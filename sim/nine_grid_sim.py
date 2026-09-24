# -*- coding: utf-8 -*-
"""数字宫格关卡仿真器（sim/nine_grid_sim.py）

定位：本模块只提供"世界 + 相机 + 带噪声动作"三件事，算法（视觉检测、GN
地图定位、FSM、容错）全部走 `levels/nine_grid.py` 的真实代码路径。断言在
`tests/test_nine_grid_sim.py`，本模块不依赖任何测试。

与 goodluck_sim 的仿真策略差异
----------------------------
- goodluck 的算法吃的是 AprilTag PnP 的位姿结果 → 模拟器重写 `solve_pnp`
  直接给"真值位姿 + 噪声"，用 matplotlib 画 2D 轨迹；
- 本关的算法吃的是**图像** → 模拟器必须重写 `capture_frame`，用真实内参/
  畸变把面板+黑字渲染成 2592×1944 合成帧，视觉链路才跑得起来。
两者共同约定：只重写 I/O 接缝（继承 RobotState），算法零改动。

仿真建模了什么 / 没建模什么
--------------------------
建模：场地几何（33.33cm 格、位置 6 恒空）、面板与黑字尺寸、真实相机内参+
径向畸变、画幅裁切、动作打滑（前进 ±8~15%、横移 ±20%、大转 ±10%、
小转 max(0, N(2.0,1.5)) 经常被"地面吞掉"）。
未建模：真机拍照耗时（2026-09-13 现场实测 fswebcam 2592x1944 -S 3 = 0.62s/张、
走代码路径 capture_frame() = 0.70s/张；**本仿真把帧当瞬时**）、舵机到位时间/抖动、
光照与白平衡对 HSV 的影响、相机-机体真实偏移（本模块 CAM_BODY_OFFSET=0）、
真实打滑分布、面板物理尺寸误差——这些必须在 P3 现场实测。

运行
----
    python -m sim.nine_grid_sim [--seed N] [--random-layout] [--quiet]
    python sim/nine_grid_sim.py            # 任意目录也可
退出码：0 = 7/7 且布局正确；1 = 否则（便于脚本/CI）。

集成测试：python tests/test_nine_grid_sim.py
"""

import argparse
import contextlib
import io
import os
import sys
from collections import Counter, namedtuple

# 允许直接 `python sim/nine_grid_sim.py` 运行（与 sim/goodluck_sim.py 一致）
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import cv2

from core.camera_config import (
    CAMERA_INTRINSIC, CAMERA_DISTORTION, HEAD_CENTER, SERVO_DEG_PER_US,
    CAM_PITCH_MOUNT_OFFSET_DEG, CAM_HEIGHT_STANDING_CM,
)
from core.ground_homography import grid_cell_center
from core.robot_core import RobotState
from levels.nine_grid import NineGridLevel, DISABLED_ACTIONS, _action_cn
from vision.nine_grid_detector import COLOR_TO_ID

# =====================================================================
# 场景常量（仿真的"真值"；与关卡假设相互独立，偏差即测试要抓的 bug）
# =====================================================================
FRAME_W, FRAME_H = 2592, 1944
PANEL_HALF_CM = 14.0        # 面板半边 14cm（33cm 格减缝）
DIGIT_HALF_CM = (4.0, 6.0)  # 黑色数字块半尺寸
_IMAGE_RECT = np.array([[0.0, 0.0], [FRAME_W, 0.0],
                        [FRAME_W, FRAME_H], [0.0, FRAME_H]],
                       dtype=np.float32)

# 各颜色满足阈值的 HSV（与 tests/test_nine_grid_detector.py 一致）
COLOR_HSV = {
    "red": (3, 200, 200), "orange": (12, 200, 200),
    "yellow": (27, 180, 200), "green": (65, 200, 100),
    "blue": (107, 230, 130), "purple": (124, 150, 150),
    "pink": (165, 120, 150),
}
PANEL_BGR = {COLOR_TO_ID[c]: cv2.cvtColor(np.full((1, 1, 3), hsv, np.uint8),
                                          cv2.COLOR_HSV2BGR)[0, 0].tolist()
             for c, hsv in COLOR_HSV.items()}

# 缺省仿真布局：位置6(左下)恒空，位置8空，其余 1..7（固定"随机"布局）
SIM_LAYOUT = {0: 5, 1: 2, 2: 7, 3: 1, 4: 4, 5: 6, 6: None, 7: 3, 8: None}


def camera_rotation(bearing_deg, pitch_pulse,
                    pitch_offset_deg=CAM_PITCH_MOUNT_OFFSET_DEG):
    """世界->相机旋转矩阵（与 tests/test_ground_homography 同一构造，已验证）

    刻意保留本模块的独立副本：仿真是"世界的替身"，与生产投影实现
    （levels/nine_grid._camera_rotation）互为交叉验证，不共享代码。
    **但相机安装下俯偏移与高度必须与 camera_config 同源**（2026-09-11）：
    历史上 sim 用名义舵机角、关卡用名义角+安装偏移，渲染与投影不一致，
    端到端仿真必然失败（实测 pitch1200：27.0° vs 45.5°）。
    有效俯角 = 名义脉宽角 + pitch_offset_deg。
    """
    a = np.radians((1500 - pitch_pulse) * SERVO_DEG_PER_US
                   + pitch_offset_deg)
    f = np.radians(bearing_deg)
    sa, ca = np.sin(a), np.cos(a)
    sf, cf = np.sin(f), np.cos(f)
    r = np.array([cf, -sf, 0.0])
    d = np.array([-sa * sf, -sa * cf, -ca])
    v = np.array([ca * sf, ca * cf, -sa])
    return np.vstack([r, d, v])


class SimNineGridRobot(RobotState):
    """九宫格仿真机器人：场地系位姿 + 打滑噪声运动 + 合成相机"""

    CAM_HEIGHT = CAM_HEIGHT_STANDING_CM   # 与 camera_config 同源（站立相机高度）
    CAM_BODY_OFFSET = 0.0  # 仿真中相机即机体中心（关卡常量另算，端到端容差内）

    def __init__(self, layout=SIM_LAYOUT, seed=3, viewer=None, deform=None):
        super().__init__(tag_poses={})
        self.pos = np.array([50.0, -20.0])   # 入口外居中
        self.heading = 0.0                   # 场地系 bearing（度，右正）
        self.pitch = 1500
        self.head = HEAD_CENTER
        self.layout = layout                 # cell -> digit
        self.rng = np.random.RandomState(seed)
        self.n_captures = 0
        # 单张拍照耗时（秒）：告诉关卡"这个环境下拍照有多贵"，用于把单格时间
        # 预算折算成拍照数预算（见 levels/nine_grid.py 的 VIS_CAPTURE_COST_*）。
        # 真机 ≈0.7s/张（2026-09-13 实测）；仿真里帧是瞬时的，取极小值 → 仿真下
        # **不会**被拍照预算先熔断（回归数字保持可比），而真机上那道闸才真正生效。
        self.capture_cost_s = 0.001
        self.action_log = []
        # 可选图形化 viewer（sim/nine_grid_view.NineGridView）；None = 纯无头
        self.viewer = viewer
        # 地板形变（2026-09-11 用户约束：本关地板会形变，机体俯仰按 ±15° 考虑）：
        # 只改**世界侧**的相机姿态/高度，关卡内常数保持不动 → 精确复现"常数失配"。
        self.deform_tilt_deg = 0.0
        self.deform_height_cm = 0.0
        self.deform_actions = 0
        # 当前目标数字（由 sim/level 侧每格开始前写入；供"按格触发阶跃"使用）
        self.current_digit = None
        self._step_done_digits = set()
        self._deform = dict(deform) if deform else None
        self._deform_rng = np.random.RandomState(int(seed) + 977)

    # ---- I/O 接缝 ----

    def set_head(self, pulse, move_time_ms=500):
        self.head = pulse
        self.current_head_pulse = pulse  # 与真机 RobotState.set_head 行为一致

    def set_deform_digit(self, digit):
        """关卡开新格时通知 sim 当前目标数字（仅供"按格触发阶跃"使用）

        真机 RobotState 没有这个方法 → 关卡侧用 getattr 兜底调用，不影响真机。
        """
        self.current_digit = int(digit)

    def set_pitch(self, pulse, move_time_ms=500):
        self.pitch = pulse

    def run_action(self, name, times=1):
        for _ in range(max(1, times)):
            self._apply_action(name)
            self._update_deform()
        self.action_log.append((name, times))
        if self.viewer is not None:
            self.viewer.on_action(self, name, times)

    def _update_deform(self):
        """地板形变模型：每动作一次随机游走 + 可选一次性阶跃

        deform 配置（run_simulation 透传）：
          sigma_tilt_deg / sigma_h_cm  每动作随机游走步长（默认 0 = 无形变）
          max_tilt_deg  / max_h_cm     形变幅度上限（默认 15° / 3cm）
          step_after_actions           第 N 个动作后施加一次性阶跃
          step_at_digit                第 N 格（目标数字）开始时施加一次性阶跃
          step_tilt_deg / step_h_cm    阶跃幅度
        注意：形变只在**动作**时变化 —— 静止扫头部（布局扫）期间保持不变，
        与现场"站定扫描时地板不再继续形变"一致。
        **为什么要有 step_at_digit（2026-09-13 新增）**：`step_after_actions` 与
        动作流强耦合——任何改变动作数的代码改动都会把阶跃触发点挪到另一个
        格子，于是同一个"回归测试"在两次改动之间测的根本不是同一个场景
        （实测：HEAD 244/191 过、本轮 206 张不过，部分原因就在这里）。
        按格触发让"第几格踩上形变"成为**场景定义的一部分**，与被测代码的
        动作数解耦；旧参数保留（用于复现历史日志），但新回归一律用 step_at_digit。
        """
        cfg = self._deform
        if not cfg:
            return
        self.deform_actions += 1
        if cfg.get("sigma_tilt_deg"):
            self.deform_tilt_deg += float(
                self._deform_rng.normal(0.0, float(cfg["sigma_tilt_deg"])))
        if cfg.get("sigma_h_cm"):
            self.deform_height_cm += float(
                self._deform_rng.normal(0.0, float(cfg["sigma_h_cm"])))
        if cfg.get("step_after_actions") \
                and self.deform_actions == int(cfg["step_after_actions"]):
            self.deform_tilt_deg += float(cfg.get("step_tilt_deg", 0.0))
            self.deform_height_cm += float(cfg.get("step_h_cm", 0.0))
        # 按格触发（与被测代码的动作数解耦，见上文说明）：关卡每开一格会调用
        # set_deform_digit(K)，第 K 格开始后的**第一个动作**上施加阶跃。
        k = cfg.get("step_at_digit")
        if k is not None and self.current_digit == int(k) \
                and int(k) not in self._step_done_digits:
            self._step_done_digits.add(int(k))
            self.deform_tilt_deg += float(cfg.get("step_tilt_deg", 0.0))
            self.deform_height_cm += float(cfg.get("step_h_cm", 0.0))
        mt = float(cfg.get("max_tilt_deg", 15.0))
        mh = float(cfg.get("max_h_cm", 3.0))
        self.deform_tilt_deg = float(np.clip(self.deform_tilt_deg, -mt, mt))
        self.deform_height_cm = float(np.clip(self.deform_height_cm, -mh, mh))

    def _apply_action(self, name):
        th = np.radians(self.heading)
        fwd = np.array([np.sin(th), np.cos(th)])
        right = np.array([np.cos(th), -np.sin(th)])
        slip = self.rng.normal
        # bearing 约定右正（见 core/ground_homography）：左转 = heading 减小
        if name == "go_forward":
            self.pos += fwd * 5.0 * (1 + slip(0, 0.08))
        elif name == "go_forward_one_step":
            self.pos += fwd * 2.0 * (1 + slip(0, 0.15))
        elif name == "back_one_step":
            self.pos -= fwd * 3.2 * (1 + slip(0, 0.15))
        elif name == "left_move":
            self.pos -= right * 1.9 * (1 + slip(0, 0.20))
        elif name == "right_move":
            self.pos += right * 2.2 * (1 + slip(0, 0.20))
        elif name == "turn_left":
            self.heading -= 22.0 * (1 + slip(0, 0.10))
        elif name == "turn_right":
            self.heading += 25.7 * (1 + slip(0, 0.10))
        elif name == "turn_left_small_step":
            self.heading -= max(0.0, self.rng.normal(2.0, 1.5))
        elif name == "turn_right_small_step":
            self.heading += max(0.0, self.rng.normal(2.0, 1.5))
        elif name == "stand":
            pass
        else:
            raise ValueError(f"仿真未实现动作: {name}")

    def capture_frame(self):
        self.n_captures += 1
        # 头部左转(脉宽>1500)为正 → 相机方位角减小（与真机一致）
        bearing = self.heading - (self.head - HEAD_CENTER) * SERVO_DEG_PER_US
        # 形变只加到"世界侧"：相机安装偏移 + 地板形变倾角、站立高度 + 形变高度差
        R = camera_rotation(bearing, self.pitch,
                            CAM_PITCH_MOUNT_OFFSET_DEG + self.deform_tilt_deg)
        C = np.array([self.pos[0], self.pos[1],
                      self.CAM_HEIGHT + self.deform_height_cm])

        frame = np.full((FRAME_H, FRAME_W, 3), 90, np.uint8)
        for cell, digit in self.layout.items():
            if digit is None:
                continue
            self._draw_panel(frame, R, C, cell, digit)
        if self.viewer is not None:
            self.viewer.on_frame(self, frame)
        return frame

    def _draw_panel(self, frame, R, C, cell, digit):
        cx, cy = grid_cell_center(cell)

        def proj(dx, dy):
            p = np.array([cx + dx, cy + dy, 0.0])
            pc = R @ (p - C)
            return pc

        def quad(hx, hy):
            corners = [proj(-hx, -hy), proj(hx, -hy), proj(hx, hy), proj(-hx, hy)]
            if any(c[2] <= 0.05 for c in corners):
                return None
            norm = np.array([c[:2] / c[2] for c in corners])
            pix, _ = cv2.projectPoints(
                np.column_stack([norm, np.ones(4)]).reshape(-1, 1, 3)
                .astype(np.float32),
                np.zeros(3), np.zeros(3), CAMERA_INTRINSIC, CAMERA_DISTORTION)
            pix = pix[:, 0, :]
            # 角点贴近相机平面（z 略大于 0.05）时投影会溢出为 ±inf，
            # astype(int32) 得 INT_MIN 后 cv2.fillPoly 会扫描 ~2^31 行
            # （实测 26s/次，整轮仿真被拖到 800s+）。
            # 正确处理 = 裁到画幅：真实相机只看到交集，直接丢弃会让"贴边
            # 仍可见的细条"在仿真里消失（比现实更难，掩盖定位缺陷）。
            if not np.all(np.isfinite(pix)):
                return None
            pix = np.clip(pix, -1e6, 1e6)
            area, inter = cv2.intersectConvexConvex(
                pix.astype(np.float32), _IMAGE_RECT)
            if inter is None or len(inter) < 3 or area <= 1e-9:
                return None
            return np.round(inter).astype(np.int32)

        panel = quad(PANEL_HALF_CM, PANEL_HALF_CM)
        if panel is not None:
            cv2.fillPoly(frame, [panel], PANEL_BGR[digit])
        digit_quad = quad(DIGIT_HALF_CM[0], DIGIT_HALF_CM[1])
        if digit_quad is not None:
            cv2.fillPoly(frame, [digit_quad], (0, 0, 0))


# =====================================================================
# 运行入口（CLI 与测试共用）
# =====================================================================

SimRun = namedtuple("SimRun", "robot level stats")


def random_layout(seed):
    """由 seed 生成合法随机布局：位置 6 恒空，7 个数字占其余 8 格中的 7 格"""
    rng = np.random.RandomState(seed)
    cells = [c for c in range(9) if c != 6]
    chosen = list(rng.permutation(cells)[:7])
    digits = list(rng.permutation(range(1, 8)))
    layout = {c: None for c in range(9)}
    for cell, digit in zip(chosen, digits):
        layout[int(cell)] = int(digit)
    return layout


def run_simulation(layout=SIM_LAYOUT, seed=3, quiet=False, viewer=None,
                   deform=None):
    """跑一遍完整关卡（布局扫→1..7），不做断言；返回 SimRun(robot, level, stats)

    stats 键：ok_all / results / layout_ok / digit_cell / truth / captures /
              actions / action_counts / small_turn_deg / small_turn_usable /
              banned_used / deform_tilt_deg / deform_height_cm
    quiet=True 时吞掉关卡逐行日志（只留返回值供调用方打印摘要）。
    viewer：可选图形化 viewer（需有 attach(robot, level) 与 on_action/on_frame）；
            传入后由 viewer 决定节奏（暂停/单步），None = 纯无头。
    deform：地板形变注入（见 SimNineGridRobot._update_deform）；None = 无形变。
    """
    robot = SimNineGridRobot(layout=layout, seed=seed, viewer=viewer,
                             deform=deform)
    level = NineGridLevel(robot)
    if viewer is not None:
        viewer.attach(robot, level)
    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            ok_all = level.run_level()
    else:
        ok_all = level.run_level()

    truth = {d: c for c, d in layout.items() if d is not None}
    used = {a for a, _ in robot.action_log}
    stats = {
        "ok_all": bool(ok_all),
        "results": list(level.results),
        "layout_ok": level.digit_cell == truth,
        "digit_cell": dict(level.digit_cell),
        "truth": truth,
        "captures": robot.n_captures,
        "actions": len(robot.action_log),
        "action_counts": dict(Counter(a for a, _ in robot.action_log)),
        "small_turn_deg": level._small_turn_deg,
        "small_turn_usable": level._small_turn_usable,
        "banned_used": sorted(used & set(DISABLED_ACTIONS)),
        "cell_trips": dict(level.cell_trips),
        # 逐格落点：{digit: [离格心 cm, 来源]}；来源应为 "真值"（仿真有 state.pos）
        "panel_landing": {d: (None if v is None else [round(v[0], 2), v[1]])
                          for d, v in level.panel_landing.items()},
        "deform_tilt_deg": robot.deform_tilt_deg,
        "deform_height_cm": robot.deform_height_cm,
    }
    return SimRun(robot, level, stats)


def _action_counts_cn(counts):
    """{动作名: 次数} → "左小转 12 次、前进一步 40 次"（按次数降序，日志用）"""
    if not counts:
        return "（无）"
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return "、".join(_action_cn(a, n) for a, n in items)


def _print_summary(stats):
    ok_n = sum(1 for _, ok in stats["results"] if ok)
    print(f"[仿真] 布局识别: {'成功' if stats['layout_ok'] else '失败'}"
          f"｜数字→格位: {stats['digit_cell']}")
    print(f"[仿真] 到位: {ok_n}/{len(stats['results'])}"
          f"｜拍照 {stats['captures']} 张｜动作 {stats['actions']} 次")
    print(f"[仿真] 动作统计: {_action_counts_cn(stats['action_counts'])}")
    print(f"[仿真] 单步小转角估计: {stats['small_turn_deg']:.1f}°/次"
          f"（{'可用' if stats['small_turn_usable'] else '不可用，改用大角度转向'}）")
    if stats["banned_used"]:
        print(f"[仿真] 警告: 出现禁用动作 {stats['banned_used']}")
    if stats.get("deform_tilt_deg") or stats.get("deform_height_cm"):
        print(f"[仿真] 结束时的地板形变: 俯仰 {stats['deform_tilt_deg']:+.1f}°"
              f"｜高度 {stats['deform_height_cm']:+.1f}cm")


def main(argv=None):
    ap = argparse.ArgumentParser(description="数字宫格关卡仿真")
    ap.add_argument("--seed", type=int, default=3,
                    help="动作噪声/随机布局种子（默认 3）")
    ap.add_argument("--random-layout", action="store_true",
                    help="由 seed 生成合法随机布局（默认用固定 SIM_LAYOUT）")
    ap.add_argument("--quiet", action="store_true",
                    help="不打印关卡逐行日志，只打印摘要")
    ap.add_argument("--deform", type=float, default=0.0, metavar="SIGMA_TILT_DEG",
                    help="地板形变：每动作俯仰随机游走步长（度），"
                         "幅度上限 ±15°、高度 ±2cm（0=无形变）")
    args = ap.parse_args(argv)

    layout = random_layout(args.seed) if args.random_layout else SIM_LAYOUT
    print(f"[仿真] 布局: {layout}")
    deform = None
    if args.deform:
        deform = {"sigma_tilt_deg": args.deform, "sigma_h_cm": 0.5,
                  "max_tilt_deg": 15.0, "max_h_cm": 2.0}
        print(f"[仿真] 注入地板形变: 每次动作俯仰随机游走 "
              f"{deform['sigma_tilt_deg']:.1f}°、高度随机游走 "
              f"{deform['sigma_h_cm']:.1f}cm（俯仰上限 ±"
              f"{deform['max_tilt_deg']:.0f}°、高度上限 ±"
              f"{deform['max_h_cm']:.0f}cm）")
    try:
        run = run_simulation(layout=layout, seed=args.seed, quiet=args.quiet,
                             deform=deform)
    except Exception as e:  # 布局扫失败等：CLI 友好退出，便于脚本判断
        print(f"[仿真] 关卡异常: {type(e).__name__}: {e}")
        return 1
    _print_summary(run.stats)
    return 0 if (run.stats["ok_all"] and run.stats["layout_ok"]) else 1


if __name__ == "__main__":
    sys.exit(main())
