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
未建模：真机拍照耗时（fswebcam 2~4s/张）、舵机到位时间/抖动、光照与白平衡
对 HSV 的影响、相机-机体真实偏移（本模块 CAM_BODY_OFFSET=0）、真实打滑分布、
面板物理尺寸误差——这些必须在 P3 现场实测。

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
)
from core.ground_homography import grid_cell_center
from core.robot_core import RobotState
from levels.nine_grid import NineGridLevel, DISABLED_ACTIONS
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


def camera_rotation(bearing_deg, pitch_pulse):
    """世界->相机旋转矩阵（与 tests/test_ground_homography 同一构造，已验证）

    刻意保留本模块的独立副本：仿真是"世界的替身"，与生产投影实现
    （levels/nine_grid._camera_rotation）互为交叉验证，不共享代码。
    """
    a = np.radians((1500 - pitch_pulse) * SERVO_DEG_PER_US)
    f = np.radians(bearing_deg)
    sa, ca = np.sin(a), np.cos(a)
    sf, cf = np.sin(f), np.cos(f)
    r = np.array([cf, -sf, 0.0])
    d = np.array([-sa * sf, -sa * cf, -ca])
    v = np.array([ca * sf, ca * cf, -sa])
    return np.vstack([r, d, v])


class SimNineGridRobot(RobotState):
    """九宫格仿真机器人：场地系位姿 + 打滑噪声运动 + 合成相机"""

    CAM_HEIGHT = 39.0
    CAM_BODY_OFFSET = 0.0  # 仿真中相机即机体中心（关卡常量另算，端到端容差内）

    def __init__(self, layout=SIM_LAYOUT, seed=3):
        super().__init__(tag_poses={})
        self.pos = np.array([50.0, -20.0])   # 入口外居中
        self.heading = 0.0                   # 场地系 bearing（度，右正）
        self.pitch = 1500
        self.head = HEAD_CENTER
        self.layout = layout                 # cell -> digit
        self.rng = np.random.RandomState(seed)
        self.n_captures = 0
        self.action_log = []

    # ---- I/O 接缝 ----

    def set_head(self, pulse, move_time_ms=500):
        self.head = pulse
        self.current_head_pulse = pulse  # 与真机 RobotState.set_head 行为一致

    def set_pitch(self, pulse, move_time_ms=500):
        self.pitch = pulse

    def run_action(self, name, times=1):
        for _ in range(max(1, times)):
            self._apply_action(name)
        self.action_log.append((name, times))

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
        R = camera_rotation(bearing, self.pitch)
        C = np.array([self.pos[0], self.pos[1], self.CAM_HEIGHT])

        frame = np.full((FRAME_H, FRAME_W, 3), 90, np.uint8)
        for cell, digit in self.layout.items():
            if digit is None:
                continue
            self._draw_panel(frame, R, C, cell, digit)
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


def run_simulation(layout=SIM_LAYOUT, seed=3, quiet=False):
    """跑一遍完整关卡（布局扫→1..7），不做断言；返回 SimRun(robot, level, stats)

    stats 键：ok_all / results / layout_ok / digit_cell / truth / captures /
              actions / action_counts / small_turn_deg / small_turn_usable /
              banned_used
    quiet=True 时吞掉关卡逐行日志（只留返回值供调用方打印摘要）。
    """
    robot = SimNineGridRobot(layout=layout, seed=seed)
    level = NineGridLevel(robot)
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
    }
    return SimRun(robot, level, stats)


def _print_summary(stats):
    ok_n = sum(1 for _, ok in stats["results"] if ok)
    print(f"[sim] 布局解算: {'OK' if stats['layout_ok'] else 'FAIL'}  "
          f"数字→格: {stats['digit_cell']}")
    print(f"[sim] 到达: {ok_n}/{len(stats['results'])}  "
          f"拍照 {stats['captures']} 张 / 动作 {stats['actions']} 次")
    print(f"[sim] 动作统计: {stats['action_counts']}")
    print(f"[sim] 小转估计: {stats['small_turn_deg']:.1f}°/次 "
          f"({'可用' if stats['small_turn_usable'] else '已弃用(大转兜底)'})")
    if stats["banned_used"]:
        print(f"[sim] ⚠ 使用了禁用动作: {stats['banned_used']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="数字宫格关卡仿真")
    ap.add_argument("--seed", type=int, default=3,
                    help="动作噪声/随机布局种子（默认 3）")
    ap.add_argument("--random-layout", action="store_true",
                    help="由 seed 生成合法随机布局（默认用固定 SIM_LAYOUT）")
    ap.add_argument("--quiet", action="store_true",
                    help="不打印关卡逐行日志，只打印摘要")
    args = ap.parse_args(argv)

    layout = random_layout(args.seed) if args.random_layout else SIM_LAYOUT
    print(f"[sim] 布局: {layout}")
    try:
        run = run_simulation(layout=layout, seed=args.seed, quiet=args.quiet)
    except Exception as e:  # 布局扫失败等：CLI 友好退出，便于脚本判断
        print(f"[sim] 关卡异常: {type(e).__name__}: {e}")
        return 1
    _print_summary(run.stats)
    return 0 if (run.stats["ok_all"] and run.stats["layout_ok"]) else 1


if __name__ == "__main__":
    sys.exit(main())
