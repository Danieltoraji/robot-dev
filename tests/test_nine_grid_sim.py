# -*- coding: utf-8 -*-
"""数字宫格关卡 PC 仿真集成测试（tests/test_nine_grid_sim.py）

合成场地渲染（七色面板+黑色数字块，含畸变投影）+ 带噪声运动模型
（模拟打滑：位移 ±10~20%、小转角噪声大且偶尔趋零），继承 RobotState
只重写 I/O 接缝（run_action/capture_frame/set_head/set_pitch），
端到端跑 NineGridLevel.run_level()：布局扫描 → 1..7 逐格导航 → 到达确认。
视觉检测、GN 地图定位、自适应转向、卡滞守卫全部走真实代码路径。

运行（PC 上跑；不进快速单测）：
    python tests/test_nine_grid_sim.py
    python -m tests.test_nine_grid_sim
"""

import os
import sys

# 允许从任意目录直接运行本文件（脚本目录在 sys.path[0]，仓库根不在）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import cv2

from core.camera_config import CAMERA_INTRINSIC, CAMERA_DISTORTION, \
    HEAD_CENTER, SERVO_DEG_PER_US
from core.ground_homography import grid_cell_center
from core.robot_core import RobotState
from levels.nine_grid import NineGridLevel, PITCH_NAV, PITCH_DOWN
from vision.nine_grid_detector import COLOR_TO_ID

FRAME_W, FRAME_H = 2592, 1944
PANEL_HALF_CM = 14.0    # 面板半边 14cm（33cm 格减缝）
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

# 仿真布局：位置6(左下)恒空，位置8空，其余 1..7（固定"随机"布局）
SIM_LAYOUT = {0: 5, 1: 2, 2: 7, 3: 1, 4: 4, 5: 6, 6: None, 7: 3, 8: None}


def camera_rotation(bearing_deg, pitch_pulse):
    """世界->相机旋转（与 tests/test_ground_homography 同一构造，已验证）"""
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


def test_banned_actions_rejected():
    """动作白名单硬门：本场地禁用的大步幅动作必须被拒绝（防误用摔倒）"""
    robot = SimNineGridRobot()
    level = NineGridLevel(robot)
    for action in ("go_forward", "go_forward_fast", "climb_stairs"):
        try:
            level._act(action, 1)
        except ValueError:
            continue
        raise AssertionError(f"{action} 未被白名单拒绝")
    level._act("go_forward_one_step", 1)  # 白名单内动作应可执行
    print("  动作白名单：go_forward/go_forward_fast 被拒，one_step 可执行 ✓")


def test_render_guard_fast():
    """渲染守卫回归：角点贴近相机平面时不得产生退化多边形（曾 26s/帧）

    背景：z_cam 略大于 0.05 的角点投影为 ±inf，astype(int32) 得 INT_MIN，
    cv2.fillPoly 会扫描 ~2^31 行。此处遍历"格内/格间 + 两档俯仰 + 广角头部"
    组合，断言单帧渲染 <0.1s，防止该缺陷回归。
    """
    import time
    from core.camera_config import HEAD_WIDE_LEFT, HEAD_WIDE_RIGHT
    robot = SimNineGridRobot()
    slow = []
    for (x, y) in ((50, -20), (50, 10), (50, 30), (50, 45), (20, 45), (80, 45)):
        for heading in (0.0, 90.0, -90.0):
            for pitch in (PITCH_NAV, PITCH_DOWN):
                for head in (HEAD_CENTER, HEAD_WIDE_LEFT, HEAD_WIDE_RIGHT):
                    robot.pos = np.array([float(x), float(y)])
                    robot.heading = heading
                    robot.pitch = pitch
                    robot.head = head
                    t0 = time.perf_counter()
                    robot.capture_frame()
                    dt = time.perf_counter() - t0
                    if dt > 0.1:
                        slow.append((x, y, heading, pitch, head, dt))
    assert not slow, f"渲染超时（退化多边形回归）: {slow[:5]}"
    print("  渲染守卫：极端位姿/俯仰/头部组合全部 <0.1s ✓")


def test_sim_full_run():
    robot = SimNineGridRobot()
    level = NineGridLevel(robot)
    ok_all = level.run_level()

    # 布局应与真值一致
    truth = {d: c for c, d in SIM_LAYOUT.items() if d is not None}
    assert level.digit_cell == truth, \
        f"布局解算错误: {level.digit_cell} != 真值 {truth}"

    # 全部 7 格应确认到达
    failed = [k for k, ok in level.results if not ok]
    assert ok_all and not failed, \
        f"未确认到达的面板: {failed}，结果 {level.results}"

    # 本场地禁用 go_forward / go_forward_fast（小面板+打滑地板易摔倒）：
    # 白名单硬门之外再加一条行为断言，防止后续改动绕过 _act 直接调 state.act
    used = {a for a, _ in robot.action_log}
    banned = used & {"go_forward", "go_forward_fast"}
    assert not banned, f"使用了本场地禁用的大步幅动作: {banned}"

    # 定位次数护栏（真机时间预算的代理指标；超限说明 FSM 在空转）
    # 基线 215 张（2026-09-08：裁切感知观测 + min_panels=1 + 20cm 切低头 + 禁用 go_forward）
    assert robot.n_captures < 350, \
        f"拍照次数 {robot.n_captures} 超护栏，检查是否出现定位风暴"

    print(f"\n[仿真] 全部 7 格到达确认 ✓  拍照 {robot.n_captures} 张，"
          f"动作 {len(robot.action_log)} 次")
    from collections import Counter
    print("[仿真] 动作统计:", dict(Counter(a for a, _ in robot.action_log)))
    print("[仿真] 小转最终估计: "
          f"{level._small_turn_deg:.1f}°/次 "
          f"({'可用' if level._small_turn_usable else '已弃用(大转兜底)'})")


if __name__ == "__main__":
    test_banned_actions_rejected()
    test_render_guard_fast()
    test_sim_full_run()
    print("数字宫格仿真集成测试通过 ✓")
