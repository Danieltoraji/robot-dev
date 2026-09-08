# -*- coding: utf-8 -*-
"""数字宫格关卡 PC 仿真集成测试（tests/test_nine_grid_sim.py）

合成场地渲染（七色面板+黑色数字块，含畸变投影）+ 带噪声运动模型
（模拟打滑：位移 ±10~20%、小转角噪声大且偶尔趋零），继承 RobotState
只重写 I/O 接缝（run_action/capture_frame/set_head/set_pitch），
端到端跑 NineGridLevel.run_level()：布局扫描 → 1..7 逐格导航 → 到达确认。
视觉检测、逐帧单应定位、自适应转向、卡滞守卫全部走真实代码路径。

运行（约 1~2 分钟，PC 上跑；不进快速单测）：
    python tests/test_nine_grid_sim.py
"""

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
            return np.round(pix[:, 0, :]).astype(np.int32)

        panel = quad(PANEL_HALF_CM, PANEL_HALF_CM)
        if panel is not None:
            cv2.fillPoly(frame, [panel], PANEL_BGR[digit])
        digit_quad = quad(DIGIT_HALF_CM[0], DIGIT_HALF_CM[1])
        if digit_quad is not None:
            cv2.fillPoly(frame, [digit_quad], (0, 0, 0))


def test_sim_full_run():
    robot = SimNineGridRobot()
    level = NineGridLevel(robot)
    done = level.run_level()

    # 布局应与真值一致
    truth = {d: c for c, d in SIM_LAYOUT.items() if d is not None}
    assert level.digit_cell == truth, \
        f"布局解算错误: {level.digit_cell} != 真值 {truth}"

    # 全部 7 格应确认到达
    failed = [k for k, ok in done if not ok]
    assert not failed, f"未确认到达的面板: {failed}，结果 {done}"

    print(f"\n[仿真] 全部 7 格到达确认 ✓  拍照 {robot.n_captures} 张，"
          f"动作 {len(robot.action_log)} 次")
    from collections import Counter
    print("[仿真] 动作统计:", dict(Counter(a for a, _ in robot.action_log)))
    print("[仿真] 小转最终估计: "
          f"{level._small_turn_deg:.1f}°/次 "
          f"({'可用' if level._small_turn_usable else '已弃用(大转兜底)'})")


if __name__ == "__main__":
    test_sim_full_run()
    print("数字宫格仿真集成测试通过 ✓")
