# -*- coding: utf-8 -*-
"""上下楼梯与识别跨障关卡仿真器（sim/stairs_hurdle_sim.py）

定位与 nine_grid_sim 相同：本模块只提供"世界 + 相机 + 带噪声动作"，
算法（红色带检测、地面度量、FSM）全部走 vision/ core/ levels/ 的真实
代码路径，只重写 I/O 接缝（继承 RobotState）。

场景模型（场景系：原点=第一级台阶基线中点，y 前 x 右 z 上，单位 cm）
------------------------------------------------------------------
  y<0        平地（入口侧，机器人从上一关抵达）
  y=0        第一级台阶立面（红色胶条下半段贴此，z 0~1.5）
  y∈[0,15]   踏面1 z=2.5（胶条上半段贴此，y 0~4）
  y=15       第二级立面
  y∈[15,30]  顶部平台 z=5
  y=30/45    下行两级立面
  y∈[30,45]  踏面 z=2.5/0
  y>45       平地，红色横杆在 y=45+bar_dist（25~40 随机）
未建模：楼梯木结构的遮挡与外观（合成帧只渲染红色目标+纯色背景；
遮挡已在方案里几何论证过：接近段横杆越过台阶沿可见）、拍照耗时、
光照/白平衡、相机-机体偏移（CAM_BODY_OFFSET=0）。

动作名义位移（"真值"，与关卡参数表无关；打滑噪声仿 nine_grid_sim）
------------------------------------------------------------------
climb_stairs: 前进 18cm 且 z+2.5；down_floor: 前进 13cm 且 z-2.5；
hurdles: 前进 10cm。跳变建模为动作组的净位移。

运行
----
    python -m sim.stairs_hurdle_sim [--seed N] [--quiet]
集成测试：python tests/test_stairs_hurdle_sim.py
"""

import argparse
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2

from core.camera_config import (
    CAMERA_INTRINSIC, CAMERA_DISTORTION, HEAD_CENTER, SERVO_DEG_PER_US,
)
from core.robot_core import RobotState

FRAME_W, FRAME_H = 2592, 1944
_IMAGE_RECT = np.array([[0.0, 0.0], [FRAME_W, 0.0],
                        [FRAME_W, FRAME_H], [0.0, FRAME_H]],
                       dtype=np.float32)

# 红色 BGR（HSV(3,200,200)），与 nine_grid_sim 同法生成
RED_BGR = cv2.cvtColor(np.full((1, 1, 3), (3, 200, 200), np.uint8),
                       cv2.COLOR_HSV2BGR)[0, 0].tolist()


def camera_rotation(bearing_deg, pitch_deg):
    """世界->相机旋转（与 tests/test_ground_homography 同构造，刻意独立副本）"""
    a = np.radians(pitch_deg)
    f = np.radians(bearing_deg)
    sa, ca = np.sin(a), np.cos(a)
    sf, cf = np.sin(f), np.cos(f)
    r = np.array([cf, -sf, 0.0])
    d = np.array([-sa * sf, -sa * cf, -ca])
    v = np.array([ca * sf, ca * cf, -sa])
    return np.vstack([r, d, v])


def tessellate_x(corners, n=12):
    """把 (bl,br,tr,tl) 四边形沿 x 细分成 n 段小四边形。

    畸变下地面直线投影像素是曲线，单段长边的直线弦与曲线差可达 ~1.5cm
    （真值被渲染污染）。细分后每段弦差可忽略。
    """
    bl, br, tr, tl = corners
    edges = np.linspace(0.0, 1.0, n + 1)
    quads = []
    for a, b in zip(edges[:-1], edges[1:]):
        quads.append(np.array([bl + (br - bl) * a, bl + (br - bl) * b,
                               tl + (tr - tl) * b, tl + (tr - tl) * a]))
    return quads


def project_quad(frame, R, C, corners, color):
    """3D 四边形(4,3)渲染到帧（含真实内参/畸变/画幅裁切）。成功返回 True"""
    pc = (R @ (np.asarray(corners, dtype=np.float64) - C).T).T
    if np.any(pc[:, 2] <= 0.05):
        return False
    norm = pc[:, :2] / pc[:, 2:3]
    pix, _ = cv2.projectPoints(
        np.column_stack([norm, np.ones(4)]).reshape(-1, 1, 3).astype(np.float32),
        np.zeros(3), np.zeros(3), CAMERA_INTRINSIC, CAMERA_DISTORTION)
    pix = pix[:, 0, :]
    if not np.all(np.isfinite(pix)):
        return False
    pix = np.clip(pix, -1e6, 1e6)
    area, inter = cv2.intersectConvexConvex(pix.astype(np.float32), _IMAGE_RECT)
    if inter is None or len(inter) < 3 or area <= 1e-9:
        return False
    cv2.fillPoly(frame, [np.round(inter).astype(np.int32)], color)
    return True


class StairsScene:
    """场地几何真值（cm）。红带方向偏转用于方位真值扰动测试。"""

    def __init__(self, tape_width=40.0, tape_yaw_deg=0.0, tape_visible=True,
                 bar_dist=30.0, bar_yaw_deg=0.0, bar_visible=True,
                 bar_half_len=40.0):
        self.step_rise = 2.5
        self.step_run = 15.0
        self.tape_width = tape_width
        self.tape_yaw = np.radians(tape_yaw_deg)
        self.tape_visible = tape_visible
        self.bar_dist = bar_dist            # 距楼梯下沿出口（y=45）
        self.bar_yaw = np.radians(bar_yaw_deg)
        self.bar_visible = bar_visible
        self.bar_half_len = bar_half_len

        self.bar_y = 45.0 + bar_dist        # 横杆地面线位置（场景系 y）

    def _rot_x(self, pts, yaw):
        """绕 z 轴旋转（方向偏转用）"""
        c, s = np.cos(yaw), np.sin(yaw)
        return pts @ np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]).T

    def red_quads(self):
        """所有红色目标四边形（场景系，长边已细分逼近畸变曲线）"""
        quads = []
        w2 = self.tape_width / 2.0
        if self.tape_visible:
            rise = self.step_rise * 0.6   # 立面段 1.5cm
            tread = 4.0                    # 踏面段深 4cm
            riser = np.array([[-w2, 0.0, 0.0], [w2, 0.0, 0.0],
                              [w2, 0.0, rise], [-w2, 0.0, rise]])
            tread_q = np.array([[-w2, 0.0, self.step_rise], [w2, 0.0, self.step_rise],
                                [w2, tread, self.step_rise], [-w2, tread, self.step_rise]])
            for base in (riser, tread_q):
                quads += [self._rot_x(q, self.tape_yaw)
                          for q in tessellate_x(base, n=12)]
        if self.bar_visible:
            L = self.bar_half_len
            t = 2.0
            bar = np.array([[-L, self.bar_y + t / 2, 0.0], [L, self.bar_y + t / 2, 0.0],
                            [L, self.bar_y + t / 2, t], [-L, self.bar_y + t / 2, t]])
            quads += [self._rot_x(q, self.bar_yaw) for q in tessellate_x(bar, n=24)]
        return quads


class SimStairsRobot(RobotState):
    """场景系位姿 + 打滑噪声动作 + 合成相机（红带渲染）"""

    CAM_HEIGHT = 39.0
    CAM_BODY_OFFSET = 0.0

    def __init__(self, scene: StairsScene, seed=0, start=(0.0, -60.0),
                 heading_deg=0.0, wobble_sigma_deg=0.3):
        super().__init__(tag_poses={})
        self.scene = scene
        self.pos = np.array(start, dtype=np.float64)
        self.z = 0.0                        # 脚下平面高度（台阶上为 2.5/5）
        self.heading = heading_deg          # 右正（度）
        self.pitch = 1500
        self.head = HEAD_CENTER
        self.rng = np.random.RandomState(seed)
        self.wobble_sigma_deg = wobble_sigma_deg
        self.n_captures = 0
        self.action_log = []

    # ---- I/O 接缝 ----

    def set_head(self, pulse, move_time_ms=500):
        self.head = pulse
        self.current_head_pulse = pulse

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
        if name == "go_forward_one_step":
            self.pos += fwd * 2.0 * (1 + slip(0, 0.15))
        elif name == "go_forward_one_small_step":
            self.pos += fwd * 1.5 * (1 + slip(0, 0.15))
        elif name == "back_one_step":
            self.pos -= fwd * 3.2 * (1 + slip(0, 0.15))
        elif name == "left_move":
            self.pos -= right * 1.9 * (1 + slip(0, 0.20))
        elif name == "right_move":
            self.pos += right * 2.2 * (1 + slip(0, 0.20))
        elif name == "turn_left_small_step":
            self.heading -= max(0.0, self.rng.normal(2.0, 1.5))
        elif name == "turn_right_small_step":
            self.heading += max(0.0, self.rng.normal(2.0, 1.5))
        elif name == "turn_left":
            self.heading -= 22.0 * (1 + slip(0, 0.10))
        elif name == "turn_right":
            self.heading += 25.7 * (1 + slip(0, 0.10))
        elif name == "stand":
            pass
        elif name == "climb_stairs":
            self.pos += fwd * 18.0 * (1 + slip(0, 0.10))
            self.z += self.scene.step_rise
        elif name == "down_floor":
            self.pos += fwd * 13.0 * (1 + slip(0, 0.10))
            self.z -= self.scene.step_rise
        elif name == "hurdles":
            self.pos += fwd * 10.0 * (1 + slip(0, 0.10))
        else:
            raise ValueError(f"仿真未实现动作: {name}")

    def capture_frame(self):
        self.n_captures += 1
        bearing = self.heading - (self.head - HEAD_CENTER) * SERVO_DEG_PER_US
        wobble = 0.0
        if self.wobble_sigma_deg > 0:
            wobble = float(self.rng.normal(0.0, self.wobble_sigma_deg))
        pitch_deg = (1500 - self.pitch) * SERVO_DEG_PER_US + wobble
        R = camera_rotation(bearing, pitch_deg)
        C = np.array([self.pos[0], self.pos[1], self.CAM_HEIGHT + self.z])
        frame = np.full((FRAME_H, FRAME_W, 3), 90, np.uint8)
        for corners in self.scene.red_quads():
            project_quad(frame, R, C, corners, RED_BGR)
        return frame


def make_frame(red_quads, bearing_deg=0.0, pitch_deg=45.0, cam_xy=(0.0, 0.0),
               cam_z=39.0, color=RED_BGR):
    """独立帧合成工具（精度扫描/HSV 鲁棒性测试用，不经过 RobotState）"""
    R = camera_rotation(bearing_deg, pitch_deg)
    C = np.array([cam_xy[0], cam_xy[1], cam_z])
    frame = np.full((FRAME_H, FRAME_W, 3), 90, np.uint8)
    for corners in red_quads:
        project_quad(frame, R, C, corners, color)
    return frame


def bar_line_quads(forward_cm, bearing_deg=0.0, lateral_cm=0.0,
                   half_len=40.0, height_off_cm=0.0):
    """在机器人本地系生成一条"横杆下沿"合成目标（回归测试用）

    forward: 下沿中点前向距离；bearing: 直线方向偏角；lateral: 中点横偏；
    height_off: 下沿相对脚下平面的高度（爬楼段方位场景用）。
    返回细分四边形列表（含 2cm 高度让红色可见，下沿严格在 height_off 高度线
    上；细分是为逼近畸变曲线，避免渲染弦差污染真值）。
    """
    th = np.radians(bearing_deg)
    dx, dy = np.cos(th), np.sin(th)          # 直线方向（沿带长）
    cx, cy = lateral_cm, forward_cm          # 下沿中点
    p0 = np.array([cx - dx * half_len, cy - dy * half_len, height_off_cm])
    p1 = np.array([cx + dx * half_len, cy + dy * half_len, height_off_cm])
    up = np.array([0.0, 0.0, 2.0])
    return tessellate_x(np.array([p0, p1, p1 + up, p0 + up]), n=24)


def main(argv=None):
    ap = argparse.ArgumentParser(description="上下楼梯与识别跨障关卡仿真")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    from levels.stairs_hurdle import StairsHurdleLevel
    scene = StairsScene(bar_dist=float(np.random.RandomState(args.seed).choice([25.0, 32.0, 40.0])))
    robot = SimStairsRobot(scene, seed=args.seed,
                           start=(0.0, -60.0), heading_deg=0.0)
    level = StairsHurdleLevel(robot)
    ok = level.run_level()
    print(f"[sim] ok={ok} 到达 y={robot.pos[1]:.1f} 拍照 {robot.n_captures} 张 "
          f"动作 {len(robot.action_log)} 次")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
