# -*- coding: utf-8 -*-
"""寻线+循迹模拟器 — 视觉/决策分离三模式

将机器人流程分为两条链路：
  - 视觉链路：set_head → capture_frame → detector.detect → LineResult
  - 决策链路：LineResult → seek_line/track_line → run_action/set_head

三种运行模式：
  --mode vision   : 只测视觉链路（合成帧 → 真实 LineDetector → 检查输出）
  --mode decision : 只测决策链路（预设 LineResult → 真实算法 → 检查动作序列）
  --mode full     : 联调（合成帧 → 真实检测器 → 真实算法 → 模拟运动）

运行：
  python -m sim.line_tracking_sim --mode vision
  python -m sim.line_tracking_sim --mode decision
  python -m sim.line_tracking_sim --mode full
  python -m sim.line_tracking_sim              # 默认 full
"""

import os
import sys
import time
import types
import argparse
import numpy as np
import cv2

import matplotlib
if not os.environ.get("MPLBACKEND"):
    try:
        matplotlib.use("TkAgg")
    except Exception:
        matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon as MplPolygon
from matplotlib.transforms import Affine2D

# 配置中文字体
for _font in ["Microsoft YaHei", "SimHei", "WenQuanYi Micro Hei", "Arial Unicode MS"]:
    try:
        matplotlib.rcParams["font.sans-serif"] = [_font] + matplotlib.rcParams["font.sans-serif"]
        break
    except Exception:
        continue
matplotlib.rcParams["axes.unicode_minus"] = False

# 项目根目录加入 sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# =====================================================================
# Phase 1: mock hiwonder + 导入 line_seeker_tracking
# =====================================================================

class _MockController:
    """模拟 hiwonder.Controller.Controller — set_pwm_servo_pulse 为 no-op"""
    def __init__(self, board=None):
        self._head_pulse = 1500
        self._pitch_pulse = 1500

    def set_pwm_servo_pulse(self, servo_id, pulse, move_time_ms):
        if servo_id == 2:
            self._head_pulse = pulse
        elif servo_id == 1:
            self._pitch_pulse = pulse


def _inject_mock_hiwonder():
    """在 sys.modules 注入 mock hiwonder 包，让 line_seeker_tracking 导入不报错"""
    # hiwonder 包
    hiwonder_mod = types.ModuleType("hiwonder")
    hiwonder_mod.__path__ = []
    sys.modules["hiwonder"] = hiwonder_mod

    # hiwonder.ActionGroupControl
    agc_mod = types.ModuleType("hiwonder.ActionGroupControl")
    def _mock_run_action(name, times=1, with_stand=True, path=""):
        pass
    agc_mod.runActionGroup = _mock_run_action
    sys.modules["hiwonder.ActionGroupControl"] = agc_mod
    hiwonder_mod.ActionGroupControl = agc_mod

    # hiwonder.ros_robot_controller_sdk
    rrc_mod = types.ModuleType("hiwonder.ros_robot_controller_sdk")
    class _MockBoard:
        def __init__(self):
            pass
    rrc_mod.Board = _MockBoard
    sys.modules["hiwonder.ros_robot_controller_sdk"] = rrc_mod
    hiwonder_mod.ros_robot_controller_sdk = rrc_mod

    # hiwonder.Controller
    ctl_mod = types.ModuleType("hiwonder.Controller")
    ctl_mod.Controller = _MockController
    sys.modules["hiwonder.Controller"] = ctl_mod
    hiwonder_mod.Controller = ctl_mod


_inject_mock_hiwonder()

# 现在可以安全导入 line_seeker_tracking
import levels.line_seeker_tracking as lst


# =====================================================================
# Phase 2: 配置常量
# =====================================================================

# 噪声参数
ACTION_ERROR_STD = 0.0       # 步长误差比例（0.1=±10%）
TURN_ERROR_STD = 0.0          # 转向误差标准差（度）
LINE_JITTER_STD = 0.0         # 线渲染抖动（像素）

# 动画参数
ANIM_PAUSE_SEC = 0.05
MAX_SIM_STEPS = 500

# 机器人初始状态
INITIAL_POS = np.array([5.0, 20.0], dtype=np.float64)
INITIAL_ANGLE_DEG = 0.0       # 0=朝东

# 赛道定义：含直道+直角弯的路径 (x, y) cm
# 赛道定义：含直道+直角弯的路径 (x, y) cm
# 各直线段 ≥60cm，确保 D_FAR=35cm 视野内直道中段看不到拐角后线段
LINE_WAYPOINTS = [
    (10, 20),    # 起点
    (70, 20),    # P1: 左弯（直道 60cm）
    (70, 100),   # P2: 右弯（竖直 80cm）
    (150, 100),  # P3: 右弯（水平 80cm）
    (150, 20),   # P4: 左弯（竖直 80cm）
    (210, 20),   # 终点（水平 60cm）
]

# 相机渲染参数
RENDER_WIDTH = 640
RENDER_HEIGHT = 480
D_NEAR = 5.0                  # 近端距离（cm）
D_FAR = 35.0                  # 远端距离（cm）—— 降至 35cm 更接近真实相机
W_NEAR = 20.0                 # 近端半宽（cm）
W_FAR = 50.0                  # 远端半宽（cm）
LINE_THICKNESS = 12           # 线宽（像素）

# 动作标定值（来自 levels/goodluck.py）
FORWARD_ONE_STEP_CM = 2.0
FORWARD_CM = 5.0
BACK_CM = 3.2
LEFT_MOVE_CM = 1.9
RIGHT_MOVE_CM = 2.2
TURN_LEFT_DEG = 22.0
TURN_RIGHT_DEG = 25.7

# 机器人边界框（cm）
ROBOT_WIDTH_CM = 26.0
ROBOT_LENGTH_CM = 10.0


# =====================================================================
# Phase 3: SimState — 虚拟机器人状态
# =====================================================================

class SimState:
    """维护机器人的真实位置/朝向/头部脉宽/轨迹

    坐标系：x=东, y=北, angle=0 朝东, 正角=左转(逆时针)
    """

    def __init__(self, pos, angle_deg):
        self.pos = np.array(pos, dtype=np.float64).copy()
        self.angle = np.radians(angle_deg)
        self.head_pulse = 1500
        self.trajectory = [self.pos.copy()]
        self.step_count = 0
        self.last_action = "init"
        self.action_log = []     # 记录所有动作（decision 模式用）

    @property
    def orientation_vec(self):
        return np.array([np.cos(self.angle), np.sin(self.angle)])

    @property
    def head_angle_rad(self):
        """头部偏转角度（弧度），正=左转"""
        return np.radians((self.head_pulse - 1500) * 0.09)

    def _record(self, action_name):
        self.last_action = action_name
        self.trajectory.append(self.pos.copy())
        self.action_log.append({
            "step": self.step_count,
            "action": action_name,
            "pos": self.pos.copy(),
            "angle": self.angle,
        })

    def apply_forward(self, cm, name="forward"):
        actual = cm
        if ACTION_ERROR_STD > 0:
            actual = cm * (1.0 + np.random.normal(0, ACTION_ERROR_STD))
        self.pos = self.pos + actual * self.orientation_vec
        self._record(name)

    def apply_back(self, cm, name="back"):
        actual = cm
        if ACTION_ERROR_STD > 0:
            actual = cm * (1.0 + np.random.normal(0, ACTION_ERROR_STD))
        self.pos = self.pos - actual * self.orientation_vec
        self._record(name)

    def apply_left_move(self, cm, name="left_move"):
        actual = cm
        if ACTION_ERROR_STD > 0:
            actual = cm * (1.0 + np.random.normal(0, ACTION_ERROR_STD))
        left_dir = np.array([-np.sin(self.angle), np.cos(self.angle)])
        self.pos = self.pos + actual * left_dir
        self._record(name)

    def apply_right_move(self, cm, name="right_move"):
        actual = cm
        if ACTION_ERROR_STD > 0:
            actual = cm * (1.0 + np.random.normal(0, ACTION_ERROR_STD))
        right_dir = np.array([np.sin(self.angle), -np.cos(self.angle)])
        self.pos = self.pos + actual * right_dir
        self._record(name)

    def apply_turn(self, deg, name="turn"):
        """转向：正=左转(逆时针)，负=右转(顺时针)"""
        actual_deg = deg
        if TURN_ERROR_STD > 0:
            noise = np.random.normal(0, TURN_ERROR_STD)
            if deg >= 0:
                actual_deg = abs(deg + noise)
            else:
                actual_deg = -abs(deg + noise)
        self.angle += np.radians(actual_deg)
        self._record(name)


# =====================================================================
# Phase 4: LinePath — 虚拟赛道
# =====================================================================

class LinePath:
    """虚拟赛道：waypoints → 密集点列"""

    def __init__(self, waypoints, step_cm=1.0):
        self.waypoints = [np.array(wp, dtype=np.float64) for wp in waypoints]
        self.points = self._generate_points(step_cm)

    def _generate_points(self, step_cm):
        pts = []
        for i in range(len(self.waypoints) - 1):
            p0 = self.waypoints[i]
            p1 = self.waypoints[i + 1]
            dist = np.linalg.norm(p1 - p0)
            n = max(2, int(dist / step_cm))
            for j in range(n):
                t = j / n
                pts.append(p0 + t * (p1 - p0))
        pts.append(self.waypoints[-1].copy())
        return np.array(pts)

    def world_to_robot(self, point, sim_state):
        """世界坐标 → 机器人坐标系 (forward, right)"""
        dx = point[0] - sim_state.pos[0]
        dy = point[1] - sim_state.pos[1]
        cos_a = np.cos(sim_state.angle)
        sin_a = np.sin(sim_state.angle)
        forward = cos_a * dx + sin_a * dy
        right = sin_a * dx - cos_a * dy
        return forward, right

    def robot_to_camera(self, forward, right, head_angle_rad):
        """机器人坐标系 → 相机坐标系 (f_cam, r_cam)"""
        cos_h = np.cos(head_angle_rad)
        sin_h = np.sin(head_angle_rad)
        f_cam = cos_h * forward - sin_h * right
        r_cam = sin_h * forward + cos_h * right
        return f_cam, r_cam


# =====================================================================
# Phase 5: FrameRenderer — 合成相机帧
# =====================================================================

class FrameRenderer:
    """合成相机帧：虚拟赛道 → 透视投影 → 640×480 BGR ndarray"""

    def __init__(self, line_path):
        self.line_path = line_path
        self.W = RENDER_WIDTH
        self.H = RENDER_HEIGHT
        self._H_matrix = self._compute_homography()

    def _compute_homography(self):
        """地面相机系 → 图像像素的单应矩阵

        源点(相机系 forward,right): 4 个梯形顶点
        目标点(图像): 映射到图像下半部（ROI 区域）
        """
        src = np.array([
            [D_NEAR, -W_NEAR],   # 近左
            [D_NEAR,  W_NEAR],   # 近右
            [D_FAR,   -W_FAR],   # 远左
            [D_FAR,   W_FAR],    # 远右
        ], dtype=np.float32)
        dst = np.array([
            [0,           self.H],           # 近左 → 图像左下
            [self.W,      self.H],           # 近右 → 图像右下
            [0,           self.H // 2],       # 远左 → 图像左中
            [self.W,      self.H // 2],       # 远右 → 图像右中
        ], dtype=np.float32)
        return cv2.getPerspectiveTransform(src, dst)

    def render(self, sim_state):
        """渲染当前视角的合成帧"""
        frame = np.full((self.H, self.W, 3), (40, 120, 40), dtype=np.uint8)

        # 赛道点列 World → Robot → Camera
        head_rad = sim_state.head_angle_rad
        cam_pts = []
        for wp in self.line_path.points:
            fwd, rgt = self.line_path.world_to_robot(wp, sim_state)
            f_cam, r_cam = self.line_path.robot_to_camera(fwd, rgt, head_rad)
            if f_cam > 0.5:   # 只取相机前方的点
                cam_pts.append([f_cam, r_cam])

        if len(cam_pts) < 2:
            return frame

        cam_arr = np.array(cam_pts, dtype=np.float32).reshape(-1, 1, 2)
        img_pts = cv2.perspectiveTransform(cam_arr, self._H_matrix).reshape(-1, 2)

        # 过滤图像范围内的连续段
        segments = []
        current_seg = []
        for pt in img_pts:
            x, y = pt
            if -50 <= x <= self.W + 50 and 0 <= y <= self.H:
                current_seg.append((int(x), int(y)))
            else:
                if len(current_seg) >= 2:
                    segments.append(current_seg)
                current_seg = []
        if len(current_seg) >= 2:
            segments.append(current_seg)

        # 画红色线
        for seg in segments:
            pts = np.array(seg, dtype=np.int32)
            cv2.polylines(frame, [pts], False, (0, 0, 255), LINE_THICKNESS)

        return frame


# =====================================================================
# Phase 6: Visualizer — 双视图可视化
# =====================================================================

class Visualizer:
    """matplotlib 双视图：俯视图 + 相机视图"""

    def __init__(self, line_path, mode="full"):
        self.line_path = line_path
        self.mode = mode
        self.fig, (self.ax_top, self.ax_cam) = plt.subplots(
            1, 2, figsize=(14, 7))
        self._draw_static_top()
        # 相机视图初始化
        self.ax_cam.set_title("相机视图", fontsize=12)
        self.ax_cam.set_xticks([])
        self.ax_cam.set_yticks([])
        self.cam_im = self.ax_cam.imshow(
            np.zeros((RENDER_HEIGHT, RENDER_WIDTH, 3), dtype=np.uint8))

        # 动态层句柄
        self.dynamic_artists = []
        self.fig.canvas.manager.set_window_title(
            f"寻线循迹模拟器 [{mode} 模式]")

    def _draw_static_top(self):
        ax = self.ax_top
        ax.set_xlim(-10, 220)
        ax.set_ylim(-10, 110)
        ax.set_aspect("equal")
        ax.set_title("俯视图", fontsize=12)
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")
        ax.grid(True, linestyle="--", alpha=0.3)

        # 赛道线
        pts = self.line_path.points
        ax.plot(pts[:, 0], pts[:, 1], "r-", linewidth=3, alpha=0.8)
        # 路点标注
        for i, wp in enumerate(self.line_path.waypoints):
            ax.plot(wp[0], wp[1], "ro", markersize=8, markeredgecolor="black")
            ax.annotate(f"P{i}", (wp[0], wp[1]),
                        textcoords="offset points", xytext=(6, 6),
                        fontsize=8, color="darkred")

    def update(self, sim_state, frame=None, action_text="", extra_info=""):
        """重绘动态层"""
        # 清除旧动态层
        for artist in self.dynamic_artists:
            artist.remove()
        self.dynamic_artists = []

        ax = self.ax_top
        # 轨迹
        traj = np.array(sim_state.trajectory)
        traj_line, = ax.plot(traj[:, 0], traj[:, 1], "c-", linewidth=1.5, alpha=0.7)
        self.dynamic_artists.append(traj_line)

        # 机器人位置
        pos = sim_state.pos
        ori = sim_state.orientation_vec
        angle_deg = np.degrees(sim_state.angle)

        # 机器人边框
        box_x = pos[0] - ROBOT_LENGTH_CM / 2
        box_y = pos[1] - ROBOT_WIDTH_CM / 2
        t = Affine2D().rotate_deg_around(pos[0], pos[1], angle_deg) + ax.transData
        robot_box = Rectangle(
            (box_x, box_y), ROBOT_LENGTH_CM, ROBOT_WIDTH_CM,
            facecolor=(1.0, 0.6, 0.6, 0.3), edgecolor="red", linewidth=1.2)
        robot_box.set_transform(t)
        ax.add_patch(robot_box)
        self.dynamic_artists.append(robot_box)

        # 朝向箭头
        arrow = ax.annotate(
            "", xy=(pos[0] + ori[0] * 5, pos[1] + ori[1] * 5),
            xytext=(pos[0], pos[1]),
            arrowprops=dict(arrowstyle="->", color="magenta", lw=2.5))
        self.dynamic_artists.append(arrow)
        ax.plot(pos[0], pos[1], "mo", markersize=5)
        self.dynamic_artists.append(
            ax.plot(pos[0], pos[1], "mo", markersize=5)[0])

        # 相机 FOV 梯形
        head_rad = sim_state.head_angle_rad
        cos_h, sin_h = np.cos(head_rad), np.sin(head_rad)
        fwd = sim_state.orientation_vec
        right_dir = np.array([sin_h * np.cos(sim_state.angle) + cos_h * np.sin(sim_state.angle),
                              sin_h * np.sin(sim_state.angle) - cos_h * np.cos(sim_state.angle)])
        # 简化：FOV 四角
        fov_pts = []
        for d, w in [(D_NEAR, W_NEAR), (D_NEAR, -W_NEAR),
                     (D_FAR, -W_FAR), (D_FAR, W_FAR)]:
            p = pos + d * fwd + w * right_dir
            fov_pts.append(p)
        fov_poly = MplPolygon(fov_pts, closed=True,
                              facecolor=(0.5, 0.8, 1.0, 0.15),
                              edgecolor="blue", linewidth=1, linestyle="--")
        ax.add_patch(fov_poly)
        self.dynamic_artists.append(fov_poly)

        # 动作文字
        info_text = action_text
        if extra_info:
            info_text += "\n" + extra_info
        txt = ax.text(
            0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=8, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
        self.dynamic_artists.append(txt)

        # 相机视图
        if frame is not None:
            self.cam_im.set_data(frame)
            self.ax_cam.set_title(f"相机视图 (head={sim_state.head_pulse})", fontsize=10)

        if ANIM_PAUSE_SEC > 0 and matplotlib.get_backend().lower() != "agg":
            plt.pause(ANIM_PAUSE_SEC)
        else:
            self.fig.canvas.draw_idle()


# =====================================================================
# Phase 7: Vision 模式 — 视觉链路独立测试
# =====================================================================

# 预设测试位置：(pos_x, pos_y, angle_deg, head_pulse, 描述)
# 拐角专项：不同距离接近拐角（30/45/55cm）+ 过弯后视角 + 转弯中途 45° 视角
VISION_TEST_CASES = [
    # ---- 直道：距 P1(70,20) >35cm，视野内看不到拐角后线段 ----
    (15, 20, 0, 1500, "直道居中", "follow"),
    (25, 20, 0, 1500, "直道居中2", "follow"),
    (30, 20, 0, 1500, "直道居中3", "follow"),
    (25, 18, 0, 1500, "直道偏左（线在右前方）", "follow"),
    (25, 22, 0, 1500, "直道偏右（线在左前方）", "follow"),
    (55, 20, 0, 1500, "接近弯道", "corner"),
    (58, 22, 0, 1500, "弯道入口附近", "corner"),
    (70, 25, 90, 1500, "已左转，朝北直道", "follow"),
    (70, 50, 90, 1500, "朝北直道中段", "follow"),
    (5, 80, 0, 1500, "远离赛道（应丢线）", "none"),
    (25, 20, 0, 1950, "头部左转40.5°", "any"),
    (25, 20, 0, 1050, "头部右转-40.5°", "any"),
    # ---- 拐角专项：P1(70,20) 左弯，不同距离接近 ----
    (30, 20, 0, 1500, "拐角专项: 距P1弯40cm（远距接近）", "follow"),
    (50, 20, 0, 1500, "拐角专项: 距P1弯20cm（中距接近）", "corner"),
    (63, 20, 0, 1500, "拐角专项: 距P1弯7cm（近距接近）", "corner"),
    # ---- 拐角专项：P1 弯距拐角 55/65cm（超远距，超出视野）----
    (5, 20, 0, 1500, "拐角专项: 距P1弯65cm（超远距接近）", "follow"),
    (15, 20, 0, 1500, "拐角专项: 距P1弯55cm（远距接近2）", "follow"),
    # ---- 拐角专项：P2(70,100) 右弯，过弯前/后视角 ----
    (70, 85, 90, 1500, "拐角专项: 距P2弯15cm（北上接近右弯）", "corner"),
    (70, 95, 90, 1500, "拐角专项: 距P2弯5cm（右弯入口）", "any"),
    (72, 98, 0, 1500, "拐角专项: 过P2弯后朝东看（新方向线）", "follow"),
    # ---- 拐角专项：P2 弯距拐角 30/45/55cm ----
    (70, 70, 90, 1500, "拐角专项: 距P2弯30cm（远距北上接近）", "corner"),
    (70, 55, 90, 1500, "拐角专项: 距P2弯45cm（超远距北上接近）", "follow"),
    (70, 45, 90, 1500, "拐角专项: 距P2弯55cm（极远距北上接近）", "follow"),
    # ---- 拐角专项：转弯中途 45° 视角（L 形不对称透视）----
    (66, 22, 45, 1500, "拐角专项: 转弯中途45°视角看P1弯", "any"),
    (72, 92, 135, 1500, "拐角专项: 转弯中途135°视角看P2弯", "any"),
    # ---- 拐角专项：过弯后视角（新方向线确认）----
    (72, 25, 90, 1500, "拐角专项: 过P1弯后朝北看（新方向线）", "follow"),
    (72, 30, 90, 1500, "拐角专项: 过P1弯后朝北看2（新方向线）", "follow"),
    (72, 95, 0, 1500, "拐角专项: 过P2弯后朝东看（新方向线）", "follow"),
    (75, 98, 0, 1500, "拐角专项: 过P2弯后朝东看2（新方向线）", "follow"),
    # ---- 拐角专项：P3(150,100) 右弯，不同距离 ----
    (135, 100, 0, 1500, "拐角专项: 距P3弯15cm（东向接近右弯）", "corner"),
    (125, 100, 0, 1500, "拐角专项: 距P3弯25cm（远距东向接近右弯）", "corner"),
    (115, 100, 0, 1500, "拐角专项: 距P3弯35cm（超远距东向接近右弯）", "corner"),
    # ---- 拐角专项：P4(150,20) 左弯，不同距离（朝南=270°接近）----
    (150, 35, 270, 1500, "拐角专项: 距P4弯15cm（南下接近左弯）", "corner"),
    (150, 40, 270, 1500, "拐角专项: 距P4弯20cm（远距南下接近左弯）", "corner"),
]


def run_vision_test():
    """Vision 模式：在预设位置渲染合成帧 → 跑真实 LineDetector → 检查输出"""
    print("=" * 60)
    print("===== Vision 模式：视觉链路独立测试 =====")
    print("=" * 60)

    line_path = LinePath(LINE_WAYPOINTS)
    renderer = FrameRenderer(line_path)
    sim_state = SimState(INITIAL_POS, INITIAL_ANGLE_DEG)
    viz = Visualizer(line_path, mode="vision")

    detector = lst.detector   # 真实 LineDetector

    # 统计
    pass_count = 0
    fail_count = 0
    explosion_count = 0       # |lateral_offset| > 500 视为拟合爆炸
    corner_expected = 0
    corner_detected = 0
    corner_correct_dir = 0

    for idx, (px, py, ang, head, desc, expect) in enumerate(VISION_TEST_CASES):
        print(f"\n--- 测试 {idx + 1}/{len(VISION_TEST_CASES)}: {desc} ---")
        sim_state.pos = np.array([px, py], dtype=np.float64)
        sim_state.angle = np.radians(ang)
        sim_state.head_pulse = head

        # 渲染合成帧
        frame = renderer.render(sim_state)

        # 跑真实 LineDetector
        result = detector.detect(frame)

        # 判定
        passed = False
        if expect == "none":
            passed = not result.exists
        elif expect == "any":
            passed = result.exists
        elif result.exists:
            p = result.primary
            if expect == "corner":
                passed = (p.orientation == "corner")
            elif expect == "follow":
                passed = (p.orientation == "follow")

        if passed:
            pass_count += 1
        else:
            fail_count += 1

        if result.exists:
            p = result.primary
            print(f"  检测到线: orientation={p.orientation}  "
                  f"lookahead_x={p.lookahead_x:.1f}  "
                  f"lateral_offset={p.lateral_offset:.1f}  "
                  f"heading={p.heading_deg:.1f}  "
                  f"curvature={p.curvature:.4f}  "
                  f"straightness={p.straightness:.3f}  "
                  f"{'✓' if passed else '✗'}")
            extra = (f"检测结果:\n"
                     f"  orientation={p.orientation}\n"
                     f"  lookahead_x={p.lookahead_x:.1f}\n"
                     f"  lateral_offset={p.lateral_offset:.1f}\n"
                     f"  heading={p.heading_deg:.1f}\n"
                     f"  curvature={p.curvature:.4f}\n"
                     f"  straightness={p.straightness:.3f}\n"
                     f"  期望={expect} {'✓' if passed else '✗'}")

            # 拐角方向统计
            if expect == "corner":
                corner_expected += 1
                if p.orientation == "corner":
                    corner_detected += 1
                    # P1 弯是左弯（curvature<0），P2 弯是右弯（curvature>0）
                    # P3 弯是右弯（curvature>0），P4 弯是左弯（curvature<0）
                    is_correct_dir = False
                    if "P1" in desc and p.curvature < 0:
                        is_correct_dir = True
                    elif "P2" in desc and p.curvature > 0:
                        is_correct_dir = True
                    elif "P3" in desc and p.curvature > 0:
                        is_correct_dir = True
                    elif "P4" in desc and p.curvature < 0:
                        is_correct_dir = True
                    if is_correct_dir:
                        corner_correct_dir += 1
                    print(f"  [拐角统计] 检出 corner，curvature={p.curvature:+.1f}"
                          f"（{'方向正确' if is_correct_dir else '方向错误!'}）")
            if abs(p.lateral_offset) > 500 or abs(p.lookahead_x) > 500:
                explosion_count += 1
                print(f"  [警告] 拟合爆炸: lateral_offset={p.lateral_offset:.1f} "
                      f"lookahead_x={p.lookahead_x:.1f}")
        else:
            print(f"  未检测到线 (exists=False)  {'✓' if passed else '✗'}")
            extra = f"检测结果: exists=False\n期望={expect} {'✓' if passed else '✗'}"

        viz.update(sim_state, frame,
                   action_text=f"测试 {idx + 1}: {desc}\n"
                               f"pos=({px},{py}) angle={ang}° head={head}",
                   extra_info=extra)

    # 统计汇总
    print("\n" + "=" * 60)
    print("===== Vision 测试统计 =====")
    print(f"通过: {pass_count}/{len(VISION_TEST_CASES)}")
    print(f"失败: {fail_count}/{len(VISION_TEST_CASES)}")
    print(f"拟合爆炸: {explosion_count} 次（|lateral_offset|>500 或 |lookahead_x|>500）")
    print(f"拐角专项: 期望 corner {corner_expected} 个，检出 {corner_detected}，方向正确 {corner_correct_dir}")
    if fail_count == 0 and explosion_count == 0:
        print(">>> Vision 测试全部通过 <<<")
    else:
        print(f">>> 有 {fail_count} 个测试未通过，需检查 <<<")

    print("\n===== Vision 模式测试完成 =====")
    if matplotlib.get_backend().lower() != "agg":
        print("关闭图形窗口退出。")
        plt.show()


# =====================================================================
# Phase 8: Decision 模式 — 决策链路独立测试
# =====================================================================

def _make_line_result(orientation="follow", lookahead_x=0.0,
                      lateral_offset=0.0, heading_deg=0.0,
                      curvature=0.0, straightness=0.95,
                      exists=True):
    """构造预设 LineResult"""
    from vision.detection import LineResult, LineSegment
    if not exists:
        return LineResult(exists=False)
    seg = LineSegment(
        orientation=orientation,
        heading_deg=heading_deg,
        curvature=curvature,
        straightness=straightness,
        lateral_offset=lateral_offset,
        lookahead_x=lookahead_x,
    )
    return LineResult(exists=True, primary=seg, others=[], confidence=straightness)


# 预设 LineResult 序列：每个场景是一组连续帧的检测结果
DECISION_TEST_SCENARIOS = {
    "A_直道前进": [
        _make_line_result("follow", lookahead_x=2.0, straightness=0.95),
        _make_line_result("follow", lookahead_x=-1.0, straightness=0.95),
        _make_line_result("follow", lookahead_x=3.0, straightness=0.95),
        _make_line_result("follow", lookahead_x=0.5, straightness=0.95),
        _make_line_result("follow", lookahead_x=-2.0, straightness=0.95),
    ],
    "B_偏左修正": [
        _make_line_result("follow", lookahead_x=-25.0, straightness=0.90),
        _make_line_result("follow", lookahead_x=-15.0, straightness=0.92),
        _make_line_result("follow", lookahead_x=-5.0, straightness=0.95),
        _make_line_result("follow", lookahead_x=1.0, straightness=0.95),
    ],
    "C_右弯": [
        _make_line_result("follow", lookahead_x=5.0, straightness=0.90),
        _make_line_result("corner", curvature=1.0, straightness=0.70),
        _make_line_result("corner", curvature=1.0, straightness=0.70),
        _make_line_result("follow", lookahead_x=30.0, straightness=0.88),
        _make_line_result("follow", lookahead_x=10.0, straightness=0.92),
        _make_line_result("follow", lookahead_x=2.0, straightness=0.95),
    ],
    "D_丢线恢复": [
        _make_line_result("follow", lookahead_x=1.0, straightness=0.95),
        _make_line_result(exists=False),
        _make_line_result(exists=False),
        _make_line_result("follow", lookahead_x=3.0, straightness=0.95),
    ],
    "E_横线跨越": [
        _make_line_result("follow", lookahead_x=2.0, straightness=0.95),
        _make_line_result("cross", heading_deg=90.0, straightness=0.95),
        _make_line_result("follow", lookahead_x=1.0, straightness=0.95),
    ],
}


def run_decision_test():
    """Decision 模式：预设 LineResult → 真实算法 → 检查动作序列"""
    print("=" * 60)
    print("===== Decision 模式：决策链路独立测试 =====")
    print("=" * 60)

    line_path = LinePath(LINE_WAYPOINTS)
    sim_state = SimState(INITIAL_POS, INITIAL_ANGLE_DEG)
    viz = Visualizer(line_path, mode="decision")

    # 为每个场景创建独立的 mock detect
    for scenario_name, results in DECISION_TEST_SCENARIOS.items():
        print(f"\n{'=' * 40}")
        print(f"  场景: {scenario_name}")
        print(f"  预设 {len(results)} 帧 LineResult")
        print(f"{'=' * 40}")

        # 重置状态
        sim_state.__init__(INITIAL_POS, INITIAL_ANGLE_DEG)
        sim_state.step_count = 0

        # mock detect 返回预设序列
        call_idx = [0]  # 闭包计数器

        def mock_detect(frame):
            if call_idx[0] < len(results):
                r = results[call_idx[0]]
                call_idx[0] += 1
                return r
            return _make_line_result(exists=False)

        # mock capture_frame 返回空白帧
        def mock_capture_frame():
            return np.zeros((RENDER_HEIGHT, RENDER_WIDTH, 3), dtype=np.uint8)

        # mock run_action 记录并模拟运动
        def mock_run_action(name, times=1, with_stand=True):
            for _ in range(times):
                sim_state.step_count += 1
                if sim_state.step_count > MAX_SIM_STEPS:
                    raise RuntimeError(f"步数超限({MAX_SIM_STEPS})")
                if name == "stand":
                    sim_state._record("stand")
                elif name == "go_forward_one_step":
                    sim_state.apply_forward(FORWARD_ONE_STEP_CM, name)
                elif name == "go_forward":
                    sim_state.apply_forward(FORWARD_CM, name)
                elif name == "back_one_step":
                    sim_state.apply_back(BACK_CM, name)
                elif name == "left_move":
                    sim_state.apply_left_move(LEFT_MOVE_CM, name)
                elif name == "right_move":
                    sim_state.apply_right_move(RIGHT_MOVE_CM, name)
                elif name == "turn_left":
                    sim_state.apply_turn(TURN_LEFT_DEG, name)
                elif name == "turn_right":
                    sim_state.apply_turn(-TURN_RIGHT_DEG, name)
                elif name == "turn_left_small_step":
                    sim_state.apply_turn(TURN_LEFT_DEG / 2, name)
                elif name == "turn_right_small_step":
                    sim_state.apply_turn(-TURN_RIGHT_DEG / 2, name)
                else:
                    sim_state._record(name)
                viz.update(sim_state, frame=None,
                           action_text=f"场景: {scenario_name}\n"
                                       f"step={sim_state.step_count} action={name}",
                           extra_info=f"动作序列: {[a['action'] for a in sim_state.action_log[-10:]]}")

        # mock set_head
        def mock_set_head(pulse, move_time_ms=500):
            sim_state.head_pulse = pulse

        # 保存原始函数
        orig_capture = lst.capture_frame
        orig_detect = lst.detector.detect
        orig_run_action = lst.run_action
        orig_set_head = lst.set_head
        orig_ctl = lst._ctl

        try:
            # monkey-patch
            lst.capture_frame = mock_capture_frame
            lst.detector.detect = mock_detect
            lst.run_action = mock_run_action
            lst.set_head = mock_set_head
            lst._ctl = _MockController()

            # 跑算法（只跑 track_line，跳过 seek）
            try:
                lst.track_line(lst.detector)
            except RuntimeError as e:
                print(f"  [算法异常] {e}")

            # 打印动作序列
            actions = [a["action"] for a in sim_state.action_log]
            print(f"  动作序列 ({len(actions)} 步): {actions}")
            print(f"  最终位置: ({sim_state.pos[0]:.1f}, {sim_state.pos[1]:.1f})  "
                  f"角度: {np.degrees(sim_state.angle):.1f}°")

        finally:
            # 恢复原始函数
            lst.capture_frame = orig_capture
            lst.detector.detect = orig_detect
            lst.run_action = orig_run_action
            lst.set_head = orig_set_head
            lst._ctl = orig_ctl

    print("\n===== Decision 模式测试完成 =====")
    if matplotlib.get_backend().lower() != "agg":
        print("关闭图形窗口退出。")
        plt.show()


# =====================================================================
# Phase 9: Full 模式 — 联调
# =====================================================================

def run_full_test():
    """Full 模式：合成帧 → 真实检测器 → 真实算法 → 模拟运动"""
    print("=" * 60)
    print("===== Full 模式：视觉+决策联调 =====")
    print("=" * 60)

    line_path = LinePath(LINE_WAYPOINTS)
    renderer = FrameRenderer(line_path)
    sim_state = SimState(INITIAL_POS, INITIAL_ANGLE_DEG)
    viz = Visualizer(line_path, mode="full")

    # mock capture_frame → 合成帧
    def mock_capture_frame():
        return renderer.render(sim_state)

    # mock run_action → 模拟运动 + 可视化
    def mock_run_action(name, times=1, with_stand=True):
        for _ in range(times):
            sim_state.step_count += 1
            if sim_state.step_count > MAX_SIM_STEPS:
                raise RuntimeError(f"步数超限({MAX_SIM_STEPS})，算法可能不收敛")
            if name == "stand":
                sim_state._record("stand")
            elif name == "go_forward_one_step":
                sim_state.apply_forward(FORWARD_ONE_STEP_CM, name)
            elif name == "go_forward":
                sim_state.apply_forward(FORWARD_CM, name)
            elif name == "back_one_step":
                sim_state.apply_back(BACK_CM, name)
            elif name == "left_move":
                sim_state.apply_left_move(LEFT_MOVE_CM, name)
            elif name == "right_move":
                sim_state.apply_right_move(RIGHT_MOVE_CM, name)
            elif name == "turn_left":
                sim_state.apply_turn(TURN_LEFT_DEG, name)
            elif name == "turn_right":
                sim_state.apply_turn(-TURN_RIGHT_DEG, name)
            elif name == "turn_left_small_step":
                sim_state.apply_turn(TURN_LEFT_DEG / 2, name)
            elif name == "turn_right_small_step":
                sim_state.apply_turn(-TURN_RIGHT_DEG / 2, name)
            else:
                sim_state._record(name)

            frame = renderer.render(sim_state)
            viz.update(sim_state, frame,
                       action_text=f"step={sim_state.step_count} action={name}\n"
                                   f"pos=({sim_state.pos[0]:.1f}, {sim_state.pos[1]:.1f})\n"
                                   f"angle={np.degrees(sim_state.angle):.1f}°")

    # mock set_head
    def mock_set_head(pulse, move_time_ms=500):
        sim_state.head_pulse = pulse

    # 保存原始函数
    orig_capture = lst.capture_frame
    orig_run_action = lst.run_action
    orig_set_head = lst.set_head
    orig_ctl = lst._ctl

    try:
        # monkey-patch
        lst.capture_frame = mock_capture_frame
        lst.run_action = mock_run_action
        lst.set_head = mock_set_head
        lst._ctl = _MockController()

        # 初始画面
        frame = renderer.render(sim_state)
        viz.update(sim_state, frame, "模拟器就绪\n等待启动...")

        # 跑完整算法
        lst.run_line_tracking()

    except RuntimeError as e:
        print(f"[Full 模式] 算法异常: {e}")
    except Exception as e:
        print(f"[Full 模式] 未预期异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 恢复原始函数
        lst.capture_frame = orig_capture
        lst.run_action = orig_run_action
        lst.set_head = orig_set_head
        lst._ctl = orig_ctl

    # 打印摘要
    print(f"\n===== Full 模式结束 =====")
    print(f"总步数: {sim_state.step_count}")
    print(f"最终位置: ({sim_state.pos[0]:.1f}, {sim_state.pos[1]:.1f})  "
          f"角度: {np.degrees(sim_state.angle):.1f}°")
    if sim_state.action_log:
        print(f"\n动作序列摘要 ({len(sim_state.action_log)} 步):")
        for sa in sim_state.action_log:
            print(f"  #{sa['step']:>3d}  {sa['action']:<28s}  "
                  f"pos=({sa['pos'][0]:6.1f}, {sa['pos'][1]:6.1f})")

    if matplotlib.get_backend().lower() != "agg":
        print("关闭图形窗口退出。")
        plt.show()


# =====================================================================
# Phase 10: 统一入口
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="寻线+循迹模拟器（视觉/决策分离三模式）")
    parser.add_argument(
        "--mode", choices=["vision", "decision", "full"],
        default="full", help="运行模式（默认 full）")
    args = parser.parse_args()

    if args.mode == "vision":
        run_vision_test()
    elif args.mode == "decision":
        run_decision_test()
    else:
        run_full_test()


if __name__ == "__main__":
    main()
