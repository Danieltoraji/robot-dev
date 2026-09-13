#!/usr/bin/env python3
# coding=utf-8
"""放下蓝色方块后「后退→找红线→循迹」独立调试脚本。

巡线算法移植自 levels/line_seeker_tracking.py（基于 vision/line_detector.py 的
LineDetector），替代原先基于单一 line_cx 的简单巡线。流程:
  1. back_fast 后退 back_steps_after_place 步 (假定方块已放下、手臂已复位)
  2. stand
  3. seek_line(): 头部扫描 + 身体旋转 + 平移试探, 找到红色路径线并对准
  4. track_line(): 沿线行进, 处理直道/直角弯/丢线, far_ry 判定终点

用法(真机):
    /home/pi/jupyter-env/bin/python3 line_follow_debug.py [--dry-run]
"""

import argparse
import glob
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, "/home/pi/TonyPi")
sys.path.insert(0, "/home/pi/TonyPi/Functions")
sys.path.insert(0, "/home/pi/TonyPi/HiwonderSDK")

# 让脚本能从任意目录 import 到 vision 包内的 line_detector 模块
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_VISION_DIR = _PROJECT_ROOT / "vision"
for _extra in (_PROJECT_ROOT, _VISION_DIR):
    _extra_str = str(_extra)
    if _extra_str not in sys.path:
        sys.path.insert(0, _extra_str)

from line_detector import LineDetector

import hiwonder.ActionGroupControl as AGC
import hiwonder.ros_robot_controller_sdk as rrc
import hiwonder.yaml_handle as yaml_handle
from hiwonder.Controller import Controller
from CameraCalibration.CalibrationConfig import calibration_param_path


LOG_FILE = "/home/pi/JustForTestNoUse/BlueCube_test/line_follow_debug.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler(LOG_FILE, mode="a"), logging.StreamHandler()],
)
log = logging.getLogger("line_follow_debug")


# ---- 摄像头分辨率 ----
CAM_SIZE = (640, 480)

# Head scanning.
HEAD_STEP_X = 15
HEAD_STEP_Y = 15

# 夹爪闭合程度: 1.0=最紧(8=0,16=1000), 0.5=半开, 0=全开(500)
GRIP_SCALE = 1.0
LOCK_SERVOS = {"6": 700, "7": 820, "8": int(500 - 500*GRIP_SCALE), "14": 300, "15": 180, "16": int(500 + 500*GRIP_SCALE)}

# =====================================================================
# 红色路径 HSV 范围 + LineDetector 配置（来自 line_seeker_tracking.py）
# =====================================================================

RED_LOW = (0, 0, 50)
RED_HIGH = (17, 255, 255)
RED_LOW_2 = (149, 0, 50)
RED_HIGH_2 = (180, 255, 255)

# =====================================================================
# 循迹决策阈值（工作分辨率 640px 宽下，与 line_seeker_tracking 一致）
# =====================================================================

T_CENTER = 12.0        # |lookahead_x| ≤ 此值 → 居中，可前进
T_BIG = 35.0           # |lookahead_x| > 此值 → 需转向修朝向
LOOKAHEAD_CLAMP = 200.0  # lookahead_x 绝对值钳制上限（防止多项式外推爆炸）
T_LAT = 40.0           # 对齐阶段侧边出线阈值

# 寻线参数
SEEK_BODY_TURNS = 14       # 身体单向扫描最大次数（turn_right×14 ≈ 360° 全周）
SEEK_MAX_PROBES = 3        # 全周扫描失败后的平移试探轮数
SEEK_PROBE_FORWARD = 5     # 每轮试探前进步数
SEEK_PROBE_LEFT = 5        # 每轮试探左移步数
SEEK_PROBE_RIGHT = 10      # 每轮试探右移步数（覆盖左移回程+更远）
TRACK_MAX_BACKS = 3        # 循迹丢线时连续后退最大次数，超限回寻线
ALIGN_MAX_STEPS = 20       # 对齐阶段最大步数（防死循环）
ALIGN_HEADING_TOL = 15.0   # 对齐阶段 heading_deg 容忍阈值（度）

# 转弯闭环参数
HEADING_BIG = 25.0             # |heading_deg| ≥ 此值 → 大步粗修朝向
HEADING_SMALL = 8.0            # 此值 ≤ |heading_deg| < HEADING_BIG → 小步精修朝向
FOLLOW_HEADING_MIN = 15.0      # follow 循迹：|heading| > 此值优先修朝向
FOLLOW_HEADING_MAX = 35.0      # follow 循迹：|heading| > 此值疑为拐角误判/抖动，忽略朝向只看横向
CORNER_EXIT_HEADING = 12.0     # 过弯出口：|heading| ≤ 此值视为已对准新方向
CORNER_INITIAL_TURN_STEPS = 2  # 定点转弯初始大步数
CORNER_MAX_CUMULATIVE_DEG = 135.0  # 单弯累计最大转角
CORNER_MAX_TURN_STEPS = 15     # 定点转弯闭环最大迭代次数（防死循环）
STUCK_MAX = 6                  # 循迹中连续 cross/丢线 超过此值则回寻线

# 终点判定参数
LINE_END_FAR_RY = 100.0        # follow 线远端 ry 小于此值视为线末端接近
LINE_END_FORWARD_MAX = 8       # 终点前进最大步数（丢线即停）
LINE_END_BLIND_MAX = 2         # 丢线后再盲目前进步数

# 头部扫描顺序：(servo2 相对中心偏移, 对应身体转角修正角度，左正右负)
HEAD_SCAN_SEQUENCE = [
    (0, 0.0),
    (450, 40.5),
    (-450, -40.5),
    (700, 63.0),
    (-700, -63.0),
]

# 动作标定值（来自 levels/goodluck.py 实机标定）
TURN_LEFT_DEG = 22.0
TURN_RIGHT_DEG = 25.7
TURN_LEFT_SMALL_DEG = TURN_LEFT_DEG / 2    # ≈ 11.0°
TURN_RIGHT_SMALL_DEG = TURN_RIGHT_DEG / 2  # ≈ 12.85°

# 直角弯「前进接近」阶段参数
CORNER_APPROACH_MAX = 16        # 接近阶段最大前进次数
CORNER_APPROACH_FAR_RY = 120.0  # follow 线远端 ry 小于此值视为拐角已接近
CORNER_APPROACH_ELBOW_RY = 40.0  # corner 肘点前向距离小于此值视为拐角点已接近


class ActionRunner:
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.lock_servos = ""

    def run(self, name: str, times: int = 1, lock: bool = False, with_stand: bool = False):
        if not name or times <= 0:
            return
        lock_servos = self.lock_servos if lock else ""
        if self.dry_run:
            log.info("[dry-run] action=%s times=%s lock=%s with_stand=%s", name, times, bool(lock_servos), with_stand)
            time.sleep(0.05)
            return
        log.info("action=%s times=%s lock=%s with_stand=%s", name, times, bool(lock_servos), with_stand)
        # 防止runningAction卡True(上一动作异常中断时会发生),否则所有动作被runAction跳过
        if AGC.runningAction:
            log.warning("runningAction stuck before '%s', resetting", name)
            AGC.runningAction = False
        AGC.runActionGroup(name, times=times, lock_servos=lock_servos, with_stand=with_stand)
        if AGC.runningAction:
            log.warning("runningAction stuck after '%s', resetting", name)
            AGC.runningAction = False

    def lock_hands(self):
        self.lock_servos = LOCK_SERVOS
        log.info("hand servos locked")

    def unlock_hands(self):
        self.lock_servos = ""
        log.info("hand servos unlocked")


class LineFollowDebug:
    def __init__(self, args):
        self.args = args
        self.actions = ActionRunner(dry_run=args.dry_run)

        self.servo_data = yaml_handle.get_yaml_data(yaml_handle.servo_file_path)
        self.servo1 = int(self.servo_data["servo1"])
        self.servo2 = int(self.servo_data["servo2"])
        self.x_dis = self.servo2
        self.y_dis = self.servo1
        self.head_mode = "left_right"
        self.head_turn = "left_right"
        self.d_x = HEAD_STEP_X
        self.d_y = HEAD_STEP_Y

        self.line_head_delta = int(args.line_head_delta)
        self.max_total_steps = int(args.max_total_steps)
        self.last_debug_save = 0.0

        # 巡线检测器（与 line_seeker_tracking.py 相同配置）
        self.detector = LineDetector(
            hsv_ranges=[(RED_LOW, RED_HIGH), (RED_LOW_2, RED_HIGH_2)],
            line_color="light",
            min_area=80,
            lookahead_ratio=0.5,
            straightness_thresh=0.85,
            work_width=640,
        )

        self.running = False
        self.services_stopped = False
        self.robot_initialized = False

        param_data = np.load(calibration_param_path + ".npz")
        mtx = param_data["mtx_array"]
        dist = param_data["dist_array"]
        newcameramtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, CAM_SIZE, 0, CAM_SIZE)
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, CAM_SIZE, 5)

        # Reuse AGC's board/controller.
        self.board = AGC.board
        self.ctl = AGC.ctl

    def init_robot(self):
        if not self.args.no_kill and not self.args.dry_run:
            for pattern in ("TonyPi.py", "Joystick.py", "transport_color.py", "blue_pickup_overhead.py"):
                subprocess.run(["pkill", "-f", pattern], stderr=subprocess.DEVNULL)
            # 停掉开机自启服务,否则pkill后会被立刻重启,抢串口导致动作组set_bus_servo_pulse失败
            subprocess.run(["sudo", "systemctl", "stop", "tonypi", "joystick"], stderr=subprocess.DEVNULL)
            self.services_stopped = True
            time.sleep(2.0)
            # 重新初始化串口: AGC.board在import时创建,若当时TonyPi.py占串口会连接不良,stop后必须重建
            AGC.board = rrc.Board()
            AGC.ctl = Controller(AGC.board)
            self.board = AGC.board
            self.ctl = AGC.ctl
        self.set_head_center(duration=500)
        self.actions.run("stand")
        self.robot_initialized = True

    def set_head(self, x: Optional[int] = None, y: Optional[int] = None, duration: int = 100):
        if x is not None:
            self.x_dis = int(x)
        if y is not None:
            self.y_dis = int(y)
        if self.args.dry_run:
            log.info("[dry-run] head x=%d y=%d duration=%d", self.x_dis, self.y_dis, duration)
            return
        self.ctl.set_pwm_servo_pulse(1, self.y_dis, duration)
        self.ctl.set_pwm_servo_pulse(2, self.x_dis, duration)

    def set_head_center(self, duration: int = 300):
        self.head_mode = "left_right"
        self.head_turn = "left_right"
        self.d_x = HEAD_STEP_X
        self.d_y = HEAD_STEP_Y
        self.set_head(self.servo2, self.servo1, duration)
        time.sleep(duration / 1000.0 + 0.05)

    def open_camera(self):
        open_once = yaml_handle.get_yaml_data("/boot/camera_setting.yaml")["open_once"]
        candidates = sorted(glob.glob("/dev/v4l/by-id/*-video-index0"))
        candidates.extend("/dev/video%d" % index for index in range(10))
        if open_once:
            candidates.append("http://127.0.0.1:8080/?action=stream?dummy=param.mjpg")
        for device in candidates:
            camera = cv2.VideoCapture(device)
            if not camera.isOpened():
                camera.release()
                continue
            if str(device).startswith("/dev/"):
                camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("Y", "U", "Y", "V"))
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_SIZE[0])
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_SIZE[1])
            for _ in range(8):
                ok, frame = camera.read()
                if ok and frame is not None:
                    log.info("camera opened: %s frame=%dx%d", device, frame.shape[1], frame.shape[0])
                    return camera
                time.sleep(0.05)
            camera.release()
        raise RuntimeError("no working USB camera capture node found")

    # =====================================================================
    # 底层感知/动作适配（移植自 line_seeker_tracking.py）
    # =====================================================================

    def capture_frame(self):
        """从实时流取一帧并去畸变；失败返回 None"""
        if self.camera is None:
            return None
        ok, frame = self.camera.read()
        if not ok or frame is None:
            return None
        return cv2.remap(frame, self.mapx, self.mapy, cv2.INTER_LINEAR)

    def _head_lr(self, offset: int, duration: int = 400):
        """左右转头看线（offset 相对 servo2 中心，正=左）"""
        self.set_head(self.servo2 + offset, self.servo1 + self.line_head_delta, duration)
        time.sleep(duration / 1000.0 + 0.05)

    def _run_action(self, name: str, times: int = 1):
        """执行动作组，每步回站立（对齐 line_seeker_tracking 语义）"""
        self.actions.run(name, times=times, with_stand=True)

    def _turn_by_angle(self, angle_deg):
        """按角度闭环转向：正=左转（逆时针），负=右转（顺时针）。

        用「大步 turn_left/right + 小步 turn_left/right_small_step」组合逼近目标角度。
        返回近似实际转角（度，左正右负）。
        """
        if abs(angle_deg) < HEADING_SMALL:
            return 0.0
        if angle_deg > 0:
            big_deg, small_deg = TURN_LEFT_DEG, TURN_LEFT_SMALL_DEG
            big_action, small_action = "turn_left", "turn_left_small_step"
            sign = 1.0
        else:
            big_deg, small_deg = TURN_RIGHT_DEG, TURN_RIGHT_SMALL_DEG
            big_action, small_action = "turn_right", "turn_right_small_step"
            sign = -1.0

        remaining = abs(angle_deg)
        actual = 0.0
        big_count = int(remaining // big_deg)
        remaining -= big_count * big_deg
        small_count = int(round(remaining / small_deg))

        if big_count > 0:
            self._run_action(big_action, times=big_count)
            actual += big_count * big_deg
        if small_count > 0:
            self._run_action(small_action, times=small_count)
            actual += small_count * small_deg

        log.info("[转向] 目标 %+.1f° → %s×%d + %s×%d ≈ %+.1f°",
                 angle_deg, big_action, big_count, small_action, small_count, sign * actual)
        return sign * actual

    def _correct_heading(self, heading_deg):
        """按观测 heading 闭环修朝向：heading>0=线在前方偏右 → 身体右转（负角）"""
        return self._turn_by_angle(-heading_deg)

    # =====================================================================
    # Phase 1: 寻线阶段
    # =====================================================================

    def _scan_head(self, prefer_side=None):
        """头部扫描：按顺序转头检测，找到线即返回。

        返回 (found, head_offset, head_angle_deg, result)
        """
        sequence = list(HEAD_SCAN_SEQUENCE)
        if prefer_side == "left":
            sequence.sort(key=lambda t: -t[1])
        elif prefer_side == "right":
            sequence.sort(key=lambda t: t[1])
        for offset, angle_deg in sequence:
            self._head_lr(offset)
            frame = self.capture_frame()
            if frame is None:
                continue
            result = self.detector.detect(frame)
            if result.exists:
                log.info("[寻线] 头部偏移 %d（%+.1f°）发现线：orientation=%s lookahead_x=%.1f",
                         offset, angle_deg, result.primary.orientation,
                         result.primary.lookahead_x)
                return True, offset, angle_deg, result
        return False, 0, 0.0, None

    def _turn_body_to_head_angle(self, head_angle_deg, max_turns=3):
        """根据头部偏转角度执行身体转向，使身体正对线"""
        if abs(head_angle_deg) < 1.0:
            self._head_lr(0)
            return 0.0

        if head_angle_deg > 0:
            turns = max(1, min(max_turns, int(round(head_angle_deg / TURN_LEFT_DEG))))
            log.info("[寻线] 头部左偏 %.1f°，身体左转 %d 次", head_angle_deg, turns)
            self._head_lr(0)
            self._run_action("turn_left", times=turns)
            return turns * TURN_LEFT_DEG
        else:
            turns = max(1, min(max_turns, int(round(abs(head_angle_deg) / TURN_RIGHT_DEG))))
            log.info("[寻线] 头部右偏 %.1f°，身体右转 %d 次", head_angle_deg, turns)
            self._head_lr(0)
            self._run_action("turn_right", times=turns)
            return -turns * TURN_RIGHT_DEG

    def _align_to_line(self, context="seek"):
        """统一对准线：优先级 corner→完成；cross→跨越；follow 依次处理 heading/lateral/lookahead"""
        tag = "对齐" if context == "seek" else "转弯后对准"
        for step in range(ALIGN_MAX_STEPS):
            self._head_lr(0)
            frame = self.capture_frame()
            if frame is None:
                log.info("[%s] 拍照失败", tag)
                return False
            result = self.detector.detect(frame)
            if not result.exists:
                log.info("[%s] 丢线", tag)
                return False

            p = result.primary
            lx = max(-LOOKAHEAD_CLAMP, min(LOOKAHEAD_CLAMP, p.lookahead_x))
            lateral = p.lateral_offset
            log.info("[%s] step=%d orientation=%s lookahead_x=%.1f heading=%.1f lateral=%.1f",
                     tag, step, p.orientation, lx, p.heading_deg, lateral)

            if p.orientation == "corner":
                if context == "seek":
                    log.info("[%s] 检测到拐角，直接进入循迹", tag)
                    return True
                turn_sign = -1.0 if p.curvature > 0 else 1.0
                big_deg = TURN_LEFT_DEG if turn_sign > 0 else TURN_RIGHT_DEG
                direction = "右" if turn_sign < 0 else "左"
                log.info("[%s] 仍是拐角（%s），补转 %+.1f°", tag, direction, turn_sign * big_deg)
                self._turn_by_angle(turn_sign * big_deg)
                continue

            if p.orientation == "cross":
                log.info("[%s] 检测到横线，跨越", tag)
                self._run_action("go_forward_one_step")
                continue

            if abs(p.heading_deg) > ALIGN_HEADING_TOL:
                log.info("[%s] heading=%.1f°，闭环修朝向", tag, p.heading_deg)
                self._correct_heading(p.heading_deg)
                continue

            if abs(lateral) > T_LAT:
                if lateral > 0:
                    log.info("[%s] lateral=%.1f > %.1f，右移对齐", tag, lateral, T_LAT)
                    self._run_action("right_move")
                else:
                    log.info("[%s] lateral=%.1f < -%.1f，左移对齐", tag, lateral, T_LAT)
                    self._run_action("left_move")
                continue

            if abs(lx) > T_CENTER:
                if lx > 0:
                    log.info("[%s] lookahead_x=%.1f，右移对齐", tag, lx)
                    self._run_action("right_move")
                else:
                    log.info("[%s] lookahead_x=%.1f，左移对齐", tag, lx)
                    self._run_action("left_move")
                continue

            log.info("[%s] lookahead_x=%.1f ≤ %.1f，heading=%.1f°，对准完成",
                     tag, lx, T_CENTER, p.heading_deg)
            return True

        log.info("[%s] 超过最大步数，返回", tag)
        return False

    def seek_line(self):
        """寻线主函数：头部扫描 → 单向身体扫描（全周）→ 平移试探，返回 True 表示寻线成功"""
        log.info("=" * 50)
        log.info("===== 开始寻线 =====")
        log.info("=" * 50)

        self._run_action("stand")
        self._head_lr(0)

        found, _, head_angle, _ = self._scan_head()
        if found:
            self._turn_body_to_head_angle(head_angle)
            if self._align_to_line():
                log.info("===== 寻线成功，进入循迹 =====")
                return True
            log.info("[寻线] 对齐阶段丢线，进入身体扫描")

        for body_turn in range(SEEK_BODY_TURNS):
            log.info("[寻线] 身体扫描第 %d/%d 次（右转）", body_turn + 1, SEEK_BODY_TURNS)
            self._run_action("turn_right")
            self._head_lr(0)
            frame = self.capture_frame()
            if frame is not None:
                result = self.detector.detect(frame)
                if result.exists:
                    log.info("[寻线] 身体扫描第 %d 次发现线：orientation=%s lookahead_x=%.1f",
                             body_turn + 1, result.primary.orientation,
                             result.primary.lookahead_x)
                    if self._align_to_line():
                        log.info("===== 寻线成功，进入循迹 =====")
                        return True
                    log.info("[寻线] 第 %d 次身体扫描后对齐失败，继续转", body_turn + 1)

        for probe in range(SEEK_MAX_PROBES):
            log.info("[寻线] 平移试探第 %d/%d 轮", probe + 1, SEEK_MAX_PROBES)
            for action_name, times in (
                ("go_forward_one_step", SEEK_PROBE_FORWARD),
                ("left_move", SEEK_PROBE_LEFT),
                ("right_move", SEEK_PROBE_RIGHT),
            ):
                self._run_action(action_name, times=times)
                found, _, head_angle, _ = self._scan_head()
                if found:
                    self._turn_body_to_head_angle(head_angle)
                    if self._align_to_line():
                        log.info("===== 寻线成功，进入循迹 =====")
                        return True
            log.info("[寻线] 第 %d 轮平移试探无果", probe + 1)

        log.info("===== 寻线失败：未找到线 =====")
        self._head_lr(0)
        return False

    # =====================================================================
    # Phase 2: 循迹阶段
    # =====================================================================

    def _handle_corner(self, curvature):
        """弯道处理（闭环）：前进接近拐角 → 逐步定点转弯 → 直到 follow 线 heading 归零。

        curvature < 0 → 左弯（左转，正角）；curvature > 0 → 右弯（右转，负角）。
        转弯量由「实时 heading」决定，不再依赖 corner_deg（透视下低估真实直角弯）。
        返回 True 表示弯已转过，False 表示异常（丢线或超限）。
        """
        turn_sign = -1.0 if curvature > 0 else 1.0
        direction = "左" if turn_sign > 0 else "右"
        big_deg = TURN_LEFT_DEG if turn_sign > 0 else TURN_RIGHT_DEG
        small_deg = TURN_LEFT_SMALL_DEG if turn_sign > 0 else TURN_RIGHT_SMALL_DEG

        log.info("[转弯] %s弯（curvature=%.2f），先前进接近拐角", direction, curvature)

        # 阶段1：前进接近拐角点
        for approach in range(CORNER_APPROACH_MAX):
            self._head_lr(0)
            frame = self.capture_frame()
            if frame is None:
                log.info("[转弯] 拍照失败")
                return False
            result = self.detector.detect(frame)
            reached = False
            if result.exists:
                p = result.primary
                if p.orientation == "cross":
                    log.info("[转弯] 检测到横线，已到拐角点，开始定点转弯")
                    reached = True
                elif p.orientation == "follow" and 0 < p.far_ry < CORNER_APPROACH_FAR_RY:
                    log.info("[转弯] 线末端接近（far_ry=%.0f < %.0f），拐角在前方，开始定点转弯",
                             p.far_ry, CORNER_APPROACH_FAR_RY)
                    reached = True
                elif p.orientation == "corner" and 0 < p.elbow_ry < CORNER_APPROACH_ELBOW_RY:
                    log.info("[转弯] 拐角点已接近（elbow_ry=%.0f < %.0f），开始定点转弯",
                             p.elbow_ry, CORNER_APPROACH_ELBOW_RY)
                    reached = True
                else:
                    log.info("[转弯] 接近中 orientation=%s（继续前进）", p.orientation)
            else:
                log.info("[转弯] 接近中丢线（继续前进）")
            if reached:
                break
            self._run_action("go_forward_one_step")
            log.info("[转弯] 接近拐角前进 %d/%d 步", approach + 1, CORNER_APPROACH_MAX)
        else:
            log.info("[转弯] 接近阶段前进用尽，按当前位置定点转弯")

        # 阶段2：逐步闭环定点转弯
        cumulative_angle = self._turn_by_angle(turn_sign * CORNER_INITIAL_TURN_STEPS * big_deg)
        log.info("[转弯] 初始转 %d×%.0f° ≈ %+.1f°，开始闭环精修",
                 CORNER_INITIAL_TURN_STEPS, big_deg, cumulative_angle)

        for step in range(CORNER_MAX_TURN_STEPS):
            self._head_lr(0)
            frame = self.capture_frame()
            if frame is None:
                log.info("[转弯] 拍照失败，跳过")
                continue
            result = self.detector.detect(frame)

            if result is None or not result.exists:
                log.info("[转弯] 闭环 %d: 丢线，前进一步找线", step + 1)
                self._run_action("go_forward_one_step")
                continue

            p = result.primary
            lx = max(-LOOKAHEAD_CLAMP, min(LOOKAHEAD_CLAMP, p.lookahead_x))
            log.info("[转弯] 闭环 %d: orientation=%s lookahead_x=%.1f heading=%.1f 累计=%+.1f°",
                     step + 1, p.orientation, lx, p.heading_deg, cumulative_angle)

            if p.orientation == "follow":
                h = p.heading_deg
                if abs(h) <= CORNER_EXIT_HEADING:
                    log.info("[转弯] 弯已转过（heading=%.1f° ≤ %.1f°），横向误差交给对齐阶段）",
                             h, CORNER_EXIT_HEADING)
                    return True
                if turn_sign * h < 0:
                    log.info("[转弯] 未转够（heading=%+.1f° 偏%s侧），继续转", h, direction)
                    cumulative_angle += self._turn_by_angle(turn_sign * big_deg)
                else:
                    log.info("[转弯] 转过（heading=%+.1f° 偏反侧），反向小步回正", h)
                    cumulative_angle += self._turn_by_angle(-turn_sign * small_deg)
            elif p.orientation in ("corner", "cross"):
                log.info("[转弯] 仍在拐角内（%s），继续转", p.orientation)
                cumulative_angle += self._turn_by_angle(turn_sign * big_deg)
            else:
                continue

            if abs(cumulative_angle) >= CORNER_MAX_CUMULATIVE_DEG:
                log.info("[转弯] 累计转角超安全上限，返回异常")
                return False

        log.info("[转弯] 闭环超过 %d 次，返回异常", CORNER_MAX_TURN_STEPS)
        return False

    def _handle_line_lost(self):
        """循迹丢线处理：后退找线，超限则回寻线。返回 "track"（找回）或 "seek"（需重寻）"""
        for back_num in range(TRACK_MAX_BACKS):
            log.info("[丢线] 后退 %d/%d 次找线", back_num + 1, TRACK_MAX_BACKS)
            self._run_action("back_one_step")
            self._head_lr(0)
            frame = self.capture_frame()
            if frame is None:
                continue
            result = self.detector.detect(frame)
            if result.exists:
                log.info("[丢线] 后退后找回线：orientation=%s", result.primary.orientation)
                return "track"

        log.info("[丢线] 后退用尽仍丢线，回寻线阶段")
        return "seek"

    def track_line(self):
        """循迹主函数：沿线行进，处理直道/直角弯/丢线，far_ry 判定终点。

        返回 True 表示正常结束（到达线末端或达到步数上限），False 表示异常。
        """
        log.info("=" * 50)
        log.info("===== 开始循迹 =====")
        log.info("=" * 50)

        total_steps = 0
        stuck_counter = 0  # 连续 cross/丢线 计数（防卡死）

        while total_steps < self.max_total_steps:
            total_steps += 1

            self._head_lr(0)
            frame = self.capture_frame()
            if frame is None:
                log.info("[循迹] step=%d 取帧失败，跳过", total_steps)
                continue

            # 每 0.5s 存一帧 remap 快照，配合 log 排查
            now = time.time()
            if now - self.last_debug_save >= 0.5:
                self.last_debug_save = now
                snap_path = "/home/pi/codes/pictures/line_track_%s.jpg" % time.strftime("%H%M%S")
                os.makedirs("/home/pi/codes/pictures", exist_ok=True)
                cv2.imwrite(snap_path, frame)

            result = self.detector.detect(frame)

            if not result.exists:
                stuck_counter += 1
                if stuck_counter > STUCK_MAX:
                    log.info("[循迹] 卡住 %d 步（丢线+横线），回寻线", stuck_counter)
                    if self.seek_line():
                        stuck_counter = 0
                        continue
                    log.info("[循迹] 重新寻线失败，循迹结束")
                    return False
                log.info("[循迹] step=%d 丢线", total_steps)
                status = self._handle_line_lost()
                if status == "seek":
                    if self.seek_line():
                        stuck_counter = 0
                        continue
                    log.info("[循迹] 重新寻线失败，循迹结束")
                    return False
                continue

            p = result.primary
            lx_clamped = max(-LOOKAHEAD_CLAMP, min(LOOKAHEAD_CLAMP, p.lookahead_x))
            log.info("[循迹] step=%d orientation=%s lookahead_x=%.1f lateral_offset=%.1f "
                     "heading=%.1f far_ry=%.0f curvature=%.4f",
                     total_steps, p.orientation, lx_clamped, p.lateral_offset,
                     p.heading_deg, p.far_ry, p.curvature)

            if p.orientation == "corner":
                stuck_counter = 0
                if self._handle_corner(p.curvature):
                    log.info("[循迹] 转弯完成，重新对准")
                    if not self._align_to_line(context="corner"):
                        log.info("[循迹] 转弯后对准失败，回寻线")
                        if not self.seek_line():
                            log.info("[循迹] 重新寻线失败，循迹结束")
                            return False
                else:
                    log.info("[循迹] 转弯异常，尝试继续循迹")
                continue

            if p.orientation == "cross":
                stuck_counter += 1
                if stuck_counter > STUCK_MAX:
                    log.info("[循迹] 卡住 %d 步（丢线+横线），回寻线", stuck_counter)
                    if self.seek_line():
                        stuck_counter = 0
                        continue
                    log.info("[循迹] 重新寻线失败，循迹结束")
                    return False
                log.info("[循迹] 横线（卡住计数 %d），跨越", stuck_counter)
                self._run_action("go_forward_one_step")
                continue

            # follow 线
            stuck_counter = 0
            # 终点判定：线远端 ry 显著小于视野高度说明线末端已进入视野
            if 0 < p.far_ry < LINE_END_FAR_RY:
                log.info("[循迹] 线末端接近（far_ry=%.0f < %.0f），终点前进",
                         p.far_ry, LINE_END_FAR_RY)
                forward_steps = 0
                for _ in range(LINE_END_FORWARD_MAX):
                    self._run_action("go_forward_one_step")
                    forward_steps += 1
                    self._head_lr(0)
                    f2 = self.capture_frame()
                    r2 = self.detector.detect(f2) if f2 is not None else None
                    if r2 is None or not r2.exists:
                        break
                for _ in range(LINE_END_BLIND_MAX):
                    self._run_action("go_forward_one_step")
                    forward_steps += 1
                log.info("[循迹] 到达线末端（终点前进 %d 步），循迹结束", forward_steps)
                return True

            lx = lx_clamped
            if FOLLOW_HEADING_MIN < abs(p.heading_deg) <= FOLLOW_HEADING_MAX:
                log.info("[循迹] heading=%.1f° 偏，闭环修朝向", p.heading_deg)
                self._correct_heading(p.heading_deg)
                continue
            if abs(p.heading_deg) > FOLLOW_HEADING_MAX:
                log.info("[循迹] heading=%.1f° 过大（疑拐角/抖动），忽略朝向，按横向处理",
                         p.heading_deg)
            if abs(lx) > T_BIG:
                if lx > 0:
                    log.info("[循迹] lookahead_x=%.1f > %.1f，右转修朝向", lx, T_BIG)
                    self._run_action("turn_right")
                else:
                    log.info("[循迹] lookahead_x=%.1f < -%.1f，左转修朝向", lx, T_BIG)
                    self._run_action("turn_left")
            elif abs(lx) > T_CENTER:
                if lx > 0:
                    log.info("[循迹] lookahead_x=%.1f，右移对齐", lx)
                    self._run_action("right_move")
                else:
                    log.info("[循迹] lookahead_x=%.1f，左移对齐", lx)
                    self._run_action("left_move")
            else:
                log.info("[循迹] 居中，前进（lookahead_x=%.1f）", lx)
                self._run_action("go_forward_one_step")

        log.info("[循迹] 达到最大步数 %d（固定步数兜底），循迹结束", self.max_total_steps)
        return True

    def run(self):
        camera = None
        try:
            camera = self.open_camera()
            self.camera = camera
            for _ in range(30):
                ok, frame = camera.read()
                if ok and frame is not None:
                    break
                time.sleep(0.05)

            self.init_robot()
            # init_robot 杀了tonypi服务会重置USB相机, 重新打开并预热
            if camera is not None:
                camera.release()
            camera = self.open_camera()
            self.camera = camera
            for _ in range(30):
                ok, frame = camera.read()
                if ok and frame is not None:
                    break
                time.sleep(0.05)

            log.info(
                "line-follow debug start dry_run=%s back_steps=%d max_total_steps=%d",
                self.args.dry_run,
                self.args.back_steps_after_place,
                self.max_total_steps,
            )

            # 后退 (假定方块已放下、手臂已复位)
            self.actions.run("back_fast", times=self.args.back_steps_after_place, with_stand=True)
            self.actions.run("stand")

            # 寻线 + 循迹
            if not self.seek_line():
                log.warning("寻线失败，循迹终止")
            else:
                self.track_line()

        except BaseException as exc:
            log.exception("fatal error: %s", exc)
        finally:
            self.running = False
            if camera is not None:
                camera.release()
            cv2.destroyAllWindows()
            if self.robot_initialized and not self.args.dry_run:
                AGC.stopActionGroup()
                self.actions.run("stand")
            if self.services_stopped:
                subprocess.run(["sudo", "systemctl", "start", "tonypi", "joystick"], stderr=subprocess.DEVNULL)
            log.info("line-follow debug exit")


def parse_args():
    parser = argparse.ArgumentParser(description="放下蓝色方块后: 后退→找红线→循迹 独立调试")
    parser.add_argument("--dry-run", action="store_true", help="log actions without executing motion")
    parser.add_argument("--no-kill", action="store_true", help="do not kill existing TonyPi/Joystick processes")
    parser.add_argument("--back-steps-after-place", type=int, default=12, help="放置后后退步数(退远一点才够转身找到身后红线)")
    parser.add_argument("--max-total-steps", type=int, default=200, help="循迹总步数安全上限(固定步数兜底, 防死循环)")
    parser.add_argument("--line-head-delta", type=int, default=60, help="巡线时低头角度(相对servo1, 60=看远)")
    args = parser.parse_args()
    if min(args.back_steps_after_place, args.max_total_steps) < 0:
        parser.error("movement counts must be nonnegative")
    return args


def main():
    args = parse_args()
    task = LineFollowDebug(args)
    task.run()


if __name__ == "__main__":
    main()
