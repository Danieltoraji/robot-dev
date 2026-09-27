#!/usr/bin/env python3
# coding=utf-8
"""Pick one tagless blue sponge and place it on the board marked Tag 38."""

import argparse
import glob
import json
import logging
import math
import os
import subprocess
import sys
import time
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, "/home/pi/TonyPi")
sys.path.insert(0, "/home/pi/TonyPi/Functions")
sys.path.insert(0, "/home/pi/TonyPi/HiwonderSDK")

import hiwonder.ActionGroupControl as AGC
import hiwonder.Misc as Misc
import hiwonder.apriltag as apriltag
import hiwonder.ros_robot_controller_sdk as rrc
import hiwonder.yaml_handle as yaml_handle
from hiwonder.Controller import Controller
from CameraCalibration.CalibrationConfig import calibration_param_path


LOG_FILE = "/home/pi/Robot_Competition/levels/apriltag_sorting_task.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler(LOG_FILE, mode="a"), logging.StreamHandler()],
)
log = logging.getLogger("sorting_task")


# ---- Tunable task parameters -------------------------------------------------

TARGET_COLOR = "blue"
# 标定好的蓝色 LAB 阈值(由 test_programs_NoUseInMain/blue_cube_lab_calibration.py 标定)
# 直接写死在代码里, 不再依赖 lab_config.yaml 里的 blue 项
BLUE_LAB_MIN = [0, 0, 0]
BLUE_LAB_MAX = [153, 123, 125]
DEFAULT_TARGET_TAG_ID = 38
FRAME_SIZE = (320, 240)
CENTER_X = 350

# Color approach thresholds, derived from existing /home/pi/codes examples.
COLOR_FAR_Y = 250   # go_forward到cy<=250(跟官方Transport一致)
COLOR_NEAR_Y = 340
COLOR_TOO_NEAR_Y = 390
COLOR_X_TURN = 80
COLOR_X_LARGE = 40
COLOR_X_FINE = 20
COLOR_AREA_MIN = 500

# Tag approach thresholds. These need final field tuning.
BOARD_DEPTH_M = 0.08
TAG_PLACE_NEAR_M = 0.17 + BOARD_DEPTH_M
TAG_PLACE_FAR_M = 0.25 + BOARD_DEPTH_M
TAG_PLACE_BIG_M = 0.60  # 大步靠近切换阈值(>0.6m用go_forward大步, 否则小步)
TAG_FAR_Y = 235
TAG_NEAR_Y = 335
TAG_TOO_NEAR_Y = 390
TAG_X_TURN = 90
TAG_X_LARGE = 45
TAG_X_FINE = 22

# 终点tag穿过参数(align_test同步)
END_YAW_LOWER = -15    # yaw下限
END_YAW_UPPER = 5      # yaw上限
END_FORWARD_TOL = 0.07  # 纵深修正阈值(米)
WALK_STEPS = 10        # 终点直行步数
MAX_TURN = 30          # 终点找tag最大小步转次数

# Number of final small forward steps before grasping / placing.
PICK_FINAL_STEPS = 2
STEP4_MAX_FORWARDS = 10  # step4凑近最多连续前进步数, 到顶直接判定到位(防止推着海绵无限前进)
MAX_PICK_RETRIES = 3  # 抓取验证失败最大重试次数
PLACE_FINAL_STEPS = 1

# Head scanning.
HEAD_STEP_X = 15
HEAD_STEP_Y = 15
HEAD_SCAN_LEFT = 300    # 头扫左极限(相对servo2脉冲, 原200)
HEAD_SCAN_RIGHT = 500   # 头扫右极限(相对servo2脉冲, 原400)
INIT_HEAD_PITCH = 1100  # 初始化时头部俯仰中位(低头的中间参数, 原为yaml servo1)

# 夹爪闭合程度: 1.0=最紧(8=0,16=1000), 0.5=半开, 0=全开(500)
GRIP_SCALE = 1.0
LOCK_SERVOS = {"6": 700, "7": 820, "8": int(500 - 500*GRIP_SCALE), "14": 300, "15": 180, "16": int(500 + 500*GRIP_SCALE)}

# 摄像头分辨率(640x480)
CAM_SIZE = (640, 480)

# ---- 红色胶带巡线(参考 TonyPi/Functions/VisualPatrol.py) ----
LINE_CENTER_X = 340            # 巡线画面中心(640 宽)
LINE_TURN_THRESHOLD = 59       # |dx|>59 才动作(借鉴RedLinePatrol)
SEARCH_LINE_ALIGN_THRESHOLD = 40  # search_line对正阈值(看到红线后转到|dx|<40才开始巡线)
LINE_LOST_TIMEOUT = 2.0        # 相机无帧超时(秒)
LINE_LOST_HOLD = 1.5           # 丢线后保持方向前进的秒数(不立即转)
MAX_LOST_TURNS = 10            # 保持超时后左大转找的最大次数
MAX_CONSEC_MOVES = 2           # 同方向连续平移上限, 超过强制前进一步重判(打断卡尔曼滞后过冲链条)
FOLLOW_CONFIRM_S = 2.0         # 巡线确认等待秒(静置后判断线是否竖直, 防转身后画面甩动误判)
FOLLOW_MAX_ADJUST = 30         # 未竖直时的调整次数上限, 超限直接直行
VERTICAL_ANGLE_TOL = 25       # 红线主轴与竖直方向夹角<=此值(度)视为线竖直, 用平移不用转身(方向判据, 与线宽/距离无关)
LINE_ROI = [                   # 三段 ROI (y1, y2, x1, x2, 权重), 越靠脚边权重越大
    # 调整: 去掉最下段(440-480, 脚下区域基本看不到线), 在中间段上方补一段(300-340)
    (240, 280, 0, 640, 0.1),
    (300, 340, 0, 640, 0.3),
    (340, 380, 0, 640, 0.6),
]
# HSV 红色双段阈值(红色横跨0°, 两段合并; S/V排除黑白线)
RED_H_LOW1, RED_H_HIGH1 = 0, 10
RED_H_LOW2, RED_H_HIGH2 = 160, 180
RED_S_LOW, RED_S_HIGH = 80, 255
RED_V_LOW, RED_V_HIGH = 80, 255

# ---- 卡尔曼滤波参数(可被 calib_config.json 覆盖) ----
KF_CX_Q, KF_CX_R = 1.0, 4.0
KF_CY_Q, KF_CY_R = 1.0, 4.0
KF_DIST_Q, KF_DIST_R = 0.002, 0.01
KF_OFFSET_Q, KF_OFFSET_R = 0.002, 0.01
KF_ANGLE_Q, KF_ANGLE_QV, KF_ANGLE_R = 5.0, 1.0, 10.0
KF_AREA_Q, KF_AREA_R = 100.0, 500.0
KF_LINE_Q, KF_LINE_R = 30.0, 10.0
# 巡线中心滤波: 巡线循环每执行一个动作(~1.2s)才更新一次滤波, Q=1时增益仅~0.24、
# 稳态滞后~60px, 机器人按滞后值决策导致平移过冲之字形; Q=30增益~0.63, 滞后降到~12px

# ---- 标定配置加载(calib_config.json 由 tools/calib_tuner.py 生成) ----
# 白名单: 允许被配置文件覆盖的模块常量
CALIB_CONST_KEYS = [
    "BLUE_LAB_MIN", "BLUE_LAB_MAX",
    "RED_H_LOW1", "RED_H_HIGH1", "RED_H_LOW2", "RED_H_HIGH2",
    "RED_S_LOW", "RED_S_HIGH", "RED_V_LOW", "RED_V_HIGH",
    "COLOR_AREA_MIN", "CENTER_X",
    "HEAD_SCAN_LEFT", "HEAD_SCAN_RIGHT",
    "COLOR_FAR_Y", "COLOR_NEAR_Y", "COLOR_TOO_NEAR_Y",
    "COLOR_X_TURN", "COLOR_X_LARGE", "COLOR_X_FINE",
    "BOARD_DEPTH_M", "TAG_PLACE_NEAR_M", "TAG_PLACE_FAR_M", "TAG_PLACE_BIG_M",
    "TAG_FAR_Y", "TAG_NEAR_Y", "TAG_TOO_NEAR_Y",
    "TAG_X_TURN", "TAG_X_LARGE", "TAG_X_FINE",
    "END_YAW_LOWER", "END_YAW_UPPER", "WALK_STEPS", "MAX_TURN",
    "PICK_FINAL_STEPS", "PLACE_FINAL_STEPS", "MAX_PICK_RETRIES", "STEP4_MAX_FORWARDS",
    "LINE_CENTER_X", "LINE_TURN_THRESHOLD", "SEARCH_LINE_ALIGN_THRESHOLD",
    "VERTICAL_ANGLE_TOL", "LINE_LOST_TIMEOUT", "LINE_LOST_HOLD", "MAX_LOST_TURNS",
    "MAX_CONSEC_MOVES", "FOLLOW_CONFIRM_S", "FOLLOW_MAX_ADJUST",
    "KF_CX_Q", "KF_CX_R", "KF_CY_Q", "KF_CY_R",
    "KF_DIST_Q", "KF_DIST_R", "KF_OFFSET_Q", "KF_OFFSET_R",
    "KF_ANGLE_Q", "KF_ANGLE_QV", "KF_ANGLE_R",
    "KF_AREA_Q", "KF_AREA_R", "KF_LINE_Q", "KF_LINE_R",
]
# 允许被配置文件覆盖的命令行参数(dest名), 仅当命令行未显式指定时生效
CALIB_ARG_KEYS = [
    "pick_area_threshold", "pick_y_threshold", "pick_too_near_y",
    "post_pick_left_turns", "post_pick_forward_steps", "back_steps_after_place",
    "line_head_delta", "line_final_steps", "line_search_turns",
]
_ARG_OPTION_STRINGS = {
    "pick_area_threshold": "--pick-area-threshold",
    "pick_y_threshold": "--pick-y-threshold",
    "pick_too_near_y": "--pick-too-near-y",
    "post_pick_left_turns": "--post-pick-left-turns",
    "post_pick_forward_steps": "--post-pick-forward-steps",
    "back_steps_after_place": "--back-steps-after-place",
    "line_head_delta": "--line-head-delta",
    "line_final_steps": "--line-final-steps",
    "line_search_turns": "--line-search-turns",
}


def load_calib_config():
    """加载脚本同目录的 calib_config.json; 不存在或损坏时返回 None(行为与无配置完全一致)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib_config.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (ValueError, OSError) as exc:
        log.warning("calib_config.json 解析失败, 已忽略: %s", exc)
        return None


CALIB_CONFIG = load_calib_config()
if CALIB_CONFIG:
    _applied = []
    for _key in CALIB_CONST_KEYS:
        if _key in CALIB_CONFIG:
            globals()[_key] = CALIB_CONFIG[_key]
            _applied.append(_key)
    log.info("calib_config.json 覆盖 %d 个常量: %s", len(_applied), _applied)


@dataclass
class ColorTarget:
    cx: int = -1
    cy: int = -1
    angle: float = 0.0
    area: float = 0.0


class Kalman1D:
    """一维卡尔曼滤波(带速度): 用于angle, 状态=[值, 速度]"""
    def __init__(self, x0=0.0, Q_angle=5.0, Q_v=1.0, R=4.0):
        self.x = float(x0)   # 估计值
        self.v = 0.0          # 速度估计
        self.Q_angle = Q_angle  # 角度过程噪声(大, 能跟上突变)
        self.Q_v = Q_v          # 速度过程噪声
        self.R = R              # 观测噪声
        self.P = np.array([[100.0, 0.0], [0.0, 100.0]])  # 协方差矩阵

    def update(self, z):
        """输入测量值z, 返回平滑后的估计值"""
        self.x = self.x + self.v
        Q_mat = np.array([[self.Q_angle, 0.0], [0.0, self.Q_v]])
        self.P = self.P + Q_mat
        K = self.P[:, 0] / (self.P[0, 0] + self.R)
        innov = float(z) - self.x
        self.x = self.x + K[0] * innov
        self.v = self.v + K[1] * innov
        self.P = (np.eye(2) - np.outer(K, [1.0, 0.0])) @ self.P
        return self.x


class Kalman1DConst:
    """一维卡尔曼滤波(常数模型): 用于cx/cy/dist/offset/area/line_cx"""
    def __init__(self, x0=0.0, Q=1.0, R=4.0):
        self.x = float(x0)
        self.Q = Q
        self.R = R
        self.P = 1.0
    def update(self, z):
        self.P = self.P + self.Q
        K = self.P / (self.P + self.R)
        self.x = self.x + K * (float(z) - self.x)
        self.P = (1 - K) * self.P
        return self.x


@dataclass
class TagTarget:
    tag_id: int
    cx: int
    cy: int
    angle: float
    area: float
    distance: float = 0.0   # 3D直线距离(米)
    offset: float = 0.0     # 水平左右偏移(米), 正=机器人在tag右
    forward: float = 0.0    # 纵深距离(米), 正=机器人在tag前
    yaw: float = 0.0       # 机器人朝向偏离tag法线的角度(度), 正对=0


# AprilTag 5cm×5cm 实际角点(单位米, 半边2.5cm), 用于solvePnP
# 检测器corners顺序: [左上, 右上, 右下, 左下] (顺时针)
TAG_OBJECT_POINTS = np.array([
    [-0.025,  0.025, 0],  # 左上
    [ 0.025,  0.025, 0],  # 右上
    [ 0.025, -0.025, 0],  # 右下
    [-0.025, -0.025, 0],  # 左下
], dtype=np.float64)


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


class SortingTask:
    def __init__(self, args):
        self.args = args
        self.actions = ActionRunner(dry_run=args.dry_run)

        self.lab_data = yaml_handle.get_yaml_data(yaml_handle.lab_file_path)
        self.servo_data = yaml_handle.get_yaml_data(yaml_handle.servo_file_path)
        self.servo1 = int(self.servo_data["servo1"])
        self.servo2 = int(self.servo_data["servo2"])
        self.x_dis = self.servo2
        self.y_dis = self.servo1
        self.head_mode = "left_right"
        self.head_turn = "left_right"  # 官方scan用
        self.d_x = HEAD_STEP_X
        self.d_y = HEAD_STEP_Y
        self.last_search_turn = 0.0

        self.target_tag_id = int(args.target_tag)
        self.line_head_delta = int(args.line_head_delta)
        self.state = "SEARCH_OBJECT"
        self.last_state = None
        self.state_enter_time = time.time()
        self.start_time = time.time()
        self.kf_cx = Kalman1DConst(0.0, Q=KF_CX_Q, R=KF_CX_R)  # cx 卡尔曼(R大=更信预测=更平滑)
        self.kf_cy = Kalman1DConst(0.0, Q=KF_CY_Q, R=KF_CY_R)  # cy 卡尔曼
        self.kf_dist = Kalman1DConst(0.0, Q=KF_DIST_Q, R=KF_DIST_R)  # PnP distance 卡尔曼(米)
        self.kf_offset = Kalman1DConst(0.0, Q=KF_OFFSET_Q, R=KF_OFFSET_R)  # PnP offset 卡尔曼(米)
        self.kf_angle = Kalman1D(0.0, Q_angle=KF_ANGLE_Q, Q_v=KF_ANGLE_QV, R=KF_ANGLE_R)  # 角度卡尔曼(带速度模型)
        self.kf_area = Kalman1DConst(0.0, Q=KF_AREA_Q, R=KF_AREA_R)  # area 卡尔曼(像素)
        self.kf_line_cx = Kalman1DConst(LINE_CENTER_X, Q=KF_LINE_Q, R=KF_LINE_R)  # 巡线红线中心x卡尔曼(初始画面中心,防起步假偏移)
        self.line_is_vertical = False  # 巡线平行判定(线竖直时用平移)
        self.color_step = 1  # approach_color step状态机(参考官方Transport)
        self.step4_forward_steps = 0  # step4凑近连续前进步数(兜底计数)
        self.pick_retries = 0  # 抓取失败重试计数
        # 双线程共享(检测线程写, 动作线程读)
        import threading
        self._lock = threading.Lock()
        self.color_target = ColorTarget()
        self.tags = []
        self.display_frame = None
        self.running = False
        self.services_stopped = False
        self.robot_initialized = False
        self.debug_gui = bool(args.debug and os.environ.get("DISPLAY"))
        self.last_debug_save = 0.0
        self.debug_save_path = "/home/pi/codes/images/sorting_task_debug.jpg"

        self.detector = apriltag.Detector(searchpath=["/usr/local/lib"])

        param_data = np.load(calibration_param_path + ".npz")
        mtx = param_data["mtx_array"]
        dist = param_data["dist_array"]
        newcameramtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, CAM_SIZE, 0, CAM_SIZE)
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, CAM_SIZE, 5)
        # PnP用: 图像已remap去畸变, 所以用newcameramtx + 零畸变系数
        self.cam_mtx = newcameramtx
        self.cam_dist = np.zeros((5, 1), dtype=np.float64)

        # Reuse AGC's board/controller. Creating extra board objects for motion
        # can compete for serial responses.
        self.board = AGC.board
        self.ctl = AGC.ctl

    def enter_state(self, state: str):
        if state == self.state:
            return
        log.info("state %s -> %s", self.state, state)
        self.state = state
        self.state_enter_time = time.time()

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
        self.set_head(y=INIT_HEAD_PITCH, duration=300)  # 初始化低头中位 1100
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

    def scan_head_or_turn(self, lock: bool = False, turn_direction: str = "right"):
        """官方Transport搜索逻辑(照搬): 头扫+到极限换向+扫到底转身"""
        if not hasattr(self, "_scan_start"):
            self._scan_start = True
            self._scan_time = 0.0
        if self._scan_start:
            self._scan_start = False
            self._scan_time = time.time()
        else:
            if time.time() - self._scan_time > 0.5:
                if 0 < self.servo2 - self.x_dis <= abs(self.d_x) and self.d_y > 0:
                    # 头扫到底了(回到中位附近), 转身
                    self.x_dis = self.servo2
                    self.y_dis = self.servo1
                    self.set_head(self.x_dis, self.y_dis, 20)
                    turn_action = "turn_right"  # 大步右转
                    log.info("scan: 扫完右转2大步")
                    self.actions.run(turn_action, times=2, lock=lock)
                elif self.head_turn == "left_right":
                    self.x_dis += self.d_x
                    if self.x_dis > self.servo2 + HEAD_SCAN_RIGHT or self.x_dis < self.servo2 - HEAD_SCAN_LEFT:
                        self.head_turn = "up_down"
                        self.d_x = -self.d_x
                elif self.head_turn == "up_down":
                    self.y_dis += self.d_y
                    if self.y_dis > self.servo1 + 300 or self.y_dis < self.servo1:
                        self.head_turn = "left_right"
                        self.d_y = -self.d_y
                self.set_head(self.x_dis, self.y_dis, 20)
                time.sleep(0.02)

    def detect_blue(self, img) -> Tuple[ColorTarget, np.ndarray]:
        display = img.copy()

        img_h, img_w = img.shape[:2]
        frame_resize = cv2.resize(img, FRAME_SIZE, interpolation=cv2.INTER_NEAREST)
        frame_lab = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2LAB)
        frame_gb = cv2.GaussianBlur(frame_lab, (3, 3), 3)
        mask = cv2.inRange(
            frame_gb,
            tuple(BLUE_LAB_MIN),
            tuple(BLUE_LAB_MAX),
        )
        eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]

        best_contour = None
        best_area = 0.0
        for contour in contours:
            area = math.fabs(cv2.contourArea(contour))
            if area > best_area:
                best_area = area
                best_contour = contour

        if best_contour is None or best_area < COLOR_AREA_MIN:
            return ColorTarget(), display

        rect = cv2.minAreaRect(best_contour)
        # 不做立体检测(方块表面均匀会误拒), 直接用颜色中心点定位
        box = cv2.boxPoints(rect).astype(np.intp)
        for i in range(4):
            box[i, 0] = int(Misc.map(box[i, 0], 0, FRAME_SIZE[0], 0, img_w))
            box[i, 1] = int(Misc.map(box[i, 1], 0, FRAME_SIZE[1], 0, img_h))
        cv2.drawContours(display, [box], -1, (0, 255, 255), 2)
        cx = int((box[0, 0] + box[2, 0]) / 2)
        cy = int((box[0, 1] + box[2, 1]) / 2)
        # 卡尔曼滤波平滑cx/cy, 减少检测抖动
        cx = int(self.kf_cx.update(cx))
        cy = int(self.kf_cy.update(cy))
        cv2.circle(display, (cx, cy), 5, (0, 255, 255), -1)
        angle_f = float(self.kf_angle.update(float(rect[2])))
        area_f = float(self.kf_area.update(float(best_area)))
        return ColorTarget(cx, cy, angle_f, area_f), display

    def detect_tags(self, img) -> Tuple[List[TagTarget], np.ndarray]:
        display = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(gray, return_image=False)
        tags: List[TagTarget] = []
        for detection in detections:
            if str(detection.tag_family, encoding="utf-8") != "tag36h11":
                continue
            corners = np.rint(detection.corners).astype(np.intp)
            cv2.drawContours(display, [corners], -1, (0, 255, 255), 2)
            cx, cy = int(detection.center[0]), int(detection.center[1])
            angle = float(math.degrees(math.atan2(corners[0][1] - corners[1][1], corners[0][0] - corners[1][0])))
            area = float(abs(cv2.contourArea(corners)))
            # IPPE平面4点精确解, 返回两个解
            retval, rvecs, tvecs, reproj_errors = cv2.solvePnPGeneric(
                TAG_OBJECT_POINTS, detection.corners.astype(np.float64),
                self.cam_mtx, self.cam_dist, flags=cv2.SOLVEPNP_IPPE)
            if retval:
                # 选pitch绝对值最小的解(tag竖直贴, 平视时pitch≈0)
                best_idx = 0
                min_pitch = 999
                for i, (rv, tv) in enumerate(zip(rvecs, tvecs)):
                    R2, _ = cv2.Rodrigues(rv)
                    pitch2 = math.degrees(math.asin(max(-1, min(1, -float(R2[2,0])))))
                    if abs(pitch2) < min_pitch:
                        min_pitch = abs(pitch2)
                        best_idx = i
                rvec = rvecs[best_idx]
                tvec = tvecs[best_idx]
                R, _ = cv2.Rodrigues(rvec)
                cam_in_tag = (-R.T @ tvec).ravel()  # (3,1)->(3,), numpy2下元素才是标量(math.sqrt/float不接受1元素数组)
                distance = float(math.sqrt(cam_in_tag[0]**2 + cam_in_tag[1]**2 + cam_in_tag[2]**2))
                offset = float(-cam_in_tag[0])
                forward = float(abs(cam_in_tag[2]))
                fwd_in_tag = R.T @ np.array([0.0, 0.0, 1.0])
                yaw = float(math.degrees(math.atan2(fwd_in_tag[0], fwd_in_tag[2])))
                if yaw > 90: yaw -= 180
                elif yaw < -90: yaw += 180
                # 卡尔曼平滑
                if distance > 0:
                    distance = float(self.kf_dist.update(distance))
                    offset = float(self.kf_offset.update(offset))
                    forward = float(self.kf_dist.update(forward))
                    yaw = float(self.kf_angle.update(yaw))
            else:
                distance = 0.0
                offset = 0.0
                forward = 0.0
                yaw = 0.0
            tag = TagTarget(int(detection.tag_id), cx, cy, angle, area, distance, offset, forward, yaw)
            tags.append(tag)
            cv2.circle(display, (cx, cy), 5, (0, 255, 255), -1)
            cv2.putText(display, f"id:{tag.tag_id} d={distance:.2f} o={offset:.2f}", (cx + 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        return tags, display

    def choose_target_area_tag(self, tags: List[TagTarget]) -> Optional[TagTarget]:
        candidates = [tag for tag in tags if tag.tag_id == self.target_tag_id]
        if not candidates:
            return None
        return max(candidates, key=lambda tag: tag.area)

    def approach_color(self, target: ColorTarget) -> bool:
        if target.cx < 0:
            self.color_step = 1
            self.scan_head_or_turn(lock=False)
            return False

        if self.x_dis != self.servo2:
            if self.x_dis > self.servo2:
                self.actions.run("turn_left_small_step")
            else:
                self.actions.run("turn_right_small_step")
            self.set_head_center(duration=300)
            return False

        dx = target.cx - CENTER_X
        log.info("sponge target cx=%d cy=%d area=%.1f dx=%d step=%d", target.cx, target.cy, target.area, dx, self.color_step)

        # 参考官方Transport step状态机: 1对正转身 2接近 3精调 4凑近
        if self.color_step == 1:  # 左右调整对中(转身)
            if abs(dx) > 170 and target.cy > COLOR_NEAR_Y:
                self.actions.run("back_fast")
            elif abs(dx) > COLOR_X_TURN:
                # 远区大转弯, 近区小转弯
                if target.cy <= COLOR_FAR_Y:
                    self.actions.run("turn_right" if dx > 0 else "turn_left")
                else:
                    self.actions.run("turn_right_small_step" if dx > 0 else "turn_left_small_step")
            elif target.cy <= COLOR_FAR_Y:
                self.actions.run("go_forward")  # 远区大步靠近
            else:
                self.color_step = 2
        elif self.color_step == 2:  # 接近并横向对正; 海绵不使用轮廓角度
            if target.cy > COLOR_NEAR_Y:
                self.actions.run("back_fast")
            elif abs(dx) > 150:
                self.actions.run("right_move_30" if dx > 0 else "left_move_30")
            elif abs(dx) > COLOR_X_LARGE:
                self.actions.run("right_move_30" if dx > 0 else "left_move_30")
            else:
                self.color_step = 3
        elif self.color_step == 3:  # 精细调整(小平移)
            if target.cy > COLOR_TOO_NEAR_Y:
                self.actions.run("back_fast")
            elif target.cy <= COLOR_FAR_Y:
                self.actions.run("go_forward")  # 远区大步靠近
            elif abs(dx) > COLOR_X_LARGE:
                self.actions.run("right_move_30" if dx > 0 else "left_move_30")
            elif abs(dx) > COLOR_X_FINE:
                self.actions.run("right_move" if dx > 0 else "left_move")
            else:
                self.color_step = 4
        elif self.color_step == 4:  # 凑近(小步前进)
            if target.cy > COLOR_TOO_NEAR_Y:
                self.step4_forward_steps = 0
                self.actions.run("back_fast")
            elif target.area >= self.args.pick_area_threshold:
                # 面积已足够大(足够近), 不再追着海绵走, 直接进入抓取判定
                log.info("sponge aligned for final pickup approach (area=%.0f)", target.area)
                self.step4_forward_steps = 0
                self.color_step = 1  # reset for next
                return True
            elif target.cy < COLOR_NEAR_Y:
                # 兜底: 连续前进到步数上限还没满足面积, 说明已在推着海绵走, 直接判定到位
                self.actions.run("go_forward_one_step")
                self.step4_forward_steps += 1
                log.info("sponge step4 forward %d/%d (area=%.0f cy=%d)",
                         self.step4_forward_steps, STEP4_MAX_FORWARDS, target.area, target.cy)
                if self.step4_forward_steps >= STEP4_MAX_FORWARDS:
                    log.info("sponge aligned for pickup (step4 forward limit %d reached)", STEP4_MAX_FORWARDS)
                    self.step4_forward_steps = 0
                    self.color_step = 1  # reset for next
                    return True
            elif abs(dx) > COLOR_X_FINE:
                self.step4_forward_steps = 0
                self.color_step = 3  # 偏了回step3调整
            else:
                log.info("sponge aligned for final pickup approach")
                self.step4_forward_steps = 0
                self.color_step = 1  # reset for next
                return True
        return False

    def creep_to_pick_distance(self):
        """Approach the tagless sponge using only color position and area."""
        for _ in range(30):
            with self._lock:
                ct = self.color_target
            if ct.cx < 0:
                time.sleep(0.1)
                continue

            dx = ct.cx - CENTER_X
            log.info("final sponge approach area=%.0f cx=%d cy=%d dx=%d", ct.area, ct.cx, ct.cy, dx)
            if ct.cy >= self.args.pick_too_near_y:
                log.info("sponge too near (cy=%d), back", ct.cy)
                self.actions.run("back_fast")
                return False
            if ct.area >= self.args.pick_area_threshold or ct.cy >= self.args.pick_y_threshold:
                log.info("at sponge pickup distance (area=%.0f cy=%d)", ct.area, ct.cy)
                return True
            if abs(dx) > COLOR_X_LARGE:
                self.actions.run("right_move_30" if dx > 0 else "left_move_30")
            elif abs(dx) > COLOR_X_FINE:
                self.actions.run("right_move" if dx > 0 else "left_move")
            else:
                self.actions.run("go_forward_one_step")
        return False

    def pick_object(self):
        log.info("pickup tagless sponge with existing action '%s'", self.args.pick_action)
        self.set_head_center(duration=300)
        self.actions.run("go_forward_one_step", times=PICK_FINAL_STEPS)
        self.actions.run("stand")
        self.actions.run(self.args.pick_action)
        # 夹爪按GRIP_SCALE闭合(覆盖move_up末值), 方便调整夹持力度
        grip_positions = [
            [8, int(500 - 500*GRIP_SCALE)], [16, int(500 + 500*GRIP_SCALE)]
        ]
        if self.args.dry_run:
            log.info("[dry-run] close hand servos: %s", grip_positions)
        else:
            self.board.bus_servo_set_position(0.5, grip_positions)
        time.sleep(0.6)
        self.actions.lock_hands()

    def verify_pickup(self) -> bool:
        """低头检查地上是否还有蓝色海绵. 返回True=没抓到需重抓, False=抓到了"""
        log.info("verify_pickup: 低头检查地上是否还有海绵")
        self.set_head(self.servo2, self.servo1 + 220, duration=400)
        time.sleep(0.6)
        seen = 0
        for _ in range(15):
            with self._lock:
                ct = self.color_target
            if ct.cx >= 0 and ct.area > 2000:
                seen += 1
                log.info("verify_pickup: 地上看到海绵 area=%.0f cx=%d (连续%d帧)", ct.area, ct.cx, seen)
            else:
                seen = 0
            if seen >= 3:
                self.set_head_center(duration=300)
                return True
            time.sleep(0.1)
        self.set_head_center(duration=300)
        log.info("verify_pickup: 地上无海绵, 抓取成功")
        return False

    def position_for_target_search(self):
        """Move from the pickup point into the left-side Tag 38 search lane."""
        log.info(
            "post-pick route: turn left %d times, then forward %d steps",
            self.args.post_pick_left_turns,
            self.args.post_pick_forward_steps,
        )
        self.actions.run(
            "turn_left",
            times=self.args.post_pick_left_turns,
            lock=True,
        )
        self.actions.run(
            "go_forward",
            times=self.args.post_pick_forward_steps,
            lock=True,
        )
        self.actions.run("stand", lock=True)
        self.set_head_center(duration=300)

    def approach_tag(self, tag: Optional[TagTarget]) -> bool:
        if tag is None:
            self.scan_head_or_turn(lock=True, turn_direction="left")
            return False

        if self.x_dis != self.servo2:
            if self.x_dis > self.servo2:
                self.actions.run("turn_left_small_step", lock=True)
            else:
                self.actions.run("turn_right_small_step", lock=True)
            self.set_head_center(duration=300)
            return False

        # 1.先转身对正(tag cx到中间), 2.前进靠近, 3.近区平移微调
        dx = tag.cx - CENTER_X
        log.info("target tag id=%d cx=%d cy=%d dx=%d dist=%.2fm", tag.tag_id, tag.cx, tag.cy, dx, tag.distance)
        if tag.distance > TAG_PLACE_FAR_M or tag.distance == 0:
            # 远区: 先转身对正(cx到中间), 再前进
            if abs(dx) > TAG_X_TURN:
                # 远区大转弯(只大偏移>150), 中等偏移小转弯, 防过冲振荡
                if tag.distance > TAG_PLACE_BIG_M and abs(dx) > 150:
                    self.actions.run("turn_right" if dx > 0 else "turn_left", lock=True)
                else:
                    self.actions.run("turn_right_small_step" if dx > 0 else "turn_left_small_step", lock=True)
                time.sleep(0.3)  # 转身后等画面稳定, 防过冲反向转
            elif tag.distance > TAG_PLACE_FAR_M:
                # 对正了, 前进靠近: 远用大步, 近用小步
                self.actions.run("go_forward" if tag.distance > TAG_PLACE_BIG_M else "go_forward_one_step", lock=True)
            elif tag.cy <= TAG_FAR_Y:
                # 无PnP用像素cy(远区大步)
                self.actions.run("go_forward", lock=True)
            else:
                log.info("target area aligned for placing")
                return True
        elif tag.distance >= TAG_PLACE_NEAR_M:
            # 近区: 大偏移先转身对正(快), 小偏移平移精调
            if abs(dx) > TAG_X_TURN:
                self.actions.run("turn_right_small_step" if dx > 0 else "turn_left_small_step", lock=True)
            elif abs(dx) > TAG_X_LARGE:
                self.actions.run("right_move_30" if dx > 0 else "left_move_30", lock=True)
            elif abs(dx) > TAG_X_FINE:
                self.actions.run("right_move" if dx > 0 else "left_move", lock=True)
            else:
                log.info("at place distance %.2fm, place", tag.distance)
                return True
        else:
            # 比墙上Tag目标距离更近
            log.info("too near %.2fm, back", tag.distance)
            self.actions.run("back_fast", lock=True)
        return False

    def place_custom(self):
        """放置: 弯腰下放→松夹→等2秒→手臂复位→等2秒→起身站立"""
        ls = LOCK_SERVOS
        open8, open16 = 950, 50  # 夹爪松开(大)
        mid6, mid7, mid8, mid14, mid15, mid16 = 575, 800, 725, 425, 200, 275  # 站立手臂中位
        # 帧1 弯腰下放(腿弯腰, 手臂LOCK, 夹爪闭)
        bend = [500, 125, 900, 750, 500,  ls['6'], ls['7'], ls['8'],
                500, 875, 100, 250, 500,  ls['14'], ls['15'], ls['16'],  500, 500]
        # 帧2 松夹(腿弯腰, 手臂LOCK, 夹爪开)
        bend_open = list(bend)
        bend_open[7] = open8
        bend_open[15] = open16
        # 帧3 手臂复位(腿保持弯腰, 手臂回中位, 夹爪中位)
        arm_reset = [500, 125, 900, 750, 500,  mid6, mid7, mid8,
                     500, 875, 100, 250, 500,  mid14, mid15, mid16,  500, 500]
        # 帧4 起身站立(腿站立, 手臂中位, 夹爪中位)
        stand = [500, 388, 500, 594, 500,  mid6, mid7, mid8,
                 500, 612, 500, 406, 500,  mid14, mid15, mid16,  500, 500]
        log.info("place: 弯腰下放")
        self._play_place_frame(bend, 1.0)
        log.info("place: 松夹")
        self._play_place_frame(bend_open, 0.8)
        log.info("place: 等夹爪松开+物体落地(2秒)")
        time.sleep(2.0)
        log.info("place: 手臂复位")
        self._play_place_frame(arm_reset, 1.0)
        log.info("place: 等手臂复位完整(2秒)")
        time.sleep(2.0)
        log.info("place: 起身站立")
        self._play_place_frame(stand, 1.0)

    def _play_place_frame(self, frame, dur):
        positions = [[i + 1, frame[i]] for i in range(18)]
        if self.args.dry_run:
            log.info("[dry-run] place frame duration=%.1fs positions=%s", dur, positions)
        else:
            self.board.bus_servo_set_position(dur, positions)
        time.sleep(dur + 0.2)

    def place_object(self):
        self.actions.run("go_forward_one_step", times=PLACE_FINAL_STEPS, lock=True)
        self.actions.run("stand", lock=True)
        self.actions.unlock_hands()
        self.actions.run("put_down")
        self.actions.run("back_fast", times=self.args.back_steps_after_place, with_stand=True)
        self.actions.run("stand")

    # ---- 红色胶带巡线 ----

    def _get_area_max_contour(self, contours, min_area=100):
        """返回面积最大的轮廓(照搬 VisualPatrol.getAreaMaxContour)"""
        area_max = 0.0
        cnt_max = None
        for c in contours:
            a = math.fabs(cv2.contourArea(c))
            if a > area_max:
                area_max = a
                if a > min_area:
                    cnt_max = c
        return cnt_max, area_max

    def detect_red_line(self, img) -> Tuple[int, np.ndarray]:
        """巡线: HSV双段红色检测 + 3ROI加权. 返回(中心x或-1, display). 设self.line_is_vertical."""
        display = img.copy()
        gb = cv2.GaussianBlur(img, (3, 3), 3)
        hsv = cv2.cvtColor(gb, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, np.array([RED_H_LOW1, RED_S_LOW, RED_V_LOW]),
                         np.array([RED_H_HIGH1, RED_S_HIGH, RED_V_HIGH]))
        m2 = cv2.inRange(hsv, np.array([RED_H_LOW2, RED_S_LOW, RED_V_LOW]),
                         np.array([RED_H_HIGH2, RED_S_HIGH, RED_V_HIGH]))
        mask = cv2.bitwise_or(m1, m2)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.dilate(cv2.erode(mask, k), k)
        mask[:, 0:160] = 0
        mask[:, 480:640] = 0
        centroid_x_sum = 0.0
        weight_sum = 0.0
        self.line_is_vertical = False
        for r in LINE_ROI:
            sub = mask[r[0]:r[1], r[2]:r[3]]
            cnts = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)[-2]
            cnt_large, _ = self._get_area_max_contour(cnts)
            if cnt_large is not None:
                rect = cv2.minAreaRect(cnt_large)
                box = np.intp(cv2.boxPoints(rect))
                cx = float(box[0][0] + box[2][0]) / 2.0
                centroid_x_sum += cx * r[4]
                weight_sum += r[4]
                box[:, 1] = box[:, 1] + r[0]
                cv2.drawContours(display, [box], -1, (0, 0, 255), 2)
        # 竖直判定: 用三段ROI合并区域的红线轮廓PCA主轴方向(与竖直夹角<=VERTICAL_ANGLE_TOL).
        # 单带只有40px高, 轮廓被带宽截断, 线宽>40px时截断块主轴反而变水平;
        # 合并区域足够高(140px), 主轴方向才反映线真实方向, 且与线宽/距离无关.
        # (之前高宽比判据在近处宽线时比值仅~1.1, 永远判不竖直)
        _y1 = min(r[0] for r in LINE_ROI)
        _y2 = max(r[1] for r in LINE_ROI)
        _mcnts = cv2.findContours(mask[_y1:_y2, :], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)[-2]
        _mlarge, _ = self._get_area_max_contour(_mcnts)
        if _mlarge is not None:
            _m = cv2.moments(_mlarge)
            if _m["m00"] > 0:
                _mu11 = _m["mu11"] / _m["m00"]
                _mu20 = _m["mu20"] / _m["m00"]
                _mu02 = _m["mu02"] / _m["m00"]
                _angle = math.degrees(0.5 * math.atan2(2.0 * _mu11, _mu20 - _mu02))
                if abs(_angle) >= 90.0 - VERTICAL_ANGLE_TOL:
                    self.line_is_vertical = True
        if weight_sum > 0:
            line_cx = int(self.kf_line_cx.update(centroid_x_sum / weight_sum))
            cv2.circle(display, (line_cx, 460), 8, (0, 255, 255), -1)
            return line_cx, display
        return -1, display

    def follow_line(self, max_steps: int, stop_on_blue: bool = False, label: str = ""):
        """确认式直行循迹: 静置FOLLOW_CONFIRM_S秒后判断线是否竖直;
        竖直则一次直行max_steps步; 否则按dx调整朝向再静置重判, 循环"""
        log.info("follow_line %s: confirm=%.1fs straight=%d stop_on_blue=%s",
                 label, FOLLOW_CONFIRM_S, max_steps, stop_on_blue)
        self.set_head(self.servo2, self.servo1 + self.line_head_delta, duration=400)
        time.sleep(0.5)
        steps = 0
        self.lost_turns = 0
        no_frame_start = None
        adjust_count = 0
        while True:
            # 静置确认期: 等画面稳定后再判断(转身/调整后画面会甩动)
            time.sleep(FOLLOW_CONFIRM_S)
            ok, frame = self.camera.read()
            if not ok or frame is None:
                if no_frame_start is None:
                    no_frame_start = time.time()
                if time.time() - no_frame_start > LINE_LOST_TIMEOUT:
                    log.info("follow_line: 相机长时间无帧, 停")
                    break
                continue
            no_frame_start = None
            frame = cv2.remap(frame, self.mapx, self.mapy, cv2.INTER_LINEAR)
            if stop_on_blue:
                ct, _ = self.detect_blue(frame)
                if ct.cx >= 0 and ct.area >= COLOR_AREA_MIN * 2:
                    log.info("follow_line: 蓝海绵出现(area=%.0f), 停止巡线", ct.area)
                    break
            line_cx, line_display = self.detect_red_line(frame)
            now = time.time()
            if now - self.last_debug_save >= 0.5:
                self.last_debug_save = now
                snap_path = "/home/pi/codes/pictures/line_%s.jpg" % time.strftime("%H%M%S")
                os.makedirs("/home/pi/codes/pictures", exist_ok=True)
                cv2.imwrite(snap_path, line_display)
            if line_cx >= 0 and self.line_is_vertical:
                # 线竖直确认: 一次直行max_steps步, 结束循迹
                log.info("follow_line: 线竖直确认, 直行%d步", max_steps)
                self.actions.run("go_forward_one_step", times=max_steps)
                steps += max_steps
                break
            # 未竖直: 调整后重新静置判断
            if adjust_count >= FOLLOW_MAX_ADJUST:
                log.info("follow_line: 调整%d次仍未确认竖直, 直行%d步", adjust_count, max_steps)
                self.actions.run("go_forward_one_step", times=max_steps)
                steps += max_steps
                break
            adjust_count += 1
            if line_cx < 0:
                # 丢线: 左大转找回
                self.lost_turns += 1
                if self.lost_turns > MAX_LOST_TURNS:
                    log.info("follow_line: 丢线转%d次未找回停", MAX_LOST_TURNS)
                    break
                log.info("follow_line: 丢线左大转找 %d/%d", self.lost_turns, MAX_LOST_TURNS)
                self.actions.run("turn_left")
            else:
                self.lost_turns = 0
                dx = line_cx - LINE_CENTER_X
                self.actions.run("turn_right_small_step" if dx > 0 else "turn_left_small_step")
                log.info("follow_line: 未竖直(dx=%d) 小步转调整 %d/%d", dx, adjust_count, FOLLOW_MAX_ADJUST)
        self.set_head_center(duration=300)
        log.info("follow_line %s done: steps=%d", label, steps)

    def search_line_and_follow(self, max_steps: int, max_turns: int = 8):
        """放置后: 右转找红线, 找到就沿红线走到终点(走满max_steps). 返回是否找到并走完."""
        log.info("search_line: 右转找红线最多%d次", max_turns)
        # 重置巡线中心卡尔曼, 避免画面中心旧值拉偏首帧
        self.kf_line_cx = Kalman1DConst(LINE_CENTER_X, Q=1.0, R=10.0)
        self.set_head(self.servo2, self.servo1 + self.line_head_delta, duration=400)
        time.sleep(0.5)
        found = False
        last_side = 0   # 最近一次看到红线时的 dx 符号(+1右/-1左), 用于丢线后反向回找
        turns = 0
        while turns < max_turns:
            ok, frame = self.camera.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue
            frame = cv2.remap(frame, self.mapx, self.mapy, cv2.INTER_LINEAR)
            line_cx, line_display = self.detect_red_line(frame)
            snap_path = "/home/pi/codes/pictures/searchline_%s_%d.jpg" % (time.strftime("%H%M%S"), turns)
            os.makedirs("/home/pi/codes/pictures", exist_ok=True)
            cv2.imwrite(snap_path, line_display)
            if line_cx >= 0:
                found = True
                dx = line_cx - LINE_CENTER_X
                if dx != 0:
                    last_side = 1 if dx > 0 else -1
                # 对正条件: 红色矩形必须竖着(机器人已面向红线) 且 位于画面正中, 才进入巡线
                if self.line_is_vertical and abs(dx) <= SEARCH_LINE_ALIGN_THRESHOLD:
                    log.info("search_line: 第%d次红线竖直且居中 cx=%d, 开始巡线", turns + 1, line_cx)
                    break
                if self.line_is_vertical:
                    # 线已竖直但偏: 用平移横向对正(保持朝向不变, 避免转身过冲甩出视野)
                    move_action = "right_move" if dx > 0 else "left_move"
                    log.info("search_line: 第%d次红线竖直但偏 dx=%d, 平移对正", turns + 1, dx)
                else:
                    # 线还是横/斜的: 继续向右转找正面(直到红色矩形竖起来)
                    move_action = "turn_right_small_step"
                    log.info("search_line: 第%d次看到红线但未竖直, 继续右转找正面", turns + 1)
                self.actions.run(move_action)
                turns += 1
                time.sleep(0.1)  # 判断间隔(原0.3, 缩短加快找线)
                continue
            # 未看到红线
            if found and last_side != 0:
                # 之前看到过线现在丢了: 朝上次方向的反方向小步回找, 不继续朝原方向扫描
                back_action = "turn_left_small_step" if last_side > 0 else "turn_right_small_step"
                log.info("search_line: 红线丢失, 反向回找 %s (last_side=%d)", back_action, last_side)
                self.actions.run(back_action)
            else:
                self.actions.run("turn_right_small_step")  # 小转弯找红线(避免转过头)
            turns += 1
            time.sleep(0.1)  # 判断间隔(原0.3, 缩短加快找线)
        if found:
            self.follow_line(max_steps, stop_on_blue=False, label="to_end")
            return True
        log.warning("search_line: %d次右转未找到红线, 结束巡线", max_turns)
        self.set_head_center(duration=300)
        return False

    def navigate_to_end_tag(self, tag_id):
        """放置后: 找终点tag, 小步转直到-15<yaw<5后大步直行10步穿过"""
        log.info("navigate to end tag %d (yaw_range=[%d,%d] walk=10)", tag_id, END_YAW_LOWER, END_YAW_UPPER)
        self.ctl.set_pwm_servo_pulse(1, 1500, 500)
        time.sleep(0.3)
        # 阶段1: 小步转直到-15<yaw<5
        yaw_locked = False
        for step in range(MAX_TURN):
            ok, frame = self.camera.read()
            if not ok or frame is None:
                time.sleep(0.02); continue
            frame = cv2.remap(frame, self.mapx, self.mapy, cv2.INTER_LINEAR)
            tags, _ = self.detect_tags(frame)
            target = None
            for t in tags:
                if t.tag_id == tag_id:
                    target = t; break
            if target is None:
                log.info("end tag %d step=%d not found, turn_right_small_step", tag_id, step)
                self.actions.run("turn_right_small_step", lock=False)
                self.ctl.set_pwm_servo_pulse(1, 1500, 200)
                time.sleep(0.5)
                continue
            yaw = target.yaw
            log.info("end tag %d step=%d yaw=%.1f offset=%.3f forward=%.3f cx=%d",
                     tag_id, step, yaw, target.offset, target.forward, target.cx)
            if END_YAW_LOWER < yaw < END_YAW_UPPER:
                log.info("end tag %d YAW LOCKED at %.1f, walk %d steps", tag_id, yaw, WALK_STEPS)
                yaw_locked = True
                break
            # yaw在5~10之间, 转一步就开始走
            if 5 < yaw < 10:
                log.info("end tag %d yaw=%.1f in [5,10], turn one step then walk", tag_id, yaw)
                self.actions.run("turn_right_small_step", lock=False)
                self.ctl.set_pwm_servo_pulse(1, 1500, 200)
                time.sleep(0.3)
                yaw_locked = True
                break
            action = "turn_right_small_step" if yaw > 0 else "turn_left_small_step"
            log.info("end tag %d yaw=%.1f -> %s", tag_id, yaw, action)
            self.actions.run(action, lock=False)
            self.ctl.set_pwm_servo_pulse(1, 1500, 200)
            time.sleep(0.5)
        if not yaw_locked:
            log.info("end tag %d 转了%d步没锁定, 直行", tag_id, MAX_TURN)
        # 阶段2开始前重置卡尔曼(小步转→直行, 角速度变化)
        self.kf_angle = Kalman1D(0.0, Q_angle=5.0, Q_v=1.0, R=10.0)
        log.info("end tag %d [RESET] 卡尔曼重置, 进入直行阶段", tag_id)
        # 阶段2: 大步直行10步, 每步后检查yaw超范围就转一步调整
        for step in range(WALK_STEPS):
            ok, frame = self.camera.read()
            if not ok or frame is None:
                time.sleep(0.02); continue
            frame = cv2.remap(frame, self.mapx, self.mapy, cv2.INTER_LINEAR)
            tags, _ = self.detect_tags(frame)
            target = None
            for t in tags:
                if t.tag_id == tag_id:
                    target = t; break
            if target is not None:
                yaw = target.yaw
                log.info("end tag %d walk=%d/%d yaw=%.1f offset=%.3f forward=%.3f",
                         tag_id, step + 1, WALK_STEPS, yaw, target.offset, target.forward)
                if yaw < END_YAW_LOWER or yaw > END_YAW_UPPER:
                    action = "turn_right_small_step" if yaw > 0 else "turn_left_small_step"
                    log.info("end tag %d yaw=%.1f out of range, %s to adjust", tag_id, yaw, action)
                    self.actions.run(action, lock=False)
                    self.ctl.set_pwm_servo_pulse(1, 1500, 200)
                    time.sleep(0.3)
            else:
                log.info("end tag %d walk=%d/%d tag lost, go forward", tag_id, step + 1, WALK_STEPS)
            self.actions.run("go_forward_fast", lock=False)
            self.ctl.set_pwm_servo_pulse(1, 1500, 200)
            time.sleep(0.3)
        log.info("end tag %d done", tag_id)

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

    def action_loop(self):
        """动作线程: 读最新检测结果, 执行approach/pick/place(跟官方move一样)"""
        while self.running and self.state != "FINISH":
            with self._lock:
                ct = self.color_target
                tags = list(self.tags)
            if self.state == "SEARCH_OBJECT":
                if self.approach_color(ct):
                    if self.args.blue_pickup_test:
                        self.enter_state("PICK_OBJECT")
                    elif self.creep_to_pick_distance():
                        self.enter_state("PICK_OBJECT")
            elif self.state == "PICK_OBJECT":
                self.pick_object()
                if self.args.blue_pickup_test:
                    self.enter_state("BLUE_PICKUP_HOLD")
                elif self.verify_pickup():
                    self.pick_retries += 1
                    if self.pick_retries > MAX_PICK_RETRIES:
                        log.error("verify_pickup: 抓取失败%d次, 放弃", self.pick_retries)
                        self.enter_state("FINISH")
                    else:
                        log.info("verify_pickup: 没抓到, 回 SEARCH_OBJECT 重试 %d/%d", self.pick_retries, MAX_PICK_RETRIES)
                        self.color_step = 1
                        self.actions.run("back_fast", times=2)
                        self.enter_state("SEARCH_OBJECT")
                else:
                    self.pick_retries = 0
                    self.position_for_target_search()
                    self.enter_state("SEARCH_TARGET_AREA")
            elif self.state == "BLUE_PICKUP_HOLD":
                log.info("blue pickup test: holding %.1fs", self.args.hold_after_pick)
                time.sleep(max(0.0, self.args.hold_after_pick))
                self.enter_state("FINISH")
            elif self.state == "SEARCH_TARGET_AREA":
                target = self.choose_target_area_tag(tags)
                if self.approach_tag(target):
                    self.enter_state("PLACE_OBJECT")
            elif self.state == "PLACE_OBJECT":
                self.place_object()
                self.enter_state("FINISH")
            time.sleep(0.01)

    def run(self):
        camera = None
        worker = None
        try:
            camera = self.open_camera()
            self.camera = camera
            camera_ready = False
            for _ in range(30):
                ok, frame = camera.read()
                if ok and frame is not None:
                    camera_ready = True
                    break
                time.sleep(0.05)
            if not camera_ready:
                raise RuntimeError("camera opened but returned no frames; motion was not started")

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
                "single-sponge task start dry_run=%s pick=%s target_tag=%d",
                self.args.dry_run,
                self.args.pick_action,
                self.target_tag_id,
            )

            self.running = True
            worker = threading.Thread(target=self.action_loop, daemon=True)
            worker.start()

            while self.running and self.state != "FINISH":
                ok, frame = camera.read()
                if not ok or frame is None:
                    time.sleep(0.02)
                    continue
                frame = cv2.remap(frame, self.mapx, self.mapy, cv2.INTER_LINEAR)
                color_target, display = self.detect_blue(frame)
                tags, _ = self.detect_tags(frame)
                with self._lock:
                    self.color_target = color_target
                    self.tags = tags
                    self.display_frame = display

                now = time.time()
                if now - self.last_debug_save >= 5.0:
                    self.last_debug_save = now
                    snap_path = "/home/pi/codes/pictures/snap_%s.jpg" % time.strftime("%H%M%S")
                    os.makedirs("/home/pi/codes/pictures", exist_ok=True)
                    cv2.imwrite(snap_path, display)

                if self.args.debug and self.debug_gui:
                    cv2.imshow("sorting_task", display)
                    if cv2.waitKey(1) == 27:
                        log.info("ESC pressed")
                        break

            self.running = False
            worker.join(timeout=2)

            # 放置后: 转身找红线 → 沿红线走到终点; 找不到红线时回退找终点tag
            if self.state == "FINISH" and not self.args.blue_pickup_test:
                line_ok = self.search_line_and_follow(
                    self.args.line_final_steps, self.args.line_search_turns)
                if not line_ok and self.args.end_tag:
                    log.info("search_line 未找到红线, 回退到找终点 tag %d", self.args.end_tag)
                    self.navigate_to_end_tag(self.args.end_tag)

        except BaseException as exc:
            log.exception("fatal error: %s", exc)
        finally:
            self.running = False
            if worker is not None:
                worker.join(timeout=2)
            if camera is not None:
                camera.release()
            if self.debug_gui:  # 无GUI编译的OpenCV(机器人无DISPLAY)下destroyAllWindows会抛cv2.error
                cv2.destroyAllWindows()
            if self.robot_initialized and not self.args.dry_run:
                AGC.stopActionGroup()
                if not (self.args.blue_pickup_test and self.args.skip_test_putdown):
                    self.actions.unlock_hands()
                    self.actions.run("stand")
            if self.services_stopped:
                subprocess.run(["sudo", "systemctl", "start", "tonypi", "joystick"], stderr=subprocess.DEVNULL)
            log.info("task exit")


def parse_args():
    parser = argparse.ArgumentParser(description="Pick one tagless blue sponge and place it on a Tag 38 board")
    parser.add_argument("--debug", action="store_true", help="show camera window")
    parser.add_argument("--debug-save-interval", type=float, default=0.5, help="seconds between saved debug frames when no DISPLAY is available")
    parser.add_argument("--dry-run", action="store_true", help="log actions without executing motion")
    parser.add_argument("--no-kill", action="store_true", help="do not kill existing TonyPi/Joystick processes")
    parser.add_argument("--pick-action", default="move_up", help="existing action group used to pick the sponge")
    parser.add_argument("--target-tag", type=int, default=DEFAULT_TARGET_TAG_ID, help="AprilTag ID on the target board")
    parser.add_argument("--end-tag", type=int, default=26, help="终点tag id, 放置后走到该tag(0=禁用)")
    parser.add_argument("--post-pick-left-turns", type=int, default=8,
                        help="抓取后左转次数(当前8)")
    parser.add_argument("--post-pick-forward-steps", type=int, default=5)
    parser.add_argument("--back-steps-after-place", type=int, default=12, help="放置后后退步数(退远一点才够转身找到身后红线)")
    # 红色胶带巡线
    parser.add_argument("--line-initial-steps", type=int, default=8, help="开机沿红色胶带走几步到海绵区(0=禁用,看见蓝海绵提前停)")
    parser.add_argument("--line-search-turns", type=int, default=30, help="放置后右转找红线最多几次(看到即停)")
    parser.add_argument("--line-final-steps", type=int, default=20,
                        help="放置后沿红线走到终点的步数(确认线竖直后一次直行的步数)")
    parser.add_argument("--line-head-delta", type=int, default=60, help="巡线时低头角度(相对servo1, 60=看远)")
    parser.add_argument("--pick-area-threshold", type=float, default=4762.0,
                        help="海绵面积达到此值即视为足够近, 触发抓取")
    parser.add_argument("--pick-y-threshold", type=int, default=350,
                        help="海绵cy达到此值即视为足够近, 触发抓取")
    parser.add_argument("--pick-too-near-y", type=int, default=420)
    parser.add_argument(
        "--blue-pickup-test",
        action="store_true",
        help="minimal test: detect blue object, approach it, pick it up, then optionally put it down",
    )
    parser.add_argument("--hold-after-pick", type=float, default=5.0, help="seconds to hold the object in blue pickup test")
    parser.add_argument("--skip-test-putdown", action="store_true", help="in blue pickup test, do not run put_down after holding")
    args = parser.parse_args()
    # calib_config.json 覆盖参数(命令行显式指定的优先)
    _cli_opts = {a.split("=", 1)[0] for a in sys.argv if a.startswith("--")}
    if CALIB_CONFIG:
        for _dest in CALIB_ARG_KEYS:
            if _dest in CALIB_CONFIG and _ARG_OPTION_STRINGS[_dest] not in _cli_opts:
                _new = CALIB_CONFIG[_dest]
                try:
                    setattr(args, _dest, type(getattr(args, _dest))(_new))
                except (TypeError, ValueError) as exc:
                    log.warning("calib_config 参数 %s=%r 类型不合法, 忽略: %s", _dest, _new, exc)
    if min(args.post_pick_left_turns, args.post_pick_forward_steps, args.back_steps_after_place,
           args.line_initial_steps, args.line_search_turns, args.line_final_steps) < 0:
        parser.error("movement counts must be nonnegative")
    if args.pick_area_threshold <= 0:
        parser.error("--pick-area-threshold must be positive")
    if not 0 < args.pick_y_threshold < args.pick_too_near_y <= CAM_SIZE[1]:
        parser.error("pickup Y thresholds must satisfy 0 < pick-y-threshold < pick-too-near-y <= 480")
    return args


def main():
    args = parse_args()
    task = SortingTask(args)
    task.run()


if __name__ == "__main__":
    main()
