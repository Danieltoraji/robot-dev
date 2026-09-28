#!/usr/bin/python3
# coding=utf8
"""
RedLinePatrol.py — 红色线专用巡线

功能：
  只追踪红色线，黑线、白线及其他颜色线不会干扰。
  使用 HSV 色彩空间阈值提取红色区域，
  再用 ROI 加权法计算线中心，控制机器人沿线行走。

  支持：虚线抗干扰、十字路口抗干扰、闭合环形线抗干扰。

运行方式：
  sudo systemctl stop tonypi
  python3 RedLinePatrol.py

可调参数见下方 ============ 可调参数 ============ 区域
"""
import sys
from pathlib import Path

TONYPI_DIR = Path('/home/pi/TonyPi')
HIWONDER_SDK_DIR = TONYPI_DIR / 'HiwonderSDK'
ACTION_GROUP_DIR = TONYPI_DIR / 'ActionGroups'

for extra_path in (TONYPI_DIR, HIWONDER_SDK_DIR):
    extra_path_str = str(extra_path)
    if extra_path_str not in sys.path:
        sys.path.insert(0, extra_path_str)

import math
import os
import sys
import threading
import time

import cv2
import numpy as np

# hiwonder SDK 仅存在于机器人上。PC 离线时用桩对象顶替，保证模块可导入
# 与离线回放（动作调用变成 no-op，视觉逻辑照常）；机器人上行为不变。
try:
    import hiwonder.ActionGroupControl as AGC
    import hiwonder.Camera as Camera
    import hiwonder.Misc as Misc
    import hiwonder.ros_robot_controller_sdk as rrc
    import hiwonder.yaml_handle as yaml_handle
    from hiwonder.Controller import Controller
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False
    print("RedLinePatrolV5：未找到 hiwonder SDK，进入离线模式（动作不执行）")

    class _NoopController:
        def set_pwm_servo_pulse(self, *args, **kwargs):
            pass

    class _NoopBoard:
        pass

    class _MockAGC:
        @staticmethod
        def runActionGroup(*args, **kwargs):
            pass

    class _MockMisc:
        @staticmethod
        def map(v, in_min, in_max, out_min, out_max):
            return out_min + (v - in_min) * (out_max - out_min) / max(
                1.0, float(in_max - in_min))

    class _MockCamera:
        pass

    class _MockYamlHandle:
        @staticmethod
        def get_yaml_data(*args, **kwargs):
            return {}

    AGC = _MockAGC()
    Camera = _MockCamera()
    Misc = _MockMisc()
    rrc = _NoopBoard()
    yaml_handle = _MockYamlHandle()
    Controller = _NoopController

if sys.version_info.major == 2:
    print("Please run this program with python3!")
    sys.exit(0)

try:
    if __name__ == "__main__":
        from CameraCalibration.CalibrationConfig import *
    else:
        from Functions.CameraCalibration.CalibrationConfig import *
except ImportError:
    # PC 离线：标定路径不可用，仅影响独立运行入口加载 npz。
    calibration_param_path = None

# ============ 可调参数 ============
# --- 红色 HSV 阈值 ---
# 红色在 HSV 中横跨 0 度，需要两段范围合并
RED_H_LOW1 = 0  # 第一段色相下限
RED_H_HIGH1 = 6  # 第一段色相上限
RED_H_LOW2 = 149  # 第二段色相下限
RED_H_HIGH2 = 180  # 第二段色相上限
RED_S_LOW = 31  # 饱和度下限（越高越严格，排除浅色/白色）
RED_S_HIGH = 255  # 饱和度上限
RED_V_LOW = 50  # 明度下限（越高越排除暗色/黑色）
RED_V_HIGH = 255  # 明度上限

MIN_CONTOUR_AREA = 50  # 最小轮廓面积，过滤噪点
TURN_THRESHOLD = 55  # 偏差超过此像素值才转向
TURN_STOP_THRESHOLD = 40  # 转向后回到此偏差内才恢复直行，防止左右反复过冲
TURN_ACTION_COOLDOWN = 0.8  # 转向动作后的视觉刷新等待时间
TURN_CONFIRM_FRAMES = 3  # 转向后等待新的视觉帧，避免连续原地转
# 竖线检测：轮廓高宽比超过此值视为"机器人与线平行"，改用横向平移而非转向
# 转弯步幅较大，因此这里故意放宽，不要求像素级精确平行。
VERTICAL_LINE_RATIO = 2.0  # 普通巡线使用的阈值
TURN_PARALLEL_RATIO = 0.75  # 近处 ROI 只有 40px 高，粗红线转正时高/宽可能小于 1.2
TURN_PARALLEL_ANGLE_TOL = 25  # 最近红线主轴与竖直方向的夹角容忍度（度）
TURN_PARALLEL_MIN_HEIGHT = 18  # 排除只有几像素高的横线/噪点
LINE_PARALLEL_SLANT_TOL = 20  # 中/近两段线中心 x 之差阈值（像素）
TURN_PARALLEL_CONFIRM_FRAMES = 2  # 连续满足多少帧才确认大致平行
TURN_CENTER_TOLERANCE = 70  # 最近红线允许偏离画面中心的像素数
LINE_THICKNESS_MAX = 80  # 线状轮廓的"厚度"上限，用来排除直角弯角块
NEAR_LINE_MIN_Y = 400  # 只把画面下部（近处）红线算入转弯完成判定
TURN_REARM_COOLDOWN = 1.5  # 转弯结束后短时间内不重复触发同一个弯
TURN_MAX_ACTIONS = 8  # 单次直角弯最多执行的原地转弯动作数，防止视觉失效后无限旋转
TURN_MAX_SECONDS = 12.0  # 单次转弯最长持续时间（秒）
POST_TURN_LOOK_SECONDS = 2.0  # 转弯结束后短暂放宽弯道检测，抓很近的下一个直角弯
POST_TURN_CORNER_Y = 400  # 急弯判断时角块底部的放宽阈值（像素）
LATERAL_LEFT_ACTION = "left_move"  # 线在左侧时向左平移
LATERAL_RIGHT_ACTION = "right_move"  # 线在右侧时向右平移
FORWARD_ACTION = "go_forward_one_step"
TURN_LEFT_ACTION = "turn_left"
TURN_RIGHT_ACTION = "turn_right"


MORPH_KERNEL_SIZE = 3  # 形态学核大小
ROI_LEFT_CROP = 160  # ROI 左侧裁剪像素（排除边缘干扰）
ROI_RIGHT_CROP = 160  # ROI 右侧裁剪像素

# --- 虚线抗干扰参数 ---
LINE_LOST_TIMEOUT = 1.5  # 线丢失超过此秒数才判定真正丢线（秒）
LINE_MEMORY_FRAMES = 8  # 记忆最近 N 帧的线中心，用于丢线时预测方向

# --- 直角转弯参数 ---
CROSS_WIDTH_RATIO = 2.5  # 轮廓宽度 / 正常线宽 超过此比值判定为直角弯
NORMAL_LINE_WIDTH = 40  # 正常线条在 ROI 中的大致宽度（像素），根据实际调整
CORNER_TRIGGER_Y = 400  # 直角弯横线底部到达此行后触发；原代码硬编码 460，识别过晚

last_center_y = 240
APPROACH_STEPS = 3  # 转弯前直行步数
APPROACH_STEP_INTERVAL = 0.7  # 每步直行的时间间隔（秒），保证真的走出余量
# --- 闭合线（环形轨道）抗干扰参数 ---
MAX_CENTER_JUMP = 120  # 帧间线中心最大跳变（像素），超过则忽略（防止跳到环另一侧）
PREFER_NEAREST_CONTOUR = True  # True: 多轮廓时选离上次中心最近的，False: 选最大面积的
SHOW_DISPLAY = False  # 无显示环境保持 False，避免 cv2.imshow 崩溃
HEAD_TILT_OFFSET = (
    60  # 头部仰角微调，正值抬头，负值低头（叠加在 servo_config 基础值上）
)
# ==================================
# --- 直角弯参数（替代原十字路口逻辑） ---
running = False
turn_started = False  # 新增：是否正在转弯中
TURN_DIRECTION = 0  # 新增：-1左转，1右转
board = rrc.Board() if HARDWARE_AVAILABLE else None
ctl = Controller(board) if HARDWARE_AVAILABLE else None

servo_data = None


def load_config():
    global servo_data
    servo_data = yaml_handle.get_yaml_data(yaml_handle.servo_file_path)


if HARDWARE_AVAILABLE:
    load_config()


def initMove():
    """固定头部朝下看地面线"""
    ctl.set_pwm_servo_pulse(1, servo_data["servo1"] + HEAD_TILT_OFFSET, 500)
    ctl.set_pwm_servo_pulse(2, servo_data["servo2"], 500)


# 巡线状态
line_center_x = -1
line_center_frame = 0
img_centerx = 320
line_lost_time = 0  # 线丢失起始时间戳
line_history = []  # 最近 N 帧的线中心 x 坐标
last_valid_center = 320  # 最后一次有效线中心（虚线间隙时用）

# 进弯直行状态：检测到直角弯后先走几步再转弯
approach_active = False  # 是否处于进弯直行阶段
approach_last_step_time = 0  # 上一步直行的时间戳
approach_steps_done = 0  # 已完成直行步数
line_is_vertical = False  # 当前帧线条是否竖直（机器人与线平行）

# 转弯状态下，单独记录"最近红线"，避免远处红线干扰完成判定
nearest_line_center_x = -1
nearest_line_bottom_y = -1
nearest_line_is_parallel = False
turn_parallel_count = 0
turn_frame_count = 0  # 当前转弯已经持续的帧数
turn_action_count = 0  # 当前直角弯已执行的转弯动作数
turn_start_time = 0.0  # 当前直角弯开始时间
turn_end_time = 0  # 上次转弯结束时间戳
post_turn_look_end = 0  # 急弯判断窗口结束时间戳

# demoV4 总控接口：已完成直角弯计数与最近一次转弯完成时间（终点判定的
# 进度门限与转弯冷却期用）；V5 无主动搜索，search_active 恒为 False。
corner_index = 0
last_turn_completed_at = 0.0
search_active = False


def get_corner_index():
    """已完成直角弯数（demoV4 终点判定进度门限用）。"""
    return corner_index

enter = False


def reset():
    global \
        line_center_x, \
        line_center_frame, \
        line_lost_time, \
        line_history, \
        last_valid_center, \
        line_is_vertical, \
        nearest_line_center_x, \
        nearest_line_bottom_y, \
        nearest_line_is_parallel
    global turn_started, TURN_DIRECTION
    global approach_active, approach_last_step_time, approach_steps_done
    global turn_parallel_count, turn_frame_count, turn_action_count, turn_start_time
    global turn_end_time, post_turn_look_end
    global corner_index, last_turn_completed_at

    line_center_x = -1
    line_center_frame = 0
    line_lost_time = 0
    line_history = []
    last_valid_center = 320
    line_is_vertical = False
    nearest_line_center_x = -1
    nearest_line_bottom_y = -1
    nearest_line_is_parallel = False
    turn_parallel_count = 0
    turn_frame_count = 0
    turn_action_count = 0
    turn_start_time = 0.0
    turn_end_time = 0
    post_turn_look_end = 0
    turn_started = False
    TURN_DIRECTION = 0
    approach_active = False
    approach_last_step_time = 0
    approach_steps_done = 0
    corner_index = 0
    last_turn_completed_at = 0.0


def init():
    global enter
    print("RedLinePatrol Init")
    load_config()
    initMove()
    enter = True


def start():
    global running
    running = True
    print("RedLinePatrol Start")


def stop():
    global running
    running = False
    reset()
    print("RedLinePatrol Stop")


def exit():
    global enter, running
    enter = False
    running = False
    reset()
    AGC.runActionGroup("stand_slow")
    print("RedLinePatrol Exit")


# ============ 运动控制线程 ============
def move():
    global line_center_x, turn_started

    turning = False
    turn_direction = 0
    turn_frame = 0
    last_turn_time = 0

    while True:
        if enter and running:
            # ===== 转弯或接近弯道时，完全暂停巡线控制 =====
            if turn_started or approach_active:  # ← 改这里
                time.sleep(0.01)
                continue
            # ==============================================

            if line_center_x != -1:
                offset = line_center_x - img_centerx
                now = time.time()
                fresh_frames = line_center_frame - turn_frame

                if (
                    abs(offset) <= TURN_STOP_THRESHOLD
                    and fresh_frames >= TURN_CONFIRM_FRAMES
                ):
                    turning = False
                    turn_direction = 0

                if turning:
                    if (
                        now - last_turn_time < TURN_ACTION_COOLDOWN
                        or fresh_frames < TURN_CONFIRM_FRAMES
                    ):
                        time.sleep(0.01)
                    else:
                        turning = False
                        turn_frame = line_center_frame
                elif fresh_frames < TURN_CONFIRM_FRAMES:
                    time.sleep(0.01)
                elif abs(offset) <= TURN_THRESHOLD:
                    AGC.runActionGroup(FORWARD_ACTION, times=1)
                    turn_direction = 0
                elif line_is_vertical:
                    if offset > 0:
                        AGC.runActionGroup(LATERAL_RIGHT_ACTION, times=1)
                    else:
                        AGC.runActionGroup(LATERAL_LEFT_ACTION, times=1)
                    turn_direction = 0
                elif offset > TURN_THRESHOLD:
                    if turn_direction != -1:
                        AGC.runActionGroup(TURN_RIGHT_ACTION)
                        turning = True
                        turn_direction = 1
                        turn_frame = line_center_frame
                        last_turn_time = time.time()
                    else:
                        AGC.runActionGroup(FORWARD_ACTION, times=1)
                        turn_direction = 0
                elif offset < -TURN_THRESHOLD:
                    if turn_direction != 1:
                        AGC.runActionGroup(TURN_LEFT_ACTION)
                        turning = True
                        turn_direction = -1
                        turn_frame = line_center_frame
                        last_turn_time = time.time()
                    else:
                        AGC.runActionGroup(FORWARD_ACTION, times=1)
                        turn_direction = 0
                else:
                    time.sleep(0.01)
            else:
                # 直行期间丢线：不做动作等待（避免在赛道尽头因转弯调整而原地打转）。
                # 真正转直角弯时的丢线由 turn_started 转弯分支负责继续转。
                turning = False
                turn_direction = 0
                time.sleep(0.01)
        else:
            turning = False
            turn_direction = 0
            time.sleep(0.1)


if HARDWARE_AVAILABLE:
    th = threading.Thread(target=move)
    th.daemon = True
    th.start()
else:
    th = None


# ============ ROI 配置 ============
# (y_start, y_end, x_start, x_end, weight)
roi = [
    (350, 380, 0, 640, 0.1),  # 远处范围缩小
    (340, 380, 0, 640, 0.3),  # 中部
    (400, 440, 0, 640, 0.4),  # 近处上（脚下短直道）
    (440, 480, 0, 640, 0.6),  # 近处下（贴近机器人）
]

roi_h1 = roi[0][0]
roi_h2 = roi[1][0] - roi[0][0]
roi_h3 = roi[2][0] - roi[1][0]
roi_h4 = roi[3][0] - roi[2][0]
roi_h_list = [roi_h1, roi_h2, roi_h3, roi_h4]

size = (640, 480)
img_w, img_h = None, None


def getAreaMaxContour(contours):
    """找出面积最大的轮廓"""
    contour_area_max = 0
    area_max_contour = None

    for c in contours:
        area = math.fabs(cv2.contourArea(c))
        if area > contour_area_max:
            contour_area_max = area
            if area > MIN_CONTOUR_AREA:
                area_max_contour = c

    return area_max_contour, contour_area_max


def detect_red_line(roi_img):
    """
    在 ROI 图像中检测红色线。

    方法：转 HSV → 双范围阈值（红色横跨 0°）→ 合并 → 形态学去噪
    黑线饱和度/明度低，白线饱和度低，其他颜色色相不同，都不会被误判。
    """
    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)

    # 红色需要两段范围（色相在 0 附近环绕）
    lower1 = np.array([RED_H_LOW1, RED_S_LOW, RED_V_LOW])
    upper1 = np.array([RED_H_HIGH1, RED_S_HIGH, RED_V_HIGH])
    lower2 = np.array([RED_H_LOW2, RED_S_LOW, RED_V_LOW])
    upper2 = np.array([RED_H_HIGH2, RED_S_HIGH, RED_V_HIGH])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    # 形态学处理去噪
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
    )
    mask = cv2.erode(mask, kernel)
    mask = cv2.dilate(mask, kernel)

    # 裁剪左右边缘，减少边缘干扰
    mask[:, 0:ROI_LEFT_CROP] = 0
    mask[:, (640 - ROI_RIGHT_CROP) : 640] = 0

    return mask


def run(img):
    """处理每一帧图像"""
    global \
        line_center_x, \
        line_center_frame, \
        line_lost_time, \
        line_history, \
        last_valid_center, \
        nearest_line_center_x, \
        nearest_line_bottom_y, \
        nearest_line_is_parallel
    global line_is_vertical
    global img_w, img_h
    global TURN_DIRECTION, turn_started
    global approach_active, approach_last_step_time, approach_steps_done
    global turn_parallel_count, turn_frame_count, turn_action_count, turn_start_time
    global turn_end_time, post_turn_look_end
    global corner_index, last_turn_completed_at

    display_image = img.copy()
    img_h, img_w = img.shape[:2]
    line_center_frame += 1

    if not enter:
        return display_image

    frame_resize = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)
    frame_gb = cv2.GaussianBlur(frame_resize, (3, 3), 3)

    centroid_x_sum = 0
    weight_sum = 0
    n = 0
    # 每帧重新选择"最近红线"作为转弯完成判定对象
    nearest_info = None
    near_upper_x = None  # 近处上段（脚下短直道）的线中心 x
    near_lower_x = None  # 近处下段的线中心 x

    for r in roi:
        roi_h = roi_h_list[n]
        n += 1

        blobs = frame_gb[r[0] : r[1], r[2] : r[3]]
        mask = detect_red_line(blobs)
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)[-2]

        if len(cnts) == 0:
            continue

        valid_cnts = [
            c for c in cnts if math.fabs(cv2.contourArea(c)) >= MIN_CONTOUR_AREA
        ]
        if len(valid_cnts) == 0:
            continue

        contour_info = []
        for c in valid_cnts:
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cx_mapped = int(Misc.map(cx, 0, size[0], 0, img_w)) if img_w else cx
                # cy 是 ROI 内局部坐标，必须先加 ROI 顶边 r[0] 才是整帧坐标。
                cy_mapped = (
                    int(Misc.map(r[0] + cy, 0, size[1], 0, img_h))
                    if img_h
                    else r[0] + cy
                )
                rect = cv2.minAreaRect(c)
                angle = rect[2] % 180
                rect_w, rect_h = rect[1]
                thickness = min(rect_w, rect_h)
                # 轮廓主轴与竖直方向的夹角（度），0 表示接近竖直
                mu20 = M["mu20"]
                mu02 = M["mu02"]
                mu11 = M["mu11"]
                if mu20 + mu02 > 0:
                    theta = 0.5 * math.atan2(2.0 * mu11, mu20 - mu02)
                    vert_dev = abs(90.0 - abs(math.degrees(theta)))
                else:
                    vert_dev = 90.0
                bx, by, bw, bh = cv2.boundingRect(c)
                bottom_y = by + bh
                bottom_y_mapped = int(Misc.map(r[0] + bottom_y, 0, size[1], 0, img_h))
                contour_info.append(
                    {
                        "contour": c,
                        "cx": cx_mapped,
                        "cy": cy_mapped,
                        "angle": angle,
                        "area": math.fabs(cv2.contourArea(c)),
                        "bw": bw,
                        "bh": bh,
                        "thickness": thickness,
                        "vert_dev": vert_dev,
                        "bottom_y": bottom_y_mapped,
                    }
                )

        if len(contour_info) == 0:
            continue

        # 转弯判定只用"线状且靠近画面底部"的轮廓：
        # 排除直角弯的宽角块，也排除远处红线。
        line_like = [
            info
            for info in contour_info
            if info["thickness"] <= LINE_THICKNESS_MAX
            and info["bottom_y"] >= NEAR_LINE_MIN_Y
        ]
        if line_like:
            roi_nearest_info = max(line_like, key=lambda info: info["bottom_y"])
            if nearest_info is None or roi_nearest_info["bottom_y"] > nearest_info["bottom_y"]:
                nearest_info = roi_nearest_info

        if last_valid_center > 0 and len(contour_info) > 1:
            for info in contour_info:
                dist = math.sqrt(
                    (info["cx"] - last_valid_center) ** 2
                    + (info["cy"] - last_center_y) ** 2
                )
                info["dist"] = dist
            contour_info.sort(key=lambda x: x["dist"])
            best_info = contour_info[0]
            cnt_large = best_info["contour"]
        else:
            contour_info.sort(key=lambda x: x["area"], reverse=True)
            cnt_large = contour_info[0]["contour"]

        if cnt_large is not None:
            rect = cv2.minAreaRect(cnt_large)
            box = np.intp(cv2.boxPoints(rect))

            rect_long = max(rect[1][0], rect[1][1])

            # 获取轮廓底部 Y 坐标（车头方向）。
            # boundingRect 的 y、h 都是 ROI 内局部量；原代码漏加 y，
            # 会把位于 ROI 底部的细横线错误算到 ROI 顶部附近，造成漏检/晚检。
            _, by, _, bh = cv2.boundingRect(cnt_large)
            corner_bottom_y = r[0] + by + bh
            corner_bottom_y = int(Misc.map(corner_bottom_y, 0, size[1], 0, img_h))

            # 判断：长边超过阈值 AND 弯道靠近画面底部。
            # 转弯期间 turn_started=True，关闭新的直角弯触发，
            # 避免当前转弯过程再次被识别成另一个弯道。
            post_turn_look = time.time() < post_turn_look_end
            if (
                not turn_started
                and (
                    post_turn_look
                    or time.time() - turn_end_time > TURN_REARM_COOLDOWN
                )
                and rect_long > NORMAL_LINE_WIDTH * CROSS_WIDTH_RATIO
                and corner_bottom_y
                > (POST_TURN_CORNER_Y if post_turn_look else CORNER_TRIGGER_Y)
            ):
                if not approach_active:  # 防止重复触发
                    approach_active = True
                    approach_last_step_time = 0.0
                    approach_steps_done = 0
                    print(
                        f"🔴 弯道已靠近车头 (y={corner_bottom_y})，直行 {APPROACH_STEPS} 步"
                    )

            _, _, bw, bh = cv2.boundingRect(cnt_large)
            line_is_vertical = bw > 0 and bh / bw >= VERTICAL_LINE_RATIO

            for i in range(4):
                # boxPoints 给出的是当前 ROI 内坐标，直接加当前 ROI 的顶边。
                # 原公式假定 ROI 连续且等距；当前 ROI 有重叠和空档，会把框画错位置。
                box[i, 1] = box[i, 1] + r[0]
                box[i, 1] = int(Misc.map(box[i, 1], 0, size[1], 0, img_h))
            for i in range(4):
                box[i, 0] = int(Misc.map(box[i, 0], 0, size[0], 0, img_w))

            cv2.drawContours(display_image, [box], -1, (0, 0, 255), 2)

            pt1_x, pt1_y = box[0, 0], box[0, 1]
            pt3_x, pt3_y = box[2, 0], box[2, 1]
            center_x = (pt1_x + pt3_x) / 2
            center_y = (pt1_y + pt3_y) / 2
            if n == 3:
                near_upper_x = center_x
            elif n == 4:
                near_lower_x = center_x
            cv2.circle(
                display_image, (int(center_x), int(center_y)), 5, (0, 0, 255), -1
            )

            centroid_x_sum += center_x * r[4]
            weight_sum += r[4]

    # 只保留当前画面中最靠近机器人的红线信息。远处红线不会参与转弯完成判定。
    if nearest_info is not None:
        nearest_line_center_x = int(nearest_info["cx"])
        nearest_line_bottom_y = int(nearest_info["bottom_y"])
        nearest_bw = max(int(nearest_info["bw"]), 1)
        # 转正判定：近处上/下两段线中心竖直对齐（两段都在脚下这条短直道上，
        # 不会被远处那条直道带偏）；或近处线高宽比/夹角（放宽过）。
        near_slant_ok = (
            near_upper_x is not None
            and near_lower_x is not None
            and abs(near_lower_x - near_upper_x) <= LINE_PARALLEL_SLANT_TOL
        )
        # 近处 ROI 只有约 40px 高，红线靠近镜头后会变粗；即使已经竖直，
        # bh / bw 也可能只有 0.8～1.0。原来的 1.2 会让“肉眼已转正”仍判 False。
        shape_angle_parallel = (
            nearest_info["bh"] >= TURN_PARALLEL_MIN_HEIGHT
            and (nearest_info["bh"] / nearest_bw) >= TURN_PARALLEL_RATIO
            and nearest_info["vert_dev"] <= TURN_PARALLEL_ANGLE_TOL
        )
        nearest_line_is_parallel = shape_angle_parallel or near_slant_ok
        cv2.circle(
            display_image,
            (nearest_line_center_x, min(nearest_line_bottom_y, img_h - 5)),
            8,
            (0, 255, 0),
            -1,
        )
    else:
        nearest_line_center_x = -1
        nearest_line_bottom_y = -1
        nearest_line_is_parallel = False

        # ========== 直角弯处理（视觉闭环，无时间硬等） ==========
        # ========== 直角弯处理（步数控制 + 视觉闭环） ==========
    if approach_active:
        # 直行期间用当前帧的加权中心刷新方向信号：
        # 进弯前线中心在画面中央，用旧值判断左右容易转反；
        # 越靠近弯道，拐向一侧的线越明显，中心会明确偏向该侧。
        if weight_sum != 0:
            last_valid_center = int(centroid_x_sum / weight_sum)

        now = time.time()
        # 每步直行之间留时间间隔，保证机器人真的走出余量，
        # 而不是在几帧内瞬间走完步数就立刻转弯。
        if (
            approach_steps_done < APPROACH_STEPS
            and now - approach_last_step_time >= APPROACH_STEP_INTERVAL
            and running
        ):
            AGC.runActionGroup(FORWARD_ACTION, times=1)
            line_center_x = last_valid_center
            line_lost_time = 0
            print(f"🚶 直行中... 第 {approach_steps_done + 1}/{APPROACH_STEPS} 步")
            # 从动作实际完成时开始计间隔，避免首次 go_forward 启动动作较长时
            # 下一帧立刻再走一步，造成弯前连续前冲。
            approach_last_step_time = time.time()
            approach_steps_done += 1

        # 步数走满且最后一步已经走完，才开始转弯
        if (
            approach_steps_done >= APPROACH_STEPS
            and now - approach_last_step_time >= APPROACH_STEP_INTERVAL
            and running
        ):
            turn_start_time = time.time()
            turn_action_count = 1
            if last_valid_center < img_centerx:
                TURN_DIRECTION = -1
                AGC.runActionGroup(TURN_LEFT_ACTION)
                print(f"🔄 开始左转 (last_valid_center={last_valid_center})")
            else:
                TURN_DIRECTION = 1
                AGC.runActionGroup(TURN_RIGHT_ACTION)
                print(f"🔄 开始右转 (last_valid_center={last_valid_center})")
            turn_started = True
            turn_parallel_count = 0
            turn_frame_count = 0
            approach_active = False
            print("   转弯中暂时关闭新的直角弯检测")

        cv2.circle(display_image, (line_center_x, img_h - 30), 10, (0, 200, 0), -1)

    elif turn_started and running:
        # 转弯完成只看"最近红线"：先大致平行，再用横移把它调到视野中央。
        turn_frame_count += 1

        turn_elapsed = time.time() - turn_start_time if turn_start_time > 0 else 0.0
        if turn_action_count >= TURN_MAX_ACTIONS or turn_elapsed >= TURN_MAX_SECONDS:
            # 原代码没有任何上限，只要一次转正判定失败就会永远沿同一方向转。
            AGC.runActionGroup("stand")
            print(
                f"⚠️ 转弯保护停止：actions={turn_action_count}, "
                f"elapsed={turn_elapsed:.1f}s，交回普通巡线重新判断"
            )
            turn_started = False
            TURN_DIRECTION = 0
            turn_parallel_count = 0
            turn_frame_count = 0
            turn_action_count = 0
            turn_start_time = 0.0
            turn_end_time = time.time()
            post_turn_look_end = time.time() + POST_TURN_LOOK_SECONDS
            # 弯确实到了、转弯动作也做满了，只是没转到"平行居中"——
            # 同样计一个已完成弯，保证 demoV4 终点判定的进度门限可用
            #（此前保护停止不计数，corner_index 停在低值导致永远进不了射门）。
            corner_index += 1
            last_turn_completed_at = time.time()

        elif nearest_info is not None:
            if turn_frame_count % 8 == 1:
                print(
                    f"🔄 转弯中 dir={TURN_DIRECTION} frame={turn_frame_count} "
                    f"nearest_bh={nearest_info['bh']} bw={nearest_info['bw']} "
                    f"dev={nearest_info['vert_dev']:.0f} parallel={nearest_line_is_parallel}"
                )

            if nearest_line_is_parallel:
                turn_parallel_count += 1
            else:
                turn_parallel_count = 0

            if 0 < turn_parallel_count < TURN_PARALLEL_CONFIRM_FRAMES:
                # 第一帧看见“已转正”后原地保持，等待下一帧确认。
                # 原代码此时又执行一次约 1 秒的完整转弯动作，必然容易转过头。
                print(
                    f"⏸️ 已疑似转正，保持不动等待确认 "
                    f"({turn_parallel_count}/{TURN_PARALLEL_CONFIRM_FRAMES})"
                )

            elif turn_parallel_count >= TURN_PARALLEL_CONFIRM_FRAMES:
                nearest_offset = nearest_line_center_x - img_centerx
                line_center_x = nearest_line_center_x
                last_valid_center = nearest_line_center_x
                line_lost_time = 0

                if abs(nearest_offset) > TURN_CENTER_TOLERANCE:
                    if nearest_offset > 0:
                        AGC.runActionGroup(LATERAL_RIGHT_ACTION, times=1)
                        print(f"↔️ 最近红线已大致平行，向右平移 (offset={nearest_offset})")
                    else:
                        AGC.runActionGroup(LATERAL_LEFT_ACTION, times=1)
                        print(f"↔️ 最近红线已大致平行，向左平移 (offset={nearest_offset})")
                    turn_parallel_count = 0
                else:
                    # 不额外向前走；先退出转弯状态，让普通巡线根据新帧接管。
                    turn_started = False
                    TURN_DIRECTION = 0
                    turn_parallel_count = 0
                    turn_frame_count = 0
                    turn_action_count = 0
                    turn_start_time = 0.0
                    turn_end_time = time.time()
                    post_turn_look_end = time.time() + POST_TURN_LOOK_SECONDS
                    corner_index += 1
                    last_turn_completed_at = time.time()
                    print(f"✅ 转弯完成：最近红线大致平行且已居中 (offset={nearest_offset})")
            else:
                if TURN_DIRECTION < 0:
                    AGC.runActionGroup(TURN_LEFT_ACTION)
                else:
                    AGC.runActionGroup(TURN_RIGHT_ACTION)
                turn_action_count += 1
        else:
            # 最近红线暂时丢失时不能认为转弯完成，继续按锁定方向寻找。
            turn_parallel_count = 0
            if turn_frame_count % 8 == 1:
                print(f"🔄 转弯中 dir={TURN_DIRECTION} frame={turn_frame_count} 最近红线不可用，继续转")
            if TURN_DIRECTION < 0:
                AGC.runActionGroup(TURN_LEFT_ACTION)
            else:
                AGC.runActionGroup(TURN_RIGHT_ACTION)
            turn_action_count += 1

    elif weight_sum != 0:
        current_center = int(centroid_x_sum / weight_sum)
        if running:
            jump = abs(current_center - last_valid_center)
            if jump > MAX_CENTER_JUMP and len(line_history) >= 3:
                line_center_x = last_valid_center
            else:
                line_center_x = current_center
                last_valid_center = current_center
            line_lost_time = 0
            line_history.append(current_center)
            if len(line_history) > LINE_MEMORY_FRAMES:
                line_history.pop(0)
        cv2.circle(display_image, (line_center_x, img_h - 30), 10, (0, 255, 255), -1)

    else:
        if running:
            if line_lost_time == 0:
                line_lost_time = time.time()
            lost_duration = time.time() - line_lost_time
            if lost_duration < LINE_LOST_TIMEOUT:
                line_center_x = last_valid_center
            else:
                line_center_x = -1

    return display_image


# ============ 独立运行入口 ============
if __name__ == "__main__":
    # 加载相机标定参数
    param_data = np.load(calibration_param_path + ".npz")
    mtx = param_data["mtx_array"]
    dist = param_data["dist_array"]
    newcameramtx, _ = cv2.getOptimalNewCameraMatrix(
        mtx, dist, (640, 480), 0, (640, 480)
    )
    mapx, mapy = cv2.initUndistortRectifyMap(
        mtx, dist, None, newcameramtx, (640, 480), 5
    )

    init()
    start()

    open_once = yaml_handle.get_yaml_data("/boot/camera_setting.yaml")["open_once"]
    if open_once:
        my_camera = cv2.VideoCapture(
            "http://127.0.0.1:8080/?action=stream?dummy=param.mjpg"
        )
    else:
        my_camera = Camera.Camera()
        my_camera.camera_open()

    AGC.runActionGroup("stand")
    print("红色巡线已启动，按 Ctrl+C 退出")
    print(
        f"红色 HSV 范围: H=[{RED_H_LOW1}-{RED_H_HIGH1}]|[{RED_H_LOW2}-{RED_H_HIGH2}], "
        f"S=[{RED_S_LOW}-{RED_S_HIGH}], V=[{RED_V_LOW}-{RED_V_HIGH}]"
    )

    try:
        while True:
            ret, img = my_camera.read()
            if ret:
                frame = img.copy()
                frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
                display = run(frame)
                if SHOW_DISPLAY:
                    cv2.imshow("RedLinePatrol", display)
                    key = cv2.waitKey(1)
                    if key == 27:
                        break
                else:
                    time.sleep(0.01)
            else:
                time.sleep(0.01)
    except KeyboardInterrupt:
        pass

    my_camera.camera_close()
    if SHOW_DISPLAY:
        cv2.destroyAllWindows()