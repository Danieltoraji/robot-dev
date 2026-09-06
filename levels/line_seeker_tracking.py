# -*- coding: utf-8 -*-
"""红色路径寻线 + 循迹模块（不依赖 AprilTag 定位）。

两阶段流程：
  1. 寻线（seek_line）：头部扫描 + 身体旋转找到红色路径线并对准
  2. 循迹（track_line）：沿线行进，处理直道 / 直角弯 / 丢线

动作组来自 TonyPi 框架，动作名和标定值参考 levels/goodluck.py：
  go_forward_one_step ≈ 2cm   back_one_step ≈ 3.2cm
  left_move ≈ 1.9cm           right_move ≈ 2.2cm
  turn_left ≈ 22°             turn_right ≈ 25.7°
"""

import cv2
import numpy as np
import subprocess
import time

# =====================================================================
# 路径设置：让脚本能从任意目录 import 到 vision 包内的模块
# =====================================================================
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_VISION_DIR = str(_PROJECT_ROOT / "vision")
if _VISION_DIR not in sys.path:
    sys.path.insert(0, _VISION_DIR)

TONYPI_DIR = Path('/home/pi/TonyPi')
HIWONDER_SDK_DIR = TONYPI_DIR / 'HiwonderSDK'
ACTION_GROUP_DIR = TONYPI_DIR / 'ActionGroups'
for extra_path in (TONYPI_DIR, HIWONDER_SDK_DIR):
    extra_path_str = str(extra_path)
    if extra_path_str not in sys.path:
        sys.path.insert(0, extra_path_str)

from line_detector import LineDetector, LineResult, LineSegment

import hiwonder.ActionGroupControl as AGC

try:
    import hiwonder.ros_robot_controller_sdk as rrc
    from hiwonder.Controller import Controller
    _board = rrc.Board()
    _ctl = Controller(_board)
except Exception:
    _ctl = None


# =====================================================================
# 动作执行
# =====================================================================

def run_action(name, times=1, with_stand=True):
    """执行动作组（同步阻塞，一个动作跑完才返回）"""
    AGC.runActionGroup(
        name,
        times=times,
        with_stand=with_stand,
        path=str(ACTION_GROUP_DIR) + '/',
    )


# =====================================================================
# 头部舵机控制
# =====================================================================

HEAD_CENTER = 1500
HEAD_RIGHT = 1050
HEAD_LEFT = 1950
HEAD_WIDE_RIGHT = 800
HEAD_WIDE_LEFT = 2200
HEAD_MOVE_TIME_MS = 500
HEAD_MOVE_TIME_MIN_MS = 100

_current_head_pulse = HEAD_CENTER


def set_head(pulse, move_time_ms=HEAD_MOVE_TIME_MS):
    """转动头部舵机并等待到位；目标与当前位置相同则跳过"""
    global _current_head_pulse
    if pulse == _current_head_pulse:
        return
    if _ctl is None:
        print("[WARN] 舵机未初始化，set_head 跳过")
        return
    delta = abs(pulse - _current_head_pulse)
    dynamic_time = max(HEAD_MOVE_TIME_MIN_MS, int(move_time_ms * delta / 900))
    _ctl.set_pwm_servo_pulse(2, pulse, dynamic_time)
    time.sleep(dynamic_time / 1000.0 + 0.2)
    _current_head_pulse = pulse


# =====================================================================
# 拍照
# =====================================================================

CAMERA_WIDTH = 2592
CAMERA_HEIGHT = 1944


def capture_frame():
    """拍照并读取为内存帧（BGR ndarray）；失败返回 None"""
    timestamp = int(time.time())
    filename = f"/home/pi/Pictures/photo_tjz_tracking_{timestamp}.jpg"
    cmd = (f"fswebcam -r {CAMERA_WIDTH}x{CAMERA_HEIGHT} "
           f"--no-banner -S 3 {filename}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"拍照失败: {result.stderr}")
        return None
    frame = cv2.imread(filename)
    if frame is None:
        print("读图失败：", filename)
    return frame


# =====================================================================
# 红色路径 HSV 范围 + LineDetector 配置
# =====================================================================

red_low = (0, 0, 50)
red_high = (17, 255, 255)
red_low_2 = (149, 0, 50)
red_high_2 = (180, 255, 255)
# 红色路径 HSV 范围，有两段，(H,S,V)，0~10 偏紫红，170~180 偏橙红
# 数值已经过 HSV_threshold_test_program.py 调试

detector = LineDetector(
    hsv_ranges=[(red_low, red_high), (red_low_2, red_high_2)],
    line_color="light",
    min_area=80,
    lookahead_ratio=0.5,
    straightness_thresh=0.85,
    work_width=640,
)


# =====================================================================
# 循迹决策阈值（工作分辨率 640px 宽下，需实机标定调整）
# =====================================================================

T_CENTER = 12.0        # |lookahead_x| ≤ 此值 → 居中，可前进
T_BIG = 35.0           # |lookahead_x| > 此值 → 需转向修朝向
T_HEADING_SEEK = 10.0  # 寻线对齐阶段 heading_deg 容忍阈值（度）

SEEK_MAX_BODY_TURNS = 8    # 寻线阶段身体旋转最大次数
TRACK_MAX_BACKS = 3        # 循迹丢线时连续后退最大次数，超限回寻线
CORNER_MAX_TURNS = 6       # 单个直角弯最大转向次数（安全限）
MAX_TOTAL_STEPS = 500     # 总步数安全上限（防死循环）
ALIGN_MAX_STEPS = 20      # 对齐阶段最大步数（防死循环）

# 头部扫描顺序：(脉宽, 对应身体转角修正的近似角度，左正右负)
HEAD_SCAN_SEQUENCE = [
    (HEAD_CENTER, 0.0),
    (HEAD_LEFT, 40.5),
    (HEAD_RIGHT, -40.5),
    (HEAD_WIDE_LEFT, 63.0),
    (HEAD_WIDE_RIGHT, -63.0),
]

# 动作标定值（来自 levels/goodluck.py 实机标定）
TURN_LEFT_DEG = 22.0
TURN_RIGHT_DEG = 25.7


# =====================================================================
# Phase 1: 寻线阶段
# =====================================================================

def _scan_head(detector):
    """头部扫描：按顺序转头拍照检测，找到线即返回

    返回 (found, head_pulse, head_angle_deg, result)
    """
    for pulse, angle_deg in HEAD_SCAN_SEQUENCE:
        set_head(pulse)
        frame = capture_frame()
        if frame is None:
            continue
        result = detector.detect(frame)
        if result.exists:
            print(f"[寻线] 头部脉宽 {pulse}（{angle_deg:+.1f}°）发现线："
                  f"orientation={result.primary.orientation} "
                  f"lookahead_x={result.primary.lookahead_x:.1f}")
            return True, pulse, angle_deg, result
    return False, HEAD_CENTER, 0.0, None


def _turn_body_to_head_angle(head_angle_deg):
    """根据头部偏转角度执行身体转向，使身体正对线"""
    if abs(head_angle_deg) < 1.0:
        set_head(HEAD_CENTER)
        return

    if head_angle_deg > 0:
        # 线在左侧，身体左转
        turns = max(1, min(3, int(round(head_angle_deg / TURN_LEFT_DEG))))
        print(f"[寻线] 头部左偏 {head_angle_deg:.1f}°，身体左转 {turns} 次")
        set_head(HEAD_CENTER)
        run_action("turn_left", times=turns)
    else:
        # 线在右侧，身体右转
        turns = max(1, min(3, int(round(abs(head_angle_deg) / TURN_RIGHT_DEG))))
        print(f"[寻线] 头部右偏 {head_angle_deg:.1f}°，身体右转 {turns} 次")
        set_head(HEAD_CENTER)
        run_action("turn_right", times=turns)


def _align_to_line(detector):
    """对准线：循环拍照检测，根据 lookahead_x 和 heading_deg 调整

    返回 True 表示已对准，False 表示丢线
    """
    for step in range(ALIGN_MAX_STEPS):
        set_head(HEAD_CENTER)
        frame = capture_frame()
        if frame is None:
            print("[对齐] 拍照失败")
            return False
        result = detector.detect(frame)
        if not result.exists:
            print("[对齐] 丢线")
            return False

        p = result.primary
        print(f"[对齐] step={step} orientation={p.orientation} "
              f"lookahead_x={p.lookahead_x:.1f} heading={p.heading_deg:.1f}")

        if p.orientation == "corner":
            print("[对齐] 检测到拐角，直接进入循迹")
            return True

        if p.orientation == "cross":
            print("[对齐] 检测到横线，跨越")
            run_action("go_forward_one_step")
            continue

        # follow 线
        lx = p.lookahead_x
        if abs(lx) > T_BIG:
            if lx > 0:
                print(f"[对齐] lookahead_x={lx:.1f} > {T_BIG}，右转修朝向")
                run_action("turn_right")
            else:
                print(f"[对齐] lookahead_x={lx:.1f} < -{T_BIG}，左转修朝向")
                run_action("turn_left")
        elif abs(lx) > T_CENTER:
            if lx > 0:
                print(f"[对齐] lookahead_x={lx:.1f}，右移对齐")
                run_action("right_move")
            else:
                print(f"[对齐] lookahead_x={lx:.1f}，左移对齐")
                run_action("left_move")
        else:
            print(f"[对齐] lookahead_x={lx:.1f} ≤ {T_CENTER}，对准完成")
            return True

    print("[对齐] 超过最大步数，返回")
    return False


def seek_line(detector):
    """寻线主函数：头部扫描 + 身体旋转找到线并对准

    返回 True 表示寻线成功（已对准线，可进入循迹），False 表示未找到线
    """
    print("=" * 50)
    print("===== 开始寻线 =====")
    print("=" * 50)

    run_action("stand")
    set_head(HEAD_CENTER)

    # 第一轮：头部扫描
    found, head_pulse, head_angle, result = _scan_head(detector)

    if found:
        _turn_body_to_head_angle(head_angle)
        if _align_to_line(detector):
            print("===== 寻线成功，进入循迹 =====")
            return True
        print("[寻线] 对齐阶段丢线，进入身体扫描")

    # 身体扫描：旋转身体寻找线
    for body_turn in range(SEEK_MAX_BODY_TURNS):
        print(f"[寻线] 身体扫描第 {body_turn + 1}/{SEEK_MAX_BODY_TURNS} 次")
        # 交替左右转，扩大搜索范围
        if body_turn % 2 == 0:
            run_action("turn_right")
        else:
            run_action("turn_left")

        found, head_pulse, head_angle, result = _scan_head(detector)
        if found:
            _turn_body_to_head_angle(head_angle)
            if _align_to_line(detector):
                print("===== 寻线成功，进入循迹 =====")
                return True
            print(f"[寻线] 第 {body_turn + 1} 次身体扫描后仍丢线，继续")

    print("===== 寻线失败：未找到线 =====")
    set_head(HEAD_CENTER)
    return False


# =====================================================================
# Phase 2: 循迹阶段
# =====================================================================

def _handle_corner(detector, curvature):
    """直角弯处理：根据 curvature 方向连续转向，直到线变为 follow

    curvature < 0 → 左弯（turn_left）
    curvature > 0 → 右弯（turn_right）
    返回 True 表示弯已转过，False 表示异常（丢线或超限）
    """
    if curvature < 0:
        turn_action = "turn_left"
        reverse_action = "turn_right"
        direction = "左"
    else:
        turn_action = "turn_right"
        reverse_action = "turn_left"
        direction = "右"

    print(f"[转弯] {direction}弯（curvature={curvature:.2f}），开始转向")

    for turn_num in range(CORNER_MAX_TURNS):
        run_action(turn_action)
        print(f"[转弯] 第 {turn_num + 1}/{CORNER_MAX_TURNS} 次 {turn_action}")

        set_head(HEAD_CENTER)
        frame = capture_frame()
        if frame is None:
            print("[转弯] 拍照失败")
            return False
        result = detector.detect(frame)

        if not result.exists:
            # 丢线，可能转向过度，反向转一次尝试找回
            print("[转弯] 转向后丢线，反向转一次尝试找回")
            run_action(reverse_action)
            set_head(HEAD_CENTER)
            frame = capture_frame()
            if frame is not None:
                result = detector.detect(frame)
                if result.exists and result.primary.orientation == "follow":
                    print("[转弯] 反向转后找回线，弯已转过")
                    return True
            print("[转弯] 反向转后仍未找回线")
            return False

        p = result.primary
        print(f"[转弯] 检测到 orientation={p.orientation} "
              f"lookahead_x={p.lookahead_x:.1f}")

        if p.orientation == "follow":
            if abs(p.lookahead_x) > T_BIG:
                print(f"[转弯] follow 但 lookahead_x={p.lookahead_x:.1f} 较大，继续转向")
                continue
            print("[转弯] follow 且居中，弯已转过")
            return True

        if p.orientation == "corner":
            # 仍是拐角，检查方向是否一致
            if (curvature < 0 and p.curvature < 0) or (curvature > 0 and p.curvature > 0):
                print(f"[转弯] 仍是同方向拐角（curvature={p.curvature:.2f}），继续转")
                continue
            # 方向变了，按新方向处理
            print(f"[转弯] 拐角方向变化（curvature={p.curvature:.2f}），按新方向处理")
            if p.curvature < 0:
                turn_action = "turn_left"
                reverse_action = "turn_right"
                direction = "左"
            else:
                turn_action = "turn_right"
                reverse_action = "turn_left"
                direction = "右"
            curvature = p.curvature
            continue

        if p.orientation == "cross":
            print("[转弯] 遇到横线，跨越")
            run_action("go_forward_one_step")
            return True

    print(f"[转弯] 超过最大转向次数 {CORNER_MAX_TURNS}，返回异常")
    return False


def _handle_line_lost(detector):
    """丢线处理：后退找线，超限则回寻线

    返回 "track" 表示找回线继续循迹，"seek" 表示需要重新寻线
    """
    for back_num in range(TRACK_MAX_BACKS):
        print(f"[丢线] 后退 {back_num + 1}/{TRACK_MAX_BACKS} 次找线")
        run_action("back_one_step")

        set_head(HEAD_CENTER)
        frame = capture_frame()
        if frame is None:
            continue
        result = detector.detect(frame)
        if result.exists:
            print(f"[丢线] 后退后找回线：orientation={result.primary.orientation}")
            return "track"

    print("[丢线] 后退用尽仍丢线，回寻线阶段")
    return "seek"


def track_line(detector):
    """循迹主函数：沿线行进，处理直道 / 直角弯 / 丢线，无限循环

    返回 True 表示正常结束（达到 MAX_TOTAL_STEPS），False 表示异常
    """
    print("=" * 50)
    print("===== 开始循迹 =====")
    print("=" * 50)

    total_steps = 0

    while total_steps < MAX_TOTAL_STEPS:
        total_steps += 1

        set_head(HEAD_CENTER)
        frame = capture_frame()
        if frame is None:
            print(f"[循迹] step={total_steps} 拍照失败，跳过")
            continue

        result = detector.detect(frame)

        if not result.exists:
            print(f"[循迹] step={total_steps} 丢线")
            status = _handle_line_lost(detector)
            if status == "seek":
                if seek_line(detector):
                    continue
                else:
                    print("[循迹] 重新寻线失败，循迹结束")
                    return False
            continue

        p = result.primary
        print(f"[循迹] step={total_steps} orientation={p.orientation} "
              f"lookahead_x={p.lookahead_x:.1f} lateral_offset={p.lateral_offset:.1f} "
              f"heading={p.heading_deg:.1f} curvature={p.curvature:.4f}")

        if p.orientation == "corner":
            if not _handle_corner(detector, p.curvature):
                print("[循迹] 转弯异常，尝试继续循迹")
            continue

        if p.orientation == "cross":
            print("[循迹] 横线，跨越")
            run_action("go_forward_one_step")
            continue

        # follow 线
        lx = p.lookahead_x
        if abs(lx) > T_BIG:
            if lx > 0:
                print(f"[循迹] lookahead_x={lx:.1f} > {T_BIG}，右转修朝向")
                run_action("turn_right")
            else:
                print(f"[循迹] lookahead_x={lx:.1f} < -{T_BIG}，左转修朝向")
                run_action("turn_left")
        elif abs(lx) > T_CENTER:
            if lx > 0:
                print(f"[循迹] lookahead_x={lx:.1f}，右移对齐")
                run_action("right_move")
            else:
                print(f"[循迹] lookahead_x={lx:.1f}，左移对齐")
                run_action("left_move")
        else:
            print(f"[循迹] 居中，前进（lookahead_x={lx:.1f}）")
            run_action("go_forward_one_step")

    print(f"[循迹] 达到最大步数 {MAX_TOTAL_STEPS}，循迹结束")
    return True


# =====================================================================
# Phase 3: 入口函数
# =====================================================================

def run_line_tracking():
    """寻线 + 循迹完整流程入口

    流程：stand → seek_line → track_line
    无限循迹直到手动停止或达到 MAX_TOTAL_STEPS 安全上限。
    """
    print("=" * 50)
    print("===== 启动寻线 + 循迹 =====")
    print("=" * 50)

    run_action("stand")
    set_head(HEAD_CENTER)
    _ctl.set_pwm_servo_pulse(1, 1050, 500)

    # Phase 1: 寻线
    if not seek_line(detector):
        print("寻线失败，程序终止。")
        return False

    # Phase 2: 循迹
    track_line(detector)

    print("===== 循迹结束 =====")
    return True


if __name__ == "__main__":
    run_line_tracking()
