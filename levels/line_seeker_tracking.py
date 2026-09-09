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


def _turn_by_angle(angle_deg):
    """按角度闭环转向：正=左转（逆时针），负=右转（顺时针）。

    用「大步 turn_left/right + 小步 turn_left/right_small_step」组合逼近目标角度，
    取代旧的「按标定值固定转 N 次」开环转弯。返回近似实际转角（度，左正右负）。
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
    # 大步覆盖主体角度
    big_count = int(remaining // big_deg)
    remaining -= big_count * big_deg
    # 余量用小步逼近（不足半小步则舍去，避免过度补转）
    small_count = int(round(remaining / small_deg))

    if big_count > 0:
        run_action(big_action, times=big_count)
        actual += big_count * big_deg
    if small_count > 0:
        run_action(small_action, times=small_count)
        actual += small_count * small_deg

    print(f"[转向] 目标 {angle_deg:+.1f}° → {big_action}×{big_count} + "
          f"{small_action}×{small_count} ≈ {sign * actual:+.1f}°")
    return sign * actual


def _correct_heading(heading_deg):
    """按观测 heading 闭环修朝向：heading>0=线在前方偏右 → 身体右转（负角）。

    误差小用小步精修，误差大用大步粗修（由 _turn_by_angle 自适应）。
    返回实际转向角（左正右负）。
    """
    return _turn_by_angle(-heading_deg)


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

# 寻线参数
SEEK_BODY_TURNS = 14       # 身体单向扫描最大次数（turn_right×14 ≈ 360° 全周）
SEEK_MAX_PROBES = 3        # 全周扫描失败后的平移试探轮数
SEEK_PROBE_FORWARD = 5     # 每轮试探前进步数
SEEK_PROBE_LEFT = 5        # 每轮试探左移步数
SEEK_PROBE_RIGHT = 10      # 每轮试探右移步数（覆盖左移回程+更远）
TRACK_MAX_BACKS = 3        # 循迹丢线时连续后退最大次数，超限回寻线
MAX_TOTAL_STEPS = 500     # 总步数安全上限（防死循环）
ALIGN_MAX_STEPS = 20      # 对齐阶段最大步数（防死循环）
ALIGN_HEADING_TOL = 15.0  # 对齐阶段 heading_deg 容忍阈值（度）

# 转弯闭环参数（闭环决定转弯程度：按每次观测到的 heading / corner_deg 决策，
# 不再按标定值固定转几次，也不再假设弯角为 90°）
LOOKAHEAD_CLAMP = 200.0        # lookahead_x 绝对值钳制上限（防止多项式外推爆炸）
HEADING_BIG = 25.0             # |heading_deg| ≥ 此值 → 大步粗修朝向（turn_left/right）
HEADING_SMALL = 8.0            # 此值 ≤ |heading_deg| < HEADING_BIG → 小步精修朝向
FOLLOW_HEADING_MIN = 15.0      # follow 循迹：|heading| > 此值优先修朝向（线倾斜时 lookahead 不可靠）
FOLLOW_HEADING_MAX = 35.0      # follow 循迹：|heading| > 此值疑为拐角误判/抖动，忽略朝向只看横向
CORNER_EXIT_HEADING = 12.0     # 过弯出口：|heading| ≤ 此值视为已对准新方向
CORNER_INITIAL_TURN_RATIO = 0.85  # 初始按观测拐角角 × 此比例转（略欠转，防过冲）
CORNER_MAX_CUMULATIVE_DEG = 150.0  # 单弯累计最大转角（安全上限，适应任意弯角）
CORNER_SEEK_FORWARD_MAX = 4    # 出口确认丢线后前进找线最大次数
# 循迹卡死检测参数（防止 cross+丢线 循环）
STUCK_MAX = 6                  # 循迹中连续 cross/丢线 超过此值则回寻线

# 终点判定参数（线末端接近 → 前进到尽头停止）
LINE_END_FAR_RY = 100.0        # follow 线远端 ry 小于此值（约 7~8cm）视为线末端接近
LINE_END_FORWARD_MAX = 8       # 终点前进最大步数（丢线即停，防止越过终点）
LINE_END_BLIND_MAX = 2         # 丢线后再盲目前进步数（抵达线末端，直线短距前进安全）

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
# 小步转向角度 = 正常转向的一半（用于闭环精修朝向）
TURN_LEFT_SMALL_DEG = TURN_LEFT_DEG / 2    # ≈ 11.0°
TURN_RIGHT_SMALL_DEG = TURN_RIGHT_DEG / 2  # ≈ 12.85°

# 直角弯「前进接近」阶段参数
# 检测到 corner 时拐角可能仍在视野远端（D_FAR=35cm 处），原地转弯会转错位置。
# 先前进到拐角点附近（elbow_px.y 足够大）再定点转弯。
CORNER_APPROACH_MAX = 20     # 接近阶段最大前进次数（每步约 2cm，最多约 40cm）
CORNER_APPROACH_Y = 200.0    # elbow_px.y ≥ 此值视为拐角已到脚下附近（像素，越大越近）


# =====================================================================
# Phase 1: 寻线阶段
# =====================================================================

def _scan_head(detector, prefer_side=None):
    """头部扫描：按顺序转头拍照检测，找到线即返回

    prefer_side: "left"/"right"/None — 优先扫指定侧（过弯后线大概率在转向侧）
    返回 (found, head_pulse, head_angle_deg, result)
    """
    sequence = list(HEAD_SCAN_SEQUENCE)
    if prefer_side == "left":
        sequence.sort(key=lambda t: -t[1])      # 左侧（大角度）优先
    elif prefer_side == "right":
        sequence.sort(key=lambda t: t[1])       # 右侧（小角度）优先
    for pulse, angle_deg in sequence:
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


def _turn_body_to_head_angle(head_angle_deg, max_turns=3):
    """根据头部偏转角度执行身体转向，使身体正对线

    max_turns 限制最大转向次数（转弯流程中避免过度转向）
    返回实际转向角度（左正右负），未转返回 0.0
    """
    if abs(head_angle_deg) < 1.0:
        set_head(HEAD_CENTER)
        return 0.0

    if head_angle_deg > 0:
        # 线在左侧，身体左转
        turns = max(1, min(max_turns, int(round(head_angle_deg / TURN_LEFT_DEG))))
        print(f"[寻线] 头部左偏 {head_angle_deg:.1f}°，身体左转 {turns} 次")
        set_head(HEAD_CENTER)
        run_action("turn_left", times=turns)
        return turns * TURN_LEFT_DEG
    else:
        # 线在右侧，身体右转
        turns = max(1, min(max_turns, int(round(abs(head_angle_deg) / TURN_RIGHT_DEG))))
        print(f"[寻线] 头部右偏 {head_angle_deg:.1f}°，身体右转 {turns} 次")
        set_head(HEAD_CENTER)
        run_action("turn_right", times=turns)
        return -turns * TURN_RIGHT_DEG


def _align_to_line(detector, context="seek"):
    """统一对准线：优先级 corner→完成；cross→跨越；follow 依次处理

    follow 处理优先级：
      1. 大幅欠转（|heading| > 25°）→ 先转向拉正线。线严重倾斜时最近点
         与远前视点横向方向相反（如 lateral=+56 而 lookahead=-58），
         横移会左右反复振荡，必须先转向把线拉回近似垂直。
      2. 侧边出线（|lateral_offset| > T_LAT）→ 横移
      3. |lookahead_x| > T_CENTER → 横移
      4. 居中 → 完成

    context="seek"  — 寻线后对齐（corner 直接进入循迹）
    context="corner" — 过弯后对齐（先修朝向再修横向，heading 优先）
    返回 True 表示已对准，False 表示丢线/超限
    """
    tag = "对齐" if context == "seek" else "转弯后对准"
    for step in range(ALIGN_MAX_STEPS):
        set_head(HEAD_CENTER)
        frame = capture_frame()
        if frame is None:
            print(f"[{tag}] 拍照失败")
            return False
        result = detector.detect(frame)
        if not result.exists:
            print(f"[{tag}] 丢线")
            return False

        p = result.primary
        lx = max(-LOOKAHEAD_CLAMP, min(LOOKAHEAD_CLAMP, p.lookahead_x))
        lateral = p.lateral_offset
        print(f"[{tag}] step={step} orientation={p.orientation} "
              f"lookahead_x={lx:.1f} heading={p.heading_deg:.1f} "
              f"lateral={lateral:.1f}")

        if p.orientation == "corner":
            if context == "seek":
                print(f"[{tag}] 检测到拐角，直接进入循迹")
                return True
            # 过弯后仍见拐角：按其方向与观测拐角角继续补转（闭环，非固定一次）
            turn_sign = -1.0 if p.curvature > 0 else 1.0
            residual = p.corner_deg if p.corner_deg > 0 else 30.0
            direction = "右" if turn_sign < 0 else "左"
            print(f"[{tag}] 仍是拐角（{direction}），补转 {residual:.1f}°")
            _turn_by_angle(turn_sign * residual * CORNER_INITIAL_TURN_RATIO)
            continue

        if p.orientation == "cross":
            print(f"[{tag}] 检测到横线，跨越")
            run_action("go_forward_one_step")
            continue

        # follow 线：优先级判断
        # ① 大幅欠转先修朝向（|heading| 超过容忍阈值时线严重倾斜，最近点与远
        #    前视点横向方向相反，横移会左右振荡；先闭环修朝向把线拉回近似垂直，
        #    误差大用大步、误差小用小步精修，不再固定单次 turn_left/right）
        if abs(p.heading_deg) > ALIGN_HEADING_TOL:
            print(f"[{tag}] heading={p.heading_deg:.1f}°，闭环修朝向")
            _correct_heading(p.heading_deg)
            continue

        # ② 侧边出线（lateral_offset 过大说明线已在视野边缘）
        T_LAT = 40.0  # 侧边出线阈值
        if abs(lateral) > T_LAT:
            if lateral > 0:
                print(f"[{tag}] lateral={lateral:.1f} > {T_LAT}，右移对齐")
                run_action("right_move")
            else:
                print(f"[{tag}] lateral={lateral:.1f} < -{T_LAT}，左移对齐")
                run_action("left_move")
            continue

        # ③ 前视点居中检查（横向对齐后 lookahead 才准确）
        if abs(lx) > T_CENTER:
            if lx > 0:
                print(f"[{tag}] lookahead_x={lx:.1f}，右移对齐")
                run_action("right_move")
            else:
                print(f"[{tag}] lookahead_x={lx:.1f}，左移对齐")
                run_action("left_move")
            continue

        # ④ 居中，对准完成
        print(f"[{tag}] lookahead_x={lx:.1f} ≤ {T_CENTER}，"
              f"heading={p.heading_deg:.1f}°，对准完成")
        return True

    print(f"[{tag}] 超过最大步数，返回")
    return False


def seek_line(detector):
    """寻线主函数：头部扫描 → 单向身体扫描（全周）→ 平移试探

    策略（解决「仅旋转不一定对准线」）：
      1. 头部扫描（±63° 覆盖前方 126° 扇区）
      2. 身体单向右转扫描（turn_right×14 ≈ 360° 全周，不左右交替抵消）
      3. 全周无果 → 平移试探：前进 5 步 + 左移 5 步 + 右移 10 步，
         每步后头部扫描（机器人可能平行于线或距离线太远）
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

    # 身体扫描：单向右转（每转一次仅中心位拍照，14 次 ≈ 360° 全周）
    for body_turn in range(SEEK_BODY_TURNS):
        print(f"[寻线] 身体扫描第 {body_turn + 1}/{SEEK_BODY_TURNS} 次（右转）")
        run_action("turn_right")

        # 每轮仅中心位拍照（省时间，全周覆盖靠身体旋转）
        set_head(HEAD_CENTER)
        frame = capture_frame()
        if frame is not None:
            result = detector.detect(frame)
            if result.exists:
                head_angle = 0.0
                print(f"[寻线] 身体扫描第 {body_turn + 1} 次发现线："
                      f"orientation={result.primary.orientation} "
                      f"lookahead_x={result.primary.lookahead_x:.1f}")
                if _align_to_line(detector):
                    print("===== 寻线成功，进入循迹 =====")
                    return True
                print(f"[寻线] 第 {body_turn + 1} 次身体扫描后对齐失败，继续转")

    # 全周无果：平移试探（前进 + 左右平移，覆盖「平行于线」和「距离太远」两种情况）
    for probe in range(SEEK_MAX_PROBES):
        print(f"[寻线] 平移试探第 {probe + 1}/{SEEK_MAX_PROBES} 轮")
        for action_name, times in (
            ("go_forward_one_step", SEEK_PROBE_FORWARD),
            ("left_move", SEEK_PROBE_LEFT),
            ("right_move", SEEK_PROBE_RIGHT),
        ):
            run_action(action_name, times=times)
            found, head_pulse, head_angle, result = _scan_head(detector)
            if found:
                _turn_body_to_head_angle(head_angle)
                if _align_to_line(detector):
                    print("===== 寻线成功，进入循迹 =====")
                    return True
        print(f"[寻线] 第 {probe + 1} 轮平移试探无果")

    print("===== 寻线失败：未找到线 =====")
    set_head(HEAD_CENTER)
    return False


# =====================================================================
# Phase 2: 循迹阶段
# =====================================================================

def _handle_corner(detector, curvature):
    """弯道处理（闭环）：前进接近拐角 → 按观测拐角角定点转弯 → 出口确认

    curvature < 0 → 左弯（左转，正角）
    curvature > 0 → 右弯（右转，负角）
    返回 True 表示弯已转过（新方向线已确认），False 表示异常（丢线或超限）

    不再按标定值预设转弯次数：初始转弯量由每次观测到的拐角弯折角 corner_deg
    决定，出口阶段按实时 heading 闭环精修，从而适应不同弯角角度与不同直线段
    长度/间距的路线（通用化，不预设任何路线形状）。
    """
    turn_sign = -1.0 if curvature > 0 else 1.0  # 左弯 +1，右弯 -1
    direction = "左" if turn_sign > 0 else "右"
    prefer_side = "left" if turn_sign > 0 else "right"

    print(f"[转弯] {direction}弯（curvature={curvature:.2f}），先前进接近拐角")

    # ---- 阶段1：前进接近拐角点，同时记录观测到的拐角弯折角 ----
    observed_corner_deg = None
    for approach in range(CORNER_APPROACH_MAX):
        set_head(HEAD_CENTER)
        frame = capture_frame()
        if frame is None:
            print("[转弯] 拍照失败")
            return False
        result = detector.detect(frame)
        if result.exists:
            p = result.primary
            if p.orientation == "corner" and p.elbow_px is not None:
                if p.corner_deg > 0:
                    observed_corner_deg = p.corner_deg
                elbow_y = p.elbow_px[1]
                if elbow_y >= CORNER_APPROACH_Y:
                    print(f"[转弯] 拐角已接近（elbow_px.y={elbow_y:.0f} ≥ "
                          f"{CORNER_APPROACH_Y:.0f}），开始定点转弯")
                    break
                print(f"[转弯] 接近中 elbow_px.y={elbow_y:.0f}（< {CORNER_APPROACH_Y:.0f}）")
            elif p.orientation == "cross":
                # 拐角已在脚下（检测成横线），说明已到拐角点
                print("[转弯] 检测到横线，已到拐角点，开始定点转弯")
                break
            else:
                print(f"[转弯] 接近中 orientation={p.orientation}（继续前进）")
        else:
            print("[转弯] 接近中丢线（继续前进）")
        run_action("go_forward_one_step")
        print(f"[转弯] 接近拐角前进 {approach + 1}/{CORNER_APPROACH_MAX} 步")
    else:
        print("[转弯] 接近阶段前进用尽，按当前位置定点转弯")

    # ---- 阶段2：按观测拐角角定点转弯（闭环初始量，非固定次数） ----
    if observed_corner_deg is None:
        observed_corner_deg = 90.0  # 未观测到拐角角时的保守默认
        print(f"[转弯] 未观测到拐角角，用默认 {observed_corner_deg:.0f}°")
    required = turn_sign * observed_corner_deg * CORNER_INITIAL_TURN_RATIO
    cumulative_angle = _turn_by_angle(required)
    print(f"[转弯] 定点转弯：观测角 {observed_corner_deg:.1f}° × "
          f"{CORNER_INITIAL_TURN_RATIO} ≈ 目标 {required:+.1f}°，"
          f"实际 {cumulative_angle:+.1f}°")

    # ---- 阶段3：出口确认（按实时 heading 闭环精修，不再固定次数） ----
    forward_seek_count = 0
    for confirm in range(CORNER_APPROACH_MAX):
        set_head(HEAD_CENTER)
        frame = capture_frame()
        if frame is not None:
            result = detector.detect(frame)
        else:
            result = None

        if result is None or not result.exists:
            # 丢线：优先转向侧扫描找线
            found, _, _, result = _scan_head(detector, prefer_side=prefer_side)
            set_head(HEAD_CENTER)
            if not found:
                if forward_seek_count >= CORNER_SEEK_FORWARD_MAX:
                    print("[转弯] 出口确认丢线前进找线用尽，返回异常")
                    return False
                forward_seek_count += 1
                print(f"[转弯] 出口确认 {confirm + 1}: 丢线，"
                      f"前进一步再找（{forward_seek_count}/"
                      f"{CORNER_SEEK_FORWARD_MAX}）")
                run_action("go_forward_one_step")
                continue

        p = result.primary
        lx = max(-LOOKAHEAD_CLAMP, min(LOOKAHEAD_CLAMP, p.lookahead_x))
        print(f"[转弯] 出口确认 {confirm + 1}: orientation={p.orientation} "
              f"lookahead_x={lx:.1f} heading={p.heading_deg:.1f}")

        if p.orientation == "follow":
            # 见 follow 线但 heading 仍偏：闭环精修朝向，再确认
            if abs(p.heading_deg) > CORNER_EXIT_HEADING:
                correction = -p.heading_deg  # 修朝向的转向量（正=左，负=右）
                # 拐角刚转过时 heading 不可靠（L 形拐角会被误判成 follow，heading
                # 符号可能与真实朝向相反）。若修朝向方向与过弯方向强烈相反，说明
                # 大概率是「尚未转过的拐角」被误判成 follow，此时不能反向转，否则
                # 会把刚转过去的弯又转回来，导致来回振荡。改为继续按原方向补转。
                if turn_sign * correction < -CORNER_EXIT_HEADING:
                    print(f"[转弯] follow 但 heading={p.heading_deg:+.1f}° 与"
                          f"过弯方向相反，疑为拐角误判，继续按{direction}转")
                    cumulative_angle += _turn_by_angle(turn_sign * 30.0)
                    if abs(cumulative_angle) >= CORNER_MAX_CUMULATIVE_DEG:
                        print("[转弯] 累计转角超安全上限，返回异常")
                        return False
                    continue
                print(f"[转弯] 见 follow 线 heading={p.heading_deg:.1f}° 偏，"
                      f"精修朝向")
                cumulative_angle += _correct_heading(p.heading_deg)
                if abs(cumulative_angle) >= CORNER_MAX_CUMULATIVE_DEG:
                    print("[转弯] 累计转角超安全上限，返回异常")
                    return False
                continue
            print(f"[转弯] 弯已转过（heading={p.heading_deg:.1f}° ≤ "
                  f"{CORNER_EXIT_HEADING}°），横向误差交给对齐阶段）")
            return True

        if p.orientation == "cross":
            print("[转弯] 出口确认见横线，跨越")
            run_action("go_forward_one_step")
            continue

        # 仍是 corner：按观测拐角角继续补余量
        residual_deg = p.corner_deg if p.corner_deg > 0 else 30.0
        print(f"[转弯] 出口确认仍见拐角，继续补转 {residual_deg:.1f}°")
        cumulative_angle += _turn_by_angle(turn_sign * residual_deg)
        if abs(cumulative_angle) >= CORNER_MAX_CUMULATIVE_DEG:
            print("[转弯] 累计转角超安全上限，返回异常")
            return False

    print(f"[转弯] 出口确认超过 {CORNER_APPROACH_MAX} 次，返回异常")
    return False


def _handle_line_lost(detector):
    """循迹丢线处理：后退找线，超限则回寻线

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
    stuck_counter = 0  # 连续 cross/丢线 计数（防卡死）

    while total_steps < MAX_TOTAL_STEPS:
        total_steps += 1

        set_head(HEAD_CENTER)
        frame = capture_frame()
        if frame is None:
            print(f"[循迹] step={total_steps} 拍照失败，跳过")
            continue

        result = detector.detect(frame)

        if not result.exists:
            stuck_counter += 1
            if stuck_counter > STUCK_MAX:
                print(f"[循迹] 卡住 {stuck_counter} 步（丢线+横线），回寻线")
                if seek_line(detector):
                    stuck_counter = 0
                    continue
                else:
                    print("[循迹] 重新寻线失败，循迹结束")
                    return False
            print(f"[循迹] step={total_steps} 丢线")
            status = _handle_line_lost(detector)
            if status == "seek":
                if seek_line(detector):
                    stuck_counter = 0
                    continue
                else:
                    print("[循迹] 重新寻线失败，循迹结束")
                    return False
            continue

        p = result.primary
        # 钳制 lookahead_x 防止多项式外推爆炸
        lx_clamped = max(-LOOKAHEAD_CLAMP, min(LOOKAHEAD_CLAMP, p.lookahead_x))
        print(f"[循迹] step={total_steps} orientation={p.orientation} "
              f"lookahead_x={lx_clamped:.1f} lateral_offset={p.lateral_offset:.1f} "
              f"heading={p.heading_deg:.1f} far_ry={p.far_ry:.0f} "
              f"curvature={p.curvature:.4f}")

        if p.orientation == "corner":
            stuck_counter = 0  # 重置卡死计数
            if _handle_corner(detector, p.curvature):
                # 转弯成功，用统一对齐函数对准新方向的线
                print("[循迹] 转弯完成，重新对准")
                if not _align_to_line(detector, context="corner"):
                    print("[循迹] 转弯后对准失败，回寻线")
                    if not seek_line(detector):
                        print("[循迹] 重新寻线失败，循迹结束")
                        return False
            else:
                print("[循迹] 转弯异常，尝试继续循迹")
            continue

        if p.orientation == "cross":
            stuck_counter += 1
            if stuck_counter > STUCK_MAX:
                print(f"[循迹] 卡住 {stuck_counter} 步（丢线+横线），回寻线")
                if seek_line(detector):
                    stuck_counter = 0
                    continue
                else:
                    print("[循迹] 重新寻线失败，循迹结束")
                    return False
            print(f"[循迹] 横线（卡住计数 {stuck_counter}），跨越")
            run_action("go_forward_one_step")
            continue

        # follow 线
        stuck_counter = 0  # 重置卡死计数
        # 终点判定：线远端 ry 显著小于视野高度说明线末端已进入视野（快走完）
        if 0 < p.far_ry < LINE_END_FAR_RY:
            print(f"[循迹] 线末端接近（far_ry={p.far_ry:.0f} < "
                  f"{LINE_END_FAR_RY:.0f}），终点前进")
            forward_steps = 0
            for _ in range(LINE_END_FORWARD_MAX):
                run_action("go_forward_one_step")
                forward_steps += 1
                set_head(HEAD_CENTER)
                f2 = capture_frame()
                r2 = detector.detect(f2) if f2 is not None else None
                if r2 is None or not r2.exists:
                    break
            # 丢线后再盲目前进少量步数抵达线末端（直线、heading 已对齐，短距前进安全）
            for _ in range(LINE_END_BLIND_MAX):
                run_action("go_forward_one_step")
                forward_steps += 1
            print(f"[循迹] 到达线末端（终点前进 {forward_steps} 步），循迹结束")
            return True
        lx = lx_clamped
        # 朝向误差优先：线明显倾斜时，lookahead 的横向符号可能与真实朝向相反
        # （线在机器人正前方斜穿时尤为明显），若仍按 lookahead 转向会与朝向修正
        # 互相打架导致来回振荡。故只要 heading 明显偏，就先闭环修朝向。
        #
        # 但 heading 异常大（> FOLLOW_HEADING_MAX）时多半是拐角被误判成 follow
        # 或分割抖动——真实直线循迹几乎不会出现 35°+ 的朝向误差。此时 heading
        # 不可信，不能按其做大步转向（否则会把刚对准的方向又打歪），改由下方
        # lookahead/lateral 横向逻辑兜底，并借横移改变视角让拐角重新被正确识别。
        if FOLLOW_HEADING_MIN < abs(p.heading_deg) <= FOLLOW_HEADING_MAX:
            print(f"[循迹] heading={p.heading_deg:.1f}° 偏，闭环修朝向")
            _correct_heading(p.heading_deg)
            continue
        if abs(p.heading_deg) > FOLLOW_HEADING_MAX:
            print(f"[循迹] heading={p.heading_deg:.1f}° 过大（疑拐角/抖动），"
                  f"忽略朝向，按横向处理")
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
