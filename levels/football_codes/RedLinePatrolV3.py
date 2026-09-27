#!/usr/bin/python3
# coding=utf8
"""红线巡线 V3：赛道固定为直角弯，转向顺序查表（CORNER_TURN_SEQUENCE），
横移纠偏、路口接近锁存和有界丢线搜索。

V3 在 V2 算法基础上补充了 demoV4.py / patrol_end_recovery.py 期望的
总控接口：line_center_x、line_lost_time、turn_started、approach_active、
detect_red_line()、roi、MIN_CONTOUR_AREA、calibration_param_path。
算法与参数与 V2 保持一致。
"""

import os
import sys
import cv2
import time
import math
import threading
import numpy as np

# 支持从 Robot_control_self_module/levels/football_codes 直接运行：TonyPi
# 框架目录提供 hiwonder SDK 与 Functions 包（CameraCalibration 等）。在
# TonyPi 目录下运行时该路径已在 sys.path 中，此块自动跳过；PC 离线模式下
# 该目录不存在，同样跳过。
_TONYPI_DIR = '/home/pi/TonyPi'
if os.path.isdir(_TONYPI_DIR) and _TONYPI_DIR not in sys.path:
    sys.path.insert(0, _TONYPI_DIR)

# hiwonder SDK 仅存在于机器人上。Windows 导入失败时进入离线模式：
# 视觉与决策（run()/decide_action()）仍可被 PC 模拟器离线调用，
# 运动线程不启动，比赛代码在机器人上的行为不变。
try:
    import hiwonder.Camera as Camera
    import hiwonder.ros_robot_controller_sdk as rrc
    from hiwonder.Controller import Controller
    import hiwonder.ActionGroupControl as AGC
    import hiwonder.yaml_handle as yaml_handle
    HARDWARE_AVAILABLE = True
except ImportError:
    Camera = None
    rrc = None
    Controller = None
    AGC = None
    yaml_handle = None
    HARDWARE_AVAILABLE = False
    print('RedLinePatrolV3：未找到 hiwonder SDK，进入离线模式（不启动运动线程）', flush=True)

if sys.version_info.major == 2:
    print('Please run this program with python3!')
    sys.exit(0)

# 兼容两种部署结构：机器人上以 Functions 包形式运行，Windows 工作区直接
# 从本目录导入 CameraCalibration。
try:
    from Functions.CameraCalibration.CalibrationConfig import calibration_param_path
except ImportError:
    from CameraCalibration.CalibrationConfig import calibration_param_path


# ============ 可调参数 ============
FRAME_SIZE = (640, 480)
IMAGE_CENTER_X = FRAME_SIZE[0] // 2

# 红色 HSV 双区间
RED_H_LOW1 = 0
RED_H_HIGH1 = 17
RED_H_LOW2 = 149
RED_H_HIGH2 = 180
RED_S_LOW = 0
RED_V_LOW = 50

# 视觉区域和轮廓
VISION_TOP_Y = 170
VISION_SIDE_CROP = 80
MIN_PATH_AREA = 180
MIN_BAND_PIXELS = 45
MORPH_OPEN_SIZE = 3
MORPH_CLOSE_SIZE = 7

# 从远到近的水平采样带。算法使用这些带的中心拟合前方方向。
SAMPLE_BANDS = (
    (180, 230),
    (230, 280),
    (280, 330),
    (330, 380),
    (380, 430),
    (430, 480),
)
SAMPLE_BAND_HEIGHT = 50  # 各采样带等高，用于计算横条填充率

# 正常巡线阈值
CENTER_DEADBAND = 42              # 近处红线离画面中心超过此值才横移
HEADING_TURN_THRESHOLD = 5        # 极缓弯也立即转向
HEADING_MAX = 50.0                # heading 绝对值上限；超过视为路口/噪声，置 0 忽略朝向
HEADING_HINT_THRESHOLD = 25       # 丢线时用最后朝向猜搜索方向的阈值
VERTICAL_HEADING_TOLERANCE = 5    # 与转向阈值衔接，避免决策死区
FILTER_ALPHA = 0.5                # 位置和方向的一阶滤波系数
FILTER_RESET_JUMP = 140           # 位置突变过大时直接重置滤波

# 路口检测：赛道固定为直角弯，只需判断"是否到路口"，不再靠视觉猜方向
HORIZONTAL_MIN_WIDTH = 90         # 普通线宽 52-75，直角弯横条 94-160，90 只对真实横向岔路触发
HORIZONTAL_FILL_MIN = 0.30        # 横条采样带填充率下限：单条横条填充率高(~0.7+)，
                                  # 转过弯后"前方竖线+脚下残留"的水平跨度填充率低(~0.14)，
                                  # 用填充率排除多线跨度被误判成横条。
CORNER_DIRECTION_DEADBAND = 15    # 横条中点相对主线中心偏移超过此值才判向（否则歧义=0）
CORNER_TURN_TRIGGER_Y = 350       # 路口特征（横条）下移到该深度即开始转身。
                                  # 曾设 380：横条在画面底部因透视/裁剪变稀疏（fill_ratio
                                  # 掉到 0.30 以下被过滤），corner_y 达不到 380 就消失，
                                  # 机器人横向略偏（横条偏左被裁剪）时转弯完全不触发、冲过路口。
                                  # 实机数据：接近阶段横条稳定在 ~246，到路口升到 343~388；
                                  # 曾降到 330 保证在横条饱满时触发，但实测偏早（机器人离弯
                                  # 还有余量就转身）。最近日志横条在 y≈370 仍能检出
                                  # （horizontal_seen=True），y≈376 才因填充率不足消失，
                                  # 故上调到 350 更接近弯再转，同时低于 370 失效点保留余量。
CORNER_LATCH_SECONDS = 2.0        # 路口特征短暂漏检时继续保持"已到路口"状态
CORNER_DEBUG_INTERVAL = 0.5       # 调试打印节流：每隔多久打印一次检测到的最大宽度/行

# 转弯后确认期：转弯刚结束时画面里可能仍残留横条（corner_ready 持续 True），
# 直接响应会把残留横条误判成下一个路口导致连续转弯。确认期内屏蔽 corner_ready，
# 横条残留说明转弯没转够、朝原方向小步微调；横条消失（面对直道）连续数帧后恢复。
POST_TURN_CLEAR_FRAMES = 5        # 连续 horizontal_seen=False 多少帧才认定已面对直道
POST_TURN_MAX_EXTRA_TURNS = 12    # 竖线可见时最多额外小步微调次数（防卡死）
POST_TURN_MAX_EXTRA_FORWARDS = 8  # 横条残留、竖线未现时最多前进出弯步数（防盲行）
POST_TURN_MAX_EXTRA_SWINGS = 3    # 前进出弯用尽后横条仍不退场，最多反向回摆小步数
                                  #（转弯过转时把新直道带回视野；竖线一出现即回 heading 闭环）
CORNER_BAR_DWELL_TIMEOUT = 12.0   # 正常循迹中横条持续可见超过此时长，启动路口滞留保护
CORNER_BAR_DWELL_SWINGS = 3       # 滞留保护每次最多反向回摆小步数（横条退场后重置）
POST_TURN_MIN_FORWARD_STEPS = 3   # 转弯完成后，前进/横移累计达到此次数才允许触发
                                  # 下一个路口：保证转完弯必须真正驶离路口，防止残留
                                  # 横条在弯口原地连转漏弯。残留横条随前进自然退场，
                                  # 真路口的横条随接近变深，无需硬编码时间或距离。

# 赛道固定，转弯顺序提前写死（从起点开始，依次对应每个路口的转向）
CORNER_TURN_SEQUENCE = ['left', 'right', 'right', 'left']

# 与 CORNER_TURN_SEQUENCE 平行：对应弯在触发转弯后、主转之前先前进
# CORNER_PRE_TURN_FORWARD_STEPS 步，把转身支点推进到路口。
# 260927 实车（居中门限生效后）：触发时横条已压到最深（corner_y≈387），
# 横条驱动的弯前前进把机器人推过了路口，主转后出口线落在身后（精修
# "无线停止"、红线出现在 -221px 左边缘）。居中门限已替代本补偿的作用，
# 全部关闭；如某弯位仍偏早，可单独打开并按 MAX_STEPS 限幅。
CORNER_PRE_TURN_FORWARD = (False, False, False, False)
# 横条驱动的弯前前进：触发转弯后继续前进，直到横条消失（横条随接近而
# 填充率/宽度坍塌，消失=已走到路口上方），再经 corner_pending 锁存多走
# 约 1 步后原地主转。步数随每个弯的实际距离自适应；上限
# CORNER_PRE_TURN_FORWARD_MAX_STEPS 步兜底，防止横条与场地红色连成片、
# 永不消失时一直前进。前进过头仍在线上，比支点偏早（转完离线）安全。
CORNER_PRE_TURN_FORWARD_MAX_STEPS = 4

# 转弯采用「主转固定步数 + 精修」两阶段：
#   主转按标定角度（左 22°/次、右 25.7°/次）转到接近 90°——左 4 步≈88°、
#   右 3 步≈77°。转弯中间新直道在画面里倾斜、被采样带误判成横条（宽>90），
#   direction_samples 为空、heading 无法计算，所以主转阶段不看视觉、只按
#   标定角度开环转固定步数，避免像旧版那样用「横条消失/竖线 heading」在转弯
#   中间判断失效而原地打转 308°。
#   精修阶段转弯已基本到位：横条还在→继续转大步兜底；竖线出现→按 heading
#   符号闭环（未转够续转大步、转过反向小步回摆），直到竖线正对。
CORNER_MAIN_STEPS_LEFT = 4        # 左转主转步数：4×22°=88°
CORNER_MAIN_STEPS_RIGHT = 3       # 右转主转步数：3×25.7°≈77°（欠转，留给精修竖线闭环补足）。
                                  # 曾设 4（103°）：260927 实车回放显示转弯合计过转约 113°，
                                  # 新直道不出现在视野、横条压脚不退场，机器人在路口滞留约
                                  # 20s 后侧冲出赛道。旧版 3 步的 231° 过冲源自旧精修的横条
                                  # offset 大步续转，现已禁用（横条分支只同向小步 2 次），
                                  # 3 步可以安全使用。
CORNER_FINE_MAX_STEPS = 6         # 精修阶段最多步数（未转够用大步，横条残留与转过用小步）

# 260927 回放修复：转弯方向的控制权交给标定步数与顺序表，视觉只做触发与
# "新直道出现"的退出确认；不再用横条中点偏移判向（偏姿态下符号会反转）。
CORNER_TRIGGER_CENTER_DEADBAND = 20   # 触发时近端线中心须在画面中心 ±20px 内
                                      #（弯1实测 ±4px；弯2/3/4 触发时 +20~33px，支点偏早）
CORNER_FINE_BAR_MAX_STEPS = 2         # 竖线未出现时，精修最多按查表方向小步续转的次数
                                      #（右小步 5.2°、左小步 8.625°，各方向各用各值）
CORNER_BAR_FREE_REARM_FRAMES = 5      # 上一弯完成后连续无横条帧数达到此值，
                                      # 才允许触发下一弯（防残留横条串弯）

# 转弯节奏与提速：主转开环不看视觉，每步只需等动作完成；精修每步只需
# 一张新帧（15fps 下足够），正常巡线仍用 NORMAL_ACTION_SETTLE=1.0。
# 注意：0.3/0.5 曾导致弯3转完丢线未找回（机器人未停稳/视觉未稳定嫌疑），
# 暂回退到 1.0 与正常巡线一致，定位元凶后再逐步调小。
TURN_MAIN_STEP_SETTLE = 1.0
TURN_FINE_STEP_SETTLE = 1.0
FINE_SMALL_STREAK_ESCALATE = 2    # 连续小步转弯达到此次数仍未明显改善 -> 升级一次同方向大步
FINE_SMALL_PROGRESS_MIN = 3.0     # 小步窗口内误差（px）改善小于此值视为"效果不够明显"
CORNER_EXIT_HEADING_PX = 20.0     # 转弯完成判据：前方竖线 heading 绝对值 ≤ 此值视为已对准新方向
                                  # （正常直线循迹 heading 在 ±20px 内，精细对齐交给后续循迹）。
CORNER_MIN_VERTICAL_SPAN = 40.0   # 前方竖线存在的判据：direction_samples 的垂直跨度需 ≥ 此值，
                                  # 否则说明竖线尚未出现（横条主导），继续按原方向大步转。
CORNER_VERTICAL_SIGN_MIN_SPAN = 100.0  # 竖线 heading 符号可信所需的最小垂直跨度：span 贴近
                                  # 40px 下限时斜率被放大到 ±数百（260927 实车 rh=-412.6 被
                                  # 判"未转够"连续大步续转，弯1 累计转约 149° 过冲），符号
                                  # 不可信，只允许小步试探，禁止大步续转。

# 动作与安全限制
FORWARD_ACTION = 'go_forward_fast'
TURN_LEFT_ACTION = 'turn_left'
TURN_RIGHT_ACTION = 'turn_right'
TURN_LEFT_SMALL_ACTION = 'turn_left_small_step'    # 直行时的实时方向微调，转角更小
TURN_RIGHT_SMALL_ACTION = 'turn_right_small_step'
LATERAL_LEFT_ACTION = 'left_move_fast'
LATERAL_RIGHT_ACTION = 'right_move_fast'
NORMAL_ACTION_SETTLE = 1.0        # 每个正常动作单元完成后停 1 秒观察
MAX_NO_PROGRESS_TURNS = 16        # 小转角修正每次改善的角度更小，需要更多次才能判定"无进展"
TURN_PROGRESS_MIN = 3             # 同上，进展阈值相应调小
TURN_ACTION_TIMES = 1             # 与 cmd_keyboard.py 的 q/e 配置一致
TURN_WITH_STAND = True

# 丢线搜索：记忆方向 3 次，未找到再反向 6 次
LOST_CONFIRM_SECONDS = 0.45
SEARCH_PRIMARY_MAX_TURNS = 3
SEARCH_REVERSE_MAX_TURNS = 6
SEARCH_OBSERVE_SECONDS = 1.0

HEAD_TILT_OFFSET = 40
SHOW_DISPLAY = False

# 调试帧保存（默认关闭，不影响比赛运行）：置 True 后把每一帧校正后图像
# 存到 DEBUG_SAVE_DIR，供 PC 端 test_programs_NoUseInMain/redline_route_debug.py
# 离线复盘（demoV4 与独立运行都经过 run()，两条路径均覆盖）。
# PC 拉取：python tools/pull_from_robot.py --remote /home/pi/codes/pictures/patrol --dest <本地目录>
SAVE_DEBUG_FRAMES = True
DEBUG_SAVE_DIR = '/home/pi/codes/pictures/patrol'
DEBUG_SAVE_EVERY_N = 1
# ==================================

# V3 总控接口：脚下红线轮廓最小面积（与 demoV4.py PatrolEndDetector 配合）。
MIN_CONTOUR_AREA = 50

# V3 总控接口：脚下近处 ROI。demoV4.py 取 roi[-1] 作为脚下区域，
# 复用 V2 的 VISION_SIDE_CROP 左右裁剪，底部到 480 即画面最下方。
roi = (
    (430, 480, VISION_SIDE_CROP, FRAME_SIZE[0] - VISION_SIDE_CROP),
)


if HARDWARE_AVAILABLE:
    board = rrc.Board()
    ctl = Controller(board)
else:
    board = None
    ctl = None
servo_data = None

enter = False
running = False

# V3 总控接口：当前红线中心 x（-1 表示丢线）、丢线起始时间、直角弯状态。
line_center_x = -1
line_lost_time = 0
turn_started = False
approach_active = False
corner_ready = False
search_active = False     # 丢线搜索进行中（demoV4 终点判定据此屏蔽候选终点）
last_turn_completed_at = 0.0  # 最近一次路口转弯完成时间（monotonic），终点判定冷却期用

_state_lock = threading.Lock()
_vision_state = None
_frame_id = 0
_filtered_near_x = None
_filtered_heading = None
_corner_pending = False
_corner_pending_y = 0
_corner_pending_last_seen = 0
_corner_index = 0          # 下一个待完成的路口在 CORNER_TURN_SEQUENCE 中的下标
_corner_direction = 0       # 最近一次横条判出的转向：+1右 -1左 0歧义/无
_corner_offset_x = 0.0      # 判向时的横条中点相对主线中心偏移（像素）
_last_corner_debug_time = 0


def direction_name(direction):
    return '右' if direction > 0 else '左'


def turn_action_for(direction):
    return TURN_RIGHT_ACTION if direction > 0 else TURN_LEFT_ACTION


def small_turn_action_for(direction):
    return TURN_RIGHT_SMALL_ACTION if direction > 0 else TURN_LEFT_SMALL_ACTION


def expected_corner_direction(corner_index):
    """按赛道固定的转弯顺序表查询第 corner_index 个路口该往哪转。
    顺序已跑完时返回 0（不再强制转弯）。"""
    if corner_index >= len(CORNER_TURN_SEQUENCE):
        return 0
    return 1 if CORNER_TURN_SEQUENCE[corner_index] == 'right' else -1


def empty_vision_state(frame_id=0):
    return {
        'frame_id': frame_id,
        'timestamp': time.time(),
        'visible': False,
        'last_seen_time': 0,
        'near_x': IMAGE_CENTER_X,
        'near_y': FRAME_SIZE[1] - 1,
        'heading': 0.0,
        'raw_heading': 0.0,
        'vertical_span': 0.0,
        'horizontal_corner': False,
        'corner_pending': False,
        'corner_y': 0,
        'corner_ready': False,
        'corner_direction': 0,
        'corner_offset_x': 0.0,
        'sample_count': 0,
        'confidence': 0.0,
    }


_vision_state = empty_vision_state()


def load_config():
    global servo_data
    servo_data = yaml_handle.get_yaml_data(yaml_handle.servo_file_path)


def initMove():
    ctl.set_pwm_servo_pulse(1, servo_data['servo1'] + HEAD_TILT_OFFSET, 500)
    ctl.set_pwm_servo_pulse(2, servo_data['servo2'], 500)


def reset():
    global _vision_state, _frame_id, _filtered_near_x, _filtered_heading
    global _corner_pending, _corner_pending_y, _corner_pending_last_seen
    global _corner_index
    global _corner_direction, _corner_offset_x
    global line_center_x, line_lost_time, turn_started, approach_active, corner_ready
    global search_active
    global last_turn_completed_at

    with _state_lock:
        _frame_id = 0
        _vision_state = empty_vision_state()
        _filtered_near_x = None
        _filtered_heading = None
        _corner_pending = False
        _corner_pending_y = 0
        _corner_pending_last_seen = 0
        _corner_index = 0
        _corner_direction = 0
        _corner_offset_x = 0.0
        line_center_x = -1
        line_lost_time = 0
        turn_started = False
        approach_active = False
        corner_ready = False
        search_active = False
        last_turn_completed_at = 0.0


def init():
    global enter
    load_config()
    initMove()
    reset()
    enter = True
    print('RedLinePatrolV3 Init')


def start():
    global running
    running = True
    print('RedLinePatrolV3 Start')


def stop():
    global running
    running = False
    reset()
    print('RedLinePatrolV3 Stop')


def exit():
    global enter, running
    enter = False
    running = False
    reset()
    AGC.runActionGroup('stand_slow')
    print('RedLinePatrolV3 Exit')


def detect_red_line(frame):
    """检测整帧红色区域的二值 mask（不做 ROI 裁剪）。

    供 demoV4.py 的 PatrolEndDetector.detect_foot_line() 复用，
    保证总控与巡线模块使用同一套红色 HSV 与形态学定义。
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower1 = np.array([RED_H_LOW1, RED_S_LOW, RED_V_LOW])
    upper1 = np.array([RED_H_HIGH1, 255, 255])
    lower2 = np.array([RED_H_LOW2, RED_S_LOW, RED_V_LOW])
    upper2 = np.array([RED_H_HIGH2, 255, 255])
    mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1),
                          cv2.inRange(hsv, lower2, upper2))

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (MORPH_OPEN_SIZE, MORPH_OPEN_SIZE))
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (MORPH_CLOSE_SIZE, MORPH_CLOSE_SIZE))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    return mask


def build_red_mask(frame):
    mask = detect_red_line(frame)
    mask[:VISION_TOP_Y, :] = 0
    mask[:, :VISION_SIDE_CROP] = 0
    mask[:, FRAME_SIZE[0] - VISION_SIDE_CROP:] = 0
    return mask


def select_path_contour(mask, reference_x):
    contours = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    candidates = [c for c in contours
                  if math.fabs(cv2.contourArea(c)) >= MIN_PATH_AREA]
    if not candidates:
        return None, 0.0

    def contour_score(contour):
        area = math.fabs(cv2.contourArea(contour))
        x, y, width, height = cv2.boundingRect(contour)
        distance = max(0, x - reference_x, reference_x - (x + width))
        # 优先选择与上次路径相连的红色区域，面积用于排除小噪点。
        return area - distance * 6 + (y + height) * 0.4

    selected = max(candidates, key=contour_score)
    return selected, math.fabs(cv2.contourArea(selected))


def sample_path(path_mask):
    """返回 (y, center_x, left_x, right_x, pixel_count) 采样点。"""
    samples = []
    for y_start, y_end in SAMPLE_BANDS:
        ys, xs = np.nonzero(path_mask[y_start:y_end, :])
        if xs.size < MIN_BAND_PIXELS:
            continue
        samples.append((
            float(y_start + np.median(ys)),
            float(np.median(xs)),
            int(np.min(xs)),
            int(np.max(xs)),
            int(xs.size),
        ))
    return samples


def analyze_path_samples(samples):
    """把采样点转换成近处位置、方向误差、路口提示和路口方向。
    路口方向由最宽横条中点相对主线近处中心的横向偏移判断，供 move() 与
    查表结果做一致性校验；方向本身仍来自 CORNER_TURN_SEQUENCE。"""
    if not samples:
        return None

    near = max(samples, key=lambda sample: sample[0])
    direction_samples = [
        sample for sample in samples
        if sample[3] - sample[2] < HORIZONTAL_MIN_WIDTH
    ]
    raw_heading = 0.0
    if len(direction_samples) >= 2:
        direction_near = max(direction_samples, key=lambda sample: sample[0])
        direction_ahead = min(direction_samples, key=lambda sample: sample[0])
        vertical_span = direction_near[0] - direction_ahead[0]
        if vertical_span >= 40:
            raw_heading = ((direction_ahead[1] - direction_near[1]) * 200.0 /
                           vertical_span)
    else:
        vertical_span = 0.0

    # heading 是「远端 - 近端」横向像素差经斜率放大后的值，正常直线循迹在
    # ±20px 内。路口附近 direction_samples 只剩远端窄带、vertical_span 贴近
    # 40px 下限时会被放大到 ±数百（实机见 heading=-461.4），远超可信范围。
    # 超过上限说明是路口/噪声，置 0 忽略朝向修正，让 corner_pending/
    # corner_ready 逻辑接管，避免在接近路口时乱转方向。
    # raw_heading 保留未截断的原始值，供转弯闭环判断「未转够 / 转过」的符号。
    heading = raw_heading
    if abs(heading) > HEADING_MAX:
        heading = 0.0

    # 路口特征：某条采样带的红色区域宽度远超正常直线段宽度，说明这里出现了
    # 岔路（横向的红线段）。只需要知道"到没到"，不需要判断岔路方向。
    # 用填充率排除「前方竖线 + 脚下残留」的水平跨度被误判成横条。
    horizontal_seen = False
    corner_y = 0
    max_width = 0
    max_width_y = 0
    max_width_left_x = 0
    max_width_right_x = 0
    max_width_fill_ratio = 0.0
    for sample_y, _, left_x, right_x, count in samples:
        width = right_x - left_x
        if width > max_width:
            max_width = width
            max_width_y = sample_y
            max_width_left_x = left_x
            max_width_right_x = right_x
            max_width_fill_ratio = (count / (width * SAMPLE_BAND_HEIGHT)
                                    if width > 0 else 0.0)
        if width >= HORIZONTAL_MIN_WIDTH:
            fill_ratio = (count / (width * SAMPLE_BAND_HEIGHT)
                          if width > 0 else 0.0)
            if fill_ratio >= HORIZONTAL_FILL_MIN:
                horizontal_seen = True
                corner_y = max(corner_y, sample_y)

    corner_ready = horizontal_seen and corner_y >= CORNER_TURN_TRIGGER_Y

    # 判向：最宽横条（直角弯的横向红线段）只向转弯方向延伸，其中点相对
    # 主线近处中心的横向偏移方向即转弯方向。偏移过小视为歧义，返回 0。
    corner_direction = 0
    corner_offset_x = 0.0
    if horizontal_seen and max_width_right_x > max_width_left_x:
        corner_center_x = (max_width_left_x + max_width_right_x) / 2.0
        corner_offset_x = corner_center_x - near[1]
        if corner_offset_x >= CORNER_DIRECTION_DEADBAND:
            corner_direction = 1
        elif corner_offset_x <= -CORNER_DIRECTION_DEADBAND:
            corner_direction = -1

    return {
        'near_x': near[1],
        'near_y': near[0],
        'heading': heading,
        'raw_heading': raw_heading,
        'vertical_span': vertical_span,
        'horizontal_corner': horizontal_seen,
        'corner_y': corner_y,
        'corner_ready': corner_ready,
        'corner_direction': corner_direction,
        'corner_offset_x': corner_offset_x,
        'sample_count': len(samples),
        'max_width': max_width,
        'max_width_y': max_width_y,
        'max_width_left_x': max_width_left_x,
        'max_width_right_x': max_width_right_x,
        'max_width_fill_ratio': max_width_fill_ratio,
    }


def filter_measurements(near_x, heading):
    global _filtered_near_x, _filtered_heading

    if (_filtered_near_x is None or
            abs(near_x - _filtered_near_x) > FILTER_RESET_JUMP):
        _filtered_near_x = float(near_x)
    else:
        _filtered_near_x = (FILTER_ALPHA * near_x +
                            (1.0 - FILTER_ALPHA) * _filtered_near_x)

    if _filtered_heading is None:
        _filtered_heading = float(heading)
    else:
        _filtered_heading = (FILTER_ALPHA * heading +
                             (1.0 - FILTER_ALPHA) * _filtered_heading)

    return _filtered_near_x, _filtered_heading


def get_vision_state():
    with _state_lock:
        return dict(_vision_state)


def clear_pending_corner():
    global _vision_state, _corner_pending
    global _corner_pending_y, _corner_pending_last_seen
    global _corner_direction, _corner_offset_x

    with _state_lock:
        _corner_pending = False
        _corner_pending_y = 0
        _corner_pending_last_seen = 0
        _corner_direction = 0
        _corner_offset_x = 0.0
        _vision_state = dict(_vision_state)
        _vision_state['corner_pending'] = False
        _vision_state['corner_ready'] = False
        _vision_state['corner_direction'] = 0
        _vision_state['corner_offset_x'] = 0.0


def advance_corner_index():
    """完成一次路口转弯后，指向赛道顺序表里的下一个路口。"""
    global _corner_index
    with _state_lock:
        _corner_index += 1


def get_corner_index():
    with _state_lock:
        return _corner_index


class LostLineSearch:
    def __init__(self):
        self.reset()

    def reset(self):
        self.active = False
        self.exhausted = False
        self.primary_direction = 0
        self.primary_turns = 0
        self.reverse_turns = 0

    def next_turn(self, preferred_direction):
        if self.exhausted:
            return 0, '', 0, 0
        if not self.active:
            self.active = True
            self.primary_direction = 1 if preferred_direction >= 0 else -1

        if self.primary_turns < SEARCH_PRIMARY_MAX_TURNS:
            self.primary_turns += 1
            return (self.primary_direction, '正向', self.primary_turns,
                    SEARCH_PRIMARY_MAX_TURNS)

        if self.reverse_turns < SEARCH_REVERSE_MAX_TURNS:
            self.reverse_turns += 1
            return (-self.primary_direction, '反向', self.reverse_turns,
                    SEARCH_REVERSE_MAX_TURNS)

        self.exhausted = True
        return 0, '', 0, 0


def choose_search_direction(state, corner_index):
    """丢线后猜一个方向去找线：赛道固定，优先用当前路口该转的方向。"""
    expected = expected_corner_direction(corner_index)
    if expected != 0:
        return expected, '赛道顺序'
    if abs(state['heading']) >= HEADING_HINT_THRESHOLD:
        return (1 if state['heading'] > 0 else -1), '最后方向'
    if abs(state['near_x'] - IMAGE_CENTER_X) >= CENTER_DEADBAND:
        return (1 if state['near_x'] > IMAGE_CENTER_X else -1), '最后位置'
    return 1, '默认方向'


def observe_for_line(timeout):
    """搜索转向后持续检查视觉，发现红线立即返回。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if get_vision_state()['visible']:
            return True
        time.sleep(0.02)
    return False


class Decision:
    """decide_action 的返回值：本轮要执行的动作计划与等待/观察指令。

    真机 move() 执行器与离线模拟器共用：actions 逐条执行（每条可带 settle
    睡眠），wait>0 表示本轮不执行动作、等待后重新决策，observe 表示动作
    执行完后按 SEARCH_OBSERVE_SECONDS 观察红线（丢线搜索用）。
    """
    def __init__(self):
        self.actions = []       # [(action_name, times, with_stand, settle), ...]
        self.wait = 0.0         # 本轮不执行动作，等待该秒数后重新决策
        self.settle_after = 0.0 # 全部动作执行完后的额外睡眠
        self.observe = False    # 动作执行完后观察红线（丢线搜索用）
        self.search_count = 0   # observe 命中的搜索次数（供打印）
        self.label = ''         # 本轮决策描述（模拟器显示用）


class PatrolSession:
    """move() 决策链的跨帧会话状态（纯数据，无硬件）。

    模拟器与真机 move() 共用同一份状态定义，保证离线回放与真机行为一致。
    corner_index 在"暂停巡线"（running=False）期间保留，与重构前模块级
    _corner_index 的保留语义一致（patrol_end_recovery 漏弯恢复依赖它）。
    """
    def __init__(self):
        self.last_handled_frame = -1
        self.last_visible_state = empty_vision_state()
        self.search = LostLineSearch()
        self.search_exhausted_reported = False
        self.last_action_label = ''
        self.last_turn_direction = 0
        self.last_turn_error = None
        self.no_progress_turns = 0
        self.turn_hold_reported = False
        self.no_line_since = 0
        self.ever_seen_line = False
        self.post_turn_active = False
        self.post_turn_direction = 0
        self.post_turn_clear_frames = 0
        self.post_turn_extra_turns = 0
        # 转弯后前进进度：最近一次转弯完成后，累计执行的前进/横移动作次数。
        # 达到 POST_TURN_MIN_FORWARD_STEPS 才允许触发下一个路口，防止
        # 残留横条在弯口原地连转漏弯（初始化即达标：尚未转过弯）。
        self.post_turn_progress_steps = POST_TURN_MIN_FORWARD_STEPS
        self.corner_index = 0
        # 横条退场门限：上一弯完成后需连续 CORNER_BAR_FREE_REARM_FRAMES 帧
        # 无横条才允许触发下一弯（初始无上一弯，直接允许触发）。
        self.corner_rearmed = True
        self.bar_free_frames = 0
        # 精修横条分支计数：竖线未现时最多同向小步续转 CORNER_FINE_BAR_MAX_STEPS 次
        self.turn_bar_steps = 0
        # M8 路口滞留保护：出弯前进用尽后的反向回摆步数，与正常循迹中横条
        # 持续可见的起始时间（横条退场即重置）。
        self.post_turn_swing_done = 0
        self.bar_dwell_start = 0.0
        self.bar_dwell_swings = 0
        # 转弯状态机（重构抽取：原 move() 的主转+精修拆成逐帧决策）
        self.turn_active = False
        self.turn_corner = 0
        self.turn_steps = 0
        self.turn_main_steps_total = 0
        self.turn_fine_remaining = 0
        self.turn_pre_forward_remaining = 0   # 转弯前还需前进几步（仅查表指定的弯）
        self.turn_main_issued = False         # 主转固定步数是否已下发
        self.turn_small_streak = 0            # 连续小步转弯计数（升级大步用）
        self.turn_streak_base_error = 0.0     # 小步序列起始误差（px）

    def reset(self, keep_corner_index=False):
        corner_index = self.corner_index if keep_corner_index else 0
        self.__init__()
        self.corner_index = corner_index


def _finish_turn(session, now, decision):
    """转弯主转+精修全部结束后的收尾，与重构前 move() 转弯完成段一一对应。"""
    global turn_started, last_turn_completed_at
    turn_started = False
    last_turn_completed_at = now
    print('V3 巡线：路口{}转弯{}步（主{}步+精修{}步）'.format(
        session.corner_index + 1, session.turn_steps,
        session.turn_main_steps_total,
        session.turn_steps - session.turn_main_steps_total))
    session.post_turn_active = True
    session.post_turn_direction = session.turn_corner
    session.post_turn_clear_frames = 0
    session.post_turn_extra_turns = 0
    # 转弯刚完成：清零前进进度，必须先真正驶离路口（前进/横移累计
    # POST_TURN_MIN_FORWARD_STEPS 次）才允许触发下一个路口。
    session.post_turn_progress_steps = 0
    session.corner_index += 1
    # 刚转完弯：横条残留期间禁止触发下一弯（串弯防护），横条连续退场
    # CORNER_BAR_FREE_REARM_FRAMES 帧后由 decide_action 重新允许。
    session.corner_rearmed = False
    session.bar_free_frames = 0
    session.turn_bar_steps = 0
    session.post_turn_swing_done = 0
    session.bar_dwell_start = 0.0
    session.bar_dwell_swings = 0
    # 同步模块级 corner_index：demoV4 终点判定的进度门限（M9）经
    # get_corner_index() 读取，此前该值从未随转弯更新。
    global _corner_index
    _corner_index = session.corner_index
    clear_pending_corner()
    session.last_turn_direction = 0
    session.last_turn_error = None
    session.no_progress_turns = 0
    session.turn_hold_reported = False
    session.turn_active = False
    decision.settle_after = NORMAL_ACTION_SETTLE
    decision.label = '路口{}转弯完成'.format(session.corner_index)


def _decide_turn_fine_step(state, session, now, decision):
    """转弯进行中的一步决策：对应重构前 move() 精修 for 循环的一次迭代。

    真机上每帧调用一次，等价于原循环内每次 get_vision_state() 后推进一步；
    模拟器逐帧喂图，自然推进精修直至完成。连续小步改善不足时升级为同方向
    大步一次，加速收敛。
    """
    vis = state.get('visible', False)
    span = state.get('vertical_span', 0.0)
    raw_h = state.get('raw_heading', 0.0)
    hc = state.get('horizontal_corner', False)

    if session.turn_fine_remaining <= 0:
        _finish_turn(session, now, decision)
        return decision

    # 完成：前方竖线存在且已正对。
    if (vis and span >= CORNER_MIN_VERTICAL_SPAN and
            abs(raw_h) <= CORNER_EXIT_HEADING_PX):
        print('V3 转弯精修：第{}步 完成 span={:.0f} raw_h={:.1f}'.format(
            session.turn_steps, span, raw_h))
        _finish_turn(session, now, decision)
        return decision

    vertical_usable = vis and span >= CORNER_MIN_VERTICAL_SPAN
    error = (abs(raw_h) if vertical_usable
             else abs(state.get('corner_offset_x', 0.0)))
    small_action = None   # 本轮若用小步时的动作
    big_action = None     # 同方向大步（小步改善不足时升级用）

    if vertical_usable:
        # 竖线出现但未正对。符号判据（与 line_seeker 的
        # turn_sign*heading<0 等价，因 corner_turn 右=+1、
        # 左=-1 恰与 turn_sign 相反）：
        #   corner_turn*raw_h>0 → 未转够（线在转弯侧）→ 续转大步
        #   corner_turn*raw_h<0 → 转过（线在反侧）→ 反向小步回摆
        if session.turn_corner * raw_h > 0:
            if (span < CORNER_VERTICAL_SIGN_MIN_SPAN
                    or abs(raw_h) > HEADING_MAX):
                # span 过窄或斜率过大（实车 rh=-412.6）时符号不可信：只按
                # 查表方向小步试探，禁止大步续转（防 149° 式过冲）。
                print('V3 转弯精修：第{}步 竖线信号不可信小步试探 span={:.0f} raw_h={:.1f}'.format(
                    session.turn_steps, span, raw_h))
                session.turn_steps += 1
                session.turn_fine_remaining -= 1
                decision.actions = [(small_turn_action_for(session.turn_corner),
                                     1, True, TURN_FINE_STEP_SETTLE)]
                decision.label = '转弯精修第{}步（竖线信号不可信，小步试探）'.format(
                    session.turn_steps)
                return decision
            print('V3 转弯精修：第{}步 未转够续转 span={:.0f} raw_h={:.1f}'.format(
                session.turn_steps, span, raw_h))
            step_action = turn_action_for(session.turn_corner)
            session.turn_small_streak = 0
        else:
            print('V3 转弯精修：第{}步 转过回摆 span={:.0f} raw_h={:.1f}'.format(
                session.turn_steps, span, raw_h))
            small_action = small_turn_action_for(-session.turn_corner)
            big_action = turn_action_for(-session.turn_corner)
    elif hc:
        # 横条还在、竖线未出现：偏姿态下横条中点偏移 offset_x 的符号会反转
        # （260927 回放：弯2/3 精修 6 步全部"横条反侧回摆"，与主转方向相反，
        # 净转角归零导致走过弯），不再用 offset_x 判向。只按查表方向保守
        # 小步续转 CORNER_FINE_BAR_MAX_STEPS 次，用尽后收尾交给转弯后确认期
        # （竖线 heading 闭环，或前进出弯）。
        if session.turn_bar_steps >= CORNER_FINE_BAR_MAX_STEPS:
            print('V3 转弯精修：第{}步 横条持续、竖线未现，停止精修'.format(
                session.turn_steps))
            _finish_turn(session, now, decision)
            return decision
        session.turn_bar_steps += 1
        session.turn_steps += 1
        session.turn_fine_remaining -= 1
        decision.actions = [(small_turn_action_for(session.turn_corner),
                             1, True, TURN_FINE_STEP_SETTLE)]
        decision.label = '转弯精修第{}步（横条：同向小步）'.format(session.turn_steps)
        return decision
    else:
        # 无横条无竖线：可能转过或短暂丢线，停止精修。
        print('V3 转弯精修：第{}步 无线停止 span={:.0f} raw_h={:.1f}'.format(
            session.turn_steps, span, raw_h))
        _finish_turn(session, now, decision)
        return decision

    # 小步→大步升级：连续 FINE_SMALL_STREAK_ESCALATE 次小步、误差改善不足
    # FINE_SMALL_PROGRESS_MIN 时，本轮升级为同方向大步一次，再回到小步。
    if small_action is not None:
        if session.turn_small_streak == 0:
            session.turn_streak_base_error = error
            session.turn_small_streak = 1
            step_action = small_action
        elif session.turn_streak_base_error - error < FINE_SMALL_PROGRESS_MIN:
            step_action = big_action
            print('V3 转弯精修：第{}步 小步改善不足升级大步 err={:.1f}->{:.1f}'.format(
                session.turn_steps, session.turn_streak_base_error, error))
            session.turn_small_streak = 0
        else:
            step_action = small_action
            session.turn_small_streak = 0

    session.turn_steps += 1
    session.turn_fine_remaining -= 1
    decision.actions = [(step_action, 1, True, TURN_FINE_STEP_SETTLE)]
    decision.label = '转弯精修第{}步'.format(session.turn_steps)
    return decision


def decide_action(state, session, now):
    """纯决策：根据当前视觉状态与会话状态，返回本轮动作计划。

    与真机 move() 执行器配套使用，不直接驱动任何硬件；离线模拟器可传入
    虚拟时间逐帧调用，决策优先级链与打印日志和重构前 move() 完全一致。
    """
    global turn_started, search_active
    decision = Decision()

    # 转弯进行中：先执行转弯前前进（仅查表指定的弯，横条驱动、步数自适应），
    # 再一次性下发主转，之后逐帧精修。整个转弯过程不受"帧未更新"去重与
    # 丢线分支影响。
    if session.turn_active:
        if session.turn_pre_forward_remaining > 0:
            # 横条（或 corner_pending 锁存内）仍在视野：继续前进接近路口。
            if state.get('horizontal_corner') or state.get('corner_pending'):
                session.turn_pre_forward_remaining -= 1
                decision.actions = [(FORWARD_ACTION, 1, False, NORMAL_ACTION_SETTLE)]
                decision.label = '转弯前前进（横条仍在，最多再走{}步）'.format(
                    session.turn_pre_forward_remaining)
                return decision
            # 横条消失：已走到路口上方，本轮直接转入主转。
            session.turn_pre_forward_remaining = 0
        if not session.turn_main_issued:
            session.turn_main_issued = True
            session.turn_steps = session.turn_main_steps_total
            decision.actions = [
                (turn_action_for(session.turn_corner), 1, True,
                 TURN_MAIN_STEP_SETTLE)
            ] * session.turn_main_steps_total
            decision.label = '路口{}向{}转（主转{}步）'.format(
                session.corner_index + 1,
                direction_name(session.turn_corner),
                session.turn_main_steps_total)
            return decision
        return _decide_turn_fine_step(state, session, now, decision)

    if state['visible']:
        session.no_line_since = 0
        session.ever_seen_line = True
        session.last_visible_state = state
        if session.search.active or session.search.exhausted:
            print('V3 丢线搜索：重新发现红线，恢复巡线')
            search_active = False
            clear_pending_corner()
            session.search.reset()
            session.search_exhausted_reported = False
            session.last_handled_frame = state['frame_id']
            decision.wait = 0.05
            decision.label = '重新发现红线'
            return decision

        if state['frame_id'] == session.last_handled_frame:
            decision.wait = 0.01
            return decision
        session.last_handled_frame = state['frame_id']

        # 横条退场门限（防残留横条串下一弯）：连续无横条达
        # CORNER_BAR_FREE_REARM_FRAMES 帧后重新允许触发路口。
        if state['horizontal_corner']:
            session.bar_free_frames = 0
        else:
            session.bar_free_frames += 1
        if (not session.corner_rearmed and
                session.bar_free_frames >= CORNER_BAR_FREE_REARM_FRAMES):
            session.corner_rearmed = True
            print('V3 巡线：横条已离开视野，允许触发下一路口')

        # 路口滞留保护计时（M8）：正常循迹中（非转弯/非确认期）横条持续
        # 可见的起始时间；横条退场或进入转弯流程即重置。
        if (state['horizontal_corner'] and not session.turn_active
                and not session.post_turn_active):
            if session.bar_dwell_start == 0:
                session.bar_dwell_start = now
        else:
            session.bar_dwell_start = 0.0
            session.bar_dwell_swings = 0

        center_error = state['near_x'] - IMAGE_CENTER_X
        heading = state['heading']
        action = None
        action_label = ''
        turn_direction = 0
        turn_error = 0

        # 转弯后确认期：转弯刚结束画面里可能仍残留横条（corner_ready 持续
        # True），直接响应会把残留横条误判成下一个路口导致连续转弯。确认期
        # 内屏蔽 corner_ready：横条残留说明转弯没转够、朝原方向小步微调；
        # 横条消失（面对直道）连续数帧后才恢复前进与路口检测。
        corner_ready_flag = state['corner_ready']
        if session.post_turn_active:
            corner_ready_flag = False
            if state['horizontal_corner']:
                session.post_turn_clear_frames = 0
                span = state.get('vertical_span', 0.0)
                raw_h = state.get('raw_heading', 0.0)
                if span >= CORNER_MIN_VERTICAL_SPAN:
                    # 竖线可见：heading 判向微调（唯一可信的判向通道）。
                    if abs(raw_h) <= CORNER_EXIT_HEADING_PX:
                        session.post_turn_active = False
                        clear_pending_corner()
                        print('V3 巡线：转弯后竖线正对，恢复正常循迹')
                        decision.label = '转弯后竖线正对'
                        return decision
                    if session.post_turn_extra_turns >= POST_TURN_MAX_EXTRA_TURNS:
                        session.post_turn_active = False
                        print('V3 巡线：转弯后确认超次，恢复正常循迹')
                    else:
                        session.post_turn_extra_turns += 1
                        adjust_direction = (session.post_turn_direction
                                            if session.post_turn_direction * raw_h > 0
                                            else -session.post_turn_direction)
                        action = small_turn_action_for(adjust_direction)
                        action_label = '转弯确认微调向{}（{}/{}）'.format(
                            direction_name(adjust_direction),
                            session.post_turn_extra_turns, POST_TURN_MAX_EXTRA_TURNS)
                        if action_label != session.last_action_label:
                            print('V3 巡线：{}'.format(action_label))
                            session.last_action_label = action_label
                        decision.actions = [(action, TURN_ACTION_TIMES, True,
                                             NORMAL_ACTION_SETTLE)]
                        decision.label = action_label
                        return decision
                else:
                    # 横条残留、竖线未现：不再按 offset_x 判向回摆（偏姿态
                    # 符号会反转，260927 回放中弯1后 132 次原地微调即源于
                    # 此），先前进驶出路口（横条随前进自然退场）；仍不退场
                    # 则做少量反向回摆小步，把新直道带回视野（260927 实车：
                    # 右转过转约 20° 时竖线永不出现、机器人在路口滞留约
                    # 20s 后侧冲出赛道）。前进与回摆都有上限，防止盲目直行。
                    if session.post_turn_extra_turns >= POST_TURN_MAX_EXTRA_FORWARDS:
                        if session.post_turn_swing_done < POST_TURN_MAX_EXTRA_SWINGS:
                            session.post_turn_swing_done += 1
                            action = small_turn_action_for(-session.post_turn_direction)
                            action_label = '转弯后横条不退场，反向回摆（{}/{}）'.format(
                                session.post_turn_swing_done, POST_TURN_MAX_EXTRA_SWINGS)
                            decision.actions = [(action, 1, True, NORMAL_ACTION_SETTLE)]
                            decision.label = action_label
                            return decision
                        session.post_turn_active = False
                        print('V3 巡线：转弯后确认超次，恢复正常循迹')
                    else:
                        session.post_turn_extra_turns += 1
                        action = FORWARD_ACTION
                        action_label = '转弯后横条残留，前进出弯（{}/{}）'.format(
                            session.post_turn_extra_turns, POST_TURN_MAX_EXTRA_FORWARDS)
                        decision.actions = [(action, 1, False, NORMAL_ACTION_SETTLE)]
                        decision.label = action_label
                        return decision
            else:
                session.post_turn_clear_frames += 1
                if session.post_turn_clear_frames >= POST_TURN_CLEAR_FRAMES:
                    session.post_turn_active = False
                    clear_pending_corner()
                    print('V3 巡线：转弯后确认完成，恢复正常循迹')

        # 路口触发条件：corner_ready、当前帧真看到横条（排除 corner_pending
        # 锁存把已消失横条"记忆"成路口），转完弯后已前进/横移离开路口至少
        # POST_TURN_MIN_FORWARD_STEPS 次，横条自上一弯起连续退场过
        # （corner_rearmed），且近端线已居中（CORNER_TRIGGER_CENTER_DEADBAND：
        # 偏姿态下横条深度信号偏早，弯2/3/4 触发时 nx=+20~33px 而弯1 仅
        # ±4px，居中门限把支点推回路口）。
        if (corner_ready_flag and state['horizontal_corner']
                and session.post_turn_progress_steps >= POST_TURN_MIN_FORWARD_STEPS
                and session.corner_rearmed
                and abs(center_error) <= CORNER_TRIGGER_CENTER_DEADBAND):
            corner_turn = expected_corner_direction(session.corner_index)
            visual_dir = state.get('corner_direction', 0)
            offset_x = state.get('corner_offset_x', 0.0)
            if corner_turn != 0:
                if visual_dir != 0 and visual_dir != corner_turn:
                    # 偏姿态下横条中点偏移符号会反转（260927 回放：右转的
                    # 弯2/3 全程判左），视觉判向只做记录，不再回退/覆盖
                    # 顺序表，避免朝错误方向转（回放中曾致 ci 回退 10 次）。
                    print('V3 方向校验：视觉判向{} 与赛道顺序{} 不一致（offset={:.0f}），忽略视觉判向'.format(
                        direction_name(visual_dir), direction_name(corner_turn),
                        offset_x))
                action_label = '路口{}按顺序执行向{}转 corner_y={:.0f}'.format(
                    session.corner_index + 1, direction_name(corner_turn),
                    state['corner_y'])
                if action_label != session.last_action_label:
                    print('V3 巡线：{}'.format(action_label))
                    session.last_action_label = action_label
                turn_started = True
                # 转弯两阶段：主转（固定步数开环，接近 90°）→ 精修（竖线
                # 正对即停）。转弯中间新直道被误判成横条、heading 不可用，
                # 故主转只看标定角度；精修阶段竖线竖直、heading 可靠。
                # 第 2/3/4 弯在转弯前先前进（横条驱动：走到横条消失为止，
                # 步数自适应），把转身支点推进到路口（横条深度信号被身体
                # 遮挡时会偏早触发）。主转与前进由转弯路由逐帧下发，精修
                # 由后续帧反复调用推进。
                main_steps = (CORNER_MAIN_STEPS_LEFT if corner_turn < 0
                              else CORNER_MAIN_STEPS_RIGHT)
                session.turn_active = True
                session.turn_corner = corner_turn
                session.turn_steps = 0
                session.turn_main_steps_total = main_steps
                session.turn_fine_remaining = CORNER_FINE_MAX_STEPS
                session.turn_main_issued = False
                session.turn_small_streak = 0
                session.turn_streak_base_error = 0.0
                session.turn_bar_steps = 0
                session.turn_pre_forward_remaining = (
                    CORNER_PRE_TURN_FORWARD_MAX_STEPS
                    if (session.corner_index < len(CORNER_PRE_TURN_FORWARD)
                        and CORNER_PRE_TURN_FORWARD[session.corner_index])
                    else 0)
                decision.label = '路口{}向{}转（转弯前横条驱动前进）'.format(
                    session.corner_index + 1, direction_name(corner_turn))
                return decision
            else:
                # 转弯顺序表已用完（到终点），不再理会路口特征，继续直行。
                action = FORWARD_ACTION
                action_label = '路口顺序已完成，直行'
        elif state['horizontal_corner']:
            # 路口滞留保护（M8）：横条长时间不退场说明转弯后机器人仍面对
            # 路口（典型过转），此时只横移/前进永远出不去（260927 实车滞留
            # 约 20s）。做少量反向回摆小步把新直道带回视野；竖线出现后由
            # 正常循迹/转弯逻辑接管。
            if (session.bar_dwell_start
                    and now - session.bar_dwell_start >= CORNER_BAR_DWELL_TIMEOUT
                    and session.post_turn_direction != 0
                    and session.bar_dwell_swings < CORNER_BAR_DWELL_SWINGS):
                session.bar_dwell_swings += 1
                action = small_turn_action_for(-session.post_turn_direction)
                action_label = '路口滞留回摆（{}/{}）'.format(
                    session.bar_dwell_swings, CORNER_BAR_DWELL_SWINGS)
                if action_label != session.last_action_label:
                    print('V3 巡线：{}'.format(action_label))
                    session.last_action_label = action_label
                decision.actions = [(action, 1, True, NORMAL_ACTION_SETTLE)]
                decision.label = action_label
                return decision
            # 横条存在（接近路口或转弯后残留）：heading 被横条干扰，转向
            # 修正会原地打转。改为横向对中：红线中心偏右→右移、偏左→左移；
            # 已对中则前进（接近路口或走出路口），让横条自然变化。
            # 接近路口（corner_ready）时用更紧的 CORNER_TRIGGER_CENTER_DEADBAND，
            # 先把支点摆正再触发，避免弯2/3/4 偏姿态提前转。
            band = (CORNER_TRIGGER_CENTER_DEADBAND if corner_ready_flag
                    else CENTER_DEADBAND)
            if abs(center_error) >= band:
                action = (LATERAL_RIGHT_ACTION if center_error > 0
                          else LATERAL_LEFT_ACTION)
                action_label = '路口横移向{}'.format(
                    direction_name(1 if center_error > 0 else -1))
            else:
                action = FORWARD_ACTION
                action_label = '路口前进'
        elif abs(heading) >= HEADING_TURN_THRESHOLD:
            path_direction = 1 if heading > 0 else -1
            turn_direction = path_direction
            turn_error = abs(heading)
            action_label = '实时方向修正执行向{}'.format(
                direction_name(turn_direction))
        elif state['corner_pending']:
            action = FORWARD_ACTION
            action_label = '接近下一路口'
        elif (abs(center_error) >= CENTER_DEADBAND and
              abs(heading) <= VERTICAL_HEADING_TOLERANCE):
            action = (LATERAL_RIGHT_ACTION if center_error > 0
                      else LATERAL_LEFT_ACTION)
            action_label = '横移向{}'.format(
                direction_name(1 if center_error > 0 else -1))
        else:
            action = FORWARD_ACTION
            action_label = '前进'

        if turn_direction != 0:
            if (turn_direction == session.last_turn_direction and
                    session.last_turn_error is not None and
                    turn_error >= session.last_turn_error - TURN_PROGRESS_MIN):
                session.no_progress_turns += 1
            else:
                session.no_progress_turns = 0
            session.last_turn_direction = turn_direction
            session.last_turn_error = turn_error

            if session.no_progress_turns >= MAX_NO_PROGRESS_TURNS:
                if not session.turn_hold_reported:
                    print('V3 巡线：连续转向未改善，暂停动作等待视觉变化')
                    session.turn_hold_reported = True
                decision.wait = 0.05
                decision.label = '连续转向未改善，暂停'
                return decision

            # 直行时的实时方向微调用小转角动作，避免转角过大反复过冲
            action = small_turn_action_for(turn_direction)
        else:
            session.last_turn_direction = 0
            session.last_turn_error = None
            session.no_progress_turns = 0
            session.turn_hold_reported = False

        if action_label != session.last_action_label:
            print('V3 巡线：{} center={:.0f} heading={:.1f}'.format(
                action_label, center_error, heading))
            session.last_action_label = action_label
        if turn_direction != 0:
            decision.actions = [(action, TURN_ACTION_TIMES, True,
                                 NORMAL_ACTION_SETTLE)]
        else:
            decision.actions = [(action, 1, False, NORMAL_ACTION_SETTLE)]
            # 前进/横移计入转弯后前进进度（原地小步微调不计入）。
            if action in (FORWARD_ACTION, LATERAL_LEFT_ACTION,
                          LATERAL_RIGHT_ACTION):
                if session.post_turn_progress_steps < POST_TURN_MIN_FORWARD_STEPS:
                    session.post_turn_progress_steps += 1
        decision.label = action_label
        return decision

    # —— 丢线分支 ——
    # 摄像头首帧到达前不搜索，防止启动站立动作期间机器人先转一步。
    if not session.ever_seen_line:
        decision.wait = 0.02
        return decision

    lost_from = state['last_seen_time'] or session.last_visible_state['last_seen_time']
    if lost_from == 0:
        if session.no_line_since == 0:
            session.no_line_since = now
        lost_from = session.no_line_since
    if now - lost_from < LOST_CONFIRM_SECONDS:
        decision.wait = 0.02
        return decision

    preferred_direction, source = choose_search_direction(
        session.last_visible_state, session.corner_index)
    search_direction, phase, count, limit = session.search.next_turn(
        preferred_direction)
    if search_direction == 0:
        if not session.search_exhausted_reported:
            print('V3 丢线搜索：正向 3 次、反向 6 次均未找到红线，停止等待')
            session.search_exhausted_reported = True
        search_active = False
        decision.wait = 0.05
        return decision

    action = turn_action_for(search_direction)
    print('V3 丢线搜索：{}阶段向{}转 {}/{}（依据：{}）'.format(
        phase, direction_name(search_direction), count, limit, source))
    search_active = True
    decision.actions = [(action, TURN_ACTION_TIMES, True, 0.0)]
    decision.observe = True
    decision.search_count = count
    decision.label = '丢线搜索：{}阶段向{}转 {}/{}'.format(
        phase, direction_name(search_direction), count, limit)
    return decision


def move():
    """运动执行器：循环读取视觉状态 → decide_action 决策 → 执行动作。

    决策链本体在 decide_action（纯函数），此处只负责硬件执行与睡眠节奏，
    与重构前一致：每条动作后 NORMAL_ACTION_SETTLE 睡眠、无进展 0.05s、
    帧未更新 0.01s、暂停 0.1s、丢线搜索动作后按 SEARCH_OBSERVE_SECONDS 观察。
    """
    global turn_started
    session = PatrolSession()
    while True:
        if not (enter and running):
            session.reset(keep_corner_index=True)
            turn_started = False
            time.sleep(0.1)
            continue

        state = get_vision_state()
        decision = decide_action(state, session, time.monotonic())
        if decision.wait:
            time.sleep(decision.wait)
            continue
        for action_name, times, with_stand, settle in decision.actions:
            if with_stand:
                AGC.runActionGroup(action_name, times=times,
                                   with_stand=TURN_WITH_STAND)
            else:
                AGC.runActionGroup(action_name, times=times)
            if settle:
                time.sleep(settle)
        if decision.observe:
            if observe_for_line(SEARCH_OBSERVE_SECONDS):
                print('V3 丢线搜索：第 {} 次转向后发现红线'.format(
                    decision.search_count))
        if decision.settle_after:
            time.sleep(decision.settle_after)


if HARDWARE_AVAILABLE:
    motion_thread = threading.Thread(target=move)
    motion_thread.daemon = True
    motion_thread.start()
else:
    motion_thread = None


_save_debug_warned = False


def _save_debug_frame(frame):
    """把当前校正帧写入 DEBUG_SAVE_DIR（调试用；异常只告警一次）。"""
    global _save_debug_warned
    try:
        if not os.path.isdir(DEBUG_SAVE_DIR):
            os.makedirs(DEBUG_SAVE_DIR)
        path = os.path.join(DEBUG_SAVE_DIR, 'patrol_%06d.jpg' % _frame_id)
        cv2.imwrite(path, frame)
    except Exception as exc:
        if not _save_debug_warned:
            print('RedLinePatrolV3：调试帧保存失败（只告警一次）：{}'.format(exc),
                  flush=True)
            _save_debug_warned = True


def run(img, now=None):
    """视觉分析主入口。now 为时间戳（默认真实单调时间）。

    离线模拟器按"帧号/fps"传入虚拟时间，使 corner_pending 锁存等时间逻辑
    可确定性回放；机器人上不传参，行为与之前完全一致。
    """
    global _vision_state, _frame_id
    global _corner_pending, _corner_pending_y, _corner_pending_last_seen
    global _corner_direction, _corner_offset_x
    global _last_corner_debug_time
    global line_center_x, line_lost_time, approach_active, corner_ready

    if now is None:
        now = time.monotonic()

    if not enter:
        return img

    source_height, source_width = img.shape[:2]
    frame = cv2.resize(img, FRAME_SIZE, interpolation=cv2.INTER_NEAREST)
    display = frame.copy()
    mask = build_red_mask(frame)

    with _state_lock:
        reference_x = (_filtered_near_x if _filtered_near_x is not None
                       else IMAGE_CENTER_X)

    contour, contour_area = select_path_contour(mask, reference_x)
    samples = []
    analysis = None
    if contour is not None:
        path_mask = np.zeros_like(mask)
        cv2.drawContours(path_mask, [contour], -1, 255, thickness=-1)
        samples = sample_path(path_mask)
        analysis = analyze_path_samples(samples)
        cv2.drawContours(display, [contour], -1, (0, 0, 255), 2)

    global _last_corner_debug_time
    if analysis is not None and now - _last_corner_debug_time >= CORNER_DEBUG_INTERVAL:
        _last_corner_debug_time = now
        print('V3 路口调试：max_width={:.0f}@y={:.0f} horizontal_seen={} corner_y={:.0f} dir={} offset={:.0f}'.format(
            analysis['max_width'], analysis['max_width_y'],
            analysis['horizontal_corner'], analysis['corner_y'],
            analysis['corner_direction'], analysis['corner_offset_x']))

    with _state_lock:
        _frame_id += 1
        previous_last_seen = _vision_state['last_seen_time']

        if analysis is not None:
            near_x, heading = filter_measurements(
                analysis['near_x'], analysis['heading'])
            if analysis['horizontal_corner']:
                _corner_pending = True
                _corner_pending_y = max(_corner_pending_y, analysis['corner_y'])
                _corner_pending_last_seen = now
                if analysis['corner_direction'] != 0:
                    _corner_direction = analysis['corner_direction']
                    _corner_offset_x = analysis['corner_offset_x']
            elif (_corner_pending and
                  now - _corner_pending_last_seen > CORNER_LATCH_SECONDS):
                _corner_pending = False
                _corner_pending_y = 0
                _corner_pending_last_seen = 0
                _corner_direction = 0
                _corner_offset_x = 0.0

            corner_ready = analysis['corner_ready'] or (
                _corner_pending and _corner_pending_y >= CORNER_TURN_TRIGGER_Y)
            confidence = min(1.0, contour_area / 2500.0)
            confidence *= min(1.0, analysis['sample_count'] / 3.0)
            line_center_x = int(round(near_x))
            line_lost_time = 0
            approach_active = bool(_corner_pending)
            _vision_state = {
                'frame_id': _frame_id,
                'timestamp': now,
                'visible': True,
                'last_seen_time': now,
                'near_x': near_x,
                'near_y': analysis['near_y'],
                'heading': heading,
                'raw_heading': analysis['raw_heading'],
                'vertical_span': analysis['vertical_span'],
                # 路口特征确认后保持接近状态，短暂漏检也不恢复横移。
                'horizontal_corner': analysis['horizontal_corner'],
                'corner_pending': _corner_pending,
                'corner_y': (_corner_pending_y if _corner_pending
                             else analysis['corner_y']),
                'corner_ready': corner_ready,
                'corner_direction': _corner_direction,
                'corner_offset_x': _corner_offset_x,
                'sample_count': analysis['sample_count'],
                'confidence': confidence,
                # 调试可视化扩展字段（decide_action 不使用，仅供离线模拟器绘制）：
                'samples': samples,
                'max_width': analysis['max_width'],
                'max_width_y': analysis['max_width_y'],
                'max_width_left_x': analysis['max_width_left_x'],
                'max_width_right_x': analysis['max_width_right_x'],
                'max_width_fill_ratio': analysis['max_width_fill_ratio'],
            }
        else:
            if line_center_x != -1:
                line_lost_time = now
            line_center_x = -1
            approach_active = bool(_corner_pending)
            corner_ready = False
            _vision_state = dict(_vision_state)
            _vision_state.update({
                'frame_id': _frame_id,
                'timestamp': now,
                'visible': False,
                'last_seen_time': previous_last_seen,
            })
        state_for_display = dict(_vision_state)

    for sample_y, sample_x, _, _, _ in samples:
        cv2.circle(display, (int(sample_x), int(sample_y)), 5, (0, 255, 255), -1)

    if state_for_display['visible']:
        near_point = (int(state_for_display['near_x']),
                      int(state_for_display['near_y']))
        cv2.circle(display, near_point, 9, (0, 255, 0), -1)

    cv2.line(display, (IMAGE_CENTER_X, VISION_TOP_Y),
             (IMAGE_CENTER_X, FRAME_SIZE[1] - 1), (255, 0, 0), 1)

    if SAVE_DEBUG_FRAMES and _frame_id % DEBUG_SAVE_EVERY_N == 0:
        _save_debug_frame(frame)

    if (source_width, source_height) != FRAME_SIZE:
        display = cv2.resize(display, (source_width, source_height),
                             interpolation=cv2.INTER_LINEAR)
    return display


if __name__ == '__main__':
    if not HARDWARE_AVAILABLE:
        print('RedLinePatrolV3：离线模式无法独立运行相机循环，请改用 '
              'test_programs_NoUseInMain/redline_route_debug.py 离线复盘')
        sys.exit(0)
    param_data = np.load(calibration_param_path + '.npz')
    mtx = param_data['mtx_array']
    dist = param_data['dist_array']
    newcameramtx, _ = cv2.getOptimalNewCameraMatrix(
        mtx, dist, FRAME_SIZE, 0, FRAME_SIZE)
    mapx, mapy = cv2.initUndistortRectifyMap(
        mtx, dist, None, newcameramtx, FRAME_SIZE, 5)

    init()
    start()

    open_once = yaml_handle.get_yaml_data('/boot/camera_setting.yaml')['open_once']
    if open_once:
        camera = cv2.VideoCapture(
            'http://127.0.0.1:8080/?action=stream?dummy=param.mjpg')
    else:
        camera = Camera.Camera()
        camera.camera_open()

    AGC.runActionGroup('stand')
    print('红线巡线 V3 已启动，按 Ctrl+C 退出')

    try:
        while True:
            ret, image = camera.read()
            if not ret:
                time.sleep(0.01)
                continue
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            display_image = run(image)
            if SHOW_DISPLAY:
                cv2.imshow('RedLinePatrolV3', display_image)
                if cv2.waitKey(1) == 27:
                    break
            else:
                time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        exit()
        if hasattr(camera, 'camera_close'):
            camera.camera_close()
        else:
            camera.release()
        if SHOW_DISPLAY:
            cv2.destroyAllWindows()
