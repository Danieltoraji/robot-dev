#!/usr/bin/python3
# coding=utf8
"""红线巡线 V3：赛道固定为直角弯，转向顺序查表（CORNER_TURN_SEQUENCE），
横移纠偏、路口接近锁存和有界丢线搜索。

V3 在 V2 算法基础上补充了 demoV4.py / patrol_end_recovery.py 期望的
总控接口：line_center_x、line_lost_time、turn_started、approach_active、
detect_red_line()、roi、MIN_CONTOUR_AREA、calibration_param_path。
算法与参数与 V2 保持一致。
"""

import sys
import cv2
import time
import math
import threading
import numpy as np

import hiwonder.Camera as Camera
import hiwonder.ros_robot_controller_sdk as rrc
from hiwonder.Controller import Controller
import hiwonder.ActionGroupControl as AGC
import hiwonder.yaml_handle as yaml_handle

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
POST_TURN_MAX_EXTRA_TURNS = 12    # 横条残留时最多额外小步微调次数（防卡死）

# 赛道固定，转弯顺序提前写死（从起点开始，依次对应每个路口的转向）
CORNER_TURN_SEQUENCE = ['left', 'right', 'right', 'left']

# 转弯采用「主转固定步数 + 精修」两阶段：
#   主转按标定角度（左 22°/次、右 25.7°/次）转到接近 90°——左 4 步≈88°、
#   右 3 步≈77°。转弯中间新直道在画面里倾斜、被采样带误判成横条（宽>90），
#   direction_samples 为空、heading 无法计算，所以主转阶段不看视觉、只按
#   标定角度开环转固定步数，避免像旧版那样用「横条消失/竖线 heading」在转弯
#   中间判断失效而原地打转 308°。
#   精修阶段转弯已基本到位：横条还在→继续转大步兜底；竖线出现→按 heading
#   符号闭环（未转够续转大步、转过反向小步回摆），直到竖线正对。
CORNER_MAIN_STEPS_LEFT = 4        # 左转主转步数：4×22°=88°
CORNER_MAIN_STEPS_RIGHT = 4       # 右转主转步数：4×25.7°≈103°。曾设 3（77°）：机器人
                                  # 横向略偏（offset≈50）时，3 步后横条仍 180px 宽不消失、
                                  # 竖线不出现，精修 6 步全浪费在「横条续转」大步上，累计
                                  # 231° 过冲。4 步 103° 更稳地跳过转弯中间（弯1 左转 4 步
                                  # 88° 已验证竖线可靠出现）。
CORNER_FINE_MAX_STEPS = 6         # 精修阶段最多步数（未转够用大步，横条残留与转过用小步）
CORNER_EXIT_HEADING_PX = 20.0     # 转弯完成判据：前方竖线 heading 绝对值 ≤ 此值视为已对准新方向
                                  # （正常直线循迹 heading 在 ±20px 内，精细对齐交给后续循迹）。
CORNER_MIN_VERTICAL_SPAN = 40.0   # 前方竖线存在的判据：direction_samples 的垂直跨度需 ≥ 此值，
                                  # 否则说明竖线尚未出现（横条主导），继续按原方向大步转。

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
# ==================================

# V3 总控接口：脚下红线轮廓最小面积（与 demoV4.py PatrolEndDetector 配合）。
MIN_CONTOUR_AREA = 50

# V3 总控接口：脚下近处 ROI。demoV4.py 取 roi[-1] 作为脚下区域，
# 复用 V2 的 VISION_SIDE_CROP 左右裁剪，底部到 480 即画面最下方。
roi = (
    (430, 480, VISION_SIDE_CROP, FRAME_SIZE[0] - VISION_SIDE_CROP),
)


board = rrc.Board()
ctl = Controller(board)
servo_data = None

enter = False
running = False

# V3 总控接口：当前红线中心 x（-1 表示丢线）、丢线起始时间、直角弯状态。
line_center_x = -1
line_lost_time = 0
turn_started = False
approach_active = False
corner_ready = False
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
    for sample_y, _, left_x, right_x, count in samples:
        width = right_x - left_x
        if width > max_width:
            max_width = width
            max_width_y = sample_y
            max_width_left_x = left_x
            max_width_right_x = right_x
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


def move():
    global turn_started, last_turn_completed_at
    last_handled_frame = -1
    last_visible_state = empty_vision_state()
    search = LostLineSearch()
    search_exhausted_reported = False
    last_action_label = ''
    last_turn_direction = 0
    last_turn_error = None
    no_progress_turns = 0
    turn_hold_reported = False
    no_line_since = 0
    ever_seen_line = False
    post_turn_active = False
    post_turn_direction = 0
    post_turn_clear_frames = 0
    post_turn_extra_turns = 0

    while True:
        if not (enter and running):
            search.reset()
            search_exhausted_reported = False
            last_handled_frame = -1
            last_turn_direction = 0
            last_turn_error = None
            no_progress_turns = 0
            turn_hold_reported = False
            no_line_since = 0
            ever_seen_line = False
            post_turn_active = False
            post_turn_direction = 0
            post_turn_clear_frames = 0
            post_turn_extra_turns = 0
            last_visible_state = empty_vision_state()
            turn_started = False
            time.sleep(0.1)
            continue

        state = get_vision_state()
        if state['visible']:
            no_line_since = 0
            ever_seen_line = True
            last_visible_state = state
            if search.active or search.exhausted:
                print('V3 丢线搜索：重新发现红线，恢复巡线')
                clear_pending_corner()
                search.reset()
                search_exhausted_reported = False
                last_handled_frame = state['frame_id']
                time.sleep(0.05)
                continue

            if state['frame_id'] == last_handled_frame:
                time.sleep(0.01)
                continue
            last_handled_frame = state['frame_id']

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
            if post_turn_active:
                corner_ready_flag = False
                if state['horizontal_corner']:
                    post_turn_clear_frames = 0
                    if post_turn_extra_turns >= POST_TURN_MAX_EXTRA_TURNS:
                        post_turn_active = False
                        print('V3 巡线：转弯后确认超次，恢复正常循迹')
                    else:
                        post_turn_extra_turns += 1
                        # 转弯后横条残留：竖线可见时用 heading 判向；竖线不可见
                        # （转弯中间）时用横条中点偏移 offset_x 判向，避免盲目
                        # 前进走出路口（弯2 曾因此前进 4 步走过横条后丢线）。
                        #   corner_turn*(offset_x)>0 → 没转够 → 续转
                        #   corner_turn*(offset_x)<0 → 转过 → 回摆
                        span = state.get('vertical_span', 0.0)
                        raw_h = state.get('raw_heading', 0.0)
                        offset_x = state.get('corner_offset_x', 0.0)
                        if (span >= CORNER_MIN_VERTICAL_SPAN and
                                abs(raw_h) <= CORNER_EXIT_HEADING_PX):
                            post_turn_active = False
                            clear_pending_corner()
                            print('V3 巡线：转弯后竖线正对，恢复正常循迹')
                            continue
                        if span >= CORNER_MIN_VERTICAL_SPAN:
                            # 竖线可见，用 heading 判向
                            adjust_direction = (post_turn_direction
                                                if post_turn_direction * raw_h > 0
                                                else -post_turn_direction)
                        else:
                            # 竖线不可见，用横条偏移判向（offset=0 保守续转）
                            adjust_direction = (post_turn_direction
                                                if post_turn_direction * offset_x >= 0
                                                else -post_turn_direction)
                        action = small_turn_action_for(adjust_direction)
                        action_label = '转弯确认微调向{}（{}/{}）'.format(
                            direction_name(adjust_direction),
                            post_turn_extra_turns, POST_TURN_MAX_EXTRA_TURNS)
                        if action_label != last_action_label:
                            print('V3 巡线：{}'.format(action_label))
                            last_action_label = action_label
                        AGC.runActionGroup(action, times=TURN_ACTION_TIMES,
                                           with_stand=TURN_WITH_STAND)
                        time.sleep(NORMAL_ACTION_SETTLE)
                        continue
                else:
                    post_turn_clear_frames += 1
                    if post_turn_clear_frames >= POST_TURN_CLEAR_FRAMES:
                        post_turn_active = False
                        clear_pending_corner()
                        print('V3 巡线：转弯后确认完成，恢复正常循迹')

            if corner_ready_flag:
                corner_index = get_corner_index()
                corner_turn = expected_corner_direction(corner_index)
                if corner_turn != 0:
                    visual_dir = state.get('corner_direction', 0)
                    if visual_dir != 0 and visual_dir != corner_turn:
                        print('V3 方向校验：视觉判向{} 与赛道顺序{} 不一致（offset={:.0f}）'.format(
                            direction_name(visual_dir), direction_name(corner_turn),
                            state.get('corner_offset_x', 0.0)))
                    action_label = '路口{}按顺序执行向{}转 corner_y={:.0f}'.format(
                        corner_index + 1, direction_name(corner_turn), state['corner_y'])
                    if action_label != last_action_label:
                        print('V3 巡线：{}'.format(action_label))
                        last_action_label = action_label
                    turn_started = True
                    # 转弯两阶段：主转（固定步数开环，接近 90°）→ 精修（竖线
                    # 正对即停）。转弯中间新直道被误判成横条、heading 不可用，
                    # 故主转只看标定角度；精修阶段竖线竖直、heading 可靠。
                    main_steps = (CORNER_MAIN_STEPS_LEFT if corner_turn < 0
                                  else CORNER_MAIN_STEPS_RIGHT)
                    turn_steps = 0
                    for _ in range(main_steps):
                        AGC.runActionGroup(
                            turn_action_for(corner_turn), times=1,
                            with_stand=TURN_WITH_STAND)
                        turn_steps += 1
                        time.sleep(NORMAL_ACTION_SETTLE)
                    for _ in range(CORNER_FINE_MAX_STEPS):
                        latest = get_vision_state()
                        vis = latest.get('visible', False)
                        span = latest.get('vertical_span', 0.0)
                        raw_h = latest.get('raw_heading', 0.0)
                        hc = latest.get('horizontal_corner', False)
                        # 完成：前方竖线存在且已正对。
                        if (vis and span >= CORNER_MIN_VERTICAL_SPAN and
                                abs(raw_h) <= CORNER_EXIT_HEADING_PX):
                            print('V3 转弯精修：第{}步 完成 span={:.0f} raw_h={:.1f}'.format(
                                turn_steps, span, raw_h))
                            break
                        if vis and span >= CORNER_MIN_VERTICAL_SPAN:
                            # 竖线出现但未正对。符号判据（与 line_seeker 的
                            # turn_sign*heading<0 等价，因 corner_turn 右=+1、
                            # 左=-1 恰与 turn_sign 相反）：
                            #   corner_turn*raw_h>0 → 未转够（线在转弯侧）→ 续转大步
                            #   corner_turn*raw_h<0 → 转过（线在反侧）→ 反向小步回摆
                            if corner_turn * raw_h > 0:
                                print('V3 转弯精修：第{}步 未转够续转 span={:.0f} raw_h={:.1f}'.format(
                                    turn_steps, span, raw_h))
                                AGC.runActionGroup(
                                    turn_action_for(corner_turn), times=1,
                                    with_stand=TURN_WITH_STAND)
                            else:
                                print('V3 转弯精修：第{}步 转过回摆 span={:.0f} raw_h={:.1f}'.format(
                                    turn_steps, span, raw_h))
                                AGC.runActionGroup(
                                    small_turn_action_for(-corner_turn), times=1,
                                    with_stand=TURN_WITH_STAND)
                        elif hc:
                            # 横条还在：转弯中间竖线不可见、heading 失效，改用
                            # 横条中点相对主线中心的偏移 offset_x 判向：
                            #   corner_turn*offset_x > 死区 → 没转够 → 续转大步
                            #   corner_turn*offset_x < -死区 → 转过 → 回摆小步
                            #   |offset_x| ≤ 死区 → 横条对中 → 小步续转（保守）
                            offset_x = latest.get('corner_offset_x', 0.0)
                            if corner_turn * offset_x > CORNER_DIRECTION_DEADBAND:
                                print('V3 转弯精修：第{}步 横条偏侧续转 offset={:.0f}'.format(
                                    turn_steps, offset_x))
                                AGC.runActionGroup(
                                    turn_action_for(corner_turn), times=1,
                                    with_stand=TURN_WITH_STAND)
                            elif corner_turn * offset_x < -CORNER_DIRECTION_DEADBAND:
                                print('V3 转弯精修：第{}步 横条反侧回摆 offset={:.0f}'.format(
                                    turn_steps, offset_x))
                                AGC.runActionGroup(
                                    small_turn_action_for(-corner_turn), times=1,
                                    with_stand=TURN_WITH_STAND)
                            else:
                                print('V3 转弯精修：第{}步 横条对中小步续转 offset={:.0f}'.format(
                                    turn_steps, offset_x))
                                AGC.runActionGroup(
                                    small_turn_action_for(corner_turn), times=1,
                                    with_stand=TURN_WITH_STAND)
                        else:
                            # 无横条无竖线：可能转过或短暂丢线，停止精修。
                            print('V3 转弯精修：第{}步 无线停止 span={:.0f} raw_h={:.1f}'.format(
                                turn_steps, span, raw_h))
                            break
                        turn_steps += 1
                        time.sleep(NORMAL_ACTION_SETTLE)
                    turn_started = False
                    last_turn_completed_at = time.monotonic()
                    print('V3 巡线：路口{}转弯{}步（主{}步+精修{}步）'.format(
                        corner_index + 1, turn_steps,
                        main_steps, turn_steps - main_steps))
                    post_turn_active = True
                    post_turn_direction = corner_turn
                    post_turn_clear_frames = 0
                    post_turn_extra_turns = 0
                    advance_corner_index()
                    clear_pending_corner()
                    last_turn_direction = 0
                    last_turn_error = None
                    no_progress_turns = 0
                    turn_hold_reported = False
                    time.sleep(NORMAL_ACTION_SETTLE)
                    continue
                else:
                    # 转弯顺序表已用完（到终点），不再理会路口特征，继续直行。
                    action = FORWARD_ACTION
                    action_label = '路口顺序已完成，直行'
            elif state['horizontal_corner']:
                # 横条存在（接近路口或转弯后残留）：heading 被横条干扰，转向
                # 修正会原地打转。改为横向对中：红线中心偏右→右移、偏左→左移；
                # 已对中则前进（接近路口或走出路口），让横条自然变化。
                if abs(center_error) >= CENTER_DEADBAND:
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
                action_label = '横移向{}'.format(direction_name(1 if center_error > 0 else -1))
            else:
                action = FORWARD_ACTION
                action_label = '前进'

            if turn_direction != 0:
                if (turn_direction == last_turn_direction and
                        last_turn_error is not None and
                        turn_error >= last_turn_error - TURN_PROGRESS_MIN):
                    no_progress_turns += 1
                else:
                    no_progress_turns = 0
                last_turn_direction = turn_direction
                last_turn_error = turn_error

                if no_progress_turns >= MAX_NO_PROGRESS_TURNS:
                    if not turn_hold_reported:
                        print('V3 巡线：连续转向未改善，暂停动作等待视觉变化')
                        turn_hold_reported = True
                    time.sleep(0.05)
                    continue

                # 直行时的实时方向微调用小转角动作，避免转角过大反复过冲
                action = small_turn_action_for(turn_direction)
            else:
                last_turn_direction = 0
                last_turn_error = None
                no_progress_turns = 0
                turn_hold_reported = False

            if action_label != last_action_label:
                print('V3 巡线：{} center={:.0f} heading={:.1f}'.format(
                    action_label, center_error, heading))
                last_action_label = action_label
            if turn_direction != 0:
                AGC.runActionGroup(
                    action, times=TURN_ACTION_TIMES,
                    with_stand=TURN_WITH_STAND)
            else:
                AGC.runActionGroup(action, times=1)
            time.sleep(NORMAL_ACTION_SETTLE)
            continue

        # 摄像头首帧到达前不搜索，防止启动站立动作期间机器人先转一步。
        if not ever_seen_line:
            time.sleep(0.02)
            continue

        lost_from = state['last_seen_time'] or last_visible_state['last_seen_time']
        if lost_from == 0:
            if no_line_since == 0:
                no_line_since = time.time()
            lost_from = no_line_since
        if time.time() - lost_from < LOST_CONFIRM_SECONDS:
            time.sleep(0.02)
            continue

        preferred_direction, source = choose_search_direction(
            last_visible_state, get_corner_index())
        search_direction, phase, count, limit = search.next_turn(preferred_direction)
        if search_direction == 0:
            if not search_exhausted_reported:
                print('V3 丢线搜索：正向 3 次、反向 6 次均未找到红线，停止等待')
                search_exhausted_reported = True
            time.sleep(0.05)
            continue

        action = turn_action_for(search_direction)
        print('V3 丢线搜索：{}阶段向{}转 {}/{}（依据：{}）'.format(
            phase, direction_name(search_direction), count, limit, source))
        AGC.runActionGroup(
            action, times=TURN_ACTION_TIMES,
            with_stand=TURN_WITH_STAND)
        if observe_for_line(SEARCH_OBSERVE_SECONDS):
            print('V3 丢线搜索：第 {} 次转向后发现红线'.format(count))


motion_thread = threading.Thread(target=move)
motion_thread.daemon = True
motion_thread.start()


def run(img):
    global _vision_state, _frame_id
    global _corner_pending, _corner_pending_y, _corner_pending_last_seen
    global _corner_direction, _corner_offset_x
    global _last_corner_debug_time
    global line_center_x, line_lost_time, approach_active, corner_ready

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

    now = time.time()
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

    if (source_width, source_height) != FRAME_SIZE:
        display = cv2.resize(display, (source_width, source_height),
                             interpolation=cv2.INTER_LINEAR)
    return display


if __name__ == '__main__':
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
