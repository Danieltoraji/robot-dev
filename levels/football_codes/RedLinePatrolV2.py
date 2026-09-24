#!/usr/bin/python3
# coding=utf8
"""红线巡线 V2：赛道固定为直角弯，转向顺序查表（CORNER_TURN_SEQUENCE），
横移纠偏、路口接近锁存和有界丢线搜索。"""

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

if __name__ == '__main__':
    from CameraCalibration.CalibrationConfig import *
else:
    from Functions.CameraCalibration.CalibrationConfig import *


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

# 正常巡线阈值
CENTER_DEADBAND = 42              # 近处红线离画面中心超过此值才横移
HEADING_TURN_THRESHOLD = 5        # 极缓弯也立即转向
HEADING_HINT_THRESHOLD = 25       # 丢线时用最后朝向猜搜索方向的阈值
VERTICAL_HEADING_TOLERANCE = 5    # 与转向阈值衔接，避免决策死区
FILTER_ALPHA = 0.5                # 位置和方向的一阶滤波系数
FILTER_RESET_JUMP = 140           # 位置突变过大时直接重置滤波

# 路口检测：赛道固定为直角弯，只需判断"是否到路口"，不再靠视觉猜方向
HORIZONTAL_MIN_WIDTH = 70         # 实测偏保守偏大导致漏检，调小以更容易触发
CORNER_TURN_TRIGGER_Y = 380       # 路口特征接近画面底部后才开始转身，调小以更早触发
CORNER_LATCH_SECONDS = 2.0        # 路口特征短暂漏检时继续保持"已到路口"状态
CORNER_DEBUG_INTERVAL = 0.5       # 调试打印节流：每隔多久打印一次检测到的最大宽度/行

# 赛道固定，转弯顺序提前写死（从起点开始，依次对应每个路口的转向）
CORNER_TURN_SEQUENCE = ['left', 'left', 'right', 'right', 'left', 'right']
CORNER_TURN_REPEATS = 3           # 单次90°整弯需要的 turn_left/turn_right 执行次数
                                  # 需用 action_test.py 实测单次转向角度后校准（90 / 单次角度）

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


board = rrc.Board()
ctl = Controller(board)
servo_data = None

enter = False
running = False

_state_lock = threading.Lock()
_vision_state = None
_frame_id = 0
_filtered_near_x = None
_filtered_heading = None
_corner_pending = False
_corner_pending_y = 0
_corner_pending_last_seen = 0
_corner_index = 0          # 下一个待完成的路口在 CORNER_TURN_SEQUENCE 中的下标
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
        'horizontal_corner': False,
        'corner_pending': False,
        'corner_y': 0,
        'corner_ready': False,
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

    with _state_lock:
        _frame_id = 0
        _vision_state = empty_vision_state()
        _filtered_near_x = None
        _filtered_heading = None
        _corner_pending = False
        _corner_pending_y = 0
        _corner_pending_last_seen = 0
        _corner_index = 0


def init():
    global enter
    load_config()
    initMove()
    reset()
    enter = True
    print('RedLinePatrolV2 Init')


def start():
    global running
    running = True
    print('RedLinePatrolV2 Start')


def stop():
    global running
    running = False
    reset()
    print('RedLinePatrolV2 Stop')


def exit():
    global enter, running
    enter = False
    running = False
    reset()
    AGC.runActionGroup('stand_slow')
    print('RedLinePatrolV2 Exit')


def build_red_mask(frame):
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
    """把采样点转换成近处位置、方向误差和路口提示。
    路口方向不再由视觉判断——赛道固定，方向来自 CORNER_TURN_SEQUENCE 查表。"""
    if not samples:
        return None

    near = max(samples, key=lambda sample: sample[0])
    direction_samples = [
        sample for sample in samples
        if sample[3] - sample[2] < HORIZONTAL_MIN_WIDTH
    ]
    if len(direction_samples) >= 2:
        direction_near = max(direction_samples, key=lambda sample: sample[0])
        direction_ahead = min(direction_samples, key=lambda sample: sample[0])
        vertical_span = direction_near[0] - direction_ahead[0]
        heading = ((direction_ahead[1] - direction_near[1]) * 200.0 /
                   vertical_span) if vertical_span >= 40 else 0.0
    else:
        vertical_span = 0.0
        heading = 0.0

    # 路口特征：某条采样带的红色区域宽度远超正常直线段宽度，说明这里出现了
    # 岔路（横向的红线段）。只需要知道"到没到"，不需要判断岔路方向。
    horizontal_seen = False
    corner_y = 0
    max_width = 0
    max_width_y = 0
    for sample_y, _, left_x, right_x, _ in samples:
        width = right_x - left_x
        if width > max_width:
            max_width = width
            max_width_y = sample_y
        if width >= HORIZONTAL_MIN_WIDTH:
            horizontal_seen = True
            corner_y = max(corner_y, sample_y)

    corner_ready = horizontal_seen and corner_y >= CORNER_TURN_TRIGGER_Y

    return {
        'near_x': near[1],
        'near_y': near[0],
        'heading': heading,
        'vertical_span': vertical_span,
        'horizontal_corner': horizontal_seen,
        'corner_y': corner_y,
        'corner_ready': corner_ready,
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

    with _state_lock:
        _corner_pending = False
        _corner_pending_y = 0
        _corner_pending_last_seen = 0
        _vision_state = dict(_vision_state)
        _vision_state['corner_pending'] = False
        _vision_state['corner_ready'] = False


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
            last_visible_state = empty_vision_state()
            time.sleep(0.1)
            continue

        state = get_vision_state()
        if state['visible']:
            no_line_since = 0
            ever_seen_line = True
            last_visible_state = state
            if search.active or search.exhausted:
                print('V2 丢线搜索：重新发现红线，恢复巡线')
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

            if state['corner_ready']:
                corner_index = get_corner_index()
                corner_turn = expected_corner_direction(corner_index)
                if corner_turn != 0:
                    action_label = '路口{}按顺序执行向{}转 corner_y={:.0f}'.format(
                        corner_index + 1, direction_name(corner_turn), state['corner_y'])
                    if action_label != last_action_label:
                        print('V2 巡线：{}'.format(action_label))
                        last_action_label = action_label
                    AGC.runActionGroup(
                        turn_action_for(corner_turn), times=CORNER_TURN_REPEATS,
                        with_stand=TURN_WITH_STAND)
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
                        print('V2 巡线：连续转向未改善，暂停动作等待视觉变化')
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
                print('V2 巡线：{} center={:.0f} heading={:.1f}'.format(
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
                print('V2 丢线搜索：正向 3 次、反向 6 次均未找到红线，停止等待')
                search_exhausted_reported = True
            time.sleep(0.05)
            continue

        action = turn_action_for(search_direction)
        print('V2 丢线搜索：{}阶段向{}转 {}/{}（依据：{}）'.format(
            phase, direction_name(search_direction), count, limit, source))
        AGC.runActionGroup(
            action, times=TURN_ACTION_TIMES,
            with_stand=TURN_WITH_STAND)
        if observe_for_line(SEARCH_OBSERVE_SECONDS):
            print('V2 丢线搜索：第 {} 次转向后发现红线'.format(count))


motion_thread = threading.Thread(target=move)
motion_thread.daemon = True
motion_thread.start()


def run(img):
    global _vision_state, _frame_id
    global _corner_pending, _corner_pending_y, _corner_pending_last_seen
    global _last_corner_debug_time

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
        print('V2 路口调试：max_width={:.0f}@y={:.0f} horizontal_seen={} corner_y={:.0f}'.format(
            analysis['max_width'], analysis['max_width_y'],
            analysis['horizontal_corner'], analysis['corner_y']))

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
            elif (_corner_pending and
                  now - _corner_pending_last_seen > CORNER_LATCH_SECONDS):
                _corner_pending = False
                _corner_pending_y = 0
                _corner_pending_last_seen = 0

            corner_ready = analysis['corner_ready'] or (
                _corner_pending and _corner_pending_y >= CORNER_TURN_TRIGGER_Y)
            confidence = min(1.0, contour_area / 2500.0)
            confidence *= min(1.0, analysis['sample_count'] / 3.0)
            _vision_state = {
                'frame_id': _frame_id,
                'timestamp': now,
                'visible': True,
                'last_seen_time': now,
                'near_x': near_x,
                'near_y': analysis['near_y'],
                'heading': heading,
                # 路口特征确认后保持接近状态，短暂漏检也不恢复横移。
                'horizontal_corner': analysis['horizontal_corner'],
                'corner_pending': _corner_pending,
                'corner_y': (_corner_pending_y if _corner_pending
                             else analysis['corner_y']),
                'corner_ready': corner_ready,
                'sample_count': analysis['sample_count'],
                'confidence': confidence,
            }
        else:
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
    print('红线巡线 V2 已启动，按 Ctrl+C 退出')

    try:
        while True:
            ret, image = camera.read()
            if not ret:
                time.sleep(0.01)
                continue
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            display_image = run(image)
            if SHOW_DISPLAY:
                cv2.imshow('RedLinePatrolV2', display_image)
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
