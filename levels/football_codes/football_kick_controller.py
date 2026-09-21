#!/usr/bin/python3
# coding=utf8
"""FootballKick 射门控制器。

本模块不导入 FootballKick.py 的完整主程序，避免导入时初始化相机、控制板
和后台线程。它保留 FootballKick.py 的动作组方向，但把流程拆成更安全的
非阻塞状态机：

    射门前球门线预对齐 -> 对准足球 -> 保守接近足球正后方
    -> 球门柱小幅辅助校正 -> 踢球

重要安全约束：
  * 足球检测不到时不继续前进，避免盲目前进把球踢跑；
  * 严格踢球距离使用足球检测框中心 cy 判断；检测框底部 bottom_y 仅用于接近阶段的分段速度参考；
  * 足球进入安全距离后不再前进，只允许必要的小幅姿态修正；
  * 球门柱只负责粗略辅助居中，足球位置不满足安全踢球条件时不会踢；
  * 越线判断阶段只使用当前帧的足球框，不使用过期框。
"""

import os
import time

import cv2
import numpy as np

try:
    from Functions.goal_line_judge import check_ball_crossed_goal_line
    from Functions.red_goal_line_detector import RedGoalLineEstimator
except ImportError:
    from goal_line_judge import check_ball_crossed_goal_line
    from red_goal_line_detector import RedGoalLineEstimator


class FootballDetector:
    def __init__(self, model_path=None, conf_threshold=0.45, iou_threshold=0.45, input_size=640):
        self.model_path = model_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "models", "football_best_win.onnx"
        )
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.input_size = int(input_size)
        if not os.path.exists(self.model_path):
            raise IOError("足球模型不存在: {}".format(self.model_path))
        self.net = cv2.dnn.readNetFromONNX(self.model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def detect(self, frame):
        if frame is None or getattr(frame, "size", 0) == 0:
            return None
        height, width = frame.shape[:2]
        scale = min(float(self.input_size) / width, float(self.input_size) / height)
        new_width, new_height = int(width * scale), int(height * scale)
        resized = cv2.resize(frame, (new_width, new_height))
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        dx = (self.input_size - new_width) // 2
        dy = (self.input_size - new_height) // 2
        canvas[dy:dy + new_height, dx:dx + new_width] = resized
        blob = cv2.dnn.blobFromImage(
            canvas, 1.0 / 255.0, (self.input_size, self.input_size),
            swapRB=True, crop=False,
        )
        self.net.setInput(blob)
        predictions = self.net.forward()[0]
        if predictions.ndim != 2 or predictions.shape[1] < 6:
            return None
        scores = predictions[:, 4] * predictions[:, 5]
        mask = scores > self.conf_threshold
        if not np.any(mask):
            return None
        selected = predictions[mask]
        selected_scores = scores[mask]
        boxes = []
        for item in selected:
            cx, cy, box_w, box_h = item[:4]
            boxes.append([
                float(cx - box_w / 2), float(cy - box_h / 2),
                float(box_w), float(box_h),
            ])
        indices = cv2.dnn.NMSBoxes(
            boxes, selected_scores.tolist(), self.conf_threshold, self.iou_threshold
        )
        if indices is None or len(indices) == 0:
            return None
        flat = np.asarray(indices).flatten().astype(int)
        best = int(max(flat, key=lambda index: float(selected_scores[index])))
        x1, y1, box_w, box_h = boxes[best]
        cx = (x1 + box_w / 2.0 - dx) / scale
        cy = (y1 + box_h / 2.0 - dy) / scale
        return (
            int(cx), int(cy), int(box_w / scale), int(box_h / scale),
            float(selected_scores[best]),
        )


class FootballKickController:
    """逐帧驱动的射门前朝向预对齐、保守接近和射门状态机。"""

    IDLE = "IDLE"
    ORIENTING_GOAL = "ORIENTING_GOAL"
    ALIGNING_BALL = "ALIGNING_BALL"
    APPROACHING_BALL = "APPROACHING_BALL"
    GOALPOST_ASSIST = "GOALPOST_ASSIST"
    KICKING = "KICKING"
    CHECKING_GOAL = "CHECKING_GOAL"
    RESULT = "RESULT"

    def __init__(
        self,
        motion,
        ball_confidence=0.45,
        action_interval_s=0.55,
        kick_settle_s=0.45,
        goal_timeout_s=6.0,
        prepare_timeout_s=20.0,
        goal_side="above",
        line_margin_px=3.0,
        ball_center_tolerance_px=29.0,
        ball_lane_tolerance_px=85.0,
        left_kick_target_offset_px=-71.0,
        right_kick_target_offset_px=68.0,
        kick_target_tolerance_px=29.0,
        kick_lane_tolerance_px=85.0,
        ball_turn_threshold_px=180.0,
        ball_far_bottom_y=235.0,
        ball_medium_bottom_y=325.0,
        ball_safe_bottom_y=405.0,
        ball_safe_center_y=428.0,
        ball_safe_height_px=90.0,
        ball_safe_confirm_frames=3,
        goalpost_center_tolerance_px=45.0,
        goalpost_turn_threshold_px=120.0,
        goalpost_min_count=2,
        goalpost_confirm_frames=5,
        goalpost_search_interval_s=1.2,
        max_micro_adjustments=4,
        line_flat_angle_deg=8.0,
        line_turn_angle_deg=12.0,
        line_confirm_frames=2,
        line_min_points=25,
        skip_goal_result_check=False,
        detector=None,
        line_estimator=None,
    ):
        self.motion = motion
        self.detector = detector or FootballDetector(conf_threshold=ball_confidence)
        self.action_interval_s = max(0.0, float(action_interval_s))
        self.kick_settle_s = max(0.0, float(kick_settle_s))
        self.goal_timeout_s = max(0.5, float(goal_timeout_s))
        self.prepare_timeout_s = max(1.0, float(prepare_timeout_s))
        self.goal_side = goal_side
        self.line_margin_px = max(0.0, float(line_margin_px))

        # 旧参数名保留兼容，但足球不再以摄像头中心作为严格射门基准。
        # 现在的基准是“当前选择的射门脚在图像中的目标位置”。
        legacy_target_tolerance = (
            29.0 if ball_center_tolerance_px is None
            else float(ball_center_tolerance_px)
        )
        legacy_lane_tolerance = (
            85.0 if ball_lane_tolerance_px is None
            else float(ball_lane_tolerance_px)
        )
        self.left_kick_target_offset_px = float(left_kick_target_offset_px)
        self.right_kick_target_offset_px = float(right_kick_target_offset_px)
        self.kick_target_tolerance_px = max(
            1.0,
            legacy_target_tolerance if kick_target_tolerance_px is None
            else float(kick_target_tolerance_px),
        )
        self.kick_lane_tolerance_px = max(
            self.kick_target_tolerance_px,
            legacy_lane_tolerance if kick_lane_tolerance_px is None
            else float(kick_lane_tolerance_px),
        )
        self.ball_turn_threshold_px = max(
            self.kick_lane_tolerance_px, float(ball_turn_threshold_px)
        )
        # 外部旧代码若读取这两个属性，得到的是新的脚部窗口阈值。
        self.ball_center_tolerance_px = self.kick_target_tolerance_px
        self.ball_lane_tolerance_px = self.kick_lane_tolerance_px
        self.ball_far_bottom_y = float(ball_far_bottom_y)
        self.ball_medium_bottom_y = float(ball_medium_bottom_y)
        self.ball_safe_bottom_y = float(ball_safe_bottom_y)
        self.ball_safe_center_y = float(ball_safe_center_y)
        self.ball_safe_height_px = float(ball_safe_height_px)
        self.ball_safe_confirm_frames = max(1, int(ball_safe_confirm_frames))
        self.goalpost_center_tolerance_px = max(1.0, float(goalpost_center_tolerance_px))
        self.goalpost_turn_threshold_px = max(
            self.goalpost_center_tolerance_px, float(goalpost_turn_threshold_px)
        )
        self.goalpost_min_count = max(2, int(goalpost_min_count))
        self.goalpost_confirm_frames = max(1, int(goalpost_confirm_frames))
        # 两柱暂未同时出现时，用身体转向搜索；动作间隔避免连续抖动。
        self.goalpost_search_interval_s = max(0.4, float(goalpost_search_interval_s))
        # 在宽射门走廊内连续微调达到上限后，仍允许尝试射门，
        # 避免机器人因每次微移过小而永远无法进入严格窗口。
        self.max_micro_adjustments = max(1, int(max_micro_adjustments))

        # 横移动作加入方向锁和滞回：ActionGroup 执行一次横移后，
        # 足球/球门柱的画面位置可能暂时越过目标。如果只看单帧符号，
        # 就会出现“左移 -> 右移 -> 左移”的来回摆动。
        self.goalpost_center_release_tolerance_px = min(
            self.goalpost_center_tolerance_px,
            max(1.0, self.goalpost_center_tolerance_px * 0.65),
        )
        self.goalpost_reverse_threshold_px = max(
            self.goalpost_center_tolerance_px + 15.0,
            self.goalpost_center_tolerance_px * 1.5,
        )
        self.ball_reverse_threshold_px = max(
            self.kick_target_tolerance_px + 15.0,
            self.kick_target_tolerance_px * 1.5,
        )

        # 红线只决定“平移还是转身”，不决定左右方向。采用滞回，
        # 避免角度在临界值附近时动作类型来回切换。
        self.line_flat_angle_deg = max(0.0, float(line_flat_angle_deg))
        self.line_turn_angle_deg = max(
            self.line_flat_angle_deg + 0.1, float(line_turn_angle_deg)
        )
        self.line_confirm_frames = max(1, int(line_confirm_frames))
        self.line_min_points = max(5, int(line_min_points))
        # V4 展示模式使用：踢球动作稳定后直接结束本次射门，
        # 不进入 CHECKING_GOAL；默认 False，保持 V3 原有越线判断。
        self.skip_goal_result_check = bool(skip_goal_result_check)

        self.line_estimator = line_estimator or RedGoalLineEstimator()
        self.reset()

    def reset(self):
        self.state = self.IDLE
        self.started_at = None
        self.kick_started_at = None
        self.next_action_at = 0.0
        self.last_ball = None
        self.last_goal_result = None
        self.last_line = None
        self.crossed = False
        self.result_message = ""
        self.ball_safe_streak = 0
        self.goalpost_center_streak = 0
        self.micro_adjustment_count = 0
        self.goalpost_search_count = 0
        self.last_goalpost_search_at = 0.0
        self.last_goalpost_center_x = None
        self.goalpost_correction_direction = None
        self.ball_correction_direction = None
        self.selected_kick_action = None
        self.last_action = None
        self.last_action_at = None
        self.last_action_reason = ""
        self.last_line_pixels = 0
        self.last_line_angle_deg = None
        self.line_adjustment_mode = None
        self.line_mode_candidate = None
        self.line_mode_streak = 0
        self.line_estimator.reset()

    def _clear_lateral_correction_locks(self):
        """清除球门柱和足球的横移方向锁。"""
        self.goalpost_correction_direction = None
        self.ball_correction_direction = None

    @property
    def active(self):
        return self.state != self.IDLE

    @property
    def waiting_for_goalposts(self):
        return self.state in (self.ORIENTING_GOAL, self.GOALPOST_ASSIST)

    def detect_ball(self, frame):
        self.last_ball = self.detector.detect(frame)
        return self.last_ball

    def start(self, now=None):
        now = time.monotonic() if now is None else float(now)
        self.state = self.ORIENTING_GOAL
        self.started_at = now
        self.kick_started_at = None
        self.next_action_at = 0.0
        self.last_ball = None
        self.last_goal_result = None
        self.last_line = None
        self.crossed = False
        self.ball_safe_streak = 0
        self.goalpost_center_streak = 0
        self.micro_adjustment_count = 0
        self.goalpost_search_count = 0
        self.last_goalpost_search_at = 0.0
        self.last_goalpost_center_x = None
        self.goalpost_correction_direction = None
        self.ball_correction_direction = None
        self.selected_kick_action = None
        self.last_action = None
        self.last_action_at = None
        self.last_action_reason = ""
        self.last_line_pixels = 0
        self.last_line_angle_deg = None
        self.line_adjustment_mode = None
        self.line_mode_candidate = None
        self.line_mode_streak = 0
        self.result_message = "已停止导航，先寻找球门柱并调平球门线"
        self.line_estimator.reset()
        # ActionGroup 完成后机器人保持站立，后续每一步都重新检测足球。
        self.motion.run("stand")

    def _run_action(self, action, now):
        if now < self.next_action_at:
            return False
        self.motion.run(action)
        self.last_action = action
        self.last_action_at = now
        self.next_action_at = now + self.action_interval_s
        return True

    @staticmethod
    def _ball_metrics(ball):
        cx, cy, width, height, _ = ball
        bottom_y = float(cy) + max(0.0, float(height)) / 2.0
        return float(cx), float(cy), float(width), float(height), bottom_y

    def _ball_is_safe(self, ball):
        _cx, cy, _width, _height, _bottom_y = self._ball_metrics(ball)
        # 实测以足球检测框中心 cy 判断最终踢球距离更稳定。
        # bottom_y 和检测框高度只保留给接近阶段和调试观察使用。
        return cy >= self.ball_safe_center_y

    def _kick_target_x(self, frame_width, action):
        """返回某个射门动作对应的实际击球目标 x 坐标。"""
        image_center = float(frame_width) / 2.0
        if action == "left_shot_fast":
            return image_center + self.left_kick_target_offset_px
        if action == "right_shot_fast":
            return image_center + self.right_kick_target_offset_px
        raise ValueError("未知射门动作: {}".format(action))

    def _select_kick_action(self, ball, frame_width):
        """根据足球离左右脚目标窗口的距离选择射门脚，并在一次射门中保持。"""
        if self.selected_kick_action in ("left_shot_fast", "right_shot_fast"):
            return self.selected_kick_action
        cx = float(ball[0])
        image_center = float(frame_width) / 2.0
        left_target = self._kick_target_x(frame_width, "left_shot_fast")
        right_target = self._kick_target_x(frame_width, "right_shot_fast")
        left_distance = abs(cx - left_target)
        right_distance = abs(cx - right_target)
        if abs(left_distance - right_distance) < 1.0:
            action = "left_shot_fast" if cx <= image_center else "right_shot_fast"
        else:
            action = "left_shot_fast" if left_distance < right_distance else "right_shot_fast"
        self.selected_kick_action = action
        return action

    def _ball_target_metrics(self, ball, frame_width, action=None):
        action = action or self._select_kick_action(ball, frame_width)
        target_x = self._kick_target_x(frame_width, action)
        delta = float(ball[0]) - target_x
        return action, target_x, delta

    def _ball_is_in_kick_lane(self, ball, frame_width, action=None):
        _action, _target_x, delta = self._ball_target_metrics(ball, frame_width, action)
        return abs(delta) <= self.kick_lane_tolerance_px

    def _ball_is_in_kick_window(self, ball, frame_width, action=None):
        _action, _target_x, delta = self._ball_target_metrics(ball, frame_width, action)
        return abs(delta) <= self.kick_target_tolerance_px

    def _ball_correction_action(self, delta, tolerance=None):
        """足球偏离目标时，优先横移；偏差很大才转身。"""
        delta = float(delta)
        limit = self.kick_lane_tolerance_px if tolerance is None else float(tolerance)
        if abs(delta) <= limit:
            return None
        if delta < -self.ball_turn_threshold_px:
            return "turn_left"
        if delta > self.ball_turn_threshold_px:
            return "turn_right"
        # 对准阶段也避免用大横移反复越过目标；在宽走廊内使用微移。
        if abs(delta) <= self.kick_lane_tolerance_px:
            return "left_move_20" if delta < 0 else "right_move_20"
        if delta < 0:
            return "left_move_fast"
        return "right_move_fast"

    def _line_turn_action(self, angle_deg):
        """根据球门线倾斜方向选择转身动作，直到红线接近水平。"""
        angle_deg = float(angle_deg)
        if angle_deg < -self.line_flat_angle_deg:
            # 图像 y 轴向下：负斜率表示需要向左转回正。
            return "turn_left"
        if angle_deg > self.line_flat_angle_deg:
            return "turn_right"
        return None

    def _ball_translation_correction_action(self, delta, tolerance):
        """红线已水平后，用带方向锁的横移把足球移入目标窗口。

        足球刚越过目标时，先站立等待下一帧稳定，不因一次检测抖动立即
        反向。只有明显越过另一侧的反向阈值，才切换横移方向。
        返回 ``"stand"`` 表示暂缓反向，调用方不能把它当作已进入踢球窗口。
        """
        delta = float(delta)
        tolerance = float(tolerance)
        if abs(delta) <= tolerance:
            self.ball_correction_direction = None
            return None

        desired = "left" if delta < 0 else "right"
        locked = self.ball_correction_direction
        if locked is None:
            self.ball_correction_direction = desired
        elif locked != desired:
            # 小幅越过目标先保持站立；明显越过才允许换向。
            if abs(delta) < self.ball_reverse_threshold_px:
                return "stand"
            self.ball_correction_direction = desired

        direction = self.ball_correction_direction
        if abs(delta) <= self.kick_lane_tolerance_px:
            return "left_move_20" if direction == "left" else "right_move_20"
        return "left_move_fast" if direction == "left" else "right_move_fast"

    def _ball_is_centered(self, ball, frame_width, tolerance=None):
        """兼容旧调用：centered 现在表示落在当前射门脚目标窗口。"""
        limit = self.kick_target_tolerance_px if tolerance is None else float(tolerance)
        _action, _target_x, delta = self._ball_target_metrics(ball, frame_width)
        return abs(delta) <= limit

    @staticmethod
    def _goalpost_center(goalposts):
        """用水平跨度最大的两根柱估计球门中央。

        GoalPostDetector 的结果按置信度排序，不保证前两项就是左右两根
        球门柱。直接平均前两项可能把同一根柱的重复框当成两个柱，导致
        机器人误以为已经正对球门。选择水平距离最大的检测对，更符合
        “左柱 + 右柱”的几何含义。
        """
        if not goalposts or len(goalposts) < 2:
            return None
        best_left = None
        best_right = None
        best_span = -1.0
        for left_index in range(len(goalposts) - 1):
            for right_index in range(left_index + 1, len(goalposts)):
                left = goalposts[left_index]
                right = goalposts[right_index]
                span = abs(float(right[0]) - float(left[0]))
                if span > best_span:
                    best_span = span
                    best_left = left
                    best_right = right
        if best_left is None or best_right is None:
            return None
        return (float(best_left[0]) + float(best_right[0])) / 2.0

    def _next_goalpost_search_action(self):
        """在足球保持安全距离时，以身体交替转向搜索两根球门柱。

        不使用左右转头舵机；交替方向可避免只朝一侧连续转动而越过球门。
        如果转动后足球暂时消失，主状态机会站立等待，绝不盲目前进。
        """
        # 连续几次向同一方向形成扫描，再反向扫回来；每一步都由
        # GOALPOST_ASSIST 的下一帧重新检测，不会连续盲转。
        direction_block = (self.goalpost_search_count // 3) % 2
        return "turn_left" if direction_block == 0 else "turn_right"

    def _kick(self, ball, frame_width, now):
        action = self._select_kick_action(ball, frame_width)
        self.motion.run(action)
        self.last_action = action
        self.last_action_at = now
        self.last_action_reason = "达到踢球条件，执行射门动作"
        self.state = self.KICKING
        self.kick_started_at = now
        self.result_message = "执行 {}，等待球门线结果".format(action)
        return action

    @staticmethod
    def _line_angle_deg(line):
        """将红线方向归一化到 [-90, 90] 度。图像中水平线为 0 度。"""
        if not line:
            return None
        dx = float(line.get("dx", 0.0))
        dy = float(line.get("dy", 0.0))
        if abs(dx) < 1e-6:
            return 90.0 if dy >= 0 else -90.0
        angle = float(np.degrees(np.arctan2(dy, dx)))
        while angle > 90.0:
            angle -= 180.0
        while angle < -90.0:
            angle += 180.0
        return angle

    def _update_line_mode(self, line, pixel_count):
        """根据红线角度更新平移/转身模式，并使用滞回和连续帧确认。"""
        angle = self._line_angle_deg(line)
        self.last_line_angle_deg = angle
        self.last_line_pixels = int(pixel_count or 0)
        reliable = angle is not None and self.last_line_pixels >= self.line_min_points
        if not reliable:
            self.line_mode_candidate = None
            self.line_mode_streak = 0
            return None

        abs_angle = abs(float(angle))
        previous = self.line_adjustment_mode
        if previous == "turn":
            candidate = "translate" if abs_angle <= self.line_flat_angle_deg else "turn"
        elif previous == "translate":
            candidate = "turn" if abs_angle >= self.line_turn_angle_deg else "translate"
        elif abs_angle <= self.line_flat_angle_deg:
            candidate = "translate"
        elif abs_angle >= self.line_turn_angle_deg:
            candidate = "turn"
        else:
            candidate = None

        if candidate is None:
            self.line_mode_candidate = None
            self.line_mode_streak = 0
            return None
        if candidate == self.line_adjustment_mode:
            self.line_mode_candidate = candidate
            self.line_mode_streak = self.line_confirm_frames
        elif candidate == self.line_mode_candidate:
            self.line_mode_streak += 1
        else:
            self.line_mode_candidate = candidate
            self.line_mode_streak = 1
        if self.line_mode_streak >= self.line_confirm_frames:
            self.line_adjustment_mode = candidate
        return self.line_adjustment_mode

    def _goalpost_correction_action(self, delta, line_mode=None):
        """根据球门中央偏差和红线方向选择转身或带方向锁的横移。

        红线接近水平时，机器人已经基本正对球门，优先横向平移；
        红线明显倾斜时，优先转身建立身体朝向。横移过程中采用滞回：
        回到释放阈值内才解除方向锁，小幅反向只站立等待，避免来回摆动。
        """
        delta = float(delta)
        center = self.goalpost_center_tolerance_px
        turn = self.goalpost_turn_threshold_px

        if line_mode != "translate":
            self.goalpost_correction_direction = None

        if abs(delta) <= center:
            if abs(delta) <= self.goalpost_center_release_tolerance_px:
                self.goalpost_correction_direction = None
            return None

        if line_mode == "translate":
            desired = "left" if delta < 0 else "right"
            locked = self.goalpost_correction_direction
            if locked is None:
                self.goalpost_correction_direction = desired
            elif locked != desired:
                # 只因小幅越过球门中央时，先站立，不要立刻反向。
                if abs(delta) < self.goalpost_reverse_threshold_px:
                    return "stand"
                self.goalpost_correction_direction = desired

            direction = self.goalpost_correction_direction
            # 红线已水平后，球门中央的中小偏差也只做微移；
            # 只有偏差明显超出宽走廊时才使用大步横移。
            if abs(delta) <= self.kick_lane_tolerance_px:
                return "left_move_20" if direction == "left" else "right_move_20"
            return "left_move_fast" if direction == "left" else "right_move_fast"
        if line_mode == "turn":
            return "turn_left" if delta < 0 else "turn_right"
        if abs(delta) > turn:
            return "turn_left" if delta < 0 else "turn_right"
        return "left_move_fast" if delta < 0 else "right_move_fast"

    def update(self, frame, ball=None, goalposts=None, now=None):
        now = time.monotonic() if now is None else float(now)
        # 对准/接近阶段只接受当前帧足球；检测不到就停，不用旧框继续走。
        if ball is not None:
            self.last_ball = ball

        info = {
            "state": self.state,
            "ball": ball,
            "line": self.last_line,
            "line_source": "unavailable",
            "goal_result": self.last_goal_result,
            "resolved": False,
            "crossed": False,
            "message": self.result_message,
        }
        if ball is not None and self.state != self.IDLE:
            action, target_x, target_delta = self._ball_target_metrics(
                ball, frame.shape[1], self.selected_kick_action
            )
            info.update({
                "selected_kick_action": action,
                "kick_target_x": target_x,
                "kick_target_delta": target_delta,
                "kick_lane_ok": self._ball_is_in_kick_lane(
                    ball, frame.shape[1], action
                ),
                "kick_window_ok": self._ball_is_in_kick_window(
                    ball, frame.shape[1], action
                ),
                "micro_adjustment_count": self.micro_adjustment_count,
                "max_micro_adjustments": self.max_micro_adjustments,
            })
        if self.state == self.IDLE:
            return info

        # 方向锁只在球门柱辅助横移阶段有效；离开该阶段后立即清空，
        # 防止上一阶段的横移方向影响下一阶段或下一次射门。
        if self.state != self.GOALPOST_ASSIST:
            self._clear_lateral_correction_locks()

        line_pixels = 0
        if goalposts:
            line, line_pixels = self.line_estimator.estimate(frame, goalposts)
            if line is not None:
                self.last_line = line
        line_mode = self._update_line_mode(self.last_line, line_pixels)
        goalpost_center = self._goalpost_center(goalposts or [])
        info.update({
            "goalpost_count": len(goalposts or []),
            "goalpost_center_x": goalpost_center,
            "line": self.last_line,
            "line_pixels": int(self.last_line_pixels),
            "line_angle_deg": (
                None if self.last_line_angle_deg is None
                else round(float(self.last_line_angle_deg), 2)
            ),
            "line_reliable": bool(
                self.last_line_angle_deg is not None
                and self.last_line_pixels >= self.line_min_points
            ),
            "line_adjustment_mode": line_mode,
            "line_mode_stable": self.line_adjustment_mode,
        })

        if self.state in (
            self.ORIENTING_GOAL,
            self.ALIGNING_BALL,
            self.APPROACHING_BALL,
            self.GOALPOST_ASSIST,
        ) and self.started_at is not None:
            if now - self.started_at >= self.prepare_timeout_s:
                self.state = self.RESULT
                self.crossed = False
                self.result_message = "足球对准/球门柱辅助超时，返回当前球门重试"

        if self.state == self.ORIENTING_GOAL:
            # 停止 Tag 导航后先锁定身体朝向：在球还没有接近前，
            # 只允许搜索球门柱和转身调平红色球门线，不允许向足球前进。
            if len(goalposts or []) < self.goalpost_min_count:
                self.goalpost_center_streak = 0
                if now - self.last_goalpost_search_at >= self.goalpost_search_interval_s:
                    search_action = self._next_goalpost_search_action()
                    if self._run_action(search_action, now):
                        self.goalpost_search_count += 1
                        self.last_goalpost_search_at = now
                        self.result_message = (
                            "射门前未同时看到两根球门柱，执行身体搜索 {}；保持站立，不接近足球"
                        ).format(search_action)
                    else:
                        self.result_message = (
                            "射门前等待身体搜索动作完成；仍需两根球门柱，不接近足球"
                        )
                        self._run_action("stand", now)
                else:
                    self.result_message = (
                        "射门前等待球门柱搜索间隔；保持站立，不接近足球"
                    )
                    self._run_action("stand", now)
            elif (
                self.last_line_angle_deg is None
                or self.last_line_pixels < self.line_min_points
            ):
                self.goalpost_center_streak = 0
                self.result_message = (
                    "已看到两根球门柱，但红色球门线暂不可靠；保持站立，不接近足球"
                )
                self._run_action("stand", now)
            elif abs(float(self.last_line_angle_deg)) > self.line_flat_angle_deg:
                self.goalpost_center_streak = 0
                correction_action = self._line_turn_action(
                    self.last_line_angle_deg
                )
                self.result_message = (
                    "射门前球门线倾斜 {:.1f}°，先转身调平，执行 {}；不接近足球"
                ).format(self.last_line_angle_deg, correction_action)
                self.last_action_reason = "射门前先将球门线调至水平"
                self._run_action(correction_action, now)
            else:
                self.goalpost_center_streak += 1
                self.result_message = (
                    "射门前球门线已接近水平，连续确认 ({}/{}); 暂不接近足球"
                ).format(
                    self.goalpost_center_streak, self.line_confirm_frames
                )
                self._run_action("stand", now)
                if self.goalpost_center_streak >= self.line_confirm_frames:
                    self.state = self.ALIGNING_BALL
                    self.ball_safe_streak = 0
                    self.goalpost_center_streak = 0
                    self.result_message = (
                        "射门前朝向已确认，开始按原逻辑对准并接近足球"
                    )

        elif self.state == self.ALIGNING_BALL:
            if ball is None:
                self.result_message = "已停止，等待重新检测足球，不继续前进"
                self._run_action("stand", now)
            elif not self._ball_is_safe(ball):
                # 第一阶段只判断 centre_y 是否达到安全距离。
                # 足球还远时不做任何左右调整，避免机器人过早横移/转身。
                _cx, center_y, _width, _height, bottom_y = self._ball_metrics(ball)
                self.ball_safe_streak = 0
                if bottom_y < self.ball_far_bottom_y:
                    action = "go_forward_fast"
                elif bottom_y < self.ball_medium_bottom_y:
                    action = "go_forward"
                else:
                    action = "go_forward_one_step"
                self.result_message = (
                    "足球尚未达到安全距离，centre_y={:.0f} < {:.0f}，先执行 {}，暂不水平调整"
                ).format(center_y, self.ball_safe_center_y, action)
                self._run_action(action, now)
            else:
                # centre_y 达到安全阈值，只说明可以停止前进；
                # 下一阶段仍然先调球门线水平，不能马上做足球水平调整。
                self.state = self.APPROACHING_BALL
                self.ball_safe_streak = 0
                self.result_message = (
                    "足球已达到安全距离（centre_y={:.0f}），停止前进；下一步先调平球门线，暂不左右平移"
                ).format(float(ball[1]))

        elif self.state == self.APPROACHING_BALL:
            if ball is None:
                self.result_message = "接近中暂时看不到足球，立即停步等待"
                self._run_action("stand", now)
            elif not self._ball_is_safe(ball):
                # 如果 centre_y 暂时跌回安全阈值以下，先恢复安全距离；
                # 在再次达到阈值前仍禁止左右微调。
                _cx, center_y, _width, _height, bottom_y = self._ball_metrics(ball)
                self.ball_safe_streak = 0
                if bottom_y < self.ball_far_bottom_y:
                    action = "go_forward_fast"
                elif bottom_y < self.ball_medium_bottom_y:
                    action = "go_forward"
                else:
                    action = "go_forward_one_step"
                self.result_message = (
                    "足球 centre_y={:.0f} < 安全阈值 {:.0f}，先执行 {}，禁止水平调整"
                ).format(center_y, self.ball_safe_center_y, action)
                self._run_action(action, now)
            else:
                # 总体顺序必须是：
                #   1) 先只按 centre_y 接近足球；
                #   2) 达到安全距离后，进入球门线/身体朝向调整；
                #   3) 红线水平以后，才允许左右平移和足球水平微调。
                # 因此这里绝不根据足球水平偏差做 left/right_move。
                self.ball_safe_streak += 1
                self.result_message = (
                    "足球已到安全距离（centre_y={:.0f}），先停止前进并进入球门线调平确认 "
                    "({}/{})；暂不左右平移"
                ).format(float(ball[1]), self.ball_safe_streak, self.ball_safe_confirm_frames)
                self._run_action("stand", now)
                if self.ball_safe_streak >= self.ball_safe_confirm_frames:
                    self.state = self.GOALPOST_ASSIST
                    self.goalpost_center_streak = 0
                    self.micro_adjustment_count = 0
                    self.last_goalpost_center_x = None
                    self.result_message = "足球已达到严格踢球距离，先寻找球门柱和红线，转身使球门线水平"

        elif self.state == self.GOALPOST_ASSIST:
            if ball is None:
                self._clear_lateral_correction_locks()
                self.result_message = "球门柱辅助期间看不到足球，保持站立，不调整、不踢球"
                self._run_action("stand", now)
            elif not self._ball_is_safe(ball):
                self._clear_lateral_correction_locks()
                self.state = self.APPROACHING_BALL
                self.ball_safe_streak = 0
                self.goalpost_center_streak = 0
                self.result_message = "足球距离仍不够近，返回保守接近"
            else:
                # 射门前严格分两步：先把身体朝向调正，再做横向位置调整。
                center_x = self._goalpost_center(goalposts or [])
                if center_x is None or len(goalposts or []) < self.goalpost_min_count:
                    self._clear_lateral_correction_locks()
                    self.goalpost_center_streak = 0
                    if now - self.last_goalpost_search_at >= self.goalpost_search_interval_s:
                        search_action = self._next_goalpost_search_action()
                        if self._run_action(search_action, now):
                            self.goalpost_search_count += 1
                            self.last_goalpost_search_at = now
                            self.result_message = (
                                "足球已保护，未同时看到两根球门柱，执行身体搜索 {}"
                            ).format(search_action)
                        else:
                            self.result_message = (
                                "足球已保护，等待身体搜索动作完成；仍需看到两根球门柱"
                            )
                            self._run_action("stand", now)
                    else:
                        self.result_message = (
                            "足球已保护，等待身体搜索间隔；仍需看到两根球门柱"
                        )
                        self._run_action("stand", now)
                elif (
                    self.last_line_angle_deg is None
                    or self.last_line_pixels < self.line_min_points
                ):
                    self._clear_lateral_correction_locks()
                    self.goalpost_center_streak = 0
                    self.result_message = "等待可靠的红色球门线，暂不横移或踢球"
                    self._run_action("stand", now)
                elif abs(float(self.last_line_angle_deg)) > self.line_flat_angle_deg:
                    self._clear_lateral_correction_locks()
                    self.goalpost_center_streak = 0
                    correction_action = self._line_turn_action(
                        self.last_line_angle_deg
                    )
                    self.result_message = (
                        "球门线仍倾斜 {:.1f}°，先转身使红线水平，执行 {}"
                    ).format(self.last_line_angle_deg, correction_action)
                    self.last_action_reason = "先将球门线调至水平，再进行横移"
                    self._run_action(correction_action, now)
                else:
                    # 红线已经水平，身体朝向固定；此后只允许左右平移。
                    self.last_goalpost_center_x = center_x
                    image_center = float(frame.shape[1]) / 2.0
                    goal_delta = center_x - image_center
                    correction_action = self._goalpost_correction_action(
                        goal_delta, line_mode="translate"
                    )
                    if correction_action is not None:
                        # 身体按球门柱做过一次横移后，足球的画面偏差必须重新评估。
                        self.ball_correction_direction = None
                        self.goalpost_center_streak = 0
                        self.micro_adjustment_count = 0
                        self.result_message = (
                            "红线已水平，球门中央偏差 {:.0f}px，执行横移 {}"
                        ).format(goal_delta, correction_action)
                        self.last_action_reason = "红线水平后的球门中央横向校正"
                        self._run_action(correction_action, now)
                    else:
                        action, target_x, ball_delta = self._ball_target_metrics(
                            ball, frame.shape[1], self.selected_kick_action
                        )
                        ball_correction = self._ball_translation_correction_action(
                            ball_delta, self.kick_target_tolerance_px
                        )
                        if ball_correction is not None:
                            # 在宽射门走廊内微调次数达到上限后，不再无限等待严格窗口。
                            # 只要偏差没有“特别夸张”，经过连续确认就尝试射门。
                            # ``stand`` 表示足球刚小幅越过目标，正在等待稳定；
                            # 不能把这种情况当作宽走廊内可直接射门。
                            in_wide_lane = (
                                ball_correction != "stand"
                                and abs(float(ball_delta)) <= self.kick_lane_tolerance_px
                            )
                            if in_wide_lane and self.micro_adjustment_count >= self.max_micro_adjustments:
                                self.goalpost_center_streak += 1
                                self.result_message = (
                                    "足球仍在宽射门走廊内（偏差 {:.0f}px），微调已达上限 {}/{}，"
                                    "尝试射门确认 ({}/{})"
                                ).format(
                                    ball_delta,
                                    self.micro_adjustment_count,
                                    self.max_micro_adjustments,
                                    self.goalpost_center_streak,
                                    self.goalpost_confirm_frames,
                                )
                                self.last_action_reason = "微调次数达到上限，宽走廊内尝试射门"
                                self._run_action("stand", now)
                                if self.goalpost_center_streak >= self.goalpost_confirm_frames:
                                    self._kick(ball, frame.shape[1], now)
                            else:
                                self.goalpost_center_streak = 0
                                if not in_wide_lane:
                                    self.micro_adjustment_count = 0
                                self.result_message = (
                                    "红线已水平，横移调整足球到{}目标 x={:.0f}，偏差 {:.0f}px，执行 {}"
                                ).format(action, target_x, ball_delta, ball_correction)
                                self.last_action_reason = "红线水平后的足球横向校正"
                                ran = self._run_action(ball_correction, now)
                                if ran and ball_correction in ("left_move_20", "right_move_20"):
                                    self.micro_adjustment_count += 1
                        else:
                            # 严格窗口内直接按原有连续确认逻辑射门。
                            self.micro_adjustment_count = 0
                            self.goalpost_center_streak += 1
                            self.result_message = (
                                "身体已正对球门，足球位于{}最终窗口 x={:.0f}（偏差 {:.0f}px），"
                                "球门中央连续确认 ({}/{})"
                            ).format(
                                action, target_x, ball_delta,
                                self.goalpost_center_streak,
                                self.goalpost_confirm_frames,
                            )
                            self._run_action("stand", now)
                            if self.goalpost_center_streak >= self.goalpost_confirm_frames:
                                self._kick(ball, frame.shape[1], now)
        elif self.state == self.KICKING:
            self.result_message = "踢球动作执行后稳定画面"
            if self.kick_started_at is not None and now - self.kick_started_at >= self.kick_settle_s:
                if self.skip_goal_result_check:
                    self.crossed = False
                    self.state = self.RESULT
                    self.result_message = "踢球动作已完成；展示模式不判断是否进门"
                else:
                    self.state = self.CHECKING_GOAL
                    self.result_message = "检查足球是否完全越过球门线"

        elif self.state == self.CHECKING_GOAL:
            line, _ = self.line_estimator.estimate(frame, goalposts or [])
            if line is not None:
                self.last_line = line
            result = check_ball_crossed_goal_line(
                ball,
                goalposts or [],
                image_width=frame.shape[1],
                goal_line=self.last_line,
                goal_line_source="red_pixels" if self.last_line is not None else "goalposts_fallback",
                goal_side=self.goal_side,
                margin_px=self.line_margin_px,
                previous_streak=0,
                confirm_frames=1,
            )
            self.last_goal_result = result
            info["line_source"] = result.get("line_source", "unavailable")
            if result.get("whole_crossed"):
                self.crossed = True
                self.state = self.RESULT
                self.result_message = "足球整体完全越过球门线，射门成功"
            elif self.kick_started_at is not None and now - self.kick_started_at >= self.goal_timeout_s:
                self.crossed = False
                self.state = self.RESULT
                self.result_message = "超时未检测到完全越线，本次射门失败，准备重试"
            else:
                self.result_message = "等待足球完全越线（足球框四个角均需越线）"

        elif self.state == self.RESULT:
            pass

        info.update({
            "state": self.state,
            "line": self.last_line,
            "line_source": (self.last_goal_result or {}).get("line_source", "unavailable"),
            "goal_result": self.last_goal_result,
            "resolved": self.state == self.RESULT,
            "crossed": bool(self.crossed),
            "message": self.result_message,
            "last_action": self.last_action,
            "last_action_at": self.last_action_at,
            "last_action_reason": self.last_action_reason,
        })
        return info