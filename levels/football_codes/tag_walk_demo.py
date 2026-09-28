#!/usr/bin/python3
# coding=utf8
"""
tag_walk_demo.py — AprilTag 导航 + 球门柱对齐走路 Demo

当前 Demo 流程：
  摄像头
    -> AprilTag 路线导航（锁定 Tag -> 接近 -> 连续丢失后判定到达）
    -> 先完成当前必经 Tag 路线；导航中的足球只检测、不打断路线
    -> 进入球门搜索 Tag 区域后，确认足球才停止 Tag 动作
    -> 再由足球/球门柱状态机对准并保守接近其正后方（看不到球就停）
    -> 足球进入安全区域后检测两根球门柱并做小幅辅助校正
    -> 复刻 FootballKick.py 的左右脚射门动作
    -> 用红色球门线及其延长线判断足球是否完全越线
    -> 每个球门最多射一脚；无论是否确认越线都推进下一阶段
    -> 第二球门射门结束后沿出口 Tag 路线到 Tag 37

球门柱检测沿用 Functions/finalkick.py 的模型和预处理逻辑，但通过
Functions/goalpost_detector.py 独立封装，避免导入 finalkick 的完整射门状态机。
模型：Functions/weights/best.onnx

实机运行：
  sudo systemctl stop tonypi
  cd /home/pi/TonyPi/Functions
  python3 tag_walk_demo.py --run

默认是 dry-run，只检测并打印，不执行动作；实机必须显式加 --run。
"""

from __future__ import print_function

import argparse
import time

# 本地优先：Robot_Competition 副本为唯一真源（含 apriltag 库搜索路径修复）；
# TonyPi/Functions 副本可能滞后或缺修复。
try:
    from tag_route_demo import (
        AprilTagDetector,
        ActionGroupMotion,
        DEMO_PLAN,
        NullMotion,
        TagRouteNavigator,
    )
    from goalpost_detector import GoalPostDetector, draw_goalposts, goal_center_x
    from football_kick_controller import FootballKickController
except ImportError:
    from Functions.tag_route_demo import (
        AprilTagDetector,
        ActionGroupMotion,
        DEMO_PLAN,
        NullMotion,
        TagRouteNavigator,
    )
    from Functions.goalpost_detector import (
        GoalPostDetector,
        draw_goalposts,
        goal_center_x,
    )
    from Functions.football_kick_controller import FootballKickController


def open_camera():
    """使用与 locate_web.py 相同的原始 Camera 取帧方式。"""
    import hiwonder.Camera as Camera

    # locate_web.py 始终使用 Camera.Camera()；这里也不再根据
    # camera_setting.yaml 切换到 MJPEG 网络流，避免两条视频链路的
    # 分辨率、压缩和延迟差异影响 AprilTag 检测结果。
    camera = Camera.Camera()
    camera.camera_open()
    time.sleep(1.0)
    return camera, False


def close_camera(camera, is_cv_capture):
    if camera is None:
        return
    try:
        if is_cv_capture:
            camera.release()
        else:
            camera.camera_close()
    except Exception as exc:
        print("关闭摄像头时出现异常：{}".format(exc))


def draw_observations(frame, observations, goalposts, event, goal_event, ball=None, shot_info=None):
    """绘制 Tag、球门柱、足球、球门线和当前任务状态。"""
    import cv2
    import numpy as np

    image = frame.copy()
    for observation in observations:
        color = (0, 255, 0)
        if observation.corners is not None:
            corners = np.asarray(observation.corners, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(image, [corners], True, color, 2)
        cv2.putText(
            image,
            "Tag {}".format(observation.tag_id),
            (int(observation.center_x) - 30, int(observation.center_y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    image = draw_goalposts(image, goalposts)
    if ball is not None:
        cx, cy, width, height, confidence = ball
        x1 = int(cx - width / 2)
        y1 = int(cy - height / 2)
        x2 = int(cx + width / 2)
        y2 = int(cy + height / 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(image, (int(cx), int(cy)), 3, (0, 255, 0), -1)
        cv2.putText(
            image,
            "Football {:.2f}".format(confidence),
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    if shot_info:
        line = shot_info.get("line")
        if line is not None:
            cv2.line(
                image,
                tuple(map(int, line["extended_p1"])),
                tuple(map(int, line["extended_p2"])),
                (0, 0, 255),
                2,
            )
        text = "Shot: {} | {}".format(
            shot_info.get("state", "-"), shot_info.get("message", "")
        )
        cv2.putText(
            image, text[:105], (10, 84), cv2.FONT_HERSHEY_SIMPLEX,
            0.48, (0, 255, 255), 2,
        )

    cv2.putText(
        image,
        "{}: {}".format(event.kind, event.stage),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        image,
        "Goal: {}".format(goal_event),
        (10, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 0, 255),
        2,
    )
    return image


class GoalPostApproachController:
    """球门柱搜索、身体对齐和靠近控制器。

    这里只做球门柱搜索和身体 yaw 对齐；实际足球接近与射门由
    FootballKickController 负责。
    一次 action group 完成后，等待 action_interval_s 再发下一个动作，
    避免在同一动作尚未完成时重复下发动作。
    """

    SEARCHING = "SEARCHING_GOALPOST"
    ALIGNING_LEFT = "ALIGNING_GOAL_LEFT"
    ALIGNING_RIGHT = "ALIGNING_GOAL_RIGHT"
    WAITING_CENTER = "WAITING_GOAL_CENTER"
    APPROACHING = "APPROACHING_GOAL"
    DONE = "GOAL_APPROACH_DONE"

    def __init__(
        self,
        motion,
        center_tolerance_px=55.0,
        confirm_frames=3,
        min_posts=2,
        forward_steps=2,
        action_interval_s=0.45,
    ):
        self.motion = motion
        self.center_tolerance_px = float(center_tolerance_px)
        self.confirm_frames = max(1, int(confirm_frames))
        self.min_posts = max(2, int(min_posts))
        self.forward_steps = max(0, int(forward_steps))
        self.action_interval_s = max(0.0, float(action_interval_s))
        self.reset()

    def reset(self):
        self.center_streak = 0
        self.forward_count = 0
        self.last_center_x = None
        self.done = False
        self.next_action_at = 0.0

    def _run_action(self, action_name, now):
        if now < self.next_action_at:
            return False
        self.motion.run(action_name)
        self.next_action_at = now + self.action_interval_s
        return True

    def update(self, posts, frame_width, now=None):
        """处理一帧球门柱检测，返回状态字符串。"""
        now = time.monotonic() if now is None else float(now)
        if self.done:
            return self.DONE

        # 只有同时检测到至少两根球门柱，才允许进入“居中确认”。
        # 仅看到一根柱子时不能可靠计算球门中央，也不能触发后续阶段切换。
        if len(posts) < self.min_posts:
            self.center_streak = 0
            self.last_center_x = None
            return "{} ({}/{})".format(
                self.SEARCHING, len(posts), self.min_posts
            )

        # 模型偶尔可能产生多于两根候选框；优先使用置信度最高的两根，
        # 避免误检框把球门中心拉偏。goalpost_detector 返回结果按置信度降序排列。
        selected_posts = list(posts[:self.min_posts])
        center_x = goal_center_x(selected_posts)
        if center_x is None:
            self.center_streak = 0
            self.last_center_x = None
            return "{} ({}/{})".format(
                self.SEARCHING, len(posts), self.min_posts
            )

        self.last_center_x = center_x
        image_center = float(frame_width) / 2.0
        if center_x < image_center - self.center_tolerance_px:
            self.center_streak = 0
            if self._run_action("turn_left", now):
                return self.ALIGNING_LEFT
            return "WAIT_GOAL_ALIGN_ACTION"

        if center_x > image_center + self.center_tolerance_px:
            self.center_streak = 0
            if self._run_action("turn_right", now):
                return self.ALIGNING_RIGHT
            return "WAIT_GOAL_ALIGN_ACTION"

        self.center_streak += 1
        if self.center_streak < self.confirm_frames:
            return self.WAITING_CENTER

        if self.forward_count < self.forward_steps:
            if self._run_action("go_forward_fast", now):
                self.forward_count += 1
                return "{} {}/{}".format(
                    self.APPROACHING,
                    self.forward_count,
                    self.forward_steps,
                )
            return "WAIT_GOAL_APPROACH_ACTION"

        self.done = True
        return self.DONE


class NavigationHeadController:
    """导航头部控制：俯仰观察，并在丢失目标 Tag 时左右扫描。"""

    def __init__(
        self,
        execute_actions=False,
        look_down_offset=10,
        pan_search_offset=500,
    ):
        self.enabled = bool(execute_actions)
        self.current_pitch = None
        self.current_pan = None
        self.normal_pitch = 500
        self.normal_pan = 500
        self.pan_search_offset = max(1, int(pan_search_offset))
        self.search_pitch = self.normal_pitch - int(look_down_offset)
        self.controller = None
        if self.enabled:
            import hiwonder.ros_robot_controller_sdk as rrc
            import hiwonder.yaml_handle as yaml_handle
            from hiwonder.Controller import Controller
            servo_data = yaml_handle.get_yaml_data(yaml_handle.servo_file_path)
            self.normal_pitch = int(servo_data["servo1"])
            self.normal_pan = int(servo_data["servo2"])
            self.search_pitch = self.normal_pitch - int(look_down_offset)
            self.controller = Controller(rrc.Board())

        # 与 locate_web.py 的实机约定一致：servo2 脉宽增大是向左看，
        # 脉宽减小是向右看。只保存命令位置，不依赖舵机反馈。
        self.left_pan = self.normal_pan + self.pan_search_offset
        self.right_pan = self.normal_pan - self.pan_search_offset

    def set_pan(self, pan, duration=350):
        pan = int(pan)
        if self.current_pan == pan:
            return False
        self.current_pan = pan
        if self.enabled:
            self.controller.set_pwm_servo_pulse(2, pan, int(duration))
        return True

    def center_pan(self, duration=350):
        return self.set_pan(self.normal_pan, duration=duration)

    def look_left(self, duration=350):
        return self.set_pan(self.left_pan, duration=duration)

    def look_right(self, duration=350):
        return self.set_pan(self.right_pan, duration=duration)

    def pan_direction(self, deadband=30):
        """返回当前头部相对身体的方向：left/right/None（基本居中）。"""
        if self.current_pan is None:
            return None
        delta = int(self.current_pan) - int(self.normal_pan)
        if delta > int(deadband):
            return "left"
        if delta < -int(deadband):
            return "right"
        return None

    def set_pitch(self, pitch, duration=350):
        pitch = int(pitch)
        if self.current_pitch == pitch:
            return False
        self.current_pitch = pitch
        if self.enabled:
            self.controller.set_pwm_servo_pulse(1, pitch, int(duration))
        return True

    def lower_for_tag_search(self):
        return self.set_pitch(self.search_pitch)

    def restore_normal(self):
        return self.set_pitch(self.normal_pitch)

class TagWalkDemo:
    """把 Tag 导航、球门柱对齐和 FootballKick 射门流程接起来。"""

    def __init__(
        self,
        execute_actions=False,
        tag_confirm_frames=3,
        tag_lost_confirm_frames=5,
        tag_lost_confirm_s=0.6,
        min_tag_approach_s=1.0,
        approach_action_interval_s=0.45,
        arrival_height_px=110.0,
        center_tolerance_px=65.0,
        search_action="turn_left",
        exit_search_action="turn_right",
        search_interval_s=1.5,
        head_look_down_offset=10,
        head_pan_search_offset=500,
        head_stabilize_s=0.8,
        head_guided_turn_settle_s=0.8,
        low_tag_search_timeout_s=3.0,
        goalpost_confidence=0.30,
        goalpost_confirm_frames=5,
        max_micro_adjustments=4,
        goalpost_min_count=2,
        goalpost_center_tolerance_px=45.0,
        goalpost_turn_threshold_px=120.0,
        goal_forward_steps=2,
        goal_action_interval_s=0.45,
        goal_search_timeout_s=20.0,
        stage2_goal_search_wait_s=2.0,
        goalpost_detect_interval=2,
        ball_confirm_frames=2,
        ball_confidence=0.45,
        kick_action_interval_s=0.55,
        kick_settle_s=0.45,
        kick_goal_timeout_s=6.0,
        kick_prepare_timeout_s=20.0,
        kick_attempts_per_goal=1,
        advance_after_kick_without_goal_check=False,
        ball_loss_timeout_s=8.0,
        goal_side="above",
        goal_line_margin_px=3.0,
        ball_center_tolerance_px=None,
        ball_lane_tolerance_px=None,
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
        max_seconds=600.0,
    ):
        motion = ActionGroupMotion() if execute_actions else NullMotion()
        self.motion = motion
        self.execute_actions = bool(execute_actions)
        self.head_controller = NavigationHeadController(
            execute_actions=self.execute_actions,
            look_down_offset=head_look_down_offset,
            pan_search_offset=head_pan_search_offset,
        )
        self.navigator = TagRouteNavigator(
            plan=DEMO_PLAN,
            motion=motion,
            tag_confirm_frames=tag_confirm_frames,
            tag_lost_confirm_frames=tag_lost_confirm_frames,
            tag_lost_confirm_s=tag_lost_confirm_s,
            min_tag_approach_s=min_tag_approach_s,
            approach_action_interval_s=approach_action_interval_s,
            arrival_height_px=arrival_height_px,
            center_tolerance_px=center_tolerance_px,
            goal_search_timeout_s=goal_search_timeout_s,
        )
        self.detector = AprilTagDetector()
        self.goalpost_detector = GoalPostDetector(
            conf_threshold=goalpost_confidence,
        )
        self.goal_controller = GoalPostApproachController(
            motion=motion,
            center_tolerance_px=goalpost_center_tolerance_px,
            confirm_frames=goalpost_confirm_frames,
            min_posts=goalpost_min_count,
            forward_steps=goal_forward_steps,
            action_interval_s=goal_action_interval_s,
        )
        # 旧搜索动作参数保留接口兼容；Tag 丢失后不再按固定方向盲转。
        # 新流程由头部先扫描，确认目标方向后才让身体朝同方向转一步。
        self.search_action = search_action
        self.exit_search_action = exit_search_action
        self.search_interval_s = max(0.1, float(search_interval_s))
        self.head_stabilize_s = max(0.0, float(head_stabilize_s))
        self.head_guided_turn_settle_s = max(
            0.0, float(head_guided_turn_settle_s)
        )
        self.low_tag_search_timeout_s = max(0.0, float(low_tag_search_timeout_s))
        self.stage2_goal_search_wait_s = max(0.0, float(stage2_goal_search_wait_s))
        # 第二阶段到达 Tag82 后，每次进入 SEARCH_GOAL 会话最多执行一次左转。
        # 用 navigator 的 goal_search_started_at 区分首次进入和射门失败后的重试。
        self.stage2_goal_search_session_started_at = None
        self.stage2_goal_search_turn_done = False
        self.goalpost_detect_interval = max(1, int(goalpost_detect_interval))
        self.max_seconds = float(max_seconds)
        self.started_at = time.monotonic()
        self.tag_search_started_at = None
        self.low_search_until = None
        self.head_scan_sequence = ("left", "center", "right", "center")
        self.head_scan_index = 0
        self.head_scan_settle_until = 0.0
        self.head_scan_next_move_at = 0.0
        self.head_guided_turn_settle_until = 0.0
        self.last_printed = None
        self.last_visible_tag_ids = ()
        self.goalpost_frame_counter = 0
        self.cached_goalposts = []
        self.ball_confirm_frames = max(1, int(ball_confirm_frames))
        self.ball_seen_streak = 0
        self.last_ball = None
        self.ball_priority_active = False
        self.ball_ignore_until = 0.0
        self.kick_attempts_per_goal = max(1, int(kick_attempts_per_goal))
        # 默认 False 保持 V3：射门后仍由控制器判断是否越线。
        # demoV4 显式设为 True：踢球动作完成后直接推进路线。
        self.advance_after_kick_without_goal_check = bool(
            advance_after_kick_without_goal_check
        )
        self.kick_attempts_this_goal = 0
        self.ball_loss_timeout_s = max(0.0, float(ball_loss_timeout_s))
        self.ball_missing_since = None
        self.kick_controller = FootballKickController(
            motion=motion,
            ball_confidence=ball_confidence,
            action_interval_s=kick_action_interval_s,
            kick_settle_s=kick_settle_s,
            goal_timeout_s=kick_goal_timeout_s,
            prepare_timeout_s=kick_prepare_timeout_s,
            goal_side=goal_side,
            line_margin_px=goal_line_margin_px,
            skip_goal_result_check=self.advance_after_kick_without_goal_check,
            ball_center_tolerance_px=ball_center_tolerance_px,
            ball_lane_tolerance_px=ball_lane_tolerance_px,
            left_kick_target_offset_px=left_kick_target_offset_px,
            right_kick_target_offset_px=right_kick_target_offset_px,
            kick_target_tolerance_px=kick_target_tolerance_px,
            kick_lane_tolerance_px=kick_lane_tolerance_px,
            ball_turn_threshold_px=ball_turn_threshold_px,
            ball_far_bottom_y=ball_far_bottom_y,
            ball_medium_bottom_y=ball_medium_bottom_y,
            ball_safe_bottom_y=ball_safe_bottom_y,
            ball_safe_center_y=ball_safe_center_y,
            ball_safe_height_px=ball_safe_height_px,
            ball_safe_confirm_frames=ball_safe_confirm_frames,
            goalpost_center_tolerance_px=goalpost_center_tolerance_px,
            goalpost_turn_threshold_px=goalpost_turn_threshold_px,
            goalpost_min_count=goalpost_min_count,
            goalpost_confirm_frames=goalpost_confirm_frames,
            max_micro_adjustments=max_micro_adjustments,
        )

    def _print_event(self, text):
        if text != self.last_printed:
            print(text, flush=True)
            self.last_printed = text

    def _print_navigation_event(self, event):
        message = "[{}] [{}] {}".format(event.stage, event.kind, event.message)
        if event.kind == "SEARCH_TAG":
            current_step = self.navigator.current_step
            expected_ids = list(current_step.tags) if current_step is not None else []
            if self.last_visible_tag_ids:
                message += (
                    "；当前目标 Tag={}，当前画面检测到 Tag={}"
                    "，没有检测到当前目标"
                ).format(expected_ids, list(self.last_visible_tag_ids))
            else:
                message += "；当前目标 Tag={}，当前帧未检测到任何 Tag".format(
                    expected_ids
                )
        self._print_event(message)

    def _prepare_tag_head_scan_sequence(self):
        """按当前路线阶段设置转头顺序；出口阶段优先向右看。"""
        current_stage = self.navigator.current_stage
        if current_stage is not None and current_stage.goal_tag is None:
            # 第二球门后的退场路线沿用“优先向右找”的场地经验，但只是
            # 先向右转头观察；没有视觉确认时身体仍然保持不动。
            self.head_scan_sequence = ("right", "center", "left", "center")
        else:
            self.head_scan_sequence = ("left", "center", "right", "center")
        self.head_scan_index = 0

    def _reset_tag_search_state(self, center_pan=True):
        """结束本轮 Tag 头部扫描；可选择让水平舵机回到中位。"""
        self.tag_search_started_at = None
        self.low_search_until = None
        self.head_scan_index = 0
        self.head_scan_settle_until = 0.0
        self.head_scan_next_move_at = 0.0
        if center_pan:
            self.head_controller.center_pan()

    def _move_head_for_tag_scan(self, direction, now):
        """移动头部到下一个扫描位置，并留出舵机稳定和观察时间。"""
        if direction == "left":
            self.head_controller.look_left()
            label = "左侧"
        elif direction == "right":
            self.head_controller.look_right()
            label = "右侧"
        else:
            self.head_controller.center_pan()
            label = "正前方"

        self.head_scan_settle_until = now + self.head_stabilize_s
        self.head_scan_next_move_at = (
            self.head_scan_settle_until + self.search_interval_s
        )
        self._print_event(
            "[HEAD SEARCH] 身体保持不动，头部转向{}搜索目标 Tag".format(label)
        )

    def _align_body_from_head_search(self, direction, now):
        """目标已在转头视野中确认：身体朝头部方向转一步，再回中重找。"""
        body_action = "turn_left" if direction == "left" else "turn_right"
        if self.execute_actions:
            self.navigator.motion.run(body_action)

        # 身体转完后头部回中，并清除旧视角下的锁定；下一轮必须在
        # 新身体朝向下重新确认目标，避免直接按偏转相机坐标前进。
        self.head_controller.center_pan()
        self.navigator._clear_target_lock()
        self.navigator._clear_target_streak()
        # ActionGroup 可能是同步阻塞调用，因此以动作真正返回后的时间
        # 计算稳定窗口，保证头部回中后确实留出完整的稳定时间。
        aligned_at = time.monotonic()
        self.head_guided_turn_settle_until = (
            aligned_at + self.head_guided_turn_settle_s + self.head_stabilize_s
        )
        self._reset_tag_search_state(center_pan=False)
        self._print_event(
            "[HEAD GUIDED TURN] 头部在{}发现并确认目标 Tag，身体执行 {}；"
            "头部回中后重新确认".format(
                "左侧" if direction == "left" else "右侧",
                body_action,
            )
        )

    def _search_when_no_target(self, event):
        """先转头扫描目标 Tag；确认目标方位后才转动身体。

        搜索顺序为：先在正前方静止观察，再左右转头循环扫描。普通路线
        优先看左侧；第二球门后的退场路线优先看右侧，保留既定场地经验。
        没有检测并确认目标时，身体始终不转，避免按固定方向盲目搜索。
        如果在偏转视角中连续确认了目标，身体只朝相同方向转一步，头部
        回中后重新确认；若仍未对准，会再次扫描并做下一次有依据的转身。
        """
        now = time.monotonic()

        # 到达球门搜索点后恢复正常头部姿态；这里搜索的是足球，不是 Tag。
        if self.navigator.state == self.navigator.SEARCH_GOAL:
            self.head_controller.restore_normal()
            self.head_controller.center_pan()

            # stage_index=1 对应 STAGE2：到达 Tag82 后寻找第二球门足球。
            # 该既有逻辑只属于球门区找球，不属于“找不到目标 Tag”的路线搜索。
            if self.navigator.stage_index == 1:
                search_started_at = self.navigator.goal_search_started_at
                if search_started_at != self.stage2_goal_search_session_started_at:
                    self.stage2_goal_search_session_started_at = search_started_at
                    self.stage2_goal_search_turn_done = False

                if (
                    not self.stage2_goal_search_turn_done
                    and search_started_at is not None
                    and now - search_started_at >= self.stage2_goal_search_wait_s
                ):
                    if self.execute_actions:
                        self.motion.run("turn_left")
                    self.stage2_goal_search_turn_done = True
                    self._print_event(
                        "[SEARCH] 到达第二球门 Tag82 后仍未看到足球，执行一次左转搜索"
                    )
            else:
                self.stage2_goal_search_session_started_at = None
                self.stage2_goal_search_turn_done = False

            self._reset_tag_search_state(center_pan=False)
            return

        # 转头视角中看到目标时先保持头部不动，利用 navigator 原有的
        # 连续帧机制确认。确认锁定后，再依据当前头部方向转动身体。
        if event.kind in ("TRACK_TAG", "TAG_LOCKED"):
            head_direction = self.head_controller.pan_direction()
            if head_direction is not None:
                if event.kind == "TRACK_TAG":
                    self._print_event(
                        "[HEAD SEARCH] 已在{}看到目标 Tag，保持头部不动并等待连续确认".format(
                            "左侧" if head_direction == "left" else "右侧"
                        )
                    )
                    return
                self._align_body_from_head_search(head_direction, now)
                return

            # 正前方确认到目标，不需要额外转身，直接交给正常接近逻辑。
            self._reset_tag_search_state(center_pan=True)
            return

        if event.kind != "SEARCH_TAG":
            # 已经在接近 Tag、切换状态或完成任务，停止扫描并回中。
            self._reset_tag_search_state(center_pan=True)
            return

        if self.tag_search_started_at is None:
            self._prepare_tag_head_scan_sequence()
            self.tag_search_started_at = now
            self.head_controller.lower_for_tag_search()
            self.head_controller.center_pan()
            self.head_scan_settle_until = now + self.head_stabilize_s
            self.low_search_until = (
                self.head_scan_settle_until + self.low_tag_search_timeout_s
            )
            self.head_scan_next_move_at = self.low_search_until
            self._print_event(
                "[HEAD SEARCH] 未看到目标 Tag，身体保持不动；先低头朝正前方观察 {:.1f} 秒".format(
                    self.low_tag_search_timeout_s
                )
            )
            return

        # 舵机运动和稳定期间不改变扫描位置。
        if now < self.head_scan_settle_until:
            return

        # 初次正前方观察窗口，或当前扫描位置的停留窗口尚未结束。
        if self.low_search_until is not None and now < self.low_search_until:
            return
        if now < self.head_scan_next_move_at:
            return

        direction = self.head_scan_sequence[self.head_scan_index]
        self.head_scan_index = (self.head_scan_index + 1) % len(
            self.head_scan_sequence
        )
        self._move_head_for_tag_scan(direction, now)

    def _detect_goalposts(self, frame, allow=False):
        """只在足球优先模式运行球门柱检测。

        足球出现后 Tag 导航已经暂停；控制器先锁定球门线朝向，
        再按原逻辑接近足球，最后用球门柱做小幅辅助，避免导航和
        球门柱动作互相覆盖。
        """
        if (
            not allow
            or not self.ball_priority_active
            or self.kick_controller.state not in (
                self.kick_controller.ORIENTING_GOAL,
                self.kick_controller.GOALPOST_ASSIST,
                self.kick_controller.KICKING,
                self.kick_controller.CHECKING_GOAL,
            )
        ):
            self.cached_goalposts = []
            self.goalpost_frame_counter = 0
            return []

        self.goalpost_frame_counter += 1
        if self.goalpost_frame_counter % self.goalpost_detect_interval == 0 or not self.cached_goalposts:
            self.cached_goalposts = self.goalpost_detector.detect(frame)
        return self.cached_goalposts

    def _reset_shot_session(self, now, ignore_ball_s=0.0, reset_attempts=False):
        """清理当前足球优先会话；阶段推进时同时清零本球门射门次数。"""
        self.kick_controller.reset()
        self.goal_controller.reset()
        self.cached_goalposts = []
        self.goalpost_frame_counter = 0
        self.ball_seen_streak = 0
        self.last_ball = None
        self.ball_priority_active = False
        self.ball_missing_since = None
        self.ball_ignore_until = float(now) + max(0.0, float(ignore_ball_s))
        if reset_attempts:
            self.kick_attempts_this_goal = 0

    def process_frame(self, frame):
        """处理一帧图像；必经 Tag 导航优先，球门区内再切换到射门。"""
        now = time.monotonic()
        observations = self.detector.detect(frame)
        self.last_visible_tag_ids = tuple(
            sorted({int(observation.tag_id) for observation in observations})
        )

        event = None
        goal_event = "未进入足球优先模式"
        shot_info = None
        goalposts = []
        ball = None

        # 出口阶段没有球门，不因偶然检测框触发射门；球优先模式内则必须持续检测。
        has_goal_stage = (
            self.navigator.state != self.navigator.DONE
            and self.navigator.current_stage.goal_tag is not None
        )
        if has_goal_stage or self.kick_controller.active:
            ball = self.kick_controller.detect_ball(frame)
            self.last_ball = ball

        # Tag 导航的前置路线不能被足球检测打断：
        # 第一阶段必须先到 Tag 103、执行左转，再到 Tag 26/50；
        # 只有进入球门搜索状态后，足球才可以触发射门状态机。
        goal_search_states = (
            self.navigator.SEARCH_GOAL,
            self.navigator.WAIT_SHOT,
        )
        if (
            not self.ball_priority_active
            and not self.kick_controller.active
            and now >= self.ball_ignore_until
            and ball is not None
            and self.navigator.state in goal_search_states
        ):
            self.ball_seen_streak += 1
        elif not self.ball_priority_active and not self.kick_controller.active:
            self.ball_seen_streak = 0

        if (
            not self.ball_priority_active
            and not self.kick_controller.active
            and self.ball_seen_streak >= self.ball_confirm_frames
            and self.navigator.state in goal_search_states
        ):
            event = self.navigator.notify_goal_ready()
            if event.kind == "BALL_PRIORITY_STOP":
                self.ball_priority_active = True
                self.ball_missing_since = None
                self.goal_controller.reset()
                self.cached_goalposts = []
                self.goalpost_frame_counter = 0
                self.kick_controller.start(now=now)
                self._print_navigation_event(event)
                self._print_event(
                    "[{}] 检测到足球，已停止 Tag 导航；先调平球门线，再对准并接近足球".format(
                        self.navigator.current_stage.name
                    )
                )

        shot_mode_active = self.ball_priority_active or self.kick_controller.active
        if shot_mode_active:
            if ball is None:
                if self.ball_missing_since is None:
                    self.ball_missing_since = now
            else:
                self.ball_missing_since = None

            if (
                self.ball_loss_timeout_s > 0
                and self.ball_missing_since is not None
                and now - self.ball_missing_since >= self.ball_loss_timeout_s
            ):
                next_event = self.navigator.notify_goal_abandoned(
                    "射门阶段连续 {:.1f} 秒未看到足球，放弃当前球门并进入下一阶段".format(
                        self.ball_loss_timeout_s
                    ),
                    event_kind="BALL_LOSS_ADVANCE",
                )
                self._print_navigation_event(next_event)
                self._reset_shot_session(now, ignore_ball_s=4.0, reset_attempts=True)
                goal_event = "连续丢球超时，已放弃当前球门并推进路线"
                shot_info = {
                    "state": "BALL_LOSS_TIMEOUT",
                    "resolved": True,
                    "crossed": False,
                    "message": goal_event,
                }
                return next_event, observations, [], goal_event, ball, shot_info
        else:
            self.ball_missing_since = None

        if shot_mode_active:
            # 球优先模式中永远不调用 TagRouteNavigator.update()，避免导航动作覆盖保护动作。
            goalposts = self._detect_goalposts(frame, allow=True)
            shot_info = self.kick_controller.update(
                frame,
                ball=ball,
                goalposts=goalposts,
                now=now,
            )
            state = shot_info.get("state", self.kick_controller.state)
            if state == self.kick_controller.ORIENTING_GOAL:
                goal_event = "足球优先：先转身使球门线水平，再接近足球"
            elif state == self.kick_controller.ALIGNING_BALL:
                goal_event = "足球优先：调整机器人姿态，使足球进入当前射门脚目标走廊"
            elif state == self.kick_controller.APPROACHING_BALL:
                goal_event = "足球优先：保守接近足球正后方，检测不到球就停"
            elif state == self.kick_controller.GOALPOST_ASSIST:
                goal_event = "足球已在安全区域：球门柱只做小幅辅助校正"
            elif state in (self.kick_controller.KICKING, self.kick_controller.CHECKING_GOAL):
                goal_event = "已完成足球/球门柱准备，执行射门并快速判断完全越线"
            elif state == self.kick_controller.RESULT:
                goal_event = "本次射门已得到结果"

            self._print_event(
                "[{}] {}".format(
                    self.navigator.current_stage.name,
                    shot_info.get("message", shot_info.get("state", "")),
                )
            )
            if event is None:
                # 只用于画面显示，不触发任何动作。
                event = self.navigator._event(
                    "BALL_PRIORITY",
                    goal_event,
                    goal_tag=self.navigator.current_stage.goal_tag,
                    step_name=(
                        self.navigator.current_step.name
                        if self.navigator.current_step is not None else None
                    ),
                )

            if shot_info.get("resolved"):
                crossed = bool(shot_info.get("crossed"))
                self.kick_attempts_this_goal += 1
                if self.advance_after_kick_without_goal_check:
                    next_event = self.navigator.notify_goal_abandoned(
                        "射门动作已完成；V4 展示模式不判断是否进门，直接进入下一阶段",
                        event_kind="KICK_COMPLETED_ADVANCE",
                    )
                elif crossed:
                    next_event = self.navigator.notify_shot_result(True)
                elif self.kick_attempts_this_goal >= self.kick_attempts_per_goal:
                    next_event = self.navigator.notify_goal_abandoned(
                        "本球门已完成第 {}/{} 脚，未确认越线；已达到射门上限，进入下一阶段".format(
                            self.kick_attempts_this_goal,
                            self.kick_attempts_per_goal,
                        ),
                        event_kind="SHOT_LIMIT_ADVANCE",
                    )
                else:
                    next_event = self.navigator.notify_shot_result(False)

                self._print_navigation_event(next_event)
                stage_advanced = next_event.kind in (
                    "STAGE_ADVANCE",
                    "SHOT_LIMIT_ADVANCE",
                    "KICK_COMPLETED_ADVANCE",
                )
                # 只要已经推进到下一阶段，就忽略上一球门的残留足球框，
                # 防止反弹或模型残留在下一阶段入口处触发假启动。
                self._reset_shot_session(
                    now,
                    ignore_ball_s=4.0 if stage_advanced else 0.0,
                    reset_attempts=stage_advanced,
                )
                event = next_event

        else:
            # 头部扫描舵机尚未稳定，或刚按头部信息转完身体时，暂停路线
            # 状态机；避免使用运动中的模糊画面，也避免头未回中就开始接近。
            if now < self.head_guided_turn_settle_until:
                event = self.navigator._event(
                    "HEAD_GUIDED_TURN_SETTLE",
                    "已按头部发现方向转身，等待头部回中稳定后重新确认目标 Tag",
                    step_name=(
                        self.navigator.current_step.name
                        if self.navigator.current_step is not None else None
                    ),
                )
                self._print_navigation_event(event)
            elif (
                self.tag_search_started_at is not None
                and now < self.head_scan_settle_until
            ):
                event = self.navigator._event(
                    "HEAD_SCAN_SETTLE",
                    "头部正在移动或稳定，身体保持不动，暂不使用本帧导航",
                    step_name=(
                        self.navigator.current_step.name
                        if self.navigator.current_step is not None else None
                    ),
                )
                self._print_navigation_event(event)
            else:
                # 只有没有足球优先事件时，才允许 TagRouteNavigator 发出路线动作。
                event = self.navigator.update(
                    observations,
                    frame_width=frame.shape[1],
                    goal_ready=False,
                    now=now,
                )
                self._print_navigation_event(event)
                self._search_when_no_target(event)

            if self.navigator.state == self.navigator.SEARCH_GOAL:
                goal_event = "球门搜索区域：检测到足球后停步，启动球门柱与射门流程"
            elif self.navigator.state == self.navigator.NAVIGATE:
                goal_event = "Tag 导航中：先完成必经路线，暂不因足球中断"

        return event, observations, goalposts, goal_event, ball, shot_info

    def run(self, display=False):
        import cv2

        camera = None
        is_cv_capture = False
        try:
            # 进入本阶段前先确认站稳，避免上一个阶段（如巡线）残留动作导致摔倒
            if self.execute_actions:
                import hiwonder.ActionGroupControl as AGC
                AGC.runActionGroup("stand")
                time.sleep(1.0)
            # 启动时水平舵机回中；之后找不到目标 Tag 时先左右转头扫描，
            # 只有在偏转视角中确认目标后，才按头部方向转动身体。
            self.head_controller.center_pan()
            camera, is_cv_capture = open_camera()
            print("Tag + 球门柱对齐走路 Demo 已启动", flush=True)
            print("执行动作：{}".format("是" if self.execute_actions else "否（dry-run）"), flush=True)
            print("流程：完成必经 Tag 导航 -> 进入球门搜索区后发现足球停步 -> 球门柱居中 -> FootballKick 射门 -> 越线成功后进入下一阶段", flush=True)
            print("按 Ctrl+C 停止", flush=True)

            while True:
                if time.monotonic() - self.started_at > self.max_seconds:
                    print("超过最大运行时间，自动停止", flush=True)
                    break

                ret, frame = camera.read()
                if not ret or frame is None:
                    time.sleep(0.05)
                    continue

                event, observations, goalposts, goal_event, ball, shot_info = self.process_frame(frame)
                if display:
                    image = draw_observations(
                        frame,
                        observations,
                        goalposts,
                        event,
                        goal_event,
                        ball=ball,
                        shot_info=shot_info,
                    )
                    cv2.imshow("TagWalkGoalPostDemo", image)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break

                if self.navigator.state == self.navigator.DONE:
                    print("已到达出口 Tag 37，走路 Demo 完成", flush=True)
                    break

                time.sleep(0.02)
        finally:
            close_camera(camera, is_cv_capture)
            if display:
                cv2.destroyAllWindows()
            if self.execute_actions:
                try:
                    self.head_controller.restore_normal()
                    import hiwonder.ActionGroupControl as AGC
                    AGC.runActionGroup("stand")
                except Exception as exc:
                    print("恢复站立动作失败：{}".format(exc), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="TonyPi AprilTag + FootballKick 带球射门 Demo"
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="实机执行动作；不加此参数时只检测和打印，不驱动机器人",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="显示 OpenCV 画面；SSH 无桌面时不要使用",
    )
    parser.add_argument("--confirm-frames", type=int, default=3)
    parser.add_argument(
        "--tag-lost-confirm-frames",
        type=int,
        default=5,
        help="目标 Tag 连续丢失多少帧后才允许判定到达",
    )
    parser.add_argument(
        "--tag-lost-confirm-seconds",
        type=float,
        default=0.6,
        help="目标 Tag 至少连续丢失多少秒后才允许判定到达",
    )
    parser.add_argument(
        "--min-tag-approach-seconds",
        type=float,
        default=1.0,
        help="锁定 Tag 后至少接近多久，避免刚漏检就误判到达",
    )
    parser.add_argument(
        "--approach-action-interval",
        type=float,
        default=0.45,
        help="接近 Tag 时相邻动作组之间的最短间隔",
    )
    parser.add_argument("--arrival-height", type=float, default=110.0)
    parser.add_argument("--center-tolerance", type=float, default=65.0)
    parser.add_argument(
        "--head-look-down-offset",
        type=int,
        default=10,
        help="Tag 搜索时 servo1 相对正常位置的轻微低头偏移量（参考 locate_web）",
    )
    parser.add_argument(
        "--head-pan-search-offset",
        type=int,
        default=500,
        help="找不到目标 Tag 时，头部向左右扫描的 servo2 脉宽偏移量",
    )
    parser.add_argument(
        "--head-stabilize",
        type=float,
        default=0.8,
        help="每次转头或低头后等待云台稳定的秒数",
    )
    parser.add_argument(
        "--head-guided-turn-settle",
        type=float,
        default=0.8,
        help="依据头部方向转身后，重新识别目标 Tag 前的额外稳定时间",
    )
    parser.add_argument(
        "--low-tag-search-timeout",
        type=float,
        default=3.0,
        help="低头朝正前方寻找 Tag 的秒数；超时后开始左右转头扫描",
    )
    parser.add_argument(
        "--search-action",
        default="turn_left",
        help="旧参数兼容项；当前 Tag 搜索不再按固定动作盲目转身",
    )
    parser.add_argument("--search-interval", type=float, default=1.5)
    parser.add_argument("--goalpost-conf", type=float, default=0.30)
    parser.add_argument("--goalpost-confirm-frames", type=int, default=5)
    parser.add_argument(
        "--max-micro-adjustments", type=int, default=4,
        help="宽射门走廊内使用 *_move_10 微调的最大次数，达到后仍会尝试射门",
    )
    parser.add_argument(
        "--goalpost-min-count",
        type=int,
        default=2,
        help="至少同时检测到几根球门柱才允许确认球门中央；默认 2",
    )
    parser.add_argument("--goalpost-center-tolerance", type=float, default=45.0)
    parser.add_argument(
        "--goalpost-turn-threshold", type=float, default=120.0,
        help="球门中央偏差超过该值才转身；中间区域优先左右平移",
    )
    parser.add_argument(
        "--goal-forward-steps",
        type=int,
        default=0,
        help="球门柱居中后向前走几步；设为 0 可只测试对齐",
    )
    parser.add_argument("--goal-action-interval", type=float, default=0.45)
    parser.add_argument(
        "--goal-search-timeout",
        type=float,
        default=20.0,
        help="在一个球门 Tag 附近寻找球门柱的最长秒数，超时转备用 Tag",
    )
    parser.add_argument("--goalpost-detect-interval", type=int, default=2)
    parser.add_argument("--ball-confirm-frames", type=int, default=2)
    parser.add_argument("--ball-conf", type=float, default=0.45)
    parser.add_argument("--kick-action-interval", type=float, default=0.55)
    parser.add_argument("--kick-settle", type=float, default=0.45)
    parser.add_argument("--kick-goal-timeout", type=float, default=6.0)
    parser.add_argument(
        "--kick-prepare-timeout", type=float, default=20.0,
        help="从看到足球到球门柱辅助完成的最长时间",
    )
    parser.add_argument("--goal-side", choices=("above", "below"), default="above")
    parser.add_argument("--goal-line-margin", type=float, default=3.0)
    parser.add_argument(
        "--ball-center-tolerance", type=float, default=None,
        help="旧参数兼容项；建议改用 --kick-target-tolerance",
    )
    parser.add_argument(
        "--ball-lane-tolerance", type=float, default=None,
        help="旧参数兼容项；建议改用 --kick-lane-tolerance",
    )
    parser.add_argument(
        "--left-kick-target-offset", type=float, default=-71.0,
        help="left_shot_fast 的足球目标相对图像中心偏移，左为负",
    )
    parser.add_argument(
        "--right-kick-target-offset", type=float, default=68.0,
        help="right_shot_fast 的足球目标相对图像中心偏移，右为正",
    )
    parser.add_argument(
        "--kick-target-tolerance", type=float, default=29.0,
        help="最终射门时足球允许偏离对应射门脚目标的像素数",
    )
    parser.add_argument(
        "--kick-lane-tolerance", type=float, default=85.0,
        help="接近和球门柱辅助阶段允许的宽射门走廊半径",
    )
    parser.add_argument(
        "--ball-turn-threshold", type=float, default=180.0,
        help="足球相对射门脚目标偏差超过该值才转身，否则优先横移",
    )
    parser.add_argument("--ball-far-bottom-y", type=float, default=235.0)
    parser.add_argument("--ball-medium-bottom-y", type=float, default=325.0)
    parser.add_argument(
        "--ball-safe-bottom-y", type=float, default=405.0,
        help="旧参数兼容项；接近阶段仍保留检测框底部的分段距离参考",
    )
    parser.add_argument(
        "--ball-safe-center-y", type=float, default=428.0,
        help="足球检测框中心 cy 达到该 y 后，才允许进入严格踢球距离判断",
    )
    parser.add_argument("--ball-safe-height", type=float, default=90.0)
    parser.add_argument("--ball-safe-confirm-frames", type=int, default=3)
    parser.add_argument("--max-seconds", type=float, default=600.0)
    return parser.parse_args()


def main():
    args = parse_args()
    demo = TagWalkDemo(
        execute_actions=args.run,
        tag_confirm_frames=args.confirm_frames,
        tag_lost_confirm_frames=args.tag_lost_confirm_frames,
        tag_lost_confirm_s=args.tag_lost_confirm_seconds,
        min_tag_approach_s=args.min_tag_approach_seconds,
        approach_action_interval_s=args.approach_action_interval,
        arrival_height_px=args.arrival_height,
        center_tolerance_px=args.center_tolerance,
        head_look_down_offset=args.head_look_down_offset,
        head_pan_search_offset=args.head_pan_search_offset,
        head_stabilize_s=args.head_stabilize,
        head_guided_turn_settle_s=args.head_guided_turn_settle,
        low_tag_search_timeout_s=args.low_tag_search_timeout,
        search_action=args.search_action,
        search_interval_s=args.search_interval,
        goalpost_confidence=args.goalpost_conf,
        goalpost_confirm_frames=args.goalpost_confirm_frames,
        max_micro_adjustments=args.max_micro_adjustments,
        goalpost_min_count=args.goalpost_min_count,
        goalpost_center_tolerance_px=args.goalpost_center_tolerance,
         goalpost_turn_threshold_px=args.goalpost_turn_threshold,
        goal_forward_steps=args.goal_forward_steps,
        goal_action_interval_s=args.goal_action_interval,
        goal_search_timeout_s=args.goal_search_timeout,
        goalpost_detect_interval=args.goalpost_detect_interval,
         ball_confirm_frames=args.ball_confirm_frames,
         ball_confidence=args.ball_conf,
         kick_action_interval_s=args.kick_action_interval,
         kick_settle_s=args.kick_settle,
         kick_goal_timeout_s=args.kick_goal_timeout,
         kick_prepare_timeout_s=args.kick_prepare_timeout,
         goal_side=args.goal_side,
         goal_line_margin_px=args.goal_line_margin,
         ball_center_tolerance_px=args.ball_center_tolerance,
         ball_lane_tolerance_px=args.ball_lane_tolerance,
         left_kick_target_offset_px=args.left_kick_target_offset,
         right_kick_target_offset_px=args.right_kick_target_offset,
         kick_target_tolerance_px=args.kick_target_tolerance,
         kick_lane_tolerance_px=args.kick_lane_tolerance,
         ball_turn_threshold_px=args.ball_turn_threshold,
         ball_far_bottom_y=args.ball_far_bottom_y,
         ball_medium_bottom_y=args.ball_medium_bottom_y,
         ball_safe_bottom_y=args.ball_safe_bottom_y,
         ball_safe_center_y=args.ball_safe_center_y,
         ball_safe_height_px=args.ball_safe_height,
         ball_safe_confirm_frames=args.ball_safe_confirm_frames,
        max_seconds=args.max_seconds,
    )
    try:
        demo.run(display=args.display)
    except KeyboardInterrupt:
        print("收到 Ctrl+C，停止 Demo", flush=True)


if __name__ == "__main__":
    main()










