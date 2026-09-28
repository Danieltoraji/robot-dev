#!/usr/bin/python3
# coding=utf8
"""巡线终点复核与漏转直角弯恢复。

稳定丢线不再直接判定第一阶段结束：先让头部扫描第一条 Tag 路线的目标
Tag；只有连续确认目标 Tag 才允许切换阶段。若完整扫描仍未发现目标，则
逐步后退，并复用 RedLinePatrolV3 原有的直角弯检测与转弯闭环。
"""

from __future__ import print_function

import time


class RecoveryEvent:
    def __init__(self, kind, message, display_frame=None, visible_tag_ids=()):
        self.kind = kind
        self.message = message
        self.display_frame = display_frame
        self.visible_tag_ids = tuple(visible_tag_ids or ())


class PatrolEndRecoveryController:
    """在候选巡线终点处，用 Tag 复核；失败时后退恢复漏掉的直角弯。"""

    IDLE = "IDLE"
    HEAD_SCAN = "HEAD_SCAN"
    BACKTRACK = "BACKTRACK"
    TURN_RECOVERY = "TURN_RECOVERY"
    TAG_FOUND = "TAG_FOUND"
    RECOVERED = "RECOVERED"
    FAILED = "FAILED"

    def __init__(
        self,
        redline,
        execute_actions=False,
        target_tag_ids=(103,),
        tag_confirm_frames=3,
        head_pan_offset=500,
        head_settle_s=0.8,
        head_observe_s=1.0,
        max_back_steps=8,
        back_step_interval_s=0.8,
        turn_recovery_timeout_s=25.0,
        turn_recovery_no_turn_timeout_s=4.0,
        detector_factory=None,
    ):
        self.redline = redline
        self.execute_actions = bool(execute_actions)
        self.target_tag_ids = tuple(int(tag_id) for tag_id in target_tag_ids)
        self.tag_confirm_frames = max(1, int(tag_confirm_frames))
        self.head_pan_offset = max(1, int(head_pan_offset))
        self.head_settle_s = max(0.0, float(head_settle_s))
        self.head_observe_s = max(0.1, float(head_observe_s))
        self.max_back_steps = max(1, int(max_back_steps))
        self.back_step_interval_s = max(0.1, float(back_step_interval_s))
        self.turn_recovery_timeout_s = max(1.0, float(turn_recovery_timeout_s))
        self.turn_recovery_no_turn_timeout_s = max(
            0.5, float(turn_recovery_no_turn_timeout_s)
        )
        self.detector_factory = detector_factory
        self.detector = None

        self.scan_sequence = ("center", "left", "center", "right", "center")
        self.state = self.IDLE
        self.scan_index = 0
        self.current_head_direction = "center"
        self.head_settle_until = 0.0
        self.head_observe_until = 0.0
        self.tag_seen_streak = 0
        self.last_visible_tag_ids = ()
        self.back_steps_done = 0
        self.backtrack_ready_at = 0.0
        self.next_back_step_at = 0.0
        self.turn_started_seen = False
        self.turn_recovery_started_at = 0.0

    @property
    def active(self):
        return self.state in (self.HEAD_SCAN, self.BACKTRACK, self.TURN_RECOVERY)

    def _event(self, kind, message, display_frame=None):
        return RecoveryEvent(
            kind,
            message,
            display_frame=display_frame,
            visible_tag_ids=self.last_visible_tag_ids,
        )

    def _ensure_detector(self):
        if self.detector is not None:
            return
        if self.detector_factory is None:
            # 本地优先：Robot_Competition 副本为唯一真源（含 apriltag 库
            # 搜索路径修复）；TonyPi/Functions 副本可能滞后或缺修复。
            try:
                from tag_route_demo import AprilTagDetector
            except ImportError:
                from Functions.tag_route_demo import AprilTagDetector
            self.detector_factory = AprilTagDetector
        self.detector = self.detector_factory()

    @staticmethod
    def _clamp_servo_pulse(value):
        # 本项目 servo_config 与现有头部代码使用 0~1000 脉宽；避免偏移越界。
        return max(0, min(1000, int(value)))

    def _set_head_direction(self, direction, now):
        servo_data = getattr(self.redline, "servo_data", None) or {}
        normal_pitch = int(servo_data.get("servo1", 1500))
        normal_pan = int(servo_data.get("servo2", 1500))
        if direction == "left":
            pan = normal_pan + self.head_pan_offset
        elif direction == "right":
            pan = normal_pan - self.head_pan_offset
        else:
            pan = normal_pan

        if self.execute_actions:
            controller = getattr(self.redline, "ctl", None)
            if controller is None:
                raise RuntimeError("RedLinePatrolV3 未提供头部舵机控制器 ctl")
            controller.set_pwm_servo_pulse(1, normal_pitch, 350)
            controller.set_pwm_servo_pulse(
                2, self._clamp_servo_pulse(pan), 350
            )

        self.current_head_direction = direction
        self.head_settle_until = now + self.head_settle_s
        self.head_observe_until = self.head_settle_until + self.head_observe_s
        self.tag_seen_streak = 0

    def _restore_patrol_head(self):
        if self.execute_actions:
            self.redline.initMove()

    def _pause_patrol(self):
        # 不能调用 redline.stop()，因为 stop() 会 reset()，从而丢掉原巡线
        # 状态；这里只暂停运动线程，之后仍要复用原直角弯闭环。
        self.redline.running = False
        self.redline.AGC.runActionGroup("stand")

    def start(self, now=None):
        """开始终点复核。调用前应已稳定确认脚下红线消失。"""
        now = time.monotonic() if now is None else float(now)
        try:
            self._pause_patrol()
            self._ensure_detector()
            self.state = self.HEAD_SCAN
            self.scan_index = 1
            self.last_visible_tag_ids = ()
            self.back_steps_done = 0
            self.turn_started_seen = False
            self._set_head_direction(self.scan_sequence[0], time.monotonic())
            return self._event(
                "END_CHECK_STARTED",
                "候选终点已停车；先抬头并转头搜索目标 Tag {}，暂不结束第一阶段".format(
                    list(self.target_tag_ids)
                ),
            )
        except Exception as exc:
            self.state = self.FAILED
            return self._event(
                self.FAILED,
                "无法启动巡线终点 Tag 复核，已安全停止：{}".format(exc),
            )

    def _reset_patrol_line_state(self):
        """清除候选终点与恢复过程留下的巡线状态，身体保持暂停。"""
        self.redline.running = False
        if hasattr(self.redline, "line_center_x"):
            self.redline.line_center_x = -1
        if hasattr(self.redline, "line_lost_time"):
            self.redline.line_lost_time = 0
        if hasattr(self.redline, "approach_active"):
            self.redline.approach_active = False
        if hasattr(self.redline, "corner_ready"):
            self.redline.corner_ready = False
        if hasattr(self.redline, "turn_started"):
            self.redline.turn_started = False

    def _begin_backtrack(self, now):
        self._restore_patrol_head()
        # 清除候选终点造成的丢线记忆；身体仍保持暂停，由本控制器逐步后退。
        self._reset_patrol_line_state()

        self.state = self.BACKTRACK
        self.back_steps_done = 0
        # 从恢复巡线俯视姿态的调用返回后计时，保证完整稳定窗口。
        restored_at = time.monotonic()
        self.backtrack_ready_at = restored_at + self.head_settle_s
        self.next_back_step_at = self.backtrack_ready_at
        return self._event(
            "TAG_NOT_FOUND_BACKTRACK",
            "完整转头扫描未找到目标 Tag；判定第一阶段尚未结束，开始逐步后退寻找漏识别的直角弯",
        )

    def _update_head_scan(self, raw_frame, now):
        if now >= self.head_settle_until:
            try:
                observations = self.detector.detect(raw_frame)
            except Exception as exc:
                self.state = self.FAILED
                return self._event(
                    self.FAILED,
                    "终点复核期间 AprilTag 检测失败，已安全停止：{}".format(exc),
                    display_frame=raw_frame,
                )

            self.last_visible_tag_ids = tuple(
                sorted({int(observation.tag_id) for observation in observations})
            )
            target_visible = any(
                int(observation.tag_id) in self.target_tag_ids
                for observation in observations
            )
            if target_visible:
                self.tag_seen_streak += 1
            else:
                self.tag_seen_streak = 0

            if self.tag_seen_streak >= self.tag_confirm_frames:
                # 这里只确认阶段结束，不直接按单帧位置行动。Tag 导航启动后会
                # 再用自己的连续帧和“先转头、再转身”逻辑锁定并接近 Tag。
                found_direction = self.current_head_direction
                self._set_head_direction("center", now)
                self.state = self.TAG_FOUND
                return self._event(
                    self.TAG_FOUND,
                    "已在{}连续确认目标 Tag {}；第一阶段结束，进入 Tag 导航".format(
                        {
                            "left": "左侧",
                            "right": "右侧",
                            "center": "正前方",
                        }.get(found_direction, "当前视野"),
                        list(self.target_tag_ids),
                    ),
                    display_frame=raw_frame,
                )

        if now >= self.head_observe_until:
            if self.scan_index >= len(self.scan_sequence):
                return self._begin_backtrack(now)
            direction = self.scan_sequence[self.scan_index]
            self.scan_index += 1
            self._set_head_direction(direction, now)
            return self._event(
                "HEAD_SCAN_MOVE",
                "身体保持不动，头部转向{}继续搜索目标 Tag {}".format(
                    {"left": "左侧", "right": "右侧", "center": "正前方"}[
                        direction
                    ],
                    list(self.target_tag_ids),
                ),
                display_frame=raw_frame,
            )

        return self._event(
            "HEAD_SCAN",
            "正在{}搜索目标 Tag {}；当前看到 Tag {}，连续确认 {}/{} 帧".format(
                {"left": "左侧", "right": "右侧", "center": "正前方"}[
                    self.current_head_direction
                ],
                list(self.target_tag_ids),
                list(self.last_visible_tag_ids),
                self.tag_seen_streak,
                self.tag_confirm_frames,
            ),
            display_frame=raw_frame,
        )

    def _update_backtrack(self, corrected_frame, now):
        if now < self.backtrack_ready_at:
            return self._event(
                "PATROL_HEAD_SETTLE",
                "头部正在恢复巡线俯视姿态；身体保持不动，稳定后再开始后退找弯",
                display_frame=corrected_frame,
            )

        # running=False 时 redline.run() 只做视觉分析，不会由巡线线程前进；
        # 一旦它原有的直角弯检测将 corner_ready 置真（宽度达标且 corner_y
        # 足够靠下，与 move() 真正转弯的判据一致），就停止后退。
        display_frame = self.redline.run(corrected_frame)
        if getattr(self.redline, "corner_ready", False):
            self.redline.AGC.runActionGroup("stand")
            self.redline.running = True
            self.state = self.TURN_RECOVERY
            self.turn_started_seen = False
            self.turn_recovery_started_at = time.monotonic()
            return self._event(
                "CORNER_REACQUIRED",
                "后退过程中第一次稳定识别到直角弯；停止后退，恢复原弯前接近和转弯闭环",
                display_frame=display_frame,
            )

        if now >= self.next_back_step_at:
            if self.back_steps_done >= self.max_back_steps:
                self.state = self.FAILED
                self.redline.running = False
                self.redline.AGC.runActionGroup("stand")
                return self._event(
                    self.FAILED,
                    "已后退 {} 步仍未识别到直角弯；为避免无限倒退，已安全停止".format(
                        self.back_steps_done
                    ),
                    display_frame=display_frame,
                )

            self.redline.AGC.runActionGroup("back", times=1)
            self.back_steps_done += 1
            # 动作组可能同步阻塞，所以从动作返回时重新计时。
            self.next_back_step_at = time.monotonic() + self.back_step_interval_s
            return self._event(
                "BACKTRACK_STEP",
                "未看到直角弯，后退第 {}/{} 步；每步后重新观察".format(
                    self.back_steps_done, self.max_back_steps
                ),
                display_frame=display_frame,
            )

        return self._event(
            "BACKTRACK_OBSERVE",
            "后退后保持观察，等待直角弯进入视野（已后退 {}/{} 步）".format(
                self.back_steps_done, self.max_back_steps
            ),
            display_frame=display_frame,
        )

    def _update_turn_recovery(self, corrected_frame, now):
        display_frame = self.redline.run(corrected_frame)
        approach_active = bool(getattr(self.redline, "approach_active", False))
        turn_started = bool(getattr(self.redline, "turn_started", False))
        if turn_started:
            self.turn_started_seen = True

        if self.turn_started_seen and not approach_active and not turn_started:
            self.state = self.RECOVERED
            return self._event(
                self.RECOVERED,
                "漏识别直角弯的原有转弯闭环已完成；继续第一阶段巡线",
                display_frame=display_frame,
            )

        # 假找回兜底：corner_ready 短暂满足却始终未真正触发转弯（turn_started
        # 从未置真），说明后退时误把普通红线宽度当成了直角弯。回到后退阶段
        # 继续寻找，避免干等 turn_recovery_timeout_s 后才 FAILED。
        if not self.turn_started_seen and (
            now - self.turn_recovery_started_at
            >= self.turn_recovery_no_turn_timeout_s
        ):
            self._restore_patrol_head()
            self._reset_patrol_line_state()
            self.state = self.BACKTRACK
            self.back_steps_done = 0
            self.turn_started_seen = False
            self.backtrack_ready_at = time.monotonic() + self.head_settle_s
            self.next_back_step_at = self.backtrack_ready_at
            return self._event(
                "BACKTRACK_RETRY",
                "找回信号未真正触发转弯，回到后退阶段继续寻找直角弯",
                display_frame=display_frame,
            )

        if now - self.turn_recovery_started_at >= self.turn_recovery_timeout_s:
            self.state = self.FAILED
            self.redline.running = False
            self.redline.AGC.runActionGroup("stand")
            return self._event(
                self.FAILED,
                "直角弯恢复超过 {:.1f} 秒仍未完成；已安全停止".format(
                    self.turn_recovery_timeout_s
                ),
                display_frame=display_frame,
            )

        return self._event(
            "TURN_RECOVERY",
            "已找回直角弯，正在执行原有弯前接近/转弯/视觉转正逻辑",
            display_frame=display_frame,
        )

    def update(self, raw_frame, corrected_frame, now=None):
        now = time.monotonic() if now is None else float(now)
        if self.state == self.HEAD_SCAN:
            return self._update_head_scan(raw_frame, now)
        if self.state == self.BACKTRACK:
            return self._update_backtrack(corrected_frame, now)
        if self.state == self.TURN_RECOVERY:
            return self._update_turn_recovery(corrected_frame, now)
        return self._event(self.state, "巡线终点恢复控制器当前状态：{}".format(self.state))
