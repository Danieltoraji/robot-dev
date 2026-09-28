#!/usr/bin/python3
# coding=utf8
"""
tag_route_demo.py — 基于 AprilTag 的快速演示导航

本模块只负责：
  1. 按预先配置的 Tag 路线导航；
  2. 支持候选 Tag（例如 2 或 72）；
  3. 允许跳过 optional Tag；找到当前 Tag 后锁定并接近，直到连续丢失才判定到达；
  4. 在球门区域把控制权交给上层射门模块；
  5. 射门结果由上层通过 notify_shot_result() 回传。

本模块不负责：
  - 足球检测、球门柱检测和踢球；
  - 球是否越过球门线的判断；
  - 修改 FootballKick.py、finalkick.py 或 Running.py。

当前已确认的路线：

第一球门：
  到达第一个 Tag 65（或路线以后改为 26）后，再 turn_left x7 做初始方向校正；
  65（必须找到） -> 26（必须找到、第一次找球门柱）
  -> 50（可跳过、26 处没找到球门柱时的备用位置） -> 球门 Tag 61

第二球门：
  第一球门阶段完成后：back x5 -> turn_right x3
  直接寻找 Tag 82（必须找到、找球门柱） -> 球门 Tag 106

出口：
  第二球门阶段完成后：back x4 -> turn_right x3
  46（可跳过） -> 108（必须找到） -> 37（必须找到、出口）

重要：
  “Tag 65 和 Tag 26 都必须找到”；“Tag 50 可跳过”表示找到 Tag 26 后先搜索球门柱，失败时才寻找 Tag 50，
  而是说在 Tag 26 处已经找到球门柱并完成射门准备时，不必再去 Tag 50。
"""

from __future__ import print_function

import time
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# 路线配置
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RouteStep:
    """一个路线节点。tags 中多个 ID 表示等价候选。"""

    name: str
    tags: Tuple[int, ...]
    required: bool = True
    goal_checkpoint: bool = False
    fallback_for_goal_step: Optional[int] = None


@dataclass(frozen=True)
class StagePlan:
    name: str
    goal_tag: Optional[int]
    steps: Tuple[RouteStep, ...]
    # 进入本阶段后，第一次确认指定 Tag 时执行的动作。
    initial_trigger_tags: Tuple[int, ...] = ()
    initial_actions: Tuple[Tuple[str, int], ...] = ()
    after_success_actions: Tuple[Tuple[str, int], ...] = ()


STAGE1 = StagePlan(
    name="stage1_goal_61",
    goal_tag=61,
    steps=(
        RouteStep("stage1_required_65", (65,), required=True),
        RouteStep(
            "stage1_goal_search_at_26",
            (26,),
            required=True,
            goal_checkpoint=True,
            fallback_for_goal_step=2,
        ),
        RouteStep(
            "stage1_goal_search_fallback_at_50",
            (50,),
            required=False,
            goal_checkpoint=True,
        ),
    ),
    # 机器人刚进入场地时朝前；到达第一个路线 Tag 后固定左转七步校正。
    initial_trigger_tags=(65,),
    initial_actions=(("turn_left", 7),),
    # 第一球门射门确认后，进入第二球门路线前的重新定向。
    after_success_actions=(("back", 5), ("turn_right", 3)),
)

STAGE2 = StagePlan(
    name="stage2_goal_106",
    goal_tag=106,
    steps=(
        # 第二阶段不再经过 Tag 2 或 Tag 72，直接寻找 Tag 82。
        RouteStep(
            "stage2_goal_search_at_82",
            (82,),
            required=True,
            goal_checkpoint=True,
        ),
    ),
    # 第二球门射门确认后，进入出口路线前的重新定向。
    after_success_actions=(("back", 4), ("turn_right", 3)),
)

EXIT_ROUTE = StagePlan(
    name="exit",
    goal_tag=None,
    steps=(
        RouteStep("exit_optional_46", (46,), required=False),
        RouteStep("exit_required_108", (108,), required=True),
        RouteStep("exit_required_37", (37,), required=True),
    ),
)

DEMO_PLAN: Tuple[StagePlan, ...] = (STAGE1, STAGE2, EXIT_ROUTE)


# ---------------------------------------------------------------------------
# AprilTag 检测结果
# ---------------------------------------------------------------------------

# 与 locate_robot.py 保持一致：先在加白边的图像上检测，再把坐标移回原图。
LOCATE_ROBOT_DETECTION_BORDER_PAD = 64

@dataclass
class TagObservation:
    tag_id: int
    center_x: float
    center_y: float
    width: float
    height: float
    area: float
    corners: object = None


class AprilTagDetector:
    """按 locate_robot.py 的方式检测 AprilTag。

    这里复用 locate_robot.py 中已验证的关键流程：
      - tag36h11；
      - refine_edges / refine_decode；
      - quad_decimate=1.0、quad_contours=True；
      - 四周加 64 像素白边，检测后减回该偏移。

    导航只需要图像坐标，因此保留的是移回原始画面后的 raw detection，
    不把 locate_robot 的 PnP 全局定位流程混入 Tag 导航。
    """

    def __init__(self, border_pad=LOCATE_ROBOT_DETECTION_BORDER_PAD):
        import cv2
        import numpy as np
        import hiwonder.apriltag as apriltag

        self.cv2 = cv2
        self.np = np
        self.border_pad = int(border_pad)
        options = apriltag.DetectorOptions(
            families="tag36h11",
            border=1,
            nthreads=4,
            quad_decimate=1.0,
            quad_blur=0.0,
            refine_edges=True,
            refine_decode=True,
            refine_pose=False,
            quad_contours=True,
        )
        self.detector = apriltag.Detector(
            options=options,
            searchpath=apriltag._get_demo_searchpath(),
        )

    def detect(self, frame) -> List[TagObservation]:
        cv2 = self.cv2
        np = self.np
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pad = self.border_pad
        gray_padded = cv2.copyMakeBorder(
            gray, pad, pad, pad, pad,
            cv2.BORDER_CONSTANT,
            value=255,
        )
        detections_padded = self.detector.detect(
            gray_padded, return_image=False
        )
        # 与 locate_robot._remove_detection_padding() 相同：
        # 检测结果先恢复到原始 640x480 图像坐标系，再交给导航逻辑。
        raw_detections = [
            self._remove_detection_padding(detection, pad)
            for detection in detections_padded
        ]

        observations = []
        for detection in raw_detections:
            corners = np.asarray(detection.corners, dtype=np.float32).reshape(4, 2)
            center = np.asarray(detection.center, dtype=np.float32).reshape(2)

            x_min = float(corners[:, 0].min())
            x_max = float(corners[:, 0].max())
            y_min = float(corners[:, 1].min())
            y_max = float(corners[:, 1].max())
            contour_area = abs(float(cv2.contourArea(corners.astype(np.float32))))
            observations.append(TagObservation(
                tag_id=int(detection.tag_id),
                center_x=float(center[0]),
                center_y=float(center[1]),
                width=max(0.0, x_max - x_min),
                height=max(0.0, y_max - y_min),
                area=contour_area,
                corners=corners,
            ))
        return observations

    @staticmethod
    def _remove_detection_padding(detection, pad):
        """把 padding 图像坐标移回原始图像坐标。"""
        np = __import__("numpy")
        corners = np.asarray(detection.corners, dtype=np.float32).copy()
        corners[:, 0] -= pad
        corners[:, 1] -= pad
        center = np.asarray(detection.center, dtype=np.float32).copy()
        center[0] -= pad
        center[1] -= pad
        # hiwonder.apriltag 的 Detection 是 namedtuple；保留其余字段。
        return detection._replace(corners=corners, center=center)


# ---------------------------------------------------------------------------
# 动作执行器
# ---------------------------------------------------------------------------

class ActionGroupMotion:
    """对 TonyPi ActionGroupControl 的很薄一层封装。

    导航状态机也可以接收一个假的 motion 对象进行电脑端测试，只要它提供
    run(action_name) 方法即可。
    """

    def __init__(self, action_group_control=None):
        if action_group_control is None:
            import hiwonder.ActionGroupControl as AGC
            action_group_control = AGC
        self.agc = action_group_control

    def run(self, action_name: str) -> None:
        self.agc.runActionGroup(action_name)

    def run_repeated(self, action_name: str, repeats: int) -> None:
        for _ in range(int(repeats)):
            self.run(action_name)


class NullMotion:
    """测试用动作执行器：只记录动作，不驱动机器人。"""

    def __init__(self):
        self.actions = []

    def run(self, action_name: str) -> None:
        self.actions.append(action_name)


# ---------------------------------------------------------------------------
# 导航状态机
# ---------------------------------------------------------------------------

@dataclass
class NavigationEvent:
    kind: str
    state: str
    stage: str
    message: str
    goal_tag: Optional[int] = None
    step_name: Optional[str] = None


class TagRouteNavigator:
    """基于视觉 Tag 的路线状态机。

    外部主循环每得到一帧图像，就调用一次 update()：

        observations = tag_detector.detect(frame)
        event = navigator.update(observations, frame_width=640,
                                 goal_ready=goal_detector(frame))

    当 event.kind == "GOAL_READY" 时，上层停止 Tag 导航并启动 FootballKick。
    射门完成后，上层调用：

        navigator.notify_shot_result(crossed=True)

    底层接口仍保留 crossed=False 的重试能力；当前 TagWalkDemo 默认每个球门
    最多一脚，未越线时由上层调用 notify_goal_abandoned() 推进下一阶段。
    """

    NAVIGATE = "NAVIGATE"
    SEARCH_GOAL = "SEARCH_GOAL"
    WAIT_SHOT = "WAIT_SHOT"
    DONE = "DONE"

    def __init__(
        self,
        plan: Sequence[StagePlan] = DEMO_PLAN,
        motion=None,
        tag_confirm_frames: int = 3,
        tag_lost_confirm_frames: int = 5,
        tag_lost_confirm_s: float = 0.6,
        min_tag_approach_s: float = 1.0,
        approach_action_interval_s: float = 0.45,
        arrival_height_px: float = 110.0,  # 兼容旧参数；当前不再用于推进路线
        center_tolerance_px: float = 65.0,
        goal_search_timeout_s: float = 4.0,
        action_sleep_s: float = 0.0,
    ):
        self.plan = tuple(plan)
        self.motion = motion if motion is not None else NullMotion()
        self.tag_confirm_frames = max(1, int(tag_confirm_frames))
        self.tag_lost_confirm_frames = max(1, int(tag_lost_confirm_frames))
        self.tag_lost_confirm_s = max(0.0, float(tag_lost_confirm_s))
        self.min_tag_approach_s = max(0.0, float(min_tag_approach_s))
        self.approach_action_interval_s = max(0.0, float(approach_action_interval_s))
        self.arrival_height_px = float(arrival_height_px)
        self.center_tolerance_px = float(center_tolerance_px)
        self.goal_search_timeout_s = float(goal_search_timeout_s)
        self.action_sleep_s = float(action_sleep_s)
        self.reset()

    def reset(self) -> None:
        self.stage_index = 0
        self.step_index = 0
        self.state = self.NAVIGATE
        self.goal_search_started_at = None
        self.last_target_id = None
        self.target_streak = 0
        self.locked_target_id = None
        self.target_lost_streak = 0
        self.target_locked_at = None
        self.last_target_seen_at = None
        self.next_approach_action_at = 0.0
        self.initial_actions_done = False
        self.last_event = None

    @property
    def current_stage(self) -> StagePlan:
        return self.plan[self.stage_index]

    @property
    def current_step(self) -> Optional[RouteStep]:
        stage = self.current_stage
        if self.step_index >= len(stage.steps):
            return None
        return stage.steps[self.step_index]

    def update(
        self,
        observations: Iterable[TagObservation],
        frame_width: int = 640,
        goal_ready: bool = False,
        now: Optional[float] = None,
    ) -> NavigationEvent:
        """处理一帧检测结果；锁定当前 Tag，接近到连续丢失后再切换节点。"""
        now = time.monotonic() if now is None else float(now)
        observations = list(observations or [])

        if self.state == self.DONE:
            return self._event("DONE", "任务已完成")

        if self.state == self.WAIT_SHOT:
            return self._event("WAIT_SHOT", "等待上层射门模块回传越线结果")

        if self.state == self.SEARCH_GOAL:
            if goal_ready:
                self.state = self.WAIT_SHOT
                return self._event(
                    "GOAL_READY",
                    "已检测到球门区域，交给上层射门模块",
                    goal_tag=self.current_stage.goal_tag,
                    step_name=self.current_step.name,
                )
            if self.goal_search_started_at is None:
                self.goal_search_started_at = now
            elapsed = now - self.goal_search_started_at
            if elapsed >= self.goal_search_timeout_s:
                fallback = self.current_step.fallback_for_goal_step
                if fallback is not None:
                    self.step_index = int(fallback)
                    self.state = self.NAVIGATE
                    self.goal_search_started_at = None
                    self._clear_target_streak()
                    return self._event(
                        "GOAL_SEARCH_FALLBACK",
                        "当前球门位置未找到球门柱，前往备用 Tag",
                        step_name=self.current_step.name,
                    )
                return self._event(
                    "GOAL_SEARCH_TIMEOUT",
                    "当前球门位置未找到球门柱，且没有备用位置",
                    goal_tag=self.current_stage.goal_tag,
                    step_name=self.current_step.name,
                )
            return self._event(
                "SEARCHING_GOAL",
                "在当前 Tag 附近寻找球门柱",
                goal_tag=self.current_stage.goal_tag,
                step_name=self.current_step.name,
            )

        # 普通路线导航。先允许在“尚未锁定 Tag”时跳过 optional 节点；
        # 一旦锁定了某个 Tag，就必须先完成它的接近流程，不能被其他 Tag 打断。
        visible = self._best_visible_by_id(observations)
        if self.locked_target_id is None:
            self._skip_optional_steps_if_later_target_visible(visible)

        step = self.current_step
        if step is None:
            return self._event("ROUTE_ERROR", "当前路线没有可执行的步骤")

        # 已锁定目标：持续追踪并接近它。目标暂时漏检时不立即切换，
        # 只有连续丢失且满足最短接近时间，才把它视为已经到达。
        if self.locked_target_id is not None:
            locked = visible.get(int(self.locked_target_id))
            if locked is not None:
                self.target_lost_streak = 0
                self.last_target_seen_at = now
                action_message = self._approach_locked_target(
                    locked, frame_width, now
                )
                return self._event(
                    "APPROACH_TAG",
                    action_message,
                    step_name=step.name,
                )

            self.target_lost_streak += 1
            last_seen_at = (
                self.last_target_seen_at
                if self.last_target_seen_at is not None
                else now
            )
            lost_for = max(0.0, now - last_seen_at)
            locked_for = max(0.0, now - (self.target_locked_at or now))
            if (
                self.target_lost_streak >= self.tag_lost_confirm_frames
                and lost_for >= self.tag_lost_confirm_s
            ):
                if locked_for < self.min_tag_approach_s:
                    target_id = self.locked_target_id
                    self._clear_target_lock()
                    self._clear_target_streak()
                    return self._event(
                        "REACQUIRE_TAG",
                        "目标 Tag 过早丢失，暂不判定到达，重新寻找 Tag {}".format(
                            target_id
                        ),
                        step_name=step.name,
                    )
                return self._arrive_at_locked_target(now, step)

            return self._event(
                "APPROACH_TAG_LOST",
                "接近过程中暂时未看到已锁定的 Tag {}，等待连续丢失确认".format(
                    self.locked_target_id
                ),
                step_name=step.name,
            )

        # 尚未锁定目标：先连续确认当前路线目标，确认后才开始接近。
        target = self._select_target(step, visible)
        if target is None:
            self._clear_target_streak()
            return self._event(
                "SEARCH_TAG",
                "未看到当前目标 Tag，继续搜索",
                step_name=step.name,
            )

        self._update_target_streak(target.tag_id)
        if self.target_streak < self.tag_confirm_frames:
            return self._event(
                "TRACK_TAG",
                "目标 Tag 已看到，等待连续帧确认",
                step_name=step.name,
            )

        self.locked_target_id = int(target.tag_id)
        self.target_lost_streak = 0
        self.target_locked_at = now
        self.last_target_seen_at = now
        self.next_approach_action_at = 0.0
        # 初始转向不能在“刚看到第一个 Tag”时执行；否则机器人还没有
        # 走到第一个 Tag，就可能提前改变朝向。初始转向在
        # _arrive_at_locked_target() 中、确认已经到达该 Tag 后执行。
        self._clear_target_streak()
        return self._event(
            "TAG_LOCKED",
            "已锁定 Tag {}，开始接近；连续丢失后才判定到达".format(
                self.locked_target_id
            ),
            step_name=step.name,
        )
    def _complete_goal_stage(self, message, event_kind):
        """完成当前球门阶段并进入下一阶段。"""
        completed_stage = self.stage_index
        self._run_actions(self.current_stage.after_success_actions)
        self.stage_index += 1
        self.step_index = 0
        self.goal_search_started_at = None
        self.initial_actions_done = False
        self._clear_target_streak()
        self._clear_target_lock()

        if self.stage_index >= len(self.plan):
            self.state = self.DONE
            return self._event(event_kind, message)

        self.state = self.NAVIGATE
        return self._event(
            event_kind,
            message,
            goal_tag=self.current_stage.goal_tag,
        )

    def notify_shot_result(self, crossed: bool) -> NavigationEvent:
        """由上层射门模块回传越线结果。

        只要当前阶段的足球已经确认完全越过球门线，就完成当前球门阶段。
        不能因为机器人是在 Tag 导航状态 NAVIGATE 中看到足球，就把已经
        成功的第一球门射门误判成“尚未完成必经 Tag”。

        Tag 路线仍然保留：如果没有看到足球，机器人继续按 65 -> 26 -> 50、
        直接寻找 Tag 82 等路线寻找球门区域；如果已经看到球并成功进门，射门
        本身就是当前球门任务的完成条件，随后立即执行后退/转向进入下一阶段。
        """
        if self.state != self.WAIT_SHOT:
            return self._event("UNEXPECTED_SHOT_RESULT", "当前不在等待射门结果状态")

        if not crossed:
            # 不改变当前路线阶段：留在当前球门搜索状态，继续找球并重试。
            self.state = self.SEARCH_GOAL
            self.goal_search_started_at = time.monotonic()
            return self._event(
                "SHOT_FAILED_RETRY",
                "足球未越线，保留当前球门位置并允许重新射门",
                goal_tag=self.current_stage.goal_tag,
                step_name=self.current_step.name if self.current_step is not None else None,
            )

        # 成功越线即完成当前球门阶段，不再回到第一球门的剩余 Tag 路线。
        return self._complete_goal_stage(
            "射门越线确认，进入下一阶段路线",
            "STAGE_ADVANCE",
        )

    def notify_goal_abandoned(
        self,
        message="放弃当前球门，进入下一阶段路线",
        event_kind="GOAL_ABANDONED_ADVANCE",
    ) -> NavigationEvent:
        """上层因射门次数达到上限或持续丢球而放弃当前球门。"""
        if self.state not in (self.SEARCH_GOAL, self.WAIT_SHOT):
            return self._event(
                "UNEXPECTED_GOAL_ABANDON",
                "当前不在球门搜索或等待射门状态，不能推进球门阶段",
            )
        return self._complete_goal_stage(message, event_kind)

    def notify_goal_completed_without_shot(self) -> NavigationEvent:
        """Demo 专用：已检测并对齐球门柱，跳过射门直接进入下一阶段。"""
        if self.state != self.SEARCH_GOAL:
            return self._event(
                "UNEXPECTED_GOAL_COMPLETE",
                "当前不在球门搜索状态，不能结束球门阶段",
            )
        return self._complete_goal_stage(
            "已检测并对齐球门柱，跳过射门进入下一阶段路线",
            "GOAL_APPROACH_COMPLETE",
        )
    def notify_goal_not_found(self) -> NavigationEvent:
        """允许上层显式结束当前球门位置搜索，通常会触发备用 Tag。"""
        if self.state != self.SEARCH_GOAL:
            return self._event("UNEXPECTED_GOAL_RESULT", "当前不在寻找球门状态")
        fallback = self.current_step.fallback_for_goal_step
        if fallback is None:
            return self._event("GOAL_NOT_FOUND", "未找到球门柱，当前步骤无备用 Tag")
        self.step_index = int(fallback)
        self.state = self.NAVIGATE
        self.goal_search_started_at = None
        self._clear_target_streak()
        return self._event("GOAL_SEARCH_FALLBACK", "前往备用球门搜索 Tag", step_name=self.current_step.name)

    def notify_goal_ready(self) -> NavigationEvent:
        """在已经进入球门搜索区域后暂停 Tag 导航，交给上层射门。

        NAVIGATE 阶段即使画面中出现足球，也必须继续完成当前 Tag 路线；
        只有到达带有 goal_checkpoint 的路线步骤、进入 SEARCH_GOAL，
        才允许上层把控制权交给足球/球门柱射门状态机。
        """
        if self.state not in (self.SEARCH_GOAL, self.WAIT_SHOT):
            return self._event(
                "UNEXPECTED_GOAL_READY",
                "尚未进入球门搜索区域，继续 Tag 导航，不启动射门",
            )
        self.state = self.WAIT_SHOT
        return self._event(
            "BALL_PRIORITY_STOP",
            "检测到足球，已停止 Tag 导航，先对准足球再寻找球门柱",
            goal_tag=self.current_stage.goal_tag,
            step_name=self.current_step.name if self.current_step is not None else None,
        )

    def _run_initial_actions_if_needed(self, tag_id: int) -> bool:
        stage = self.current_stage
        if self.initial_actions_done:
            return False
        if int(tag_id) not in stage.initial_trigger_tags:
            return False
        self._run_actions(stage.initial_actions)
        self.initial_actions_done = True
        return True

    def _best_visible_by_id(self, observations):
        result = {}
        for observation in observations:
            old = result.get(int(observation.tag_id))
            if old is None or observation.area > old.area:
                result[int(observation.tag_id)] = observation
        return result

    def _select_target(self, step, visible):
        candidates = [visible[tag_id] for tag_id in step.tags if tag_id in visible]
        if not candidates:
            return None
        # 同组候选优先选择面积较大的，即通常更近、检测更稳定的 Tag。
        return max(candidates, key=lambda item: item.area)

    def _skip_optional_steps_if_later_target_visible(self, visible):
        """只向前跳过 optional 节点；遇到 required 节点立即停止。"""
        while True:
            step = self.current_step
            if step is None or step.required:
                return
            later_index = None
            for index in range(self.step_index + 1, len(self.current_stage.steps)):
                later = self.current_stage.steps[index]
                if any(tag_id in visible for tag_id in later.tags):
                    later_index = index
                    break
                if later.required:
                    # 不能因为看到了更远处的 Tag 就跳过一个必须经过的节点。
                    return
            if later_index is None:
                return
            self.step_index = later_index
            self._clear_target_streak()

    def _update_target_streak(self, tag_id):
        if self.last_target_id == int(tag_id):
            self.target_streak += 1
        else:
            self.last_target_id = int(tag_id)
            self.target_streak = 1

    def _clear_target_streak(self):
        self.last_target_id = None
        self.target_streak = 0

    def _clear_target_lock(self):
        self.locked_target_id = None
        self.target_lost_streak = 0
        self.target_locked_at = None
        self.last_target_seen_at = None
        self.next_approach_action_at = 0.0

    def _approach_locked_target(self, target, frame_width, now):
        """根据锁定 Tag 的水平位置进行小幅修正或前进。"""
        if now < self.next_approach_action_at:
            return (
                "已锁定 Tag {}，等待上一个动作完成".format(
                    self.locked_target_id
                )
            )

        image_center = float(frame_width) / 2.0
        if target.center_x < image_center - self.center_tolerance_px:
            action = "turn_left"
            action_message = "Tag {} 偏左，执行 turn_left 对准".format(
                self.locked_target_id
            )
        elif target.center_x > image_center + self.center_tolerance_px:
            action = "turn_right"
            action_message = "Tag {} 偏右，执行 turn_right 对准".format(
                self.locked_target_id
            )
        else:
            action = "go_forward_fast"
            action_message = "Tag {} 已基本居中，执行 go_forward_fast 接近".format(
                self.locked_target_id
            )

        self.motion.run(action)
        self.next_approach_action_at = now + self.approach_action_interval_s
        return action_message

    def _arrive_at_locked_target(self, now, step):
        reached_name = step.name
        reached_tag = self.locked_target_id
        self._clear_target_lock()
        self._clear_target_streak()

        # 第一阶段的初始左转是在真正到达第一个 Tag 后执行，
        # 然后再开始搜索下一个路线 Tag，而不是在刚看到 Tag 时转身。
        initial_correction = self._run_initial_actions_if_needed(reached_tag)

        if step.goal_checkpoint:
            self.state = self.SEARCH_GOAL
            self.goal_search_started_at = now
            return self._event(
                "TAG_ARRIVED_GOAL_SEARCH",
                "Tag {} 已连续丢失，认为已到达球门路线点，进入球门柱搜索".format(
                    reached_tag
                ),
                goal_tag=self.current_stage.goal_tag,
                step_name=reached_name,
            )

        self.step_index += 1
        if self.current_step is None:
            if self.stage_index == len(self.plan) - 1:
                self.state = self.DONE
                return self._event(
                    "MISSION_COMPLETE",
                    "Tag {} 已到达，任务完成".format(reached_tag),
                    step_name=reached_name,
                )
            return self._event(
                "ROUTE_REACHED_END",
                "Tag {} 已到达，已完成本阶段路线节点".format(reached_tag),
                step_name=reached_name,
            )

        if initial_correction:
            return self._event(
                "INITIAL_CORRECTION",
                "已到达第一个 Tag {}，执行初始左转 7 步；开始寻找下一个路线 Tag".format(
                    reached_tag
                ),
                step_name=reached_name,
            )

        return self._event(
            "TAG_ARRIVED",
            "Tag {} 已连续丢失，认为已到达；切换下一个路线目标".format(
                reached_tag
            ),
            step_name=reached_name,
        )

    def _run_actions(self, actions):
        for action_name, repeats in actions:
            for _ in range(int(repeats)):
                self.motion.run(action_name)
                self._sleep_after_action()

    def _sleep_after_action(self):
        if self.action_sleep_s > 0:
            time.sleep(self.action_sleep_s)

    def _event(self, kind, message, goal_tag=None, step_name=None):
        event = NavigationEvent(
            kind=kind,
            state=self.state,
            stage=self.current_stage.name,
            message=message,
            goal_tag=goal_tag,
            step_name=step_name,
        )
        self.last_event = event
        return event


# ---------------------------------------------------------------------------
# 便捷入口
# ---------------------------------------------------------------------------

def build_demo_navigator(execute_actions=False, **kwargs):
    """构造当前项目路线的导航器。

    默认使用 NullMotion，适合电脑端验证路线状态机；机器人实机运行时传入
    execute_actions=True，才会使用 ActionGroupControl 执行动作。
    """
    motion = ActionGroupMotion() if execute_actions else NullMotion()
    return TagRouteNavigator(plan=DEMO_PLAN, motion=motion, **kwargs)


if __name__ == "__main__":
    print("Tag route demo configuration")
    for stage in DEMO_PLAN:
        print("-", stage.name, "goal=", stage.goal_tag)
        for index, step in enumerate(stage.steps):
            print("  {}. {} tags={} required={} goal_checkpoint={}".format(
                index, step.name, step.tags, step.required, step.goal_checkpoint
            ))







