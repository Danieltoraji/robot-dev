# -*- coding: utf-8 -*-
"""数字宫格 —— **参考原版通关代码**主流程（levels/nine_grid_original/level.py）

来源：`reference code/九宫格视觉导航/`（叶雨岑、陶鲁玥，2026-08）
      ── 主控流程是 `robot/mainv0.2.ipynb` 的 CELL 4（go_to / turn_to /
         search_target / arrived）+ CELL 5（初始化与逐格调用），本文件是它的
      逐行搬运版；动作层与视觉层分别搬在同包的
      `action.py` / `vision.py` / `classifier.py`。

**这是什么**：这条链路不是本仓库写的算法，而是"别人已经通关的那一版"原样接入，
用来与现行的 `nine_grid`（统一决策）和 `nine_grid_three_stage`（三段式）做对照
（A/B）。原版的通关思路是四层有限状态机：

    搜索 SEARCH（search_target：最多 20 次大转，前视/低头各试一遍）
        ↓ 找到
    方向跟踪 TRACKING（turn_to：|yaw|>30° 大转；8°~30° 按**在线估计**的小转角批量微调）
        ↓ 已对准
    前进修正 MOVE_FORWARD（go_to 的 while proximity < ARRIVE_THRESHOLD：
        大步前进 → 重拍 → 对准；框宽变化 <5px 判为"没走近" → 后退 + 越障重试）
        ↓ 框宽够近
    最终接近 FINAL_APPROACH（arrived：低头拍**颜色占比**，
        "看到了颜色 → 颜色消失" 判为压过目标格；期间小步前进 + 小转修正）

**入口**：`python main.py nine_grid_original`（包入口见 `__init__.py`）

=====================================================================
与参考版的差异（全部只为"接入本仓库"，**没有一处改判定**）
=====================================================================
  1. `run_level(state)` 是本仓库的关卡入口约定（参考版是 notebook 里手动逐个
     `go_to(1..7)`）。本文件把它写成 `for k in 1..7: go_to(k)`——与参考版
     CELL 5 的末次测试 `while cur < 7: go_to(cur+1, 0, 0)` 一致：`go_to` 内部
     按 id 覆盖三个门限，所以外面传什么值都无所谓。
  2. 门限表：参考版是 `go_to()` 里 7 段 `if/elif`，这里原值不动地写成模块级
     `GO_TO_PARAMS`（1:1 转写，见文件内注释与 docs 常量表）。
  3. 拍照/动作改为可绑定 `RobotState`（复用真机同一条拍照命令、并让 main.py 的
     trace 记到动作数）；不绑定时退回参考版自己的 fswebcam 调用。
  4. `search_target` 20 次转完仍找不到目标时，参考版是抛 `RuntimeError` 直接
     中断 notebook；这里是捕获它、打印、**结束本关并返回 False**（顺序计分下
     断一格后面全不算，不该闷头往下走，交人工判断）。

⚠️ **本轮不改任何标定常量**（用户要求）：所有阈值都是参考版在**另一台机器人、
   另一块场地**上标的。哪些必须重标、现行 nine_grid 的对应值是多少，
   见 `docs/关卡算法/彩色数字九宫格-nine_grid/参考原版通关代码-nine_grid_original.md`。

⚠️ **本关没有仿真器**：参考版的视觉 ↔ 动作耦合（颜色占比 + 框宽 + 云台姿态）
   没有对应的仿真模型，直接照搬会在仿真里给出假结论。要验证只能上真机。
"""

import time

from . import action, vision
from .vision import (
    capture_image,
    id_to_color,
    identify,
    identify_color,
)


# =====================================================================
# 门限表（参考版 go_to() 里的 7 段 if/elif，**数值一字未改**）
# =====================================================================
# id → (ARRIVE_THRESHOLD, COLOR_THRESHOLD, COLOR_LOST_THRESHOLD)
#   ARRIVE_THRESHOLD：目标框宽（px @1280×980）达到多少就转入最终接近段
#   COLOR_THRESHOLD ："看到目标颜色"的整帧占比门
#   COLOR_LOST_THRESHOLD："颜色已消失（=压过该格）"的整帧占比门
#
# ⚠️ 参考版内部有两处**自相矛盾**的记录（照抄保留，未擅自统一）：
#   · id 6：函数内表是 0.002 / 0.003；notebook CELL 13 调用时写 0.01 / 0.003，
#     被注释掉的旧表也写 0.01 / 0.003 ⇒ 实际生效的是 **0.002 / 0.003**。
#   · id 4：CELL 11 调用时写 0.18 / 0.12，函数内表是 **0.14 / 0.12**（表生效）。
GO_TO_PARAMS = {
    1: (450, 0.10, 0.05),
    2: (450, 0.05, 0.01),
    3: (370, 0.05, 0.015),
    4: (450, 0.14, 0.10),
    5: (450, 0.17, 0.10),
    6: (450, 0.002, 0.003),
    7: (450, 0.15, 0.10),
}

# go_to() 的默认门限（参考版 CELL 1 的模块级初值；id 不在表里时才用）
DEFAULT_ARRIVE_THRESHOLD = 430
DEFAULT_COLOR_THRESHOLD = 0.0
DEFAULT_COLOR_LOST_THRESHOLD = 0.0

# 远距前进阶段的"前进效率"判据：一次前进后框宽变化 < 此值 ⇒ 判为没走近
FORWARD_EFFICIENCY_MIN_PX = 5

# ---------------------------------------------------------------------
# turn_to() 的自适应小转参数
# ---------------------------------------------------------------------
BIG_TURN_THRESHOLD = 30       # |yaw| > 30° ⇒ 直接大转
EPSILON = 8                   # |yaw| ≤ 8° ⇒ 认为已经对准
INITIAL_SMALL_TURN_ANGLE = 4  # 一次小转的**初估**角度（在线估计会把它纠回来）
MIN_EFFECTIVE_TURN = 0.5      # 一批小转平均每步 < 0.5° ⇒ 判为"小转失效"，改大转
TURN_ESTIMATE_ALPHA = 0.3     # 小转角度的指数滑动平均系数

# ---------------------------------------------------------------------
# arrived()（最终接近段）的参数
# ---------------------------------------------------------------------
ARRIVED_SMALL_TURN_INIT = 3.0     # 最终段的小转初估角度（参考版局部值，不是 4）
ARRIVED_MIN_EFFECTIVE_TURN = 0.5  # id 6 时取 0（紫牌阈值本来就低，放宽）
ARRIVED_MIN_RATIO_CHANGE = 0.002  # 小步前进后颜色占比变化 < 此值 ⇒ 判为没走近
ARRIVED_MIN_RATIO_CHANGE_ID6 = 0  # id 6 时取 0
MAX_FINAL_TURNS = 3               # 已经看到颜色后，一批小转最多 3 次
PURPLE_ID = 6                     # 需要特殊放宽的紫牌

# 数字识别连续丢失几次就重新搜索（参考版：累计到 1 次即重搜）
LOST_BEFORE_SEARCH = 1

# ---------------------------------------------------------------------
# 搜索段
# ---------------------------------------------------------------------
SEARCH_MAX_TURNS = 20   # search_target 最多左转多少次（参考版字面量 20 / README 亦记 20）

# ---------------------------------------------------------------------
# 【接入新增】进场动作开关
# ---------------------------------------------------------------------
# 参考版 notebook 在通关前手动执行过 CELL 2/3：`over_hurdle()`（爬 + 后退两步，
# 即从上一位关卡/栏架进入九宫格场地）+ `move_backward_one()`。
# 本仓库的进场由前序关卡（stairs_hurdle 等）负责，故**默认不自动执行**；
# 需要复现参考版完整流程时把这里改成 True（或单独调用 enter_field()）。
ENTER_FIELD_ON_START = False


def enter_field():
    """参考版 CELL 2 + CELL 3 的进场动作（越障进入九宫格 + 后退一步）"""
    action.over_hurdle()
    action.move_backward_one()


# =====================================================================
# 顶层导航 go_to()
# =====================================================================

def go_to(id, COLOR_THRESHOLD=None, COLOR_LOST_THRESHOLD=None):
    """从当前位置走向数字 id 的面板（参考版 CELL 4 的逐行搬运）

    两个颜色门限参数保留在签名里是为了与参考版一致，但**表里的值总是覆盖它们**
    （参考版 7 段 if/elif 的语义）；id 不在表里时才用传入值/默认值。
    """
    ARRIVE_THRESHOLD, COLOR_THRESHOLD, COLOR_LOST_THRESHOLD = GO_TO_PARAMS.get(
        id,
        (DEFAULT_ARRIVE_THRESHOLD,
         DEFAULT_COLOR_THRESHOLD if COLOR_THRESHOLD is None else COLOR_THRESHOLD,
         DEFAULT_COLOR_LOST_THRESHOLD if COLOR_LOST_THRESHOLD is None
         else COLOR_LOST_THRESHOLD),
    )

    action.servo_look_forward()

    image = capture_image()

    proximity = turn_to(id, image)

    # =====================================================
    # 前进效率检测：一次前进后 proximity 变化 < 5，
    # 认为机器人前进没有接近目标 ⇒ 后退一步 + 越障脱困，再重新识别
    # =====================================================

    previous_proximity = proximity

    # --------------------
    # 第一阶段：远距离导航（大步前进）
    # --------------------
    while proximity < ARRIVE_THRESHOLD:

        action.move_forward()

        image = capture_image()

        proximity = turn_to(id, image)

        proximity_change = abs(
            proximity - previous_proximity
        )

        print(
            f"Forward efficiency: "
            f"proximity {previous_proximity:.1f} -> "
            f"{proximity:.1f}, "
            f"change={proximity_change:.1f}"
        )

        if proximity_change < FORWARD_EFFICIENCY_MIN_PX:

            print(
                "⚠️ Forward movement ineffective "
                f"(proximity change < {FORWARD_EFFICIENCY_MIN_PX}), "
                "moving backward one step."
            )

            action.move_backward_one()
            action.over_hurdle()

            # 后退会改变距离与偏角 ⇒ 必须重新识别，不能沿用旧 proximity
            image = capture_image()

            proximity = turn_to(id, image)

            print(
                f"After backward recovery: "
                f"proximity={proximity:.1f}"
            )

        previous_proximity = proximity

    # --------------------
    # 第二阶段：接近目标（小步前进 + 颜色检测）
    # --------------------
    arrived(
        id,
        COLOR_THRESHOLD,
        COLOR_LOST_THRESHOLD
    )

    return


# =====================================================================
# 方向跟踪 turn_to()
# =====================================================================

def turn_to(id, image):
    """把机器人转向目标数字；返回 proximity（框宽）

    自适应小转：小转一步的实际角度**在线估计**（指数滑动平均），
    一批小转后按实际转过的角度更新估计；若平均每步 < MIN_EFFECTIVE_TURN
    （地面摩擦导致小转几乎无效），直接升级为一次大转。
    """

    # 当前估计的一次小转实际角度
    small_turn_angle = INITIAL_SMALL_TURN_ANGLE

    # 第一次优先使用已经拍好的 image
    success, id_direction, proximity = identify(
        id,
        image
    )

    # 第一次识别失败才搜索
    if not success:

        id_direction, proximity = search_target(id)

    lost_count = 0

    while True:

        # =================================
        # 1. 已经对准
        # =================================
        if abs(id_direction) <= EPSILON:

            print(
                f"Turn finished. "
                f"yaw={id_direction:.1f}°, "
                f"proximity={proximity}"
            )

            return proximity

        # =================================
        # 2. 大角度修正
        # =================================
        if abs(id_direction) > BIG_TURN_THRESHOLD:

            print(
                f"Large correction: "
                f"{id_direction:.1f} deg"
            )

            if id_direction > 0:
                action.turn_big_angle_left()
            else:
                action.turn_big_angle_right()

        # =================================
        # 3. 小角度修正
        # =================================
        else:

            turn_count = max(
                1,
                int(abs(id_direction) / small_turn_angle)
            )

            print(
                f"Small correction: "
                f"yaw={id_direction:.1f}°, "
                f"estimated turn="
                f"{small_turn_angle:.2f}°, "
                f"count={turn_count}"
            )

            yaw_before = id_direction

            for _ in range(turn_count):

                if id_direction > 0:
                    action.turn_small_angle_left()
                else:
                    action.turn_small_angle_right()

            # 给机器人一点时间稳定
            time.sleep(0.1)

            # 一批小转结束后重新识别
            image = capture_image()

            success, new_direction, proximity = identify(
                id,
                image
            )

            # 小转后目标丢失
            if not success:

                print(
                    f"Target {id} lost after "
                    f"{turn_count} small turns, "
                    f"searching directly..."
                )

                id_direction, proximity = search_target(id)

                lost_count = 0

            else:

                lost_count = 0

                # 计算这一批小转的实际平均效率
                actual_total_turn = abs(
                    yaw_before - new_direction
                )

                actual_turn_per_step = (
                    actual_total_turn / turn_count
                )

                print(
                    f"Small turn result: "
                    f"{yaw_before:.2f}° -> "
                    f"{new_direction:.2f}°"
                )

                print(
                    f"Actual total turn: "
                    f"{actual_total_turn:.2f}°"
                )

                print(
                    f"Actual turn per step: "
                    f"{actual_turn_per_step:.2f}°"
                )

                # 小于 MIN_EFFECTIVE_TURN ⇒ 小转基本没效果，改大转
                if (
                    actual_turn_per_step
                    < MIN_EFFECTIVE_TURN
                ):

                    print(
                        f"Small turn ineffective: "
                        f"{actual_turn_per_step:.2f}° "
                        f"< {MIN_EFFECTIVE_TURN}°"
                    )

                    print(
                        "Switch to large turn."
                    )

                    if new_direction > 0:
                        action.turn_big_angle_left()
                    else:
                        action.turn_big_angle_right()

                    time.sleep(0.1)

                    # 大转后立即重新识别
                    image = capture_image()

                    success, id_direction, proximity = identify(
                        id,
                        image
                    )

                    if success:
                        lost_count = 0

                    else:
                        lost_count += 1

                else:

                    # 有效小转：指数滑动平均更新小转角度估计
                    small_turn_angle = (
                        (1 - TURN_ESTIMATE_ALPHA)
                        * small_turn_angle
                        +
                        TURN_ESTIMATE_ALPHA
                        * actual_turn_per_step
                    )

                    print(
                        f"Updated small turn angle: "
                        f"{small_turn_angle:.2f}°"
                    )

                    id_direction = new_direction

        # =================================
        # 4. 大转后重新识别（小转正常时这里也会再次确认状态）
        # =================================
        time.sleep(0.1)

        image = capture_image()

        success, new_direction, proximity = identify(
            id,
            image
        )

        if success:

            id_direction = new_direction
            lost_count = 0

            continue

        lost_count += 1

        print(
            f"Target {id} lost "
            f"({lost_count}/3)"
        )

        # 连续丢失到阈值 ⇒ 重新搜索
        if lost_count >= LOST_BEFORE_SEARCH:

            print(
                f"Searching target {id} again..."
            )

            id_direction, proximity = search_target(
                id
            )

            lost_count = 0


# =====================================================================
# 目标搜索 search_target()
# =====================================================================

def search_target(id):
    """搜索目标数字：当前姿态（前视→低头）各识别一次，找不到就左转大转，最多 20 次

    找到时返回 `(机身系方向, proximity)`，方向 = 相机系 yaw + 当前云台 yaw。
    """
    for body_turn in range(SEARCH_MAX_TURNS):

        # 当前姿态搜索（先前视）
        action.servo_look_forward()
        image = capture_image()

        success, camera_direction, proximity = identify(
            id,
            image
        )

        if success:

            direction = (
                camera_direction
                + action.CURRENT_YAW
            )

            print(
                f"Found target {id}: "
                f"camera_yaw={camera_direction:.1f}°, "
                f"current_yaw={action.CURRENT_YAW:.1f}°, "
                f"direction={direction:.1f}°"
            )

            return direction, proximity

        # 再低头
        action.servo_look_down()
        image = capture_image()

        success, camera_direction, proximity = identify(
            id,
            image
        )

        if success:

            direction = (
                camera_direction
                + action.CURRENT_YAW
            )

            print(
                f"Found target {id}: "
                f"camera_yaw={camera_direction:.1f}°, "
                f"current_yaw={action.CURRENT_YAW:.1f}°, "
                f"direction={direction:.1f}°"
            )

            return direction, proximity

        # 没找到 ⇒ 机身向左大转一次
        print(
            f"Target {id} not found. "
            f"Robot turn left "
            f"({body_turn + 1}/{SEARCH_MAX_TURNS})"
        )

        action.turn_big_angle_left()

        time.sleep(0.5)

    raise RuntimeError(
        f"未找到目标数字{id}"
    )


# =====================================================================
# 到达判定 arrived()
# =====================================================================

def arrived(id, COLOR_THRESHOLD, COLOR_LOST_THRESHOLD):
    """最终接近：低头看**颜色占比**，看到过颜色后再消失 ⇒ 判定已压过目标格

    与参考版逐行一致的关键点：
      · `seen_color` 只在"明确看到颜色"后置真；中间的过渡区**不算**消失；
      · `seen_color` 为真后，**一次**真正的颜色丢失即判到达；
      · 数字识别若在 `seen_color` 之后丢失，也直接判到达；
      · 小步前进后颜色占比几乎不变 ⇒ 后退 + 越障脱困（并重拍，不沿用旧数据）；
      · 小转效率过低 ⇒ 同样后退 + 越障脱困。
    """
    color = id_to_color(id)

    action.servo_look_down()

    seen_color = False

    target_lost_count = 0

    # 小转参数（**参考版在这里是局部量，与 turn_to 的模块级同名量不同值**）
    small_turn_angle = ARRIVED_SMALL_TURN_INIT

    # id 6（紫牌）阈值本来就低，进一步放宽到 0
    if id == PURPLE_ID:
        MIN_EFFECTIVE_TURN = 0
    else:
        MIN_EFFECTIVE_TURN = ARRIVED_MIN_EFFECTIVE_TURN

    TURN_ESTIMATE_ALPHA = 0.3

    if id == PURPLE_ID:
        MIN_RATIO_CHANGE = ARRIVED_MIN_RATIO_CHANGE_ID6
    else:
        MIN_RATIO_CHANGE = ARRIVED_MIN_RATIO_CHANGE

    previous_ratio = None
    last_move_was_forward = False

    while True:

        # 1. 拍照
        image = capture_image()

        # 2. 颜色检测
        ratio = identify_color(
            color,
            image
        )

        print(
            f"arrived ratio: "
            f"{ratio:.3f}, "
            f"seen_color={seen_color}, "
            f"lost_count={target_lost_count}"
        )

        # 3. 颜色状态更新
        if ratio > COLOR_THRESHOLD:

            seen_color = True

            # 颜色明确出现 ⇒ 之前的"丢失"不能继续累计
            target_lost_count = 0

        # 4. 已经看到颜色后，颜色真正消失
        if (
            seen_color
            and ratio < COLOR_LOST_THRESHOLD
        ):

            target_lost_count += 1

            print(
                f"Target temporary lost "
                f"({target_lost_count}/1)"
            )

            # 一次真正的颜色丢失 ⇒ 认为机器人已经压过目标格
            if target_lost_count >= 1:

                action.servo_look_forward()

                print(
                    f"arrived{id}: "
                    f"color lost after seeing target"
                )

                return True

        else:

            # ratio 落在两个阈值之间 ⇒ 不认为颜色真正消失
            if ratio >= COLOR_LOST_THRESHOLD:

                target_lost_count = 0

        # 5. 数字识别
        success, yaw, proximity = identify(
            id,
            image
        )

        # 6. 数字识别失败
        if not success:

            print(
                f"Target {id} temporary lost."
            )

            # 最终判定主要靠"看到颜色 → 颜色消失"，所以这里也能收尾
            if seen_color:

                print(
                    f"arrived{id}: "
                    f"target lost after seeing color."
                )

                action.servo_look_forward()

                return True

            # 数字丢失时直接小步前进，尝试重新接近目标；
            # 此处的 ratio 变化不能用于后退决策。
            action.move_forward_small()

            previous_ratio = ratio
            last_move_was_forward = True

            continue

        # 7. 数字重新识别成功
        print(
            f"Final yaw: {yaw:.1f}°, "
            f"proximity={proximity}"
        )

        target_lost_count = 0

        # 8. 已经基本对准
        if abs(yaw) <= EPSILON:

            # 前进之前检查 ratio 变化（只在"上一步是前进"时才有意义）
            if previous_ratio is not None and last_move_was_forward:

                ratio_change = abs(
                    ratio - previous_ratio
                )

                print(
                    f"Forward efficiency: "
                    f"ratio {previous_ratio:.4f} -> "
                    f"{ratio:.4f}, "
                    f"change={ratio_change:.4f}"
                )

                if ratio_change < MIN_RATIO_CHANGE:

                    print(
                        "⚠️ Small forward movement ineffective "
                        "(ratio change < MIN_RATIO_CHANGE), "
                        "moving backward one step."
                    )

                    action.move_backward_one()
                    action.over_hurdle()

                    # 后退之后重新计算方向与颜色
                    image = capture_image()

                    ratio = identify_color(
                        color,
                        image
                    )

                    success, yaw, proximity = identify(
                        id,
                        image
                    )

                    print(
                        f"After backward recovery: "
                        f"yaw={yaw if success else None}, "
                        f"ratio={ratio:.4f}, "
                        f"proximity={proximity if success else None}"
                    )

                    previous_ratio = ratio
                    last_move_was_forward = False

                    continue

            print(
                f"Target {id} already centered."
            )

            action.move_forward_small()

            previous_ratio = ratio
            last_move_was_forward = True

            continue

        # 9. 按当前估计的小转角，计算这一批要转多少次
        turn_count = max(
            1,
            int(abs(yaw) / small_turn_angle)
        )

        # 10. 已经看到颜色 ⇒ 非常接近目标，最多 3 次小转
        if seen_color:

            turn_count = min(
                turn_count,
                MAX_FINAL_TURNS
            )

        print(
            f"Final correction: "
            f"yaw={yaw:.2f}°, "
            f"estimated_turn="
            f"{small_turn_angle:.2f}°, "
            f"count={turn_count}"
        )

        yaw_before = yaw

        for _ in range(turn_count):

            if yaw > 0:

                action.turn_small_angle_left()

            else:

                action.turn_small_angle_right()

        # 13. 小转后重新拍照
        time.sleep(0.1)

        image = capture_image()

        ratio = identify_color(
            color,
            image
        )

        success, new_yaw, new_proximity = identify(
            id,
            image
        )

        # 14. 小转后目标丢失：这里不能判"前进效率低"（刚做的是转动）
        if not success:

            print(
                f"Target {id} lost after "
                f"{turn_count} small turns."
            )

            previous_ratio = ratio
            last_move_was_forward = True

            action.move_forward_small()

            continue

        # 15. 计算实际小转效率
        actual_total_turn = abs(
            yaw_before - new_yaw
        )

        actual_turn_per_step = (
            actual_total_turn / turn_count
        )

        print(
            f"Small turn result: "
            f"{yaw_before:.2f}° -> "
            f"{new_yaw:.2f}°"
        )

        print(
            f"Actual turn per step: "
            f"{actual_turn_per_step:.2f}°"
        )

        # 16. 判断小转是否有效
        if actual_turn_per_step >= MIN_EFFECTIVE_TURN:

            small_turn_angle = (
                (1 - TURN_ESTIMATE_ALPHA)
                * small_turn_angle
                +
                TURN_ESTIMATE_ALPHA
                * actual_turn_per_step
            )

            print(
                f"Updated small turn angle: "
                f"{small_turn_angle:.2f}°"
            )

            last_move_was_forward = False

        else:

            print(
                f"Small turn ineffective: "
                f"{actual_turn_per_step:.2f}°"
            )

            # 小转效率过低 ⇒ 改变相对位置后重来
            action.move_backward_one()
            action.over_hurdle()

            previous_ratio = ratio
            last_move_was_forward = False

            continue

        # 17/18. 更新 ratio 参考值；下一轮重新拍照、重新判 yaw/ratio/seen_color
        previous_ratio = ratio

        continue


# =====================================================================
# 关卡入口
# =====================================================================

def run_level(state):
    """参考原版通关流程：按 1→7 顺序依次 go_to 每块面板

    返回 True = 7 块都走完；False = 中途 `search_target` 20 次大转仍找不到目标
    （参考版此处直接抛异常中断，这里改成干净的失败返回，交人工判断）。
    """
    print("[参考原版] 这是 reference code 里那一版通关代码的原样接入，"
          "常量尚未按本机器人/本场地重标 —— 上现场前先看 docs 的常量表。")
    if ENTER_FIELD_ON_START:
        enter_field()

    # 拍照与动作接到传进来的 state（复用真机链路；不接也能跑，退回参考版的实现）
    vision.bind_state(state)
    action.bind_state(state)

    action.init_servo()

    for digit in range(1, 8):
        arrive, color_th, lost_th = GO_TO_PARAMS[digit]
        print(f"===== [参考原版] 前往面板 {digit} "
              f"（框宽门 {arrive}px，色占比门 {color_th} → 丢失门 {lost_th}）=====")
        try:
            go_to(digit)
        except RuntimeError as e:
            print(f"[参考原版] 面板 {digit} 搜索失败：{e} —— "
                  f"顺序计分下不跳格，本关到此为止，请人工判断。")
            return False

    print("[参考原版] 7 块面板流程走完（是否真的踩中微动开关以裁判系统为准）")
    return True
