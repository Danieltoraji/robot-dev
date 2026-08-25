# -*- coding: utf-8 -*-
"""
goodluck 关卡（levels/goodluck.py）

赛道说明：机器人从入口出发，沿规定线路行进至出口，行进途中完成四次直角转弯。
赛道图示四个点位粘贴不同ID的AprilTag标签；机器人识别标签、测算距离，
并在合适时机执行转弯动作。

本模块封装了该关卡的赛道数据、导航算法和入口函数 run_level(state)。
通用能力（定位、动作执行、几何工具）从 robot_core 导入。
"""

import time
from collections import namedtuple
import numpy as np

from robot_core import RobotState, distance_point_to_rect, HEAD_CENTER


# =====================================================================
# 赛道数据
# =====================================================================
# 这里是定义每个AprilTag的世界坐标，单位为厘米。
# 添加AprilTag的世界坐标，四个点的顺序为左上，右上，右下，左下。也即从左上角开始顺时针。

tag_poses = {}
tag_poses["36"] = np.array(
    [[44.7, 23.8, 41.5], [44.7, 18.8, 41.5], [44.7, 18.8, 36.5], [44.7, 23.8, 36.5]],
    dtype=np.float64,
)
tag_poses["37"] = np.array(
    [[20.6, 100, 41.7], [25.6, 100, 41.7], [25.6, 100, 36.7], [20.6, 100, 36.7]],
    dtype=np.float64,
)
tag_poses["38"] = np.array(
    [[95, 82.2, 41.7], [95, 77.2, 41.7], [95, 77.2, 36.7], [95, 82.2, 36.7]],
    dtype=np.float64,
)
tag_poses["39"] = np.array(
    [[76.5, 0, 42.1], [71.5, 0, 42.1], [71.5, 0, 37.1], [76.5, 0, 37.1]],
    dtype=np.float64,
)

# 赛道墙壁（不可通行区域），格式 [x_min, x_max, y_min, y_max]，单位cm
# 来自赛道说明：左墙、中墙、右墙；外框底/顶边在 distance_to_walls 中单独处理
WALLS = [
    [0, 5, 40, 100],    # 左墙
    [45, 55, 0, 60],    # 中墙
    [95, 100, 40, 100], # 右墙
]

# =====================================================================
# 决策算法常量
# =====================================================================
ORIENTATION_THRESHOLD = 0.26  # 朝向差异模长阈值，约15°
POSITION_THRESHOLD = 3.0  # 位置差异模长阈值，单位cm
STOP_TIME = 3  # 到达目标点后停留时间，单位秒（比赛规则要求停靠 3 秒）
OBSTACLE_THRESHOLD = 8.0  # 避障容忍阈值，离墙最近距离小于此值则排除该动作
SAFE_MARGIN_CM = 3.0  # 安全点额外余量。实际安全点距离 = OBSTACLE_THRESHOLD + POSITION_THRESHOLD + SAFE_MARGIN_CM，确保机器人离墙足够远。
CORRIDOR_CLEAR_CM = OBSTACLE_THRESHOLD + 3.0  # 走廊净空校验阈值（路点串沿线离墙最小距离）
ORIENT_FREEZE_DIST_CM = 10.0  # 动态朝向冻结距离阈值（cm）：距目标 > 此值时用连线方向，≤ 此值时切指定朝向或冻结连线方向（防震荡）。调大防震荡，调小扩大连线方向范围，但需 > POSITION_THRESHOLD

# =====================================================================
# 批量直行参数（"一次定位，多步行动"优化）
# =====================================================================
# 仅在"朝向已对准 + 安全走廊 + 纯直行 + 远离目标"时启用批量直行，
# 减少长直走廊段的重复定位次数。转向/横移/危险区/近目标点仍每步定位。
# 2026-08-25 标定后批量直行一律使用 go_forward（5cm/步，实测比 one_step 更直）。
BATCH_FORWARD_MAX_STEPS = 4        # 单次批量直行上限步数（4步×5cm=20cm）
BATCH_FORWARD_MIN_WALL_DIST = 18.0 # 批量直行要求的最小离墙距离（cm），需 > OBSTACLE_THRESHOLD
BATCH_FORWARD_MIN_DIST = 12.0      # 距目标 > 此值才启用批量（cm），确保远离精确停靠区
# 批量直行允许的最大朝向偏差（度）：
# 4 步行程 20cm，横向偏移 = 20·sinθ ≤ 1.1cm，需 < POSITION_THRESHOLD(3cm)
GO_FORWARD_BATCH_MAX_ANGLE_DEG = 3.0

# =====================================================================
# 统一路点模型与整条赛道路点表
# =====================================================================
# Waypoint(pos, stop, orientation)：
#   pos         目标位置 [x, y]（cm）
#   stop        到达后停留秒数；>0 = 停靠评分点，0 = 经过（转向点/中间走廊路点）
#   orientation 目标朝向 [dx, dy]（单位向量）；None = 不强制朝向（走廊内顺向路点，
#               沿当前朝向继续走，不额外转向）
#
# 路点表说明：
#  - 停靠点/转向点是比赛要求经过的点（编号1~5）。
#  - 中间走廊路点（orientation=None、stop=0）用于长程规划：把机器人提前引入
#    开阔走廊，避开贴墙角的危险走廊（见各段注释），坐标建议用 assert_corridor_clear 校验。
Waypoint = namedtuple("Waypoint", ["pos", "stop", "orientation", "bypass_position", "bypass_condition"])

ROUTE = [
    Waypoint([14.7, 21.3], STOP_TIME, [1, 0], None, None),   # 停靠点1
    Waypoint([23.1, 30.0], 0.0, None, 30.5, "y+"),            # 中间路点①：先横移到x≈23，避开左墙底角
    Waypoint([23.1, 70.0], STOP_TIME, [0, 1], None, None),   # 停靠点2
    Waypoint([40.0, 80.0], 0.0, None, 41.0, "x+"),            # 中间路点②：先上行到y≈80，绕过中墙上方
    Waypoint([65.0, 79.7], STOP_TIME, [1, 0], None, None),   # 停靠点3
    Waypoint([74.0, 75.0], 0.0, None, 74.0, "y-"),            # 中间路点③：先东移到x≈74，远离中墙后下行
    Waypoint([74.0, 30.0], STOP_TIME, [0, -1], None, None),  # 停靠点4
    Waypoint([82.0, 20.0], 0.0, None, 82.0, "x+"),
    Waypoint([100.0, 20.0], STOP_TIME, [1, 0], None, None),  # 停靠点5
]

# =====================================================================
# 动作组参数常量（2026-08-25 实机标定完成）
# =====================================================================
# 实机标定结论：以下 7 个动作的位移/转角为实测值，决策算法只采用这些动作。
# 其余动作（go_forward_one_small_step、turn_left_small_step、turn_right_small_step）
# 实机表现不可靠，已从决策算法移除，相关常量一并删除。
FORWARD_CM = 5.0  # go_forward（实测 5.0cm/次）
FORWARD_ONE_STEP_CM = 2.0  # go_forward_one_step（实测 2.0cm/次）
BACK_FAST_CM = 3.2  # back_one_step（实测 3.2cm/次）
LEFT_MOVE_CM = 1.9  # left_move（实测 1.9cm/次）
RIGHT_MOVE_CM = 2.2  # right_move（实测 2.2cm/次）
TURN_LEFT_DEG = 22.0  # turn_left（实测 22.0°/次）
TURN_RIGHT_DEG = 25.7  # turn_right（实测 25.7°/次）
FORWARD_BIAS = 0.0  # 前进方向偏好权重，避免原地转圈


# =====================================================================
# 关卡几何函数（依赖赛道数据）
# =====================================================================

def distance_to_walls(pos):
    """计算位置到最近墙壁的距离（cm）

    包含3个矩形墙 + 外框底边(y=0) + 顶边(y=100)。
    左右外框已被左/右墙覆盖；出入口 x=0/x=100, y∈[0,40] 为开口不约束。
    """
    pos = np.array(pos, dtype=np.float64)
    min_dist = float("inf")
    for rect in WALLS:
        min_dist = min(min_dist, distance_point_to_rect(pos, rect))
    # 外框底边、顶边
    min_dist = min(min_dist, float(pos[1]))       # y=0
    min_dist = min(min_dist, float(100 - pos[1])) # y=100
    return min_dist

def nearest_safe_point(pos):
    """找离 pos 最近的安全点（distance_to_walls >= SAFE_THRESHOLD）

    搜索阈值 SAFE_THRESHOLD = OBSTACLE_THRESHOLD + POSITION_THRESHOLD + SAFE_MARGIN_CM，
    确保返回的安全点离机器人足够远（> POSITION_THRESHOLD），必触发平移动作，
    避免危险区边界处"需逃离但无需导航"的死循环。
    若 pos 本身安全直接返回；否则以 1cm 步长、16方向螺旋搜索，
    返回首个安全点；兜底返回搜索范围内 distance_to_walls 最大的点。
    """
    SAFE_THRESHOLD = OBSTACLE_THRESHOLD + POSITION_THRESHOLD + SAFE_MARGIN_CM
    pos = np.array(pos, dtype=np.float64)
    if distance_to_walls(pos) >= SAFE_THRESHOLD:
        return pos

    # 16方向螺旋搜索，步长1cm，最大搜索半径100cm
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    best_point = pos.copy()
    best_dist = distance_to_walls(pos)
    for step in range(1, 101):
        for a in angles:
            candidate = pos + step * np.array([np.cos(a), np.sin(a)])
            d = distance_to_walls(candidate)
            if d >= SAFE_THRESHOLD:
                return candidate
            if d > best_dist:
                best_dist = d
                best_point = candidate.copy()
    print(f"nearest_safe_point: 未找到满足阈值的安全点，返回最优点 {best_point} (距离 {best_dist:.2f}cm)")
    return best_point


# =====================================================================
# 导航算法
# =====================================================================

def calc_distance(position_diff):
    """计算水平距离"""
    return np.linalg.norm(position_diff)

def decide_panning_action(state, current_pos, target_pos, orientation_xOy):
    """贪心平移策略：模拟各方向动作后的预期位置，选最接近目标的

    state: RobotState 实例（用于访问常量，当前实现直接用模块常量）。
    current_pos/target_pos: 绝对坐标（2D）。
    对每个候选动作执行后的绝对位置做避障过滤（离墙距离 >= OBSTACLE_THRESHOLD），
    全部被排除时返回 None（信号：机器人已在危险区）。
    前进方向加偏好权重 FORWARD_BIAS。
    """
    position_diff = np.array(current_pos) - np.array(target_pos)
    # 各动作的位移向量（机体坐标系）
    # orientation_xOy 为机体朝向，左转为 [-oy[1], oy[0]]，右转为 [oy[1], -oy[0]]
    oy = orientation_xOy
    left_dir = np.array([-oy[1], oy[0]])
    right_dir = np.array([oy[1], -oy[0]])

    # 模拟各动作后的预期 position_diff（2D）
    # 候选集 = 2026-08-25 实机标定的可靠动作（go_forward_one_small_step 已移除）
    candidates = {
        "go_forward": position_diff + FORWARD_CM * oy,
        "go_forward_one_step": position_diff + FORWARD_ONE_STEP_CM * oy,
        "back_one_step": position_diff - BACK_FAST_CM * oy,
        "left_move": position_diff + LEFT_MOVE_CM * left_dir,
        "right_move": position_diff + RIGHT_MOVE_CM * right_dir,
    }

    # 计算每个候选动作执行后到目标的距离，前进方向减去偏好权重
    # 避障过滤：执行后绝对位置离墙距离 < 阈值则排除
    best_action = None
    best_score = float("inf")
    for action, new_pd in candidates.items():
        new_pos = np.array(target_pos) + new_pd
        wall_dist = distance_to_walls(new_pos)
        if wall_dist < OBSTACLE_THRESHOLD:
            print(
                f"  {action}: 执行后位置 {new_pos} 离墙 {wall_dist:.2f}cm < {OBSTACLE_THRESHOLD}cm，排除"
            )
            continue
        score = calc_distance(new_pd)
        if action.startswith("go_forward"):
            score -= FORWARD_BIAS
        print(
            f"  {action}: 执行后距离 {calc_distance(new_pd):.2f}cm 离墙 {wall_dist:.2f}cm (score={score:.2f})"
        )
        if score < best_score:
            best_score = score
            best_action = action

    if best_action is None:
        print("所有平移动作均被避障排除，机器人可能已在危险区。")
        return None

    print("最佳平移动作：", best_action)
    return best_action

def decide_rotation_action(state, target):
    """转向策略：计算带符号角度差，用实测可靠的大步转向执行

    state: RobotState 实例（用于访问 current_orientation）。
    target: 目标朝向（单位向量）。
    返回 (action_name, times)。

    2026-08-25 实机标定后：小步转向（turn_*_small_step）不可靠已弃用，
    转向只使用 turn_left(22.0°) / turn_right(25.7°)。每轮循环重新定位、
    重新决策，转角误差由闭环收敛（times 恒为 1）。
    """
    target = np.array(target, dtype=np.float64)  # 目标朝向
    # 当前朝向到目标朝向的带符号角度差
    # current_orientation × target 的 z 分量符号决定左/右转
    cross = state.current_orientation[0] * target[1] - state.current_orientation[1] * target[0]
    dot = np.dot(state.current_orientation, target)
    dot = np.clip(dot, -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(dot))

    if cross > 0:
        print(f"需左转 {angle_deg:.1f}°（turn_left，实测 22.0°/次）")
        return "turn_left", 1
    else:
        print(f"需右转 {angle_deg:.1f}°（turn_right，实测 25.7°/次）")
        return "turn_right", 1

def navigate_to_target(state, target_pos, target_orientation=None, stop_time=0.0, bypass_position=None, bypass_condition=None):
    """统一路点导航原语：定位 → 危险检测 → 动态朝向修正 → 平移接近 → 到达检查 → 按需停留

    state: RobotState 实例。
    target_pos: 目标位置 [x, y]（cm）。
    target_orientation: 目标朝向 [dx, dy]（单位向量）；None 表示不强制朝向。
    stop_time: 到达后停留秒数；0 表示不停留（转向点/中间路点）。

    动态朝向策略：
      - 距目标 > ORIENT_FREEZE_DIST_CM：目标朝向 = 当前位置→目标连线方向（动态更新），
        使转弯段更平滑（斜切接近）。
      - 距目标 ≤ ORIENT_FREEZE_DIST_CM：
        · 有指定朝向 → 切换到指定朝向（确保停靠点评分朝向精确）。
        · 无指定朝向(None) → 冻结进入近距离时的连线方向，之后固定不变（防震荡）。
      - 逃离危险区时跳过朝向修正（effective_orient=None），优先平移离开。
    """
    target_pos = np.asarray(target_pos, dtype=np.float64)
    if target_orientation is not None:
        target_orientation = np.asarray(target_orientation, dtype=np.float64)
        target_orientation = target_orientation / np.linalg.norm(target_orientation)

    if stop_time > 0:
        phase = "停靠点"
    elif target_orientation is not None:
        phase = "转向点"
    else:
        phase = "中间路点"
    print(f"\n===== 开始导航至{phase}：{target_pos} =====")
    if target_orientation is not None:
        print(f"指定朝向：{target_orientation}")

    frozen_orient = None  # 冻结的连线方向（走廊点近距离时使用，防震荡）

    while True:
        print("\n=== 新一轮导航循环开始 ===")
        print(f"目标位置：{target_pos}，目标朝向：{target_orientation}，目标停留时间：{stop_time}s")

        # 1. 定位（含头部扫描+身体转动重试）
        if not state.locate_with_retry():
            print(f"无法定位，导航至{phase} {target_pos} 失败。")
            return False

        print("当前机体位置：", state.current_position)
        print("当前机体朝向：", state.current_orientation)

        # 1.5 跳过该点的条件
        if bypass_condition is not None and bypass_position is not None:
            if (bypass_condition == "x+" and state.current_position[0] > bypass_position) or \
               (bypass_condition == "x-" and state.current_position[0] < bypass_position) or \
               (bypass_condition == "y+" and state.current_position[1] > bypass_position) or \
               (bypass_condition == "y-" and state.current_position[1] < bypass_position):
                print(f"满足跳过条件 {bypass_condition} > {bypass_position}, 跳过该路点导航。")
                print("\n=== 本轮导航循环结束 ===")
                return True

        # 2. 危险检测：离墙距离 < 阈值 → 临时导航至最近安全点
        wall_dist = distance_to_walls(state.current_position)
        if wall_dist < OBSTACLE_THRESHOLD:
            target = nearest_safe_point(state.current_position)
            escaping = True
            print(
                f"机器人处于危险区（离墙 {wall_dist:.2f}cm < {OBSTACLE_THRESHOLD}cm），"
                f"临时导航至安全点 {target}"
            )
        else:
            target = target_pos
            escaping = False

        # 3. 计算位置差异。位置差异用于后续的计算，平移决策仍读取原始数据。
        pd = np.array(target) - np.array(state.current_position)
        dist = np.linalg.norm(pd)
        print("位置差异pd：", pd)

        # 4. 动态计算有效目标朝向。除非在逃，否则都是有效目标朝向。
        if escaping:
            effective_orient = None  # 逃离时跳过朝向修正
            orient_source = "无（逃离）"
        elif dist > ORIENT_FREEZE_DIST_CM:
            # 远距离：连线方向（指向目标的单位向量），动态更新
            effective_orient = pd / dist
            frozen_orient = effective_orient.copy()  # 重置冻结（回到远距离）
            orient_source = "连线方向"
        else:
            # 近距离：有指定朝向则用指定，无则冻结连线方向
            if target_orientation is not None:
                effective_orient = target_orientation
                orient_source = "指定朝向"
            else:
                if frozen_orient is None:
                    frozen_orient = pd / dist if dist > 0 else state.current_orientation.copy()
                    if (target_orientation is None and np.dot(pd, state.current_orientation) < 0):
                        frozen_orient = -frozen_orient
                    print(f"  冻结方向：{frozen_orient}")

                effective_orient = frozen_orient
                orient_source = "冻结方向"

        # 5. 计算朝向差异。朝向差异用于后续的计算，以及决定是否旋转。旋转决策仍读取原始数据。
        od = None
        if effective_orient is not None:
            od = effective_orient - np.array(state.current_orientation)
            print(f"目标朝向（{orient_source}）：{effective_orient}  朝向差异od：{od}")

        # 6. 朝向优先：非逃离且朝向差异超阈值时修正
        if (not escaping and od is not None
                and np.linalg.norm(od) > ORIENTATION_THRESHOLD):
            action, times = decide_rotation_action(state, effective_orient)
            state.run_action(action, times)
            continue

        # 7. 平移接近：pd 模 > 阈值 → 贪心选最优方向
        if dist > POSITION_THRESHOLD:
            action = decide_panning_action(state, state.current_position, target, state.current_orientation)
            if action is None:
                # 极端兜底：所有动作被排除
                print("警告：无安全平移动作可选，按兵不动。")
                # state.run_action("back_one_step")
            else:
                # ★ 批量直行优化：朝向已对准 + 安全走廊 + 纯直行 + 远离目标时批量执行
                # 2026-08-25 标定后：批量直行一律用 go_forward（5cm/步，实测比 one_step 更直），
                # 仅在朝向偏差 ≤ GO_FORWARD_BATCH_MAX_ANGLE_DEG 时启用；偏差更大回退单步。
                if (action in ("go_forward", "go_forward_one_step")
                        and not escaping
                        and wall_dist >= BATCH_FORWARD_MIN_WALL_DIST
                        and dist > BATCH_FORWARD_MIN_DIST
                        and od is not None
                        and np.linalg.norm(od) <= ORIENTATION_THRESHOLD):
                    # od 模长 = 2*sin(θ/2)，θ = 2*arcsin(‖od‖/2)
                    angle_deg = np.degrees(2 * np.arcsin(min(np.linalg.norm(od) / 2, 1.0)))
                    if angle_deg <= GO_FORWARD_BATCH_MAX_ANGLE_DEG:
                        batch_steps = min(BATCH_FORWARD_MAX_STEPS, int(dist / FORWARD_CM))
                        # 路径预检：整段批量路径离墙距离是否充足
                        if batch_steps > 1 and check_segment_clear(
                            state.current_position, state.current_orientation,
                            batch_steps, FORWARD_CM, BATCH_FORWARD_MIN_WALL_DIST
                        ):
                            print(f"  ★ 批量直行 {batch_steps} 步（go_forward，距目标 {dist:.1f}cm，离墙 {wall_dist:.1f}cm）")
                            state.run_action("go_forward", times=batch_steps)
                            continue
                state.run_action(action)
            continue
        print("\n=== 本轮导航循环结束 ===")
        # 8. 到达
        if escaping:
            print(f"===== 已到达安全点 {target}，恢复原目标导航 =====")
            continue
        if stop_time > 0:
            print(f"===== 到达{phase} {target_pos}，停留 {stop_time} 秒 =====")
            state.run_action("stand")
            time.sleep(stop_time)
        else:
            print(f"===== 到达{phase} {target_pos}，不停留 =====")
        return True

def check_segment_clear(start_pos, direction, steps, step_cm, min_dist):
    """校验批量直行整段路径的离墙距离

    start_pos: 起点位置 [x, y]（cm）
    direction: 机体朝向单位向量 [dx, dy]
    steps: 批量步数
    step_cm: 每步前进距离（cm）
    min_dist: 最小离墙距离要求（cm）

    沿路径每步采样一次，任一点离墙 < min_dist 返回 False。
    """
    start_pos = np.asarray(start_pos, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    for k in range(1, steps + 1):
        sample = start_pos + k * step_cm * direction
        if distance_to_walls(sample) < min_dist:
            print(f"  [批量预检] 第{k}步位置 {sample} 离墙 {distance_to_walls(sample):.2f}cm < {min_dist}cm，不可批量")
            return False
    return True


def assert_corridor_clear(polyline, min_dist=CORRIDOR_CLEAR_CM):
    """校验一串路点折线走廊的净空：沿线每 1cm 采样 distance_to_walls

    任一采样点离墙 < min_dist 打印警告并返回 False（用于排查路点是否贴墙）。
    """
    pts = [np.asarray(p, dtype=np.float64) for p in polyline]
    ok = True
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        seg_len = np.linalg.norm(b - a)
        n = max(int(seg_len), 2)
        for k in range(n + 1):
            sample = a + (b - a) * (k / n)
            d = distance_to_walls(sample)
            if d < min_dist:
                print(f"  [走廊警告] 段 {a}→{b} 采样 {sample} 离墙 {d:.2f}cm < {min_dist}cm")
                ok = False
    print(f"  [走廊校验] {'通过，全部净空 >= %.1fcm' % min_dist if ok else '存在贴墙段，请调整路点'}")
    return ok


# =====================================================================
# 关卡入口函数
# =====================================================================

def run_level(state):
    """goodluck 关卡主流程：沿 ROUTE 路点串依次导航

    state: RobotState 实例（需已用 tag_poses 初始化）。
    """
    # 上线前校验路点走廊净空
    assert_corridor_clear([wp.pos for wp in ROUTE])

    state.run_action("stand")
    state.set_head(HEAD_CENTER)

    for i, wp in enumerate(ROUTE, 1):
        if not navigate_to_target(state, wp.pos, wp.orientation, wp.stop, wp.bypass_position, wp.bypass_condition):
            print(f"导航至路点 {i}（{wp.pos}）失败，程序终止。")
            return False
        print(f"已{'到达停靠点' if wp.stop > 0 else '通过路点'} {i}：{wp.pos}")

    print("===== 全程完成 =====")
    return True
