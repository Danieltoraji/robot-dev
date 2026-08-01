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

# 以下定义机器人定点站立课目的目标点坐标和朝向。
stand_poses = {}
stand_poses["1"] = np.array([14.7, 21.3], dtype=np.float64)
stand_poses["2"] = np.array([23.1, 70], dtype=np.float64)
stand_poses["3"] = np.array([65, 79.7], dtype=np.float64)
stand_poses["4"] = np.array([74, 30], dtype=np.float64)


# =====================================================================
# 决策算法常量
# =====================================================================
ORIENTATION_THRESHOLD = 0.19  # 朝向差异模长阈值，约11°，是2sin(11°/2)的值
POSITION_THRESHOLD = 3.0  # 位置差异模长阈值，单位cm
STOP_TIME = 3  # 到达目标点后停留时间，单位秒
OBSTACLE_THRESHOLD = 15.0  # 避障容忍阈值，离墙最近距离小于此值则排除该动作
SAFE_MARGIN_CM = 3.0  # 安全点额外余量。实际安全点距离 = OBSTACLE_THRESHOLD + POSITION_THRESHOLD + SAFE_MARGIN_CM，确保机器人离墙足够远。

target_poses = {}
target_poses["1"] = np.array([20.0, 21.3], dtype=np.float64)
target_poses["2"] = np.array([23.1, 70], dtype=np.float64)
target_poses["3"] = np.array([65, 79.7], dtype=np.float64)
target_poses["4"] = np.array([74, 30], dtype=np.float64)

target_orientations = {}
target_orientations["1"] = np.array([1, 0], dtype=np.float64)
target_orientations["2"] = np.array([0, 1], dtype=np.float64)
target_orientations["3"] = np.array([1, 0], dtype=np.float64)
target_orientations["4"] = np.array([0, -1], dtype=np.float64)


# =====================================================================
# 动作组参数常量（所有的数值都需要重新标定！！！）
# =====================================================================
FORWARD_ONE_STEP_CM = 4.0  # go_forward_one_step（待标定）
FORWARD_ONE_SMALL_STEP_CM = 2.0  # go_forward_one_small_step（待标定）
BACK_ONE_STEP_CM = 4.0  # back_one_step（待标定）
LEFT_MOVE_CM = 2.9  # left_move（待标定）
RIGHT_MOVE_CM = 2.1  # right_move（待标定）
TURN_LEFT_SMALL_STEP_DEG = 21.0  # turn_left_small_step（待标定）
TURN_RIGHT_SMALL_STEP_DEG = 21.0  # turn_right_small_step（待标定）
TURN_LEFT_DEG = 30.0  # turn_left（估算值，待标定）
TURN_RIGHT_DEG = 30.0  # turn_right（估算值，待标定）
FORWARD_BIAS = 0.5  # 前进方向偏好权重，避免原地转圈
CAMERA_FORWARD_OFFSET_CM = 2.0  # 摄像头中心相对旋转中心的前后偏移（旋转中心在后方，cm）
TURN_LEFT_RADIUS_CM = 5.0  # 左转圆周运动半径（cm）
TURN_RIGHT_RADIUS_CM = 5.0  # 右转圆周运动半径（cm）


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

def calculate_diff(state, target_pos, target_id):
    """计算当前位置/朝向与目标点的差异

    state: RobotState 实例。
    target_pos: 目标位置坐标数组（2D）。
    target_id: 目标编号，用于查 target_orientations。
    """
    position_diff = np.array(state.current_position) - np.array(target_pos)
    orientation_diff = np.array(state.current_orientation) - np.array(
        target_orientations[target_id]
    )
    return position_diff, orientation_diff

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
    candidates = {
        "go_forward_one_step": position_diff + FORWARD_ONE_STEP_CM * oy,
        "go_forward_one_small_step": position_diff + FORWARD_ONE_SMALL_STEP_CM * oy,
        "back_one_step": position_diff - BACK_ONE_STEP_CM * oy,
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

def decide_rotation_action(state, orientation_diff):
    """转向策略：计算带符号角度差，小角度差用小步，大角度差用大步

    state: RobotState 实例（用于访问 current_orientation）。
    返回 (action_name, times)。
    """
    target = state.current_orientation - orientation_diff  # 目标朝向
    # 当前朝向到目标朝向的带符号角度差
    # current_orientation × target 的 z 分量符号决定左/右转
    cross = state.current_orientation[0] * target[1] - state.current_orientation[1] * target[0]
    dot = np.dot(state.current_orientation, target)
    dot = np.clip(dot, -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(dot))

    if cross > 0:
        direction = "left"
        print(f"需左转 {angle_deg:.1f}°")
    else:
        direction = "right"
        print(f"需右转 {angle_deg:.1f}°")

    if direction == "left":
        if angle_deg > TURN_LEFT_DEG:
            return "turn_left", 1
        else:
            return "turn_left_small_step", 1
    else:
        if angle_deg > TURN_RIGHT_DEG:
            return "turn_right", 1
        else:
            return "turn_right_small_step", 1

def navigate_to_target(state, target_id, poses, stop_time=STOP_TIME):
    """通用导航函数：定位→对准朝向→平移接近→到达检查→按需停留

    state: RobotState 实例。
    poses: 目标位置字典（stand_poses 或 target_poses）。
    stop_time: 到达后停留秒数；0 表示不停留（转向起止点）。
    """
    phase = "停靠点" if stop_time > 0 else "转向点"
    print(f"\n===== 开始导航至{phase} {target_id} =====")
    print(
        f"目标坐标：{poses[target_id]}，目标朝向：{target_orientations[target_id]}"
    )

    while True:
        # 1. 定位（含头部扫描+身体转动重试）
        if not state.locate_with_retry():
            print(f"无法定位，导航至{phase} {target_id} 失败。")
            return False

        print("当前机体位置：", state.current_position)
        print("当前机体朝向：", state.current_orientation)

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
            target = poses[target_id]
            escaping = False

        # 3. 计算差异
        pd, od = calculate_diff(state, target, target_id)
        print("位置差异pd：", pd)
        print("朝向差异od：", od)

        # 4. 朝向优先：od 模 > 阈值 → 旋转修正（逃离时跳过，优先平移离开）
        if not escaping and np.linalg.norm(od) > ORIENTATION_THRESHOLD:
            action, times = decide_rotation_action(state, od)
            state.run_action(action, times)
            continue

        # 5. 平移接近：pd 模 > 阈值 → 贪心选最优方向
        if np.linalg.norm(pd) > POSITION_THRESHOLD:
            action = decide_panning_action(state, state.current_position, target, state.current_orientation)
            if action is None:
                # 极端兜底：所有动作被排除，强制后退尝试离开危险区
                print("警告：无安全平移动作可选，强制后退尝试脱离危险区。")
                state.run_action("back_one_step")
            else:
                state.run_action(action)
            continue

        # 6. 到达
        if escaping:
            print(f"===== 已到达安全点 {target}，恢复原目标导航 =====")
            continue
        if stop_time > 0:
            print(f"===== 到达{phase} {target_id}，停留 {stop_time} 秒 =====")
            state.run_action("stand")
            time.sleep(stop_time)
        else:
            print(f"===== 到达{phase} {target_id}，不停留 =====")
        return True


# =====================================================================
# 关卡入口函数
# =====================================================================

def run_level(state):
    """goodluck 关卡主流程

    state: RobotState 实例（需已用 tag_poses 初始化）。
    流程：站立→头部回正→依次导航4个停靠点/转向点→开环走出出口。
    """
    state.run_action("stand")
    state.set_head(HEAD_CENTER)

    for tid in ["1", "2", "3", "4"]:
        # 停靠阶段：到达 stand_poses，停留 3 秒
        if not navigate_to_target(state, tid, stand_poses, STOP_TIME):
            print(f"导航至停靠点 {tid} 失败，程序终止。")
            return False
        print(f"已到达停靠点 {tid}。")

        # 准备转向阶段：到达 target_poses，不停留
        if not navigate_to_target(state, tid, target_poses, 0):
            print(f"导航至转向点 {tid} 失败，程序终止。")
            return False
        print(f"已到达转向点 {tid}。")

    # 第5点：开环走出出口
    print("\n===== 到达第4个转向点，准备开环走出出口 =====")
    state.run_action("go_forward", times=3)
    state.run_action("turn_left", times=3)
    state.run_action("go_forward", times=6)
    state.run_action("stand")
    print("===== 全程完成 =====")
    return True
