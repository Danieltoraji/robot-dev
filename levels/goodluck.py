# -*- coding: utf-8 -*-
"""
goodluck 关卡（levels/goodluck.py）

赛道说明：机器人从入口出发，沿规定线路行进至出口，行进途中完成四次直角转弯。
赛道图示四个点位粘贴不同ID的AprilTag标签；机器人识别标签、测算距离，
并在合适时机执行转弯动作。

本模块封装了该关卡的赛道数据、导航算法和入口函数 run_level(state)。
通用能力（定位、动作执行、几何工具）从 robot_core 导入。
"""

import sys
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
tag_poses["151"] = np.array([[45.01, 16.12, 31.40], [45.01, 11.12, 31.49], [45.01, 11.04, 26.49], [45.01, 16.04, 26.40]],dtype=np.float64,)
tag_poses["152"] = np.array([[45.01, 42.35, 30.84], [45.01, 37.35, 31.01], [45.01, 37.17, 26.02], [45.01, 42.17, 25.84]],dtype=np.float64,)
tag_poses["153"] = np.array([[18.19, 101.38, 19.77], [23.17, 101.38, 20.22], [23.62, 101.38, 15.24], [18.64, 101.38, 14.79]],dtype=np.float64,)
tag_poses["154"] = np.array([[18.06, 101.38, 32.04], [23.06, 101.38, 32.05], [23.07, 101.38, 27.05], [18.07, 101.38, 27.04]],dtype=np.float64,)
tag_poses["155"] = np.array([[73.14, 101.38, 28.13], [78.14, 101.38, 28.14], [78.16, 101.38, 23.14], [73.16, 101.38, 23.13]],dtype=np.float64,)
tag_poses["156"] = np.array([[95.00, 82.57, 27.41], [95.00, 77.57, 27.54], [95.00, 77.43, 22.54], [95.00, 82.43, 22.41]],dtype=np.float64,)
tag_poses["157"] = np.array([[95.00, 60.44, 28.09], [95.00, 55.47, 28.62], [95.00, 54.94, 23.65], [95.00, 59.91, 23.12]],dtype=np.float64,)
tag_poses["158"] = np.array([[55.23, 16.49, 29.62], [55.23, 21.49, 29.67], [55.23, 21.54, 24.67], [55.23, 16.54, 24.62]],dtype=np.float64,)
tag_poses["159"] = np.array([[27.98, 22.66, 0.00], [27.84, 17.67, 0.00], [22.84, 17.81, 0.00], [22.99, 22.81, 0.00]],dtype=np.float64,)
tag_poses["160"] = np.array([[79.57, 15.48, 0.00], [74.57, 15.50, 0.00], [74.58, 20.50, 0.00], [79.58, 20.48, 0.00]],dtype=np.float64,)
tag_poses["161"] = np.array([[82.36, 0.00, 27.93], [77.37, 0.00, 27.65], [77.64, 0.00, 22.66], [82.63, 0.00, 22.93]],dtype=np.float64,)
tag_poses["162"] = np.array([[22.51, 78.43, 0.00], [27.51, 78.50, 0.00], [27.58, 73.50, 0.00], [22.58, 73.43, 0.00]],dtype=np.float64,)
tag_poses["163"] = np.array([[77.93, 78.45, 0.00], [77.65, 73.46, 0.00], [72.66, 73.75, 0.00], [72.94, 78.74, 0.00]],dtype=np.float64,)

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
STOP_TIME = 0  # 到达目标点后停留秒数。2026-08-30 提速改造取消停靠（原比赛规则 3 秒），
#               恢复停靠改回 3 即可（ROUTE 停靠点引用自动生效）
OBSTACLE_THRESHOLD = 13.0  # 避障容忍阈值，离墙最近距离小于此值则排除该动作
SAFE_MARGIN_CM = 3.0  # 安全点额外余量。实际安全点距离 = OBSTACLE_THRESHOLD + POSITION_THRESHOLD + SAFE_MARGIN_CM，确保机器人离墙足够远。
CORRIDOR_CLEAR_CM = OBSTACLE_THRESHOLD + 3.0  # 走廊净空校验阈值（路点串沿线离墙最小距离）
ORIENT_FREEZE_DIST_CM = 10.0  # 动态朝向冻结距离阈值（cm）：距目标 > 此值时用连线方向，≤ 此值时切指定朝向或冻结连线方向（防震荡）。调大防震荡，调小扩大连线方向范围，但需 > POSITION_THRESHOLD

# =====================================================================
# 批量直行参数（"一次定位，多步行动"优化）
# =====================================================================
# 仅在"朝向已对准 + 纯直行 + 远离目标"时启用批量直行。
# 2026-08-25 标定后批量直行一律使用 go_forward（5cm/步，实测比 one_step 更直）。
# 2026-08-30 v6：BATCH_FORWARD_MIN_WALL_DIST / BATCH_FORWARD_MIN_DIST 已删除——
# 上限由调度层按航向自适应计算（min(开阔 6 步, 中心带余量/sin(航向差))，
# 走廊硬顶 5 步），近目标截短由调度层"批量以切换线为界"接管；
# 路径预检阈值改为 OBSTACLE_THRESHOLD+2（15cm）。
GO_FORWARD_BATCH_MAX_STEPS = 6        # 批量直行绝对上限（旧调用默认；调度层自适应可更小）
GO_FORWARD_BATCH_MAX_ANGLE_DEG = 3.0  # 批量直行允许的最大朝向偏差（度）

# =====================================================================
# Tolerant 直行参数（在长直段容忍小航向误差，到拐点再统一大转）
# =====================================================================
TOLERANT_ANGLE_DEG = 20.0             # 远段不转向的最大航向误差（度）
APPROACH_DIST_CM = 15.0               # 距拐点多远切换回严格对准
MIN_SEG_LEN_FOR_TOLERANT = 15.0       # 直段长度低于此值不使用 tolerant
MIN_CORNER_ANGLE_FOR_TOLERANT = 30.0  # 下一拐点转角低于此值不使用 tolerant

# 拐点接近冻结：进入该距离后冻结参考方向，避免近距离转向位移造成 RRLL
APPROACH_FREEZE_DIST_CM = 20.0

# 执行期管廊约束：横向偏离当前段中线超过此值触发重规划
CORRIDOR_DEVIATION_LIMIT_CM = 6.0

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
    # 2026-08-30 提速改造：停靠点 orientation 一并取消（到达即走，不再对准转身），
    # 恢复停靠朝向时把 None 改回原值：1→[1,0]、2→[0,1]、3→[1,0]、4→[0,-1]、5→[1,0]
    Waypoint([14.7, 21.3], STOP_TIME, None, None, None),   # 停靠点1
    Waypoint([23.1, 30.0], 0.0, None, 30.5, "y+"),            # 中间路点①：先横移到x≈23，避开左墙底角
    Waypoint([23.1, 70.0], STOP_TIME, None, None, None),   # 停靠点2
    Waypoint([40.0, 80.0], 0.0, None, 41.0, "x+"),            # 中间路点②：先上行到y≈80，绕过中墙上方
    Waypoint([65.0, 79.7], STOP_TIME, None, None, None),   # 停靠点3
    # 2026-08-30 撞墙修复：bypass 原为 (74.0,"y-") 只查 y 不查 x——"先东移远离
    # 中墙再下行"被跳过（实测在 x=64.5 触发，贴中墙东面 x=55 下行撞墙）。
    # 改为 x≥72 才允许跳过：保证东移到位后才直落。
    Waypoint([74.0, 75.0], 0.0, None, 72.0, "x+"),          # 中间路点③：先东移到x≥72，远离中墙后下行
    Waypoint([74.0, 30.0], STOP_TIME, None, None, None),  # 停靠点4（--end-at-last-stop 在此结束）
    Waypoint([82.0, 20.0], 0.0, None, 82.0, "x+"),
    Waypoint([100.0, 20.0], STOP_TIME, None, None, None),  # 停靠点5（出口）
]

# =====================================================================
# 提前结束开关（2026-08-30 提速改造）
# =====================================================================
# --end-at-last-stop：走到停靠点4 [74,30] 即结束流程，跳过中间路点④与出口段
# （出口段日后由其它方法处理）。真机用法：python main.py goodluck --end-at-last-stop
END_AT_LAST_STOP = "--end-at-last-stop" in sys.argv
END_AFTER_POS = [74.0, 30.0]  # 提前结束的判定路点（停靠点4）
# v6 A* 导航目标与到达阈值
GOAL_POS = [97.5, 20.0]      # 出口（栅格中心 97.5；出口开口 x=100 y∈[0,40]）
GOAL_THRESHOLD_CM = 6.0      # 终点到达判定（cm）

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
# Batch 审计（默认关闭，不影响正常行为）
# =====================================================================

class AuditRecorder:
    """记录批量直行/批量转向的提议、实际执行与拒绝原因。"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.segments = []
        self.batch_events = []
        self.turn_events = []
        self._current_segment = None
        self._current_segment_idx = None
        self._current_segment_start_locate = None

    def begin_segment(self, wp, locate_count):
        self._current_segment_idx = len(self.segments)
        self._current_segment = {
            "semantics": wp.semantics,
            "xy": [float(wp.xy[0]), float(wp.xy[1])],
            "actions": 0,
            "locates": 0,
            "start_locate": locate_count,
        }
        self.segments.append(self._current_segment)

    def record_action(self):
        if self._current_segment is not None:
            self._current_segment["actions"] += 1

    def end_segment(self, locate_count):
        if self._current_segment is not None:
            self._current_segment["locates"] = locate_count - self._current_segment["start_locate"]
            self._current_segment = None
            self._current_segment_idx = None

    def record_batch(self, proposed, candidate, actual, reject_reason,
                     angle_deg=None, angle_gate=None, precheck_ok=None,
                     precheck_min_clearance=None):
        self.batch_events.append({
            "segment_idx": self._current_segment_idx,
            "proposed": proposed,
            "candidate": candidate,
            "actual": actual,
            "reject_reason": reject_reason,
            "angle_deg": angle_deg,
            "angle_gate": angle_gate,
            "precheck_ok": precheck_ok,
            "precheck_min_clearance": precheck_min_clearance,
        })

    def record_turn(self, action, proposed_times, actual_times, angle_deg=None):
        self.turn_events.append({
            "segment_idx": self._current_segment_idx,
            "action": action,
            "proposed_times": proposed_times,
            "actual_times": actual_times,
            "angle_deg": angle_deg,
        })

    def to_dict(self):
        return {
            "segments": self.segments,
            "batch_events": self.batch_events,
            "turn_events": self.turn_events,
        }


AUDIT = None  # 默认关闭；由 enable_audit() 开启


def enable_audit():
    """开启 batch 审计并清空旧数据。"""
    global AUDIT
    if AUDIT is None:
        AUDIT = AuditRecorder()
    AUDIT.reset()


def get_audit_data():
    """返回当前审计数据；未开启时返回 None。"""
    return AUDIT.to_dict() if AUDIT is not None else None


def _sim_locate_count(state):
    """从模拟器状态读取累计定位次数；非模拟环境返回 0。"""
    sim = getattr(state, "_sim", None)
    return sim.locate_count if sim is not None else 0


def _audit_action():
    """审计：记录一次已执行动作。"""
    if AUDIT is not None:
        AUDIT.record_action()


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

def decide_panning_action(state, current_pos, target_pos, orientation_xOy, escaping=False):
    """贪心平移策略：模拟各方向动作后的预期位置，选最接近目标的

    state: RobotState 实例（用于访问常量，当前实现直接用模块常量）。
    current_pos/target_pos: 绝对坐标（2D）。
    escaping: 危险区逃离模式。全部候选被避障排除时不再返回 None（按兵不动），
        而是返回"预测位置离墙距离最大"的一步（2026-08-30 撞墙复盘：逃离的
        单步过滤在深度入区时会全排除导致死锁——朝向平行墙面时一步补不回
        5cm+ 的缺口；最优一步保证危险区内始终有进展）。
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
    fallback_action = None
    fallback_dist = -1.0
    for action, new_pd in candidates.items():
        new_pos = np.array(target_pos) + new_pd
        wall_dist = distance_to_walls(new_pos)
        if wall_dist > fallback_dist:
            fallback_dist = wall_dist
            fallback_action = action
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
        if escaping:
            # 危险区兜底：无净空步可选时执行离墙增益最大的一步，保证进展
            print(f"危险区无净空步可选，执行离墙增益最大的一步：{fallback_action} "
                  f"(执行后离墙 {fallback_dist:.2f}cm)")
            return fallback_action
        print("所有平移动作均被避障排除，机器人可能已在危险区。")
        return None

    print("最佳平移动作：", best_action)
    return best_action

def decide_rotation_action(state, target):
    """转向策略：计算带符号角度差，用实测可靠的大步转向执行（批量连转）

    state: RobotState 实例（用于访问 current_orientation）。
    target: 目标朝向（单位向量）。
    返回 (action_name, times)。

    2026-08-25 实机标定后：小步转向（turn_*_small_step）不可靠已弃用，
    转向只使用 turn_left(22.0°) / turn_right(25.7°)。
    2026-08-30 提速改造：需要角 ≥ 1.5 步时一次连转 times 步（round 取整，上限 3），
    省去连转链中的中间定位（每省一次 ≈2s）。过冲 ≤ 一个步长，处于
    ORIENTATION_THRESHOLD(15°) 容忍内，转完一次定位自纠。
    """
    target = np.array(target, dtype=np.float64)  # 目标朝向
    # 当前朝向到目标朝向的带符号角度差
    # current_orientation × target 的 z 分量符号决定左/右转
    cross = state.current_orientation[0] * target[1] - state.current_orientation[1] * target[0]
    dot = np.dot(state.current_orientation, target)
    dot = np.clip(dot, -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(dot))

    if cross > 0:
        step_deg, name = TURN_LEFT_DEG, "turn_left"
    else:
        step_deg, name = TURN_RIGHT_DEG, "turn_right"
    times = max(1, min(3, int(round(angle_deg / step_deg))))
    if AUDIT is not None:
        AUDIT.record_turn(name, times, times, angle_deg=angle_deg)
    print(f"需转向 {angle_deg:.1f}°（{name}，实测 {step_deg}°/次 × {times}）")
    return name, times

def navigate_to_target(state, target_pos, target_orientation=None, stop_time=0.0,
                       bypass_position=None, bypass_condition=None,
                       semantics="precise", max_batch_steps=None,
                       batch_angle_limit_deg=None,
                       pass_line=None, prev_waypoint=None, locate_fail_limit=2,
                       tolerant=False):
    """统一路点导航原语：定位 → 危险检测 → 动态朝向修正 → 平移接近 → 到达检查

    state: RobotState 实例。
    target_pos: 目标位置 [x, y]（cm）。
    target_orientation: 目标朝向 [dx, dy]；None 表示不强制朝向。
    stop_time: 到达后停留秒数；0 不停留。
    --- v6 混合架构扩展参数（调度层传入）---
    semantics: 子目标语义。"precise"（拐点/终点：精确到达 3cm + 微调权限，
        冻结/取反逻辑生效）或 "pass"（直段切点：法向越线即到达，不做
        到点微调，冻结/取反被门控跳过——目标在侧后方时由远距连线方向
        + 横移/倒退候选自然处理）。
    max_batch_steps: 批量直行上限（调度层按航向自适应计算；None = 用
        旧默认逻辑 GO_FORWARD_BATCH 上限，兼容旧调用）。
    batch_angle_limit_deg: 批量直行的角度门上限（度）。调度层公式已用
        sin(航向差) 惩罚了角度（偏角越大步数越少），执行层再用固定 3°
        门二次否决会让自适应批量被拦（2026-08-31 主因修复）——传入时
        取 max(默认 3°, 此值)；None = 旧行为 3°（兼容旧调用）。
    pass_line: (prev_wp_xy, cur_wp_xy) 二元组——pass 语义的法向越线判定
        基准（prev→cur 连线，机器人越过 cur 的法向线即通过）。
    prev_waypoint: 上一个子目标坐标（护栏/越线计算的段起点）。
    locate_fail_limit: 连续定位失败 retry 次数上限，超过则安全终止
        （stand + 返回 False；不无限等待也不盲走）。

    动态朝向策略（precise 语义；pass 语义被门控跳过近距部分）：
      - 距目标 > ORIENT_FREEZE_DIST_CM：连线方向动态更新。
      - 距目标 ≤ ORIENT_FREEZE_DIST_CM：
        · 有指定朝向 → 指定朝向；无 → 冻结连线方向（目标在正后方时
          取反——机器人倒退接近而非 180° 调头）。
      - 逃离危险区时跳过朝向修正，优先平移离开。
    """
    target_pos = np.asarray(target_pos, dtype=np.float64)
    if target_orientation is not None:
        target_orientation = np.asarray(target_orientation, dtype=np.float64)
        target_orientation = target_orientation / np.linalg.norm(target_orientation)

    phase = "子目标" if semantics == "pass" else "目标点"
    print(f"\n===== 开始导航至{phase}：{target_pos}（语义 {semantics}）=====")

    frozen_orient = None
    locate_fails = 0

    while True:
        # 1. 定位（含头部扫描+身体转动重试）；连续失败达上限 → 安全终止
        if not state.locate_with_retry():
            locate_fails += 1
            if locate_fails >= locate_fail_limit:
                print(f"连续 {locate_fails} 次定位失败（retry 级），安全终止。")
                state.act("stand")
                return False
            print(f"定位失败 {locate_fails}/{locate_fail_limit}，重试。")
            continue

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

        # Tolerant 远段：允许更大的不转向误差；近段恢复严格阈值
        if tolerant and semantics != "pass" and dist > APPROACH_DIST_CM:
            turn_threshold_deg = TOLERANT_ANGLE_DEG
        else:
            turn_threshold_deg = np.degrees(2 * np.arcsin(min(ORIENTATION_THRESHOLD / 2, 1.0)))
        turn_threshold_norm = 2 * np.sin(np.radians(turn_threshold_deg) / 2)

        # 3.4 执行期管廊约束：偏离当前段中线过远则触发重规划
        if not escaping and prev_waypoint is not None:
            from path_planner import route_offset
            lateral, _ = route_offset(state.current_position, prev_waypoint, target_pos)
            if abs(lateral) > CORRIDOR_DEVIATION_LIMIT_CM:
                print(f"  [管廊] 横向偏差 {lateral:.1f}cm > {CORRIDOR_DEVIATION_LIMIT_CM}cm，触发重规划")
                return "replan"

        # 3.5 pass 语义：法向越线判定（在危险检测后、朝向修正前）
        #    prev→cur 段方向 d；机器人相对 cur 的向量 v；
        #    along = v·d > 0（已越过 cur 的法向线）且 |side| 有界（仍在段附近，
        #    防止在下一段远处误判）→ 视为通过，立即返回。
        if semantics == "pass" and pass_line is not None:
            (px0, py0), (px1, py1) = pass_line[0], pass_line[1]
            sx, sy = float(state.current_position[0]), float(state.current_position[1])
            seg_dx, seg_dy = px1 - px0, py1 - py0
            seg_len = np.hypot(seg_dx, seg_dy)
            if seg_len > 1e-9:
                vx, vy = sx - px1, sy - py1
                along = vx * seg_dx + vy * seg_dy
                side = vx * seg_dy - vy * seg_dx
                if along > 0 and abs(side) < 25.0:
                    print(f"===== 通过直段切点 {target_pos}（越线判定）=====")
                    return True

        # 4. 动态计算有效目标朝向。pass 语义门控：朝向目标 = 路径段方向
        #    （prev→cur 连线方向，固定值）。首版错误地用"到切点的连线方向"——
        #    切点在前方不断移动，朝向修正是追逐目标永不收敛（步进间差 15~19°
        #    恒过不了阈值 → 转一点走一步无限循环；一次转向过冲后目标反转
        #    106~170°，×3 连转冲进危险区）。路径段方向是常量，转到位即收敛。
        if escaping:
            effective_orient = None  # 逃离时跳过朝向修正
            orient_source = "无（逃离）"
        elif semantics == "pass" and pass_line is not None:
            (px0, py0), (px1, py1) = pass_line[0], pass_line[1]
            seg = np.array([px1 - px0, py1 - py0], dtype=np.float64)
            seg_n = np.linalg.norm(seg)
            effective_orient = seg / seg_n if seg_n > 1e-9 else pd / dist
            frozen_orient = None
            orient_source = "路径段方向"
        elif dist > APPROACH_FREEZE_DIST_CM:
            # 远距离：连线方向（指向目标的单位向量），动态更新
            effective_orient = pd / dist
            frozen_orient = effective_orient.copy()  # 重置冻结（回到远距离）
            orient_source = "连线方向"
        else:
            # 近距离（仅 precise 语义）：提前进入冻结窗口，避免转向位移造成 RRLL
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
                and np.linalg.norm(od) > turn_threshold_norm):
            action, times = decide_rotation_action(state, effective_orient)
            state.act(action, times)
            _audit_action()
            continue

        # 7. 平移接近：pd 模 > 阈值 → 贪心选最优方向
        if dist > POSITION_THRESHOLD:
            action = decide_panning_action(state, state.current_position, target,
                                           state.current_orientation, escaping=escaping)
            if action is None:
                # 极端兜底：所有动作被排除
                print("警告：无安全平移动作可选，按兵不动。")
                # state.run_action("back_one_step")
            else:
                # ★ 批量直行优化：朝向已对准 + 纯直行 + 远离目标时批量执行
                # 2026-08-25 标定后：批量直行一律用 go_forward（5cm/步，实测比 one_step 更直），
                # 仅在朝向偏差 ≤ GO_FORWARD_BATCH_MAX_ANGLE_DEG 时启用；偏差更大回退单步。
                # 2026-08-30 v6：批量上限改为外部 max_batch_steps（调度层按航向自适应：
                # min(开阔 6 步, 中心带余量/sin(航向差))，走廊硬顶 5 步）；
                # None 时退回 GO_FORWARD_BATCH_MAX_STEPS（兼容旧调用）。
                # 动态刷新批量上限：每次按当前位姿/航向重新计算，避免转正后仍被旧上限卡住
                if max_batch_steps is not None:
                    fresh_cap, fresh_angle_limit = adaptive_batch_steps(state, target_pos, dist)
                    batch_cap = fresh_cap
                    effective_angle_limit = fresh_angle_limit
                else:
                    batch_cap = GO_FORWARD_BATCH_MAX_STEPS
                    effective_angle_limit = batch_angle_limit_deg
                if action in ("go_forward", "go_forward_one_step") and not escaping and od is not None:
                    # od 模长 = 2*sin(θ/2)，θ = 2*arcsin(‖od‖/2)
                    angle_deg = np.degrees(2 * np.arcsin(min(np.linalg.norm(od) / 2, 1.0)))
                    # 角度门：调度层传入 batch_angle_limit_deg 时以调度层标准为准
                    # （其公式已按 sin(航向差) 惩罚步数，固定 3° 二次否决是 2026-08-31
                    # 主因 bug——pass 段朝向差稳定 5° 被拦成每轮 5cm 单步+定位）
                    angle_gate = max(GO_FORWARD_BATCH_MAX_ANGLE_DEG,
                                     effective_angle_limit or 0.0)
                    if np.linalg.norm(od) <= turn_threshold_norm and angle_deg <= angle_gate:
                        batch_steps = min(batch_cap, int(dist / FORWARD_CM))
                        # 路径预检：整段批量路径离墙距离是否充足
                        if batch_steps > 1 and check_segment_clear(
                            state.current_position, state.current_orientation,
                            batch_steps, FORWARD_CM, OBSTACLE_THRESHOLD + 2.0
                        ):
                            print(f"  ★ 批量直行 {batch_steps} 步（go_forward，距目标 {dist:.1f}cm，离墙 {wall_dist:.1f}cm）")
                            if AUDIT is not None:
                                AUDIT.record_batch(batch_cap, batch_steps, batch_steps, "none",
                                                   angle_deg=angle_deg, angle_gate=angle_gate,
                                                   precheck_ok=True)
                            state.act("go_forward", times=batch_steps)
                            _audit_action()
                            continue
                        else:
                            reason = "dist_cut" if batch_steps <= 1 else "precheck_fail"
                            if AUDIT is not None:
                                AUDIT.record_batch(batch_cap, batch_steps, 1, reason,
                                                   angle_deg=angle_deg, angle_gate=angle_gate,
                                                   precheck_ok=False)
                    else:
                        if AUDIT is not None:
                            AUDIT.record_batch(batch_cap, 1, 1, "angle_gate",
                                               angle_deg=angle_deg, angle_gate=angle_gate)
                state.act(action)
                _audit_action()
            continue
        print("\n=== 本轮导航循环结束 ===")
        # 8. 到达
        if escaping:
            print(f"===== 已到达安全点 {target}，恢复原目标导航 =====")
            continue
        if stop_time > 0:
            print(f"===== 到达{phase} {target_pos}，停留 {stop_time} 秒 =====")
            state.act("stand")
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
# 调度层 + 关卡入口（v6 混合架构：A* 子目标 + 旧执行层）
# =====================================================================

# 航向自适应批量参数（v6：激进度由实测对准度决定，不由地理标签决定）
BATCH_ADAPTIVE_MAX_STEPS = 6     # 开阔区绝对上限
BATCH_CORRIDOR_HARD_CAP = 5      # 走廊内（净空 <21cm）硬顶
GUARDRAIL_MARGIN_CM = 1.0        # 护栏安全垫：护栏阈值 = 中心带余量 − 此值
REPLAN_DEVIATION_CM = 12.0       # 偏离路径中线超过此值 → 重规划
REPLAN_NO_PROGRESS_ROUNDS = 4    # 同一子目标 N 轮无进展 → 重规划


def adaptive_batch_steps(state, target, dist_cm):
    """航向自适应批量：返回 (步数, 允许航向差角)

    物理依据：横向漂移 = 批长 × sin(航向残差) ≤ 中心带余量。
    步数 = min(开阔 6, 中心带余量/sin(允许角)/5cm)，走廊（净空<21）硬顶 5。
    允许角 = 航向差本身（对准 <3° 时全额步数）——激进度由实测对准度决定。
    返回的第二值传给执行层 batch_angle_limit_deg（统一批量判定，
    避免 3° 固定门二次否决调度层已按 sin(角) 惩罚过的批量）。
    """
    ori = np.asarray(state.current_orientation, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64) - np.asarray(state.current_position, dtype=np.float64)
    tgt_n = np.linalg.norm(tgt)
    if tgt_n < 1e-9:
        return 1, 0.0
    from path_planner import clearance
    clr = clearance(float(state.current_position[0]), float(state.current_position[1]))
    cap = BATCH_CORRIDOR_HARD_CAP if clr < 21.0 else BATCH_ADAPTIVE_MAX_STEPS
    # 航向与目标连线夹角
    cos_a = float(np.clip(np.dot(ori, tgt / tgt_n), -1.0, 1.0))
    angle_deg = np.degrees(np.arccos(cos_a))
    margin = clr - 13.0  # 中心带余量（cm）
    if angle_deg < 1.0:
        return cap, 15.0        # 对准：地理上限全额；角度门给足（后续轮闭环）
    if margin <= 0.5:
        return 1, 15.0          # 已贴墙（理论上逃离会先接管）：单步保底
    # 允许批长 = margin / sin(angle)，步数 = 批长 / 5cm；允许角 = 当前航向差
    import math as _m
    allowed_cm = margin / _m.sin(_m.radians(angle_deg))
    steps = int(allowed_cm / FORWARD_CM)
    return max(1, min(cap, steps, BATCH_ADAPTIVE_MAX_STEPS)), angle_deg


def run_route(state, route, goal_name="终点"):
    """调度层：沿 A* 子目标序列逐段导航（v6 核心，~80 行）

    职责：子目标索引推进（只前进不后退）+ 语义分发（pass/corner/goal）+
    批量截短（发射前投影越线检查）+ 护栏/重规划守护。
    定位与动作全部由 navigate_to_target（执行层）完成。
    仲裁互斥：escaping 由执行层内处理，本层每轮只看导航结果。
    """
    from path_planner import clearance, plan_route
    wps = route.waypoints
    idx = 0
    prev_xy = list(state.current_position) if state.current_position is not None else list(wps[0].xy)
    no_progress = 0
    last_dist = None

    while idx < len(wps):
        wp = wps[idx]
        viz = getattr(state, "_viz", None)
        if viz is not None and hasattr(viz, "set_waypoint_idx"):
            viz.set_waypoint_idx(idx)
        dist = float(np.hypot(*(np.asarray(wp.xy) - np.asarray(state.current_position))))
        print(f"\n◆ 调度：子目标 {idx+1}/{len(wps)} {wp.semantics} {wp.xy}（距 {dist:.1f}cm）")

        # 护栏与重规划守护（执行层每轮定位后，本层在子目标切换间检查）
        if no_progress >= REPLAN_NO_PROGRESS_ROUNDS:
            print("  [守护] 多轮无进展，重规划。")
            return "replan"
        if last_dist is not None and dist > last_dist + 2.0:
            no_progress += 1
        else:
            no_progress = 0
        last_dist = dist

        semantics = "pass" if wp.semantics == "pass" else "precise"
        pass_line = (prev_xy, wp.xy) if wp.semantics == "pass" else None

        # Tolerant 判断：当前段足够长，且到达当前拐点后需要大转角
        next_wp = wps[idx + 1] if idx + 1 < len(wps) else None
        seg_len = float(np.hypot(wp.xy[0] - prev_xy[0], wp.xy[1] - prev_xy[1]))
        turn_angle = 0.0
        if next_wp is not None:
            in_dir = np.asarray(wp.xy, dtype=np.float64) - np.asarray(prev_xy, dtype=np.float64)
            out_dir = np.asarray(next_wp.xy, dtype=np.float64) - np.asarray(wp.xy, dtype=np.float64)
            n1 = np.linalg.norm(in_dir)
            n2 = np.linalg.norm(out_dir)
            if n1 > 1e-9 and n2 > 1e-9:
                cos_a = float(np.clip(np.dot(in_dir, out_dir) / (n1 * n2), -1.0, 1.0))
                turn_angle = float(np.degrees(np.arccos(cos_a)))
        tolerant = (next_wp is not None and seg_len >= MIN_SEG_LEN_FOR_TOLERANT
                    and turn_angle >= MIN_CORNER_ANGLE_FOR_TOLERANT)

        # 批量截短：本批终点+余量投影越线 → 传小上限给执行层
        max_steps, angle_limit = adaptive_batch_steps(state, wp.xy, dist)
        if wp.semantics == "pass" and dist < FORWARD_CM * (max_steps + 2):
            max_steps = max(1, int(dist / FORWARD_CM))  # 接近切换线：截短

        if AUDIT is not None:
            AUDIT.begin_segment(wp, _sim_locate_count(state))

        ok = navigate_to_target(state, wp.xy, None, 0.0, None, None,
                                semantics=semantics, max_batch_steps=max_steps,
                                batch_angle_limit_deg=angle_limit,
                                pass_line=pass_line, prev_waypoint=prev_xy,
                                tolerant=tolerant)
        if AUDIT is not None:
            AUDIT.end_segment(_sim_locate_count(state))
        if ok == "replan":
            return "replan"  # 执行期管廊越界，触发重规划
        if ok is False:
            return False  # 定位失败终止（执行层已 stand）
        # ok=True（到达/通过）或 None（不会出现；函数只返回 bool）
        prev_xy = list(wp.xy)
        idx += 1
        last_dist = None

    print(f"===== 到达{goal_name} =====")
    return True


def run_level(state):
    """goodluck 关卡主流程（v6 混合架构）

    A* 规划子目标序列 → 调度层逐段交给旧 navigate_to_target 执行。
    旧 ROUTE 手工路点体系保留于 git 历史；--end-at-last-stop = 目标改 [74,30]。
    """
    from path_planner import plan_route
    goal = END_AFTER_POS if END_AT_LAST_STOP else GOAL_POS
    print(f"导航目标：{goal}{'（--end-at-last-stop）' if END_AT_LAST_STOP else '（出口）'}")

    state.act("stand")
    state.set_head(HEAD_CENTER)
    state.raise_head()

    replans = 0
    while True:
        if not state.locate_with_retry():
            print("开局定位失败，程序终止。")
            return False
        route = plan_route(list(state.current_position), list(goal))
        if route is None:
            print("A* 无可行路径，程序终止。")
            return False
        print(f"[A*] 路线 {route.length_cm:.0f}cm，{len(route.waypoints)} 个子目标")
        viz = getattr(state, "_viz", None)
        if viz is not None and hasattr(viz, "set_route"):
            viz.set_route(route)

        result = run_route(state, route, goal_name="终点")
        if result is True:
            print("===== 全程完成 =====")
            return True
        if result is False:
            print("导航失败（定位终止），程序终止。")
            return False
        # result == "replan"
        replans += 1
        if replans > 3:
            print("重规划超限，程序终止。")
            return False
        print(f"[守护] 重规划 {replans}/3")
