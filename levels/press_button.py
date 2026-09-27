# -*- coding: utf-8 -*-
"""机器人智按按钮：拍照识别层 + 动作库常量。
调参地点在第178行。
识别层：拍照 → apriltag 库识别 → 返回 {id: 四角}。只跑真机，不留 PC 分支。
"""

import inspect
import os
import sys
import time

import cv2
import numpy as np

# 允许直接运行本文件时找到仓库根
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =====================================================================
# 拍摄识别层
# =====================================================================

# 读图失败重试：真机上 fswebcam 报"已保存"后偶尔文件还没落盘，imread 返回 None，
# 紧接着 cvtColor 就抛 !_src.empty()（实测偶发）。等一下再读，最多重试这么多次。
read_retries = 3
read_retry_wait_s = 0.4


def detect_tags(state):
    """拍照，识别画面里所有 AprilTag。

    返回 {tag_id: 四角}；四角是 4×2 的像素坐标数组，顺序 左上/右上/右下/左下。
    角点顺序不用重排：apriltag 库返回的顺序经真机验证就是 左上→右上→右下→左下
    （等价 core.robot_core.TAG_CORNER_PERM 的恒等映射）。
    拍照失败或一个都没检出时返回 {}。

    读图失败会**重新拍一张**再试（不是重读同一个文件）。原因（真机实测 2026-09-26）：
    core.capture_image() 的照片名是秒级时间戳 /home/pi/Pictures/photo_<秒>.jpg，
    同一秒内拍两张 → 后一张覆盖前一张 → 读到的可能是被截断的坏文件，
    cv2.imread 返回 None、紧接着 cvtColor 抛 !_src.empty()。
    重读同一个坏路径没用，必须换一张新照片（新时间戳）。
    """
    for attempt in range(read_retries + 1):
        path = state.capture_image()
        if path is None:
            print("[press_button] 拍照失败")
            return {}
        try:
            results = state.detect_apriltag(path)
        except Exception as e:                   # noqa: BLE001 —— 读图/检测异常都当本帧无效
            if attempt < read_retries:
                time.sleep(read_retry_wait_s)
                print("[press_button] 读图/检测失败，重新拍一张（第 %d 次）：%s"
                      % (attempt + 1, path))
                continue
            print("[press_button] 检测失败（已重拍 %d 次）：%s: %s"
                  % (read_retries, type(e).__name__, e))
            return {}
        tags = {}
        for r in results:
            tags[int(r.tag_id)] = np.asarray(r.corners, dtype=np.float64).reshape(4, 2)
        return tags
    return {}


# =====================================================================
# 动作组常量
# =====================================================================
# 动作名前缀 A_ 是 core 的既有写法（levels/goodluck.py），直接沿用。
# 12 个动作名已于 2026-09-26 在真机 192.168.31.209 上逐一核对：
#   ls /home/pi/TonyPi/ActionGroups/ → 全部存在，press.d6a 也在
# 调用时只写动作名、不写 .d6a 后缀（AGC.runActionGroup 自己拼路径）。

# --- 站位与按压：press.d6a 8 帧 / 5500ms（Servo3/4/6/7/8/11/12/14/15/16）---
A_STAND = "stand"                    # 1 帧 / 500ms
A_PRESS = "press"                    # 8 帧 / 5500ms，单帧 Time: 1000 500 500 400 1000 600 1000 500

# --- 原地转向（大步：粗修）---
A_TURN_L = "turn_left"               # 实测 22.0°/次
A_TURN_R = "turn_right"              # 实测 25.7°/次
TURN_L_DEG = 22.0                    # levels/goodluck.py:126（2026-08-25 实机标定）
TURN_R_DEG = 25.7                    # levels/goodluck.py:127

# --- 原地转向（小步：精修）---
A_TURN_L_SMALL = "turn_left_small_step"     # 实测 8.625°/次
A_TURN_R_SMALL = "turn_right_small_step"    # 实测 5.200°/次
TURN_L_SMALL_DEG = 8.625
TURN_R_SMALL_DEG = 5.200
# 来源：新流程设计稿 §5.1/§5.2（每组 20 次×2 组：左 ±3σ=1.875°、右 ±3σ=0.150°）。
# ★ 两个方向必须各用各自的值：8.625/22.0 = 0.392，5.200/25.7 = 0.202，比例不同不能互换。

# --- 前进 / 后退 ---
A_FWD = "go_forward"                 # 实测 5.0cm/次
A_FWD_ONE = "go_forward_one_step"    # 实测 2.0cm/次
A_BACK = "back_one_step"             # 实测 3.2cm/次
FORWARD_CM = 5.0                     # levels/goodluck.py:121
FORWARD_ONE_CM = 2.0                 # levels/goodluck.py:122
BACK_CM = 3.2                        # levels/goodluck.py:123（原名 BACK_FAST_CM）

# --- 左右横移 ---
A_MOVE_L = "left_move"               # 实测 1.9cm/次
A_MOVE_R = "right_move"              # 实测 2.2cm/次
LEFT_MOVE_CM = 1.9                   # levels/goodluck.py:124
RIGHT_MOVE_CM = 2.2                  # levels/goodluck.py:125

# --- 头部（不走动作组，走舵机：state.set_head(pulse)）---
# 常量在 core/camera_config.py:29-36，用的时候从 core 引，别在这儿重抄
# HEAD_CENTER=1500 / HEAD_RIGHT=1050(-40.5°) / HEAD_LEFT=1950(+40.5°)
# HEAD_WIDE_RIGHT=800(-63°) / HEAD_WIDE_LEFT=2200(+63°)


# =====================================================================
# 按按钮：寻找 → 接近 → 按下
# =====================================================================
# 相机内参（值取自 core/camera_config.py，别在这儿改）
focal_x = 1944.903664123011      # 水平焦距 fx
focal_y = 1950.095436307893      # 竖直焦距 fy
image_center_x = 1283.069051100245   # 画面中心横向 = 主点（**不是** 2592/2=1296）
image_center_y = 983.1983420212778   # 画面中心纵向 = 主点
# 径向畸变（不用切向 p1/p2，core 里就是 0）
distort_k1 = -0.384402275498781
distort_k2 = 0.284681889150075

tag_size_cm = 4.5                # 标签黑框边长（2026-09-26 真机卷尺实测：整块黑框 4.5cm）
target_ids = (102, 101)          # 本关要按的两个按钮（先 102 后 101）
solve_iters = 3                  # 联立求解的迭代次数
min_cos_view = 0.4               # 视线与标签法线夹角超过 ~66° 就不信这个读数

# 头部转向（头不走动作组，走舵机 state.set_head(pulse)；脉宽→角度 =(pulse−1500)×0.09）
HEAD_CENTER = 1500               # 回正
HEAD_LEFT = 1950                 # 左转 +40.5°
HEAD_RIGHT = 1050                # 右转 −40.5°
PITCH_LEVEL = 1500               # 俯仰水平（1500=水平，越小越低头）
HEAD_SETTLE_S = 0.5              # 初始化头部后等舵机停稳的时间（秒）

# ---------------------------------------------------------------------
# 对准用的像素容差（画面竖着切五份：最中间那份不用转）
# ---------------------------------------------------------------------
# 换算关系：转 1° 标签在画面上横向移动 focal_x × π/180 = 33.9 px
px_per_deg = focal_x * np.pi / 180.0
# 各动作一步推动多少像素（由实测角度换算，不是猜的）：
#   turn_left            22.000° → 747 px
#   turn_right           25.700° → 872 px
#   turn_left_small_step  8.625° → 293 px
#   turn_right_small_step 5.200° → 177 px
#
# 死区半宽 = 339px（= 10.0°）的依据：
#   · 必须 ≥ 最大的一步小转（左小转 8.625° = 293px），否则转一步会整个跨过死区、永远回不来
#   · 用户定：对准允许偏 10°，即 339px
#   收敛性：配上"选转完离中心最近的动作"这条规则，全画面 100% 收敛、最多 2 步
#   （按区间直接定动作只有 293/1291 收敛，已实测否决）
dead_zone_px = 10.0 * focal_x * np.pi / 180.0
# 左/右两侧"大步区"的下界（画面切五份的那两条内边界）
big_turn_px = 872.0
# 转动候选（度，左正右负）：不动 / 右小 / 左小 / 右大 / 左大
turn_choices_deg = (0.0, TURN_R_SMALL_DEG, -TURN_L_SMALL_DEG, TURN_R_DEG, -TURN_L_DEG)

# ---------------------------------------------------------------------
# 判据常量：逐目标独立（两个按钮的面板位置/朝向可能不同，阈值就得分开调）
# ---------------------------------------------------------------------
# 每个目标点一套。改哪个目标的哪个数，就改 target_defaults 下面那一份。
# 打印时读的是这里的值，所以现场按日志调完，回来改这里。
target_defaults = dict(
    # 第一阶段：纵深小于它就进第二阶段
    target_distance_cm=25.0,
    # 第二阶段：距离超过它就退回第一阶段
    phase2_too_far_cm=35.0,
    # "可以按"的距离区间
    press_low_cm=20.0,
    press_high_cm=23.0,
    # 标签中心离画面中心的容差（px）
    phase2_center_px=60.0,
    # 标签法线垂直于视线的容差：视线与法线夹角的余弦要大于它
    #   1.000 = 完全正对；0.985 ≈ 偏 10°；0.966 ≈ 偏 15°
    square_cos_min=0.985,
)
# 每个目标的独立覆盖（只写要改的那几个，其余继承 target_defaults）
target_overrides = {
    102: dict(press_low_cm=18.0, press_high_cm=21.0),
    101: dict(press_low_cm=20.0, press_high_cm=23.0),
}


class TargetParams:
    """一个目标点的全部判据阈值（读的是 attribute，改的是上面那两张表）。"""

    def __init__(self, tag_id):
        self.tag_id = tag_id
        for k, v in target_defaults.items():
            setattr(self, k, v)
        for k, v in target_overrides.get(tag_id, {}).items():
            if k not in target_defaults:
                raise KeyError("未知的逐目标阈值：%s" % k)
            setattr(self, k, v)

    def __repr__(self):
        return ("TargetParams(%d: 进二阶段<%.1f 退回>%.1f 可按%.1f~%.1f 中心±%.0fpx 正对>%.3f)"
                % (self.tag_id, self.target_distance_cm, self.phase2_too_far_cm,
                   self.press_low_cm, self.press_high_cm, self.phase2_center_px,
                   self.square_cos_min))

camera_matrix = np.array([[focal_x, 0.0, image_center_x],
                          [0.0, focal_y, image_center_y],
                          [0.0, 0.0, 1.0]], dtype=np.float64)
distort_coeffs = np.array([distort_k1, distort_k2, 0.0, 0.0], dtype=np.float64)


def undistort_corners(corners):
    """四个角点去畸变 → 消除径向畸变后的理想像素坐标（4×2）。

    只算这 4 个点，不做整图 remap（整图 remap 2592×1944 又慢又会重采样，
    把 apriltag 的亚像素角点精度削掉约 0.5px）。
    不去畸变的代价（合成实测，真值纵深 30cm）：标签在画面中心差 0.5%、
    偏 600px 差 8.2%、偏 1100px 差 22.2%，方向一律是"以为更远"。
    """
    pts = np.asarray(corners, dtype=np.float64).reshape(-1, 1, 2)
    normalized = cv2.undistortPoints(pts, camera_matrix, distort_coeffs).reshape(-1, 2)
    return np.column_stack([focal_x * normalized[:, 0] + image_center_x,
                            focal_y * normalized[:, 1] + image_center_y])


def left_right_edge_lengths(corners):
    """四角 → (左竖边长, 右竖边长)"""
    left = float(np.linalg.norm(corners[3] - corners[0]))
    right = float(np.linalg.norm(corners[2] - corners[1]))
    return left, right


def top_bottom_widths(corners):
    """四角 → (顶边宽, 底边宽)，都是横边在画面上的像素长度"""
    top = float(abs(corners[1][0] - corners[0][0]))
    bottom = float(abs(corners[2][0] - corners[3][0]))
    return top, bottom


def tag_center_x(corners):
    """标签中心的横向像素坐标"""
    return float(corners[:, 0].mean())


def tag_reading(corners):
    """一帧里这个标签的两个读数：(纵深 cm, 正对程度)。

    正对程度 = 视线与标签法线夹角的余弦，1.0 = 标签正对镜头、0 = 完全侧对。
    求解不了时返回 (0.0, 0.0)。
    """
    if corners is None:
        return 0.0, 0.0
    corners = undistort_corners(corners)

    top, bottom = top_bottom_widths(corners)
    width_px = (top + bottom) / 2.0
    if width_px <= 0.0:
        return 0.0, 0.0

    left, right = left_right_edge_lengths(corners)
    if left <= 0.0 or right <= 0.0:
        return 0.0, 0.0
    edge_ratio = left / right

    # 偏角方向：标签在画面右半边 ⇒ 标签法线偏右
    apart = 1.0 if tag_center_x(corners) > image_center_x else -1.0

    depth = focal_x * tag_size_cm / width_px          # 先按正对取初值
    cos_view = 1.0
    for _ in range(solve_iters):
        sin_view = (edge_ratio - 1.0) / (edge_ratio + 1.0) * (2.0 * depth / tag_size_cm)
        sin_view = max(-1.0, min(1.0, apart * abs(sin_view)))
        cos_view = float(np.sqrt(1.0 - sin_view * sin_view))
        depth = focal_x * tag_size_cm * cos_view / width_px

    if cos_view < min_cos_view:
        return 0.0, 0.0
    return float(depth), cos_view


def tag_depth_cm(corners):
    """标签的纵深（cm）：沿相机光轴方向，相机到标签平面的距离。

    进来先对 4 个角点去畸变（不去畸变，画面边缘的标签会被判成远 22%）。

    怎么算：宽度 w 定尺度，左右两条竖边的长短比定标签还偏着多少角度，
    两个未知数（纵深、偏角）联立迭代 solve_iters 次。

    三个已知边界（实测合成数据，不是猜的）：
      · 标签正对镜头时精确到 0.0%（侧偏 10/20/30/40cm 都测过）
      · 标签自己偏着（斜看）会低估：偏到视线夹角 34° 时低 25%
      · **只有标签在画面中心时，读数才等于真正的纵深**：标签偏在画面一侧时，
        透视本身让左右竖边一长一短，求解器会把"位置造成的长短差"错当成
        "标签转过去的偏角"（侧偏 30cm 时甚至会算出 cos>1）。
        所以用这个读数前必须先对准（第一阶段就是干这个的）。
      · 视线与标签法线夹角超过 min_cos_view 对应的角度（约 66°）直接返回 0.0，
        当作"读数不可用"，免得把崩掉的数当成"还没到"
    """
    return tag_reading(corners)[0]


def distance_between(state, tag_id, low_cm, high_cm):
    """拍一张照片，判断目标标签的纵深是否落在 [low_cm, high_cm] 里。是 → True。

    返回 False 也包括"这一帧没检出该标签"和"读数不可用"。
    """
    tags = detect_tags(state)
    if tag_id not in tags:
        return False
    depth = tag_depth_cm(tags[tag_id])
    return low_cm < depth < high_cm


def distance_under(state, tag_id, cm=None):
    """拍一张照片，判断目标标签的纵深是否小于 cm。是 → True。

    cm 不给就用该目标自己的 target_distance_cm（逐目标阈值）。
    """
    if cm is None:
        cm = TargetParams(tag_id).target_distance_cm
    return distance_between(state, tag_id, 0.0, cm)


def tag_is_square_on(corners, cos_min):
    """标签法线是否垂直于视线（即标签是否正对镜头）。是 → True。

    cos_min 由调用方给（逐目标阈值 TargetParams.square_cos_min）。
    """
    if corners is None:
        return False
    _, cos_view = tag_reading(corners)
    return cos_view > cos_min


def turn_for_offset(offset_px):
    """标签中心偏离画面中心 offset_px 时，该下发的 (动作名, 转的度数)。

    画面竖着切五份：|offset| < dead_zone_px 是最中间那份 → 不用转（返回 None）。
    其余四份里怎么选：**算出每个候选动作转完之后离中心还剩多远，选剩得最少的那个**
    （候选 = 不动 / 左小 / 右小 / 左大 / 右大）。

    为什么不是"在里面两份就用小转"：偏差 600px 时，小左转完还剩 307px（还在外面），
    而大左转完只剩 −147px（进死区了）。按区间直接定动作会让它一直差一点、收敛不了。
    """
    if abs(offset_px) < dead_zone_px:
        return None

    best_action, best_deg, best_left = None, None, None
    for deg in turn_choices_deg:
        left = abs(offset_px - deg * px_per_deg)     # 转完之后偏差还剩多少
        if best_left is None or left < best_left:
            best_left = left
            best_deg = deg
            # 度数为正 = 向右转（把标签往画面右边推）
            if deg == 0.0:
                best_action = None
            elif deg > 0:
                best_action = A_TURN_R if deg > 20.0 else A_TURN_R_SMALL
            else:
                best_action = A_TURN_L if deg < -20.0 else A_TURN_L_SMALL

    if best_action is None:
        return None
    return best_action, best_deg


def log_measure(tag_id, corners, stage, reason, action, times=1):
    """每轮决策打一行：观测量 + 命中哪条判据 + 下发什么动作。

    真机上滚动看这一行就能判断是"在收敛"还是"卡住了"。
    """
    offset = tag_center_x(corners) - image_center_x
    depth, cos_view = tag_reading(corners)
    print("[按按钮][%d][%s] 偏差%+7.1fpx 纵深%6.2fcm 正对%.3f | 判定:%s | 动作:%s×%d"
          % (tag_id, stage, offset, depth, cos_view, reason, action, times))


def log_search(tag_id, what):
    """找目标时打一行（看到了什么 / 下一步怎么找）"""
    print("[按按钮][%d][寻找] %s" % (tag_id, what))


def next_action(corners, target, stage="二"):
    """纯判据：给定一帧角点 + 该目标的阈值，返回该做的 (动作, 次数, 命中判据)。

    `stage` 是"一"时用第一阶段判据（对准 + 靠近），否则用第二阶段判据。
    阈值全部来自 `target`（TargetParams，逐目标独立）。
    抽出来是为了让 --probe 打印的"下一步会做什么"和真正执行时是同一段逻辑。
    """
    offset = tag_center_x(corners) - image_center_x
    depth, _ = tag_reading(corners)

    if stage == "一":
        choice = turn_for_offset(offset)
        if choice is not None:
            return choice[0], 1, "阶段一:偏出死区(%+.0fpx)" % offset
        if not (0.0 < depth < target.target_distance_cm):
            return A_FWD, 1, "阶段一:还不够近(%.2fcm，目标<%.0f)" % (depth, target.target_distance_cm)
        return None, 0, "阶段一:对准且够近 → 进第二阶段"

    if depth > target.phase2_too_far_cm or depth <= 0.0:
        return None, 0, "阶段二:太远/读数不可用(%.2fcm>%.0f) → 退回第一阶段" % (
            depth, target.phase2_too_far_cm)
    if abs(offset) > target.phase2_center_px:
        return (A_MOVE_L if offset < 0 else A_MOVE_R), 1, "阶段二:不在视野中心(%+.0fpx)" % offset
    if not tag_is_square_on(corners, target.square_cos_min):
        choice = turn_for_offset(offset)
        act = choice[0] if choice is not None else A_TURN_R_SMALL
        return act, 1, "阶段二:标签法线没垂直视线"
    if not (target.press_low_cm < depth < target.press_high_cm):
        return (A_FWD if depth > target.press_high_cm else A_BACK), 1, \
               "阶段二:距离%.2fcm 不在 %.0f~%.0f" % (depth, target.press_low_cm, target.press_high_cm)
    return A_PRESS, 1, "阶段二:全部满足 → 按下"


def press_button(state, tag_id):
    """寻找 → 接近 → 按下 一个按钮。

    分两个阶段，每轮拍一张照片：

    第一阶段（靠近）：对准 + 走到 target_distance_cm 以内
        看不到该 id      → 头左右摇着找（左转头 → 右转头）
                           两个方向都摇过还是没有 → 头回正 + 身体右转两步，再摇一遍
        偏出最中间那份    → 按 turn_for_offset 选出的动作转一步
        对准了但不够近    → 直行一步
        对准且够近        → 进第二阶段

    第二阶段（精对准 + 按下）：
        看不到该 id        → 同第一阶段
        距离 > 35cm        → 退回第一阶段继续靠近
        标签不在视野中心   → 左右平移一步
        标签法线不垂直视线 → 转身（用不跨过死区的那个动作）
        距离不在 14~17cm   → 前进 / 后退一步
        以上都满足         → 按下，然后后退一步脱离

    **没有次数上限、不放弃**：唯一的出口是"按下去"。
    每一步的决定都会打印出来（判定 + 动作 + 观测量）。
    """
    target = TargetParams(tag_id)            # 该目标点的全部阈值（逐目标独立）
    _servo(state, "set_head", HEAD_CENTER)   # 头回正（每次按按钮都先摆正）
    print("[按按钮][%d] 开始：头回正，开始寻找 | %r" % (tag_id, target))

    # ---------------- 第一 / 第二阶段共用的找目标 ----------------
    look_left = True                         # 头先往哪边摇
    head_tries = 0                           # 这一轮摇了几次（左、右各算一次）

    def find_target():
        """拍一张照片：看到目标返回它的角点，看不到就按下面的顺序找。

            if 没看到:
                左转头 → 看到了? 左转身两次 → goto A
                右转头 → 看到了? 右转身两次 → goto A
                右转身两次
            A: 看到目标 → 返回它的角点
        """
        nonlocal look_left, head_tries
        while True:
            tags = detect_tags(state)
            seen = tags.get(tag_id)

            if seen is None:
                # ① 左转头
                state.set_head(HEAD_LEFT)
                log_search(tag_id, "没看到 → 左转头(%d)" % HEAD_LEFT)
                look_left = True                     # 已经偏到左边
                head_tries = 1
                tags = detect_tags(state)
                seen = tags.get(tag_id)
                if seen is not None:
                    log_search(tag_id, "左转头看到了 → 左转身两次")
                    state.act(A_TURN_L, 2)
                    _servo(state, "set_head", HEAD_CENTER)
                    head_tries = 0
                    look_left = True
                    continue                         # goto A

                # ② 右转头
                state.set_head(HEAD_RIGHT)
                log_search(tag_id, "左转头也没有 → 右转头(%d)" % HEAD_RIGHT)
                look_left = False                    # 已经偏到右边
                head_tries = 1
                tags = detect_tags(state)
                seen = tags.get(tag_id)
                if seen is not None:
                    log_search(tag_id, "右转头看到了 → 右转身两次")
                    state.act(A_TURN_R, 2)
                    _servo(state, "set_head", HEAD_CENTER)
                    head_tries = 0
                    look_left = True
                    continue                         # goto A

                # ③ 两边都没有 → 右转身两次，回头再来一遍
                _servo(state, "set_head", HEAD_CENTER)
                log_search(tag_id, "左右转头都没有 → 头回正 + 右转身两次")
                state.act(A_TURN_R, 2)
                head_tries = 0
                look_left = True
                continue

            # A: 看到目标了 —— 头回正后重新取一帧，保证读数不是歪着头拍的
            _servo(state, "set_head", HEAD_CENTER)
            if head_tries:
                return None
            return seen

    # ---------------- 第一阶段：走近到该目标的 target_distance_cm ----------------
    def approach():
        """对准 + 靠近，直到纵深 < target.target_distance_cm 才返回。"""
        while True:
            corners = find_target()
            if corners is None:
                continue
            action, times, reason = next_action(corners, target, stage="一")
            if action is None:
                log_measure(tag_id, corners, "阶段一", reason, "(无)", 0)
                return corners               # 对准且够近 → 交给第二阶段
            log_measure(tag_id, corners, "阶段一", reason, action, times)
            state.act(action, times)

    # ---------------- 第二阶段：精对准 + 按下 ----------------
    while True:
        corners = find_target()
        if corners is None:
            continue

        action, times, reason = next_action(corners, target, stage="二")
        if action is None:
            log_measure(tag_id, corners, "阶段二", reason, "(退回)", 0)
            approach()                       # 太远 → 退回第一阶段靠近
            continue

        log_measure(tag_id, corners, "阶段二", reason, action, times)
        state.act(action, times)

        if action == A_PRESS:
            state.act(A_BACK)                # 后退一步脱离
            state.act(A_PRESS)                # 再按一次，保险
            state.act(A_BACK)                # 后退一步脱离
            state.set_head(HEAD_CENTER)
            print("[按按钮][%d] 已按下，后退脱离，完成" % tag_id)
            return True


# =====================================================================
# 主控制流
# =====================================================================

def _servo(state, method_name, pulse):
    """给头部舵机下发脉宽，尽量强制下发（不管"记录值 == 目标值"）。

    真机 core 的 set_head/set_pitch 带 force 参数（用它绕过"相同脉宽直接 return"），
    但有的版本没这个参数 —— 先看签名再决定传不传，不靠 try/except 猜
    （否则别处的 TypeError 会被吞掉）。
    """
    method = getattr(state, method_name)
    try:
        accepts_force = "force" in inspect.signature(method).parameters
    except (TypeError, ValueError):
        accepts_force = False
    if accepts_force:
        method(pulse, force=True)
    else:
        method(pulse)


def init_head(state):
    """关卡开始前初始化头部：俯仰摆到水平、转头回正。

    为什么要强制下发：RobotState.__init__ 把 current_head_pulse 记为 HEAD_CENTER、
    current_pitch_pulse 记为 1500，但**都不发指令**；而 set_head/set_pitch 在
    "目标脉宽 == 当前记录值"时直接 return ⇒ 不强制就等于空操作，光轴还是歪的。
    （实测光轴不水平会直接污染纵深读数：俯角 10° 偏 1.7%、20° 偏 5.2%。）
    """
    print("[按按钮] 初始化头部：俯仰 → 水平(%d)，转头 → 回正(%d)" % (PITCH_LEVEL, HEAD_CENTER))
    _servo(state, "set_pitch", PITCH_LEVEL)
    _servo(state, "set_head", HEAD_CENTER)
    time.sleep(HEAD_SETTLE_S)            # 等舵机停稳再拍照
    print("[按按钮] 头部初始化完成（脉宽：俯仰=%s，转头=%s）"
          % (getattr(state, "current_pitch_pulse", "?"),
             getattr(state, "current_head_pulse", "?")))


def run_level(state):
    """关卡入口（main.py 调用）。

    初始化头部 → 开局 → 前进5步 → 左转5步 → 按第一个按钮 → 后退5步 → 右转5步
              → 按第二个按钮 → 后退5步 → 右转5步
    """
    init_head(state)

    state.act(A_STAND)
    state.act(A_FWD, 5)
    state.act(A_TURN_L, 5)
    press_button(state, target_ids[0])   # 第一个按钮
    state.act(A_BACK, 5)
    state.act(A_TURN_R, 5)
    press_button(state, target_ids[1])   # 第二个按钮
    state.act(A_BACK, 5)
    state.act(A_TURN_R, 5)


def probe(state, tag_ids=target_ids):
    """只测量、不动：拍一张，打印判据与"下一步会做什么"（一个动作都不下发）。"""
    print("[探针] 拍照 + 识别（不发任何动作）...")
    tags = detect_tags(state)
    print("[探针] 检出 %d 个 tag：%s" % (len(tags), sorted(tags)))
    for tid, c in sorted(tags.items()):
        offset = tag_center_x(c) - image_center_x
        depth, cos_view = tag_reading(c)
        param = TargetParams(tid)
        print("  id=%d  中心u=%7.1f  偏差%+7.1fpx  纵深%7.2fcm  正对%.3f  %s"
              % (tid, tag_center_x(c), offset, depth, cos_view,
                 "正对" if cos_view > param.square_cos_min else "没正对"))
    for tid in tag_ids:
        param = TargetParams(tid)
        print("  目标 %d 阈值：进二阶段<%.1fcm  退回>%.1fcm  可按%.1f~%.1fcm  中心±%.0fpx  正对>%.3f"
              % (tid, param.target_distance_cm, param.phase2_too_far_cm,
                 param.press_low_cm, param.press_high_cm,
                 param.phase2_center_px, param.square_cos_min))
        if tid not in tags:
            print("     本帧没看到")
            continue
        a1, t1, r1 = next_action(tags[tid], param, stage="一")
        a2, t2, r2 = next_action(tags[tid], param, stage="二")
        print("     阶段一判据 → %s×%d   (%s)" % (a1 or "无", t1, r1))
        print("     阶段二判据 → %s×%d   (%s)" % (a2 or "无", t2, r2))
    return 0


if __name__ == "__main__":
    import sys as _sys

    from core.robot_core import RobotState

    _st = RobotState(tag_poses={})

    if "--probe" in _sys.argv:
        # 只测量不动。注意：这条路**不初始化头部**，所以头不在位时很可能一个 tag 都看不到
        raise SystemExit(probe(_st))

    if "--run" in _sys.argv:
        # 直接跑关卡（含头部初始化），等价于 python main.py press_button
        run_level(_st)
        raise SystemExit(0)

    print("用法：")
    print("  python levels/press_button.py --run      # 跑整个关卡（会初始化头部，然后开始动）")
    print("  python levels/press_button.py --probe    # 只拍一张看判据，不动（不初始化头部）")
    print()
    print("提示：不带参数只会拍一张照片，头部不初始化 —— 头不在位时通常一个 tag 都看不到。")
    print("      要跑关卡请用 --run，或 python main.py press_button")
