# -*- coding: utf-8 -*-
"""机器人智按按钮：拍照识别层 + 动作库常量。

识别层：拍照 → apriltag 库识别 → 返回 {id: 四角}。只跑真机，不留 PC 分支。
"""

import os
import sys

import cv2
import numpy as np

# 允许直接运行本文件时找到仓库根
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =====================================================================
# 拍摄识别层
# =====================================================================

def detect_tags(state):
    """拍照，识别画面里所有 AprilTag。

    返回 {tag_id: 四角}；四角是 4×2 的像素坐标数组，顺序 左上/右上/右下/左下。
    角点顺序不用重排：apriltag 库返回的顺序经真机验证就是 左上→右上→右下→左下
    （等价 core.robot_core.TAG_CORNER_PERM 的恒等映射）。
    拍照失败或一个都没检出时返回 {}。
    """
    path = state.capture_image()
    if path is None:
        print("[press_button] 拍照失败")
        return {}

    tags = {}
    for r in state.detect_apriltag(path):
        tags[int(r.tag_id)] = np.asarray(r.corners, dtype=np.float64).reshape(4, 2)
    return tags


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

tag_size_cm = 5.0                # 标签黑框边长
target_distance_cm = 30.0        # 纵深小于它就够近
solve_iters = 3                  # 联立求解的迭代次数
min_cos_view = 0.4               # 视线与标签法线夹角超过 ~66° 就不信这个读数

# 头部转向（头不走动作组，走舵机 state.set_head(pulse)；脉宽→角度 =(pulse−1500)×0.09）
HEAD_CENTER = 1500               # 回正
HEAD_LEFT = 1950                 # 左转 +40.5°
HEAD_RIGHT = 1050                # 右转 −40.5°

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


def tag_depth_cm(corners):
    """标签的纵深（cm）：沿相机光轴方向，相机到标签平面的距离。

    进来先对 4 个角点去畸变（不去畸变，画面边缘的标签会被判成远 22%）。

    怎么算：宽度 w 定尺度，左右两条竖边的长短比定标签还偏着多少角度，
    两个未知数（纵深、偏角）联立迭代 solve_iters 次。

    两个已知边界（实测合成数据，不是猜的）：
      · 标签正对镜头时精确到 0.0%（侧偏 10/20/30/40cm 都测过）
      · 标签自己偏着（斜看）会低估：偏到视线夹角 34° 时低 25%
      · 视线与标签法线夹角超过 min_cos_view 对应的角度（约 66°）直接返回 0.0，
        当作"读数不可用"，免得把崩掉的数当成"还没到"
    """
    if corners is None:
        return 0.0
    corners = undistort_corners(corners)

    top, bottom = top_bottom_widths(corners)
    width_px = (top + bottom) / 2.0
    if width_px <= 0.0:
        return 0.0

    left, right = left_right_edge_lengths(corners)
    if left <= 0.0 or right <= 0.0:
        return 0.0
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
        return 0.0
    return float(depth)


def distance_under(state, tag_id, cm=target_distance_cm):
    """拍一张照片，判断目标标签的纵深是否小于 cm。是 → True。

    返回 False 也包括"这一帧没检出该标签"和"读数不可用"。
    """
    tags = detect_tags(state)
    if tag_id not in tags:
        return False
    depth = tag_depth_cm(tags[tag_id])
    return 0.0 < depth < cm


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


def press_button(state, tag_id):
    """寻找 → 接近 → 按下 一个按钮。

    目前只实现了第一阶段（靠近）：对准 + 走到 30cm 以内。
    每轮拍一张照片：
        看不到该 id      → 头左右摇着找（左转头 → 右转头）
                           两个方向都摇过还是没有 → 头回正 + 身体右转两步，再摇一遍
        偏出最中间那份    → 按 turn_for_offset 选出的动作转一步
        对准了但不够近    → 直行一步
        对准且够近        → 第一阶段完成

    **没有次数上限、不放弃**：唯一的出口是"对准且够近"。
    """
    state.current_head_pulse = None          # 头部默认值 1500 且不发指令，先清空
    state.set_head(HEAD_CENTER)

    look_left = True                         # 头先往哪边摇
    head_tries = 0                           # 这一轮摇了几次（左、右各算一次）
    while True:
        tags = detect_tags(state)

        if tag_id not in tags:
            # 阶段一（1）：看不到就找
            if head_tries < 2:
                # 头左右摇：先左转头，再右转头
                state.set_head(HEAD_LEFT if look_left else HEAD_RIGHT)
                look_left = not look_left
                head_tries += 1
            else:
                # 两下都没摇到 → 头回正，身体右转两步，再从头的左摇重新来
                state.set_head(HEAD_CENTER)
                state.act(A_TURN_R, 2)
                head_tries = 0
                look_left = True
            continue

        corners = tags[tag_id]
        state.set_head(HEAD_CENTER)          # 看到目标就把头回正再判
        head_tries = 0
        look_left = True

        choice = turn_for_offset(tag_center_x(corners) - image_center_x)
        if choice is not None:
            # 阶段一（2）：没对准，转一步
            state.act(choice[0])
        elif not (0.0 < tag_depth_cm(corners) < target_distance_cm):
            # 阶段一（3）：对准了但还不够近，直行
            state.act(A_FWD)
        else:
            return True                      # 对准且够近，第一阶段完成


# =====================================================================
# 主控制流
# =====================================================================

def run_level(state):
    """关卡入口（main.py 调用）。

    开局 → 前进5步 → 左转5步 → 按第一个按钮 → 后退5步 → 右转5步
         → 按第二个按钮 → 后退5步 → 右转5步
    """
    state.current_pitch_pulse = None     # 俯仰默认值 1500 且不发指令，必须先清空
    state.set_pitch(1500)                # 光轴水平，否则几何不成立

    state.act(A_STAND)
    state.act(A_FWD, 5)
    state.act(A_TURN_L, 5)
    press_button(state, 102)             # 第一个按钮
    state.act(A_BACK, 5)
    state.act(A_TURN_R, 5)
    press_button(state, 101)             # 第二个按钮
    state.act(A_BACK, 5)
    state.act(A_TURN_R, 5)


if __name__ == "__main__":
    from core.robot_core import RobotState

    st = RobotState(tag_poses={})
    tags = detect_tags(st)
    print(f"检出 {len(tags)} 个 tag：{sorted(tags)}")
    for tid, c in tags.items():
        print(f"  id={tid}")
        for name, p in zip(("TL", "TR", "BR", "BL"), c):
            print("     %s  u=%8.2f  v=%8.2f" % (name, p[0], p[1]))
