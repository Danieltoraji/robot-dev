# -*- coding: utf-8 -*-
"""机器人智按按钮：拍照识别层 + 动作库常量。

识别层：拍照 → apriltag 库识别 → 返回 {id: 四角}。只跑真机，不留 PC 分支。
"""

import os
import sys

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

def press_button(state, tag_id):
    """寻找 → 接近 → 按下 一个按钮。桩函数，待实现。

    返回 True = 按到了；False = 没按到。
    """
    pass


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
