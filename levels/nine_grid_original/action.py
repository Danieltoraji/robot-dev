# -*- coding: utf-8 -*-
"""参考原版九宫格的**动作与云台层**（搬运自参考代码，不参与本仓库其它关卡）

来源：`reference code/九宫格视觉导航/robot/action.py`（叶雨岑、陶鲁玥，2026-08）
用途：`levels/nine_grid_original/`（参考原版通关流程）专用的动作原语与云台姿态。
隔离：本模块只被 `nine_grid_original` 这个包 import；`levels/nine_grid.py`、
      `levels/nine_grid_three_stage.py`、`levels/nine_grid_shared.py` 一行都不碰。

与参考版的差异（**只有两处，都不改任何判定**）：
  1. 硬件句柄改用 `core.robot_core` 里已有的 `AGC` / `ctl`（同一套 hiwonder SDK，
     导入失败时同样降级为 None）。参考版在文件顶层 `rrc.Board()`，
     在 PC 上 import 即崩；这里改成"用到才报错"，保证 `python main.py <其它关卡>`
     不会因为本模块而失败。
  2. 保真保留了 `with_stand=` 参数（它决定动作组结束后机器人是否回到站立姿态），
     因此没有走 `RobotState.run_action()`（那里不带这个参数）。

⚠️ 现场待标定（本轮不动，见 docs/关卡算法/彩色数字九宫格-nine_grid/
   参考原版通关代码-nine_grid_original.md 的常量表）：
   `LOOK_FORWARD_PULSE`（参考版自己就写着 TODO 现场标定）、`LOOK_DOWN_PULSE`。
"""

import time

# 硬件句柄：与 core.robot_core 共用同一份（try/except 降级在那里做过了）
from core.robot_core import AGC, ctl


# ---------------- 云台参数 ----------------

YAW_SERVO = 2
PITCH_SERVO = 1

YAW_CENTER = 1500


# ---------------- 云台俯仰 ----------------

# 导航姿态（略微下俯，保证转弯时不会拍到机器人自己）
LOOK_FORWARD_PULSE = 1200      # TODO：现场标定
# 检测姿态（最低头）
LOOK_DOWN_PULSE = 1000

CURRENT_YAW = 0

MAX_YAW = 90


# 绑定的 RobotState（接入用；见 bind_state）
_STATE = None


def bind_state(state):
    """把 RobotState 借给动作层（一局开始时调一次即可）

    只影响一件事：**不带 `with_stand`** 的动作组改走 `state.run_action()`，
    好让 `main.py` 的 trace（TraceRobotState）能统计动作次数——两条路最终都是
    `AGC.runActionGroup(name, times=times)`，行为完全一致。
    带 `with_stand` 的动作组仍直连 AGC（`RobotState.run_action` 没有这个参数）。
    """
    global _STATE
    _STATE = state


def _require_hardware():
    """硬件不可用时报清晰错误（参考版是 import 期直接崩，这里推迟到调用点）"""
    if AGC is None or ctl is None:
        raise RuntimeError(
            "hiwonder 硬件库不可用（非机器人环境？）：本关卡只能在真机上跑。"
            "参考原版通关代码没有仿真器，见 levels/nine_grid_original/level.py 的文件头。")
    return AGC, ctl


def _run(name, times=1, with_stand=None):
    """执行动作组；`with_stand` 为 None 时用 SDK 默认值（与参考版逐字一致）"""
    if with_stand is None and _STATE is not None:
        _STATE.run_action(name, times=times)
        return
    agc, _ = _require_hardware()
    if with_stand is None:
        agc.runActionGroup(name, times=times)
    else:
        agc.runActionGroup(name, times=times, with_stand=with_stand)


# =====================================================================
# 动作组（名字与 with_stand 全部照抄参考版）
# =====================================================================

def over_hurdle():
    """参考版的"越障脱困"：爬一次 + 后退两步（现场用于卡住时逃出）"""
    _run('climb_stairs', times=1, with_stand=True)
    _run('back_one_step', times=2, with_stand=False)


def move_forward():
    _run('go_forward_one_step', times=5, with_stand=False)


def move_backward_one():
    _run('back_one_step', times=1, with_stand=False)


def move_forward_1():
    _run('go_forward', times=3, with_stand=True)


def move_forward_fast():
    _run('go_forward_fast', times=3, with_stand=True)


def move_forward_small():
    """小步前进：接近目标后用，提高停车精度"""
    _run('go_forward_one_step', times=2, with_stand=True)


def turn_small_angle_right():
    _run('turn_right_small_step', times=1)


def turn_small_angle_left():
    _run('turn_left_small_step', times=1)


def turn_big_angle_left():
    _run('turn_left', times=2)


def turn_big_angle_right():
    _run('turn_right', times=2)


# =====================================================================
# 云台（头部 yaw=ID2 / 俯仰 pitch=ID1，与参考版同 ID）
# =====================================================================

def angle_to_pulse(angle):
    """角度 → yaw 舵机脉宽（参考版：10μs/度，中心 1500）"""
    pulse = 1500 + 10 * angle
    return pulse


def turn_to_angle(angle):
    """把 yaw 舵机转到指定角度（**通关流程里没有调用**，照抄保留）"""
    global CURRENT_YAW
    _, c = _require_hardware()

    angle = max(-MAX_YAW, min(MAX_YAW, angle))
    pulse = angle_to_pulse(angle)

    c.set_pwm_servo_pulse(
        YAW_SERVO,
        pulse,
        500
    )

    CURRENT_YAW = angle

    time.sleep(0.5)


def servo_look_forward():
    """导航姿态：用于识别数字、转向、搜索目标"""
    _, c = _require_hardware()

    c.set_pwm_servo_pulse(
        PITCH_SERVO,
        LOOK_FORWARD_PULSE,
        500
    )

    time.sleep(0.1)


def servo_look_down():
    """检测姿态（最低头）：用于颜色识别"""
    _, c = _require_hardware()

    c.set_pwm_servo_pulse(
        PITCH_SERVO,
        LOOK_DOWN_PULSE,
        500
    )

    time.sleep(0.5)


def reset_servo():
    """水平回正 + 云台恢复导航俯角"""
    global CURRENT_YAW
    _, c = _require_hardware()

    c.set_pwm_servo_pulse(
        YAW_SERVO,
        YAW_CENTER,
        500
    )

    c.set_pwm_servo_pulse(
        PITCH_SERVO,
        LOOK_FORWARD_PULSE,
        500
    )

    CURRENT_YAW = 0

    time.sleep(0.5)


def init_servo():
    """比赛开始调用一次；初始为导航姿态"""
    _, c = _require_hardware()

    c.set_pwm_servo_pulse(
        PITCH_SERVO,
        LOOK_FORWARD_PULSE,
        500
    )

    c.set_pwm_servo_pulse(
        YAW_SERVO,
        YAW_CENTER,
        500
    )

    time.sleep(0.5)
