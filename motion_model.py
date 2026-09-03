# -*- coding: utf-8 -*-
"""
motion_model.py —— 机器人动作前向模型 + 直线长度整数化

动作模型 JSON 格式：
{
  "go_forward": {
      "dx": 5.0, "dy": 0.0, "dtheta": 0.0,
      "cov": [[0.1, 0.0, 0.0], ...]
  },
  ...
}

约定：
- 每个动作的 (dx, dy, dtheta) 是“机体局部坐标系”下的增量；
- dtheta 单位为度；
- 应用动作时先把局部增量旋转到全局，再累加到当前位姿。
"""

import json
import math
import os

import numpy as np

# 默认动作模型路径
DEFAULT_MODEL_PATH = os.path.join("result", "motion_model.json")

# 动作表（用于区分转向/直行/横移）
TURN_ACTIONS = {"turn_left", "turn_right"}
STRAIGHT_ACTIONS = {"go_forward", "go_forward_one_step"}
SIDE_ACTIONS = {"left_move", "right_move", "back_one_step"}


# =====================================================================
# 直线长度整数化
# =====================================================================

def quantize_straight(L):
    """把目标直线长度 L 量化为可执行动作组合。

    返回 (actual_length, actions, error)
      actual_length: 实际执行总长
      actions: 动作名字典 {"go_forward": n5, "go_forward_one_step": n2}
      error: L - actual_length（允许 ±1cm 左右）
    """
    if L is None or L < 0:
        raise ValueError(f"L must be >= 0, got {L}")

    # 四舍五入到最接近的 0.5cm？这里按实数处理：先取最近整数？规则针对连续值也适用。
    # 为了保证 |error| 尽量小，对非整数也使用同一规则，误差可能达到 1.5cm。
    # 用 round 到最近的 0.5 也可以，但这里先按用户规则实现。
    # 为避免浮点误差，先转为“十分之一 cm”整数？
    # 简单实现：按 L 最近整数处理，但误差统计会略有偏差。
    L_int = int(round(L))
    n5 = L_int // 5
    r = L_int - n5 * 5

    if r == 0:
        n2 = 0
        error = L - (n5 * 5)
    elif r == 1:
        n2 = 0
        error = L - (n5 * 5)          # 少走 1cm
    elif r == 2:
        n2 = 1
        error = L - (n5 * 5 + 2)
    elif r == 3:
        n2 = 1
        error = L - (n5 * 5 + 2)      # 少走 1cm
    else:  # r == 4
        n5 = n5 + 1
        n2 = 0
        error = L - (n5 * 5)          # 多走 1cm

    # 去掉零动作
    actions = {}
    if n5 > 0:
        actions["go_forward"] = n5
    if n2 > 0:
        actions["go_forward_one_step"] = n2
    actual_length = n5 * 5 + n2 * 2
    return actual_length, actions, error


# =====================================================================
# 模型加载与位姿变换
# =====================================================================

def load_motion_model(path=DEFAULT_MODEL_PATH):
    """加载动作模型 JSON。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data


def _rot(theta_deg):
    """旋转矩阵（将局部向量转到全局）。"""
    a = math.radians(theta_deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def apply_action(state, action_model, action_name, count=1):
    """对当前位姿应用某个动作 count 次。

    state: (x, y, theta_deg)
    action_model: 动作模型 dict（含 dx/dy/dtheta）
    action_name: 字符串
    count: 正整数
    返回新的 (x, y, theta_deg)
    """
    x, y, theta = state
    if action_name not in action_model:
        raise KeyError(f"Unknown action: {action_name}")

    model = action_model[action_name]
    dx = float(model.get("dx", 0.0))
    dy = float(model.get("dy", 0.0))
    dtheta = float(model.get("dtheta", 0.0))

    for _ in range(count):
        # 局部增量转到全局
        g = _rot(theta) @ np.array([dx, dy])
        x += float(g[0])
        y += float(g[1])
        theta += dtheta
    return (x, y, theta)


def forward_model(state, action_sequence, action_model=None):
    """对动作序列做前向推演。

    action_sequence: [("go_forward", 3), ("turn_left", 1), ...]
    返回末端位姿和中间位姿列表。
    """
    if action_model is None:
        action_model = load_motion_model()
    states = [tuple(state)]
    cur = tuple(state)
    for name, count in action_sequence:
        cur = apply_action(cur, action_model, name, count)
        states.append(cur)
    return cur, states


# =====================================================================
# 便捷：从模拟器真值生成“标称动作模型”的工具函数
# =====================================================================

def local_delta_from_global(dx_global, dy_global, theta_deg):
    """把全局位移转回起始机体局部坐标。"""
    inv = _rot(-theta_deg)
    v = inv @ np.array([dx_global, dy_global])
    return float(v[0]), float(v[1])
