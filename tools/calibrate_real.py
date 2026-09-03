#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calibrate_real.py —— 真机动作模型自标定（一次性引导脚本）

用法：
    python tools/calibrate_real.py [--repeats 20] [--out result/motion_model.json]

说明：
- 使用 RobotState + AprilTag locate_with_retry() 测量动作前后位姿。
- 对每个动作/count 连续采样；若机器人漂出安全区域或定位失败，脚本会暂停并提示你手动归位。
- 标定完成后输出 motion_model.json（含均值/协方差）。

安全：
- 第一次执行任何动作前，请先手动/点动确认动作方向和幅度。
- 周围保持空旷，准备好急停。
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot_core import RobotState, ctl
from levels import goodluck as gl


# =====================================================================
# 配置
# =====================================================================

# 头部中正：水平（pan）与俯仰（pitch）都回到 1500
PITCH_CENTER_PULSE = 1500

# 安全标定区域（世界坐标，cm）；超出后建议人工归位
SAFE_BOX = {
    "x_min": -30.0, "x_max": 130.0,
    "y_min": -30.0, "y_max": 130.0,
}

ACTION_LIST = [
    ("go_forward", 1),
    ("go_forward_one_step", 1),
    ("turn_left", 1),
    ("turn_right", 1),
    ("turn_left", 2),
    ("turn_right", 2),
    ("turn_left", 3),
    ("turn_right", 3),
    ("left_move", 1),
    ("right_move", 1),
    ("back_one_step", 1),
]


# =====================================================================
# 工具
# =====================================================================

def heading_from_orientation(ori):
    """由机体朝向单位向量得到角度（度）。"""
    return float(np.degrees(np.arctan2(ori[1], ori[0])))


def measure_pose(state):
    """定位一次；成功返回 (x,y,θ)，失败返回 None。"""
    if not state.locate_with_retry():
        return None
    if state.current_position is None or state.current_orientation is None:
        return None
    return (
        float(state.current_position[0]),
        float(state.current_position[1]),
        heading_from_orientation(state.current_orientation),
    )


def local_delta(before, after):
    """把世界系位移转成起始机体局部系位移。"""
    x0, y0, th0 = before
    x1, y1, th1 = after
    dx_g = x1 - x0
    dy_g = y1 - y0
    a = math.radians(th0)
    dx = math.cos(a) * dx_g + math.sin(a) * dy_g
    dy = -math.sin(a) * dx_g + math.cos(a) * dy_g
    return dx, dy, th1 - th0


def in_safe_box(pos):
    return (SAFE_BOX["x_min"] <= pos[0] <= SAFE_BOX["x_max"] and
            SAFE_BOX["y_min"] <= pos[1] <= SAFE_BOX["y_max"])


def wait_for_reposition(message):
    print("\n" + "=" * 60)
    print(message)
    print("请手动把机器人放到一个标签可见、动作前后都能定位的起始位姿。")
    input("就绪后按回车继续...")
    print("=" * 60)


# =====================================================================
# 主流程
# =====================================================================

def collect_samples(state, action_name, count, repeats):
    samples = []
    print(f"\n----- 开始标定: {action_name} × {count}（目标 {repeats} 次） -----")
    while len(samples) < repeats:
        before = measure_pose(state)
        if before is None:
            wait_for_reposition("当前无法定位。")
            continue
        if not in_safe_box(before):
            wait_for_reposition(f"当前位置 {before[:2]} 超出安全区域。")

        print(f"\n[{len(samples)+1}/{repeats}] before = ({before[0]:.2f}, {before[1]:.2f}, {before[2]:.2f}°)")
        print(f"即将执行: {action_name} × {count}")

        # 安全提示：第一次执行某动作时请确认
        state.act(action_name, times=count)
        time.sleep(0.3)

        after = measure_pose(state)
        if after is None:
            wait_for_reposition("动作后定位失败，本次数据作废。")
            continue
        if not in_safe_box(after):
            wait_for_reposition(f"动作后位置 {after[:2]} 超出安全区域，本次数据作废。")
            continue

        d = local_delta(before, after)
        samples.append(d)
        print(f"after  = ({after[0]:.2f}, {after[1]:.2f}, {after[2]:.2f}°)")
        print(f"Δ      = (dx={d[0]:.3f}, dy={d[1]:.3f}, dθ={d[2]:.3f})")

        # 如果已经漂出安全区域，提前提示归位，避免下一次动作后无法定位
        if not in_safe_box(after):
            wait_for_reposition("机器人已接近安全区域边缘，请移回安全起始位姿。")

    arr = np.array(samples)
    mean = arr.mean(axis=0)
    cov = np.cov(arr, rowvar=False) if arr.shape[0] >= 2 else np.zeros((3, 3))
    print(f"\n完成 {action_name} × {count}: mean=({mean[0]:.3f},{mean[1]:.3f},{mean[2]:.3f})")
    return samples, mean, cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--out", default="result/motion_model.json")
    ap.add_argument("--actions", nargs="*", default=None,
                    help="只标定指定动作，例如 --actions go_forward turn_left")
    args = ap.parse_args()

    print("正在初始化 RobotState...")
    state = RobotState(tag_poses=gl.tag_poses)
    # 确保头部“横竖都是正的”：水平回中 + 俯仰回中（不抬头/不低头）
    state.set_head(gl.HEAD_CENTER)
    if ctl is not None:
        ctl.set_pwm_servo_pulse(1, PITCH_CENTER_PULSE, 500)
        ctl.set_pwm_servo_pulse(2, PITCH_CENTER_PULSE, 500)
        time.sleep(0.3)

    wait_for_reposition("请把机器人放到标定起始位姿。")

    if args.actions is not None:
        action_list = [a for a in ACTION_LIST if a[0] in args.actions]
    else:
        action_list = ACTION_LIST

    model = {}
    count_table = {}
    for action_name, count in action_list:
        samples, mean, cov = collect_samples(state, action_name, count, args.repeats)
        entry = {
            "dx": float(mean[0]),
            "dy": float(mean[1]),
            "dtheta": float(mean[2]),
            "cov": cov.tolist(),
            "count": count,
            "samples": samples,
        }
        if count == 1:
            model[action_name] = {k: v for k, v in entry.items() if k != "samples"}
            # samples 单独存，避免 JSON 太大；如需要可保留
            model[action_name + "_samples"] = samples
        else:
            count_table.setdefault(action_name, {})[count] = {
                k: v for k, v in entry.items() if k != "samples"
            }
            count_table.setdefault(action_name + "_samples", {})[count] = samples

    output = {
        "go_forward": model.get("go_forward"),
        "go_forward_one_step": model.get("go_forward_one_step"),
        "turn_left": model.get("turn_left"),
        "turn_right": model.get("turn_right"),
        "left_move": model.get("left_move"),
        "right_move": model.get("right_move"),
        "back_one_step": model.get("back_one_step"),
        "count_table": count_table,
        "samples": {k: v for k, v in model.items() if k.endswith("_samples")},
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n标定完成，已写入 {args.out}")
    print("提示：count_table 用于检查 count>1 是否线性；运行时使用单步 dx/dy/dtheta/cov。")


if __name__ == "__main__":
    main()
