#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calibrate_actions.py —— 在模拟器中生成动作模型 JSON（P1 的仿真版）。

真实环境使用时，把数据采集换成 AprilTag 定位得到的动作前后位姿，
并保留本脚本的拟合/输出结构。

用法：
    python tools/calibrate_actions.py [--repeats 1] [--noisy]

默认关闭噪声，生成标称模型；加 --noisy 则使用模拟器默认噪声并输出均值/协方差。
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import goodluck_sim as gs


def _apply_to_sim(sim, action_name, count=1):
    """在 sim 上执行动作，返回 (before, after)。"""
    before = (float(sim.pos[0]), float(sim.pos[1]), float(np.degrees(np.arctan2(sim.orientation[1], sim.orientation[0]))))
    for _ in range(count):
        if action_name == "go_forward":
            sim.apply_forward(gs.gl.FORWARD_CM, action_name)
        elif action_name == "go_forward_one_step":
            sim.apply_forward(gs.gl.FORWARD_ONE_STEP_CM, action_name)
        elif action_name == "back_one_step":
            sim.apply_back(gs.gl.BACK_FAST_CM, action_name)
        elif action_name == "left_move":
            sim.apply_left_move(gs.gl.LEFT_MOVE_CM, action_name)
        elif action_name == "right_move":
            sim.apply_right_move(gs.gl.RIGHT_MOVE_CM, action_name)
        elif action_name == "turn_left":
            sim.apply_turn(gs.gl.TURN_LEFT_DEG, action_name)
        elif action_name == "turn_right":
            sim.apply_turn(-gs.gl.TURN_RIGHT_DEG, action_name)
        else:
            raise KeyError(action_name)
    after = (float(sim.pos[0]), float(sim.pos[1]), float(np.degrees(np.arctan2(sim.orientation[1], sim.orientation[0]))))
    return before, after


def _local_delta(before, after):
    """用起始航向把全局位移转成局部位移。"""
    x0, y0, th0 = before
    x1, y1, th1 = after
    dx_g = x1 - x0
    dy_g = y1 - y0
    a = np.radians(th0)
    c, s = np.cos(a), np.sin(a)
    dx = c * dx_g + s * dy_g
    dy = -s * dx_g + c * dy_g
    return dx, dy, th1 - th0


def calibrate(action_names, repeats, noisy):
    if noisy:
        # 保留模拟器默认噪声
        pass
    else:
        gs.ACTION_ERROR_STD = 0.0
        gs.ACTION_ERROR_BIAS = 0.0
        gs.TURN_ERROR_STD = 0.0

    model = {}
    for name in action_names:
        samples = []  # (dx, dy, dtheta)
        for _ in range(repeats):
            sim = gs.SimState(np.array([0.0, 0.0]), np.array([1.0, 0.0]))
            before, after = _apply_to_sim(sim, name)
            samples.append(_local_delta(before, after))
        arr = np.array(samples)
        mean = arr.mean(axis=0)
        if arr.shape[0] >= 2:
            cov = np.cov(arr, rowvar=False)
        else:
            cov = np.zeros((3, 3))
        model[name] = {
            "dx": float(mean[0]),
            "dy": float(mean[1]),
            "dtheta": float(mean[2]),
            "cov": cov.tolist(),
        }
        print(f"{name:>22}: dx={mean[0]:6.3f} dy={mean[1]:6.3f} dtheta={mean[2]:7.3f}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--noisy", action="store_true")
    ap.add_argument("--out", default="result/motion_model.json")
    args = ap.parse_args()

    action_names = [
        "go_forward",
        "go_forward_one_step",
        "turn_left",
        "turn_right",
        "left_move",
        "right_move",
        "back_one_step",
    ]

    model = calibrate(action_names, args.repeats, args.noisy)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False, indent=2)
    print(f"\n已写入 {args.out}")


if __name__ == "__main__":
    main()
