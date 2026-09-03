#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_real_trace.py —— 分析真机 trace 日志：定位成功率、动作前后位姿残差、路点到达误差等。

用法：
    python tools/analyze_real_trace.py C:\\path\\real_trace_xxx.txt [more files...]
"""

import argparse
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from motion_model import load_motion_model, forward_model


# =====================================================================
# 解析
# =====================================================================

POS_RE = re.compile(r"位置：\s*\[\s*([-\d.]+)\s+([-\d.]+)\s*\]")
ORI_RE = re.compile(r"朝向：\s*\[\s*([-\d.]+)\s+([-\d.]+)\s*\]")
ACTION_RE = re.compile(r"动作 #\d+:\s*(\w+)(?:\s*×\s*(\d+))?")


def parse_trace(path):
    """返回字典，包含定位统计、位姿序列、动作序列、路点到达误差。"""
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    attempts = 0
    successes = 0
    single_frame_failures = 0
    danger_events = 0
    terminal_failures = 0

    poses = []          # (x, y, theta_deg, line_no)
    actions = []        # (action_name, count, start_pose_index, line_no)
    waypoint_errors = []  # (wp, err_cm)

    last_pose_idx = None
    pending_action = None

    for idx, line in enumerate(lines, 1):
        if "定位尝试" in line:
            attempts += 1
        if "定位成功" in line:
            successes += 1
        if "PnP结果未通过" in line or "单帧定位失败" in line:
            single_frame_failures += 1
        if "机器人处于危险区" in line:
            danger_events += 1
        if ("定位重试超限" in line or "安全终止" in line or
                re.search(r"连续\s*\d+\s*次定位失败", line)):
            terminal_failures += 1

        m_pos = POS_RE.search(line)
        if m_pos and "定位成功" in line:
            x = float(m_pos.group(1))
            y = float(m_pos.group(2))
            m_ori = ORI_RE.search(line)
            if m_ori:
                ox = float(m_ori.group(1))
                oy = float(m_ori.group(2))
                theta = math.degrees(math.atan2(oy, ox))
            else:
                theta = None
            pose_idx = len(poses)
            poses.append((x, y, theta))
            last_pose_idx = pose_idx

        m_act = ACTION_RE.search(line)
        if m_act:
            name = m_act.group(1)
            count = int(m_act.group(2)) if m_act.group(2) else 1
            if name != "stand":
                actions.append((name, count, last_pose_idx))
                pending_action = (name, count, last_pose_idx)

        # 路点到达
        wp_match = re.search(r"===== 到达(?:中间路点|目标点|终点)\s*\[([-\d.]+)\s+([-\d.]+)\]", line)
        if wp_match and last_pose_idx is not None and poses[last_pose_idx][2] is not None:
            wx = float(wp_match.group(1))
            wy = float(wp_match.group(2))
            px, py, _ = poses[last_pose_idx]
            err = math.hypot(px - wx, py - wy)
            waypoint_errors.append(((wx, wy), err))

    # 去掉开头没有动作关联的 pose（如开局定位）
    # 计算每个动作的观测/预测残差：需要 action 前后都有 pose
    model = load_motion_model()
    residuals = []
    for i in range(len(actions)):
        name, count, start_idx = actions[i]
        if start_idx is None or start_idx + 1 >= len(poses):
            continue
        start_pose = poses[start_idx]
        end_pose = poses[start_idx + 1]
        if start_pose[2] is None or end_pose[2] is None:
            continue
        # 预测：从 start_pose 用名义模型执行动作
        try:
            pred, _ = forward_model(start_pose, [(name, count)], model)
        except Exception:
            continue
        pos_err = math.hypot(end_pose[0] - pred[0], end_pose[1] - pred[1])
        # 角度差（如果可计算）
        th_err = None
        if pred[2] is not None and end_pose[2] is not None:
            d = abs(pred[2] - end_pose[2]) % 360
            th_err = d if d <= 180 else 360 - d
        residuals.append({
            "action": f"{name}×{count}",
            "pos_err_cm": pos_err,
            "theta_err_deg": th_err,
        })

    return {
        "path": path,
        "attempts": attempts,
        "successes": successes,
        "single_frame_failures": single_frame_failures,
        "danger_events": danger_events,
        "terminal_failures": terminal_failures,
        "poses": poses,
        "actions": actions,
        "waypoint_errors": waypoint_errors,
        "residuals": residuals,
    }


def report(data):
    p = data["path"]
    print(f"\n===== {p} =====")
    attempts = data["attempts"]
    successes = data["successes"]
    print(f"定位尝试(内部扫描): {attempts}")
    print(f"定位成功(完成调用): {successes}")
    completed = successes + data['terminal_failures']
    if completed:
        print(f"定位调用成功率: {successes / completed * 100:.1f}% (完成 {completed} 次调用)")
    if successes:
        print(f"平均每次成功所需内部扫描: {attempts / successes:.2f}")
    print(f"单帧 PnP 失败次数: {data['single_frame_failures']}")
    print(f"危险区事件: {data['danger_events']}")
    print(f"终止/连续失败: {data['terminal_failures']}")

    residuals = data["residuals"]
    if residuals:
        pos_errs = [r["pos_err_cm"] for r in residuals]
        th_errs = [r["theta_err_deg"] for r in residuals if r["theta_err_deg"] is not None]
        print(f"\n动作间残差样本数: {len(residuals)}")
        print(f"位置残差 (名义模型 vs 下一次定位):")
        print(f"  均值 {np.mean(pos_errs):.2f} cm, 中位 {np.median(pos_errs):.2f} cm, "
              f"p90 {np.percentile(pos_errs, 90):.2f} cm, p95 {np.percentile(pos_errs, 95):.2f} cm, max {np.max(pos_errs):.2f} cm")
        if th_errs:
            print(f"航向残差:")
            print(f"  均值 {np.mean(th_errs):.2f}°, 中位 {np.median(th_errs):.2f}°, "
                  f"p90 {np.percentile(th_errs, 90):.2f}°, p95 {np.percentile(th_errs, 95):.2f}°, max {np.max(th_errs):.2f}°")

        # 按动作类型分
        by_action = {}
        for r in residuals:
            by_action.setdefault(r["action"], []).append(r["pos_err_cm"])
        print("\n按动作的位置残差中位数:")
        for act in sorted(by_action):
            arr = by_action[act]
            print(f"  {act:>20}: n={len(arr):3d} med={np.median(arr):.2f} cm p90={np.percentile(arr,90):.2f} cm")

    if data["waypoint_errors"]:
        errs = [e[1] for e in data["waypoint_errors"]]
        print(f"\n路点到达误差样本: {len(errs)}")
        print(f"  均值 {np.mean(errs):.2f} cm, 中位 {np.median(errs):.2f} cm, max {np.max(errs):.2f} cm")
        for wp, err in data["waypoint_errors"]:
            print(f"  wp {wp}: {err:.2f} cm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()
    for p in args.paths:
        data = parse_trace(p)
        report(data)


if __name__ == "__main__":
    main()
