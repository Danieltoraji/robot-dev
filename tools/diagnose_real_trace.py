#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""diagnose_real_trace.py —— 深入分析真机定位误差来源。

输出：
- 每个动作类型的局部系平均残差 (dx, dy, dtheta)，判断是否有系统性偏差；
- 观测位移 / 命令位移比例；
- 残差与动作长度/次数的相关性；
- 按区域分组的残差；
- 离群点与定位重试的关系。
"""

import argparse
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from motion_model import load_motion_model, forward_model

POS_RE = re.compile(r"位置：\s*\[\s*([-\d.]+)\s+([-\d.]+)\s*\]")
ORI_RE = re.compile(r"朝向：\s*\[\s*([-\d.]+)\s+([-\d.]+)\s*\]")
ACTION_RE = re.compile(r"动作 #\d+:\s*(\w+)(?:\s*×\s*(\d+))?")
RETRY_RE = re.compile(r"定位尝试\s+(\d)/5")


def local_delta(before, after):
    x0, y0, th0 = before
    x1, y1, th1 = after
    dx_g = x1 - x0
    dy_g = y1 - y0
    a = math.radians(th0)
    dx = math.cos(a) * dx_g + math.sin(a) * dy_g
    dy = -math.sin(a) * dx_g + math.cos(a) * dy_g
    return dx, dy, th1 - th0


def parse(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    poses = []          # (x,y,theta)
    pose_line = []      # line no of pose for retry context
    actions = []        # (name,count,start_pose_idx)
    # 记录每个 pose 前最近的定位重试次数
    pose_retry = []     # retry level at success line

    last_pose_idx = None
    last_retry = 0
    for idx, line in enumerate(lines, 1):
        m_retry = RETRY_RE.search(line)
        if m_retry:
            last_retry = int(m_retry.group(1))
        m_pos = POS_RE.search(line)
        if m_pos and "定位成功" in line:
            x = float(m_pos.group(1)); y = float(m_pos.group(2))
            m_ori = ORI_RE.search(line)
            theta = None
            if m_ori:
                theta = math.degrees(math.atan2(float(m_ori.group(2)), float(m_ori.group(1))))
            poses.append((x, y, theta))
            pose_line.append(idx)
            pose_retry.append(last_retry)
            last_pose_idx = len(poses) - 1
        m_act = ACTION_RE.search(line)
        if m_act and m_act.group(1) != "stand":
            actions.append((m_act.group(1), int(m_act.group(2)) if m_act.group(2) else 1, last_pose_idx))
    return poses, actions, pose_retry


def signed_residuals(path):
    poses, actions, pose_retry = parse(path)
    model = load_motion_model()
    recs = []
    for i, (name, count, start_idx) in enumerate(actions):
        if start_idx is None or start_idx + 1 >= len(poses):
            continue
        s = poses[start_idx]
        e = poses[start_idx + 1]
        if s[2] is None or e[2] is None:
            continue
        pred, _ = forward_model(s, [(name, count)], model)
        # 预测在起始局部系下的 delta
        pred_local = local_delta(s, pred)
        obs_local = local_delta(s, e)
        # 残差 = 观测 - 预测
        recs.append({
            "action": f"{name}×{count}",
            "name": name,
            "count": count,
            "res_dx": obs_local[0] - pred_local[0],
            "res_dy": obs_local[1] - pred_local[1],
            "res_dth": obs_local[2] - pred_local[2],
            "obs_disp": math.hypot(obs_local[0], obs_local[1]),
            "pred_disp": math.hypot(pred_local[0], pred_local[1]),
            "x": s[0], "y": s[1],
            "retry": pose_retry[start_idx + 1],
            "pos_err": math.hypot(e[0] - pred[0], e[1] - pred[1]),
        })
    return recs


def report(path):
    recs = signed_residuals(path)
    print(f"\n===== {path} =====")
    print(f"样本数: {len(recs)}")

    # 1. 按动作类型看平均残差
    by = defaultdict(list)
    for r in recs:
        by[r["action"]].append(r)
    print("\n--- 按动作类型平均残差 (观测-预测) ---")
    print(f"{'action':>20} {'n':>3} {'dx(cm)':>8} {'dy(cm)':>8} {'dth(°)':>8} {'|pos|(cm)':>10}")
    for act in sorted(by):
        arr = by[act]
        print(f"{act:>20} {len(arr):>3} "
              f"{np.mean([r['res_dx'] for r in arr]):>8.2f} "
              f"{np.mean([r['res_dy'] for r in arr]):>8.2f} "
              f"{np.mean([r['res_dth'] for r in arr]):>8.2f} "
              f"{np.mean([r['pos_err'] for r in arr]):>10.2f}")

    # 2. 整体均值残差
    print("\n--- 整体均值残差 ---")
    print(f"dx = {np.mean([r['res_dx'] for r in recs]):.3f} cm")
    print(f"dy = {np.mean([r['res_dy'] for r in recs]):.3f} cm")
    print(f"dtheta = {np.mean([r['res_dth'] for r in recs]):.3f} °")

    # 3. 观测位移/命令位移
    print("\n--- 位移比例 (obs/pred) ---")
    straight = [r for r in recs if r["name"] in ("go_forward", "go_forward_one_step")]
    if straight:
        ratios = [r["obs_disp"] / r["pred_disp"] for r in straight if r["pred_disp"] > 0.1]
        print(f"直行动作平均比例: {np.mean(ratios):.3f} (1.0=正好)")
    for r in recs:
        if r["pred_disp"] > 0.1:
            ratio = r["obs_disp"] / r["pred_disp"]
            if ratio > 1.5 or ratio < 0.5:
                print(f"  离群位移: {r['action']} ratio={ratio:.2f} pos_err={r['pos_err']:.2f} at ({r['x']:.1f},{r['y']:.1f})")

    # 4. 按区域分组
    print("\n--- 按 x 区域的平均位置残差 ---")
    bins = defaultdict(list)
    for r in recs:
        bins[int(r["x"] // 10) * 10].append(r)
    for xb in sorted(bins):
        arr = bins[xb]
        print(f"x∈[{xb},{xb+10}] n={len(arr):2d} dx={np.mean([r['res_dx'] for r in arr]):+.2f} dy={np.mean([r['res_dy'] for r in arr]):+.2f} pos_err={np.mean([r['pos_err'] for r in arr]):.2f}")

    # 5. 定位重试与残差关系
    print("\n--- 定位重试级别与残差 ---")
    retry_groups = defaultdict(list)
    for r in recs:
        retry_groups[r["retry"]].append(r["pos_err"])
    for k in sorted(retry_groups):
        arr = retry_groups[k]
        print(f"retry={k}: n={len(arr):2d} mean_pos_err={np.mean(arr):.2f} cm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()
    for p in args.paths:
        report(p)


if __name__ == "__main__":
    main()
