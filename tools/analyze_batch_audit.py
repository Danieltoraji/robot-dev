#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""analyze_batch_audit.py —— 解析 batch_audit_*.json 并输出 Batch 审计报告。

用法：
    python tools/analyze_batch_audit.py [result/batch_audit_end.json ...]
"""

import json
import sys
from collections import Counter, defaultdict

import numpy as np


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def analyze_one(path):
    data = load(path)
    goal = data["goal"]
    audits = data["audits"]

    ok_count = 0
    crash_count = 0
    locates = []
    steps = []
    elapsed = []
    min_wall = []

    # batch
    batch_opp = 0          # candidate >= 2 (距离截短后仍可批量)
    batch_candidate_steps_sum = 0
    batch_actual_steps_sum = 0
    reject_counter = Counter()
    saved_locates = 0

    # turn
    turn_opp = 0           # proposed_times >= 2
    turn_actual_times_sum = 0
    turn_proposed_times_sum = 0

    # segment
    seg_locate_ratio = []
    seg_locates = []
    seg_actions = []

    for a in audits:
        if a is None:
            continue
        ok_count += 1  # audits only recorded for completed runs; failure may still have audit
        # Note: stats are not in audit file; we can't recover ok/crash from audit alone.
        # We'll compute only audit-derived metrics here; use v6 report for ok/crash.

        for ev in a.get("batch_events", []):
            candidate = ev.get("candidate", 0)
            actual = ev.get("actual", 0)
            reason = ev.get("reject_reason")
            # 只把“距离截短后仍 >=2 步”的视为真正的批量机会
            if candidate >= 2:
                batch_opp += 1
                batch_candidate_steps_sum += candidate
                batch_actual_steps_sum += actual
                if actual < candidate:
                    reject_counter[reason] += 1
                    saved_locates += max(0, candidate - actual)

        for ev in a.get("turn_events", []):
            pt = ev.get("proposed_times", 0)
            at = ev.get("actual_times", 0)
            if pt >= 2:
                turn_opp += 1
                turn_proposed_times_sum += pt
                turn_actual_times_sum += at

        for seg in a.get("segments", []):
            seg_locates.append(seg.get("locates", 0))
            seg_actions.append(seg.get("actions", 0))
            if seg.get("actions"):
                seg_locate_ratio.append(seg["locates"] / seg["actions"])

    print(f"\n===== Batch Audit Report: {goal} =====")
    print(f"审计 seed 数（含 audit 记录）: {len(audits)}")

    print("\n--- 直行批量 ---")
    if batch_opp:
        utilization = batch_actual_steps_sum / batch_candidate_steps_sum
        print(f"批量机会 (距离截短后 candidate>=2): {batch_opp}")
        print(f"候选步数总和: {batch_candidate_steps_sum}")
        print(f"实际步数总和: {batch_actual_steps_sum}")
        print(f"批量利用率: {utilization * 100:.1f}%")
        print(f"错失原因分布 (candidate>=2 但 actual<candidate): {dict(reject_counter)}")
        print(f"若全部按候选执行，预计少定位: ~{saved_locates} 次 (按 0.5s/次 ≈ {saved_locates * 0.5:.0f}s)")
    else:
        print("无批量机会 (candidate>=2)")

    print("\n--- 转向批量 ---")
    if turn_opp:
        turn_util = turn_actual_times_sum / turn_proposed_times_sum
        print(f"批量转向机会 (proposed>=2): {turn_opp}")
        print(f"提议总次数: {turn_proposed_times_sum}")
        print(f"实际总次数: {turn_actual_times_sum}")
        print(f"转向批量利用率: {turn_util * 100:.1f}%")
    else:
        print("无批量转向机会 (proposed>=2)")

    if seg_locates:
        print("\n--- 子目标定位/动作 ---")
        print(f"子目标平均定位: {np.mean(seg_locates):.2f}")
        print(f"子目标平均动作: {np.mean(seg_actions):.2f}")
        print(f"locates/actions 平均: {np.mean(seg_locate_ratio):.2f}")
        print(f"定位最多的子目标 top5: {sorted(seg_locates, reverse=True)[:5]}")

    print("\n" + "=" * 50)


def main():
    paths = sys.argv[1:] or [
        "result/batch_audit_end.json",
        "result/batch_audit_exit.json",
    ]
    for p in paths:
        try:
            analyze_one(p)
        except FileNotFoundError:
            print(f"未找到 {p}，跳过")


if __name__ == "__main__":
    main()
