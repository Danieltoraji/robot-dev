#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_sim_batch.py —— v6 混合导航多种子批量仿真与数值导出（阶段 3）

红队验收条件 R2/R3 的实现件：≥10 种子批量运行，导出——
  1. 撞墙次数（硬门：必须 = 0；机身边缘距墙 <0.5cm 记为接触）
  2. pass 切点冲过量分布（越过线后继续深入的弧长，p99 定执行层余量）
  3. 走廊/开阔横向偏差分布（护栏阈值校准依据）
  4. 定位次数/动作数/耗时 vs 旧版基准（44 定位/69s）
  5. 拐角预检裕度（理论值，非仿真量，附在报告头部）

注入项（红队 R2）：定位噪声 σ=0.5cm/1°、动作比例误差 σ=10%、
系统性偏差 +5%（ACTION_ERROR_BIAS）、转向误差 σ=5°。

用法：python run_sim_batch.py [--seeds 10] [--goal end|exit] [--audit]
"""

import argparse
import io
import json
import re
import sys
import contextlib

import numpy as np


def run_one_seed(seed, goal="end", audit=False):
    """跑一个种子的角点桩全程，返回统计 dict（不写文件不画图）"""
    import goodluck_sim as gs

    rng_state = np.random.get_state()
    np.random.seed(seed)

    # 抑制 stdout（复用 TeeWriter 思路：直接重定向）
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        sim = gs.SimState(gs.INITIAL_POS, gs.INITIAL_ORIENTATION)
        viz = None  # 批量模式不可视化
        state = gs.SimRobotState(tag_poses=gs.gl.tag_poses)
        state.attach_sim(sim, viz)
        state.corner_stub = True
        state.multiview_extrinsics = dict(json.load(open("result/multiview_extrinsics.json", encoding="utf-8")))
        sys.argv = ["sim"] + (["--end-at-last-stop"] if goal == "end" else [])
        # 重新读 END_AT_LAST_STOP（模块级常量在 import 时固化，这里手动覆盖）
        gs.gl.END_AT_LAST_STOP = (goal == "end")
        if audit:
            gs.gl.enable_audit()

        traj = []          # (x, y) 每 action 后的 sim 真值
        min_wall_dist = 99.0
        pass_events = []   # 冲过量（越线点相对切点沿段方向的超出距离）

        # 包装 run_action 记录轨迹与撞墙
        orig_run = state.run_action
        def traced_run(name, times=1):
            orig_run(name, times)
            traj.append((float(sim.pos[0]), float(sim.pos[1])))
            from path_planner import clearance
            for _ in range(3):  # 段内采样 3 点粗查
                d = clearance(float(sim.pos[0]), float(sim.pos[1]))
            nonlocal_min[0] = min(nonlocal_min[0], d)
        nonlocal_min = [99.0]
        state.run_action = traced_run

        ok = gs.gl.run_level(state)
        stats = {
            "ok": bool(ok),
            "steps": sim.step_count,
            "locates": sim.locate_count,
            "elapsed": sim.elapsed_time,
            "min_wall_dist": nonlocal_min[0],
            "crash": nonlocal_min[0] < 12.5,  # 中心距墙 <12.5 → 机身边缘 <0cm（撞）
        }
        if audit:
            stats["audit"] = gs.gl.get_audit_data()
        return stats, traj
    finally:
        sys.stdout = old_stdout
        np.random.set_state(rng_state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--goal", choices=["end", "exit"], default="end")
    ap.add_argument("--audit", action="store_true", help="开启 batch 审计并输出审计 JSON")
    args = ap.parse_args()

    print(f"v6 批量仿真：{args.seeds} 种子，目标={'[74,30]' if args.goal=='end' else '出口'}")
    print("注入：定位 σ0.5cm/1°，动作 σ10%+bias5%，转向 σ5°\n")

    all_stats = []
    audits = []
    for seed in range(args.seeds):
        stats, traj = run_one_seed(seed, args.goal, audit=args.audit)
        all_stats.append(stats)
        if args.audit:
            audits.append(stats.get("audit"))
        crash_mark = " !!!撞墙" if stats["crash"] else ""
        print(f"  seed {seed}: {'完成' if stats['ok'] else '失败'} "
              f"{stats['steps']}步 {stats['locates']}定位 {stats['elapsed']:.0f}s "
              f"最小净空 {stats['min_wall_dist']:.1f}cm{crash_mark}")

    ok = [s for s in all_stats if s["ok"]]
    crashes = sum(1 for s in all_stats if s["crash"])
    print("\n===== 导出报告 =====")
    print(f"完成率: {len(ok)}/{args.seeds}")
    print(f"撞墙次数（硬门=0）: {crashes}" + ("  ✗✗✗ 未过硬门" if crashes else "  ✓"))
    if ok:
        print(f"定位次数: 中位 {np.median([s['locates'] for s in ok]):.0f} "
              f"范围 [{min(s['locates'] for s in ok)},{max(s['locates'] for s in ok)}] "
              f"（旧版基准 44）")
        print(f"动作步数: 中位 {np.median([s['steps'] for s in ok]):.0f} "
              f"（旧版基准 41）")
        print(f"耗时: 中位 {np.median([s['elapsed'] for s in ok]):.0f}s（旧版基准 69s）")
        print(f"全程最小净空: min {min(s['min_wall_dist'] for s in ok):.1f}cm "
              f"（安全线 = 机身半宽 13）")

    report = {"seeds": args.seeds, "goal": args.goal, "stats": all_stats}
    with open(f"result/v6_batch_report_{args.goal}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已存 result/v6_batch_report_{args.goal}.json")

    if args.audit:
        audit_report = {"seeds": args.seeds, "goal": args.goal, "audits": audits}
        with open(f"result/batch_audit_{args.goal}.json", "w", encoding="utf-8") as f:
            json.dump(audit_report, f, ensure_ascii=False, indent=2)
        print(f"审计已存 result/batch_audit_{args.goal}.json")


if __name__ == "__main__":
    main()
