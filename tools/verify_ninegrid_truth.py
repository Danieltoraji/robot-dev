#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""verify_ninegrid_truth.py —— 用仿真真值评测 nine_grid 的**面板级识别**

为什么需要它
------------
本仓此前无法回答"识别准不准"这个问题在仿真里的版本：仿真器的"数字"曾是**纯黑
实心矩形**（`sim/nine_grid_sim.py` 旧 `_draw_panel` 的 `fillPoly((0,0,0))`），
数字链路根本没有输入信号。P0 把字形换成**现场照片同源的真字形**并给渲染加了
面板级真值（`_frame_truth`），本工具就是它的消费点。

它量的是三件事（全部以**渲染真值**为基准，不是以关卡自报为准）：
  1. **面板检出率**：真值里"可见"的面板，检测器这一帧检出了几个（配对用像素距离）；
  2. **数字正确率**：配对上的观测，`o.digit`（颜色主判，必要时含形状改判）是否等于
     真值数字。这就是"数字判据在仿真里准不准"；
  3. **颜色歧义/形状改判的干预率**：`o.ambiguous` / `shape_override` 的触发次数。

用法
----
    python tools/verify_ninegrid_truth.py --seed 3
    python tools/verify_ninegrid_truth.py --seed 3 --seeds 7,11 --out archive/result/p0_probe
    python tools/verify_ninegrid_truth.py --seed 3 --arbitrate   # 另附 SVM 对照（诊断用）

⚠️ 配对阈值 `--match-px` 默认 150 原生 px：真值给的是**色块外接框中心**，而检测器
未裁切时给"对角线交点"、裁切时给"凸包质心"，两者本就不完全同一，故阈值不能太小。
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import io
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import levels.nine_grid as NG  # noqa: E402
import sim.nine_grid_sim as SIM  # noqa: E402

# 判定"真值里这个面板算可见"的门：可见面积占比 / 被遮挡比例
VISIBLE_MIN_FRAC = 0.05
OCCLUDED_MAX = 0.60


class TruthProbeViewer:
    """挂在仿真的 viewer 钩子上，逐帧记录（渲染真值, 检测观测）

    为什么用 viewer 钩子而不是包 `capture_frame`：`robot.viewer` 是仿真**现成**的
    接缝（`run_simulation(viewer=...)` 已支持），且 `attach(robot, level)` 在
    `run_level()` 之前调用 ⇒ 这里一定拿得到关卡的 detector。包 `capture_frame`
    则要在 robot/level 两个构造之间插桩，曾因初始化顺序踩坑。

    ⚠️ 检测在**关卡之外**独立跑一遍（本工具不读关卡内部的观测），所以
    "记录到的判定"不会反过来污染关卡决策。
    """

    def __init__(self, arbitrate=False, shape=False, limit=0):
        self.arbitrate = bool(arbitrate)
        self.shape = bool(shape)
        self.limit = int(limit)
        self.records = []
        self.robot = None
        self.level = None

    def attach(self, robot, level):
        self.robot = robot
        self.level = level

    def on_action(self, robot, name, times):
        pass

    def close(self):
        pass

    def on_frame(self, robot, frame):
        if self.level is None or self.limit and len(self.records) >= self.limit:
            return
        truth = list(getattr(robot, "_last_frame_truth", []) or [])
        obs = self.level.detector.detect_panels(
            frame, arbitrate=self.arbitrate, drop_border=False, shape=self.shape)
        self.records.append({"truth": truth, "obs": obs})


def capture_frames(seed, layout=None, arbitrate=False, shape=False, limit=0):
    """跑一局（静默），返回 (逐帧记录, 关卡结果)"""
    layout = SIM.SIM_LAYOUT if layout is None else layout
    probe = TruthProbeViewer(arbitrate=arbitrate, shape=shape, limit=limit)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run = SIM.run_simulation(layout=layout, seed=seed, quiet=True,
                                 viewer=probe)
    return probe.records, run


def pair_frame(truth, obs, match_px):
    """真值面板 ↔ 检测观测 的贪心配对（按距离最近，且**同色**才允许配）

    返回 [(truth_item, obs|None, dist_px|None), ...]
    """
    used = set()
    out = []
    for t in truth:
        best, best_d = None, None
        for i, o in enumerate(obs):
            if i in used:
                continue
            if NG_ok_color(o) != t["digit"]:
                continue
            d = float(np.hypot(o.center_px[0] - t["cx"], o.center_px[1] - t["cy"]))
            if best_d is None or d < best_d:
                best, best_d = i, d
        if best is not None and best_d <= match_px:
            used.add(best)
            out.append((t, obs[best], best_d))
        else:
            out.append((t, None, None))
    return out


def NG_ok_color(o):
    """观测的"颜色主判数字"（= o.color_id；形状改判另算，见 shape_override）"""
    return int(o.color_id)


def report(records, match_px, tag):
    n_truth = n_vis = n_hit = n_digit_ok = 0
    n_amb = n_shape = n_shape_changed = 0
    miss_by_digit = collections.Counter()
    wrong_by_digit = collections.Counter()
    err = []
    for rec in records:
        for t, o, d in pair_frame(rec["truth"], rec["obs"], match_px):
            n_truth += 1
            visible = (t["visible_frac"] >= VISIBLE_MIN_FRAC
                       and t["occluded_frac"] <= OCCLUDED_MAX)
            if not visible:
                continue
            n_vis += 1
            if o is None:
                miss_by_digit[t["digit"]] += 1
                continue
            n_hit += 1
            err.append(d)
            if o.ambiguous:
                n_amb += 1
            if getattr(o, "shape_override", False):
                n_shape += 1
                if o.shape_digit != o.color_id:
                    n_shape_changed += 1
            if int(o.digit) == int(t["digit"]):
                n_digit_ok += 1
            else:
                wrong_by_digit[(t["digit"], int(o.digit))] += 1
    print("=" * 70)
    print("面板级识别评测（%s）" % tag)
    print("=" * 70)
    print("  真值面板出现次数（全部帧合计）: %d" % n_truth)
    print("  其中【判为可见】: %d（可见面积≥%.2f 且 遮挡≤%.2f）"
          % (n_vis, VISIBLE_MIN_FRAC, OCCLUDED_MAX))
    if n_vis:
        print("  检出率  : %d/%d = %.1f%%" % (n_hit, n_vis, 100.0 * n_hit / n_vis))
        print("  数字正确: %d/%d = %.1f%%"
              % (n_digit_ok, n_hit, 100.0 * n_digit_ok / max(1, n_hit)))
    if err:
        e = np.asarray(err)
        print("  配对像素误差: 中位 %.0f  p90 %.0f  max %.0f"
              % (float(np.median(e)), float(np.percentile(e, 90)), float(e.max())))
    print("  颜色歧义观测: %d｜形状改判触发: %d（其中真的改了颜色主判: %d）"
          % (n_amb, n_shape, n_shape_changed))
    if miss_by_digit:
        print("  漏检按数字: %s" % dict(sorted(miss_by_digit.items())))
    if wrong_by_digit:
        print("  数字判错 (真值→判定): %s" % dict(sorted(wrong_by_digit.items())))
    return {"truth": n_truth, "visible": n_vis, "hit": n_hit,
            "digit_ok": n_digit_ok, "ambiguous": n_amb,
            "shape_override": n_shape}


def main(argv=None):
    ap = argparse.ArgumentParser(description="用仿真真值评测面板级识别")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--seeds", default="", help="逗号分隔的多种子（覆盖 --seed）")
    ap.add_argument("--match-px", type=float, default=150.0)
    ap.add_argument("--arbitrate", action="store_true",
                    help="同时跑 SVM 数字仲裁（诊断用，生产路径不用它）")
    ap.add_argument("--shape", action="store_true",
                    help="同时跑形状模板仲裁（颜色歧义时可能改判）")
    args = ap.parse_args(argv)

    seeds = ([int(s) for s in args.seeds.split(",") if s.strip()]
             if args.seeds else [args.seed])
    total = collections.Counter()
    for sd in seeds:
        records, run = capture_frames(sd, arbitrate=args.arbitrate,
                                      shape=args.shape)
        st = report(records, args.match_px,
                    "seed=%d  帧数=%d  关卡到达=%s"
                    % (sd, len(records), run.stats.get("ok_all")))
        for k, v in st.items():
            total[k] += v
    if len(seeds) > 1:
        print("\n" + "=" * 70)
        print("合计（%d 个种子）" % len(seeds))
        print("=" * 70)
        if total["visible"]:
            print("  检出率  : %d/%d = %.1f%%"
                  % (total["hit"], total["visible"],
                     100.0 * total["hit"] / total["visible"]))
            print("  数字正确: %d/%d = %.1f%%"
                  % (total["digit_ok"], total["hit"],
                     100.0 * total["digit_ok"] / max(1, total["hit"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
