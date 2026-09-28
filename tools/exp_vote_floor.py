# -*- coding: utf-8 -*-
"""临时实验 8：为什么新网格的票数掉到 3？（投票门 vs 网格变小的相互作用）

`LAYOUT_SCAN_VOTE_MIN_COMBOS = 10` 是**绝对票数**下限，它隐含假设"成功组合
应该有几十个"——那是 120 组（17 偏移 × 8 高度）时代的事。网格改成"实测值 ±
几档"（17 × 4 = 68 组）后，成功组合数按比例掉到 3，旧的绝对门就把**本来能过的
布局**判成了"投票不足"。

本脚本逐组合打印 (偏移, 高度) → 是否 7/7、刚体 RMS，并把票数按高度汇总，
给出新门该取多少的依据。

用法：python tools/exp_vote_floor.py
"""

import os
import sys
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import numpy as np

import levels.nine_grid_shared as ngs
from core.ground_homography import GroundHomography, grid_cell_center
from levels.nine_grid_shared import NineGridShared, assign_grid_cells
from tools.replay_ninegrid import (
    ReplayRobot, build_schedule, find_photos, sweep_and_collect, PHOTO_TRUTH,
)


def make_aggregate(obs_by_digit, pitch_ref):
    """与 _fit_grid._aggregate 同规则；额外返回"主档"点的刚体残差"""
    def _agg(off, hcm):
        med, clean, wts = {}, set(), {}
        for d, lst in obs_by_digit.items():
            pts = np.array([GroundHomography.from_pose(
                (0.0, 0.0), hcm, pitch, head_pulse=pulse,
                pitch_offset_deg=off).pixels_to_ground([px], head_pulse=pulse)[0]
                for pitch, pulse, px, _cl in lst])
            flags = np.array([cl for _p, _h, _px, cl in lst])
            if len(pts) > 1:
                dm = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
                n_in = (dm <= ngs.CLUSTER_RADIUS_CM).sum(axis=1)
                keep = dm[int(np.argmax(n_in))] <= ngs.CLUSTER_RADIUS_CM
            else:
                keep = np.ones(len(pts), dtype=bool)
            n_clean = int(np.count_nonzero(keep & ~flags))
            use = pts[keep & ~flags] if n_clean >= 1 else pts[keep]
            med[d] = np.median(use, axis=0)
            if n_clean >= 1:
                clean.add(d)
                wts[d] = 1.0
            else:
                wts[d] = ngs.GRID_FIT_CLIPPED_WEIGHT
        return med, clean, wts
    return _agg


def main():
    photos = find_photos()
    robot = ReplayRobot(build_schedule(photos, "schedule"), verbose=False)
    level = NineGridShared(robot)
    obs_by_digit = {}
    for pitch, head, d, px, cl in sweep_and_collect(robot, level, diag=False):
        obs_by_digit.setdefault(d, []).append((int(pitch), int(head),
                                               np.asarray(px, float), bool(cl)))
    agg = make_aggregate(obs_by_digit, 1200)

    measured = ngs.measured_cam_height_cm()
    grid = [measured + dh for dh in ngs.LAYOUT_SCAN_HEIGHT_OFFSETS_CM]
    print(f"实测高度 {measured}cm；网格高度 {[round(h,1) for h in grid]}；"
          f"偏移采样 {len(ngs.LAYOUT_SCAN_OFFSET_SAMPLES_DEG)} 个；"
          f"总组合 {len(grid) * len(ngs.LAYOUT_SCAN_OFFSET_SAMPLES_DEG)}")
    print(f"旧门：绝对票数 ≥{ngs.LAYOUT_SCAN_VOTE_MIN_COMBOS}、"
          f"占比 ≥{ngs.LAYOUT_SCAN_VOTE_MIN_FRAC}\n")

    print("== 逐 (偏移, 高度) 结果（✓=7/7 且 = 照片真值）==")
    print("   高度   通过的偏移")
    tally = defaultdict(list)
    n_total = n_hit = 0
    for h in grid:
        oks = []
        for off in ngs.LAYOUT_SCAN_OFFSET_SAMPLES_DEG:
            n_total += 1
            med, clean, wts = agg(float(off), float(h))
            if len(med) < 7:
                continue
            cells, _i, ranked = assign_grid_cells(med, weights=wts, clean=clean)
            if cells is None or len(cells) != 7 or not ranked:
                continue
            n_hit += 1
            tally[h].append((float(off), cells == PHOTO_TRUTH,
                             ranked[0]["rms_clean"]))
            if cells == PHOTO_TRUTH:
                oks.append(f"{off:.1f}")
        print(f"  {h:6.1f}   {', '.join(oks) if oks else '（无）'}")
    print(f"\n  成功组合 {n_hit}/{n_total}；全部 = 照片真值: "
          f"{all(ok for v in tally.values() for _o, ok, _r in v)}")
    print("\n== 各高度的成功组合数与干净子集刚体 RMS ==")
    for h in sorted(tally):
        rows = tally[h]
        rms = min(r for _o, _ok, r in rows)
        print(f"  {h:6.1f}cm: {len(rows):2d} 个组合，最好 RMS {rms:.2f}cm，"
              f"偏移 {[round(o, 1) for o, _ok, _r in rows]}")


if __name__ == "__main__":
    main()
