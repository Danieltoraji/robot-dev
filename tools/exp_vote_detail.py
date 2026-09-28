# -*- coding: utf-8 -*-
"""临时实验 7：投票明细——自标定到底在哪些 (偏移, 高度) 上投票？

用与 `_fit_grid._aggregate` 相同的聚合规则（最大一致簇 + 簇内干净优先），
对每个 (偏移, 高度) 组合调用**真实的** assign_grid_cells，记录它解出的布局，
再按高度汇总。回答：
  · 下界 36 那一档（以及 33.9 那一档）能不能投票？
  · 票是不是全挤在 45~55？

用法：python tools/exp_vote_detail.py
"""

import os
import sys
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import numpy as np

from core.ground_homography import GroundHomography, grid_cell_center
from levels.nine_grid_shared import (
    CLUSTER_RADIUS_CM, GRID_FIT_CLIPPED_WEIGHT, NineGridShared, assign_grid_cells,
)
from tools.replay_ninegrid import (
    ReplayRobot, build_schedule, find_photos, sweep_and_collect, PHOTO_TRUTH,
)


def make_aggregate(obs_by_digit):
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
                n_in = (dm <= CLUSTER_RADIUS_CM).sum(axis=1)
                keep = dm[int(np.argmax(n_in))] <= CLUSTER_RADIUS_CM
            else:
                keep = np.ones(len(pts), dtype=bool)
            n_clean = int(np.count_nonzero(keep & ~flags))
            use = pts[keep & ~flags] if n_clean >= 1 else pts[keep]
            med[d] = np.median(use, axis=0)
            if n_clean >= 1:
                clean.add(d)
                wts[d] = 1.0
            else:
                wts[d] = GRID_FIT_CLIPPED_WEIGHT
        return med, clean, wts
    return _agg


def sweep(obs_by_digit, hmin, hstep, hmax=70.0):
    agg = make_aggregate(obs_by_digit)
    votes = defaultdict(list)          # (off, h) -> 解出的 cells
    for off in np.arange(0.0, 40.0 + 1e-9, 2.5):
        for h in np.arange(hmin, hmax + 1e-9, hstep):
            med, clean, wts = agg(float(off), float(h))
            if len(med) < 7:
                continue
            cells, _info, ranked = assign_grid_cells(med, weights=wts, clean=clean)
            if cells is None or len(cells) != 7 or not ranked:
                continue
            votes[(float(off), float(h))] = cells
    return votes


def report(votes, tag):
    print(f"\n== {tag} ==")
    n_win = sum(1 for c in votes.values() if c == PHOTO_TRUTH)
    print(f"  成功组合 {len(votes)} 个；其中解出 = 照片真值 的 {n_win} 个")
    by_h = defaultdict(lambda: [0, 0])
    for (off, h), cells in votes.items():
        by_h[h][0] += 1
        if cells == PHOTO_TRUTH:
            by_h[h][1] += 1
    print("   高度   成功组合  与真值一致")
    for h in sorted(by_h):
        n, m = by_h[h]
        print(f"  {h:5.1f}   {n:8d}  {m:8d}" + ("   ← 真值(卷尺)" if abs(h - 33.9) < .1 else ""))


def main():
    photos = find_photos()
    robot = ReplayRobot(build_schedule(photos, "schedule"), verbose=False)
    level = NineGridShared(robot)
    obs_by_digit = {}
    for pitch, head, d, px, cl in sweep_and_collect(robot, level, diag=False):
        obs_by_digit.setdefault(d, []).append((int(pitch), int(head),
                                               np.asarray(px, float), bool(cl)))
    report(sweep(obs_by_digit, 36.0, 5.0), "现状网格 高度[36,70] 步5")
    report(sweep(obs_by_digit, 33.9, 5.0), "把真值 33.9 塞进网格（步5）")
    report(sweep(obs_by_digit, 34.0, 1.0), "细网格 高度[34,70] 步1")


if __name__ == "__main__":
    main()
