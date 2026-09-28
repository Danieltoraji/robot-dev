# -*- coding: utf-8 -*-
"""临时实验 4：两个"相机高度"证据谁说了算

证据 A（卷尺标定，.worktrees/stairs/archive/result/ruler_profile.json）：
  14 个卷尺刻度 3~60cm，含畸变物理模型 RMS 2.72px
  → h_eff = 32.73cm（光心到刻度平面）、θ = 55.11°（自竖直）、卷尺厚 1.20
  → 光心离地 = 33.93cm，观测档 pitch=1100 的有效俯角 57.34°（名义 36° + 偏移 21.3°）

证据 B（现场布局扫自标定）：48cm（重放）/ 51cm（历史记录）

本脚本把证据 A 的几何搬过来，固定 h=33.93cm，在安装偏移上细扫，
看九宫格照片能不能接受这个高度（格阵刚性 + 像素残差）。

用法：python tools/exp_ruler_vs_photos.py
"""

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import numpy as np

from core.ground_homography import GRID_CELL_CM, GroundHomography, grid_cell_center
from levels.nine_grid_shared import (
    CLUSTER_RADIUS_CM, NineGridShared, fit_rigid_2d, project_ground_to_pixel,
)
from tools.replay_ninegrid import (
    ReplayRobot, build_schedule, find_photos, sweep_and_collect, PHOTO_TRUTH,
)

H_RULER = 33.93     # 卷尺标定：光心离地
OFF_RULER = 21.3    # 卷尺标定 θ=57.34° @pitch1100 ⇒ 偏移 21.3°（与档位无关的常数）


def collect():
    photos = find_photos()
    robot = ReplayRobot(build_schedule(photos, "schedule"), verbose=False)
    level = NineGridShared(robot)
    obs = {}
    for pitch, head, d, px, cl in sweep_and_collect(robot, level, diag=False):
        obs.setdefault(d, []).append((int(pitch), int(head),
                                      np.asarray(px, float), bool(cl)))
    return obs


def evaluate(obs, cells, off, hcm, clean_only=False):
    ds = [d for d in sorted(cells) if d in obs]
    if len(ds) < 6:
        return None
    med, sel = {}, {}
    for d in ds:
        lst = [e for e in obs[d] if (not clean_only) or (not e[3])]
        if not lst:
            return None
        pts = np.array([GroundHomography.from_pose(
            (0.0, 0.0), hcm, p, head_pulse=hp,
            pitch_offset_deg=off).pixels_to_ground([px], head_pulse=hp)[0]
            for p, hp, px, _c in lst])
        if len(pts) > 1:
            dm = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
            n_in = (dm <= CLUSTER_RADIUS_CM).sum(axis=1)
            keep = dm[int(np.argmax(n_in))] <= CLUSTER_RADIUS_CM
        else:
            keep = np.ones(len(pts), dtype=bool)
        med[d] = np.median(pts[keep], axis=0)
        sel[d] = [k for k in np.where(keep)[0]]
    gs = np.array([grid_cell_center(cells[d]) for d in ds])
    ps = np.array([med[d] for d in ds])
    R, t, rms, per = fit_rigid_2d(gs, ps)
    pos = -R.T @ t
    fwd = R.T @ np.array([0.0, 1.0])
    th = float(np.arctan2(fwd[0], fwd[1]))
    n_ok = int(np.count_nonzero(per <= 0.35 * GRID_CELL_CM))
    px_res = []
    for d in ds:
        for k in sel[d]:
            p, head, px, _c = obs[d][k]
            q = project_ground_to_pixel(grid_cell_center(cells[d]), pos[0], pos[1],
                                        th, p, head, pitch_offset_deg=off,
                                        cam_height_cm=hcm)
            q = np.asarray(q, float).ravel()[:2]
            if np.all(np.isfinite(q)):
                px_res.append(float(np.linalg.norm(q - px)))
    px_res = np.array(px_res) if px_res else np.array([np.inf])
    return dict(rms=float(rms), res_max=float(per.max()), n_ok=n_ok,
                med_px=float(np.median(px_res)), max_px=float(px_res.max()),
                n=len(px_res))


def main():
    obs = collect()
    n_cl = sum(1 for v in obs.values() for e in v if e[3])
    n_all = sum(len(v) for v in obs.values())
    print(f"观测 {n_all} 条（其中裁切 {n_cl} 条 / 干净 {n_all - n_cl} 条）")
    print(f"证据A 卷尺标定：高度 {H_RULER}cm、安装偏移 {OFF_RULER}°"
          f"（pitch1100 有效俯角 {36 + OFF_RULER:.2f}°）\n")

    print("== 固定卷尺标定高度 33.93cm，细扫安装偏移（全观测 / 只看干净观测）==")
    print("   偏移     格阵RMS  最大残差 覆盖  中位像素  最大像素 | 干净集: 格阵RMS 中位像素")
    for o in np.arange(10.0, 31.01, 1.0):
        r = evaluate(obs, PHOTO_TRUTH, float(o), H_RULER)
        rc = evaluate(obs, PHOTO_TRUTH, float(o), H_RULER, clean_only=True)
        if r is None:
            continue
        s = f"  {o:+5.1f}°  {r['rms']:7.2f}  {r['res_max']:8.2f}  {r['n_ok']}/7 " \
            f"{r['med_px']:8.1f}  {r['max_px']:8.1f}"
        if rc:
            s += f" |  {rc['rms']:7.2f}  {rc['med_px']:8.1f}"
        print(s)

    print("\n== 固定高度、让偏移取最优：不同高度的最好成绩（干净观测） ==")
    print("   高度    最优偏移  格阵RMS  最大残差 覆盖  中位像素")
    for h in (30.0, 33.93, 36.0, 38.0, 40.0, 44.0, 48.0, 52.0, 56.0):
        best = None
        for o in np.arange(0.0, 41.0, 1.0):
            rc = evaluate(obs, PHOTO_TRUTH, float(o), h, clean_only=True)
            if rc and (best is None or rc["med_px"] < best[0]["med_px"]):
                best = (rc, o)
        if best is None:
            print(f"  {h:5.1f}   —")
            continue
        rc, o = best
        print(f"  {h:5.1f}  {o:+6.1f}°  {rc['rms']:7.2f}  {rc['res_max']:8.2f} "
              f"{rc['n_ok']}/7  {rc['med_px']:8.1f}")


if __name__ == "__main__":
    main()
