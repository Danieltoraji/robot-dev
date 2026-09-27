# -*- coding: utf-8 -*-
"""临时实验 3：像素域代价面——照片自己说相机高度是多少？

用现场真实照片（10 帧 / 53 条观测）与照片读图真值格阵，对每个 (安装偏移,
相机高度) 做：
  ① 反投影到机器人系 → 用**已知格阵归属**做刚体拟合（格阵可行性）；
  ② 用拟合出的位姿正投影回像素 → 中位/最大重投影残差（像素域代价，就是
     `calibrate_pixel_pose` 的门所用的量）。

这样得到的是"照片能容忍哪些高度"，与自标定搜到的 48cm、卷尺标定的 33.9cm
都能直接对比。

用法：python tools/exp_pixel_cost.py
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


def build_obs(pix_obs):
    obs = {}
    for pitch, head, d, px, cl in pix_obs:
        obs.setdefault(d, []).append((int(pitch), int(head),
                                      np.asarray(px, float), bool(cl)))
    return obs


def evaluate(obs, cells, off, hcm):
    """→ (rms_lattice_cm, res_max_cm, med_px, max_px, n_pts)"""
    ds = sorted(cells)
    if any(d not in obs for d in ds):
        return None
    # 每个数字先按 CLUSTER_RADIUS_CM 取最大一致簇（与 _aggregate 同规则）
    med, sel = {}, {}
    for d in ds:
        lst = obs[d]
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
        sel[d] = np.where(keep)[0]
    gs = np.array([grid_cell_center(cells[d]) for d in ds])
    ps = np.array([med[d] for d in ds])
    R, t, rms, per = fit_rigid_2d(gs, ps)
    res_max = float(per.max())
    # 位姿：机器人系原点在场地系的坐标 + 航向
    pos = -R.T @ t
    fwd = R.T @ np.array([0.0, 1.0])
    th = float(np.arctan2(fwd[0], fwd[1]))
    px_res = []
    for d in ds:
        for k in sel[d]:
            _p, head, px, _c = obs[d][k]
            q = project_ground_to_pixel(grid_cell_center(cells[d]),
                                        pos[0], pos[1], th, _p, head,
                                        pitch_offset_deg=off, cam_height_cm=hcm)
            q = np.asarray(q, float).ravel()[:2]
            if np.all(np.isfinite(q)):
                px_res.append(float(np.linalg.norm(q - px)))
    px_res = np.array(px_res) if px_res else np.array([np.inf])
    return (float(rms), res_max, float(np.median(px_res)),
            float(px_res.max()), len(px_res))


def main():
    photos = find_photos()
    robot = ReplayRobot(build_schedule(photos, "schedule"), verbose=False)
    level = NineGridShared(robot)
    obs = build_obs(sweep_and_collect(robot, level, diag=False))
    print(f"观测 {sum(len(v) for v in obs.values())} 条；真值格阵 {PHOTO_TRUTH}\n")

    print("== 像素域代价面：中位重投影残差 (px)，行=高度，列=安装偏移 ==")
    offs = list(range(0, 41, 2))
    print("  h\\off " + "".join(f"{o:6d}" for o in offs))
    table = {}
    for h in range(28, 71, 2):
        row = f"  {h:5d} "
        for o in offs:
            r = evaluate(obs, PHOTO_TRUTH, float(o), float(h))
            if r is None:
                row += "     -"
                continue
            table[(h, o)] = r
            row += f"{min(r[2], 999):6.0f}"
        print(row)

    print("\n== 每个高度上的最优 （偏移 + 中位残差 + 格阵 RMS） ==")
    for h in range(28, 71, 2):
        cand = [(table[(h, o)], o) for o in offs if (h, o) in table]
        if not cand:
            continue
        (rms, res_max, med_px, max_px, n), o = min(cand, key=lambda c: c[0][2])
        mark = ""
        if med_px <= 40.0:
            mark = "  ★过像素门(40px)"
        if abs(h - 34) <= 1:
            mark += "   ← 卷尺标定高度"
        if abs(h - 48) <= 1:
            mark += "   ← 现场重放自标定高度"
        print(f"  {h:3d}cm 偏移{o:+3d}°  中位 {med_px:7.1f}px  最大 {max_px:7.1f}px  "
              f"格阵RMS {rms:5.2f}cm  最大残差 {res_max:5.2f}cm  ({n}点){mark}")


if __name__ == "__main__":
    main()
