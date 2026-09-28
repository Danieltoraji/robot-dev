# -*- coding: utf-8 -*-
"""临时实验：布局扫自标定的 (安装偏移, 相机高度) 可解性地图

问题（2026-09-27 讨论）：现场实测相机光心离地 33.9cm，而
`LAYOUT_SCAN_HEIGHT_MIN_CM = 36.0`。那个"扫不到真值"到底有多严重？

做法（全部用现场真实照片的观测，不跑 FSM）：
  1. ReplayRobot 取 10 帧 → 53 条观测（与 tools/replay_ninegrid.py --diag 同源）；
  2. 在 (偏移, 高度) 的细网格上重算"机器人系点 → 已知真值格心"的刚体拟合，
     按布局扫自己的两道硬门（assign_grid_cells 的覆盖门 0.35 格距 + 刚体 RMS 门
     8cm）判定该组合是否"可行"；
  3. 打印可行域地图、每格距/偏移的残差，以及真值附近的行为。

用法：python tools/exp_height_degeneracy.py
"""

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import numpy as np

from core.camera_config import HEAD_CENTER, SERVO_DEG_PER_US
from core.ground_homography import GRID_CELL_CM, GroundHomography, grid_cell_center
from levels.nine_grid_shared import (
    CLUSTER_RADIUS_CM, GRID_FIT_RMS_MAX_CM, NineGridShared, PITCH_DOWN, PITCH_NAV,
    fit_rigid_2d,
)
from tools.replay_ninegrid import (
    PHOTO_TRUTH, ReplayRobot, build_schedule, find_photos, sweep_and_collect,
)


def agg_points(obs, off, hcm):
    """复刻 _fit_grid._aggregate 的前半段：逐数字 → 机器人系点 → 中位点

    返回 (med, wts)：簇内"干净优先"（同 GRID_FIT_CLIPPED_WEIGHT 规则）。
    """
    med, wts = {}, {}
    for d, lst in obs.items():
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
        wts[d] = 1.0 if n_clean >= 1 else 0.3
    return med, wts


def evaluate(obs, truth, off, hcm, cache):
    """→ (rms_cm, res_max_cm, n_cells_ok, ok)；不合规则返回 None"""
    key = (float(off), float(hcm))
    got = cache.get(key)
    if got is None:
        got = agg_points(obs, off, hcm)
        cache[key] = got
    med, wts = got
    ds = sorted(truth)
    if any(d not in med for d in ds):
        return None
    gs = np.array([grid_cell_center(truth[d]) for d in ds])
    ps = np.array([med[d] for d in ds])
    ws = np.array([wts[d] for d in ds])
    R, t, rms, per = fit_rigid_2d(gs, ps, ws)
    res_max = float(np.max(per))
    # 覆盖门：每个点离它被指派的格心 ≤ 0.35 格距
    ok_cov = res_max <= 0.35 * GRID_CELL_CM
    ok_rms = rms <= GRID_FIT_RMS_MAX_CM
    n_ok = int(np.count_nonzero(per <= 0.35 * GRID_CELL_CM))
    return (float(rms), res_max, n_ok, ok_cov and ok_rms)


def main():
    photos = find_photos()
    if not photos:
        print("夹具缺失")
        return 1
    robot = ReplayRobot(build_schedule(photos, "schedule"), verbose=False)
    level = NineGridShared(robot)
    pix_obs = sweep_and_collect(robot, level, diag=False)
    obs = {}
    for pitch, head, d, px, cl in pix_obs:
        obs.setdefault(d, []).append((int(pitch), int(head), px, bool(cl)))
    n_clean = sum(1 for v in obs.values() for e in v if not e[3])
    print(f"观测 {len(pix_obs)} 条；未裁切 {n_clean} 条，数字 {sorted(obs)}")
    print(f"照片读图真值 cells = {PHOTO_TRUTH}")
    print(f"格距 {GRID_CELL_CM:.2f}cm；覆盖门 0.35 格距 = "
          f"{0.35 * GRID_CELL_CM:.1f}cm；刚体 RMS 门 {GRID_FIT_RMS_MAX_CM}cm\n")

    cache = {}
    print("== (偏移, 高度) 可行域地图：数字 = 7 个面板里落在覆盖门内的个数 ==")
    print("   行=高度 (cm)，列=安装偏移 (度)；'.' = 该组合 7/7 且 RMS≤8cm\n")
    offs = [float(o) for o in range(0, 41, 2)]
    hcms = [float(h) for h in range(30, 71, 2)]
    header = "  h\\off " + "".join(f"{o:5.0f}" for o in offs)
    print(header)
    best = []
    for h in hcms:
        row = f"  {h:5.0f} "
        for o in offs:
            r = evaluate(obs, PHOTO_TRUTH, o, h, cache)
            if r is None:
                row += "    -"
                continue
            rms, res_max, n_ok, ok = r
            row += "    *" if ok else f"{n_ok:5d}"
            if ok:
                best.append((rms, o, h, res_max))
        print(row)
    best.sort()
    print("\n== 可行组合按 RMS 排序（前 20） ==")
    for rms, o, h, res_max in best[:20]:
        print(f"  偏移 {o:+5.1f}°  高度 {h:5.1f}cm   RMS {rms:5.2f}cm   "
              f"最大残差 {res_max:5.2f}cm")

    print("\n== 几个关键组合的细节 ==")
    for o, h, tag in [(0.0, 36.0, "网格下界角点"),
                      (5.0, 36.0, "网格下界附近"),
                      (10.0, 36.0, "网格下界附近(偏移10)"),
                      (25.0, 33.9, "真值(仅单档等效偏移)"),
                      (12.8, 33.9, "真值高度+网格最佳偏移"),
                      (30.0, 48.0, "现场重放实际标定值"),
                      (18.5, 56.0, "camera_config 缺省")]:
        r = evaluate(obs, PHOTO_TRUTH, o, h, cache)
        if r is None:
            print(f"  {tag:26s} 偏移{o:+5.1f}° 高度{h:5.1f}cm → 观测不足")
            continue
        rms, res_max, n_ok, ok = r
        print(f"  {tag:26s} 偏移{o:+5.1f}° 高度{h:5.1f}cm → "
              f"RMS {rms:5.2f}cm 最大残差 {res_max:5.2f}cm "
              f"覆盖 {n_ok}/7 {'★可行' if ok else '✗被门拒'}")

    print("\n== 每个面板的残差随高度（偏移取该高度下的最佳值） ==")
    print("   高度   最佳偏移     RMS    最大残差   说明")
    for h in [30.0, 32.0, 33.9, 36.0, 38.0, 40.0, 44.0, 48.0, 52.0, 56.0, 60.0]:
        cand = [(evaluate(obs, PHOTO_TRUTH, o, h, cache), o)
                for o in np.arange(0.0, 40.1, 1.0)]
        cand = [(r, o) for r, o in cand if r is not None]
        if not cand:
            print(f"  {h:5.1f}   —")
            continue
        (rms, res_max, n_ok, ok), o = min(cand, key=lambda cr: cr[0][0])
        print(f"  {h:5.1f}  {o:+6.1f}°  {rms:6.2f}  {res_max:7.2f}   "
              f"覆盖 {n_ok}/7 {'★可行' if ok else '✗被门拒'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
