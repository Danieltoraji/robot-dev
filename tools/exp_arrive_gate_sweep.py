# -*- coding: utf-8 -*-
"""临时实验 10：面板"唯一正前方"逐距离推进 —— 到达门到底卡在哪一条

现场日志（用户 2026-09-27 真机）：
  [决策] 面板1 档=绿·直行 偏角 +0.3°｜横偏 0.6%画幅｜框宽 1180px｜紫 0.319 橙 0.284 → 前进一步×3
  [决策] 面板1 档=绿·直行 偏角 +5.6°｜横偏 5.4%画幅｜框宽 1156px｜紫 0.499 橙 0.040 → 前进一步×5
第二帧**紫 0.499 ≥ 0.35 且 橙 0.040 ≤ 0.10**，却仍判"没到达"。本脚本把面板放在
正前方（横向 0 偏）逐距离推进，用**真实的**投影 + 真实的 `_zone_masks` 复算
六条到达判据，看是哪一条在什么距离上失效。

用法：python tools/exp_arrive_gate_sweep.py
"""

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import numpy as np

from core.ground_homography import GRID_CELL_CM, grid_cell_center
from levels.nine_grid import (
    ARRIVE_ASYMMETRY_MAX, ARRIVE_CENTER_MAX, ARRIVE_MAX_WIDTH_HEIGHT_RATIO,
    ARRIVE_MIN_TARGET_COVER, ARRIVE_ORANGE_MAX, ARRIVE_PURPLE_MIN,
    PITCH_DOWN, zone_masks,
)
from levels.nine_grid_shared import (
    NineGridShared, PANEL_HALF_CM, measured_cam_height_cm,
    project_ground_to_pixel,
)
from core.camera_config import HEAD_CENTER  # noqa: E402

HALF = PANEL_HALF_CM


def hull_area(pts):
    """投影点集的凸包面积（px²）；点太少时返回 0"""
    if len(pts) < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    try:
        from scipy.spatial import ConvexHull
        return float(ConvexHull(pts).volume)
    except Exception:
        # 退化/共线：用轴对齐外接框的一半兜底
        return 0.5 * float((x.max() - x.min()) * (y.max() - y.min()))


def panel_pixels(cam_xy, d_center, off_deg=25.0, head=HEAD_CENTER,
                 lateral_cm=0.0):
    """面板中心在正前方 d_center 处时，色块网格点的投影像素"""
    cx = cam_xy[0] + lateral_cm
    cy = cam_xy[1] + d_center
    n = 24
    xs = np.linspace(cx - HALF, cx + HALF, n)
    ys = np.linspace(cy - HALF, cy + HALF, n)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    px = project_ground_to_pixel(pts, cam_xy[0], cam_xy[1], 0.0, PITCH_DOWN,
                                 head, pitch_offset_deg=off_deg,
                                 cam_height_cm=measured_cam_height_cm())
    return px


def evaluate(d_center, off_deg=25.0, head=HEAD_CENTER, lateral_cm=0.0):
    """→ dict：可见份额、四区份额、六条判据的通过情况

    lateral_cm：面板中心相对相机正前方的横向偏移（cm，正=画面右侧）
    """
    from core.camera_config import CAMERA_WIDTH, CAMERA_HEIGHT
    cam_xy = (50.0, 50.0 - d_center)          # 光心在面板正后方
    px = panel_pixels(cam_xy, d_center, off_deg=off_deg, head=head,
                      lateral_cm=lateral_cm)
    W, H = float(CAMERA_WIDTH), float(CAMERA_HEIGHT)
    vis = ((px[:, 0] >= 0) & (px[:, 0] <= W) & (px[:, 1] >= 0) & (px[:, 1] <= H))
    n_all, n_vis = len(px), int(vis.sum())
    if n_vis == 0:
        return None
    vp = px[vis]
    whole = hull_area(vp) / (W * H)
    zm = zone_masks(int(W), int(H))
    keys = ["purple", "orange", "green", "blueL", "blueR"]
    cnt = {k: 0 for k in keys}
    for x, y in vp:
        if not (0 <= x < W and 0 <= y < H):
            continue
        for k in keys:
            mm = zm.get(k)
            if mm is not None and mm.size and mm[int(y), int(x)]:
                cnt[k] += 1
                break
    tot = sum(cnt.values())
    if tot == 0:
        return None
    shares = {k: cnt[k] / tot for k in keys}
    asym = (cnt["blueL"] - cnt["blueR"]) / tot
    bw = float(vp[:, 0].max() - vp[:, 0].min())
    bh = float(vp[:, 1].max() - vp[:, 1].min())
    ratio = bw / bh if bh > 1 else float("inf")
    return dict(whole=whole, pur=shares["purple"], org=shares["orange"],
                asym=asym, ratio=ratio, vis=n_vis / n_all, bw=bw, bh=bh,
                ymin=float(vp[:, 1].min()), ymax=float(vp[:, 1].max()))


def main():
    print(f"相机高 {measured_cam_height_cm()}cm / 低头 {PITCH_DOWN} / 偏移 +25°；"
          f"面板半边 {HALF}cm，正前方推进\n")
    print("  中心距离  可见%   whole   紫      橙      不对称  宽/高  框宽px  行范围      "
          "紫 橙 覆盖 形状 不对称")
    for d in (60, 50, 45, 40, 37, 35, 33, 31, 30, 29, 28, 27, 26, 25, 20):
        r = evaluate(float(d))
        if r is None:
            print(f"  {d:6.0f}cm   —— 不可见")
            continue
        g = lambda ok: "✓ " if ok else "✗ "  # noqa: E731
        gs = (g(r["pur"] >= ARRIVE_PURPLE_MIN), g(r["org"] <= ARRIVE_ORANGE_MAX),
              g(r["whole"] >= ARRIVE_MIN_TARGET_COVER),
              g(r["ratio"] <= ARRIVE_MAX_WIDTH_HEIGHT_RATIO),
              g(abs(r["asym"]) <= ARRIVE_ASYMMETRY_MAX))
        allok = "  ← **全部满足**" if all(x.strip() == "✓" for x in gs) else ""
        print(f"  {d:6.0f}cm {r['vis']*100:5.0f}%  {r['whole']:.3f}  "
              f"{r['pur']:.3f}  {r['org']:.3f}  {r['asym']:+.3f}  "
              f"{r['ratio']:5.2f}  {r['bw']:6.0f}  "
              f"{r['ymin']:5.0f}~{r['ymax']:5.0f}  "
              + " ".join(gs) + allok)
    print(f"\n门槛：紫≥{ARRIVE_PURPLE_MIN} 橙≤{ARRIVE_ORANGE_MAX} "
          f"whole≥{ARRIVE_MIN_TARGET_COVER} 宽/高≤{ARRIVE_MAX_WIDTH_HEIGHT_RATIO} "
          f"不对称≤{ARRIVE_ASYMMETRY_MAX} 居中≤{ARRIVE_CENTER_MAX}")


if __name__ == "__main__":
    main()
