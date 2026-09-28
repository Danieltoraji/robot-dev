# -*- coding: utf-8 -*-
"""真机照片实验 v2：色块轮廓能不能当"正方形四角"用（修正 v1 的度量 bug）

v1 的错：把轮廓点到"边的无限延长线"的距离当偏离，平行边的点会全部被算进来 ⇒ 偏离≈面板宽度。
v2 改为：轮廓点到**四边形边界折线**的最短距离（点到 4 条线段的最小值），这才是"这块轮廓有多像四边形"。
另外报告 area(轮廓)/area(四边形) 的面积比。
"""
import os
import sys
import glob
import numpy as np
import cv2

ROOT = r"F:\coding\Projects\RoboTrack\robot-dev"
sys.path.insert(0, ROOT)

from vision.nine_grid_detector import (  # noqa: E402
    normalize_illumination, build_color_mask, MIN_AREA_WORK, WORK_WIDTH,
)

DIRS = [os.path.join(ROOT, "tests", "fixtures", "field_photos"),
        os.path.join(ROOT, "tests", "fixtures", "nine_grid_truth")]
COLORS = ("red", "orange", "yellow", "green", "blue", "purple", "pink")


def dist_to_quad(pts, q):
    """每个点到四边形边界折线的最短距离"""
    best = np.full(len(pts), np.inf)
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        ab = b - a
        L2 = float(ab @ ab)
        if L2 < 1e-9:
            continue
        t = np.clip(((pts - a) @ ab) / L2, 0.0, 1.0)
        proj = a + t[:, None] * ab
        d = np.linalg.norm(pts - proj, axis=1)
        best = np.minimum(best, d)
    return best


rows = []
for d in DIRS:
    for path in sorted(glob.glob(os.path.join(d, "*.jpg"))):
        name = os.path.basename(path)
        if name.startswith("_"):
            continue
        img = cv2.imread(path)
        if img is None:
            continue
        scale = img.shape[1] / float(WORK_WIDTH)
        work = cv2.resize(img, (WORK_WIDTH, int(round(img.shape[0] / scale))))
        work, _ = normalize_illumination(work)
        hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
        for color in COLORS:
            try:
                mask = build_color_mask(hsv, color)
            except Exception:
                continue
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            cnt = max(cnts, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            if area < MIN_AREA_WORK:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            q = approx.reshape(-1, 2).astype(np.float64) if len(approx) == 4 else None
            if q is None:
                rows.append(dict(photo=name, color=color, nv=len(approx), ok=False))
                continue
            pts = cnt.reshape(-1, 2).astype(np.float64)
            dev = dist_to_quad(pts, q).max()
            qa = abs(cv2.contourArea(q.astype(np.float32)))
            rows.append(dict(photo=name, color=color, nv=4, ok=True, dev=float(dev),
                             area_ratio=float(area / qa) if qa > 0 else float("nan"),
                             side=float(max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[1]))),
                             hull_ratio=float(area / max(cv2.contourArea(cv2.convexHull(cnt)), 1e-6))))

ok = [r for r in rows if r.get("ok")]
print(f"候选 {len(rows)} 个（面积 >= {MIN_AREA_WORK}），其中 approxPolyDP 给出 4 点的 {len(ok)} 个 "
      f"（{100.0 * len(ok) / max(len(rows), 1):.0f}%）\n")
if ok:
    dev = np.array([r["dev"] for r in ok])
    side = np.array([r["side"] for r in ok])
    ar = np.array([r["area_ratio"] for r in ok])
    hr = np.array([r["hull_ratio"] for r in ok])
    print("轮廓点到四边形边界的最大偏离（工作分辨率 1296 宽）:")
    print(f"   中位 {np.median(dev):6.1f}px   P90 {np.percentile(dev, 90):6.1f}px   max {dev.max():6.1f}px")
    print("对应面板边长 px:")
    print(f"   中位 {np.median(side):6.1f}px   min {side.min():6.1f}   max {side.max():6.1f}")
    print("归一化（偏离 / 边长）:")
    rel = dev / np.maximum(side, 1.0)
    print(f"   中位 {np.median(rel) * 100:5.2f}%   P90 {np.percentile(rel, 90) * 100:5.2f}%   max {rel.max() * 100:5.2f}%")
    print("面积比 轮廓/四边形 :")
    print(f"   中位 {np.median(ar):5.3f}   min {ar.min():5.3f}   max {ar.max():5.3f}")
    print("凸实性 轮廓/凸包 :")
    print(f"   中位 {np.median(hr):5.3f}   min {hr.min():5.3f}")
    print("\n最好/最差各 6 例：")
    for r in sorted(ok, key=lambda r: r["dev"])[:6]:
        print(f"   BEST  {r['photo']:<34}{r['color']:<8} dev={r['dev']:6.1f}px side={r['side']:6.0f} "
              f"rel={r['dev'] / r['side'] * 100:5.2f}%  area={r['area_ratio']:.3f}")
    for r in sorted(ok, key=lambda r: -r["dev"])[:6]:
        print(f"   WORST {r['photo']:<34}{r['color']:<8} dev={r['dev']:6.1f}px side={r['side']:6.0f} "
              f"rel={r['dev'] / r['side'] * 100:5.2f}%  area={r['area_ratio']:.3f}")

print("\n顶点数分布（全部候选）:")
from collections import Counter  # noqa: E402
print("  ", dict(sorted(Counter(r["nv"] for r in rows).items())))
