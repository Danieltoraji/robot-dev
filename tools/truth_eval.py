#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""truth_eval.py —— 在 nine_grid 可信真值集上评测候选判据

真值集：`tests/fixtures/nine_grid_truth/truth.json`（人工目视裁定，73 观测 = 68 真 / 5 假）
重建真值集：`python tools/build_truth_matrix.py`

本脚本对每个候选判据输出**混淆矩阵**，其中「误杀真面板数」是最不可接受的一列：
多拦一个假货只损失一点鲁棒性，误杀一个真面板直接输掉一局。

用法：
    python tools/truth_eval.py
    python tools/truth_eval.py --frames tests/fixtures/nine_grid_truth --only-fails
"""

import argparse
import glob
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import cv2
import numpy as np

from core.camera_config import CAMERA_DISTORTION, CAMERA_INTRINSIC
from vision.nine_grid_detector import (NineGridDetector, build_color_mask,
                                       extract_glyph_mask, normalize_illumination)

DEFAULT_FRAMES = os.path.join(_ROOT, "tests/fixtures/nine_grid_truth")


def load_truth(path):
    """key = (photo, color, rank)，rank = 该颜色内按 hull 面积降序的名次。

    用 rank 而非面积做键：面积随 JPEG 重编码/检测器微调漂移，rank 稳定得多。
    """
    with open(path, encoding="utf-8") as f:
        gt = json.load(f)
    return {(r["photo"], r["color"], r["rank"]): r["verdict"] for r in gt["rows"]}, gt


def fit_quad(poly):
    """把凸包轮廓拟合成 4 个角点（面板物理上是方形，角点才有射影意义）。

    注意：`PanelObservation.hull_poly` 是 approxPolyDP(eps=2.0) 的**轮廓抽稀**，
    典型 5~12 个点，直接当四边形用是错的（早期版本因此让本判据几乎永不生效）。
    这里先加大 eps 直到只剩 4 点，失败则退回最小外接矩形的 4 角。
    """
    pts = np.asarray(poly, np.float32).reshape(-1, 1, 2)
    if len(pts) < 4:
        return None
    peri = float(cv2.arcLength(pts, True))
    for f in (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10):
        ap = cv2.approxPolyDP(pts, f * peri, True)
        if len(ap) == 4:
            return ap.reshape(4, 2)
    box = cv2.boxPoints(cv2.minAreaRect(pts))
    return np.asarray(box, np.float32).reshape(4, 2)


def ray_metric(poly, K, DIST):
    """去畸变 4 角 → 单位方阵单应 → (列范数比, 列夹角)。

    数学上免疫相机高度/俯仰：完整观测下真面板 ≈ (1.0, 90°)，而畸变碎片会大幅偏离。
    注意：观测被画幅裁切（缺边）时该不变量少一个自由度，结论不可用。
    """
    q = fit_quad(poly)
    if q is None:
        return None, None
    p = q.reshape(-1, 1, 2).astype(np.float32)
    n = cv2.undistortPoints(p, K, DIST).reshape(-1, 2).astype(np.float32)
    if len(n) != 4:
        return None, None
    src = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    H = cv2.getPerspectiveTransform(src, n)
    a, b = H[:, 0][:2], H[:, 1][:2]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return None, None
    cos = min(1.0, abs(float(np.dot(a, b))) / (na * nb))
    return float(max(na / nb, nb / na)), float(np.degrees(np.arccos(cos)))


def main(argv=None):
    ap = argparse.ArgumentParser(description="在真值集上评测 nine_grid 候选判据")
    ap.add_argument("--frames", default=DEFAULT_FRAMES)
    ap.add_argument("--truth", default=None, help="默认 <frames>/truth.json")
    ap.add_argument("--min-obs", type=int, default=4)
    ap.add_argument("--only-fails", action="store_true", help="只打印假货清单后退出")
    args = ap.parse_args(argv)

    tpath = args.truth or os.path.join(args.frames, "truth.json")
    truth, gt = load_truth(tpath)
    det = NineGridDetector()
    K, DIST = CAMERA_INTRINSIC, CAMERA_DISTORTION

    rows, seen, unmatched = [], set(), []
    per_color = {}
    for p in sorted(glob.glob(os.path.join(args.frames, "*.jpg"))):
        base = os.path.basename(p)
        if base == "truth_matrix.jpg":
            continue
        fr = cv2.imread(p)
        if fr is None:
            continue
        obs = det.detect_panels(fr)
        if len(obs) < args.min_obs:
            continue
        H, W = fr.shape[:2]
        sc = W / det.work_width
        for o in obs:
            x, y, w, h = (int(v) for v in o.bbox)
            key = (base, o.color, x, y, w, h)
            if key in seen:
                continue
            seen.add(key)
            per_color.setdefault(o.color, []).append((float(o.hull_area), o, fr, base, sc))

    for color, items in per_color.items():
        items.sort(key=lambda z: -z[0])
        for rank, (hull, o, fr, base, sc) in enumerate(items, 1):
            x, y, w, h = (int(v) for v in o.bbox)
            hw = hull / (sc * sc)
            ga = 0.0
            sub = fr[y:y + h, x:x + w]
            if sub.size:
                wk = cv2.resize(sub, (max(1, w // 2), max(1, h // 2)))
                wk, _ = normalize_illumination(wk)
                hsv = cv2.cvtColor(wk, cv2.COLOR_BGR2HSV)
                cm = build_color_mask(hsv, color)
                gm, _ = extract_glyph_mask(hsv, cm, (0, 0, wk.shape[1], wk.shape[0]))
                ga = 0.0 if gm is None else float(cv2.countNonZero(gm))
            asp, orth = (ray_metric(o.hull_poly, K, DIST)
                         if len(o.hull_poly) >= 4 else (None, None))
            hit = truth.get((base, color, rank))
            if hit is None:
                unmatched.append((base, color, rank, hw))
                hit = "TRUE"
            rows.append(dict(photo=base, color=color, rank=rank, hw=hw, asp=asp,
                             orth=orth, clip=int(o.clipped),
                             ev=(None if o.digit_evidence is None
                                 else float(o.digit_evidence)),
                             glyph=ga, hw_ratio=h / max(1.0, w),
                             complete=(int(o.clipped) == 0 and len(o.hull_poly) >= 4),
                             false=(hit == "FALSE")))

    nf = sum(1 for r in rows if r["false"])
    nT = len(rows) - nf
    print(f"真值集：{len(rows)} 观测 = {nT} 真 / {nf} 假")
    if unmatched:
        print(f"\n⚠️ {len(unmatched)} 条观测在真值集里找不到匹配（判据表可能已过期，"
              f"请重跑 tools/build_truth_matrix.py）:")
        for base, color, rank, hw in unmatched:
            print(f"     {color:>8} rank={rank:>2d} A={hw:>9.0f} {base}")
        return 3

    print("\n假货清单（核对真值匹配是否正确）:")
    for r in rows:
        if r["false"]:
            print(f"    {r['color']:>8} rank={r['rank']:>2d} A={r['hw']:>8.0f} "
                  f"h/w={r['hw_ratio']:.2f} clip={r['clip']} "
                  f"glyph={r['glyph']:.0f} {r['photo']}")
    if args.only_fails:
        return 0

    def evaluate(name, pred):
        fn = sum(1 for r in rows if (not r["false"]) and pred(r))
        tp = sum(1 for r in rows if r["false"] and pred(r))
        flag = "  ← 零误杀" if fn == 0 else "  **有误杀**"
        print(f"  {name:<50} 拦假 {tp}/{nf}   误杀真 {fn}/{nT}{flag}")
        return fn, tp

    print("\n" + "=" * 96)
    print("候选判据评测（『误杀真』是最不可接受的一列）")
    print("=" * 96)
    for a in (300000, 200000, 160000, 150000, 147000, 140000, 100000, 50000):
        evaluate(f"hull_work > {a}", lambda r, a=a: r["hw"] > a)
    evaluate("h/w > 1.5", lambda r: r["hw_ratio"] > 1.5)
    evaluate("h/w < 0.35", lambda r: r["hw_ratio"] < 0.35)
    evaluate("h/w > 1.5 或 < 0.35",
             lambda r: r["hw_ratio"] > 1.5 or r["hw_ratio"] < 0.35)
    evaluate("glyph == 0", lambda r: r["glyph"] == 0.0)
    evaluate("clipped==1 且 glyph==0", lambda r: r["clip"] == 1 and r["glyph"] == 0.0)
    evaluate("clipped==1 且 glyph==0 且 h/w∉[0.5,1.3]",
             lambda r: r["clip"] == 1 and r["glyph"] == 0.0
             and not (0.5 <= r["hw_ratio"] <= 1.3))
    evaluate("geom: complete 且 (asp>1.3 或 orth∉[75,105])",
             lambda r: r["complete"] and r["asp"] is not None
             and (r["asp"] > 1.3 or not (75 <= r["orth"] <= 105)))

    print("\n" + "=" * 96)
    print("真面板 h/w 实测跨度（决定 ASPECT 门下界能压到多低）")
    print("=" * 96)
    tr = [r["hw_ratio"] for r in rows if not r["false"]]
    fa = [r["hw_ratio"] for r in rows if r["false"]]
    print(f"  真面板 h/w: min={min(tr):.3f}  max={max(tr):.3f}  "
          f"（共 {len(tr)} 条）")
    print(f"  假货   h/w: {sorted(round(v, 3) for v in fa)}")
    print(f"  ⇒ 真的 h/w 区间是否完全包住假的: "
          f"{min(tr) <= min(fa) and max(tr) >= max(fa)}")

    print("\n" + "=" * 96)
    print("几何射线度量不变量（仅对 complete 观测有意义）")
    print("=" * 96)
    print(f"  [诊断] complete 真 {sum(1 for r in rows if r['complete'])} 条"
          f"；asp 可用 {sum(1 for r in rows if r.get('asp') is not None)} 条")
    cT = [r for r in rows if not r["false"] and r["complete"] and r["asp"]]
    cF = [r for r in rows if r["false"] and r["complete"] and r["asp"]]
    print(f"  complete 真 {len(cT)} 条；complete 假 {len(cF)} 条")
    if cT:
        print(f"    真: asp {min(r['asp'] for r in cT):.3f}~{max(r['asp'] for r in cT):.3f}  "
              f"orth {min(r['orth'] for r in cT):.1f}~{max(r['orth'] for r in cT):.1f}°")
    if cF:
        print(f"    假: asp {min(r['asp'] for r in cF):.3f}~{max(r['asp'] for r in cF):.3f}  "
              f"orth {min(r['orth'] for r in cF):.1f}~{max(r['orth'] for r in cF):.1f}°")
    print(f"  ⚠️ 全部 {nf} 个假货中 complete 的只有 {len(cF)} 个 ⇒ "
          f"该不变量对本真值集的假货**基本不可用**（假货多为画幅裁切碎片，缺边即少一自由度）")

    ncl = sum(1 for r in rows if not r["false"] and not r["complete"])
    print(f"\n  ⚠️ 代价核算：真面板里被画幅裁切的有 {ncl}/{nT} 条。"
          f"若用『必须 complete』当硬门，虽能拦掉 {nf}/{nf} 个假货，"
          f"但会连 {ncl} 条真面板一起丢掉 ⇒ **不可用**。")
    print("  ⇒ 结论：单帧像素域没有可用判别器；必须走多帧探针累积证据。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
