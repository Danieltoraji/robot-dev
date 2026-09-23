#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_truth_matrix.py —— 从现场帧生成 nine_grid「可信真值集」联络表

产出（默认写入 tests/fixtures/nine_grid_truth/）：
    truth_matrix.jpg    一张大图：7 个颜色各占一行，行内先 TRUE 列后 FALSE 列
    truth.json          程序化真值集：逐条 {color, photo, bbox, area_work, ...}
    index.json          按颜色的原始索引（人工标注用）

真值口径见 truth.json 的 `note` 字段与同目录 README.md。

用法：
    python tools/build_truth_matrix.py
    python tools/build_truth_matrix.py --frames tests/fixtures/nine_grid_truth
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

from vision.nine_grid_detector import NineGridDetector

DEFAULT_FRAMES = os.path.join(_ROOT, "tests/fixtures/nine_grid_truth")


def verdict_for(color, photo, rank, area_work, false_list):
    """按 (颜色, 照片, 名次) 匹配人工裁定的假货清单。

    用 rank（该颜色内按 hull 面积降序的名次）而不是面积做键：面积会随
    JPEG 重编码/检测器微调而漂移，rank 稳定得多；面积只作交叉校验告警。
    """
    for item in false_list:
        if item["color"] != color or item["rank"] != rank:
            continue
        if item["photo"] != photo:
            print(f"  ⚠️ 真值键不一致：{color} rank={rank} 真值记的是 "
                  f"{item['photo']}，本次是 {photo}", file=sys.stderr)
        a = item.get("area_work")
        if a is not None and abs(area_work - a) > max(2.0, 0.02 * a):
            print(f"  ⚠️ {color} rank={rank} 面积漂移：真值 {a:.0f} vs 本次 "
                  f"{area_work:.0f}（判据阈值若按面积标定需复核）", file=sys.stderr)
        return "FALSE", item.get("why", "")
    return "TRUE", ""


def main(argv=None):
    ap = argparse.ArgumentParser(description="生成 nine_grid 真值分类总图")
    ap.add_argument("--frames", default=DEFAULT_FRAMES,
                    help="现场帧目录（默认 tests/fixtures/nine_grid_truth）")
    ap.add_argument("--outdir", default=None, help="输出目录（默认 = --frames）")
    ap.add_argument("--pad", type=float, default=0.40, help="bbox 外扩比例")
    ap.add_argument("--cell", type=int, default=430, help="每格边长 px")
    ap.add_argument("--min-obs", type=int, default=4,
                    help="该帧至少检出这么多颜色才纳入（滤掉对着房间的帧）")
    ap.add_argument("--truth", default=None,
                    help="已有人工裁定 JSON（默认读 <outdir>/ground_truth.json）")
    args = ap.parse_args(argv)

    outdir = args.outdir or args.frames
    tpath = args.truth or os.path.join(outdir, "ground_truth.json")
    if not os.path.isfile(tpath):
        print(f"缺少人工裁定文件: {tpath}", file=sys.stderr)
        return 2
    with open(tpath, encoding="utf-8") as f:
        gt = json.load(f)
    false_list = gt["false_observations"]

    det = NineGridDetector()

    per_color, seen = {}, set()
    for p in sorted(glob.glob(os.path.join(args.frames, "*.jpg"))):
        if os.path.basename(p) == "truth_matrix.jpg":
            continue
        fr = cv2.imread(p)
        if fr is None:
            continue
        obs = det.detect_panels(fr)
        if len(obs) < args.min_obs:
            print(f"  [跳过] {os.path.basename(p)}：仅 {len(obs)} 个观测（<{args.min_obs}）")
            continue
        H, W = fr.shape[:2]
        sc = W / det.work_width
        for o in obs:
            x, y, w, h = (int(v) for v in o.bbox)
            key = (os.path.basename(p), o.color, x, y, w, h)
            if key in seen:
                continue
            seen.add(key)
            px, py = int(w * args.pad), int(h * args.pad)
            x0, y0 = max(0, x - px), max(0, y - py)
            x1, y1 = min(W, x + w + px), min(H, y + h + py)
            crop = fr[y0:y1, x0:x1].copy()
            if crop.size == 0:
                continue
            cv2.rectangle(crop, (x - x0, y - y0), (x - x0 + w, y - y0 + h),
                          (0, 255, 255), 6)
            area_work = round(float(o.hull_area) / (sc * sc), 1)
            per_color.setdefault(o.color, []).append({
                "color": o.color, "photo": os.path.basename(p),
                "bbox": [x, y, w, h], "area_work": area_work,
                "h_over_w": round(h / max(1.0, w), 3), "clip": int(o.clipped),
                "ev": (None if o.digit_evidence is None
                       else round(float(o.digit_evidence), 4)),
                "crop": crop,
            })

    # ---- 名次（该颜色内按 hull 面积降序）→ 裁定 ----
    for color, items in per_color.items():
        items.sort(key=lambda z: -z["area_work"])
        for i, it in enumerate(items, 1):
            it["rank"] = i
            it["verdict"], it["why"] = verdict_for(
                color, it["photo"], i, it["area_work"], false_list)

    os.makedirs(outdir, exist_ok=True)
    index = {}
    for color, items in per_color.items():
        index[color] = [{k: v for k, v in it.items() if k != "crop"} for it in items]

    # ---- 版面 ----
    C, HEAD, ROWLAB = args.cell, 46, 150
    colors = sorted(per_color.keys())
    layout = []
    for color in colors:
        items = per_color[color]
        items.sort(key=lambda z: (z["verdict"] != "TRUE", -z["area_work"]))
        tr = [z for z in items if z["verdict"] == "TRUE"]
        fa = [z for z in items if z["verdict"] == "FALSE"]
        layout.append((color, tr, fa))
    cols = max(max(len(t), len(f), 1) for _, t, f in layout)
    Wt = ROWLAB + cols * C
    Ht = HEAD + sum(C for _ in layout)
    sheet = np.full((Ht, Wt, 3), 252, np.uint8)
    cv2.putText(sheet, "nine_grid truth matrix   (T = real panel / F = false positive)",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(sheet, "yellow box = detector bbox | #n A=area(work res) h/w clip ev",
                (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (110, 110, 110), 1, cv2.LINE_AA)

    rows, y = [], HEAD
    for color, tr, fa in layout:
        cv2.putText(sheet, color.upper(), (8, y + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(sheet, f"{len(tr)}T / {len(fa)}F", (8, y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1, cv2.LINE_AA)
        cv2.line(sheet, (0, y), (Wt, y), (200, 200, 200), 1)
        for ci in range(cols):
            x0 = ROWLAB + ci * C
            if ci < len(tr):
                tag, col, it = "TRUE", (40, 150, 40), tr[ci]
            elif ci - len(tr) < len(fa):
                tag, col, it = "FALSE", (40, 40, 210), fa[ci - len(tr)]
            else:
                continue
            cv2.rectangle(sheet, (x0, y), (x0 + C - 1, y + C - 1), (170, 170, 170), 1)
            cv2.putText(sheet, tag, (x0 + 6, y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
            img = it.pop("crop")
            ih, iw = img.shape[:2]
            s = min((C - 10) / max(1, iw), (C - 70) / max(1, ih))
            img = cv2.resize(img, (max(1, int(iw * s)), max(1, int(ih * s))))
            sheet[y + 46:y + 46 + img.shape[0], x0 + 5:x0 + 5 + img.shape[1]] = img
            n = len(rows) + 1
            meta = (f"#{n} A={it['area_work']:.0f} h/w={it['h_over_w']:.2f} "
                    f"c{it['clip']} ev={'--' if it['ev'] is None else format(it['ev'], '.3f')}")
            cv2.putText(sheet, meta, (x0 + 6, y + C - 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(sheet, it["photo"].replace("photo_", "").replace(".jpg", ""),
                        (x0 + 6, y + C - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 100), 1, cv2.LINE_AA)
            it["n"] = n
            rows.append(it)
        y += C

    mpath = os.path.join(outdir, "truth_matrix.jpg")
    cv2.imwrite(mpath, sheet, [cv2.IMWRITE_JPEG_QUALITY, 82])

    json.dump({"note": gt.get("note", ""), "colors": index},
              open(os.path.join(outdir, "index.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump({
        "note": gt.get("note", ""),
        "false_reasons": {f"{i['color']}:{i['area_work']:.0f}": i.get("why", "")
                          for i in false_list},
        "rows": [{k: v for k, v in r.items() if k != "crop"} for r in rows],
    }, open(os.path.join(outdir, "truth.json"), "w", encoding="utf-8"),
        ensure_ascii=False, indent=1)

    nT = sum(1 for r in rows if r["verdict"] == "TRUE")
    print(f"\n写出 {mpath}  ({Wt}x{Ht})")
    print(f"共 {len(rows)} 观测： TRUE={nT}  FALSE={len(rows)-nT}")
    for color, tr, fa in layout:
        print(f"  {color:<7} {len(tr)}T/{len(fa)}F")
    return 0


if __name__ == "__main__":
    sys.exit(main())
