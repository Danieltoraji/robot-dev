#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gen_ninegrid_glyph_assets.py —— 生成"仿真器渲染用"的数字字形资产

为什么要它（2026-09-25）
------------------------
仿真器目前把面板上的数字画成**纯黑实心矩形**（`sim/nine_grid_sim.py::_draw_panel`
的 `fillPoly(..., (0,0,0))`），所以**数字识别链（SVM / 形状模板）在仿真里没有信号**：
实测同一批仿真帧上 SVM 与颜色主判的一致率只有 5.4%（≈随机），而"格 4 被判成数字 1"
这类日志根本无从复现或验证。本工具产出**与真机同源**的字形（取自现场照片），
让仿真画真数字 ⇒ 数字判据的仿真结论才可迁移到真机。

与 `gen_ninegrid_digit_templates.py` 的分工
------------------------------------------
那个工具产出**匹配用**模板（`digit_templates.npz`，经 `normalize_mask` 归一化到
28×28 画布，**原生宽高比已丢**）。本工具产出**渲染用**字形：
  - 保留**原生墨迹宽高比**（宽高比本身是判据，见 `digit_recognizer.normalize_mask`）；
  - 逐数字取"质量最好"的样本，而不是全体中位（现场帧里有大量阴影/边缘/手遮挡）；
  - **朝向不归一化**：用户裁定（2026-09-25）"现场字形朝向随机，不需要考虑这一点，
    并将仿真器的朝向也作为随机的" ⇒ 字形按提取出来的样子存，仿真端**每个面板
    随机转 90°×k**（种子固定，见 `sim/nine_grid_sim.glyph_rotate_k`）。
    ⚠️ 因此资产里**没有**也不该有 `rotate_k` 字段——朝向不是资产的属性。

为什么不用 `nine_grid_detector.extract_glyph_mask` 直接取最大连通域
------------------------------------------------------------------
实测两个坑（都已在此处规避，**不改生产检测器**）：
  1. **坐标空间**：`detect_panels` 返回**原生**像素，而 hsv 在**工作分辨率**
     （1296）。少了 `× 1/scale` 会取到完全无关的区域（曾据此得出"红色面板
     的 V 中位=127、墨迹阈值 89"的假结论）。本工具一律显式换算。
  2. **数字是色域里的洞**：`extract_glyph_mask` 取"最大色域轮廓填充后**内缩**"
     作有效区，而数字本身就是被内缩掉的那部分 ⇒ 墨迹与有效区**零交集**
     （实测 `dark ∩ inner = 0`，数字 1 因此永远取不到）。
     本工具改用"墨迹 ∩ 面板填充（外扩一点）"，并按**紧凑度 + 靠面板质心**择优选点。

运行
----
    python tools/gen_ninegrid_glyph_assets.py                # 只评估 + 打印质量表
    python tools/gen_ninegrid_glyph_assets.py --sheet        # 出候选联络表（人眼抽查质量）
    python tools/gen_ninegrid_glyph_assets.py --write        # 写 models/nine_grid/digit_glyphs.npz
"""
from __future__ import annotations

import argparse
import collections
import glob
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from vision.nine_grid_detector import (  # noqa: E402
    NineGridDetector, normalize_illumination, build_color_mask, _find_contours,
)

PHOTO_DIR = os.path.join(_ROOT, "tests", "fixtures", "field_photos")
OUT_DIR = os.path.join(_ROOT, "archive", "result", "p0_probe")
ASSET_PATH = os.path.join(_ROOT, "models", "nine_grid", "digit_glyphs.npz")

# ---- 墨迹判据（本工具专用，不进生产路径）----
V_FRAC = 0.70        # 墨迹 = V < V_FRAC × 面板 V 中位
MIN_AREA = 150       # 墨迹连通域最小面积（工作分辨率）
MIN_FILL = 0.30      # 紧凑度下限：拒掉细长阴影/边缘高光（实测那些 fill≈0.15）
MIN_SIDE = 8         # 墨迹最小边长（工作分辨率）
MAX_DIST = 0.30      # 墨迹质心到面板质心的距离 / 面板对角线 的上限
FILL_BORDER_PX = 5   # 允许墨迹越出面板填充的像素（手写/模糊会让笔画溢出）
CLOSE_PX = 7         # 闭运算：把手写笔画连成一个连通域

# 质量分（越大越好）：面积权重 × 紧凑度 × 居中奖励 × 未裁切奖励
W_FILL, W_CENTER, BONUS_CLEAN = 1.0, 1.0, 0.35


def glyph_candidates(hsv, cm, bbox):
    """色块内墨迹候选 → [(mask, diag), ...]（按质量降序）

    bbox 是**原生**像素；内部换算到工作分辨率（见模块 docstring 的坑 1）。
    """
    H, W = hsv.shape[:2]
    sc = 1.0
    # 调用方传入的 hsv 与 bbox 的分辨率比由调用方保证一致（见 collect 的换算）
    x0, y0 = max(0, int(bbox[0])), max(0, int(bbox[1]))
    x1, y1 = min(W, int(bbox[0] + bbox[2])), min(H, int(bbox[1] + bbox[3]))
    w, h = x1 - x0, y1 - y0
    if w < 12 or h < 12:
        return []
    sub = cm[y0:y1, x0:x1]
    if int(np.count_nonzero(sub)) < 50:
        return []
    cnts = sorted(_find_contours(sub), key=cv2.contourArea, reverse=True)
    if not cnts:
        return []
    filled = np.zeros((h, w), np.uint8)
    cv2.drawContours(filled, [cnts[0]], -1, 255, -1)
    M = cv2.moments(filled)
    if M["m00"] <= 0:
        return []
    pcx, pcy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    diag = float(np.hypot(w, h))

    v = hsv[y0:y1, x0:x1, 2]
    vmed = float(np.median(v[sub > 0]))
    ink = ((v < V_FRAC * vmed).astype(np.uint8)) * 255
    border = cv2.dilate(filled, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * FILL_BORDER_PX + 1,) * 2))
    ink = cv2.bitwise_and(ink, border)
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (CLOSE_PX, CLOSE_PX)))

    n, lab, st, cent = cv2.connectedComponentsWithStats(
        (ink > 0).astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        a = int(st[i, 4])
        bx, by, bw, bh = (int(t) for t in st[i, :4])
        if a < MIN_AREA or bw < MIN_SIDE or bh < MIN_SIDE:
            continue
        fill = a / float(bw * bh)
        if fill < MIN_FILL:
            continue
        cx, cy = cent[i]
        dist = float(np.hypot(cx - pcx, cy - pcy)) / diag
        if dist > MAX_DIST:
            continue
        m = (lab == i).astype(np.uint8) * 255
        ys, xs = np.nonzero(m)
        crop = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        if crop.shape[0] < MIN_SIDE or crop.shape[1] < 4:
            continue
        out.append((crop, {"area": a, "fill": fill, "dist": dist,
                           "v_med": vmed, "aspect": crop.shape[1] / float(crop.shape[0])}))
    out.sort(key=lambda t: -t[1]["area"])
    return out


def collect():
    """扫描现场帧 → {digit: [候选, ...]}（候选按质量分降序）"""
    det = NineGridDetector()
    found = collections.defaultdict(list)
    for p in sorted(glob.glob(os.path.join(PHOTO_DIR, "*.jpg"))):
        if "debug" in os.path.basename(p):
            continue
        img = cv2.imread(p)
        if img is None:
            continue
        ww = int(det.work_width)
        sc = img.shape[1] / float(ww)
        work = cv2.resize(img, (ww, int(round(img.shape[0] / sc))))
        work, _ = normalize_illumination(work)
        hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
        cms = {}
        for o in det.detect_panels(img):
            digit = o.color_id
            if not (1 <= digit <= 7):
                continue
            wb = tuple(int(round(v / sc)) for v in o.bbox)   # 原生 → 工作分辨率
            if wb[2] < 40 or wb[3] < 40:
                continue
            cm = cms.get(o.color)
            if cm is None:
                cm = cms[o.color] = build_color_mask(hsv, o.color)
            for crop, diag in glyph_candidates(hsv, cm, wb):
                q = (diag["area"] * diag["fill"]
                     * (1.0 - W_CENTER * diag["dist"])
                     * (1.0 + (BONUS_CLEAN if not o.clipped else 0.0)))
                found[digit].append({
                    "photo": os.path.basename(p), "clipped": bool(o.clipped),
                    "mask": crop, "q": float(q), **diag,
                })
    for d in found:
        found[d].sort(key=lambda e: -e["q"])
    return found


def rotations(mask, k):
    """90°×k 的旋转（k=0..3），返回同尺寸画布上的旋转结果"""
    m = np.rot90(mask, k)
    return np.ascontiguousarray(m)


def contact_sheet(found, path):
    """候选联络表：每数字 Top-3 × 四个朝向（人眼抽查字形质量用）

    ⚠️ 这张表**不用于确认朝向**（朝向已裁定为随机、不入资产）；只看"字形是不是
    真的那个数字、有没有吃到阴影/边缘"。
    """
    rows = []
    for d in range(1, 8):
        lst = found.get(d, [])[:3]
        tiles = []
        for e in lst:
            for k in range(4):
                r = rotations(e["mask"], k)
                g = cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)
                hh = 84
                g = cv2.resize(g, (max(1, int(g.shape[1] * hh / g.shape[0])), hh),
                               interpolation=cv2.INTER_NEAREST)
                pad = np.zeros((hh + 16, g.shape[1] + 6, 3), np.uint8)
                pad[16:16 + hh, 3:3 + g.shape[1]] = g
                cv2.putText(pad, "k=%d" % k, (3, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 255), 1)
                tiles.append(pad)
            tiles.append(np.full((hh + 16, 8, 3), 90, np.uint8))
        if not tiles:
            tiles = [np.zeros((100, 80, 3), np.uint8)]
        strip = np.hstack(tiles)
        head = np.zeros((24, strip.shape[1], 3), np.uint8)
        cv2.putText(head, "digit %d   candidates=%d   (each row: one sample, k=0..3)"
                    % (d, len(found.get(d, []))), (4, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        rows.append(np.vstack([head, strip]))
    W = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 3, 3, 0, W - r.shape[1],
                               cv2.BORDER_CONSTANT, value=(45, 45, 45))
            for r in rows]
    cv2.imwrite(path, np.vstack(rows))
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description="生成仿真渲染用数字字形资产")
    ap.add_argument("--sheet", action="store_true", help="出候选联络表")
    ap.add_argument("--write", action="store_true",
                    help="写 models/nine_grid/digit_glyphs.npz")
    args = ap.parse_args(argv)

    found = collect()
    print("=" * 76)
    print("字形候选质量表（现场帧目录 %s）" % os.path.relpath(PHOTO_DIR, _ROOT))
    print("=" * 76)
    missing = []
    weak = []
    for d in range(1, 8):
        lst = found.get(d, [])
        if not lst:
            missing.append(d)
            print("  数字 %d : 0 个候选   ← 缺" % d)
            continue
        clean = [e for e in lst if not e["clipped"]]
        if not clean:
            weak.append(d)
        ar = [e["aspect"] for e in lst]
        top = lst[0]
        flag = "" if clean else "   ← ⚠ 无未裁切样本，字形可能是面板弧带而非数字"
        print("  数字 %d : %2d 个候选（未裁切 %2d）｜w/h 中位 %.2f｜最佳 q=%.0f "
              "fill=%.2f dist=%.2f  %s%s"
              % (d, len(lst), len(clean), float(np.median(ar)), top["q"],
                 top["fill"], top["dist"], top["photo"], flag))
    print("\n缺字形的数字:", missing if missing else "无")
    if weak:
        print("⚠ 只靠裁切样本的数字:", weak,
              "（现场帧里这些数字没拍到完整面板；字形质量存疑，建议现场补拍）")
    os.makedirs(OUT_DIR, exist_ok=True)
    if args.sheet:
        p = contact_sheet(found, os.path.join(OUT_DIR, "glyph_contact.png"))
        print("联络表:", os.path.relpath(p, _ROOT))
    if args.write:
        data, meta = {}, {}
        for d in range(1, 8):
            lst = found.get(d, [])
            if not lst:
                print("!! 数字 %d 无候选，拒绝写出（缺字形的资产会让仿真画出空面板）" % d)
                return 2
            m = np.ascontiguousarray(lst[0]["mask"])
            data["d%d" % d] = m.astype(np.uint8)
            meta["%d_source" % d] = np.array([lst[0]["photo"]])
            meta["%d_aspect" % d] = np.array([lst[0]["aspect"]])
            meta["%d_n_candidates" % d] = np.array([len(lst)])
            meta["%d_n_clean" % d] = np.array([sum(1 for e in lst if not e["clipped"])])
            meta["%d_quality" % d] = np.array([lst[0]["q"]])
        meta["schema"] = np.array(["glyph_render_v1"])
        meta["note"] = np.array([
            "仿真渲染用字形（取自 tests/fixtures/field_photos 现场帧）。"
            "保留原生墨迹宽高比；朝向不归一化——仿真端每面板随机 90°×k。"
            "字段 <d>_n_clean==0 表示该数字只有裁切样本，字形质量存疑。"])
        np.savez_compressed(ASSET_PATH, **data, **meta)
        print("已写:", os.path.relpath(ASSET_PATH, _ROOT), "（%d 个数字）" % len(data))
    return 0


if __name__ == "__main__":
    sys.exit(main())
