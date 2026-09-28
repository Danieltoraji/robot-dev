# -*- coding: utf-8 -*-
"""从现场实拍帧生成九宫格数字模板（models/nine_grid/digit_templates.npz）

为什么自己做模板而不复用 models/nine_grid/digit_classifier_mask.pkl：
那个 SVM 是**别处**训练的（不同字体/分辨率/光照），在 13 张现场帧上与颜色
主判的一致率只有 23%——拿它仲裁颜色等于用噪声否决真值。本工具用现场帧
自建模板：颜色主判给出标签（调色板 66/66 唯一命中，标签可信），墨迹掩膜
给出样本，逐数字取中位模板。

墨迹提取（与运行时的形状仲裁一致）：
  色域腐蚀后的内缩区域 → V < 0.6×色块 V 中位 的像素 → 开运算 →
  取最大连通域（黑字） → 裁到外框 → 缩放到 28×28。
"缩放到方形"会顺带纠正掠视角的纵向压缩（远排面板数字被压扁）。

留一评估（LOO）：对每个样本，用**去掉它本人**的中位模板重新匹配，统计
top-1 命中率与"间隔"分布——这是模板能否用来仲裁的唯一依据。

运行：
    python tools/gen_ninegrid_digit_templates.py            # 只评估 + 打印
    python tools/gen_ninegrid_digit_templates.py --write     # 写 npz
"""

import argparse
import glob
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import cv2
import numpy as np

from vision.digit_recognizer import DigitRecognizer
from vision.nine_grid_detector import (
    NineGridDetector, COLOR_TO_ID, normalize_illumination, build_color_mask,
    extract_glyph_mask,
)

TEMPLATE_PATH = os.path.join(os.path.dirname(_HERE), "models", "nine_grid",
                             "digit_templates.npz")
WORK_WIDTH = 1296


def glyph_mask(work_bgr, hsv, color_mask, bbox):
    """兼容壳：直接复用运行时的墨迹提取（保证训练/推理同一链路）"""
    return extract_glyph_mask(hsv, color_mask, bbox)


def collect(frames):
    """逐帧取样本 → {digit: [归一化模板图, ...]}，附来源标签"""
    det = NineGridDetector()
    samples = {}
    meta = []
    for path in frames:
        frame = cv2.imread(path)
        if frame is None:
            continue
        tag = os.path.basename(path)
        scale = frame.shape[1] / WORK_WIDTH
        work = cv2.resize(frame, (WORK_WIDTH, int(round(frame.shape[0] / scale))))
        obs = det.detect_panels(frame, arbitrate=False, drop_border=False)
        # 与运行时同一条链路：工作帧 → 光照归一化 → HSV
        work_n, _ = normalize_illumination(work)
        hsv = cv2.cvtColor(work_n, cv2.COLOR_BGR2HSV)
        for o in obs:
            if not o.has_digit_evidence():
                continue          # 场内同色杂物（木框/地垫）不入模板
            x, y, w, h = [int(v / scale) for v in o.bbox]
            cm = build_color_mask(hsv, o.color)
            m, diag = glyph_mask(work_n, hsv, cm, (x, y, w, h))
            if m is None:
                meta.append((tag, o.color, "跳过", diag.get("why", "")))
                continue
            norm = DigitRecognizer.normalize_mask(m)
            if norm is None:
                meta.append((tag, o.color, "跳过", "归一化失败"))
                continue
            samples.setdefault(o.color_id, []).append(norm)
            meta.append((tag, o.color, "样本", f"面积 {diag.get('area', 0):.0f}"))
    return samples, meta


def make_templates(samples, exclude=None):
    """逐数字中位模板；exclude=(digit, index) 时留一（该样本不参与）"""
    out = {}
    for digit, arrs in samples.items():
        use = [a for i, a in enumerate(arrs)
               if exclude is None or (digit, i) != exclude]
        if not use:
            continue
        stack = np.stack([a.astype(np.float32) for a in use])
        out[str(digit)] = np.median(stack, axis=0).astype(np.uint8)
    return out


def loo_report(samples):
    """留一评估 → (命中数, 总数, 命中时的间隔列表, 明细)"""
    ok = total = 0
    margins = []
    rows = []
    for digit, arrs in sorted(samples.items()):
        for i, arr in enumerate(arrs):
            tmpl = make_templates(samples, exclude=(digit, i))
            rec = DigitRecognizer(templates=tmpl)
            pred, conf, _ = rec.match_mask(arr)
            total += 1
            hit = (pred == digit)
            ok += int(hit)
            if hit:
                margins.append(conf)
            rows.append((digit, pred, round(conf, 3), hit))
    return ok, total, margins, rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="九宫格数字模板生成（现场帧自建）")
    ap.add_argument("--frames", nargs="*", default=None)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)

    paths = []
    for item in (args.frames or ["tests/fixtures/field_photos"]):
        p = item
        if os.path.isdir(p):
            paths += sorted(glob.glob(os.path.join(p, "*.jpg")))
        else:
            paths.append(p)
    paths = [q for q in paths if "debug" not in os.path.basename(q).lower()]
    if not paths:
        print("[模板] 没有帧，退出")
        return 1
    print(f"[模板] 帧 {len(paths)} 张")
    samples, meta = collect(paths)
    print("样本数：" + "，".join(f"{d}→{len(v)}"
                                for d, v in sorted(samples.items())))
    lack = [d for d in range(1, 8) if d not in samples]
    if lack:
        print(f"[模板] 警告：数字 {lack} 没有样本（现场帧里该面板贴画幅边、数字被"
              f"切掉）——形状仲裁对这些数字没有发言权（不会误改判），"
              f"补充帧后重跑即可")
    for tag, color, kind, note in meta:
        print(f"   {tag[:20]:22s} {color:7s} {kind}  {note}")

    templates = make_templates(samples)
    if not templates:
        print("[模板] 无可用样本，退出")
        return 1
    ok, total, margins, rows = loo_report(samples)
    print(f"\n留一评估：top-1 命中 {ok}/{total} = {ok / max(1, total) * 100:.1f}%")
    if margins:
        print(f"  命中时间隔：中位 {np.median(margins):.3f} "
              f"最小 {min(margins):.3f}")
    bad = [(d, p, c) for d, p, c, hit in rows if not hit]
    if bad:
        print(f"  误判 {len(bad)} 例："
              + "，".join(f"真{d}→{p}({c})" for d, p, c in bad[:12]))
    # 间隔门限扫描：仲裁只在"间隔足够大"时才敢用形状否决颜色，
    # 故真正要看的指标是"过了门限的那部分里有多少是对的"。
    print("  间隔门限扫描（用它选 SHAPE_OVERRIDE_MIN_CONF）：")
    for thr in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.50):
        sel = [hit for _d, _p, c, hit in rows if c >= thr]
        n_ok = sum(1 for h in sel if h)
        print(f"    间隔 ≥{thr:.2f}: 采纳 {len(sel):2d}/{total}"
              + (f"，其中正确 {n_ok}/{len(sel)} = "
                 f"{n_ok / len(sel) * 100:.0f}%" if sel else "（无）"))

    if args.write:
        os.makedirs(os.path.dirname(TEMPLATE_PATH), exist_ok=True)
        np.savez_compressed(
            TEMPLATE_PATH,
            **{f"d{k}": v for k, v in templates.items()},
            frames=np.array([os.path.basename(p) for p in paths]),
            loo_ok=np.array([ok]), loo_total=np.array([total]))
        print(f"[模板] 已写 {TEMPLATE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
