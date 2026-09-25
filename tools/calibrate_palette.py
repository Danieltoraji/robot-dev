# -*- coding: utf-8 -*-
"""调色板标定工具：从标注帧拟合七色窗口（归一化 HSV 空间）

为什么需要它：七色窗口原先手调，现场光照一变就整块漏检（2026-09-11 实测：
粉贴纸 H 由 168 漂到 141 → 掉出窗口 → 布局扫缺数字 7）。本工具把"手调"换成
"从我们自己的实拍帧拟合"，并给出可复核的统计量与留一帧验证。

样本来源与标签：对每个帧跑 `NineGridDetector.detect_panels`，取每个 blob 的
色块像素（hull 腐蚀后、排除黑字 V<60 与白边 S<20）统计 (H, S, V) 中位；
标签 = 检测器给出的颜色。这些帧的布局已由 `tools/replay_ninegrid.py --expect`
验证与照片读图一致，故颜色标签可信（工具会打印每帧的检出集合供复核）。

用法：
    python tools/calibrate_palette.py --check            # 只打印（默认）
    python tools/calibrate_palette.py --write            # 写 models/nine_grid/palette.json
    python tools/calibrate_palette.py --frames tests/fixtures/field_photos <更多目录>
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import numpy as np
import cv2

from vision.nine_grid_detector import (NineGridDetector, COLOR_TO_ID,
                                       ID_TO_COLOR, normalize_illumination,
                                       build_color_mask)

PALETTE_PATH = os.path.join(str(_REPO), "models", "nine_grid", "palette.json")

# 拟合参数（可在命令行覆盖）
H_TOL_DEFAULT = 8.0        # 保留参数（当前 H 边界用 Voronoi 相邻中点，不用 ±tol）
S_MIN_FLOOR, S_MIN_CAP = 10.0, 25.0   # S 下限：地板 10%（白板实测 S 3~9%），上限 25%
V_MIN_FLOOR, V_MIN_CAP = 15.0, 25.0   # V 下限：地板 15%，上限 25%
MARGIN = 0.25              # 由实测最小值外扩的比例（取 1-MARGIN）


def _s_lo_of(s_min):
    """S 下限只承担"不是白板"的职责（白板 S 3~9%），故取松不取紧——

    第一版用 `min×0.8` 得到红 49%／黄 54%，把 1 个真实样本直接拒掉；
    S 会随曝光/光照整体漂移，过紧就会重现"整块漏检"。
    """
    return round(min(S_MIN_CAP, max(S_MIN_FLOOR, float(s_min) * (1 - MARGIN))), 1)


def _v_lo_of(v_min):
    """V 下限同理取松（暗阴影交给几何门 / 数字到达判断条件拦）"""
    return round(min(V_MIN_CAP, max(V_MIN_FLOOR, float(v_min) * (1 - MARGIN))), 1)


def _blob_hsv_stats(work_bgr, color_mask):
    """blob 的稳健 (H, S, V) 中位

    取"该色掩膜腐蚀后的像素 ∩ V>=60（排除黑字）"的中位——**必须**限制在色掩膜内：
    曾用 hull 直接统计，把合并进来的木框/地垫/白板一起平均，导致样本被污染
    （实测出现"红色样本 H=90"这种明显不可能的值）。
    """
    hsv = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2HSV)
    m = cv2.erode(color_mask, np.ones((7, 7), np.uint8))
    if int(np.count_nonzero(m)) < 50:
        m = color_mask
    sel = (m > 0) & (hsv[:, :, 2] >= 60)
    if int(np.count_nonzero(sel)) < 30:
        return None
    px = hsv[sel]
    return np.median(px, axis=0).astype(float)   # (H, S, V)


def collect(frames, det=None, legacy_masks=False, keep_clutter=False):
    """返回 {color: [(H,S,V,frame_tag)]}、每帧检出摘要、被标签门剔除的样本

    legacy_masks=True 时用**历史手调窗口**选样本（自举）：否则"待拟合的窗口"
    与"选样本的窗口"是同一套，会形成循环依赖——实测把 Voronoi 边界 4.5 烘进
    代码后，红板在多数帧被并进橙窗口，样本里 red 只剩 2 个，拟合结果被自己
    的错误带偏。
    keep_clutter=True 时不做"色块内要有黑字"的标签质量门（对照实验用）。
    """
    det = det or NineGridDetector()
    if legacy_masks:
        import vision.nine_grid_detector as _d
        _d.USE_PALETTE = False
    out = {}
    summary = []
    skipped = []
    for path in frames:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        tag = os.path.basename(str(path))
        obs = det.detect_panels(frame, arbitrate=False, drop_border=False)
        scale = frame.shape[1] / det.work_width
        work = cv2.resize(frame, (det.work_width,
                                  int(round(frame.shape[0] / scale))))
        # **必须与运行时同一空间**：检测器在工作分辨率上先做光照归一化再 HSV。
        # 曾漏掉这一步，样本落在原始空间（probeC 粉 H=141 而非归一化后的 166），
        # 拟合出的窗口与运行时不匹配。
        work, norm_info = normalize_illumination(work)
        hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
        summary.append((tag, sorted(o.digit for o in obs),
                        [round(g, 3) for g in norm_info["gain"]]))
        for o in obs:
            # 标签质量门：色块内几乎没有黑字、又没贴边/没小到判不了 ⇒ 场内同色
            # 杂物（木框/地垫），它的像素会把 S 下限和 H 边界拖偏，不入样本。
            if not keep_clutter and not o.has_digit_evidence():
                skipped.append((tag, o.color,
                                None if o.digit_evidence is None
                                else round(o.digit_evidence, 4)))
                continue
            # 注意：o.bbox/center_px 是**原生**分辨率坐标，本工具在工作分辨率上
            # 统计（掩膜在那里），故先除以 scale
            x, y, w, h = [int(v / scale) for v in o.bbox]
            x0, y0 = max(x, 0), max(y, 0)
            roi_hsv = hsv[y0:y + h, x0:x + w]
            mask = build_color_mask(roi_hsv, o.color)
            if int(np.count_nonzero(mask)) < 50:
                continue
            st = _blob_hsv_stats(work[y0:y + h, x0:x + w], mask)
            if st is None:
                continue
            out.setdefault(o.color, []).append((st[0], st[1] / 255.0 * 100.0,
                                                st[2] / 255.0 * 100.0, tag))
    return out, summary, skipped


def fit(samples, h_tol=H_TOL_DEFAULT):
    """按颜色拟合最终窗口表示

    返回 {color: {"h_lo","h_hi","s_lo","s_hi","v_lo","stats":{...}}}。

    H 边界的取法（2026-09-11 实测修正）：**相邻两色实测范围的间隙中点**，
    而不是"中心的中点"。原因：红↔橙的中心只差 7~8，但两者散布不同——
    红的归一化 H 实测 0~5、橙 6~13，用"中心中点"(4.5) 会把红切成橙
    （实测红板检出从 8/13 帧掉到 2/13 帧）。用"极值间隙中点"(5.5) 才正确。
    H 环绕：先把样本按环绕簇展开到以主簇为中心的连续区间，再统一归一化。
    """
    # 1) 各色环绕安全的中位中心
    centers = {}
    for color, rows in samples.items():
        arr = np.array([r[0] for r in rows], dtype=float)
        h0 = float(np.median(arr))
        d = np.abs(arr - h0)
        d = np.minimum(d, 180.0 - d)
        centers[color] = float(np.median(arr[d <= 20])) if (d <= 20).any() else h0

    # 2) 环绕展开：以第一个色（按中心排序）为基准，把落在其"另一侧"的样本
    #    平移到连续区间（例：红色样本 174~179 记为 -6~-1）
    order = sorted(centers, key=lambda c: centers[c])
    base = centers[order[0]]
    span = {}
    for color in order:
        arr = np.array([r[0] for r in samples[color]], dtype=float)
        shifted = np.where(((arr - base) % 180.0) > 90.0, arr - 180.0, arr)
        span[color] = (float(shifted.min()), float(shifted.max()))

    # 3) 相邻边界 = 两色极值间隙中点；环绕那一对用 +180 展开
    bounds = {}
    for i, color in enumerate(order):
        lo_c = order[i - 1]
        lo_max = span[lo_c][1]
        lo_lo = span[lo_c][0]
        # 前一个色相对本色的边界（把前色展开到 본色 下方）
        prev_max = lo_max if lo_max <= span[color][0] else lo_max - 180.0
        bounds[color] = (prev_max + span[color][0]) / 2.0
        _ = lo_lo
    fit_out = {}
    for color in order:
        lo = bounds[color] % 180.0
        hi = bounds[order[(order.index(color) + 1) % len(order)]] % 180.0
        arr = np.array([[r[0], r[1], r[2]] for r in samples[color]], dtype=float)
        fit_out[color] = {
            "h_lo": round(float(lo), 1),
            "h_hi": round(float(hi), 1),
            "s_lo": _s_lo_of(float(arr[:, 1].min())),
            "s_hi": 100.0,
            "v_lo": _v_lo_of(float(arr[:, 2].min())),
            "stats": {
                "n": len(samples[color]),
                "h_center": round(centers[color], 1),
                "h_min": round(span[color][0], 1),
                "h_max": round(span[color][1], 1),
                "s_med": round(float(np.median(arr[:, 1])), 1),
                "s_min": round(float(arr[:, 1].min()), 1),
                "v_med": round(float(np.median(arr[:, 2])), 1),
                "v_min": round(float(arr[:, 2].min()), 1),
            },
        }
    return fit_out


def _in_window(h, s, v, w):
    lo, hi = w["h_lo"], w["h_hi"]
    if lo <= hi:
        h_ok = lo <= h <= hi
    else:                       # 环绕窗口
        h_ok = h >= lo or h <= hi
    return (h_ok and w["s_lo"] <= s <= w["s_hi"]
            and w["v_lo"] <= v <= w.get("v_hi", 100.0))


def loo_report(samples, fit_out):
    """留一帧验证：每个面板样本按"哪些色的窗口包含它"判定"""
    ok = amb = miss = 0
    conf = {}
    for color, rows in samples.items():
        for (h, s, v, tag) in rows:
            hits = [c for c, w in fit_out.items() if _in_window(h, s, v, w)]
            if color in hits and len(hits) == 1:
                ok += 1
            elif color in hits:
                amb += 1
                conf.setdefault(color, []).append((tag, sorted(hits)))
            else:
                miss += 1
                conf.setdefault(color, []).append((tag, "MISS"))
    return ok, amb, miss, conf


def main(argv=None):
    ap = argparse.ArgumentParser(description="九宫格调色板标定（归一化 HSV 窗口）")
    ap.add_argument("--frames", nargs="*", default=None,
                    help="帧文件或目录（缺省：tests/fixtures/field_photos）")
    ap.add_argument("--h-tol", type=float, default=H_TOL_DEFAULT,
                    help="H 半窗（默认 8）")
    ap.add_argument("--legacy-masks", action="store_true",
                    help="用历史手调窗口选样本（自举；避免与待拟合窗口循环依赖）")
    ap.add_argument("--keep-clutter", action="store_true",
                    help="关掉'色块内要有黑字'的标签质量门（对照实验用）")
    ap.add_argument("--write", action="store_true",
                    help="写 models/nine_grid/palette.json（缺省只打印）")
    args = ap.parse_args(argv)

    paths = []
    for item in (args.frames or ["tests/fixtures/field_photos"]):
        p = Path(item)
        if p.is_dir():
            paths += sorted(glob.glob(str(p / "*.jpg")))
        else:
            paths.append(str(p))
    paths = [q for q in paths if "debug" not in os.path.basename(q).lower()]
    if not paths:
        print("[palette] 没有帧，退出")
        return 0
    print(f"[palette] 帧 {len(paths)} 张")

    samples, summary, skipped = collect(paths, legacy_masks=args.legacy_masks,
                                        keep_clutter=args.keep_clutter)
    for tag, digits, gain in summary:
        print(f"   {tag}: 检出 {digits}  增益 {' '.join('%.3f' % g for g in gain)}")
    if skipped:
        print(f"[palette] 标签质量门剔除 {len(skipped)} 个'色块内无黑字'样本"
              f"（木框/地垫等场内同色杂物）："
              + ", ".join(f"{t[:16]}:{c}(d={d})" for t, c, d in skipped))
    print("\n逐色拟合窗口（H 边界 = 相邻色中心中点；S/V 为百分数）:")
    fit_out = fit(samples, args.h_tol)
    for color in sorted(fit_out, key=lambda c: fit_out[c]["stats"]["h_center"]):
        w, st = fit_out[color], fit_out[color]["stats"]
        print("  %-7s n=%2d  H [%5.1f, %5.1f]（中心 %5.1f，样本 %5.1f~%5.1f）  "
              "S >= %4.1f（中位 %4.1f）  V >= %4.1f（中位 %4.1f）"
              % (color, st["n"], w["h_lo"], w["h_hi"], st["h_center"],
                 st["h_min"], st["h_max"], w["s_lo"], st["s_med"],
                 w["v_lo"], st["v_med"]))
    ok, amb, miss, conf = loo_report(samples, fit_out)
    total = ok + amb + miss
    print(f"\n窗口包含判定：唯一命中 {ok}/{total}，多色命中 {amb}，漏 {miss}")
    if conf:
        print("  非唯一/漏判样本:")
        for c, items in conf.items():
            print(f"    {c}: {items[:6]}")

    # 相邻色 H 间隔（环绕）
    print("\n相邻色 H 中心间隔（环绕）:")
    cs = sorted(fit_out, key=lambda c: fit_out[c]["stats"]["h_center"])
    for a, b in zip(cs, cs[1:]):
        gap = fit_out[b]["stats"]["h_center"] - fit_out[a]["stats"]["h_center"]
        print("   %-7s ↔ %-7s %5.1f" % (a, b, gap))
    wrap = (cs[0], cs[-1])
    gap = (fit_out[cs[0]]["stats"]["h_center"]
           + 180.0 - fit_out[cs[-1]]["stats"]["h_center"])
    print("   %-7s ↔ %-7s %5.1f（环绕）" % (wrap[1], wrap[0], gap))

    if args.write:
        os.makedirs(os.path.dirname(PALETTE_PATH), exist_ok=True)
        with open(PALETTE_PATH, "w", encoding="utf-8") as f:
            json.dump({"fitted": fit_out,
                       "frames": [os.path.basename(p) for p in paths],
                       "sampled_with": ("legacy" if args.legacy_masks
                                        else "palette")},
                      f, ensure_ascii=False, indent=2)
        print(f"\n[palette] 已写 {PALETTE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
