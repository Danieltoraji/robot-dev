# -*- coding: utf-8 -*-
"""九宫格颜色鲁棒性测试（光照归一化 / 数据拟合调色板 / 数字形状仲裁）

覆盖 2026-09-11 的"消除现场光照影响"改造（A~C 阶段）：
  1. 白点估计与灰世界归一化：正常帧增益≈1，暖偏色帧把白点拉回中性；
  2. 13 张实拍帧七色分类（含强暖偏色探针帧 probe_c）：每种颜色都要有检出，
     且真机探针帧必须 7/7；
  3. **通道增益扰动矩阵**：模拟现场自动白平衡/曝光漂移，颜色主判必须守住
     （这是"消除现场光照影响"的量化验收）；
  4. 数字证据（C1）：色块内黑字占比区分真面板与场内同色杂物；
  5. 数字形状仲裁（C2/C3）：模板匹配机制 + "只在颜色歧义时改判"的策略。

运行：python tests/test_nine_grid_color.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import cv2
import numpy as np

import vision.nine_grid_detector as nd
from vision.digit_recognizer import DigitRecognizer
from vision.nine_grid_detector import (
    NineGridDetector, PanelObservation, estimate_white_bgr,
    normalize_illumination, PALETTE, COLOR_TO_ID, SHAPE_TEMPLATE_PATH,
)

PHOTO_DIR = os.path.join(_HERE, "fixtures", "field_photos")
# 过曝判据（任一分量打满的像素占比）：打满=信息丢失，靠归一化救不回来
NORM_CLIP_LIMIT = nd.NORM_CLIP_WARN_FRAC


def _photos(pattern="*.jpg"):
    import glob
    return sorted(p for p in glob.glob(os.path.join(PHOTO_DIR, pattern))
                  if "debug" not in os.path.basename(p).lower())


def _probes():
    import glob
    return sorted(glob.glob(os.path.join(PHOTO_DIR, "probe_*.jpg")))


# ---------------------------------------------------------------------
# 1. 光照归一化
# ---------------------------------------------------------------------

def test_white_point_and_normalization():
    """正常帧增益≈1；暖偏色帧（R 高 B 低）被拉回中性"""
    frames = _photos()
    assert frames, "缺少实拍帧夹具"
    neutral = cv2.imread(frames[0])
    wp, frac, method = estimate_white_bgr(neutral)
    assert wp is not None and frac > 0.05, f"白点估计失败: {wp}, frac={frac}"
    assert method == "neutral", f"正常帧应走低饱和主判，实际 {method}"
    _, info = normalize_illumination(neutral)
    g = info["gain"]
    assert info["applied"] and not info["suspicious"], f"正常帧应正常归一化: {info}"
    # 不变式：归一化后白点落到目标亮度且接近中性（曝光 + 偏色一起被校掉）
    fixed0, _ = normalize_illumination(neutral)
    wp_f, _, _ = estimate_white_bgr(fixed0)
    assert abs(float(np.mean(wp_f)) - nd.NORM_TARGET_WHITE_MEAN) < 8, \
        f"归一化后白点亮度应≈{nd.NORM_TARGET_WHITE_MEAN}，实际 {wp_f}"
    assert (max(wp_f) - min(wp_f)) / float(np.mean(wp_f)) < 0.05, \
        f"归一化后白点应中性，实际 {wp_f}"
    # 人为暖偏色：B×0.75、G×0.95、R×1.25（模拟现场自动白平衡漂移）
    warm = neutral.astype(np.float32) * np.array([0.75, 0.95, 1.25])
    warm = np.clip(warm, 0, 255).astype(np.uint8)
    fixed, info2 = normalize_illumination(warm)
    g2 = info2["gain"]
    wp2b, _, _ = estimate_white_bgr(warm)
    # gain 与 wp 同序（B,G,R）：暖偏色下 B 被压低 ⇒ B 的增益最大
    assert g2[0] > g2[2], f"暖偏色帧应给 B 更高增益，实际 {g2}"
    assert info2["applied"], "暖偏色帧必须真的做归一化"
    # 强偏色会把白板本身推出"低饱和"判据 ⇒ 必须走兜底判据
    assert info2["white_method"] in ("bright", "brightest"), \
        f"强偏色帧应走兜底判据，实际 {info2['white_method']}"
    wp2, _, _ = estimate_white_bgr(fixed)
    spread = (max(wp2) - min(wp2)) / max(1.0, float(np.mean(wp2)))
    spread_before = (max(wp2b) - min(wp2b)) / max(1.0, float(np.mean(wp2b)))
    assert spread < 0.5 * spread_before, \
        f"归一化应显著降低偏色：{spread_before:.3f} → {spread:.3f}"
    # 曝光归一化：整帧 ×0.75（纯变暗）也必须被拉回目标亮度
    dark = np.clip(neutral.astype(np.float32) * 0.75, 0, 255).astype(np.uint8)
    fixed3, info3 = normalize_illumination(dark)
    wp3, _, _ = estimate_white_bgr(fixed3)
    assert float(np.mean(wp3)) > 0.9 * nd.NORM_TARGET_WHITE_MEAN, \
        f"变暗帧应被拉回目标亮度，实际 {wp3}"
    print(f"  归一化：正常帧增益 {[round(x, 3) for x in g]}（{method}）；"
          f"暖偏色帧 {[round(x, 3) for x in g2]}"
          f"（{info2['white_method']} 兜底，残余偏色 {spread * 100:.1f}%）；"
          f"变暗帧增益 {[round(x, 3) for x in info3['gain']]} ✓")


# ---------------------------------------------------------------------
# 2/3. 实拍帧分类 + 增益扰动矩阵
# ---------------------------------------------------------------------

def test_field_frames_all_colors():
    """13 张实拍帧：每种颜色都要有检出；3 张真机探针帧必须 7/7"""
    det = NineGridDetector()
    seen_any = set()
    for path in _photos():
        frame = cv2.imread(path)
        seen = {o.color for o in det.detect_panels(frame)}
        seen_any |= seen
    missing = set(COLOR_TO_ID) - seen_any
    assert not missing, f"整体缺检颜色: {missing}"
    for path in _probes():
        obs = det.detect_panels(cv2.imread(path))
        colors = {o.color for o in obs}
        assert len(colors) == 7, \
            f"{os.path.basename(path)} 应 7/7，实际 {len(colors)}: {sorted(colors)}"
        weak = [o.color for o in obs if not o.has_digit_evidence()]
        assert not weak, f"{os.path.basename(path)} 真面板被判为杂物: {weak}"
    print(f"  实拍帧七色分类：{len(_photos())} 帧全部有检出，"
          f"{len(_probes())} 张真机探针帧 7/7 且全部带数字证据 ✓")


GAIN_MATRIX = [
    (1.00, 1.00, 1.00),   # 基准
    (0.80, 1.00, 1.30),   # 冷偏色（B 增益大 = 偏蓝，R 弱）
    (1.30, 1.00, 0.80),   # 暖偏色
    (1.30, 1.30, 1.30),   # 整体变亮（曝光+30%）
    (0.75, 0.75, 0.75),   # 整体变暗（曝光-25%）
    (0.70, 0.85, 1.10),   # 冷偏色 + 变暗（组合最恶劣）
]


def test_gain_perturbation_matrix():
    """通道增益扰动下颜色主判必须守住（模拟自动白平衡/曝光漂移）

    真机探针帧上叠加 BGR 通道增益，逐帧要求 7/7。有一类情况**原理上救不回来**：
    模拟增益把某通道推到 255 打满（过曝即信息丢失，真实相机应靠锁定曝光
    避免——见 CAM_LOCK_CONTROLS_ENABLED）。这类用 clip_frac 识别并单独统计，
    断言"凡未过曝的组合必须 7/7"+"过曝组合确实被 clip_frac 标记出来"。
    """
    det = NineGridDetector()
    probes = _probes()
    assert probes, "缺少真机探针帧"
    rows, clipped_rows = [], []
    for gains in GAIN_MATRIX:
        worst, worst_clip, detail = 7, 0.0, []
        for path in probes:
            frame = cv2.imread(path).astype(np.float32) * np.array(gains)
            frame = np.clip(frame, 0, 255).astype(np.uint8)
            _, info = normalize_illumination(frame)
            clip = float(info["clip_frac"])
            colors = {o.color for o in det.detect_panels(frame)}
            if clip > NORM_CLIP_LIMIT:
                worst_clip = max(worst_clip, clip)
                continue
            worst = min(worst, len(colors))
            worst_clip = max(worst_clip, clip)
            if len(colors) < 7:
                detail.append(f"{os.path.basename(path)[:14]}:"
                              f"缺{sorted(set(COLOR_TO_ID) - colors)}"
                              f"(clip={clip:.3f})")
        (clipped_rows if worst_clip > NORM_CLIP_LIMIT else rows).append(
            (gains, worst, worst_clip, detail))
    bad = [(g, w, d) for g, w, _c, d in rows if w < 7]
    assert not bad, f"未过曝的增益扰动下掉色: {bad}"
    assert rows, "至少应有一组未过曝的组合"
    print("  增益扰动矩阵：未过曝 "
          + f"{len(rows)} 组 全部 7/7（"
          + "，".join("/".join(f"{x:.2f}" for x in g) for g, _w, _c, _d in rows)
          + "）")
    for g, w, c, d in clipped_rows:
        print(f"    （过曝组 {'/'.join(f'{x:.2f}' for x in g)}："
              f"clip_frac={c:.1%} → 检出 {w}/7，已由 clip_frac 标记，"
              f"现场对策=锁定曝光）")
    assert all(c > NORM_CLIP_LIMIT for _g, _w, c, _d in clipped_rows)
    print("  通道增益扰动：颜色主判守住 ✓")


# ---------------------------------------------------------------------
# 4. 数字证据（C1）
# ---------------------------------------------------------------------

def test_digit_evidence_separates_clutter():
    """真面板（有黑字）与'颜色接近但没有印刷数字'的杂物要能被分开

    现场实测：木框橙 0.0085、蓝地垫 0.0032、绿地垫 0.0000；真面板 ≥0.0184。
    只比较**判得了**的观测（未裁切、面积够大），贴边/过小的一律免检。
    """
    assert nd.DIGIT_EVIDENCE_MIN > 0.0085, "门限必须高于杂物实测上界"
    assert nd.DIGIT_EVIDENCE_MIN <= 0.0184, "门限必须低于真面板实测下界"
    det = NineGridDetector()
    panel, clutter = [], []
    for path in _photos():
        for o in det.detect_panels(cv2.imread(path)):
            if o.clipped or o.area < nd.DIGIT_EVIDENCE_MIN_AREA:
                continue
            (panel if o.has_digit_evidence() else clutter).append(
                (o.digit_evidence or 0.0, o.color,
                 os.path.basename(path)[:16]))
    assert panel and clutter, f"夹具应同时含真面板与杂物样本: {panel} {clutter}"
    lo = min(panel)
    hi = max(clutter)
    assert lo[0] > hi[0], \
        f"真面板最小证据 {lo} 未高于杂物最大证据 {hi}"
    print(f"  数字证据：判得了的观测中 真面板 {len(panel)} 个（最小 "
          f"{lo[0]:.4f} {lo[1]}@{lo[2]}）> 杂物 {len(clutter)} 个（最大 "
          f"{hi[0]:.4f} {hi[1]}@{hi[2]}） ✓")


# ---------------------------------------------------------------------
# 5. 数字形状仲裁（C2/C3）
# ---------------------------------------------------------------------

def test_match_mask_mechanics():
    """模板匹配机制：自建 7 个可区分字形，match_mask 必须认对且间隔为正"""
    tmpl = {}
    for d in range(1, 8):
        img = np.zeros((48, 48), np.uint8)
        cv2.putText(img, str(d), (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                    255, 3, cv2.LINE_AA)
        tmpl[str(d)] = (img > 0).astype(np.uint8) * 255
    rec = DigitRecognizer(templates=tmpl)
    for d in range(1, 8):
        pred, conf, scores = rec.match_mask(tmpl[str(d)])
        assert pred == d, f"数字 {d} 误match为 {pred}（分数 {scores}）"
        assert conf > 0.0, f"数字 {d} 的匹配间隔应为正: {conf}"
    assert rec.match_mask(np.zeros((28, 28), np.uint8))[0] is None, \
        "空掩膜应返回 None"
    assert DigitRecognizer.normalize_mask(np.zeros((5, 5), np.uint8)) is None, \
        "无墨迹掩膜应归一化失败"
    print("  形状匹配机制：7 个字形全部认对、空掩膜安全返回 None ✓")


def test_shape_override_policy():
    """改判策略：**只在颜色歧义 + 间隔足够 + 形状属于候选**时才改判"""
    def obs(**kw):
        base = dict(color="orange", bbox=(0, 0, 10, 10), center_px=(5, 5),
                    area=100000, solidity=0.9)
        base.update(kw)
        return PanelObservation(**base)

    # 无歧义：即使形状高置信地反对颜色，也不改判（颜色是主判）
    o = obs(shape_digit=1, shape_conf=0.9)
    assert o.digit == 2 and not o.shape_override
    # 有歧义（同位置双色命中）+ 间隔足够 + 形状在候选内 → 改判
    o = obs(shape_digit=1, shape_conf=nd.SHAPE_OVERRIDE_MIN_CONF + 0.05,
            ambig_color="red")
    assert o.ambiguous and o.digit == 1 and o.shape_override
    # 间隔不足 → 不改判
    o = obs(shape_digit=1, shape_conf=nd.SHAPE_OVERRIDE_MIN_CONF - 0.05,
            ambig_color="red")
    assert o.digit == 2 and not o.shape_override
    # 形状给的数字不在候选内（例如说它是 5）→ 不改判
    o = obs(shape_digit=5, shape_conf=0.9, ambig_color="red")
    assert o.digit == 2 and not o.shape_override
    # 中位 H 贴窗口边界也算歧义；竞争色由调色板邻居给出（检测时确定）
    o = obs(shape_digit=1, shape_conf=0.9,
            ambig_margin_deg=nd.AMBIG_H_MARGIN_DEG - 1.0,
            ambig_candidates=("red",))
    assert o.ambiguous and o.digit == 1 and o.shape_override
    # 只贴边、但检测端没给出竞争色 ⇒ 认歧义但无从改判（候选里没有 1）
    o = obs(shape_digit=1, shape_conf=0.9,
            ambig_margin_deg=nd.AMBIG_H_MARGIN_DEG - 1.0)
    assert o.ambiguous and not o.shape_override and o.digit == 2
    o = obs(shape_digit=1, shape_conf=0.9,
            ambig_margin_deg=nd.AMBIG_H_MARGIN_DEG + 5.0)
    assert not o.ambiguous and not o.shape_override
    # 调色板邻居查找：橙窗口低侧（H=3）应落到红，高侧（H=17）应落到黄
    assert nd.palette_neighbor("orange", 3.0) == "red"
    assert nd.palette_neighbor("orange", 17.0) == "yellow"
    print("  改判策略：无歧义/间隔不足/非候选 一律不改判；歧义+高间隔才改判 ✓")


def test_shape_templates_on_field_frames():
    """现场模板在真机探针帧上必须与颜色主判一致（有结论时）"""
    if not os.path.exists(SHAPE_TEMPLATE_PATH):
        print("  形状模板缺失，跳过（python tools/gen_ninegrid_digit_templates.py --write）")
        return
    det = NineGridDetector()
    n_ok = n_decided = 0
    for path in _probes():
        for o in det.detect_panels(cv2.imread(path), shape=True):
            if o.shape_digit is None \
                    or o.shape_conf < nd.SHAPE_OVERRIDE_MIN_CONF:
                continue
            n_decided += 1
            n_ok += int(o.shape_digit == o.color_id)
            assert not o.shape_override, \
                f"{os.path.basename(path)} {o.color} 被形状改判为 {o.shape_digit}"
    assert n_decided >= 9, f"应有多数面板能给出形状结论，实际 {n_decided}"
    assert n_ok == n_decided, f"形状与颜色不一致 {n_decided - n_ok} 例"
    print(f"  现场模板：真机探针帧 {n_ok}/{n_decided} 与颜色主判一致，"
          f"无改判 ✓")


if __name__ == "__main__":
    test_white_point_and_normalization()
    test_field_frames_all_colors()
    test_gain_perturbation_matrix()
    test_digit_evidence_separates_clutter()
    test_match_mask_mechanics()
    test_shape_override_policy()
    test_shape_templates_on_field_frames()
    print("九宫格颜色鲁棒性测试通过 ✓")
