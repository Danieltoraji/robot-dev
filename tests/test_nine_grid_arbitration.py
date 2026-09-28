# -*- coding: utf-8 -*-
"""同色候选仲裁单测（vision/nine_grid_detector.pick_same_color）

背景（2026-09-28 现场）：木框会被黄/橙色掩膜命中，面积常比真面板还大
（93k vs 62k px），纯按面积择大会追着木框跑。实测 277 个观测上没有任何
单一颜色/形状/纹理指标能完全分开真面板与杂物，故用三级仲裁：
黑字证据 → 纯度 → 面积，前两级都平局时弃权。

硬约束：**单候选时逐位不变**（近距离到达段数字出画/被机身挡住时只有一个
候选，这条规则不得影响它）。
"""
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from vision.nine_grid_detector import (  # noqa: E402
    pick_same_color, is_trustworthy, evidence_strong,
    DIGIT_EVIDENCE_MIN, PURITY_MIN,
)


class _Obs:
    """最小替身：仲裁只看 digit_evidence / purity / hull_area"""

    def __init__(self, color, ev, purity, area):
        self.color = color
        self.digit_evidence = ev
        self.purity = purity
        self.hull_area = area

    def has_digit_evidence(self):          # 软判据（贴边一律 True）——故意留着
        return True


def test_single_candidate_is_noop():
    """单候选：无论证据/纯度多低，都原样返回（保护到达段）"""
    for ev, purity in ((0.0, 0.0), (0.05, 0.0), (0.0, 0.9)):
        o = _Obs("yellow", ev, purity, 5000)
        assert pick_same_color([o]) is o, "单候选必须 no-op"
    assert pick_same_color([]) is None
    print("  单候选 no-op（含 证据0/纯度0 的近距离情形） ✓")


def test_evidence_tier_beats_bigger_clutter():
    """① 黑字证据优先：木框更大也不选它"""
    wood = _Obs("yellow", 0.000, 0.00, 145000)
    panel = _Obs("yellow", 0.074, 0.55, 120000)
    win = pick_same_color([wood, panel])
    assert win is panel, "有证据的真面板必须胜过更大的木框"
    assert evidence_strong(panel) and not evidence_strong(wood)
    print("  证据级：真面板(0.074, 12万px) 胜过木框(0.000, 14.5万px) ✓")


def test_purity_tier_when_no_evidence():
    """② 无证据时看纯度（数字被遮挡的真面板）"""
    wood = _Obs("yellow", 0.000, 0.00, 145000)
    occluded = _Obs("yellow", 0.000, 0.72, 30000)
    assert pick_same_color([wood, occluded]) is occluded
    print("  纯度级：数字被遮挡但色块仍纯(0.72) 胜过木框(0.00) ✓")


def test_abstain_when_both_tiers_tie():
    """③ 两级都平局 ⇒ 弃权（宁可这帧没观测，也不硬选大的）"""
    a = _Obs("yellow", 0.001, 0.05, 145000)
    b = _Obs("yellow", 0.000, 0.00, 90000)
    assert pick_same_color([a, b]) is None, "两级都不过时必须弃权"
    assert not is_trustworthy(a) and not is_trustworthy(b)
    print("  平局弃权：两个都不信 → None（不按面积硬选） ✓")


def test_trustworthy_gate():
    """不可信 = 证据不硬 且 纯度低于 PURITY_MIN（布局扫"定数字"用）"""
    assert is_trustworthy(_Obs("yellow", 0.074, 0.00, 1000))          # 证据硬
    assert is_trustworthy(_Obs("yellow", 0.000, PURITY_MIN + 0.01, 9))  # 够纯
    assert not is_trustworthy(_Obs("yellow", 0.000, PURITY_MIN - 0.01, 90000))
    # 关键：软判据 has_digit_evidence() 对贴边观测一律 True，不能拿来当硬门
    wood = _Obs("yellow", 0.000, 0.000, 145000)
    assert wood.has_digit_evidence() and not is_trustworthy(wood), \
        "软判据必须在仲裁处被绕过（木框永远贴边）"
    assert DIGIT_EVIDENCE_MIN > 0.0095 and PURITY_MIN >= 0.2
    print("  可信门：证据硬 或 纯度≥%.2f；贴边木框被软判据放过但被硬门拦住 ✓"
          % PURITY_MIN)


if __name__ == "__main__":
    print("同色候选仲裁单测：")
    test_single_candidate_is_noop()
    test_evidence_tier_beats_bigger_clutter()
    test_purity_tier_when_no_evidence()
    test_abstain_when_both_tiers_tie()
    test_trustworthy_gate()
    print("全部通过 ✓")
