# -*- coding: utf-8 -*-
"""歧义修复测试（E）：颜色歧义 + 格阵共识的交换裁决

场景：某块面板的颜色落在两色窗口边界（红↔橙现场只差 7°），形状仲裁又没能
定案，于是"格号"可能被安错。修复思路不是再猜颜色，而是用**格阵共识**复核：
其余无歧义数字先定出刚体变换，再看歧义数字的观测点离"本格"还是"竞争格"更近。

运行：python tests/test_nine_grid_ambiguity.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import numpy as np

from core.ground_homography import grid_cell_center
from levels.nine_grid_shared import repair_ambiguous_digits, GRID_CELL_CM


def _scene(swap=True, jitter=0.0):
    """数字→格 的真值布局（观察点=真值格心；robot 系与场地系此处重合）

    真值：1→格0、2→格1、3→格4、4→格6、5→格3、6→格2、7→格5
    swap=True 时把 1/3 的格号安反（模拟颜色歧义导致误判）。
    """
    truth = {1: 0, 2: 1, 3: 4, 4: 6, 5: 3, 6: 2, 7: 5}
    med = {d: np.array(grid_cell_center(c), dtype=float) for d, c in truth.items()}
    cells = dict(truth)
    if swap:
        cells[1], cells[3] = truth[3], truth[1]
    if jitter:
        rng = np.random.default_rng(7)
        for d in med:
            med[d] = med[d] + rng.normal(0, jitter, 2)
    pick = {"med": med, "clean": set(truth)}
    return cells, pick, truth


def test_swap_repair_recovers_truth():
    cells, pick, truth = _scene(swap=True)
    fixed, notes = repair_ambiguous_digits(cells, pick, {3: {1}})
    assert fixed == truth, f"未修复：{fixed} != 真值 {truth}"
    assert notes and "交换" in notes[0]
    print(f"  交换修复：{notes[0][:52]}… ✓")


def test_no_repair_without_ambiguity():
    """没有歧义记录 ⇒ 一个格号都不许动（修复只在有歧义时介入）"""
    cells, pick, truth = _scene(swap=True)
    fixed, notes = repair_ambiguous_digits(cells, pick, {})
    assert fixed == cells and not notes
    print("  无歧义不介入 ✓")


def test_no_repair_when_geometry_inconclusive():
    """观测点两边都不明显（余量不足）⇒ 保持原判（宁可不猜）"""
    cells, pick, truth = _scene(swap=False)      # 格号本来就对
    fixed, notes = repair_ambiguous_digits(cells, pick, {3: {1}})
    assert fixed == truth and not notes, f"不该动：{fixed} / {notes}"
    print("  几何不明确时不猜 ✓")


def test_reference_too_few():
    """可信参照不足 3 个 ⇒ 原样返回（无从定刚体变换）"""
    cells, pick, _truth = _scene(swap=True)
    pick = {"med": {d: v for d, v in pick["med"].items() if d in (1, 3, 5)},
            "clean": {1, 3, 5}}
    fixed, notes = repair_ambiguous_digits(cells, pick, {3: {1}})
    assert fixed == cells and not notes
    print("  参照不足不介入 ✓")


def test_repair_survives_noise_and_margin():
    """加 1.5cm 观测噪声仍能修复；余量抬到 40cm（> 格距）则不再交换"""
    cells, pick, truth = _scene(swap=True, jitter=1.5)
    fixed, _notes = repair_ambiguous_digits(cells, pick, {3: {1}})
    assert fixed == truth, f"带噪声未修复：{fixed}"
    fixed2, notes2 = repair_ambiguous_digits(cells, pick, {3: {1}}, margin_cm=2 * GRID_CELL_CM)
    assert fixed2 == cells and not notes2, "余量超格距时不该交换"
    print(f"  噪声 1.5cm 仍修复；余量 {2 * GRID_CELL_CM:.0f}cm 时拒绝交换 ✓")


if __name__ == "__main__":
    test_swap_repair_recovers_truth()
    test_no_repair_without_ambiguity()
    test_no_repair_when_geometry_inconclusive()
    test_reference_too_few()
    test_repair_survives_noise_and_margin()
    print("歧义修复测试通过 ✓")
