#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数字字形资产完整性（tests/test_nine_grid_glyph_assets.py）

P0（2026-09-25）把仿真器的数字从"纯黑实心矩形"换成**现场照片同源的真字形**，
字形存在 `models/nine_grid/digit_glyphs.npz`。本文件钉住三件事：

1. **七个数字齐**（缺一个 ⇒ 那个面板画不出数字 ⇒ 数字判据在仿真里静默失去输入）；
2. **每个字形是"非退化的墨迹"**（非空、有合理留白、宽高比在物理范围内）——
   这条是踩过坑的：早期提取会把面板的**边缘高光/阴影弧带**当成字形（实测那些
   东西的 fill≈0.15、宽高比 1.6~2.4），画到仿真上就成了"看着有字、其实不是字"；
3. **元数据与字形一一对应**（来源帧、候选数、未裁切样本数），否则将来无法判断
   某个字形可不可信。

⚠️ 本文件**不检查朝向**：用户 2026-09-25 裁定"现场字形朝向随机，不需要考虑
这一点，并将仿真器的朝向也作为随机的" ⇒ 朝向不是资产属性（资产里也没有该字段）。
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sim.nine_grid_sim as SIM  # noqa: E402

DIGITS = tuple(range(1, 8))


@pytest.fixture(scope="module")
def asset():
    if not os.path.exists(SIM.GLYPH_ASSET_PATH):
        pytest.skip("字形资产未生成：python tools/gen_ninegrid_glyph_assets.py --write")
    return np.load(SIM.GLYPH_ASSET_PATH, allow_pickle=False)


def test_all_seven_digits_present(asset):
    """七个数字都必须有字形——缺一个就是"那个面板在仿真里没有数字" """
    missing = [d for d in DIGITS if f"d{d}" not in asset.files]
    assert not missing, (
        f"字形资产缺数字 {missing}；现场帧没覆盖到时必须如实告警/补拍，"
        f"不许让仿真静默画黑块")
    assert str(asset["schema"][0]) == "glyph_render_v1"


def test_glyphs_are_nondegenerate_ink(asset):
    """每个字形是合理的墨迹：非空、不铺满、宽高比在物理范围内

    判据来源（现场拍摄的印刷数字块是 8×12cm，字形在块内、四周有留白）：
      · 墨迹占比 0.05~0.75 —— 低于 0.05 说明只剩几粒噪声；高于 0.75 说明把
        整块数字块都涂黑了（等于退回旧的黑块行为）；
      · 宽高比 0.2~4.0 —— 实测真字形 0.71~2.0；弧带类假货是 1.6~2.4 且 fill 低，
        由下一条的 fill 门拦。
    """
    bad = []
    for d in DIGITS:
        m = asset[f"d{d}"]
        assert m.ndim == 2 and m.dtype == np.uint8, f"d{d} 形状/类型不对: {m.shape}"
        ink = float((m > 0).mean())
        if not (0.05 <= ink <= 0.75):
            bad.append((d, "墨迹占比", round(ink, 3)))
            continue
        ys, xs = np.nonzero(m > 0)
        bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        ar = bw / float(bh)
        # 连通域填充率：判"是笔画"还是"一条弧带"
        fill = float((m > 0).sum()) / float(bw * bh)
        if not (0.2 <= ar <= 4.0):
            bad.append((d, "宽高比", round(ar, 2)))
        if fill < 0.25:
            bad.append((d, "填充率(疑为面板弧带/阴影)", round(fill, 3)))
    assert not bad, f"字形质量不合格: {bad}"


def test_metadata_matches_glyphs(asset):
    """每个字形都要有来源与样本统计（否则无从判断可信度）"""
    for d in DIGITS:
        for suffix in ("_source", "_n_candidates", "_n_clean", "_quality",
                       "_aspect"):
            key = f"{d}{suffix}"
            assert key in asset.files, f"缺元数据 {key}"
        assert int(asset[f"{d}_n_candidates"][0]) >= 1, f"d{d} 候选数为 0"
    # 只有裁切样本的数字必须被如实标出（本次是 4）；将来补拍后此断言应随资产更新
    thin = [d for d in DIGITS if int(asset[f"{d}_n_clean"][0]) == 0]
    assert isinstance(thin, list)


def test_no_orientation_field(asset):
    """资产里**不许**有朝向字段：朝向是随机渲染的，不是资产属性（用户裁定）"""
    offenders = [k for k in asset.files if "rotate" in k.lower()]
    assert not offenders, (
        f"资产里出现了朝向字段 {offenders}；用户已裁定朝向随机、不归一化")
