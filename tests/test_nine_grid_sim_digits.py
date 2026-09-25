#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""仿真器"真字形"渲染与真值层（tests/test_nine_grid_sim_digits.py）

背景：仿真器曾把面板上的数字画成**纯黑实心矩形**，导致 SVM/形状模板/任何数字判据
在仿真里都没有输入信号。P0（2026-09-25）改为画**现场照片同源的真字形**并加面板级
真值（`SimNineGridRobot._last_frame_truth`）。本文件钉住四件事：

1. **面板上真的有字形**（不是黑块、不是空面板）；
2. **朝向是"确定性的随机"**——按格位取 k、同一格永远同一个 k（不动动作噪声的
   随机流，否则对照实验不可比）；
3. **缺资产时显式退化并留痕**（不许静默画成空面板）；
4. **真值层可用**：格式正确、与渲染一致、且**关卡不消费它**。
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.camera_config import HEAD_CENTER  # noqa: E402
from levels.nine_grid_shared import PITCH_DOWN  # noqa: E402
import sim.nine_grid_sim as SIM  # noqa: E402


def _robot_at_entry():
    r = SIM.SimNineGridRobot()
    r.pos = np.array([50.0, -20.0])
    r.heading = 0.0
    r.pitch = PITCH_DOWN
    r.head = HEAD_CENTER
    return r


def test_glyph_table_has_all_digits():
    table = SIM.glyph_table()
    assert sorted(table) == list(range(1, 8)), f"字形表不全: {sorted(table)}"
    for d, m in table.items():
        assert m.ndim == 2 and (m > 0).any(), f"d{d} 是空字形"


def test_rotate_k_is_deterministic_and_in_range():
    """朝向随机但**按格位确定**：同格永远同 k、k∈[0,3]、并且确实不是恒定值"""
    ks = [SIM.glyph_rotate_k(c) for c in range(9)]
    assert all(0 <= k <= 3 for k in ks)
    assert [SIM.glyph_rotate_k(c) for c in range(9)] == ks, "同格 k 不稳定"
    assert len(set(ks)) > 1, "所有格朝向相同 ⇒ 没有体现'朝向随机'"


def test_panel_has_ink_inside_color_block():
    """渲染出来的面板里，色块内部应当有**深色墨迹**（真字形）

    判据：在真值给的面板 bbox 内，黑色像素占该 bbox 的 1%~25%。旧的黑块实现
    也在范围内，所以另有下一条按"墨迹形状"区分块与字。
    """
    r = _robot_at_entry()
    frame = r.capture_frame()
    truth = r._last_frame_truth
    assert truth, "真值层为空"
    checked = 0
    for t in truth:
        x, y, w, h = t["bbox"]
        if w < 120 or h < 120:            # 太小的面板不看，避免采样噪声
            continue
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
        if x1 - x0 < 60 or y1 - y0 < 60:
            continue
        roi = frame[y0:y1, x0:x1]
        dark = float((roi.sum(axis=2) < 150).mean())
        assert 0.005 <= dark <= 0.30, (
            f"数字{t['digit']} 面板内深色占比 {dark:.3f} 不合理"
            f"（0 或极小=没画上字；过大=退回黑块）")
        checked += 1
    assert checked >= 1, "没有可检查的面板（站位或布局变了？）"


def test_glyph_is_not_a_solid_block():
    """字形**不是实心块**——用"墨迹的周长/面积"把"字"与"块"分开

    实心矩形的紧凑度 4πA/P² ≈ 0.785；笔画字形远低于此（本仓字形实测 ≤0.5）。
    这条正是 P0 要防的回归：如果哪天资产丢失、代码悄悄退回 `fillPoly(黑块)`，
    上面那条占比判据仍然通过，只有这一条会红。
    """
    import cv2
    checked = 0
    for d, m in SIM.glyph_table().items():
        cnts, _ = cv2.findContours((m * 255).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        assert cnts, f"d{d} 没有轮廓"
        c = max(cnts, key=cv2.contourArea)
        a, p = cv2.contourArea(c), cv2.arcLength(c, True)
        assert a > 0 and p > 0
        compact = 4.0 * np.pi * a / (p * p)
        assert compact <= 0.60, (
            f"d{d} 太像实心块（紧凑度 {compact:.2f}；字形应 ≤0.60）"
            "——字形资产可能被黑块渲染取代")
        checked += 1
    assert checked == 7


def test_missing_asset_degrades_loudly(monkeypatch, capsys):
    """资产不可用时：不抛异常、退回黑块、并**在 stdout 显式告警**（不静默）"""
    monkeypatch.setattr(SIM, "GLYPH_ASSET_PATH",
                        os.path.join(os.path.dirname(SIM.GLYPH_ASSET_PATH),
                                     "__no_such_glyphs__.npz"))
    monkeypatch.setattr(SIM, "_GLYPH_CACHE", None)
    monkeypatch.setattr(SIM, "_GLYPH_FAIL_WARNED", False)
    table = SIM.glyph_table()
    out = capsys.readouterr().out
    assert table == {}, "缺资产时应返回空表"
    assert "字形资产不可用" in out, f"缺资产没有告警：{out!r}"


def test_frame_truth_format_and_consistency():
    """真值层格式：逐面板给出 digit/cell/bbox/visible_frac/occluded_by，且与布局一致"""
    r = _robot_at_entry()
    r.capture_frame()
    truth = r._last_frame_truth
    keys = {"digit", "cell", "cx", "cy", "bbox", "visible_frac",
            "occluded_frac", "occluded_by", "clipped", "world_xy"}
    assert truth, "真值层为空"
    for t in truth:
        assert keys <= set(t), f"真值字段不全: {sorted(t)}"
        assert 1 <= t["digit"] <= 7
        assert 0 <= t["cell"] <= 8
        d = SIM.SIM_LAYOUT[t["cell"]]
        assert d == t["digit"], f"真值 digit/cell 与布局不符: {t} vs 布局 {d}"
        assert 0.0 <= t["visible_frac"] <= 1.0001, t["visible_frac"]
        assert 0.0 <= t["occluded_frac"] <= 1.0001, t["occluded_frac"]
        assert t["cx"] == pytest.approx(t["bbox"][0] + t["bbox"][2] / 2.0)
        assert isinstance(t["occluded_by"], list)


def test_truth_layer_is_not_consumed_by_level():
    """真值层只给评测用：关卡侧**不许**读它（真机没有这个属性）

    源码级断言：`levels/nine_grid.py` 里不得出现 `_last_frame_truth` /
    `_frame_truth`。它一旦被读，仿真与真机就走上了不同分支。
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(root, "levels", "nine_grid.py"),
               encoding="utf-8").read()
    for token in ("_last_frame_truth", "_frame_truth"):
        assert token not in src, f"关卡代码里出现了仿真真值 {token}（真机没有它）"
