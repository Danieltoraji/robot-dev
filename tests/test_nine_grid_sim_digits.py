#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""仿真器数字渲染与真值层（tests/test_nine_grid_sim_digits.py）

背景：仿真器曾把面板上的数字画成**纯黑实心矩形**，导致 SVM/形状模板/任何数字判据
在仿真里都没有输入信号。随后一版改成"现场照片裁出来的字形贴图"，但 2026-09-25
体检证明那版有三个硬伤：墨迹带着源照片的透视斜切、渲染时被强行塞进固定 8×12cm
方框（宽高比各数字被拉坏）、且贴图角点序与世界 y 反向 ⇒ **数字被画成上下镜像**
（现场不可能出现印反的数字）。现在改为**字体渲染**（`sim.nine_grid_sim.glyph_mask`），
不依赖任何外部资产。

本文件钉住五件事：
1. **面板上真的有字形**（不是黑块、不是空面板）；
2. **朝向是"确定性的随机"**——按格位取 k、同一格永远同一个 k（不动动作噪声的
   随机流，否则对照实验不可比）；
3. **渲染出来的字不许是镜像**（角点序回归护栏，见 `test_rendered_glyph_not_mirrored`）；
4. **字形不是实心块**（用外框填充率把"字"与"黑砖"分开）；
5. **真值层可用**：格式正确、与渲染一致、且**关卡不消费它**。
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

from core.camera_config import HEAD_CENTER  # noqa: E402
from core.ground_homography import grid_cell_center  # noqa: E402
from levels.nine_grid_shared import PITCH_DOWN, project_ground_to_pixel  # noqa: E402
import sim.nine_grid_sim as SIM  # noqa: E402


def _robot_at_entry():
    r = SIM.SimNineGridRobot()
    r.pos = np.array([50.0, -20.0])
    r.heading = 0.0
    r.pitch = PITCH_DOWN
    r.head = HEAD_CENTER
    return r


def _one_panel_robot(digit, cell=4, dist=45.0):
    """场地里只放一块面板并正对相机（隔离"这个数字画得对不对"）"""
    layout = {k: None for k in range(9)}
    layout[cell] = digit
    r = SIM.SimNineGridRobot(layout=layout, seed=3)
    c = grid_cell_center(cell)
    r.pos = np.array([c[0], c[1] - dist])
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
    """渲染出来的面板里，色块内部应当有**深色墨迹**（数字）"""
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
    """字形**不是实心块**——用"墨迹在自身外框里的填充率"把"字"与"黑砖"分开

    实心矩形（旧实现的黑块）填充率恒为 1.00；笔画字形实测 0.56~0.69。
    这条正是 P0 要防的回归：如果哪天渲染悄悄退回 `fillPoly(黑块)`，上面的
    "深色占比"判据仍然通过，只有这一条会红。
    """
    fill = {d: float(m.mean()) for d, m in SIM.glyph_table().items()}
    bad = {d: round(v, 2) for d, v in fill.items() if v > 0.80}
    assert not bad, (f"这些字形太像实心块（外框填充率 {bad}；字形应 ≤0.80）"
                     "——渲染可能退回了黑块")


def test_rendered_glyph_not_mirrored():
    """渲染出来的数字**不许是镜像**（历史 bug 的回归护栏）

    判据用**独立于渲染代码**的物理朝向：站在入口看这块面板时，数字的上沿在
    远处（世界 +y）、左边在世界 −x。据此自己算四角点、自己把帧反投影回
    "正视画布"，再与字体字形比对：正立（恒等）的 IoU 必须明显高于上下翻。

    曾经这里按 [(-x,-y), (+x,-y), (+x,+y), (-x,+y)] 贴图 ⇒ 每个数字上下颠倒
    （"7" 看起来像 "L"、"5" 像 "2"），IoU 上下翻 0.85 / 恒等 0.24。
    """
    digit, cell = 6, 4      # 6 的正/反立差异最大（实测 IoU 0.80 vs 0.44）
    g = SIM.glyph_mask(digit)
    assert g is not None
    orig_k = SIM.glyph_rotate_k
    SIM.glyph_rotate_k = lambda c: 0        # 朝向另有一条测试，这里只看镜像
    try:
        r = _one_panel_robot(digit, cell)
        frame = r.capture_frame()
    finally:
        SIM.glyph_rotate_k = orig_k
    hx, hy = SIM.glyph_block_half_cm(digit, cell)
    cx, cy = grid_cell_center(cell)
    # 自己定义"站在入口看"的四角点顺序 TL,TR,BR,BL（上=远=+y）
    world = np.array([[cx - hx, cy + hy], [cx + hx, cy + hy],
                      [cx + hx, cy - hy], [cx - hx, cy - hy]])
    pix = project_ground_to_pixel(world, r.pos[0], r.pos[1],
                                  np.radians(r.heading), r.pitch, r.head)
    W, H = max(60, g.shape[1] * 4), max(60, g.shape[0] * 4)
    dst = np.array([[0.0, 0.0], [W - 1.0, 0.0],
                    [W - 1.0, H - 1.0], [0.0, H - 1.0]], np.float32)
    Hm = cv2.getPerspectiveTransform(pix.astype(np.float32), dst)
    flat = cv2.warpPerspective(frame, Hm, (W, H))
    gray = cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY)
    _, ink = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = cv2.erode(ink, np.ones((5, 5), np.uint8))
    ref = cv2.resize(g * 255, (W, H), interpolation=cv2.INTER_NEAREST)

    def iou(a, b):
        inter = np.count_nonzero((a > 0) & (b > 0))
        union = np.count_nonzero((a > 0) | (b > 0))
        return inter / union if union else 0.0

    same = iou(ref, ink)
    flip = iou(np.flipud(ref), ink)
    assert same > 0.6, f"渲染出来的字形与字体字形差太远（IoU {same:.2f}）"
    assert same > flip + 0.2, (
        f"数字被画成上下镜像了：恒等 IoU {same:.2f} vs 上下翻 {flip:.2f}"
        "（检查 `_glyph_quad` 的角点顺序：掩膜 TL 必须落在世界 (−x, +y)）")


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
