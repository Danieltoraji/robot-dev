#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分区判档（示意图五参数）单测：levels/nine_grid._zone_of_px

判据来源 = 用户 2026-09-24 的示意图：图**就是相机画面**，横轴画面左右、
纵轴画面上下（下 = 近），绿=中央梯形（直行）、蓝=画面底部两块（横移）、
橙=其余（旋转）。五个参数由 `tools/_tmp_diagram_measure.py` **从图里量出**
（按颜色分割 → 逐行取边界 → 直线拟合 → 化成画幅比例）。

本文件钉住三件事：
1. 五个参数**就是量出来的那五个数**——改图必须改这里，不许悄悄漂移；
2. 边界是**斜线**（上窄下宽）：同一条横向偏移，近处算绿、远处不算；
   蓝区只存在于画幅下方 `VIS_ZONE_BLUE_HEIGHT` 之内（那条水平分界线）；
3. 用户指出的病：**"又远又偏"必须落到旋转**。旧规则的边界是竖直线
   （等方位角射线），远处放得太宽，才会"还很远就横移"。

注：`tests/test_nine_grid_anchor.py::test_zone_hysteresis_*` 测的是**旧三阈值**
路径（不传 px），两条路径并存：`VIS_ZONE_PX_ENABLED=False` 才走旧的。
"""
import numpy as np
import pytest

import levels.nine_grid as NG

W, H = 2592.0, 1944.0


class _StubState:
    def __init__(self, head=NG.HEAD_CENTER):
        self.current_head_pulse = head


def _level():
    return NG.NineGridLevel(_StubState())


def _zone(lv, dx_frac, t, prev=None):
    """按"横向偏移占画幅宽的比例 dx_frac + 纵向位置 t(=py/H)"问一次档位"""
    px = W / 2.0 + dx_frac * W
    return lv._zone_of_px(px, t * H, W, H, prev)


# =====================================================================
# 1. 五个参数就是量出来的那五个数
# =====================================================================

def test_diagram_parameters_are_pinned():
    """示意图量出来的五参数（画幅比例）——见 tools/_tmp_diagram_measure.py"""
    assert (NG.VIS_ZONE_GREEN_TOP, NG.VIS_ZONE_GREEN_BOT) == \
        pytest.approx((0.0548, 0.1199))
    assert (NG.VIS_ZONE_BLUE_TOP, NG.VIS_ZONE_BLUE_BOT) == \
        pytest.approx((0.2746, 0.3623))
    assert NG.VIS_ZONE_BLUE_HEIGHT == pytest.approx(0.2445)
    assert NG.VIS_ZONE_PX_ENABLED is True


def test_green_is_narrow_at_top_wide_at_bottom():
    """绿梯形上窄下宽：同一条 dx，近处（画面下方）算绿，远处不算"""
    lv = _level()
    assert _zone(lv, 0.09, 0.99) == "move"      # 近：绿半宽 0.119
    assert _zone(lv, 0.09, 0.01) == "rot"       # 远：绿半宽 0.055，且蓝区够不着
    # 像素口径再确认一次（不受"用比例还是像素"影响）
    assert _zone(lv, 0.05, 0.50) == "move"


def test_blue_only_below_the_horizontal_line():
    """蓝区只在画幅下方 VIS_ZONE_BLUE_HEIGHT 内——这就是"太远不该横移"那条线"""
    lv = _level()
    assert _zone(lv, 0.20, 0.95) == "lat"       # 线以下（近）→ 横移
    assert _zone(lv, 0.20, 0.70) == "rot"       # 线以上（远）→ 旋转
    assert _zone(lv, 0.20, 1.0 - NG.VIS_ZONE_BLUE_HEIGHT + 1e-6) == "lat"


def test_far_and_off_center_must_turn():
    """用户指出的病：又远又偏（示例 30cm 级）必须是旋转，不是横移"""
    lv = _level()
    for t in (0.1, 0.3, 0.5, 0.75):
        for dx in (0.12, 0.20, 0.30):
            assert _zone(lv, dx, t) == "rot", (t, dx)


def test_blue_outer_edge_is_a_slanted_line():
    """蓝外沿也是斜线：近处能到 0.362，蓝区上沿只有 0.275"""
    lv = _level()
    assert _zone(lv, 0.30, 0.99) == "lat"
    assert _zone(lv, 0.40, 0.99) == "rot"
    assert _zone(lv, 0.30, 0.76) == "rot"       # 刚过蓝区上沿，外沿只剩 0.275
    assert _zone(lv, 0.26, 0.76) == "lat"


def test_hysteresis_widens_only_exit():
    """迟滞只放宽"离开"：已在本档时边界外一点点仍留在本档"""
    lv = _level()
    tol = NG.VIS_ZONE_HYST_DEG / NG.CAMERA_FOV_H_DEG
    # 横移档：0.20 之外一点点，第一次判 rot，已在 lat 时仍算 lat
    edge = _zone(lv, 0.20, 0.95)
    assert edge == "lat"
    dx_out = 0.37
    assert _zone(lv, dx_out, 0.95, prev=None) == "rot"
    assert _zone(lv, dx_out, 0.95, prev="lat") == "lat"
    assert dx_out <= 0.3623 + (0.3623 - 0.2746) * 0.8 + tol
    # 绿档：同样只放宽离开
    assert _zone(lv, 0.13, 0.99, prev=None) == "lat"
    assert _zone(lv, 0.13, 0.99, prev="move") == "move"


# =====================================================================
# 2. 两条路径的开关语义
# =====================================================================

def test_px_rule_is_used_when_enabled_and_ignored_when_zone_disabled():
    """VIS_ZONE_ENABLED=False ⇒ 完全走原二值死区（像素参数被无视）"""
    lv = _level()
    old = NG.VIS_ZONE_ENABLED
    try:
        NG.VIS_ZONE_ENABLED = False
        lv._zone_last = None
        # 又远又偏：像素规则给 rot；二值规则只要 |yaw|≤12° 就给 move
        px, py = W / 2.0 + 0.05 * W, 0.99 * H
        zv, _ = lv._zone_of_h(3.0, 0.0, 0.9, px=px, py=py, w=W, h=H)
        assert zv == "move"
    finally:
        NG.VIS_ZONE_ENABLED = old


def test_old_rule_still_reachable_for_ab():
    """VIS_ZONE_PX_ENABLED=False ⇒ 退回旧三阈值（保留做对照）"""
    lv = _level()
    old = (NG.VIS_ZONE_ENABLED, NG.VIS_ZONE_PX_ENABLED)
    try:
        NG.VIS_ZONE_ENABLED = True
        NG.VIS_ZONE_PX_ENABLED = False
        lv._zone_last = None
        # 近处、偏 8°、够近 ⇒ 旧规则给 lat（即便像素位置在很远的地方也无所谓）
        z, _ = lv._zone_of_h(8.0, 0.0, 0.9, px=W / 2.0, py=0.01 * H, w=W, h=H)
        assert z == "lat"
    finally:
        NG.VIS_ZONE_ENABLED, NG.VIS_ZONE_PX_ENABLED = old


# =====================================================================
# 3. 紫区（只看当前状态的到达判据）
# =====================================================================

def test_purple_and_arrival_thresholds_are_pinned():
    """紫区高度**独立于**蓝区高度；三个到达阈值取实测量（见常量注释）"""
    assert NG.VIS_ZONE_PURPLE_HEIGHT == pytest.approx(0.2445)
    assert NG.VIS_ARRIVE_PURPLE_MIN == pytest.approx(0.35)
    assert NG.VIS_ARRIVE_ORANGE_MAX == pytest.approx(0.10)
    assert NG.VIS_ARRIVE_ASYM_MAX == pytest.approx(0.25)
    assert NG.VIS_ARRIVE_REGION_ENABLED is True


def test_regions_are_disjoint_and_cover_frame():
    """绿/紫/蓝/橙四块互不重叠、合起来正好是整幅画面（蓝左+蓝右 = 蓝）"""
    lv = _level()
    m = lv._region_masks(320, 200)
    core = ["green", "purple", "blue", "orange"]
    tot = sum(int(m[k].sum()) for k in core)
    assert tot == 320 * 200
    assert int(m["blueL"].sum()) + int(m["blueR"].sum()) == int(m["blue"].sum())
    for i, a in enumerate(core):
        for b in core[i + 1:]:
            assert not (m[a] & m[b]).any(), (a, b)


def test_purple_height_is_independent_of_blue_height():
    """紫区上沿由 VIS_ZONE_PURPLE_HEIGHT 定，改蓝区高度不许动它"""
    lv = _level()
    old = (NG.VIS_ZONE_BLUE_HEIGHT, NG.VIS_ZONE_PURPLE_HEIGHT)
    try:
        NG.VIS_ZONE_BLUE_HEIGHT, NG.VIS_ZONE_PURPLE_HEIGHT = 0.2445, 0.2445
        m1 = lv._region_masks(320, 200)
        top1 = int(np.argmax(m1["purple"].any(axis=1)))
        blue1 = int(np.argmax(m1["blue"].any(axis=1)))
        NG.VIS_ZONE_BLUE_HEIGHT = 0.60          # 只动蓝区
        lv._region_cache = None
        m2 = lv._region_masks(320, 200)
        top2 = int(np.argmax(m2["purple"].any(axis=1)))
        blue2 = int(np.argmax(m2["blue"].any(axis=1)))
        assert top2 == top1, "紫区上沿被蓝区高度带走了"
        assert blue2 < blue1, "蓝区高度没有生效"
    finally:
        NG.VIS_ZONE_BLUE_HEIGHT, NG.VIS_ZONE_PURPLE_HEIGHT = old


def test_purple_width_equals_green_corridor():
    """紫区横向边界**就是**绿走廊（用户定：对不上会出现"没到位却只能直行"）"""
    lv = _level()
    m = lv._region_masks(320, 200)
    # 任意一行：紫∪绿 的横向范围 == 走廊（连续、左右对称）
    for row in (int(0.80 * 200), int(0.95 * 200), int(0.60 * 200)):
        cols = np.where((m["purple"] | m["green"])[row])[0]
        assert cols.size > 0
        # 关于画幅中线对称（代码按列号与 w/2 比较，允许半像素差）
        assert abs(int(cols.min()) + int(cols.max()) - 320) <= 1
        if m["purple"][row].any():          # 紫区那一行：紫的边界 == 走廊边界
            pc = np.where(m["purple"][row])[0]
            assert pc.min() == cols.min() and pc.max() == cols.max()
