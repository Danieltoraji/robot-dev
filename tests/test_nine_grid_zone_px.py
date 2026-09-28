#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分区判档（示意图五参数）单测：levels/nine_grid._zone_at_pixel

判据来源 = 用户 2026-09-24 的示意图：图**就是相机画面**，横轴画面左右、
纵轴画面上下（下 = 近），绿=中央梯形（直行）、蓝=画面底部两块（横移）、
橙=其余（旋转）。五个参数由 `tools/_tmp_diagram_measure.py` **从图里量出**
（按颜色分割 → 逐行取边界 → 直线拟合 → 化成画幅比例）。

本文件钉住三件事：
1. 五个参数**就是量出来的那五个数**——改图必须改这里，不许悄悄漂移；
2. 边界是**斜线**（上窄下宽）：同一条横向偏移，近处算绿、远处不算；
   蓝区只存在于画幅下方 `ZONE_BLUE_HEIGHT` 之内（那条水平分界线）；
3. 用户指出的病：**"又远又偏"必须落到旋转**。旧规则的边界是竖直线
   （等方位角射线），远处放得太宽，才会"还很远就横移"。

判档只有这一条路径（旧三阈值与两个开关已随"删用不上的代码"一步移除）。

2026-09-25 追加第 3 节：判据抽成模块级纯函数后的不变量 ——
`zone_at_pixel`（判档）/ `zone_masks`（掩膜）/ `zone_lines`（分割线，调试镜像的
分区叠加用）必须**共用同一套公式**：逐点判档 == 掩膜标签；画出来的线 == 边界
（两侧异区、且覆盖全部跳变）；运行时改 `ZONE_*` 仍然生效。
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
    return lv._zone_at_pixel(px, t * H, W, H, prev)


# =====================================================================
# 1. 五个参数就是量出来的那五个数
# =====================================================================

def test_diagram_parameters_are_pinned():
    """示意图量出来的五参数（画幅比例）——见 tools/_tmp_diagram_measure.py"""
    assert (NG.ZONE_GREEN_TOP, NG.ZONE_GREEN_BOT) == \
        pytest.approx((0.0548, 0.1199))
    assert (NG.ZONE_BLUE_TOP, NG.ZONE_BLUE_BOT) == \
        pytest.approx((0.2746, 0.3623))
    assert NG.ZONE_BLUE_HEIGHT == pytest.approx(0.2445)


def test_green_is_narrow_at_top_wide_at_bottom():
    """绿梯形上窄下宽：同一条 dx，近处（画面下方）算绿，远处不算"""
    lv = _level()
    assert _zone(lv, 0.09, 0.99) == "forward"      # 近：绿半宽 0.119
    assert _zone(lv, 0.09, 0.01) == "turn"       # 远：绿半宽 0.055，且蓝区够不着
    # 按像素算再确认一次（不受"用比例还是像素"影响）
    assert _zone(lv, 0.05, 0.50) == "forward"


def test_blue_only_below_the_horizontal_line():
    """蓝区只在画幅下方 ZONE_BLUE_HEIGHT 内——这就是"太远不该横移"那条线"""
    lv = _level()
    assert _zone(lv, 0.20, 0.95) == "side"       # 线以下（近）→ 横移
    assert _zone(lv, 0.20, 0.70) == "turn"       # 线以上（远）→ 旋转
    assert _zone(lv, 0.20, 1.0 - NG.ZONE_BLUE_HEIGHT + 1e-6) == "side"


def test_far_and_off_center_must_turn():
    """用户指出的病：又远又偏（示例 30cm 级）必须是旋转，不是横移"""
    lv = _level()
    for t in (0.1, 0.3, 0.5, 0.75):
        for dx in (0.12, 0.20, 0.30):
            assert _zone(lv, dx, t) == "turn", (t, dx)


def test_blue_outer_edge_is_a_slanted_line():
    """蓝外沿也是斜线：近处能到 0.362，蓝区上沿只有 0.275"""
    lv = _level()
    assert _zone(lv, 0.30, 0.99) == "side"
    assert _zone(lv, 0.40, 0.99) == "turn"
    assert _zone(lv, 0.30, 0.76) == "turn"       # 刚过蓝区上沿，外沿只剩 0.275
    assert _zone(lv, 0.26, 0.76) == "side"


def test_hysteresis_widens_only_exit():
    """迟滞只放宽"离开"：已在本档时边界外一点点仍留在本档"""
    lv = _level()
    tol = NG.ZONE_HYSTERESIS_DEG / NG.CAMERA_FOV_H_DEG
    # 横移档：0.20 之外一点点，第一次判 rot，已在 lat 时仍算 lat
    edge = _zone(lv, 0.20, 0.95)
    assert edge == "side"
    dx_out = 0.37
    assert _zone(lv, dx_out, 0.95, prev=None) == "turn"
    assert _zone(lv, dx_out, 0.95, prev="side") == "side"
    assert dx_out <= 0.3623 + (0.3623 - 0.2746) * 0.8 + tol
    # 绿档：同样只放宽离开
    assert _zone(lv, 0.13, 0.99, prev=None) == "side"
    assert _zone(lv, 0.13, 0.99, prev="forward") == "forward"


# =====================================================================
# 2. 紫区（只看当前状态的到达判据）
# =====================================================================

def test_purple_and_arrival_thresholds_are_pinned():
    """紫区高度**独立于**蓝区高度；三个到达阈值取实测量（见常量注释）

    ⚠️ `ZONE_PURPLE_HEIGHT` 的取值史：
      · 0.2445（示意图比例）只覆盖画幅底边上方约 **4cm** 地面 ⇒ 紫份额要到离格心
        ≤4cm 才达 0.35（15cm 处仅 0.323、12cm 0.332、8cm 0.365），而低头档单步
        2.652cm ⇒ 单格要走 ~14 次前进决策；
      · 2026-09-25（P1 重写）抬到 0.62 ⇒ 覆盖约 **16cm**（在 ±半格 16.7cm 内），
        判据在 ~40cm 起就满足紫门（40cm→0.345、30cm→0.505、20cm→0.546）；
      · **2026-09-26 调参减半到 0.31**（现场手调，覆盖约 8cm 地面）——到达点更靠
        格心，代价是紫门要走到 ~8cm 才成立。断言跟着当前值走，**不是**历史值。
    这条与"蓝区高度"**仍然独立**（见下一条测试）。
    """
    assert NG.ZONE_PURPLE_HEIGHT == pytest.approx(0.31)
    assert NG.ARRIVE_PURPLE_MIN == pytest.approx(0.35)
    # ★ 2026-09-28 按**真机四张到达参考帧**重标：橙门 0.10 → **0.30**。
    #   旧值 0.10 把四个"期望到达"帧全判 ✗（实测橙 0.164~0.252），机器人因此
    #   永远不判到达。依据见 docs/关卡算法/.../refs/ 与核对报告 §8.4。
    assert NG.ARRIVE_ORANGE_MAX == pytest.approx(0.30)
    # ★ 同日按用户要求**短路**两条（0.0 = 关闭，见常量块留档）：
    #   "整帧占比 ≥0.065" 在 33.9cm 几何下对真到达恒判 ✗；"宽/高 ≤2.15" 的
    #   真到达簇落到 2.83~3.08（旧门是 h=56cm 时代标的）。
    assert NG.ARRIVE_MIN_TARGET_COVER == pytest.approx(0.0)
    assert NG.ARRIVE_MAX_WIDTH_HEIGHT_RATIO == pytest.approx(0.0)
    # 2026-09-25 由 0.25 改为 0.55：该量是**角度量**（同横偏 @4cm/@10cm/@20cm 读数
    # 差 2~3 倍），且面板 28cm 宽于绿走廊（±7cm）⇒ 永远到不了 0.5 以上多少，
    # 0.25 实际只容许 ±1.7cm 横偏而蓝档稳定在 ±5cm。扫门实测 0.25→10/21、
    # 0.55→20/21；落地精度复核（3 种子）为 6.1~10.1cm，未因放门而压偏。
    assert NG.ARRIVE_ASYMMETRY_MAX == pytest.approx(0.55)
    assert NG.ARRIVE_CENTER_MAX == pytest.approx(0.12)


def test_regions_are_disjoint_and_cover_frame():
    """绿/紫/蓝/橙四块互不重叠、合起来正好是整幅画面（蓝左+蓝右 = 蓝）"""
    lv = _level()
    m = lv._zone_masks(320, 200)
    core = ["green", "purple", "blue", "orange"]
    tot = sum(int(m[k].sum()) for k in core)
    assert tot == 320 * 200
    assert int(m["blueL"].sum()) + int(m["blueR"].sum()) == int(m["blue"].sum())
    for i, a in enumerate(core):
        for b in core[i + 1:]:
            assert not (m[a] & m[b]).any(), (a, b)


def test_purple_height_is_independent_of_blue_height():
    """紫区上沿由 ZONE_PURPLE_HEIGHT 定，改蓝区高度不许动它"""
    lv = _level()
    old = (NG.ZONE_BLUE_HEIGHT, NG.ZONE_PURPLE_HEIGHT)
    try:
        NG.ZONE_BLUE_HEIGHT, NG.ZONE_PURPLE_HEIGHT = 0.2445, 0.2445
        m1 = lv._zone_masks(320, 200)
        top1 = int(np.argmax(m1["purple"].any(axis=1)))
        blue1 = int(np.argmax(m1["blue"].any(axis=1)))
        NG.ZONE_BLUE_HEIGHT = 0.60          # 只动蓝区
        lv._region_cache = None
        m2 = lv._zone_masks(320, 200)
        top2 = int(np.argmax(m2["purple"].any(axis=1)))
        blue2 = int(np.argmax(m2["blue"].any(axis=1)))
        assert top2 == top1, "紫区上沿被蓝区高度带走了"
        assert blue2 < blue1, "蓝区高度没有生效"
    finally:
        NG.ZONE_BLUE_HEIGHT, NG.ZONE_PURPLE_HEIGHT = old


def test_purple_width_equals_green_corridor():
    """紫区横向边界**就是**绿走廊（用户定：对不上会出现"没到位却只能直行"）"""
    lv = _level()
    m = lv._zone_masks(320, 200)
    # 任意一行：紫∪绿 的横向范围 == 走廊（连续、左右对称）
    for row in (int(0.80 * 200), int(0.95 * 200), int(0.60 * 200)):
        cols = np.where((m["purple"] | m["green"])[row])[0]
        assert cols.size > 0
        # 关于画幅中线对称（代码按列号与 w/2 比较，允许半像素差）
        assert abs(int(cols.min()) + int(cols.max()) - 320) <= 1
        if m["purple"][row].any():          # 紫区那一行：紫的边界 == 走廊边界
            pc = np.where(m["purple"][row])[0]
            assert pc.min() == cols.min() and pc.max() == cols.max()


# =====================================================================
# 3. 纯函数化（2026-09-25）：判档/掩膜/分割线共用一套公式
# =====================================================================
# 这三种东西现在都在 levels/nine_grid.py 的模块级纯函数里：
#   zone_at_pixel 判档 ｜ zone_masks 掩膜 ｜ zone_lines 分割线（调试叠加用）
# 下面钉住：① 类方法只是薄包装；② 判档与掩膜逐像素一致；③ 画出来的线**就是**
# 判据的边界（一条不多、一条不少）；④ 运行时改 ZONE_* 仍然生效（record_run --tune）。


def _label_grid(m):
    """四块掩膜 → 标签栅格（0=绿 1=紫 2=蓝 3=橙）"""
    lab = np.full(m["green"].shape, 3, dtype=np.int8)
    lab[m["green"]] = 0
    lab[m["purple"]] = 1
    lab[m["blue"]] = 2
    return lab


def _zone_from_label(lab_val):
    """四块 → 三档（紫区在判档里和绿区同属"直行"）。"""
    return {0: "forward", 1: "forward", 2: "side", 3: "turn"}[int(lab_val)]


def _seg_dist(pts, a, b):
    """点集到线段的距离（像素单位；pts 已经是像素坐标）"""
    a = np.asarray(a, dtype=float)
    ab = np.asarray(b, dtype=float) - a
    t = np.clip(((pts - a) @ ab) / float(ab @ ab), 0.0, 1.0)
    return np.linalg.norm(pts - (a + t[:, None] * ab), axis=1)


def _boundary_gap_px(x, y, w, h):
    """像素 (x,y) 到最近一条分区边界线的距离（像素）——贴线像素的容差判定用"""
    t = float(y) / h
    dx = abs(float(x) - w / 2.0) / w
    gaps = [abs(dx - NG.zone_green_half(t)) * w,
            abs(t - NG.zone_purple_top_t()) * h]
    tb = NG.zone_blue_top_t()
    if t >= tb:
        bh = float(NG.ZONE_BLUE_HEIGHT)
        u = ((t - tb) / bh) if bh > 0 else 1.0
        gaps.append(abs(dx - NG.zone_blue_half(u)) * w)
    else:
        gaps.append(abs(t - tb) * h)
    return min(gaps)


def test_methods_are_thin_wrappers_over_pure_functions():
    """类方法 = 模块级纯函数的薄包装（老调用点、探针、测试都不受影响）"""
    lv = _level()
    for dx, t, prev in ((0.02, 0.99, None), (0.20, 0.95, None), (0.20, 0.95, "side"),
                        (0.30, 0.20, None), (0.13, 0.99, "forward")):
        px = W / 2.0 + dx * W
        assert (lv._zone_at_pixel(px, t * H, W, H, prev)
                == NG.zone_at_pixel(px, t * H, W, H, prev)), (dx, t, prev)
    m1, m2 = lv._zone_masks(320, 200), NG.zone_masks(320, 200)
    assert set(m1) == set(m2)
    for k in m1:
        assert np.array_equal(m1[k], m2[k]), k


def test_runtime_zone_constant_override_still_works():
    """运行时改 ZONE_* 必须**仍然能改参**（tools/record_run.py --tune 就靠这个）

    2026-09-25 把判据抽成纯函数时最容易踩的坑：把常量改成 import 期捕获/默认参数
    ⇒ 改参**静默失效**（本仓库反复踩过的那类）。这里照 record_run 的手法改一遍，
    要求掩膜与判档都跟着变。
    """
    lv = _level()
    old = (NG.ZONE_GREEN_TOP, NG.ZONE_GREEN_BOT, NG.ZONE_BLUE_HEIGHT)
    try:
        NG.ZONE_GREEN_TOP, NG.ZONE_GREEN_BOT, NG.ZONE_BLUE_HEIGHT = \
            0.0329, 0.0719, 0.60
        lv._region_cache = None
        m = lv._zone_masks(320, 200)
        blue_top = int(np.argmax(m["blue"].any(axis=1)))
        assert blue_top == int(round((1.0 - 0.60) * 200)), \
            f"蓝区上沿没跟着 ZONE_BLUE_HEIGHT 走（{blue_top}）"
        # 走廊收窄后，同样的横向偏移在近处也不再算直行
        assert NG.zone_at_pixel(W / 2.0 + 0.09 * W, 0.99 * H, W, H) == "side"
        m2 = lv._zone_masks(320, 200)          # 第二次取缓存，不该变
        assert np.array_equal(m["blue"], m2["blue"])
    finally:
        NG.ZONE_GREEN_TOP, NG.ZONE_GREEN_BOT, NG.ZONE_BLUE_HEIGHT = old


def test_point_classifier_agrees_with_masks():
    """逐点判档 == 掩膜标签（绿/紫→直行、蓝→横移、橙→旋转），逐像素对齐"""
    w, h = 320, 240
    lab = _label_grid(NG.zone_masks(w, h))
    ties = 0
    for y in range(0, h, 2):
        for x in range(0, w, 2):
            want = _zone_from_label(lab[y, x])
            got = NG.zone_at_pixel(x, y, w, h)
            if got == want:
                continue
            # 恰好压在边界线上的像素：掩膜是 float32 栅格、判档是 float64
            # （差 ~1e-7），允许这一点点"贴线"差异，但不许是分块判错。
            assert _boundary_gap_px(x, y, w, h) < 1e-3, \
                f"({x},{y}) 判档 {got} != 掩膜 {want}，且不贴边界线"
            ties += 1
    assert ties <= 4, f"贴线容差用得太多（{ties} 个），多半是判据与掩膜真的走散了"


def test_zone_lines_separate_the_zones():
    """画出来的每一段都**压在**分区边界上：段两侧 ±4px 必须分属不同区块"""
    w, h = 640, 480
    lab = _label_grid(NG.zone_masks(w, h))
    lines = NG.zone_lines()
    assert len(lines) == 7, f"缺省参数下应有 7 段（绿2+紫1+蓝上沿2+蓝外沿2），实为 {len(lines)}"
    bad = []
    for (a, b) in lines:
        ax, ay = a[0] * w, a[1] * h
        bx, by = b[0] * w, b[1] * h
        L = float(np.hypot(bx - ax, by - ay))
        nx, ny = -(by - ay) / L, (bx - ax) / L        # 单位法向
        for s in np.linspace(0.1, 0.9, 21):
            cx, cy = ax + (bx - ax) * s, ay + (by - ay) * s
            sides = []
            for sgn in (1.0, -1.0):
                px = int(round(cx + sgn * nx * 4.0))
                py = int(round(cy + sgn * ny * 4.0))
                px = min(max(px, 0), w - 1)
                py = min(max(py, 0), h - 1)
                sides.append(int(lab[py, px]))
            if sides[0] == sides[1]:
                bad.append(((round(a[0], 4), round(a[1], 4),
                             round(b[0], 4), round(b[1], 4)), round(s, 2), sides))
    assert not bad, f"有线段没有压在边界上（两侧同区）: {bad[:3]}"


def test_zone_lines_cover_every_boundary():
    """反向：掩膜里**每一处**标签跳变都要被某一段线覆盖（不许漏画）"""
    w, h = 640, 480
    lab = _label_grid(NG.zone_masks(w, h))
    pts = []
    ys, xs = np.nonzero(lab[:, 1:] != lab[:, :-1])       # 左右相邻不同 → 竖边界
    if xs.size:
        pts.append(np.stack([xs + 0.5, ys.astype(float)], axis=1))
    ys, xs = np.nonzero(lab[1:, :] != lab[:-1, :])       # 上下相邻不同 → 横边界
    if xs.size:
        pts.append(np.stack([xs.astype(float), ys + 0.5], axis=1))
    pts = np.concatenate(pts, axis=0)
    assert pts.shape[0] > 500, "边界点太少，掩膜多半没算对"
    dist = np.full(pts.shape[0], np.inf)
    for (a, b) in NG.zone_lines():
        dist = np.minimum(dist, _seg_dist(pts, (a[0] * w, a[1] * h),
                                          (b[0] * w, b[1] * h)))
    worst = pts[int(np.argmax(dist))]
    assert float(dist.max()) <= 2.0, \
        f"有边界没被分割线覆盖：最远 {dist.max():.2f}px，位置 {worst}"


def test_zone_lines_are_normalized_and_degenerate_safe():
    """线是归一化坐标（0~1），且参数退化时只跳过对应段、不抛异常"""
    for seg in NG.zone_lines():
        for q in seg:
            assert 0.0 <= q[0] <= 1.0 and 0.0 <= q[1] <= 1.0
    old = (NG.ZONE_BLUE_HEIGHT, NG.ZONE_PURPLE_HEIGHT)
    try:
        NG.ZONE_BLUE_HEIGHT, NG.ZONE_PURPLE_HEIGHT = 0.0, 0.0    # 两块都退化成空
        lines = NG.zone_lines()
        assert len(lines) == 2, f"只该剩绿走廊两条边，实为 {len(lines)}"
        for seg in lines:
            for q in seg:
                assert 0.0 <= q[0] <= 1.0 and 0.0 <= q[1] <= 1.0
        NG.ZONE_BLUE_HEIGHT, NG.ZONE_PURPLE_HEIGHT = 1.5, 1.2    # 越界也走同一条路
        assert len(NG.zone_lines()) == 2
    finally:
        NG.ZONE_BLUE_HEIGHT, NG.ZONE_PURPLE_HEIGHT = old
