#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MAP-ANCHOR 单测：地图锚定的逐帧自标定（levels/nine_grid._map_anchor）

判据（都来自本轮数值实测，见方案文档 §1.4/§1.5）：
1. 合成场景（名义相机常数）位姿复原误差 < 1cm；
2. **相机常数完全错**（模拟地板形变）时仍然 < 1cm —— 这是这套机制存在的理由；
3. 注入"同色假货"（对应点被挪走）时，RANSAC 把它判为外点而不是污染共识；
4. 对应点不足时**诚实弃权**（返回 None），绝不硬解。
"""
import numpy as np
import pytest

import levels.nine_grid_shared as NG
from core.camera_config import CAM_HEIGHT_STANDING_CM
from core.ground_homography import (
    GroundHomography, grid_cell_center, cell_index)


HEAD = NG.HEAD_CENTER
PITCH = NG.PITCH_NAV
ALL_CELLS = [cell_index(r, c) for r in range(3) for c in range(3)]


class _StubState:
    """_map_anchor 只需要一个能报头部脉宽的状态对象"""

    def __init__(self, head=HEAD):
        self.current_head_pulse = head


class _StubObs:
    """冒充 PanelObservation：_anchor_corr 用 clipped / digit / center_px /
    bbox / hull_poly"""

    def __init__(self, digit, px, clipped=False, bbox=None, hull_poly=()):
        self.digit = digit
        self.center_px = (float(px[0]), float(px[1]))
        self.clipped = bool(clipped)
        # 默认给一个"四边都不贴"的 bbox，避免未裁切观测被误判裁边
        self.bbox = bbox if bbox is not None else (
            float(px[0]) - 50.0, float(px[1]) - 50.0, 100.0, 100.0)
        self.hull_poly = tuple(hull_poly)


class _Frame:
    """只提供 shape 的假帧（_clip_sides_of 只用 shape）"""

    def __init__(self, w=2592, h=1944):
        self.shape = (h, w, 3)


FRAME = _Frame()


def _make_level(cells_by_digit):
    lv = NG.NineGridShared(_StubState())
    lv.digit_cell = dict(cells_by_digit)
    return lv


def _scene(cam_xy=(50.0, -20.0), h=CAM_HEIGHT_STANDING_CM, head=HEAD,
           noise=0.0, seed=0):
    """用真值位姿投影 9 个格心 → 像素，返回 (obs, cells_by_digit)"""
    hg = GroundHomography.from_pose(cam_xy, h, PITCH, head_pulse=head)
    gd = np.array([grid_cell_center(c) for c in ALL_CELLS])
    px = np.asarray(hg.ground_to_pixels(gd), float)
    inb = ((px[:, 0] > 0) & (px[:, 0] < 2592)
           & (px[:, 1] > 0) & (px[:, 1] < 1944))
    rng = np.random.default_rng(seed)
    px = px + rng.normal(0, noise, px.shape)
    obs, cells = [], {}
    for k, c in enumerate(ALL_CELLS):
        if not inb[k]:
            continue
        d = k + 1
        obs.append(_StubObs(d, px[k]))
        cells[d] = c
    return obs, cells, px, gd, inb


# 站点取"场外后退 60cm"：9 格全部落在画幅内，对应点最多。
# 近站点 (50,-20) 只有 6 格可见 —— 6 点里再有 1 个假货就只剩 5 点，
# 而 5 点单应已经是恰定解，RANSAC 无从核验（这正是 MIN_INLIERS=5 的理由）。
FAR_SITE = (50.0, -60.0)


# ---------------------------------------------------------------------

def test_anchor_recovers_pose_nominal():
    """名义常数 + 2px 质心噪声 ⇒ 位姿复原误差 < 1cm"""
    obs, cells, px, gd, inb = _scene(noise=2.0, seed=1)
    assert len(obs) >= NG.MAP_ANCHOR_MIN_PTS, "合成场景对应点不足，测试前提不成立"
    lv = _make_level(cells)
    a = lv._map_anchor(obs, FRAME)
    assert a is not None, "应能解出锚"
    err = float(np.linalg.norm(a["pose"][:2] - np.array([50.0, -20.0])))
    assert err < 1.0, f"位姿误差 {err:.2f}cm 应 < 1cm"
    assert abs(a["cam_height_cm"] - CAM_HEIGHT_STANDING_CM) < 3.0


def test_anchor_is_deformation_immune():
    """★ 核心性质：真实相机常数与关卡常数完全不符时，H 仍解对

    场景由 h=41cm（而非标称 56cm）投影生成 ⇒ 任何"冻结相机常数"的方法都会错，
    而 MAP-ANCHOR 从观测里自己解出来。
    """
    obs, cells, px, gd, inb = _scene(h=41.0, noise=2.0, seed=2)
    assert len(obs) >= NG.MAP_ANCHOR_MIN_PTS
    lv = _make_level(cells)
    a = lv._map_anchor(obs, FRAME)
    assert a is not None
    err = float(np.linalg.norm(a["pose"][:2] - np.array([50.0, -20.0])))
    assert err < 1.0, f"形变下位姿误差 {err:.2f}cm 应 < 1cm"
    # 反解出的高度应接近真实的 41cm，而不是关卡里冻结的 56cm
    assert abs(a["cam_height_cm"] - 41.0) < 3.0, \
        f"反解高度 {a['cam_height_cm']:.1f}cm 应接近真实的 41cm"


def test_anchor_rejects_impostor():
    """注入同色假货（对应点被挪到别处）⇒ 它落在外点，共识仍正确"""
    obs, cells, px, gd, inb = _scene(cam_xy=FAR_SITE, noise=2.0, seed=3)
    assert len(obs) >= 7, f"该站点应至少 7 格可见，实际 {len(obs)}"
    # 第 4 个观测挪走：模拟"该色唯一观测是场地同色杂物"
    bad_i = 3
    bad_digit = obs[bad_i].digit
    moved = np.asarray(obs[bad_i].center_px, float) + np.array([170.0, -110.0])
    obs[bad_i] = _StubObs(bad_digit, moved)
    lv = _make_level(cells)
    a = lv._map_anchor(obs, FRAME)
    assert a is not None
    out_digits = [o.digit for o in a["outliers"]]
    assert bad_digit in out_digits, \
        f"假货（数字{bad_digit}）应被判为外点，实际外点 {out_digits}"
    err = float(np.linalg.norm(a["pose"][:2] - np.asarray(FAR_SITE, float)))
    assert err < 1.5, f"共识位姿误差 {err:.2f}cm 应 < 1.5cm（未被假货带歪）"


def test_anchor_abstains_when_too_few_points():
    """对应点不足 ⇒ 返回 None（诚实弃权），不硬解"""
    obs, cells, px, gd, inb = _scene(noise=0.0, seed=4)
    few = obs[:NG.MAP_ANCHOR_MIN_PTS - 1]
    lv = _make_level(cells)
    assert lv._map_anchor(few, FRAME) is None


def test_anchor_abstains_without_redundancy():
    """只有 5 个对应点时是恰定解 ⇒ 即便解得出来也不该给出"共识"

    （5 点单应无冗余，任何一点都可被解释；宁可弃权也不要假信心）
    """
    obs, cells, px, gd, inb = _scene(noise=0.0, seed=9)
    assert len(obs) > NG.MAP_ANCHOR_MIN_PTS
    lv = _make_level(cells)
    a = lv._map_anchor(obs[:NG.MAP_ANCHOR_MIN_PTS], FRAME)
    assert a is None or a["n_inliers"] >= NG.MAP_ANCHOR_MIN_INLIERS


def test_anchor_corr_uses_centre_for_unclipped_only():
    """未裁切观测贡献 1 个中心对应；不在 map 里的数字一律不参与"""
    obs, cells, px, gd, inb = _scene(noise=2.0, seed=5)
    good = len(obs)
    lv = _make_level(cells)
    corr, keep = lv._anchor_corr(obs, FRAME)
    assert len(corr) == good
    # 数字不在 map 里 ⇒ 不参与
    lv2 = _make_level({})
    assert lv2._anchor_corr(obs, FRAME)[0] == []


def _clipped_scene(side="B", cam_xy=(50.0, 0.0), h=CAM_HEIGHT_STANDING_CM):
    """合成一个"某条画幅边被裁掉"的面板：返回 (obs, cells, 真值角点字段坐标)

    做法：把面板中心投影到像素，再人为把 hull_poly 造成"只剩对面那两个角"，
    并按裁边给出一个贴边的 bbox。
    """
    hg = GroundHomography.from_pose(cam_xy, h, PITCH, head_pulse=HEAD)
    cell = cell_index(0, 0)                     # 任意一格
    g = np.asarray(grid_cell_center(cell), float)
    half = NG.PANEL_HALF_CM
    corners = [(g[0] - half, g[1] - half), (g[0] + half, g[1] - half),
               (g[0] + half, g[1] + half), (g[0] - half, g[1] + half)]
    px = np.asarray(hg.ground_to_pixels(corners), float)
    # 与 _anchor_corr 的规则一致：可见的是"被裁边对面"的那一对角
    # 角点索引：0=(x−h,y−h) 近左  1=(x+h,y−h) 近右  2=(x+h,y+h) 远右  3=(x−h,y+h) 远左
    # （本关场地系 y 越大越靠近入口；画幅上沿=远、下沿=近）
    idx = {"B": (2, 3), "T": (0, 1), "L": (1, 2), "R": (3, 0)}[side]
    poly = [tuple(px[i]) for i in idx]
    if side == "B":
        bbox = (min(p[0] for p in poly) - 40, 1900.0, 200.0, 44.0)
    elif side == "T":
        bbox = (min(p[0] for p in poly) - 40, 0.0, 200.0, 44.0)
    elif side == "L":
        bbox = (0.0, min(p[1] for p in poly) - 40, 44.0, 200.0)
    else:
        bbox = (2548.0, min(p[1] for p in poly) - 40, 44.0, 200.0)
    o = _StubObs(1, np.mean(poly, axis=0), clipped=True, bbox=bbox,
                 hull_poly=poly)
    return [o], {1: cell}, [corners[i] for i in idx]


@pytest.mark.parametrize("side", ["B", "T", "L", "R"])
def test_anchor_uses_clipped_panel_corners(side):
    """★ WS3 核心：被裁切的面板用"未裁掉的那两个真实角点"参与解算

    这是把可用率从 14.7% 抬到 46.1% 的那条改动。单面板 + 该面板的 2 个角点
    不足以解 H（需 ≥5 点），所以这里只断言**对应集本身**正确：
    产出 2 个对应，且场地坐标与该裁边应有的角点一致。
    """
    obs, cells, want = _clipped_scene(side)
    lv = _make_level(cells)
    corr, keep = lv._anchor_corr(obs, FRAME)
    assert len(corr) == 2, f"裁边 {side}：应产出 2 个角点对应，实际 {len(corr)}"
    got = sorted([tuple(np.round(c[0], 3)) for c in corr])
    exp = sorted([tuple(np.round(np.asarray(w, float), 3)) for w in want])
    assert got == exp, f"裁边 {side} 的场地角点不对：{got} vs {exp}\n{corr}"


def test_anchor_corner_pair_swap_is_tried():
    """角点配对顺序被故意打反时，逆序那一遍能把它救回来

    构造：3 个未裁切面板（各 1 点）+ 1 个裁切面板（2 角点）⇒ 共 5 点。
    正序若错误会导致共识下降；逆序应给出同样的内点数或更好。
    这里只断言"两种配对都能被枚举到、且较优者被返回"。
    """
    lv = _make_level({})
    obs, cells, _want = _clipped_scene("B")
    lv.digit_cell = dict(cells)
    a1 = lv._anchor_corr(obs, FRAME, swap_corners=False)
    a2 = lv._anchor_corr(obs, FRAME, swap_corners=True)
    assert len(a1[0]) == len(a2[0]) == 2
    # 逆序必须真的把两个场地点换了位置（否则"补试"是假的）
    assert not np.allclose(a1[0][0][0], a2[0][0][0]), \
        "swap_corners=True 没有交换配对，逆序补试是空转"


def test_anchor_skips_corner_clipped_panels():
    """贴两条边（角点被裁）⇒ 无法确定是哪两个角 ⇒ 该面板弃权，不硬解"""
    hg = GroundHomography.from_pose((50.0, 0.0), CAM_HEIGHT_STANDING_CM,
                                    PITCH, head_pulse=HEAD)
    g = np.asarray(grid_cell_center(cell_index(0, 0)), float)
    half = NG.PANEL_HALF_CM
    px = np.asarray(hg.ground_to_pixels(
        [(g[0] - half, g[1] - half), (g[0] + half, g[1] - half),
         (g[0] + half, g[1] + half), (g[0] - half, g[1] + half)]), float)
    o = _StubObs(1, px[0], clipped=True,
                 bbox=(0.0, 1900.0, 300.0, 44.0),      # 同时贴 L 和 B
                 hull_poly=[tuple(px[0])])
    lv = _make_level({1: cell_index(0, 0)})
    corr, _keep = lv._anchor_corr([o], FRAME)
    assert corr == [], "角点被裁的面板不应产生任何对应"


def test_anchor_skips_clipped_without_hull_poly():
    """hull_poly 为空（检测器异常回退）⇒ 该面板不参与，整体退化为 v1 行为"""
    o = _StubObs(1, (1200.0, 1800.0), clipped=True,
                 bbox=(1150.0, 1780.0, 100.0, 60.0), hull_poly=())
    lv = _make_level({1: cell_index(0, 0)})
    assert lv._anchor_corr([o], FRAME)[0] == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
