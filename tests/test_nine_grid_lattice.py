# -*- coding: utf-8 -*-
"""格阵拟合与布局扫自标定单元测试（tests/test_nine_grid_lattice.py）

背景（2026-09-11 接手勘察，三个 P0 缺陷之二、之三）：
- `lattice_assign` 曾枚举 D4 **八个**朝向（含 4 个镜像）。机器人系与场地系
  同为右手系（x 右 / y 前 / z 上），格阵基 e2 = CCW90(e1) 也是右手构造，
  故"格阵整数坐标 → 场地格号"的真解必为真旋转；镜像候选却能通过旧的全部
  约束并被选中 → 布局被转置 + `_pose_bootstrap` 崩（"光轴无向下分量"）。
  真实照片实测分离度：真旋转 RMS 0.62cm vs 镜像 18.75cm。
- `_lattice_grid_fit` 的参数网格"多数票不足"分支返回 2 元组而调用方解包
  4 元组 → 该分支必崩（本该是"前进一步重扫"）。

本测试锁定：
1. 物理生成器（真旋转 + 噪声）下 **从不出错**（歧义可接受），入口范围内
   正确率 ≥90%；
2. 候选集里**只有真旋转族**（镜像族不得出现）；
3. 覆盖门：只有 6 个数字的观测 → 判失败（整函数返回 cells=None）；
4. 参数网格自标定 + 位姿自举：从合成像素观测恢复布局、(偏移, 高度) 与位姿；
5. 裁切观测偏差（注入 ~6cm）被降权吸收，布局仍正确。

运行：python tests/test_nine_grid_lattice.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # 仓库根

import numpy as np

from core.camera_config import (
    CAMERA_WIDTH, CAMERA_HEIGHT, HEAD_CENTER,
)
from core.ground_homography import grid_cell_center, cell_index, GRID_CELL_CM
from levels.nine_grid_shared import (
    NineGridShared, lattice_assign, PITCH_NAV, PITCH_DOWN,
    LATTICE_RMS_MAX_CM,
)
from sim.nine_grid_sim import SIM_LAYOUT

TRUTH = {d: c for c, d in SIM_LAYOUT.items() if d is not None}
_TRANSFORMS = [("id", lambda r, c: (r, c)), ("rot90", lambda r, c: (c, 2 - r)),
               ("rot180", lambda r, c: (2 - r, 2 - c)),
               ("rot270", lambda r, c: (2 - c, r)),
               ("flipLR", lambda r, c: (r, 2 - c)),
               ("flipUD", lambda r, c: (2 - r, c)),
               ("transpose", lambda r, c: (c, r)),
               ("antitrans", lambda r, c: (2 - c, 2 - r))]
PROPER = {"id", "rot90", "rot180", "rot270"}


def _variants(layout):
    """layout(digit->cell) 的全部 D4 变体：{名称: layout'}"""
    out = {}
    for name, fn in _TRANSFORMS:
        cells = {d: fn(*divmod(c, 3)) for d, c in layout.items()}
        idx = [a * 3 + b for a, b in cells.values()]
        if len(set(idx)) == len(idx):
            out[name] = {d: a * 3 + b for d, (a, b) in cells.items()}
    return out


def _family_of(cells):
    """cells 相对真值属于哪个 D4 变体（None = 不是任何变体）"""
    for name, lay in _variants(TRUTH).items():
        if lay == cells:
            return name
    return None


def _synth_points(rng, x, y, bearing_deg, sigma=3.0):
    """物理生成器：机器人网格位置 (x,y)、朝向 ±45°、观测噪声 N(0,sigma)"""
    th = np.radians(bearing_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return {d: R @ (grid_cell_center(c) - np.array([x, y]))
            + rng.normal(0, sigma, 2)
            for d, c in TRUTH.items()}


def test_physical_never_wrong():
    """物理生成器：0 错判；入口范围内正确率 ≥90%"""
    rng = np.random.RandomState(11)
    n = ok = amb = wrong = 0
    for _ in range(300):
        pts = _synth_points(rng, rng.uniform(25, 75), rng.uniform(-45, -5),
                            rng.uniform(-45, 45))
        cells, info, _ranked = lattice_assign(pts, clean=set(pts))
        n += 1
        if cells is None:
            amb += 1
        elif cells == TRUTH:
            ok += 1
        else:
            wrong += 1
            if wrong <= 2:
                print(f"    错判样例: {cells}（{info}）")
    assert wrong == 0, f"物理生成器出现 {wrong}/{n} 错判（必须为 0）"
    assert ok / n >= 0.90, f"入口范围内正确率 {ok}/{n} < 90%"
    print(f"  物理生成器 {n} 组：正确 {ok}、歧义 {amb}、错判 {wrong} ✓")


def test_only_proper_family_returned():
    """候选集只允许真旋转族：镜像族（transpose 等）不得出现"""
    rng = np.random.RandomState(12)
    seen, improper = set(), 0
    for _ in range(120):
        pts = _synth_points(rng, rng.uniform(25, 75), rng.uniform(-45, -5),
                            rng.uniform(-45, 45))
        cells, _info, ranked = lattice_assign(pts, clean=set(pts))
        for cand in ranked:
            fam = _family_of(cand["cells"])
            seen.add(fam)
            if fam is not None and fam not in PROPER:
                improper += 1
    assert improper == 0, f"候选集出现 {improper} 个镜像族（手性门失效）"
    assert "id" in seen, f"候选集竟无真值族（seen={sorted(map(str, seen))}）"
    print(f"  候选族分布: {sorted(str(s) for s in seen)}（镜像族 0 个）✓")


def test_too_few_points_refused():
    """可见数字 <4 无法定朝向：必须拒绝而不是猜"""
    pts = {d: grid_cell_center(c) for d, c in list(TRUTH.items())[:3]}
    cells, info, ranked = lattice_assign(pts)
    assert cells is None and ranked == [], f"3 点竟给出布局 {cells}"
    print(f"  3 点输入被拒：{info} ✓")


def _synth_pix_obs(x, y, bearing_deg, layout=TRUTH, off_deg=22.0, h_cm=50.0,
                   bias_cm=None, bias_digit=None, margin_px=12.0):
    """合成像素观测：用关卡自身的正向投影生成（与拟合侧模型自洽）

    bias_cm/bias_digit：把该数字的面板中心在地面平移 bias_cm（模拟裁切质心
    偏差），并把该数字全部观测标为 clipped（走降权路径）。
    """
    from levels.nine_grid_shared import project_ground_to_pixel
    th = np.radians(bearing_deg)
    shift = np.array([0.0, bias_cm or 0.0])
    obs = []
    for pitch in (PITCH_NAV, PITCH_DOWN):
        for head in (HEAD_CENTER, 1950, 1050, 2200, 800):
            for d, c in layout.items():
                g = grid_cell_center(c)
                if bias_cm and d == bias_digit:
                    g = g + shift
                px = project_ground_to_pixel(g, x, y, th, pitch, head,
                                             off_deg, h_cm)[0]
                clipped = not (margin_px <= px[0] <= CAMERA_WIDTH - margin_px
                               and margin_px <= px[1] <= CAMERA_HEIGHT - margin_px)
                if clipped:
                    continue      # 只用未裁切观测（裁切偏差由专门用例覆盖）
                obs.append((pitch, head, d, px, False))
    return obs


def test_self_calibration_and_pose_bootstrap():
    """合成像素观测：自标定出布局 + (偏移, 高度)，位姿自举回到真值"""
    level = NineGridShared(None)
    x, y, brg = 50.0, -20.0, 0.0
    obs = _synth_pix_obs(x, y, brg)
    assert len({e[2] for e in obs}) == 7, "合成观测应覆盖 7 个数字"
    fit = level._lattice_grid_fit(obs)
    assert fit.cells == TRUTH, f"布局解算错误: {fit.cells} != {TRUTH}（{fit.info}）"
    assert abs(fit.offset_deg - 22.0) <= 5.0, \
        f"自标定偏移 {fit.offset_deg:.1f}° 偏离真值 22° 超 5°"
    assert abs(fit.cam_height_cm - 50.0) <= 7.5, \
        f"自标定高度 {fit.cam_height_cm:.1f}cm 偏离真值 50cm 超一个网格步"
    assert level._pose_bootstrap(fit), "位姿自举失败"
    err_xy = float(np.hypot(level.pose[0] - x, level.pose[1] - y))
    err_th = abs(np.degrees(level.pose[2] - np.radians(brg)))
    assert err_xy < 2.0 and err_th < 2.0, \
        f"位姿自举误差 {err_xy:.2f}cm / {err_th:.2f}° 超限"
    print(f"  自标定: 偏移 {fit.offset_deg:+.1f}°（真值 22°）、高度 "
          f"{fit.cam_height_cm:.0f}cm（真值 50cm）；布局正确；位姿误差 "
          f"{err_xy:.2f}cm/{err_th:.2f}° ✓")


def test_coverage_requires_all_digits():
    """只有 6 个数字的观测：整函数判失败（不允许返回残缺布局）"""
    level = NineGridShared(None)
    obs = [e for e in _synth_pix_obs(50.0, -20.0, 0.0) if e[2] != 4]
    fit = level._lattice_grid_fit(obs)
    assert fit.cells is None, f"缺 1 个数字竟给出布局 {fit.cells}"
    assert level._pose_bootstrap(fit) is False, "残缺拟合不应通过位姿自举"
    print(f"  6 数字观测被拒：{fit.info} ✓")


def test_clipped_bias_downweighted():
    """单块板只有带 ~6cm 偏差的裁切观测：降权后布局仍正确"""
    level = NineGridShared(None)
    obs = _synth_pix_obs(50.0, -20.0, 0.0)
    biased = [(p, h, 4, px, True) for (p, h, d, px, _c) in obs if d == 4]
    others = [e for e in obs if e[2] != 4]
    assert biased, "测试前提：数字4 应有观测"
    # 裁切质心偏差：把数字4 的观测像素按 +6cm 地面位移重新投影
    biased = _synth_pix_obs(50.0, -20.0, 0.0, bias_cm=6.0, bias_digit=4,
                            margin_px=-1e9)      # margin 负值 → 全部保留
    biased = [e for e in biased if e[2] == 4]
    biased = [(p, h, d, px, True) for (p, h, d, px, _c) in biased]
    fit = level._lattice_grid_fit(others + biased)
    assert fit.cells == TRUTH, f"裁切偏差导致布局错误: {fit.cells}（{fit.info}）"
    assert 4 in fit.weights and fit.weights[4] < 1.0, \
        "仅裁切观测的数字应被降权"
    assert any("贴边观测" in w for w in fit.warnings), \
        f"应给出贴边（裁切）观测告警：{fit.warnings}"
    print(f"  裁切偏差容忍：权重 {fit.weights[4]:.1f}，布局仍正确 ✓")


def test_outlier_never_yields_wrong_layout():
    """假阳/大偏差观测：只允许"拒绝"或"仍然正确"，绝不允许给出错布局

    实测（2026-09-11）：单块板注入偏差 ≤0.6 格距（20cm）时，参数网格会**吸收**
    它（(偏置, 高度) 沿退化谷移动，实测 22°/50cm → 25~29°/53~56cm），布局仍是
    真值——好处是鲁棒，代价是自标定常数被带偏几度/几厘米；≥0.9 格距（30cm）
    时覆盖门/RMS 门直接拒绝（cells=None）。
    """
    level = NineGridShared(None)
    obs = _synth_pix_obs(50.0, -20.0, 0.0)
    from levels.nine_grid_shared import project_ground_to_pixel
    for bias in (10.0, 20.0, 30.0, 40.0, 50.0):
        fake = []
        for (pitch, head, d, _px, _c) in obs:
            if d != 7:
                continue
            g = grid_cell_center(TRUTH[7]) + np.array([0.0, bias])
            px = project_ground_to_pixel(g, 50.0, -20.0, 0.0, pitch, head,
                                         22.0, 50.0)[0]
            fake.append((pitch, head, d, px, False))
        fit = level._lattice_grid_fit([e for e in obs if e[2] != 7] + fake)
        assert fit.cells is None or fit.cells == TRUTH, \
            f"注入 {bias:.0f}cm 偏差后给出错布局: {fit.cells}"
        if bias >= 30.0:
            assert fit.cells is None, \
                f"注入 {bias:.0f}cm（≥0.9 格距）偏差应被拒绝，实际 {fit.cells}"
    print("  假阳/大偏差：≤20cm 被吸收（仍正确）、≥30cm 被拒，从不错判 ✓")


def test_grid_convention():
    """格号约定：格6=入口视角左下（恒空），格0=远排左；cell_index 自洽"""
    assert cell_index(2, 0) == 6 and cell_index(0, 0) == 0
    assert cell_index(0, 2) == 2 and cell_index(2, 2) == 8
    c6 = grid_cell_center(6)
    c0 = grid_cell_center(0)
    assert np.allclose(c6, [GRID_CELL_CM * 0.5, GRID_CELL_CM * 0.5]), c6
    assert np.allclose(c0, [GRID_CELL_CM * 0.5, GRID_CELL_CM * 2.5]), c0
    print(f"  格号约定：格6={np.round(c6, 1)}（左下）、格0={np.round(c0, 1)}"
          "（远排左）✓")


if __name__ == "__main__":
    test_grid_convention()
    test_too_few_points_refused()
    test_physical_never_wrong()
    test_only_proper_family_returned()
    test_self_calibration_and_pose_bootstrap()
    test_coverage_requires_all_digits()
    test_clipped_bias_downweighted()
    test_outlier_never_yields_wrong_layout()
    print(f"全部格阵拟合测试通过 ✓（干净子集 RMS 门 {LATTICE_RMS_MAX_CM:.0f}cm）")
