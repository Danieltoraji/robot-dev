# -*- coding: utf-8 -*-
"""格阵拟合与布局扫自标定单元测试（tests/test_nine_grid_lattice.py）

背景（2026-09-11 接手勘察，三个 P0 缺陷之二、之三）：
- `assign_grid_cells` 曾枚举 D4 **八个**朝向（含 4 个镜像）。机器人系与场地系
  同为右手系（x 右 / y 前 / z 上），格阵基 e2 = CCW90(e1) 也是右手构造，
  故"格阵整数坐标 → 场地格号"的真解必为真旋转；镜像候选却能通过旧的全部
  约束并被选中 → 布局被转置 + `_pose_from_panels` 崩（"光轴无向下分量"）。
  真实照片实测分离度：真旋转 RMS 0.62cm vs 镜像 18.75cm。
- `_fit_grid` 的参数网格"多数票不足"分支返回 2 元组而调用方解包
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
    NineGridShared, assign_grid_cells, PITCH_NAV, PITCH_DOWN,
    GRID_FIT_RMS_MAX_CM,
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
        cells, info, _ranked = assign_grid_cells(pts, clean=set(pts))
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
        cells, _info, ranked = assign_grid_cells(pts, clean=set(pts))
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
    cells, info, ranked = assign_grid_cells(pts)
    assert cells is None and ranked == [], f"3 点竟给出布局 {cells}"
    print(f"  3 点输入被拒：{info} ✓")


def _synth_pix_obs(x, y, bearing_deg, layout=TRUTH, off_deg=22.0, h_cm=None,
                   bias_cm=None, bias_digit=None, margin_px=12.0,
                   off_by_pitch=None):
    """合成像素观测：用关卡自身的正向投影生成（与拟合侧模型自洽）

    缺省 h_cm = 卷尺实测高度（33.9cm）——**与生产缺省一致**：否则合成的"真值
    高度"会落在 `_fit_grid` 的可采纳带（实测值 ±8cm）之外，自标定会被正确地
    拒绝（见 test_selfcal_height_falls_back_to_measured）。
    bias_cm/bias_digit：把该数字的面板中心在地面平移 bias_cm（模拟裁切质心
    偏差），并把该数字全部观测标为 clipped（走降权路径）。
    off_by_pitch：{pitch: 安装偏移}——模拟"换档位后偏移不同"（真实机器人就是
    这样：pitch1200 与 pitch1040 的有效偏移不一样）；缺省全部用 off_deg。
    """
    from levels.nine_grid_shared import project_ground_to_pixel
    if h_cm is None:
        from levels.nine_grid_shared import measured_cam_height_cm
        h_cm = measured_cam_height_cm()
    th = np.radians(bearing_deg)
    shift = np.array([0.0, bias_cm or 0.0])
    obs = []
    for pitch in (PITCH_NAV, PITCH_DOWN):
        off_p = off_deg if not off_by_pitch else off_by_pitch.get(pitch, off_deg)
        for head in (HEAD_CENTER, 1950, 1050, 2200, 800):
            for d, c in layout.items():
                g = grid_cell_center(c)
                if bias_cm and d == bias_digit:
                    g = g + shift
                px = project_ground_to_pixel(g, x, y, th, pitch, head,
                                             off_p, h_cm)[0]
                clipped = not (margin_px <= px[0] <= CAMERA_WIDTH - margin_px
                               and margin_px <= px[1] <= CAMERA_HEIGHT - margin_px)
                if clipped:
                    continue      # 只用未裁切观测（裁切偏差由专门用例覆盖）
                obs.append((pitch, head, d, px, False))
    return obs


def test_self_calibration_and_pose_from_panels():
    """合成像素观测：自标定出布局 + (偏移, 高度)，位姿自举回到真值"""
    from levels.nine_grid_shared import measured_cam_height_cm
    level = NineGridShared(None)
    x, y, brg = 50.0, 15.0, 0.0
    obs = _synth_pix_obs(x, y, brg)
    assert len({e[2] for e in obs}) == 7, "合成观测应覆盖 7 个数字"
    fit = level._fit_grid(obs)
    assert fit.cells == TRUTH, f"布局解算错误: {fit.cells} != {TRUTH}（{fit.info}）"
    assert abs(fit.offset_deg - 22.0) <= 5.0, \
        f"自标定偏移 {fit.offset_deg:.1f}° 偏离真值 22° 超 5°"
    # 高度：合成场景的真值就是"实测高度"，所以无论走哪条路都应落在它附近
    h_true = measured_cam_height_cm()
    assert abs(fit.cam_height_cm - h_true) <= 2.0, \
        f"自标定高度 {fit.cam_height_cm:.1f}cm 偏离真值 {h_true:.1f}cm 超 2cm"
    assert level._pose_from_panels(fit), "位姿自举失败"
    err_xy = float(np.hypot(level.pose[0] - x, level.pose[1] - y))
    err_th = abs(np.degrees(level.pose[2] - np.radians(brg)))
    assert err_xy < 2.0 and err_th < 2.0, \
        f"位姿自举误差 {err_xy:.2f}cm / {err_th:.2f}° 超限"
    print(f"  自标定: 偏移 {fit.offset_deg:+.1f}°（真值 22°）、高度 "
          f"{fit.cam_height_cm:.1f}cm（真值 {h_true:.1f}cm，"
          f"{fit.height_source}）；布局正确；位姿误差 "
          f"{err_xy:.2f}cm/{err_th:.2f}° ✓")


def test_coverage_requires_all_digits():
    """只有 6 个数字的观测：整函数判失败（不允许返回残缺布局）"""
    level = NineGridShared(None)
    obs = [e for e in _synth_pix_obs(50.0, 15.0, 0.0) if e[2] != 4]
    fit = level._fit_grid(obs)
    assert fit.cells is None, f"缺 1 个数字竟给出布局 {fit.cells}"
    assert level._pose_from_panels(fit) is False, "残缺拟合不应通过位姿自举"
    print(f"  6 数字观测被拒：{fit.info} ✓")


def test_clipped_bias_downweighted():
    """单块板只有带 ~6cm 偏差的裁切观测：降权后布局仍正确"""
    level = NineGridShared(None)
    obs = _synth_pix_obs(50.0, 15.0, 0.0)
    biased = [(p, h, 4, px, True) for (p, h, d, px, _c) in obs if d == 4]
    others = [e for e in obs if e[2] != 4]
    assert biased, "测试前提：数字4 应有观测"
    # 裁切质心偏差：把数字4 的观测像素按 +6cm 地面位移重新投影
    biased = _synth_pix_obs(50.0, 15.0, 0.0, bias_cm=6.0, bias_digit=4,
                            margin_px=-1e9)      # margin 负值 → 全部保留
    biased = [e for e in biased if e[2] == 4]
    biased = [(p, h, d, px, True) for (p, h, d, px, _c) in biased]
    fit = level._fit_grid(others + biased)
    assert fit.cells == TRUTH, f"裁切偏差导致布局错误: {fit.cells}（{fit.info}）"
    assert 4 in fit.weights and fit.weights[4] < 1.0, \
        "仅裁切观测的数字应被降权"
    assert any("贴边观测" in w for w in fit.warnings), \
        f"应给出贴边（裁切）观测告警：{fit.warnings}"
    print(f"  裁切偏差容忍：权重 {fit.weights[4]:.1f}，布局仍正确 ✓")


def test_outlier_never_yields_wrong_layout():
    """假阳/大偏差观测：只允许"拒绝"或"仍然正确"，绝不允许给出错布局

    实测（2026-09-11，旧 36~70cm 网格）：单块板注入偏差 ≤0.6 格距（20cm）时，
    参数网格会**吸收**它（(偏置, 高度) 沿退化谷移动），布局仍是真值；≥0.9 格距
    （30cm）时覆盖门/RMS 门直接拒绝（cells=None）。

    ⚠️ 2026-09-28：**本用例不再钉"多少厘米必须被拒"**。两道门都动过：
      ① 高度网格从 36~70 改成"实测值 ± 几档"（吸收能力变强，门槛 30→40cm）；
      ② `LAYOUT_SCAN_VOTE_MIN_COMBOS` 从 3 降到 **1**（现场真机"得票 1/1"
         导致连扫三轮全废、一格都不走）——**票数下限本来就是"混淆面板"的
         主要防线**：注入偏差后往往只剩 1 个参数组合能凑出 7/7，降到 1
         就等于放它过去（本用例实测：40cm 注入从"被拒"变成"被吸收"）。
    所以这里只钉**安全契约**：任何偏差下都不允许给出"错布局"（cells 要么 None、
    要么等于真值）。量化的拒绝门槛要等"注入偏差 ↔ 覆盖门/RMS 门"重新标定。
    """
    level = NineGridShared(None)
    obs = _synth_pix_obs(50.0, 15.0, 0.0)
    from levels.nine_grid_shared import (
        measured_cam_height_cm, project_ground_to_pixel,
    )
    h_true = measured_cam_height_cm()
    rejected = []
    for bias in (10.0, 20.0, 30.0, 40.0, 50.0):
        fake = []
        for (pitch, head, d, _px, _c) in obs:
            if d != 7:
                continue
            g = grid_cell_center(TRUTH[7]) + np.array([0.0, bias])
            px = project_ground_to_pixel(g, 50.0, 15.0, 0.0, pitch, head,
                                         22.0, h_true)[0]
            fake.append((pitch, head, d, px, False))
        fit = level._fit_grid([e for e in obs if e[2] != 7] + fake)
        assert fit.cells is None or fit.cells == TRUTH, \
            f"注入 {bias:.0f}cm 偏差后给出错布局: {fit.cells}"
        if fit.cells is None:
            rejected.append(bias)
    assert rejected, "至少要有某个偏差档被拒（否则覆盖门/RMS 门形同虚设）"
    print(f"  假阳/大偏差：从不错判；被拒偏差档 {rejected} ✓")


def test_cell6_may_be_occupied():
    """★ 位置 6 **不是**恒空（2026-09-28 现场确认）：6 有面板也必须解对、且不告警

    旧代码把"格6 恒空"当规则硬约束、拿它给候选朝向排序/筛选，还会对"格6 被占"
    打"与比赛规则冲突"的告警。那条规则早已取消 ⇒ 本用例保证代码不再因此
    降级、告警或选错朝向。
    """
    level = NineGridShared(None)
    variants = []
    # ① 每一块板轮流挪进格6（格6 原本那块换到它原来的位置）
    for d in sorted(TRUTH):
        v = dict(TRUTH)
        old = v[d]
        holder = [x for x, c in v.items() if c == 6]
        v[d] = 6
        if holder:
            v[holder[0]] = old
        variants.append((f"数字{d} 挪进格6", v))
    # ② 格6 空着、两个空位都占满的布局（证明"格6 空"不再是隐含前提）
    v = {}
    empties = [c for c in range(9) if c not in set(TRUTH.values())]
    cells = [c for c in range(9) if c != 6]
    for d, c in zip(sorted(TRUTH), cells):
        v[d] = c
    variants.append(("格6 空、其余 8 格里占 7 格", v))
    n = 0
    for tag, layout in variants:
        if 6 not in layout.values():
            continue
        obs = _synth_pix_obs(50.0, 15.0, 0.0, layout=layout)
        if len({e[2] for e in obs}) != 7:
            continue
        fit = level._fit_grid(obs)
        assert fit.cells == layout, \
            f"{tag}：解出 {fit.cells} != 真值 {layout}（{fit.info}）"
        assert not any("格位 6" in w or "规则冲突" in w for w in fit.warnings), \
            f"{tag}：不应再对位置 6 发作何告警，实际 {fit.warnings}"
        n += 1
    assert n >= 1, "测试前提：至少要构造出一个'位置6 有面板'的布局"
    print(f"  位置6 可被占用：{n} 种布局全部解对、且无告警 ✓")


def test_selfcal_height_falls_back_to_measured():
    """★ 核心契约：自标定没过门/跑飞时，**绝不把未验证的高度写进导航**

    背景（2026-09-27 核对，见 docs/关卡算法/彩色数字九宫格-nine_grid/
    核对报告-2026-09-27-相机高度自标定.md）：旧实现把"参数网格投票的中位数"
    直接写回 `_cam_height_cm`——现场照片重放因此报出 48.5cm（卷尺实测 33.9cm，
    差 43%），而同一轮的像素域精修自报未通过。格阵刚性对高度本来就弱
    （每个网格点都给出同一个正确布局），所以"网格给的高度"根本不是结论。

    本用例把"精修失败"和"精修跑飞"两种情形都钉死：
      ① 精修返回 None（现场最常见：残差过门不了）→ 高度 = 实测值；
      ② 精修给出一个偏离实测值 12cm 的高度 → 同样回退实测值（可采纳带 8cm）。
    两种情形下**布局都必须仍然正确**（布局靠格阵 + 投票，与高度无关）。
    """
    from unittest.mock import patch
    import levels.nine_grid_shared as ngs

    level = NineGridShared(None)
    obs = _synth_pix_obs(50.0, 15.0, 0.0, off_deg=22.0, h_cm=33.9)
    assert len({e[2] for e in obs}) == 7, "测试前提：合成观测应覆盖 7 个数字"

    with patch.object(ngs, "measured_cam_height_cm", return_value=33.9):
        # ① 精修整体失败（现场：残差过不了门）
        with patch.object(ngs, "calibrate_pixel_pose",
                          return_value=(None, None, None)):
            fit = level._fit_grid(obs)
        assert fit.cells == TRUTH, f"布局仍须正确，实际 {fit.cells}（{fit.info}）"
        assert fit.calib_ok is False and fit.height_source == "实测回退", \
            f"应标记为实测回退，实际 {fit.height_source}/{fit.calib_ok}"
        assert abs(fit.cam_height_cm - 33.9) < 1e-6, \
            f"回退高度应等于实测 33.9cm，实际 {fit.cam_height_cm:.1f}cm"
        assert any("实测" in w for w in fit.warnings), \
            f"应给出显式告警：{fit.warnings}"

        # ② 精修顺利收敛、但**解出的高度就是离实测值 12cm**（物理矛盾或实测值错）
        #    → 自标定值仍不得写回，必须回退实测值 + 显式告警。
        obs_wild = _synth_pix_obs(50.0, 15.0, 0.0, off_deg=22.0, h_cm=45.9)
        fit2 = level._fit_grid(obs_wild)
        assert fit2.cells == TRUTH, f"布局仍须正确，实际 {fit2.cells}"
        assert fit2.calib_ok is False and fit2.height_source == "实测回退", \
            f"跑飞的高度不得被采纳，实际 {fit2.height_source}/{fit2.calib_ok}"
        assert abs(fit2.cam_height_cm - 33.9) < 1e-6, \
            f"应回退实测 33.9cm，实际 {fit2.cam_height_cm:.1f}cm"
        assert any("偏离卷尺实测" in w for w in fit2.warnings), \
            f"应给出'两种测量矛盾'的告警：{fit2.warnings}"
    print("  高回退契约：精修失败/跑飞 ⇒ 高度=实测值、布局仍正确 ✓")


def test_shared_height_recovers_non_grid_value():
    """多档共享高度：两档偏移不同、真值高度不在网格上 → 仍能定出高度

    单档观测里"相机高度"与"该档安装偏移"不可分（代码自己会告警"退化"）。
    共享一个高度 + 每档一个偏移，才让高度成为可辨识量——这也是唯一能让高度
    取**连续值**（不落在网格点上）的机制。
    """
    level = NineGridShared(None)
    obs = _synth_pix_obs(50.0, 15.0, 0.0, h_cm=41.0,
                         off_by_pitch={PITCH_NAV: 10.0, PITCH_DOWN: 22.0})
    assert len({e[2] for e in obs}) == 7, "测试前提：合成观测应覆盖 7 个数字"
    assert len({e[0] for e in obs}) == 2, "测试前提：应有两档俯仰的观测"
    fit = level._fit_grid(obs)
    assert fit.cells == TRUTH, f"布局解算错误: {fit.cells}（{fit.info}）"
    assert fit.calib_ok, f"像素域精修应过门，实际 {fit.height_source}：{fit.warnings}"
    assert abs(fit.cam_height_cm - 41.0) <= 2.0, \
        f"共享高度应解出 41.0cm±2，实际 {fit.cam_height_cm:.1f}cm"
    print(f"  共享高度：解出 {fit.cam_height_cm:.1f}cm（真值 41.0cm）、"
          f"偏移 {fit.offset_deg:+.1f}°、来源 {fit.height_source} ✓")



def test_grid_convention():
    """格号约定：格6=入口视角左下，格0=远排左；cell_index 自洽

    （"格6 恒空"这条规则已取消：位置 6 有面板完全正常，见 2026-09-28 现场确认。）
    """
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
    test_self_calibration_and_pose_from_panels()
    test_coverage_requires_all_digits()
    test_clipped_bias_downweighted()
    test_outlier_never_yields_wrong_layout()
    test_cell6_may_be_occupied()
    test_selfcal_height_falls_back_to_measured()
    test_shared_height_recovers_non_grid_value()
    print(f"全部格阵拟合测试通过 ✓（干净子集 RMS 门 {GRID_FIT_RMS_MAX_CM:.0f}cm）")
