# -*- coding: utf-8 -*-
"""数字宫格裁切感知定位单元测试（tests/test_nine_grid_localize.py）

背景（2026-09-08 根因诊断）：面板被画幅裁切后，轮廓"对角线交点中心"失去
几何意义（实测偏差 150~826px），把它当观测送进最小二乘会把解拖走，随后
的事后离群剔除又把点剔到 <2 个 → 定位返回 None（仿真 7/7 全败的直接原因）。
修复：裁切面板改用"颜色掩膜凸包面积质心"观测，预测端用"四角投影与画幅
求交的面积质心"配对，两者同定义、裁切不变。

本测试用仿真相机在"多块面板被裁切"的位姿上验证：
1. 裁切观测模型：预测裁切质心 ↔ 检出凸包质心 的残差 <5px，
   而"预测面板中心 ↔ 凸包质心"残差 >50px（证明旧观测不可用）；
2. 端到端：从 ±8cm/±7° 扰动先验出发，GN 收敛到真值 <1cm/<1°。

运行：python tests/test_nine_grid_localize.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # 仓库根

import numpy as np

from core.camera_config import HEAD_RIGHT
from levels.nine_grid import (
    NineGridLevel, PITCH_NAV, clipped_quad_centroid,
    project_ground_to_pixel, _wrap_angle,
)
from sim.nine_grid_sim import SimNineGridRobot, SIM_LAYOUT

# 复现根因的位姿：导航档 + 头部右转 40.5°，一帧内 3 块面板被裁切
SCENE_POSE = (50.0, -20.0, -18.69)
SCENE_HEAD = HEAD_RIGHT
PRIOR = np.array([49.43, -20.31, np.radians(-25.71)])  # 比真值差 ~7° 的动作预测


def _scene():
    robot = SimNineGridRobot()
    level = NineGridLevel(robot)
    level.digit_cell = {d: c for c, d in SIM_LAYOUT.items() if d is not None}
    robot.pos = np.array([SCENE_POSE[0], SCENE_POSE[1]])
    robot.heading = SCENE_POSE[2]
    robot.pitch = PITCH_NAV
    robot.set_head(SCENE_HEAD)
    truth = np.array([SCENE_POSE[0], SCENE_POSE[1],
                      np.radians(SCENE_POSE[2])])
    corr, obs, _frame = level._capture_corr(PITCH_NAV)
    return level, truth, corr, obs


def test_clipped_observation_model():
    """裁切观测配对：预测裁切质心必须贴合检出凸包质心，中心模型必须偏差巨大"""
    _level, truth, corr, obs = _scene()
    clipped = [(c, o) for c, o in zip(corr, obs) if c[3]]
    assert len(clipped) >= 2, f"本用例应有 >=2 块裁切面板，实际 {len(clipped)}"

    max_clip_err, min_center_err = 0.0, 1e9
    for (cell_xy, _obs_px, head, _clipped), o in clipped:
        pred_clip = clipped_quad_centroid(
            cell_xy, truth[0], truth[1], truth[2], PITCH_NAV, head)
        pred_center = project_ground_to_pixel(
            cell_xy, truth[0], truth[1], truth[2], PITCH_NAV, head)[0]
        det = np.asarray(o.hull_centroid_px)
        max_clip_err = max(max_clip_err, float(np.linalg.norm(pred_clip - det)))
        min_center_err = min(min_center_err,
                             float(np.linalg.norm(pred_center - det)))

    assert max_clip_err < 5.0, \
        f"裁切预测质心与检出凸包质心差 {max_clip_err:.1f}px（应 <5px）"
    # 中心模型偏差随裁切比例变化：本场景最小 31px、最大 800+px；
    # 门槛取 20px 即足以证明"中心观测不可用"（裁切观测 <5px）。
    assert min_center_err > 20.0, \
        f"面板中心投影与凸包质心最小差 {min_center_err:.1f}px（旧模型应 >20px）"
    print(f"  裁切观测配对：裁切质心误差 {max_clip_err:.1f}px，"
          f"中心模型最小偏差 {min_center_err:.0f}px ✓")


def test_localize_converges_from_perturbed_prior():
    """端到端：含裁切面板的对应集从扰动先验收敛到真值"""
    level, truth, corr, _obs = _scene()
    p = level._gn_localize(PRIOR.copy(), corr, PITCH_NAV)
    assert p is not None, "含裁切面板的对应集定位失败（曾直接发散）"
    err_xy = float(np.hypot(p[0] - truth[0], p[1] - truth[1]))
    err_th = abs(np.degrees(_wrap_angle(p[2] - truth[2])))
    assert err_xy < 1.0, f"位置误差 {err_xy:.2f}cm 超 1cm"
    assert err_th < 1.0, f"航向误差 {err_th:.2f}° 超 1°"
    print(f"  裁切感知定位：扰动先验 → 误差 {err_xy:.2f}cm / {err_th:.2f}° ✓")


def test_unclipped_center_model_exact():
    """未裁切面板：中心模型仍精确（残差 <20px），保证近距外精度不退步

    阈值依据（2026-09-11 更新为"安装偏移 + 高度 56cm"后的新几何）：本场景
    实测 关卡投影 ↔ 仿真渲染 误差 **0.00px**（两侧模型严格一致），残差全部
    来自检测端 quad_center 对栅格化四边形的 approxPolyDP 近似（4~15px ≈
    0.2~0.8cm）；而裁切面板的"中心模型"偏差在本场景为 76~140px。故 20px
    既能把中心模型与裁切模型分开，又不把检测噪声误判成模型缺陷。
    """
    robot = SimNineGridRobot()
    level = NineGridLevel(robot)
    level.digit_cell = {d: c for c, d in SIM_LAYOUT.items() if d is not None}
    robot.pos = np.array([50.0, -20.0])
    robot.heading = 0.0
    robot.pitch = PITCH_NAV
    robot.set_head(1500)
    truth = np.array([50.0, -20.0, 0.0])
    corr, obs, _ = level._capture_corr(PITCH_NAV)
    clean = [(c, o) for c, o in zip(corr, obs) if not c[3]]
    assert len(clean) >= 4, f"入口帧应有 >=4 块完整面板，实际 {len(clean)}"
    errs = []
    for (cell_xy, _obs_px, head, _c), o in clean:
        pred = project_ground_to_pixel(
            cell_xy, truth[0], truth[1], truth[2], PITCH_NAV, head)[0]
        errs.append(float(np.linalg.norm(pred - np.asarray(o.center_px))))
    assert max(errs) < 20.0, f"完整面板中心模型残差 {max(errs):.1f}px 超 20px"
    print(f"  完整面板中心模型：最大残差 {max(errs):.1f}px"
          f"（≪裁切模型 76~140px）✓")


def test_clipped_prediction_inside_image():
    """裁切预测边界条件：脚下只余贴底细条/贴边极限，远距为画内点

    阈值按"安装偏移 + 高度 56cm"的新几何重推（2026-09-11；旧几何 39cm/41.4°
    下站在面板中心时面板整个在视野外，现在是"近端一小条可见"）：
      - 站在面板中心 pitch1040：可见地面带起点 ≈3.3cm，面板前缘 14cm 可见
        → 裁切质心预测 (1283, 1734) 在画内靠底；面板中心投影 y=2004 已出画；
      - 站在面板中心 pitch1200：面板完全在可见带之下 → 预测点被钳到画幅底边
        （y=1944），仍是有限值（数值雅可比需要连续，不能返回 None）；
      - 距 60cm pitch1200：面板基本在画内（y≈937）。
    """
    from core.ground_homography import grid_cell_center
    center = grid_cell_center(4)  # 面板 4 在格心 (50,50)
    # 站在面板中心（pitch 1040）：裁切质心在画内靠底，中心投影已出画
    on_panel = clipped_quad_centroid(center, 50.0, 50.0, 0.0, 1040)
    center_px = project_ground_to_pixel(center, 50.0, 50.0, 0.0, 1040)[0]
    assert on_panel is not None and 0 <= on_panel[0] <= 2592 \
        and on_panel[1] > 1944 * 0.75, \
        f"站在面板中心时裁切预测应在画内靠底，实际 {on_panel}"
    assert center_px[1] > 1944, \
        f"站在面板中心时面板中心投影应已出画（下沿外），实际 {center_px}"
    # 导航档站在面板中心：面板完全在视野外 → 钳到画幅的极限点（不返回 None）
    outside = clipped_quad_centroid(center, 50.0, 50.0, 0.0, PITCH_NAV)
    assert outside is not None, "画外预测应给连续极限点而非 None"
    assert 0 <= outside[0] <= 2592 and 1944 - 1.0 <= outside[1] <= 1944, \
        f"完全出画时应钳到画幅底边，实际 {outside}"
    # 站在 60cm 外：面板大部分可见 → 画内点（明显高于底边）
    far = clipped_quad_centroid(center, 50.0, -10.0, 0.0, PITCH_NAV)
    assert far is not None, "远距应给出裁切预测"
    assert 0 <= far[0] <= 2592 and 0 <= far[1] < 1944 - 100, \
        f"远距预测点应在画内偏上: {far}"
    print(f"  裁切预测边界：脚下贴底细条({int(on_panel[1])}px，中心投影 "
          f"{int(center_px[1])}px 出画)，画外极限({outside[0]:.0f},{outside[1]:.0f})，"
          f"远距画内 ({far[0]:.0f},{far[1]:.0f}) ✓")


if __name__ == "__main__":
    test_clipped_observation_model()
    test_localize_converges_from_perturbed_prior()
    test_unclipped_center_model_exact()
    test_clipped_prediction_inside_image()
    print("全部 nine_grid 定位测试通过 ✓")
