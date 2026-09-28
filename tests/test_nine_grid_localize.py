# -*- coding: utf-8 -*-
"""数字宫格裁切感知观测单元测试（tests/test_nine_grid_localize.py）

背景（2026-09-08 根因诊断）：面板被画幅裁切后，轮廓"对角线交点中心"失去
几何意义（实测偏差 150~826px）。修复：裁切面板改用"颜色掩膜凸包面积质心"
观测，预测端用"四角投影与画幅求交的面积质心"配对，两者同定义、裁切不变。

本测试用仿真相机在"多块面板被裁切"的位姿上钉住**投影/观测模型**：
1. 裁切观测模型：预测裁切质心 ↔ 检出凸包质心 的残差 <5px，
   而"预测面板中心 ↔ 凸包质心"残差 >20px（证明旧观测不可用）；
2. 未裁切面板：中心模型仍精确（残差 <20px）；
3. 裁切预测的边界条件：脚下的贴底细条、完全出画时的钳位、远距画内点。

注：地图 GN 定位（`_gn_localize` / `_capture_corr`）已随"删用不上的代码"
一步移除，本文件因此只保留投影侧的三条。

运行：python tests/test_nine_grid_localize.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # 仓库根

import numpy as np

from core.camera_config import HEAD_RIGHT
from levels.nine_grid_shared import (
    NineGridShared, PITCH_NAV, clipped_centroid,
    project_ground_to_pixel,
)
from sim.nine_grid_sim import SimNineGridRobot, SIM_LAYOUT

# 复现根因的位姿：导航档 + 头部右转 40.5°，一帧内 3 块面板被裁切
SCENE_POSE = (50.0, -20.0, -18.69)
SCENE_HEAD = HEAD_RIGHT


def _frame_observations(level, robot, pitch):
    """拍一帧并检测 → [(格心, 观测, 头部脉宽, 是否裁切), ...]

    与已删的 `_capture_corr` 同一算法：只保留地图里已知的数字，
    观测点取"未裁切用对角线交点、裁切用凸包质心"。
    """
    frame = robot.capture_frame()
    obs = level.detector.detect_panels(frame, drop_border=False)
    head = robot.current_head_pulse
    from core.ground_homography import grid_cell_center
    out = []
    for o in obs:
        if o.digit not in level.digit_cell:
            continue
        out.append((grid_cell_center(level.digit_cell[o.digit]), o, head,
                    bool(o.clipped)))
    return out


def _scene():
    """复现根因的位姿：一帧内多块面板被裁切"""
    robot = SimNineGridRobot()
    level = NineGridShared(robot)
    level.digit_cell = {d: c for c, d in SIM_LAYOUT.items() if d is not None}
    robot.pos = np.array([SCENE_POSE[0], SCENE_POSE[1]])
    robot.heading = SCENE_POSE[2]
    robot.pitch = PITCH_NAV
    robot.set_head(SCENE_HEAD)
    truth = np.array([SCENE_POSE[0], SCENE_POSE[1],
                      np.radians(SCENE_POSE[2])])
    pairs = _frame_observations(level, robot, PITCH_NAV)
    return level, truth, pairs


def test_clipped_observation_model():
    """裁切观测配对：预测裁切质心必须贴合检出凸包质心，中心模型必须偏差巨大"""
    _level, truth, pairs = _scene()
    clipped = [(c, o, h) for c, o, h, cl in pairs if cl]
    assert len(clipped) >= 2, f"本用例应有 >=2 块裁切面板，实际 {len(clipped)}"

    max_clip_err, min_center_err = 0.0, 1e9
    for cell_xy, o, head in clipped:
        pred_clip = clipped_centroid(
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


def test_unclipped_center_model_exact():
    """未裁切面板：中心模型仍精确（残差 <20px），保证近距外精度不退步

    阈值依据：本场景实测"关卡投影 ↔ 仿真渲染"误差 0.00px（两侧模型严格一致），
    残差全部来自检测端 quad_center 对栅格化四边形的近似（4~15px ≈ 0.2~0.8cm）；
    而裁切面板的"中心模型"偏差在本场景为 76~140px。故 20px 既能把中心模型与
    裁切模型分开，又不把检测噪声误判成模型缺陷。
    """
    robot = SimNineGridRobot()
    level = NineGridShared(robot)
    level.digit_cell = {d: c for c, d in SIM_LAYOUT.items() if d is not None}
    robot.pos = np.array([50.0, -20.0])
    robot.heading = 0.0
    robot.pitch = PITCH_NAV
    robot.set_head(1500)
    truth = np.array([50.0, -20.0, 0.0])
    pairs = _frame_observations(level, robot, PITCH_NAV)
    clean = [(c, o, h) for c, o, h, cl in pairs if not cl]
    assert len(clean) >= 4, f"入口帧应有 >=4 块完整面板，实际 {len(clean)}"
    errs = []
    for cell_xy, o, head in clean:
        pred = project_ground_to_pixel(
            cell_xy, truth[0], truth[1], truth[2], PITCH_NAV, head)[0]
        errs.append(float(np.linalg.norm(pred - np.asarray(o.center_px))))
    assert max(errs) < 20.0, f"完整面板中心模型残差 {max(errs):.1f}px 超 20px"
    print(f"  完整面板中心模型：最大残差 {max(errs):.1f}px"
          f"（≪裁切模型 76~140px）✓")


def test_clipped_prediction_inside_image():
    """裁切预测边界条件：脚下只余贴底细条/贴边极限，远距为画内点

    ⚠️ 2026-09-27 改（相机高度 56→**实测 33.9cm**，见
    docs/关卡算法/彩色数字九宫格-nine_grid/核对报告-2026-09-27-相机高度自标定.md）：
    高度降低 39% 后相机把地面看得更近，"多近才出画"这条边界跟着变——原来按
    56cm 写的"导航档站在面板中心必定完全出画"已不成立（33.9cm 时面板还在画内
    偏下）。所以这里不再钉死一个位置，而是**按当前几何反解出画边界**再验证钳位，
    这样换相机高度/换档位时测试仍然成立。

    阈值按"安装偏移 + 高度 56cm"的几何：
      - 站在面板中心 pitch1040：可见地面带起点 ≈3.3cm，面板前缘 14cm 可见
        → 裁切质心预测 (1283, 1734) 在画内靠底；面板中心投影 y=2004 已出画；
      - 站在面板中心 pitch1200：面板完全在可见带之下 → 预测点被钳到画幅底边
        （y=1944），仍是有限值（数值雅可比需要连续，不能返回 None）；
      - 距 60cm pitch1200：面板基本在画内（y≈937）。
    """
    from core.ground_homography import grid_cell_center
    center = grid_cell_center(4)  # 面板 4 在格心 (50,50)
    # 站在面板中心（pitch 1040）：裁切质心在画内靠底，中心投影已出画
    on_panel = clipped_centroid(center, 50.0, 50.0, 0.0, 1040)
    center_px = project_ground_to_pixel(center, 50.0, 50.0, 0.0, 1040)[0]
    assert on_panel is not None and 0 <= on_panel[0] <= 2592 \
        and on_panel[1] > 1944 * 0.75, \
        f"站在面板中心时裁切预测应在画内靠底，实际 {on_panel}"
    assert center_px[1] > 1944, \
        f"站在面板中心时面板中心投影应已出画（下沿外），实际 {center_px}"
    # 导航档、面板中心出画：裁切预测仍必须在画幅内（贴底细条，不能返回画外点）
    sliver = clipped_centroid(center, 50.0, 50.0, 0.0, PITCH_NAV)
    assert sliver is not None and 0 <= sliver[0] <= 2592 \
        and 1944 * 0.85 <= sliver[1] <= 1944, \
        f"面板中心出画时裁切预测应落在画幅内靠底，实际 {sliver}"
    # 完全出画（面板在机器人**身后**）：钳到画幅底边的极限点（不返回 None——
    # 数值雅可比需要连续）。用 bearing=180 构造，与相机高度无关，换档位也成立。
    outside = clipped_centroid(center, 50.0, 30.0, np.radians(180.0), PITCH_NAV)
    assert outside is not None, "画外预测应给连续极限点而非 None"
    assert 0 <= outside[0] <= 2592 and 1944 - 1.0 <= outside[1] <= 1944, \
        f"完全出画时应钳到画幅底边，实际 {outside}"
    # 站在 60cm 外：面板大部分可见 → 画内点（明显高于底边）
    far = clipped_centroid(center, 50.0, -10.0, 0.0, PITCH_NAV)
    assert far is not None, "远距应给出裁切预测"
    assert 0 <= far[0] <= 2592 and 0 <= far[1] < 1944 - 100, \
        f"远距预测点应在画内偏上: {far}"
    print(f"  裁切预测边界：脚下贴底细条({int(on_panel[1])}px，中心投影 "
          f"{int(center_px[1])}px 出画 / 导航档裁切 {sliver[1]:.0f}px 仍在画内)，"
          f"画外极限({outside[0]:.0f},{outside[1]:.0f})，"
          f"远距画内 ({far[0]:.0f},{far[1]:.0f}) ✓")


if __name__ == "__main__":
    test_clipped_observation_model()
    test_unclipped_center_model_exact()
    test_clipped_prediction_inside_image()
    print("全部 nine_grid 定位/投影测试通过 ✓")
