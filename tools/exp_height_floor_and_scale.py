# -*- coding: utf-8 -*-
"""临时实验 5：① 把高度下界降到 30cm，布局扫会报什么？② 面板像素尺寸定标

① 直接改模块常量后跑**真实的** _fit_grid，看结果是不是还是 48cm 那一档
   （= "下界改小也救不回来"的判定），还是掉到真值附近。
② 面板物理边长已知（色块约 28cm ± 半宽 14cm），所以它在像素里的尺寸是一个
   **与相机高度无关的绝对尺度尺**：预测尺寸/实测尺寸 的比值直接给出
   真实高度 / 自标定高度。这是打破 (偏移, 高度) 退化谷最便宜的一招。

用法：python tools/exp_height_floor_and_scale.py
"""

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import numpy as np

import levels.nine_grid_shared as ngs
from core.ground_homography import GroundHomography, grid_cell_center
from levels.nine_grid_shared import NineGridShared
from tools.replay_ninegrid import (
    ReplayRobot, build_schedule, find_photos, sweep_and_collect, PHOTO_TRUTH,
)

PANEL_HALF_CM = 14.0        # 色块半边（nine_grid_shared.PANEL_HALF_CM 同值）


def run_scan(hmin, hmax, tag):
    """用"实测高度锚定带"模拟不同的搜索带（2026-09-27：网格已改成
    `measured + LAYOUT_SCAN_HEIGHT_OFFSETS_CM`，下界不再是固定 36/30/25）"""
    old = ngs.LAYOUT_SCAN_HEIGHT_OFFSETS_CM
    measured = ngs.measured_cam_height_cm()
    ngs.LAYOUT_SCAN_HEIGHT_OFFSETS_CM = tuple(
        h - measured for h in np.arange(hmin, hmax + 1e-9, 5.0))
    try:
        photos = find_photos()
        robot = ReplayRobot(build_schedule(photos, "schedule"), verbose=False)
        level = NineGridShared(robot)
        print(f"\n--- {tag}：候选高度 {[round(measured + d, 1) for d in ngs.LAYOUT_SCAN_HEIGHT_OFFSETS_CM]} ---")
        fit = level._fit_grid(sweep_and_collect(robot, level, diag=False))
        print(f"    cells = {fit.cells}   {'= 照片真值 ✓' if fit.cells == PHOTO_TRUTH else '✗ 与真值不符'}")
        print(f"    自标定：偏移 {fit.offset_deg:+.1f}° / 高度 {fit.cam_height_cm:.1f}cm"
              f"（{fit.height_source}）")
        for w in fit.warnings:
            print(f"    警告: {w}")
    finally:
        ngs.LAYOUT_SCAN_HEIGHT_OFFSETS_CM = old


def scale_check():
    """面板像素尺寸 ↔ 物理尺寸：反推真实高度"""
    photos = find_photos()
    robot = ReplayRobot(build_schedule(photos, "schedule"), verbose=False)
    level = NineGridShared(robot)
    print("\n== 面板像素尺寸定标（未裁切观测；预测/实测 边长比） ==")
    print("   高度假设 | 各面板 预测边长/实测边长（比值 1.0 = 该高度正确）")
    corners = np.array([[-PANEL_HALF_CM, -PANEL_HALF_CM],
                        [PANEL_HALF_CM, -PANEL_HALF_CM],
                        [PANEL_HALF_CM, PANEL_HALF_CM],
                        [-PANEL_HALF_CM, PANEL_HALF_CM]], float)
    for pitch, head, _p in robot.schedule:
        robot.set_pitch(pitch)
        robot.set_head(head)
        frame = robot.capture_frame()
        if frame is None:
            continue
        obs = level.detector.detect_panels(frame, arbitrate=True, drop_border=False)
        for o in obs:
            if o.clipped or len(o.hull_poly) < 3:
                continue
            g = grid_cell_center(PHOTO_TRUTH[o.digit])
            # 实测边长：凸包面积 → 等效正方形边长（原生 px）
            poly = np.asarray(o.hull_poly, float)
            x, y = poly[:, 0], poly[:, 1]
            area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            side_obs = float(np.sqrt(area))
            row = []
            for h in (33.9, 40.0, 48.0, 56.0):
                hg = GroundHomography.from_pose((0.0, 0.0), h, pitch,
                                                head_pulse=head,
                                                pitch_offset_deg=0.0)
                # 位置不影响尺寸；取格心正上方 55cm 处做参照
                p4 = hg.ground_to_pixels(g[None, :] + corners)
                if p4 is None or not np.all(np.isfinite(p4)):
                    row.append(float("nan"))
                    continue
                side_pred = float(np.sqrt(0.5 * abs(
                    np.dot(p4[:, 0], np.roll(p4[:, 1], 1))
                    - np.dot(p4[:, 1], np.roll(p4[:, 0], 1)))))
                row.append(side_obs / side_pred if side_pred > 1e-9 else float("nan"))
            print(f"   pitch={pitch} head={head:4d} 数字{o.digit} "
                  f"实测边长 {side_obs:6.1f}px | "
                  + "  ".join(f"h={h:4.0f}: {r:5.2f}" for h, r in
                              zip((33.9, 40.0, 48.0, 56.0), row)))
    print("\n判读：比值 = 实测/预测。若 h=33.9 那一列 ≈1.0，则相机确实在 33.9cm；")
    print("      若 h=48 那一列 ≈1.0，则自标定是对的、卷尺那个 33.9 需要复查。")
    print("      注意几何位置取自真值格阵、且假设站位使面板在 30~100cm 内——")
    print("      这只做量级判断（比值随假设高度按 1/h 变化）。")


if __name__ == "__main__":
    run_scan(24.0, 70.0, "宽带 24~70cm（旧写法的等价物）")
    run_scan(30.0, 70.0, "把带的下沿放到 30cm")
    run_scan(33.9, 33.9, "只留实测值一档")
    scale_check()
