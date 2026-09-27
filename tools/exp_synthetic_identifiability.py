# -*- coding: utf-8 -*-
"""临时实验 6：合成"理想观测"——把搜索范围这个变量彻底隔离掉

用真实相机模型（含畸变、与工程同一套 project_ground_to_pixel）生成**无噪声**
观测：真实 (高度, 偏移) 已知，站位与现场照片同量级（面板 30~90cm、±45° 头部
yaw），跑**真实的** _fit_grid，看它能不能把真值找回来。

  · 若 h*=33.9 时自标定回不来（被推到 40+）而 h*=48 能回来 → 病根在"可辨识性"
    （下界只是表症）；
  · 若 h*=33.9 能回来 → 那就纯粹是搜索下界的问题，改常量即可。

用法：python tools/exp_synthetic_identifiability.py
"""

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import numpy as np

import levels.nine_grid_shared as ngs
from core.ground_homography import (
    CAMERA_TO_BODY_FORWARD_CM, GRID_CELL_CM, grid_cell_center,
)
from levels.nine_grid_shared import (
    HEAD_CENTER, PITCH_NAV, NineGridShared, project_ground_to_pixel,
)

CELLS = {1: 8, 2: 5, 3: 4, 4: 6, 5: 3, 6: 0, 7: 1}


class _NullRobot:
    def set_pitch(self, *a, **k):
        pass

    def set_head(self, *a, **k):
        pass

    def run_action(self, *a, **k):
        pass

    def act(self, *a, **k):
        pass

    def capture_frame(self):
        return None


def synth_obs(h_true, off_true, cop_x, cop_y, theta_rad,
              heads=(HEAD_CENTER, 1950, 1050, 2200, 800), pitch=PITCH_NAV):
    """生成理想观测 [(pitch, head, digit, 像素, 裁切=False), ...]"""
    out = []
    for head in heads:
        for d, cell in CELLS.items():
            g = grid_cell_center(cell)
            q = project_ground_to_pixel(g, cop_x, cop_y, theta_rad, pitch, head,
                                        pitch_offset_deg=off_true,
                                        cam_height_cm=h_true)
            q = np.asarray(q, float).ravel()[:2]
            if not np.all(np.isfinite(q)):
                continue
            if not (-1500 <= q[0] <= 2592 + 1500 and -1500 <= q[1] <= 1944 + 1500):
                continue
            if not (0 <= q[0] <= 2592 and 0 <= q[1] <= 1944):
                continue                     # 只保留完全可见（不裁切）
            out.append((pitch, head, d, q, False))
    return out


def main():
    level = NineGridShared(_NullRobot())
    th = np.radians(10.0)                     # 站位：台面南侧偏一点
    cop_x, cop_y = 50.0, 0.0 - CAMERA_TO_BODY_FORWARD_CM
    print(f"站位（光心）({cop_x:.1f}, {cop_y:.1f})，航向 {np.degrees(th):.1f}°；"
          f"pitch={PITCH_NAV}，头部档 {5} 个")
    print(f"搜索网格（实测锚定带）：高度 "
          f"{[round(ngs.measured_cam_height_cm() + d, 1) for d in ngs.LAYOUT_SCAN_HEIGHT_OFFSETS_CM]}cm；"
          f"偏移 [{ngs.LAYOUT_SCAN_OFFSET_MIN_DEG:.0f}, "
          f"{ngs.LAYOUT_SCAN_OFFSET_MAX_DEG:.0f}]° / 步 "
          f"{ngs.LAYOUT_SCAN_OFFSET_STEP_DEG:.1f}°\n")
    print("  真值高度  真值偏移  观测数  解出布局  解出高度  解出偏移  位姿误差  RMS")
    for h_true in (30.0, 33.9, 36.0, 40.0, 48.0, 51.0, 56.0):
        for off_true in (8.0, 21.3, 25.0, 30.0):
            obs = synth_obs(h_true, off_true, cop_x, cop_y, th)
            fit = level._fit_grid(obs)
            if fit.cells is None:
                print(f"  {h_true:8.1f}  {off_true:+8.1f}  {len(obs):6d}   "
                      f"✗ 失败（{fit.info[:28]}）")
                continue
            ok = "✓" if fit.cells == CELLS else "✗"
            msg = ""
            if fit.pose is not None:
                msg = (f"  ({fit.pose[0] - cop_x:+.1f},"
                       f"{fit.pose[1] - cop_y:+.1f})cm")
            else:
                msg = "  (未精修)"
            print(f"  {h_true:8.1f}  {off_true:+8.1f}  {len(obs):6d}   {ok}       "
                  f"{fit.cam_height_cm:7.1f}  {fit.offset_deg:+8.1f}{msg}  "
                  f"{fit.rms_clean_cm if fit.rms_clean_cm is not None else float('nan'):4.2f}cm")


def main_floor():
    """实测锚定带：把带设成"只围绕真值 33.9"→ 看它能不能精确落回来

    （旧版这里改的是固定的高度区间下界；2026-09-27 起网格改成
    `measured + LAYOUT_SCAN_HEIGHT_OFFSETS_CM`，所以改成改"带"。）
    """
    old = ngs.LAYOUT_SCAN_HEIGHT_OFFSETS_CM
    ngs.LAYOUT_SCAN_HEIGHT_OFFSETS_CM = (-2.0, 0.0, 2.0, 5.0)
    try:
        print("\n== 实测锚定带 [31.9, 33.9, 35.9, 38.9]：真值 33.9 / 偏移 21.3° ==")
        level = NineGridShared(_NullRobot())
        obs = synth_obs(33.9, 21.3, 50.0, -CAMERA_TO_BODY_FORWARD_CM,
                        np.radians(10.0))
        fit = level._fit_grid(obs)
        print(f"   解得 高度 {fit.cam_height_cm:.2f}cm / 偏移 {fit.offset_deg:+.2f}°"
              f"（真值 33.9 / +21.3）来源 {fit.height_source} RMS "
              f"{fit.rms_clean_cm if fit.rms_clean_cm is not None else float('nan'):.2f}cm")
        for w in fit.warnings:
            print(f"   警告: {w}")
    finally:
        ngs.LAYOUT_SCAN_HEIGHT_OFFSETS_CM = old


def main_noise():
    """理想观测 + 小的裁切质心偏差：高度还能被定住吗？"""
    print("\n== 观测带上现场量级的位置偏差（模拟裁切质心偏移）==")
    level = NineGridShared(_NullRobot())
    th = np.radians(10.0)
    rng = np.random.default_rng(7)
    print("  真值高度  偏差幅度  解出高度  解出偏移  位姿误差")
    for h_true in (33.9, 48.0):
        base = synth_obs(h_true, 21.3, 50.0, -CAMERA_TO_BODY_FORWARD_CM, th)
        for amp_cm in (0.0, 1.0, 3.0, 6.0):
            obs = []
            for k, (pitch, head, d, px, cl) in enumerate(base):
                # 在"预测像素"上加偏差：等价于观测中心偏了 amp_cm
                jitter = rng.normal(0.0, amp_cm, 2)
                obs.append((pitch, head, d, px + jitter * 20.0, cl))  # ~20px/cm
            fit = level._fit_grid(obs)
            if fit.cells is None:
                print(f"  {h_true:8.1f}  {amp_cm:8.1f}   ✗ 拟合失败（{fit.info[:24]}）")
                continue
            pose = (f"({fit.pose[0] - 50.0:+.1f},{fit.pose[1] + CAMERA_TO_BODY_FORWARD_CM:+.1f})cm"
                    if fit.pose is not None else "(未精修)")
            print(f"  {h_true:8.1f}  {amp_cm:6.1f}cm  {fit.cam_height_cm:8.1f}  "
                  f"{fit.offset_deg:+8.1f}  {pose}")


if __name__ == "__main__":
    main()
    main_floor()
    main_noise()

