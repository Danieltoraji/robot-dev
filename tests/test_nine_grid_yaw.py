#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""yaw 的**增益标定**不变量（2026-09-24 定稿）

`_see_target` 的 yaw 是**伺服误差**（横向像素偏移换算的"名义度"），
不是可信的角度数值。本测试不驱动任何控制路径，只用纯数学把两件事钉住：

1. **增益 ≈ 0.734**：`yaw = -(dx/W)·CAMERA_FOV_H_DEG` 与真实相机方位角之比。
   - 名义 FOV 60° ≠ 真实 `2·atan(1296/1944.9) = 67.36°`
   - 线性代替 atan
   - 两条合成后随距离缓变：20cm 处 0.696、55cm 处 0.752、130cm 处 0.719
2. **这个增益是承重的**：整族 yaw 域阈值（`VIS_ALIGN_TOL_DEG` / `VIS_BIG_TURN_DEG`
   / `VIS_ALIGN_WORSEN_DEG` / 改善判据 0.5° / 小转规划下限）都在这个坐标下标定。

2026-09-24 实测（16 种子、实测原语）把控制侧换成"去畸变 + atan"的精确公式：
    到达 112/112 → **72/111**，真值落点最大 4.3cm → **128cm**，超半格 0 → 35；
补偿死区到 15/16/17°（对齐等效真实角度 16.3°）也只回到 35/48 ~ 43/48（基线 56/56）。
⇒ 所以**不要**单独把 yaw "修准"。本测试就是那道闸：增益一变就必须整族重标。
"""
import math

import numpy as np
import pytest

import levels.nine_grid as NG
from core.camera_config import (
    CAMERA_INTRINSIC, CAM_HEIGHT_STANDING_CM, CAM_PITCH_MOUNT_OFFSET_DEG)

W = float(NG.CAMERA_WIDTH)
FX = float(CAMERA_INTRINSIC[0, 0])
CX = float(CAMERA_INTRINSIC[0, 2])


def _exact_azimuth(u, v):
    """真实相机方位角（用未畸变归一化平面；只作对照，不进控制路径）"""
    import cv2
    from core.camera_config import CAMERA_DISTORTION
    und = cv2.undistortPoints(np.array([[[u, v]]], np.float64),
                              CAMERA_INTRINSIC, CAMERA_DISTORTION)
    return -float(np.degrees(math.atan2(float(und[0, 0, 0]), 1.0)))


def _sweep():
    """两档俯仰 × 9 个方位角 × 7 个距离：(真实方位角, 关卡 yaw)"""
    import cv2
    import sim.nine_grid_sim as SIM
    from core.camera_config import CAMERA_DISTORTION
    rows = []
    for pitch in (NG.PITCH_NAV, NG.PITCH_DOWN):
        R = SIM.camera_rotation(0.0, pitch, CAM_PITCH_MOUNT_OFFSET_DEG)
        C = np.array([0.0, 0.0, CAM_HEIGHT_STANDING_CM])
        for d in (20.0, 30.0, 40.0, 55.0, 70.0, 100.0, 130.0):
            for bear in (3, 6, 9, 12, 15, 20, 25, 30, 33):
                X = d * math.sin(math.radians(bear))
                Y = d * math.cos(math.radians(bear))
                pc = R @ (np.array([X, Y, 0.0]) - C)
                if pc[2] <= 0.05:
                    continue
                norm = np.array([[[pc[0] / pc[2], pc[1] / pc[2], 1.0]]],
                                np.float32)
                pix = cv2.projectPoints(norm, np.zeros(3), np.zeros(3),
                                        CAMERA_INTRINSIC, CAMERA_DISTORTION)
                u = float(pix[0][0, 0, 0])
                v = float(pix[0][0, 0, 1])
                true_az = -math.degrees(math.atan2(pc[0], pc[2]))
                yaw = -(u - CX) / W * NG.CAMERA_FOV_H_DEG
                rows.append((true_az, yaw, _exact_azimuth(u, v)))
    return rows


def test_yaw_gain_is_about_0734():
    """增益（关卡 yaw / 真实方位角）必须仍在 0.734 附近

    这个倍率一变，`VIS_ALIGN_TOL_DEG` / `VIS_BIG_TURN_DEG` /
    `VIS_ALIGN_WORSEN_DEG` 等**整族阈值**都要重标（见模块 docstring 的实测）。
    """
    rows = _sweep()
    ratios = [r[1] / r[2] for r in rows if abs(r[2]) > 3.0]
    assert len(ratios) > 50, "扫描样本太少"
    med = float(np.median(ratios))
    assert 0.70 < med < 0.77, (
        f"yaw 增益 {med:.4f} 偏离已知的 0.734 —— 若确实改了公式/内参/FOV，"
        "必须整族重标 yaw 域阈值，不能只改一个")


def test_yaw_gain_is_distance_dependent_but_bounded():
    """增益随距离缓变（0.69~0.76），但不得发散——这是"当增益用"的前提"""
    rows = _sweep()
    ratios = [r[1] / r[2] for r in rows if abs(r[2]) > 3.0]
    assert 0.65 < min(ratios) and max(ratios) < 0.82, \
        f"增益区间 [{min(ratios):.3f}, {max(ratios):.3f}] 超预期，增益不再稳定"


def test_fov_constant_is_nominal_not_true():
    """名义 FOV 必须与真实 FOV 有已知差距（60° vs 67.36°）——这是增益的成因之一"""
    true_fov = 2.0 * math.degrees(math.atan(W / 2.0 / FX))
    assert abs(true_fov - NG.CAMERA_FOV_H_DEG) > 5.0, \
        "名义 FOV 与真实 FOV 已一致——请更新 CAMERA_FOV_H_DEG 的注释与增益说明"
    assert abs(true_fov - 67.36) < 0.1


def test_exact_yaw_model_is_NOT_in_control_path():
    """**回归闸**：不允许把精确 yaw 公式重新接回控制路径

    2026-09-24 实测它会把到达率从 112/112 打到 72/111。要走这条路必须
    整族重标 + 独立 A/B，并连同本测试一起改。
    """
    assert not hasattr(NG, "YAW_EXACT_MODEL"), \
        "YAW_EXACT_MODEL 又被加回来了——请先读 CAMERA_FOV_H_DEG 的注释"
    src = open(NG.__file__, encoding="utf-8").read()
    # `_see_target` 里不得出现 undistortPoints（它属于"修准 yaw"那条被否证的路）
    i = src.find("def _see_target")
    j = src.find("def _see_target_any")
    assert i > 0 and j > i
    assert "undistortPoints" not in src[i:j], \
        "_see_target 里出现了 undistortPoints —— 精确 yaw 被接回控制路径了"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
