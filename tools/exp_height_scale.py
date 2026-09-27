# -*- coding: utf-8 -*-
"""临时实验 2：相机高度→地面映射的"各向异性缩放"到底有多敏感

几何事实（本脚本用真实投影代码验证）：
  若相机真实位姿 (h*, off*)，却被按 (h, off) 反投影，则同一批地面点的
  **径向距离会被乘上一个随视角变化的系数 k**。要是 k 全场恒定，格阵拟合
  就完全分辨不出高度（纯缩放可被刚体拟合吸收）；k 随视角变化越大，格阵的
  刚性残差对高度越敏感。

本脚本逐点算 k，并给出 k 的极差（对格阵拟合 RMD 的一阶估计）。

用法：python tools/exp_height_scale.py
"""

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import numpy as np

from core.ground_homography import GroundHomography

TRUE_H = 33.9           # 现场卷尺标定（.worktrees/stairs/levels/stairs_hurdle.py）
TRUE_OFF = 21.3         # 由卷尺标定 θ=57.34°（pitch1100 名义 36°）反推
PITCH = 1200
HEADS = (1500, 1950, 1050, 2200, 800)
DIST = 55.0             # 站位：面板大致在 30~100cm


def main():
    print(f"真值 高度 {TRUE_H}cm / 偏移 {TRUE_OFF}° / pitch {PITCH}")
    print("每个头部档取同一批真实地面点（绕相机光心方位角 0~90°），")
    print("比较 (h,off) 反投影距离与真值反投影距离之比 k：\n")
    print("   假设高度  假设偏移     k 中位    k 极差    等效格阵残差估计")
    for h in (30.0, 33.9, 36.0, 40.0, 45.0, 48.0, 51.0, 56.0, 60.0):
        for off in (TRUE_OFF, 30.0):
            ks = []
            for head in HEADS:
                hg_t = GroundHomography.from_pose(
                    (0.0, 0.0), TRUE_H, PITCH, head_pulse=head,
                    pitch_offset_deg=TRUE_OFF)
                hg_h = GroundHomography.from_pose(
                    (0.0, 0.0), h, PITCH, head_pulse=head,
                    pitch_offset_deg=off)
                # 真值下：面板在机器人系的位置（绕相机方位 0/±20/±40°）
                for psi_deg in (0.0, 20.0, -20.0, 40.0, -40.0):
                    psi = np.radians(psi_deg)
                    g = np.array([[DIST * np.sin(psi), DIST * np.cos(psi)]])
                    px = hg_t.ground_to_pixels(g)
                    if px is None or not np.all(np.isfinite(px)):
                        continue
                    back = hg_h.pixels_to_ground(px, head_pulse=head)[0]
                    d_true = float(np.linalg.norm(g[0]))
                    d_back = float(np.linalg.norm(back))
                    if d_back > 1e-6:
                        ks.append(d_back / d_true)
            if not ks:
                continue
            ks = np.array(ks)
            spread = float(ks.max() - ks.min())
            print(f"   {h:6.1f}  {off:+7.1f}°  {np.median(ks):8.3f}  "
                  f"{spread:7.3f}   {spread * DIST / 2:8.2f}cm  "
                  f"(半格阵残差 @ {DIST:.0f}cm)")
    print("\n判读：k 极差 ≈ 0 ⇒ 高度不可辨（纯缩放，刚体拟合吃得掉）；")
    print("      k 极差 0.1 量级 ⇒ 100cm 处径向残差 ~10cm ⇒ 会被 RMS/覆盖门拦下。")


if __name__ == "__main__":
    main()
