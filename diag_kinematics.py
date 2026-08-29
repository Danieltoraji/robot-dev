#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diag_kinematics.py —— 转头运动学链诊断（阶段3 辅助）

用现有采集数据（无需重新采集）回答三个问题：
  1. 转角标称是否准确？   单帧 PnP 光轴 yaw 差 vs 标称 θ：线性拟合斜率应≈1
  2. 光心偏心模型是否自洽？各帧相机位置差能否被 [Rz(θ_i)−Rz(θ_0)]·e 解释
  3. 相机俯仰是否稳定？   光轴 z 分量标准差

原理：
    机体不动、只转头时，单帧 PnP 解出的相机位姿应满足：
      光轴 yaw:  ψ_i = φ + k·θ_i          → 拟合 k，残差应≈0
      相机位置:  p_i = p_B + Rz(φ+θ_i)·e  → 帧间差只由 e 旋转引起，拟合 e 看残差
    若朝向残差大 → k≠1 或 θ 标称错（「转角相对精确」假设破）；
    若位置残差大 → 偏心模型缺项（如 t_H 转轴偏移）或机体真的动了。

用法：
    python diag_kinematics.py --data result/multiview_20260825_201411.npz
"""

import argparse
import json

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

from levels.goodluck import tag_poses
from multiview_pose import K, DIST


def load_npz(path):
    data = np.load(path, allow_pickle=True)
    pulses = data["pulses"]
    thetas = data["thetas_nom"]
    f_idx = data["frame_idx"]
    t_ids = data["tag_ids"]
    corners = data["corners"]
    frames = []
    for fi in range(len(pulses)):
        mask = f_idx == fi
        if not mask.any():
            continue
        ids = t_ids[mask]
        pts3d = np.vstack([tag_poses[str(tid)] for tid in ids])
        obs = corners[mask].reshape(-1, 4, 2).reshape(-1, 2)
        frames.append((float(thetas[fi]), pts3d, obs))
    meta = {}
    if "meta" in data and len(data["meta"]):
        try:
            meta = json.loads(str(data["meta"][0]))
        except Exception:
            meta = {}
    return frames, meta


def rot2(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def main():
    parser = argparse.ArgumentParser(description="转头运动学链诊断")
    parser.add_argument("--data", required=True, help="collect_multi_view.py 生成的 npz")
    args = parser.parse_args()

    if cv2 is None:
        print("需要 cv2 做单帧 PnP。")
        return

    frames, meta = load_npz(args.data)
    if len(frames) < 2:
        print("有效帧 < 2。")
        return

    # ---- 每帧单帧 PnP ----
    rows = []
    for fi, (theta, pts3d, obs) in enumerate(frames):
        ok, rvec, tvec = cv2.solvePnP(pts3d, obs, K, DIST)
        if not ok:
            print(f"帧{fi} θ={theta:+.1f}°: PnP 失败")
            continue
        R, _ = cv2.Rodrigues(rvec)
        pos = (-R.T @ tvec).flatten()
        ori = (R.T @ np.array([0.0, 0.0, 1.0])).flatten()
        yaw = np.degrees(np.arctan2(ori[1], ori[0]))
        proj, _ = cv2.projectPoints(pts3d, rvec, tvec, K, DIST)
        err = float(np.mean(np.linalg.norm(proj[:, 0, :] - obs, axis=1)))
        rows.append({"fi": fi, "theta": theta, "pos": pos[:2], "yaw": yaw,
                     "ori_z": ori[2], "reproj": err})

    if len(rows) < 2:
        print("单帧 PnP 成功帧 < 2。")
        return

    # 基准：|θ| 最小帧
    base = min(rows, key=lambda r: abs(r["theta"]))
    print("=" * 72)
    print(f"数据: {args.data}")
    print(f"基准帧: 帧{base['fi']} θ={base['theta']:+.1f}°")
    print("=" * 72)
    print(f"{'帧':<4}{'θ标称':>8}{'yaw':>9}{'Δyaw':>8}{'Δpos(cm)':>9}{'reproj':>7}")
    for r in rows:
        dy = r["yaw"] - base["yaw"]
        dp = float(np.linalg.norm(r["pos"] - base["pos"]))
        print(f"{r['fi']:<4}{r['theta']:>+7.1f}{r['yaw']:>9.2f}{dy:>+8.2f}{dp:>9.2f}"
              f"{r['reproj']:>7.2f}")

    # ---- 1) 转角标称检验：Δyaw = k·Δθ（过原点线性拟合） ----
    dtheta = np.array([r["theta"] - base["theta"] for r in rows], dtype=np.float64)
    dyaw = np.array([r["yaw"] - base["yaw"] for r in rows], dtype=np.float64)
    k_hat = float(np.sum(dtheta * dyaw) / np.sum(dtheta * dtheta))
    res_deg = float(np.sqrt(np.mean((dyaw - k_hat * dtheta) ** 2)))
    print()
    print(f"[1] 转角标称检验: Δyaw = k·Δθ, k = {k_hat:.3f}, 残差 RMS = {res_deg:.2f}°")
    if abs(k_hat - 1.0) < 0.03 and res_deg < 2.0:
        print("    → 转角标称准确（k≈1 且线性），「转角相对精确」假设成立")
    elif abs(k_hat - 1.0) >= 0.03:
        print(f"    → k={k_hat:.3f} ≠ 1：实际转角与标称有 {abs(1 - k_hat) * 100:.0f}% 偏差"
              f"（舵机非线性或角度标称需修正）")
    else:
        print(f"    → 线性残差 {res_deg:.2f}° 偏大：θ 记录/舵机非线性/机体转动，需排查")

    # ---- 2) 偏心模型检验：Δp = [Rz(θ_i)−Rz(θ_0)]·e ----
    phi = base["yaw"] - k_hat * base["theta"]
    print()
    print(f"[2] 偏心模型检验（假设机体不动，机体朝向 φ≈{phi:.1f}°）:")
    A_list, b_list = [], []
    for r in rows:
        A_list.append(rot2(r["theta"]) - rot2(base["theta"]))
        b_list.append(r["pos"] - base["pos"])
    A = np.array(A_list).reshape(-1, 2)
    b = np.array(b_list).reshape(-1)
    e_hat, _, _, sv = np.linalg.lstsq(A, b, rcond=None)
    res_cm = float(np.sqrt(np.mean((b - A @ e_hat) ** 2)))
    print(f"    拟合偏心 e = ({e_hat[0]:.2f}, {e_hat[1]:.2f})cm, 残差 RMS = {res_cm:.2f}cm")
    print(f"    奇异值 = {sv[0]:.3f}, {sv[1]:.3f}"
          + ("（帧间角度跨度不足，e 可能不可辨识）" if sv[1] < 0.05 else ""))
    if res_cm < 2.0:
        print("    → 偏心模型自洽：帧间位置差可由 e 旋转解释")
    elif res_cm < 5.0:
        print("    → 轻微失配：可能缺 t_H（转轴偏移）或机体微动")
    else:
        print("    → 显著失配：偏心模型不完整（缺 t_H？）或机体在转头时真的动了")

    # ---- 3) 俯仰稳定性 ----
    zs = np.array([r["ori_z"] for r in rows])
    pitch_mean = float(np.degrees(np.arcsin(np.clip(-zs.mean(), -1.0, 1.0))))
    pitch_std = float(np.std(zs))
    print()
    print(f"[3] 俯仰稳定性: 光轴 z 分量均值={zs.mean():.3f}（≈俯仰 {pitch_mean:.1f}°），"
          f"标准差={pitch_std:.3f}")
    if pitch_std < 0.02:
        print("    → 俯仰稳定（转头不改变俯仰，模型假设成立）")
    else:
        print("    → 俯仰随转头变化：机体倾斜或头部俯仰耦合，模型需加项")

    # ---- 4) 汇总 ----
    print()
    print("=" * 72)
    print("诊断结论:")
    print(f"  转角标称: k={k_hat:.3f}（残差 {res_deg:.2f}°）")
    print(f"  偏心模型: e=({e_hat[0]:.2f},{e_hat[1]:.2f})cm（残差 {res_cm:.2f}cm）")
    print(f"  俯仰: {pitch_mean:.1f}° ± {pitch_std:.3f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
