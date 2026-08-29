#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
synthetic_multi_view.py —— 阶段1：多帧联合位姿求解（A 方案）合成数据验证

背景：
    单帧 PnP 存在「旋转↔平移」歧义：转头时旋转被吸收进平动，
    导致位置虚增（实测 27° 档位置错 15cm、朝向丢 25°）。
    本脚本验证「转头多帧 + 已知转角」联合求解能否消除歧义，
    并恢复机体位姿 / 光心偏心 / 相机俯仰。

场景矩阵（每场景独立生成观测、独立求解）：
    A 同标签 3 帧      理想摆动（摆动中心对准标签，3 帧同一标签）
    B 同墙不同标签     单面墙、帧间标签集合不同 ← 赛道最常见场景
    C 仅 2 帧          回正失败，只有转头帧有标签
    D 不同墙（非共面） 帧间观测来自两个墙面（概念验证非共面增益）

用法（需 numpy + scipy；TonyPi 已确认有 scipy）：
    python synthetic_multi_view.py
    python synthetic_multi_view.py --noise 0.5 --seed 2

输出：
    每场景：真实/初值/求解参数表、位置/朝向/偏心/俯仰恢复误差、残差 RMS；
    A 场景附单帧 PnP 对比（演示歧义：右转帧位置虚增）。
"""

import argparse

import numpy as np

try:
    from scipy.optimize import least_squares
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

try:
    import cv2
except Exception:
    cv2 = None

# 正向几何模型（camera_pose/project_points/residual/solve_joint）、参数表与
# 相机内参已收敛到 multiview_pose.py（单一实现）；本脚本保留合成场景、
# 判定与报告逻辑，作为阶段1合成验证器。
from multiview_pose import (
    camera_pose, project_points, residual, solve_joint,
    K, DIST, P_NAMES, P_INDEX, P_UNIT, P_BOUNDS,
)


# =====================================================================
# 标签库（世界坐标角点，左上→右上→右下→左下）
# =====================================================================

def tag_on_x_wall(x, y_center, z_center, half=2.5):
    """x=常数 竖墙上的标签"""
    return np.array([
        [x, y_center + half, z_center + half],
        [x, y_center - half, z_center + half],
        [x, y_center - half, z_center - half],
        [x, y_center + half, z_center - half],
    ], dtype=np.float64)


def tag_on_y_wall(y, x_center, z_center, half=2.5):
    """y=常数 竖墙上的标签"""
    return np.array([
        [x_center - half, y, z_center + half],
        [x_center + half, y, z_center + half],
        [x_center + half, y, z_center - half],
        [x_center - half, y, z_center - half],
    ], dtype=np.float64)


TAG36 = tag_on_x_wall(44.7, 21.3, 39.0)          # 模拟中墙西面 tag36
TAG40X = tag_on_x_wall(44.7, 28.5, 39.0)         # 同墙近邻（B 场景）
WALL_A = tag_on_x_wall(60.0, 52.0, 39.0)         # D 场景：东墙
WALL_B = tag_on_y_wall(60.0, 50.0, 39.0)         # D 场景：北墙


# =====================================================================
# 单帧 PnP 对照（歧义演示用）
# =====================================================================

def single_frame_pnp(pts3d, obs):
    """单帧 cv2.solvePnP：返回 (相机世界位置, 光轴方向, reproj)；无 cv2 返回 None"""
    if cv2 is None:
        return None
    ok, rvec, tvec = cv2.solvePnP(pts3d, obs, K, DIST)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    pos = (-R.T @ tvec).flatten()
    ori = (R.T @ np.array([0.0, 0.0, 1.0])).flatten()
    proj, _ = cv2.projectPoints(pts3d, rvec, tvec, K, DIST)
    err = float(np.mean(np.linalg.norm(proj[:, 0, :] - obs, axis=1)))
    return pos, ori, err


# =====================================================================
# 场景定义
# =====================================================================

# (name, 真实参数, 标签列表, frames=[(theta_deg, [标签索引...])])
# 真实参数：x_B, y_B, phi, e_x, e_y, z_c, pitch（e/z/pitch 各场景相同，聚焦场景差异）
COMMON = {"e_x": 2.5, "e_y": 1.0, "z_c": 39.0, "pitch": 10.0, "k": 1.0}  # pitch>0 = 俯视 10°，k=1

SCENARIOS = [
    (
        "A 同标签3帧",
        [24.0, 21.0, 0.0, COMMON["e_x"], COMMON["e_y"], COMMON["z_c"], COMMON["pitch"], COMMON["k"]],
        [TAG36],
        [(0.0, [0]), (-15.0, [0]), (15.0, [0])],
    ),
    (
        "B 同墙不同标签",
        [24.0, 24.5, 0.0, COMMON["e_x"], COMMON["e_y"], COMMON["z_c"], COMMON["pitch"], COMMON["k"]],
        [TAG36, TAG40X],
        [(0.0, [0, 1]), (-15.0, [0]), (15.0, [1])],
    ),
    (
        "C 仅2帧",
        [24.0, 21.0, 0.0, COMMON["e_x"], COMMON["e_y"], COMMON["z_c"], COMMON["pitch"], COMMON["k"]],
        [TAG36],
        [(-15.0, [0]), (15.0, [0])],
    ),
    (
        "D 不同墙非共面",
        [40.0, 40.0, 45.0, COMMON["e_x"], COMMON["e_y"], COMMON["z_c"], COMMON["pitch"], COMMON["k"]],
        [WALL_A, WALL_B],
        [(0.0, [0, 1]), (-15.0, [0]), (15.0, [1])],
    ),
]


# =====================================================================
# 主流程
# =====================================================================

def run_scenario(name, p_true, tags, frames_spec, noise_std, rng, show_pnp):
    # 合成观测
    frames = []
    for theta_deg, tag_idx in frames_spec:
        pts3d = np.vstack([tags[i] for i in tag_idx])
        R_cw, t_cw = camera_pose(p_true, theta_deg)
        obs, ok = project_points(pts3d, R_cw, t_cw)
        if not ok.all():
            print(f"[{name}] 警告: 有角点在相机后方，场景定义有误！")
        obs = obs + rng.normal(0.0, noise_std, obs.shape)
        frames.append((theta_deg, pts3d, obs))

    # 初值：真实值加扰动
    p0 = p_true.copy()
    p0[P_INDEX["x_B"]] += 15.0
    p0[P_INDEX["y_B"]] -= 10.0
    p0[P_INDEX["phi_deg"]] += 12.0
    p0[P_INDEX["e_x"]] = 0.0
    p0[P_INDEX["e_y"]] = 0.0
    p0[P_INDEX["z_c"]] += 5.0
    p0[P_INDEX["pitch_deg"]] = 0.0
    p0[P_INDEX["k_head"]] = 0.95   # k 初值带 5% 扰动，验证可辨识性

    try:
        p_est, rms = solve_joint(frames, p0)
    except RuntimeError as e:
        print(f"\n[{name}] 求解失败: {e}")
        return

    d_pos = np.linalg.norm(p_est[0:2] - p_true[0:2])
    d_phi = abs(p_est[P_INDEX["phi_deg"]] - p_true[P_INDEX["phi_deg"]])
    d_e = np.linalg.norm(p_est[3:5] - p_true[3:5])
    d_pitch = abs(p_est[P_INDEX["pitch_deg"]] - p_true[P_INDEX["pitch_deg"]])

    print()
    print("=" * 72)
    print(f"场景: {name}   噪声 σ={noise_std}px")
    print(f"帧: {[(t, idx) for t, idx in frames_spec]}")
    print(f"{'参数':<10}{'真实':>12}{'初值':>12}{'求解':>12}{'误差':>10}")
    for name_i in P_NAMES:
        i = P_INDEX[name_i]
        unit = P_UNIT[i]
        print(f"{name_i:<10}{p_true[i]:>10.2f}{unit}{p0[i]:>10.2f}{unit}"
              f"{p_est[i]:>10.2f}{unit}{p_est[i] - p_true[i]:>+9.2f}{unit}")
    print(f"残差 RMS: {rms:.3f} px")
    print(f"机体位置误差: {d_pos:.2f} cm   机体朝向误差: {d_phi:.2f} °   "
          f"偏心恢复误差: {d_e:.2f} cm   俯仰恢复误差: {d_pitch:.2f} °")
    verdict = "通过" if (d_pos < 1.0 and d_phi < 1.0) else "未达标"
    print(f">>> {name}: {verdict}")

    # A 场景附单帧 PnP 对比（歧义演示）
    if show_pnp and name.startswith("A"):
        print()
        print("  单帧 PnP 对比（同一场景，有 cv2 时）:")
        for theta_deg, pts3d, obs in frames:
            res = single_frame_pnp(pts3d, obs)
            if res is None:
                continue
            pos, ori, err = res
            R_cw_t, t_cw_t = camera_pose(p_true, theta_deg)
            p_true_cam = -R_cw_t.T @ t_cw_t
            d_pos_cam = float(np.linalg.norm(pos - p_true_cam))
            ori_true = R_cw_t.T @ np.array([0.0, 0.0, 1.0])
            cos_a = float(np.clip(np.dot(ori, ori_true), -1.0, 1.0))
            d_ori = float(np.degrees(np.arccos(cos_a)))
            print(f"    θ={theta_deg:+6.1f}°  相机位置误差 {d_pos_cam:6.2f}cm  "
                  f"光轴误差 {d_ori:6.2f}°  reproj {err:.2f}px")


def main():
    parser = argparse.ArgumentParser(description="多帧联合位姿求解 合成验证（阶段1）")
    parser.add_argument("--noise", type=float, default=0.5, help="角点像素噪声 σ（默认 0.5）")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--no-pnp", action="store_true", help="跳过单帧 PnP 对比")
    args = parser.parse_args()

    if not HAS_SCIPY:
        print("错误: 需要 scipy（TonyPi 已确认安装）。")
        return

    rng = np.random.default_rng(args.seed)
    print("=" * 72)
    print(f"多帧联合位姿求解 合成验证（阶段1）  噪声 σ={args.noise}px  seed={args.seed}")
    print(f"未知量: {P_NAMES}  头转系数 k 作为未知量求解（边界 0.8~1.2）")
    print("=" * 72)

    for name, p_true, tags, frames_spec in SCENARIOS:
        run_scenario(name, p_true, tags, frames_spec, args.noise, rng, not args.no_pnp)

    print()
    print("=" * 72)
    print("判定标准: 位置<1cm 且 朝向<1° = 歧义消除")
    print("重点看 B 场景（单面墙、帧间标签不同）——它是赛道的真实主场景。")
    print("=" * 72)


if __name__ == "__main__":
    main()
