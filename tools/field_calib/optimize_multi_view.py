#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimize_multi_view.py —— 阶段3：真机采集数据 离线联合求解（A 方案）

读取 collect_multi_view.py 生成的 npz，对全部帧的角点观测做联合最小二乘：
    未知量: x_B, y_B, phi, e_x, e_y, z_c, pitch（头转系数 k 固定=1）
几何模型与阶段1合成验证完全一致（复用 synthetic_multi_view 的相机模型）。

输出：
    机体位姿 / 光心偏心 / 俯仰 求解结果与残差 RMS；
    与单帧 PnP 逐帧对比（位置/光轴误差）——检验联合解是否压掉了歧义；
    标定结果保存 archive/result/multiview_calib_<时间戳>.json（供阶段4集成）。

用法：
    python -m tools.field_calib.optimize_multi_view --data archive/result/multiview_<时间戳>.npz
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

# 允许直接运行本文件
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.paths import RESULT_DIR
from core.robot_core import TAG_CORNER_PERM
from levels.goodluck import tag_poses

# 几何模型与参数定义已收敛到 multiview_pose.py（单一实现）
from core.multiview_pose import (
    camera_pose, project_points, residual,
    K, DIST, P_NAMES, P_INDEX, P_UNIT, P_BOUNDS,
)


def load_npz(path):
    """npz → frames=[(theta_deg, pts3d, obs)]"""
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
            meta = {"raw": str(data["meta"][0])}
    return frames, meta


def initial_guesses(frames):
    """返回多组初值 [(名称, p0), ...]，避免单帧 PnP 歧义导致局部最优

    组1：第一帧单帧 PnP 解（有 cv2 时）
    组2：经验初值（场地中部）
    组3：按当前数据分布的经验初值（第一帧单帧解附近 + e 非零扰动）
    全部 clip 到参数边界内，保证 x0 可行。
    """
    lo = np.array(P_BOUNDS[0], dtype=np.float64)
    hi = np.array(P_BOUNDS[1], dtype=np.float64)
    guesses = []
    pnp_p0 = None
    if cv2 is not None:
        for theta_deg, pts3d, obs in frames:
            ok, rvec, tvec = cv2.solvePnP(pts3d, obs, K, DIST)
            if not ok:
                continue
            R, _ = cv2.Rodrigues(rvec)
            pos_cam = (-R.T @ tvec).flatten()
            ori_cam = (R.T @ np.array([0.0, 0.0, 1.0])).flatten()
            yaw = np.degrees(np.arctan2(ori_cam[1], ori_cam[0]))
            pitch0 = np.degrees(np.arcsin(np.clip(-ori_cam[2], -1.0, 1.0)))
            pnp_p0 = np.array([
                pos_cam[0], pos_cam[1],
                yaw - theta_deg,          # 机体朝向 ≈ 光轴 yaw − 头转角
                0.0, 0.0,                 # e 初值 0
                pos_cam[2], pitch0,
                1.0,                      # k 初值 1
            ], dtype=np.float64)
            break
    if pnp_p0 is not None:
        guesses.append(("单帧PnP", np.clip(pnp_p0, lo, hi)))
        # 组3：单帧解附近 + 偏心扰动
        g3 = pnp_p0.copy()
        g3[P_INDEX["e_x"]] = 2.0
        g3[P_INDEX["e_y"]] = 1.0
        guesses.append(("单帧PnP+e扰动", np.clip(g3, lo, hi)))
    guesses.append(("经验1", np.clip(np.array(
        [50.0, 50.0, 90.0, 0.0, 0.0, 40.0, 0.0, 1.0], dtype=np.float64), lo, hi)))
    guesses.append(("经验2", np.clip(np.array(
        [15.0, 25.0, 20.0, 0.0, 0.0, 35.0, -14.0, 1.0], dtype=np.float64), lo, hi)))
    return guesses


def frame_pnp_report(frames):
    """每帧单帧 PnP 质量报告：相机位置 / reproj；返回 (rows, 位置中位数)"""
    rows = []
    for fi, (theta_deg, pts3d, obs) in enumerate(frames):
        if cv2 is None:
            break
        ok, rvec, tvec = cv2.solvePnP(pts3d, obs, K, DIST)
        if not ok:
            rows.append((fi, theta_deg, None, None))
            continue
        R, _ = cv2.Rodrigues(rvec)
        pos = (-R.T @ tvec).flatten()
        proj, _ = cv2.projectPoints(pts3d, rvec, tvec, K, DIST)
        err = float(np.mean(np.linalg.norm(proj[:, 0, :] - obs, axis=1)))
        rows.append((fi, theta_deg, pos, err))
    pos_list = [r[2][:2] for r in rows if r[2] is not None]
    med = np.median(np.array(pos_list), axis=0) if pos_list else None
    return rows, med


def frame_residual(p, frame):
    """单帧残差（供逐帧 RMS 打印）"""
    return residual(p, [frame])


def _preprocess_fix_negatives(argv):
    """argparse 负数值兼容：--fix-e -0.2,-5.9 → --fix-e=-0.2,-5.9

    参数值以 '-' 开头时 argparse 会误判为选项，预处理合并为等号形式。
    """
    fix_opts = {"--fix-e", "--fix-pitch", "--fix-z", "--fix-k"}
    out = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in fix_opts and i + 1 < len(argv):
            nxt = argv[i + 1]
            if nxt.startswith("-") and not nxt.startswith("--"):
                out.append(f"{a}={nxt}")
                i += 2
                continue
        out.append(a)
        i += 1
    return out


def main():
    parser = argparse.ArgumentParser(description="真机多帧数据离线联合求解（阶段3）")
    parser.add_argument("--data", required=True, help="collect_multi_view.py 生成的 npz 路径")
    parser.add_argument("--drop", type=str, default="",
                        help="手动剔除帧号（逗号分隔，如 0,1）")
    parser.add_argument("--auto-drop", action="store_true",
                        help="自动剔除单帧 PnP 位置偏离中位数 >20cm 的帧")
    parser.add_argument("--max-theta", type=float, default=30.0,
                        help="只求解 |θ|≤此值的帧（默认 30°，排除 ±81° 发现帧；设 999 用全部）")
    parser.add_argument("--fix-e", type=str, default="",
                        help="固定光心偏心 e_x,e_y（cm，如 -0.2,-5.9；diag_kinematics 输出）")
    parser.add_argument("--fix-pitch", type=float, default=None,
                        help="固定俯仰角（度，如 -15；diag_kinematics 输出）")
    parser.add_argument("--fix-z", type=float, default=None,
                        help="固定相机高度 z_c（cm，可选）")
    parser.add_argument("--fix-k", type=float, default=None,
                        help="固定头转系数 k（诊断 k≈1，可固定 1.0）")
    args = parser.parse_args(_preprocess_fix_negatives(sys.argv[1:]))

    # ---- 固定参数：用边界锁死（lb==ub），残差函数无需改动 ----
    lo = list(P_BOUNDS[0])
    hi = list(P_BOUNDS[1])
    fixed_idx = set()
    if args.fix_e:
        try:
            ex, ey = [float(x) for x in args.fix_e.split(",")]
        except ValueError:
            print("--fix-e 格式错误（应为 e_x,e_y）")
            return
        lo[3] = hi[3] = ex
        lo[4] = hi[4] = ey
        fixed_idx.update({3, 4})
    if args.fix_pitch is not None:
        lo[6] = hi[6] = args.fix_pitch
        fixed_idx.add(6)
    if args.fix_z is not None:
        lo[5] = hi[5] = args.fix_z
        fixed_idx.add(5)
    if args.fix_k is not None:
        lo[7] = hi[7] = args.fix_k
        fixed_idx.add(7)
    bounds = (np.array(lo, dtype=np.float64), np.array(hi, dtype=np.float64))
    if fixed_idx:
        fixed_names = [P_NAMES[i] for i in sorted(fixed_idx)]
        print(f"固定参数: {fixed_names} = {[lo[i] for i in sorted(fixed_idx)]}")

    frames, meta = load_npz(args.data)
    if len(frames) < 2:
        print(f"有效帧数 {len(frames)} < 2，无法联合求解。")
        return

    n_corners = sum(len(obs) for _, _, obs in frames)
    print("=" * 72)
    print(f"数据: {args.data}")
    print(f"有效帧数: {len(frames)}   角点总数: {n_corners}")
    if meta:
        m = json.loads(meta) if isinstance(meta, str) else meta
        print(f"元信息: {m}")
    print(f"参数边界: {P_BOUNDS}")
    print("=" * 72)

    # ---- 帧质量报告（单帧 PnP） ----
    report, med_pos = frame_pnp_report(frames)
    if report:
        print("帧质量报告（单帧 PnP，同一机体位置应彼此接近）:")
        for fi, theta, pos, err in report:
            if pos is None:
                print(f"  帧{fi} θ={theta:+.1f}°: PnP 失败")
                continue
            if med_pos is not None:
                d = float(np.linalg.norm(pos[:2] - med_pos))
            else:
                d = 0.0
            flag = "  ← 疑似坏帧" if d > 20.0 else ""
            print(f"  帧{fi} θ={theta:+.1f}°: pos=({pos[0]:6.1f},{pos[1]:6.1f})  "
                  f"reproj={err:5.2f}px  距中位{d:5.1f}cm{flag}")
        print()

    # ---- 坏帧剔除 ----
    drop_set = set()
    if args.drop:
        try:
            drop_set = {int(x) for x in args.drop.split(",") if x.strip() != ""}
        except ValueError:
            print("--drop 格式错误（应为逗号分隔帧号）")
            return
    if args.auto_drop and report and med_pos is not None:
        for fi, theta, pos, err in report:
            if pos is not None and np.linalg.norm(pos[:2] - med_pos) > 20.0:
                drop_set.add(fi)
    if drop_set:
        keep = [(fi, f) for fi, f in enumerate(frames) if fi not in drop_set]
        print(f"剔除帧 {sorted(drop_set)}，保留 {len(keep)} 帧")
        frames = [f for _, f in keep]
        if len(frames) < 2:
            print("剔除后有效帧 < 2，无法求解。")
            return

    # ---- 角度过滤：±81° 发现帧的运动学与大角度偏差大，默认排除 ----
    if args.max_theta is not None and args.max_theta < 999.0:
        keep = [f for f in frames if abs(f[0]) <= args.max_theta]
        dropped = [f[0] for f in frames if abs(f[0]) > args.max_theta]
        if dropped:
            print(f"按 |θ|≤{args.max_theta:g}° 排除发现帧 θ={[f'{t:+.1f}' for t in dropped]}，"
                  f"保留 {len(keep)} 帧")
        frames = keep
        if len(frames) < 2:
            print("角度过滤后有效帧 < 2，无法求解。可用 --max-theta 999 或 --drop 调整。")
            return

    from scipy.optimize import least_squares

    # ---- 求解：固定参数从优化变量中剔除（trf 不支持 lb==ub） ----
    free_idx = [i for i in range(len(P_NAMES)) if i not in fixed_idx]
    free_lo = np.array([P_BOUNDS[0][i] for i in free_idx], dtype=np.float64)
    free_hi = np.array([P_BOUNDS[1][i] for i in free_idx], dtype=np.float64)
    fixed_vals = {i: lo[i] for i in fixed_idx}

    def pack(p_free):
        """自由参数子集 → 全 8 参数向量"""
        p = np.zeros(len(P_NAMES), dtype=np.float64)
        for i, v in fixed_vals.items():
            p[i] = v
        for j, i in enumerate(free_idx):
            p[i] = p_free[j]
        return p

    def unpack(p_full):
        return np.array([p_full[i] for i in free_idx], dtype=np.float64)

    def res_fixed(p_free, frames):
        return residual(pack(p_free), frames)

    # ---- 多初值求解，选残差最小者 ----
    best = None
    for gname, p0 in initial_guesses(frames):
        p0_free = np.clip(unpack(p0), free_lo, free_hi)
        try:
            res = least_squares(
                lambda p: res_fixed(p, frames), p0_free,
                bounds=(free_lo, free_hi), method="trf", max_nfev=4000,
                xtol=1e-12, ftol=1e-12, gtol=1e-12,
            )
        except ValueError as e:
            print(f"[初值 {gname}] 求解失败: {e}")
            continue
        p_est_g = pack(res.x)
        rms_g = float(np.sqrt(np.mean(res_fixed(res.x, frames) ** 2)))
        print(f"[初值 {gname}] cost={res.cost:.1f}  rms={rms_g:.2f}px  "
              f"success={res.success}  nfev={res.nfev}  "
              f"pitch={p_est_g[P_INDEX['pitch_deg']]:.2f}°  k={p_est_g[P_INDEX['k_head']]:.3f}")
        if best is None or res.cost < best[0].cost:
            best = (res, p_est_g)
    if best is None:
        print("所有初值均求解失败。")
        return
    res, p_est = best
    rms = float(np.sqrt(np.mean(res_fixed(res.x, frames) ** 2)))

    print()
    print(f"{'参数':<10}{'求解':>12}")
    for name in P_NAMES:
        i = P_INDEX[name]
        print(f"{name:<10}{p_est[i]:>10.2f}{P_UNIT[i]}")
    print(f"残差 RMS: {rms:.3f} px   success={res.success}   optimality={res.optimality:.3g}")
    print(f"机体位置: ({p_est[0]:.2f}, {p_est[1]:.2f})  机体朝向: {p_est[2]:.2f}°")
    print(f"光心偏心: e=({p_est[3]:.2f}, {p_est[4]:.2f})cm  相机高度: {p_est[5]:.2f}cm  "
          f"俯仰: {p_est[6]:.2f}°")

    print()
    print("每帧联合残差 RMS（定位坏帧用）:")
    for fi, (theta_deg, pts3d, obs) in enumerate(frames):
        fr = frame_residual(p_est, (theta_deg, pts3d, obs))
        print(f"  帧{fi} θ={theta_deg:+.1f}°: {float(np.sqrt(np.mean(fr ** 2))):.2f}px")

    # ---- 单帧 PnP 对比 ----
    if cv2 is not None:
        print()
        print("逐帧对比（单帧 PnP vs 联合解重投影）:")
        print(f"{'帧':<6}{'θ':>8}{'单帧位置':>22}{'联合位置':>22}{'位置差':>8}")
        for fi, (theta_deg, pts3d, obs) in enumerate(frames):
            res_pnp = None
            if cv2 is not None:
                ok, rvec, tvec = cv2.solvePnP(pts3d, obs, K, DIST)
                if ok:
                    R, _ = cv2.Rodrigues(rvec)
                    res_pnp = (-R.T @ tvec).flatten()
            R_cw, t_cw = camera_pose(p_est, theta_deg)
            p_c_joint = -R_cw.T @ t_cw
            if res_pnp is not None:
                d = float(np.linalg.norm(res_pnp[:2] - p_c_joint[:2]))
                print(f"{fi:<6}{theta_deg:>+7.1f}°({res_pnp[0]:>6.1f},{res_pnp[1]:>6.1f})"
                      f"({p_c_joint[0]:>6.1f},{p_c_joint[1]:>6.1f}){d:>8.2f}cm")
            else:
                print(f"{fi:<6}{theta_deg:>+7.1f}°    —        "
                      f"({p_c_joint[0]:>6.1f},{p_c_joint[1]:>6.1f})")
        print("注: 单帧位置 = 该帧相机位置；联合位置 = 由机体位姿+转角重投影的相机位置。")
        print("    位置差大 = 单帧歧义明显；位置差小 = 两者一致。")

    # ---- 保存标定结果 ----
    os.makedirs(RESULT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    calib_path = os.path.join(RESULT_DIR, f"multiview_calib_{ts}.json")
    calib = {
        "x_B": float(p_est[0]), "y_B": float(p_est[1]),
        "phi_deg": float(p_est[2]),
        "e_x_cm": float(p_est[3]), "e_y_cm": float(p_est[4]),
        "z_c_cm": float(p_est[5]), "pitch_deg": float(p_est[6]),
        "k_head": float(p_est[7]),
        "rms_px": rms,
        "data": args.data,
    }
    with open(calib_path, "w", encoding="utf-8") as f:
        json.dump(calib, f, ensure_ascii=False, indent=2)
    print(f"\n标定结果已保存: {calib_path}")
    print("（e / z_c / pitch 可固化到 robot_core 供阶段4集成使用）")


if __name__ == "__main__":
    main()
