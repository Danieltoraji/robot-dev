# -*- coding: utf-8 -*-
"""
multiview_pose.py —— 多视角联合位姿求解共享模块（单一实现，多方复用）

使用方：
  - synthetic_multi_view.py / optimize_multi_view.py / diag_kinematics.py（离线工具）
  - survey_field.py（场地自标定 Bundle Adjustment）
  - robot_core.py（运行时三档联解）
  - goodluck_sim.py（角点级仿真桩）

依赖约束：本模块会被 robot_core 在机器人上运行时导入，
因此只依赖 numpy/scipy/cv2 与 camera_config，
绝不 import robot_core / levels（避免循环依赖与硬件依赖）。

参数化（8 维向量 p，与历史离线工具一致）：
    p = [x_B, y_B, phi_deg, e_x, e_y, z_c, pitch_deg, k_head]
    机体位置(cm) / 机体朝向(度) / 光心偏心(cm) / 相机高度(cm) /
    俯仰(度, >0 为俯视) / 头转系数（实际转角 = k × 标称转角）
    相机位姿由 camera_pose(p, theta_deg) 给出，theta 为头部标称转角（度）。

消歧原理：单帧 4 共面点存在重投影都很准的两支解（IPPE two-fold）；
多帧共享同一套世界参数后，错误支的帧间相对旋转与舵机已知转角矛盾，
只有正确支能同时压低所有帧残差——帧间视差由光心偏心 e 的公转提供。
"""

import json
import os

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

from camera_config import (
    CAMERA_INTRINSIC, CAMERA_DISTORTION,
    PNP_FIELD_MIN, PNP_FIELD_MAX, PNP_CAM_Z_MIN, PNP_CAM_Z_MAX,
    PNP_ORI_Z_MAX, PNP_ORI_XY_MIN, PNP_REPROJ_ERR_MAX_PX,
)

# 兼容别名（离线工具的历史导入名）
K = CAMERA_INTRINSIC
DIST = CAMERA_DISTORTION

# ---------------------------------------------------------------------
# 参数向量与边界（自 synthetic_multi_view.py 迁入，保持一致）
# ---------------------------------------------------------------------
P_NAMES = ["x_B", "y_B", "phi_deg", "e_x", "e_y", "z_c", "pitch_deg", "k_head"]
P_INDEX = {n: i for i, n in enumerate(P_NAMES)}
P_UNIT = ["cm", "cm", "deg", "cm", "cm", "cm", "deg", "x"]
P_BOUNDS = (
    [-10.0, -10.0, -180.0, -10.0, -10.0, 5.0, -30.0, 0.8],
    [110.0, 110.0, 360.0, 10.0, 10.0, 80.0, 30.0, 1.2],
)

TAG_HALF = 2.5  # tag 物理半边长（cm）

# 外参缺省值：e=0 即「光心=机体中心」假设；z_c/pitch 为粗略经验值。
# 正式使用前应以 survey_field.py 的标定产物 multiview_extrinsics.json 为准。
# tilt_deg/tilt_phase_deg：头部偏航轴的倾斜（幅值+方位）。真机实测转头时
# pitch 随 θ 漂 2~4°（2026-08-29），即偏航轴不竖直；0 = 旧「纯竖直轴」模型。
DEFAULT_EXTRINSICS = {"e_x": 0.0, "e_y": 0.0, "z_c": 39.0, "pitch_deg": 0.0, "k_head": 1.0,
                      "tilt_deg": 0.0, "tilt_phase_deg": 0.0}
EXTRINSICS_KEYS = ("e_x", "e_y", "z_c", "pitch_deg", "k_head",
                   "tilt_deg", "tilt_phase_deg")


def normalize_extrinsics(ext=None):
    """补全外参字典缺失键为缺省值，返回新字典"""
    out = dict(DEFAULT_EXTRINSICS if ext is None else ext)
    for k in EXTRINSICS_KEYS:
        out.setdefault(k, DEFAULT_EXTRINSICS[k])
    return out


def load_extrinsics(path):
    """读取 multiview_extrinsics.json → 外参 dict；不存在/格式错返回 None"""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ext = {k: float(data[k]) for k in EXTRINSICS_KEYS if k in data}
    except Exception:
        return None
    return ext or None


# =====================================================================
# 正向几何模型（自 synthetic_multi_view.py 迁入，行为不变）
# =====================================================================

def rot_z(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_from_rvec(rvec):
    """Rodrigues 旋转向量 → 旋转矩阵（纯 numpy，避免运行时强依赖 cv2）"""
    v = np.asarray(rvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(v))
    if theta < 1e-12:
        return np.eye(3)
    k = v / theta
    kx = np.array([[0.0, -k[2], k[1]],
                   [k[2], 0.0, -k[0]],
                   [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * kx + (1.0 - np.cos(theta)) * (kx @ kx)


def camera_pose(p, theta_deg):
    """由参数向量 p 与头转角 θ，构造世界→相机 (R_cw, t_cw)（纯竖直偏航轴模型）

    实际头转角: θ_act = k_head × θ_nom
    光轴（相机 z）世界方向 = 水平前向(yaw) 俯视 pitch：
        a = (cos p·cos yaw, cos p·sin yaw, −sin p)     pitch>0 为俯视
    相机右（x）世界方向 = 水平右手边：r = (sin yaw, −cos yaw, 0)
    相机下（y）世界方向 = a × r
    光心世界位置: p_C = p_B + Rz(yaw) @ e + (0, 0, z_c)
    世界→相机:    X_cam = R_cw @ (X_world − p_C)
    """
    theta_act = p[P_INDEX["k_head"]] * theta_deg
    yaw = np.radians(p[P_INDEX["phi_deg"]] + theta_act)
    pitch = np.radians(p[P_INDEX["pitch_deg"]])
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)

    a = np.array([cp * cy, cp * sy, -sp])     # 光轴（相机 z）世界方向
    r = np.array([sy, -cy, 0.0])              # 相机右（x）世界方向
    d = np.cross(a, r)                        # 相机下（y）世界方向
    R_wc = np.column_stack([r, d, a])         # 相机→世界
    R_cw = R_wc.T                             # 世界→相机

    p_b = np.array([p[P_INDEX["x_B"]], p[P_INDEX["y_B"]], 0.0])
    R_yaw = rot_z(p[P_INDEX["phi_deg"]] + theta_act)
    e3 = np.array([p[P_INDEX["e_x"]], p[P_INDEX["e_y"]], 0.0])
    p_c = p_b + R_yaw @ e3 + np.array([0.0, 0.0, p[P_INDEX["z_c"]]])
    return R_cw, -R_cw @ p_c


def camera_pose_tilted(p, theta_deg, tilt_deg=0.0, tilt_phase_deg=0.0):
    """带偏航轴倾斜的相机位姿（tilt=0 时与 camera_pose 完全一致）

    头部旋转不再是绕竖直轴，而是绕倾斜轴：M(θ) = W·Rz(kθ)·W⁻¹，
    W 为绕水平方向 (cos ψ, sin ψ, 0) 倾斜 tilt_deg 的旋转。
    相机姿态 = Rz(φ)·M(θ)·B(pitch)，光心 = p_B + Rz(φ)·M(θ)·e + (0,0,z_c)。
    真机依据：2026-08-29 精化坐标逐帧 PnP 显示 pitch 随 θ 漂 2~4°（轴倾斜 ~2°），
    纯竖直轴模型在此数据上残差卡 ~8px。
    """
    if tilt_deg == 0.0:
        return camera_pose(p, theta_deg)
    theta_act = p[P_INDEX["k_head"]] * theta_deg
    phi = p[P_INDEX["phi_deg"]]
    pitch = np.radians(p[P_INDEX["pitch_deg"]])
    sp, cp = np.sin(pitch), np.cos(pitch)

    axis = np.array([np.cos(np.radians(tilt_phase_deg)),
                     np.sin(np.radians(tilt_phase_deg)), 0.0])
    W = rot_from_rvec(axis * np.radians(tilt_deg))
    M = W @ rot_z(theta_act) @ W.T

    # B(pitch)：pitch 后的相机基（yaw=0 时），满足 Rz(y)·B == 旧模型的 [r|d|a]
    B = np.array([
        [0.0, -sp, cp],
        [-1.0, 0.0, 0.0],
        [0.0, -cp, -sp],
    ])
    R_wc = rot_z(phi) @ M @ B
    R_cw = R_wc.T

    p_b = np.array([p[P_INDEX["x_B"]], p[P_INDEX["y_B"]], 0.0])
    e3 = np.array([p[P_INDEX["e_x"]], p[P_INDEX["e_y"]], 0.0])
    p_c = p_b + rot_z(phi) @ (M @ e3) + np.array([0.0, 0.0, p[P_INDEX["z_c"]]])
    return R_cw, -R_cw @ p_c


def ext_tilt(ext):
    """从外参 dict 提取 (tilt_deg, tilt_phase_deg)（缺省 0）"""
    ext = normalize_extrinsics(ext)
    return float(ext.get("tilt_deg", 0.0)), float(ext.get("tilt_phase_deg", 0.0))


def project_points(pts3d, R_cw, t_cw):
    """手写投影（径向畸变 k1,k2）；返回 (uv, 前方掩码)"""
    X = pts3d @ R_cw.T + t_cw
    z = X[:, 2]
    ok = z > 0.1
    x = X[:, 0] / z
    y = X[:, 1] / z
    r2 = x * x + y * y
    f = 1.0 + DIST[0] * r2 + DIST[1] * r2 * r2
    xd, yd = x * f, y * f
    u = K[0, 0] * xd + K[0, 2]
    v = K[1, 1] * yd + K[1, 2]
    return np.stack([u, v], axis=1), ok


def residual(p, frames, ext=None):
    """所有帧所有可见角点的 2D 重投影残差（观测 - 投影）

    ext 提供偏航轴倾斜（tilt_deg/tilt_phase_deg，缺省 0 = 纯竖直轴）。
    """
    tilt = ext_tilt(ext) if ext is not None else (0.0, 0.0)
    rs = []
    for theta_deg, pts3d, obs in frames:
        R_cw, t_cw = camera_pose_tilted(p, theta_deg, *tilt)
        proj, ok = project_points(pts3d, R_cw, t_cw)
        if not ok.all():
            # 有角点跑到相机后方：给大惩罚，防止优化利用不可见几何
            rs.append((np.ones(proj.shape) * 1e3).reshape(-1))
            continue
        rs.append((proj - obs).reshape(-1))
    return np.concatenate(rs)


def solve_joint(frames, p0):
    """全 8 参数联合最小二乘（离线工具用）；返回 (p_est, rms_px)"""
    if not HAS_SCIPY:
        raise RuntimeError("未安装 scipy，无法求解")
    res = least_squares(
        lambda p: residual(p, frames), p0,
        bounds=P_BOUNDS, method="trf", max_nfev=2000,
        xtol=1e-12, ftol=1e-12, gtol=1e-12,
    )
    return res.x, float(np.sqrt(np.mean(res.fun ** 2)))


# =====================================================================
# 位姿工具
# =====================================================================

def camera_optical_center(p, theta_deg, ext=None):
    """某头转角下的相机光心世界坐标（3 维）"""
    R_cw, t_cw = camera_pose_tilted(p, theta_deg, *ext_tilt(ext))
    return (-R_cw.T @ t_cw).flatten()


def body_orientation_xy(p):
    """机体朝向单位向量 [dx, dy]（由 phi_deg）"""
    phi = np.radians(p[P_INDEX["phi_deg"]])
    return np.array([np.cos(phi), np.sin(phi)])


# =====================================================================
# 初始化：单帧位姿候选（IPPE 双解枚举）与联合求解种子
# =====================================================================

def frame_pose_candidates(pts3d, obs):
    """单帧位姿候选（初始化用），按重投影误差升序返回 [(rvec, tvec, reproj_px)]

    平面观测（单标签）用 solvePnPGeneric/IPPE 取全部解，显式枚举歧义两支，
    替代「ITERATIVE 第一个解落在哪支看运气」的旧做法；
    非平面观测（多标签联合）无翻转歧义，IPPE 不可用时回退 ITERATIVE 单解。
    """
    if cv2 is None:
        return []
    obj = np.asarray(pts3d, dtype=np.float64)
    img = np.asarray(obs, dtype=np.float64)
    cands = []
    try:
        ok, rvecs, tvecs, errs = cv2.solvePnPGeneric(
            obj, img, K, DIST, flags=cv2.SOLVEPNP_IPPE)
        if ok:
            for rvec, tvec, e in zip(rvecs, tvecs, errs):
                # OpenCV 各版本 errs 形状不一（标量/(n,1) 数组），统一取首个元素
                e_val = float(np.ravel(e)[0]) if np.asarray(e).size else 0.0
                cands.append((np.asarray(rvec, dtype=np.float64).reshape(3),
                              np.asarray(tvec, dtype=np.float64).reshape(3),
                              e_val))
    except cv2.error:
        pass
    if not cands:
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, DIST)
        if ok:
            proj, _ = cv2.projectPoints(obj, rvec, tvec, K, DIST)
            err = float(np.mean(np.linalg.norm(proj[:, 0, :] - img, axis=1)))
            cands.append((rvec.reshape(3), tvec.reshape(3), err))
    cands.sort(key=lambda c: c[2])
    return cands


def seed_from_camera_pose(rvec, tvec, theta_deg, ext):
    """由单帧解出的相机位姿反推 8 参数初值（外参取 ext，机体位姿取反推值）"""
    R = rot_from_rvec(rvec)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
    pc = (-R.T @ tvec).flatten()
    ori = (R.T @ np.array([0.0, 0.0, 1.0])).flatten()
    yaw = np.degrees(np.arctan2(ori[1], ori[0]))
    pitch_deg = np.degrees(np.arcsin(np.clip(-ori[2], -1.0, 1.0)))
    phi = yaw - ext["k_head"] * theta_deg
    e_xy = np.array([ext["e_x"], ext["e_y"]])
    body_xy = pc[:2] - rot_z(phi)[:2, :2] @ e_xy
    return np.array([body_xy[0], body_xy[1], phi,
                     ext["e_x"], ext["e_y"], ext["z_c"], pitch_deg, ext["k_head"]],
                    dtype=np.float64)


def build_seeds(frames, ext, max_per_frame=2):
    """由各帧位姿候选生成联合求解初值集合（歧义分支枚举 + 经验兜底）"""
    ext = normalize_extrinsics(ext)
    seeds = []
    for theta_deg, pts3d, obs in frames:
        for rvec, tvec, _ in frame_pose_candidates(pts3d, obs)[:max_per_frame]:
            seeds.append(seed_from_camera_pose(rvec, tvec, theta_deg, ext))
    seeds.append(np.array([50.0, 50.0, 90.0, ext["e_x"], ext["e_y"],
                           ext["z_c"], 0.0, ext["k_head"]]))
    seeds.append(np.array([15.0, 25.0, 20.0, ext["e_x"], ext["e_y"],
                           ext["z_c"], -14.0, ext["k_head"]]))
    return seeds


# =====================================================================
# 运行时入口：固定外参的 3 参数联合求解 + 门控
# =====================================================================

def solve_joint_3dof(frames, ext=None, seeds=None, max_nfev=1000):
    """固定外参 ext，联合最小二乘只解机体位姿 (x_B, y_B, phi_deg)

    frames: [(theta_deg, pts3d, obs), ...]，theta 为头部标称转角（度）
    ext:    外参 dict（缺键补 DEFAULT_EXTRINSICS）
    返回 (p_est 8 维或 None, rms_px, info dict)；
    info 含 per_frame_rms / n_frames / n_seeds。
    """
    if not HAS_SCIPY:
        raise RuntimeError("未安装 scipy，无法联合求解")
    ext = normalize_extrinsics(ext)
    if not frames:
        return None, float("inf"), {}
    if seeds is None:
        seeds = build_seeds(frames, ext)
    lo = np.array([P_BOUNDS[0][0], P_BOUNDS[0][1], P_BOUNDS[0][2]], dtype=np.float64)
    hi = np.array([P_BOUNDS[1][0], P_BOUNDS[1][1], P_BOUNDS[1][2]], dtype=np.float64)

    def res3(p3):
        p = np.array([p3[0], p3[1], p3[2],
                      ext["e_x"], ext["e_y"], ext["z_c"],
                      ext["pitch_deg"], ext["k_head"]], dtype=np.float64)
        return residual(p, frames, ext=ext)

    best = None
    for p0 in seeds:
        p0c = np.clip(np.asarray(p0, dtype=np.float64)[:3], lo, hi)
        try:
            r = least_squares(res3, p0c, bounds=(lo, hi), method="trf",
                              max_nfev=max_nfev)
        except Exception:
            continue
        rms = float(np.sqrt(np.mean(r.fun ** 2)))
        if best is None or rms < best[1]:
            p_full = np.array([r.x[0], r.x[1], r.x[2],
                               ext["e_x"], ext["e_y"], ext["z_c"],
                               ext["pitch_deg"], ext["k_head"]], dtype=np.float64)
            best = (p_full, rms)
    if best is None:
        return None, float("inf"), {"n_seeds": len(seeds)}
    p_est, rms = best
    per_frame = [float(np.sqrt(np.mean(residual(p_est, [f]) ** 2))) for f in frames]
    info = {"per_frame_rms": per_frame, "n_frames": len(frames), "n_seeds": len(seeds)}
    return p_est, rms, info


def gate_solution(p_est, frames, ext=None, reproj_max=None):
    """联合解物理合理性门控；返回问题列表（空 = 通过）

    检查项与 robot_core 单帧门控同源（camera_config.PNP_*）：
      - 各帧重投影 rms ≤ reproj_max（默认 PNP_REPROJ_ERR_MAX_PX）
      - θ=0 光心在场地范围内、相机高度合理
      - 各帧光轴近水平（|z| 分量、XY 下限）
    """
    if reproj_max is None:
        reproj_max = PNP_REPROJ_ERR_MAX_PX
    problems = []
    p_c = camera_optical_center(p_est, 0.0)
    if not (PNP_FIELD_MIN <= p_c[0] <= PNP_FIELD_MAX and
            PNP_FIELD_MIN <= p_c[1] <= PNP_FIELD_MAX):
        problems.append(f"光心({p_c[0]:.1f},{p_c[1]:.1f})超出场地范围")
    if not (PNP_CAM_Z_MIN <= p_est[P_INDEX["z_c"]] <= PNP_CAM_Z_MAX):
        problems.append(f"相机高度z={p_est[P_INDEX['z_c']]:.1f}cm不合理")
    for theta_deg, _, _ in frames:
        R_cw, t_cw = camera_pose_tilted(p_est, theta_deg, *ext_tilt(ext))
        ori = R_cw.T @ np.array([0.0, 0.0, 1.0])
        if abs(ori[2]) > PNP_ORI_Z_MAX:
            problems.append(f"θ={theta_deg:+.1f}° 朝向不水平(z分量{ori[2]:.2f})")
        if np.linalg.norm(ori[:2]) < PNP_ORI_XY_MIN:
            problems.append(f"θ={theta_deg:+.1f}° 朝向接近垂直")
    for i, f in enumerate(frames):
        fr = residual(p_est, [f], ext=ext)
        rms = float(np.sqrt(np.mean(fr ** 2)))
        if rms > reproj_max:
            problems.append(f"帧{i}(θ={f[0]:+.1f}°) 重投影{rms:.2f}px过大")
    return problems
