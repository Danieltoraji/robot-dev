#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
survey_field.py —— 场地自标定：多站位联合 Bundle Adjustment 精化 tag_poses

背景：
    tag_poses 为手工尺量值，各标签间相对几何互差 ~1cm（表现为多标签帧
    单帧 PnP 重投影 10~20px，而单标签帧 <1px）；墙体名义坐标（WALLS）
    与实物搭建亦有偏差。本工具用机器人采集的多站位多帧观测，把
    「每站位机体位姿 + 每标签面内坐标 + 墙面实际偏移 + 相机外参」
    联合解出，输出与导航坐标系自洽的精化 tag_poses。

平面分组与锚定（与 WALLS 导航坐标系的关系）：
    - 依 tag_poses 记录值自动按平面分组（某坐标恒定 → 该平面；聚类容差 2cm）；
    - 硬锚定三组以固定整体 6 自由度基准：东墙 x=95、南墙 y=0、地面 z=0
      （锚在哪面墙只影响结果表达在哪个名义系里；ROUTE/WALLS 均以名义系
      定义，故锚定到名义值即与导航自洽）；
    - 其余平面（中墙西/东面、北墙…）平面偏移作为待估量，弱先验拉向记录值
      ——墙是理想值、实物有偏差，故不能把标签焊死在名义平面上；
    - 左墙无标签不可观测，WALLS 保持名义值（8cm 避障余量可吸收其偏差）。

标签参数化（每标签 3 自由度，约束在其物理平面内）：
    平面 d·n + 面内坐标 (cu,cv) + 面内旋转 alpha；
    基准角点偏移取该组参考标签记录值（精确的 ±h 组合），保证 alpha=0、
    (cu,cv)=记录值 时完全复现记录角点——即从记录值出发做微调。

采集协议（机器人上执行，每站位一次）：
    沿线路 5 个站位（入口 + 四个停靠点附近），每站位跑一次 collect_multi_view.py；
    尽量让每个标签出现在 ≥2 个站位。

用法：
    python survey_field.py result/multiview_A.npz result/multiview_B.npz ...
    python survey_field.py --selftest        # PC 端合成数据自检，无需真机数据

输出：
    result/tag_poses_refined.json + 可直接粘贴回 levels/goodluck.py 的代码块
    result/multiview_extrinsics.json（供运行时三档联解使用）
    逐标签/逐站位残差报告（揪出贴歪/量错的单个标签）
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

try:
    from scipy.optimize import least_squares
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

from camera_config import CAMERA_INTRINSIC, CAMERA_DISTORTION, PITCH_UP_PULSE
from multiview_pose import (
    camera_pose, project_points, frame_pose_candidates, rot_from_rvec,
    normalize_extrinsics, EXTRINSICS_KEYS,
)

# 平面锚定：命中 (轴, 名义值) 的组平面偏移固定为名义值（固定整体基准）
ANCHOR_RULES = (
    ("z", 0.0, 2.5, 0.0),    # 地面
    ("y", 0.0, 2.5, 0.0),    # 南墙
    ("x", 95.0, 3.0, 95.0),  # 东墙
)
GROUP_LABELS = {
    ("x", 45.0): "中墙西面", ("x", 55.0): "中墙东面", ("x", 95.0): "东墙",
    ("y", 100.0): "北墙", ("y", 0.0): "南墙", ("z", 0.0): "地面",
}


def group_label(gkey):
    axis, _, val = gkey.partition("@")
    return GROUP_LABELS.get((axis, float(val)), gkey)


# =====================================================================
# 数据加载
# =====================================================================

def load_station(path, max_theta, known_ids):
    """npz → (站名, frames)；frames = [(theta_deg, {tid: corners(4,2)})]

    与 optimize_multi_view.load_npz 同源逻辑，但保留每标签独立角点。
    """
    data = np.load(path, allow_pickle=True)
    pulses = data["pulses"]
    thetas = data["thetas_nom"]
    f_idx = data["frame_idx"]
    t_ids = data["tag_ids"]
    corners = data["corners"]

    frames = []
    for fi in range(len(pulses)):
        theta = float(thetas[fi])
        if max_theta is not None and max_theta < 999.0 and abs(theta) > max_theta:
            continue
        mask = f_idx == fi
        if not mask.any():
            continue
        tags = {}
        for row in np.where(mask)[0]:
            tid = str(int(t_ids[row]))
            if tid not in known_ids:
                print(f"[{os.path.basename(path)}] 跳过未知标签 {tid}")
                continue
            tags[tid] = corners[row].reshape(4, 2).astype(np.float64)
        if tags:
            frames.append((theta, tags))
    return os.path.basename(path), frames


# =====================================================================
# 平面分组（依 tag_poses 记录值自动聚类）
# =====================================================================

def _plane_of(corners, tid):
    """检测标签所在平面：返回 (轴名, 常数坐标值)；常数轴不唯一则报错"""
    corners = np.asarray(corners, dtype=np.float64)
    const_axes = []
    for axis, name in ((0, "x"), (1, "y"), (2, "z")):
        col = corners[:, axis]
        if float(col.max() - col.min()) < 0.5:
            const_axes.append((name, float(col.mean())))
    if len(const_axes) != 1:
        raise ValueError(
            f"tag{tid} 角点无法唯一确定所在平面（常数轴: {const_axes}）"
            "——请检查 tag_poses 是否存在退化/重复角点")
    return const_axes[0]


def build_groups(tag_poses, cluster_tol=2.0):
    """把 tag_poses 按平面聚类 → groups

    groups: {gkey: dict(
        axis, axis_val(常数坐标均值,用于锚定/展示),
        d_rec(平面方程值均值 n·X), s(法向沿轴符号 ±1),
        n(法向单位矢), u, v,
        pat: {tid: (a(4,), b(4,)) 该标签自身角点的面内基准偏移},
        tags=[tid,...])}
    gkey 形如 "x@45"（轴@记录均值四舍五入）。

    注意两点易错处：
    - n 的方向随标签记录朝向而变（如朝东面墙 n=-x），平面方程须用 n·X=d，
      不能把轴坐标值直接当 d；
    - 「左上→右上」是观察者相对的，同组标签的角点朝向模式可能不同
      （地面标签尤其如此），故基准偏移按每标签自身计算。
    """
    planes = {}
    for tid, corners in tag_poses.items():
        axis, d = _plane_of(corners, tid)
        planes.setdefault(axis, []).append((tid, d, np.asarray(corners, np.float64)))

    clusters = []  # (axis, [tid, axis值, corners] 列表)
    for axis in ("x", "y", "z"):
        items = sorted(planes.get(axis, []), key=lambda it: it[1])
        cluster = []
        for it in items:
            if cluster and it[1] - cluster[-1][1] > cluster_tol:
                clusters.append((axis, cluster))
                cluster = []
            cluster.append(it)
        if cluster:
            clusters.append((axis, cluster))

    ax_idx = {"x": 0, "y": 1, "z": 2}
    axis_vec = np.zeros(3)
    groups_tmp = {}
    for axis, cluster in clusters:
        gkey = f"{axis}@{round(float(np.mean([c[1] for c in cluster])))}"
        ref_corners = cluster[0][2]

        # 基底强制取坐标轴方向（符号取自参考标签棱边方向）：
        # 墙面是平的、与轴对齐；若用记录标签的棱边直接当基底，
        # 记录误差带来的法向倾斜会被放大成锚定平面的系统性错配。
        axv = np.zeros(3)
        axv[ax_idx[axis]] = 1.0
        u_raw = ref_corners[1] - ref_corners[0]
        u_raw = u_raw / np.linalg.norm(u_raw)
        v_raw = ref_corners[3] - ref_corners[0]
        v_raw = v_raw / np.linalg.norm(v_raw)
        n_raw = np.cross(u_raw, v_raw)
        n_raw = n_raw / np.linalg.norm(n_raw)
        s = 1.0 if float(n_raw @ axv) >= 0 else -1.0
        n = s * axv
        i1, i2 = [i for i in (0, 1, 2) if i != ax_idx[axis]]
        e1 = np.zeros(3); e1[i1] = 1.0
        e2 = np.zeros(3); e2[i2] = 1.0
        u = e1 if float(u_raw @ e1) >= 0 else -e1
        v = e2 if float(v_raw @ e2) >= 0 else -e2

        pat, d_vals = {}, []
        for tid, _, corners in cluster:
            c = corners.mean(axis=0)
            d_vals.append(float(c @ n))
            pat[tid] = (np.array([(ci - c) @ u for ci in corners]),
                        np.array([(ci - c) @ v for ci in corners]))

        groups_tmp[gkey] = {
            "axis": axis,
            "axis_val": float(np.mean([c[1] for c in cluster])),
            "d_rec": float(np.mean(d_vals)),
            "s": s,
            "n": n, "u": u, "v": v,
            "pat": pat,
            "tags": [c[0] for c in cluster],
        }
    return groups_tmp


def anchor_value(group):
    """返回该组硬锚定的轴坐标值（名义平面）；不锚定（偏移待估）返回 None"""
    axis, d = group["axis"], group["axis_val"]
    for a_axis, a_nominal, tol, a_fixed in ANCHOR_RULES:
        if axis == a_axis and abs(d - a_nominal) <= tol:
            return a_fixed
    return None


# =====================================================================
# 参数装配
# =====================================================================

class SurveyProblem:
    """场地自标定联合 BA：变量装配 / 残差 / 精化角点重建

    变量布局:
      [0 .. 3S-1]           各站位 (x_B, y_B, phi_deg)
      [ext_off .. +4]       e_x, e_y, z_c, pitch_deg, k_head
      [tag_off .. +3T-1]    每标签 (cu, cv, alpha_deg)
      [g_off  .. +G-1]      各待估平面的偏移量 ddelta（相对记录均值）
    """

    def __init__(self, stations, groups, tag_poses, ext_init,
                 floor_weight=1.0, prior_sigma_cm=1.0, px_per_cm=None,
                 st_prior_sigma_cm=5.0, phi_prior_sigma_deg=5.0):
        self.stations = stations
        self.groups = groups
        self.tag_ids = list(tag_poses.keys())
        self.ext_init = normalize_extrinsics(ext_init)
        self.floor_weight = floor_weight
        self._group_of = {tid: g for g, gr in groups.items() for tid in gr["tags"]}

        self.n_st = len(stations)
        self.ext_off = 3 * self.n_st
        self.tag_off = self.ext_off + len(EXTRINSICS_KEYS)
        self.g_off = self.tag_off + 3 * len(self.tag_ids)
        self.free_groups = [g for g, gr in groups.items() if anchor_value(gr) is None]
        # 锚定的轴坐标值 → 模型平面值 d = 轴值 × s（s = n 沿轴符号）
        self.anchor_d = {g: anchor_value(gr) * gr["s"] for g, gr in groups.items()
                         if anchor_value(gr) is not None}
        self.n_var = self.g_off + len(self.free_groups)

        # 每标签初值 (cu, cv, alpha)，基于其所在组 u/v 坐标系
        self.tag_init = {}
        for gkey, gr in groups.items():
            for tid in gr["tags"]:
                c = np.asarray(tag_poses[tid], np.float64).mean(axis=0)
                self.tag_init[tid] = np.array([c @ gr["u"], c @ gr["v"], 0.0])

        # 弱先验的像素换算：σ_cm × px/cm
        if px_per_cm is None:
            px_per_cm = float(CAMERA_INTRINSIC[0, 0]) / 60.0  # 典型标签距离 ~60cm
        self.px_per_cm = px_per_cm
        self.prior_sigma_cm = prior_sigma_cm
        self.st_prior_sigma_cm = st_prior_sigma_cm
        self.phi_prior_sigma_deg = phi_prior_sigma_deg
        # 站位规范固定先验（拉向 PnP 种子；σ 宽松，只用于压平整体平移弱方向）
        self.st_seeds = None
        # 面内规范固定先验：各组标签面内中心均值拉向记录值均值
        # （法向有平面先验/锚定钉住；面内没有则标签网络沿墙滑动，
        #   近锚定端不动、远端漂移 ~1cm。组均值 σ≈1cm，不约束个体分布）
        self.ip_prior_sigma_cm = 1.0
        self.group_ip_rec = {}
        for gkey, gr in self.groups.items():
            us = [self.tag_init[t][0] for t in gr["tags"]]
            vs = [self.tag_init[t][1] for t in gr["tags"]]
            self.group_ip_rec[gkey] = (float(np.mean(us)), float(np.mean(vs)))

        # 观测记录：(站号, 帧号, 标签)，active 控制离群剔除
        self.obs = []
        for si, (_, frames) in enumerate(stations):
            for fi, (theta, tags) in enumerate(frames):
                for tid in tags:
                    if tid in self.tag_init:
                        self.obs.append((si, fi, tid))
        self.active = np.ones(len(self.obs), dtype=bool)

    # ---- 变量打包 ----
    def pack_init(self, st_seeds):
        x0 = np.zeros(self.n_var)
        for si, seed in enumerate(st_seeds):
            x0[3 * si:3 * si + 3] = seed
        for i, k in enumerate(EXTRINSICS_KEYS):
            x0[self.ext_off + i] = self.ext_init[k]
        for j, tid in enumerate(self.tag_ids):
            x0[self.tag_off + 3 * j: self.tag_off + 3 * j + 3] = self.tag_init[tid]
        return x0

    def bounds(self):
        lo = np.full(self.n_var, -np.inf)
        hi = np.full(self.n_var, np.inf)
        for si in range(self.n_st):
            lo[3 * si:3 * si + 2], hi[3 * si:3 * si + 2] = -30.0, 130.0
            lo[3 * si + 2], hi[3 * si + 2] = -720.0, 720.0
        ext_lo = {"e_x": -15.0, "e_y": -15.0, "z_c": 5.0, "pitch_deg": -30.0, "k_head": 0.7}
        ext_hi = {"e_x": 15.0, "e_y": 15.0, "z_c": 80.0, "pitch_deg": 30.0, "k_head": 1.3}
        for i, k in enumerate(EXTRINSICS_KEYS):
            lo[self.ext_off + i], hi[self.ext_off + i] = ext_lo[k], ext_hi[k]
        for j in range(len(self.tag_ids)):
            b0 = self.tag_off + 3 * j
            lo[b0], hi[b0] = -80.0, 180.0            # cu
            lo[b0 + 1], hi[b0 + 1] = -80.0, 180.0    # cv
            lo[b0 + 2], hi[b0 + 2] = -180.0, 180.0   # alpha
        for gi in range(len(self.free_groups)):
            lo[self.g_off + gi], hi[self.g_off + gi] = -20.0, 20.0
        return lo, hi

    # ---- 场景重建 ----
    def station_p8(self, x, si):
        e = {k: x[self.ext_off + i] for i, k in enumerate(EXTRINSICS_KEYS)}
        return np.array([x[3 * si], x[3 * si + 1], x[3 * si + 2],
                         e["e_x"], e["e_y"], e["z_c"], e["pitch_deg"], e["k_head"]],
                        dtype=np.float64)

    def tag_corners(self, x, tid):
        j = self.tag_ids.index(tid)
        cu, cv, alpha = x[self.tag_off + 3 * j: self.tag_off + 3 * j + 3]
        gkey = self._group_of[tid]
        gr = self.groups[gkey]
        d = self.anchor_d.get(gkey)
        if d is None:
            d = gr["d_rec"] + x[self.g_off + self.free_groups.index(gkey)]
        rad = np.radians(alpha)
        a, b = gr["pat"][tid]  # 该标签自身的角点面内基准偏移
        a_r = a * np.cos(rad) - b * np.sin(rad)
        b_r = a * np.sin(rad) + b * np.cos(rad)
        center = d * gr["n"] + cu * gr["u"] + cv * gr["v"]
        return center[None, :] + a_r[:, None] * gr["u"][None, :] + b_r[:, None] * gr["v"][None, :]

    # ---- 残差 ----
    def residual(self, x):
        poses = {}
        for si, (_, frames) in enumerate(self.stations):
            p8 = self.station_p8(x, si)
            for fi, (theta, _) in enumerate(frames):
                poses[(si, fi)] = camera_pose(p8, theta)
        rs = []
        for oi, (si, fi, tid) in enumerate(self.obs):
            if not self.active[oi]:
                continue
            theta, tags = self.stations[si][1][fi]
            R_cw, t_cw = poses[(si, fi)]
            proj, ok = project_points(self.tag_corners(x, tid), R_cw, t_cw)
            if not ok.all():
                rs.append(np.ones(8) * 1e3)  # 角点跑到相机后方：大惩罚
                continue
            blk = (proj - tags[tid]).reshape(-1)
            gr = self.groups[self._group_of[tid]]
            if gr["axis"] == "z" and abs(gr["axis_val"]) <= 2.5:
                blk = blk * self.floor_weight  # 地面标签擦视角，降权
            rs.append(blk)
        for gi in range(len(self.free_groups)):
            # 平面偏移弱先验：σ_cm 厘米偏移 ≈ 1 个残差单位的拉力
            # （此前误写成 ddelta/(σ_cm×px_per_cm)，弱了 ~100 倍，规范方向放任）
            rs.append(np.array([x[self.g_off + gi] * self.px_per_cm / self.prior_sigma_cm]))
        # 面内规范固定：各组面内中心均值拉向记录值（压平面内滑动弱方向）
        for gkey, (u_rec, v_rec) in self.group_ip_rec.items():
            idxs = [self.tag_off + 3 * j for j, t in enumerate(self.tag_ids)
                    if t in self.groups[gkey]["tags"]]
            u_est = float(np.mean([x[i] for i in idxs]))
            v_est = float(np.mean([x[i + 1] for i in idxs]))
            rs.append(np.array([
                (u_est - u_rec) * self.px_per_cm / self.ip_prior_sigma_cm,
                (v_est - v_rec) * self.px_per_cm / self.ip_prior_sigma_cm,
            ]))
        if self.st_seeds is not None:
            # 站位规范固定先验：拉向单帧 PnP 种子（记录坐标推得，~1-3cm 精度）。
            # 没有它，「全体站位+标签面内坐标整体平移」方向几乎无约束。
            for si in range(self.n_st):
                sx, sy, sphi = self.st_seeds[si]
                gx, gy, gphi = x[3 * si], x[3 * si + 1], x[3 * si + 2]
                dphi = (gphi - sphi + 180.0) % 360.0 - 180.0
                px_per_deg = self.px_per_cm * 1.05  # 60cm 处 1° ≈ 1.05cm 弧长
                rs.append(np.array([
                    (gx - sx) * self.px_per_cm / self.st_prior_sigma_cm,
                    (gy - sy) * self.px_per_cm / self.st_prior_sigma_cm,
                    dphi * px_per_deg / self.phi_prior_sigma_deg,
                ]))
        return np.concatenate(rs)

    # ---- 逐观测 rms（含先验项跳过）----
    def per_obs_rms(self, x):
        r = self.residual(x)
        out = np.full(len(self.obs), np.nan)
        off = 0
        for oi in range(len(self.obs)):
            if self.active[oi]:
                out[oi] = np.sqrt(np.mean(r[off:off + 8] ** 2))
                off += 8
        return out


# =====================================================================
# 初始化与求解
# =====================================================================

def station_seeds(stations, tag_poses_ref, ext_init):
    """每站位初值：取该站标签最多的一帧做多标签单帧解 → 反推机体位姿"""
    ext = normalize_extrinsics(ext_init)
    seeds = []
    for _, frames in stations:
        theta, tags = max(frames, key=lambda ft: len(ft[1]))
        pts3d = np.vstack([np.asarray(tag_poses_ref[t], np.float64) for t in tags])
        obs = np.vstack([tags[t] for t in tags])
        cands = frame_pose_candidates(pts3d, obs)
        if not cands:
            seeds.append(np.array([50.0, 50.0, 0.0]))
            continue
        rvec, tvec, _ = cands[0]
        R = rot_from_rvec(rvec)
        pc = (-R.T @ tvec).flatten()
        ori = (R.T @ np.array([0.0, 0.0, 1.0])).flatten()
        yaw = np.degrees(np.arctan2(ori[1], ori[0]))
        phi = yaw - ext["k_head"] * theta
        e_xy = np.array([ext["e_x"], ext["e_y"]])
        c, s = np.cos(np.radians(phi)), np.sin(np.radians(phi))
        body = pc[:2] - np.array([[c, -s], [s, c]]) @ e_xy
        seeds.append(np.array([body[0], body[1], phi]))
    return seeds


def run_survey(stations, tag_poses, ext_init, fix, floor_weight,
               loss="soft_l1", verbose=True):
    """执行场地自标定，返回 (problem, x_est, report_dict)"""
    groups = build_groups(tag_poses)
    ext_init = normalize_extrinsics(ext_init)
    for k, v in fix.items():
        if v is not None:
            ext_init[k] = float(v)

    prob = SurveyProblem(stations, groups, tag_poses, ext_init,
                         floor_weight=floor_weight)

    if verbose:
        print("=" * 72)
        print(f"场地自标定：{len(stations)} 站位 / {len(prob.obs)} 条标签观测 / "
              f"{len(prob.tag_ids)} 标签 / {len(prob.free_groups)} 个待估平面偏移")
        anchored = ", ".join(f"{group_label(g)}={d:g}" for g, d in prob.anchor_d.items())
        estimated = [group_label(g) for g in prob.free_groups]
        print(f"锚定平面: {anchored or '无'}    待估平面: {estimated or '无'}")
        print(f"变量 {prob.n_var} 维，loss={loss}(f_scale=2px)")
        print("=" * 72)

    seeds = station_seeds(stations, tag_poses, ext_init)
    prob.st_seeds = seeds
    lo, hi = prob.bounds()
    x0 = np.clip(prob.pack_init(seeds), lo, hi)
    # 第一遍：尺量坐标种子 → 精化标签
    res = least_squares(prob.residual, x0, bounds=(lo, hi), method="trf",
                        loss=loss, f_scale=2.0, max_nfev=4000)
    # 第二遍：用第一遍精化标签重算站位种子（规范固定先验随之收紧）
    refined1 = refined_tag_poses(prob, res.x)
    seeds2 = station_seeds(stations, refined1, ext_init)
    prob.st_seeds = seeds2
    x02 = res.x.copy()
    for si in range(prob.n_st):
        x02[3 * si:3 * si + 3] = seeds2[si]
    res = least_squares(prob.residual, np.clip(x02, lo, hi), bounds=(lo, hi),
                        method="trf", loss=loss, f_scale=2.0, max_nfev=4000)

    # ---- 离群观测剔除后重解一次 ----
    rms1 = prob.per_obs_rms(res.x)
    med = float(np.nanmedian(rms1))
    thresh = max(4.0, 3.0 * med)
    bad = [i for i, v in enumerate(rms1) if not np.isnan(v) and v > thresh]
    if bad and int(prob.active.sum()) > len(bad):
        for i in bad:
            prob.active[i] = False
        if verbose:
            dropped = [f"站{prob.obs[i][0]}帧{prob.obs[i][1]}tag{prob.obs[i][2]}"
                       for i in bad]
            print(f"[离群剔除] {len(bad)} 条观测 rms>{thresh:.1f}px: {dropped}，重解中…")
        res = least_squares(prob.residual, res.x, bounds=(lo, hi), method="trf",
                            loss=loss, f_scale=2.0, max_nfev=4000)

    x = res.x
    r = prob.residual(x)
    inlier_sq = []
    off = 0
    for oi in range(len(prob.obs)):
        if prob.active[oi]:
            inlier_sq.append(np.mean(r[off:off + 8] ** 2))
            off += 8
    rms_px = float(np.sqrt(np.mean(inlier_sq)))

    report = {
        "rms_px": rms_px,
        "cost": float(res.cost),
        "n_obs_total": len(prob.obs),
        "n_obs_active": int(prob.active.sum()),
    }
    return prob, x, report


# =====================================================================
# 结果输出
# =====================================================================

def refined_tag_poses(prob, x):
    return {tid: prob.tag_corners(x, tid).astype(np.float64) for tid in prob.tag_ids}


def print_report(prob, x, report, tag_poses):
    print()
    print("=" * 72)
    print(f"收敛：cost={report['cost']:.1f}  有效观测 {report['n_obs_active']}"
          f"/{report['n_obs_total']}  总 rms={report['rms_px']:.3f}px")
    print("=" * 72)

    print("\n[外参]")
    for i, k in enumerate(EXTRINSICS_KEYS):
        print(f"  {k:<10} {x[prob.ext_off + i]:>8.3f}")

    print("\n[平面偏移]（轴坐标空间；待估组：偏移 = 相对记录均值的平移量）")
    for gkey, gr in prob.groups.items():
        a = anchor_value(gr)
        if a is not None:
            print(f"  {group_label(gkey):<6}({gkey:<7}) 锚定 {gr['axis']}={a:7.2f}"
                  f"  （记录均值 {gr['axis_val']:.2f}）")
        else:
            dd = x[prob.g_off + prob.free_groups.index(gkey)]
            print(f"  {group_label(gkey):<6}({gkey:<7}) 估计 {gr['axis']}="
                  f"{gr['axis_val'] + dd * gr['s']:7.2f}"
                  f"  偏移 {dd:+5.2f}cm（记录均值 {gr['axis_val']:.2f}）")

    print("\n[逐标签]（moved = 精化中心相对记录中心位移；rms 大者疑似贴歪/量错）")
    obs_rms = {}
    for oi, v in enumerate(prob.per_obs_rms(x)):
        if not np.isnan(v):
            obs_rms.setdefault(prob.obs[oi][2], []).append(v)
    for tid in prob.tag_ids:
        new_c = prob.tag_corners(x, tid).mean(axis=0)
        rec_c = np.asarray(tag_poses[tid], np.float64).mean(axis=0)
        moved = float(np.linalg.norm(new_c - rec_c))
        rr = obs_rms.get(tid)
        rr_txt = f"{np.mean(rr):5.2f}px ×{len(rr)}" if rr else " 未观测  ×0"
        flag = ""
        if rr and np.mean(rr) > 2.0 * report["rms_px"] + 1.0:
            flag = "  ← 疑似贴歪/量错，请复核"
        print(f"  tag{tid:<4} {rr_txt}  moved={moved:5.2f}cm"
              f"  新中心=({new_c[0]:6.2f},{new_c[1]:6.2f},{new_c[2]:6.2f}){flag}")

    print("\n[逐站位]")
    for si, (name, frames) in enumerate(prob.stations):
        p8 = prob.station_p8(x, si)
        print(f"  {name:<36} 机体=({p8[0]:6.2f},{p8[1]:6.2f})"
              f" 朝向={p8[2]:7.2f}°  帧数={len(frames)}")


def print_paste_block(refined, rms_px, n_stations, ts):
    print("\n" + "=" * 72)
    print(f"以下代码块可整体替换 levels/goodluck.py 的 tag_poses 段落"
          f"（生成于 {ts}，rms={rms_px:.2f}px，站位 {n_stations}）：")
    print("=" * 72)
    print("tag_poses = {}")
    for tid, corners in refined.items():
        pts = ", ".join("[" + ", ".join(f"{v:.2f}" for v in row) + "]" for row in corners)
        print(f'tag_poses["{tid}"] = np.array([{pts}],dtype=np.float64,)')


def save_outputs(refined, prob, x, report, fix, out_dir, ts):
    os.makedirs(out_dir, exist_ok=True)
    ext = {k: float(x[prob.ext_off + i]) for i, k in enumerate(EXTRINSICS_KEYS)}

    tag_json = {
        "tag_poses": {tid: refined[tid].tolist() for tid in refined},
        "rms_px": report["rms_px"],
        "n_stations": len(prob.stations),
        "planes": {
            gkey: ({"anchored_axis_val": anchor_value(gr),
                    "recorded_axis_val": gr["axis_val"]}
                   if gkey in prob.anchor_d else
                   {"estimated_axis_val": gr["axis_val"] +
                    float(x[prob.g_off + prob.free_groups.index(gkey)]) * gr["s"],
                    "recorded_axis_val": gr["axis_val"]})
            for gkey, gr in prob.groups.items()
        },
        "generated": ts,
    }
    tag_path = os.path.join(out_dir, "tag_poses_refined.json")
    with open(tag_path, "w", encoding="utf-8") as f:
        json.dump(tag_json, f, ensure_ascii=False, indent=2)

    ext_json = dict(ext)
    ext_json.update({
        "pitch_pulse": PITCH_UP_PULSE,
        "rms_px": report["rms_px"],
        "source": "survey_field",
        "n_stations": len(prob.stations),
        "fixed": {k: v for k, v in fix.items() if v is not None},
        "generated": ts,
    })
    ext_path = os.path.join(out_dir, "multiview_extrinsics.json")
    with open(ext_path, "w", encoding="utf-8") as f:
        json.dump(ext_json, f, ensure_ascii=False, indent=2)

    print(f"\n已保存: {tag_path}")
    print(f"已保存: {ext_path}  （同步到机器人后，运行时三档联解自动加载）")
    return tag_path, ext_path


# =====================================================================
# 合成数据自检（PC 端，无需真机数据）
# =====================================================================

def perturb_tag(corners, rng, shift_cm=1.0, normal_cm=0.5, ang_deg=2.0):
    """保持平面性的标签扰动：面内平移 + 绕轴对齐法向旋转 + 沿法向微移

    模拟尺量记录与实物贴放之间的偏差（自检用）。
    旋转必须绕轴对齐法向（而非标签自身微倾的法向），
    否则角点会被转离平面，产生模型无法表达的剪切残差（~4px）。
    """
    axis_name, _ = _plane_of(corners, "?")
    ax = {"x": 0, "y": 1, "z": 2}[axis_name]
    n = np.zeros(3)
    n[ax] = 1.0
    i1, i2 = [i for i in (0, 1, 2) if i != ax]
    u = np.zeros(3)
    u[i1] = 1.0
    v = np.zeros(3)
    v[i2] = 1.0
    corners = np.asarray(corners, np.float64)
    center = corners.mean(axis=0)
    R = rot_from_rvec(n * np.radians(rng.uniform(-ang_deg, ang_deg)))
    out = center + (R @ (corners - center).T).T
    out = out + (rng.uniform(-shift_cm, shift_cm) * u
                 + rng.uniform(-shift_cm, shift_cm) * v
                 + rng.uniform(-normal_cm, normal_cm) * n)
    return out.astype(np.float64)


def selftest(noise_px=0.5, seed=0):
    """合成 5 站位观测（标签初值人为扰动 ~1cm），验证 BA 能否恢复真值"""
    from levels.goodluck import tag_poses as truth_poses

    rng = np.random.default_rng(seed)
    # pitch 17° 贴近真机 optimize_multi_view 的俯仰估计（~16.9°）
    ext_true = {"e_x": 2.5, "e_y": 1.0, "z_c": 39.0, "pitch_deg": 17.0, "k_head": 0.97}
    p8_true = np.array([0, 0, 0, ext_true["e_x"], ext_true["e_y"],
                        ext_true["z_c"], ext_true["pitch_deg"], ext_true["k_head"]],
                       dtype=np.float64)

    st_true = [  # 6 站位（入口+沿途+东侧），模拟真实采集协议
        (12.0, 22.0, 30.0),
        (20.0, 38.0, 70.0),
        (50.0, 80.0, 10.0),
        (75.0, 62.0, -120.0),
        (85.0, 30.0, -100.0),
        (65.0, 55.0, -150.0),
    ]
    thetas = [-30.0, -10.0, 10.0, 30.0]  # 贴近真实协议（多档转角，观测更密）
    W, H, margin = 2592.0, 1944.0, 40.0
    MID_WALL = (45.0, 55.0, 0.0, 60.0)  # 中墙 XY 范围（WALLS），遮挡检测用

    def mid_wall_blocks(p0_xy, p1_xy):
        """相机→标签连线是否穿过中墙矩形（Liang-Barsky 参数区间内即遮挡）"""
        x0, y0 = p0_xy
        x1, y1 = p1_xy
        dx, dy = x1 - x0, y1 - y0
        t0, t1 = 0.02, 0.98
        for p, q in ((-dx, x0 - MID_WALL[0]), (dx, MID_WALL[1] - x0),
                     (-dy, y0 - MID_WALL[2]), (dy, MID_WALL[3] - y0)):
            if abs(p) < 1e-9:
                if q < 0:
                    return False
            else:
                t = q / p
                if p < 0:
                    t0 = max(t0, t)   # 进入区间
                else:
                    t1 = min(t1, t)   # 离开区间
        return t0 < t1

    def render(p8, theta, pts3d, cam_xy):
        R_cw, t_cw = camera_pose(p8, theta)
        proj, ok = project_points(pts3d, R_cw, t_cw)
        if not ok.all():
            return None
        if ((proj[:, 0] < margin) | (proj[:, 0] > W - margin) |
                (proj[:, 1] < margin) | (proj[:, 1] > H - margin)).any():
            return None
        for corner in pts3d:  # 中墙遮挡：任一角点被挡则整标签不可用
            if mid_wall_blocks(cam_xy, corner[:2]):
                return None
        return proj + rng.normal(0.0, noise_px, proj.shape)

    stations = []
    seen = {tid: 0 for tid in truth_poses}

    # 真值场景须与模型假设自洽（否则 selftest 误报模型偏差）：
    # - 锚定组标签的常数坐标精确取锚定值；
    # - 自由组标签也统一到该组单一真实平面（同一面墙上的标签物理上共面——
    #   记录值里 151(x=44.5) 与 152(x=45.3) 的互差是尺量噪声，不是真有两面墙）。
    groups_true = build_groups(truth_poses)
    truth_snapped = {tid: c.copy() for tid, c in truth_poses.items()}
    ax_idx = {"x": 0, "y": 1, "z": 2}
    for gkey, gr in groups_true.items():
        a = anchor_value(gr)
        plane = a if a is not None else gr["axis_val"]
        for tid in gr["tags"]:
            truth_snapped[tid][:, ax_idx[gr["axis"]]] = plane

    for si, (sx, sy, sphi) in enumerate(st_true):
        p8 = p8_true.copy()
        p8[0], p8[1], p8[2] = sx, sy, sphi
        frames = []
        for theta in thetas:
            tags = {}
            for tid, corners in truth_snapped.items():
                proj = render(p8, theta, corners, (sx, sy))
                if proj is not None:
                    tags[tid] = proj
                    seen[tid] += 1
            if tags:
                frames.append((theta, tags))
        if not frames:
            print(f"[selftest] 站位 {si} ({sx},{sy}) 无可见标签，请调整站位")
            return False
        stations.append((f"st{si}", frames))

    # 初值：记录值做保平面扰动（模拟尺量/贴放误差）、外参用缺省（e=0,pitch=0,k=1）
    tag_init = {tid: perturb_tag(c, rng) for tid, c in truth_poses.items()}
    ext_init = normalize_extrinsics(None)

    prob = SurveyProblem(stations, build_groups(tag_init), tag_init, ext_init)
    seeds = station_seeds(stations, tag_init, ext_init)
    prob.st_seeds = seeds
    lo, hi = prob.bounds()
    x0 = np.clip(prob.pack_init(seeds), lo, hi)
    res = least_squares(prob.residual, x0, bounds=(lo, hi), method="trf",
                        loss="soft_l1", f_scale=2.0, max_nfev=4000)

    # ---- 第二遍：用第一遍的精化标签重算站位种子（精度 ~0.5cm，远优于尺量
    # 坐标种子的 ~2cm），规范固定先验随之收紧，自由平面组的弱方向漂移减小 ----
    refined1 = refined_tag_poses(prob, res.x)
    seeds2 = station_seeds(stations, refined1, ext_init)
    prob.st_seeds = seeds2
    x02 = res.x.copy()  # 标签/外参/平面偏移沿用第一遍解，只换站位种子
    for si in range(prob.n_st):
        x02[3 * si:3 * si + 3] = seeds2[si]
    res = least_squares(prob.residual, np.clip(x02, lo, hi), bounds=(lo, hi),
                        method="trf", loss="soft_l1", f_scale=2.0, max_nfev=4000)
    x = res.x
    rms = float(np.sqrt(np.mean(prob.residual(x)[:8 * int(prob.active.sum())] ** 2)))

    errs = []
    for tid in prob.tag_ids:
        if seen[tid] == 0:
            continue
        new_c = prob.tag_corners(x, tid).mean(axis=0)
        true_c = np.asarray(truth_snapped[tid], np.float64).mean(axis=0)
        errs.append((tid, float(np.linalg.norm(new_c - true_c)), seen[tid]))
    st_errs = []
    for si, (sx, sy, sphi) in enumerate(st_true):
        p8 = prob.station_p8(x, si)
        dxy = float(np.hypot(p8[0] - sx, p8[1] - sy))
        dphi = abs((p8[2] - sphi + 180.0) % 360.0 - 180.0)
        st_errs.append((si, dxy, dphi))
    k_est = x[prob.ext_off + 4]

    # 运行时指标（最终目的）：用精化坐标做多标签帧单帧 PnP，
    # 重投影应从尺量坐标的 ~4-20px 降到噪声水平（<1px）
    def frame_pnp_reproj(poses3d, tags):
        try:
            import cv2
        except Exception:
            return None
        obj = np.vstack([poses3d[t] for t in tags])
        img = np.vstack([tags[t] for t in tags])
        ok, rvec, tvec = cv2.solvePnP(obj, img, CAMERA_INTRINSIC, CAMERA_DISTORTION)
        if not ok:
            return None
        proj, _ = cv2.projectPoints(obj, rvec, tvec, CAMERA_INTRINSIC, CAMERA_DISTORTION)
        return float(np.mean(np.linalg.norm(proj[:, 0, :] - img, axis=1)))

    rr_ref, rr_rec = [], []
    refined = {tid: prob.tag_corners(x, tid) for tid in prob.tag_ids}
    for _, frames in stations:
        for theta, tags in frames:
            if len(tags) < 2:
                continue
            e1 = frame_pnp_reproj(tag_init, tags)
            e2 = frame_pnp_reproj(refined, tags)
            if e1 is not None and e2 is not None:
                rr_rec.append(e1)
                rr_ref.append(e2)

    print("=" * 72)
    print(f"[selftest] 合成 {len(stations)} 站位 / 噪声 σ={noise_px}px / "
          f"观测到 {len(errs)}/{len(truth_poses)} 个标签  收敛 rms={rms:.3f}px")
    # 门槛说明：绝对位置被规范固定先验钉在「记录值组均值」上，记录值本身
    # ±1cm 的误差决定了绝对精度地板（2 标签组均值误差可达 ~1cm）；
    # 方法保证的核心是运行时一致性（多标签帧 PnP 重投影），该指标为硬门槛。
    anchored_tags = {t for g in prob.anchor_d for t in prob.groups[g]["tags"]}
    for tid, e, n in sorted(errs, key=lambda t: -t[1]):
        if n < 2:
            gate = "  （单站可见，保持先验，不设门槛）"
        elif tid in anchored_tags:
            gate = "  （锚定面，门槛 1.5cm）"
        else:
            gate = "  （自由面，门槛 2.0cm——受墙体公差与规范地板限制）"
        print(f"  tag{tid:<4} 中心误差 {e:5.2f}cm  可见站数 {n}{gate}")
    for si, dxy, dphi in st_errs:
        print(f"  站位{si} 位置误差 {dxy:5.2f}cm  朝向误差 {dphi:5.2f}°")
    print(f"  k_head 估计 {k_est:.3f}（真值 {ext_true['k_head']}，误差 {abs(k_est - ext_true['k_head']):.3f}）")
    if rr_ref:
        print(f"  运行时指标: 多标签帧单帧 PnP 重投影 "
              f"记录坐标 {np.mean(rr_rec):.2f}px → 精化 {np.mean(rr_ref):.2f}px "
              f"(max {np.max(rr_ref):.2f}px)")

    gate_ok = all(e < (1.5 if tid in anchored_tags else 2.0)
                  for tid, e, n in errs if n >= 2)
    ok = (gate_ok and
          max(dxy for _, dxy, _ in st_errs) < 1.5 and
          max(dphi for _, _, dphi in st_errs) < 1.5 and
          abs(k_est - ext_true["k_head"]) < 0.03 and
          (not rr_ref or (np.mean(rr_ref) < 1.0 and np.max(rr_ref) < 2.0)))
    print(f">>> selftest: {'通过' if ok else '未达标'}"
          "（判定: 锚定面标签<1.5cm/自由面<2.0cm、站位<1.5cm/1.5°、k<0.03、"
          "多标签帧 PnP 重投影 avg<1px 且 max<2px）")
    return ok


# =====================================================================
# 主流程
# =====================================================================

def _fix_args(args):
    fix = {"e_x": None, "e_y": None, "z_c": None, "pitch_deg": None, "k_head": None}
    if args.fix_e:
        try:
            ex, ey = [float(v) for v in args.fix_e.split(",")]
        except ValueError:
            print("--fix-e 格式错误（应为 e_x,e_y）")
            sys.exit(1)
        fix["e_x"], fix["e_y"] = ex, ey
    fix["z_c"] = args.fix_z
    fix["pitch_deg"] = args.fix_pitch
    fix["k_head"] = args.fix_k
    return fix


def main():
    parser = argparse.ArgumentParser(description="场地自标定：多站位联合精化 tag_poses")
    parser.add_argument("npzs", nargs="*", help="collect_multi_view.py 生成的 npz（每站位一个）")
    parser.add_argument("--selftest", action="store_true", help="合成数据自检（无需真机数据）")
    parser.add_argument("--max-theta", type=float, default=30.0,
                        help="只用 |θ|≤此值的帧（默认 30，排除 ±81° 发现帧；999=全用）")
    parser.add_argument("--fix-e", type=str, default="", help="固定光心偏心 e_x,e_y（cm）")
    parser.add_argument("--fix-z", type=float, default=None, help="固定相机高度（cm）")
    parser.add_argument("--fix-pitch", type=float, default=None, help="固定俯仰（度）")
    parser.add_argument("--fix-k", type=float, default=None, help="固定头转系数")
    parser.add_argument("--floor-weight", type=float, default=1.0,
                        help="地面(z=0)标签残差权重（擦视角噪声大可调 0.3~0.5）")
    parser.add_argument("--loss", type=str, default="soft_l1",
                        help="scipy loss: soft_l1/huber/linear（默认 soft_l1）")
    parser.add_argument("--out", type=str, default="result", help="输出目录")
    args = parser.parse_args()

    if not HAS_SCIPY:
        print("错误: 需要 scipy")
        sys.exit(1)

    if args.selftest:
        ok = selftest()
        sys.exit(0 if ok else 1)

    if not args.npzs:
        print("错误: 请给至少一个站位 npz（每站位跑一次 collect_multi_view.py），"
              "或用 --selftest 做合成自检")
        sys.exit(1)

    from levels.goodluck import tag_poses
    stations = []
    for p in args.npzs:
        name, frames = load_station(p, args.max_theta, set(tag_poses.keys()))
        if not frames:
            print(f"[{name}] 过滤后无有效帧，跳过（可调 --max-theta）")
            continue
        stations.append((name, frames))
    if len(stations) < 2:
        print("错误: 有效站位 < 2，平面偏移与外参不可辨识。请补采站位。")
        sys.exit(1)

    fix = _fix_args(args)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prob, x, report = run_survey(stations, tag_poses, normalize_extrinsics(None),
                                 fix, args.floor_weight, loss=args.loss)
    refined = refined_tag_poses(prob, x)
    print_report(prob, x, report, tag_poses)
    print_paste_block(refined, report["rms_px"], len(stations), ts)
    save_outputs(refined, prob, x, report, fix, args.out, ts)


if __name__ == "__main__":
    main()
