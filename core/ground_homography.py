# -*- coding: utf-8 -*-
"""
地面单应标定与应用（ground_homography.py）——数字宫格关卡的度量定位基础。

本关卡场地无 AprilTag，机器人"我在哪 / 目标在哪 / 朝向差多少"全部依赖
地面几何。原理：相机拍地面时，地面点与去畸变后的像素点之间存在单应 H；
对 H（地面→归一化图像方向）做平面单应分解（Zhang 分解）即可恢复相机位姿，
再用已知相机高度做质量校验。

能力
----
1. solve(): 由 >=4 对「像素点 ↔ 地面点(cm)」最小二乘求 H；
2. pixels_to_ground(): 像素 -> undistortPoints 归一化 -> H -> 场地坐标；
   透视收缩与径向畸变在这一步一次性校正（本仓库"识别在像素空间、
   仅度量校正"决策的落地）；
3. decompose(): H -> (R, t, C)。C 的地面投影作为机器人位置锚点，
   视轴落地方向作为机器人前方 => 一帧同时得到位置/朝向/目标相对量；
4. head 补偿：头部偏航非中位时，按"绕世界竖直轴（经光心）旋转头部角"
   变换相机系坐标后再过 H（布局预扫宽扫档使用，格级精度足够；
   精确度量仍须中位头部档）。

约定
----
- 场地坐标系：x 向右、y 向前（入口在 y=0 侧）、原点=宫格左下角，单位 cm，
  与 goodluck 场地系同构；ESP32 位置编号 0..8（左上=0，左下=6 恒空）。
- 像素坐标为 fswebcam 原生 2592x1944 分辨率（与 camera_config 内参一致）。
- H 按俯仰舵机脉宽分档标定（俯仰不同 => 相机姿态不同 => H 不同）；
  运行时俯仰脉宽必须与标定档一致，否则度量失效。
- 头部脉宽-角度、俯仰档位常量沿用 camera_config（单一真源）。
"""

import json
import os
from datetime import datetime

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

from core.camera_config import (
    CAMERA_INTRINSIC, CAMERA_DISTORTION, HEAD_CENTER, SERVO_DEG_PER_US,
)
from core.paths import RESULT_DIR

# 宫格几何：1m 底板 3x3 布局，格间距（含缝）= 33.33cm
GRID_CELL_CM = 100.0 / 3.0

# 标定产物（多俯仰档共存）：{pitch_pulse: H条目}
HOMOGRAPHY_PATH = os.path.join(RESULT_DIR, "ninegrid_homography.json")

# 相机光心地面投影相对机器人双脚中心的固定偏移（cm，前正）。
# 粗值=头在躯干正上方、双脚中心略靠后；P3 实测后修正。
CAMERA_TO_BODY_FORWARD_CM = 4.0


def grid_cell_center(grid_pos):
    """ESP32 位置编号 0..8 -> 场地系格中心 (x, y) cm。

    位置布局（从入口上方俯视，入口在 y=0 一侧）：
        0 1 2   （最远排，y 最大）
        3 4 5
        6 7 8   （最近排；6=左下恒空）
    """
    col = grid_pos % 3
    row = grid_pos // 3  # 0 = 最远排
    x = (col + 0.5) * GRID_CELL_CM
    y = (2 - row + 0.5) * GRID_CELL_CM
    return np.array([x, y], dtype=np.float64)


def grid_intersections():
    """宫格 4x4 网格交点场地坐标（标定点击点的参考答案），顺序先行后列"""
    return [(c * GRID_CELL_CM, r * GRID_CELL_CM)
            for r in range(4) for c in range(4)]


def nearest_cell(field_xy, tol_cm=GRID_CELL_CM / 2.0):
    """场地坐标 -> 最近宫格位置编号；离格心超过 tol（默认半格距）返回 None"""
    best, best_d = None, tol_cm
    for g in range(9):
        c = grid_cell_center(g)
        d = float(np.hypot(field_xy[0] - c[0], field_xy[1] - c[1]))
        if d < best_d:
            best, best_d = g, d
    return best


def head_pulse_to_angle_deg(pulse):
    """头部脉宽 -> 偏航角（度，左转为正），与 robot_core.pulse_to_angle 一致"""
    return (pulse - HEAD_CENTER) * SERVO_DEG_PER_US


class GroundHomography:
    """单一俯仰档位的地面单应（含位姿分解与坐标变换）"""

    def __init__(self, H_ground_to_norm, pitch_pulse, head_pulse=HEAD_CENTER,
                 points=None, quality=None):
        self.H = np.asarray(H_ground_to_norm, dtype=np.float64)
        self.H /= self.H[2, 2]
        self.pitch_pulse = int(pitch_pulse)
        self.head_pulse = int(head_pulse)
        self.points = points or []
        self.quality = quality or {}
        self._decomposed = None
        self._H_inv = None

    # -----------------------------------------------------------------
    # 标定
    # -----------------------------------------------------------------

    @classmethod
    def from_pose(cls, cam_xy, cam_z, pitch_pulse, bearing_deg=0.0,
                  head_pulse=HEAD_CENTER):
        """由已知相机位姿解析构造 H（无标定文件时的开机自举）

        cam_xy: 光心地面投影 (场地系 cm)；cam_z: 相机高度 cm（实测 ≈39）
        pitch_pulse: 俯仰舵机脉宽（(1500-pulse)*0.09 = 俯角，越小越低头）
        精度受位姿假设限制（±3cm 级），只够布局扫的格归属判断；
        精确度量用 tools/calib_ninegrid.py 点击标定覆盖。
        """
        alpha = np.radians((1500 - pitch_pulse) * SERVO_DEG_PER_US)
        f = np.radians(bearing_deg)
        sa, ca = np.sin(alpha), np.cos(alpha)
        sf, cf = np.sin(f), np.cos(f)
        # 世界->相机旋转的行 = 相机三轴在世界系方向（见 tests/test_ground_homography）
        r = np.array([cf, -sf, 0.0])
        d = np.array([-sa * sf, -sa * cf, -ca])
        v = np.array([ca * sf, ca * cf, -sa])
        R = np.vstack([r, d, v])
        C = np.array([cam_xy[0], cam_xy[1], cam_z], dtype=np.float64)
        t = -R @ C
        H = np.column_stack([R[:, 0], R[:, 1], t])
        hg = cls(H, pitch_pulse, head_pulse)
        R2, t2, C2, diff = hg.decompose()
        hg.quality = {"h_col_norm_diff": float(diff),
                      "camera_center": [float(x) for x in C2],
                      "camera_height_cm": float(C2[2]),
                      "source": "analytic"}
        return hg

    @classmethod
    def solve(cls, points_px, points_ground, pitch_pulse,
              head_pulse=HEAD_CENTER):
        """由像素点-地面点对应求 H（最小二乘，>=4 对）

        points_px:      (N,2) 原生分辨率像素坐标
        points_ground:  (N,2) 场地坐标 cm
        """
        if cv2 is None:
            raise RuntimeError("ground_homography 需要 cv2")
        px = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
        gd = np.asarray(points_ground, dtype=np.float64).reshape(-1, 2)
        if len(px) < 4:
            raise ValueError(f"单应标定至少需要4对点，收到 {len(px)} 对")
        norm = _undistort_to_norm(px)  # (N,2) 归一化平面坐标
        H, _ = cv2.findHomography(
            gd.reshape(-1, 1, 2).astype(np.float64),
            norm.reshape(-1, 1, 2).astype(np.float64),
            method=0)  # 普通最小二乘；点击点由人工保证质量，不用 RANSAC
        if H is None:
            raise ValueError("单应求解失败（点近共线或重复？）")
        hg = GroundHomography(H, pitch_pulse, head_pulse,
                              points=[{"px": p.tolist(), "ground": g.tolist()}
                                      for p, g in zip(px, gd)])
        hg._check_quality()
        return hg

    def _check_quality(self):
        """质量检查：H 前两列模长应近似相等（真实单应的性质）。

        只记录 quality 字典不打印——逐帧自标定单应会频繁调用，
        告警与否由调用方按场景决定（标定工具打印，运行时按阈值丢弃）。
        """
        n1 = np.linalg.norm(self.H[:, 0])
        n2 = np.linalg.norm(self.H[:, 1])
        diff = abs(n1 - n2) / ((n1 + n2) / 2.0)
        R, t, C, diff = self.decompose()
        self.quality = {
            "h_col_norm_diff": float(diff),
            "camera_center": [float(v) for v in C],
            "camera_height_cm": float(C[2]),
        }
        return self.quality

    # -----------------------------------------------------------------
    # 坐标变换
    # -----------------------------------------------------------------

    def pixels_to_ground(self, pts_px, head_pulse=HEAD_CENTER):
        """像素坐标 -> 场地坐标 (N,2) cm

        H 存的是 地面->归一化图像 方向，此处取逆使用。
        head_pulse != 标定头部档时做偏航补偿（布局宽扫用，格级精度）。
        """
        pts = np.asarray(pts_px, dtype=np.float64).reshape(-1, 2)
        norm = _undistort_to_norm(pts)
        theta = np.radians(head_pulse_to_angle_deg(head_pulse)
                           - head_pulse_to_angle_deg(self.head_pulse))
        if abs(theta) > 1e-9:
            # 头部绕世界竖直轴（经光心）左转 θ：两相机系坐标差
            # p_A = R·Q(θ)·Rᵀ·p_B（R=标定档位姿；推导与验证见单测）。
            # 忽略舵机轴与光心的杠杆臂——布局扫格级精度足够，精确度量
            # 仍应使用中位头部档。
            R, _, _, _ = self.decompose()
            c, s = np.cos(theta), np.sin(theta)
            Q = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            M = R @ Q @ R.T
            pb = np.column_stack([norm, np.ones(len(norm))])
            pa = pb @ M.T
            norm = pa[:, :2] / pa[:, 2:3]
        if self._H_inv is None:
            self._H_inv = np.linalg.inv(self.H)
        return _apply_h(self._H_inv, norm)

    def ground_to_pixels(self, pts_ground):
        """场地坐标 -> 像素坐标（含畸变，调试叠加用）"""
        gd = np.asarray(pts_ground, dtype=np.float64).reshape(-1, 2)
        norm = _apply_h(self.H, gd)
        pts3 = np.column_stack([norm, np.ones(len(norm))]).reshape(-1, 1, 3)
        pix, _ = cv2.projectPoints(pts3.astype(np.float32),
                                   np.zeros(3), np.zeros(3),
                                   CAMERA_INTRINSIC, CAMERA_DISTORTION)
        return pix[:, 0, :]

    # -----------------------------------------------------------------
    # 位姿分解
    # -----------------------------------------------------------------

    def decompose(self):
        """H(地面->归一化图像) -> (R, t, C, col_norm_diff)

        R: 世界->相机旋转；t: 场地原点在相机系坐标；C: 相机中心（场地系，
        z=高度 cm）。场地坐标以 cm 给出 => 尺度自动确定，无需高度锚定；
        C_z 与实际相机高度的偏差即标定质量的直接度量。
        """
        if self._decomposed is not None:
            return self._decomposed
        H = self.H
        h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
        n1, n2 = np.linalg.norm(h1), np.linalg.norm(h2)
        diff = abs(n1 - n2) / ((n1 + n2) / 2.0)
        lam = 2.0 / (n1 + n2)
        r1, r2, t = h1 * lam, h2 * lam, h3 * lam
        if t[2] < 0:  # 相机在地面之上 => 场地原点在相机系 z>0
            r1, r2, t = -r1, -r2, -t
        R = np.column_stack([r1, r2, np.cross(r1, r2)])
        U, _, Vt = np.linalg.svd(R)  # 投影回 SO(3) 吸收点击噪声
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        C = -R.T @ t
        self._decomposed = (R, t, C, diff)
        return self._decomposed

    def ground_pose(self):
        """相机地面锚点与地面视线方向

        返回 (pos_xy, fwd_xy)：
        - pos_xy: 光心地面投影 (场地系 cm) —— 机器人位置锚点
        - fwd_xy: 视轴与地面交点方向（单位向量，场地系）—— 机器人"前方"
        """
        R, t, C, _ = self.decompose()
        w = R.T @ np.array([0.0, 0.0, 1.0])  # 相机光轴的世界方向
        if w[2] > -1e-6:
            raise ValueError("相机未俯视地面（光轴无向下分量），H 无效")
        s = -C[2] / w[2]
        hit = C + s * w
        fwd = hit[:2] - C[:2]
        fwd = fwd / np.linalg.norm(fwd)
        return C[:2].copy(), fwd

    def target_in_robot_frame(self, target_xy, head_pulse=HEAD_CENTER):
        """目标场地坐标 -> 机器人系相对量

        返回 (forward_cm, lateral_cm, bearing_err_deg)：
        - forward:  目标沿地面视线方向、从锚点（含相机-机体偏移修正）到
                    目标的纵向距离（前正）
        - lateral:  垂直视线方向的横向偏移（右正左负）
        - bearing_err: 目标方位角 - 视线方位角（度，右正左负）
        """
        pos, fwd = self.ground_pose()
        # 相机锚点在机体中心前方（前正偏移），机体中心 = 锚点后退
        pos = pos - fwd * CAMERA_TO_BODY_FORWARD_CM
        d = np.asarray(target_xy, dtype=np.float64) - pos
        right = np.array([fwd[1], -fwd[0]])  # 视线方向右转 90°（右正左负）
        forward = float(d @ fwd)
        lateral = float(d @ right)
        bearing_err = np.degrees(np.arctan2(lateral, forward))
        return forward, lateral, float(bearing_err)

    # -----------------------------------------------------------------
    # 持久化
    # -----------------------------------------------------------------

    def to_dict(self):
        return {
            "pitch_pulse": self.pitch_pulse,
            "head_pulse": self.head_pulse,
            "H_ground_to_norm": [[float(v) for v in row] for row in self.H],
            "points": self.points,
            "quality": self.quality,
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    @classmethod
    def from_dict(cls, d):
        hg = cls(np.array(d["H_ground_to_norm"], dtype=np.float64),
                 d["pitch_pulse"], d.get("head_pulse", HEAD_CENTER),
                 d.get("points"), d.get("quality"))
        hg._decomposed = None
        return hg

    def save(self, path=HOMOGRAPHY_PATH):
        """保存/更新到多档位标定文件（按键=俯仰脉宽）"""
        store = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    store = json.load(f)
            except (json.JSONDecodeError, ValueError):
                store = {}  # 空文件/损坏文件：重建
        store[str(self.pitch_pulse)] = self.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(store, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, pitch_pulse, path=HOMOGRAPHY_PATH):
        """按俯仰脉宽读取标定；缺失返回 None"""
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            store = json.load(f)
        d = store.get(str(int(pitch_pulse)))
        return cls.from_dict(d) if d else None


# =====================================================================
# 模块级工具
# =====================================================================

def _undistort_to_norm(pts_px):
    """像素 -> 去畸变归一化平面坐标 (N,2)"""
    pts = np.asarray(pts_px, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.undistortPoints(pts, CAMERA_INTRINSIC, CAMERA_DISTORTION)
    return out.reshape(-1, 2)


def _apply_h(H, pts_xy):
    """归一化/任意平面齐次点过单应，返回 (N,2)"""
    pts = np.asarray(pts_xy, dtype=np.float64).reshape(-1, 2)
    hom = np.column_stack([pts, np.ones(len(pts))])
    out = hom @ H.T
    return out[:, :2] / out[:, 2:3]
