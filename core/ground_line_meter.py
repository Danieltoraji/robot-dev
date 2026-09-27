# -*- coding: utf-8 -*-
"""红色横带地面度量（ground_line_meter.py）——上下楼梯与识别跨障关卡的测距/对正核心。

原理（机器人本地系地面单应）
----------------------------
`GroundHomography.from_pose(cam_xy=(0,0), cam_z, pitch_pulse, head=1500)`
构造的单应，其地面坐标系是"随机器人走"的本地系：原点=当前光心地面投影、
y=视轴落地方向（前方）、x=右。相机高度与俯仰恒定时，相机相对该本地系的
位姿每帧相同 => 同一个 H 在机器人平移/旋转后依然有效（与 nine_grid 按
pitch 档标定后逐帧换位置复用同一 H 的用法同构；tests 验证）。

标定来源（2026-09-25 现场结论）
--------------------------------
本关**不走全 H 标定**：`tools/calib_stairs_hurdle.py` 需要的"机体坐标原点"
现场找不到（相机光心的地面投影没法在地面上标出来），已在工具文件头标注放弃。
`CALIB_PATH` 保留只是为了让旧产物自动生效，**目前不存在该文件**。

实际参数由 `tools/calib_ruler_profile.py`（卷尺刻度点击 + 含畸变物理模型）解出，
经 `build_meter()` 传入 `GroundHomography.from_pose`：光心离地 33.9cm、
安装下俯偏移 19.1°，现场 worst 距离误差 0.42cm。

度量输出：最近红色簇的下沿点在本地地面系的两轮 TLS 直线拟合 =>
(前向距离, 横向偏移, 方位角误差, 内点率)。
- 前向距离/横向偏移仅在"目标下沿在机器人脚下的地面平面上"时有效
  （胶条与横杆的下沿=地面线，平地段成立）；
- 站在台阶上时目标低于脚下平面，投影是绕光心落地点的均匀径向缩放：
  方向（bearing）不变、距离偏小 ~×0.76-0.89 —— 本流程把测距放在下完楼之后，
  所以只有降级时才会碰上这一条。

符号约定（与 nine_grid_sim 一致，右正左负）
------------------------------------------
- bearing_err_deg = +θ 表示机体相对横杆右偏 θ（应 turn_left 修正）；
- lateral_cm > 0 表示横杆可见段中心在机体右侧（应 right_move 对中）。
"""

import math
import os
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - 与 ground_homography 同样的非机器人环境屏蔽
    cv2 = None

from core.camera_config import (
    CAMERA_DISTORTION, CAMERA_INTRINSIC, HEAD_CENTER, SERVO_DEG_PER_US,
)
from core.ground_homography import GroundHomography
from core.paths import RESULT_DIR
from vision.red_line_detector import RedBandComponent

# 本关标定产物（与 ninegrid_homography.json 分开，避免互相覆盖）
CALIB_PATH = os.path.join(RESULT_DIR, "stairs_hurdle_calib.json")

CLUSTER_GAP_CM = 5.0   # 连通域按前向距离聚类的间隙阈值（胶条与横杆相距 >20cm）
RESIDUAL_TOL_CM = 2.0  # 直线拟合内点阈（垂直距离）
MIN_FIT_POINTS = 6     # 拟合所需最少下沿点


@dataclass
class LineMeasurement:
    """一次红色带度量结果（本地地面系）"""
    exists: bool = False
    reason: str = ""
    forward_cm: float = float("nan")       # 拟合直线与机体前方轴线交点的距离（直行将穿越横带的距离）
    lateral_cm: float = float("nan")       # 可见段中心横向偏移（右正）
    bearing_err_deg: float = float("nan")  # 直线方向偏角（机体右偏为正）
    inlier_ratio: float = 0.0
    n_points: int = 0
    n_clusters: int = 0
    secondary_forward_cm: float = float("nan")  # 次近簇前向距离（排障日志用）
    ground_pts: Optional[np.ndarray] = None     # 最近簇全部地面点（调试叠加用）


class GroundLineMeter:
    """下沿点 -> 本地地面系投影 -> 聚类 -> 两轮 TLS 直线拟合"""

    def __init__(self, homography: GroundHomography, degraded=False):
        self.hg = homography
        self.degraded = bool(degraded)  # True=from_pose 自举（未经现场标定）

    # ------------------------------------------------------------------

    def measure(self, components: Sequence[RedBandComponent],
                window_cm=None) -> LineMeasurement:
        """度量一帧。window_cm=(lo,hi) 时只认该前向距离窗口内的簇

        胶条（楼梯根部）与木条（楼梯之后）同色、同检测、都横跨赛道，
        唯一区别是在运动轴上的位置。调用方按当前步骤指定窗口，比
        "取最近的红东西"可靠——后者在胶条退到脚边被遮挡时，会无声地
        跳到远处的木条上，而那正好发生在最不能出错的时刻。
        window_cm=None 时保持原有"取最近簇"行为。
        """
        m = LineMeasurement()
        if not components:
            m.reason = "无红色连通域"
            return m
        pts_by_comp = []
        for comp in components:
            gd = self.hg.pixels_to_ground(comp.bottom_pts)
            pts_by_comp.append(gd)
        m.n_clusters = len(pts_by_comp)

        # 按前向距离聚类（组件级：每组件取下沿点平均 y），取最近簇
        comp_fwd = [float(np.mean(gd[:, 1])) for gd in pts_by_comp]
        order = np.argsort(comp_fwd)
        clusters = [[order[0]]]
        for idx in order[1:]:
            if comp_fwd[idx] - comp_fwd[clusters[-1][-1]] <= CLUSTER_GAP_CM:
                clusters[-1].append(idx)
            else:
                clusters.append([idx])
        cluster_fwd = [float(np.mean([comp_fwd[i] for i in cl]))
                       for cl in clusters]   # 沿 clusters 顺序递增，[0] 即最近簇
        if window_cm is None:
            nearest = clusters[0]
            if len(clusters) > 1:
                m.secondary_forward_cm = min(comp_fwd[i] for i in clusters[1])
        else:
            lo, hi = float(window_cm[0]), float(window_cm[1])
            keep = [ci for ci, f in enumerate(cluster_fwd) if lo <= f <= hi]
            if not keep:
                m.reason = (f"窗口 {lo:.0f}~{hi:.0f}cm 内无目标"
                            f"（共 {len(clusters)} 簇，最近 {cluster_fwd[0]:.1f}cm）")
                return m
            nearest = clusters[keep[0]]
            rest = [cluster_fwd[ci] for ci in range(len(clusters))
                    if ci != keep[0]]
            if rest:
                m.secondary_forward_cm = float(min(rest))
        pts = np.vstack([pts_by_comp[i] for i in nearest])
        m.ground_pts = pts
        m.n_points = len(pts)
        if len(pts) < MIN_FIT_POINTS:
            m.reason = f"下沿点不足（{len(pts)}<{MIN_FIT_POINTS}）"
            return m

        # 两轮 TLS 直线拟合（先全点，剔除外点后重拟合）
        inliers = pts
        for _ in range(2):
            direction, center = _tls_line(inliers)
            normal = np.array([-direction[1], direction[0]])
            resid = np.abs((inliers - center) @ normal)
            keep = resid <= RESIDUAL_TOL_CM
            if keep.all() or keep.sum() < MIN_FIT_POINTS:
                break
            inliers = inliers[keep]
        m.inlier_ratio = len(inliers) / len(pts)
        if len(inliers) < MIN_FIT_POINTS:
            m.reason = "内点不足，直线拟合失败"
            return m

        direction, center = _tls_line(inliers)
        if direction[0] < 0:  # 直线方向双向等价，统一成 x 分量非负
            direction = -direction
        m.exists = True
        m.bearing_err_deg = float(math.degrees(math.atan2(direction[1], direction[0])))
        # 前向距离 = 拟合直线与机体前方轴线（x=0）的交点——"继续直行将穿越
        # 横带的距离"。对正横带（bearing≈0）时等于可见段 y 中位数；斜带时
        # 可见段 y 中位数会偏离真值（可见窗口不对称），必须用交点。
        if abs(direction[0]) < 0.2:  # 直线近平行于前向轴，交点病态，退化为中位数
            m.forward_cm = float(np.median(inliers[:, 1]))
        else:
            m.forward_cm = float(center[1] - center[0] * direction[1] / direction[0])
        # 横向 = 可见段横坐标极差中点。不用中位数：下沿点按图像列采样，
        # 斜视/偏置时地面 x 密度不对称会使中位数有偏；极差中点对直带无偏，
        # 且"把可见段中点对到视轴"正是横移对中需要的语义。
        m.lateral_cm = float((np.max(inliers[:, 0]) + np.min(inliers[:, 0])) / 2.0)
        return m


# =====================================================================
# 构造与几何辅助
# =====================================================================

def build_meter(pitch_pulse, cam_height_cm, path=CALIB_PATH,
                head_pulse=HEAD_CENTER,
                pitch_offset_deg=0.0) -> GroundLineMeter:
    """优先读现场标定档；缺失降级 from_pose 解析自举（degraded=True）

    pitch_offset_deg：相机相对俯仰舵机的安装下俯偏移（度），见
    core.camera_config.CAM_PITCH_MOUNT_OFFSET_DEG。降级路径必须传它，
    否则自举出来的几何整体偏浅（相机实际比名义角度更低头）。
    """
    hg = GroundHomography.load(pitch_pulse, path=path)
    if hg is not None:
        return GroundLineMeter(hg, degraded=False)
    hg = GroundHomography.from_pose((0.0, 0.0), cam_height_cm,
                                    pitch_pulse, head_pulse=head_pulse,
                                    pitch_offset_deg=pitch_offset_deg)
    return GroundLineMeter(hg, degraded=True)


def pitch_down_deg(pitch_pulse) -> float:
    """俯仰脉宽 -> 名义俯角（度）：1500=水平，越小越低头

    不含安装偏移；完整俯角 = 本值 + pitch_offset_deg。
    """
    return (1500 - pitch_pulse) * SERVO_DEG_PER_US


_HALF_VFOV_DEG = None


def half_vfov_deg() -> float:
    """画幅垂直半视场（度，含畸变实算并缓存）

    ⚠️ 不要用针孔公式 atan((H/2)/fy)：它给 26.49°，**低估 2.5°**。
    本镜头桶形畸变很强（k1=−0.384），画幅边缘对应的真实角度更大，
    实算为 29.02°（2026-09-25 纠正）。凡是"用针孔公式推能看见多远/
    多近"的结论都不可信。
    """
    global _HALF_VFOV_DEG
    if _HALF_VFOV_DEG is None:
        if cv2 is None:
            _HALF_VFOV_DEG = 29.02  # 无 cv2 时的实测缺省（见 docstring）
        else:
            cx = CAMERA_INTRINSIC[0, 2]
            edge = np.array([[[cx, 0.0]]], dtype=np.float64)
            un = cv2.undistortPoints(edge, CAMERA_INTRINSIC, CAMERA_DISTORTION)
            _HALF_VFOV_DEG = float(math.degrees(math.atan(abs(float(un[0, 0, 1])))))
    return _HALF_VFOV_DEG


def min_visible_ground_cm(cam_height_cm, pitch_pulse,
                          pitch_offset_deg=0.0) -> float:
    """固定俯仰下"地面线目标"的可见下界（cm）

    下缘俯角 = 名义俯角 + 安装偏移 + 半视场；地面最近可见距离 =
    cam_z / tan(下缘俯角)。触发阈值窗口必须设在此值之上，否则"原地
    复测确认"会在盲区内失败。

    2026-09-25 纠正两处（旧版都让"最近能看见多近"偏大）：
    - 漏掉安装偏移（相机装在头上时向下偏约 15~17°）；
    - 用针孔公式算半视场（26.49°），实际含畸变为 29.02°。
    修正后 1040 档、h=34.5cm 约 3cm，与现场实测可见带 2~70cm 一致。
    下缘俯角已达或超过铅垂时返回 0（画面下缘就在脚下，再近的盲区是
    机体自遮挡，与镜头无关），旧版此处返回 inf 是错的。
    """
    total = math.radians(pitch_down_deg(pitch_pulse) + pitch_offset_deg
                         + half_vfov_deg())
    if total >= math.pi / 2:
        return 0.0
    if total <= 1e-6:
        return float("inf")
    return cam_height_cm / math.tan(total)


def _tls_line(pts: np.ndarray):
    """总最小二乘直线拟合：返回 (单位方向向量, 点集均值中心)"""
    center = pts.mean(axis=0)
    centered = pts - center
    # 主方向 = 协方差主特征向量（SVD，等价特征分解）
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    norm = np.linalg.norm(direction)
    return direction / norm, center
