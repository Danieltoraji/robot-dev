# -*- coding: utf-8 -*-
"""红色横带检测器（red_line_detector.py）——上下楼梯与识别跨障关卡的像素级感知。

目标：第一级台阶的红色胶条（下沿=地面线）与平地红色横杆（2x2cm 截面，
下沿=地面线）。两者共用同一检测：HSV 红色双区间 -> 形态学开闭 ->
连通域 -> 逐连通域提取**下沿采样点**（每列取该连通域最底部的像素）。

下沿点交给 core/ground_line_meter.py 投影到机器人本地地面系做直线拟合，
本模块只负责像素层，不做任何几何换算（与仓库"识别在像素空间、仅度量
校正"的决策一致，不做全图去畸变）。

红色在 HSV 色环两端，用双区间取并（继承参考实现 Final_Edition.py 的
阈值，现场用 tools/debug_vision.py 或 debug_server 复标）。
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - 与 robot_core 同样的非机器人环境屏蔽
    cv2 = None

# 红色双区间（继承参考实现；h∈[0,180]）
DEFAULT_RED_RANGES = (
    ((0, 50, 50), (12, 255, 255)),
    ((160, 50, 50), (179, 255, 255)),
)


@dataclass
class RedBandComponent:
    """单个红色连通域：面积 + 下沿采样点（像素坐标，x 向右 y 向下）。"""
    label_id: int
    area_px: int
    bottom_pts: np.ndarray                     # (K,2) float64, 列 [x, y]
    centroid_px: Tuple[float, float]


@dataclass
class RedLineDetection:
    """一帧的红色带检测结果（可能多目标：胶条+横杆可同帧）。"""
    exists: bool = False
    components: List[RedBandComponent] = field(default_factory=list)
    mask: Optional[np.ndarray] = None          # 调试用（debug_server/离线调参）

    @property
    def total_area_px(self) -> int:
        return sum(c.area_px for c in self.components)


class RedLineDetector:
    """红色横带检测：连通域分割 + 下沿点提取。

    hsv_ranges   : ((lower,upper),(lower,upper)) 红色双区间
    min_area     : 连通域最小面积（px），过滤噪点（参考实现 500@2592x1944）
    column_step  : 下沿点按列采样步长（px）；2592 宽下 8px 约 150 点/横杆
    morph_kernel : 形态学开闭核边长（px）
    """

    def __init__(self, hsv_ranges=DEFAULT_RED_RANGES, min_area=500,
                 column_step=8, morph_kernel=5, return_mask=False):
        self.ranges = [((np.array(lo, np.uint8)), np.array(hi, np.uint8))
                       for lo, hi in hsv_ranges]
        self.min_area = int(min_area)
        self.column_step = int(column_step)
        self.morph_kernel = int(morph_kernel)
        self.return_mask = bool(return_mask)

    # ------------------------------------------------------------------

    def detect(self, frame) -> RedLineDetection:
        result = RedLineDetection()
        if cv2 is None:
            raise RuntimeError("red_line_detector 需要 cv2")
        if frame is None or frame.size == 0:
            return result

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = None
        for lo, hi in self.ranges:
            m = cv2.inRange(hsv, lo, hi)
            mask = m if mask is None else cv2.bitwise_or(mask, m)
        if self.return_mask:
            result.mask = mask

        k = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.morph_kernel, self.morph_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        for comp_id in range(1, n):  # 0 是背景
            area = int(stats[comp_id, cv2.CC_STAT_AREA])
            if area < self.min_area:
                continue
            bottom_pts = self._bottom_edge_points(labels, comp_id, stats[comp_id])
            if len(bottom_pts) < 2:
                continue  # 不足以拟合直线
            cx, cy = centroids[comp_id]
            result.components.append(RedBandComponent(
                label_id=comp_id,
                area_px=area,
                bottom_pts=bottom_pts,
                centroid_px=(float(cx), float(cy)),
            ))
        result.exists = len(result.components) > 0
        return result

    # ------------------------------------------------------------------

    def _bottom_edge_points(self, labels, comp_id, stats) -> np.ndarray:
        """连通域每列最底部像素，按 column_step 采样 -> (K,2) [x, y]"""
        x0 = int(stats[cv2.CC_STAT_LEFT])
        w = int(stats[cv2.CC_STAT_WIDTH])
        ys, xs = np.nonzero(labels[...] == comp_id)
        if len(xs) == 0:
            return np.empty((0, 2))
        col_bottom = np.full(labels.shape[1], -1, dtype=np.int32)
        np.maximum.at(col_bottom, xs, ys.astype(np.int32))
        cols = np.arange(x0, x0 + w, self.column_step)
        cols = cols[(cols < len(col_bottom)) & (col_bottom[cols] >= 0)]
        if len(cols) == 0:
            return np.empty((0, 2))
        return np.column_stack([cols, col_bottom[cols]]).astype(np.float64)
