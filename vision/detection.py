# -*- coding: utf-8 -*-
"""视觉检测统一数据模型。

所有检测器都返回这里定义的标准结构；关卡层只依赖这些结构与检测器接口，
不依赖具体检测器实现，便于换模型 / 换后端 / 仿真替换，避免跨关卡混乱。

坐标约定：
  - bbox / center_px：图像像素坐标（原点左上角，x 向右、y 向下）。
  - world_xy：可选，世界坐标 (x, y) cm，需配合 PnP / 地面标定投影，未投影为 None。
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Detection:
    """单个目标检测结果（通用目标 / 颜色块 / YOLO 通用）。

    cls          类别名，如 "ball" / "obstacle" / "digit_3"
    confidence   置信度 0~1
    bbox         (x, y, w, h) 图像像素，左上角 + 宽高
    center_px    (cx, cy) 图像像素中心
    world_xy     可选，世界坐标 (x, y) cm
    """
    cls: str
    confidence: float
    bbox: Tuple[float, float, float, float]
    center_px: Tuple[float, float]
    world_xy: Optional[Tuple[float, float]] = None


@dataclass
class LineResult:
    """巡线检测结果。

    offset_x   线质心相对画面中心的横向偏移（像素，右正左负），用于转向误差。
    heading    可选，线方向角（弧度），弯道判断用。
    """
    exists: bool
    offset_x: float = 0.0
    heading: Optional[float] = None
    confidence: float = 0.0


@dataclass
class DigitResult:
    """数字识别结果。"""
    digits: str
    confidence: float
    center_px: Optional[Tuple[float, float]] = None
    world_xy: Optional[Tuple[float, float]] = None
