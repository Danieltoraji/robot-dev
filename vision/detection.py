# -*- coding: utf-8 -*-
"""视觉检测统一数据模型。

所有检测器都返回这里定义的标准结构；关卡层只依赖这些结构与检测器接口，
不依赖具体检测器实现，便于换模型 / 换后端 / 仿真替换，避免跨关卡混乱。

坐标约定：
  - bbox / center_px：图像像素坐标（原点左上角，x 向右、y 向下）。
  - world_xy：可选，世界坐标 (x, y) cm，需配合 PnP / 地面标定投影，未投影为 None。
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List


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
class LineSegment:
    """一条中心线（一个连通组件）的几何描述。

    orientation    : "follow" 顺线(可跟随) / "cross" 横线(停止线等) / "corner" 拐角
    heading_deg    : 切线方向角（相对前向，右正左负，度）；cross 时为 ±90
    curvature      : follow 时为拟合曲率（正=向右弯）；corner 时为转向方向（+1 右 / -1 左）
    straightness   : 直线度 0~1（1=完全直线，趋近 0=拐角 / 急弧）
    lateral_offset : follow 时机器人处横向偏移（像素，右正左负）
    lookahead_x    : follow 时前视点横向偏移（纯追踪转向量）
    points         : 中心线像素点列 [(x, y), ...]，调试可视化用
    """
    orientation: str = "none"
    heading_deg: float = 0.0
    curvature: float = 0.0
    straightness: float = 0.0
    lateral_offset: float = 0.0
    lookahead_x: float = 0.0
    points: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class LineResult:
    """巡线检测结果（可能含多条线）。

    primary : 主目标线（离机器人最近、且优先取 follow 线）
    others  : 其余线（双线 / 横线 / 拐角等，交给关卡决定怎么用）
    """
    exists: bool = False
    primary: Optional[LineSegment] = None
    others: List[LineSegment] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class DigitResult:
    """数字识别结果。"""
    digits: str
    confidence: float
    center_px: Optional[Tuple[float, float]] = None
    world_xy: Optional[Tuple[float, float]] = None
