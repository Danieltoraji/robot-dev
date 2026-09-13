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
    heading_deg    : 切线方向角（相对前向，右正左负，度）；cross 时为 ±90；
                     corner 时为近臂弦向角（机器人当前与来线的夹角）
    curvature      : follow 时为拟合曲率（正=向右弯）；corner 时为转向方向（+1 右 / -1 左，
                     由远臂末端相对肘点的横向符号决定）
    corner_deg     : corner 时观测到的拐角弯折角（近臂→远臂方向变化角，0~180°，恒正），
                     由肘部检测返回，用于闭环决定转弯程度；非 corner 为 0
    straightness   : 直线度 0~1（1=完全直线，趋近 0=拐角 / 急弧）；仅作 cross 判据与置信度，
                     corner 判据为肘部检测（弦向角突变），见 line_detector.py
    lateral_offset : 机器人处横向偏移（像素，右正左负）；follow/corner 均为最近点实测 rx，
                     不做多项式外推（防 L 形拟合爆炸）
    lookahead_x    : follow 时前视点横向偏移（纯追踪转向量）；拟合病态时回退为
                     ry≈L 处实测点 rx（插值不外推）
    far_ry         : 线远端（最远点）的 ry 值（像素，前正，越大越远）。线延伸到
                     视野远尽头时 ≈ roi_h；线末端进入视野（线快走完）时显著变小。
                     用于终点判定（线末端接近）。cross 时为 0。
    elbow_px       : corner 时肘点（拐点）图像坐标 (x, y)，非 corner 为 None
    elbow_ry       : corner 时肘点（拐角点）的前向距离（像素，roi_h - elbow_y）。
                     越小越近（拐角点接近脚下）。用于「前进接近拐角」阶段的终止
                     判据：肘点 ry 降到阈值以下说明拐角点已到脚下，可开始定点转弯。
                     非 corner 为 0
    points         : 中心线像素点列 [(x, y), ...]，调试可视化用
    """
    orientation: str = "none"
    heading_deg: float = 0.0
    curvature: float = 0.0
    corner_deg: float = 0.0
    straightness: float = 0.0
    lateral_offset: float = 0.0
    lookahead_x: float = 0.0
    far_ry: float = 0.0
    elbow_px: Optional[Tuple[float, float]] = None
    elbow_ry: float = 0.0
    points: List[Tuple[float, float]] = field(default_factory=list)
    # 地面单应映射结果（cm，右正左负/前正），仅在启用单应映射时填充，否则为 None
    lateral_offset_cm: Optional[float] = None   # 最近点横向偏移
    nearest_forward_cm: Optional[float] = None  # 最近点纵向距离
    lookahead_cm: Optional[float] = None        # lookahead 点横向偏移
    lookahead_forward_cm: Optional[float] = None  # lookahead 点纵向距离


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
