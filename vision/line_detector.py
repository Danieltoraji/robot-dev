# -*- coding: utf-8 -*-
"""巡线检测器：灰度阈值 + 质心，输出线相对画面中心的横向偏移。

纯传统 CV，无模型依赖，可在 RPi 上运行。阈值由关卡构造时传入。
"""

import cv2
import numpy as np

from .detection import LineResult


class LineDetector:
    """基于单通道阈值 + 质心的巡线检测器。

    line_color : "dark"（黑线，默认）或 "light"（浅色线）
    thresh     : 二值化阈值 0~255；None 时用大津法(OTSU)自动求阈值
    roi_ratio  : 只取画面底部该比例高度的区域（0~1），避免看太远导致误判
    """

    def __init__(self, line_color="dark", thresh=None, roi_ratio=0.5):
        self.line_color = line_color
        self.thresh = thresh
        self.roi_ratio = roi_ratio

    def detect(self, frame) -> LineResult:
        if frame is None or frame.size == 0:
            return LineResult(exists=False, confidence=0.0)

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 只看画面底部近处（即时修正用），避开远处无关内容
        roi = gray[int(h * (1.0 - self.roi_ratio)):, :]

        thresh_type = cv2.THRESH_BINARY_INV if self.line_color == "dark" else cv2.THRESH_BINARY
        if self.thresh is None:
            thresh_type |= cv2.THRESH_OTSU
            thresh_val = 0
        else:
            thresh_val = self.thresh
        _, binary = cv2.threshold(roi, thresh_val, 255, thresh_type)

        # 用白色像素的质心代表线位置（比单轮廓更稳）
        ys, xs = np.where(binary > 0)
        if len(xs) < 5:
            return LineResult(exists=False, confidence=0.0)

        cx = float(xs.mean())
        offset_x = cx - w / 2.0  # 右正左负
        # 线像素占比作为置信度
        confidence = min(1.0, len(xs) / binary.size)

        return LineResult(exists=True, offset_x=offset_x, confidence=confidence)
