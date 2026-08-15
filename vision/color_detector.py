# -*- coding: utf-8 -*-
"""颜色 / 球体检测器：HSV 阈值 + 轮廓，输出连通块列表。

纯传统 CV，无模型依赖，可在 RPi 上运行。颜色范围由关卡构造时传入。
"""

import cv2
import numpy as np

from .detection import Detection


class ColorBlobDetector:
    """检测指定 HSV 颜色范围内的连通块（如彩色球体）。

    hsv_range : (lower, upper)，各为 (h, s, v) 三元组；h∈[0,180]，s/v∈[0,255]
    label     : 命中该颜色的目标类别名，写入 Detection.cls
    min_area  : 最小连通域面积（像素），用于过滤噪点
    """

    def __init__(self, hsv_range, label="blob", min_area=50):
        self.lower = np.array(hsv_range[0], dtype=np.uint8)
        self.upper = np.array(hsv_range[1], dtype=np.uint8)
        self.label = label
        self.min_area = min_area

    @staticmethod
    def _find_contours(binary):
        # 兼容 OpenCV 3/4 的 findContours 返回值差异
        cnts = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return cnts[0] if len(cnts) == 2 else cnts[1]

    def detect(self, frame):
        results = []
        if frame is None or frame.size == 0:
            return results

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        # 开运算去噪
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=2)

        for c in self._find_contours(mask):
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            cx, cy = x + w / 2.0, y + h / 2.0
            # 圆形度（越接近 1 越像球），作为「球体」置信度信号，由关卡设阈值
            peri = cv2.arcLength(c, True)
            circularity = (4 * np.pi * area) / (peri * peri) if peri > 0 else 0.0
            confidence = float(min(1.0, circularity))
            results.append(Detection(
                cls=self.label,
                confidence=confidence,
                bbox=(float(x), float(y), float(w), float(h)),
                center_px=(cx, cy),
            ))
        return results
