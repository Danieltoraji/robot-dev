# -*- coding: utf-8 -*-
"""数字识别（tier1：连通域分割 + 模板匹配）。

决策 D4 结论：比赛数字通常是规整印刷体，先用「轮廓分割 + 模板匹配」——
无重依赖、可解释、可在 RPi 上跑；后续若精度不足，用同一 detect() 接口
替换为 OCR / 小 CNN 后端，关卡层无需改动。

模板由关卡在构造时注入（dict：数字字符 → 二值模板图，白字黑底）。
"""

import cv2
import numpy as np

from .detection import DigitResult

_CANONICAL = (28, 28)  # 模板匹配统一尺寸


class DigitRecognizer:
    """数字识别器（模板匹配档）。

    templates : {str: ndarray}，数字字符 → 模板图（二值，白字黑底）
    thresh    : 二值化阈值 0~255；None 用大津法(OTSU)
    min_area  : 最小数字连通域面积（像素）
    """

    def __init__(self, templates=None, thresh=None, min_area=30):
        self.thresh = thresh
        self.min_area = min_area
        self.templates = {}
        if templates:
            for ch, img in templates.items():
                g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                g = cv2.resize(g, _CANONICAL)
                self.templates[str(ch)] = g.astype(np.float32) / 255.0

    @staticmethod
    def _find_contours(binary):
        cnts = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return cnts[0] if len(cnts) == 2 else cnts[1]

    def detect(self, frame) -> DigitResult:
        if frame is None or frame.size == 0:
            return DigitResult("", 0.0)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.thresh is None:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(gray, self.thresh, 255, cv2.THRESH_BINARY_INV)

        # 分割出每个数字，按 x 从左到右排序
        boxes = []
        for c in self._find_contours(binary):
            x, y, w, h = cv2.boundingRect(c)
            if w * h < self.min_area:
                continue
            boxes.append((x, y, w, h))
        if not boxes:
            return DigitResult("", 0.0)
        boxes.sort(key=lambda b: b[0])

        if not self.templates:
            # 未注入模板：返回空串，仅保留位置信息供调试
            return DigitResult("", 0.0, center_px=self._center(boxes))

        digits, scores = [], []
        for (x, y, w, h) in boxes:
            roi = binary[y:y + h, x:x + w]
            roi = cv2.resize(roi, _CANONICAL).astype(np.float32) / 255.0
            best, best_score = None, -2.0
            for ch, tmpl in self.templates.items():
                score = cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED).item()
                if score > best_score:
                    best, best_score = ch, score
            if best is not None:
                digits.append(best)
                scores.append(best_score)

        if not digits:
            return DigitResult("", 0.0, center_px=self._center(boxes))

        conf = float(np.mean(scores))
        return DigitResult("".join(digits), conf, center_px=self._center(boxes))

    @staticmethod
    def _center(boxes):
        xs = [b[0] + b[2] / 2.0 for b in boxes]
        ys = [b[1] + b[3] / 2.0 for b in boxes]
        return (float(np.mean(xs)), float(np.mean(ys)))
