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
                # 必须与 match_mask 的查询端**同一套归一化**：曾经这里用
                # "拉成正方"、查询端用"保比缩放"，两端不一致 ⇒ 自匹配分数
                # 反而低于异类（实测合成字形里 '1' 自匹配 0.44 < '3' 的 0.52）。
                norm = self.normalize_mask(g)
                if norm is None:
                    norm = cv2.resize(g, _CANONICAL)
                self.templates[str(ch)] = norm.astype(np.float32) / 255.0

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

    # ------------------------------------------------------------------
    # 单字符掩膜匹配（九宫格面板仲裁用）
    # ------------------------------------------------------------------

    @classmethod
    def normalize_mask(cls, mask):
        """单字符掩膜 → 统一归一化图（裁剪墨迹外框 + **保比缩放** + 居中）

        保比缩放（高对齐到画布、宽度按比例、水平居中）而不是"拉成正方"：
        拉方会把 "1" 撑成粗竖条，与 "7" 的相关系数反而更高（实测合成字形下
        "1" 被判成 "7"）。字形**长宽比本身**就是判据——真机远排的纵向压缩
        由色块在画面里的位置决定，不靠这一步校正。
        """
        if mask is None or mask.size == 0:
            return None
        m = mask if mask.ndim == 2 else cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        m = (m > 0).astype(np.uint8) * 255
        ys, xs = np.nonzero(m)
        if len(xs) < 4:
            return None
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        roi = m[y0:y1, x0:x1]
        if roi.shape[0] < 3 or roi.shape[1] < 3:
            return None
        th, tw = _CANONICAL
        inner_h = int(round(th * 0.85))          # 上下留白，避免贴边
        scale = inner_h / float(roi.shape[0])
        new_w = max(2, min(tw, int(round(roi.shape[1] * scale))))
        resized = cv2.resize(roi, (new_w, inner_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((th, tw), np.uint8)
        ox = (tw - new_w) // 2
        oy = (th - inner_h) // 2
        canvas[oy:oy + inner_h, ox:ox + new_w] = resized
        return canvas

    def match_mask(self, mask):
        """单字符掩膜 → (digit:int|None, conf:float, scores:{digit: 分数})

        conf = 最佳分 − 次佳分（**间隔**，不是绝对分）：绝对分受分辨率/模糊
        影响大，而"哪个模板明显更像"才是仲裁需要的判据。
        """
        if not self.templates:
            return None, 0.0, {}
        norm = self.normalize_mask(mask)
        if norm is None:
            return None, 0.0, {}
        roi = norm.astype(np.float32) / 255.0
        scores = {}
        for ch, tmpl in self.templates.items():
            scores[int(ch)] = float(
                cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED).item())
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        best, best_s = ranked[0]
        second_s = ranked[1][1] if len(ranked) > 1 else -1.0
        return best, float(best_s - second_s), scores

    def scores_for(self, mask):
        """完整分数表（诊断用）"""
        norm = self.normalize_mask(mask)
        if norm is None or not self.templates:
            return {}
        roi = norm.astype(np.float32) / 255.0
        return {int(ch): float(cv2.matchTemplate(roi, t, cv2.TM_CCOEFF_NORMED).item())
                for ch, t in self.templates.items()}
