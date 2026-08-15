# -*- coding: utf-8 -*-
"""巡线检测器：中心线提取 + 骨架化 + 弧长参数化 + 多项式拟合 + 纯追踪。

设计依据：《巡线识别算法设计.md》。能力：
  - 灰度模式（黑/白线，向后兼容）与 HSV 颜色模式（识别特定颜色线，红色自动合并两段）
  - 多条线（连通域分离）、横向线（分类 cross）、拐角（分类 corner）
  - 输出纯追踪前视点 lookahead_x，供关卡做左/右移、左/右转、前进/后退决策

所有输出量均在工作分辨率（work_width 宽）像素坐标系下，关卡阈值按此标定。
"""

import cv2
import numpy as np

from .detection import LineResult, LineSegment


class LineDetector:
    def __init__(self, line_color="dark", thresh=None, roi_ratio=0.5,
                 hsv_range=None, hsv_ranges=None,
                 min_area=80, lookahead_ratio=0.5, straightness_thresh=0.85,
                 work_width=640):
        self.line_color = line_color
        self.thresh = thresh
        self.roi_ratio = roi_ratio
        self.hsv_range = hsv_range          # 单一 HSV 范围，红色自动补 170~180 段
        self.hsv_ranges = hsv_ranges        # 额外多段 HSV（可选 list）
        self.min_area = min_area
        self.lookahead_ratio = lookahead_ratio
        self.straightness_thresh = straightness_thresh
        self.work_width = work_width

    # =====================================================================
    # Step 0：二值掩膜
    # =====================================================================
    def _make_mask(self, roi):
        if self.hsv_range is not None or self.hsv_ranges is not None:
            return self._color_mask(roi)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        ttype = cv2.THRESH_BINARY_INV if self.line_color == "dark" else cv2.THRESH_BINARY
        if self.thresh is None:
            ttype |= cv2.THRESH_OTSU
            th = 0
        else:
            th = self.thresh
        _, binary = cv2.threshold(gray, th, 255, ttype)
        return binary

    def _color_mask(self, roi):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        ranges = []
        if self.hsv_range is not None:
            lo, hi = self.hsv_range
            ranges.append((lo, hi))
            # 红色在 HSV 中分两段：h∈[0,~10] ∪ [~170,180]，自动补高段
            if lo[0] <= 8 and hi[0] <= 20:
                ranges.append(((170, lo[1], lo[2]), (180, hi[1], hi[2])))
        if self.hsv_ranges:
            ranges.extend(self.hsv_ranges)
        mask = None
        for lo, hi in ranges:
            m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
            mask = m if mask is None else cv2.bitwise_or(mask, m)
        return mask if mask is not None else np.zeros(roi.shape[:2], np.uint8)

    # =====================================================================
    # Step 1：连通域分割 + Step 2：骨架化 + Step 3：弧长参数化
    # =====================================================================
    @staticmethod
    def _zhang_suen(binary):
        """Zhang-Suen 细化：粗带 → 1px 中轴（纯 numpy，无额外依赖）。"""
        img = (binary > 0).astype(np.uint8)
        changed = True
        while changed:
            changed = False
            for step in (1, 2):
                h, w = img.shape
                p = np.zeros((h + 2, w + 2), np.uint8)
                p[1:-1, 1:-1] = img
                P2 = p[0:-2, 1:-1]; P3 = p[0:-2, 2:]; P4 = p[1:-1, 2:]; P5 = p[2:, 2:]
                P6 = p[2:, 1:-1]; P7 = p[2:, 0:-2]; P8 = p[1:-1, 0:-2]; P9 = p[0:-2, 0:-2]
                B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
                seq = [P2, P3, P4, P5, P6, P7, P8, P9]
                A = np.zeros_like(img)
                for i in range(8):
                    A += ((seq[i] == 0) & (seq[(i + 1) % 8] == 1)).astype(np.uint8)
                if step == 1:
                    cond = (B >= 2) & (B <= 6) & (A == 1) & (P2 * P4 * P6 == 0) & (P4 * P6 * P8 == 0)
                else:
                    cond = (B >= 2) & (B <= 6) & (A == 1) & (P2 * P4 * P8 == 0) & (P2 * P6 * P8 == 0)
                cond &= (img == 1)
                if cond.any():
                    changed = True
                    img[cond] = 0
        return (img * 255).astype(np.uint8)

    def _extract_segments(self, mask):
        """连通域 → 骨架化 → 追踪，返回 list[list[(x,y)]]。"""
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        segments = []
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < self.min_area:
                continue
            comp = (labels == i).astype(np.uint8)
            skel = self._zhang_suen(comp)
            segments.extend(self._trace(skel))
        return segments

    @staticmethod
    def _trace(skel):
        """把 1px 骨架追踪成有序点列（像素坐标 (x, y)）。"""
        ys, xs = np.nonzero(skel)
        if len(ys) == 0:
            return []
        pts = set(zip(xs.tolist(), ys.tolist()))

        def nbrs(p):
            x, y = p
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    q = (x + dx, y + dy)
                    if q in pts:
                        yield q

        deg = {p: sum(1 for _ in nbrs(p)) for p in pts}
        endpoints = [p for p in pts if deg[p] == 1]
        visited = set()
        segs = []

        def walk(start):
            seg = [start]
            visited.add(start)
            prev, cur = None, start
            while True:
                nxt = None
                for n in nbrs(cur):
                    if n == prev:
                        continue
                    if n not in visited:
                        nxt = n
                        break
                if nxt is None:
                    break
                visited.add(nxt)
                seg.append(nxt)
                if deg[nxt] != 2:  # 到达端点或分支点
                    break
                prev, cur = cur, nxt
            return seg

        for ep in endpoints:
            if ep not in visited:
                segs.append(walk(ep))
        # 闭合环 / 分支间残余：从任意未访问点继续
        for p in list(pts):
            if p not in visited:
                segs.append(walk(p))
        return segs

    # =====================================================================
    # Step 4~6：拟合 + 分类 + 纯追踪
    # =====================================================================
    def _analyze(self, seg, roi_w, roi_h):
        if len(seg) < 3:
            return None
        pts = np.array(seg, dtype=np.float64)          # (N,2) 图像坐标 x,y
        x, y = pts[:, 0], pts[:, 1]
        cx, cy = x.mean(), y.mean()
        xc, yc = x - cx, y - cy
        var_x = float(np.dot(xc, xc) / len(x))
        var_y = float(np.dot(yc, yc) / len(x))
        cov_xy = float(np.dot(xc, yc) / len(x))
        # 特征值 → 直线度
        tr = var_x + var_y
        det = var_x * var_y - cov_xy * cov_xy
        disc = max(0.0, (tr / 2.0) ** 2 - det)
        lam1 = tr / 2.0 + np.sqrt(disc)
        lam2 = tr / 2.0 - np.sqrt(disc)
        straightness = lam1 / (lam1 + lam2 + 1e-9)

        if straightness < self.straightness_thresh:
            orientation = "corner"
        elif var_y >= var_x:
            orientation = "follow"
        else:
            orientation = "cross"

        seg_obj = LineSegment(
            orientation=orientation,
            straightness=float(straightness),
            points=[(float(px), float(py)) for px, py in seg],
        )

        if orientation == "follow":
            # 机器人系：rx = x - 中心(右正)，ry = 底 - y(前正，0=机器人处)
            rx = x - roi_w / 2.0
            ry = roi_h - y
            a, b, c = np.polyfit(ry, rx, 2)            # rx = a·ry² + b·ry + c
            L = self.lookahead_ratio * roi_h
            seg_obj.curvature = float(a)               # 曲率(右弯正)
            seg_obj.lateral_offset = float(c)          # 机器人处横向偏移
            seg_obj.heading_deg = float(np.degrees(np.arctan(b)))  # 切线角
            seg_obj.lookahead_x = float(a * L * L + b * L + c)     # 前视点偏移
        elif orientation == "corner":
            near = seg[int(np.argmax(y))]              # 近端(离机器人近)
            far = seg[int(np.argmin(y))]               # 远端
            seg_obj.curvature = 1.0 if far[0] > near[0] else -1.0  # 转向方向
        else:  # cross
            seg_obj.heading_deg = 90.0
        return seg_obj

    # =====================================================================
    def detect(self, frame):
        if frame is None or frame.size == 0:
            return LineResult(exists=False)

        h, w = frame.shape[:2]
        roi = frame[int(h * (1.0 - self.roi_ratio)):, :]
        if roi.shape[1] > self.work_width:
            scale = self.work_width / roi.shape[1]
            roi = cv2.resize(roi, (self.work_width, int(roi.shape[0] * scale)))
        roi_h, roi_w = roi.shape[:2]

        mask = self._make_mask(roi)
        analyzed = []
        for seg in self._extract_segments(mask):
            s = self._analyze(seg, roi_w, roi_h)
            if s is not None:
                analyzed.append(s)

        if not analyzed:
            return LineResult(exists=False)

        def nearness(s):
            return max((p[1] for p in s.points), default=0.0)

        primary = max(analyzed, key=lambda s: (s.orientation == "follow", nearness(s)))
        others = [s for s in analyzed if s is not primary]
        confidence = float(primary.straightness)
        return LineResult(exists=True, primary=primary, others=others, confidence=confidence)
