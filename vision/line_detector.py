# -*- coding: utf-8 -*-
"""巡线检测器：中心线提取 + 骨架化 + 弧长参数化 + 多项式拟合 + 纯追踪。

设计依据：《视觉识别算法详解.md》第 1 章。能力：
  - 灰度模式（黑/白线，向后兼容）与 HSV 颜色模式（识别特定颜色线，红色自动合并两段）
  - 多条线（连通域分离）、横向线（分类 cross）、拐角（分类 corner，肘部检测）
  - 输出纯追踪前视点 lookahead_x，供关卡做左/右移、左/右转、前进/后退决策

corner 判据（肘部检测，替代旧 PCA 直线度判据）：
  透视几何下 L 形拐角两臂长度极不对称（近处纵臂短、远处横臂长），PCA 直线度
  可高达 0.95+，旧判据判不出拐角。改为沿骨架点列滑窗检测弦向角突变：
  近端→远端方向变化 > ELBOW_ANGLE_THRESH 且两臂各 ≥ ELBOW_MIN_ARM_PX 判 corner。
  渐弯线（S 弯/弧线）窗口间方向渐变不触发，仍判 follow，天然区分急折。

所有输出量均在工作分辨率（work_width 宽）像素坐标系下，关卡阈值按此标定。
"""

import cv2
import numpy as np

from vision.detection import LineResult, LineSegment

# 肘部检测参数（corner 判据）
ELBOW_ANGLE_THRESH = 45.0   # 弦向角变化超过此值（度）判为肘部（拐角）
ELBOW_WINDOW = 8            # 滑窗半宽（点数），沿点列取 P[i±w] 计算弦向
ELBOW_MIN_NEAR_ARM_PX = 6.0   # 肘部近臂（肘点→近端）全弧长下限（像素）
ELBOW_MIN_FAR_ARM_PX = 15.0   # 肘部远臂（肘点→远端）全弧长下限（像素）
# 注：臂长按「肘点到段端点的全弧长」计，而非滑窗弦长——机器人贴近拐角时
# 近臂在图像中只有 ~10px，但仍是真拐角；远臂必须足够长以排除噪声毛刺。

# 骨架追踪段长度下限：拐角块状结构细化的毛刺残段（通常 <10 点）会被
# 丢弃，避免其被误判为 follow 并因优先级抢赢真正的 corner 段。
MIN_SEGMENT_LEN = 10

# follow 稳健拟合参数
FIT_MIN_RY_SPAN = 30.0      # ry 跨度小于此值（像素）时降为一次拟合（防 polyfit 病态）
FIT_MIN_POINTS = 8          # 点数小于此值时直接量测，不拟合
FIT_MAX_RMS = 4.0           # 二次拟合 RMS 残差超过此值（像素）改判 corner
LOOKAHEAD_CLAMP = 200.0     # lookahead_x 绝对值钳制上限（防止多项式外推爆炸）


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
            for seg in self._trace(skel):
                if len(seg) >= MIN_SEGMENT_LEN:
                    segments.append(seg)
        return segments

    @staticmethod
    def _trace(skel):
        """把 1px 骨架追踪成有序点列（像素坐标 (x, y)）。

        在分支点（度≥3）不停止，选择「最接近当前行进方向」的分支继续走，
        保证 L 形拐角的两臂留在同一条点列里（旧版在拐角分支点断开，
        L 形被拆成近臂/远臂两条独立线段，拐角永远检测不到）。
        """
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
        # 端点确定性排序：优先从「y 最大（离机器人最近）」的端点开始追踪，
        # 保证 L 形拐角的主路径被完整追踪（set 无序遍历会因 hash seed 不同
        # 产生非确定性的段划分）。
        endpoints = sorted((p for p in pts if deg[p] == 1), key=lambda p: (-p[1], p[0]))
        visited = set()
        segs = []

        def walk(start):
            seg = [start]
            visited.add(start)
            prev, cur = None, start
            while True:
                cand = [n for n in nbrs(cur) if n != prev and n not in visited]
                if not cand:
                    break
                if len(cand) == 1:
                    nxt = cand[0]
                else:
                    # 分支点：选与当前行进方向夹角最小的分支（保持直行趋势）
                    d = (cur[0] - prev[0], cur[1] - prev[1]) if prev else (0, 0)
                    def dir_score(q):
                        dq = (q[0] - cur[0], q[1] - cur[1])
                        dot = d[0] * dq[0] + d[1] * dq[1]
                        nq = (dq[0] ** 2 + dq[1] ** 2) ** 0.5 or 1.0
                        nd = (d[0] ** 2 + d[1] ** 2) ** 0.5 or 1.0
                        return -(dot / (nq * nd))   # cos 大 → 分数小 → 优先
                    nxt = min(cand, key=dir_score)
                visited.add(nxt)
                seg.append(nxt)
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
    @staticmethod
    def _find_elbow(pts):
        """肘部检测：沿点列滑窗找弦向角突变点（拐角判据）。

        pts: (N,2) 有序点列，首点为近端（y 最大，离机器人最近）
        返回 (turn_deg, elbow_idx) 或 None。
        turn_deg: 近臂→远臂的方向变化角（0~180°，取绝对值）

        臂长按「肘点到段端点的全弧长」计算（骨架点间距≈1px）：
        - 近臂 = 肘点→近端（pts[0]）的弧长，阈值低（机器人可贴近拐角）
        - 远臂 = 肘点→远端（pts[-1]）的弧长，阈值高（排除噪声毛刺）
        取「最近端第一个满足条件」的肘部（近拐角优先，防远拐角抢赢）。
        """
        n = len(pts)
        w = ELBOW_WINDOW
        if n < 2 * w + 2:
            return None
        # 累计弧长表：arc[i] = 点 0→i 的路径长度
        diffs = np.diff(pts, axis=0)
        step_len = np.hypot(diffs[:, 0], diffs[:, 1])
        arc = np.concatenate([[0.0], np.cumsum(step_len)])
        total_arc = arc[-1]
        for i in range(w, n - w):
            near_arm = arc[i]            # 肘点→近端弧长
            far_arm = total_arc - arc[i]  # 肘点→远端弧长
            if near_arm < ELBOW_MIN_NEAR_ARM_PX or far_arm < ELBOW_MIN_FAR_ARM_PX:
                continue
            v1 = pts[i] - pts[i - w]          # 近臂弦向（指向肘点）
            v2 = pts[i + w] - pts[i]          # 远臂弦向（离开肘点）
            n1 = np.hypot(*v1)
            n2 = np.hypot(*v2)
            if n1 < 1e-6 or n2 < 1e-6:
                continue
            cos_a = float(np.dot(v1, v2) / (n1 * n2))
            cos_a = max(-1.0, min(1.0, cos_a))
            turn_deg = float(np.degrees(np.arccos(cos_a)))
            if turn_deg > ELBOW_ANGLE_THRESH:
                return (turn_deg, i)   # 最近端优先，找到即返回
        return None

    def _analyze(self, seg, roi_w, roi_h):
        if len(seg) < 3:
            return None
        pts = np.array(seg, dtype=np.float64)          # (N,2) 图像坐标 x,y
        # 统一「近端在前」：首点 y 最大（图像底部，离机器人最近）
        if pts[0, 1] < pts[-1, 1]:
            pts = pts[::-1]
        x, y = pts[:, 0], pts[:, 1]
        cx, cy = x.mean(), y.mean()
        xc, yc = x - cx, y - cy
        var_x = float(np.dot(xc, xc) / len(x))
        var_y = float(np.dot(yc, yc) / len(x))
        cov_xy = float(np.dot(xc, yc) / len(x))
        # 特征值 → 直线度（保留作 cross 判据与 confidence）
        tr = var_x + var_y
        det = var_x * var_y - cov_xy * cov_xy
        disc = max(0.0, (tr / 2.0) ** 2 - det)
        lam1 = tr / 2.0 + np.sqrt(disc)
        lam2 = tr / 2.0 - np.sqrt(disc)
        straightness = lam1 / (lam1 + lam2 + 1e-9)

        # ---- 肘部检测（corner 主判据）----
        elbow = self._find_elbow(pts)

        # ---- 分类 ----
        if elbow is not None:
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

        # 机器人系：rx = x - 中心(右正)，ry = 底 - y(前正，0=机器人处)
        rx = x - roi_w / 2.0
        ry = roi_h - y

        if orientation == "follow":
            # ---- 稳健拟合 ----
            ry_span = float(ry.max() - ry.min())
            # lateral_offset 统一为「最近点实测 rx」
            nearest_rx = float(rx[np.argmax(ry)])
            seg_obj.lateral_offset = nearest_rx
            # 线远端 ry（终点判定用：线延伸到视野尽头≈roi_h，线末端接近则变小）
            seg_obj.far_ry = float(ry.max())

            if len(x) < FIT_MIN_POINTS or ry_span < FIT_MIN_RY_SPAN:
                # 点太少 / 跨度太小：直接量测，不拟合（防 polyfit 病态）
                seg_obj.heading_deg = float(np.degrees(np.arctan(
                    (rx[0] - rx[-1]) / max(1e-6, abs(ry[0] - ry[-1]))
                ))) if abs(ry[0] - ry[-1]) > 1e-6 else 0.0
                # lookahead：ry≈L 处实测点 rx（插值不外推）
                L = self.lookahead_ratio * roi_h
                seg_obj.lookahead_x = self._lookahead_by_measure(pts, roi_w, roi_h, L)
            else:
                # 二次拟合，但先校验 RMS；若超限改判 corner
                a, b, c = np.polyfit(ry, rx, 2)        # rx = a·ry² + b·ry + c
                fit_rx = a * ry ** 2 + b * ry + c
                rms = float(np.sqrt(np.mean((rx - fit_rx) ** 2)))
                if rms > FIT_MAX_RMS:
                    # 改判 corner（方向用远/近端 x 符号）
                    seg_obj.orientation = "corner"
                    near_pt = pts[np.argmax(y)]
                    far_pt = pts[np.argmin(y)]
                    seg_obj.curvature = 1.0 if far_pt[0] > near_pt[0] else -1.0
                    # corner 时 lateral_offset 已在上面设为最近点实测
                    return seg_obj
                L = self.lookahead_ratio * roi_h
                seg_obj.curvature = float(a)           # 曲率(右弯正)
                seg_obj.heading_deg = float(np.degrees(np.arctan(b)))  # 切线角
                # lookahead_x：拟合值，但做病态保护（外推过大时回退实测插值）
                lookahead_fit = float(a * L * L + b * L + c)
                # 若拟合值异常大（外推爆炸），回退为 ry≈L 处实测插值
                if abs(lookahead_fit) > LOOKAHEAD_CLAMP:
                    seg_obj.lookahead_x = self._lookahead_by_measure(pts, roi_w, roi_h, L)
                else:
                    seg_obj.lookahead_x = lookahead_fit
        elif orientation == "corner":
            elbow_turn_deg, elbow_idx = elbow
            elbow_pt = pts[elbow_idx]
            seg_obj.elbow_px = (float(elbow_pt[0]), float(elbow_pt[1]))
            # 拐角点前向距离：roi 底部=机器人处(ry≈0)，顶部=视野远端(ry≈roi_h)
            seg_obj.elbow_ry = float(roi_h - elbow_pt[1])
            # 观测到的拐角弯折角（0~180°，恒正），供闭环决定转弯程度
            seg_obj.corner_deg = float(elbow_turn_deg)
            # 转向方向：远臂末端相对肘点的横向符号（右=+1，左=-1）
            far_end = pts[-1]
            seg_obj.curvature = 1.0 if far_end[0] > elbow_pt[0] else -1.0
            # 近臂弦向角（含图像底部那段，反映机器人当前与线的夹角）
            near_arm = pts[max(0, elbow_idx - ELBOW_WINDOW)] - pts[0]
            if np.hypot(*near_arm) > 1e-6:
                seg_obj.heading_deg = float(np.degrees(np.arctan2(near_arm[0], -near_arm[1])))
            # 最近点横向偏移（直接量测，不外推）
            seg_obj.lateral_offset = float(rx[np.argmax(ry)])
            # 线远端 ry（拐角时远臂延伸到视野远处，仅作参考）
            seg_obj.far_ry = float(ry.max())
            # corner 时也设 lookahead_x：用实测插值（不外推，防爆炸）
            L = self.lookahead_ratio * roi_h
            corner_la = self._lookahead_by_measure(pts, roi_w, roi_h, L)
            seg_obj.lookahead_x = max(-LOOKAHEAD_CLAMP, min(LOOKAHEAD_CLAMP, corner_la))
        else:  # cross
            seg_obj.heading_deg = 90.0
        return seg_obj

    @staticmethod
    def _lookahead_by_measure(pts, roi_w, roi_h, L):
        """在实测点列上取 ry≈L 处的 rx（线性插值，不外推）。

        若 L 超出点列 ry 范围，取最近端点的 rx（保守回退）。
        """
        rx = pts[:, 0] - roi_w / 2.0
        ry = roi_h - pts[:, 1]
        order = np.argsort(ry)
        ry_s, rx_s = ry[order], rx[order]
        if L <= ry_s[0]:
            return float(rx_s[0])
        if L >= ry_s[-1]:
            return float(rx_s[-1])
        return float(np.interp(L, ry_s, rx_s))

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

        # 主线优先级：follow > corner > cross，同级取离机器人最近（y 最大）
        def priority(s):
            return (s.orientation == "follow", s.orientation == "corner", nearness(s))

        primary = max(analyzed, key=priority)
        others = [s for s in analyzed if s is not primary]
        confidence = float(primary.straightness)
        return LineResult(exists=True, primary=primary, others=others, confidence=confidence)
