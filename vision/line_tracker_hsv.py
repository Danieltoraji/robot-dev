# -*- coding: utf-8 -*-
"""方案一：HSV 自适应阈值 + ROI 动态搜索 + 单应映射 的巡线检测器。

继承 vision/line_detector.py 的 LineDetector，完整保留其「ROI 裁剪 → 连通域
分割 → 骨架化 → 弧长参数化 → 多项式拟合 → follow/cross/corner 分类 → 纯追踪」
流程（直角弯过弯闭环依赖这些分类），仅替换两处：

  1. _make_mask：固定红色色相范围 + S/V 通道 OTSU 自适应阈值 + 形态学滤波，
     替代原固定 S/V 的 cv2.inRange，使分割随局部光照自适应；
  2. detect：ROI 找不到线时向上扩大 ROI 动态搜索；并对最近点 / lookahead 点
     做地面单应映射，输出横向偏移与距离（cm）。

单应：GroundHomography.from_pose 解析自举（相机位姿假设 + 俯仰脉宽），运行期
把 640x480 工作坐标按比例放大到原生 2592x1944 再喂 pixels_to_ground。横向偏移
是相对相机光心投影的垂直视线方向距离，与机器人在场地中的绝对位置无关，只随
相机朝向变化；循迹时朝向近似固定，可作为辅助横向反馈（from_pose 精度 ±3cm 级）。
"""

import cv2
import numpy as np

from vision.detection import LineResult
from vision.line_detector import LineDetector
from core.ground_homography import GroundHomography
from core.camera_config import CAMERA_WIDTH, CAMERA_HEIGHT, HEAD_CENTER


class LineTrackerHSV(LineDetector):
    def __init__(self, hsv_ranges=None, line_color="light", min_area=80,
                 lookahead_ratio=0.5, straightness_thresh=0.85, work_width=640,
                 roi_ratio=0.5, roi_expand_ratios=(0.6, 0.7),
                 pitch_pulse=1050, cam_height_cm=39.0, head_pulse=HEAD_CENTER,
                 use_adaptive=True, morph_open_iters=1, morph_close_iters=2):
        super().__init__(
            line_color=line_color,
            hsv_ranges=hsv_ranges,
            min_area=min_area,
            lookahead_ratio=lookahead_ratio,
            straightness_thresh=straightness_thresh,
            work_width=work_width,
        )
        self.roi_ratio = roi_ratio
        self.roi_expand_ratios = tuple(roi_expand_ratios)
        self.use_adaptive = use_adaptive
        self.morph_open_iters = morph_open_iters
        self.morph_close_iters = morph_close_iters
        self.pitch_pulse = pitch_pulse

        # 地面单应（解析自举）：相机朝 +y、中位头部；仅用于相对相机的 cm 输出
        self.hg = None
        try:
            self.hg = GroundHomography.from_pose(
                (0.0, 0.0), cam_height_cm, pitch_pulse,
                bearing_deg=0.0, head_pulse=head_pulse,
            )
        except Exception:
            self.hg = None  # 解析失败时退化为纯像素模式（cm 字段保持 None）

    # =================================================================
    # Step 0：HSV 自适应阈值二值掩膜（override）
    # =================================================================
    def _make_mask(self, roi):
        ranges = self.hsv_ranges or ([self.hsv_range] if self.hsv_range else None)
        if ranges is None:
            return super()._make_mask(roi)

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # 固定红色色相范围（颜色特异性，红分两段由 hsv_ranges 传入）
        hue_mask = None
        for lo, hi in ranges:
            m = cv2.inRange(h, lo[0], hi[0])
            hue_mask = m if hue_mask is None else cv2.bitwise_or(hue_mask, m)

        if self.use_adaptive:
            # S / V 通道 OTSU 自适应阈值（局部光照自适应）。
            # 直接用 threshold 返回的二值图 dst（语义：src > 阈值 归前景），
            # 避免手写 inRange 的「>=」多一个等号把背景值也纳入前景。
            #   S 通道：高饱和 = 彩色线（始终取 src > 阈值）
            #   V 通道：线相对背景的明暗方向由 line_color 决定——
            #     "light"（线比背景亮）取 src > 阈值；"dark"（线比背景暗，如白底红线）
            #     取 src <= 阈值（BINARY_INV），否则会把更亮的白底误选为前景。
            _, s_bin = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            v_type = (cv2.THRESH_BINARY_INV if self.line_color == "dark"
                      else cv2.THRESH_BINARY)
            _, v_bin = cv2.threshold(v, 0, 255, v_type + cv2.THRESH_OTSU)
            mask = cv2.bitwise_and(hue_mask, s_bin)
            mask = cv2.bitwise_and(mask, v_bin)
        else:
            # 回退：固定 S/V 下界（取第一段范围的下界）；S>=下界 已能排除白底
            # （白底 S=0），V 通道无需区分明暗方向，保留原固定下界即可。
            lo = ranges[0][0]
            mask = cv2.bitwise_and(hue_mask, cv2.inRange(s, lo[1], 255))
            mask = cv2.bitwise_and(mask, cv2.inRange(v, lo[2], 255))

        # 形态学滤波：开运算去噪点，闭运算补线宽缺口
        if self.morph_open_iters or self.morph_close_iters:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            if self.morph_open_iters:
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,
                                        iterations=self.morph_open_iters)
            if self.morph_close_iters:
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,
                                        iterations=self.morph_close_iters)
        return mask

    # =================================================================
    # detect：ROI 动态搜索 + 单应映射（override）
    # =================================================================
    def detect(self, frame):
        if frame is None or frame.size == 0:
            return LineResult(exists=False)

        ratios = (self.roi_ratio,) + self.roi_expand_ratios
        for ratio in ratios:
            result = self._detect_one_roi(frame, ratio)
            if result.exists:
                return result
        return LineResult(exists=False)

    def _detect_one_roi(self, frame, roi_ratio):
        h, w = frame.shape[:2]
        y0 = int(h * (1.0 - roi_ratio))
        roi = frame[y0:, :]
        scale = 1.0
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

        def priority(s):
            return (s.orientation == "follow", s.orientation == "corner", nearness(s))

        primary = max(analyzed, key=priority)
        others = [s for s in analyzed if s is not primary]
        confidence = float(primary.straightness)

        self._fill_cm_fields(primary, roi_w, roi_h, y0, scale, h, w)

        return LineResult(exists=True, primary=primary, others=others, confidence=confidence)

    # =================================================================
    # 单应映射
    # =================================================================
    def _fill_cm_fields(self, seg, roi_w, roi_h, roi_y0, scale, frame_h, frame_w):
        if self.hg is None:
            return
        pts = np.array(seg.points, dtype=np.float64)
        if len(pts) == 0:
            return

        # 最近点（lateral_offset 的参考点）：points 中 y 最大
        nearest = pts[np.argmax(pts[:, 1])]
        npix = self._work_to_native(nearest[0], nearest[1], roi_w, roi_h,
                                    roi_y0, scale, frame_h, frame_w)
        fwd, lat = self._px_to_robot_frame(npix)
        if fwd is not None:
            seg.nearest_forward_cm = float(fwd)
            seg.lateral_offset_cm = float(lat)

        # lookahead 点：rx=lookahead_x, ry=L（cross 无前视点，跳过）
        if seg.orientation != "cross":
            L = self.lookahead_ratio * roi_h
            look_x = seg.lookahead_x + roi_w / 2.0
            look_y = roi_h - L
            lpix = self._work_to_native(look_x, look_y, roi_w, roi_h,
                                        roi_y0, scale, frame_h, frame_w)
            fwd, lat = self._px_to_robot_frame(lpix)
            if fwd is not None:
                seg.lookahead_forward_cm = float(fwd)
                seg.lookahead_cm = float(lat)

    @staticmethod
    def _work_to_native(work_x, work_y, roi_w, roi_h, roi_y0, scale,
                        frame_h, frame_w):
        # 工作坐标 → ROI 坐标（去 scale）
        roi_x = work_x / scale
        roi_y = work_y / scale
        # ROI 坐标 → 整帧坐标
        frame_x = roi_x
        frame_y = roi_y0 + roi_y
        # 整帧（采集分辨率）→ 原生（CAMERA_WIDTH x CAMERA_HEIGHT）
        native_x = frame_x * (CAMERA_WIDTH / frame_w)
        native_y = frame_y * (CAMERA_HEIGHT / frame_h)
        return native_x, native_y

    def _px_to_robot_frame(self, native_px):
        """原生像素 → 相对相机的 (forward_cm, lateral_cm)；失败返回 (None, None)"""
        try:
            ground = self.hg.pixels_to_ground([native_px])[0]
            forward, lateral, _ = self.hg.target_in_robot_frame(ground)
            return float(forward), float(lateral)
        except Exception:
            return None, None
