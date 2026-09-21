#!/usr/bin/python3
# coding=utf8
"""无硬件依赖的红色球门线估计器。

实现与 Functions/goal_line_web.py 已验证的红线拟合逻辑一致，
但不打开摄像头、不初始化机器人，只接收当前帧和球门柱检测框。
"""

import cv2
import numpy as np

try:
    from Functions.goal_line_judge import _estimate_goal_line
except ImportError:
    from goal_line_judge import _estimate_goal_line


class RedGoalLineEstimator:
    """在两个球门柱附近 ROI 内拟合红色球门线，并做时域平滑。"""

    def __init__(
        self,
        band_px=70,
        min_points=25,
        min_span_px=35,
        smoothing_alpha=0.35,
        hold_frames=8,
    ):
        self.band_px = float(band_px)
        self.min_points = max(5, int(min_points))
        self.min_span_px = max(10, int(min_span_px))
        self.smoothing_alpha = float(smoothing_alpha)
        self.hold_frames = max(0, int(hold_frames))
        self.reset()

    def reset(self):
        self.smoothed_line = None
        self.missed_frames = 0

    @staticmethod
    def _line_from_points(x1, y1, x2, y2, image_width):
        dx = float(x2) - float(x1)
        dy = float(y2) - float(y1)
        norm = float(np.hypot(dx, dy))
        if norm < 1.0 or abs(dx) < 1e-9:
            return None
        left_y = float(y1) + (0.0 - float(x1)) * dy / dx
        right_x = float(image_width - 1)
        right_y = float(y1) + (right_x - float(x1)) * dy / dx
        return {
            "p1": (round(float(x1)), round(float(y1))),
            "p2": (round(float(x2)), round(float(y2))),
            "extended_p1": (0, round(left_y)),
            "extended_p2": (image_width - 1, round(right_y)),
            "x1": float(x1),
            "y1": float(y1),
            "dx": dx,
            "dy": dy,
            "norm": norm,
        }

    def _smooth(self, current, image_width):
        previous = self.smoothed_line
        if current is None:
            return previous
        if previous is None:
            return current

        current_slope = current["dy"] / current["dx"]
        current_intercept = current["y1"] - current_slope * current["x1"]
        previous_slope = previous["dy"] / previous["dx"]
        previous_intercept = previous["y1"] - previous_slope * previous["x1"]

        # 球门和机器人已经基本静止时，异常大角度跳变不可信。
        if abs(current_slope - previous_slope) > 0.45:
            return previous

        alpha = self.smoothing_alpha
        slope = alpha * current_slope + (1.0 - alpha) * previous_slope
        intercept = alpha * current_intercept + (1.0 - alpha) * previous_intercept
        return self._line_from_points(
            0.0,
            intercept,
            float(image_width - 1),
            slope * float(image_width - 1) + intercept,
            image_width,
        )

    def estimate(self, frame, goalposts):
        """返回 ``(line, pixel_count)``；line 兼容 goal_line_judge 的格式。"""
        if frame is None or getattr(frame, "size", 0) == 0:
            return self.smoothed_line, 0
        height, width = frame.shape[:2]
        expected = _estimate_goal_line(goalposts, width)
        if expected is None:
            return self.smoothed_line, 0

        post_x = sorted(
            float(post[0]) for post in (goalposts or []) if len(post) >= 4
        )
        if len(post_x) < 2:
            return self.smoothed_line, 0

        x_min = max(0, int(post_x[0] - 55))
        x_max = min(width - 1, int(post_x[-1] + 55))
        if x_max - x_min < self.min_span_px:
            return self.smoothed_line, 0

        y_expected_min = expected["y1"] + (x_min - expected["x1"]) * expected["dy"] / expected["dx"]
        y_expected_max = expected["y1"] + (x_max - expected["x1"]) * expected["dy"] / expected["dx"]
        y_min = max(0, int(min(y_expected_min, y_expected_max) - self.band_px))
        y_max = min(height - 1, int(max(y_expected_min, y_expected_max) + self.band_px))
        if y_max <= y_min:
            return self.smoothed_line, 0

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        low1 = np.array([0, 65, 35], dtype=np.uint8)
        high1 = np.array([12, 255, 255], dtype=np.uint8)
        low2 = np.array([168, 65, 35], dtype=np.uint8)
        high2 = np.array([180, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, low1, high1)
        mask |= cv2.inRange(hsv, low2, high2)
        roi = mask[y_min:y_max + 1, x_min:x_max + 1]
        kernel = np.ones((3, 3), np.uint8)
        roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel, iterations=1)
        roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel, iterations=1)

        local_y, local_x = np.where(roi > 0)
        pixel_count = int(len(local_x))
        if pixel_count < self.min_points:
            self.missed_frames += 1
            return self.smoothed_line, pixel_count

        xs = local_x.astype(np.float64) + x_min
        ys = local_y.astype(np.float64) + y_min
        expected_y = expected["y1"] + (xs - expected["x1"]) * expected["dy"] / expected["dx"]
        near_expected = np.abs(ys - expected_y) <= self.band_px
        xs = xs[near_expected]
        ys = ys[near_expected]
        pixel_count = int(len(xs))
        if pixel_count < self.min_points or float(xs.max() - xs.min()) < self.min_span_px:
            self.missed_frames += 1
            return self.smoothed_line, pixel_count

        keep = np.ones(len(xs), dtype=bool)
        for _ in range(3):
            if int(keep.sum()) < self.min_points:
                self.missed_frames += 1
                return self.smoothed_line, int(keep.sum())
            slope, intercept = np.polyfit(xs[keep], ys[keep], 1)
            residual = np.abs(ys - (slope * xs + intercept))
            threshold = max(5.0, float(np.percentile(residual[keep], 70)) * 1.8)
            keep = residual <= threshold

        if int(keep.sum()) < self.min_points:
            self.missed_frames += 1
            return self.smoothed_line, int(keep.sum())

        slope, intercept = np.polyfit(xs[keep], ys[keep], 1)
        current = self._line_from_points(
            float(xs[keep].min()),
            slope * float(xs[keep].min()) + intercept,
            float(xs[keep].max()),
            slope * float(xs[keep].max()) + intercept,
            width,
        )
        if current is None:
            self.missed_frames += 1
            return self.smoothed_line, 0

        self.smoothed_line = self._smooth(current, width)
        self.missed_frames = 0
        return self.smoothed_line, int(keep.sum())
