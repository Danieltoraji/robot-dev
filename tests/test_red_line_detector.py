# -*- coding: utf-8 -*-
"""red_line_detector 单元测试：合成帧验证连通域分割与下沿点提取

运行：python -m tests.test_red_line_detector
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from vision.red_line_detector import RedLineDetector

FRAME_W, FRAME_H = 2592, 1944
BG_GRAY = 90


def make_band(bottom_left, bottom_right, thickness, hue=3, sat=200, val=200):
    """画一条红色横带（下沿 = bottom_left -> bottom_right 直线）"""
    frame = np.full((FRAME_H, FRAME_W, 3), BG_GRAY, np.uint8)
    bgr = cv2.cvtColor(np.full((1, 1, 3), (hue, sat, val), np.uint8),
                       cv2.COLOR_HSV2BGR)[0, 0].tolist()
    pts = np.array([bottom_left,
                    bottom_right,
                    (bottom_right[0], bottom_right[1] - thickness),
                    (bottom_left[0], bottom_left[1] - thickness)],
                   dtype=np.int32)
    cv2.fillPoly(frame, [pts], bgr)
    return frame


def fit_angle_deg(pts):
    """下沿点的最小二乘方向角（度，相对 x 轴）"""
    c = pts.mean(axis=0)
    u, s, vt = np.linalg.svd(pts - c, full_matrices=False)
    v = vt[0]
    if v[0] < 0:
        v = -v
    return float(np.degrees(np.arctan2(v[1], v[0])))


def test_horizontal_band():
    frame = make_band((500, 1500), (2000, 1500), 60)
    det = RedLineDetector().detect(frame)
    assert det.exists and len(det.components) == 1
    comp = det.components[0]
    assert comp.area_px > 500
    assert abs(np.median(comp.bottom_pts[:, 1]) - 1500) <= 1.5, \
        f"下沿 y {np.median(comp.bottom_pts[:, 1])} 偏离 1500"
    assert fit_angle_deg(comp.bottom_pts) < 0.5, "水平带方向角应 ≈0"
    print("  水平横带：检出 1 分量，下沿/方向正确 ✓")


def test_dual_range_red():
    frame = np.maximum(
        make_band((300, 1000), (1200, 1000), 50, hue=5),
        make_band((1400, 1000), (2300, 1000), 50, hue=170))
    det = RedLineDetector().detect(frame)
    assert det.exists and len(det.components) == 2, \
        f"HSV 双区间应各检出一段，实际 {len(det.components)}"
    print("  HSV 双区间（h=5 与 h=170）：两段均检出 ✓")


def test_min_area_filter():
    frame = make_band((500, 1500), (2000, 1500), 60)
    cv2.rectangle(frame, (100, 100), (112, 108), (50, 50, 220), -1)  # 96px 噪点
    det = RedLineDetector().detect(frame)
    assert len(det.components) == 1, "小面积噪点应被过滤"
    print("  面积过滤：96px 噪点被滤除 ✓")


def test_tilted_band():
    angle_true = fit_angle_deg(np.array([[300.0, 1200.0], [2200.0, 1500.0]]))
    frame = make_band((300, 1200), (2200, 1500), 60)
    det = RedLineDetector().detect(frame)
    assert det.exists
    angle = fit_angle_deg(det.components[0].bottom_pts)
    assert abs(angle - angle_true) < 0.5, f"斜带方向角 {angle:.2f} vs {angle_true:.2f}"
    print(f"  斜带：方向角 {angle:.2f}° 与真值 {angle_true:.2f}° 一致 ✓")


def test_morphology_closes_holes():
    frame = make_band((500, 1500), (2000, 1500), 60)
    # 带内抠几个洞（模拟反光/纹理断裂）
    for cx in (800, 1200, 1600):
        cv2.rectangle(frame, (cx, 1465), (cx + 20, 1485), (BG_GRAY,) * 3, -1)
    det = RedLineDetector().detect(frame)
    assert det.exists and len(det.components) == 1, "闭运算应连接带内空洞"
    print("  闭运算：带内小洞被填补，保持单连通域 ✓")


def test_hsv_robustness():
    """HSV 扰动鲁棒性（方案 §4 验收线：单帧检出率 ≥95%）

    扰动集：V ±30%、色相向橙(H<12)/粉(H>160)边界偏移、阴影(V×0.35)。
    用合成相机帧（真实内参/畸变渲染，30cm 处横杆）。
    """
    from sim.stairs_hurdle_sim import make_frame, bar_line_quads

    def hsv_shift(frame, dh=0, vscale=1.0):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = np.clip(hsv[..., 0] + dh, 0, 179)
        hsv[..., 2] = np.clip(hsv[..., 2] * vscale, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    base = make_frame(bar_line_quads(30.0))
    perturbations = []
    for vs in (0.7, 1.0, 1.3):
        for dh in (0, 5, 8, -5, -8):
            perturbations.append((dh, vs))
    perturbations.append((0, 0.35))  # 阴影
    det = RedLineDetector()
    hits = 0
    misses = []
    for dh, vs in perturbations:
        d = det.detect(hsv_shift(base, dh, vs))
        if d.exists:
            hits += 1
        else:
            misses.append((dh, vs))
    rate = hits / len(perturbations)
    assert rate >= 0.95, f"检出率 {rate:.0%} <95%，未检出组合 {misses}"
    print(f"  HSV 扰动 {len(perturbations)} 例检出率 {rate:.0%} ✓")


if __name__ == "__main__":
    test_horizontal_band()
    test_dual_range_red()
    test_min_area_filter()
    test_tilted_band()
    test_morphology_closes_holes()
    test_hsv_robustness()
    print("全部 red_line_detector 测试通过 ✓")
