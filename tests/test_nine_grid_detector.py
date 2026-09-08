# -*- coding: utf-8 -*-
"""nine_grid_detector 单元测试：七色合成图端到端

对每种颜色构造满足阈值（含绿/蓝动态规则）的 HSV 合成面板，走
HSV->BGR 转换模拟真实相机输入，断言检测器：正确分类七色、中心定位、
黑色数字 mask 提取、color_ratio、跨色去重不误报。

运行：python tests/test_nine_grid_detector.py
"""

import os
import sys

# 允许从任意目录直接运行本文件（脚本目录在 sys.path[0]，仓库根不在）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import cv2

from vision.nine_grid_detector import (
    NineGridDetector, build_color_mask, extract_digit_mask,
    COLOR_TO_ID, DigitArbiter,
)

FRAME_W, FRAME_H = 2592, 1944
SQUARE = 700   # 面板边长（原生像素，≈1.2m 处的 33cm 面板）

# 各颜色满足阈值/动态规则的 HSV 取值（合成真值）
COLOR_HSV = {
    "red": (3, 200, 200),
    "orange": (12, 200, 200),
    "yellow": (27, 180, 200),
    "green": (65, 200, 100),    # H_deg=130, S/V=2.0
    "blue": (107, 230, 130),    # H_deg=214, S=90%, V=51%
    "purple": (124, 150, 150),
    "pink": (165, 120, 150),
}


def make_frame(color_name, square_xy=(946, 622), digit=True):
    """灰底 + 色块（+黑色数字块）的合成帧"""
    hsv = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    hsv[:, :, 1] = 0
    hsv[:, :, 2] = 90       # 灰底（V=90，不触发任何彩色阈值）
    x, y = square_xy
    h, s, v = COLOR_HSV[color_name]
    hsv[y:y + SQUARE, x:x + SQUARE] = (h, s, v)
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    if digit:
        # 黑色"数字"块：面板中央 140x180
        cx, cy = x + SQUARE // 2, y + SQUARE // 2
        cv2.rectangle(frame, (cx - 70, cy - 90), (cx + 70, cy + 90),
                      (0, 0, 0), -1)
    return frame


def approx(v, ref, tol):
    assert abs(v - ref) <= tol, f"{v:.2f} != {ref:.2f}±{tol}"


def test_seven_colors():
    det = NineGridDetector()
    for color, digit_id in COLOR_TO_ID.items():
        frame = make_frame(color, digit=False)
        obs = det.detect_panels(frame)
        assert len(obs) == 1, f"{color}: 应检出1块，实际{len(obs)} {obs}"
        o = obs[0]
        assert o.color == color, f"{color}: 误判为 {o.color}"
        assert o.digit == digit_id
        cx, cy = o.center_px
        approx(cx, 946 + SQUARE / 2, 8)
        approx(cy, 622 + SQUARE / 2, 8)
    print("  七色分类与中心定位 ✓")


def test_black_digit_mask():
    """彩色面板上的黑色数字应被相对阈值提取"""
    for color in ("blue", "green"):  # 特殊判定色重点测
        frame = make_frame(color)
        x, y = 946, 622
        roi = frame[y:y + SQUARE, x:x + SQUARE]
        mask = extract_digit_mask(roi)
        # 数字块 140x180=25200，允许形态学 ±30% 误差
        area = cv2.countNonZero(mask)
        assert 0.7 * 25200 < area < 1.5 * 25200, \
            f"{color}: 数字mask面积{area}异常"
        # mask 应集中在中央
        ys, xs = np.nonzero(mask)
        approx(float(xs.mean()), SQUARE / 2, 60)
        approx(float(ys.mean()), SQUARE / 2, 60)
    print("  黑色数字 mask 提取（含绿/蓝特殊色） ✓")


def test_color_ratio():
    det = NineGridDetector()
    frame = make_frame("red", digit=False)
    ratio = det.color_ratio(frame, "red")
    assert 0.05 < ratio < 0.20, f"红色占比 {ratio:.3f} 异常"
    assert det.color_ratio(frame, "blue") < 0.005
    print(f"  color_ratio（红 {ratio:.3f}，蓝 <0.005） ✓")


def test_arbitration_interface():
    """仲裁接口：模型可用时跑通、字段齐全（合成图形预测值不断言）"""
    det = NineGridDetector()
    frame = make_frame("blue")
    obs = det.detect_panels(frame, arbitrate=True)
    assert len(obs) == 1
    o = obs[0]
    if DigitArbiter().available():
        assert o.model_digit is not None, "模型可用时应给出仲裁结果"
        assert 1 <= o.model_digit <= 7 or o.model_digit == 0
        print(f"  仲裁接口：svm={o.model_digit}({o.model_conf:.2f}) "
              f"conflict={o.arb_conflict} ✓")
    else:
        assert o.model_digit is None
        print("  仲裁接口：模型不可用已降级 ✓")


def test_no_false_on_empty():
    det = NineGridDetector()
    gray = np.full((FRAME_H, FRAME_W, 3), 90, np.uint8)
    assert det.detect_panels(gray) == []
    assert det.detect_panels(None) == []
    print("  空场景零误报 ✓")


if __name__ == "__main__":
    test_seven_colors()
    test_black_digit_mask()
    test_color_ratio()
    test_arbitration_interface()
    test_no_false_on_empty()
    print("全部 nine_grid_detector 测试通过 ✓")
