# -*- coding: utf-8 -*-
"""line_tracker_hsv 单元测试：合成红色线端到端

对 follow（竖直红直线）/ corner（L 形红折线）/ cross（水平红横线）三种几何
构造合成帧，走 LineTrackerHSV 全链路（HSV 自适应阈值 + 形态学 + 骨架化 +
拟合分类 + 地面单应 cm 输出），断言：
  - 分类正确（follow / corner / cross）
  - 像素量符号与几何真值一致（lateral_offset / lookahead_x 右正左负）
  - 单应 cm 字段被填充，且横向符号与像素量一致、纵向为正
  - ROI 动态搜索：线只出现在底部 ROI 之外时靠扩大搜索找回
  - 空场景 / None 零误报

运行：python tests/test_line_tracker_hsv.py
"""

import os
import sys

# 允许从任意目录直接运行本文件（脚本目录在 sys.path[0]，仓库根不在）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2

from vision.line_tracker_hsv import LineTrackerHSV


FRAME_W, FRAME_H = 640, 480
ROI_RATIO = 0.5
Y0 = int(FRAME_H * (1.0 - ROI_RATIO))          # 240，默认 ROI 底边

# 红色路径 HSV 两段（与 levels/line_seeker_tracking.py 一致）
RED_LOW = (0, 0, 50)
RED_HIGH = (17, 255, 255)
RED_LOW_2 = (149, 0, 50)
RED_HIGH_2 = (180, 255, 255)

LINE_BGR = (0, 0, 200)   # 纯红 BGR：h≈0 / s≈255 / v≈200
BG_GRAY = 90             # 灰底 V（S=0），自适应阈值下不触发红线
BG_DARK = 0              # 黑底（固定阈值回退路径用，V<50 才能与红线分离）


def make_frame(bg=BG_GRAY):
    """灰底帧（S=0，hue 虽在红段内但 S/V 被自适应阈值滤掉）"""
    return np.full((FRAME_H, FRAME_W, 3), bg, dtype=np.uint8)


def make_detector(**kw):
    """与 line_follow_debug.py 相同配置的默认检测器"""
    kw.setdefault("hsv_ranges", [(RED_LOW, RED_HIGH), (RED_LOW_2, RED_HIGH_2)])
    kw.setdefault("line_color", "light")
    kw.setdefault("min_area", 80)
    kw.setdefault("lookahead_ratio", 0.5)
    kw.setdefault("straightness_thresh", 0.85)
    kw.setdefault("work_width", FRAME_W)
    kw.setdefault("roi_ratio", ROI_RATIO)
    kw.setdefault("pitch_pulse", 1050)
    kw.setdefault("cam_height_cm", 39.0)
    return LineTrackerHSV(**kw)


def approx(v, ref, tol):
    assert abs(v - ref) <= tol, f"{v:.2f} != {ref:.2f}±{tol}"


def draw_vertical(frame, x, y_top, y_bot, half_w=3, color=LINE_BGR):
    cv2.rectangle(frame, (x - half_w, y_top), (x + half_w, y_bot), color, -1)


def draw_horizontal(frame, y, x_left, x_right, half_w=3, color=LINE_BGR):
    cv2.rectangle(frame, (x_left, y - half_w), (x_right, y + half_w), color, -1)


def test_homography_enabled():
    """解析自举单应应成功（无 hiwonder 依赖），cm 字段才有意义"""
    det = make_detector()
    assert det.hg is not None, "GroundHomography.from_pose 解析自举失败"
    print("  单应解析自举启用（pitch=1050, cam_z=39cm） ✓")


def test_follow_right():
    """竖直红线偏右 → follow，像素/厘米横向均为正（右正左负）"""
    frame = make_frame()
    draw_vertical(frame, FRAME_W // 2 + 40, Y0, FRAME_H - 1)
    p = make_detector().detect(frame).primary
    assert p is not None and p.orientation == "follow", f"orientation={p.orientation}"
    assert p.lateral_offset > 0, f"lateral_offset={p.lateral_offset} 应>0"
    assert p.lookahead_x > 0, f"lookahead_x={p.lookahead_x} 应>0"
    assert p.lateral_offset_cm is not None and p.lateral_offset_cm > 0, \
        f"lateral_offset_cm={p.lateral_offset_cm} 应>0"
    assert p.lookahead_cm is not None and p.lookahead_cm > 0, \
        f"lookahead_cm={p.lookahead_cm} 应>0"
    assert p.nearest_forward_cm > 0 and p.lookahead_forward_cm > 0, "纵向应为正"
    print(f"  follow(右)：lateral={p.lateral_offset:.1f}px→{p.lateral_offset_cm:.1f}cm "
          f"lookahead={p.lookahead_x:.1f}px→{p.lookahead_cm:.1f}cm ✓")


def test_follow_left():
    """竖直红线偏左 → 横向符号为负（与像素一致）"""
    frame = make_frame()
    draw_vertical(frame, FRAME_W // 2 - 40, Y0, FRAME_H - 1)
    p = make_detector().detect(frame).primary
    assert p is not None and p.orientation == "follow", f"orientation={p.orientation}"
    assert p.lateral_offset < 0, f"lateral_offset={p.lateral_offset} 应<0"
    assert p.lookahead_x < 0, f"lookahead_x={p.lookahead_x} 应<0"
    assert p.lateral_offset_cm is not None and p.lateral_offset_cm < 0, \
        f"lateral_offset_cm={p.lateral_offset_cm} 应<0"
    assert p.lookahead_cm is not None and p.lookahead_cm < 0, \
        f"lookahead_cm={p.lookahead_cm} 应<0"
    print(f"  follow(左)：lateral={p.lateral_offset:.1f}px→{p.lateral_offset_cm:.1f}cm "
          f"lookahead={p.lookahead_x:.1f}px→{p.lookahead_cm:.1f}cm ✓")


def test_cross():
    """水平红线 → cross（heading=±90），lookahead_cm 不填充"""
    frame = make_frame()
    draw_horizontal(frame, Y0 + 80, FRAME_W // 4, 3 * FRAME_W // 4)
    p = make_detector().detect(frame).primary
    assert p is not None and p.orientation == "cross", f"orientation={p.orientation}"
    assert abs(abs(p.heading_deg) - 90.0) < 1e-6, f"heading={p.heading_deg}"
    assert p.lookahead_cm is None, "cross 不应填 lookahead_cm"
    assert p.lateral_offset_cm is not None, "cross 应填最近点横向 cm"
    print(f"  cross：heading={p.heading_deg:.0f}° "
          f"lateral_cm={p.lateral_offset_cm:.1f} ✓")


def test_corner_right():
    """L 形右折 → corner，curvature>0，肘点/弯折角被填"""
    frame = make_frame()
    x0 = FRAME_W // 2 + 40            # 竖直近臂 x
    draw_vertical(frame, x0, Y0 + 50, FRAME_H - 1)          # 近臂（底部）
    draw_horizontal(frame, Y0 + 50, x0, FRAME_W - 40)       # 远臂（向右）
    p = make_detector().detect(frame).primary
    assert p is not None and p.orientation == "corner", f"orientation={p.orientation}"
    assert p.curvature > 0, f"右折 curvature 应>0，实际 {p.curvature}"
    assert p.elbow_px is not None, "corner 应给出肘点"
    assert p.corner_deg > 0, f"corner_deg={p.corner_deg} 应>0"
    assert p.lateral_offset_cm is not None, "corner 应填最近点横向 cm"
    print(f"  corner(右折)：curvature={p.curvature:+.1f} "
          f"corner_deg={p.corner_deg:.0f}° elbow_ry={p.elbow_ry:.0f}px ✓")


def test_roi_dynamic_search():
    """线只出现在默认 ROI(底部50%)之外时，靠扩大 ROI(→70%)找回"""
    frame = make_frame()
    # 线在 y∈[0.3H, 0.38H]：不在默认 ROI(y≥0.5H) 内，需扩到 0.7(y≥0.3H)
    draw_vertical(frame, FRAME_W // 2, int(0.3 * FRAME_H), int(0.38 * FRAME_H))
    r = make_detector().detect(frame)
    assert r.exists, "扩大 ROI 后应找回线"
    assert r.primary.orientation == "follow", f"orientation={r.primary.orientation}"
    print("  ROI 动态搜索：线在默认 ROI 外，扩大后找回 ✓")


def test_fixed_threshold_fallback():
    """use_adaptive=False 回退固定 S/V 阈值（黑底，V 差足够分离）"""
    frame = make_frame(bg=BG_DARK)
    draw_vertical(frame, FRAME_W // 2, Y0, FRAME_H - 1)
    p = make_detector(use_adaptive=False).detect(frame).primary
    assert p is not None and p.orientation == "follow", f"orientation={p.orientation}"
    print("  固定阈值回退：黑底红线检出 follow ✓")


def test_no_false_on_empty():
    """纯灰底 / None 零误报"""
    det = make_detector()
    assert not det.detect(make_frame()).exists, "灰底不应检出线"
    assert not det.detect(None).exists, "None 不应检出线"
    print("  空场景零误报 ✓")


if __name__ == "__main__":
    test_homography_enabled()
    test_follow_right()
    test_follow_left()
    test_cross()
    test_corner_right()
    test_roi_dynamic_search()
    test_fixed_threshold_fallback()
    test_no_false_on_empty()
    print("全部 line_tracker_hsv 测试通过 ✓")
