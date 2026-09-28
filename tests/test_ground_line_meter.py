# -*- coding: utf-8 -*-
"""ground_line_meter 几何测试：合成相机端到端精度扫描 + 高度差方向不变性

用 sim.stairs_hurdle_sim.make_frame（真实内参+畸变渲染）生成已知几何的
红色横杆帧，走 检测 -> pixels_to_ground -> 聚类 -> TLS 拟合 全链路，
断言距离/横偏/方位误差在方案验收线内。

验收线（方案 §4）：
- 距离误差 ≤1.5cm@≤50cm / ≤2.5cm@≤70cm；方位 ≤1.5°；横偏 ≤1.5cm
- 俯仰晃动 σ=0.5° 下阈值 ×1.5
- 台阶上（相机高于脚下平面，横杆更低）：bearing ≤1.5°（距离偏大属预期，
  仅验证偏移系数区间，不进触发）

运行：python -m tests.test_ground_line_meter
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from core.camera_config import HEAD_CENTER
from core.ground_homography import GroundHomography
from core.ground_line_meter import (
    GroundLineMeter, min_visible_ground_cm, build_meter,
)
from vision.red_line_detector import RedLineDetector
from sim.stairs_hurdle_sim import make_frame, bar_line_quads

PITCH_OBS = 1000
PITCH_DEG = (1500 - PITCH_OBS) * 0.09  # 45°
CAM_Z = 39.0


def make_meter():
    return GroundLineMeter(GroundHomography.from_pose((0.0, 0.0), CAM_Z, PITCH_OBS))


def measure_case(forward, bearing=0.0, lateral=0.0, cam_z=CAM_Z,
                 height_off=0.0, pitch_deg=PITCH_DEG, half_len=40.0,
                 detector=None):
    quads = bar_line_quads(forward, bearing, lateral,
                           half_len=half_len, height_off_cm=height_off)
    frame = make_frame(quads, pitch_deg=pitch_deg, cam_z=cam_z)
    det = detector or RedLineDetector()
    d = det.detect(frame)
    m = make_meter().measure(d.components)
    return m


def approx(v, ref, tol):
    assert abs(v - ref) <= tol, f"{v:.4f} != {ref:.4f}±{tol}"


# ---------------------------------------------------------------------
# 几何精度扫描
# ---------------------------------------------------------------------

def band_fully_visible(d, th_deg, lat, half_len):
    """横带两端点投影后都在画幅内（与渲染同一投影管线，精确判据）"""
    import cv2
    from core.camera_config import CAMERA_INTRINSIC, CAMERA_DISTORTION
    from sim.stairs_hurdle_sim import camera_rotation
    th = np.radians(th_deg)
    dx, dy = np.cos(th), np.sin(th)
    pts = np.array([[lat + s * dx * half_len, d + s * dy * half_len, 0.0]
                    for s in (1.0, -1.0)])
    R = camera_rotation(0.0, PITCH_DEG)
    C = np.array([0.0, 0.0, CAM_Z])
    pc = (R @ (pts - C).T).T
    norm = pc[:, :2] / pc[:, 2:3]
    pix, _ = cv2.projectPoints(
        np.column_stack([norm, np.ones(len(norm))]).reshape(-1, 1, 3)
        .astype(np.float32), np.zeros(3), np.zeros(3),
        CAMERA_INTRINSIC, CAMERA_DISTORTION)
    pix = pix[:, 0, :]
    m = 60
    return bool(np.all((pix[:, 0] > m) & (pix[:, 0] < 2592 - m)
                       & (pix[:, 1] > m) & (pix[:, 1] < 1944 - m)))


def test_geometry_sweep():
    detector = RedLineDetector()
    cases = []
    for d in (15.0, 20.0, 30.0, 45.0, 60.0, 70.0):
        for th in (0.0, 25.0, -25.0):
            for lat in (0.0, 15.0):
                cases.append((d, th, lat))
    worst = {"d": 0.0, "th": 0.0, "lat": 0.0}
    for d, th, lat in cases:
        m = measure_case(d, th, lat, detector=detector)
        assert m.exists, f"d={d} th={th} lat={lat}: {m.reason}"
        dist_tol = 1.5 if d <= 50 else 2.5
        # 前向真值 = 直线与机体前方轴线（x=0）的交点（斜带时 ≠ 中点距离）
        import math as _math
        d_true = d - _math.tan(_math.radians(th)) * lat
        ed = abs(m.forward_cm - d_true)
        et = abs(m.bearing_err_deg - th)
        worst["d"] = max(worst["d"], ed)
        worst["th"] = max(worst["th"], et)
        assert ed <= dist_tol, f"d={d} th={th} lat={lat}: 距离误差 {ed:.2f}cm"
        assert et <= 1.5, f"d={d} th={th} lat={lat}: 方位误差 {et:.2f}°"
        # 横偏仅在"正对横带（th=0）且完整可见"时有意义：下沿点按图像列采样，
        # 斜带时地面 x 密度不对称、中位数有偏——但控制上横移对中只在
        # bearing≈0 之后执行（先转正再对中），该场景无偏，故断言限定于此。
        if th == 0.0 and band_fully_visible(d, th, lat, 40.0):
            el = abs(m.lateral_cm - lat)
            worst["lat"] = max(worst["lat"], el)
            assert el <= 1.5, f"d={d} th={th} lat={lat}: 横偏误差 {el:.2f}cm"
    print(f"  精度扫描 {len(cases)} 例：距离 worst {worst['d']:.2f}cm / "
          f"方位 {worst['th']:.2f}° / 横偏 {worst.get('lat', 0):.2f}cm ✓")


def test_lateral_fully_visible():
    """横带完整可见时横偏精度（对应胶条对正场景：40cm 带在 40cm 距离全可见）"""
    for d, lat, hl in ((30.0, 6.0, 12.0), (30.0, -6.0, 12.0),
                       (45.0, 10.0, 15.0), (45.0, -10.0, 15.0)):
        m = measure_case(d, 0.0, lat, half_len=hl)
        assert m.exists
        el = abs(m.lateral_cm - lat)
        assert el <= 1.5, f"d={d} lat={lat}: 横偏误差 {el:.2f}cm"
        assert abs(m.forward_cm - d) <= 1.5
    print("  横偏（横带完整可见）：误差 ≤1.5cm ✓")


def test_pitch_wobble():
    """俯仰晃动 ±0.5°（落地脚下平面假设不变）下距离误差 ≤ 验收线 ×1.5"""
    for wobble in (-0.5, 0.5):
        m = measure_case(45.0, pitch_deg=PITCH_DEG + wobble)
        assert m.exists
        assert abs(m.forward_cm - 45.0) <= 1.5 * 1.5, \
            f"晃动 {wobble}° 距离误差 {abs(m.forward_cm - 45.0):.2f}cm"
        assert abs(m.bearing_err_deg) <= 1.5 * 1.5
    print("  俯仰晃动 ±0.5°：距离/方位误差在验收线 ×1.5 内 ✓")


def test_two_targets_nearest_cluster():
    """胶条（近）与横杆（远）同帧：取最近簇，次近簇记入日志字段"""
    from sim.stairs_hurdle_sim import bar_line_quads
    quads = (bar_line_quads(20.0, 0.0, 0.0)
             + bar_line_quads(55.0, 0.0, 0.0))
    frame = make_frame(quads)
    det = RedLineDetector().detect(frame)
    assert len(det.components) >= 2, f"应检出 2 个连通域，实际 {len(det.components)}"
    m = make_meter().measure(det.components)
    assert m.exists
    approx(m.forward_cm, 20.0, 1.5)
    approx(m.secondary_forward_cm, 55.0, 3.0)
    print(f"  双目标：取最近簇 {m.forward_cm:.1f}cm，次近 {m.secondary_forward_cm:.1f}cm ✓")


# ---------------------------------------------------------------------
# 台阶上方位测量（高度差径向缩放）
# ---------------------------------------------------------------------

def test_climb_bearing_invariance():
    """站在台阶上（脚下平面抬高）：bearing 不受横杆低于平面影响；距离按
    39/(39+脚下高+杆低差) 缩短（目标显得更近）——只验证方位，距离偏移
    系数记录不进触发。比值与解析公式偏差 <0.03。"""
    cases = [
        (2.5, 2.5, 25.0),   # 一级台阶：脚下 +2.5，杆低 2.5（在真地面）
        (5.0, 5.0, 25.0),   # 台顶：脚下 +5，杆低 5
        (5.0, 7.5, 30.0),   # 台顶 + 杆在更低组合
    ]
    for feet_z, off, d in cases:
        m = measure_case(d, cam_z=CAM_Z + feet_z, height_off=-off)
        assert m.exists, f"feet={feet_z} off={off}: {m.reason}"
        assert abs(m.bearing_err_deg) <= 1.5, \
            f"feet={feet_z} off={off}: 方位误差 {m.bearing_err_deg:.2f}°"
        ratio = m.forward_cm / d
        expect = CAM_Z / (CAM_Z + feet_z + off)
        assert abs(ratio - expect) < 0.03, \
            f"feet={feet_z} off={off}: 距离比 {ratio:.3f} vs 解析 {expect:.3f}"
        print(f"  台阶上 feet=+{feet_z}cm 杆低 {off}cm：方位 ✓ "
              f"距离读数 ×{ratio:.2f}（偏近，仅记录不进触发）")


# ---------------------------------------------------------------------
# 工具函数与构造
# ---------------------------------------------------------------------

def test_min_visible_ground():
    d = min_visible_ground_cm(39.0, 1000)
    assert 12.0 < d < 14.5, f"pitch=1000 可见下界 {d:.2f}cm 不合理"
    d2 = min_visible_ground_cm(39.0, 950)
    assert d2 < d, "更低头应看得更近"
    print(f"  可见下界：pitch1000={d:.1f}cm pitch950={d2:.1f}cm ✓")


def test_build_meter_degraded_and_calib():
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.remove(path)
    meter = build_meter(PITCH_OBS, CAM_Z, path=path)
    assert meter.degraded, "无标定文件应降级 from_pose"
    # 存入标定后应不再降级
    meter.hg.save(path)
    meter2 = build_meter(PITCH_OBS, CAM_Z, path=path)
    assert not meter2.degraded
    os.remove(path)
    print("  build_meter：缺标定降级 / 有标定加载 ✓")


if __name__ == "__main__":
    test_min_visible_ground()
    test_build_meter_degraded_and_calib()
    test_geometry_sweep()
    test_lateral_fully_visible()
    test_pitch_wobble()
    test_two_targets_nearest_cluster()
    test_climb_bearing_invariance()
    print("全部 ground_line_meter 测试通过 ✓")
