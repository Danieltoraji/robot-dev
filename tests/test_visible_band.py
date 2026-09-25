# -*- coding: utf-8 -*-
"""仿真器可见地面带回归（tests/test_visible_band.py）

把现场实测的"1040 档可见 2~70cm"变成回归门。

为什么需要它：2026-09-25 之前，仿真器用 CAM_HEIGHT=39 且**漏掉相机的安装
下俯偏移**，实际渲染出的可见地面带是 [13.9, 177.7]cm —— 与现场实测的
[2, 70]cm **远近两头都反着**。所有基于那个仿真器的端到端测试，验证的都是
一个不存在的世界。

实测结果（h=34.5cm、偏移 15°、pitch 1040，木条下沿严格摆在 d 处）：
    近界 2.2~2.5cm，远界 69~70cm，带内测距误差 ≤0.04cm
与现场 [2, 70]cm 一致，也与解析式 min_visible_ground_cm 同量级。

运行：python tests/test_visible_band.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.camera_config import HEAD_CENTER
from core.ground_line_meter import build_meter, min_visible_ground_cm
from sim.stairs_hurdle_sim import SimStairsRobot, StairsScene
from vision.red_line_detector import RedLineDetector

PITCH = 1040                    # 现场观测档
BAR_HALF_THICK_CM = 1.0         # 场景把木条近侧面放在 bar_y + 半厚处
# 现场实测的可靠可见范围（2026-09-25 用户给出）
FIELD_NEAR_CM, FIELD_FAR_CM = 2.0, 70.0
# 旧版（错误相机模型）实际渲染出的带，用于反向回归
OLD_NEAR_CM, OLD_FAR_CM = 13.9, 177.7

PROBE_CM = [1.5, 2.0, 2.5, 3.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0,
            60.0, 65.0, 69.0, 70.0, 80.0, 90.0, 120.0]


def make_rig():
    """建一套"机器人 + 场景 + 度量"（场景里只有木条，无胶条）"""
    scene = StairsScene(bar_dist=30.0)
    scene.tape_visible = False
    robot = SimStairsRobot(scene, seed=0, start=(0.0, 0.0),
                           heading_deg=0.0, wobble_sigma_deg=0.0)
    robot.set_pitch(PITCH)
    robot.set_head(HEAD_CENTER)
    meter = build_meter(PITCH, robot.CAM_HEIGHT,
                        pitch_offset_deg=robot.CAM_PITCH_OFFSET_DEG)
    return scene, robot, meter


def measure_at(scene, robot, meter, detector, d_cm):
    """把木条**下沿**摆到正前方 d_cm 处，返回测距读数（不可见返回 None）"""
    scene.bar_y = float(d_cm) - BAR_HALF_THICK_CM
    det = detector.detect(robot.capture_frame())
    if not det.exists:
        return None
    m = meter.measure(det.components)
    return float(m.forward_cm) if m.exists else None


def scan_band():
    """扫一遍探测距离，返回 {d: 读数 or None}"""
    scene, robot, meter = make_rig()
    det = RedLineDetector()
    return {d: measure_at(scene, robot, meter, det, d) for d in PROBE_CM}


def test_camera_model_is_not_the_old_one():
    """仿真器的相机模型本身不得退回旧值"""
    assert abs(SimStairsRobot.CAM_HEIGHT - 39.0) > 1.0, \
        "CAM_HEIGHT 退回 39cm（应为卷尺实测 34.5）"
    assert SimStairsRobot.CAM_PITCH_OFFSET_DEG > 5.0, \
        "安装下俯偏移被清零（这是旧版可见带失真的根因）"
    print(f"  相机模型：h={SimStairsRobot.CAM_HEIGHT}cm "
          f"offset={SimStairsRobot.CAM_PITCH_OFFSET_DEG}° ✓")


def test_band_matches_field():
    band = scan_band()
    visible = sorted(d for d, v in band.items() if v is not None)
    near, far = min(visible), max(visible)
    # 近端：现场 2cm；这里 2.5cm 可见、2.0cm 不可见（几何近界约 2.4cm）
    assert band[2.5] is not None, "2.5cm 处应可见"
    # 远端：现场 70cm；这里 69cm 可见、70cm 不可见（几何远界约 69.5cm）
    assert band[69.0] is not None, "69cm 处应可见"
    assert band[70.0] is None, "70cm 处不应可见"
    assert abs(near - FIELD_NEAR_CM) <= 1.0, f"近界 {near}cm 与现场 2cm 差太多"
    assert abs(far - FIELD_FAR_CM) <= 1.5, f"远界 {far}cm 与现场 70cm 差太多"
    print(f"  渲染可见带 = [{near:.1f}, {far:.1f}]cm"
          f"（现场 [{FIELD_NEAR_CM:.0f}, {FIELD_FAR_CM:.0f}]cm；"
          f"旧模型 [{OLD_NEAR_CM}, {OLD_FAR_CM}]）✓")


def test_old_wrong_band_is_gone():
    """反向回归：旧模型的两个端点特征都必须消失"""
    band = scan_band()
    # 旧模型看不到 13.9cm 以内 —— 现在 10cm 必须可见
    assert band[10.0] is not None, "10cm 处应可见（旧模型此处是盲区）"
    # 旧模型能看到 177.7cm —— 现在 120cm 必须不可见
    assert band[120.0] is None, "120cm 处不应可见（旧模型此处可见）"
    print("  旧模型两端特征均已消失（10cm 可见 / 120cm 不可见）✓")


def test_accuracy_within_band():
    band = scan_band()
    worst = 0.0
    for d in [2.5, 5.0, 10.0, 20.0, 40.0, 60.0, 69.0]:
        v = band[d]
        assert v is not None, f"{d}cm 应可见"
        worst = max(worst, abs(v - d))
    assert worst <= 0.5, f"带内测距 worst 误差 {worst:.2f}cm 超 0.5cm"
    print(f"  带内测距 worst 误差 {worst:.2f}cm（≤0.5cm）✓")


def test_analytic_matches_render():
    """解析近界必须与渲染近界同量级（畸变上下不对称会带来小偏差）"""
    h = SimStairsRobot.CAM_HEIGHT
    off = SimStairsRobot.CAM_PITCH_OFFSET_DEG
    analytic = min_visible_ground_cm(h, PITCH, pitch_offset_deg=off)
    assert 1.5 <= analytic <= 4.0, f"解析近界 {analytic:.2f}cm 不合理"
    band = scan_band()
    assert band[2.5] is not None and band[2.0] is None, \
        "渲染近界应落在 (2.0, 2.5] 之间"
    print(f"  解析近界 {analytic:.2f}cm 与渲染近界（2.0 不可见 / 2.5 可见）一致 ✓")


if __name__ == "__main__":
    test_camera_model_is_not_the_old_one()
    test_band_matches_field()
    test_old_wrong_band_is_gone()
    test_accuracy_within_band()
    test_analytic_matches_render()
    print("全部可见带测试通过 ✓")
