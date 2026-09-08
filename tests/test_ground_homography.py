# -*- coding: utf-8 -*-
"""ground_homography 单元测试：合成相机端到端验证

用已知位姿（位置/俯仰/偏航）+ camera_config 真实内参畸变，把宫格网格交点
投影成像素，再走 solve -> pixels_to_ground -> decompose -> ground_pose ->
target_in_robot_frame 全链路，断言恢复误差在容差内。头部偏航补偿用
"相机绕光心偏航"的精确模型合成，验证补偿公式与符号。

运行：python -m tests.test_ground_homography
"""

import numpy as np

from core.camera_config import CAMERA_INTRINSIC, CAMERA_DISTORTION
from core.ground_homography import (
    GroundHomography, grid_cell_center, grid_intersections,
    head_pulse_to_angle_deg, CAMERA_TO_BODY_FORWARD_CM,
)

K = CAMERA_INTRINSIC
D = CAMERA_DISTORTION

# 真值相机姿态（与 nav 俯仰档 1200 一致：俯角 27°）
CAM_POS = np.array([50.0, -20.0, 39.0])   # 场地系 cm，入口外 20cm 居中
PITCH_DEG = 27.0                          # (1500-1200)*0.09


def camera_rotation(bearing_deg, pitch_deg):
    """世界->相机旋转矩阵（列=世界基向量在相机系的坐标）

    场地系 x 右 y 前 z 上；相机 x 右 y 下(图像) z 前(视轴)。
    """
    a = np.radians(pitch_deg)
    f = np.radians(bearing_deg)
    sa, ca = np.sin(a), np.cos(a)
    sf, cf = np.sin(f), np.cos(f)
    # 相机三轴在世界系的方向 = 世界->相机旋转矩阵的行（p_cam = R @ p_world）
    r = np.array([cf, -sf, 0.0])                    # 图像右
    d = np.array([-sa * sf, -sa * cf, -ca])         # 图像下
    v = np.array([ca * sf, ca * cf, -sa])           # 视轴（前下）
    return np.vstack([r, d, v])


def project(pts_world, bearing_deg):
    """世界点 -> 像素（含畸变），相机位于 CAM_POS、俯仰 PITCH_DEG、偏航 bearing"""
    R = camera_rotation(bearing_deg, PITCH_DEG)
    p_cam = (R @ (np.asarray(pts_world, dtype=np.float64) - CAM_POS).T).T
    if np.any(p_cam[:, 2] <= 0):
        raise ValueError("测试点在相机背后")
    norm = p_cam[:, :2] / p_cam[:, 2:3]
    pts3 = np.column_stack([norm, np.ones(len(norm))]).reshape(-1, 1, 3)
    import cv2
    pix, _ = cv2.projectPoints(pts3, np.zeros(3), np.zeros(3), K, D)
    return pix[:, 0, :]


def visible_frame(bearing_deg, margin=100):
    """返回当前偏航档下「视野内」网格交点的 (像素, 地面坐标)

    与真实标定一致：只能点击看得见的点。图像外的点畸变逆运算不收敛
    （本机 k1=-0.384/k2=+0.285 在归一化半径>1 后不单调），必须剔除。
    """
    inters = grid_intersections()
    px = project(np.array([[x, y, 0.0] for x, y in inters]), bearing_deg)
    w, h = 2592, 1944
    keep = [(p, g) for p, g, (x, y) in
            zip(px, inters, inters)
            if margin < p[0] < w - margin and margin < p[1] < h - margin]
    px_v = np.array([k[0] for k in keep])
    gd_v = np.array([k[1] for k in keep], dtype=np.float64)
    return px_v, gd_v


def approx(v, ref, tol):
    assert abs(v - ref) <= tol, f"{v:.4f} != {ref:.4f}±{tol}"


def test_solve_and_backproject():
    """标定 -> 留出点反投影误差"""
    px_all, gd_all = visible_frame(0.0)
    assert len(px_all) >= 8, f"视野内标定点不足: {len(px_all)}"
    px_train, gd_train = px_all[:-3], gd_all[:-3]
    px_out, gd_out = px_all[-3:], gd_all[-3:]
    hg = GroundHomography.solve(px_train, gd_train, 1200)
    gd = hg.pixels_to_ground(px_out)
    for (tx, ty), (gx, gy) in zip(gd_out, gd):
        approx(float(gx), tx, 1e-3)
        approx(float(gy), ty, 1e-3)
    print("  solve/反投影：留出点误差 <1e-3 cm ✓")


def test_decompose_pose():
    """位姿分解恢复相机位置与高度"""
    px_all, gd_all = visible_frame(0.0)
    hg = GroundHomography.solve(px_all, gd_all, 1200)
    R, t, C, diff = hg.decompose()
    approx(float(C[0]), CAM_POS[0], 0.1)
    approx(float(C[1]), CAM_POS[1], 0.1)
    approx(float(C[2]), CAM_POS[2], 0.2)
    assert diff < 0.02, f"H 两列模长差 {diff:.3f} 过大"
    # 视轴落地方向：俯视 27° 时地面视线应指向 +y（正前）
    pos, fwd = hg.ground_pose()
    approx(float(fwd[0]), 0.0, 1e-6)
    approx(float(fwd[1]), 1.0, 1e-6)
    print(f"  分解：C=({C[0]:.2f},{C[1]:.2f},{C[2]:.2f}) 视线方向 ✓")


def test_robot_frame():
    """目标相对量：forward/lateral/bearing 与几何真值一致"""
    px_all, gd_all = visible_frame(0.0)
    hg = GroundHomography.solve(px_all, gd_all, 1200)
    # 正前目标：场地 (50,50)，相机锚点 (50,-20) → 纵向 70 + 机体偏移修正
    fwd_cm, lat_cm, bearing = hg.target_in_robot_frame((50.0, 50.0))
    approx(fwd_cm, 70.0 + CAMERA_TO_BODY_FORWARD_CM, 0.1)
    approx(lat_cm, 0.0, 1e-3)
    approx(bearing, 0.0, 1e-3)
    # 右侧目标 (70,30)：横向 +20（右正）
    fwd_cm, lat_cm, bearing = hg.target_in_robot_frame((70.0, 30.0))
    approx(lat_cm, 20.0, 0.1)
    assert bearing > 0, "右侧目标方位差应为正（右正左负）"
    # 左侧目标 (30,30)：横向 -20
    _, lat_cm, bearing = hg.target_in_robot_frame((30.0, 30.0))
    approx(lat_cm, -20.0, 0.1)
    assert bearing < 0, "左侧目标方位差应为负"
    print("  机器人系相对量：纵向/横向/方位差符号与幅值 ✓")


def test_head_yaw_compensation():
    """头部偏航帧经补偿后应恢复场地坐标（精确模型，含 ±40.5° 两档）"""
    px_c, gd_c = visible_frame(0.0)
    hg = GroundHomography.solve(px_c, gd_c, 1200)

    for head_deg in (40.5, -40.5, 20.0):
        pulse = HEAD_PULSE(head_deg)
        # 头部左转(θ>0) → 相机方位角 = -θ（向 -x 侧转）
        px_yaw, gd_yaw = visible_frame(-head_deg)
        gd = hg.pixels_to_ground(px_yaw, head_pulse=pulse)
        for (tx, ty), (gx, gy) in zip(gd_yaw, gd):
            approx(float(gx), tx, 0.05)
            approx(float(gy), ty, 0.05)
    print("  头部偏航补偿：±40.5°/±20° 档恢复误差 <0.05 cm ✓")


def HEAD_PULSE(angle_deg):
    return 1500 + int(round(angle_deg / 0.09))


def test_grid_geometry():
    """宫格位置编号 → 格中心（左下 6 恒空的布局约定）"""
    c0 = grid_cell_center(0)   # 左上（最远排最左）
    approx(float(c0[0]), 100.0 / 6, 1e-9)
    approx(float(c0[1]), 100.0 - 100.0 / 6, 1e-9)
    c6 = grid_cell_center(6)   # 左下恒空
    approx(float(c6[0]), 100.0 / 6, 1e-9)
    approx(float(c6[1]), 100.0 / 6, 1e-9)
    c8 = grid_cell_center(8)   # 右下
    approx(float(c8[0]), 100.0 - 100.0 / 6, 1e-9)
    print("  宫格几何：位置编号映射 ✓")


def test_noise_robustness():
    """0.5px 点击噪声下留出点误差应 <0.5cm（对应验收线 2cm@1m）"""
    rng = np.random.RandomState(7)
    px_all, gd_all = visible_frame(0.0)
    px_train, gd_train = px_all[:-3] + rng.normal(0, 0.5, (len(px_all) - 3, 2)), gd_all[:-3]
    px_out, gd_out = px_all[-3:], gd_all[-3:]
    hg = GroundHomography.solve(px_train, gd_train, 1200)
    gd = hg.pixels_to_ground(px_out)
    errs = [np.hypot(g[0] - t[0], g[1] - t[1]) for g, t in zip(gd, gd_out)]
    assert max(errs) < 0.5, f"噪声下留出点误差 {max(errs):.3f}cm 超限"
    R, t, C, diff = hg.decompose()
    assert abs(C[2] - 39.0) < 1.0, f"噪声下高度恢复 {C[2]:.2f}cm 偏差过大"
    print(f"  噪声鲁棒：0.5px 噪声留出点最大误差 {max(errs):.3f}cm，"
          f"高度恢复 {C[2]:.1f}cm ✓")


def test_save_load_roundtrip(tmp_path=None):
    """保存/加载往返一致"""
    import os
    import tempfile
    px_all, gd_all = visible_frame(0.0)
    hg = GroundHomography.solve(px_all, gd_all, 1200)
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    hg.save(path)
    hg2 = GroundHomography.load(1200, path=path)
    assert hg2 is not None
    gd1 = hg.pixels_to_ground(px_all)
    gd2 = hg2.pixels_to_ground(px_all)
    assert np.max(np.abs(gd1 - gd2)) < 1e-9
    os.remove(path)
    print("  持久化：save/load 往返 ✓")


def test_from_pose_analytic():
    """解析单应（无标定自举）：分解回位姿 + 反投影一致 + 格归属"""
    from core.ground_homography import GroundHomography, nearest_cell
    hg = GroundHomography.from_pose((50.0, -20.0), 39.0, 1200)
    R, t, C, diff = hg.decompose()
    approx(float(C[0]), 50.0, 1e-6)
    approx(float(C[1]), -20.0, 1e-6)
    approx(float(C[2]), 39.0, 1e-6)
    px_all, gd_all = visible_frame(0.0)
    gd = hg.pixels_to_ground(px_all)
    errs = [max(abs(g[0] - t0[0]), abs(g[1] - t0[1]))
            for g, t0 in zip(gd, gd_all)]
    assert max(errs) < 0.1, f"解析单应反投影误差 {max(errs):.3f}cm"
    # 格归属：格心附近判入，远处/格间返回 None
    assert nearest_cell((16.9, 83.1)) == 0
    assert nearest_cell((0.0, 0.0)) is None
    assert nearest_cell((200.0, 0.0)) is None
    print("  解析单应 from_pose：位姿往返/反投影/格归属 ✓")


if __name__ == "__main__":
    test_grid_geometry()
    test_solve_and_backproject()
    test_decompose_pose()
    test_robot_frame()
    test_head_yaw_compensation()
    test_noise_robustness()
    test_save_load_roundtrip()
    test_from_pose_analytic()
    print("全部 ground_homography 测试通过 ✓")
