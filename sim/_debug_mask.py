# -*- coding: utf-8 -*-
"""诊断：case13 (10,20,0) 拐角帧的掩膜连通域 + 关键赛道点的图像坐标"""
import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.line_tracking_sim import (LinePath, FrameRenderer, SimState,
                                   LINE_WAYPOINTS)

line_path = LinePath(LINE_WAYPOINTS)
renderer = FrameRenderer(line_path)
sim_state = SimState(np.array([10.0, 20.0]), 0.0)
sim_state.head_pulse = 1500

# 1) 关键赛道点的图像坐标
print("=== 关键赛道点图像坐标（全图 640x480）===")
test_pts = [
    ("近臂(15,20)", (15, 20)), ("近臂(25,20)", (25, 20)),
    ("近臂(35,20)", (35, 20)), ("近臂末(39,20)", (39, 20)),
    ("拐点(40,20)", (40, 20)), ("远臂(40,25)", (40, 25)),
    ("远臂(40,40)", (40, 40)), ("远臂(40,55)", (40, 55)),
    ("远臂(40,59)", (40, 59)), ("P2段(50,60)", (50, 60)),
    ("P2段(60,60)", (60, 60)), ("P2段(70,60)", (70, 60)),
]
for name, wp in test_pts:
    fwd, rgt = line_path.world_to_robot(np.array(wp, dtype=np.float64), sim_state)
    f_cam, r_cam = line_path.robot_to_camera(fwd, rgt, sim_state.head_angle_rad)
    if f_cam <= 0.5:
        print(f"  {name}: 相机后方 (f={f_cam:.1f})")
        continue
    arr = np.array([[f_cam, r_cam]], dtype=np.float32).reshape(-1, 1, 2)
    img = cv2.perspectiveTransform(arr, renderer._H_matrix).reshape(-1, 2)[0]
    in_roi = 240 <= img[1] <= 480
    print(f"  {name}: ground(f={fwd:.0f},r={rgt:.0f}) cam(f={f_cam:.0f},r={r_cam:.0f})"
          f" -> img({img[0]:.0f},{img[1]:.0f})  ROI内={in_roi}")

# 2) 渲染帧 + 掩膜连通域
frame = renderer.render(sim_state)
roi = frame[int(480 * 0.5):, :]
print(f"\n=== ROI shape: {roi.shape} ===")
import levels.line_seeker_tracking as lst_import  # noqa: F401 (确保 detector 可导入)
det = None
from sim.line_tracking_sim import lst
det = lst.detector
mask = det._make_mask(roi)
n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
print(f"连通域数（含背景）: {n}")
for i in range(1, n):
    x, y, w, h, area = stats[i]
    print(f"  组件{i}: bbox=({x},{y})-({x+w},{y+h}) w={w} h={h} area={area}")

# 3) 保存掩膜图
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_frames",
                   "case13_mask.png")
cv2.imwrite(out, mask)
print(f"\n掩膜已保存: {out}")

# 4) 骨架段
segs = det._extract_segments(mask)
print(f"\n骨架段数: {len(segs)}")
for i, s in enumerate(segs):
    xs = [p[0] for p in s]; ys = [p[1] for p in s]
    print(f"  段{i}: {len(s)}点 x∈[{min(xs):.0f},{max(xs):.0f}] "
          f"y∈[{min(ys):.0f},{max(ys):.0f}] 首={s[0]} 尾={s[-1]}")
