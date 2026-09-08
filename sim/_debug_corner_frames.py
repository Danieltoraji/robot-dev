# -*- coding: utf-8 -*-
"""临时调试：渲染拐角专项用例的合成帧并保存 PNG + 打印检测细节"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sim.line_tracking_sim as sim_mod
from sim.line_tracking_sim import (LinePath, FrameRenderer, SimState,
                                   LINE_WAYPOINTS, VISION_TEST_CASES)
import cv2

line_path = LinePath(LINE_WAYPOINTS)
renderer = FrameRenderer(line_path)
sim_state = SimState(np.array([5.0, 20.0]), 0.0)

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_frames")
os.makedirs(out_dir, exist_ok=True)

cases = [c for c in VISION_TEST_CASES if "拐角专项" in c[4]]
for px, py, ang, head, desc in cases:
    sim_state.pos = np.array([px, py], dtype=np.float64)
    sim_state.angle = np.radians(ang)
    sim_state.head_pulse = head
    frame = renderer.render(sim_state)
    fname = os.path.join(out_dir, f"case_{px}_{py}_{ang}_{head}.png")
    cv2.imwrite(fname, frame)
    print(f"saved: {fname}  ({desc})")

    # 检测细节
    result = sim_mod.lst.detector.detect(frame)
    if result.exists:
        p = result.primary
        print(f"  primary: {p.orientation} lx={p.lookahead_x:.1f} "
              f"lat={p.lateral_offset:.1f} head={p.heading_deg:.1f} "
              f"straight={p.straightness:.3f} npts={len(p.points)}")
        for j, o in enumerate(result.others):
            print(f"  other[{j}]: {o.orientation} lx={o.lookahead_x:.1f} "
                  f"npts={len(o.points)} straight={o.straightness:.3f}")
        # 打印 primary 点列的首尾（看形状）
        if p.points:
            pts = p.points
            print(f"  points head: {pts[:3]}")
            print(f"  points mid: {pts[len(pts)//2-1:len(pts)//2+1]}")
            print(f"  points tail: {pts[-3:]}")
    else:
        print("  exists=False")
    print()
