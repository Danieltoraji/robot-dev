# -*- coding: utf-8 -*-
"""临时：极性判定——连发 4 次同名小步，看 |Ψ−1| 是变小（对）还是变大（反）。

结果只用来决定 turn_sign 的默认值。人守着，随时 Ctrl+C。
"""
import os
import sys
import time

os.chdir("/home/pi/Robot_Competition")
sys.path.insert(0, "/home/pi/Robot_Competition")

from core.robot_core import RobotState
from levels import press_button as pb

st = RobotState(tag_poses={})
try:
    st.set_head(pb.HEAD_CENTER, force=True)
except TypeError:
    st.current_head_pulse = None
    st.set_head(pb.HEAD_CENTER)


def measure(tag=102):
    p = st.capture_image()
    for r in st.detect_apriltag(p):
        if int(r.tag_id) == tag:
            m = pb.corner_metrics(pb.undistort_corners(r.corners))
            z, yaw = pb.depth_and_yaw(m)
            return "Ψ=%.4f β=%+.0fpx z=%.1fcm yaw=%.1f° |Ψ-1|=%.4f" % (
                m["psi"], m["beta"], z, yaw, abs(m["psi"] - 1.0))
    return "未检出"


print("基线: %s" % measure())
print("\n--- 连发 4 次 turn_right_small_step x1 ---")
for i in range(4):
    st.act("turn_right_small_step", 1)
    time.sleep(0.8)
    print("  右%2d: %s" % (i + 1, measure()))
print("\n--- 连发 4 次 turn_left_small_step x1 （回摆）---")
for i in range(4):
    st.act("turn_left_small_step", 1)
    time.sleep(0.8)
    print("  左%2d: %s" % (i + 1, measure()))
