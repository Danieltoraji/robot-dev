# -*- coding: utf-8 -*-
"""临时：受控对照——转一步前后各测一次（在机器人 Jupyter 核心里跑，用 exec_on_robot 发送）"""
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
print("头部回中；head=%s pitch=%s" % (getattr(st, "current_head_pulse", "?"),
                                     getattr(st, "current_pitch_pulse", "?")))


def measure(tag=102):
    p = st.capture_image()
    rs = st.detect_apriltag(p)
    ids = [int(r.tag_id) for r in rs]
    for r in rs:
        if int(r.tag_id) == tag:
            m = pb.corner_metrics(pb.undistort_corners(r.corners))
            z, yaw = pb.depth_and_yaw(m)
            return "id=%d Ψ=%.3f β=%+.0fpx γ=%+.0fpx w=%.0f z=%.1f tilt=%.3f yaw=%.1f°" % (
                tag, m["psi"], m["beta"], m["gamma"], m["w"], z, m["tilt"], yaw)
    return "未检出 %d（本帧见 %s）" % (tag, ids)


print("\n=== 基线（不动作，连测 2 次）===")
for i in range(2):
    print("  #%d %s" % (i + 1, measure()))

print("\n=== 发一次 turn_right_small_step ×1 ===")
st.act("turn_right_small_step", 1)
time.sleep(1.0)
for i in range(3):
    print("  #%d %s" % (i + 1, measure()))

print("\n=== 再发一次反向 turn_left_small_step ×1（试回位）===")
st.act("turn_left_small_step", 1)
time.sleep(1.0)
for i in range(2):
    print("  #%d %s" % (i + 1, measure()))
