import sys, numpy as np
sys.path.insert(0, ".")
from sim.line_tracking_sim import LinePath, FrameRenderer, SimState, LINE_WAYPOINTS
from vision.line_detector import LineDetector

lp = LinePath(LINE_WAYPOINTS)
fr = FrameRenderer(lp)
det = LineDetector()

# 重现 P1 转弯后状态：robot 在 P1 转弯 88 度后朝 88 度
# 从日志 step 17 对准完成时约 (69.6, 19.8)，然后循迹前进到 step 40
# 循迹中 step32 有一次 left_move
st = SimState(np.array([69.6, 19.8]), 88.0)
# step 18-31: 前进 14 步
st.apply_forward(2.0*14)
# step 32: left_move 1.9cm
st.apply_left_move(1.9)
# step 33-39: 前进 7 步 (33,34,35,36,37,38,39 = 7步)
st.apply_forward(2.0*7)
print(f"step40 pos={st.pos} angle={np.degrees(st.angle):.1f}")
frame = fr.render(st)
r = det.detect(frame)
if r.exists:
    p = r.primary
    print(f"  orient={p.orientation} elbow_y={p.elbow_px} lx={p.lookahead_x:.1f} heading={p.heading_deg:.1f}")

# 接近阶段 15 步
for i in range(15):
    st.apply_forward(2.0)
    if i in (0, 7, 14):
        f2 = fr.render(st)
        r2 = det.detect(f2)
        if r2.exists:
            pp = r2.primary
            print(f"  接近{i+1}: pos={st.pos} orient={pp.orientation} elbow_y={None if pp.elbow_px is None else pp.elbow_px[1]:.0f}")
print(f"接近结束 pos={st.pos} angle={np.degrees(st.angle):.1f}")

# 右转 3 次
st.apply_turn(-25.7*3)
print(f"右转3次后 pos={st.pos} angle={np.degrees(st.angle):.1f}")
f3 = fr.render(st)
r3 = det.detect(f3)
if r3.exists:
    pp = r3.primary
    print(f"  orient={pp.orientation} lx={pp.lookahead_x:.1f} heading={pp.heading_deg:.1f}")
else:
    print("  none")
