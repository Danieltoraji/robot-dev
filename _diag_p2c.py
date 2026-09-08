import sys, numpy as np
sys.path.insert(0, ".")
from sim.line_tracking_sim import LinePath, FrameRenderer, SimState, LINE_WAYPOINTS
from vision.line_detector import LineDetector

lp = LinePath(LINE_WAYPOINTS)
fr = FrameRenderer(lp)
det = LineDetector()

st = SimState(np.array([70.21, 91.82]), 88.0)
# 右转 3 次后
st.apply_turn(-25.7*3)
print(f"右转3次后 pos={st.pos} angle={np.degrees(st.angle):.1f}")
r = det.detect(fr.render(st))
if r.exists:
    p = r.primary
    print(f"  orient={p.orientation} lx={p.lookahead_x:.1f} heading={p.heading_deg:.1f} lateral={p.lateral_offset:.1f}")

# 尝试右移几次看 heading 变化（模拟对准）
for i in range(4):
    st.apply_right_move(2.2)
    r = det.detect(fr.render(st))
    if r.exists:
        p = r.primary
        print(f"  右移{i+1}: pos={np.round(st.pos,1)} lx={p.lookahead_x:.1f} heading={p.heading_deg:.1f} lateral={p.lateral_offset:.1f}")
    else:
        print(f"  右移{i+1}: none pos={np.round(st.pos,1)}")
