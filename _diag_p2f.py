import sys, numpy as np
sys.path.insert(0, ".")
from sim.line_tracking_sim import LinePath, FrameRenderer, SimState, LINE_WAYPOINTS
from vision.line_detector import LineDetector

lp = LinePath(LINE_WAYPOINTS)
fr = FrameRenderer(lp)
det = LineDetector()

# P2 转3次后，前进若干步，看 heading/lateral 变化
st = SimState(np.array([70.21, 91.82]), 88.0)
st.apply_turn(-25.7*3)
print(f"转3次: angle={np.degrees(st.angle):.1f}")
for i in range(8):
    r = det.detect(fr.render(st))
    p = r.primary
    print(f"  前进{i}步: pos={np.round(st.pos,1)} orient={p.orientation} lx={p.lookahead_x:.1f} heading={p.heading_deg:.1f} lateral={p.lateral_offset:.1f}")
    st.apply_forward(2.0)
