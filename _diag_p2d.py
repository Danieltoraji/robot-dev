import sys, numpy as np
sys.path.insert(0, ".")
from sim.line_tracking_sim import LinePath, FrameRenderer, SimState, LINE_WAYPOINTS
from vision.line_detector import LineDetector

lp = LinePath(LINE_WAYPOINTS)
fr = FrameRenderer(lp)
det = LineDetector()

st = SimState(np.array([70.21, 91.82]), 88.0)
st.apply_turn(-25.7*3)  # 右转3次
for i in range(6):
    st.apply_left_move(1.9)
    r = det.detect(fr.render(st))
    if r.exists:
        p = r.primary
        print(f"左移{i+1}: pos={np.round(st.pos,1)} orient={p.orientation} lx={p.lookahead_x:.1f} heading={p.heading_deg:.1f} lateral={p.lateral_offset:.1f}")
    else:
        print(f"左移{i+1}: none pos={np.round(st.pos,1)}")
