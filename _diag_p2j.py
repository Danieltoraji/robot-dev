import sys, numpy as np
sys.path.insert(0, ".")
from sim.line_tracking_sim import LinePath, FrameRenderer, SimState, LINE_WAYPOINTS
from vision.line_detector import LineDetector

lp = LinePath(LINE_WAYPOINTS)
fr = FrameRenderer(lp)
det = LineDetector()

# P2 转3次后，模拟带 heading 修正的循迹
st = SimState(np.array([70.0, 98.0]), 88.0)
st.apply_turn(-25.7*3)
print(f"起点: pos={np.round(st.pos,1)} angle={np.degrees(st.angle):.1f}")
for step in range(30):
    r = det.detect(fr.render(st))
    if not r.exists:
        print(f"step{step}: 丢线 pos={np.round(st.pos,1)}"); break
    p = r.primary
    act = ""
    if p.orientation != "follow":
        print(f"step{step}: orient={p.orientation} pos={np.round(st.pos,1)}"); break
    if abs(p.heading_deg) > 10:
        if p.heading_deg > 0:
            st.apply_turn(-25.7); act="右转"
        else:
            st.apply_turn(22.0); act="左转"
    elif abs(p.lookahead_x) > 12:
        if p.lookahead_x > 0:
            st.apply_right_move(2.2); act="右移"
        else:
            st.apply_left_move(1.9); act="左移"
    else:
        st.apply_forward(2.0); act="前进"
    print(f"step{step}: {act} h={p.heading_deg:.1f} lx={p.lookahead_x:.1f} lat={p.lateral_offset:.1f} pos={np.round(st.pos,1)} angle={np.degrees(st.angle):.1f}")
