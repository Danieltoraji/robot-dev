import sys, numpy as np
sys.path.insert(0, ".")
from sim.line_tracking_sim import LinePath, FrameRenderer, SimState, LINE_WAYPOINTS
from vision.line_detector import LineDetector

lp = LinePath(LINE_WAYPOINTS)
fr = FrameRenderer(lp)
det = LineDetector()

st = SimState(np.array([70.21, 91.82]), 88.0)
st.apply_turn(-25.7*3)   # 转3次 欠转
for i in range(5):
    st.apply_forward(2.0)  # 前进5步远离拐角
print(f"前进5步后: pos={np.round(st.pos,1)} angle={np.degrees(st.angle):.1f}")
r = det.detect(fr.render(st)); p = r.primary
print(f"  检测: orient={p.orientation} lx={p.lookahead_x:.1f} heading={p.heading_deg:.1f} lateral={p.lateral_offset:.1f}")

# 右转一次修正
st.apply_turn(-25.7)
r = det.detect(fr.render(st)); p = r.primary
print(f"右转1次后: angle={np.degrees(st.angle):.1f} orient={p.orientation} lx={p.lookahead_x:.1f} heading={p.heading_deg:.1f} lateral={p.lateral_offset:.1f}")

# 左移对齐看是否可行
for i in range(4):
    st.apply_left_move(1.9)
    r = det.detect(fr.render(st)); p = r.primary
    print(f"  左移{i+1}: pos={np.round(st.pos,1)} orient={p.orientation} lx={p.lookahead_x:.1f} heading={p.heading_deg:.1f} lateral={p.lateral_offset:.1f}")
