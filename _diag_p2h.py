import sys, numpy as np
sys.path.insert(0, ".")
from sim.line_tracking_sim import LinePath, FrameRenderer, SimState, LINE_WAYPOINTS
from vision.line_detector import LineDetector

lp = LinePath(LINE_WAYPOINTS)
fr = FrameRenderer(lp)
det = LineDetector()

# 场景：robot 沿竖直线朝北，接近 P2 拐角(70,100)，到不同 y 位置后右转3/4次
for y0 in [91.82, 98, 102, 106]:
    for nt in [3, 4]:
        st = SimState(np.array([70.0, y0]), 88.0)
        st.apply_turn(-25.7*nt)
        r = det.detect(fr.render(st))
        p = r.primary
        print(f"y0={y0} 转{nt}次: angle={np.degrees(st.angle):.1f} orient={p.orientation} lx={p.lookahead_x:.1f} heading={p.heading_deg:.1f} lateral={p.lateral_offset:.1f}")
    print()
