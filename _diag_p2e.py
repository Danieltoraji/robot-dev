import sys, numpy as np
sys.path.insert(0, ".")
from sim.line_tracking_sim import LinePath, FrameRenderer, SimState, LINE_WAYPOINTS
from vision.line_detector import LineDetector

lp = LinePath(LINE_WAYPOINTS)
fr = FrameRenderer(lp)
det = LineDetector()

# 转3次 vs 转4次对比
for nturns, tag in [(3, "转3次"), (4, "转4次")]:
    st = SimState(np.array([70.21, 91.82]), 88.0)
    st.apply_turn(-25.7*nturns)
    r = det.detect(fr.render(st))
    p = r.primary
    print(f"{tag}: angle={np.degrees(st.angle):.1f} orient={p.orientation} lx={p.lookahead_x:.1f} heading={p.heading_deg:.1f} lateral={p.lateral_offset:.1f}")
