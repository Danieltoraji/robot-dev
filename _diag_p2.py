import sys, os, numpy as np
sys.path.insert(0, ".")
from sim.line_tracking_sim import LinePath, FrameRenderer, SimState, LINE_WAYPOINTS, _inject_mock_hiwonder
from vision.line_detector import LineDetector
from levels import line_seeker_tracking as lst

lp = LinePath(LINE_WAYPOINTS)
fr = FrameRenderer(lp)
det = LineDetector()

# 右弯 P2 (70,100)：机器人朝北(90度) 沿 y 接近
for y in [70, 80, 85, 90, 93, 95, 97, 98, 99, 100]:
    st = SimState(np.array([70.0, y]), 90.0)  # 朝北
    frame = fr.render(st)
    r = det.detect(frame)
    if r.exists:
        p = r.primary
        ep = p.elbow_px
        ey = None if ep is None else round(ep[1],1)
        print(f"y={y} -> orient={p.orientation} elbow_y={ey} curvature={p.curvature:.2f} heading={p.heading_deg:.1f}")
    else:
        print(f"y={y} -> none")
