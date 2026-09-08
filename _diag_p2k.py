import sys, numpy as np
sys.path.insert(0, ".")
from sim.line_tracking_sim import LinePath, FrameRenderer, SimState, LINE_WAYPOINTS
from vision.line_detector import LineDetector

lp = LinePath(LINE_WAYPOINTS)
fr = FrameRenderer(lp)
det = LineDetector()

def align_then_heading(st, max_steps=25, heading_thresh=8.0):
    for i in range(max_steps):
        r = det.detect(fr.render(st))
        if not r.exists:
            print(f"  step{i}: 丢线"); return False
        p = r.primary
        if p.orientation != "follow":
            print(f"  step{i}: orient={p.orientation}"); return False
        lat = p.lateral_offset; lx = p.lookahead_x; h = p.heading_deg
        # ① 粗对齐 lateral
        if abs(lat) > 40:
            act = "右移" if lat > 0 else "左移"
            st.apply_right_move(2.2) if lat > 0 else st.apply_left_move(1.9)
        # ② 细对齐 lookahead
        elif abs(lx) > 12:
            act = "右移" if lx > 0 else "左移"
            st.apply_right_move(2.2) if lx > 0 else st.apply_left_move(1.9)
        # ③ heading 修正
        elif abs(h) > heading_thresh:
            act = "右转" if h > 0 else "左转"
            st.apply_turn(-25.7) if h > 0 else st.apply_turn(22.0)
        else:
            print(f"  step{i}: 完成 h={h:.1f} lx={lx:.1f} lat={lat:.1f} angle={np.degrees(st.angle):.1f} pos={np.round(st.pos,1)}")
            return True
        print(f"  step{i}: {act} h={h:.1f} lx={lx:.1f} lat={lat:.1f} angle={np.degrees(st.angle):.1f} pos={np.round(st.pos,1)}")
    return False

print("=== P2 (y0=98) 转3次后，对齐→heading修正 ===")
st = SimState(np.array([70.0, 98.0]), 88.0)
st.apply_turn(-25.7*3)
align_then_heading(st)

print("=== P1 (y0=20 转4次) 对齐→heading修正 ===")
st = SimState(np.array([70.0, 16.0]), 0.0)
st.apply_turn(22.0*4)  # 左转88°
align_then_heading(st)
