import sys, numpy as np
sys.path.insert(0, ".")
from sim.line_tracking_sim import LinePath, FrameRenderer, SimState, LINE_WAYPOINTS
from vision.line_detector import LineDetector

lp = LinePath(LINE_WAYPOINTS)
fr = FrameRenderer(lp)
det = LineDetector()

def simulate(y0, nturns):
    st = SimState(np.array([70.0, y0]), 88.0)
    st.apply_turn(-25.7*nturns)
    results = []
    for i in range(10):
        r = det.detect(fr.render(st))
        if not r.exists:
            results.append((None,)); break
        p = r.primary
        results.append((p.orientation, round(p.lookahead_x,1), round(p.heading_deg,1), round(p.lateral_offset,1), np.round(st.pos,1)))
        # 简单横向对齐策略：lateral>40 右移，lookahead>12 右移，<-12 左移
        if p.orientation == "follow":
            if abs(p.lateral_offset) > 40:
                st.apply_right_move(2.2) if p.lateral_offset > 0 else st.apply_left_move(1.9)
            elif abs(p.lookahead_x) > 12:
                st.apply_right_move(2.2) if p.lookahead_x > 0 else st.apply_left_move(1.9)
            else:
                break
        else:
            break
    return results

for y0 in [95, 97, 98, 99, 100, 101, 102]:
    res = simulate(y0, 3)
    first = res[0]
    last = res[-1]
    print(f"y0={y0}: 起点 orient={first[0]} lx={first[1]} h={first[2]} lat={first[3]}  |  对齐{len(res)}步后 h={last[2] if len(last)>1 else '-'} lat={last[3] if len(last)>1 else '-'} pos={last[4] if len(last)>1 else '-'}")
