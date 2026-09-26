import sys; sys.path.insert(0, ".")
import numpy as np
import levels.press_button as pb

class R:
    def __init__(s,t,c): s.tag_id, s.corners = t, c

class FakeState:
    def __init__(s, center_px, width_px, appear_after=0, tag=102, grow=1.6, deg_per_px=None):
        s.center, s.width, s.n = center_px, width_px, 0
        s.appear_after, s.tag, s.calls = appear_after, tag, []
        s.grow = grow
        s.dpp = deg_per_px if deg_per_px is not None else (180.0/2592.0)
        s.current_head_pulse = None; s.head = 1500
    def capture_image(s): s.n += 1; return "/tmp/f.jpg"
    def detect_apriltag(s, p):
        if s.n <= s.appear_after: return []
        w, cy = s.width, 972.0
        c = np.array([[s.center-w/2, cy-w/2],[s.center+w/2, cy-w/2],
                      [s.center+w/2, cy+w/2],[s.center-w/2, cy+w/2]])
        return [R(s.tag, c)]
    def set_head(s, pulse, move_time_ms=500):
        s.calls.append(("head", pulse)); s.head = pulse
    def act(s, name, times=1):
        s.calls.append(("act", name))
        px = s.dpp * 1944.903664123011 / 57.2958
        if name == "turn_left":               s.center -= pb.TURN_L_DEG / s.dpp
        elif name == "turn_right":            s.center += pb.TURN_R_DEG / s.dpp
        elif name == "turn_left_small_step":  s.center -= pb.TURN_L_SMALL_DEG / s.dpp
        elif name == "turn_right_small_step": s.center += pb.TURN_R_SMALL_DEG / s.dpp
        elif name == "go_forward":            s.width *= s.grow

import signal
def run(name, **kw):
    st = FakeState(**kw)
    ok = pb.press_button(st, 102)
    acts = [c[1] for c in st.calls if c[0]=="act"]
    print("%-30s 返回=%-5s 拍=%-3d 动作数=%-3d %s" % (name, ok, st.n, len(acts), acts[:6]))
    return st

print("分界: Q1/Q2=%.0f  Q3/Q4=%.0f  死区=[%.1f,%.1f]" % (pb.quarter_1_px, pb.quarter_4_px, pb.align_left_px, pb.align_right_px))
print()
run("Q1 最左 300",  center_px=300.0,  width_px=200.0)
run("Q2 中左 900",  center_px=900.0,  width_px=200.0)
run("Q3 中右 1600", center_px=1600.0, width_px=200.0)
run("Q4 最右 2300", center_px=2300.0, width_px=200.0)
run("正中+够近",     center_px=1283.0, width_px=400.0)
run("正中+太远",     center_px=1283.0, width_px=200.0)
