import sys
sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import levels.press_button as pb

class R:
    def __init__(s,t,c): s.tag_id, s.corners = t, c

class FakeState:
    def __init__(s, center_px, width_px, appear_after=0, grow=1.6):
        s.center, s.width, s.n = center_px, width_px, 0
        s.appear_after, s.calls = appear_after, []
        s.grow = grow; s.current_head_pulse = None
    def capture_image(s): s.n += 1; return "/tmp/f.jpg"
    def detect_apriltag(s, p):
        if s.n <= s.appear_after: return []
        w, cy = s.width, 972.0
        return [R(102, np.array([[s.center-w/2,cy-w/2],[s.center+w/2,cy-w/2],
                                 [s.center+w/2,cy+w/2],[s.center-w/2,cy+w/2]]))]
    def set_head(s, pulse, move_time_ms=500): s.calls.append(("head", pulse))
    def act(s, name, times=1):
        s.calls.append(("act", name))
        # 假真机：转向按"名义角度"改变标签在画面上的横向位置（600px ≈ 40° 视场）
        dpp = 600.0 / 40.5
        if name == "turn_left":               s.center -= pb.TURN_L_DEG * dpp
        elif name == "turn_right":            s.center += pb.TURN_R_DEG * dpp
        elif name == "turn_left_small_step":  s.center -= pb.TURN_L_SMALL_DEG * dpp
        elif name == "turn_right_small_step": s.center += pb.TURN_R_SMALL_DEG * dpp
        elif name == "go_forward":            s.width *= s.grow

def run(name, **kw):
    st = FakeState(**kw)
    ok = pb.press_button(st, 102)
    acts = [c[1] for c in st.calls if c[0]=="act"]
    heads = [c[1] for c in st.calls if c[0]=="head"]
    print("%-22s 返回=%-5s 拍=%-3d 动作=%s" % (name, ok, st.n, acts[:7]), flush=True)
    if len(heads) > 2: print("      头部: %s" % heads[:6], flush=True)

print("分界: Q1/Q2=%.0f Q3/Q4=%.0f 死区=[%.1f,%.1f]" % (pb.quarter_1_px, pb.quarter_4_px, pb.align_left_px, pb.align_right_px), flush=True)
for label, kw in (
    ("Q1 最左 300",  dict(center_px=300.0,  width_px=200.0)),
    ("Q2 中左 900",  dict(center_px=900.0,  width_px=200.0)),
    ("Q3 中右 1600", dict(center_px=1600.0, width_px=200.0)),
    ("Q4 最右 2300", dict(center_px=2300.0, width_px=200.0)),
    ("正中+够近",     dict(center_px=1283.0, width_px=400.0)),
    ("正中+太远",     dict(center_px=1283.0, width_px=200.0)),
    ("前4帧看不到",   dict(center_px=1283.0, width_px=400.0, appear_after=4)),
    ("永远看不到",    dict(center_px=0.0, width_px=0.0, appear_after=10**9)),
):
    run(label, **kw)
