import sys, time
sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import levels.press_button as pb
PX = pb.px_per_deg
class R:
    def __init__(s,t,c): s.tag_id, s.corners = t, c
class FakeState:
    """假真机，带真机时间成本：拍照 0.65s，动作 1.0s，转头 0.5s"""
    def __init__(s, bearing, width=180.0):
        s.bearing_abs, s.width, s.n, s.t = bearing, width, 0, 0.0
        s.body, s.head, s.calls, s.current_head_pulse = 0.0, 0.0, [], None
    def capture_image(s): s.n += 1; s.t += 0.65; return "/tmp/f.jpg"
    def detect_apriltag(s, p):
        rel = s.bearing_abs - s.body - s.head
        if abs(rel) > 33.68: return []
        cu = pb.image_center_x + rel*PX
        w, cy = s.width, 972.0
        return [R(102, np.array([[cu-w/2,cy-w/2],[cu+w/2,cy-w/2],
                                 [cu+w/2,cy+w/2],[cu-w/2,cy+w/2]]))]
    def set_head(s, pulse, move_time_ms=500):
        s.calls.append(("head", pulse)); s.t += 0.5; s.head = (pulse-1500)*0.09
    def act(s, name, times=1):
        s.calls.append(("act", name, times)); s.t += 1.0*times
        if name == "turn_right":              s.body += pb.TURN_R_DEG*times
        elif name == "turn_left":             s.body -= pb.TURN_L_DEG*times
        elif name == "turn_right_small_step": s.body += pb.TURN_R_SMALL_DEG*times
        elif name == "turn_left_small_step":  s.body -= pb.TURN_L_SMALL_DEG*times
        elif name == "go_forward":            s.width *= 1.25

print("方位扫描（真实时间成本；上限 120 秒等效）:", flush=True)
print("%8s %7s %7s %8s %9s  %s" % ("方位","结果","拍数","等效秒","累计右转","动作摘要"), flush=True)
bad=[]
for b in (-180,-150,-120,-80,-45,-33,0,33,45,80,120,150,180):
    st = FakeState(b)
    tick = [0.0]
    # 用真实时间成本限制：每步 +0.65s，超过 120s 就认定"现场不可接受"
    def guarded():
        r = pb.press_button(st, 102)
        return r
    t0 = time.time()
    import threading
    box={}
    th = threading.Thread(target=lambda: box.update(r=guarded()), daemon=True); th.start()
    th.join(4.0)
    stuck = th.is_alive()
    acts=[]
    for c in st.calls:
        if c[0]=="act": acts.append(("%s×%d"%(c[1][:11],c[2])) if c[2]!=1 else c[1][:13])
    print("%8d %7s %7d %8.1f %8.1f°  %s%s" % (
        b, str(box.get("r")), st.n, st.t, st.body, "★超时 " if stuck else "", acts[:4]), flush=True)
    if box.get("r") is not True and not stuck: bad.append(b)
print(flush=True)
print("未收敛方位:", bad if bad else "无", flush=True)
