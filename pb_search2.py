import sys, time
sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import levels.press_button as pb
PX = pb.px_per_deg

class R:
    def __init__(s,t,c): s.tag_id, s.corners = t, c
class FakeState:
    """假真机：rel = 标签方位 - 身体朝向 - 头偏角；|rel|>33.68° 即看不到。
    符号约定：身体右转 → 朝向角增大 → 原方位的标签 rel 变小（往画面左移）。"""
    def __init__(s, tag_bearing_deg, width_px=180.0):
        s.bearing, s.width, s.n = tag_bearing_deg, width_px, 0
        s.body, s.head, s.calls, s.current_head_pulse = 0.0, 0.0, [], None
    def capture_image(s): s.n += 1; return "/tmp/f.jpg"
    def detect_apriltag(s, p):
        rel = s.bearing - s.body - s.head
        if abs(rel) > 33.68: return []
        cu = pb.image_center_x + rel*PX
        w, cy = s.width, 972.0
        return [R(102, np.array([[cu-w/2,cy-w/2],[cu+w/2,cy-w/2],
                                 [cu+w/2,cy+w/2],[cu-w/2,cy+w/2]]))]
    def set_head(s, pulse, move_time_ms=500):
        s.calls.append(("head", pulse)); s.head = (pulse-1500)*0.09
    def act(s, name, times=1):
        s.calls.append(("act", name, times))
        if name == "turn_left":               s.body -= pb.TURN_L_DEG*times
        elif name == "turn_right":            s.body += pb.TURN_R_DEG*times
        elif name == "turn_left_small_step":  s.body -= pb.TURN_L_SMALL_DEG*times
        elif name == "turn_right_small_step": s.body += pb.TURN_R_SMALL_DEG*times

print("A. 标签在视野内（|方位|<=33°）→ 直接对准到位", flush=True)
for b in (-30,-10,0,10,30):
    st = FakeState(b)
    print("  方位%4d° → 返回=%s 拍=%d" % (b, pb.press_button(st,102), st.n), flush=True)
print(flush=True)
print("B. 标签在视野外 → 转头找 → 不行身体转两步 → 最终找到", flush=True)
for b in (45, 80, 120, 180, -45, -80, -120, -180):
    st = FakeState(b)
    r = pb.press_button(st,102)
    two = [(c[1],c[2]) for c in st.calls if c[0]=="act" and c[2]==2]
    print("  方位%5d° → 返回=%s 拍=%-4d 身体两步转序列=%s" % (b, r, st.n, two[:4]), flush=True)
print(flush=True)
print("C. 扫描实况（标签在 +120°）前 14 个调用:", flush=True)
st = FakeState(120)
import threading
threading.Thread(target=lambda: pb.press_button(st,102), daemon=True).start()
time.sleep(0.5)
seq = [(c[0], c[1]) if c[0]=="head" else (c[1], c[2]) for c in st.calls[:14]]
print("  ", seq, flush=True)
