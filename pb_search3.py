import sys, time
sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import levels.press_button as pb
PX = pb.px_per_deg

print("=== 先定符号：left_move / turn_left 后，目标在画面上的位置应该往哪边移？ ===", flush=True)
print("相机系 x 向右、u = fx*X/Z + cx0；标签在机器人右侧 ⇒ X>0 ⇒ u > cx0", flush=True)
print("要把它转到画面中心 ⇒ 必须右转（把机器人朝向转向标签）", flush=True)
print("⇒ 因此：标签在画面右侧(u>cx0) → 下发 turn_right；左侧 → turn_left", flush=True)
print(flush=True)
print("turn_for_offset 的实际输出:", flush=True)
for off in (-1000, -400, 0, 400, 1000):
    print("  画面偏差 %6d → %s" % (off, pb.turn_for_offset(off)), flush=True)
print(flush=True)

class R:
    def __init__(s,t,c): s.tag_id, s.corners = t, c
class FakeState:
    """假真机（符号按上面结论）：turn_right 让机器人朝向转向右侧标签 ⇒ 标签在画面上左移"""
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
        # 右转 = 朝向转向右侧 ⇒ bearing 相对朝向变小 ⇒ 标签在画面上左移
        if name == "turn_left":               s.body -= pb.TURN_L_DEG*times
        elif name == "turn_right":            s.body += pb.TURN_R_DEG*times
        elif name == "turn_left_small_step":  s.body -= pb.TURN_L_SMALL_DEG*times
        elif name == "turn_right_small_step": s.body += pb.TURN_R_SMALL_DEG*times

print("A. 标签在视野内 → 直接对准到位:", flush=True)
for b in (-30,-10,0,10,30):
    st = FakeState(b)
    print("  方位%4d° → 返回=%s 拍=%d 动作=%s" % (b, pb.press_button(st,102), st.n,
        [c[1] for c in st.calls if c[0]=="act"][:3]), flush=True)
print(flush=True)
print("B. 标签在视野外 → 转头找 → 身体转两步换方位:", flush=True)
for b in (45, 80, 120, 180, -45, -80, -120, -180):
    st = FakeState(b)
    r = pb.press_button(st,102)
    two = [(c[1],c[2]) for c in st.calls if c[0]=="act" and c[2]==2]
    print("  方位%5d° → 返回=%s 拍=%-4d 身体两步转=%s" % (b, r, st.n, two[:4]), flush=True)
