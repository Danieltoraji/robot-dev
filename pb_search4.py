import sys
sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import levels.press_button as pb
PX = pb.px_per_deg

# 用最朴素的方式打表：不动代码，手算"假设机器人右转 25.7° 后，标签相对朝向的方位角怎么变"
print("标签在机器人右前方 30°（bearing=+30）。机器人右转 25.7° 后，标签在哪？", flush=True)
print("  直觉：机器人脸转向右边 ⇒ 原本右前方的标签变得几乎正前方 ⇒ 相对方位应从 +30 变到 +4.3", flush=True)
print("  即 rel_new = rel_old - 25.7 = 30 - 25.7 = 4.3  → 相对方位【减小】", flush=True)
print(flush=True)
print("验算：turn_for_offset(+1018px) 给出 turn_right（向右转），若右转让 rel 减小，则收敛 ✓", flush=True)
print("     turn_for_offset(-1018px) 给出 turn_left（向左转），若左转让 rel 增大，则收敛 ✓", flush=True)
print(flush=True)

class R:
    def __init__(s,t,c): s.tag_id, s.corners = t, c
class FakeState:
    def __init__(s, bearing, width=180.0):
        s.bearing, s.width, s.n = bearing, width, 0
        s.rel, s.calls, s.current_head_pulse = bearing, [], None
    def capture_image(s): s.n += 1; return "/tmp/f.jpg"
    def detect_apriltag(s, p):
        if abs(s.rel) > 33.68: return []
        cu = pb.image_center_x + s.rel*PX
        w, cy = s.width, 972.0
        return [R(102, np.array([[cu-w/2,cy-w/2],[cu+w/2,cy-w/2],
                                 [cu+w/2,cy+w/2],[cu-w/2,cy+w/2]]))]
    def set_head(s, pulse, move_time_ms=500):
        s.calls.append(("head", pulse))
        # 头左转(+40.5°)= 视线转向机器人左侧 ⇒ 相对方位【增大】
        s.rel = s.bearing - (pulse-1500)*0.09 if False else s.rel
    def act(s, name, times=1):
        s.calls.append(("act", name, times))
        # ★ 右转 ⇒ 相对方位减小（朝向追向右侧标签）
        if name == "turn_right":            s.rel -= pb.TURN_R_DEG*times
        elif name == "turn_left":           s.rel += pb.TURN_L_DEG*times
        elif name == "turn_right_small_step": s.rel -= pb.TURN_R_SMALL_DEG*times
        elif name == "turn_left_small_step":  s.rel += pb.TURN_L_SMALL_DEG*times

print("A. 标签在视野内 → 对准到位:", flush=True)
for b in (-30,-10,0,10,30):
    st = FakeState(b)
    r = pb.press_button(st,102)
    acts = [c[1] for c in st.calls if c[0]=="act"]
    print("  方位%4d° → 返回=%s 拍=%d 动作=%s 结束rel=%.1f" % (b, r, st.n, acts[:3], st.rel), flush=True)
print(flush=True)
print("B. 视野外 → 转头找 → 身体转两步:", flush=True)
for b in (45, 80, 120, 180, -45, -80, -120, -180):
    st = FakeState(b)
    r = pb.press_button(st,102)
    two = [(c[1],c[2]) for c in st.calls if c[0]=="act" and c[2]==2]
    print("  方位%5d° → 返回=%s 拍=%-4d 身体两步转=%s" % (b, r, st.n, two[:4]), flush=True)
