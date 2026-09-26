import sys, time
sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import levels.press_button as pb
PX = pb.px_per_deg

class R:
    def __init__(s,t,c): s.tag_id, s.corners = t, c
class FakeState:
    def __init__(s, center_px, width_px, appear_after=0, grow=1.25, stop_grow_at=None):
        s.center, s.width, s.n, s.appear_after = center_px, width_px, 0, appear_after
        s.calls, s.grow, s.stop_grow_at = [], grow, stop_grow_at
        s.current_head_pulse = None
    def capture_image(s): s.n += 1; return "/tmp/f.jpg"
    def detect_apriltag(s, p):
        if s.n <= s.appear_after: return []
        if abs(s.center - pb.image_center_x) > 2592/2: return []
        w, cy = s.width, 972.0
        return [R(102, np.array([[s.center-w/2,cy-w/2],[s.center+w/2,cy-w/2],
                                 [s.center+w/2,cy+w/2],[s.center-w/2,cy+w/2]]))]
    def set_head(s, pulse, move_time_ms=500): s.calls.append(("head", pulse))
    def act(s, name, times=1):
        s.calls.append(("act", name))
        if name == "turn_left":               s.center += pb.TURN_L_DEG * PX
        elif name == "turn_right":            s.center -= pb.TURN_R_DEG * PX
        elif name == "turn_left_small_step":  s.center += pb.TURN_L_SMALL_DEG * PX
        elif name == "turn_right_small_step": s.center -= pb.TURN_R_SMALL_DEG * PX
        elif name == "go_forward":
            if s.stop_grow_at is None or s.width < s.stop_grow_at:
                s.width *= s.grow

print("=== A. 正常场景扫描（17 个起点，早停=够近即回）===", flush=True)
ok = 0; tot = 0; worst = 0
for off in (-1290,-1200,-1000,-800,-600,-400,-339,-100,0,100,339,400,600,800,1000,1200,1290):
    st = FakeState(pb.image_center_x+off, 180.0)
    good = pb.press_button(st, 102)
    nt = sum(1 for c in st.calls if c[0]=="act" and "turn" in c[1])
    ok += good; tot += 1; worst = max(worst, nt)
print("  收敛 %d/%d，单次最多转向 %d 步" % (ok, tot, worst), flush=True)
print(flush=True)

print("=== B. 边界：恰好卡在死区上 ===", flush=True)
for off in (338, 339, 340, 341):
    st = FakeState(pb.image_center_x+off, 400.0)
    print("  偏差%4d → %s" % (off, pb.press_button(st,102)), flush=True)
print(flush=True)

print("=== C. 搜索场景 ===", flush=True)
st = FakeState(pb.image_center_x, 400.0, appear_after=5)
print("  前5帧看不到 → 返回=%s, 头部动作=%s" % (
    pb.press_button(st,102), [c[1] for c in st.calls if c[0]=="head"][:7]), flush=True)
st = FakeState(0, 0, appear_after=10**9)
t0=time.time(); r = pb.press_button(st,102)
print("  永远看不到 → 返回=%s, 拍照=%d, 用时%.2fs" % (r, st.n, time.time()-t0), flush=True)
print(flush=True)

print("=== D. 危险场景：直行永远到不了 30cm（真机可能发生：卡住/地面打滑/够不着）===", flush=True)
st = FakeState(pb.image_center_x, 180.0, stop_grow_at=200.0)
t0 = time.time()
try:
    r = pb.press_button(st, 102)
    print("  返回=%s, 拍照=%d, 用时%.2fs" % (r, st.n, time.time()-t0), flush=True)
except KeyboardInterrupt:
    print("  被中断", flush=True)
