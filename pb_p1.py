import sys; sys.path.insert(0, ".")
import numpy as np
import levels.press_button as pb

FX = 1944.903664123011
class R:
    def __init__(s,t,c): s.tag_id, s.corners = t, c
class FakeState:
    """脚本化真机：目标在画面某个横向位置，宽 w；转头/转向会改变位置。"""
    def __init__(s, center_px, width_px, appear_after=0, tag=102):
        s.center, s.width, s.n = center_px, width_px, 0
        s.appear_after, s.tag, s.calls = appear_after, tag, []
        s.current_head_pulse = None; s.head = 1500
    def capture_image(s): s.n += 1; return "/tmp/f.jpg"
    def detect_apriltag(s, p):
        if s.n <= s.appear_after: return []
        w, cy = s.width, 972.0
        c = np.array([[s.center-w/2, cy-w/2],[s.center+w/2, cy-w/2],
                      [s.center+w/2, cy+w/2],[s.center-w/2, cy+w/2]])
        return [R(s.tag, c)]
    def set_head(s, pulse, move_time_ms=500):
        s.calls.append(("head", pulse))
        if pulse == 1500: s.head = 1500
        else: s.head = "side"
    def act(s, name, times=1):
        s.calls.append(("act", name))
        if name == "turn_left":        s.center -= pb.TURN_L_DEG * 34
        elif name == "turn_right":     s.center += pb.TURN_R_DEG * 34
        elif name == "turn_left_small_step":  s.center -= pb.TURN_L_SMALL_DEG * 34
        elif name == "turn_right_small_step": s.center += pb.TURN_R_SMALL_DEG * 34
        elif name == "go_forward":     s.width *= 1.12

def show(name, st, ok):
    acts = [c[1] for c in st.calls if c[0]=="act"]
    print("%-34s 返回=%-5s 拍=%-3d 动作=%s" % (name, ok, st.n, acts[:8]))

print("分界常量: Q1/Q2=%.0f  Q3/Q4=%.0f  死区=[%.1f, %.1f]" % (
    pb.quarter_1_px, pb.quarter_4_px, pb.align_left_px, pb.align_right_px))
print()
# Q1 最左 → 大步左转
st = FakeState(300.0, 200.0); show("Q1 最左(300) → 大步左转", st, pb.press_button(st, 102))
# Q2 中左 → 小步左转
st = FakeState(900.0, 200.0); show("Q2 中左(900) → 小步左转", st, pb.press_button(st, 102))
# Q3 中右 → 小步右转
st = FakeState(1600.0, 200.0); show("Q3 中右(1600) → 小步右转", st, pb.press_button(st, 102))
# Q4 最右 → 大步右转
st = FakeState(2300.0, 200.0); show("Q4 最右(2300) → 大步右转", st, pb.press_button(st, 102))
print()
# 已对准且够近 → 直接 True
st = FakeState(1283.0, 400.0); show("正中(1283)+够近 → 直接到位", st, pb.press_button(st, 102))
# 对准但太远 → 直行
st = FakeState(1283.0, 200.0); show("正中+太远 → 直行靠近", st, pb.press_button(st, 102))
# 看不到 → 左右交替转头找
st = FakeState(1283.0, 400.0, appear_after=4); show("前4帧看不到 → 交替转头", st, pb.press_button(st, 102))
print("  头部动作序列:", [c[1] for c in st.calls if c[0]=="head"][:6])
# 永远看不到 → False（不死循环）
st = FakeState(0,0, appear_after=10**9); show("永远看不到 → False", st, pb.press_button(st, 102))
print()
print("边界值检查:")
for c in (647.9, 648.1, 1943.9, 1944.1, 1242.0, 1243.1, 1322.9, 1323.1):
    side = "Q1" if c < pb.quarter_1_px else "Q2" if c < pb.align_left_px else "死区" if c <= pb.align_right_px else "Q3" if c <= pb.quarter_4_px else "Q4"
    print("  中心 %.1f → %s" % (c, side))
