import sys, time
sys.path.insert(0, "/home/pi/Robot_Competition")
import levels.press_button as pb
import core.robot_core as rc

st = rc.RobotState(tag_poses={})
pb.init_head(st)
print("START")
ok = 0
for i in range(6):
    tags = pb.detect_tags(st)
    if tags:
        ok += 1
        print("shot%d ids=%s" % (i + 1, sorted(tags)))
    else:
        print("shot%d NONE" % (i + 1))
    time.sleep(0.2)
print("END ok=%d/6" % ok)
