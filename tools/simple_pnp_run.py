#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simple_pnp_run.py —— 独立基线程序：最简单的 PnP 定位 + 现行导航决策

设计约束（2026-08-30 应用户要求建立）：
  1. 忽略现有全部定位算法——无门控、无帧救援、无联合求解、无宽扫、
     无头转系数 k/轴倾斜修正。定位 = 拍照 → 检测已知 AprilTag →
     全部角点合并 → cv2.solvePnP 一次求解 → 头部角度补偿（标称脉宽换算）。
  2. 平移/转向/避障/危险区/批量直行/转弯批量化/路点表/bypass 全部
     沿用 2026-08-30 现行逻辑（快照复制，含 OBSTACLE=13、STOP_TIME=0、
     路点6 bypass x≥72）。
  3. 完全独立：不 import 本项目任何模块、不改动任何现有文件；
     硬件库（hiwonder/apriltag/cv2）与标准库除外。

用途：作为对照基线（定位退化版的整机行为、故障排查、演示兜底）。
注意：无门控意味着单标签平面歧义（镜像分支）不会被拦截，本文件
      的定位结果可能偶发大偏差——这是"最简单"的代价，属预期。

用法（机器人项目根目录）：
    python -m tools.simple_pnp_run
    python -m tools.simple_pnp_run --end-at-last-stop   # 到停靠点4 [74,30] 结束
"""

import sys
import time
import subprocess
from collections import namedtuple

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None
try:
    import apriltag
except Exception:
    apriltag = None
try:
    import hiwonder.ActionGroupControl as AGC
except Exception:
    AGC = None
try:
    import hiwonder.ros_robot_controller_sdk as rrc
    from hiwonder.Controller import Controller
    _board = rrc.Board()
    ctl = Controller(_board)
except Exception:
    ctl = None

# =====================================================================
# 常量（快照自 camera_config.py / levels/goodluck.py 2026-08-30 版）
# =====================================================================
CAMERA_INTRINSIC = np.array(
    ((1.944903664123011e03, 0, 1.283069051100245e03),
     (0, 1.950095436307893e03, 9.831983420212778e02),
     (0, 0, 1)), dtype=np.double)
CAMERA_DISTORTION = np.array([-0.384402275498781, 0.284681889150075, 0, 0])
CAMERA_WIDTH, CAMERA_HEIGHT = 2592, 1944

HEAD_CENTER, HEAD_RIGHT, HEAD_LEFT = 1500, 1050, 1950
HEAD_MOVE_TIME_MS, HEAD_MOVE_TIME_MIN_MS = 500, 100
SERVO_DEG_PER_US = 0.09
PITCH_UP_PULSE = 1800
TAG_CORNER_PERM = np.array([0, 1, 2, 3], dtype=np.int64)  # 角点顺序已验证为恒等映射
MAX_LOCATE_RETRIES = 5

# =====================================================================
# 赛道数据（快照自 levels/goodluck.py 2026-08-30：survey 精化坐标）
# =====================================================================
tag_poses = {}
tag_poses["151"] = np.array([[45.01, 16.12, 31.40], [45.01, 11.12, 31.49], [45.01, 11.04, 26.49], [45.01, 16.04, 26.40]],dtype=np.float64,)
tag_poses["152"] = np.array([[45.01, 42.35, 30.84], [45.01, 37.35, 31.01], [45.01, 37.17, 26.02], [45.01, 42.17, 25.84]],dtype=np.float64,)
tag_poses["153"] = np.array([[18.19, 101.38, 19.77], [23.17, 101.38, 20.22], [23.62, 101.38, 15.24], [18.64, 101.38, 14.79]],dtype=np.float64,)
tag_poses["154"] = np.array([[18.06, 101.38, 32.04], [23.06, 101.38, 32.05], [23.07, 101.38, 27.05], [18.07, 101.38, 27.04]],dtype=np.float64,)
tag_poses["155"] = np.array([[73.14, 101.38, 28.13], [78.14, 101.38, 28.14], [78.16, 101.38, 23.14], [73.16, 101.38, 23.13]],dtype=np.float64,)
tag_poses["156"] = np.array([[95.00, 82.57, 27.41], [95.00, 77.57, 27.54], [95.00, 77.43, 22.54], [95.00, 82.43, 22.41]],dtype=np.float64,)
tag_poses["157"] = np.array([[95.00, 60.44, 28.09], [95.00, 55.47, 28.62], [95.00, 54.94, 23.65], [95.00, 59.91, 23.12]],dtype=np.float64,)
tag_poses["158"] = np.array([[55.23, 16.49, 29.62], [55.23, 21.49, 29.67], [55.23, 21.54, 24.67], [55.23, 16.54, 24.62]],dtype=np.float64,)
tag_poses["159"] = np.array([[27.98, 22.66, 0.00], [27.84, 17.67, 0.00], [22.84, 17.81, 0.00], [22.99, 22.81, 0.00]],dtype=np.float64,)
tag_poses["160"] = np.array([[79.57, 15.48, 0.00], [74.57, 15.50, 0.00], [74.58, 20.50, 0.00], [79.58, 20.48, 0.00]],dtype=np.float64,)
tag_poses["161"] = np.array([[82.36, 0.00, 27.93], [77.37, 0.00, 27.65], [77.64, 0.00, 22.66], [82.63, 0.00, 22.93]],dtype=np.float64,)
tag_poses["162"] = np.array([[22.51, 78.43, 0.00], [27.51, 78.50, 0.00], [27.58, 73.50, 0.00], [22.58, 73.43, 0.00]],dtype=np.float64,)
tag_poses["163"] = np.array([[77.93, 78.45, 0.00], [77.65, 73.46, 0.00], [72.66, 73.75, 0.00], [72.94, 78.74, 0.00]],dtype=np.float64,)

WALLS = [
    [0, 5, 40, 100],    # 左墙
    [45, 55, 0, 60],    # 中墙
    [95, 100, 40, 100], # 右墙
]

# =====================================================================
# 决策常量（快照：含 2026-08-30 提速改造与用户手调 OBSTACLE=13）
# =====================================================================
ORIENTATION_THRESHOLD = 0.26   # 朝向差异模长阈值，约15°
POSITION_THRESHOLD = 3.0       # 位置差异模长阈值，cm
STOP_TIME = 0                  # 2026-08-30 取消停靠（原 3 秒）
OBSTACLE_THRESHOLD = 13.0      # 避障阈值 = 机身半宽（用户 2026-08-30 手调）
SAFE_MARGIN_CM = 3.0
CORRIDOR_CLEAR_CM = OBSTACLE_THRESHOLD + 3.0
ORIENT_FREEZE_DIST_CM = 10.0

BATCH_FORWARD_MAX_STEPS = 6
BATCH_FORWARD_MIN_WALL_DIST = 18.0
BATCH_FORWARD_MIN_DIST = 12.0
GO_FORWARD_BATCH_MAX_ANGLE_DEG = 3.0

FORWARD_CM = 5.0
FORWARD_ONE_STEP_CM = 2.0
BACK_FAST_CM = 3.2
LEFT_MOVE_CM = 1.9
RIGHT_MOVE_CM = 2.2
TURN_LEFT_DEG = 22.0
TURN_RIGHT_DEG = 25.7
FORWARD_BIAS = 0.0

Waypoint = namedtuple("Waypoint", ["pos", "stop", "orientation", "bypass_position", "bypass_condition"])

END_AT_LAST_STOP = "--end-at-last-stop" in sys.argv
END_AFTER_POS = [74.0, 30.0]

ROUTE = [
    Waypoint([14.7, 21.3], STOP_TIME, None, None, None),   # 停靠点1
    Waypoint([23.1, 30.0], 0.0, None, 30.5, "y+"),         # 中间路点①
    Waypoint([23.1, 70.0], STOP_TIME, None, None, None),   # 停靠点2
    Waypoint([40.0, 80.0], 0.0, None, 41.0, "x+"),         # 中间路点②
    Waypoint([65.0, 79.7], STOP_TIME, None, None, None),   # 停靠点3
    Waypoint([74.0, 75.0], 0.0, None, 72.0, "x+"),         # 中间路点③：先东移 x≥72 再下行
    Waypoint([74.0, 30.0], STOP_TIME, None, None, None),   # 停靠点4（--end-at-last-stop 结束点）
    Waypoint([82.0, 20.0], 0.0, None, 82.0, "x+"),         # 中间路点④
    Waypoint([100.0, 20.0], STOP_TIME, None, None, None),  # 停靠点5（出口）
]


# =====================================================================
# 纯几何工具（快照）
# =====================================================================

def distance_point_to_rect(pos, rect):
    x, y = pos[0], pos[1]
    x_min, x_max, y_min, y_max = rect
    dx = max(x_min - x, 0, x - x_max)
    dy = max(y_min - y, 0, y - y_max)
    return float(np.sqrt(dx * dx + dy * dy))


def distance_to_walls(pos):
    pos = np.array(pos, dtype=np.float64)
    min_dist = float("inf")
    for rect in WALLS:
        min_dist = min(min_dist, distance_point_to_rect(pos, rect))
    min_dist = min(min_dist, float(pos[1]))        # 外框底边 y=0
    min_dist = min(min_dist, float(100 - pos[1]))  # 外框顶边 y=100
    return min_dist


def nearest_safe_point(pos):
    SAFE_THRESHOLD = OBSTACLE_THRESHOLD + POSITION_THRESHOLD + SAFE_MARGIN_CM
    pos = np.array(pos, dtype=np.float64)
    if distance_to_walls(pos) >= SAFE_THRESHOLD:
        return pos
    best_point, best_dist = pos.copy(), distance_to_walls(pos)
    for radius in range(1, 30):
        for k in range(16):
            a = 2 * np.pi * k / 16
            step = radius / 3.0
            candidate = pos + step * np.array([np.cos(a), np.sin(a)])
            d = distance_to_walls(candidate)
            if d >= SAFE_THRESHOLD:
                return candidate
            if d > best_dist:
                best_dist = d
                best_point = candidate.copy()
    print(f"nearest_safe_point: 未找到满足阈值的安全点，返回最优点 {best_point} (距离 {best_dist:.2f}cm)")
    return best_point


def check_segment_clear(start_pos, direction, steps, step_cm, min_dist):
    start_pos = np.asarray(start_pos, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    for k in range(1, steps + 1):
        sample = start_pos + k * step_cm * direction
        if distance_to_walls(sample) < min_dist:
            print(f"  [批量预检] 第{k}步位置 {sample} 离墙 {distance_to_walls(sample):.2f}cm < {min_dist}cm，不可批量")
            return False
    return True


def assert_corridor_clear(polyline, min_dist=CORRIDOR_CLEAR_CM):
    pts = [np.asarray(p, dtype=np.float64) for p in polyline]
    ok = True
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        seg_len = np.linalg.norm(b - a)
        n = max(int(seg_len), 2)
        for k in range(n + 1):
            sample = a + (b - a) * (k / n)
            if distance_to_walls(sample) < min_dist:
                print(f"  [走廊警告] 段 {a}→{b} 采样 {sample} 离墙 {distance_to_walls(sample):.2f}cm < {min_dist}cm")
                ok = False
    print(f"  [走廊校验] {'通过，全部净空 >= %.1fcm' % min_dist if ok else '存在贴墙段，请调整路点'}")
    return ok


# =====================================================================
# 机器人状态：硬件 I/O + 最简 PnP 定位
# =====================================================================

class SimplePnPState:
    """硬件状态与最简定位：三档头扫（回正→右→左），每档一次 cv2.solvePnP。

    刻意不做的事（与主程序的差异）：无合理性门控、无逐标签剔除、
    无多帧联合、无宽扫档、无头转系数 k/轴倾斜修正（用标称脉宽换算）。
    """

    def __init__(self, poses):
        self.tag_poses = poses
        self.current_position = None
        self.current_orientation = None
        self.current_head_pulse = HEAD_CENTER

    # ---- 硬件 ----
    def run_action(self, name, times=1):
        AGC.runActionGroup(name, times=times)

    def set_head(self, pulse, move_time_ms=HEAD_MOVE_TIME_MS):
        if pulse == self.current_head_pulse:
            return
        delta = abs(pulse - self.current_head_pulse)
        dynamic_time = max(HEAD_MOVE_TIME_MIN_MS, int(move_time_ms * delta / 900))
        ctl.set_pwm_servo_pulse(2, pulse, dynamic_time)
        time.sleep(dynamic_time / 1000.0 + 0.2)
        self.current_head_pulse = pulse

    def raise_head(self):
        ctl.set_pwm_servo_pulse(1, PITCH_UP_PULSE, 500)

    def pulse_to_angle(self, pulse):
        return (pulse - HEAD_CENTER) * SERVO_DEG_PER_US

    def capture_image(self):
        timestamp = int(time.time())
        filename = f"/home/pi/Pictures/photo_{timestamp}.jpg"
        cmd = (f"fswebcam -r {CAMERA_WIDTH}x{CAMERA_HEIGHT} "
               f"--no-banner -S 3 {filename}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"拍照失败: {result.stderr}")
            return None
        return filename

    # ---- 最简单帧定位：拍照 → 检测 → solvePnP，无任何门控 ----
    def locate_pose(self, head_pulse):
        """一档拍照+PnP。成功写入 current_position/orientation 并返回 True。"""
        self.set_head(head_pulse)
        filename = self.capture_image()
        if filename is None:
            return False
        image = cv2.imread(filename)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        options = apriltag.DetectorOptions(families="tag36h11")
        results = apriltag.Detector(options).detect(gray)

        objlist, imglist = [], []
        for r in results:
            tid = str(r.tag_id)
            if tid not in self.tag_poses:
                continue
            print("[INFO] Detected AprilTag ID:", tid)
            objlist.extend(self.tag_poses[tid])
            imglist.extend(np.asarray(r.corners, dtype=np.float64)[TAG_CORNER_PERM])

        if len(objlist) < 4:
            print("未检测到已知标签，本档定位失败。")
            return False

        ok, rvec, tvec = cv2.solvePnP(
            np.asarray(objlist), np.asarray(imglist),
            CAMERA_INTRINSIC, CAMERA_DISTORTION)
        if not ok:
            print("solvePnP 失败。")
            return False
        R = cv2.Rodrigues(rvec)[0]
        pos_3d = (-np.linalg.inv(R) @ tvec).flatten()
        ori_3d = (np.linalg.inv(R) @ (np.array([[0.0], [0.0], [1.0]]) - tvec)).flatten() - pos_3d
        ori_3d /= np.linalg.norm(ori_3d)

        self.current_position = pos_3d[:2]
        self.current_orientation = ori_3d[:2]
        norm = np.linalg.norm(self.current_orientation)
        if norm != 0:
            self.current_orientation /= norm
        # 头部角度补偿：转头档位成功时把相机朝向旋回机体朝向（标称换算）
        if head_pulse != HEAD_CENTER:
            theta = np.radians(self.pulse_to_angle(head_pulse))
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            self.current_orientation = np.array(
                [[cos_t, sin_t], [-sin_t, cos_t]]) @ self.current_orientation
            n = np.linalg.norm(self.current_orientation)
            if n != 0:
                self.current_orientation /= n
        return True

    def locate_with_scan(self):
        """三档头扫：回正→右转→左转，任一档 PnP 成功即返回。"""
        self.current_position = None
        self.current_orientation = None
        for pulse, name in ((HEAD_CENTER, "回正"), (HEAD_RIGHT, "右转"), (HEAD_LEFT, "左转")):
            if self.locate_pose(pulse):
                print(f"定位成功（头部{name}）。位置：", self.current_position,
                      "朝向：", self.current_orientation)
                self.set_head(HEAD_CENTER)
                return True
        self.set_head(HEAD_CENTER)
        print("三档扫描均失败。")
        return False

    def locate_with_retry(self):
        """扫描失败 → 身体同向连转重试（沿用现行策略）。"""
        for attempt in range(MAX_LOCATE_RETRIES):
            print(f"--- 定位尝试 {attempt + 1}/{MAX_LOCATE_RETRIES} ---")
            if self.locate_with_scan():
                return True
            print("头部扫描失败，身体左转尝试重新定位。")
            self.run_action("turn_left")
        print("定位重试超限，程序终止。")
        return False


# =====================================================================
# 导航决策（快照自 levels/goodluck.py 2026-08-30 现行版）
# =====================================================================

def decide_panning_action(state, current_pos, target_pos, orientation_xOy):
    """贪心平移策略：模拟各方向动作后的预期位置，选最接近目标的。"""
    position_diff = np.array(current_pos) - np.array(target_pos)
    oy = orientation_xOy
    left_dir = np.array([-oy[1], oy[0]])
    right_dir = np.array([oy[1], -oy[0]])

    candidates = {
        "go_forward": position_diff + FORWARD_CM * oy,
        "go_forward_one_step": position_diff + FORWARD_ONE_STEP_CM * oy,
        "back_one_step": position_diff - BACK_FAST_CM * oy,
        "left_move": position_diff + LEFT_MOVE_CM * left_dir,
        "right_move": position_diff + RIGHT_MOVE_CM * right_dir,
    }

    best_action, best_score = None, float("inf")
    for action, new_pd in candidates.items():
        new_pos = np.array(target_pos) + new_pd
        wall_dist = distance_to_walls(new_pos)
        if wall_dist < OBSTACLE_THRESHOLD:
            print(f"  {action}: 执行后位置 {new_pos} 离墙 {wall_dist:.2f}cm < {OBSTACLE_THRESHOLD}cm，排除")
            continue
        score = np.linalg.norm(new_pd)
        if action.startswith("go_forward"):
            score -= FORWARD_BIAS
        print(f"  {action}: 执行后距离 {np.linalg.norm(new_pd):.2f}cm 离墙 {wall_dist:.2f}cm (score={score:.2f})")
        if score < best_score:
            best_score = score
            best_action = action

    if best_action is None:
        print("所有平移动作均被避障排除，机器人可能已在危险区。")
        return None
    print("最佳平移动作：", best_action)
    return best_action


def decide_rotation_action(state, target):
    """转向策略：批量连转 times = round(需要角/步长)，上限 3（现行版）。"""
    target = np.array(target, dtype=np.float64)
    cross = state.current_orientation[0] * target[1] - state.current_orientation[1] * target[0]
    dot = np.clip(np.dot(state.current_orientation, target), -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(dot))
    if cross > 0:
        step_deg, name = TURN_LEFT_DEG, "turn_left"
    else:
        step_deg, name = TURN_RIGHT_DEG, "turn_right"
    times = max(1, min(3, int(round(angle_deg / step_deg))))
    print(f"需转向 {angle_deg:.1f}°（{name}，实测 {step_deg}°/次 × {times}）")
    return name, times


def navigate_to_target(state, target_pos, target_orientation=None, stop_time=0.0,
                       bypass_position=None, bypass_condition=None):
    """统一路点导航原语（现行版逻辑：危险检测/动态朝向/批量直行/bypass）。"""
    target_pos = np.asarray(target_pos, dtype=np.float64)
    if target_orientation is not None:
        target_orientation = np.asarray(target_orientation, dtype=np.float64)
        target_orientation = target_orientation / np.linalg.norm(target_orientation)

    phase = "停靠点" if stop_time > 0 else ("转向点" if target_orientation is not None else "中间路点")
    print(f"\n===== 开始导航至{phase}：{target_pos} =====")

    frozen_orient = None
    while True:
        print("\n=== 新一轮导航循环开始 ===")

        if not state.locate_with_retry():
            print(f"无法定位，导航至{phase} {target_pos} 失败。")
            return False
        print("当前机体位置：", state.current_position)
        print("当前机体朝向：", state.current_orientation)

        # bypass 跳过
        if bypass_condition is not None and bypass_position is not None:
            if (bypass_condition == "x+" and state.current_position[0] > bypass_position) or \
               (bypass_condition == "x-" and state.current_position[0] < bypass_position) or \
               (bypass_condition == "y+" and state.current_position[1] > bypass_position) or \
               (bypass_condition == "y-" and state.current_position[1] < bypass_position):
                print(f"满足跳过条件 {bypass_condition} > {bypass_position}, 跳过该路点导航。")
                return True

        # 危险检测
        wall_dist = distance_to_walls(state.current_position)
        if wall_dist < OBSTACLE_THRESHOLD:
            target = nearest_safe_point(state.current_position)
            escaping = True
            print(f"机器人处于危险区（离墙 {wall_dist:.2f}cm < {OBSTACLE_THRESHOLD}cm），"
                  f"临时导航至安全点 {target}")
        else:
            target = target_pos
            escaping = False

        pd = np.array(target) - np.array(state.current_position)
        dist = np.linalg.norm(pd)
        print("位置差异pd：", pd)

        # 动态朝向
        if escaping:
            effective_orient, orient_source = None, "无（逃离）"
        elif dist > ORIENT_FREEZE_DIST_CM:
            effective_orient = pd / dist
            frozen_orient = effective_orient.copy()
            orient_source = "连线方向"
        else:
            if target_orientation is not None:
                effective_orient, orient_source = target_orientation, "指定朝向"
            else:
                if frozen_orient is None:
                    frozen_orient = pd / dist if dist > 0 else state.current_orientation.copy()
                    if np.dot(pd, state.current_orientation) < 0:
                        frozen_orient = -frozen_orient
                    print("  冻结方向：", frozen_orient)
                effective_orient, orient_source = frozen_orient, "冻结方向"

        od = None
        if effective_orient is not None:
            od = effective_orient - np.array(state.current_orientation)
            print(f"目标朝向（{orient_source}）：{effective_orient}  朝向差异od：{od}")

        # 朝向修正
        if not escaping and od is not None and np.linalg.norm(od) > ORIENTATION_THRESHOLD:
            action, times = decide_rotation_action(state, effective_orient)
            state.run_action(action, times)
            continue

        # 平移接近
        if dist > POSITION_THRESHOLD:
            action = decide_panning_action(state, state.current_position, target, state.current_orientation)
            if action is None:
                print("警告：无安全平移动作可选，按兵不动。")
            else:
                if (action in ("go_forward", "go_forward_one_step")
                        and not escaping
                        and wall_dist >= BATCH_FORWARD_MIN_WALL_DIST
                        and dist > BATCH_FORWARD_MIN_DIST
                        and od is not None
                        and np.linalg.norm(od) <= ORIENTATION_THRESHOLD):
                    angle_deg = np.degrees(2 * np.arcsin(min(np.linalg.norm(od) / 2, 1.0)))
                    if angle_deg <= GO_FORWARD_BATCH_MAX_ANGLE_DEG:
                        batch_steps = min(BATCH_FORWARD_MAX_STEPS, int(dist / FORWARD_CM))
                        if batch_steps > 1 and check_segment_clear(
                                state.current_position, state.current_orientation,
                                batch_steps, FORWARD_CM, BATCH_FORWARD_MIN_WALL_DIST):
                            print(f"  ★ 批量直行 {batch_steps} 步（go_forward，距目标 {dist:.1f}cm，离墙 {wall_dist:.1f}cm）")
                            state.run_action("go_forward", times=batch_steps)
                            continue
                state.run_action(action)
            continue

        print("\n=== 本轮导航循环结束 ===")
        if escaping:
            print(f"===== 已到达安全点 {target}，恢复原目标导航 =====")
            continue
        if stop_time > 0:
            print(f"===== 到达{phase} {target_pos}，停留 {stop_time} 秒 =====")
            state.run_action("stand")
            time.sleep(stop_time)
        else:
            print(f"===== 到达{phase} {target_pos}，不停留 =====")
        return True


# =====================================================================
# 主流程
# =====================================================================

def main():
    print("=" * 60)
    print("simple_pnp_run —— 最简 PnP 基线程序（独立运行，不影响主程序）")
    print(f"end-at-last-stop: {END_AT_LAST_STOP}")
    print("=" * 60)
    if cv2 is None or apriltag is None or AGC is None or ctl is None:
        print("错误：缺少硬件/视觉库（cv2 / apriltag / hiwonder），请在机器人上运行。")
        sys.exit(1)

    assert_corridor_clear([wp.pos for wp in ROUTE])
    state = SimplePnPState(tag_poses)
    state.run_action("stand")
    state.set_head(HEAD_CENTER)

    for i, wp in enumerate(ROUTE, 1):
        if not navigate_to_target(state, wp.pos, wp.orientation, wp.stop,
                                  wp.bypass_position, wp.bypass_condition):
            print(f"导航至路点 {i}（{wp.pos}）失败，程序终止。")
            return
        print(f"已{'到达停靠点' if wp.stop > 0 else '通过路点'} {i}：{wp.pos}")
        if END_AT_LAST_STOP and wp.pos == END_AFTER_POS:
            print("===== --end-at-last-stop 生效：到达最后停靠点，提前结束 =====")
            return
    print("===== 全程完成 =====")


if __name__ == "__main__":
    main()
