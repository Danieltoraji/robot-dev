#!/usr/bin/env python3
# coding=utf-8
"""放下蓝色方块后「后退找线→右转进入线→循迹」独立调试脚本。

实际场景：白底红线，只有直线（无直角弯/横线）。
检测器用方案一 LineTrackerHSV（红色 hue 固定 + S/V OTSU 自适应阈值 + ROI
动态搜索 + 地面单应 cm 输出，line_color="dark" 适配白底红线）。流程:
  1. back_and_find_line(): 边后退边低头找红线，找到即停
  2. turn_right_to_enter_line(): 固定右转约 90°，把横向线转成纵向（follow）并对准
  3. track_line(): 沿线直行，丢线后退找回，步数上限兜底

用法(真机):
    /home/pi/jupyter-env/bin/python3 line_follow_debug.py [--dry-run]
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import cv2

sys.path.insert(0, "/home/pi/TonyPi")
sys.path.insert(0, "/home/pi/TonyPi/Functions")
sys.path.insert(0, "/home/pi/TonyPi/HiwonderSDK")

# 让脚本能从任意目录 import 到 vision 包内的 line_detector 模块
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_VISION_DIR = _PROJECT_ROOT / "vision"
for _extra in (_PROJECT_ROOT, _VISION_DIR):
    _extra_str = str(_extra)
    if _extra_str not in sys.path:
        sys.path.insert(0, _extra_str)

from camera_stream import CameraStream
from line_tracker_hsv import LineTrackerHSV

import hiwonder.ActionGroupControl as AGC
import hiwonder.ros_robot_controller_sdk as rrc
import hiwonder.yaml_handle as yaml_handle
from hiwonder.Controller import Controller


LOG_FILE = "/home/pi/JustForTestNoUse/BlueCube_test/line_follow_debug.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler(LOG_FILE, mode="a"), logging.StreamHandler()],
)
log = logging.getLogger("line_follow_debug")


# ---- 摄像头分辨率 ----
CAM_SIZE = (640, 480)

# 循迹俯仰脉宽（低头看线，与 levels/line_seeker_tracking.py 实机一致）
TRACK_PITCH_PULSE = 1050
# 相机光心高度（cm，与 core/ground_homography 解析自举缺省一致）
CAM_HEIGHT_CM = 39.0

# Head scanning.
HEAD_STEP_X = 15
HEAD_STEP_Y = 15

# 夹爪闭合程度: 1.0=最紧(8=0,16=1000), 0.5=半开, 0=全开(500)
GRIP_SCALE = 1.0
LOCK_SERVOS = {"6": 700, "7": 820, "8": int(500 - 500*GRIP_SCALE), "14": 300, "15": 180, "16": int(500 + 500*GRIP_SCALE)}

# =====================================================================
# 红色路径 HSV 范围 + LineDetector 配置（来自 line_seeker_tracking.py）
# =====================================================================

RED_LOW = (0, 0, 50)
RED_HIGH = (17, 255, 255)
RED_LOW_2 = (149, 0, 50)
RED_HIGH_2 = (180, 255, 255)

# =====================================================================
# 循迹决策阈值（工作分辨率 640px 宽下，与 line_seeker_tracking 一致）
# =====================================================================

T_CENTER = 12.0        # |lookahead_x| ≤ 此值 → 居中，可前进
T_BIG = 35.0           # |lookahead_x| > 此值 → 需转向修朝向
LOOKAHEAD_CLAMP = 200.0  # lookahead_x 绝对值钳制上限（防止多项式外推爆炸）

# 找线 / 进入线参数
ENTER_TURN_DEG = 90.0       # 右转进入线角度（横向线→纵向线，负=右转）
TRACK_MAX_BACKS = 3         # 循迹丢线时连续后退最大次数
ALIGN_MAX_STEPS = 20        # 对准阶段最大步数（防死循环）

# 转弯 / 循迹闭环参数
HEADING_BIG = 25.0          # |heading_deg| ≥ 此值 → 转向拉正
HEADING_SMALL = 8.0         # 小步精修阈值（_turn_by_angle 内部用）
FOLLOW_HEADING_MIN = 15.0   # follow 循迹：|heading| > 此值优先修朝向

# 动作标定值（来自 levels/goodluck.py 实机标定）
TURN_LEFT_DEG = 22.0
TURN_RIGHT_DEG = 25.7
TURN_LEFT_SMALL_DEG = TURN_LEFT_DEG / 2    # ≈ 11.0°
TURN_RIGHT_SMALL_DEG = TURN_RIGHT_DEG / 2  # ≈ 12.85°


class ActionRunner:
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.lock_servos = ""

    def run(self, name: str, times: int = 1, lock: bool = False, with_stand: bool = False):
        if not name or times <= 0:
            return
        lock_servos = self.lock_servos if lock else ""
        if self.dry_run:
            log.info("[dry-run] action=%s times=%s lock=%s with_stand=%s", name, times, bool(lock_servos), with_stand)
            time.sleep(0.05)
            return
        log.info("action=%s times=%s lock=%s with_stand=%s", name, times, bool(lock_servos), with_stand)
        # 防止runningAction卡True(上一动作异常中断时会发生),否则所有动作被runAction跳过
        if AGC.runningAction:
            log.warning("runningAction stuck before '%s', resetting", name)
            AGC.runningAction = False
        AGC.runActionGroup(name, times=times, lock_servos=lock_servos, with_stand=with_stand)
        if AGC.runningAction:
            log.warning("runningAction stuck after '%s', resetting", name)
            AGC.runningAction = False

    def lock_hands(self):
        self.lock_servos = LOCK_SERVOS
        log.info("hand servos locked")

    def unlock_hands(self):
        self.lock_servos = ""
        log.info("hand servos unlocked")


class LineFollowDebug:
    def __init__(self, args):
        self.args = args
        self.actions = ActionRunner(dry_run=args.dry_run)

        self.servo_data = yaml_handle.get_yaml_data(yaml_handle.servo_file_path)
        self.servo1 = int(self.servo_data["servo1"])
        self.servo2 = int(self.servo_data["servo2"])
        self.x_dis = self.servo2
        self.y_dis = self.servo1
        self.head_mode = "left_right"
        self.head_turn = "left_right"
        self.d_x = HEAD_STEP_X
        self.d_y = HEAD_STEP_Y

        self.max_total_steps = int(args.max_total_steps)

        # 巡线检测器：HSV 自适应阈值 + ROI 动态搜索 + 地面单应映射（方案一）
        self.detector = LineTrackerHSV(
            hsv_ranges=[(RED_LOW, RED_HIGH), (RED_LOW_2, RED_HIGH_2)],
            line_color="dark",
            min_area=80,
            lookahead_ratio=0.5,
            straightness_thresh=0.85,
            work_width=640,
            roi_ratio=0.5,
            pitch_pulse=TRACK_PITCH_PULSE,
            cam_height_cm=CAM_HEIGHT_CM,
        )

        self.running = False
        self.services_stopped = False
        self.robot_initialized = False

        # Reuse AGC's board/controller.
        self.board = AGC.board
        self.ctl = AGC.ctl

    def init_robot(self):
        if not self.args.no_kill and not self.args.dry_run:
            for pattern in ("TonyPi.py", "Joystick.py", "transport_color.py", "blue_pickup_overhead.py"):
                subprocess.run(["pkill", "-f", pattern], stderr=subprocess.DEVNULL)
            # 停掉开机自启服务,否则pkill后会被立刻重启,抢串口导致动作组set_bus_servo_pulse失败
            subprocess.run(["sudo", "systemctl", "stop", "tonypi", "joystick"], stderr=subprocess.DEVNULL)
            self.services_stopped = True
            time.sleep(2.0)
            # 重新初始化串口: AGC.board在import时创建,若当时TonyPi.py占串口会连接不良,stop后必须重建
            AGC.board = rrc.Board()
            AGC.ctl = Controller(AGC.board)
            self.board = AGC.board
            self.ctl = AGC.ctl
        self.set_head_center(duration=500)
        self.actions.run("stand")
        self.robot_initialized = True

    def set_head(self, x: Optional[int] = None, y: Optional[int] = None, duration: int = 100):
        if x is not None:
            self.x_dis = int(x)
        if y is not None:
            self.y_dis = int(y)
        if self.args.dry_run:
            log.info("[dry-run] head x=%d y=%d duration=%d", self.x_dis, self.y_dis, duration)
            return
        self.ctl.set_pwm_servo_pulse(1, self.y_dis, duration)
        self.ctl.set_pwm_servo_pulse(2, self.x_dis, duration)

    def set_head_center(self, duration: int = 300):
        self.head_mode = "left_right"
        self.head_turn = "left_right"
        self.d_x = HEAD_STEP_X
        self.d_y = HEAD_STEP_Y
        self.set_head(self.servo2, self.servo1, duration)
        time.sleep(duration / 1000.0 + 0.05)

    def _make_stream(self):
        open_once = False
        try:
            open_once = yaml_handle.get_yaml_data("/boot/camera_setting.yaml")["open_once"]
        except Exception:
            open_once = False
        return CameraStream(
            cam_size=CAM_SIZE,
            open_once_url="http://127.0.0.1:8080/?action=stream?dummy=param.mjpg" if open_once else None,
        )

    # =====================================================================
    # 底层感知/动作适配（移植自 line_seeker_tracking.py）
    # =====================================================================

    def capture_frame(self):
        """从采集线程取最新一帧（已由 CameraStream 后台抓帧）；失败返回 None"""
        if self.camera is None:
            return None
        return self.camera.read()

    def _head_lr(self, offset: int, duration: int = 400):
        """左右转头看线（offset 相对 servo2 中心，正=左；俯仰固定低头看线）"""
        self.set_head(self.servo2 + offset, TRACK_PITCH_PULSE, duration)
        time.sleep(duration / 1000.0 + 0.05)

    def _run_action(self, name: str, times: int = 1):
        """执行动作组，每步回站立（对齐 line_seeker_tracking 语义）"""
        self.actions.run(name, times=times, with_stand=True)

    def _turn_by_angle(self, angle_deg):
        """按角度闭环转向：正=左转（逆时针），负=右转（顺时针）。

        用「大步 turn_left/right + 小步 turn_left/right_small_step」组合逼近目标角度。
        返回近似实际转角（度，左正右负）。
        """
        if abs(angle_deg) < HEADING_SMALL:
            return 0.0
        if angle_deg > 0:
            big_deg, small_deg = TURN_LEFT_DEG, TURN_LEFT_SMALL_DEG
            big_action, small_action = "turn_left", "turn_left_small_step"
            sign = 1.0
        else:
            big_deg, small_deg = TURN_RIGHT_DEG, TURN_RIGHT_SMALL_DEG
            big_action, small_action = "turn_right", "turn_right_small_step"
            sign = -1.0

        remaining = abs(angle_deg)
        actual = 0.0
        big_count = int(remaining // big_deg)
        remaining -= big_count * big_deg
        small_count = int(round(remaining / small_deg))

        if big_count > 0:
            self._run_action(big_action, times=big_count)
            actual += big_count * big_deg
        if small_count > 0:
            self._run_action(small_action, times=small_count)
            actual += small_count * small_deg

        log.info("[转向] 目标 %+.1f° → %s×%d + %s×%d ≈ %+.1f°",
                 angle_deg, big_action, big_count, small_action, small_count, sign * actual)
        return sign * actual

    def _correct_heading(self, heading_deg):
        """按观测 heading 闭环修朝向：heading>0=线在前方偏右 → 身体右转（负角）"""
        return self._turn_by_angle(-heading_deg)

    # =====================================================================
    # Phase 1: 后退找线 + 右转进入线
    # =====================================================================

    def back_and_find_line(self):
        """后退找线：边后退边低头检测红线，找到即返回 True。

        白底红线只有直线：后退过程中线进入摄像头视野即算找到，无需头部扫描
        或身体旋转（原 seek_line 的 360° 扫描 / 平移试探已删除）。
        """
        log.info("=" * 50)
        log.info("===== 开始后退找线 =====")
        log.info("=" * 50)

        self._run_action("stand")
        self._head_lr(0)

        for step in range(self.args.back_steps_after_place):
            self._run_action("back_one_step")
            self._head_lr(0)
            frame = self.capture_frame()
            if frame is None:
                continue
            result = self.detector.detect(frame)
            if result.exists:
                p = result.primary
                log.info("[后退找线] 第 %d 步发现线：orientation=%s lookahead_x=%.1f",
                         step + 1, p.orientation, p.lookahead_x)
                return True

        log.info("[后退找线] 后退 %d 步未发现线", self.args.back_steps_after_place)
        return False

    def _align_to_line(self):
        """对准线（仅直道 follow）：先修朝向，再按 lookahead_x 横移，居中即完成。

        白底红线只有直线，不做 corner/cross 分支；非 follow 时前进一步观察。
        返回 True 表示已对准。
        """
        for step in range(ALIGN_MAX_STEPS):
            self._head_lr(0)
            frame = self.capture_frame()
            if frame is None:
                log.info("[对准] 拍照失败")
                return False
            result = self.detector.detect(frame)
            if not result.exists:
                log.info("[对准] 丢线")
                return False

            p = result.primary
            lx = max(-LOOKAHEAD_CLAMP, min(LOOKAHEAD_CLAMP, p.lookahead_x))
            log.info("[对准] step=%d orientation=%s lookahead_x=%.1f heading=%.1f",
                     step, p.orientation, lx, p.heading_deg)

            if p.orientation != "follow":
                log.info("[对准] 非 follow（%s），前进一步观察", p.orientation)
                self._run_action("go_forward_one_step")
                continue

            if abs(p.heading_deg) > HEADING_BIG:
                log.info("[对准] heading=%.1f°，先转向拉正", p.heading_deg)
                self._correct_heading(p.heading_deg)
                continue

            if abs(lx) > T_CENTER:
                if lx > 0:
                    log.info("[对准] lookahead_x=%.1f，右移对齐", lx)
                    self._run_action("right_move")
                else:
                    log.info("[对准] lookahead_x=%.1f，左移对齐", lx)
                    self._run_action("left_move")
                continue

            log.info("[对准] lookahead_x=%.1f ≤ %.1f，heading=%.1f°，对准完成",
                     lx, T_CENTER, p.heading_deg)
            return True

        log.info("[对准] 超过最大步数，返回")
        return False

    def turn_right_to_enter_line(self):
        """右转进入线：固定右转约 90°，把横向线转成纵向（follow），再对准。"""
        log.info("=" * 50)
        log.info("===== 右转进入线 =====")
        log.info("=" * 50)
        self._turn_by_angle(-ENTER_TURN_DEG)  # 负 = 右转
        self._head_lr(0)
        return self._align_to_line()

    # =====================================================================
    # Phase 2: 循迹阶段
    # =====================================================================

    def _handle_line_lost(self):
        """循迹丢线处理：后退找线。返回 True=找回，False=找回失败。"""
        for back_num in range(TRACK_MAX_BACKS):
            log.info("[丢线] 后退 %d/%d 次找线", back_num + 1, TRACK_MAX_BACKS)
            self._run_action("back_one_step")
            self._head_lr(0)
            frame = self.capture_frame()
            if frame is None:
                continue
            result = self.detector.detect(frame)
            if result.exists:
                log.info("[丢线] 后退后找回线：orientation=%s", result.primary.orientation)
                return True

        log.info("[丢线] 后退用尽仍丢线")
        return False

    def track_line(self):
        """循迹主函数：沿线直行，丢线后退找回，步数上限兜底。

        白底红线只有直线：不做直角弯 / 横线终点判定（原 far_ry 终点前进已删除），
        循迹靠步数上限结束（真机上可据线长调 --max-total-steps）。
        返回 True 表示正常结束，False 表示异常。
        """
        log.info("=" * 50)
        log.info("===== 开始循迹 =====")
        log.info("=" * 50)

        total_steps = 0
        while total_steps < self.max_total_steps:
            total_steps += 1

            self._head_lr(0)
            frame = self.capture_frame()
            if frame is None:
                log.info("[循迹] step=%d 取帧失败，跳过", total_steps)
                continue

            result = self.detector.detect(frame)

            if not result.exists:
                log.info("[循迹] step=%d 丢线", total_steps)
                if not self._handle_line_lost():
                    log.info("[循迹] 丢线无法找回，循迹结束")
                    return False
                continue

            p = result.primary
            lx = max(-LOOKAHEAD_CLAMP, min(LOOKAHEAD_CLAMP, p.lookahead_x))
            log.info("[循迹] step=%d orientation=%s lookahead_x=%.1f heading=%.1f "
                     "lateral=%.1f lat_cm=%s look_cm=%s",
                     total_steps, p.orientation, lx, p.heading_deg, p.lateral_offset,
                     ("%.1f" % p.lateral_offset_cm) if p.lateral_offset_cm is not None else "-",
                     ("%.1f" % p.lookahead_cm) if p.lookahead_cm is not None else "-")

            # 只有直线：非 follow 视为尚未进入线或已到线末端，前进一步观察
            if p.orientation != "follow":
                log.info("[循迹] 非 follow（%s），前进一步", p.orientation)
                self._run_action("go_forward_one_step")
                continue

            # follow 直道
            if FOLLOW_HEADING_MIN < abs(p.heading_deg) <= HEADING_BIG:
                log.info("[循迹] heading=%.1f°，闭环修朝向", p.heading_deg)
                self._correct_heading(p.heading_deg)
                continue

            if abs(lx) > T_BIG:
                if lx > 0:
                    log.info("[循迹] lookahead_x=%.1f > %.1f，右转修朝向", lx, T_BIG)
                    self._run_action("turn_right")
                else:
                    log.info("[循迹] lookahead_x=%.1f < -%.1f，左转修朝向", lx, T_BIG)
                    self._run_action("turn_left")
            elif abs(lx) > T_CENTER:
                if lx > 0:
                    log.info("[循迹] lookahead_x=%.1f，右移对齐", lx)
                    self._run_action("right_move")
                else:
                    log.info("[循迹] lookahead_x=%.1f，左移对齐", lx)
                    self._run_action("left_move")
            else:
                log.info("[循迹] 居中，前进（lookahead_x=%.1f）", lx)
                self._run_action("go_forward_one_step")

        log.info("[循迹] 达到最大步数 %d，循迹结束", self.max_total_steps)
        return True

    def run(self):
        stream = None
        try:
            stream = self._make_stream()
            stream.start()
            self.camera = stream

            self.init_robot()
            # init_robot 杀了tonypi服务会重置USB相机, 重建并预热
            stream.release()
            stream = self._make_stream()
            stream.start()
            self.camera = stream

            log.info(
                "line-follow debug start dry_run=%s back_steps=%d max_total_steps=%d",
                self.args.dry_run,
                self.args.back_steps_after_place,
                self.max_total_steps,
            )

            # 后退找线 → 右转进入线 → 循迹
            if not self.back_and_find_line():
                log.warning("后退找线失败，循迹终止")
            elif not self.turn_right_to_enter_line():
                log.warning("右转进入线失败，循迹终止")
            else:
                self.track_line()

        except BaseException as exc:
            log.exception("fatal error: %s", exc)
        finally:
            self.running = False
            if stream is not None:
                stream.release()
            cv2.destroyAllWindows()
            if self.robot_initialized and not self.args.dry_run:
                AGC.stopActionGroup()
                self.actions.run("stand")
            if self.services_stopped:
                subprocess.run(["sudo", "systemctl", "start", "tonypi", "joystick"], stderr=subprocess.DEVNULL)
            log.info("line-follow debug exit")


def parse_args():
    parser = argparse.ArgumentParser(description="放下蓝色方块后: 后退找线→右转进入线→循迹 独立调试")
    parser.add_argument("--dry-run", action="store_true", help="log actions without executing motion")
    parser.add_argument("--no-kill", action="store_true", help="do not kill existing TonyPi/Joystick processes")
    parser.add_argument("--back-steps-after-place", type=int, default=12, help="后退找线步数(每步后退并检测红线, 找到即停)")
    parser.add_argument("--max-total-steps", type=int, default=200, help="循迹总步数安全上限(固定步数兜底, 防死循环)")
    args = parser.parse_args()
    if min(args.back_steps_after_place, args.max_total_steps) < 0:
        parser.error("movement counts must be nonnegative")
    return args


def main():
    args = parse_args()
    task = LineFollowDebug(args)
    task.run()


if __name__ == "__main__":
    main()
