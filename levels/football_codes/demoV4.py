#!/usr/bin/python3
# coding=utf8
"""
demoV4.py — 赛道巡线结束后自动进入 Tag 导航/射门流程

基于 demoV2，新增 V4 展示流程规则：每个球门只射一脚；踢球动作完成后不判断
是否进门，直接执行阶段间动作并进入下一阶段。

整体流程：
    RedLinePatrolV3 红线巡线
        -> 连续确认看不到红线，只作为候选终点
        -> 停车转头确认 Tag103；若没有 Tag，则后退找回漏识别直角弯
    tag_walk_demo
        -> 使用 tag_route_demo 的 AprilTag 路线状态机
        -> 找球、球门柱对齐、踢球；不进行进门结果判断
        -> 第二球门完成后沿出口路线离场

注意：
    1. 不修改 RedLinePatrolV3.py；这里只调用它已有的 init/start/run/stop/exit 接口。
    2. tag_route_demo.py 是路线状态机，实际包含摄像头循环和射门衔接的是 tag_walk_demo.py。
    3. 推荐在机器人上运行：
         sudo systemctl stop tonypi
         cd /home/pi/TonyPi/Functions
         python3 demoV4.py --run
"""

from __future__ import print_function

import argparse
import os
import sys
import time


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 支持从 Robot_control_self_module/levels/football_codes 直接运行：TonyPi
# 框架目录提供 hiwonder SDK 与 Functions 包（CameraCalibration 等）。在
# TonyPi 目录下运行时该路径已在 sys.path 中，此块自动跳过。
_TONYPI_DIR = '/home/pi/TonyPi'
if os.path.isdir(_TONYPI_DIR) and _TONYPI_DIR not in sys.path:
    sys.path.insert(0, _TONYPI_DIR)

try:
    # 优先本地同目录导入（Robot_control_self_module 副本为唯一真源）；
    # 本地缺失时回退 Functions（TonyPi 框架运行路径）。
    import RedLinePatrolV3 as redline
except ImportError:
    from Functions import RedLinePatrolV3 as redline

try:
    from patrol_end_recovery import PatrolEndRecoveryController
except ImportError:
    from Functions.patrol_end_recovery import PatrolEndRecoveryController


class PatrolEndDetector:
    """检测候选巡线终点，而不是直接判定第一阶段结束。

    终点判断优先使用脚下近处 ROI：
      1. 原巡线模块定义的下部近处 ROI（脚下区域）连续无红线；
      2. 如果脚下 ROI 检测不可用，再退化为整体 line_center_x 连续丢失。

    这样即使远处仍能看到红线，只要机器人脚下已经离开赛道，也能进入下一阶段；
    同时仍会避开直角弯处理期间的短暂丢线。RedLinePatrolV3.py 本身保持不变。
    """

    # 转弯完成后脚下无红线的冷却期（秒）。转弯刚结束红线仍停在转弯后的前方、
    # 尚未压到脚下，这段时间禁止判定候选终点，避免转弯一结束就被误判。
    TURN_END_COOLDOWN_S = 2.5

    def __init__(self, end_confirm_s=0.8, min_corner_index=3):
        self.end_confirm_s = max(0.0, float(end_confirm_s))
        self.min_corner_index = max(0, int(min_corner_index))
        self.seen_line = False
        self.absent_since = None
        self.foot_absent_since = None
        self.last_foot_line_present = None
        self._foot_detect_error_reported = False

    @staticmethod
    def _scale_roi(frame, roi):
        """把 RedLinePatrolV3 的 640x480 ROI 映射到当前帧尺寸。"""
        frame_h, frame_w = frame.shape[:2]
        y0, y1, x0, x1 = roi[:4]
        sx = float(frame_w) / 640.0
        sy = float(frame_h) / 480.0
        return (
            max(0, min(frame_h, int(round(y0 * sy)))),
            max(0, min(frame_h, int(round(y1 * sy)))),
            max(0, min(frame_w, int(round(x0 * sx)))),
            max(0, min(frame_w, int(round(x1 * sx)))),
        )

    def detect_foot_line(self, frame):
        """检测脚下近处 ROI 是否存在足够大的红线轮廓。

        优先复用 RedLinePatrolV3.detect_red_line() 的 HSV 和形态学参数，避免
        总控和巡线模块使用两套不一致的红线定义。检测失败时返回 None，届时
        update() 会退化为原来的整体 line_center_x 判断。
        """
        if frame is None:
            return None
        try:
            import cv2

            rois = getattr(redline, "roi", None)
            if not rois:
                return None
            y0, y1, x0, x1 = self._scale_roi(frame, rois[-1])
            if y1 <= y0 or x1 <= x0:
                return None

            mask = redline.detect_red_line(frame[y0:y1, x0:x1])
            contours = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )[-2]
            min_area = float(getattr(redline, "MIN_CONTOUR_AREA", 50))
            return any(cv2.contourArea(contour) >= min_area for contour in contours)
        except Exception as exc:
            if not self._foot_detect_error_reported:
                print("脚下红线检测失败，退化为整体丢线判断：{}".format(exc), flush=True)
                self._foot_detect_error_reported = True
            return None

    def update(self, frame=None):
        now = time.monotonic()
        line_seen = getattr(redline, "line_center_x", -1) != -1
        foot_line_present = self.detect_foot_line(frame)
        self.last_foot_line_present = foot_line_present

        if line_seen:
            self.seen_line = True
            self.absent_since = None
        elif self.seen_line and self.absent_since is None:
            self.absent_since = now

        if foot_line_present is True:
            self.foot_absent_since = None
        elif foot_line_present is False and self.foot_absent_since is None:
            self.foot_absent_since = now

        # 启动时如果相机暂时没有拍到红线，不允许直接判定到达终点。
        if not self.seen_line:
            return False

        # 进度门限（M9）：顺序表完成足够多的弯之前，中途离线不得判定候选
        # 终点。260927 实车：弯2 后侧冲出赛道触发终点复核，恢复流程在画面
        # 左边缘红线处左右横移震荡。若漏弯导致真到终点时进度不足，可用
        # --patrol-end-min-corner-index 调低，或置 0 关闭此门限。
        try:
            if redline.get_corner_index() < self.min_corner_index:
                self.absent_since = None
                self.foot_absent_since = None
                return False
        except Exception:
            pass

        # 直角弯处理或弯前接近阶段的丢线由巡线逻辑自己恢复。
        if getattr(redline, "turn_started", False) or getattr(redline, "approach_active", False):
            self.absent_since = None
            self.foot_absent_since = None
            return False

        # 丢线搜索进行中不判定候选终点：搜索转向期间整体丢线是预期状态，
        # 终点判定会把还没搜完的搜索打断。
        if getattr(redline, "search_active", False):
            self.absent_since = None
            self.foot_absent_since = None
            return False

        # 转弯刚结束的空窗期：红线仍停在转弯后的前方、尚未压到脚下，
        # 这段时间禁止判定候选终点，避免转弯一结束就被误判。
        if now - getattr(redline, "last_turn_completed_at", 0.0) < self.TURN_END_COOLDOWN_S:
            self.absent_since = None
            self.foot_absent_since = None
            return False


        # 脚下 ROI 是主要终点证据：远处仍有红线不影响切换。
        # 只有检测不可用时，才回退到整体 line_center_x 的丢线状态。
        if foot_line_present is not None:
            if foot_line_present is True or self.foot_absent_since is None:
                return False
            # 整体仍能看到红线：红线还在前方，是转弯后空窗而非真正终点。
            if line_seen:
                self.foot_absent_since = None
                return False
            return now - self.foot_absent_since >= self.end_confirm_s

        if line_seen or self.absent_since is None:
            return False
        return now - self.absent_since >= self.end_confirm_s


def open_camera():
    """沿用 locate_web/tag_walk_demo 使用的原始 TonyPi 摄像头接口。"""
    import hiwonder.Camera as Camera

    camera = Camera.Camera()
    camera.camera_open()
    time.sleep(1.0)
    return camera


def close_camera(camera):
    if camera is None:
        return
    try:
        camera.camera_close()
    except Exception as exc:
        print("关闭巡线摄像头时出现异常：{}".format(exc), flush=True)


def build_undistort_maps():
    """加载 RedLinePatrolV3 独立运行入口使用的相机标定映射。"""
    import cv2
    import numpy as np

    calibration_param_path = getattr(redline, "calibration_param_path", None)
    if not calibration_param_path:
        raise RuntimeError("RedLinePatrolV3 未提供 calibration_param_path")

    param_file = calibration_param_path + ".npz"
    if not os.path.exists(param_file):
        raise FileNotFoundError("找不到相机标定文件：{}".format(param_file))

    param_data = np.load(param_file)
    mtx = param_data["mtx_array"]
    dist = param_data["dist_array"]
    newcameramtx, _ = cv2.getOptimalNewCameraMatrix(
        mtx, dist, (640, 480), 0, (640, 480)
    )
    mapx, mapy = cv2.initUndistortRectifyMap(
        mtx, dist, None, newcameramtx, (640, 480), 5
    )
    return mapx, mapy


def _install_dry_run_action_logger():
    """让不带 --run 时的巡线阶段只打印动作，不驱动舵机。"""
    original = redline.AGC.runActionGroup

    def log_action(action_name, *args, **kwargs):
        suffix = ""
        if args:
            suffix += " args={}".format(args)
        if kwargs:
            suffix += " kwargs={}".format(kwargs)
        print("[DRY-RUN][RedLinePatrolV3] {}{}".format(action_name, suffix), flush=True)

    redline.AGC.runActionGroup = log_action
    return original


def run_red_line_stage(args):
    """运行原始 RedLinePatrolV3，直到稳定确认红线消失。"""
    import cv2

    camera = None
    original_action = None
    patrol_started = False
    reached_end = False

    if not args.run:
        original_action = _install_dry_run_action_logger()

    try:
        mapx, mapy = build_undistort_maps()
        redline.reset()
        redline.init()
        redline.start()
        patrol_started = True

        camera = open_camera()
        redline.AGC.runActionGroup("stand")
        print(
            "红线巡线阶段已启动；脚下稳定丢线后将先转头确认 Tag103",
            flush=True,
        )
        print(
            "巡线动作：{}；终点确认时间：{:.2f}s".format(
                "真实执行" if args.run else "dry-run（只打印动作）",
                args.patrol_end_confirm_seconds,
            ),
            flush=True,
        )

        end_detector = PatrolEndDetector(
            args.patrol_end_confirm_seconds,
            min_corner_index=args.patrol_end_min_corner_index)
        end_recovery = PatrolEndRecoveryController(
            redline=redline,
            execute_actions=args.run,
            target_tag_ids=(103,),
            tag_confirm_frames=args.patrol_end_tag_confirm_frames,
            head_pan_offset=args.patrol_end_head_pan_offset,
            head_settle_s=args.patrol_end_head_settle_seconds,
            head_observe_s=args.patrol_end_head_observe_seconds,
            max_back_steps=args.patrol_recovery_max_back_steps,
            back_step_interval_s=args.patrol_recovery_back_interval,
            turn_recovery_timeout_s=args.patrol_recovery_turn_timeout,
        )
        started_at = time.monotonic()
        last_status = None

        while True:
            if time.monotonic() - started_at > args.patrol_max_seconds:
                print("巡线阶段超过最大运行时间，未确认终点，停止总控流程", flush=True)
                return False

            ret, frame = camera.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            corrected = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

            if end_recovery.active:
                recovery_event = end_recovery.update(frame, corrected)
                display_frame = (
                    recovery_event.display_frame
                    if recovery_event.display_frame is not None
                    else corrected.copy()
                )
                combined_status = recovery_event.message
                if combined_status != last_status:
                    print(
                        "[PATROL END RECOVERY][{}] {}".format(
                            recovery_event.kind, combined_status
                        ),
                        flush=True,
                    )
                    last_status = combined_status

                if end_recovery.state == end_recovery.TAG_FOUND:
                    reached_end = True
                    print(
                        "[RedLinePatrolV3] 已连续确认 Tag103，第一阶段结束，进入 Tag/射门阶段",
                        flush=True,
                    )
                    return True

                if end_recovery.state == end_recovery.RECOVERED:
                    # 漏掉的直角弯已经由原巡线闭环完成。清空候选终点的
                    # 丢线计时，继续第一阶段，避免本帧立即再次触发终点复核。
                    end_detector = PatrolEndDetector(
                        args.patrol_end_confirm_seconds,
                        min_corner_index=args.patrol_end_min_corner_index)
                    last_status = None
                    print(
                        "[RedLinePatrolV3] 已恢复漏识别直角弯，继续巡线",
                        flush=True,
                    )
                    time.sleep(0.01)
                    continue

                if end_recovery.state == end_recovery.FAILED:
                    print(
                        "[RedLinePatrolV3] 终点复核/漏弯恢复失败，已安全停止",
                        flush=True,
                    )
                    return False

                line_present = getattr(redline, "line_center_x", -1) != -1
            else:
                display_frame = redline.run(corrected)
                reached_end_now = end_detector.update(corrected)
                line_present = getattr(redline, "line_center_x", -1) != -1
                foot_line = end_detector.last_foot_line_present
                status = "检测到红线" if line_present else "当前未检测到红线"
                if foot_line is True:
                    foot_status = "脚下有红线"
                elif foot_line is False:
                    foot_status = "脚下无红线"
                else:
                    foot_status = "脚下红线检测不可用"
                combined_status = "{}；{}".format(status, foot_status)
                if combined_status != last_status:
                    print("[RedLinePatrolV3] {}".format(combined_status), flush=True)
                    last_status = combined_status

                if reached_end_now:
                    recovery_event = end_recovery.start()
                    print(
                        "[PATROL END RECOVERY][{}] {}".format(
                            recovery_event.kind, recovery_event.message
                        ),
                        flush=True,
                    )
                    last_status = recovery_event.message
                    if end_recovery.state == end_recovery.FAILED:
                        return False

            if args.display:
                cv2.putText(
                    display_frame,
                    "RedLinePatrolV3: {}".format(combined_status),
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0) if line_present else (0, 0, 255),
                    2,
                )
                cv2.imshow("CourseDemo-RedLinePatrolV3", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    print("收到退出按键，停止总控流程", flush=True)
                    return False

            time.sleep(0.01)
    finally:
        # 先停止 RedLinePatrolV3 的运动线程，再关闭相机，最后执行退出动作。
        if patrol_started:
            try:
                redline.stop()
            except Exception as exc:
                print("停止 RedLinePatrolV3 时出现异常：{}".format(exc), flush=True)

        close_camera(camera)

        if patrol_started:
            try:
                redline.exit()
                # 巡线动作线程已确认退出；给 stand_slow 留出稳定时间，
                # 再初始化 Tag/射门阶段的舵机和动作组。
                time.sleep(0.3)
            except Exception as exc:
                print("退出 RedLinePatrolV3 时出现异常：{}".format(exc), flush=True)

        if original_action is not None:
            redline.AGC.runActionGroup = original_action

        if args.display:
            cv2.destroyWindow("CourseDemo-RedLinePatrolV3")

    return reached_end


def run_tag_shot_stage(args):
    """进入现有 Tag 导航 + 足球射门总流程。"""
    try:
        from Functions.tag_walk_demo import TagWalkDemo
    except ImportError:
        from tag_walk_demo import TagWalkDemo

    print(
        "开始进入 Tag 导航/射门阶段（V4：每门只射 1 脚，踢完直接推进，不判断进门）",
        flush=True,
    )
    demo = TagWalkDemo(
        execute_actions=args.run,
        ball_confidence=args.ball_conf,
        goalpost_confirm_frames=args.goalpost_confirm_frames,
        kick_attempts_per_goal=1,
        advance_after_kick_without_goal_check=True,
        ball_loss_timeout_s=args.ball_loss_timeout,
        max_seconds=args.tag_max_seconds,
    )
    demo.run(display=args.display)


def parse_args():
    parser = argparse.ArgumentParser(
        description="V4 展示：巡线后完成 Tag 导航，每门踢一脚后直接推进"
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="真实执行机器人动作；不加时巡线和射门阶段均为 dry-run",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="显示巡线阶段的调试窗口；无桌面环境不要使用",
    )
    parser.add_argument(
        "--patrol-end-confirm-seconds",
        type=float,
        default=0.8,
        help="脚下近处 ROI 连续无红线后，额外确认多久才切换",
    )
    parser.add_argument(
        "--patrol-end-min-corner-index",
        type=int,
        default=3,
        help="终点判定的进度门限：完成至少该数量的直角弯后才允许判定候选"
        "终点（防中途离线误判；漏弯场景可调低或置 0 关闭）",
    )
    parser.add_argument(
        "--patrol-end-tag-confirm-frames",
        type=int,
        default=3,
        help="候选巡线终点处连续看到 Tag103 多少帧才允许结束第一阶段",
    )
    parser.add_argument(
        "--patrol-end-head-pan-offset",
        type=int,
        default=500,
        help="候选终点复核时头部左右扫描的 servo2 脉宽偏移",
    )
    parser.add_argument(
        "--patrol-end-head-settle-seconds",
        type=float,
        default=0.8,
        help="候选终点复核时每次转头后的稳定等待时间",
    )
    parser.add_argument(
        "--patrol-end-head-observe-seconds",
        type=float,
        default=1.0,
        help="候选终点复核时每个头部方向的观察时间",
    )
    parser.add_argument(
        "--patrol-recovery-max-back-steps",
        type=int,
        default=8,
        help="未找到 Tag103 时，为找回漏识别直角弯最多后退多少步",
    )
    parser.add_argument(
        "--patrol-recovery-back-interval",
        type=float,
        default=0.8,
        help="漏弯恢复时每次后退动作后的视觉观察间隔",
    )
    parser.add_argument(
        "--patrol-recovery-turn-timeout",
        type=float,
        default=25.0,
        help="找回直角弯后，原转弯闭环允许的最长完成时间",
    )
    parser.add_argument(
        "--patrol-max-seconds",
        type=float,
        default=300.0,
        help="巡线阶段最大运行时间",
    )
    parser.add_argument(
        "--tag-max-seconds",
        type=float,
        default=600.0,
        help="Tag/射门阶段最大运行时间",
    )
    parser.add_argument(
        "--ball-conf",
        type=float,
        default=0.40,
        help="足球模型置信度阈值，默认 0.40",
    )
    parser.add_argument(
        "--goalpost-confirm-frames",
        type=int,
        default=2,
        help="最终射门窗口连续确认次数，默认 2",
    )
    parser.add_argument(
        "--ball-loss-timeout",
        type=float,
        default=8.0,
        help="射门阶段连续看不到足球多少秒后放弃当前球门，默认 8 秒",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("整体赛道 Demo V4 启动（踢完即推进，不判断进门）", flush=True)
    print("第一阶段：RedLinePatrolV3 红线巡线", flush=True)
    print("第二阶段：tag_route_demo 路线导航 + tag_walk_demo 射门流程", flush=True)

    if not run_red_line_stage(args):
        print("未完成巡线终点确认，整体流程结束", flush=True)
        return

    run_tag_shot_stage(args)
    print("整体赛道 Demo V4 结束", flush=True)


if __name__ == "__main__":
    main()
