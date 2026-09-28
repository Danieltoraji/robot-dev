# -*- coding: utf-8 -*-
"""上下楼梯与识别跨障关卡（levels/stairs_hurdle.py）

流程：对正（不计时段）→ 闭环接近楼梯 → 上楼（逐动作方位微调）→ 下楼
→ 条件后退 → 闭环接近横杆 → 跨杆 → 出场。

感知：vision/red_line_detector.py（HSV 红双区间连通域下沿点）
度量：core/ground_line_meter.py（机器人本地系地面单应 + 直线拟合）
    - 平地段距离/横偏/方位全部有效（胶条与横杆下沿=地面线）
    - 台阶段只用 bearing（横杆低于脚下平面，距离受径向缩放偏差）

控制约定（右正左负，与 nine_grid/goodluck 一致）：
    bearing_err > 0 机体右偏 → turn_left 修正
    lateral > 0 目标在机体右侧 → right_move 对中

失败降级（不弃赛）：视觉丢失 → 变俯仰档重试 → 航位推算盲走 → 开环保底。
唯一接受的中止点：ALIGN 粗接近超限（起跑前停下好过乱撞）。

运行：python main.py stairs_hurdle；集成仿真：python tests/test_stairs_hurdle_sim.py
"""

import math
import time
from collections import deque

import numpy as np

from core.camera_config import HEAD_CENTER
from core.ground_line_meter import (
    build_meter, min_visible_ground_cm,
)
from core.robot_core import RobotState
from vision.red_line_detector import RedLineDetector

# =====================================================================
# 观测位姿（M3 现场标定后核对）
# =====================================================================
PITCH_OBS = 1000        # 观测俯仰（参考队实测 8cm 内才盲区）
CAM_HEIGHT_CM = 39.0    # 站立光心高度（M3 卷尺实测覆盖；标定协议与之绑定）
HEAD_PULSE = HEAD_CENTER

# =====================================================================
# 触发阈值（默认由可见下界 d_min 推导；M3 实测 σ 后按窗口公式改写）
# =====================================================================
D_ALIGN_CM = 40.0       # 对正完成点（胶条前向距离）
D_MIN_MARGIN = 3.0
D_HURDLE_MARGIN = 4.0
CONFIRM_SLACK_CM = 1.5  # 原地复测确认的放宽量
ROLLING_N = 3           # 行进中滚动中位数窗口

# =====================================================================
# 容差与循环上限
# =====================================================================
TOL_ROT_DEG = 2.5       # 对正航向容差（运行时取 max(本值, 量子/2)）
TOL_LAT_CM = 3.0
TOL_ROT_CLIMB_DEG = 3.0  # 爬楼段逐动作方位微调容差（现场标定）
ALIGN_ROUNDS = 8
RECOVER_RETRIES = 3
COARSE_LIMIT_CM = 100.0  # 粗接近累计上限（超限停机报警）
TARGET_JUMP_CM = 8.0     # 前向读数正向跳变超此 = 疑似胶条/横杆切换（按丢失处理）

# =====================================================================
# 动作与名义位移（M3"三常数实测仪式"覆盖；预测/盲走用）
# =====================================================================
A_STAND = "stand"
A_FWD = "go_forward_one_step"
A_FWD_SMALL = "go_forward_one_small_step"
A_BACK = "back_one_step"
A_LEFT = "left_move"
A_RIGHT = "right_move"
A_TURN_L = "turn_left_small_step"
A_TURN_R = "turn_right_small_step"
A_CLIMB = "climb_stairs"
A_DOWN = "down_floor"
A_HURDLE = "hurdles"

FWD_STEP_CM = 2.0
FWD_SMALL_CM = 1.5
BACK_STEP_CM = 3.2
TURN_QUANTUM_DEG = 2.0   # turn_small 单次角度（在线不可测，M3 实测覆盖）

# 参考实现继承的开环参数（Final_Edition.py；横杆不可见时的回退值）
CLIMB_INTERVAL_S = 1.0
DOWN_INTERVAL_S = 1.2
TOP_BACK_STEPS = 1
TOP_FALLBACK_TURN_R = 2
EXIT_STEPS = 6           # 跨杆后前进出光束（宁少勿多，M3 标定）

# =====================================================================
# 计时（按固件 80s 档保守预算；现场确认 150s 档后放宽）
# =====================================================================
TOTAL_BUDGET_S = None    # None=不启用看门狗（仿真默认）；真机传 75.0
SETTLE_S = 0.3           # 动作后静置（标定协议组成部分，勿单方面改）
FRAMES_TRIGGER = 3       # 触发复测帧数（看门狗经济模式降到 1）


class StairsHurdleLevel:
    """FSM 主控。state 为 RobotState（或仿真子类）。"""

    def __init__(self, state: RobotState, cam_height_cm=CAM_HEIGHT_CM,
                 total_budget_s=TOTAL_BUDGET_S, settle_s=SETTLE_S,
                 detector=None, verbose=True):
        self.state = state
        self.cam_height = cam_height_cm
        self.total_budget = total_budget_s
        self.settle_s = settle_s
        self.verbose = verbose
        self.detector = detector or RedLineDetector()
        self.d_min = min_visible_ground_cm(cam_height_cm, PITCH_OBS)
        self.d_climb = self.d_min + D_MIN_MARGIN
        self.d_hurdle = self.d_min + D_HURDLE_MARGIN
        # 运行时状态
        self.meter = None
        self._fwd_hist = deque(maxlen=ROLLING_N)
        self._last_valid_fwd = None
        self._photo_every = 1
        self._frames_trigger = FRAMES_TRIGGER
        self._deadline = None
        self._start_t = None

    # ------------------------------------------------------------------
    # 基础设施
    # ------------------------------------------------------------------

    def _reset_target_hist(self):
        """切换追踪目标时清空滚动历史与上次有效距离"""
        self._fwd_hist.clear()
        self._last_valid_fwd = None

    def _log(self, msg):
        if self.verbose:
            print(f"[stairs_hurdle] {msg}")

    def _sleep(self, s=None):
        time.sleep(self.settle_s if s is None else s)

    def _check_watchdog(self):
        if self.total_budget is None or self._deadline is None:
            return
        elapsed = time.time() - self._start_t
        if elapsed > self.total_budget and self._photo_every == 1:
            self._photo_every = 2
            self._frames_trigger = 1
            self._log(f"预算看门狗：已用 {elapsed:.0f}s，进入经济模式"
                      f"（{self._photo_every} 步一拍，复测 {self._frames_trigger} 帧）")

    def _turn_steps(self, err_deg):
        """航向误差 -> turn_small 步数（方向内聚到调用方）"""
        quantum = max(TURN_QUANTUM_DEG, 1e-6)
        tol = max(TOL_ROT_DEG, TURN_QUANTUM_DEG / 2.0)
        if abs(err_deg) <= tol:
            return 0
        return int(math.ceil(abs(err_deg) / quantum))

    def _turn(self, err_deg):
        """按方位误差修正航向：>0 机体右偏 → turn_left"""
        n = self._turn_steps(err_deg)
        if n <= 0:
            return
        action = A_TURN_L if err_deg > 0 else A_TURN_R
        self.state.act(action, times=n)
        self._log(f"航向修正 {err_deg:+.1f}° -> {action} x{n}")

    # ------------------------------------------------------------------
    # 测量
    # ------------------------------------------------------------------

    def _measure_once(self):
        """单帧 拍照->检测->度量；拍照失败重试 3 次。失败返回 None"""
        for _ in range(3):
            frame = self.state.capture_frame()
            if frame is not None:
                det = self.detector.detect(frame)
                return self.meter.measure(det.components)
            self._log("拍照失败，重试")
        return None

    def _forward_valid(self, m):
        """滚动中位数 + 目标切换守卫；返回 (ok, forward_cm)，m 为 None/丢失返回 (False, None)"""
        if m is None or not m.exists:
            return False, None
        fwd = m.forward_cm
        if (self._last_valid_fwd is not None
                and fwd > self._last_valid_fwd + TARGET_JUMP_CM):
            self._log(f"前向读数跳变 {self._last_valid_fwd:.1f}->{fwd:.1f}cm，"
                      "疑似胶条/横杆切换，按丢失处理")
            return False, None
        self._last_valid_fwd = fwd
        self._fwd_hist.append(fwd)
        return True, float(np.median(self._fwd_hist))

    def _measure_forward(self, label=""):
        """标准测量：1 帧 + 滚动中位数。返回 (ok, forward_cm, measurement)"""
        self._check_watchdog()
        m = self._measure_once()
        ok, fwd = self._forward_valid(m)
        if ok:
            self._log(f"{label} 前向 {fwd:.1f}cm 方位 {m.bearing_err_deg:+.1f}°"
                      f" 横偏 {m.lateral_cm:+.1f}cm 内点率 {m.inlier_ratio:.2f}")
        else:
            self._log(f"{label} 目标丢失（{m.reason if m else '拍照失败'}）")
        return ok, fwd, m

    def _pitch_retry(self):
        """变俯仰档重试（降级精度）：1050/950 各一帧，成功返回 measurement"""
        saved = self.state.current_pitch_pulse
        try:
            for pulse in (1050, 950):
                self.state.set_pitch(pulse)
                m = self._measure_once()
                if m is not None and m.exists:
                    self._log(f"俯仰 {pulse} 档找回目标（降级精度）")
                    return m
        finally:
            self.state.set_pitch(saved)
        return None

    # ------------------------------------------------------------------
    # FSM 状态
    # ------------------------------------------------------------------

    def run_level(self):
        try:
            self._start_t = time.time()
            self._deadline = (self._start_t + self.total_budget
                              if self.total_budget else None)
            self._init_pose()
            self.meter = build_meter(PITCH_OBS, self.cam_height)
            if self.meter.degraded:
                self._log("警告：无现场标定，使用 from_pose 自举（±3cm 级），"
                          "请先运行 tools/calib_stairs_hurdle.py")
            if not self._align():
                return self._finish(False, "对正失败，停机报警")
            self._approach(self.d_climb, "楼梯段")
            self._climb()
            self._reset_target_hist()  # 目标从胶条切换为横杆，历史距离作废
            self._recover()
            self._approach(self.d_hurdle, "横杆段")
            self._cross()
            self._exit()
            return self._finish(True)
        except Exception as e:  # 任何异常都落站立，不留倒姿
            self._log(f"异常：{type(e).__name__}: {e}")
            self._safe_stand()
            return self._finish(False, f"异常退出: {e}")

    def _init_pose(self):
        self.state.act(A_STAND)
        self.state.set_head(HEAD_PULSE)
        self.state.set_pitch(PITCH_OBS)
        self._sleep(0.5)

    def _finish(self, ok, why=""):
        self._log(("完成" if ok else "失败") + (f"：{why}" if why else ""))
        return ok

    def _safe_stand(self):
        try:
            self.state.act(A_STAND)
        except Exception:
            pass

    # -- 对正（不计时段） ------------------------------------------------

    def _align(self):
        # 粗接近：递增步数直到检出胶条
        walked = 0.0
        batch = 2
        while True:
            ok, fwd, _ = self._measure_forward("对正粗测")
            if ok:
                break
            self.state.act(A_FWD, times=batch)
            walked += batch * FWD_STEP_CM
            self._sleep()
            if walked > COARSE_LIMIT_CM:
                self._log(f"粗接近 {walked:.0f}cm 未见胶条，超限停机")
                return False
            batch = min(8, batch * 2)
        # 细对正：旋转至直线水平 → 横移至居中（横移每步由下轮测量复核）
        for round_i in range(ALIGN_ROUNDS):
            ok, fwd, m = self._measure_forward("对正细测")
            if not ok:
                m2 = self._pitch_retry()
                if m2 is None:
                    continue
                m = m2
            if abs(m.bearing_err_deg) > TOL_ROT_DEG:
                self._turn(m.bearing_err_deg)
            elif abs(m.lateral_cm) > TOL_LAT_CM:
                action = A_RIGHT if m.lateral_cm > 0 else A_LEFT
                self.state.act(action, times=1)
                self._log(f"横移对中 {m.lateral_cm:+.1f}cm -> {action}")
            else:
                self._log(f"对正达标（方位 {m.bearing_err_deg:+.1f}° "
                          f"横偏 {m.lateral_cm:+.1f}cm）")
                break
        else:
            self._log("对正轮数耗尽，接受当前误差继续")
        # 前进到 D_ALIGN
        return self._approach(D_ALIGN_CM, "对正收尾", confirm=False)

    # -- 闭环接近（一步一拍） --------------------------------------------

    def _approach(self, target_cm, label, confirm=True):
        """行进测量直到触发。返回 'ok'（正常触发）或 'blind'（航位推算到位）"""
        steps_since_photo = 0
        while True:
            self._check_watchdog()
            if steps_since_photo < self._photo_every:
                self.state.act(A_FWD, times=1)
                steps_since_photo += 1
                self._sleep()
                continue
            steps_since_photo = 0
            ok, fwd, _ = self._measure_forward(label)
            if ok and fwd <= target_cm:
                if not confirm:
                    self._log(f"{label} 到达 {fwd:.1f}cm")
                    return "ok"
                dm = self._confirm_burst()
                if dm is not None and dm <= target_cm + CONFIRM_SLACK_CM:
                    self._log(f"{label} 触发确认：复测中位 {dm:.1f}cm ≤ "
                              f"{target_cm + CONFIRM_SLACK_CM:.1f}cm")
                    self.trigger_fwd = fwd
                    return "ok"
            if not ok:
                outcome = self._lost_recovery(target_cm, label)
                if outcome == "ok":
                    return "ok"
                if outcome == "blind":
                    return "blind"
                # 'retry'：找回目标，继续循环
            # 近距切小步：进入 d_min+6cm 内盲区风险大，用小步防冲过
            if fwd is None or fwd < self.d_min + 6.0:
                self.state.act(A_FWD_SMALL, times=1)
            else:
                self.state.act(A_FWD, times=1)
            self._sleep()

    def _confirm_burst(self):
        """原地突发复测，返回复测前向中位数（失败 None）"""
        reads = []
        for _ in range(self._frames_trigger):
            m = self._measure_once()
            if m is not None and m.exists:
                reads.append(m.forward_cm)
        if not reads:
            return None
        return float(np.median(reads))

    def _lost_recovery(self, target_cm, label):
        """接近段目标丢失降级链。返回 'ok' / 'blind' / 'retry'"""
        m = self._pitch_retry()
        if m is not None:
            return "retry"
        # 航位推算盲走到目标：按上次有效距离换算小步数
        if self._last_valid_fwd is not None:
            residual = self._last_valid_fwd - target_cm
            steps = max(0, int(round(residual / FWD_SMALL_CM)))
            self._log(f"{label} 视觉丢失，航位推算盲走 {steps} 小步"
                      f"（上次有效 {self._last_valid_fwd:.1f}cm）")
            if steps:
                self.state.act(A_FWD_SMALL, times=steps)
            dm = self._confirm_burst()
            if dm is not None and dm <= target_cm + CONFIRM_SLACK_CM:
                return "ok"
            self._log(f"{label} 盲走后仍不可见，按当前位置继续（不弃赛）")
            return "blind"
        self._log(f"{label} 无历史距离，按当前位置继续（不弃赛）")
        return "blind"

    # -- 楼梯段 ----------------------------------------------------------

    def _climb(self):
        self._log("开始上楼")
        for i in (1, 2):
            self.state.act(A_CLIMB)
            self.state.act(A_FWD_SMALL)
            self._sleep(CLIMB_INTERVAL_S)
            self._climb_bearing_fix(f"上楼第{i}级后")
        # 顶部：几何位置调整（继承参考）+ 实测方位修正
        self.state.act(A_BACK, times=TOP_BACK_STEPS)
        if not self._climb_bearing_fix("顶部"):
            self.state.act(A_TURN_R, times=TOP_FALLBACK_TURN_R)
            self._log(f"顶部不可见横杆，回退开环补偿（右转 x{TOP_FALLBACK_TURN_R}）")
        self._log("开始下楼")
        for i in (1, 2):
            self.state.act(A_DOWN)
            self._sleep(DOWN_INTERVAL_S)
            self._climb_bearing_fix(f"下楼第{i}级后")
        self.state.act(A_STAND)

    def _climb_bearing_fix(self, label):
        """台阶上方位检测与小步微调。只用 bearing（横杆低于脚下平面时距离
        读数偏近 ×0.76~0.89，不可作触发依据），返回是否可见"""
        m = self._measure_once()
        if m is None or not m.exists:
            self._log(f"{label}：横杆不可见，跳过方位微调")
            return False
        if abs(m.bearing_err_deg) > TOL_ROT_CLIMB_DEG:
            n = int(math.ceil(abs(m.bearing_err_deg) / TURN_QUANTUM_DEG))
            action = A_TURN_L if m.bearing_err_deg > 0 else A_TURN_R
            self.state.act(action, times=n)
            self._log(f"{label}：方位 {m.bearing_err_deg:+.1f}° -> {action} x{n}"
                      f"（微调标定值 TOL_ROT_CLIMB={TOL_ROT_CLIMB_DEG}°/量子"
                      f"{TURN_QUANTUM_DEG}°，待 M3 实测）")
        else:
            self._log(f"{label}：方位 {m.bearing_err_deg:+.1f}° 在容差内")
        return True

    def _recover(self):
        """条件后退（不盲退）：横杆不可见才退，避免撞上刚走过的台阶"""
        for attempt in range(RECOVER_RETRIES + 1):
            m = self._measure_once()
            if m is not None and m.exists:
                self._log(f"下楼后横杆可见（{m.forward_cm:.1f}cm），无需后退")
                return
            if attempt < RECOVER_RETRIES:
                self._log(f"下楼后横杆不可见，后退 1 小步再试"
                          f"（{attempt + 1}/{RECOVER_RETRIES}）")
                self.state.act(A_BACK, times=1)
        self._log("后退预算耗尽仍不可见，按当前位置继续（不弃赛）")

    # -- 跨杆与出场 ------------------------------------------------------

    def _cross(self):
        self._log("执行跨杆动作组")
        self.state.act(A_HURDLE)
        self.state.act(A_STAND)
        self._sleep(0.5)

    def _exit(self):
        self.state.act(A_FWD, times=EXIT_STEPS)
        self._log(f"出场前进 {EXIT_STEPS} 步（步数待 M3 依杆-终点距离标定）")


def run_level(state: RobotState) -> bool:
    """main.py LEVELS 入口"""
    return StairsHurdleLevel(state).run_level()


tag_poses = {}  # 本关无 AprilTag
