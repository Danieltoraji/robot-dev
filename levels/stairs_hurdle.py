# -*- coding: utf-8 -*-
"""上下楼梯与识别跨障关卡（levels/stairs_hurdle.py）—— 四步流程重制版

方案文档：docs/关卡算法/上下楼梯与识别跨障-stairs-hurdle/完整方案-2026-09-25-流程重制.md

流程
----
    1 对准并走到起爬点   目标=红胶条（楼梯根部）  闭环 + 末段推算
    2 上下台阶           4 个动作组 + 4 次右小转  整段开环（写死）
    3 对准并走到起跨点   目标=木条                闭环 + 末段推算
    4 跨栏                hurdles                 开环（动作组）

三条设计原则
------------
1. **摆位为主，视觉为辅**：位置与朝向由人摆定，机器人只修"人摆不准"的
   那部分（主要是航向），不做多轮搜索式对正，也不做长距离盲搜。
2. **除动作组执行期间外一律闭环**：上下台阶/跨栏是黑箱动作组，执行期间
   插不进判断；其余每一步都必须有视觉确认。
3. **按步骤指定目标**：胶条与木条同色、同检测、都横跨赛道，靠"距离窗口"
   区分。不用"取最近的红东西"——胶条退到脚边被遮挡时，最近的红东西会
   无声地变成远处的木条，而那正好发生在最不能出错的时刻。

关键数字（全部现场实测，见方案 §2 与 core/motion_calib.py）
----------------------------------------------------------
    pitch 1100、光心离地 33.9cm、安装下俯偏移 19.1°、可见带 3.5~71cm
    前进统一用 go_forward_one_step（实测 1.8cm/步）
    起跨点必须在木条前 1cm 以内
    末段约 3cm 必被机体自遮挡，只能按已标定步长推算

控制约定（右正左负，与 nine_grid/goodluck 一致）
    bearing_err > 0 机体右偏 → turn_left 修正
    lateral > 0 目标在机体右侧 → right_move 对中

运行：python main.py stairs_hurdle
仿真：python tests/test_stairs_hurdle_sim.py
"""

import math
import time

import numpy as np

from core import motion_calib as mc
from core.camera_config import HEAD_CENTER
from core.ground_line_meter import build_meter
from core.robot_core import RobotState
from vision.red_line_detector import RedLineDetector

# =====================================================================
# 相机与观测位（2026-09-25 现场卷尺标定，勿凭公式改）
# =====================================================================
# 标定来源：14 个卷尺刻度（读数 3~60cm），含畸变物理模型 RMS=2.72px。
#   h_eff（光心离刻度平面）= 32.73cm，卷尺厚 λ=1.20cm
#     ⇒ 光心离地 = 32.73 + 1.20 = 33.93cm
#   拟合俯角 55.11°，本档名义 (1500-1100)*0.09 = 36.0°
#     ⇒ 安装下俯偏移 = 55.11 - 36.0 = 19.11°
# 这个 19.11° 与 camera_config 的 18.5°、九宫格自标定的 19.0° 三者吻合。
PITCH_OBS = 1100          # 观测俯仰档（换舵机后重新标定的档位）
CAM_HEIGHT_CM = 33.9      # 光心离地：标定 h_eff 32.73 + 卷尺厚 1.20
PITCH_OFFSET_DEG = 19.1   # 相机相对俯仰舵机的安装下俯偏移（标定值）
HEAD_PULSE = HEAD_CENTER
#: 观测档找不到目标时的重试档（几何随档位重建，精度降级）
FALLBACK_PITCHES = (1050, 1150)

# =====================================================================
# 动作名
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

# =====================================================================
# 现场实测常数（2026-09-25 现场）
# =====================================================================
#: hurdles 动作组的净前进量（cm）——现场实测 18
HURDLE_FWD_NOMINAL_CM = 18.0
#: 摆位先验：机器人开局（人摆好之后）机体中心离胶条的距离（cm）
#:
#: ⚠ **未实测**。取 60cm 的依据：可见带远界约 71cm，摆位必须让胶条在视野里；
#:   同时别贴太近，否则第1步的视觉闭环无事可做。上机时量一次人摆位的实际
#:   距离填进来——这个数只影响**降级路径**（胶条完全看不见时），不影响正常流程。
PLACEMENT_DIST_CM = 60.0

# ⚠ 已弃用 go_forward_one_small_step（现场结论：步长太短、准直性极差）。
# 全程前进统一用 go_forward_one_step，末段推算也用它。
# 步长取自 core/motion_calib.FWD_STEP_CM（现场报 1.8cm）。

# =====================================================================
# 目标距离窗口（按步骤指定，见设计原则 3）
# =====================================================================
# ⚠️ 真正的判别手段是**步骤顺序**（第1步时木条还远在可见带之外；下完台阶后
# 胶条已在身后），窗口是第二道保险：把明显不属于本步的读数挡掉，避免机器人
# 一旦冲过头、木条进入视野时被当成胶条继续往里走。
# 窗口上界不要超过可见带远界（1100 档由标定参数算出约 71cm），否则等于没挡。
WINDOW_TAPE = (0.0, 55.0)        # 第1步：红胶条（楼梯根部）
WINDOW_BAR_FLAT = (0.0, 50.0)    # 第3步：下完台阶后在平地看木条

# =====================================================================
# 停点
# =====================================================================
#: 起爬点：机器人中心距胶条的距离（照参考实现；楼梯段整体开环写死）
CLIMB_STOP_CM = 7.0

# ⚠⚠ 起跨点必须**先换算口径**，否则会站在原地压着木条 ──────────────
# 测距工具读的是「**光心地面投影** → 木条」，而"机器人在栏杆前几厘米"
# 说的是「**脚尖** → 木条」。两者差一个常数：脚尖在光心投影**前方** 4.24cm
# （2026-09-25 卷尺标定 + 现场复核 +4.55~+4.66 恒定）。
#
# 也就是说：**工具读数 1cm 时，脚尖已经越过木条 3.24cm**——脚正踩在杆上，
# 一碰就是 −2 分（封顶 −6）。必须把"脚尖离杆多少"折成工具读数。
#
# 另一个好处：5.44cm 落在可见带（3.5~71cm）**里面**，最后一段不需要
# 航位推算，起跨点的精度回到测量精度（约 0.4cm）。
TOE_AHEAD_CM = 4.24          # 脚尖在光心地面投影前方的距离（标定实测）
#: 起跨点的**目标窗口上界**：脚尖离木条不超过这么多（现场给的验收线 0~1.2cm）。
#:
#: 为什么取窗口**上界**当瞄准点：落点只会比瞄准点更近（步子是往前迈的），
#: 所以瞄准上界，最坏情况也只是贴到窗口下界，不会越过去。
HURDLE_TOE_GAP_CM = 1.2
#: 窗口**下界**：脚尖至少留这么多，别真的贴上杆面。
#: 步子的保守长度总有误差，留 0.2cm 当接触余量——贴到 0 就是碰上了（−2 分）。
HURDLE_TOE_GAP_MIN_CM = 0.2
#: 折成测距工具的读数口径（原点=光心地面投影）：1.2 + 4.24 = 5.44cm
HURDLE_STOP_CM = HURDLE_TOE_GAP_CM + TOE_AHEAD_CM
#: 起爬点允许冲过头的量：动作组是黑箱，本来就吃一定范围的站位差，
#: 比胶条停点近 3cm 以内无所谓（不像木条，近了会撞）。
CLIMB_OVERSHOOT_OK_CM = 3.0
#: 起跨点的"到位放宽量"。**这里必须是 0**。
#:
#: 通用的 CONFIRM_SLACK_CM 是 1cm，但停点窗口总共只有 1.0cm 宽（0.2~1.2），
#: 1cm 的放宽量会让机器人在余量还有 2.2cm 时就宣布"到位"，把整个窗口丢掉
#: （实测丢掉 12/20 个种子）。哪怕只放 0.3cm，也是直接把窗口上界从 1.2
#: 抬到 1.5——seed 1 就是这么以 1.26cm 掉出窗口的。
#: 这点复测噪声由 `_measure_median` 的复测中位数兜，不靠放宽量。
HURDLE_ARRIVE_SLACK_CM = 0.0

# =====================================================================
# 末段的步子阶梯（整步 → 小步）
# =====================================================================
# ⚠ "落点只能落在 1.8cm 的格点上"是个**错的前提**：实测两次量
#   go_forward_one_step 得到 2.652 和 1.8（差 1.47 倍），步长本身就不稳定，
#   根本不存在固定格点。既然不是格点，正确做法就是**能迈多小就迈多小**，
#   而不是"余量不足一整步就不迈"——那等于自己放弃了最后 1~2cm。
#
# 所以末段用阶梯：先把整步用完，再用 go_forward_one_small_step 一点点收。
# 现场说 small_step"步长太短、准直性极差"——那是拿它走**长距离**的结论；
# 最后收 1~2cm 正是它唯一称职的场合（参考实现也是这么用的）。
#: 小步的名义长度（cm）——**只在 core/motion_calib.FWD_SMALL_STEP_CM 没填时**
#: 用这个兜底（老 FSM 的名义值）。
#:
#: 唯一真源是 core/motion_calib.FWD_SMALL_STEP_CM，现场标定填那里就行，
#: 不必改本文件。用的时候**现读**（见 small_step_nominal_cm），不缓存——
#: 免得标定完因为 import 顺序或进程复用而没生效。
SMALL_STEP_NOMINAL_CM = 1.5


def small_step_nominal_cm():
    """小步的名义长度：现场标定值优先，没填时退回 SMALL_STEP_NOMINAL_CM"""
    return mc.FWD_SMALL_STEP_CM or SMALL_STEP_NOMINAL_CM


#: 阶梯用的安全系数。比 FWD_STEP_SAFETY（1.10，远距批量用）小：
#: 批量要防的是"十几步累计走过头"，阶梯防的只是"一步之内冲过窗口下界"，
#: 而实测单步 ±3σ 只有 2.1%。1.10 用在阶梯上会把本来安全又刚好贴窗的一小步
#: 也挡掉——实测因此丢掉 7/20 个种子的落点。
LADDER_SAFETY = 1.05
#: 落点判窗口时的浮点容差（cm）。边界上会出现 `1.6-1.1 = 0.49999999999999994`
#: 这种情形，不加容差会把刚好落在窗口上界的组合判掉。
LANDING_EPS = 1e-9
#: 后退调节的组合上限：一次调节里最多做几个动作（退回 + 小步）。
#: 4 个动作能覆盖"后退 0.9~2.5cm"这一档常见取值；再多就是浪费时间了。
BACK_OFF_MAX_ACTIONS = 4

# =====================================================================
# 楼梯段：整段开环写死（现场结论）
# =====================================================================
# 上/下台阶动作本身会把机体转歪，现场用固定"右小转"逐步补回来，比顶上
# 用视觉测方位更省事也更稳（视觉那一版要额外一帧 + 判据，收益不明显）。
# 顺序：climb-右小转-climb-右小转-down-右小转-down-右小转
STAIR_SEQUENCE = (
    (A_CLIMB, 1), (A_TURN_R, 1),
    (A_CLIMB, 1), (A_TURN_R, 1),
    (A_DOWN, 1), (A_TURN_R, 1),
    (A_DOWN, 1), (A_TURN_R, 1),
)

# =====================================================================
# 收尾开关
# =====================================================================
#: True = 跨栏后**直接结束**（现场直接切下一关，不走离场步）
#: False = 按跨障区几何推算步数走到终点线
SKIP_EXIT = True

# =====================================================================
# 容差与轮数上限
# =====================================================================
#: 航向死区下限（仅用于文档/断言；实际死区按方向取半步长，见 _rot_tol）
TOL_ROT_DEG = mc.ALIGN_TOL_MIN_DEG       # 10.0
#: 航向死区 = 该方向步长的一半 + 本余量（测量噪声）
BEARING_MARGIN_DEG = 1.5
#: 横偏只在"跑偏了"的时候才纠（两个目标都横跨赛道，本来就无需精确居中）
LATERAL_GUARD_CM = 15.0
MAX_ROUNDS = 20
#: 到位复测的放宽量（也用作"算到位"的判据：读数 <= 目标 + 本值即停）
CONFIRM_SLACK_CM = 1.0

#: 远距前进允许"保守批量"，近距一步一测（防冲过）
#: 批量仍然闭环：步数由**实测读数**与**步长上界**共同限定，下一帧复测兜底。
#: 步长上界 = 实测均值 × 安全系数（实测组间极差 0.9cm/20步 = 4.5%）。
FWD_STEP_SAFETY = 1.10
FAR_BATCH_MIN_CM = 8.0
MAX_BATCH_STEPS = 6
#: 比这更近就不再做横偏修正（可见段太短，横向中点不再代表带中心）
LATERAL_CORRECT_MIN_CM = 20.0

# =====================================================================
# 跨障区（出场步数推算）
# =====================================================================
CROSS_ZONE_CM = 50.0        # 楼梯下沿出口 -> 终点线（说明书画定）
EXIT_MARGIN_CM = 15.0       # 再多走一点，保证"整体越过"
EXIT_STEPS_MAX = 20
BAR_DIST_FALLBACK_CM = 32.0  # 未测到杆距时的缺省（25~40 的中值）

# =====================================================================
# 时序
# =====================================================================
SETTLE_S = 0.3
CAPTURE_RETRY = 3
FRAMES_CONFIRM = 3


class StairsHurdleLevel:
    """四步流程主控。state 为 RobotState（或仿真子类）。"""

    def __init__(self, state: RobotState, cam_height_cm=CAM_HEIGHT_CM,
                 pitch_offset_deg=PITCH_OFFSET_DEG, settle_s=SETTLE_S,
                 detector=None, verbose=True):
        self.state = state
        self.cam_height = cam_height_cm
        self.pitch_offset_deg = pitch_offset_deg
        self.settle_s = settle_s
        self.verbose = verbose
        self.detector = detector or RedLineDetector()
        self.meter = None
        self.bar_dist_cm = None        # 下完楼后测到的"楼梯出口 -> 木条"
        self.hurdle_trigger_fwd = None  # 起跨时木条的前向读数（遥测/断言用）
        self.small_step_cm = 0.0       # 本局实测到的最大小步位移（只往上修）
        self.step_cm = 0.0             # 本局实测到的最大整步位移（只往上修）
        self.last_action = None        # 最近一次 _advance 用的动作名

    # ------------------------------------------------------------------
    # 基础设施
    # ------------------------------------------------------------------

    def _log(self, msg):
        if self.verbose:
            print(f"[stairs_hurdle] {msg}")

    def _sleep(self, s=None):
        time.sleep(self.settle_s if s is None else s)

    def _build_meter(self, pitch):
        self.meter = build_meter(pitch, self.cam_height,
                                 pitch_offset_deg=self.pitch_offset_deg)

    def _safe_stand(self):
        try:
            self.state.act(A_STAND)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 测量
    # ------------------------------------------------------------------

    def _capture(self):
        for _ in range(CAPTURE_RETRY):
            frame = self.state.capture_frame()
            if frame is not None:
                return frame
            self._log("拍照失败，重试")
        return None

    def _measure(self, window):
        """单帧：拍照 -> 检测 -> 按窗口筛选 -> 度量。失败返回 None"""
        frame = self._capture()
        if frame is None:
            return None
        det = self.detector.detect(frame)
        if not det.exists:
            return None
        m = self.meter.measure(det.components, window_cm=window)
        return m if m.exists else None

    def _pitch_retry(self, window):
        """变俯仰档重试（降级精度）。几何随档位重建，结束恢复观测档"""
        saved = self.state.current_pitch_pulse
        try:
            for pulse in FALLBACK_PITCHES:
                self.state.set_pitch(pulse)
                self._build_meter(pulse)
                m = self._measure(window)
                if m is not None:
                    self._log(f"俯仰 {pulse} 档找回目标（降级精度）")
                    return m
        finally:
            self.state.set_pitch(saved)
            self._build_meter(saved)
        return None

    def _measure_median(self, window, n=FRAMES_CONFIRM):
        """原地多帧中位数（到位确认用）。返回 (前向, 横偏, 方位) 或 None"""
        fwds, lats, bears = [], [], []
        for _ in range(n):
            m = self._measure(window)
            if m is not None:
                fwds.append(m.forward_cm)
                lats.append(m.lateral_cm)
                bears.append(m.bearing_err_deg)
        if not fwds:
            return None
        return (float(np.median(fwds)), float(np.median(lats)),
                float(np.median(bears)))

    # ------------------------------------------------------------------
    # 动作
    # ------------------------------------------------------------------

    def _rot_tol(self, bearing_deg):
        """该方向上的航向死区 = 半步长 + 测量噪声余量

        死区取**半步长**而不是整步长：一步之后残余 |误差| <= 步长 − 死区
        <= 死区，必然落进死区，不会极限环。若按整步长取（左转就是 10°），
        10° 以内的偏差会被全部放过，机器人带着它走完整段路，横向漂移到
        十几厘米——而那正是"跑到木条旁边"的原因。
        """
        step = (mc.TURN_LEFT_DEG if bearing_deg > 0 else mc.TURN_RIGHT_DEG)
        return step / 2.0 + BEARING_MARGIN_DEG

    def _correct(self, m):
        """航向与横偏修正：**一次只走一步**，复测由调用方负责

        实测左转一步就有 8.625°、且组间极差大（±7%），批量 ceil(误差/步长)
        会过冲。一步一测同时还顺带在线观测了真实步长。

        横偏只纠"跑偏了"的大偏差：两个目标都横跨赛道（胶条 40cm、木条
        80cm），精确居中本来就无必要；而且目标近/被画幅裁切时"可见段中点"
        不再代表带的中心，读数会乱跳，横移还带一点后退分量，越纠越远。
        """
        if abs(m.bearing_err_deg) > self._rot_tol(m.bearing_err_deg):
            action = A_TURN_L if m.bearing_err_deg > 0 else A_TURN_R
            self.state.act(action, times=1)
            self._log(f"  航向 {m.bearing_err_deg:+.1f}° -> {action} x1")
            return True
        if (abs(m.lateral_cm) > LATERAL_GUARD_CM
                and m.forward_cm > LATERAL_CORRECT_MIN_CM):
            action = A_RIGHT if m.lateral_cm > 0 else A_LEFT
            self.state.act(action, times=1)
            self._log(f"  横偏 {m.lateral_cm:+.1f}cm -> {action} x1")
            return True
        return False

    def _scale_est(self):
        """本局步长相对名义值的缩放（由实测整步推得）

        实测两次量 go_forward_one_step 得到 2.652 与 1.8（差 1.47 倍），说明
        **局间**步长会整体变。整步和小步受同一个缩放影响，所以一旦量到整步，
        就能把**小步**的估计一起校正——小步往往到末段才第一次走，靠它自己
        实测已经太晚（第一小步的决策就是靠这个估计做的）。
        """
        if self.step_cm <= 0:
            return 1.0
        return max(0.5, self.step_cm / mc.FWD_STEP_CM)

    def _ladder(self):
        """步子阶梯：(动作, 名义长度, 保守上界)，从大到小

        名义长度用于"预计落点会不会落进窗口"，保守上界用于"最坏情况会不会
        冲过窗口下界"。两者必须分开：用上界去预测落点会一律偏悲观，把本来
        安全又刚好贴窗的一小步也挡掉（实测挡掉 7/20 个种子的落点）。
        """
        f_nom = max(mc.FWD_STEP_CM, self.step_cm)
        s_nom = max(small_step_nominal_cm() * self._scale_est(),
                    self.small_step_cm)
        return ((A_FWD, f_nom, f_nom * LADDER_SAFETY),
                (A_FWD_SMALL, s_nom, s_nom * LADDER_SAFETY))

    def _full_bound(self):
        """整步的批量上界（远距批量用，宁可欠走）"""
        return max(mc.FWD_STEP_CM, self.step_cm) * FWD_STEP_SAFETY

    def _observe_step(self, advance_cm, steps):
        """用一次前进前后的读数差标定步长（只记录，取运行最大值）

        只往上修：实测到更长的步子立刻变保守，绝不因为一次巧合把上界调小。
        """
        if steps <= 0 or advance_cm <= 0:
            return
        per = advance_cm / steps
        if per > self.step_cm:
            self._log(f"  整步实测 {per:.2f}cm/步"
                      f"（上界基准 {self.step_cm:.2f} → {per:.2f}）")
            self.step_cm = per

    def _observe_small_step(self, advance_cm):
        """用一次小步前后的读数差标定小步长度（只记录，取运行最大值）"""
        if advance_cm <= 0:
            return
        if advance_cm > self.small_step_cm:
            self._log(f"  小步实测 {advance_cm:.2f}cm"
                      f"（上界基准 {self.small_step_cm:.2f} → {advance_cm:.2f}）")
            self.small_step_cm = advance_cm

    def _advance(self, forward_cm, stop_cm, bias="short", overshoot_ok_cm=0.0):
        """前进一步或一批；返回实际走的步数（0 = 走不动了）

        远距"保守批量"不是开环：步数由**实测读数**除以**步长上界**得到
        （用上界 => 宁可欠走），下一帧立刻复测兜底。近距（余量 < 8cm）
        退回一步一测。

        近距用**步子阶梯**，两级决策：
        1. 先找"预计落点正好落在窗口里"的步子（从大到小），有就迈；
        2. 都不在窗口里，就迈"最坏情况也安全"的最大的那一步，先靠近再说；
        3. 连最小的步子都不安全，就地停。

        `overshoot_ok_cm` 是"冲过停点多少还算无害"（起跨点 = 窗口上界 − 下界；
        起爬点 = 黑箱动作组的站位宽容度）。

        ⚠ 旧版这里是"余量不足一个整步就不迈步"，理由写的是"落点只能落在
        1.8cm 的格点上"。**那个前提是错的**：实测两次量 go_forward_one_step
        得到 2.652 和 1.8，步长本身就不稳定，没有固定格点。既然不是格点，
        能迈多小就该迈多小，否则等于白白放弃最后 1~2cm。

        ⚠ 整步仍然全程用于**长距离**（这才是"small_step 准直性极差"的适用
        场合：拿它走几十厘米会越走越偏）。小步只在末段收最后一点余量时用，
        这也正是参考实现的用法。
        """
        room = forward_cm - stop_cm
        if room <= 0:
            return 0
        if room > FAR_BATCH_MIN_CM:
            steps = max(1, min(MAX_BATCH_STEPS,
                               int(room / self._full_bound())))
            self.state.act(A_FWD, times=steps)
            self._sleep()
            self.last_action = A_FWD
            self._log(f"  远距批量前进 {steps} 步（余 {room:.1f}cm）")
            return steps
        if bias != "short":
            self.state.act(A_FWD, times=1)
            self._sleep()
            self.last_action = A_FWD
            return 1
        ladder = self._ladder()
        # 1) 预计落点落在窗口里的步子，优先（从大到小）
        for action, nom, bound in ladder:
            if room + overshoot_ok_cm < bound:
                continue
            landing = room - nom            # 落点相对停点的余量
            if -overshoot_ok_cm <= landing <= LANDING_EPS:
                self._go(action, room, nom, window_hit=True)
                return 1
        # 2) 都不在窗口里，就迈最坏情况也安全的最大的那一步
        for action, nom, bound in ladder:
            if room + overshoot_ok_cm >= bound:
                self._go(action, room, nom, window_hit=False)
                return 1
        # 3) 前进走不动了：试试**后退调节**
        if self._back_off(room, overshoot_ok_cm):
            return 1
        self._log(f"  余 {room:.2f}cm 已小于最小步上界 "
                  f"{ladder[-1][2]:.2f}cm，不再前进")
        return 0

    def _back_off(self, room, overshoot_ok_cm):
        """后退调节：前进走不动时，用"退 + 进小步"的组合把落点凑进窗口

        单退一步再进一小步，净位移是 `小步 − 后退`。**只有当后退比小步短**
        它才是一档更细的步子；如果后退更长（很可能，`back_one_step` 大约就是
        整步反过来那么长），单步组合毫无用处。所以这里搜的是**小组合**：

            净前进 = k × 小步 − m × 后退      （k ≥ 1, m ≥ 1）

        这些净位移一般和"整步/小步"互质，凑起来等于凭空多出好几档细步子。
        例：小步 1.5、后退 1.8 时可选净位移 {0.9, 1.2, 1.5, 2.7}——
        相当于量化步长从 1.5cm 降到 0.3cm，1.0cm 宽的窗口就基本必中。

        安全性：**先退后进**。退完的位置离目标更远，随后每一步前进都比最终
        落点更远，所以只要最终落点不越过窗口下界，中间任何一步都不会。
        整段组合走完再复测一次，读数说了算。

        `BACK_STEP_CM` 没标定（None）时直接返回 False = 关闭本功能。
        """
        back = mc.BACK_STEP_CM
        if not back or back <= 0:
            return False
        # 后退也是同一套腿部动作，本局步长缩放同样作用于它——不折算的话
        # 预测的净位移会整体偏，搜出来的组合落不到窗口里（实测就是这么
        # 20 个种子里只触发 2 次的）。
        back = back * self._scale_est()
        _, s_nom, _ = self._ladder()[-1]
        # 动作最少优先，其次净位移最大；k 上限 3、m 上限 2 覆盖
        # "后退 0.9~2.5cm" 这一档常见取值
        best = None
        for total in range(2, BACK_OFF_MAX_ACTIONS + 1):
            for k in range(1, min(3, total - 1) + 1):
                m = total - k
                if m < 1 or m > 2:
                    continue
                net = k * s_nom - m * back
                if net <= 0:
                    continue
                landing = room - net
                # 容一点浮点误差：边界上有 "1.6-1.1 = 0.49999999999999994"
                # 这种情形，不加容差会把刚好落在窗口下界的组合判掉
                if -overshoot_ok_cm <= landing <= LANDING_EPS:
                    best = (k, m, net, landing)
                    break
            if best:
                break
        if best is None:
            return False
        k, m, net, landing = best
        self._log(f"  余 {room:.2f}cm 前进走不动，后退调节：退 {m} 步"
                  f"（{m * back:.2f}cm）再进小步 {k} 次"
                  f"（净前进约 {net:.2f}cm），预计落点 {landing:+.2f}cm")
        for _ in range(m):
            self.state.act(A_BACK, times=1)
            self._sleep()
        for _ in range(k):
            self.state.act(A_FWD_SMALL, times=1)
            self._sleep()
        # 组合的净位移不等于单步，**不能**拿它去标定小步长度
        self.last_action = "back_off"
        return True

    def _go(self, action, room, nom, window_hit):
        self.state.act(action, times=1)
        self._sleep()
        self.last_action = action
        self._log(f"  {'小步' if action == A_FWD_SMALL else '整步'}前进 1 次"
                  f"（余 {room:.2f}cm，预计落点 {room - nom:+.2f}cm"
                  f"{'，贴窗' if window_hit else ''}）")

    def _dead_reckon(self, distance_cm, label, bias="short"):
        """按标定步长推算走完末段

        bias='short'：宁可少走（起爬点少走只是跨得早；走过头就是踢杆）。
        先用整步吃掉大部分，余量交给小步收尾（小步不参与远距推算——
        它准直性差，只适合最后一点点）。
        """
        if distance_cm <= 0:
            return 0
        small = small_step_nominal_cm()
        big = max(0.0, distance_cm - small)
        steps = int(math.floor(big / mc.FWD_STEP_CM)) if big > 0 else 0
        if steps > 0:
            self.state.act(A_FWD, times=steps)
            self._sleep()
            self._log(f"{label}：推算前进 {steps} 步（约 "
                      f"{steps * mc.FWD_STEP_CM:.1f}cm / 目标 {distance_cm:.1f}cm）")
        rest = distance_cm - steps * mc.FWD_STEP_CM
        if rest > small * 0.5:
            self.state.act(A_FWD_SMALL, times=1)
            self._sleep()
            self.last_action = A_FWD_SMALL
            self._log(f"{label}：余 {rest:.1f}cm 补一小步")
            return steps + 1
        if steps <= 0:
            self._log(f"{label}：余量 {distance_cm:.1f}cm 不足一步，不再走")
        return steps

    # ------------------------------------------------------------------
    # 第1步／第5步共用的闭环接近
    # ------------------------------------------------------------------

    def _approach(self, window, stop_cm, label, bias="short", max_misses=3,
                  overshoot_ok_cm=0.0, arrive_slack_cm=None):
        """边走边修，直到读数 <= stop_cm（+ 到位放宽量）并复测确认

        到位判定放在修正之前：近距时"可见段中点"因为只剩一小截而变得不可信，
        此时再去纠正横偏只会来回摆。

        目标连续丢失（多半是进入机体自遮挡盲区）时，用最后有效读数推算余量。
        走到阶梯里最小的步子也放不下时（_advance 返回 0）就地接受。

        步长/小步的实测位移都会被回读利用（_observe_step / _observe_small_step），
        上界只往上修——本局走出来的步子比名义值长，后面就按长的算。

        `arrive_slack_cm` 是"到位放宽量"。默认 CONFIRM_SLACK_CM（1cm），但
        **起跨点绝不能用 1cm**：停点窗口总共只有 1.2cm 宽，1cm 的放宽量会让
        机器人在余量还有 2.2cm 时就宣布到位，白白丢掉整个窗口（实测就是这么
        丢掉 12/20 个种子的）。

        返回 True=到位/已按推算处理/到步子极限，False=始终未见目标。
        """
        slack = CONFIRM_SLACK_CM if arrive_slack_cm is None else arrive_slack_cm
        last = None
        misses = 0
        prev = None            # (读数, 动作名, 步数)：上一次前进之后的状态
        for _ in range(MAX_ROUNDS):
            m = self._measure(window)
            if m is None and misses == 0:
                # 只在一次丢失的开头试变档（变档要动舵机，省时间）
                m = self._pitch_retry(window)
            if m is None:
                misses += 1
                if misses < max_misses:
                    self._log(f"{label}：未见目标（{misses}/{max_misses}）重试")
                    self._sleep()
                    continue
                if last is None:
                    self._log(f"{label}：连续 {misses} 次未见目标")
                    return False
                self._log(f"{label}：目标连续丢失 {misses} 次"
                          "（多半进入机体自遮挡）")
                self._dead_reckon(max(0.0, last - stop_cm), label, bias)
                return True
            misses = 0
            last = m.forward_cm
            if prev is not None:
                advanced = prev[0] - m.forward_cm
                if prev[1] == A_FWD_SMALL:
                    self._observe_small_step(advanced)
                elif prev[1] == A_FWD:
                    self._observe_step(advanced, prev[2])
            prev = None
            self._log(f"{label}：前向 {m.forward_cm:.1f}cm "
                      f"方位 {m.bearing_err_deg:+.1f}° "
                      f"横偏 {m.lateral_cm:+.1f}cm")
            if m.forward_cm <= stop_cm + slack:
                conf = self._measure_median(window)
                c = conf[0] if conf is not None else m.forward_cm
                if c <= stop_cm + slack:
                    self._log(f"{label}：到位（复测中位 {c:.1f}cm）")
                    return True
                self._log(f"{label}：复测 {c:.1f}cm 未达，继续")
                if self._advance(c, stop_cm, bias, overshoot_ok_cm) == 0:
                    self._log(f"{label}：最小步子也放不下，就此处停")
                    return True
                prev = (c, self.last_action, 1)
                continue
            if self._correct(m):
                self._sleep()
                continue
            n = self._advance(m.forward_cm, stop_cm, bias, overshoot_ok_cm)
            if n == 0:
                self._log(f"{label}：最小步子也放不下，就此处停")
                return True
            prev = (m.forward_cm, self.last_action, n)
        self._log(f"{label}：轮数耗尽，按当前位置继续")
        return True

    def _approach_prior(self, window, stop_cm, label, prior_cm,
                        overshoot_ok_cm=0.0, arrive_slack_cm=None):
        """闭环接近；目标始终不见时，按摆位先验盲走——但**边走边接着找**

        "盲走"只是起点，不是一路撞到底：每走一小批就再拍一帧，目标一旦重新
        出现立刻交回闭环。摆位先验估错时，错得越多越早看见目标，所以这个
        兜底的风险是有界的。

        ⚠ 与旧版的区别：旧版"未见目标就走 5 大步"物理意图不明（13cm，既不
        是走完全程也不是有意义的一小段），实际效果是机器人停在离楼梯 30cm
        开外，第2步的动作组够不着、第3步连木条都看不见。改成"按先验走到
        起爬点"之后，降级路径才真的走完关卡。
        """
        if self._approach(window, stop_cm, label, overshoot_ok_cm=overshoot_ok_cm,
                          arrive_slack_cm=arrive_slack_cm):
            return
        self._log(f"{label}：目标始终未见，改按摆位先验走"
                  f"（先验 {prior_cm:.0f}cm - 停点 {stop_cm:.1f}cm）")
        left = prior_cm - stop_cm
        while left > FAR_BATCH_MIN_CM:
            n = max(1, min(MAX_BATCH_STEPS, int(left / self._full_bound())))
            self.state.act(A_FWD, times=n)
            self._sleep()
            left -= n * mc.FWD_STEP_CM
            self._log(f"{label}：先验盲走 {n} 步（余 {max(0.0, left):.1f}cm）")
            if self._measure(window) is not None:
                self._log(f"{label}：目标重新出现，交回闭环")
                self._approach(window, stop_cm, label,
                               overshoot_ok_cm=overshoot_ok_cm,
                               arrive_slack_cm=arrive_slack_cm)
                return
        self._dead_reckon(left, label)

    # ------------------------------------------------------------------
    # 四个步骤
    # ------------------------------------------------------------------

    def _step1_to_climb_point(self):
        self._log("第1步：对准并走到起爬点（目标=红胶条）")
        self._approach_prior(WINDOW_TAPE, CLIMB_STOP_CM, "接近楼梯",
                             PLACEMENT_DIST_CM,
                             overshoot_ok_cm=CLIMB_OVERSHOOT_OK_CM)

    def _step2_stairs(self):
        """第2步：上下台阶——**整段开环写死**

        顺序：climb-右小转-climb-右小转-down-右小转-down-右小转。

        为什么不用视觉：现场结论是这套固定补偿更省事也更稳。上楼/下楼动作
        本身会把机体转歪，"逐动作右小转"就是补它；中间插一帧测方位要额外
        拍照 + 判据，收益不明显，还多一个失败点。
        """
        self._log(f"第2步：上下台阶（开环写死 {len(STAIR_SEQUENCE)} 个动作）")
        for i, (action, times) in enumerate(STAIR_SEQUENCE, 1):
            self.state.act(action, times=times)
            self._sleep()
            self._log(f"  {i}/{len(STAIR_SEQUENCE)} {action} x{times}")
        self.state.act(A_STAND)
        self._sleep()

    def _step3_to_hurdle_point(self):
        """第3步：对准并走到起跨点（目标=木条，方向与距离都用）

        这一段是全场罚分（碰撞，封顶 −6）的唯一来源，测量冗余度最高。
        现场验收线：**脚尖离木条 0.2 ~ 1.2cm**（见 HURDLE_TOE_GAP_*）。

        末段用步子阶梯贴窗：先整步靠近，再用小步收最后一点余量，
        所以落点能贴进窗口，而不是停在 2~3cm 外。
        """
        self._log("第3步：对准并走到起跨点（目标=木条）")
        m0 = self._measure(WINDOW_BAR_FLAT)
        if m0 is not None:
            self.bar_dist_cm = m0.forward_cm
            self._log(f"木条距台阶出口 {self.bar_dist_cm:.1f}cm")
        self._approach_prior(WINDOW_BAR_FLAT, HURDLE_STOP_CM, "接近木条",
                             self.bar_dist_cm or BAR_DIST_FALLBACK_CM,
                             overshoot_ok_cm=(HURDLE_TOE_GAP_CM
                                              - HURDLE_TOE_GAP_MIN_CM),
                             arrive_slack_cm=HURDLE_ARRIVE_SLACK_CM)
        conf = self._measure_median(WINDOW_BAR_FLAT, n=2)
        self.hurdle_trigger_fwd = (conf[0] if conf is not None
                                   else HURDLE_STOP_CM)
        toe_gap = self.hurdle_trigger_fwd - TOE_AHEAD_CM
        ok = HURDLE_TOE_GAP_MIN_CM <= toe_gap <= HURDLE_TOE_GAP_CM
        self._log(f"起跨点就位：读数 {self.hurdle_trigger_fwd:.2f}cm"
                  f"（光心口径）⇒ 脚尖离木条 {toe_gap:.2f}cm，"
                  f"目标 {HURDLE_TOE_GAP_MIN_CM:.1f}~{HURDLE_TOE_GAP_CM:.1f}cm"
                  f" ⇒ {'达标' if ok else '⚠ 超窗'}")
        if toe_gap < 0.0:
            self._log("⚠ 脚尖已越过木条（负值）——读数口径可能搞反了，"
                      "检查 TOE_AHEAD_CM 的符号")

    def _step4_hurdle(self):
        """第4步：跨栏（+ 可选离场）

        跨栏动作组实测净前进 18cm，比原来估的 10cm 大一倍——所以跨完之后
        到终点常常已经没剩多少路了。
        """
        self._log("第4步：跨栏")
        self.state.act(A_HURDLE)
        self.state.act(A_STAND)
        self._sleep(0.5)
        if SKIP_EXIT:
            self._log("收尾开关 SKIP_EXIT=True：跨栏后直接结束（现场切下一关）")
            return
        steps = self._exit_steps()
        if steps > 0:
            self.state.act(A_FWD, times=steps)
            self._log(f"出场前进 {steps} 步")
        self._sleep()

    def _exit_steps(self):
        """出场步数由跨障区几何与实测杆距推算，不用固定常数

        跨障区自台阶出口起 50cm；木条在实测距离 bar 处，起跨点在木条前
        HURDLE_STOP_CM，跨栏动作本身再带人前进 HURDLE_FWD_NOMINAL_CM。

        步长用**本局实测值**（取名义值与实测的较大者）：局间步长会整体漂移
        （实测两次量差 1.47 倍），拿名义值算步数会成比例地走不到位。
        多走一点没有代价，走不到终点线才是白干。
        """
        bar = self.bar_dist_cm if self.bar_dist_cm else BAR_DIST_FALLBACK_CM
        advanced = (bar - HURDLE_STOP_CM) + HURDLE_FWD_NOMINAL_CM
        remaining = (CROSS_ZONE_CM + EXIT_MARGIN_CM) - advanced
        steps = int(math.ceil(remaining / max(mc.FWD_STEP_CM, self.step_cm)))
        return int(max(0, min(EXIT_STEPS_MAX, steps)))

    # ------------------------------------------------------------------
    # 入口
    # ------------------------------------------------------------------

    def _servo(self, fn_name, pulse):
        """优先强制下发舵机脉冲；宿主没有 force 参数时退回普通调用

        为什么要强制：RobotState.__init__ 把 current_head_pulse /
        current_pitch_pulse 预设成 1500，**并不代表舵机真的在那里**。
        刚上电、刚换过舵机、被手掰过时，普通 set_* 会因为"已在目标位"直接
        return，一次脉冲都发不出去——后面所有几何全错，而且看不出来。
        """
        fn = getattr(self.state, fn_name, None)
        if fn is None:
            return
        try:
            fn(pulse, force=True)
        except TypeError:
            fn(pulse)

    def _log_calibration(self):
        """开场常数自检——一眼看出标定值有没有被吃进来

        现场标定完最怕的是"填了但没生效"（拼错名字、填错文件、进程复用旧值）。
        所以这里把末段用到的三个数全打出来，没标定的直接点名去填哪一行。
        """
        fwd_small = mc.FWD_SMALL_STEP_CM
        back = mc.BACK_STEP_CM
        parts = [f"整步 {mc.FWD_STEP_CM:.2f}cm"]
        parts.append(
            f"小步 {fwd_small:.2f}cm" if fwd_small
            else f"小步 {SMALL_STEP_NOMINAL_CM:.2f}cm（⚠ 未标定 · 名义值："
                 f"填 core/motion_calib.py 的 FWD_SMALL_STEP_CM）")
        parts.append(
            f"后退 {back:.2f}cm" if back
            else "后退 —（未标定 · 后退调节关闭："
                 "填 core/motion_calib.py 的 BACK_STEP_CM）")
        parts.append(f"脚尖偏移 {TOE_AHEAD_CM:.2f}cm")
        self._log("常数自检：" + " | ".join(parts))

    def _init_pose(self):
        """开场初始化：站直 → 头部 yaw 回中位 → 俯仰到观测档

        顺序不能反：先回中位再定俯仰，两者都强制下发（见 _servo）。
        换舵机之后这一步尤其不能省。
        """
        self.state.act(A_STAND)
        self._servo("set_head", HEAD_PULSE)
        self._servo("set_pitch", PITCH_OBS)
        self._log(f"头部初始化：yaw={HEAD_PULSE}（中位）pitch={PITCH_OBS}（观测档）")
        self._build_meter(PITCH_OBS)
        self._log_calibration()
        self._sleep(0.5)

    def _finish(self, ok, why=""):
        self._log(("完成" if ok else "失败") + (f"：{why}" if why else ""))
        return ok

    def run_level(self):
        try:
            self._init_pose()
            self._step1_to_climb_point()
            self._step2_stairs()
            self._step3_to_hurdle_point()
            self._step4_hurdle()
            return self._finish(True)
        except Exception as e:  # 任何异常都落站立，不留倒姿
            self._log(f"异常：{type(e).__name__}: {e}")
            self._safe_stand()
            return self._finish(False, f"异常退出: {e}")


def run_level(state: RobotState) -> bool:
    """main.py LEVELS 入口"""
    return StairsHurdleLevel(state).run_level()


tag_poses = {}  # 本关无 AprilTag
