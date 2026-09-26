# -*- coding: utf-8 -*-
"""stairs_hurdle 四步流程端到端仿真测试

复用 sim/stairs_hurdle_sim.py 的合成相机（真实内参+畸变+安装偏移）与带噪声
动作，跑 levels/stairs_hurdle.py 的真实流程。算法零 mock，只重写 I/O 接缝。

覆盖：
- N 个随机种子全流程走完，且真正跨过木条；
- **起跨时脚尖还在木条前面**（碰撞 −2/次、封顶 −6 是唯一罚分来源）；
- 起跨点落在门限内（末段推算的结果）；
- 每步目标唯一性（第1步只有胶条、下楼后只有木条）；
- 楼梯段确实是写死的 4 动作 + 4 右小转，且走完站在平地；
- 动作组净位移与场景几何自洽（两次上楼落在顶部平台、两次下楼落在平地）；
- 降级路径（目标全程不可见）不弃赛。

运行：python tests/test_stairs_hurdle_sim.py [--quick]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from core.ground_line_meter import build_meter
from levels.stairs_hurdle import (
    StairsHurdleLevel, PITCH_OBS, WINDOW_TAPE, WINDOW_BAR_FLAT,
    HURDLE_STOP_CM, HURDLE_TOE_GAP_CM, HURDLE_TOE_GAP_MIN_CM, TOE_AHEAD_CM,
    small_step_nominal_cm, SMALL_STEP_NOMINAL_CM, FWD_STEP_SAFETY,
    LADDER_SAFETY, STAIR_SEQUENCE, HURDLE_FWD_NOMINAL_CM, SKIP_EXIT,
    A_CLIMB, A_DOWN, A_TURN_R, A_STAND,
)
from core import motion_calib as mc
from sim.stairs_hurdle_sim import SimStairsRobot, StairsScene
from vision.red_line_detector import RedLineDetector

CAPTURE_GUARD = 90          # 单局拍照上限（时间代理指标）
N_SEEDS = 20
FINISH_LINE_CM = 45.0 + 50.0    # 楼梯下沿出口 45 + 跨障区 50（说明书画定）
#: 仿真里的脚尖偏移是独立真值，必须与关卡取值一致（不一致说明有一边写错了）
SIM_TOE_AHEAD_CM = SimStairsRobot.TOE_AHEAD_CM
#: 一个整步的保守长度（末段阶梯用完它之后交给小步）
STEP_BOUND_CM = mc.FWD_STEP_CM * FWD_STEP_SAFETY
#: 一个小步的保守长度（阶梯里最小的那级）
SMALL_BOUND_CM = small_step_nominal_cm() * LADDER_SAFETY


def make_run(seed):
    """按种子生成 (scene, robot, level)，模拟"自由摆位但摆得不错"的起点

    额外挂一个钩子：在跨栏**之前**记下机体位置，用来校验"脚尖没有先越过木条"
    （碰撞 −2/次、封顶 −6，是全场唯一的罚分来源，见 test_toe_never_crosses_bar）。
    """
    rng = np.random.RandomState(seed)
    scene = StairsScene(
        bar_dist=float(rng.choice([25.0, 32.0, 40.0])),
        bar_yaw_deg=float(rng.uniform(-2.0, 2.0)),
        tape_yaw_deg=float(rng.uniform(-1.5, 1.5)),
    )
    robot = SimStairsRobot(
        scene, seed=seed,
        start=(float(rng.uniform(-8, 8)), float(rng.uniform(-48, -42))),
        heading_deg=float(rng.uniform(-6, 6)),
    )
    level = StairsHurdleLevel(robot, settle_s=0.0, verbose=False)
    level.trigger_pose = {}
    orig_hurdle = level._step4_hurdle

    def hooked():
        level.trigger_pose["y"] = float(robot.pos[1])
        level.trigger_pose["x"] = float(robot.pos[0])
        return orig_hurdle()

    level._step4_hurdle = hooked
    return scene, robot, level


# ---------------------------------------------------------------------
# 端到端
# ---------------------------------------------------------------------

def test_e2e_seeds():
    n_ok = 0
    worst_cap = 0
    gaps = []
    t0 = time.time()
    for seed in range(N_SEEDS):
        scene, robot, level = make_run(seed)
        ok = level.run_level()
        assert ok, f"seed={seed} 未完成"
        n_ok += 1
        # 起跨点读数落在门限内（光心口径）。
        # 下界 = TOE_AHEAD_CM：读数再低就意味着脚尖已经越过木条（下面还会单独断言）。
        # 上界 = 目标 + 最小步子（走不动时最远就差这么多）。
        tf = level.hurdle_trigger_fwd
        assert tf is not None, f"seed={seed} 未记录起跨读数"
        assert TOE_AHEAD_CM - 0.4 <= tf <= HURDLE_STOP_CM + SMALL_BOUND_CM + 0.4, \
            f"seed={seed} 起跨读数 {tf:.2f}cm 偏离目标 {HURDLE_STOP_CM:.2f}cm 太多"
        # 跨栏之前脚尖必须还在木条**前面**（这是 −2 分碰撞的唯一来源）
        ty = level.trigger_pose["y"]
        toe_gap = scene.bar_y - (ty + SIM_TOE_AHEAD_CM)
        assert toe_gap >= HURDLE_TOE_GAP_MIN_CM - 0.1, \
            f"seed={seed} 起跨时脚尖离木条只剩 {toe_gap:.2f}cm（贴上了会撞杆）"
        assert toe_gap <= HURDLE_TOE_GAP_CM + SMALL_BOUND_CM + 0.3, \
            f"seed={seed} 起跨时脚尖离木条 {toe_gap:.2f}cm，停得太远"
        gaps.append(toe_gap)
        # 必须真正跨过木条（没有停在杆前或杆上）
        assert robot.pos[1] > scene.bar_y + 5.0, \
            f"seed={seed} 终了 y={robot.pos[1]:.1f} 未跨过木条 {scene.bar_y:.1f}"
        # 落点合理性：
        #   SKIP_EXIT=True 时跨完就交棒，落点 = 杆位 + 跨栏净前进 18cm，
        #   木条最远在 85cm 处，所以允许越过终点线一点（那不是错误）；
        #   但不许冲出去太远——那说明步长/位移推算整体偏大。
        #   SKIP_EXIT=False 时才有"必须过终点线"的要求。
        if SKIP_EXIT:
            assert robot.pos[1] < FINISH_LINE_CM + 25.0, \
                f"seed={seed} 终了 y={robot.pos[1]:.1f} 冲得离谱"
        else:
            assert robot.pos[1] >= FINISH_LINE_CM, \
                f"seed={seed} 终了 y={robot.pos[1]:.1f} 未过终点线 {FINISH_LINE_CM:.0f}"
        worst_cap = max(worst_cap, robot.n_captures)
        assert robot.n_captures <= CAPTURE_GUARD, \
            f"seed={seed} 拍照 {robot.n_captures} 张超护栏"
    in_window = sum(1 for g in gaps
                    if HURDLE_TOE_GAP_MIN_CM <= g <= HURDLE_TOE_GAP_CM)
    need = window_quota(N_SEEDS)
    tail = "全部跨过木条" + ("（SKIP_EXIT：落点=杆位+18cm）" if SKIP_EXIT
                            else "并越过终点线")
    print(f"  端到端 {n_ok}/{N_SEEDS} 完成，{tail} ✓")
    print(f"    起跨落点：{in_window}/{N_SEEDS} 落进验收窗口 "
          f"{HURDLE_TOE_GAP_MIN_CM}~{HURDLE_TOE_GAP_CM}cm"
          f"（全部 {min(gaps):.2f}~{max(gaps):.2f}cm）｜"
          f"小步名义 {small_step_nominal_cm():.2f}cm，"
          f"门槛 {need}/{N_SEEDS}｜最多拍照 {worst_cap} 张，"
          f"耗时 {time.time() - t0:.0f}s")
    assert in_window >= need, \
        f"只有 {in_window}/{N_SEEDS} 落进窗口，低于门槛 {need}"


def window_quota(n):
    """落窗配额：**随标定值自动收紧**，不需要人改

    落点是在"可用净位移"的格子上取的，窗口宽
    `win = HURDLE_TOE_GAP_CM − HURDLE_TOE_GAP_MIN_CM = 1.0cm`。
    能拿到多少比例，取决于**最小的可用净位移**（量化步长）：

    - 只有整步/小步：量化 = 小步长。
    - 填了 BACK_STEP_CM：`_back_off` 会搜"k 个小步 − m 个后退"的组合，
      净位移之间能插进约 0.35cm 的细档，量化大幅变细。

    所以小步标定得越短、后退常量填上，门槛自动抬高；标定到量化 ≤ 窗口宽
    就要求 100%。乘 0.6 是给贪心阶梯打折——它不是最优规划，达不到理论上限
    （实测约在上限的 0.6~0.75 倍）。
    """
    win = HURDLE_TOE_GAP_CM - HURDLE_TOE_GAP_MIN_CM
    quantum = small_step_nominal_cm()
    if mc.BACK_STEP_CM:
        quantum = min(quantum, 0.35)     # 组合搜索插出来的细档
    frac = min(1.0, win / quantum)
    return max(n // 3, int(n * frac * 0.6))


def test_back_off():
    """后退调节：标定了 BACK_STEP_CM 之后必须真的用上，且落点更贴窗

    机制：还没到停点、但再往前一步就会冲过窗口时，先退一步再进小步，
    净位移 = 小步 − 后退，比任何单次动作都细一档。这是让落点精度**超过
    最小步长**的唯一办法。

    常量为 None（未标定）时必须**安静地关掉**，不能乱退。
    """
    assert mc.BACK_STEP_CM is None, "本测试假定现场还没标定后退常量"
    # 未标定时：_back_off 一律不动作
    lv = StairsHurdleLevel.__new__(StairsHurdleLevel)
    lv.step_cm = 0.0
    lv.small_step_cm = 0.0
    assert lv._back_off(0.6, 1.0) is False, "未标定后退常量时不该退"

    # 临时标定一个后退值（仿真那边也要同步，否则量的不是同一台机器人）。
    # 取 1.8：`back_one_step` 大致就是整步反过来那么长，是最可能的真值。
    sim_back = 1.8
    mc.BACK_STEP_CM = sim_back
    SimStairsRobot.BACK_STEP_CM_NOMINAL = sim_back
    try:
        used = 0
        gaps = []
        for seed in range(N_SEEDS):
            scene, robot, level = make_run(seed)
            assert level.run_level(), f"seed={seed} 未完成"
            used += sum(1 for a, _ in robot.action_log if a == "back_one_step")
            gap = scene.bar_y - (level.trigger_pose["y"] + SIM_TOE_AHEAD_CM)
            assert gap >= HURDLE_TOE_GAP_MIN_CM - 0.1, \
                f"seed={seed} 后退调节后脚尖离杆只剩 {gap:.2f}cm"
            gaps.append(gap)
    finally:
        mc.BACK_STEP_CM = None
        SimStairsRobot.BACK_STEP_CM_NOMINAL = 3.2
    in_win = sum(1 for g in gaps
                 if HURDLE_TOE_GAP_MIN_CM <= g <= HURDLE_TOE_GAP_CM)
    print(f"  后退调节：后退常量 {sim_back:.2f}cm 时用过 {used} 次，"
          f"落点 {in_win}/{N_SEEDS} 落进窗口（{min(gaps):.2f}~{max(gaps):.2f}cm）"
          f"｜门槛 {window_quota(N_SEEDS)}/{N_SEEDS} ✓")
    assert used > 0, "标定了后退常量却一次都没用上，机制没接上"
    assert in_win >= window_quota(N_SEEDS), \
        f"后退调节只把 {in_win}/{N_SEEDS} 送进窗口，没起到作用"


def test_calibration_plumbing():
    """现场标定值必须"填了就生效"，不需要改别的任何文件

    现场流程是：量小步/后退 → 把 core/motion_calib.py 里的 None 换成数值 →
    直接跑。所以这里把"填了之后会怎样"逐条钉死：

    - `small_step_nominal_cm()` 现读，**不能**在 import 时缓存住旧值
      （缓存了就会出现"填了没生效"，而且看不出来）；
    - 步子阶梯、开场自检日志都跟着变；
    - `BACK_STEP_CM` 没填时后退调节必须**安静关闭**，填了才启用。
    """
    unset_small, unset_back = mc.FWD_SMALL_STEP_CM, mc.BACK_STEP_CM
    assert unset_small is None and unset_back is None, \
        "这条测试假定现场还没标定；等标定值填进 motion_calib 后请删掉本测试"

    class _FakeState:
        def __init__(self):
            self.acts = []

        def act(self, name, times=1):
            self.acts.append((name, times))

    def bare_level():
        lv = StairsHurdleLevel.__new__(StairsHurdleLevel)
        lv.state = _FakeState()
        lv.settle_s = 0.0
        lv.verbose = False
        lv.step_cm = 0.0
        lv.small_step_cm = 0.0
        lv.last_action = None
        return lv

    # —— 未标定：退回名义值，后退调节关闭 ——
    assert small_step_nominal_cm() == SMALL_STEP_NOMINAL_CM, \
        "没标定时应退回名义值"
    lv = bare_level()
    assert abs(lv._ladder()[-1][1] - SMALL_STEP_NOMINAL_CM) < 1e-9
    assert lv._back_off(0.3, 1.0) is False, "未标定后退常量时不该退"
    assert lv.state.acts == [], "未标定时不该发任何动作"

    # —— 填上标定值：立刻生效（现读，不是 import 时缓存）——
    mc.FWD_SMALL_STEP_CM = 0.8
    mc.BACK_STEP_CM = 1.1
    try:
        assert small_step_nominal_cm() == 0.8, \
            "填了小步标定值却没生效——八成是在模块级缓存住了旧值"
        lv = bare_level()
        assert abs(lv._ladder()[-1][1] - 0.8) < 1e-9, "阶梯没用上标定的小步"
        # 自检日志必须把两个数打出来，且不再报"未标定"
        log = []
        lv._log = log.append
        lv._log_calibration()
        line = " ".join(str(x) for x in log)
        assert "0.80cm" in line and "1.10cm" in line, \
            f"开场自检没打出标定值：{line}"
        assert "未标定" not in line, f"自检仍报未标定：{line}"
        # 后退调节启用：小步 0.8、后退 1.1 时，"2 小步 − 1 后退"净前进 0.5
        lv = bare_level()
        assert lv._back_off(0.5, 1.0) is True, "填了后退常量却仍不调节"
        assert lv.state.acts == [("back_one_step", 1),
                                 ("go_forward_one_small_step", 1),
                                 ("go_forward_one_small_step", 1)], \
            f"后退调节动作序列不对：{lv.state.acts}"
    finally:
        mc.FWD_SMALL_STEP_CM = unset_small
        mc.BACK_STEP_CM = unset_back
    assert small_step_nominal_cm() == SMALL_STEP_NOMINAL_CM, "恢复失败"
    print(f"  标定接入：填 FWD_SMALL_STEP_CM=0.8 / BACK_STEP_CM=1.1 后，"
          f"阶梯、自检日志、后退调节全部立刻生效 ✓")


def test_toe_frame_arithmetic():
    """停点口径必须自洽：读数是"光心投影→杆"，不是"脚尖→杆"

    这是最容易搞错、后果最重的一处：
      - 工具读数 = 光心投影 → 木条（相机在头部，投影大致落在脚踝上方）
      - "机器人在栏杆前几厘米" = 脚尖 → 木条
      - 脚尖在光心投影**前方** TOE_AHEAD_CM，所以 读数 = 脚尖距 + TOE_AHEAD_CM

    如果照字面把 HURDLE_STOP_CM 取成 1.0（工具读数），脚尖就已经越过木条
    3.24cm——脚正踩在杆上，一碰 −2 分。
    """
    assert SIM_TOE_AHEAD_CM == TOE_AHEAD_CM, \
        f"仿真真值 {SIM_TOE_AHEAD_CM} 与关卡取值 {TOE_AHEAD_CM} 不一致"
    assert TOE_AHEAD_CM > 0, "脚尖应在光心投影前方（现场实测 +4.24）"
    assert abs(HURDLE_STOP_CM - (HURDLE_TOE_GAP_CM + TOE_AHEAD_CM)) < 1e-9, \
        "HURDLE_STOP_CM 必须是'脚尖离杆距离 + 脚尖偏移'，不能直接用脚尖距离"
    assert HURDLE_STOP_CM > 3.5, \
        "停点必须落在可见带（近界 3.5cm）之内，否则最后一段只能靠航位推算"
    # 若照字面把停点取成"脚尖距离"（1.2），工具读数 1.2 时脚尖已经在杆后 3.04cm
    assert HURDLE_TOE_GAP_CM < TOE_AHEAD_CM, \
        "要求的离杆余量小于脚尖偏移，说明口径换算没做"
    print(f"  停点口径：工具读数 {HURDLE_STOP_CM:.2f}cm = 脚尖离杆窗口上界 "
          f"{HURDLE_TOE_GAP_CM:.1f}cm + 脚尖偏移 {TOE_AHEAD_CM:.2f}cm；"
          f"落在可见带内（>3.5cm），末段无需航位推算 ✓")


# ---------------------------------------------------------------------
# 目标唯一性：胶条与木条不会在同一距离窗口里出现
# ---------------------------------------------------------------------

def _meter_for(robot):
    return build_meter(PITCH_OBS, robot.CAM_HEIGHT,
                       pitch_offset_deg=robot.CAM_PITCH_OFFSET_DEG)


def test_goal_uniqueness():
    """目标不会混淆：第1步全程看不见木条；下楼后胶条已在身后

    这是"按步骤切目标"能成立的前提。注意判别手段是**步骤顺序**而不是窗口
    宽度——下楼后木条在 25cm，落进任何够宽的窗口里；窗口只是第二道保险。
    """
    scene = StairsScene(bar_dist=25.0)
    robot = SimStairsRobot(scene, seed=0, start=(0.0, 0.0), heading_deg=0.0,
                           wobble_sigma_deg=0.0)
    robot.set_pitch(PITCH_OBS)
    meter = _meter_for(robot)
    det = RedLineDetector()

    def measure(window):
        m = meter.measure(det.detect(robot.capture_frame()).components,
                          window_cm=window)
        return m.forward_cm if m.exists else None

    # 第1步全程（起点 -> 起爬点）：木条必须始终不可见，胶条可见。
    # 注意起爬点是 y=-7（离胶条 7cm），不是 y=0——y=0 已经站在胶条上了，
    # 不在实际工作范围内。1100 档远界约 72cm，y=-7 时木条在 78cm，刚好出界。
    probe_ys = (-45.0, -30.0, -20.0, -12.0, -7.0)
    tape_seen = bar_seen = 0
    for y in probe_ys:
        robot.pos = np.array([0.0, y])
        if measure(WINDOW_TAPE) is not None:
            tape_seen += 1
        if measure((60.0, 95.0)) is not None:
            bar_seen += 1
    assert tape_seen >= 4, f"第1步胶条应基本全程可见，实际 {tape_seen}/5"
    assert bar_seen == 0, f"第1步不该看见木条，实际 {bar_seen}/5 次看见"

    # 下楼后：脚下 0~10cm 内没有任何红色目标（胶条在身后），木条在 25cm 处
    robot.pos = np.array([0.0, 45.0])
    near = measure((0.0, 10.0))
    bar = measure(WINDOW_BAR_FLAT)
    assert near is None, f"下楼后脚下不该还有红色目标，实际 {near}"
    assert bar is not None and 20.0 < bar < 30.0, f"下楼后木条读数 {bar} 不合理"
    print(f"  目标唯一性：第1步胶条 {tape_seen}/5 可见且木条 0/5 可见；"
          f"下楼后木条 {bar:.1f}cm、脚下无目标 ✓")


# ---------------------------------------------------------------------
# 楼梯段：写死的动作序列
# ---------------------------------------------------------------------

def test_stair_sequence_hardcoded():
    """第2步必须原样下发写死的序列，并且走完站在平地

    这一段不做视觉：上/下楼动作组之间只夹"右小转"补航向。测试要盯两件事——
    序列没被改（改了就是悄悄换策略），以及走完之后确实回到 z=0 的平地
    （没卡在台阶上）。
    """
    expect = [A_CLIMB, A_TURN_R, A_CLIMB, A_TURN_R,
              A_DOWN, A_TURN_R, A_DOWN, A_TURN_R]
    assert [a for a, _ in STAIR_SEQUENCE] == expect, \
        f"楼梯段序列被改动：{[a for a, _ in STAIR_SEQUENCE]}"
    assert all(n == 1 for _, n in STAIR_SEQUENCE), "楼梯段每个动作都只发 1 次"

    headings = []
    for seed in range(8):
        scene = StairsScene(bar_dist=30.0)
        robot = SimStairsRobot(scene, seed=seed, start=(0.0, 0.0),
                               heading_deg=0.0, wobble_sigma_deg=0.0)
        level = StairsHurdleLevel(robot, settle_s=0.0, verbose=False)
        level._build_meter(PITCH_OBS)
        robot.action_log.clear()
        level._step2_stairs()
        got = [a for a, _ in robot.action_log if a != A_STAND]
        assert got == expect, f"seed={seed} 实际下发 {got}"
        assert abs(robot.z) < 1e-6, f"seed={seed} 楼梯段走完 z={robot.z} 不在平地"
        # 每步位移在仿真里是名义值 ±10%，4 步累计 σ≈2.4cm，所以只要求
        # "确实下来了"；"下来之后木条还看得见"由端到端测试兜。
        assert robot.pos[1] >= 40.0, \
            f"seed={seed} 楼梯段走完 y={robot.pos[1]:.1f} 还在楼梯上"
        headings.append(robot.heading)
    # 写死的 4 次右小转是按现场观察的累计左偏配的；仿真里的偏置是名义值，
    # 只要残余航向还在第3步能修的范围内即可（左转一步 8.6°）。
    worst = max(abs(h) for h in headings)
    assert worst <= 20.0, f"楼梯段走完残余航向 {worst:.1f}° 过大"
    print(f"  楼梯段：8 种子序列完全一致、均落平地 z=0；"
          f"残余航向最大 {worst:.1f}°（写死补偿 vs 仿真名义偏置）✓")


# ---------------------------------------------------------------------
# 动作组净位移与场景几何自洽
# ---------------------------------------------------------------------

def test_action_groups_match_scene():
    """两次上楼必须落在顶部平台、两次下楼必须落到平地。

    旧仿真的 climb_stairs 走 18cm 而踏面只有 15cm，两次上楼会冲过平台
    （y≈36 > 30）——那样测出来的"通过"是在一个不存在的世界里通过的。
    """
    scene = StairsScene(bar_dist=30.0)
    robot = SimStairsRobot(scene, seed=0, start=(0.0, 0.0), heading_deg=0.0,
                           wobble_sigma_deg=0.0)
    for _ in range(2):
        robot.run_action("climb_stairs")
    assert abs(robot.z - 5.0) < 1e-6, f"两次上楼后 z={robot.z} 应为 5"
    y_top = float(robot.pos[1])
    assert 15.0 <= y_top <= 30.0, \
        f"两次上楼后 y={y_top:.1f} 不在顶部平台 [15,30]"
    for _ in range(2):
        robot.run_action("down_floor")
    assert abs(robot.z) < 1e-6, f"两次下楼后 z={robot.z} 应为 0"
    y_flat = float(robot.pos[1])
    assert y_flat >= 45.0, f"两次下楼后 y={y_flat:.1f} 未到平地（≥45）"
    print(f"  动作组自洽：两次上楼 y={y_top:.1f}（平台 [15,30] 内）、"
          f"两次下楼 y={y_flat:.1f}（平地 ≥45）✓")


# ---------------------------------------------------------------------
# 降级路径
# ---------------------------------------------------------------------

def test_degraded_paths():
    """目标全程不可见时也不弃赛：第1步走摆位先验、第3步按先验起跨"""
    cases = [("胶条不可见", dict(tape_visible=False)),
             ("木条不可见", dict(bar_visible=False)),
             ("两个都不可见", dict(tape_visible=False, bar_visible=False))]
    for name, kw in cases:
        scene = StairsScene(bar_dist=30.0, **kw)
        robot = SimStairsRobot(scene, seed=0, start=(0.0, -45.0),
                               heading_deg=0.0)
        level = StairsHurdleLevel(robot, settle_s=0.0, verbose=False)
        ok = level.run_level()
        assert ok, f"{name}：应走降级链完成而不是弃赛"
        assert robot.pos[1] > 45.0, f"{name}：终了 y={robot.pos[1]:.1f} 没走完楼梯段"
        assert robot.n_captures <= CAPTURE_GUARD, f"{name}：拍照超护栏"
    print("  降级路径：胶条/木条/两者全不可见，均不弃赛 ✓")


# ---------------------------------------------------------------------
# 收尾开关
# ---------------------------------------------------------------------

def test_exit_switch():
    """SKIP_EXIT=False 时必须真的走到终点线（开关不能是死代码）

    现场默认 SKIP_EXIT=True（跨完直接交棒），但开关要真的能翻——翻过去之后
    出场步数由 `_exit_steps` 按跨障区几何与实测杆距算，不能再用固定常数。
    """
    import levels.stairs_hurdle as lv
    assert lv.SKIP_EXIT is True, "现场结论是跨完直接切下一关，默认应为 True"
    lv.SKIP_EXIT = False
    try:
        for seed in range(6):
            scene, robot, level = make_run(seed)
            assert level.run_level(), f"seed={seed} 未完成"
            assert robot.pos[1] >= FINISH_LINE_CM, \
                f"seed={seed} 终了 y={robot.pos[1]:.1f} 未过终点线 {FINISH_LINE_CM:.0f}"
            assert robot.pos[1] <= FINISH_LINE_CM + 30.0, \
                f"seed={seed} 终了 y={robot.pos[1]:.1f} 冲过终点线太远"
    finally:
        lv.SKIP_EXIT = True
    print(f"  收尾开关：SKIP_EXIT=False 时 6/6 走到终点线 y≥{FINISH_LINE_CM:.0f} ✓"
          f"（跨栏净前进按实测 {HURDLE_FWD_NOMINAL_CM:.0f}cm 计）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="跳过 20 种子端到端，只跑结构性测试")
    args = ap.parse_args()
    test_goal_uniqueness()
    test_action_groups_match_scene()
    test_stair_sequence_hardcoded()
    test_calibration_plumbing()
    test_toe_frame_arithmetic()
    test_degraded_paths()
    test_exit_switch()
    test_back_off()
    if not args.quick:
        test_e2e_seeds()
    print("全部 stairs_hurdle 端到端测试通过 ✓")
