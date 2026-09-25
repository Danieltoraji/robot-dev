# -*- coding: utf-8 -*-
"""stairs_hurdle 六步流程端到端仿真测试

复用 sim/stairs_hurdle_sim.py 的合成相机（真实内参+畸变+安装偏移）与带噪声
动作，跑 levels/stairs_hurdle.py 的真实流程。算法零 mock，只重写 I/O 接缝。

覆盖：
- N 个随机种子全流程走完，且越过终点线；
- 起跨点落在门限内（末段推算的结果）；
- 每步目标唯一性（第1步只有胶条、下楼后只有木条）；
- 第3步顶上对正能修掉上楼带来的航向偏差；
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
    WINDOW_BAR_TOP, HURDLE_STOP_CM, CONFIRM_SLACK_CM, TOL_ROT_DEG,
)
from sim.stairs_hurdle_sim import SimStairsRobot, StairsScene
from vision.red_line_detector import RedLineDetector

CAPTURE_GUARD = 90          # 单局拍照上限（时间代理指标）
N_SEEDS = 20
FINISH_LINE_CM = 45.0 + 50.0    # 楼梯下沿出口 45 + 跨障区 50（说明书画定）


def make_run(seed):
    """按种子生成 (scene, robot, level)，模拟"自由摆位但摆得不错"的起点"""
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
    return scene, robot, level


# ---------------------------------------------------------------------
# 端到端
# ---------------------------------------------------------------------

def test_e2e_seeds():
    n_ok = 0
    worst_cap = 0
    t0 = time.time()
    for seed in range(N_SEEDS):
        scene, robot, level = make_run(seed)
        ok = level.run_level()
        assert ok, f"seed={seed} 未完成"
        n_ok += 1
        # 起跨点：末段推算应落在门限内（偏差偏"早"是安全方向）
        tf = level.hurdle_trigger_fwd
        assert tf is not None, f"seed={seed} 未记录起跨读数"
        assert 0.0 <= tf <= HURDLE_STOP_CM + CONFIRM_SLACK_CM + 0.5, \
            f"seed={seed} 起跨读数 {tf:.1f}cm 超门限"
        # 越过终点线
        assert robot.pos[1] >= FINISH_LINE_CM, \
            f"seed={seed} 终了 y={robot.pos[1]:.1f} 未过终点线 {FINISH_LINE_CM:.0f}"
        # 必须真正跨过木条（没有停在杆前或杆上）
        assert robot.pos[1] > scene.bar_y + 5.0, \
            f"seed={seed} 终了 y={robot.pos[1]:.1f} 未跨过木条 {scene.bar_y:.1f}"
        worst_cap = max(worst_cap, robot.n_captures)
        assert robot.n_captures <= CAPTURE_GUARD, \
            f"seed={seed} 拍照 {robot.n_captures} 张超护栏"
    print(f"  端到端 {n_ok}/{N_SEEDS} 完成，起跨点全落门限，全部越过终点线 ✓"
          f"（最多拍照 {worst_cap} 张，耗时 {time.time() - t0:.0f}s）")


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
# 第3步顶上对正
# ---------------------------------------------------------------------

def test_step3_corrects_drift():
    """上楼会把人带歪；第3步必须用木条方向把它修进容差"""
    ok_cnt = 0
    for seed in range(8):
        scene = StairsScene(bar_dist=float([25.0, 32.0, 40.0][seed % 3]))
        robot = SimStairsRobot(scene, seed=seed, start=(0.0, 0.0),
                               heading_deg=0.0, wobble_sigma_deg=0.0)
        robot.set_pitch(PITCH_OBS)
        # 摆到顶部平台，并注入一个明显的航向偏差（模拟上楼带歪）
        robot.pos = np.array([0.0, 24.0])
        robot.z = 5.0
        robot.heading = -18.0 + seed
        level = StairsHurdleLevel(robot, settle_s=0.0, verbose=False)
        level._build_meter(PITCH_OBS)
        before = robot.heading
        level._step3_align_on_top()
        after = robot.heading
        assert abs(after) <= TOL_ROT_DEG, \
            f"seed={seed} 顶上对正后航向 {after:+.1f}° 仍超容差 {TOL_ROT_DEG}"
        assert abs(after) < abs(before), \
            f"seed={seed} 顶上对正没起作用：{before:+.1f}° -> {after:+.1f}°"
        ok_cnt += 1
    print(f"  顶上对正：{ok_cnt}/8 把注入的航向偏差修进容差 ±{TOL_ROT_DEG:.0f}° ✓")


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
    """目标全程不可见时也不弃赛：第1步走摆位先验、第5步按先验起跨"""
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
        assert robot.n_captures <= CAPTURE_GUARD, f"{name}：拍照超护栏"
    print("  降级路径：胶条/木条/两者全不可见，均不弃赛 ✓")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="跳过 20 种子端到端，只跑结构性测试")
    args = ap.parse_args()
    test_goal_uniqueness()
    test_action_groups_match_scene()
    test_step3_corrects_drift()
    test_degraded_paths()
    if not args.quick:
        test_e2e_seeds()
    print("全部 stairs_hurdle 端到端测试通过 ✓")
