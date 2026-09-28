# -*- coding: utf-8 -*-
"""stairs_hurdle FSM 端到端仿真测试

复用 sim/stairs_hurdle_sim.py 的合成相机（真实内参+畸变）+ 打滑动作模型，
跑 levels/stairs_hurdle.py 的真实 FSM。算法零 mock，只重写 I/O 接缝。

验收线（方案 §4）：
- 20 个随机种子全流程到达 DONE；
- 跨杆起跨距离全部落在可见窗口内；
- 爬楼段方位微调后航向收敛；
- 降级路径（横杆全程不可见）不弃赛可达 DONE；
- 拍照张数护栏（时间代理指标）。

运行：python -m tests.test_stairs_hurdle_sim [--quick]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from core.ground_line_meter import min_visible_ground_cm
from levels.stairs_hurdle import (
    StairsHurdleLevel, PITCH_OBS, CAM_HEIGHT_CM, CONFIRM_SLACK_CM,
)
from sim.stairs_hurdle_sim import SimStairsRobot, StairsScene

CAPTURE_GUARD = 120  # 单局拍照上限（时间代理指标）


def make_run(seed):
    """按种子生成 (scene, robot, level)"""
    rng = np.random.RandomState(seed)
    scene = StairsScene(
        bar_dist=float(rng.choice([25.0, 32.0, 40.0])),
        bar_yaw_deg=float(rng.uniform(-3.0, 3.0)),
        tape_yaw_deg=float(rng.uniform(-2.0, 2.0)),
    )
    robot = SimStairsRobot(
        scene, seed=seed,
        start=(float(rng.uniform(-12, 12)), float(rng.uniform(-70, -55))),
        heading_deg=float(rng.uniform(-15, 15)),
    )
    level = StairsHurdleLevel(robot, settle_s=0.0, total_budget_s=None,
                              verbose=False)
    return scene, robot, level


def test_e2e_20_seeds():
    d_min = min_visible_ground_cm(CAM_HEIGHT_CM, PITCH_OBS)
    n_ok = 0
    t0 = time.time()
    for seed in range(20):
        scene, robot, level = make_run(seed)
        ok = level.run_level()
        assert ok, f"seed={seed} 未完成：trigger_fwd={getattr(level, 'trigger_fwd', None)}"
        n_ok += 1
        # 跨杆起跨距离在可见窗口内（无盲走段起跨）
        tf = level.trigger_fwd
        assert d_min - 1.0 <= tf <= level.d_hurdle + CONFIRM_SLACK_CM + 2.0, \
            f"seed={seed} 起跨距离 {tf:.1f}cm 出窗 [{d_min:.1f}, " \
            f"{level.d_hurdle + CONFIRM_SLACK_CM + 2.0:.1f}]"
        # 跨杆时航向收敛（爬楼段方位微调 + 对正的联合效果）
        assert abs(robot.heading) <= 10.0, f"seed={seed} 跨杆航向 {robot.heading:.1f}°"
        # 整体越过横杆（跨杆 +10cm，出场 +12cm）
        assert robot.pos[1] >= scene.bar_y - 3.0, \
            f"seed={seed} 终点 y={robot.pos[1]:.1f} < 杆 {scene.bar_y:.1f}"
        assert robot.n_captures <= CAPTURE_GUARD, \
            f"seed={seed} 拍照 {robot.n_captures} 张超护栏"
    print(f"  端到端 {n_ok}/20 到达 DONE，起跨距离全落窗，航向收敛 ✓"
          f"（耗时 {time.time() - t0:.0f}s）")


def test_bar_invisible_degradation():
    """横杆全程不可见：降级链（pitch 重试→盲走/按当前位置）不弃赛可达 DONE"""
    for seed in (0, 1):
        scene, robot, level = make_run(seed)
        scene.bar_visible = False
        ok = level.run_level()
        assert ok, f"seed={seed} 横杆不可见时应走降级链完成"
        assert robot.n_captures <= CAPTURE_GUARD
    print("  降级路径：横杆全程不可见仍完成全流程（不弃赛）✓")


def test_tape_invisible_alarms():
    """全场无红：粗接近超限停机报警（唯一允许的失败点）。

    注：仅"胶条缺失但横杆可见"的场景超出纯红色检测的判别力（会把横杆误
    认作胶条）——现场以胶条存在为前提，彻底解法是 P2 的台阶边缘检测。
    """
    scene, robot, level = make_run(2)
    scene.tape_visible = False
    scene.bar_visible = False
    ok = level.run_level()
    assert not ok, "无红色目标应粗接近超限停机报警"
    print("  无红目标：粗接近超限停机报警 ✓")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="只跑降级路径不跑 20 种子")
    args = ap.parse_args()
    test_tape_invisible_alarms()
    test_bar_invisible_degradation()
    if not args.quick:
        test_e2e_20_seeds()
    print("全部 stairs_hurdle FSM 仿真测试通过 ✓")
