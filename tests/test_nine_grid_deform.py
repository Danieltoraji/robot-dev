# -*- coding: utf-8 -*-
"""地板形变鲁棒性集成测试（tests/test_nine_grid_deform.py）

背景（2026-09-11 用户现场约束）：本关地板会形变，机器人在板上行走时机体俯仰
/相机高度会发生**大幅变化**（按 ±15° 考虑）。任何"冻结相机常数 + 度量投影"
的判据在这种形变下都会失效——量化：h=50cm 时可见地面带在有效俯角
42°/57°/72° 下分别是 19~184 / 5~86 / 0~50 cm（4 倍范围摆动）。
故导航必须走**视觉伺服**（像素偏移/框宽/颜色占比等相对量）。

本测试在仿真里注入形变（只改世界侧的相机姿态/高度，关卡内常数保持不动）：
  1. 基线（无形变）；
  2. 随机游走：σ=3°/动作 俯仰 + σ=0.5cm/动作 高度（限幅 ±15° / ±2cm）；
  3. 一次性阶跃：第 3 格附近 +15° 俯仰（模拟"踩上/离开板"）。
三场景都要求：布局正确 + 1..7 全部确认到达 + 拍照数 < 护栏。

运行：python tests/test_nine_grid_deform.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # 仓库根

from sim.nine_grid_sim import run_simulation

# 拍照数护栏（空气/时间代理指标；基线 219 张，见 test_nine_grid_sim.py）
CAPTURE_GUARD = 350


def _check(tag, stats):
    assert stats["layout_ok"], \
        f"[{tag}] 布局解算错误: {stats['digit_cell']} != 真值 {stats['truth']}"
    failed = [k for k, ok in stats["results"] if not ok]
    assert stats["ok_all"] and not failed, \
        f"[{tag}] 未确认到达的面板: {failed}"
    assert not stats["banned_used"], \
        f"[{tag}] 使用了禁用动作: {stats['banned_used']}"
    assert stats["captures"] < CAPTURE_GUARD, \
        f"[{tag}] 拍照 {stats['captures']} 张超护栏 {CAPTURE_GUARD}（定位风暴？）"
    print(f"  [{tag}] 布局 ✓ 7/7 ✓ 拍照 {stats['captures']} 张 / "
          f"动作 {stats['actions']} 次；形变终值 俯仰 "
          f"{stats['deform_tilt_deg']:+.1f}° / 高度 {stats['deform_height_cm']:+.1f}cm")


def test_baseline_no_deform():
    """基线：无形变（防止改造回归）"""
    run = run_simulation(seed=3, quiet=True)
    _check("基线", run.stats)


def test_random_walk_deform():
    """随机游走形变：σ=3°/动作（限幅 ±15°）+ σ=0.5cm/动作（限幅 ±2cm）"""
    deform = {"sigma_tilt_deg": 3.0, "sigma_h_cm": 0.5,
              "max_tilt_deg": 15.0, "max_h_cm": 2.0}
    run = run_simulation(seed=3, quiet=True, deform=deform)
    _check("随机游走±15°", run.stats)


def test_step_deform():
    """一次性阶跃：第 60 个动作后俯仰 +15°（模拟踩上/离开板的一次性变化）"""
    deform = {"sigma_tilt_deg": 0.0, "sigma_h_cm": 0.0,
              "step_after_actions": 60, "step_tilt_deg": 15.0,
              "step_h_cm": -2.0, "max_tilt_deg": 15.0, "max_h_cm": 2.0}
    run = run_simulation(seed=3, quiet=True, deform=deform)
    _check("阶跃+15°", run.stats)


if __name__ == "__main__":
    test_baseline_no_deform()
    test_random_walk_deform()
    test_step_deform()
    print("地板形变鲁棒性测试通过 ✓")
