# -*- coding: utf-8 -*-
"""顺序计分 ⇒ 绝不跳格、绝不因耗时放弃一格（2026-09-25 策略）的回归测试

背景：本关按 1→7 依次到位计分，**顺序断一格后面全不算**。旧的"单格超时/拍照
超限就记未确认、接着做下一格"在顺序计分下等于把后面的分全丢掉，已取消：

  ① 一格没确认 ⇒ `run_level` **继续磨这一格**，`go_to_panel(digit, attempt)`
     的 attempt 递增（子类用它换策略）；
  ② 时间/拍照数/动作数只打印提醒，**不停止**；
  ③ 唯一能自己停下来的自动保护是**离场护栏**（位姿越出场地外框）。

这个文件用小脚本化的 `_drive_to_panel` 直接验这三条，不跑真实视觉（快）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import levels.nine_grid as NG  # noqa: E402
import levels.nine_grid_three_stage as THREE  # noqa: E402
from core.camera_config import HEAD_CENTER  # noqa: E402


class _StubState:
    """只提供公共件需要的接口（不碰真机/仿真）"""

    def __init__(self):
        self.current_head_pulse = HEAD_CENTER
        self.pitch = 1500
        self.pos = np.array([50.0, -20.0])
        self.actions = []

    def act(self, name, times=1):
        self.actions.append((name, times))

    def set_head(self, pulse, move_time_ms=500):
        self.current_head_pulse = pulse

    def set_pitch(self, pulse, move_time_ms=500):
        self.pitch = pulse

    def capture_frame(self):        # 本文件不跑视觉；万一被调到就报错
        raise AssertionError("这个测试不该真的拍照")


def _make(level_cls, outcome, router=None):
    """造一个关卡：布局扫跳过、逐格走法由 `outcome` 脚本决定

    outcome: {digit: [True/False, ...]} —— 该数字第 n 次尝试是否成功
             （列表用完后按最后一个值一直重复）
    """
    lv = level_cls(_StubState())
    lv.digit_cell = {d: d - 1 for d in range(1, 8)}   # 1→格0 … 7→格6
    lv.cell_conflict = set()
    lv.layout_scan = lambda: None
    calls = []

    base = router if router is not None else level_cls._drive_to_panel

    def scripted(self, digit, attempt=1):
        calls.append((digit, attempt))
        # 子类的契约：先开本次尝试的计数（真实现里 _drive_to_panel 第一句就是它）
        self._begin_cell_attempt(digit, attempt)
        seq = outcome.get(digit, [True])
        ok = seq[min(attempt - 1, len(seq) - 1)]
        if ok:
            self._arrive_evidence = f"脚本第 {attempt} 次尝试判到位"
        return ok

    lv._drive_to_panel = scripted.__get__(lv, level_cls)
    return lv, calls


# =====================================================================
# 1. 绝不跳格：失败就重试同一格，attempt 递增
# =====================================================================

def test_retries_same_panel_until_confirmed():
    """面板 3 前两次失败 ⇒ 第三次仍在磨面板 3，且 4 必须等 3 确认后才开始"""
    lv, calls = _make(NG.NineGridLevel, {3: [False, False, True]})
    ok = lv.run_level()

    assert ok, f"脚本里第 3 次尝试会成功，run_level 却说没完成：{lv.results}"
    assert lv.results == [(d, True) for d in range(1, 8)], lv.results
    assert lv.attempt_log == {1: 1, 2: 1, 3: 3, 4: 1, 5: 1, 6: 1, 7: 1}, \
        lv.attempt_log
    # 面板 3 的三次尝试必须在面板 4 之前**连续**发生（没有跳过去又回来）
    seq3 = [a for (d, a) in calls if d == 3]
    assert seq3 == [1, 2, 3], seq3
    first4 = next(i for i, (d, _a) in enumerate(calls) if d == 4)
    last3 = max(i for i, (d, _a) in enumerate(calls) if d == 3)
    assert last3 < first4, f"面板4 在面板3 之前被碰过：{calls}"
    print(f"  绝不跳格 ✓ 面板3 磨了 3 次；调用序列 {calls}")


def test_three_stage_also_never_skips():
    """三段式同样不许跳格（两套办法共用 run_level 的这条规矩）"""
    lv, calls = _make(THREE.NineGridThreeStageLevel, {5: [False, True]})
    ok = lv.run_level()
    assert ok and lv.attempt_log[5] == 2, (ok, lv.attempt_log)
    print(f"  三段式绝不跳格 ✓ 调用序列 {calls}")


# =====================================================================
# 2. 时间只提醒、不停止
# =====================================================================

def test_time_passing_does_not_stop_the_run():
    """把"开始时刻"挪到很久以前（远超参考用时）⇒ 仍然继续磨这一格

    旧行为：超过单格硬预算/全局看门狗就返回 False（本格记未确认、去下一格）。
    新行为：只打印"[重试] … 已超参考 Ns"，继续重试。
    """
    lv, calls = _make(NG.NineGridLevel, {2: [False, False, False, True]})
    lv.layout_t0 = 0.0                    # 整局"已经跑了几十年"
    lv._cell_t0 = 0.0                     # 本格也一样
    ok = lv.run_level()
    assert ok, "时间超参考值不该让 run_level 收手"
    assert lv.attempt_log[2] == 4, lv.attempt_log
    assert not lv.cell_trips, f"不该有'被终止'的格：{lv.cell_trips}"
    print(f"  超时不停止 ✓ 面板2 磨了 4 次；cell_trips={lv.cell_trips}")


def test_no_time_limit_constants_left():
    """钉住"没有单格时间上限这类常量"：删干净了才不会有人再挂回去"""
    for name in ("TARGET_TIME_BUDGET_S", "TOTAL_TIME_BUDGET_S",
                 "CELL_LIMIT_TIMEOUT_S", "CELL_LIMIT_FRAMES",
                 "CELL_LIMIT_ACTIONS", "CELL_LIMIT_BUDGET_FRAC",
                 "CELL_LIMIT_TIMEOUT_MIN_S"):
        assert not hasattr(NG.NineGridShared, name) \
            and not hasattr(NG, name), \
            f"{name} 又回来了——顺序计分下不能有'耗时太长就放弃一格'的闸"
    assert hasattr(NG.NineGridShared, "_give_up_attempt"), \
        "换成'本次尝试收手、重试本格'的接口不见了"
    print("  时间类上限常量已清空 ✓")


# =====================================================================
# 3. 只剩离场护栏能停
# =====================================================================

def test_only_arena_guard_stops_the_whole_run():
    """位姿越界 ⇒ 整局收手（不再去找下一格），并留下可观测记录"""
    lv, calls = _make(NG.NineGridLevel, {1: [True], 2: [False]})

    def out_of_field(self, digit, attempt=1):
        calls.append((digit, attempt))
        self._begin_cell_attempt(digit, attempt)
        if digit == 1:
            return True                    # 第一格正常到位
        self.pose = np.array([999.0, 999.0, 0.0])   # 第二格起越出场地外框
        self._arena_guard()                          # 动作路径上会调它
        return False

    lv._drive_to_panel = out_of_field.__get__(lv, NG.NineGridLevel)
    ok = lv.run_level()
    assert not ok
    assert lv._abort_level and "位姿越界" in lv._abort_level, lv._abort_level
    assert lv.results == [(1, True), (2, False)], lv.results
    assert 3 not in [d for d, _ in lv.results], "越界后不该再去找下一格"
    print(f"  只有离场护栏硬停 ✓ {lv._abort_level}")


if __name__ == "__main__":
    test_retries_same_panel_until_confirmed()
    test_three_stage_also_never_skips()
    test_time_passing_does_not_stop_the_run()
    test_no_time_limit_constants_left()
    test_only_arena_guard_stops_the_whole_run()
    print("顺序计分（绝不跳格 / 超时不停止 / 只有离场护栏硬停）测试通过 ✓")
