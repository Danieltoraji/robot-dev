#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""到达门常量不变量（2026-09-24 新增）

为什么要把常数钉在测试里：这套判据的**阈值与几何是绑死的**——
`COLOR_DROP_FRAC=0.55` 能成立的前提是"色块压过时会真的从视野里消失"，
而 PITCH_DOWN（实际俯角 59.9°、画幅下沿对应地面 3.5cm）下这个前提**不成立**：
站在 28cm 面板中心时远侧一半仍在画面里，占比只能降到峰值的 ~0.67。
所以真正可达的判据是"前压到底 + 回落一大截"的兜底（`ARRIVE_TAIL_ACCEPT_FRAC`）。

现场教训（2026-09-24 实测，见 `完整方案-2026-09-24-*.md` §1.2）：
- 出厂（press 45 + 兜底关）：实测原语 16 种子 87/111，落点最大 38.8cm，超半格 4 次；
  名义原语 4 种子 22/28，落点最大 45.0cm，超半格 3 次。
- **新默认（press 60 + 兜底 0.80）**：实测原语 16 种子 **112/112**、拍照 3446→2828、
  落点最大 38.8→**4.3cm**；名义原语 4 种子 **28/28**、拍照 1132→892、
  落点最大 45.0→**5.7cm**；超半格 **0/0**。
把这两个数一起改掉（例如只把 DROP_FRAC 放宽到 0.70）会**提前停**、牺牲落点
（名义参数面板 3/4/5 的落点从 2.2/0.7/3.4cm 推到 6.8/5.5/5.2cm）⇒ 踩不到开关。
"""
import levels.nine_grid_three_stage as NG


def test_arrival_gate_constants_pinned():
    # 名义判据：色块"出现→消失"。**不要放宽它去换到达率**（会提前停）。
    assert NG.COLOR_DROP_FRAC == 0.55
    # 兜底判据：只在"前压到底 + 回落一大截"时放行；0 = 回退到旧行为。
    assert NG.ARRIVE_TAIL_ACCEPT_FRAC == 0.80
    assert 0.0 < NG.ARRIVE_TAIL_PRESS_FRAC <= 1.0
    # 兜底阈必须**高于**几何可达下限（~0.67），否则等于没放宽；
    # 又必须**低于** 1.0，否则永远不触发。
    assert 0.67 < NG.ARRIVE_TAIL_ACCEPT_FRAC < 1.0


def test_arrival_gate_tail_accept_requires_press():
    """兜底必须挂在"前压接近封顶"上——否则它就不是兜底，是提前停。"""
    assert NG.ARRIVE_TAIL_PRESS_FRAC >= 0.8, \
        "前压比例门槛太低会让兜底提前触发，牺牲落点精度"


def test_arrival_press_budget_covers_peak_to_centre():
    """前压预算必须覆盖"交棒 → 占比峰值 → 格心"的全行程

    真值几何实测（36 次到达尝试）：交棒在 45~48cm，**占比峰值固定在离格心 14~19cm**
    （中位 16.1cm），峰值之后还要再走完这 16cm 才到格心。
    旧值 45cm 恰好在格心处用尽 ⇒ 判据以 0.4%~3.7% 之差漏判 ⇒ 整格重搜 ⇒ 落点 45cm。
    """
    assert NG.ARRIVE_PRESS_MAX_CM >= 55.0, \
        "预算不足以覆盖峰值→格心的行程，会在格心处用尽（实测病根）"
    assert NG.ARRIVE_PRESS_MAX_CM < NG.ARRIVE_PRESS_PLAUSIBLE_CM, \
        "前压上限必须小于'越界一整格'的物理合理性门"

