#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_quantize_straight.py —— QuantizeStraight 单元测试。

用法：
    python tools/test_quantize_straight.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motion_model import quantize_straight


def check(L, expected_n5=None, expected_n2=None, max_abs_error=1.0):
    actual, actions, error = quantize_straight(L)
    n5 = actions.get("go_forward", 0)
    n2 = actions.get("go_forward_one_step", 0)

    # 规则校验：最多 1 个小步
    assert n2 <= 1, f"L={L}: n2={n2} > 1"

    # 总长校验
    total = n5 * 5 + n2 * 2
    assert abs(total - actual) < 1e-9, f"L={L}: total {total} != actual {actual}"

    # 误差校验
    assert abs(error) <= max_abs_error + 1e-9, f"L={L}: error {error}"

    if expected_n5 is not None:
        assert n5 == expected_n5, f"L={L}: n5 {n5} != {expected_n5}"
    if expected_n2 is not None:
        assert n2 == expected_n2, f"L={L}: n2 {n2} != {expected_n2}"

    return actual, actions, error


def main():
    # L=0 应无动作
    actual, actions, error = quantize_straight(0)
    assert actual == 0 and actions == {} and abs(error) < 1e-9

    # 用户给出的关键例子
    # L=18 -> 15 + 1*2，误差 +1
    check(18, expected_n5=3, expected_n2=1)
    # L=19 -> 20 - 1，误差 -1
    check(19, expected_n5=4, expected_n2=0)

    # 全覆盖 1..30
    for L in range(1, 31):
        check(L)

    # 一些非整数（允许误差稍大？这里按规则应 <=1.5）
    for L in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]:
        actual, actions, error = quantize_straight(L)
        assert abs(error) <= 1.5 + 1e-9, f"L={L}: error {error}"

    print("QuantizeStraight 全部测试通过")


if __name__ == "__main__":
    main()
