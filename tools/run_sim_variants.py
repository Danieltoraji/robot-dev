# -*- coding: utf-8 -*-
"""
run_sim_variants.py —— 以不同噪声配置批量跑模拟器（不改 goodluck_sim.py 源码）

用法：
    python tools/run_sim_variants.py ideal|noisy|stress [次数]
    ideal  : 全零噪声（验证决策逻辑正确性）
    noisy  : 默认噪声（定位 0.5cm/1.0°，动作 ±10%，转向 5°）
    stress : 压测噪声（定位 2.4cm/5.0°，动作 ±10%，转向 8°）
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import goodluck_sim as sim

VARIANTS = {
    "ideal":  dict(LOCATE_NOISE_STD=0.0, LOCATE_ANGLE_NOISE_STD=0.0,
                   ACTION_ERROR_STD=0.0, TURN_ERROR_STD=0.0),
    "noisy":  dict(LOCATE_NOISE_STD=0.5, LOCATE_ANGLE_NOISE_STD=1.0,
                   ACTION_ERROR_STD=0.1, TURN_ERROR_STD=5.0),
    "stress": dict(LOCATE_NOISE_STD=2.4, LOCATE_ANGLE_NOISE_STD=5.0,
                   ACTION_ERROR_STD=0.1, TURN_ERROR_STD=8.0),
}


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "ideal"
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    cfg = VARIANTS[variant]
    sim.LOCATE_NOISE_STD = cfg["LOCATE_NOISE_STD"]
    sim.LOCATE_ANGLE_NOISE_STD = cfg["LOCATE_ANGLE_NOISE_STD"]
    sim.ACTION_ERROR_STD = cfg["ACTION_ERROR_STD"]
    sim.TURN_ERROR_STD = cfg["TURN_ERROR_STD"]
    print(f"[variant={variant}] {cfg}")
    ok = 0
    for i in range(rounds):
        print(f"\n{'='*20} 第 {i+1}/{rounds} 轮（{variant}）{'='*20}")
        sim.run_simulation()
        ok += 1
    print(f"\n[variant={variant}] 完成 {ok}/{rounds} 轮")


if __name__ == "__main__":
    main()
