#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_layout_scan.py —— 只跑布局扫描（不动底盘前进以外无动作）并给出结论

为什么单独有这个工具：整局 `run_level` 会一路驱动机器人；调试"布局到底扫对
没有"时只想要第一阶段，且要一份**可存档、可对比**的结构化结论（数字→格、
自标定常数、位姿、告警）。

注意：布局扫描在"多解/缺数字/自举不过"时会**前进一步（2cm）重扫**，最多 3 轮
（见 levels/nine_grid.layout_scan）——这是设计行为，不是失控。

用法（机器人仓库根目录）：
    /home/pi/jupyter-env/bin/python3 tools/run_layout_scan.py
    /home/pi/jupyter-env/bin/python3 tools/run_layout_scan.py --no-lock --out /tmp/scan.json
退出码：0 成功 / 2 三轮未定 / 3 其它异常
"""

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from core.robot_core import RobotState, lock_camera_controls
from levels.nine_grid import NineGridLevel, PITCH_NAV, TOTAL_TIME_BUDGET_S


def main(argv=None):
    ap = argparse.ArgumentParser(description="九宫格布局扫描（单阶段）")
    ap.add_argument("--no-lock", action="store_true", help="跳过相机锁定")
    ap.add_argument("--budget", type=float, default=600.0, help="本阶段时间预算(s)")
    ap.add_argument("--out", default="field_layout_scan.json")
    args = ap.parse_args(argv)

    lock_ok, lock_info = (False, {"reason": "已跳过"})
    if not args.no_lock:
        lock_ok, lock_info = lock_camera_controls(force=True)
        print(f"[扫描] 相机锁定 {'成功' if lock_ok else '未生效'}: "
              f"{lock_info.get('reason') or lock_info.get('ctrls')}")
        if lock_info.get("mismatch"):
            print(f"[扫描] 读回不一致: {lock_info['mismatch']}")

    state = RobotState()
    level = NineGridLevel(state)
    level.deadline = time.time() + args.budget
    # 拍照计数：layout_scan 不走 _count_frame，故在 capture_frame 上挂个钩子
    shots = {"n": 0}
    _orig_capture = state.capture_frame

    def _counting_capture(*a, **kw):
        shots["n"] += 1
        return _orig_capture(*a, **kw)

    state.capture_frame = _counting_capture
    t0, err, rc = time.time(), None, 0
    try:
        level.layout_scan()
    except RuntimeError as e:
        err, rc = str(e), 2
    except Exception as e:                      # noqa: BLE001
        err, rc = f"{type(e).__name__}: {e}", 3
    dt = time.time() - t0

    cell_digit = {c: d for d, c in (level.digit_cell or {}).items()}
    out = {
        "ok": rc == 0,
        "error": err,
        "digit_cell": {int(k): int(v) for k, v in (level.digit_cell or {}).items()},
        "cell_digit": {int(k): int(v) for k, v in cell_digit.items()},
        "pose": None if level.pose is None else [round(float(v), 2) for v in level.pose],
        "pose_deg": (None if level.pose is None
                     else round(float(level.pose[2]) * 57.29578, 1)),
        "pitch_offset_deg": round(float(level._pitch_offset_deg), 1),
        "cam_height_cm": round(float(level._cam_height_cm), 1),
        "effective_pitch_deg_nav": round(float(level._effective_pitch_deg(PITCH_NAV)), 1),
        "cell_conflict": sorted(int(v) for v in (level.cell_conflict or ())),
        "frames": dict(level.phase_frames or {}),
        "photos": int(shots["n"]),
        "elapsed_s": round(dt, 1),
        "lock": {"ok": lock_ok, "ctrls": lock_info.get("ctrls"),
                 "mismatch": lock_info.get("mismatch"),
                 "reason": lock_info.get("reason")},
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\n================ 布局扫描结论 ================")
    if rc == 0:
        print(f"数字→格: {out['digit_cell']}")
        print(f"格→数字: {out['cell_digit']}   格6是否被占: "
              f"{'是（与规则冲突，需核对标号约定）' if 6 in cell_digit else '否'}")
        print(f"位姿: ({out['pose'][0]}, {out['pose'][1]}) "
              f"航向 {out['pose_deg']}°" if out["pose"] else "位姿: 无")
        print(f"自标定: 安装偏移 {out['pitch_offset_deg']:+.1f}° / 高度 "
              f"{out['cam_height_cm']:.0f}cm（导航档有效俯角 "
              f"{out['effective_pitch_deg_nav']:.1f}°）")
        print(f"仲裁冲突格: {out['cell_conflict']}")
    else:
        print(f"失败（rc={rc}）: {err}")
    print(f"拍照 {out['photos']} 张，耗时 {dt:.0f}s（预算 {args.budget:.0f}s / 全局 "
          f"{TOTAL_TIME_BUDGET_S:.0f}s）")
    print(f"结论已写 {args.out}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
