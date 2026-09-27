# -*- coding: utf-8 -*-
"""临时实验 9b：喂给像素域精修的观测该不该含"裁切质心"观测？

现场照片 55 条观测里 37 条是裁切质心（偏差实测 1.3~8.4cm），精修解出的偏移
被顶到 +41.8°（贴 55° 上界）、中位残差 45.4px。

本脚本**从真实 _fit_grid 自己的起点出发**（拦一次 calibrate_pixel_pose 拿到
entries/pose0/off0/h0/refs），再分别用"含裁切 / 干净优先 / 只要干净"重跑，
保证三种喂法只差"喂了哪些观测"。

用法：python tools/exp_calib_inputs.py
"""

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import numpy as np

import levels.nine_grid_shared as ngs
from levels.nine_grid_shared import NineGridShared, calibrate_pixel_pose
from tools.replay_ninegrid import (
    ReplayRobot, build_schedule, find_photos, sweep_and_collect, PHOTO_TRUTH,
)

CAPTURED = {}
_ORIG = ngs.calibrate_pixel_pose


def spy(entries, p0, off0, h0, iters=12, refs=None):
    if not CAPTURED:
        CAPTURED.update(entries=list(entries), p0=np.array(p0, float),
                        off0=float(off0), h0=float(h0), refs=refs)
    return _ORIG(entries, p0, off0, h0, iters=iters, refs=refs)


def main():
    photos = find_photos()
    robot = ReplayRobot(build_schedule(photos, "schedule"), verbose=False)
    level = NineGridShared(robot)
    pix_obs = sweep_and_collect(robot, level, diag=False)

    ngs.calibrate_pixel_pose = spy
    fit = level._fit_grid(pix_obs)
    ngs.calibrate_pixel_pose = _ORIG
    print(f"真实 _fit_grid：cells={'=真值 ✓' if fit.cells == PHOTO_TRUTH else '✗'} "
          f"高度 {fit.cam_height_cm:.1f}cm（{fit.height_source}）"
          f" 偏移 {fit.offset_deg:+.1f}°")
    if not CAPTURED:
        print("没有捕获到精修调用")
        return
    ent = CAPTURED["entries"]
    n_cl = sum(1 for e in ent if e[3])
    print(f"起点：位姿 ({CAPTURED['p0'][0]:.1f},{CAPTURED['p0'][1]:.1f},"
          f"{np.degrees(CAPTURED['p0'][2]):.1f}°)、偏移 {CAPTURED['off0']:+.1f}°、"
          f"高度 {CAPTURED['h0']:.1f}cm；entries {len(ent)} 条（裁切 {n_cl} 条）\n")

    print(f"  {'喂法':20s} {'条目':>4s} {'高度':>7s} {'偏移':>8s} "
          f"{'中位px':>8s} {'RMS px':>8s}")
    sets = [("含裁切（现状）", ent),
            ("只要干净观测", [e for e in ent if not e[3]])]
    for tag, sub in sets:
        if len(sub) < 4:
            print(f"  {tag:20s} {len(sub):4d}   条数不足")
            continue
        p5, rms_all, med_px = _ORIG(sub, CAPTURED["p0"], CAPTURED["off0"],
                                    CAPTURED["h0"], refs=CAPTURED["refs"])
        if p5 is None:
            print(f"  {tag:20s} {len(sub):4d}   未收敛")
            continue
        print(f"  {tag:20s} {len(sub):4d} {p5[4]:7.1f} {p5[3]:+8.1f} "
              f"{med_px:8.1f} {rms_all:8.1f}")


if __name__ == "__main__":
    main()
