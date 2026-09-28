#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""measure_anchor_cost.py —— 锚解算耗时随**对应点数**的变化（不动机器人）

⚠️ 只拍照 + 纯计算：不发动作、不动舵机，机器人原地不动。

它回答的问题
------------------------------------------------
"统一决策"（三档分区 + 不分三阶段）打算**每帧**解一次 MAP-ANCHOR 拿实测位姿。
锚解算是 RANSAC（枚举 4 子集，上限 `MAP_POSE_MAX_COMBOS`），**开销随对应点数
增长**；而对应点数取决于"画面里能看到几块面板"（远时多、贴近时少）。
⇒ 本脚本量出「对应点数 N → 锚解算耗时」曲线与**最坏情况**，再折算成
   "每帧多花多少秒、单格 70s 预算还够不够"。

怎么做（不需要移动）
------------------------------------------------
用**当前站位**拍一张帧取真实观测；对应点不足时用带抖动的副本补齐到目标 N。
N 只影响 RANSAC 的计算量，所以耗时是有代表性的（几何退化只影响"解不算得出来"，
不影响单次求解的代价）。
计时**直接调生产的 `_map_pose`**（只把 `_map_pose_correction` 换成注入的合成对应集），
不复制任何算法代码。

怎么跑
------------------------------------------------
    # PC（先把仓库同步到机器人）
    python tools/sync_to_robot.py
    python tools/exec_on_robot.py --cmd "cd /home/pi/Robot_Competition && /home/pi/jupyter-env/bin/python3 tools/measure_anchor_cost.py" --timeout 900

    # 或 SSH 进机器人直接跑
    cd /home/pi/Robot_Competition && source /home/pi/jupyter-env/bin/activate
    python tools/measure_anchor_cost.py --reps 7

可选参数
    --reps N       每个点数档重复次数（默认 7）
    --photo PATH   用已存在的照片（默认自动拍一张）
    --nmax N       最大对应点数（默认 12）
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np  # noqa: E402


def _read_with_retry(path, tries=4, wait=0.15):
    """fswebcam 写盘与 cv2.imread 之间偶发竞态 ⇒ 重试 + 校验尺寸"""
    import cv2
    for _ in range(tries):
        try:
            if path and os.path.exists(path) and os.path.getsize(path) > 1024:
                fr = cv2.imread(path)
                if fr is not None and fr.size:
                    return fr
        except OSError:
            pass
        time.sleep(wait)
    return None


class _ObsStub:
    """极简观测桩：_map_pose 只用 digit / clipped / center_px / bbox / hull_poly"""

    def __init__(self, digit):
        self.digit = int(digit)
        self.clipped = False
        self.center_px = (0.0, 0.0)
        self.bbox = (0.0, 0.0, 10.0, 10.0)
        self.hull_poly = ()


def main(argv=None):
    ap = argparse.ArgumentParser(description="锚解算耗时 vs 对应点数（不动机器人）")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--photo", default=None)
    ap.add_argument("--nmax", type=int, default=12)
    args = ap.parse_args(argv)

    from core.robot_core import RobotState
    from vision.nine_grid_detector import NineGridDetector
    import levels.nine_grid_shared as NG

    st = RobotState(tag_poses={})
    det = NineGridDetector()

    # ---------------- 取帧（只拍照） ----------------
    print("=" * 72)
    print("取帧（只拍照；不发动作、不动舵机）")
    print("=" * 72)
    frame = _read_with_retry(args.photo) if args.photo else None
    if frame is not None:
        print(f"  用指定照片 {args.photo}  {frame.shape}")
    else:
        for _ in range(3):
            fn = st.capture_image()
            frame = _read_with_retry(fn)
            if frame is not None:
                print(f"  已拍：{fn}  {frame.shape}")
                break
            time.sleep(0.4)
    if frame is None:
        print("  ❌ 拿不到可用帧（fswebcam 或读图失败）")
        return 1

    # ---------------- 真实观测 → 基对应点 ----------------
    obs = det.detect_panels(frame, arbitrate=False, drop_border=False)
    lv = NG.NineGridShared(st)
    lv.digit_cell = {d: d - 1 for d in range(1, 8)}   # 合成映射，仅用于计时
    corr, _keep = lv._map_pose_correction(obs, frame)
    base = [(np.asarray(c[0], float), np.asarray(c[1], float)) for c in corr]
    print(f"  本帧观测 {len(obs)} 个；可作对应的点 {len(base)} 个")
    if not base:
        print("  （本帧无可用对应 ⇒ 用合成点量耗时；数值仍代表 RANSAC 代价）")
        base = [(np.array([20.0 + 30.0 * i, 50.0]),
                 np.array([900.0 + 300.0 * i, 900.0])) for i in range(3)]

    rng = np.random.default_rng(12345)
    orig_corr = lv._map_pose_correction

    def bench_anchor(n, reps):
        """把 _map_pose_correction 换成 N 个合成对应，计时生产的 `_map_pose`"""
        pts = []
        while len(pts) < n:
            g, p = base[len(pts) % len(base)]
            pts.append((g + rng.normal(0, 0.5, 2), p + rng.normal(0, 1.5, 2)))
        setattr(lv, "_map_pose_correction",
                lambda o, f, swap_corners=False: (pts, [_ObsStub(i + 1)
                                                        for i in range(n)]))
        lv._anchor_quiet = True        # 静音，避免刷屏影响计时
        try:
            lv._map_pose([], frame, why="bench")     # 预热
            ts = []
            for _ in range(reps):
                t = time.perf_counter()
                lv._map_pose([], frame, why="bench")
                ts.append(time.perf_counter() - t)
            return min(ts), sum(ts) / len(ts)
        finally:
            setattr(lv, "_map_pose_correction", orig_corr)
            lv._anchor_quiet = False

    # ---------------- 逐点数档计时 ----------------
    print("\n" + "=" * 72)
    print(f"锚解算耗时 vs 对应点数（reps={args.reps}，取最小 / 均值）")
    print("=" * 72)
    print(f"  {'对应点数N':>9s} {'4子集数':>8s} {'实际枚举':>8s} "
          f"{'最小(s)':>10s} {'均值(s)':>10s}")
    worst = 0.0
    for n in range(NG.MAP_POSE_MIN_PTS, args.nmax + 1):
        combos = math.comb(n, 4)
        used = min(combos, NG.MAP_POSE_MAX_COMBOS)
        mn, av = bench_anchor(n, args.reps)
        worst = max(worst, av)
        print(f"  {n:>9d} {combos:>8d} {used:>8d} {mn:>10.4f} {av:>10.4f}")

    # ---------------- 折算 ----------------
    # 单格取样时长：关卡 2026-09-25 起**取消了一切时间闸**（顺序计分下
    # 放弃一格等于把后面的分全丢），所以这个数只属于本测量脚本自己的折算。
    budget = 70.0
    print("\n" + "=" * 72)
    print("折算")
    print("=" * 72)
    print(f"  最坏一档的锚解算均值 = {worst:.4f}s")
    print(f"  统一决策若**每帧**解一次锚、单格按 ~40 帧估："
          f"锚解算合计 ≈ {worst * 40:.1f}s ≈ 单格 {budget:.0f}s 预算的 "
          f"{worst * 40 / budget * 100:.0f}%")
    print("\n  判读：把这一行与 `measure_frame_cost.py` 的『每帧感知』相加——"
          "\n        单格总时间仍 < 70s 且帧数够 ~40 张，则统一决策在算力上可行。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
