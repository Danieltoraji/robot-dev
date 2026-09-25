#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""measure_frame_cost.py —— 真机**单帧感知**各环节耗时（不动机器人）

⚠️ 本脚本**只做两件事**：调 `fswebcam` 拍静态照片、在内存里跑识别算法。
   **不发任何动作**（不 go_forward / 不 turn / 不 left_move），
   **不动任何舵机**（不 set_head / 不 set_pitch）。机器人原地不动。

它回答的问题
------------------------------------------------
1. 现在"拍一张 + 识别一次"到底要多久？（决定单格 70s 预算里能拍几张）
2. 三个候选改动各自要加多少开销？
   - **现状**：拍照 + 色占比 + **单色**检测
   - **WS4 几何到达判据**：现状 + 每 3 帧一次 **全色**检测 + 锚解算
   - **统一决策**（三档分区 + 不分三阶段）：每帧都要 **全色**检测 + 锚解算

怎么跑
------------------------------------------------
方式一（PC，推荐；先把仓库同步到机器人）：
    python tools/sync_to_robot.py
    python tools/exec_on_robot.py --cmd "cd /home/pi/Robot_Competition && /home/pi/jupyter-env/bin/python3 tools/measure_frame_cost.py" --timeout 900

方式二（SSH 进机器人直接跑）：
    ssh pi@192.168.31.209
    cd /home/pi/Robot_Competition && source /home/pi/jupyter-env/bin/activate
    python tools/measure_frame_cost.py --reps 5

可选参数
    --reps N      每个环节重复次数（默认 5，取最小值与均值）
    --no-capture  跳过拍照环节，只用一张已存在的照片（更快；配合 --photo）
    --photo PATH  指定已存在的照片路径（默认自动拍一张）
"""
from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np  # noqa: E402


def _read_with_retry(path, tries=4, wait=0.15):
    """fswebcam 写盘与 cv2.imread 之间偶发竞态 ⇒ 重试 + 校验尺寸

    （历史现象：`capture_image()` 用 `int(time.time())` 命名，同一秒内两次拍照
      会撞名；且没有 fsync。直接读会偶发返回 None。）
    """
    import cv2
    for k in range(tries):
        try:
            if os.path.exists(path) and os.path.getsize(path) > 1024:
                fr = cv2.imread(path)
                if fr is not None and fr.size:
                    return fr
        except OSError:
            pass
        time.sleep(wait)
    return None


def bench(fn, reps, warmup=1):
    """返回 (最小值, 均值, 全部样本)；抛异常按 NaN 记，不中断整轮测量"""
    for _ in range(warmup):
        try:
            fn()
        except Exception:
            break
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        try:
            fn()
        except Exception as e:
            print(f"    ⚠ 该环节抛异常：{type(e).__name__}: {e}")
            return float("nan"), float("nan"), []
        ts.append(time.perf_counter() - t)
    return min(ts), sum(ts) / len(ts), ts


def main(argv=None):
    ap = argparse.ArgumentParser(description="真机单帧感知耗时（不动机器人）")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--no-capture", action="store_true",
                    help="跳过拍照环节（用 --photo 或已存在的一张照片）")
    ap.add_argument("--photo", default=None, help="用这张已存在的照片")
    args = ap.parse_args(argv)

    from core.robot_core import RobotState
    from vision.nine_grid_detector import NineGridDetector
    import levels.nine_grid_shared as NG

    st = RobotState(tag_poses={})
    det = NineGridDetector()

    # ---------------- 取一张帧（后续环节都在它上面做，避免拍照噪声干扰）----
    print("=" * 72)
    print("取帧")
    print("=" * 72)
    photo = args.photo
    frame = None
    if photo:
        frame = _read_with_retry(photo)
        print(f"  指定照片 {photo}: "
              f"{'OK ' + str(frame.shape) if frame is not None else '读不出'}")
    if frame is None:
        for attempt in range(3):
            fn = st.capture_image()
            if fn:
                frame = _read_with_retry(fn)
                if frame is not None:
                    photo = fn
                    break
            time.sleep(0.4)
    if frame is None:
        print("  ❌ 拿不到可用帧（fswebcam 或读图失败）。"
              "请先单独确认拍照链路：")
        print("     fswebcam -r 2592x1944 --no-banner -S 3 /tmp/t.jpg && "
              "ls -l /tmp/t.jpg")
        return 1
    print(f"  帧 {frame.shape}（后续环节都在这张帧上重复跑）")

    # ---------------- 各环节 ----------------
    print("\n" + "=" * 72)
    print(f"各环节耗时（reps={args.reps}，取最小值 / 均值，单位秒）")
    print("=" * 72)
    rows = []

    if not args.no_capture:
        mn, av, _ = bench(lambda: _read_with_retry(
            st.capture_image() or ""), args.reps, warmup=0)
        rows.append(("① 拍照（fswebcam + 读图）", mn, av))

    mn, av, _ = bench(lambda: det.color_ratio(frame, "red"), args.reps)
    rows.append(("② 色占比 color_ratio（单色）", mn, av))

    mn, av, _ = bench(lambda: det.detect_panels(
        frame, colors=["red"], arbitrate=False, drop_border=False), args.reps)
    rows.append(("③ detect_panels 单色", mn, av))

    mn, av, obs_box = bench(lambda: det.detect_panels(
        frame, arbitrate=False, drop_border=False), args.reps)
    rows.append(("④ detect_panels 全 7 色", mn, av))

    # ⑤ 锚解算（需要 level 实例；digit_cell 用"数字 d → 格 d-1"的合成映射，
    #    这里只量**耗时**，映射内容不影响 RANSAC 的计算量级）
    lv = NG.NineGridShared(st)
    lv.digit_cell = {d: d - 1 for d in range(1, 8)}
    obs = det.detect_panels(frame, arbitrate=False, drop_border=False)
    corr, _keep = lv._map_pose_correction(obs, frame)
    mn, av, _ = bench(lambda: lv._map_pose(obs, frame, why="bench"), args.reps)
    rows.append((f"⑤ 锚解算 _map_pose（本帧对应点 {len(corr)} 个）", mn, av))

    print(f"  {'环节':44s} {'最小':>9s} {'均值':>9s}")
    for name, mn, av in rows:
        print(f"  {name:44s} {mn:9.3f} {av:9.3f}")

    # ---------------- 折算 ----------------
    def get(i):
        return rows[i][1] if i < len(rows) else 0.0

    cap = get(0) if not args.no_capture else float("nan")
    ratio = get(1 if not args.no_capture else 0)
    one = get(2 if not args.no_capture else 1)
    all7 = get(3 if not args.no_capture else 2)
    anchor = get(4 if not args.no_capture else 3)

    print("\n" + "=" * 72)
    print("折算：三种候选的『每帧感知』开销 与 单格预算能拍多少帧")
    print("=" * 72)
    cur = cap + ratio + one
    ws4 = cap + ratio + one + (all7 + anchor) / 3.0
    uni = cap + ratio + all7 + anchor
    budget = NG.CELL_LIMIT_TIMEOUT_S
    print(f"  单格硬预算 CELL_LIMIT_TIMEOUT_S = {budget:.0f}s")
    for name, per in (("现状（色占比 + 单色检测）", cur),
                      ("+ WS4（每 3 帧一次全色+锚）", ws4),
                      ("统一决策（每帧全色+锚）", uni)):
        if per != per or per <= 0:                    # NaN
            print(f"  {name:34s} —— 缺拍照耗时，无法折算（去掉 --no-capture 再跑）")
            continue
        print(f"  {name:34s} 每帧 {per:6.3f}s → 单格可拍 "
              f"{budget / per:6.1f} 帧（现状整局约 200 张 / 单格最多 ~40 张）")

    print("\n提示：单格 70s 之外还有『全局 780s』与『每格 45~70s 自适应』两道闸；"
          "\n      若上面的帧数 ≥ 40，说明算力不是瓶颈。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
