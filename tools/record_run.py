#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""record_run.py —— 把仿真跑成**能看的录像**（mp4 + 逐帧 PNG）

为什么需要它：`tools/ab_ninegrid.py` 和图形界面**跑的是同一个仿真、同一份关卡
代码**，区别只在输出——AB 把画面关掉只数数字，图形界面把过程画出来。只报 AB
的数字而不看过程，就会犯"拿关卡自报的到达当仿真判定"这种错（2026-09-24 实犯）。

用法
----
    # 录一段 mp4 + 每 N 帧存一张 PNG（最快速度，不开窗）
    python tools/record_run.py --seed 3 --out archive/result/rec_base

    # 一边录一边开窗（SPACE 暂停 / S 单步 / D 检出叠加 / +/- 调速 / Q 退出）
    python tools/record_run.py --seed 3 --live --out archive/result/rec_unified \
        --unified

    # 选算法（默认 = 真机默认的三段式）
    --baseline            三段式（默认开关）
    --unified             三档分区 + 统一决策
    --no-region           统一决策用旧的"占比回落"到达判据
    --tune                绿走廊×0.6 + 蓝区高 0.60
    --primitives real     运动原语用现场实测值（与 AB 同口径）

画面里能看到：左边相机帧（含检出框），右边场地俯视图（真值面板布局、**估计位姿
vs 真值位姿**两条轨迹），底下一行状态（阶段/目标/拍照数/前后横向差）。
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import levels.nine_grid as NG  # noqa: E402
import sim.nine_grid_view as VIEW  # noqa: E402
import sim.nine_grid_sim as SIM  # noqa: E402
import ab_ninegrid as AB  # noqa: E402


class Recorder(VIEW.NineGridView):
    """把 NineGridView 的"显示"换成"写文件"；--live 时退回原显示

    录像画面顶部那行字（banner）用 ASCII：cv2.putText 画不了中文。
    文案约定：
      `SIM seed=3 layout=fixed primitives=real` —— 本局配置（种子/布局/运动原语）；
      `NAV unitary | arrive by region` / `NAV unitary | arrive by color drop`
      / `NAV staged (default)` —— 本局导航算法与到达判据；
      `arrive: underfoot purple >= 0.35, far orange <= 0.10`
      —— 区域判据的两个门槛（脚下紫区占比、远处橙区占比，见 levels/nine_grid 的
      VIS_ARRIVE_PURPLE_MIN / VIS_ARRIVE_ORANGE_MAX），录像时按实际常量填写。
    """

    def __init__(self, out_dir, live=False, every=10, fps=12, show_detect=True,
                 delay_ms=1):
        super().__init__(delay_ms=delay_ms, show_detect=show_detect)
        self.out_dir = out_dir
        self.live = bool(live)
        self.every = max(1, int(every))
        self.fps = fps
        self.n = 0
        self.writer = None
        # 录像画面第二行提示（ASCII）：本局到达判据用哪套、门槛是多少
        self._banner_hint = ""
        self.frames_dir = os.path.join(out_dir, "frames")
        os.makedirs(self.frames_dir, exist_ok=True)

    def on_frame(self, robot, frame):
        self._last_frame = frame
        if self.live:
            # ★ 必须走原版的 on_frame：里面有"暂停时阻塞等按键 / S 单步 / R 重开"
            # 那段逻辑。第一版我在这里只写了 imshow+waitKey，把暂停/单步丢了 ⇒
            # 按键读到了却没人理，表现就是"仿真器不受键盘控制"（用户报的问题）。
            # 显示与写文件统一由下面的 _show 负责（原版是 imshow，这里加写文件）。
            super().on_frame(robot, frame)
        else:
            self._write(self._compose(frame))

    def _show(self, frame, banner=None):
        view = self._compose(frame)
        if banner:
            cv2.rectangle(view, (0, 0), (view.shape[1], 34), (0, 0, 0), -1)
            cv2.putText(view, banner, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 1)
            if self._banner_hint:
                # 亚像素行：把"到达判据是什么"补在配置行下方（ASCII，字号更小）
                cv2.putText(view, self._banner_hint, (12, 47),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1)
        self._write(view)
        if self.live:
            cv2.imshow(VIEW.WIN, view)

    def release(self):
        """收尾录像（幂等）：按键退出/异常退出也一定要走到，否则 mp4 是坏的"""
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def finish(self, stats):
        ok_n = sum(1 for _, ok in stats.get("results", []) if ok)
        banner = (f"done self={ok_n}/{len(stats.get('results', []))} "
                  f"caps={stats.get('captures', 0)}")
        print(f"[rec] {banner}")
        if self._last_frame is not None:
            view = self._compose(self._last_frame)
            cv2.rectangle(view, (0, 0), (view.shape[1], 34), (0, 0, 0), -1)
            cv2.putText(view, banner, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 1)
            self._write(view, force=True)
            if self.live:
                cv2.imshow(VIEW.WIN, view)
        self.release()
        print(f"[rec] 帧目录 {self.frames_dir}（每 {self.every} 帧一张）")

    def _write(self, view, force=False):
        self.n += 1
        if self.writer is None:
            h, w = view.shape[:2]
            path = os.path.join(self.out_dir, "run.mp4")
            self.writer = cv2.VideoWriter(
                path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
            print(f"[rec] 录像 {path}  {w}x{h} @{self.fps}fps")
        self.writer.write(view)
        if force or self.n % self.every == 0:
            cv2.imwrite(os.path.join(self.frames_dir,
                                     f"f{self.n:04d}.png"), view)


def main(argv=None):
    ap = argparse.ArgumentParser(description="把一局仿真录成能看的文件")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--layout", choices=("fixed", "random"), default="fixed")
    ap.add_argument("--out", default="archive/result/rec")
    ap.add_argument("--live", action="store_true", help="同时开窗（可暂停/单步）")
    ap.add_argument("--every", type=int, default=10, help="每 N 帧存一张 PNG")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--delay", type=int, default=30,
                    help="--live 时每帧等待 ms（与 sim.nine_grid_view 同默认）")
    ap.add_argument("--baseline", action="store_true", help="三段式（默认开关）")
    ap.add_argument("--unified", action="store_true",
                    help="连续导航（不分搜索/对准/接近三段，逐帧分区决策）")
    ap.add_argument("--no-region", action="store_true",
                    help="连续导航下改用旧的“颜色占比回落”到达判据")
    ap.add_argument("--tune", action="store_true",
                    help="绿走廊×0.6 + 蓝区高 0.60")
    ap.add_argument("--primitives", choices=("nominal", "real"), default="real")
    args = ap.parse_args(argv)

    restores = []
    if args.primitives == "real":
        restores.append(AB.install_real_kinematics())
        restores.append(AB.set_level_real())
    # 算法开关必须**在起跑之前**设好（下方循环里每一局都按当前开关值跑）
    if args.unified:
        NG.UNIFIED_NAV_ENABLED = True
        NG.VIS_ZONE_ENABLED = True
    if args.no_region:
        NG.VIS_ARRIVE_REGION_ENABLED = False
    if args.tune:
        NG.VIS_ZONE_GREEN_TOP = 0.0329
        NG.VIS_ZONE_GREEN_BOT = 0.0719
        NG.VIS_ZONE_BLUE_HEIGHT = 0.60
    # 终端打印用的完整中文说明（录像顶栏用 ASCII 简写 + 第二行阈值）
    algo = ("分段导航（默认：搜索→对准→接近→到达）"
            if args.baseline or not args.unified
            else ("连续导航＋颜色占比回落到达判据" if args.no_region
                  else "连续导航＋区域到达判据"))
    banner_nav = ("NAV staged (default)" if args.baseline or not args.unified
                  else ("NAV unitary | arrive color-drop" if args.no_region
                        else "NAV unitary | arrive region"))
    if not args.baseline and args.unified:
        if args.no_region:
            hint = (f"arrive: peak color {NG.VIS_COLOR_SEEN_MIN:.2f} then "
                    f"drop below {NG.VIS_COLOR_DROP_FRAC:.2f} of peak")
        else:
            hint = (f"arrive: underfoot purple >= {NG.VIS_ARRIVE_PURPLE_MIN:.2f}"
                    f" | far orange <= {NG.VIS_ARRIVE_ORANGE_MAX:.2f}"
                    f" | L/R imbalance <= {NG.VIS_ARRIVE_ASYM_MAX:.2f}")
    else:
        hint = ""
    layout = SIM.random_layout(args.seed) if args.layout == "random" \
        else SIM.SIM_LAYOUT
    seed = args.seed
    while True:
        print(f"\n[rec] {algo}｜seed={seed}｜布局={args.layout}"
              f"｜运动原语={args.primitives}｜输出={args.out}")
        if hint:
            print(f"[rec] 录像顶栏: {banner_nav} / {hint}")
        rec = Recorder(args.out, live=args.live, every=args.every, fps=args.fps,
                       delay_ms=args.delay)
        rec._banner_hint = hint
        run = None
        try:
            # ★ 关卡每一帧的决策日志**直接打在终端**（搜索/对准/行进/到达/确认/
            #   遥测都在里面）。第一版这里用 redirect_stdout 吞掉了 —— 用户就是
            #   要看这个，吞掉纯属多此一举。
            run = SIM.run_simulation(layout=layout, seed=seed, quiet=False,
                                     viewer=rec)
            rec.finish(run.stats)
        except VIEW.ViewerRestart:
            print("[rec] 用户按 R：换 seed 重开")
            seed += 1
            layout = SIM.random_layout(seed) if args.layout == "random" \
                else SIM.SIM_LAYOUT
            continue
        except VIEW.ViewerQuit:
            print("[rec] 用户按 Q/ESC 退出（已保存到此刻的录像）")
            return 0
        finally:
            rec.release()          # 无论怎么退出都要收尾，否则 mp4 是坏的
            for f in reversed(restores):
                f()
        if run is None:
            return 1
        st = run.stats
        ok = sum(1 for _, o in st["results"] if o)
        truth_ok = sum(1 for d, o in st["results"] if o
                       and (st["panel_landing"].get(d) or [1e9])[0] <= 100 / 6)
        tight = sum(1 for d, o in st["results"] if o
                    and (st["panel_landing"].get(d) or [1e9])[0] <= 5.5)
        print(f"[rec] 自报到达 {ok}/7 ｜ 仿真确认到位(≤半格) {truth_ok}/7 ｜ "
              f"踩进开关区(≤5.5cm) {tight}/7")
        print("[rec] 逐格真值落点: "
              + ", ".join(f"{d}:{(st['panel_landing'].get(d) or ['—'])[0]}"
                          for d, _ in st["results"]))
        return 0


if __name__ == "__main__":
    sys.exit(main())
