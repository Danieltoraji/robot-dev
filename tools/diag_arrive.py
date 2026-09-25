# -*- coding: utf-8 -*-
"""真值诊断：把关卡"自认"的位姿/到达与仿真真值逐格对照（tools/diag_arrive.py）

**为什么需要它**（2026-09-13 假到达事故）：
本关的到达判据是**视觉推断**（低头颜色占比峰值→回落），微动开关接 ESP32、
Pi 侧读不到状态。于是出现了一种最难发现的失败："日志判到达 ✓、机器人其实还
在离格心 89.5cm 处"。只看 `results` 里的 7/7 完全看不出来。

本工具跑三个场景，逐格打出五个**互相独立**的数字，用来区分两类完全不同的故障：

  | 数字             | 含义                              | 故障指向                     |
  |------------------|-----------------------------------|------------------------------|
  | 落点真值         | 跑完该格时真值离该格格心          | 大 ⇒ **假到达**（身份误判）  |
  | 位姿漂移         | 跑完该格时 |死推位姿 − 真值|      | 大而落点小 ⇒ 位姿不可信      |
  | 到达残差         | 关卡自报的落点残差（死推）        | 与落点真值对比 ⇒ 残差是否可信|
  | 峰值帧证据       | 颜色峰值那一帧的面板 hull / 画幅  | <0.08 ⇒ 到达判断条件会拒绝（正确） |
  | 到达依据         | 关卡自报"凭什么算到达"            | 看是哪一条判据接手           |

**怎么读**：`落点真值` 是第一优先级——它超半格（16.7cm）就是假到达，
不管 `结果` 列是不是 ✓。`位姿漂移` 大但 `落点真值` 小是正常的（形变下位姿
会漂，视觉伺服仍能把机器人做对）；**不要用位姿去否决到达**（会误杀正确到达，
实测：面板5 位姿漂移 105cm 但真值离格心只有 13cm）。

用法：
    python tools/diag_arrive.py                # 三场景
    python tools/diag_arrive.py --scenes BASE  # 只跑一个
    python tools/diag_arrive.py --quiet        # 只出表，不打印关卡日志
"""

import argparse
import contextlib
import io
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import levels.nine_grid as ng                      # noqa: E402
from levels.nine_grid_shared import GRID_CELL_CM, grid_cell_center   # noqa: E402
from vision.nine_grid_detector import NineGridDetector        # noqa: E402
from sim.nine_grid_sim import run_simulation                 # noqa: E402

# 场景定义（与 tests/test_nine_grid_deform.py 的计分场景保持一致）
SCENES = {
    "BASE": {},
    "RW": {"deform": {"sigma_tilt_deg": 3.0, "sigma_h_cm": 0.5,
                      "max_tilt_deg": 15.0, "max_h_cm": 2.0}},
    "STEP3": {"deform": {"sigma_tilt_deg": 0.0, "sigma_h_cm": 0.0,
                         "step_at_digit": 3, "step_tilt_deg": 15.0,
                         "step_h_cm": -2.0,
                         "max_tilt_deg": 15.0, "max_h_cm": 2.0}},
    "STEP_LEGACY": {"deform": {"sigma_tilt_deg": 0.0, "sigma_h_cm": 0.0,
                               "step_after_actions": 60, "step_tilt_deg": 8.0,
                               "step_h_cm": -2.0,
                               "max_tilt_deg": 15.0, "max_h_cm": 2.0}},
}
HALF_CELL_CM = GRID_CELL_CM / 2.0


def _install_probes(rows):
    """给关卡/检测器挂只读探针，收集"到达段"的峰值帧证据

    探针只读取，不改变任何控制流——所以用它得到的结论可以直接对应生产行为。
    """
    ctx = {"digit": None, "peak": 0.0, "peak_cover": -1.0}

    _orig_cr = NineGridDetector.color_ratio

    def color_ratio(self, frame, color_name):
        v = float(_orig_cr(self, frame, color_name))
        if ctx["digit"] is not None and v > ctx["peak"]:
            ctx["peak"] = v
            ctx["peak_cover"] = -1.0          # 峰值帧换人，证据重等
            ctx["_peak_frame_pending"] = True
        return v

    _orig_dp = NineGridDetector.detect_panels

    def detect_panels(self, frame, colors=None, arbitrate=False,
                      drop_border=False, shape=False):
        out = _orig_dp(self, frame, colors=colors, arbitrate=arbitrate,
                       drop_border=drop_border, shape=shape)
        if ctx["digit"] is not None and ctx.get("_peak_frame_pending") and out:
            area = float(frame.shape[0] * frame.shape[1])
            best = max(float(o.hull_area) for o in out)
            ctx["peak_cover"] = best / max(area, 1.0)
            ctx["_peak_frame_pending"] = False
        return out

    _orig_ar = ng.NineGridLevel._walk_until_underfoot

    def arrive_visual(self, digit, t_end):
        ctx.update({"digit": digit, "peak": 0.0, "peak_cover": -1.0,
                    "_peak_frame_pending": False})
        r = _orig_ar(self, digit, t_end)
        rows.append({
            "digit": digit,
            "cell": self.digit_cell.get(digit),
            "ok": bool(r),
            "peak_cover": ctx["peak_cover"],
            "evidence": self._arrive_evidence,
        })
        ctx["digit"] = None
        return r

    NineGridDetector.color_ratio = color_ratio
    NineGridDetector.detect_panels = detect_panels
    ng.NineGridLevel._walk_until_underfoot = arrive_visual


def run_scene(tag, quiet=False):
    """跑一个场景，返回逐格诊断行"""
    rows = []
    _install_probes(rows)
    kw = dict(SCENES[tag])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run = run_simulation(seed=3, quiet=False, **kw)
    return run, rows, buf.getvalue()


def report(tag, run, rows):
    stats = run.stats
    land = stats.get("panel_landing", {})
    n_ok = sum(1 for _, ok in stats["results"] if ok)
    print(f"\n=== {tag} === 确认 {n_ok}/{len(stats['results'])}  "
          f"布局 {'OK' if stats['layout_ok'] else 'FAIL'}  "
          f"拍照 {stats['captures']}  动作 {stats['actions']}  "
          f"形变 俯仰{stats['deform_tilt_deg']:+.1f}°/高度"
          f"{stats['deform_height_cm']:+.1f}cm")
    if stats.get("cell_trips"):
        print(f"  ⚠ 被安全项终止的格: {stats['cell_trips']}")
    print(f"  {'面板':<4}{'格':<4}{'结果':<6}{'落点真值':>9}{'来源':>6}"
          f"{'峰值证据':>9}  到达依据")
    fake = []
    gross_fake = []      # 跨格级假到达（> 一整格）：**必须为 0**，否则判据形同虚设
    near_miss = []       # 半格 ~ 一整格：落点偏差，可能压不到微动开关，只记录
    for r in rows:
        d = r["digit"]
        rec = land.get(d)
        off = rec[0] if rec else float("nan")
        src = rec[1] if rec else "?"
        ev = (r["evidence"] or "（未到达）")
        flag = ""
        if r["ok"] and off > HALF_CELL_CM:
            gross = off > GRID_CELL_CM
            flag = "  ← 假到达(跨格)!" if gross else "  ← 落点超半格"
            (gross_fake if gross else near_miss).append((d, off))
        cover = ("—" if r["peak_cover"] < 0 else f"{r['peak_cover']:.3f}")
        print(f"  {d:<4}{str(r['cell']):<4}{'✓' if r['ok'] else '✗':<6}"
              f"{off:>8.1f}cm{src:>6}{cover:>9}  {ev[:64]}{flag}")
    if gross_fake:
        print(f"  ⚠ **跨格级假到达**（结果 ✓ 但真值离格心超一整格 "
              f"{GRID_CELL_CM:.1f}cm）: "
              + ", ".join(f"面板{d} {o:.1f}cm" for d, o in gross_fake)
              + "  ← 身份误判，必须修判据")
    if near_miss:
        print(f"  ⚠ 落点超半格（{HALF_CELL_CM:.1f}cm，仍在同格内）: "
              + ", ".join(f"面板{d} {o:.1f}cm" for d, o in near_miss)
              + "  ← 可能压不到微动开关；不是身份误判，据此判断是否收紧判据")
    if not gross_fake and not near_miss:
        print("  ✓ 无假到达（所有 ✓ 的面板落点真值都在半格内）")
    return gross_fake, near_miss


def main(argv=None):
    ap = argparse.ArgumentParser(description="nine_grid 真值到达诊断")
    ap.add_argument("--scenes", default="BASE,RW,STEP3,STEP_LEGACY",
                    help="逗号分隔，可选 " + "/".join(SCENES))
    ap.add_argument("--quiet", action="store_true", help="不打印关卡逐行日志")
    ap.add_argument("--log", metavar="TAG",
                    help="把指定场景（如 STEP3）的关卡日志全量打印出来")
    args = ap.parse_args(argv)

    total_gross, total_near = 0, 0
    for tag in [t.strip() for t in args.scenes.split(",") if t.strip()]:
        if tag not in SCENES:
            print(f"未知场景 {tag}，可选: {list(SCENES)}")
            continue
        run, rows, log = run_scene(tag)
        if args.log == tag:
            print(log)
        g, n = report(tag, run, rows)
        total_gross += len(g)
        total_near += len(n)
    print(f"\n合计：跨格级假到达 {total_gross} 处"
          f"（必须为 0）、落点超半格 {total_near} 处（容忍记录）")
    if total_gross:
        print("← 有身份误判，必须修判据，不要靠调参")
    return 1 if total_gross else 0


if __name__ == "__main__":
    sys.exit(main())
