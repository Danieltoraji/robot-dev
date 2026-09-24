#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ab_ninegrid.py —— nine_grid 多种子 A/B 对照工具（入版本库，带**活性断言**）

为什么需要它（2026-09-24 立）
------------------------------------------------
2026-09-23 有两条结论是**因为对照装置坏掉**而产生的，代价很大：

1. `tools/_tmp_recalib.py` 的 `run(seeds, zone=False)` 里 `zone` 参数**从未被引用**
   （死参数）⇒ 它扫的 17 组规划常量全是同一条二值路径 ⇒ "49/56 vs 54/56"
   其实是 A/A 对照，已作废。
2. `tools/_tmp_recalib2.py` 的 A/B 机制是**对的**，但它报出"四个阈值臂逐位相同"
   时，没人去证明"被测分支真的执行了" ⇒ 结论"分区律没有可测出的增益"被写进
   交接文档，直到 2026-09-24 才查明真正原因是 `_align_visual:2280` 的裸死区把
   `_zone_of` 的结果短路掉，**蓝档动作从未执行**。

⇒ 本工具的硬规则：**任何一臂都必须先证明"被测分支确实执行且两臂计数不同"，
否则该组结果直接标 INVALID，不参与比较。** 这条比数字本身重要。

用法
------------------------------------------------
    python tools/ab_ninegrid.py --list
    python tools/ab_ninegrid.py --arms zone_off zone_on
    python tools/ab_ninegrid.py --arms zone_off zone_on --primitives real
    python tools/ab_ninegrid.py --arms zone_off zone_on --layout random \
        --seeds 3 7 11 21 42 100 202 303 --json archive/result/ab_zone.json
    python tools/ab_ninegrid.py --arms anchor_report anchor_use --deform 1.5

设计约束
------------------------------------------------
- 只 **in-memory** 改模块属性 / 类属性（与 `_tmp_recalib.py` 同一套机制），
  不写任何文件；退出时恢复原状。
- 一臂 = (setup, teardown)。setup 返回一个"期望两臂不同的计数器名"。
- 逐种子跑 `sim.nine_grid_sim.run_simulation`，统计到达格数 / 拍照 / 动作 /
  布局是否正确 / 离场护栏是否触发。
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

import levels.nine_grid as NG  # noqa: E402
import sim.nine_grid_sim as SIM  # noqa: E402

DEFAULT_SEEDS = [3, 7, 11, 21, 42, 100, 202, 303]

# 现场实测运动原语（runbook §2 测量 1）。与 tools/_tmp_recalib.py 的 REAL 同源，
# 这里复制一份以避免依赖 gitignore 的临时脚本。
REAL = dict(fwd=2.652, left=2.497, right=2.200,
            tls=8.625, trs=5.200, sigma=0.06)


# =====================================================================
# 原语：名义 / 实测
# =====================================================================

def install_real_kinematics():
    """把仿真侧的运动原语换成现场实测值（并记录还原句柄）"""
    orig = SIM.SimNineGridRobot._apply_action

    def ap(self, name):
        th = np.radians(self.heading)
        fwd = np.array([np.sin(th), np.cos(th)])
        right = np.array([np.cos(th), -np.sin(th)])
        s = self.rng.normal
        if name == "go_forward":
            self.pos += fwd * 5.0 * (1 + s(0, 0.08))
        elif name == "go_forward_one_step":
            self.pos += fwd * REAL["fwd"] * (1 + s(0, REAL["sigma"]))
        elif name == "back_one_step":
            self.pos -= fwd * 3.2 * (1 + s(0, 0.15))
        elif name == "left_move":
            self.pos -= right * REAL["left"] * (1 + s(0, REAL["sigma"]))
        elif name == "right_move":
            self.pos += right * REAL["right"] * (1 + s(0, REAL["sigma"]))
        elif name == "turn_left":
            self.heading -= 22.0 * (1 + s(0, 0.10))
        elif name == "turn_right":
            self.heading += 25.7 * (1 + s(0, 0.10))
        elif name == "turn_left_small_step":
            self.heading -= REAL["tls"] * (1 + s(0, 0.12))
        elif name == "turn_right_small_step":
            self.heading += REAL["trs"] * (1 + s(0, 0.12))
        elif name == "stand":
            pass
        else:
            raise ValueError(name)

    SIM.SimNineGridRobot._apply_action = ap

    def restore():
        SIM.SimNineGridRobot._apply_action = orig
    return restore


def set_level_real():
    """把关卡侧的规划常量也换成实测值（返回还原句柄）

    ⚠️ 交接文档 §6.9：单独换这一半或那一半都会把基线打挂，需独立立项。
    本工具只在 `--primitives real` 下成对调用，供"实测口径"对照用，
    **不改仓库里的任何默认值**。
    """
    names = ["FORWARD_ONE_STEP_CM", "LEFT_MOVE_CM", "RIGHT_MOVE_CM",
             "FIELD_SMALL_TURN_STEP_DEG_LEFT", "FIELD_SMALL_TURN_STEP_DEG_RIGHT",
             "FIELD_SMALL_TURN_STEP_DEG", "ACTION_MODEL"]
    # 某些常量在历史版本里不存在（例如 *_LEFT/_RIGHT 是后来加的）——只还原存在的，
    # 否则 A/B 会在 setup 阶段就崩（曾真的崩过：FIELD_SMALL_TURN_STEP_DEG_LEFT）。
    old = {n: getattr(NG, n) for n in names if hasattr(NG, n)}
    names = list(old)
    old_model = dict(NG.ACTION_MODEL)
    NG.FORWARD_ONE_STEP_CM = REAL["fwd"]
    NG.LEFT_MOVE_CM = REAL["left"]
    NG.RIGHT_MOVE_CM = REAL["right"]
    NG.FIELD_SMALL_TURN_STEP_DEG_LEFT = REAL["tls"]
    NG.FIELD_SMALL_TURN_STEP_DEG_RIGHT = REAL["trs"]
    NG.FIELD_SMALL_TURN_STEP_DEG = REAL["tls"]
    NG.ACTION_MODEL["go_forward_one_step"] = ("fwd", REAL["fwd"])
    NG.ACTION_MODEL["left_move"] = ("lat", -REAL["left"])
    NG.ACTION_MODEL["right_move"] = ("lat", REAL["right"])
    NG.ACTION_MODEL["turn_left_small_step"] = ("turn", -REAL["tls"])
    NG.ACTION_MODEL["turn_right_small_step"] = ("turn", REAL["trs"])

    def restore():
        for n, v in old.items():
            setattr(NG, n, v)
        NG.ACTION_MODEL.clear()
        NG.ACTION_MODEL.update(old_model)
    return restore


# =====================================================================
# 仪器：计数 _zone_of 返回值与真实执行的动作
# =====================================================================

class Probe:
    """记录被测分支的"活性证据"：_zone_of 返回计数 + 真正下发的动作计数"""

    def __init__(self):
        self.zone = Counter()
        self.acts = Counter()
        self.anchor = Counter()
        self._orig_zone = NG.NineGridLevel._zone_of
        # 2026-09-24：分区判档改成"按示意图五参数、吃画面像素"后，真正被调用的是
        # `_zone_of_px`（`_zone_of_h` 内部转发）。两个都包，计数才反映实际路径。
        self._orig_zone_px = getattr(NG.NineGridLevel, "_zone_of_px", None)
        self._orig_act = NG.NineGridLevel._act
        self._orig_anchor = getattr(NG.NineGridLevel, "_map_anchor", None)

    def install(self):
        probe = self

        def zone(self_, yaw, off_cm=0.0, near=1.0):
            r = probe._orig_zone(self_, yaw, off_cm, near)
            probe.zone[r[0] if isinstance(r, tuple) else r] += 1
            return r

        if probe._orig_zone_px is not None:
            def zone_px(self_, px, py, w, h, prev=None):
                r = probe._orig_zone_px(self_, px, py, w, h, prev)
                probe.zone[r] += 1
                return r

            NG.NineGridLevel._zone_of_px = zone_px

        def act(self_, action, times=1):
            probe.acts[action] += 1
            return probe._orig_act(self_, action, times)

        NG.NineGridLevel._zone_of = zone
        NG.NineGridLevel._act = act
        if probe._orig_anchor is not None:
            def anc(self_, *a, **kw):
                out = probe._orig_anchor(self_, *a, **kw)
                probe.anchor["ok" if out is not None else "none"] += 1
                return out
            NG.NineGridLevel._map_anchor = anc

    def remove(self):
        NG.NineGridLevel._zone_of = self._orig_zone
        if self._orig_zone_px is not None:
            NG.NineGridLevel._zone_of_px = self._orig_zone_px
        NG.NineGridLevel._act = self._orig_act
        if self._orig_anchor is not None:
            NG.NineGridLevel._map_anchor = self._orig_anchor

    def snapshot(self):
        return {"zone": dict(self.zone), "acts": dict(self.acts),
                "anchor": dict(self.anchor)}

    def reset(self):
        self.zone.clear()
        self.acts.clear()
        self.anchor.clear()


# =====================================================================
# 臂定义：一臂 = setup() -> restore()
# =====================================================================

def arm_generic(sets):
    """通用臂：把 `--set NAME=VALUE` 的常量存进 levels.nine_grid（内存内）"""
    def setup(**kw):
        old = {}
        for k, v in sets.items():
            if hasattr(NG, k):
                old[k] = getattr(NG, k)
            setattr(NG, k, v)

        def restore():
            for k, v in old.items():
                setattr(NG, k, v)
        return restore
    return setup


def make_generic_arm(name, sets):
    # 通用臂的活性判据用**动作流**（`acts`）：改任何控制常量都必须体现在动作上，
    # 否则就是"改了但没生效"。比只比分区计数更严格。
    ARMS[name] = (arm_generic(sets), f"自定义: {sets}", "acts", True)


def _zone_common(move=None, rot=None, near=None, enabled=True,
                 hyst=None, no_rot_tail=None):
    old = {n: getattr(NG, n) for n in
           ("VIS_ZONE_ENABLED", "VIS_ZONE_MOVE_DEG", "VIS_ZONE_ROT_DEG",
            "VIS_ZONE_ROT_NEAR")}
    old_extra = {}
    for n, v in (("VIS_ZONE_HYST_DEG", hyst),
                 ("VIS_ZONE_NO_ROT_TAIL", no_rot_tail)):
        if hasattr(NG, n):
            old_extra[n] = getattr(NG, n)
        if v is not None:
            setattr(NG, n, v)
    NG.VIS_ZONE_ENABLED = enabled
    if move is not None:
        NG.VIS_ZONE_MOVE_DEG = move
    if rot is not None:
        NG.VIS_ZONE_ROT_DEG = rot
    if near is not None:
        NG.VIS_ZONE_ROT_NEAR = near

    def restore():
        for n, v in old.items():
            setattr(NG, n, v)
        for n, v in old_extra.items():
            setattr(NG, n, v)
    return restore


def arm_zone_off(**kw):
    """二值死区基线（VIS_ZONE_ENABLED=False）"""
    return _zone_common(enabled=False)


def arm_zone_on(**kw):
    """三档分区律（阈值取代码现值）"""
    return _zone_common(enabled=True)


def arm_anchor_report(**kw):
    """MAP-ANCHOR 报告模式（只算不用；只动 VIS_ZONE_ENABLED 保持一致）"""
    old = getattr(NG, "MAP_ANCHOR_ENABLED", None)
    old_ro = getattr(NG, "MAP_ANCHOR_REPORT_ONLY", None)
    had = hasattr(NG, "MAP_ANCHOR_ENABLED")
    if had:
        NG.MAP_ANCHOR_ENABLED = True
        NG.MAP_ANCHOR_REPORT_ONLY = True

    def restore():
        if had:
            NG.MAP_ANCHOR_ENABLED = old
            NG.MAP_ANCHOR_REPORT_ONLY = old_ro
    return restore


def arm_anchor_use(**kw):
    """MAP-ANCHOR 接进控制回路"""
    old = getattr(NG, "MAP_ANCHOR_ENABLED", None)
    old_ro = getattr(NG, "MAP_ANCHOR_REPORT_ONLY", None)
    had = hasattr(NG, "MAP_ANCHOR_ENABLED")
    if had:
        NG.MAP_ANCHOR_ENABLED = True
        NG.MAP_ANCHOR_REPORT_ONLY = False

    def restore():
        if had:
            NG.MAP_ANCHOR_ENABLED = old
            NG.MAP_ANCHOR_REPORT_ONLY = old_ro
    return restore


ARMS = {
    "zone_off": (arm_zone_off, "二值死区基线（zone 关）", "zone", False),
    "zone_on": (arm_zone_on, "三档分区律（zone 开，代码现值阈值）", "zone", True),
    "anchor_report": (arm_anchor_report, "MAP-ANCHOR 报告模式（只算不用）",
                      "anchor", False),
    "anchor_use": (arm_anchor_use, "MAP-ANCHOR 接控制回路", "anchor", True),
    # 活性自检臂：两个**故意不同**的阈值。它们只用来证明"装置能测出分区差异"，
    # 不声称行为改变（near=0.0 与 near=9.0 都会改变 'lat' 计数）。
    "zone_near_lo": (lambda **k: _zone_common(enabled=True, near=0.0),
                     "分区律 near=0.0（活性自检）", "zone", True),
    "zone_near_hi": (lambda **k: _zone_common(enabled=True, near=9.0),
                     "分区律 near=9.0 ⇒ 蓝档禁用（活性自检）", "zone", True),
    "zone_tail": (lambda **k: _zone_common(enabled=True, no_rot_tail=True),
                  "分区律 + 收尾段禁用旋转（VIS_ZONE_NO_ROT_TAIL）",
                  "zone", True),
    "zone_nohyst_tail": (lambda **k: _zone_common(enabled=True, hyst=0.0,
                                                  no_rot_tail=True),
                         "分区律（无迟滞）+ 收尾禁转", "zone", True),
    "tail_only": (lambda **k: _zone_common(enabled=False, no_rot_tail=True),
                  "**二值基线 + 收尾禁转**（与分区律解耦的独立变量）",
                  "zone", True),
    "zone_nohyst": (lambda **k: _zone_common(enabled=True, hyst=0.0),
                    "分区律但无迟滞（对照）", "zone", True),
}


# =====================================================================
# 主流程
# =====================================================================

def run_arm(name, seeds, layout_mode, deform, primitives):
    setup, _desc, _lv, _exp = ARMS[name]
    restore = setup()
    probe = Probe()
    probe.install()
    rows = []
    try:
        for sd in seeds:
            layout = (SIM.random_layout(sd) if layout_mode == "random"
                      else SIM.SIM_LAYOUT)
            probe.reset()
            err = None
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    r = SIM.run_simulation(layout=layout, seed=sd, quiet=True,
                                           deform=deform)
                st = r.stats
                arrived = sum(1 for _, ok in st["results"] if ok)
                ncell = len(st["results"])
                # 关卡级诊断（可选属性；老代码没有就为空）
                _lvl = getattr(r, "level", None)
                dr_hits = list(getattr(_lvl, "_dr_hits", []) or [])
                dr_land = [float(v) for v in
                           (getattr(_lvl, "_dr_hit_landing", []) or [])]
                an_hits = int(getattr(_lvl, "_anchor_arrive_hits", 0) or 0)
                an_d = [float(v) for v in
                        (getattr(_lvl, "_anchor_arrive_d", []) or [])]
                rows.append({
                    "seed": sd, "arrived": arrived, "cells": ncell,
                    "self_ok": {str(k): bool(ok) for k, ok in st["results"]},
                    "dr_hits": dr_hits, "dr_landing": dr_land,
                    "anchor_arrive_hits": an_hits, "anchor_arrive_d": an_d,
                    "layout_ok": bool(st["layout_ok"]),
                    "captures": st["captures"], "actions": st["actions"],
                    "aborted": bool(st["cell_trips"]) and ncell < 7,
                    "landing": {str(k): v for k, v in
                                (st.get("panel_landing") or {}).items()},
                    "zone": dict(probe.zone), "acts": dict(probe.acts),
                    "anchor": dict(probe.anchor),
                })
            except Exception as e:  # 布局扫失败等
                err = f"{type(e).__name__}: {e}"
                rows.append({"seed": sd, "arrived": 0, "cells": 7,
                             "layout_ok": False, "captures": 0, "actions": 0,
                             "aborted": True, "error": err,
                             "zone": dict(probe.zone), "acts": dict(probe.acts),
                             "anchor": dict(probe.anchor)})
            finally:
                probe.reset()
    finally:
        probe.remove()
        restore()
    return rows


def summarize(rows):
    # 落点残差（**安全性指标**，比到达数更重要）：到达判据放松后最危险的是
    # "提前宣布到达"——机器人站在离格心很远的地方就不再压了。真值落点只在仿真里
    # 拿得到（state.pos），故用 panel_landing 的来源字段过滤，只统计"真值"。
    land = []
    for r in rows:
        for v in (r.get("landing") or {}).values():
            if isinstance(v, (list, tuple)) and len(v) == 2 and v[1] == "真值":
                land.append(float(v[0]))
    # ★ 自报 vs 真值（2026-09-24 加）：`st["results"]` 里的 ok 是**关卡自己**的
    # 判断（go_to_panel 返回 True），**不是仿真器的判定**。以前只报这个数，
    # 会把"提前喊到达"算成成功（实测有格子自报到达却停在 25~75cm 外）。
    # 这里补上三个口径：
    #   arrived        = 自报到达（旧口径，只能当"关卡以为自己到了"）
    #   arrived_true   = 自报到达 **且** 真值落点 ≤ 半格（仿真确认在目标格上）
    #   arrived_tight  = 自报到达 **且** 真值落点 ≤ 5.5cm（仿真确认踩进开关区）
    a_true = a_tight = 0
    for r in rows:
        self_ok = r.get("self_ok") or {}
        lnd = r.get("landing") or {}
        for k, ok in self_ok.items():
            if not ok:
                continue
            v = lnd.get(str(k))
            if not (isinstance(v, (list, tuple)) and len(v) == 2
                    and v[1] == "真值"):
                continue
            d = float(v[0])
            if d <= 100.0 / 6.0:
                a_true += 1
            if d <= 5.5:
                a_tight += 1
    return {
        "arrived": sum(r["arrived"] for r in rows),
        "arrived_true": a_true,
        "arrived_tight": a_tight,
        "cells": sum(r["cells"] for r in rows),
        "captures": sum(r["captures"] for r in rows),
        "actions": sum(r["actions"] for r in rows),
        "layout_ok": sum(1 for r in rows if r["layout_ok"]),
        "per_seed": [r["arrived"] for r in rows],
        "land_n": len(land),
        "land_med_cm": (float(np.median(land)) if land else None),
        "land_max_cm": (float(np.max(land)) if land else None),
        "land_over_half_cell": int(sum(1 for v in land if v > 100.0 / 6.0)),
        "zone": dict(sum((Counter(r["zone"]) for r in rows), Counter())),
        "acts": dict(sum((Counter(r["acts"]) for r in rows), Counter())),
        "anchor": dict(sum((Counter(r["anchor"]) for r in rows), Counter())),
        # 死推判据命中证据（WS2）：命中次数 + 命中时的**真值**离格心距离
        "dr_hits": dict(sum((Counter(r.get("dr_hits") or []) for r in rows),
                            Counter())),
        "dr_land": [v for r in rows for v in (r.get("dr_landing") or [])],
        # 几何到达判据（WS4）活性证据
        "anchor_arrive_hits": sum(r.get("anchor_arrive_hits", 0) for r in rows),
        "anchor_arrive_d": [v for r in rows
                            for v in (r.get("anchor_arrive_d") or [])],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="nine_grid 多种子 A/B（带活性断言）")
    ap.add_argument("--arms", nargs="+", default=["zone_off", "zone_on"],
                    choices=sorted(ARMS))
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--layout", choices=("fixed", "random"), default="random",
                    help="fixed=SIM_LAYOUT（seed 只影响动作噪声）；"
                         "random=由 seed 生成合法随机布局")
    ap.add_argument("--primitives", choices=("nominal", "real"),
                    default="nominal")
    ap.add_argument("--deform", type=float, default=0.0, metavar="SIGMA_DEG")
    ap.add_argument("--json", default=None, help="把结果写成 JSON")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    help="把 levels.nine_grid 的模块常量改成 VALUE（内存内），"
                         "可重复；每次出现会生成一个独立的臂，名字形如 "
                         "'set:VIS_COLOR_DROP_FRAC=0.75'")
    ap.add_argument("--list", action="store_true", help="列出可用臂")
    args = ap.parse_args(argv)

    # `--set` 生成的臂追加进 ARMS。一次 `--set` 可以给**多个**赋值（逗号分隔）
    # ⇒ 它们属于同一个臂（用于"组合改动"的对照，例如同时改两个常量）。
    for item in args.set:
        sets = {}
        for part in item.split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                ap.error(f"--set 需要 NAME=VALUE 形式，收到 {part!r}")
            k, v = part.split("=", 1)
            k = k.strip()
            val = parse_value(v)
            sets[k] = val
        nm = f"set:{item.strip()}"
        make_generic_arm(nm, sets)
        args.arms.append(nm)

    if args.list:
        for k in sorted(ARMS):
            print(f"  {k:14s} {ARMS[k][1]}")
        return 0
    restores = []
    if args.primitives == "real":
        restores.append(install_real_kinematics())
        restores.append(set_level_real())
    deform = None
    if args.deform:
        deform = {"sigma_tilt_deg": args.deform, "sigma_h_cm": 0.5,
                  "max_tilt_deg": 15.0, "max_h_cm": 2.0}

    print(f"[AB] 臂={args.arms} 种子={args.seeds} 布局={args.layout} "
          f"原语={args.primitives} 形变={args.deform}")

    results = {}
    try:
        for name in args.arms:
            rows = run_arm(name, args.seeds, args.layout, deform,
                           args.primitives)
            results[name] = {"rows": rows, "sum": summarize(rows)}
            s = results[name]["sum"]
            print(f"\n[AB] 臂 {name}（{ARMS[name][1]}）")
            print(f"     自报到达 {s['arrived']}/{s['cells']}"
                  f"（**关卡自己说的**，不是仿真判定）"
                  f"  逐种子 {s['per_seed']}  布局OK {s['layout_ok']}/{len(args.seeds)}")
            print(f"     ★ 仿真确认到位（真值 ≤半格）：{s['arrived_true']}/{s['cells']}"
                  f"；踩进开关区（真值 ≤5.5cm）：{s['arrived_tight']}/{s['cells']}")
            print(f"     拍照 {s['captures']}  动作 {s['actions']}")
            print(f"     真值落点: n={s['land_n']} "
                  f"中位 {fmt_cm(s['land_med_cm'])} 最大 {fmt_cm(s['land_max_cm'])} "
                  f"超半格 {s['land_over_half_cell']} 次")
            if s.get("dr_hits") or s.get("dr_land"):
                dl = np.array(s["dr_land"]) if s.get("dr_land") else np.array([])
                extra = ("—" if dl.size == 0 else
                         f"中位 {np.median(dl):.1f}cm 最大 {dl.max():.1f}cm "
                         f">5.5cm {int((dl > 5.5).sum())}/{dl.size}")
                print(f"     死推判据命中 {s['dr_hits']}；命中时真值离格心: {extra}")
            if s.get("anchor_arrive_hits") or s.get("anchor_arrive_d"):
                ad = (np.array(s["anchor_arrive_d"])
                      if s.get("anchor_arrive_d") else np.array([]))
                print(f"     几何到达判据(WS4): 成立 {s['anchor_arrive_hits']} 次；"
                      f"锚可用帧 {ad.size}"
                      + ("" if ad.size == 0 else
                         f"，实测离格心中位 {np.median(ad):.1f}cm "
                         f"最小 {ad.min():.1f}cm"))
            print(f"     _zone_of 返回 {s['zone']}")
            print(f"     动作计数 {dict(sorted(s['acts'].items()))}")
            if s["anchor"]:
                print(f"     MAP-ANCHOR {s['anchor']}")
    finally:
        for r in reversed(restores):
            r()

    # ---- 活性断言：比数字更重要的是"被测分支真的执行了吗" ----
    print("\n" + "=" * 68)
    print("活性断言（先证明分支执行，再谈数字）")
    print("=" * 68)
    base = args.arms[0]
    bsum = results[base]["sum"]
    ok_all = True
    for name in args.arms[1:]:
        _s, _d, k, expects_behavior = ARMS[name]
        asum = results[name]["sum"]
        branch_differs = (cnt_of(bsum, k) != cnt_of(asum, k))
        acts_differ = (bsum.get("acts", {}) != asum.get("acts", {}))
        tag = "OK"
        why = []
        if not branch_differs:
            tag = "INVALID ARM"
            why.append(f"{k} 计数完全相同 ⇒ 被测分支可能根本没执行")
        if expects_behavior and not acts_differ:
            tag = "INVALID ARM"
            why.append("动作流逐项相同 ⇒ 该臂没有改变任何行为"
                       "（这正是 2026-09-23 两次错判的根因）")
        if tag != "OK":
            ok_all = False
        print(f"  {name:14s} vs {base:14s} [{k}]: "
              f"{fmt_cnt(bsum, k)} vs {fmt_cnt(asum, k)}  → {tag}")
        if acts_differ:
            diff = {a: (bsum.get('acts', {}).get(a, 0),
                        asum.get('acts', {}).get(a, 0))
                    for a in set(bsum.get('acts', {})) | set(asum.get('acts', {}))
                    if bsum.get('acts', {}).get(a, 0) != asum.get('acts', {}).get(a, 0)}
            print(f"     动作流差异（基线→该臂）: {diff}")
        for w in why:
            print(f"     ⚠ {w}")
    # ---- 两两同一性：任何两个"声称不同"的臂若逐项相同，也要报警 ----
    # （2026-09-24 加：`zone_tail` 与 `zone_on` 逐项相同时，只跟基线比是发现不了的）
    print("\n  两两同一性（claimed-different arms must differ）:")
    names = list(args.arms)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = results[names[i]]["sum"], results[names[j]]["sum"]
            same = (a["arrived"] == b["arrived"] and a["cells"] == b["cells"]
                    and a["captures"] == b["captures"]
                    and a["actions"] == b["actions"]
                    and a.get("acts") == b.get("acts"))
            exp_a = ARMS[names[i]][3]
            exp_b = ARMS[names[j]][3]
            if same and (exp_a or exp_b):
                ok_all = False
                print(f"    {names[i]} ≡ {names[j]} 逐项相同 → INVALID"
                      "（两臂声称有行为差异，但一个动作都没差）")
            else:
                print(f"    {names[i]} vs {names[j]}: "
                      f"{'相同（均为对照臂，可接受）' if same else '不同'}")
    # 分区律专有断言：开了分区却一次 'lat' 都没有 = 中间档不可达
    for name in args.arms:
        if name.startswith("zone") and name != "zone_off":
            lat = results[name]["sum"]["zone"].get("lat", 0)
            if lat == 0 and name != "zone_near_hi":
                print(f"  {name:14s} ⚠ 分区开启但 'lat' 返回 0 次"
                      " ⇒ 中间档不可达（查 _align_visual 的入口判据）")
                ok_all = False
            else:
                print(f"  {name:14s} 'lat' 返回 {lat} 次")
    print("=" * 68)

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"args": vars(args), "arms": results,
                       "liveness_ok": ok_all}, f, ensure_ascii=False, indent=2)
        print(f"[AB] 结果已写入 {args.json}")
    return 0 if ok_all else 2


def fmt_cm(v):
    return "—" if v is None else f"{v:.1f}cm"


def parse_value(v):
    """把 `--set NAME=VALUE` 的右值转成 Python 值

    ⚠️ 必须显式处理 true/false：`bool("False")` 是 **True**，所以 `--set
    VIS_ZONE_ENABLED=False` 会**悄悄打开**分区律——这正是本工具最该防的那种
    "看起来改了其实没改（或改反了）"的坑。
    """
    s = v.strip()
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def cnt_of(sumdict, key):
    """活性强度：把该计数器组的**全部键值**规范化成可比较的元组

    （2026-09-24 修：原来只比 'lat' 次数，会漏掉"只改 rot/move 不改 lat"的臂）
    """
    d = sumdict.get(key, {})
    return tuple(sorted((str(k), int(v)) for k, v in d.items()))


def fmt_cnt(sumdict, key):
    d = sumdict.get(key, {})
    if key == "zone":
        return f"lat={d.get('lat', 0)}/{dict(d)}"
    return f"{dict(d)}"


if __name__ == "__main__":
    sys.exit(main())
