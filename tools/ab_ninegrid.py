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
    python tools/ab_ninegrid.py --arms unified three_stage --seeds 3 7 11 21
    python tools/ab_ninegrid.py --arms unified three_stage --primitives real
    python tools/ab_ninegrid.py --arms unified three_stage --layout random \
        --seeds 3 7 11 21 42 100 202 303 --json archive/result/ab.json
    python tools/ab_ninegrid.py --arms unified --deform 1.5
    # 临时改常量对照（内存内，不改仓库默认值）；
    # `--set` 自己会追加一根独立的臂，臂名形如 set:shared.XXX=0.75
    python tools/ab_ninegrid.py --arms unified \
        --set shared.VIS_COLOR_DROP_FRAC=0.75

设计约束
------------------------------------------------
- 只 **in-memory** 改模块属性 / 类属性（与 `_tmp_recalib.py` 同一套机制），
  不写任何文件；退出时恢复原状。
- 一臂 = (setup, teardown)。setup 返回一个"期望两臂不同的计数器名"。
- 逐种子跑 `sim.nine_grid_sim.run_simulation`，统计到达格数 / 拍照 / 动作 /
  布局是否正确 / 离场护栏是否触发。
- 选算法只走 `run_simulation(three_stage=)`：两条路线已是两个模块，
  臂的名字就是路线名。
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

import levels.nine_grid as NG  # noqa: E402  统一决策（默认路线）
import levels.nine_grid_shared as SH  # noqa: E402  两条路线共用
import levels.nine_grid_three_stage as THREE  # noqa: E402  三段式
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
    # 常量按归属分开改：动作模型与步长在共用模块，对准死区那条不变量在三段式。
    old_sh = {n: getattr(SH, n) for n in
              ("FORWARD_ONE_STEP_CM", "LEFT_MOVE_CM", "RIGHT_MOVE_CM")}
    old_three = {n: getattr(THREE, n) for n in
                 ("FIELD_SMALL_TURN_STEP_DEG",) if hasattr(THREE, n)}
    old_model = dict(SH.ACTION_MODEL)
    SH.FORWARD_ONE_STEP_CM = REAL["fwd"]
    SH.LEFT_MOVE_CM = REAL["left"]
    SH.RIGHT_MOVE_CM = REAL["right"]
    if hasattr(THREE, "FIELD_SMALL_TURN_STEP_DEG"):
        THREE.FIELD_SMALL_TURN_STEP_DEG = REAL["tls"]
    SH.ACTION_MODEL["go_forward_one_step"] = ("fwd", REAL["fwd"])
    SH.ACTION_MODEL["left_move"] = ("lat", -REAL["left"])
    SH.ACTION_MODEL["right_move"] = ("lat", REAL["right"])
    SH.ACTION_MODEL["turn_left_small_step"] = ("turn", -REAL["tls"])
    SH.ACTION_MODEL["turn_right_small_step"] = ("turn", REAL["trs"])

    def restore():
        for n, v in old_sh.items():
            setattr(SH, n, v)
        for n, v in old_three.items():
            setattr(THREE, n, v)
        SH.ACTION_MODEL.clear()
        SH.ACTION_MODEL.update(old_model)
    return restore


# =====================================================================
# 仪器：计数 _zone_of 返回值与真实执行的动作
# =====================================================================

class Probe:
    """记录被测分支的"活性证据"：判档返回计数 + 真正下发的动作计数

    两条路线现在是两个类，所以探针挂在**共用基类**上（一次覆盖两条路线）；
    像素判档只有统一决策那条有，单独挂它。
    """

    def __init__(self):
        self.zone = Counter()
        self.acts = Counter()
        self.anchor = Counter()
        self._orig_act = SH.NineGridShared._act
        self._orig_anchor = SH.NineGridShared._map_anchor
        self._orig_zone_px = getattr(NG.NineGridLevel, "_zone_of_px", None)

    def install(self):
        probe = self

        def act(self_, action, times=1):
            probe.acts[action] += 1
            return probe._orig_act(self_, action, times)

        def anc(self_, *a, **kw):
            out = probe._orig_anchor(self_, *a, **kw)
            probe.anchor["ok" if out is not None else "none"] += 1
            return out

        SH.NineGridShared._act = act
        SH.NineGridShared._map_anchor = anc
        if probe._orig_zone_px is not None:
            def zone_px(self_, px, py, w, h, prev=None):
                r = probe._orig_zone_px(self_, px, py, w, h, prev)
                probe.zone[r] += 1
                return r

            NG.NineGridLevel._zone_of_px = zone_px

    def remove(self):
        SH.NineGridShared._act = self._orig_act
        SH.NineGridShared._map_anchor = self._orig_anchor
        if self._orig_zone_px is not None:
            NG.NineGridLevel._zone_of_px = self._orig_zone_px

    def snapshot(self):
        return {"zone": dict(self.zone), "acts": dict(self.acts),
                "anchor": dict(self.anchor)}

    def reset(self):
        self.zone.clear()
        self.acts.clear()
        self.anchor.clear()


# =====================================================================
# 臂定义：一根臂 = 一条决策路线（或 `--set` 现场改常量）
# =====================================================================
# 两条路线拆成两个模块之后，"选哪条"不再靠改开关，而是 run_simulation(three_stage=)
# ⇒ 路线臂不需要 setup/restore，只需要一个名字。
#
# 已经答过的问题（旧臂的结论留档，别再重问一遍）：
#   · 三档分区律 vs 二值死区：16 种子统计上不可区分（78.6% vs 78.4%），且多花
#     约 5% 拍照 ⇒ 分区律没有可测增益（旧臂 zone_off/zone_on/zone_near_*/
#     zone_tail/zone_nohyst/zone_nohyst_tail/tail_only 的结论）。
#   · 迟滞：名义口径有益、实测口径有害，差异都在 ±3 格（≈1σ）⇒ 不足以定论。
#   · 收尾禁转：与"无迟滞"逐项完全相同（工具当时正确判 INVALID）⇒ 恒等操作。
#   · 地图锚：只算不用时指标逐位相同，可用率仅 5~16% ⇒ 现在只作核验与遥测。


def _noop_setup(**_kw):
    return lambda: None


ARMS = {
    "unified": (_noop_setup, "统一决策（三档分区＋一个循环，默认路线）",
                "acts", True),
    "three_stage": (_noop_setup, "三段式（现场发货的稳定实现）", "acts", True),
}


def arm_generic(sets):
    """通用臂：把 `--set 模块.常量=VALUE` 的值存进对应模块（内存内）

    模块名：shared（两条路线共用）/ level（统一决策）/ three_stage（三段式）。
    不带模块名时三个模块都试，命中多处直接报错——避免"改了但没生效"。
    """
    mods = {"shared": SH, "level": NG, "three_stage": THREE}
    resolved = []
    for key, val in sets.items():
        if "." in key:
            mname, cname = key.split(".", 1)
            if mname not in mods:
                raise KeyError(f"未知模块 {mname!r}（可用：{sorted(mods)}）")
            resolved.append((mods[mname], cname, val))
            continue
        hits = [m for m in mods.values() if hasattr(m, key)]
        if not hits:
            raise KeyError(f"三个模块里都没有常量 {key!r}")
        resolved.append((hits[-1], key, val))

    def setup(**_kw):
        old = [(m, n, getattr(m, n)) for m, n, _v in resolved]
        for m, n, v in resolved:
            setattr(m, n, v)
        return lambda: [setattr(m, n, v) for m, n, v in old]
    return setup


def make_generic_arm(name, sets):
    # 通用臂的活性判据用**动作流**（acts）：改任何控制常量都必须体现在动作上，
    # 否则就是"改了但没生效"。
    ARMS[name] = (arm_generic(sets), f"自定义: {sets}", "acts", True)


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
                                           deform=deform,
                                           three_stage=(name == "three_stage"))
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
    ap.add_argument("--arms", nargs="+", default=["unified", "three_stage"],
                    choices=sorted(ARMS))
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--layout", choices=("fixed", "random"), default="random",
                    help="fixed=SIM_LAYOUT（seed 只影响动作噪声）；"
                         "random=由 seed 生成合法随机布局")
    ap.add_argument("--primitives", choices=("nominal", "real"),
                    default="nominal")
    ap.add_argument("--deform", type=float, default=0.0, metavar="SIGMA_DEG")
    ap.add_argument("--json", default=None, help="把结果写成 JSON")
    ap.add_argument("--set", action="append", default=[],
                    metavar="[模块.]NAME=VALUE",
                    help="把某个模块的常量改成 VALUE（内存内），可重复；模块 ∈ "
                         "shared / level / three_stage，省略则三个都找。"
                         "每次出现生成一根独立的臂，名字形如 "
                         "'set:shared.VIS_COLOR_DROP_FRAC=0.75'")
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
    """把 `--set [模块.]NAME=VALUE` 的右值转成 Python 值

    ⚠️ 必须显式处理 true/false：`bool("False")` 是 **True**，所以 `--set
    shared.某开关=False` 会**悄悄打开**它——这正是本工具最该防的那种
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
