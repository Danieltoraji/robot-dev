#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""只同步"新版 nine_grid（统一决策）"这条路线要的文件
（tools/sync_nine_grid_to_robot.py）

为什么要有这个：`sync_to_robot.py` 的默认集合是
    main.py + debug.sh + core/ vision/ levels/ tools/ models/
那条命令会连带传**全部关卡**（goodluck / stairs_hurdle / nine_grid_three_stage /
nine_grid_original / press_button / football_shooter…）、`tools/` 下 100+ 个文件、
40MB 的 models，**把机器人上其它关卡的现场手调值一起覆盖掉**。
本脚本把范围钉死在"让 `python main.py nine_grid` 跑起来"的最小集上。

=====================================================================
最小集怎么来的（2026-09-27 实测，不是读代码猜的）
=====================================================================
机器人当时是一棵**跨分支拼起来的**树：core/ vision/ 是 press_button 时期的老版本，
levels/ 里混着另一分支的 stairs_hurdle / press_button。实测结果：

  1. 基线（机器人现状）跑不了新版：
     `import levels.nine_grid` →
     `ImportError: cannot import name 'CAM_HEIGHT_STANDING_CM' from 'core.camera_config'`
     同一个错误也让机器人上的 `levels.nine_grid_three_stage` 起不来。
  2. 把候选文件逐个换成本地版做**留一法**，能复现 import 失败的才是必修项：
       core/camera_config.py      ← CAM_HEIGHT_STANDING_CM 等新常量
       core/ground_homography.py  ← cell_index
       core/robot_core.py         ← CAM_AUTO_EXPOSURE_ENABLED
       vision/nine_grid_detector.py ← normalize_illumination（整模块由 472 行涨到 999 行）
  3. 只做 import 试验会漏掉**运行期**才 import 的东西：
     `vision/nine_grid_detector.py` 的形状仲裁在运行时才
     `from .digit_recognizer import DigitRecognizer` 并调用
     `match_mask()` —— 机器人上那份老 digit_recognizer 没有 match_mask ⇒ 必修。
  4. `main.py` + `core/trace.py`：
     · main.py 决定 `python main.py nine_grid` 能不能起来、跑完有没有
       `archive/result/real_trace_*.txt`（本地版 TRACE_ENABLED=True）；
     · trace.py 少了本地那三处 `getattr(level, "WALLS"/"ROUTE"/"tag_poses", …)`
       兜底，九宫格存轨迹图会抛 "module 'levels.nine_grid' has no attribute
       'WALLS'" —— 机器人上那些 35407 字节的 real_trajectory_*.png 就是这么来的。
  5. 机器人上**已经逐字节一致**、因此不必重复传的：
     levels/nine_grid_shared.py、levels/nine_grid_three_stage.py、
     levels/nine_grid_original/*、models/nine_grid/*、core/paths.py、
     vision/red_line_detector.py、archive/result/ninegrid_homography.json。
     （它们仍列在 ROUTE_FILES 里：换新 SD 卡/重装机器人时这套清单要能独立跑通。）

同步后已在本地用"机器人树"预演过：levels.nine_grid / nine_grid_three_stage /
nine_grid_original / goodluck / stairs_hurdle / press_button / main **全部 import 通过**。

⚠️ 关于 `--full`：本仓库默认**总是全量推这几个文件**。本地清单
（archive/sync_manifest.json）记的是"我从这个工作树推过什么"，而机器人上的同名文件
可能被**别的工作树**后推覆盖过（2026-09-27 实测：本地清单说 camera_config.py 已同步，
机器人上却是老版本）。清单在这种情况下会静默跳过，所以这里不给它机会。

用法（PC 仓库根目录）：

    python tools/sync_nine_grid_to_robot.py --check     # 干跑：只列将传什么
    python tools/sync_nine_grid_to_robot.py             # 真传（总是全量推这 8 个）
    python tools/sync_nine_grid_to_robot.py --verify    # 只做同步后的逐字节核对
    python tools/sync_nine_grid_to_robot.py --pull      # 传完顺手把现场 trace 拉回来

机器人地址默认 http://192.168.31.209:8888（可用环境变量 ROBOT_HOST 覆盖）。
"""

import argparse
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

# ★ 最小集（改这里之前先重跑留一法，别凭感觉加文件）
ROUTE_FILES = [
    # ---- 路线本体 ----
    "levels/nine_grid.py",             # 新版（统一决策）本体
    "levels/nine_grid_shared.py",      # 共用基类（机器人上已一致，防换卡重装）
    "levels/nine_grid_three_stage.py",  # main.py 会 import 它，缺了 main 起不来
    # ---- 它 import 的 core / vision（留一法实测的必修项）----
    "core/camera_config.py",
    "core/ground_homography.py",
    "core/robot_core.py",
    "vision/nine_grid_detector.py",
    "vision/digit_recognizer.py",      # 运行期形状仲裁用（match_mask）
    # ---- 入口与 trace ----
    "main.py",                         # 关卡注册表（含 press_button）+ TRACE_ENABLED
    "core/trace.py",                   # 九宫格存轨迹图要那三处 getattr 兜底
]

# 同步完做逐字节核对（离线核对本地文件是否真的等于远端）
VERIFY_FILES = ROUTE_FILES


def _run(cmd):
    return subprocess.call(cmd, cwd=_REPO)


def cmd_sync(check=False):
    paths = ROUTE_FILES
    missing = [p for p in paths if not os.path.isfile(os.path.join(_REPO, p))]
    if missing:
        print("[必要文件] ❌ 下列文件在本地不存在，先解决再同步：")
        for p in missing:
            print(f"    {p}")
        return 2

    cmd = [sys.executable, os.path.join(_HERE, "sync_to_robot.py"),
           "--paths"] + paths
    # 总是忽略本地清单：机器人上的同名文件可能被别的工作树推过（见文件头）
    cmd.append("--full")
    if check:
        cmd.append("--check")

    print(f"[必要文件] 范围：{len(paths)} 个文件"
          f"（十关只传九宫格这一条路线；tools/ models/ 一概不动）")
    print(f"[必要文件] 命令：{' '.join(cmd[1:])}")
    print("-" * 68)
    rc = _run(cmd)
    if rc != 0:
        print(f"[必要文件] ❌ sync_to_robot.py 退出码 {rc}")
        return rc
    return 0


def cmd_verify():
    """把远端这几个文件拉回来逐字节比对（部署一致性：不信任 PUT 的回执）"""
    sys.path.insert(0, _HERE)
    from sync_to_robot import DEFAULT_HOST, DEFAULT_PASSWORD, JupyterClient

    root = os.environ.get("ROBOT_REMOTE_ROOT", "Robot_Competition")
    client = JupyterClient(DEFAULT_HOST, DEFAULT_PASSWORD, timeout=120)
    print(f"[核对] 远端 {DEFAULT_HOST} 的 {root}/ —— 逐字节比对 "
          f"{len(VERIFY_FILES)} 个文件")
    print("-" * 68)
    bad = 0
    for rel in VERIFY_FILES:
        local_path = os.path.join(_REPO, rel.replace("/", os.sep))
        with open(local_path, "rb") as f:
            local = f.read()
        remote = client.get_file_bytes(f"{root}/{rel}")
        if remote is None:
            print(f"  [远端缺失] {rel}")
            bad += 1
        elif remote == local:
            print(f"  [一致] {rel} ({len(local)} 字节)")
        else:
            print(f"  [不一致] {rel}：本地 {len(local)} 字节 / 远端 {len(remote)} 字节")
            bad += 1
    print("-" * 68)
    if bad:
        print(f"[核对] ❌ {bad} 个文件没对齐")
        return 1
    print("[核对] ✅ 全部逐字节一致")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="只同步新版 nine_grid（统一决策）这条路线要的 10 个文件")
    ap.add_argument("--check", action="store_true",
                    help="干跑：只列出将同步的文件，不写远端（每次改完代码先跑）")
    ap.add_argument("--verify", action="store_true",
                    help="只做核对：把远端这几个文件拉回来逐字节比对，不写远端")
    ap.add_argument("--no-verify", action="store_true",
                    help="真传之后不自动核对（默认会核对）")
    ap.add_argument("--pull", action="store_true",
                    help="传完顺手把机器人上的现场 trace 拉回 PC 存档")
    args = ap.parse_args()

    if args.verify:
        return cmd_verify()

    rc = cmd_sync(check=args.check)
    if rc != 0:
        return rc

    if args.check:
        print("[必要文件] （干跑，未写远端；确认无误后去掉 --check）")
        return 0

    if not args.no_verify:
        print("-" * 68)
        rc = cmd_verify()
        if rc != 0:
            return rc

    if args.pull:
        print("-" * 68)
        print("[必要文件] 拉回现场 trace（archive/result/real_trace_*.txt）…")
        pull = [sys.executable, os.path.join(_HERE, "pull_from_robot.py"),
                "--remote", "archive/result", "--dest", "archive/result"]
        rc = _run(pull)
        if rc != 0:
            print(f"[必要文件] ⚠️ 拉取退出码 {rc}（同步本身已成功且已核对）")

    print("-" * 68)
    print("[必要文件] ✅ 完成")
    print("    上机验证：ssh 到机器人后")
    print("      cd /home/pi/Robot_Competition")
    print("      python3 main.py nine_grid              # 新版（统一决策）")
    print("      python3 main.py nine_grid_three_stage  # 三段式（对照）")
    print("      python3 main.py nine_grid_original     # 参考原版（对照）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
