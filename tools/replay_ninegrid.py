# -*- coding: utf-8 -*-
"""真实照片重放（tools/replay_ninegrid.py）——布局扫的现场验收工具

用途：不接机器人，把现场照片按扫描时序喂给**真实的**
`NineGridLevel.layout_scan()`，验收：
  1. 布局是否解出、是否与照片读图一致（`--expect` 做断言）；
  2. 参数自标定常数（安装偏移 / 相机高度）是否落在合理范围；
  3. 位姿自举是否成功（刚体残差、位置、航向）；
  4. `--diag` 逐帧检出 / 逐数字跨帧离散度 / 参数网格投票 —— 用来把
     "照片映射错" 与 "算法错" 分开归因。

用法：
    python tools/replay_ninegrid.py                 # 夹具目录 + 默认映射
    python tools/replay_ninegrid.py --diag          # 逐帧/逐数字诊断
    python tools/replay_ninegrid.py --expect        # 与照片读图对照（不符则非零退出）
    python tools/replay_ninegrid.py --map head-major
    python tools/replay_ninegrid.py --fixtures archive/result/field_photos

照片→(俯仰档, 头部档) 映射（**用户 2026-09-11 表示未确认**，故默认为假设值，
可覆盖）：文件名顺序 = 扫描时序，前 5 张 pitch=PITCH_NAV(1200)，head 依次
1500/1950/1050/2200/800；后 5 张 pitch=PITCH_DOWN(1040)，head 同序。

已知的现场照片读图真值（本次接手勘察逐张目视 + 与修复后检出对齐）：
    板南侧机位、朝北：远排 = 紫6 / 粉7 / 空，中排 = 蓝5 / 黄3 / 橙2，
    近排 = 绿4 / 空(被机身遮挡) / 红1
    → 数字→格 = PHOTO_TRUTH（见下）
注意：**该机位下"位置6 被绿4 占据"是预期现象**（机位不在入口侧，或标号
约定与裁判 .ino 不同），真机从入口进场时扫描结果必须满足"格6 恒空"。
"""

import argparse
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import cv2

from core.camera_config import HEAD_CENTER
from levels.nine_grid import (
    NineGridLevel, PITCH_NAV, PITCH_DOWN,
)

# 现场照片读图真值（板南侧机位；见模块 docstring）
PHOTO_TRUTH = {1: 8, 2: 5, 3: 4, 4: 6, 5: 3, 6: 0, 7: 1}
# 孤立照（扫描前不同位姿，重放必须剔除）
EXCLUDE = ("photo_1789022559.jpg", "_ruler", "_debug")
HEAD_ORDER = (HEAD_CENTER, 1950, 1050, 2200, 800)

DEFAULT_FIXTURES = ("tests/fixtures/field_photos",
                    "archive/result/field_photos")


def find_photos(fixtures=None):
    """找现场照片（排除孤立照/调试图），返回按文件名排序的路径列表"""
    dirs = [fixtures] if fixtures else list(DEFAULT_FIXTURES)
    for d in dirs:
        p = Path(d) if os.path.isabs(d) else _REPO / d
        if not p.is_dir():
            continue
        photos = [q for q in sorted(p.glob("photo_1789022*.jpg"))
                  if not any(k in q.name for k in EXCLUDE)]
        if photos:
            return photos
    return []


def build_schedule(photos, mode="schedule"):
    """照片 → [(pitch, head, path)]；mode 见 --map"""
    n = len(photos)
    if mode == "head-major":
        # 每档头部各拍两档俯仰（先全部 pitch1200 再 pitch1040 以外的排法）
        order = []
        for head in HEAD_ORDER:
            for pitch in (PITCH_NAV, PITCH_DOWN):
                order.append((pitch, head))
        return [(order[k][0], order[k][1], photos[k])
                for k in range(min(n, len(order)))]
    order = [(pitch, head) for pitch in (PITCH_NAV, PITCH_DOWN)
             for head in HEAD_ORDER]
    if mode == "nav-only":
        order = [(PITCH_NAV, head) for head in HEAD_ORDER]
    elif mode == "down-only":
        order = [(PITCH_DOWN, head) for head in HEAD_ORDER]
    return [(order[k][0], order[k][1], photos[k])
            for k in range(min(n, len(order)))]


class ReplayRobot:
    """照片重放机器人：只实现关卡用到的 I/O 接缝（不接 RobotState/AGC）

    动作 `act`/`run_action` 只记日志——照片是**静止机位**拍的，重放时"前进一步"
    没有新视角，重扫会重新拿到同一组照片（工具会打印复用次数，避免误读）。
    """

    def __init__(self, schedule, verbose=True):
        self.schedule = list(schedule)
        self.by_key = {}
        # 同一 (pitch, head) 可能有多个视角的照片（现场多拍时）
        for pitch, head, path in self.schedule:
            self.by_key.setdefault((int(pitch), int(head)), []).append(path)
        self.current_head_pulse = HEAD_CENTER
        self.current_pitch_pulse = PITCH_NAV
        self.verbose = verbose
        self.captures = 0
        self.reused = 0
        self.actions = []
        self.frames = {}

    # ---- 关卡用到的接缝 ----
    def set_pitch(self, pulse, move_time_ms=500):
        self.current_pitch_pulse = int(pulse)

    def set_head(self, pulse, move_time_ms=500):
        self.current_head_pulse = int(pulse)

    def run_action(self, name, times=1):
        self.actions.append((name, times))

    def act(self, name, times=1):
        self.run_action(name, times)

    def capture_frame(self):
        key = (int(self.current_pitch_pulse), int(self.current_head_pulse))
        lst = self.by_key.get(key)
        if not lst:
            return None
        idx = self.frames.get(key, 0)
        if idx >= len(lst):
            idx = len(lst) - 1
            self.reused += 1
        self.frames[key] = idx + 1
        self.captures += 1
        return cv2.imread(str(lst[idx]))


def sweep_and_collect(robot, level, diag=False):
    """按关卡的扫描顺序取一遍观测（与 layout_scan 同序，便于诊断）

    返回 pix_obs = [(pitch, head, digit, 观测像素, 裁切?), ...]
    """
    pix_obs = []
    for pitch, head, _path in robot.schedule:
        robot.set_pitch(pitch)
        robot.set_head(head)
        frame = robot.capture_frame()
        if frame is None:
            continue
        obs = level.detector.detect_panels(frame, arbitrate=True,
                                           drop_border=False)
        if diag:
            items = " ".join(
                "%d@(%.0f,%.0f)%s" % (
                    o.digit,
                    (o.hull_centroid_px if o.clipped else o.center_px)[0],
                    (o.hull_centroid_px if o.clipped else o.center_px)[1],
                    "C" if o.clipped else "")
                for o in sorted(obs, key=lambda o: o.digit))
            print("  [帧] pitch=%d head=%4d 检出 %d: %s"
                  % (pitch, head, len(obs), items))
        for o in obs:
            px = o.hull_centroid_px if o.clipped else o.center_px
            pix_obs.append((pitch, head, o.digit, px, o.clipped))
    return pix_obs


def run_diag(robot, level):
    """逐帧检出 + 参数网格拟合诊断（不跑 FSM）"""
    print("[diag] 逐帧检出（C = 被画幅裁切，用凸包质心观测）:")
    pix_obs = sweep_and_collect(robot, level, diag=True)
    seen = sorted({e[2] for e in pix_obs})
    clean = {e[2] for e in pix_obs if not e[4]}
    print(f"[diag] 观测汇总: {len(pix_obs)} 条，数字 {seen}，"
          f"未裁切覆盖 {len(clean)}/7")
    fit = level._lattice_grid_fit(pix_obs)
    print(f"[diag] 拟合: cells={fit.cells}")
    print(f"[diag] 说明: {fit.info}")
    for w in fit.warnings:
        print(f"[diag] 警告: {w}")
    if fit.spread_cm:
        print("[diag] 逐数字跨帧离散（cm，含观测条数）:")
        for d in sorted(fit.spread_cm):
            n = sum(1 for e in pix_obs if e[2] == d)
            print("      数字%d: 离散 %.1f  (%d 条%s)"
                  % (d, fit.spread_cm[d], n,
                     "，仅裁切" if fit.weights.get(d, 1.0) < 1 else ""))
    if fit.cells is not None:
        print(f"[diag] 自标定: 安装偏移 {fit.offset_deg:+.1f}° / 高度 "
              f"{fit.cam_height_cm:.0f}cm；刚体残差 "
              f"{fit.rms_clean_cm if fit.rms_clean_cm is not None else fit.rms_all_cm:.1f}cm")
    return fit


def main(argv=None):
    ap = argparse.ArgumentParser(description="数字宫格布局扫·真实照片重放")
    ap.add_argument("--fixtures", default=None,
                    help="照片目录（缺省依次找 tests/fixtures/field_photos、"
                         "archive/result/field_photos）")
    ap.add_argument("--map", default="schedule",
                    choices=("schedule", "head-major", "nav-only", "down-only"),
                    help="照片→(俯仰,头部) 映射（缺省=扫描时序假设）")
    ap.add_argument("--diag", action="store_true", help="逐帧/逐数字诊断模式")
    ap.add_argument("--expect", action="store_true",
                    help="与照片读图真值对照，不符则非零退出")
    args = ap.parse_args(argv)

    photos = find_photos(args.fixtures)
    if not photos:
        print("[replay] 未找到现场照片（--fixtures 指定目录），跳过")
        return 0
    schedule = build_schedule(photos, args.map)
    print(f"[replay] 照片 {len(photos)} 张，映射 {args.map}，"
          f"实际用 {len(schedule)} 帧")
    for pitch, head, path in schedule:
        print(f"         pitch={pitch} head={head:4d}  {path.name}")

    robot = ReplayRobot(schedule)
    level = NineGridLevel(robot)

    if args.diag:
        run_diag(robot, level)
        return 0

    t0 = time.time()
    try:
        level.layout_scan()
    except RuntimeError as e:
        print(f"[replay] 布局扫失败: {e}")
        return 1
    dt = time.time() - t0
    print(f"[replay] 布局: {level.digit_cell}")
    print(f"[replay] 自标定: 安装偏移 {level._pitch_offset_deg:+.1f}° / 高度 "
          f"{level._cam_height_cm:.0f}cm")
    print(f"[replay] 位姿自举: ({level.pose[0]:.1f},{level.pose[1]:.1f}) "
          f"航向 {level.pose[2] * 180.0 / 3.141592653589793:.1f}°")
    print(f"[replay] 拍照 {robot.captures} 张（复用 {robot.reused} 次）"
          f"耗时 {dt:.1f}s")
    if 6 in level.digit_cell.values():
        print("[replay] 注意：结果里位置6 被占——该机位不在入口侧（或标号约定"
              "不同）时属预期；真机入口侧扫描必须满足格6 恒空")

    if args.expect:
        ok = level.digit_cell == PHOTO_TRUTH
        print(f"[replay] 与照片读图对照: {'一致 ✓' if ok else '不一致 ✗'}"
              f"（期望 {PHOTO_TRUTH}）")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
