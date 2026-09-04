#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_multi_view.py —— 阶段2：真机多帧定位数据采集（A 方案）

流程：
    1. 发现阶段：回正 / 右转(-40.5°) / 左转(+40.5°) 各拍 1 帧，检测标签；
    2. 选目标标签：优先回正帧、其次右转帧、最后左转帧（选像素最靠近画面中心者）；
    3. 估算目标方位角 θ₀（头转角 + 像素横向偏移换算）；
    4. 精拍阶段：以 θ₀ 为中心摆动 ±--sweep（默认 15°）拍 3 帧
       （超出舵机实测行程 ±90° 时整体平移回界内）；
    5. 全部帧的角点观测落盘 archive/result/multiview_<时间戳>.npz，
       供 tools/field_calib/optimize_multi_view.py 离线联合求解。

用法（机器人项目根目录）：
    python -m tools.field_calib.collect_multi_view
    python -m tools.field_calib.collect_multi_view --sweep 9
    python -m tools.field_calib.collect_multi_view --scan-left 2250 --scan-right 750   # 发现档位可调

发现阶段档位：
    默认右转 600 / 左转 2400（±81°，舵机实测行程 ±90° 内留 9° 余量）；
    可分别用 --scan-right / --scan-left 调整（脉宽 500~2500 范围内）。

输出：
    archive/result/multiview_<时间戳>.npz：
      pulses      (N,)      每帧头部脉宽
      thetas_nom  (N,)      每帧标称转角（度）
      frame_idx   (K,)      每条角点记录所属帧
      tag_ids     (K,)      每条角点记录所属标签 id
      corners     (K,4,2)   角点像素（已按 TAG_CORNER_PERM 与 tag_poses 对齐）
      meta        (1,)      采集元信息（JSON 字符串）
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

# 允许直接运行本文件
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.paths import RESULT_DIR
from core.robot_core import RobotState, HEAD_CENTER, HEAD_RIGHT, HEAD_LEFT, TAG_CORNER_PERM
from levels.goodluck import tag_poses

FX = 1.944903664123011e03   # 相机内参 fx（像素→角度换算）
CX = 1.283069051100245e03   # 主点 u
SERVO_DEG_PER_US = 0.09
PULSE_MAX, PULSE_MIN = 2500, 500   # 舵机实测行程：控制值 500~2500 = 回正向左右各 90°
DEFAULT_SCAN_RIGHT = 600           # 发现阶段默认右转脉宽（-81°）
DEFAULT_SCAN_LEFT = 2400           # 发现阶段默认左转脉宽（+81°）


def detect_known(state, filename):
    """拍照文件 → 已知标签 {id: corners(4,2 已按 TAG_CORNER_PERM 重排)}"""
    dets = state.detect_apriltag(filename)
    out = {}
    for d in dets:
        tid = str(d.tag_id)
        if tid not in tag_poses:
            print(f"  [WARN] 跳过未知标签 {tid}")
            continue
        out[tid] = np.asarray(d.corners, dtype=np.float64)[TAG_CORNER_PERM]
    return out


def theta_of_pulse(pulse):
    return (pulse - 1500) * SERVO_DEG_PER_US


def pulse_for_theta(theta_deg):
    return int(round(1500 + theta_deg / SERVO_DEG_PER_US))


def estimate_target_theta(detect_theta_deg, center_px):
    """发现帧头转角 + 标签中心像素 → 正对标签所需头转角 θ₀（度）

    画面右侧(u>主点) = 世界方位角减小：θ₀ = θ_detect − atan((u−cx)/fx)
    """
    offset = np.degrees(np.arctan2(center_px[0] - CX, FX))
    return detect_theta_deg - offset


def pulse_name(pulse):
    if pulse == HEAD_CENTER:
        return "回正"
    if pulse < HEAD_CENTER:
        return f"右转({pulse})"
    return f"左转({pulse})"


def main():
    parser = argparse.ArgumentParser(description="真机多帧定位数据采集（阶段2）")
    parser.add_argument("--sweep", type=float, default=15.0, help="精拍摆动半幅（度，默认 15）")
    parser.add_argument("--scan-right", type=int, default=DEFAULT_SCAN_RIGHT,
                        help=f"发现阶段右转脉宽（默认 {DEFAULT_SCAN_RIGHT}，-81°）")
    parser.add_argument("--scan-left", type=int, default=DEFAULT_SCAN_LEFT,
                        help=f"发现阶段左转脉宽（默认 {DEFAULT_SCAN_LEFT}，+81°）")
    args = parser.parse_args()

    if not (PULSE_MIN <= args.scan_right <= PULSE_MAX and
            PULSE_MIN <= args.scan_left <= PULSE_MAX and
            args.scan_right < HEAD_CENTER < args.scan_left):
        print(f"错误: 发现档位越界或方向错误（右转须 <1500 < 左转，范围 {PULSE_MIN}~{PULSE_MAX}）")
        return

    scan = [("回正", HEAD_CENTER), ("右转", args.scan_right), ("左转", args.scan_left)]

    os.makedirs(RESULT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    npz_path = os.path.join(RESULT_DIR, f"multiview_{ts}.npz")

    state = RobotState(tag_poses=tag_poses)
    state.set_head(HEAD_CENTER)

    print("=" * 70)
    print("多帧定位数据采集（阶段2）")
    print(f"发现阶段: 回正/右转({args.scan_right},{theta_of_pulse(args.scan_right):+.1f}°)"
          f"/左转({args.scan_left},{theta_of_pulse(args.scan_left):+.1f}°) 各 1 帧")
    print(f"精拍阶段: 以目标方位角为中心 ±{args.sweep:g}° 摆动 3 帧")
    print(f"输出: {npz_path}")
    print("提醒: 采集全程机体保持不动，只动头。")
    print("=" * 70)

    # ---- 发现阶段 ----
    frames = []   # (pulse, theta_nom, {tag_id: corners})
    for name, pulse in scan:
        state.set_head(pulse)
        filename = state.capture_image()
        if filename is None:
            print(f"[发现 {name}] 拍照失败，跳过")
            continue
        tags = detect_known(state, filename)
        theta = theta_of_pulse(pulse)
        frames.append((pulse, theta, tags))
        desc = ", ".join(f"tag{tid}@{np.mean(c, axis=0).round(0).astype(int)}"
                         for tid, c in tags.items())
        print(f"[发现 {name}] 脉宽 {pulse} (θ={theta:+.1f}°) 标签: {desc or '无'}")

    # ---- 选目标标签（最靠近画面中心者） ----
    target = None
    for pulse, theta, tags in frames:
        if tags:
            tid = min(tags, key=lambda t: np.sum((np.mean(tags[t], axis=0) - [CX, 972.0]) ** 2))
            target = (pulse, theta, tid, tags[tid])
            break
    if target is None:
        print("\n三个发现帧均未检测到已知标签。请换位置/方向后重跑。")
        return

    t_pulse, t_theta, t_tid, t_corners = target
    t_center = np.mean(t_corners, axis=0)
    theta0 = estimate_target_theta(t_theta, t_center)
    print(f"\n目标: tag{t_tid}（来自{pulse_name(t_pulse)}帧）中心像素 {t_center.round(1)}")
    print(f"正对标签所需头转角 θ₀ ≈ {theta0:+.1f}°")

    # ---- 精拍阶段（超行程则整体平移回界内） ----
    swings = [theta0 - args.sweep, theta0, theta0 + args.sweep]
    if max(swings) > 54.0:
        shift = 54.0 - max(swings)
        swings = [s + shift for s in swings]
    if min(swings) < -54.0:
        shift = -54.0 - min(swings)
        swings = [s + shift for s in swings]

    print(f"精拍摆动: {[f'{s:+.1f}°' for s in swings]}")
    for s in swings:
        pulse = pulse_for_theta(s)
        state.set_head(pulse)
        filename = state.capture_image()
        if filename is None:
            print(f"[精拍 θ={s:+.1f}°] 拍照失败，跳过")
            continue
        tags = detect_known(state, filename)
        frames.append((pulse, s, tags))
        desc = ", ".join(f"tag{tid}@{np.mean(c, axis=0).round(0).astype(int)}"
                         for tid, c in tags.items())
        print(f"[精拍 θ={s:+.1f}°] 脉宽 {pulse} 标签: {desc or '无'}")

    state.set_head(HEAD_CENTER)

    # ---- 落盘 ----
    pulses, thetas = [], []
    f_idx, t_ids, corners = [], [], []
    for fi, (pulse, theta, tags) in enumerate(frames):
        pulses.append(pulse)
        thetas.append(theta)
        for tid, c in tags.items():
            f_idx.append(fi)
            t_ids.append(int(tid))
            corners.append(c)
    if not t_ids:
        print("\n未采集到任何角点观测，未保存文件。请换位置重跑。")
        return
    meta = json.dumps({
        "ts": ts,
        "sweep": args.sweep,
        "scan_right": args.scan_right,
        "scan_left": args.scan_left,
        "n_frames": len(frames),
        "target_tag": t_tid,
        "theta0": float(theta0),
        "note": "corners 已按 TAG_CORNER_PERM 与 tag_poses 对齐",
    })
    np.savez(npz_path,
             pulses=np.array(pulses, dtype=np.int64),
             thetas_nom=np.array(thetas, dtype=np.float64),
             frame_idx=np.array(f_idx, dtype=np.int64),
             tag_ids=np.array(t_ids, dtype=np.int64),
             corners=np.array(corners, dtype=np.float64),
             meta=np.array([meta]))
    print("\n" + "=" * 70)
    print(f"采集完成: {len(frames)} 帧, {len(t_ids)} 条角点观测")
    print(f"已保存: {npz_path}")
    print("下一步: python -m tools.field_calib.optimize_multi_view --data " + npz_path)
    print("=" * 70)


if __name__ == "__main__":
    main()
