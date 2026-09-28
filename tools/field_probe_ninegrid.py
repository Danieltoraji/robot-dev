#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""field_probe_ninegrid.py —— 九宫格现场探针（在**机器人上**运行）

一次跑完现场验收的第一步：
  1. 锁相机白平衡/对焦（core.robot_core.lock_camera_controls，带读回复验）；
  2. 按"俯仰 × 头部"档位采若干帧存盘（默认存 field_probe/）；
  3. 逐帧跑**真实**检测器，打印：白点/增益/过曝比例、检出的颜色、
     各面板的数字证据、歧义面板——即"锁定后光照还漂不漂、颜色还准不准"；
  4. 写 probe_summary.json（帧文件 ↔ 白点/增益/检出），供 PC 侧复标调色板
     与复核；帧本身用 tools/pull_from_robot.py 拉回 PC。

关键约定（复标别踩）：**存盘的是原图，不做光照归一化**。运行时是
「缩到 work_width → normalize_illumination → HSV」，PC 侧的
tools/calibrate_palette.py 已逐字复刻这三步；拉原图才能让 PC 在与运行时
**完全相同的空间**里采样。若在这里先归一化再存盘，等于把空间烘死，
PC 侧再归一化一次就是二次处理（历史事故：漏了这层对齐，采样落在原始空间，
probeC 粉 H=141 而非归一化后的 166，拟合出的窗口与运行时不匹配）。

用法（机器人仓库根目录）：
    /home/pi/jupyter-env/bin/python3 tools/field_probe_ninegrid.py
    /home/pi/jupyter-env/bin/python3 tools/field_probe_ninegrid.py \
        --frames 12 --tag 20260928-新场地     # 复标：按场次单独成目录
    /home/pi/jupyter-env/bin/python3 tools/field_probe_ninegrid.py \
        --frames 12 --out field_probe --no-lock
"""

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from core.camera_config import (HEAD_CENTER, HEAD_LEFT, HEAD_RIGHT,
                                HEAD_WIDE_LEFT, HEAD_WIDE_RIGHT)
from core.robot_core import (CAM_AUTO_EXPOSURE_ENABLED, RobotState,
                             auto_calibrate_exposure, lock_camera_controls)
from vision.nine_grid_detector import (NineGridDetector, COLOR_TO_ID,
                                       normalize_illumination)

try:
    from levels.nine_grid_shared import PITCH_NAV, PITCH_DOWN
except ImportError:      # 关卡层改动时不要阻断探针
    PITCH_NAV, PITCH_DOWN = 1200, 1040


def main(argv=None):
    ap = argparse.ArgumentParser(description="九宫格现场探针")
    ap.add_argument("--frames", type=int, default=9, help="目标帧数（默认 9）")
    ap.add_argument("--out", default="field_probe", help="存帧目录")
    ap.add_argument("--tag", default=None,
                    help="本次场次名：帧与摘要写入 <out>/<tag>/，复标时按场次拉回、"
                         "互不混样（缺省沿用 <out>/ 平铺，与旧流程兼容）")
    ap.add_argument("--no-lock", action="store_true", help="跳过相机锁定")
    ap.add_argument("--no-auto-exposure", action="store_true",
                    help="跳过开局自动曝光闭环（默认与真实关卡一致：先跑 "
                         "auto_calibrate_exposure 把画面均亮调到目标值）")
    ap.add_argument("--unlock-first", action="store_true",
                    help="先恢复自动白平衡再锁（对照：验证锁定是否真的生效）")
    args = ap.parse_args(argv)

    out_dir = args.out if not args.tag else os.path.join(args.out, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    ok_lock, lock_info = (False, {"reason": "已跳过"})
    if not args.no_lock:
        if args.unlock_first:
            import subprocess
            dev = "/dev/video0"
            subprocess.run(f"v4l2-ctl -d {dev} -c white_balance_automatic=1",
                           shell=True, capture_output=True, text=True)
            print("[探针] 已先恢复自动白平衡（对照用）")
        ok_lock, lock_info = lock_camera_controls(force=True)
    print(f"[探针] 相机锁定: {'成功' if ok_lock else '未生效'} "
          f"({lock_info.get('reason') or lock_info.get('ctrls')})")
    if lock_info.get("mismatch"):
        print(f"[探针] 读回不一致: {lock_info['mismatch']}")

    state = RobotState()
    # 与真实关卡同工况：关卡的九宫格流程在开局会跑一次自动曝光闭环
    # （levels/nine_grid_shared.py 里 CAM_AUTO_EXPOSURE_ENABLED 分支）。
    # 探针若跳过它，拍的帧就落在"锁定值"这一档，而复标出来的窗口要服务的是
    # "闭环后"那一档——曝光本可由 normalize_illumination 的曝光基准抵消，
    # 但噪声/过曝形态不同，没必要制造这层偏差。
    exp_info = None
    if CAM_AUTO_EXPOSURE_ENABLED and not args.no_auto_exposure:
        exp_info = auto_calibrate_exposure(state)
        print(f"[探针] 自动曝光: ok={exp_info['ok']} 曝光={exp_info['exposure']} "
              f"增益={exp_info['gain']} 均亮={exp_info['mean']} "
              f"过曝={exp_info['clip']}（{exp_info['shots']} 张）")
    det = NineGridDetector()
    heads = [HEAD_CENTER, HEAD_LEFT, HEAD_RIGHT, HEAD_WIDE_LEFT, HEAD_WIDE_RIGHT]
    pitches = [PITCH_NAV, PITCH_DOWN]
    rows = []
    n_seen = 0
    i = 0
    while n_seen < args.frames and i < len(pitches) * len(heads) * 2:
        pitch = pitches[i % len(pitches)]
        head = heads[(i // len(pitches)) % len(heads)]
        i += 1
        state.set_pitch(pitch)
        state.set_head(head)
        frame = state.capture_frame()
        if frame is None:
            print("[探针] 拍照失败，跳过一帧")
            continue
        _, info = normalize_illumination(frame)      # 原生分辨率上的诊断估计
        obs = det.detect_panels(frame, shape=True)
        dnorm = dict(det.last_norm or {})            # 检测器真正用的那一次（工作分辨率）
        colors = sorted(o.color for o in obs)
        name = f"field_{int(time.time())}_{pitch}_{head}.jpg"
        # 存**原图**：PC 侧 calibrate_palette.py 自己复刻"缩到工作分辨率 →
        # normalize_illumination"，两边空间一致（见文件头说明）
        path = os.path.join(out_dir, name)
        try:
            import cv2
            cv2.imwrite(path, frame)
        except Exception as e:                      # noqa: BLE001
            print(f"[探针] 存帧失败({e})，仍继续")
            path = ""
        ev = {o.color: (None if o.digit_evidence is None
                        else round(o.digit_evidence, 4)) for o in obs}
        amb = sorted(o.color for o in obs if o.ambiguous)
        rows.append({
            "file": name, "pitch": pitch, "head": head,
            "white_bgr": info.get("white_bgr"),
            "gain": [round(g, 3) for g in info.get("gain", [])],
            "white_method": info.get("white_method"),
            "clip_frac": round(float(info.get("clip_frac", 0.0)), 4),
            # 检测器实际用的归一化（工作分辨率）：applied=False 表示这帧
            # **没归一化**，其颜色样本落在原始空间、与调色板不同空间——拟合前必看
            "det_norm": {"applied": bool(dnorm.get("applied")),
                         "white_method": dnorm.get("white_method"),
                         "gain": [round(g, 3) for g in dnorm.get("gain", [])],
                         "suspicious": bool(dnorm.get("suspicious")),
                         "clip_frac": round(float(dnorm.get("clip_frac", 0.0)), 4)},
            "colors": colors, "n_colors": len(colors),
            "digit_evidence": ev, "ambiguous": amb,
            "shape_decided": {o.color: o.shape_digit for o in obs
                              if o.shape_digit is not None},
        })
        wp = info.get("white_bgr")
        print(f"  帧{i}: pitch={pitch} head={head} 检出 {len(colors)}/7 "
              f"{colors}  白点={None if wp is None else [round(v) for v in wp]} "
              f"增益={[round(g, 3) for g in info.get('gain', [])]} "
              f"过曝={info.get('clip_frac', 0.0):.1%} 歧义={amb}")
        n_seen += 1

    good = [r for r in rows if r["n_colors"] == 7]
    print(f"\n[探针] 共 {len(rows)} 帧，7/7 帧 {len(good)} 个；"
          f"逐色检出 "
          + ", ".join(f"{c}:{sum(1 for r in rows if c in r['colors'])}"
                      for c in COLOR_TO_ID))
    if rows:
        clips = [r["clip_frac"] for r in rows]
        print(f"[探针] 过曝比例 最大 {max(clips):.1%}"
              + ("（>5% 建议锁曝光：core.robot_core.CAM_LOCK_EXPOSURE=True）"
                 if max(clips) > 0.05 else "（可接受）"))
        wps = [r["white_bgr"] for r in rows if r["white_bgr"]]
        if len(wps) > 1:
            spread = max(abs(a - b) for a, b in zip(wps[0], wps[-1]))
            print(f"[探针] 白点跨帧最大漂移 {spread:.0f}（0-255）"
                  + "——锁定生效时应接近 0" if spread <= 6 else
                  f"[探针] 白点跨帧最大漂移 {spread:.0f}——仍偏大，检查锁定")
        # 复标前置检查：没归一化的帧，其颜色样本与调色板不在同一空间
        not_norm = [r["file"] for r in rows
                    if not (r.get("det_norm") or {}).get("applied")]
        if not_norm:
            print(f"[探针] ⚠ {len(not_norm)}/{len(rows)} 帧检测器**未做光照归一化**"
                  f"（白点估不出）：{not_norm[:3]}——这些帧的颜色样本不在调色板空间，"
                  f"复标前先看 probe_summary.json 的 det_norm")
        susp = [r["file"] for r in rows
                if (r.get("det_norm") or {}).get("suspicious")]
        if susp:
            print(f"[探针] ⚠ {len(susp)} 帧归一化增益超出告警区间"
                  f"（光照偏离参考白点），样本噪声偏大：{susp[:3]}")
    with open(os.path.join(out_dir, "probe_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump({"tag": args.tag,
                   "lock": {"ok": ok_lock,
                            "ctrls": lock_info.get("ctrls"),
                            "readback": lock_info.get("readback"),
                            "mismatch": lock_info.get("mismatch"),
                            "reason": lock_info.get("reason")},
                   "auto_exposure": exp_info,
                   "frames": rows}, f, ensure_ascii=False, indent=2)
    print(f"[探针] 摘要已写 {os.path.join(out_dir, 'probe_summary.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
