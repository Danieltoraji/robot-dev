#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_layout_scan.py —— 只跑布局扫描（不动底盘前进以外无动作）并给出结论

为什么单独有这个工具：整局 `run_level` 会一路驱动机器人；调试"布局到底扫对
没有"时只想要第一阶段，且要一份**可存档、可对比**的结构化结论（数字→格、
自标定常数、位姿、告警）。

注意：布局扫描在"多解/缺数字/自举不过"时会**前进一步（2cm）重扫**，最多 3 轮
（见 levels/nine_grid.layout_scan）——这是设计行为，不是失控。

用法（机器人仓库根目录）：
    /home/pi/jupyter-env/bin/python3 tools/run_layout_scan.py
    /home/pi/jupyter-env/bin/python3 tools/run_layout_scan.py --no-lock --out /tmp/scan.json
退出码：0 成功 / 2 三轮未定 / 3 其它异常
"""

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from core.robot_core import (RobotState, lock_camera_controls,
                                auto_calibrate_exposure,
                                CAM_AUTO_EXPOSURE_ENABLED)
from levels.nine_grid_shared import NineGridShared, PITCH_NAV, print_layout


def main(argv=None):
    ap = argparse.ArgumentParser(description="九宫格布局扫描（单阶段）")
    ap.add_argument("--no-lock", action="store_true",
                    help="跳过相机锁定与自动曝光标定")
    ap.add_argument("--budget", type=float, default=600.0, help="本阶段时间预算(s)")
    ap.add_argument("--out", default="field_layout_scan.json")
    args = ap.parse_args(argv)

    lock_ok, lock_info = (False, {"reason": "已跳过"})
    if not args.no_lock:
        lock_ok, lock_info = lock_camera_controls(force=True)
        print(f"[扫描] 相机锁定 {'成功' if lock_ok else '未生效'}: "
              f"{lock_info.get('reason') or lock_info.get('ctrls')}")
        if lock_info.get("mismatch"):
            print(f"[扫描] 读回不一致: {lock_info['mismatch']}")

    state = RobotState()
    if CAM_AUTO_EXPOSURE_ENABLED and not args.no_lock:
        calib = auto_calibrate_exposure(state)
        lock_info["auto_exposure"] = {k: calib[k] for k in
                                      ("ok", "exposure", "gain", "mean", "clip")}
    level = NineGridShared(state)
    level.deadline = time.time() + args.budget
    # 拍照计数 + **逐帧光照记录**：layout_scan 不走 _count_frame，故在
    # capture_frame 上挂钩子。光照记录是"换灯后还能不能认色"的直接证据
    # （白点/增益/过曝比例），出问题时可据此判断是光源变了还是算法退化了。
    shots = {"n": 0}
    norms = []

    def _info_of(frame):
        try:
            from vision.nine_grid_detector import normalize_illumination
            _out, info = normalize_illumination(frame)
            return {"white_bgr": None if info.get("white_bgr") is None
                    else [round(float(v)) for v in info["white_bgr"]],
                    "gain": [round(float(g), 3) for g in info.get("gain", [])],
                    "method": info.get("white_method"),
                    "clip_frac": round(float(info.get("clip_frac", 0.0)), 4)}
        except Exception as e:                  # noqa: BLE001
            return {"error": f"{type(e).__name__}: {e}"}

    _orig_capture = state.capture_frame

    def _counting_capture(*a, **kw):
        shots["n"] += 1
        frame = _orig_capture(*a, **kw)
        if frame is not None:
            norms.append(_info_of(frame))
        return frame

    state.capture_frame = _counting_capture
    t0, err, rc = time.time(), None, 0
    try:
        level.layout_scan()
    except RuntimeError as e:
        err, rc = str(e), 2
    except Exception as e:                      # noqa: BLE001
        err, rc = f"{type(e).__name__}: {e}", 3
    dt = time.time() - t0

    cell_digit = {c: d for d, c in (level.digit_cell or {}).items()}
    out = {
        "ok": rc == 0,
        "error": err,
        "digit_cell": {int(k): int(v) for k, v in (level.digit_cell or {}).items()},
        "cell_digit": {int(k): int(v) for k, v in cell_digit.items()},
        "pose": None if level.pose is None else [round(float(v), 2) for v in level.pose],
        "pose_deg": (None if level.pose is None
                     else round(float(level.pose[2]) * 57.29578, 1)),
        "pitch_offset_deg": round(float(level._pitch_offset_deg), 1),
        "cam_height_cm": round(float(level._cam_height_cm), 1),
        # 相机高度来源（2026-09-27）："像素域精定" = 自标定过了像素门且落在
        # 实测值附近；"实测回退" = 用卷尺实测常数（见核对报告-2026-09-27）。
        # 现场判读：出现"实测回退"不是失败，但说明这局的**距离尺度没被自标定
        # 验证过**，量距离类结论要打问号（导航/到达判决不读它）。
        "height_source": (None if level._last_fit is None
                          else level._last_fit.height_source),
        "calib_ok": (None if level._last_fit is None
                     else bool(level._last_fit.calib_ok)),
        "effective_pitch_deg_nav": round(float(level._effective_pitch_deg(PITCH_NAV)), 1),
        "cell_conflict": sorted(int(v) for v in (level.cell_conflict or ())),
        "frames": dict(level.phase_frames or {}),
        "photos": int(shots["n"]),
        "lighting": norms,
        "elapsed_s": round(dt, 1),
        "lock": {"ok": lock_ok, "ctrls": lock_info.get("ctrls"),
                 "mismatch": lock_info.get("mismatch"),
                 "reason": lock_info.get("reason")},
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\n================ 布局扫描结论 ================")
    if rc == 0:
        print(f"数字→格: {out['digit_cell']}")
        print("数字→格位（远排在上一行、0 = 该格没有数字）：")
        print_layout(out["digit_cell"], prefix="  ")
        print(f"格→数字: {out['cell_digit']}   格6是否被占: "
              f"{'是（与规则冲突，需核对标号约定）' if 6 in cell_digit else '否'}")
        print(f"位姿: ({out['pose'][0]}, {out['pose'][1]}) "
              f"航向 {out['pose_deg']}°" if out["pose"] else "位姿: 无")
        print(f"自标定: 安装偏移 {out['pitch_offset_deg']:+.1f}° / 高度 "
              f"{out['cam_height_cm']:.0f}cm（导航档有效俯角 "
              f"{out['effective_pitch_deg_nav']:.1f}°）"
              f"｜高度来源: {out['height_source'] or '无（未跑布局扫）'}"
              f"{'（自标定通过像素门）' if out['calib_ok'] else '（用卷尺实测值）'}")
        print(f"仲裁冲突格: {out['cell_conflict']}")
    else:
        print(f"失败（rc={rc}）: {err}")
    print(f"拍照 {out['photos']} 张，耗时 {dt:.0f}s"
          f"（本工具自定的预算 {args.budget:.0f}s；关卡侧已无时间闸）")
    if norms:
        wps = [n["white_bgr"] for n in norms if n.get("white_bgr")]
        clips = [n.get("clip_frac", 0.0) for n in norms]
        gains = [g for n in norms for g in n.get("gain", [])]
        meth = sorted({n.get("method") for n in norms if n.get("method")})
        if wps:
            print(f"光照: 白点 {wps[0]} → {wps[-1]}（{len(wps)} 帧都有白点，"
                  f"判据 {meth}）")
        print(f"      过曝 最大 {max(clips):.1%}"
              + "（>5% 建议重扫相机参数：tools/field_camera_sweep.py）"
              if max(clips) > 0.05 else f"      过曝 最大 {max(clips):.1%}（可接受）")
        if gains:
            print(f"      归一化增益范围 {min(gains):.3f}~{max(gains):.3f}"
                  "（1.0 附近=光源本来就近中性）")
    print(f"结论已写 {args.out}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
