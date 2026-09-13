#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""field_camera_sweep.py —— 相机参数现场扫描（在**机器人上**运行）

换场地/换灯光后，`core.robot_core.CAM_*` 里那组常数必须重新扫：
  1. **曝光**：逐个候选曝光值拍照，看"过曝比例"与"七色检出数"——
     过曝是不可逆的信息丢失，橙/黄最容易被洗白（实测该场地自动曝光
     过曝 21~54% ⇒ 橙 0/12、黄 0/12 帧检出）；
  2. **白平衡温度**：逐个候选色温拍照，看白板是否中性（白点 R/B 越接近 1
     越准，判据是否回到"低饱和"）。色温不对时归一化要硬扛一个大偏色，
     噪声被放大、调色板窗口被拖偏；
  3. **重复性**：同一机位连拍 N 张，均亮/白点应完全一致（判"锁定是否生效"）。

用法（机器人仓库根目录）：
    /home/pi/jupyter-env/bin/python3 tools/field_camera_sweep.py --what all
    /home/pi/jupyter-env/bin/python3 tools/field_camera_sweep.py --what wb \
        --temps 4000 5200 5800 6500 --exposure 100
输出末尾会给出"建议常数"，把它抄进 core/robot_core.py（或在现场用
`--write-env` 打印可直接粘贴的环境变量形式）。
"""

import argparse
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from core.camera_config import CAMERA_WIDTH, CAMERA_HEIGHT
from core.robot_core import CAM_V4L2_DEVICE
from vision.nine_grid_detector import (NineGridDetector, estimate_white_bgr,
                                       normalize_illumination)


def v4l(*args, device=CAM_V4L2_DEVICE):
    return subprocess.run(f"v4l2-ctl -d {device} " + " ".join(args),
                          shell=True, capture_output=True, text=True)


def shot(skip=10, path="/tmp/sweep.jpg"):
    import cv2
    subprocess.run(f"fswebcam -r {CAMERA_WIDTH}x{CAMERA_HEIGHT} --no-banner "
                   f"-S {skip} {path}", shell=True, capture_output=True,
                   text=True)
    return cv2.imread(path)


def measure(frame, det=None):
    """→ dict(mean, clip, white_bgr, white_method, rb, n_colors, colors)"""
    det = det or NineGridDetector()
    _, info = normalize_illumination(frame)
    wp, _frac, method = estimate_white_bgr(frame)
    obs = det.detect_panels(frame, shape=True)
    rb = None if wp is None else float(wp[2]) / max(1.0, float(wp[0]))
    return {"mean": float(frame.mean()),
            "clip": float(info.get("clip_frac", 0.0)),
            "white_bgr": None if wp is None else [round(float(v)) for v in wp],
            "white_method": method, "rb": None if rb is None else round(rb, 2),
            "n_colors": len(obs), "colors": sorted(o.color for o in obs)}


def sweep_exposure(exps, det, skip):
    print("== 曝光扫描（固定当前白平衡）==")
    rows = []
    for e in exps:
        v4l("-c auto_exposure=1", f"-c exposure_time_absolute={int(e)}")
        fr = shot(skip)
        if fr is None:
            print(f"  曝光 {e}: 拍照失败")
            continue
        m = measure(fr, det)
        rows.append((e, m))
        print(f"  曝光 {e:4d}: 均亮 {m['mean']:6.1f} 过曝 {m['clip']:5.1%} "
              f"白点 {m['white_bgr']}({m['white_method']}) 检出 {m['n_colors']}/7 "
              f"{m['colors']}")
    ok = [r for r in rows if r[1]["clip"] <= 0.05]
    best = max(ok or rows, key=lambda r: r[1]["n_colors"]) if rows else None
    if best:
        print(f"  ⇒ 建议曝光 exposure_time_absolute={best[0]}"
              f"（过曝 {best[1]['clip']:.1%}、检出 {best[1]['n_colors']}/7）")
    return best[0] if best else None


def sweep_wb(temps, exposure, det, skip):
    print("== 白平衡温度扫描 ==")
    if exposure:
        v4l("-c auto_exposure=1", f"-c exposure_time_absolute={int(exposure)}")
    rows = []
    for t in temps:
        v4l("-c white_balance_automatic=0", f"-c white_balance_temperature={int(t)}")
        fr = shot(skip)
        if fr is None:
            print(f"  {t}K: 拍照失败")
            continue
        m = measure(fr, det)
        rows.append((t, m))
        print(f"  {t:4d}K: 白点 {m['white_bgr']}({m['white_method']}) "
              f"R/B {m['rb']} 过曝 {m['clip']:5.1%} 检出 {m['n_colors']}/7 "
              f"{m['colors']}")
    cand = [r for r in rows if r[1]["rb"] is not None
            and r[1]["white_method"] == "neutral"]
    if not cand:
        cand = [r for r in rows if r[1]["rb"] is not None]
    best = min(cand, key=lambda r: abs(r[1]["rb"] - 1.0)) if cand else None
    if best:
        print(f"  ⇒ 建议色温 white_balance_temperature={best[0]}"
              f"（R/B {best[1]['rb']}、白点判据 {best[1]['white_method']}）")
    return best[0] if best else None


def check_repeat(n, det, skip):
    print(f"== 重复性检查（同机位连拍 {n} 张）==")
    vals = []
    for i in range(n):
        fr = shot(skip)
        if fr is None:
            continue
        m = measure(fr, det)
        vals.append(m)
        print(f"  #{i}: 均亮 {m['mean']:6.1f} 白点 {m['white_bgr']} "
              f"检出 {m['n_colors']}/7")
    if len(vals) > 1:
        d = max(v["mean"] for v in vals) - min(v["mean"] for v in vals)
        print(f"  ⇒ 均亮极差 {d:.2f}（锁定生效时应 <1）")
    return vals


def main(argv=None):
    ap = argparse.ArgumentParser(description="相机参数现场扫描")
    ap.add_argument("--what", default="all",
                    choices=["all", "exposure", "wb", "repeat"])
    ap.add_argument("--exposures", nargs="*", type=int,
                    default=[60, 100, 130, 160, 200])
    ap.add_argument("--temps", nargs="*", type=int,
                    default=[4200, 4800, 5200, 5600, 5800, 6200])
    ap.add_argument("--exposure", type=int, default=None,
                    help="扫白平衡时固定的曝光（缺省用当前值）")
    ap.add_argument("--skip", type=int, default=10, help="fswebcam -S 跳帧数")
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args(argv)

    det = NineGridDetector()
    exp = args.exposure
    if args.what in ("all", "exposure"):
        exp = sweep_exposure(args.exposures, det, args.skip) or exp
    if args.what in ("all", "wb"):
        sweep_wb(args.temps, exp, det, args.skip)
    if args.what in ("all", "repeat"):
        check_repeat(args.repeat, det, args.skip)
    print("\n把建议值写进 core/robot_core.py 的 CAM_WB_TEMPERATURE_K / "
          "CAM_EXPOSURE_ABS，并置 CAM_LOCK_EXPOSURE=True；"
          "然后跑 tools/field_probe_ninegrid.py 复验七色检出。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
