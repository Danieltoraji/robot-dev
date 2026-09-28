# -*- coding: utf-8 -*-
"""红色目标静态测距验证（tools/verify_red_distance.py）——M3 验收工具

作用
----
在真机上验证"本地系单应 + 红色带检测"的测距精度与重复精度，产出：
- 各距离点的测距误差（对照卷尺量出的真值）
- 同点位多次测量的散布 σ（喂给关卡触发窗口宽度公式）
- 方位角/横偏读数（供对正容差核对）

M3 验收线（方案 §5）：d≤50cm 误差 ≤2cm、重复 σ ≤1cm。

用法
----
真机交互（把红色目标摆在已知距离，输入真值回车即测）：
    python tools/verify_red_distance.py --pitch 1000
单帧离线分析（已知距离的已有照片）：
    python tools/verify_red_distance.py --image photo.jpg --dist 32.0

交互命令：输入数字=该真值距离下测一轮；s=统计汇总；q=退出。
注意：测量协议须与运行时一致——stand 姿态、头部 1500、pitch=观测档、
拍照前静置（工具内固定 0.3s）；机器人每次挪位后输入下一组真值。
"""

import argparse
import os
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from core.camera_config import HEAD_CENTER
from core.ground_line_meter import build_meter, CALIB_PATH
from core.robot_core import RobotState
from vision.red_line_detector import RedLineDetector

SETTLE_S = 0.3  # 与关卡运行时协议一致


def parse_args():
    ap = argparse.ArgumentParser(description="红色目标静态测距验证")
    ap.add_argument("--pitch", type=int, default=1000)
    ap.add_argument("--cam-z", type=float, default=39.0,
                    help="站立实测相机高度 cm（与标定时一致）")
    ap.add_argument("--frames", type=int, default=3,
                    help="每轮测量帧数（取中位数），默认 3")
    ap.add_argument("--image", default=None, help="离线单帧分析模式")
    ap.add_argument("--dist", type=float, default=None,
                    help="离线模式的真值距离 cm")
    return ap.parse_args()


def measure_once(detector, meter, state):
    frame = state.capture_frame()
    if frame is None:
        return None
    det = detector.detect(frame)
    m = meter.measure(det.components)
    return m


def measure_round(detector, meter, state, frames):
    """一轮 = frames 帧，返回有效测量的 (forward, bearing, lateral) 中位数"""
    reads = []
    for _ in range(frames):
        m = measure_once(detector, meter, state)
        if m is not None and m.exists:
            reads.append((m.forward_cm, m.bearing_err_deg, m.lateral_cm))
        else:
            print("  本帧未检出（拍照失败或无红目标）")
        time.sleep(SETTLE_S)
    if not reads:
        return None
    arr = np.asarray(reads)
    return tuple(float(np.median(arr[:, i])) for i in range(3))


def main():
    args = parse_args()
    detector = RedLineDetector()
    meter = build_meter(args.pitch, args.cam_z)
    if meter.degraded:
        print(f"⚠ 未找到标定 {CALIB_PATH}，使用 from_pose 自举（±3cm 级）——"
              "请先运行 tools/calib_stairs_hurdle.py")
    state = None
    if args.image:
        import cv2
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"读图失败: {args.image}")
            sys.exit(1)
        assert args.dist is not None, "离线模式需 --dist 真值"
        det = detector.detect(frame)
        m = meter.measure(det.components)
        if not m.exists:
            print("未检出红色目标")
            sys.exit(1)
        print(f"真值 {args.dist:.1f}cm -> 测量 {m.forward_cm:.2f}cm "
              f"(误差 {m.forward_cm - args.dist:+.2f})  方位 "
              f"{m.bearing_err_deg:+.2f}°  横偏 {m.lateral_cm:+.2f}cm")
        return

    state = RobotState()
    state.set_head(HEAD_CENTER)
    state.set_pitch(args.pitch)

    # records: {真值距离: [(fwd,bearing,lat), ...]}
    records = {}
    print("\n输入真值距离(cm)测一轮；s=汇总；q=退出。"
          "同一距离可测多次（评估重复精度 σ）。")
    while True:
        s = input("\n真值距离(cm)/s/q: ").strip().lower()
        if s == "q":
            break
        if s == "s":
            print("\n===== 汇总 =====")
            for d, rs in sorted(records.items()):
                fwd = np.asarray([r[0] for r in rs])
                err = fwd - d
                sigma = float(np.std(fwd)) if len(fwd) > 1 else float("nan")
                print(f"  真值 {d:6.1f}cm  n={len(rs)}  "
                      f"误差均值 {err.mean():+.2f}cm  重复σ {sigma:.2f}cm")
                if d <= 50.0:
                    if abs(err.mean()) > 2.0:
                        print("    ⚠ 误差超 2cm（M3 验收线），建议重标")
                    if len(rs) > 1 and sigma > 1.0:
                        print("    ⚠ 重复 σ 超 1cm（站立晃动/协议不一致？）")
            print("================")
            continue
        try:
            d = float(s)
        except ValueError:
            print("请输入数字 / s / q")
            continue
        r = measure_round(detector, meter, state, args.frames)
        if r is None:
            print("本轮全部帧未检出")
            continue
        fwd, bearing, lat = r
        records.setdefault(d, []).append((fwd, bearing, lat))
        print(f"真值 {d:.1f}cm -> 测量 {fwd:.2f}cm (误差 {fwd - d:+.2f})  "
              f"方位 {bearing:+.2f}°  横偏 {lat:+.2f}cm")


if __name__ == "__main__":
    main()
