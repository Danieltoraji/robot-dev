# -*- coding: utf-8 -*-
"""红色目标静态测距验证（tools/verify_red_distance.py）——M3 验收工具

**这不是标定工具**，它不产生任何参数、不写任何文件，只把测距结果量给你看。
标定由 tools/calib_ruler_profile.py 做（拍卷尺点刻度），它标出相机几何。

作用
----
在真机上验证"本地系地面模型 + 红色带检测"的测距精度与重复精度，产出：
- 各距离点的测距误差（对照卷尺量出的真值）
- 同点位多次测量的散布 σ（喂给关卡触发窗口宽度公式）
- 方位角/横偏读数（供对正容差核对）

⚠ 距离口径（最容易搞错的一点）
------------------------------
本工具读到的是「**光心地面投影** → 红目标」的距离；现场卷尺量到的是
「**脚尖** → 红目标」的距离。两者相差一个常数，2026-09-25 卷尺标定解出
**+4.24cm**（脚尖在光心投影前方 4.24cm，见 --ref-offset）。

所以判据不是"读数 ≈ 卷尺值"，而是：

    读数 − (卷尺值 + 4.24)  ≈  0

换个说法：**读数减卷尺值应该恒等于 +4.24**，摆多远都一样。不是常数就是有问题。

M3 验收线（方案 §7）：折算到光心口径后 d≤50cm 误差 ≤2cm、重复 σ ≤1cm；
近段 2~10cm 误差 ≤1cm（这条守的是起跨点）。

用法
----
真机交互（把红色目标摆在已知距离，输入卷尺读数回车即测）：
    python tools/verify_red_distance.py --pitch 1100
单帧离线分析（已知距离的已有照片）：
    python tools/verify_red_distance.py --image photo.jpg --dist 32.0 --pitch 1100

交互命令：输入数字=该卷尺读数下测一轮；s=统计汇总；q=退出。
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

#: 观测档与相机常数（2026-09-25 卷尺标定；改这里前先确认关卡 PITCH_OBS 一致）
PITCH_OBS = 1100
CAM_HEIGHT_CM = 33.9        # 光心离地 = h_eff 32.73 + 卷尺厚 1.20
PITCH_OFFSET_DEG = 19.1     # 安装下俯偏移 = 拟合俯角 55.11 − 名义 36.0

#: 卷尺零点（脚尖）相对光心地面投影的前向偏移（cm）。
#: 2026-09-25 卷尺标定解出 +4.24。现场卷尺多半从脚尖量，而本工具读的是
#: 光心投影口径 ⇒ 判据是"读数 − 卷尺值 ≈ 常数"，不是 "≈ 0"。
TOE_TO_CAMERA_CM = 4.24


def parse_args():
    ap = argparse.ArgumentParser(description="红色目标静态测距验证（检查工具，非标定）")
    ap.add_argument("--pitch", type=int, default=PITCH_OBS,
                    help=f"俯仰舵机脉宽（须与关卡 PITCH_OBS 一致），默认 {PITCH_OBS}")
    ap.add_argument("--cam-z", type=float, default=CAM_HEIGHT_CM,
                    help=f"站立实测相机高度 cm，默认 {CAM_HEIGHT_CM}")
    ap.add_argument("--pitch-offset", type=float, default=PITCH_OFFSET_DEG,
                    help=f"相机相对俯仰舵机的安装下俯偏移（度），默认 {PITCH_OFFSET_DEG}。"
                         "漏掉它测距会偏 1.7 倍——这是换舵机后新增的参数")
    ap.add_argument("--ref-offset", type=float, default=TOE_TO_CAMERA_CM,
                    help=f"你量距离时用的参考点比光心地面投影靠前多少 cm，"
                         f"默认 {TOE_TO_CAMERA_CM}（卷尺零点在脚尖）。"
                         "从别的参考点量就改这个值")
    ap.add_argument("--frames", type=int, default=3,
                    help="每轮测量帧数（取中位数），默认 3")
    ap.add_argument("--image", default=None, help="离线单帧分析模式")
    ap.add_argument("--dist", type=float, default=None,
                    help="离线模式的卷尺读数 cm（从 --ref-offset 指定的参考点量）")
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
    # ⚠ 必须带 pitch_offset_deg：漏掉它，模型会以为相机只低头 36°（实际 55.1°），
    # 测距偏 1.7 倍（真值 30cm 会报 49cm）。换舵机后新增的参数，别漏。
    meter = build_meter(args.pitch, args.cam_z,
                        pitch_offset_deg=args.pitch_offset)
    ref = args.ref_offset
    if meter.degraded:
        print(f"说明：未找到 {CALIB_PATH}，用解析模型（光心离地 {args.cam_z:.1f}cm + "
              f"安装偏移 {args.pitch_offset:.1f}°）——这是**正常且可用**的，"
              "那两个常数由 tools/calib_ruler_profile.py 卷尺标定给出（±3cm 级）。")
    print(f"距离口径：本工具报的是「光心地面投影 → 目标」；你输入的卷尺读数按"
          f"「参考点 → 目标」，参考点在光心投影前方 {ref:.2f}cm。")
    print(f"  ⇒ 判据：**读数 − 卷尺值 应恒等于 {ref:+.2f}cm**（摆多远都一样）")
    state = None
    if args.image:
        import cv2
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"读图失败: {args.image}")
            sys.exit(1)
        assert args.dist is not None, "离线模式需 --dist 卷尺读数"
        det = detector.detect(frame)
        m = meter.measure(det.components)
        if not m.exists:
            print("未检出红色目标")
            sys.exit(1)
        true_cam = args.dist + ref
        print(f"卷尺 {args.dist:.1f}cm (+{ref:.2f} 折算到光心 {true_cam:.2f}cm)"
              f" -> 测量 {m.forward_cm:.2f}cm "
              f"(误差 {m.forward_cm - true_cam:+.2f})  方位 "
              f"{m.bearing_err_deg:+.2f}°  横偏 {m.lateral_cm:+.2f}cm")
        return

    state = RobotState()
    state.set_head(HEAD_CENTER)
    state.set_pitch(args.pitch)

    # records: {卷尺读数: [(fwd,bearing,lat), ...]}
    records = {}
    print("\n输入卷尺读数(cm，从参考点量)测一轮；s=汇总；q=退出。"
          "同一距离可测多次（评估重复精度 σ）。")
    while True:
        s = input("\n卷尺读数(cm)/s/q: ").strip().lower()
        if s == "q":
            break
        if s == "s":
            print("\n===== 汇总 =====")
            for d, rs in sorted(records.items()):
                fwd = np.asarray([r[0] for r in rs])
                true_cam = d + ref               # 折算到光心口径的真值
                err = fwd - true_cam
                sigma = float(np.std(fwd)) if len(fwd) > 1 else float("nan")
                print(f"  卷尺 {d:6.1f}cm(光心口径 {true_cam:6.2f})  n={len(rs)}  "
                      f"误差均值 {err.mean():+.2f}cm  重复σ {sigma:.2f}cm")
                if true_cam <= 10.0 and abs(err.mean()) > 1.0:
                    print("    ⚠ 近距(≤10cm)误差超 1cm——这条守的是起跨点")
                elif true_cam <= 50.0 and abs(err.mean()) > 2.0:
                    print("    ⚠ 误差超 2cm（验收线）")
                if len(rs) > 1 and sigma > 1.0:
                    print("    ⚠ 重复 σ 超 1cm（站立晃动/协议不一致？）")
            print("  提示：各点的「读数 − 卷尺值」应恒等于 "
                  f"{ref:+.2f}cm。若随距离漂移，说明几何还不对。")
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
        print(f"卷尺 {d:.1f}cm -> 测量 {fwd:.2f}cm  "
              f"(差值 {fwd - d:+.2f}，应≈{ref:+.2f})  "
              f"方位 {bearing:+.2f}°  横偏 {lat:+.2f}cm")


if __name__ == "__main__":
    main()
