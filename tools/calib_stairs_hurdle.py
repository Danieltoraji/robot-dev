# -*- coding: utf-8 -*-
"""上下楼梯与识别跨障关卡：机器人本地系地面单应标定（tools/calib_stairs_hurdle.py）

为什么需要它
------------
关卡的测距/对正依赖"机器人本地系"地面单应（原点=光心地面投影、y=前、
x=右）。缺省用 GroundHomography.from_pose 解析自举（±3cm 级）；现场用
本工具做 solve() 全 H 标定，一次性吸收相机高度/俯仰/偏航/横滚零偏。

标定一次，处处可用：本地系随机器人移动，标定产物与标定时的站位无关
（只要：地面平整、相机高度与俯仰不变、头部居中）。

现场操作流程（约 20 分钟）
------------------------
1. 机器人摆到观测姿态：stand，头 pitch=1000（与关卡 PITCH_OBS 一致）、
   头部回正（1500）。**之后不要再动俯仰/头部。**
2. 确定光心地面投影点 O：相机正下方地面位置（卷尺/吊线，从机身参考点量）。
3. 在 O 周围摆 4~6 个红色目标（胶带片/红色边角料），目标下沿按本地系
   坐标摆放（卷尺量，x 右正 y 前正 cm）。**必须含横向偏移的点**，否则
   偏航/横滚零偏不可观。建议布局见 --layout。
4. 拍照（本工具直接拍，或 --image 用已有照片），按顺序点击每个目标的
   下沿中点（与运行时度量的"下沿点"同语义）。
5. 保存 -> archive/result/stairs_hurdle_calib.json。**机器人位置自此后
   可任意移动**（本地系性质），但相机高度/俯仰变了必须重标。
6. 用 tools/verify_red_distance.py 在**另一个站位**复核测距精度。

用法
----
    python tools/calib_stairs_hurdle.py --pitch 1000            # 真机拍照+点击
    python tools/calib_stairs_hurdle.py --pitch 1000 --image photo.jpg
    python tools/calib_stairs_hurdle.py --pitch 1000 --image photo.jpg \
        --px "1200,1500; 1500,1600; 900,1700; 1600,1400" \
        --ground "0,25; 10,35; -10,40; 5,55"

交互：左键点击 → u 撤销 → r 重置 → s 保存 → q/ESC 退出
"""

import argparse
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from core.camera_config import HEAD_CENTER
from core.ground_line_meter import CALIB_PATH
from core.ground_homography import GroundHomography

# 建议的红色目标摆放布局（本地系 cm，x 右 y 前）：覆盖近/中/远 + 左右偏移
SUGGESTED_LAYOUT = [
    (0.0, 25.0), (0.0, 45.0), (0.0, 65.0),
    (-15.0, 35.0), (15.0, 35.0), (-12.0, 60.0),
]


def parse_args():
    ap = argparse.ArgumentParser(description="上下楼梯跨障关卡本地系单应标定")
    ap.add_argument("--pitch", type=int, default=1000,
                    help="俯仰舵机脉宽（须与关卡 PITCH_OBS 一致），默认 1000")
    ap.add_argument("--head", type=int, default=HEAD_CENTER,
                    help="头部脉宽（标定档），默认 1500（中位）")
    ap.add_argument("--image", default=None,
                    help="已有照片路径；缺省用机器人相机拍一张")
    ap.add_argument("--out", default=CALIB_PATH,
                    help=f"输出 JSON，默认 {CALIB_PATH}")
    ap.add_argument("--ground", default=None,
                    help="脚本模式：本地系地面坐标 'x,y;x,y;...'（cm）")
    ap.add_argument("--px", default=None,
                    help="脚本模式：像素坐标 'x,y;x,y;...'（原生分辨率）")
    return ap.parse_args()


def _parse_pairs(s):
    out = []
    for part in s.replace("，", ",").split(";"):
        part = part.strip()
        if not part:
            continue
        x, y = part.split(",")
        out.append((float(x), float(y)))
    return out


def load_frame(args):
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"读图失败: {args.image}")
            sys.exit(1)
        print(f"读入照片: {args.image}  {frame.shape[1]}x{frame.shape[0]}")
        return frame
    from core.robot_core import RobotState
    state = RobotState()
    state.set_head(args.head)
    state.set_pitch(args.pitch)
    frame = state.capture_frame()
    if frame is None:
        print("拍照失败（机器人未连接？可用 --image 指定已有照片）")
        sys.exit(1)
    print(f"已拍照: {frame.shape[1]}x{frame.shape[0]}")
    return frame


def solve_and_report(px, ground, pitch, head, out_path):
    if len(px) < 4:
        print(f"至少需要 4 对点，当前 {len(px)} 对")
        return None
    xs = [g[0] for g in ground]
    ys = [g[1] for g in ground]
    if max(xs) - min(xs) < 15.0:
        print("  ⚠ 地面点横向跨度 <15cm：偏航/横滚零偏不可观，"
              "建议补充左右偏移的目标重新标定")
    hg = GroundHomography.solve(px, ground, pitch, head_pulse=head)
    errs = []
    for p, g in zip(px, ground):
        pred = hg.ground_to_pixels([g])[0]
        errs.append(float(np.hypot(pred[0] - p[0], pred[1] - p[1])))
    R, t, C, diff = hg.decompose()
    print(f"\n单应求解成功（{len(px)} 对点，pitch={pitch} head={head}，"
          f"本地系原点=光心地面投影）")
    print(f"  重投影误差: 中位 {np.median(errs):.2f}px  最大 {max(errs):.2f}px")
    print(f"  反推相机相对本地系: 高度 {C[2]:.1f}cm（应≈站立实测值）、"
          f"水平 ({C[0]:.1f},{C[1]:.1f})——本地系标定下应≈(0,0)")
    print(f"  H 两列模长差: {diff:.4f}（<0.02 为佳）")
    if max(errs) > 20:
        print("  ⚠ 最大重投影误差 >20px，检查点选/坐标")
    if abs(C[0]) > 5 or abs(C[1]) > 5:
        print("  ⚠ 反推水平位置偏离原点 >5cm，点选/坐标可能有系统差")
    hg.save(out_path)
    print(f"已保存: {out_path}")
    print("下一步：换一个站位跑 tools/verify_red_distance.py 复核")
    return hg


class ClickCalibrator:
    """交互点击标定（左键记录，u 撤销，r 重置，s 保存，q/ESC 退出）"""

    def __init__(self, frame, ground_pts, pitch, head, out_path):
        self.frame = frame
        self.ground_pts = ground_pts
        self.pitch = pitch
        self.head = head
        self.out_path = out_path
        self.clicks = []
        self.scale = min(1.0, 1280.0 / frame.shape[1])
        self.win = "calib_stairs_hurdle"
        self.hg = None

    def to_native(self, p):
        return (p[0] / self.scale, p[1] / self.scale)

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.clicks) < len(self.ground_pts):
            self.clicks.append(self.to_native((x, y)))
            self.render()

    def render(self):
        vis = self.frame.copy()
        for i, p in enumerate(self.clicks):
            cv2.circle(vis, (int(p[0]), int(p[1])), 18, (0, 255, 0), 5)
            cv2.putText(vis, str(i + 1), (int(p[0]) + 20, int(p[1]) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        vis = cv2.resize(vis, None, fx=self.scale, fy=self.scale)
        n = len(self.clicks)
        if n < len(self.ground_pts):
            g = self.ground_pts[n]
            tip = f"click #{n + 1}: local ({g[0]:.0f},{g[1]:.0f})cm  u/r/s/q"
        else:
            tip = f"done {n} pts — s=save q=quit"
        cv2.putText(vis, tip, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, (0, 255, 255), 3)
        cv2.imshow(self.win, vis)

    def run(self):
        print("\n本地系点击标定：按提示顺序点击红色目标下沿中点"
              "（u 撤销 / r 重置 / s 保存 / q 退出）")
        print("点击顺序（本地系 cm，x 右 y 前，原点=光心地面投影）：")
        for i, g in enumerate(self.ground_pts):
            print(f"  {i + 1:2d}. ({g[0]:6.1f}, {g[1]:6.1f})")
        cv2.namedWindow(self.win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.win, self.on_mouse)
        self.render()
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("u") and self.clicks:
                self.clicks.pop()
                self.render()
            elif key == ord("r"):
                self.clicks = []
                self.render()
            elif key == ord("s"):
                if len(self.clicks) < 4:
                    print("至少 4 个点才能求解")
                    continue
                ground = self.ground_pts[:len(self.clicks)]
                self.hg = solve_and_report(self.clicks, ground, self.pitch,
                                           self.head, self.out_path)
        cv2.destroyAllWindows()
        return self.hg


def main():
    args = parse_args()
    if args.px or args.ground:
        if not (args.px and args.ground):
            print("脚本模式需同时给 --px 与 --ground")
            sys.exit(1)
        px = _parse_pairs(args.px)
        ground = _parse_pairs(args.ground)
        if len(px) != len(ground):
            print(f"点数不一致: px={len(px)} ground={len(ground)}")
            sys.exit(1)
        solve_and_report(px, ground, args.pitch, args.head, args.out)
        return

    frame = load_frame(args)
    ClickCalibrator(frame, SUGGESTED_LAYOUT, args.pitch, args.head,
                    args.out).run()


if __name__ == "__main__":
    main()
