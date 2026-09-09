# -*- coding: utf-8 -*-
"""数字宫格地面单应点击标定工具（tools/calib_ninegrid.py）

为什么需要它
------------
布局预扫把"色块像素 → 场地格"需要地面单应 H。无标定文件时用
`GroundHomography.from_pose` 解析自举，但那只依赖"机器人起始位姿假设"
（缺省 (50,-20)/0°），只够格归属；现场必须点击标定覆盖，才能让起始位姿
假设误差不影响布局归属。

用法
----
交互点击（需桌面环境；机器人上可用 x11 转发或先拍照拷到 PC）：
    python tools/calib_ninegrid.py --pitch 1200 --image photo.jpg
    python tools/calib_ninegrid.py --pitch 1040          # 直接用相机拍
    python tools/calib_ninegrid.py --pitch 1200 --image photo.jpg --head 1500

脚本模式（无 GUI，坐标事先量好）：
    python tools/calib_ninegrid.py --pitch 1200 --image photo.jpg \
        --px "100,1800; 2500,1800; 2500,300; 100,300" \
        --ground "0,0; 100,0; 100,100; 0,100"

交互操作：鼠标左键按屏幕提示顺序点击 → u 撤销 → r 重置 → s 保存 → q/ESC 退出。
点击顺序缺省为宫格 4×4 交点（先行后列，从 y=0 近边开始）；可用 --ground 自定义。

产物：archive/result/ninegrid_homography.json（按俯仰脉宽分档，多档共存）
"""

import argparse
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from core.camera_config import HEAD_CENTER
from core.ground_homography import (
    GroundHomography, HOMOGRAPHY_PATH, grid_intersections,
)


def parse_args():
    ap = argparse.ArgumentParser(description="数字宫格地面单应点击标定")
    ap.add_argument("--pitch", type=int, default=1200,
                    help="俯仰舵机脉宽（导航 1200 / 低头 1040），默认 1200")
    ap.add_argument("--head", type=int, default=HEAD_CENTER,
                    help="头部脉宽（标定档），默认 1500（中位）")
    ap.add_argument("--image", default=None,
                    help="已有照片路径；缺省用机器人相机拍一张")
    ap.add_argument("--out", default=HOMOGRAPHY_PATH,
                    help=f"输出 JSON，默认 {HOMOGRAPHY_PATH}")
    ap.add_argument("--ground", default=None,
                    help="脚本模式：地面坐标 'x,y;x,y;...'（cm）")
    ap.add_argument("--px", default=None,
                    help="脚本模式：像素坐标 'x,y;x,y;...'（原生分辨率）")
    ap.add_argument("--max-points", type=int, default=16,
                    help="交互模式最多点击点数，默认 16（4×4 宫格交点）")
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


def default_ground_points(n):
    """缺省点击顺序：宫格 4×4 交点（先行后列，y=0 近边开始）"""
    pts = grid_intersections()
    return pts[:n]


def load_frame(args):
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"读图失败: {args.image}")
            sys.exit(1)
        print(f"读入照片: {args.image}  {frame.shape[1]}x{frame.shape[0]}")
        return frame
    # 真机拍照（与关卡同一条 fswebcam 链路）
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
    hg = GroundHomography.solve(px, ground, pitch, head_pulse=head)
    # 反投影自检
    errs = []
    for p, g in zip(px, ground):
        pred = hg.ground_to_pixels([g])[0]
        errs.append(float(np.hypot(pred[0] - p[0], pred[1] - p[1])))
    R, t, C, diff = hg.decompose()
    print(f"\n单应求解成功（{len(px)} 对点，pitch={pitch} head={head}）")
    print(f"  重投影误差: 中位 {np.median(errs):.2f}px  最大 {max(errs):.2f}px")
    print(f"  分解相机中心: ({C[0]:.1f}, {C[1]:.1f}, {C[2]:.1f}) cm"
          f"（高度应 ≈39cm；偏差大说明点选/俯仰档不对）")
    print(f"  H 两列模长差: {diff:.4f}（<0.02 为佳）")
    if max(errs) > 20:
        print("  ⚠ 最大重投影误差 >20px，建议检查点选顺序/像素坐标")
    if not (25.0 <= C[2] <= 55.0):
        print("  ⚠ 分解高度超出 25~55cm，单应可能不可信（点近共线？）")
    hg.save(out_path)
    print(f"已保存: {out_path}")
    return hg


class ClickCalibrator:
    """交互点击标定（鼠标左键记录，u 撤销，r 重置，s 保存，q/ESC 退出）"""

    def __init__(self, frame, ground_pts, pitch, head, out_path):
        self.frame = frame
        self.ground_pts = ground_pts
        self.pitch = pitch
        self.head = head
        self.out_path = out_path
        self.clicks = []          # 原生分辨率像素
        self.scale = min(1.0, 1280.0 / frame.shape[1])
        self.win = "calib_ninegrid"
        self.hg = None

    def to_display(self, p):
        return (int(round(p[0] * self.scale)), int(round(p[1] * self.scale)))

    def to_native(self, p):
        return (p[0] / self.scale, p[1] / self.scale)

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.clicks) < len(self.ground_pts):
            self.clicks.append(self.to_native((x, y)))
            self.render()

    def render(self):
        vis = self.frame.copy()
        # 已点：绿圈 + 序号；下一个：黄字提示
        for i, p in enumerate(self.clicks):
            cv2.circle(vis, (int(p[0]), int(p[1])), 18, (0, 255, 0), 5)
            cv2.putText(vis, str(i + 1), (int(p[0]) + 20, int(p[1]) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        vis = cv2.resize(vis, None, fx=self.scale, fy=self.scale)
        n = len(self.clicks)
        if n < len(self.ground_pts):
            g = self.ground_pts[n]
            tip = f"click #{n + 1}: ground ({g[0]:.0f},{g[1]:.0f})cm  u=undo r=reset q=quit"
        else:
            tip = f"done {n} pts — s=save q=quit"
        cv2.putText(vis, tip, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, (0, 255, 255), 3)
        cv2.imshow(self.win, vis)

    def run(self):
        print("\n交互点击标定开始：按屏幕提示顺序点击（u 撤销 / r 重置 / s 保存 / q 退出）")
        print("点击顺序（场地系 cm，x 右 y 前，入口在 y=0 一侧）：")
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
    ground_pts = default_ground_points(args.max_points)
    ClickCalibrator(frame, ground_pts, args.pitch, args.head,
                    args.out).run()


if __name__ == "__main__":
    main()
