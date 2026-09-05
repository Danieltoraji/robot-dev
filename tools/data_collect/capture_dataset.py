#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
capture_dataset.py —— 足球/球门数据集批量采集（机器人端）

按《足球与球门识别训练部署方案》§3 的覆盖矩阵批量拍照：fswebcam 静态照
（与定位链路同款命令，2592x1944 + -S 3 稳曝光），输出到
    <out>/<场景>/<场景>_<头部档>_<序号>.jpg

每个场景一个子目录，后续 split_dataset.py 以子目录名为场景键做整体划分，
同一场景的连拍不会同时进 train 与 val/test。

覆盖矩阵与拍摄 checklist 见 docs/视觉能力/足球球门数据采集规范.md。

用法（机器人上，项目根目录）：
    python3 tools/data_collect/capture_dataset.py --scene ball_near --count 10
    python3 tools/data_collect/capture_dataset.py --scene goal_light_bg --count 15 --head left
    python3 tools/data_collect/capture_dataset.py --scene negative --count 20 --interval 2

本脚本只做拍照与命名，不依赖 hiwonder SDK；头部档位由操作者先手动调好，
用 --head 记录进文件名。
"""

import argparse
import os
import subprocess
import sys
import time

# 允许直接运行本文件
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.camera_config import CAMERA_WIDTH, CAMERA_HEIGHT

HEAD_LABELS = ("center", "right", "left", "wide_right", "wide_left")


def capture_one(path):
    """拍一张，成功返回 True（fswebcam 命令与 RobotState.capture_image 保持一致）"""
    cmd = (f"fswebcam -r {CAMERA_WIDTH}x{CAMERA_HEIGHT} "
           f"--no-banner -S 3 {path}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  拍照失败: {result.stderr.strip()}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="足球/球门数据集批量采集（fswebcam 静态照）")
    parser.add_argument("--scene", required=True,
                        help="场景名（子目录与文件名前缀，如 ball_near / goal_light_bg / negative）")
    parser.add_argument("--count", type=int, default=10, help="本场景拍照张数（默认 10）")
    parser.add_argument("--head", choices=HEAD_LABELS, default="center",
                        help="头部档位标注（仅记录进文件名，请先手动把头部调到对应位置）")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="相邻两张间隔秒（默认 1.0，给摆拍留时间）")
    parser.add_argument("--out", default="/home/pi/dataset_raw", help="输出根目录")
    parser.add_argument("--start", type=int, default=1, help="起始序号（同场景续拍时用）")
    args = parser.parse_args()

    scene_dir = os.path.join(args.out, args.scene)
    os.makedirs(scene_dir, exist_ok=True)

    print("=" * 60)
    print(f"场景: {args.scene} | 头部: {args.head} | 张数: {args.count}")
    print("拍前自查: 本次要覆盖的距离/角度/光照组合是否已就位？")
    print("=" * 60)

    ok = 0
    for i in range(args.count):
        idx = args.start + i
        path = os.path.join(scene_dir, f"{args.scene}_{args.head}_{idx:03d}.jpg")
        print(f"[{idx}] -> {path}")
        if capture_one(path):
            ok += 1
        if i < args.count - 1 and args.interval > 0:
            time.sleep(args.interval)

    print(f"完成: {ok}/{args.count} 张已存入 {scene_dir}")
    if ok < args.count:
        sys.exit(1)


if __name__ == "__main__":
    main()
