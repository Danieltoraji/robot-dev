#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench_yolo.py —— 检测后端单帧延迟基准（PC / 树莓派通用）

对一批实拍图计时 detect() 全流程（letterbox + 推理 + NMS，不含拍照），
输出 平均/中位/P95/最大 延迟与等效帧率。
验收线（《足球与球门识别训练部署方案》§6.6）：RPi5 单帧推理 ≤ 1000ms。

用法：
    python tools/bench_yolo.py --model models/football_goal_640.onnx --images <实拍目录>
    python tools/bench_yolo.py --model best.pt --image photo.jpg --warmup 5
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.yolo_detector import YoloDetector, LocalYoloBackend, OnnxYoloBackend

IMG_EXTS = (".jpg", ".jpeg", ".png")


def build_backend(args):
    if args.backend == "onnx" or (args.backend == "auto"
                                  and args.model.lower().endswith(".onnx")):
        return OnnxYoloBackend(args.model, conf=args.conf, iou=args.iou,
                               input_size=args.input_size)
    return LocalYoloBackend(args.model, conf=args.conf, iou=args.iou,
                            device=args.device)


def collect_images(args):
    paths = []
    for item in [args.image] if args.image else []:
        paths.append(item)
    if args.images:
        for name in sorted(os.listdir(args.images)):
            if name.lower().endswith(IMG_EXTS):
                paths.append(os.path.join(args.images, name))
    if not paths:
        print("错误: 请用 --image 或 --images 提供图片")
        sys.exit(1)
    return paths[:args.max_images]


def main():
    parser = argparse.ArgumentParser(description="检测后端单帧延迟基准")
    parser.add_argument("--model", required=True, help=".onnx 或 .pt 模型路径")
    parser.add_argument("--images", help="实拍图目录")
    parser.add_argument("--image", help="单张图片（与 --images 二选一或混用）")
    parser.add_argument("--backend", choices=("auto", "onnx", "local"), default="auto",
                        help="onnx=OnnxYoloBackend，local=ultralytics；auto 按扩展名")
    parser.add_argument("--conf", type=float, default=0.45)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--device", default=None, help="local 后端设备（cpu/cuda:0）")
    parser.add_argument("--warmup", type=int, default=3, help="预热次数（不计入统计）")
    parser.add_argument("--max-images", type=int, default=50)
    parser.add_argument("--save-dir", help="可选：保存带框可视化到此目录")
    args = parser.parse_args()

    paths = collect_images(args)
    det = YoloDetector(build_backend(args))

    frames = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"[WARN] 读图失败: {p}")
        else:
            frames.append(img)
    if not frames:
        print("没有可用图片")
        sys.exit(1)

    for _ in range(args.warmup):        # 预热：懒加载/内存分配不计入统计
        det.detect(frames[0])

    ms_list = []
    for i, frame in enumerate(frames):
        t0 = time.perf_counter()
        dets = det.detect(frame)
        ms_list.append((time.perf_counter() - t0) * 1000.0)
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            for d in dets:
                x, y, w, h = [int(v) for v in d.bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
                cv2.putText(frame, f"{d.cls} {d.confidence:.2f}", (x, max(y - 8, 16)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(args.save_dir, os.path.basename(paths[i])), frame)

    ms = np.array(ms_list)
    print("=" * 56)
    print(f"模型: {args.model} | 图片: {len(frames)} 张 | 预热: {args.warmup}")
    print(f"平均 {ms.mean():7.1f} ms | 中位 {np.median(ms):7.1f} ms | "
          f"P95 {np.percentile(ms, 95):7.1f} ms")
    print(f"最小 {ms.min():7.1f} ms | 最大 {ms.max():7.1f} ms | "
          f"等效帧率 {1000.0 / ms.mean():5.2f} fps")
    budget = "RPi5 ≤1000ms"
    print(f"验收线: {budget} -> {'达标 ✓' if ms.mean() <= 1000 else '超时 ✗'}")
    print("=" * 56)


if __name__ == "__main__":
    main()
