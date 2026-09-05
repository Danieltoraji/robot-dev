#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_onnx_vs_pt.py —— 部署后端一致性校验：ultralytics vs OnnxYoloBackend

同一批图分别用 LocalYoloBackend(.pt) 与 OnnxYoloBackend(.onnx) 检测，
按「类别相同 + IoU ≥ --min-iou」做一一匹配，输出逐图匹配统计与最差 IoU。
全部匹配退出码 0，否则 1（CI/入库前把关用）。

前提：onnx 由同一 best.pt 导出且输入尺寸一致，conf/iou 参数相同。
用法（PC 或机器人，需 ultralytics + onnxruntime）：
    python tools/verify_onnx_vs_pt.py --pt best.pt \
        --onnx models/football_goal_640.onnx --images <目录或单张图>
"""

import argparse
import os
import sys

import cv2

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.yolo_detector import YoloDetector, LocalYoloBackend, OnnxYoloBackend

IMG_EXTS = (".jpg", ".jpeg", ".png")


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="pt 与 onnx 后端检测结果一致性校验")
    parser.add_argument("--pt", required=True, help="ultralytics 权重（best.pt）")
    parser.add_argument("--onnx", required=True, help="同权重导出的 onnx")
    parser.add_argument("--images", required=True, help="图片目录或单张图片")
    parser.add_argument("--conf", type=float, default=0.45)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--input-size", type=int, default=640, help="onnx 导出输入边长")
    parser.add_argument("--min-iou", type=float, default=0.9, help="判定为同一目标的 IoU 下限")
    args = parser.parse_args()

    if os.path.isdir(args.images):
        files = sorted(os.path.join(args.images, f) for f in os.listdir(args.images)
                       if f.lower().endswith(IMG_EXTS))
    else:
        files = [args.images]
    if not files:
        print("没有找到图片")
        sys.exit(1)

    pt = YoloDetector(LocalYoloBackend(args.pt, conf=args.conf, iou=args.iou,
                                       device="cpu"))
    onnx = YoloDetector(OnnxYoloBackend(args.onnx, conf=args.conf, iou=args.iou,
                                        input_size=args.input_size))

    total_pt = total_onnx = matched = 0
    worst = 1.0
    for path in files:
        frame = cv2.imread(path)
        if frame is None:
            print(f"[WARN] 读图失败: {path}")
            continue
        dets_pt = pt.detect(frame)
        dets_onnx = onnx.detect(frame)
        total_pt += len(dets_pt)
        total_onnx += len(dets_onnx)

        used, n_match = set(), 0
        for d in dets_pt:
            best_j, best_v = -1, 0.0
            for j, e in enumerate(dets_onnx):
                if j in used or e.cls != d.cls:
                    continue
                v = iou(d.bbox, e.bbox)
                if v > best_v:
                    best_v, best_j = v, j
            if best_j >= 0 and best_v >= args.min_iou:
                used.add(best_j)
                n_match += 1
                worst = min(worst, best_v)
        matched += n_match
        print(f"{os.path.basename(path)}: pt={len(dets_pt)} onnx={len(dets_onnx)} 匹配={n_match}")

    print("-" * 50)
    print(f"总计: pt={total_pt} onnx={total_onnx} 匹配={matched} 最差IoU={worst:.3f}")
    if total_pt > 0 and matched == total_pt == total_onnx:
        print("一致 ✓")
        sys.exit(0)
    print("不一致 ✗（检查导出输入尺寸/conf/iou 是否与校验参数一致）")
    sys.exit(1)


if __name__ == "__main__":
    main()
