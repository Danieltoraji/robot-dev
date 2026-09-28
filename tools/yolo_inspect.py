#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yolo_inspect.py —— PC 端 YOLO 检查器（cv2 GUI，离线验证模型本身）

对指定文件夹的图片逐张跑 OnnxYoloBackend，窗口内叠加显示框 + 类别 + 置信度，
trackbar 实时调置信度阈值。与机器人完全解耦。

用法（PC 上，需 onnxruntime + opencv-python）：
    python tools/yolo_inspect.py --dir datasets/label_work/images
    python tools/yolo_inspect.py --dir archive/result/dataset_raw/ball_1350c --conf 30

按键：D/→ 下一张   A/← 上一张   S 保存当前标注图   Q/ESC 退出
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from vision.yolo_detector import OnnxYoloBackend

WINDOW = "yolo_inspect"


def main():
    ap = argparse.ArgumentParser(description="YOLO 检查器（cv2 GUI）")
    ap.add_argument("--dir", required=True, help="图片文件夹")
    ap.add_argument("--model", default="models/football_goal_ball_v1_640.onnx")
    ap.add_argument("--conf", type=int, default=45, help="初始置信度阈值 %%（默认 45）")
    ap.add_argument("--input-size", type=int, default=640)
    ap.add_argument("--out", default="archive/result/yolo_inspect", help="S 键保存目录")
    args = ap.parse_args()

    files = sorted(f for f in glob.glob(os.path.join(args.dir, "**", "*.jpg"),
                                        recursive=True))
    if not files:
        print(f"错误: {args.dir} 下没有 jpg")
        sys.exit(1)
    print(f"共 {len(files)} 张 | 模型: {args.model}")

    backend = OnnxYoloBackend(args.model, conf=args.conf / 100.0,
                              iou=0.45, input_size=args.input_size)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    idx = [0]

    def on_conf(v):
        backend.conf = v / 100.0
        show(idx[0])  # 阈值变化后重画当前图

    def show(i):
        idx[0] = i % len(files)
        frame = cv2.imread(files[idx[0]])
        dets = []
        if frame is None:
            # 读图失败时用占位画布，避免后续 putText/frame.shape 崩溃
            frame = np.zeros((480, 960, 3), np.uint8)
            cv2.putText(frame, "READ FAIL: " + os.path.basename(files[idx[0]]),
                        (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                        (0, 0, 255), 3)
        else:
            dets = backend.detect(frame)
            for d in dets:
                x, y, w, h = [int(v) for v in d.bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 6)
                cv2.putText(frame, f"{d.cls} {d.confidence:.2f}",
                            (x, max(y - 14, 40)), cv2.FONT_HERSHEY_SIMPLEX,
                            2.2, (0, 255, 0), 4)
        label = f"[{idx[0] + 1}/{len(files)}] {os.path.basename(files[idx[0]])}  dets={len(dets)}"
        cv2.putText(frame, label, (40, frame.shape[0] - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)
        cv2.imshow(WINDOW, frame)

    cv2.createTrackbar("conf %", WINDOW, args.conf, 100, on_conf)
    show(0)

    print("按键: D/→ 下一张  A/← 上一张  S 保存当前图  Q/ESC 退出")
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key in (ord("d"), 83):        # d / 右方向键（部分平台 83）
            show(idx[0] + 1)
        elif key in (ord("a"), 81):        # a / 左方向键（部分平台 81）
            show(idx[0] - 1)
        elif key == ord("s"):
            os.makedirs(args.out, exist_ok=True)
            frame = cv2.imread(files[idx[0]])
            if frame is None:
                print("读图失败，未保存:", files[idx[0]])
            else:
                dets = backend.detect(frame)
                for d in dets:
                    x, y, w, h = [int(v) for v in d.bbox]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 6)
                    cv2.putText(frame, f"{d.cls} {d.confidence:.2f}",
                                (x, max(y - 14, 40)), cv2.FONT_HERSHEY_SIMPLEX, 2.2,
                                (0, 255, 0), 4)
                path = os.path.join(args.out, os.path.basename(files[idx[0]]))
                cv2.imwrite(path, frame)
                print("已保存:", path)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
