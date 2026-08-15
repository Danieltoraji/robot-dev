# -*- coding: utf-8 -*-
"""视觉检测器离线调试工具：读一张照片 → 跑检测器 → 画结果 → 显示/保存。

在 PC 上离线调参，不占用机器人、可复现。用法见《视觉调试指南.md》，或：
  python tools/debug_vision.py --help
"""

import argparse
import sys
from pathlib import Path

# 让脚本能从任意目录 import 到仓库根的 vision 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from vision.line_detector import LineDetector
from vision.color_detector import ColorBlobDetector
from vision.digit_recognizer import DigitRecognizer
from vision.yolo_detector import YoloDetector, LocalYoloBackend, RemoteYoloBackend


def load_templates(template_dir):
    templates = {}
    for p in sorted(Path(template_dir).glob("*.png")):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        templates[p.stem] = img
    return templates


def draw_line(frame, result, roi_ratio):
    h, w = frame.shape[:2]
    cy = int(h * (1.0 - roi_ratio / 2.0))
    cv2.line(frame, (0, cy), (w, cy), (255, 255, 0), 1)
    if result.exists:
        cx = int(result.offset_x + w / 2.0)
        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), 2)
        cv2.putText(frame, f"offset={result.offset_x:.1f} conf={result.confidence:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "no line", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


def draw_boxes(frame, detections):
    for d in detections:
        x, y, w, h = [int(v) for v in d.bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{d.cls} {d.confidence:.2f}"
        cv2.putText(frame, label, (x, max(0, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def _show_or_save(frame, out_path):
    if out_path:
        cv2.imwrite(out_path, frame)
        print(f"结果已保存: {out_path}")
    else:
        cv2.imshow("debug_vision", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="视觉检测器离线调试")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("line")
    p.add_argument("--image", required=True)
    p.add_argument("--thresh", type=int, default=None)
    p.add_argument("--roi", type=float, default=0.5)
    p.add_argument("--out", default=None)

    p = sub.add_parser("color")
    p.add_argument("--image", required=True)
    p.add_argument("--hsv", required=True, help="h_min,s_min,v_min,h_max,s_max,v_max")
    p.add_argument("--label", default="blob")
    p.add_argument("--min-area", type=int, default=50)
    p.add_argument("--out", default=None)

    p = sub.add_parser("digit")
    p.add_argument("--image", required=True)
    p.add_argument("--templates", default=None)
    p.add_argument("--thresh", type=int, default=None)
    p.add_argument("--min-area", type=int, default=30)
    p.add_argument("--out", default=None)

    p = sub.add_parser("yolo")
    p.add_argument("--image", required=True)
    p.add_argument("--model", default=None, help="本地模型路径")
    p.add_argument("--url", default=None, help="远程服务地址（优先于 --model）")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--out", default=None)

    args = ap.parse_args()
    frame = cv2.imread(args.image)
    if frame is None:
        print(f"读图失败: {args.image}")
        sys.exit(1)
    out = frame.copy()

    if args.cmd == "line":
        det = LineDetector(thresh=args.thresh, roi_ratio=args.roi)
        r = det.detect(out)
        draw_line(out, r, args.roi)
        print(r)

    elif args.cmd == "color":
        hsv = [int(v) for v in args.hsv.split(",")]
        det = ColorBlobDetector((hsv[:3], hsv[3:]), label=args.label, min_area=args.min_area)
        rs = det.detect(out)
        draw_boxes(out, rs)
        print(rs)

    elif args.cmd == "digit":
        templates = load_templates(args.templates) if args.templates else {}
        det = DigitRecognizer(templates=templates, thresh=args.thresh, min_area=args.min_area)
        r = det.detect(out)
        cv2.putText(out, f"digits={r.digits} conf={r.confidence:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        print(r)

    elif args.cmd == "yolo":
        if args.url:
            backend = RemoteYoloBackend(args.url)
        elif args.model:
            backend = LocalYoloBackend(args.model, conf=args.conf)
        else:
            print("yolo 需指定 --model 或 --url")
            sys.exit(1)
        rs = YoloDetector(backend).detect(out)
        draw_boxes(out, rs)
        print(rs)

    _show_or_save(out, args.out)


if __name__ == "__main__":
    main()
