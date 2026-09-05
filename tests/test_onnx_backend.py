# -*- coding: utf-8 -*-
"""OnnxYoloBackend 前后处理纯逻辑测试（不依赖 onnxruntime 与模型文件）

覆盖：
  1. letterbox：2592x1944（项目相机分辨率）→ 640x640 的 scale/pad 正确、内容位置不变形；
  2. decode_yolo_output：置信度过滤、按类别 NMS 抑制重复框、
     letterbox → 原图坐标还原，以及 (N,4+nc)/(4+nc,N)/(1,4+nc,N) 三种布局一致。

运行：python tests/test_onnx_backend.py   （或 pytest tests/）
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

from vision.yolo_detector import letterbox, decode_yolo_output  # noqa: E402

NAMES = {0: "football", 1: "goal"}
CONF, IOU = 0.5, 0.45


def test_letterbox_4to3():
    frame = np.zeros((1944, 2592, 3), np.uint8)
    canvas, scale, pad_x, pad_y = letterbox(frame, 640)
    assert canvas.shape == (640, 640, 3)
    assert abs(scale - 640.0 / 2592.0) < 1e-9
    assert pad_x == 0
    assert pad_y == 80          # (640 - round(1944*scale)) // 2 = (640-480)//2
    assert canvas[79, 320, 0] == 114   # 上灰边
    assert canvas[80, 320, 0] == 0     # 图像内容区（黑帧）


def test_letterbox_keeps_content_position():
    frame = np.zeros((1944, 2592, 3), np.uint8)
    frame[450:550, 950:1050] = 255  # 原图 (1000, 500) 处白块
    canvas, scale, pad_x, pad_y = letterbox(frame, 640)
    cx, cy = int(1000 * scale) + pad_x, int(500 * scale) + pad_y
    assert canvas[cy, cx, 0] == 255


def _make_rows():
    """预测行（letterbox 像素 cxcywh + 2 类得分）：
    A 类0 高分；B = A 的近重复（应被 NMS 抑制）；
    C 类1 低分（被 conf 过滤）；D 类1 保留；
    再补 8 条低于阈值的噪声行，使 N=12 > 4+nc，模拟真实 ONNX 输出
    （固定 8400 锚点，N 恒大于 4+nc——解码器据此判定是否转置）。"""
    rows = [
        [320.0, 320.0, 100.0, 50.0, 0.9, 0.1],   # A
        [322.0, 320.0, 100.0, 50.0, 0.6, 0.1],   # B（与 A IoU≈0.96）
        [500.0, 400.0, 60.0, 60.0, 0.05, 0.3],   # C
        [480.0, 300.0, 80.0, 80.0, 0.1, 0.7],    # D
    ]
    for k in range(8):                            # 噪声行（低于阈值，全部被滤掉）
        rows.append([10.0 + k, 600.0 - k, 20.0, 20.0, 0.02, 0.01])
    return np.array(rows, dtype=np.float32)


def test_decode():
    frame = np.zeros((1944, 2592, 3), np.uint8)
    scale = 640.0 / 2592.0
    pad_x, pad_y = 0, 80
    dets = decode_yolo_output(_make_rows(), frame.shape, scale, pad_x, pad_y,
                              NAMES, CONF, IOU)

    assert len(dets) == 2, f"应剩 A/D 两条，实际 {len(dets)}"
    # 置信度降序：A(0.9) 在前，D(0.7) 在后（float32 存储须用容差比较）
    np.testing.assert_allclose([d.confidence for d in dets], [0.9, 0.7], atol=1e-5)
    assert [d.cls for d in dets] == ["football", "goal"]

    a, d = dets
    # A: letterbox(320,320,100,50) -pad(0,80) -> 原图坐标
    exp_a = ((320 - 50) / scale, (320 - 25 - pad_y) / scale,
             100 / scale, 50 / scale)
    np.testing.assert_allclose(a.bbox, exp_a, atol=1e-3)
    np.testing.assert_allclose(a.center_px,
                               (exp_a[0] + exp_a[2] / 2, exp_a[1] + exp_a[3] / 2),
                               atol=1e-3)


def test_decode_layouts_equal():
    frame = np.zeros((1944, 2592, 3), np.uint8)
    scale, pad_x, pad_y = 640.0 / 2592.0, 0, 80
    rows = _make_rows()
    ref = decode_yolo_output(rows, frame.shape, scale, pad_x, pad_y, NAMES, CONF, IOU)
    # (4+nc, N) 转置布局 与 (1, 4+nc, N) 批量布局，结果须一致
    for alt in (rows.T, rows.T[np.newaxis]):
        got = decode_yolo_output(alt, frame.shape, scale, pad_x, pad_y, NAMES, CONF, IOU)
        assert got == ref


def main():
    test_letterbox_4to3()
    test_letterbox_keeps_content_position()
    test_decode()
    test_decode_layouts_equal()
    print("tests/test_onnx_backend.py: 全部通过")


if __name__ == "__main__":
    main()
