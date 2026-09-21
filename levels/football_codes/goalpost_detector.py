#!/usr/bin/python3
# coding=utf8
"""
goalpost_detector.py — 可独立复用的球门柱 YOLO 检测器

此模块只负责球门柱检测，不负责踢球、越线判断或机器人导航。
检测模型沿用 Functions/finalkick.py 中使用的：
    Functions/weights/best.onnx

返回的每个检测结果为：
    (cx, cy, w, h, confidence)
其中坐标均为输入原图像素坐标。
"""

from __future__ import print_function

import os

cv2 = None
np = None


def _ensure_dependencies():
    global cv2, np
    if cv2 is None or np is None:
        import cv2 as _cv2
        import numpy as _np
        cv2 = _cv2
        np = _np


DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "weights", "best.onnx"
)


class GoalPostDetector:
    """使用 OpenCV DNN 推理球门柱，接口与 finalkick.detect_goalposts 对齐。"""

    def __init__(
        self,
        model_path=DEFAULT_MODEL_PATH,
        conf_threshold=0.30,
        iou_threshold=0.45,
        input_size=640,
    ):
        _ensure_dependencies()
        self.model_path = model_path
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.input_size = int(input_size)

        if not os.path.exists(self.model_path):
            raise IOError("球门柱模型不存在: {}".format(self.model_path))

        self.net = cv2.dnn.readNetFromONNX(self.model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def _preprocess(self, frame):
        height, width = frame.shape[:2]
        scale = min(float(self.input_size) / width, float(self.input_size) / height)
        new_width = int(width * scale)
        new_height = int(height * scale)

        canvas = np.full(
            (self.input_size, self.input_size, 3), 114, dtype=np.uint8
        )
        dx = (self.input_size - new_width) // 2
        dy = (self.input_size - new_height) // 2
        resized = cv2.resize(frame, (new_width, new_height))
        canvas[dy:dy + new_height, dx:dx + new_width] = resized

        blob = cv2.dnn.blobFromImage(
            canvas,
            1.0 / 255.0,
            (self.input_size, self.input_size),
            swapRB=True,
            crop=False,
        )
        return blob, scale, dx, dy

    def detect(self, frame):
        _ensure_dependencies()
        """返回所有 NMS 后的球门柱检测结果，按置信度从高到低排列。"""
        if frame is None or getattr(frame, "size", 0) == 0:
            return []

        blob, scale, dx, dy = self._preprocess(frame)
        self.net.setInput(blob)
        outputs = self.net.forward()
        predictions = outputs[0]

        if predictions.ndim != 2 or predictions.shape[1] < 6:
            return []

        object_confidence = predictions[:, 4]
        class_confidence = (
            predictions[:, 5]
            if predictions.shape[1] == 6
            else predictions[:, 5:].max(axis=1)
        )
        scores = object_confidence * class_confidence
        mask = scores > self.conf_threshold
        if not np.any(mask):
            return []

        filtered = predictions[mask]
        filtered_scores = scores[mask]
        boxes = []
        for detection in filtered:
            cx, cy, width, height = detection[:4]
            boxes.append([
                float(cx - width / 2.0),
                float(cy - height / 2.0),
                float(width),
                float(height),
            ])

        indices = cv2.dnn.NMSBoxes(
            boxes,
            filtered_scores.tolist(),
            self.conf_threshold,
            self.iou_threshold,
        )
        if indices is None or len(indices) == 0:
            return []

        results = []
        for index in np.asarray(indices).flatten():
            x1, y1, width, height = boxes[int(index)]
            center_x = (x1 + width / 2.0 - dx) / scale
            center_y = (y1 + height / 2.0 - dy) / scale
            width = width / scale
            height = height / scale
            results.append(
                (
                    int(center_x),
                    int(center_y),
                    int(width),
                    int(height),
                    float(filtered_scores[int(index)]),
                )
            )

        results.sort(key=lambda item: item[4], reverse=True)
        return results



def goal_center_x(posts):
    """返回当前检测到的球门柱集合中心 x；没有检测结果时返回 None。"""
    if not posts:
        return None
    return int(sum(post[0] for post in posts) / float(len(posts)))



def draw_goalposts(frame, posts, color=(255, 0, 255)):
    _ensure_dependencies()
    """在图像上绘制球门柱检测框，供走路 Demo 调试显示。"""
    image = frame.copy()
    for cx, cy, width, height, confidence in posts:
        x1 = int(cx - width / 2)
        y1 = int(cy - height / 2)
        x2 = int(cx + width / 2)
        y2 = int(cy + height / 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.circle(image, (int(cx), int(cy)), 3, color, -1)
        cv2.putText(
            image,
            "GoalPost {:.0f}%".format(confidence * 100.0),
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
    return image


