# -*- coding: utf-8 -*-
"""YOLO 目标检测器：本地 / 远程双后端，可随时切换。

决策 D1 结论：同时提供本地（ultralytics/onnx 等）与远程（HTTP → 算力端 Flask）
两套后端，二者实现同一 detect(frame) 接口；关卡通过构造参数选择后端，切换时
关卡代码不变。

本地后端懒加载 ultralytics（未安装也不影响 import 本模块）；远程后端仅用
标准库 urllib + cv2。
"""

import cv2
import json
import urllib.request

from .detection import Detection


class LocalYoloBackend:
    """本地推理后端：ultralytics YOLO 模型。

    model_path : 模型权重路径（.pt / .onnx 等，由 ultralytics 决定支持格式）
    conf / iou : 置信度 / NMS IoU 阈值
    device     : 推理设备（None 自动；"cpu" / "cuda:0" / "0" 等）
    """

    def __init__(self, model_path, conf=0.25, iou=0.45, device=None):
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO  # 懒加载，避免无此库时影响 import
            self._model = YOLO(self.model_path)
        return self._model

    def detect(self, frame):
        model = self._load()
        results = model.predict(frame, conf=self.conf, iou=self.iou,
                                device=self.device, verbose=False)
        out = []
        for r in results:
            names = r.names
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                out.append(Detection(
                    cls=names.get(cls_id, str(cls_id)),
                    confidence=conf,
                    bbox=(x1, y1, x2 - x1, y2 - y1),
                    center_px=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                ))
        return out


class RemoteYoloBackend:
    """远程推理后端：HTTP POST 图片到算力端（如 KV260 的 Flask 服务）。

    协议：POST 原始 JPEG 字节，Content-Type: image/jpeg；
    返回 JSON {"detections": [{"cls": ..., "confidence": ..., "bbox": [x, y, w, h]}, ...]}。
    网络异常时打印告警并返回空列表（由关卡决定重试或其它处理）。
    """

    def __init__(self, url, timeout=5.0):
        self.url = url
        self.timeout = timeout

    def detect(self, frame):
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            print("[RemoteYoloBackend] 图像编码失败")
            return []
        req = urllib.request.Request(
            self.url, data=buf.tobytes(),
            headers={"Content-Type": "image/jpeg"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"[RemoteYoloBackend] 请求失败: {e}")
            return []

        out = []
        for d in payload.get("detections", []):
            x, y, w, h = d.get("bbox", [0, 0, 0, 0])
            out.append(Detection(
                cls=str(d.get("cls", "unknown")),
                confidence=float(d.get("confidence", 0.0)),
                bbox=(float(x), float(y), float(w), float(h)),
                center_px=(float(x) + float(w) / 2.0, float(y) + float(h) / 2.0),
            ))
        return out


class YoloDetector:
    """YOLO 检测器统一入口，持有 local / remote 后端之一。

    用法：
      # 本地
      det = YoloDetector(LocalYoloBackend("yolov8n.pt", conf=0.5))
      # 远程（切换只需换后端，关卡代码不变）
      det = YoloDetector(RemoteYoloBackend("http://192.168.1.10:5000/detect"))
    """

    def __init__(self, backend):
        self.backend = backend

    def detect(self, frame):
        return self.backend.detect(frame)
