# -*- coding: utf-8 -*-
"""YOLO 目标检测器：本地 / 远程多后端，可随时切换。

决策 D1 结论：本地（ultralytics / onnxruntime）与远程（HTTP → 算力端 Flask）
实现同一 detect(frame) 接口；关卡通过构造参数选择后端，切换时关卡代码不变。

本地后端两种：
  - LocalYoloBackend ：ultralytics（.pt / .onnx 均可），开发调试用；
  - OnnxYoloBackend  ：onnxruntime 纯 CPU，无 ultralytics 依赖，RPi 本地部署主链路。
远程后端仅用标准库 urllib + cv2。

letterbox / 输出解码为模块级函数，便于离线单测（tests/test_onnx_backend.py）。
"""

import cv2
import json
import urllib.request

import numpy as np

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


def letterbox(frame, size):
    """保持长宽比缩放到 size x size，短边居中填灰（与 ultralytics 预处理一致）

    返回 (canvas, scale, pad_x, pad_y)；
    换算关系：letterbox 坐标 = 原图坐标 * scale + pad。
    """
    h, w = frame.shape[:2]
    scale = min(size / h, size / w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y


def decode_yolo_output(pred, orig_shape, scale, pad_x, pad_y,
                       names, conf_thres, iou_thres):
    """解析 YOLOv8/11 检测头 ONNX 输出 → Detection 列表（原图像素坐标）

    pred      : (4+nc, N) 或 (N, 4+nc)；前 4 列为 cxcywh（letterbox 像素），
                其余为各类别得分（导出时已含 sigmoid，v8/11 无 objectness 列）。
    orig_shape: 原图 (h, w, ...)，用于坐标还原与越界裁剪。
    """
    pred = np.asarray(pred)
    if pred.ndim == 3:
        pred = pred[0]
    if pred.shape[0] < pred.shape[1]:      # (4+nc, N) -> (N, 4+nc)
        pred = pred.T

    boxes_cxcywh = pred[:, :4]
    class_scores = pred[:, 4:]
    cls_ids = class_scores.argmax(axis=1)
    confs = class_scores.max(axis=1)

    keep = confs > conf_thres
    boxes_cxcywh, confs, cls_ids = boxes_cxcywh[keep], confs[keep], cls_ids[keep]

    detections = []
    orig_h, orig_w = orig_shape[:2]
    for cid in np.unique(cls_ids):
        m = cls_ids == cid
        cb = boxes_cxcywh[m].astype(np.float64)
        cs = confs[m]
        # cxcywh -> xywh（NMSBoxes 输入格式），按类别独立 NMS
        xywh = np.stack([cb[:, 0] - cb[:, 2] / 2, cb[:, 1] - cb[:, 3] / 2,
                         cb[:, 2], cb[:, 3]], axis=1)
        idx = cv2.dnn.NMSBoxes(xywh.tolist(), cs.tolist(), conf_thres, iou_thres)
        if idx is None or len(idx) == 0:
            continue
        for i in np.array(idx).flatten():
            x, y, w, h = xywh[i]
            # letterbox 坐标 -> 原图坐标，并裁剪到图内
            x1 = min(max((x - pad_x) / scale, 0.0), orig_w - 1.0)
            y1 = min(max((y - pad_y) / scale, 0.0), orig_h - 1.0)
            w1 = min(w / scale, orig_w - x1)
            h1 = min(h / scale, orig_h - y1)
            detections.append(Detection(
                cls=names.get(int(cid), str(int(cid))),
                confidence=float(cs[i]),
                bbox=(x1, y1, w1, h1),
                center_px=(x1 + w1 / 2.0, y1 + h1 / 2.0),
            ))
    # 置信度降序，便于关卡直接取最高分目标
    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections


class OnnxYoloBackend:
    """本地 ONNX 推理后端：onnxruntime CPU，无 ultralytics 依赖。

    RPi 本地部署主链路（《足球与球门识别训练部署方案》§6.4）：
    fswebcam 原图 → letterbox(640) → onnxruntime → 解码+按类别 NMS → Detection。

    model_path : 导出的 .onnx（固定输入尺寸，如 640）
    conf / iou : 置信度 / NMS IoU 阈值
    input_size : 导出时的输入边长（默认 640）
    names      : 类别 id -> 名称；默认与 data.yaml（0=football, 1=goal）对齐
    """

    def __init__(self, model_path, conf=0.25, iou=0.45, input_size=640, names=None):
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.input_size = input_size
        self.names = dict(names) if names else {0: "football", 1: "goal"}
        self._session = None
        self._input_name = None

    def _load(self):
        if self._session is None:
            import onnxruntime as ort  # 懒加载，避免无此库时影响 import 本模块
            self._session = ort.InferenceSession(
                self.model_path, providers=["CPUExecutionProvider"])
            self._input_name = self._session.get_inputs()[0].name
        return self._session

    def detect(self, frame):
        session = self._load()
        canvas, scale, pad_x, pad_y = letterbox(frame, self.input_size)
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 255.0, swapRB=True)  # (1,3,size,size)
        pred = session.run(None, {self._input_name: blob})[0]
        return decode_yolo_output(pred, frame.shape, scale, pad_x, pad_y,
                                  self.names, self.conf, self.iou)


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
