#!/usr/bin/python3
# coding=utf8
"""
goal_line_web.py — 独立球门线越线判断测试网页

用途：
    不启动 finalkick.py，不控制机器人运动，只打开摄像头、运行足球/球门柱
    检测，并把 goal_line_judge.check_ball_crossed_goal_line() 的结果显示在网页上。

运行：
    cd /home/pi/TonyPi/Functions
    python3 goal_line_web.py

浏览器：
    http://<机器人IP>:5002
"""

import json
import os
import sys
import threading
import time

_FUNCTIONS_DIR = os.path.dirname(os.path.abspath(__file__))
if _FUNCTIONS_DIR not in sys.path:
    sys.path.insert(0, _FUNCTIONS_DIR)

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request

import hiwonder.Camera as Camera

if __name__ == "__main__":
    from CameraCalibration.CalibrationConfig import *
else:
    from Functions.CameraCalibration.CalibrationConfig import *

from goal_line_judge import check_ball_crossed_goal_line, _estimate_goal_line


# ======================== 参数 ========================
WEB_PORT = 5002
STREAM_QUALITY = 75
FRAME_INTERVAL = 0.01
GOALPOST_REFRESH_INTERVAL = 3
INPUT_SIZE = 640
FOOTBALL_CONF = 0.40
GOALPOST_CONF = 0.30
IOU_THRESHOLD = 0.45
IMG_W, IMG_H = 640, 480

_DIR = os.path.dirname(os.path.abspath(__file__))
FOOTBALL_MODEL_PATH = os.path.join(_DIR, "models", "football_best_win.onnx")
GOALPOST_MODEL_PATH = os.path.join(_DIR, "weights", "best.onnx")


# ======================== 初始化 ========================
param_data = np.load(calibration_param_path + ".npz")
mtx = param_data["mtx_array"]
dist = param_data["dist_array"]
newcameramtx, _ = cv2.getOptimalNewCameraMatrix(
    mtx, dist, (IMG_W, IMG_H), 0, (IMG_W, IMG_H)
)
mapx, mapy = cv2.initUndistortRectifyMap(
    mtx, dist, None, newcameramtx, (IMG_W, IMG_H), 5
)

print(f"加载足球模型: {FOOTBALL_MODEL_PATH}")
football_net = cv2.dnn.readNetFromONNX(FOOTBALL_MODEL_PATH)
football_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
football_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

print(f"加载球门柱模型: {GOALPOST_MODEL_PATH}")
goalpost_net = cv2.dnn.readNetFromONNX(GOALPOST_MODEL_PATH)
goalpost_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
goalpost_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

app = Flask(__name__)
stop_event = threading.Event()
frame_lock = threading.Lock()

latest_jpeg = None
camera_ready = False
settings = {
    "goal_side": "above",
    "margin_px": 3.0,
    "confirm_frames": 1,
}
current_streak = 0
goal_event_latched = False  # 任务成功锁存：足球整体完全越线才算成功
smoothed_red_line = None
current_status = {
    "camera_ready": False,
    "ball": None,
    "goalposts": [],
    "line_valid": False,
    "partial_crossed": False,
    "whole_crossed": False,
    "confirmed": False,
    "streak": 0,
    "line": None,
    "bbox": None,
    "signed_distances": [],
    "goal_side": settings["goal_side"],
    "margin_px": settings["margin_px"],
    "confirm_frames": settings["confirm_frames"],
    "processing_fps": 0.0,
    "goal_event_latched": False,
    "task_success": False,
    "line_source": "unavailable",
    "red_line_valid": False,
    "red_pixel_count": 0,
    "last_error": "",
    "updated_at": 0.0,
}


# ======================== YOLO 检测 ========================
def _yolo_preprocess(frame):
    height, width = frame.shape[:2]
    scale = min(INPUT_SIZE / width, INPUT_SIZE / height)
    new_width, new_height = int(width * scale), int(height * scale)
    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    dx = (INPUT_SIZE - new_width) // 2
    dy = (INPUT_SIZE - new_height) // 2
    canvas[dy:dy + new_height, dx:dx + new_width] = cv2.resize(
        frame, (new_width, new_height)
    )
    blob = cv2.dnn.blobFromImage(
        canvas, 1.0 / 255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True, crop=False
    )
    return blob, scale, dx, dy


def detect_football(frame):
    """返回 (cx, cy, w, h, conf)，未检测到时返回 None。"""
    blob, scale, dx, dy = _yolo_preprocess(frame)
    football_net.setInput(blob)
    predictions = football_net.forward()[0]
    scores = predictions[:, 4] * predictions[:, 5]
    mask = scores > FOOTBALL_CONF
    if not np.any(mask):
        return None

    selected_predictions = predictions[mask]
    selected_scores = scores[mask]
    boxes = [
        [item[0] - item[2] / 2, item[1] - item[3] / 2, item[2], item[3]]
        for item in selected_predictions
    ]
    indices = cv2.dnn.NMSBoxes(
        boxes, selected_scores.tolist(), FOOTBALL_CONF, IOU_THRESHOLD
    )
    if len(indices) == 0:
        return None

    index = int(np.array(indices).flatten()[0])
    x1, y1, width, height = boxes[index]
    cx = (x1 + width / 2 - dx) / scale
    cy = (y1 + height / 2 - dy) / scale
    return (
        int(cx), int(cy), int(width / scale), int(height / scale),
        float(selected_scores[index]),
    )


def detect_goalposts(frame):
    """返回球门柱列表 [(cx, cy, w, h, conf), ...]。"""
    blob, scale, dx, dy = _yolo_preprocess(frame)
    goalpost_net.setInput(blob)
    predictions = goalpost_net.forward()[0]
    object_confidence = predictions[:, 4]
    class_confidence = (
        predictions[:, 5]
        if predictions.shape[1] == 6
        else predictions[:, 5:].max(axis=1)
    )
    scores = object_confidence * class_confidence
    mask = scores > GOALPOST_CONF
    if not np.any(mask):
        return []

    selected_predictions = predictions[mask]
    selected_scores = scores[mask]
    boxes = [
        [item[0] - item[2] / 2, item[1] - item[3] / 2, item[2], item[3]]
        for item in selected_predictions
    ]
    indices = cv2.dnn.NMSBoxes(
        boxes, selected_scores.tolist(), GOALPOST_CONF, IOU_THRESHOLD
    )
    if len(indices) == 0:
        return []

    result = []
    for index in np.array(indices).flatten():
        x1, y1, width, height = boxes[int(index)]
        cx = (x1 + width / 2 - dx) / scale
        cy = (y1 + height / 2 - dy) / scale
        result.append(
            (
                int(cx), int(cy), int(width / scale), int(height / scale),
                float(selected_scores[int(index)]),
            )
        )
    return result


# ======================== 红色球门线检测 ========================
RED_HSV_LOW_1 = np.array([0, 65, 35], dtype=np.uint8)
RED_HSV_HIGH_1 = np.array([12, 255, 255], dtype=np.uint8)
RED_HSV_LOW_2 = np.array([168, 65, 35], dtype=np.uint8)
RED_HSV_HIGH_2 = np.array([180, 255, 255], dtype=np.uint8)
RED_LINE_BAND_PX = 70
RED_LINE_MIN_POINTS = 25
RED_LINE_MIN_SPAN_PX = 35


def _line_from_points(x1, y1, x2, y2, image_width):
    """构造与 goal_line_judge 兼容的无限延长线。"""
    dx = float(x2) - float(x1)
    dy = float(y2) - float(y1)
    norm = float(np.hypot(dx, dy))
    if norm < 1.0 or abs(dx) < 1e-9:
        return None
    left_y = float(y1) + (0.0 - float(x1)) * dy / dx
    right_x = float(image_width - 1)
    right_y = float(y1) + (right_x - float(x1)) * dy / dx
    return {
        "p1": (round(float(x1)), round(float(y1))),
        "p2": (round(float(x2)), round(float(y2))),
        "extended_p1": (0, round(left_y)),
        "extended_p2": (image_width - 1, round(right_y)),
        "x1": float(x1),
        "y1": float(y1),
        "dx": dx,
        "dy": dy,
        "norm": norm,
    }


def _smooth_red_line(previous, current, alpha=0.35):
    """对红线斜率和截距做简单 EMA，降低单帧红色噪声导致的抖动。"""
    if current is None:
        return previous
    if previous is None:
        return current

    current_slope = current["dy"] / current["dx"]
    current_intercept = current["y1"] - current_slope * current["x1"]
    previous_slope = previous["dy"] / previous["dx"]
    previous_intercept = previous["y1"] - previous_slope * previous["x1"]

    # 机器人停在球门前时，红线角度不应突然跳变；异常结果直接丢弃。
    if abs(current_slope - previous_slope) > 0.45:
        return previous

    slope = alpha * current_slope + (1.0 - alpha) * previous_slope
    intercept = alpha * current_intercept + (1.0 - alpha) * previous_intercept
    return _line_from_points(
        0,
        intercept,
        IMG_W - 1,
        slope * (IMG_W - 1) + intercept,
        IMG_W,
    )


def detect_red_goal_line(frame, goalposts):
    """
    在球门柱附近直接检测红色球门线。

    返回：
        line: 与 goal_line_judge 兼容的直线字典；失败时为 None
        pixel_count: 参与拟合的红色像素数
    """
    expected = _estimate_goal_line(goalposts, IMG_W)
    if expected is None:
        return None, 0

    post_x = sorted([float(post[0]) for post in goalposts if len(post) >= 4])
    if len(post_x) < 2:
        return None, 0

    x_min = max(0, int(post_x[0] - 55))
    x_max = min(IMG_W - 1, int(post_x[-1] + 55))
    if x_max - x_min < RED_LINE_MIN_SPAN_PX:
        return None, 0

    # 只在预计球门线附近取样，避免把场地其它红线当成球门线。
    y_expected_at_xmin = expected["y1"] + (x_min - expected["x1"]) * expected["dy"] / expected["dx"]
    y_expected_at_xmax = expected["y1"] + (x_max - expected["x1"]) * expected["dy"] / expected["dx"]
    y_min = max(0, int(min(y_expected_at_xmin, y_expected_at_xmax) - RED_LINE_BAND_PX))
    y_max = min(IMG_H - 1, int(max(y_expected_at_xmin, y_expected_at_xmax) + RED_LINE_BAND_PX))
    if y_max <= y_min:
        return None, 0

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, RED_HSV_LOW_1, RED_HSV_HIGH_1)
    mask |= cv2.inRange(hsv, RED_HSV_LOW_2, RED_HSV_HIGH_2)
    roi = mask[y_min:y_max + 1, x_min:x_max + 1]
    kernel = np.ones((3, 3), np.uint8)
    roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel, iterations=1)
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel, iterations=1)

    local_y, local_x = np.where(roi > 0)
    if len(local_x) < RED_LINE_MIN_POINTS:
        return None, int(len(local_x))

    xs = local_x.astype(np.float64) + x_min
    ys = local_y.astype(np.float64) + y_min
    expected_y = expected["y1"] + (xs - expected["x1"]) * expected["dy"] / expected["dx"]
    near_expected = np.abs(ys - expected_y) <= RED_LINE_BAND_PX
    xs = xs[near_expected]
    ys = ys[near_expected]
    if len(xs) < RED_LINE_MIN_POINTS or float(xs.max() - xs.min()) < RED_LINE_MIN_SPAN_PX:
        return None, int(len(xs))

    # 迭代剔除红色噪点后拟合 y = slope*x + intercept。
    keep = np.ones(len(xs), dtype=bool)
    for _ in range(3):
        if int(keep.sum()) < RED_LINE_MIN_POINTS:
            return None, int(keep.sum())
        slope, intercept = np.polyfit(xs[keep], ys[keep], 1)
        residual = np.abs(ys - (slope * xs + intercept))
        threshold = max(5.0, float(np.percentile(residual[keep], 70)) * 1.8)
        keep = residual <= threshold

    if int(keep.sum()) < RED_LINE_MIN_POINTS:
        return None, int(keep.sum())

    slope, intercept = np.polyfit(xs[keep], ys[keep], 1)
    line = _line_from_points(
        float(xs[keep].min()),
        slope * float(xs[keep].min()) + intercept,
        float(xs[keep].max()),
        slope * float(xs[keep].max()) + intercept,
        IMG_W,
    )
    return line, int(keep.sum()) if line is not None else 0


# ======================== 绘图与状态 ========================
def _safe_point(point):
    return int(point[0]), int(point[1])


def draw_result(frame, ball, goalposts, result):
    display = frame.copy()

    for index, (cx, cy, width, height, confidence) in enumerate(goalposts):
        x1, y1 = cx - width // 2, cy - height // 2
        x2, y2 = cx + width // 2, cy + height // 2
        cv2.rectangle(display, (x1, y1), (x2, y2), (255, 120, 0), 2)
        cv2.putText(
            display,
            f"Post#{index + 1} {confidence:.2f}",
            (x1, max(16, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 120, 0),
            2,
        )

    line = result.get("line")
    if line is not None:
        if result.get("line_source") == "red_pixels":
            line_color = (0, 255, 0)
        else:
            line_color = (0, 165, 255) if result.get("confirmed") else (0, 0, 255)
        cv2.line(
            display,
            _safe_point(line["extended_p1"]),
            _safe_point(line["extended_p2"]),
            line_color,
            3,
        )
        cv2.putText(
            display,
            f"Goal line / extension ({result.get('line_source', 'unknown')})",
            (8, max(18, min(IMG_H - 8, line["extended_p1"][1] - 8))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            line_color,
            2,
        )

    if ball is not None:
        cx, cy, width, height, confidence = ball
        x1, y1 = cx - width // 2, cy - height // 2
        x2, y2 = cx + width // 2, cy + height // 2
        if result.get("confirmed"):
            ball_color = (0, 165, 255)
        elif result.get("whole_crossed"):
            ball_color = (0, 255, 255)
        else:
            ball_color = (0, 255, 0)
        cv2.rectangle(display, (x1, y1), (x2, y2), ball_color, 3)
        cv2.circle(display, (cx, cy), 4, ball_color, -1)
        cv2.putText(
            display,
            f"Ball {confidence:.2f}",
            (x1, max(16, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            ball_color,
            2,
        )

    if not result.get("line_valid"):
        status_text = "Need 2 goalposts"
        status_color = (0, 200, 255)
    elif result.get("confirmed"):
        status_text = "WHOLE BBOX CROSSED / CONFIRMED"
        status_color = (0, 165, 255)
    elif result.get("whole_crossed"):
        status_text = f"Whole crossing candidate {result.get('streak', 0)}"
        status_color = (0, 255, 255)
    elif result.get("partial_crossed"):
        status_text = "PARTIAL CROSSING"
        status_color = (0, 255, 255)
    elif ball is None:
        status_text = "Ball not detected"
        status_color = (180, 180, 180)
    else:
        status_text = "Not crossed"
        status_color = (100, 220, 100)

    cv2.rectangle(display, (0, 0), (IMG_W, 34), (25, 25, 25), -1)
    cv2.putText(
        display,
        status_text,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        status_color,
        2,
    )
    return display


def _json_ball(ball):
    if ball is None:
        return None
    cx, cy, width, height, confidence = ball
    return {
        "cx": cx,
        "cy": cy,
        "w": width,
        "h": height,
        "conf": round(confidence * 100, 1),
    }


def _json_posts(posts):
    return [
        {
            "cx": post[0],
            "cy": post[1],
            "w": post[2],
            "h": post[3],
            "conf": round(post[4] * 100, 1),
        }
        for post in posts
    ]


def camera_thread():
    global latest_jpeg, camera_ready, current_streak, current_status
    global goal_event_latched, smoothed_red_line

    camera = Camera.Camera()
    goalposts = []
    smoothed_red_line = None
    frame_index = 0
    fps_started_at = time.monotonic()
    processed_frames = 0
    processing_fps = 0.0
    try:
        camera.camera_open()
        camera_ready = True
        print(f"摄像头已开启，请访问 http://<机器人IP>:{WEB_PORT}")

        while not stop_event.is_set():
            ret, raw_frame = camera.read()
            if not ret or raw_frame is None:
                time.sleep(0.01)
                continue

            frame = cv2.remap(raw_frame, mapx, mapy, cv2.INTER_LINEAR)
            try:
                # 球门柱在测试过程中基本不动，不必每帧重新推理；
                # 把计算量留给足球检测，可以减少漏掉快速越线瞬间的概率。
                if frame_index % GOALPOST_REFRESH_INTERVAL == 0 or not goalposts:
                    goalposts = detect_goalposts(frame)
                ball = detect_football(frame)
                red_line_raw, red_pixel_count = detect_red_goal_line(frame, goalposts)
                if red_line_raw is not None:
                    smoothed_red_line = _smooth_red_line(smoothed_red_line, red_line_raw)
                with frame_lock:
                    config = settings.copy()
                    previous_streak = current_streak
                    latched = goal_event_latched

                if smoothed_red_line is not None:
                    line_for_judge = smoothed_red_line
                    line_source = "red_pixels"
                else:
                    line_for_judge = None
                    line_source = "goalposts_fallback"

                result = check_ball_crossed_goal_line(
                    ball,
                    goalposts,
                    image_width=IMG_W,
                    goal_line=line_for_judge,
                    goal_line_source=line_source,
                    goal_side=config["goal_side"],
                    margin_px=config["margin_px"],
                    previous_streak=previous_streak,
                    confirm_frames=config["confirm_frames"],
                )

                # 任务标准：足球检测框四个角全部越过球门线，才认为射门成功。
                # 快速反弹时，可能只在一帧看到 whole_crossed，因此锁存任务成功。
                # partial_crossed 仅保留为调试信息，不触发任务完成。
                if result["whole_crossed"]:
                    latched = True

                display = draw_result(frame, ball, goalposts, result)
                if latched:
                    cv2.rectangle(display, (0, 34), (IMG_W, 68), (25, 95, 45), -1)
                    cv2.putText(
                        display,
                        "TASK SUCCESS: PARTIAL BBOX CROSSED / LATCHED",
                        (8, 58),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (120, 255, 150),
                        2,
                    )
                processed_frames += 1
                elapsed = time.monotonic() - fps_started_at
                if elapsed >= 1.0:
                    processing_fps = processed_frames / elapsed
                    processed_frames = 0
                    fps_started_at = time.monotonic()
                cv2.putText(
                    display,
                    f"Process FPS: {processing_fps:.1f}",
                    (IMG_W - 190, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (220, 220, 220),
                    2,
                )

                status = {
                    "camera_ready": True,
                    "ball": _json_ball(ball),
                    "goalposts": _json_posts(goalposts),
                    "line_valid": result["line_valid"],
                    "partial_crossed": result["partial_crossed"],
                    "whole_crossed": result["whole_crossed"],
                    "confirmed": result["confirmed"],
                    "streak": result["streak"],
                    "line": result["line"],
                    "line_source": result["line_source"],
                    "red_line_valid": smoothed_red_line is not None,
                    "red_pixel_count": red_pixel_count,
                    "bbox": result["bbox"],
                    "signed_distances": [round(v, 1) for v in result["signed_distances"]],
                    "goal_side": config["goal_side"],
                    "margin_px": config["margin_px"],
                    "confirm_frames": config["confirm_frames"],
                    "processing_fps": round(processing_fps, 1),
                    "goal_event_latched": latched,
                    "task_success": latched,
                    "last_error": "",
                    "updated_at": time.time(),
                }
                encoded_ok, buffer = cv2.imencode(
                    ".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, STREAM_QUALITY]
                )
                if not encoded_ok:
                    continue

                with frame_lock:
                    current_streak = result["streak"]
                    goal_event_latched = latched
                    current_status = status
                    latest_jpeg = buffer.tobytes()
                frame_index += 1
            except Exception as exc:
                with frame_lock:
                    current_status["last_error"] = repr(exc)
                time.sleep(0.1)

            time.sleep(FRAME_INTERVAL)
    finally:
        camera_ready = False
        try:
            camera.camera_close()
        except Exception:
            pass


def generate_mjpeg():
    while not stop_event.is_set():
        with frame_lock:
            jpeg = latest_jpeg
        if jpeg is not None:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(FRAME_INTERVAL)


# ======================== 网页 ========================
HTML_PAGE = r"""
<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TonyPi 球门线越线测试</title>
<style>
body { background:#151923; color:#e9edf5; font-family:Arial,sans-serif; margin:0; padding:18px; }
main { max-width:1100px; margin:auto; }
h1 { margin:0 0 8px; font-size:24px; }
.note { color:#aeb8c8; margin-bottom:14px; }
.grid { display:grid; grid-template-columns:minmax(0, 2fr) minmax(280px, 1fr); gap:16px; }
.card { background:#202634; border-radius:10px; padding:14px; box-shadow:0 2px 10px #0005; }
img { width:100%; background:#000; border-radius:6px; display:block; }
.row { display:flex; justify-content:space-between; gap:10px; padding:8px 0; border-bottom:1px solid #343c4d; }
.row:last-child { border-bottom:0; }
.label { color:#aeb8c8; }
.value { font-weight:bold; text-align:right; }
.ok { color:#57df8a; } .warn { color:#ffd166; } .bad { color:#ff7070; }
.controls { display:flex; flex-wrap:wrap; align-items:end; gap:10px; margin-top:12px; }
label { color:#aeb8c8; font-size:13px; display:flex; flex-direction:column; gap:4px; }
select,input,button { background:#30394c; border:1px solid #4a5871; color:#fff; border-radius:5px; padding:7px 9px; }
button { cursor:pointer; } button:hover { background:#43516b; }
pre { white-space:pre-wrap; word-break:break-word; color:#bac4d4; font-size:12px; }
@media(max-width:760px) { .grid { grid-template-columns:1fr; } }
</style>
</head>
<body>
<main>
<h1>⚽ TonyPi 球门线越线判断测试</h1>
<div class="note">此页面只运行视觉检测和独立判断函数，不会控制机器人行走或踢球。</div>
<div class="grid">
  <div class="card">
    <img src="/video_feed" alt="camera stream">
    <div class="controls">
      <label>球门内部方向
        <select id="goal_side"><option value="above">图像上方</option><option value="below">图像下方</option></select>
      </label>
      <label>离线安全距离(px)<input id="margin_px" type="number" min="0" step="0.5" value="3"></label>
      <label>确认帧数<input id="confirm_frames" type="number" min="1" step="1" value="1"></label>
      <button onclick="applyConfig()">应用参数</button>
      <button onclick="resetJudge()">清除连续帧</button>
    </div>
  </div>
  <div class="card">
    <div class="row"><span class="label">摄像头</span><span id="camera" class="value">--</span></div>
    <div class="row"><span class="label">足球</span><span id="ball" class="value">--</span></div>
    <div class="row"><span class="label">球门柱数量</span><span id="posts" class="value">--</span></div>
    <div class="row"><span class="label">球门线</span><span id="line" class="value">--</span></div>
    <div class="row"><span class="label">球门线来源</span><span id="line_source" class="value">--</span></div>
    <div class="row"><span class="label">红色拟合像素</span><span id="red_pixels" class="value">--</span></div>
    <div class="row"><span class="label">部分越线</span><span id="partial" class="value">--</span></div>
    <div class="row"><span class="label">整体越线</span><span id="whole" class="value">--</span></div>
    <div class="row"><span class="label">连续确认</span><span id="streak" class="value">--</span></div>
    <div class="row"><span class="label">整体越线确认</span><span id="confirmed" class="value">--</span></div>
    <div class="row"><span class="label">任务成功（完全越线）</span><span id="task_success" class="value">--</span></div>
    <div class="row"><span class="label">后端处理帧率</span><span id="fps" class="value">--</span></div>
    <div class="row"><span class="label">四角带符号距离</span><span id="distances" class="value">--</span></div>
    <div class="row"><span class="label">错误</span><span id="error" class="value">--</span></div>
    <h3>调试数据</h3>
    <pre id="raw">--</pre>
  </div>
</div>
</main>
<script>
function setValue(id, text, cls='value') {
  const e = document.getElementById(id); e.textContent = text; e.className = cls;
}
function yesNo(v) { return v ? '✅ 是' : '❌ 否'; }
function update() {
  fetch('/status').then(r => r.json()).then(d => {
    setValue('camera', d.camera_ready ? '✅ 已连接' : '❌ 未连接', d.camera_ready ? 'value ok' : 'value bad');
    setValue('ball', d.ball ? `✅ (${d.ball.cx},${d.ball.cy}) ${d.ball.w}×${d.ball.h}, ${d.ball.conf}%` : '❌ 未检测', d.ball ? 'value ok' : 'value bad');
    setValue('posts', `${(d.goalposts || []).length}`, (d.goalposts || []).length >= 2 ? 'value ok' : 'value warn');
    setValue('line', d.line_valid ? '✅ 已估计' : '❌ 需要两个柱子', d.line_valid ? 'value ok' : 'value warn');
    setValue('line_source', d.line_source || '--', d.line_source === 'red_pixels' ? 'value ok' : 'value warn');
    setValue('red_pixels', `${d.red_pixel_count || 0}`, d.red_line_valid ? 'value ok' : 'value warn');
    setValue('partial', yesNo(d.partial_crossed), d.partial_crossed ? 'value warn' : 'value');
    setValue('whole', yesNo(d.whole_crossed), d.whole_crossed ? 'value warn' : 'value');
    setValue('streak', `${d.streak} / ${d.confirm_frames}`, d.streak ? 'value warn' : 'value');
    setValue('confirmed', yesNo(d.confirmed), d.confirmed ? 'value ok' : 'value');
    setValue('task_success', yesNo(d.task_success), d.task_success ? 'value ok' : 'value');
    setValue('fps', `${d.processing_fps || 0} FPS`);
    setValue('distances', JSON.stringify(d.signed_distances || []));
    setValue('error', d.last_error || '--', d.last_error ? 'value bad' : 'value');
    document.getElementById('raw').textContent = JSON.stringify(d, null, 2);
  }).catch(() => setValue('error', '无法读取 /status', 'value bad'));
}
function applyConfig() {
  const body = {
    goal_side: document.getElementById('goal_side').value,
    margin_px: Number(document.getElementById('margin_px').value),
    confirm_frames: Number(document.getElementById('confirm_frames').value)
  };
  fetch('/config', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)})
    .then(() => update());
}
function resetJudge() { fetch('/reset').then(() => update()); }
setInterval(update, 300); update();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/status")
def status():
    with frame_lock:
        return jsonify(current_status)


@app.route("/reset", methods=["GET", "POST"])
def reset_judge():
    global current_streak, goal_event_latched
    with frame_lock:
        current_streak = 0
        goal_event_latched = False
        current_status["goal_event_latched"] = False
        current_status["task_success"] = False
        current_status["confirmed"] = False
    return jsonify({"success": True})


@app.route("/config", methods=["POST"])
def configure():
    payload = request.get_json(silent=True) or request.form
    goal_side = str(payload.get("goal_side", "above"))
    try:
        margin_px = float(payload.get("margin_px", 3.0))
        confirm_frames = int(payload.get("confirm_frames", 1))
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "参数格式错误"}), 400

    if goal_side not in ("above", "below"):
        return jsonify({"success": False, "error": "goal_side 只能是 above 或 below"}), 400
    if margin_px < 0 or confirm_frames < 1:
        return jsonify({"success": False, "error": "margin_px >= 0 且 confirm_frames >= 1"}), 400

    global current_streak, goal_event_latched
    with frame_lock:
        settings.update({
            "goal_side": goal_side,
            "margin_px": margin_px,
            "confirm_frames": confirm_frames,
        })
        current_streak = 0
        goal_event_latched = False
    return jsonify({"success": True, **settings})


if __name__ == "__main__":
    print("=" * 60)
    print("TonyPi 独立球门线越线判断测试网页")
    print(f"浏览器访问: http://<机器人IP>:{WEB_PORT}")
    print("本程序不会控制机器人运动或执行踢球动作")
    print("按 Ctrl+C 退出")
    print("=" * 60)

    worker = threading.Thread(target=camera_thread, daemon=True)
    worker.start()
    try:
        app.run(host="0.0.0.0", port=WEB_PORT, threaded=True, use_reloader=False)
    finally:
        stop_event.set()

