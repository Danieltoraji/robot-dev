#!/usr/bin/python3
# coding=utf8
"""
web_distance.py — YOLOv5 足球检测 + 测距 Web 可视化界面

基于 football_web.py 架构，增加摄像头测距功能

功能：
  在浏览器中实时显示摄像头画面 + YOLOv5 足球检测 + 距离测量

运行方式：
  sudo systemctl stop tonypi
  python3 web_distance.py

  然后浏览器访问：http://192.168.31.203:5001

模型文件：models/football_best_win.onnx
"""

import sys
import os
import cv2
import time
import threading
import numpy as np
from flask import Flask, Response, render_template_string

import hiwonder.Camera as Camera
import hiwonder.yaml_handle as yaml_handle

if __name__ == '__main__':
    from CameraCalibration.CalibrationConfig import *
else:
    from Functions.CameraCalibration.CalibrationConfig import *

# ============ 可调参数 ============
# --- YOLO 推理参数 ---
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'football_best_win.onnx')
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
INPUT_SIZE = 640

# --- 测距参数 ---
BALL_DIAMETER = 6.3             # 足球真实直径（cm）

# --- Web 服务参数 ---
WEB_PORT = 5001
STREAM_QUALITY = 70
FRAME_INTERVAL = 0.05
# ==================================

if sys.version_info.major == 2:
    print('Please run this program with python3!')
    sys.exit(0)

# 加载相机标定参数
param_data = np.load(calibration_param_path + '.npz')
mtx = param_data['mtx_array']
dist = param_data['dist_array']
fx = mtx[0, 0]  # 焦距
newcameramtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (640, 480), 0, (640, 480))
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (640, 480), 5)

print(f"加载 YOLO 模型: {MODEL_PATH}")
net = cv2.dnn.readNetFromONNX(MODEL_PATH)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
print("YOLO 模型加载完成")

app = Flask(__name__)

# ============ 全局状态 ============
latest_raw_frame = None
latest_frame = None
detection_info = {
    'detected': False,
    'cx': 0, 'cy': 0, 'w': 0, 'h': 0,
    'confidence': 0.0,
    'distance_cm': -1.0,
    'infer_ms': 0.0,
    'last_error': '',
}
frame_lock = threading.Lock()
camera_frames = 0
detect_frames = 0
last_detect_time = 0.0
last_error = ''


def detect_football_yolo(frame):
    """用 YOLOv5 ONNX 检测足球，返回 (cx, cy, w, h, conf) 或 None"""
    img_h, img_w = frame.shape[:2]

    scale = min(INPUT_SIZE / img_w, INPUT_SIZE / img_h)
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)
    resized = cv2.resize(frame, (new_w, new_h))

    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    dx = (INPUT_SIZE - new_w) // 2
    dy = (INPUT_SIZE - new_h) // 2
    canvas[dy:dy+new_h, dx:dx+new_w] = resized

    blob = cv2.dnn.blobFromImage(canvas, 1.0/255.0, (INPUT_SIZE, INPUT_SIZE),
                                  swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    outputs = net.forward()
    infer_ms = (time.time() - start) * 1000

    predictions = outputs[0]
    obj_conf = predictions[:, 4]
    cls_conf = predictions[:, 5]
    scores = obj_conf * cls_conf

    mask = scores > CONF_THRESHOLD
    if not np.any(mask):
        return None, infer_ms

    filtered = predictions[mask]
    filtered_scores = scores[mask]

    boxes = []
    for det in filtered:
        cx, cy, w, h = det[0], det[1], det[2], det[3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        boxes.append([x1, y1, w, h])

    boxes = np.array(boxes)
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), filtered_scores.tolist(),
                                CONF_THRESHOLD, IOU_THRESHOLD)

    if len(indices) == 0:
        return None, infer_ms

    idx = int(np.array(indices).flatten()[0])
    best_box = boxes[idx]
    best_score = filtered_scores[idx]

    x1, y1, w, h = best_box
    cx = x1 + w / 2
    cy = y1 + h / 2

    cx = (cx - dx) / scale
    cy = (cy - dy) / scale
    w = w / scale
    h = h / scale

    return (int(cx), int(cy), int(w), int(h), float(best_score)), infer_ms


def calc_distance(w, h):
    """小孔成像测距：distance = (real_diameter * fx) / pixel_diameter"""
    pixel_diameter = max(w, h)
    if pixel_diameter > 0:
        return (BALL_DIAMETER * fx) / pixel_diameter
    return -1


def draw_detection(display, cx, cy, w, h, confidence, distance_cm):
    """在画面上绘制检测框和距离"""
    if confidence >= 0.8:
        color = (0, 255, 0)
    elif confidence >= 0.6:
        color = (0, 255, 255)
    else:
        color = (0, 128, 255)

    x1 = cx - w // 2
    y1 = cy - h // 2
    x2 = cx + w // 2
    y2 = cy + h // 2
    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
    cv2.circle(display, (cx, cy), 3, color, -1)

    cv2.putText(display, f"Football {confidence*100:.0f}%", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 距离显示
    if distance_cm > 0:
        dist_text = f"Distance: {distance_cm:.1f} cm"
        if distance_cm < 30:
            dist_color = (0, 255, 255)
        elif distance_cm < 60:
            dist_color = (0, 255, 0)
        else:
            dist_color = (255, 200, 0)
        cv2.putText(display, dist_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, dist_color, 2)

    return display


# ============ 线程函数 ============

def camera_thread():
    """摄像头采集 + 画面叠加"""
    global latest_raw_frame, latest_frame, camera_frames, last_error

    cap = None
    # 依次尝试打开摄像头
    for idx in [0, 1, 2, -1]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, test = cap.read()
            if ret and test is not None:
                print(f"  摄像头已打开 (index={idx})")
                break
            cap.release()
        cap = None

    if cap is None:
        print("  cv2.VideoCapture 失败，尝试 Camera 模块...")
        try:
            cap = Camera.Camera()
            cap.camera_open()
            print("  摄像头已打开 (Camera 模块)")
        except:
            print("  无法打开摄像头！")
            return

    while True:
        try:
            if isinstance(cap, Camera.Camera):
                ret, frame = cap.read()
            else:
                ret, frame = cap.read()
        except:
            ret = False

        if not ret or frame is None:
            time.sleep(0.01)
            continue

        camera_frames += 1

        # 去畸变
        frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

        display = frame.copy()

        with frame_lock:
            info = detection_info.copy()

        if info['detected']:
            display = draw_detection(display, info['cx'], info['cy'],
                                     info['w'], info['h'],
                                     info['confidence'] / 100.0,
                                     info['distance_cm'])
        else:
            cv2.putText(display, "No football detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)

        with frame_lock:
            latest_raw_frame = frame
            latest_frame = display

        time.sleep(FRAME_INTERVAL)


def detection_thread():
    """YOLO 检测 + 测距"""
    global detection_info, detect_frames, last_detect_time, last_error

    while True:
        try:
            with frame_lock:
                frame = latest_raw_frame

            if frame is None:
                time.sleep(0.05)
                continue

            result, infer_ms = detect_football_yolo(frame)
            detect_frames += 1
            last_detect_time = time.time()

            with frame_lock:
                if result is not None:
                    cx, cy, w, h, confidence = result
                    distance_cm = round(calc_distance(w, h), 1)
                    detection_info = {
                        'detected': True,
                        'cx': cx, 'cy': cy, 'w': w, 'h': h,
                        'confidence': round(confidence * 100, 1),
                        'distance_cm': distance_cm,
                        'infer_ms': round(infer_ms, 1),
                        'last_error': last_error,
                    }
                else:
                    detection_info = {
                        'detected': False,
                        'cx': 0, 'cy': 0, 'w': 0, 'h': 0,
                        'confidence': 0.0,
                        'distance_cm': -1.0,
                        'infer_ms': round(infer_ms, 1),
                        'last_error': last_error,
                    }
        except Exception as e:
            with frame_lock:
                last_error = repr(e)

        time.sleep(0.05)


def generate_mjpeg():
    """MJPEG 流生成器"""
    while True:
        with frame_lock:
            frame = latest_frame

        if frame is None:
            time.sleep(0.05)
            continue

        _, buffer = cv2.imencode('.jpg', frame,
                                 [cv2.IMWRITE_JPEG_QUALITY, STREAM_QUALITY])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(FRAME_INTERVAL)


# ============ Web 路由 ============

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TonyPi 足球测距 (YOLOv5)</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
        }
        h1 { font-size: 1.5em; margin-bottom: 15px; color: #00d4ff; }
        .container {
            display: flex; flex-wrap: wrap; gap: 20px;
            justify-content: center; max-width: 1200px; width: 100%;
        }
        .video-box {
            background: #16213e; border-radius: 12px; padding: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }
        .video-box img {
            width: 640px; max-width: 100%; height: auto; border-radius: 8px;
        }
        .info-panel {
            background: #16213e; border-radius: 12px; padding: 20px;
            min-width: 280px; box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }
        .info-panel h2 {
            font-size: 1.1em; color: #00d4ff; margin-bottom: 15px;
            border-bottom: 1px solid #333; padding-bottom: 8px;
        }
        .distance-box {
            font-size: 2em; font-weight: bold; text-align: center;
            padding: 15px; border-radius: 12px; margin-bottom: 15px;
        }
        .distance-box.far { background: #1a2a4e; color: #ffc800; }
        .distance-box.mid { background: #0a2a1e; color: #00ff88; }
        .distance-box.near { background: #2a2a0a; color: #ffcc00; }
        .distance-box.none { background: #2a1a1a; color: #ff6666; }

        .status { font-size: 1.1em; padding: 8px; border-radius: 8px; text-align: center; margin-bottom: 15px; }
        .status.detected { background: #0a3d0a; color: #4cff4c; }
        .status.not-detected { background: #3d0a0a; color: #ff6b6b; }

        .metric { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #222; }
        .metric-label { color: #aaa; }
        .metric-value { font-weight: bold; }
        .model-badge { display: inline-block; background: #0a3d3d; color: #4cffff; padding: 3px 10px; border-radius: 12px; font-size: 0.75em; margin-bottom: 12px; }
    </style>
</head>
<body>
    <h1>⚽ TonyPi 足球测距</h1>
    <div class="container">
        <div class="video-box">
            <img src="/video_feed" alt="Live Feed">
        </div>
        <div class="info-panel">
            <h2>检测 & 测距</h2>
            <span class="model-badge">YOLOv5s + 单目测距</span>

            <div id="distance-box" class="distance-box none">⚪ 等待检测...</div>
            <div id="status" class="status not-detected">No Football</div>

            <div class="metric">
                <span class="metric-label">置信度</span>
                <span class="metric-value" id="val-conf">--</span>
            </div>
            <div class="metric">
                <span class="metric-label">推理耗时</span>
                <span class="metric-value" id="val-infer">--</span>
            </div>
            <div class="metric">
                <span class="metric-label">像素直径</span>
                <span class="metric-value" id="val-diameter">--</span>
            </div>
            <div class="metric">
                <span class="metric-label">位置 (x, y)</span>
                <span class="metric-value" id="val-pos">--</span>
            </div>
            <div class="metric">
                <span class="metric-label">帧数</span>
                <span class="metric-value" id="val-frames">--</span>
            </div>
            <div class="metric">
                <span class="metric-label">参数</span>
                <span class="metric-value" id="val-params">--</span>
            </div>
        </div>
    </div>

    <script>
        function updateInfo() {
            fetch('/status')
                .then(r => r.json())
                .then(data => {
                    const statusEl = document.getElementById('status');
                    const distBox = document.getElementById('distance-box');

                    document.getElementById('val-infer').textContent = (data.infer_ms || 0).toFixed(1) + ' ms';
                    document.getElementById('val-frames').textContent = data.camera_frames + ' / ' + data.detect_frames;

                    if (data.detected) {
                        statusEl.textContent = 'Football Detected!';
                        statusEl.className = 'status detected';
                        document.getElementById('val-conf').textContent = data.confidence.toFixed(1) + '%';
                        document.getElementById('val-pos').textContent = '(' + data.cx + ', ' + data.cy + ')';
                        document.getElementById('val-diameter').textContent = Math.max(data.w, data.h) + ' px';
                        document.getElementById('val-params').textContent = data.ball_diameter + 'cm × ' + data.focal.toFixed(0) + 'px';

                        if (data.distance_cm > 0) {
                            var d = data.distance_cm;
                            distBox.textContent = d.toFixed(1) + ' cm';
                            distBox.className = 'distance-box';
                            if (d < 30) distBox.className += ' near';
                            else if (d < 60) distBox.className += ' mid';
                            else distBox.className += ' far';
                        } else {
                            distBox.textContent = '⚪ 计算中...';
                            distBox.className = 'distance-box none';
                        }
                    } else {
                        statusEl.textContent = 'No Football';
                        statusEl.className = 'status not-detected';
                        distBox.textContent = '⚪ 未检测到足球';
                        distBox.className = 'distance-box none';
                        document.getElementById('val-conf').textContent = '--';
                        document.getElementById('val-pos').textContent = '--';
                        document.getElementById('val-diameter').textContent = '--';
                    }
                })
                .catch(function() {});
        }
        updateInfo();
        setInterval(updateInfo, 500);
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_PAGE)


@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    import json
    with frame_lock:
        info = detection_info.copy()
        info['camera_frames'] = camera_frames
        info['detect_frames'] = detect_frames
        info['ball_diameter'] = BALL_DIAMETER
        info['focal'] = round(fx, 1)
        info['last_error'] = last_error
    return json.dumps(info), 200, {'Content-Type': 'application/json'}


if __name__ == '__main__':
    print("=" * 50)
    print("TonyPi 足球测距 Web 服务 (YOLOv5 + 单目测距)")
    print("=" * 50)
    print(f"模型: {MODEL_PATH}")
    print(f"足球直径: {BALL_DIAMETER} cm")
    print(f"焦距: {fx:.1f} px")
    print(f"端口: {WEB_PORT}")
    print(f"浏览器打开: http://192.168.31.203:{WEB_PORT}")
    print("按 Ctrl+C 退出")
    print("=" * 50)

    cam_thread = threading.Thread(target=camera_thread, daemon=True)
    cam_thread.start()
    det_thread = threading.Thread(target=detection_thread, daemon=True)
    det_thread.start()

    app.run(host='0.0.0.0', port=WEB_PORT, threaded=True, use_reloader=False)
