#!/usr/bin/python3
# coding=utf8
"""
goal_web.py — 足球越线检测 Web 可视化

功能：
  在浏览器中实时显示：
  - 足球检测（YOLO 绿色框）
  - 红色球门线检测（颜色阈值）
  - 头部扫视状态（是否在低头/左右找线）
  - 越线判定结果（本次射门是否已结束）

运行方式：
  sudo systemctl stop tonypi
  python3 goal_web.py

  浏览器访问：http://<机器人IP>:5001
"""

import os
import sys
import time
import threading
import cv2
import numpy as np
from flask import Flask, Response, render_template_string, jsonify

import hiwonder.Camera as Camera
import hiwonder.yaml_handle as yaml_handle

# 将 TonyPi 根目录加入路径，确保导入正常工作
_GOAL_WEB_DIR = os.path.dirname(os.path.abspath(__file__))
_TONYPY_DIR = os.path.dirname(_GOAL_WEB_DIR)  # Functions/ 的上级
if _TONYPY_DIR not in sys.path:
    sys.path.insert(0, _TONYPY_DIR)

from Functions.CameraCalibration.CalibrationConfig import *

# ============ 导入越线检测模块 ============
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from goal_detector import GoalDetector

# ============ 可调参数 ============
WEB_PORT = 5001
STREAM_QUALITY = 70
FRAME_INTERVAL = 0.05
# ==================================

if sys.version_info.major == 2:
    print("Please run this program with python3!")
    sys.exit(0)

# 加载标定参数（去畸变用）
param_data = np.load(calibration_param_path + ".npz")
mtx = param_data["mtx_array"]
dist = param_data["dist_array"]
newcameramtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (640, 480), 0, (640, 480))
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (640, 480), 5)

app = Flask(__name__)

# ============ 全局状态 ============
latest_frame = None  # 带检测标注的画面
latest_jpeg = None
frame_lock = threading.Lock()

# 检测器
detector = GoalDetector()

# 检测结果（给 JSON API 用）
current_result = {
    "shot_resolved": False,
    "status": "等待检测",
    "football": None,
    "goal_line": None,
    "crossed_line": False,
    "scan_state": "idle",
}

camera_ready = False


def camera_and_detect_thread():
    """单线程：采集 → 去畸变 → 检测 → 绘图 → 推流"""
    global latest_frame, latest_jpeg, current_result, camera_ready

    cam = Camera.Camera()
    cam.camera_open()
    camera_ready = True
    time.sleep(1.0)

    print(f"摄像头已开启，访问 http://<机器人IP>:{WEB_PORT}")

    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        # 去畸变
        frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

        # 越线检测 + 绘图
        result = detector.check_goal(frame)
        display = detector.draw_result(frame, result)

        # 更新全局状态
        with frame_lock:
            latest_frame = display.copy()
            current_result = result

        # JPEG 压缩
        _, buffer = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, STREAM_QUALITY])
        with frame_lock:
            latest_jpeg = buffer.tobytes()

        time.sleep(FRAME_INTERVAL)


def generate_mjpeg():
    """MJPEG 流"""
    while True:
        with frame_lock:
            jpeg = latest_jpeg
        if jpeg is None:
            time.sleep(0.05)
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
        time.sleep(FRAME_INTERVAL)


# ============ Web 路由 ============

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TonyPi 越线检测</title>
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
            display: block;
        }
        .info-panel {
            background: #16213e; border-radius: 12px; padding: 20px;
            min-width: 280px; box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            display: flex; flex-direction: column; gap: 12px;
        }
        .info-panel h2 {
            font-size: 1.1em; color: #00d4ff; margin-bottom: 5px;
            border-bottom: 1px solid #333; padding-bottom: 8px;
        }

        .goal-badge {
            font-size: 1.8em;
            font-weight: bold;
            text-align: center;
            padding: 20px;
            border-radius: 12px;
            transition: all 0.3s;
        }
        .goal-badge.goal {
            background: #3d1a0a;
            color: #ff8833;
            animation: pulse 1s infinite;
        }
        .goal-badge.no-goal {
            background: #0a1a3d;
            color: #6699ff;
        }
        .goal-badge.searching {
            background: #1a1a1a;
            color: #888;
        }
        @keyframes pulse {
            0% { box-shadow: 0 0 10px #ff8833; }
            50% { box-shadow: 0 0 40px #ff6600; }
            100% { box-shadow: 0 0 10px #ff8833; }
        }

        .status-line { text-align: center; font-size: 1em; padding: 6px; border-radius: 6px; }

        .grid {
            display: grid; grid-template-columns: auto 1fr; gap: 4px 12px;
            font-size: 0.85em;
        }
        .grid .label { color: #888; }
        .grid .value { color: #ddd; }
        .grid .value.ok { color: #4cff4c; }
        .grid .value.fail { color: #ff6b6b; }

        .legend {
            font-size: 0.82em; color: #aaa; padding: 8px;
            background: #111; border-radius: 6px;
        }
        .legend span { margin-right: 12px; }
        .dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 4px; }
        .dot-green { background: #4cff4c; }
        .dot-red { background: #ff3333; }
        .dot-orange { background: #ff8833; }

        .btn {
            padding: 12px 20px; border: none; border-radius: 8px;
            font-size: 1em; font-weight: bold; cursor: pointer;
            background: #00d4ff; color: #1a1a2e;
        }
        .btn:hover { background: #33ddff; }

        .fps { color: #555; font-size: 0.8em; text-align: center; }
    </style>
</head>
<body>
    <h1>⚽ TonyPi 越线检测</h1>
    <div class="container">
        <div class="video-box">
            <img src="/video_feed" id="video">
        </div>
        <div class="info-panel">
            <div id="goal-badge" class="goal-badge searching">⏳ 等待检测...</div>
            <div id="status-line" class="status-line">--</div>

            <button class="btn" onclick="resetSession()">🔄 重置本次射门（reset_session）</button>

            <h2>检测状态</h2>
            <div class="grid">
                <span class="label">⚽ 足球</span>
                <span class="value" id="val-football">--</span>
                <span class="label">🔴 球门线</span>
                <span class="value" id="val-line">--</span>
                <span class="value" id="val-dist">--</span>
                <span class="label">🚧 已过线</span>
                <span class="value" id="val-crossed">--</span>
                <span class="label">👀 扫视状态</span>
                <span class="value" id="val-scan">--</span>
            </div>

            <div class="legend">
                <span><span class="dot dot-green"></span>足球</span>
                <span><span class="dot dot-red"></span>球门线</span>
                <span><span class="dot dot-orange"></span>已结束</span>
            </div>
        </div>
    </div>

    <script>
        function resetSession() {
            fetch('/reset').then(r => r.json()).catch(() => {});
        }

        function updateStatus() {
            fetch('/status')
                .then(r => r.json())
                .then(d => {
                    const badge = document.getElementById('goal-badge');
                    const sl = document.getElementById('status-line');

                    if (d.shot_resolved) {
                        badge.textContent = '🚧 已越线，射门结束';
                        badge.className = 'goal-badge goal';
                        sl.textContent = d.status || '本次射门已结束';
                        sl.style.color = '#ff8833';
                    } else if (d.football) {
                        badge.textContent = '⚽ 跟踪中';
                        badge.className = 'goal-badge no-goal';
                        sl.textContent = d.status || '球未过线';
                        sl.style.color = '#6699ff';
                    } else {
                        badge.textContent = '⏳ 搜索中...';
                        badge.className = 'goal-badge searching';
                        sl.textContent = d.status || '等待检测';
                        sl.style.color = '#888';
                    }

                    const fb = d.football;
                    document.getElementById('val-football').textContent =
                        fb ? '✅ 已检测 (' + fb.conf + '%)' : '❌ 未检测';
                    document.getElementById('val-football').className = 'value' + (fb ? ' ok' : ' fail');
                    document.getElementById('val-line').textContent =
                        d.goal_line ? '✅ 已检测 (红色)' : '❌ 未检测';
                    document.getElementById('val-line').className = 'value' + (d.goal_line ? ' ok' : ' fail');
                    document.getElementById('val-dist').textContent =
                        fb && fb.distance_cm > 0 ? fb.distance_cm + ' cm' : '--';
                    document.getElementById('val-crossed').textContent =
                        d.crossed_line === true ? '✅ 是' :
                        d.crossed_line === false ? '❌ 否' : '--';
                    document.getElementById('val-crossed').className = 'value' +
                        (d.crossed_line === true ? ' ok' :
                         d.crossed_line === false ? ' fail' : '');
                    document.getElementById('val-scan').textContent = d.scan_state || '--';
                })
                .catch(() => {});
        }

        updateStatus();
        setInterval(updateStatus, 300);
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/video_feed")
def video_feed():
    return Response(generate_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    import json
    with frame_lock:
        r = current_result.copy()
    return json.dumps(r), 200, {"Content-Type": "application/json"}


@app.route("/reset")
def reset():
    detector.reset_session()
    return jsonify({"success": True})


if __name__ == "__main__":
    print("=" * 50)
    print("TonyPi 越线检测 Web 服务")
    print("=" * 50)
    print(f"端口: {WEB_PORT}")
    print(f"浏览器打开: http://<机器人IP>:{WEB_PORT}")
    print(f"\n检测内容:")
    print(f"  ⚽ YOLO → 足球 (football_best_win.onnx)")
    print(f"  🔴 颜色 → 红色球门线 (lab_config.yaml: red)")
    print(f"  👀 找不到线时自动低头/左右扫视")
    print(f"\n按 Ctrl+C 退出")
    print("=" * 50)

    thread = threading.Thread(target=camera_and_detect_thread, daemon=True)
    thread.start()

    # 等待摄像头就绪
    while not camera_ready:
        time.sleep(0.1)

    app.run(host="0.0.0.0", port=WEB_PORT, threaded=True, use_reloader=False)
