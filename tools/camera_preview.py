# -*- coding: utf-8 -*-
"""
摄像头实时预览工具（与关卡无关）

用途：
  真机调试/调节摄像头时，直接查看摄像头画面。

两种模式：
  1. GUI 模式（默认，有桌面环境时）：
       python -m tools.camera_preview
       打开 OpenCV 窗口，按 s 保存截图，按 q / ESC 退出。
  2. 网页流模式（适合 SSH / 无显示器）：
       python -m tools.camera_preview --stream
       或直接运行，脚本在 Linux 无 DISPLAY 时自动切换为网页流模式。
       然后浏览器打开 http://<树莓派IP>:8080 即可实时查看。

可选 AprilTag 叠加：
    python -m tools.camera_preview --apriltag
    python -m tools.camera_preview --stream --apriltag
"""

import argparse
import os
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# 允许直接运行本工具
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.paths import RESULT_DIR

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="摄像头实时预览工具")
    parser.add_argument("--camera", type=int, default=0,
                        help="摄像头设备索引，默认 0")
    parser.add_argument("--width", type=int, default=1280,
                        help="期望画面宽度，默认 1280")
    parser.add_argument("--height", type=int, default=720,
                        help="期望画面高度，默认 720")
    parser.add_argument("--apriltag", action="store_true",
                        help="开启 AprilTag 检测叠加（需要 apriltag 库）")
    parser.add_argument("--stream", action="store_true",
                        help="强制使用网页流模式，而不是 OpenCV 窗口")
    parser.add_argument("--port", type=int, default=8080,
                        help="网页流模式端口，默认 8080")
    parser.add_argument("--quality", type=int, default=80,
                        help="网页流 JPEG 质量，默认 80")
    return parser.parse_args()


def should_use_stream(args):
    """无桌面环境时自动使用网页流模式"""
    if args.stream:
        return True
    # Linux 下没有 DISPLAY/WAYLAND_DISPLAY 时，OpenCV 窗口无法显示
    if sys.platform.startswith("linux"):
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            return True
    return False


def draw_overlay(frame):
    """绘制中心十字线和三分线网格，方便对准摄像头"""
    h, w = frame.shape[:2]
    # 三分线
    for i in (1, 2):
        cv2.line(frame, (w * i // 3, 0), (w * i // 3, h), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.line(frame, (0, h * i // 3), (w, h * i // 3), (0, 255, 0), 1, cv2.LINE_AA)
    # 中心十字
    cx, cy = w // 2, h // 2
    cv2.line(frame, (cx - 30, cy), (cx + 30, cy), (0, 0, 255), 2, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - 30), (cx, cy + 30), (0, 0, 255), 2, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), 2, cv2.LINE_AA)
    return frame


def draw_apriltag(frame, detector):
    """检测并绘制 AprilTag 边框与 ID"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    try:
        results = detector.detect(gray)
    except Exception as e:
        cv2.putText(frame, f"apriltag error: {e}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return frame

    for r in results:
        corners = r.corners.astype(int)
        cv2.polylines(frame, [corners], True, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"id={r.tag_id}", (corners[0][0], corners[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    return frame


def make_detector():
    """创建 AprilTag 检测器；失败返回 None"""
    try:
        import apriltag
        options = apriltag.DetectorOptions(families="tag36h11")
        return apriltag.Detector(options)
    except Exception as e:
        print(f"[camera_preview] 无法启用 AprilTag 检测: {e}")
        return None


def process_frame(frame, detector):
    """叠加网格和 AprilTag，返回处理后的画面"""
    display = frame.copy()
    display = draw_overlay(display)
    if detector is not None:
        display = draw_apriltag(display, detector)
    return display


def run_gui(args, detector):
    """OpenCV 窗口模式"""
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[camera_preview] 无法打开摄像头，索引: {args.camera}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    os.makedirs(RESULT_DIR, exist_ok=True)
    print("[camera_preview] 按 s 保存截图，按 q / ESC 退出")

    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[camera_preview] 读取画面失败，摄像头可能被占用或断开。")
                break

            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps = fps * 0.9 + (1.0 / dt) * 0.1

            display = process_frame(frame, detector)
            h, w = display.shape[:2]
            info = f"Camera {args.camera} | {w}x{h} | {fps:.1f} FPS"
            cv2.putText(display, info, (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display, "s=save q=quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("Camera Preview", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q 或 ESC
                break
            if key == ord("s"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = os.path.join(RESULT_DIR, f"camera_snapshot_{timestamp}.png")
                cv2.imwrite(path, frame)
                print(f"[camera_preview] 截图已保存: {path}")
    except cv2.error as e:
        print(f"[camera_preview] GUI 模式失败（可能没有显示器）: {e}")
        print("请使用: python -m tools.camera_preview --stream")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[camera_preview] 已退出")


# =====================================================================
# 网页流模式
# =====================================================================

class MjpegStreamer:
    """后台抓帧，保存最新 JPEG 帧供 HTTP 流输出"""

    def __init__(self, args, detector):
        self.args = args
        self.detector = detector
        self.cap = cv2.VideoCapture(args.camera)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头，索引: {args.camera}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

        self.lock = threading.Lock()
        self.latest_frame = None
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)

    def start(self):
        self.thread.start()

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            display = process_frame(frame, self.detector)
            ok, jpeg = cv2.imencode(
                ".jpg", display,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.args.quality],
            )
            if ok:
                with self.lock:
                    self.latest_frame = jpeg.tobytes()

    def release(self):
        self.running = False
        self.cap.release()


class StreamHandler(BaseHTTPRequestHandler):
    streamer = None

    def do_GET(self):
        if self.path != "/":
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        try:
            while True:
                with self.streamer.lock:
                    frame = self.streamer.latest_frame
                if frame is not None:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                time.sleep(0.03)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, fmt, *args):
        # 减少刷屏，只记录非流请求
        if not self.path.startswith("/"):
            super().log_message(fmt, *args)


def get_lan_ip():
    """获取一个本机局域网 IP，用于提示访问地址"""
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def run_stream(args, detector):
    """启动 MJPEG HTTP 流，浏览器实时查看"""
    try:
        streamer = MjpegStreamer(args, detector)
    except Exception as e:
        print(f"[camera_preview] {e}")
        return

    streamer.start()
    StreamHandler.streamer = streamer

    # 端口被占用时自动向后尝试，避免 "Address already in use"
    server = None
    used_port = args.port
    for port in range(args.port, args.port + 20):
        try:
            server = ThreadingHTTPServer(("0.0.0.0", port), StreamHandler)
            used_port = port
            break
        except OSError as e:
            if e.errno == 98:  # Address already in use
                continue
            raise

    if server is None:
        print(f"[camera_preview] 端口 {args.port}~{args.port + 19} 均被占用，无法启动。")
        streamer.release()
        return

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    ip = get_lan_ip()
    url = f"http://{ip}:{used_port}"
    print("=" * 50)
    print("[camera_preview] 网页流模式已启动")
    if used_port != args.port:
        print(f"[camera_preview] 端口 {args.port} 被占用，已自动改用端口 {used_port}")
    print(f"  本机访问: {url}")
    print(f"  同一局域网内其它电脑访问: {url}")
    print("  按 Ctrl+C 停止")
    print("=" * 50)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[camera_preview] 正在停止...")
    finally:
        streamer.release()
        server.shutdown()
        server.server_close()
        print("[camera_preview] 已退出")


def main():
    args = parse_args()

    detector = None
    if args.apriltag:
        detector = make_detector()
        if detector is not None:
            print("[camera_preview] AprilTag 检测已开启 (tag36h11)")

    if should_use_stream(args):
        run_stream(args, detector)
    else:
        run_gui(args, detector)


if __name__ == "__main__":
    main()
