# -*- coding: utf-8 -*-
"""双核视频流采集封装：后台线程持续抓帧，主控线程取最新帧处理。

设计（方案一「双核视频流处理」）：
  - 采集线程 _capture_loop 独占摄像头，循环 cap.read()，只保存最新一帧；
  - 处理线程（调用方）通过 read() 拿最新帧，与采集并发，帧始终最新、延迟低。

相比「处理线程每步重新 read 摄像头」的单线程模式，采集线程持续刷新缓冲区，
处理端总是拿到刚出炉的帧，避免动作结束后再等曝光/缓冲的额外延迟。

设备枚举/参数沿用 test_programs_NoUseInMain/line_follow_debug.py 原 open_camera()
逻辑：优先 /dev/v4l/by-id 下的 -video-index0，回退 /dev/videoN，支持 open_once
时追加 mjpg 流 URL。不依赖 hiwonder，保持 vision 模块纯净。
"""

import glob
import logging
import threading
import time

import cv2


class CameraStream:
    """后台抓帧 + 最新帧读取的摄像头流。

    cam_size      : 期望分辨率 (w, h)，仅对 /dev/ 设备生效
    open_once_url : 相机 open_once 时的 mjpg 流 URL（可选，由调用方传入）
    """

    def __init__(self, cam_size=(640, 480), open_once_url=None):
        self.cam_size = cam_size
        self.open_once_url = open_once_url
        self.log = logging.getLogger("camera_stream")

        self.device = None
        self._cap = None
        self._lock = threading.Lock()
        self._latest = None
        self._running = False
        self._thread = None

    # -----------------------------------------------------------------
    # 设备枚举 + 打开
    # -----------------------------------------------------------------

    def _build_candidates(self):
        candidates = sorted(glob.glob("/dev/v4l/by-id/*-video-index0"))
        candidates.extend("/dev/video%d" % index for index in range(10))
        if self.open_once_url:
            candidates.append(self.open_once_url)
        return candidates

    def _open_capture(self):
        """逐个候选打开并预热，返回 (cap, first_frame)；全失败抛 RuntimeError"""
        for device in self._build_candidates():
            cap = cv2.VideoCapture(device)
            if not cap.isOpened():
                cap.release()
                continue
            if str(device).startswith("/dev/"):
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("Y", "U", "Y", "V"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_size[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_size[1])
            for _ in range(8):
                ok, frame = cap.read()
                if ok and frame is not None:
                    self.device = str(device)
                    self.log.info(
                        "camera opened: %s frame=%dx%d",
                        device, frame.shape[1], frame.shape[0],
                    )
                    return cap, frame
                time.sleep(0.05)
            cap.release()
        raise RuntimeError("no working USB camera capture node found")

    # -----------------------------------------------------------------
    # 生命周期
    # -----------------------------------------------------------------

    def start(self):
        """打开摄像头并启动后台抓帧线程"""
        cap, first = self._open_capture()
        self._cap = cap
        with self._lock:
            self._latest = first
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        while self._running:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            with self._lock:
                self._latest = frame

    def read(self):
        """取最新一帧；无帧返回 None。返回引用，调用方应同步处理、勿跨线程持有"""
        with self._lock:
            return self._latest

    def release(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        with self._lock:
            self._latest = None
