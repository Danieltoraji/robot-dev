#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_server.py —— 闯关运行中的被动镜像调试服务器（机器人端）

设计原则（详见《视觉调试指南》三工具分工）：
  - 不占摄像头：只监视关卡正常拍照落盘的照片（默认 /home/pi/Pictures/photo_*.jpg）；
  - 零侵入：不 import core.robot_core（其模块级会连舵机串口，与闯关的 main.py
    双进程抢串口会干扰舵机指令）；PnP 数学用 camera_config 常量本地复刻；
  - 事件驱动：只有新照片才触发分析（一张只分析一次），无新照片时看门狗仅扫目录名；
  - 分析在机器人端（部署真实形态），叠加绘制在 PC 浏览器（前端 canvas，可分层开关）。

用法（机器人上，jupyter-env 解释器以获得 apriltag/onnxruntime/cv2）：
    /home/pi/jupyter-env/bin/python3 tools/debug_server.py --port 8081
浏览器打开 http://<机器人IP>:8081
"""

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# 允许直接运行本文件
if __package__ in (None, ""):
    _HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(_HERE))   # 仓库根（core/ vision/ models/）

import numpy as np

from core.camera_config import (
    CAMERA_INTRINSIC, CAMERA_DISTORTION,
    PNP_FIELD_MIN, PNP_FIELD_MAX, PNP_CAM_Z_MIN, PNP_CAM_Z_MAX,
    reproj_gate_px,
)

PHOTO_PREFIX = "photo_"
DEFAULT_MODEL = "models/football_goal_ball_v1_640.onnx"


# =====================================================================
# PnP 定位：与 core.robot_core.solve_pnp_pose 相同的数学。
# 刻意本地复刻而非 import robot_core——后者模块级实例化舵机串口 Board，
# 调试服务器若 import 会与闯关的 main.py 双进程争抢串口。
# =====================================================================

def pnp_pose(objlist, imglist):
    """返回 (pos_3d[cm], ori_3d 单位向量, 重投影误差 px)，失败 None"""
    obj = np.asarray(objlist, dtype=np.float64)
    img = np.asarray(imglist, dtype=np.float64)
    ok, rvec, tvec = cv2_solvePnP(obj, img)
    if not ok:
        return None
    proj, _ = cv2_project(obj, rvec, tvec)
    reproj = float(np.mean(np.linalg.norm(proj[:, 0, :] - img, axis=1)))
    rot = cv2_rodrigues(rvec)
    pos = (-np.linalg.inv(rot) @ tvec).flatten()
    ori = (np.linalg.inv(rot) @ (np.array([[0.0], [0.0], [1.0]]) - tvec)).flatten() - pos
    n = np.linalg.norm(ori)
    if n:
        ori = ori / n
    return pos, ori, reproj


_cv2 = None


def _cv():
    global _cv2
    if _cv2 is None:
        import cv2
        _cv2 = cv2
    return _cv2


def cv2_solvePnP(obj, img):
    cv2 = _cv()
    return cv2.solvePnP(obj, img, CAMERA_INTRINSIC, CAMERA_DISTORTION)


def cv2_project(obj, rvec, tvec):
    cv2 = _cv()
    return cv2.projectPoints(obj, rvec, tvec, CAMERA_INTRINSIC, CAMERA_DISTORTION)


def cv2_rodrigues(rvec):
    return _cv().Rodrigues(rvec)[0]


# =====================================================================
# 分析器集合（懒加载；enabled 由网页 POST /config 实时切换）
# =====================================================================

class Analyzers:
    def __init__(self, args):
        self.args = args
        self.enabled = {"tag": not args.no_tag,
                        "yolo": not args.no_yolo,
                        "digit": args.enable_digit}
        self._tag_det = None
        self._tag_poses = None
        self._yolo = None
        self._digit = None
        self.status = {"tag": "未加载", "yolo": "未加载", "digit": "未加载"}

    # ---- tag：apriltag + 可选 PnP（levels.goodluck.tag_poses 存在时）----
    def _load_tag(self):
        import apriltag
        self._tag_det = apriltag.Detector(
            apriltag.DetectorOptions(families="tag36h11"))
        try:
            from levels.goodluck import tag_poses
            self._tag_poses = {str(k): v for k, v in tag_poses.items()}
        except Exception as e:
            self._tag_poses = None
            self.status["tag"] += f"（无关卡 tag_poses，跳过 PnP: {e}）"

    def run_tag(self, frame):
        if self._tag_det is None:
            self._load_tag()
        gray = _cv().cvtColor(frame, _cv().COLOR_BGR2GRAY)
        tags, pose = [], None
        objlist, imglist = [], []
        known = 0
        for r in self._tag_det.detect(gray):
            corners = [[round(float(x), 1), round(float(y), 1)]
                       for x, y in r.corners]
            tags.append({"id": int(r.tag_id), "corners": corners})
            if self._tag_poses and str(r.tag_id) in self._tag_poses:
                known += 1
                objlist.extend(self._tag_poses[str(r.tag_id)])
                imglist.append(np.asarray(r.corners, dtype=np.float64))
        if known:
            res = pnp_pose(objlist, imglist)
            if res:
                pos, ori, reproj = res
                problems = []
                if not (PNP_FIELD_MIN <= pos[0] <= PNP_FIELD_MAX
                        and PNP_FIELD_MIN <= pos[1] <= PNP_FIELD_MAX):
                    problems.append("位置超出场地")
                if not (PNP_CAM_Z_MIN <= pos[2] <= PNP_CAM_Z_MAX):
                    problems.append("相机高度不合理")
                if reproj > reproj_gate_px(known):
                    problems.append(f"重投影 {reproj:.1f}px 超门控")
                pose = {"position": [round(pos[0], 1), round(pos[1], 1)],
                        "cam_z": round(float(pos[2]), 1),
                        "orientation": [round(float(ori[0]), 3), round(float(ori[1]), 3)],
                        "reproj": round(reproj, 2),
                        "tags_used": known,
                        "problems": problems}
        return {"tags": tags, "pose": pose}

    # ---- yolo：OnnxYoloBackend ----
    def _load_yolo(self):
        from vision.yolo_detector import OnnxYoloBackend
        model = self.args.model
        if not os.path.isabs(model):
            model = os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), model)
        self._yolo = OnnxYoloBackend(model, conf=self.args.conf, iou=0.45,
                                     input_size=640, names={0: "football"})

    def run_yolo(self, frame):
        if self._yolo is None:
            self._load_yolo()
        dets = self._yolo.detect(frame)
        return {"yolo": [{"cls": d.cls, "conf": round(d.confidence, 3),
                          "bbox": [round(v, 1) for v in d.bbox]} for d in dets]}

    # ---- digit：DigitRecognizer（模板缺失自动禁用）----
    def _load_digit(self):
        from vision.digit_recognizer import DigitRecognizer
        tdir = self.args.digit_templates
        templates = {}
        if os.path.isdir(tdir):
            cv2 = _cv()
            for fn in sorted(os.listdir(tdir)):
                if fn.endswith(".png") and os.path.splitext(fn)[0].isdigit():
                    tpl = cv2.imread(os.path.join(tdir, fn), 0)
                    if tpl is not None:
                        templates[os.path.splitext(fn)[0]] = tpl
        if not templates:
            raise RuntimeError(f"数字模板目录为空: {tdir}")
        self._digit = DigitRecognizer(templates=templates)

    def run_digit(self, frame):
        if self._digit is None:
            self._load_digit()
        r = self._digit.detect(frame)
        return {"digit": {"digits": r.digits, "conf": round(r.confidence, 3),
                          "center": list(r.center_px) if r.center_px else None}}

    def run(self, name, frame):
        runner = {"tag": self.run_tag, "yolo": self.run_yolo,
                  "digit": self.run_digit}[name]
        t0 = time.perf_counter()
        try:
            out = runner(frame)
            self.status[name] = "ok"
            return out, round((time.perf_counter() - t0) * 1000)
        except Exception as e:
            self.status[name] = f"错误: {e}"
            return {name: None}, 0


# =====================================================================
# 状态与看门狗
# =====================================================================

class Hub:
    """共享状态 + 照片看门狗（事件驱动分析）"""

    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.analyzers = Analyzers(args)
        self.photo_path = None
        self.photo_jpeg = b""
        self.photo_mtime = 0.0
        self.photo_ts = 0.0
        self.results = {}
        self.errors = {}
        self.analysis_ms = {}
        self._last_identity = None

    def _jpeg_complete(self, path):
        try:
            with open(path, "rb") as f:
                f.seek(-2, os.SEEK_END)
                return f.read(2) == b"\xff\xd9"
        except OSError:
            return False

    def watchdog_loop(self):
        while True:
            try:
                self.scan_once()
            except Exception as e:
                print(f"[watchdog] {e}")
            time.sleep(0.5)

    def scan_once(self):
        d = self.args.photo_dir
        if not os.path.isdir(d):
            return
        newest, newest_key = None, None
        for fn in os.listdir(d):
            if not (fn.startswith(PHOTO_PREFIX) and fn.endswith(".jpg")):
                continue
            p = os.path.join(d, fn)
            try:
                st = os.stat(p)
            except OSError:
                continue
            key = (st.st_mtime_ns, st.st_size)
            if newest_key is None or key > newest_key:
                newest, newest_key = p, key
        if newest is None or newest_key == self._last_identity:
            return
        if not self._jpeg_complete(newest):
            return  # 还没写完，下轮再见
        self._last_identity = newest_key
        self.analyze(newest, newest_key[0] / 1e9)

    def analyze(self, path, mtime_s):
        frame = _cv().imread(path)
        if frame is None:
            return
        with open(path, "rb") as f:
            jpeg = f.read()
        results, errors, ms = {}, {}, {}
        for name, on in list(self.analyzers.enabled.items()):
            if not on:
                continue
            out, cost = self.analyzers.run(name, frame)
            results.update(out if out else {})
            if out and out.get(name) is None:
                errors[name] = "分析失败"
            ms[name] = cost
        with self.lock:
            self.photo_path = path
            self.photo_jpeg = jpeg
            self.photo_mtime = mtime_s
            self.photo_ts = time.time()
            self.results = results
            self.errors = errors
            self.analysis_ms = ms
        print(f"[analyze] {os.path.basename(path)} "
              f"tags={len(results.get('tags', []))} "
              f"yolo={len(results.get('yolo', []))} "
              f"pose={'有' if results.get('pose') else '无'} {ms}")

    def state_payload(self):
        with self.lock:
            return {
                "photo_age_s": round(time.time() - self.photo_ts, 1)
                               if self.photo_ts else None,
                "photo_file": os.path.basename(self.photo_path) if self.photo_path else None,
                "analyzers": dict(self.analyzers.enabled),
                "analyzer_status": dict(self.analyzers.status),
                "results": self.results,
                "errors": self.errors,
                "analysis_ms": self.analysis_ms,
            }

    def photo_bytes(self):
        with self.lock:
            return self.photo_jpeg


# =====================================================================
# HTTP
# =====================================================================

PAGE_HTML = """<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>机器人视觉调试镜像</title>
<style>
 body { font-family: sans-serif; background: #1c1c1c; color: #ddd; margin: 12px; }
 .wrap { position: relative; max-width: 1280px; }
 img, canvas { width: 100%; display: block; }
 canvas { position: absolute; top: 0; left: 0; }
 .panel { display: flex; gap: 24px; flex-wrap: wrap; margin-bottom: 8px; }
 fieldset { border: 1px solid #555; border-radius: 6px; min-width: 180px; }
 legend { padding: 0 6px; color: #8cf; }
 #status { font-family: monospace; white-space: pre-wrap; background: #111;
           padding: 8px; border-radius: 6px; }
</style></head>
<body>
<div class="panel">
 <fieldset><legend>分析器（机器人端）</legend>
  <label><input type="checkbox" id="en_tag" checked> tag 定位</label><br>
  <label><input type="checkbox" id="en_yolo" checked> yolo 球/门</label><br>
  <label><input type="checkbox" id="en_digit"> digit 数字</label>
 </fieldset>
 <fieldset><legend>图层显示</legend>
  <label><input type="checkbox" id="ly_tagbox" checked> tag 角点</label><br>
  <label><input type="checkbox" id="ly_tagid" checked> tag id</label><br>
  <label><input type="checkbox" id="ly_pose" checked> 定位信息</label><br>
  <label><input type="checkbox" id="ly_yolo" checked> yolo 框+置信度</label><br>
  <label><input type="checkbox" id="ly_digit" checked> digit</label>
 </fieldset>
 <fieldset><legend>状态</legend><div id="status">连接中...</div></fieldset>
</div>
<div class="wrap">
 <img id="photo" alt="">
 <canvas id="overlay"></canvas>
</div>
<script>
const img = document.getElementById('photo');
const canvas = document.getElementById('overlay');
let state = null;

function draw() {
  canvas.width = img.naturalWidth || 640;
  canvas.height = img.naturalHeight || 480;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  if (!state || !state.results) return;
  const k = canvas.width / (state.image_width || canvas.width);
  const box = (b, color, w) => { ctx.strokeStyle = color; ctx.lineWidth = w;
    ctx.strokeRect(b[0]*k, b[1]*k, b[2]*k, b[3]*k); };
  const text = (t, x, y, color) => { ctx.fillStyle = color; ctx.font = '28px monospace';
    ctx.fillText(t, x*k, y*k); };
  const ly = id => document.getElementById(id).checked;
  if (ly('ly_yolo')) for (const d of state.results.yolo || []) {
    box(d.bbox, '#00ff00', 6);
    if (document.getElementById('ly_conf').checked)
      text(d.cls + ' ' + d.conf.toFixed(2), d.bbox[0], Math.max(d.bbox[1]-10, 30), '#00ff00');
  }
  if (ly('ly_tagbox')) for (const t of state.results.tags || []) {
    ctx.strokeStyle = '#00ffff'; ctx.lineWidth = 4; ctx.beginPath();
    t.corners.forEach((p, i) => i ? ctx.lineTo(p[0]*k, p[1]*k)
                                  : ctx.moveTo(p[0]*k, p[1]*k));
    ctx.closePath(); ctx.stroke();
    if (document.getElementById('ly_tagid').checked)
      text('id=' + t.id, t.corners[0][0], t.corners[0][1] - 10, '#00ffff');
  }
  let sl = 30;
  if (ly('ly_pose') && state.results.pose) {
    const p = state.results.pose;
    const warn = p.problems && p.problems.length ? ' ⚠' + p.problems.join(',') : '';
    text(`定位 (${p.position[0]}, ${p.position[1]})cm 朝向(${p.orientation[0]}, ${p.orientation[1]}) 重投影${p.reproj}px${warn}`,
         20, sl += 40, p.problems.length ? '#ff8888' : '#ffff88');
  }
  if (ly('ly_digit') && state.results.digit && state.results.digit.digits) {
    const dg = state.results.digit;
    if (dg.center) box([dg.center[0]-60, dg.center[1]-60, 120, 120], '#ff8800', 4);
    text('digit ' + dg.digits + ' ' + dg.conf.toFixed(2), 20, canvas.height - 20, '#ff8800');
  }
}

function updateStatus() {
  const s = document.getElementById('status');
  if (!state) { s.textContent = '无数据'; return; }
  const lines = [`照片: ${state.photo_file || '无'} (年龄 ${state.photo_age_s}s)`];
  for (const k of ['tag', 'yolo', 'digit'])
    lines.push(`${k}: ${state.analyzers[k] ? '开' : '关'} | ${state.analyzer_status[k] || ''} | ${state.analysis_ms[k] ? state.analysis_ms[k] + 'ms' : '-'}`);
  if (state.errors && Object.keys(state.errors).length)
    lines.push('错误: ' + JSON.stringify(state.errors));
  s.textContent = lines.join('\\n');
}

function refresh() {
  fetch('/api/state').then(r => r.json()).then(s => { state = s; updateStatus(); })
                      .catch(() => { document.getElementById('status').textContent = '连接断开...'; });
  img.src = '/photo?t=' + Date.now();
}
img.onload = draw;
img.onerror = () => { /* 无照片时静默 */ };
for (const id of ['en_tag', 'en_yolo', 'en_digit'])
  document.getElementById(id).onchange = () => {
    fetch('/config', {method: 'POST',
      body: JSON.stringify({analyzers: {tag: en_tag.checked, yolo: en_yolo.checked,
                                        digit: en_digit.checked}})});
  };
for (const id of ['ly_tagbox', 'ly_tagid', 'ly_pose', 'ly_yolo', 'ly_conf', 'ly_digit'])
  document.getElementById(id).onchange = draw;
setInterval(refresh, 1000);
refresh();
</script>
</body></html>
"""


class Handler(BaseHTTPRequestHandler):
    hub = None

    def log_message(self, fmt, *args):  # 静默访问日志
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        if ctype.startswith("image/") or ctype == "application/json":
            self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            self._send(200, PAGE_HTML.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/photo":
            data = self.hub.photo_bytes()
            if not data:
                self._send(404, b"no photo yet", "text/plain")
            else:
                self._send(200, data, "image/jpeg")
        elif path == "/api/state":
            payload = self.hub.state_payload()  # state_payload 内部已持锁（Lock 不可重入，勿外层再包）
            payload["image_width"] = 2592  # 前端坐标基准（photo 恒为原生分辨率）
            payload["image_height"] = 1944
            self._send(200, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                       "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path.split("?")[0] != "/config":
            self._send(404, b"not found", "text/plain")
            return
        n = int(self.headers.get("Content-Length", 0))
        try:
            cfg = json.loads(self.rfile.read(n).decode("utf-8"))
            for k, v in cfg.get("analyzers", {}).items():
                if k in self.hub.analyzers.enabled:
                    self.hub.analyzers.enabled[k] = bool(v)
            self._send(200, b'{"ok": true}', "application/json")
        except (ValueError, json.JSONDecodeError):
            self._send(400, b'{"ok": false}', "application/json")


# =====================================================================
# 入口
# =====================================================================

def main():
    ap = argparse.ArgumentParser(description="被动镜像视觉调试服务器（不占摄像头）")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--photo-dir", default="/home/pi/Pictures",
                    help="关卡照片落盘目录（默认 /home/pi/Pictures）")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"YOLO onnx 模型（默认 {DEFAULT_MODEL}）")
    ap.add_argument("--conf", type=float, default=0.45, help="YOLO 置信度阈值")
    ap.add_argument("--digit-templates", default="tools/digit_templates",
                    help="数字模板目录（缺失则 digit 分析器禁用）")
    ap.add_argument("--no-tag", action="store_true", help="初始关闭 tag 分析器")
    ap.add_argument("--no-yolo", action="store_true", help="初始关闭 yolo 分析器")
    ap.add_argument("--enable-digit", action="store_true", help="初始启用 digit 分析器")
    args = ap.parse_args()

    hub = Hub(args)
    Handler.hub = hub
    threading.Thread(target=hub.watchdog_loop, daemon=True).start()

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    server.daemon_threads = True
    print(f"调试镜像服务器: http://0.0.0.0:{args.port}  "
          f"(照片目录 {args.photo_dir}，分析器 tag/yolo 开、digit 关)")
    print("Ctrl+C 退出")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
