#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_server.py —— 闯关运行中的被动镜像调试服务器（机器人端）

设计原则（详见《视觉调试指南》三工具分工）：
  - 不占摄像头：只监视关卡正常拍照落盘的照片（默认 /home/pi/Pictures/photo_*.jpg）；
  - 零侵入：不改 core/levels/main.py 任何调用链；tag 分析器**复用关卡内部
    逻辑**（core.robot_core 的同款 apriltag 检测 / solve_pnp_pose / 完整
    pnp_pose_problems 门控）——import 会打开串口句柄（SDK 可导入时）但调试
    服务器从不写串口，与 TonyPi.py/main.py 共持串口的现状一致；异常或
    --no-core 时自动降级本地简化实现；yolo/digit/line 复用 vision/ 共享包；
  - 事件驱动：只有新照片才触发分析（一张只分析一次），无新照片时看门狗仅扫
    目录名；
  - 分析在机器人端（部署真实形态），叠加绘制在 PC 浏览器（前端 canvas，可分
    层开关）。

用法（机器人上，jupyter-env 解释器以获得 apriltag/onnxruntime/cv2）：
    /home/pi/jupyter-env/bin/python3 tools/debug_server.py --port 8081
浏览器打开 http://<机器人IP>:8081
"""

import argparse
import json
import os
import subprocess
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
    CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_INTRINSIC, CAMERA_DISTORTION,
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
                        "digit": args.enable_digit,
                        "line": not args.no_line}
        self._tag_det = None
        self._tag_poses = None
        self._yolo = None
        self._digit = None
        self._line = None
        self.mode = None      # tag 分析模式："core"（复用关卡逻辑）/"local"（降级）
        self._rc = None
        self._state = None
        self.status = {"tag": "未加载", "yolo": "未加载",
                       "digit": "未加载", "line": "未加载"}

    # ---- tag：apriltag + 可选 PnP（levels.goodluck.tag_poses 存在时）----
    def _load_tag(self):
        """双模式：首选复用关卡内部逻辑（core.robot_core 的同款 apriltag 检测、
        solve_pnp_pose、完整 pnp_pose_problems 门控）；core 不可用（PC 无 SDK、
        apriltag 缺失或 --no-core）时降级为本地简化实现。"""
        self.mode = "local"
        if not self.args.no_core:
            try:
                import core.robot_core as rc
                from levels.goodluck import tag_poses
                if rc.apriltag is None:
                    raise RuntimeError("core.apriltag 不可用")
                self._rc = rc
                self._state = rc.RobotState(tag_poses=tag_poses)
                self._tag_poses = {str(k): v for k, v in tag_poses.items()}
                self.mode = "core"
                self.status["tag"] = "core 复用模式"
                return
            except Exception as e:
                self.status["tag"] = f"core 不可用({e})，降级本地实现"
        import apriltag
        self._tag_det = apriltag.Detector(
            apriltag.DetectorOptions(families="tag36h11"))
        try:
            from levels.goodluck import tag_poses
            self._tag_poses = {str(k): v for k, v in tag_poses.items()}
        except Exception as e:
            self._tag_poses = None
            self.status["tag"] += f"（无关卡 tag_poses，跳过 PnP: {e}）"

    def run_tag(self, frame, photo_path=None):
        if self.mode is None:
            self._load_tag()
        if self.mode == "core":
            return self._run_tag_core(photo_path)
        return self._run_tag_local(frame)

    def _run_tag_core(self, photo_path):
        """复用关卡内部逻辑：RobotState.detect_apriltag + solve_pnp_pose +
        pnp_pose_problems（完整门控），与闯关看到的结果同源。"""
        rc, state = self._rc, self._state
        if not photo_path or not os.path.exists(photo_path):
            return {"tags": [], "pose": None}  # 照片已被清理（如 tjz 滚动删除）
        try:
            dets = state.detect_apriltag(photo_path) or []
        except Exception as e:
            self.status["tag"] = f"检测异常: {e}"
            dets = []
        tags, objlist, imglist, known = [], [], [], 0
        for r in dets:
            corners = [[round(float(x), 1), round(float(y), 1)]
                       for x, y in r.corners]
            tags.append({"id": int(r.tag_id), "corners": corners})
            tid = str(r.tag_id)
            if tid in state.tag_poses:
                known += 1
                objlist.extend(state.tag_poses[tid])
                imglist.extend(
                    np.asarray(r.corners, dtype=np.float64)[rc.TAG_CORNER_PERM])
        pose = None
        if known:
            res = rc.solve_pnp_pose(objlist, imglist)
            if res:
                pos, ori, reproj = res
                pose = {"position": [round(float(pos[0]), 1),
                                     round(float(pos[1]), 1)],
                        "cam_z": round(float(pos[2]), 1),
                        "orientation": [round(float(ori[0]), 3),
                                        round(float(ori[1]), 3)],
                        "reproj": round(reproj, 2),
                        "tags_used": known,
                        "problems": rc.pnp_pose_problems(
                            pos, ori, reproj, n_tags=known)}
        return {"tags": tags, "pose": pose}

    def _run_tag_local(self, frame):
        """降级路径：本地 apriltag 检测 + camera_config 常量的简化 PnP 复刻"""
        if self._tag_det is None:
            import apriltag
            self._tag_det = apriltag.Detector(
                apriltag.DetectorOptions(families="tag36h11"))
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
                # 注意 extend（逐点拼接）而非 append——append 会把 4×2 整块塞成
                # 一项，solvePnP 收到畸形点集直接断言崩溃（真机踩坑 2026-09-08）
                imglist.extend(np.asarray(r.corners, dtype=np.float64))
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

    # ---- digit：与关卡同一条链路（色块 → 字形掩膜 → 仲裁/形状裁决）----
    #
    # 2026-09-23 重写。旧实现把**整帧**交给 vision.digit_recognizer.DigitRecognizer，
    # 而那个识别器是为**单字符掩膜**设计的（min_area 默认仅 30px²、且没有任何
    # 置信度门槛）。实测整帧会产生 84 个候选框（其中 62 个是皮尺刻度级碎片），
    # 每个框都强行输出一个数字 ⇒ 页面底部出现一条 80+ 位的无意义数字串。
    #
    # 现在改为复刻关卡的实际链路（见 levels/nine_grid.layout_scan）：
    #   detect_panels(arbitrate=True, shape=True)
    #     → 每个色块的 .color / .color_id / .digit / .shape_digit / .shape_conf
    #   其中 .digit 即关卡认定的数字：颜色为主判，形状仅在**颜色有歧义**且
    #   间隔 ≥ SHAPE_OVERRIDE_MIN_CONF(0.30) 时才允许改判。
    #
    # 注意：不再需要 --digit-templates（那些 arial/consolas 字体模板只服务于
    # 旧的整帧识别）；形状仲裁用的是现场自建 models/nine_grid/digit_templates.npz。
    def run_digit(self, frame):
        from vision.nine_grid_detector import NineGridDetector
        if self._digit is None:
            self._digit = NineGridDetector()
        obs = self._digit.detect_panels(frame, arbitrate=True, shape=True)
        if not obs:
            return {"digit": {"digits": "", "conf": 0.0, "center": None,
                              "panels": [], "note": "未检出任何色块"}}
        # 按可见面积降序，前端只画前几个以免刷屏
        obs = sorted(obs, key=lambda o: o.hull_area, reverse=True)[:6]
        panels = []
        for o in obs:
            x, y, w, h = (int(v) for v in o.bbox)
            panels.append({
                "color": o.color,
                "color_id": int(o.color_id),
                "digit": int(o.digit),              # ← 关卡认定的数字
                "shape_digit": (None if o.shape_digit is None
                                else int(o.shape_digit)),
                "shape_conf": round(float(o.shape_conf), 3),
                "overridden": bool(o.shape_override),
                "clipped": int(o.clipped),
                "bbox": [x, y, w, h],
                # 凸包多边形（原生像素 [[x,y],...]）：只包住色块本身，比 bbox
                # 更贴形。前端优先画它，画不出再退回 bbox 方框。
                "poly": [[round(px, 1), round(py, 1)] for px, py in o.hull_poly],
            })
        top = panels[0]
        conf = top["shape_conf"] if top["overridden"] else 1.0
        label = " ".join(
            f"{p['color']}→{p['digit']}" + ("*" if p["overridden"] else "")
            for p in panels)
        cx = sum(p["bbox"][0] + p["bbox"][2] / 2.0 for p in panels) / len(panels)
        cy = sum(p["bbox"][1] + p["bbox"][3] / 2.0 for p in panels) / len(panels)
        return {"digit": {"digits": label, "conf": round(float(conf), 3),
                          "center": [cx, cy], "panels": panels}}

    def _load_line(self):
        # 兼容两种导入风格：仓库版 line_detector 用 `from vision.detection
        # import ...`；机器人上的并行会话版本用平铺 `from detection import ...`
        # （需将 vision/ 目录加入 sys.path 才能解析）。
        vdir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "vision")
        if vdir not in sys.path:
            sys.path.insert(0, vdir)
        try:
            from vision.line_detector import LineDetector
        except ModuleNotFoundError:
            from line_detector import LineDetector
        a, b = self.args.line_hsv.split(":")
        hsv = (tuple(int(v) for v in a.split(",")),
               tuple(int(v) for v in b.split(",")))
        # 颜色模式认红线（line_detector 对 h≤10 自动补 170~180 段）；
        # roi_ratio=1.0：调试要观察全画面的线，而非巡线关卡的下半幅
        self._line = LineDetector(line_color="color", hsv_range=hsv,
                                  roi_ratio=1.0, work_width=640)

    def run_line(self, frame):
        if self._line is None:
            self._load_line()
        h, w = frame.shape[:2]
        r = self._line.detect(frame)
        if not r.exists or r.primary is None:
            return {"line": None}
        # 工作坐标（640 宽、ROI 自 h*(1-roi_ratio) 起）→ 原图坐标
        s = w / self._line.work_width
        roi_top = h * (1.0 - self._line.roi_ratio)
        prim = r.primary
        pts = [[round(px * s, 1), round(roi_top + py * s, 1)]
               for px, py in prim.points]
        return {"line": {"orientation": prim.orientation,
                         "heading_deg": round(prim.heading_deg, 1),
                         "confidence": round(r.confidence, 3),
                         "others": len(r.others),
                         "points": pts}}

    def run(self, name, frame, photo_path=None):
        t0 = time.perf_counter()
        try:
            if name == "tag":
                out = self.run_tag(frame, photo_path)
            elif name == "yolo":
                out = self.run_yolo(frame)
            elif name == "digit":
                out = self.run_digit(frame)
            else:
                out = self.run_line(frame)
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

    DISPLAY_WIDTH = 960  # /photo 服务的降采样宽度（弱上行链路友好，~150KB）

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
        self._warned_corrupt = set()
        self._capturing = False
        self.capture_note = "就绪"

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
        entries = []
        for fn in os.listdir(d):
            if not (fn.startswith(PHOTO_PREFIX) and fn.endswith(".jpg")):
                continue
            p = os.path.join(d, fn)
            try:
                st = os.stat(p)
            except OSError:
                continue
            if st.st_size < 1024:
                continue  # 0 字节/明显废件，直接无视
            entries.append(((st.st_mtime_ns, st.st_size), p))
        if not entries:
            return
        entries.sort(reverse=True)
        # 取最新的“完整”照片：损坏文件（无 FFD9）跳过并继续向前找，
        # 避免一张损坏的最新文件永久阻塞分析（真机踩坑 2026-09-08：
        # 重启瞬间正在写的照片被截断，mtime 恰好最新）
        for key, path in entries:
            if key == self._last_identity:
                return  # 最新完整照片已分析过 → 无新内容
            if self._jpeg_complete(path):
                self._last_identity = key
                self.analyze(path, key[0] / 1e9)
                return
            if path not in self._warned_corrupt:
                self._warned_corrupt.add(path)
                print(f"[watchdog] 跳过损坏文件: {path}", flush=True)
        # 全部损坏：本轮什么都不做

    def analyze(self, path, mtime_s):
        frame = _cv().imread(path)
        if frame is None:
            time.sleep(0.4)
            frame = _cv().imread(path)
        if frame is None:
            # 照片在落盘/清理竞争中暂不可读（如 tjz 进程滚动删除）：
            # 清除断点标记，下轮扫描会分析当时最新的完整照片
            self._last_identity = None
            print(f"[analyze] 照片暂不可读，下轮重试: {path}")
            return
        with open(path, "rb") as f:
            jpeg = f.read()
        results, errors, ms = {}, {}, {}
        for name, on in list(self.analyzers.enabled.items()):
            if not on:
                continue
            out, cost = self.analyzers.run(name, frame, self.photo_path)
            if out:
                results.update(out)   # 异常时 run 返回 {name: 文本}，由状态栏展示
            ms[name] = cost

        # /photo 服务降采样小图（弱上行链路友好），结果坐标同步缩放
        h, w = frame.shape[:2]
        scale = min(1.0, self.DISPLAY_WIDTH / w)
        if scale < 1.0:
            small = _cv().resize(frame, (int(w * scale), int(h * scale)),
                                 interpolation=_cv().INTER_AREA)
            ok, buf = _cv().imencode(".jpg", small,
                                     [_cv().IMWRITE_JPEG_QUALITY, 70])
            jpeg = buf.tobytes() if ok else jpeg
        if scale < 1.0:
            for t in results.get("tags", []):
                t["corners"] = [[round(x * scale, 1), round(y * scale, 1)]
                                for x, y in t["corners"]]
            for d in results.get("yolo", []):
                d["bbox"] = [round(v * scale, 1) for v in d["bbox"]]
            dg = results.get("digit")
            if dg and dg.get("center"):
                dg["center"] = [round(v * scale, 1) for v in dg["center"]]
            # ⚠️ panels[].bbox 也必须缩 —— 否则页面上的面板框按原生分辨率画，
            # 在降采样显示图上会放大约 1/scale 倍（960/2592 → 2.7×）而错位。
            # 2026-09-23 实测踩坑：文字（用 center，已缩放）大致在位，框却巨大
            # 且套不住面板，看起来像"识别错了"，实际是坐标契约不一致。
            if dg:
                for p in dg.get("panels") or []:
                    p["bbox"] = [round(v * scale, 1) for v in p["bbox"]]
                    if p.get("poly"):
                        p["poly"] = [[round(px * scale, 1), round(py * scale, 1)]
                                     for px, py in p["poly"]]
            ln = results.get("line")
            if ln and ln.get("points"):
                ln["points"] = [[round(x * scale, 1), round(y * scale, 1)]
                                for x, y in ln["points"]]

        with self.lock:
            self.photo_path = path
            self.photo_jpeg = jpeg
            self.photo_mtime = mtime_s
            self.photo_ts = time.time()
            self.results = results
            self.errors = errors
            self.analysis_ms = ms
            self.display_w = int(w * scale)
            self.display_h = int(h * scale)
        print(f"[analyze] {os.path.basename(path)} "
              f"tags={len(results.get('tags', []))} "
              f"yolo={len(results.get('yolo', []))} "
              f"pose={'有' if results.get('pose') else '无'} {ms}")

    def trigger_capture(self):
        """网页按钮触发机器人拍照（fswebcam 直拍，与关卡拍照同款命令）。
        ⚠ 闯关运行时点击可能与关卡抢相机——仅供关卡未运行时手动观察用。"""
        if self._capturing:
            self.capture_note = "上一次拍照还在进行中"
            return
        self._capturing = True
        self.capture_note = "拍照中（约 3 秒）..."
        threading.Thread(target=self._do_capture, daemon=True).start()

    def _do_capture(self):
        try:
            path = os.path.join(self.args.photo_dir,
                                f"photo_debug_{int(time.time() * 1000)}.jpg")
            cmd = (f"fswebcam -r {CAMERA_WIDTH}x{CAMERA_HEIGHT} "
                   f"--no-banner -S 3 {path}")
            r = subprocess.run(cmd, shell=True, capture_output=True,
                               text=True, timeout=30)
            if (r.returncode == 0 and os.path.exists(path)
                    and self._jpeg_complete(path)):
                self.capture_note = "拍照完成，画面即将更新"
                return
            if os.path.exists(path):
                os.remove(path)
            self.capture_note = "拍照失败（相机被占用？闯关中请勿点此按钮）"
        except Exception as e:
            self.capture_note = f"拍照异常: {e}"
        finally:
            self._capturing = False

    def state_payload(self):
        with self.lock:
            return {
                "photo_age_s": round(time.time() - self.photo_ts, 1)
                               if self.photo_ts else None,
                "photo_file": os.path.basename(self.photo_path) if self.photo_path else None,
                "image_width": getattr(self, "display_w", None) or 2592,
                "image_height": getattr(self, "display_h", None) or 1944,
                "analyzers": dict(self.analyzers.enabled),
                "analyzer_status": dict(self.analyzers.status),
                "tag_mode": self.analyzers.mode,
                "capturing": self._capturing,
                "capture_note": self.capture_note,
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
  <label><input type="checkbox" id="en_digit"> digit 数字</label><br>
  <label><input type="checkbox" id="en_line" checked> line 巡线</label>
 </fieldset>
 <fieldset><legend>图层显示</legend>
  <label><input type="checkbox" id="ly_tagbox" checked> tag 角点</label><br>
  <label><input type="checkbox" id="ly_tagid" checked> tag id</label><br>
  <label><input type="checkbox" id="ly_pose" checked> 定位信息</label><br>
  <label><input type="checkbox" id="ly_yolo" checked> yolo 框</label><br>
  <label><input type="checkbox" id="ly_conf" checked> 置信度文本</label><br>
  <label><input type="checkbox" id="ly_digit" checked> digit</label><br>
  <label><input type="checkbox" id="ly_line" checked> 巡线（红）</label>
 </fieldset>
 <fieldset><legend>状态</legend><div id="status">连接中...</div></fieldset>
 <fieldset><legend>机器人拍照</legend>
  <button id="btn_capture" style="padding:6px 14px;">📷 触发拍照</button>
  <div id="cap_note" style="margin-top:6px;font-size:13px;color:#aaa;">
   闯关运行时勿点（会与关卡抢相机）</div>
 </fieldset>
</div>
<div class="wrap">
 <img id="photo" alt="">
 <canvas id="overlay"></canvas>
</div>
<script>
const capBtn = document.getElementById('btn_capture');
capBtn.onclick = () => {
  capBtn.disabled = true;
  document.getElementById('cap_note').textContent = '拍照中（约 3 秒）...';
  fetch('/capture', {method: 'POST'}).then(r => r.json()).catch(() => {});
};
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
  if (ly('ly_digit') && state.results.digit && state.results.digit.panels) {
    // 与关卡同一链路：每个色块画**凸包多边形**（只包色块本身，比 bbox 贴形）
    // 并标注 颜色→关卡认定的数字（* = 被形状仲裁改判）
    const dg = state.results.digit;
    for (const p of dg.panels) {
      const col = p.overridden ? '#ff44ff' : '#ff8800';
      if (p.poly && p.poly.length >= 3) {
        ctx.strokeStyle = col; ctx.lineWidth = 5; ctx.beginPath();
        p.poly.forEach((q, i) => i ? ctx.lineTo(q[0]*k, q[1]*k)
                                   : ctx.moveTo(q[0]*k, q[1]*k));
        ctx.closePath(); ctx.stroke();
      } else {
        box(p.bbox, col, 5);   // 退路：多边形拿不到时仍画方框
      }
      let s = p.color + '→' + p.digit + (p.overridden ? '*' : '');
      if (document.getElementById('ly_conf').checked) {
        s += p.overridden ? ' Δ' + p.shape_conf.toFixed(2)
                          : (p.shape_digit != null ? ' vs' + p.shape_digit : '') +
                            (p.clipped ? ' clip' : '');
      }
      text(s, p.bbox[0], Math.max(p.bbox[1] - 10, 30), col);
    }
    const summary = dg.note ? ('digit: ' + dg.note)
                            : ('digit ' + dg.digits);
    text(summary, 20, canvas.height - 20, '#ff8800');
  }
  if (ly('ly_line') && state.results.line && state.results.line.points) {
    const L = state.results.line;
    ctx.strokeStyle = '#ff5555'; ctx.lineWidth = 5; ctx.beginPath();
    L.points.forEach((p, i) => i ? ctx.lineTo(p[0]*k, p[1]*k)
                                 : ctx.moveTo(p[0]*k, p[1]*k));
    ctx.stroke();
    const p0 = L.points[0];
    text(`line ${L.orientation} ${L.heading_deg}° (${L.points.length}点)`,
         Math.max(p0[0]-10, 10), Math.max(p0[1]-14, 30), '#ff5555');
  }
}

function updateStatus() {
  const s = document.getElementById('status');
  const note = document.getElementById('cap_note');
  const capBtn = document.getElementById('btn_capture');
  if (state) {
    if (state.capture_note) note.textContent = state.capture_note;
    capBtn.disabled = !!state.capturing;
  }
  if (!state) { s.textContent = '无数据'; return; }
  const lines = [`照片: ${state.photo_file || '无'} (年龄 ${state.photo_age_s}s)`];
  for (const k of ['tag', 'yolo', 'digit', 'line'])
    lines.push(`${k}: ${state.analyzers[k] ? '开' : '关'} | ${state.analyzer_status[k] || ''} | ${state.analysis_ms[k] ? state.analysis_ms[k] + 'ms' : '-'}`);
  if (state.tag_mode) lines.push(`tag 模式: ${state.tag_mode}`);
  if (state.errors && Object.keys(state.errors).length)
    lines.push('错误: ' + JSON.stringify(state.errors));
  s.textContent = lines.join('\\n');
}

let photoLoading = false;
function refreshState() {
  fetch('/api/state').then(r => r.json()).then(s => { state = s; updateStatus(); })
                      .catch(() => { document.getElementById('status').textContent = '连接断开（重试中）...'; });
}
function refreshPhoto() {
  if (photoLoading) return;   // 上一张没加载完不再发请求（弱链路防连接池耗尽）
  photoLoading = true;
  img.src = '/photo?t=' + Date.now();
}
img.onload = () => { photoLoading = false; draw(); };
img.onerror = () => { photoLoading = false; };
setInterval(refreshState, 1000);
setInterval(refreshPhoto, 2000);
for (const id of ['en_tag', 'en_yolo', 'en_digit', 'en_line'])
  document.getElementById(id).onchange = () => {
    fetch('/config', {method: 'POST',
      body: JSON.stringify({analyzers: {tag: en_tag.checked, yolo: en_yolo.checked,
                                        digit: en_digit.checked, line: en_line.checked}})});
  };
for (const id of ['ly_tagbox', 'ly_tagid', 'ly_pose', 'ly_yolo', 'ly_conf', 'ly_digit', 'ly_line'])
  document.getElementById(id).onchange = draw;
refreshState();
refreshPhoto();
</script>
</body></html>
"""


class Handler(BaseHTTPRequestHandler):
    hub = None
    timeout = 30  # 挂起的慢客户端（弱上行写 1MB 图）30s 强制断开，防线程耗尽

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
            self._send(200, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                       "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        path = self.path.split("?")[0]
        if path == "/capture":
            self.hub.trigger_capture()
            self._send(200, b'{"ok": true}', "application/json")
            return
        if path != "/config":
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
    # 注：2026-09-23 起 digit 分析器改走关卡链路（detect_panels + 形状仲裁），
    # 不再使用 arial/consolas 字体模板，故 --digit-templates 已移除。
    ap.add_argument("--no-tag", action="store_true", help="初始关闭 tag 分析器")
    ap.add_argument("--no-yolo", action="store_true", help="初始关闭 yolo 分析器")
    ap.add_argument("--enable-digit", action="store_true", help="初始启用 digit 分析器")
    ap.add_argument("--no-core", action="store_true",
                    help="tag 分析不复用 core.robot_core，强制本地简化实现")
    ap.add_argument("--no-line", action="store_true", help="初始关闭 line 巡线分析器")
    ap.add_argument("--line-hsv", default="0,90,80:10,255,255",
                    help="巡线 HSV 范围 'h,s,v:h,s,v'（默认红线，自动补 170~180 段）")
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
