#!/usr/bin/python3
# coding=utf8
"""足球射门阈值调试网页。

安全设计：
  * 默认只采集摄像头和检测足球，不自动走路、不自动踢球；
  * 页面可以实时修改“足够近”的阈值，观察当前检测框是否通过；
  * 手动射门按钮只执行一次正式 FootballKick 使用的 left_shot_fast /
    right_shot_fast，不执行自动对准和前进；
  * 适合在真实场地中把机器人摆到认为合适的射门位置后，验证阈值和踢球动作。

启动：
  sudo systemctl stop tonypi
  cd /home/pi/TonyPi/Functions
  python3 football_kick_debug_web.py --port 5002
"""

from __future__ import print_function

import argparse
import json
import threading
import time

import cv2
from flask import Flask, Response, jsonify, render_template_string, request

try:
    import hiwonder.Camera as Camera
    import hiwonder.ActionGroupControl as AGC
except ImportError:
    Camera = None
    AGC = None

try:
    from Functions.football_kick_controller import FootballDetector, FootballKickController
    from Functions.goalpost_detector import GoalPostDetector
    from Functions.red_goal_line_detector import RedGoalLineEstimator
except ImportError:
    from football_kick_controller import FootballDetector, FootballKickController
    from goalpost_detector import GoalPostDetector
    from red_goal_line_detector import RedGoalLineEstimator


FRAME_INTERVAL = 0.05
JPEG_QUALITY = 75

app = Flask(__name__)
frame_lock = threading.Lock()
action_lock = threading.Lock()
latest_frame = None
latest_display = None
latest_ball = None
latest_goalposts = []
latest_info = {}
last_error = ""
safe_streak = 0
last_action = ""
last_action_at = 0.0
controller_last_action = ""
controller_last_action_at = 0.0
controller_enabled = False
controller_armed = False

config = {
    "ball_confidence": 0.45,
    "safe_bottom_y": 405.0,
    # 旧参数保留兼容；当前距离判定不再使用 height。
    "safe_height_px": 90.0,
    "safe_confirm_frames": 3,
    "lane_tolerance_px": 85.0,
    "target_tolerance_px": 35.0,
    "left_kick_target_offset_px": -45.0,
    "right_kick_target_offset_px": 45.0,
    "line_flat_angle_deg": 8.0,
    "line_turn_angle_deg": 12.0,
    "line_min_points": 25,
    "line_confirm_frames": 2,
    "goalpost_center_tolerance_px": 45.0,
    "goalpost_turn_threshold_px": 120.0,
}


class DebugMotion:
    """状态机动作记录器；默认只记录建议动作，不驱动机器人。"""

    def run(self, action):
        global controller_last_action, controller_last_action_at
        controller_last_action = str(action)
        controller_last_action_at = time.time()
        if not controller_armed or AGC is None:
            return
        if not action_lock.acquire(blocking=False):
            return
        try:
            AGC.runActionGroup(str(action))
        finally:
            action_lock.release()


preview_line_estimator = None
preview_controller = None


def target_x_for(frame_width, action):
    center_x = float(frame_width) / 2.0
    offset = (
        config["left_kick_target_offset_px"]
        if action == "left_shot_fast"
        else config["right_kick_target_offset_px"]
    )
    return center_x + float(offset)


def select_action_for(cx, frame_width):
    left_target = target_x_for(frame_width, "left_shot_fast")
    right_target = target_x_for(frame_width, "right_shot_fast")
    left_distance = abs(float(cx) - left_target)
    right_distance = abs(float(cx) - right_target)
    if abs(left_distance - right_distance) < 1.0:
        return "left_shot_fast" if float(cx) <= float(frame_width) / 2.0 else "right_shot_fast"
    return "left_shot_fast" if left_distance < right_distance else "right_shot_fast"


def metrics_for(ball, frame_width):
    global safe_streak
    if ball is None:
        safe_streak = 0
        return {
            "detected": False,
            "safe_streak": 0,
            "distance_ok": False,
            "confirmed_ok": False,
            "lane_ok": False,
            "target_ok": False,
            "center_ok": False,
            "message": "当前没有检测到足球",
        }

    cx, cy, width, height, confidence = ball
    bottom_y = float(cy) + float(height) / 2.0
    image_center = float(frame_width) / 2.0
    dx = float(cx) - image_center
    action = select_action_for(cx, frame_width)
    target_x = target_x_for(frame_width, action)
    target_dx = float(cx) - target_x
    # 近距离时检测框可能被画面底部截断，height 会变小；距离硬判定只使用 bottom_y。
    distance_ok = bottom_y >= float(config["safe_bottom_y"])
    lane_ok = abs(target_dx) <= float(config["lane_tolerance_px"])
    target_ok = abs(target_dx) <= float(config["target_tolerance_px"])
    if distance_ok:
        safe_streak += 1
    else:
        safe_streak = 0
    confirmed_ok = safe_streak >= int(config["safe_confirm_frames"])

    if not distance_ok:
        message = "距离条件未通过：观察 bottom_y"
    elif not confirmed_ok:
        message = "距离条件已通过，等待连续确认帧"
    elif not lane_ok:
        message = "足球偏离当前射门脚的宽踢球走廊"
    elif not target_ok:
        message = "距离足够，但足球尚未进入当前射门脚目标窗口"
    else:
        message = "距离和射门脚目标窗口均通过（本页不自动踢球）"

    return {
        "detected": True,
        "cx": int(cx),
        "cy": int(cy),
        "w": int(width),
        "h": int(height),
        "confidence": round(float(confidence), 4),
        "bottom_y": round(bottom_y, 1),
        "dx": round(dx, 1),
        "target_dx": round(target_dx, 1),
        "target_x": round(target_x, 1),
        "left_target_x": round(target_x_for(frame_width, "left_shot_fast"), 1),
        "right_target_x": round(target_x_for(frame_width, "right_shot_fast"), 1),
        "distance_ok": bool(distance_ok),
        "safe_streak": int(safe_streak),
        "confirmed_ok": bool(confirmed_ok),
        "lane_ok": bool(lane_ok),
        "target_ok": bool(target_ok),
        # 旧前端字段保留为别名，避免旧页面脚本出错。
        "center_ok": bool(target_ok),
        "suggested_action": action,
        "message": message,
    }


def draw_debug(frame, ball, info, goalposts=None):
    image = frame.copy()
    height, width = image.shape[:2]
    center_x = width // 2
    cv2.line(image, (center_x, 0), (center_x, height), (255, 255, 0), 2)

    left_target = int(round(target_x_for(width, "left_shot_fast")))
    right_target = int(round(target_x_for(width, "right_shot_fast")))
    target_tol = int(config["target_tolerance_px"])
    lane_tol = int(config["lane_tolerance_px"])
    cv2.line(image, (left_target, 0), (left_target, height), (255, 80, 255), 2)
    cv2.line(image, (right_target, 0), (right_target, height), (80, 180, 255), 2)
    for x, color in (
        (left_target - lane_tol, (255, 180, 255)),
        (left_target + lane_tol, (255, 180, 255)),
        (right_target - lane_tol, (80, 220, 255)),
        (right_target + lane_tol, (80, 220, 255)),
    ):
        cv2.line(image, (x, 0), (x, height), color, 1)
    for x, color in (
        (left_target - target_tol, (255, 0, 255)),
        (left_target + target_tol, (255, 0, 255)),
        (right_target - target_tol, (0, 140, 255)),
        (right_target + target_tol, (0, 140, 255)),
    ):
        cv2.line(image, (x, 0), (x, height), color, 2)

    # 距离阈值线与实际红色球门线使用不同颜色，避免两条线重叠时混淆。
    safe_y = max(0, min(height - 1, int(config["safe_bottom_y"])))
    cv2.line(image, (0, safe_y), (width, safe_y), (0, 255, 255), 2)
    cv2.putText(
        image, "safe_bottom_y={}".format(int(config["safe_bottom_y"])),
        (12, max(20, safe_y - 8)), cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (0, 255, 255), 2,
    )
    cv2.putText(image, "L target", (max(4, left_target - 38), 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 80, 255), 1)
    cv2.putText(image, "R target", (min(width - 70, right_target + 5), 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 180, 255), 1)

    posts = goalposts or []
    for cx, cy, box_w, box_h, confidence in posts:
        x1 = int(cx - box_w / 2)
        y1 = int(cy - box_h / 2)
        x2 = int(cx + box_w / 2)
        y2 = int(cy + box_h / 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.circle(image, (int(cx), int(cy)), 3, (255, 0, 255), -1)
        cv2.putText(image, "Post {:.0f}%".format(float(confidence) * 100.0),
                    (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 0, 255), 1)
    goal_center = info.get("goalpost_center_x")
    if goal_center is not None:
        gx = int(round(float(goal_center)))
        cv2.line(image, (gx, 0), (gx, height), (255, 0, 255), 2)
        cv2.putText(image, "goal center", (max(4, gx + 4), 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)

    line = info.get("line")
    if line:
        p1 = tuple(map(int, line.get("extended_p1", (0, 0))))
        p2 = tuple(map(int, line.get("extended_p2", (width - 1, height - 1))))
        cv2.line(image, p1, p2, (0, 0, 255), 3)
        angle = info.get("line_angle_deg")
        line_text = "red line angle={} deg pixels={} mode={}".format(
            "--" if angle is None else "{:.1f}".format(float(angle)),
            info.get("line_pixels", 0),
            info.get("line_adjustment_mode") or "fallback",
        )
        cv2.putText(image, line_text, (12, 108), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (0, 0, 255), 2)

    if ball is None:
        cv2.putText(image, "NO FOOTBALL", (12, 55), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)
    else:
        cx, cy, box_w, box_h, confidence = ball
        x1 = int(cx - box_w / 2)
        y1 = int(cy - box_h / 2)
        x2 = int(cx + box_w / 2)
        y2 = int(cy + box_h / 2)
        color = (0, 255, 0) if info.get("target_ok") and info.get("distance_ok") else (0, 165, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        cv2.circle(image, (int(cx), int(cy)), 4, color, -1)
        cv2.putText(image, "Football {:.1f}%".format(float(confidence) * 100),
                    (x1, max(24, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        text = "bottom={:.1f} h={} dx={:.1f} target_dx={:.1f}".format(
            info.get("bottom_y", 0), int(box_h), info.get("dx", 0), info.get("target_dx", 0)
        )
        cv2.putText(image, text, (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)
    cv2.putText(image, "state={} action={} armed={}".format(
        info.get("controller_state", "IDLE"),
        info.get("controller_suggested_action") or info.get("suggested_action") or "--",
        "YES" if controller_armed else "NO",
    ), (12, height - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 2)
    cv2.putText(image, "cyan=center; yellow=safe_bottom_y; red=goal line; purple=posts", (12, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
    return image

def camera_worker():
    global latest_frame, latest_display, last_error
    if Camera is None:
        last_error = "hiwonder.Camera unavailable"
        return
    camera = Camera.Camera()
    camera.camera_open()
    time.sleep(1.0)
    while True:
        try:
            ret, frame = camera.read()
            if not ret or frame is None:
                last_error = "camera read failed"
                time.sleep(0.05)
                continue
            with frame_lock:
                latest_frame = frame.copy()
                ball = latest_ball
                posts = list(latest_goalposts)
                info = dict(latest_info)
            display = draw_debug(frame, ball, info, posts)
            with frame_lock:
                latest_display = display
            time.sleep(FRAME_INTERVAL)
        except Exception as exc:
            last_error = repr(exc)
            time.sleep(0.1)


def _preview_line(frame, posts):
    global preview_line_estimator
    if preview_line_estimator is None:
        preview_line_estimator = RedGoalLineEstimator()
    line, pixels = preview_line_estimator.estimate(frame, posts or [])
    angle = None
    if line and abs(float(line.get("dx", 0.0))) > 1e-6:
        angle = float(__import__("math").degrees(__import__("math").atan2(
            float(line.get("dy", 0.0)), float(line.get("dx", 0.0))
        )))
        while angle > 90.0:
            angle -= 180.0
        while angle < -90.0:
            angle += 180.0
    if line and pixels >= int(config["line_min_points"]):
        aa = abs(angle or 0.0)
        if aa <= float(config["line_flat_angle_deg"]):
            mode = "translate"
        elif aa >= float(config["line_turn_angle_deg"]):
            mode = "turn"
        else:
            mode = "ambiguous"
    else:
        mode = "unreliable"
    center = None
    if len(posts or []) >= 2:
        xs = sorted(float(post[0]) for post in posts)
        center = (xs[0] + xs[-1]) / 2.0
    return line, int(pixels or 0), angle, mode, center


def detection_worker():
    global latest_ball, latest_goalposts, latest_info, last_error
    global preview_controller
    try:
        detector = FootballDetector(conf_threshold=float(config["ball_confidence"]))
        post_detector = GoalPostDetector(conf_threshold=0.30)
        preview_controller = FootballKickController(
            DebugMotion(), detector=detector,
            line_flat_angle_deg=config["line_flat_angle_deg"],
            line_turn_angle_deg=config["line_turn_angle_deg"],
            line_confirm_frames=config["line_confirm_frames"],
            line_min_points=config["line_min_points"],
            goalpost_center_tolerance_px=config["goalpost_center_tolerance_px"],
            goalpost_turn_threshold_px=config["goalpost_turn_threshold_px"],
        )
    except Exception as exc:
        last_error = "detector init failed: {}".format(exc)
        return
    while True:
        try:
            with frame_lock:
                frame = None if latest_frame is None else latest_frame.copy()
            if frame is None:
                time.sleep(0.05)
                continue
            ball = detector.detect(frame)
            posts = post_detector.detect(frame)
            info = metrics_for(ball, frame.shape[1])
            line, pixels, angle, mode, center = _preview_line(frame, posts)
            info.update({
                "goalpost_count": len(posts),
                "goalpost_center_x": None if center is None else round(center, 1),
                "line": line,
                "line_pixels": pixels,
                "line_angle_deg": None if angle is None else round(angle, 2),
                "line_reliable": bool(line is not None and pixels >= int(config["line_min_points"])),
                "line_adjustment_mode": mode,
                "controller_state": "IDLE",
                "controller_suggested_action": None,
            })
            if controller_enabled and preview_controller is not None:
                _sync_controller_config(preview_controller)
                cinfo = preview_controller.update(frame, ball=ball, goalposts=posts)
                info.update({
                    "controller_state": cinfo.get("state"),
                    "controller_message": cinfo.get("message"),
                    "controller_suggested_action": cinfo.get("last_action"),
                    "controller_last_action_reason": cinfo.get("last_action_reason"),
                    "controller_resolved": cinfo.get("resolved"),
                    "controller_crossed": cinfo.get("crossed"),
                    "line": cinfo.get("line") or line,
                    "line_pixels": cinfo.get("line_pixels", pixels),
                    "line_angle_deg": cinfo.get("line_angle_deg", angle),
                    "line_reliable": cinfo.get("line_reliable", info["line_reliable"]),
                    "line_adjustment_mode": cinfo.get("line_adjustment_mode") or mode,
                    "goal_result": cinfo.get("goal_result"),
                })
            with frame_lock:
                latest_ball = ball
                latest_goalposts = posts
                latest_info = info
        except Exception as exc:
            last_error = repr(exc)
        time.sleep(0.05)

def mjpeg_stream():
    while True:
        with frame_lock:
            frame = None if latest_display is None else latest_display.copy()
        if frame is None:
            time.sleep(0.05)
            continue
        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        time.sleep(FRAME_INTERVAL)


HTML = r"""
<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TonyPi 完整射门调试</title>
<style>
body{font-family:Arial,sans-serif;background:#172033;color:#eee;margin:0;padding:18px}
main{max-width:1450px;margin:auto}.grid{display:grid;grid-template-columns:minmax(0,2fr) minmax(360px,1fr);gap:18px}
.card{background:#222d44;border-radius:12px;padding:16px;box-shadow:0 3px 15px #0005;margin-bottom:18px}
img{width:100%;border-radius:8px}.row{display:flex;justify-content:space-between;gap:12px;margin:7px 0;border-bottom:1px solid #ffffff18;padding-bottom:6px}
label{display:block;margin:10px 0}.value{font-weight:bold;color:#74e6ff}.ok{color:#5cff79}.bad{color:#ff7676}.warn{color:#ffd166}
input[type=range]{width:100%}button{font-size:1rem;padding:11px;margin:4px 0;border:0;border-radius:8px;cursor:pointer;background:#0d8bf2;color:white;width:100%}
button.danger{background:#d9485f}button.safe{background:#65748b}button:disabled{background:#555;cursor:not-allowed}.hint{font-size:.9rem;color:#b8c4d8;line-height:1.45}
.badge{display:inline-block;padding:3px 8px;border-radius:12px;background:#46536b;margin:2px}.warning{background:#5b3b20;border-left:4px solid #ffd166;padding:10px;margin:8px 0}
</style></head>
<body><main><h1>TonyPi 完整射门调试页</h1>
<p class="hint">本页覆盖：找球 → 足球目标窗口对准 → 保守接近 → 球门柱辅助 → 红线姿态判断 → 射门 → 越线检查。默认是仿真模式，只显示状态机建议，不执行机器人动作。</p>
<div class="warning"><b>安全提示：</b>只有点击“允许机器人动作”后，调试状态机才会真正调用动作组。未允许时，所有动作只记录为建议；正式场地测试请先保持仿真模式。</div>
<div class="grid"><div>
<div class="card"><img src="/video_feed"></div>
<div class="card"><h2>状态机控制</h2>
<div class="row"><span>状态机</span><span id="controller_state">IDLE</span></div>
<div class="row"><span>当前信息</span><span id="controller_message">--</span></div>
<div class="row"><span>建议/上一次动作</span><span id="controller_action">--</span></div>
<div class="row"><span>红线策略</span><span id="line_mode">--</span></div>
<div class="row"><span>是否允许真实动作</span><span id="armed">否</span></div>
<button id="start_controller">启动状态机（仿真）</button>
<button id="stop_controller" class="safe">停止状态机</button>
<button id="arm_controller" class="danger">允许机器人动作：否（点击切换）</button>
<div id="controller_result" class="hint"></div>
</div>
<div class="card"><h2>手动动作验证</h2>
<button id="kick_gate">按当前条件执行射门</button>
<button id="kick_manual" class="danger">强制手动射门（仅要求检测到足球）</button>
<div id="kick_result" class="hint"></div></div>
</div>
<div>
<div class="card"><h2>实时视觉与判断</h2><div id="message">等待画面...</div>
<div class="row"><span>足球</span><span id="detected">--</span></div>
<div class="row"><span>足球 cx / 中心偏差</span><span id="dx">--</span></div>
<div class="row"><span>射门脚目标 x / 偏差</span><span id="target">--</span></div>
<div class="row"><span>bottom_y / 检测框高度 h</span><span id="distance_values">--</span></div>
<div class="row"><span>距离条件</span><span id="distance_ok">--</span></div>
<div class="row"><span>连续确认</span><span id="streak">--</span></div>
<div class="row"><span>宽走廊 / 最终窗口</span><span id="window_ok">--</span></div>
<div class="row"><span>球门柱数量 / 中心 x</span><span id="posts">--</span></div>
<div class="row"><span>红线角度 / 像素数</span><span id="line_angle">--</span></div>
<div class="row"><span>红线可靠性</span><span id="line_reliable">--</span></div>
<div class="row"><span>推荐动作</span><span id="action">--</span></div>
</div>
<div class="card"><h2>可调参数</h2>
<label>left_kick_target_offset_px: <span class="value" id="left_kick_target_offset_px_v"></span><input id="left_kick_target_offset_px" type="range" min="-150" max="80" step="1"></label>
<label>right_kick_target_offset_px: <span class="value" id="right_kick_target_offset_px_v"></span><input id="right_kick_target_offset_px" type="range" min="-80" max="150" step="1"></label>
<label>kick_target_tolerance_px: <span class="value" id="target_tolerance_px_v"></span><input id="target_tolerance_px" type="range" min="10" max="100" step="1"></label>
<label>kick_lane_tolerance_px: <span class="value" id="lane_tolerance_px_v"></span><input id="lane_tolerance_px" type="range" min="20" max="180" step="1"></label>
<label>safe_bottom_y: <span class="value" id="safe_bottom_y_v"></span><input id="safe_bottom_y" type="range" min="250" max="450" step="1"></label>
<label>safe_confirm_frames: <span class="value" id="safe_confirm_frames_v"></span><input id="safe_confirm_frames" type="range" min="1" max="8" step="1"></label>
<label>红线水平阈值（度）: <span class="value" id="line_flat_angle_deg_v"></span><input id="line_flat_angle_deg" type="range" min="2" max="20" step="1"></label>
<label>红线倾斜阈值（度）: <span class="value" id="line_turn_angle_deg_v"></span><input id="line_turn_angle_deg" type="range" min="4" max="30" step="1"></label>
<div class="hint">h 只用于观察，不参与距离硬判定。红线接近水平时建议横移，明显倾斜时建议转身；临界区保留上一策略或回退。</div>
</div></div></div>
<script>
const names=['left_kick_target_offset_px','right_kick_target_offset_px','target_tolerance_px','lane_tolerance_px','safe_bottom_y','safe_confirm_frames','line_flat_angle_deg','line_turn_angle_deg'];
function setText(id,v,cls){const e=document.getElementById(id);e.textContent=v;e.className=cls||''}
function applyConfig(data){for(const n of names){if(data[n]!==undefined){const e=document.getElementById(n);e.value=data[n];document.getElementById(n+'_v').textContent=data[n]}}}
function update(){fetch('/status').then(r=>r.json()).then(d=>{
 if(d.config)applyConfig(d.config);
 setText('message',d.controller_message||d.message||d.last_error||'--');
 setText('controller_state',d.controller_state||'IDLE',d.controller_state==='RESULT'?'ok':'');
 setText('controller_message',d.controller_message||'未启动状态机');
 setText('controller_action',(d.controller_suggested_action||d.last_controller_action||'--')+(d.controller_armed?'（真实）':'（仿真）'));
 setText('armed',d.controller_armed?'是':'否',d.controller_armed?'bad':'ok');
 setText('detected',d.detected?'是':'否',d.detected?'ok':'bad');
 setText('dx',d.dx??'--'); setText('target',(d.target_x??'--')+' / '+(d.target_dx??'--'));
 setText('distance_values',(d.bottom_y??'--')+' / '+(d.h??'--'));
 setText('distance_ok',d.distance_ok?'通过':'未通过',d.distance_ok?'ok':'bad');
 setText('streak',(d.safe_streak||0)+' / '+(d.config?.safe_confirm_frames||'--'),d.confirmed_ok?'ok':'warn');
 setText('window_ok',(d.lane_ok?'宽走廊通过':'宽走廊未通过')+' / '+(d.target_ok?'最终通过':'最终未通过'),d.target_ok?'ok':'warn');
 setText('posts',(d.goalpost_count||0)+' / '+(d.goalpost_center_x??'--'));
 setText('line_angle',(d.line_angle_deg??'--')+'° / '+(d.line_pixels||0));
 setText('line_reliable',d.line_reliable?'可靠':'不可靠',d.line_reliable?'ok':'warn');
 setText('line_mode',(d.line_adjustment_mode||'unreliable')+'（平移/转身策略）');
 setText('action',d.controller_suggested_action||d.suggested_action||'--');
 }).catch(()=>{})}
for(const n of names){document.getElementById(n).addEventListener('input',()=>{document.getElementById(n+'_v').textContent=document.getElementById(n).value;const body={};body[n]=Number(document.getElementById(n).value);fetch('/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})})}
function post(url,body){return fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})}).then(r=>r.json())}
document.getElementById('start_controller').onclick=()=>post('/controller/start').then(d=>document.getElementById('controller_result').textContent=d.message||JSON.stringify(d));
document.getElementById('stop_controller').onclick=()=>post('/controller/stop').then(d=>document.getElementById('controller_result').textContent=d.message||JSON.stringify(d));
document.getElementById('arm_controller').onclick=()=>{fetch('/status').then(r=>r.json()).then(d=>post('/controller/arm',{armed:!d.controller_armed})).then(d=>document.getElementById('controller_result').textContent=d.message||JSON.stringify(d)))};
function doKick(enforce){if(!confirm('确认执行一次正式射门动作？请确认机器人周围和球门方向安全。'))return;const b1=document.getElementById('kick_gate'),b2=document.getElementById('kick_manual');b1.disabled=true;b2.disabled=true;fetch('/kick?enforce='+(enforce?'1':'0'),{method:'POST'}).then(r=>r.json()).then(d=>document.getElementById('kick_result').textContent=d.message||JSON.stringify(d)).catch(e=>document.getElementById('kick_result').textContent=String(e)).finally(()=>{b1.disabled=false;b2.disabled=false})}
document.getElementById('kick_gate').onclick=()=>doKick(true);document.getElementById('kick_manual').onclick=()=>doKick(false);
update();setInterval(update,500);
</script></body></html>
"""




@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/video_feed")
def video_feed():
    return Response(mjpeg_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


def _sync_controller_config(controller):
    controller.ball_safe_bottom_y = float(config["safe_bottom_y"])
    controller.ball_safe_confirm_frames = max(1, int(config["safe_confirm_frames"]))
    controller.kick_target_tolerance_px = max(1.0, float(config["target_tolerance_px"]))
    controller.kick_lane_tolerance_px = max(
        controller.kick_target_tolerance_px, float(config["lane_tolerance_px"])
    )
    controller.ball_center_tolerance_px = controller.kick_target_tolerance_px
    controller.ball_lane_tolerance_px = controller.kick_lane_tolerance_px
    controller.left_kick_target_offset_px = float(config["left_kick_target_offset_px"])
    controller.right_kick_target_offset_px = float(config["right_kick_target_offset_px"])
    controller.line_flat_angle_deg = float(config["line_flat_angle_deg"])
    controller.line_turn_angle_deg = max(
        controller.line_flat_angle_deg + 0.1, float(config["line_turn_angle_deg"])
    )
    controller.line_min_points = max(5, int(config["line_min_points"]))
    controller.line_confirm_frames = max(1, int(config["line_confirm_frames"]))
    controller.goalpost_center_tolerance_px = float(config["goalpost_center_tolerance_px"])
    controller.goalpost_turn_threshold_px = max(
        controller.goalpost_center_tolerance_px,
        float(config["goalpost_turn_threshold_px"]),
    )


@app.route("/status")
def status():
    with frame_lock:
        data = dict(latest_info)
    data["config"] = dict(config)
    data["last_action"] = last_action
    data["last_action_at"] = last_action_at
    data["last_error"] = last_error
    data["controller_enabled"] = bool(controller_enabled)
    data["controller_armed"] = bool(controller_armed)
    data["last_controller_action"] = controller_last_action
    data["last_controller_action_at"] = controller_last_action_at
    return jsonify(data)


@app.route("/controller/start", methods=["POST"])
def controller_start():
    global controller_enabled
    if preview_controller is None:
        return jsonify({"ok": False, "message": "状态机尚未初始化，请等待摄像头线程启动"}), 503
    _sync_controller_config(preview_controller)
    preview_controller.start()
    controller_enabled = True
    return jsonify({"ok": True, "message": "已启动状态机仿真；当前不会执行真实动作"})


@app.route("/controller/stop", methods=["POST"])
def controller_stop():
    global controller_enabled, controller_armed
    controller_enabled = False
    controller_armed = False
    if preview_controller is not None:
        preview_controller.reset()
        # 即使状态机正在仿真，复位时也记录 stand；若此前已授权，
        # DebugMotion 会把它发送给机器人，避免停在前进/转身动作中间。
        preview_controller.motion.run("stand")
    return jsonify({"ok": True, "message": "已停止状态机并关闭真实动作"})


@app.route("/controller/arm", methods=["POST"])
def controller_arm():
    global controller_armed
    payload = request.get_json(silent=True) or {}
    controller_armed = bool(payload.get("armed", False))
    return jsonify({
        "ok": True,
        "controller_armed": controller_armed,
        "message": ("已允许状态机执行真实动作，请确认场地安全" if controller_armed
                     else "已关闭真实动作，恢复仿真模式"),
    })


@app.route("/config", methods=["POST"])
def update_config():
    payload = request.get_json(silent=True) or {}
    allowed = {"safe_bottom_y", "safe_height_px", "safe_confirm_frames",
               "lane_tolerance_px", "target_tolerance_px",
               "left_kick_target_offset_px", "right_kick_target_offset_px",
               "line_flat_angle_deg", "line_turn_angle_deg",
               "line_min_points", "line_confirm_frames",
               "goalpost_center_tolerance_px", "goalpost_turn_threshold_px",
               # 旧页面字段：写入时同步到新的目标窗口阈值。
               "center_tolerance_px"}
    for key in allowed:
        if key not in payload:
            continue
        value = float(payload[key])
        if key == "safe_confirm_frames":
            value = max(1, min(8, int(value)))
        elif key == "safe_bottom_y":
            value = max(250.0, min(450.0, value))
        elif key == "safe_height_px":
            value = max(40.0, min(140.0, value))
        elif key in ("left_kick_target_offset_px", "right_kick_target_offset_px"):
            value = max(-150.0, min(150.0, value))
        elif key in ("target_tolerance_px", "center_tolerance_px"):
            value = max(10.0, min(100.0, value))
            if key == "center_tolerance_px":
                config["target_tolerance_px"] = value
                continue
        elif key == "line_flat_angle_deg":
            value = max(2.0, min(20.0, value))
        elif key == "line_turn_angle_deg":
            value = max(4.0, min(30.0, value))
        elif key == "line_min_points":
            value = max(5.0, min(500.0, int(value)))
        elif key == "line_confirm_frames":
            value = max(1.0, min(8.0, int(value)))
        elif key == "goalpost_center_tolerance_px":
            value = max(10.0, min(150.0, value))
        elif key == "goalpost_turn_threshold_px":
            value = max(30.0, min(250.0, value))
        else:
            value = max(20.0, min(180.0, value))
        config[key] = value
    return jsonify({"ok": True, "config": config})


@app.route("/kick", methods=["POST"])
def kick():
    global last_action, last_action_at
    enforce = request.args.get("enforce", "1") != "0"
    if AGC is None:
        return jsonify({"ok": False, "message": "ActionGroupControl 不可用"}), 500
    if not action_lock.acquire(blocking=False):
        return jsonify({"ok": False, "message": "已有动作正在执行，请稍候"}), 409
    try:
        with frame_lock:
            ball = latest_ball
            info = dict(latest_info)
        if ball is None or not info.get("detected"):
            return jsonify({"ok": False, "message": "当前没有足球检测框，拒绝射门"}), 400
        if enforce and not (info.get("confirmed_ok") and info.get("target_ok")):
            return jsonify({"ok": False, "message": "当前距离或射门脚目标窗口尚未通过，未执行射门；可调参数或使用强制手动射门"}), 400
        action = info.get("suggested_action")
        if action not in ("left_shot_fast", "right_shot_fast"):
            return jsonify({"ok": False, "message": "无法根据当前足球位置决定左右脚动作"}), 400
        AGC.runActionGroup(action)
        last_action = action
        last_action_at = time.time()
        return jsonify({"ok": True, "action": action,
                        "message": "已执行 {}；本页不会自动判断进球结果".format(action)})
    except Exception as exc:
        return jsonify({"ok": False, "message": repr(exc)}), 500
    finally:
        action_lock.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5002)
    args = parser.parse_args()
    threading.Thread(target=camera_worker, daemon=True).start()
    threading.Thread(target=detection_worker, daemon=True).start()
    print("football_kick_debug_web: http://<robot-ip>:{}".format(args.port), flush=True)
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
