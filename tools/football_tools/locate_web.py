#!/usr/bin/python3
# coding=utf8
"""
locate_web.py — AprilTag 定位可视化网站

功能：
  - 显示 1m×1m 场地俯视图
  - 标注 load_pos.py 当前的全部 AprilTag 位置
  - 实时显示机器人位置和朝向
  - 端口 5001

运行：
  sudo systemctl stop tonypi
  python3 locate_web.py
  浏览器访问：http://<机器人IP>:5001
"""

import sys
import time
import threading
import json

import numpy as np
from flask import Flask, Response, render_template_string, request
import cv2
import hiwonder.Camera as Camera
import hiwonder.ros_robot_controller_sdk as rrc
import hiwonder.yaml_handle as yaml_handle
from hiwonder.Controller import Controller

from locate_robot import RobotLocator, FIXED_TAG_IDS
from load_pos import load_tag_pos

# 加载舵机配置
servo_data = yaml_handle.get_yaml_data(yaml_handle.servo_file_path)

# ============ 配置 ============
WEB_PORT = 5001
STREAM_QUALITY = 75
UPDATE_INTERVAL = 0.5    # 秒

# 场地坐标范围（cm，以场地中心为原点(0,0)）
# 场地尺寸：100cm(宽) × 100cm(高)
FIELD_X_MIN = -50
FIELD_X_MAX =  50
FIELD_Y_MIN = -50
FIELD_Y_MAX =  50

# --- 头部搜索/稳定定位参数 ---
# servo1：当前标定约定下，脉宽增大是抬头，减小是低头。
# 搜索时比正式定位姿态抬高一些，但仍低于平视，减少身体遮挡。
HEAD_TILT_LOCALIZE = servo_data['servo1'] - 30
HEAD_TILT_SEARCH = servo_data['servo1'] - 10
HEAD_TILT_LOOK_DOWN = HEAD_TILT_LOCALIZE  # 兼容旧变量名

HEAD_PAN_CENTER = servo_data['servo2']
# servo2 的脉宽方向与人直观左右相反：脉宽增大=机器人向左转，减小=向右转。
HEAD_PAN_LEFT  = servo_data['servo2'] + 500
HEAD_PAN_RIGHT = servo_data['servo2'] - 500
HEAD_PAN_STEP  = 200
HEAD_SWEEP_INTERVAL = 2.0       # 每个搜索位置停留秒数
HEAD_SETTLE_SECONDS = 1.5        # 回到固定定位姿态后的稳定等待时间
MIN_TAGS_TO_LOCK = 2             # 自动搜索阶段至少看到这么多有效 Tag 才锁定
STABLE_FRAMES_TO_LOCK = 4        # 连续若干帧满足条件，避免单帧误锁定
LOST_FRAMES_TO_RESTART = 6       # 稳定定位连续丢失后重新搜索
# ==================================

# --- 舵机/板子初始化 ---
board = rrc.Board()
ctl = Controller(board)

# 头部当前命令值及定位状态
head_x = servo_data['servo2']
head_y = HEAD_TILT_LOCALIZE
auto_sweep_mode = True
localization_state = 'AUTO_SEARCH'
head_lock = threading.Lock()

app = Flask(__name__)

# 全局状态
latest_frame = None
latest_result = None
latest_diagnostics = {
    'raw_tag_ids': [],
    'valid_tag_ids': [],
    'detections': [],
}
frame_lock = threading.Lock()

# Tag 中心坐标（用于地图显示），始终直接从 load_pos.py 读取
TAG_CENTERS = {}


def compute_tag_centers():
    """从 load_pos.py 当前数据计算所有 Tag 的中心坐标。"""
    global TAG_CENTERS
    TAG_CENTERS = {}
    all_poses = load_tag_pos()
    for key, corners in all_poses.items():
        tid = int(key)
        TAG_CENTERS[tid] = (
            float(corners[:, 0].mean()),
            float(corners[:, 1].mean()),
        )


def set_head(pan=None, tilt=None, duration=300):
    """设置头部目标脉宽，并同步网页显示值。"""
    global head_x, head_y
    with head_lock:
        if pan is not None:
            head_x = int(pan)
        if tilt is not None:
            head_y = int(tilt)
        current_x, current_y = head_x, head_y

    ctl.set_pwm_servo_pulse(2, current_x, duration)
    ctl.set_pwm_servo_pulse(1, current_y, duration)


def get_head_status():
    with head_lock:
        return head_x, head_y


def camera_and_locate_thread():
    """采图、显示所有 Tag，并执行自动搜索→固定姿态稳定定位状态机。"""
    global latest_frame, latest_result, latest_diagnostics
    global auto_sweep_mode, localization_state

    # 自动搜索初始姿态：抬高一点但仍低于平视，避免身体遮挡左右视线。
    set_head(HEAD_PAN_CENTER, HEAD_TILT_SEARCH, 500)
    time.sleep(0.8)

    locator = RobotLocator()

    cam = Camera.Camera()
    cam.camera_open()
    time.sleep(1.0)
    print(f"[locate_web] 摄像头已开启，访问 http://<机器人IP>:{WEB_PORT}")
    print("[locate_web] 自动模式：高一点的俯仰角左右扫视，找到足够 Tag 后回中稳定定位")
    print("[locate_web] 手动模式：方向键调头，定位结果仅作当前头部姿态调试参考")

    # 左右搜索顺序：中心 → 左 → 中 → 右。
    sweep_positions = [HEAD_PAN_CENTER, HEAD_PAN_LEFT,
                       HEAD_PAN_CENTER, HEAD_PAN_RIGHT]
    sweep_idx = 0
    last_sweep_move = time.time()
    stable_frames = 0
    lost_frames = 0
    settle_until = 0.0
    was_auto = auto_sweep_mode

    while True:
        now = time.time()

        # Tab/API 切回自动模式后，从搜索姿态重新开始。
        if auto_sweep_mode and not was_auto:
            localization_state = 'AUTO_SEARCH'
            stable_frames = 0
            lost_frames = 0
            sweep_idx = 0
            set_head(HEAD_PAN_CENTER, HEAD_TILT_SEARCH, 300)
            last_sweep_move = now
        was_auto = auto_sweep_mode

        # 自动模式的云台状态机。
        if auto_sweep_mode:
            if localization_state == 'MANUAL':
                localization_state = 'AUTO_SEARCH'
                stable_frames = 0
                set_head(HEAD_PAN_CENTER, HEAD_TILT_SEARCH, 300)
                last_sweep_move = now

            if (localization_state == 'AUTO_SEARCH'
                    and now - last_sweep_move >= HEAD_SWEEP_INTERVAL):
                sweep_idx = (sweep_idx + 1) % len(sweep_positions)
                set_head(sweep_positions[sweep_idx], HEAD_TILT_SEARCH, 300)
                last_sweep_move = now

        else:
            localization_state = 'MANUAL'

        ret, frame = cam.read()
        if not ret or frame is None:
            time.sleep(0.05)
            continue

        result = locator.locate_once(frame)
        diagnostics = locator.get_last_diagnostics()
        valid_count = len(diagnostics['valid_tag_ids'])

        # 自动搜索：连续多帧看到足够有效 Tag 后，回中并切换到正式定位俯仰角。
        if auto_sweep_mode and localization_state == 'AUTO_SEARCH':
            if valid_count >= MIN_TAGS_TO_LOCK and result is not None:
                stable_frames += 1
            else:
                stable_frames = 0

            if stable_frames >= STABLE_FRAMES_TO_LOCK:
                localization_state = 'AUTO_SETTLING'
                stable_frames = 0
                lost_frames = 0
                set_head(HEAD_PAN_CENTER, HEAD_TILT_LOCALIZE, 400)
                settle_until = now + HEAD_SETTLE_SECONDS

        elif auto_sweep_mode and localization_state == 'AUTO_SETTLING':
            if now >= settle_until:
                localization_state = 'AUTO_STABLE'
                lost_frames = 0

        elif auto_sweep_mode and localization_state == 'AUTO_STABLE':
            if valid_count < MIN_TAGS_TO_LOCK or result is None:
                lost_frames += 1
            else:
                lost_frames = 0
            if lost_frames >= LOST_FRAMES_TO_RESTART:
                localization_state = 'AUTO_SEARCH'
                stable_frames = 0
                lost_frames = 0
                sweep_idx = 0
                set_head(HEAD_PAN_CENTER, HEAD_TILT_SEARCH, 300)
                last_sweep_move = now

        # 正式自动定位只在固定头部姿态稳定后发布；手动模式用于调试当前姿态。
        if (not auto_sweep_mode) or localization_state == 'AUTO_STABLE':
            published_result = result
        else:
            published_result = None

        # 在视频上画出所有原始检测框；绿色=load_pos.py 中有效，红色=未纳入定位。
        display = frame.copy()
        for item in diagnostics['detections']:
            pts = np.asarray(item['corners'], dtype=np.int32).reshape(-1, 1, 2)
            color = (0, 220, 0) if item['valid'] else (0, 0, 255)
            cv2.polylines(display, [pts], True, color, 2)
            x0, y0 = pts[0, 0]
            cv2.putText(display, f"#{item['id']}", (int(x0), int(y0) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if published_result:
            tags_str = ','.join(str(t) for t in published_result['tags_used'])
            txt = (f"({published_result['x']:.0f},{published_result['y']:.0f})cm "
                   f"yaw={published_result['yaw_deg']:+.0f} "
                   f"{published_result['confidence']} "
                   f"tags={len(published_result['tags_used'])}:{tags_str}")
            cv2.putText(display, txt, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 80), 2)
        else:
            cv2.putText(display, "No stable pose", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 80, 255), 2)

        head_now_x, head_now_y = get_head_status()
        state_txt = (f"state={localization_state} head=({head_now_x},{head_now_y}) "
                     f"raw={len(diagnostics['raw_tag_ids'])} "
                     f"valid={valid_count}")
        cv2.putText(display, state_txt, (10, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 220, 0), 1)

        with frame_lock:
            latest_frame = display
            latest_result = published_result
            latest_diagnostics = diagnostics

        time.sleep(UPDATE_INTERVAL)

def generate_mjpeg():
    while True:
        with frame_lock:
            frame = latest_frame
        if frame is None:
            time.sleep(0.05)
            continue
        _, buf = cv2.imencode('.jpg', frame,
                              [cv2.IMWRITE_JPEG_QUALITY, STREAM_QUALITY])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + buf.tobytes() + b'\r\n')
        time.sleep(0.05)


# ============ Flask 路由 ============

@app.route('/')
def index():
    tag_data = [{'id': tid, 'x': cx, 'y': cy}
                for tid, (cx, cy) in TAG_CENTERS.items()]
    return render_template_string(
        HTML_PAGE,
        tag_data=json.dumps(tag_data),
        x_min=FIELD_X_MIN, x_max=FIELD_X_MAX,
        y_min=FIELD_Y_MIN, y_max=FIELD_Y_MAX,
    )

@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def api_status():
    with frame_lock:
        r = latest_result
        d = latest_diagnostics
    current_x, current_y = get_head_status()
    payload = {
        'found':            r is not None,
        'head_x':           current_x,
        'head_y':           current_y,
        'auto_sweep':       auto_sweep_mode,
        'localization_state': localization_state,
        'raw_tag_ids':      d['raw_tag_ids'],
        'valid_tag_ids':    d['valid_tag_ids'],
        'raw_tag_count':    len(d['raw_tag_ids']),
        'valid_tag_count':  len(d['valid_tag_ids']),
    }
    if r is not None:
        payload.update({
            'x':            round(r['x'], 1),
            'y':            round(r['y'], 1),
            'yaw_deg':      round(r['yaw_deg'], 1),
            'confidence':   r['confidence'],
            'tags_used':    r['tags_used'],
            'reproj_error': round(r['reproj_error'], 3),
        })
    return json.dumps(payload), 200, {'Content-Type': 'application/json'}


@app.route('/api/head')
def api_head_move():
    """键盘调试：手动控制头部俯仰/旋转。
       dx=+1 为机器人左转头，dx=-1 为机器人右转头；
       dy=+1 抬头，dy=-1 低头。"""
    global auto_sweep_mode, localization_state
    dx = int(request.args.get('dx', 0))
    dy = int(request.args.get('dy', 0))
    reset_head = request.args.get('reset', '0') == '1'

    current_x, current_y = get_head_status()
    step = 50
    if reset_head:
        new_x, new_y = HEAD_PAN_CENTER, HEAD_TILT_LOCALIZE
    else:
        new_x = current_x + dx * step
        new_y = current_y + dy * step
    new_x = max(500, min(2500, new_x))
    new_y = max(800, min(2200, new_y))

    # 手动调头应立即接管自动状态机，避免后台线程下一帧把头抢回去。
    auto_sweep_mode = False
    localization_state = 'MANUAL'
    set_head(new_x, new_y, 200)
    return json.dumps({
        'head_x': new_x,
        'head_y': new_y,
        'auto_sweep': auto_sweep_mode,
        'localization_state': localization_state,
    }), 200, {'Content-Type': 'application/json'}


@app.route('/api/auto_sweep')
def api_auto_sweep():
    """切换自动搜索/固定姿态定位模式。?mode=on/?mode=off"""
    global auto_sweep_mode, localization_state
    mode = request.args.get('mode', 'toggle')
    if mode == 'on':
        auto_sweep_mode = True
    elif mode == 'off':
        auto_sweep_mode = False
    else:
        auto_sweep_mode = not auto_sweep_mode

    if auto_sweep_mode:
        localization_state = 'AUTO_SEARCH'
    else:
        localization_state = 'MANUAL'
    print(f"[locate_web] 自动扫视: {'ON' if auto_sweep_mode else 'OFF'}")
    return json.dumps({
        'auto_sweep': auto_sweep_mode,
        'localization_state': localization_state,
    }), 200, {'Content-Type': 'application/json'}


# ============ HTML 页面 ============
HTML_PAGE = """
<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>TonyPi 定位可视化</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #111; color: #eee; font-family: 'Segoe UI', sans-serif; }
h1 { text-align: center; padding: 12px; color: #0df; font-size: 1.3em; letter-spacing: 2px; }
.main { display: flex; flex-wrap: wrap; gap: 16px; justify-content: center; padding: 0 16px 20px; }
.panel { background: #1a1a2e; border-radius: 10px; padding: 12px;
         box-shadow: 0 4px 16px rgba(0,0,0,.5); }
canvas { display: block; }
.video-box img { width: 480px; max-width: 100%; border-radius: 6px; }
.info h2 { color: #0df; font-size: 1em; margin-bottom: 10px; }
.row { display: flex; justify-content: space-between; padding: 6px 0;
       border-bottom: 1px solid #333; font-size: .9em; }
.row .lbl { color: #aaa; }
.row .val { font-weight: bold; }
.conf-HIGH   { color: #4f4; }
.conf-MEDIUM { color: #ff4; }
.conf-LOW    { color: #f84; }
.conf-none   { color: #888; }
#tag-list { margin-top: 8px; font-size: .8em; color: #0df; }
</style>
</head>
<body>
<h1>TonyPi AprilTag 定位</h1>
<div class="main">
  <!-- 场地地图 -->
  <div class="panel">
    <canvas id="map" width="300" height="300"></canvas>
  </div>

  <!-- 摄像头 -->
  <div class="panel video-box">
    <img src="/video_feed" alt="Camera">
  </div>

  <!-- 状态面板 -->
  <div class="panel info" style="min-width:240px">
    <h2>定位状态</h2>
    <div class="row"><span class="lbl">X (cm)</span><span class="val" id="vx">--</span></div>
    <div class="row"><span class="lbl">Y (cm)</span><span class="val" id="vy">--</span></div>
    <div class="row"><span class="lbl">偏航角</span><span class="val" id="vyaw">--</span></div>
    <div class="row"><span class="lbl">置信度</span><span class="val" id="vconf">--</span></div>
    <div class="row"><span class="lbl">重投影误差</span><span class="val" id="vreproj">--</span></div>
    <div class="row"><span class="lbl">用于定位 Tag</span><span class="val" id="vntags">--</span></div>
    <div class="row"><span class="lbl">定位状态</span><span class="val" id="vstate">--</span></div>
    <div class="row"><span class="lbl">原始/有效 Tag</span><span class="val" id="vtagcounts">--</span></div>
    <div id="tag-list"></div>
    <div class="row" style="border-bottom:none;margin-top:8px;color:#0df;font-size:.85em">
      <span class="lbl">头部 PWM</span><span class="val" id="vhead">--</span>
    </div>
    <div style="font-size:.7em;color:#666;margin-top:6px">
      ↑↓ 俯仰 &nbsp; ←→ 旋转 &nbsp; 空格 复位 &nbsp; Tab 切换扫视
    </div>
  </div>
</div>

<script>
// ===== 场地 & Tag 数据（由 Python 注入）=====
const TAGS  = {{ tag_data | safe }};
const X_MIN = {{ x_min }};   // -50
const X_MAX = {{ x_max }};   // +50
const Y_MIN = {{ y_min }};   // -50
const Y_MAX = {{ y_max }};   // +50
const FW = X_MAX - X_MIN;    // 100 cm
const FH = Y_MAX - Y_MIN;    // 100 cm

// ===== 画布设置（1:1 正方形，宽300×高300）=====
const canvas = document.getElementById('map');
const ctx    = canvas.getContext('2d');
const PAD    = 42;
const CW = canvas.width  - PAD * 2;   // 可绘区宽 px
const CH = canvas.height - PAD * 2;   // 可绘区高 px

// 坐标映射：世界 cm → 画布 px（Y轴镜像：世界+Y朝上，画布Y朝下）
function wx(x) { return PAD + (x - X_MIN) / FW * CW; }
function wy(y) { return PAD + (Y_MAX - y) / FH * CH; }

// ===== 绘制场地 =====
function drawField() {
  ctx.fillStyle = '#0d1117';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // 网格线（每25cm一格）
  ctx.strokeStyle = '#223'; ctx.lineWidth = 1;
  ctx.fillStyle = '#445';   ctx.font = '10px sans-serif';
  for (let g = X_MIN; g <= X_MAX; g += 25) {
    ctx.beginPath(); ctx.moveTo(wx(g), wy(Y_MAX)); ctx.lineTo(wx(g), wy(Y_MIN)); ctx.stroke();
    ctx.fillText(g, wx(g) - (g < 0 ? 10 : 6), wy(Y_MIN) + 13);
  }
  for (let g = Y_MIN; g <= Y_MAX; g += 25) {
    ctx.beginPath(); ctx.moveTo(wx(X_MIN), wy(g)); ctx.lineTo(wx(X_MAX), wy(g)); ctx.stroke();
    ctx.fillText(g, PAD - 34, wy(g) + 4);
  }

  // 原点十字虚线
  ctx.strokeStyle = '#f804'; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(wx(0), wy(Y_MAX)); ctx.lineTo(wx(0), wy(Y_MIN)); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(wx(X_MIN), wy(0)); ctx.lineTo(wx(X_MAX), wy(0)); ctx.stroke();
  ctx.setLineDash([]);

  // 原点标记
  ctx.fillStyle = '#f80'; ctx.font = 'bold 10px sans-serif';
  ctx.fillText('(0,0)', wx(0) + 4, wy(0) - 4);

  // 场地外框
  ctx.strokeStyle = '#05f'; ctx.lineWidth = 2;
  ctx.strokeRect(wx(X_MIN), wy(Y_MAX), CW, CH);

  // 轴标签
  ctx.fillStyle = '#05f'; ctx.font = 'bold 11px sans-serif';
  ctx.fillText('+X →', wx(X_MAX) - 30, wy(Y_MIN) + 28);
  ctx.fillText('+Y ↑', wx(X_MIN) - 2, wy(Y_MAX) - 6);
  ctx.fillStyle = '#888'; ctx.font = '10px sans-serif';
  ctx.fillText(`${FW}×${FH} cm`, wx(X_MIN) + 4, wy(Y_MAX) + 14);

  // 所有固定 Tag（青色圆点 + 编号）
  TAGS.forEach(t => {
    const px = wx(t.x), py = wy(t.y);
    ctx.fillStyle = '#0df';
    ctx.beginPath(); ctx.arc(px, py, 5, 0, Math.PI*2); ctx.fill();
    ctx.fillStyle = '#fff'; ctx.font = 'bold 9px sans-serif';
    ctx.fillText('#' + t.id, px + 7, py - 4);
  });
}

// ===== 机器人状态 =====
let robot = null;

function drawRobot() {
  if (!robot) return;
  const px = wx(robot.x);
  const py = wy(robot.y);
  const yawRad = -robot.yaw_deg * Math.PI / 180;  // 负号：y轴翻转补偿
  const arrowLen = 28;

  // 方向箭头
  ctx.strokeStyle = '#f40';
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(px, py);
  ctx.lineTo(px + Math.cos(yawRad) * arrowLen,
             py + Math.sin(yawRad) * arrowLen);
  ctx.stroke();

  // 箭头头部
  const hx = px + Math.cos(yawRad) * arrowLen;
  const hy = py + Math.sin(yawRad) * arrowLen;
  const a1 = yawRad + 2.5, a2 = yawRad - 2.5;
  ctx.beginPath();
  ctx.moveTo(hx, hy);
  ctx.lineTo(hx + Math.cos(a1)*8, hy + Math.sin(a1)*8);
  ctx.lineTo(hx + Math.cos(a2)*8, hy + Math.sin(a2)*8);
  ctx.closePath();
  ctx.fillStyle = '#f40'; ctx.fill();

  // 机器人圆点
  ctx.beginPath(); ctx.arc(px, py, 9, 0, Math.PI*2);
  ctx.fillStyle = '#f40'; ctx.fill();
  ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke();

  // 坐标标注
  ctx.fillStyle = '#fff'; ctx.font = '11px sans-serif';
  ctx.fillText(`(${robot.x.toFixed(0)},${robot.y.toFixed(0)})`, px+12, py+4);
}

function redraw() {
  drawField();
  drawRobot();
}

// ===== 状态更新 =====
function updateStatus() {
  fetch('/api/status')
    .then(r => r.json())
    .then(data => {
      // 无论当前是否已经形成稳定位姿，都实时显示头部实际命令值和 Tag 诊断。
      autoSweep = !!data.auto_sweep;
      document.getElementById('vhead').textContent =
        `servo1=${data.head_y}  servo2=${data.head_x}`;
      document.getElementById('vstate').textContent =
        (data.localization_state || '--') + (data.auto_sweep ? '（自动）' : '（手动）');
      document.getElementById('vtagcounts').textContent =
        `${data.raw_tag_count || 0}/${data.valid_tag_count || 0}`;
      document.getElementById('tag-list').textContent =
        '原始: [' + (data.raw_tag_ids || []).join(', ') + ']  ' +
        '有效: [' + (data.valid_tag_ids || []).join(', ') + ']';

      if (data.found) {
        robot = data;

        document.getElementById('vx').textContent    = data.x.toFixed(1) + ' cm';
        document.getElementById('vy').textContent    = data.y.toFixed(1) + ' cm';
        document.getElementById('vyaw').textContent  = data.yaw_deg.toFixed(1) + '°';
        document.getElementById('vntags').textContent = data.tags_used.length;
        document.getElementById('vreproj').textContent = data.reproj_error.toFixed(3) + ' px';

        const confEl = document.getElementById('vconf');
        confEl.textContent = data.confidence;
        confEl.className = 'val conf-' + data.confidence;
      } else {
        robot = null;
        ['vx','vy','vyaw','vntags','vreproj'].forEach(id =>
          document.getElementById(id).textContent = '--');
        const c = document.getElementById('vconf');
        c.textContent = '未形成稳定位姿'; c.className = 'val conf-none';
      }
      redraw();
    })
    .catch(() => {});
}

// ===== 键盘手动调头（调试用）=====
let autoSweep = true;
let headTimer = null;
let pending_dx = 0, pending_dy = 0;
document.addEventListener('keydown', e => {
  // Tab 切换自动/手动扫视模式
  if (e.key === 'Tab') {
    e.preventDefault();
    autoSweep = !autoSweep;
    fetch('/api/auto_sweep?mode=' + (autoSweep ? 'on' : 'off'));
    document.getElementById('vhead').textContent = '扫视:' + (autoSweep ? '开' : '关');
    return;
  }
  if (e.key === ' ') {
    e.preventDefault();
    fetch('/api/head?reset=1').then(r => r.json()).then(d => {
      document.getElementById('vhead').textContent = d.head_x + ',' + d.head_y;
    });
    updateStatus();
    return;
  }
  const map = { 'ArrowUp': [0,1], 'ArrowDown': [0,-1], 'ArrowLeft': [1,0], 'ArrowRight': [-1,0] };
  const d = map[e.key];
  if (!d) return;
  e.preventDefault();
  pending_dx += d[0]; pending_dy += d[1];
  if (headTimer) clearTimeout(headTimer);
  headTimer = setTimeout(() => {
    fetch(`/api/head?dx=${pending_dx}&dy=${pending_dy}`).then(r => r.json()).then(d => {
      document.getElementById('vhead').textContent = d.head_x + ',' + d.head_y;
    });
    pending_dx = 0; pending_dy = 0;
    updateStatus();
  }, 40);
});

drawField();
updateStatus();
setInterval(updateStatus, 500);
</script>
</body>
</html>
"""


# ============ 主入口 ============
if __name__ == '__main__':
    if sys.version_info.major == 2:
        print('请使用 Python 3！'); sys.exit(0)

    print("=" * 50)
    print("AprilTag 定位可视化  Web Server")
    print(f"端口: {WEB_PORT}")
    print(f"固定Tag ID: {FIXED_TAG_IDS}")
    print(f"场地范围: X [{FIELD_X_MIN}, {FIELD_X_MAX}] cm  Y [{FIELD_Y_MIN}, {FIELD_Y_MAX}] cm")
    print("=" * 50)

    compute_tag_centers()
    print(f"已加载 {len(TAG_CENTERS)} 个 Tag 坐标")

    t = threading.Thread(target=camera_and_locate_thread, daemon=True)
    t.start()

    app.run(host='0.0.0.0', port=WEB_PORT,
            threaded=True, use_reloader=False)


