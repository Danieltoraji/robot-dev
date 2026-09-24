# -*- coding: utf-8 -*-
"""数字宫格模拟器图形界面（sim/nine_grid_view.py）

用途：把 `sim/nine_grid_sim.py` 的仿真跑成"可看、可停、可单步"的窗口。

界面（单窗口，约 1060×540）
--------------------------
    +----------------------------+----------------------+
    | 相机窗格（合成帧 ×0.25）    | 俯视场地（100×100cm）|
    |  · 预测面板中心（估计位姿） |  · 3×3 格线 + 格号    |
    |  · 目标格投影（橙圈）       |  · 面板（真值布局）   |
    |  · 检出框（暂停/--detect）  |  · 估计/真值位姿+尾迹 |
    +----------------------------+----------------------+
    | 状态栏：step/caps/target/fwd/lat/bearing/est_err/phase |
    +-------------------------------------------------------+

按键：SPACE 暂停/继续 · S 单步 · R 重开 · Q/ESC 退出 · D 切换检出叠加 ·
      +/- 调速 · H 帮助。

运行
----
    python -m sim.nine_grid_view [--seed N] [--random-layout] [--delay MS]
                                 [--detect] [--headless]
无图形环境（DISPLAY 不可用 / opencv-headless 构建）时自动退回无头运行，
等价于 `python -m sim.nine_grid_sim`；也可显式 `--headless`。

实现要点
--------
- 钩子式阻塞 viewer：仿真只重写 I/O 接缝，算法线程在 `capture_frame` 里
  调用 `on_frame`，viewer 在这里渲染并处理按键（暂停时阻塞）。
- 估计/真值比较统一用"相机光心地面投影"：`level.pose` 是光心锚点，仿真里
  `robot.pos` 也是光心（CAM_BODY_OFFSET=0）；关卡内部"机体=光心后退 4cm"
  不参与显示，避免把两者混起来。
- `cv2.putText` 不支持中文，窗口内文字一律用 ASCII，中文只出现在控制台。
- `--delay 0` 映射为 `waitKey(1)`（`waitKey(0)` 是永久阻塞，不能当"最快"）。
"""

import argparse
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from core.ground_homography import grid_cell_center
from levels.nine_grid import project_ground_to_pixel
from sim.nine_grid_sim import (
    FRAME_H, FRAME_W, PANEL_BGR, PANEL_HALF_CM, SIM_LAYOUT,
    random_layout, run_simulation, _print_summary,
)

# 窗口标题与 H 键帮助（ASCII：cv2 画不了中文；中文说明见 main() 启动时的提示）
WIN = ("nine_grid sim  |  SPACE pause  S step  R restart  Q quit  D detect  "
       "+/- speed")
CAM_SCALE = 0.25
MAP_SIZE = 400
MAP_MARGIN = 16
STATUS_H = 52
GAP = 12
# 俯视图视野（cm）：略大于场地，让入口(50,-20)也可见
VIEW_MIN, VIEW_MAX = -25.0, 110.0
HELP = ("keys: SPACE pause/resume | S step one frame | R restart | Q/ESC quit | "
        "D detection boxes on/off | +/- slower/faster | H this help")


class ViewerQuit(Exception):
    """用户要求退出"""


class ViewerRestart(Exception):
    """用户要求重开（main 循环用 seed+1 重建）"""


# =====================================================================
# 纯绘图函数（无窗口，可单测）
# =====================================================================

def _cm_to_map(x, y, size=MAP_SIZE, margin=MAP_MARGIN):
    """场地坐标 cm -> 俯视图像素（x 右、y 向上，图像 y 需翻转）"""
    s = (size - 2 * margin) / (VIEW_MAX - VIEW_MIN)
    px = margin + (x - VIEW_MIN) * s
    py = size - margin - (y - VIEW_MIN) * s
    return int(round(px)), int(round(py))


def draw_field_map(robot, level, trail=(), size=MAP_SIZE):
    """俯视图：格线/面板（真值布局）/估计映射/目标/估计与真值位姿/尾迹"""
    img = np.full((size, size, 3), 245, np.uint8)
    s = (size - 2 * MAP_MARGIN) / (VIEW_MAX - VIEW_MIN)

    # 3×3 格线（0/33.3/66.7/100 cm）
    for i in range(4):
        c = i * 100.0 / 3.0
        cv2.line(img, _cm_to_map(0, c, size), _cm_to_map(100, c, size),
                 (200, 200, 200), 1)
        cv2.line(img, _cm_to_map(c, 0, size), _cm_to_map(c, 100, size),
                 (200, 200, 200), 1)
    cv2.rectangle(img, _cm_to_map(0, 0, size), _cm_to_map(100, 100, size),
                  (120, 120, 120), 2)

    # 格号
    for cell in range(9):
        c = grid_cell_center(cell)
        cv2.putText(img, str(cell), (_cm_to_map(c[0] - 4, c[1] + 10, size)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1)

    # 面板（真值布局，按数字配色）+ 估计映射（青圈）
    half_px = int(round(PANEL_HALF_CM * s))
    for cell, digit in robot.layout.items():
        if digit is None:
            continue
        cx, cy = _cm_to_map(*grid_cell_center(cell), size=size)
        cv2.rectangle(img, (cx - half_px, cy - half_px),
                      (cx + half_px, cy + half_px), PANEL_BGR[digit], -1)
        cv2.rectangle(img, (cx - half_px, cy - half_px),
                      (cx + half_px, cy + half_px), (90, 90, 90), 1)
        cv2.putText(img, str(digit), (cx - 8, cy + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    for digit, cell in getattr(level, "digit_cell", {}).items():
        cx, cy = _cm_to_map(*grid_cell_center(cell), size=size)
        cv2.circle(img, (cx, cy), half_px + 6, (255, 200, 0), 2)
        # 估计与真值不一致时连线（布局扫错的直观信号）
        truth_cell = next((c for c, d in robot.layout.items() if d == digit), None)
        if truth_cell is not None and truth_cell != cell:
            tx, ty = _cm_to_map(*grid_cell_center(truth_cell), size=size)
            cv2.line(img, (cx, cy), (tx, ty), (0, 0, 255), 2)

    # 目标格（当前 SEEK 的数字）
    target = getattr(level, "_current_digit", None)
    if target in getattr(level, "digit_cell", {}):
        cx, cy = _cm_to_map(*grid_cell_center(level.digit_cell[target]), size=size)
        cv2.circle(img, (cx, cy), half_px + 12, (0, 165, 255), 3)

    # 估计轨迹尾迹
    if len(trail) > 1:
        pts = np.array([_cm_to_map(x, y, size) for x, y in trail], np.int32)
        cv2.polylines(img, [pts], False, (0, 180, 0), 1, cv2.LINE_AA)

    # 真值位姿（黑圈 + 短朝向线）
    tx, ty = _cm_to_map(robot.pos[0], robot.pos[1], size)
    cv2.circle(img, (tx, ty), 5, (0, 0, 0), 2)
    th = np.radians(robot.heading)
    cv2.line(img, (tx, ty),
             _cm_to_map(robot.pos[0] + np.sin(th) * 12,
                        robot.pos[1] + np.cos(th) * 12, size),
             (0, 0, 0), 2)

    # 估计位姿（品红实心 + 朝向线）——与真值同为"相机光心地面投影"
    ex, ey = _cm_to_map(level.pose[0], level.pose[1], size)
    cv2.circle(img, (ex, ey), 5, (255, 0, 255), -1)
    eth = level.pose[2]
    cv2.line(img, (ex, ey),
             _cm_to_map(level.pose[0] + np.sin(eth) * 12,
                        level.pose[1] + np.cos(eth) * 12, size),
             (255, 0, 255), 2)

    # 入口标记
    ix, iy = _cm_to_map(50, -20, size)
    cv2.drawMarker(img, (ix, iy), (128, 0, 128), cv2.MARKER_TRIANGLE_UP, 14, 2)
    cv2.putText(img, "entry", (ix - 22, iy - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 0, 128), 1)
    cv2.putText(img, "magenta=est  black=truth  orange=target",
                (6, size - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1)
    return img


def draw_camera_pane(frame, robot, level, obs=None):
    """相机窗格：合成帧缩放 + 预测面板中心（估计位姿）+ 可选检出框"""
    pane = cv2.resize(frame, None, fx=CAM_SCALE, fy=CAM_SCALE,
                      interpolation=cv2.INTER_AREA)
    h, w = pane.shape[:2]
    pitch, head = robot.pitch, robot.current_head_pulse

    # 预测：用估计位姿把已知格心投影回图像
    for digit, cell in getattr(level, "digit_cell", {}).items():
        px = project_ground_to_pixel(grid_cell_center(cell), level.pose[0],
                                     level.pose[1], level.pose[2], pitch, head)[0]
        if not np.all(np.isfinite(px)):
            continue
        x, y = int(round(px[0] * CAM_SCALE)), int(round(px[1] * CAM_SCALE))
        if not (0 <= x < w and 0 <= y < h):
            continue
        color = (0, 165, 255) if digit == getattr(level, "_current_digit", None) \
            else (255, 0, 255)
        cv2.drawMarker(pane, (x, y), color, cv2.MARKER_CROSS, 16, 2)
        cv2.putText(pane, f"{digit}", (x + 8, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 检出（原生像素 -> 窗格坐标）
    for o in (obs or []):
        bx, by, bw, bh = [v * CAM_SCALE for v in o.bbox]
        cv2.rectangle(pane, (int(bx), int(by)),
                      (int(bx + bw), int(by + bh)), (0, 255, 0), 1)
        tag = f"{o.color}->{o.digit}" + (" C" if o.clipped else "")
        cv2.putText(pane, tag, (int(bx), max(12, int(by) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    cv2.putText(pane, f"pitch={pitch} head={head}", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return pane


def draw_status_bar(width, robot, level, stats, last_action=""):
    """状态栏：步数/拍照/目标相对量/估计误差/FSM 阶段"""
    bar = np.full((STATUS_H, width, 3), 35, np.uint8)
    digit = getattr(level, "_current_digit", None)
    cell = getattr(level, "digit_cell", {}).get(digit, "-")
    try:
        fwd, lat, bearing = level.target_relative(digit)
        rel = f"fwd={fwd:+5.1f} lat={lat:+5.1f} bearing={bearing:+5.1f}"
    except Exception:
        rel = "fwd=  -   lat=  -   bearing=  -"
    # 估计/真值都用相机光心地面投影比较
    exy = np.asarray(level.pose[:2]) - np.asarray(robot.pos)
    eth = np.degrees(np.arctan2(np.sin(level.pose[2] - np.radians(robot.heading)),
                                np.cos(level.pose[2] - np.radians(robot.heading))))
    line1 = (f"step={stats.get('actions', 0):3d}  caps={stats.get('captures', 0):3d}  "
             f"target={digit}@cell{cell}  phase={getattr(level, 'phase', '?')}")
    line2 = (f"{rel}  est_err={np.hypot(*exy):4.1f}cm/{abs(eth):4.1f}deg  "
             f"last={last_action}")
    cv2.putText(bar, line1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (220, 220, 220), 1)
    cv2.putText(bar, line2, (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (180, 220, 180), 1)
    return bar


# =====================================================================
# Viewer
# =====================================================================

class NineGridView:
    """阻塞式图形 viewer：挂到 SimNineGridRobot 的 I/O 钩子上

    只依赖 duck-typing 的 on_action/on_frame/attach，仿真侧对 viewer=None
    完全无感（测试/无头路径不受影响）。
    """

    def __init__(self, delay_ms=30, show_detect=False):
        self.delay_ms = max(1, int(delay_ms))
        self.show_detect = bool(show_detect)
        self.paused = False
        self.step_once = False
        self.trail = []
        self.robot = None
        self.level = None
        self.last_action = ""
        self._last_frame = None
        self._window_ready = False

    # ---- 生命周期 ----

    def attach(self, robot, level):
        self.robot = robot
        self.level = level
        self.trail = [(float(level.pose[0]), float(level.pose[1]))]

    def close(self):
        if self._window_ready:
            cv2.destroyWindow(WIN)
            self._window_ready = False

    # ---- 仿真钩子 ----

    def on_action(self, robot, name, times):
        """动作后：更新尾迹与"最近动作"标签（不渲染，等下一次拍照）"""
        self.last_action = f"{name}x{times}" if times > 1 else name
        if self.level is not None:
            self.trail.append((float(self.level.pose[0]),
                               float(self.level.pose[1])))
            if len(self.trail) > 500:
                self.trail = self.trail[-500:]

    def on_frame(self, robot, frame):
        """拍照后：渲染 + 按键（暂停时阻塞；R/Q 抛异常给 main 处理）"""
        self._last_frame = frame
        self._ensure_window()
        self._show(frame)
        if not self.paused:
            self._handle_key(cv2.waitKey(self.delay_ms) & 0xFF)
            self._check_alive()
            return
        # 暂停：阻塞等按键；SPACE 恢复，S 放行一次，R/Q 抛异常
        while True:
            self._check_alive()
            self._handle_key(cv2.waitKey(0) & 0xFF)
            if not self.paused:
                return
            if self.step_once:
                self.step_once = False
                return
            self._show(frame)  # D/±/H 等按键后重绘

    def finish(self, stats):
        """跑完：显示终局画面，等 R 重开 / Q 退出"""
        ok_n = sum(1 for _, ok in stats.get("results", []) if ok)
        banner = (f"done {ok_n}/{len(stats.get('results', []))}  "
                  f"captures={stats.get('captures', 0)}  "
                  f"layout={'ok' if stats.get('layout_ok') else 'FAIL'}  "
                  f"R restart / Q quit")
        print(f"[view] {banner}")
        if self._last_frame is None:   # 极端情况：还没拍到任何帧
            return
        self._ensure_window()
        self._show(self._last_frame, banner=banner)
        while True:
            self._check_alive()
            key = cv2.waitKey(0) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                return
            if key in (ord("r"), ord("R")):
                raise ViewerRestart()

    # ---- 内部 ----

    def _ensure_window(self):
        if self._window_ready:
            return
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        self._window_ready = True

    def _check_alive(self):
        """窗口被 X 关闭时抛 ViewerQuit，避免算法线程永久阻塞"""
        try:
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                raise ViewerQuit()
        except cv2.error as e:
            raise ViewerQuit() from e

    def _handle_key(self, key):
        if key in (255, -1):
            return
        if key in (27, ord("q"), ord("Q")):
            raise ViewerQuit()
        if key == ord(" "):
            self.paused = not self.paused
            print(f"[view] {'已暂停（SPACE 继续，S 单步）' if self.paused else '继续运行'}")
        elif key in (ord("s"), ord("S")):
            self.step_once = True
        elif key in (ord("r"), ord("R")):
            raise ViewerRestart()
        elif key in (ord("d"), ord("D")):
            self.show_detect = not self.show_detect
            print(f"[view] 检出框叠加: {'开' if self.show_detect else '关'}")
        elif key in (ord("+"), ord("=")):
            self.delay_ms = min(500, self.delay_ms * 2)
            print(f"[view] 每帧等待 {self.delay_ms}ms")
        elif key in (ord("-"), ord("_")):
            self.delay_ms = max(1, self.delay_ms // 2)
            print(f"[view] 每帧等待 {self.delay_ms}ms")
        elif key in (ord("h"), ord("H")):
            print(f"[view] 按键说明: {HELP}")

    def _detections(self, frame):
        """检出叠加：只在暂停/单步/--detect 时跑检测器（单帧 ~120ms）"""
        if not (self.show_detect or self.paused or self.step_once):
            return None
        try:
            return self.level.detector.detect_panels(frame)
        except Exception:
            return None

    def _compose(self, frame):
        cam = draw_camera_pane(frame, self.robot, self.level,
                               obs=self._detections(frame))
        field = draw_field_map(self.robot, self.level, self.trail)
        if field.shape[0] < cam.shape[0]:
            pad = cam.shape[0] - field.shape[0]
            field = cv2.copyMakeBorder(field, pad // 2, pad - pad // 2, 0, 0,
                                       cv2.BORDER_CONSTANT, value=(245, 245, 245))
        gap = np.full((cam.shape[0], GAP, 3), 35, np.uint8)
        top = np.hstack([cam, gap, field])
        stats = {"actions": len(self.robot.action_log),
                 "captures": self.robot.n_captures}
        return np.vstack([top, draw_status_bar(top.shape[1], self.robot,
                                               self.level, stats,
                                               last_action=self.last_action)])

    def _show(self, frame, banner=None):
        view = self._compose(frame)
        if banner:
            cv2.rectangle(view, (0, 0), (view.shape[1], 34), (0, 0, 0), -1)
            cv2.putText(view, banner, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 1)
        cv2.imshow(WIN, view)


# =====================================================================
# CLI
# =====================================================================

def _gui_available():
    """无 DISPLAY / opencv-headless 构建时返回 False（不抛异常）"""
    try:
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        cv2.destroyWindow(WIN)
        return True
    except cv2.error:
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description="数字宫格模拟器图形界面")
    ap.add_argument("--seed", type=int, default=3,
                    help="动作噪声/随机布局种子（默认 3）")
    ap.add_argument("--random-layout", action="store_true",
                    help="由 seed 生成合法随机布局")
    ap.add_argument("--delay", type=int, default=30,
                    help="每帧等待 ms（0=最快，映射为 waitKey(1)）")
    ap.add_argument("--detect", action="store_true",
                    help="运行中也跑检测器叠加（慢，整轮约 +25s）")
    ap.add_argument("--headless", action="store_true",
                    help="不开窗，等价 python -m sim.nine_grid_sim")
    ap.add_argument("--quiet", action="store_true",
                    help="吞掉关卡逐行日志（无头/脚本场景）")
    args = ap.parse_args(argv)

    headless = args.headless or not _gui_available()
    if headless and not args.headless:
        print("[view] 未检测到图形界面（DISPLAY 不可用？）——改为无窗口运行")
        print("[view] 说明：本命令用于在本机开窗查看；远程 SSH 场景请加 "
              "--headless，或参考 tools/camera_preview.py --stream 的网页流方案")
    if not headless:
        # 窗口内文字只能是 ASCII，这里补一份中文按键说明（只打印一次）
        print("[view] 按键：SPACE 暂停／继续｜S 单步放行一帧｜R 换 seed 重开"
              "｜Q 或 ESC 退出｜D 检出框叠加开关｜+／- 调慢／调快｜H 本说明")

    seed = args.seed
    while True:
        layout = random_layout(seed) if args.random_layout else SIM_LAYOUT
        if headless:
            run = run_simulation(layout=layout, seed=seed, quiet=args.quiet)
            _print_summary(run.stats)
            return 0 if (run.stats["ok_all"] and run.stats["layout_ok"]) else 1

        print(f"[view] 按键说明: {HELP}")
        viewer = NineGridView(delay_ms=args.delay, show_detect=args.detect)
        try:
            run = run_simulation(layout=layout, seed=seed, viewer=viewer,
                                 quiet=args.quiet)
        except ViewerRestart:
            viewer.close()
            seed += 1
            print(f"[view] 重新开始，seed 改为 {seed}")
            continue
        except ViewerQuit:
            viewer.close()
            print("[view] 用户已退出")
            return 0
        try:
            viewer.finish(run.stats)
            viewer.close()
            return 0 if (run.stats["ok_all"] and run.stats["layout_ok"]) else 1
        except ViewerRestart:
            viewer.close()
            seed += 1
            print(f"[view] 重新开始，seed 改为 {seed}")
        except ViewerQuit:
            viewer.close()
            print("[view] 用户已退出")
            return 0


if __name__ == "__main__":
    sys.exit(main())
