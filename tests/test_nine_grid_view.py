# -*- coding: utf-8 -*-
"""数字宫格图形界面测试（tests/test_nine_grid_view.py）

GUI 本身无法在无头 CI 里验证，但界面依赖的两块可以：
  1. 纯绘图函数（俯视图/相机窗格/状态栏）在无窗口下返回合法图像；
  2. viewer 钩子契约：SimNineGridRobot 在 viewer=None 时行为不变，
     传入 viewer 时 on_action/on_frame 被正确调用。

运行：python tests/test_nine_grid_view.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # 仓库根

import numpy as np

from levels.nine_grid import NineGridLevel
from sim.nine_grid_sim import SIM_LAYOUT, SimNineGridRobot
from sim.nine_grid_view import (
    NineGridView, draw_camera_pane, draw_field_map, draw_status_bar,
)


class RecordingViewer:
    """鸭子类型 viewer：只记录钩子调用，不碰 cv2 窗口"""

    def __init__(self):
        self.attached = None
        self.actions = []
        self.frames = 0

    def attach(self, robot, level):
        self.attached = (robot, level)

    def on_action(self, robot, name, times):
        self.actions.append((name, times))

    def on_frame(self, robot, frame):
        self.frames += 1


def _scene():
    robot = SimNineGridRobot()
    level = NineGridLevel(robot)
    level.digit_cell = {d: c for c, d in SIM_LAYOUT.items() if d is not None}
    level._current_digit = 1
    level.phase = "APPROACH"
    return robot, level


def test_draw_functions_headless():
    """三个绘图函数在无窗口下返回合法 BGR 图像（两个不同位姿）"""
    robot, level = _scene()
    for (x, y, heading) in ((50.0, -20.0, 0.0), (30.0, 40.0, -35.0)):
        robot.pos = np.array([x, y])
        robot.heading = heading
        level.pose = np.array([x, y, np.radians(heading)])
        field = draw_field_map(robot, level, trail=[(50, -20), (x, y)])
        assert field.ndim == 3 and field.shape[2] == 3, f"俯视图形状异常 {field.shape}"
        assert field.dtype == np.uint8

        frame = robot.capture_frame()
        cam = draw_camera_pane(frame, robot, level, obs=None)
        assert cam.ndim == 3 and cam.shape[2] == 3, f"相机窗格形状异常 {cam.shape}"
        assert cam.dtype == np.uint8

        bar = draw_status_bar(cam.shape[1], robot, level,
                              {"actions": 3, "captures": 5})
        assert bar.shape[1] == cam.shape[1] and bar.ndim == 3
    print("  绘图函数（俯视图/相机窗格/状态栏）：无窗口返回合法图像 ✓")


def test_compose_without_window():
    """NineGridView._compose 不依赖窗口即可拼出完整画面"""
    robot, level = _scene()
    viewer = NineGridView()
    viewer.attach(robot, level)
    frame = robot.capture_frame()
    view = viewer._compose(frame)
    assert view.ndim == 3 and view.shape[2] == 3
    assert view.shape[0] > frame.shape[0] // 4, "合成画面高度异常"
    print(f"  画面合成（不建窗口）：{view.shape[1]}x{view.shape[0]} ✓")


def test_viewer_hooks_contract():
    """钩子契约：viewer=None 行为不变；传入 viewer 时钩子被正确调用"""
    # 无 viewer：与原来完全一致
    plain = SimNineGridRobot()
    plain.run_action("go_forward_one_step", 1)
    plain.capture_frame()
    assert plain.action_log == [("go_forward_one_step", 1)]
    assert plain.n_captures == 1

    # 有 viewer：动作/拍照钩子按序触发，参数正确
    rec = RecordingViewer()
    robot = SimNineGridRobot(viewer=rec)
    robot.run_action("go_forward_one_step", 2)
    robot.run_action("turn_right", 1)
    f1 = robot.capture_frame()
    f2 = robot.capture_frame()
    assert rec.actions == [("go_forward_one_step", 2), ("turn_right", 1)]
    assert rec.frames == 2
    assert f1.shape == f2.shape
    print("  viewer 钩子契约：on_action/on_frame 按序触发，viewer=None 无副作用 ✓")


if __name__ == "__main__":
    test_draw_functions_headless()
    test_compose_without_window()
    test_viewer_hooks_contract()
    print("数字宫格图形界面测试通过 ✓")
