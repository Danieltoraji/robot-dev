# -*- coding: utf-8 -*-
"""
真机轨迹记录与日志输出（可选功能）

供 main.py 在 TRACE_ENABLED = True 时使用：
  - TeeWriter：把 stdout 同时输出到终端和日志文件
  - TraceRecorder：记录动作、定位、时间戳
  - TraceRobotState：继承 RobotState，重写 run_action / locate_with_retry 实现无侵入记录
  - save_trajectory_png：把定位轨迹保存为 PNG

默认关闭时 main.py 不会 import 本模块，不影响现有真机逻辑。
"""

import os
import time
from datetime import datetime

from core.paths import RESULT_DIR
from core.robot_core import RobotState


class TeeWriter:
    """将 stdout 同时写入终端和文件"""

    def __init__(self, file_path, original_stdout):
        self.file = open(file_path, "w", encoding="utf-8")
        self.stdout = original_stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


class TraceRecorder:
    """记录真机运行过程中的动作和定位结果"""

    def __init__(self, output_dir=RESULT_DIR):
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(output_dir, f"real_trace_{timestamp}.txt")
        self.png_path = os.path.join(output_dir, f"real_trajectory_{timestamp}.png")
        self.start_time = time.time()
        self.actions = []
        self.locates = []
        self.action_count = 0
        self.locate_count = 0

    def record_action(self, name, times=1, position=None, orientation=None):
        """记录一次动作调用；position/orientation 为最近一次定位结果"""
        self.action_count += 1
        self.actions.append({
            "seq": self.action_count,
            "action": name,
            "times": times,
            "pos": None if position is None else list(position),
            "orientation": None if orientation is None else list(orientation),
            "elapsed": time.time() - self.start_time,
        })

    def record_locate(self, position, orientation=None):
        """记录一次成功定位得到的位置/朝向"""
        self.locate_count += 1
        self.locates.append({
            "seq": self.locate_count,
            "pos": list(position),
            "orientation": None if orientation is None else list(orientation),
            "elapsed": time.time() - self.start_time,
        })


class TraceRobotState(RobotState):
    """带记录能力的 RobotState 子类；不改变原 RobotState 的任何行为"""

    def __init__(self, tag_poses=None, recorder=None):
        super().__init__(tag_poses)
        self._recorder = recorder

    def run_action(self, name, times=1):
        if self._recorder is not None:
            pos = None if self.current_position is None else self.current_position.copy()
            ori = None if self.current_orientation is None else self.current_orientation.copy()
            self._recorder.record_action(name, times, pos, ori)
            if times > 1:
                print(f"[trace] 动作 #{self._recorder.action_count}: {name} × {times}")
            else:
                print(f"[trace] 动作 #{self._recorder.action_count}: {name}")
        super().run_action(name, times)

    def locate_with_retry(self):
        ok = super().locate_with_retry()
        if ok and self._recorder is not None and self.current_position is not None:
            ori = None if self.current_orientation is None else self.current_orientation.copy()
            self._recorder.record_locate(self.current_position.copy(), ori)
        return ok


def save_trajectory_png(recorder, level, path=None):
    """保存真机轨迹图；matplotlib 缺失时只警告，不影响主流程"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        import numpy as np
    except Exception as e:
        print(f"[trace] 无法保存轨迹图（缺少 matplotlib/numpy？）：{e}")
        return False

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.set_aspect("equal")
    ax.set_title("RoboTrack 真机轨迹", fontsize=13)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.grid(True, linestyle="--", alpha=0.3)

    # 赛道外框
    ax.add_patch(Rectangle((0, 0), 100, 100, fill=False, edgecolor="black", linewidth=2))

    # 墙壁
    for rect in level.WALLS:
        x_min, x_max, y_min, y_max = rect
        ax.add_patch(Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            facecolor="gray", edgecolor="black", alpha=0.5, hatch="//",
        ))

    # 路点
    for i, wp in enumerate(level.ROUTE, 1):
        if wp.stop > 0:
            ax.plot(wp.pos[0], wp.pos[1], "bs", markersize=9, markeredgecolor="black")
            ax.annotate(f"停{i}", (wp.pos[0], wp.pos[1]), textcoords="offset points",
                        xytext=(6, 6), fontsize=8, color="blue")
        elif wp.orientation is not None:
            ax.plot(wp.pos[0], wp.pos[1], "ro", markersize=8, markeredgecolor="black")
            ax.annotate(f"转{i}", (wp.pos[0], wp.pos[1]), textcoords="offset points",
                        xytext=(6, -10), fontsize=8, color="red")
        else:
            ax.plot(wp.pos[0], wp.pos[1], "g.", markersize=5)

    # AprilTag
    for tid, pts in level.tag_poses.items():
        p = pts[0][:2]
        ax.plot(p[0], p[1], "g^", markersize=7)
        ax.annotate(f"Tag{tid}", (p[0], p[1]), textcoords="offset points",
                    xytext=(6, 6), fontsize=7, color="green")

    # 定位轨迹
    if recorder.locates:
        traj = np.array([loc["pos"] for loc in recorder.locates])
        ax.plot(traj[:, 0], traj[:, 1], "c-", linewidth=1.5, alpha=0.7, label="定位轨迹")
        ax.plot(traj[0, 0], traj[0, 1], "go", markersize=8, label="起点")
        ax.plot(traj[-1, 0], traj[-1, 1], "r*", markersize=14, label="终点")
        ax.legend(loc="upper left")

    save_path = path or recorder.png_path
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[trace] 轨迹图已保存: {save_path}")
    return True
