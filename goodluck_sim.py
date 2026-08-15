# -*- coding: utf-8 -*-
"""
寻路算法测试程序（模拟器 + 可视化）

架构说明（重构后）：
  - 通用层 robot_core.py 提供 RobotState 类（定位/动作/头部舵机等 I/O 能力）。
  - 关卡层 levels/goodluck.py 提供赛道数据、导航算法和 run_level(state) 入口。
  - 本模拟器创建 SimRobotState(RobotState) 子类，重写 run_action / solve_pnp /
    set_head 三个 I/O 方法为桩函数（更新模拟器状态 + 刷新可视化），
    算法函数通过 state.xxx() 调用命中桩函数，无需 monkey-patch。

运行：python goodluck_sim.py
"""

import os
import sys
import time
import numpy as np
import matplotlib

# 后端选择：尊重 MPLBACKEND 环境变量；否则优先交互式（TkAgg）支持动画
if not os.environ.get("MPLBACKEND"):
    try:
        matplotlib.use("TkAgg")
    except Exception:
        matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
from datetime import datetime

# 配置中文字体（Windows: Microsoft YaHei / SimHei；缺失则回退默认）
for _font in ["Microsoft YaHei", "SimHei", "WenQuanYi Micro Hei", "Arial Unicode MS"]:
    try:
        matplotlib.rcParams["font.sans-serif"] = [_font] + matplotlib.rcParams["font.sans-serif"]
        break
    except Exception:
        continue
matplotlib.rcParams["axes.unicode_minus"] = False  # 负号显示

# 导入通用层与关卡层
from robot_core import RobotState, HEAD_CENTER, HEAD_RIGHT, HEAD_LEFT, HEAD_MOVE_TIME_MS
from levels import goodluck as gl


# =====================================================================
# 模拟器配置（噪声开关，默认全 0 = 理想模式）
# =====================================================================

LOCATE_NOISE_STD = 0.5          # 定位位置噪声标准差（cm），0=无噪声
LOCATE_ANGLE_NOISE_STD = 1.0    # 定位朝向角噪声标准差（度），0=无噪声
ACTION_ERROR_STD = 0.1          # 动作步长误差标准差（比例，0.1=±10%），0=无误差
TURN_ERROR_STD = 5.0            # 转向角度误差标准差（度），0=无误差
ANIM_PAUSE_SEC = 0.01            # 每步动画刷新间隔（秒）
MAX_SIM_STEPS = 500            # 模拟最大动作步数，防止算法不收敛时无限循环卡死

'''压测模式
LOCATE_NOISE_STD = 2.4          # 定位位置噪声标准差（cm），0=无噪声
LOCATE_ANGLE_NOISE_STD = 5.0    # 定位朝向角噪声标准差（度），0=无噪声
ACTION_ERROR_STD = 0.1          # 动作步长误差标准差（比例，0.1=±10%），0=无误差
TURN_ERROR_STD = 8.0            # 转向角度误差标准差（度），0=无误差
ANIM_PAUSE_SEC = 0.01            # 每步动画刷新间隔（秒）
MAX_SIM_STEPS = 500            # 模拟最大动作步数，防止算法不收敛时无限循环卡死
'''

'''稍大噪声模式
LOCATE_NOISE_STD = 1.0          # 定位位置噪声标准差（cm），0=无噪声
LOCATE_ANGLE_NOISE_STD = 3.0    # 定位朝向角噪声标准差（度），0=无噪声
ACTION_ERROR_STD = 0.1          # 动作步长误差标准差（比例，0.1=±10%），0=无误差
TURN_ERROR_STD = 5.0            # 转向角度误差标准差（度），0=无误差
ANIM_PAUSE_SEC = 0.01            # 每步动画刷新间隔（秒）
MAX_SIM_STEPS = 500            # 模拟最大动作步数，防止算法不收敛时无限循环卡死
'''

'''一般模式
LOCATE_NOISE_STD = 0.5          # 定位位置噪声标准差（cm），0=无噪声
LOCATE_ANGLE_NOISE_STD = 1.0    # 定位朝向角噪声标准差（度），0=无噪声
ACTION_ERROR_STD = 0.1          # 动作步长误差标准差（比例，0.1=±10%），0=无误差
TURN_ERROR_STD = 5.0            # 转向角度误差标准差（度），0=无误差
ANIM_PAUSE_SEC = 0.01            # 每步动画刷新间隔（秒）
MAX_SIM_STEPS = 500            # 模拟最大动作步数，防止算法不收敛时无限循环卡死
'''

'''理想模式
LOCATE_NOISE_STD = 0.0          # 定位位置噪声标准差（cm），0=无噪声
LOCATE_ANGLE_NOISE_STD = 0.0    # 定位朝向角噪声标准差（度），0=无噪声
ACTION_ERROR_STD = 0.0          # 动作步长误差标准差（比例，0.1=±10%），0=无误差
TURN_ERROR_STD = 0.0            # 转向角度误差标准差（度），0=无误差
ANIM_PAUSE_SEC = 0.01            # 每步动画刷新间隔（秒）
MAX_SIM_STEPS = 500            # 模拟最大动作步数，防止算法不收敛时无限循环卡死
'''

# 输出目录与文件（result/ 子目录，文件名含日期时间）
_RESULT_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_DIR = "result"
TRAJECTORY_PNG_PATH = os.path.join(RESULT_DIR, f"trajectory_{_RESULT_TIMESTAMP}.png")

# 机器人边界框尺寸（以几何中心为中心）
ROBOT_WIDTH_CM = 26.0          # 机器人边界框宽（cm）
ROBOT_LENGTH_CM = 10.0         # 机器人边界框长（cm）

# 日志输出到文件
LOG_TO_FILE = True             # 是否输出日志到文件
LOG_FILE_PATH = os.path.join(RESULT_DIR, f"simulation_log_{_RESULT_TIMESTAMP}.txt")

# 各动作耗时（秒），用于实时累计完成时间
ACTION_TIME_SEC = {
    "stand": 1.0,
    "go_forward_one_step": 1.0,
    "go_forward_one_small_step": 0.8,
    "go_forward": 1.0,
    "back_one_step": 1.0,
    "back": 1.0,
    "left_move": 1.2,
    "right_move": 1.2,
    "turn_left": 1.5,
    "turn_left_small_step": 0.8,
    "turn_right": 1.5,
    "turn_right_small_step": 0.8,
}
BATCH_FORWARD_MAX_ANGLE_DEG = 15.0 # 批量直行允许的最大朝向偏差（度）；偏差越大允许步数越少，超过此值不批量
LOCATE_TIME_SEC = 0.5  # 每次定位耗时（秒），可调。真实硬件拍照+检测+PnP约数秒

# 机器人初始状态（入口附近，朝东）
INITIAL_POS = np.array([2.0, 20.0], dtype=np.float64)
INITIAL_ORIENTATION = np.array([1.0, 0.0], dtype=np.float64)


class SimState:
    """模拟器状态：维护机器人的真实位置/朝向/轨迹

    所有 apply_* 方法按机体坐标系更新状态，并追加轨迹点。
    """

    def __init__(self, pos, orientation):
        self.pos = np.array(pos, dtype=np.float64).copy()
        self.orientation = np.array(orientation, dtype=np.float64).copy()
        norm = np.linalg.norm(self.orientation)
        if norm != 0:
            self.orientation /= norm
        self.head_pulse = HEAD_CENTER
        self.trajectory = [self.pos.copy()]
        self.last_action = "init"
        self.step_count = 0
        self.locate_count = 0
        self.elapsed_time = 0.0
        # 记录每步动作信息（序号、动作名、位置、朝向），用于日志和可视化
        self.step_actions = []

    def _record(self, action_name):
        self.last_action = action_name
        self.trajectory.append(self.pos.copy())
        self.step_actions.append({
            "step": self.step_count,
            "action": action_name,
            "pos": self.pos.copy(),
            "orientation": self.orientation.copy(),
            "time": self.elapsed_time,
        })

    def apply_forward(self, cm):
        """沿当前朝向前进 cm 厘米"""
        actual_cm = cm
        if ACTION_ERROR_STD > 0:
            actual_cm = cm * (1.0 + np.random.normal(0, ACTION_ERROR_STD))
        self.pos = self.pos + actual_cm * self.orientation
        self._record("forward")

    def apply_back(self, cm):
        """沿当前朝向后退 cm 厘米"""
        actual_cm = cm
        if ACTION_ERROR_STD > 0:
            actual_cm = cm * (1.0 + np.random.normal(0, ACTION_ERROR_STD))
        self.pos = self.pos - actual_cm * self.orientation
        self._record("back")

    def apply_left_move(self, cm):
        """机体左侧横移 cm 厘米（左转为 [-oy[1], oy[0]]）"""
        actual_cm = cm
        if ACTION_ERROR_STD > 0:
            actual_cm = cm * (1.0 + np.random.normal(0, ACTION_ERROR_STD))
        left_dir = np.array([-self.orientation[1], self.orientation[0]])
        self.pos = self.pos + actual_cm * left_dir
        self._record("left_move")

    def apply_right_move(self, cm):
        """机体右侧横移 cm 厘米（右转为 [oy[1], -oy[0]]）"""
        actual_cm = cm
        if ACTION_ERROR_STD > 0:
            actual_cm = cm * (1.0 + np.random.normal(0, ACTION_ERROR_STD))
        right_dir = np.array([self.orientation[1], -self.orientation[0]])
        self.pos = self.pos + actual_cm * right_dir
        self._record("right_move")

    def apply_turn(self, deg):
        """圆周运动转向：机体绕旋转中心做圆弧运动（正=左转，负=右转）

        旋转中心 = 机体位置 - d·朝向 + R·侧向方向
          左转：侧向 = left_dir = [-oy, ox]，圆心在左后方
          右转：侧向 = right_dir = [oy, -ox]，圆心在右后方
        机体绕圆心旋转 α 度（左转正、右转负），位置和朝向同步更新。

        噪声模型（方向锁定）：旋转方向由命令角度 deg 锁定，噪声只改变旋转
        幅度（不足/过冲），永不变号——模拟真实舵机转向"过头或不足"而非反向。
        """
        actual_deg = deg
        if TURN_ERROR_STD > 0:
            noise = np.random.normal(0, TURN_ERROR_STD)
            # 方向由命令锁定，噪声只作用于幅度：左转只转 [0,+∞)，右转只转 (-∞,0]
            if deg >= 0:
                actual_deg = abs(deg + noise)
            else:
                actual_deg = -abs(deg + noise)
        alpha = np.radians(actual_deg)
        oy = self.orientation
        # 旋转中心：先向后退 d·O，再侧向偏移 R。圆心方向由命令角度决定（与旋转方向一致）
        base = self.pos - gl.CAMERA_FORWARD_OFFSET_CM * oy
        if deg >= 0:  # 左转（命令）
            left_dir = np.array([-oy[1], oy[0]])
            center = base + gl.TURN_LEFT_RADIUS_CM * left_dir
        else:  # 右转（命令）
            right_dir = np.array([oy[1], -oy[0]])
            center = base + gl.TURN_RIGHT_RADIUS_CM * right_dir
        # 旋转矩阵 R(α) = [[cos, -sin], [sin, cos]]
        cos_a, sin_a = np.cos(alpha), np.sin(alpha)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        # P' = K + R·(P - K)，O' = R·O
        self.pos = center + R @ (self.pos - center)
        self.orientation = R @ self.orientation
        norm = np.linalg.norm(self.orientation)
        if norm != 0:
            self.orientation /= norm
        self._record("turn")


# 全局模拟器实例与可视化实例（run_simulation 中重新初始化）
sim = None
viz = None


class Visualizer:
    """matplotlib 可视化：绘制赛道、目标点、机器人箭头、历史轨迹

    静态层（赛道/墙壁/目标点）在 __init__ 绘制一次；
    动态层（机器人箭头/轨迹/动作文字）在 update() 每步重绘。
    """

    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self._draw_static()
        # 动态层句柄
        self.robot_arrow = None
        self.robot_box = None
        self.traj_line = None
        self.action_text = None
        self.step_markers = []  # 步骤序号标记列表
        self.fig.canvas.manager.set_window_title("寻路算法模拟器")

    def _draw_static(self):
        ax = self.ax
        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 105)
        ax.set_aspect("equal")
        ax.set_title("RoboTrack 寻路算法模拟", fontsize=13)
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")
        ax.grid(True, linestyle="--", alpha=0.3)

        # 外框
        ax.add_patch(Rectangle((0, 0), 100, 100, fill=False, edgecolor="black", linewidth=2))

        # 墙壁
        for rect in gl.WALLS:
            x_min, x_max, y_min, y_max = rect
            ax.add_patch(Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                facecolor="gray", edgecolor="black", alpha=0.5, hatch="//",
            ))

        # 路点（ROUTE）—— 停靠点=蓝色方块，转向点=红色圆点，中间走廊路点=绿色小点
        for i, wp in enumerate(gl.ROUTE, 1):
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

        # AprilTag 位置（取各 tag 第一点近似）—— 绿色三角
        for tid, pts in gl.tag_poses.items():
            p = pts[0][:2]
            ax.plot(p[0], p[1], "g^", markersize=7)
            ax.annotate(f"Tag{tid}", (p[0], p[1]), textcoords="offset points",
                        xytext=(6, 6), fontsize=7, color="green")

        # 入口/出口标注
        ax.annotate("入口", (0, 20), textcoords="offset points",
                    xytext=(-30, 0), fontsize=9, color="purple")
        ax.annotate("出口", (100, 20), textcoords="offset points",
                    xytext=(8, 0), fontsize=9, color="purple")

    def update(self, sim_state, action_text=""):
        """重绘动态层并刷新"""
        ax = self.ax
        # 移除旧的动态元素
        for artist in [self.robot_arrow, self.robot_box, self.traj_line, self.action_text] + self.step_markers:
            if artist is not None:
                artist.remove()
        self.step_markers = []
        # 历史轨迹
        traj = np.array(sim_state.trajectory)
        self.traj_line, = ax.plot(traj[:, 0], traj[:, 1], "c-", linewidth=1.5, alpha=0.7)
        # 步骤序号标记：在每个轨迹点旁标注序号
        for i, point in enumerate(sim_state.trajectory):
            if i == 0:
                label = "S"  # 起点
            else:
                # 从 step_actions 获取动作名缩写
                if i - 1 < len(sim_state.step_actions):
                    sa = sim_state.step_actions[i - 1]
                    label = f"#{sa['step']}"
                else:
                    label = f"#{i}"
            marker = ax.annotate(
                label,
                (point[0], point[1]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=6,
                color="darkblue",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.1", facecolor="yellow", alpha=0.7, edgecolor="none"),
            )
            self.step_markers.append(marker)
        # 机器人位置箭头
        pos = sim_state.pos
        ori = sim_state.orientation
        # 机器人边界框：以 pos 为中心，长边沿朝向旋转
        angle_deg = np.degrees(np.arctan2(ori[1], ori[0]))
        # 矩形左下角（未旋转时）：以中心为原点偏移
        box_x = pos[0] - ROBOT_LENGTH_CM / 2
        box_y = pos[1] - ROBOT_WIDTH_CM / 2
        t = Affine2D().rotate_deg_around(pos[0], pos[1], angle_deg) + ax.transData
        self.robot_box = Rectangle(
            (box_x, box_y), ROBOT_LENGTH_CM, ROBOT_WIDTH_CM,
            facecolor=(1.0, 0.6, 0.6, 0.3), edgecolor="red", linewidth=1.2,
        )
        self.robot_box.set_transform(t)
        ax.add_patch(self.robot_box)
        self.robot_arrow = ax.annotate(
            "",
            xy=(pos[0] + ori[0] * 4, pos[1] + ori[1] * 4),
            xytext=(pos[0], pos[1]),
            arrowprops=dict(arrowstyle="->", color="magenta", lw=2.5),
        )
        ax.plot(pos[0], pos[1], "mo", markersize=6)
        # 动作文字
        self.action_text = ax.text(
            0.02, 0.98, action_text, transform=ax.transAxes,
            fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )
        # 交互后端用 pause 触发重绘；非交互后端用 draw
        if ANIM_PAUSE_SEC > 0 and matplotlib.get_backend().lower() != "agg":
            plt.pause(ANIM_PAUSE_SEC)
        else:
            self.fig.canvas.draw_idle()


class SimRobotState(RobotState):
    """模拟器桩：继承 RobotState，重写 I/O 方法为模拟器逻辑

    算法函数通过 state.run_action() / state.solve_pnp() / state.set_head()
    调用时，命中此处的桩函数，而非真实硬件操作。
    """

    def __init__(self, tag_poses=None):
        super().__init__(tag_poses)
        # 模拟器状态由 run_simulation 注入
        self._sim = None
        self._viz = None

    def attach_sim(self, sim_instance, viz_instance):
        """注入模拟器状态与可视化实例"""
        self._sim = sim_instance
        self._viz = viz_instance

    def run_action(self, name, times=1):
        """桩：解析动作名，更新模拟器状态，并刷新可视化

        每次行动打印统一日志块，直观对比"实际状态"与"算法感知（带噪声）状态"：
          ==================第N次行动开始===================
            动作名、实际位置/朝向、带噪声位置/带输出朝向、执行结果
          ==================第N次行动结束===================
        """
        if times > 1:
            print(f"\n{'='*50}")
            print(f"  ★ 连续直行: {name} × {times} 步（跳过中间定位）")
            print(f"{'='*50}")
        for _ in range(times):
            if self._sim.step_count >= MAX_SIM_STEPS:
                raise RuntimeError(
                    f"模拟步数超限({MAX_SIM_STEPS})，算法可能不收敛。中止以防卡死。"
                )
            self._sim.step_count += 1
            step_num = self._sim.step_count

            # ===== 行动开始：实际状态 vs 感知（带噪声）状态 =====
            print(f"\n{'='*18}第{step_num}次行动开始{'='*18}")
            real_pos = self._sim.pos
            real_ori = self._sim.orientation
            if self.current_position is not None:
                noisy_pos_str = f"({self.current_position[0]:.2f}, {self.current_position[1]:.2f})"
            else:
                noisy_pos_str = "未定位"
            if self.current_orientation is not None:
                noisy_ori_str = f"({self.current_orientation[0]:.2f}, {self.current_orientation[1]:.2f})"
            else:
                noisy_ori_str = "未定位"
            print(f"  动作: {name}")
            print(f"  实际位置=({real_pos[0]:.2f}, {real_pos[1]:.2f})，带噪声位置={noisy_pos_str}")
            print(f"  实际朝向=({real_ori[0]:.2f}, {real_ori[1]:.2f})，带输出朝向={noisy_ori_str}")

            # ===== 执行动作 =====
            if name == "stand":
                self._sim._record("stand")
            elif name == "go_forward_one_step":
                self._sim.apply_forward(gl.FORWARD_ONE_STEP_CM)
            elif name == "go_forward_one_small_step":
                self._sim.apply_forward(gl.FORWARD_ONE_SMALL_STEP_CM)
            elif name == "go_forward":
                # 连续前进：按一步常量模拟
                self._sim.apply_forward(gl.FORWARD_ONE_STEP_CM)
            elif name == "back_one_step":
                self._sim.apply_back(gl.BACK_ONE_STEP_CM)
            elif name == "back":
                self._sim.apply_back(gl.BACK_ONE_STEP_CM)
            elif name == "left_move":
                self._sim.apply_left_move(gl.LEFT_MOVE_CM)
            elif name == "right_move":
                self._sim.apply_right_move(gl.RIGHT_MOVE_CM)
            elif name == "turn_left":
                self._sim.apply_turn(gl.TURN_LEFT_DEG)
            elif name == "turn_left_small_step":
                self._sim.apply_turn(gl.TURN_LEFT_SMALL_STEP_DEG)
            elif name == "turn_right":
                self._sim.apply_turn(-gl.TURN_RIGHT_DEG)
            elif name == "turn_right_small_step":
                self._sim.apply_turn(-gl.TURN_RIGHT_SMALL_STEP_DEG)
            else:
                print(f"[sim] 未知动作，忽略: {name}")
                self._sim._record(name)
            # 累加动作耗时
            self._sim.elapsed_time += ACTION_TIME_SEC.get(name, 1.0)
            print(f"  ✦ 执行后: 位置=({self._sim.pos[0]:.2f}, {self._sim.pos[1]:.2f})  "
                  f"朝向=({self._sim.orientation[0]:.2f}, {self._sim.orientation[1]:.2f})  "
                  f"累计时间={self._sim.elapsed_time:.1f}s")

            # ===== 可视化刷新 =====
            if self._viz is not None:
                batch_tag = f" × {times} 步（连续直行）" if times > 1 else ""
                self._viz.update(self._sim, f"第{step_num}次行动: {name}{batch_tag}\n"
                                f"实际位置: ({real_pos[0]:.1f}, {real_pos[1]:.1f})\n"
                                f"带噪声位置: {noisy_pos_str}\n"
                                f"实际朝向: ({real_ori[0]:.1f}, {real_ori[1]:.1f})\n"
                                f"带输出朝向: {noisy_ori_str}\n"
                                f"已用时间: {self._sim.elapsed_time:.1f}s  动作: {step_num}步  定位: {self._sim.locate_count}次")

            # ===== 行动结束 =====
            print(f"{'='*18}第{step_num}次行动结束{'='*18}")

    def solve_pnp(self):
        """桩：直接返回模拟器真实状态（可注入噪声），不拍照不检测

        同时兼作死循环检测点：每次 navigate_to_target 循环都会调用定位，
        若定位次数超限则抛异常中止，防止算法逻辑死循环（不执行动作时
        MAX_SIM_STEPS 无法触发）。
        """
        self._sim.locate_count += 1
        if self._sim.locate_count > MAX_SIM_STEPS * 3:
            raise RuntimeError(
                f"定位次数超限({self._sim.locate_count})，算法可能陷入死循环。中止以防卡死。"
            )
        # 累加定位耗时（真实硬件拍照+检测+PnP耗时显著）
        self._sim.elapsed_time += LOCATE_TIME_SEC
        print(f"  📍 定位 #{self._sim.locate_count}（耗时 {LOCATE_TIME_SEC}s，累计 {self._sim.elapsed_time:.1f}s）")
        pos = self._sim.pos.copy()
        ori = self._sim.orientation.copy()
        print(f"触发定位。真实位置为 ({pos[0]:.1f}, {pos[1]:.1f})，朝向为 ({ori[0]:.2f}, {ori[1]:.2f})")
        # 注入定位噪声
        if LOCATE_NOISE_STD > 0:
            pos = pos + np.random.normal(0, LOCATE_NOISE_STD, size=2)
        if LOCATE_ANGLE_NOISE_STD > 0:
            ang = np.radians(np.random.normal(0, LOCATE_ANGLE_NOISE_STD))
            c, s = np.cos(ang), np.sin(ang)
            R = np.array([[c, -s], [s, c]])
            ori = R @ ori
            n = np.linalg.norm(ori)
            if n != 0:
                ori /= n
        self.current_position = pos
        self.current_orientation = ori
        if self._viz is not None:
            self._viz.update(self._sim, f"📍 定位 #{self._sim.locate_count}\n"
                            f"位置: ({self._sim.pos[0]:.1f}, {self._sim.pos[1]:.1f})  "
                            f"朝向: ({self._sim.orientation[0]:.2f}, {self._sim.orientation[1]:.2f})\n"
                            f"已用时间: {self._sim.elapsed_time:.1f}s  动作: {self._sim.step_count}步  定位: {self._sim.locate_count}次")
        print(f"返回定位结果（带噪声）: 位置=({pos[0]:.2f}, {pos[1]:.2f})  朝向=({ori[0]:.2f}, {ori[1]:.2f}) ")
        return True

    def set_head(self, pulse, move_time_ms=HEAD_MOVE_TIME_MS):
        """桩：只更新头部脉宽记录，不调舵机"""
        self._sim.head_pulse = pulse
        self.current_head_pulse = pulse


def save_trajectory_png(path=TRAJECTORY_PNG_PATH):
    """保存当前 matplotlib 图为 PNG（含完整赛道+最终轨迹+终点姿态）"""
    if viz is None:
        print("[save_trajectory_png] 可视化未初始化，跳过保存。")
        return
    # 保存前刷新最终画面，确保所有步骤序号标记都已绘制
    if sim is not None:
        viz.update(sim, f"模拟完成\n总步数: {sim.step_count}  定位次数: {sim.locate_count}  总耗时: {sim.elapsed_time:.1f}s")
    viz.fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[save_trajectory_png] 轨迹图已保存: {path}")


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


def run_simulation():
    """主模拟流程：初始化 → 创建桩 state → 执行关卡 → 保存轨迹图"""
    global sim, viz

    # 确保输出目录存在
    os.makedirs(RESULT_DIR, exist_ok=True)

    # 日志 tee：同时输出到终端和文件
    tee = None
    original_stdout = sys.stdout
    if LOG_TO_FILE:
        tee = TeeWriter(LOG_FILE_PATH, original_stdout)
        sys.stdout = tee
        print(f"[log] 日志同时输出到文件: {LOG_FILE_PATH}")

    print("=" * 60)
    print("寻路算法模拟器启动")
    print(f"噪声配置: 定位位置σ={LOCATE_NOISE_STD}cm, 定位朝向σ={LOCATE_ANGLE_NOISE_STD}°, "
          f"动作步长σ={ACTION_ERROR_STD}, 转向σ={TURN_ERROR_STD}°")
    print(f"耗时配置: 定位={LOCATE_TIME_SEC}s/次")
    print("=" * 60)

    # 1. 初始化模拟器状态与可视化
    sim = SimState(INITIAL_POS, INITIAL_ORIENTATION)
    viz = Visualizer()
    viz.update(sim, "模拟器就绪\n等待启动...")

    # 2. 创建桩 RobotState，注入模拟器与可视化
    state = SimRobotState(tag_poses=gl.tag_poses)
    state.attach_sim(sim, viz)
    print("[sim] 已创建 SimRobotState，I/O 方法已替换为桩函数")

    try:
        # 3. 执行关卡主流程
        gl.run_level(state)

    except Exception as e:
        print(f"[run_simulation] 模拟过程异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 4. 保存最终轨迹图
        save_trajectory_png()
        print(f"\n总耗时: {sim.elapsed_time:.1f}s  总动作步数: {sim.step_count}  定位次数: {sim.locate_count}")
        locate_time = sim.locate_count * LOCATE_TIME_SEC
        action_time = sim.elapsed_time - locate_time
        print(f"  其中: 定位耗时 {locate_time:.1f}s ({sim.locate_count}次 × {LOCATE_TIME_SEC}s), "
              f"动作耗时 {action_time:.1f}s ({sim.step_count}步)")
        # 打印完整动作步骤摘要
        if sim.step_actions:
            print(f"\n{'='*60}")
            print(f"  动作步骤摘要（共 {len(sim.step_actions)} 步）")
            print(f"{'='*60}")
            for sa in sim.step_actions:
                print(f"  #{sa['step']:>3d}  {sa['action']:<28s}  "
                      f"位置=({sa['pos'][0]:6.1f}, {sa['pos'][1]:6.1f})  "
                      f"时间={sa['time']:.1f}s")
            print(f"{'='*60}")
        print("\n模拟结束。")
        # 恢复 stdout
        if tee is not None:
            sys.stdout = original_stdout
            tee.close()
            print(f"[log] 日志已保存: {LOG_FILE_PATH}")
        # 交互后端保持窗口；非交互后端直接退出
        if matplotlib.get_backend().lower() != "agg":
            print("关闭图形窗口退出。")
            plt.show()


if __name__ == "__main__":
    run_simulation()
