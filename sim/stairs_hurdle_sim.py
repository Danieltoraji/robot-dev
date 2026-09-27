# -*- coding: utf-8 -*-
"""上下楼梯与识别跨障关卡仿真器（sim/stairs_hurdle_sim.py）

定位与 nine_grid_sim 相同：本模块只提供"世界 + 相机 + 带噪声动作"，
算法（红色带检测、地面度量、FSM）全部走 vision/ core/ levels/ 的真实
代码路径，只重写 I/O 接缝（继承 RobotState）。

场景模型（场景系：原点=第一级台阶基线中点，y 前 x 右 z 上，单位 cm）
------------------------------------------------------------------
  y<0        平地（入口侧，机器人从上一关抵达）
  y=0        第一级台阶立面（红色胶条下半段贴此，z 0~1.5）
  y∈[0,15]   踏面1 z=2.5（胶条上半段贴此，y 0~4）
  y=15       第二级立面
  y∈[15,30]  顶部平台 z=5
  y=30/45    下行两级立面
  y∈[30,45]  踏面 z=2.5/0
  y>45       平地，红色横杆在 y=45+bar_dist（25~40 随机）
相机模型（2026-09-25 重制）
--------------------------
站立光心离地 CAM_HEIGHT=33.9cm、相对俯仰舵机的安装下俯偏移
CAM_PITCH_OFFSET_DEG=19.1°（**必须计入**），都由 2026-09-25 卷尺标定给出
（tools/calib_ruler_profile.py，14 点，RMS 2.72px）。
旧版漏掉偏移且用 39cm，渲染出的可见带是 [13.9, 177.7]cm，而现场是
[3.5, 71]cm——远近两头都反着。

动作"真值"（噪声仿 nine_grid_sim）
----------------------------------
位移与转角取自 core/motion_calib（实测单一真源），并叠加实测散布。

⚠️ **不要在这里写与关卡共享的"猜测值"。** 历史事故：仿真把
turn_left_small_step 写成 N(2.0, 1.5)°，与关卡共享同一个错数（真值
8.625°），于是端到端 20/20 全绿，而真机对正阶段必然来回过冲。
仿真读实测真源，关卡把同一份数值当预测值——两者角色不同，数值同源。

动作组净位移（CLIMB/DOWN_FWD_CM）**仍是名义值，尚未实测**，
取值使场景自洽：两次上楼落在顶部平台（y 15~30），两次下楼落在平地
（y≥45）。跨栏的 HURDLE_FWD_CM=18cm 是现场实测。上楼/下楼的航向偏置
同理（参考实现在顶部固定"右转 2 小步"补偿，按实测右转 5.2°/步折合约
10.4°，即两次上楼累计左偏约 10°）。

未建模：楼梯木结构的遮挡与外观（合成帧只渲染红色目标+纯色背景；
遮挡已在方案里几何论证过：接近段横杆越过台阶沿可见）、拍照耗时、
光照/白平衡、相机-机体偏移（CAM_BODY_OFFSET=0）、三楼以上的机体自遮挡。

运行
----
    python -m sim.stairs_hurdle_sim [--seed N] [--quiet]
集成测试：python tests/test_stairs_hurdle_sim.py
"""

import argparse
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2

from core import motion_calib as mc
from core.camera_config import (
    CAMERA_INTRINSIC, CAMERA_DISTORTION, HEAD_CENTER, SERVO_DEG_PER_US,
)
from core.robot_core import RobotState

FRAME_W, FRAME_H = 2592, 1944
_IMAGE_RECT = np.array([[0.0, 0.0], [FRAME_W, 0.0],
                        [FRAME_W, FRAME_H], [0.0, FRAME_H]],
                       dtype=np.float32)

# 红色 BGR（HSV(3,200,200)），与 nine_grid_sim 同法生成
RED_BGR = cv2.cvtColor(np.full((1, 1, 3), (3, 200, 200), np.uint8),
                       cv2.COLOR_HSV2BGR)[0, 0].tolist()


def camera_rotation(bearing_deg, pitch_deg):
    """世界->相机旋转（与 tests/test_ground_homography 同构造，刻意独立副本）"""
    a = np.radians(pitch_deg)
    f = np.radians(bearing_deg)
    sa, ca = np.sin(a), np.cos(a)
    sf, cf = np.sin(f), np.cos(f)
    r = np.array([cf, -sf, 0.0])
    d = np.array([-sa * sf, -sa * cf, -ca])
    v = np.array([ca * sf, ca * cf, -sa])
    return np.vstack([r, d, v])


def tessellate_x(corners, n=12):
    """把 (bl,br,tr,tl) 四边形沿 x 细分成 n 段小四边形。

    畸变下地面直线投影像素是曲线，单段长边的直线弦与曲线差可达 ~1.5cm
    （真值被渲染污染）。细分后每段弦差可忽略。
    """
    bl, br, tr, tl = corners
    edges = np.linspace(0.0, 1.0, n + 1)
    quads = []
    for a, b in zip(edges[:-1], edges[1:]):
        quads.append(np.array([bl + (br - bl) * a, bl + (br - bl) * b,
                               tl + (tr - tl) * b, tl + (tr - tl) * a]))
    return quads


def project_quad(frame, R, C, corners, color):
    """3D 四边形(4,3)渲染到帧（含真实内参/畸变/画幅裁切）。成功返回 True"""
    pc = (R @ (np.asarray(corners, dtype=np.float64) - C).T).T
    if np.any(pc[:, 2] <= 0.05):
        return False
    norm = pc[:, :2] / pc[:, 2:3]
    pix, _ = cv2.projectPoints(
        np.column_stack([norm, np.ones(4)]).reshape(-1, 1, 3).astype(np.float32),
        np.zeros(3), np.zeros(3), CAMERA_INTRINSIC, CAMERA_DISTORTION)
    pix = pix[:, 0, :]
    if not np.all(np.isfinite(pix)):
        return False
    pix = np.clip(pix, -1e6, 1e6)
    area, inter = cv2.intersectConvexConvex(pix.astype(np.float32), _IMAGE_RECT)
    if inter is None or len(inter) < 3 or area <= 1e-9:
        return False
    cv2.fillPoly(frame, [np.round(inter).astype(np.int32)], color)
    return True


class StairsScene:
    """场地几何真值（cm）。红带方向偏转用于方位真值扰动测试。"""

    def __init__(self, tape_width=40.0, tape_yaw_deg=0.0, tape_visible=True,
                 bar_dist=30.0, bar_yaw_deg=0.0, bar_visible=True,
                 bar_half_len=40.0):
        self.step_rise = 2.5
        self.step_run = 15.0
        self.tape_width = tape_width
        self.tape_yaw = np.radians(tape_yaw_deg)
        self.tape_visible = tape_visible
        self.bar_dist = bar_dist            # 距楼梯下沿出口（y=45）
        self.bar_yaw = np.radians(bar_yaw_deg)
        self.bar_visible = bar_visible
        self.bar_half_len = bar_half_len

        self.bar_y = 45.0 + bar_dist        # 横杆地面线位置（场景系 y）

    def _rot_x(self, pts, yaw):
        """绕 z 轴旋转（方向偏转用）"""
        c, s = np.cos(yaw), np.sin(yaw)
        return pts @ np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]).T

    def red_quads(self):
        """所有红色目标四边形（场景系，长边已细分逼近畸变曲线）"""
        quads = []
        w2 = self.tape_width / 2.0
        if self.tape_visible:
            rise = self.step_rise * 0.6   # 立面段 1.5cm
            tread = 4.0                    # 踏面段深 4cm
            riser = np.array([[-w2, 0.0, 0.0], [w2, 0.0, 0.0],
                              [w2, 0.0, rise], [-w2, 0.0, rise]])
            tread_q = np.array([[-w2, 0.0, self.step_rise], [w2, 0.0, self.step_rise],
                                [w2, tread, self.step_rise], [-w2, tread, self.step_rise]])
            for base in (riser, tread_q):
                quads += [self._rot_x(q, self.tape_yaw)
                          for q in tessellate_x(base, n=12)]
        if self.bar_visible:
            L = self.bar_half_len
            t = 2.0                       # 截面 2×2cm（说明书）
            y0 = self.bar_y               # 木条**近棱**的地面线 = bar_y
            # ⚠ 必须渲染成"近立面 + 顶面"两片，不能只画中间一片。
            # 检测器取的是红色区域的**下沿**，而木条下沿就是近立面底边
            # （y = bar_y）。旧版只在 bar_y + t/2 画一片竖直面，等于把可检出的
            # 地面线整体后移 1cm——"起跨点离杆多远"的断言会跟着错 1cm，
            # 而 1cm 正是这里唯一在乎的尺度。
            near = np.array([[-L, y0, 0.0], [L, y0, 0.0],
                             [L, y0, t], [-L, y0, t]])
            top = np.array([[-L, y0, t], [L, y0, t],
                            [L, y0 + t, t], [-L, y0 + t, t]])
            for base in (near, top):
                quads += [self._rot_x(q, self.bar_yaw)
                          for q in tessellate_x(base, n=24)]
        return quads


class SimStairsRobot(RobotState):
    """场景系位姿 + 打滑噪声动作 + 合成相机（红带渲染）

    动作真值取自 core/motion_calib（实测单一真源），噪声按其散布给。
    动作组的净位移与航向偏置是**名义值**，见模块 docstring。
    """

    #: 站立时相机光心离地高度（cm）——2026-09-25 卷尺标定：
    #: h_eff（光心离刻度平面）32.73 + 卷尺厚 1.20 = 33.93
    CAM_HEIGHT = 33.9
    #: 相机相对俯仰舵机的安装下俯偏移（度）——标定值：
    #: 1100 档拟合俯角 55.11° − 名义 (1500-1100)*0.09=36.0° = 19.11°
    CAM_PITCH_OFFSET_DEG = 19.1
    #: 相机-机体水平偏移（未建模）
    CAM_BODY_OFFSET = 0.0
    #: 脚尖在光心地面投影**前方**的距离（cm）——2026-09-25 卷尺标定 +4.24，
    #: 现场复核 +4.55~+4.66 恒定。相机装在头部，其地面投影大致落在脚踝上方，
    #: 脚尖自然在前。
    #:
    #: ⚠ 这个数不是可有可无的细节：测距工具读的是"光心投影→目标"，而"机器人
    #:   在栏杆前几厘米"说的是"脚尖→目标"。**工具读数 1cm 时脚尖已经越过木条
    #:   3.24cm，等于站在杆上**。本仿真把它当真值建模，好让测试能验证
    #:   "跨栏之前脚尖没有先越过木条"。
    TOE_AHEAD_CM = 4.24

    #: 动作组净位移（cm，名义值）：两次上楼落在顶部平台、两次下楼落在平地
    #: 取 12cm 而不是踏面深度 15cm，是给 ±10% 的打滑噪声留余量——旧版 18cm
    #: 两次上楼会冲到 y≈36（平台只到 30），测出来的"通过"在一个不存在的世界里
    CLIMB_FWD_CM = 12.0
    DOWN_FWD_CM = 12.0
    #: 跨栏动作组净前进 18cm——2026-09-25 现场实测（旧值 10cm 是估的，小了一半）
    HURDLE_FWD_CM = 18.0

    #: 上楼/下楼造成的航向偏置与散布（度，名义值）
    CLIMB_YAW_BIAS_DEG = -5.0
    CLIMB_YAW_SIGMA_DEG = 2.0
    DOWN_YAW_BIAS_DEG = -1.5
    DOWN_YAW_SIGMA_DEG = 1.5

    #: 未实测动作的名义长度（仅仿真用；关卡不得当真值）。
    #: ⚠ 这两个数以 core/motion_calib 的**现场标定值**为准：填了就用实测值，
    #:   没填才退回名义值。仿真读的是"实测真源"，不是从关卡抄过来的猜测。
    FWD_SMALL_CM_NOMINAL = mc.FWD_SMALL_STEP_CM or 1.5
    BACK_STEP_CM_NOMINAL = mc.BACK_STEP_CM or 3.2
    RIGHT_MOVE_CM_NOMINAL = 2.5

    #: go_forward_one_step 的单步相对散布。
    #:
    #: 旧版写的是 15%——那**不是实测值，是随手填的**：实测 ±3σ 只有 0.055cm
    #: （占步长 2.1%），组间极差 0.9cm/20步（2.5%）。15% 在 1.8cm 步长上是
    #: σ=0.27cm、3σ=0.81cm，比实测大一个量级，等于让关卡去对付一个不存在的
    #: 机器人（末段"能不能贴到杆前 1.2cm"这种判断会被它整个淹掉）。
    FWD_STEP_SIGMA_REL = (mc.FWD_STEP_3SIGMA_CM / 3.0) / mc.FWD_STEP_CM_LEGACY
    #: 局间步长整体漂移（每局抽一次，作用于所有整步）。
    #: 依据：实测两次量 go_forward_one_step 得到 2.652 与 1.8（差 1.47 倍）——
    #: **局间**步长确实会整体变（重装、电量、地面）。所以模型是
    #: "局间大漂移 + 局内小散布"，而不是把大漂移摊到每一步上。
    #: 关卡那边靠 _observe_step 在局内实测并把上界往上修来应对。
    FWD_SCALE_SIGMA = 0.12

    def __init__(self, scene: StairsScene, seed=0, start=(0.0, -60.0),
                 heading_deg=0.0, wobble_sigma_deg=0.3):
        super().__init__(tag_poses={})
        self.scene = scene
        self.pos = np.array(start, dtype=np.float64)
        self.z = 0.0                        # 脚下平面高度（台阶上为 2.5/5）
        self.heading = heading_deg          # 右正（度）
        self.pitch = 1500
        self.head = HEAD_CENTER
        self.rng = np.random.RandomState(seed)
        self.wobble_sigma_deg = wobble_sigma_deg
        self.n_captures = 0
        self.action_log = []
        # 组间漂移：整局内转角整体偏移（实测左转组间极差 25°/20步 =
        # 单步 ±7%；右转仅 ±1%）。建模成每局一次的缩放，逼关卡
        # "转一步、复测一次"，而不是把实测值当冻结常数。
        self._turn_scale = {
            "left": 1.0 + float(self.rng.normal(0.0, 0.030)),
            "right": 1.0 + float(self.rng.normal(0.0, 0.005)),
        }
        # 局间步长整体漂移（每局一次；见 FWD_SCALE_SIGMA 说明）
        self._step_scale = 1.0 + float(self.rng.normal(0.0, self.FWD_SCALE_SIGMA))

    # ---- I/O 接缝 ----

    def set_head(self, pulse, move_time_ms=500, force=False):
        self.head = pulse
        self.current_head_pulse = pulse

    def set_pitch(self, pulse, move_time_ms=500, force=False):
        # 必须同步 current_pitch_pulse：关卡的变档重试靠它恢复原档，
        # 旧版只改 self.pitch，一次重试之后机器人就被永久留在错误档位
        self.pitch = pulse
        self.current_pitch_pulse = pulse

    def run_action(self, name, times=1):
        for _ in range(max(1, times)):
            self._apply_action(name)
        self.action_log.append((name, times))

    def _apply_action(self, name):
        th = np.radians(self.heading)
        fwd = np.array([np.sin(th), np.cos(th)])
        right = np.array([np.cos(th), -np.sin(th)])
        slip = self.rng.normal
        if name == "go_forward_one_step":
            # 实测散布（很小）+ 本局的整体步长漂移（可能很大）
            self.pos += fwd * (mc.FWD_STEP_CM * self._step_scale
                               * (1 + slip(0, self.FWD_STEP_SIGMA_REL)))
        elif name == "go_forward_one_small_step":
            self.pos += fwd * (self.FWD_SMALL_CM_NOMINAL * self._step_scale
                               * (1 + slip(0, self.FWD_STEP_SIGMA_REL)))
        elif name == "back_one_step":
            # 后退：本局步长缩放同样作用于它（同一套腿部动作幅度）
            self.pos -= fwd * (self.BACK_STEP_CM_NOMINAL * self._step_scale
                               * (1 + slip(0, 0.15)))
        elif name == "left_move":
            self.pos -= right * mc.LEFT_MOVE_CM * (1 + slip(0, 0.20))
        elif name == "right_move":
            self.pos += right * self.RIGHT_MOVE_CM_NOMINAL * (1 + slip(0, 0.20))
        elif name == "turn_left_small_step":
            # 实测 8.625°/步（±3σ 1.875），每局还有整体缩放漂移
            self.heading -= max(0.0, self.rng.normal(
                mc.TURN_LEFT_DEG * self._turn_scale["left"],
                mc.TURN_LEFT_3SIGMA_DEG / 3.0))
        elif name == "turn_right_small_step":
            # 实测 5.200°/步（±3σ 0.150），比左转稳得多
            self.heading += max(0.0, self.rng.normal(
                mc.TURN_RIGHT_DEG * self._turn_scale["right"],
                mc.TURN_RIGHT_3SIGMA_DEG / 3.0))
        elif name == "turn_left":
            self.heading -= 22.0 * (1 + slip(0, 0.10))
        elif name == "turn_right":
            self.heading += 25.7 * (1 + slip(0, 0.10))
        elif name == "stand":
            pass
        elif name == "climb_stairs":
            self.pos += fwd * self.CLIMB_FWD_CM * (1 + slip(0, 0.10))
            self.z += self.scene.step_rise
            self.heading += float(self.rng.normal(self.CLIMB_YAW_BIAS_DEG,
                                                  self.CLIMB_YAW_SIGMA_DEG))
        elif name == "down_floor":
            self.pos += fwd * self.DOWN_FWD_CM * (1 + slip(0, 0.10))
            self.z -= self.scene.step_rise
            self.heading += float(self.rng.normal(self.DOWN_YAW_BIAS_DEG,
                                                  self.DOWN_YAW_SIGMA_DEG))
        elif name == "hurdles":
            self.pos += fwd * self.HURDLE_FWD_CM * (1 + slip(0, 0.10))
        else:
            raise ValueError(f"仿真未实现动作: {name}")

    def capture_frame(self):
        self.n_captures += 1
        bearing = self.heading - (self.head - HEAD_CENTER) * SERVO_DEG_PER_US
        wobble = 0.0
        if self.wobble_sigma_deg > 0:
            wobble = float(self.rng.normal(0.0, self.wobble_sigma_deg))
        # 有效俯角 = 名义舵机角 + 安装下俯偏移（漏掉偏移可见带会整体失真）
        pitch_deg = ((1500 - self.pitch) * SERVO_DEG_PER_US
                     + self.CAM_PITCH_OFFSET_DEG + wobble)
        R = camera_rotation(bearing, pitch_deg)
        C = np.array([self.pos[0], self.pos[1], self.CAM_HEIGHT + self.z])
        frame = np.full((FRAME_H, FRAME_W, 3), 90, np.uint8)
        for corners in self.scene.red_quads():
            project_quad(frame, R, C, corners, RED_BGR)
        return frame


def make_frame(red_quads, bearing_deg=0.0, pitch_deg=45.0, cam_xy=(0.0, 0.0),
               cam_z=39.0, color=RED_BGR):
    """独立帧合成工具（精度扫描/HSV 鲁棒性测试用，不经过 RobotState）"""
    R = camera_rotation(bearing_deg, pitch_deg)
    C = np.array([cam_xy[0], cam_xy[1], cam_z])
    frame = np.full((FRAME_H, FRAME_W, 3), 90, np.uint8)
    for corners in red_quads:
        project_quad(frame, R, C, corners, color)
    return frame


def bar_line_quads(forward_cm, bearing_deg=0.0, lateral_cm=0.0,
                   half_len=40.0, height_off_cm=0.0):
    """在机器人本地系生成一条"横杆下沿"合成目标（回归测试用）

    forward: 下沿中点前向距离；bearing: 直线方向偏角；lateral: 中点横偏；
    height_off: 下沿相对脚下平面的高度（爬楼段方位场景用）。
    返回细分四边形列表（含 2cm 高度让红色可见，下沿严格在 height_off 高度线
    上；细分是为逼近畸变曲线，避免渲染弦差污染真值）。
    """
    th = np.radians(bearing_deg)
    dx, dy = np.cos(th), np.sin(th)          # 直线方向（沿带长）
    cx, cy = lateral_cm, forward_cm          # 下沿中点
    p0 = np.array([cx - dx * half_len, cy - dy * half_len, height_off_cm])
    p1 = np.array([cx + dx * half_len, cy + dy * half_len, height_off_cm])
    up = np.array([0.0, 0.0, 2.0])
    return tessellate_x(np.array([p0, p1, p1 + up, p0 + up]), n=24)


def main(argv=None):
    ap = argparse.ArgumentParser(description="上下楼梯与识别跨障关卡仿真")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    from levels.stairs_hurdle import StairsHurdleLevel
    scene = StairsScene(bar_dist=float(np.random.RandomState(args.seed).choice([25.0, 32.0, 40.0])))
    # 起点：入口光束在楼梯下沿 30cm 以内，机器人摆在光束之外
    robot = SimStairsRobot(scene, seed=args.seed,
                           start=(0.0, -45.0), heading_deg=0.0)
    level = StairsHurdleLevel(robot)
    ok = level.run_level()
    print(f"[sim] ok={ok} 到达 y={robot.pos[1]:.1f} 拍照 {robot.n_captures} 张 "
          f"动作 {len(robot.action_log)} 次")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
