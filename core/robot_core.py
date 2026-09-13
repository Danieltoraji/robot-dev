# -*- coding: utf-8 -*-
"""
通用机器人核心层（robot_core.py）

提供所有关卡共享的基础能力：
  - 硬件初始化（动作组、舵机、相机内参）
  - RobotState 类：封装机器人运行时状态（位置/朝向/头部脉宽），
    并提供定位（AprilTag + PnP）、动作执行、头部舵机控制等方法。
  - 纯几何工具函数（distance_point_to_rect）

各关卡只需 import RobotState 并传入自己的 tag_poses 即可复用全部定位能力。
关卡特定的赛道数据（墙壁、目标点等）和导航算法留在关卡文件中。
"""

import json
import os
import time
import subprocess
import numpy as np

# 硬件/视觉库导入：在非机器人环境用 try/except 屏蔽，
# 模拟器通过继承 RobotState 重写 I/O 方法，不调用真实硬件。
try:
    import hiwonder.ActionGroupControl as AGC  # 动作库
except Exception:
    AGC = None
try:
    import apriltag
except Exception:
    apriltag = None
try:
    import cv2
except Exception:
    cv2 = None
try:
    import hiwonder.ros_robot_controller_sdk as rrc
    from hiwonder.Controller import Controller
    board = rrc.Board()
    ctl = Controller(board)
except Exception:
    rrc = None
    ctl = None
    board = None

# 相机内参/头部舵机/PnP门控常量：单一真源在 camera_config.py，
# 此处导入以保持 robot_core.CAMERA_INTRINSIC 等既有引用方式不变。
from core.camera_config import (
    CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_INTRINSIC, CAMERA_DISTORTION,
    HEAD_CENTER, HEAD_RIGHT, HEAD_LEFT, HEAD_WIDE_RIGHT, HEAD_WIDE_LEFT,
    HEAD_MOVE_TIME_MS, HEAD_MOVE_TIME_MIN_MS, SERVO_DEG_PER_US, PITCH_UP_PULSE,
    PNP_FIELD_MIN, PNP_FIELD_MAX, PNP_CAM_Z_MIN, PNP_CAM_Z_MAX,
    PNP_ORI_Z_MAX, PNP_ORI_XY_MIN, PNP_REPROJ_ERR_MAX_PX, reproj_gate_px,
    AMBIGUITY_SEP_MIN_CM, EXPECT_SELECT_MARGIN_CM, EXPECT_RADIUS_MARGIN_CM,
)

from core.paths import RESULT_DIR


# =====================================================================
# 定位参数常量
# =====================================================================
MAX_LOCATE_RETRIES = 5  # 定位失败最大重试次数
# 注意：导航决策常量（朝向阈值、位置阈值、避障阈值、停靠时间、安全余量等）
# 属于关卡层，由各关卡在 levels/<level>.py 中自行定义，不要放回 core，
# 以免出现「core 与关卡各持一份」的双重真相、造成跨关卡混乱。

# 多视角联合定位外参文件（tools/field_calib/survey_field.py 产出；缺失时降级为缺省外参）
MULTIVIEW_EXTRINSICS_PATH = os.path.join(RESULT_DIR, "multiview_extrinsics.json")

# =====================================================================
# 相机参数锁定（2026-09-11：消除现场光照/自动白平衡漂移）
# =====================================================================
# 现场实测（真机探针帧 photo_1789213170）：同一场地同一相机的白底板 BGR 从
# [153,155,149] 漂到 [209,186,180]（R/B 差 16%），粉贴纸 H 由 168 掉到 141，
# 整块掉出手写窗口而漏检。事后归一化能救，但**源头锁住才是根治**：
# fswebcam 每次调用都重新测光/白平衡（命令里 `-S 3` 只是丢 3 帧），
# 故在一局开始时用 v4l2-ctl 把白平衡与对焦钉死，全程不再漂。
# 取值来源：真机 `v4l2-ctl --list-ctrls` 读到的当前值（4000K/313/166）。
# 曝光默认**不锁**（CAM_LOCK_EXPOSURE=False）：亮度漂移已由
# vision.nine_grid_detector.normalize_illumination 归一化消除；而手动曝光在
# 新场地上有欠曝/过曝风险，且过曝是不可逆的信息丢失。需要时再开。
#
# ===== 2026-09-13 场地实测更新（机器人上场地后逐项扫出来的）=====
# 自动曝光（auto_exposure=3）在该场地会把白板打到过曝 21~54%，橙/黄贴纸被
# 洗成白色 → 12 帧里橙 0 次、黄 0 次检出；改手动曝光后稳定：
#   曝光 60/100/150/200/250 → 过曝 0%/0%/59%/57%/21%，检出 5/7/2/2/3
#   ⇒ 取 exposure_time_absolute=100（过曝 ≈1.3%，连拍 6 张均亮 143.4 完全一致）
# 白平衡温度扫描（固定曝光 100，看白板是否中性，R/B 越接近 1 越准）：
#   2800K/3400K 全画面偏色（白点估不出）→ 4000K R/B 0.54 → 4600K 0.60
#   → **5200K 0.92（判据回到"低饱和"）** → **5800K 0.96、白点均值 162≈归一化目标**
#   → 6500K 1.28 偏蓝   ⇒ 取 5800K（原 4000K 是上一场地的值，在本场地太暖）
# 对焦：`focus_absolute` 写入被驱动拒绝（Permission denied，值停在默认 270）；
# 有效的只有 `focus_automatic_continuous=0`（关掉连续对焦、锁在当前焦平面）。
# 故 CAM_FOCUS_ABS 记为实际生效值 270；写不进去不影响（不再来回拉风箱）。
CAM_LOCK_CONTROLS_ENABLED = True
CAM_LOCK_EXPOSURE = True
CAM_V4L2_DEVICE = "/dev/video0"
CAM_WB_TEMPERATURE_K = 5800
CAM_EXPOSURE_ABS = 100
CAM_FOCUS_ABS = 270
# 现场复扫（换场地/换灯后照做）：python tools/field_camera_sweep.py
# 判据：白点 R/B 最接近 1、过曝 <5%、同一机位连拍均亮一致。
_CAM_LOCK_DONE = None      # 已应用的配置（同一次运行内不重复设置）


def camera_lock_controls(device=None, wb_k=None, exposure=None, focus=None,
                         lock_exposure=None):
    """→ 有序的 (控制名, 值) 列表（纯函数，便于单测；不含 IO）

    auto_exposure=1 = Manual、white_balance_automatic=0 = 手动白平衡、
    focus_automatic_continuous=0 = 手动对焦（V4L2 菜单约定）。
    """
    device = device or CAM_V4L2_DEVICE
    wb_k = CAM_WB_TEMPERATURE_K if wb_k is None else wb_k
    exposure = CAM_EXPOSURE_ABS if exposure is None else exposure
    focus = CAM_FOCUS_ABS if focus is None else focus
    lock_exposure = CAM_LOCK_EXPOSURE if lock_exposure is None else lock_exposure
    ctrls = [("white_balance_automatic", 0),
             ("white_balance_temperature", int(wb_k)),
             ("focus_automatic_continuous", 0),
             ("focus_absolute", int(focus))]
    if lock_exposure:
        ctrls += [("auto_exposure", 1),
                  ("exposure_time_absolute", int(exposure))]
    return device, ctrls


def build_camera_lock_cmd(device, ctrls):
    """设置命令（一次调用设完全部控制，避免中间态被 fswebcam 抢到）"""
    pairs = ",".join(f"{k}={v}" for k, v in ctrls)
    return f"v4l2-ctl -d {device} -c {pairs}"


def build_camera_lock_cmds(device, ctrls):
    """**逐条**设置命令 —— 真机实测（2026-09-13）：合成一条 `-c k1=v1,k2=v2`
    时，只要其中一个控制没有写权限（`focus_absolute: Permission denied`），
    v4l2-ctl 整条返回 255、**其余控制也不会被设置**（白平衡就白锁了）。
    逐条设置后单点失败不影响其它控制。
    """
    return [(k, v, f"v4l2-ctl -d {device} -c {k}={v}") for k, v in ctrls]


def build_camera_readback_cmd(device, ctrls):
    """读回命令（验收/诊断：确认锁定真的生效）"""
    names = ",".join(k for k, _v in ctrls)
    return f"v4l2-ctl -d {device} --get-ctrl={names}"


def parse_v4l2_values(text):
    """`v4l2-ctl --get-ctrl` 输出 → {控制名: 值}（容忍空白与冒号变体）"""
    out = {}
    for line in (text or "").splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k, v = k.strip(), v.strip()
        try:
            out[k] = int(v)
        except ValueError:
            continue
    return out


def lock_camera_controls(device=None, wb_k=None, exposure=None, focus=None,
                         lock_exposure=None, force=False, quiet=False):
    """锁定白平衡/对焦（可选曝光）→ (是否成功, 明细字典)

    非 Linux（PC 开发）或设备不存在/无 v4l2-ctl 时返回 (False, 原因)，
    **不抛异常**：锁不上只是精度风险，不该让整局跑不起来（归一化仍可兜）。
    """
    device, ctrls = camera_lock_controls(device, wb_k, exposure, focus,
                                         lock_exposure)
    info = {"device": device, "ctrls": dict(ctrls), "applied": False,
            "readback": {}, "mismatch": [], "failed": [], "reason": ""}
    if not CAM_LOCK_CONTROLS_ENABLED and not force:
        info["reason"] = "锁定已关闭(CAM_LOCK_CONTROLS_ENABLED=False)"
        return False, info
    global _CAM_LOCK_DONE
    if _CAM_LOCK_DONE == info["ctrls"] and not force:
        info["applied"] = True
        info["reason"] = "本次运行已锁定过，跳过"
        return True, info
    if os.name != "posix" or not os.path.exists(device):
        info["reason"] = f"非 Linux 或无 {device}（PC 开发环境按未锁处理）"
        return False, info
    failed = []
    for name, want, cmd in build_camera_lock_cmds(device, ctrls):
        try:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                               timeout=10)
        except (OSError, subprocess.SubprocessError) as e:
            failed.append(f"{name}: 调用失败({e})")
            continue
        if r.returncode != 0:
            failed.append(f"{name}: rc={r.returncode} "
                          f"{r.stderr.strip().splitlines()[-1][:60] if r.stderr.strip() else ''}")
    info["failed"] = failed
    info["applied"] = len(failed) < len(ctrls)
    if not info["applied"]:
        info["reason"] = "全部控制设置失败: " + "; ".join(failed)
        return False, info
    if failed:
        info["reason"] = "部分控制设置失败: " + "; ".join(failed)
    try:
        rb = subprocess.run(build_camera_readback_cmd(device, ctrls),
                            shell=True, capture_output=True, text=True,
                            timeout=10)
        got = parse_v4l2_values(rb.stdout)
    except (OSError, subprocess.SubprocessError):
        got = {}
    info["readback"] = got
    for k, want in ctrls:
        if k in got and got[k] != want:
            info["mismatch"].append(f"{k}: 期望 {want} 实读 {got[k]}")
    _CAM_LOCK_DONE = dict(ctrls)
    if not quiet:
        print(f"[相机] 锁定 {device}: "
              + ", ".join(f"{k}={v}" for k, v in ctrls)
              + (f"；失败 {failed}" if failed else "")
              + (f"；读回不一致 {info['mismatch']}" if info["mismatch"] else "；读回一致"))
    return info["applied"], info


# =====================================================================
# 位置连续性守卫（2026-08-30，依据真机 trace 一次 12cm 跳变坏定位）
# =====================================================================
# 共面标签对的错误分支可能通过重投影门控（真机实测 (22,38)→(34,41) 跳变，
# 触发危险的避墙绕路）。定位结果若偏离上次成功定位超过
# 「动作位移预算 + 裕度」则拒绝本次解，扫描换下一档重试。
# 预算是粗略上界（含动作误差与定位噪声），不用关卡层的标定精值。
MOTION_BUDGET_CM = {
    "go_forward": 5.5, "go_forward_one_step": 2.5,
    "go_forward_one_small_step": 1.5,
    "back_one_step": 3.5,
    "back": 3.5, "left_move": 2.5, "right_move": 2.5,
    "turn_left": 8.0, "turn_right": 8.0, "stand": 0.0,
    # 上下楼梯与识别跨障关卡的场景动作（净水平位移上界，M3 实测后修正；
    # 预算仅用于连续性日志，不拦截动作）
    "climb_stairs": 20.0, "down_floor": 20.0, "hurdles": 15.0,
}
CONTINUITY_MARGIN_CM = 5.0  # 预算之外的额外裕度（定位噪声 + 动作散布）

# =====================================================================
# AprilTag 角点顺序（P0 修复 · 真机实测确认）
# =====================================================================
# tag_poses 世界坐标顺序约定（levels/goodluck.py）：左上→右上→右下→左下。
# 2026-08-24 真机实测（tools/field_calib/verify_corner_order.py，标签 36/37/38/39 共 6 组观测，
# 含正对与斜视角）确认：apriltag 库返回的 r.corners 顺序与 tag_poses 完全一致，
# 即恒等映射。此常量把该事实显式化，防止换库/换版本/重贴标签后再次踩坑；
# 若未来顺序变化，只需修改此常量并重跑 tools/field_calib/verify_corner_order.py 验证。
TAG_CORNER_PERM = np.array([0, 1, 2, 3], dtype=np.int64)


# =====================================================================
# 纯几何工具函数（模块级，不依赖任何关卡数据）
# =====================================================================

def distance_point_to_rect(pos, rect):
    """点到矩形的最短距离（点在矩形内返回0）

    pos: [x, y]，rect: [x_min, x_max, y_min, y_max]
    """
    x, y = pos[0], pos[1]
    x_min, x_max, y_min, y_max = rect
    dx = max(x_min - x, 0, x - x_max)
    dy = max(y_min - y, 0, y - y_max)
    return float(np.sqrt(dx * dx + dy * dy))


def solve_pnp_pose(objlist, imglist):
    """solvePnP + 提取相机位姿 + 平均重投影误差

    返回 (pos_3d, ori_3d, reproj_err) 或 None（求解失败）。
    pos_3d/ori_3d 为世界坐标系下的相机位置 / 光轴单位向量。
    """
    obj = np.asarray(objlist, dtype=np.float64)
    img = np.asarray(imglist, dtype=np.float64)
    ifsuccess, rvec, tvec = cv2.solvePnP(obj, img, CAMERA_INTRINSIC, CAMERA_DISTORTION)
    if not ifsuccess:
        return None
    proj, _ = cv2.projectPoints(obj, rvec, tvec, CAMERA_INTRINSIC, CAMERA_DISTORTION)
    reproj_err = float(np.mean(np.linalg.norm(proj[:, 0, :] - img, axis=1)))
    rotateMatrix = cv2.Rodrigues(rvec)[0]
    pos_3d = (-np.linalg.inv(rotateMatrix) @ tvec).flatten()
    ori_3d = (np.linalg.inv(rotateMatrix) @ (np.array([[0.0], [0.0], [1.0]]) - tvec)).flatten() - pos_3d
    norm = np.linalg.norm(ori_3d)
    if norm != 0:
        ori_3d = ori_3d / norm
    return pos_3d, ori_3d, reproj_err


def tag_plane_axis(corners):
    """返回标签所在平面的判别键（轴名@常数，如 "x@45.0"）；无唯一平面返回 None

    用于帧救援的共面判定：非共面标签子集的 PnP 无平面歧义，可安全接受。
    """
    corners = np.asarray(corners, dtype=np.float64)
    for axis, name in ((0, "x"), (1, "y"), (2, "z")):
        col = corners[:, axis]
        if float(col.max() - col.min()) < 0.5:
            return f"{name}@{float(col.mean()):.1f}"
    return None


def pnp_pose_problems(pos_3d, ori_3d, reproj_err, n_tags=1):
    """PnP 位姿合理性检查，返回问题列表（空列表 = 通过）

    拦截角点顺序错误/镜像等「重投影误差小但物理不合理」的假位姿。
    重投影阈值按标签数自适应：单标签 2px（正确解 <1px、镜像解 ≥2.8px，
    可分隔）；多标签 8px（标签世界坐标残余不一致 ~4px，错误分支仍 >15px）。
    """
    problems = []
    x, y, z = pos_3d
    if not (PNP_FIELD_MIN <= x <= PNP_FIELD_MAX and PNP_FIELD_MIN <= y <= PNP_FIELD_MAX):
        problems.append(f"相机位置({x:.1f},{y:.1f})超出场地范围")
    if not (PNP_CAM_Z_MIN <= z <= PNP_CAM_Z_MAX):
        problems.append(f"相机高度z={z:.1f}cm不合理")
    if abs(ori_3d[2]) > PNP_ORI_Z_MAX:
        problems.append(f"相机朝向不水平(z分量{ori_3d[2]:.2f})")
    if np.linalg.norm(ori_3d[:2]) < PNP_ORI_XY_MIN:
        problems.append("朝向接近垂直")
    if reproj_err > reproj_gate_px(n_tags):
        problems.append(f"重投影误差{reproj_err:.1f}px过大(阈值{reproj_gate_px(n_tags):.0f}px)")
    return problems


# =====================================================================
# RobotState 类
# =====================================================================

class RobotState:
    """机器人运行时状态与基础 I/O 能力

    封装位置/朝向/头部脉宽等运行时状态，提供定位、动作执行、
    头部舵机控制等方法。各关卡通过传入 tag_poses 初始化即可复用。

    属性:
        current_position: 当前机体位置 [x, y]（cm），None 表示未定位
        current_orientation: 当前机体朝向 [dx, dy]（单位向量），None 表示未定位
        current_head_pulse: 头部舵机当前脉宽
        tag_poses: AprilTag 世界坐标字典，用于 PnP 求解
    """

    def __init__(self, tag_poses=None):
        self.current_position = None
        self.current_orientation = None
        self.current_head_pulse = HEAD_CENTER
        self.current_pitch_pulse = 1500  # 俯仰舵机（ID1），1500=水平，越小越低头
        self.tag_poses = tag_poses if tag_poses is not None else {}
        # 多视角外参：优先读 survey_field.py 标定产物；缺失则用缺省值
        # （e=0 即「光心=机体中心」假设），首次联合求解时打印告警
        from core.multiview_pose import load_extrinsics, normalize_extrinsics
        ext = load_extrinsics(MULTIVIEW_EXTRINSICS_PATH)
        self.multiview_extrinsics = normalize_extrinsics(ext)
        self._extrinsics_from_file = ext is not None
        self._extrinsics_warned = False
        # 位置连续性守卫：上次成功定位的位置 + 自此以来的动作位移预算
        self._last_located_pos = None
        self.motion_budget_cm = 0.0

    # -----------------------------------------------------------------
    # 动作执行
    # -----------------------------------------------------------------

    def run_action(self, name, times=1):
        """执行动作组"""
        AGC.runActionGroup(name, times=times)

    def act(self, name, times=1):
        """执行动作并累积位移预算（连续性守卫用）；定位决策统一走本方法

        直接调 run_action 不计预算，其定位结果可能被连续性守卫误拒。
        """
        self.motion_budget_cm += MOTION_BUDGET_CM.get(name, 6.0) * max(1, times)
        self.run_action(name, times)

    def _expected_position(self):
        """位置期望（上次成功定位），无则 None——仅用于歧义选支，不否决"""
        return self._last_located_pos

    def _accept_position(self, pos_xy):
        """记录定位基准；位置跳变时仅打日志（不拒绝——见 2026-08-30 复盘）

        曾实现为"跳变超预算即拒绝"，但真机复盘显示坏读数可能出现在跳变的
        另一侧（先到的读数才是异常），拒绝正确解会让机器人在错误位置上
        继续行动，比跳变本身更危险。改为仅记录，供事后诊断。
        """
        if self._last_located_pos is not None:
            jump = float(np.linalg.norm(np.asarray(pos_xy) - self._last_located_pos))
            if jump > self.motion_budget_cm + CONTINUITY_MARGIN_CM:
                print(f"[连续性观测] 位置跳变 {jump:.1f}cm（预算 "
                      f"{self.motion_budget_cm:.1f}+{CONTINUITY_MARGIN_CM:.0f}cm）——记录不拦截")
        self._last_located_pos = np.array(pos_xy, dtype=np.float64)
        self.motion_budget_cm = 0.0
        return True

    # -----------------------------------------------------------------
    # 头部舵机控制
    # -----------------------------------------------------------------

    def set_head(self, pulse, move_time_ms=HEAD_MOVE_TIME_MS):
        """转动头部舵机并等待到位；目标与当前位置相同则跳过。

        需要旋转时，按脉宽差（角度差）动态缩放等待时间。
        900μs≈90°为满量程，最小 HEAD_MOVE_TIME_MIN_MS。
        """
        if pulse == self.current_head_pulse:
            return  # 已在目标位置，无需等待
        delta = abs(pulse - self.current_head_pulse)
        dynamic_time = max(HEAD_MOVE_TIME_MIN_MS, int(move_time_ms * delta / 900))
        ctl.set_pwm_servo_pulse(2, pulse, dynamic_time)
        time.sleep(dynamic_time / 1000.0 + 0.2)
        self.current_head_pulse = pulse

    def raise_head(self):
        """转动头部至 PITCH_UP_PULSE（固定抬头；外参标定须与此俯仰一致）"""
        ctl.set_pwm_servo_pulse(1, PITCH_UP_PULSE, 500)

    def set_pitch(self, pulse, move_time_ms=500):
        """俯仰舵机（ID1）通用控制：1500=水平，可调约 950~2000，越小越低头

        与目标脉宽相同则跳过；转动后按转动时长等待舵机到位。
        （数字宫格等需要多俯仰档切换的关卡使用；AprilTag 关卡仍用 raise_head。）
        """
        if ctl is None:
            return
        if pulse == self.current_pitch_pulse:
            return
        ctl.set_pwm_servo_pulse(1, pulse, move_time_ms)
        time.sleep(move_time_ms / 1000.0 + 0.3)
        self.current_pitch_pulse = pulse

    def pulse_to_angle(self, pulse):
        """舵机脉宽→角度（度），右转为负，左转为正"""
        return (pulse - HEAD_CENTER) * SERVO_DEG_PER_US

    def compensate_head_offset(self, head_pulse):
        """补偿头部偏转角：将相机朝向转换为机体朝向

        solvePnP 解出的是相机朝向，转头时相机随头部转动但机体不动。
        机体朝向 = R(-θ_head) @ 相机朝向
        """
        if self.current_orientation is None:
            return
        theta = np.radians(self.pulse_to_angle(head_pulse))
        # R(-θ) = [[cosθ, sinθ], [-sinθ, cosθ]]
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R_neg = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
        self.current_orientation = R_neg @ self.current_orientation
        norm = np.linalg.norm(self.current_orientation)
        if norm != 0:
            self.current_orientation /= norm

    # -----------------------------------------------------------------
    # 拍照与 AprilTag 检测
    # -----------------------------------------------------------------

    def capture_image(self):
        """拍照，返回文件路径；失败返回 None"""
        timestamp = int(time.time())
        filename = f"/home/pi/Pictures/photo_{timestamp}.jpg"
        cmd = (f"fswebcam -r {CAMERA_WIDTH}x{CAMERA_HEIGHT} "
               f"--no-banner -S 3 {filename}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"拍照失败: {result.stderr}")
            return None

        print(f"照片已保存: {filename}")
        return filename

    def capture_frame(self):
        """拍照并读取为内存帧（BGR ndarray），供视觉识别使用；失败返回 None

        复用 capture_image() 的拍照链路，多一步 cv2.imread 得到图像数组。
        注：本方法与 capture_image 走同一 fswebcam 命令，视觉识别每次调用
        同样会保存一张照片到 Pictures 目录（与现有定位流程一致）。
        """
        filename = self.capture_image()
        if filename is None:
            return None
        frame = cv2.imread(filename)
        if frame is None:
            print("读图失败：", filename)
        return frame

    def detect_apriltag(self, filename):
        """检测图片中的 AprilTag，返回检测结果列表"""
        print("[INFO] loading image...")
        image = cv2.imread(filename)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        print("[INFO] detecting AprilTags...")
        options = apriltag.DetectorOptions(families="tag36h11")
        detector = apriltag.Detector(options)
        results = detector.detect(gray)
        print("[INFO] {} total AprilTags detected".format(len(results)))
        return results

    # -----------------------------------------------------------------
    # 定位（AprilTag + PnP）
    # -----------------------------------------------------------------

    def solve_pnp(self):
        """拍照并执行 AprilTag 检测 + PnP 求解，成功返回 True

        成功时设置 self.current_position 和 self.current_orientation（相机坐标系）。
        P0 修复：
          - 角点按 TAG_CORNER_PERM 显式重排后与 tag_poses 配对（真机实测为恒等映射）；
          - 位姿须通过物理合理性门控（pnp_pose_problems），假位姿一律返回 False，
            由上层定位重试或安全终止，绝不输出错误位姿；
          - 未知 tag id 跳过，不再 KeyError 崩溃。
        """
        objlist = []
        imglist = []
        filename = self.capture_image()
        if filename is None:
            return False
        for r in self.detect_apriltag(filename):
            tid = str(r.tag_id)
            if tid not in self.tag_poses:
                print(f"[WARN] 检测到未知 AprilTag ID {tid}（tag_poses 无此标签），跳过。")
                continue
            print("[INFO] Detected AprilTag ID: {}".format(tid))
            objlist.extend(self.tag_poses[tid])
            imglist.extend(np.asarray(r.corners, dtype=np.float64)[TAG_CORNER_PERM])

        if len(objlist) < 4 or len(objlist) % 4 != 0:
            print("检测到的AprilTag角点不足一组(4点)，无法计算相机位姿。")
            return False

        result = solve_pnp_pose(objlist, imglist)
        if result is None:
            print("PnP求解失败，无法计算相机位姿。")
            return False
        pos_3d, ori_3d, reproj_err = result

        problems = pnp_pose_problems(pos_3d, ori_3d, reproj_err,
                                     n_tags=len(imglist) // 4)
        if problems:
            print(f"PnP结果未通过合理性门控（{'；'.join(problems)}），本次定位失败。")
            print("提示：若频繁出现，请检查标签粘贴/角点顺序（TAG_CORNER_PERM）或相机内参。")
            return False

        self.current_position = pos_3d[:2]
        self.current_orientation = ori_3d[:2]
        norm = np.linalg.norm(self.current_orientation)
        if norm != 0:
            self.current_orientation /= norm
        return True

    def _locate_pose_once(self, head_pulse):
        """单档定位：拍照→检测→单帧 PnP→门控（含逐标签剔除救援）；成功则写入定位状态。

        返回 (ok, frame)：frame=(theta_nom, objlist, imglist) 供联合求解复用
        （检测到已知标签但单帧未通过门控时仍有值；无标签时为 None）。
        theta_nom 为舵机标称转角（度，未乘头转系数 k——k 由
        multiview_pose.camera_pose 内部施加，预乘会双重放大）。
        救援逻辑（2026-08-30 提速）：全集门控失败且标签 ≥3 时，逐一剔除一个
        标签重解；子集须 ≥2 标签且跨 ≥2 个平面（非共面才无平面歧义）。
        依据真机 trace：含地面标签的混合帧（157+160+161 等）全集 14~26px
        失败，剔除后子集 ~4px 通过，省一次换档重试（2~4s/次）。
        模拟器通过重写本方法接入（与旧 solve_pnp 桩同一接缝层次）。
        """
        theta = (head_pulse - HEAD_CENTER) * SERVO_DEG_PER_US
        filename = self.capture_image()
        if filename is None:
            return False, None
        tag_obs = {}  # {tid: (obj4, img4)}
        for r in self.detect_apriltag(filename):
            tid = str(r.tag_id)
            if tid not in self.tag_poses:
                print(f"[WARN] 检测到未知 AprilTag ID {tid}（tag_poses 无此标签），跳过。")
                continue
            print("[INFO] Detected AprilTag ID: {}".format(tid))
            tag_obs[tid] = (self.tag_poses[tid],
                            np.asarray(r.corners, dtype=np.float64)[TAG_CORNER_PERM])

        objlist = [p for pair in tag_obs.values() for p in pair[0]]
        imglist = [p for pair in tag_obs.values() for p in pair[1]]
        frame = (theta, objlist, imglist) if objlist else None
        if not tag_obs:
            print("检测到的AprilTag角点不足一组(4点)，无法计算相机位姿。")
            return False, frame

        def _try(tags_subset):
            """对给定标签子集做 PnP+门控，通过返回 (pos, ori)，否则 None"""
            obj = [p for t in tags_subset for p in tag_obs[t][0]]
            img = [p for t in tags_subset for p in tag_obs[t][1]]
            result = solve_pnp_pose(obj, img)
            if result is None:
                return None
            pos_3d, ori_3d, reproj_err = result
            if pnp_pose_problems(pos_3d, ori_3d, reproj_err, n_tags=len(tags_subset)):
                return None
            return pos_3d, ori_3d, reproj_err

        planes = {t: tag_plane_axis(self.tag_poses[t]) for t in tag_obs}
        solution = None

        if len(planes) == 1:
            # 共面帧（含单标签）——C3 三层歧义治理（2026-08-30）。
            # 普查依据：共面帧 75% 门控后仅一支活（零成本）；双活时 ITERATIVE
            # 落错支 1/4，须显式裁决。
            from core.multiview_pose import frame_pose_candidates, rot_from_rvec
            live = []
            for rvec, tvec, err in frame_pose_candidates(objlist, imglist):
                R = rot_from_rvec(rvec)
                pos_3d = (-R.T @ tvec).flatten()
                ori_3d = (R.T @ np.array([0.0, 0.0, 1.0])).flatten()
                if not pnp_pose_problems(pos_3d, ori_3d, err, n_tags=len(tag_obs)):
                    live.append((pos_3d, ori_3d, err))
            if len(live) == 1:
                solution = live[0]                      # 层1：门控筛支
            elif len(live) >= 2:
                sep = float(np.hypot(*(live[0][0][:2] - live[1][0][:2])))
                if sep < AMBIGUITY_SEP_MIN_CM:
                    solution = min(live, key=lambda s: s[2])   # 两支重合，无歧义
                else:
                    # 层2：期望选支（只在"干脆可分"时选，否则交给层3联合裁决）
                    exp = self._expected_position()
                    if exp is not None:
                        radius = self.motion_budget_cm + EXPECT_RADIUS_MARGIN_CM
                        d = [float(np.hypot(*(s[0][:2] - exp))) for s in live[:2]]
                        decisive = abs(d[0] - d[1]) > EXPECT_SELECT_MARGIN_CM
                        inside = [i for i in range(2) if d[i] <= radius]
                        if decisive and len(inside) == 1:
                            picked = inside[0]
                            print(f"[歧义消解] 双支间隔{sep:.1f}cm：选{['A','B'][picked]}支"
                                  f"（距期望 {d[picked]:.1f}cm / 另支 {d[1-picked]:.1f}cm）")
                            solution = live[picked]
                    if solution is None:
                        why = "期望不可分" if exp is not None else "无期望"
                        print(f"[歧义待裁决] 共面帧双支均过门控（间隔{sep:.1f}cm，{why}），"
                              "帧入联合缓存")
                        return False, frame            # 层3：交给多帧联合求解
        else:
            # 多平面帧：几何本身无歧义，ITERATIVE 单解 + 帧救援（剔除高噪标签）
            solution = _try(list(tag_obs))
            if solution is None and len(tag_obs) >= 3:
                rescued = None
                for drop in tag_obs:
                    subset = [t for t in tag_obs if t != drop]
                    if len(subset) < 2:
                        continue
                    if len({planes[t] for t in subset}) < 2:
                        continue  # 共面子集有平面歧义，不接受
                    sol = _try(subset)
                    # rescued 是 (解, 被剔标签) 二元组，比较解的 reproj 用 rescued[0][2]
                    if sol is not None and (rescued is None or sol[2] < rescued[0][2]):
                        rescued = (sol, drop)
                if rescued is not None:
                    (pos_3d, ori_3d, reproj_err), dropped = rescued
                    print(f"[帧救援] 剔除标签 {dropped} 后通过门控（reproj={reproj_err:.2f}px）")
                    solution = (pos_3d, ori_3d, reproj_err)

        if solution is None:
            print("PnP结果未通过合理性门控（含歧义治理/剔除救援），本次单帧定位失败。")
            return False, frame

        pos_3d, ori_3d, reproj_err = solution
        if not self._accept_position(pos_3d[:2]):
            return False, frame
        self.current_position = pos_3d[:2]
        self.current_orientation = ori_3d[:2]
        norm = np.linalg.norm(self.current_orientation)
        if norm != 0:
            self.current_orientation /= norm
        return True, frame

    def _locate_joint(self, frames):
        """多帧联合定位兜底：固定外参解机体位姿 (x_B, y_B, phi)，门控后写入状态。

        frames: [(theta_deg, objlist, imglist), ...]（同一站位、机体不动时采集）
        门控失败时自动剔除单帧残差最大的一帧重试一次；仍未通过返回 False。
        语义与单帧路径一致：current_position=光心地面投影，orientation=机体朝向。
        """
        try:
            from core.multiview_pose import solve_joint_3dof, gate_solution, \
                camera_optical_center, body_orientation_xy
        except Exception as e:
            print(f"[联合定位] multiview_pose 导入失败: {e}")
            return False

        if not self._extrinsics_from_file and not self._extrinsics_warned:
            print(f"[联合定位] 警告: 未找到 {MULTIVIEW_EXTRINSICS_PATH}，"
                  f"使用缺省外参 {self.multiview_extrinsics}（e=0 即光心=机体中心假设）。"
                  "建议先跑 survey_field.py 生成标定外参。")
            self._extrinsics_warned = True

        # 联合解门控阈值：帧内标签越多、帧数越多，聚合的不一致越大；
        # 按最大帧标签数取自适应阈值再乘 1.5（多帧联合的统计涨落）。
        max_tags = max((len(f[2]) // 4 for f in frames), default=1)
        joint_gate = reproj_gate_px(max_tags) * 1.5

        def _apply(fr):
            p_est, rms, info = solve_joint_3dof(fr, self.multiview_extrinsics)
            if p_est is None:
                return None, None, None, None
            problems = gate_solution(p_est, fr, self.multiview_extrinsics,
                                     reproj_max=joint_gate)
            return p_est, rms, problems, fr

        # 留一法枚举：歧义/坏帧会把联合解带偏，使"残差最大帧"反而不是坏帧
        # （直接剔除会剔错）。候选 = 全集 + 逐一去掉每帧，取过门控且 rms 最小者。
        candidates = [frames] + [f for i in range(len(frames))
                                 if (f := [x for j, x in enumerate(frames) if j != i])]
        best = None
        for cand in candidates:
            p_c, rms_c, problems_c, _ = _apply(cand)
            if not problems_c and (best is None or rms_c < best[1]):
                if len(cand) < len(frames):
                    dropped = [f[0] for f in frames if f not in cand]
                    print(f"[联合定位] 剔除帧 θ={dropped} 后通过门控…")
                best = (p_c, rms_c, cand)
        if best is None:
            print("[联合定位] 所有帧组合均未通过门控，联合求解失败。")
            return False
        p_est, rms, used = best

        joint_pos = camera_optical_center(p_est, 0.0)[:2]
        if not self._accept_position(joint_pos):
            return False
        self.current_position = joint_pos
        self.current_orientation = body_orientation_xy(p_est)
        print(f"[联合定位] 成功：{len(used)} 帧 rms={rms:.2f}px "
              f"光心=({self.current_position[0]:.1f},{self.current_position[1]:.1f}) "
              f"机体朝向=({self.current_orientation[0]:.2f},{self.current_orientation[1]:.2f})")
        return True

    def locate_with_scan(self):
        """三级头部扫描定位：回正→右转→左转，每档拍照+单帧PnP（快路径不变）

        三档单帧全部失败时，用缓存的多档角点做固定外参联合求解兜底
        （消除单标签平面歧义），仍失败才返回 False，头部回到中位。
        """
        self.current_position = None
        self.current_orientation = None
        frames = []

        # 1. 头部回正拍照（快路径：单帧成功即返回，行为与历史版本一致）
        self.set_head(HEAD_CENTER)
        ok, frame = self._locate_pose_once(HEAD_CENTER)
        if frame:
            frames.append(frame)
        if ok:
            print("定位成功（头部回正）。位置：", self.current_position,
                  "朝向：", self.current_orientation)
            return True

        # 2/3. 头部右转、左转拍照（单帧成功即返回；失败则角点入缓存）
        for pulse, name in ((HEAD_RIGHT, "右转"), (HEAD_LEFT, "左转")):
            self.set_head(pulse)
            ok, frame = self._locate_pose_once(pulse)
            if frame:
                frames.append(frame)
            if ok:
                self.compensate_head_offset(pulse)
                print(f"定位成功（头部{name}）。位置：", self.current_position,
                      "机体朝向：", self.current_orientation)
                self.set_head(HEAD_CENTER)
                return True

        # 4. 联合求解兜底：≥2 帧角点即可联解
        if len(frames) >= 2:
            if self._locate_joint(frames):
                self.set_head(HEAD_CENTER)
                return True

        # 5. 第二档宽扫（±63°）：覆盖三档扫不到的方位（如出口走廊朝东时
        #    东墙标签在 ~71-79°）。宽扫帧优先走单帧，失败并入联合缓存。
        for pulse, name in ((HEAD_WIDE_RIGHT, "宽右"), (HEAD_WIDE_LEFT, "宽左")):
            self.set_head(pulse)
            ok, frame = self._locate_pose_once(pulse)
            if frame:
                frames.append(frame)
            if ok:
                self.compensate_head_offset(pulse)
                print(f"定位成功（头部{name}）。位置：", self.current_position,
                      "机体朝向：", self.current_orientation)
                self.set_head(HEAD_CENTER)
                return True
        if len(frames) >= 2 and any(abs(f[0]) > 41.0 for f in frames):
            if self._locate_joint(frames):
                self.set_head(HEAD_CENTER)
                return True

        # 全部失败，头部回正
        self.set_head(HEAD_CENTER)
        print("三级头部扫描均失败。")
        return False

    def locate_with_retry(self):
        """转头定位失败时身体转动重试：头部扫描→同向连转改变视角→有限次后报错退出

        2026-08-30 修正：原「左右交替转身」净旋转≈0，5 次后朝向几乎不变，
        出口走廊类死区永远看不到新标签（角点级仿真实测卡死）。改为同向连转
        （5×22°=110°），能扫到侧后方位的标签；定位成功后由导航层转回。
        """
        for attempt in range(MAX_LOCATE_RETRIES):
            print(f"--- 定位尝试 {attempt + 1}/{MAX_LOCATE_RETRIES} ---")
            if self.locate_with_scan():
                return True
            # 头部扫描全失败 → 身体同向连转（累计旋转，扫到不同方位）
            # 2026-08-25 实机标定：小步转向不可靠，改用实测可靠的大步转向（见 levels 动作常量）
            print("头部扫描失败，身体左转尝试重新定位。")
            self.act("turn_left")
        print("定位重试超限，程序终止。")
        return False
