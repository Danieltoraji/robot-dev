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


# =====================================================================
# 头部舵机参数常量
# =====================================================================
HEAD_CENTER = 1500
HEAD_RIGHT = 1050
HEAD_LEFT = 1950
HEAD_MOVE_TIME_MS = 500  # 头部舵机转动等待时间，单位ms，对应旋动90°的时间。
HEAD_MOVE_TIME_MIN_MS = 100  # 小角度转头最小等待时间，单位ms
# 舵机脉宽→角度线性映射：angle_deg = (pulse - 1500) * SERVO_DEG_PER_US
# 标准500-2500μs对应±90°，即 90°/1000μs = 0.09
SERVO_DEG_PER_US = 0.09

# =====================================================================
# 定位参数常量
# =====================================================================
MAX_LOCATE_RETRIES = 5  # 定位失败最大重试次数
# 注意：导航决策常量（朝向阈值、位置阈值、避障阈值、停靠时间、安全余量等）
# 属于关卡层，由各关卡在 levels/<level>.py 中自行定义，不要放回 core，
# 以免出现「core 与关卡各持一份」的双重真相、造成跨关卡混乱。

# =====================================================================
# 相机内参（从原 solve_pnp 局部变量提取为模块级常量）
# =====================================================================
CAMERA_INTRINSIC = np.array(
    (
        [1.944903664123011e03, 0, 1.283069051100245e03],
        [0, 1.950095436307893e03, 9.831983420212778e02],
        [0, 0, 1],
    ),
    dtype=np.double,
)
CAMERA_DISTORTION = np.array([-0.384402275498781, 0.284681889150075, 0, 0])

# =====================================================================
# AprilTag 角点顺序（P0 修复 · 真机实测确认）
# =====================================================================
# tag_poses 世界坐标顺序约定（levels/goodluck.py）：左上→右上→右下→左下。
# 2026-08-24 真机实测（verify_corner_order.py，标签 36/37/38/39 共 6 组观测，
# 含正对与斜视角）确认：apriltag 库返回的 r.corners 顺序与 tag_poses 完全一致，
# 即恒等映射。此常量把该事实显式化，防止换库/换版本/重贴标签后再次踩坑；
# 若未来顺序变化，只需修改此常量并重跑 verify_corner_order.py 验证。
TAG_CORNER_PERM = np.array([0, 1, 2, 3], dtype=np.int64)

# =====================================================================
# PnP 位姿合理性门控（P0 修复）
# =====================================================================
# solvePnP 对错误角点顺序也能给出「重投影误差≈0」的假位姿（正方形标签的
# 旋转/镜像歧义）。定位结果必须通过以下物理合理性检查，否则视为定位失败
# （由 locate_with_scan/locate_with_retry 重试或安全终止），绝不输出假位姿。
# 阈值依据：2026-08-24 实测——正确解重投影误差 <1px、斜仰视角朝向 z 分量
# 可达 ±0.32；错误（镜像/旋转）解位置均跑出场地、高度超界或 z 分量 ≥0.64。
PNP_FIELD_MIN, PNP_FIELD_MAX = -10.0, 110.0   # 相机位置须在场地范围内（cm，含余量）
PNP_CAM_Z_MIN, PNP_CAM_Z_MAX = 5.0, 80.0      # 相机高度合理范围（cm）
PNP_ORI_Z_MAX = 0.4                           # 相机朝向 z 分量上限（近水平；实测斜视角可达 ±0.32）
PNP_ORI_XY_MIN = 0.1                          # 相机朝向 XY 分量下限（防垂直朝下/朝上）
PNP_REPROJ_ERR_MAX_PX = 20.0                  # 平均重投影误差上限（px，正确解实测 <1px）


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


def pnp_pose_problems(pos_3d, ori_3d, reproj_err):
    """PnP 位姿合理性检查，返回问题列表（空列表 = 通过）

    拦截角点顺序错误/镜像等「重投影误差小但物理不合理」的假位姿。
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
        problems.append("相机朝向接近垂直")
    if reproj_err > PNP_REPROJ_ERR_MAX_PX:
        problems.append(f"重投影误差{reproj_err:.1f}px过大")
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
        self.tag_poses = tag_poses if tag_poses is not None else {}

    # -----------------------------------------------------------------
    # 动作执行
    # -----------------------------------------------------------------

    def run_action(self, name, times=1):
        """执行动作组"""
        AGC.runActionGroup(name, times=times)

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
        cmd = f"fswebcam -r 2592x1944 --no-banner -S 3 {filename}"
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

        problems = pnp_pose_problems(pos_3d, ori_3d, reproj_err)
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

    def locate_with_scan(self):
        """三级头部扫描定位：回正→右转→左转，每步拍照+PnP

        成功时 current_position/current_orientation 已补偿为机体坐标系，返回 True。
        失败返回 False，头部回到中位。
        """
        self.current_position = None
        self.current_orientation = None

        # 1. 头部回正拍照
        self.set_head(HEAD_CENTER)
        if self.solve_pnp():
            print(
                "定位成功（头部回正）。位置：",
                self.current_position,
                "朝向：",
                self.current_orientation,
            )
            return True

        # 2. 头部右转拍照
        self.set_head(HEAD_RIGHT)
        if self.solve_pnp():
            self.compensate_head_offset(HEAD_RIGHT)
            print(
                "定位成功（头部右转）。位置：",
                self.current_position,
                "机体朝向：",
                self.current_orientation,
            )
            self.set_head(HEAD_CENTER)
            return True

        # 3. 头部左转拍照
        self.set_head(HEAD_LEFT)
        if self.solve_pnp():
            self.compensate_head_offset(HEAD_LEFT)
            print(
                "定位成功（头部左转）。位置：",
                self.current_position,
                "机体朝向：",
                self.current_orientation,
            )
            self.set_head(HEAD_CENTER)
            return True

        # 全部失败，头部回正
        self.set_head(HEAD_CENTER)
        print("三级头部扫描均失败。")
        return False

    def locate_with_retry(self):
        """转头定位失败时身体转动重试：头部扫描→交替左右小步转动→有限次后报错退出"""
        for attempt in range(MAX_LOCATE_RETRIES):
            print(f"--- 定位尝试 {attempt + 1}/{MAX_LOCATE_RETRIES} ---")
            if self.locate_with_scan():
                return True
            # 头部扫描全失败 → 身体小幅转动改变视角
            if attempt % 2 == 0:
                print("头部扫描失败，身体左转小步尝试重新定位。")
                self.run_action("turn_left_small_step")
            else:
                print("头部扫描失败，身体右转小步尝试重新定位。")
                self.run_action("turn_right_small_step")
        print("定位重试超限，程序终止。")
        return False
