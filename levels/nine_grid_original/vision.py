# -*- coding: utf-8 -*-
"""参考原版九宫格的**视觉层**：拍照 + 颜色识别 + 目标方位/远近（搬运自参考代码）

来源：
  · `reference code/九宫格视觉导航/robot/capture.py`（拍照）
  · `reference code/九宫格视觉导航/robot/identify.py`（颜色 mask / identify / yaw / proximity）
用途：`levels/nine_grid_original/`（参考原版通关流程）专用。与本仓库的
      `vision/nine_grid_detector.py`（现行九宫格的检测器）**没有任何共用代码**，
      两边互不影响。

与参考版的差异（**只有两处，判定与阈值一字未改**）：
  1. 拍照改为"绑定 RobotState 就借它的拍照链路，否则用参考版自己的 fswebcam 命令"。
     绑定的目的是复用真机的同一台相机与同一条拍照命令（`2592x1944 -S 3`），
     参考版命令与本仓库 `core.camera_config.CAMERA_WIDTH/HEIGHT` 本来就一致。
  2. 图像缩放抽成常量 `CAPTURE_RESIZE`（参考版硬编码 `cv2.resize(img, (1280,980))`）。
     ⚠️ 这个缩放**不是小事**：参考版所有像素阈值（`ARRIVE_THRESHOLD` 的框宽、
     `NEAR/MID_THRESHOLD`）都是在这张 1280×980 的图上标定的，换分辨率必须整套重标。

⚠️ 与本仓库现行九宫格的两处**约定相反**，接线时别混（详见 docs 的常量表）：
  · yaw 符号：参考版**左侧为正**（`calculate_yaw` 前面有负号）⇒ 本仓库 `_body_angle_deg`
    是**右侧为正**；
  · 颜色窗口：参考版自带一套 HSV 窗口 + 绿/蓝特殊判据，与 `vision/nine_grid_detector`
    的 `COLOR_THRESHOLDS` 是**两套独立标定**，互不通用。
"""

import os
import subprocess
import time
from enum import Enum

import cv2
import numpy as np

from .classifier import classify_candidates


# =====================================================================
# 摄像头参数
# =====================================================================

CAMERA_FOV = 60.0

# 拍照分辨率与缩放（参考版 capture.py：fswebcam 2592x1944 → resize 1280x980）
CAPTURE_RESOLUTION = "2592x1944"
CAPTURE_RESIZE = (1280, 980)
CAPTURE_SKIP_FRAMES = 3


# =====================================================================
# proximity 阈值（单位：目标框宽 px，在 CAPTURE_RESIZE 的图上量）
# =====================================================================

NEAR_THRESHOLD = 250
MID_THRESHOLD = 120


# 绑定的 RobotState（接入用；None 时退回参考版自带的 fswebcam 调用）
_STATE = None


def bind_state(state):
    """把 RobotState 借给拍照函数（一局开始时调一次即可）"""
    global _STATE
    _STATE = state


# =====================================================================
# 拍照
# =====================================================================

def capture_image():
    """拍照并返回 BGR 图像数组（失败返回 None）

    参考版行为：fswebcam 拍 2592x1944 存到 ~/Pictures，读回来再缩放到 1280x980。
    绑定过 RobotState 时走它的 `capture_frame()`（同一台相机、同一条命令，
    且由 core 统一决定分辨率），缩放与返回语义保持一致。
    """
    if _STATE is not None:
        img = _STATE.capture_frame()
        if img is None:
            return None
        return cv2.resize(img, CAPTURE_RESIZE, interpolation=cv2.INTER_AREA)

    pictures_dir = os.path.expanduser("~/Pictures")
    os.makedirs(pictures_dir, exist_ok=True)

    timestamp = int(time.time())
    filename = os.path.join(pictures_dir, f"photo_{timestamp}.jpg")

    cmd = (f"fswebcam -r {CAPTURE_RESOLUTION} --no-banner "
           f"-S {CAPTURE_SKIP_FRAMES} {filename}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"拍照失败: {result.stderr}")
        return None

    print(f"照片已保存: {filename}")

    img = cv2.imread(filename)
    if img is None:
        print("文件已保存，但 OpenCV 无法读取，可能文件损坏")
        return None

    return cv2.resize(img, CAPTURE_RESIZE, interpolation=cv2.INTER_AREA)


# =====================================================================
# 颜色枚举
# =====================================================================

class Color(Enum):

    RED = 1
    ORANGE = 2
    YELLOW = 3
    GREEN = 4
    BLUE = 5
    PURPLE = 6
    PINK = 7


def id_to_color(id):

    mapping = {

        1: Color.RED,
        2: Color.ORANGE,
        3: Color.YELLOW,
        4: Color.GREEN,
        5: Color.BLUE,
        6: Color.PURPLE,
        7: Color.PINK,

    }

    return mapping.get(id)


# =====================================================================
# 普通 HSV 颜色范围
# =====================================================================

COLOR_RANGES = {

    Color.RED:
    [
        (
            np.array([0, 100, 70]),
            np.array([10, 255, 255])
        ),

        (
            np.array([160, 100, 70]),
            np.array([179, 255, 255])
        )
    ],

    Color.ORANGE:
    [
        (
            np.array([10, 100, 70]),
            np.array([25, 255, 255])
        )
    ],

    Color.YELLOW:
    [
        (
            np.array([20, 100, 70]),
            np.array([35, 255, 255])
        )
    ],

    Color.BLUE:
    [
        (
            np.array([85, 100, 50]),
            np.array([125, 255, 255])
        )
    ],

    Color.PURPLE:
    [
        (
            np.array([125, 80, 50]),
            np.array([160, 255, 255])
        )
    ],

    Color.PINK:
    [
        (
            np.array([145, 20, 50]),
            np.array([179, 255, 255])
        )
    ]
}


# =====================================================================
# 绿色判定（H 用普通 HSV 的 0~360°；S/V ≥ 1.15）
# =====================================================================

GREEN_H_MIN_DEG = 100.0
GREEN_H_MAX_DEG = 160.0

GREEN_SV_RATIO_MIN = 1.15


# =====================================================================
# 颜色 Mask
# =====================================================================

def build_color_mask(hsv, color):

    # 绿色特殊处理：H ∈ [100°,160°] 且 S/V ≥ 1.15
    if color == Color.GREEN:

        H = hsv[:, :, 0].astype(np.float32)
        S = hsv[:, :, 1].astype(np.float32)
        V = hsv[:, :, 2].astype(np.float32)

        # OpenCV H -> 普通角度
        H_deg = H * 2.0

        # S/V
        sv_ratio = (
            S /
            np.maximum(V, 1.0)
        )

        mask = (
            (H_deg >= GREEN_H_MIN_DEG)
            &
            (H_deg <= GREEN_H_MAX_DEG)
            &
            (sv_ratio >= GREEN_SV_RATIO_MIN)
        )

        return (
            mask.astype(np.uint8)
            * 255
        )

    # 其他颜色：普通 HSV 长方体
    if color not in COLOR_RANGES:

        return np.zeros(
            hsv.shape[:2],
            dtype=np.uint8
        )

    mask = np.zeros(
        hsv.shape[:2],
        dtype=np.uint8
    )

    for lower, upper in COLOR_RANGES[color]:

        current = cv2.inRange(
            hsv,
            lower,
            upper
        )

        mask = cv2.bitwise_or(
            mask,
            current
        )

    return mask


# =====================================================================
# 颜色识别：目标颜色在**整幅画面**中的面积比例
# =====================================================================

def identify_color(color, image):

    if (
        color is None
        or not isinstance(color, Color)
    ):

        return 0.0

    if image is None:

        return 0.0

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    mask = build_color_mask(
        hsv,
        color
    )

    # 去除孤立噪声
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    area = cv2.countNonZero(
        mask
    )

    total = (
        image.shape[0]
        *
        image.shape[1]
    )

    if total == 0:

        return 0.0

    return area / total


# =====================================================================
# 计算 yaw（**左侧为正**，与参考版一致）
# =====================================================================

def calculate_yaw(
    box,
    image_width
):

    x, y, w, h = box

    center_x = (
        x + w / 2
    )

    dx = (
        center_x
        -
        image_width / 2
    )

    # 左侧为正
    yaw = (
        -(dx / image_width)
        *
        CAMERA_FOV
    )

    return yaw


# =====================================================================
# proximity：远/中/近（按目标框宽分档；主判据用的是框宽原始值）
# =====================================================================

def proximity_level(box_width):

    if box_width >= NEAR_THRESHOLD:

        return "NEAR"

    elif box_width >= MID_THRESHOLD:

        return "MID"

    else:

        return "FAR"


# =====================================================================
# 主识别函数
# =====================================================================

def identify(
    id,
    image=None,
    topk=5
):

    """用 candidate_classifier 找指定数字牌 → (success, yaw, proximity)

    proximity = 目标框宽（px，在 CAPTURE_RESIZE 的图上）。
    """

    if (
        not isinstance(id, int)
        or not 1 <= id <= 7
    ):

        return (
            False,
            None,
            None
        )

    if image is None:

        image = capture_image()

    if image is None:

        return (
            False,
            None,
            None
        )

    H, W = image.shape[:2]

    candidates = classify_candidates(
        image,
        topk=topk
    )

    if len(candidates) == 0:

        return (
            False,
            None,
            None
        )

    target_candidates = [

        c

        for c in candidates

        if c["final_digit"] == id

    ]

    if len(target_candidates) == 0:

        return (
            False,
            None,
            None
        )

    target = max(
        target_candidates,
        key=lambda x:
            x["final_score"]
    )

    yaw = calculate_yaw(
        target["box"],
        W
    )

    x, y, w, h = target["box"]

    proximity = int(w)

    return (
        True,
        yaw,
        proximity
    )
