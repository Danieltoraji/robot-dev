import cv2
import numpy as np
from enum import Enum

from capture import capture_image
from candidate_classifier import classify_candidates


# =====================================================
# 摄像头参数
# =====================================================

CAMERA_FOV = 60.0


# =====================================================
# proximity 阈值
# =====================================================

NEAR_THRESHOLD = 250
MID_THRESHOLD = 120


# =====================================================
# 颜色枚举
# =====================================================

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


# =====================================================
# 普通 HSV 颜色范围
# =====================================================

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


# =====================================================
# 绿色判定
#
# 普通 HSV:
# H: 0~360°
#
# OpenCV:
# H: 0~179
#
# 深绿色：
# H = 109°~153°
# S/V = 1.35~2.48
#
# 水绿色：
# 大多数 S/V < 1
#
# 因此采用：
#
# H ∈ [100°, 160°]
# S/V >= 1.15
#
# =====================================================

GREEN_H_MIN_DEG = 100.0
GREEN_H_MAX_DEG = 160.0

GREEN_SV_RATIO_MIN = 1.15


# =====================================================
# 颜色 Mask
# =====================================================

def build_color_mask(hsv, color):

    # -------------------------------------------------
    # 绿色特殊处理
    # -------------------------------------------------

    if color == Color.GREEN:

        H = hsv[:, :, 0].astype(
            np.float32
        )

        S = hsv[:, :, 1].astype(
            np.float32
        )

        V = hsv[:, :, 2].astype(
            np.float32
        )

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

    # -------------------------------------------------
    # 其他颜色
    # -------------------------------------------------

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


# =====================================================
# 颜色识别
# =====================================================

def identify_color(color, image):

    """
    检测目标颜色在整幅画面中的面积比例。

    特别说明：
        GREEN 不使用简单 HSV 长方体，
        而使用：

            H + S/V

        关系进行判断。
    """

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

    # -------------------------------------------------
    # 去除孤立噪声
    # -------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # -------------------------------------------------
    # 面积比例
    # -------------------------------------------------

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


# =====================================================
# 计算 yaw
# =====================================================

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


# =====================================================
# proximity
# =====================================================

def proximity_level(box_width):

    if box_width >= NEAR_THRESHOLD:

        return "NEAR"

    elif box_width >= MID_THRESHOLD:

        return "MID"

    else:

        return "FAR"


# =====================================================
# 主识别函数
# =====================================================

def identify(
    id,
    image=None,
    topk=5
):

    """
    使用 candidate_classifier
    寻找指定数字牌。

    返回：

        success
        yaw
        proximity
    """

    # -------------------------------------------------
    # 校验 ID
    # -------------------------------------------------

    if (
        not isinstance(id, int)
        or not 1 <= id <= 7
    ):

        return (
            False,
            None,
            None
        )

    # -------------------------------------------------
    # 获取图像
    # -------------------------------------------------

    if image is None:

        image = capture_image()

    if image is None:

        return (
            False,
            None,
            None
        )

    H, W = image.shape[:2]

    # -------------------------------------------------
    # 候选识别
    # -------------------------------------------------

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

    # -------------------------------------------------
    # 找目标数字
    # -------------------------------------------------

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

    # -------------------------------------------------
    # 选择最高评分候选
    # -------------------------------------------------

    target = max(
        target_candidates,
        key=lambda x:
            x["final_score"]
    )

    # -------------------------------------------------
    # yaw
    # -------------------------------------------------

    yaw = calculate_yaw(
        target["box"],
        W
    )

    # -------------------------------------------------
    # proximity
    # -------------------------------------------------

    x, y, w, h = target["box"]

    proximity = int(w)

    return (
        True,
        yaw,
        proximity
    )


# =====================================================
# 单独测试
# =====================================================

if __name__ == "__main__":

    success, yaw, proximity = identify(5)

    print(
        "success:",
        success
    )

    print(
        "yaw:",
        yaw
    )

    print(
        "proximity:",
        proximity
    )

    if success:

        print(
            "level:",
            proximity_level(
                proximity
            )
        )