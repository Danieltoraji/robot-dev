import cv2
import numpy as np


# =====================================================
# HSV 颜色范围
#
# OpenCV:
# H: 0~179
# S: 0~255
# V: 0~255
#
# 注意：
# green / blue 不再使用简单的 HSV 长方体，
# 而使用 H + S/V 的关系进行判断。
# =====================================================

COLOR_RANGES = {

    "red":
    [
        (
            np.array([0, 120, 100]),
            np.array([6, 255, 255])
        ),

        (
            np.array([170, 120, 80]),
            np.array([179, 255, 255])
        )
    ],

    "orange":
    [
        (
            np.array([8, 120, 120]),
            np.array([16, 255, 255])
        )
    ],

    "yellow":
    [
        (
            np.array([22, 100, 120]),
            np.array([32, 255, 255])
        )
    ],

    # blue 在 build_color_mask() 中特殊处理
    # 因此这里不再设置普通 HSV 范围。

    "purple":
    [
        (
            np.array([108, 80, 50]),
            np.array([140, 255, 255])
        )
    ],

    "pink":
    [
        (
            np.array([160, 35, 80]),    # H 下限，S 下限，V 下限
            np.array([170, 255, 255])   # H 上限，S 上限，V 上限
        )
    ]
}


COLOR_TO_ID = {

    "red": 1,
    "orange": 2,
    "yellow": 3,
    "green": 4,
    "blue": 5,
    "purple": 6,
    "pink": 7
}


# =====================================================
# 绿色判定参数
# =====================================================

# 你的真实数据：
#
# 深绿色：
# H = 109° ~ 153°
# S/V ≈ 1.35 ~ 2.48
#
# 水绿色：
# H = 121° ~ 169°
# S/V ≈ 0.27 ~ 0.98
#
# 因此这里不再分别限制 S 和 V，
# 而是利用 S/V 的关系。
#
# H 使用“普通 HSV 的 0~360°”表示，
# 代码内部会自动转换为 OpenCV H。


GREEN_H_MIN_DEG = 100.0
GREEN_H_MAX_DEG = 160.0

GREEN_SV_RATIO_MIN = 1.15


# =====================================================
# 蓝色判定参数
# =====================================================

# 真实标准蓝色数据：
#
# 215°, 92%, 44%
# 213°, 92%, 43%
# 219°, 72%, 52%
# 215°, 58%, 53%
# 214°, 96%, 61%
# 211°, 94%, 64%
# 218°, 75%, 50%
# 208°, 84%, 60%
# 211°, 81%, 63%
# 211°, 83%, 60%
#
# 因此目标蓝色主要集中在：
#
# H = 208° ~ 219°
# S = 58% ~ 96%
# V = 43% ~ 64%
#
# 杂色则主要表现为：
#
# 174° ~ 205°：
#     H 明显偏低
#
# 208° ~ 209°：
#     H 接近蓝色，但 S 明显偏低
#
# 因此不能简单使用：
#
#     H_MIN <= H <= H_MAX
#     S_MIN <= S <= S_MAX
#
# 而采用 H + S 的联合判定。
#
# H 使用普通 HSV 的 0~360°。
# =====================================================

BLUE_H_MIN_DEG = 207.0
BLUE_H_MAX_DEG = 220.0

BLUE_V_MIN_PERCENT = 35.0
BLUE_V_MAX_PERCENT = 75.0


def blue_s_min(h_deg):
    """
    根据 Hue 动态计算蓝色所需的最低 Saturation。

    在 H < 215° 时：
        H 越靠近 208°，
        所需 S 越高。

    在 H >= 215° 时：
        S 最低要求为 55%。

    该关系根据目前采集到的
    标准蓝色 / 杂色数据确定。
    """

    if h_deg < 215.0:

        return (
            65.0
            -
            1.43 * (h_deg - 208.0)
        )

    return 55.0


# =====================================================
# 参数
# =====================================================

MIN_AREA = 15000
MAX_AREA = 500000


# =====================================================
# 建立颜色 Mask
# =====================================================

def build_color_mask(hsv, color_name):
    """
    根据颜色名称建立二值 mask。

    普通颜色：
        使用传统 HSV 范围。

    green：
        使用 H + S/V 的关系进行判断。

    blue：
        使用 H + 动态 S + V 的关系进行判断。

    参数
    ----
    hsv : OpenCV HSV image
    color_name : str

    返回
    ----
    mask : uint8
        0 / 255
    """

    # -------------------------------------------------
    # 特殊处理绿色
    # -------------------------------------------------

    if color_name == "green":

        H = hsv[:, :, 0].astype(np.float32)
        S = hsv[:, :, 1].astype(np.float32)
        V = hsv[:, :, 2].astype(np.float32)

        # OpenCV H: 0~179
        # 普通 HSV H: 0~360°
        H_deg = H * 2.0

        # 防止 V=0
        sv_ratio = S / np.maximum(V, 1.0)

        mask = (
            (H_deg >= GREEN_H_MIN_DEG)
            &
            (H_deg <= GREEN_H_MAX_DEG)
            &
            (sv_ratio >= GREEN_SV_RATIO_MIN)
        )

        return (
            mask.astype(np.uint8) * 255
        )

    # -------------------------------------------------
    # 特殊处理蓝色
    # -------------------------------------------------

    if color_name == "blue":

        H = hsv[:, :, 0].astype(np.float32)
        S = hsv[:, :, 1].astype(np.float32)
        V = hsv[:, :, 2].astype(np.float32)

        # -------------------------------------------------
        # OpenCV HSV → 普通 HSV
        # -------------------------------------------------

        H_deg = H * 2.0

        S_percent = (
            S / 255.0 * 100.0
        )

        V_percent = (
            V / 255.0 * 100.0
        )

        # -------------------------------------------------
        # H 范围
        # -------------------------------------------------

        h_valid = (
            (H_deg >= BLUE_H_MIN_DEG)
            &
            (H_deg <= BLUE_H_MAX_DEG)
        )

        # -------------------------------------------------
        # V 范围
        # -------------------------------------------------

        v_valid = (
            (V_percent >= BLUE_V_MIN_PERCENT)
            &
            (V_percent <= BLUE_V_MAX_PERCENT)
        )

        # -------------------------------------------------
        # 动态 S 阈值
        # -------------------------------------------------

        # 对每一个像素，根据它自己的 H
        # 计算对应的最低 S。
        #
        # H < 215°：
        #
        #     S_min = 65 - 1.43 * (H - 208)
        #
        # H >= 215°：
        #
        #     S_min = 55
        #
        # 这里直接使用 numpy 向量化计算，
        # 不需要逐像素 Python 循环。

        s_min = np.where(
            H_deg < 215.0,
            65.0 - 1.43 * (H_deg - 208.0),
            55.0
        )

        s_valid = (
            S_percent >= s_min
        )

        # -------------------------------------------------
        # 最终蓝色 Mask
        # -------------------------------------------------

        mask = (
            h_valid
            &
            s_valid
            &
            v_valid
        )

        return (
            mask.astype(np.uint8) * 255
        )

    # -------------------------------------------------
    # 其他颜色
    # -------------------------------------------------

    if color_name not in COLOR_RANGES:

        return np.zeros(
            hsv.shape[:2],
            dtype=np.uint8
        )

    mask = np.zeros(
        hsv.shape[:2],
        dtype=np.uint8
    )

    for lower, upper in COLOR_RANGES[color_name]:

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
# 提取数字
# =====================================================

def extract_digit_mask(roi):
    """
    从颜色牌 ROI 中提取黑色数字。

    思路：

        1. 找到 ROI 中的主颜色
        2. 每个像素计算与主颜色的 HSV 距离
        3. Otsu 自动阈值
        4. 形态学去噪
        5. 保留较大的连通区域

    返回：
        digit_mask
            白色 = 数字
            黑色 = 背景
    """

    hsv = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2HSV
    )

    # -------------------------------------------------
    # 去掉黑色像素
    # -------------------------------------------------

    valid = (
        (hsv[:, :, 1] > 40)
        &
        (hsv[:, :, 2] > 40)
    )

    pixels = hsv[valid]

    if len(pixels) < 100:

        return np.zeros(
            roi.shape[:2],
            np.uint8
        )

    # -------------------------------------------------
    # 主颜色
    # -------------------------------------------------

    h0 = np.median(
        pixels[:, 0]
    )

    s0 = np.median(
        pixels[:, 1]
    )

    v0 = np.median(
        pixels[:, 2]
    )

    # -------------------------------------------------
    # HSV 距离
    # -------------------------------------------------

    H = hsv[:, :, 0].astype(np.float32)
    S = hsv[:, :, 1].astype(np.float32)
    V = hsv[:, :, 2].astype(np.float32)

    dh = np.abs(H - h0)

    # Hue 环绕
    dh = np.minimum(
        dh,
        180 - dh
    )

    ds = np.abs(S - s0)
    dv = np.abs(V - v0)

    # H 权重小一些
    distance = (
        0.5 * dh
        +
        1.0 * ds
        +
        1.2 * dv
    )

    distance = cv2.normalize(
        distance,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    # -------------------------------------------------
    # Otsu
    # -------------------------------------------------

    _, digit_mask = cv2.threshold(
        distance,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # -------------------------------------------------
    # 去噪
    # -------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    digit_mask = cv2.morphologyEx(
        digit_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    digit_mask = cv2.morphologyEx(
        digit_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # -------------------------------------------------
    # 保留较大连通域
    # -------------------------------------------------

    contours, _ = cv2.findContours(
        digit_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    final = np.zeros_like(
        digit_mask
    )

    roi_area = roi.shape[0] * roi.shape[1]

    for c in contours:

        area = cv2.contourArea(c)

        if area < roi_area * 0.001:
            continue

        cv2.drawContours(
            final,
            [c],
            -1,
            255,
            -1
        )

    return final


# =====================================================
# 洞分析
# =====================================================

def analyze_holes(mask, contour):

    x, y, w, h = cv2.boundingRect(
        contour
    )

    roi_mask = mask[
        y:y+h,
        x:x+w
    ]

    inv = cv2.bitwise_not(
        roi_mask
    )

    contours, _ = cv2.findContours(
        inv,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    hole_area = 0
    hole_count = 0

    for c in contours:

        area = cv2.contourArea(c)

        if area > 30:

            hole_area += area
            hole_count += 1

    return hole_area, hole_count


# =====================================================
# 主函数
# =====================================================

def extract_digit_roi(image):

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    results = []

    # -------------------------------------------------
    # 遍历颜色
    # -------------------------------------------------

    for color_name in COLOR_TO_ID.keys():

        mask = build_color_mask(
            hsv,
            color_name
        )

        # -------------------------------------------------
        # 连接破碎区域
        # -------------------------------------------------

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3)
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        # -------------------------------------------------
        # 找轮廓
        # -------------------------------------------------

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # -------------------------------------------------
        # 遍历候选区域
        # -------------------------------------------------

        for cnt in contours:

            area = cv2.contourArea(cnt)

            if area < MIN_AREA:
                continue

            if area > MAX_AREA:
                continue

            x, y, w, h = cv2.boundingRect(
                cnt
            )

            ratio = max(
                w / h,
                h / w
            )

            if ratio > 3:
                continue

            # -------------------------------------------------
            # Solidity
            # -------------------------------------------------

            hull = cv2.convexHull(cnt)

            hull_area = cv2.contourArea(
                hull
            )

            if hull_area == 0:
                continue

            solidity = (
                area / hull_area
            )

            # -------------------------------------------------
            # ROI
            # -------------------------------------------------

            roi = image[
                y:y+h,
                x:x+w
            ]

            digit_mask = extract_digit_mask(
                roi
            )

            gray = cv2.cvtColor(
                roi,
                cv2.COLOR_BGR2GRAY
            )

            contrast = np.std(
                gray
            )

            # -------------------------------------------------
            # 洞
            # -------------------------------------------------

            hole_area, hole_count = analyze_holes(
                mask,
                cnt
            )

            hole_ratio = (
                hole_area /
                (area + 1)
            )

            # -------------------------------------------------
            # 评分
            # -------------------------------------------------

            area_score = min(
                area / 80000,
                1
            )

            if 0.05 < hole_ratio < 0.8:

                hole_score = 1

            elif hole_ratio <= 1.2:

                hole_score = 0.5

            else:

                hole_score = 0

            if 1 <= hole_count <= 10:

                count_score = 1

            elif hole_count <= 20:

                count_score = 0.5

            else:

                count_score = 0

            contrast_score = min(
                contrast / 60,
                1
            )

            score = (
                0.30 * area_score
                +
                0.25 * solidity
                +
                0.25 * hole_score
                +
                0.10 * count_score
                +
                0.10 * contrast_score
            )

            results.append({

                "roi": roi,

                "digit_mask": digit_mask,

                "gray": gray,

                "box": (
                    x,
                    y,
                    w,
                    h
                ),

                "area": area,

                "ratio": ratio,

                "solidity": solidity,

                "contrast": contrast,

                "hole_ratio": hole_ratio,

                "hole_count": hole_count,

                "score": score,

                "color": color_name,

                "color_id":
                    COLOR_TO_ID[color_name],

                "mask":
                    mask[y:y+h, x:x+w]
            })

    # -------------------------------------------------
    # 按评分排序
    # -------------------------------------------------

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results