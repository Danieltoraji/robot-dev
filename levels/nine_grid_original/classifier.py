# -*- coding: utf-8 -*-
"""参考原版九宫格的**候选区域 + 数字分类层**（搬运自参考代码）

来源：
  · `reference code/九宫格视觉导航/robot/extract_digit_roi.py`（HSV 分割 / ROI / 数字 Mask / 洞分析）
  · `reference code/九宫格视觉导航/robot/candidate_classifier.py`（HOG + SVM 数字识别与融合打分）
用途：`levels/nine_grid_original/`（参考原版通关流程）专用；上游是同包的 `vision.identify()`。

两个模块合并到一个文件，是因为它们本来就是一条链路（分类器 import ROI 提取器），
且都只服务这一条参考链路；`COLOR_RANGES` 在参考版里就是**两份不同的表**
（本文件的按颜色名索引，vision 里的按 Color 枚举索引），合并时必须保持两份独立。

与参考版的差异（**判定、权重、阈值一字未改**）：
  1. 模型路径：参考版是 `./models/digit_classifier_mask.pkl`（相对当前工作目录，
     且**在 import 时**就 `joblib.load`）。这里改成
     `core.paths.PROJECT_ROOT/models/nine_grid/digit_classifier_mask.pkl`，
     并**懒加载**——否则 `python main.py <其它关卡>` 会因为在 PC 上 import 本模块
     而直接崩掉（不满足"不影响其它关卡"）。
     模型文件与本仓库已有的 `models/nine_grid/digit_classifier_mask.pkl`
     经 SHA256 比对**逐字节相同**（4F60F827…A59BC），故不再复制一份 28MB 的副本。
  2. 依赖缺失（joblib/skimage 没装）时**降级为纯颜色**：`model_digit=None`、
     `model_conf=0.0`，`final_digit` 仍按颜色给（参考版会直接 import 崩溃）。
     机器人端依赖已具备（`/home/pi/jupyter-env` 里有 scikit-learn/skimage/joblib）。
"""

import os

import cv2
import numpy as np

from core.paths import PROJECT_ROOT


# =====================================================
# HSV 颜色范围（按颜色名索引；green / blue 走特殊判据）
#
# OpenCV: H 0~179、S 0~255、V 0~255
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

    # blue 在 build_color_mask() 中特殊处理，这里不再设普通 HSV 范围

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
# 绿色判定参数（深绿 H 109~153°、S/V 1.35~2.48；水绿 S/V < 1）
# =====================================================

GREEN_H_MIN_DEG = 100.0
GREEN_H_MAX_DEG = 160.0

GREEN_SV_RATIO_MIN = 1.15


# =====================================================
# 蓝色判定参数（标准蓝 H 208~219°、S 58~96%、V 43~64%；用 H+S 联合判定）
# =====================================================

BLUE_H_MIN_DEG = 207.0
BLUE_H_MAX_DEG = 220.0

BLUE_V_MIN_PERCENT = 35.0
BLUE_V_MAX_PERCENT = 75.0


def blue_s_min(h_deg):
    """按 Hue 动态给出蓝色所需的最低 Saturation（%）

    H < 215°：H 越靠近 208°，所需 S 越高（65 - 1.43*(H-208)）；
    H ≥ 215°：最低 55%。
    """
    if h_deg < 215.0:

        return (
            65.0
            -
            1.43 * (h_deg - 208.0)
        )

    return 55.0


# =====================================================
# 候选区域面积门（px²，在 CAPTURE_RESIZE 的图上量）
# =====================================================

MIN_AREA = 15000
MAX_AREA = 500000


# =====================================================
# 建立颜色 Mask
# =====================================================

def build_color_mask(hsv, color_name):
    """颜色名 → 二值 mask（green / blue 用 H+S 联合判据，其余用 HSV 长方体）"""

    # 绿色
    if color_name == "green":

        H = hsv[:, :, 0].astype(np.float32)
        S = hsv[:, :, 1].astype(np.float32)
        V = hsv[:, :, 2].astype(np.float32)

        # OpenCV H: 0~179 → 普通 HSV H: 0~360°
        H_deg = H * 2.0

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

    # 蓝色
    if color_name == "blue":

        H = hsv[:, :, 0].astype(np.float32)
        S = hsv[:, :, 1].astype(np.float32)
        V = hsv[:, :, 2].astype(np.float32)

        H_deg = H * 2.0

        S_percent = (
            S / 255.0 * 100.0
        )

        V_percent = (
            V / 255.0 * 100.0
        )

        h_valid = (
            (H_deg >= BLUE_H_MIN_DEG)
            &
            (H_deg <= BLUE_H_MAX_DEG)
        )

        v_valid = (
            (V_percent >= BLUE_V_MIN_PERCENT)
            &
            (V_percent <= BLUE_V_MAX_PERCENT)
        )

        # 逐像素按自己的 H 算最低 S（向量化，等价于 blue_s_min）
        s_min = np.where(
            H_deg < 215.0,
            65.0 - 1.43 * (H_deg - 208.0),
            55.0
        )

        s_valid = (
            S_percent >= s_min
        )

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

    # 其他颜色
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
# 从 ROI 里提取黑色数字（主颜色 HSV 距离 → Otsu → 形态学 → 保留大连通域）
# =====================================================

def extract_digit_mask(roi):
    """返回 digit_mask：白色 = 数字，黑色 = 背景"""

    hsv = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2HSV
    )

    # 去掉黑色像素
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

    # 主颜色
    h0 = np.median(
        pixels[:, 0]
    )

    s0 = np.median(
        pixels[:, 1]
    )

    v0 = np.median(
        pixels[:, 2]
    )

    # HSV 距离
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

    # Otsu
    _, digit_mask = cv2.threshold(
        distance,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 去噪
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

    # 保留较大连通域
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
# 洞分析（数字的闭合孔洞 = 数字牌的重要特征）
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
# 候选区域提取（按颜色分割 → 几何筛选 → 打分）
# =====================================================

def extract_digit_roi(image):

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    results = []

    for color_name in COLOR_TO_ID.keys():

        mask = build_color_mask(
            hsv,
            color_name
        )

        # 连接破碎区域
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3)
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

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

            # Solidity
            hull = cv2.convexHull(cnt)

            hull_area = cv2.contourArea(
                hull
            )

            if hull_area == 0:
                continue

            solidity = (
                area / hull_area
            )

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

            hole_area, hole_count = analyze_holes(
                mask,
                cnt
            )

            hole_ratio = (
                hole_area /
                (area + 1)
            )

            # 评分
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

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results


# =====================================================
# 数字模型（HOG + SVM；懒加载，缺依赖时降级为纯颜色）
# =====================================================

DIGIT_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "models", "nine_grid", "digit_classifier_mask.pkl")

# HOG 参数必须与训练一致，改动即失配
HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
)

_classifier = None
_load_failed = False


def get_classifier():
    """懒加载 SVM 分类器；不可用返回 None（原因只打印一次）"""
    global _classifier, _load_failed

    if _classifier is not None:
        return _classifier
    if _load_failed:
        return None

    try:
        import warnings
        import joblib
        with warnings.catch_warnings():
            # pkl 训练端 sklearn 版本较旧，新版本加载会有版本告警（实测不影响预测）
            warnings.simplefilter("ignore", UserWarning)
            _classifier = joblib.load(DIGIT_MODEL_PATH)
        print(f"[参考原版] 数字模型已载入: {DIGIT_MODEL_PATH}")
    except ImportError as e:
        print(f"[参考原版] 缺少依赖({e})，数字识别降级为**纯颜色**。"
              "机器人端安装: pip install scikit-learn scikit-image joblib")
        _load_failed = True
    except OSError as e:
        print(f"[参考原版] 数字模型载入失败({e})，数字识别降级为**纯颜色**。")
        _load_failed = True

    return _classifier


def extract_hog(img):
    """HOG 特征（与训练保持一致）"""
    from skimage.feature import hog

    return hog(
        img,
        **HOG_PARAMS
    )


def predict_digit(digit_mask):
    """digit_mask → (digit, confidence)；模型不可用时 → (None, 0.0)"""
    classifier = get_classifier()

    if classifier is None:

        return None, 0.0

    img = cv2.resize(
        digit_mask,
        (64, 64)
    )

    feature = extract_hog(
        img
    )

    feature = feature.reshape(
        1, -1
    )

    pred = classifier.predict(
        feature
    )[0]

    confidence = 0.5

    if hasattr(
        classifier,
        "predict_proba"
    ):

        confidence = max(
            classifier.predict_proba(feature)[0]
        )

    return int(pred), float(confidence)


# =====================================================
# 主函数：候选区域 → 颜色/模型/洞三路融合
# =====================================================

def classify_candidates(
        image,
        topk=5
):

    candidates = extract_digit_roi(
        image
    )

    if len(candidates) == 0:

        return []

    results = []

    for c in candidates[:topk]:

        color = c["color"]

        color_digit = c["color_id"]

        # digit_mask 模型
        model_digit, model_conf = predict_digit(
            c["digit_mask"]
        )

        # 洞可信度
        hole_count = c["hole_count"]

        hole_ratio = c["hole_ratio"]

        if 2 <= hole_count <= 10:

            hole_count_score = 1.0

        elif hole_count <= 20:

            hole_count_score = 0.5

        else:

            hole_count_score = 0

        if 0.05 < hole_ratio < 0.8:

            hole_ratio_score = 1.0

        elif hole_ratio < 1.2:

            hole_ratio_score = 0.5

        else:

            hole_ratio_score = 0

        hole_score = (

            0.6 * hole_count_score

            +

            0.4 * hole_ratio_score

        )

        # 最终数字判断
        digit_score = {}

        # 颜色权重
        digit_score[color_digit] = (
            0.7
        )

        # 模型权重（模型不可用时不给这一票）
        if model_digit is not None:

            digit_score[model_digit] = (
                digit_score.get(model_digit, 0)
                +
                0.2 * model_conf
            )

        # hole 只影响置信度
        final_digit = max(
            digit_score,
            key=digit_score.get
        )

        # 是否一致
        digit_match = (
            model_digit == color_digit
        )

        # candidate 总评分
        final_score = (

            0.65 * c["score"]

            +

            0.20 * model_conf

            +

            0.15 * hole_score

        )

        results.append({

            "box": c["box"],

            "roi": c["roi"],

            "digit_mask": c["digit_mask"],

            # extract
            "extract_score":
                c["score"],

            # color
            "color":
                color,

            "color_digit":
                color_digit,

            # model
            "model_digit":
                model_digit,

            "model_conf":
                model_conf,

            # hole
            "hole_ratio":
                hole_ratio,

            "hole_count":
                hole_count,

            "hole_score":
                hole_score,

            # final
            "final_digit":
                final_digit,

            "digit_match":
                digit_match,

            "final_score":
                final_score

        })

    results.sort(

        key=lambda x: x["final_score"],

        reverse=True

    )

    return results
