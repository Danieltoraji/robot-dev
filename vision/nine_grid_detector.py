# -*- coding: utf-8 -*-
"""数字宫格面板检测器（nine_grid_detector.py）

视觉方案（2026-09-08 定稿，详见 docs/关卡算法/数字宫格攻略.md）：
  - 主判 = 颜色（HSV）。关卡设计即 颜色↔数字 一一映射（1红..7粉），
    色块占面板 2/3 面积，远距可识别；HSV 分离亮度 V，色相 H 对光照稳。
  - 仲裁 = 参考 HOG+SVM 模型（黑色数字，形状特征跨光照稳）。
    只做否决/标记，不作主判：模型与颜色结论冲突时标"待近距复核"。
  - 黑色数字 mask 提取沿用参考代码"主色 HSV 距离 + Otsu"（相对阈值，
    对光照稳），HOG 参数必须与 pkl 训练一致（64x64/9 orientation/8x8 cell/
    2x2 block/L2-Hys），改动即失配。

阈值单一真源：本模块 COLOR_THRESHOLDS（收敛参考代码两套并存的问题）。
绿色/蓝色沿用参考实战结论：不用简单长方体，用 H+S/V 关系判定。
现场换光照只改这里 + tools/debug_vision.py ninegrid 子命令验证。

像素坐标约定：detect_panels 返回原生分辨率（2592x1944）坐标，与
core/ground_homography（按原生分辨率标定）直接对接；内部降采样加速。
"""

import os

import cv2
import numpy as np

from core.paths import PROJECT_ROOT

# =====================================================================
# HSV 阈值（单一真源）
# =====================================================================
# 2026-09-10 现场实测更新（11 张布局扫照片，采样见开发实录）：
# 实测贴纸 H(度)/S%/V% —— 黄42-60/30-96/63-73、橙14-26/27-86/61、
# 蓝206-214/41-83/43-56、粉336/26/63、紫242-248/32-47/33-42、红2-8/65-80/39-52。
# 白色面板地板 S 仅 3~9%、阴影 V 7~10%、外围绿色地垫会过绿色规则（靠格归属排除）。
# 自动白平衡跨帧漂移可使同一贴纸 H 漂 ~10°，故 H 区间取宽、S 下限区分地板。
# 普通颜色：(h,s,v) 长方体（OpenCV H 0-179），红为双段（H 环绕）
COLOR_THRESHOLDS = {
    "red": [((0, 110, 90), (5, 255, 255)),
            ((172, 110, 90), (180, 255, 255))],
    "orange": [((6, 85, 110), (15, 255, 255))],
    "yellow": [((17, 70, 90), (34, 255, 255))],
    # green/blue 走 build_color_mask 特殊判定，不放长方体
    "purple": [((114, 55, 60), (136, 255, 255))],
    "pink": [((160, 45, 80), (171, 255, 255))],
}

COLOR_TO_ID = {"red": 1, "orange": 2, "yellow": 3,
               "green": 4, "blue": 5, "purple": 6, "pink": 7}
ID_TO_COLOR = {v: k for k, v in COLOR_TO_ID.items()}

# 绿色：H_deg∈[100,160] 且 S/V>=1.15 且 V>=16%（深绿 S/V 1.35~2.48，水绿
# 0.27~0.98 被排除）；V 下限排除深色阴影（实测 V7~10% 的阴影曾通过比例规则）
GREEN_H_MIN_DEG, GREEN_H_MAX_DEG = 100.0, 160.0
GREEN_SV_RATIO_MIN = 1.15
GREEN_V_MIN_PERCENT = 16.0

# 蓝色：H∈[198,222]deg，V∈[25%,75%]，S 平底下限 32%（现场实测 S 低至 41%，
# 参考队的动态 S 下限 55~70% 在本场地会把蓝整块拒掉，故弃用动态规则）
BLUE_H_MIN_DEG, BLUE_H_MAX_DEG = 198.0, 222.0
BLUE_V_MIN_PERCENT, BLUE_V_MAX_PERCENT = 25.0, 75.0
BLUE_S_MIN_PERCENT = 32.0


def blue_s_min_percent(h_deg):
    """蓝色 S 下限（%）。历史版本随 H 动态（55~70%），2026-09-10 现场实测
    本场地蓝色贴纸 S 低至 41%，改为平底下限。保留函数以兼容既有调用。"""
    return BLUE_S_MIN_PERCENT


# 面板几何门槛（工作分辨率下；工作宽 1296 = 原生一半）
WORK_WIDTH = 1296
MIN_AREA_WORK = 3500     # ≈原生 14000px²；最远格(~2.2m)面板约 2 万 px²(工作分辨率)
MAX_AREA_WORK = 1_500_000
# 掠视角前向收缩：视距 d 处面板 bbox 长宽比 ≈ sqrt(d²+39²)/39（相机高39cm）。
# 4.5 ≈ 容忍 1.7m 视距（布局扫远排必然 3.2 左右）；细长噪声条仍被排除。
MAX_ASPECT_RATIO = 4.5
MIN_SOLIDITY = 0.35      # 哑光贴纸色块凸实性下限（软门槛滤碎噪）


def build_color_mask(hsv, color_name):
    """单色二值 mask。green/blue 特殊判定，其余 inRange 长方体并集。"""
    if color_name == "green":
        H = hsv[:, :, 0].astype(np.float32) * 2.0
        S = hsv[:, :, 1].astype(np.float32)
        V = hsv[:, :, 2].astype(np.float32)
        sv_ratio = S / np.maximum(V, 1.0)
        v_pct = V / 255.0 * 100.0
        mask = ((H >= GREEN_H_MIN_DEG) & (H <= GREEN_H_MAX_DEG)
                & (sv_ratio >= GREEN_SV_RATIO_MIN)
                & (v_pct >= GREEN_V_MIN_PERCENT))
        return mask.astype(np.uint8) * 255

    if color_name == "blue":
        H = hsv[:, :, 0].astype(np.float32) * 2.0
        S = hsv[:, :, 1] / 255.0 * 100.0
        V = hsv[:, :, 2] / 255.0 * 100.0
        h_ok = (H >= BLUE_H_MIN_DEG) & (H <= BLUE_H_MAX_DEG)
        v_ok = (V >= BLUE_V_MIN_PERCENT) & (V <= BLUE_V_MAX_PERCENT)
        return ((h_ok & (S >= BLUE_S_MIN_PERCENT) & v_ok)
                .astype(np.uint8) * 255)

    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in COLOR_THRESHOLDS.get(color_name, []):
        mask |= cv2.inRange(hsv, np.array(lower, np.uint8),
                            np.array(upper, np.uint8))
    return mask


def extract_digit_mask(roi_bgr):
    """从彩色面板 ROI 提取黑色数字 mask（白色=数字）。

    主色（中值 HSV）→ 加权 HSV 距离（H 权重低）→ Otsu → 开闭去噪
    → 去小连通域。相对阈值，跨光照稳（继承参考代码实测结论）。
    """
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    valid = (hsv[:, :, 1] > 40) & (hsv[:, :, 2] > 40)
    pixels = hsv[valid]
    if len(pixels) < 100:
        return np.zeros(roi_bgr.shape[:2], np.uint8)
    h0, s0, v0 = (np.median(pixels[:, i]) for i in range(3))
    H = hsv[:, :, 0].astype(np.float32)
    S = hsv[:, :, 1].astype(np.float32)
    V = hsv[:, :, 2].astype(np.float32)
    dh = np.abs(H - h0)
    dh = np.minimum(dh, 180 - dh)  # Hue 环绕
    dist = 0.5 * dh + 1.0 * np.abs(S - s0) + 1.2 * np.abs(V - v0)
    dist = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    out = np.zeros_like(mask)
    roi_area = mask.shape[0] * mask.shape[1]
    cnts = _find_contours(mask)
    for c in cnts:
        if cv2.contourArea(c) >= roi_area * 0.001:
            cv2.drawContours(out, [c], -1, 255, -1)
    return out


def _find_contours(binary):
    cnts = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cnts[0] if len(cnts) == 2 else cnts[1]


def quad_center(cnt):
    """四边形面板的透视中心 = 对角线交点

    bbox 中心在透视下偏向近端（近大远小），随视角倾斜系统性偏差可达
    面板尺寸的 10~20%，会污染单应对应点（实测 C_z 漂移 41→63cm）。
    对角线交点是投影不变量，才是透视正确的"中心"。
    approx 未得 4 点时退回实心轮廓一阶矩质心。
    """
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    if len(approx) == 4:
        p = approx.reshape(4, 2).astype(np.float64)  # 轮廓顺序（环形）
        d1, d2 = p[2] - p[0], p[3] - p[1]            # 两条对角线方向
        denom = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(denom) > 1e-6:
            t = ((p[1][0] - p[0][0]) * d2[1] - (p[1][1] - p[0][1]) * d2[0]) / denom
            return (float(p[0][0] + t * d1[0]), float(p[0][1] + t * d1[1]))
    # 兜底：approx != 4 时用实心轮廓一阶矩质心。
    # 注意 _filled_contour 返回的是 **bbox 局部坐标掩膜**，质心必须加回
    # bbox 原点——漏加会让观测中心系统性偏移数百~上千 px（现场实测：
    # 2701 橙 2 的 bbox 原点 (1862,580)、返回值却是掩膜内的 (382,239)）。
    x, y, w, h = cv2.boundingRect(cnt)
    m = cv2.moments(_filled_contour(cnt))
    if m["m00"] > 0:
        return (x + float(m["m10"] / m["m00"]), y + float(m["m01"] / m["m00"]))
    return (x + w / 2.0, y + h / 2.0)


def _filled_contour(cnt):
    """轮廓 → 实心 mask（消数字孔洞对质心的影响）"""
    x, y, w, h = cv2.boundingRect(cnt)
    mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(mask, [cnt - [x, y]], -1, 255, -1)
    return mask


# =====================================================================
# SVM 数字仲裁器
# =====================================================================

DIGIT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "nine_grid",
                                "digit_classifier_mask.pkl")

# HOG 参数必须与 pkl 训练一致，改动即失配
HOG_PARAMS = dict(orientations=9, pixels_per_cell=(8, 8),
                  cells_per_block=(2, 2), block_norm="L2-Hys")


class DigitArbiter:
    """HOG+SVM 黑色数字分类（懒加载；依赖缺失时降级为不可用）"""

    def __init__(self, model_path=DIGIT_MODEL_PATH):
        self.model_path = model_path
        self._clf = None
        self._load_failed = False

    def available(self):
        if self._clf is not None:
            return True
        if self._load_failed:
            return False
        try:
            import warnings
            import joblib
            with warnings.catch_warnings():
                # pkl 训练于 sklearn 1.6.1，加载端版本新时会有版本告警；
                # 实测 1.9.0 加载+预测正常，告警不阻断，保留记录
                warnings.simplefilter("ignore", UserWarning)
                self._clf = joblib.load(self.model_path)
            print(f"[数字仲裁] SVM 模型已加载: {self.model_path}")
        except ImportError as e:
            print(f"[数字仲裁] 依赖缺失({e})，仲裁停用（仅颜色主判）。"
                  "机器人端安装: pip install scikit-learn scikit-image joblib")
            self._load_failed = True
        except OSError as e:
            print(f"[数字仲裁] 模型加载失败({e})，仲裁停用（仅颜色主判）。")
            self._load_failed = True
        return self._clf is not None

    def predict(self, digit_mask):
        """digit_mask(64x64 或任意，内部缩放) -> (digit:int, confidence:float)"""
        if not self.available():
            return None, 0.0
        from skimage.feature import hog
        img = cv2.resize(digit_mask, (64, 64))
        feature = hog(img, **HOG_PARAMS).reshape(1, -1)
        pred = int(self._clf.predict(feature)[0])
        conf = 0.5
        if hasattr(self._clf, "predict_proba"):
            conf = float(max(self._clf.predict_proba(feature)[0]))
        return pred, conf


# =====================================================================
# 面板观测与检测器
# =====================================================================

class PanelObservation:
    """单个面板色块观测（像素坐标为原生分辨率）

    clipped / hull_centroid_px 是"裁切感知定位"的输入：
    - 未被画幅裁切时，center_px（轮廓对角线交点）就是面板中心的精确投影；
    - 被裁切时对角线交点失去几何意义（实测偏差可达数百 px），此时必须用
      hull_centroid_px（颜色掩膜凸包的面积质心）作为观测，并在定位端用
      "预测裁切四边形质心"与之配对。
    """

    __slots__ = ("color", "color_id", "bbox", "center_px", "area",
                 "solidity", "model_digit", "model_conf",
                 "clipped", "hull_centroid_px", "hull_area")

    def __init__(self, color, bbox, center_px, area, solidity,
                 model_digit=None, model_conf=0.0, clipped=False,
                 hull_centroid_px=None, hull_area=0.0):
        self.color = color
        self.color_id = COLOR_TO_ID[color]
        self.bbox = bbox                      # (x,y,w,h) 原生像素
        self.center_px = center_px            # (cx,cy) 原生像素
        self.area = area                      # 工作分辨率下面积
        self.solidity = solidity
        self.model_digit = model_digit        # SVM 仲裁结果（None=未跑/不可用）
        self.model_conf = model_conf
        self.clipped = bool(clipped)          # 是否被画幅边缘裁切
        self.hull_centroid_px = hull_centroid_px or center_px
        self.hull_area = float(hull_area)

    @property
    def digit(self):
        """主判：颜色→数字。仲裁只否决，不改主判（见类 docstring）。"""
        return self.color_id

    @property
    def arb_conflict(self):
        """仲裁冲突：SVM 高置信地反对颜色结论 → 标记待近距复核"""
        return (self.model_digit is not None and self.model_conf >= 0.6
                and self.model_digit != self.color_id
                and 1 <= self.model_digit <= 7)

    def __repr__(self):
        arb = f", svm={self.model_digit}({self.model_conf:.2f})" \
            if self.model_digit is not None else ""
        return (f"Panel({self.color}->{self.color_id}, "
                f"c_px=({self.center_px[0]:.0f},{self.center_px[1]:.0f}){arb})")


class NineGridDetector:
    """七色面板检测 + 数字仲裁

    work_width: 内部降采样工作宽（RPi 提速；返回坐标已放大回原生）
    """

    def __init__(self, work_width=WORK_WIDTH, arbiter=None,
                 min_area=MIN_AREA_WORK, max_area=MAX_AREA_WORK):
        self.work_width = work_width
        self.min_area = min_area
        self.max_area = max_area
        self.arbiter = arbiter if arbiter is not None else DigitArbiter()
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # -----------------------------------------------------------------

    def detect_panels(self, frame, colors=None, arbitrate=False,
                      drop_border=False):
        """检测画面中的面板色块，按面积降序返回 [PanelObservation]

        colors:      限定颜色列表（None=全部七色）
        arbitrate:   对每个候选跑 SVM 数字仲裁（慢，布局扫/复核时开）
        drop_border: 丢弃被画幅裁切的面板。裁切后 center_px（对角线交点）
                     与面积都系统性偏差，只适合"定归属"的布局扫；GN 定位
                     路径默认关闭——裁切面板改用 hull_centroid_px 参与解算
                     （见 PanelObservation 与 levels/nine_grid._gn_run）。

        同色碎块合并：黑色数字块/画幅裁切会把同一面板的颜色掩膜切成多块
        （bbox 互不重叠，跨色 IoU 去重抓不到）。本关卡每种颜色只有一块面板，
        故把"间距小于较大块尺寸"的同色碎块并成一个观测，取凸包为可见区域。
        """
        if frame is None or frame.size == 0:
            return []
        scale = frame.shape[1] / self.work_width
        work = cv2.resize(frame, (self.work_width,
                                  int(round(frame.shape[0] / scale))))
        hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
        border_px = 8  # 工作分辨率下贴边判据
        work_h = work.shape[0]

        raw = []
        for color in (colors or COLOR_TO_ID.keys()):
            mask = build_color_mask(hsv, color)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
            frags = []
            for cnt in _find_contours(mask):
                area = cv2.contourArea(cnt)
                if not (self.min_area <= area <= self.max_area):
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                if max(w / h, h / w) > MAX_ASPECT_RATIO:
                    continue
                hull_area = cv2.contourArea(cv2.convexHull(cnt))
                if hull_area <= 0:
                    continue
                solidity = area / hull_area
                if solidity < MIN_SOLIDITY:
                    continue
                frags.append((area, (x, y, w, h), solidity, cnt))
            if not frags:
                continue

            # 同色碎块合并（数字孔洞/画幅裁切造成的分裂）
            frags.sort(key=lambda f: f[0], reverse=True)
            main = frags[0]
            merge_gap = max(main[1][2], main[1][3])
            group = [main] + [f for f in frags[1:]
                              if _bbox_gap(f[1], main[1]) <= merge_gap]
            hull = _hull_of([g[3] for g in group])
            area = float(cv2.contourArea(hull))
            if area <= 0:
                continue
            x, y, w, h = cv2.boundingRect(hull)
            clipped = (x < border_px or y < border_px
                       or x + w > self.work_width - border_px
                       or y + h > work_h - border_px)
            if drop_border and clipped:
                continue
            raw.append((color, area, (x, y, w, h), main[2], hull,
                        quad_center(hull), _contour_centroid(hull), clipped))

        # 跨色去重：同区域多色命中时保留面积大者（阈值交界抖动）
        raw.sort(key=lambda r: r[1], reverse=True)
        kept = []
        for item in raw:
            dup = False
            for k in kept:
                if _iou(item[2], k[2]) > 0.5:
                    dup = True
                    break
            if not dup:
                kept.append(item)

        results = []
        for (color, area, (x, y, w, h), solidity, _hull, center,
             hull_centroid, clipped) in kept:
            obs = PanelObservation(
                color=color,
                bbox=(x * scale, y * scale, w * scale, h * scale),
                center_px=(center[0] * scale, center[1] * scale),
                area=area,
                solidity=solidity,
                clipped=clipped,
                hull_centroid_px=(hull_centroid[0] * scale,
                                  hull_centroid[1] * scale),
                hull_area=area * scale * scale,
            )
            if arbitrate:
                roi = work[y:y + h, x:x + w]
                mask_roi = build_color_mask(hsv[y:y + h, x:x + w], color)
                digit_mask = _digit_mask_for_arbitration(roi, mask_roi)
                obs.model_digit, obs.model_conf = self.arbiter.predict(
                    digit_mask)
            results.append(obs)
        return results

    def color_ratio(self, frame, color_name):
        """整帧中某颜色的面积占比（CONFIRM 到达判定用）"""
        if frame is None or frame.size == 0:
            return 0.0
        scale = frame.shape[1] / self.work_width
        work = cv2.resize(frame, (self.work_width,
                                  int(round(frame.shape[0] / scale))))
        hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
        mask = build_color_mask(hsv, color_name)
        return float(cv2.countNonZero(mask)) / mask.size

    def annotate(self, frame, observations):
        """调试可视化：框出面板并标注颜色→数字/仲裁结果"""
        vis = frame.copy()
        for o in observations:
            x, y, w, h = (int(v) for v in o.bbox)
            label = f"{o.color}->{o.digit}"
            if o.model_digit is not None:
                label += f" svm:{o.model_digit}({o.model_conf:.2f})"
            if o.arb_conflict:
                label += " !复核"
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 6)
            cv2.drawMarker(vis, (int(o.center_px[0]), int(o.center_px[1])),
                           color, cv2.MARKER_CROSS, 60, 4)
            cv2.putText(vis, label, (x, max(40, y - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 4)
        return vis


def _digit_mask_for_arbitration(roi_bgr, color_mask_roi):
    """仲裁输入：黑色数字 mask（优先参考法；主色 mask 反相兜底）

    参考法（extract_digit_mask）在 ROI 过小时 Otsu 可能不稳；
    颜色 mask 反相（彩色背景之外即数字+阴影）作兜底融合。
    """
    m = extract_digit_mask(roi_bgr)
    if cv2.countNonZero(m) > 0.005 * m.size:
        return m
    inv = cv2.bitwise_not(color_mask_roi)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel)


def _iou(a, b):
    """(x,y,w,h) 框 IoU"""
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1 = min(ax0 + aw, bx0 + bw)
    iy1 = min(ay0 + ah, by0 + bh)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _bbox_gap(a, b):
    """两 (x,y,w,h) 框的最短间距（相交为 0）"""
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    dx = max(0.0, max(ax0 - (bx0 + bw), bx0 - (ax0 + aw)))
    dy = max(0.0, max(ay0 - (by0 + bh), by0 - (ay0 + ah)))
    return float(np.hypot(dx, dy))


def _hull_of(contours):
    """多轮廓点集 → 凸包（同色碎块合并后的可见面板区域）"""
    pts = np.vstack([c.reshape(-1, 2) for c in contours])
    return cv2.convexHull(pts.astype(np.int32))


def _contour_centroid(cnt):
    """轮廓/多边形的面积质心（裁切不变；数字孔洞被凸包填掉）"""
    m = cv2.moments(cnt)
    if m["m00"] > 1e-9:
        return (float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"]))
    x, y, w, h = cv2.boundingRect(cnt)
    return (x + w / 2.0, y + h / 2.0)
