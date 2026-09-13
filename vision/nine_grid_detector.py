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
    "pink": [((137, 30, 80), (175, 255, 255))],
}

# =====================================================================
# 七色调色板（归一化 HSV 空间；2026-09-11 起为**生效阈值**）
# =====================================================================
# 由 tools/calibrate_palette.py 从 13 张实拍标注帧（tests/fixtures/field_photos，
# 含 3 张真机探针帧）拟合，规整为整数 H 区间：
#   - H 边界 = **相邻色实测样本之间的缝隙中点**（gap-midpoint）→ 互相排斥、
#     不留缝、不重叠。先试过"相邻色中心中点"（Voronoi，红↔橙 = 4.5），结果
#     红在 13 帧里只命中 2 帧：红贴纸实测 H 落在 178~5，中心中点把 5 判给橙。
#     改成缝隙中点（红 ≤5 / 橙 ≥6）后 73/73 唯一命中、0 多命中、0 漏检。
#     实测相邻色 H 中心间隔：红↔橙 7、橙↔黄 12、蓝↔紫 18、粉↔红 13.5（环绕），
#     其余 ≥42。红↔橙只差 7° 是硬伤：H 单独判不了，交给形状仲裁（C3）兜。
#   - S/V 下限只承担"不是白板、不是黑字"的职责（白板 S 3~9%、黑字 V<24%），
#     故意取松：S/V 会随曝光与光照整体漂移，取紧就重现"整块漏检"（第一版
#     拟合用 min×0.8 得到红 S≥49%、黄 S≥54%，直接拒掉真实样本）。
#     场景杂物（木框/地垫）交给数字证据门 + 格阵共识拦，不靠颜色窗口硬扛。
#   - 现场复标：python tools/calibrate_palette.py --write
#     （写 models/nine_grid/palette.json，运行时优先读它，失败回退本表）。
USE_PALETTE = True
PALETTE = {          # color: (h_lo, h_hi, s_lo%, v_lo%)；h_lo > h_hi = 环绕窗口
    "red":    (176, 5, 25.0, 25.0),     # 环绕另一段 (0, 5] 见 _palette_ranges
    "orange": (6, 15, 25.0, 25.0),
    "yellow": (16, 42, 22.9, 25.0),
    "green":  (43, 87, 25.0, 19.7),
    "blue":   (88, 113, 25.0, 25.0),
    "purple": (114, 143, 25.0, 25.0),
    "pink":   (144, 175, 10.0, 25.0),   # 粉贴纸本身饱和度低（实测 S 12~35%）
}
PALETTE_JSON_PATH = os.path.join(PROJECT_ROOT, "models", "nine_grid",
                                 "palette.json")
_PALETTE_OVERRIDE = None       # (生效色数, 路径) —— _active_palette 的缓存


def _active_palette():
    """生效调色板：models/nine_grid/palette.json 存在则覆盖（现场复标产物）"""
    global _PALETTE_OVERRIDE
    if _PALETTE_OVERRIDE is not None:
        return _PALETTE_OVERRIDE
    pal = dict(PALETTE)
    try:
        import json
        if os.path.exists(PALETTE_JSON_PATH):
            with open(PALETTE_JSON_PATH, "r", encoding="utf-8") as f:
                fitted = json.load(f).get("fitted", {})
            n = 0
            for color, w in fitted.items():
                if color in pal and "h_lo" in w:
                    pal[color] = (int(round(w["h_lo"])), int(round(w["h_hi"])),
                                  float(w["s_lo"]), float(w["v_lo"]))
                    n += 1
            print(f"[调色板] 已加载现场复标 {PALETTE_JSON_PATH}（{n} 色覆盖）")
    except (OSError, ValueError, KeyError) as e:
        print(f"[调色板] {PALETTE_JSON_PATH} 读取失败({e})，用代码内缺省窗口")
    _PALETTE_OVERRIDE = pal
    return pal


def _palette_ranges(color_name, palette=None):
    """调色板窗口 → [(h_lo, h_hi, s_lo8, v_lo8), ...]（S/V 换算到 0-255；红两段）"""
    pal = palette if palette is not None else _active_palette()
    w = pal.get(color_name)
    if w is None:
        return []
    h_lo, h_hi, s_lo, v_lo = w
    s8, v8 = int(round(s_lo * 2.55)), int(round(v_lo * 2.55))
    if h_lo <= h_hi:
        return [(int(h_lo), int(h_hi), s8, v8)]
    return [(0, int(h_hi), s8, v8), (int(h_lo), 179, s8, v8)]   # 环绕拆两段

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
# 同色碎块合并准则（见 detect_panels）：间距上限 = 主块短边 × MERGE_GAP_FRAC，
# 面积比下限 = 主块面积 × MERGE_AREA_FRAC。
# 依据：黑字把色块切成两块时间距 ≈ 笔画宽（工作分辨率 20~40px，取 0.35×短边
# 有余量）；而场内同色杂物（木框、蓝地垫）与面板间距远大于此、面积比也更小。
MERGE_GAP_FRAC = 0.35
MERGE_AREA_FRAC = 0.15

# 数字证据（C1，"这块色斑像不像印刷面板"的结构判据）
# ---------------------------------------------------------------------
# 真面板 = 色块 **内部印着黑色数字**；场内同色杂物（木框 H12/S33%/V126、
# 蓝地垫 H105/S38%、深绿地垫）只是"颜色接近"，没有黑字。用"凸包内 V<60 的
# 像素占比"衡量数字证据，实测（13 张实拍帧的合并观测）：
#   真面板（未裁切）0.0184~0.1262；粉 7 最差帧（强暖偏色 probe_c）0.0184
#   同色杂物        0.0000~0.0085（木框橙 0.0085／蓝地垫 0.0032／绿地垫 0）
# 但**证据低 ≠ 杂物**：实测 photo_1789022703/706 的真面板"绿 4"呈 0.0000
# ——数字被场内白色物件挡住，色块本身没被画幅裁切。故：
#   - 只对**极端深色块**（证据 > DIGIT_EVIDENCE_MAX，实测杂物 0.2650 vs
#     真面板上界 0.1262）做硬过滤：大片深色 ⇒ 阴影/深色物件，必非印刷面板；
#   - 低证据只作**软信号**：写进观测的 digit_evidence，供导航 `_see_target`
#     在"真面板 vs 木框"之间择大时优先取有证据的那块（木框橙实测 93k px
#     比真橙面板 62k 还大，纯按面积择大会追着木框跑）。
#   宁可放过、不可误杀：漏检可以用 color_ratio 兜（到达判据不依赖掩膜分类），
#   误杀真面板会让导航看不到目标。
DIGIT_EVIDENCE_ENABLED = True
DIGIT_V_MAX = 60              # "黑字"判据（归一化工作帧：黑字 V≈15~55，贴纸自身 V≥70）
DIGIT_EVIDENCE_MIN = 0.012    # 有证据判据（软，供 _see_target 优先择取）
DIGIT_EVIDENCE_MAX = 0.22     # 硬门上限：占比过大 = 大片阴影/深色物件
DIGIT_EVIDENCE_MIN_AREA = 5000   # 面积下限（工作分辨率）：低于此判不了，免检

# 形状仲裁（C3）：**只在颜色有歧义时**才允许用数字形状改判
# ---------------------------------------------------------------------
# 颜色是主判（现场光照下 66/66 唯一命中），形状只在两色边界打架时当裁判：
#   - 歧义信号一：同一位置被两种颜色的掩膜同时命中（跨色去重 IoU > AMBIG_IOU）；
#   - 歧义信号二：色块中位 H 距本窗口边界 < AMBIG_H_MARGIN_DEG（红↔橙只差 7°，
#     现场光照漂移足以把红贴纸推到橙窗口边）。
#   - 裁决依据：现场帧自建的数字模板（tools/gen_ninegrid_digit_templates.py）。
#     留一实测（2026-09-13，25 帧 71 样本）：间隔 ≥0.30 时 24/24 正确（覆盖
#     24/71），≥0.20 时 35/37（95%），≥0.15 时 43/45（96%）。取 0.30——
#     改判颜色是"能翻盘"的动作，宁可少判也不要用形状把对的颜色改错
#     （旧 SVM 在现场帧与颜色一致率仅 23%，等于用噪声否决真值）。
SHAPE_ENABLED = True
SHAPE_TEMPLATE_PATH = os.path.join(PROJECT_ROOT, "models", "nine_grid",
                                   "digit_templates.npz")
SHAPE_OVERRIDE_MIN_CONF = 0.30
AMBIG_IOU = 0.5
AMBIG_H_MARGIN_DEG = 4.0
# 墨迹（黑色数字）提取参数：形状仲裁与模板生成共用，保证训练/推理同一链路
GLYPH_ERODE_PX = 7
GLYPH_V_FRAC = 0.60
GLYPH_MIN_AREA = 40


def extract_glyph_mask(hsv, color_mask, bbox):
    """色块内的黑色数字墨迹掩膜（工作分辨率）→ (mask|None, 诊断)

    数字是**色域里的洞**（黑字不满足该色窗口），必须先"填洞"再内缩——直接
    对色域腐蚀会让洞跟着变大，墨迹像素全被排除（实测 dark∩inner=0/5387）。
    填洞 = 取最大外轮廓实心填充（外轮廓天然包含洞）。
    """
    x, y, w, h = [int(v) for v in bbox]
    cm = color_mask[y:y + h, x:x + w]
    if cm.size == 0 or int(np.count_nonzero(cm)) < 50:
        return None, {"why": "色域为空"}
    cnts = sorted(_find_contours(cm), key=cv2.contourArea, reverse=True)
    if not cnts:
        return None, {"why": "无色域轮廓"}
    filled = np.zeros((h, w), np.uint8)
    cv2.drawContours(filled, [cnts[0]], -1, 255, -1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                    (GLYPH_ERODE_PX, GLYPH_ERODE_PX))
    inner = cv2.erode(filled, ker)
    if cv2.countNonZero(inner) < 50:
        return None, {"why": "内缩后无像素"}
    v = hsv[y:y + h, x:x + w, 2]
    vmed = float(np.median(v[cm > 0]))
    dark = ((v < GLYPH_V_FRAC * vmed).astype(np.uint8)) * 255
    dark = cv2.bitwise_and(dark, inner)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    cnts = [c for c in _find_contours(dark)
            if cv2.contourArea(c) >= GLYPH_MIN_AREA]
    if not cnts:
        return None, {"why": "无墨迹连通域", "v_med": vmed}
    big = max(cnts, key=cv2.contourArea)
    out = np.zeros_like(dark)
    cv2.drawContours(out, [big], -1, 255, -1)
    return out, {"area": float(cv2.contourArea(big)), "v_med": vmed}


# =====================================================================
# 光照归一化（2026-09-11：消除现场光照/自动白平衡漂移）
# =====================================================================
# 现场实测（真机探针帧 photo_1789213170）：同一场地、同一相机的白底板 BGR
# 从 [153,155,149] 漂到 [209,186,180]（R/B 差 16%，相机 white_balance_automatic=1
# 所致）→ 粉贴纸 H 由 168 掉到 141，整块掉出手写窗口而漏检（布局扫缺数字 7）。
# 用画面里的**白底板**做灰世界归一化后，粉 H 回到 164（粉−紫间隔 24→46），
# 对正常帧几乎无影响（168→169）。
# 白参考 = 低饱和(S<WHITE_S_MAX)且较亮(V>WHITE_V_MIN)像素的中位 BGR。
# 归一化只是"修正光源"，不改变场景几何；过曝（分量打满）是信息损失，
# 归一化救不回来 → 单独统计 clip_frac 供真机提示"锁/降曝光"。
NORM_ENABLED = True
WHITE_S_MAX = 40            # 白参考像素的 S 上限（OpenCV 0-255）
WHITE_V_MIN = 120           # 白参考像素的 V 下限
WHITE_MIN_SAMPLES = 200     # 白参考像素数下限
WHITE_MIN_FRAC = 0.005      # 白参考像素占比下限（低于此不归一化）
NORM_GAIN_MIN, NORM_GAIN_MAX = 0.60, 1.80   # 硬限幅（防白点估计异常时放大噪声）
NORM_GAIN_WARN_MIN, NORM_GAIN_WARN_MAX = 0.75, 1.40   # 告警区间（现场需关注）
# 曝光归一化目标：把白点三通道均值拉到该值。**只做白平衡是不够的**——均匀
# 变暗/变亮（曝光漂移）时白点跟着一起缩放，纯白平衡增益恒为 1、等于没做，
# 实测 -25% 曝光下深绿面板 V 跌破调色板下限而整块漏检。13 张实拍帧白点均值
# 中位 154.3（范围 135~192），故取 155 为目标。
NORM_TARGET_WHITE_MEAN = 155.0
NORM_CLIP_LEVEL = 250       # "打满"判据
NORM_CLIP_WARN_FRAC = 0.05  # 过曝告警阈值（任一分量打满的像素占比）
WHITE_BRIGHT_PCTL = 95      # "亮参考"分位（白参考的亮度下限按它取相对值）
WHITE_V_REL = 0.70          # 白参考亮度下限 = 该比例 × 亮参考分位
WHITE_S_FALLBACK = 120      # 兜底一的饱和度上限（白板强偏色后 S≈100~115）
WHITE_V_FALLBACK_REL = 0.55
WHITE_FALLBACK_MIN_FRAC = 0.01


def estimate_white_bgr(frame):
    """估计光源白点 → (wp_bgr|None, frac, 方法)

    主判：低饱和 + 较亮像素的中位 BGR（本场地 = 白底板，占比很大）。
    **兜底**：强偏色会把白底板本身推到 S>WHITE_S_MAX（实测 B×0.75 的暖偏色
    帧里白板变 (115,147,186)、S=98 ⇒ 主判一个样本都选不出来，归一化直接
    不生效），此时改用"最亮 WHITE_BRIGHT_PCTL 分位以上的像素"——白底板仍是
    画面最亮的大面积区域，取它的中位色彩与饱和度判据无关。
    大面积帧按 1/4 抽样估计（全局增益与分辨率无关，省时间）。
    """
    if frame is None or frame.size == 0:
        return None, 0.0, "none"
    small = frame[::4, ::4] if frame.shape[1] > 800 else frame
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    # 白参考还必须"够亮"：否则会挑到画面里的灰暗色块（机器人本体、阴影里的
    # 浅灰地面）——它们也满足"低饱和"，但完全不代表光源。实测人为双重偏色
    # （B×0.7,R×1.1 叠在 probe_c 强暖偏上）时，主判挑中的"白点"亮度只有
    # 全场 95 分位的 64%，比色随之为近中性 ⇒ 归一化形同没做、粉 7 被推成红。
    bright_ref = float(np.percentile(hsv[:, :, 2], 95))
    v_floor = max(float(WHITE_V_MIN), WHITE_V_REL * bright_ref)
    m = (hsv[:, :, 1] < WHITE_S_MAX) & (hsv[:, :, 2] > v_floor)
    n = int(np.count_nonzero(m))
    frac = n / float(m.size) if m.size else 0.0
    if n >= WHITE_MIN_SAMPLES and frac >= WHITE_MIN_FRAC:
        return np.median(small[m].astype(np.float64), axis=0), frac, "neutral"
    # 兜底一：放宽饱和度上限（强偏色下白板自身 S 会升到 100+，但仍远低于
    # 彩贴纸的 180~220），同时要求够亮。**不能只按"最亮"取**——暖偏色下
    # 橙/黄贴纸比白板还亮（实测 B×0.75/R×1.25 后橙面板 V=250 > 白板 186），
    # 纯最亮会把橙面板当白点，增益反被带飞。
    v_floor2 = max(float(WHITE_V_MIN), WHITE_V_FALLBACK_REL * bright_ref)
    m = (hsv[:, :, 1] < WHITE_S_FALLBACK) & (hsv[:, :, 2] > v_floor2)
    n = int(np.count_nonzero(m))
    frac2 = n / float(m.size) if m.size else 0.0
    if n >= WHITE_MIN_SAMPLES and frac2 >= WHITE_FALLBACK_MIN_FRAC:
        return np.median(small[m].astype(np.float64), axis=0), frac2, "bright"
    # 不再往"纯最亮"退：合成/无白底场景里最亮的往往是彩贴纸本身（实测合成帧
    # 灰底 V=90、面板 V=200 ⇒ 纯最亮会把面板当白点，整帧被带偏、误检丛生）。
    # 找不到可信白点就**不归一化**（返回原帧），由现场锁曝光兜底。
    return None, frac, "none"


def normalize_illumination(frame, enabled=None):
    """灰世界归一化 → (frame_u8, info)

    info = {"applied": bool, "white_bgr": [b,g,r]|None, "gain": [gb,gg,gr],
            "white_frac": float, "clip_frac": float, "suspicious": bool}
    白参考不足或 NORM_ENABLED=False 时原样返回（applied=False）。
    """
    use = NORM_ENABLED if enabled is None else bool(enabled)
    info = {"applied": False, "white_bgr": None, "gain": [1.0, 1.0, 1.0],
            "white_frac": 0.0, "clip_frac": 0.0, "suspicious": False}
    if frame is None or frame.size == 0:
        return frame, info
    if not use:
        return frame, info
    small = frame[::4, ::4] if frame.shape[1] > 800 else frame
    info["clip_frac"] = float(np.count_nonzero(
        (small[:, :, 0] >= NORM_CLIP_LEVEL) | (small[:, :, 1] >= NORM_CLIP_LEVEL)
        | (small[:, :, 2] >= NORM_CLIP_LEVEL))) / float(small.shape[0]
        * small.shape[1]) if small.size else 0.0
    wp, frac, method = estimate_white_bgr(frame)
    info["white_frac"] = float(frac)
    info["white_method"] = method
    if wp is None:
        return frame, info
    gain = NORM_TARGET_WHITE_MEAN / np.maximum(wp, 1.0)
    info["white_bgr"] = [float(v) for v in wp]
    info["suspicious"] = bool(np.any(gain < NORM_GAIN_WARN_MIN)
                              or np.any(gain > NORM_GAIN_WARN_MAX))
    gain = np.clip(gain, NORM_GAIN_MIN, NORM_GAIN_MAX)
    info["gain"] = [float(v) for v in gain]
    out = np.clip(frame.astype(np.float32) * gain.reshape(1, 1, 3),
                  0, 255).astype(np.uint8)
    info["applied"] = True
    return out, info


def build_color_mask(hsv, color_name):
    """单色二值 mask

    缺省走**归一化空间的 PALETTE（Voronoi H 分界 + 松 S/V 下限）**；
    USE_PALETTE=False 时回退历史实现（green/blue 走 H+S/V 特殊判定，
    其余走 COLOR_THRESHOLDS 长方体）——保留以便现场应急与对照。
    """
    if USE_PALETTE:
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for h_lo, h_hi, s_lo, v_lo in _palette_ranges(color_name):
            mask |= cv2.inRange(hsv, np.array([h_lo, s_lo, v_lo], np.uint8),
                                np.array([h_hi, 255, 255], np.uint8))
        return mask

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
                 "clipped", "hull_centroid_px", "hull_area", "digit_evidence",
                 "shape_digit", "shape_conf", "ambig_color", "ambig_margin_deg",
                 "ambig_candidates")

    def __init__(self, color, bbox, center_px, area, solidity,
                 model_digit=None, model_conf=0.0, clipped=False,
                 hull_centroid_px=None, hull_area=0.0, digit_evidence=None,
                 shape_digit=None, shape_conf=0.0, ambig_color=None,
                 ambig_margin_deg=None, ambig_candidates=()):
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
        self.digit_evidence = digit_evidence   # 凸包内黑字占比（None=未算）
        self.shape_digit = shape_digit         # 形状仲裁结果（None=未跑/不可用）
        self.shape_conf = float(shape_conf)    # 匹配间隔（最佳−次佳），越大越敢用
        self.ambig_color = ambig_color         # 主要竞争色（跨色去重的落败者）
        self.ambig_margin_deg = ambig_margin_deg   # 中位 H 距本窗口边界的角度
        self.ambig_candidates = tuple(ambig_candidates)   # 全部竞争色

    @property
    def digit(self):
        """主判：颜色→数字。形状只在该颜色**有歧义**时才允许改判（C3）"""
        if self.shape_override:
            return self.shape_digit
        return self.color_id

    @property
    def ambiguous(self):
        """颜色是否有歧义：同位置双色命中，或中位 H 贴着本窗口边界"""
        if self.ambig_candidates or self.ambig_color is not None:
            return True
        return (self.ambig_margin_deg is not None
                and self.ambig_margin_deg < AMBIG_H_MARGIN_DEG)

    @property
    def candidates(self):
        """歧义时的候选数字（颜色主判 + 全部竞争色）"""
        cands = {self.color_id}
        for c in list(self.ambig_candidates) + (
                [self.ambig_color] if self.ambig_color else []):
            if c in COLOR_TO_ID:
                cands.add(COLOR_TO_ID[c])
        return cands

    @property
    def shape_override(self):
        """是否由形状改判（诊断/日志用）"""
        if not (SHAPE_ENABLED and self.shape_digit is not None):
            return False
        if self.shape_conf < SHAPE_OVERRIDE_MIN_CONF:
            return False
        if self.shape_digit == self.color_id:
            return False
        return self.ambiguous and self.shape_digit in self.candidates

    @property
    def arb_conflict(self):
        """仲裁冲突：SVM 高置信地反对颜色结论 → 标记待近距复核"""
        return (self.model_digit is not None and self.model_conf >= 0.6
                and self.model_digit != self.color_id
                and 1 <= self.model_digit <= 7)

    def has_digit_evidence(self):
        """是否"看起来像印刷面板"（软判据，见 DIGIT_EVIDENCE_*）

        证据足够 → True；证据不足但**判不了**（贴画幅边、太小、未算）→ True
        （宁可放过）；只有"面积够大、未裁切、却几乎没有黑字"才 False——
        现场实测这类多半是木框/地垫等场内同色杂物。
        """
        e = self.digit_evidence
        if e is None or self.clipped or self.area < DIGIT_EVIDENCE_MIN_AREA:
            return True
        return e >= DIGIT_EVIDENCE_MIN

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
        # 形状仲裁器（现场自建数字模板；懒加载，缺文件即降级为不可用）
        self._shape_arb = None
        self._shape_tried = False
        # 最近一次 detect_panels/color_ratio 的光照归一化信息（诊断/工具用）
        self.last_norm = None
        # 最近一次 detect_panels 被数字证据门拒绝的观测（诊断/工具用）
        self.last_rejected = []

    @property
    def shape_arb(self):
        """形状仲裁器：models/nine_grid/digit_templates.npz（缺失=不可用）"""
        if not self._shape_tried:
            self._shape_tried = True
            try:
                from .digit_recognizer import DigitRecognizer
                if os.path.exists(SHAPE_TEMPLATE_PATH):
                    data = np.load(SHAPE_TEMPLATE_PATH, allow_pickle=False)
                    tmpl = {k[1:]: data[k] for k in data.files
                            if k.startswith("d") and k[1:].isdigit()}
                    if tmpl:
                        self._shape_arb = DigitRecognizer(templates=tmpl)
                        print(f"[形状仲裁] 数字模板已加载: {SHAPE_TEMPLATE_PATH}"
                              f"（{len(tmpl)} 个数字）")
            except (OSError, ValueError, KeyError) as e:
                print(f"[形状仲裁] 模板加载失败({e})，形状仲裁停用")
        return self._shape_arb

    # -----------------------------------------------------------------

    def detect_panels(self, frame, colors=None, arbitrate=False,
                      drop_border=False, shape=False):
        """检测画面中的面板色块，按面积降序返回 [PanelObservation]

        colors:      限定颜色列表（None=全部七色）
        arbitrate:   对每个候选跑 SVM 数字仲裁（慢，布局扫/复核时开）
        shape:       对每个候选跑**数字形状仲裁**（现场自建模板；颜色有歧义
                     时才用于改判，见 PanelObservation.digit）
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
        # 光照归一化（工作分辨率上做：掩膜/面积都在这一层，等价且省时间）
        work, self.last_norm = normalize_illumination(work)
        hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
        border_px = 8  # 工作分辨率下贴边判据
        work_h = work.shape[0]
        self.last_rejected = []

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

            # 同色碎块合并（只针对"同一块面板被黑字/画幅切开"的情形）
            #
            # 历史 bug（2026-09-11 真机探针帧 photo_1789127583）：旧准则
            # gap <= max(主块宽,高) 过于宽松——黄面板（工作 bbox 393x212）与
            # 画面左侧木框的同色碎块间距 266px 也被合并 → 合并 hull 横跨到
            # 左边框 → 面板被误判 clipped=True，观测中心偏 600 原生 px。
            # 现准则：间距 ≤ 主块**短边**的 MERGE_GAP_FRAC（黑字笔画在工作
            # 分辨率约 20~40px，0.35×短边足够桥接），且候选面积 ≥ 主块的
            # MERGE_AREA_FRAC（挡住木框/地垫等场地同色杂物）。
            frags.sort(key=lambda f: f[0], reverse=True)
            main = frags[0]
            merge_gap = MERGE_GAP_FRAC * min(main[1][2], main[1][3])
            group = [main] + [
                f for f in frags[1:]
                if f[0] >= MERGE_AREA_FRAC * main[0]
                and _bbox_gap(f[1], main[1]) <= merge_gap]
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
            # 数字证据门（C1）：色块内没有黑字且不贴边 ⇒ 场内同色杂物
            ev = digit_evidence(hsv[:, :, 2], hull, (x, y, w, h))
            keep, why = digit_evidence_ok(area, clipped, ev)
            if not keep:
                self.last_rejected.append((color, (x, y, w, h), ev, why))
                continue
            raw.append((color, area, (x, y, w, h), main[2], hull,
                        quad_center(hull), _contour_centroid(hull), clipped, ev))

        # 跨色去重：同区域多色命中时保留面积大者（阈值交界抖动）。
        # 落败色**不丢**：记到胜者的 ambig_color 上——同位置双色命中就是
        # 颜色歧义的直接证据，交给形状仲裁裁决（见 PanelObservation.digit）。
        raw.sort(key=lambda r: r[1], reverse=True)
        kept = []
        losers = {}          # id(胜者元组) -> 落败色名
        for item in raw:
            dup = False
            for k in kept:
                if _iou(item[2], k[2]) > AMBIG_IOU:
                    dup = True
                    losers.setdefault(id(k), item[0])
                    break
            if not dup:
                kept.append(item)

        results = []
        for item in kept:
            (color, area, (x, y, w, h), solidity, _hull, center,
             hull_centroid, clipped, ev) = item
            # 颜色歧义度：色块像素的中位 H 距本窗口两侧边界的角度（取小者）。
            # 红↔橙窗口只差 7°，中位 H 贴边就说明光照把贴纸推到了边界。
            mask_full = build_color_mask(hsv, color)
            sub = mask_full[y:y + h, x:x + w]
            hmed = (float(np.median(hsv[y:y + h, x:x + w, 0][sub > 0]))
                    if np.count_nonzero(sub) else None)
            amb_margin = None
            amb_list = []
            if hmed is not None:
                h_lo, h_hi, _s, _v = _active_palette()[color]
                d_lo = (float(hmed) - h_lo) % 180.0
                d_hi = (h_hi - float(hmed)) % 180.0
                amb_margin = float(min(d_lo, d_hi))
                # 中位 H 贴边 ⇒ 竞争色 = 窗口边界另一侧的颜色（查询窗口外 1°，
                # 查窗口内会命中本色而返回 None——曾据此把竞争色全丢掉）
                if d_lo < AMBIG_H_MARGIN_DEG:
                    nb = palette_neighbor(color, h_lo - 1.0)
                    if nb:
                        amb_list.append(nb)
                if d_hi < AMBIG_H_MARGIN_DEG:
                    nb = palette_neighbor(color, h_hi + 1.0)
                    if nb:
                        amb_list.append(nb)
            loser = losers.get(id(item))
            if loser and loser not in amb_list:
                amb_list.append(loser)
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
                digit_evidence=ev,
                ambig_color=loser,
                ambig_margin_deg=amb_margin,
                ambig_candidates=amb_list,
            )
            if arbitrate:
                roi = work[y:y + h, x:x + w]
                mask_roi = build_color_mask(hsv[y:y + h, x:x + w], color)
                digit_mask = _digit_mask_for_arbitration(roi, mask_roi)
                obs.model_digit, obs.model_conf = self.arbiter.predict(
                    digit_mask)
            if shape and SHAPE_ENABLED:
                gm, _diag = extract_glyph_mask(hsv, mask_full, (x, y, w, h))
                if gm is not None and self.shape_arb is not None:
                    obs.shape_digit, obs.shape_conf, _ = \
                        self.shape_arb.match_mask(gm)
            results.append(obs)
        return results

    def color_ratio(self, frame, color_name):
        """整帧中某颜色的面积占比（低头到达判定用）

        与 detect_panels 同一条光照归一化链路（否则颜色占比会被现场光照带偏，
        v3 到达判据用的是"占比峰值/跌落"相对量，但也必须在同一归一化下比较）。
        """
        if frame is None or frame.size == 0:
            return 0.0
        scale = frame.shape[1] / self.work_width
        work = cv2.resize(frame, (self.work_width,
                                  int(round(frame.shape[0] / scale))))
        work, self.last_norm = normalize_illumination(work)
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


def palette_neighbor(color_name, h_query):
    """调色板中"另一个"包含该 H 的颜色名（色块中位 H 贴边时找竞争色）

    窗口按整数 H 平铺（相邻窗口留 0~1° 缝），故对 h_query 逐色判断包含关系，
    取第一个非本色命中者。
    """
    pal = _active_palette()
    for other, (h_lo, h_hi, _s, _v) in pal.items():
        if other == color_name:
            continue
        h = float(h_query) % 180.0
        if h_lo <= h_hi:
            hit = h_lo <= h <= h_hi
        else:
            hit = h >= h_lo or h <= h_hi
        if hit:
            return other
    return None


def digit_evidence(v_plane, hull, bbox):
    """凸包内"印刷黑字"像素占比（数字证据，见 DIGIT_EVIDENCE_* 说明）

    v_plane: 工作帧的 HSV V 通道；hull: 合并后的凸包轮廓；bbox: 其 (x,y,w,h)
    """
    x, y, w, h = bbox
    m = np.zeros((h, w), np.uint8)
    cv2.drawContours(m, [hull - [x, y]], -1, 255, -1)
    inside = m > 0
    n = int(np.count_nonzero(inside))
    if n <= 0:
        return 0.0
    dark = np.count_nonzero(inside & (v_plane[y:y + h, x:x + w] < DIGIT_V_MAX))
    return dark / float(n)


def digit_evidence_ok(area, clipped, evidence):
    """数字证据硬门 → (是否保留, 原因)；只有"极端深色块"才丢

    低证据**不丢**（可能是数字被遮挡的真面板，见 DIGIT_EVIDENCE_* 说明），
    原因串 "low_evidence"/"clipped"/"too_small" 仍返回 True，只作日志。
    """
    if not DIGIT_EVIDENCE_ENABLED:
        return True, ""
    if area < DIGIT_EVIDENCE_MIN_AREA:
        return True, "too_small"
    if evidence > DIGIT_EVIDENCE_MAX:
        return False, "dark_blob"
    if evidence >= DIGIT_EVIDENCE_MIN:
        return True, ""
    return True, "clipped" if clipped else "low_evidence"


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
