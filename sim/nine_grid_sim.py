# -*- coding: utf-8 -*-
"""数字宫格关卡仿真器（sim/nine_grid_sim.py）

定位：本模块只提供"世界 + 相机 + 带噪声动作"三件事，算法（视觉检测、GN
地图定位、FSM、容错）全部走 `levels/nine_grid.py` 的真实代码路径。断言在
`tests/test_nine_grid_sim.py`，本模块不依赖任何测试。

与 goodluck_sim 的仿真策略差异
----------------------------
- goodluck 的算法吃的是 AprilTag PnP 的位姿结果 → 模拟器重写 `solve_pnp`
  直接给"真值位姿 + 噪声"，用 matplotlib 画 2D 轨迹；
- 本关的算法吃的是**图像** → 模拟器必须重写 `capture_frame`，用真实内参/
  畸变把面板+黑字渲染成 2592×1944 合成帧，视觉链路才跑得起来。
两者共同约定：只重写 I/O 接缝（继承 RobotState），算法零改动。

仿真建模了什么 / 没建模什么
--------------------------
建模：场地几何（33.33cm 格）、面板与数字尺寸（字体渲染，随机 90°×k
朝向）、真实相机内参+径向畸变、画幅裁切、动作打滑（前进 ±8~15%、横移 ±20%、
大转 ±10%、**小转按现场实测值左 8.625°/右 5.200° 各 ±12%**——左右不对称是现场
真实特性，见 `levels/nine_grid_shared` 的单一真源）。
未建模：真机拍照耗时（2026-09-13 现场实测 fswebcam 2592x1944 -S 3 = 0.62s/张、
走代码路径 capture_frame() = 0.70s/张；**本仿真把帧当瞬时**）、舵机到位时间/抖动、
光照与白平衡对 HSV 的影响、相机-机体真实偏移（本模块 CAM_BODY_OFFSET=0）、
真实打滑分布、面板物理尺寸误差——这些必须在 P3 现场实测。

运行
----
    python -m sim.nine_grid_sim [--seed N] [--random-layout] [--quiet]
    python sim/nine_grid_sim.py            # 任意目录也可
退出码：0 = 7/7 且布局正确；1 = 否则（便于脚本/CI）。

仿真侧的保险丝（SIM_FUSE_*）
--------------------------
2026-09-25 起，**关卡不再因为"耗时太长"放弃任何一格**（顺序计分：跳一格等于
后面全丢；改成一格没确认就一直磨这一格）。这在真机上由人来叫停，在仿真里由
本模块的两条保险丝叫停：拍照数或动作数超过 `SIM_FUSE_CAPTURES/ACTIONS` 就抛
`SimFuseTripped`，`run_simulation` 捕获后照常返回 stats（`fuse_tripped` 写明
原因）。**这是仿真器的实验保护，不是关卡策略**——关卡自己的停手理由只剩离场
护栏一条。

集成测试：python tests/test_nine_grid_sim.py
"""

import argparse
import contextlib
import io
import os
import sys
from collections import Counter, namedtuple
# 允许直接 `python sim/nine_grid_sim.py` 运行（与 sim/goodluck_sim.py 一致）
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import cv2

from core.camera_config import (
    CAMERA_INTRINSIC, CAMERA_DISTORTION, HEAD_CENTER, SERVO_DEG_PER_US,
    CAM_PITCH_MOUNT_OFFSET_DEG,
)
from core.ground_homography import grid_cell_center
from core.robot_core import RobotState
from levels.nine_grid import NineGridLevel
from levels.nine_grid_shared import (
    BACK_ONE_STEP_CM, CAM_HEIGHT_CM, DISABLED_ACTIONS, FORWARD_ONE_STEP_CM,
    LEFT_MOVE_CM, NO_LAYOUT_ENV_VAR, RIGHT_MOVE_CM, TURN_LEFT_DEG,
    TURN_LEFT_SMALL_DEG, TURN_RIGHT_DEG, TURN_RIGHT_SMALL_DEG,
    action_name_cn, layout_enabled, print_layout,
)
from levels.nine_grid_three_stage import NineGridThreeStageLevel
from vision.nine_grid_detector import COLOR_TO_ID

# =====================================================================
# 场景常量（仿真的"真值"；与关卡假设相互独立，偏差即测试要抓的 bug）
# =====================================================================
FRAME_W, FRAME_H = 2592, 1944
PANEL_HALF_CM = 14.0        # 面板半边 14cm（33cm 格减缝）
# ---- 仿真侧保险丝（见模块头说明；关卡不再自己放弃任何一格） ----
# 取值：常规整局 ~200~600 张；留 5 倍余量，只拦"真的停不下来"的实验。
SIM_FUSE_CAPTURES = 3000
SIM_FUSE_ACTIONS = 3000


class SimFuseTripped(BaseException):
    """仿真保险丝跳闸（继承 BaseException：不被任何 `except Exception` 吞掉）

    语义上等同"操作员按下停止"：关卡策略本身没有这个停手理由（顺序计分下
    绝不放弃一格），是仿真这边为了不让一次实验无限跑下去而叫停。
    """
_IMAGE_RECT = np.array([[0.0, 0.0], [FRAME_W, 0.0],
                        [FRAME_W, FRAME_H], [0.0, FRAME_H]],
                       dtype=np.float32)

# 各颜色满足阈值的 HSV（与 tests/test_nine_grid_detector.py 一致）
COLOR_HSV = {
    "red": (3, 200, 200), "orange": (12, 200, 200),
    "yellow": (27, 180, 200), "green": (65, 200, 100),
    "blue": (107, 230, 130), "purple": (124, 150, 150),
    "pink": (165, 120, 150),
}
PANEL_BGR = {COLOR_TO_ID[c]: cv2.cvtColor(np.full((1, 1, 3), hsv, np.uint8),
                                          cv2.COLOR_HSV2BGR)[0, 0].tolist()
             for c, hsv in COLOR_HSV.items()}

# 缺省仿真布局：位置8空，其余 1..7（固定"随机"布局）
SIM_LAYOUT = {0: 5, 1: 2, 2: 7, 3: 1, 4: 4, 5: 6, 6: None, 7: 3, 8: None}

# =====================================================================
# 面板数字：**用字体画**（2026-09-25 用户裁定；替换掉"现场照片裁贴图"那版）
# =====================================================================
# 为什么不再用现场照片裁出来的字形贴图（旧实现，已删）：
#   1. 裁出来的墨迹**带着那张照片的透视斜切**——再投影到地面等于斜切两次；
#   2. 渲染时掩膜被强行塞进固定 8×12cm 方框 ⇒ 每个数字的宽高比被各自拉坏
#      （而 `digit_recognizer.normalize_mask` 明确把**宽高比**当判据）；
#   3. 裁切样本会混进面板边缘的阴影弧带（旧资产的 d4 根本不是数字，是弧带）；
#   4. 现场帧没拍到干净样本的数字只能凑合（旧资产里 d4 的 n_clean==0）。
# 字体渲染没有上述任何一个问题：字形**正立、无斜切、宽高比原生**，且 1..7
# 全部可画，不依赖任何外部资产。
#
# 朝向（用户 2026-09-25 裁定，保留）："现场字形朝向随机，并将仿真器的朝向也
# 作为随机的" ⇒ 按面板格位**确定性随机**转 90°×k（固定 seed ⇒ 同一局可复现）。
# ⚠️ 随机的是**旋转**（贴纸真的可能被转 90° 摆放），**不是镜像**——现场不可能
# 出现印反的数字（历史 bug：贴图角点序与世界 y 反向，见 `_glyph_quad`）。
GLYPH_FONT = cv2.FONT_HERSHEY_DUPLEX   # 粗笔画无衬线，接近现场印刷体
GLYPH_FONT_SCALE = 3.0
GLYPH_FONT_THICKNESS = 9               # ≈ 字高的 14%（现场笔画/字高实测 ~0.15）
GLYPH_ROTATE_SEED = 20260925     # 面板朝向的随机种子（固定 ⇒ 可复现）
# 数字墨迹的**高度**（正立时沿面板纵向）；宽度按字形自身宽高比。28cm 面板上
# 取 10.5cm 的依据：现场 44 个未裁切面板实测"墨迹高 / 色块高"中位 **0.38**
# ⇒ 0.38×28 ≈ 10.6cm（用同一个量法在仿真帧上量，字高 12cm 时该比值 0.41、
# 黑字占比 digit_evidence 0.106，而现场中位只有 0.071 ⇒ 12cm 偏大）。
# ⚠️ 这个尺寸不是"随便定的"：统一决策在**面板 4**上的到位判定对字的大小很敏感
# （实测字高 8/10.5/12cm ⇒ 整局拍照 726/809/1631 张，面板 4 尝试 1/2/5 次）。
DIGIT_HEIGHT_CM = 10.5
_GLYPH_CACHE = {}


def glyph_mask(digit):
    """数字 → 二值掩膜（白=墨迹，**紧裁到墨迹外框**，不归一化宽高比）

    紧裁是刻意的：宽高比是数字判据的一部分（"1" 就该是细高的），缩放交给
    投影那一端（`glyph_block_half_cm` 只钉高度，宽度跟字形走）。
    """
    g = _GLYPH_CACHE.get(int(digit))
    if g is not None:
        return g
    canvas = np.zeros((400, 400), np.uint8)
    txt = str(int(digit))
    (_tw, th), _base = cv2.getTextSize(txt, GLYPH_FONT, GLYPH_FONT_SCALE,
                                       GLYPH_FONT_THICKNESS)
    cv2.putText(canvas, txt, (40, 40 + th), GLYPH_FONT, GLYPH_FONT_SCALE,
                255, GLYPH_FONT_THICKNESS, cv2.LINE_AA)
    ys, xs = np.nonzero(canvas > 127)
    if len(xs) == 0:               # 字体不可用（理论到不了这里）
        return None
    g = (canvas[ys.min():ys.max() + 1, xs.min():xs.max() + 1] > 127
         ).astype(np.uint8)
    _GLYPH_CACHE[int(digit)] = g
    return g


def glyph_table():
    """{digit: 二值掩膜(白=墨迹)}（1..7；老接口名，测试与审计脚本都读它）"""
    table = {}
    for d in range(1, 8):
        m = glyph_mask(d)
        if m is not None:
            table[d] = m
    return table


def glyph_block_half_cm(digit, cell):
    """该面板数字块的世界**半尺寸** (hx, hy) cm

    高度恒为 `DIGIT_HEIGHT_CM`，宽度 = 高度 × 字形自身宽高比；转了 90°×k 之后
    宽高**互换**（物理上就是那张贴纸转了 90°）。数字块的世界尺寸因此随数字变化，
    不再把"1"撑成 8cm 宽的黑板砖。
    """
    g = glyph_mask(digit)
    asp = (g.shape[1] / float(g.shape[0])) if g is not None else 0.6
    h = DIGIT_HEIGHT_CM
    w = h * asp
    if glyph_rotate_k(cell) % 2:
        w, h = h, w
    return w / 2.0, h / 2.0


def glyph_rotate_k(cell):
    """面板数字的朝向：按格位确定性随机 90°×k（k=0..3）

    为什么随机而不是固定正立：现场字形朝向随机（用户裁定）。用**格位**而不是
    全局 RNG 决定，是为了不动动作噪声的随机流（改朝向不该改变打滑序列，否则
    仿真对照实验就不可比了）。
    """
    rng = np.random.RandomState(GLYPH_ROTATE_SEED + int(cell) * 7919)
    return int(rng.randint(0, 4))


def camera_rotation(bearing_deg, pitch_pulse,
                    pitch_offset_deg=CAM_PITCH_MOUNT_OFFSET_DEG):
    """世界->相机旋转矩阵（与 tests/test_ground_homography 同一构造，已验证）

    刻意保留本模块的独立副本：仿真是"世界的替身"，与生产投影实现
    （levels/nine_grid.camera_rotation）互为交叉验证，不共享代码。
    **但相机安装下俯偏移与高度必须与 camera_config 同源**（2026-09-11）：
    历史上 sim 用名义舵机角、关卡用名义角+安装偏移，渲染与投影不一致，
    端到端仿真必然失败（实测 pitch1200：27.0° vs 45.5°）。
    有效俯角 = 名义脉宽角 + pitch_offset_deg。
    """
    a = np.radians((1500 - pitch_pulse) * SERVO_DEG_PER_US
                   + pitch_offset_deg)
    f = np.radians(bearing_deg)
    sa, ca = np.sin(a), np.cos(a)
    sf, cf = np.sin(f), np.cos(f)
    r = np.array([cf, -sf, 0.0])
    d = np.array([-sa * sf, -sa * cf, -ca])
    v = np.array([ca * sf, ca * cf, -sa])
    return np.vstack([r, d, v])


def _project_quad_raw(R, C, cx, cy, hx, hy):
    """地面矩形（中心 (cx,cy)、半尺寸 (hx,hy)）→ 画幅内的可见多边形 | None

    ⚠️ 返回的是**裁到画幅**的多边形（真实相机只看到交集）。三个守卫都是踩过坑的：
      · 任一角点跑到相机平面附近（z ≤ 0.05）⇒ 投影溢出为 ±inf，`astype(int32)`
        得 INT_MIN，`cv2.fillPoly` 会扫描 ~2^31 行（实测 26s/帧，整轮仿真 800s+）；
      · `isfinite` 检查兜住上一条漏网的 inf/nan；
      · `clip(±1e6)` 再进 `intersectConvexConvex`。
    """
    corners = []
    for dx, dy in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)):
        p = np.array([cx + dx, cy + dy, 0.0])
        pc = R @ (p - C)
        if pc[2] <= 0.05:
            return None
        corners.append(pc)
    norm = np.array([c[:2] / c[2] for c in corners])
    pix, _ = cv2.projectPoints(
        np.column_stack([norm, np.ones(4)]).reshape(-1, 1, 3).astype(np.float32),
        np.zeros(3), np.zeros(3), CAMERA_INTRINSIC, CAMERA_DISTORTION)
    pix = pix[:, 0, :]
    if not np.all(np.isfinite(pix)):
        return None
    pix = np.clip(pix, -1e6, 1e6)
    # ⚠️ 返回顺序是 (面积:float, 交点:ndarray(N,1,2))——不是 (points, area)
    area, inter = cv2.intersectConvexConvex(pix.astype(np.float32), _IMAGE_RECT)
    if inter is None or area <= 1e-9:
        return None
    pts = np.asarray(inter, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 3:
        return None
    return pts


def _project_quad(R, C, cx, cy, hx, hy):
    """同 `_project_quad_raw`，但返回**取整后的 int32** 多边形（供 fillPoly）"""
    poly = _project_quad_raw(R, C, cx, cy, hx, hy)
    return None if poly is None else np.round(poly).astype(np.int32)


def _glyph_quad(R, C, cx, cy, digit, cell):
    """数字块的**四角点**（原生 px）→ (4,2) float32 或 None

    与 `_project_quad` 同一套投影/守卫，但返回的是**未裁剪的四角点**（裁剪后的
    多边形不能用来做透视变换——那会把字形拉伸到错误的位置）。
    任一角点不合法（跑到相机平面附近/画幅外很远）就返回 None：该面板只画色块。

    ★ 角点顺序 = **掩膜的栅格序**（TL,TR,BR,BL），且 TL 必须是"站在入口看这块
    面板时的左上角"：数字的上沿在**远处**（世界 +y）、左边在世界 −x。
    历史 bug（2026-09-25 体检发现）：这里曾经按 [(-x,-y), (+x,-y), (+x,+y),
    (-x,+y)] 给点，即掩膜 TL 落到**近边**（图像左下）⇒ 每个数字都被画成
    **上下镜像**（"7" 看起来像 "L"、"5" 像 "2"），而现场不可能出现印反的数字。
    因此顺序写成 [(-x,+y), (+x,+y), (+x,-y), (-x,-y)]，并有单测钉住。
    """
    hx, hy = glyph_block_half_cm(digit, cell)
    pts = []
    for dx, dy in ((-hx, hy), (hx, hy), (hx, -hy), (-hx, -hy)):
        pc = R @ (np.array([cx + dx, cy + dy, 0.0]) - C)
        if pc[2] <= 0.05:
            return None
        pts.append(pc[:2] / pc[2])
    pix, _ = cv2.projectPoints(
        np.column_stack([pts, np.ones(4)]).reshape(-1, 1, 3).astype(np.float32),
        np.zeros(3), np.zeros(3), CAMERA_INTRINSIC, CAMERA_DISTORTION)
    pix = pix[:, 0, :]
    if not np.all(np.isfinite(pix)):
        return None
    if np.max(np.abs(pix)) > 1e6:
        return None
    # 面板/数字块**完全在画幅外**时不必画（省一次 warp；裁切仍由 warp 的
    # 目标画幅自然处理）
    if (pix[:, 0].max() < 0 or pix[:, 0].min() > FRAME_W
            or pix[:, 1].max() < 0 or pix[:, 1].min() > FRAME_H):
        return None
    return pix.astype(np.float32)


def _draw_glyph(frame, R, C, cell, digit, cx, cy):
    """在 already-drawn 的色块上画**数字**（黑字，字体渲染）

    实现 = 把字形掩膜（按格位朝向转 90°×k）透视变换到数字块四角点，再把
    "掩膜为真"的像素涂黑。**没有逐像素 Python 循环**（那是 26s/帧那条退化
    路径的同类写法）。
    """
    g = glyph_mask(digit)
    if g is None:
        return False
    quad = _glyph_quad(R, C, cx, cy, digit, cell)
    if quad is None:
        return False
    k = glyph_rotate_k(cell)
    if k:
        g = np.ascontiguousarray(np.rot90(g, k))
    src = np.array([[0.0, 0.0], [g.shape[1] - 1.0, 0.0],
                    [g.shape[1] - 1.0, g.shape[0] - 1.0],
                    [0.0, g.shape[0] - 1.0]], dtype=np.float32)
    # ⚠️ warp 的目标必须**裁到数字块的外接框**，不能整幅 2592×1944：
    # 整幅 warp 实测 0.090s/帧（渲染守卫线是 0.1s/帧），只差 10% 就踩线。
    x0 = int(np.floor(quad[:, 0].min())) - 2
    y0 = int(np.floor(quad[:, 1].min())) - 2
    x1 = int(np.ceil(quad[:, 0].max())) + 2
    y1 = int(np.ceil(quad[:, 1].max())) + 2
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(FRAME_W, x1), min(FRAME_H, y1)
    if x1c <= x0c or y1c <= y0c:
        return False                       # 数字块完全在画幅外
    Hm = cv2.getPerspectiveTransform(src, quad - np.array([x0c, y0c],
                                                          dtype=np.float32))
    warped = cv2.warpPerspective(g * 255, Hm, (x1c - x0c, y1c - y0c),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    view = frame[y0c:y1c, x0c:x1c]
    view[warped > 127] = (0, 0, 0)
    return True


def _project_quad_full(R, C, cx, cy, hx, hy):
    """地面矩形 → 投影后的**四角点**（未裁到画幅）+ 多边形面积

    与 `_glyph_quad` 同一套守卫。返回 (4,2) float32 或 None。
    用途：`_frame_truth` 要算"可见面积 **占该面板本帧总投影面积** 的比例"——
    分母必须是**同一投影下的未裁切面积**，不能拿物理 cm² 去比（踩过：264.6）。
    """
    pts = []
    for dx, dy in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)):
        pc = R @ (np.array([cx + dx, cy + dy, 0.0]) - C)
        if pc[2] <= 0.05:
            return None
        pts.append(pc[:2] / pc[2])
    pix, _ = cv2.projectPoints(
        np.column_stack([pts, np.ones(4)]).reshape(-1, 1, 3).astype(np.float32),
        np.zeros(3), np.zeros(3), CAMERA_INTRINSIC, CAMERA_DISTORTION)
    pix = pix[:, 0, :]
    if not np.all(np.isfinite(pix)) or np.max(np.abs(pix)) > 1e6:
        return None
    return pix.astype(np.float32)


def _frame_truth(R, C, layout):
    """本帧的**面板真值**（供"数字判据在仿真里准不准"这类评测用）

    每项：{digit, cell, cx, cy, bbox(x,y,w,h 原生 px), visible_frac, clipped}
      · cx,cy   = **色块中心**的投影（未裁切时等于对角线交点；裁切时会被裁边拉偏，
                  所以同时给 `clipped` 与 `bbox` 让调用方自行取舍）
      · visible_frac = 可见面积 / 未裁切面积 ⇒ 用来判"是不是被别的东西挡住了"，
                  也用来排除"只露一条边"的观测
      · occluded_by  = 盖在它上面的数字（`面板6` 在部分种子下被别格面板压住）

    ⚠️ 这是**真值**，不是检测结果：它由渲染时同一套 `_project_quad_raw` 算出，
    因此与画出来的像素逐点对应（错也错得一致，不会自证）。
    """
    out = []
    polys = {}
    for cell, digit in layout.items():
        if digit is None:
            continue
        cx, cy = grid_cell_center(cell)
        quads = {}
        for name, (hx, hy) in (("full", (PANEL_HALF_CM, PANEL_HALF_CM)),
                               ("digit", glyph_block_half_cm(digit, cell))):
            poly = _project_quad_raw(R, C, cx, cy, hx, hy)
            if poly is None:
                continue
            xs, ys = poly[:, 0], poly[:, 1]
            # 分母 = **同一投影下的未裁切面积**（不是物理 cm²！）
            full = _project_quad_full(R, C, cx, cy, hx, hy)
            total = (float(cv2.contourArea(full)) if full is not None else 0.0)
            quads[name] = {
                "poly": poly,
                "bbox": (int(xs.min()), int(ys.min()),
                         int(xs.max() - xs.min()), int(ys.max() - ys.min())),
                "area": float(cv2.contourArea(poly)),
                "total_area": total,
            }
        if "full" not in quads:
            continue
        polys[int(cell)] = quads
    for cell, quads in polys.items():
        digit = int(layout[cell])
        q = quads["full"]
        cx, cy = grid_cell_center(cell)
        # 被同帧其它面板（更近的那些）盖住的比例：投影后色块多边形的交叠
        occl = 0.0
        occluders = []
        for other, qo in polys.items():
            if other == cell:
                continue
            # ⚠️ (面积, 交点)，见 _project_quad_raw 的注释
            ia, ipts = cv2.intersectConvexConvex(
                q["poly"], qo["full"]["poly"])
            if ipts is not None and ia > 0.02 * max(q["area"], 1.0):
                occl += float(ia)
                occluders.append(int(layout[other]))
        cx_px, cy_px = (q["bbox"][0] + q["bbox"][2] / 2.0,
                        q["bbox"][1] + q["bbox"][3] / 2.0)
        out.append({
            "digit": digit, "cell": int(cell),
            "cx": float(cx_px), "cy": float(cy_px),
            "bbox": q["bbox"],
            "visible_frac": (q["area"] / q["total_area"]
                             if q["total_area"] > 0 else 0.0),
            "occluded_frac": min(1.0, occl / max(q["area"], 1.0)),
            "occluded_by": sorted(occluders),
            "clipped": bool(q["area"] < 0.999 * max(q["total_area"], 1e-9)),
            "digit_bbox": (quads["digit"]["bbox"] if "digit" in quads else None),
            # 世界系真值（诊断用；真机没有）
            "world_xy": (float(cx), float(cy)),
        })
    return sorted(out, key=lambda e: e["digit"])


class SimNineGridRobot(RobotState):
    """九宫格仿真机器人：场地系位姿 + 打滑噪声运动 + 合成相机"""

    CAM_HEIGHT = CAM_HEIGHT_CM   # 与关卡同源（levels.nine_grid_shared 的实测值）
    CAM_BODY_OFFSET = 0.0  # 仿真中相机即机体中心（关卡常量另算，端到端容差内）

    def __init__(self, layout=SIM_LAYOUT, seed=3, viewer=None, deform=None,
                 fuse_captures=None, fuse_actions=None):
        super().__init__(tag_poses={})
        self.pos = np.array([50.0, -20.0])   # 入口外居中
        self.heading = 0.0                   # 场地系 bearing（度，右正）
        self.pitch = 1500
        self.head = HEAD_CENTER
        self.layout = layout                 # cell -> digit
        self.rng = np.random.RandomState(seed)
        self.n_captures = 0
        # 仿真侧保险丝（见模块头）：调用方可调小，好在测试里快速跑完"磨不下来"的
        # 场景；None = 用默认值。
        self.fuse_captures = (SIM_FUSE_CAPTURES if fuse_captures is None
                              else int(fuse_captures))
        self.fuse_actions = (SIM_FUSE_ACTIONS if fuse_actions is None
                             else int(fuse_actions))
        # 单张拍照耗时（秒）：告诉关卡"这个环境下拍照有多贵"，供它把"拍了多少张"
        # 折算成时间估计（见 levels/nine_grid_shared.py 的 CAPTURE_COST_S）。
        # 真机 ≈0.7s/张（2026-09-13 实测）；仿真里帧是瞬时的，取极小值 ⇒ 仿真
        # 打出来的"按单张折算"看着很小，这是符合事实的（回归数字也因此可比）。
        self.capture_cost_s = 0.001
        self.action_log = []
        # 可选图形化 viewer（sim/nine_grid_view.NineGridView）；None = 纯无头
        self.viewer = viewer
        # 地板形变（2026-09-11 用户约束：本关地板会形变，机体俯仰按 ±15° 考虑）：
        # 只改**世界侧**的相机姿态/高度，关卡内常数保持不动 → 精确复现"常数失配"。
        self.deform_tilt_deg = 0.0
        self.deform_height_cm = 0.0
        self.deform_actions = 0
        # 当前目标数字（由 sim/level 侧每格开始前写入；供"按格触发阶跃"使用）
        self.current_digit = None
        self._step_done_digits = set()
        self._deform = dict(deform) if deform else None
        self._deform_rng = np.random.RandomState(int(seed) + 977)

    # ---- I/O 接缝 ----

    def set_head(self, pulse, move_time_ms=500):
        self.head = pulse
        self.current_head_pulse = pulse  # 与真机 RobotState.set_head 行为一致

    def set_deform_digit(self, digit):
        """关卡开新格时通知 sim 当前目标数字（仅供"按格触发阶跃"使用）

        真机 RobotState 没有这个方法 → 关卡侧用 getattr 兜底调用，不影响真机。
        """
        self.current_digit = int(digit)

    def set_pitch(self, pulse, move_time_ms=500):
        self.pitch = pulse

    def run_action(self, name, times=1):
        for _ in range(max(1, times)):
            self._apply_action(name)
            self._update_deform()
        self.action_log.append((name, times))
        self._check_fuse()
        if self.viewer is not None:
            self.viewer.on_action(self, name, times)

    def _check_fuse(self):
        """仿真侧的保险丝（**不是**关卡策略，见模块头 `SIM_FUSE_*` 说明）"""
        if self.n_captures > self.fuse_captures:
            raise SimFuseTripped(
                f"仿真保险丝：拍照 {self.n_captures} 张 > {self.fuse_captures}"
                "（关卡本身不再放弃任何一格，是仿真这边停的实验）")
        if len(self.action_log) > self.fuse_actions:
            raise SimFuseTripped(
                f"仿真保险丝：动作 {len(self.action_log)} 次 > {self.fuse_actions}"
                "（关卡本身不再放弃任何一格，是仿真这边停的实验）")

    def _update_deform(self):
        """地板形变模型：每动作一次随机游走 + 可选一次性阶跃

        deform 配置（run_simulation 透传）：
          sigma_tilt_deg / sigma_h_cm  每动作随机游走步长（默认 0 = 无形变）
          max_tilt_deg  / max_h_cm     形变幅度上限（默认 15° / 3cm）
          step_after_actions           第 N 个动作后施加一次性阶跃
          step_at_digit                第 N 格（目标数字）开始时施加一次性阶跃
          step_tilt_deg / step_h_cm    阶跃幅度
        注意：形变只在**动作**时变化 —— 静止扫头部（布局扫）期间保持不变，
        与现场"站定扫描时地板不再继续形变"一致。
        **为什么要有 step_at_digit（2026-09-13 新增）**：`step_after_actions` 与
        动作流强耦合——任何改变动作数的代码改动都会把阶跃触发点挪到另一个
        格子，于是同一个"回归测试"在两次改动之间测的根本不是同一个场景
        （实测：HEAD 244/191 过、本轮 206 张不过，部分原因就在这里）。
        按格触发让"第几格踩上形变"成为**场景定义的一部分**，与被测代码的
        动作数解耦；旧参数保留（用于复现历史日志），但新回归一律用 step_at_digit。
        """
        cfg = self._deform
        if not cfg:
            return
        self.deform_actions += 1
        if cfg.get("sigma_tilt_deg"):
            self.deform_tilt_deg += float(
                self._deform_rng.normal(0.0, float(cfg["sigma_tilt_deg"])))
        if cfg.get("sigma_h_cm"):
            self.deform_height_cm += float(
                self._deform_rng.normal(0.0, float(cfg["sigma_h_cm"])))
        if cfg.get("step_after_actions") \
                and self.deform_actions == int(cfg["step_after_actions"]):
            self.deform_tilt_deg += float(cfg.get("step_tilt_deg", 0.0))
            self.deform_height_cm += float(cfg.get("step_h_cm", 0.0))
        # 按格触发（与被测代码的动作数解耦，见上文说明）：关卡每开一格会调用
        # set_deform_digit(K)，第 K 格开始后的**第一个动作**上施加阶跃。
        k = cfg.get("step_at_digit")
        if k is not None and self.current_digit == int(k) \
                and int(k) not in self._step_done_digits:
            self._step_done_digits.add(int(k))
            self.deform_tilt_deg += float(cfg.get("step_tilt_deg", 0.0))
            self.deform_height_cm += float(cfg.get("step_h_cm", 0.0))
        mt = float(cfg.get("max_tilt_deg", 15.0))
        mh = float(cfg.get("max_h_cm", 3.0))
        self.deform_tilt_deg = float(np.clip(self.deform_tilt_deg, -mt, mt))
        self.deform_height_cm = float(np.clip(self.deform_height_cm, -mh, mh))

    def _apply_action(self, name):
        th = np.radians(self.heading)
        fwd = np.array([np.sin(th), np.cos(th)])
        right = np.array([np.cos(th), -np.sin(th)])
        slip = self.rng.normal
        # ★ 所有平移/转向量都**导入**自 `levels/nine_grid_shared`（现场实测值，
        # 单一真源），不再手抄名义值——历史 bug：仿真抄的是 2.0/1.9/2.0°，而成
        # 品真机是 2.652/2.497/8.625|5.200 ⇒ "仿真跑的是一个不存在的机器人"。
        # 打滑系数取 `tools/ab_ninegrid.REAL` 那套"现场实测原语"的同一组数
        # （平移 6%、后退 15%、大转 10%、小转 12%）：现场 20 步/10 步的**组间**
        # 极差折合 1.7%~4.5%，平移取 6% 略保守（只有总量数据，无法分辨每步抖动）。
        # bearing 约定右正（见 core/ground_homography）：左转 = heading 减小
        if name == "go_forward":
            # 本关卡禁用的大步幅动作（见 DISABLED_ACTIONS）；量级沿用 5cm 旧值，
            # 现场未实测——关卡永远不会下发它，只在"白名单硬门"测试里用到。
            self.pos += fwd * 5.0 * (1 + slip(0, 0.08))
        elif name == "go_forward_one_step":
            self.pos += fwd * FORWARD_ONE_STEP_CM * (1 + slip(0, 0.06))
        elif name == "back_one_step":
            self.pos -= fwd * BACK_ONE_STEP_CM * (1 + slip(0, 0.15))
        elif name == "left_move":
            self.pos -= right * LEFT_MOVE_CM * (1 + slip(0, 0.06))
        elif name == "right_move":
            self.pos += right * RIGHT_MOVE_CM * (1 + slip(0, 0.06))
        elif name == "turn_left":
            self.heading -= TURN_LEFT_DEG * (1 + slip(0, 0.10))
        elif name == "turn_right":
            self.heading += TURN_RIGHT_DEG * (1 + slip(0, 0.10))
        elif name == "turn_left_small_step":
            # 现场实测（同一真源）：左手两套舵机/连杆与右手不同，左右差 66% 是
            # 真实特性。旧模型 `max(0, N(2.0,1.5))` 还额外造出 9% 的"这一步原地
            # 不动"，而现场 20 步极差只有 25°/2°——没有"被地面吞掉"这回事。
            self.heading -= TURN_LEFT_SMALL_DEG * (1 + slip(0, 0.12))
        elif name == "turn_right_small_step":
            self.heading += TURN_RIGHT_SMALL_DEG * (1 + slip(0, 0.12))
        elif name == "stand":
            pass
        else:
            raise ValueError(f"仿真未实现动作: {name}")

    def capture_frame(self):
        self.n_captures += 1
        self._check_fuse()
        # 头部左转(脉宽>1500)为正 → 相机方位角减小（与真机一致）
        bearing = self.heading - (self.head - HEAD_CENTER) * SERVO_DEG_PER_US
        # 形变只加到"世界侧"：相机安装偏移 + 地板形变倾角、站立高度 + 形变高度差
        R = camera_rotation(bearing, self.pitch,
                            CAM_PITCH_MOUNT_OFFSET_DEG + self.deform_tilt_deg)
        C = np.array([self.pos[0], self.pos[1],
                      self.CAM_HEIGHT + self.deform_height_cm])

        frame = np.full((FRAME_H, FRAME_W, 3), 90, np.uint8)
        for cell, digit in self.layout.items():
            if digit is None:
                continue
            self._draw_panel(frame, R, C, cell, digit)
        # 本帧真值（面板级）——挂在 state 上，供评测工具/测试读取。
        # 关卡侧**不消费**它（真机没有这个属性），故不影响任何决策路径。
        self._last_frame_truth = _frame_truth(R, C, self.layout)
        if self.viewer is not None:
            self.viewer.on_frame(self, frame)
        return frame

    def _draw_panel(self, frame, R, C, cell, digit):
        cx, cy = grid_cell_center(cell)
        panel = _project_quad(R, C, cx, cy, PANEL_HALF_CM, PANEL_HALF_CM)
        if panel is not None:
            cv2.fillPoly(frame, [np.round(panel).astype(np.int32)],
                         PANEL_BGR[digit])
        # 数字：字体渲染（见文件头"面板数字"）。字体画不出来（理论上不会）时
        # 退回黑块——宁可画个块，也不让面板看起来"没有数字"。
        if not _draw_glyph(frame, R, C, cell, digit, cx, cy):
            block = _project_quad(R, C, cx, cy,
                                  *glyph_block_half_cm(digit, cell))
            if block is not None:
                cv2.fillPoly(frame, [np.round(block).astype(np.int32)], (0, 0, 0))


# =====================================================================
# 运行入口（CLI 与测试共用）
# =====================================================================

SimRun = namedtuple("SimRun", "robot level stats")


def random_layout(seed):
    """由 seed 生成合法随机布局：7 个数字占 9 格中的 7 格（留 2 个空位）"""
    rng = np.random.RandomState(seed)
    cells = [c for c in range(9) if c != 6]
    chosen = list(rng.permutation(cells)[:7])
    digits = list(rng.permutation(range(1, 8)))
    layout = {c: None for c in range(9)}
    for cell, digit in zip(chosen, digits):
        layout[int(cell)] = int(digit)
    return layout


def run_simulation(layout=SIM_LAYOUT, seed=3, quiet=False, viewer=None,
                   deform=None, three_stage=False,
                   fuse_captures=None, fuse_actions=None, no_layout=False):
    """跑一遍完整关卡（布局扫→1..7），不做断言；返回 SimRun(robot, level, stats)

    两种决策办法各是一个模块，这里按 `three_stage` 选：
      · False（默认）= levels/nine_grid.py 统一决策（三档分区 + 一个循环）；
      · True         = levels/nine_grid_three_stage.py 三段式（现场发货的稳定版）。

    no_layout=True：短路掉全部布局流程（置 NINEGRID_NO_LAYOUT 环境变量，关卡侧
    由 nine_grid_shared.layout_enabled() 统一判定）。用于回归"关掉布局还能不能
    逐格走完"，此时 `layout_ok` 无意义（digit_cell 必然为空）。

    stats 键：decision / ok_all / results / layout_ok / digit_cell / truth /
              captures / actions / action_counts / small_turn_deg /
              small_turn_usable / banned_used / cell_trips / attempt_log /
              fuse_tripped / panel_landing / deform_tilt_deg / deform_height_cm
    quiet=True 时吞掉关卡逐行日志（只留返回值供调用方打印摘要）。
    viewer：可选图形化 viewer（需有 attach(robot, level) 与 on_action/on_frame）；
            传入后由 viewer 决定节奏（暂停/单步），None = 纯无头。
    deform：地板形变注入（见 SimNineGridRobot._update_deform）；None = 无形变。
    保险丝：拍照/动作超过 SIM_FUSE_*（可用 fuse_captures/fuse_actions 调小）时抛
            SimFuseTripped（仿真侧保护，不是关卡策略）；这里捕获并把原因写进
            stats["fuse_tripped"]，照常返回结果。
    """
    from levels.nine_grid_shared import NO_LAYOUT_ENV_VAR, layout_enabled
    no_layout = bool(no_layout) or not layout_enabled()
    if no_layout:
        # 常量为 False 时环境变量写不写都关着；no_layout=True 时补上环境变量，
        # 好让日志/子进程看到同一个开关（环境变量是"只减不增"的临时关闭位）。
        os.environ.setdefault(NO_LAYOUT_ENV_VAR, "1")
    robot = SimNineGridRobot(layout=layout, seed=seed, viewer=viewer,
                             deform=deform, fuse_captures=fuse_captures,
                             fuse_actions=fuse_actions)
    level_cls = NineGridThreeStageLevel if three_stage else NineGridLevel
    level = level_cls(robot)
    if viewer is not None:
        viewer.attach(robot, level)
    fuse = None
    try:
        if quiet:
            with contextlib.redirect_stdout(io.StringIO()):
                ok_all = level.run_level()
        else:
            ok_all = level.run_level()
    except SimFuseTripped as e:
        # 关卡"绝不放弃一格"⇒ 可能一直磨下去；这里是仿真自己的刹车。
        fuse = str(e)
        ok_all = False
        print(f"[仿真] {fuse}")

    truth = {d: c for c, d in layout.items() if d is not None}
    used = {a for a, _ in robot.action_log}
    layout_ran = layout_enabled()      # 跑完再问一次：布局流程到底跑没跑
    stats = {
        "decision": "三段式" if three_stage else "统一决策",
        "ok_all": bool(ok_all),
        "results": list(getattr(level, "results", [])),
        # 短路布局时 digit_cell 必然为空 ⇒ layout_ok 无意义（用 layout_ran 区分）
        "layout_ok": level.digit_cell == truth,
        "layout_ran": layout_ran,
        "digit_cell": dict(level.digit_cell),
        "truth": truth,
        "captures": robot.n_captures,
        "actions": len(robot.action_log),
        "action_counts": dict(Counter(a for a, _ in robot.action_log)),
        "small_turn_deg": level._small_turn_deg,
        "small_turn_usable": level._small_turn_usable,
        "banned_used": sorted(used & set(DISABLED_ACTIONS)),
        "cell_trips": dict(level.cell_trips),
        "attempt_log": dict(getattr(level, "attempt_log", {})),
        "fuse_tripped": fuse,
        # 逐格落点：{digit: [离格心 cm, 来源]}；来源应为 "真值"（仿真有 state.pos）
        "panel_landing": {d: (None if v is None else [round(v[0], 2), v[1]])
                          for d, v in level.panel_landing.items()},
        "deform_tilt_deg": robot.deform_tilt_deg,
        "deform_height_cm": robot.deform_height_cm,
    }
    return SimRun(robot, level, stats)


def _action_counts_cn(counts):
    """{动作名: 次数} → "左小转 12 次、前进一步 40 次"（按次数降序，日志用）"""
    if not counts:
        return "（无）"
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return "、".join(action_name_cn(a, n) for a, n in items)


def _run_ok(stats):
    """退出码判据：7/7 到位，（布局流程跑过时）还要求布局识别正确

    短路布局时 `digit_cell` 必然为空、`layout_ok` 恒 False，那不是失败——
    这种情况只看到位情况（`ok_all`）。
    """
    return bool(stats["ok_all"] and
                (stats["layout_ok"] or not stats.get("layout_ran", True)))


def _print_summary(stats):
    ok_n = sum(1 for _, ok in stats["results"] if ok)
    print(f"[仿真] 决策方式: {stats.get('decision', '?')}")
    if stats.get("layout_ran", True):
        print(f"[仿真] 布局识别: {'成功' if stats['layout_ok'] else '失败'}"
              f"｜数字→格位: {stats['digit_cell']}")
        if stats.get("layout_ok"):
            # 给人看的形式：远排在上、0 = 该格没有数字（见 format_layout）
            print("[仿真] 数字→格位图（远排在上一行、0 = 该格没有数字）：")
            print_layout(stats["digit_cell"], prefix="[仿真]   ")
    else:
        print("[仿真] 布局流程: **已短路**（不扫场/不拟合/不反推位姿）"
              "｜逐格导航全靠视觉")
    print(f"[仿真] 到位: {ok_n}/{len(stats['results'])}"
          f"｜拍照 {stats['captures']} 张｜动作 {stats['actions']} 次")
    print(f"[仿真] 动作统计: {_action_counts_cn(stats['action_counts'])}")
    print(f"[仿真] 单步小转角估计: {stats['small_turn_deg']:.1f}°/次"
          f"（{'可用' if stats['small_turn_usable'] else '不可用，改用大角度转向'}）")
    if stats["banned_used"]:
        print(f"[仿真] 警告: 出现禁用动作 {stats['banned_used']}")
    if stats.get("fuse_tripped"):
        print(f"[仿真] ⚠ 保险丝跳闸：{stats['fuse_tripped']}")
    if stats.get("attempt_log"):
        print(f"[仿真] 逐格尝试次数: {stats['attempt_log']}"
              "（>1 = 那一格磨了不止一次；顺序计分下这是正常代价）")
    if stats.get("deform_tilt_deg") or stats.get("deform_height_cm"):
        print(f"[仿真] 结束时的地板形变: 俯仰 {stats['deform_tilt_deg']:+.1f}°"
              f"｜高度 {stats['deform_height_cm']:+.1f}cm")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="数字宫格关卡仿真（**默认开窗**；--no-ui 只跑无头）")
    ap.add_argument("--seed", type=int, default=3,
                    help="动作噪声/随机布局种子（默认 3）")
    ap.add_argument("--random-layout", action="store_true",
                    help="由 seed 生成合法随机布局（默认用固定 SIM_LAYOUT）")
    ap.add_argument("--quiet", action="store_true",
                    help="不打印关卡逐行日志，只打印摘要")
    ap.add_argument("--deform", type=float, default=0.0,
                    metavar="SIGMA_TILT_DEG",
                    help="地板形变：每动作俯仰随机游走步长（度），"
                         "幅度上限 ±15°、高度 ±2cm（0=无形变）")
    ap.add_argument("--three-stage", action="store_true",
                    help="跑**三段式**（现场发货的稳定实现，"
                         "levels/nine_grid_three_stage.py）。不加则跑统一决策"
                         "（三档分区＋一个循环，levels/nine_grid.py）。")
    ap.add_argument("--no-layout", action="store_true",
                    help="**短路掉全部布局流程**（等价 NINEGRID_NO_LAYOUT=1）："
                         "不扫场/不拟合格阵/不自标定/不反推位姿，逐格导航全靠"
                         "视觉。用于回归'布局定不出来时这一关还跑不跑得起来'。"
                         "此时逐格正确性看 ok_all，不看 layout_ok。")
    ap.add_argument("--no-ui", action="store_true",
                    help="不开窗（脚本 / CI 用）；默认开窗可视化")
    ap.add_argument("--delay", type=int, default=30,
                    help="开窗时每帧等待 ms（0=最快，映射为 waitKey(1)）")
    ap.add_argument("--detect", action="store_true",
                    help="开窗时也跑检测器叠加（慢，整轮约 +25s）")
    args = ap.parse_args(argv)

    # ★ 短路布局流程（CLI 或环境变量）：必须在创建关卡前置好环境变量——关卡侧
    #   在 run_level() 里读它（nine_grid_shared.layout_enabled()）。
    if args.no_layout:
        os.environ.setdefault(NO_LAYOUT_ENV_VAR, "1")
    no_layout = not layout_enabled()
    print(f"[仿真] 决策方式: {'三段式' if args.three_stage else '统一决策'}"
          + ("｜布局流程: **已短路**（不扫场/不拟合/不反推位姿）"
             if no_layout else "｜布局流程: 开启"))
    layout = random_layout(args.seed) if args.random_layout else SIM_LAYOUT
    print(f"[仿真] 布局: {layout}")
    deform = None
    if args.deform:
        deform = {"sigma_tilt_deg": args.deform, "sigma_h_cm": 0.5,
                  "max_tilt_deg": 15.0, "max_h_cm": 2.0}
        print(f"[仿真] 注入地板形变: 每次动作俯仰随机游走 "
              f"{deform['sigma_tilt_deg']:.1f}°、高度随机游走 "
              f"{deform['sigma_h_cm']:.1f}cm（俯仰上限 ±"
              f"{deform['max_tilt_deg']:.0f}°、高度上限 ±"
              f"{deform['max_h_cm']:.0f}cm）")

    show_window = not args.no_ui
    if show_window:
        try:
            from sim.nine_grid_view import (
                NineGridView, ViewerQuit, ViewerRestart, _gui_available, HELP,
            )
        except Exception as e:            # 缺 cv2 高层面板等
            print(f"[仿真] 图形界面不可用（{type(e).__name__}: {e}）"
                  "——改为无窗口运行")
            show_window = False
        else:
            if not _gui_available():
                print("[仿真] 未检测到图形界面（DISPLAY 不可用？）"
                      "——改为无窗口运行")
                print("[仿真] 远程 SSH 场景请加 --no-ui；也可参考 "
                      "tools/camera_preview.py --stream 的网页流方案")
                show_window = False

    if not show_window:
        try:
            run = run_simulation(layout=layout, seed=args.seed,
                                 quiet=args.quiet, deform=deform,
                                 three_stage=args.three_stage,
                                 no_layout=no_layout)
        except Exception as e:            # 布局扫失败等：CLI 友好退出
            print(f"[仿真] 关卡异常: {type(e).__name__}: {e}")
            return 1
        _print_summary(run.stats)
        return 0 if _run_ok(run.stats) else 1

    # ---- 开窗模式：按键 与 换 seed 重开 ----
    print("[仿真] 按键：SPACE 暂停／继续｜S 单步放行一帧｜R 换 seed 重开"
          "｜Q 或 ESC 退出｜D 检出框叠加开关｜+／- 调慢／调快｜H 本说明")
    print(f"[仿真] 按键说明: {HELP}")
    seed = args.seed
    while True:
        layout = random_layout(seed) if args.random_layout else SIM_LAYOUT
        viewer = NineGridView(delay_ms=args.delay, show_detect=args.detect)
        try:
            run = run_simulation(layout=layout, seed=seed, viewer=viewer,
                                 quiet=args.quiet, deform=deform,
                                 three_stage=args.three_stage,
                                 no_layout=no_layout)
        except ViewerRestart:
            viewer.close()
            seed += 1
            print(f"[仿真] 重新开始，seed 改为 {seed}")
            continue
        except ViewerQuit:
            viewer.close()
            print("[仿真] 用户已退出")
            return 0
        try:
            viewer.finish(run.stats)
            viewer.close()
            return 0 if _run_ok(run.stats) else 1
        except ViewerRestart:
            viewer.close()
            seed += 1
            print(f"[仿真] 重新开始，seed 改为 {seed}")
        except ViewerQuit:
            viewer.close()
            print("[仿真] 用户已退出")
            return 0


if __name__ == "__main__":
    sys.exit(main())
