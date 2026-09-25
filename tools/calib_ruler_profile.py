# -*- coding: utf-8 -*-
"""标尺刻度 → 距离公式标定辅助（tools/calib_ruler_profile.py）

为什么需要它
------------
把一张"地上摆着卷尺"的照片变成「像素 ↔ 地面距离」公式，有两条路：

- **凭眼睛在图上画整条横线**（上一版做法）。精度不够，而且**极易把配对搞反**。
  2026-09-25 复核发现：标尺帧上明明标着"70 在最上、2 在最下"，
  分析脚本却把最上面那条当成了 2cm；而当初用来"确认配对"的交比检验对
  反转不变，根本区分不了正反。两者相差 6 倍（349.6px vs 58.7px）。
- **在图上精确点击标尺刻度**（本工具）。每个刻度点一次，带缩放与十字线，
  误差可比画线小一个量级；距离值由点击顺序决定，结构上不可能配反。

本工具还顺手做三件事：
  1. 拟合后直接用**厘米**报残差（不是只报像素），因为最终要的是距离精度；
  2. 同时给"自由射影模型"和"物理相机模型（含畸变）"的结果，看得出畸变
     到底有没有帮忙；
  3. 检查 y 随距离的走向，走向反了会明确报警。

用法
----
    # 交互点击（推荐）：按提示顺序点标尺刻度
    python tools/calib_ruler_profile.py --image photo.jpg

    # 只关心近段（起跨点要的是近距零点精度）
    python tools/calib_ruler_profile.py --image photo.jpg --dists 2,3,4,5,6,8,10,15,20

    # 卷尺零点不在光心地面投影点上时，用 --offset 补（正=卷尺零点在原点前方）
    python tools/calib_ruler_profile.py --image photo.jpg --dists 5,10,15,20 --offset 12

    # 已有像素点，只重算拟合
    python tools/calib_ruler_profile.py --image photo.jpg --px "1200,1819;1210,1401" \
        --dists 2,10 --fit-only

两条现场口径（都不知道就别猜，第一条工具能自己解）
------------------------------------------------
**① 卷尺零点压在机器人脚尖上。** 光心的地面投影点现场根本找不到，不要让人
去测它。用 `--fit-offset` 把「脚尖 → 光心地面投影」的纵向距离解出来。

⚠ `core/ground_homography.py` 的 `CAMERA_TO_BODY_FORWARD_CM = 4.0` 是**仿真参数**，
不是实测值，不能拿它当这个偏移的真值对照。

**② 刻度印在卷尺上表面，比地面高约 1cm。** 这是已知的测量缺陷——刻度点不是
地面点。用 `--ruler-height`（默认 1.0）补偿：几何上"目标抬升 λ"严格等价于
"相机降低 λ"，不补会引入约 λ/h ≈ 3% 的比例误差（40cm 处就是 1.2cm）。

⚠ 零点偏移在自由射影模型下**不可辨识**（`y=(a+bd)/(c+d)` 对 d 平移不变，
实测偏移 ±10cm 残差一字不变），只有物理模型解得出来，且**必须带畸变**
（不带会偏 1.5cm）。

交互
----
    左键   记录当前待点的刻度
    滚轮   以光标为中心缩放（点刻度必须放大，否则差几个像素）
    右键拖动 / 方向键   平移
    n 跳过当前刻度   u 撤销   r 重置
    f 拟合并打印     s 保存 JSON    q/ESC 退出
"""

import argparse
import json
import os
import sys
from datetime import datetime

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from core.camera_config import CAMERA_DISTORTION, CAMERA_INTRINSIC
from core.paths import RESULT_DIR

DEFAULT_OUT = os.path.join(RESULT_DIR, "ruler_profile.json")
#: 默认刻度序列：近段密（起跨点看的就是近段），远段疏
DEFAULT_DISTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20,
                 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]

WIN_W, WIN_H = 1280, 900


# =====================================================================
# 拟合
# =====================================================================

def _ls(fun, x0, **kw):
    from scipy.optimize import least_squares
    return least_squares(fun, x0, **kw)


def fit_mobius(d, y):
    """自由射影模型 y = (a + b·d)/(c + d)

    这是"任意针孔相机 + 平地面"的通解（相机高度/俯角/焦距/主点全被 a,b,c
    吸收），因此与标尺在画面里的位置、朝向无关。它给出**能达到的最好结果**，
    物理模型不可能比它更好。
    """
    best = None
    for a0 in (-2000., 0., 2000.):
        for b0 in (-20000., 0., 20000.):
            for c0 in (1., 50., 500.):
                try:
                    r = _ls(lambda p: (p[0] + p[1] * d) / (p[2] + d) - y,
                            [a0, b0, c0], method="lm", max_nfev=200000)
                except Exception:
                    continue
                if best is None or r.cost < best.cost:
                    best = r
    return best.x if best is not None else None


def inv_mobius(p, yv):
    a, b, c = p
    den = yv - b
    if abs(den) < 1e-12:
        return float("nan")
    return (a - yv * c) / den


def phys_y(d, h, th, fy, cy, use_dist, k1, k2, ruler_h=0.0):
    """物理针孔地面投影（可含径向畸变）。仅在标尺接近光轴平面时严格成立

    ruler_h：刻度点离地高度（cm）。**刻度印在卷尺上表面，卷尺厚约 1cm，
    所以刻度点其实在 z=+ruler_h，不是地面点。**

    几何上"目标抬升 λ"严格等价于"相机降低 λ"：
        (0,d,λ) 在相机高 h 下的投影 ≡ (0,d,0) 在相机高 (h−λ) 下的投影
    （把 h 与 d 同乘一个系数，yn 不变）。不修正会引入随距离增长的系统性
    比例误差，量级 λ/h（h=34.5、λ=1 时约 2.9%，40cm 处就是 1.2cm）。
    近距零点虽然受影响小，但它正是起跨点要的东西，不能不管。
    """
    he = h - ruler_h
    yn = (he * np.cos(th) - d * np.sin(th)) / (d * np.cos(th) + he * np.sin(th))
    if use_dist:
        yn = yn * (1.0 + k1 * yn * yn + k2 * yn ** 4)
    return fy * yn + cy


def inv_phys(yv, h, th, fy, cy, use_dist, k1, k2, ruler_h=0.0):
    from scipy.optimize import brentq
    f = lambda dd: phys_y(dd, h, th, fy, cy, use_dist, k1, k2, ruler_h) - yv
    try:
        return brentq(f, 0.3, 400.0)
    except Exception:
        return float("nan")


def fit_phys(d, y, h_lock=None, use_dist=True, ruler_h=0.0):
    """拟合 (h, theta)；fy/cy 锁死真实内参。h_lock 给定时只解俯角"""
    fy0, cy0 = float(CAMERA_INTRINSIC[1, 1]), float(CAMERA_INTRINSIC[1, 2])
    k1, k2 = float(CAMERA_DISTORTION[0]), float(CAMERA_DISTORTION[1])
    hs = [h_lock] if h_lock else [20., 30., 34.5, 43., 50., 60., 80.]
    best = None
    for h in hs:
        for t0 in np.arange(20.0, 88.0, 2.0):
            f = lambda q: phys_y(d, h, np.radians(q[0]), fy0, cy0,
                                 use_dist, k1, k2, ruler_h) - y
            try:
                r = _ls(f, [t0], bounds=([3.0], [89.5]), max_nfev=200000)
            except Exception:
                continue
            if best is None or r.cost < best[0].cost:
                best = (r, h)
    if best is None:
        return None
    r, h = best
    # q[0] 本身就是"度"（模型内部才转弧度），不要再 degrees() 一次
    return h, float(r.x[0]), fy0, cy0, k1, k2


def solve_zero_offset(d_read, y, h, use_dist=True, ruler_h=0.0):
    """锁死高度与真实内参，拟合 (俯角, 零点偏移)

    刻度读数是"离卷尺零点"的距离，真实地面距离 = 读数 + 偏移。
    **偏移在自由射影模型下不可辨识**（y=(a+bd)/(c+d) 对 d 平移不变），
    所以只能靠物理模型解出来，或直接量。

    本项目现场约定：卷尺零点压在**机器人脚尖**上。光心的地面投影点现场
    根本找不到，所以别让用户去测它——让这里解出来。输出的"零点偏移"就是
    **脚尖到光心地面投影的纵向距离**。

    ⚠ 注意：`core/ground_homography.py` 的 `CAMERA_TO_BODY_FORWARD_CM = 4.0`
    是**仿真参数**，不是实测值，不能拿来当这个偏移的真值对照。
    """
    fy0, cy0 = float(CAMERA_INTRINSIC[1, 1]), float(CAMERA_INTRINSIC[1, 2])
    k1, k2 = float(CAMERA_DISTORTION[0]), float(CAMERA_DISTORTION[1])
    best = None
    for t0 in (0., 5., 10., 15., 20., 30.):
        for th0 in np.arange(30.0, 86.0, 4.0):
            f = lambda q: phys_y(d_read + q[1], h, np.radians(q[0]),
                                 fy0, cy0, use_dist, k1, k2, ruler_h) - y
            try:
                r = _ls(f, [th0, t0], bounds=([5.0, -30.0], [89.0, 80.0]),
                        max_nfev=200000)
            except Exception:
                continue
            if best is None or r.cost < best.cost:
                best = r
    return best


def report(d, y, x=None, fit_offset=False, height=34.5, ruler_h=1.0):
    """跑全部拟合并打印。返回结果字典

    height ：相机光心离地高度（cm），卷尺实测
    ruler_h：刻度点离地高度（cm），默认 1.0 —— 刻度印在卷尺上表面，
             卷尺比地面高约 1cm，这是已知的测量缺陷，必须补偿
    """
    d = np.asarray(d, float)
    y = np.asarray(y, float)
    out = {"n_points": int(len(d)), "dists_cm": d.tolist(), "y_px": y.tolist(),
           "height_cm": height, "ruler_h_cm": ruler_h}
    if x is not None:
        cx = float(CAMERA_INTRINSIC[0, 2])
        out["x_px"] = list(map(float, x))
        out["x_spread_px"] = float(np.ptp(x))
        out["x_offset_from_cx_px"] = float(np.mean(np.abs(np.asarray(x) - cx)))

    print(f"\n===== 拟合（{len(d)} 个刻度点）=====")
    print(f"  高度假设 {height:.1f}cm；刻度点离地 {ruler_h:.1f}cm"
          f"（卷尺厚，已按『等效降低相机』补偿；不补会差约 "
          f"{100 * ruler_h / max(height - ruler_h, 1e-6):.1f}%）")

    # 走向自检：远处在上（y 小）是正常俯视；反了说明配对/顺序有问题
    slope_sign = np.sign(np.polyfit(d, y, 1)[0])
    if slope_sign < 0:
        print("  走向自检：y 随距离递减 ✓（远处在上，俯视相机正常朝向）")
        out["direction_ok"] = True
    else:
        print("  ⚠ 走向自检：y 随距离**递增**——等于要求一台倒置的相机。"
              "检查点击顺序是否与刻度值反了")
        out["direction_ok"] = False

    if x is not None and out["x_offset_from_cx_px"] > 60:
        print(f"  ⚠ 刻度点偏离画面中心列 {out['x_offset_from_cx_px']:.0f}px："
              "物理模型（h,θ）只在近光轴处严格成立，自由射影模型不受影响")

    if len(d) < 4:
        print("  点太少（<4），不做拟合")
        return out

    # --- 自由射影（能达到的上限）---
    p = fit_mobius(d, y)
    if p is not None:
        rms = float(np.sqrt(np.mean(((p[0] + p[1] * d) / (p[2] + d) - y) ** 2)))
        dh = np.array([inv_mobius(p, yv) for yv in y])
        e = dh - d
        print(f"  自由射影模型 (a,b,c 三参)      RMS={rms:6.2f}px | "
              f"距离误差 worst={np.max(np.abs(e)):5.2f}cm  {np.round(e, 2)}")
        out["mobius"] = {"abc": list(map(float, p)), "rms_px": rms,
                         "dist_err_cm": e.tolist()}

    # --- 物理模型：高度自由 / 高度锁死实测值 ---
    for tag, h_lock in (("高度自由      ", None), ("高度锁死 34.5cm", 34.5)):
        for use_dist in (True, False):
            r = fit_phys(d, y, h_lock=h_lock, use_dist=use_dist, ruler_h=ruler_h)
            if r is None:
                continue
            h, th, fy0, cy0, k1, k2 = r
            pred = phys_y(d, h, np.radians(th), fy0, cy0, use_dist, k1, k2, ruler_h)
            rms = float(np.sqrt(np.mean((pred - y) ** 2)))
            dh = np.array([inv_phys(yv, h, np.radians(th), fy0, cy0,
                                    use_dist, k1, k2, ruler_h) for yv in y])
            e = dh - d
            lbl = "含畸变" if use_dist else "无畸变"
            print(f"  物理模型 {tag} {lbl}  h={h:5.1f}cm 俯角={th:4.1f}° | "
                  f"RMS={rms:6.2f}px | 距离误差 worst={np.max(np.abs(e)):5.2f}cm  "
                  f"{np.round(e, 2)}")
            key = f"phys_{'free' if h_lock is None else 'h34.5'}_{'dist' if use_dist else 'nodist'}"
            out[key] = {"h_cm": h, "theta_deg": th, "rms_px": rms,
                        "dist_err_cm": e.tolist()}

    print("\n  判读：")
    print("   · 自由射影的 RMS 是这类数据的上限；物理模型明显更差说明刻度值/几何不自洽")
    print("   · 「含畸变 vs 无畸变」的差就是桶形畸变帮了多少忙")
    print("   · 最终要的是**近段距离误差**（起跨点靠的是近距零点），不是总 RMS")

    # --- 零点偏移：现场卷尺压在脚尖上，这个数就是"脚尖到光心地面投影" ---
    if fit_offset:
        print(f"\n  --- 零点偏移拟合（高度锁死 {height:.1f}cm，fy/cy 锁死）---")
        for use_dist in (True, False):
            r = solve_zero_offset(d, y, height, use_dist=use_dist, ruler_h=ruler_h)
            if r is None:
                continue
            th, t = float(r.x[0]), float(r.x[1])
            fy0, cy0 = float(CAMERA_INTRINSIC[1, 1]), float(CAMERA_INTRINSIC[1, 2])
            k1, k2 = float(CAMERA_DISTORTION[0]), float(CAMERA_DISTORTION[1])
            rms = float(np.sqrt(np.mean(r.fun ** 2)))
            dh = np.array([inv_phys(yv, height, np.radians(th), fy0, cy0,
                                    use_dist, k1, k2, ruler_h) for yv in y]) - t
            e = dh - d
            print(f"    {'含畸变' if use_dist else '无畸变'}  俯角={th:4.1f}°  "
                  f"零点偏移={t:+6.2f}cm | RMS={rms:6.2f}px | "
                  f"距离误差 worst={np.max(np.abs(e)):5.2f}cm")
            key = f"offset_{'dist' if use_dist else 'nodist'}"
            out[key] = {"theta_deg": th, "zero_offset_cm": t, "rms_px": rms,
                        "height_cm": height, "dist_err_cm": e.tolist()}
        print("    · 零点偏移 = 卷尺零点(脚尖)到光心地面投影的纵向距离，应为正")
        print("    · 刻度点离地高度已按 --ruler-height 补偿；它主要影响**比例**"
              "（约 λ/h），对零点偏移本身只差零点几厘米")
    return out


# =====================================================================
# 交互
# =====================================================================

class RulerClicker:
    """带缩放/平移的刻度点击器（缩放着实必要：刻度是细线，不放大点不准）"""

    def __init__(self, frame, dists, out_path, image_path=None, pitch=None,
                 fit_offset=False, height=34.5, ruler_h=1.0):
        self.img = frame
        self.h, self.w = frame.shape[:2]
        self.dists = list(dists)
        self.out_path = out_path
        self.image_path = image_path
        self.pitch = pitch
        self.fit_offset = fit_offset
        self.height = height
        self.ruler_h = ruler_h
        self.points = []          # [(d_cm, x_px, y_px)]
        self.skipped = []
        self.i = 0
        # 视图矩形（原生坐标）
        s = min(WIN_W / self.w, WIN_H / self.h)
        vw, vh = self.w * s, self.h * s
        self.x0 = (self.w - vw) / 2.0
        self.y0 = (self.h - vh) / 2.0
        self.vw, self.vh = vw, vh
        self.win = "calib_ruler_profile"
        self.cursor = None
        self.drag = None

    # ---- 坐标变换 ----
    def to_native(self, u, v):
        return self.x0 + u * self.vw / WIN_W, self.y0 + v * self.vh / WIN_H

    def to_disp(self, x, y):
        return ((x - self.x0) * WIN_W / self.vw, (y - self.y0) * WIN_H / self.vh)

    def zoom(self, factor, u, v):
        nx, ny = self.to_native(u, v)
        self.vw = float(np.clip(self.vw / factor, 30.0, self.w * 2.0))
        self.vh = float(np.clip(self.vh / factor, 20.0, self.h * 2.0))
        self.x0 = nx - (u / WIN_W) * self.vw
        self.y0 = ny - (v / WIN_H) * self.vh

    def pan(self, dx, dy):
        self.x0 += dx * self.vw / WIN_W
        self.y0 += dy * self.vh / WIN_H

    # ---- 事件 ----
    def on_mouse(self, event, u, v, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.cursor = (u, v)
        elif event == cv2.EVENT_MOUSEWHEEL:
            self.zoom(1.25 if flags > 0 else 1 / 1.25, u, v)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.drag = (u, v)
        elif event == cv2.EVENT_RBUTTONUP:
            self.drag = None
        elif event == cv2.EVENT_MOUSEMOVE and self.drag:
            self.pan(u - self.drag[0], v - self.drag[1])
            self.drag = (u, v)
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.record(*self.to_native(u, v))

    def record(self, x, y):
        if self.i >= len(self.dists):
            print("刻度已点完（f 拟合 / s 保存 / u 撤销）")
            return
        self.points.append((float(self.dists[self.i]), float(x), float(y)))
        print(f"  #{len(self.points):2d}  d={self.dists[self.i]:>5.1f}cm  "
              f"px=({x:7.1f},{y:7.1f})")
        self.i += 1

    # ---- 绘制 ----
    def render(self):
        # 取视图矩形并在原生图上裁切+缩放（保证放大后仍是原生像素）
        x0, y0 = int(round(self.x0)), int(round(self.y0))
        x1, y1 = int(round(self.x0 + self.vw)), int(round(self.y0 + self.vh))
        cx0, cy0 = max(0, x0), max(0, y0)
        cx1, cy1 = min(self.w, x1), min(self.h, y1)
        if cx1 <= cx0 or cy1 <= cy0:
            return False
        crop = self.img[cy0:cy1, cx0:cx1]
        vis = cv2.resize(crop, (WIN_W, WIN_H), interpolation=cv2.INTER_NEAREST)
        sx = WIN_W / self.vw
        sy = WIN_H / self.vh

        for i, (d, px, py) in enumerate(self.points):
            u = (px - self.x0) * sx
            v = (py - self.y0) * sy
            cv2.drawMarker(vis, (int(u), int(v)), (0, 255, 0),
                           cv2.MARKER_CROSS, 26, 2)
            cv2.putText(vis, f"{d:.0f}", (int(u) + 12, int(v) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if self.cursor:
            u, v = self.cursor
            nx, ny = self.to_native(u, v)
            cv2.line(vis, (u, 0), (u, WIN_H), (0, 200, 255), 1)
            cv2.line(vis, (0, v), (WIN_W, v), (0, 200, 255), 1)
            cv2.putText(vis, f"({nx:.0f},{ny:.0f})  放大 {self.w / self.vw:.2f}x",
                        (12, WIN_H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 200, 255), 2)

        if self.i < len(self.dists):
            tip = (f"click #{self.i + 1}: d = {self.dists[self.i]:.0f} cm   "
                   f"| wheel zoom  n skip  u undo  f fit  s save  q quit")
        else:
            tip = "all ticks clicked  |  f fit  s save  u undo  q quit"
        cv2.putText(vis, tip, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2)
        cv2.imshow(self.win, vis)
        return True

    def run(self):
        print(f"\n标尺刻度点击标定：按提示顺序点击刻度")
        print(f"待点刻度（cm）：{self.dists}")
        print("放大后再点（刻度是细线）；点完按 f 看拟合，s 保存")
        cv2.namedWindow(self.win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.win, self.on_mouse)
        if not self.render():
            print("图像为空")
            return None
        res = None
        while True:
            k = cv2.waitKey(20) & 0xFF
            if k in (27, ord("q")):
                break
            if k == ord("u") and self.points:
                d, _, _ = self.points.pop()
                self.i = max(0, self.i - 1)
                print(f"  撤销 d={d:.0f}cm")
            elif k == ord("n"):
                if self.i < len(self.dists):
                    self.skipped.append(self.dists[self.i])
                    print(f"  跳过 d={self.dists[self.i]:.0f}cm")
                    self.i += 1
            elif k == ord("r"):
                self.points, self.skipped, self.i = [], [], 0
                print("  已重置")
            elif k == ord("f"):
                res = self.fit_and_report()
            elif k == ord("s"):
                res = self.fit_and_report(save=True)
            self.render()
        cv2.destroyAllWindows()
        return res

    def fit_and_report(self, save=False):
        if len(self.points) < 3:
            print("至少 3 个点")
            return None
        d = [p[0] for p in self.points]
        x = [p[1] for p in self.points]
        y = [p[2] for p in self.points]
        res = report(d, y, x=x, fit_offset=self.fit_offset,
                      height=self.height, ruler_h=self.ruler_h)
        res.update({"image": self.image_path, "pitch": self.pitch,
                    "fit_offset": self.fit_offset, "height_cm": self.height,
                    "skipped_cm": self.skipped,
                    "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "points": [{"d_cm": p[0], "x_px": p[1], "y_px": p[2]}
                               for p in self.points]})
        if save:
            os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
            with open(self.out_path, "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)
            print(f"\n已保存: {self.out_path}")
        return res


# =====================================================================
# 入口
# =====================================================================

def _parse_floats(s):
    return [float(v) for v in s.replace("，", ",").split(",") if v.strip()]


def _parse_px(s):
    out = []
    for part in s.replace("，", ",").split(";"):
        part = part.strip()
        if not part:
            continue
        px, py = part.split(",")
        out.append((float(px), float(py)))
    return out


def parse_args():
    ap = argparse.ArgumentParser(description="标尺刻度 → 距离公式标定辅助")
    ap.add_argument("--image", required=True, help="标尺照片（原生分辨率）")
    ap.add_argument("--dists", default=None,
                    help="按点击顺序对应的距离(cm)，逗号分隔；缺省用内置序列")
    ap.add_argument("--offset", type=float, default=0.0,
                    help="卷尺零点相对光心地面投影点的偏移(cm)，正=零点在原点前方；"
                         "会加到所有刻度读数上。**不知道这个值时不要猜，用 --fit-offset**")
    ap.add_argument("--fit-offset", action="store_true",
                    help="让工具用物理模型把零点偏移解出来（现场卷尺压在脚尖上时用这个）")
    ap.add_argument("--height", type=float, default=34.5,
                    help="相机光心离地高度(cm)，卷尺实测，默认 34.5")
    ap.add_argument("--ruler-height", type=float, default=1.0,
                    help="刻度点离地高度(cm)：刻度印在卷尺上表面，卷尺比地面高约 "
                         "1cm，属已知测量缺陷，默认 1.0 并自动补偿")
    ap.add_argument("--px", default=None, help="脚本模式：像素点 'x,y;x,y;...'")
    ap.add_argument("--fit-only", action="store_true", help="只算拟合，不开窗口")
    ap.add_argument("--pitch", type=int, default=None, help="记录用：俯仰脉宽")
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"输出 JSON，默认 {DEFAULT_OUT}")
    return ap.parse_args()


def main():
    args = parse_args()
    frame = cv2.imread(args.image)
    if frame is None:
        print(f"读图失败: {args.image}")
        sys.exit(1)
    print(f"读入: {args.image}  {frame.shape[1]}x{frame.shape[0]}")
    dists = _parse_floats(args.dists) if args.dists else list(DEFAULT_DISTS)
    if args.offset:
        print(f"刻度读数统一加偏移 {args.offset:+.1f}cm（卷尺零点 -> 光心地面投影点）")
        dists = [d + args.offset for d in dists]

    if args.px:
        pts = _parse_px(args.px)
        if len(pts) != len(dists):
            print(f"点数不一致: px={len(pts)} dists={len(dists)}")
            sys.exit(1)
        res = report(dists, [p[1] for p in pts], x=[p[0] for p in pts],
                     fit_offset=args.fit_offset, height=args.height,
                     ruler_h=args.ruler_height)
        res.update({"image": args.image, "pitch": args.pitch,
                    "fit_offset": args.fit_offset, "height_cm": args.height,
                    "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "points": [{"d_cm": d, "x_px": p[0], "y_px": p[1]}
                               for d, p in zip(dists, pts)]})
        if not args.fit_only:
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)
            print(f"\n已保存: {args.out}")
        return

    RulerClicker(frame, dists, args.out, image_path=args.image,
                 pitch=args.pitch, fit_offset=args.fit_offset,
                 height=args.height, ruler_h=args.ruler_height).run()


if __name__ == "__main__":
    main()
