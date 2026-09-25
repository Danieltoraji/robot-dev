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
    # 交互点击（推荐）：按提示顺序点标尺刻度。**不需要任何已知参数。**
    python tools/calib_ruler_profile.py --image photo.jpg

    # 只关心近段（起跨点要的是近距零点精度）
    python tools/calib_ruler_profile.py --image photo.jpg --dists 2,3,4,5,6,8,10,15,20

    # 已有像素点，只重算拟合
    python tools/calib_ruler_profile.py --image photo.jpg --px "1200,1819;1210,1401" \
        --dists 2,10 --fit-only

不需要猜任何东西
----------------
**刻度平面高度、俯角、零点偏移三个量全部由拟合解出**，不要再传"高度大概
34.5、卷尺大概厚 1cm"这类猜测值当默认——那只会给出一个看着确定、其实建立
在猜测上的数。实测（合成数据 + 2px 点击噪声）：三个量都解得出来，
零点偏移只飘 ±0.16cm。

两条现场口径（工具已自动处理，写在这里免得再踩）
------------------------------------------------
**① 卷尺零点压在机器人脚尖上。** 光心的地面投影点现场根本找不到，不要让人
去测它——解出来的"零点偏移"就是「脚尖 → 光心地面投影」的纵向距离。

⚠ `core/ground_homography.py` 的 `CAMERA_TO_BODY_FORWARD_CM = 4.0` 是**仿真参数**，
不是实测值，不能拿它当这个偏移的真值对照。

**② 刻度印在卷尺上表面，比地面高约 1cm**，所以刻度点**不是地面点**。
工具解出的是「光心离**刻度平面**的高度」——卷尺厚度与相机高度在模型里只以
差值出现，**永远分不开**，所以不必也不能分开给。要量**地面**目标时还得把
卷尺厚度加回去，影响约 λ/h 的比例（40cm 约 1cm，2cm 仅 0.06cm，对起跨点无所谓）。

⚠ 零点偏移在自由射影模型下**不可辨识**（`y=(a+bd)/(c+d)` 对 d 平移不变，
实测偏移 ±10cm 残差一字不变），只有物理模型解得出来。

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


def fit_all(d_read, y, use_dist=True):
    """三参全自由：刻度平面高度、俯角、零点偏移

    **默认就用这个，任何参数都不需要猜。**

    已实测这三个量都能从"刻度-像素"数据本身定出来：合成数据（真值
    h_eff=33.5cm / 俯角 58.0° / 零点偏移 12.0cm）在 2px 点击噪声下反复
    拟合，零点偏移只飘 ±0.16cm。所以不要再拿"高度大概 34.5、卷尺大概厚
    1cm"这种猜测值当默认——那只会给出一个看着确定、其实建立在猜测上的数。

    刻度读数是"离卷尺零点"的距离，**真实地面距离 = 读数 + 零点偏移**。
    偏移在自由射影模型下不可辨识（y=(a+bd)/(c+d) 对 d 平移不变），
    只有物理模型解得出来。

    本项目现场约定：卷尺零点压在**机器人脚尖**上（光心的地面投影点现场
    根本找不到），所以解出的偏移就是**脚尖 → 光心地面投影**的纵向距离。

    ⚠ 能定出来的只是 **h_eff = 光心离「刻度平面」的高度 = h − λ**：
    卷尺厚度 λ 与相机高度 h 在模型里只以 h−λ 的形式出现，**两者永远分不开**。
    要量**地面**目标时还得把 λ 加回去（λ 只有大概值），影响是约 λ/h 的
    比例误差——40cm 处约 1cm，2cm 处只有 0.06cm，所以对起跨点无所谓。

    ⚠ `core/ground_homography.py` 的 `CAMERA_TO_BODY_FORWARD_CM = 4.0` 是
    **仿真参数**，不是实测值，不能拿来当这个偏移的真值对照。
    """
    fy0, cy0 = float(CAMERA_INTRINSIC[1, 1]), float(CAMERA_INTRINSIC[1, 2])
    k1, k2 = float(CAMERA_DISTORTION[0]), float(CAMERA_DISTORTION[1])
    best = None
    for t0 in (0., 5., 10., 15., 20., 30.):
        for th0 in (35., 45., 52., 58., 65., 72., 80.):
            for h0 in (20., 27., 33.5, 42., 55., 75.):
                f = lambda q: phys_y(d_read + q[2], q[0], np.radians(q[1]),
                                     fy0, cy0, use_dist, k1, k2) - y
                try:
                    r = _ls(f, [h0, th0, t0],
                            bounds=([8.0, 15.0, -40.0], [150.0, 89.0, 80.0]),
                            max_nfev=400000)
                except Exception:
                    continue
                if best is None or r.cost < best.cost:
                    best = r
    return best


def solve_zero_offset(d_read, y, h, use_dist=True, ruler_h=0.0):
    """锁死高度，只拟合 (俯角, 零点偏移)。作为 fit_all 的交叉校验用"""
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


def emit_mapping(y, fit_x, use_dist=True, ruler_thickness=None, n_rows=25):
    """把拟合结果变成**可直接用的像素 → 距离映射**

    拟合出来的是"刻度平面"上的映射（因为刻度点就长在那个高度上）。要量
    **地面**目标（木条根部、胶条根部），整体乘一个系数：

        a = h / h_eff = 1 + λ / h_eff        （λ = 卷尺厚度）

    乘完之后对地面目标精确——合成数据实测残留 0.000cm。

    为什么只需要一个系数：把距离 d 乘以 k 与把相机高度 h 除以 k 是**同一个
    投影**，所以"刻度高了 λ"在数据里长得和"相机矮了 λ"一模一样。这一个
    数据**结构上**分不出两者，能定出的形状完全正确，只差这个整体倍数。

    λ 不必很准：估错 1mm，70cm 处差 0.2cm，2cm 处差 0.006cm。
    """
    he, th, t = float(fit_x[0]), float(fit_x[1]), float(fit_x[2])
    fy0 = float(CAMERA_INTRINSIC[1, 1])
    cy0 = float(CAMERA_INTRINSIC[1, 2])
    k1 = float(CAMERA_DISTORTION[0])
    k2 = float(CAMERA_DISTORTION[1])
    a = (1.0 + float(ruler_thickness) / he) if ruler_thickness is not None else None

    ys = np.linspace(float(np.min(y)), float(np.max(y)), n_rows)
    rows = []
    for yv in ys:
        # inv_phys 解出的就是"该像素对应的地面水平距离"（含零点偏移），
        # 不要再减零点偏移——减了就退回成卷尺读数了
        dt = inv_phys(yv, he, np.radians(th), fy0, cy0, use_dist, k1, k2)
        rows.append({"y_px": float(yv), "d_tick_cm": float(dt),
                     "d_floor_cm": (float(dt * a) if a else None)})
    return {"h_eff_cm": he, "theta_deg": th, "zero_offset_cm": t,
            "fy": fy0, "cy": cy0, "k1": k1, "k2": k2,
            "ruler_thickness_cm": ruler_thickness, "scale_a": a,
            "table": rows}


def print_mapping(m):
    """打印映射表与用法"""
    print("\n  ===== 像素 → 距离 映射 =====")
    print(f"    模型参数（刻度平面）：高度 h_eff={m['h_eff_cm']:.2f}cm  "
          f"俯角={m['theta_deg']:.2f}°  零点偏移={m['zero_offset_cm']:+.2f}cm")
    if m["scale_a"] is None:
        print("    ⚠ 没给 --ruler-thickness，**绝对刻度会偏**：地面读数整体"
              f"偏小约 λ/h_eff（λ=1cm 时约 {100*1.0/m['h_eff_cm']:.1f}%）")
        print("      形状是对的，只差这一个倍数。量一下卷尺厚度填进去即可。")
    else:
        print(f"    卷尺厚度 λ={m['ruler_thickness_cm']:.2f}cm  ⇒  "
              f"缩放系数 a = 1 + λ/h_eff = {m['scale_a']:.5f}")
        print(f"    **地面距离 = 刻度平面读数 × a**（+{(m['scale_a']-1)*100:.2f}%）")
    print()
    hdr = f"    {'y_px':>8} {'刻度平面cm':>12}"
    if m["scale_a"] is not None:
        hdr += f" {'地面cm':>10}"
    print(hdr)
    for row in m["table"]:
        line = f"    {row['y_px']:>8.0f} {row['d_tick_cm']:>12.2f}"
        if row["d_floor_cm"] is not None:
            line += f" {row['d_floor_cm']:>10.2f}"
        print(line)
    print("\n    用法：按上表线性插值即可（y 越大 = 越近）。"
          "表已存进 JSON 的 mapping 段。")
    print(f"    ⚠ 表只覆盖你点过的刻度范围（地面 "
          f"{m['table'][-1]['d_tick_cm']:.1f}~{m['table'][0]['d_tick_cm']:.1f}cm）"
          "——超出即为**外推**，不保证准。")
    print("    ⚠ 要量地面目标（木条根部、胶条根部）用**地面cm**列；"
          "刻度平面cm 列系统性偏短，只作标定原始记录。")


def report(d, y, x=None, lock_height=None, lock_ruler_h=0.0,
           ruler_thickness=None):
    """跑全部拟合并打印。返回结果字典

    lock_height    ：可选的交叉校验——先锁死"光心离刻度平面的高度"再拟合。
                     缺省 None = 不锁，三个参数全自由（**推荐**）。
    ruler_thickness：卷尺厚度 λ（cm），**量出来的，不是猜的**。给了它才会
                     输出对**地面**目标可直接用的映射（= 刻度平面读数 ×
                     (1+λ/h_eff)）。不给则形状照样对，只是绝对刻度偏小
                     约 λ/h_eff（λ=1cm 时约 3%）。
    """
    d = np.asarray(d, float)
    y = np.asarray(y, float)
    out = {"n_points": int(len(d)), "dists_cm": d.tolist(), "y_px": y.tolist()}
    if x is not None:
        cx = float(CAMERA_INTRINSIC[0, 2])
        out["x_px"] = list(map(float, x))
        out["x_spread_px"] = float(np.ptp(x))
        out["x_offset_from_cx_px"] = float(np.mean(np.abs(np.asarray(x) - cx)))

    print(f"\n===== 拟合（{len(d)} 个刻度点）=====")

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
              "物理模型只在近光轴处严格成立，自由射影模型不受影响")

    if len(d) < 4:
        print("  点太少（<4），不做拟合")
        return out

    # --- 自由射影：与几何无关的上限 ---
    p = fit_mobius(d, y)
    if p is not None:
        rms = float(np.sqrt(np.mean(((p[0] + p[1] * d) / (p[2] + d) - y) ** 2)))
        dh = np.array([inv_mobius(p, yv) for yv in y])
        e = dh - d
        print(f"  自由射影模型（**针孔族**的上限）  RMS={rms:6.2f}px | "
              f"距离误差 worst={np.max(np.abs(e)):5.2f}cm  {np.round(e, 2)}")
        out["mobius"] = {"abc": list(map(float, p)), "rms_px": rms,
                         "dist_err_cm": e.tolist()}

    fy0 = float(CAMERA_INTRINSIC[1, 1])
    cy0 = float(CAMERA_INTRINSIC[1, 2])
    k1 = float(CAMERA_DISTORTION[0])
    k2 = float(CAMERA_DISTORTION[1])

    # --- 主结果：三参全自由，不需要任何猜测值 ---
    print("\n  --- 主结果：刻度平面高度 / 俯角 / 零点偏移（三参全自由）---")
    best_free = None
    for use_dist in (True, False):
        r = fit_all(d, y, use_dist=use_dist)
        if r is None:
            continue
        if use_dist:
            best_free = r
        he, th, t = float(r.x[0]), float(r.x[1]), float(r.x[2])
        rms = float(np.sqrt(np.mean(r.fun ** 2)))
        dh = np.array([inv_phys(yv, he, np.radians(th), fy0, cy0,
                                use_dist, k1, k2) for yv in y]) - t
        e = dh - d
        print(f"    {'含畸变' if use_dist else '无畸变'}  刻度平面高度={he:5.2f}cm  "
              f"俯角={th:5.2f}°  零点偏移={t:+6.2f}cm | RMS={rms:6.2f}px | "
              f"距离误差 worst={np.max(np.abs(e)):5.2f}cm  {np.round(e, 2)}")
        out[f"free_{'dist' if use_dist else 'nodist'}"] = {
            "h_eff_cm": he, "theta_deg": th, "zero_offset_cm": t,
            "rms_px": rms, "dist_err_cm": e.tolist()}
    print("    · 零点偏移 = 卷尺零点(脚尖) → 光心地面投影的纵向距离，应为正")
    print("    · 刻度平面高度 = 光心离**刻度平面**的高度 = 相机离地高度 − 卷尺厚度；")
    print("      这两者模型里只以差值出现，**永远分不开**，所以量地面目标时")
    print("      还需把卷尺厚度加回去（影响约 λ/h 的比例，40cm 约 1cm、2cm 仅 0.06cm）")

    # --- 可选交叉校验：锁死高度 ---
    if lock_height is not None:
        print(f"\n  --- 交叉校验：锁死刻度平面高度 {lock_height:.2f}cm ---")
        for use_dist in (True, False):
            r = solve_zero_offset(d, y, lock_height, use_dist=use_dist,
                                  ruler_h=lock_ruler_h)
            if r is None:
                continue
            th, t = float(r.x[0]), float(r.x[1])
            rms = float(np.sqrt(np.mean(r.fun ** 2)))
            dh = np.array([inv_phys(yv, lock_height, np.radians(th), fy0, cy0,
                                    use_dist, k1, k2, lock_ruler_h)
                           for yv in y]) - t
            e = dh - d
            print(f"    {'含畸变' if use_dist else '无畸变'}  俯角={th:5.2f}°  "
                  f"零点偏移={t:+6.2f}cm | RMS={rms:6.2f}px | "
                  f"距离误差 worst={np.max(np.abs(e)):5.2f}cm")
            out[f"locked_{'dist' if use_dist else 'nodist'}"] = {
                "h_eff_cm": lock_height, "theta_deg": th, "zero_offset_cm": t,
                "rms_px": rms, "dist_err_cm": e.tolist()}

    print("\n  判读：")
    print("   · 针孔族的上限是自由射影；含畸变的物理模型不属该族，**可以更好**")
    print("   · 「含畸变 vs 无畸变」的差就是桶形畸变帮了多少忙")
    print("   · 最终要的是**近段距离误差**（起跨点靠的是近距零点），不是总 RMS")

    # --- 最终交付物：可直接用的像素 → 距离映射 ---
    if best_free is not None:
        m = emit_mapping(y, best_free.x, use_dist=True,
                         ruler_thickness=ruler_thickness)
        print_mapping(m)
        out["mapping"] = m
    return out


# =====================================================================
# 交互
# =====================================================================

class RulerClicker:
    """带缩放/平移的刻度点击器（缩放着实必要：刻度是细线，不放大点不准）"""

    def __init__(self, frame, dists, out_path, image_path=None, pitch=None,
                 lock_height=None, ruler_thickness=None):
        self.img = frame
        self.h, self.w = frame.shape[:2]
        self.dists = list(dists)
        self.out_path = out_path
        self.image_path = image_path
        self.pitch = pitch
        self.lock_height = lock_height
        self.ruler_thickness = ruler_thickness
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
        res = report(d, y, x=x, lock_height=self.lock_height,
                      ruler_thickness=self.ruler_thickness)
        res.update({"image": self.image_path, "pitch": self.pitch,
                    "lock_height": self.lock_height,
                    "ruler_thickness_cm": self.ruler_thickness,
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
                    help="已知零点偏移(cm)时先加上；**不知道就别填**，缺省由拟合解出")
    ap.add_argument("--ruler-thickness", type=float, default=None,
                    help="卷尺厚度(cm)：量出来的（卡尺/尺子比一下），不是猜的。"
                         "给了它才会输出对地面目标的映射（读数 × (1+λ/h_eff)）；"
                         "不给则形状照样对，只是绝对刻度偏小约 λ/h_eff（λ=1cm 约 3%）")
    ap.add_argument("--lock-height", type=float, default=None,
                    help="可选：锁死『光心离刻度平面的高度』(cm) 做交叉校验。"
                         "缺省不锁——三个参数全自由，不需要任何猜测值")
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
                     lock_height=args.lock_height,
                     ruler_thickness=args.ruler_thickness)
        res.update({"image": args.image, "pitch": args.pitch,
                    "lock_height": args.lock_height,
                    "ruler_thickness_cm": args.ruler_thickness,
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
                 pitch=args.pitch, lock_height=args.lock_height,
                 ruler_thickness=args.ruler_thickness).run()


if __name__ == "__main__":
    main()
