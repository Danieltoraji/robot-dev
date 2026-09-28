# -*- coding: utf-8 -*-
"""用一张"期望的到达帧"标定分区参数（tools/calib_zone_from_photo.py）

为什么需要：分区（绿走廊/蓝区/紫区）是**画在人眼看到的画面里**的东西，而
`ZONE_PURPLE_HEIGHT` / `ZONE_BLUE_HEIGHT` / 绿蓝半宽这几个数决定"多近才算站上去"。
在相机高度从 56cm 改成实测 33.9cm 之后，这几个数对应的**地面距离全变了**，
靠解析式来回推很容易错（本脚本第一版就把行→距离反解写反过）。

做法：拿一张**现场确认过"这就是正好到达"**的照片，直接
  ① 跑真实检测（同一个 detector，同一套归一化）拿到目标观测；
  ② 用真实 `_zone_shares` 链路算四区份额（与关卡判据同源）；
  ③ 把**分区边界线**画到照片上，直接看紫带有没有盖住目标面板；
  ④ 把关键读数（t、dx、框宽、四区份额、以及"当前参数下紫/蓝带覆盖的地面距离"）
     一并打印 —— 参数改前/改后各跑一次即可对比。

用法：
    python tools/calib_zone_from_photo.py --image <到达参考图> --digit 3
    python tools/calib_zone_from_photo.py --image <图> --digit 3 --sweep-purple
"""

import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

import cv2
import numpy as np

import levels.nine_grid as NG
from levels.nine_grid import NineGridLevel
from levels.nine_grid_shared import (
    PITCH_DOWN, NineGridShared, measured_cam_height_cm, project_ground_to_pixel,
)
from vision.nine_grid_detector import (
    ID_TO_COLOR, build_color_mask, normalize_illumination,
)


class _NullRobot:
    """只喂给 NineGridShared 用于取 `_zone_shares` 的窄接口"""

    current_head_pulse = 1500
    current_pitch_pulse = PITCH_DOWN

    def set_head(self, *a, **k):
        pass

    def set_pitch(self, *a, **k):
        pass

    def act(self, *a, **k):
        pass

    def run_action(self, *a, **k):
        pass

    def capture_frame(self):
        return None


def zone_shares(level, frame, color):
    """优先用关卡自己的 _zone_shares（同一份实现），没有才用本文件的同源副本"""
    fn = getattr(level, "_zone_shares", None)
    if fn is not None:
        return fn(frame, color)
    return _zone_shares_fallback(level.detector, frame, color)


def _zone_shares_fallback(detector, frame, color):
    """与 `NineGridLevel._zone_shares` **逐行同源**的四区份额

    （那一份挂在统一决策类上，本工具不 import 它，避免把整个关卡拖进来；
    归一化链路/掩膜/分区都调同一批函数，读数可比。）
    """
    ww = int(detector.work_width)
    scale = frame.shape[1] / float(ww)
    work = cv2.resize(frame, (ww, int(round(frame.shape[0] / scale))))
    work, detector.last_norm = normalize_illumination(work)
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    mask = build_color_mask(hsv, color)
    tot = int(cv2.countNonZero(mask))
    if tot <= 0:
        return 0.0, 0.0, 0.0, 0.0
    m = mask.astype(bool)
    regs = NG.zone_masks(mask.shape[1], mask.shape[0])
    n = {k: int(m[v].sum()) for k, v in regs.items()}
    return (tot / float(mask.size), n["purple"] / tot, n["orange"] / tot,
            (n["blueL"] - n["blueR"]) / tot)


def ground_dist_of_row(t, pitch=None, head=1500, off=25.0, h=None):
    """画幅归一化行 t → 该行看到的地面距离（cm）——用**关卡自己的投影**反查

    注意方向：t 越小（越靠画幅顶边）看到的地面越远。二分时务必用这个方向，
    写反了会得到"紫带覆盖 8~1cm"这种荒谬结果（第一版就踩了）。
    """
    h = measured_cam_height_cm() if h is None else h
    pitch = PITCH_DOWN if pitch is None else pitch
    lo, hi = 0.5, 900.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        y = float(project_ground_to_pixel(np.array([[50.0, 50.0 + mid]]), 50.0, 50.0,
                                          0.0, pitch, head, pitch_offset_deg=off,
                                          cam_height_cm=h)[0][1])
        if y > t * 1944.0:      # 该行比我们要的更低（更近）→ 距离取小了
            lo = mid
        else:
            hi = mid
    return hi


def imread_unicode(path):
    """Windows 上 cv2.imread 读不了含中文的路径 ⇒ 用 imdecode 绕开"""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None


def imwrite_unicode(path, img):
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return bool(ok)


def main():
    ap = argparse.ArgumentParser(description="用到达参考帧标定分区参数")
    ap.add_argument("--image", required=True)
    ap.add_argument("--digit", type=int, required=True, help="目标数字（1..7）")
    ap.add_argument("--out", default=None, help="叠加图输出路径")
    ap.add_argument("--sweep-purple", action="store_true",
                    help="扫一遍 ZONE_PURPLE_HEIGHT，打印各档的紫/橙份额")
    args = ap.parse_args()

    frame = imread_unicode(args.image)
    if frame is None:
        print(f"读不到图片: {args.image}")
        return 2
    H, W = frame.shape[:2]
    lv = NineGridLevel(_NullRobot())
    color = ID_TO_COLOR[args.digit]

    obs = lv.detector.detect_panels(frame, colors=[color], arbitrate=False,
                                    drop_border=False)
    o = max(obs, key=lambda x: x.hull_area) if obs else None
    print(f"画幅 {W}x{H}｜目标色 {color}（数字 {args.digit}）")
    if o is None:
        print("  ⚠️ 没检出目标面板")
    else:
        px, py = (o.hull_centroid_px if o.clipped else o.center_px)
        bx, by, bw, bh = (float(v) for v in o.bbox)
        print(f"  检出: 中心 ({px:.0f},{py:.0f})｜框 ({bw:.0f}x{bh:.0f})"
              f"｜clipped={o.clipped}｜宽/高 {bw / max(bh, 1):.2f}")
        print(f"  归一化: t={py / H:.3f}｜dx={abs(px - W / 2) / W:.3f}"
              f"（门槛 居中≤{NG.ARRIVE_CENTER_MAX}）")
        print(f"  档位判定: {NG.zone_at_pixel(px, py, W, H)}")

    sh = zone_shares(lv, frame, color)
    if sh is not None and o is not None:
        px, py = (o.hull_centroid_px if o.clipped else o.center_px)
        g = []
        lv._arrive_pixels_ok(sh, o, float(px), float(W), collect=g)
        print(f"  四区份额: whole={sh[0]:.3f} 紫={sh[1]:.3f} 橙={sh[2]:.3f} "
              f"不对称={sh[3]:+.3f}")
        print("  到达门读数: " + "｜".join(g))

    tp, tb = NG.zone_purple_top_t(), NG.zone_blue_top_t()
    print(f"\n当前参数: ZONE_PURPLE_HEIGHT={NG.ZONE_PURPLE_HEIGHT}（上沿 t={tp:.3f}）"
          f"｜ZONE_BLUE_HEIGHT={NG.ZONE_BLUE_HEIGHT}（上沿 t={tb:.3f}）")
    print(f"  紫带覆盖地面 {ground_dist_of_row(tp):.0f} ~ {ground_dist_of_row(1.0):.0f} cm"
          f"｜蓝区覆盖 {ground_dist_of_row(tb):.0f} ~ {ground_dist_of_row(1.0):.0f} cm")
    print(f"  画幅底边看到的地面 {ground_dist_of_row(1.0):.0f} cm"
          f"｜画幅顶边 {ground_dist_of_row(0.0):.0f} cm")

    if args.sweep_purple:
        print("\n扫 ZONE_PURPLE_HEIGHT（改这个数 = 移动紫带上沿）：")
        print("   紫高   上沿t   紫带覆盖地面(cm)   紫份额   橙份额   档位")
        old = NG.ZONE_PURPLE_HEIGHT
        try:
            for ph in (0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85):
                NG.ZONE_PURPLE_HEIGHT = ph
                sh2 = zone_shares(lv, frame, color)
                t2 = NG.zone_purple_top_t()
                if o is not None:
                    px2, py2 = (o.hull_centroid_px if o.clipped else o.center_px)
                    z = NG.zone_at_pixel(px2, py2, W, H)
                else:
                    z = "?"
                print("  %5.2f  %6.3f   %6.0f ~ %6.0f    %.3f    %.3f    %s"
                      % (ph, t2, ground_dist_of_row(t2), ground_dist_of_row(1.0),
                         sh2[1] if sh2 else -1, sh2[2] if sh2 else -1, z))
        finally:
            NG.ZONE_PURPLE_HEIGHT = old

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.image)),
                                   "zone_overlay.jpg")
    img = frame.copy()
    for name, xy in NG.zone_lines().items() if isinstance(NG.zone_lines(), dict) else []:
        pass
    # 画分割线：紫带上下沿与蓝区上沿（横线）+ 绿/蓝走廊边界（斜线按公式取点）
    def hline(t, colr, txt):
        y = int(t * H)
        cv2.line(img, (0, y), (W, y), colr, 3)
        cv2.putText(img, txt, (20, max(30, y - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, colr, 3)
    hline(tp, (255, 0, 255), "purple top")
    hline(tb, (255, 128, 0), "blue top")
    for t in np.linspace(0, 1, 40):
        y = int(t * H)
        half = NG.zone_green_half(t) * W
        cv2.circle(img, (int(W / 2 - half), y), 3, (0, 255, 0), -1)
        cv2.circle(img, (int(W / 2 + half), y), 3, (0, 255, 0), -1)
        if t >= tb:
            u = (t - tb) / max(NG.ZONE_BLUE_HEIGHT, 1e-9)
            hb = NG.zone_blue_half(u) * W
            cv2.circle(img, (int(W / 2 - hb), y), 3, (255, 128, 0), -1)
            cv2.circle(img, (int(W / 2 + hb), y), 3, (255, 128, 0), -1)
    cv2.imwrite(out, img) if False else imwrite_unicode(out, img)
    print(f"\n叠加图已写出: {out}（绿点=绿走廊边、橙点=蓝外沿、横线=紫/蓝上沿）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
