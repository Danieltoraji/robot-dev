# -*- coding: utf-8 -*-
"""视觉实测转弯角度（竖直红线参照版）：只对「竖直连通域」做 PCA 测角。

竖直红线（高 > 宽）的角度对机器人 yaw 最敏感：正对前方竖线 ≈ ±90°，
机器人转 θ 角，竖线主轴角度随之变化约 θ。比整帧 PCA（横条+竖线混合）可靠得多。

对 turn_left / turn_right 各测 2 次，输出净转角。
"""
import time

import cv2
import numpy as np

import hiwonder.Camera as Camera
import hiwonder.ActionGroupControl as AGC

try:
    from Functions import RedLinePatrolV3 as redline
except ImportError:
    import RedLinePatrolV3 as redline

FRAME = (640, 480)
MIN_RED_PIXELS = 300


def build_maps():
    param_data = np.load(redline.calibration_param_path + '.npz')
    mtx = param_data['mtx_array']
    dist = param_data['dist_array']
    newcameramtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, FRAME, 0, FRAME)
    return cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, FRAME, 5)


def vertical_axis_angle(mask):
    """对竖直连通域（高>宽）的红色像素做 PCA，返回主轴角度（度）。
    竖直红线≈±90。返回 (angle, n, linearity, rect)。"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < 500:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if h <= w * 1.2:   # 只取竖直区域（高明显大于宽）
            continue
        if area > best_area:
            best_area = area
            best = (x, y, w, h)
    if best is None:
        return None, 0, None, None
    x, y, w, h = best
    sub = mask[y:y + h, x:x + w]
    ys, xs = np.nonzero(sub)
    n = xs.size
    if n < MIN_RED_PIXELS:
        return None, n, None, best
    mx, my = xs.mean() + x, ys.mean() + y
    xc = xs - xs.mean()
    yc = ys - ys.mean()
    cov_xx = float((xc * xc).mean())
    cov_yy = float((yc * yc).mean())
    cov_xy = float((xc * yc).mean())
    tr = cov_xx + cov_yy
    det = cov_xx * cov_yy - cov_xy * cov_xy
    disc = max(0.0, (tr / 2.0) ** 2 - det)
    e1 = tr / 2.0 + np.sqrt(disc)
    e2 = tr / 2.0 - np.sqrt(disc)
    linearity = 1.0 - (e2 / e1) if e1 > 1e-6 else 0.0
    theta = 0.5 * np.arctan2(2.0 * cov_xy, cov_xx - cov_yy)
    return np.degrees(theta), n, linearity, best


def grab_angle(cam, mapx, mapy):
    angles, counts, lins, rects = [], [], [], []
    for _ in range(5):
        ret, frame = cam.read()
        if not ret or frame is None:
            continue
        corrected = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
        mask = redline.build_red_mask(corrected)
        ang, n, lin, rect = vertical_axis_angle(mask)
        if ang is not None:
            angles.append(ang)
            counts.append(n)
            lins.append(lin)
            rects.append(rect)
        time.sleep(0.08)
    if not angles:
        return None, 0, 0.0, None
    i = int(np.argmax(counts))
    return float(angles[i]), int(counts[i]), float(lins[i]), rects[i]


def norm(d):
    return (d + 180.0) % 360.0 - 180.0


def measure(cam, mapx, mapy, name):
    a0, n0, l0, r0 = grab_angle(cam, mapx, mapy)
    if a0 is None:
        print('%-24s 画面无竖直红线，跳过' % name)
        return None
    AGC.runActionGroup(name, times=1, with_stand=True)
    time.sleep(0.6)
    a1, n1, l1, r1 = grab_angle(cam, mapx, mapy)
    if a1 is None:
        print('%-24s 转后无竖直红线 (前角=%+.1f)' % (name, a0))
        return None
    d = norm(a1 - a0)
    print('%-24s 前角=%+6.1f 后角=%+6.1f 净转角=%+7.2f (直线度%.2f->%.2f rect%s->%s)'
          % (name, a0, a1, d, l0, l1, r0, r1))
    return d


def main():
    cam = Camera.Camera()
    cam.camera_open()
    time.sleep(1.5)
    AGC.runActionGroup('stand')
    time.sleep(1.0)
    mapx, mapy = build_maps()

    a, n, l, r = grab_angle(cam, mapx, mapy)
    print('初始竖直红线：角度=%+.1f 像素=%d 直线度=%.2f rect=%s' % (a, n, l, r))

    for name in ['turn_left', 'turn_right']:
        vals = []
        for _ in range(2):
            v = measure(cam, mapx, mapy, name)
            if v is not None:
                vals.append(v)
            time.sleep(1.0)
        if vals:
            print('==> %-20s 平均净转角 = %+7.2f 度 (%s)'
                  % (name, sum(vals) / len(vals),
                     ', '.join('%+.1f' % v for v in vals)))
        print('---')

    cam.camera_close()
    print('done')


if __name__ == '__main__':
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc()
        print('FAILED')
