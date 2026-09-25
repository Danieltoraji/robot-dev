# -*- coding: utf-8 -*-
"""视觉实测转弯角度：转弯前后用 PCA 拟合红线主轴方向，角度差即机器人转角。

陀螺仪已损坏（读数恒定），故改用视觉。机器人需站在赛道上、画面内有红线。
对 turn_left / turn_right / *_small_step 各测 3 次，输出净转角。
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
MIN_RED_PIXELS = 300  # 红线像素少于该值判定为无红线


def build_maps():
    import cv2
    param_data = np.load(redline.calibration_param_path + '.npz')
    mtx = param_data['mtx_array']
    dist = param_data['dist_array']
    newcameramtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, FRAME, 0, FRAME)
    return cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, FRAME, 5)


def fit_axis_angle(mask):
    """PCA 主方向角（度）。竖直红线≈±90，水平横条≈0。返回 (angle, n, linearity)。"""
    ys, xs = np.nonzero(mask)
    n = xs.size
    if n < MIN_RED_PIXELS:
        return None, n, None
    mx, my = xs.mean(), ys.mean()
    xc = xs - mx
    yc = ys - my
    cov_xx = float((xc * xc).mean())
    cov_yy = float((yc * yc).mean())
    cov_xy = float((xc * yc).mean())
    trace = cov_xx + cov_yy
    det = cov_xx * cov_yy - cov_xy * cov_xy
    disc = max(0.0, (trace / 2.0) ** 2 - det)
    eig1 = trace / 2.0 + np.sqrt(disc)
    eig2 = trace / 2.0 - np.sqrt(disc)
    linearity = 1.0 - (eig2 / eig1) if eig1 > 1e-6 else 0.0
    theta = 0.5 * np.arctan2(2.0 * cov_xy, cov_xx - cov_yy)
    return np.degrees(theta), n, linearity


def grab_angle(cam, mapx, mapy):
    """抓多帧取中位数角度，稳定自动曝光。"""
    angles = []
    counts = []
    lins = []
    for _ in range(5):
        ret, frame = cam.read()
        if not ret or frame is None:
            continue
        corrected = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
        mask = redline.detect_red_line(corrected)
        ang, n, lin = fit_axis_angle(mask)
        if ang is not None:
            angles.append(ang)
            counts.append(n)
            lins.append(lin)
        time.sleep(0.08)
    if not angles:
        return None, 0, 0.0
    return float(np.median(angles)), int(np.median(counts)), float(np.median(lins))


def measure(cam, mapx, mapy, name):
    a0, n0, l0 = grab_angle(cam, mapx, mapy)
    if a0 is None:
        print('%-24s 画面无红线，跳过' % name)
        return None
    AGC.runActionGroup(name, times=1, with_stand=True)
    time.sleep(0.6)
    a1, n1, l1 = grab_angle(cam, mapx, mapy)
    if a1 is None:
        print('%-24s 转后无红线，跳过 (前角=%+.1f)' % (name, a0))
        return None
    d = (a1 - a0 + 180.0) % 360.0 - 180.0  # 归一化到 -180..180
    print('%-24s 前角=%+6.1f  后角=%+6.1f  净转角=%+7.2f  (像素%d->%d 直线度%.2f->%.2f)'
          % (name, a0, a1, d, n0, n1, l0, l1))
    return d


def main():
    cam = Camera.Camera()
    cam.camera_open()
    time.sleep(1.5)
    AGC.runActionGroup('stand')
    time.sleep(1.0)

    mapx, mapy = build_maps()

    # 先报告当前视角
    a, n, l = grab_angle(cam, mapx, mapy)
    print('初始红线：角度=%+.1f 像素=%d 直线度=%.2f' % (a, n, l))

    for name in ['turn_left', 'turn_right']:
        vals = []
        for _ in range(2):
            v = measure(cam, mapx, mapy, name)
            if v is not None:
                vals.append(v)
            time.sleep(1.0)
        if vals:
            print('==> %-20s 平均净转角 = %+7.2f 度  (%s)'
                  % (name, sum(vals) / len(vals),
                     ', '.join('%+.1f' % v for v in vals)))
        print('---')

    cam.camera_close()
    print('done')


if __name__ == '__main__':
    main()
