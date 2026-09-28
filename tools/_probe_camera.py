# -*- coding: utf-8 -*-
"""远程摄像头探针：抓一帧，检测红线，判断机器人当前视角。不做任何动作。"""
import os
import time

import cv2
import numpy as np

try:
    from Functions import RedLinePatrolV3 as redline
except ImportError:
    import RedLinePatrolV3 as redline

try:
    import hiwonder.Camera as Camera
    import hiwonder.ActionGroupControl as AGC
except ImportError:
    Camera = None
    AGC = None


def main():
    if Camera is None:
        print("PROBE_ERROR: hiwonder.Camera 不可用")
        return

    cam = Camera.Camera()
    cam.camera_open()
    time.sleep(1.5)

    if AGC is not None:
        try:
            AGC.runActionGroup('stand')
            time.sleep(1.0)
        except Exception as exc:
            print("PROBE_STAND_FAIL: {}".format(exc))

    # 多读几帧让自动曝光稳定
    last = None
    for _ in range(8):
        ret, frame = cam.read()
        if ret and frame is not None:
            last = frame
        time.sleep(0.1)

    cam.camera_close()

    if last is None:
        print("PROBE_ERROR: 读不到帧")
        return

    h, w = last.shape[:2]
    print("PROBE_FRAME: shape={}x{}".format(w, h))

    # 校正
    try:
        param_data = np.load(redline.calibration_param_path + '.npz')
        mtx = param_data['mtx_array']
        dist = param_data['dist_array']
        newcameramtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (640, 480), 0, (640, 480))
        mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (640, 480), 5)
        corrected = cv2.remap(last, mapx, mapy, cv2.INTER_LINEAR)
    except Exception as exc:
        print("PROBE_UNDISTORT_FAIL: {}".format(exc))
        corrected = last

    # 红线检测
    try:
        mask = redline.detect_red_line(corrected)
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        red_pixels = int(cv2.countNonZero(mask))
        areas = sorted([cv2.contourArea(c) for c in contours], reverse=True)[:5]
        print("PROBE_RED: pixels={} top_areas={}".format(red_pixels, [round(a, 1) for a in areas]))
        if contours:
            big = max(contours, key=cv2.contourArea)
            x, y, bw, bh = cv2.boundingRect(big)
            print("PROBE_RED_BBOX: x={} y={} w={} h={} aspect={:.2f}".format(
                x, y, bw, bh, float(bw) / max(1, bh)))
        # PCA 主方向角：竖直红线≈±90，水平横条≈0
        ys, xs = np.nonzero(mask)
        if xs.size > 300:
            mx, my = xs.mean(), ys.mean()
            xc = xs - mx
            yc = ys - my
            cxx = float((xc * xc).mean())
            cyy = float((yc * yc).mean())
            cxy = float((xc * yc).mean())
            theta = 0.5 * np.arctan2(2.0 * cxy, cxx - cyy)
            print("PROBE_RED_ANGLE: {:.1f} deg".format(np.degrees(theta)))
    except Exception as exc:
        print("PROBE_RED_FAIL: {}".format(exc))

    # 保存（/home/pi 供 pull_from_robot 拉取）
    for out in ('/tmp/probe_frame.jpg', '/home/pi/probe_frame.jpg'):
        try:
            cv2.imwrite(out, corrected)
            print("PROBE_SAVED: {} size={}".format(out, os.path.getsize(out)))
        except Exception as exc:
            print("PROBE_SAVE_FAIL {}: {}".format(out, exc))

    # 亮度统计（判断镜头朝向：天花板/地板/前方）
    gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
    print("PROBE_BRIGHTNESS: mean={:.1f} min={} max={}".format(
        float(gray.mean()), int(gray.min()), int(gray.max())))


if __name__ == '__main__':
    main()
