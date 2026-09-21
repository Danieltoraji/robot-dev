#!/usr/bin/python3
# coding=utf8
"""相机标定配置（单一真源）。

提供：
    calibration_param_path : 标定 npz 路径（不带 .npz 后缀）
    CAMERA_INTRINSIC       : 实际输出分辨率下的内参矩阵 (3, 3)
    CAMERA_DISTORTION      : 畸变系数 (k1, k2, p1, p2)

标定原始数据由 fswebcam -r 2592x1944 拍照标定得到；机器人实时视频流实际
输出 640x480，两者为等比缩放（2592/640 = 1944/480 = 4.05），故将内参按
比例缩放到 640x480（畸变系数在归一化坐标下与分辨率无关，保持不变）。

calibration_param.npz 由本文件数据生成，键为 mtx_array / dist_array。
"""

import os

import numpy as np

# ---- 标定拍照分辨率与机器人实际输出分辨率 ----
CALIB_WIDTH, CALIB_HEIGHT = 2592, 1944   # fswebcam 标定拍照分辨率
CAMERA_WIDTH, CAMERA_HEIGHT = 640, 480   # 机器人实时视频流输出分辨率

# ---- 原始标定内参（2592x1944 分辨率下）----
_CALIB_INTRINSIC = np.array(
    (
        [1.944903664123011e03, 0, 1.283069051100245e03],
        [0, 1.950095436307893e03, 9.831983420212778e02],
        [0, 0, 1],
    ),
    dtype=np.double,
)

# ---- 畸变系数 (k1, k2, p1, p2)，归一化坐标下与分辨率无关 ----
CAMERA_DISTORTION = np.array(
    [-0.384402275498781, 0.284681889150075, 0, 0], dtype=np.double
)

# ---- 内参缩放到实际输出分辨率 640x480 ----
_scale_x = float(CAMERA_WIDTH) / float(CALIB_WIDTH)
_scale_y = float(CAMERA_HEIGHT) / float(CALIB_HEIGHT)
CAMERA_INTRINSIC = _CALIB_INTRINSIC.copy()
CAMERA_INTRINSIC[0, 0] *= _scale_x
CAMERA_INTRINSIC[1, 1] *= _scale_y
CAMERA_INTRINSIC[0, 2] *= _scale_x
CAMERA_INTRINSIC[1, 2] *= _scale_y


# 使用相对路径随文件自动定位，部署到任何目录都能找到标定文件。
calibration_param_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "calibration_param"
)
