#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
蓝色方块 LAB 阈值标定工具（适配 apriltag_sorting_task.py 的 detect_blue）

背景：
  现有代码 detect_blue() 使用 LAB 色彩空间做颜色分割：
      frame_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
      mask = cv2.inRange(frame_lab, tuple(lab_data["blue"]["min"]),
                                     tuple(lab_data["blue"]["max"]))
  本工具用于交互式标定 lab_config.yaml 里 "blue" 的 min / max 三个值。

LAB 通道含义（OpenCV，范围均为 0~255）：
  L: 亮度（0=黑, 255=白）
  A: 绿→红（128=中性, <128 偏绿, >128 偏红）
  B: 蓝→黄（128=中性, <128 偏蓝, >128 偏黄）
  蓝色方块特征：B 通道明显小于 128（偏蓝），A 通道略低于 128。

功能：
  1. 加载测试图片（配置图片列表，或自动读取脚本同目录 test_images/ 文件夹）
  2. 6 个滑块实时调整 L/A/B 上下限
  3. 可开关"高斯模糊 + 腐蚀膨胀"，完全复刻 detect_blue 的预处理流程
  4. 鼠标点击图片任意位置，打印该点 BGR / LAB 值（快速定位蓝色范围）
  5. 自动框出 mask 最大连通域（对应"蓝色方块"），显示中心/宽高/面积
  6. 按键：
       [N] 下一张   [B] 上一张   [P] 打印阈值（可直接复制到 lab_config.yaml）
       [S] 保存截图 [R] 重置     [Q] 退出
"""

import cv2
import numpy as np
import os
import glob
import time

# =====================================================================
# 【请修改此处】配置测试图片路径
# =====================================================================

# 方式1：直接列出图片完整路径（推荐，保证按你想要的顺序显示）
IMAGE_PATHS = [
    "E:/Robot_Competition_others/lift_BlueCube/Cube_threhold/1.jpg",
    "E:/Robot_Competition_others/lift_BlueCube/Cube_threhold/2.jpg",
    "E:/Robot_Competition_others/lift_BlueCube/Cube_threhold/3.jpg",
    "E:/Robot_Competition_others/lift_BlueCube/Cube_threhold/4.jpg",
    "E:/Robot_Competition_others/lift_BlueCube/Cube_threhold/5.jpg"
]

# 方式2：IMAGE_PATHS 为空时，自动读取脚本同目录下的 test_images/ 文件夹
IMAGE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_images")

# LAB 默认阈值（蓝色：B 通道 < 128 偏蓝）
DEFAULT_L_LOW, DEFAULT_L_HIGH = 0, 255
DEFAULT_A_LOW, DEFAULT_A_HIGH = 0, 255
DEFAULT_B_LOW, DEFAULT_B_HIGH = 0, 128

# 最小轮廓面积（像素），小于此面积的最大连通域不显示，避免噪声误标
MIN_CONTOUR_AREA = 200

# 窗口名称
WINDOW_TRACKBARS = "Trackbars"
WINDOW_ORIGINAL = "Original (click to sample LAB)"
WINDOW_MASK = "Mask (White=Blue)"
WINDOW_RESULT = "Result (Blue Extracted)"


def nothing(_x):
    pass


def load_images(paths):
    """加载图片，返回 (图片列表, 文件名列表)。paths 为空时读 IMAGE_FOLDER。"""
    if not paths:
        if os.path.isdir(IMAGE_FOLDER):
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                paths.extend(sorted(glob.glob(os.path.join(IMAGE_FOLDER, ext))))
        if not paths:
            print(f"⚠️  未配置 IMAGE_PATHS，且未找到文件夹 {IMAGE_FOLDER}")
            print("    请把蓝色方块照片放进该文件夹，或在 IMAGE_PATHS 里填写路径。")
            return [], []

    images, names = [], []
    for path in paths:
        if not os.path.exists(path):
            print(f"⚠️  跳过（不存在）-> {path}")
            continue
        img = cv2.imread(path)
        if img is None:
            print(f"⚠️  跳过（无法读取）-> {path}")
            continue
        images.append(img)
        names.append(os.path.basename(path))
        print(f"✅ 加载: {os.path.basename(path)} ({img.shape[1]}x{img.shape[0]})")

    if not images:
        print("❌ 没有加载到任何有效图片。")
    return images, names


def create_trackbars():
    cv2.namedWindow(WINDOW_TRACKBARS, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("L_Low", WINDOW_TRACKBARS, DEFAULT_L_LOW, 255, nothing)
    cv2.createTrackbar("L_High", WINDOW_TRACKBARS, DEFAULT_L_HIGH, 255, nothing)
    cv2.createTrackbar("A_Low", WINDOW_TRACKBARS, DEFAULT_A_LOW, 255, nothing)
    cv2.createTrackbar("A_High", WINDOW_TRACKBARS, DEFAULT_A_HIGH, 255, nothing)
    cv2.createTrackbar("B_Low", WINDOW_TRACKBARS, DEFAULT_B_LOW, 255, nothing)
    cv2.createTrackbar("B_High", WINDOW_TRACKBARS, DEFAULT_B_HIGH, 255, nothing)
    # 预处理开关: 0=关, 1=开(复刻 detect_blue 的高斯模糊+腐蚀膨胀)
    cv2.createTrackbar("Preprocess", WINDOW_TRACKBARS, 1, 1, nothing)


def get_thresholds():
    l_low = cv2.getTrackbarPos("L_Low", WINDOW_TRACKBARS)
    l_high = cv2.getTrackbarPos("L_High", WINDOW_TRACKBARS)
    a_low = cv2.getTrackbarPos("A_Low", WINDOW_TRACKBARS)
    a_high = cv2.getTrackbarPos("A_High", WINDOW_TRACKBARS)
    b_low = cv2.getTrackbarPos("B_Low", WINDOW_TRACKBARS)
    b_high = cv2.getTrackbarPos("B_High", WINDOW_TRACKBARS)
    preprocess = cv2.getTrackbarPos("Preprocess", WINDOW_TRACKBARS)
    return (l_low, a_low, b_low), (l_high, a_high, b_high), bool(preprocess)


def process_image(img, lo, hi, preprocess):
    """复刻 detect_blue: LAB 分割 + 可选预处理。返回 (mask, result, 最大轮廓信息)。"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    if preprocess:
        lab = cv2.GaussianBlur(lab, (3, 3), 3)

    mask = cv2.inRange(lab, np.array(lo), np.array(hi))

    if preprocess:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.dilate(cv2.erode(mask, k), k)

    result = cv2.bitwise_and(img, img, mask=mask)

    contour_info = None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area >= MIN_CONTOUR_AREA:
            x, y, w, h = cv2.boundingRect(largest)
            cx, cy = x + w // 2, y + h // 2
            contour_info = (x, y, w, h, cx, cy, area)
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.circle(result, (cx, cy), 5, (0, 255, 255), -1)

    return mask, result, contour_info


def draw_overlay(img, lo, hi, idx, total, name, contour_info, preprocess):
    """在图上叠加阈值 / 图片信息 / 最大轮廓信息。"""
    out = img.copy()
    lines = [
        f"[{idx + 1}/{total}] {name}",
        f"L[{lo[0]:3d}-{hi[0]:3d}] A[{lo[1]:3d}-{hi[1]:3d}] B[{lo[2]:3d}-{hi[2]:3d}]  pre={'on' if preprocess else 'off'}",
        "Keys: [N]ext [B]ack [P]rint [S]ave [R]eset [Q]uit",
    ]
    if contour_info is not None:
        x, y, w, h, cx, cy, area = contour_info
        lines.append(f"cube: center=({cx},{cy}) size={w}x{h} area={area:.0f}")
    for i, text in enumerate(lines):
        yy = 25 + i * 28
        cv2.putText(out, text, (12, yy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, (10, yy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def print_thresholds(lo, hi):
    print("\n" + "=" * 60)
    print("当前蓝色 LAB 阈值（可直接复制到 lab_config.yaml 的 blue 项）:")
    print("=" * 60)
    print("blue:")
    print(f"  min: [{lo[0]}, {lo[1]}, {lo[2]}]")
    print(f"  max: [{hi[0]}, {hi[1]}, {hi[2]}]")
    print()
    print("对应代码调用:")
    print(f"  mask = cv2.inRange(lab, ({lo[0]},{lo[1]},{lo[2]}), ({hi[0]},{hi[1]},{hi[2]}))")
    print("=" * 60 + "\n")


def on_mouse(event, x, y, _flags, _param):
    """点击时打印该点 BGR/LAB 值，帮助确定阈值范围。"""
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    img = on_mouse.img
    if img is None or x >= img.shape[1] or y >= img.shape[0]:
        return
    b, g, r = img[y, x]
    lab = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2LAB)[0][0]
    l, a, bb = lab
    print(f"🖱️  点({x:4d},{y:4d})  BGR=({int(b):3d},{int(g):3d},{int(r):3d})  "
          f"LAB=({int(l):3d},{int(a):3d},{int(bb):3d})")


def reset_trackbars():
    cv2.setTrackbarPos("L_Low", WINDOW_TRACKBARS, DEFAULT_L_LOW)
    cv2.setTrackbarPos("L_High", WINDOW_TRACKBARS, DEFAULT_L_HIGH)
    cv2.setTrackbarPos("A_Low", WINDOW_TRACKBARS, DEFAULT_A_LOW)
    cv2.setTrackbarPos("A_High", WINDOW_TRACKBARS, DEFAULT_A_HIGH)
    cv2.setTrackbarPos("B_Low", WINDOW_TRACKBARS, DEFAULT_B_LOW)
    cv2.setTrackbarPos("B_High", WINDOW_TRACKBARS, DEFAULT_B_HIGH)
    cv2.setTrackbarPos("Preprocess", WINDOW_TRACKBARS, 1)
    print("🔄 阈值已重置为默认值")


def main():
    images, names = load_images(IMAGE_PATHS)
    if not images:
        return

    total = len(images)
    current_idx = 0
    img = images[current_idx]
    img_name = names[current_idx]

    on_mouse.img = img
    create_trackbars()
    cv2.namedWindow(WINDOW_ORIGINAL)
    cv2.namedWindow(WINDOW_MASK)
    cv2.namedWindow(WINDOW_RESULT)
    cv2.setMouseCallback(WINDOW_ORIGINAL, on_mouse)

    print("\n" + "=" * 60)
    print("🎮 蓝色方块 LAB 标定工具已启动")
    print("=" * 60)
    print(f"📸 共 {total} 张图片")
    print("  [N] 下一张   [B] 上一张   [P] 打印阈值")
    print("  [S] 保存截图 [R] 重置     [Q] 退出")
    print("  🖱️  在 Original 窗口点击可查看该点 BGR/LAB 值")
    print("  💡 Preprocess 滑块=1 时复刻 detect_blue 的高斯模糊+腐蚀膨胀")
    print("=" * 60 + "\n")

    while True:
        lo, hi, preprocess = get_thresholds()
        mask, result, contour_info = process_image(img, lo, hi, preprocess)

        cv2.imshow(WINDOW_ORIGINAL, draw_overlay(img, lo, hi, current_idx, total, img_name, contour_info, preprocess))
        cv2.imshow(WINDOW_MASK, mask)
        cv2.imshow(WINDOW_RESULT, result)
        cv2.setWindowTitle(WINDOW_ORIGINAL, f"Original [{current_idx + 1}/{total}]")
        cv2.setWindowTitle(WINDOW_RESULT, f"Result [{current_idx + 1}/{total}]")

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("👋 退出")
            break
        elif key == ord('n'):
            current_idx = (current_idx + 1) % total
            img, img_name = images[current_idx], names[current_idx]
            on_mouse.img = img
            print(f"📸 切换: [{current_idx + 1}/{total}] {img_name}")
        elif key == ord('b'):
            current_idx = (current_idx - 1) % total
            img, img_name = images[current_idx], names[current_idx]
            on_mouse.img = img
            print(f"📸 切换: [{current_idx + 1}/{total}] {img_name}")
        elif key == ord('p'):
            print_thresholds(lo, hi)
        elif key == ord('s'):
            save_dir = os.path.dirname(os.path.abspath(__file__))
            stamp = time.strftime("%Y%m%d_%H%M%S")
            snap = os.path.join(save_dir, f"blue_lab_calib_{stamp}.jpg")
            cv2.imwrite(snap, result)
            print(f"💾 已保存截图 -> {snap}")
        elif key == ord('r'):
            reset_trackbars()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
