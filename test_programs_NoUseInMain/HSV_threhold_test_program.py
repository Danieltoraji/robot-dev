#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HSV阈值调试工具 - 多图增强版
功能：
  1. 加载指定文件夹下的所有测试图片
  2. 拖动滑动条实时调整HSV阈值
  3. 按 'N' 键切换到下一张图片，按 'B' 键切换到上一张
  4. 按 'P' 键打印当前阈值（便于复制到代码）
  5. 按 'R' 键重置阈值
  6. 按 'Q' 键退出
"""

import cv2
import numpy as np
import os
import glob

# =====================================================================
# 【请修改此处】配置你的多张测试图片路径
# =====================================================================

# 方式1：直接列出所有图片的完整路径（推荐）
IMAGE_PATHS = [
    "E:/Robot_Competition_others/robot-dev/archive/football_samples/1.png",
    "E:/Robot_Competition_others/robot-dev/archive/football_samples/2.png",
    "E:/Robot_Competition_others/robot-dev/archive/football_samples/3.png",
    "E:/Robot_Competition_others/robot-dev/archive/football_samples/4.png",
    "E:/Robot_Competition_others/robot-dev/archive/football_samples/5.png",
    "E:/Robot_Competition_others/robot-dev/archive/football_samples/6.png",
    "E:/Robot_Competition_others/robot-dev/archive/football_samples/7.png",
    "E:/Robot_Competition_others/robot-dev/archive/football_samples/8.png",
    "E:/Robot_Competition_others/robot-dev/archive/football_samples/9.png",
    "E:/Robot_Competition_others/robot-dev/archive/football_samples/11.png",
    "E:/Robot_Competition_others/robot-dev/archive/football_samples/22.png",
    "E:/Robot_Competition_others/robot-dev/archive/football_samples/33.png"
]

# 方式2：自动读取某个文件夹下的所有图片（取消注释并注释掉上面的列表）
# IMAGE_PATHS = []
# folder = "./test_images"  # 把你的图片放在这个文件夹里
# if os.path.exists(folder):
#     # 支持常见图片格式
#     for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
#         IMAGE_PATHS.extend(glob.glob(os.path.join(folder, ext)))

# 如果上面的列表为空，程序会报错退出
# =====================================================================

# HSV默认值（针对红色路径优化）
DEFAULT_H_LOW = 0
DEFAULT_H_HIGH = 17
DEFAULT_S_LOW = 0
DEFAULT_S_HIGH = 255
DEFAULT_V_LOW = 50
DEFAULT_V_HIGH = 255

DEFAULT_H2_LOW = 149
DEFAULT_H2_HIGH = 180

# 窗口名称
WINDOW_TRACKBARS = "Trackbars"
WINDOW_ORIGINAL = "Original Image"
WINDOW_MASK = "Mask (White=Path)"
WINDOW_RESULT = "Result (Path Extracted)"

def nothing(x):
    pass

def load_images(paths):
    """加载所有图片，返回有效图片列表及其文件名"""
    images = []
    names = []
    for path in paths:
        if not os.path.exists(path):
            print(f"⚠️  警告：文件不存在，已跳过 -> {path}")
            continue
        img = cv2.imread(path)
        if img is None:
            print(f"⚠️  警告：无法读取图片（可能格式损坏）-> {path}")
            continue
        images.append(img)
        names.append(os.path.basename(path))
        print(f"✅ 加载成功: {os.path.basename(path)} ({img.shape[1]}x{img.shape[0]})")
    
    if not images:
        print("\n❌ 错误：没有加载到任何有效图片！")
        print("   请检查 IMAGE_PATHS 列表中的路径是否正确。")
        return [], []
    
    return images, names

def create_trackbars():
    cv2.namedWindow(WINDOW_TRACKBARS)
    cv2.createTrackbar("H_Low", WINDOW_TRACKBARS, DEFAULT_H_LOW, 180, nothing)
    cv2.createTrackbar("H_High", WINDOW_TRACKBARS, DEFAULT_H_HIGH, 180, nothing)
    cv2.createTrackbar("S_Low", WINDOW_TRACKBARS, DEFAULT_S_LOW, 255, nothing)
    cv2.createTrackbar("S_High", WINDOW_TRACKBARS, DEFAULT_S_HIGH, 255, nothing)
    cv2.createTrackbar("V_Low", WINDOW_TRACKBARS, DEFAULT_V_LOW, 255, nothing)
    cv2.createTrackbar("V_High", WINDOW_TRACKBARS, DEFAULT_V_HIGH, 255, nothing)
    cv2.createTrackbar("H2_Low", WINDOW_TRACKBARS, DEFAULT_H2_LOW, 180, nothing)
    cv2.createTrackbar("H2_High", WINDOW_TRACKBARS, DEFAULT_H2_HIGH, 180, nothing)
    cv2.createTrackbar("Show_Info", WINDOW_TRACKBARS, 1, 1, nothing)

def get_hsv_thresholds():
    h_low = cv2.getTrackbarPos("H_Low", WINDOW_TRACKBARS)
    h_high = cv2.getTrackbarPos("H_High", WINDOW_TRACKBARS)
    s_low = cv2.getTrackbarPos("S_Low", WINDOW_TRACKBARS)
    s_high = cv2.getTrackbarPos("S_High", WINDOW_TRACKBARS)
    v_low = cv2.getTrackbarPos("V_Low", WINDOW_TRACKBARS)
    v_high = cv2.getTrackbarPos("V_High", WINDOW_TRACKBARS)
    h2_low = cv2.getTrackbarPos("H2_Low", WINDOW_TRACKBARS)
    h2_high = cv2.getTrackbarPos("H2_High", WINDOW_TRACKBARS)
    show_info = cv2.getTrackbarPos("Show_Info", WINDOW_TRACKBARS)
    
    return {
        'range1': ((h_low, s_low, v_low), (h_high, s_high, v_high)),
        'range2': ((h2_low, s_low, v_low), (h2_high, s_high, v_high)),
        'show_info': show_info
    }

def process_image(img, thresholds):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    lo1, hi1 = thresholds['range1']
    mask1 = cv2.inRange(hsv, np.array(lo1), np.array(hi1))
    
    lo2, hi2 = thresholds['range2']
    mask2 = cv2.inRange(hsv, np.array(lo2), np.array(hi2))
    
    mask = cv2.bitwise_or(mask1, mask2)
    
    # 可选：形态学降噪（如果图片噪声大，可以取消注释）
    # kernel = np.ones((3, 3), np.uint8)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    result = cv2.bitwise_and(img, img, mask=mask)
    return mask, result

def draw_info(img, thresholds, img_index, total_count, img_name):
    """在图片上叠加HSV阈值和图片信息"""
    lo1, hi1 = thresholds['range1']
    lo2, hi2 = thresholds['range2']
    
    lines = [
        f"[{img_index+1}/{total_count}] {img_name}",
        f"Range1: H[{lo1[0]:3d}-{hi1[0]:3d}] S[{lo1[1]:3d}-{hi1[1]:3d}] V[{lo1[2]:3d}-{hi1[2]:3d}]",
        f"Range2: H[{lo2[0]:3d}-{hi2[0]:3d}] S[{lo2[1]:3d}-{hi2[1]:3d}] V[{lo2[2]:3d}-{hi2[2]:3d}]",
        "Keys: [N]ext [B]ack [P]rint [R]eset [Q]uit"
    ]
    
    img_copy = img.copy()
    for i, text in enumerate(lines):
        y = 25 + i * 28
        # 黑色描边
        cv2.putText(img_copy, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.55, (0, 0, 0), 3, cv2.LINE_AA)
        # 白色文字
        cv2.putText(img_copy, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.55, (255, 255, 255), 1, cv2.LINE_AA)
    
    return img_copy

def print_thresholds(thresholds):
    lo1, hi1 = thresholds['range1']
    lo2, hi2 = thresholds['range2']
    
    print("\n" + "="*60)
    print("当前HSV阈值（可直接用于 LineDetector 初始化）:")
    print("="*60)
    print(f"hsv_ranges=[")
    print(f"    (({lo1[0]:3d}, {lo1[1]:3d}, {lo1[2]:3d}), ({hi1[0]:3d}, {hi1[1]:3d}, {hi1[2]:3d})),")
    print(f"    (({lo2[0]:3d}, {lo2[1]:3d}, {lo2[2]:3d}), ({hi2[0]:3d}, {hi2[1]:3d}, {hi2[2]:3d}))")
    print(f"]")
    print("="*60 + "\n")

def main():
    # 1. 加载所有图片
    images, names = load_images(IMAGE_PATHS)
    if not images:
        return
    
    total = len(images)
    current_idx = 0
    img = images[current_idx]
    img_name = names[current_idx]
    
    # 2. 创建窗口
    create_trackbars()
    cv2.namedWindow(WINDOW_ORIGINAL)
    cv2.namedWindow(WINDOW_MASK)
    cv2.namedWindow(WINDOW_RESULT)
    
    # 3. 显示提示信息
    print("\n" + "="*60)
    print("🎮 HSV调试工具已启动（多图模式）")
    print("="*60)
    print(f"📸 共加载 {total} 张测试图片")
    print("操作说明：")
    print("  - [N] 键：切换到下一张图片")
    print("  - [B] 键：切换到上一张图片")
    print("  - [P] 键：打印当前HSV阈值")
    print("  - [R] 键：重置阈值到默认值")
    print("  - [Q] 键：退出程序")
    print("="*60 + "\n")
    
    # 4. 主循环
    while True:
        # 获取当前阈值
        thresholds = get_hsv_thresholds()
        
        # 处理图像
        mask, result = process_image(img, thresholds)
        
        # 显示原始图（带覆盖信息）
        if thresholds['show_info']:
            display_img = draw_info(img.copy(), thresholds, current_idx, total, img_name)
            display_result = draw_info(result.copy(), thresholds, current_idx, total, img_name)
        else:
            display_img = img
            display_result = result
        
        cv2.imshow(WINDOW_ORIGINAL, display_img)
        cv2.imshow(WINDOW_MASK, mask)
        cv2.imshow(WINDOW_RESULT, display_result)
        
        # 更新窗口标题，显示当前图片序号
        cv2.setWindowTitle(WINDOW_ORIGINAL, f"Original [{current_idx+1}/{total}]")
        cv2.setWindowTitle(WINDOW_RESULT, f"Result [{current_idx+1}/{total}]")
        
        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\n👋 退出程序")
            break
        
        elif key == ord('n'):  # 下一张
            current_idx = (current_idx + 1) % total
            img = images[current_idx]
            img_name = names[current_idx]
            print(f"📸 切换到: [{current_idx+1}/{total}] {img_name}")
        
        elif key == ord('b'):  # 上一张（B=Back）
            current_idx = (current_idx - 1) % total
            img = images[current_idx]
            img_name = names[current_idx]
            print(f"📸 切换到: [{current_idx+1}/{total}] {img_name}")
        
        elif key == ord('p'):
            print_thresholds(thresholds)
        
        elif key == ord('r'):
            cv2.setTrackbarPos("H_Low", WINDOW_TRACKBARS, DEFAULT_H_LOW)
            cv2.setTrackbarPos("H_High", WINDOW_TRACKBARS, DEFAULT_H_HIGH)
            cv2.setTrackbarPos("S_Low", WINDOW_TRACKBARS, DEFAULT_S_LOW)
            cv2.setTrackbarPos("S_High", WINDOW_TRACKBARS, DEFAULT_S_HIGH)
            cv2.setTrackbarPos("V_Low", WINDOW_TRACKBARS, DEFAULT_V_LOW)
            cv2.setTrackbarPos("V_High", WINDOW_TRACKBARS, DEFAULT_V_HIGH)
            cv2.setTrackbarPos("H2_Low", WINDOW_TRACKBARS, DEFAULT_H2_LOW)
            cv2.setTrackbarPos("H2_High", WINDOW_TRACKBARS, DEFAULT_H2_HIGH)
            print("🔄 阈值已重置为默认值")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()