# -*- coding: utf-8 -*-
"""
一键诊断：AprilTag 角点顺序 + 通用 solvePnP

在机器人上直接运行：
    python -m tools.field_calib.debug_pnp_permutations

它会：
  1. 拍一张照片
  2. 打印实际图像分辨率 image.shape
  3. 检测 AprilTag
  4. 对每个已知 tag 尝试 8 种角点排列（4 旋转 + 4 镜像）
  5. 用通用 cv2.solvePnP 求解并打印重投影误差、相机位置、相机朝向
  6. 按“重投影误差小 + 位置在场地内 + 高度合理 + 朝向水平”给一个最佳候选
"""

import time
import subprocess

import cv2
import numpy as np
import apriltag


CAMERA_INTRINSIC = np.array(
    (
        [1.944903664123011e03, 0, 1.283069051100245e03],
        [0, 1.950095436307893e03, 9.831983420212778e02],
        [0, 0, 1],
    ),
    dtype=np.double,
)

CAMERA_DISTORTION = np.array([-0.384402275498781, 0.284681889150075, 0, 0])


def load_tag_pos():
    tag_poses = {}
    tag_poses['36'] = np.array(
        [[44.7, 23.8, 41.5], [44.7, 18.8, 41.5], [44.7, 18.8, 36.5], [44.7, 23.8, 36.5]],
        dtype=np.float64,
    )
    tag_poses['37'] = np.array(
        [[20.6, 95.5, 41.7], [25.6, 95.5, 41.7], [25.6, 95.5, 36.7], [20.6, 95.5, 36.7]],
        dtype=np.float64,
    )
    tag_poses['38'] = np.array(
        [[95, 82.2, 41.7], [95, 77.2, 41.7], [95, 77.2, 36.7], [95, 82.2, 36.7]],
        dtype=np.float64,
    )
    tag_poses['39'] = np.array(
        [[76.5, 0.5, 42.1], [71.5, 0.5, 42.1], [71.5, 0.5, 37.1], [76.5, 0.5, 37.1]],
        dtype=np.float64,
    )
    return tag_poses


def capture_image():
    """拍照，返回文件路径；失败返回 None"""
    timestamp = int(time.time())
    filename = f"/home/pi/Pictures/photo_{timestamp}.jpg"
    cmd = f"fswebcam -r 2592x1944 --no-banner -S 3 {filename}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"拍照失败: {result.stderr}")
        return None

    print(f"照片已保存: {filename}")
    return filename


def detect_apriltag(filename):
    """检测图片中的 AprilTag，返回检测结果列表"""
    print("[INFO] loading image...")
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print("[INFO] detecting AprilTags...")
    options = apriltag.DetectorOptions(families="tag36h11")
    detector = apriltag.Detector(options)
    results = detector.detect(gray)
    print("[INFO] {} total AprilTags detected".format(len(results)))
    return results


def main():
    tag_poses = load_tag_pos()

    filename = capture_image()
    if filename is None:
        print("无法拍摄照片，程序终止。")
        return

    image = cv2.imread(filename)
    if image is None:
        print("读图失败，程序终止。")
        return

    h, w = image.shape[:2]
    print(f"image.shape = {image.shape}")
    if (w, h) != (2592, 1944):
        print(f"[WARN] 实际分辨率 ({w}x{h}) 与内参对应的 2592x1944 不一致！")

    results = detect_apriltag(filename)
    if len(results) == 0:
        print("没有检测到 AprilTag，无法诊断。")
        return

    for r in results:
        tag_id = str(r.tag_id)
        if tag_id not in tag_poses:
            print(f"[WARN] 跳过未知 AprilTag ID: {tag_id}")
            continue

        corners_world = tag_poses[tag_id]
        corners_img = np.asarray(r.corners, dtype=np.float64)

        print(f"\n===== Tag {tag_id} =====")
        print("corners_img =")
        print(corners_img)

        base_idx = np.arange(4)
        permutations = []
        for k in range(4):
            permutations.append(np.roll(base_idx, k))
        for k in range(4):
            permutations.append(np.roll(base_idx[::-1], k))

        best = None
        best_score = float("inf")

        for pi, perm in enumerate(permutations):
            img = corners_img[perm]

            ok, rvec, tvec = cv2.solvePnP(
                corners_world,
                img,
                CAMERA_INTRINSIC,
                CAMERA_DISTORTION,
            )
            if not ok:
                continue

            proj, _ = cv2.projectPoints(
                corners_world,
                rvec,
                tvec,
                CAMERA_INTRINSIC,
                CAMERA_DISTORTION,
            )
            err = float(np.mean(np.linalg.norm(proj[:, 0, :] - img, axis=1)))

            R = cv2.Rodrigues(rvec)[0]
            pos = -np.linalg.inv(R) @ tvec
            ori = np.linalg.inv(R) @ (np.array([[0], [0], [1]]) - tvec) - pos

            pos_flat = pos.flatten()
            ori_flat = ori.flatten()

            print(f"perm {pi}: {perm}")
            print(f"  reproj_err = {err:.3f}")
            print(f"  pos        = {pos_flat}")
            print(f"  ori        = {ori_flat}")

            x, y, z = pos_flat
            score = err
            if not (-10 <= x <= 110 and -10 <= y <= 110 and 5 <= z <= 80):
                score += 1000
            # 相机高度更可能在 10~50cm，过高/过低都要扣分
            if not (10 <= z <= 50):
                score += 500
            # 相机朝向应接近水平，z 分量过大说明解不合理
            if abs(ori_flat[2]) > 0.3:
                score += 200
            if np.linalg.norm(ori_flat[:2]) < 0.1:
                score += 1000
            if err > 100:
                score += 1000

            if score < best_score:
                best_score = score
                best = (pi, perm, err, pos_flat, ori_flat)

        if best is not None:
            pi, perm, err, pos, ori = best
            print(f"\n>>> 最佳候选: perm {pi} {perm}")
            print(f"    reproj_err = {err:.3f}")
            print(f"    pos        = {pos}")
            print(f"    ori        = {ori}")
        else:
            print("\n>>> 没有找到合理候选")


if __name__ == "__main__":
    main()
