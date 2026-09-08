#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_dataset_from_labelwork.py —— X-AnyLabeling 标注成果 → YOLO 训练集

输入：--src（label_work 目录，images/ 内含 jpg 与 labelme 风格 json）
输出：--dst（标准 YOLO 布局 images/+labels/{train,val,test} + data.yaml）

规则：
  - 场景组从文件名解析（如 ball_1250r_right_1250_003.jpg → ball_1250r），
    同场景组整体进同一 split（防连拍泄漏）；split 归属由 SCENE_SPLIT 硬编码；
  - 无 json 的图 = 空标注（负样本）；
  - 标签名归一化为小写后必须 ∈ LABELS，非法标签的框剔除并告警；
  - 框坐标裁剪到图像边界，宽高 <0.001 的退化框剔除并告警。
"""

import argparse
import glob
import json
import os
import re
import shutil
import sys
import zipfile

LABEL = "football"
SCENE_SPLIT = {
    # train：近中距 + 左侧（主体）
    "ball_1040c": "train", "ball_1040c_neg": "train",
    "ball_1150c": "train", "ball_1150c_neg": "train",
    "ball_1250c": "train", "ball_1250c_neg": "train",
    "ball_1250l": "train", "ball_1250l_neg": "train",
    # val：右侧（侧向泛化检查）
    "ball_1250r": "val", "ball_1250r_neg": "val",
    # test：1350 远距（小目标检查）
    "ball_1350c": "test", "ball_1350c_neg": "test",
}
YAW_WORDS = ("center", "right", "left", "wide_right", "wide_left")


def scene_of(stem):
    """ball_1250r_right_1250_003 → ball_1250r；ball_1040c_neg_center_1040_001 → ball_1040c_neg"""
    parts = stem.split("_")
    if len(parts) < 3 or parts[0] != "ball":
        return None
    scene = f"ball_{parts[1]}"
    if len(parts) >= 4 and parts[2] == "neg":
        scene += "_neg"
    return scene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="datasets/label_work")
    ap.add_argument("--dst", default="datasets/football_goal")
    ap.add_argument("--zip", action="store_true", help="完成后打包 dst 为 zip")
    args = ap.parse_args()

    img_dir = os.path.join(args.src, "images")
    jpgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    if not jpgs:
        print("错误：src/images 下没有 jpg")
        sys.exit(1)

    stats, warnings = {}, []
    seen_scenes = set()
    n_box_total = 0
    items = []  # (jpg_path, scene, split, boxes)

    for jpg in jpgs:
        stem = os.path.splitext(os.path.basename(jpg))[0]
        scene = scene_of(stem)
        if scene not in SCENE_SPLIT:
            warnings.append(f"[场景未知] {stem} -> 解析为 '{scene}'，未纳入数据集")
            continue
        seen_scenes.add(scene)
        split = SCENE_SPLIT[scene]

        jsn = os.path.join(img_dir, stem + ".json")
        boxes, labels_seen = [], set()
        if os.path.exists(jsn):
            data = json.load(open(jsn, encoding="utf-8"))
            img = None  # 尺寸懒加载（仅在需要裁剪/校验时读图）
            W = H = None
            for shp in data.get("shapes", []):
                lbl = str(shp.get("label", "")).strip().lower()
                if lbl != LABEL:
                    labels_seen.add(lbl or "<空>")
                    continue
                pts = shp.get("points", [])
                if shp.get("shape_type") != "rectangle" or len(pts) < 2:
                    warnings.append(f"[格式异常] {stem}: 非矩形/点数异常的框被剔除")
                    continue
                if W is None:
                    import cv2
                    im = cv2.imread(jpg)
                    H, W = im.shape[:2]
                # 兼容两种存法：2 点对角线（labelme 经典）/ 4 点四角（X-AnyLabeling v4）
                xs = [float(p[0]) for p in pts]
                ys = [float(p[1]) for p in pts]
                x1, x2 = max(0.0, min(xs)), min(float(W), max(xs))
                y1, y2 = max(0.0, min(ys)), min(float(H), max(ys))
                bw, bh = x2 - x1, y2 - y1
                if bw < 1 or bh < 1:
                    warnings.append(f"[退化框] {stem}: 宽高<1px 被剔除")
                    continue
                boxes.append(((x1 + bw / 2) / W, (y1 + bh / 2) / H,
                              bw / W, bh / H))
            if labels_seen:
                warnings.append(f"[非 football 标签] {stem}: {sorted(labels_seen)}（框已剔除）")

        n = len(boxes)
        n_box_total += n
        key = (scene, split)
        stats.setdefault(key, [0, 0])
        stats[key][0] += 1
        stats[key][1] += n
        # 正样本场景 0 框告警（可能有意为之，仅提示）
        if n == 0 and "_neg" not in scene:
            warnings.append(f"[正样本 0 框] {stem}（确认有意当负样本则忽略）")
        items.append((jpg, scene, split, boxes))

    # 组装输出目录
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(args.dst, "images", split), exist_ok=True)
        os.makedirs(os.path.join(args.dst, "labels", split), exist_ok=True)
    for jpg, scene, split, boxes in items:
        stem = os.path.splitext(os.path.basename(jpg))[0]
        shutil.copy2(jpg, os.path.join(args.dst, "images", split,
                                       os.path.basename(jpg)))
        with open(os.path.join(args.dst, "labels", split, stem + ".txt"),
                  "w", encoding="utf-8") as f:
            for xc, yc, bw, bh in boxes:
                f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    with open(os.path.join(args.dst, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(f"path: /root/datasets/football_goal\n"
                f"train: images/train\nval: images/val\ntest: images/test\n\n"
                f"names:\n  0: {LABEL}\n")

    # 报告
    print("=" * 62)
    print(f"{'场景组':<18}{'split':<7}{'图数':>5}{'框数':>6}")
    for (scene, split) in sorted(stats, key=lambda k: (k[1], k[0])):
        n_img, n_box = stats[(scene, split)]
        print(f"{scene:<18}{split:<7}{n_img:>5}{n_box:>6}")
    missing = set(SCENE_SPLIT) - seen_scenes
    tot = {s: (0, 0) for s in ("train", "val", "test")}
    for (scene, split), (n_img, n_box) in stats.items():
        tot[split] = (tot[split][0] + n_img, tot[split][1] + n_box)
    print("-" * 62)
    for s in ("train", "val", "test"):
        print(f"{s}: {tot[s][0]} 图 / {tot[s][1]} 框")
    print(f"合计: {sum(t[0] for t in tot.values())} 图 / {n_box_total} 框")
    if missing:
        print(f"[注意] 预期场景组缺失（全被删光或未拍）: {sorted(missing)}")
    if warnings:
        print(f"\n告警 {len(warnings)} 条:")
        for w in warnings:
            print(" ", w)

    if args.zip:
        base = os.path.dirname(args.dst.rstrip("/"))
        zip_path = os.path.join(base, "football_goal.zip")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for root, _dirs, files in os.walk(args.dst):
                for fn in files:
                    full = os.path.join(root, fn)
                    z.write(full, os.path.relpath(full, base))
        print(f"\n已打包: {zip_path} ({os.path.getsize(zip_path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
