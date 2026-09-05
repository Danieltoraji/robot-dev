#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
split_dataset.py —— 标注数据按场景划分 train/val/test（YOLO detect 目录结构）

输入目录（--src）两种形态都支持：
    1. 场景子目录（capture_dataset.py 的输出，推荐）：
        src/<场景名>/*.jpg + 同名 .txt；场景键 = 子目录名
    2. 平铺：src/*.jpg + 同名 .txt；
       场景键 = 文件名第一个下划线前的字段（ball_near_center_001.jpg → ball）

同一场景的所有图整体进同一 split，避免同一连拍同时进 train 与 val/test
造成指标虚高（《足球与球门识别训练部署方案》§3.4 划分原则）。
负样本（无目标图）允许没有 .txt，划分时自动补空标签。

输出（--dst）：
    dst/images/{train,val,test}/  dst/labels/{train,val,test}/

用法：
    python tools/train/split_dataset.py --src /data/labeled --dst datasets/football_goal
    python tools/train/split_dataset.py --src /data/labeled --within-scene  # 场景数很少时场景内按比例切
"""

import argparse
import os
import random
import shutil
import sys

IMG_EXTS = (".jpg", ".jpeg", ".png")
SPLITS = ("train", "val", "test")


def find_image_label_pairs(src):
    """遍历 src，返回 [(image_path, label_path 或 None), ...]"""
    pairs = []
    for root, _dirs, files in os.walk(src):
        for name in sorted(files):
            if not name.lower().endswith(IMG_EXTS):
                continue
            img = os.path.join(root, name)
            label = os.path.splitext(img)[0] + ".txt"
            pairs.append((img, label if os.path.exists(label) else None))
    return pairs


def scene_key(img_path, src):
    """场景键：子目录名；平铺时取文件名首个下划线前的字段"""
    rel = os.path.relpath(img_path, src)
    parts = rel.split(os.sep)
    if len(parts) > 1:
        return parts[0]
    return os.path.basename(rel).split("_")[0]


def assign_splits(pairs, src, val_ratio, test_ratio, seed, within_scene):
    """返回 [(img, label, split), ...]；组级划分（默认）或组内按比例划分"""
    rng = random.Random(seed)
    groups = {}
    for img, label in pairs:
        groups.setdefault(scene_key(img, src), []).append((img, label))

    assignment = []
    for scene in sorted(groups):
        items = sorted(groups[scene])
        if within_scene:
            # 场景数很少（如 <5 个）时兜底：场景内按比例随机切
            rng.shuffle(items)
            n = len(items)
            n_val = int(round(n * val_ratio))
            n_test = int(round(n * test_ratio))
            splits = (["val"] * n_val + ["test"] * n_test
                      + ["train"] * (n - n_val - n_test))
            assignment.extend((img, label, s) for (img, label), s in zip(items, splits))
        else:
            r = rng.random()
            if r < test_ratio:
                s = "test"
            elif r < test_ratio + val_ratio:
                s = "val"
            else:
                s = "train"
            assignment.extend((img, label, s) for img, label in items)
            print(f"场景 {scene}: {len(items)} 张 -> {s}")
    return assignment


def materialize(assignment, dst):
    """复制图像与标签到 YOLO 目录结构，返回各 split 计数与补空标签数"""
    counts = {s: 0 for s in SPLITS}
    created_empty_labels = 0
    for img, label, split in assignment:
        scene = os.path.basename(os.path.dirname(img))
        name = os.path.basename(img)
        # 平铺/重名兜底：目标名不含场景前缀时补上，防跨场景同名覆盖
        if scene and not name.startswith(scene + "_"):
            name = f"{scene}_{name}"
        os.makedirs(os.path.join(dst, "images", split), exist_ok=True)
        os.makedirs(os.path.join(dst, "labels", split), exist_ok=True)
        shutil.copy2(img, os.path.join(dst, "images", split, name))
        if label is None:
            label = os.path.splitext(img)[0] + ".txt"
            with open(label, "w", encoding="utf-8"):
                pass  # 负样本补空标签
            created_empty_labels += 1
        shutil.copy2(label, os.path.join(dst, "labels", split, name))
        counts[split] += 1
    return counts, created_empty_labels


def main():
    parser = argparse.ArgumentParser(description="标注数据按场景划分 train/val/test")
    parser.add_argument("--src", required=True, help="标注目录（场景子目录或平铺 jpg+txt）")
    parser.add_argument("--dst", default="datasets/football_goal", help="输出根目录")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="val 比例（默认 0.1）")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="test 比例（默认 0.1）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认 42，保证可复现）")
    parser.add_argument("--within-scene", action="store_true",
                        help="场景内按比例切（场景数很少时用；默认整场景同 split）")
    args = parser.parse_args()

    pairs = find_image_label_pairs(args.src)
    if not pairs:
        print(f"错误: {args.src} 下没有找到图片")
        sys.exit(1)

    print(f"共 {len(pairs)} 张（其中 {sum(1 for _, l in pairs if l is None)} 张无标签，将按负样本补空标签）")
    assignment = assign_splits(pairs, args.src, args.val_ratio, args.test_ratio,
                               args.seed, args.within_scene)
    counts, created = materialize(assignment, args.dst)

    print("-" * 40)
    for s in SPLITS:
        print(f"{s}: {counts[s]} 张")
    print(f"已补空标签（负样本）: {created} 张 -> {args.dst}")
    if counts["val"] == 0 or counts["test"] == 0:
        print("[WARN] val/test 为空：场景数太少导致整场景划分落空，"
              "请增加场景数或改用 --within-scene")


if __name__ == "__main__":
    main()
