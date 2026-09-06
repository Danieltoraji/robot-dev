#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
capture_dataset.py —— 足球/球门数据集批量采集（机器人端，脚本驱动头部舵机）

按《足球与球门识别训练部署方案》§3 的覆盖矩阵批量拍照：fswebcam 静态照
（与定位链路同款命令，2592x1944 + -S 3 稳曝光），输出到
    <out>/<场景>/<场景>_<偏航档>_<俯仰脉宽>_<序号>.jpg

脚本直接驱动头部舵机（偏航 ID2 走 RobotState.set_head 动态等待；俯仰 ID1
固定 500ms），一条命令即可按 --head 顺序扫过多个偏航档位，无需人工摆位。
SDK 不可用或显式 --manual 时退回"人工摆位"模式：脚本只记录档位标签，
每个档位拍照前提示确认。

覆盖矩阵与拍摄 checklist 见 docs/视觉能力/足球球门数据采集规范.md。

用法（机器人上，项目根目录）：
    # 单档：中位偏航 + 俯仰 1040，拍 12 张
    python3 tools/data_collect/capture_dataset.py --scene ball_050 --count 12 --pitch 1040
    # 一条命令覆盖左右扫（每个偏航档各拍 count 张）
    python3 tools/data_collect/capture_dataset.py --scene ball_050 --count 6 --pitch 1150 --head center right left
    # 宽扫
    python3 tools/data_collect/capture_dataset.py --scene ball_100 --count 4 --pitch 1350 --head wide_right wide_left

俯仰为舵机脉宽：1500=水平，越小越低头（实测 1040=最近地面点、
1350=看穿全场球门，实用带 1040~1350，可调约 950~2000）。
"""

import argparse
import os
import subprocess
import sys
import time

# 允许直接运行本文件
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.camera_config import (
    CAMERA_WIDTH, CAMERA_HEIGHT,
    HEAD_CENTER, HEAD_RIGHT, HEAD_LEFT, HEAD_WIDE_RIGHT, HEAD_WIDE_LEFT,
)

YAW_GEARS = {"center": HEAD_CENTER, "right": HEAD_RIGHT, "left": HEAD_LEFT,
             "wide_right": HEAD_WIDE_RIGHT, "wide_left": HEAD_WIDE_LEFT}
PITCH_MIN, PITCH_MAX = 500, 2500      # 舵机硬限
PITCH_PRACTICAL = (950, 2000)         # 实测实用范围
PITCH_CMD_MS = 500                    # 俯仰转动指令时长
PITCH_WAIT_S = 0.8                    # 俯仰转动余量等待


def init_head_control():
    """初始化头部舵机控制；SDK 不可用返回 (None, None)，退回人工摆位模式"""
    try:
        from core.robot_core import RobotState, ctl
        if ctl is None:
            print("[WARN] 舵机 SDK 不可用（core.robot_core.ctl 为 None），退回人工摆位模式")
            return None, None
        state = RobotState()
        # set_head 有"目标==当前则跳过"逻辑，而初始值 HEAD_CENTER 是假设值；
        # 置 0 强制首个指令真实下发（物理头部可能停在任意位置）
        state.current_head_pulse = 0
        return state, ctl
    except Exception as e:
        print(f"[WARN] 头部舵机控制初始化失败（{e}），退回人工摆位模式")
        return None, None


def set_pitch(ctl, pulse, last_pulse):
    """俯仰舵机（ID1）：与上次目标相同则跳过，否则转动并等待到位"""
    if pulse == last_pulse:
        return pulse
    ctl.set_pwm_servo_pulse(1, pulse, PITCH_CMD_MS)
    time.sleep(PITCH_WAIT_S)
    return pulse


def capture_one(path):
    """拍一张并校验文件真实有效，成功返回 True；失败会删除无效文件防污染数据集"""
    cmd = (f"fswebcam -r {CAMERA_WIDTH}x{CAMERA_HEIGHT} "
           f"--no-banner -S 3 {path}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    def _fail(reason):
        print(f"  拍照异常：{reason} -> {path}")
        print("  （相机可能被占用：检查视频流/预览工具是否已关，或镜头未就绪）")
        if os.path.exists(path):
            os.remove(path)   # 删除无效文件，防止 0 字节/截断废片混进数据集
        return False

    if result.returncode != 0:
        return _fail(f"fswebcam 退出码 {result.returncode}: {result.stderr.strip()}")
    # 真机踩坑（2026-09-05）：相机被占用时 fswebcam 可能返回 0 但留下空文件/截断文件
    if not os.path.exists(path) or os.path.getsize(path) < 1024:
        return _fail("文件未生成或过小")
    with open(path, "rb") as fp:
        fp.seek(-2, os.SEEK_END)
        if fp.read(2) != b"\xff\xd9":
            return _fail("JPEG 结束标记缺失（文件截断）")
    try:
        import cv2
        if cv2.imread(path) is None:
            return _fail("文件无法解码")
    except ImportError:
        pass  # 无 cv2 时以上述标记检查为准
    return True


def resolve_start(scene_dir, scene, gear, pitch, start_arg):
    """--start 未指定时自动续号：接场景目录中同档位已有照片的最大序号继续

    这样同一命令可反复执行（换球位/机器人位后重跑），文件名自动接续，
    不需要人工数序号。显式传 --start 时以传入值为准。
    """
    if start_arg is not None:
        return start_arg
    prefix = f"{scene}_{gear}_{pitch}_"
    max_idx = 0
    if not os.path.isdir(scene_dir):
        return 1
    for name in os.listdir(scene_dir):
        if name.startswith(prefix) and name.endswith(".jpg"):
            try:
                max_idx = max(max_idx, int(name[len(prefix):-4]))
            except ValueError:
                continue
    return max_idx + 1


def shoot_gear(scene_dir, scene, gear, pitch, count, interval, start):
    """在当前头部档位下拍 count 张，返回成功张数"""
    ok = 0
    for i in range(count):
        idx = start + i
        path = os.path.join(scene_dir, f"{scene}_{gear}_{pitch}_{idx:03d}.jpg")
        print(f"[{gear} {pitch}] {idx} -> {path}")
        if capture_one(path):
            ok += 1
        if i < count - 1 and interval > 0:
            time.sleep(interval)
    return ok


def main():
    parser = argparse.ArgumentParser(description="足球/球门数据集批量采集（脚本驱动头部舵机）")
    parser.add_argument("--scene", required=True,
                        help="场景名（子目录与文件名前缀，如 ball_050 / goal_100 / negative）")
    parser.add_argument("--count", type=int, default=10,
                        help="每个偏航档位拍照张数（默认 10）")
    parser.add_argument("--head", nargs="+", choices=sorted(YAW_GEARS),
                        default=["center"],
                        help="偏航档位序列，逐档拍摄（默认 center；左右扫用 --head center right left）")
    parser.add_argument("--pitch", type=int, default=1040,
                        help=f"俯仰舵机脉宽（默认 1040；1500=水平，越小越低头，"
                             f"实用 {PITCH_PRACTICAL[0]}~{PITCH_PRACTICAL[1]}）")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="相邻两张间隔秒（默认 1.0，给摆拍微扰留时间）")
    parser.add_argument("--out", default="/home/pi/dataset_raw", help="输出根目录")
    parser.add_argument("--start", type=int, default=None,
                        help="起始序号（默认自动续号：接该档已有照片继续编号）")
    parser.add_argument("--manual", action="store_true",
                        help="强制人工摆位模式（不驱动舵机，仅记录档位标签）")
    args = parser.parse_args()

    if not (PITCH_MIN <= args.pitch <= PITCH_MAX):
        print(f"错误: 俯仰脉宽 {args.pitch} 超出舵机范围 {PITCH_MIN}~{PITCH_MAX}")
        sys.exit(1)
    if not (PITCH_PRACTICAL[0] <= args.pitch <= PITCH_PRACTICAL[1]):
        print(f"[WARN] 俯仰 {args.pitch} 在实用带 {PITCH_PRACTICAL} 之外，请确认是否有意如此")

    if args.manual:
        state = ctl = None
    else:
        state, ctl = init_head_control()
    servo_ok = state is not None

    scene_dir = os.path.join(args.out, args.scene)
    os.makedirs(scene_dir, exist_ok=True)

    print("=" * 64)
    mode = "舵机模式" if servo_ok else "人工摆位模式"
    print(f"场景: {args.scene} | 偏航档: {' '.join(args.head)} | 俯仰: {args.pitch} "
          f"| 每档张数: {args.count} | {mode}")
    print("拍前自查: 相机未被占用？本次要覆盖的距离/光照组合已就位？")
    print("=" * 64)

    last_pitch = None
    total_ok = 0
    for gear in args.head:
        if servo_ok:
            state.set_head(YAW_GEARS[gear])       # 动态等待到位
            last_pitch = set_pitch(ctl, args.pitch, last_pitch)
        else:
            try:
                input(f"[人工模式] 调好头部（偏航 {gear}={YAW_GEARS[gear]}，"
                      f"俯仰 {args.pitch}）后回车继续...")
            except EOFError:
                print("（无交互终端，等 3 秒后开始）")
                time.sleep(3)
        start = resolve_start(scene_dir, args.scene, gear, args.pitch, args.start)
        total_ok += shoot_gear(scene_dir, args.scene, gear, args.pitch,
                               args.count, args.interval, start)

    expected = args.count * len(args.head)
    print("-" * 64)
    print(f"完成: {total_ok}/{expected} 张已存入 {scene_dir}")
    if total_ok < expected:
        sys.exit(1)


if __name__ == "__main__":
    main()
