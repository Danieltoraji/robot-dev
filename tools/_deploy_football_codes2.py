#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""临时部署：把 football_codes2 的 V4 链路 + 修改后的 RedLinePatrolV3.py
部署到机器人 /home/pi/TonyPi/Functions/（运行时目录）。

纪律（与 _deploy_patrol_v3.py 一致）：
  - 覆盖前把远端旧文件备份为 <名>.bak_<时间戳>；
  - 上传后 GET 回读逐字节校验；
  - 模型/标定先比对字节，一致则跳过，不覆盖机器人侧产物。
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sync_to_robot import JupyterClient  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REMOTE_ROOT = "TonyPi/Functions"

# (本地相对仓库根, 远端相对 TonyPi/Functions 的文件名)
FILES = [
    ("levels/football_codes/RedLinePatrolV3.py", "RedLinePatrolV3.py"),
    ("levels/football_codes2/demoV4.py", "demoV4.py"),
    ("levels/football_codes2/patrol_end_recovery.py", "patrol_end_recovery.py"),
    ("levels/football_codes2/tag_route_demo.py", "tag_route_demo.py"),
    ("levels/football_codes2/tag_walk_demo.py", "tag_walk_demo.py"),
    ("levels/football_codes2/goalpost_detector.py", "goalpost_detector.py"),
    ("levels/football_codes2/football_kick_controller.py",
     "football_kick_controller.py"),
    ("levels/football_codes2/goal_line_judge.py", "goal_line_judge.py"),
    ("levels/football_codes2/red_goal_line_detector.py",
     "red_goal_line_detector.py"),
    ("levels/football_codes2/README.md", "README.md"),
    ("levels/football_codes2/CameraCalibration/CalibrationConfig.py",
     "CameraCalibration/CalibrationConfig.py"),
    # 模型：机器人已有同名文件，字节一致则跳过，不一致先备份再覆盖。
    ("levels/football_codes2/models/football_best_win.onnx",
     "models/football_best_win.onnx"),
    ("levels/football_codes2/weights/best.onnx", "weights/best.onnx"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get(
        "ROBOT_HOST", "http://192.168.31.209:8888"))
    ap.add_argument("--password", default="pi")
    ap.add_argument("--check", action="store_true",
                    help="干跑：只比对字节，不上传")
    args = ap.parse_args()

    client = JupyterClient(args.host, args.password, timeout=300)
    print("已登录 {}".format(args.host))

    changed = []
    for local_rel, remote_name in FILES:
        local_path = os.path.join(BASE, local_rel)
        with open(local_path, "rb") as f:
            data = f.read()
        remote_path = REMOTE_ROOT + "/" + remote_name
        old = client.get_file_bytes(remote_path)
        if old == data:
            print("[相同] {} ({})".format(remote_path, len(data)))
        elif old is None:
            print("[远端缺失] {} ({})".format(remote_path, len(data)))
            changed.append((remote_path, None, data))
        else:
            print("[不同] {} (local {} / remote {})".format(
                remote_path, len(data), len(old)))
            changed.append((remote_path, old, data))

    print("-" * 56)
    print("共 {} 个文件，需更新 {} 个{}".format(
        len(FILES), len(changed), "（干跑，未写远端）" if args.check else ""))
    if args.check:
        return

    for remote_path, old, data in changed:
        if old is not None:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            backup = remote_path + ".bak_" + stamp
            ok = client.put_file_bytes(backup, old)
            print("备份旧文件 -> {}: {}".format(backup, "OK" if ok else "FAILED"))
        ok = client.put_file_bytes(remote_path, data)
        print("上传 -> {}: {}".format(remote_path, "OK" if ok else "FAILED"))
        verify = client.get_file_bytes(remote_path)
        print("校验: {}".format("通过" if verify == data else "失败"))
    print("部署完成")


if __name__ == "__main__":
    main()
