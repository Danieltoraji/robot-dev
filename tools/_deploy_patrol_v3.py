#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""临时部署脚本：把本地 RedLinePatrolV3.py 部署到机器人 TonyPi/Functions/。"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sync_to_robot import JupyterClient  # noqa: E402

LOCAL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "levels", "football_codes", "RedLinePatrolV3.py")
REMOTE = "TonyPi/Functions/RedLinePatrolV3.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get(
        "ROBOT_HOST", "http://192.168.31.209:8888"))
    ap.add_argument("--password", default="pi")
    args = ap.parse_args()

    with open(LOCAL, "rb") as f:
        data = f.read()
    print(f"本地文件: {LOCAL} ({len(data)} bytes)")

    client = JupyterClient(args.host, args.password, timeout=60)
    print(f"已登录 {args.host}")

    old = client.get_file_bytes(REMOTE)
    if old is not None:
        if old == data:
            print("远端已是最新，无需部署")
            return
        stamp = time.strftime("%Y%m%d_%H%M%S")
        backup = REMOTE + ".bak_" + stamp
        ok = client.put_file_bytes(backup, old)
        print(f"备份远端旧文件 -> {backup}: {'OK' if ok else 'FAILED'}")

    ok = client.put_file_bytes(REMOTE, data)
    print(f"上传新文件 -> {REMOTE}: {'OK' if ok else 'FAILED'}")

    verify = client.get_file_bytes(REMOTE)
    if verify == data:
        print("校验通过：远端字节与本地一致")
    else:
        print(f"校验失败：远端 {len(verify) if verify is not None else None} bytes")


if __name__ == "__main__":
    main()
