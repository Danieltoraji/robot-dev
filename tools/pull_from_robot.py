#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pull_from_robot.py —— 机器人 → PC 文件拉取（Jupyter Contents API，零第三方依赖）

tools/sync_to_robot.py 的反向：把机器人上的文件（如 dataset_raw 采集照片）
拉回 PC。远端目录递归遍历，已存在且字节一致的文件跳过（可中断续拉）。

用法（PC 仓库根目录）：
    python tools/pull_from_robot.py --remote dataset_raw --dest archive/result/robot_pull
    python tools/pull_from_robot.py --remote dataset_raw/test --dest . --check
"""

import argparse
import json
import os
import sys
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sync_to_robot import DEFAULT_HOST, DEFAULT_PASSWORD, JupyterClient


def list_remote_files(client, path):
    """递归列出远端目录，返回 [(远端完整路径, size)]；路径不存在返回 None"""
    s, b = client._request("GET", f"/api/contents/{urllib.parse.quote(path)}?content=1")
    if s != 200:
        return None
    info = json.loads(b)
    if info.get("type") == "file":
        return [(info["path"], info.get("size") or 0)]
    out = []
    for child in info.get("content") or []:
        if child.get("type") == "directory":
            out.extend(list_remote_files(client, child["path"]) or [])
        else:
            out.append((child["path"], child.get("size") or 0))
    return out


def main():
    parser = argparse.ArgumentParser(description="机器人 → PC 文件拉取")
    parser.add_argument("--remote", required=True,
                        help="远端路径（Jupyter 根下，如 dataset_raw）")
    parser.add_argument("--dest", required=True, help="本地目标目录")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--password", default=DEFAULT_PASSWORD)
    parser.add_argument("--check", action="store_true", help="干跑：只列出将拉取的文件")
    args = parser.parse_args()

    client = JupyterClient(args.host, args.password)
    files = list_remote_files(client, args.remote)
    if files is None:
        print(f"远端不存在: {args.remote}")
        sys.exit(1)

    prefix = args.remote.rstrip("/") + "/"
    total = downloaded = skipped = failed = 0
    for remote_path, size in sorted(files):
        rel = remote_path[len(prefix):] if remote_path.startswith(prefix) else remote_path
        local_path = os.path.join(args.dest, rel.replace("/", os.sep))
        total += 1
        if os.path.exists(local_path):
            with open(local_path, "rb") as f:
                if f.read() == client.get_file_bytes(remote_path):
                    skipped += 1
                    continue
        if args.check:
            print(f"[将拉取] {rel} ({size} 字节)")
            downloaded += 1
            continue
        data = client.get_file_bytes(remote_path)
        if data is None:
            print(f"[失败] {rel}")
            failed += 1
            continue
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(data)
        downloaded += 1
        print(f"[已拉取] {rel} ({len(data)} 字节)")

    print("-" * 56)
    print(f"完成: 拉取 {downloaded} / 跳过(一致) {skipped} / 失败 {failed} / 共 {total}"
          + ("（干跑）" if args.check else ""))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
