#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在机器人上执行本地 Python 脚本文件（绕开 PowerShell 命令行引号问题）。

用法：
    python tools/_exec_file_on_robot.py tools/_measure_turn_angle.py [--timeout 300]
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exec_on_robot import login, create_session, execute  # noqa: E402

HOST = "http://192.168.31.209:8888"
PASSWORD = "pi"


def main():
    if len(sys.argv) < 2:
        print("用法: python tools/_exec_file_on_robot.py <脚本.py> [--timeout 秒]")
        sys.exit(1)
    script = sys.argv[1]
    timeout = 300
    if "--timeout" in sys.argv:
        timeout = float(sys.argv[sys.argv.index("--timeout") + 1])

    with open(script, encoding="utf-8") as f:
        code = f.read()

    opener, xsrf, cookie = login(HOST, PASSWORD)
    sess = create_session(opener, HOST, xsrf)
    kernel_id = sess["kernel"]["id"]
    print(f"[内核已启动 {kernel_id[:8]}…]（RPi 上启动约 3~10s）")
    time.sleep(3)

    try:
        out, idle = execute(HOST, cookie, kernel_id, code, timeout)
        print(out, end="" if out.endswith("\n") else "\n")
        if not idle:
            print(f"[WARN] 执行未在 {timeout}s 内完成（内核已终止）")
            sys.exit(1)
    finally:
        import urllib.request
        import urllib.error
        req = urllib.request.Request(
            f"{HOST}/api/sessions/{sess['id']}", method="DELETE",
            headers={"X-XSRFToken": xsrf})
        try:
            opener.open(req, timeout=15)
        except (urllib.error.HTTPError, urllib.error.URLError):
            pass


if __name__ == "__main__":
    main()
