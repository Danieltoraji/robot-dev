#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exec_on_robot.py —— 在机器人上远程执行代码（Jupyter 内核会话，需 websocket-client）

依赖：pip install websocket-client（PC 端）。
为什么不用 Jupyter 终端 API：终端（terminado）在真机上输出时序不稳定
（启动横幅吞输入、ANSI 乱码、易超时）；内核会话协议确定性强，输出可靠。

注意：机器人上含 numpy/cv2/onnxruntime/ultralytics 的解释器是
/home/pi/jupyter-env/bin/python3（Jupyter 的 venv）；系统 python3 是否同环境
未验证，跑依赖包的命令建议显式用该路径。

用法（PC 仓库根目录）：
    python tools/exec_on_robot.py --code "print(1+1)"
    python tools/exec_on_robot.py --cmd "uname -a"
    python tools/exec_on_robot.py --cmd "/home/pi/jupyter-env/bin/python3 -m pip list" --timeout 300
"""

import argparse
import http.cookiejar
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

DEFAULT_HOST = "http://192.168.31.209:8888"
DEFAULT_PASSWORD = "pi"


def login(host, password, timeout=30):
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    resp = opener.open(host + "/login", timeout=timeout)
    m = re.search(r'name="_xsrf" value="([^"]+)"', resp.read().decode("utf-8", "replace"))
    token = m.group(1) if m else next(
        (c.value for c in jar if c.name == "_xsrf"), None)
    opener.open(host + "/login",
                data=urllib.parse.urlencode({"_xsrf": token, "password": password}).encode(),
                timeout=timeout)
    cookie = "; ".join(f"{c.name}={c.value}" for c in jar)
    return opener, token, cookie


def create_session(opener, host, xsrf):
    body = json.dumps({"kernel": {"name": "python3"}, "name": "exec_on_robot",
                       "type": "notebook", "path": "exec_on_robot/probe.ipynb"}).encode()
    req = urllib.request.Request(host + "/api/sessions", data=body, method="POST",
                                 headers={"Content-Type": "application/json",
                                          "X-XSRFToken": xsrf})
    return json.loads(opener.open(req, timeout=30).read())


def execute(host, cookie, kernel_id, code, deadline_s):
    import websocket  # 懒加载，给出友好提示

    ws = websocket.create_connection(f"{host.replace('http', 'ws', 1)}/api/kernels/{kernel_id}/channels",
                                     timeout=15, header={"Cookie": cookie})
    msg_id = uuid.uuid4().hex
    ws.send(json.dumps({
        "header": {"msg_id": msg_id, "username": "pc", "session": uuid.uuid4().hex,
                   "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "msg_type": "execute_request", "version": "5.3"},
        "parent_header": {}, "metadata": {},
        "content": {"code": code, "silent": False, "store_history": False,
                    "user_expressions": {}, "allow_stdin": False, "stop_on_error": True},
        "buffers": [], "signature": "", "channel": "shell",
    }))

    outputs, got_idle = [], False
    deadline = time.time() + deadline_s
    while time.time() < deadline:
        try:
            raw = ws.recv()
        except websocket.WebSocketTimeoutException:
            continue
        msg = json.loads(raw)
        if msg.get("parent_header", {}).get("msg_id") != msg_id:
            continue
        mt = msg.get("header", {}).get("msg_type")
        c = msg.get("content", {})
        if mt == "stream":
            outputs.append(c.get("text", ""))
        elif mt == "execute_result":
            outputs.append(c.get("data", {}).get("text/plain", "") + "\n")
        elif mt == "error":
            outputs.append("[ERROR] " + c.get("ename", "") + ": " + c.get("evalue", ""))
        elif mt == "status" and c.get("execution_state") == "idle":
            got_idle = True
            break
    ws.close()
    return "".join(outputs), got_idle


def main():
    parser = argparse.ArgumentParser(description="机器人上远程执行代码（Jupyter 内核）")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--password", default=DEFAULT_PASSWORD)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--code", help="要执行的 Python 代码")
    group.add_argument("--cmd", help="要执行的 shell 命令（subprocess 包一层）")
    parser.add_argument("--timeout", type=float, default=120,
                        help="执行超时秒（默认 120）")
    args = parser.parse_args()

    try:
        import websocket  # noqa: F401
    except ImportError:
        print("缺少依赖: pip install websocket-client")
        sys.exit(1)

    if args.cmd is not None:
        esc = args.cmd.replace("\\", "\\\\").replace('"', '\\"')
        code = ('import subprocess\n'
                f'r = subprocess.run("{esc}", shell=True, capture_output=True, '
                f'text=True, timeout={args.timeout:.0f})\n'
                'print(r.stdout)\n'
                'if r.stderr: print("[stderr]", r.stderr)\n'
                'print("[exit]", r.returncode)\n')
    else:
        code = args.code

    opener, xsrf, cookie = login(args.host, args.password)
    sess = create_session(opener, args.host, xsrf)
    kernel_id = sess["kernel"]["id"]
    print(f"[内核已启动 {kernel_id[:8]}…]（RPi 上启动约 3~10s）")
    time.sleep(3)

    try:
        out, idle = execute(args.host, cookie, kernel_id, code, args.timeout)
        print(out, end="" if out.endswith("\n") else "\n")
        if not idle:
            print(f"[WARN] 执行未在 {args.timeout}s 内完成（内核已终止）")
            sys.exit(1)
    finally:
        req = urllib.request.Request(
            f"{args.host}/api/sessions/{sess['id']}", method="DELETE",
            headers={"X-XSRFToken": xsrf})
        try:
            opener.open(req, timeout=15)
        except (urllib.error.HTTPError, urllib.error.URLError):
            print("[WARN] 会话清理失败，可在 Jupyter 网页里手动关闭")


if __name__ == "__main__":
    main()
