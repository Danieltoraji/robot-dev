# -*- coding: utf-8 -*-
"""磁盘满时的应急通道：经 Jupyter **终端** API 在机器人上执行命令

为什么需要：磁盘写满 ⇒ Jupyter 内核会话起不来（写不了 session 文件，HTTP 500），
`tools/exec_on_robot.py` 整条路断掉；但 `/api/terminals` 仍可用。
本脚本用终端跑命令并抓回输出，用来查/清磁盘、验证文件。

用法：
    python tools/robot_shell.py "df -h /"
    python tools/robot_shell.py --timeout 120 "du -sh /home/pi/Pictures"
"""

import argparse
import json
import re
import sys
import time
import uuid
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import urllib.error  # noqa: E402
import urllib.request  # noqa: E402

from exec_on_robot import DEFAULT_HOST, DEFAULT_PASSWORD, login  # noqa: E402


def run(cmd, timeout=60):
    opener, xsrf, cookie = login(DEFAULT_HOST, DEFAULT_PASSWORD)
    body = json.dumps({}).encode()
    req = urllib.request.Request(DEFAULT_HOST + "/api/terminals", data=body,
                                 method="POST",
                                 headers={"Content-Type": "application/json",
                                          "X-XSRFToken": xsrf})
    name = json.loads(opener.open(req, timeout=30).read())["name"]

    import websocket
    ws = websocket.create_connection(
        f"{DEFAULT_HOST.replace('http', 'ws', 1)}/api/terminals/{name}/channels",
        timeout=15, header={"Cookie": cookie})
    marker = f"__DONE_{uuid.uuid4().hex[:8]}__"
    # 只回显我们要的：命令输出 + 结束标记（终端会把输入回显出来，故用 marker 切分）
    ws.send(json.dumps(["stdin", f"{cmd}; echo {marker}\n"]))
    buf, t0 = "", time.time()
    while time.time() - t0 < timeout:
        try:
            msg = json.loads(ws.recv())
        except Exception:
            break
        if isinstance(msg, list) and len(msg) > 1 and msg[0] == "stdout":
            buf += msg[1]
            if marker in buf:
                break
    ws.close()
    # 去掉回显的命令行本身与标记
    out = buf.split(marker)[0]
    out = re.sub(r"^.*?" + re.escape(cmd.split(";")[0][:20]), "", out,
                 count=1, flags=re.S) if out else out
    return out.strip("\r\n")


def main():
    ap = argparse.ArgumentParser(description="经 Jupyter 终端 API 在机器人上跑命令")
    ap.add_argument("cmd")
    ap.add_argument("--timeout", type=int, default=60)
    args = ap.parse_args()
    print(run(args.cmd, timeout=args.timeout))
    return 0


if __name__ == "__main__":
    sys.exit(main())
