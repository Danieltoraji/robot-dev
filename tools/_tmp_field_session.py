#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""_tmp_field_session.py —— 现场在机器人上跑测量脚本（PC 侧包装器，**走 Jupyter 终端**）

为什么不用内核会话：`exec_on_robot.py` 走 Jupyter **内核**，而内核没有 stdin，
交互脚本里的 `input()` 会直接抛
    StdinNotImplementedError: raw_input was called, but this frontend does not
    support input requests.
而本项目的测量脚本（`_tmp_measure_motion.py` / `_tmp_dist_sweep.py`）**必须交互**
（停下等你量卷尺、等你摆面板后回车）。所以本包装器改走 Jupyter **终端**
（terminado：`POST /api/terminals` + `ws /terminals/websocket/<name>`），
那是真 TTY：`input()` 可用、ANSI 正常，并且会先
`source /home/pi/jupyter-env/bin/activate` 激活 venv（用户实测该方式可用）。

用法（PC 仓库根目录）：
    python tools/_tmp_field_session.py --check
    python tools/_tmp_field_session.py --script _tmp_measure_motion.py --args "--group 1a --run"
    python tools/_tmp_field_session.py --script _tmp_dist_sweep.py --args "--tag m2 --distances 40..10,-2 --repeats 2"
    python tools/_tmp_field_session.py --focus          # 对焦探针（非交互）
    python tools/_tmp_field_session.py --down           # 清理遗留终端

交互说明：脚本提示实时打印；**你的输入直接敲、回车发送**。
Ctrl-C 中断等待并关闭终端。

权限范围：默认只允许机器人 `tools/` 下的白名单脚本；传其它名字会被拒。
"""

import argparse
import http.cookiejar
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

ROBOT_ROOT = "/home/pi/Robot_Competition"
VENV_ACTIVATE = "/home/pi/jupyter-env/bin/activate"
HOST = os.environ.get("ROBOT_HOST", "http://192.168.31.209:8888")
PASSWORD = os.environ.get("ROBOT_PASSWORD", "pi")

# 允许运行的脚本（防止误跑其它东西）
ALLOWED = {
    "_tmp_focus_probe.py",
    "_tmp_measure_motion.py",
    "_tmp_dist_sweep.py",
    "run_layout_scan.py",
    "field_probe_ninegrid.py",
    "field_camera_sweep.py",
    "diag_arrive.py",
}
TOOLS_PREFIX = "tools/"


class TermClient:
    """最小 terminado 客户端：登录 → 建/删终端 → 提供 cookie 供 websocket 握手"""

    def __init__(self, host=HOST, password=PASSWORD, timeout=30):
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.jar))
        self._login(password)

    def _login(self, password):
        html = self.opener.open(self.host + "/login", timeout=self.timeout
                                ).read().decode("utf-8", "replace")
        m = re.search(r'name="_xsrf" value="([^"]+)"', html)
        self.xsrf = m.group(1) if m else next(
            (c.value for c in self.jar if c.name == "_xsrf"), None)
        self.opener.open(
            self.host + "/login",
            data=urllib.parse.urlencode({"_xsrf": self.xsrf,
                                         "password": password}).encode(),
            timeout=self.timeout)
        if not any(c.name.startswith("username") for c in self.jar):
            raise RuntimeError("登录失败：未取得会话 cookie（检查密码/地址）")

    def _req(self, method, path, obj=None):
        data = None
        headers = {"X-XSRFToken": self.xsrf}
        if obj is not None:
            data = json.dumps(obj).encode()
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self.host + path, data=data,
                                     method=method, headers=headers)
        return self.opener.open(req, timeout=self.timeout)

    def new_terminal(self):
        return json.loads(self._req("POST", "/api/terminals").read())["name"]

    def del_terminal(self, name):
        try:
            self._req("DELETE", f"/api/terminals/{urllib.parse.quote(str(name))}")
        except (urllib.error.HTTPError, urllib.error.URLError, OSError):
            pass

    def cookie_header(self):
        return "; ".join(f"{c.name}={c.value}" for c in self.jar)


def _make_line_reader():
    """返回 read_local_line()：有整行输入则返回该行；无输入返回 None；EOF 返回 ""

    Windows：select 对管道 stdin 会抛 OSError(10038)，改用 msvcrt.kbhit()。
    POSIX：用 select + readline。
    """
    if os.name == "nt":
        import msvcrt

        def read_nt():
            if not msvcrt.kbhit():
                return None
            return sys.stdin.readline()
        return read_nt

    import select

    def read_posix():
        r, _, _ = select.select([sys.stdin], [], [], 0.2)
        if not r:
            return None
        return sys.stdin.readline()
    return read_posix


def run_interactive(term, name, cmd, ready_marker="__RC__"):
    """在终端里跑 cmd：实时转发输出，并把本地敲的行发回机器人

    terminado 消息是 JSON 数组：["stdout", 文本] / ["disconnect", ...]
    """
    import queue
    import threading
    import websocket

    ws_url = (term.host.replace("http", "ws", 1)
              + f"/terminals/websocket/{name}")
    ws = websocket.create_connection(ws_url, timeout=5,
                                     header={"Cookie": term.cookie_header()})

    q = queue.Queue()
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            try:
                raw = ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            except Exception:
                break
            if not raw:
                continue
            try:
                msg = json.loads(raw)
            except ValueError:
                continue
            if isinstance(msg, list) and len(msg) >= 2 and msg[0] == "stdout":
                q.put(("out", msg[1]))
            elif isinstance(msg, list) and msg[0] == "disconnect":
                q.put(("disconnect", ""))
                break

    threading.Thread(target=reader, daemon=True).start()

    def send(text):
        ws.send(json.dumps(["stdin", text]))

    read_local_line = _make_line_reader()
    buf = ""
    # ⚠️ terminado 的坑：shell 启动横幅会吞掉过早发送的输入（本仓库文档已记录）。
    # 先发一个握手标记，等它回来再发真正的命令 —— 这样能确认 shell 已就绪。
    # 另注：终端默认 shell 是 dash，`source` 是 bash 语法会报 not found，
    # 故这里用 POSIX 的 `.`；并且显式用 venv 解释器，不依赖 activate 状态。
    send(f"echo {ready_marker}READY\n")
    t0 = time.time()
    handshake = False
    while time.time() - t0 < 10:
        try:
            raw = ws.recv()
        except websocket.WebSocketTimeoutException:
            continue
        except Exception:
            break
        try:
            msg = json.loads(raw)
        except ValueError:
            continue
        if isinstance(msg, list) and len(msg) >= 2 and msg[0] == "stdout":
            buf += msg[1]
            if f"{ready_marker}READY" in buf:
                handshake = True
                break
    if not handshake:
        print("[WARN] 终端握手未在 10s 内完成，仍继续发送命令（可能吞输入）")

    send(f"cd {ROBOT_ROOT} && . {VENV_ACTIVATE} && "
         f"export PYTHONIOENCODING=utf-8 && {cmd}; echo {ready_marker}$?\n")

    try:
        while True:
            drained = False
            while True:
                try:
                    kind, text = q.get_nowait()
                except queue.Empty:
                    break
                drained = True
                if kind == "disconnect":
                    raise KeyboardInterrupt
                buf += text
                sys.stdout.write(text)
                sys.stdout.flush()
                m = re.search(ready_marker + r"(\d+)", buf)
                if m:
                    return int(m.group(1))
            if not drained:
                # 本地有输入就发过去（交互提示的处理）。
                # 注意：Windows 上 select 只支持 socket，对管道 stdin 会抛
                # OSError(WinError 10038)，故 Windows 单独走 msvcrt 轮询。
                line = read_local_line()
                if line is None:
                    time.sleep(0.15)
                elif line == "":
                    break
                else:
                    send(line)
    finally:
        stop.set()
        try:
            ws.close()
        except Exception:
            pass
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="现场跑机器人侧测量脚本（走 Jupyter 终端）")
    ap.add_argument("--check", action="store_true",
                    help="列机器人现状（脚本清单 + 相机状态），不跑测量")
    ap.add_argument("--focus", action="store_true", help="跑对焦探针（非交互）")
    ap.add_argument("--script", default=None, help="要跑的脚本名（白名单内）")
    ap.add_argument("--args", default="", help="传给该脚本的参数（整串，加引号）")
    ap.add_argument("--down", action="store_true", help="清理遗留终端")
    args = ap.parse_args(argv)

    term = TermClient()

    if args.down:
        for n in range(1, 12):
            term.del_terminal(n)
        print("已清理 1..11 号终端")
        return 0

    if args.check:
        cmd = ("echo '--- 白名单脚本 ---'; ls -la tools/_tmp_*.py "
               "tools/run_layout_scan.py 2>&1; echo '--- 相机 ---'; "
               "v4l2-ctl -d /dev/video0 --get-ctrl=focus_absolute,"
               "focus_automatic_continuous,white_balance_temperature,"
               "auto_exposure,exposure_time_absolute")
    elif args.focus:
        cmd = (f"python {TOOLS_PREFIX}_tmp_focus_probe.py "
               + (args.args or "--values 270 370 420 470 500 --repeats 2 "
                               "--out /tmp/focus_probe"))
    elif args.script:
        script = os.path.basename(args.script)
        if script not in ALLOWED:
            print(f"拒绝：{script} 不在白名单 {sorted(ALLOWED)}")
            return 2
        cmd = f"python {TOOLS_PREFIX}{script} {args.args}".strip()
    else:
        ap.error("给 --check / --focus / --script / --down 之一")

    name = term.new_terminal()
    print(f"[终端 {name} 已建立] 将执行: {cmd}\n" + "-" * 68)
    try:
        rc = run_interactive(term, name, cmd)
        print("-" * 68 + f"\n[命令退出码 {rc}]")
        return 0
    except KeyboardInterrupt:
        print("\n[中断] 正在关闭终端…")
        return 130
    finally:
        term.del_terminal(name)


if __name__ == "__main__":
    sys.exit(main())
