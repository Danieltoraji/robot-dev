#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""robot_files.py —— 机器人文件读写小工具（Jupyter Contents API，标准库实现）

用途：sync_to_robot.py 只能推「本地 → 机器人」，但在实机上常需要
「先看机器人上是什么、再只改一处推回去」。本工具补这一段：

    python tools/robot_files.py --host http://192.168.31.209:8888 ls  Robot_Competition
    python tools/robot_files.py ... get Robot_Competition/main.py                  # 打印到 stdout
    python tools/robot_files.py ... get Robot_Competition/main.py --out archive/pull/main.py.robot
    python tools/robot_files.py ... put Robot_Competition/main.py --from archive/pull/main.py.robot
    python tools/robot_files.py ... rm  Robot_Competition/levels/_tmp.py

注意：Contents API 对**小文件**可能直接返回 UTF-8 文本、对大文件返回 base64，
本工具按返回的 `format` 字段分别处理（踩过一次坑）。
"""
import argparse
import base64
import http.cookiejar
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_HOST = os.environ.get("ROBOT_HOST", "http://192.168.43.81:8888")
DEFAULT_PASSWORD = "pi"


def login(host, password, timeout=30):
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    html = opener.open(host.rstrip("/") + "/login", timeout=timeout).read().decode("utf-8", "replace")
    m = re.search(r'name="_xsrf" value="([^"]+)"', html)
    token = m.group(1) if m else next((c.value for c in jar if c.name == "_xsrf"), None)
    opener.open(host.rstrip("/") + "/login",
                data=urllib.parse.urlencode({"_xsrf": token, "password": password}).encode(),
                timeout=timeout)
    if not any(c.name.startswith("username") for c in jar):
        raise RuntimeError("登录失败：未取得会话 cookie（检查密码/地址）")
    return opener, host.rstrip("/"), token


def _req(opener, host, path, token, content=False, method="GET", body=None):
    # ★ 每个请求都要带 X-XSRFToken，否则写操作（PUT/DELETE）返回 403 Forbidden
    url = f"{host}/api/contents/{urllib.parse.quote(path)}" + ("?content=1" if content else "")
    headers = {"X-XSRFToken": token}
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        raw = opener.open(req, timeout=120).read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{method} {path} → HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:200]}") from None
    # DELETE 成功返回 204/空体；不能无条件 json.loads
    return json.loads(raw.decode("utf-8")) if raw.strip() else None


def read_bytes(opener, host, token, path):
    r = _req(opener, host, path, token, content=True)
    if r.get("type") != "file":
        raise RuntimeError(f"{path} 不是文件（type={r.get('type')}）")
    c = r["content"]
    # format="base64" ⇒ 二进制；否则是 UTF-8 文本（小文件时 Jupyter 会这么返回）
    if r.get("format") == "base64":
        return base64.b64decode(c)
    return c.encode("utf-8")


def write_bytes(opener, host, token, path, data, fmt=None):
    parent = path.rsplit("/", 1)[0] if "/" in path else ""
    if parent:
        try:
            _req(opener, host, parent, token)
        except RuntimeError:
            _mkdirs(opener, host, token, parent)
    if fmt is None:
        try:
            data.decode("utf-8")
            fmt = "text"
        except UnicodeDecodeError:
            fmt = "base64"
    body = {"type": "file", "format": fmt,
            "content": data.decode("utf-8") if fmt == "text" else base64.b64encode(data).decode("ascii")}
    _req(opener, host, path, token, method="PUT", body=body)


def _mkdirs(opener, host, token, path):
    parts = path.split("/")
    for i in range(1, len(parts) + 1):
        sub = "/".join(parts[:i])
        try:
            _req(opener, host, sub, token)
        except RuntimeError:
            _req(opener, host, sub, token, method="PUT", body={"type": "directory"})


def main():
    ap = argparse.ArgumentParser(description="机器人文件读写（Jupyter Contents API）")
    ap.add_argument("cmd", choices=["ls", "get", "put", "rm"])
    ap.add_argument("path")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--password", default=DEFAULT_PASSWORD)
    ap.add_argument("--out", help="get：写入本地文件（缺省打印到 stdout）")
    ap.add_argument("--from", dest="src", help="put：本地源文件")
    args = ap.parse_args()

    opener, host, token = login(args.host, args.password)

    if args.cmd == "ls":
        r = _req(opener, host, args.path, token)
        if r.get("type") != "directory":
            print(f"{args.path}\t{r.get('type')}\t{r.get('size')}")
            return
        for c in r.get("content", []):
            print("%-40s %-10s %s" % (c["name"], c["type"], c.get("size")))
        return

    if args.cmd == "get":
        data = read_bytes(opener, host, token, args.path)
        if args.out:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "wb") as f:
                f.write(data)
            print(f"已写入 {args.out}（{len(data)} 字节）")
        else:
            sys.stdout.write(data.decode("utf-8", "replace"))
        return

    if args.cmd == "put":
        if not args.src:
            raise SystemExit("put 需要 --from <本地文件>")
        with open(args.src, "rb") as f:
            data = f.read()
        write_bytes(opener, host, token, args.path, data)
        back = read_bytes(opener, host, token, args.path)
        same = back == data
        print(f"已推送 {args.path}（{len(data)} 字节）；回读校验 {'一致' if same else '不一致！'}")
        if not same:
            sys.exit(2)
        return

    if args.cmd == "rm":
        _req(opener, host, args.path, token, method="DELETE")
        print(f"已删除 {args.path}")


if __name__ == "__main__":
    main()
