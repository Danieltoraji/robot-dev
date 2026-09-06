#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sync_to_robot.py —— PC → 机器人代码同步（Jupyter Contents API，零第三方依赖）

机器人端 Jupyter: http://192.168.31.209:8888（密码登录）；
目标目录: Robot_control_self_module（即 /home/pi/Robot_control_self_module）。

原理：
  1. GET /login 取 _xsrf → POST /login（password）拿会话 cookie；
  2. 遍历本地待同步文件，逐个 GET /api/contents/<path>?content=1 比对字节；
  3. 缺失或内容不同才 PUT（base64 格式，父目录自动补建）。

同步纪律：PC 为唯一真源，整文件覆盖；本脚本**不做远端删除**（多余文件需
人工清理，防误删机器人侧产物）。

用法（PC 仓库根目录）：
    python tools/sync_to_robot.py --check              # 干跑：只列出将同步的文件
    python tools/sync_to_robot.py                      # 同步默认集合（见下）
    python tools/sync_to_robot.py --paths core main.py # 只同步指定路径
    python tools/sync_to_robot.py --host http://IP:8888 --password xxx

默认同步集合: main.py + core/ vision/ levels/ tools/ models/
（sim/ docs/ tests/ archive/ release/ 仅 PC 使用，不进机器人。）
"""

import argparse
import base64
import http.cookiejar
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_HOST = "http://192.168.31.209:8888"
DEFAULT_PASSWORD = "pi"
DEFAULT_REMOTE_ROOT = "Robot_control_self_module"
DEFAULT_PATHS = ["main.py", "core", "vision", "levels", "tools", "models"]

SKIP_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints", ".zcode", "archive", "release"}
SKIP_EXTS = (".pyc", ".pyo", ".npz", ".npz.bak")


class JupyterClient:
    """最小 Jupyter REST 客户端：密码登录 + Contents API（标准库实现）"""

    def __init__(self, host, password, timeout=30):
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.jar))
        self._login(password)

    def _login(self, password):
        resp = self._open_with_retry(self.host + "/login")
        html = resp.read().decode("utf-8", "replace")
        m = re.search(r'name="_xsrf" value="([^"]+)"', html)
        token = m.group(1) if m else self._cookie("_xsrf")
        self._open_with_retry(urllib.request.Request(
            self.host + "/login",
            data=urllib.parse.urlencode({"_xsrf": token, "password": password}).encode(),
            method="POST"))
        if not self._cookie("username"):
            raise RuntimeError("登录失败：未取得会话 cookie（检查密码/地址）")
        self.xsrf = token

    def _open_with_retry(self, req, tries=3):
        """链路抖动重试（2026-09-05：机器人 WiFi 省电导致的频繁超时，已关省电，
        但仍保留重试兜底）"""
        last = None
        for i in range(tries):
            try:
                return self.opener.open(req, timeout=self.timeout)
            except (urllib.error.URLError, urllib.error.HTTPError,
                    TimeoutError, OSError) as e:
                if isinstance(e, urllib.error.HTTPError) and e.code < 500:
                    raise  # 4xx 是业务错误，重试无意义
                last = e
                time.sleep(1.5 * (i + 1))
        raise last

    def _cookie(self, prefix):
        for c in self.jar:
            if c.name.startswith(prefix):
                return c.value
        return None

    def _request(self, method, path, obj=None):
        url = f"{self.host}{path}"
        headers = {"X-XSRFToken": self.xsrf}
        data = None
        if obj is not None:
            data = json.dumps(obj).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            resp = self._open_with_retry(req)
            return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def get_file_bytes(self, remote_path):
        """读远端文件原始字节；不存在返回 None"""
        s, b = self._request("GET", f"/api/contents/{urllib.parse.quote(remote_path)}?content=1")
        if s != 200:
            return None
        info = json.loads(b)
        content = info.get("content")
        if info.get("format") == "base64":
            return base64.b64decode(content)
        if isinstance(content, str):
            return content.encode("utf-8")
        return None

    def ensure_remote_dir(self, remote_dir):
        parts = remote_dir.split("/")
        for i in range(1, len(parts) + 1):
            d = "/".join(parts[:i])
            s, _ = self._request("GET", f"/api/contents/{urllib.parse.quote(d)}")
            if s == 404:
                self._request("PUT", f"/api/contents/{urllib.parse.quote(d)}",
                              {"type": "directory"})

    def put_file_bytes(self, remote_path, data):
        payload = {"type": "file", "format": "base64",
                   "content": base64.b64encode(data).decode("ascii")}
        s, _ = self._request("PUT", f"/api/contents/{urllib.parse.quote(remote_path)}",
                             payload)
        return s in (200, 201)


def collect_local_files(base, rel):
    """返回 [(相对仓库根的本地路径, 相对远端根的路径)]，保持稳定排序"""
    local = os.path.join(base, rel)
    pairs = []
    if os.path.isfile(local):
        pairs.append((local, rel.replace(os.sep, "/")))
        return pairs
    for root, dirs, files in os.walk(local):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for name in sorted(files):
            if name.lower().endswith(SKIP_EXTS):
                continue
            full = os.path.join(root, name)
            remote_rel = os.path.relpath(full, base).replace(os.sep, "/")
            pairs.append((full, remote_rel))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="PC → 机器人代码同步（Jupyter API）")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--password", default=DEFAULT_PASSWORD)
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT,
                        help="机器人侧目标目录（Jupyter 根下，默认 Robot_control_self_module）")
    parser.add_argument("--paths", nargs="+", default=DEFAULT_PATHS,
                        help="要同步的本地文件/目录（相对仓库根）")
    parser.add_argument("--check", action="store_true",
                        help="干跑：只显示会新增/更新哪些文件，不写远端")
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pairs = []
    for rel in args.paths:
        if not os.path.exists(os.path.join(base, rel)):
            print(f"[WARN] 本地不存在，跳过: {rel}")
            continue
        pairs.extend(collect_local_files(base, rel))
    if not pairs:
        print("没有可同步的文件")
        sys.exit(1)

    client = JupyterClient(args.host, args.password)
    print(f"已登录 {args.host}，目标 {args.remote_root}/，"
          f"待同步 {len(pairs)} 个文件{'（干跑）' if args.check else ''}")

    added = updated = unchanged = failed = 0
    for local_path, rel in pairs:
        remote_path = f"{args.remote_root}/{rel}"
        with open(local_path, "rb") as f:
            data = f.read()
        if len(data) > 80 * 1024 * 1024:
            print(f"[WARN] 超大文件跳过: {rel} ({len(data) // 1024 // 1024}MB)")
            failed += 1
            continue
        remote = client.get_file_bytes(remote_path)
        if remote == data:
            unchanged += 1
            continue
        tag = "新增" if remote is None else "更新"
        if args.check:
            print(f"[将{tag}] {rel} ({len(data)} 字节)")
        else:
            parent = remote_path.rsplit("/", 1)[0]
            if "/" in remote_path:
                client.ensure_remote_dir(parent)
            if client.put_file_bytes(remote_path, data):
                print(f"[已{tag}] {rel} ({len(data)} 字节)")
            else:
                print(f"[失败] {rel}")
                failed += 1
                continue
        added += (remote is None)
        updated += (remote is not None)

    print("-" * 56)
    print(f"完成: 新增 {added} / 更新 {updated} / 无变化 {unchanged} / 失败 {failed}"
          + ("（干跑，未写远端）" if args.check else ""))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
