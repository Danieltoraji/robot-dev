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
（sim/ docs/ tests/ archive/ release/ 仅 PC 使用，不进机器人；
  例外：archive/result/ninegrid_homography.json 是数字宫格运行时标定产物，附加同步。）
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

DEFAULT_HOST = os.environ.get("ROBOT_HOST", "http://192.168.31.209:8888")
DEFAULT_PASSWORD = "pi"
DEFAULT_REMOTE_ROOT = "Robot_control_self_module"
DEFAULT_PATHS = ["main.py", "debug.sh", "core", "vision", "levels", "tools", "models"]

# 机器人运行时需要的标定产物：archive/ 默认不同步（PC 专用），这些文件例外。
# 数字宫格的地面单应按俯仰档存于 archive/result/ninegrid_homography.json，
# 缺失会退回 from_pose 解析自举（布局归属精度下降）——必须随代码一起上机。
EXTRA_FILES = [
    "archive/result/ninegrid_homography.json",
    # 上下楼梯与识别跨障：本地系地面单应标定（缺失退回 from_pose 自举，
    # 测距精度下降到 ±3cm 级）——必须随代码一起上机
    "archive/result/stairs_hurdle_calib.json",
]

SKIP_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints", ".zcode", "archive", "release"}
SKIP_EXTS = (".pyc", ".pyo", ".npz", ".npz.bak")


class JupyterClient:
    """最小 Jupyter REST 客户端：密码登录 + Contents API（标准库实现）"""

    def __init__(self, host, password, timeout=120):
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.jar))
        self._login(password)

    def _login(self, password):
        _status, body = self._open_with_retry(self.host + "/login")
        html = body.decode("utf-8", "replace")
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
        """链路抖动重试：请求与响应体读取都在保护范围内
        （2026-09-07 教训：大文件 GET 的 body 读取中途超时，仅包 open 不够）"""
        last = None
        for i in range(tries):
            try:
                resp = self.opener.open(req, timeout=self.timeout)
                return resp.status, resp.read()
            except urllib.error.HTTPError:
                raise  # 4xx/5xx 是业务结果，重试无意义
            except (urllib.error.URLError, TimeoutError, OSError) as e:
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
            return self._open_with_retry(req)
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def remote_meta(self, remote_path):
        """远端文件元信息（含 size）；不存在返回 None"""
        s, b = self._request("GET", f"/api/contents/{urllib.parse.quote(remote_path)}")
        if s != 200:
            return None
        return json.loads(b)

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
    parser.add_argument("--no-manifest", action="store_true",
                        help="不用本地清单，退回远端逐字节比对（慢，不推荐）")
    parser.add_argument("--full", action="store_true",
                        help="忽略清单全量重传（结束后重建清单）")
    parser.add_argument("--request-timeout", type=int, default=30,
                        help="单请求超时秒（默认 30；大文件弱链路可调大，如 150）")
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pairs = []
    for rel in args.paths:
        if not os.path.exists(os.path.join(base, rel)):
            print(f"[WARN] 本地不存在，跳过: {rel}")
            continue
        pairs.extend(collect_local_files(base, rel))
    # 标定产物例外（archive/ 默认跳过，但机器人运行时需要）
    for rel in EXTRA_FILES:
        local = os.path.join(base, rel)
        if os.path.isfile(local):
            print(f"[标定产物] 附加同步: {rel}")
            pairs.append((local, rel.replace(os.sep, "/")))
    if not pairs:
        print("没有可同步的文件")
        sys.exit(1)

    client = JupyterClient(args.host, args.password, timeout=args.request_timeout)
    print(f"已登录 {args.host}，目标 {args.remote_root}/，"
          f"待同步 {len(pairs)} 个文件{'（干跑）' if args.check else ''}")

    # 本地同步清单：记录每个已推送文件的 (size, mtime_ns)。
    # 命中清单 = 本地直接跳过，零远端请求——根因：每文件 2 个远端比对请求 ×
    # 几十个文件，在忙于跑框架的机器人 Jupyter 上是分钟级的卡顿源。
    added = updated = unchanged = failed = 0
    manifest_path = os.path.join(base, "archive", "sync_manifest.json")
    manifest = {}
    if not args.no_manifest and os.path.exists(manifest_path):
        try:
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
        except (ValueError, OSError):
            manifest = {}

    for local_path, rel in pairs:
        remote_path = f"{args.remote_root}/{rel}"
        st = os.stat(local_path)
        entry = manifest.get(rel)
        if (not args.full and entry is not None
                and entry.get("size") == st.st_size
                and entry.get("mtime_ns") == st.st_mtime_ns):
            unchanged += 1
            continue
        with open(local_path, "rb") as f:
            data = f.read()
        if len(data) > 80 * 1024 * 1024:
            print(f"[WARN] 超大文件跳过: {rel} ({len(data) // 1024 // 1024}MB)")
            failed += 1
            continue
        is_new = entry is None
        tag = "新增" if is_new else "更新"
        # 单文件网络隔离：一个文件的超时/失败只计失败并继续，
        # 不中止整个同步（2026-09-07 教训：弱链路上大文件 PUT 反复超时曾卡死全局）
        try:
            if args.check:
                print(f"[将{tag}] {rel} ({st.st_size} 字节)", flush=True)
            else:
                parent = remote_path.rsplit("/", 1)[0]
                if "/" in remote_path:
                    client.ensure_remote_dir(parent)
                if not client.put_file_bytes(remote_path, data):
                    print(f"[失败] {rel}")
                    failed += 1
                    continue
                manifest[rel] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
            added += is_new
            updated += not is_new
        except (TimeoutError, urllib.error.URLError, OSError) as e:
            print(f"[网络失败，跳过] {rel}: {e}")
            failed += 1
            continue

    if not args.check:
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=1)

    print("-" * 56)
    print(f"完成: 新增 {added} / 更新 {updated} / 无变化 {unchanged} / 失败 {failed}"
          + ("（干跑，未写远端）" if args.check else ""))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
