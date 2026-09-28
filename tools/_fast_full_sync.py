#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""并行全量同步（比赛应急用）—— tools/_fast_full_sync.py

为什么另写一个：`sync_to_robot.py` 是单线程、每文件 PUT 后等响应。
实测这台 Pi 的瓶颈是**它自己的 base64 编解码 CPU**，不是网络；54MB 的模型单请求
要几十秒，直接撞默认 30s 超时。单线程串行传 535MB 是分钟级起步，中间还随时超时。

这个脚本做四件事：
  1. **先跳过一样的不重传**（见下），重跑基本是秒级，只补真正缺的；
  2. **多线程并发**（默认 8 路上传 / 16 路探测），把 Pi 的 CPU 吃满；
  3. **每文件独立超时**（默认 300s）+ **默认不重试**，单文件最坏就是 300s；
  4. **逐文件输出**：每成功一个打一行，失败也打一行并带原因。

跳过判据（安全第一）：
  远端文件**存在**且**字节数等于本地**才跳过。不比对 mtime（Jupyter 的 PUT 不保真
  mtime），也不只看本地清单 —— 实测客户端超时的时候 Pi 那边常常**已经写完了**，
  只看清单会把这种文件白传一遍。字节数不等或取不到远端信息，一律照传。

范围与 `sync_to_robot.py` 完全一致：main.py + debug.sh + core/ vision/ levels/
tools/ models/ + EXTRA_FILES（两份标定 json）；archive/ release/ sim/ docs/
tests/ 照样不上机。跳过 __pycache__/.git/.ipynb_checkpoints/.zcode 与
*.pyc/*.pyo/*.npz；.sh/.bash 自动 CRLF→LF。

用法（PC 仓库根目录）：
    python tools/_fast_full_sync.py --check     # 只列清单，不写远端
    python tools/_fast_full_sync.py             # 真传（默认 8 并发、不重试）
    python tools/_fast_full_sync.py --full      # 忽略跳过判据，全部重传
    python tools/_fast_full_sync.py -j 4 --timeout 120
"""
import argparse
import base64
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sync_to_robot as S  # noqa: E402  复用它的登录/收集/建目录实现

PRINT_LOCK = threading.Lock()


def log(msg):
    with PRINT_LOCK:
        print(msg, flush=True)


def put_one(client, remote_path, data):
    """单次上传（不重试）；返回 (成功?, 错误原因)"""
    try:
        body = json.dumps({"type": "file", "format": "base64",
                           "content": base64.b64encode(data).decode("ascii")}).encode("utf-8")
        req = urllib.request.Request(
            client.host + "/api/contents/" + urllib.parse.quote(remote_path),
            data=body, method="PUT",
            headers={"Content-Type": "application/json",
                     "X-XSRFToken": client.xsrf})
        resp = client.opener.open(req, timeout=client.timeout)
        if resp.status in (200, 201):
            return True, None
        return False, f"HTTP {resp.status}"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code} {e.read()[:120]}"
    except Exception as e:                           # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def run_pool(items, jobs, handler):
    """把 items 丢给 jobs 个线程跑 handler(item)；返回已处理的顺序不保证"""
    lock = threading.Lock()
    queue = list(items)

    def worker():
        while True:
            with lock:
                if not queue:
                    return
                item = queue.pop(0)
            handler(item)

    threads = [threading.Thread(target=worker, daemon=True)
               for _ in range(max(1, jobs))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def main():
    ap = argparse.ArgumentParser(description="并行全量同步到机器人（跳过远端已一致的）")
    ap.add_argument("--host", default=S.DEFAULT_HOST)
    ap.add_argument("--password", default=S.DEFAULT_PASSWORD)
    ap.add_argument("--remote-root", default=S.DEFAULT_REMOTE_ROOT)
    ap.add_argument("--timeout", type=float, default=300.0, help="上传请求超时秒")
    ap.add_argument("--tries", type=int, default=1, help="每文件上传次数（默认 1=不重试）")
    ap.add_argument("-j", "--jobs", type=int, default=8, help="上传并发线程数")
    ap.add_argument("--probe-jobs", type=int, default=16, help="探测远端大小并发线程数")
    ap.add_argument("--full", action="store_true", help="忽略跳过判据，全部重传")
    ap.add_argument("--check", action="store_true", help="只列清单，不写远端")
    args = ap.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    pairs = []
    for rel in S.DEFAULT_PATHS:
        if not os.path.exists(os.path.join(base, rel)):
            log(f"[WARN] 本地不存在，跳过: {rel}")
            continue
        pairs.extend(S.collect_local_files(base, rel))
    for rel in S.EXTRA_FILES:
        local = os.path.join(base, rel)
        if os.path.isfile(local):
            pairs.append((local, rel.replace(os.sep, "/")))

    if not pairs:
        log("没有可同步的文件")
        return 1

    local_sizes = {rel: os.path.getsize(local) for local, rel in pairs}
    total_bytes = sum(local_sizes.values())
    log(f"本地共 {len(pairs)} 个文件，合计 {total_bytes / 1048576:.1f} MB")

    if args.check:
        for local, rel in pairs:
            log(f"[将传] {rel} ({local_sizes[rel]} 字节)")
        return 0

    client = S.JupyterClient(args.host, args.password, timeout=args.timeout)
    log(f"已登录 {args.host} → {args.remote_root}/；"
        f"上传并发 {args.jobs}、探测并发 {args.probe_jobs}、"
        f"上传超时 {args.timeout:.0f}s、每次尝试 {args.tries} 次")

    # ---- 第一阶段：先建目录，再并发探测远端大小，算出真正要传的 ----
    for d in sorted({os.path.dirname(f"{args.remote_root}/{rel}")
                     for _l, rel in pairs}):
        if d:
            client.ensure_remote_dir(d)

    to_upload = []
    skipped = []
    probe_lock = threading.Lock()

    if args.full:
        to_upload = list(pairs)
        probe_el = 0.0
        log("--full：忽略跳过判据，全部重传")
    else:
        # ★ 按**目录**批量取远端大小：一次 GET 目录列表就能拿到该目录下所有文件的
        # size（实测 0.03s/目录），远快于"每个文件一个请求"——后者 248 个请求串起来
        # 要 200 秒以上，比上传本身还慢（2026-09-28 实测 226s，就是"卡住"的真身）。
        remote_sizes = {}
        dirs = sorted({os.path.dirname(rel) for _l, rel in pairs})
        t_probe = time.time()

        def probe_dir(d):
            path = f"{args.remote_root}/{d}" if d else args.remote_root
            try:
                status, body = client._request(
                    "GET", f"/api/contents/{urllib.parse.quote(path)}?content=1")
            except Exception:                        # noqa: BLE001  取不到就照传
                return
            if status != 200:
                return
            try:
                info = json.loads(body)
            except ValueError:
                return
            found = {}
            prefix = f"{d}/" if d else ""
            for item in (info.get("content") or []):
                if item.get("type") != "file":
                    continue
                size = item.get("size")
                if isinstance(size, int):
                    found[f"{prefix}{item.get('name')}"] = size
            if found:
                with probe_lock:
                    remote_sizes.update(found)

        run_pool(dirs, args.probe_jobs, probe_dir)
        probe_el = time.time() - t_probe

        for pair in pairs:
            _local, rel = pair
            size = remote_sizes.get(rel)
            if size is not None and size == local_sizes[rel]:
                skipped.append((rel, size))
            else:
                to_upload.append(pair)

    skip_bytes = sum(s for _r, s in skipped)
    todo_bytes = sum(local_sizes[rel] for _l, rel in to_upload)
    log(f"探测完成（{probe_el:.1f}s）：远端已一致 **跳过 {len(skipped)} 个**"
        f"（{skip_bytes / 1048576:.1f}MB），待传 {len(to_upload)} 个"
        f"（{todo_bytes / 1048576:.1f}MB）")

    if not to_upload:
        log("没有需要传的文件 —— 机器人代码已是最新")
        return 0

    for rel, size in sorted(skipped):
        log(f"[跳过] 远端一致  {size:>10} 字节  {rel}")

    # ---- 第二阶段：并发上传 ----
    lock = threading.Lock()
    state = {"ok": 0, "fail": 0, "bytes": 0}
    failed = []
    t0 = time.time()

    def upload(pair):
        local, rel = pair
        size = local_sizes[rel]
        with open(local, "rb") as f:
            data = f.read()
        if rel.lower().endswith((".sh", ".bash")) and b"\r\n" in data:
            data = data.replace(b"\r\n", b"\n")
        remote_path = f"{args.remote_root}/{rel}"
        err = None
        for attempt in range(1, max(1, args.tries) + 1):
            ok, err = put_one(client, remote_path, data)
            if ok:
                break
            if attempt < args.tries:
                time.sleep(1.5 * attempt)
        el = time.time() - t0
        with lock:
            if ok:
                state["ok"] += 1
                state["bytes"] += size
                n = state["ok"] + state["fail"]
                speed = state["bytes"] / 1048576 / max(el, 0.001)
                log(f"[{n}/{len(to_upload)}] {el:6.1f}s  {size / 1048576:6.2f}MB  "
                    f"累计 {speed:5.2f}MB/s  {rel}")
            else:
                state["fail"] += 1
                n = state["ok"] + state["fail"]
                failed.append((rel, err))
                log(f"[{n}/{len(to_upload)}] {el:6.1f}s  失败  {rel}  ← {err}")

    run_pool(to_upload, args.jobs, upload)

    el = time.time() - t0
    log("-" * 60)
    log(f"完成：上传成功 {state['ok']} / 失败 {state['fail']} / 跳过 {len(skipped)}"
        f"（共 {len(pairs)} 个），上传耗时 {el:.1f}s"
        f"（{state['bytes'] / 1048576:.1f}MB，"
        f"{state['bytes'] / 1048576 / max(el, 0.001):.2f}MB/s）")
    if failed:
        log("失败清单（重跑同一条命令即可；上一轮超时但已写成功的会自动跳过）：")
        for rel, err in failed:
            log(f"  {rel}  ← {err}")
        return 1
    log("全部对齐。核对命令：python tools/sync_nine_grid_to_robot.py --verify")
    return 0


if __name__ == "__main__":
    sys.exit(main())
