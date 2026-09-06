# 机器人 Jupyter 访问与操作指南（PC ↔ RPi）

> 面向任何人 / AI：如何从 PC 操作机器人（同步代码、远程执行、文件拉取），
> 含底层协议全文档与已知坑。本文 2026-09-05 全部实测验证。
> 网络不通先看《[机器人网络连接排查](机器人网络连接排查.md)》。

---

## 0. 速查表（常用任务 → 一条命令）

在 **PC 仓库根目录**执行（`F:\coding\Projects\RoboTrack\robot-dev`）：

| 任务 | 命令 |
|---|---|
| 同步代码到机器人（增量，只传有变化的） | `python tools/sync_to_robot.py` |
| 同步前干跑（只列出将传什么） | `python tools/sync_to_robot.py --check` |
| 在机器人上执行 shell 命令 | `python tools/exec_on_robot.py --cmd "uname -a"` |
| 在机器人上执行 Python 代码 | `python tools/exec_on_robot.py --code "print(1+1)"` |
| 拉取机器人上的文件/目录到 PC | `python tools/pull_from_robot.py --remote dataset_raw --dest archive/result/dataset_raw` |
| 拉取前干跑 | `python tools/pull_from_robot.py --remote dataset_raw --dest <目录> --check` |
| 网页版 Jupyter（人工浏览/下载） | http://192.168.31.209:8888/tree/Robot_control_self_module （密码 `pi`） |

---

## 1. 连接参数

| 项 | 值 |
|---|---|
| Jupyter 地址 | `http://192.168.31.209:8888` |
| 密码 | `pi` |
| 机器人代码目录 | `/home/pi/Robot_control_self_module`（Jupyter API 路径 `Robot_control_self_module/...`） |
| PC 端工具 | `tools/sync_to_robot.py`、`tools/exec_on_robot.py`、`tools/pull_from_robot.py` |
| PC 端依赖 | sync/pull **零依赖**（纯标准库）；exec 需 `pip install websocket-client`（已装） |
| IP 获取方式 | DHCP 动态分配（建议路由器做 IP 保留；换网络后 IP 可能变，工具用 `--host` 覆盖） |

---

## 2. 三个工具详解

### 2.1 sync_to_robot.py —— PC → 机器人代码同步

- **增量同步**：逐文件与远端比对字节，只上传缺失/不同的文件；无变化跳过；
- **不做远端删除**（设计如此，防误删机器人侧产物）；多余文件需人工清理；
- 自动跳过 `__pycache__/.git/archive/release` 与 `.pyc/.npz` 等；
- 默认同步集合：`main.py + core/ vision/ levels/ tools/ models/`；
- 每个上传都带落盘确认，失败退出码 1。

```bash
python tools/sync_to_robot.py                   # 全量默认集合
python tools/sync_to_robot.py --check           # 干跑
python tools/sync_to_robot.py --paths core main.py   # 只同步指定路径
python tools/sync_to_robot.py --host http://IP:8888 --password xxx  # 换地址
```

### 2.2 exec_on_robot.py —— 远程执行（内核会话方式）

- 每次调用：启动一个全新内核 → 执行代码 → 收集全部输出 → **删除会话**（无残留）；
- RPi 上内核启动需 3~10s（脚本已内置等待）；
- `--cmd` 是 shell 命令，机器人在服务端用 `subprocess.run(shell=True)` 包一层，
  自动回显 stdout/stderr 和退出码；`--timeout` 同时是 subprocess 和内核等待的超时。

```bash
python tools/exec_on_robot.py --cmd "ls /home/pi" --timeout 60
python tools/exec_on_robot.py --code "import cv2; print(cv2.__version__)" --timeout 120
```

- **转义规则**：`--cmd` 里的双引号和反斜杠会被脚本自动转义；但**单引号嵌套**
  尽量避免（shell+Python 双层引号），复杂逻辑写成一行 Python 用 `--code`；
- 长任务（训练、批量拍照）务必给足 `--timeout`（默认 120s）。

### 2.3 pull_from_robot.py —— 机器人 → PC 文件拉取

- 递归遍历远端目录，逐文件下载；**已存在且字节一致的跳过**（可中断续拉）；
- 以**机器人为准**刷新本地时，先删本地目录再拉（工具不删本地多余文件）。

```bash
python tools/pull_from_robot.py --remote dataset_raw --dest archive/result/dataset_raw
```

---

## 3. 底层协议（供 AI / 开发者重实现或扩展）

三个工具本质是对 Jupyter Server REST + WebSocket API 的最小封装。
以下协议细节足以**不依赖任何现成代码**从零实现。

### 3.1 登录（取得会话）

1. `GET /login` → 从返回 HTML 中取 `name="_xsrf" value="..."`（或同名 cookie）；
2. `POST /login`，form 编码：`_xsrf=<token>&password=pi`；
   成功后 cookie jar 中出现 `username-<IP>-8888`（cookie 名含主机 IP，IP 变则名变）；
3. 之后所有**写操作**（POST/PUT/DELETE）必须带请求头 `X-XSRFToken: <token>`。

### 3.2 Contents API（文件/目录操作）

| 操作 | 请求 | 说明 |
|---|---|---|
| 列目录 | `GET /api/contents/<path>?content=1` | 返回 JSON，`type: directory/file`，子项在 `content` 数组 |
| 读文件 | `GET /api/contents/<path>?content=1` | 文本文件 `format:"text"`；二进制 `format:"base64"` |
| 写文件 | `PUT /api/contents/<path>` body `{"type":"file","format":"base64","content":"<b64>"}` | 新建返回 **201**，覆盖返回 **200** |
| 建目录 | `PUT /api/contents/<path>` body `{"type":"directory"}` | **父目录不存在时写文件会失败**，需先逐级建目录 |
| 删除 | `DELETE /api/contents/<path>` | 返回 **204**；无回收站，慎用 |

- 路径中的中文/特殊字符需 URL encode；
- 一次 PUT 建议 ≤ 数十 MB（大文件先分块或压缩）。

### 3.3 内核会话执行（远程执行的标准姿势，推荐）

```
1. POST /api/sessions
   body: {"kernel":{"name":"python3"},"name":"task名","type":"notebook",
          "path":"exec_on_robot/probe.ipynb"}
   → 返回 JSON，记下 id（会话 id）与 kernel.id（内核 id）

2. 等待内核就绪：RPi 上 sleep 3~10s

3. WebSocket 连接 ws://<host>/api/kernels/<kernel_id>/channels
   请求头带登录 Cookie

4. 发 execute_request（shell channel）：
{
  "header": {"msg_id":"<uuid>", "username":"pc", "session":"<uuid>",
             "date":"<ISO时间>", "msg_type":"execute_request", "version":"5.3"},
  "parent_header": {}, "metadata": {},
  "content": {"code":"<Python代码>", "silent":false, "store_history":false,
              "user_expressions":{}, "allow_stdin":false, "stop_on_error":true},
  "buffers": [], "signature": "", "channel": "shell"
}
   ※ signature 留空即可（该 Jupyter 未启用 HMAC 签名校验）

5. 在同一 WebSocket 上（iopub 混流）读消息，只认 parent_header.msg_id 匹配的：
   - stream          → content.text（stdout/stderr）
   - execute_result  → content.data["text/plain"]
   - error           → content.ename/evalue/traceback
   - status idle     → 执行结束，退出循环

6. DELETE /api/sessions/<会话id>   ← 必做，否则内核残留占内存
```

### 3.4 终端 API（terminado）——**不推荐**

`POST /api/terminals` 建终端 → `ws /terminals/websocket/<name>`。
消息格式为 `["stdout", "..."]` 数组。**真机上实测不稳定**：shell 启动横幅吞掉
过早发送的命令、ANSI 转义噪声大、recv 易超时。需要跑 shell 时用
3.3 的会话方式包 `subprocess`，不要用终端 API。

---

## 4. 机器人环境事实（2026-09-05 实测）

| 项 | 值 |
|---|---|
| 硬件 | Raspberry Pi 5 Model B Rev 1.1，aarch64 |
| 系统 Python | 3.11.2（系统 `python3` 与 `/home/pi/jupyter-env/bin/python3` 均可用） |
| Jupyter venv | `/home/pi/jupyter-env`（Jupyter 内核即用它） |
| 关键包 | numpy 1.23.5 / opencv-python 4.9.0.80 / **onnxruntime 1.23.2 / ultralytics 8.3.221** / apriltag 0.0.16 / scipy 1.11.4 |
| sudo | `sudo -n` 免密可用 |
| 磁盘 | 28GB，余 ~8GB（2026-09-05） |
| 相机 | fswebcam 静态照（`-r 2592x1944 -S 3`），设备 `/dev/video0` |

**解释器选择**：跑视觉/推理相关命令显式用 `/home/pi/jupyter-env/bin/python3`
（包版本有保证）；跑采集/舵机类系统 python3 已验证可用。

---

## 5. 已知坑与对策（全部踩过）

1. **相机被占用静默失败**：视频流工具持有 `/dev/video0` 时，fswebcam 退出码
   0 但不产生文件。对策：采集脚本已带落盘校验（报"拍照异常"）；
   排查 `fuser -v /dev/video0` 找到占用 PID，`kill <pid>` 释放；
   **拍摄期间关闭视频流/预览工具**。
2. **WiFi 省电吞包**：症状是"经常连不上、时好时坏"。判据与修复见
   《[机器人网络连接排查](机器人网络连接排查.md)》（已修复并持久化）。
   三个工具均已内置 3 次退避重试。
3. **内核启动延迟**：会话创建后立刻发请求会丢失。工具已内置等待；
   自己实现时 sleep ≥3s 或轮询 `/api/kernels/<id>`。
4. **PC 侧 Git Bash heredoc 吃反斜杠**：`<<'EOF'` 中 `\\` 会变 `\`，
   内嵌多行 Python 会被破坏。对策：把代码 Write 成临时 .py 文件再执行。
5. **exec 的双层引号**：`--cmd` 内容走"PC shell → Python 参数 → 机器人 shell"
   三层，嵌套引号易错。复杂任务优先 `--code` 写 Python。
6. **同名覆盖**：Contents API 的 PUT 直接覆盖同名文件且无确认。
   批量重命名/合并目录时先查目标名是否已存在（2026-09-05 曾因序号冲突
   覆盖掉 1 张负样本）。
7. **数据一致性方向**：机器人是采集源头，"以云端（机器人）为准，覆盖本地"——
   本地镜像刷新 = 删除后整拉，不做增量合并。

---

## 6. 给 AI 操作者的边界提醒

- 大多数操作（同步、编译验证、拍照、拉取、诊断）**直接执行**，无需询问；
- **破坏性操作先确认**：删除机器人上的文件/目录、kill 未知进程、
  覆盖未备份的数据、apt/pip 改动环境；
- 远端代码目录以 **PC 仓库为唯一真源**（sync 单向 PC→机器人）；
  反向发现"机器人上多出来的文件"时，先弄清来源再决定去留；
- 长任务用 `--timeout` 给足时间而不是反复重试；内核会话每次用完即删；
- 所有工具失败时退出码非 0，脚本化时检查之。

---

## 7. 相关文档

- 《[机器人网络连接排查](机器人网络连接排查.md)》——WiFi 省电案例与诊断顺序
- 《[足球球门数据采集规范](../视觉能力/足球球门数据采集规范.md)》——采集命令与相机注意事项
- `docs/视觉能力/足球与球门识别训练部署方案.md` ——部署链路（模型 → RPi）
