# Web 调试工具与 `cmd_keyboard.py`

本文件说明 `Functions` 目录下所有 Web 调试程序，以及终端键盘控制程序 `cmd_keyboard.py`。

这些程序主要用于视觉调试、定位调试和动作验证，不属于 `demoV4.py` 的主运行入口。除特别说明外，Web 页面只在浏览器中显示摄像头和检测结果；但带有“射门”“动作授权”按钮的页面可能会直接控制机器人，使用前必须确认场地安全。

## 运行前准备

程序应在 TonyPi 机器人上运行，并确认以下服务和文件可用：

```bash
sudo systemctl stop tonypi
cd /home/pi/TonyPi/Functions
```

需要的基础环境：

- Python 3
- TonyPi `hiwonder` SDK
- `numpy`
- `opencv-python`
- `Flask`
- `hiwonder.apriltag`（AprilTag 定位工具需要）
- 摄像头、舵机控制板和对应动作组

通用本地资源：

```text
CameraCalibration/calibration_param.npz
models/football_best_win.onnx
weights/best.onnx
```

其中：

- `models/football_best_win.onnx`：足球检测模型
- `weights/best.onnx`：球门柱检测模型
- `CameraCalibration/calibration_param.npz`：相机标定参数

机器人还必须预装动作组，例如 `stand`、`stand_slow`、`go_forward`、`turn_left`、`turn_right`、`left_shot_fast` 和 `right_shot_fast`。

## 工具总览

| 程序 | 默认端口 | 主要用途 | 默认是否控制机器人 |
|---|---:|---|---|
| `football_web.py` | 5000 | 足球检测画面和置信度显示 | 会初始化摄像头和舵机，主要用于检测 |
| `web_distance.py` | 5001 | 足球检测和单目距离估计 | 会初始化摄像头 |
| `goal_web.py` | 5001 | 足球、红色球门线和越线状态显示 | 会进行头部扫视，可能控制头部 |
| `goal_line_web.py` | 5002 | 独立球门线越线判断测试 | 不控制行走和踢球 |
| `locate_web.py` | 5001 | AprilTag 场地定位和机器人姿态显示 | 会控制头部扫视 |
| `football_kick_debug_web.py` | 5002 | 足球射门阈值和手动射门调试 | 手动点击射门按钮时会踢球 |
| `football_kick_debug_web_full.py` | 5002 | 完整射门状态机调试页面 | 默认仿真，授权后可控制动作 |
| `cmd_keyboard.py` | 无 | 终端键盘控制身体和头部 | 会直接控制机器人 |

同一时间不要启动默认端口相同的两个 Web 程序。可以使用命令行参数给两个射门调试页面改端口；其他页面需要修改脚本中的 `WEB_PORT`。

## 1. `football_web.py`

实时显示摄像头画面、足球检测框、置信度、检测中心点和推理状态。使用 `models/football_best_win.onnx`。

启动：

```bash
python3 football_web.py
```

浏览器访问：

```text
http://<机器人IP>:5000
```

主要页面接口：

- `/`：检测页面
- `/video_feed`：MJPEG 视频流
- `/status`：JSON 检测状态

置信度阈值在脚本中的 `CONF_THRESHOLD` 配置，当前默认值为 `0.1`，主要用于调试。

## 2. `web_distance.py`

在足球检测基础上，根据足球实际直径和相机焦距估计机器人到足球的距离。足球直径参数在脚本中为 `BALL_DIAMETER = 6.3` cm。

启动：

```bash
python3 web_distance.py
```

浏览器访问：

```text
http://<机器人IP>:5001
```

主要页面接口：

- `/`：检测和测距页面
- `/video_feed`：MJPEG 视频流
- `/status`：JSON 检测和距离信息

测距结果依赖相机标定参数和足球真实直径。更换摄像头后，需要重新标定或重新确认焦距。

## 3. `goal_web.py`

显示以下信息：

- 足球检测
- 红色球门线检测
- 足球到相机的估计距离
- 头部低头和左右扫描状态
- 足球是否越过球门线

启动：

```bash
python3 goal_web.py
```

浏览器访问：

```text
http://<机器人IP>:5001
```

依赖本地文件：

- `goal_detector.py`
- `models/football_best_win.onnx`
- `CameraCalibration/calibration_param.npz`

`goal_detector.py` 会通过 TonyPi 的 `yaml_handle.lab_file_path` 尝试读取红色阈值配置；如果读取失败，会使用代码中的默认红色阈值。该页面可能通过舵机调整头部视角，但不会自动行走或自动踢球。

页面接口：

- `/`：越线检测页面
- `/video_feed`：MJPEG 视频流
- `/status`：当前检测状态
- `/reset`：重置当前射门检测会话

## 4. `goal_line_web.py`

这是独立的球门线越线判断工具，使用足球模型和球门柱模型，同时显示球门线估计和越线判断结果。

启动：

```bash
python3 goal_line_web.py
```

浏览器访问：

```text
http://<机器人IP>:5002
```

依赖本地文件：

- `goal_line_judge.py`
- `models/football_best_win.onnx`
- `weights/best.onnx`
- `CameraCalibration/calibration_param.npz`

本程序不会控制机器人行走，也不会执行踢球动作。

页面可调整：

- 球门线方向：`above` 或 `below`
- 越线安全边距 `margin_px`
- 连续确认帧数 `confirm_frames`

页面接口：

- `/`：判断页面
- `/video_feed`：MJPEG 视频流
- `/status`：JSON 状态
- `/config`：更新判断参数
- `/reset`：重置判断状态

## 5. `locate_web.py`

使用 AprilTag 计算机器人在约 `1m × 1m` 场地中的位置和朝向，显示：

- `x`、`y` 坐标
- yaw 偏航角
- 使用的 Tag 数量
- 重投影误差
- 当前定位置信度

启动：

```bash
python3 locate_web.py
```

浏览器访问：

```text
http://<机器人IP>:5001
```

依赖本地文件：

- `locate_robot.py`
- `load_pos.py`
- `CameraCalibration/calibration_param.npz`

`load_pos.py` 中的 `FIXED_TAG_COORDS` 是场地 Tag 坐标的唯一来源。更换场地或移动 Tag 后，需要先更新这些坐标。

页面接口：

- `/`：场地定位页面
- `/video_feed`：摄像头视频流
- `/api/status`：JSON 定位状态
- `/api/head`：头部手动调整和复位
- `/api/auto_sweep`：打开或关闭自动扫视

网页键盘控制：

- `Tab`：切换自动扫视
- 方向键：手动调整头部
- 空格：头部回到默认位置

## 6. `football_kick_debug_web.py`

简化版足球射门阈值调试页面，主要用于：

- 查看足球检测框
- 查看足球中心点和射门目标窗口
- 调整安全距离、连续确认帧数和目标容差
- 手动执行一次左脚或右脚射门动作

默认只进行检测，网页点击射门按钮时才执行动作。

启动：

```bash
python3 football_kick_debug_web.py --port 5002
```

浏览器访问：

```text
http://<机器人IP>:5002
```

可用参数：

```bash
python3 football_kick_debug_web.py --help
python3 football_kick_debug_web.py --host 0.0.0.0 --port 5010
```

依赖：

- `football_kick_controller.py`
- `models/football_best_win.onnx`
- TonyPi `hiwonder.Camera`
- TonyPi `hiwonder.ActionGroupControl`

页面接口：

- `/`：调试页面
- `/video_feed`：MJPEG 视频流
- `/status`：检测和阈值状态
- `/config`：更新阈值
- `/kick`：执行一次射门；默认会先检查安全条件

如需强制执行网页选择的射门动作，接口可使用 `/kick?enforce=0`。这会绕过部分安全条件，只应在确认机器人周围安全时使用。

## 7. `football_kick_debug_web_full.py`

完整射门状态机调试页面，除了足球检测，还显示：

- 足球接近状态
- 射门脚选择
- 球门柱检测
- 球门线估计
- 状态机建议动作

启动：

```bash
python3 football_kick_debug_web_full.py --port 5002
```

浏览器访问：

```text
http://<机器人IP>:5002
```

启动后默认是仿真模式：

- `/controller/start`：启动状态机仿真
- `/controller/stop`：停止状态机并复位
- `/controller/arm`：授权或取消真实动作

只有在网页中明确授权状态机后，状态机才会发送真实动作。页面中的“手动射门”按钮仍然可能直接执行 `left_shot_fast` 或 `right_shot_fast`。

依赖：

- `football_kick_controller.py`
- `goalpost_detector.py`
- `red_goal_line_detector.py`
- `goal_line_judge.py`
- `models/football_best_win.onnx`
- `weights/best.onnx`

## 8. `cmd_keyboard.py`

终端键盘控制工具，不需要图形界面。它会打开摄像头，同时在终端读取单字符按键。

启动：

```bash
python3 cmd_keyboard.py
```

控制键：

| 按键 | 功能 |
|---|---|
| `w` | 前进 |
| `s` | 后退 |
| `a` | 左移 |
| `d` | 右移 |
| `q` | 左转 |
| `e` | 右转 |
| `u` | 头部向上 |
| `l` | 头部向下 |
| `p` | 拍照 |
| 空格 | 停止动作并回到默认姿态 |
| `Esc` 或 `Ctrl+C` | 退出 |

拍摄的照片会保存到：

```text
Functions/debug_vision_pic/
```

退出时程序会恢复终端设置、停止动作组并关闭摄像头。如果程序异常退出导致终端不回显，可以执行：

```bash
reset
```

## 端口冲突

默认端口如下：

```text
5000  football_web.py
5001  web_distance.py / goal_web.py / locate_web.py
5002  goal_line_web.py / football_kick_debug_web.py / football_kick_debug_web_full.py
```

同一时间只能运行一个使用相同端口的程序。两个 `football_kick_debug_web` 程序支持命令行改端口：

```bash
python3 football_kick_debug_web.py --port 5010
python3 football_kick_debug_web_full.py --port 5011
```

其他 Web 程序需要修改对应脚本中的 `WEB_PORT` 后再启动。

## 安全建议

1. 启动 Web 调试页前，先停止 `tonypi` 服务，避免多个程序同时控制摄像头或舵机。
2. 首次只查看画面和状态，不要点击射门、动作授权或手动控制按钮。
3. 使用 `football_kick_debug_web_full.py` 时，确认页面显示“仿真”后再调试状态机；只有场地安全时才授权真实动作。
4. 使用 `cmd_keyboard.py` 前，确认机器人四周没有人员和障碍物，并准备好急停方式。
5. 结束后按 `Ctrl+C`，确认摄像头和动作组已停止。

