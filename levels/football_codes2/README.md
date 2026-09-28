# TonyPi Demo V4

这是 TonyPi 机器人赛道展示流程的最终版入口。程序先完成红线巡线，再通过 AprilTag 导航到两个球门，完成足球检测、球门柱对齐和射门，最后沿出口路线离场。

## 运行流程

```text
RedLinePatrolV3 红线巡线
        ↓
终点候选确认，扫描 Tag 103
        ↓
AprilTag 路线导航
        ↓
足球检测与球门柱对齐
        ↓
每个球门射门 1 次
        ↓
不判断是否进球，直接进入下一阶段
        ↓
第二个球门完成后沿出口路线离场
```

V4 的关键规则是：每个球门只执行一次射门；踢球动作完成后不等待进球判定，直接推进路线。

## 最终版文件

提交时请保留以下目录结构：

```text
Functions/
├── demoV4.py
├── RedLinePatrolV3.py
├── patrol_end_recovery.py
├── tag_walk_demo.py
├── tag_route_demo.py
├── goalpost_detector.py
├── football_kick_controller.py
├── goal_line_judge.py
├── red_goal_line_detector.py
├── CameraCalibration/
│   ├── CalibrationConfig.py
│   └── calibration_param.npz
├── models/
│   └── football_best_win.onnx
└── weights/
    └── best.onnx
```

### 文件作用

| 文件 | 作用 |
|---|---|
| `demoV4.py` | 总控入口，串联巡线、Tag 导航和射门 |
| `RedLinePatrolV3.py` | 红线识别、巡线和直角弯处理 |
| `patrol_end_recovery.py` | 巡线终点复核、Tag 103 检测和漏弯恢复 |
| `tag_route_demo.py` | AprilTag 路线状态机 |
| `tag_walk_demo.py` | Tag 导航、足球搜索、球门柱对齐和射门衔接 |
| `goalpost_detector.py` | 使用 `weights/best.onnx` 检测球门柱 |
| `football_kick_controller.py` | 使用足球模型控制接近、对齐和射门 |
| `goal_line_judge.py` | 球门线越线判断工具 |
| `red_goal_line_detector.py` | 红色球门线估计工具 |
| `CameraCalibration/CalibrationConfig.py` | 相机标定路径和标定参数配置 |
| `CameraCalibration/calibration_param.npz` | 相机内参和畸变参数 |
| `models/football_best_win.onnx` | 足球检测模型 |
| `weights/best.onnx` | 球门柱检测模型 |

## 不需要提交的文件

以下文件不是 V4 运行链的一部分，通常不需要一起提交：

- `demoV2.py`、`demoV3.py`
- `FootballKick.py`、`finalkick.py` 等旧版独立程序
- `RedLinePatrolV3_backup_*.py`
- `captures/`
- `__pycache__/`
- `Functions/Functions/` 内的重复副本
- `.pt` 模型、其他备用 `.onnx` 模型和测试脚本

不要把外层 `Functions/` 和内层 `Functions/Functions/` 的同名文件混用；它们不是完全相同的版本。最终入口使用外层文件。

## 运行环境

程序需要运行在安装了 TonyPi 软件环境的机器人上，并依赖：

- TonyPi `hiwonder` SDK
- Python 3
- `numpy`
- `opencv-python`
- `hiwonder.apriltag`
- 机器人动作组文件，例如 `stand`、`go_forward`、`turn_left`、`left_shot_fast` 等

动作组不是 Python 文件的一部分，需要确认机器人系统中已经安装完整。

## 部署

推荐将程序放在机器人目录：

```text
/home/pi/TonyPi/Functions/
```

部署后确认模型和标定文件存在：

```bash
ls -l /home/pi/TonyPi/Functions/models/football_best_win.onnx
ls -l /home/pi/TonyPi/Functions/weights/best.onnx
ls -l /home/pi/TonyPi/Functions/CameraCalibration/calibration_param.npz
```

`CalibrationConfig.py` 默认使用以下标定参数路径：

```text
/home/pi/TonyPi/Functions/CameraCalibration/calibration_param.npz
```

如果实际安装目录不同，需要同步修改 `CameraCalibration/CalibrationConfig.py` 中的 `calibration_param_path`。

## 运行方式

运行前先停止 TonyPi 的默认控制服务，避免多个程序同时控制舵机：

```bash
sudo systemctl stop tonypi
cd /home/pi/TonyPi/Functions
```

### Dry-run 模式

不加 `--run` 时，程序不会执行机器人动作，只进行流程和视觉逻辑运行，并打印动作名称：

```bash
python3 demoV4.py
```

### 实机模式

确认摄像头、动作组和场地均已准备好后运行：

```bash
python3 demoV4.py --run
```

如果需要在有桌面的环境中显示巡线调试窗口，可以使用：

```bash
python3 demoV4.py --run --display
```

没有桌面环境时不要使用 `--display`。

## 常用参数

```bash
python3 demoV4.py --help
```

常用参数如下：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--run` | 关闭 | 真实执行机器人动作；不加则为 dry-run |
| `--display` | 关闭 | 显示巡线调试窗口 |
| `--patrol-end-confirm-seconds` | `0.8` | 脚下连续无红线的终点确认时间 |
| `--patrol-end-tag-confirm-frames` | `3` | Tag 103 连续确认帧数 |
| `--patrol-max-seconds` | `300` | 巡线阶段最大运行时间 |
| `--tag-max-seconds` | `600` | Tag/射门阶段最大运行时间 |
| `--ball-conf` | `0.40` | 足球检测置信度阈值 |
| `--goalpost-confirm-frames` | `2` | 射门前球门柱确认帧数 |
| `--ball-loss-timeout` | `8` | 连续丢球后放弃当前球门的时间 |

例如，适当降低足球检测阈值：

```bash
python3 demoV4.py --run --ball-conf 0.35
```

## 调试建议

1. 首次运行先使用 dry-run：

   ```bash
   python3 demoV4.py
   ```

2. 确认模型文件和标定文件路径正确。
3. 确认机器人没有被 `tonypi` 服务或其他程序占用。
4. 确认动作组名称完整，尤其是巡线、转向、侧移和左右脚射门动作。
5. 实机运行时保持急停或断电措施可用，不要在机器人周围放置障碍物。

## 常见问题

### 找不到模型

检查以下文件是否存在，并确认大小不是 0：

```bash
ls -lh models/football_best_win.onnx
ls -lh weights/best.onnx
```

### 找不到相机标定文件

检查：

```bash
ls -lh CameraCalibration/calibration_param.npz
```

如果程序提示的路径是 `/home/pi/TonyPi/Functions/...`，请把标定文件放到该路径，或修改 `CalibrationConfig.py`。

### 舵机没有动作或动作冲突

确认已经执行：

```bash
sudo systemctl stop tonypi
```

同时检查对应动作组是否安装在 TonyPi 的动作组目录中。

### Tag 一直识别不到

检查摄像头是否正常、Tag 编号和摆放方向是否符合路线配置，并避免使用过暗、反光或严重模糊的画面。

## 安全提示

`--run` 会直接控制真实机器人。第一次部署或修改参数后，应先在空旷区域使用 dry-run 检查流程，再进行低风险实机测试。

