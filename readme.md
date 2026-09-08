# 具身智能机器人代码开发（robot-dev）

啊，伟大的搭档！我又来了！

本仓库用于放置 TonyPi 机器人上运行的代码。目前有两个已实现的关卡：

- `goodluck`：AprilTag 视觉定位 + 固定路点闭环导航（稳定方案）；
- `nine_grid`（数字宫格）：颜色主判 + GN 地图定位，按 1→7 顺序到达随机布局的七块数字面板。

## 这是什么

一个「智能闯关」机器人项目：机器人从赛道入口出发，靠 **AprilTag 视觉定位 + 闭环导航**，
沿规定线路依次经过评分点/转向点、完成转弯、走出出口。

仓库还包括一套为攻克后续关卡准备的**通用视觉识别能力（`vision/`）**：巡线、识别数字、
识别球体、YOLO 检测器等。

## 当前算法状态（2026-09）

- **当前方案**：固定 `ROUTE` 路点 + `navigate_to_target()` 闭环导航，代码在
  `levels/goodluck.py`。
- **为什么是固定路点**：比赛规则已明确固定路线，不需要通用 A* 自主规划；固定路点
  逻辑简单、参数少、实机可预测。
- **历史实验**：A* / v6 混合导航 / 可执行直线-圆弧路径规划等曾在一段时间内尝试，
  现已在提交 `80aedd6` 整体回退到稳定边界 `bcf4547` 的固定路点版本。
- **实验代码去向**：相关代码已从当前工作区移除，仅保留在 git 历史；方案文档保留在
  `docs/关卡算法/可执行直线-圆弧路径规划方案.md` 作为留档。
- **目录整理**：2026-09 起代码按职责收进 `core/ levels/ vision/ sim/ tools/`；
  运行/标定产物统一写入 `archive/result/`（gitignore），历史快照在 `archive/backup/`。

## 快速开始

```bash
# 运行 goodluck 关卡（真机，完整 ROUTE）
python main.py goodluck

# 走到停靠点 4 后提前结束（出口段交给其它方法）
python main.py goodluck --end-at-last-stop

# 运行数字宫格关卡（真机）
python main.py nine_grid

# 运行模拟器（PC，验证算法）
python -m sim.goodluck_sim            # goodluck：2D 可视化 + 位姿桩
python -m sim.nine_grid_sim           # 数字宫格：合成相机图像，跑真实视觉链路
python -m sim.nine_grid_sim --random-layout --seed 11   # 随机布局探索
python -m sim.nine_grid_view          # 数字宫格图形界面（相机窗格 + 俯视图，可暂停/单步）
python -m sim.nine_grid_view --headless   # 无图形环境（SSH）退回无头运行

# 批量跑模拟变体（可选，goodluck）
python -m sim.run_sim_variants ideal 1
```

> 当前 `STOP_TIME = 0`，即到达路点不停留（2026-08-30 提速改造）。
> 若比赛要求恢复停靠评分，把 `levels/goodluck.py` 中 `STOP_TIME` 改回 `3` 即可。

## 代码结构

```
robot-dev/
├── main.py            # 入口：LEVELS 关卡注册表 + 关卡分发
├── core/              # 运行共享核心
│   ├── robot_core.py      # RobotState（定位/动作/头部/几何/取帧）
│   ├── camera_config.py   # 相机内参/头部舵机常量
│   ├── ground_homography.py # 地面单应（数字宫格定位基础）
│   ├── multiview_pose.py  # 多视角联合位姿求解
│   ├── trace.py           # 真机轨迹记录
│   └── paths.py           # 项目路径 / RESULT_DIR 统一入口
├── levels/            # 关卡层
│   ├── goodluck.py        # 赛道数据 + ROUTE + 导航算法
│   └── nine_grid.py       # 数字宫格：布局扫 + GN 定位 + FSM
├── vision/            # 视觉识别能力：巡线/颜色/数字/九宫格/YOLO 检测器
│   └── nine_grid_detector.py  # 七色面板 + HOG/SVM 数字仲裁
├── sim/               # 模拟器（只提供世界/传感器/带噪声动作，算法走真实代码）
│   ├── goodluck_sim.py    # goodluck：位姿桩 + matplotlib 可视化
│   ├── nine_grid_sim.py   # 数字宫格：合成相机图像 + 真实视觉链路
│   ├── nine_grid_view.py  # 数字宫格图形界面（相机窗格 + 俯视图，暂停/单步/重开）
│   └── run_sim_variants.py
├── tests/             # 单测 + 仿真集成测试（断言在 tests/，环境在 sim/）
├── tools/             # 开发/调试/标定工具
│   ├── debug_vision.py / gen_digit_templates.py
│   ├── calib_ninegrid.py  # 数字宫格地面单应点击标定
│   ├── camera_preview.py
│   ├── sync_to_robot.py / exec_on_robot.py / pull_from_robot.py
│   └── field_calib/       # 真机标定/多视角/场地工具
├── models/            # 模型权重（含 nine_grid/digit_classifier_mask.pkl）
├── docs/              # 文档（见 docs/README.md）
├── reference code/    # 赛道方/参考队代码（只读）
├── archive/           # 归档：backup/（历史快照）+ result/（运行产物，gitignore）
└── .zcode/            # 本地会话计划（一般不动）
```

## 常用工具命令

```bash
# 摄像头预览
python -m tools.camera_preview [--stream --apriltag]

# 独立 PnP 对照基线
python -m tools.simple_pnp_run [--end-at-last-stop]

# 视觉检测器离线调试
python tools/debug_vision.py --help

# 数字模板生成
python tools/gen_digit_templates.py --help

# 真机标定/多视角（示例）
python -m tools.field_calib.collect_multi_view
python -m tools.field_calib.optimize_multi_view --data archive/result/multiview_*.npz
python -m tools.field_calib.survey_field archive/result/multiview_A.npz ...
```

## 文档

文档统一放在 [`docs/`](./docs/) 下，按主题分类、各司其职。
**入口见 [`docs/README.md`](./docs/README.md)**，其中列有每份文档的职责与推荐阅读路径。

当前 goodluck 算法细节见 [`docs/关卡算法/introduction.md`](./docs/关卡算法/introduction.md)，
定位原理见 [`docs/关卡算法/定位逻辑详解.md`](./docs/关卡算法/定位逻辑详解.md)。

## 日后任务

- **完善视觉识别**：巡线、数字、球体等仍需实机调参；YOLO 尚未接入实际模型。
- **攻克其它关卡**：已准备 `vision/` 视觉识别能力，详见 `docs/视觉能力/`。
- **按比赛规则微调 goodluck**：现场若需要恢复停靠或调整阈值，在 `levels/goodluck.py`
  中集中修改，并用 `python -m sim.goodluck_sim` 先验证。

开发愉快，开学愉快！

钉小呆Xiaodai, 20260816（2026-09 更新算法状态与目录整理）
