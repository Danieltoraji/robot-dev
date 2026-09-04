# 具身智能机器人代码开发（robot-dev）

啊，伟大的搭档！我又来了！

本仓库用于放置 TonyPi 机器人上运行的代码。目前有一个已实现的关卡 `goodluck`，
采用 **AprilTag 视觉定位 + 固定路点闭环导航** 的稳定方案。

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

# 运行模拟器（PC，验证算法）
python -m sim.goodluck_sim

# 批量跑模拟变体（可选）
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
│   ├── multiview_pose.py  # 多视角联合位姿求解
│   ├── trace.py           # 真机轨迹记录
│   └── paths.py           # 项目路径 / RESULT_DIR 统一入口
├── levels/            # 关卡层：goodluck.py（赛道数据 + ROUTE + 导航算法）
├── vision/            # 视觉识别能力：巡线/颜色/数字/YOLO 检测器
├── sim/               # 模拟器
│   ├── goodluck_sim.py    # SimRobotState 继承重写 I/O
│   └── run_sim_variants.py
├── tools/             # 开发/调试/标定工具
│   ├── debug_vision.py / gen_digit_templates.py
│   ├── camera_preview.py
│   ├── simple_pnp_run.py
│   └── field_calib/       # 真机标定/多视角/场地工具
├── archive/           # 归档：backup/（历史快照）+ result/（运行产物，gitignore）
├── docs/              # 文档（见 docs/README.md）
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
