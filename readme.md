# 具身智能机器人代码开发（robot-dev）

啊，伟大的搭档！我又来了！

本次更新主要内容：规范了项目结构，增加了视觉功能，整理了文档。

视觉功能包括：巡线、识别数字、识别球体、YOLO 检测器。前三者需要实机拍摄图片进一步调参，最后一个还没有接入视觉模型。

本仓库用于放置 TonyPi 机器人上运行的代码。

## 这是什么

一个「智能闯关」机器人项目：机器人从赛道入口出发，靠 **AprilTag 视觉定位 + 闭环导航**，
沿规定线路依次停靠评分点、完成转弯、走出出口。目前有一个已实现的关卡 `goodluck`，
以及一套为攻克后续关卡准备的**通用视觉识别能力（`vision/`）**。

## 快速开始

```bash
# 运行 goodluck 关卡（真机）
python main.py goodluck

# 运行模拟器（PC，验证算法）
python goodluck_sim.py
```

## 代码结构

```
robot-dev/
├── main.py            # 入口：LEVELS 关卡注册表 + 关卡分发
├── robot_core.py      # 通用核心：RobotState（定位/动作/头部/几何/取帧）
├── levels/            # 关卡层：goodluck.py（赛道数据 + 导航算法）
├── vision/            # 视觉识别能力：巡线/颜色/数字/YOLO 检测器
├── tools/             # 调试工具：debug_vision.py / gen_digit_templates.py
├── goodluck_sim.py    # 模拟器：SimRobotState 继承重写 I/O
└── docs/              # 文档（见 docs/README.md）
```

## 文档

文档统一放在 [`docs/`](./docs/) 下，按主题分类、各司其职。
**入口见 [`docs/README.md`](./docs/README.md)**，其中列有每份文档的职责与推荐阅读路径。

## 日后任务

- **完善视觉识别**：需要进行实机调参。此外，YOLO还没有接入实际模型，需要我们添加模型并进行训练。
- **攻克其它关卡**：已准备 `vision/` 视觉识别能力（巡线、识别数字、识别球体、YOLO 目标检测），详见 `docs/视觉能力/`。
- **继续优化 goodluck 算法**：压缩时间、批量直行调优等（动作组标定已于 2026-08-25 实机完成，决策算法已切换到实测可靠动作集），详见 `docs/关卡算法/introduction.md`。

开发愉快，开学愉快！

钉小呆Xiaodai, 20260816
