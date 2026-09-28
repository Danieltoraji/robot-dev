# tag 分析器重构：复用关卡内部逻辑 + 协作纪律入档

## 结论回顾

yolo/digit/line 三分析器**已复用** `vision/` 共享包（与关卡同一份代码），无需改动。唯一"另写"的是 **tag 分析器**（apriltag 检测 + PnP + 门控的本地简化复刻，因当时顾虑 import core.robot_core 的串口副作用）。评估结论：复用可行——import 的唯一副作用是打开串口句柄（只读不写），机器人上多进程共持串口本就是现状；且完整门控（`pnp_pose_problems`）比复刻版更可信。

## 实现（只改 `tools/debug_server.py`，~70 行）

1. **双模式 tag 分析器**：
   - `_load_tag` 首选 **core 模式**：`import core.robot_core` + `from levels.goodluck import tag_poses` + `RobotState(tag_poses=...)`；前提校验 `rc.apriltag is not None`
   - 失败（PC 无 SDK/apriltag，或显式 `--no-core`）→ **local 模式**：现有本地实现原样保留（apriltag + 简化 PnP 复刻）
   - 状态栏标注模式："core 复用模式" / "本地简化实现（降级）"
2. **core 模式 run_tag**（检测需要文件路径，`Analyzers.run` 增加 `photo_path` 参数透传，其余分析器忽略）：
   - `state.detect_apriltag(photo_path)` —— 关卡同款检测
   - 逐标签 `TAG_CORNER_PERM` 对齐 + `state.tag_poses` 配对（未知 id 跳过并计数）
   - `rc.solve_pnp_pose(objlist, imglist)` 解位姿
   - `rc.pnp_pose_problems(pos, ori, reproj, n_tags=known)` —— 完整门控（替代复刻版 3 条简化检查）
   - JSON 输出结构不变（tags/pose/problems），**前端零改动**
3. **串口安全**（docstring 写明）：import 打开串口句柄但调试服务器从不写；如现场出现异常，`--no-core` 一键退回本地实现
4. **文档**：《视觉调试指南》§3.6 补两点——三分析器与关卡的代码同源关系（含"关卡改参数不自动同步调试参数"的维护点）；**协作纪律**：多工作树向同一机器人同步时后写覆盖（本次 line_detector 分叉实例），换会话前先拉齐

## 明确不做

- 不改 core.robot_core / levels / main.py 任何一行
- 不删本地简化实现（PC 降级路径）
- yolo/digit/line 三个分析器与 vision/ 包不动

## 验收

1. 机器人重启服务器：`/api/state` 显示 core 复用模式；画面中 goodluck 标签（151~163）出现角点，pose 带完整 problems 门控（对照关卡内部定位结果应一致）
2. 闯关运行时舵机指令正常（串口只读共持无干扰）
3. PC 上（无 SDK/apriltag）服务器启动自动落 local 模式，tag 分析器状态明确提示降级