# 循迹弯 2/3/4 修复实施记录（260927）

依据：pulled_patrol_260927 回放分析（见会话分析结论）。改动文件：
- `levels/football_codes/RedLinePatrolV3.py`（全部算法改动）
- `test_programs_NoUseInMain/redline_route_debug.py`（面板展示新常量）

## 已实施的修改

M1 删除"视觉与查表强矛盾→ci 回退→改按视觉方向转"，视觉判向只保留记录打印；
   顺序表耗尽（corner_turn==0）恒直行。
M2 精修阶段横条分支：禁用 offset_x 判向，改为按查表方向小步续转
   ≤ CORNER_FINE_BAR_MAX_STEPS(2) 次，用尽即收尾。
M3 转弯后确认期：竖线可见→heading 判向微调（保留，含 12 次上限）；竖线不可见
   且横条残留→前进出弯，上限 POST_TURN_MAX_EXTRA_FORWARDS(8) 步后恢复正常循迹。
M4 rearm 门限：转弯完成后横条须连续退场 CORNER_BAR_FREE_REARM_FRAMES(5) 帧，
   才允许触发下一弯（PatrolSession.corner_rearmed / bar_free_frames）。
M5 触发增加居中门限 |center_error| ≤ CORNER_TRIGGER_CENTER_DEADBAND(20px)；
   接近路口（corner_ready）时横移对中死区收紧为 20px（平时 42px）。
M6 模拟器决策面板新增上述常量展示。

新增常量：CORNER_TRIGGER_CENTER_DEADBAND=20、CORNER_FINE_BAR_MAX_STEPS=2、
CORNER_BAR_FREE_REARM_FRAMES=5、POST_TURN_MAX_EXTRA_FORWARDS=8。

## 回放验证结果（pulled_patrol_260927，4452 帧全量回放，修改前→后）

| 断言 | 修改前 | 修改后 |
|---|---|---|
| ci 回退次数 | 13（弯1区 10 + 尾部 3） | **0**（ci 单调不减） |
| 精修反向判向（反侧回摆/偏侧续转/横条对中） | 弯2/3 精修 12 步全是反侧回摆 | **0**；改为"同向小步"共 4 次（每弯 2 次） |
| 确认期竖线未现仍按 offset 微调 | 弯1 后 132 次原地微调 | **0**；改为"前进出弯"≤8 步 |
| 弯1 转弯步数 | 主4+精修6=10 步（且被假重触发 10 次） | 主4+精修2=6 步，**只触发 1 次** |
| 弯2/3/4 串弯（2.7s 连吃） | 发生 | 657-725 偏姿态横条前不再触发（居中门限+rearm 拦下，走"路口横移"居中） |
| 尾部假右转（帧 3114/3142/3170） | 3 轮（回退后按视觉右转） | **0**；3116 居中且视觉与查表一致时正常触发弯2一次 |
| py_compile | — | 通过 |

## 已知回放局限（真机复验项）

- 回放是决策回放器：决策不移动机器人，录像为修复前采集；"横条 4 步内不消失"、
  "前进出弯 8 步后横条仍在"等是回放静态画面的产物，真机上机器人真实运动会改变
  画面，横条会随前进退场。
- 弯 3/4 在本录像中因机器人全程右偏 20~33px 被居中门限拦下，属预期行为；真机
  验证时机器人会真实执行横移对中，之后应能依次触发。

## 真机部署与验收步骤（未执行，待进行）

1. 本地 py_compile（已通过）。
2. `tools/_deploy_patrol_v3.py` 部署到机器人（自动备份+字节校验）。
3. 机器人端 py_compile；`SAVE_DEBUG_FRAMES=True` 跑一次 demoV4 巡线。
4. `tools/pull_from_robot.py` 拉回新帧，回放验收：四弯各触发一次、ci 依次
   1→2→3→4、转弯后横条真实退场、尾部不丢线或丢线可快速找回。
