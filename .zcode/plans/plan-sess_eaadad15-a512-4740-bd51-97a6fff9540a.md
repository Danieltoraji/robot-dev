## 目标（只改 levels/football_codes/RedLinePatrolV3.py，模拟器自动同步验证）

### 修改 1：转弯提速 —— 缩短转弯期每步等待 + 小步→大步升级

1. 新增常量：
   - `TURN_MAIN_STEP_SETTLE = 0.3`（主转开环不看视觉，每步只等动作完成即可）
   - `TURN_FINE_STEP_SETTLE = 0.5`（精修每步只需一张新帧，15fps 下足够）
   - `FINE_SMALL_STREAK_ESCALATE = 2`（连续小步达此次数仍未明显改善 → 升级一次大步）
   - `FINE_SMALL_PROGRESS_MIN = 3.0`（窗口内误差 px 改善小于此值视为"效果不够明显"）
   - 正常巡线/横移/转弯后确认期的 `NORMAL_ACTION_SETTLE = 1.0` 保持不变。
2. 主转 plan 与精修步的 settle 分别改为上述两个新常量（动作元组第 4 项，执行器无需改动）。
3. `PatrolSession` 新增 `turn_small_streak = 0`、`turn_streak_base_error = 0.0`。
4. `_decide_turn_fine_step` 升级逻辑：误差取 `|raw_h|`（竖线可见）或 `|offset_x|`（横条主导）。
   - 大步分支照旧并清零 streak；
   - 小步分支（转过回摆/横条反侧回摆/横条对中小步续转）：第一次小步记基准误差（streak=1）；第二次小步若 `基准误差 - 当前误差 < FINE_SMALL_PROGRESS_MIN`，本次升级为同方向大步（正向小步→`turn_action_for(corner_turn)`，反向小步→`turn_action_for(-corner_turn)`），新增打印 `V3 转弯精修：第N步 小步改善不足升级大步`，streak 清零回到小步；改善足够则保持小步并重开窗口。
   - 完成/无线停止分支不变，`CORNER_FINE_MAX_STEPS=6` 兜底。

### 修改 2：转弯前前进 1 步（仅第 2/3/4 弯）

1. 新增常量：`CORNER_PRE_TURN_FORWARD = (False, True, True, True)`（与 CORNER_TURN_SEQUENCE 平行；弯 1 已验证不偏早，保持 False）、`CORNER_PRE_TURN_FORWARD_STEPS = 1`（可调 0/2）。
2. `PatrolSession` 新增 `turn_pre_forward_remaining = 0`、`turn_main_issued = False`。
3. `decide_action` 转弯路由重构为三段：`turn_active` 时若还有前进步数 → 下发 1 步 `go_forward_fast`（settle 1.0s）并减计数；主转未下发 → 一次性下发 4 步主转（settle=TURN_MAIN_STEP_SETTLE）置 `turn_main_issued`；否则进入精修。
4. 路口触发处改为只登记状态（turn_active、turn_corner、turn_main_steps_total、turn_fine_remaining、turn_main_issued=False、按查表设置前进步数），主转 plan 改由路由下发；`turn_started=True`、视觉判向校验打印、"主X步+精修Y步"完成打印语义不变（前进步不计入转弯步数）。

### 模拟器配套
`redline_route_debug.py` 决策面板补显示 `pre_forward`、`main_issued`、`small_streak`。

### 离线验证（先跑通再部署）
- 弯 2 触发 → 动作序列 `[go_forward_fast, turn_right×4, …]`；弯 1 触发 → 直接 `[turn_left×4, …]`。
- 对中 offset≈0 持续 → 精修序列 `[小步, 小步, 大步, …]`，断言第 3 步升级为大步；竖线未转够（大步）与改善足够的小步回摆保持原行为。
- settle 值断言：主转 0.3、精修 0.5。
- 回归：既有 5 分支用例、连弯防护两场景、archive 30 帧回放。

### 部署
`py_compile` → `_deploy_patrol_v3.py` 部署 TonyPi/Functions（自动备份）→ Robot_Competition 镜像同步（备份+字节校验）→ 机器人端 py_compile。

### 风险与备注
- 缩短等待只改转弯期节奏、不碰决策链；若真机动作衔接异常，把 0.3/0.5 常量调回 0.5/0.8 即可。
- 前进 1 步只影响第 2/3/4 弯；若过冲调 0、仍偏早调 2。
- 升级机制最坏带来一步过冲后回摆，由 6 步上限与判向闭环兜底。后续若仍嫌慢，再议"主转竖线出现提前退出"和"前进 1 步后横条偏侧时先横移对中"。