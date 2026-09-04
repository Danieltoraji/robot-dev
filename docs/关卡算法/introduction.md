# goodluck 关卡算法说明（当前稳定版）

> 📌 **现状说明**：本文描述的是**当前正在使用的固定路点稳定算法**，对应代码
> `levels/goodluck.py`（稳定边界提交 `bcf4547`，退回提交 `80aedd6`）。
> 早期版本中本文是“原始寻路算法设计文档”，现已按当前代码彻底重写；
> 若想了解曾被尝试但未采用的 A*/直线-圆弧方案，见
> [`可执行直线-圆弧路径规划方案.md`](./可执行直线-圆弧路径规划方案.md)。

---

## 1. 方案一句话

比赛规则要求固定路线，因此当前算法**不做通用路径规划**，而是：

1. 在 `levels/goodluck.py` 中维护一张固定 `ROUTE` 路点表；
2. 用 AprilTag + PnP 视觉定位获取当前位姿；
3. 对每个路点调用 `navigate_to_target()` 闭环导航；
4. 到达（或满足跳过条件）后进入下一个路点，直到完成/提前结束。

---

## 2. 为什么回到固定路点

| 时间线 | 内容 |
|--------|------|
| `bcf4547` 之前 | 固定路点法，实机可跑通，作为稳定边界 |
| `742f639` ~ 之后 | 尝试 A*、v6 混合导航、可执行直线-圆弧路径规划等新方案 |
| `80aedd6` | 比赛规则变化，决定**回退到固定路点稳定版本**，本次回退作为一次提交 |

新方案（A* / 直线-圆弧）并非简单“更差”，而是：
- 比赛已不需要通用自主规划能力；
- 固定路点逻辑简单、参数少、现场可预测；
- 新方案引入了更多调参面（A* 路点、重规划、定位滤波等），在规则收紧后性价比不高。

因此当前工作区**不保留**新方案的代码；方案文档作为历史留档保留在
`docs/关卡算法/可执行直线-圆弧路径规划方案.md`。

---

## 3. 代码结构

当前 goodluck 由三层组成：

```
main.py                # 根入口 + LEVELS 关卡注册表
core/robot_core.py     # RobotState：定位/动作/头部/几何等通用能力
levels/goodluck.py     # 赛道数据、ROUTE 路点表、导航算法、run_level
sim/goodluck_sim.py    # 模拟器：SimRobotState 继承重写 I/O
```

依赖方向：`main → level → core`，`sim/goodluck_sim → level + core`。

- 关卡内只关心 goodluck 的墙、标签、路点和决策参数；
- 通用定位、动作执行放在 `core/robot_core.py`，不掺入关卡规则。

---

## 4. 赛道与路点表

### 4.1 赛道约束

代码中的不可通行区域：

| 区域 | 范围（cm） |
|------|-----------|
| 左墙 | `0≤x≤5, 40≤y≤100` |
| 中墙 | `45≤x≤55, 0≤y≤60` |
| 右墙 | `95≤x≤100, 40≤y≤100` |
| 外框底边 | `y=0` |
| 外框顶边 | `y=100` |

出入口在 `x=0` / `x=100` 且 `0≤y≤40` 的区域开口。

### 4.2 Waypoint 模型

```python
Waypoint(pos, stop, orientation, bypass_position, bypass_condition)
```

| 字段 | 含义 |
|------|------|
| `pos` | 目标位置 `[x, y]`（cm） |
| `stop` | 到达后停留秒数；`>0` 为停靠评分点，`0` 为经过/中间路点 |
| `orientation` | 目标朝向；`None` 表示不强制朝向 |
| `bypass_position` / `bypass_condition` | 若当前位置已满足条件，则跳过该路点 |

> 当前 `ROUTE` 所有路点的 `orientation` 均为 `None`（2026-08-30 提速改造取消停靠前对准），
> 需要恢复停靠朝向时按代码注释改回 `[1,0]`、`[0,1]` 等原值即可。

### 4.3 当前 ROUTE

| # | 位置 | stop | 说明 | bypass |
|---|------|------|------|--------|
| 1 | `[14.7, 21.3]` | `STOP_TIME` | 停靠点 1 | 无 |
| 2 | `[23.1, 30.0]` | 0 | 中间路点①：先横移到 x≈23，避开左墙底角 | `y>30.5` 跳过 |
| 3 | `[23.1, 70.0]` | `STOP_TIME` | 停靠点 2 | 无 |
| 4 | `[40.0, 80.0]` | 0 | 中间路点②：先上行到 y≈80，绕过中墙上方 | `x>41` 跳过 |
| 5 | `[65.0, 79.7]` | `STOP_TIME` | 停靠点 3 | 无 |
| 6 | `[74.0, 75.0]` | 0 | 中间路点③：先东移到 x≥72，远离中墙后下行 | `x>72` 跳过 |
| 7 | `[74.0, 30.0]` | `STOP_TIME` | 停靠点 4（`--end-at-last-stop` 在此结束） | 无 |
| 8 | `[82.0, 20.0]` | 0 | 中间路点④：出口走廊引导 | `x>82` 跳过 |
| 9 | `[100.0, 20.0]` | `STOP_TIME` | 停靠点 5（出口） | 无 |

> `STOP_TIME` 当前为 `0`：到达即走，不再停留。若比赛规则要求停靠评分，
> 把 `STOP_TIME` 改回 `3` 即可，所有引用该常量的路点自动恢复停留。

---

## 5. 导航算法

### 5.1 主循环

`run_level()` 先做路点走廊净空校验（`assert_corridor_clear`），然后对 `ROUTE`
逐点调用 `navigate_to_target()`。单个路点的导航主循环：

```text
while True:
  1. locate_with_retry()          # AprilTag/PnP 定位
  2. bypass 判断                   # 若已越过该路点则直接返回 True
  3. 危险检测                      # 离墙 < OBSTACLE_THRESHOLD → 临时目标 = 最近安全点
  4. 计算 pd = target - current
  5. 动态计算有效朝向 effective_orient
  6. 朝向优先：‖od‖ > ORIENTATION_THRESHOLD → 旋转（可连转）→ continue
  7. 平移接近：dist > POSITION_THRESHOLD
       ├─ 单步贪心平移；满足条件时启用“批量直行”
       └─ continue
  8. 到达检查：危险区则恢复原目标；否则按 stop_time 停留/直接通过 → return True
```

### 5.2 动态朝向策略

| 情况 | 有效目标朝向 |
|------|-------------|
| 逃离危险区 | `None`（跳过朝向修正，优先平移离开） |
| `dist > ORIENT_FREEZE_DIST_CM` | 当前位置 → 目标的连线方向（动态更新） |
| `dist ≤ ORIENT_FREEZE_DIST_CM` 且指定了朝向 | 使用指定朝向 |
| `dist ≤ ORIENT_FREEZE_DIST_CM` 且未指定朝向 | 冻结进入近距离时的连线方向（防震荡） |

`ORIENT_FREEZE_DIST_CM = 10.0`，需大于 `POSITION_THRESHOLD = 3.0`。

### 5.3 旋转

- 只用实测可靠的大步转向：`turn_left`（22.0°/次）、`turn_right`（25.7°/次）；
- 小步转向（`turn_*_small_step`）已弃用，不再参与决策；
- 2026-08-30 起支持一次连转：需要角度 ≥ 1.5 步时 `state.act(action, times)`，
  单次最多连转 3 步，减少中间定位次数。

### 5.4 平移

`decide_panning_action()` 模拟每个候选动作执行后的位置，按“到目标距离更近”打分：

| 候选动作 | 位移 |
|----------|------|
| `go_forward` | 5.0 cm |
| `go_forward_one_step` | 2.0 cm |
| `back_one_step` | 3.2 cm |
| `left_move` | 1.9 cm |
| `right_move` | 2.2 cm |

- 执行后位置离墙 `< OBSTACLE_THRESHOLD` 的动作被排除；
- 前进动作有 `FORWARD_BIAS = 0.0` 的偏好权重（当前为 0）；
- 危险区中若无任何净空动作，选择“执行后离墙最远”的一步，保证持续有进展；
- 非危险区中全部动作被排除时按兵不动，避免乱撞。

### 5.5 批量直行

在**长直走廊段**减少重复定位：满足以下条件时一次执行多个 `go_forward`：

- 选择动作是前进类；
- 不在逃离危险区；
- 当前位置离墙 ≥ `BATCH_FORWARD_MIN_WALL_DIST`；
- 距目标 > `BATCH_FORWARD_MIN_DIST`；
- 朝向偏差足够小（≤ `GO_FORWARD_BATCH_MAX_ANGLE_DEG`）；
- 整段路径预检离墙充足。

批量步数上限 `BATCH_FORWARD_MAX_STEPS = 6`，即单次最多约 30cm。

### 5.6 危险区 / 安全点

当 `distance_to_walls(current_pos) < OBSTACLE_THRESHOLD` 时：

1. 临时目标 = `nearest_safe_point(current_pos)`；
2. 安全点阈值 = `OBSTACLE_THRESHOLD + POSITION_THRESHOLD + SAFE_MARGIN_CM`；
3. 保证安全点足够远，避免“已到安全点但还在危险区”的死循环。

### 5.7 提前结束模式

`--end-at-last-stop` 传入时，走到停靠点 4 `[74.0, 30.0]` 即结束，
跳过中间路点④和出口段。真机用法：

```bash
python main.py goodluck --end-at-last-stop
```

---

## 6. 参数表（当前值）

### 6.1 决策阈值

| 常量 | 当前值 | 含义 |
|------|--------|------|
| `ORIENTATION_THRESHOLD` | 0.26 | 朝向差异模长阈值（约 15°） |
| `POSITION_THRESHOLD` | 3.0 cm | 位置到达容差 |
| `STOP_TIME` | 0 s | 到达停靠点后的停留时间（提速改造后为 0） |
| `OBSTACLE_THRESHOLD` | 13.0 cm | 避障安全距离 |
| `SAFE_MARGIN_CM` | 3.0 cm | 安全点额外余量 |
| `CORRIDOR_CLEAR_CM` | 16.0 cm | 路点走廊净空校验阈值 |
| `ORIENT_FREEZE_DIST_CM` | 10.0 cm | 动态朝向冻结距离 |

### 6.2 批量直行参数

| 常量 | 当前值 | 含义 |
|------|--------|------|
| `BATCH_FORWARD_MAX_STEPS` | 6 | 单次批量最多步数 |
| `BATCH_FORWARD_MIN_WALL_DIST` | 18.0 cm | 批量要求的最小离墙距离 |
| `BATCH_FORWARD_MIN_DIST` | 12.0 cm | 距目标大于该值才批量 |
| `GO_FORWARD_BATCH_MAX_ANGLE_DEG` | 3.0° | 批量直行最大朝向偏差 |

### 6.3 动作参数（2026-08-25 实机标定）

| 常量 | 值 | 动作 |
|------|-----|------|
| `FORWARD_CM` | 5.0 cm | `go_forward` |
| `FORWARD_ONE_STEP_CM` | 2.0 cm | `go_forward_one_step` |
| `BACK_FAST_CM` | 3.2 cm | `back_one_step` |
| `LEFT_MOVE_CM` | 1.9 cm | `left_move` |
| `RIGHT_MOVE_CM` | 2.2 cm | `right_move` |
| `TURN_LEFT_DEG` | 22.0° | `turn_left` |
| `TURN_RIGHT_DEG` | 25.7° | `turn_right` |
| `FORWARD_BIAS` | 0.0 | 前进偏好权重 |

---

## 7. 运行方式

### 7.1 真机

```bash
# 完整 ROUTE
python main.py goodluck

# 走到停靠点 4 后提前结束（出口段交给其它方法）
python main.py goodluck --end-at-last-stop
```

### 7.2 模拟器

```bash
# 默认使用 goodluck 关卡
python -m sim.goodluck_sim

# 无界面模式（Windows PowerShell）
$env:MPLBACKEND="Agg"; python -m sim.goodluck_sim
```

---

## 8. 相关文档

- 定位原理与失败模式：`docs/关卡算法/定位逻辑详解.md`
- 赛道规则与材料：`docs/比赛相关材料/各关卡说明/goodluck关卡赛道说明.md`
- 架构审查：`docs/项目结构/架构分析报告.md`
- 历史方案（未采用）：`docs/关卡算法/可执行直线-圆弧路径规划方案.md`
