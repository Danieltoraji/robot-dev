# 参考原版通关代码接入（`nine_grid_original`）

> 状态：**已接入，常量未重标**（2026-09-26）
> 入口：`python main.py nine_grid_original`
> 代码：`levels/nine_grid_original/`（**一个包，四个文件**）
> 　　　`level.py`（关卡流程：go_to / turn_to / search_target / arrived + run_level）
> 　　　`action.py`（动作组 + 云台）
> 　　　`vision.py`（拍照 + 颜色识别）
> 　　　`classifier.py`（候选区域 + HOG/SVM 数字）
> 　　　`__init__.py`（包入口：再导出 `run_level`，`main.py` 用的就是它）

---

## 一、这是什么、从哪来

`reference code/九宫格视觉导航/` 是**别人已经通关的那一版**九宫格代码（叶雨岑、陶鲁玥，
2026-08；说明书与 README 同目录）。本轮把它**逐行搬运**进本仓库，作为与现行两条路线
并列的第三条路（对照臂）：

| 路线 | 模块 | 想法 |
| :--- | :--- | :--- |
| 统一决策（默认） | `levels/nine_grid.py` | 三档分区（绿直行/蓝横移/橙旋转）+ 一个循环，每帧同一套判据 |
| 三段式（上现场优先） | `levels/nine_grid_three_stage.py` | 搜索→对准→接近→到达四段，各带各的切换判据 |
| **参考原版（本次接入）** | `levels/nine_grid_original/`（包） | 四层状态机：搜索 → 方向跟踪 → 前进修正 → 最终接近（颜色出现又消失=压过该格） |

**三条路线互不共用代码**：原版自带一整套视觉链路（自己的 HSV 窗口、自己的 HOG+SVM
分类器、自己的拍照函数），既没有 import `nine_grid_shared`，也没有 import
`vision/nine_grid_detector`；反过来现行两条路线也一行都没改。`main.py` 只多了一行 import
和一个 `LEVELS` 条目（注册的是这个**包**，包内 `__init__.py` 再导出 `run_level`）。

### 文件对应关系（搬运对照表）

| 参考版文件 | 本仓库文件（包内） | 说明 |
| :--- | :--- | :--- |
| `robot/mainv0.2.ipynb` CELL 4 / CELL 5 | `levels/nine_grid_original/level.py` | `go_to` / `turn_to` / `search_target` / `arrived` + 逐格入口 |
| `robot/action.py` | `levels/nine_grid_original/action.py` | 动作组原语 + 云台姿态 |
| `robot/capture.py` + `robot/identify.py` | `levels/nine_grid_original/vision.py` | 拍照、颜色 mask、`identify()`、yaw、proximity |
| `robot/extract_digit_roi.py` + `robot/candidate_classifier.py` | `levels/nine_grid_original/classifier.py` | 候选区域提取 + HOG/SVM 融合打分 |
| —— | `levels/nine_grid_original/__init__.py` | 包入口：`from .level import go_to, run_level` |
| `robot/models/digit_classifier_mask.pkl` | `models/nine_grid/digit_classifier_mask.pkl` | **已有同一份**（SHA256 `4F60F827…A59BC` 逐字节相同），未再复制副本 |

> 包内相互引用用**相对导入**（`from . import action, vision`、`from .vision import ...`），
> 与 `vision/__init__.py` 的既有风格一致；`main.py` 的
> `from levels import nine_grid_original as level_nine_grid_original` 一字未动。

---

## 二、原版的通关思路（读代码前先看这段）

```
搜索 SEARCH          search_target()：当前姿态「前视→低头」各识别一次，
                    找不到就机身左转大转；最多 20 次，之后抛异常
      ↓ 找到
方向跟踪 TRACKING    turn_to()：|yaw| > 30° 走大转；8°~30° 走小转，
                    小转一步的实际角度**在线估计**（指数滑动平均 α=0.3；
                    一批小转平均每步 < 0.5° ⇒ 判为地面摩擦导致小转失效，升级大转）
      ↓ 已对准
前进修正 MOVE_FORWARD go_to() 的 while：大步前进 → 重拍 → 对准；
                    框宽变化 < 5px ⇒ 判「没走近」→ 后退一步 + 越障脱困 + 重新识别
      ↓ 框宽 ≥ ARRIVE_THRESHOLD
最终接近 FINAL_APPROACH arrived()：云台切最低头，看**目标色整帧占比**：
                    占比 > COLOR_THRESHOLD          ⇒ 记住"看到过颜色"
                    看到过 且 占比 < COLOR_LOST     ⇒ 判定已压过该格，收工
                    （中间的过渡带不算消失；看到过颜色后数字丢失也算到达）
```

与原版 notebook 的组织差别只有一处：原版是人工逐格跑
`go_to(1) … go_to(7)`，本文件把它写成 `run_level(state)` 里的
`for digit in 1..7: go_to(digit)`——与原版 CELL 5 的收尾测试
`while cur < 7: go_to(cur+1, 0, 0)` 同义（`go_to` 内部按 id 覆盖三个门限，
外面传什么值都不影响）。

---

## 三、与参考版的差异（全部只为接入，**没有一处改判定**）

1. **入口形状**：`run_level(state)`（本仓库约定）替代 notebook 的手动逐格调用。
2. **门限表**：原版 `go_to()` 里 7 段 `if/elif` → 模块级 `GO_TO_PARAMS`（数值一字未改）。
3. **拍照/动作可绑定 `RobotState`**：绑定后复用真机的同一条拍照命令
   （`fswebcam -r 2592x1944 --no-banner -S 3`，与 `core.camera_config` 一致），
   并让 `main.py` 的 trace 能统计动作次数；不绑定则退回参考版自己的实现。
   带 `with_stand=` 的动作组直连 SDK（`RobotState.run_action` 没有这个参数），逐字保留。
4. **模型路径 + 懒加载**：原版 `./models/digit_classifier_mask.pkl` 是相对当前工作目录
   且在 **import 期**就 `joblib.load`；改成仓库根相对路径 + 首次使用时才加载
   （否则 `python main.py goodluck` 会因为 import 本模块而变慢甚至崩掉），
   依赖缺失时降级为**纯颜色判定**。
5. **搜索失败的收尾**：原版 `search_target()` 20 次找不到就抛 `RuntimeError` 中断
   notebook；这里捕获它、打印、**返回 False 结束本关**（顺序计分下断一格后面全不算，
   不接着做下一格，交人工判断）。
6. **进场动作**：原版 notebook 在通关前手动执行 CELL 2/3（`over_hurdle()` +
   `move_backward_one()`，即越障进入九宫格）。本仓库的进场由前序关卡负责，
   故 `ENTER_FIELD_ON_START = False`（要复现完整流程可改 True 或调 `enter_field()`）。

⚠️ **本关没有仿真器**：原版的判据建立在"颜色占比 + 框宽 + 云台姿态"上，本仓库的
`sim/nine_grid_sim.py` 没有对应模型，硬接只会给出假结论。验证只能上真机。

---

## 四、★ 待改动常量清单（本轮**未改**，上现场前逐条过）

单位与坐标约定先记住三条：
**① 所有像素量（框宽、面积）都定义在 `CAPTURE_RESIZE = (1280, 980)` 这张图上**
（原版 `capture.py` 把 2592×1944 缩放到 1280×980，纵向被压了 2%）；
**② yaw 符号：原版"左侧为正"**（与本仓库 `_body_angle_deg` 的"右正"相反）；
**③ 原版把 `CAMERA_FOV = 60°` 当真实角度用**（现行九宫格明确写着它只是伺服增益）。

### A 组：不重标就上不了场（判据直接失效）

| 常量 | 位置 | 原版值 | 为什么必须改 | 现行 nine_grid 对应值 |
| :--- | :--- | :--- | :--- | :--- |
| `LOOK_FORWARD_PULSE` | action | `1200` | 原版自己写着 `TODO：现场标定`；它决定"前视"能看到哪一带地面，直接影响远处能否识别到目标 | `PITCH_NAV = 1200`（数值巧合相同，但机型/安装不同，仍需现场确认） |
| `LOOK_DOWN_PULSE` | action | `1000` | 最终到达判据（颜色占比）**全靠这个低头档**；差 40μs ≈ 3.6°（按 0.09°/μs），整条颜色占比曲线会平移 | `PITCH_DOWN = 1040`（低头档，可见地面带 3.8~85.7cm，2026-09-24 复算） |
| `GO_TO_PARAMS`（7 行 × 3 个门限） | level | 见下表 | 在**另一台机器人 + 另一块场地**、1280×980 图上标的；本仓库的相机、灯光、贴纸、面板尺寸都不同 | 现行路线不用"框宽门 + 颜色占比门"这套判据，无对应值；颜色窗口见 `vision/nine_grid_detector.COLOR_THRESHOLDS` |
| `CAMERA_FOV` | vision | `60.0` | 原版用它把"画面横向偏移 px"换成**角度**，再据此算小转步数 ⇒ 标错就转过头/转不到位 | `CAMERA_FOV_H_DEG = 60.0`，但注释明确"这是**伺服增益**、不是真实 FOV"（实测横向 ~154.8px/°，`nine_grid.py` 里另有 0.39 的增益修正） |
| `CAPTURE_RESIZE` | vision | `(1280, 980)` | 不是"改不改"的问题而是"动一处动全身"：换分辨率 ⇒ 框宽门、邻近阈值、面积门**全部**按比例重标 | 现行九宫格用原生 `2592x1944`（`core.camera_config.CAMERA_WIDTH/HEIGHT`）＋工作宽 `WORK_WIDTH = 1296` |
| `COLOR_RANGES`（两份） | vision / classifier | 逐色 HSV 窗口，见源码 | 换场地/换灯/换打印必须重标；⚠️ 原版这里**有两份不同的表**（vision 里按 `Color` 枚举索引、classifier 里按颜色名索引），改的时候别只改一份 | `vision/nine_grid_detector.COLOR_THRESHOLDS` + `PALETTE`（见第五节末） |
| `GREEN_H_MIN/MAX_DEG = 100/160`、`GREEN_SV_RATIO_MIN = 1.15` | vision / classifier | 同左 | 绿牌判据；两份文件各存一份（本仓库照抄，保持两份独立） | `GREEN_SV_RATIO_MIN = 1.15`（一致）、`GREEN_V_MIN_PERCENT = 16.0`、绿窗口在 `COLOR_THRESHOLDS['green']` / `PALETTE['green']` |
| `BLUE_H_MIN/MAX_DEG = 207/220`、`BLUE_V_MIN/MAX_PERCENT = 35/75`、`blue_s_min()` 的 `65 − 1.43·(H−208)` / `55` | classifier | 同左 | 蓝牌用 H+S 联动判据（不是简单长方体）；换贴纸/换灯必重标 | 现行只用简单下限 `BLUE_S_MIN_PERCENT = 32.0` + `COLOR_THRESHOLDS['blue']` / `PALETTE['blue']` |
| `MIN_AREA = 15000` / `MAX_AREA = 500000` | classifier | 同左 | 候选区域面积门（px²），**严格绑定分辨率与相机高度** | `MIN_AREA_WORK = 3500` / `MAX_AREA_WORK = 1500000`（定义在 `WORK_WIDTH = 1296` 的图上） |
| `FORWARD_EFFICIENCY_MIN_PX = 5` | level | `5` | "前进一步框宽变化 < 5px ⇒ 判没走近"——框宽是分辨率相关量 | 现行用**整帧色占比**判前进无效（`NO_PROGRESS_COVER = 0.004`，理由见 `nine_grid.py` 注释：框宽在近距裁切后反向塌陷） |
| `ARRIVED_MIN_RATIO_CHANGE = 0.002`（id 6 取 0） | level | `0.002` | "小步前进后颜色占比变化"门，依赖低头档与色块尺寸 | `NO_PROGRESS_COVER = 0.004`（同为占比增量，但口径是整帧目标色份额） |

#### `GO_TO_PARAMS` 原值（id → 框宽门, 看到色门, 丢失色门）

| id | ARRIVE_THRESHOLD (px@1280×980) | COLOR_THRESHOLD | COLOR_LOST_THRESHOLD |
| --- | --- | --- | --- |
| 1 | 450 | 0.16 | 0.10 |
| 2 | 450 | 0.05 | 0.01 |
| 3 | **370** | 0.05 | 0.015 |
| 4 | 450 | **0.14** | 0.12 |
| 5 | 450 | 0.17 | 0.10 |
| 6 | 450 | **0.002** | **0.003** |
| 7 | 450 | 0.15 | 0.10 |

原版内部有两处**自相矛盾**的记录（照抄保留、未擅自统一）：id 6 的 notebook 调用写
`0.01/0.003`（还有一份被注释掉的旧表也写 `0.01`），函数内表写 `0.002/0.003`
⇒ **实际生效的是 0.002/0.003**；id 4 的调用写 `0.18/0.12`，函数内表写 `0.14/0.12`
⇒ **表生效**。

### B 组：需要核对但不一定动（对照用）

| 常量 | 原版值 | 说明 | 现行 nine_grid 对应值 |
| :--- | :--- | :--- | :--- |
| `CAPTURE_RESOLUTION` / `CAPTURE_SKIP_FRAMES` | `2592x1944` / `3` | 与 `core.camera_config` 一致 ✓ | `CAMERA_WIDTH/HEIGHT = 2592, 1944` |
| 相机锁定（白平衡/曝光/对焦） | **原版不锁** | 本仓库一局开始会锁（`lock_camera_controls`、`auto_calibrate_exposure`）。接入版**保持原样不锁**——若要启用，等于换掉颜色门的标定前提，必须与 A 组一起重标 | `CAM_WB_TEMPERATURE_K = 5800`、`CAM_EXPOSURE_ABS = 100`、`CAM_AUTO_EXPOSURE_ENABLED = True` |
| `NEAR_THRESHOLD = 250` / `MID_THRESHOLD = 120` | 同左 | **通关流程没用它们**（只用框宽原始值比 `ARRIVE_THRESHOLD`），仅 `proximity_level()` 打印用 | 现行无对应（不用远近分档） |
| `BIG_TURN_THRESHOLD = 30` | `30` | 大/小转分界（角度制，依赖 `CAMERA_FOV` 的标定） | 现行按**画面分区**判档，不用角度门（`ZONE_*`） |
| `EPSILON = 8` | `8` | "已对准"门 | `ARRIVE_CENTER_MAX = 0.12`（画幅比例，≈7°） |
| `INITIAL_SMALL_TURN_ANGLE = 4` / `ARRIVED_SMALL_TURN_INIT = 3.0` | `4` / `3.0` | 小转步长**在线估计**的初值（不是硬判据） | `SMALL_TURN_INIT_DEG = TURN_LEFT_SMALL_DEG = 8.625`（实测值） |
| `MIN_EFFECTIVE_TURN = 0.5`（id 6 取 0） | `0.5` | 小转失效门 | — |
| `TURN_ESTIMATE_ALPHA = 0.3` | `0.3` | 指数滑动平均系数 | — |
| `MAX_FINAL_TURNS = 3` | `3` | 看到颜色后一批小转的上限 | — |
| `LOST_BEFORE_SEARCH = 1` | `1` | 连续丢失几次就重新搜索 | 现行 `FIND_MAX_FRAMES = 90` / `FIND_STRIDE = 1`（重捕获） |
| `SEARCH_MAX_TURNS = 20` | `20` | 搜索最多左转 20 次 | `FIND_MAX_TURN_DEG = 720.0`（允许转 2 整圈） |
| `YAW_CENTER = 1500`、`YAW_SERVO = 2`、`PITCH_SERVO = 1`、`MAX_YAW = 90` | 同左 | 舵机 ID 与本仓库一致 ✓ | `HEAD_CENTER = 1500`、ID2=yaw、ID1=pitch |
| `angle_to_pulse()` 的 **10 μs/度** | `1500 + 10*angle` | 与 `SERVO_DEG_PER_US = 0.09`（≈11.1 μs/度）差 ~10%；`turn_to_angle()` 用到，**通关流程没调用** | `SERVO_DEG_PER_US = 0.09` |
| `HOG_PARAMS` | 9 方向 / 8×8 / 2×2 / L2-Hys | **不许动**：必须与 pkl 训练一致，改了必失配 | 同值 |
| `DIGIT_MODEL_PATH` | `./models/digit_classifier_mask.pkl` | 已改成本仓库相对路径，文件与本仓库已有的那份**逐字节相同** ⇒ 不需要重训/重传 | `models/nine_grid/digit_classifier_mask.pkl` |

### C 组：动作原语（名字同一套 SDK，**步幅要用本仓库实测值去理解**）

| 原版调用 | 实际下发 | 本仓库实测单步 | 换算 |
| :--- | :--- | :--- | :--- |
| `move_forward()` | `go_forward_one_step` × 5 | `FORWARD_ONE_STEP_CM = 2.652` | 一次 ≈ **13.3cm** |
| `move_forward_small()` | `go_forward_one_step` × 2 | 同上 | 一次 ≈ **5.3cm** |
| `move_backward_one()` | `back_one_step` × 1 | `BACK_ONE_STEP_CM = 3.2` | ≈ 3.2cm |
| `turn_big_angle_left/right()` | `turn_left` / `turn_right` × 2 | `TURN_LEFT_DEG = 22.0` / `TURN_RIGHT_DEG = 25.7` | 一次 ≈ **44° / 51°** |
| `turn_small_angle_left/right()` | `turn_left_small_step` / `turn_right_small_step` × 1 | `TURN_LEFT_SMALL_DEG = 8.625` / `TURN_RIGHT_SMALL_DEG = 5.200` | 与在线估计初值 4°/3° 相差 2~3 倍（估计很快会纠回来） |
| `over_hurdle()` | `climb_stairs` × 1 + `back_one_step` × 2 | `MOTION_BUDGET_CM["climb_stairs"] = 20.0` | ⚠️ 本仓库把 `climb_stairs` 当作**上下楼梯关卡的真实动作**；原版拿它当"越障脱困"。在九宫格场地里执行是否安全，需现场确认 |
| `move_forward_1()` / `move_forward_fast()` | `go_forward` / `go_forward_fast` × 3 | — | ⚠️ 本仓库**明令禁用**这两个动作（`DISABLED_ACTIONS`，5cm 步幅在 33cm 小面板 + 打滑地板上易摔倒）。原版**通关流程里没有调用**它们，这里照抄保留但不用 |

---

## 五、现行 `nine_grid` 的常量清单（供对照，本次一行未改）

### 5.1 关卡决策门限 —— `levels/nine_grid.py`（29 个）

| 常量 | 值 | 含义 |
| :--- | :--- | :--- |
| `TURN_LEFT_SMALL_DEG` / `TURN_RIGHT_SMALL_DEG` | 8.625 / 5.200 | 再导出（真源在 `nine_grid_shared`） |
| `ZONE_NAME_CN` | `{'forward': 绿·直行, 'side': 蓝·横移, 'turn': 橙·旋转}` | 档位中文（日志/调试镜像共用） |
| `ZONE_GREEN_TOP` / `ZONE_GREEN_BOT` | 0.0548 / 0.1199 | 绿（直行）梯形半宽：画幅顶边 / 底边 |
| `ZONE_BLUE_TOP` / `ZONE_BLUE_BOT` | 0.2746 / 0.3623 | 蓝（横移）外沿半宽：蓝区上沿 / 画幅底边 |
| `ZONE_BLUE_HEIGHT` | 0.2445 | 蓝区高度 / 画幅高 |
| `ZONE_PURPLE_HEIGHT` | 0.62 | 紫（=到达）区高度 / 画幅高 |
| `ARRIVE_PURPLE_MIN` | 0.35 | 到达：紫区份额下限 |
| `ARRIVE_ORANGE_MAX` | 0.10 | 到达：橙区份额上限 |
| `ARRIVE_ASYMMETRY_MAX` | 0.55 | 到达：左右溢出不对称上限 |
| `ARRIVE_CENTER_MAX` | 0.12 | 到达前置：横向偏移上限（画幅比例） |
| `ARRIVE_MIN_TARGET_COVER` | 0.065 | 到达前置：目标色整帧占比下限 |
| `ARRIVE_MAX_WIDTH_HEIGHT_RATIO` | 2.15 | 到达前置：目标框宽/高上限（斜切片识别） |
| `ARRIVE_ON_CELL_TOL_CM` | 16.7 | 格的同一性核验：距格心容差 |
| `ARRIVE_ON_CELL_REQUIRED` | False | 锚不可用时是否也拒绝放行 |
| `REANCHOR_MAX_POSE_JUMP_CM` | 40.0 | 到达后视觉反推位姿的弃用门 |
| `ZONE_HYSTERESIS_DEG` | 2.0 | 分区迟滞（省拍照；只放宽"离开"） |
| `UNIFIED_MAX_STEPS` | 400 | 一次尝试的步数上限（超了换招，不放弃本格） |
| `FIND_MAX_TURN_DEG` / `FIND_MAX_FRAMES` / `FIND_STRIDE` | 720.0 / 90 / 1 | 重捕获（目标丢了怎么找回来） |
| `FORWARD_COVER_MID` / `FORWARD_COVER_NEAR` | 0.085 / 0.115 | 前进快慢分档（整帧色占比） |
| `FORWARD_FAR_STEPS` / `FORWARD_MID_STEPS` | 5 / 3 | 远/中距一次前进的步数 |
| `NO_PROGRESS_COVER` / `NO_PROGRESS_PATIENCE` | 0.004 / 2 | 前进无效脱困判据 |

### 5.2 共用底座 —— `levels/nine_grid_shared.py`（54 个）

**相机与俯仰档**

| 常量 | 值 | 含义 |
| :--- | :--- | :--- |
| `CAM_HEIGHT_CM` | = `CAM_HEIGHT_STANDING_CM`(56.0) | 相机光心离地高度（布局扫会运行时自标定刷新） |
| `PITCH_NAV` | 1200 | 前视导航/布局预扫档 |
| `PITCH_DOWN` | 1040 | 低头档（可见地面带 3.8~85.7cm） |
| `PANEL_HALF_CM` / `PANEL_WIDTH_CM` | 14.0 / 28.0 | 面板色块半边长 / 边长 |
| `CAMERA_FOV_H_DEG` | 60.0 | **伺服增益**（不是真实 FOV），只用于"往哪边转/偏多少" |

**动作白名单与**现场实测**位移模型（★ 单一真源）**

| 常量 | 值 | 含义 |
| :--- | :--- | :--- |
| `DISABLED_ACTIONS` | `('go_forward', 'go_forward_fast')` | 本场地禁用的大步幅动作（`_act` 硬门拒绝） |
| `FORWARD_ONE_STEP_CM` | 2.652 | `go_forward_one_step` 实测 |
| `BACK_ONE_STEP_CM` | 3.2 | `back_one_step`（名义值，未单独实测） |
| `LEFT_MOVE_CM` / `RIGHT_MOVE_CM` | 2.497 / 2.200 | 左移实测 / 右移名义（待补测） |
| `TURN_LEFT_DEG` / `TURN_RIGHT_DEG` | 22.0 / 25.7 | 大转名义值 |
| `TURN_LEFT_SMALL_DEG` / `TURN_RIGHT_SMALL_DEG` | 8.625 / 5.200 | 小转**现场实测** |
| `ACTION_MODEL` | 由上列常量生成 | 动作 → (类型, 名义增量)；仿真器与 A/B 工具都**导入**它 |
| `SMALL_TURN_INIT_DEG` | = `TURN_LEFT_SMALL_DEG` | 自适应小转的初估 |

**定位/地图锚/布局扫描**

| 常量 | 值 | 含义 |
| :--- | :--- | :--- |
| `FIT_CAUCHY_C_PX` | 4.0 | 格阵拟合的 Cauchy 核尺度 |
| `MAP_POSE_MIN_PTS` / `MAP_POSE_MIN_INLIERS` / `MAP_POSE_INLIER_PX` | 5 / 5 / 6.0 | 地图锚：对应点下限 / 内点下限 / 内点残差门 |
| `MAP_POSE_H_RANGE_CM` | (25.0, 90.0) | 锚反解相机高度合理区间 |
| `MAP_POSE_COL_DIFF_MAX` | 0.05 | 单应 H 列模长差上限 |
| `MAP_POSE_TOL_CM` | 60.0 | 锚位姿距场心的荒谬门 |
| `MAP_POSE_MAX_COMBOS` | 80 | RANSAC 枚举子集上限 |
| `MAP_POSE_CORNER_MARGIN_PX` | 12.0 | 判"是否被画幅裁切"的边距 |
| `TURN_BUDGET_DEG` | 360.0 | 单格累计命令转角参考上限（超了换招，不放弃本格） |
| `ARENA_X_CM` / `ARENA_Y_CM` | (-33,133) / (-48,133) | 离场护栏（唯一自动停止保护） |
| `CELL_TIME_REFERENCE_S` / `TOTAL_TIME_REFERENCE_S` | 70.0 / 780.0 | 单格/整局用时参考线（**只提醒**） |
| `CELL_PROGRESS_EVERY_FRAMES` / `CAPTURE_COST_S` | 30 / 0.9 | 进度打印频率 / 单张拍照耗时估计 |
| `GRID_FIT_INLIER_CELL` / `GRID_FIT_RMS_MAX_CM` / `GRID_FIT_RMS_MAX_ALL_CM` | 0.35 / 8.0 / 0.35·格宽 | 格阵拟合：内点比例门 / 残差门 |
| `GRID_FIT_CLIPPED_WEIGHT` | 0.3 | 被裁切观测的权重 |
| `GRID_FIT_CENTER_PRIOR_X_CM` / `GRID_FIT_CENTER_MARGIN_CM` | 50.0 / 10.0 | 格阵中心先验与容差 |
| `LAYOUT_SCAN_OFFSET_MIN/MAX/STEP_DEG` | 0.0 / 40.0 / 2.5 | 安装偏移自标定网格 |
| `LAYOUT_SCAN_HEIGHT_MIN/MAX/STEP_CM` | 36.0 / 70.0 / 5.0 | 相机高度自标定网格 |
| `LAYOUT_SCAN_VOTE_MIN_FRAC` / `LAYOUT_SCAN_VOTE_MIN_COMBOS` | 0.6 / 10 | 布局投票门 |
| `PIXEL_CALIB_MED_MAX_PX` | 40.0 | 像素标定中位残差门 |
| `SPREAD_MAX_CM` / `CLUSTER_RADIUS_CM` / `AMBIG_SWAP_MARGIN_CM` | 8.0 / 6.0 / 8.0 | 观测聚合与消歧 |
| `LAYOUT_SCAN_MIN_CLEAN_DIGITS` | 6 | 扫描终止：未裁切观测覆盖的数字数 |
| `POSE_FROM_PANELS_RMS_MAX_CM` / `POSE_FROM_PANELS_X_RANGE` / `POSE_FROM_PANELS_Y_RANGE` | 0.35·格宽 / (-10,110) / (-45,105) | 由面板反推位姿的门与范围 |

### 5.3 相机/头部硬件常量 —— `core/camera_config.py`（九宫格继承使用的部分）

| 常量 | 值 | 含义 |
| :--- | :--- | :--- |
| `CAMERA_WIDTH` / `CAMERA_HEIGHT` | 2592 / 1944 | 拍照分辨率（内参与畸变都对应它） |
| `CAMERA_INTRINSIC` | fx 1944.904, fy 1950.095, cx 1283.069, cy 983.198 | 相机内参 |
| `CAMERA_DISTORTION` | k1 −0.384402, k2 0.284682 | 桶形畸变系数 |
| `HEAD_CENTER` / `HEAD_RIGHT` / `HEAD_LEFT` | 1500 / 1050 / 1950 | 头部 yaw 三档 |
| `HEAD_WIDE_RIGHT` / `HEAD_WIDE_LEFT` | 800 / 2200 | 第二档宽扫（±63°） |
| `SERVO_DEG_PER_US` | 0.09 | 脉宽→角度 |
| `CAM_PITCH_MOUNT_OFFSET_DEG` | 18.5 | 相机安装下俯偏移（布局扫会自标定刷新） |
| `CAM_HEIGHT_STANDING_CM` | 56.0 | 站立时相机离地高度 |

### 5.4 颜色窗口 —— `vision/nine_grid_detector.py`

```python
COLOR_THRESHOLDS = {
    "red":    [((0, 110, 90), (5, 255, 255)), ((172, 110, 90), (180, 255, 255))],
    "orange": [((6, 85, 110), (15, 255, 255))],
    "yellow": [((17, 70, 90), (34, 255, 255))],
    "purple": [((114, 55, 60), (136, 255, 255))],
    "pink":   [((137, 30, 80), (175, 255, 255))],
}
GREEN_SV_RATIO_MIN = 1.15     # 绿/蓝走 H+S 联合判据，不用长方体
GREEN_V_MIN_PERCENT = 16.0
BLUE_S_MIN_PERCENT = 32.0
PALETTE = {  # 手动标定的色相窗口（USE_PALETTE=True 时优先）
    "red": (176, 5, 25.0, 25.0), "orange": (6, 15, 25.0, 25.0),
    "yellow": (16, 42, 22.9, 25.0), "green": (43, 87, 25.0, 19.7),
    "blue": (88, 113, 25.0, 25.0), "purple": (114, 143, 25.0, 25.0),
    "pink": (144, 175, 10.0, 25.0),
}
```

> 对照要点：原版的颜色窗口与上面这一套**是两套独立标定**，数值不同、判据形状也不同
> （原版蓝色用 `H+S` 联动曲线，现行只用 `S` 下限 + 色相窗口；原版绿用 `H∈[100°,160°]`
> + `S/V ≥ 1.15`，现行还额外有 `GREEN_V_MIN_PERCENT`）。**不要**把两边互相抄。

---

## 六、怎么跑、怎么验证

```bash
# 真机跑参考原版（无仿真器）
python main.py nine_grid_original

# 只跑本文件的回归测试（19 项，不碰硬件/相机/模型）
python -m pytest tests/test_nine_grid_original.py -q
```

测试守的是三件事：**接入正确**（`main.py` 注册、其它四关与现行常量未动）、
**不拖累别人**（import 期不碰硬件、不加载 28MB 模型，空图短路）、
**搬运没走样**（门限表逐字比对、yaw 左侧为正、小转失效率升级大转、
"看到颜色 → 颜色消失"才判到达、颜色优先于模型那一票）。

### 上现场前的检查单

1. A 组常量按本机器人/本场地重标（尤其 `LOOK_DOWN_PULSE`、`CAMERA_FOV`、颜色窗口）；
2. 确认 `models/nine_grid/digit_classifier_mask.pkl` 在机器人上（`tools/sync_to_robot.py`
   默认同步 `models/`，`SKIP_EXTS` 不含 `.pkl`）；
3. 机器人端 `joblib / scikit-learn / scikit-image` 已装（`/home/pi/jupyter-env` 已有）；
   缺了不会崩，但会**降级为纯颜色判定**，日志里会打印一行提示；
4. `climb_stairs`（原版的越障脱困）在本场地的安全性需人工确认；
5. 与另外两条路线**分开跑、分开记分**，别混着看遥测。
