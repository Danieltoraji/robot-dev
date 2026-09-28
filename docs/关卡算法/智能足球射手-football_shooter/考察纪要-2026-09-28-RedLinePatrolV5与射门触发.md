# RedLinePatrolV5 考察纪要（2026-09-28）

> 缘起：远端提交 `355e7a4`（`260928_添加的football_codes2纯粹是云盘源代码，忽略即可；
> 目前循迹用的RedLinePatrolV5避免了硬编码，几乎可以走，在第3个弯的地方容易转过丢线；
> 目前问题似乎不能触发足球射门，添加看到tag+无红线判断路径后依然无法触发，很奇怪`）
> 里同时提到"V5 在用"和"射门不触发"。本文件是把这两件事查清后的记录。
>
> **一句话结论：V5 本体没坏，问题有两个** ——
> ① 仓库里的 V5 只是**云盘快照**，不是真机上正在跑的那一份（同名文件在两个目录里、内容不同）；
> ② "射门不触发"的门**不在巡线/V5 里**，而在 `tag_walk_demo` 的两道状态门控上，
> "看到 tag / 无红线"这两个条件**根本不是射门的触发条件**。

---

## 1. V5 本体是什么（在 main 上已入库）

- 文件：`levels/football_codes/RedLinePatrolV5.py`（35,471 字节，随 `355e7a4` 入库）。
- 身份：**TonyPi 上的独立红巡线程序**。文件头自称 `RedLinePatrol.py`，运行方式是
  `sudo systemctl stop tonypi; python3 RedLinePatrol.py`，硬编码 `/home/pi/TonyPi`
  路径，用 hiwonder SDK（拿不到 SDK 时自动降级为桩对象）。
- 离线可用性（本机实测）：**能导入**，打印"未找到 hiwonder SDK，进入离线模式（动作不执行）"，
  并暴露 `init / start / run / stop / exit / detect_red_line`。
- 关键常量：`CORNER_TRIGGER_Y = 400`（注释写明"原代码硬编码 460，识别过晚"）、
  `LINE_LOST_TIMEOUT = 1.5`、`APPROACH_STEPS = 3`、`FORWARD_ACTION = "go_forward_one_step"`。
  ⇒ **"第 3 个弯转过丢线"就是它这一版要解决的问题**，不是代码缺陷。

## 2. 真问题：同名文件在两处、内容不同，仓库里那份不是"在跑的那份"

| 文件 | `levels/football_codes/` | `levels/football_codes2/` |
|---|---|---|
| `RedLinePatrolV5.py` | **有**（35 KB） | **没有** |
| `RedLinePatrolV3.py` | 78,625 B | 75,647 B ← **内容不同** |
| `demoV4.py` | 22,603 B | 19,915 B ← **内容不同** |
| `football_kick_controller.py` | 两份相同 | |
| `goal_line_judge.py` | 两份相同 | |
| `models/*`（球检测模型 5 个） | 两份相同 | |

提交信息自己说 "football_codes2 纯粹是云盘源代码，忽略即可"，但两个目录**同时在仓库里**，
而且 `demoV4.py` 的导入是"先 `import RedLinePatrolV5`、失败才 `from Functions import …`"
⇒ 真机上跑的可能是机器人本地那份，**不是仓库这份**。

**后果**：仓库里的 V5 ≠ 提交者正在用的 V5（后者改了没提交）。
任何"按仓库代码复现问题"的尝试都会对不上号。

## 3. "射门不触发"的门在哪里（已定位到行）

链路：`demoV4`（两阶段调度）→ 巡线段（V5）→ `tag_walk_demo`（Tag 路线 + 射门）→
`football_kick_controller`。**巡线/V5 不参与射门触发。**

门控在 `levels/football_codes/tag_walk_demo.py:751-790`：

```python
has_goal_stage = (navigator.state != navigator.DONE
                  and navigator.current_stage.goal_tag is not None)
if has_goal_stage or kick_controller.active:
    ball = kick_controller.detect_ball(frame)          # 需要 models/football_best_win.onnx

goal_search_states = (navigator.SEARCH_GOAL, navigator.WAIT_SHOT)
if (not ball_priority_active and not kick_controller.active
        and now >= ball_ignore_until and ball is not None
        and navigator.state in goal_search_states):    # ← 门①
    ball_seen_streak += 1
...
if (... ball_seen_streak >= ball_confirm_frames       # ← 门②（默认 2 帧）
        and navigator.state in goal_search_states):    # ← 门① 再要一次
    event = navigator.notify_goal_ready()              # → BALL_PRIORITY_STOP → 才进射门
```

**两道门必须同时成立**：

- **门① `navigator.state ∈ {SEARCH_GOAL, WAIT_SHOT}`** —— 即机器人必须已经**走到球门搜索点**
  （`tag_route_demo` 的路线：`103 → 26/50 → … → 82`，只有 `goal_checkpoint=True` 的步才切进去）。
  **"看到 tag"本身不够，"没有红线"更不在这个判据里。**
- **门② 球连续 ≥ `ball_confirm_frames`（默认 2）帧被检测到**，检测器
  `FootballDetector`（`conf 0.45`，模型 `models/football_best_win.onnx`；模型缺失会直接
  `IOError`，不会静默跳过）。另外还有 `ball_ignore_until` 冷却期会临时屏蔽球。

⇒ "加了看到 tag + 无红线判断后仍不触发"的最可能原因：**门① 不成立**（改动加在了
巡线/终点判定那一侧，而射门要的是 navigator 状态）。第二可能是门②不成立（球不在视野里、
或正处在转弯后的 `ball_ignore_until` 冷却期）。

## 4. 建议（**本文档不含任何代码改动**）

这三件事需要提交者对齐，别的改动在他活跃工作区上做会撞车：

1. **真源定哪一个**：`football_codes/` 还是 `football_codes2/`？（他自己的注释说 2 是云盘、可忽略）
2. **他本地那份 V5 的 diff 要不要提交**？否则仓库里的 V5 永远只是快照，问题无法在仓库复现。
3. **射门要触发，先确认 navigator 已进 `SEARCH_GOAL`**；把"看到 tag / 无红线"当成触发条件是
   错的方向——它只影响"终点判定/进入 Tag 阶段"，不影响射门。

## 5. 自查方法（几步就能确认到底卡在哪道门）

在机器人上跑 `demoV4` 时看日志里这几项：

- `[RedLinePatrolV3] 已连续确认 Tag103，第一阶段结束，进入 Tag/射门阶段`
  —— **没出现这一行** ⇒ 卡在第一阶段（巡线终点确认），跟射门无关；
- navigator 是否进过 `SEARCH_GOAL`（`tag_route_demo` 的状态机）；
- 球检测是否稳定给出框（模型文件在不在、`conf 0.45` 是否过高、球是否在视野内）；
- 是否落在 `ball_ignore_until` 冷却期（转弯刚结束时球会被临时忽略）。

---

### 附：本纪要涉及的证据位置

| 结论 | 证据 |
|---|---|
| V5 能离线导入、接口齐全 | `levels/football_codes/RedLinePatrolV5.py`（本机实测导入输出"离线模式"） |
| 两目录同名文件内容不同 | 逐文件哈希比对（V3、demoV4 不同；其余相同） |
| 射门门控在 tag_walk_demo | `levels/football_codes/tag_walk_demo.py:751-790` |
| 球检测器/模型/阈值 | `levels/football_codes/football_kick_controller.py:35-44`、`:111`、`:146` |
| 球门搜索点与路线 | `levels/football_codes/tag_route_demo.py:72-110`、`:309-310`、`:546-551` |
| 提交原文 | `git show -s 355e7a4` |
