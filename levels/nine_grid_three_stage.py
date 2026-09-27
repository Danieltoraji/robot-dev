# -*- coding: utf-8 -*-
"""数字宫格关卡（三段式办法，levels/nine_grid_three_stage.py）

现场发货并验证过的实现：搜索 → 对准 → 接近 → 低头到达 → 蹭步确认，
四段各有各的切换判据（对准死区 / 框宽交棒 / 颜色占比回落）。

  搜索  导航档 + 头部五档扫 + 盲走（`_search_target`）
  对准  可见目标时按像素 yaw 转（批量小转 + 在线 EMA 估计步长，`_turn_to_face_target`）
  接近  按框宽/色占比分档前压，顺带厘米级横移纠偏（`_walk_closer`）
  到达  整帧色占比"出现→回落"判到位 + 前压兜底（`_walk_until_underfoot`）
  确认  蹭步（后退 3.2cm + 前进 2.652cm）压过格心，确认微动开关（`_press_switch`）

到达判据为什么用"峰值跟踪 + 相对跌落"而不是绝对阈值：参考实现的每色阈值
（0.002~0.17）在本机俯角下判不到（实测峰值仅 0.13）；峰值/跌落是**尺度无关**的
量，与颜色、光照、俯仰都无关。历史结论"低头色占比不可用"是旧相机模型
（39cm/41.4°，可见地面带 15.8cm 起）下的结论，现在几何已修正。

两条**死推**判据（`ARRIVE_STOP_FORWARD_CM` / `ARRIVE_FALLBACK_CM`）保留为
"视觉全盲时的最后手段"：命中事件与命中时的真值落点都进遥测
（`_note_dead_reckoning_hit`），供现场复核。

本模块只写自己这条办法；共用机制（布局扫描、投影、地图像核验、单格进度与
离场护栏、运动原语）与现场背景见 levels/nine_grid_shared.py 的文件头。
新办法（三档分区＋一个循环）见 levels/nine_grid.py。

★ 顺序计分 ⇒ 绝不跳格（2026-09-25）：一格没确认就**一直磨这一格**（本模块
一次尝试内部还有 RETRY_LIMIT 轮；收手后由 run_level 换办法重试本格）。
时间/拍照数/动作数只打印提醒、不停止，唯一会自己停下来的是离场护栏。

入口：python main.py nine_grid_three_stage
"""


import time
import numpy as np

from levels.nine_grid_shared import (
    CAMERA_FOV_H_DEG,
    FORWARD_ONE_STEP_CM,
    LEFT_MOVE_CM,
    NineGridShared,
    PANEL_WIDTH_CM,
    PITCH_DOWN,
    PITCH_NAV,
    RIGHT_MOVE_CM,
    TURN_LEFT_DEG,
    TURN_RIGHT_DEG,
    TURN_LEFT_SMALL_DEG,
    TURN_RIGHT_SMALL_DEG,
    action_name_cn,
)

from core.camera_config import (
    HEAD_CENTER,
    HEAD_LEFT,
    HEAD_RIGHT,
    HEAD_WIDE_LEFT,
    HEAD_WIDE_RIGHT,
    SERVO_DEG_PER_US,
)

from core.ground_homography import (
    GRID_CELL_CM,
    grid_cell_center,
)



# =====================================================================
# 到达与对准容差（面板 33cm，微动开关区≈中心 2/3≈±5.5cm）
# =====================================================================
BIG_TURN_SKIP_DEG = 8.0   # 残余航向差小于此就不必再补一次大转（`_big_turn`）
TURN_EMA_ALPHA = 0.3          # 指数滑动平均
TURN_EMA_ALPHA_FAST = 0.5     # 前几批用快收敛，抵消初值偏差
TURN_EMA_FAST_UPDATES = 2     # 快收敛批次数
MIN_EFFECTIVE_TURN_DEG = 0.5  # 每次实际转角低于此 = 被地面吞掉
SMALL_TURN_FAIL_BATCHES = 2   # 连续多少批失效才升级大转（单批噪声不误判）
SMALL_TURN_MAX_STEPS = 6      # 一轮小转批量上限
# ★ 关于 yaw 的正确认识（2026-09-24 定稿，此前一轮走过弯路，勿重蹈）：
#
#   `yaw` **不是**一个可信的角度数值，而是"横向像素偏移"换来的**伺服误差**，
#   单位只是"名义度"。它与真实相机方位角的比值实测 ≈ **0.734**
#   （20cm 处 0.696、55cm 处 0.752、130cm 处 0.719 —— 随距离缓变）。
#   成因：名义 FOV 60° ≠ 真实 2·atan(1296/1944.9)=67.36°，且用线性代替 atan。
#
#   为什么**不要**去"修正"它：
#   闭环在零点处两套公式同为零 ⇒ 收敛点不受影响；而这个 0.734 倍率事实上是
#   **整族 yaw 域阈值共同的隐含增益标定**（死区 ALIGN_DEAD_ZONE_DEG、
#   大转分界 BIG_TURN_DEAD_ZONE_DEG、变差判据 ALIGN_WORSEN_DEG、
#   改善判据里的 0.5°、小转规划下限 …）。
#   2026-09-24 实测（16 种子、实测原语）把控制侧换成"去畸变+atan"的精确公式：
#       到达 112/112 → **72/111**，真值落点最大 4.3cm → **128cm**，超半格 0 → 35；
#   即使把死区从 12° 补偿到 15/16/17°（对齐线性公式的等效真实角度 16.3°），
#   也只回到 35/48 ~ 43/48（基线 56/56）。⇒ 单独"修准"yaw 会让这族门限互相失配。
#   ⇒ **结论：yaw 保持线性公式，把它当无量纲伺服误差用；不要试图给它标定角度。**
#   若哪天真要用真实角度，必须**整族阈值一起重标**，属独立立项。
#   纯数学对照（供查证）：`tests/test_nine_grid_yaw.py`。
#   相关背景：评审《分区控制律与地图核验方案》Q2 —— "作为伺服增益无所谓（闭环
#   吸收），但不要用它去反推角度数值"。
# 对准死区 = **一个真实动作分辨率**（2026-09-13 真机"在死区里打转"复盘）：
# 现场实测小转一次 左 8.625°/右 5.200°（2026-09-25 现场量得，取代此前的"目视
# ≈10°"），而旧死区收到 3.5° **小于一个步长** → 每轮都规划 1 步、每步过冲
# 6~10°、永远进不了死区：
#     yaw +4° →（转 1 步，实际 -10°）→ -6° →（转 1 步）→ +4° → … 无限振荡
# 6 轮后对准放弃 → 回到搜索再转 → 现场看到的就是"在死区莫名其妙打转"，
# 最后转向叠加盲走把机器人带出场地（2026-09-13 手工重跑实录）。
# 结论：**死区必须 ≥ 一个步长**（12° ≥ 左转 8.625° + 打滑裕量），不追求度数精度。
# 也试过"用在线单步估计自适应取死区"（clip(1.2×估计, 4°, 12°)）：估计一旦崩到
# 0.8°，死区跟着收到 4° → 与 22° 大转步长严重失配 → 反而把标称场景（无变形的
# 基线）打挂（面板5 未确认）。**现场只有一个可靠数字：小转一步多大，所以死区就
# 取一个步长，不自适应**。
# ⚠️ 已知遗留（现场测量文档 §2 的判读，**尚未落地**）：死区 12° 是**右转**一步
# 5.2° 的 2.3 倍，右转时"一步跨不出死区"⇒ 该文档建议压到 4~5°。那是一次独立的
# 策略改动（会动到整族阈值），不在"小转角对齐标定"这次改动的范围内。
# 落点精度交给低头段的**厘米级横移纠偏**（ARRIVE_SIDE_VIA_SIDESTEP：
# 2.2cm/步）+ 920px 交棒点——角度的分辨率被机械步长卡死，位置的分辨率没有。
ALIGN_DEAD_ZONE_DEG = 12.0        # 对准死区（≥ 一个真实小转步长）
BIG_TURN_DEAD_ZONE_DEG = 30.0         # 大/小转角分界（参考 BIG_TURN_THRESHOLD=30）
# **真机小转一步的真实角度**（2026-09-25 起取现场实测值：左 8.625 / 右 5.200，
# 见 `levels/nine_grid_shared.py` 的单一真源；此前写的是"目视 ≈10°、裕量取 11°"）。
# 左右不对称 ⇒ 这条不变量取**较大者**（死区 ≥ 大步长 ⇒ 自然也 ≥ 小步长）。
# 只用于"死区 ≥ 一个步长"这条不变量的自检，**不参与任何控制计算**。
# 为什么单独立成一个常量：现场唯一可靠的数字就是"小转一步多大"，而它决定了
# 死区的下限。详见文件末尾 `_check_align_invariant()`——死区小于一个步长会产生
# "规划 1 步 → 过冲 → 反向再规划 1 步 → 再过冲"的极限环（现场"在死区里莫名
# 打转"+6 轮后对准放弃→回搜索再转）。这条不变量以前只写在注释里，结果死区被
# 改成 3.5°（远小于 10° 的步长）而没有任何东西拦住——现在由代码强制。
SMALL_TURN_STEP_DEG = max(TURN_LEFT_SMALL_DEG, TURN_RIGHT_SMALL_DEG)
# 小转角闭环的规划下限（°/步）：规划步数 n = floor(|yaw| / max(EMA估计, 此值))。
# 为什么需要下限：曾出现"小转被地面吞掉"→ EMA 估计崩到 0.3°/次 → n 被算成
# SMALL_TURN_MAX_STEPS，一次请求 6 步 → 过冲 → 再算 n → 振荡。
# ⚠️ 2026-09-25 起"被地面吞掉"已不在仿真模型里（现场 20 步极差只有 25°/2°，
# 实测没有这回事），但这条下限保留作保险：它不改变正常情形（估计 8.6/5.2 远大于
# 0.8），只在估计异常时才起作用。有下限后最多请求 6 小步，且**每轮都重新拍帧
# 复测 yaw**，过冲由下一轮吸收——收敛靠闭环，不靠这个常数准。
SMALL_TURN_MIN_STEP_DEG = 0.8
# 到达触发：目标框宽（px）——"目标已进入 ~35cm"的**相对深度**代理
# （2026-09-11 由 820 收紧到 920，真机"踩不到微动开关"的根因之一）。
# 换算（针孔 + 沿轴深度）：box_px ≈ fx·W / Z，Z = d·cosθ + h·sinθ
#   fx = 1944.9px（camera_config 内参）、W = 28cm（色块宽；33cm 格含缝，
#   现场说法"面板 33cm"含缝，用 33 算只差 15%）、θ = 有效俯角
#   （pitch1200 = 名义27° + 安装18.5° = 45.5°）、h = 相机高 56cm
#   （camera_config 现值 = 现场双帧拟合 + 布局扫运行时自标定；用户给的
#    39cm 是该文件的**历史粗值**，按 39 算 Z 会小 22cm、框宽虚大 ~40%）。
#   目标在 35cm 处：Z = 35·cos45.5° + 56·sin45.5° = 24.5+39.9 = 64.4cm
#   → 28cm 色块跨视轴的像宽 ≈ 1944.9×28/64.4 = 845px；实测 918px（色块自身
#   有沿轴尺寸 + 斜视投影 + 径向畸变，实测比针孔估算大 ~8%）。
# 实测（sim，目标正前方、h=56cm、安装偏移 18.5°，pitch1200 正对色块 bbox 宽）：
#   60cm→744  50cm→810  45cm→845  40cm→880  35cm→918  32cm→940
#   30cm→956（≤30cm 起 bbox 贴画幅下沿被裁切，框宽饱和失真 → 不能再细）
# 取 920px ⇔ ≈35cm（离裁切拐点 32cm 还有 2cm 裕量）：
#   - 旧值 820px ⇔ ≈48cm，交棒太早 → 低头段还得盲推 ~50cm 才压到格心，
#     这 50cm 里横向偏差没人纠（旧代码只在"从未见到颜色"时纠横），
#     压过去时偏半个开关区是常态 —— 这就是"踩不到按钮"；
#   - 35cm 交棒把盲推距离砍掉 ~13cm，且 pitch1040 在 34cm 处色块仍完整可见
#     （实测 box 902px、占比 0.0996、未裁切），低头段一开始就有视觉反馈。
# 框宽不是精细距离（近距接近饱和），精细到达仍由低头颜色"峰值→回落"判定。
ARRIVE_BOX_PX = 920.0
BATCH_MAX_STEPS = 5         # 远距批量上限（5×2.652≈13cm，仍是小步幅）
# 低头到达判据（**尺度无关**版本，2026-09-11 实测重设计）：
# 参考实现用每色绝对占比阈值（0.002~0.17），但我们的相机俯角大、色块占比峰值
# 只有 ~0.13（实测红1：0.047 → 峰值 0.130 → 0.015），照抄阈值会永远判不到
# "看到"。改为记录**本次接近的占比峰值**：峰值超过绝对下限后，占比跌到峰值
# 的 COLOR_DROP_FRAC 以下即判"已压过"——与颜色/光照/俯仰无关。
COLOR_SEEN_MIN = 0.02       # "看到过颜色"的绝对下限（防噪声）
COLOR_DROP_FRAC = 0.55      # 占比跌破峰值的此比例 → 判定压过该格
# ---- 前压到底的兜底回落判据（2026-09-24 新增，默认关闭）----
# 病因（真值位姿逐帧对齐实测，见方案文档 §1.2.1）：PITCH_DOWN 的实际俯角是 59.9°，
# 画幅下沿对应地面 **3.5cm** ⇒ 站在 28cm 面板中心时远侧那一半**仍在画面里**
# ⇒ 占比只能降到峰值的 **0.67**，**永远到不了 0.55**；于是前压封顶 45cm、
# 死推兜底又没接住 ⇒ 本格记未确认（seed 11 面板1 真值离格心仅 3.5cm、
# seed 7 面板2 仅 4.1cm，两次都是这个签名）。
# 处置：**不动 0.55 的名义值**（它管的是"色块真的消失"），只在"前压预算已经
# 用掉 ARRIVE_TAIL_PRESS_FRAC 以上、占比也真的回落了一大截"时再放行一次。
# 这样既不提前停（不牺牲落点精度），又能把"已经站到格心上却判不出来"救回来。
# 取 0.75：比可达下限 0.67 留 8 个百分点的裕量；比峰值低 25% 才算"真的在回落"。
# 取 0.80（16 种子实测最优，且比 0.75 更好）。**已设为默认**（唯一一处改默认行为的
# 改动）：见下方 A/B 实测表——它在**每一个**指标上都优于 0.55 基线，包括落点精度。
#   ┌ 算法 ──────────────┬ 到达 ──────┬ 拍照 ─┬ 落点中位 ─┬ 落点最大 ─┬ 超半格 ┐
#   │ 实测原语 16 种子 0.55│ 87/111     │ 3446  │ 2.6cm     │ 38.8cm    │ 4      │
#   │ 实测原语 16 种子 0.80│ **111/112**│ 2946  │ 2.8cm     │ **11.3cm**│ **0**  │
#   │ 名义原语 4 种子  0.55│ 22/28      │ 1132  │ 2.4cm     │ 45.0cm    │ 3      │
#   │ 名义原语 4 种子  0.80│ **28/28**  │ 767   │ 3.0cm     │ **8.8cm** │ **0**  │
#   └─────────────────────┴────────────┴───────┴───────────┴───────────┴────────┘
# 它只可能"救回"已经站到格心上却判不出来的那些格：判据要求前压已用掉 90% 预算，
# 因此**不会提前停**。唯一的代价是落在 5.5~8.8cm 区间的格从 5 个变成 7 个
# （都仍在半格内，且原本那 5 个里有 3 个是 16.7cm 以上的灾难落点）——这一条
# **必须真机复核**（仿真不建模微动开关，行程余量只能现场量）。
# 一键回退：把 `ARRIVE_TAIL_ACCEPT_FRAC` 改回 `0.0`。
ARRIVE_TAIL_ACCEPT_FRAC = 0.80   # 0 = 关闭（旧行为，结果完全一致）
ARRIVE_TAIL_PRESS_FRAC = 0.9     # 前压达到 PRESS_MAX_CM 的此比例后才允许兜底
# 实测（sim 红1/紫6 等）：占比曲线 0.06 → 峰值 0.13 → 平稳回落；取 0.55 是
# "跌掉 45%" 的保守证据，同时避免机器人刚好停在峰值附近时判不出来。
ARRIVE_MAX_STEPS = 30       # 低头段迭代上限（**一次尝试**的上限，走满就换办法重试）
# 迭代上限与"分段前压"对齐：前压封顶 45cm、精压段 2cm/帧 → 最多 23 帧，
# 再加 SIDE_MAX_CORRECTIONS(4) 次纠横帧 + 首帧 = 28 ≤ 30，够用不空转。
# ---- 分段前压（2026-09-11 真机"踩不到微动开关"根因，重设计） ----
# 旧实现的收尾是"一次死推兜底"：低头段迭代走完就直接看死推距离（≤15cm 即
# 判成功）。它有三个毛病：① 15cm 已经大于微动开关有效区（±5.5cm）；② 这段
# 盲推里**横向偏差不纠**（旧代码只在"从未见到颜色"时纠横，而交棒时颜色早就
# 见到了）；③ 段中不复查颜色，压过头/没压到都看不出来。
# 改为**分段前压 + 每段复查占比峰值**：每段只压 2~6cm，段间必拍一帧复查
# （占比跌破峰值 → 立即判到位；占比还在涨 → 继续压；横向偏差超容差 → 先
# 小幅纠横再压）。总前压距离封顶 ARRIVE_PRESS_MAX_CM。
#
# ★ 2026-09-24：45 → **60cm**（本轮最大的一处收益，实测驱动，见方案 §1.2）。
#   旧值 45cm 的推导（"交棒 ≈35cm + 面板半径 14cm ≈ 44cm"）漏了一件事：
#   **占比峰值稳定出现在离格心 14~19cm 处**（36 次真值几何实测，中位 16.1cm），
#   而不是在格心上；峰值之后还要再走完这 16cm 才能到格心。交棒实测在 45~48cm，
#   于是 45cm 预算**恰好在格心处用尽**，判据以 0.4%~3.7% 之差漏判（占比谷值/峰值
#   实测 0.551/0.571/0.608 vs 要求 <0.55）⇒ 本格记未确认 ⇒ 重搜 ⇒ 盲走 39~42cm
#   ⇒ 落点变成 45cm。**"压不到位"是假象，机器人其实每次都压到格上了**
#   （最近真值距离 0.77 / 3.93 / 0.84cm）。
#   60cm 依据：交棒 ≈48cm + 峰值后行程 ≈16cm ≈ 64cm，取 60 并保留
#   ARRIVE_PRESS_PLAUSIBLE_CM=70 这道"越界一整格"的上限不变
#   （60cm 前压自 48cm 交棒起算 = 越过格心约 12cm，仍远在 33.3cm 格内）。
#   实测（tools/ab_ninegrid.py）：
#     实测原语 16 种子 87/111 → **112/112**，拍照 3446→2828，落点最大 38.8→4.3cm；
#     名义原语  4 种子 22/28  → **28/28**， 拍照 1132→892， 落点最大 45.0→5.7cm；
#     超半格落点 4/3 → **0/0**。
ARRIVE_PRESS_MAX_CM = 60.0
ARRIVE_PRESS_COARSE_STEPS = 3   # 占比还在涨时每段步数（3×2.652≈8cm，赶路）
ARRIVE_PRESS_FINE_STEPS = 1     # 到峰/过峰后每段步数（2.652cm，精压）
# ---- 到达到达判断条件（2026-09-13 假到达根因修复，语义修正） ----
# 背景（真机"到 2 之后找 3 异常"的正解所在，实测复现）：形变阶跃 +15° 场景里
# 面板 3 **在离格心 89.5cm 处被判"到达"**。抓到的当帧证据是：
#   ratio 0.0419 →（连续 3 帧）→ 0.0088  ⇒ 跌破峰值 55% ⇒ 判"压过"
#   而那个"色块"观测是：clip、bbox 1100×380、hull 210162 —— **h/w = 0.35**，
#   它只是恰好擦过画幅下沿的**细长条**，随后被 2 次横移推出画幅。
# 也就是说：**"占比峰值→回落"描述的只是"整帧该颜色像素变少了"**，而颜色变少
# 有三种完全不同的物理原因（① 机器人真的压过；② 目标从画幅侧边滑出；
# ③ 另一个同色面板/地板色块离开视野），旧判据把 ②③ 也当成了 ①。
# 修法：**"压过"必须由一次"面板级"的视觉证据支持** —— 即本次接近里至少有一帧
# 看到过一块**形状与尺度都像面板**的同色区域（不是擦边条）。
# 门限取值（三场景全量实测，见 tools/diag_arrive.py 的"证据"列）：
#   合法到达（BASE/RW/STEP 全部确认格）: 峰值帧色块/画幅 ∈ [0.118, 0.184]
#   唯一的假到达（STEP 面板3）          : 峰值帧色块/画幅 = 0.042（h/w = 0.35）
# 取 0.08 = 落在分离带中间，两侧各留 ~2 倍余量（首版取 0.12 会把合法值 0.118
# 误杀——**这个值必须贴着实测下界取，不能"取个好看的整数"**）。它是**投影
# 不变量**（不依赖光照、不依赖相机俯仰、不依赖位姿），所以抗形变。
# 为什么用"峰值帧的面板面积"而不是"当前帧的形状"：机器人**真正压上去**时
# 色块面积会大到 12~18% 画幅，这一点与光照无关；而"形状"在临门那几帧会退化
# 成画幅下沿的粗边（实测 clip、h/w 0.29~0.46），拿形状当判据会把合法到达
# 一起误杀（首版就踩了这个坑）。所以证据取**占比峰值那一帧**的面板级面积：
# 语义上与"峰值"同步，尺度上不引入新依赖。而"当前帧是不是擦边条"只用来否决
# **横移控制**（见 _walk_until_underfoot）——擦边条的凸包质心不是面板中心的合理估计。
ARRIVE_MIN_BOX_ASPECT = 0.5   # 证据区域的 h/w 下限（擦边条 ≈0.35）
ARRIVE_MIN_BOX_COVER = 0.08   # 证据区域面积占画幅比下限（假到达 0.042）
# ★ 2026-09-23 修正：上一条只对**未裁切**观测生效（见 _walk_until_underfoot 的用法）。
#   理由（真值集量化，tests/fixtures/nine_grid_truth 的 73 条）：
#     真面板 h/w 实测跨度 0.222~2.989，其中 **17/68（25%）落在 <0.5**；
#     而这 17 条里有 **15 条是 clip==1**（被画幅裁切）。
#   裁切后的 bbox 量的是"可见残片"而不是面板，拿它算长宽比没有意义——
#   一块真面板被下沿切掉大半，可见部分自然又宽又扁。
#   原来的用法（一律拿 bbox 的 h/w 判"是不是擦边条"，判到就**跳过横移纠偏**）
#   会让这 15 条真面板在该纠横时不做纠横，一路盲压。
#   改法：只对未裁切观测做擦边条判定。裁切观测保留原有的"质心不可信"保护
#   （它本来就只对**未**裁切观测改用 center_px，见下方 px 的取法），
#   所以这里放开不会把"用擦边条质心做横移"那个历史 bug 放回来。

# 前压物理合理性上限（与视觉判据**独立**的第二道防线）：交棒 ≈35cm（见
# ARRIVE_BOX_PX）+ 一格 34cm ≈ 70cm —— 一次接近里前压超过这个距离，说明
# "落下去的"是更远处的东西，不是目标被压过。它只用**本格内累积的前压量**
# （每格到达后 _reanchor_pose 重置，格内只累积几步 2.65cm 的模型误差），
# 因此**不受长期位姿漂移影响**，可以放心用来否决视觉判据。
ARRIVE_PRESS_PLAUSIBLE_CM = 70.0
# 死推兜底（15/12cm → 6/5cm）：只在**分段前压已执行完**之后使用。判据收紧到
# 微动开关真实有效区：面板 33cm 的中心 2/3 ≈ ±5.5cm；蹭步（back3.2+forward2.65）
# 净 -0.55cm，故纵向 ≤6cm、横向 ≤5cm 才算"蹭步后仍能压到开关"。旧值 15cm 会把
# "离格心 15cm"也判成到位（真机现象：日志判成功、按钮没响、白丢 10 分）。
ARRIVE_FALLBACK_CM = 6.0    # 视觉判据走不完时，死推距离兜底（纵向 cm）
ARRIVE_FALLBACK_SIDE_CM = 5.0   # 同上（横向 cm）
# 死推到位即停压（2026-09-13 形变阶跃场景新增的**第二个独立判据**）：
# "颜色峰值回落"在相机俯仰被地板形变改掉时会失效（deform 阶跃 +15° 实测：
# 前压 44cm 占比仍不回落 → 该判据根本发不出声），而**格内相对死推距离**
# （每格到达后由 _reanchor_pose 重置，格内只累积几步 2.65cm 的模型误差）与俯仰
# 无关。任一判据成立即判到位，遥测里写清是哪一条（诚实遥测）。
# 取 2.0cm：蹭步 = back3.2 + forward2.65（净 -0.55cm），停在格心前 2cm、蹭步后
# 再退 0.55cm ⇒ 落点 ≈ 格心 -2.55cm，仍在开关有效区（±5.5cm）内。
# ⚠️ 这个 2.0cm 是**独立的停止距离**，不随单步实测值变（蹭步用的是
# FORWARD_ONE_STEP_CM，落点因此随实测值移动，上面的换算按 2.652cm 计）。
ARRIVE_STOP_FORWARD_CM = 2.0

# =====================================================================
# 三段式（现场发货的稳定实现；三档分区那套新办法见 levels/nine_grid.py）
# =====================================================================
# 本模块是『搜索 → 对准 → 接近 → 到达』四段各带各的切换判据那套老办法，
# 一路发到现场、跑通过的就是它。两种办法共用 levels/nine_grid_shared.py
# 里的机制（布局扫描、投影、位置核对、单格进度与离场护栏、运动原语）。

# 连续多少次"检不出目标面板"才认输去重搜。
TARGET_LOST_TOLERATE = 4

# 纠横：**每次最多 3 步（6.6cm），且"看到色块之后"也继续纠**（2026-09-13 实测）。
# ① 执行方式用横移而不是小转：小转一步 5.2~8.6° 在 35cm 上前压只能挪
# 35·sin5.2° ≈ 3.2cm（右转）~5.3cm（左转），而横移一步就是 2.2cm、方向直接可控；
# 纠 7cm 偏差横移 3 步 vs 小转 2 次（且每次都要重新对准），横移更细更稳；
# ② 窗口不限制在"看到色块之前"：形变随机游走场景实测，把窗口收回
#    peak < COLOR_SEEN_MIN 后，面板 6/7 直接丢（占比已在涨、横偏还没纠完就
#    开始前压 → 压出去时偏半个开关区）。这与现场"踩不到微动开关"同源。
# dx_cm 是投影不变量，相机俯仰被形变改掉时会失真——所以次数有上限
# （SIDE_MAX_CORRECTIONS），纠过头由下一帧复测接管，不会一路横着走。
SIDE_MAX_CORRECTIONS = 4     # 低头段横向纠偏次数上限（防抖动）
NO_PROGRESS_BOX_PX = 5.0          # 前进无效判据（参考 proximity_change<5）
SIDE_TOL_CM = 3.0            # 低头段横向容差（由 dx/box_w*PANEL_WIDTH_CM 换算）
# 纠横执行方式开关（可回退）：True = 横移（left/right_move，cm 级、直接消
# 偏差）；False = 旧行为（turn_*_small_step，度级——小转一步 5.2~8.6° 在 35cm
# 上前压只能挪 35·sin5.2° ≈ 3.2cm，纠 10cm 偏差要 3~4 次，且每次都要重对准）。
# 换算式 dx_cm = (px − W/2)/box_w · PANEL_WIDTH_CM 是投影不变量（抗形变），
# sim 实测精度：真值横偏 5/10cm（@30cm）→ 估 4.7/9.2cm。
ARRIVE_SIDE_VIA_SIDESTEP = True
ARRIVE_SIDE_MAX_STEPS = 3    # 单次纠横步数上限（3×2.2cm≈6.6cm，防纠过头）
FIND_MAX_ROUNDS = 4       # 搜索轮次上限（每轮 = 头部五档扫 + 转一步）
FIND_BLIND_CM = 25.0      # 盲走判据：死推纵向大于此值时先走近再由视觉接管
FIND_BLIND_MAX_STEPS = 10  # 单轮盲走上限（10×2.652≈26.5cm；2026-09-25 前按
                           # 名义 2cm 估作 20cm，实测步长更大 ⇒ 同样步数走得更远）
# 单格盲走**总量**上限（2026-09-13 真机"其它异常行动"）：搜索靠"死推提示"决定
# 往哪转/往哪走，而提示来自死推位姿——位姿在长时间盲走+转向后会发散（仿真形变
# 阶跃场景实测：一次失败的搜索盲走约 1m，随后位姿与真相差 111.9cm）。盲走越多
# 位姿越离谱、下一格的提示越不可用：宁可本格记未确认，也不要在场上瞎走。
# 60cm ≈ 2 格——目标在 2 格内都没进过视野，就不是"再走走"能解决的。
FIND_BLIND_TOTAL_CM = 60.0
FIND_MAX_FRAMES = 26      # 一次尝试的搜索拍照上限（走满就换办法重试本格）
NO_PROGRESS_TURNS = 4        # 连续 N 次转向后 |yaw| 都没变小 → 打转判据
ALIGN_MAX_FLIPS = 3         # 死区外左右来回摆 N 次 → 判定打转（直接收手）
ALIGN_WORSEN_DEG = 5.0      # 单次转向把 |yaw| 转大超过此值 = "变差"


def _check_align_invariant():
    """导入时自检：**对准死区必须 ≥ 一个真实小转步长**（设计不变量）

    为什么要在 import 时硬失败，而不是写在注释里：
    2026-09-13 真机"在死区里莫名其妙打转、最后转出场地"的根因就是这个不变量
    被破坏了——死区当时是 3.5°，而现场小转一步 ≈10°：
        yaw +4° →（规划 1 步，实际转过 10°）→ yaw −6° →（再规划 1 步）→ +4° …
    每轮都过冲、永远进不了死区；6 轮后对准放弃 → 回搜索再转 ⇒ 人看到的就是
    "打转"。这不是"参数没调好"，是**两个常数之间的关系**错了，所以只有代码
    断言能拦住下一次。

    同时校验死区不能超过"大/小转角分界"（否则大转兜底永远用不上、精度失控）。
    """
    if ALIGN_DEAD_ZONE_DEG < SMALL_TURN_STEP_DEG:
        raise AssertionError(
            f"ALIGN_DEAD_ZONE_DEG({ALIGN_DEAD_ZONE_DEG}°) < 现场小转一步"
            f"({SMALL_TURN_STEP_DEG}°，见该常量注释)：死区小于一个步长时"
            "闭环必然过冲振荡（真机现场表现='在死区里打转'）。"
            "要改小死区，必须先用现场日志证明小转一步真的变小了。")
    if ALIGN_DEAD_ZONE_DEG > BIG_TURN_DEAD_ZONE_DEG:
        raise AssertionError(
            f"ALIGN_DEAD_ZONE_DEG({ALIGN_DEAD_ZONE_DEG}°) > "
            f"BIG_TURN_DEAD_ZONE_DEG({BIG_TURN_DEAD_ZONE_DEG}°)：死区比大转分界还大，"
            "对准会直接接受任意姿态")


_check_align_invariant()
RETRY_LIMIT = 2                # 单格 CONFIRM 失败重试上限
# =====================================================================
# 三段式：对准 → 接近 → 低头到达（现场发货的稳定实现）
# =====================================================================


class NineGridThreeStageLevel(NineGridShared):
    """数字宫格（三段式）：对准 → 接近 → 低头到达，三段各有各的判据

    这是现场发货并验证过的实现，行为已冻结：任何改动都必须能复现
    `--three-stage --no-ui --seed 3` 的 7/7、拍照 191、动作 167。
    共用机制见 levels/nine_grid_shared.py。
    """
    def go_to_panel(self, digit, attempt=1):
        """前往第 digit 块面板（三段式）：返回是否确认到位

        `attempt` = 本格第几次尝试（外层在一格没确认时**继续磨这一格**，顺序
        计分下跳格等于丢掉后面的分）。

        逐格落点在返回前留档（诊断与回归用，不参与控制）。
        """
        try:
            return self._drive_to_panel(digit, attempt)
        finally:
            self._record_landing(digit)
    def _drive_to_panel(self, digit, attempt=1):
        self._begin_cell_attempt(digit, attempt)
        self._target_seen = False
        self._loc_count = 0
        self._current_digit = digit
        # 仿真接缝：把"当前目标数字"告知 sim，让地板形变可以**按格**注入
        # （见 sim/nine_grid_sim._update_deform 的 step_at_digit 说明）。
        # 真机 RobotState 没有这个方法 → getattr 兜底，对真机零影响。
        setter = getattr(self.state, "set_deform_digit", None)
        if setter is not None:
            setter(digit)
        for _try in range(RETRY_LIMIT + 1):
            # 唯一的自动停手是离场护栏（_abort_level）；时间不再是停手依据。
            if self._abort_level:
                print(f"[面板{digit}] 整局收手：{self._abort_level}")
                return False
            if self._attempt_stuck:
                break
            if self._seek_align_approach(digit):
                if self._walk_until_underfoot(digit) \
                        and self._press_switch(digit):
                    self._reanchor_pose(digit)
                    return True
            print(f"[面板{digit}] 第 {_try + 1} 轮未到位")
            if _try < RETRY_LIMIT:
                self._act("back_one_step", 1)
        return False
    def _seek_align_approach(self, digit):
        """搜索 → 对准 → 接近（框宽达标即返回 True）

        对准期目标短暂丢失（转过冲/近距窄视野）不算失败：重搜一次再试。
        任一子环节判定"这次尝试不行" → 直接返回 False（不在这里再重试，
        交回外层：同一格换办法再来）。
        """
        if self._search_target(digit) is None:
            print(f"[搜索] 未找到面板{digit}（可能被遮挡或光照不足）")
            return False
        self._target_seen = True
        for _try in range(2):
            if self._attempt_stuck:
                return False
            if self._turn_to_face_target(digit) is not None:
                return self._walk_closer(digit)
            print(f"[对准] 目标{digit}丢失 → 重新搜索一次")
            if self._search_target(digit) is None:
                break
        # 目标始终不入视野，但死推说到位了：交给低头颜色判据裁决（抗形变、
        # 不需要看见目标数字本身——色块压过与否是机器人与色块的相对关系）。
        if self._attempt_stuck:
            return False
        if digit in self.digit_cell:
            rel = self.target_relative(digit)
            if rel is not None:
                fwd, lat, _b = rel
                if abs(fwd) <= ARRIVE_FALLBACK_CM and abs(lat) <= 20.0:
                    print(f"[对准] 目标{digit}不可见，但按动作推算已到位"
                          f"（纵向 {fwd:+.1f}cm、横向 {lat:+.1f}cm）"
                          f"→ 交由低头档颜色判据裁决")
                    self._target_seen = True
                    return True
        print(f"[对准] 目标{digit}反复丢失，本轮对准收手（稍后换办法重试本格）")
        return False
    def _find_target_panel(self, digit, pitch=PITCH_NAV, head=None):
        """拍一帧找目标色块 → (obs, frame, yaw_deg, box_px)；没看到返回 None

        yaw 是**机体系**偏角：像素偏移换算（仅作伺服增益，FOV 取名义值不影响
        收敛点）**再加上当前头部角**（左正）——否则在偏头档找到目标后会转错
        角度（实测：宽扫档找到目标、按像素偏移转，目标立刻丢失）。
        box_px 优先取 bbox 宽；被画幅裁切时用 sqrt(hull_area)（等效边长）兜底。
        """
        self._count_frame()
        if head is not None:
            self.state.set_head(head)
        frame = self._capture()
        if frame is None:
            return None
        obs = self.detector.detect_panels(frame,
                                          colors=[self._target_color(digit)],
                                          arbitrate=False, drop_border=False)
        if not obs:
            return None
        # 同色择优：优先取"带数字证据"的观测，其次才按面积——现场实测木框被
        # 橙色掩膜命中且比真橙面板还大（93k vs 62k px），纯按面积择大会追着
        # 木框跑（见 nine_grid_detector.DIGIT_EVIDENCE_*）。
        cand = [o for o in obs if o.has_digit_evidence()] or list(obs)
        o = max(cand, key=lambda x: x.hull_area)
        w = float(frame.shape[1])
        center_x = float((o.hull_centroid_px if o.clipped
                          else o.center_px)[0])
        center_y = float((o.hull_centroid_px if o.clipped
                          else o.center_px)[1])
        head_deg = ((self.state.current_head_pulse - HEAD_CENTER)
                    * SERVO_DEG_PER_US)
        # yaw = 横向像素偏移换算的**伺服误差**（名义度，不是真实角度；
        # 真实相机方位角 ≈ 本值 / 0.734，详见 CAMERA_FOV_H_DEG 注释）。
        # 头部档角是真实伺服角，加回来把 yaw 变到"机体系"。
        yaw = -(center_x - w / 2.0) / w * CAMERA_FOV_H_DEG + head_deg
        box = float(o.bbox[2])
        if o.clipped:
            box = max(box, float(np.sqrt(max(o.hull_area, 1.0))))
        return o, frame, float(yaw), box
    def _find_any_panel(self, digit, pitch=PITCH_NAV):
        """先在当前头部档找；找不到再头部五档扫（零身体位移）。同 _find_target_panel"""
        seen = self._find_target_panel(digit, pitch)
        if seen is not None:
            return seen
        for head in (HEAD_LEFT, HEAD_RIGHT, HEAD_WIDE_LEFT, HEAD_WIDE_RIGHT):
            seen = self._find_target_panel(digit, pitch, head)
            if seen is not None:
                return seen
        self.state.set_head(HEAD_CENTER)
        return None
    def _align_step(self, why):
        """打转升级阶梯（打转判据的统一出口）：换策略在先，收手在后

        1) 弃用小转、改用大转（小转被地面吞掉是已知现场情形）；
        2) 放弃本次对准（返回 None，让调用方走"死推到位→低头到达"或重搜）；
        3) 仍打转 → **本次尝试**收手（外层换办法重试本格，绝不放弃这一格）。
        返回值：True = 调用方可以继续（已换策略）；False = 调用方应 return None。

        为什么不直接收手（2026-09-13 仿真复核）：标称场景里一次摆动就收手会白丢
        整格；打转真正要的结果是"别再原地转"，不是"这格不要了"。而"别再原地转"
        另有 TURN_BUDGET_DEG(360°) 兜底（转满就换办法重试），所以这条阶梯不会无限升。
        """
        self._align_stall = 0
        self._align_worsen = 0
        self._align_flips = 0
        self._align_ladder += 1
        if self._align_ladder == 1:
            self._small_turn_usable = False
            print(f"[对准] 原地打转（{why}）→ 措施一：停用小转角，改用大角度转向")
            return True
        if self._align_ladder == 2:
            print(f"[对准] 原地打转（{why}）→ 措施二：放弃本次对准"
                  "（交由按动作推算／低头档路径，不再原地转向）")
            return False
        # 措施三：本次尝试收手（**不是**放弃本格！外层会换办法重试同一格）。
        # 旧行为是熔断本格、接着做下一格——顺序计分下那是净亏。
        self._give_up_attempt(f"打转判据连续触发 3 次（{why}，本次尝试已转 "
                              f"{self._cell_turn_cmd_deg:.0f}°）——换办法重试本格")
        return False
    def _search_target(self, digit):
        """搜索目标：当前朝向头部五档 → 按死推提示转一步 → 后段前进探测

        地图只用来定"往哪转"（±30° 容差足够，死推位姿每格到达后由
        _reanchor_pose 重置，因此不会漂到不可用）。返回 (obs, frame) 或 None。
        """
        self.phase = f"SEARCH{digit}"
        frames = 0
        blind_cm = 0.0
        for rnd in range(FIND_MAX_ROUNDS):
            if frames >= FIND_MAX_FRAMES or self._attempt_stuck:
                break
            for head in (HEAD_CENTER, HEAD_LEFT, HEAD_RIGHT,
                         HEAD_WIDE_LEFT, HEAD_WIDE_RIGHT):
                if frames >= FIND_MAX_FRAMES or self._attempt_stuck:
                    break
                seen = self._find_target_panel(digit, PITCH_NAV, head)
                frames += 1
                if seen is not None:
                    self.state.set_head(HEAD_CENTER)
                    print(f"[搜索] 面板{digit} 已找到（第 {rnd + 1} 轮，"
                          f"用掉 {frames} 帧）")
                    return seen[0], seen[1]
            self.state.set_head(HEAD_CENTER)
            # 补一帧**低头档**（2026-09-13 形变场景）：导航档可见地面带在形变后
            # 会整体移走（h=56cm 时 45.5°→18.5~165cm；形变到 60° 时约 11~96cm），
            # 目标可能正好落在导航档之外、低头档之内（**3.8~85.7cm**，2026-09-24
            # 按现行常数复算；旧注释的 0.16~1.47m 是按 h=39cm 推的，已作废）。
            # 只在导航档整轮没找到时补 1 帧（真机 ~0.7s + 两次俯仰伺服）：
            # 比"整轮五档低头扫"便宜 5 倍。
            self.state.set_pitch(PITCH_DOWN)
            seen_down = self._find_target_panel(digit, PITCH_DOWN, HEAD_CENTER)
            frames += 1
            self.state.set_pitch(PITCH_NAV)
            if seen_down is not None:
                print(f"[搜索] 面板{digit} 在低头档找到（第 {rnd + 1} 轮，"
                      f"用掉 {frames} 帧）")
                return seen_down[0], seen_down[1]
            # 按死推提示转向（提示无效时固定左转）：**按提示大小转足**，
            # 一次转 1~5 个大转步（22~129°）——动作不花帧，只有头部扫花帧，
            # 这样"目标在身后 180°"也能两轮内覆盖到。
            hint_rel = (self.target_relative(digit)
                        if digit in self.digit_cell else None)
            hint = hint_rel[2] if hint_rel is not None else 0.0
            if abs(hint) < 5.0:
                hint = -TURN_LEFT_DEG      # 负 = 目标在左侧 → 左转
            k = int(np.clip(round(abs(hint) / TURN_RIGHT_DEG), 1, 5))
            self._act("turn_right" if hint > 0 else "turn_left", k)
            frames += 1
            # 盲走近目标（死推指引，动作不花帧）：地板形变会把可见地面带整体
            # 推走（±15° → 可见带在 0~50cm 与 19~184cm 间摆动），此时目标
            # 可能压根不在任何俯仰档的视野里，只能先走近再由视觉接管。
            fwd_rel = (self.target_relative(digit)
                       if digit in self.digit_cell else None)
            if fwd_rel is not None:
                fwd = fwd_rel[0]
                if fwd > FIND_BLIND_CM:
                    if blind_cm >= FIND_BLIND_TOTAL_CM:
                        print(f"[搜索] 面板{digit} 按动作推算已盲走累计 {blind_cm:.0f}cm 仍未入"
                              f"视野（上限 {FIND_BLIND_TOTAL_CM:.0f}cm）"
                              "→ 终止（不再继续盲走，位姿提示已不可信）")
                        break
                    n = int(np.clip(round((fwd - FIND_BLIND_CM) / 4.0
                                          / FORWARD_ONE_STEP_CM),
                                    1, FIND_BLIND_MAX_STEPS))
                    print(f"[搜索] 面板{digit} 未进入视野（按动作推算纵向 {fwd:.0f}cm）"
                          f"→ 盲走 {n} 步")
                    self._act("go_forward_one_step", n)
                    blind_cm += n * FORWARD_ONE_STEP_CM
        print(f"[搜索] 面板{digit} 搜索次数用尽（共 {frames} 帧）")
        return None
    def _turn_to_face_target(self, digit, max_iters=6, first_seen=None):
        """闭环对准（误差源 = 像素 yaw）→ (obs, frame, box) 或 None

        误差来自画面偏移而不是地图方位——形变不影响它。
        **小转角闭环（2026-09-11 真机"转向精度太粗/找下一格异常"重设计）**：
        只要误差在大转分界（BIG_TURN_DEAD_ZONE_DEG）以内，就永远走"小步多次逼近"，
        每轮重新拍帧复测 yaw，规划步数 = floor(|yaw| / max(EMA估计, 下限))。
        为什么不沿用"小转被吞就全局弃用小转、全走大转"：真机日志正是这条
        升级链把整局带偏——弃用小转后只剩 22°/25.7° 量化大转，在 8°（现 3.5°）
        死区外反复过冲 → _big_turn 判"振荡"后干脆**不动** → 对准循环空转
        （日志连打"大转修正 yaw -34.2°"却没有任何动作）→ 丢目标 → 重搜。
        "走到数字 2 之后找数字 3 异常"与"其它异常行动"都出自这里。
        仿真标定的"小转 1.7°/次"（2026-09-25 起：现场实测左 8.625°/右 5.200°）
        只作规划参考（且有下限保护），现场以闭环
        收敛为准：不依赖固定 °/次 常数，估计偏大偏小都由下一轮复测吸收。
        first_seen：调用方刚拍到的那一帧（搜索/接近的观测）可直接复用，
        省一次拍照（真机 ~0.7s/张）。
        """
        self.phase = f"ALIGN{digit}"
        prev_yaw, prev_n = None, 0
        last_act, last_n = None, 0
        for _ in range(max_iters):
            if self._attempt_stuck:
                return None
            if first_seen is not None:
                seen, first_seen = first_seen, None
            else:
                seen = self._find_any_panel(digit)
            if seen is None:
                if last_act is not None:
                    undo = {"turn_left": "turn_right",
                            "turn_right": "turn_left",
                            "turn_left_small_step": "turn_right_small_step",
                            "turn_right_small_step": "turn_left_small_step"}[last_act]
                    print(f"[对准] 目标{digit}丢失：回退 "
                          f"{action_name_cn(undo, max(1, last_n // 2))} 后重试")
                    self._act(undo, max(1, last_n // 2))
                    last_act, last_n = None, 0
                    continue
                return None
            obs, frame, yaw, box = seen
            # 用"转过前后 yaw 的变化"更新单次小转实际角（过冲按 |a|+|b| 折算，
            # 否则 per_step 变负 → EMA 崩向 0 → 下一轮请求更多步 → 更过冲）。
            # 注意：EMA 只用于**规划步数**，不再用它决定"是否弃用小转"。
            if prev_yaw is not None and prev_n:
                if abs(yaw) < abs(prev_yaw):
                    per = (abs(prev_yaw) - abs(yaw)) / prev_n
                else:
                    per = (abs(prev_yaw) + abs(yaw)) / prev_n
                per = float(np.clip(per, 0.3, 10.0))
                alpha = (TURN_EMA_ALPHA_FAST
                         if self._turn_updates < TURN_EMA_FAST_UPDATES
                         else TURN_EMA_ALPHA)
                self._turn_updates += 1
                self._small_turn_deg = ((1 - alpha) * self._small_turn_deg
                                        + alpha * per)
                # 仍记录"被地面吞掉"的次数（日志诊断用），但本办法**不再
                # 因为这条就弃用小转**。
                if per < MIN_EFFECTIVE_TURN_DEG:
                    self._small_turn_fail += 1
                    if self._small_turn_fail >= SMALL_TURN_FAIL_BATCHES:
                        self._small_turn_usable = False
                else:
                    self._small_turn_fail = 0
                # ---- 打转判据（2026-09-13 现场"在死区里打转"）----
                # 转过一次之后 |yaw| 反而没变小 → 记一次；连续 NO_PROGRESS_TURNS
                # 次都没变小 = 这套闭环在这一格收敛不了（机械步长吃掉误差/打滑/
                # 目标不稳）：先试**一次**大转兜底（小转被地面吞掉是已知现场情形），
                # 再不动就让本次尝试收手——绝不再"再转一次试试"。
                if abs(yaw) < abs(prev_yaw) - 0.5:
                    self._align_stall = 0
                    self._align_worsen = 0
                else:
                    self._align_stall += 1
                    # 方向自检：同号误差被**转得更大**（>ALIGN_WORSEN_DEG）
                    # = 转向方向反了（左右约定/头部符号）。连续 2 次就翻符号重试；
                    # 翻错也只是再错 2 次，被停转判据兜住。
                    if abs(yaw) > abs(prev_yaw) + ALIGN_WORSEN_DEG \
                            and (yaw > 0) == (prev_yaw > 0):
                        self._align_worsen += 1
                    else:
                        self._align_worsen = 0
                    if self._align_worsen >= 2:
                        self._align_sign = -self._align_sign
                        self._align_stall = 0
                        self._align_worsen = 0
                        print(f"[对准] 方向校验：按同方向继续转向后误差反而增大"
                              f"（{prev_yaw:+.1f}°→{yaw:+.1f}°）→ 反转转向符号"
                              f"（现 {self._align_sign:+.0f}）")
                    elif self._align_stall >= NO_PROGRESS_TURNS:
                        if self._align_step(
                                f"连续{self._align_stall}次转向 |yaw| 无改善"
                                f"（{abs(prev_yaw):.1f}°→{abs(yaw):.1f}°）"):
                            self._big_turn(-yaw * self._align_sign)
                            prev_yaw, prev_n = None, 0
                            continue
                        else:
                            return None
                # 左右来回摆（误差跨过 0、两边都还在死区外）= 打转的直接证据
                # （现场"在死区莫名其妙打转"的另一半：步长 > 误差，一步就过冲）
                if (yaw > 0) != (prev_yaw > 0) \
                        and abs(yaw) > ALIGN_DEAD_ZONE_DEG:
                    self._align_flips += 1
                    if self._align_flips > ALIGN_MAX_FLIPS:
                        if not self._align_step(
                                f"死区外左右来回摆 {self._align_flips} 次"
                                f"（{prev_yaw:+.1f}°→{yaw:+.1f}°）"):
                            return None
            # ---- 是否已对准：偏角在一个真实小转步长以内 ----
            if abs(yaw) <= ALIGN_DEAD_ZONE_DEG:
                print(f"[对准] 已对准（偏角 {yaw:+.1f}°，框宽 {box:.0f}px）")
                return obs, frame, box
            # 大转**只作兜底**：① 误差本来就大（>30°）；② 大转振荡记忆未生效时
            # 才用。振荡已发生时不再调 _big_turn（它会直接 return，等于空转），
            # 落到下面的小转闭环——保证每一轮都发出**真实动作**，不会空转。
            if abs(yaw) > BIG_TURN_DEAD_ZONE_DEG and not self._turn_oscillation:
                print(f"[对准] 以大角度转向兜底修正，偏角 {yaw:+.1f}°")
                # _big_turn 用"右正"约定，故取负；_align_sign 见方向自检
                self._big_turn(-yaw * self._align_sign)
                prev_yaw, prev_n = None, 0
                continue
            # 小转角闭环：floor 保证"规划转角 ≤ 剩余误差"（不计划性过冲）；
            # 估计角有下限保护（见 SMALL_TURN_MIN_STEP_DEG），
            # 真实收敛靠下一轮复测。
            step = max(self._small_turn_deg, SMALL_TURN_MIN_STEP_DEG)
            n = int(np.clip(int(np.floor(abs(yaw) / step)),
                            1, SMALL_TURN_MAX_STEPS))
            # yaw > 0 = 目标在画面左侧 → 左转（符号由方向自检标定）
            action = ("turn_left_small_step" if yaw * self._align_sign > 0
                      else "turn_right_small_step")
            self._act(action, n)
            self._last_turn = (action, n)
            last_act, last_n = action, n
            print(f"[对准] 小转角转向 {n} 次（偏角 {yaw:+.1f}°，估计 "
                  f"{self._small_turn_deg:.1f}°/次）")
            prev_yaw, prev_n = yaw, n
        return None
    def _walk_closer(self, digit, max_iters=60):
        """接近：目标框宽达 ARRIVE_BOX_PX 即交棒低头段

        距离只用框宽（相对量，抗形变）；卡滞判据用"框宽是否还在涨"
        （参考实现：Δproximity < 5 → 后退一步重识别）。
        """
        self.phase = f"APPROACH{digit}"
        arr = float(ARRIVE_BOX_PX)
        prev_box = None
        stall = 0
        for _ in range(max_iters):
            if self._attempt_stuck:
                return False
            seen = self._find_target_panel(digit)
            # ---- 是否已对准：偏角在一个真实小转步长以内 ----
            if seen is None:
                got = self._turn_to_face_target(digit, first_seen=seen)
                if got is None:
                    print(f"[接近] 目标{digit}丢失，交回搜索段")
                    return False
                box = got[2]
            else:
                if abs(seen[2]) <= ALIGN_DEAD_ZONE_DEG:
                    # 已对准：直接前进，**不再多拍一帧**（收尾禁转只影响"没对准"
                    # 时用什么手段修正，不改变"已对准就前进"这条语义）
                    box = seen[3]
                else:
                    got = self._turn_to_face_target(digit, first_seen=seen)
                    if got is None:
                        print(f"[接近] 目标{digit}丢失，交回搜索段")
                        return False
                    box = got[2]
            print(f"[接近] 框宽 {box:.0f}px（到位线 {arr:.0f}px）")
            if box >= arr:
                # 交棒 yaw 只作遥测：死区 12° 下它最大可到 ~12°，
                # 由此产生的横向残余由低头段的厘米级横移纠偏吸收。
                self._cell_handoff_yaw = float(seen[2]) if seen is not None \
                    else None
                if self._cell_handoff_yaw is not None:
                    print(f"[接近] 交接时残余偏角 {self._cell_handoff_yaw:+.1f}°"
                          f"（死区 {ALIGN_DEAD_ZONE_DEG:.0f}°，"
                          "横向残余交低头段纠）")
                return True
            if prev_box is not None and abs(box - prev_box) < NO_PROGRESS_BOX_PX:
                stall += 1
                if stall >= 2:
                    print("[接近] 前进无效（框宽未增大）：后退一步后重新识别")
                    self._act("back_one_step", 1)
                    stall, prev_box = 0, None
                    continue
            else:
                stall = 0
            prev_box = box
            if box < 0.6 * arr:
                n = BATCH_MAX_STEPS
            elif box < 0.85 * arr:
                n = 3
            else:
                n = 1
            self._act("go_forward_one_step", n)
        print("[接近] 循环次数已达上限")
        return False
    def _walk_until_underfoot(self, digit):
        """低头到达：颜色占比"出现 → 消失"判定压过目标格（尺度无关版）

        判据 = 记录本次接近的占比峰值，峰值过绝对下限后占比跌破峰值的
        COLOR_DROP_FRAC → 已压过该格。它描述的是"色块从机器人视野里
        由出现到消失"这一**机器人与色块的相对几何关系**，与相机高度/俯仰
        无关 → 抗地板形变。
        **横向纠偏是厘米级的**（2026-09-13 接线修正）：dx/box_w×PANEL_WIDTH_CM
        是投影不变量，换算成横向厘米后用 left/right_move 一步 2.2~2.5cm 直接消掉，
        而不是"小转 1 步"（一步 5.2~8.6° 在 35cm 上只挪 3.2~5.3cm，纠 7cm 要
        2~3 次，且每次都要重新对准）。
        纠偏**不再限制在"见到颜色之前"**：交棒点死区 12° → 横向残余可达
        35·tan12° ≈ 7cm，这段正是"踩不到微动开关"的来源；只要还在前进前压
        阶段，就允许继续纠（总次数上限 SIDE_MAX_CORRECTIONS）。
        前压总量封顶 ARRIVE_PRESS_MAX_CM（45cm）：压满仍未判到回落就交
        死推兜底复核，绝不一路压出去。
        **到达判断条件（2026-09-13 假到达修复）**：见 ARRIVE_MIN_BOX_ASPECT / ARRIVE_MIN_BOX_COVER 常量
        注释。"占比峰值→回落"只是"整帧色像素变少"，必须叠加"本次接近里见过
        一块形状/尺度都像面板的同色区域"，才允许判"压过"；另有与前压量挂钩的
        物理合理性上限 ARRIVE_PRESS_PLAUSIBLE_CM。被拒的次数与原因进
        遥测——现场靠它区分"视觉没到位"和"门太严"。
        """
        self.phase = f"ARRIVE{digit}"
        self.state.set_pitch(PITCH_DOWN)
        color = self._target_color(digit)
        peak = 0.0
        prev_ratio = None
        lat_fixes = 0
        press_cm = 0.0
        # 到达到达判断条件：峰值帧的"面板级"证据 + 拒因（诚实遥测）
        peak_cover = -1.0       # 占比峰值那一帧的面板 hull / 画幅（-1 = 未测到）
        ev_cover_best = 0.0     # 本次接近里见过的最大 hull/画幅（仅遥测）
        ev_aspect_best = 0.0    # 上述那帧的 h/w（仅遥测）
        ev_reject = 0           # 因证据不足被拒的"回落"次数（诚实遥测）
        ev_reasons = []         # 去重后的拒因
        # 迭代上限只限制**一次尝试**：走满就收手，外层换办法重试本格。
        # （旧代码在这里按『剩余时间不足』把上限砍到 60%——2026-09-25 取消：
        #   时间不再是停手依据，见 nine_grid_shared 的文件头。）
        iters_max = ARRIVE_MAX_STEPS
        try:
            for _ in range(iters_max):
                if self._attempt_stuck:
                    return False
                self._count_frame()
                self._cell_frames += 1      # 低头段不走 _capture()，这里补记
                frame = self.state.capture_frame()
                if frame is None:
                    return False
                ratio = self.detector.color_ratio(frame, color)
                # 峰值帧也升格（见 ARRIVE_MIN_BOX_COVER 注释）：峰值出现
                # 的那一帧才是"面板在画面里最大/最完整"的一帧，它的面板级面积
                # 才是该用来验证"这个峰值真的来自一块面板"的证据。
                if ratio > peak:
                    peak = ratio
                    peak_cover = -1.0
                print(f"[到达] 颜色占比 {ratio:.4f}（峰值 {peak:.4f}，"
                      f"判据 <{COLOR_DROP_FRAC * peak:.4f}）")
                # 到达判断条件素材：同一帧里找"最像面板"的同色区域（保留观测里的
                # 最大 hull）。它在后面既给"是否见过面板"作证，也决定能不能
                # 用它的质心做横移纠偏（擦边条不是合格的控制对象）。
                obs = self.detector.detect_panels(frame, colors=[color],
                                                  arbitrate=False,
                                                  drop_border=False)
                frame_area = float(frame.shape[0] * frame.shape[1])
                ev = max(obs, key=lambda x: x.hull_area) if obs else None
                # 到达判断条件素材：本次接近里见过的**最大**面板级同色区域（单调不减，
                # 只会因为有东西真的靠近而变大）；以及"当前帧的最大观测是不是
                # 画幅下沿的擦边条"（只影响横移控制与临门帧的形状审查）。
                ev_cover_now = 0.0
                ev_strip_now = False
                if ev is not None:
                    bw = max(float(ev.bbox[2]), 1.0)
                    bh = max(float(ev.bbox[3]), 1.0)
                    ev_cover_now = float(ev.hull_area) / max(frame_area, 1.0)
                    if ev_cover_now > ev_cover_best:
                        ev_cover_best = ev_cover_now
                        ev_aspect_best = bh / bw
                    ev_strip_now = (int(ev.clipped) == 0
                                    and bh / bw < ARRIVE_MIN_BOX_ASPECT)
                if peak == ratio:       # 本帧就是峰值帧：钉住它的面板级证据
                    peak_cover = max(peak_cover, ev_cover_now)
                # ★ 前压到底的兜底回落（见 ARRIVE_TAIL_ACCEPT_FRAC）：
                #   默认 0 ⇒ tail_accept 恒 False ⇒ 判据与原来一个数都不差。
                tail_accept = (
                    ARRIVE_TAIL_ACCEPT_FRAC > 0.0
                    and press_cm >= ARRIVE_PRESS_MAX_CM
                    * ARRIVE_TAIL_PRESS_FRAC
                    and ratio < ARRIVE_TAIL_ACCEPT_FRAC * peak)
                if peak >= COLOR_SEEN_MIN \
                        and (ratio < COLOR_DROP_FRAC * peak or tail_accept):
                    # ① 到达判断条件：**峰值那一帧**必须真见过"大到像面板"的同色区域，                    #    不能只靠整帧色像素变少（后者可能是侧滑出画/别的同色块
                    #    离开视野）。用峰值帧而不是"历史最大"：形变会把整条曲线
                    #    的尺度一起改掉（同一面板在不同俯仰下 box 面积差 30%），
                    #    只有跟着峰值自身的语义走才不引入新的尺度依赖。
                    if peak_cover < ARRIVE_MIN_BOX_COVER:
                        ev_reject += 1
                        why = (f"峰值帧面板证据仅占画幅 {peak_cover:.3f}"
                               f"<{ARRIVE_MIN_BOX_COVER}")
                        if why not in ev_reasons:
                            ev_reasons.append(why)
                        print(f"[到达] 占比已回落，但证据不足（{why}）"
                              "→ 不判到达（继续前压/兜底）")
                    # ② 物理合理性：前压超过"交棒 + 一格"就不可能是在压目标
                    elif press_cm > ARRIVE_PRESS_PLAUSIBLE_CM:
                        why = (f"前压{press_cm:.0f}cm>"
                               f"{ARRIVE_PRESS_PLAUSIBLE_CM:.0f}cm")
                        ev_reject += 1
                        if why not in ev_reasons:
                            ev_reasons.append(why)
                        print(f"[到达] 占比已回落，但{why}（已越过一整格）"
                              "→ 不判到达（继续前压/兜底）")
                    else:
                        self._arrive_evidence = (
                            f"低头颜色占比由峰值 {peak:.4f} 回落至 {ratio:.4f}"
                            f"（判据 <{COLOR_DROP_FRAC * peak:.4f}"
                            f"{'，前压封顶后兜底' if tail_accept else ''}）；"
                            f"已前压 {press_cm:.0f}cm，横向修正 {lat_fixes} 次｜"
                            f"证据 峰值帧色块占画幅 {peak_cover:.3f}"
                            f"（高宽比 {ev_aspect_best:.2f}，阈值"
                            f"{ARRIVE_MIN_BOX_COVER}）"
                            f"｜到达判断条件拒止 {ev_reject} 次"
                            + (f"（{'；'.join(ev_reasons)}）"
                               if ev_reasons else ""))
                        print(f"[到达] 颜色占比由峰值 {peak:.4f} 降至 {ratio:.4f}"
                              f"（证据 峰值帧色块占画幅 {peak_cover:.3f}）"
                              "→ 判定已越过目标格")
                        return True
                if ev is not None and lat_fixes < SIDE_MAX_CORRECTIONS:
                    o = ev
                    bw = max(float(o.bbox[2]), 1.0)
                    bh = max(float(o.bbox[3]), 1.0)
                    # 横移控制的对象健全性（2026-09-13 实测）：擦边条（细长）的
                    # 凸包质心不是合理的面板中心估计，拿它做横移会把目标推出
                    # 画幅——阶跃场景的假到达正是"用 1100×380 的擦边条做 2 次
                    # ×3 步横移"。遇到这种对象就**先前进**，让观测变干净再纠横。
                    if ev_strip_now:
                        print(f"[到达] 观测呈细长条（高宽比 {bh / bw:.2f}"
                              f"<{ARRIVE_MIN_BOX_ASPECT}，完整可见）"
                              "→ 跳过横移纠偏，先前压取干净观测")
                        o = None
                    if o is not None:
                        px = o.hull_centroid_px if o.clipped else o.center_px
                        box = max(float(o.bbox[2]), 1.0)
                        dx_cm = ((float(px[0]) - frame.shape[1] / 2.0) / box
                                 * PANEL_WIDTH_CM)
                        if abs(dx_cm) > SIDE_TOL_CM:
                            if ARRIVE_SIDE_VIA_SIDESTEP:
                                step_cm = (RIGHT_MOVE_CM if dx_cm > 0
                                           else LEFT_MOVE_CM)
                                n_lat = int(np.clip(
                                    round(abs(dx_cm) / step_cm),
                                    1, ARRIVE_SIDE_MAX_STEPS))
                                print(f"[到达] 横向偏差 {dx_cm:+.1f}cm → 以平移修正 "
                                      f"{'右' if dx_cm > 0 else '左'}{n_lat}步"
                                      f"（{lat_fixes + 1}/{SIDE_MAX_CORRECTIONS}）")
                                self._act("right_move" if dx_cm > 0
                                          else "left_move", n_lat)
                            else:
                                print(f"[到达] 横向偏差 {dx_cm:+.1f}cm → 以小转角转向修正"
                                      f"（{lat_fixes + 1}/{SIDE_MAX_CORRECTIONS}）")
                                self._act("turn_right_small_step" if dx_cm > 0
                                          else "turn_left_small_step", 1)
                            lat_fixes += 1
                            continue
                # 第二个独立判据：**死推到位即停压**（见 ARRIVE_STOP_FORWARD_CM）。
                # 它让"踩到格心"这件事不完全依赖颜色曲线的形状——相机俯仰被地板
                # 形变改掉时，颜色判据可能永远不回落（实测前压 44cm 仍不回落），
                # 而死推距离不受俯仰影响。任一判据成立即到位，依据写清哪一条。
                if digit in self.digit_cell:
                    fwd_now, lat_now, _bn = self.target_relative(digit)
                    if abs(fwd_now) <= ARRIVE_STOP_FORWARD_CM \
                            and abs(lat_now) <= ARRIVE_FALLBACK_SIDE_CM:
                        self._note_dead_reckoning_hit("stop", digit, fwd_now, lat_now)
                        ev_txt = (f"峰值帧色块占画幅 {peak_cover:.3f}"
                                  f"（高宽比 {ev_aspect_best:.2f}）")
                        self._arrive_evidence = (
                            f"按动作推算已到位（纵向 {fwd_now:+.1f}、"
                            f"横向 {lat_now:+.1f}cm；已前压 {press_cm:.0f}cm；"
                            f"颜色占比峰值 {peak:.4f} 未回落｜"
                            f"{ev_txt}｜到达判断条件拒止 {ev_reject} 次）")
                        print(f"[到达] 按动作推算已到位（纵向 {fwd_now:+.1f} "
                              f"横向 {lat_now:+.1f}cm）→ 判定到达")
                        return True
                # 步长自适应：占比**还在涨**时用 3 步批量（≈8cm）赶路，一旦不再涨
                # （到峰/过峰）改单步（2.652cm）精停。实测：用"与峰值比"判据会
                # 在爬升段误判成"接近峰值"而一路单步挪（占比 0.06→0.13 花了
                # 18 次迭代＝低头段占了全程 61% 的拍照）。
                rising = prev_ratio is None or ratio >= prev_ratio
                prev_ratio = ratio
                press_steps = (ARRIVE_PRESS_COARSE_STEPS if rising
                               else ARRIVE_PRESS_FINE_STEPS)
                if press_cm + press_steps * FORWARD_ONE_STEP_CM \
                        > ARRIVE_PRESS_MAX_CM:
                    print(f"[到达] 前压已达上限 {press_cm:.0f}cm"
                          f"（{ARRIVE_PRESS_MAX_CM:.0f}cm）且占比未回落"
                          " → 交由按动作推算兜底复核")
                    break
                press_cm += press_steps * FORWARD_ONE_STEP_CM
                self._act("go_forward_one_step", press_steps)
            print(f"[到达] 循环次数／前压额度用尽（峰值占比 {peak:.4f}，"
                  f"已前压 {press_cm:.0f}cm）")
            # 兜底：视觉判据没走完（例如峰值后占比掉得不够）时，用**死推距离**
            # 复核——位姿每格到达后已重置，短程死推（≤1格）精度足够。
            # ⚠️ 没有地图（布局扫失败/被关掉）时 target_relative 返回 None：
            #    这条兜底只能弃权，不能崩（2026-09-28 修）。
            rel = self.target_relative(digit)
            if rel is not None:
                fwd, lat, _b = rel
            else:
                print("[到达] 无格心映射（布局扫失败/已关）→ 死推距离兜底弃权")
                return False
            if abs(fwd) <= ARRIVE_FALLBACK_CM \
                    and abs(lat) <= ARRIVE_FALLBACK_SIDE_CM:
                self._note_dead_reckoning_hit("fallback", digit, fwd, lat)
                self._arrive_evidence = (
                    f"按动作推算距离兜底（纵向 {fwd:+.1f}、横向 {lat:+.1f}cm；"
                    f"已前压 {press_cm:.0f}cm 后占比仍未回落，"
                    f"峰值 {peak:.4f}）")
                print(f"[到达] 按动作推算距离兜底判定到位（纵向 {fwd:+.1f} "
                      f"横向 {lat:+.1f}cm）")
                return True
            return False
        finally:
            self.state.set_pitch(PITCH_NAV)
    def _note_dead_reckoning_hit(self, kind, digit, fwd, lat):
        """记录"死推判据命中"这一事件，并留下**真值落点**（若可得）

        为什么需要它（2026-09-24）：本关到达段有两条**死推**判据
        （循环内 `ARRIVE_STOP_FORWARD_CM` ±2cm/±5cm；循环后
        `ARRIVE_FALLBACK_CM` ±6cm/±5cm）。而死推位姿的实测剖面是：
        距上次重锚 ≤7 个动作时中位 ≤2.2cm，但 8~15 个动作升到 6.8cm(P90 18.7)、
        16~31 个动作 9.9cm(P90 32.2)；**到达段本身要 30~50 个动作**。
        ⇒ 在它最该被信任的地方恰恰最不可信。这两条判据到底是"救回了不少格"
        还是"制造了假到达"，只能靠"命中时的真值落点"来判——本函数就是那个证据。
        真机没有 `state.pos`，记 None（不影响真机行为）。
        """
        self._dr_hits.append(kind)
        truth = getattr(self.state, "pos", None)
        if truth is not None and digit in self.digit_cell:
            c = np.asarray(grid_cell_center(self.digit_cell[digit]), float)
            d = float(np.linalg.norm(np.asarray(truth, float)[:2] - c))
            self._dr_hit_landing.append(d)
            # 只有落点明显偏（> 开关半宽）时才出声，避免正常格刷屏
            if d > 5.5:
                print(f"[到达] 警告：按动作推算的判据（{kind}）成立，但真实位置距格心 {d:.1f}cm"
                      f"（>5.5cm 开关半宽）——纵向 {fwd:+.1f} 横向 {lat:+.1f}cm")
    def _press_switch(self, digit):
        """到达收尾：蹭步压微动开关 + 仲裁冲突近距复核

        **诚实声明（2026-09-13）**：微动开关是场地侧器件（计分由 ESP32 读），
        Pi 侧**读不到它的状态**——本函数只能保证"机器人已按判据压过格心一遍"，
        不能保证按钮真的触发。因此日志打印**依据**（到达判据 + 落点死推残差）
        而不是无条件"到达 ✓"；残差大时明确打警告，供赛后与计分表核对。
        """
        self.phase = f"CONFIRM{digit}"
        self._act("back_one_step", 1)
        self._act("go_forward_one_step", 1)
        cell = self.digit_cell.get(digit)
        if cell in self.cell_conflict:
            try:
                _frame = self._capture()
                _obs = [] if _frame is None else self.detector.detect_panels(
                    _frame, colors=[self._target_color(digit)],
                    arbitrate=True, drop_border=False)
            except Exception:
                _obs = []
            for o in _obs:
                if o.digit == digit:
                    verdict = "一致" if o.model_digit == digit \
                        else "仍冲突（以颜色为准，赛后人工核对）"
                    print(f"[复核] 格{cell} 近距数字模型判定为 {o.model_digit}"
                          f"({o.model_conf:.2f}) vs 颜色={digit}——{verdict}")
        self.state.act("stand")
        # 落点残差：位姿此刻仍是死推值（_reanchor_pose 在返回后才重置），
        # 所以这是"机器人与期望格心的差距"的独立估计，不是自证。
        resid = None
        if digit in self.digit_cell:
            fwd, lat, _b = self.target_relative(digit)
            resid = float(np.hypot(fwd, lat))
            self._cell_arrive_resid_cm = resid
        shaky = "  警告: 距格心超过半格，按钮可能未压到" \
            if (resid is not None and resid > GRID_CELL_CM / 2) else ""
        resid_txt = "无数据（无格心映射）" if resid is None else f"{resid:.1f}cm"
        print(f"[确认] 面板{digit} 已压过格心｜判定依据: {self._arrive_evidence or '无'}"
              f"｜距格心 {resid_txt}{shaky}")
        print("[确认] 微动开关状态在 Pi 侧不可读——是否触发以场地计分为准")
        return True
    def _reanchor_pose(self, digit):
        """到达后把位置压回格心，航向不动（老办法就是这么跑的）

        三段式的到达由它自己那套判据负责（`_walk_until_underfoot` +
        `_press_switch`），位姿只用来选方向，所以这里只重置位置。
        用视觉+地图反推位姿是统一决策那边的事，见 levels/nine_grid.py。
        """
        if digit not in self.digit_cell:
            return
        resid = self._cell_arrive_resid_cm
        c = grid_cell_center(self.digit_cell[digit])
        self.pose = np.array([c[0], c[1], self.pose[2]])
        print(f"[锚定] 位置重置为格心 {c[0]:.0f},{c[1]:.0f}"
              f"（航向保持 {np.degrees(self.pose[2]):.1f}°"
              f"{'' if resid is None else f'，距格心 {resid:.1f}cm'}）")
    def _big_turn(self, bearing):
        """大步转向兜底：量化到 22°/25.7°，过冲由下一轮闭环吸收

        防振荡：大转步长（22/25.7°）大于"不必再转"的门限（`BIG_TURN_SKIP_DEG`
        = 8°）的两倍时，小误差会在门限外来回转（9°→-13°→9°…）。若上一次大转后
        误差符号翻转且幅度没有明显改善（1.0° 以内），停止转向，改用"带航向偏置
        的接近"（横偏交给 sidestep）。
        """
        if abs(bearing) <= BIG_TURN_SKIP_DEG:
            return
        if self._last_big_turn is not None:
            last_sign, last_abs = self._last_big_turn
            if last_sign * bearing < 0 and last_abs <= abs(bearing) + 1.0:
                print(f"[转向] 大角度转向出现振荡（{last_abs:.0f}° → {abs(bearing):.0f}°），"
                      "停止转向，改带航向偏置接近")
                self._turn_oscillation = True
                return
        step = TURN_RIGHT_DEG if bearing > 0 else TURN_LEFT_DEG
        n = int(np.clip(round(abs(bearing) / step), 1, 3))
        action = "turn_right" if bearing > 0 else "turn_left"
        self._act(action, n)
        self._last_turn = (action, n)
        self._last_big_turn = (1 if bearing > 0 else -1, abs(bearing))
        print(f"[转向] 大角度转向 {action_name_cn(action, n)}（航向差 {bearing:+.1f}°）")


def run_level(state):
    """数字宫格关卡入口（三段式）：python main.py nine_grid_three_stage"""
    level = NineGridThreeStageLevel(state)
    return level.run_level()
