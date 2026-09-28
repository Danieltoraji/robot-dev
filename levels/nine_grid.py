# -*- coding: utf-8 -*-
"""
数字宫格关卡（levels/nine_grid.py）—— 统一决策：三档分区 ＋ 一个循环

任务：1m×1m 九宫格（3x3），7 块数字面板(1~7)随机布局，
机器人按 1→7 顺序依次到达各面板中心（踩中微动开关），限时 15 分钟。
评分：顺序正确每格 10 分（70），限时完成 +30。

这个模块 = 新办法（2026-09-25 重写，默认）。它做的唯一判断是**看目标落在
画面哪一档**：

    绿档（直行）/ 蓝档（横移）/ 橙档（旋转）——每帧算一次，下发**一个**原语。

到达判据（只看当前这一帧，**全部是画面量**，全过才算到）：
    目标居中 ≤ ARRIVE_CENTER_MAX；紫/橙/左右不对称三区份额。
    目标够大 ≥ ARRIVE_MIN_TARGET_COVER 与形状不扁（宽/高 ≤
    ARRIVE_MAX_WIDTH_HEIGHT_RATIO）**已按用户要求短路**（两个常量都是 0.0 =
    关闭，见常量块的留档说明）：在实测 33.9cm 的相机高度下这两条对真到达恒判 ✗。
    紫区 ≥ ARRIVE_PURPLE_MIN、橙区 ≤ ARRIVE_ORANGE_MAX、
    左右不对称 ≤ ARRIVE_ASYMMETRY_MAX。
（2026-09-26 短路）**位姿不参与到达判决**：曾经还有一道"格的同一性核验"
（锚解实测位姿距目标格心 ≤16.7cm 才放行），实测它会用不可信的位姿否决**正确**
到达（形变场景位姿漂 105cm 而真值离格心只有 13cm），已摘掉其否决权。锚测量保留为
`ARRIVE_ON_CELL_PROBE` 的纯诊断探针，只进日志、不改变控制流。
⚠️ 但它**不是**"面板 5 磨三次"的原因：实测那三次是搜索阶段
`TURN_BUDGET_DEG=360` 转满一圈收手（改用位姿否决的那一版同样如此，
锚在那几帧全是"unknown"）——排查时别把两件事混起来。
到位的最后一帧之后调用 `_reanchor_pose`：用视觉+地图反推实测位姿，解不出
才退回"格心 + 原航向"。

与老办法（levels/nine_grid_three_stage.py）的分工：
    · 两套办法共用 levels/nine_grid_shared.py 的感知、预算、护栏、投影、
      布局扫描；本模块只写"怎么决策"，共用背景见那边的文件头。
    · 老办法按"搜索→对准→接近→到达"四段各带各的切换判据；本办法不分段，
      判据每帧同一套。

⚠️ **已知差距（上现场前必读）**：本办法到达前**不做蹭步**
（`back_one_step` + `go_forward_one_step`），所以微动开关不一定触发；
仿真里本办法 6/7、老办法 7/7，差距同源。**上现场先用
`python main.py nine_grid_three_stage`。**

★ 顺序计分 ⇒ 绝不跳格（2026-09-25）：本关按 1→7 依次到位计分，顺序断一格
后面全不算。所以 run_level 在一格没确认时**一直磨这一格**（每次尝试换搜索方向），
时间/拍照数/动作数只打印提醒、不停止；唯一会自己停下来的是离场护栏。

入口：python main.py nine_grid
"""

import time
import numpy as np
import cv2

from levels.nine_grid_shared import (
    CAMERA_FOV_H_DEG,
    NineGridShared,
    PITCH_DOWN,
    PITCH_NAV,
    action_name_cn,
    layout_enabled,
)
from levels.nine_grid_shared import (
    TURN_LEFT_SMALL_DEG as SH_TURN_LEFT_SMALL_DEG,
    TURN_RIGHT_SMALL_DEG as SH_TURN_RIGHT_SMALL_DEG,
)

from core.camera_config import (
    HEAD_CENTER,
    HEAD_LEFT,
    HEAD_RIGHT,
    HEAD_WIDE_LEFT,
    HEAD_WIDE_RIGHT,
)

from core.ground_homography import (
    grid_cell_center,
)

from vision.nine_grid_detector import (
    build_color_mask,
    normalize_illumination,
    pick_same_color,
)


# =====================================================================
# 运动原语的**现场实测值**（runbook §2 测量 1；2026-09-25 P1 落地）
# =====================================================================
# ⚠️ 2026-09-25 体检后：真源已挪到 `levels/nine_grid_shared.py`（那里同时生成
# `ACTION_MODEL`，并被 `sim/nine_grid_sim._apply_action` 与 `tools/ab_ninegrid.REAL`
# 导入）。此处只做**向后兼容的再导出**，新代码请直接用 shared 里的定义——
# 之前三个模块各存一份（关卡 8.625、ACTION_MODEL 2.0、仿真 2.0），改一处不动另一处
# 就是"仿真跑的是不存在的机器人"这类事故的温床。
#
# 为什么必须单独列出来：`ACTION_MODEL` 里的小转此前写的是**名义 ±2.0°/步**，
# 而实测是左 **8.625°** / 右 **5.200°**（差 330%/160%）。旧代码拿 `ACTION_MODEL`
# 去规划步数 ⇒ `n = floor(|yaw| / 2.0)` 会把 16.5° 的偏差算成 6 步，真机一次转
# **51.75°**（转过头 3 倍）→ 丢目标 → 重搜。
#
# ★ 使用纪律（《需求规格-2026-09-25-统一决策重写》§7.7）：
#   这两个常量**只允许**三个用途：(a) 下发动作；(b) 死推位姿的先验；(c) 遥测/
#   测试里的期望值。**不许进任何判据**（不许再做 `n = f(角度/步长)` 的除法）。
#   统一决策一次决策只下发一个原语，因此不需要"每步多少度"这种规划常数。
TURN_LEFT_SMALL_DEG = SH_TURN_LEFT_SMALL_DEG     # 再导出（真源见 shared）
TURN_RIGHT_SMALL_DEG = SH_TURN_RIGHT_SMALL_DEG   # 再导出（真源见 shared）


# 档位 → 中文（日志用；与用户示意图的绿/蓝/橙一一对应）
# 2026-09-25 提为公开名：调试镜像（tools/debug_server.py 的「分区判据」叠加）也用它
# 标注判据点落在哪一档 —— 档位的中文说法只留这一份。
ZONE_NAME_CN = {"forward": "绿·直行", "side": "蓝·横移", "turn": "橙·旋转"}
_ZONE_NAME_CN = ZONE_NAME_CN          # 旧私有名的兼容别名（别在新代码里用）
# ---- ★ 现行规则：按用户 2026-09-24 的示意图，**从画面像素直接判档** ----
# 示意图就是相机画面本身：横轴 = 画面左右，纵轴 = 画面上下（下 = 近）。
#   绿（中央梯形，上窄下宽）→ 直行
#   蓝（画面底部两块梯形）  → 横移
#   橙（其余）              → 旋转
# 与旧规则的差别**全在边界形状**：
#   ① 旧规则的边界是**竖直线**（|yaw| ≤ 常数 ⇔ 画面横向偏移恒定 ⇔ 地上"等方位角"
#      的射线束）⇒ 近处只管住很窄一条、远处放得很宽 —— 与图正好反着。
#      图里的边界是**斜线**（上窄下宽）⇔ 地上近似**等宽度走廊**（见下方换算）。
#   ② 旧规则的"够近"用框宽/画幅高（near ≥ 0.35 ⇔ 实测地面距离 ~65cm）；
#      图里蓝区上沿是**一条水平线**，实测换算只有 **~16cm**。
#      ⇒ 用户指出的"还很远就横移"正是这条线画得太高（旧门 65cm vs 图 16cm）。
#
# 五个参数**全部从图里量出来**（tools/_tmp_diagram_measure.py：按颜色分割 →
# 逐行取边界 → 直线拟合 → 在画幅顶边/蓝区上沿/画幅底边求值，化成画幅比例，
# 故换分辨率不用改）：
#   绿梯形半宽  顶边 0.0548 → 底边 0.1199
#   蓝外沿半宽  蓝区上沿 0.2746 → 画幅底边 0.3623
#   蓝区高度    0.2445（从画幅底边往上量）
# 换算成"地上左右差几厘米"（低头档 1040：俯角 59.9°、h=56cm）：
#   绿 = 离中心线 ±6.7cm(85cm 处) ~ ±8.0cm(3.5cm 处) ⇒ 近似**恒定 ±7cm 走廊**
#   蓝外沿 = ±20.7cm(15.9cm 处) ~ ±24.3cm(3.5cm 处) ⇒ 近似**恒定 ±22cm 走廊**
#   蓝区上沿 = 地面距离 **15.9cm**
# 一句话：目标在中心线 ±7cm 内 → 直行；比 16cm 更近且偏在 ±22cm 内 → 横移；
# 其余（偏太多，或既不在走廊里又还比 16cm 远）→ 旋转。
#
# ★ **只决定"做哪种动作"，不决定"做几次"**（用户 2026-09-24 明确）：
#   直行次数按框宽分档、旋转次数按每步实测角算、横移一次一步 —— 都留在原处。
ZONE_GREEN_TOP = 0.0548       # 绿：画幅**顶边**处的半宽 / 画幅宽
ZONE_GREEN_BOT = 0.1199       # 绿：画幅**底边**处的半宽 / 画幅宽
ZONE_BLUE_TOP = 0.2746        # 蓝外沿：**蓝区上沿**处的半宽 / 画幅宽
ZONE_BLUE_BOT = 0.3623        # 蓝外沿：画幅**底边**处的半宽 / 画幅宽
ZONE_BLUE_HEIGHT = 0.2445     # 蓝区高度 / 画幅高（从底边往上量）
# ---- 紫区：**只看当前状态**的到达判据（用户 2026-09-24 方案）----
# 示意图底部中央那块紫 = "面板已经在我脚下、而且基本对正"的区域：
#   横向边界 = 绿走廊，**故意绑死**（用户定）：一旦比走廊宽/窄，就会出现
#              "没到位却只能直行"或"该横移却判到达"的死区 —— 对不上得不偿失。
#   高度     = **自己的参数**（用户明确要求）：它决定"多近才算站上去"，
#              不能被蓝区高度（横移范围）带着走。实测教训：共用一个参数时把它
#              调成 0.60（≈脚下 40cm）后，判据在每格离格心 33cm 处就成立了
#              （28 格全部提前成立）；改回 0.2445（≈脚下 16cm）时是 3.0~6.8cm。
# ★★ 2026-09-25 实测校准（P1 重写）：紫区**必须覆盖"格心附近"那块地面**，
#    否则到达判据在单格预算内**不可达**。
#
# 病：紫区高度沿用示意图量出的 0.2445 ⇒ 紫带上沿只到画幅底边上方约 4cm 地面。
#     低头档 1040 + 相机高 57cm 的实测（目标正前方，逐档量紫/橙份额）：
#        距离 15cm → 紫 0.323（差 0.027）｜12cm → 0.332｜10cm → 0.339｜8cm → 0.365★
#        6cm  → 0.410 但橙 0.103（>0.10 门，**差 0.003 被卡**）｜4cm → 0.468 且橙 0.063★
#     ⇒ 判据只在 **≤4cm** 成立。而"框宽→距离"几乎不可用（实测 70cm=688px、
#       30cm=924px、≤15cm 起裁切饱和在 ~1000px），单步 2.652cm ⇒ 从 40cm 走到
#       4cm 要 ~14 次前进决策，7 格就是 100+ 帧，直接撞 140 张/格上限。
#     旧实现之所以"能到"，是因为它用的是**旧到达判据**（色占比回落，峰值出现在
#       离格心 14~19cm），而不是这条紫区判据——紫区判据在旧代码里从未当过主判据。
#
# 校准：紫带上沿抬到画幅底边上方 **0.62** ⇒ 覆盖到约 16cm 地面（与旧判据的
#     峰值距离同量级，落在±半格 16.7cm 之内），判据在 ~16cm 处成立，
#     单格只需 ~8 次前进决策。
# ⚠️ 真机复核项：该值由**仿真几何**标定（h=57cm、俯角 59.9°）。现场若相机高度/
#     俯角不同，必须用测量 2（近距剖面）重标——见需求规格 §10 风险 1。
#
# ★ 2026-09-26 调参：**0.62 → 0.31（减半）**，只动高度、走廊横向边界不动。
#   高度决定"多近才算站上去"：0.31 覆盖画幅底边上方约 **8cm** 地面
#   （0.62 是约 16cm），比 ±半格 16.7cm 紧，到达点更靠格心。
#   ⚠️ 代价（上一条的实测数据推出来）：紫份额按距离单调，40cm→0.345、30cm→0.505；
#   上半段压掉一半 ⇒ 紫门 0.35 大约要走到**离格心 ~8cm** 才成立，单格前进决策数
#   随之上升（历史上 4cm 那版要 ~14 次/格）。重标/回退时看这一行。
ZONE_PURPLE_HEIGHT = 0.31
# ---- 到达门的三个份额阈值：全部由实测定（老办法 28 次真到达 vs 5 次假到达；工具
# tools/_tmp_region_ratio.py）。单位 = **目标色像素的份额**：
#   紫区份额：真到达 min 0.446 / 中位 0.621 ｜ 假到达 全部 **0.000** ⇒ 取 0.35
#   橙区份额：真到达 max 0.018 ｜ 假到达 中位 0.707（3 格 0.61~1.00）⇒ 取 0.10
#   左右溢出不对称 |蓝左−蓝右|：真到达 中位 0.099 / P90 0.199 / max 0.251，
#       换算 ≈ 每 1cm 横向偏差对应 0.078 不对称 ⇒ 最初取 0.25（≈横向 3.2cm），
#       **2026-09-25 已改为 0.55，理由见下面那段**。
#       这一条对应"要站在格子中央才能触发压感"（用户要求），
#       并且是**状态量**：偏心时色块溢到左右蓝区的量不对称，不用任何历史。
ARRIVE_PURPLE_MIN = 0.35      # 紫区份额下限
# ★ 2026-09-28 按**现场到达参考帧**重标（用户提供的四张图，存
#   docs/关卡算法/彩色数字九宫格-nine_grid/refs/）：
#     状态        紫      橙      不对称   居中
#     A 临界(直视) 0.472   0.164   -0.071   0.015   ← 用户：这是到达临界
#     B 中央(直视) 0.504   0.024   -0.111   0.039   ← 用户：这是正好到场中央
#     C 临界(45°)  0.353   0.212   -0.060   0.010
#     D 中央(45°)  0.630   0.252   -0.111   0.039
#   ⇒ **紫门 0.35 是对的**（C 恰好压线 0.353，说明它确实是"临界"）；
#     **橙门 0.10 是错的**：四个"期望到达"帧的橙在 0.164~0.252，全部被判 ✗，
#     机器人因此永远不判到达（现场 5 帧实测紫 0.51~0.53/橙 0.05~0.06 与 0.28~0.32
#     之间反复，也是同一件事）。
#   橙像素到底出在哪（同一次实测，逐区像素分布）：
#     · 大部分是**场上红色杂物**（红边框/红条）落进"其余"区，与有没有开到板上无关；
#     · 少量是面板上沿漏到紫带以外。
#   ⇒ 新阈值取 **0.30**：覆盖四个期望到达帧（含余量到 0.252），又远低于
#     旧标定的假到达 0.707。判"还太远"的职责交给**紫门**（面板远时投影高、
#     紫区拿不到像素：实测 50cm 外紫 0.000、橙 0.73）。
ARRIVE_ORANGE_MAX = 0.30
ARRIVE_ASYMMETRY_MAX = 0.55        # 左右溢出不对称上限
# ★ 2026-09-25 由 0.25 改为 **0.55**（依据是**扫门实测 + 落地精度复核**）：
#   ① 这个量是**角度量**，不是距离量 —— 同一横偏在不同距离读数差 2~3 倍（实测）：
#         横偏   @4cm   @10cm  @20cm
#          0cm   0.012  0.021  0.012
#          2cm   0.159  0.123  0.069
#          4cm   0.293  0.226  0.125
#          6cm   0.403  0.303  0.169
#      ⇒ 固定阈值天然随距离漂移，"多少算偏心"没有唯一答案。
#   ② 它**永远到不了 0.5 以上多少**：面板 28cm 比绿走廊（±7cm）宽，超出走廊的部分
#      全算橙 ⇒ 判据上限被几何压住，0.25 实际只容许 ±1.7cm 横偏，而蓝档把机器人
#      稳定在 ±5cm 量级 ⇒ 到达判据几乎永不成立。
#   ③ 扫门实测（3 种子 21 格）：0.25→10/21、**0.55→20/21**、0.70 与 0.55 同。
#   ④ 落地精度复核（0.55 + 每帧都转，3 种子）：落点
#      [7.0, 8.4, 6.1, 7.4, 6.9, 10.1, 8.8]cm —— **全部 ≤10.1cm**，都在半格
#      16.7cm 内，且多数在微动开关半宽 5.5cm 附近。
#      ⇒ 放宽不对称门**没有**换来"压偏也判到达"。
#   ⚠️ 更正确的做法是用"几何横偏 cm"（由（机体方位角, 像宽/画幅）反解，实测误差
#      ~1cm，只差 `_center_column_px` 的 13px≈0.7° 常数偏置）——留作后续升级，届时这个
#      经验门可以整条去掉。
# 到达门前置条件：目标观测相对**机体系中线**的横向偏移上限（画幅比例）
#   0.12 ≈ 311px @2592（机体系方位角 ~7°）—— 与绿走廊的"近处半宽 0.1199"
#   同量级：站在走廊里才允许宣布到达。
#   ⚠️ 设 0 表示关闭该前置条件（仅用于对照实验，不许当默认）。
ARRIVE_CENTER_MAX = 0.12
# ★ 到达门前置条件二：目标色**整帧占比下限**（目标必须"够大"= 真的压在脚下）
#   ⚠️ **2026-09-28 已按用户要求短路（0.0 = 关闭）**。原值与理由（留档，别再自己加回来）：
#     为什么当初加了它（P1 实测的假到达机制）：目标**边缘斜视的残缺投影**会让四个
#     整帧份额全部失真 —— 远侧一半出画幅 ⇒ 橙区拿 0（本该 0.48）、紫区拿 0.637
#     ⇒ 门在离目标 66cm 处成立。实测：真到达整帧占比 0.078~0.150、骗门帧 0.042
#     ⇒ 取 0.065。
#     为什么现在关：用户 2026-09-28 现场判定"面板被画幅切掉下沿的那几帧（紫 0.51~0.53、
#     橙 0.05~0.06、居中 ≤0.02）**就是到达**"，并明确"没要求过这两条限制"。
#     而且在 h=33.9cm + 低头 1040 的几何下它**必然**卡住："画幅底边只看到 23.5cm 处
#     的地面" ⇒ 面板中心一进 23.5cm 下沿就出画幅 ⇒ whole 必然掉到 0.058 左右，
#     而"紫/橙要过"又要求走到 ~15cm ⇒ 两个区间不相交，门永不成立（现场 5 帧实测）。
#   设 0 = 关闭该判据；要恢复请连同几何重标一起做（见核对报告 §8.4）。
ARRIVE_MIN_TARGET_COVER = 0.0
#   ⚠️ 2026-09-25 留档实测：两条门互补 —— 把 0.065 放到 0.055/0.045/0.020 三档结果
#   **一个数都不差**（24/28、拍照 2386、中位 8.2、最大 10.5、超半格 0）；而**只留形状门**
#   会退化到 27/28 但**超半格 2**（出现 75.5cm 落点）。⇒ 两条一起关之后，"假到达"
#   的防线只剩紫/橙/不对称/居中四条的**整帧份额**，落点精度必须重新实测
#   （到达时已打印"据画面估计面板中心距"，用现场几局把门槛收回来）。
# 形状门：目标观测 bbox 的**宽/高比上限**。
#   ⚠️ **2026-09-28 已按用户要求短路（0.0 = 关闭）**。原值与理由（留档）：
#       标定工具 `tools/_tmp_arrive_gate_probe.py`（门全放开、逐帧记特征、真值打标签）：
#         真到达（11 帧）宽/高 **1.53 ~ 2.07**（中位 1.66）
#         假到达（ 5 帧）宽/高 **2.25 ~ 2.49**（又扁又斜的残缺投影）⇒ 取 2.15。
#       但那批标定是在 **h=56cm** 的几何下做的；相机高度改成实测 33.9cm 后
#       "真到达簇"整簇挪进了原来的"假到达"区间（现场实测 2.83~3.08），
#       这条门在当前几何下对**真到达**恒判 ✗。
#   设 0 = 关闭该判据。
ARRIVE_MAX_WIDTH_HEIGHT_RATIO = 0.0
#   依据：正常到达时实测位置应在格心附近（半格 16.7cm 内）；40cm 已超过一整格，
#   说明 RANSAC 把观测配错了格 ⇒ 宁可退回旧行为，也不采纳一个离谱的位姿。
ARRIVE_ON_CELL_TOL_CM = 16.7     # 距格心容差（= 半格；压感区半宽 5.5cm，
#                                     但锚本身有位姿误差，取半格更稳）
# ⚠️ 已废弃（2026-09-26）：判决改为**只吃画面量**，锚的结论不再进到达门，
# 这个开关也就无处可施了。留着只为让旧日志/旧文档里的名字查得到。
# 解不了锚时"是否也拒绝"已经没有意义——我们不再拿锚拒绝任何一次到达。
ARRIVE_ON_CELL_REQUIRED = False   # 【废弃】锚不可用时是否也拒绝（不再被读取）
# ★ 到达时的锚测量：**纯诊断探针，不参与判决**（2026-09-26 新增）
#   True  = 画面判据成立后，额外跑一次 `_verify_target_cell`（多一遍 detect_panels
#           + _map_pose），只把"画面说到了、地图说不在"的分歧打进日志，供后续用
#           真值统计"核验本来会否决几次"。到达后的 `_reanchor_pose` 不受影响：
#           它照旧自己跑一次锚并采纳实测位姿（位置与航向都换）。
#   False = 连这次测量也不跑（到达帧少一遍全色检测）。
#   ⚠️ 置 False **不会**让否决权回来：否决逻辑已从判决路径删除（见 `_drive_to_panel`
#      到达段）。要恢复"用位姿否决到达"，必须回滚 2026-09-26 那次短路提交。
ARRIVE_ON_CELL_PROBE = True
# 到达后用视觉+地图反推位姿时，实测位置距"目标格心"超过此值就判为误解、弃用
# （40cm 已超过一整格 ⇒ 宁可退回"格心 + 原航向"，也不采纳一个离谱的位姿）。
REANCHOR_MAX_POSE_JUMP_CM = 40.0
# ---- 迟滞（2026-09-24 新增，评审 Q3/§6.6 要求）----
# 作用：**省拍照**，不是防发散。每次分区切换 = 一次决策 + 可能一次拍照（0.7s）。
# 做法：**只放宽"离开"条件，不放宽"进入"条件**——
#   进入窄档仍按名义边界；已经在窄档时，要放宽 ZONE_HYSTERESIS_DEG 才离开。
# 这样切换频率降为原来的 1/(1+HYST/带宽)，而"进入"语义不被污染。
# 取 2.0°：一个真实小转步长是 5.2~8.6°，迟滞小于一个步长就等于没有迟滞；
# 取太大（>半步长）会让窄区实际变宽、把 6° 门悄悄改成 9°。2° 是"能吸收
# 单帧噪声、又不改变门的名义值"的最小可用量。
ZONE_HYSTERESIS_DEG = 2.0
# 一次**尝试**走满多少步就收手换招（2026-09-25：从 150 提到 400）。
# 它不再是"这格不要了"的依据——走满只表示"这一招没磨下来"，外层会换搜索方向
# 重试**同一格**（顺序计分，绝不跳格）。提到 400 是因为"磨一格"本来就该允许
# 比"一次干净利落的走法"长得多：常规单格实测 20~90 步。
UNIFIED_MAX_STEPS = 400
# ★ 重捕获（P1）：目标不见了怎么办 —— 纯视觉、逐步转、有上限
#   为什么需要它：统一决策**不用死推提示**（用户明确放弃），而"上一格跑完时目标
#   经常在 180° 身后、距离可达 2 格"（实测面板3 距上一格 79cm）。若只做几小步
#   自转就放弃，整局会在第 2 格之后连续丢格。
#   机理：**逐步转过去 → 每一步都拍照复测**。转多少完全由视觉闭环决定（转多了
#   下一步反向吸收）。⚠️ 2026-09-25 起**连"往哪边转"也不再读地图方位**（航向误差
#   逐格累积、到 d7 有 −105.7°，方位会指反），改为"上次在画面哪一侧 + 重试反向"。
#   ⇒ 现状：`self.pose` **既不参与选方向，也不参与到达判决**（2026-09-26 短路）。
#   还在读它的只有两处，都要单独对待：`_arena_guard`（离场围栏，按位姿判越界，
#   是唯一硬停）与 `_reanchor_pose`（到达后**重写**位姿，不是读它做判断）。
FIND_MAX_TURN_DEG = 720.0   # 单次重捕获允许累计转过的角度（2 整圈；见下）
FIND_MAX_FRAMES = 90        # 单次重捕获拍照上限
# 为什么是 720°／90 帧（实测定的，不是拍的）：
#   上一格跑完时目标常在**正后方**。旧值（420°/40 帧 + 每 4 帧才转一步）实测只能
#   转过 ≈55°，于是"目标在身后"的整格**一步未动就放弃**——seed 3 有 4/7 格是这种
#   （逐格真值轨迹：起点=终点，一格都没走）。
#   算账：小转一步实测左 8.625°/右 5.200° ⇒ 转 180° 要 21~35 步；每 2 帧转一步
#   ⇒ 42~70 帧。取 90 帧/720°（=2 整圈）覆盖最坏情况（提示方向错、要绕一圈）。
#   ⚠️ 与"单格 360° 转向预算"的关系：预算是防"在原地打转"，重捕获是**有目标方向**
#   的搜索 ⇒ 二者分别记账（见 `_find_target_again`）。
FIND_STRIDE = 1             # 每丢几帧转一步
#   ★★ 取 1（每帧都转）是本轮**最大的单点收益**，实测（3 种子 21 格）：
#     每 2 帧转一步 → 到达 10/21、落点 [7.0, 8.4, 65.6, 32.8, 14.1, 64.9, 53.3]、
#       拍照 1709；
#     每帧都转     → 到达 **20/21**、落点 [7.0, 8.4, 6.1, 7.4, 6.9, 10.1, 8.8]、
#       拍照 1407。
#     为什么：转 180° 要 21~35 步，"隔一帧再转"就吃掉 42~70 帧、正好卡在 90 帧的
#     重捕获上限附近 ⇒ **扫不完就被判失败**（实测失败格恰好停在 117≈110+头部 5 帧，
#     而当时那道 140 帧的硬上限从未触发 ⇒ 不是预算不够，是这个节流把它卡死了；
#     2026-09-25 起这类上限已整族取消，见 levels/nine_grid_shared.py 文件头）。
#     每步都复测本来就是这套循环要的语义；"隔几帧再转"省下的是动作，赔上的是整格。
#   为什么需要它：`_body_angle_deg` 用的是 `CAMERA_FOV_H_DEG=60` 这条**伺服增益**，
#   而画面真实横向标定是 ~154.8px/°（±33.7° 半视场 / 1296px）⇒ 60/154.8 = 0.388。
#   实测（固定站位只改真航向，跟同一个面板）：真航向 −20/0/+20° 时
#     几何方位 +20.00/0.00/−20.00，而 `_body_angle_deg` 读 −7.36/+0.32/+7.99
#     ⇒ 比值 −7.36/20 = 0.368、7.99/20 = 0.400 ⇒ 取 0.39。
#   ⚠️ 偏差 0.03 就足以把"修 15°"变成"修 11°或 19°"⇒ 拿它当**修复量**用之前
#   必须先按上面这套参数现场重标（真机 FOV 与这 154.8px/° 不同）。这也是本项目
#   `test_nine_grid_yaw.py` 反复强调的那条：yaw 是**增益**，不是角度。
# ---- 前进的快慢两档（**整帧色占比**分界，见 `_drive_to_panel` 的说明）----
# 实测 `whole`↔距离：60cm→0.056｜50→0.072｜40→0.092｜35→0.103｜30→0.115
#                    25→0.126｜20→0.137｜15→0.147（此后随裁切不再增）
FORWARD_COVER_MID = 0.085          # < 此值（≈42cm 以外）走大步
FORWARD_COVER_NEAR = 0.115         # < 此值（≈30cm 以外）走中步；≥ 此值恒 1 步
# ★ 目标丢失后的搜索小转：**一次发几步**（2026-09-28 用户要求：改成一次 3 步）
#   原来一次 1 步、每步拍帧复测 ⇒ 目标在身后要磨十几轮。3 步 ≈15.5~25.9°
#   （左 8.625°/右 5.200° × 3），远小于 ±33.7° 半视场，不会跳过视野。
SEARCH_TURN_STEPS = 3
# ★ 本格开始时**先抬头看一眼**（用户 2026-09-28 要求）：抬头档视野远，目标在
#   中远处更容易被看到；看到了就进主循环朝它走（判档会判"绿·直行"）。
#   置 False 可一键关掉（它只多花 1 张照片；主循环仍用低头档）。
SEARCH_LOOK_UP_FIRST = True
SEARCH_LOOK_UP_PITCH = PITCH_NAV
FORWARD_FAR_STEPS = 5            # 远距一次前进步数（5×2.652 ≈ 13cm）
FORWARD_MID_STEPS = 3            # 中距一次前进步数（3×2.652 ≈ 8cm）
# ★ 2026-09-28：从**参考原版**搬回来的"前进不力 ⇒ 跨步"机制。
#   原版（levels/nine_grid_original/level.py:157-207）：每前进一次比 `proximity`
#   （框宽）变化，变化 < `FORWARD_EFFICIENCY_MIN_PX=5`（@1280 宽画幅）就判前进不力，
#   随后 move_backward_one() + over_hurdle() 脱困、重拍重识别。
#   这里保留"用框宽判不力"的语义，阈值按本工程画幅换算并留实测余量：
#     实测"走一步(2.652cm)应带来的框宽增量" = 25~50cm 段 **~30~39px**
#     （投影算出来的，见 tools/exp_* 与现场 1146↔1164px 的帧间抖动 ~6~16px）
#   ⇒ 取 25px：比抖动上限高一档、比正常增量低一档。
FORWARD_PROGRESS_MIN_PX = 25.0
#   ⚠️ 2026-09-28 **已废弃**（用户现场证伪）：框宽在"路线与面板平行"时**本来就不变**
#   （现场实测连续两帧 1158→1158px），于是被判"前进不力"⇒ 跨一大步 ⇒ **直接冲出**。
#   现行判据见下：用**到达判据各量**的变化，不看框宽。
# ★ 现行"前进不力"判据（用户 2026-09-28 指定）：
#   若上一个动作是**直行**，取"到达判据"的观测量组成向量
#       v = (紫, 橙, 整帧占比)
#   （到达门里能连续读到的三个量），算前后两帧的**差值平方和**
#       d² = Σ (v_i - v_i')²
#   连续 `NO_PROGRESS_PATIENCE` 次 d² < `NO_PROGRESS_D2_MIN` ⇒ 判"前进不力"。
#   为什么比框宽/占比单量好：这三个量合起来描述"面板在我脚下的程度"，
#   机器人真在走近时它们必然同时变化；而框宽会被视角直接抵消（平行路线不变）。
#   阈值取 1.5e-4：现场同一姿势下这三个量的帧间抖动 ≈ 0.001~0.005，
#   正常走近一步的合成变化 ≈ 0.01~0.05 ⇒ 门取在"抖动以下、真变化以上"。
NO_PROGRESS_D2_MIN = 1.5e-4     # 差值平方和下限（低于此 = 这一帧几乎没变化）
NO_PROGRESS_PATIENCE = 2        # 连续几次才判不力（单帧抖动不误判）
# ★ "不力"之后的处置（用户指定，**严禁任何大批量直行**）：
#       后退 ×2  →  越障（hurdles，净前进 ≈15cm）  →  后退 ×1
#   为什么不能像之前那样"跨一大步（go_forward_one_step ×5）"：那正是把机器人
#   直接送出格子的原因（现场：一次"不力"直接冲出）。
NO_PROGRESS_RECOVER_ACTIONS = (
    ("back_one_step", 2),
    ("hurdles", 1),
    ("back_one_step", 1),
)
#   ⚠️ 旧做法（`go_forward_one_step ×5` 跨步）**已废止**：现场实测"一次不力直接冲出"。
#   脱困一律走 NO_PROGRESS_RECOVER_ACTIONS（后退×2 → 越障 → 后退×1）。
FORWARD_STRIDE_STEPS = 5        # 仅留给"远距正常快进"分档，不再用于脱困
# =====================================================================
# ★ 小转角**批量**（2026-09-28，用户要求）：按在线 EMA 估计一次小转的实际角，
#   一次决策可下发**至多 8 步**连续转向。**左/右各自独立**，且每下发一批就
#   在**下一帧立刻**用"批前偏角 → 批后偏角"更新该方向的 EMA。
# =====================================================================
# 为什么需要：统一决策循环原先是"一次决策 = 一个 turn_*_small_step"，
# 于是"每步实际只转 3~5°"时就得靠一帧一帧慢慢磨（现场实测：偏角 +25°→+13.5°
# 用了 10 帧，每帧约 1° 的读数变化）。参考原版的做法是**按估计角算批量**：
#   levels/nine_grid_original/level.py:736
#     turn_count = max(1, int(abs(yaw) / small_turn_angle))
#   （`MAX_FINAL_TURNS=3` 只在"已经看到目标颜色"的收尾段夹一层上限）
# 三段式那边也早有同款闭环（`nine_grid_three_stage.py:711`），本关统一决策缺这一块。
# ★ 左右独立（现场实测左 8.625°/右 5.200°，差 66%，是真实特性）：
#   两个方向各存一份 EMA，互不污染；下发哪一侧就只更新哪一侧。
# 语义与三段式对齐（那边是"估计越小组数越多，每轮重新拍帧复测、过冲由下一轮吸收"）：
#   每批转向后立刻用偏角变化更新该方向 EMA，再
#   n = clip(floor(|偏角| / max(该方向EMA, 下限)), 1, 8)。
# ⚠️ 下限（0.8°/步）必须有：没有它时"估计崩到 0.3°"会把 n 算到上限、一次转过冲
#   （三段式的注释里记着这个坑）。有下限后最多请求 8 小步，且每轮都复测。
TURN_BATCH_MAX_STEPS = 8        # 一轮小转批量上限（用户指定：至多 8 步）
TURN_BATCH_MIN_STEP_DEG = 0.8   # 规划下限（°/步）：防"估计崩到 0"⇒ 一次转过冲
TURN_BATCH_EMA_ALPHA = 0.3      # 单步实际角的指数滑动平均
TURN_BATCH_EMA_ALPHA_FAST = 0.5  # 前几批用快收敛，抵消初值偏差
TURN_BATCH_EMA_FAST_UPDATES = 2  # 快收敛批次数
TURN_BATCH_MIN_EFFECTIVE_DEG = 0.5  # 单步实测低于此 = 被地面吞掉（只记日志）
# 真实"像素 ↔ 角度"换算（**不是** yaw_gain 那条伺服增益）：±33.7° 半视场 / 1296px
#   ⇒ ≈154.8 px/°（见 `_body_angle_deg` 的实测说明：那条 0.388 的比值正是
#     60/(154.8) 的产物）。EMA 必须用**真实角**，否则量纲自己就错了。
TURN_BATCH_FOV_PX_PER_DEG = 154.8
# 方向 → (动作名, EMA 属性名, 中文名)
TURN_BATCH_DIRS = {
    "left": ("turn_left_small_step", "_small_turn_left_deg", "左"),
    "right": ("turn_right_small_step", "_small_turn_right_deg", "右"),
}
# ★ 2026-09-28 新增的近距保险丝：**面板已经"看全了"就恒走 1 步**。
#   为什么（现场实测，用户 2026-09-28 明确质问"最后连走五大步是哪个傻逼想出来的"）：
#     "整帧占比"在近距**不是单调量**——面板下沿一出画幅，whole 就往下掉
#     （实测 24cm→0.118、现场 15cm 处只剩 0.058），于是分档一路判"还很远"、
#     一路发大档；而一次 5 步 = 13cm，在 20cm 的尺度上是一脚踩过去，
#     于是"到不了 + 冲过板"同时发生（日志：紫 0.513/橙 0.063/居中 0.012
#     这一帧之后仍走 ×5）。
#   判据用**框宽**（面板完整可见时它在近距趋于饱和，实测稳定在 1142~1156px，
#   一旦明显更小就说明还没走近），比 whole 单调：
#     框宽 ≥ FORWARD_BOX_NEAR_PX（面板基本看全）⇒ n = 1
FORWARD_BOX_NEAR_PX = 1100.0
# ★ 近距判定余量（2026-09-28 现场事故补）：紫份额只要到"门 − 这个余量"就认为
#   面板已经压进紫带 ⇒ 前进恒走 1 步。现场面板6 那帧紫 0.340（门 0.35，只差
#   0.01）却仍走 ×5 ⇒ 踩过去。取 0.05（约等于现场帧间紫的自然抖动上限）。
NEAR_ARRIVE_MARGIN = 0.05
# ---- 前进无效脱困（**2026-09-28 改成用框宽判**，见循环里那段长注释）----
# 旧版用整帧色占比判（NO_PROGRESS_COVER / NO_PROGRESS_PATIENCE，已删除）：
# 那个量在近距不单调（面板下沿出画幅后 whole 正常下降），近距永远判不出卡住。
# 现行判据 = 框宽增量 < FORWARD_PROGRESS_MIN_PX（参考原版同款语义），
# 脱困阶梯 = 跨步（FORWARD_STRIDE_STEPS）→ 后退一步重识别。


# =====================================================================
# ★ 统一判据的**纯函数实现**（模块级；无 self、无硬件、不碰相机）
# =====================================================================
# 「判据」= 上面那五个示意图参数（+ 紫区高度）在画面里划出的分区：看目标点落在
# 哪一块，就下发哪个动作。这里把它写成三个**同一套公式**的纯函数：
#
#   zone_at_pixel(px, py, w, h, prev)  逐点判档（决策用；返回 forward/side/turn）
#   zone_masks(w, h)                   四块区域的布尔掩膜（到达判据的份额用）
#   zone_lines()                       分区分割线（**画图用**；归一化画幅坐标）
#
# 为什么单独拎成模块级函数（2026-09-25）：
#   ① **公式只留一份**：掩膜、判档、分割线都调下面三个"半边宽度"函数。以前掩膜与
#      判档各写一遍边界表达式，只能靠测试盯着它们别走散；现在画出来的线与判据
#      在结构上就是同一条式子（tests/test_nine_grid_zone_px.py 再钉一层）。
#   ② **调试镜像能直接复用**：`tools/debug_server.py` 用它们把"判据点＋分区线"
#      叠加到真机照片上（见 debug.sh 的 zone 选项）——不复制几何、不复制阈值，
#      现场看到的线就是关卡真正在用的线。
#   ③ ⚠️ **函数体在调用时读本模块全局常量**（不是 import 期捕获、不做默认参数）：
#      `tools/record_run.py --tune` 与若干测试靠运行时改 `ZONE_*` 改参，改成
#      "常量捕获"会让改参**静默失效**——那正是本仓库反复踩过的那类坑。
#
# 坐标约定（三处一致）：x 归一化用**画幅宽**、y 归一化用**画幅高**；t=0 是画幅
# 顶边（远）、t=1 是底边（近）；中线 0.5 是"机体正前方"（头部折算见
# `_center_column_px`）。

def zone_green_half(t):
    """绿走廊半宽 / 画幅宽（上窄下宽的**斜线**；t = 纵向归一化位置）"""
    return ZONE_GREEN_TOP + (ZONE_GREEN_BOT - ZONE_GREEN_TOP) * t


def zone_blue_half(u):
    """蓝外沿半宽 / 画幅宽（斜线；u = 在蓝区高度内的归一化位置，0=蓝区上沿）"""
    return ZONE_BLUE_TOP + (ZONE_BLUE_BOT - ZONE_BLUE_TOP) * u


def zone_purple_top_t():
    """紫区上沿的纵向位置（t；紫区占画幅下方 ZONE_PURPLE_HEIGHT）"""
    return 1.0 - float(ZONE_PURPLE_HEIGHT)


def zone_blue_top_t():
    """蓝区上沿的纵向位置（t；蓝区占画幅下方 ZONE_BLUE_HEIGHT）——"太远不该横移"那条线"""
    return 1.0 - float(ZONE_BLUE_HEIGHT)


def zone_at_pixel(px, py, w, h, prev=None):
    """★ 按示意图判档：绿/蓝/橙三块**画在画面里**，看目标落在哪一块

    只用目标在画面里的位置 (px, py)，不用角度、不用框宽：
        t  = py / h            （0 = 画幅顶边(远)，1 = 画幅底边(近)）
        dx = |px − w/2| / w    （横向偏移占画幅宽的比例）
        绿半宽 = 绿上沿 + (绿下沿 − 绿上沿)·t      —— 斜线（上窄下宽）
        dx ≤ 绿半宽                        → "forward"（直行）
        否则若 t ≥ 1 − 蓝区高（在蓝区高度内）：
            蓝半宽 = 蓝上宽 + (蓝下宽 − 蓝上宽)·u，u = 在蓝区内的归一化高度
            dx ≤ 蓝半宽                    → "side"（横移）
        其余                                → "turn"（旋转）

    迟滞（ZONE_HYSTERESIS_DEG，只放宽"离开"条件）按 `度/画幅角` 折算成横向比例，
    语义与旧规则一致：已在本档时边界放宽一点点，避免一两像素抖动来回切档。

    ⚠️ 这是**判据真源**；类方法 `NineGridLevel._zone_at_pixel` 只是它的薄包装
    （老调用点不用改），调试镜像与仿真也直接调本函数。
    """
    if w <= 0 or h <= 0:
        return "turn"
    t = float(np.clip(py / h, 0.0, 1.0))
    dx = abs(float(px) - w / 2.0) / w
    hy = (ZONE_HYSTERESIS_DEG / CAMERA_FOV_H_DEG) if ZONE_HYSTERESIS_DEG > 0 else 0.0
    half_green = zone_green_half(t)
    if prev == "forward":
        half_green += hy
    if dx <= half_green:
        return "forward"
    bh = float(ZONE_BLUE_HEIGHT)
    if t >= zone_blue_top_t():
        u = (t - zone_blue_top_t()) / bh if bh > 0 else 1.0
        half_blue = zone_blue_half(u)
        if prev == "side":
            half_blue += hy
        if dx <= half_blue:
            return "side"
    return "turn"


def zone_masks(w, h):
    """把示意图四块区域栅格化成布尔掩膜（**无缓存**的纯函数；缓存见类方法）

    上(远) ┌───────────────┐
           │ 橙 │  绿  │ 橙 │   绿 = 走廊内 且 紫区高度**以上**
           ├────┼─────┼────┤   紫 = 走廊内 且 紫区高度**以内**
           │ 橙 │蓝│紫│蓝│ 橙│   蓝 = 蓝区高度内 且 走廊外、蓝外沿以内
           └───────────────┘   橙 = 其余
    下(近)
    「走廊」= |dx| ≤ 绿半宽(t)（上窄下宽的斜线）；紫区的横向边界**就是**走廊，
    不另设宽度（见 ZONE_PURPLE_HEIGHT 注释）。蓝左/蓝右按画幅中线切开，
    供"左右溢出不对称"（站在格子中央的判据）用。
    """
    w, h = int(w), int(h)
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    dx = np.abs(xs - w / 2.0) / w                       # (w,)
    t = ys / h                                          # (h,)
    half_green = zone_green_half(t)
    in_corr = dx[None, :] <= half_green[:, None]        # (h,w) 走廊
    below_p = t >= zone_purple_top_t()
    purple = in_corr & below_p[:, None]
    green = in_corr & (~below_p)[:, None]
    bh = float(ZONE_BLUE_HEIGHT)
    below_b = t >= zone_blue_top_t()
    u = (np.clip((t - zone_blue_top_t()) / bh, 0.0, 1.0) if bh > 0
         else np.ones_like(t))
    half_blue = zone_blue_half(u)
    blue = below_b[:, None] & (~in_corr) & (dx[None, :] <= half_blue[:, None])
    orange = ~(green | purple | blue)
    left = (xs < w / 2.0)[None, :]
    return {"green": green, "purple": purple, "blue": blue, "orange": orange,
            "blueL": blue & left, "blueR": blue & (~left)}


def zone_lines():
    """分区分割线（**归一化画幅坐标** 0~1，与分辨率无关）——给叠加图用

    返回 [[(fx0, fy0), (fx1, fy1)], ...]：
      ① 绿走廊左/右斜边（画幅顶边 → 底边）；
      ② 紫区上沿（横线，只画走廊内那一段）；
      ③ 蓝区上沿左/右两段（横线，从走廊外沿到蓝外沿）；
      ④ 蓝外沿左/右斜边（蓝区上沿 → 画幅底边）。
    缺省参数下正好 **7 段**；参数退化（如蓝区高 ≥1 或 ≤0）时跳过对应段而不是
    抛异常——调试工具不该因为改了个参数就崩。

    为什么返回归一化坐标而不是像素：分区几何本来就是"占画幅多少"定义的，
    照片分辨率、/photo 的降采样都不影响它（前端按 canvas 宽高直接乘即可）。
    """
    segs = []

    def _seg(x0, y0, x1, y1):
        x0, x1 = float(np.clip(x0, 0.0, 1.0)), float(np.clip(x1, 0.0, 1.0))
        y0, y1 = float(np.clip(y0, 0.0, 1.0)), float(np.clip(y1, 0.0, 1.0))
        if abs(x1 - x0) < 1e-9 and abs(y1 - y0) < 1e-9:
            return                              # 零长段（参数退化）不画
        segs.append([[round(x0, 6), round(y0, 6)],
                     [round(x1, 6), round(y1, 6)]])

    # ① 走廊（绿/紫的外边界）
    g_top, g_bot = zone_green_half(0.0), zone_green_half(1.0)
    _seg(0.5 - g_top, 0.0, 0.5 - g_bot, 1.0)
    _seg(0.5 + g_top, 0.0, 0.5 + g_bot, 1.0)

    # ② 紫区上沿（到达判据的那条线；紫区比蓝区高，通常落在画幅中部）
    tp = zone_purple_top_t()
    if 0.0 < tp < 1.0:
        hp = zone_green_half(tp)
        _seg(0.5 - hp, tp, 0.5 + hp, tp)

    # ③④ 蓝区（横移档）：上沿横线 + 外沿斜线
    tb = zone_blue_top_t()
    if 0.0 < tb < 1.0:
        inner = zone_green_half(tb)             # 蓝上沿的"内端"= 走廊边
        outer = zone_blue_half(0.0)             # 蓝上沿的"外端"= 蓝外沿
        if outer > inner:
            _seg(0.5 - outer, tb, 0.5 - inner, tb)
            _seg(0.5 + inner, tb, 0.5 + outer, tb)
        b_top, b_bot = zone_blue_half(0.0), zone_blue_half(1.0)
        _seg(0.5 - b_top, tb, 0.5 - b_bot, 1.0)
        _seg(0.5 + b_top, tb, 0.5 + b_bot, 1.0)
    return segs


# =====================================================================
# 统一决策：三档分区 + 一个循环走完全程
# =====================================================================


class NineGridLevel(NineGridShared):
    """数字宫格（统一决策）：看目标落在画面哪一档，一个循环走到目标

    每帧一次拍照、一次判档、一个动作；不看历史峰值、不用推算位姿做提示、
    不用 yaw 的数值做规划。共用机制见 levels/nine_grid_shared.py。
    """
    def go_to_panel(self, digit, attempt=1):
        """前往第 digit 块面板（统一决策）：返回是否确认到位

        `attempt` = 本格第几次尝试。外层在不确认时**继续磨这一格**（顺序计分，
        跳格等于丢掉后面的分），每次把 attempt +1 传进来；本办法用它**换搜索
        方向**（奇偶次反向），这是重试时唯一能换的东西。

        逐格落点在返回前留档（诊断与回归用，不参与控制）——必须在返回前记，
        因为调用方随后就会走向下一格（或重试本格）。
        """
        try:
            return self._drive_to_panel(digit, attempt)
        finally:
            self._record_landing(digit)
    def _drive_to_panel(self, digit, attempt=1):
        """★ 统一决策（P1，唯一路径）：三档分区 ＋ 一个循环走完全程

        与旧三段式（`_seek_align_approach` + `_walk_until_underfoot`）的对照：

        | | 三段式（已删） | 本法 |
        |---|---|---|
        | 决策依据 | ALIGN `\\|yaw\\|≤12°` / APPROACH 框宽≥920px / ARRIVE 色占比回落，**三段三套** | **每帧同一套** `_zone_at_body_pixel` → 直行/横移/旋转 |
        | 交棒 | 有交棒点 ⇒ 交棒 yaw 残余留给低头段兜 | **没有交棒点**，也就没有这个漏 |
        | 每次下发几个原语 | 批量（`n = floor(\\|yaw\\|/步长)`，最多 6） | **恒为 1 个**（用户明确放弃批量） |
        | yaw 的用途 | 数值 → 规划步数 | **只作伺服增益**，判档纯像素 |
        | 死推 | 搜索提示/盲走/护栏都吃它 | **只进遥测**，不进决策 |
        | 取景档 | 导航档/低头档两档切换 | **全程 1040**（用户："1040 是最好的"） |

        到达判据沿用"只看当前帧"的四量（紫/橙/左右不对称/几何锚），
        不含任何历史 ⇒ 原地转身、偏头、遮挡都不会让它误成立。

        `attempt > 1` = 上一次尝试没磨下来，这次**换搜索方向**再试（不放弃本格）。
        """
        self._begin_cell_attempt(digit, attempt)
        self._target_seen = False
        self._loc_count = 0
        self._current_digit = digit
        self._turn_updates = 0
        setter = getattr(self.state, "set_deform_digit", None)
        if setter is not None:
            setter(digit)
        color = self._target_color(digit)
        self.phase = f"DRIVE{digit}"
        pitch0 = getattr(self.state, "pitch", None)
        self.state.set_pitch(PITCH_DOWN)
        # ★ 2026-09-28 用户要求：**本格开始先抬头看一眼**——抬头档（PITCH_NAV）
        #   看得远，目标面板在中远处更容易被看到；看见就直接进主循环（判档会
        #   判成"绿·直行"就朝它走），不必先低头盲搜。
        #   注意：抬头档只用于**快速确认"目标在不在视野里"**，主循环仍用低头档
        #   （低头档是"能压到格心"的那一档，见 `本关全程 1040` 的决策）。
        if SEARCH_LOOK_UP_FIRST:
            self._look_up_and_check(digit, color)
            self.state.set_pitch(PITCH_DOWN)
        lost = 0
        turns = 0
        self._reacq_deg = 0.0          # 本轮重捕获已累计转过的角度
        self._reacq_frames = 0
        self._reacq_dir = None         # 本轮重捕获的转向方向（一次定死）
        if attempt > 1:
            # ★ 重试时唯一能换的东西：**搜索方向**。单格失败最常见的原因就是
            #   "目标在身后、而这一侧扫不到"，奇偶次反向能直接换个半圆去找。
            self._last_seen_side = 1.0 if attempt % 2 == 0 else -1.0
            print(f"[重试] 面板{digit} 第 {attempt} 次尝试：换搜索方向"
                  f"（{'左' if attempt % 2 == 0 else '右'}转起扫）")
        prev_zone = None
        prev_cov = None
        prev_org = None
        prev_box = None                # 上一帧框宽（仅诊断/日志用）
        prev_gate_vec = None           # 上一帧的 (紫,橙,占比)：前进不力判据的基准
        prev_turn = None               # 上一次转向前的偏角（日志用）
        prev_turn_n = 0                # 上一次转向下发的步数
        prev_turn_dir = None           # 上一次转向的方向（"left"/"right"）
        prev_px_of_turn = None         # 上一次转向前的目标像素列（算单步实际角用）
        stall = 0
        try:
            for _ in range(UNIFIED_MAX_STEPS):
                if self._attempt_stuck:
                    # 本次尝试的招数用尽（转角转满一圈等）→ 收手，外层换办法重试
                    return False
                p = self._capture_and_measure(color)
                if p is None:
                    return False
                self._reacq_frames += 1
                sh = p["shares"]
                # ---- 到达证据（只看当前帧，**且全是画面量**）----
                #   判据本身 = `_arrive_pixels_ok()`：居中 ∧ 够大 ∧ 形状不扁 ∧
                #   紫够多 ∧ 橙够少 ∧ 左右不过偏 —— 六条的理由与标定数据都在
                #   那个函数的注释里（原先堆在这里，2026-09-26 归位）。
                # ★ **位姿不参与判决**（2026-09-26 短路）：原先还有一道"格的
                #   同一性核验"（锚解实测位姿距目标格心 ≤16.7cm 才放行）。实测它会
                #   拿一个**不可信**的位姿否决**正确**到达（形变场景实测位姿漂
                #   105cm 而真值离格心只有 13cm）——上位法早已写明"不要用位姿去
                #   否决到达"。现在锚测量只在 `ARRIVE_ON_CELL_PROBE` 下跑一次，
                #   **只进日志、不改变控制流**。
                if sh is not None:
                    pur, org, asym = float(sh[1]), float(sh[2]), float(sh[3])
                    if self._arrive_pixels_ok(sh, p["obs"], p["px"], p["w"]):
                        self._arrive_probe_note(digit, p["frame"])
                        self._arrive_evidence = (
                            f"脚下色块判据：紫区占比 {pur:.3f}（≥"
                            f"{ARRIVE_PURPLE_MIN}），橙区占比 {org:.3f}（≤"
                            f"{ARRIVE_ORANGE_MAX}），左右不对称 {asym:+.3f}"
                            f"（≤{ARRIVE_ASYMMETRY_MAX}）")
                        print(f"[行进] 面板{digit} 区域判据成立（紫 {pur:.3f}、"
                              f"橙 {org:.3f}、不对称 {asym:+.3f}）→ 判定到达")
                        # ★★ 到达后**重建位姿**（P1 重写此前漏掉的一步，后果严重）：
                        # 统一循环原来只设 `_arrive_evidence` 就返回 True，
                        # `_reanchor_pose` 只被旧链路调用 ⇒ 新循环里位姿**从不重建**、
                        # 一直靠死推漂移。实测（seed21 逐格 pose_θ−真值航向）：
                        #   d1 −0.0° → d3 −7.7° → d5 −49.2° → d7 **−105.7°**
                        # ⇒ "地图方位"到后面方向都指反、搜索朝错方向连转 140°+，
                        #   也就是用户看到的"异常的多次旋转"。
                        # 现在调用 `_reanchor_pose`：它按用户澄清改用
                        # **视觉+地图反推的实测位姿**（`_map_pose`），
                        # 解不出才退回"格心 + 原航向"。
                        self._reanchor_pose(digit)
                        return True
                o = p["obs"]
                if o is None:
                    lost += 1
                    # 目标丢失帧的偏角不可比（观测消失）⇒ 作废转向批量的测量基准，
                    # 免得重捕获期间转的那些步被算进"单步实际角"的 EMA。
                    prev_turn, prev_turn_n, prev_turn_dir = None, 0, None
                    turns, stepped = self._find_target_again(digit, lost, turns, p)
                    if turns < 0:
                        return False
                    self._reacq_deg += abs(stepped)
                    if stepped and (self._reacq_deg > FIND_MAX_TURN_DEG
                                    or self._reacq_frames > FIND_MAX_FRAMES):
                        # 本次尝试的搜索招数用尽（转完参考额度仍没见到目标）：
                        # 收手，外层会**换搜索方向重试本格**，绝不放弃这一格。
                        self._give_up_attempt(
                            f"重捕获已转 {self._reacq_deg:.0f}°／"
                            f"{self._reacq_frames} 帧仍未见到目标")
                        return False
                    continue
                lost = 0
                turns = 0
                self._reacq_deg = 0.0
                self._reacq_frames = 0
                self._reacq_dir = None
                # ---- 三档分区（判档只吃像素）----
                head_pulse = getattr(self.state, "current_head_pulse",
                                     HEAD_CENTER)
                zone, dx_body = self._zone_at_body_pixel(
                    p["px"], p["py"], p["w"], p["h"], head_pulse,
                    self._zone_last)
                self._zone_last = zone
                # 记"最后一次看到它在哪一侧"——丢失后自转搜索靠它定方向
                self._last_seen_side = (1.0 if p["px"] < p["w"] / 2.0 else -1.0)
                px_body = p["px"] - (self._center_column_px(p["w"], head_pulse)
                                     - p["w"] / 2.0)
                act = self._action_for_zone(zone, px_body, p["w"])
                yaw_gain = self._body_angle_deg(p["px"], p["w"], head_pulse)
                # ---- 小转角**批量**：按在线估计的单步实际角算这一轮转几步 ----
                # （2026-09-28，用户要求；机制说明见常量块 TURN_BATCH_MAX_STEPS）
                # ① **立刻更新**上一次那批所属方向的 EMA（左/右各自独立）：
                #    必须"哪一侧发的批就更新哪一侧"，且**第一帧可测就更新**
                #    （用户要求：完成左/右批量后就立刻更新）。过冲时按 |a|+|b|
                #    折算，否则 per 变负 → EMA 崩向 0 → 下一轮请求更多步 → 更过冲
                #    （三段式注释里记过这个坑）。
                if (zone == "turn" and prev_turn is not None
                        and prev_turn_n > 0 and prev_turn_dir in TURN_BATCH_DIRS):
                    _act_name, attr, cn = TURN_BATCH_DIRS[prev_turn_dir]
                    # ★★ 2026-09-28 现场事故（用户指出）：偏角读数从 +9.3° 一路
                    #   涨到 +10.8°（**机器人一步都没转过去**），而这里把
                    #   "读数变了"当成"转过去了"，EMA 反而稳步上涨到 9.8°/步。
                    #   两个根因：
                    #   ① 量的单位错：`yaw_gain` 是**增益折算值**（×0.388），
                    #      不是真实角度；用一个未标定的增益去标定"单步实际角"，
                    #      量纲自己就先错了。改用**像素变化**换算真实角
                    #      （TURN_BATCH_FOV_PX_PER_DEG）。
                    #   ② 没有"没转过去"的判别：读数朝**错方向**变化时不能更新 EMA。
                    #      ⇒ 只有"转后读数往该方向变"（绝对值变小，或过冲后反向）
                    #      且单步实测 ≥ TURN_BATCH_MIN_EFFECTIVE_DEG 才更新。
                    #   顺带：EMA 上限 = 该方向已知的最快单步（左 8.625/右 5.2），
                    #      不可能比"名义值"还快（旧 clip 上限 10.0 会一直涨）。
                    _px = float(p["px"])
                    per = None
                    if prev_px_of_turn is not None:
                        per = abs(_px - prev_px_of_turn) / TURN_BATCH_FOV_PX_PER_DEG \
                            / prev_turn_n
                    # 方向自检：读数必须朝"目标更靠近居中"的方向变，否则判"没转过去"
                    moved_right_way = (
                        (prev_turn_dir == "left" and _px < prev_px_of_turn)
                        or (prev_turn_dir == "right" and _px > prev_px_of_turn)
                    ) if prev_px_of_turn is not None else True
                    cap = float({"left": TURN_LEFT_SMALL_DEG,
                                 "right": TURN_RIGHT_SMALL_DEG}[prev_turn_dir])
                    if per is None or per < TURN_BATCH_MIN_EFFECTIVE_DEG \
                            or not moved_right_way:
                        # 没测到 / 几乎没动 / 朝错方向 ⇒ **不更新 EMA**，
                        # 并交给打滑保护计数（它会连续两次后 ≈ 后退一步脱困）
                        per_show = 0.0 if per is None else per
                        print(f"[转向] {cn}转这一批**没转过去**（读数 "
                              f"{(0.0 if prev_px_of_turn is None else prev_px_of_turn):.0f}"
                              f"→{_px:.0f}px，约 {per_show:.2f}°/步）"
                              f"→ 不更新 EMA")
                        if self._check_turn_slip(per_show, direction_cn=cn,
                                                 n_steps=prev_turn_n):
                            prev_turn, prev_turn_n, prev_turn_dir = None, 0, None
                            prev_px_of_turn = None
                            prev_zone = None
                            continue
                    else:
                        per = float(np.clip(per, 0.3, cap))
                        alpha = (TURN_BATCH_EMA_ALPHA_FAST
                                 if self._turn_updates < TURN_BATCH_EMA_FAST_UPDATES
                                 else TURN_BATCH_EMA_ALPHA)
                        self._turn_updates += 1
                        before = float(getattr(self, attr))
                        setattr(self, attr, (1.0 - alpha) * before + alpha * per)
                        print(f"[转向] {cn}转批量实测更新：{before:.1f} → "
                              f"{getattr(self, attr):.1f}°/步"
                              f"（上一批 {prev_turn_n} 步，像素 {prev_px_of_turn:.0f}→"
                              f"{_px:.0f}，单步 {per:.2f}°，上限 {cap:.1f}°）")
                        # 统一决策里 `_small_turn_deg` 只作"聚合显示"用（三段式仍用它
                        # 规划），这里同步成左右均值，免得日志里那个数永远停在初值。
                        self._small_turn_deg = 0.5 * (self._small_turn_left_deg
                                                      + self._small_turn_right_deg)
                        # ★ 旋转打滑保护（用户 2026-09-28）：连续 TURN_SLIP_STRIKES
                        #   次实测单步角 < TURN_SLIP_MIN_DEG ⇒ 后退一步 + 清 EMA。
                        if self._check_turn_slip(per, direction_cn=cn,
                                                 n_steps=prev_turn_n):
                            prev_turn, prev_turn_n, prev_turn_dir = None, 0, None
                            prev_px_of_turn = None
                            prev_zone = None
                            continue
                # ② 算批量：偏角大 ⇒ 多转几步；**该方向**估计角小 ⇒ 自动多转
                n_turn = 1
                turn_dir = None
                if zone == "turn":
                    turn_dir = "left" if px_body < p["w"] / 2.0 else "right"
                    est = float(getattr(self, TURN_BATCH_DIRS[turn_dir][1]))
                    step_est = max(est, TURN_BATCH_MIN_STEP_DEG)
                    n_turn = int(np.clip(int(np.floor(abs(yaw_gain) / step_est)),
                                         1, TURN_BATCH_MAX_STEPS))
                # ---- 前进的**快慢两档**：远距多走几步、近距一步一复测 ----
                # ★ 分档量一开始只用**整帧色占比 `whole`**；2026-09-28 现场证伪了
                #   它的单调性：面板下沿一出画幅，whole 就往下掉（实测 24cm→0.118，
                #   现场 15cm 处只剩 0.058），于是分档一路判"还很远"、一路发大档
                #   （日志：紫 0.513/橙 0.063/居中 0.012 那一帧之后仍走 ×5 = 13cm，
                #    直接踩过面板）。所以现在多一条**框宽保险丝**：面板一旦"基本看全"
                #   （框宽 ≥ FORWARD_BOX_NEAR_PX，实测完整可见时稳定在 1142~1156px），
                #   就恒走 1 步（2.652cm），把最后这段交给"一步一复测"。
                # 实测 `whole`↔距离（低头 1040，正前方）：
                #   60cm→0.056｜50→0.072｜40→0.092｜35→0.103｜30→0.115｜25→0.126
                #   20→0.137｜15→0.147（**此后随裁切不再增**）
                n = 1
                if zone == "forward" and sh is not None:
                    cov = float(sh[0])
                    if cov < FORWARD_COVER_MID:
                        n = FORWARD_FAR_STEPS
                    elif cov < FORWARD_COVER_NEAR:
                        n = FORWARD_MID_STEPS
                    box = float(p.get("box") or 0.0)
                    if box >= FORWARD_BOX_NEAR_PX:
                        n = 1          # 看全了 ⇒ 一步一复测（近距保险丝）
                    # ★★ 2026-09-28 现场事故：面板6 那一段"紫 0.34 / 橙 0.30 / 框宽 946px"
                    #   仍被判成"远"⇒ 连发 ×5 直行，直接把面板踩过去。
                    #   根因：分档量 `whole`（0.041）在近距**必然**小 —— 面板下沿出了
                    #   画幅，可见面积本来就少；于是"已经站在板上"被读成"还很远"。
                    #   真正的"快到了"签名在**到达门那几个量**上：
                    #     · 紫接近门（≥ 门 − NEAR_MARGIN）⇒ 面板已经压进紫带；
                    #     · 橙低于门（≤ 门）⇒ 面板不再留在远处。
                    #   任一成立 ⇒ **恒走 1 步**（2.652cm，一步一复测）。
                    #   宁可多拍几帧，也绝不再一次跨 13cm 把面板送过去。
                    if sh[1] >= ARRIVE_PURPLE_MIN - NEAR_ARRIVE_MARGIN \
                            or sh[2] <= ARRIVE_ORANGE_MAX:
                        n = 1
                # ---- 前进不力判据（**2026-09-28 用户指定，替代框宽法**）----
                # 规则：若**上一个动作是直行**，取到达判据的三个观测量
                #       v = (紫, 橙, 整帧占比)
                #   算前后两帧的**差值平方和** d² = Σ(v_i − v_i')²；
                #   连续 NO_PROGRESS_PATIENCE 次 d² < NO_PROGRESS_D2_MIN ⇒ 判"不为力"。
                # 为什么废弃框宽：**路线与面板平行时框宽本来就不变**（现场实测
                #   连续两帧 1158→1158px），照样判"不力"，然后一次跨步直接冲出。
                # 为什么三个量一起看：它们合起来才是"面板在我脚下的程度"，
                #   真在走近时必然同时变；单看任何一个都会找到"看不出变化"的视角。
                # ★ 判"不力"之后**严禁任何大批量直行**，一律走
                #   NO_PROGRESS_RECOVER_ACTIONS（后退×2 → 越障 → 后退×1）。
                cov_now = float(sh[0]) if sh is not None else 0.0
                org_now = float(sh[2]) if sh is not None else 0.0
                pur_now = float(sh[1]) if sh is not None else 0.0
                vec_now = (pur_now, org_now, cov_now)
                fwd_now = zone == "forward"
                d2 = None
                if prev_zone == "forward" and fwd_now and sh is not None \
                        and prev_gate_vec is not None:
                    d2 = sum((a - b) ** 2 for a, b in zip(vec_now, prev_gate_vec))
                if fwd_now and sh is not None and d2 is not None \
                        and d2 < NO_PROGRESS_D2_MIN:
                    stall += 1
                    print(f"[行进] 面板{digit} 前进无力迹象 {stall}/"
                          f"{NO_PROGRESS_PATIENCE}：d²={d2:.2e} < "
                          f"{NO_PROGRESS_D2_MIN:.1e}"
                          f"（紫 {prev_gate_vec[0]:.3f}→{pur_now:.3f}、橙 "
                          f"{prev_gate_vec[1]:.3f}→{org_now:.3f}、占比 "
                          f"{prev_gate_vec[2]:.3f}→{cov_now:.3f}）")
                    if stall >= NO_PROGRESS_PATIENCE:
                        print(f"[脱困] 面板{digit} 连续 {stall} 次几乎无变化 → "
                              f"后退×2 → 越障 → 后退×1（**不发大步前进**）")
                        for _act_name, _times in NO_PROGRESS_RECOVER_ACTIONS:
                            self._act(_act_name, _times)
                        if self._attempt_stuck:
                            return False
                        stall, prev_gate_vec = 0, None
                        prev_zone = None
                        prev_cov, prev_org = None, None
                        continue
                else:
                    stall = 0
                prev_gate_vec = vec_now if fwd_now else None
                # 转向批量的登记：本帧这一轮**朝哪个方向**转了几步 ⇒ 下一帧
                # （第一帧可测时）立刻更新**同方向**的 EMA（见上面 ①）
                if zone == "turn":
                    prev_turn, prev_turn_n = yaw_gain, n_turn
                    prev_turn_dir = turn_dir
                    prev_px_of_turn = float(p["px"])   # 下一帧算像素差用
                else:
                    prev_turn, prev_turn_n, prev_turn_dir = None, 0, None
                    prev_px_of_turn = None
                if n_turn > 1:
                    _cn = TURN_BATCH_DIRS[turn_dir][2] if turn_dir else "?"
                    print(f"[转向] 面板{digit} {_cn}转批量 {n_turn} 步"
                          f"（偏角 {yaw_gain:+.1f}°，{_cn}向估计 "
                          f"{getattr(self, TURN_BATCH_DIRS[turn_dir][1]):.1f}°/步）")
                # ★ 2026-09-27：把**到达门六条判据的读数**一起打出来。
                #   为什么（真机复盘）：帧2 的"紫 0.499✓ 橙 0.040✓ 却仍未判到达"
                #   无法从日志回答——`_arrive_pixels_ok` 里够大/形状/不对称三条
                #   原本没有任何读数。这里只**多打一行**；判决仍由上面那次调用
                #   （不带 collect）给出，collect 不参与任何控制流。
                gates = []
                self._arrive_pixels_ok(sh, p["obs"], p["px"], p["w"],
                                       collect=gates)
                print(f"[决策] 面板{digit} 档={ZONE_NAME_CN[zone]} "
                      f"偏角 {yaw_gain:+.1f}°｜横偏 {dx_body * 100:.1f}%画幅"
                      f"｜框宽 {p['box']:.0f}px"
                      + (f"｜紫 {sh[1]:.3f} 橙 {sh[2]:.3f}" if sh else "")
                      + f" → {action_name_cn(act, n)}")
                print(f"[到达门] 面板{digit} 六条判据（全过才判到达）："
                      + "｜".join(gates))
                self._act(act, n if zone != "turn" else n_turn)
                prev_zone = zone
            # 一次尝试的步数用完 = 这一招没磨下来 → 收手，外层换办法重试本格
            self._give_up_attempt(
                f"本次尝试走满 {UNIFIED_MAX_STEPS} 步仍未确认到位")
            return False
        finally:
            if pitch0 is not None:
                self.state.set_pitch(pitch0)
    # =================================================================
    # 到达判决（2026-09-26 短路：**只吃画面量，位姿不参与**）
    # =================================================================
    def _arrive_pixels_ok(self, shares, obs, px, w, collect=None):
        """★ 到达判决（**纯画面量**）：六条全过才 True

        `shares` = `_capture_and_measure` 的 (whole, purple, orange, asym)；
        `obs`    = 目标观测（可为 None）；`px`/`w` = 目标画面列与画幅宽。

        ★ 本方法**不许**碰 `self.pose`、不许调 `_map_pose`、不许拍帧
        （`tests/test_nine_grid_unified_zone.py` 有回归测试钉这条），
        因为"用不可信的位姿否决正确到达"正是 2026-09-26 要短路掉的东西。


        六条判据（全部实测标定过，改阈值前先读 `levels/nine_grid.py` 的常量注释）：

          ① 居中 `ARRIVE_CENTER_MAX`：份额是**整帧**统计量，目标偏在一侧时色块与
             分区边界的交叠关系完全变了 —— 实测"面板在正前方"要到 ≤4cm 才成立，
             而"面板偏在侧面"时判据会**提前成立**（落点实测差整整一格 33cm，
             8 个种子里有 5 个出现 66~71cm 的落点）。站在格心上时面板必然在正前方
             ⇒ 这一条是"压到了"的必要条件，顺带堵掉"横向没对正就宣布到达"。
          ② 够大 `ARRIVE_MIN_TARGET_COVER`：已定位的骗门机制（seed11 d3，站位
             (41.4,82.6)、目标在 66cm 外）：航向 −40° 时目标是**边缘斜视的残缺
             投影**（远侧一半出画幅）⇒ 橙区拿到 0.000（本该 0.48）、紫区 0.637
             ⇒ 门成立。四个整帧份额在残缺投影下全部失真 ⇒ 必须另加尺度判据。
             选型（都实测过，别重复走）：✅ 整帧目标色占比 ≥0.065（真到达
             0.078~0.150、骗门帧 0.042 ⇒ 4 种子**超半格 4→0**、最大落点
             66.9→10.5cm；代价：到达率 96.4%→85.7%）；
             ❌ `hull_area ≥ 300000` 反而更差（超半格 5、78~80cm 落点，因为邻格
             面板被近距离看到时 hull 同样很大）；❌ 收紧 center_max 到
             0.04/0.06/0.08 到达率掉到 10~21/28 且仍有 78~82cm 落点。
          ② 够大 `ARRIVE_MIN_TARGET_COVER`：**已按用户要求短路（0.0）**。留档理由：
             当初为挡"边缘斜视的残缺投影"（远侧一半出画幅 ⇒ 橙 0、紫 0.637 ⇒ 66cm
             处骗门）而加；但实测 33.9cm 几何下它必然卡住（画幅底边只看到 23.5cm 处
             地面 ⇒ 面板中心一进 23.5cm 下沿就出画幅 ⇒ whole 掉到 0.058；而紫/橙
             要过又要求走到 ~15cm ⇒ 两区间不相交）。
          ③ 形状不扁 `ARRIVE_MAX_WIDTH_HEIGHT_RATIO`：**已按用户要求短路（0.0）**。
             留档理由：原 2.15 是 h=56cm 几何下标定的（真到达 1.53~2.07 / 假到达
             2.25~2.49），相机改成实测 33.9cm 后真到达簇挪到 2.83~3.08 ⇒ 恒判 ✗。
          ④⑤⑥ 紫/橙/不对称：阈值与依据见模块常量块 `ARRIVE_PURPLE_MIN` /
              `ARRIVE_ORANGE_MAX` / `ARRIVE_ASYMMETRY_MAX`（含"不对称是角度量、
             不能当距离量用"那条实测）。

        ⚠️ 关掉②③之后，"假到达"的防线只剩紫/橙/不对称/居中四条整帧份额 ⇒
        **落点精度必须用现场几局重新实测**（到达时已打印"据画面估计面板中心距"，
        收尾也可用微动开关当真值）。留档实测（09-25）：只留形状门会退化到 27/28
        但出现 75.5cm 落点；两条一起关的落点分布尚未测。

        collect = 传入一个 list 时，把**六条判据的读数**逐条记进它
        （形如 ["紫 0.319/>=0.35 x", ...]）。**只用于打日志**：判断逻辑一个字
        都没变（每条仍是同一个比较、同一个门槛）。2026-09-27 真机复盘需要它
        ——当时帧2 紫 0.499 OK 橙 0.040 OK 却仍未判到达，而 [决策] 那行只印了
        6 条里的 3 条，后三条（够大/形状/不对称）没有任何读数，只能靠猜。
        """
        ok = True

        def _gate(passed, txt):
            nonlocal ok
            if not passed:
                ok = False
            if collect is not None:
                collect.append(f"{txt} {'✓' if passed else '✗'}")

        if shares is None:
            if collect is not None:
                collect.append("份额不可用 ✗")
            return False
        whole, pur, org, asym = shares
        _gate(pur >= ARRIVE_PURPLE_MIN, f"紫 {pur:.3f}/≥{ARRIVE_PURPLE_MIN}")
        _gate(org <= ARRIVE_ORANGE_MAX, f"橙 {org:.3f}/≤{ARRIVE_ORANGE_MAX}")
        _gate(abs(asym) <= ARRIVE_ASYMMETRY_MAX,
              f"不对称 {asym:+.3f}/≤{ARRIVE_ASYMMETRY_MAX}")
        if ARRIVE_CENTER_MAX > 0.0 and obs is not None:
            dx = abs(float(px) - self._center_column_px(w)) / float(w)
            _gate(dx <= ARRIVE_CENTER_MAX, f"居中 {dx:.3f}/≤{ARRIVE_CENTER_MAX}")
        elif ARRIVE_CENTER_MAX > 0.0:
            if collect is not None:
                collect.append("居中 无观测(跳过)")
        if ARRIVE_MIN_TARGET_COVER > 0.0:
            _gate(float(whole) >= ARRIVE_MIN_TARGET_COVER,
                  f"够大 {whole:.3f}/≥{ARRIVE_MIN_TARGET_COVER}")
        elif collect is not None:
            # 2026-09-28 用户要求短路：不再参与判决，但**读数保留在日志里**
            # （现场还要靠它观察"面板被画幅切掉多少"）
            collect.append(f"够大 {whole:.3f}（已短路）")
        if ARRIVE_MAX_WIDTH_HEIGHT_RATIO > 0.0 and obs is not None:
            bw = float(obs.bbox[2])
            bh = float(obs.bbox[3])
            if bh > 1.0:
                _gate(bw / bh <= ARRIVE_MAX_WIDTH_HEIGHT_RATIO,
                      f"宽/高 {bw / bh:.2f}/≤{ARRIVE_MAX_WIDTH_HEIGHT_RATIO}")
            elif collect is not None:
                collect.append("宽/高 退化(高≤1px) ✗")
            if bh <= 1.0:
                ok = False
        elif ARRIVE_MAX_WIDTH_HEIGHT_RATIO > 0.0:
            if collect is not None:
                collect.append("宽/高 无观测(跳过)")
        elif obs is not None and float(obs.bbox[3]) > 1.0 and collect is not None:
            # 2026-09-28 用户要求短路：不参与判决，读数照样打（现场靠它看被切了多少）
            collect.append("宽/高 %.2f（已短路）"
                           % (float(obs.bbox[2]) / float(obs.bbox[3])))
        return ok

    def _arrive_probe_note(self, digit, frame):
        """到达时的锚测量：**纯诊断，绝不改变控制流**（`ARRIVE_ON_CELL_PROBE`）

        只在探针说 "no"（画面说到了、地图说不在）时打一行，供后续用真值统计
        "这道核验本来会否决几次"。`yes`/`unknown` 不打（避免噪音，也避免把"弃权"
        当结论写进留档）。`_reanchor_pose` 随后会自己再跑一次锚并采纳实测位姿，
        与本探针无关。
        """
        if not ARRIVE_ON_CELL_PROBE:
            return None
        try:
            verdict, why = self._verify_target_cell(frame, digit)
        except Exception as e:                      # 探针不许影响判决
            print(f"[核验] 面板{digit} 探针异常（{type(e).__name__}: {e}）"
                  "——仅记录，不影响到达判决")
            return None
        if verdict == "no":
            print(f"[核验] 面板{digit} 画面判到达，但地图锚认为不在目标格"
                  f"（{why}）——仅记录，**不否决**")
        return verdict

    def _verify_target_cell(self, frame, digit):
        """格的同一性核验（用户方案 ②-a）：实测位姿是否真的落在目标格上？

        ⚠️ **本函数不参与到达判决**（2026-09-26 短路）：唯一调用点是
        `_arrive_probe_note` 这个诊断探针。曾经它是到达门的最后一道"与门"，
        实测会拿不可信的位姿否决**正确**到达，已摘掉否决权（见模块文件头、
        常量块 `ARRIVE_ON_CELL_PROBE`）。要恢复否决必须回滚那次提交。

        返回 (verdict, reason)：
          "yes" = 锚解出的**实测位姿**距目标格心 ≤ ARRIVE_ON_CELL_TOL_CM
          "no"  = 解出来了、但明显不在目标格（"站在别格"的假到达签名）
          "unknown" = 锚不可用（诚实弃权；旧开关 `ARRIVE_ON_CELL_REQUIRED` 已废弃）

        为什么当初要靠它（而不是继续加份额/形状阈值）：到达门那几个量都是**单帧
        整帧统计**，在"站在别的格子上、看到一块完整且居中的面板"这种视角下**全部
        正常**（实测有一帧各项合规而真值离目标 100cm）。只有"这一帧的几何能不能被
        地图解释"才能分辨——这正是用户方案第 2 条第一项要求的核验。

        实现复用 `_map_pose`：它自带共识门与诚实弃权，所以"解出来了"本身就是
        可信的证据。
        """
        try:
            full = self.detector.detect_panels(frame, drop_border=False)
        except Exception:
            return "unknown", "全色检测异常"
        anc = self._map_pose(full, frame, why=f"ONCELL{digit}")
        if anc is None:
            return "unknown", "锚不可用（共识不足/对应点不够）"
        cell = self.digit_cell.get(digit)
        if cell is None:
            return "unknown", "无该数字的格位"
        try:
            pos = np.asarray(anc["pose"], float)[:2]
            d = float(np.linalg.norm(pos - np.asarray(
                grid_cell_center(cell), float)))
        except Exception:
            return "unknown", "锚结果无法解读"
        if d <= ARRIVE_ON_CELL_TOL_CM:
            return "yes", f"实测位姿距目标格心 {d:.1f}cm"
        return "no", (f"实测位姿距目标格心 {d:.1f}cm"
                      f"（>{ARRIVE_ON_CELL_TOL_CM:.0f}cm）"
                      f"——目标面板不在脚下，疑为邻格同色面板")
    def _find_target_again(self, digit, lost, turns, p):
        """目标丢失后的**重捕获** → 返回 (turns, deg_turned)；返回 <0 表示本次尝试收手

        纯视觉、逐步转、一次尝试内仍有上限：
          ① 丢的第 1 帧先只动**头部**扫四档（零身体位移、不烧转向预算、无翻车风险）；
          ② 之后每丢 `TARGET_LOST_TOLERATE` 帧 ⇒ 朝上次看到它的一侧转**一小步**
             （一次一个原语），每一步都拍照复测，转过冲由下一步反向吸收；
          ③ 累计转过 `FIND_MAX_TURN_DEG` 或拍够 `FIND_MAX_FRAMES` 仍不见
             ⇒ **本次尝试收手**（不是放弃本格！外层会换搜索方向重试同一格，
             顺序计分下"这格不要了"等于把后面全丢掉）。
        """
        if lost == 1:
            for h in (HEAD_WIDE_LEFT, HEAD_LEFT, HEAD_RIGHT, HEAD_WIDE_RIGHT,
                      HEAD_CENTER):
                if self._attempt_stuck:
                    return -1, turns
                self.state.set_head(h)
                q = self._capture_and_measure(self._target_color(digit))
                if q is None:
                    return -1, turns
                if q["obs"] is not None:
                    print(f"[重捕获] 面板{digit} 头部档 {h} 扫到"
                          f"（画面列 {q['px']:.0f}px）→ 复测接手")
                    self.state.set_head(HEAD_CENTER)
                    return 0, 0.0
            self.state.set_head(HEAD_CENTER)
            print(f"[重捕获] 面板{digit} 头部四档未扫到 → 按地图方位逐步转过去")
            return turns, 0.0
        if lost % FIND_STRIDE != 0:
            return turns, 0.0
        # ★ 转向方向**一次定死、中途不改**（见上：反复翻向会退化成原地摆动）
        # ★★ 2026-09-25 实测更正：**不再用地图方位决定首转方向**。
        #   理由（seed21 逐格实测死推航向误差 pose_θ − true_heading）：
        #       d1 −0.0° → d2 −1.4° → d3 **−7.7°** → d4 −38.7° → d5 −49.2°
        #       → d6 −70.0° → d7 **−105.7°**
        #   航向误差**逐格累积且不修**（`_reanchor_pose` 只锚位置、航向保持死推值），
        #   到 d3（首次失败）已 7.7°、d7 已 105.7° ⇒ 地图方位**方向都会指反**。
        #   实测证据：digit3 前 26 帧**全是 `turn_left_small_step`**（整格转 107 次 /
        #   走 15 次）—— 朝错方向连转 140°+，而相机横向视场本身就有 ±46°
        #   ⇒ 朝对方向时最多转 (360°−92°)=268°、朝错时也只需多转 92°，
        #   **根本不该出现"反复转"**。
        #   ⇒ 方向改为：① 有"最后看到它的一侧"就用它；② 否则固定**左转**单向扫；
        #      ③ 扫满约一圈仍未见到才反向（见 `_reacq_flipped`）。
        #   这样搜索**完全不依赖 pose 航向**，"异常的多次旋转"从根上消失。
        if self._reacq_dir is None:
            side = getattr(self, "_last_seen_side", None)
            if side is not None:
                self._reacq_dir = 1.0 if float(side) >= 0.0 else -1.0
            else:
                self._reacq_dir = 1.0        # 无先验：固定左转单向扫
        want_left = self._reacq_dir > 0.0
        step = TURN_LEFT_SMALL_DEG if want_left else -TURN_RIGHT_SMALL_DEG
        act = "turn_left_small_step" if want_left else "turn_right_small_step"
        # ★ 2026-09-28 用户要求：**找不到目标的小转一次发 3 步**（原来一次 1 步，
        #   每步都要拍一帧复测 ⇒ 目标在身后时要磨十几轮）。
        #   实际转角 = 3 × (左 8.625° / 右 5.200°) ≈ 15.5~25.9°，
        #   比相机横向视场（±33.7° 半视场）小得多，不会跳过视野。
        turns += 1
        print(f"[重捕获] 面板{digit} 连续 {lost} 帧未见 → "
              f"{action_name_cn(act, SEARCH_TURN_STEPS)}"
              f"（第 {turns} 次，累计{action_name_cn(act, SEARCH_TURN_STEPS)}"
              f"≈{self._reacq_deg:.0f}°）")
        self._act(act, SEARCH_TURN_STEPS)
        return turns, step * SEARCH_TURN_STEPS
    def _zone_masks(self, w, h):
        """把示意图四块区域栅格化成布尔掩膜（按画幅尺寸缓存，只算一次）

        几何公式与判档/分割线共用模块级的 `zone_masks(w, h)`（那里是唯一实现）；
        本方法只负责**缓存**：同一画幅尺寸下每帧复用同一批布尔数组（2592×1944
        的掩膜不小，单格几百帧都按同一个尺寸算，省下来的是实打实的拍照间隔）。
        """
        key = (int(w), int(h))
        cache = getattr(self, "_region_cache", None)
        if cache is not None and cache[0] == key:
            return cache[1]
        out = zone_masks(key[0], key[1])
        self._region_cache = (key, out)
        return out
    def _zone_at_body_pixel(self, px, py, w, h, head_pulse=None, prev=None):
        """按**机体系**判档：(px, py) → ("forward"|"side"|"turn", dx_frac)

        与 `_zone_at_pixel`（画幅中线版）的唯一区别：中线取 `_center_column_px()`，
        即把**头部角**折算掉。用户示意图的三块区域是按"机体正前方"画的，
        而偏头拍摄时"画面正中"不是"机体正前方"。

        注：当前投影模型里**没有**相机-机体**方位**安装参数（见 `_body_angle_deg`），
        故中线除头部角外没有别的平移项。
        """
        px_body = float(px) - (self._center_column_px(w, head_pulse) - w / 2.0)
        zone = self._zone_at_pixel(px_body, py, w, h, prev)
        dx = abs(px_body - w / 2.0) / w if w > 0 else 1.0
        return zone, dx
    def _action_for_zone(self, zone, px_body, w):
        """档位 → 动作名（**一次决策只下发一个原语**）

        绿 = 直行；蓝 = 横移（目标在左→左移）；橙 = 旋转（小转）。
        动作名一律是"单步"原语：批量由调用方的循环次数体现，不由 `times` 参数。
        """
        if zone == "forward":
            return "go_forward_one_step"
        if zone == "side":
            return "left_move" if px_body < w / 2.0 else "right_move"
        return ("turn_left_small_step" if px_body < w / 2.0
                else "turn_right_small_step")
    def _look_up_and_check(self, digit, color):
        """抬头档看一眼目标在不在视野里（用户 2026-09-28 要求）

        抬头档（`PITCH_NAV`）视野远，目标面板在中远处更容易被看到；看到就报一行，
        主循环随后仍用低头档接近（低头档才是"能压到格心"的那一档）。
        **纯信息**：不改任何决策状态，找不到也不额外转向（交给主循环的重捕获）。
        """
        try:
            self.state.set_pitch(SEARCH_LOOK_UP_PITCH)
            frame = self._capture()
            if frame is None:
                return False
            obs = self.detector.detect_panels(
                frame, colors=[color], arbitrate=True, drop_border=False)
            if not obs:
                print(f"[抬头] 面板{digit} 抬头档（pitch {SEARCH_LOOK_UP_PITCH}）"
                      f"没看到 → 回低头档正常流程")
                return False
            o = max(obs, key=lambda x: x.hull_area)
            px = (o.hull_centroid_px if o.clipped else o.center_px)[0]
            print(f"[抬头] 面板{digit} 抬头档看到（画面列 {px:.0f}px，"
                  f"框宽 {o.bbox[2]:.0f}px）→ 目标在视野内，进入主循环（低头档）"
                  f"朝它走")
            self._target_seen = True
            return True
        except Exception as e:
            print(f"[抬头] 探测异常（{type(e).__name__}: {e}）——忽略，回正常流程")
            return False

    def _capture_and_measure(self, color, pitch=None):
        """★ 一次拍照 → 三个观测量（统一决策的**唯一**感知入口）

        返回 dict 或 None（帧拿不到/预算耗尽）：
          obs     : 目标色最佳观测（`clipped` 时取凸包质心）或 None
          px, py  : 观测在画面里的位置（原生 px）
          w, h    : 画幅尺寸
          shares  : (whole, purple, orange, asym) 四区份额
          box     : 目标框宽（裁切时用 sqrt(hull_area) 兜底）
          frame   : 帧本身（调试/可视化用）

        为什么"一次拍照取三个量"：判档要**目标位置**，到达要**整帧色占比**——
        分开拍两次不仅慢一倍，而且两个量来自不同时刻，判档与判据会打架。
        """
        if self._attempt_stuck:
            return None
        self._count_frame()
        self._cell_frames += 1
        self._cell_progress()
        frame = self.state.capture_frame()
        if frame is None:
            return None
        sh = self._zone_shares(frame, color)
        obs_l = self.detector.detect_panels(
            frame, colors=[color], arbitrate=False, drop_border=False)
        o = pick_same_color(obs_l)
        out = {"obs": o, "frame": frame, "shares": sh,
               "w": float(frame.shape[1]), "h": float(frame.shape[0]),
               "px": None, "py": None, "box": 0.0}
        if o is not None:
            p = o.hull_centroid_px if o.clipped else o.center_px
            out["px"], out["py"] = float(p[0]), float(p[1])
            box = float(o.bbox[2])
            if o.clipped:
                box = max(box, float(np.sqrt(max(o.hull_area, 1.0))))
            out["box"] = box
        return out
    def _zone_shares(self, frame, color):
        """整帧色占比 + 分区份额 → (whole, 紫, 橙, 不对称)

        与 `NineGridDetector.color_ratio` **同一条光照归一化链路**（否则阈值不可
        比），但**一次掩膜**同时算出四块区域的份额，不重复归一化。
        「份额」= 该区域内的目标色像素 ÷ **全帧**目标色像素总数
        （所以紫+橙+绿+蓝 = 1；这也是为什么紫门 0.35 与橙门 0.10 可以并列）。
        拿不到帧/掩膜时返回 None。
        """
        if frame is None or frame.size == 0:
            return None
        try:
            ww = int(self.detector.work_width)
            scale = frame.shape[1] / float(ww)
            work = cv2.resize(frame, (ww, int(round(frame.shape[0] / scale))))
            work, self.detector.last_norm = normalize_illumination(work)
            hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
            mask = build_color_mask(hsv, color)
        except Exception:
            return None
        tot = int(cv2.countNonZero(mask))
        if tot <= 0:
            return 0.0, 0.0, 0.0, 0.0
        m = mask.astype(bool)
        regs = self._zone_masks(mask.shape[1], mask.shape[0])
        n = {k: int(m[v].sum()) for k, v in regs.items()}
        return (tot / float(mask.size), n["purple"] / tot, n["orange"] / tot,
                (n["blueL"] - n["blueR"]) / tot)
    def _zone_at_pixel(self, px, py, w, h, prev=None):
        """★ 按示意图判档（画幅中线版）——薄包装，实现与阈值见模块级 `zone_at_pixel`

        保留这个方法名是因为老调用点、测试与 `tools/ab_ninegrid.py` 的探针都挂在
        它上面（探针按类属性包一层计数）；**判据本身只有一份**，就在上面的纯函数里
        （那里也写着迟滞、边界形状与"为什么按像素不按角度"）。
        函数体读本模块全局常量 ⇒ `tools/record_run.py --tune` 的运行时改参照旧生效。
        """
        return zone_at_pixel(px, py, w, h, prev)
    def _reanchor_pose(self, digit):
        """到达后重置位姿：**优先用地图锚的实测位姿**，退而用格心 + 推算航向

        统一决策循环到达时用**视觉 + 地图**反推自身位姿（`_map_pose`：把"地图里
        已知格心的面板"与"画面里看到的色块"配对，共识后解出场地系位姿），成功就
        整体采纳（**位置与航向都换成实测值**）；失败或判为误解则退回格心 + 原航向。

        ⚠️ 方向性：这条改的是**位姿来源**（推算 → 视觉反推），不是"用推算位姿导航"
        ——后者仍是禁止的。

        总开关 `layout_enabled() == False`（见 nine_grid_shared.LAYOUT_ENABLED）
        时**整段短路**：地图锚按定义要拿"数字→格"的已知格心去配观测点，digit_cell
        为空时它连一个对应点都凑不出（下面那句 early return 本来就会命中），
        再跑一次 detect_panels 纯属浪费——到达帧的拍照预算留给视觉判据本身。
        """
        if not layout_enabled():
            # ★ 2026-09-28：布局总开关**默认关闭**（用户决定：不扫了、地图相关的
            #   都不要了）⇒ 本方法在默认路径上**永不执行**：地图锚按定义要拿
            #   "数字→格"的已知格心去配观测点，digit_cell 为空时连一个对应点都
            #   凑不出，再跑一次 detect_panels 纯属浪费。
            #   位姿维持死推值 —— 统一决策的选方向/到达判决都不读 self.pose
            #   （见本文件文件头）；**离场护栏读它，仍然生效**。
            print("[锚定] 已短路（布局总开关关闭）：跳过地图锚，位姿保持死推值")
            return
        if digit not in self.digit_cell:
            return
        resid = self._cell_arrive_resid_cm
        c = grid_cell_center(self.digit_cell[digit])
        anc = None
        frame = self._capture()
        if frame is not None:
            try:
                full = self.detector.detect_panels(frame, drop_border=False)
                anc = self._map_pose(full, frame, why=f"REANCHOR{digit}")
            except Exception:
                anc = None
        if anc is not None and REANCHOR_MAX_POSE_JUMP_CM > 0.0:
            meas = np.asarray(anc["pose"], float)
            jump = float(np.linalg.norm(meas[:2] - np.asarray(c, float)))
            if jump > REANCHOR_MAX_POSE_JUMP_CM:
                print(f"[锚定] 地图锚解出的位置距格心 {jump:.0f}cm "
                      f"（>{REANCHOR_MAX_POSE_JUMP_CM:.0f}cm）→ 判为误解，弃用")
                anc = None
        if anc is not None:
            self.pose = np.asarray(anc["pose"], float).copy()
            print(f"[锚定] **视觉+地图反推位姿**：位置 "
                  f"({self.pose[0]:.1f},{self.pose[1]:.1f})、航向 "
                  f"{np.degrees(self.pose[2]):.1f}°"
                  f"（内点 {anc['n_inliers']}，反解相机高 "
                  f"{anc.get('cam_height_cm', float('nan')):.0f}cm）"
                  f"｜对照格心 {c[0]:.0f},{c[1]:.0f}")
            return
        # ★ 2026-09-28：**死推被扬弃** ⇒ 原来这里"地图锚不可用 → 退回'格心+原航向'"
        #   的那次写位姿**已删除**：那是拿"以为的格心"当位置写回 `self.pose`，
        #   等于让死推继续冒充定位。现在锚不可用就什么都不做（`self.pose` 不再
        #   被写、也不再被推进），并把"测不到"如实打出来。
        print(f"[锚定] 地图锚不可用 → **不写位姿**（死推已扬弃）"
              f"（参考格心 {c[0]:.0f},{c[1]:.0f}"
              f"{'' if resid is None else f'，上次记录的距格心 {resid:.1f}cm'}）")


def run_level(state):
    """数字宫格关卡入口（统一决策）：python main.py nine_grid"""
    print("[提醒] 本办法到达前不做蹭步，微动开关可能不触发；"
          "上现场请先用 python main.py nine_grid_three_stage")
    level = NineGridLevel(state)
    return level.run_level()
