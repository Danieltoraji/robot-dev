# -*- coding: utf-8 -*-
"""
数字宫格关卡（levels/nine_grid.py）

任务：1m×1m 九宫格（3x3，左下位置6恒空），7 块数字面板(1~7)随机布局，
机器人按 1→7 顺序依次到达各面板中心（踩中微动开关），限时 15 分钟。
评分：顺序正确每格 10 分（70），限时完成 +30。

方案要点（2026-09-08 定稿；2026-09-11 导航改 v3 视觉伺服，详见
docs/关卡算法/数字宫格攻略.md §3.2b）：
  视觉  颜色主判（HSV 七色，1红..7粉）+ HOG+SVM 黑色数字仲裁（只否决）；
        面板中心：未裁切取四边形对角线交点（透视不变量）；被画幅裁切时
        对角线交点失效（实测偏差 150~800px），改用颜色掩膜凸包面积质心。
  导航  **v3 视觉伺服（默认）**：地板会形变 → 机体俯仰/相机高度大幅变化
        （按 ±15° 考虑，可见地面带 4 倍摆动），因此导航主链只用相对量：
        搜索（头部五档 + 按死推提示转向 / 盲走）→ 对准（像素 yaw，小转角
        闭环、死区 3.5°）→ 接近（目标框宽 ≥920px ≈ 进 35cm）→ 低头到达
        （颜色占比**峰值跟踪 + 分段前压**：跌破峰值 55% 即判压过该格）→
        蹭步确认 → 死推位姿锚到格心。
        地图（布局扫的 digit_cell + 动作模型死推位姿）只作"往哪转/盲走多远"
        的粗提示（只用来决定"往哪边找"，不产生任何到位结论）。
  定位（布局扫自标定）  布局扫同时自标定相机常数（安装下俯偏移 + 离地高度），
        并用地图像核验在**到达时**反推实测位姿（位置与航向一起换）。
        **不依赖冻结的相机常数**，因此地板形变不影响它。
  运动  小幅度动作白名单（单步 2cm 前进、~2cm 横移、3.2cm 后退）；远距
        批量上限 BATCH_MAX_STEPS=5（10cm）。本场地禁用
        go_forward / go_forward_fast（5cm 步幅在小面板+打滑地板上易摔倒，
        `_act` 硬门拒绝）。转向用「在线 EMA 估计小转实际角（误差源 = 像素
        yaw 或地图方位），连续 2 批 <0.5°/次→升级大转」+ 大转防振荡。
  容错  卡滞检测（框宽不涨/定位不改善 → 后退脱困）、目标丢失（头部扫找回
        → 回退半步 → 重搜）、分格时间预算 + **单格硬熔断**（CELL_LIMIT_*：
        时间/拍照/动作三重护栏，任一触发即"本格记未确认、立即返回"）
        + 全局看门狗 + 预算不足自动收缩低头段迭代、到达确认（颜色出现→
        消失主判 + 蹭步压微动开关）。绝不跳格：跳格断 70 分计分链（ESP32
        顺序错只日志不断链，借道穿越其他面板计分安全）。

到达确认为什么不改回"低头色占比绝对阈值"：参考实现的每色阈值（0.002~0.17）
在本机俯角下判不到（实测峰值仅 0.13）——v3 用**尺度无关**的"峰值跟踪 + 相对
跌落"，与颜色/光照/俯仰无关。历史结论"低头色占比不可用"是旧相机模型
（39cm/41.4°，可见地面带 15.8cm 起）下的结论，现在几何已修正。

布局扫描 v2（2026-09-11，无站位依赖）：机器人从前一关卡进入，初始位姿
未知不可控，因此布局扫不用任何场地系先验——每帧观测经"机器人系单应"
（from_pose((0,0), 高度, 俯仰, head_pulse=头部档)；头部偏航已并入相机朝向，
故宽扫档 ±40.5°/±63° 的观测不会被当成中位档映射——见
core.ground_homography.from_pose 的 head_in_pose）转为机器人系地面坐标，
跨帧按 digit 聚合（**未裁切观测优先**，同数字跨帧离散 >SPREAD_MAX_CM 剔除），
再在 (安装偏移, 相机高度) 参数网格上做格阵拟合（assign_grid_cells：假设-共识找
格阵基 + **4 个真旋转**硬约束（手性论证见模块内注释）+ 刚体残差门 + 覆盖率门
+ 入口侧/列序谓词 + 格6恒空优先）。胜出布局即数字→格，胜出组合中位数即自标定
常数；随后用"机器人系点 ↔ 场地格心"2D 刚体拟合**直接解出位姿**（不再解单应
再 decompose），GN 定位从此自续。格阵歧义/缺数字/自举自检不过时自动前进一步
重扫（平移视差消歧、带进近排），最多 3 轮。扫描终止判据 = 未裁切观测覆盖
≥LAYOUT_SCAN_MIN_CLEAN_DIGITS 个数字**且两档俯仰都已有观测**（只扫一档会让
"安装偏移 vs 名义俯仰角"退化，实测 (1200,25°) ≡ (1040,10°)）。

现场依赖：
  1) 无标定依赖（布局扫 v2 已去除；archive/result/ninegrid_homography.json
     与 tools/calib_ninegrid.py 保留为诊断工具）；
  2) COLOR_THRESHOLDS 现场复标（tools/debug_vision.py ninegrid 子命令）；
  3) SVM 仲裁依赖 scikit-learn/skimage/joblib（缺失自动降级纯颜色）。

"""

import time
import numpy as np
import cv2

from levels.nine_grid_shared import (
    CAMERA_FOV_H_DEG,
    NineGridShared,
    PITCH_DOWN,
    action_name_cn,
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
)


# =====================================================================
# 运动原语的**现场实测值**（runbook §2 测量 1；2026-09-25 P1 落地）
# =====================================================================
# 为什么必须单独列出来：`ACTION_MODEL` 里的小转此前写的是**名义 ±2.0°/步**，
# 而实测是左 **8.625°** / 右 **5.200°**（差 330%/160%）。旧代码拿 `ACTION_MODEL`
# 去规划步数 ⇒ `n = floor(|yaw| / 2.0)` 会把 16.5° 的偏差算成 6 步，真机一次转
# **51.75°**（转过头 3 倍）→ 丢目标 → 重搜。
#
# ★ 使用纪律（《需求规格-2026-09-25-统一决策重写》§7.7）：
#   这两个常量**只允许**两个用途：(a) 下发动作；(b) 遥测/测试里的期望值。
#   **不许进任何判据**（不许再做 `n = f(角度/步长)` 的除法）。统一决策一次决策
#   只下发一个原语，因此不需要"每步多少度"这种规划常数。
TURN_LEFT_SMALL_DEG = 8.625     # turn_left_small_step（20 步 ×2 组）
TURN_RIGHT_SMALL_DEG = 5.200    # turn_right_small_step（20 步 ×2 组）


# 档位 → 中文（日志用；与用户示意图的绿/蓝/橙一一对应）
_ZONE_NAME_CN = {"forward": "绿·直行", "side": "蓝·横移", "turn": "橙·旋转"}
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
ZONE_PURPLE_HEIGHT = 0.62
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
ARRIVE_ORANGE_MAX = 0.10      # 橙区份额上限（色块不该还留在远处）
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
#   为什么必须有（P1 实测定位到的假到达机制）：目标**边缘斜视的残缺投影**会让
#   四个整帧份额全部失真 —— 远侧一半出画幅 ⇒ 橙区拿 0（本该 0.48）、紫区拿 0.637
#   ⇒ 门在离目标 **66cm** 处成立。实测两侧取值：
#       真到达：整帧占比 0.078~0.150（低头 1040，距格心 ≤10cm）
#       骗门帧：0.042
#   取 0.065（两侧都留余量）。设 0 关闭该判据（仅用于对照实验，不许当默认）。
ARRIVE_MIN_TARGET_COVER = 0.065
#   ⚠️ 2026-09-25 实测更正：加了下面的形状门之后，本条**已不再是主要约束** ——
#   把 0.065 放到 0.055/0.045/0.020 三档，结果**一个数都不差**（24/28、拍照 2386、
#   中位 8.2、最大 10.5、超半格 0）；而**只留形状门**会退化到 27/28 但**超半格 2**
#   （出现 75.5cm 落点）。⇒ 两条门是**互补**的（占比管"够不够大"、形状管"是不是
#   斜切片"），保留两条最稳；本条的具体阈值不敏感。
# 形状门（**实际有效的那条**）：目标观测 bbox 的**宽/高比上限**。
#   标定工具 `tools/_tmp_arrive_gate_probe.py`（把门全放开、逐帧记特征、用真值打标签）：
#       真到达（11 帧）宽/高 **1.53 ~ 2.07**（中位 1.66）
#       假到达（ 5 帧）宽/高 **2.25 ~ 2.49**   ← 全是又扁又斜的残缺投影
#   ⇒ 取 **2.15**。设 0 关闭。
ARRIVE_MAX_WIDTH_HEIGHT_RATIO = 2.15
#   依据：正常到达时实测位置应在格心附近（半格 16.7cm 内）；40cm 已超过一整格，
#   说明 RANSAC 把观测配错了格 ⇒ 宁可退回旧行为，也不采纳一个离谱的位姿。
ARRIVE_ON_CELL_TOL_CM = 16.7     # 距格心容差（= 半格；压感区半宽 5.5cm，
#                                     但锚本身有位姿误差，取半格更稳）
ARRIVE_ON_CELL_REQUIRED = False   # 锚**不可用**时是否也拒绝
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
# 统一决策把它们合成一个循环 ⇒ 上限必须 ≥ 这个量级，否则会在"还没走到"时就用尽，
# 外层 RETRY 只能拿 back_one_step 兜（第一版 60 时就踩了这个坑）。
UNIFIED_MAX_STEPS = 150
# ★ 重捕获（P1）：目标不见了怎么办 —— 纯视觉、逐步转、有上限
#   为什么需要它：统一决策**不用死推提示**（用户明确放弃），而"上一格跑完时目标
#   经常在 180° 身后、距离可达 2 格"（实测面板3 距上一格 79cm）。若只做几小步
#   自转就放弃，整局会在第 2 格之后连续丢格。
#   机理：**地图方位 → 逐步转过去 → 每一步都拍照复测**。方位只用来决定"往哪边转"，
#   转多少完全由视觉闭环决定（转多了下一步反向吸收）——这不违反"不用死推"：
#   `self.pose` 是布局扫/到达锚定的结果，且**只用于选方向**，不产生任何"到位"结论。
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
#     而单格 140 帧的硬熔断从未触发 ⇒ 不是预算不够，是这个节流把它卡死了）。
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
FORWARD_FAR_STEPS = 5            # 远距一次前进步数（5×2.652 ≈ 13cm）
FORWARD_MID_STEPS = 3            # 中距一次前进步数（3×2.652 ≈ 8cm）
# ---- 前进无效脱困（用色占比判，不用框宽）----
# 现场/仿真共识：地板打滑或顶住时"前进一步"不产生位移。判据必须用**单调量**，
# 框宽在近距裁切后会反向塌陷，只能产生误报（实测把一格推过格心 28cm）。
NO_PROGRESS_COVER = 0.004            # 占比增量低于此值视为"这一步没走近"
NO_PROGRESS_PATIENCE = 2           # 连续几次没走近才后退脱困（单次抖动不误判）

# =====================================================================
# 统一决策：三档分区 + 一个循环走完全程
# =====================================================================


class NineGridLevel(NineGridShared):
    """数字宫格（统一决策）：看目标落在画面哪一档，一个循环走到目标

    每帧一次拍照、一次判档、一个动作；不看历史峰值、不用推算位姿做提示、
    不用 yaw 的数值做规划。共用机制见 levels/nine_grid_shared.py。
    """
    def go_to_panel(self, digit):
        """前往第 digit 块面板（统一决策）：返回是否确认到位

        逐格落点在返回前留档（诊断与回归用，不参与控制）——必须在返回前记，
        因为调用方随后就会走向下一格。
        """
        try:
            return self._drive_to_panel(digit)
        finally:
            self._record_landing(digit)
    def _drive_to_panel(self, digit):
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
        """
        t_end = self._begin_cell_budget(digit)
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
        lost = 0
        turns = 0
        self._reacq_deg = 0.0          # 本轮重捕获已累计转过的角度
        self._reacq_frames = 0
        self._reacq_dir = None         # 本轮重捕获的转向方向（一次定死）
        prev_zone = None
        prev_cov = None
        prev_org = None
        stall = 0
        try:
            for _ in range(UNIFIED_MAX_STEPS):
                if time.time() > t_end or self._cell_expired():
                    return False
                p = self._capture_and_measure(color)
                if p is None:
                    return False
                self._reacq_frames += 1
                sh = p["shares"]
                # ---- 到达证据（只看当前帧）----
                # ★ 前置条件：**目标必须在画面中央附近**（`ARRIVE_CENTER_MAX`）。
                # 为什么必须有：四个份额是**整帧**统计量，目标偏在一侧时色块与
                # 分区边界的交叠关系完全变了 —— 实测"面板在正前方"要到 ≤4cm 才成立，
                # 而"面板偏在侧面"时判据会**提前成立**（落点实测差整整一格 33cm，
                # 8 个种子里有 5 个出现 66~71cm 的落点）。
                # 站在格心上时面板必然在正前方 ⇒ 这一条是"压到了"的必要条件，
                # 顺带把"横向没对正就宣布到达"也堵掉。
                if sh is not None:
                    _whole, pur, org, asym = sh
                    _centered = True
                    if p["obs"] is not None and ARRIVE_CENTER_MAX > 0.0:
                        _dx = abs(p["px"] - self._center_column_px(p["w"])) / p["w"]
                        _centered = _dx <= ARRIVE_CENTER_MAX
                    # ★ 面板必须**够大**（压在脚下）——"假到达"的直接解药，已实测。
                    # 已定位的骗门机制（seed11 d3，站位 (41.4,82.6)、目标在 66cm 外）：
                    # 航向 −40° 时目标是**边缘斜视的残缺投影**（远侧一半出画幅）
                    # ⇒ 橙区拿到 **0.000**（本该 0.48）、紫区拿到 0.637 ⇒ 门成立。
                    # 四个整帧份额在"残缺投影"下全部失真 ⇒ 必须另加**尺度**判据。
                    #
                    # 判据选型（都实测过，别重复走）：
                    #   ✅ **整帧目标色占比 ≥ 0.065**（本行）：真到达 0.078~0.150、
                    #      骗门帧 0.042 ⇒ 4 种子实测**超半格 4→0**、最大落点
                    #      66.9→10.5cm。代价：到达率 96.4%→85.7%、拍照 466→609/局
                    #      （因为"站得稍远一点"的正常到达也会被这条压住）。
                    #   ❌ **目标 hull_area ≥ 300000**：看着更干净（真到达 418288 vs
                    #      骗门帧 210950，差 2 倍），但实测**反而更差** ——
                    #      超半格 5、出现 78~80cm 落点。原因：同一色面板在**邻格**
                    #      被近距离看到时 hull 同样很大，面积判不出"是不是脚下这一块"。
                    #   ❌ **收紧 center_max 到 0.04/0.06/0.08**：到达率掉到 10~21/28
                    #      且**仍有** 78~82cm 落点 ⇒ 方向错。
                    _big = (_whole >= ARRIVE_MIN_TARGET_COVER
                            if ARRIVE_MIN_TARGET_COVER > 0.0 else True)
                    # 形状门：目标观测 bbox 的**宽/高比上限** —— 专治"斜视/残缺投影"。
                    # ⚠️ 方向与上一版**相反**（上一版用高/宽比下限 0.46，只挡掉 5 帧里的
                    # 2 帧；用真值标定后发现正确方向是"宽/高比不能太大"）。
                    # 标定（tools/_tmp_arrive_gate_probe.py，11 帧真到达 / 5 帧假到达）：
                    #     真到达 宽/高 **1.53 ~ 2.07**（中位 1.66）
                    #     假到达 宽/高 **2.25 ~ 2.49**（全是又扁又斜的残缺投影）
                    #   ⇒ 取 **2.15**（两簇之间，两侧各留 ~0.08 余量）。
                    if ARRIVE_MAX_WIDTH_HEIGHT_RATIO > 0.0 and p["obs"] is not None:
                        _bw = float(p["obs"].bbox[2])
                        _bh = float(p["obs"].bbox[3])
                        if _bh > 1.0 and _bw / _bh > ARRIVE_MAX_WIDTH_HEIGHT_RATIO:
                            _big = False
                    if _centered and _big and pur >= ARRIVE_PURPLE_MIN \
                            and org <= ARRIVE_ORANGE_MAX \
                            and abs(asym) <= ARRIVE_ASYMMETRY_MAX:
                        # ★ 最后一道：**格的同一性核验**（用户方案 ②-a）。
                        # 份额/形状类判据在"站在别格、看到完整同色面板"时全部正常，
                        # 只有"这一帧几何能否被地图解释"能分辨 ⇒ 解一次实测位姿。
                        _verdict, _why = self._verify_target_cell(p["frame"], digit)
                        # "no" 一律拒绝；"unknown"（锚弃权）按开关决定
                        # ——默认放行，否则会把大量正常到达一起误杀（锚可用率仅 ~16%）。
                        _ok_cell = (_verdict == "yes"
                                    or (_verdict == "unknown"
                                        and not ARRIVE_ON_CELL_REQUIRED))
                        if not _ok_cell:
                            print(f"[行进] 面板{digit} 色块判据成立但**格的核验否决**"
                                  f"（{_why}）→ 不判到达，继续逼近")
                            prev_cov, prev_org = None, None
                            continue
                        if _verdict == "yes":
                            print(f"[核验] 面板{digit} 格的同一性通过：{_why}")
                        self._arrive_evidence = (
                            f"脚下色块判据：紫区占比 {pur:.3f}（≥"
                            f"{ARRIVE_PURPLE_MIN}），橙区占比 {org:.3f}（≤"
                            f"{ARRIVE_ORANGE_MAX}），左右不对称 {asym:+.3f}"
                            f"（≤{ARRIVE_ASYMMETRY_MAX}）"
                            + (f"；格的核验: {_why}" if _verdict != "unknown"
                               else f"；格的核验: 弃权（{_why}）"))
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
                    turns, stepped = self._find_target_again(digit, lost, turns, p)
                    if turns < 0:
                        return False
                    self._reacq_deg += abs(stepped)
                    if stepped and (self._reacq_deg > FIND_MAX_TURN_DEG
                                    or self._reacq_frames > FIND_MAX_FRAMES):
                        print(f"[重捕获] 面板{digit} 已转 {self._reacq_deg:.0f}°"
                              f"／{self._reacq_frames} 帧仍未见到目标 → 放弃本格")
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
                # ---- 前进的**快慢两档**：远距多走几步、近距一步一复测 ----
                # ★ 分档量用**整帧色占比 `whole`**，不用框宽：
                #   ① 框宽在近距会因裁切**塌陷**（实测同一格 1164→676px 跳变），
                #      而 `whole` 单调且与到达门同源（都在同一张掩膜上算）；
                #   ② 这与旧三段式的"框宽 920px 交棒"是同一个量纲错误的反面教材。
                # 实测 `whole`↔距离（低头 1040，正前方）：
                #   60cm→0.056｜50→0.072｜40→0.092｜35→0.103｜30→0.115｜25→0.126
                #   20→0.137｜15→0.147（**此后随裁切不再增**）
                # ⇒ 近距离必判成 1 步：否则一次 5 步（13cm）会直接把机器人送过格心
                #   （实测：5 步批次把"橙 0.158"一步带到"橙 0.034"并冲过格心 28cm，
                #    框宽判据还误报"前进无效"而触发后退）。
                n = 1
                if zone == "forward" and sh is not None:
                    cov = float(sh[0])
                    if cov < FORWARD_COVER_MID:
                        n = FORWARD_FAR_STEPS
                    elif cov < FORWARD_COVER_NEAR:
                        n = FORWARD_MID_STEPS
                # ---- 前进无效脱困：**两个量都没进展**才算无效 ----
                # 判据必须用单调量，且**不能只看整帧占比**：面板走到脚下时有一部分
                # 出了画幅，`whole` 会**正常下降**——只看它就会把"正在压上去"误判成
                # "没走近"，于是"前进→后退→再前进"打成一个极限环（实测面板2：
                # 橙在 0.20↔0.10 之间来回，永远差一点够不着 0.10 的到达门）。
                # 正确的无效签名 = 占比不增 **且** 橙也不减（两个距离代理同时躺平）。
                cov_now = float(sh[0]) if sh is not None else 0.0
                org_now = float(sh[2]) if sh is not None else 0.0
                no_progress = (prev_cov is not None
                               and cov_now - prev_cov < NO_PROGRESS_COVER
                               and prev_org is not None
                               and org_now > prev_org - NO_PROGRESS_COVER)
                if zone == "forward" and no_progress:
                    stall += 1
                    if stall >= NO_PROGRESS_PATIENCE:
                        print(f"[行进] 面板{digit} 无进展（占比 "
                              f"{prev_cov:.3f}→{cov_now:.3f}、橙 {prev_org:.3f}→"
                              f"{org_now:.3f}）→ 后退一步重识别")
                        self._act("back_one_step", 1)
                        stall, prev_cov, prev_org, prev_zone = 0, None, None, None
                        continue
                else:
                    stall = 0
                if zone == "forward":
                    prev_cov, prev_org = cov_now, org_now
                else:
                    prev_cov, prev_org = None, None
                print(f"[决策] 面板{digit} 档={_ZONE_NAME_CN[zone]} "
                      f"偏角 {yaw_gain:+.1f}°｜横偏 {dx_body * 100:.1f}%画幅"
                      f"｜框宽 {p['box']:.0f}px"
                      + (f"｜紫 {sh[1]:.3f} 橙 {sh[2]:.3f}" if sh else "")
                      + f" → {action_name_cn(act, n)}")
                self._act(act, n)          # ★ 一次决策只下发一个原语（前进可带步数）
                prev_zone = zone
            print(f"[行进] 面板{digit} 统一循环迭代次数已达上限")
            return False
        finally:
            if pitch0 is not None:
                self.state.set_pitch(pitch0)
    def _verify_target_cell(self, frame, digit):
        """★ 格的同一性核验（用户方案 ②-a）：实测位姿是否真的落在目标格上？

        返回 (verdict, reason)：
          "yes" = 锚解出的**实测位姿**距目标格心 ≤ ARRIVE_ON_CELL_TOL_CM ⇒ 放行
          "no"  = 解出来了、但明显不在目标格 ⇒ **拒绝**（这就是"站在别格"的假到达）
          "unknown" = 锚不可用（诚实弃权）⇒ 按 ARRIVE_ON_CELL_REQUIRED 决定放不放

        为什么必须靠它（而不是继续加份额/形状阈值）：到达门那几个量都是**单帧整帧
        统计**，在"站在别的格子上、看到一块完整且居中的同色面板"这种视角下**全部
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
        """目标丢失后的**重捕获** → 返回 (turns, deg_turned)；返回 <0 表示放弃本格

        纯视觉、逐步转、有上限：
          ① 丢的第 1 帧先只动**头部**扫四档（零身体位移、不烧转向预算、无翻车风险）；
          ② 之后每丢 `TARGET_LOST_TOLERATE` 帧 ⇒ 朝地图方位转**一小步**（一次一个
             原语），每一步都拍照复测，转过冲由下一步反向吸收；
          ③ 累计转过 `FIND_MAX_TURN_DEG` 或拍够 `FIND_MAX_FRAMES` 仍不见
             ⇒ 放弃本格（交上层记未确认），**绝不无限转**（转向是最便宜的动作，
             没有上限就会"在原地转出场"）。
        """
        if lost == 1:
            for h in (HEAD_WIDE_LEFT, HEAD_LEFT, HEAD_RIGHT, HEAD_WIDE_RIGHT,
                      HEAD_CENTER):
                if self._cell_expired():
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
        turns += 1
        print(f"[重捕获] 面板{digit} 连续 {lost} 帧未见 → {action_name_cn(act)}"
              f"（第 {turns} 步，已转 {self._reacq_deg:.0f}°）")
        self._act(act, 1)
        return turns, step
    def _zone_masks(self, w, h):
        """把示意图四块区域栅格化成布尔掩膜（按画幅尺寸缓存，只算一次）

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
        key = (int(w), int(h))
        cache = getattr(self, "_region_cache", None)
        if cache is not None and cache[0] == key:
            return cache[1]
        xs = np.arange(w, dtype=np.float32)
        ys = np.arange(h, dtype=np.float32)
        dx = np.abs(xs - w / 2.0) / w                       # (w,)
        t = ys / h                                          # (h,)
        half_green = (ZONE_GREEN_TOP
                      + (ZONE_GREEN_BOT - ZONE_GREEN_TOP) * t)
        in_corr = dx[None, :] <= half_green[:, None]        # (h,w) 走廊
        ph = float(ZONE_PURPLE_HEIGHT)
        below_p = t >= (1.0 - ph)
        purple = in_corr & below_p[:, None]
        green = in_corr & (~below_p)[:, None]
        bh = float(ZONE_BLUE_HEIGHT)
        below_b = t >= (1.0 - bh)
        u = (np.clip((t - (1.0 - bh)) / bh, 0.0, 1.0) if bh > 0
             else np.ones_like(t))
        half_blue = (ZONE_BLUE_TOP
                     + (ZONE_BLUE_BOT - ZONE_BLUE_TOP) * u)
        blue = below_b[:, None] & (~in_corr) & (dx[None, :] <= half_blue[:, None])
        orange = ~(green | purple | blue)
        left = (xs < w / 2.0)[None, :]
        out = {"green": green, "purple": purple, "blue": blue, "orange": orange,
               "blueL": blue & left, "blueR": blue & (~left)}
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
        if self._cell_expired():
            return None
        self._count_frame()
        self._cell_frames += 1
        frame = self.state.capture_frame()
        if frame is None:
            return None
        sh = self._zone_shares(frame, color)
        obs_l = self.detector.detect_panels(
            frame, colors=[color], arbitrate=False, drop_border=False)
        o = max(obs_l, key=lambda x: x.hull_area) if obs_l else None
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
        """
        if w <= 0 or h <= 0:
            return "turn"
        t = float(np.clip(py / h, 0.0, 1.0))
        dx = abs(float(px) - w / 2.0) / w
        hy = (ZONE_HYSTERESIS_DEG / CAMERA_FOV_H_DEG) if ZONE_HYSTERESIS_DEG > 0 else 0.0
        half_green = ZONE_GREEN_TOP \
            + (ZONE_GREEN_BOT - ZONE_GREEN_TOP) * t
        if prev == "forward":
            half_green += hy
        if dx <= half_green:
            return "forward"
        bh = float(ZONE_BLUE_HEIGHT)
        if t >= 1.0 - bh:
            u = (t - (1.0 - bh)) / bh if bh > 0 else 1.0
            half_blue = ZONE_BLUE_TOP \
                + (ZONE_BLUE_BOT - ZONE_BLUE_TOP) * u
            if prev == "side":
                half_blue += hy
            if dx <= half_blue:
                return "side"
        return "turn"
    def _reanchor_pose(self, digit):
        """到达后重置位姿：**优先用地图锚的实测位姿**，退而用格心 + 推算航向

        统一决策循环到达时用**视觉 + 地图**反推自身位姿（`_map_pose`：把"地图里
        已知格心的面板"与"画面里看到的色块"配对，共识后解出场地系位姿），成功就
        整体采纳（**位置与航向都换成实测值**）；失败或判为误解则退回格心 + 原航向。

        ⚠️ 方向性：这条改的是**位姿来源**（推算 → 视觉反推），不是"用推算位姿导航"
        ——后者仍是禁止的。
        """
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
        self.pose = np.array([c[0], c[1], self.pose[2]])
        print(f"[锚定] 地图锚不可用 → 退回'格心 + 原航向'"
              f"（格心 {c[0]:.0f},{c[1]:.0f}"
              f"{'' if resid is None else f'，距格心 {resid:.1f}cm'}）")


def run_level(state):
    """数字宫格关卡入口（统一决策）：python main.py nine_grid"""
    level = NineGridLevel(state)
    return level.run_level()
