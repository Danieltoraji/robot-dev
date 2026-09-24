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
        的粗提示；GN 地图定位降级为可选校正（USE_MAP_CORRECTION，失败只记
        日志、不触发恢复链）。现场应急：VIS_NAV_ENABLED=False 回 v2 地图路径。
  定位（v2 地图路径，保留为回退/诊断）  地图 GN：布局已知后 9 个格心即
        "地图"；每步用名义动作模型预测位姿，以可见面板的「投影像素 ↔ 检出
        像素」残差迭代修正。未裁切用面板中心投影 ↔ 对角线交点；已裁切用
        "预测四角投影∩画幅的面积质心 ↔ 凸包质心"。IRLS + Cauchy 鲁棒核 +
        弱先验正则（σ=8cm/8°）。**注意：它依赖冻结的相机常数，地板形变下
        会整片超门控——这正是 v3 改视觉伺服的原因。**
  运动  小幅度动作白名单（单步 2cm 前进、~2cm 横移、3.2cm 后退）；远距
        批量上限 VIS_BATCH_MAX_STEPS=5（10cm）。本场地禁用
        go_forward / go_forward_fast（5cm 步幅在小面板+打滑地板上易摔倒，
        `_act` 硬门拒绝）。转向用「在线 EMA 估计小转实际角（误差源 = 像素
        yaw 或地图方位），连续 2 批 <0.5°/次→升级大转」+ 大转防振荡。
  容错  卡滞检测（框宽不涨/定位不改善 → 后退脱困）、目标丢失（头部扫找回
        → 回退半步 → 重搜）、分格时间预算 + **单格硬熔断**（VIS_CELL_HARD_*：
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
再在 (安装偏移, 相机高度) 参数网格上做格阵拟合（lattice_assign：假设-共识找
格阵基 + **4 个真旋转**硬约束（手性论证见模块内注释）+ 刚体残差门 + 覆盖率门
+ 入口侧/列序谓词 + 格6恒空优先）。胜出布局即数字→格，胜出组合中位数即自标定
常数；随后用"机器人系点 ↔ 场地格心"2D 刚体拟合**直接解出位姿**（不再解单应
再 decompose），GN 定位从此自续。格阵歧义/缺数字/自举自检不过时自动前进一步
重扫（平移视差消歧、带进近排），最多 3 轮。扫描终止判据 = 未裁切观测覆盖
≥LAYOUT_MIN_CLEAN_DIGITS 个数字**且两档俯仰都已有观测**（只扫一档会让
"安装偏移 vs 名义俯仰角"退化，实测 (1200,25°) ≡ (1040,10°)）。

现场依赖：
  1) 无标定依赖（布局扫 v2 已去除；archive/result/ninegrid_homography.json
     与 tools/calib_ninegrid.py 保留为诊断工具）；
  2) COLOR_THRESHOLDS 现场复标（tools/debug_vision.py ninegrid 子命令）；
  3) SVM 仲裁依赖 scikit-learn/skimage/joblib（缺失自动降级纯颜色）。

注意：localize() 不切换物理俯仰，按 state 当前俯仰拍照解算；调用方负责
用 state.set_pitch 保证俯仰与期望视野一致（GN 按 pitch 档投影）。
"""

import time
from collections import namedtuple

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

from core.camera_config import (
    HEAD_CENTER, HEAD_LEFT, HEAD_RIGHT, HEAD_WIDE_LEFT, HEAD_WIDE_RIGHT,
    SERVO_DEG_PER_US, CAMERA_INTRINSIC, CAMERA_DISTORTION,
    CAMERA_WIDTH, CAMERA_HEIGHT,
    CAM_PITCH_MOUNT_OFFSET_DEG, CAM_HEIGHT_STANDING_CM,
)
from core.ground_homography import (
    GroundHomography, grid_cell_center, cell_index,
    CAMERA_TO_BODY_FORWARD_CM, GRID_CELL_CM,
)
from core.robot_core import (RobotState, lock_camera_controls,
                                auto_calibrate_exposure,
                                CAM_AUTO_EXPOSURE_ENABLED)
from vision.nine_grid_detector import NineGridDetector, ID_TO_COLOR

# 相机光心高度缺省值：来自 camera_config（机器人自身属性）；layout_scan
# 运行时用格阵拟合参数网格自标定刷新（self._cam_height_cm）
CAM_HEIGHT_CM = CAM_HEIGHT_STANDING_CM
PITCH_NAV = 1200      # 前视导航/布局预扫（下沿 53.5°；转弯不拍到自身）
PITCH_DOWN = 1040     # 低头：可见地面带 0.16~1.47m（近距/脚下判定）
PANEL_HALF_CM = 14.0  # 面板色块半边长（33cm 格含缝，色块约 28cm）——P3 实测校准

# =====================================================================
# 到达与对准容差（面板 33cm，微动开关区≈中心 2/3≈±5.5cm）
# =====================================================================
ALIGN_TOL_DEG = 8.0      # 航向差小于此视为对准
ENTER_ALIGN_TOL_DEG = 15.0  # 进入段容忍的航向差上限（粗对齐即可，横移纠偏）
ENTER_BEARING_GATE_CM = 12.0  # 进入段纵向小于此不再按 bearing 转向（方位角病态）
FAR_DIST_CM = 18.0       # > 此值允许 one_step 批量（配合 3 步上限 ≈6cm）
MID_DIST_CM = 20.0       # > 此值 APPROACH，否则 ENTER（切低头）
# 为什么 20cm 就切低头：导航档（pitch 1200）可见地面从 28.9cm 起，目标面板
# 在 20cm 处远边仅 34cm、且其它面板都在身后——继续用导航档会"整帧 0 个面板"。
# 低头档（1040）可见地面从 15.8cm 起，能持续看到目标面板的可见条。
ENTER_MIN_CM = -10.0     # 中心退到脚后此值内仍算"到位"（防过冲判据）
ENTER_TOL_CM = 4.0       # 中心进入 [ENTER_MIN, ENTER_TOL] 转 CONFIRM
LAT_TOL_CM = 4.0         # 横偏容差；超过用横移纠正
CONFIRM_LAT_MAX_CM = 6.0
TURN_NEAR_LIMIT_CM = 20.0  # 距目标中心小于此先退格再转（防面板边缘转向）

# =====================================================================
# 动作白名单与名义位移模型（预测用；实测标定值见 levels/goodluck.py）
# =====================================================================
# 本场地禁用 go_forward / go_forward_fast：5cm 步幅在 33cm 小面板+打滑地板上
# 容易重心前扑摔倒（现场结论）。前进一律用 go_forward_one_step（2cm），
# 远距允许小批量（≤3 次=6cm）。白名单外的动作由 _act 直接拒绝。
DISABLED_ACTIONS = ("go_forward", "go_forward_fast")

FORWARD_ONE_STEP_CM = 2.0   # go_forward_one_step（唯一前进原语）
BACK_ONE_STEP_CM = 3.2      # back_one_step
LEFT_MOVE_CM = 1.9
RIGHT_MOVE_CM = 2.2
TURN_LEFT_DEG = 22.0
TURN_RIGHT_DEG = 25.7
BATCH_MAX_STEPS = 3         # 单次批量 one_step 上限（3×2cm=6cm；打滑地板保守值）
BATCH_MAX_ANGLE_DEG = 4.0   # 批量直行允许的最大航向差

# (类型, 名义增量)：fwd 前向 cm（负=后退）、lat 右向 cm、turn 右转度（右正）
ACTION_MODEL = {
    "go_forward_one_step": ("fwd", FORWARD_ONE_STEP_CM),
    "back_one_step": ("fwd", -BACK_ONE_STEP_CM),
    "left_move": ("lat", -LEFT_MOVE_CM),
    "right_move": ("lat", RIGHT_MOVE_CM),
    "turn_left": ("turn", -TURN_LEFT_DEG),
    "turn_right": ("turn", TURN_RIGHT_DEG),
    "turn_left_small_step": ("turn", -2.0),
    "turn_right_small_step": ("turn", 2.0),
    "stand": (None, 0.0),
}

# =====================================================================
# 自适应小转向（继承参考方案：在线估计 + 失效升级）
# =====================================================================
SMALL_TURN_INIT_DEG = 2.0     # 小转单次角度初估（P3 实测覆盖；goodluck 实测不可靠）
TURN_EMA_ALPHA = 0.3          # 指数滑动平均
TURN_EMA_ALPHA_FAST = 0.5     # 前几批用快收敛，抵消初值偏差
TURN_EMA_FAST_UPDATES = 2     # 快收敛批次数
MIN_EFFECTIVE_TURN_DEG = 0.5  # 每次实际转角低于此 = 被地面吞掉
SMALL_TURN_FAIL_BATCHES = 2   # 连续多少批失效才升级大转（单批噪声不误判）
SMALL_TURN_MAX_STEPS = 6      # 一轮小转批量上限
BIG_TURN_THRESHOLD_DEG = 30.0  # 超此角度直接大转（小转批量太慢）
BIG_TURN_OSC_ACCEPT_DEG = 15.0  # 大转振荡时接受残余航向差的上限（<半步长）

# =====================================================================
# 定位（Gauss-Newton 地图定位）门控
# =====================================================================
GN_MAX_ITERS = 10
GN_PIXEL_GATE = 12.0      # 单帧解残差 RMS 上限（px；裁切观测噪声 ~2px，留裕量）
GN_PIXEL_GATE_MERGED = 15.0  # 头部扫合并帧的残差上限（斜视角中心偏差更大）
GN_OUTLIER_PX = 25.0      # 单点残差超此剔除后重解（误检防护；勿把裁切点剔光）
GN_JUMP_CM = 12.0         # 解算位姿相对预测的最大跳变（预算+裕度）
GN_JUMP_DEG = 30.0        # 最大航向跳变
GN_CAUCHY_C_PX = 4.0      # IRLS 鲁棒核尺度（px）：大残差（裁切野值）近零权重
GN_PRIOR_SIGMA_XY_CM = 8.0    # 弱先验正则（cm）：低观测时抑制位姿乱跑
GN_PRIOR_SIGMA_TH_DEG = 8.0   # 弱先验正则（度）
GN_RMS_OK_PX = 3.0        # 多起点早停阈值：RMS 已足够好就不再试其它起点

# =====================================================================
# 视觉伺服导航 v3（2026-09-11：抗地板形变，导航不依赖相机常数）
# =====================================================================
# 现场约束（用户）：本关地板会形变，机器人在板上行走时机体俯仰/相机高度会
# 大幅变化（按 ±15° 考虑）。量化：h=50cm、竖直半视场 26.8° 时，有效俯角
# 42°/57°/72° 的可见地面带分别是 19~184 / 5~86 / 0~50 cm（4 倍范围摆动）
# —— 任何"冻结相机常数的度量投影"（GN 地图定位、"位姿覆盖中心"判据）在这种
# 形变下都不可用（sim 实测：随机游走 ±15° 时旧导航 0/7）。
# 故导航主链改为**相对量视觉伺服**（做法同参考实现
# reference code/九宫格视觉导航：yaw=-(dx/W)·FOV、proximity=框宽、
# 到达=低头颜色"出现→消失"）；地图（布局扫结果 digit_cell + 动作模型死推位姿）
# 只用于"往哪个方向找目标"的粗提示（±30° 足够）与每格到达后的位姿重置。
VIS_NAV_ENABLED = True          # False → 退回 v2 地图路径（现场应急开关）
USE_MAP_CORRECTION = False      # True → 每格尝试一次地图 GN 校正（失败只打日志）
CAMERA_FOV_H_DEG = 60.0         # yaw = -(dx/W)*FOV：名义水平视场，仅作伺服增益
# 对准死区 = **一个真实动作分辨率**（2026-09-13 真机"在死区里打转"复盘）：
# 现场实测小转一次 ≈10°（目视；打滑本就大，不可能更准），而旧死区收到 3.5°
# **小于一个步长** → 每轮都规划 1 步、每步过冲 6~10°、永远进不了死区：
#     yaw +4° →（转 1 步，实际 -10°）→ -6° →（转 1 步）→ +4° → … 无限振荡
# 6 轮后对准放弃 → 回到搜索再转 → 现场看到的就是"在死区莫名其妙打转"，
# 最后转向叠加盲走把机器人带出场地（2026-09-13 手工重跑实录）。
# 结论：**死区必须 ≥ 一个步长**（12° ≈ 10° + 打滑裕量），不追求度数精度。
# 也试过"用在线单步估计自适应取死区"（clip(1.2×估计, 4°, 12°)）：仿真里小转被
# 地面吞掉后估计崩到 0.8°，死区跟着收到 4° → 与 22° 大转步长严重失配 → 反而
# 把标称场景（无变形的基线）打挂（面板5 未确认）。**现场只有一个可靠数字：
# 小转一步 ≈10°，所以死区就取一个步长，不自适应**。
# 落点精度交给低头段的**厘米级横移纠偏**（VIS_ARRIVE_LAT_VIA_SIDESTEP：
# 2.2cm/步）+ 920px 交棒点——角度的分辨率被机械步长卡死，位置的分辨率没有。
VIS_ALIGN_TOL_DEG = 12.0        # 对准死区（≥ 一个真实小转步长）
VIS_BIG_TURN_DEG = 30.0         # 大/小转角分界（参考 BIG_TURN_THRESHOLD=30）
# **真机小转一步的真实角度（现场目视实测 ≈10°，2026-09-13；带打滑裕量取 11°）**
# 只用于"死区 ≥ 一个步长"这条不变量的自检，**不参与任何控制计算**。
# 为什么单独立成一个常量：现场唯一可靠的数字就是"小转一步多大"，而它决定了
# 死区的下限。详见文件末尾 `_check_align_invariant()`——死区小于一个步长会产生
# "规划 1 步 → 过冲 → 反向再规划 1 步 → 再过冲"的极限环（现场"在死区里莫名
# 打转"+6 轮后对准放弃→回搜索再转）。这条不变量以前只写在注释里，结果死区被
# 改成 3.5°（远小于 10° 的步长）而没有任何东西拦住——现在由代码强制。
FIELD_SMALL_TURN_STEP_DEG = 11.0
# 小转角闭环的规划下限（°/步）：规划步数 n = floor(|yaw| / max(EMA估计, 此值))。
# 为什么需要下限：真机/仿真都出现"小转被地面吞掉"→ EMA 估计崩到 0.3°/次 →
# n 被算成 SMALL_TURN_MAX_STEPS，一次请求 6 步 → 过冲 → 再算 n → 振荡。
# 有下限后最多请求 6 小步（≈12°），且**每轮都重新拍帧复测 yaw**，过冲由下
# 一轮吸收——收敛靠闭环，不靠这个常数准。
VIS_SMALL_TURN_MIN_STEP_DEG = 0.8

# =====================================================================
# 三档分区横向控制律（v3 对准，2026-09-23）—— 默认**关闭**，见 VIS_ZONE_ENABLED
# =====================================================================
# 动机：现行二值判据（|yaw| ≤ VIS_ALIGN_TOL_DEG 就前进，否则转）把"角度分辨率"
# 当成唯一修正手段。但真机小转一步实测是**左 8.625° / 右 5.200°**（runbook §2），
# 死区 12° 对右转只有 2.3 个步长、对左转 1.4 个 ⇒ 一步跨不出去就会
# "转一步过冲 → 反号再转 → 再过冲"，现场表现就是"在死区里打转"。
# 而**横移一步是厘米级的**（实测 2.497 / 2.200 cm/步），精度远好于角度量化。
# 所以把修正手段按精度排成三档：
#
#   ★ 与用户给的图一一对应：
#     绿（直行）= 方位角很小的中央梯形。图像里两腰是直线，而"方位角恒定"
#        对应的正是**过光心的竖直平面** ⇒ 形状必然是上窄下宽的等腰梯形。
#     蓝（横移）= 方位角已明显（> MOVE_DEG）但**够近**，横移才有权威。
#     橙（旋转）= 方位角太大，**或太远** —— 这就是图里蓝/橙之间那条**水平分界线**。
#
#   ★ "够近"用尺度不变量 near = 框宽 / 画幅高，不用像素行：
#     框宽是**投影不变量**（∝1/深度，同 _arrive_visual 的 dx_cm），除以画幅高后
#     无量纲 ⇒ 换俯仰档、地板形变都不改阈值。物理依据：横移一步改变的"跨越偏差"
#     恒为 2.2~2.5cm（与距离无关），但它改变方位角的效果 ∝ 1/距离 ⇒ 1m 外一步
#     只改 ~1.4°，效率极低，该改用旋转。
#
#   ★ 曾试过用"跨越偏差厘米数"当门（off_cm = 像素偏移/框宽×PANEL_WIDTH_CM）：
#     几何上更正确（恒定地面宽度走廊，反投影自检 X 极差 0.0000cm），但**离线
#     实测更差** —— 面板被画幅裁切时框宽只剩可见窄条 ⇒ off_cm 被系统性高估，
#     而现场 74% 的观测是裁切的 ⇒ 过度纠偏。三种阈值组合都没打赢二值基线。
#     故两个门都只用方位角 + near。
VIS_ZONE_MOVE_DEG = 6.0         # 方位角 ≤ 此值（度）→ 直行（绿）
VIS_ZONE_ROT_DEG = 12.0         # 方位角 ≤ 此值（度）**且够近** → 横移（蓝）
VIS_ZONE_ROT_NEAR = 0.35        # "够近"门槛 = 框宽/画幅高（见 _near_of）
# 来源一：**整关仿真 + 实测运动原语**的 36 组网格扫描（tools/_tmp_zone_eval.py
#   --sweep，4 局共 28 格）。基线（二值 12° 死区）到达 21/28；全场最优即本组
#   6.0/12.0/0.35 → **27/28**（动作 705→911，拍照 1126→1168）。
#   次优 6/12/0.5 与 6/18/0.5 都是 26/28 但动作更少（~740）。
# 来源二：运动学级对照（tools/_tmp_zone_sim.py，50 组初值网格）。
#   那里 6/18/0.35 最好（31/50 vs 基线 25/50），与整关仿真不一致——因为运动学
#   模型没有"接近/到达"两段与识别噪声。**以整关仿真为准**（它的口径更接近真机）。
# 二值律的失败**残余都在 150~170cm**（方向搞反后一路转出去），分区律把大部分
# 救回来（残余降到 3~7cm）——横移档的价值：它不靠"转对方向"吃饭，只靠"看目标
# 偏在哪边就往哪边挪"，对方向标定免疫。
VIS_ZONE_ENABLED = False        # False → 完全走原二值死区路径（默认，见下方理由）
# ★ 为什么默认关闭（**不是没做完，是风险控制**）：
#   离线对照（tools/_tmp_zone_eval.py）显示分区律的收益**只在"仿真与关卡都用
#   实测运动常量"时才体现**：
#     仿真原语=实测 + 关卡常量=实测 → 到达 49/56(二值) vs **54/56**(分区)
#     仿真原语=名义 + 关卡常量=名义 → 两者几乎无差别
#       （12° 死区在 2° 步长下有 6 个步长、绝对够用；真实 5.2~8.6° 步长下只有
#         1.4~2.3 个步长 —— 那才是现场"在死区里打转"的根因）
#   而把仿真与关卡常量一起换成实测值，会让现有仿真基线（SIM_LAYOUT/seed=3）
#   从 7/7 掉到 5/7 —— 那是独立一项工作（runbook 3a）。
#   ⇒ 在常量仍是名义值的当下打开分区律，等于用"为 2° 步长调的阈值"去指挥一个
#     5.2~8.6° 步长的机器人，离线实测退化到 2/7。所以分两步交付：
#       ① 本次：可开关、可离线对照、**默认行为逐位不变**（本开关 False）
#       ② 运动常量换实测值那一步做完后：把本开关翻成 True 并重跑基线
#   翻开关前先跑 python tools/_tmp_zone_eval.py 看两律的到达/代价对照。

PANEL_WIDTH_CM = 28.0           # 色块宽（像素↔厘米换算基准；与 PANEL_HALF_CM 同源）
# 到达触发：目标框宽（px）——"目标已进入 ~35cm"的**相对深度**代理
# （2026-09-11 由 820 收紧到 920，真机"踩不到微动开关"的根因之一）。
# 换算（针孔 + 沿轴深度）：box_px ≈ fx·W / Z，Z = d·cosθ + h·sinθ
#   fx = 1944.9px（camera_config 内参）、W = 28cm（色块宽；33cm 格含缝，
#   现场口径"面板 33cm"含缝，用 33 算只差 15%）、θ = 有效俯角
#   （pitch1200 = 名义27° + 安装18.5° = 45.5°）、h = 相机高 56cm
#   （camera_config 现值 = 现场双帧拟合 + 布局扫运行时自标定；用户口径的
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
VIS_ARRIVE_BOX_PX = 920.0
VIS_BATCH_MAX_STEPS = 5         # 远距批量上限（5×2cm=10cm，仍是小步幅）
# 低头到达判据（**尺度无关**版本，2026-09-11 实测重设计）：
# 参考实现用每色绝对占比阈值（0.002~0.17），但我们的相机俯角大、色块占比峰值
# 只有 ~0.13（实测红1：0.047 → 峰值 0.130 → 0.015），照抄阈值会永远判不到
# "看到"。改为记录**本次接近的占比峰值**：峰值超过绝对下限后，占比跌到峰值
# 的 VIS_COLOR_DROP_FRAC 以下即判"已压过"——与颜色/光照/俯仰无关。
VIS_COLOR_SEEN_MIN = 0.02       # "看到过颜色"的绝对下限（防噪声）
VIS_COLOR_DROP_FRAC = 0.55      # 占比跌破峰值的此比例 → 判定压过该格
# 实测（sim 红1/紫6 等）：占比曲线 0.06 → 峰值 0.13 → 平稳回落；取 0.55 是
# "跌掉 45%" 的保守证据，同时避免机器人刚好停在峰值附近时判不出来。
VIS_ARRIVE_MAX_ITERS = 30       # 低头段迭代上限（时间预算护栏）
# 迭代上限与"分段前压"对齐：前压封顶 45cm、精压段 2cm/帧 → 最多 23 帧，
# 再加 VIS_LAT_MAX_CORRECTIONS(3) 次纠横帧 + 首帧 = 27 ≤ 30，够用不空转。
# ---- 分段前压（2026-09-11 真机"踩不到微动开关"根因，重设计） ----
# 旧实现的收尾是"一次死推兜底"：低头段迭代走完就直接看死推距离（≤15cm 即
# 判成功）。它有三个毛病：① 15cm 已经大于微动开关有效区（±5.5cm）；② 这段
# 盲推里**横向偏差不纠**（旧代码只在"从未见到颜色"时纠横，而交棒时颜色早就
# 见到了）；③ 段中不复查颜色，压过头/没压到都看不出来。
# 改为**分段前压 + 每段复查占比峰值**：每段只压 2~6cm，段间必拍一帧复查
# （占比跌破峰值 → 立即判到位；占比还在涨 → 继续压；横向偏差超容差 → 先
# 小幅纠横再压）。总前压距离封顶 VIS_ARRIVE_PRESS_MAX_CM。
# 45cm 依据：新交棒点 ≈35cm（见 VIS_ARRIVE_BOX_PX）+ 面板半径 14cm ≈ 压过
# 格心的理论上限；留 4cm 裕量，同时保证绝不会一路压出 1.5 格（1 格 33cm）。
VIS_ARRIVE_PRESS_MAX_CM = 45.0
VIS_ARRIVE_PRESS_COARSE_STEPS = 3   # 占比还在涨时每段步数（3×2cm=6cm，赶路）
VIS_ARRIVE_PRESS_FINE_STEPS = 1     # 到峰/过峰后每段步数（2cm，精压）
# ---- 到达证据门（2026-09-13 假到达根因修复，语义修正） ----
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
# **横移控制**（见 _arrive_visual）——擦边条的凸包质心不是面板中心的合理估计。
VIS_ARRIVE_EVIDENCE_MIN_ASPECT = 0.5   # 证据区域的 h/w 下限（擦边条 ≈0.35）
VIS_ARRIVE_EVIDENCE_MIN_COVER = 0.08   # 证据区域面积占画幅比下限（假到达 0.042）
# ★ 2026-09-23 修正：上一条只对**未裁切**观测生效（见 _arrive_visual 的用法）。
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
# VIS_ARRIVE_BOX_PX）+ 一格 34cm ≈ 70cm —— 一次接近里前压超过这个距离，说明
# "落下去的"是更远处的东西，不是目标被压过。它只用**本格内累积的前压量**
# （每格到达后 _reanchor_pose 重置，格内只累积几步 2cm 的模型误差），
# 因此**不受长期位姿漂移影响**，可以放心用来否决视觉判据。
VIS_ARRIVE_PRESS_PLAUSIBLE_CM = 70.0
# 死推兜底（15/12cm → 6/5cm）：只在**分段前压已执行完**之后使用。判据收紧到
# 微动开关真实有效区：面板 33cm 的中心 2/3 ≈ ±5.5cm；蹭步（back3.2+forward2.0）
# 净 -1.2cm，故纵向 ≤6cm、横向 ≤5cm 才算"蹭步后仍能压到开关"。旧值 15cm 会把
# "离格心 15cm"也判成到位（真机现象：日志判成功、按钮没响、白丢 10 分）。
VIS_ARRIVE_FALLBACK_CM = 6.0    # 视觉判据走不完时，死推距离兜底（纵向 cm）
VIS_ARRIVE_FALLBACK_LAT_CM = 5.0   # 同上（横向 cm）
# 死推到位即停压（2026-09-13 形变阶跃场景新增的**第二个独立判据**）：
# "颜色峰值回落"在相机俯仰被地板形变改掉时会失效（deform 阶跃 +15° 实测：
# 前压 44cm 占比仍不回落 → 该判据根本发不出声），而**格内相对死推距离**
# （每格到达后由 _reanchor_pose 重置，格内只累积几步 2cm 的模型误差）与俯仰
# 无关。任一判据成立即判到位，遥测里写清是哪一条（诚实遥测）。
# 取 2.0cm：蹭步 = back3.2 + forward2.0（净 -1.2cm），停在格心前 2cm、蹭步后
# 再退 1.2cm ⇒ 落点 ≈ 格心 -3.2cm，仍在开关有效区（±5.5cm）内。
VIS_ARRIVE_STOP_FWD_CM = 2.0
# 纠横：**每次最多 3 步（6.6cm），且"看到色块之后"也继续纠**（2026-09-13 实测）。
# ① 执行方式用横移而不是小转：小转 2° 在 35cm 上前压只能挪 1.2cm，纠 7cm 要 6 次；
# ② 窗口不限制在"看到色块之前"：形变随机游走场景实测，把窗口收回
#    peak < VIS_COLOR_SEEN_MIN 后，面板 6/7 直接丢（占比已在涨、横偏还没纠完就
#    开始前压 → 压出去时偏半个开关区）。这与现场"踩不到微动开关"同源。
# dx_cm 是投影不变量，相机俯仰被形变改掉时会失真——所以次数有上限
# （VIS_LAT_MAX_CORRECTIONS），纠过头由下一帧复测接管，不会一路横着走。
VIS_LAT_MAX_CORRECTIONS = 4     # 低头段横向纠偏次数上限（防抖动）
VIS_STALL_BOX_PX = 5.0          # 前进无效判据（参考 proximity_change<5）
VIS_MIN_RATIO_CHANGE = 0.002    # 低头段占比变化下限（参考；紫6 取 0）
VIS_LAT_TOL_CM = 3.0            # 低头段横向容差（由 dx/box_w*PANEL_WIDTH_CM 换算）
# 纠横执行方式开关（可回退）：True = 横移（left/right_move，cm 级、直接消
# 偏差）；False = 旧行为（turn_*_small_step，度级——小转 2° 在 35cm 上前压
# 只能挪 35·sin2° ≈ 1.2cm，纠 10cm 偏差要 8 次，等于没纠）。
# 换算式 dx_cm = (px − W/2)/box_w · PANEL_WIDTH_CM 是投影不变量（抗形变），
# sim 实测精度：真值横偏 5/10cm（@30cm）→ 估 4.7/9.2cm。
VIS_ARRIVE_LAT_VIA_SIDESTEP = True
VIS_ARRIVE_LAT_MAX_STEPS = 3    # 单次纠横步数上限（3×2.2cm≈6.6cm，防纠过头）
VIS_SEARCH_MAX_ROUNDS = 4       # 搜索轮次上限（每轮 = 头部五档扫 + 转一步）
VIS_SEARCH_PROBE_STEPS = 3      # （保留）早前的前进探测步数
VIS_SEARCH_BLIND_CM = 25.0      # 盲走判据：死推纵向大于此值时先走近再由视觉接管
VIS_SEARCH_BLIND_MAX_STEPS = 10  # 单轮盲走上限（10×2cm=20cm）
# 单格盲走**总量**上限（2026-09-13 真机"其它异常行动"）：搜索靠"死推提示"决定
# 往哪转/往哪走，而提示来自死推位姿——位姿在长时间盲走+转向后会发散（仿真形变
# 阶跃场景实测：一次失败的搜索盲走约 1m，随后位姿与真相差 111.9cm）。盲走越多
# 位姿越离谱、下一格的提示越不可用：宁可本格记未确认，也不要在场上瞎走。
# 60cm ≈ 2 格——目标在 2 格内都没进过视野，就不是"再走走"能解决的。
VIS_SEARCH_BLIND_TOTAL_CM = 60.0
VIS_SEARCH_MAX_FRAMES = 26      # 单格搜索拍照硬上限（时间预算护栏）
# ---- 单格转向护栏（2026-09-13 真机"一直转 → 冲出场地"复盘，安全项） ----
# 现场现象：机器人在某一格内持续转向（对准与搜索互相接管），最后转出场外。
# 旧实现只在时间/拍照数/动作数上设了上限，而**转向是最便宜的动作**（不拍照、
# 不耗帧）：没有"转够了就停"的概念，它可以在死区里空转很久。
# 对策：单格累计**命令**转角（含搜索/对准/大转/纠横的所有 turn_*）超预算即
# 熔断本格，并**拒绝执行**那最后一次转向（见 _act）——宁可本格记未确认。
# 360° 依据（180° → 360°，2026-09-13 形变随机游走实测修正）：一格里**正常**
# 的转向总量 = 对准 ≤60°（几个 10° 小步）+ 搜索换视角最多两轮大转
# （每轮 ≤129°）≈ 320°；取"整圈" 360° 只拦"已经在原地打转"的情形，
# 正常最坏路径不误伤（仿真单格实测最大 180°）。早期取 180° 会在"目标在身后、
# 一轮搜索没扫到"时误熔断——deform 随机游走场景面板 4 就是这样丢的。
VIS_TURN_BUDGET_DEG = 360.0     # 单格累计命令转角上限（一整圈）
VIS_TURN_STALL_TURNS = 4        # 连续 N 次转向后 |yaw| 都没变小 → 打转判据
VIS_ALIGN_MAX_FLIPS = 3         # 死区外左右来回摆 N 次 → 判定打转（直接收手）
VIS_ALIGN_WORSEN_DEG = 5.0      # 单次转向把 |yaw| 转大超过此值 = "变差"
# ---- 离场护栏（同上，安全项） ----
# 场地是 1m×1m 台面（格心 16.7~83.3cm），起点在台面南侧 (50,-20)（见 __init__）。
# 外框取"台面 ± 1 格"（33cm）：拦的是**彻底失控**（已经跑到台面外一整格），
# 不是防跌落预案。为什么不收得更紧：死推位姿本身在形变下实测能偏 30cm+
# （到达判据"颜色峰值回落"在俯仰 ±15° 时会误判，_reanchor_pose 又把位姿锚到
# "以为"的格心——deform 随机游走实测落点残差最大 36.6cm）。真按 ±18cm 收，
# 会拿一个已经不可信的位姿把好局误熔断（该场景实测：面板4 之后整局被误中止）。
# **真正管住"继续动"的是上面三条**：转向预算 360°、停转/来回摆判据、前压封顶。
# 越界即站立并**中止整局**（不再搜索下一格）——走下台面的代价远大于丢分。
VIS_FIELD_X_CM = (-33.0, 133.0)
VIS_FIELD_Y_CM = (-48.0, 133.0)


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
    if VIS_ALIGN_TOL_DEG < FIELD_SMALL_TURN_STEP_DEG:
        raise AssertionError(
            f"VIS_ALIGN_TOL_DEG({VIS_ALIGN_TOL_DEG}°) < 现场小转一步"
            f"({FIELD_SMALL_TURN_STEP_DEG}°，见该常量注释)：死区小于一个步长时"
            "闭环必然过冲振荡（真机现场表现='在死区里打转'）。"
            "要改小死区，必须先用现场日志证明小转一步真的变小了。")
    if VIS_ALIGN_TOL_DEG > VIS_BIG_TURN_DEG:
        raise AssertionError(
            f"VIS_ALIGN_TOL_DEG({VIS_ALIGN_TOL_DEG}°) > "
            f"VIS_BIG_TURN_DEG({VIS_BIG_TURN_DEG}°)：死区比大转分界还大，"
            "对准会直接接受任意姿态")


_check_align_invariant()

# =====================================================================
# 容错预算
# =====================================================================
TARGET_TIME_BUDGET_S = 110.0   # 单格软预算（15min/7格 ≈ 128s，留裕量）
TOTAL_TIME_BUDGET_S = 780.0    # 全局看门狗 13min（给上下场留 2min）
# ---- 单格硬熔断（2026-09-11 真机"终点不停"后新增，安全项） ----
# 现场现象：走到第 7 块面板后机器人仍在继续行动；日志显示它并没有"越界"，
# 而是第 7 格一直没确认到位，于是"搜索→对准→接近→低头→蹭步"整条链在单格内
# 反复重试，直到格预算耗尽——人看着就是"到了终点不停"。
# 旧实现的问题：超时只在**子循环入口**判（`while time.time() < t_end`），而真机
# 拍照 ~0.7s/张（2026-09-13 实测，见 VIS_CAPTURE_COST_S），一次 `_see_target_any`
# （头部五档）也有 ~4s，嵌套后单格实际能跑到 150s+，且过冲量随嵌套层数叠加。
# 对策（三重护栏，任一触发即"本格记未确认、立刻返回"）：
#   1) 时间：VIS_CELL_HARD_TIMEOUT_S，且在**每次拍照前**判（最细粒度，过冲
#      最多 1 张照片 ~0.7s，而不是一个子循环 4s+）；
#   2) 拍照数 VIS_CELL_HARD_FRAMES /   3) 动作数 VIS_CELL_HARD_ACTIONS：
#      与时钟无关的确定性护栏——即使某环节出现"不拍照也不动作"的空转或
#      "疯狂动作"的失控，也一定在有限步内退出（防新增死循环的保险丝）。
# 取值依据（时间 70s）：
#   - 总预算：全局 780s，布局扫真机实测 60~90s，剩 ~690s / 7 格 ≈ 98s/格；
#     70s 给"搜索/对准/接近/低头"四段各留失败重试余量，同时保证**最坏情况**
#     （7 格全熔断）90 + 7×70 = 580s < 780s，绝不拖过看门狗；
#   - **实测余量充足**（2026-09-13 订正）：整局 206 张 × 0.70s ≈ 2.4 分钟，
#     即使全部 7 格都跑到 70s 上限也只有 490s+90s < 780s。历史上"真机 2~4s/张"
#     的估计**偏高约 4 倍**（疑似把网络 RTT 当成了拍照耗时），曾据此误判"整局会
#     撞看门狗、单格只够 21 张"——按实测这个担心不成立。70s 仍是硬上限，
#     它管的是"某个环节卡死"，不是"正常流程不够用"。
#   - 仿真（sim，7 格全程只花 ~13s，单格最长 ≈3s；见 test_nine_grid_sim 的
#     [进度] 行）：70s 在仿真里永不触发。
VIS_CELL_HARD_TIMEOUT_S = 70.0
# 全局剩余时间按剩余格数分摊的比例（自适应收缩）：硬熔断 = min(70s,
# 剩余时间/剩余格数 × 0.9)。这样即使每格都被熔断，累计也不会超过全局看门狗
# （分摊是望远镜求和：Σ 剩余/剩余格数 = 剩余时间），0.9 再留 10% 余量。
# 下限 45s：避免最后一格因前面拖时而只剩几秒、连一次搜索都跑不完。
VIS_CELL_HARD_BUDGET_FRAC = 0.9
VIS_CELL_HARD_TIMEOUT_MIN_S = 45.0
VIS_CELL_HARD_FRAMES = 140     # 单格拍照硬上限（基线整局 206 张、单格最多 ~38 张）
VIS_CELL_HARD_ACTIONS = 300    # 单格动作硬上限（基线整局 176 次、单格最多 ~55 次）
# ---- 单格"拍照数"预算（与时间预算绑定，让真机不会把时间耗在拍照上） ----
# 为什么需要它：旧护栏 VIS_CELL_HARD_FRAMES=140 是死判据（单格最多 ~38 张，永不
# 触发），拦不住"某环节反复拍照把单格时间耗光"。做法是把"本格还剩多少时间"
# 换算成"还允许拍几张"：frame_budget = 本格硬预算秒数 ÷ 单张耗时估计。
# 单张耗时估计由 state.capture_cost_s 提供（真机默认见 VIS_CAPTURE_COST_S；
# 仿真里 SimNineGridRobot 声明一个极小值 ⇒ 该闸不会先于时间闸触发，回归可比）。
#
# **实测（2026-09-13 现场直接量，不要再用猜的）**：
#     fswebcam 2592x1944 -S 3 × 6 次 = 0.74/0.61/0.66/0.61/0.63/0.61s（均值 0.62s）
#     走代码路径 RobotState.capture_frame()（fswebcam + cv2.imread）= **0.70s**
# 历史文档里"真机 2~4s/张"的估计**偏高约 4 倍**（很可能是把 RTT/弱链路读数当成
# 了拍照耗时）。这个高估会把拍照预算压到 70/3×0.9 ≈ 21 张/格，而实测单格需求
# 最多 61 张 ⇒ **预算会先把本来能跑完的好格砍掉**。故按实测取 0.9s（留 ~30%
# 裕量覆盖进程/内存竞争），得到 ≈70 张/格，与 70s 时间闸等价而不更严。
# 换机器人/换相机请重测（tools/field_probe_ninegrid.py 可复用其锁相机链路）。
VIS_CAPTURE_COST_S = 0.9       # 机器人单张耗时（秒；2026-09-13 实测 0.70s + 裕量）
VIS_CAPTURE_COST_SAFETY = 0.9  # 再留 10% 裕量（动作/转头也吃时间）
STALL_EPS_CM = 0.5             # 卡滞判据：前进单步距目标改善低于此
STALL_CONSEC = 2               # 连续多次触发即卡滞
RETRY_LIMIT = 2                # 单格 CONFIRM 失败重试上限
ENTER_FAIL_LIMIT = 3           # 进入段连续定位失败上限
RECOVER_LIMIT = 5              # 单段定位丢失恢复上限（逐级升级，超过放弃本段）
LOC_BUDGET_PER_TARGET = 40     # 单格定位次数软护栏（超限只告警，真机时间诊断用）


# =====================================================================
# 格阵拟合（布局扫 v2：机器人系相对几何，无站位/标定依赖）
# =====================================================================
# 数学原理：可见面板是已知 3×3 刚性格阵（间距 GRID_CELL_CM，格6恒空）经未知
# 刚体变换（平移+旋转）后的带噪子集。拟合三步：
#   1) 枚举"某对面板=相邻格"假设，其余面板在该假设格阵基下整数化，
#      残差 ≤LATTICE_INLIER_CELL 格距者为内点（对角/跨格假设被共识自动否决）；
#   2) 取内点最多的假设为格阵基（要求**全部输入数字**都是内点，否则判失败）；
#   3) 枚举 **4 个真旋转**（不含镜像）作为格号朝向候选，硬约束过滤：
#      无重复占格 / 覆盖门（每块板都在自己格号 0.35 格距内）/ 刚体残差门
#      （干净子集 RMS）/ 入口侧谓词 / 列序谓词；"格6 恒空"优先但不硬杀
#      （全灭时降级并告警，避免约定不符直接崩溃）。
#
# 为什么只枚举真旋转（手性论证，2026-09-11 修复 P0）：
#   像素→机器人系（GroundHomography.from_pose((0,0),…)）与像素→场地系
#   （project_ground_to_pixel）用的是同一套投影模型，两个坐标系都是
#   x 右 / y 前 / z 上的右手系；格阵基 e2 = CCW90(e1) 也是右手构造，
#   因此"格阵整数坐标 → 场地格号"的真解必然是真旋转（det = +1 族），
#   镜像候选在几何上不可能成立。历史 bug：D4 把镜像一并枚举，镜像候选能
#   通过全部旧约束并被选中 → 布局被转置且位姿自举崩溃。真实照片实测分离度
#   （2701 干净点集）：真旋转 RMS 0.62cm vs 镜像 18.75cm。
LATTICE_INLIER_CELL = 0.35      # 格阵基整数化/覆盖门残差上限（格距单位）
LATTICE_RMS_MAX_CM = 8.0        # 干净子集刚体拟合 RMS 上限（cm）
# 阈值依据（2026-09-11 现场照片实测）：真旋转族 0.6~4.5cm（含参数网格量化：
# 偏移步 2.5° 在 1m 处 ≈4cm），镜像族 17.4~18.8cm → 取 8cm 兼顾现场裕量与
# 区分度（真/镜像相差 2 倍以上）。过紧会让现场"3 轮重扫后直接失败丢整关"。
LATTICE_RMS_MAX_ALL_CM = 0.35 * GRID_CELL_CM   # 无干净子集时的退化门（≈11.7cm）
LATTICE_CLIPPED_WEIGHT = 0.3    # 仅由裁切观测支撑的数字，刚体拟合中的权重
LATTICE_CENTER_PRIOR_X_CM = 50.0  # 入口居中先验（场地系 x，cm）
LATTICE_CENTER_MARGIN_CM = 10.0   # 居中度差距小于此且解不同 → 判歧义（重扫）

# ---- 参数网格自标定（见 NineGridLevel._lattice_grid_fit） ----
# 偏移下界必须含 0：2026-09-11 现场照片实测，pitch1040 的最优偏移落在 10°
# 边界上（有效俯角 ≈51.5°），网格下界留 0 才不会把真值卡在边界。
LAYOUT_OFFSET_MIN_DEG = 0.0
LAYOUT_OFFSET_MAX_DEG = 40.0
LAYOUT_OFFSET_STEP_DEG = 2.5
LAYOUT_HEIGHT_MIN_CM = 36.0
LAYOUT_HEIGHT_MAX_CM = 70.0
LAYOUT_HEIGHT_STEP_CM = 5.0
LAYOUT_VOTE_MIN_FRAC = 0.6      # 胜出布局须占全部成功组合的比例
LAYOUT_VOTE_MIN_COMBOS = 10     # 且组合数下限（防少数组合碰巧一致）
# 像素域五参数精修门（见 _pixel_pose_calib）：重投影 RMS 上限 = 导航的 GN 门控
# （GN_PIXEL_GATE）——自标定常数必须让导航级残差合格，否则真机会定位风暴。
# 门用**中位残差**而不是 RMS：真实照片里少数面板会因光照/阴影/半合并使
# 观测中心偏几十 px，RMS 被它们抬高（实测中位 ~20px 而 RMS ~110px），
# 但参数估计由多数好观测驱动，故用稳健的统计量判定。
PIXEL_CALIB_MED_MAX_PX = 40.0
SPREAD_MAX_CM = 8.0             # 簇内离散告警阈值（cm；超出说明该数字观测不可信）
# 歧义修复（E）交换余量：歧义数字的观测点必须"离竞争格比离本格近"这么多 cm
# 才考虑交换格号（约 1/4 格距，远大于现场观测 1~2cm 的跨帧离散）。
AMBIG_SWAP_MARGIN_CM = 8.0
CLUSTER_RADIUS_CM = 6.0         # 同数字跨帧观测聚类半径（cm）：取最大一致簇
LAYOUT_MIN_CLEAN_DIGITS = 6     # 提前结束扫描所需的"未裁切观测覆盖数字数"
POSE_BOOTSTRAP_RMS_MAX_CM = 0.35 * GRID_CELL_CM  # 位姿自举刚体残差上限
# 取"覆盖门同量级"（0.35 格距 ≈11.7cm）：布局若已被 lattice_assign 接受
# （每点 ≤0.35 格距、干净子集 RMS ≤LATTICE_RMS_MAX_CM），自举就不该用更严的
# 门把它否掉——否则会出现"布局通过但位姿被拒"的自相矛盾流程。区分"布局错"
# 靠的是覆盖门 + 手性 RMS 门 + 规则/入口侧/列序谓词，不靠本门。
POSE_BOOTSTRAP_X_RANGE = (-10.0, 110.0)   # 自举位姿合理性门（场地系 cm）
POSE_BOOTSTRAP_Y_RANGE = (-45.0, 105.0)   # 入口在场外，y 允许负值

# 布局扫一次拟合的完整结果（cells=None 表示失败；见 _lattice_grid_fit）
# pose = 像素域精修得到的位姿 (x, y, θ)（None = 未收敛，回退刚体自举）
LatticeFit = namedtuple(
    "LatticeFit",
    "cells info offset_deg cam_height_cm pts_robot clean weights "
    "rms_clean_cm rms_all_cm spread_cm res_max_cm warnings pose")


def _pixel_pose_calib(entries, p0, off0, h0, iters=12):
    """像素域五参数联合精修 (x, y, θ, 安装偏移, 相机高度) → (pose5, rms_px)

    为什么需要：格阵刚性判据只约束"这批点是不是 33.3cm 格阵"，(安装偏移,
    相机高度) 存在一条**等价退化谷**——sim 实测 (18.5°, 56cm) 与 (19.5°, 59cm)
    的刚体 RMS 都是 0.30cm，但导航用的投影对谷内位置很敏感：高度差 3cm 会让
    GN 残差从 <1px 涨到 16~23px，直接触发"定位风暴"（拍照数爆表）。
    像素域目标用的是完整投影模型（含裁切感知观测），谷被消掉。

    entries: [(格心场地坐标, 观测像素, head_pulse, 裁切?, pitch_pulse), ...]
    与导航同一残差定义 + Cauchy 鲁棒 + 弱先验（观测少时仍可解）。
    无效（面板跑到相机背后/参数越界）返回 (None, None)。
    """
    ent = [(np.asarray(g, dtype=np.float64), np.asarray(px, dtype=np.float64),
            int(head), bool(cl), int(pitch))
           for g, px, head, cl, pitch in entries]
    if len(ent) < 4:
        return None, None, None
    w_obs = np.array([1.0 if not cl else 0.6 for _g, _px, _h, cl, _p in ent])
    sig = np.array([20.0, 20.0, np.radians(20.0), 12.0, 20.0])  # 弱先验 σ
    ref = np.array([p0[0], p0[1], p0[2], float(off0), float(h0)], dtype=np.float64)

    def resid(p):
        out = np.empty((len(ent), 2))
        for k, (g, px, head, cl, pitch) in enumerate(ent):
            # 有效性守卫：观测点当初是在画幅内被检出的，若预测中心跑到画幅外
            # 很远（或面板跑到相机背后），这一步长把参数推到了不自洽处 →
            # 视为无效，由线搜索缩步。否则优化器会"收敛"到把面板甩出画外的
            # 大残差解（实测真实照片集上出现干净残差 2.6e4 px 的假收敛）。
            qc = project_ground_to_pixel(g, p[0], p[1], p[2], pitch, head,
                                         pitch_offset_deg=p[3],
                                         cam_height_cm=p[4])[0]
            if not np.all(np.isfinite(qc)) or not (
                    -1500.0 <= qc[0] <= CAMERA_WIDTH + 1500.0
                    and -1500.0 <= qc[1] <= CAMERA_HEIGHT + 1500.0):
                return None
            if cl:
                q = clipped_quad_centroid(g, p[0], p[1], p[2], pitch, head,
                                          pitch_offset_deg=p[3],
                                          cam_height_cm=p[4])
            else:
                q = qc
            if q is None:
                return None
            q = np.asarray(q, dtype=np.float64)
            if not np.all(np.isfinite(q)):
                return None
            out[k] = q - px
        return out

    p = ref.copy()
    r = resid(p)
    if r is None:
        return None, None, None
    steps = np.array([0.5, 0.5, np.radians(0.5), 0.5, 1.0])
    dmax = np.array([3.0, 3.0, np.radians(3.0), 3.0, 3.0])
    for _it in range(iters):
        per = np.linalg.norm(r, axis=1)
        # Cauchy 权重按观测给（与 _gn_run 同一鲁棒核）
        w = w_obs / (1.0 + (per / GN_CAUCHY_C_PX) ** 2)
        J = np.zeros((r.size + 5, 5))
        for j in range(5):
            e = np.zeros(5)
            e[j] = steps[j]
            r1 = resid(p + e)
            r2 = resid(p - e)
            if r1 is None or r2 is None:
                return None, None, None
            J[:r.size, j] = ((r1 - r2) / (2.0 * steps[j])).ravel()
        for j in range(5):                      # 弱先验行
            J[r.size + j, j] = 1.0 / sig[j]
        rp = np.concatenate([r.ravel(), (p - ref) / sig])
        wf = np.concatenate([np.repeat(w, 2), np.ones(5)])
        try:
            d = np.linalg.solve(J.T @ (J * wf[:, None]) + 1e-8 * np.eye(5),
                                -(J.T @ (rp * wf)))
        except np.linalg.LinAlgError:
            return None, None, None
        d = np.clip(d, -dmax, dmax)
        ok = False
        for _ls in range(6):
            cand = p + d
            if (-15.0 <= cand[3] <= 55.0 and 25.0 <= cand[4] <= 100.0
                    and abs(cand[0] - ref[0]) <= 60.0
                    and abs(cand[1] - ref[1]) <= 60.0
                    and abs(_wrap_angle(cand[2] - ref[2])) <= np.radians(40.0)):
                rc = resid(cand)
                if rc is not None:
                    p, r, ok = cand, rc, True
                    break
            d = d * 0.5
        if not ok:
            break
        if np.max(np.abs(d[:3])) < 1e-3 and abs(d[3]) < 0.02 and abs(d[4]) < 0.02:
            break
    per = np.linalg.norm(r, axis=1)
    cl = np.array([c for _g, _px, _h, c, _p in ent])
    rms_all = float(np.sqrt(np.mean(per ** 2)))
    med_clean = (float(np.median(per[~cl]))
                 if int(np.count_nonzero(~cl)) >= 3 else float(np.median(per)))
    return p, rms_all, med_clean


def _fit_failure(info, warnings=None):
    """构造失败的 LatticeFit（cells=None）；见 _lattice_grid_fit"""
    return LatticeFit(cells=None, info=info, offset_deg=0.0, cam_height_cm=0.0,
                      pts_robot={}, clean=set(), weights={}, rms_clean_cm=None,
                      rms_all_cm=None, spread_cm={}, res_max_cm=None,
                      warnings=list(warnings or []), pose=None)


def _rigid_fit_2d(gs, ps, w=None):
    """p ≈ R·g + t 的 2D **加权**刚体最小二乘（R 为真旋转）。

    gs: (N,2) 场地格心；ps: (N,2) 机器人系点；w: (N,) 权重（缺省 1）。
    返回 (R, t, rms_cm, per_point_cm)。R 把场地系映射到机器人系，故
    机器人在场地系的位置 = -Rᵀ·t（机器人系原点 p=0 的场地坐标）。
    """
    gs = np.asarray(gs, dtype=np.float64)
    ps = np.asarray(ps, dtype=np.float64)
    ww = np.ones(len(gs)) if w is None else np.asarray(w, dtype=np.float64)
    sw = float(ww.sum())
    gm = (gs * ww[:, None]).sum(axis=0) / sw
    pm = (ps * ww[:, None]).sum(axis=0) / sw
    Hm = ((gs - gm) * ww[:, None]).T @ (ps - pm)
    U, _S, Vt = np.linalg.svd(Hm)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Rm = Vt.T @ np.diag([1.0, d]) @ U.T
    tt = pm - Rm @ gm
    per = np.linalg.norm(ps - ((Rm @ gs.T).T + tt), axis=1)
    return Rm, tt, float(np.sqrt(np.mean(per ** 2))), per


def ambiguity_repair(cells, pick, ambig, margin_cm=None):
    """歧义修复（E）：用格阵共识复核颜色有歧义、形状又没定案的面板

    输入：cells = {数字: 格号}（格阵投票胜出解）；pick = 该解的拟合明细
    （含 med=各数字机器人系点、clean=可信数字集合）；ambig = {数字: {竞争数字}}。
    做法：
      1. 只用**无歧义且可信**的数字拟合刚体变换（场地格心 → 机器人系点）；
      2. 对每个歧义数字 d 与其竞争数字 c（两者都已在 cells 里）：比较 d 的观测点
         到"格(d)"与"格(c)"预测位置的距离，若后者近出一个余量以上 ⇒ 交换二者格号；
      3. 交换后用全点重算残差，**只有残差变小才接受**（否则回退）。
    返回 (cells, notes)：notes 为人类可读的修复记录（无修复时为空）。
    几何不可判（置信数字 <3、缺观测、候选不在解里）时原样返回——宁可不动。
    """
    margin_cm = AMBIG_SWAP_MARGIN_CM if margin_cm is None else margin_cm
    notes = []
    if not ambig or not cells or pick is None:
        return cells, notes
    med = pick.get("med") or {}
    pairs = []
    for d, cands in ambig.items():
        if d not in cells or d not in med:
            continue
        for c in cands:
            if c in cells and c != d and c in med and (c, d) not in pairs:
                pairs.append((d, c))
    if not pairs:
        return cells, notes
    clean = set(pick.get("clean") or ())
    base_amb = {d for pr in pairs for d in pr}
    ref = [d for d in cells if d in med and d in clean and d not in base_amb]
    if len(ref) < 3:                      # 可信参照不足：换用全部非歧义数字
        ref = [d for d in cells if d in med and d not in base_amb]
    if len(ref) < 3:
        return cells, notes

    def _resid(cells_now):
        ds = sorted(d for d in cells_now if d in med)
        gs = np.array([grid_cell_center(cells_now[d]) for d in ds])
        ps = np.array([med[d] for d in ds])
        w = np.array([1.0 if d in clean else LATTICE_CLIPPED_WEIGHT for d in ds])
        _R, _t, _rms, per = _rigid_fit_2d(gs, ps, w)
        return float(np.sqrt(np.mean(np.asarray(per) ** 2)))

    # 参照刚体变换：只用参考数字（歧义数字的观测不参与，避免"自己证明自己"）
    gs = np.array([grid_cell_center(cells[d]) for d in sorted(ref)])
    ps = np.array([med[d] for d in sorted(ref)])
    R, t, _rms, _per = _rigid_fit_2d(gs, ps, np.ones(len(ref)))

    def _pred(cell):
        return R @ grid_cell_center(cell) + t

    new_cells = dict(cells)
    for d, c in pairs:
        pd, pc = np.asarray(med[d]), np.asarray(med[c])
        d_own = float(np.linalg.norm(pd - _pred(cells[d])))
        d_other = float(np.linalg.norm(pd - _pred(cells[c])))
        # 对称检查：竞争数字 c 的观测也应更靠近格(d)，否则不动（可能只是噪声）
        c_own = float(np.linalg.norm(pc - _pred(cells[c])))
        c_other = float(np.linalg.norm(pc - _pred(cells[d])))
        if d_other + margin_cm >= d_own or c_other + margin_cm >= c_own:
            continue
        cand = dict(new_cells)
        cand[d], cand[c] = new_cells[c], new_cells[d]
        if _resid(cand) < _resid(new_cells):
            notes.append(
                f"数字 {d}↔{c} 交换格号（{new_cells[d]}↔{new_cells[c]}）："
                f"观测点距对格 {d_own:.1f}cm、距竞争格 {d_other:.1f}cm"
                f"（余量 {margin_cm:.0f}cm），交换后残差下降")
            new_cells = cand
    return new_cells, notes



def lattice_assign(points_by_digit, spacing_cm=GRID_CELL_CM, weights=None,
                   clean=None):
    """机器人系相对几何格阵拟合：{digit: (x,y)} → ({digit: cell}, info, ranked)

    points_by_digit: 各面板在**机器人系**地面坐标（像素经名义高度/俯仰/
    头部角的单应映射，与场地系无关）。
    weights: {digit: w} 刚体拟合权重（缺省 1.0；仅裁切观测支撑的数字应降权，
    见 LATTICE_CLIPPED_WEIGHT）。
    clean:   可信（未裁切）数字集合；刚体残差门优先在它上面判（<3 个时退化
    为全点门）。手性/参数错时镜像族会在这里被拒。
    返回 (cells|None, info, ranked)：
      cells  = 数字→格号，覆盖**全部**输入数字（覆盖不足即整体判失败）；
      ranked = 全部可行候选（rule_ok / cam_x / rms_* 字段），供诊断与调用方自检。

    算法（假设-共识）：
      1) 枚举"某对面板=相邻格"假设（长度带内全部有序对），把其余面板在
         该假设的格阵基下整数化，残差 ≤LATTICE_INLIER_CELL 格距者为内点；
      2) 取内点最多的假设（并列取残差和最小）；对角/跨格假设会被共识
         自动否决（第三块板落不到整数格上）；
      3) 对内点整数坐标枚举 4 个真旋转（非镜像，见本节顶部手性论证），
         硬约束过滤 + 居中先验裁决；多解不猜（判歧义，由调用方前进一步重扫）。
    """
    digits = sorted(points_by_digit)
    if len(digits) < 4:
        return None, f"可见面板仅 {len(digits)} 块（需≥4 才能定朝向）", []
    pts = {d: np.asarray(points_by_digit[d], dtype=np.float64) for d in digits}

    # ---- 1) 假设-共识：找最优格阵基与整数坐标 ----
    best = None  # ((n_inliers, -res_sum), int_coords, inliers)
    for i in digits:
        for j in digits:
            if i == j:
                continue
            v = pts[j] - pts[i]
            L = float(np.linalg.norm(v))
            if not 0.55 * spacing_cm <= L <= 1.7 * spacing_cm:
                continue
            e1 = v / L
            e2 = np.array([-e1[1], e1[0]])
            coords, res = {}, {}
            for d in digits:
                w = (pts[d] - pts[i]) / spacing_cm
                a, b = float(w @ e1), float(w @ e2)
                ia, ib = int(round(a)), int(round(b))
                coords[d] = (ia, ib)
                res[d] = max(abs(a - ia), abs(b - ib))
            inliers = [d for d in digits if res[d] <= LATTICE_INLIER_CELL]
            if len(inliers) < len(digits):
                continue  # 覆盖率门：任一块板落不到整数格 → 假设或观测有误
            intc = {d: coords[d] for d in inliers}
            as_ = [a for a, _ in intc.values()]
            bs = [b for _, b in intc.values()]
            if max(as_) - min(as_) > 2 or max(bs) - min(bs) > 2:
                continue
            if len(set(intc.values())) != len(intc):
                continue  # 两板落同一格 → 假设不成立
            score = (len(inliers), -sum(res[d] for d in inliers))
            if best is None or score > best[0]:
                best = (score, intc, inliers)
    if best is None:
        return (None, "无一致格阵假设（面板相对位置与 33cm 格阵不符，或存在"
                      "坏观测/假阳，需重扫）", [])
    _score, ij, inliers = best

    # ---- 2) 4 个真旋转枚举：硬约束 + 刚体残差门 + 入口侧/列序谓词 ----
    # 行号/列号约定：行0=远排、行2=入口排（格6-8）；列0=入口视角左
    # （标号约定单一真源 = core/ground_homography.ROW0_IS_FAR / COL0_IS_LEFT，
    #  格号构造统一走 cell_index）。
    # 入口侧判定：对每个候选朝向做 2D 刚体 Procrustes（网格系→机器人系），
    # 机器人在网格系的 y 必须小于所有可见面板的网格 y（+半格容差）——
    # 即"机器人从入口侧进入、位于全部可见面板之前"的严格表达。
    # （"入口行均值距离最近"这类启发式在侧偏+大偏航下会误判，弃用）
    # "格6 恒空"是本关规则硬约束，但它与"机位/标号约定"耦合：若 4 个真旋转
    # 里无一满足，则降级接受并打警告，避免约定不符时直接崩溃、白丢整关。
    ds = sorted(ij)                    # 覆盖率门保证 == sorted(digits)
    w_all = np.array([(weights or {}).get(d, 1.0) for d in ds])
    clean_set = set(clean or ())
    feet = []
    for rot in range(4):               # 只枚举真旋转（手性论证见本节顶部）
        mapping = {}
        for d, (a, b) in ij.items():
            aa, bb = a, b
            for _ in range(rot):
                aa, bb = bb, -aa
            mapping[d] = (aa, bb)
        amin = min(a for a, _ in mapping.values())
        bmin = min(b for _, b in mapping.values())
        cells = {d: cell_index(a - amin, b - bmin)
                 for d, (a, b) in mapping.items()}
        if len(set(cells.values())) != len(cells):
            continue  # 两板落同一格 → 该朝向不成立
        gs = np.array([grid_cell_center(cells[d]) for d in ds])
        ps = np.array([pts[d] for d in ds])
        Rm, tt, rms_all, per = _rigid_fit_2d(gs, ps, w_all)
        # 覆盖率门：每块板都必须落在自己格号的 0.35 格距内
        if float(np.max(per)) > LATTICE_INLIER_CELL * spacing_cm:
            continue
        idx = [k for k, d in enumerate(ds) if d in clean_set]
        rms_clean = (float(np.sqrt(np.mean(per[idx] ** 2)))
                     if len(idx) >= 3 else None)
        gate = (LATTICE_RMS_MAX_CM if rms_clean is not None
                else LATTICE_RMS_MAX_ALL_CM)
        if (rms_clean if rms_clean is not None else rms_all) > gate:
            continue
        cam_grid = -Rm.T @ tt
        # 入口侧精确谓词：机器人必须在全部可见面板的入口侧
        if cam_grid[1] >= float(np.min(gs[:, 1])) + GRID_CELL_CM / 2:
            continue  # 机器人不在全部可见面板的入口侧 → 朝向错
        # 列0 必须在机器人系更左（x 更小；列间 66.6cm ≫ 偏航余弦衰减）
        col_x = {}
        for d, cell in cells.items():
            col_x.setdefault(cell % 3, []).append(pts[d][0])
        if 0 in col_x and 2 in col_x \
                and float(np.mean(col_x[0])) > float(np.mean(col_x[2])):
            continue
        feet.append({"rule_ok": 6 not in cells.values(),
                     "cells": {d: int(cells[d]) for d in ds},
                     "cam_x": float(cam_grid[0]), "cam_y": float(cam_grid[1]),
                     "rms_clean": rms_clean, "rms_all": rms_all,
                     "res_max_cm": float(np.max(per))})

    if not feet:
        return None, ("4 个真旋转均不满足约束（无重复格 / 覆盖 0.35 格距门 / "
                      "刚体残差门 / 入口侧 / 列序）——可见子集不对称、观测"
                      "有误或标号约定不符"), []

    # 排序：先满足"格6恒空"（规则），再入口居中先验（场地入口在底边中央
    # x=50），最后比刚体残差。多解且居中度相近 → 判歧义（调用方前进一步重扫）。
    def _rank_key(fc):
        return (not fc["rule_ok"],
                abs(fc["cam_x"] - LATTICE_CENTER_PRIOR_X_CM),
                fc["rms_clean"] if fc["rms_clean"] is not None
                else fc["rms_all"])

    ranked = sorted(feet, key=_rank_key)
    pool = [fc for fc in ranked if fc["rule_ok"]] or ranked
    if len(pool) > 1:
        margin = (abs(pool[1]["cam_x"] - LATTICE_CENTER_PRIOR_X_CM)
                  - abs(pool[0]["cam_x"] - LATTICE_CENTER_PRIOR_X_CM))
        if margin < LATTICE_CENTER_MARGIN_CM \
                and pool[1]["cells"] != pool[0]["cells"]:
            return None, (f"朝向歧义（{len(pool)} 个可行解且居中度相近）"
                          "——前进一步重扫消解"), ranked
    best = pool[0]
    if not best["rule_ok"]:
        return (dict(best["cells"]),
                "格阵拟合成功（警告：位置6 被占——标号约定或机位存疑，"
                "需现场踩格核对）", ranked)
    if best["rms_clean"] is None:
        info = f"格阵拟合成功（{len(feet)} 个可行朝向，无干净子集→全点 RMS 门）"
    else:
        info = (f"格阵拟合成功（{len(feet)} 个可行朝向，干净子集 RMS "
                f"{best['rms_clean']:.1f}cm）")
    return dict(best["cells"]), info, ranked



class NineGridLevel:
    """数字宫格关卡：布局扫描 + 地图定位 + 逐格视觉闭环导航"""

    def __init__(self, state: RobotState, start_cam_xy=(50.0, -20.0),
                 start_bearing_deg=0.0):
        self.state = state
        self.detector = NineGridDetector()
        self.start_cam_xy = np.array(start_cam_xy, dtype=np.float64)
        self.start_bearing_deg = float(start_bearing_deg)
        self.digit_cell = {}        # 数字(1..7) -> 位置编号(0..8)
        self.cell_conflict = set()  # 布局扫时仲裁冲突的格（近距复核）
        self.deadline = None
        # 位姿 (x, y, θrad)：相机光心地面投影 + 机体航向（右正弧度）
        self.pose = np.array([self.start_cam_xy[0], self.start_cam_xy[1],
                              np.radians(self.start_bearing_deg)])
        # 自适应小转状态（整场共享：地面条件一致，一次失效全场大转）
        self._small_turn_deg = SMALL_TURN_INIT_DEG
        self._small_turn_usable = True
        self._small_turn_fail = 0   # 连续失效批次数（>=SMALL_TURN_FAIL_BATCHES 升级）
        self._turn_updates = 0      # EMA 更新次数（前几次用快收敛）
        self._last_turn = None      # (action, times)：定位丢失时撤销转向用
        self._last_big_turn = None  # (方向, 转前|bearing|)：大转防振荡
        self._turn_oscillation = False  # 大转振荡已发生 → 改用航向偏置接近
        # 相机安装/高度常数（机器人级，缺省取 camera_config）；layout_scan
        # 运行时用格阵拟合参数网格自标定刷新，GN 投影随动
        self._pitch_offset_deg = CAM_PITCH_MOUNT_OFFSET_DEG
        self._cam_height_cm = CAM_HEIGHT_CM
        # 到达确认状态（每格重置）
        self._target_seen = False   # 接近/进入段是否检出过目标数字面板
        self._loc_count = 0         # 本格定位次数（时间预算诊断）
        # 转向/离场护栏状态（见 VIS_TURN_* / VIS_FIELD_*；每格重置）
        self._cell_turn_cmd_deg = 0.0   # 本格累计**命令**转角（含搜索/对准/大转）
        self._align_stall = 0       # 连续"转向后 |yaw| 没改善"次数（打转判据）
        self._align_worsen = 0      # 连续"同号误差被转得更大"次数（方向自检）
        self._align_flips = 0       # 死区外左右来回摆的次数（打转判据）
        self._align_ladder = 0      # 打转升级阶梯档位（1 换大转 / 2 放弃对准 / 3 熔断）
        self._align_sign = 1.0      # 对准转向符号；方向自检判反了取 -1（整局保持）
        self._abort_level = None    # 非 None = 整局收手原因（离场护栏）
        # 诚实遥测：本格"到达"的依据 / 交棒 yaw / 落点死推残差（不参与决策）
        self._arrive_evidence = None
        self._cell_handoff_yaw = None
        self._cell_arrive_resid_cm = None
        # 单格硬熔断状态（每格重置；见 VIS_CELL_HARD_* 常量）
        self._cell_deadline = None  # 本格硬熔断时刻（None = 未开预算，不拦）
        self._cell_t0 = None        # 本格开始时刻（日志用）
        self._cell_frames = 0       # 本格拍照数
        self._cell_actions = 0      # 本格动作数
        self._cell_tripped = None   # 熔断原因（None = 未熔断）
        self.cell_trips = {}        # {数字: 熔断原因}——未确认格的可观测记录
        # FSM 阶段标签（纯诊断：日志/仿真可视化用，不参与任何决策）
        self.phase = "INIT"
        # 逐阶段拍照计数（真机时间预算诊断：拍照 ~0.7s/张，7 格总预算 780s）
        self.phase_frames = {}
        # 逐格**落点真值**留档：{digit: (离格心 cm, 来源)}。来源 "真值"（仿真有
        # state.pos）或 "死推"（真机只有位姿估计）。用于回归时直接抓"假到达"
        # ——视觉判 ✓ 但真值不在格上（2026-09-13 面板3 就是 89.5cm 判 ✓）。
        self.panel_landing = {}

    # =================================================================
    # 单格硬熔断（安全项：宁可早停记未确认，也不要失控）
    # =================================================================

    def _cell_budget_begin(self, digit):
        """开本格预算：返回**软预算** t_end（旧语义），并装好硬熔断闸

        软预算（TARGET_TIME_BUDGET_S）= 各子循环入口的既有超时判据，保持
        不变以最小化行为改动；硬熔断（VIS_CELL_HARD_*）= 新增的、在**每次
        拍照前**都判的细粒度闸，先于软预算生效（70s < 110s），保证任何环节
        卡住都在有限时间内退出。两者都取 min(self.deadline)，绝不越过全局
        看门狗。
        """
        now = time.time()
        if self.deadline is None:       # 直接调用 go_to_panel（测试/工具）时的兜底
            self.deadline = now + TOTAL_TIME_BUDGET_S
        t_soft = min(now + TARGET_TIME_BUDGET_S, self.deadline)
        cells_left = max(1, 8 - int(digit))          # 含本格
        share = (self.deadline - now) / cells_left * VIS_CELL_HARD_BUDGET_FRAC
        hard = float(np.clip(share, VIS_CELL_HARD_TIMEOUT_MIN_S,
                             VIS_CELL_HARD_TIMEOUT_S))
        self._cell_deadline = min(now + hard, t_soft)
        self._cell_t0 = now
        self._cell_frames = 0
        # 单格拍照预算：本格时间预算 ÷ 单张耗时估计（见 VIS_CAPTURE_COST_*）。
        # 真机上它 ≈ 70/3×0.9 = 21 张——与"70s 里真能拍几张"一致，所以它才是
        # 真机真正生效的那道闸；仿真里 state 声明单张≈0 → 该闸不会先于时间闸
        # 触发，于是回归结果不受影响，而"真机跑不完"这件事在仿真里也能被看见。
        self._cell_frame_budget = int(np.clip(
            (self._cell_deadline - now)
            / max(float(getattr(self.state, "capture_cost_s",
                                VIS_CAPTURE_COST_S)), 1e-3)
            * VIS_CAPTURE_COST_SAFETY,
            1.0, float(VIS_CELL_HARD_FRAMES)))
        self._cell_actions = 0
        self._cell_tripped = None
        # 转向护栏 + 诚实遥测（每格重置；_align_sign 是整局标定，刻意不重置）
        self._cell_turn_cmd_deg = 0.0
        self._align_stall = 0
        self._align_worsen = 0
        self._align_flips = 0
        self._align_ladder = 0
        self._arrive_evidence = None
        self._cell_handoff_yaw = None
        self._cell_arrive_resid_cm = None
        print(f"[熔断] 面板{digit} 单格护栏：硬 {hard:.0f}s"
              f"（软预算 {(t_soft - now):.0f}s，全局剩余 "
              f"{self.deadline - now:.0f}s/{cells_left}格）"
              f"｜拍照预算 {self._cell_frame_budget} 张"
              f"（单张按 {float(getattr(self.state, 'capture_cost_s', VIS_CAPTURE_COST_S)):.1f}s 估）")
        return t_soft

    def _cell_expired(self):
        """单格硬熔断判据（**每次拍照/动作前**调用；True = 本格立即收手）

        三重护栏（时间 / 拍照数 / 动作数）任一超限即 True，并把原因记在
        self._cell_tripped（供 run_level 汇总打印与 self.cell_trips 留档）。
        设计上只"允许收手"，不改变任何既有判据的语义——所以它能在不引入
        新死循环的前提下，兜住搜索/对准/接近/低头补步任一环节。
        """
        if self._cell_tripped is not None:      # 已熔断：保持熔断（不反复打印）
            return True
        reason = None
        if self._cell_actions > VIS_CELL_HARD_ACTIONS:
            reason = f"动作数 {self._cell_actions}>{VIS_CELL_HARD_ACTIONS}"
        elif self._cell_frames > VIS_CELL_HARD_FRAMES:
            reason = f"拍照数 {self._cell_frames}>{VIS_CELL_HARD_FRAMES}"
        elif self._cell_frames > self._cell_frame_budget:
            # 与时间预算等价的拍照闸（见 VIS_CAPTURE_COST_S）：真机上"拍满了"
            # 就等于"时间快用完了"，但它在**拍照前**就能判，不必等时间闸响。
            reason = (f"拍照数 {self._cell_frames}>本格预算 "
                      f"{self._cell_frame_budget}（按单张 "
                      f"{float(getattr(self.state, 'capture_cost_s', VIS_CAPTURE_COST_S)):.1f}s"
                      " 折算的时间预算）")
        elif self._cell_deadline is not None \
                and time.time() > self._cell_deadline:
            reason = (f"单格用时 {time.time() - self._cell_t0:.0f}s "
                      f"超硬上限 {VIS_CELL_HARD_TIMEOUT_S:.0f}s")
        if reason is None:
            return False
        self._cell_tripped = reason
        return True

    def _trip_cell(self, reason, abort_level=False):
        """立即熔断本格（安全项统一出口）；abort_level=True 时整局一起收手

        与 _cell_expired 的三重护栏同源：置 _cell_tripped 后，所有子循环在
        下一次 _cell_expired() 处退出，go_to_panel 返回 False，run_level 记
        "未确认"。**绝不"再试一次"**——现场教训：为了"再试一次"继续留在场上
        动，代价是机器人转出场地。
        """
        if self._cell_tripped is None:
            self._cell_tripped = reason
        print(f"[护栏] {reason} → 本格收手，站立不动")
        if abort_level and self._abort_level is None:
            self._abort_level = reason
        self.state.act("stand")
        return True

    def _field_guard(self):
        """离场护栏：死推位姿越出场地外框（台面 ±1 格）→ 熔断本格 + 中止整局

        定位是"拦彻底失控"的兜底，不是防跌落预案：死推位姿在形变场景下实测
        能偏 30cm+（到达判据本身有误差，_reanchor_pose 又把它锚到"以为"的
        格心），所以外框放到台面 ±1 格。真正管住"继续动"的是转向预算 360°、
        停转/来回摆判据、前压封顶 45cm 这三条。
        """
        if self._abort_level is not None:
            # 已判定整局收手：保持熔断（新格开预算会清 _cell_tripped，这里补回）
            if self._cell_tripped is None:
                self._cell_tripped = self._abort_level
            return True
        x = float(self.pose[0])
        y = float(self.pose[1])
        if (VIS_FIELD_X_CM[0] <= x <= VIS_FIELD_X_CM[1]
                and VIS_FIELD_Y_CM[0] <= y <= VIS_FIELD_Y_CM[1]):
            return False
        return self._trip_cell(
            f"位姿越界 (x={x:.0f}, y={y:.0f}) 超出 x{VIS_FIELD_X_CM} "
            f"y{VIS_FIELD_Y_CM}", abort_level=True)

    def _capture(self):
        """统一拍照入口：单格拍照计数（硬熔断用）+ 转发 state.capture_frame"""
        self._cell_frames += 1
        self._count_frame()
        return self.state.capture_frame()

    # =================================================================
    # 入口
    # =================================================================

    def run_level(self):
        """布局扫描 → 按 1..7 顺序逐格导航。返回 bool（全部确认到位）

        每格结果同时写入 self.results（[(数字, 是否确认), ...]），供仿真/
        真机日志与诊断使用。
        """
        self.deadline = time.time() + TOTAL_TIME_BUDGET_S
        # 锁定相机白平衡/对焦（可选曝光）：一局开始钉一次，消除 fswebcam 每次
        # 重新测光/白平衡造成的跨帧漂移（现场实测白点 R/B 差 16% → 粉 7 漏检）。
        # 锁不上不影响继续：归一化仍能兜（见 core.robot_core.lock_camera_controls）。
        lock_camera_controls()
        # 自动曝光闭环：把画面均亮拉到目标值。**换灯/换场地后不必手改常数**
        # （实测曝光写死 100 时换灯后均亮只剩 7~22，归一化失效、扫描三轮未定）。
        if CAM_AUTO_EXPOSURE_ENABLED:
            auto_calibrate_exposure(self.state)
        self.layout_scan()
        print(f"[布局] 数字→格: {self.digit_cell}  "
              f"仲裁冲突格: {sorted(self.cell_conflict)}")

        done = []
        for k in range(1, 8):
            ok = self.go_to_panel(k)
            done.append((k, ok))
            self.state.act("stand")
            elapsed = TOTAL_TIME_BUDGET_S - (self.deadline - time.time())
            note = ""
            if not ok and self._cell_tripped:
                # 熔断收尾（安全项）：本格记为未确认，继续下一格/收尾，
                # 绝不为了"再试一次"把机器人继续留在场上动。
                note = f"（单格熔断：{self._cell_tripped}）"
                self.cell_trips[k] = self._cell_tripped
                print(f"[熔断] 面板{k} 收手：{self._cell_tripped}"
                      "——记为未确认，不再纠缠")
            print(f"[进度] 面板{k} {'完成' if ok else '未确认(以微动开关实际触发为准)'}"
                  f"{note}，已用时 {elapsed:.0f}s")
            # 单格诚实遥测：一行给出现场复盘需要的全部数字（事后不用翻半个日志）
            hy = self._cell_handoff_yaw
            rs = self._cell_arrive_resid_cm
            print(f"[遥测] 面板{k} 用时{elapsed:.0f}s 拍照{self._cell_frames} "
                  f"动作{self._cell_actions} 命令转角{self._cell_turn_cmd_deg:.0f}°"
                  f" 交棒yaw{'—' if hy is None else f'{hy:+.1f}°'}"
                  f" 落点残差{'—' if rs is None else f'{rs:.1f}cm'}")
            print(f"[遥测] 面板{k} 依据: {self._arrive_evidence or '（未到达）'}")
            if self._abort_level:
                # 离场护栏：已经越界，绝不再去"找下一格"——站立收手，保已得分
                print(f"[护栏] 整局收手：{self._abort_level}"
                      "（机器人保持站立，不再动作）")
                break
            if time.time() > self.deadline:
                print("[看门狗] 全局超时，停止（保已得分，不冒进）")
                break
        self.results = done
        if self.cell_trips:
            print(f"[诊断] 单格熔断: {self.cell_trips}")
        frames = sorted(self.phase_frames.items(), key=lambda kv: -kv[1])
        print(f"[诊断] 逐阶段拍照数: {frames}  合计 {sum(self.phase_frames.values())}")
        return all(ok for _, ok in done)

    # =================================================================
    # 布局扫描（机器人系相对几何格阵拟合，无站位/标定依赖）
    # =================================================================

    def layout_scan(self):
        """头部双俯角宽扫全场 → 机器人系相对几何格阵拟合 → 数字→格 映射

        每帧观测经"机器人系单应"（名义高度+俯仰+头部角，均自身已知）转为
        机器人系地面坐标，按 digit 跨帧聚合（干净观测优先，同色=同板）后做
        格阵拟合（lattice_assign：假设-共识 + 4 真旋转硬约束 + 刚体残差门）；
        布局确定后用"机器人系点 ↔ 场地格心"刚体拟合解出位姿写入 self.pose，
        GN 定位从此自续。全程不依赖站位复位与点击标定。

        扫描终止判据：**未裁切观测覆盖 ≥LAYOUT_MIN_CLEAN_DIGITS 个数字**，
        或扫满一轮（双俯仰×五头部）。为什么不用"见到 7 种数字"：现场照片
        实测单帧常见 7 种数字但只有 3 个未裁切，裁切质心偏差 1.3~8.4cm，
        据此拟合不稳定（2026-09-11 复现）。
        歧义/缺数字/自举自检不过时自动前进一步重扫（平移视差消解朝向歧义；
        也把没入视野的近排带进来），最多 3 轮。
        """
        self.phase = "LAYOUT"
        last_err = ""
        fit = _fit_failure("未执行")
        for attempt in range(3):
            pix_obs = []        # (pitch, head, digit, 观测像素, 裁切?)
            frame_obs = []      # (pitch, head, obs)：仲裁冲突回填写
            for pitch in (PITCH_NAV, PITCH_DOWN):
                self.state.set_pitch(pitch)
                for pulse in (HEAD_CENTER, HEAD_LEFT, HEAD_RIGHT,
                              HEAD_WIDE_LEFT, HEAD_WIDE_RIGHT):
                    self.state.set_head(pulse)
                    frame = self.state.capture_frame()
                    if frame is None:
                        continue
                    # 裁切感知：贴纸小、入口视角大半面板贴画幅边——保留
                    # 裁切面板，用凸包质心观测（拟合时降权，见 _aggregate）
                    # shape=True：颜色贴窗口边界时用数字形状仲裁（C3）——
                    # 布局一次定全局，颜色错判代价最大，这里最该开形状。
                    obs = self.detector.detect_panels(frame, arbitrate=True,
                                                      drop_border=False,
                                                      shape=True)
                    frame_obs.append((pitch, pulse, obs))
                    for o in obs:
                        px = (o.hull_centroid_px if o.clipped
                              else o.center_px)
                        pix_obs.append((pitch, pulse, o.digit, px,
                                        o.clipped))
                    # 提前退出：须同时满足（a）未裁切覆盖 ≥6 个数字、
                    # （b）**两档俯仰都已有观测**。只扫一档会让"安装偏移 vs
                    # 名义俯仰角"退化（实测 (1200,25°)≡(1040,10°)），
                    # 自标定常数无法外推到另一档，且参数网格成功组合太少
                    # （现场照片实测：单档仅 6 个组合 < 多数票门槛）。
                    if len({e[2] for e in pix_obs if not e[4]}) \
                            >= LAYOUT_MIN_CLEAN_DIGITS \
                            and len({e[0] for e in pix_obs}) >= 2:
                        break
                if len({e[2] for e in pix_obs if not e[4]}) \
                        >= LAYOUT_MIN_CLEAN_DIGITS \
                        and len({e[0] for e in pix_obs}) >= 2:
                    break
            self.state.set_head(HEAD_CENTER)
            self.state.set_pitch(PITCH_NAV)

            seen = {e[2] for e in pix_obs}
            clean_digits = {e[2] for e in pix_obs if not e[4]}
            missing = [k for k in range(1, 8) if k not in seen]
            info = (f"缺数字 {missing}（未裁切覆盖 {len(clean_digits)} 个）"
                    if missing else "")
            if not missing:
                # 参数网格自标定 + 格阵拟合（见 _lattice_grid_fit）
                # 颜色歧义表：{数字: {竞争数字}}，供拟合后的格阵共识复核（E）
                ambig = {}
                n_amb = 0
                for _p, _h, obs_list in frame_obs:
                    for o in obs_list:
                        if not o.ambiguous:
                            continue
                        n_amb += 1
                        cands = {k for k in o.candidates if k != o.digit}
                        if cands:
                            ambig.setdefault(o.digit, set()).update(cands)
                if n_amb:
                    print(f"[布局] 颜色歧义观测 {n_amb} 个"
                          f"（涉及数字 {sorted(ambig)}），交由形状仲裁/格阵共识")
                fit = self._lattice_grid_fit(pix_obs, ambig)
                info = fit.info
                for w in fit.warnings:
                    print(f"[布局] 警告: {w}")
                if fit.cells is not None:
                    worst = sorted(fit.spread_cm.items(),
                                   key=lambda kv: -kv[1])[:3]
                    print(f"[布局] 观测聚合: 未裁切覆盖 {len(clean_digits)}/7 "
                          f"数字；跨帧离散最大 {[(d, round(s, 1)) for d, s in worst]}")

            if fit.cells is not None and self._pose_bootstrap(fit):
                self.digit_cell = fit.cells
                self._pitch_offset_deg = fit.offset_deg
                self._cam_height_cm = fit.cam_height_cm
                # 仲裁冲突回填：布局已知后才能把冲突面板定位到格
                for _pitch, _head, obs in frame_obs:
                    for o in obs:
                        if o.arb_conflict and o.digit in self.digit_cell:
                            cell = self.digit_cell[o.digit]
                            self.cell_conflict.add(cell)
                            print(f"[布局] 仲裁冲突：颜色{o.digit}@格{cell} "
                                  f"SVM说{o.model_digit}({o.model_conf:.2f})"
                                  "——待近距复核")
                print(f"[布局] {fit.info}；参数自标定: 安装偏移 "
                      f"{self._pitch_offset_deg:+.1f}° "
                      f"高度 {self._cam_height_cm:.0f}cm "
                      f"（有效俯角 {self._effective_pitch_deg(PITCH_NAV):.1f}°"
                      f"@pitch{PITCH_NAV}）")
                return

            if fit.cells is not None:
                # 布局解出但位姿自举自检没过：归因为自举，别误导成拟合失败
                info = (f"{fit.info}；但位姿自举自检未通过——重扫复核"
                        f"（布局 {fit.cells}）")
            last_err = info
            if attempt < 2:
                print(f"[布局] 第{attempt + 1}轮未定（{info}）——"
                      "前进一步重扫（视差消歧/带进近排）")
                self._act("go_forward_one_step", 1)
        seen_digits = sorted({e[2] for e in pix_obs})
        raise RuntimeError(
            f"布局扫描三轮未定（{last_err}）。可见数字: {seen_digits}")

    def _effective_pitch_deg(self, pitch_pulse):
        """名义脉宽角 + 自标定安装偏移 = 有效俯角（度，日志用）"""
        return ((1500 - pitch_pulse) * SERVO_DEG_PER_US
                + self._pitch_offset_deg)

    def _lattice_grid_fit(self, pix_obs, ambig=None):
        """参数网格自标定 + 格阵拟合 → LatticeFit

        相机安装偏移/高度无法精确预知（装配离散、俯仰随头部姿态微变），在
        (偏移, 高度) 网格上逐组合做"机器人系映射 + 逐数字聚合 + 格阵拟合"。
        格阵归属是离散判定——落在拟合容差谷值内的组合给出同一布局，多数票
        即布局；胜出组合的中位数即自标定常数（写回调用方）。

        pix_obs: [(pitch, head, digit, 观测像素, 裁切?), ...]（跨帧全部观测）

        聚合策略（2026-09-11 现场照片 + 真机布局扫实测驱动）：
          - **最大一致簇取模式**：同一数字的多次观测先按 CLUSTER_RADIUS_CM
            邻域聚类，取点数最多的一簇（真机实测：场内木框被橙色阈值命中、
            蓝地垫被蓝色阈值命中，且多为**未裁切**观测——单纯"干净优先的
            中位数"会被它们带到 40cm 外，格阵拟合整体失败）；
          - 簇内**干净优先**：有未裁切观测就只用未裁切观测（裁切质心偏差
            实测 1.3~8.4cm）；否则用簇内裁切观测并把权重降到
            LATTICE_CLIPPED_WEIGHT；簇外观测计入 dropped 供告警/诊断。
        退化提醒：只有单档俯仰有观测时，安装偏移与名义俯仰角不可分离
        （实测 (pitch1200, offset=25°) 与 (pitch1040, offset=10°) 等价，
        有效俯角都 ≈51.5°），此时自标定常数不可外推到另一档 → warnings。
        """
        warnings = []
        pitches_seen = {e[0] for e in pix_obs}
        if len(pitches_seen) < 2:
            warnings.append(
                f"仅 {sorted(pitches_seen)} 单档俯仰有观测：安装偏移与名义"
                "俯仰角退化，自标定常数不可外推到另一档")
        hg_cache = {}
        obs_by_digit = {}          # digit -> [(pitch, head, px, clipped), ...]
        for _pitch, _head, _d, _px, _cl in pix_obs:
            obs_by_digit.setdefault(_d, []).append(
                (int(_pitch), int(_head), _px, bool(_cl)))

        def _hg(pitch, pulse, off, hcm):
            key = (int(pitch), int(pulse), float(off), float(hcm))
            hg = hg_cache.get(key)
            if hg is None:
                hg = GroundHomography.from_pose(
                    (0.0, 0.0), float(hcm), pitch, head_pulse=pulse,
                    pitch_offset_deg=float(off))
                hg_cache[key] = hg
            return hg

        def _aggregate(off, hcm):
            """逐数字机器人系点聚合 → (med, clean, wts, spread, dropped, sel)

            现场实测（2026-09-11 真机布局扫 photo_1789127899~909）：场内木框
            被橙色阈值命中、蓝地垫被蓝色阈值命中，且它们是**未裁切**观测——
            "干净优先的中位数"会被它们带到 40cm 外，整个格阵拟合失败。
            故先按"最大一致簇"取模式，再在簇内干净优先；簇外观测计 dropped。
            sel[d] = 该数字**入选观测在 obs_by_digit[d] 中的下标**，供像素域
            精修复用（精修必须只用同一批内点，否则假阳会把重投影解带飞）。
            """
            med, clean, wts, spread, dropped, sel = {}, set(), {}, {}, {}, {}
            for d, lst in obs_by_digit.items():
                pts = np.array([_hg(pitch, pulse, off, hcm).pixels_to_ground(
                    [px], head_pulse=pulse)[0]
                    for pitch, pulse, px, _cl in lst])
                flags = np.array([cl for _p, _h, _px, cl in lst])
                if len(pts) > 1:
                    dm = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
                    n_in = (dm <= CLUSTER_RADIUS_CM).sum(axis=1)
                    keep = dm[int(np.argmax(n_in))] <= CLUSTER_RADIUS_CM
                else:
                    keep = np.ones(len(pts), dtype=bool)
                n_clean = int(np.count_nonzero(keep & ~flags))
                use_mask = keep & ~flags if n_clean >= 1 else keep
                use = pts[use_mask]
                m = np.median(use, axis=0)
                med[d] = m
                spread[d] = float(np.max(np.linalg.norm(use - m, axis=1)))
                dropped[d] = int(np.count_nonzero(~keep))
                sel[d] = [int(k) for k in np.where(use_mask)[0]]
                if n_clean >= 1:
                    clean.add(d)
                    wts[d] = 1.0
                else:
                    wts[d] = LATTICE_CLIPPED_WEIGHT
            return med, clean, wts, spread, dropped, sel

        votes = {}   # (数字→格)冻结元组 -> [每个成功组合的拟合明细, ...]
        dropped_seen = {}
        for off in np.arange(LAYOUT_OFFSET_MIN_DEG,
                             LAYOUT_OFFSET_MAX_DEG + 1e-9,
                             LAYOUT_OFFSET_STEP_DEG):
            for hcm in np.arange(LAYOUT_HEIGHT_MIN_CM,
                                 LAYOUT_HEIGHT_MAX_CM + 1e-9,
                                 LAYOUT_HEIGHT_STEP_CM):
                med, clean, wts, spread, dropped, sel = _aggregate(off, hcm)
                if len(med) < 7:
                    continue
                for d, n in dropped.items():
                    if n:
                        dropped_seen[d] = max(dropped_seen.get(d, 0), n)
                cells, _info, ranked = lattice_assign(med, weights=wts,
                                                      clean=clean)
                if cells is None or len(cells) != 7 or not ranked:
                    continue
                votes.setdefault(tuple(sorted(cells.items())), []).append({
                    "off": float(off), "hcm": float(hcm), "med": med,
                    "clean": set(clean), "wts": dict(wts),
                    "spread": dict(spread), "sel": sel,
                    "rms_clean": ranked[0]["rms_clean"],
                    "rms_all": ranked[0]["rms_all"],
                    "res_max": ranked[0]["res_max_cm"]})
        if not votes:
            return _fit_failure(
                "全部参数组合均无法一致拟合 33cm 格阵（观测含坏点、标号"
                "约定不符或缺覆盖）", warnings)
        key, combos = max(votes.items(), key=lambda kv: len(kv[1]))
        n_assign = sum(len(v) for v in votes.values())
        if len(combos) < LAYOUT_VOTE_MIN_FRAC * n_assign \
                or len(combos) < LAYOUT_VOTE_MIN_COMBOS:
            others = sorted((len(v) for k, v in votes.items() if k != key),
                            reverse=True)
            return _fit_failure(
                f"参数网格多数票不足（{len(combos)}/{n_assign}，次优 "
                f"{others[:2]}）——前进一步重扫消歧", warnings)
        cells = dict(key)
        off = float(np.median([c["off"] for c in combos]))
        hcm = float(np.median([c["hcm"] for c in combos]))
        # 取最接近中位数的那个组合的几何数据作为基准（可复现、可诊断）
        pick = min(combos, key=lambda c: (
            abs(c["off"] - off) / LAYOUT_OFFSET_STEP_DEG
            + abs(c["hcm"] - hcm) / LAYOUT_HEIGHT_STEP_CM))
        # 像素域五参数联合精修（关键步骤，见 _pixel_pose_calib）：格阵刚性判据
        # 对 (安装偏移, 相机高度) 只有一条退化谷（sim 实测 (18.5°,56cm) 与
        # (19.5°,59cm) 刚体 RMS 都是 0.30cm），而导航投影对谷内位置很敏感
        # （高度差 3cm → GN 残差 16~23px → 定位风暴）。故用"格心↔像素"重投影
        # 残差（与导航同一残差定义）联合解出 (x, y, θ, 偏移, 高度)。
        ds = sorted(pick["med"])
        gs = np.array([grid_cell_center(cells[d]) for d in ds])
        ps = np.array([pick["med"][d] for d in ds])
        ws = np.array([pick["wts"].get(d, 1.0) for d in ds])
        R0, t0, _rms0, _per0 = _rigid_fit_2d(gs, ps, ws)
        pos0 = -R0.T @ t0
        fwd0 = R0.T @ np.array([0.0, 1.0])
        pose0 = np.array([pos0[0], pos0[1], float(np.arctan2(fwd0[0], fwd0[1]))])
        entries = []
        for d, pos_list in pick["sel"].items():
            if d not in cells:
                continue
            for k in pos_list:
                pitch, head, px, cl = obs_by_digit[d][k]
                entries.append((grid_cell_center(cells[d]), px, head, cl, pitch))
        pose5, rms_all_px, med_px = _pixel_pose_calib(entries, pose0, off, hcm)
        pose = None
        if pose5 is not None and med_px is not None \
                and med_px <= PIXEL_CALIB_MED_MAX_PX:
            o2, h2 = float(pose5[3]), float(pose5[4])
            med2, clean2, wts2, spread2, _dr2, sel2 = _aggregate(o2, h2)
            cells2, _i2, ranked2 = lattice_assign(med2, weights=wts2,
                                                  clean=clean2)
            if cells2 == cells and ranked2:
                off, hcm = o2, h2
                pick = {"off": off, "hcm": hcm, "med": med2, "clean": clean2,
                        "wts": wts2, "spread": spread2, "sel": sel2,
                        "rms_clean": ranked2[0]["rms_clean"],
                        "rms_all": ranked2[0]["rms_all"],
                        "res_max": ranked2[0]["res_max_cm"]}
                pose = np.array([pose5[0], pose5[1], pose5[2]])
                warnings.append(
                    f"相机常数经像素域精修：偏移 {off:+.1f}° / 高度 {hcm:.1f}cm"
                    f"（重投影 中位 {med_px:.1f}px / RMS {rms_all_px:.1f}px）")
            else:
                warnings.append("像素域精修后的常数与胜出布局不一致——已忽略"
                                "精修，沿用格阵投票常数")
        else:
            warnings.append(
                "像素域精修未收敛（重投影 中位 "
                f"{'None' if med_px is None else format(med_px, '.1f') + 'px'} "
                f"> {PIXEL_CALIB_MED_MAX_PX:.0f}px）——沿用格阵投票常数，"
                "导航精度可能下降")
        clipped_only = sorted(d for d in cells if pick["wts"].get(d, 1.0) < 1.0)
        if clipped_only:
            warnings.append(
                f"数字 {clipped_only} 无未裁切观测，仅靠裁切质心"
                f"（偏差可达 ~8cm）")
        if dropped_seen:
            warnings.append(
                f"数字 {sorted(dropped_seen)} 有被剔除的离群观测"
                f"（最多 {max(dropped_seen.values())} 帧，疑似场内同色杂物"
                "被误检——木框/地垫）")
        bad_spread = sorted(d for d, s in pick["spread"].items()
                            if s > SPREAD_MAX_CM)
        if bad_spread:
            warnings.append(
                f"数字 {bad_spread} 簇内离散 >{SPREAD_MAX_CM:.0f}cm，观测质量差")
        if 6 in cells.values():
            warnings.append("位置6 被面板占据：与比赛规则冲突——标号约定或"
                            "机位存疑，需现场踩格核对")
        spread_max = max(pick["spread"].values()) if pick["spread"] else 0.0
        info = (f"{len(combos)}/{n_assign} 组合收敛（{len(pick['clean'])} 个"
                f"数字有干净观测，跨帧离散 ≤{spread_max:.1f}cm）")
        # 歧义修复（E）：颜色歧义面板（同位置双色命中 / 中位 H 贴窗口边界）
        # 若形状仲裁没能定案，这里用**格阵共识**复核一次——置信数字先定刚体
        # 变换，再看歧义数字的观测点离"本格"还是"竞争数字那格"更近。
        cells, rep_notes = ambiguity_repair(cells, pick, ambig or {})
        for n in rep_notes:
            warnings.append(n)
            print(f"[布局] 歧义修复: {n}")
        return LatticeFit(cells=cells, info=info, offset_deg=off,
                          cam_height_cm=hcm, pts_robot=dict(pick["med"]),
                          clean=set(pick["clean"]), weights=dict(pick["wts"]),
                          rms_clean_cm=pick["rms_clean"],
                          rms_all_cm=pick["rms_all"],
                          spread_cm=dict(pick["spread"]),
                          res_max_cm=pick["res_max"], warnings=warnings,
                          pose=pose)

    def _pose_bootstrap(self, fit):
        """位姿自举 → 直接写 self.pose，返回 bool

        两条路径：
          1) **优先用像素域精修位姿**（fit.pose，见 _pixel_pose_calib）：它由
             "格心↔像素"重投影最小二乘得到，与导航同一残差定义、同一常数，
             精度最高；
          2) 回退用"机器人系点 ↔ 场地格心"2D 刚体拟合（两系同为 z 轴向上的
             地面系，只差一个刚体变换）。
        两条路都不再解"像素↔格心"单应再 decompose——那条路把裁切质心观测
        （偏差实测 1.3~8.4cm）送进无鲁棒最小二乘，会给出非物理 H
        （历史崩溃："相机未俯视地面（光轴无向下分量）"）。
        自检：残差门、机器人在入口侧、位置落在场地合理范围；任一不满足 → False。
        """
        ds = sorted(fit.pts_robot)
        if len(ds) < 4 or not fit.cells or any(d not in fit.cells for d in ds):
            print("[布局] 位姿自举失败：拟合结果不完整")
            return False
        gs = np.array([grid_cell_center(fit.cells[d]) for d in ds])
        if fit.pose is not None:
            pos = np.asarray(fit.pose[:2], dtype=np.float64)
            th = float(fit.pose[2])
            gate_txt = "像素域精修"
        else:
            ps = np.array([fit.pts_robot[d] for d in ds])
            w = np.array([1.0 if d in fit.clean else LATTICE_CLIPPED_WEIGHT
                          for d in ds])
            R, t, rms_all, per = _rigid_fit_2d(gs, ps, w)
            idx = [k for k, d in enumerate(ds) if d in fit.clean]
            if len(idx) >= 3:
                gate_val = float(np.sqrt(np.mean(per[idx] ** 2)))
                gate_txt = f"刚体残差 干净子集 {gate_val:.1f}cm"
            else:
                gate_val, gate_txt = rms_all, f"刚体残差 全点 {rms_all:.1f}cm"
            if gate_val > POSE_BOOTSTRAP_RMS_MAX_CM:
                print(f"[布局] 位姿自举自检失败：{gate_txt} > "
                      f"{POSE_BOOTSTRAP_RMS_MAX_CM:.0f}cm（布局或观测存疑）")
                return False
            pos = -R.T @ t                  # 机器人系原点的场地坐标（相机地面投影）
            fwd = R.T @ np.array([0.0, 1.0])   # 机器人系 +y（机体前方）的场地方向
            th = float(np.arctan2(fwd[0], fwd[1]))
        if not (POSE_BOOTSTRAP_X_RANGE[0] <= pos[0] <= POSE_BOOTSTRAP_X_RANGE[1]
                and POSE_BOOTSTRAP_Y_RANGE[0] <= pos[1]
                <= POSE_BOOTSTRAP_Y_RANGE[1]):
            print(f"[布局] 位姿自举自检失败：位置 ({pos[0]:.1f},{pos[1]:.1f}) "
                  "越出场地合理范围")
            return False
        if pos[1] >= float(np.min(gs[:, 1])) + GRID_CELL_CM / 2:
            print("[布局] 位姿自举自检失败：机器人不在全部面板的入口侧")
            return False
        self.pose = np.array([pos[0], pos[1], th])
        print(f"[布局] 位姿自举: ({pos[0]:.1f},{pos[1]:.1f}) "
              f"航向{np.degrees(th):.1f}°（{gate_txt}，{len(ds)}点）")
        return True

    # =================================================================
    # 定位：Gauss-Newton 地图定位（格心=地图，动作模型=先验）
    # =================================================================

    def predict_pose(self, action, times=1):
        """按名义动作模型推进位姿预测（每步动作后调用）

        小步转向的名义角用 EMA 在线估计值（地面打滑时它更接近真值）。
        """
        kind, step = ACTION_MODEL.get(action, (None, 0.0))
        if action.endswith("_small_step") and kind == "turn":
            step = (self._small_turn_deg if step > 0 else -self._small_turn_deg)
        for _ in range(max(1, times)):
            th = self.pose[2]
            fwd = np.array([np.sin(th), np.cos(th)])
            right = np.array([np.cos(th), -np.sin(th)])
            if kind == "fwd":
                self.pose[:2] += fwd * step
            elif kind == "lat":
                self.pose[:2] += right * step
            elif kind == "turn":
                self.pose[2] += np.radians(step)

    def _capture_corr(self, pitch, arbitrate=False):
        """拍照+检测，返回 (corr, obs, frame)

        corr 元素 = (格心场地坐标, 观测像素, 采集时头部脉宽, 是否被画幅裁切)：
        - 未裁切：观测取对角线交点 center_px（面板中心的精确投影）；
        - 已裁切：观测取颜色掩膜凸包质心 hull_centroid_px（裁切不变）。
        """
        frame = self.state.capture_frame()
        if frame is None:
            return [], None, None
        obs = self.detector.detect_panels(frame, arbitrate=arbitrate)
        head = self.state.current_head_pulse
        corr = []
        for o in obs:
            if o.digit not in self.digit_cell:
                continue
            px = o.hull_centroid_px if o.clipped else o.center_px
            corr.append((grid_cell_center(self.digit_cell[o.digit]),
                         px, head, o.clipped))
        return corr, obs, frame

    def localize(self, pitch=PITCH_NAV, arbitrate=False, min_panels=1):
        """拍照 → 检测 → >=min_panels 面板 GN 精化位姿（门控后写入 self.pose）

        min_panels 缺省 1：近距时目标面板常是唯一可见面板（其它在身后/视野
        外），此时靠弱先验正则（σ=8cm/8°）约束不可观测方向仍可解——只允许
        >=2 会退化成"头部扫→0 点→恢复"的拍照风暴（实测 64% 定位走该分支）。
        面板不足时做头部左右小扫（零身体位移，零风险）把侧向面板转进视野，
        跨帧合并对应点（每点带自己的头部角）再解。
        返回 (obs_list, frame)；失败返回 None（self.pose 保持预测值）。
        """
        self._loc_count += 1
        corr, obs, frame = self._capture_corr(pitch, arbitrate)
        if len(corr) >= min_panels:
            # 单帧优先：无头部斜视角偏差，门控最严、精度最高
            pose = self._gn_localize(self.pose.copy(), corr, pitch)
            if pose is not None:
                self.pose = pose
                return obs, frame
        # 头部左右小扫合并对应点：贴近段面板常被裁切/出画，单帧可能不足
        merged = list(corr)
        for h in (HEAD_LEFT, HEAD_RIGHT):
            self.state.set_head(h)
            more, _, _ = self._capture_corr(pitch, arbitrate)
            merged.extend(more)
        self.state.set_head(HEAD_CENTER)
        if len(merged) < min_panels:
            print(f"[定位] 合并头部扫后仅 {len(merged)} 个对应点（需>={min_panels}）")
            return None
        pose = self._gn_localize(self.pose.copy(), merged, pitch,
                                 gate_px=GN_PIXEL_GATE_MERGED)
        if pose is None:
            return None
        self.pose = pose
        return obs, frame

    def _gn_localize(self, p0, corr, pitch, gate_px=GN_PIXEL_GATE):
        """以 p0=(x,y,θ) 为先验做惰性多起点 GN 定位

        先解先验起点；RMS 已达标（GN_RMS_OK_PX）就不再试扰动起点，否则补
        ±8cm/±10° 起点取最优。门控：RMS(gate_px) + 相对先验跳变。
        失败返回 None（调用方维持预测位姿）。
        """
        starts = [np.array(p0, dtype=np.float64)]
        for dxy in ((8, 0), (-8, 0), (0, 8), (0, -8)):
            st = np.array(p0, dtype=np.float64)
            st[0] += dxy[0]
            st[1] += dxy[1]
            starts.append(st)
        for dth in (np.radians(10), -np.radians(10)):
            st = np.array(p0, dtype=np.float64)
            st[2] += dth
            starts.append(st)
        best = None
        for st in starts:
            out = self._gn_run(st, corr, pitch)
            if out is None:
                continue
            p, rms = out
            if best is None or rms < best[1]:
                best = (p, rms)
            if best[1] <= GN_RMS_OK_PX:
                break
        if best is None:
            return None
        p, rms = best
        if rms > gate_px:
            print(f"[定位] 残差RMS {rms:.1f}px 超门控({gate_px:.0f}px)，丢弃")
            return None
        jump = float(np.hypot(p[0] - p0[0], p[1] - p0[1]))
        dth = abs(_wrap_angle(p[2] - np.asarray(p0)[2]))
        if jump > GN_JUMP_CM or dth > GN_JUMP_DEG:
            print(f"[定位] 位姿跳变 {jump:.1f}cm/{np.degrees(dth):.0f}° 超预算，丢弃")
            return None
        return p

    def _gn_run(self, p0, corr, pitch):
        """单起点鲁棒 GN：裁切感知观测 + IRLS(Cauchy) + 弱先验正则 + 一轮剔除

        返回 (p, rms)；rms 只统计保留对应点的残差（先验行不计），供门控使用。
        """
        # 兼容 3 元组（旧格式/外部调用）：按未裁切处理
        entries = [t if len(t) == 4 else (t[0], t[1], t[2], False)
                   for t in corr]
        cells = np.array([c for c, _, _, _ in entries], dtype=np.float64)
        dets = np.array([d for _, d, _, _ in entries], dtype=np.float64)
        heads = np.array([h for _, _, h, _ in entries], dtype=np.float64)
        clipped = np.array([b for _, _, _, b in entries], dtype=bool)
        used = np.ones(len(entries), dtype=bool)
        p0a = np.array(p0, dtype=np.float64)
        sig_xy = GN_PRIOR_SIGMA_XY_CM
        sig_th = np.radians(GN_PRIOR_SIGMA_TH_DEG)

        def residuals(pp, mask):
            """对应点残差 (n_used, 2)；预测端按裁切与否选中心/裁切质心"""
            idx = np.where(mask)[0]
            out = np.empty((len(idx), 2))
            for k, i in enumerate(idx):
                if clipped[i]:
                    pc = clipped_quad_centroid(
                        cells[i], pp[0], pp[1], pp[2], pitch, heads[i],
                        pitch_offset_deg=self._pitch_offset_deg,
                        cam_height_cm=self._cam_height_cm)
                    if pc is None:
                        return None
                else:
                    pc = project_ground_to_pixel(
                        cells[i], pp[0], pp[1], pp[2], pitch, heads[i],
                        pitch_offset_deg=self._pitch_offset_deg,
                        cam_height_cm=self._cam_height_cm)[0]
                out[k] = pc - dets[i]
            return out

        def solve_once(p):
            r = residuals(p, used)
            if r is None:
                return None
            r = r.ravel()
            n2 = r.size
            for _it in range(GN_MAX_ITERS):
                J = np.zeros((n2 + 4, 3))
                for j in range(3):
                    eps = [0.5, 0.5, np.radians(0.5)][j]
                    pp = p.copy(); pp[j] += eps
                    r1 = residuals(pp, used)
                    pp = p.copy(); pp[j] -= eps
                    r2 = residuals(pp, used)
                    if r1 is None or r2 is None:
                        return None
                    J[:n2, j] = ((r1 - r2) / (2 * eps)).ravel()
                # 弱先验正则：低观测（1~2 点）时把解拉向运动先验
                J[n2, 0] = 1.0 / sig_xy
                J[n2 + 1, 1] = 1.0 / sig_xy
                J[n2 + 2, 2] = 1.0 / sig_th
                rp = np.concatenate([r,
                                     [(p[0] - p0a[0]) / sig_xy,
                                      (p[1] - p0a[1]) / sig_xy,
                                      (p[2] - p0a[2]) / sig_th,
                                      0.0]])
                # Cauchy 鲁棒权重按"点"给（同一面板 x/y 同权），先验行恒权 1
                per = np.linalg.norm(r.reshape(-1, 2), axis=1)
                w = 1.0 / (1.0 + (per / GN_CAUCHY_C_PX) ** 2)
                w_flat = np.concatenate([np.repeat(w, 2), np.ones(4)])
                try:
                    delta = np.linalg.solve(
                        J.T @ (J * w_flat[:, None]) + 1e-6 * np.eye(3),
                        -(J.T @ (rp * w_flat)))
                except np.linalg.LinAlgError:
                    return None
                delta = np.clip(delta, [-15, -15, -np.radians(20)],
                                [15, 15, np.radians(20)])
                p = p + delta
                p[2] = _wrap_angle(p[2])
                r = residuals(p, used)
                if r is None:
                    return None
                r = r.ravel()
                if np.max(np.abs(delta[:2])) < 0.02 \
                        and abs(delta[2]) < np.radians(0.02):
                    break
            return p, r

        p = np.array(p0a, dtype=np.float64)
        out = solve_once(p)
        if out is None:
            return None
        p, r = out

        # 一轮离群剔除（误检防护）：点数 >=5 才做，剔除后仍须 >=2 点
        per_point = np.linalg.norm(r.reshape(-1, 2), axis=1)
        if used.sum() >= 5 and per_point.max() > GN_OUTLIER_PX:
            idx = np.where(used)[0]
            used[idx[per_point > GN_OUTLIER_PX]] = False
            if used.sum() < 2:
                return None
            out = solve_once(p)
            if out is None:
                return None
            p, r = out
            per_point = np.linalg.norm(r.reshape(-1, 2), axis=1)

        rms = float(np.sqrt(np.mean(per_point ** 2))) if per_point.size else 1e9
        return p, rms

    def target_relative(self, digit):
        """目标格心在机器人系的 (纵向, 横向, 航向差)。右正左负。"""
        x, y, th = self.pose
        fwd = np.array([np.sin(th), np.cos(th)])
        right = np.array([np.cos(th), -np.sin(th)])
        body = np.array([x, y]) - fwd * CAMERA_TO_BODY_FORWARD_CM
        d = grid_cell_center(self.digit_cell[digit]) - body
        forward = float(d @ fwd)
        lateral = float(d @ right)
        bearing = float(np.degrees(np.arctan2(lateral, forward)))
        return forward, lateral, bearing

    # =================================================================
    # 单格导航 FSM：视觉伺服 v3（搜索→对准→接近→低头到达→蹭步确认）
    # 地图路径 v2 保留在 _go_to_panel_map / _approach / _enter（应急开关）
    # =================================================================

    def go_to_panel(self, digit):
        """前往第 digit 块面板；返回 bool（视觉到位并完成蹭步）

        v3（缺省）：全程只用相对量（像素 yaw / 目标框宽 / 低头颜色占比），
        不依赖相机高度与俯仰——地板形变 ±15° 下仍可用。
        v2（VIS_NAV_ENABLED=False）：地图 GN 路径，仅作现场应急回退。

        **硬熔断保证**：本函数一定在"单格预算"内返回（见 _cell_budget_begin /
        _cell_expired）。任何子环节（搜索/对准/接近/低头补步）卡住，都会在
        t_end 或 VIS_CELL_HARD_* 处被 _cell_expired 拦下 → 本格记为未确认
        （results 里 False、cell_trips 记原因）→ 立刻返回 False，不再动作。
        """
        try:
            return self._go_to_panel_v3(digit)
        finally:
            # 真值落点留档（诊断/回归用，不参与控制）：**必须在返回前记**，
            # 因为调用方随后就会走向下一格，事后再取位置已经不是这一格的落点。
            # SimNineGridRobot 有 .pos（真值），真机 RobotState 两者都没有 →
            # 取不到就留 None，对真机零影响。
            self._record_landing(digit)

    def _record_landing(self, digit):
        """把"跑完本格时机器人离目标格格心的距离"记进 self.panel_landing

        真机（无真值）退化为用死推位姿 self.pose 估计，并在日志里标明来源；
        仿真里 state.pos 是真值，因此这个数字可以用来抓"假到达"。
        """
        cell = self.digit_cell.get(digit)
        if cell is None:
            self.panel_landing[digit] = None
            return
        center = np.asarray(grid_cell_center(cell), float)
        truth = getattr(self.state, "pos", None)
        if truth is not None:
            src, pos = "真值", np.asarray(truth, float)[:2]
        else:
            src, pos = "死推", np.asarray(self.pose, float)[:2]
        self.panel_landing[digit] = (float(np.linalg.norm(pos - center)), src)

    def _go_to_panel_v3(self, digit):
        if not VIS_NAV_ENABLED:
            return self._go_to_panel_map(digit)
        t_end = self._cell_budget_begin(digit)
        self._target_seen = False
        self._loc_count = 0
        self._current_digit = digit
        # 仿真接缝：把"当前目标数字"告知 sim，让地板形变可以**按格**注入
        # （见 sim/nine_grid_sim._update_deform 的 step_at_digit 说明）。
        # 真机 RobotState 没有这个方法 → getattr 兜底，对真机零影响。
        setter = getattr(self.state, "set_deform_digit", None)
        if setter is not None:
            setter(digit)
        for attempt in range(RETRY_LIMIT + 1):
            # 双重判据：软预算（旧语义）+ 硬熔断（新增，先到先算）
            if self._cell_expired():
                print(f"[面板{digit}] 单格熔断收手：{self._cell_tripped}")
                return False
            if time.time() > t_end:
                print(f"[面板{digit}] 时间预算耗尽")
                return False
            if self._seek_align_approach(digit, t_end):
                if self._arrive_visual(digit, t_end) \
                        and self._confirm_switch(digit):
                    self._reanchor_pose(digit)
                    if USE_MAP_CORRECTION:
                        self._map_correction()
                    return True
            if self._cell_expired():
                print(f"[面板{digit}] 单格熔断收手：{self._cell_tripped}")
                return False
            print(f"[面板{digit}] 第{attempt + 1}轮未到位")
            if attempt < RETRY_LIMIT:
                self._act("back_one_step", 1)
        return False

    def _seek_align_approach(self, digit, t_end):
        """搜索 → 对准 → 接近（框宽达标即返回 True）

        对准期目标短暂丢失（转过冲/近距窄视野）不算失败：重搜一次再试。
        任一子环节熔断 → 直接返回 False（不在这里再重试，交 go_to_panel 收手）。
        """
        if self._search_target(digit, t_end) is None:
            print(f"[搜索] 未找到面板{digit}（遮挡/光照？）")
            return False
        self._target_seen = True
        for _try in range(2):
            if self._cell_expired():
                return False
            if self._align_visual(digit, t_end) is not None:
                return self._approach_visual(digit, t_end)
            print(f"[对准] 目标{digit}丢失 → 重搜一次")
            if self._search_target(digit, t_end) is None:
                break
        # 目标始终不入视野，但死推说到位了：交给低头颜色判据裁决（抗形变、
        # 不需要看见目标数字本身——色块压过与否是机器人与色块的相对关系）。
        if self._cell_expired():
            return False
        if digit in self.digit_cell:
            fwd, lat, _b = self.target_relative(digit)
            if abs(fwd) <= VIS_ARRIVE_FALLBACK_CM and abs(lat) <= 20.0:
                print(f"[对准] 目标{digit}不可见但死推已到位"
                      f"（纵向{fwd:+.1f} 横向{lat:+.1f}cm）→ 交低头颜色判据")
                self._target_seen = True
                return True
        print(f"[对准] 目标{digit}反复丢失")
        return False

    def _count_frame(self):
        """按当前阶段累计拍照数（诊断；见 run_level 的汇总打印）"""
        self.phase_frames[self.phase] = self.phase_frames.get(self.phase, 0) + 1

    def _target_color(self, digit):
        """数字(1..7) → 检测器颜色名"""
        return ID_TO_COLOR[digit]

    def _see_target(self, digit, pitch=PITCH_NAV, head=None):
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
        head_deg = ((self.state.current_head_pulse - HEAD_CENTER)
                    * SERVO_DEG_PER_US)
        yaw = -(center_x - w / 2.0) / w * CAMERA_FOV_H_DEG + head_deg
        box = float(o.bbox[2])
        if o.clipped:
            box = max(box, float(np.sqrt(max(o.hull_area, 1.0))))
        return o, frame, float(yaw), box

    def _see_target_any(self, digit, pitch=PITCH_NAV):
        """先在当前头部档找；找不到再头部五档扫（零身体位移）。同 _see_target"""
        seen = self._see_target(digit, pitch)
        if seen is not None:
            return seen
        for head in (HEAD_LEFT, HEAD_RIGHT, HEAD_WIDE_LEFT, HEAD_WIDE_RIGHT):
            seen = self._see_target(digit, pitch, head)
            if seen is not None:
                return seen
        self.state.set_head(HEAD_CENTER)
        return None

    def _align_ladder_step(self, why):
        """打转升级阶梯（打转判据的统一出口）：换策略在先，熔断在最后

        1) 弃用小转、改用大转（小转被地面吞掉是已知现场情形）；
        2) 放弃本次对准（返回 None，让调用方走"死推到位→低头到达"或重搜）；
        3) 仍打转 → 熔断本格。
        返回值：True = 调用方可以继续（已换策略）；False = 调用方应 return None。

        为什么不直接熔断（2026-09-13 仿真复核）：标称场景里一次摆动就熔断会白丢
        整格；打转真正要的结果是"别再原地转"，不是"这格不要了"。而"别再原地转"
        另有 VIS_TURN_BUDGET_DEG(360°) 硬预算兜底，所以这条阶梯不会无限升。
        """
        self._align_stall = 0
        self._align_worsen = 0
        self._align_flips = 0
        self._align_ladder += 1
        if self._align_ladder == 1:
            self._small_turn_usable = False
            print(f"[对准] 打转判据（{why}）→ 策略1：弃用小转，改用大转")
            return True
        if self._align_ladder == 2:
            print(f"[对准] 打转判据（{why}）→ 策略2：放弃本次对准"
                  "（交死推/低头路径，不再原地转）")
            return False
        self._trip_cell(f"打转判据连续触发 3 次（{why}，本格已转 "
                        f"{self._cell_turn_cmd_deg:.0f}°）——判定无法收敛")
        return False

    def _search_target(self, digit, t_end):
        """搜索目标：当前朝向头部五档 → 按死推提示转一步 → 后段前进探测

        地图只用来定"往哪转"（±30° 容差足够，死推位姿每格到达后由
        _reanchor_pose 重置，因此不会漂到不可用）。返回 (obs, frame) 或 None。
        """
        self.phase = f"SEARCH{digit}"
        frames = 0
        blind_cm = 0.0
        for rnd in range(VIS_SEARCH_MAX_ROUNDS):
            if time.time() > t_end or frames >= VIS_SEARCH_MAX_FRAMES \
                    or self._cell_expired():
                break
            for head in (HEAD_CENTER, HEAD_LEFT, HEAD_RIGHT,
                         HEAD_WIDE_LEFT, HEAD_WIDE_RIGHT):
                if time.time() > t_end or frames >= VIS_SEARCH_MAX_FRAMES \
                        or self._cell_expired():
                    break
                seen = self._see_target(digit, PITCH_NAV, head)
                frames += 1
                if seen is not None:
                    self.state.set_head(HEAD_CENTER)
                    print(f"[搜索] 面板{digit} 已找到（第{rnd + 1}轮，"
                          f"用掉 {frames} 帧）")
                    return seen[0], seen[1]
            self.state.set_head(HEAD_CENTER)
            # 补一帧**低头档**（2026-09-13 形变场景）：导航档可见地面带在形变后
            # 会整体移走（名义 42° 时 19~184cm，57° 时 5~86cm），目标可能正好
            # 落在导航档之外、低头档之内（0.16~1.47m）。只在导航档整轮没找到时
            # 补 1 帧（真机 ~0.7s + 两次俯仰伺服）：比"整轮五档低头扫"便宜 5 倍。
            self.state.set_pitch(PITCH_DOWN)
            seen_down = self._see_target(digit, PITCH_DOWN, HEAD_CENTER)
            frames += 1
            self.state.set_pitch(PITCH_NAV)
            if seen_down is not None:
                print(f"[搜索] 面板{digit} 低头档找到（第{rnd + 1}轮，"
                      f"用掉 {frames} 帧）")
                return seen_down[0], seen_down[1]
            # 按死推提示转向（提示无效时固定左转）：**按提示大小转足**，
            # 一次转 1~5 个大转步（22~129°）——动作不花帧，只有头部扫花帧，
            # 这样"目标在身后 180°"也能两轮内覆盖到。
            hint = (self.target_relative(digit)[2]
                    if digit in self.digit_cell else 0.0)
            if abs(hint) < 5.0:
                hint = -TURN_LEFT_DEG      # 负 = 目标在左侧 → 左转
            k = int(np.clip(round(abs(hint) / TURN_RIGHT_DEG), 1, 5))
            self._act("turn_right" if hint > 0 else "turn_left", k)
            frames += 1
            # 盲走近目标（死推指引，动作不花帧）：地板形变会把可见地面带整体
            # 推走（±15° → 可见带在 0~50cm 与 19~184cm 之间摆动），此时目标
            # 可能压根不在任何俯仰档的视野里，只能先走近再由视觉接管。
            if digit in self.digit_cell:
                fwd = self.target_relative(digit)[0]
                if fwd > VIS_SEARCH_BLIND_CM:
                    if blind_cm >= VIS_SEARCH_BLIND_TOTAL_CM:
                        print(f"[搜索] 面板{digit} 盲走累计 {blind_cm:.0f}cm 仍未入"
                              f"视野（上限 {VIS_SEARCH_BLIND_TOTAL_CM:.0f}cm）"
                              "→ 停手（不再瞎走，位姿提示已不可信）")
                        break
                    n = int(np.clip(round((fwd - VIS_SEARCH_BLIND_CM) / 4.0
                                          / FORWARD_ONE_STEP_CM),
                                    1, VIS_SEARCH_BLIND_MAX_STEPS))
                    print(f"[搜索] 面板{digit} 未入视野（死推纵向 {fwd:.0f}cm）"
                          f"→ 盲走 {n} 步")
                    self._act("go_forward_one_step", n)
                    blind_cm += n * FORWARD_ONE_STEP_CM
        print(f"[搜索] 面板{digit} 搜索用尽（{frames} 帧）")
        return None

    def _off_cm_of(self, obs, frame, box):
        """跨越偏差（cm）= 目标横向像素偏移 / 框宽 × PANEL_WIDTH_CM

        投影不变量：等于"横向物理偏移 ÷ 面板物理宽"，与相机高度/俯仰/地板形变
        全无关（同 _arrive_visual 的 dx_cm）。
        ⚠️ 只用于**日志/遥测**，不参与分区判据：面板被画幅裁切时框宽只剩可见窄条
        ⇒ 它会系统性高估（现场 74% 观测是裁切的）。详见 VIS_ZONE_MOVE_DEG 说明。
        """
        w = float(frame.shape[1])
        center_x = float((obs.hull_centroid_px if obs.clipped
                          else obs.center_px)[0])
        if box <= 1.0:
            return 0.0
        return (center_x - w / 2.0) / box * PANEL_WIDTH_CM

    def _near_of(self, box, frame):
        """目标"够近"的尺度不变量 = 框宽 / 画幅高（见 VIS_ZONE_MOVE_DEG 说明）

        框宽是投影不变量（∝1/深度），除以画幅高后无量纲 ⇒ 换分辨率、换俯仰档
        都不用改阈值，也不受地板形变影响。远处值小 → 只许旋转；近处值大 → 允许
        横移。这正是用户图里蓝/橙之间那条**水平分界线**的物理含义。
        """
        H = float(frame.shape[0])
        return float(box) / H if H > 0 else 0.0

    def _zone_of(self, yaw, off_cm=0.0, near=1.0):
        """分区判定 → (zone, off_cm)

        zone ∈ {"move" 直行, "lat" 横移, "rot" 旋转}（对应图的绿/蓝/橙）。
        VIS_ZONE_ENABLED=False 时**完全等价于原二值死区判据**（返回 "move"/"rot"，
        从不返回 "lat"），保证默认行为逐位不变。
        """
        if not VIS_ZONE_ENABLED:
            return ("move", off_cm) if abs(yaw) <= VIS_ALIGN_TOL_DEG \
                else ("rot", off_cm)
        if abs(yaw) <= VIS_ZONE_MOVE_DEG:
            return "move", off_cm
        if abs(yaw) <= VIS_ZONE_ROT_DEG and near >= VIS_ZONE_ROT_NEAR:
            return "lat", off_cm
        return "rot", off_cm

    def _align_visual(self, digit, t_end, max_iters=6, first_seen=None):
        """闭环对准（误差源 = 像素 yaw）→ (obs, frame, box) 或 None

        误差来自画面偏移而不是地图方位——形变不影响它。
        **小转角闭环（2026-09-11 真机"转向精度太粗/找下一格异常"重设计）**：
        只要误差在大转分界（VIS_BIG_TURN_DEG）以内，就永远走"小步多次逼近"，
        每轮重新拍帧复测 yaw，规划步数 = floor(|yaw| / max(EMA估计, 下限))。
        为什么不沿用"小转被吞就全局弃用小转、全走大转"：真机日志正是这条
        升级链把整局带偏——弃用小转后只剩 22°/25.7° 量化大转，在 8°（现 3.5°）
        死区外反复过冲 → _big_turn 判"振荡"后干脆**不动** → 对准循环空转
        （日志连打"大转修正 yaw -34.2°"却没有任何动作）→ 丢目标 → 重搜。
        "走到数字 2 之后找数字 3 异常"与"其它异常行动"都出自这里。
        仿真标定的"小转 1.7°/次"只作规划参考（且有下限保护），现场以闭环
        收敛为准：不依赖固定 °/次 常数，估计偏大偏小都由下一轮复测吸收。
        first_seen：调用方刚拍到的那一帧（搜索/接近的观测）可直接复用，
        省一次拍照（真机 ~0.7s/张）。
        """
        self.phase = f"ALIGN{digit}"
        prev_yaw, prev_n = None, 0
        last_act, last_n = None, 0
        for _ in range(max_iters):
            if time.time() > t_end or self._cell_expired():
                return None
            if first_seen is not None:
                seen, first_seen = first_seen, None
            else:
                seen = self._see_target_any(digit)
            if seen is None:
                if last_act is not None:
                    undo = {"turn_left": "turn_right",
                            "turn_right": "turn_left",
                            "turn_left_small_step": "turn_right_small_step",
                            "turn_right_small_step": "turn_left_small_step"}[last_act]
                    print(f"[对准] 目标{digit}丢失：回退 {undo} "
                          f"x{max(1, last_n // 2)} 后重试")
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
                # 仍记录"被地面吞掉"的次数（v2 do_turn 会据此升级大转，
                # 日志诊断也需要），但 v3 对准**不再因为这条就弃用小转**。
                if per < MIN_EFFECTIVE_TURN_DEG:
                    self._small_turn_fail += 1
                    if self._small_turn_fail >= SMALL_TURN_FAIL_BATCHES:
                        self._small_turn_usable = False
                else:
                    self._small_turn_fail = 0
                # ---- 打转判据（2026-09-13 现场"在死区里打转"）----
                # 转过一次之后 |yaw| 反而没变小 → 记一次；连续 VIS_TURN_STALL_TURNS
                # 次都没变小 = 这套闭环在这一格收敛不了（机械步长吃掉误差/打滑/
                # 目标不稳）：先试**一次**大转兜底（小转被地面吞掉是已知现场情形），
                # 再不动就熔断本格——绝不再"再转一次试试"。
                if abs(yaw) < abs(prev_yaw) - 0.5:
                    self._align_stall = 0
                    self._align_worsen = 0
                else:
                    self._align_stall += 1
                    # 方向自检：同号误差被**转得更大**（>VIS_ALIGN_WORSEN_DEG）
                    # = 转向方向反了（左右约定/头部符号）。连续 2 次就翻符号重试；
                    # 翻错也只是再错 2 次，被停转判据兜住。
                    if abs(yaw) > abs(prev_yaw) + VIS_ALIGN_WORSEN_DEG \
                            and (yaw > 0) == (prev_yaw > 0):
                        self._align_worsen += 1
                    else:
                        self._align_worsen = 0
                    if self._align_worsen >= 2:
                        self._align_sign = -self._align_sign
                        self._align_stall = 0
                        self._align_worsen = 0
                        print(f"[对准] 方向自检：同向转向把误差转得更大"
                              f"（{prev_yaw:+.1f}°→{yaw:+.1f}°）→ 反转转向符号"
                              f"（现 {self._align_sign:+.0f}）")
                    elif self._align_stall >= VIS_TURN_STALL_TURNS:
                        if self._align_ladder_step(
                                f"连续{self._align_stall}次转向 |yaw| 无改善"
                                f"（{abs(prev_yaw):.1f}°→{abs(yaw):.1f}°）"):
                            self._big_turn(-yaw * self._align_sign)
                            prev_yaw, prev_n = None, 0
                            continue
                        return None
                # 左右来回摆（误差跨过 0、两边都还在死区外）= 打转的直接证据
                # （现场"在死区莫名其妙打转"的另一半：步长 > 误差，一步就过冲）
                if (yaw > 0) != (prev_yaw > 0) \
                        and abs(yaw) > VIS_ALIGN_TOL_DEG:
                    self._align_flips += 1
                    if self._align_flips > VIS_ALIGN_MAX_FLIPS:
                        if not self._align_ladder_step(
                                f"死区外左右来回摆 {self._align_flips} 次"
                                f"（{prev_yaw:+.1f}°→{yaw:+.1f}°）"):
                            return None
            if abs(yaw) <= VIS_ALIGN_TOL_DEG or (
                    VIS_ZONE_ENABLED and self._zone_of(
                        yaw, self._off_cm_of(obs, frame, box),
                        self._near_of(box, frame))[0] == "move"):
                # 已对准就交棒。VIS_ZONE_ENABLED=False 时后半段短路，判据与原来
                # 逐位相同（abs(yaw) <= VIS_ALIGN_TOL_DEG）。
                print(f"[对准] 已对准（yaw {yaw:+.1f}°，框宽 {box:.0f}px）")
                return obs, frame, box
            # ---- 三档分区的中间档：横移（厘米级，最准）----
            # 仅在 VIS_ZONE_ENABLED=True 时可达；关闭时 _zone_of 只会返回
            # move/rot，这里永不进入 ⇒ 默认行为零变化。
            if VIS_ZONE_ENABLED and self._zone_of(
                    yaw, self._off_cm_of(obs, frame, box),
                    self._near_of(box, frame))[0] == "lat":
                lat = "left_move" if yaw > 0 else "right_move"
                self._act(lat, 1)
                last_act, last_n = lat, 1
                print(f"[对准] 平移档 {lat}（yaw {yaw:+.1f}°，"
                      f"off {self._off_cm_of(obs, frame, box):+.1f}cm）")
                # ★ 必须更新 prev_yaw：打转判据比的是"上一步之后 |yaw| 有没有变小"，
                # 横移是**有效动作**（把目标拉回中央）；不更新会让判据拿两步前的值
                # 比、把横移序列误判成"连续无改善"→ 假熔断。
                prev_yaw, prev_n = yaw, 1
                continue
            # 大转**只作兜底**：① 误差本来就大（>30°）；② 大转振荡记忆未生效时
            # 才用。振荡已发生时不再调 _big_turn（它会直接 return，等于空转），
            # 落到下面的小转闭环——保证每一轮都发出**真实动作**，不会空转。
            if abs(yaw) > VIS_BIG_TURN_DEG and not self._turn_oscillation:
                print(f"[对准] 大转兜底修正 yaw {yaw:+.1f}°")
                # _big_turn 用"右正"约定，故取负；_align_sign 见方向自检
                self._big_turn(-yaw * self._align_sign)
                prev_yaw, prev_n = None, 0
                continue
            # 小转角闭环：floor 保证"规划转角 ≤ 剩余误差"（不计划性过冲）；
            # 估计角有下限保护（见 VIS_SMALL_TURN_MIN_STEP_DEG），
            # 真实收敛靠下一轮复测。
            step = max(self._small_turn_deg, VIS_SMALL_TURN_MIN_STEP_DEG)
            n = int(np.clip(int(np.floor(abs(yaw) / step)),
                            1, SMALL_TURN_MAX_STEPS))
            # yaw > 0 = 目标在画面左侧 → 左转（符号由方向自检标定）
            action = ("turn_left_small_step" if yaw * self._align_sign > 0
                      else "turn_right_small_step")
            self._act(action, n)
            self._last_turn = (action, n)
            last_act, last_n = action, n
            print(f"[对准] 小转{n}次（yaw {yaw:+.1f}°，估计 "
                  f"{self._small_turn_deg:.1f}°/次）")
            prev_yaw, prev_n = yaw, n
        return None

    def _approach_visual(self, digit, t_end, max_iters=60):
        """接近：目标框宽达 VIS_ARRIVE_BOX_PX 即交棒低头段

        距离只用框宽（相对量，抗形变）；卡滞判据用"框宽是否还在涨"
        （参考实现：Δproximity < 5 → 后退一步重识别）。
        """
        self.phase = f"APPROACH{digit}"
        arr = float(VIS_ARRIVE_BOX_PX)
        prev_box = None
        stall = 0
        for _ in range(max_iters):
            if time.time() > t_end or self._cell_expired():
                return False
            seen = self._see_target(digit)
            if seen is None or abs(seen[2]) > VIS_ALIGN_TOL_DEG:
                got = self._align_visual(digit, t_end, first_seen=seen)
                if got is None:
                    print(f"[接近] 目标{digit}丢失，交回搜索")
                    return False
                box = got[2]
            else:
                box = seen[3]
            print(f"[接近] 框宽 {box:.0f}px（交棒 {arr:.0f}px）")
            if box >= arr:
                # 交棒 yaw 只作遥测：死区 12° 下它最大可到 ~12°，
                # 由此产生的横向残余由低头段的厘米级横移纠偏吸收。
                self._cell_handoff_yaw = float(seen[2]) if seen is not None \
                    else None
                if self._cell_handoff_yaw is not None:
                    print(f"[接近] 交棒 yaw 残余 {self._cell_handoff_yaw:+.1f}°"
                          f"（死区 {VIS_ALIGN_TOL_DEG:.0f}°，"
                          "横向残余交低头段纠）")
                return True
            if prev_box is not None and abs(box - prev_box) < VIS_STALL_BOX_PX:
                stall += 1
                if stall >= 2:
                    print("[接近] 前进无效（框宽不涨）：后退一步重识别")
                    self._act("back_one_step", 1)
                    stall, prev_box = 0, None
                    continue
            else:
                stall = 0
            prev_box = box
            if box < 0.6 * arr:
                n = VIS_BATCH_MAX_STEPS
            elif box < 0.85 * arr:
                n = 3
            else:
                n = 1
            self._act("go_forward_one_step", n)
        print("[接近] 迭代上限用尽")
        return False

    def _arrive_visual(self, digit, t_end):
        """低头到达：颜色占比"出现 → 消失"判定压过目标格（尺度无关版）

        判据 = 记录本次接近的占比峰值，峰值过绝对下限后占比跌破峰值的
        VIS_COLOR_DROP_FRAC → 已压过该格。它描述的是"色块从机器人视野里
        由出现到消失"这一**机器人与色块的相对几何关系**，与相机高度/俯仰
        无关 → 抗地板形变。
        **横向纠偏是厘米级的**（2026-09-13 接线修正）：dx/box_w×PANEL_WIDTH_CM
        是投影不变量，换算成横向厘米后用 left/right_move 一步 2.2cm 直接消掉，
        而不是"小转 1 步"（2° 在 35cm 上只挪 1.2cm，纠 7cm 要 6 次＝等于没纠）。
        纠偏**不再限制在"见到颜色之前"**：交棒点死区 12° → 横向残余可达
        35·tan12° ≈ 7cm，这段正是"踩不到微动开关"的来源；只要还在前进前压
        阶段，就允许继续纠（总次数上限 VIS_LAT_MAX_CORRECTIONS）。
        前压总量封顶 VIS_ARRIVE_PRESS_MAX_CM（45cm）：压满仍未判到回落就交
        死推兜底复核，绝不一路压出去。
        **到达证据门（2026-09-13 假到达修复）**：见 VIS_ARRIVE_EVIDENCE_* 常量
        注释。"占比峰值→回落"只是"整帧色像素变少"，必须叠加"本次接近里见过
        一块形状/尺度都像面板的同色区域"，才允许判"压过"；另有与前压量挂钩的
        物理合理性上限 VIS_ARRIVE_PRESS_PLAUSIBLE_CM。被拒的次数与原因进
        遥测——现场靠它区分"视觉没到位"和"门太严"。
        """
        self.phase = f"ARRIVE{digit}"
        self.state.set_pitch(PITCH_DOWN)
        color = self._target_color(digit)
        peak = 0.0
        prev_ratio = None
        lat_fixes = 0
        press_cm = 0.0
        # 到达证据门：峰值帧的"面板级"证据 + 拒因（诚实遥测）
        peak_cover = -1.0       # 占比峰值那一帧的面板 hull / 画幅（-1 = 未测到）
        ev_cover_best = 0.0     # 本次接近里见过的最大 hull/画幅（仅遥测）
        ev_aspect_best = 0.0    # 上述那帧的 h/w（仅遥测）
        ev_reject = 0           # 因证据不足被拒的"回落"次数（诚实遥测）
        ev_reasons = []         # 去重后的拒因
        # 时间预算自适应：剩余时间不足时收缩迭代上限（真机拍照 ~0.7s/张，见
        # VIS_CAPTURE_COST_S；仿真里动作/拍照是瞬时的，因此这条只影响真机行为）
        iters_max = VIS_ARRIVE_MAX_ITERS
        cells_left = max(1, 8 - digit)
        if (self.deadline - time.time()) / cells_left < 90.0:
            iters_max = max(12, int(iters_max * 0.6))
        try:
            for _ in range(iters_max):
                if time.time() > t_end:
                    return False
                self._count_frame()
                self._cell_frames += 1      # 低头段不走 _capture()，这里补记
                frame = self.state.capture_frame()
                if frame is None:
                    return False
                ratio = self.detector.color_ratio(frame, color)
                # 峰值帧也升格（见 VIS_ARRIVE_EVIDENCE_MIN_COVER 注释）：峰值出现
                # 的那一帧才是"面板在画面里最大/最完整"的一帧，它的面板级面积
                # 才是该用来验证"这个峰值真的来自一块面板"的证据。
                if ratio > peak:
                    peak = ratio
                    peak_cover = -1.0
                print(f"[到达] 色占比 {ratio:.4f}（峰值 {peak:.4f}，"
                      f"判据 <{VIS_COLOR_DROP_FRAC * peak:.4f}）")
                # 证据门素材：同一帧里找"最像面板"的同色区域（保留观测里的
                # 最大 hull）。它在后面既给"是否见过面板"作证，也决定能不能
                # 用它的质心做横移纠偏（擦边条不是合格的控制对象）。
                obs = self.detector.detect_panels(frame, colors=[color],
                                                  arbitrate=False,
                                                  drop_border=False)
                frame_area = float(frame.shape[0] * frame.shape[1])
                ev = max(obs, key=lambda x: x.hull_area) if obs else None
                # 证据门素材：本次接近里见过的**最大**面板级同色区域（单调不减，
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
                                    and bh / bw < VIS_ARRIVE_EVIDENCE_MIN_ASPECT)
                if peak == ratio:       # 本帧就是峰值帧：钉住它的面板级证据
                    peak_cover = max(peak_cover, ev_cover_now)
                if peak >= VIS_COLOR_SEEN_MIN \
                        and ratio < VIS_COLOR_DROP_FRAC * peak:
                    # ① 证据门：**峰值那一帧**必须真见过"大到像面板"的同色区域，
                    #    不能只靠整帧色像素变少（后者可能是侧滑出画/别的同色块
                    #    离开视野）。用峰值帧而不是"历史最大"：形变会把整条曲线
                    #    的尺度一起改掉（同一面板在不同俯仰下 box 面积差 30%），
                    #    只有跟着峰值自身的语义走才不引入新的尺度依赖。
                    if peak_cover < VIS_ARRIVE_EVIDENCE_MIN_COVER:
                        ev_reject += 1
                        why = (f"峰值帧面板证据仅占画幅 {peak_cover:.3f}"
                               f"<{VIS_ARRIVE_EVIDENCE_MIN_COVER}")
                        if why not in ev_reasons:
                            ev_reasons.append(why)
                        print(f"[到达] 占比回落但**证据不足**（{why}）"
                              "→ 不判到达（继续前压/兜底）")
                    # ② 物理合理性：前压超过"交棒 + 一格"就不可能是在压目标
                    elif press_cm > VIS_ARRIVE_PRESS_PLAUSIBLE_CM:
                        why = (f"前压{press_cm:.0f}cm>"
                               f"{VIS_ARRIVE_PRESS_PLAUSIBLE_CM:.0f}cm")
                        ev_reject += 1
                        if why not in ev_reasons:
                            ev_reasons.append(why)
                        print(f"[到达] 占比回落但{why}（已越过一整格）"
                              "→ 不判到达（继续前压/兜底）")
                    else:
                        self._arrive_evidence = (
                            f"低头颜色峰值 {peak:.4f}→回落 {ratio:.4f}"
                            f"（<{VIS_COLOR_DROP_FRAC * peak:.4f} 判据），"
                            f"前压{press_cm:.0f}cm 纠横{lat_fixes}次｜"
                            f"证据 峰值帧色块占画幅{peak_cover:.3f}"
                            f"（h/w{ev_aspect_best:.2f}，门"
                            f"{VIS_ARRIVE_EVIDENCE_MIN_COVER}）"
                            f"｜门拒 {ev_reject} 次"
                            + (f"（{'；'.join(ev_reasons)}）"
                               if ev_reasons else ""))
                        print(f"[到达] 颜色由峰值 {peak:.4f} 跌到 {ratio:.4f}"
                              f"（证据 峰值帧色块占画幅{peak_cover:.3f}）"
                              "→ 判定压过目标格")
                        return True
                if ev is not None and lat_fixes < VIS_LAT_MAX_CORRECTIONS:
                    o = ev
                    bw = max(float(o.bbox[2]), 1.0)
                    bh = max(float(o.bbox[3]), 1.0)
                    # 横移控制的对象健全性（2026-09-13 实测）：擦边条（细长）的
                    # 凸包质心不是合理的面板中心估计，拿它做横移会把目标推出
                    # 画幅——阶跃场景的假到达正是"用 1100×380 的擦边条做 2 次
                    # ×3 步横移"。遇到这种对象就**先前进**，让观测变干净再纠横。
                    if ev_strip_now:
                        print(f"[到达] 观测是擦边条（h/w={bh / bw:.2f}"
                              f"<{VIS_ARRIVE_EVIDENCE_MIN_ASPECT}，未裁切）"
                              "→ 跳过横移纠偏，先前压取干净观测")
                        o = None
                    if o is not None:
                        px = o.hull_centroid_px if o.clipped else o.center_px
                        box = max(float(o.bbox[2]), 1.0)
                        dx_cm = ((float(px[0]) - frame.shape[1] / 2.0) / box
                                 * PANEL_WIDTH_CM)
                        if abs(dx_cm) > VIS_LAT_TOL_CM:
                            if VIS_ARRIVE_LAT_VIA_SIDESTEP:
                                step_cm = (RIGHT_MOVE_CM if dx_cm > 0
                                           else LEFT_MOVE_CM)
                                n_lat = int(np.clip(
                                    round(abs(dx_cm) / step_cm),
                                    1, VIS_ARRIVE_LAT_MAX_STEPS))
                                print(f"[到达] 横向 {dx_cm:+.1f}cm → 横移纠偏 "
                                      f"{'右' if dx_cm > 0 else '左'}{n_lat}步"
                                      f"（{lat_fixes + 1}/{VIS_LAT_MAX_CORRECTIONS}）")
                                self._act("right_move" if dx_cm > 0
                                          else "left_move", n_lat)
                            else:
                                print(f"[到达] 横向 {dx_cm:+.1f}cm → 小转纠偏"
                                      f"（{lat_fixes + 1}/{VIS_LAT_MAX_CORRECTIONS}）")
                                self._act("turn_right_small_step" if dx_cm > 0
                                          else "turn_left_small_step", 1)
                            lat_fixes += 1
                            continue
                # 第二个独立判据：**死推到位即停压**（见 VIS_ARRIVE_STOP_FWD_CM）。
                # 它让"踩到格心"这件事不完全依赖颜色曲线的形状——相机俯仰被地板
                # 形变改掉时，颜色判据可能永远不回落（实测前压 44cm 仍不回落），
                # 而死推距离不受俯仰影响。任一判据成立即到位，依据写清哪一条。
                if digit in self.digit_cell:
                    fwd_now, lat_now, _bn = self.target_relative(digit)
                    if abs(fwd_now) <= VIS_ARRIVE_STOP_FWD_CM \
                            and abs(lat_now) <= VIS_ARRIVE_FALLBACK_LAT_CM:
                        ev_txt = (f"峰值帧色块占画幅{peak_cover:.3f}"
                                  f"（h/w{ev_aspect_best:.2f}）")
                        self._arrive_evidence = (
                            f"死推到位（纵向{fwd_now:+.1f} 横向{lat_now:+.1f}cm，"
                            f"前压{press_cm:.0f}cm；色占比峰值{peak:.4f} 未回落｜"
                            f"{ev_txt}｜门拒 {ev_reject} 次）")
                        print(f"[到达] 死推到位（纵向{fwd_now:+.1f} "
                              f"横向{lat_now:+.1f}cm）→ 判定到达")
                        return True
                # 步长自适应：占比**还在涨**时用 3 步批量（6cm）赶路，一旦不再涨
                # （到峰/过峰）改单步（2cm）精停。实测：用"与峰值比"判据会
                # 在爬升段误判成"接近峰值"而一路 2cm 挪（占比 0.06→0.13 花了
                # 18 次迭代＝低头段占了全程 61% 的拍照）。
                rising = prev_ratio is None or ratio >= prev_ratio
                prev_ratio = ratio
                press_steps = (VIS_ARRIVE_PRESS_COARSE_STEPS if rising
                               else VIS_ARRIVE_PRESS_FINE_STEPS)
                if press_cm + press_steps * FORWARD_ONE_STEP_CM \
                        > VIS_ARRIVE_PRESS_MAX_CM:
                    print(f"[到达] 前压已封顶 {press_cm:.0f}cm"
                          f"（{VIS_ARRIVE_PRESS_MAX_CM:.0f}cm）且占比未回落"
                          " → 交死推兜底复核")
                    break
                press_cm += press_steps * FORWARD_ONE_STEP_CM
                self._act("go_forward_one_step", press_steps)
            print(f"[到达] 迭代/前压用尽（峰值占比 {peak:.4f}，"
                  f"已前压 {press_cm:.0f}cm）")
            # 兜底：视觉判据没走完（例如峰值后占比掉得不够）时，用**死推距离**
            # 复核——位姿每格到达后已重置，短程死推（≤1格）精度足够。
            fwd, lat, _b = self.target_relative(digit)
            if abs(fwd) <= VIS_ARRIVE_FALLBACK_CM \
                    and abs(lat) <= VIS_ARRIVE_FALLBACK_LAT_CM:
                self._arrive_evidence = (
                    f"死推兜底（纵向{fwd:+.1f} 横向{lat:+.1f}cm，"
                    f"前压{press_cm:.0f}cm 后占比未回落，峰值{peak:.4f}）")
                print(f"[到达] 死推距离兜底判定到位（纵向{fwd:+.1f} "
                      f"横向{lat:+.1f}cm）")
                return True
            return False
        finally:
            self.state.set_pitch(PITCH_NAV)

    def _confirm_switch(self, digit):
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
            loc = self.localize(PITCH_DOWN, arbitrate=True)
            if loc:
                for o in loc[0]:
                    if o.digit == digit:
                        verdict = "一致" if o.model_digit == digit \
                            else "仍冲突（以颜色为准，赛后人工核对）"
                        print(f"[复核] 格{cell} 近距SVM={o.model_digit}"
                              f"({o.model_conf:.2f}) vs 颜色={digit}——{verdict}")
        self.state.act("stand")
        # 落点残差：位姿此刻仍是死推值（_reanchor_pose 在返回后才重置），
        # 所以这是"机器人与期望格心的差距"的独立估计，不是自证。
        resid = None
        if digit in self.digit_cell:
            fwd, lat, _b = self.target_relative(digit)
            resid = float(np.hypot(fwd, lat))
            self._cell_arrive_resid_cm = resid
        shaky = "  ⚠ 残差偏大（>半格），按钮可能没压到" \
            if (resid is not None and resid > GRID_CELL_CM / 2) else ""
        resid_txt = "—（无格心映射）" if resid is None else f"{resid:.1f}cm"
        print(f"[确认] 面板{digit} 已按格心压过｜依据: {self._arrive_evidence or '无'}"
              f"｜落点残差 {resid_txt}{shaky}")
        print("[确认] 微动开关状态 Pi 侧不可读——是否触发以场地计分为准")
        return True

    def _reanchor_pose(self, digit):
        """到达后把死推位姿锚到该格格心（消除逐格漂移，保搜索提示可用）

        航向保持死推值（视觉对准已把目标对正，残余误差只影响粗提示）。
        位置断言来自"到达判定成立"（机器人压在该色块上），同时打印**残差**
        （到达时的死推距离）——它是"这次锚定有多可信"的唯一现场证据。
        试过"残差超半格就不锚定"：标称场景直接打挂（面板6 未确认）——因为标称
        场景里残差大恰恰是**死推漂移**造成的，而锚定正是修它的手段。所以锚定
        照旧，但把残差写进遥测，让现场一眼看出哪一格的"到达"不可信。
        """
        if digit not in self.digit_cell:
            return
        resid = self._cell_arrive_resid_cm
        c = grid_cell_center(self.digit_cell[digit])
        self.pose = np.array([c[0], c[1], self.pose[2]])
        print(f"[锚定] 死推位姿重置到格{self.digit_cell[digit]}"
              f"（{c[0]:.0f},{c[1]:.0f}"
              f"{'' if resid is None else f'，到达残差 {resid:.1f}cm'}）")

    def _map_correction(self):
        """可选的地图 GN 校正：成功才采纳，失败只打一行日志

        地板形变会让相机常数失配 → 残差超门控。这是**预期**行为而非错误：
        因此这里静默跳过，绝不触发恢复链（v2 的"定位风暴"根因）。
        """
        loc = self.localize(PITCH_DOWN)
        if loc is None:
            print("[校正] 地图校正跳过（残差超门控/面板不足——形变下属预期）")
            return
        print(f"[校正] 地图校正采纳: ({self.pose[0]:.1f},{self.pose[1]:.1f}) "
              f"{np.degrees(self.pose[2]):.1f}°")

    def _go_to_panel_map(self, digit):
        """v2 地图 GN 路径（VIS_NAV_ENABLED=False 时的应急回退）"""
        t_end = min(time.time() + TARGET_TIME_BUDGET_S, self.deadline)
        self._target_seen = False
        self._loc_count = 0
        self._current_digit = digit
        self.phase = f"SEEK{digit}"
        for attempt in range(RETRY_LIMIT + 1):
            if time.time() > t_end:
                print(f"[面板{digit}] 时间预算耗尽")
                return False
            if self._approach(digit, t_end) and self._enter(digit, t_end):
                return True
            if self._loc_count > LOC_BUDGET_PER_TARGET:
                print(f"[面板{digit}] 定位次数 {self._loc_count} 超软护栏"
                      f"（{LOC_BUDGET_PER_TARGET}）——真机时间预算可能不够")
            if attempt < RETRY_LIMIT:
                print(f"[面板{digit}] 未到位，退格重试 {attempt + 1}/{RETRY_LIMIT}")
                self._act("back_one_step", 2)
        return False

    def _approach(self, digit, t_end):
        """前视闭环接近：对准→横偏→前进，至 MID_DIST 内交棒 ENTER

        卡滞判据用"本次定位 vs 上次动作前的定位"，不再在动作后额外补一帧
        定位（旧实现每次动作 2 次定位，真机拍照 ~0.7s/张也扛不住）。
        """
        self.phase = "APPROACH"
        stall = 0
        prev_fwd = None
        fails = 0
        while time.time() < t_end:
            loc = self.localize(PITCH_NAV)
            if loc is None:
                fails += 1
                if not self._recover(fails):
                    return False
                continue
            fails = 0
            obs, _frame = loc
            if any(o.digit == digit for o in obs):
                self._target_seen = True
            fwd, lat, bearing = self.target_relative(digit)
            print(f"[接近] 纵向{fwd:+.1f} 横向{lat:+.1f} 航向差{bearing:+.1f}° (cm/度)")

            # 卡滞守卫：与上一次"动作前"定位比较（零额外拍照）
            if prev_fwd is not None and abs(bearing) <= ALIGN_TOL_DEG \
                    and abs(lat) <= LAT_TOL_CM:
                if prev_fwd - fwd < STALL_EPS_CM:
                    stall += 1
                    if stall >= STALL_CONSEC:
                        print(f"[接近] 连续{STALL_CONSEC}次改善<{STALL_EPS_CM}cm，"
                              "判定卡滞：后退一步脱困")
                        self._act("back_one_step", 1)
                        stall = 0
                        prev_fwd = None
                        continue
                else:
                    stall = 0

            if abs(bearing) > ALIGN_TOL_DEG:
                if self._turn_oscillation \
                        and abs(bearing) < BIG_TURN_OSC_ACCEPT_DEG:
                    # 大转量化振荡：接受残余航向差，靠横移纠偏继续接近
                    pass
                else:
                    if self._turn_oscillation:
                        self._turn_oscillation = False  # 误差已大，重新允许转向
                    self.do_turn(bearing, digit, fwd)
                    prev_fwd = None
                    continue
            if abs(lat) > LAT_TOL_CM and fwd < 2 * FAR_DIST_CM:
                self.do_sidestep(lat)
                prev_fwd = None
                continue
            if fwd <= MID_DIST_CM:
                return True  # 交棒 ENTER

            prev_fwd = fwd
            # 前进：一律 go_forward_one_step（本场地禁用 go_forward）。
            # 远距+对准良好才小批量，上限 BATCH_MAX_STEPS 步（3×2cm=6cm）。
            if fwd > FAR_DIST_CM and abs(bearing) <= BATCH_MAX_ANGLE_DEG \
                    and abs(lat) <= LAT_TOL_CM:
                n = min(BATCH_MAX_STEPS,
                        max(1, int((fwd - MID_DIST_CM) // FORWARD_ONE_STEP_CM)))
            else:
                n = 2 if fwd > 2 * MID_DIST_CM else 1
            self._act("go_forward_one_step", n)
        print("[接近] 时间预算耗尽")
        return False

    def _enter(self, digit, t_end):
        """低头精细进入：横移+单步，至中心进入脚下判定带

        纵向 < ENTER_BEARING_GATE_CM 后不再按 bearing 转向：目标中心几乎
        在脚下时方位角病态（实测 2cm 处可跳到 24°），转向只会触发
        "退格再转"空耗；此时只用横移纠偏。
        """
        self.phase = "ENTER"
        self.state.set_pitch(PITCH_DOWN)
        stall = 0
        prev_fwd = None
        fails = 0
        try:
            while time.time() < t_end:
                loc = self.localize(PITCH_DOWN)
                if loc is None:
                    # 低头视野窄可能面板不足：借前视补一次
                    self.state.set_pitch(PITCH_NAV)
                    loc = self.localize(PITCH_NAV)
                    self.state.set_pitch(PITCH_DOWN)
                if loc is None:
                    fails += 1
                    if fails >= ENTER_FAIL_LIMIT:
                        print("[进入] 连续定位失败，退格重试")
                        return False
                    self._act("back_one_step", 1)
                    continue
                fails = 0
                obs, _frame = loc
                if any(o.digit == digit for o in obs):
                    self._target_seen = True
                fwd, lat, bearing = self.target_relative(digit)
                print(f"[进入] 纵向{fwd:+.1f} 横向{lat:+.1f} 航向差{bearing:+.1f}° (cm/度)")

                if abs(bearing) > ENTER_ALIGN_TOL_DEG \
                        and fwd > ENTER_BEARING_GATE_CM:
                    self.do_turn(bearing, digit, fwd)
                    prev_fwd = None
                    continue
                if abs(lat) > CONFIRM_LAT_MAX_CM:
                    self.do_sidestep(lat)
                    prev_fwd = None
                    continue
                if fwd > ENTER_TOL_CM:
                    # 卡滞守卫（仅前进迭代，与上一次动作前比较）
                    if prev_fwd is not None \
                            and prev_fwd - fwd < STALL_EPS_CM:
                        stall += 1
                        if stall >= STALL_CONSEC + 1:
                            print("[进入] 卡滞：后退一步脱困")
                            self._act("back_one_step", 1)
                            stall = 0
                            prev_fwd = None
                            continue
                    else:
                        stall = 0
                    prev_fwd = fwd
                    self._act("go_forward_one_step", 1)
                elif fwd < ENTER_MIN_CM:
                    print("[进入] 已过冲：回蹭")
                    self._act("back_one_step", 1)
                    prev_fwd = None
                else:
                    return self._confirm(digit)
            print("[进入] 时间预算耗尽")
            return False
        finally:
            self.state.set_pitch(PITCH_NAV)

    def _confirm(self, digit):
        """到达确认：位姿"脚下覆盖中心"主判 + 接近段检出过目标 + 蹭步压开关

        为什么不用"低头色占比"主判：pitch 1040 时画面下沿对应地面 15.8cm，
        站在面板中心时面板远边仅 14cm——目标面板整个在视野外（实测占比
        0.003，旧阈值 0.10 永远达不到）。参考方案的"出现→压过→消失"是
        过渡判据，不是"站上仍可见"。
        判据：位姿（相机在面板中心 ±4cm 内）+ 接近段确实检出过目标数字面板，
        再用 back(3.2cm)+forward(2.0cm) 蹭步确保压到微动开关。
        """
        self.phase = "CONFIRM"
        fwd, lat, _bearing = self.target_relative(digit)
        on_panel = abs(fwd) <= ENTER_TOL_CM and abs(lat) <= LAT_TOL_CM
        if not on_panel:
            print(f"[确认] 位姿未覆盖中心（纵向{fwd:+.1f} 横向{lat:+.1f}cm）")
            return False
        if not self._target_seen:
            print(f"[确认] 位姿已到中心但接近段未检出面板{digit}"
                  "（颜色阈值/遮挡？），交回导航重试")
            return False

        # 蹭步：后退3.2 + 前进2.0，来回覆盖中心点，确保触发微动开关
        self._act("back_one_step", 1)
        self._act("go_forward_one_step", 1)

        # 仲裁冲突格：低头近距复核一次（只记录，不改主判）
        cell = self.digit_cell.get(digit)
        if cell in self.cell_conflict:
            loc = self.localize(PITCH_DOWN, arbitrate=True)
            if loc:
                for o in loc[0]:
                    if o.digit == digit:
                        verdict = "一致" if o.model_digit == digit \
                            else "仍冲突（以颜色为准，赛后人工核对）"
                        print(f"[复核] 格{cell} 近距SVM={o.model_digit}"
                              f"({o.model_conf:.2f}) vs 颜色={digit}——{verdict}")
        self.state.act("stand")
        # v2 路径的"到达"本身就是位姿判据（上面 on_panel 已卡 ±4cm）：
        # 记残差 0 让 _reanchor_pose 的锚定门放行（保持 v2 原有行为）。
        self._cell_arrive_resid_cm = float(np.hypot(fwd, lat))
        print(f"[确认] 面板{digit} 位姿覆盖中心（纵向{fwd:+.1f} "
              f"横向{lat:+.1f}cm）——到位；微动开关状态 Pi 侧不可读")
        return True

    # =================================================================
    # 运动原语（白名单 + 自适应转向 + 预测推进）
    # =================================================================

    def _act(self, action, times=1):
        """动作执行 + 位姿预测推进（GN 先验的基础）

        白名单硬门：本场地禁用的大步幅动作（go_forward / go_forward_fast）
        直接拒绝，避免后续改动误用导致摔倒。

        **转向护栏（2026-09-13 真机"一直转 → 转出场地"）**：转向是最便宜的
        动作（不拍照、不耗帧），旧实现没有任何"转够了就停"的概念，对准与搜索
        可以互相接管、在死区里无限转下去。这里累计单格**命令**转角，超预算就
        **拒绝执行**这次转向并熔断本格（安全项，宁可本格记未确认）。
        **离场护栏**：动作执行后统一检查死推位姿是否越出场地外框。
        """
        if action in DISABLED_ACTIONS:
            raise ValueError(
                f"本关卡禁用动作 {action}（小面板+打滑地板易摔倒）；"
                f"前进请用 go_forward_one_step 小批量")
        if action not in ACTION_MODEL:
            raise ValueError(f"未登记的动作: {action}（动作白名单见 ACTION_MODEL）")
        if self._field_guard():
            # 已经判定越界（整局收手）：拒绝再动，只保持站立
            return
        kind, delta = ACTION_MODEL[action]
        req = abs(float(delta)) * int(times)
        if kind == "turn" \
                and self._cell_turn_cmd_deg + req > VIS_TURN_BUDGET_DEG:
            self._trip_cell(
                f"单格累计命令转角 {self._cell_turn_cmd_deg:.0f}°+{req:.0f}° "
                f"超上限 {VIS_TURN_BUDGET_DEG:.0f}°（防原地打转/转出场地）")
            return
        self.state.act(action, times=times)
        # 单格动作计数（硬熔断第三重 + 遥测）：按**实际执行的原语数**记
        # （times 批量算 times 次）——2026-09-13 发现这个计数从来没被累加过，
        # 也就是说 VIS_CELL_HARD_ACTIONS（300）此前是死判据，遥测里恒显示 0。
        self._cell_actions += max(1, int(times))
        self.predict_pose(action, times)
        if kind == "turn":
            self._cell_turn_cmd_deg += req
        else:
            # 期间发生平移：旧转向不可盲目撤销，大转振荡记忆也失效
            self._last_turn = None
            self._last_big_turn = None
        self._field_guard()

    def do_turn(self, bearing, digit, fwd_cm=None):
        """按航向差转向（右正）。近目标先退格再转（防面板边缘转向）。"""
        if fwd_cm is not None and abs(fwd_cm) < TURN_NEAR_LIMIT_CM:
            print("[转向] 近目标先退格再转")
            self._act("back_one_step", 2)

        if abs(bearing) > BIG_TURN_THRESHOLD_DEG or not self._small_turn_usable:
            self._big_turn(bearing)
            return
        # 小转批量（在线估计单次角度）
        n = int(np.clip(round(abs(bearing) / self._small_turn_deg),
                        1, SMALL_TURN_MAX_STEPS))
        action = "turn_right_small_step" if bearing > 0 else "turn_left_small_step"
        self._act(action, n)
        self._last_turn = (action, n)
        # 在线估计实际单次转角（EMA）；定位失败则保持旧估计
        loc = self.localize(PITCH_NAV)
        if loc is not None:
            new_bearing = self.target_relative(digit)[2]
            # 单次实际角估计：过冲时用"已转过+过冲量"折算（否则 per_step 变负，
            # EMA 崩向 0 → 下一轮请求更多小转步 → 更过冲，恶性循环）。
            if abs(new_bearing) < abs(bearing):
                per_step = (abs(bearing) - abs(new_bearing)) / n
            else:
                per_step = (abs(bearing) + abs(new_bearing)) / n
            per_step = float(np.clip(per_step, 0.3, 10.0))
            alpha = (TURN_EMA_ALPHA_FAST
                     if self._turn_updates < TURN_EMA_FAST_UPDATES
                     else TURN_EMA_ALPHA)
            self._turn_updates += 1
            self._small_turn_deg = ((1 - alpha) * self._small_turn_deg
                                    + alpha * per_step)
            if per_step < MIN_EFFECTIVE_TURN_DEG:
                self._small_turn_fail += 1
                if self._small_turn_fail >= SMALL_TURN_FAIL_BATCHES:
                    print(f"[转向] 小转连续{self._small_turn_fail}批被地面吞掉"
                          f"（{per_step:.2f}°/次），升级大转")
                    self._small_turn_usable = False
                    self._big_turn(new_bearing)
            else:
                self._small_turn_fail = 0
                print(f"[转向] 小转{n}次 剩余{new_bearing:+.1f}° "
                      f"估计{self._small_turn_deg:.1f}°/次")

    def _big_turn(self, bearing):
        """大步转向兜底：量化到 22°/25.7°，过冲由下一轮闭环吸收

        防振荡：大转步长（22/25.7°）大于死区（8°）的两倍时，小误差会在
        ±死区外来回转（9°→-13°→9°…）。若上一次大转后误差符号翻转且幅度
        没有明显改善，停止转向，改用"带航向偏置的接近"（横偏交给 sidestep）。
        """
        if abs(bearing) <= ALIGN_TOL_DEG:
            return
        if self._last_big_turn is not None:
            last_sign, last_abs = self._last_big_turn
            if last_sign * bearing < 0 and last_abs <= abs(bearing) + 1.0:
                print(f"[转向] 大转振荡（{last_abs:.0f}°→{abs(bearing):.0f}°），"
                      "停止转向，改带航向偏置接近")
                self._turn_oscillation = True
                return
        step = TURN_RIGHT_DEG if bearing > 0 else TURN_LEFT_DEG
        n = int(np.clip(round(abs(bearing) / step), 1, 3))
        action = "turn_right" if bearing > 0 else "turn_left"
        self._act(action, n)
        self._last_turn = (action, n)
        self._last_big_turn = (1 if bearing > 0 else -1, abs(bearing))
        print(f"[转向] 大转 {action} x{n}（目标差 {bearing:+.1f}°）")

    def do_sidestep(self, lat):
        """横移纠偏。右正。"""
        n = int(np.clip(round(abs(lat) / RIGHT_MOVE_CM), 1, 3))
        action = "right_move" if lat > 0 else "left_move"
        self._act(action, n)
        print(f"[横移] {action} x{n}（横向差 {lat:+.1f}cm）")

    def _recover(self, fails):
        """定位丢失恢复（按失败次数逐级升级）；返回 False = 放弃本段

        每级动作后由调用方重试定位：成功则 fails 归零；仍失败则升级下一级。
        fails==2 的"朝目标转"是关键：目标在身后时，转身过程中可能出现
        "整帧看不到任何面板"的方位（实测格心朝南 0 个对应点），此时只能用
        运动先验位姿（动作模型推算，仍可用）算目标方位，把视角转回有面板
        的方向——纯头部扫（±40.5°）救不了。
        """
        self.phase = f"RECOVER{fails}"
        if fails > RECOVER_LIMIT:
            print(f"[恢复] 连续 {fails} 次定位失败，放弃本段")
            return False
        if fails == 1:
            print("[恢复] 低头补扫")
            self.state.set_pitch(PITCH_DOWN)
            loc = self.localize(PITCH_DOWN)
            self.state.set_pitch(PITCH_NAV)
            if loc is not None:
                print("[恢复] 低头补扫成功")
        elif fails == 2:
            digit = getattr(self, "_current_digit", None)
            if digit is not None and digit in self.digit_cell:
                bearing = self.target_relative(digit)[2]
                if abs(bearing) > ALIGN_TOL_DEG:
                    print(f"[恢复] 按运动先验朝目标转 {bearing:+.0f}° 后重扫")
                    self._big_turn(bearing)
                else:
                    print("[恢复] 按运动先验已朝目标，右转 22° 换视角重扫")
                    self._act("turn_right", 1)
            else:
                print("[恢复] 右转 22° 换视角重扫")
                self._act("turn_right", 1)
        elif fails == 3:
            print("[恢复] 后退一步重扫")
            self._act("back_one_step", 1)
        elif fails == 4:
            # 运动先验盲走一步（不依赖定位，零解算风险），把视角带离死区
            print("[恢复] 运动先验盲走一步后重扫")
            self._act("go_forward_one_step", 1)
        elif fails == 5 and self._last_turn is not None:
            # 大转过冲可能把场地转出视野——撤销上次转向回到可定位姿态
            action, n = self._last_turn
            undo = {"turn_left": "turn_right", "turn_right": "turn_left",
                    "turn_left_small_step": "turn_right_small_step",
                    "turn_right_small_step": "turn_left_small_step"}[action]
            print(f"[恢复] 撤销上次转向 {action}x{n} → {undo}x{n}")
            self._act(undo, n)
            self._last_turn = None
        return True


# =====================================================================
# 投影与工具
# =====================================================================

# 画幅矩形（裁切预测用；与 camera_config 的原生分辨率一致）
_IMAGE_RECT = np.array([[0.0, 0.0], [CAMERA_WIDTH, 0.0],
                        [CAMERA_WIDTH, CAMERA_HEIGHT],
                        [0.0, CAMERA_HEIGHT]], dtype=np.float32)


def _camera_rotation(th_rad, pitch_pulse, head_pulse,
                     pitch_offset_deg=CAM_PITCH_MOUNT_OFFSET_DEG):
    """世界->相机旋转矩阵（行 = 相机三轴在世界系方向）

    有效俯仰 = 舵机名义角 + 安装下俯偏移（2026-09-11 现场实测：名义 27°
    实际 ≈45°+，见 camera_config.CAM_PITCH_MOUNT_OFFSET_DEG）。
    头部偏航：脉宽左转(>1500)为正，相机方位角 = 机体航向 − 头部角
    （与 core/robot_core.pulse_to_angle、单测验证的补偿模型一致）。
    """
    a = np.radians((1500 - pitch_pulse) * SERVO_DEG_PER_US + pitch_offset_deg)
    f = th_rad - np.radians((head_pulse - HEAD_CENTER) * SERVO_DEG_PER_US)
    sa, ca = np.sin(a), np.cos(a)
    sf, cf = np.sin(f), np.cos(f)
    return np.vstack([np.array([cf, -sf, 0.0]),                    # 图像右
                      np.array([-sa * sf, -sa * cf, -ca]),         # 图像下
                      np.array([ca * sf, ca * cf, -sa])])          # 视轴


def project_ground_to_pixel(pts_ground, x, y, th_rad, pitch_pulse,
                            head_pulse=HEAD_CENTER,
                            pitch_offset_deg=CAM_PITCH_MOUNT_OFFSET_DEG,
                            cam_height_cm=CAM_HEIGHT_CM):
    """格心(场地系 cm) → 像素：位姿 (x,y,θ) + 俯仰/头部档 + 真实内参/畸变"""
    R = _camera_rotation(th_rad, pitch_pulse, head_pulse, pitch_offset_deg)
    pts = np.atleast_2d(np.asarray(pts_ground, dtype=np.float64))
    if pts.shape[1] == 2:
        pts = np.column_stack([pts, np.zeros(len(pts))])  # 地面 z=0
    diff = pts - np.array([x, y, cam_height_cm])  # 相机光心高度
    p_cam = (R @ diff.T).T
    norm = p_cam[:, :2] / p_cam[:, 2:3]
    pts3 = np.column_stack([norm, np.ones(len(norm))]).reshape(-1, 1, 3)
    pix, _ = cv2.projectPoints(pts3.astype(np.float32), np.zeros(3),
                               np.zeros(3), CAMERA_INTRINSIC,
                               CAMERA_DISTORTION)
    return pix[:, 0, :]


def clipped_quad_centroid(center_xy, x, y, th_rad, pitch_pulse,
                          head_pulse=HEAD_CENTER, half_cm=PANEL_HALF_CM,
                          pitch_offset_deg=CAM_PITCH_MOUNT_OFFSET_DEG,
                          cam_height_cm=CAM_HEIGHT_CM):
    """裁切感知预测：面板四角投影 → 与画幅求交 → 交多边形面积质心

    与检测端 hull_centroid_px 同定义（都是"可见区域"的面积质心），
    因此被画幅裁切时仍然成立；未裁切时也成立（只是精度略低于中心模型）。
    面板角点落到相机背后（z<=0.05）或投影非有限时返回 None（该起点判无效）；
    面板在画外时返回"四角钳到画幅后的均值"，保持数值雅可比连续。
    """
    if cv2 is None:
        return None
    R = _camera_rotation(th_rad, pitch_pulse, head_pulse)
    cx, cy = float(center_xy[0]), float(center_xy[1])
    corners = np.array([[cx - half_cm, cy - half_cm],
                        [cx + half_cm, cy - half_cm],
                        [cx + half_cm, cy + half_cm],
                        [cx - half_cm, cy + half_cm]], dtype=np.float64)
    pts = np.column_stack([corners, np.zeros(4)])
    p_cam = (R @ (pts - np.array([x, y, cam_height_cm])).T).T
    if np.any(p_cam[:, 2] <= 0.05):
        return None
    norm = p_cam[:, :2] / p_cam[:, 2:3]
    pts3 = np.column_stack([norm, np.ones(len(norm))]).reshape(-1, 1, 3)
    pix, _ = cv2.projectPoints(pts3.astype(np.float32), np.zeros(3),
                               np.zeros(3), CAMERA_INTRINSIC,
                               CAMERA_DISTORTION)
    pix = pix[:, 0, :]
    if not np.all(np.isfinite(pix)):
        return None
    # 画外角点投影可达 1e5+（面板近边贴相机平面），这是合法输入——
    # 交给 intersectConvexConvex 裁到画幅即可；只防 float32 溢出。
    pix = np.clip(pix, -1e6, 1e6)
    area, inter = cv2.intersectConvexConvex(pix.astype(np.float32),
                                            _IMAGE_RECT)
    if inter is not None and len(inter) >= 3 and area > 1e-9:
        m = cv2.moments(inter)
        if m["m00"] > 1e-9:
            return np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])
    # 投影与画幅无交集（或退化）：返回"四角钳到画幅后的均值"作极限。
    # 必须连续——数值雅可比对位姿扰动 ±0.5cm/±0.5° 时，边缘处的交集可能
    # 一步消失；若直接返回 None 会让整个起点解算失败（实测某位姿 6 点全丢）。
    return np.clip(pix, [0.0, 0.0],
                   [CAMERA_WIDTH, CAMERA_HEIGHT]).mean(axis=0)


def _wrap_angle(a):
    """角度弧度归一化到 (-π, π]"""
    while a > np.pi:
        a -= 2 * np.pi
    while a <= -np.pi:
        a += 2 * np.pi
    return a


# =====================================================================
# 关卡入口（main.py 注册）
# =====================================================================

def run_level(state: RobotState):
    """数字宫格关卡入口：python main.py nine_grid"""
    level = NineGridLevel(state)
    return level.run_level()
