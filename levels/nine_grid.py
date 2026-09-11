# -*- coding: utf-8 -*-
"""
数字宫格关卡（levels/nine_grid.py）

任务：1m×1m 九宫格（3x3，左下位置6恒空），7 块数字面板(1~7)随机布局，
机器人按 1→7 顺序依次到达各面板中心（踩中微动开关），限时 15 分钟。
评分：顺序正确每格 10 分（70），限时完成 +30。

方案要点（2026-09-08 定稿，详见 docs/关卡算法/数字宫格攻略.md）：
  视觉  颜色主判（HSV 七色，1红..7粉）+ HOG+SVM 黑色数字仲裁（只否决）；
        面板中心：未裁切取四边形对角线交点（透视不变量）；被画幅裁切时
        对角线交点失效（实测偏差 150~800px），改用颜色掩膜凸包面积质心。
  定位  地图定位（Gauss-Newton）：布局已知后，9 个格心即"地图"。每步用
        名义动作模型预测位姿（x,y,θ），以可见面板(>=2)的「投影像素 ↔
        检出像素」残差迭代修正。观测按裁切与否分两种，均与预测同定义：
        - 未裁切：面板中心投影 ↔ 对角线交点（精确、与面板尺寸无关）；
        - 已裁切：预测四角投影与画幅求交的面积质心 ↔ 凸包质心（裁切不变）。
        平方损失会被裁切野值拖走，故用 IRLS + Cauchy 鲁棒核，并加弱先验
        正则（σ=8cm/8°）保证 1~2 点也能解。用真实内参+畸变模型。
  运动  小幅度动作白名单（单步 2cm 前进、~2cm 横移、3.2cm 后退），每步
        闭环；远距+对准良好才小批量（≤3 次 one_step = 6cm）。本场地禁用
        go_forward / go_forward_fast（5cm 步幅在小面板+打滑地板上易摔倒，
        `_act` 硬门拒绝）。转向用参考方案的「在线 EMA 估计小转实际角，
        连续 2 批 <0.5°/次→升级大转」，并带大转防振荡（误差符号翻转即
        停止转向、改用航向偏置接近）。
  容错  卡滞检测（相邻两次定位目标改善<0.5cm 连续 2 次→后退脱困）、
        丢失恢复（低头补扫→按运动先验朝目标转→退格重扫→盲走一步→撤销
        转向）、分格时间预算 + 全局看门狗、到达确认（位姿"脚下覆盖中心"
        主判 + 接近段确实检出过目标面板 + 蹭步来回覆盖微动开关）。
        绝不跳格：跳格断 70 分计分链（ESP32 顺序错只日志不断链，借道
        穿越其他面板计分安全）。

到达确认为什么不用"低头色占比"主判：相机高 39cm、俯仰 1040（41.4°）时
画面下沿对应地面 15.8cm，站在面板中心时面板远边仅 14cm——目标面板整个
在视野之外（实测占比 0.003，而旧阈值 0.10）。参考方案的"出现→压过→消失"
是过渡判据，不是"站上仍可见"。

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
from core.robot_core import RobotState
from vision.nine_grid_detector import NineGridDetector

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
# 容错预算
# =====================================================================
TARGET_TIME_BUDGET_S = 110.0   # 单格预算（15min/7格 ≈ 128s，留裕量）
TOTAL_TIME_BUDGET_S = 780.0    # 全局看门狗 13min（给上下场留 2min）
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
        # FSM 阶段标签（纯诊断：日志/仿真可视化用，不参与任何决策）
        self.phase = "INIT"

    # =================================================================
    # 入口
    # =================================================================

    def run_level(self):
        """布局扫描 → 按 1..7 顺序逐格导航。返回 bool（全部确认到位）

        每格结果同时写入 self.results（[(数字, 是否确认), ...]），供仿真/
        真机日志与诊断使用。
        """
        self.deadline = time.time() + TOTAL_TIME_BUDGET_S
        self.layout_scan()
        print(f"[布局] 数字→格: {self.digit_cell}  "
              f"仲裁冲突格: {sorted(self.cell_conflict)}")

        done = []
        for k in range(1, 8):
            ok = self.go_to_panel(k)
            done.append((k, ok))
            self.state.act("stand")
            elapsed = TOTAL_TIME_BUDGET_S - (self.deadline - time.time())
            print(f"[进度] 面板{k} {'完成' if ok else '未确认(以微动开关实际触发为准)'}，"
                  f"已用时 {elapsed:.0f}s")
            if time.time() > self.deadline:
                print("[看门狗] 全局超时，停止（保已得分，不冒进）")
                break
        self.results = done
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
                    obs = self.detector.detect_panels(frame, arbitrate=True,
                                                      drop_border=False)
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
                fit = self._lattice_grid_fit(pix_obs)
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

    def _lattice_grid_fit(self, pix_obs):
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
    # 单格导航 FSM：APPROACH → ENTER → CONFIRM
    # =================================================================

    def go_to_panel(self, digit):
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
        定位（旧实现每次动作 2 次定位，真机拍照 2~4s/张扛不住）。
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
        print(f"[确认] 面板{digit} 到达 ✓")
        return True

    # =================================================================
    # 运动原语（白名单 + 自适应转向 + 预测推进）
    # =================================================================

    def _act(self, action, times=1):
        """动作执行 + 位姿预测推进（GN 先验的基础）

        白名单硬门：本场地禁用的大步幅动作（go_forward / go_forward_fast）
        直接拒绝，避免后续改动误用导致摔倒。
        """
        if action in DISABLED_ACTIONS:
            raise ValueError(
                f"本关卡禁用动作 {action}（小面板+打滑地板易摔倒）；"
                f"前进请用 go_forward_one_step 小批量")
        if action not in ACTION_MODEL:
            raise ValueError(f"未登记的动作: {action}（动作白名单见 ACTION_MODEL）")
        self.state.act(action, times=times)
        self.predict_pose(action, times)
        if ACTION_MODEL.get(action, (None, 0))[0] != "turn":
            # 期间发生平移：旧转向不可盲目撤销，大转振荡记忆也失效
            self._last_turn = None
            self._last_big_turn = None

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
