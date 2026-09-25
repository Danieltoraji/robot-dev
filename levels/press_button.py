# -*- coding: utf-8 -*-
"""机器人智按按钮关卡（levels/press_button.py）

参考实现（`reference code/机器人智按按钮/`，该目录按仓库约定不入版本库）：
    robot/press_final.py        机器人端主程序（**本次移植的对象**，167 行）
    hardware/esp32_score.ino    ESP32 计分端：GPIO13/14 微动开关、INPUT_PULLUP、
                                按下 +10 并锁存、HTTP POST JSON 上报裁判服务器
    robot/press.d6a             `press` 动作组本体（SQLite，8 帧 / 5500ms，
                                驱 Servo3/4/6/7/8/11/12/14/15/16；已收入
                                models/press_button/press.d6a）
    说明书.txt                  关卡规格

关卡规格（说明书）：1m×1m 场地，四角各一块 30×30cm 水平按钮面板（高约 15cm），
面板后部竖板上贴 tag36h11 标签（边长 5cm，与站立时头部摄像头等高），面板中央是
微动开关。两个按钮各 50 分、满分 100；机器人依次按压 102、101。
通关流程：入口触发 → 站立 → 搜索目标 → 视觉闭环对准 → 渐进接近 → 按压 →
后退脱离 → 切换下一目标 → 全部完成。

本关为什么不用 PnP / 不用定位：
    参考程序刻意不用里程计、不解算世界坐标，只用「标签在画面里的横向位置 +
    标签的像素宽度」做闭环（说明书 §2.2 称之为"视觉闭环伺服"）。所以本关
    tag_poses 留空，也不调用 locate_with_retry/solve_pnp——这是本关的设计核心，
    移植时原样保留。

────────────────────────────────────────────────────────────────────────
★ 准入门槛：本关"对准"能收敛，靠的是一个不显眼的算术条件
────────────────────────────────────────────────────────────────────────
本关的判定顺序是「先判左右、再判前进」（见 decide()）。因此标签偏出画面中心
死区时，机器人只会原地左右转；而一步转身的幅度（22.0°/25.7°）是死区半角
（2.385°）的约 9 倍，几乎每次都会转过头。它能最终对上，靠的是：

        0  <  |turn_right − turn_left|  <  2 × 死区半角(2.385°) = 4.77°

机理：落点相位每轮单调滑动 |turn_right − turn_left| = 3.7°，而"能落进死区"的
相位带宽是 4.77° > 3.7°，所以必然扫过窗口。2026-09-26 相位推演（用本模块的
decide() 在像素空间扫描，`--selftest` 会重跑）：
    · **无打滑**：全起始方位 -33.68°~+33.68°（步长 0.05°，1348 个）
      1348/1348 收敛，转向步数中位 5、p99 = 12、最多 12，无丢目标。
    · **±10% 打滑**（30 个种子、20220 个样本）：没有真死锁、也不丢目标，
      但尾巴很长 —— 中位 6、p90 = 18、p99 = 36、最多 59；其中 0.5% 的对准要
      40 步以上。把上限放宽到 400 步后，这些"慢样本"**全部收敛**（最慢 94 步）。
      ±20% 打滑同样如此（p99 = 36、最多 58）。
      ⇒ **"最多 12 步"只在无打滑时成立**；真机上对准步数是长尾分布，这一点
        直接决定了下面护栏 40 步的取舍（见"护栏"一节的相互作用说明）。
    · 左右两步**等大**（22.0/22.0 或 24.0/24.0）⇒ 530/674 = 79% 起始方位
      **死循环**：永远进不了死区、也永远不丢标签 ⇒ 真机原地无限左右摆。
    · 差得太小（22.0/22.3）⇒ 能收敛但步数暴涨（150+ 步），现场看起来就是卡死。
⇒ 所以 TURN_LEFT_DEG / TURN_RIGHT_DEG 在本关是**载荷参数**，不是注释里的参考值。
  换机器人、换动作组、重标定之后必须重跑 `--selftest` 再上场。

────────────────────────────────────────────────────────────────────────
★ 阈值口径：为什么是 324 / ±81 / 1296（而不是参考里的 160 / ±40 / 640）
────────────────────────────────────────────────────────────────────────
参考程序用 `fswebcam -r 1280x720` 拍照，画面中心 640，判定阈值写死 160 / ±40。
本仓库 RobotState.capture_image() 的采集分辨率是 2592×1944（core/camera_config），
像素宽度与采集宽度成正比（w = fx·5cm/d，fx = 0.75035·W），所以照抄常量会让停止
距离随分辨率线性漂移：

    采集宽度 W      照抄 160 时的停止距离
    1280（参考）    30.0cm   ← 参考的标定口径
    2592（本仓库）  60.8cm   ← 照抄的话会在 60cm 外就"按"，压不到按钮
    640             15.0cm   ← 太近，会撞面板

本轮裁定的口径是**按 2592 写死**（不做运行时归一化），换算如下：
    画面中心   640   → 1296  （= 2592/2，几何事实）
    停止宽度   160   → 324   （= 160 × 2592/1280 = 160 × 2.025）
    对准死区  ±40px  → ±81   （= 40  × 2.025）
换算后与参考在**角度上完全等价**：atan(81/1944.9) = atan(40/960.44) = 2.385°，
所以上面那张收敛表在 2592 口径下继续成立。324 亦与本仓库内参自洽：
324 = 1944.9 × 5cm / 30.0cm，即"5cm 标签、距离 30cm"时恰好达标（**推断值**，
现场用尺子核，见文末复测清单①）。

⚠️ 换算前提：横向视场只是被缩放、没有被裁切（现场用实拍核对，复测清单④）。
⚠️ 采集分辨率一旦变化，324/81/1296 三个数必须按 `值 × 新宽/2592` 重算。

────────────────────────────────────────────────────────────────────────
★ 与参考代码的差异（逐条留痕，便于日后对账）
────────────────────────────────────────────────────────────────────────
1. 恒定阈值改为 2592 口径（见上）。
2. **新增护栏**（参考是裸 `while True`，无任何上限）：单目标转向 ≤40 次、
   单目标搜索 ≤10 轮、全局拍照 ≤400 张。超限不静默——打印卡在哪一步、
   已用多少张，run_level 返回 False（main.py 打"关卡失败"）。
3. 决策核心抽成纯函数 decide()，并加 `--selftest` 离线自检（把准入门槛
   变成可执行断言）。
4. 参考在 `__main__` 里、task_flow 之外还有一个 `go_forward×5` 退场动作；
   本轮裁定**不保留**（关卡结束即返回，退场交给现场流程）。
5. 新增诊断计数与 `read_scores` 钩子（真机无此方法，自动跳过）。

参考代码里与说明书不一致的地方（以**代码**为准，此处仅留痕）：
    · 说明书 §5.3 写的是 `for task in TARGET_TASKS: turn_left×2` 循环；代码是
      硬编码两段：任务1 前 `turn_left×2`、任务2 前 `turn_right×2`、任务2 后再
      `turn_right×2`。本模块按代码实现。
    · 说明书 §4.1.3 伪码写 `for (i<3)`、`scores[3]`；.ino 实际是 2 个按钮
      （`buttonPins[]={13,14}`、`scores[2]`）。
    · 说明书评分表写"每个按钮 50 分、满分 100"；ESP32 每次触发只 `+10`
      （即 20/20），100 分是赛制换算。
    · 说明书 §5.2.2 说头部脉宽 1800 是"左转约 30°"；本仓库标定映射
      （SERVO_DEG_PER_US = 0.09）是 27°。本模块按本仓库映射理解。
    · 说明书标题写"按顺序按压 2 个按钮"，"四角各设一个面板"——本关只用其中
      两个（102 先、101 后）；具体贴在哪两个角属现场项（复测清单⑤）。

运行：
    python main.py press_button                     # 真机跑（需先部署 press 动作组）
    python levels/press_button.py --selftest        # 离线自检（不碰硬件、不需要仿真）

★ 部署（真机，两步都要做）
    python tools/sync_to_robot.py                   # 默认集合已含 main.py / levels / models
    动作组必须**单独放到位**：把 models/press_button/press.d6a 复制到机器人的动作组
    目录 /home/pi/TonyPi/ActionGroups/（与 stand / turn_left / go_forward 等同一个目录，
    文件名即动作组名；该路径见 levels/line_seeker_tracking.py 的 ACTION_GROUP_DIR）。
    不放的话 AGC 找不到 'press'，按压那一步会直接失败——本仓库 RobotState.run_action
    走的是 AGC 默认动作组路径，不读 models/。

────────────────────────────────────────────────────────────────────────
★ 现场复测清单（代码里不猜数，全部列为实测项）
────────────────────────────────────────────────────────────────────────
① 站在预期按压距离拍一张，量标签像素宽度是否 ≈324（核 324 ↔ 30cm 的推断）。
   量的是**黑框跨度**：tag36h11 的 6×6 位阵最外圈就是黑边框，所以"边长 5cm"指的是
   黑框（不是含白边的纸面）——量错这一圈，宽度会差 4/3 倍，324 直接失效。
② `back_fast` 单次后退距离（本仓库无此动作实测值，MOTION_BUDGET_CM 里也没有）。
③ `turn_left` / `turn_right` 实际角度（**现在是载荷参数**，直接决定能不能收敛）。
④ `fswebcam -r 2592x1944` 是否只是缩放（横向视场有没有被裁切）。
⑤ 101 / 102 实际贴在哪两个角；标签高度是否与站立时相机等高（本关不动俯仰）。
⑥ 微动开关到标签的实际距离、`press` 动作组的可触及距离（决定"够不够得着"）。
"""

import math
import sys
import time

import numpy as np

# =====================================================================
# 关卡常量（逐条标注出处；数值口径见文件头"阈值口径"）
# =====================================================================
# ---- 画面与判定阈值（参考 press_final.py: CENTER_THRESHOLD=40、画面中心 640、
#      TARGET_TASKS 里的 stop_width=160；本模块按 2592 口径换算）----
CENTER_X = 1296              # 画面中心 = 2592/2（参考 640）
CENTER_THRESHOLD_PX = 81     # 左右对准死区 ±81px（参考 ±40）
STOP_WIDTH_PX = 324          # 停止/按压的标签像素宽度阈值（参考 160）
REF_FRAME_W = 1280           # 参考程序的采集宽度（换分辨率时据此重算上面三个数）

# ---- 头部舵机脉宽（参考 turn_head_left/right/center）----
HEAD_LEFT = 1800             # 参考注释"左转约 30°"；本仓库映射 = (1800-1500)*0.09 = 27°
HEAD_RIGHT = 1200
HEAD_CENTER = 1500

# ---- 动作组名（参考用 AGC.runActionGroup；本仓库经 RobotState.act 下发同名动组）----
A_STAND = "stand"
A_FWD = "go_forward"
A_TURN_L = "turn_left"
A_TURN_R = "turn_right"
A_PRESS = "press"            # 动作组本体 = models/press_button/press.d6a（8 帧/5500ms）
A_BACK = "back_fast"

# ---- 任务表（参考 TARGET_TASKS：102 先、101 后，各 stop_width=160）----
TARGET_TASKS = [
    {"id": 102, "stop_width": STOP_WIDTH_PX, "name": "目标1"},
    {"id": 101, "stop_width": STOP_WIDTH_PX, "name": "目标2"},
]

# ---- 动作次数（参考写死：预置转向 2、按压 2、后退 6）----
PRE_TASK_TURN_TIMES = 2
PRESS_TIMES = 2
BACK_FAST_TIMES = 6

# ---- 本仓库实测的动作组标定值（同名动作组，来自 levels/goodluck.py 的实机标定）----
# 本关的控制只按**名字**下发动组，这三个数不参与控制；它们用于离线自检、以及现场
# 核算"对准要转几步 / 前进几步能到"。**turn_left/turn_right 是准入门槛的载荷参数**
# （见文件头），goodluck 若重标定，这里必须同步并重跑 --selftest。
TURN_LEFT_DEG = 22.0
TURN_RIGHT_DEG = 25.7
FORWARD_CM = 5.0

# =====================================================================
# 护栏（参考代码没有；本轮裁定加入，超限不静默）
# =====================================================================
# ⚠️ 与准入门槛的相互作用（现场必读）：无打滑时对准 ≤12 步，40 步纯属保险；
#    但 ±10% 打滑下对准步数 p99 = 36、最多 59 ⇒ 40 步这道护栏会**误杀约 0.5%
#    "只是慢、本来能收敛"的对准**。日志会打印该目标实际用了多少步，便于现场
#    区分"真死锁（转向差被破坏，40 步也走不出来）"与"只是慢"。
ALIGN_TURN_BUDGET_PER_TARGET = 40    # 单目标转向动作总次数上限（对准 + 搜索的身体转向）
SEARCH_ROUND_BUDGET_PER_TARGET = 10  # 单目标"搜索环"轮数上限（每轮最多 3 张照片）
TOTAL_CAPTURE_BUDGET = 400           # 全局拍照张数上限（真机 0.62~0.70s/张 ⇒ 约 4.5 分钟）


class BudgetExceeded(Exception):
    """护栏触发（超限即主动退出，返回 False，不做任何"猜"动作）"""


# =====================================================================
# 判定核心（纯函数：参考代码的 if/elif 顺序逐字对应）
# =====================================================================

def decide(cx, w, stop_width=STOP_WIDTH_PX):
    """一帧的决策 → 'turn_left' / 'turn_right' / 'forward' / 'press'

    严格对应参考 press_final.py 的判定顺序（**顺序本身就是设计**）：
        cx 偏左   → 左转一步（对准优先于前进）
        cx 偏右   → 右转一步
        w 不够大  → 前进一步
        否则      → 按压
    改成"边转边走"或调整三条的先后，文件头那张收敛表即失效。
    cx / w 已按 2592 口径（CENTER_X / CENTER_THRESHOLD_PX / stop_width）。
    """
    if cx < CENTER_X - CENTER_THRESHOLD_PX:
        return "turn_left"
    if cx > CENTER_X + CENTER_THRESHOLD_PX:
        return "turn_right"
    if w < stop_width:
        return "forward"
    return "press"


# =====================================================================
# 一帧观测（真机链路 + PC 兜底）
# =====================================================================

_ARUCO_DETECTOR = None       # PC 兜底用的 cv2.aruco 检测器（惰性创建）
_ARUCO_NOTICE_PRINTED = False


def _extract_from_apriltag(results, target_id):
    """apriltag 库结果 → (center_x, width, 本帧 id 列表)

    width 公式与参考**逐字相同**：`corners[1][0] - corners[0][0]`，即标签**顶边
    在 x 方向的投影长度**，不是标签的真实边长。偏角 φ 下它按 cos φ 收缩（φ=20°
    时低估 6%），即"偏得越多、越以为还远"，方向上偏保守；160（本仓库 324）这个
    标定值就是在这个口径下量的，故不许"顺手改成真实边长"。
    角点顺序：本仓库已验证 apriltag 库返回顺序 = tag_poses 约定（左上→右上→右下
    →左下，见 core/robot_core.TAG_CORNER_PERM 的实测记录），此处不再重排。
    """
    seen = []
    for r in results:
        tid = int(r.tag_id)
        seen.append(tid)
        if tid != int(target_id):
            continue
        corners = np.asarray(r.corners, dtype=np.float64)
        width = float(corners[1][0] - corners[0][0])
        center_x = float(corners[:, 0].mean())
        return center_x, width, seen
    return None, None, seen


def _extract_from_aruco(frame, target_id):
    """PC 兜底：cv2.aruco 的 DICT_APRILTAG_36h11（与 apriltag 库同一个字典）

    只在真机链路不可用时使用（PC 上未安装 apriltag 库）。为了让 width 公式的含义
    与真机一致，这里把角点**显式规范**成图像坐标的左上→右上→右下→左下：标签正对
    时该规范等价于 apriltag 库的顺序，故 width 仍等于"顶边 x 投影长度"。
    """
    global _ARUCO_DETECTOR, _ARUCO_NOTICE_PRINTED
    import cv2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if _ARUCO_DETECTOR is None:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        params = getattr(cv2.aruco, "DetectorParameters", None)
        params = params() if params is not None else cv2.aruco.DetectorParameters_create()
        _ARUCO_DETECTOR = cv2.aruco.ArucoDetector(dictionary, params)
    corners, ids, _ = _ARUCO_DETECTOR.detectMarkers(gray)
    if ids is None:
        return None, None, []
    seen = [int(i) for i in np.asarray(ids).flatten()]
    for pts, tid in zip(corners, seen):
        if tid != int(target_id):
            continue
        pts = np.asarray(pts, dtype=np.float64).reshape(4, 2)
        ordered = _order_corners_tl_tr_br_bl(pts)
        width = float(ordered[1][0] - ordered[0][0])
        center_x = float(ordered[:, 0].mean())
        return center_x, width, seen
    return None, None, seen


def _order_corners_tl_tr_br_bl(pts):
    """四角点 → 左上/右上/右下/左下（图像坐标；用于 PC 兜底的口径统一）"""
    s = pts.sum(axis=1)
    i_tl, i_br = int(np.argmin(s)), int(np.argmax(s))
    rest = [i for i in range(4) if i not in (i_tl, i_br)]
    i_tr = rest[0] if pts[rest[0]][0] > pts[rest[1]][0] else rest[1]
    i_bl = rest[1] if i_tr == rest[0] else rest[0]
    return pts[[i_tl, i_tr, i_br, i_bl]]


# =====================================================================
# 关卡主体
# =====================================================================

class PressButtonLevel:
    """本关状态机。state 为 core.robot_core.RobotState（真机）或同接口替身。

    结构逐条对应参考 press_final.py：
        run_level()         ← __main__ + task_flow()
        _task_flow()        ← task_flow()（含 try/finally 与三段预置转向）
        _track_and_press()  ← track_and_press_tag()（SEARCH/ALIGN/APPROACH/PRESS/ESCAPE）
        _search_step()      ← track_and_press_tag() 里 `if cx is None:` 那一段
    """

    def __init__(self, state, tasks=None, verbose=True):
        self.state = state
        self.tasks = list(tasks if tasks is not None else TARGET_TASKS)
        self.verbose = verbose
        # 计数（诊断用；不是控制逻辑）
        self.captures = 0
        self.actions = 0
        self._turns = 0            # 当前目标的转向次数
        self._search_rounds = 0    # 当前目标的搜索轮数
        self.results = []          # [(target_id, ok, 按压时宽度 或 None)]
        self._backend = "apriltag"  # 观测后端（诊断用）

    # ---- 基础设施 ---------------------------------------------------

    def _log(self, msg):
        if self.verbose:
            print(f"[按按钮] {msg}")

    def _act(self, name, times=1):
        self.actions += 1
        self.state.act(name, times)

    def _reset_target_counters(self):
        """开新目标（含两段预置转向）时清零护栏计数——护栏是"每目标"口径"""
        self._turns = 0
        self._search_rounds = 0
        self._last_seen = []

    def _turn(self, action, times):
        """转向统一入口：计入单目标转向预算（护栏）"""
        self._turns += int(times)
        if self._turns > ALIGN_TURN_BUDGET_PER_TARGET:
            raise BudgetExceeded(
                f"单目标转向已达 {self._turns} 次 > 上限 {ALIGN_TURN_BUDGET_PER_TARGET}"
                f"（拍照 {self.captures} 张）——对准没有收敛，按现场规则主动退出")
        self._act(action, times)

    def _spend_capture(self):
        self.captures += 1
        if self.captures > TOTAL_CAPTURE_BUDGET:
            raise BudgetExceeded(
                f"全局拍照已达 {self.captures} 张 > 上限 {TOTAL_CAPTURE_BUDGET}"
                f"（真机按 0.65s/张 ≈ {TOTAL_CAPTURE_BUDGET * 0.65 / 60:.1f} 分钟预算）")

    # ---- 观测 -------------------------------------------------------

    def _observe(self, target_id):
        """拍一张（**恰好一张**，与参考一致）→ 找 target_id → (center_x, width)

        真机链路与参考同为"先落盘再读"：state.capture_image() → detect_apriltag(path)。
        两者不可用时（PC 上未装 apriltag 库 / 读图失败）退回 cv2.aruco 读同一张照片，
        让移植在没有仿真的情况下也能自检；真机路径不经过兜底分支。
        """
        self._spend_capture()
        try:
            path = self.state.capture_image()
        except Exception as e:                       # 拍照本身失败：按"未检出"处理
            self._log(f"拍照异常：{type(e).__name__}: {e}")
            return None, None
        if path is None:
            self._log("拍照失败（capture_image 返回 None）")
            return None, None

        results = None
        try:
            results = self.state.detect_apriltag(path)
        except Exception as e:                       # 未装 apriltag / 图像读不出
            self._log(f"apriltag 链路不可用（{type(e).__name__}: {e}），改用 cv2.aruco 兜底")
            self._backend = "aruco"

        if results is not None:
            cx, w, seen = _extract_from_apriltag(results, target_id)
        else:
            cx, w, seen = self._observe_with_aruco(path, target_id)
        self._last_seen = seen
        return cx, w

    def _observe_with_aruco(self, path, target_id):
        global _ARUCO_NOTICE_PRINTED
        try:
            import cv2
        except Exception as e:
            self._log(f"cv2 也不可用（{e}），本帧视为未检出")
            return None, None, []
        frame = cv2.imread(path)
        if frame is None:
            self._log(f"读图失败：{path}")
            return None, None, []
        if not _ARUCO_NOTICE_PRINTED:
            _ARUCO_NOTICE_PRINTED = True
            self._log("注意：本次运行使用 cv2.aruco 兜底检测（真机应走 apriltag 库）")
        return _extract_from_aruco(frame, target_id)

    # ---- 搜索环（参考 `if cx is None:` 那一段）------------------------

    def _search_step(self, target_id):
        """左转头 → 右转头 → 身体左转×2；命中即回正头 + 身体转一步

        含参考里那个"身体固定转一步跟头"的启发式：头转到 ±27° 发现目标后，身体
        无脑转固定一步（22°/25.7°），会系统性过冲——这是**复现对象**，不改。
        """
        self._search_rounds += 1
        if self._search_rounds > SEARCH_ROUND_BUDGET_PER_TARGET:
            raise BudgetExceeded(
                f"单目标搜索已达 {self._search_rounds} 轮 > 上限 "
                f"{SEARCH_ROUND_BUDGET_PER_TARGET}（拍照 {self.captures} 张）")

        self.state.set_head(HEAD_LEFT)
        cx, _w = self._observe(target_id)
        if cx is not None:
            self._log(f"左侧发现目标(cx={cx:.0f})，回正头部并身体左转")
            self.state.set_head(HEAD_CENTER)
            self._turn(A_TURN_L, 1)
            return

        self._log("向右转头搜索...")
        self.state.set_head(HEAD_RIGHT)
        cx, _w = self._observe(target_id)
        if cx is not None:
            self._log(f"右侧发现目标(cx={cx:.0f})，回正头部并身体右转")
            self.state.set_head(HEAD_CENTER)
            self._turn(A_TURN_R, 1)
            return

        self._log("视野内未发现目标，身体左转搜索...")
        self.state.set_head(HEAD_CENTER)
        self._turn(A_TURN_L, PRE_TASK_TURN_TIMES)

    # ---- 单目标状态机（参考 track_and_press_tag）----------------------

    def _track_and_press(self, task):
        """SEARCH → ALIGN → APPROACH → PRESS → ESCAPE；返回是否完成该目标"""
        target_id = int(task["id"])
        stop_width = task["stop_width"]
        name = task["name"]
        self._log(f"===== 开始执行 [{name}] | 目标ID: {target_id}, "
                  f"停止宽度: {stop_width}px =====")
        self._reset_target_counters()
        self.state.set_head(HEAD_CENTER)

        while True:
            cx, w = self._observe(target_id)

            # 状态1：搜索（参考 `if cx is None:`）
            if cx is None:
                self._log(f"正前方未发现目标（本帧检出 id={self._last_seen}），向左转头搜索...")
                self._search_step(target_id)
                continue

            # 状态2/3/4：对准 → 接近 → 按压（判定顺序见 decide()）
            action = decide(cx, w, stop_width)
            if action == "turn_left":
                self._log(f"向左修正，当前偏差: {CENTER_X - cx:.0f}px（标签宽 {w:.0f}px）")
                self._turn(A_TURN_L, 1)
                continue
            if action == "turn_right":
                self._log(f"向右修正，当前偏差: {cx - CENTER_X:.0f}px（标签宽 {w:.0f}px）")
                self._turn(A_TURN_R, 1)
                continue
            if action == "forward":
                self._log(f"距离不足（当前宽度 {w:.0f} / 目标 {stop_width}），前进...")
                self._act(A_FWD, 1)
                continue

            # 状态4：按压（含参考对"够近"的诚实声明）
            self._log(f"到达 [{name}] 目标位置（标签宽 {w:.0f}px ≥ {stop_width}px），执行按压！")
            self._act(A_PRESS, PRESS_TIMES)
            self._log("按压动作组已执行（约 5.5s/次 × 2）——微动开关状态在 Pi 侧不可读，"
                      "是否触发以场地计分为准")
            # 状态5：脱离
            self._log(f"[{name}] 完成，执行后退脱离...")
            self._act(A_BACK, BACK_FAST_TIMES)
            self.state.set_head(HEAD_CENTER)
            self.results.append((target_id, True, float(w)))
            return True

    # ---- 主流程（参考 task_flow）-------------------------------------

    def _task_flow(self):
        self._log("系统启动，初始化站立...")
        self._act(A_STAND)
        time.sleep(1)                                  # 参考固定等待
        try:
            first, second = self.tasks[0], self.tasks[1]
            # 【任务1】预置左转×2（参考注释：调整起始朝向，便于搜索下一个目标）
            self._reset_target_counters()
            self._turn(A_TURN_L, PRE_TASK_TURN_TIMES)
            time.sleep(0.5)
            ok1 = self._track_and_press(first)
            time.sleep(1)

            # 【任务2】预置右转×2
            self._reset_target_counters()
            self._turn(A_TURN_R, PRE_TASK_TURN_TIMES)
            time.sleep(0.5)
            ok2 = self._track_and_press(second)
            time.sleep(1)

            # 收尾：参考在此处再右转×2，随后 __main__ 里还有 go_forward×5 退场
            # （本轮裁定不保留退场动作，只保留这次转向，保持与参考的朝向一致）
            self._turn(A_TURN_R, PRE_TASK_TURN_TIMES)
            time.sleep(0.5)
            self._log("所有任务执行完毕！")
            return ok1 and ok2
        finally:
            # 参考的 finally：回正头部 + 站立（任何退出路径都不留倒姿/歪头）
            self.state.set_head(HEAD_CENTER)
            self._act(A_STAND)

    # ---- 诊断 -------------------------------------------------------

    def _read_scores(self):
        """可选真值来源：state 提供 read_scores() 时读一次；真机无此方法 → None

        本关按压是否真的触发微动开关，**Pi 侧不可观测**（开关接 ESP32，经 WiFi
        上报裁判服务器）。真机的唯一真值是场地计分；此钩子只为后续仿真/复算留接口。
        """
        getter = getattr(self.state, "read_scores", None)
        if getter is None:
            return None
        try:
            return getter()
        except Exception as e:
            self._log(f"read_scores 读取失败：{type(e).__name__}: {e}")
            return None

    def _summary(self, ok):
        self._log(f"拍照 {self.captures} 张｜动作 {self.actions} 次"
                  f"｜检测后端 {self._backend}")
        for target_id, done, width in self.results:
            self._log(f"目标 id={target_id}：{'已按压' if done else '未完成'}"
                      f"（按压时标签宽 {width:.0f}px）")
        scores = self._read_scores()
        if scores is not None:
            self._log(f"计分端读数（read_scores）：{scores}")
        self._log(f"结果：{'成功' if ok else '失败/中止'}")

    # ---- 入口 -------------------------------------------------------

    def run_level(self):
        ok = False
        try:
            ok = self._task_flow()
        except BudgetExceeded as e:
            self._log(f"护栏触发，主动退出：{e}")
        except KeyboardInterrupt:
            self._log("任务被用户手动中断")
        except Exception as e:
            self._log(f"异常退出：{type(e).__name__}: {e}")
        self._summary(ok)
        return ok


def run_level(state):
    """main.py LEVELS 入口"""
    return PressButtonLevel(state).run_level()


tag_poses = {}   # 本关不用 PnP（纯像素伺服），赛道数据留空


# =====================================================================
# 离线自检：把文件头的"准入门槛"变成可执行断言
# =====================================================================
# 运行：python levels/press_button.py --selftest
# 不碰硬件、不需要仿真：只用本模块的 decide() 在像素空间做相位推演。

FX_2592 = 1944.903664123011          # camera_config.CAMERA_INTRINSIC[0][0]
HALF_FOV_DEG = math.degrees(math.atan(CENTER_X / FX_2592))   # 半视场 ≈ 33.68°
DEAD_ZONE_DEG = math.degrees(math.atan(CENTER_THRESHOLD_PX / FX_2592))  # ≈ 2.385°
DEAD_ZONE_DEG_REF = math.degrees(math.atan(40.0 / (FX_2592 * 1280.0 / 2592.0)))


def _cx_of(phi_deg):
    """方位角 → 标签中心像素 x（与实际判定同一套针孔关系）"""
    return CENTER_X + FX_2592 * math.tan(math.radians(phi_deg))


def _simulate_align(phi0_deg, turn_left_deg, turn_right_deg, rng=None, max_actions=60):
    """从起始方位 phi0 出发跑"对准环" → ('aligned'|'lost'|'deadlock', 转向次数)

    只用 decide()（真代码），不动 state：w 传 0 使"居中"直接落到 forward，
    即对准完成；|phi| 超出半视场即视为丢目标（标签出画）。
    """
    phi, turns = float(phi0_deg), 0
    for _ in range(max_actions):
        if abs(phi) > HALF_FOV_DEG:
            return "lost", turns
        left = turn_left_deg if rng is None else rng.gauss(turn_left_deg, turn_left_deg * 0.1)
        right = turn_right_deg if rng is None else rng.gauss(turn_right_deg, turn_right_deg * 0.1)
        action = decide(_cx_of(phi), 0.0, STOP_WIDTH_PX)
        if action == "turn_left":
            phi += left
            turns += 1
        elif action == "turn_right":
            phi -= right
            turns += 1
        else:
            return "aligned", turns
    return "deadlock", turns


def _sweep_align(turn_left_deg, turn_right_deg, rng=None, step_deg=0.05):
    """扫描全部起始方位 → (结果计数, 转向次数列表)"""
    counts = {"aligned": 0, "lost": 0, "deadlock": 0}
    turns_seen = []
    phi = -HALF_FOV_DEG
    while phi <= HALF_FOV_DEG + 1e-9:
        outcome, turns = _simulate_align(phi, turn_left_deg, turn_right_deg, rng=rng)
        counts[outcome] += 1
        if outcome == "aligned":
            turns_seen.append(turns)
        phi += step_deg
    return counts, turns_seen


def _selftest():
    """返回 True/False（进程退出码 0/1）"""
    import random

    failures = []
    print("[selftest] 阈值口径：")
    print(f"  画面中心 {CENTER_X}｜停止宽度 {STOP_WIDTH_PX}px｜死区 ±{CENTER_THRESHOLD_PX}px")
    print(f"  fx@2592 = {FX_2592:.1f} → 停止宽度 {STOP_WIDTH_PX}px 对应距离 "
          f"{FX_2592 * 5.0 / STOP_WIDTH_PX:.1f}cm（5cm 标签）")
    print(f"  死区半角 = {DEAD_ZONE_DEG:.3f}°（参考 1280 口径 = {DEAD_ZONE_DEG_REF:.3f}°）")
    if abs(DEAD_ZONE_DEG - DEAD_ZONE_DEG_REF) > 1e-9:
        failures.append("死区半角与参考口径不等价（324/81/1296 换算写错了）")

    delta = abs(TURN_RIGHT_DEG - TURN_LEFT_DEG)
    print(f"[selftest] 准入门槛：|turn_right-turn_left| = {delta:.1f}° 必须落在 "
          f"(0, {2 * DEAD_ZONE_DEG:.2f}°) 内")
    if not (0.0 < delta < 2 * DEAD_ZONE_DEG):
        failures.append(f"转向差 {delta:.1f}° 不在 (0, {2 * DEAD_ZONE_DEG:.2f}°) 内，"
                        f"本关会死循环或跳过对准窗口")

    counts, turns_seen = _sweep_align(TURN_LEFT_DEG, TURN_RIGHT_DEG)
    print(f"[selftest] 起始方位扫描 {counts['aligned'] + counts['lost'] + counts['deadlock']} 个"
          f"（-{HALF_FOV_DEG:.2f}°~+{HALF_FOV_DEG:.2f}°）：{counts}")
    if counts["aligned"] == 0:
        failures.append("没有任何起始方位能对准")
    if counts["lost"] or counts["deadlock"]:
        failures.append(f"存在丢目标/死循环：lost={counts['lost']} deadlock={counts['deadlock']}")
    if turns_seen:
        print(f"[selftest] 对准转向步数：中位 {sorted(turns_seen)[len(turns_seen) // 2]}、"
              f"最多 {max(turns_seen)}")
        if max(turns_seen) > 12:
            failures.append(f"最坏对准步数 {max(turns_seen)} > 12（与推演结论不符）")

    # 打滑敏感性：**不丢目标、无真死锁**，但尾巴很长（真机必须知道这一点）
    def noisy_case(seed, phi, cap):
        return _simulate_align(phi, TURN_LEFT_DEG, TURN_RIGHT_DEG,
                               rng=random.Random(f"{seed}:{round(phi, 3)}"),
                               max_actions=cap)

    noisy_turns, over40, stuck, n_lost = [], 0, [], 0
    for seed in range(30):
        phi = -HALF_FOV_DEG
        while phi <= HALF_FOV_DEG + 1e-9:
            outcome, turns = noisy_case(seed, phi, 60)
            if outcome == "lost":
                n_lost += 1
            elif outcome == "deadlock":
                stuck.append((seed, phi))
            else:
                noisy_turns.append(turns)
                over40 += 1 if turns > 40 else 0
            phi += 0.1
    noisy_turns.sort()
    if noisy_turns:
        def _p(q):
            return noisy_turns[min(len(noisy_turns) - 1, int(q * len(noisy_turns)))]
        print(f"[selftest] ±10% 打滑 30 种子（{len(noisy_turns) + len(stuck) + n_lost} 样本）："
              f"中位 {_p(.5)}、p90 {_p(.9)}、p99 {_p(.99)}、最多 {noisy_turns[-1]}；"
              f"越过护栏 40 步的占 {over40 / max(1, len(noisy_turns)):.1%}")
    if n_lost:
        failures.append(f"打滑后出现丢目标 {n_lost} 例")
    still_stuck = [c for c in stuck if noisy_case(c[0], c[1], 400)[0] != "aligned"]
    if still_stuck:
        failures.append(f"打滑下有 {len(still_stuck)} 例放宽到 400 步仍不收敛（真死锁）")
    elif stuck:
        print(f"[selftest] 其中 {len(stuck)} 例在 60 步内没收敛，放宽到 400 步后**全部收敛**"
              f"（打滑下是长尾，不是死锁）")

    eq_counts, _ = _sweep_align(22.0, 22.0, step_deg=0.1)
    ratio = eq_counts["deadlock"] / max(1, sum(eq_counts.values()))
    print(f"[selftest] 反例（左右等大 22.0/22.0）：{eq_counts}"
          f" → 死循环占比 {ratio:.0%}")
    if ratio < 0.5:
        failures.append(f"等步长反例不成立（死循环占比仅 {ratio:.0%}）——"
                        f"说明判定顺序或死区口径已经被改动")

    print()
    if failures:
        for f in failures:
            print(f"[selftest] 失败：{f}")
        return False
    print("[selftest] 全部通过：判定顺序、阈值口径、收敛准入条件均与推演一致。")
    return True


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(0 if _selftest() else 1)
    print("本模块是关卡实现，真机运行：python main.py press_button")
    print("离线自检：python levels/press_button.py --selftest")
