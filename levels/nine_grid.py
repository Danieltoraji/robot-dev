# -*- coding: utf-8 -*-
"""
数字宫格关卡（levels/nine_grid.py）

任务：1m×1m 九宫格（3x3，左下位置6恒空），7 块数字面板(1~7)随机布局，
机器人按 1→7 顺序依次到达各面板中心（踩中微动开关），限时 15 分钟。
评分：顺序正确每格 10 分（70），限时完成 +30。

方案要点（2026-09-08 定稿，详见 docs/关卡算法/数字宫格攻略.md）：
  视觉  颜色主判（HSV 七色，1红..7粉）+ HOG+SVM 黑色数字仲裁（只否决）；
        面板中心取四边形对角线交点（透视不变量，bbox 中心有系统性偏差）。
  定位  地图定位（Gauss-Newton）：布局已知后，9 个格心即"地图"。每步用
        名义动作模型预测位姿（x,y,θ），以可见面板(>=2)的「投影像素 ↔
        检出像素」残差迭代修正。用真实内参+畸变模型，无 AprilTag 而全程
        度量闭环；比逐帧单应（需>=4面板共视）在接近段稳健得多。
        开机布局扫描用静态单应（点击标定或解析自举）+ 头部双俯角宽扫。
  运动  小幅度动作白名单（单步 2cm 前进、~2cm 横移、3.2cm 后退），每步
        闭环；批量直行仅远距+对准良好且 <=2 步。转向用参考方案的「在线
        EMA 估计小转实际角，<0.5°/次判被地面吞掉→升级大转」。
  容错  卡滞检测（前进步距目标改善<0.5cm 连续 2 次→后退脱困）、丢失恢复
        （低头补扫→退格重扫→撤销上次转向）、分格时间预算 + 全局看门狗、
        到达确认（低头多帧色占比 + 蹭步来回覆盖微动开关）。
        绝不跳格：跳格断 70 分计分链（ESP32 顺序错只日志不断链，借道
        穿越其他面板计分安全）。

现场依赖：
  1) archive/result/ninegrid_homography.json（可选；缺失用 from_pose 解析
     自举，只影响布局归属精度，建议现场点击标定覆盖）；
  2) COLOR_THRESHOLDS 现场复标（tools/debug_vision.py ninegrid 子命令）；
  3) SVM 仲裁依赖 scikit-learn/skimage/joblib（缺失自动降级纯颜色）。

注意：localize() 不切换物理俯仰，按 state 当前俯仰拍照解算；调用方负责
用 state.set_pitch 保证俯仰与期望视野一致（GN 按 pitch 档投影）。
"""

import time

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

from core.camera_config import (
    HEAD_CENTER, HEAD_LEFT, HEAD_RIGHT, HEAD_WIDE_LEFT, HEAD_WIDE_RIGHT,
    SERVO_DEG_PER_US, CAMERA_INTRINSIC, CAMERA_DISTORTION,
)
from core.ground_homography import (
    GroundHomography, grid_cell_center, nearest_cell,
    CAMERA_TO_BODY_FORWARD_CM,
)
from core.robot_core import RobotState
from vision.nine_grid_detector import NineGridDetector, ID_TO_COLOR

CAM_HEIGHT_CM = 39.0  # 相机光心高度（站立实测，与 ground_homography 缺省一致）
PITCH_NAV = 1200      # 前视导航/布局预扫（下沿 53.5°；转弯不拍到自身）
PITCH_DOWN = 1040     # 低头：可见地面带 0.16~1.47m（近距/脚下判定）

# =====================================================================
# 到达与对准容差（面板 33cm，微动开关区≈中心 2/3≈±5.5cm）
# =====================================================================
ALIGN_TOL_DEG = 8.0      # 航向差小于此视为对准
ENTER_ALIGN_TOL_DEG = 15.0  # 进入段容忍的航向差上限（粗对齐即可，横移纠偏）
FAR_DIST_CM = 30.0       # > 此值允许 go_forward 批量
MID_DIST_CM = 12.0       # > 此值 APPROACH，否则 ENTER（切低头）
ENTER_MIN_CM = -10.0     # 中心退到脚后此值内仍算"到位"（防过冲判据）
ENTER_TOL_CM = 4.0       # 中心进入 [ENTER_MIN, ENTER_TOL] 转 CONFIRM
LAT_TOL_CM = 4.0         # 横偏容差；超过用横移纠正
CONFIRM_LAT_MAX_CM = 6.0
CONFIRM_RATIO = 0.10     # 低头帧目标色占比：站上面板应远超此值
RATIO_FRAMES = 3         # 到达确认帧数（多数表决）
TURN_NEAR_LIMIT_CM = 20.0  # 距目标中心小于此先退格再转（防面板边缘转向）

# =====================================================================
# 动作白名单与名义位移模型（预测用；实测标定值见 levels/goodluck.py）
# =====================================================================
FORWARD_CM = 5.0            # go_forward（实测更直，仅远距批量用）
FORWARD_ONE_STEP_CM = 2.0   # go_forward_one_step（主前进原语）
BACK_ONE_STEP_CM = 3.2      # back_one_step
LEFT_MOVE_CM = 1.9
RIGHT_MOVE_CM = 2.2
TURN_LEFT_DEG = 22.0
TURN_RIGHT_DEG = 25.7
BATCH_MAX_STEPS = 2         # 批量直行上限（打滑地板保守值）
BATCH_MAX_ANGLE_DEG = 4.0   # 批量直行允许的最大航向差

# (类型, 名义增量)：fwd 前向 cm（负=后退）、lat 右向 cm、turn 右转度（右正）
ACTION_MODEL = {
    "go_forward": ("fwd", FORWARD_CM),
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
SMALL_TURN_INIT_DEG = 4.0     # 小转单次角度初估
TURN_EMA_ALPHA = 0.3          # 指数滑动平均
MIN_EFFECTIVE_TURN_DEG = 0.5  # 每次实际转角低于此 = 被地面吞掉
SMALL_TURN_MAX_STEPS = 6      # 一轮小转批量上限
BIG_TURN_THRESHOLD_DEG = 30.0  # 超此角度直接大转（小转批量太慢）

# =====================================================================
# 定位（Gauss-Newton 地图定位）门控
# =====================================================================
GN_MAX_ITERS = 10
GN_PIXEL_GATE = 8.0       # 单帧解残差 RMS 上限（px；远排对角线中心量化噪声~5.6px）
GN_PIXEL_GATE_MERGED = 12.0  # 头部扫合并帧的残差上限（斜视角中心偏差 10~30px 级）
GN_OUTLIER_PX = 15.0      # 单点残差超此剔除后重解（误检防护）
GN_JUMP_CM = 12.0         # 解算位姿相对预测的最大跳变（预算+裕度）
GN_JUMP_DEG = 30.0        # 最大航向跳变

# =====================================================================
# 容错预算
# =====================================================================
TARGET_TIME_BUDGET_S = 110.0   # 单格预算（15min/7格 ≈ 128s，留裕量）
TOTAL_TIME_BUDGET_S = 780.0    # 全局看门狗 13min（给上下场留 2min）
STALL_EPS_CM = 0.5             # 卡滞判据：前进单步距目标改善低于此
STALL_CONSEC = 2               # 连续多次触发即卡滞
RETRY_LIMIT = 2                # 单格 CONFIRM 失败重试上限
ENTER_FAIL_LIMIT = 3           # 进入段连续定位失败上限


DEBUG_GN = False


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
        self._last_turn = None  # (action, times)：定位丢失时撤销转向用

    # =================================================================
    # 入口
    # =================================================================

    def run_level(self):
        """布局扫描 → 按 1..7 顺序逐格导航。返回 [(数字, 是否确认到位)]"""
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
        return done

    # =================================================================
    # 布局扫描（静态单应自举；此后全部用 GN 地图定位）
    # =================================================================

    def layout_scan(self):
        """头部双俯角宽扫全场：色块→格归属多数投票，建立 数字→格 映射

        双俯角互补：导航档看不到最近排（贴下边被裁切），低头档看不全
        远排；任一档扫齐 7 个数字即提前收束。
        """
        def hg_for(pitch):
            hg = GroundHomography.load(pitch)
            if hg is not None:
                return hg
            return GroundHomography.from_pose(
                self.start_cam_xy, 39.0, pitch, self.start_bearing_deg)

        print("[布局] 无布局档标定时用解析自举（现场建议点击标定覆盖）")
        votes = {}  # cell -> {digit: count}
        for pitch in (PITCH_NAV, PITCH_DOWN):
            self.state.set_pitch(pitch)
            hg = hg_for(pitch)
            for pulse in (HEAD_CENTER, HEAD_LEFT, HEAD_RIGHT,
                          HEAD_WIDE_LEFT, HEAD_WIDE_RIGHT):
                self.state.set_head(pulse)
                frame = self.state.capture_frame()
                if frame is None:
                    continue
                for o in self.detector.detect_panels(frame, arbitrate=True,
                                                     drop_border=True):
                    field = hg.pixels_to_ground([o.center_px],
                                                head_pulse=pulse)[0]
                    cell = nearest_cell(field)
                    if cell is None:
                        continue
                    votes.setdefault(cell, {}).setdefault(o.digit, 0)
                    votes[cell][o.digit] += 1
                    if o.arb_conflict:
                        self.cell_conflict.add(cell)
                        print(f"[布局] 仲裁冲突：颜色{o.digit}@格{cell} "
                              f"SVM说{o.model_digit}({o.model_conf:.2f})——待近距复核")
                if self._layout_complete(votes):
                    break
            if self._layout_complete(votes):
                break
        self.state.set_head(HEAD_CENTER)
        self.state.set_pitch(PITCH_NAV)

        # 每格取最高票数字；同数字多格时保留票数更高的格
        best = {}  # cell -> (digit, votes)
        for cell, dv in votes.items():
            d = max(dv, key=dv.get)
            best[cell] = (d, dv[d])
        for cell, (d, n) in best.items():
            if d not in self.digit_cell \
                    or n > best[self.digit_cell[d]][1]:
                self.digit_cell[d] = cell

        missing = [k for k in range(1, 8) if k not in self.digit_cell]
        if missing:
            raise RuntimeError(
                f"布局扫描未找到数字 {missing}：检查 HSV 阈值/光照/起始位姿。"
                f"投票记录: {votes}")

        # 补一帧中位头+导航档做 GN 精化，消掉解析自举的起始位姿假设误差
        # （宽扫帧头非中位/俯角不同，不能直接进 GN 投影模型）
        frame = self.state.capture_frame()
        if frame is not None:
            obs = self.detector.detect_panels(frame)
            corr = [(grid_cell_center(self.digit_cell[o.digit]),
                     o.center_px, HEAD_CENTER)
                    for o in obs if o.digit in self.digit_cell]
            if len(corr) >= 2:
                pose = self._gn_localize(self.pose, corr, PITCH_NAV)
                if pose is not None:
                    self.pose = pose

    def _layout_complete(self, votes):
        """已见到全部 7 个数字即可提前结束扫描"""
        seen = {max(dv, key=dv.get) for dv in votes.values()}
        return len(seen) >= 7

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
        """拍照+检测，返回 (对应点[(格心,像素,头部脉宽)], obs, frame)"""
        frame = self.state.capture_frame()
        if frame is None:
            return [], None, None
        obs = self.detector.detect_panels(frame, arbitrate=arbitrate)
        head = self.state.current_head_pulse
        corr = [(grid_cell_center(self.digit_cell[o.digit]), o.center_px, head)
                for o in obs if o.digit in self.digit_cell]
        return corr, obs, frame

    def localize(self, pitch=PITCH_NAV, arbitrate=False, min_panels=2):
        """拍照 → 检测 → >=min_panels 面板 GN 精化位姿（门控后写入 self.pose）

        面板不足 3 个时做头部左右小扫（零身体位移，零风险）把侧向面板
        转进视野，跨帧合并对应点（每点带自己的头部角）再解。
        返回 (obs_list, frame)；失败返回 None（self.pose 保持预测值）。
        """
        corr, obs, frame = self._capture_corr(pitch, arbitrate)
        if len(corr) >= min_panels:
            # 单帧优先：无头部斜视角偏差，门控最严、精度最高
            pose = self._gn_localize(self.pose.copy(), corr, pitch)
            if pose is not None:
                self.pose = pose
                return obs, frame
        # 头部左右小扫（零身体位移）合并对应点：贴近段侧板大量合法裁切，
        # 单帧经常不足；斜视角中心偏差较大用放宽门控
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
        """以 p0=(x,y,θ) 为先验做多起点 GN 定位

        先验漂移（打滑/动作噪声）会把单起点 GN 带进错误局部极小，
        因此从先验+8个扰动起点各解一次，取残差最小者。
        门控：残差 RMS(gate_px)、相对先验的跳变（动作预算 + 裕度）。
        失败返回 None（调用方维持预测位姿）。
        """
        starts = [np.array(p0, dtype=np.float64)]
        for dxy in ((12, 0), (-12, 0), (0, 12), (0, -12),
                    (12, 12), (-12, -12), (12, -12), (-12, 12)):
            st = np.array(p0, dtype=np.float64)
            st[0] += dxy[0]
            st[1] += dxy[1]
            starts.append(st)
        for dth in (np.radians(15), -np.radians(15)):
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
        """单起点 GN 迭代 + 一轮离群剔除。返回 (p, rms) 或 None。"""
        p = np.array(p0, dtype=np.float64)
        cells = np.array([c for c, _, _ in corr], dtype=np.float64)
        dets = np.array([d for _, d, _ in corr], dtype=np.float64)
        heads = np.array([h for _, _, h in corr], dtype=np.float64)
        used = np.ones(len(corr), dtype=bool)

        def residuals(pp, mask):
            # 每个对应点用自己采集时的头部角投影
            out = np.empty((int(np.sum(mask)), 2))
            k = 0
            for i in np.where(mask)[0]:
                px = project_ground_to_pixel(
                    cells[i], pp[0], pp[1], pp[2], pitch, heads[i])[0]
                out[k] = px - dets[i]
                k += 1
            return out

        r = residuals(p, used).ravel()
        for it in range(GN_MAX_ITERS):
            # 数值雅可比（3 参数，前向差分）
            J = np.zeros((r.size, 3))
            for j in range(3):
                eps = [0.5, 0.5, np.radians(0.5)][j]
                pp = p.copy(); pp[j] += eps
                r1 = residuals(pp, used)
                pp = p.copy(); pp[j] -= eps
                r2 = residuals(pp, used)
                J[:, j] = ((r1 - r2) / (2 * eps)).ravel()
            try:
                delta = np.linalg.solve(J.T @ J + 1e-6 * np.eye(3),
                                        -(J.T @ r))
            except np.linalg.LinAlgError:
                return None
            delta = np.clip(delta, [-15, -15, -np.radians(20)],
                            [15, 15, np.radians(20)])
            p = p + delta
            p[2] = _wrap_angle(p[2])
            r = residuals(p, used).ravel()
            if np.max(np.abs(delta[:2])) < 0.05                     and abs(delta[2]) < np.radians(0.05):
                break  # 已收敛

        # 离群剔除一轮（误检防护；>=4 点才有剔除的意义）
        per_point = np.linalg.norm(
            r.reshape(-1, 2), axis=1)
        if DEBUG_GN:
            cells_dbg = ", ".join(
                f"格{self.digit_cell.get(d)}@({c[0]:.0f},{c[1]:.0f})h{int(h)}"
                for (c, _, h, d) in [] )  # 占位
            print(f"[GN调试] 解=({p[0]:.1f},{p[1]:.1f},{np.degrees(p[2]):.1f}°) "
                  f"先验=({p0[0]:.1f},{p0[1]:.1f},{np.degrees(np.asarray(p0[2])):.1f}°) "
                  f"逐点残差={per_point.round(1)}")
        if used.sum() >= 4 and np.max(per_point) > GN_OUTLIER_PX:
            idx = np.where(used)[0]
            used[idx[per_point > GN_OUTLIER_PX]] = False
            if used.sum() < 2:
                print("[定位] 离群剔除后可用面板不足，丢弃")
                return None
            r = residuals(p, used).ravel()
            for it in range(GN_MAX_ITERS):
                J = np.zeros((r.size, 3))
                for j in range(3):
                    eps = [0.5, 0.5, np.radians(0.5)][j]
                    pp = p.copy(); pp[j] += eps
                    r1 = residuals(pp, used)
                    pp = p.copy(); pp[j] -= eps
                    r2 = residuals(pp, used)
                    J[:, j] = ((r1 - r2) / (2 * eps)).ravel()
                try:
                    delta = np.linalg.solve(J.T @ J + 1e-6 * np.eye(3),
                                            -(J.T @ r))
                except np.linalg.LinAlgError:
                    return None
                delta = np.clip(delta, [-15, -15, -np.radians(20)],
                                [15, 15, np.radians(20)])
                p = p + delta
                p[2] = _wrap_angle(p[2])
                r = residuals(p, used).ravel()

        rms = float(np.sqrt(np.mean(r ** 2))) if r.size else 1e9
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
        for attempt in range(RETRY_LIMIT + 1):
            if time.time() > t_end:
                print(f"[面板{digit}] 时间预算耗尽")
                return False
            if self._approach(digit, t_end) and self._enter(digit, t_end):
                return True
            if attempt < RETRY_LIMIT:
                print(f"[面板{digit}] 未到位，退格重试 {attempt + 1}/{RETRY_LIMIT}")
                self._act("back_one_step", 2)
        return False

    def _approach(self, digit, t_end):
        """前视闭环接近：对准→横偏→前进，至 MID_DIST 内交棒 ENTER"""
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
            obs, frame = loc
            fwd, lat, bearing = self.target_relative(digit)
            print(f"[接近] 纵向{fwd:+.1f} 横向{lat:+.1f} 航向差{bearing:+.1f}° (cm/度)")

            if abs(bearing) > ALIGN_TOL_DEG:
                self.do_turn(bearing, digit, fwd)
                prev_fwd = None  # 转向后纵向基准失效
                continue
            if abs(lat) > LAT_TOL_CM and fwd < 2 * FAR_DIST_CM:
                self.do_sidestep(lat)
                continue
            if fwd <= MID_DIST_CM:
                return True  # 交棒 ENTER

            # 前进（批量门控：远距+对准良好才批量，上限2步）
            n = 1
            if fwd > FAR_DIST_CM and abs(bearing) <= BATCH_MAX_ANGLE_DEG \
                    and abs(lat) <= LAT_TOL_CM:
                n = min(BATCH_MAX_STEPS,
                        max(1, int((fwd - MID_DIST_CM) // FORWARD_CM)))
                self._act("go_forward", n)
            else:
                n = 2 if fwd > 2 * MID_DIST_CM else 1
                self._act("go_forward_one_step", n)

            # 卡滞守卫：定位给出"行动后"纵向，与行动前比较
            new_fwd = self._refine_forward_quiet(digit, PITCH_NAV)
            if new_fwd is not None:
                if prev_fwd is not None and prev_fwd - new_fwd < STALL_EPS_CM:
                    stall += 1
                    if stall >= STALL_CONSEC:
                        print(f"[接近] 连续{STALL_CONSEC}步改善<{STALL_EPS_CM}cm，"
                              "判定卡滞：后退一步脱困")
                        self._act("back_one_step", 1)
                        stall = 0
                        prev_fwd = None
                        continue
                stall = 0
                prev_fwd = new_fwd
        print("[接近] 时间预算耗尽")
        return False

    def _enter(self, digit, t_end):
        """低头精细进入：横移+单步，至中心进入脚下判定带"""
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
                fwd, lat, bearing = self.target_relative(digit)
                print(f"[进入] 纵向{fwd:+.1f} 横向{lat:+.1f} 航向差{bearing:+.1f}° (cm/度)")

                if abs(bearing) > ENTER_ALIGN_TOL_DEG:
                    self.do_turn(bearing, digit, fwd)
                    prev_fwd = None
                    continue
                if abs(lat) > CONFIRM_LAT_MAX_CM:
                    self.do_sidestep(lat)
                    continue
                if fwd > ENTER_TOL_CM:
                    self._act("go_forward_one_step", 1)
                elif fwd < ENTER_MIN_CM:
                    print("[进入] 已过冲：回蹭")
                    self._act("back_one_step", 1)
                else:
                    return self._confirm(digit)

                # 卡滞守卫（仅前进迭代）
                new_fwd = self._refine_forward_quiet(digit, PITCH_DOWN)
                if new_fwd is not None and fwd > ENTER_TOL_CM:
                    if prev_fwd is not None \
                            and prev_fwd - new_fwd < STALL_EPS_CM:
                        stall += 1
                        if stall >= STALL_CONSEC + 1:
                            print("[进入] 卡滞：后退一步脱困")
                            self._act("back_one_step", 1)
                            stall = 0
                            prev_fwd = None
                            continue
                    stall = 0
                    prev_fwd = new_fwd
            print("[进入] 时间预算耗尽")
            return False
        finally:
            self.state.set_pitch(PITCH_NAV)

    def _confirm(self, digit):
        """到达确认：低头多帧色占比 + 蹭步来回压过微动开关"""
        color = ID_TO_COLOR[digit]

        def ratio_seen():
            seen = 0
            for _ in range(RATIO_FRAMES):
                frame = self.state.capture_frame()
                if frame is not None and \
                        self.detector.color_ratio(frame, color) > CONFIRM_RATIO:
                    seen += 1
            return seen >= (RATIO_FRAMES // 2 + 1)

        if not ratio_seen():
            print(f"[确认] 脚下未见面板{digit}颜色（{color}），交回导航")
            return False

        # 蹭步：后退3.2 + 前进2.0，来回覆盖中心点，确保触发微动开关
        self._act("back_one_step", 1)
        self._act("go_forward_one_step", 1)
        ok = ratio_seen()

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
        print(f"[确认] 面板{digit} 到达{'✓' if ok else '待定'}")
        return ok

    # =================================================================
    # 运动原语（白名单 + 自适应转向 + 预测推进）
    # =================================================================

    def _act(self, action, times=1):
        """动作执行 + 位姿预测推进（GN 先验的基础）"""
        self.state.act(action, times=times)
        self.predict_pose(action, times)
        if ACTION_MODEL.get(action, (None, 0))[0] != "turn":
            self._last_turn = None  # 期间发生平移，旧转向不可盲目撤销

    def do_turn(self, bearing, digit, fwd_cm=None):
        """按航向差转向（右正）。近目标先退格再转（防面板边缘转向）。"""
        if fwd_cm is not None and fwd_cm < TURN_NEAR_LIMIT_CM:
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
            per_step = (abs(bearing) - abs(new_bearing)) / n
            self._small_turn_deg = ((1 - TURN_EMA_ALPHA) * self._small_turn_deg
                                    + TURN_EMA_ALPHA * max(per_step, 0.05))
            if per_step < MIN_EFFECTIVE_TURN_DEG:
                print(f"[转向] 小转被地面吞掉（{per_step:.2f}°/次），升级大转")
                self._small_turn_usable = False
                self._big_turn(new_bearing)
            else:
                print(f"[转向] 小转{n}次 剩余{new_bearing:+.1f}° "
                      f"估计{self._small_turn_deg:.1f}°/次")

    def _big_turn(self, bearing):
        """大步转向兜底：量化到 22°/25.7°，过冲由下一轮闭环吸收"""
        if abs(bearing) <= ALIGN_TOL_DEG:
            return
        step = TURN_RIGHT_DEG if bearing > 0 else TURN_LEFT_DEG
        n = int(np.clip(round(abs(bearing) / step), 1, 3))
        action = "turn_right" if bearing > 0 else "turn_left"
        self._act(action, n)
        self._last_turn = (action, n)
        print(f"[转向] 大转 {action} x{n}（目标差 {bearing:+.1f}°）")

    def do_sidestep(self, lat):
        """横移纠偏。右正。"""
        n = int(np.clip(round(abs(lat) / RIGHT_MOVE_CM), 1, 3))
        action = "right_move" if lat > 0 else "left_move"
        self._act(action, n)
        print(f"[横移] {action} x{n}（横向差 {lat:+.1f}cm）")

    def _refine_forward_quiet(self, digit, pitch):
        """行动后补一帧定位，返回目标纵向距离（失败返回 None 不打断主流程）"""
        loc = self.localize(pitch)
        if loc is None:
            return None
        return self.target_relative(digit)[0]

    def _recover(self, fails):
        """定位丢失恢复：低头补扫→退格重扫→撤销上次转向。超限返回 False。"""
        if fails == 1:
            print("[恢复] 低头补扫")
            self.state.set_pitch(PITCH_DOWN)
            loc = self.localize(PITCH_DOWN)
            self.state.set_pitch(PITCH_NAV)
            return loc is not None
        if fails == 2:
            print("[恢复] 后退一步重扫")
            self._act("back_one_step", 1)
            return self.localize(PITCH_NAV) is not None
        if fails == 3 and self._last_turn is not None:
            # 大转过冲可能把场地转出视野——撤销上次转向回到可定位姿态
            action, n = self._last_turn
            undo = {"turn_left": "turn_right", "turn_right": "turn_left",
                    "turn_left_small_step": "turn_right_small_step",
                    "turn_right_small_step": "turn_left_small_step"}[action]
            print(f"[恢复] 撤销上次转向 {action}x{n} → {undo}x{n}")
            self._act(undo, n)
            self._last_turn = None
            return self.localize(PITCH_NAV) is not None
        print(f"[恢复] 连续 {fails} 次定位失败，放弃本段")
        return False


# =====================================================================
# 投影与工具
# =====================================================================

def project_ground_to_pixel(pts_ground, x, y, th_rad, pitch_pulse,
                            head_pulse=HEAD_CENTER):
    """格心(场地系 cm) → 像素：位姿 (x,y,θ) + 俯仰/头部档 + 真实内参/畸变

    头部偏航：脉宽左转(>1500)为正，相机方位角 = 机体航向 − 头部角
    （与 core/robot_core.pulse_to_angle、单测验证的补偿模型一致）。
    """
    a = np.radians((1500 - pitch_pulse) * SERVO_DEG_PER_US)
    f = th_rad - np.radians((head_pulse - HEAD_CENTER) * SERVO_DEG_PER_US)
    sa, ca = np.sin(a), np.cos(a)
    sf, cf = np.sin(f), np.cos(f)
    R = np.vstack([np.array([cf, -sf, 0.0]),                       # 图像右
                   np.array([-sa * sf, -sa * cf, -ca]),            # 图像下
                   np.array([ca * sf, ca * cf, -sa])])             # 视轴
    pts = np.atleast_2d(np.asarray(pts_ground, dtype=np.float64))
    if pts.shape[1] == 2:
        pts = np.column_stack([pts, np.zeros(len(pts))])  # 地面 z=0
    diff = pts - np.array([x, y, CAM_HEIGHT_CM])  # 相机光心在 z=39cm
    p_cam = (R @ diff.T).T
    norm = p_cam[:, :2] / p_cam[:, 2:3]
    pts3 = np.column_stack([norm, np.ones(len(norm))]).reshape(-1, 1, 3)
    pix, _ = cv2.projectPoints(pts3.astype(np.float32), np.zeros(3),
                               np.zeros(3), CAMERA_INTRINSIC,
                               CAMERA_DISTORTION)
    return pix[:, 0, :]


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
