#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一决策的"感知/判档"公共件（tests/test_nine_grid_unified_zone.py）

P1 重写（2026-09-25）把三档分区从"画幅中线"改为**机体系中线**，本文件钉住这件事：

1. **安装前偏必须折算**：相机相对机体前偏（运行期自标定实测 +19.0°，名义 18.5°）
   ⇒ "机体正前方"落在画幅 `W/2 + (安装偏移+头部角)×W/FOV` 列，**不是** `W/2`。
   漏掉它 = 判档中线整体偏 ~273px（19°@60° 增益），"直行"会朝机体斜前方走。
2. **判档只吃像素**：档位由目标在画面里的位置决定（用户示意图语义），不吃 yaw 数值、
   不吃框宽、不吃距离。
3. **一次决策一个原语**：档位 → 动作名，且动作名一律是"单步"原语（用户明确放弃批量）。
4. **`_capture_and_measure` 一次拍照给齐三个量**：目标观测 + 四区份额（判档与到达判据同源）。
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.camera_config import (  # noqa: E402
    HEAD_CENTER, HEAD_LEFT, HEAD_RIGHT, SERVO_DEG_PER_US,
)
import levels.nine_grid as NG  # noqa: E402
from levels.nine_grid_shared import ACTION_MODEL  # noqa: E402

W, H = 2592.0, 1944.0


class _StubState:
    """只提供公共件需要的 state 接口（不碰真机/仿真）"""

    def __init__(self, head=HEAD_CENTER):
        self.current_head_pulse = head
        self.frames = 0

    def capture_frame(self):
        self.frames += 1
        # 纯灰帧：没有目标色 ⇒ 观测为空，但四区份额可算（0）
        return np.full((int(H), int(W), 3), 90, np.uint8)


def _level(head=HEAD_CENTER, off_deg=None):
    lv = NG.NineGridLevel(_StubState(head))
    if off_deg is not None:
        lv._pitch_offset_deg = float(off_deg)
    return lv


# =====================================================================
# 1. 机体系中线（安装前偏）
# =====================================================================

def test_center_column_tracks_head_angle():
    """头部转向时机体系中线跟着移；符号必须与 `_zone_px_of` 的头部折算一致

    约定（`core/camera_config`）：`HEAD_LEFT = 1950 > HEAD_CENTER = 1500`，
    `head_deg = (pulse-1500)×0.09` ⇒ **脉宽大 = 头左转 = head_deg 为正**。
    `_center_column_px` 与 `_zone_px_of` 都用 `+head_deg×W/FOV`（同一个"头部转过去、
    目标在画面里反向移动"的补偿方向），所以这里只钉"跟着动 + 两边一致"。
    """
    lv = _level(head=HEAD_CENTER)
    base = lv._center_column_px(W)
    lv.state.current_head_pulse = HEAD_LEFT
    left_pulse = lv._center_column_px(W)
    lv.state.current_head_pulse = HEAD_RIGHT
    right_pulse = lv._center_column_px(W)
    d = abs(HEAD_LEFT - HEAD_CENTER) * SERVO_DEG_PER_US * W / 60.0
    assert left_pulse == pytest.approx(base + d)
    assert right_pulse == pytest.approx(base - d)
    assert left_pulse > base > right_pulse
    # 与既有头部折算同号：把"头偏 δ 时目标在画面里位移 −δ"补回去 ⇒ 同一个机体方向
    lv.state.current_head_pulse = HEAD_LEFT
    assert lv._body_angle_deg(left_pulse, W) == pytest.approx(0.0, abs=1e-9)


def test_body_angle_is_zero_at_center_column():
    """目标落在机体系中线上 ⇒ 机体系方位角为 0（这才是"正前方"）"""
    lv = _level()
    assert lv._body_angle_deg(lv._center_column_px(W), W) == pytest.approx(0.0, abs=1e-9)
    assert lv._body_angle_deg(W / 2.0, W) == pytest.approx(0.0, abs=1e-9)


def test_pitch_offset_is_elevation_not_yaw():
    """★ 回归闸：`_pitch_offset_deg` **不得**影响横向（判档中线/机体系方位角）

    它只进俯角 `alpha = (1500-pulse)*SERVO_DEG_PER_US + pitch_offset_deg`
    （见 `project_ground_to_pixel`），是"相机比名义再低头多少度"。
    日志把它写成"相机相对机体**前偏**"是**措辞错误**——曾据此想给判档补一个
    横向偏置，仿真实测那样会注入 ≈19°×19.7px/° ≈ 374px 的假偏置。

    做法：同一个像素列，`_pitch_offset_deg` 取 0 / 18.5 / 19 必须给出**一个数都不差**的
    中线、方位角与档位。若哪天有人把该量挪作 yaw 用，这条会红。
    """
    outs = []
    for off in (0.0, 18.5, 19.0, -15.0):
        lv = _level(off_deg=off)
        outs.append((lv._center_column_px(W),
                     lv._body_angle_deg(W / 2.0 + 200.0, W),
                     lv._zone_at_body_pixel(W / 2.0 + 200.0, 0.99 * H, W, H,
                                         prev=None)))
    for o in outs[1:]:
        assert o[0] == pytest.approx(outs[0][0], abs=1e-12)
        assert o[1] == pytest.approx(outs[0][1], abs=1e-12)
        assert o[2][0] == outs[0][2][0]
        assert o[2][1] == pytest.approx(outs[0][2][1], abs=1e-12)


# =====================================================================
# 2. 判档语义（与旧版的关系）
# =====================================================================

def test_body_zone_equals_old_zone_when_head_centered():
    """头居中时，机体系判档与旧画幅中线判档**逐点一致**（只多了头部折算这一项）"""
    lv = _level(head=HEAD_CENTER)
    for dxf in (0.02, 0.08, 0.13, 0.20, 0.30, 0.40):
        for t in (0.10, 0.50, 0.80, 0.99):
            px = W / 2.0 + dxf * W
            py = t * H
            a, da = lv._zone_at_body_pixel(px, py, W, H, prev=None)
            b = lv._zone_at_pixel(px, py, W, H, None)
            assert a == b, (dxf, t, a, b)


def test_head_off_center_shifts_the_corridor():
    """偏头拍摄时"画面正中"不再是机体正前方 ⇒ 判档必须跟着头走"""
    px, py = W / 2.0, 0.99 * H
    lv = _level(head=HEAD_CENTER)
    assert lv._zone_at_body_pixel(px, py, W, H, prev=None)[0] == "forward"
    lv.state.current_head_pulse = HEAD_LEFT          # 头左转 ⇒ 中线上移
    assert lv._zone_at_body_pixel(px, py, W, H, prev=None)[0] != "forward"
    # 放到"机体正前方"那一列就回到直行档
    assert lv._zone_at_body_pixel(lv._center_column_px(W), py, W, H,
                               prev=None)[0] == "forward"


def test_zone_uses_pixels_only_not_angle():
    """判档只吃像素：把"头偏 δ、目标在画面里反向移 δ"这两种拍法喂进去，档位相同"""
    lv = _level(head=HEAD_CENTER)
    d_deg = 9.0
    d_px = d_deg * W / 60.0
    a = lv._zone_at_body_pixel(W / 2.0 + 200.0, 0.99 * H, W, H, prev=None)
    b = lv._zone_at_body_pixel(W / 2.0 + 200.0 + d_px, 0.99 * H, W, H,
                            head_pulse=HEAD_CENTER + int(round(
                                d_deg / SERVO_DEG_PER_US)), prev=None)
    assert a[0] == b[0]
    assert a[1] == pytest.approx(b[1], abs=1e-9)


# =====================================================================
# 3. 档位 → 动作（一次一个原语）
# =====================================================================

def test_action_for_zone_is_single_step_primitive():
    """绿→前进、蓝→横移、橙→小转；且都是**单步**原语（无批量）"""
    lv = _level()
    assert lv._action_for_zone("forward", W / 2.0, W) == "go_forward_one_step"
    assert lv._action_for_zone("side", W / 2.0 - 10, W) == "left_move"
    assert lv._action_for_zone("side", W / 2.0 + 10, W) == "right_move"
    assert lv._action_for_zone("turn", W / 2.0 - 10, W) == "turn_left_small_step"
    assert lv._action_for_zone("turn", W / 2.0 + 10, W) == "turn_right_small_step"


def test_action_names_are_in_action_model():
    """动作名必须在 ACTION_MODEL 里（否则 `_act` 会 KeyError）"""
    lv = _level()
    for zone in ("forward", "side", "turn"):
        for side in (-1, 1):
            a = lv._action_for_zone(zone, W / 2.0 + side * 10, W)
            assert a in ACTION_MODEL, a


# =====================================================================
# 4. 一次拍照给齐三个量
# =====================================================================

def test_capture_and_measure_returns_shares_on_blank_frame():
    """纯灰帧：观测为空，但 shares 仍要给出（到达判据不能因为"没看到面板"就失踪）"""
    lv = _level()
    lv._cell_frames = 0
    p = lv._capture_and_measure("red")
    assert p is not None
    assert p["obs"] is None and p["px"] is None and p["box"] == 0.0
    assert p["shares"] is not None
    whole, purple, orange, asym = p["shares"]
    assert whole == pytest.approx(0.0)
    assert purple == 0.0 and orange == 0.0 and asym == 0.0
    assert p["w"] == W and p["h"] == H
    assert p["frame"].shape[:2] == (int(H), int(W))


def test_capture_and_measure_counts_frames():
    """`_capture_and_measure` 必须记拍照数（遥测与进度提醒靠它）"""
    lv = _level()
    lv._cell_frames = 0
    lv._capture_and_measure("red")
    assert lv._cell_frames == 1


# =====================================================================
# 5. 格的同一性核验（用户方案 ②-a）
# =====================================================================
def _lv_with_anchor(cell_pose):
    """构造一个 level：digit_cell 给定，`_map_pose` 返回固定实测位姿"""
    lv = _level()
    lv.digit_cell = {1: 3}
    frame = np.zeros((10, 10, 3), np.uint8)
    if cell_pose is None:
        lv._map_pose = lambda obs, fr, why="": None
    else:
        lv._map_pose = lambda obs, fr, why="": {"pose": np.asarray(cell_pose,
                                                                    float)}
    return lv, frame


def test_verify_target_cell_accepts_measured_pose_on_the_cell():
    """实测位姿落在目标格心附近 ⇒ "yes"（放行）"""
    from core.ground_homography import grid_cell_center
    g = grid_cell_center(3)
    lv, frame = _lv_with_anchor((g[0] + 5.0, g[1], 0.0))     # 偏 5cm
    v, why = lv._verify_target_cell(frame, 1)
    assert v == "yes", why


def test_verify_target_cell_vetoes_when_measured_pose_is_elsewhere():
    """★ 实测位姿明显不在目标格 ⇒ "no"（这正是"站在别格"假到达）

    这条是用户方案 ②-a 的核心：份额/形状类判据在"站在别的格子上、看到一块完整
    且居中的同色面板"时**全部正常**（实测有一帧六量全合规、真值却离目标 100cm），
    只有"这一帧几何能否被地图解释"能分辨。
    """
    from core.ground_homography import grid_cell_center
    g = grid_cell_center(3)
    lv, frame = _lv_with_anchor((g[0] + 66.0, g[1], 0.0))    # 偏 66cm ≈ 两格
    v, why = lv._verify_target_cell(frame, 1)
    assert v == "no", why
    assert "16.7" in why or "格心" in why


def test_verify_target_cell_abstains_honestly_without_anchor():
    """锚不可用 ⇒ "unknown"（诚实弃权），绝不拿退化输入硬判"""
    lv, frame = _lv_with_anchor(None)
    v, why = lv._verify_target_cell(frame, 1)
    assert v == "unknown", why
    assert "锚" in why


def test_map_pose_correction_returns_a_two_tuple():
    """★ 契约闸：`_map_pose_correction` 返回 **(对应点列表, 观测列表) 二元组**

    钉这条是因为我连续用错：写成 `len(self._map_pose_correction(...))` 会**恒等于 2**
    （元组长度）而不是对应点数 —— 据此得出"锚只有 2 对对应点、可用率 8.3%"的
    **错误结论**，还差点去改阈值。正确用法是 `corr, keep = ...; len(corr)`。
    实测（修正后）：入口站位 7 个面板全未裁切 ⇒ **7 对**、锚可解（n_in=7）；
    站在格心低头 ⇒ 6 观测 / **7 对**、锚可解（n_in=5）。
    """
    from core.ground_homography import grid_cell_center
    lv = _level()
    lv.digit_cell = {1: 3}

    class _O:
        digit = 1
        clipped = False
        center_px = (100.0, 200.0)
        hull_poly = ()
        color_id = 1

    frame = np.zeros((100, 200, 3), np.uint8)
    out = lv._map_pose_correction([_O()], frame)
    assert isinstance(out, tuple) and len(out) == 2, (
        f"_map_pose_correction 应返回 (corr, keep) 二元组，实际 {type(out)}")
    corr, keep = out
    assert len(corr) == 1, "1 个未裁切观测应给出 1 对对应点"
    assert len(keep) == 1
    g, p = corr[0][0], corr[0][1]
    assert np.allclose(g, np.asarray(grid_cell_center(3), float)), g
    assert np.allclose(p, (100.0, 200.0)), p


# =====================================================================
# 6. 到达判决：**只吃画面量，位姿不参与**（2026-09-26 短路格的同一性核验）
# =====================================================================
# 背景：这道核验曾挂在到达门的"与"条件里，实测会拿不可信的位姿否决**正确**到达
# （面板 5 连试三次才认；形变场景实测位姿漂 105cm 而真值离格心只有 13cm）。
# 现在锚测量只是诊断探针（`ARRIVE_ON_CELL_PROBE`），**不改变控制流**。

class _Box:
    """最小观测桩：`_arrive_pixels_ok` 只读 bbox[2]、bbox[3]"""

    def __init__(self, bw, bh):
        self.bbox = (0.0, 0.0, float(bw), float(bh))


# 六条判据全过的一组值（紫 0.60 ≥0.35｜橙 0.02 ≤0.10｜不对称 0.10 ≤0.55｜
# 目标在中线｜整帧占比 0.10 ≥0.065｜宽高比 1.66 ≤2.15，均为实测中位数附近）
_OK_SHARES = (0.10, 0.60, 0.02, 0.10)
_OK_OBS = _Box(1000.0, 600.0)


def test_arrive_pixels_ok_accepts_a_clean_arrival_frame():
    """六条全过 ⇒ True（先确认判据本身没被改坏）"""
    lv = _level()
    assert lv._arrive_pixels_ok(_OK_SHARES, _OK_OBS, W / 2.0, W) is True


@pytest.mark.parametrize("shares,obs,px,why", [
    ((0.10, 0.30, 0.02, 0.10), _OK_OBS, W / 2.0, "紫区不够（0.30 < 0.35）"),
    ((0.10, 0.60, 0.40, 0.10), _OK_OBS, W / 2.0, "橙区太多（0.40 > 0.30）"),
    ((0.10, 0.60, 0.02, 0.70), _OK_OBS, W / 2.0, "左右不对称（0.70 > 0.55）"),
    ((0.10, 0.60, 0.02, 0.10), _OK_OBS, W / 2.0 + 0.30 * W, "目标没在中央"),
])
def test_arrive_pixels_ok_rejects_each_violation(shares, obs, px, why):
    """**仍在生效的**四条各自都能单独否决（逐条钉住，防止有人以为某条是摆设）

    ⚠️ 2026-09-28：门从六条减到四条 ——
      · "整帧占比 ≥0.065" 与 "宽/高 ≤2.15" **已按用户要求短路**（常量置 0.0）：
        前者在实测 33.9cm 几何下对真到达恒判 ✗（面板下沿出画幅 ⇒ whole 必然小），
        后者真到达簇在 0.5cm 相机高度改标定后落到 2.83~3.08（旧门 2.15）。
        见 levels/nine_grid.py 的 `ARRIVE_MIN_TARGET_COVER` /
        `ARRIVE_MAX_WIDTH_HEIGHT_RATIO` 留档注释与核对报告 §8.4。
      · 橙门 0.10 → **0.30**（按用户四张到达参考帧重标：期望到达态实测 0.164~0.252）。
    已短路的门**不再参与判决**，所以这里只剩下四条。
    """
    lv = _level()
    assert lv._arrive_pixels_ok(shares, obs, px, W) is False, why


def test_arrive_decision_ignores_measured_pose():
    """★ 短路的核心不变量：**到达判决与锚无关**

    同一个"六条全过"的画面证据，在①锚**直接抛异常**、②锚说"我在 66cm 外、
    根本不在目标格"两种极端下，判决都必须一样，且探针不许把异常抛出去
    （探针只记录、不改变控制流）。
    """
    lv = _level()
    lv.digit_cell = {1: 3}

    def _boom(obs, fr, why=""):
        raise AssertionError("到达判决不该读锚")

    lv._map_pose = _boom                                   # ① 锚炸了
    assert lv._arrive_pixels_ok(_OK_SHARES, _OK_OBS, W / 2.0, W) is True
    assert lv._arrive_probe_note(1, np.zeros((10, 10, 3), np.uint8)) is None

    lv2, frame = _lv_with_anchor((200.0, 200.0, 0.0))      # ② 锚说在别处
    assert lv2._arrive_pixels_ok(_OK_SHARES, _OK_OBS, W / 2.0, W) is True
    assert lv2._arrive_probe_note(1, frame) == "no", "探针应记录分歧，但不否决"


def test_arrive_evidence_has_no_cell_verification():
    """✓ 到达的那一行留档里**不许**再出现"格的核验"（防止把否决悄悄接回来）

    留档是现场排查的唯一凭据；只要它重新出现，就说明短路被回滚了一半。
    """
    lv, frame = _lv_with_anchor((200.0, 200.0, 0.0))
    lv._arrive_evidence = "脚下色块判据：紫区占比 0.600（≥0.35），" \
                          "橙区占比 0.020（≤0.1），左右不对称 +0.100（≤0.55）"
    lv._arrive_probe_note(1, frame)
    assert "核验" not in lv._arrive_evidence
    assert "锚" not in lv._arrive_evidence
