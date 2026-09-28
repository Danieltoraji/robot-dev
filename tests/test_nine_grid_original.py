# -*- coding: utf-8 -*-
"""参考原版通关代码（包 levels/nine_grid_original/）的回归测试（2026-09-26）

这一版是从 `reference code/九宫格视觉导航/` 原样搬进来的对照臂，所以测试的重点
不是"算法对不对"，而是三件事：

  ① **接入正确**：main.py 注册了 `nine_grid_original`（包入口），其它关卡原封不动；
  ② **不拖累别人**：import 本包不碰硬件、不加载 28MB 的 SVM 模型
     （否则 `python main.py goodluck` 会被拖慢甚至崩掉）；
  ③ **搬运没走样**：门限表、yaw 符号、自适应小转的升级逻辑、到达判据
     （"看到颜色 → 颜色消失"）与参考版逐条一致——这几处是原版通关的关键，
     抄错一个数就变成另一套算法了。

补丁打在 `...nine_grid_original.level` / `.vision` / `.classifier` / `.action`
这些**子模块**上（不是包上）——`level.run_level` 读的是自己模块的全局名，
打在包上会静默失效。全部用桩件（monkeypatch），不碰真相机、不碰硬件、不跑视觉。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import levels.nine_grid_original as PKG  # noqa: E402  包入口（main.py 注册的那个）
import levels.nine_grid_original.classifier as C  # noqa: E402
import levels.nine_grid_original.level as N  # noqa: E402
import levels.nine_grid_original.vision as V  # noqa: E402


# =====================================================================
# 桩件
# =====================================================================

class _StubActionLog:
    """把动作层换成记账本（不碰硬件）"""

    def __init__(self):
        self.calls = []

    def _rec(self, name):
        def _f(*a, **kw):
            self.calls.append(name)
        return _f

    def __getattr__(self, name):
        return self._rec(name)

    def count(self, name):
        return self.calls.count(name)


@pytest.fixture()
def actions(monkeypatch):
    log = _StubActionLog()
    for name in ("servo_look_forward", "servo_look_down", "move_forward",
                 "move_forward_small", "move_backward_one", "over_hurdle",
                 "turn_big_angle_left", "turn_big_angle_right",
                 "turn_small_angle_left", "turn_small_angle_right",
                 "init_servo", "reset_servo"):
        monkeypatch.setattr(N.action, name, log._rec(name))
    monkeypatch.setattr(N.time, "sleep", lambda _s: None)
    return log


def _patch_identify(monkeypatch, seq):
    """把 N.identify 换成按序返回的桩；序列用完就报错（防测试写错）"""
    it = iter(seq)
    used = []

    def _f(*a, **kw):
        used.append(1)
        try:
            return next(it)
        except StopIteration:
            raise AssertionError("identify 被调用的次数超过了测试给的序列")

    monkeypatch.setattr(N, "identify", _f)
    return used


def _patch_ratio(monkeypatch, seq):
    it = iter(seq)
    used = []

    def _f(*a, **kw):
        used.append(1)
        try:
            return next(it)
        except StopIteration:
            raise AssertionError("identify_color 被调用的次数超过了测试给的序列")

    monkeypatch.setattr(N, "identify_color", _f)
    return used


# =====================================================================
# ① 接入正确
# =====================================================================

def test_registered_in_main():
    import main
    assert "nine_grid_original" in main.LEVELS
    entry = main.LEVELS["nine_grid_original"]
    # main.py 注册的是**包**，入口函数即包里再导出的 run_level
    assert entry["module"] is PKG
    assert entry["run_level"] is PKG.run_level
    assert entry["run_level"] is N.run_level
    assert entry["tag_poses"] == {}


def test_other_levels_untouched():
    """注册是纯新增：其它四个关卡一个不少，且现行九宫格常量没被动过"""
    import main
    expected = {
        "goodluck", "nine_grid", "nine_grid_three_stage",
        "nine_grid_original", "stairs_hurdle"}
    # press_button 的代码在另一条分支（.worktrees/press_button），本分支没有它
    # ⇒ 本分支不注册；机器人上有该模块 ⇒ 那边多一个 press_button 入口，这不算
    # "动了别的关卡"。据此按模块是否在来放宽断言（2026-09-27）。
    try:
        import levels.press_button  # noqa: F401
        expected.add("press_button")
    except ImportError:
        pass
    assert set(main.LEVELS) == expected
    assert main.PRESS_BUTTON_SKIP_REASON is None or "press_button" not in main.LEVELS

    import levels.nine_grid_shared as SH
    assert SH.FORWARD_ONE_STEP_CM == 2.652
    assert SH.TURN_LEFT_SMALL_DEG == 8.625
    assert SH.TURN_RIGHT_SMALL_DEG == 5.200
    assert SH.PITCH_DOWN == 1040


# =====================================================================
# ② 不拖累别人：import 期不碰硬件、不加载模型
# =====================================================================

def test_import_does_not_load_model():
    """28MB 的 SVM 模型必须懒加载：import 期不能落到 _classifier 上"""
    assert C._classifier is None
    assert C._load_failed is False


def test_blank_image_short_circuits_before_model():
    """空图没有任何候选 ⇒ 直接返回空表，连模型都不去碰"""
    blank = np.zeros((120, 160, 3), dtype=np.uint8)
    assert C.classify_candidates(blank) == []
    assert C._classifier is None


def test_hardware_error_is_clear(monkeypatch):
    """非机器人环境下动作层给出可读报错（而不是 import 期就崩）"""
    monkeypatch.setattr(N.action, "AGC", None)
    monkeypatch.setattr(N.action, "ctl", None)
    monkeypatch.setattr(N.action, "_STATE", None)
    with pytest.raises(RuntimeError, match="hiwonder"):
        N.action.move_backward_one()


# =====================================================================
# ③ 搬运没走样：门限表 / yaw 符号 / 到达判据 / 自适应小转
# =====================================================================

def test_go_to_params_copied_verbatim():
    """门限表逐字对照参考版 go_to() 里的 7 段 if/elif

    ⚠️ 2026-09-28：数字 1（0.16/0.10 → **0.1/0.05**）与数字 4（0.12 → **0.1**）
    的 COLOR/COLOR_LOST 不再逐字等于参考版——现场标定把这两格的"看到颜色 /
    颜色消失"门限往下调了（参考版那两张表在真机上会提前判"压过"）。
    其余 5 格仍逐字一致。
    """
    assert N.GO_TO_PARAMS == {
        1: (450, 0.1, 0.05),
        2: (450, 0.05, 0.01),
        3: (370, 0.05, 0.015),
        4: (450, 0.14, 0.1),
        5: (450, 0.17, 0.10),
        6: (450, 0.002, 0.003),
        7: (450, 0.15, 0.10),
    }


def test_yaw_is_left_positive():
    """参考版约定：目标在画面左侧 ⇒ yaw > 0（与本仓库现行九宫格相反）"""
    w = 1280
    left = V.calculate_yaw((100, 0, 200, 200), w)     # 中心在画面左半边
    right = V.calculate_yaw((1000, 0, 200, 200), w)   # 中心在画面右半边
    center = V.calculate_yaw((w // 2 - 50, 0, 100, 100), w)
    assert left > 0 > right
    assert center == pytest.approx(0.0, abs=1e-9)


def test_proximity_levels():
    assert V.proximity_level(V.NEAR_THRESHOLD) == "NEAR"
    assert V.proximity_level(V.NEAR_THRESHOLD - 1) == "MID"
    assert V.proximity_level(V.MID_THRESHOLD) == "MID"
    assert V.proximity_level(V.MID_THRESHOLD - 1) == "FAR"


def test_id_to_color_covers_1_to_7():
    assert [V.id_to_color(i).value for i in range(1, 8)] == [1, 2, 3, 4, 5, 6, 7]
    assert V.id_to_color(0) is None
    assert V.id_to_color(8) is None


def test_turn_to_aligned_returns_proximity(actions, monkeypatch):
    _patch_identify(monkeypatch, [(True, 3.0, 260)])
    assert N.turn_to(5, object()) == 260
    assert actions.calls == []          # 已对准 ⇒ 一个动作都不发


def test_turn_to_big_turn_then_aligned(actions, monkeypatch):
    """>30° 走大转（左侧为正 ⇒ 左转），转完复测对准即返回"""
    _patch_identify(monkeypatch, [(True, 45.0, 100), (True, 0.0, 400)])
    monkeypatch.setattr(N, "capture_image", lambda: object())
    assert N.turn_to(5, object()) == 400
    assert actions.count("turn_big_angle_left") == 1
    assert actions.count("turn_big_angle_right") == 0


def test_turn_to_escalates_when_small_turns_ineffective(actions, monkeypatch):
    """一批小转平均每步 < 0.5° ⇒ 判为小转失效，升级为一次大转（原版核心机制）"""
    _patch_identify(monkeypatch, [
        (True, 20.0, 100),    # 首次：20° ⇒ 小转分支，估计 4°/步 ⇒ 转 5 次
        (True, 19.8, 100),    # 小转后几乎没动 ⇒ 0.04°/步 < 0.5 ⇒ 升级大转
        (True, 19.8, 100),    # 大转后复测仍在 ⇒ 走段 4 再复测
        (True, 0.0, 500),     # 对准
    ])
    monkeypatch.setattr(N, "capture_image", lambda: object())
    assert N.turn_to(5, object()) == 500
    assert actions.count("turn_small_angle_left") == 5
    assert actions.count("turn_big_angle_left") == 1


def test_search_target_raises_after_20_turns(actions, monkeypatch):
    """20 次大转仍找不到 ⇒ 抛 RuntimeError（原版行为，不是死循环）"""
    _patch_identify(monkeypatch, [(False, None, None)] * (N.SEARCH_MAX_TURNS * 2))
    monkeypatch.setattr(N, "capture_image", lambda: object())
    with pytest.raises(RuntimeError, match="未找到目标数字"):
        N.search_target(3)
    assert actions.count("turn_big_angle_left") == N.SEARCH_MAX_TURNS


def test_arrived_needs_real_color_loss(actions, monkeypatch):
    """到达判据：看到颜色后，只有真掉到丢失门以下才算到达

    序列 [0.20, 0.05, 0.02, 0.004]、丢失门 0.01：
      · 帧1 0.20 > 看到门 ⇒ seen_color=True；
      · 帧2/帧3 落在"看到门与丢失门之间" ⇒ **不算**颜色消失（原版刻意留的中间带）；
      · 帧4 0.004 < 丢失门 ⇒ 判到达。
    """
    ratio_calls = _patch_ratio(monkeypatch, [0.20, 0.05, 0.02, 0.004])
    _patch_identify(monkeypatch, [(True, 0.0, 300)] * 4)
    monkeypatch.setattr(N, "capture_image", lambda: object())

    assert N.arrived(1, 0.16, 0.01) is True
    assert len(ratio_calls) == 4


def test_arrived_on_target_lost_after_seeing_color(actions, monkeypatch):
    """另一条到达路径：已经看到过颜色之后，数字识别丢失也判到达（原版第 6 段）"""
    ratio_calls = _patch_ratio(monkeypatch, [0.20, 0.20])
    _patch_identify(monkeypatch, [(True, 0.0, 300), (False, None, None)])
    monkeypatch.setattr(N, "capture_image", lambda: object())

    assert N.arrived(1, 0.16, 0.01) is True
    assert len(ratio_calls) == 2
    # 判到达时云台要回到导航姿态（原版行为）
    assert actions.count("servo_look_forward") == 1


# =====================================================================
# 逐格入口：顺序、失败即收手
# =====================================================================

def test_run_level_goes_1_to_7_in_order(monkeypatch):
    calls = []

    def _go_to(digit, *a, **kw):
        calls.append(digit)

    monkeypatch.setattr(N, "go_to", _go_to)
    monkeypatch.setattr(N.vision, "bind_state", lambda _s: None)
    monkeypatch.setattr(N.action, "bind_state", lambda _s: None)
    monkeypatch.setattr(N.action, "init_servo", lambda: None)

    assert N.run_level(object()) is True
    assert calls == [1, 2, 3, 4, 5, 6, 7]


def test_run_level_stops_when_target_not_found(monkeypatch):
    """搜索失败（顺序计分下不跳格）⇒ 干净失败返回 False，而不是接着做下一格"""
    calls = []

    def _go_to(digit, *a, **kw):
        calls.append(digit)
        if digit == 4:
            raise RuntimeError("未找到目标数字4")

    monkeypatch.setattr(N, "go_to", _go_to)
    monkeypatch.setattr(N.vision, "bind_state", lambda _s: None)
    monkeypatch.setattr(N.action, "bind_state", lambda _s: None)
    monkeypatch.setattr(N.action, "init_servo", lambda: None)

    assert N.run_level(object()) is False
    assert calls == [1, 2, 3, 4]


# =====================================================================
# 数字融合：模型缺位时降级为纯颜色（参考版会直接崩）
# =====================================================================

def _fake_candidate():
    return {
        "roi": np.zeros((10, 10, 3), np.uint8),
        "digit_mask": np.zeros((10, 10), np.uint8),
        "box": (0, 0, 10, 10),
        "score": 0.8,
        "hole_count": 3,
        "hole_ratio": 0.2,
        "color": "blue",
        "color_id": 5,
    }


def test_fusion_color_beats_weak_model(monkeypatch):
    """颜色权重 0.7 恒定高于模型那一票 0.2×conf ⇒ 颜色是主判"""
    monkeypatch.setattr(C, "extract_digit_roi", lambda _img: [_fake_candidate()])
    monkeypatch.setattr(C, "predict_digit", lambda _m: (3, 0.7))

    out = C.classify_candidates(np.zeros((20, 20, 3), np.uint8))
    assert len(out) == 1
    assert out[0]["final_digit"] == 5      # 颜色说了算
    assert out[0]["model_digit"] == 3
    assert out[0]["digit_match"] is False


def test_model_missing_degrades_to_color_only(monkeypatch):
    """依赖/模型缺失 ⇒ model_digit=None，仍按颜色给 final_digit（不崩）"""
    monkeypatch.setattr(C, "extract_digit_roi", lambda _img: [_fake_candidate()])
    monkeypatch.setattr(C, "get_classifier", lambda: None)

    out = C.classify_candidates(np.zeros((20, 20, 3), np.uint8))
    assert out[0]["final_digit"] == 5
    assert out[0]["model_digit"] is None
    assert out[0]["model_conf"] == 0.0


# =====================================================================
# 合成图端到端：颜色分割 → 候选 → identify（强制纯颜色，不加载 28MB 模型）
# =====================================================================

def test_synthetic_red_panel_is_identified_on_the_left(monkeypatch):
    """左侧放一块红牌 ⇒ identify(1) 认到、yaw > 0（左侧为正）、proximity = 框宽"""
    import cv2

    monkeypatch.setattr(C, "get_classifier", lambda: None)

    img = np.full((980, 1280, 3), 200, np.uint8)
    cv2.rectangle(img, (150, 300), (550, 700), (0, 0, 255), -1)   # BGR 红

    success, yaw, proximity = V.identify(1, img)
    assert success is True
    assert proximity == 401          # 框宽 px（150..550 含端点）
    assert yaw > 0                   # 目标在画面左半边 ⇒ 参考版符号下为正

    # 该图上没有 5 号（蓝）目标 ⇒ 认不到
    assert V.identify(5, img) == (False, None, None)
