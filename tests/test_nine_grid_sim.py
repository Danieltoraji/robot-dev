# -*- coding: utf-8 -*-
"""数字宫格仿真集成测试（tests/test_nine_grid_sim.py）

仿真环境（世界+合成相机+带噪声动作）在 `sim/nine_grid_sim.py`，本文件只做断言：
  1. 动作白名单硬门：本场地禁用的 go_forward / go_forward_fast 必须被拒；
  2. 渲染守卫：极端位姿/俯仰/头部组合不得产生退化多边形（曾 26s/帧）；
  3. 端到端：布局解算正确 + 1..7 全部确认到达 + 拍照数护栏。

运行（PC 上跑；不进快速单测）：
    python tests/test_nine_grid_sim.py
    python -m tests.test_nine_grid_sim
    python -m sim.nine_grid_sim          # 只看仿真摘要（CLI 入口）
"""

import os
import sys
import time

# 允许从任意目录直接运行本文件（脚本目录在 sys.path[0]，仓库根不在）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from core.camera_config import (
    HEAD_CENTER, HEAD_WIDE_LEFT, HEAD_WIDE_RIGHT,
)

# =====================================================================
# ★ 2026-09-28：端到端仿真用例（整局 7 格）自带"几何不匹配"豁免
# =====================================================================
# 用户已明确：**以后不仿真，全部实机验证**。而相机高度改成实测 33.9cm 之后：
#   · 仿真世界跟着变（`SimNineGridRobot.CAM_HEIGHT` 从 nine_grid_shared 导入）；
#   · 到达门却是按**真机四张到达参考帧**重标的（`ARRIVE_ORANGE_MAX` 0.10→0.30、
#     "够大/宽高比"两条按用户要求短路）——那批标定的几何是**真机的 33.9cm**，
#     而仿真渲染与那四张图并不一致 ⇒ 端到端跑不出 7/7（实测卡在面板1）。
# 实测证据：`NINEGRID_CAM_HEIGHT_CM=56 python tests/test_nine_grid_sim.py` 通过；
#   缺省（33.9）失败。**这不是导航逻辑回归**，所以这里不再让它伪装成红灯：
#   · 单测级用例（白名单/渲染守卫）任何时候都跑；
#   · 端到端用例只在"仿真几何 == 标定几何"时跑，否则**跳过并说明原因**，
#     同时给出仍然可用的跑法（`NINEGRID_CAM_HEIGHT_CM=56` 复现旧几何）。
# 什么时候可以删掉这段豁免：真机高度再变、并且用真机参考帧重新标定完之后，
#   把下面的 _SIM_GEOMETRY_MATCHES 判断去掉即可（或把它接成"与参考帧自洽"的检查）。
# =====================================================================
CAM_HEIGHT_CALIB_CM = 33.9      # 到达门标定所用几何（真机实测高度）
SIM_OLD_GEOMETRY_CM = 56.0      # 旧仿真几何（端到端用例在该几何下自洽）


def _sim_geometry_matches():
    """仿真世界的高度是否等于到达门标定所用的几何"""
    from levels.nine_grid_shared import CAM_HEIGHT_CM
    return abs(float(CAM_HEIGHT_CM) - CAM_HEIGHT_CALIB_CM) < 0.5


SKIP_MARK = "[跳过]"


def _skip_e2e(what):
    print(f"  {SKIP_MARK} {what}：仿真几何（h={_cam_h():.1f}cm）与到达门标定几何"
          f"（h={CAM_HEIGHT_CALIB_CM}cm）不一致 —— 见本文件顶部豁免说明；"
          f"要看旧几何下的端到端：NINEGRID_CAM_HEIGHT_CM=56 python {__file__}")


def _cam_h():
    from levels.nine_grid_shared import CAM_HEIGHT_CM
    return float(CAM_HEIGHT_CM)


from levels.nine_grid_shared import NineGridShared, PITCH_NAV, PITCH_DOWN  # noqa: E402
from sim.nine_grid_sim import SimNineGridRobot, run_simulation  # noqa: E402

# 拍照数护栏（真机时间预算的代理指标；超限说明 FSM 在空转）
# 基线 215 张（2026-09-08：裁切感知观测 + min_panels=1 + 20cm 切低头 + 禁用 go_forward）
CAPTURE_GUARD = 350


def test_banned_actions_rejected():
    """动作白名单硬门：本场地禁用的大步幅动作必须被拒绝（防误用摔倒）"""
    robot = SimNineGridRobot()
    level = NineGridShared(robot)
    for action in ("go_forward", "go_forward_fast", "climb_stairs"):
        try:
            level._act(action, 1)
        except ValueError:
            continue
        raise AssertionError(f"{action} 未被白名单拒绝")
    level._act("go_forward_one_step", 1)  # 白名单内动作应可执行
    print("  动作白名单：go_forward/go_forward_fast 被拒，one_step 可执行 ✓")


def test_render_guard_fast():
    """渲染守卫回归：角点贴近相机平面时不得产生退化多边形（曾 26s/帧）

    背景：z_cam 略大于 0.05 的角点投影为 ±inf，astype(int32) 得 INT_MIN，
    cv2.fillPoly 会扫描 ~2^31 行。此处遍历"格内/格间 + 两档俯仰 + 广角头部"
    组合，断言单帧渲染 <0.1s，防止该缺陷回归。
    """
    robot = SimNineGridRobot()
    # 预热：首帧要付 OpenCV/内存分配的启动成本（实测首帧可达 0.2s，
    # 与"退化多边形"无关），先空跑两帧再开始计时，否则测的是启动成本。
    robot.pos = np.array([50.0, -20.0])
    robot.heading, robot.pitch, robot.head = 0.0, PITCH_NAV, HEAD_CENTER
    for _ in range(2):
        robot.capture_frame()
    slow = []
    for (x, y) in ((50, -20), (50, 10), (50, 30), (50, 45), (20, 45), (80, 45)):
        for heading in (0.0, 90.0, -90.0):
            for pitch in (PITCH_NAV, PITCH_DOWN):
                for head in (HEAD_CENTER, HEAD_WIDE_LEFT, HEAD_WIDE_RIGHT):
                    robot.pos = np.array([float(x), float(y)])
                    robot.heading = heading
                    robot.pitch = pitch
                    robot.head = head
                    t0 = time.perf_counter()
                    robot.capture_frame()
                    dt = time.perf_counter() - t0
                    if dt > 0.1:
                        slow.append((x, y, heading, pitch, head, dt))
    assert not slow, f"渲染超时（退化多边形回归）: {slow[:5]}"
    print("  渲染守卫：极端位姿/俯仰/头部组合全部 <0.1s ✓")


def test_sim_full_run():
    """端到端：默认布局 + 默认种子必须布局正确、7/7 到达、不越护栏"""
    if not _sim_geometry_matches():
        _skip_e2e("三段式端到端")
        return
    run = run_simulation(three_stage=True)
    s = run.stats

    assert s["layout_ok"], \
        f"布局解算错误: {s['digit_cell']} != 真值 {s['truth']}"

    failed = [k for k, ok in s["results"] if not ok]
    assert s["ok_all"] and not failed, \
        f"未确认到达的面板: {failed}，结果 {s['results']}"

    # 本场地禁用 go_forward / go_forward_fast：白名单硬门之外再加一条行为
    # 断言，防止后续改动绕过 _act 直接调 state.act
    assert not s["banned_used"], \
        f"使用了本场地禁用的大步幅动作: {s['banned_used']}"

    assert s["captures"] < CAPTURE_GUARD, \
        f"拍照次数 {s['captures']} 超护栏 {CAPTURE_GUARD}，检查是否出现定位风暴"

    print(f"\n[仿真] 全部 7 格到达确认 ✓  拍照 {s['captures']} 张，"
          f"动作 {s['actions']} 次")
    print("[仿真] 动作统计:", s["action_counts"])
    print("[仿真] 小转最终估计: "
          f"{s['small_turn_deg']:.1f}°/次 "
          f"({'可用' if s['small_turn_usable'] else '已弃用(大转兜底)'})")


def test_unified_full_run():
    """端到端（**默认办法**）：统一决策也必须 7/7、布局正确、不越护栏

    为什么单列一条：默认办法此前没有端到端门（只有形变场景下的安全门），
    而它是 `python main.py nine_grid` 真正跑的那条。2026-09-25 取消"耗时太长
    就放弃这一格"之后，它从 6/7（面板 4 拍照撞上限被记未确认）变成 7/7——
    这一条就是那次策略改动的端到端证据。

    护栏 900 张：实测 652 张（三段式 189），留 ~38% 余量；它拦的是"定位风暴"，
    不是"磨了几次"（磨本身是策略允许的）。仿真保险丝另在 3000 张兜底。
    """
    if not _sim_geometry_matches():
        _skip_e2e("统一决策端到端")
        return
    run = run_simulation(three_stage=False)
    s = run.stats

    assert s["layout_ok"], \
        f"统一决策布局解算错误: {s['digit_cell']} != 真值 {s['truth']}"
    failed = [k for k, ok in s["results"] if not ok]
    assert s["ok_all"] and not failed, \
        f"未确认到达的面板: {failed}，结果 {s['results']}"
    assert not s["banned_used"], \
        f"使用了本场地禁用的大步幅动作: {s['banned_used']}"
    assert not s["fuse_tripped"], f"不该走到仿真保险丝：{s['fuse_tripped']}"
    assert s["captures"] < 900, \
        f"拍照次数 {s['captures']} 超护栏 900，检查是否出现定位风暴"
    # ★ 顺序计分：确认的格必须是 1..7 全连续（绝不跳格）
    seq = [d for d, ok in s["results"] if ok]
    assert seq == list(range(1, 8)), f"顺序断了：确认了 {seq}"

    print(f"\n[仿真·统一决策] 全部 7 格到达确认 ✓  拍照 {s['captures']} 张，"
          f"动作 {s['actions']} 次；逐格尝试次数 {s['attempt_log']}")


if __name__ == "__main__":
    test_banned_actions_rejected()
    test_render_guard_fast()
    test_sim_full_run()
    test_unified_full_run()
    if _sim_geometry_matches():
        print("数字宫格仿真集成测试通过 ✓")
    else:
        print(f"仿真单测通过 ✓（端到端用例按几何不匹配豁免："
              f"h={_cam_h():.1f}cm ≠ 标定 {CAM_HEIGHT_CALIB_CM}cm —— "
              f"见本文件顶部说明；用户已定「以后不仿真，全部实机验证」）")
