# -*- coding: utf-8 -*-
"""地板形变鲁棒性集成测试（tests/test_nine_grid_deform.py）

背景（2026-09-11 用户现场约束）：本关地板会形变，机器人在板上行走时机体俯仰
/相机高度会发生**大幅变化**（按 ±15° 考虑）。任何"冻结相机常数 + 度量投影"
的判据在这种形变下都会失效——量化：h=50cm 时可见地面带在有效俯角
42°/57°/72° 下分别是 19~184 / 5~86 / 0~50 cm（4 倍范围摆动）。
故导航必须走**视觉伺服**（像素偏移/框宽/颜色占比等相对量）。

本测试在仿真里注入形变（只改世界侧的相机姿态/高度，关卡内常数保持不动）：
  1. 基线（无形变）；
  2. 随机游走：σ=3°/动作 俯仰 + σ=0.5cm/动作 高度（限幅 ±15° / ±2cm）；
  3. 一次性阶跃：**第 3 格开始时** +15° 俯仰（模拟"踩上/离开板"）；
  4. 同上但落在第 5 格（确认不是只对某一格调过参）。
计分场景都要求：布局正确 + 1..7 全部确认到达 + **落点真值**不超半格 +
拍照数 < 护栏 + 没有被安全项终止的格（`cell_trips` 为空）。

**为什么阶梯场景改用 `step_at_digit` 而不是 `step_after_actions`（2026-09-13）**
`step_after_actions=60` 与**动作流强耦合**：任何改变动作数的代码改动都会把阶跃
触发点挪到另一格，于是同一个"回归"在两次改动之间测的根本不是同一个场景。
实测证据：HEAD（e413cf7）在该参数下 244 张全过；本轮改动后动作数变了、触发点
落到"机器人离格心 78~96cm"处，而面板 3 在那个位置**只以一个 1100×380、hull 占
画幅 3% 的擦边条出现**（俯仰档 1040 下完全不可见）——旧代码就是靠在离格心
89.5cm 处误判"到达 ✓"、把位姿锚错，进而让面板 4 的搜索提示全错；这正是真机
"到 2 之后找 3 异常"的仿真同源体。按格触发让"第几格踩上形变"成为场景定义的
一部分，与被测代码的动作数解耦。**旧耦合场景不删除**，作为只诊断不计分的用例
保留在文件末尾（test_legacy_coupled_step_diagnostic），理由写在那里。

**2026-09-25 复核：那个"不可达"结论已经过期**（被上一次的措辞误导过，记在这里）
· 方向先纠正：`step_tilt_deg` 是**加在安装下俯偏移上**的（有效俯角 = 脉宽角 +
  安装偏移），所以 **+15° 是"更低头"，不是上仰**。更低头 ⇒ 可见地面带的**远边
  往里收**，远处面板反而更容易掉出画幅。
· 直接量（把机器人放在面板 3 正南 78cm、朝向面板，渲染一帧数面板像素）：
  | 倾斜 | 导航档 1200 | 低头档 1040 |
  |---|---|---|
  | +0°  | 3.54% | 3.63% |
  | +5°  | 3.55% | 2.19% |
  | +8°  | 3.57% | 1.12% |
  | +15° | 3.65% | **0.00%（完全看不见）** |
  也就是说 15° 时**到达判据用的低头档**在 78cm 处已经看不到面板；调小到 8° 恢复
  到 1.12%，0° 时 3.63%。
· 而且"不可达"这个结论本身也不再成立：它当年是按**当时的动作数**测的，而该场景
  的动作数是会漂的；2026-09-25 用同一套参数复跑，**15° 也能 7/7**（拍照 267、
  落点 3~14cm）。所以现在两件事都做：① 阶跃俯仰按用户要求调小到 **8°**，
  ② 把"几何不可达"的说法换成上面这张实测表——结论过期了就要写清楚。

运行：python tests/test_nine_grid_deform.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # 仓库根

import numpy as np

from sim.nine_grid_sim import run_simulation
from levels.nine_grid_shared import GRID_CELL_CM

# 拍照数护栏（防"定位风暴"回归：基线 206 / 随机游走 219 / 阶跃 246 张；
# 取 260 留 ~6% 余量。注意它是**回归判据**，不是真机时间预算——后者见下）
CAPTURE_GUARD = 260
# 真机整体可行性（仿真测不出时间，只能在这里换算显式化）：
#   2026-09-13 现场实测 capture_frame() = **0.70s/张**（fswebcam 0.62s/张）。
#   全局看门狗 780s，扣掉布局扫 60~90s 与自动曝光标定 ≤5 张，剩 ~690s
#   ⇒ 约 985 张的理论上限；取 700 张（≈8.2min）作为告警线，留足机械时间裕量。
# 历史教训：这条原先按"真机 2~4s/张"取 200 张，于是**每场都误报**"上真机前应先
# 降帧"——那个 2~4s 的估计偏高约 4 倍（疑似把网络 RTT 当成了拍照耗时）。
# 校验方式：跑真机一局后看整局耗时，若 7 格总耗时 >> 3min 才需要重估这个值。
REAL_CAPTURE_BUDGET = 700
REAL_CAPTURE_COST_S = 0.70     # 真机实测单张耗时（秒），用于打印时间换算
# 落点真值容差：微动开关有效区 ±5.5cm，取半格（16.7cm）作为"至少压在正确格上"
ARRIVE_TRUTH_TOL_CM = GRID_CELL_CM / 2.0


def _landing_errors(run):
    """逐格**真值**落点误差（cm）：关卡在每格返回前留档的"离目标格格心距离"

    这是本测试的护城河。`results` 里的"确认到达"只是关卡的**视觉自我判断**，
    真机上还有"微动开关到底响没响"这一层不可观测性（Pi 读不到 ESP32 侧）。
    所以必须用仿真真值再卡一道——2026-09-13 的假到达正是"results 说 ✓、真值
    离格心 89.5cm"。

    数据由 `NineGridLevel._record_landing` 在**每格返回前**写入（事后再取位置
    已经不是那一格的落点了）；仿真里来源应为"真值"（state.pos）。
    """
    out = {}
    for digit, rec in run.stats.get("panel_landing", {}).items():
        out[digit] = float("nan") if rec is None else float(rec[0])
    return out


def _check(tag, run):
    stats = run.stats
    assert stats["layout_ok"], \
        f"[{tag}] 布局解算错误: {stats['digit_cell']} != 真值 {stats['truth']}"
    failed = [k for k, ok in stats["results"] if not ok]
    assert stats["ok_all"] and not failed, \
        f"[{tag}] 未确认到达的面板: {failed}"
    assert not stats["banned_used"], \
        f"[{tag}] 使用了禁用动作: {stats['banned_used']}"
    assert stats["captures"] < CAPTURE_GUARD, \
        f"[{tag}] 拍照 {stats['captures']} 张超护栏 {CAPTURE_GUARD}（定位风暴？）"
    # 真值落点护栏：假到达的直接拦网
    errs = _landing_errors(run)
    bad = {d: e for d, e in errs.items() if not (e <= ARRIVE_TRUTH_TOL_CM)}
    assert not bad, (
        f"[{tag}] 落点真值超半格（{ARRIVE_TRUTH_TOL_CM:.1f}cm）: "
        + ", ".join(f"面板{d} 离格心{e:.1f}cm" for d, e in sorted(bad.items()))
        + " —— 疑似'假到达'（视觉判 ✓ 但真值不在格上）")
    # 安全护栏不得被触发：形变场景本就该靠视觉伺服扛过去
    assert not stats.get("cell_trips"), \
        f"[{tag}] 出现被安全项终止的格: {stats['cell_trips']}"
    print(f"  [{tag}] 布局 ✓ 7/7 ✓ 落点真值 max {max(errs.values()):.1f}cm"
          f"（半格 {ARRIVE_TRUTH_TOL_CM:.1f}cm）；拍照 {stats['captures']} 张 / "
          f"动作 {stats['actions']} 次；形变终值 俯仰 "
          f"{stats['deform_tilt_deg']:+.1f}° / "
          f"高度 {stats['deform_height_cm']:+.1f}cm")
    # 真机时间成本换算（仿真帧是瞬时的，测不出时间，所以在这里显式打印）：
    # 正常区间就打印一行供现场比对；超告警线才喊"需要降帧"，且**不判失败**。
    est_min = stats["captures"] * REAL_CAPTURE_COST_S / 60.0
    if stats["captures"] > REAL_CAPTURE_BUDGET:
        print(f"      ⚠ 真机可行性：{stats['captures']} 张 × "
              f"{REAL_CAPTURE_COST_S}s ≈ {est_min:.1f}min 超过告警线"
              f"（{REAL_CAPTURE_BUDGET} 张 ≈ 8.2min）——上真机前应考虑降帧")
    else:
        print(f"      真机时间成本：{stats['captures']} 张 × "
              f"{REAL_CAPTURE_COST_S}s ≈ {est_min:.1f}min"
              f"（看门狗 13min，余量充足）")


def test_baseline_no_deform():
    """基线：无形变（防止改造回归）"""
    run = run_simulation(seed=3, quiet=True, three_stage=True)
    _check("基线", run)


def test_random_walk_deform():
    """随机游走形变：σ=3°/动作（限幅 ±15°）+ σ=0.5cm/动作（限幅 ±2cm）"""
    deform = {"sigma_tilt_deg": 3.0, "sigma_h_cm": 0.5,
              "max_tilt_deg": 15.0, "max_h_cm": 2.0}
    run = run_simulation(seed=3, quiet=True, deform=deform, three_stage=True)
    _check("随机游走±15°", run)


def test_step_deform_at_digit():
    """一次性阶跃：第 3 格开始时俯仰 +15°（与动作数解耦）"""
    deform = {"sigma_tilt_deg": 0.0, "sigma_h_cm": 0.0,
              "step_at_digit": 3, "step_tilt_deg": 15.0,
              "step_h_cm": -2.0, "max_tilt_deg": 15.0, "max_h_cm": 2.0}
    run = run_simulation(seed=3, quiet=True, deform=deform, three_stage=True)
    _check("阶跃+15°(第3格)", run)


def test_step_deform_multi_digit():
    """阶跃落在第 5 格：确认"按格触发"在其它格次同样成立"""
    deform = {"sigma_tilt_deg": 0.0, "sigma_h_cm": 0.0,
              "step_at_digit": 5, "step_tilt_deg": 15.0,
              "step_h_cm": -2.0, "max_tilt_deg": 15.0, "max_h_cm": 2.0}
    run = run_simulation(seed=3, quiet=True, deform=deform, three_stage=True)
    _check("阶跃+15°(第5格)", run)


def test_legacy_coupled_step_diagnostic():
    """【只诊断、不计分】旧耦合场景 `step_after_actions=60`：记录现状供对照

    **为什么不修也不删**（交接文档 §6-2 要求"决定不保就写明理由"）：
    该场景的阶跃触发点由**动作数**决定，而动作数会随任何导航逻辑改动而变，
    所以它测的不是一个固定场景（触发点在哪、离面板多远，每次改动都可能不同）。

    **2026-09-25 复核（推翻了两条旧结论）**：
      · 阶跃俯仰按用户要求由 **15° 调小到 8°**——理由见文件头那张实测表：
        低头档 1040 在 78cm 处对面板的可见面积，15° 时是 **0.00%**（完全看不见），
        8° 时 1.12%，0° 时 3.63%。更低头 ⇒ 可见地面带远边往里收 ⇒ 远处面板掉出画面。
      · 旧注释说本场景"几何上不可达、任何合法到达判据都到不了它"——**已过期**：
        它当年是按**当时的动作数**测的，而该场景的动作数会漂。2026-09-25 用同一套
        参数复跑，15° 也已经 7/7（拍照 267、落点 3~14cm）；调小到 8° 后拍照 242、
        落点 3~7cm。所以"不可达"这个说法不再成立，只保留它作为**历史教训**：
        凡是要下"某某不可达"的结论，必须写清是哪一版代码、哪个动作数下量的。

    仍然**不计分**（原因只有一条：场景不固定）——但加了一条"别磨到保险丝"的
    断言：这个场景现在是能跑通的，跑到保险丝说明真的退化了。
    """
    deform = {"sigma_tilt_deg": 0.0, "sigma_h_cm": 0.0,
              "step_after_actions": 60, "step_tilt_deg": 8.0,
              "step_h_cm": -2.0, "max_tilt_deg": 15.0, "max_h_cm": 2.0}
    run = run_simulation(seed=3, quiet=True, deform=deform, three_stage=True)
    stats = run.stats
    errs = _landing_errors(run)
    n_ok = sum(1 for _, ok in stats["results"] if ok)
    print(f"  [旧耦合场景·诊断] {n_ok}/7 确认（**不计入判定**，见 docstring）；"
          f"拍照 {stats['captures']}；落点真值 "
          + ", ".join(f"{d}:{e:.0f}cm" for d, e in sorted(errs.items())))
    # 安全不变量：不许"**跨格级**假到达"（判 ✓ 但真值超一整格 = 身份误判）。
    # 阈值取一整格而不是半格：这个场景本就是形变场景，"尽力而为"的落点允许
    # 略超半格；真正的身份误判是"跑到别的格/场外还判 ✓"（旧代码的 89.5cm）。
    fake = [d for d, ok in stats["results"]
            if ok and errs.get(d, 0.0) > GRID_CELL_CM]
    assert not fake, f"旧耦合场景出现跨格级假到达（判 ✓ 但真值超一整格）: {fake}"
    assert not stats["banned_used"], "使用了禁用动作"
    assert stats["captures"] < 500, "拍照数异常（定位风暴）"
    assert not stats["fuse_tripped"], \
        f"这个场景现在能跑通，不该磨到保险丝：{stats['fuse_tripped']}"


def test_unified_safety_only():
    """【只诊断、不计分】统一决策办法（默认办法）在"磨不下来"场景下的安全门

    ⚠️ 这是**另一个**场景（第 3 格开始时阶跃低头，`step_at_digit=3`），别和文件
    末尾那个旧耦合场景（`step_after_actions=60`）混起来——后者 2026-09-25 复核
    时是能跑通的（见那边的 docstring）。

    本办法实测：这个场景下它**在 3000 张的保险丝内一次都没确认面板 3**
    （跑到 3001 张被保险丝停掉）。原因与"低头档可见地面带"直接相关：形变把
    相机压得更低之后，低头档只看得见很近的地面，而面板 3 在触发时刻还远在
    画幅之外（文件头那张表：78cm 处低头档对面板的可见面积在 +15° 时是 0.00%）。

    2026-09-25 改了策略——顺序计分下**绝不放弃一格**（跳格 = 后面全丢），
    所以这里观察到的行为是："它会一直磨面板 3，直到仿真侧的保险丝停掉实验。"

    因此本测试钉的是**性质**而不是成绩：
      · 布局必须解对、不许用禁用动作、不许出现"跨格级"假到达；
      · 拍照数受仿真保险丝约束（这里主动调小到 500，好让测试跑得快）；
      · **已经确认的格必须是 1..n 的前缀**——这一条正是"绝不跳格"的直接证据
        （旧策略会出现"3 未确认、4/5 已确认"那种断链结果）。
    """
    deform = {"sigma_tilt_deg": 0.0, "sigma_h_cm": 0.0,
              "step_at_digit": 3, "step_tilt_deg": 15.0,
              "step_h_cm": -2.0, "max_tilt_deg": 15.0, "max_h_cm": 2.0}
    run = run_simulation(seed=3, quiet=True, deform=deform,
                         fuse_captures=500)
    stats = run.stats
    errs = _landing_errors(run)
    n_ok = sum(1 for _, ok in stats["results"] if ok)
    print(f"  [统一决策·诊断] {n_ok}/7 确认（**不计入判定**）；"
          f"拍照 {stats['captures']}；保险丝 {stats['fuse_tripped']}；落点真值 "
          + ", ".join(f"{d}:{e:.0f}cm" for d, e in sorted(errs.items())))
    assert stats["layout_ok"], f"统一决策布局解算错误: {stats['digit_cell']}"
    assert not stats["banned_used"], "使用了禁用动作"
    fake = [d for d, ok in stats["results"]
            if ok and errs.get(d, 0.0) > GRID_CELL_CM]
    assert not fake, f"统一决策出现跨格级假到达: {fake}"
    assert stats["captures"] <= 505, \
        f"拍照数没被保险丝拦住：{stats['captures']}"
    # ★ 绝不跳格：确认的格必须是 1..n 的连续前缀
    seq = [d for d, ok in stats["results"] if ok]
    assert seq == list(range(1, len(seq) + 1)), \
        f"顺序断了（确认了 {seq}）——顺序计分下这是净亏，策略不允许"
    assert stats["fuse_tripped"], \
        "这个场景本该磨不下来、由保险丝收尾；若它跑完了，说明场景变了，请复核本测试"


if __name__ == "__main__":
    test_baseline_no_deform()
    test_random_walk_deform()
    test_step_deform_at_digit()
    test_step_deform_multi_digit()
    test_legacy_coupled_step_diagnostic()
    test_unified_safety_only()
    print("地板形变鲁棒性测试通过 ✓")
