# -*- coding: utf-8 -*-
"""main.py 新接入的两条"任务路线"回归测试（2026-09-28）

`levels/line_seeker_tracking.py`（循迹）与 `levels/apriltag_sorting_task.py`
（AprilTag 分拣）都不是比赛关卡本体，是两个独立写法的任务程序，共同点是
**顶层就 import hiwonder**：PC 上 import 必失败 ⇒ main.py 只能"有就注册"。

于是本文件要证明两件事（PC 上 hiwonder 永远不在，所以必须靠桩件把
"真机上会怎样"造出来）：

  ① **PC 上不拖累别人**：`import main` 照样成功，两条路线不注册，
     但跳过原因被记进 LEVELS 之外的诊断变量里（不是静默消失）；
  ② **真机上接了就跑得通**：把两个模块塞成桩进 `sys.modules`，重载 main
     后两条路线出现在 `LEVELS` 里，且入口闭包调到的是**各自真正的入口**
     （循迹 `run_line_tracking`、分拣 `main`），返回值也按约定归一成 bool。

用的打法是重载（`importlib.reload`），因为注册发生在 `import main` 的那一刻。
每个用例结束都把 `sys.modules` 还原，避免污染别的测试文件。
"""
import importlib
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main as M  # noqa: E402

LINE_SEEKER = "levels.line_seeker_tracking"
APRILTAG = "levels.apriltag_sorting_task"


class _StubLineSeeker(types.ModuleType):
    """循迹的桩：只记调用次数、不碰硬件"""

    def __init__(self):
        super().__init__(LINE_SEEKER)
        self.calls = 0
        self.result = True

    def run_line_tracking(self):
        self.calls += 1
        return self.result


class _StubApriltag(types.ModuleType):
    """分拣的桩：main() 就是它的真正入口（= parse_args + SortingTask.run）"""

    def __init__(self):
        super().__init__(APRILTAG)
        self.calls = 0

    def main(self):
        self.calls += 1


@pytest.fixture()
def fake_modules(monkeypatch):
    """把两条路线伪装成"真机上能 import"，重载 main 让注册跑一遍。

    重载后 `main.LEVELS` 是新字典对象，所以断言一律读重载后的那个模块对象
    （`importlib.reload(M)` 返回它），不要提前把旧字典存进变量。

    收尾**必须把桩先撤掉再重载**：只重载不撤桩的话，`from levels import X` 会命中
    `sys.modules` 里的桩而"成功"，于是 main 停在"注册了但本机没有该模块"这个
    现实中不存在的状态上 —— 下一个测试文件（tests/test_nine_grid_original.py
    会核对"跳过原因是否与模块可用性一致"）就会看到自相矛盾的 main。
    """
    seeker, sorting = _StubLineSeeker(), _StubApriltag()
    monkeypatch.setitem(sys.modules, LINE_SEEKER, seeker)
    monkeypatch.setitem(sys.modules, APRILTAG, sorting)

    yield seeker, sorting

    for name, stub in ((LINE_SEEKER, seeker), (APRILTAG, sorting)):
        if sys.modules.get(name) is stub:
            del sys.modules[name]
    importlib.reload(M)              # 撤桩后再重载 = 回到真实的（PC）状态


# =====================================================================
# ① PC 上（真机模块缺席）：不注册、不崩、留下原因
# =====================================================================

def test_pc_import_main_still_works():
    """hiwonder 不在时 `import main` 必须成功——注册逻辑不能把主进程带崩"""
    assert isinstance(M.LEVELS, dict)
    # 原来那六条一个不少
    for name in ("goodluck", "nine_grid", "nine_grid_three_stage",
                 "nine_grid_original", "stairs_hurdle", "press_button"):
        assert name in M.LEVELS


def test_pc_skip_reason_recorded_not_silent():
    """缺模块时要在诊断变量里留下原因，而不是静默少一条路线"""
    try:
        import hiwonder  # noqa: F401
        pytest.skip("本机居然有 hiwonder，这条用例只对 PC 端有意义")
    except ImportError:
        pass

    for name, reason in (("line_seeker_tracking", M.LINE_SEEKER_SKIP_REASON),
                         ("apriltag_sorting_task", M.APRILTAG_SORTING_SKIP_REASON)):
        assert reason is not None, f"{name} 缺模块时没记录跳过原因"
        assert "hiwonder" in reason
        assert name not in M.LEVELS


# =====================================================================
# ② 真机上（模块可 import）：注册进来，且闭包调到真正的入口
# =====================================================================

def test_both_routes_registered_when_importable(fake_modules):
    seeker, sorting = fake_modules
    reloaded = importlib.reload(M)

    assert "line_seeker_tracking" in reloaded.LEVELS
    assert "apriltag_sorting_task" in reloaded.LEVELS
    assert reloaded.LINE_SEEKER_SKIP_REASON is None
    assert reloaded.APRILTAG_SORTING_SKIP_REASON is None
    # 原来的六条仍在（注册是纯新增）
    for name in ("goodluck", "nine_grid", "nine_grid_three_stage",
                 "nine_grid_original", "stairs_hurdle", "press_button"):
        assert name in reloaded.LEVELS


def test_entry_closures_call_real_entrypoints(fake_modules):
    """入口名不是项目约定的 run_level，包一层就是要保证调对函数"""
    seeker, sorting = fake_modules
    reloaded = importlib.reload(M)

    entry = reloaded.LEVELS["line_seeker_tracking"]
    assert entry["tag_poses"] == {}
    assert entry["run_level"](object()) is True
    assert seeker.calls == 1
    assert sorting.calls == 0, "循迹不该顺带跑到分拣"

    entry = reloaded.LEVELS["apriltag_sorting_task"]
    assert entry["tag_poses"] == {}
    assert entry["run_level"](object()) is True
    assert sorting.calls == 1
    assert seeker.calls == 1, "分拣不该顺带跑到循迹"


def test_line_seeker_return_typed_to_bool(fake_modules):
    """项目铁律：入口返回值类型要和调用方一致（main 拿它当 success 用）"""
    seeker, _sorting = fake_modules
    seeker.result = 1                       # 模块里返回的是整型 1，不是 True
    reloaded = importlib.reload(M)

    got = reloaded.LEVELS["line_seeker_tracking"]["run_level"](object())
    assert got is True and isinstance(got, bool)


def test_module_key_matches_registered_module(fake_modules):
    """`module` 键要给轨迹图工具用（getattr 找 WALLS/ROUTE/tag_poses）"""
    seeker, sorting = fake_modules
    reloaded = importlib.reload(M)

    assert reloaded.LEVELS["line_seeker_tracking"]["module"] is seeker
    assert reloaded.LEVELS["apriltag_sorting_task"]["module"] is sorting


# =====================================================================
# ③ main() 收尾那步对新入口不炸（save_trajectory_png 的 getattr 兜底）
# =====================================================================

def test_trace_png_helper_tolerates_task_modules(fake_modules):
    """轨迹图工具会拿 `module` 找 WALLS/ROUTE/tag_poses。

    这两条路线的模块没有 WALLS/ROUTE（和 nine_grid 一样），2026-09-13 真机就
    因为少兜底丢过轨迹图 ⇒ 这里用桩模块（同样没有那两个属性）直接调一次收尾函数。
    不跑整个 main()：pytest 的 stdout 捕获会让 main() 里的 TeeWriter 把
    sys.stdout 换掉，那是测试环境的坑，不是接入的问题。
    产物写进 archive/result（项目统一产物目录、已在 .gitignore 里）：
    本机沙箱不允许往"新造的目录"写文件，所以不能用临时目录。
    """
    from core.paths import RESULT_DIR
    from core.trace import TraceRecorder, save_trajectory_png

    _seeker, sorting = fake_modules
    recorder = TraceRecorder(output_dir=RESULT_DIR)
    save_trajectory_png(recorder, sorting)          # 桩模块无 WALLS/ROUTE，不该抛
    assert os.path.isfile(recorder.png_path)


# =====================================================================
# ④ 既有注册没被动过（防止改 main.py 时手抖）
# =====================================================================

def test_existing_entries_unchanged():
    """六条既有路线的入口函数与 tag_poses 必须还是原来那些对象"""
    import levels.goodluck as GL
    import levels.nine_grid as NG
    import levels.nine_grid_original as NGO
    import levels.nine_grid_three_stage as NG3
    import levels.stairs_hurdle as SH

    assert M.LEVELS["goodluck"]["run_level"] is GL.run_level
    assert M.LEVELS["nine_grid"]["run_level"] is NG.run_level
    assert M.LEVELS["nine_grid_three_stage"]["run_level"] is NG3.run_level
    assert M.LEVELS["nine_grid_original"]["run_level"] is NGO.run_level
    assert M.LEVELS["stairs_hurdle"]["run_level"] is SH.run_level
    # tag_poses 走各自的赛道数据，没有被这里统一成空
    assert M.LEVELS["goodluck"]["tag_poses"] is GL.tag_poses
