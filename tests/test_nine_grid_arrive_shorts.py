#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""到达判决"短路"的常量与源码闸（tests/test_nine_grid_arrive_shorts.py）

**钉住什么**（2026-09-26 的决定）：到达判决**只吃画面量**，地图锚解出的位姿
**不许**参与"到没到"的判断。

为什么要用"读源码"这种闸（而不是只测行为）：这次短路的**唯一失败方式**就是
有人把否决权悄悄接回来（加一个 `if verdict == "no": continue` 太容易了，
而且它在仿真里多数时候不报错、只是偶尔把正确到达否掉）。所以这里同时钉三件事：

  ① 行为面：`_arrive_pixels_ok` 与锚无关（见 test_nine_grid_unified_zone.py §6）；
  ② 数据面：**废弃开关 `ARRIVE_ON_CELL_REQUIRED` 在判决路径上零引用** ——
     它只能出现在"常量定义"与"注释/文档字符串"里，一旦被某个 `if` 读走就红；
  ③ 源码面：判决路径里不许出现按核验结论分支的写法。

背景（为什么要短路）：实测这道核验会用**不可信**的位姿否决**正确**到达 ——
面板 5 连试三次才认；形变场景里位姿漂 105cm 而真值离格心只有 13cm。
仓库的上位法早就写明"不要用位姿去否决到达"（`tools/diag_arrive.py`、
`交接文档-2026-09-13-复核版.md`）。恢复否决 = 回滚本提交。
"""
import os
import re
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import levels.nine_grid as NG  # noqa: E402

_LEVEL_SRC = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "levels", "nine_grid.py")


def _read_level_src():
    with open(_LEVEL_SRC, "r", encoding="utf-8") as f:
        return f.read()


def test_probe_switch_exists_and_is_documented():
    """`ARRIVE_ON_CELL_PROBE` 必须在，且是**纯诊断**开关

    它只决定"到达那一帧要不要多跑一次锚测量"，不决定判决；因此它的值可以随便改，
    但**不能改名/删除**（日志与文档都按这个名字找）。
    """
    assert hasattr(NG, "ARRIVE_ON_CELL_PROBE"), \
        "诊断探针开关 ARRIVE_ON_CELL_PROBE 丢了（到达时的锚测量靠它）"
    assert isinstance(NG.ARRIVE_ON_CELL_PROBE, bool)


def test_on_cell_tolerance_value_is_pinned():
    """容差仍是"半格"16.7cm —— 探针的分歧判据，别在调参时被顺手改掉"""
    assert NG.ARRIVE_ON_CELL_TOL_CM == 16.7


def test_deprecated_require_switch_is_not_read_by_any_branch():
    """★ 废弃开关在代码里**零引用**（只许留在定义与注释/文档字符串中）

    做法：用 AST 把 `levels/nine_grid.py` 里所有**表达式位置**的名字读出来
    （函数体、默认值、f-string 都算），`ARRIVE_ON_CELL_REQUIRED` 一旦被某个
    分支读走就出现在这里。注释与 docstring 不算（它们是历史说明）。
    """
    import ast
    tree = ast.parse(_read_level_src())
    used = {n.id for n in ast.walk(tree)
            if isinstance(n, ast.Name) and n.id == "ARRIVE_ON_CELL_REQUIRED"
            and isinstance(n.ctx, ast.Load)}          # 只看"读"，定义本身不算
    assert not used, (
        "ARRIVE_ON_CELL_REQUIRED 已废弃（判决不再看锚），但代码里还在读它 —— "
        "到达判决里又出现了按锚结论分支的写法")


def test_no_branch_on_cell_verification_verdict():
    """判决路径里不许按核验结论分支（历史上的 `_ok_cell` 写法）

    允许保留的唯一形态：探针里 `if verdict == "no":` **只打印**。
    这里扫的是"拿核验结论去 continue/return"的回归写法。
    """
    src = _read_level_src()
    bad = [m for m in ("_ok_cell", "格的核验否决") if m in src]
    assert not bad, (
        f"到达判决里又出现了按格的核验分支：{bad} —— "
        "要恢复「用位姿否决到达」必须回滚 2026-09-26 的短路提交，不是加个 if")


def test_probe_never_changes_control_flow():
    """探针内部**任何**异常都必须被吞掉、静默返回 None（判决不许被它带崩）"""
    lv = NG.NineGridLevel.__new__(NG.NineGridLevel)

    def _boom(frame, digit):
        raise RuntimeError("核验炸了")

    lv._verify_target_cell = _boom
    out = NG.NineGridLevel._arrive_probe_note(lv, 1, None)
    assert out is None, "核验异常时探针必须静默返回 None，绝不许影响判决"
