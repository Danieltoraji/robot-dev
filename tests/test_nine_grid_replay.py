# -*- coding: utf-8 -*-
"""真实照片重放集成测试（tests/test_nine_grid_replay.py）

把现场照片（tests/fixtures/field_photos，若缺失则自动跳过）按扫描时序喂给
**真实的** `NineGridLevel.layout_scan()`，断言：
  1. 布局 = 照片读图真值（见 tools/replay_ninegrid.PHOTO_TRUTH）；
  2. 自标定常数落在合理范围（有效俯角 45~60°、高度 35~70cm）；
  3. 位姿自举成功（不再出现历史崩溃："相机未俯视地面（光轴无向下分量）"）。

历史（2026-09-11 接手勘察）：当时该重放给出的是真值的**转置**
（{1:8,2:7,3:4,4:2,5:1,6:0,7:3}）并随后崩溃。根因四处：
  - D4 枚举含镜像候选（真解必为真旋转）；
  - `_lattice_grid_fit` 多数票不足分支返回 2 元组（调用方解包 4 元组）；
  - `from_pose(head_pulse=δ)` 未把头部偏航并入朝向 → 宽扫档观测错位 8cm~1m；
  - `quad_center` 质心兜底漏加 bbox 原点 → 部分观测中心偏移数百~上千 px。

照片机位为板南侧（不在入口侧），因此本数据集的正确结果里"位置6 被占"是
**预期**现象；真机从入口进场时扫描结果必须满足"格6 恒空"。

运行：python tests/test_nine_grid_replay.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # 仓库根

import numpy as np

from levels.nine_grid import NineGridLevel, PITCH_NAV
from tools.replay_ninegrid import (
    PHOTO_TRUTH, ReplayRobot, build_schedule, find_photos,
)


def _scan():
    photos = find_photos()
    if not photos:
        return None, None, None
    robot = ReplayRobot(build_schedule(photos, "schedule"))
    level = NineGridLevel(robot)
    level.layout_scan()
    return level, robot, photos


def test_replay_layout_matches_photo_truth():
    """重放布局必须等于照片读图真值（含 7 数字全内点）"""
    level, robot, photos = _scan()
    if level is None:
        print("  现场照片夹具缺失，跳过（见 tests/fixtures/field_photos）")
        return
    assert level.digit_cell == PHOTO_TRUTH, \
        f"重放布局 {level.digit_cell} != 照片读图 {PHOTO_TRUTH}"
    assert sorted(level.digit_cell) == list(range(1, 8)), \
        f"布局未覆盖全部 7 个数字: {sorted(level.digit_cell)}"
    print(f"  重放布局 = 照片读图 {PHOTO_TRUTH} ✓"
          f"（{len(photos)} 张照片、实拍 {robot.captures} 帧）")


def test_replay_selfcal_and_pose():
    """自标定常数在合理范围；位姿自举成功且落在场地附近"""
    level, robot, _photos = _scan()
    if level is None:
        print("  现场照片夹具缺失，跳过")
        return
    eff = (1500 - PITCH_NAV) * 0.09 + level._pitch_offset_deg
    assert 45.0 <= eff <= 60.0, f"导航档有效俯角 {eff:.1f}° 超出 45~60°"
    assert 35.0 <= level._cam_height_cm <= 70.0, \
        f"自标定高度 {level._cam_height_cm:.0f}cm 超出 35~70cm"
    assert np.all(np.isfinite(level.pose)), "位姿自举未产生有效位姿"
    assert -30.0 <= level.pose[0] <= 130.0 and -60.0 <= level.pose[1] <= 110.0, \
        f"自举位姿 ({level.pose[0]:.1f},{level.pose[1]:.1f}) 远离场地"
    print(f"  自标定：偏移 {level._pitch_offset_deg:+.1f}°（有效俯角 "
          f"{eff:.1f}°）/ 高度 {level._cam_height_cm:.0f}cm；位姿 "
          f"({level.pose[0]:.1f},{level.pose[1]:.1f}) "
          f"航向 {np.degrees(level.pose[2]):.1f}° ✓")


if __name__ == "__main__":
    test_replay_layout_matches_photo_truth()
    test_replay_selfcal_and_pose()
    print("真实照片重放测试通过 ✓")
