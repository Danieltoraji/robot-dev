# -*- coding: utf-8 -*-
"""
RoboTrack 主入口

用法：
    python main.py              # 默认运行 goodluck 关卡
    python main.py goodluck     # 显式指定 goodluck 关卡

后续新增关卡时：
    1. 在 levels/ 下新建关卡文件（如 levels/new_level.py）
    2. 实现 run_level(state) 入口函数和 tag_poses 赛道数据
    3. 在下方 LEVELS 字典中注册
    4. python main.py new_level 即可运行
"""

import sys

from core.paths import RESULT_DIR
from core.robot_core import RobotState
from levels import goodluck as level_goodluck
from levels import nine_grid as level_nine_grid
from levels import nine_grid_original as level_nine_grid_original
from levels import nine_grid_three_stage as level_nine_grid_three_stage
from levels import stairs_hurdle as level_stairs_hurdle

# 机器人智按按钮：代码在另一条分支（`.worktrees/press_button`，分支
# dxd_press_button），**本分支的 `levels/` 下没有 press_button.py**，
# 但机器人上的代码树里有它（那一分支同步过去的）。
# 所以这里写成"有就注册"：硬 import 会让本分支 `import main` 直接 ImportError，
# PC 上的自检、`tests/test_nine_grid_original.py::test_registered_in_main` 与
# trace 复现全部作废；而注册表里少了 press_button 又会把机器人上本来能跑的
# 那条路线从 `python main.py` 的入口列表里删掉。
try:
    from levels import press_button as level_press_button
except ImportError as _exc:            # 本分支没有该模块（机器人上有）
    level_press_button = None
    PRESS_BUTTON_SKIP_REASON = str(_exc)
else:
    PRESS_BUTTON_SKIP_REASON = None


# =====================================================================
# 真机调试输出开关
# =====================================================================
# True 时，直接运行 python main.py goodluck 也会：
#   - 把详细日志输出到 archive/result/real_trace_*.txt
#   - 把定位轨迹保存为 archive/result/real_trajectory_*.png
# False 时保持原有真机行为，不引入额外依赖。
# 2026-09-11 由 False 改为 True：数字宫格真机跑完整局后暴露出"终点不停/找 3
# 异常/踩不到微动开关"三类问题，定位这些都只能靠逐动作 trace（PC 仓库是唯一
# 真源，机器人侧随后同步）。trace 只是额外写两个文件，不影响控制流。
TRACE_ENABLED = True


# =====================================================================
# 关卡注册表
# =====================================================================
# 每个关卡提供：tag_poses（AprilTag 世界坐标）和 run_level（关卡入口函数）
LEVELS = {
    "goodluck": {
        "module": level_goodluck,
        "tag_poses": level_goodluck.tag_poses,
        "run_level": level_goodluck.run_level,
    },
    # 数字宫格：无 AprilTag，用地面单应定位（tag_poses 留空即可）
    # 两种决策办法各是一个模块，入口名分开，现场跑哪条一眼可辨。
    "nine_grid": {
        "module": level_nine_grid,
        "tag_poses": {},
        "run_level": level_nine_grid.run_level,
    },
    "nine_grid_three_stage": {
        "module": level_nine_grid_three_stage,
        "tag_poses": {},
        "run_level": level_nine_grid_three_stage.run_level,
    },
    # 参考原版通关代码（reference code/九宫格视觉导航 的原样接入，常量未重标）。
    # 与上面两条互不共用代码，只用于对照；详见 levels/nine_grid_original/ 这个包。
    "nine_grid_original": {
        "module": level_nine_grid_original,
        "tag_poses": {},
        "run_level": level_nine_grid_original.run_level,
    },
    # 上下楼梯与识别跨障：无 AprilTag，红色带 + 本地系地面单应
    "stairs_hurdle": {
        "module": level_stairs_hurdle,
        "tag_poses": {},
        "run_level": level_stairs_hurdle.run_level,
    },
}

# 机器人智按按钮（模块在另一分支，见文件头的 try/except）：模块在就注册，
# 不在就不注册——PC 本分支不注册，机器人上注册（保持那边的入口一样多）。
if level_press_button is not None:
    LEVELS["press_button"] = {
        "module": level_press_button,
        "tag_poses": {},
        "run_level": level_press_button.run_level,
    }


def main():
    # 解析关卡名
    if len(sys.argv) > 1:
        level_name = sys.argv[1]
    else:
        level_name = "goodluck"  # 默认关卡

    if level_name not in LEVELS:
        print(f"未知关卡: {level_name}")
        print(f"可用关卡: {', '.join(LEVELS.keys())}")
        sys.exit(1)

    print(f"===== 启动关卡: {level_name} =====")

    # 创建机器人状态实例，传入关卡的 tag_poses
    level = LEVELS[level_name]

    original_stdout = sys.stdout
    tee = None
    trace_recorder = None

    if TRACE_ENABLED:
        from core.trace import TraceRecorder, TraceRobotState, TeeWriter, save_trajectory_png

        trace_recorder = TraceRecorder(output_dir=RESULT_DIR)
        state = TraceRobotState(tag_poses=level["tag_poses"], recorder=trace_recorder)
        tee = TeeWriter(trace_recorder.log_path, original_stdout)
        sys.stdout = tee
        print(f"[trace] 真机日志将写入: {trace_recorder.log_path}")
        print(f"[trace] 轨迹图将保存为: {trace_recorder.png_path}")
    else:
        state = RobotState(tag_poses=level["tag_poses"])

    try:
        # 执行关卡
        success = level["run_level"](state)
    finally:
        if TRACE_ENABLED:
            try:
                if trace_recorder is not None:
                    print(f"[trace] 动作次数: {trace_recorder.action_count}, "
                          f"定位次数: {trace_recorder.locate_count}")
                    try:
                        save_trajectory_png(trace_recorder, level["module"])
                    except Exception as e:
                        print(f"[trace] 保存轨迹图失败: {e}")
            finally:
                if tee is not None:
                    sys.stdout = original_stdout
                    tee.close()
                    print(f"[trace] 日志已保存: {trace_recorder.log_path}")

    if success:
        print(f"===== 关卡 {level_name} 完成 =====")
    else:
        print(f"===== 关卡 {level_name} 失败 =====")


if __name__ == "__main__":
    main()
