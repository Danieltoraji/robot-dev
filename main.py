# -*- coding: utf-8 -*-
"""
RoboTrack 主入口

用法：
    python main.py              # 默认运行 goodluck 关卡
    python main.py goodluck     # 显式指定 goodluck 关卡

（没有 --help：关卡名就是第一个位置参数，传个没有的名字会列出当前可用关卡
并以退出码 1 结束。）

后续新增关卡时：
    1. 在 levels/ 下新建关卡文件（如 levels/new_level.py）
    2. 实现 run_level(state) 入口函数和 tag_poses 赛道数据
    3. 在下方 LEVELS 字典中注册
    4. python main.py new_level 即可运行

注册表里少数几条是"有就注册"（模块能 import 才多一条入口）：press_button、
line_seeker_tracking、apriltag_sorting_task。后两条顶层 import hiwonder，
**PC 上必然缺席、只有真机才注册**；缺席原因记在 *_SKIP_REASON 变量里，
不静默消失。详见下面各自的注释块。
"""

import sys

from core.paths import RESULT_DIR
from core.robot_core import RobotState
from levels import goodluck as level_goodluck
from levels import nine_grid as level_nine_grid
from levels import nine_grid_original as level_nine_grid_original
from levels import nine_grid_three_stage as level_nine_grid_three_stage
from levels import stairs_hurdle as level_stairs_hurdle

# 机器人智按按钮：**合并后 `levels/press_button.py` 已在本分支存在**（远端 main 带来），
# 但仍写成"有就注册"：这样换 SD 卡/少同步一个文件时 `import main` 不会直接崩，
# 注册表里也不会凭空少一条能跑的路线。
try:
    from levels import press_button as level_press_button
except ImportError as _exc:            # 缺该模块时降级（不注册这条路线）
    level_press_button = None
    PRESS_BUTTON_SKIP_REASON = str(_exc)
else:
    PRESS_BUTTON_SKIP_REASON = None

# 下面两条不是比赛关卡本体，是"独立写法的任务程序"，接进来的目的是让它们和其它
# 关卡一样有 main.py 统一入口（2026-09-28 用户要求）。共同点：
#   · **模块本体一行不改**（各自有活跃的现场手调值，整文件覆盖会冲掉别人的改动）；
#   · 顶层就 `import hiwonder.*`，所以 **PC 上 import 必失败**（真机才有 SDK）
#     ⇒ 只能"有就注册"，与 press_button 同一套降级写法，PC 上少一条是正常的；
#   · 都自己开关相机、自己下发动作组，**不读传进来的 state**（state 只用于轨迹记账）；
#   · 入口签名不同：循迹是 `run_line_tracking()`，抓取是 `main()`（自带 argparse，
#     现场要传参数就直接跑原文件，别从这里传）。
# 资源排他：这两条和别的关卡一样独占相机/舵机总线，一次只跑一条。

# 红色路径寻线 + 循迹：levels/line_seeker_tracking.py（入口 run_line_tracking）
try:
    from levels import line_seeker_tracking as level_line_seeker_tracking
except ImportError as _exc:
    level_line_seeker_tracking = None
    LINE_SEEKER_SKIP_REASON = str(_exc)
else:
    LINE_SEEKER_SKIP_REASON = None

# AprilTag 分拣（抓蓝色海绵 → 放到 Tag 38 板）：levels/apriltag_sorting_task.py
# 注意它在 import 期就会建 /home/pi/Robot_Competition/levels/ 目录并挂文件日志、
# import 期还会读同目录 calib_config.json；所以"能 import"本身就等于"这是真机"。
try:
    from levels import apriltag_sorting_task as level_apriltag_sorting_task
except ImportError as _exc:
    level_apriltag_sorting_task = None
    APRILTAG_SORTING_SKIP_REASON = str(_exc)
else:
    APRILTAG_SORTING_SKIP_REASON = None


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
    # 上下楼梯与识别跨障：头部单目测距（卷尺标定的相机几何）+ 黑箱动作组，
    # 不用 AprilTag，tag_poses 留空。
    # 方案：docs/关卡算法/上下楼梯与识别跨障-stairs_hurdle/完整方案-2026-09-25-流程重制.md
    # 仿真回归：python tests/test_stairs_hurdle_sim.py
    "stairs_hurdle": {
        "module": level_stairs_hurdle,
        "tag_poses": {},
        "run_level": level_stairs_hurdle.run_level,
    },
}

# 机器人智按按钮：模块在就注册（见文件头的 try/except）。
if level_press_button is not None:
    LEVELS["press_button"] = {
        "module": level_press_button,
        "tag_poses": {},
        "run_level": level_press_button.run_level,
    }

# 循迹 + AprilTag 分拣：同样"模块在就注册"。
# 为什么入口要包一层闭包，而不像上面那样直接写 `module.run_level`：
#   这两条路线的入口函数名不是项目约定的 `run_level`，包一层是为了在**不改模块本体**
#   的前提下把签名统一成 `run_level(state)`。`state` 按原样收下但不使用：它们自己开关
#   相机、自己下发动作组，不走 state.act / state.capture_frame。
def _run_line_seeker_tracking(_state):
    """循迹（levels/line_seeker_tracking.py）：stand → 寻线 → 沿线走。"""
    return bool(level_line_seeker_tracking.run_line_tracking())


def _run_apriltag_sorting(_state):
    """Apriltag 分拣（levels/apriltag_sorting_task.py）：抓蓝海绵放 Tag 38 板。

    走它自己的 `main()`（= parse_args + SortingTask.run），也就是命令行直接跑
    `python3 levels/apriltag_sorting_task.py` 的那条路；要传 `--target-tag` 之类
    参数请直接跑原文件。
    """
    level_apriltag_sorting_task.main()
    return True


if level_line_seeker_tracking is not None:
    LEVELS["line_seeker_tracking"] = {
        "module": level_line_seeker_tracking,
        "tag_poses": {},                       # 本路线不用 AprilTag
        "run_level": _run_line_seeker_tracking,
    }

if level_apriltag_sorting_task is not None:
    LEVELS["apriltag_sorting_task"] = {
        "module": level_apriltag_sorting_task,
        "tag_poses": {},                       # 任务自带 Tag 路线，不用 main 的赛道数据
        "run_level": _run_apriltag_sorting,
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
