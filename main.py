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

from robot_core import RobotState
from levels import goodluck as level_goodluck


# =====================================================================
# 真机调试输出开关
# =====================================================================
# True 时，直接运行 python main.py goodluck 也会：
#   - 把详细日志输出到 result/real_trace_*.txt
#   - 把定位轨迹保存为 result/real_trajectory_*.png
# False 时保持原有真机行为，不引入额外依赖。
TRACE_ENABLED = False


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
        from trace import TraceRecorder, TraceRobotState, TeeWriter, save_trajectory_png

        trace_recorder = TraceRecorder(output_dir="result")
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
