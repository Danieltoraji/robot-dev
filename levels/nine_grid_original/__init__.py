# -*- coding: utf-8 -*-
"""参考原版九宫格通关代码（包：`levels/nine_grid_original/`）

来自 `reference code/九宫格视觉导航/`（叶雨岑、陶鲁玥，2026-08）——**别人已经通关的
那一版**，原样搬运进来，作为与现行两条路线并列的第三条路（A/B 对照臂）：

| 路线 | 模块 | 想法 |
| :--- | :--- | :--- |
| 统一决策（默认） | `levels/nine_grid.py` | 三档分区（绿直行/蓝横移/橙旋转）+ 一个循环 |
| 三段式（上现场优先） | `levels/nine_grid_three_stage.py` | 搜索→对准→接近→到达四段 |
| **参考原版（本包）** | `levels/nine_grid_original/` | 四层状态机：搜索 → 方向跟踪 → 前进修正 → 最终接近 |

**入口**：`python main.py nine_grid_original`

**包内文件**（与参考版文件的对应关系）：

| 本包文件 | 内容 | 参考版来源 |
| :--- | :--- | :--- |
| [`level.py`](level.py) | 通关流程 `go_to` / `turn_to` / `search_target` / `arrived` + `run_level` | `robot/mainv0.2.ipynb` CELL 4/5 |
| [`action.py`](action.py) | 动作组原语 + 云台姿态（`with_stand` 逐字保留） | `robot/action.py` |
| [`vision.py`](vision.py) | 拍照 + 颜色识别 + `identify()` / yaw / proximity | `robot/capture.py` + `robot/identify.py` |
| [`classifier.py`](classifier.py) | 候选区域提取 + HOG/SVM 数字融合打分 | `robot/extract_digit_roi.py` + `robot/candidate_classifier.py` |

**隔离**：这个包不 import `nine_grid_shared`，也不 import `vision/nine_grid_detector`；
反过来现行两条路线也没有一行引用本包。`main.py` 只多一行 import 与一个 `LEVELS` 条目。

⚠️ **常量尚未按本机器人/本场地重标**（用户要求本轮不动标定常量）。待改动清单
（含与现行 `nine_grid` 常量的逐条对照）见
`docs/关卡算法/彩色数字九宫格-nine_grid/参考原版通关代码-nine_grid_original.md`。

⚠️ **本关没有仿真器**，只能上真机；细节与"与参考版的差异"见 `level.py` 文件头。
"""

from .level import go_to, run_level

# 本关无 AprilTag（与 main.LEVELS 里的 "tag_poses": {} 一致；
# main.py 的轨迹图工具会 getattr 关模块的 WALLS/ROUTE/tag_poses）
tag_poses = {}

__all__ = ["run_level", "go_to", "tag_poses"]
