# 提速改造计划（目标：190s → 约 90-105s 到停靠点4）

## Trace 诊断结论（依据）
- 定位拍照 ~2s/张是主要成本；约 30 张废照片来自：多标签帧 8.2~10.7px 被 8px 门控拒（"差一点"类，含地面标签 163/160 混合帧 14~26px）、单标签 161 帧 2.2~3.3px 卡 2px 门、2 次全档失败级联（各 ~15s）。
- 停靠 3s×4 + 停靠朝向对准 ≈ 24s；转弯链中间被定位打断 7-8 次 ≈ 15s。
- 顺带发现：外参"未找到"告警是假告警（load_extrinsics 剥掉了 source 键，判定逻辑写错，实际已加载）。

## 改动 1：取消所有停靠要求（levels/goodluck.py）
- `STOP_TIME = 3` → `0.0`（注释注明 2026-08-30 取消停靠，恢复改回即可）；ROUTE 里 5 处引用自动归零，navigate_to_target 的 `stop_time > 0` 分支自然跳过 stand+sleep。
- ROUTE 停靠点（#1/3/5/7/9）的 orientation 字段 `[1,0]`/`[0,1]`/`[1,0]`/`[0,-1]`/`[1,0]` → `None`（到达即走，近距用冻结方向，不再触发对准转身）。
- trace.py 按 stop>0 标"停N"的逻辑向后兼容（不再标注，无需改代码）。

## 改动 2：提前结束开关（levels/goodluck.py）
- 模块级 `END_AT_LAST_STOP = "--end-at-last-stop" in sys.argv`（沿用 goodluck_sim 的 sys.argv 旗标先例，main.py 无需改）+ `END_AFTER_POS = [74.0, 30.0]`。
- run_level 路点循环：该路点导航成功且旗标开启 → 打印"开关生效：到达最后停靠点，提前结束" → return True。效果：跳过 [82,20] 与出口 [100,20]。
- 真机用法：`python main.py goodluck --end-at-last-stop`；PC 仿真 `python goodluck_sim.py --corner-stub --end-at-last-stop`。

## 改动 3：帧失败救援（robot_core.py + camera_config.py）——最大收益项
- camera_config：`PNP_REPROJ_ERR_MAX_MULTI_PX` 8.0 → **11.0**（trace 实测：正常帧 ≤5px、"差一点"帧 8.2~10.7px、错误分支 ≥14px，11px 分界）。
- `_locate_pose_once` 重构为按标签收集 `{tid: (obj4, img4)}`；全集门控失败且 n≥3 时做**逐标签剔除**重解：子集须 ≥2 标签且**跨 ≥2 个平面**（新增模块级 `tag_plane_axis()` 用 tag_poses 判定；非共面子集才无平面歧义），取过紧门控且 reproj 最小者；仍无 → 失败入联合缓存（现状不变）。
- 预期救掉 157+160+161、155+156+163 等 14~26px 混合帧（剔除地面标签后子集 ~4px 通过）。

## 改动 4：连续转弯免中间定位（levels/goodluck.py）
- `decide_rotation_action` 由恒 times=1 改为 `times = clamp(round(需要角/该转向步长), 1, 3)`（左转 22°/右转 25.7° 分别除）；过冲 ≤ 一 步转角，处于 15° 朝向阈值容忍内，下一轮定位自纠。
- navigate_to_target 旋转分支直接 `run_action(action, times)`（动作组 times 参数现成）。

## 改动 5：批量直行 4→6 步（levels/goodluck.py）
- `BATCH_FORWARD_MAX_STEPS` 4 → 6（30cm/批）；触发条件不变（离墙≥18cm、距目标>12cm、朝向差≤3°），走廊预检 check_segment_clear 覆盖更长段；注释更新（6 步 30cm × sin3° = 1.6cm < 3cm 阈值）。

## 改动 6：外参假告警修复（robot_core.py）
- `__init__` 记录 `self._extrinsics_from_file = ext is not None`；`_locate_joint` 仅在 False 时告警。

## 验证（PC）
1. 角点桩全程 `--corner-stub --end-at-last-stop`：在 [74,30] 结束、无 stand/sleep、全程完成；定位次数/步数较上版下降（预期定位循环 -15~25%）。
2. 消歧回归（真机外参+镜像诱饵）：门控 11px 后镜像分支（15~50px）仍被拒，PASS 不回退。
3. survey selftest 不受影响，复跑确认。
4. 交付真机同步清单：goodluck.py、robot_core.py、camera_config.py 共 3 个文件。

## 风险与边界
- 门控 11px 裕度：错误分支实测 ≥14px，间隔 3px，可接受；若真机出现 11-13px 假解误过，回落到 9px 并依赖剔除救援。
- 转弯批量化的累积转角误差（2-3 步 ≈ ±7-9°）在 15° 阈值内，超了下一轮自纠（与现状同机制）。
- 单标签 2~3px 假失败（tag161 特性）本轮不动（每集仅多 1-2 张照片），列为后续可选：位置连续性软通过判据。