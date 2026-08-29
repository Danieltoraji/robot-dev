## 数据判读结论
1. 崩溃根因：机器人端 optimize_multi_view.py 为旧版（309 行 `best.cost`），同步即修。
2. 主矛盾：单标签帧 reproj 0.5px（相机链路可信），跨墙多标签帧 14~20px ⇒ 尺量 tag_poses 相对几何不自洽（~1cm 级）+ 墙体名义值与实物偏差。k=0.809 撞界、rms 19.7px 均为错误坐标上的症状。
3. 数据 bug：goodluck.py:35 tag161 第 4 角点与第 3 重复（应 [81.6,0,22.8]）。
4. 新布局一站 4~5 标签跨三平面 ⇒ 场地视觉自标定可行；坐标修复前 2px 门控大量误杀多标签帧。

## P0 立即修复
- 修 tag161 typo；同步 goodluck.py + optimize_multi_view.py 到机器人（PC 为唯一真源）；机器人重跑 optimize+diag 留基线（rms 大属预期）。

## P1 场地自标定 survey_field.py（新核心）
联合 BA：每站位 (x_B,y_B,φ) + 每标签面内 3dof（约束在各自物理平面）+ **北墙/中墙西面平面偏移待估**（弱先验 σ≈3cm 拉向名义值）；东墙 x=95、南墙 y=0、地面 z=0 硬锚定基准（左墙无标签保持名义）；相机外参 {e,z_c,pitch,k} 全局共享可 --fix（兼作外参标定）。多标签 PnP + IPPE 兜底初始化，Huber/坏帧剔除，地面标签降权，|θ|≤30° 过滤。
采集：5 站位（入口+四停靠点）× 现有 collect_multi_view.py，每标签尽量 ≥2 站位可见。
输出：粘贴用 tag_poses 块 + tag_poses_refined.json + multiview_extrinsics.json + 逐标签残差报告（揪贴歪标签、报告墙实际位置）。
验收：rms ≤1px；同批 npz 多标签帧 reproj 降至 ~1px；单/多标签帧互差 <2cm。

## P2 共享模块 multiview_pose.py
正向模型 + IPPE 双解枚举 + 固定外参 3 参数求解器 + 门控；synthetic/optimize/diag 改 import 消重；survey 构建于其上，一体交付。

## P3 检查点
坐标修复后先跑现行主流程全程 trace：达标则 P4 降为可选；未达标（尤其单地面标签站位）再上 P4。

## P4 运行时三档联解
locate_with_scan：回正快路径不变；②③帧角点缓存 ≥2 帧即固定外参联解，RMS/场地/朝向门控；失败剔最大残差帧重试，再落回现行逻辑。θ=k·(pulse−1500)·0.09；current_position 维持光心语义，levels 零改动。

## P5 仿真回归
goodluck_sim 同步新 tag_poses；新增角点合成桩（真值投影+噪声+镜像歧义诱饵，补上"仿真从不产生歧义"的盲区）；多 seed 成功率不回退。

## 关键风险
地面标签擦视角噪声大（降权+离群剔除）；平面偏移待估需 ≥4 站位且弱先验防弱相关；同步纪律：每次改动 PC→机器人整文件同步。