# 模型文件（models/）

存放足球/球门检测的部署用模型与版本记录。**新模型入库必须在下表登记**；
机器人端同步与 release 打包只认本表中标「当前生效」的文件。

## 版本登记

| 文件 | 基座 | 输入 | 数据版本 | val mAP50 / mAP50-95 | 导出命令 | 状态 |
|------|------|------|----------|----------------------|----------|------|
| football_goal_ball_v1_640.onnx | yolo11n | 640 | 2026-09-05 football 90/目标129 图（ball_ 12场景组，训练90图/113框） | 0.995 / 0.904（val=1250r）；**test 0.995 / 0.878**（test=1350c 远距小目标） | `yolo export model=best.pt format=onnx imgsz=640 simplify=True` | **当前生效** |
| ~~football_goal_yolo11n_640.onnx~~ | — | — | — | — | — | 未训（命名规范占位，此行保留格式参考） |
| nine_grid/digit_classifier_mask.pkl | HOG+SVM（sklearn SVC，28MB） | 64x64 黑色数字 mask | 上届清华队九宫格训练集（类别 1~7，外部数据未入库） | 无独立 val（仅作颜色主判的仲裁校验） | 非导出产物，直接同步 pkl | **当前生效**（数字宫格仲裁） |

### digit_classifier_mask.pkl 使用说明（2026-09-08 引入）

- 来源：`reference code/九宫格视觉导航/robot/models/`（上届已验收方案），拷贝入库以支持
  PC 唯一真源同步到机器人；
- 推理链固定：黑色数字 mask（vision/nine_grid_detector.extract_digit_mask）→ resize 64x64
  → skimage.hog(orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2),
  block_norm="L2-Hys") → SVC predict/predict_proba。**HOG 参数与阈值链改动即失配**；
- 依赖：scikit-learn + scikit-image + joblib（机器人 jupyter-env 需安装，缺失时
  DigitArbiter 自动降级为仅颜色主判）；训练端 sklearn 1.6.1，加载端 1.9.0 实测正常（有版本告警，已抑制）；
- 角色：只仲裁否决（conf>=0.6 且与颜色结论冲突 → 标记待近距复核），不作主判。

### ball_v1 训练记录（2026-09-06，容器云 RTX 3090）

- **数据**：`datasets/football_goal`（采集收官版 129 图，单类 football；train=1040c/1150c/1250c/1250l±neg 92 图，val=1250r±neg 17 图，test=1350c±neg 20 图）
- **训练**：yolo11n.pt 微调，150 epoch 上限 + patience=40 早停，实际 93 轮停止（best 在 53 轮），3 分钟
- **指标**：val P/R/mAP50/mAP50-95 = 0.997/1.000/0.995/0.904；test = 0.997/1.000/0.995/0.878
- **一致性校验**（`tools/verify_onnx_vs_pt.py`，test 集，conf=0.45）：onnx 28/28 检出与真值完全一致、零误报；ultralytics pt 同阈值下多 6 个误检（含 5 张双检+1 张负样本误报）——自研 letterbox 灰边反而更稳。**部署只认 onnx 后端（OnnxYoloBackend）**
- 训练产物：`confusion_matrix*.png`、`results.csv`（本目录）

> 登记字段说明：
> - **数据版本**：数据集快照标识（日期 + 每类张数，如 `2026-09-10 football200/goal210`）；
> - **val 指标**：容器云 `yolo detect val` 在 val/test 集的结果，每类 P/R 见训练 runs 目录；
> - **状态**：当前生效 / 弃用（弃用文件删除，表内保留历史行）。

## 入库与同步约定

- `.onnx`（yolo11n 约 12MB）直接入 git，保证「PC 唯一真源」；
- 同步到机器人时模型与 `vision/` 一起走整文件同步；
- **重新打包 release 时必须包含 `models/` 与 `vision/`**
  （当前 `release/goodluck` 快照不含 `vision/`，打包脚本需更新）。

## 机器人端复验一条命令

```bash
# 一致性（pt vs onnx）
python3 tools/verify_onnx_vs_pt.py --pt best.pt --onnx models/<文件>.onnx --images <实拍目录>
# 延迟（验收线：RPi5 平均 ≤1000ms）
python3 tools/bench_yolo.py --model models/<文件>.onnx --images <实拍目录>
```
