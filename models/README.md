# 模型文件（models/）

存放足球/球门检测的部署用模型与版本记录。**新模型入库必须在下表登记**；
机器人端同步与 release 打包只认本表中标「当前生效」的文件。

## 版本登记

| 文件 | 基座 | 输入 | 数据版本 | val mAP50 / mAP50-95 | 导出命令 | 状态 |
|------|------|------|----------|----------------------|----------|------|
| football_goal_yolo11n_640.onnx | yolo11n | 640 | （待训） | （待训） | `yolo export model=best.pt format=onnx imgsz=640 simplify=True` | 待训练 |

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
