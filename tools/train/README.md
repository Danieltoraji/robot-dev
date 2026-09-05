# 足球/球门检测训练脚手架（容器云）

方案依据：《docs/视觉能力/足球与球门识别训练部署方案.md》§4（训练）/ §6（部署）。
目标硬件：**树莓派 5**（ONNX 输入固定 640，无需 416/int8 分支）。

## 文件一览

| 文件 | 作用 |
|------|------|
| `data_template.yaml` | data.yaml 模板（复制到数据集根目录并改 `path`） |
| `split_dataset.py`   | 标注结果按场景划分 train/val/test（防连拍泄漏） |
| `train_baseline.sh`  | 容器云训练 + val 一条龙 |
| `export_onnx.sh`     | best.pt → ONNX(640) 导出 |

## 完整流程

1. **采集**：机器人上按《[足球球门数据采集规范](../../docs/视觉能力/足球球门数据采集规范.md)》
   跑 `tools/data_collect/capture_dataset.py`，照片拷回 PC。
2. **标注**：X-AnyLabeling / LabelImg，输出 YOLO txt（`class x_c y_c w h`，归一化）。
   负样本（无目标图）不生成 txt 即可，划分脚本会自动补空标签。
3. **划分**：
   ```bash
   python tools/train/split_dataset.py --src <标注目录> --dst datasets/football_goal
   ```
   把 `data_template.yaml` 复制为 `datasets/football_goal/data.yaml` 并改 `path`。
4. **上传容器云**：整个 `datasets/football_goal/` 目录。
5. **训练**：
   ```bash
   DATA=/workspace/datasets/football_goal bash train_baseline.sh
   ```
   第一版先用 100 epoch 快速基线（`EPOCHS=100 NAME=quick`）看收敛与 badcase 方向，
   补数据后再 200 epoch 正式训练。
6. **评估与 badcase 闭环**：重点看每类 precision / recall / mAP50 和混淆矩阵；
   逐张可视化：
   ```bash
   python tools/debug_vision.py yolo --image <实拍图> --model best.pt --conf 0.45 --out annotated.jpg
   ```
   按方案 §5.3 调优决策表补拍（重点：白门浅背景/过曝、远距小球、球在门内）。
   提效技巧：用上一版 best.pt 对新一批原图预标注（伪标签），人工只做修正。
7. **导出**：
   ```bash
   bash export_onnx.sh /workspace/runs/football_goal/<name>/weights/best.pt
   ```
8. **入库与部署验证**：onnx 复制到仓库 `models/`，在 `models/README.md` 登记版本与指标；
   机器人端：
   ```bash
   python tools/verify_onnx_vs_pt.py --pt best.pt --onnx models/<文件>.onnx --images <实拍目录>
   python tools/bench_yolo.py --model models/<文件>.onnx --images <实拍目录>
   ```

## 调优速查（详见方案 §5.3）

| 现象 | 处理 |
|------|------|
| 远距离球漏检 | 提高 imgsz 对照训练、补远距样本、降 conf |
| 白色球门浅背景漏检 | 补"白门+浅背景/过曝"难例、训练加亮度扰动 |
| 球门误检 | 补负样本（白网/栅栏/白框）、核对标注口径 |
| 两类混淆 | 检查球在门内/门前时的标注是否重叠漏标 |
