#!/usr/bin/env bash
# 足球/球门检测训练 + 评估（容器云 GPU）
# 方案依据：《docs/视觉能力/足球与球门识别训练部署方案.md》§4
#
# 用法（数据集与 data.yaml 已就位后）：
#   DATA=/workspace/datasets/football_goal bash train_baseline.sh
# 可用环境变量覆盖：MODEL(yolo11n.pt/yolov8n.pt 同管线可互换)、
#   EPOCHS(200)、IMGSZ(640，RPi5 部署固定 640；调优对照可临时提高)、
#   BATCH(32)、NAME(run 名，默认 baseline)
set -euo pipefail

DATA="${DATA:?请设置 DATA=数据集根目录（含 data.yaml）}"
MODEL="${MODEL:-yolo11n.pt}"
EPOCHS="${EPOCHS:-200}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-32}"
NAME="${NAME:-baseline}"
RUNS="${RUNS:-/workspace/runs/football_goal}"

pip install -q ultralytics
nvidia-smi || echo "[WARN] 未检测到 GPU，训练将极慢"

yolo detect train \
  data="$DATA/data.yaml" \
  model="$MODEL" \
  epochs="$EPOCHS" \
  imgsz="$IMGSZ" \
  batch="$BATCH" \
  patience=40 \
  project="$RUNS" \
  name="$NAME"

yolo detect val \
  data="$DATA/data.yaml" \
  model="$RUNS/$NAME/weights/best.pt"

echo "完成。best 权重: $RUNS/$NAME/weights/best.pt"
echo "下一步: bash export_onnx.sh $RUNS/$NAME/weights/best.pt"
