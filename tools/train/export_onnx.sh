#!/usr/bin/env bash
# 导出 ONNX（树莓派 5 部署用，输入固定 640）
# 方案依据：《docs/视觉能力/足球与球门识别训练部署方案.md》§6.3
#
# 用法：
#   bash export_onnx.sh /workspace/runs/football_goal/baseline/weights/best.pt
set -euo pipefail

PT="${1:?用法: bash export_onnx.sh <best.pt 路径>}"
IMGSZ="${IMGSZ:-640}"

pip install -q ultralytics
yolo export model="$PT" format=onnx imgsz="$IMGSZ" simplify=True

ONNX="${PT%.pt}.onnx"
echo "产物: $ONNX"
echo "入库（PC 仓库）:"
echo "  cp $ONNX models/football_goal_<run名>_${IMGSZ}.onnx"
echo "并在 models/README.md 登记版本与 val 指标"
