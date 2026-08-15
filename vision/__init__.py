# -*- coding: utf-8 -*-
"""视觉识别包（core 侧通用基础能力）。

设计约定（对齐项目「core 宁少勿多 / 关卡自治」原则）：
  - 本包只提供「检测器 + 数据模型」这类所有关卡都能复用的基础能力。
  - 检测器统一约定实现 detect(frame) 方法，输入一帧 BGR 图像。
  - 检测器的阈值 / 参数由关卡构造时传入；识别结果的语义解释与决策逻辑
    一律留在关卡层，不要写进本包，以免跨关卡混乱。
  - 具体检测器（line / color / digit / yolo）按需
    `from vision.xxx_detector import XxxDetector` 导入，避免包导入时
    强依赖 cv2 / 深度学习运行时。
"""

from typing import Protocol, runtime_checkable

import numpy as np

from .detection import Detection, LineResult, DigitResult

__all__ = ["Detection", "LineResult", "DigitResult", "Detector"]


@runtime_checkable
class Detector(Protocol):
    """检测器统一约定：实现 detect(frame) 方法即可。

    不同检测器返回不同类型：
      - 通用目标检测 → list[Detection]
      - 巡线         → LineResult
      - 数字         → DigitResult
    关卡按所用检测器的具体返回类型消费。
    """

    def detect(self, frame: np.ndarray):
        ...
