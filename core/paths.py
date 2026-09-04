# -*- coding: utf-8 -*-
"""项目路径常量（目录整理后统一入口）

所有需要写“运行/标定产物”的模块都应从本文件取 RESULT_DIR，
不要再硬编码根目录下的 result/。
"""
import os

# 仓库根目录 = core/paths.py 的上两级（core -> robot-dev）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 统一产物目录：archive/result（已加入 .gitignore）
RESULT_DIR = os.path.join(PROJECT_ROOT, "archive", "result")
