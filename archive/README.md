# archive/（统一归档目录）

本目录集中存放“历史快照”与“运行产物”，不参与当前代码包结构。

## 目录说明

| 路径 | 内容 | 是否纳入 Git |
|------|------|-------------|
| `backup/` | 历史代码/数据快照（旧一步一定位版、旧 tag_poses） | ✅ 跟踪，但**不参与运行与 import** |
| `result/` | 模拟日志、轨迹图、多视角采集、标定 json 等运行产物 | ❌ 被 `.gitignore` 忽略，由各脚本自动写入 |

## 使用约定

- 需要查看旧实现时，优先使用 `git log` / `git show`，本目录内的 backup 只作就地留档；
- 当前所有脚本的产物统一写入 `archive/result/`（通过 `core/paths.py` 的 `RESULT_DIR` 定义）；
- 不要手工把 `archive/result/` 提交进 Git。
