# nine_grid HSV 再标定流程（机器人拍照 → PC 拟合 → 回传）

> 适用场景：换场地、换灯光、换贴纸/印刷，或现场发现某色整块漏检时。
> 只管**颜色窗口**（HSV 调色板）；几何类标定（地面单应、到达门、分区比例）是另一条线，别混在一起改。

## 0. 这件事的形状

- 运行时决定颜色的只有一处：`vision/nine_grid_detector.build_color_mask()`，走 `PALETTE`（`USE_PALETTE = True`）。
- `PALETTE` 是"从我们自己的实拍帧拟合"出来的，并且可以被 `models/nine_grid/palette.json` **逐色覆盖**（存在即优先，见 `_active_palette()`）。
- 现状（已逐字节核对机器人代码）：机器人跑的是**代码内置**的 PALETTE；机器人上既没有 `palette.json`，也没有拟合用的实拍帧 ⇒ 标定一直是在 PC 上做的、结果随代码上机。
- 本流程把回路走全：**机器人拍原图 → 拉回 PC 拟合 → 写 palette.json → 回传 → 复核生效**。以后换场地只管重跑一遍。

## 1. 先懂一件事：HSV 是在哪个空间算的

运行时链路（`vision/nine_grid_detector.detect_panels`）：

```
fswebcam 原生 2592×1944 JPEG
  → cv2.resize 到 WORK_WIDTH = 1296（等宽缩放）
  → normalize_illumination()   # 灰世界白点 + 曝光基准；找不到可信白点则**不归一化**
  → cv2.cvtColor(BGR2HSV)
  → build_color_mask() 用 PALETTE
```

**所以关卡确实"处理过"**：颜色不是在原始画面上判的，而是在「工作分辨率 + 光照归一化后」的 HSV 上判的。

PC 拟合工具已经把这 3 步逐字复刻（`tools/calibrate_palette.py` 的 `collect()`），注释里还记着当年漏掉这一步的事故：样本落在原始空间，`probe_c` 粉 H=141 而不是归一化后的 166，拟合出的窗口与运行时不匹配。

⇒ **结论：不要在机器人上预处理，拉原图就对了。** 三个理由：

1. 归一化是**逐帧确定性**函数（白点由该帧自己估出来），PC 上重算 = 运行时算的同一个结果；提前归一化再存盘，等于把同一个增益算两遍，还多一次 JPEG 有损编码。
2. 原图留着诊断能力：白点、增益、过曝比例、"这帧到底归一化了没有"都能事后复核；写死成归一化图就查不了了。
3. 探针本来就是按"存原图"实现的（`cv2.imwrite(path, frame)`，`frame` 是 `capture_frame()` 的原图）。

> 唯一"例外"恰恰支持这个选择：如果哪天改了归一化参数（`NORM_*`、白点判据），旧原图可以**在 PC 上重算**；存成归一化图就没救了。

## 2. 四步流程

### 步骤 1（机器人，现场）：拍原图

前置——把探针同步到机器人（机器人上的 `tools/` 可能落后于 PC）：

```bash
python tools/sync_to_robot.py --paths tools/field_probe_ninegrid.py core/robot_core.py --full
```

机器人在场地就位、**空闲时**执行（探针会动俯仰/头部舵机与相机，别在跑关卡时跑）：

```bash
/home/pi/jupyter-env/bin/python3 tools/field_probe_ninegrid.py --frames 12 --tag 20260928-新场地
```

它做四件事：① 锁白平衡/对焦/曝光并读回校验；② `auto_calibrate_exposure()` 把画面均亮闭环到目标（**与真实关卡开局同一个动作**）；③ 按 俯仰（`PITCH_NAV` / `PITCH_DOWN`）× 5 个头部档采帧，原图存到 `field_probe/<tag>/`；④ 逐帧跑真检测器，写 `probe_summary.json`。

**判据（不达标先修现场，别急着拟合）**：

| 项 | 达标线 | 依据 |
|---|---|---|
| 过曝比例 | 所有帧 ≤ 5% | `CAM_CLIP_MAX = 0.05` |
| 白点跨帧漂移 | ≤ 6（0-255） | 相机锁定是否真的生效 |
| 光照归一化 | 每帧 `det_norm.applied = true` | 未归一化的帧**不在调色板空间** |
| 7/7 帧 | 越多越好（不必全 7/7） | 缺色样本会被步骤 3 的闸门拦住 |
| 归一化增益 | 落在 [0.75, 1.40] | 超区间 ⇒ 光照偏离参考白点，样本噪声大 |

### 步骤 2（PC）：拉回

```bash
python tools/pull_from_robot.py --remote Robot_Competition/field_probe/20260928-新场地 --dest archive/result/palette_refit/20260928-新场地
```

`archive/` 已在 `.gitignore` 里，原图不进版本库。

### 步骤 3（PC）：拟合

先跑两遍**只打印**（不带 `--write`），对比采样口径：

```bash
python tools/calibrate_palette.py --frames archive/result/palette_refit/20260928-新场地
python tools/calibrate_palette.py --frames archive/result/palette_refit/20260928-新场地 --legacy-masks
```

- **不带 `--legacy-masks`**：用当前生效的 PALETTE 选样本、贴标签。与现状连续；但如果新场地下某色已被当前窗口整块漏掉，它压根不进样本（看不到问题）。
- **带 `--legacy-masks`**：改用历史手调窗口（`COLOR_THRESHOLDS`）选样本，独立于待拟合的窗口，是"自举"口径，能暴露"当前窗口已不合适"。

两个口径都看，**唯一命中数高、且七色样本齐的那个更可信**；两者差很多 ⇒ 新场地的光/贴纸已明显偏离，值得多拍几帧。

**验收判据**：

| 项 | 达标 | 说明 |
|---|---|---|
| `唯一命中 X/Y` | 越接近 1 越好（历史验收 73/73、114/114） | 多色命中/漏检要看是谁：红↔橙 H 中心只差 7°，靠形状仲裁兜，少量歧义可接受 |
| `七色样本齐全（各 ≥3）` | 必须 | 不齐 ⇒ 工具**拒绝写入**（防"逐色覆盖"造成静默沿用旧窗口） |
| 相邻色 H 间隔 | 与上一版对比无突变 | 尤其红↔橙、粉↔红（环绕那对） |

通过后写文件：

```bash
python tools/calibrate_palette.py --frames archive/result/palette_refit/20260928-新场地 --write
```

产物 `models/nine_grid/palette.json`：含 `fitted` 七色窗口 + 用到的帧名 + 采样口径（`sampled_with`）。

### 步骤 4（PC → 机器人）：回传

```bash
python tools/sync_to_robot.py --paths models/nine_grid/palette.json --full
```

- 该命令会顺带推 `EXTRA_FILES` 里的 `archive/result/ninegrid_homography.json` 与 `stairs_hurdle_calib.json`（运行时标定产物，推了没坏处）。
- ⚠️ **不要**用 `tools/sync_nine_grid_to_robot.py`：它的 `ROUTE_FILES` 只列了 10 个代码文件，**不含 `models/`**，推不了 palette.json。

### 步骤 5（复核）：确认真的生效

不碰相机、不动舵机，纯 import。**必须先 chdir**：`exec_on_robot` 的内核工作目录是
Jupyter 根（`/home/pi`），不是仓库根，直接 `import vision` 会报
`ModuleNotFoundError: No module named 'vision'`。

```bash
python tools/exec_on_robot.py --code "import os, sys; os.chdir('/home/pi/Robot_Competition'); sys.path.insert(0, '/home/pi/Robot_Competition'); import vision.nine_grid_detector as d; print(d._active_palette())"
```

期望：输出里有 `[调色板] 已载入现场重标定文件 …（覆盖 7 种颜色）`，且七个窗口等于本地 `palette.json`。
若显示"覆盖 6 种颜色"或更少 ⇒ 回到步骤 3 看闸门输出，别就这么上场。

## 3. 坑表

| 坑 | 现象 | 规避 |
|---|---|---|
| 空间不一致 | 拟合出来的窗口在真机上整块漏检 | 只拉原图；PC 侧不要自己先做归一化/缩放 |
| 缺色静默沿用 | 以为标定过了，其实某色还是旧值 | 步骤 3 的完整性闸门（`--min-samples`） |
| 循环依赖 | 用待拟合的窗口选样本 ⇒ 错误自我强化（历史：红只剩 2 个样本） | 加 `--legacy-masks` 跑一遍做对照 |
| `palette.json` 存在即优先 | 后来改了代码默认值却不生效 | 两者只能有一个生效源；改代码默认值后删掉 `palette.json` |
| 场次混样 | 不同场地/灯的帧混进同一次拟合 | `--tag` 分目录 + `--frames` 只给本次目录 |
| 曝光工况不一致 | 探针帧与关卡帧不在同一档 | 探针默认跑 `auto_calibrate_exposure`（与关卡开局一致） |
| 机器人 `tools/` 落后 | 探针版本与 PC 不一致 | 步骤 1 前置同步；拟合工具只在 PC 跑 |
| 在跑关卡时拍帧 | 抢占相机/舵机 | 机器人空闲时执行 |

## 4. 回滚

删掉机器人上的 `models/nine_grid/palette.json` 即回到代码内置表（`_active_palette()` 找不到文件时用 `PALETTE`）。

## 5. 可选：数字模板

`models/nine_grid/digit_templates.npz`（颜色有歧义时的形状仲裁）只在**换贴纸 / 换相机 / 换印刷**后才需要重标：

```bash
python tools/gen_ninegrid_digit_templates.py --write
```

纯换灯光、换场地不必动它。

## 6. 一句话

颜色窗口跟着**灯和贴纸**走，不跟着场地几何走；HSV 一变就重跑这五步，几何类标定另算。
