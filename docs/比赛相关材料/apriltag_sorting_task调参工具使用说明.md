# apriltag_sorting_task 调参工具使用说明

> 工具文件：`tools/calib_tuner.py`（PC 端，Windows/Linux 均可）
> 配套主程序：`levels/apriltag_sorting_task.py`
> 作用：不碰代码即可调节主程序全部阈值参数，实时查看检测效果与状态机决策，
> 导出 `calib_config.json` 供机器人自动加载。

---

## 1. 工具是什么

`calib_tuner.py` 是一个 OpenCV GUI 调参器，覆盖主程序里 6 类共 69 项参数：

| 参数页 | 内容 |
|---|---|
| 1 视觉 | 蓝色 LAB 阈值、红色 HSV 阈值、最小面积、目标中心 x |
| 2 颜色接近 | cy 三档、dx 三档、抓取面积/cy 阈值、step4 兜底步数 |
| 3 Tag 放置 | 板厚、放置距离带（米）、tag 像素档位 |
| 4 巡线 | 巡线中心、直行阈值、竖直高宽比、丢线策略、低头角 |
| 5 终点+动作 | 终点 yaw 窗口、直行步数、抓取/放置补步、路线动作计数 |
| 6 卡尔曼 | 7 组滤波器的 Q/R |

**默认值自动同步**：工具用 ast 解析主文件源码提取默认值（不 import 主文件，
PC 上没有 hiwonder SDK 也能跑），所以主程序里改了参数，工具永远是最新默认值，
不会出现两边不一致。

## 2. 启动

在 PC 仓库根目录（推荐用仓库自带虚拟环境，OpenCV/numpy 已装好）：

```bash
.venv/Scripts/python.exe tools/calib_tuner.py --image archive/result/pulled_photos   # 目录
.venv/Scripts/python.exe tools/calib_tuner.py --image 某张照片.jpg                    # 单张
.venv/Scripts/python.exe tools/calib_tuner.py --dump                                 # 无GUI: 打印默认配置
```

- `--image` 给**目录**时按文件名排序，`n`/`b` 键前后翻页；
- `--dump` 不需要图形界面，打印当前全部参数的 JSON，用于脚本化检查。

## 3. 按键速查表

| 按键 | 功能 |
|---|---|
| `n` / `b` | 下一张 / 上一张照片 |
| `TAB` | 切换视图：原图 → 蓝色 LAB 掩膜 → 红色 HSV 掩膜+ROI |
| `1` ~ `6` | 切换参数页（对应上表 6 页） |
| `s` | 保存 `levels/calib_config.json` |
| `p` | 控制台打印可粘贴的 Python 常量块 |
| `r` | 全部参数重置为代码默认值 |
| `q` / `ESC` | 退出 |
| 鼠标左键 | 点击图片任意像素，打印该点 **BGR / LAB / HSV** 值 |

**图片窗口上的实时信息**（用当前滑条值对当前照片即时计算）：

- 黄色圆点 + 文字：蓝色海绵检测框中心 `cx/cy`、面积 `area`、偏差 `dx`；
- 蓝色竖线：`CENTER_X` 目标中心线；
- `step1~step4` 四行：把当前检测值喂进 `approach_color` 状态机，
  显示每个 step 会执行的**动作**（绿色 `DONE` = 判定到位进入抓取）；
- `line_cx / vertical`：红线加权中心与"线是否竖直"判定。

**Tag 模拟窗口**：`dist(mm)`、`dx(px)`、`cy(px)` 三个滑条模拟一个 tag 观测值，
实时显示 `approach_tag` 会命中的分支（PC 上没有 hiwonder 的 apriltag 库，
tag 检测不能在 PC 本地做，用模拟值调 tag 阈值最方便）。

## 4. 完整调参工作流（一次只改一个参数）

1. **机器人在场地跑一轮**（或摆拍）：主程序自动在机器人
   `/home/pi/codes/pictures/` 存快照（`snap_*` 颜色+tag、`line_*` 巡线），
   日志在 `/home/pi/Robot_Competition/levels/apriltag_sorting_task.log`。
2. **把照片拉回 PC**：
   ```bash
   python tools/pull_from_robot.py --remote codes/pictures --dest archive/result/pulled_photos
   ```
3. **打开工具调参**：对着照片调视觉阈值（点击像素读 LAB/HSV 定颜色范围，
   看掩膜视图确认只有目标区域变白）；调流程阈值时看 `step1~4` 决策行和
   Tag 模拟窗口的分支是否按预期命中。
4. **导出**：按 `s` 保存 `levels/calib_config.json`。
5. **同步上机器人**：
   ```bash
   python tools/sync_to_robot.py --remote-root Robot_Competition --paths levels
   ```
6. **重跑验证**：对比日志/快照看行为变化；有效就 git 提交存档，无效就 `r` 重置再来。

## 5. 主程序如何加载配置（优先级规则）

- 主程序启动时自动查找**脚本同目录**的 `calib_config.json`（即机器人上
  `/home/pi/Robot_Competition/levels/calib_config.json`）：
  - 文件**不存在** → 行为与原来完全一致（代码内常量）；
  - 文件存在 → 按白名单覆盖模块常量（69 项里的常量部分），
    命令行参数（如 `pick_area_threshold`）**仅在命令行未显式指定时**才被覆盖，
    即命令行传参 > calib_config.json > 代码默认值；
  - JSON 损坏 → 只打警告，不崩溃。
- 恢复出厂行为 = 删除机器人上的 `calib_config.json`（或在工具里按 `r` 后重新导出）。

## 6. 各参数标定方法精简表

> 详细方法论见每条"症状与调法"。总原则：视觉阈值用现场照片静态标定；
> 距离/角度类阈值必须先实测动作物理量（`tools/_measure_turn_angle_vision.py`
> 测转角、卷尺测步长），再换算。

### 6.1 视觉（页 1）
| 参数 | 标定方法 |
|---|---|
| `BLUE_LAB_MIN/MAX` | 场地灯光下拍海绵照片，鼠标点击海绵读 LAB 值，滑条调到只有海绵在掩膜视图里变白 |
| `RED_H/S/V` 8 项 | 同法对红胶带调 HSV；红色两段结构保留；S/V 下限 80 是排除黑白线的，误检白线就提高 |
| `COLOR_AREA_MIN` | 海绵在 2~3 米外能稳定检出即可；太高会"看不见远处的海绵" |
| `CENTER_X` | 故意偏右 30px（摄像头偏移/手在偏右）。停稳后海绵总偏→调它 |

### 6.2 颜色接近（页 2）
| 参数 | 标定方法 |
|---|---|
| `COLOR_FAR_Y/NEAR_Y/TOO_NEAR_Y` | 把海绵放到"该抓的距离"，读日志 cy 实测定近档；**这台机器头部俯角下 cy≈280 就够近了，不要照搬官方 340** |
| `COLOR_X_TURN/LARGE/FINE` | 按实测横移量换算像素；来回摆头=档太大，停位偏=精调档太大 |
| `pick_area_threshold` | 在"该抓距离"读 area，阈值定在略低 10%；太大会推着海绵走 |
| `pick_y_threshold / pick_too_near_y` | 与 cy 档位同步调 |
| `STEP4_MAX_FORWARDS` | 兜底：连续前进 N 步强制判定到位，防止推海绵无限前进 |

### 6.3 Tag 放置（页 3）
| 参数 | 标定方法 |
|---|---|
| `TAG_PLACE_NEAR_M/FAR_M` | 带海绵（锁手）在不同距离试放，成功率最高的距离带定近档 |
| `TAG_PLACE_BIG_M` | 按实测大步前进步长定，太早用大步会冲过放置点 |
| `BOARD_DEPTH_M` | 量比赛板实际厚度 |
| `TAG_FAR_Y/NEAR_Y/TOO_NEAR_Y` | 仅 PnP 失效时用（像素回退），照 6.2 方法按实测 cy 定 |
| `TAG_X_TURN/LARGE/FINE` | 与颜色横向档同理 |

### 6.4 巡线（页 4）
| 参数 | 标定方法 |
|---|---|
| `LINE_CENTER_X` | 画面中线 320，摄像头偏移时修正 |
| `LINE_TURN_THRESHOLD` | 蛇形摆动=调高；反应迟钝=调低 |
| `SEARCH_LINE_ALIGN_THRESHOLD` | 放完海绵转身找线时 |dx| 收敛到多少算对正；太小永远对不齐 |
| `VERTICAL_LINE_RATIO` | 正对红线时看 `line_*.jpg` 快照里红色矩形实测高宽比，定在该值以下 |
| `LINE_LOST_HOLD/TIMEOUT/MAX_LOST_TURNS` | 按实测转角定"左大转几次能找回线" |
| `line_head_delta` | 低头角决定红线落在哪几行 ROI，看快照调 |

### 6.5 终点 + 动作计数（页 5）
| 参数 | 标定方法 |
|---|---|
| `END_YAW_LOWER/UPPER` | 正对 tag 站好读 yaw 波动范围，窗口取略大于波动 |
| `WALK_STEPS / MAX_TURN` | 终点距离 ÷ 实测单步距离 |
| `PICK_FINAL_STEPS / PLACE_FINAL_STEPS` | 总抓空→加大；总撞上→减小 |
| `MAX_PICK_RETRIES` | 一般不动 |
| `post_pick_left_turns / post_pick_forward_steps` | 用实测转角+步长在地图上算"抓取点→Tag 搜索通道"反推 |
| `back_steps_after_place` | 退多远才够转身看到身后红线，实测 |
| `line_final_steps / line_search_turns` | 终点距离换算 / 找线最多转几次 |

### 6.6 卡尔曼（页 6）
原则：**R（观测噪声）越大越平滑但越滞后；Q（过程噪声）越大越跟手但越抖。**
目标框发抖→先加 R；动作反应慢（转身过头）→加 Q。
注意 `kf_area` 的 Q=100 是有意设大的（面积变化快），"够近但面积上不去"要警惕滤波滞后。

## 7. 已知坑

- 工具 GUI 的窗口标题与滑条标签使用**英文缩写**（与参数名一致，如 `NEAR_Y`、
  `PLACE_FAR_mm`）：OpenCV 在 Windows 上对中文控件文字会显示乱码，
  控制台输出仍为中文。
- `CENTER_X=350` 不是 320，是故意偏置，别"顺手修回"。
- 主文件里 `END_FORWARD_TOL` 和 `--line-initial-steps` 定义了但流程未接线，
  调它们没有效果。
- 主文件已适配机器人 numpy 2.x（`ravel()`/`np.intp` 写法），改代码时不要回退。
- 快照在 320×240 检测帧上算面积，但 cx/cy 映射回 640×480 全画幅——两种坐标系
  别混用（cy 阈值是 480 高度系的，area 阈值是 320×240 面积系的）。
- 换场地光照后必须重标 BLUE_LAB 和红色 HSV，其余几何阈值一般不用重标。
