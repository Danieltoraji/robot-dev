# 九宫格视觉导航 — 赛道计分与机器人控制系统

> 清华大学具身智能课程 · 赛道关卡 | 自动计分 + 自主导航

## 📖 项目简介

本项目是“具身智能挑战赛”赛道中的一个独立关卡模块。赛道采用 **1m × 1m 九宫格布局**，机器人需按顺序从入口依次到达面板 **1→2→3→4→5→6→7** 的中心区域。

系统由两个独立部分组成：

- **赛道计分系统**（ESP32-S3）：通过8个微动开关组成的3×3矩阵检测机器人到达，自动完成计分、计时和成绩上报
- **机器人控制系统**（树莓派）：基于视觉反馈的闭环导航，识别数字与颜色并自主移动

两部分完全解耦，互不依赖。

## 🧰 软硬件环境

### 赛道计分系统

| 项目 | 规格 |
|------|------|
| 主控 | ESP32-S3 |
| 传感器 | 6×6×5mm 微动开关 × 8 |
| 显示 | 0.96寸 OLED（I2C） |
| 开发工具 | Arduino IDE 2.x |
| 编程语言 | C++（Arduino框架） |
| 通信 | 串口（115200 baud）/ WiFi（HTTP POST） |

### 机器人控制系统

| 项目 | 规格 |
|------|------|
| 主控 | Raspberry Pi 4B / Zero 2W |
| 摄像头 | Raspberry Pi Camera Module 3 |
| 操作系统 | Raspberry Pi OS (Debian 12) |
| 开发语言 | Python 3.11 |
| 视觉库 | OpenCV 4.5+ |
| 机器学习 | scikit-learn（SVM） |
| 特征提取 | HOG（scikit-image） |
| 底盘控制 | Hiwonder Robot SDK |
| 动作组文件 | climb_stairs、go_forward_one_step、turn_left_small_step 等 |

## 🚀 快速开始

### 1. 赛道计分系统

1. 硬件连接
2. 打开 `esp32/NineGrid_ScoreSystem.ino`，在Arduino IDE中上传
3. 打开串口监视器（115200），输入布局数据（如 `6,5,4,3,2,1,7,8`）
4. 触发任意微动开关自动开始计时计分

### 2. 机器人控制系统

```bash
# 激活虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install numpy==1.23.5 scipy==1.11.4 opencv-python scikit-image scikit-learn joblib pillow

# 准备动作组文件
# 将动作组文件（.d6a）放置到 Hiwonder 动作组目录中

# 启动主程序
jupyter notebook robot/mainv0.2.ipynb
1. 目录结构
text
NineGrid_ScoreSystem/
├── README.md                          # 项目说明
├── esp32/                             # 赛道计分系统程序
│   └── NineGrid_ScoreSystem.ino
└── robot/                             # 机器人控制程序
    ├── mainv0.2.ipynb                 # 主控制流程
    ├── action.py                      # 动作控制（移动/转向/云台）
    ├── capture.py                     # 图像采集（fswebcam + OpenCV）
    ├── extract_digit_roi.py           # HSV颜色分割与候选区域提取
    ├── candidate_classifier.py        # HOG+SVM数字分类与融合
    ├── identify.py                    # 目标识别接口（yaw/proximity）
    └── models/digit_classifier_mask.pkl # SVM数字分类模型
3. 计分规则
评分项	分值	说明
任务完成分	70分	按正确顺序到达面板1~7各得10分
完成分	30分	15分钟内完成全部目标得30分，否则0分
总分	100分	—
4. 数据上报
比赛结束后系统输出JSON格式成绩，支持：

串口输出：直接显示在串口监视器

WiFi上报：ESP32通过HTTP POST主动上报到裁判主机（需在代码中配置正确的SERVER_URL）

关卡名称：九宫格视觉导航
组员：叶雨岑、陶鲁玥
完成时间：2026年8月

text


## 关卡说明书 — 第3部分：技术栈和开发环境

> **注意**：本部分仅描述赛道计分系统。机器人控制部分已移至第5部分。

**1. 开发环境**

| 项目 | 版本/规格 | 说明 |
|------|----------|------|
| 开发板 | ESP32-S3（清华大学硬件设计比赛专用版） | 主控芯片 |
| 开发工具 | Arduino IDE 2.x | 程序开发与烧录 |
| 编程语言 | C++（Arduino框架） | — |
| 通信协议 | 串口（UART），波特率 115200 | 调试与数据输出 |
| 通信协议 | WiFi（HTTP POST） | 成绩上报 |

**2. 关键库**

| 库名称 | 版本 | 用途 |
|--------|------|------|
| `WiFi.h` | 内置（ESP32核心库） | 连接WiFi网络，用于成绩上报 |
| `HTTPClient.h` | 内置（ESP32核心库） | 发送HTTP POST请求，上传JSON成绩 |
| `Wire.h` | 内置（Arduino核心库） | I2C通信，驱动OLED显示屏 |
| `Adafruit_SSD1306.h` | v2.5.17 | OLED显示屏驱动 |
| `Adafruit_GFX.h` | v1.12.6 | OLED图形绘制基础库 |

本项目除Adafruit系列库外，无其他第三方依赖，所有库均可通过Arduino IDE的库管理器直接安装。

**3. 硬件工具**

| 工具 | 用途 |
|------|------|
| 万用表 | 通断测试、电压检测、二极管方向判断 |
| 电烙铁（30W-60W） | 洞洞板焊接、飞线连接 |
| 热缩管（2mm/3mm/8mm） | 焊点绝缘处理 |
| 0.5mm²单股硬导线 | 微动开关引脚延长、洞洞板飞线 |
| 斜口钳 | 剪断元件多余引脚 |


## 关卡说明书 — 第5部分：机器人控制逻辑

**1. 软件架构**

本关卡的软件运行在机器人端 Raspberry Pi 上，以 Python 为主要开发语言，采用模块化设计。

主要模块如下：

| 文件 | 主要功能 |
|------|----------|
| `mainv0.2.ipynb` | 主控制流程及实验调试入口 |
| `action.py` | 机器人移动、转向及云台控制 |
| `capture.py` | 摄像头图像采集（fswebcam + OpenCV） |
| `extract_digit_roi.py` | HSV颜色分割、候选区域提取、数字Mask提取 |
| `candidate_classifier.py` | HOG + SVM 数字识别及多信息融合 |
| `identify.py` | 对外提供目标识别、yaw及距离信息 |
| `models/digit_classifier_mask.pkl` | 已训练的黑色数字 SVM 模型 |

模块调用关系为：`mainv0.2.ipynb` 作为主控入口，调用 `identify.py` 获取目标信息、调用 `action.py` 执行动作；`identify.py` 依赖 `candidate_classifier.py` 完成候选区域分类，最终由 `extract_digit_roi.py` 完成颜色分割与候选区域提取；`capture.py` 负责图像采集。

其中两个视觉核心模块的设计要点如下。

**候选区域提取（`extract_digit_roi.py`）：**

1. HSV颜色分割：针对红、橙、黄、绿、蓝、紫、粉七种颜色建立mask。其中绿色通过 H ∈ [100°, 160°] 且 S/V ≥ 1.15 判定，蓝色通过 H ∈ [207°, 220°]、动态S阈值及V范围判定，二者均不使用简单的HSV长方体；
2. 形态学处理：`MORPH_CLOSE` 连接颜色块内部断裂；
3. 轮廓检测与几何筛选：面积 15000 ~ 500000、长宽比不大于3，并用Solidity过滤细长或破碎区域；
4. 对比度与洞分析：排除低对比度区域，利用洞结构区分数字牌与纯色干扰；
5. 数字Mask提取：基于主颜色HSV距离 + Otsu阈值分割，得到黑白数字图，供SVM模型验证。

**数字分类与融合（`candidate_classifier.py`）：**

- 颜色直接映射为 `color_digit`（1-7），作为最终数字的主要依据；
- 数字Mask缩放为 64×64，提取HOG特征（9方向、8×8 cell、2×2 block、L2-Hys归一化），由SVM模型预测数字与置信度；
- 最终候选评分 = 0.65×候选框评分 + 0.20×模型置信度 + 0.15×洞评分。

**目标识别与位姿计算（`identify.py`）：**

- `identify(id, image)` 从候选结果中筛选 `final_digit == id` 的目标，取评分最高者，返回 `(success, yaw, proximity)`，是对外核心接口；
- `yaw = -(dx / image_width) × CAMERA_FOV（60°）`，目标在画面左侧时 yaw > 0，右侧时 yaw < 0；
- `proximity` 根据目标框宽度划分为 NEAR / MID / FAR 三个等级，用于距离决策；
- `identify_color(color, image)` 检测目标颜色在画面中的面积占比，供到达判定使用。

**2. 主要技术栈**

| 技术 | 版本 | 用途 |
|------|------|------|
| Python | 3.11 | 主要开发语言 |
| OpenCV | — | 图像读取、缩放、HSV转换、二值化、轮廓检测及形态学处理 |
| NumPy | 1.23.5 | 图像矩阵及数值计算 |
| SciPy | 1.11.4 | 数值计算 |
| scikit-image | — | HOG特征提取 |
| scikit-learn | — | SVM分类器训练及预测 |
| joblib | — | 模型保存与加载 |
| Pillow | — | 部分图像处理及数据增强 |
| Jupyter Notebook | — | 机器人程序调试、参数标定及实验记录 |
| fswebcam | — | 摄像头图像采集 |
| Hiwonder Robot SDK | — | 机器人底盘、舵机及动作组控制 |

**3. 开发环境**

机器人端使用 Raspberry Pi OS，并在 Python 虚拟环境中运行项目。

系统级依赖：
```bash
sudo apt update
sudo apt install -y fswebcam libcamera-dev
创建并激活虚拟环境：

bash
python3 -m venv venv
source venv/bin/activate
安装 Python 依赖：

bash
pip install numpy==1.23.5
pip install scipy==1.11.4
pip install opencv-python
pip install scikit-image
pip install scikit-learn
pip install joblib
pip install pillow
4. 开发与维护建议

由于本项目的视觉算法依赖较多参数，后续维护赛道时应尽量保持以下内容稳定：

Python大版本及NumPy、SciPy等基础库版本保持一致；

不随意更换摄像头及其安装位置，不改变云台安装角度；

更换数字牌颜色或打印方式后，应重新进行HSV标定；

数字字体、打印尺寸或拍摄条件明显变化时，应重新采集数据并训练数字分类模型。

5. 控制逻辑

本部分介绍机器人端在九宫格赛道验收过程中采用的具体控制逻辑，并作为后续使用本关卡进行实验时的 Baseline。

机器人控制程序采用层次化有限状态机组织导航过程。go_to()、turn_to()、search_target() 和 arrived() 分别承担不同层级的状态控制功能。机器人从当前位置前往指定数字牌的整体状态流转如下：

text
搜索目标（SEARCH）
        ↓
    找到目标
        ↓
方向跟踪（TRACKING）←────── 目标丢失
        ↓ 已对准
前进修正（MOVE_FORWARD）
        ↓ proximity 达到近距离
最终接近（FINAL_APPROACH）
        ↓ 颜色出现→消失
已到达（ARRIVED）→ 进入下一目标
在上述过程中，如果目标丢失则立即重新搜索；如果前进效率低下（疑似卡滞），则执行后退 + 越障脱困；目标未进入近距离区域时，继续执行“前进—重新识别—方向修正”的循环。

顶层导航 go_to()

go_to(id, COLOR_THRESHOLD, COLOR_LOST_THRESHOLD) 是完成一次“前往目标数字”任务的顶层控制函数。

程序首先根据目标数字设置该颜色专用的阈值：

id	ARRIVE_THRESHOLD	COLOR_THRESHOLD	COLOR_LOST_THRESHOLD
1	450	0.16	0.10
2	450	0.05	0.01
3	370	0.05	0.015
4	450	0.14	0.12
5	450	0.17	0.10
6	450	0.002	0.001
7	450	0.15	0.10
随后云台恢复前视姿态，拍摄第一帧并调用 turn_to() 对准目标，之后分为两个阶段：

远距离阶段：当 proximity < ARRIVE_THRESHOLD 时循环执行——大步前进 → 重新拍摄 → turn_to() 对准修正，形成“前进—识别—修正”闭环；若本次 proximity 相对上次变化量小于5，认为前进效率低、疑似卡滞，执行后退 + 越障脱困后重新识别；

最终接近阶段：调用 arrived()，切换为低头颜色检测 + 小步前进的精细控制。

方向跟踪 turn_to()

turn_to(id,
image
) 负责根据目标当前的 yaw 不断调整机器人朝向，使目标进入机器人正前方。核心参数：

BIG_TURN_THRESHOLD = 30°：大、小角度转向的分界；

epsilon = 8°：认为已对准的阈值；

INITIAL_SMALL_TURN_ANGLE = 4°：初始估计的单次小转角度。

根据 |yaw| 划分三个控制区域：

|yaw| ≤ 8°：认为已对准，返回 proximity；

|yaw| > 30°：执行一次大角度转向；

8° < |yaw| ≤ 30°：根据在线估计的小转角度执行多次微调。

由于实际转动角度受地面摩擦、负载、电池状态等影响，小转角度采用在线估计而非固定值，可减少累计角度误差。

目标搜索 search_target()

当 turn_to() 识别失败时立即进入搜索：

最多进行20次机身转动；

每次先以前视姿态识别，失败则切换到低头姿态再识别；

找到目标后返回方向信息；

20次后仍未找到则抛出 RuntimeError，防止无限搜索。

到达判定 arrived()

进入目标附近后，控制目标由“目标在哪里”转为“机器人是否已经真正经过目标格”：

云台切换至低头姿态；

循环检测目标颜色在画面中的占比；

当颜色占比超过阈值时标记为“已看到颜色”；

当“已看到颜色”且颜色占比低于丢失阈值时，认为机器人已压过目标格；

对 id 6（紫色牌）有专门降低阈值要求的处理。

关键参数汇总：

参数	当前参考值	作用
BIG_TURN_THRESHOLD	30°	大、小角度转向的分界
epsilon	8°	判断目标是否基本对准
ARRIVE_THRESHOLD	370 ~ 450	是否进入最终接近阶段（按id变化）
COLOR_THRESHOLD	按id变化	判断目标颜色是否出现
COLOR_LOST_THRESHOLD	按id变化	判断目标颜色是否消失
CAMERA_FOV	60°	yaw计算使用的水平视场角
最大搜索次数	20	防止无限搜索
Baseline 运行流程：

环境检查：确认摄像头、动作组、云台、Python环境及模型文件正常；

初始化：运行 
action
.init_servo()；

视觉识别测试：调用 
identify
(id,
image
)；

方向控制测试：单独测试 turn_to(id,
image
)；

远距离导航测试：测试 go_to(...) 闭环；

到达检测测试：单独检查 arrived(...)；

完整测试：按比赛要求执行完整任务。