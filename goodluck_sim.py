# -*- coding: utf-8 -*-
"""
寻路算法测试程序（模拟器 + 可视化）

架构说明：
  PART A —— goodluck.py 的原样副本（勿改）。算法函数与 I/O 函数全部保留。
            当 goodluck.py 算法更新后，只需把 PART A 整体替换为新版内容
            （删除末尾主流程自动执行代码），PART B 无需任何改动。
  PART B —— 模拟器 + 可视化（测试专用）。通过 monkey-patch 在运行前用桩函数
            替换 PART A 的 I/O 函数（run_action / solve_pnp / set_head 等），
            算法函数（decide_panning_action / navigate_to_target 等）原样执行。

运行：python goodluck_sim.py
"""

# =====================================================================
# ===== PART A: goodluck.py 原样副本（勿改）===========================
# =====================================================================
# 说明：此区块是 goodluck.py 的逐字副本。唯一允许的改动是：
#   1. 硬件导入用 try/except 包裹（导入失败设为 None，不影响桩函数运行）
#   2. 删除末尾主流程自动执行代码（执行入口改由 PART B 的 run_simulation() 控制）
# 算法逻辑、常量、赛道数据、I/O 函数定义全部保持原样。

import time
import subprocess  # 拍照
import numpy as np

# 硬件/视觉库导入：在非机器人环境用 try/except 屏蔽，桩函数不调用真实硬件
try:
    import hiwonder.ActionGroupControl as AGC  # 动作库，必须包含该库
except Exception:
    AGC = None
try:
    import apriltag
except Exception:
    apriltag = None
try:
    import cv2
except Exception:
    cv2 = None
try:
    import hiwonder.ros_robot_controller_sdk as rrc
    from hiwonder.Controller import Controller
    board = rrc.Board()
    ctl = Controller(board)
except Exception:
    rrc = None
    ctl = None
    board = None

# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# 从这里开始可以更改
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||


# =====================================================================
# 赛道数据
# =====================================================================
# 这里是定义每个AprilTag的世界坐标，单位为厘米。
# 添加AprilTag的世界坐标，四个点的顺序为左上，右上，右下，左下。也即从左上角开始顺时针。

tag_poses = {}
tag_poses["36"] = np.array(
    [[44.7, 23.8, 41.5], [44.7, 18.8, 41.5], [44.7, 18.8, 36.5], [44.7, 23.8, 36.5]],
    dtype=np.float64,
)
tag_poses["37"] = np.array(
    [[20.6, 100, 41.7], [25.6, 100, 41.7], [25.6, 100, 36.7], [20.6, 100, 36.7]],
    dtype=np.float64,
)
tag_poses["38"] = np.array(
    [[95, 82.2, 41.7], [95, 77.2, 41.7], [95, 77.2, 36.7], [95, 82.2, 36.7]],
    dtype=np.float64,
)
tag_poses["39"] = np.array(
    [[76.5, 0, 42.1], [71.5, 0, 42.1], [71.5, 0, 37.1], [76.5, 0, 37.1]],
    dtype=np.float64,
)

# 赛道墙壁（不可通行区域），格式 [x_min, x_max, y_min, y_max]，单位cm
# 来自赛道说明：左墙、中墙、右墙；外框底/顶边在 distance_to_walls 中单独处理
WALLS = [
    [0, 5, 40, 100],    # 左墙
    [45, 55, 0, 60],    # 中墙
    [95, 100, 40, 100], # 右墙
]

# 以下定义机器人定点站立课目的目标点坐标和朝向。
stand_poses = {}
stand_poses["1"] = np.array([14.7, 21.3], dtype=np.float64)
stand_poses["2"] = np.array([23.1, 70], dtype=np.float64)
stand_poses["3"] = np.array([65, 79.7], dtype=np.float64)
stand_poses["4"] = np.array([74, 30], dtype=np.float64)


# =====================================================================
# 决策算法常量
# =====================================================================
ORIENTATION_THRESHOLD = 0.19  # 朝向差异模长阈值，约11°，是2sin(11°/2)的值
POSITION_THRESHOLD = 3.0  # 位置差异模长阈值，单位cm
STOP_TIME = 3  # 到达目标点后停留时间，单位秒
MAX_LOCATE_RETRIES = 5  # 定位失败最大重试次数
OBSTACLE_THRESHOLD = 15.0  # 避障容忍阈值，离墙最近距离小于此值则排除该动作
SAFE_MARGIN_CM = 3.0  # 安全点额外余量。实际安全点距离 = OBSTACLE_THRESHOLD + POSITION_THRESHOLD + SAFE_MARGIN_CM，确保机器人离墙足够远。

target_poses = {}
target_poses["1"] = np.array([20.0, 21.3], dtype=np.float64)
target_poses["2"] = np.array([23.1, 70], dtype=np.float64)
target_poses["3"] = np.array([65, 79.7], dtype=np.float64)
target_poses["4"] = np.array([74, 30], dtype=np.float64)

target_orientations = {}
target_orientations["1"] = np.array([1, 0], dtype=np.float64)
target_orientations["2"] = np.array([0, 1], dtype=np.float64)
target_orientations["3"] = np.array([1, 0], dtype=np.float64)
target_orientations["4"] = np.array([0, -1], dtype=np.float64)


# =====================================================================
# 动作组参数常量（所有的数值都需要重新标定！！！）
# =====================================================================
FORWARD_ONE_STEP_CM = 4.0  # go_forward_one_step（待标定）
FORWARD_ONE_SMALL_STEP_CM = 2.0  # go_forward_one_small_step（待标定）
BACK_ONE_STEP_CM = 4.0  # back_one_step（待标定）
LEFT_MOVE_CM = 2.9  # left_move（待标定）
RIGHT_MOVE_CM = 2.1  # right_move（待标定）
TURN_LEFT_SMALL_STEP_DEG = 21.0  # turn_left_small_step（待标定）
TURN_RIGHT_SMALL_STEP_DEG = 21.0  # turn_right_small_step（待标定）
TURN_LEFT_DEG = 30.0  # turn_left（估算值，待标定）
TURN_RIGHT_DEG = 30.0  # turn_right（估算值，待标定）
FORWARD_BIAS = 0.5  # 前进方向偏好权重，避免原地转圈
CAMERA_FORWARD_OFFSET_CM = 2.0  # 摄像头中心相对旋转中心的前后偏移（旋转中心在后方，cm）
TURN_LEFT_RADIUS_CM = 5.0  # 左转圆周运动半径（cm）
TURN_RIGHT_RADIUS_CM = 5.0  # 右转圆周运动半径（cm）

# =====================================================================
# 头部舵机参数常量
# =====================================================================
HEAD_CENTER = 1500
HEAD_RIGHT = 600
HEAD_LEFT = 2400
HEAD_MOVE_TIME_MS = 500  # 头部舵机转动等待时间，单位ms，对应旋动90°的时间。
HEAD_MOVE_TIME_MIN_MS = 100  # 小角度转头最小等待时间，单位ms
# 舵机脉宽→角度线性映射：angle_deg = (pulse - 1500) * SERVO_DEG_PER_US
# 标准500-2500μs对应±90°，即 90°/1000μs = 0.09
SERVO_DEG_PER_US = 0.09


current_position = None
current_orientation = None
next_stop = "0"
current_head_pulse = HEAD_CENTER  # 头部当前位置，初始中位

def run_action(name, times=1):
    """执行动作组"""
    AGC.runActionGroup(name, times=times)

def calculate_diff(target_pos):
    """计算当前位置/朝向与目标点的差异

    target_pos: 目标位置坐标数组（2D）。
    朝向差异始终用 target_orientations（两阶段共用）。
    """
    position_diff = np.array(current_position) - np.array(target_pos)
    orientation_diff = np.array(current_orientation) - np.array(
        target_orientations[next_stop]
    )
    return position_diff, orientation_diff

def calc_distance(position_diff):
    """计算水平距离"""
    return np.linalg.norm(position_diff)

def distance_point_to_rect(pos, rect):
    """点到矩形的最短距离（点在矩形内返回0）

    pos: [x, y]，rect: [x_min, x_max, y_min, y_max]
    """
    x, y = pos[0], pos[1]
    x_min, x_max, y_min, y_max = rect
    dx = max(x_min - x, 0, x - x_max)
    dy = max(y_min - y, 0, y - y_max)
    return float(np.sqrt(dx * dx + dy * dy))

def distance_to_walls(pos):
    """计算位置到最近墙壁的距离（cm）

    包含3个矩形墙 + 外框底边(y=0) + 顶边(y=100)。
    左右外框已被左/右墙覆盖；出入口 x=0/x=100, y∈[0,40] 为开口不约束。
    """
    pos = np.array(pos, dtype=np.float64)
    min_dist = float("inf")
    for rect in WALLS:
        min_dist = min(min_dist, distance_point_to_rect(pos, rect))
    # 外框底边、顶边
    min_dist = min(min_dist, float(pos[1]))       # y=0
    min_dist = min(min_dist, float(100 - pos[1])) # y=100
    return min_dist

def nearest_safe_point(pos):
    """找离 pos 最近的安全点（distance_to_walls >= SAFE_THRESHOLD）

    搜索阈值 SAFE_THRESHOLD = OBSTACLE_THRESHOLD + POSITION_THRESHOLD + SAFE_MARGIN_CM，
    确保返回的安全点离机器人足够远（> POSITION_THRESHOLD），必触发平移动作，
    避免危险区边界处“需逃离但无需导航”的死循环。
    若 pos 本身安全直接返回；否则以 1cm 步长、16方向螺旋搜索，
    返回首个安全点；兜底返回搜索范围内 distance_to_walls 最大的点。
    """
    SAFE_THRESHOLD = OBSTACLE_THRESHOLD + POSITION_THRESHOLD + SAFE_MARGIN_CM
    pos = np.array(pos, dtype=np.float64)
    if distance_to_walls(pos) >= SAFE_THRESHOLD:
        return pos

    # 16方向螺旋搜索，步长1cm，最大搜索半径100cm
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    best_point = pos.copy()
    best_dist = distance_to_walls(pos)
    for step in range(1, 101):
        for a in angles:
            candidate = pos + step * np.array([np.cos(a), np.sin(a)])
            d = distance_to_walls(candidate)
            if d >= SAFE_THRESHOLD:
                return candidate
            if d > best_dist:
                best_dist = d
                best_point = candidate.copy()
    print(f"nearest_safe_point: 未找到满足阈值的安全点，返回最优点 {best_point} (距离 {best_dist:.2f}cm)")
    return best_point

def capture_image():
    """拍照"""
    timestamp = int(time.time())
    filename = f"/home/pi/Pictures/photo_{timestamp}.jpg"
    cmd = f"fswebcam -r 2592x1944 --no-banner -S 3 {filename}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"拍照失败: {result.stderr}")
        return None

    print(f"照片已保存: {filename}")
    return filename

def detect_apriltag(filename):
    # load the input image and convert it to grayscale
    print("[INFO] loading image...")
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # define the AprilTags detector options and then detect the AprilTags
    # in the input image
    print("[INFO] detecting AprilTags...")
    options = apriltag.DetectorOptions(families="tag36h11")
    detector = apriltag.Detector(options)
    results = detector.detect(gray)
    print("[INFO] {} total AprilTags detected".format(len(results)))
    return results

def set_head(pulse, move_time_ms=HEAD_MOVE_TIME_MS):
    """转动头部舵机并等待到位；
    目标与当前位置相同则跳过。
    需要旋转时，按脉宽差（角度差）动态缩放等待时间。900μs≈90°为满量程，最小 HEAD_MOVE_TIME_MIN_MS。
    """
    global current_head_pulse
    if pulse == current_head_pulse:
        return  # 已在目标位置，无需等待
    # 按角度差动态调整等待时间，最小 HEAD_MOVE_TIME_MIN_MS
    delta = abs(pulse - current_head_pulse)
    dynamic_time = max(HEAD_MOVE_TIME_MIN_MS, int(move_time_ms * delta / 900))
    ctl.set_pwm_servo_pulse(2, pulse, dynamic_time)
    time.sleep(dynamic_time / 1000.0 + 0.2)
    current_head_pulse = pulse

def pulse_to_angle(pulse):
    """舵机脉宽→角度（度），右转为负，左转为正"""
    return (pulse - HEAD_CENTER) * SERVO_DEG_PER_US

def compensate_head_offset(head_pulse):
    """补偿头部偏转角：将相机朝向转换为机体朝向

    solvePnP 解出的是相机朝向，转头时相机随头部转动但机体不动。
    机体朝向 = R(-θ_head) @ 相机朝向
    """
    global current_orientation
    if current_orientation is None:
        return
    theta = np.radians(pulse_to_angle(head_pulse))
    # R(-θ) = [[cosθ, sinθ], [-sinθ, cosθ]]
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R_neg = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
    current_orientation = R_neg @ current_orientation
    norm = np.linalg.norm(current_orientation)
    if norm != 0:
        current_orientation /= norm

def solve_pnp():
    """拍照并执行 AprilTag 检测 + PnP 求解，成功返回 True

    成功时设置 current_position 和 current_orientation（相机坐标系）。
    """
    intrinsic = np.array(
        (
            [1.944903664123011e03, 0, 1.283069051100245e03],
            [0, 1.950095436307893e03, 9.831983420212778e02],
            [0, 0, 1],
        ),
        dtype=np.double,
    )
    distortion = np.array([-0.384402275498781, 0.284681889150075, 0, 0])

    global current_position, current_orientation

    objlist = []
    imglist = []
    filename = capture_image()
    if filename is None:
        return False
    for r in detect_apriltag(filename):
        print("[INFO] Detected AprilTag ID: {}".format(r.tag_id))
        objlist.extend(tag_poses[str(r.tag_id)])
        imglist.extend(r.corners)

    if len(objlist) < 1:
        print("检测到的AprilTag数量不足，无法计算相机位姿。")
        return False
    ifsuccess, rvec, tvec = cv2.solvePnP(
        np.array(objlist, dtype=np.float64),
        np.array(imglist, dtype=np.float64),
        intrinsic,
        distortion,
    )
    if not ifsuccess:
        print("PnP求解失败，无法计算相机位姿。")
        return False

    rotateMatrix = cv2.Rodrigues(rvec)[0]
    pos_3d = -np.linalg.inv(rotateMatrix) @ tvec
    ori_3d = np.linalg.inv(rotateMatrix) @ (np.array([[0], [0], [1]]) - tvec) - pos_3d
    current_position = pos_3d[:2].flatten()
    current_orientation = ori_3d[:2].flatten()
    norm = np.linalg.norm(current_orientation)
    if norm != 0:
        current_orientation /= norm
    return True

def locate_with_scan():
    """三级头部扫描定位：回正→右转→左转，每步拍照+PnP

    成功时 current_position/current_orientation 已补偿为机体坐标系，返回 True。
    失败返回 False，头部回到中位。
    """
    global current_position, current_orientation
    current_position = None
    current_orientation = None

    # 1. 头部回正拍照
    set_head(HEAD_CENTER)
    if solve_pnp():
        print(
            "定位成功（头部回正）。位置：",
            current_position,
            "朝向：",
            current_orientation,
        )
        return True

    # 2. 头部右转拍照
    set_head(HEAD_RIGHT)
    if solve_pnp():
        compensate_head_offset(HEAD_RIGHT)
        print(
            "定位成功（头部右转）。位置：",
            current_position,
            "机体朝向：",
            current_orientation,
        )
        set_head(HEAD_CENTER)
        return True

    # 3. 头部左转拍照
    set_head(HEAD_LEFT)
    if solve_pnp():
        compensate_head_offset(HEAD_LEFT)
        print(
            "定位成功（头部左转）。位置：",
            current_position,
            "机体朝向：",
            current_orientation,
        )
        set_head(HEAD_CENTER)
        return True

    # 全部失败，头部回正
    set_head(HEAD_CENTER)
    print("三级头部扫描均失败。")
    return False

def locate_with_retry():
    """转头定位失败时身体转动重试：头部扫描→交替左右小步转动→有限次后报错退出"""
    for attempt in range(MAX_LOCATE_RETRIES):
        print(f"--- 定位尝试 {attempt + 1}/{MAX_LOCATE_RETRIES} ---")
        if locate_with_scan():
            return True
        # 头部扫描全失败 → 身体小幅转动改变视角
        if attempt % 2 == 0:
            print("头部扫描失败，身体左转小步尝试重新定位。")
            run_action("turn_left_small_step")
        else:
            print("头部扫描失败，身体右转小步尝试重新定位。")
            run_action("turn_right_small_step")
    print("定位重试超限，程序终止。")
    return False

def decide_panning_action(current_pos, target_pos, orientation_xOy):
    """贪心平移策略：模拟各方向动作后的预期位置，选最接近目标的

    current_pos/target_pos: 绝对坐标（2D）。
    对每个候选动作执行后的绝对位置做避障过滤（离墙距离 >= OBSTACLE_THRESHOLD），
    全部被排除时返回 None（信号：机器人已在危险区）。
    前进方向加偏好权重 FORWARD_BIAS。
    """
    position_diff = np.array(current_pos) - np.array(target_pos)
    # 各动作的位移向量（机体坐标系）
    # orientation_xOy 为机体朝向，左转为 [-oy[1], oy[0]]，右转为 [oy[1], -oy[0]]
    oy = orientation_xOy
    left_dir = np.array([-oy[1], oy[0]])
    right_dir = np.array([oy[1], -oy[0]])

    # 模拟各动作后的预期 position_diff（2D）
    candidates = {
        "go_forward_one_step": position_diff + FORWARD_ONE_STEP_CM * oy,
        "go_forward_one_small_step": position_diff + FORWARD_ONE_SMALL_STEP_CM * oy,
        "back_one_step": position_diff - BACK_ONE_STEP_CM * oy,
        "left_move": position_diff + LEFT_MOVE_CM * left_dir,
        "right_move": position_diff + RIGHT_MOVE_CM * right_dir,
    }

    # 计算每个候选动作执行后到目标的距离，前进方向减去偏好权重
    # 避障过滤：执行后绝对位置离墙距离 < 阈值则排除
    best_action = None
    best_score = float("inf")
    for action, new_pd in candidates.items():
        new_pos = np.array(target_pos) + new_pd
        wall_dist = distance_to_walls(new_pos)
        if wall_dist < OBSTACLE_THRESHOLD:
            print(
                f"  {action}: 执行后位置 {new_pos} 离墙 {wall_dist:.2f}cm < {OBSTACLE_THRESHOLD}cm，排除"
            )
            continue
        score = calc_distance(new_pd)
        if action.startswith("go_forward"):
            score -= FORWARD_BIAS
        print(
            f"  {action}: 执行后距离 {calc_distance(new_pd):.2f}cm 离墙 {wall_dist:.2f}cm (score={score:.2f})"
        )
        if score < best_score:
            best_score = score
            best_action = action

    if best_action is None:
        print("所有平移动作均被避障排除，机器人可能已在危险区。")
        return None

    print("最佳平移动作：", best_action)
    return best_action

def decide_rotation_action(orientation_diff):
    """转向策略：计算带符号角度差，小角度差用小步，大角度差用大步

    返回 (action_name, times)。
    """
    target = current_orientation - orientation_diff  # 目标朝向
    # 当前朝向到目标朝向的带符号角度差
    # current_orientation × target 的 z 分量符号决定左/右转
    cross = current_orientation[0] * target[1] - current_orientation[1] * target[0]
    dot = np.dot(current_orientation, target)
    dot = np.clip(dot, -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(dot))

    if cross > 0:
        direction = "left"
        print(f"需左转 {angle_deg:.1f}°")
    else:
        direction = "right"
        print(f"需右转 {angle_deg:.1f}°")

    if direction == "left":
        if angle_deg > TURN_LEFT_DEG:
            return "turn_left", 1
        else:
            return "turn_left_small_step", 1
    else:
        if angle_deg > TURN_RIGHT_DEG:
            return "turn_right", 1
        else:
            return "turn_right_small_step", 1

def navigate_to_target(target_id, poses, stop_time=STOP_TIME):
    """通用导航函数：定位→对准朝向→平移接近→到达检查→按需停留

    poses: 目标位置字典（stand_poses 或 target_poses）。
    stop_time: 到达后停留秒数；0 表示不停留（转向起止点）。
    """
    global next_stop
    next_stop = target_id
    phase = "停靠点" if stop_time > 0 else "转向点"
    print(f"\n===== 开始导航至{phase} {target_id} =====")
    print(
        f"目标坐标：{poses[target_id]}，目标朝向：{target_orientations[target_id]}"
    )

    while True:
        # 1. 定位（含头部扫描+身体转动重试）
        if not locate_with_retry():
            print(f"无法定位，导航至{phase} {target_id} 失败。")
            return False

        print("当前机体位置：", current_position)
        print("当前机体朝向：", current_orientation)

        # 2. 危险检测：离墙距离 < 阈值 → 临时导航至最近安全点
        wall_dist = distance_to_walls(current_position)
        if wall_dist < OBSTACLE_THRESHOLD:
            target = nearest_safe_point(current_position)
            escaping = True
            print(
                f"机器人处于危险区（离墙 {wall_dist:.2f}cm < {OBSTACLE_THRESHOLD}cm），"
                f"临时导航至安全点 {target}"
            )
        else:
            target = poses[next_stop]
            escaping = False

        # 3. 计算差异
        pd, od = calculate_diff(target)
        print("位置差异pd：", pd)
        print("朝向差异od：", od)

        # 4. 朝向优先：od 模 > 阈值 → 旋转修正（逃离时跳过，优先平移离开）
        if not escaping and np.linalg.norm(od) > ORIENTATION_THRESHOLD:
            action, times = decide_rotation_action(od)
            run_action(action, times)
            continue

        # 5. 平移接近：pd 模 > 阈值 → 贪心选最优方向
        if np.linalg.norm(pd) > POSITION_THRESHOLD:
            action = decide_panning_action(current_position, target, current_orientation)
            if action is None:
                # 极端兜底：所有动作被排除，强制后退尝试离开危险区
                print("警告：无安全平移动作可选，强制后退尝试脱离危险区。")
                run_action("back_one_step")
            else:
                run_action(action)
            continue

        # 6. 到达
        if escaping:
            print(f"===== 已到达安全点 {target}，恢复原目标导航 =====")
            continue
        if stop_time > 0:
            print(f"===== 到达{phase} {target_id}，停留 {stop_time} 秒 =====")
            run_action("stand")
            time.sleep(stop_time)
        else:
            print(f"===== 到达{phase} {target_id}，不停留 =====")
        return True

# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# 可更改部分结尾
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

# NOTE: goodluck.py 末尾的主流程自动执行代码已删除，
#       执行入口改由 PART B 的 run_simulation() 控制。


# =====================================================================
# ===== PART B: 模拟器 + 可视化（测试专用）===========================
# =====================================================================

import os
import sys
import matplotlib
# 后端选择：尊重 MPLBACKEND 环境变量；否则优先交互式（TkAgg）支持动画
if not os.environ.get("MPLBACKEND"):
    try:
        matplotlib.use("TkAgg")
    except Exception:
        matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
from datetime import datetime

# 配置中文字体（Windows: Microsoft YaHei / SimHei；缺失则回退默认）
for _font in ["Microsoft YaHei", "SimHei", "WenQuanYi Micro Hei", "Arial Unicode MS"]:
    try:
        matplotlib.rcParams["font.sans-serif"] = [_font] + matplotlib.rcParams["font.sans-serif"]
        break
    except Exception:
        continue
matplotlib.rcParams["axes.unicode_minus"] = False  # 负号显示

# ---------------------------------------------------------------------
# 模拟器配置（噪声开关，默认全 0 = 理想模式）
# ---------------------------------------------------------------------
LOCATE_NOISE_STD = 1.0          # 定位位置噪声标准差（cm），0=无噪声
LOCATE_ANGLE_NOISE_STD = 5.0    # 定位朝向角噪声标准差（度），0=无噪声
ACTION_ERROR_STD = 0.1          # 动作步长误差标准差（比例，0.1=±10%），0=无误差
TURN_ERROR_STD = 5.0            # 转向角度误差标准差（度），0=无误差
ANIM_PAUSE_SEC = 1            # 每步动画刷新间隔（秒）
MAX_SIM_STEPS = 500            # 模拟最大动作步数，防止算法不收敛时无限循环卡死

# 输出目录与文件（result/ 子目录，文件名含日期时间）
_RESULT_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_DIR = "result"
TRAJECTORY_PNG_PATH = os.path.join(RESULT_DIR, f"trajectory_{_RESULT_TIMESTAMP}.png")

# 机器人边界框尺寸（以几何中心为中心）
ROBOT_WIDTH_CM = 26.0          # 机器人边界框宽（cm）
ROBOT_LENGTH_CM = 10.0         # 机器人边界框长（cm）

# 日志输出到文件
LOG_TO_FILE = True             # 是否输出日志到文件
LOG_FILE_PATH = os.path.join(RESULT_DIR, f"simulation_log_{_RESULT_TIMESTAMP}.txt")

# 各动作耗时（秒），用于实时累计完成时间
ACTION_TIME_SEC = {
    "stand": 1.0,
    "go_forward_one_step": 1.0,
    "go_forward_one_small_step": 0.8,
    "go_forward": 1.0,
    "back_one_step": 1.0,
    "back": 1.0,
    "left_move": 1.2,
    "right_move": 1.2,
    "turn_left": 1.5,
    "turn_left_small_step": 0.8,
    "turn_right": 1.5,
    "turn_right_small_step": 0.8,
}

# 机器人初始状态（入口附近，朝东）
INITIAL_POS = np.array([2.0, 20.0], dtype=np.float64)
INITIAL_ORIENTATION = np.array([1.0, 0.0], dtype=np.float64)


class SimState:
    """模拟器状态：维护机器人的真实位置/朝向/轨迹

    所有 apply_* 方法按机体坐标系更新状态，并追加轨迹点。
    """

    def __init__(self, pos, orientation):
        self.pos = np.array(pos, dtype=np.float64).copy()
        self.orientation = np.array(orientation, dtype=np.float64).copy()
        norm = np.linalg.norm(self.orientation)
        if norm != 0:
            self.orientation /= norm
        self.head_pulse = HEAD_CENTER
        self.trajectory = [self.pos.copy()]
        self.last_action = "init"
        self.step_count = 0
        self.locate_count = 0
        self.elapsed_time = 0.0

    def _record(self, action_name):
        self.last_action = action_name
        self.trajectory.append(self.pos.copy())

    def apply_forward(self, cm):
        """沿当前朝向前进 cm 厘米"""
        actual_cm = cm
        if ACTION_ERROR_STD > 0:
            actual_cm = cm * (1.0 + np.random.normal(0, ACTION_ERROR_STD))
        self.pos = self.pos + actual_cm * self.orientation
        self._record("forward")

    def apply_back(self, cm):
        """沿当前朝向后退 cm 厘米"""
        actual_cm = cm
        if ACTION_ERROR_STD > 0:
            actual_cm = cm * (1.0 + np.random.normal(0, ACTION_ERROR_STD))
        self.pos = self.pos - actual_cm * self.orientation
        self._record("back")

    def apply_left_move(self, cm):
        """机体左侧横移 cm 厘米（左转为 [-oy[1], oy[0]]）"""
        actual_cm = cm
        if ACTION_ERROR_STD > 0:
            actual_cm = cm * (1.0 + np.random.normal(0, ACTION_ERROR_STD))
        left_dir = np.array([-self.orientation[1], self.orientation[0]])
        self.pos = self.pos + actual_cm * left_dir
        self._record("left_move")

    def apply_right_move(self, cm):
        """机体右侧横移 cm 厘米（右转为 [oy[1], -oy[0]]）"""
        actual_cm = cm
        if ACTION_ERROR_STD > 0:
            actual_cm = cm * (1.0 + np.random.normal(0, ACTION_ERROR_STD))
        right_dir = np.array([self.orientation[1], -self.orientation[0]])
        self.pos = self.pos + actual_cm * right_dir
        self._record("right_move")

    def apply_turn(self, deg):
        """圆周运动转向：机体绕旋转中心做圆弧运动（正=左转，负=右转）

        旋转中心 = 机体位置 - d·朝向 + R·侧向方向
          左转：侧向 = left_dir = [-oy, ox]，圆心在左后方
          右转：侧向 = right_dir = [oy, -ox]，圆心在右后方
        机体绕圆心旋转 α 度（左转正、右转负），位置和朝向同步更新。
        """
        actual_deg = deg
        if TURN_ERROR_STD > 0:
            actual_deg = deg + np.random.normal(0, TURN_ERROR_STD)
        alpha = np.radians(actual_deg)
        oy = self.orientation
        # 旋转中心：先向后退 d·O，再侧向偏移 R
        base = self.pos - CAMERA_FORWARD_OFFSET_CM * oy
        if actual_deg >= 0:  # 左转
            left_dir = np.array([-oy[1], oy[0]])
            center = base + TURN_LEFT_RADIUS_CM * left_dir
        else:  # 右转
            right_dir = np.array([oy[1], -oy[0]])
            center = base + TURN_RIGHT_RADIUS_CM * right_dir
        # 旋转矩阵 R(α) = [[cos, -sin], [sin, cos]]
        cos_a, sin_a = np.cos(alpha), np.sin(alpha)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        # P' = K + R·(P - K)，O' = R·O
        self.pos = center + R @ (self.pos - center)
        self.orientation = R @ self.orientation
        norm = np.linalg.norm(self.orientation)
        if norm != 0:
            self.orientation /= norm
        self._record("turn")


# 全局模拟器实例（run_simulation 中重新初始化）
sim = SimState(INITIAL_POS, INITIAL_ORIENTATION)


class Visualizer:
    """matplotlib 可视化：绘制赛道、目标点、机器人箭头、历史轨迹

    静态层（赛道/墙壁/目标点）在 __init__ 绘制一次；
    动态层（机器人箭头/轨迹/动作文字）在 update() 每步重绘。
    """

    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self._draw_static()
        # 动态层句柄
        self.robot_arrow = None
        self.robot_box = None
        self.traj_line = None
        self.action_text = None
        self.fig.canvas.manager.set_window_title("寻路算法模拟器")

    def _draw_static(self):
        ax = self.ax
        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 105)
        ax.set_aspect("equal")
        ax.set_title("RoboTrack 寻路算法模拟", fontsize=13)
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")
        ax.grid(True, linestyle="--", alpha=0.3)

        # 外框
        ax.add_patch(Rectangle((0, 0), 100, 100, fill=False, edgecolor="black", linewidth=2))

        # 墙壁
        for rect in WALLS:
            x_min, x_max, y_min, y_max = rect
            ax.add_patch(Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                facecolor="gray", edgecolor="black", alpha=0.5, hatch="//",
            ))

        # 停靠点（stand_poses）—— 蓝色方块
        for tid, p in stand_poses.items():
            ax.plot(p[0], p[1], "bs", markersize=9, markeredgecolor="black")
            ax.annotate(f"停{tid}", (p[0], p[1]), textcoords="offset points",
                        xytext=(6, 6), fontsize=8, color="blue")

        # 转向点（target_poses）—— 红色圆点
        for tid, p in target_poses.items():
            ax.plot(p[0], p[1], "ro", markersize=8, markeredgecolor="black")
            ax.annotate(f"转{tid}", (p[0], p[1]), textcoords="offset points",
                        xytext=(6, -10), fontsize=8, color="red")

        # AprilTag 位置（取各 tag 第一点近似）—— 绿色三角
        for tid, pts in tag_poses.items():
            p = pts[0][:2]
            ax.plot(p[0], p[1], "g^", markersize=7)
            ax.annotate(f"Tag{tid}", (p[0], p[1]), textcoords="offset points",
                        xytext=(6, 6), fontsize=7, color="green")

        # 入口/出口标注
        ax.annotate("入口", (0, 20), textcoords="offset points",
                    xytext=(-30, 0), fontsize=9, color="purple")
        ax.annotate("出口", (100, 20), textcoords="offset points",
                    xytext=(8, 0), fontsize=9, color="purple")

    def update(self, sim_state, action_text=""):
        """重绘动态层并刷新"""
        ax = self.ax
        # 移除旧的动态元素
        for artist in [self.robot_arrow, self.robot_box, self.traj_line, self.action_text]:
            if artist is not None:
                artist.remove()
        # 历史轨迹
        traj = np.array(sim_state.trajectory)
        self.traj_line, = ax.plot(traj[:, 0], traj[:, 1], "c-", linewidth=1.5, alpha=0.7)
        # 机器人位置箭头
        pos = sim_state.pos
        ori = sim_state.orientation
        # 机器人边界框：以 pos 为中心，长边沿朝向旋转
        angle_deg = np.degrees(np.arctan2(ori[1], ori[0]))
        # 矩形左下角（未旋转时）：以中心为原点偏移
        box_x = pos[0] - ROBOT_LENGTH_CM / 2
        box_y = pos[1] - ROBOT_WIDTH_CM / 2
        t = Affine2D().rotate_deg_around(pos[0], pos[1], angle_deg) + ax.transData
        self.robot_box = Rectangle(
            (box_x, box_y), ROBOT_LENGTH_CM, ROBOT_WIDTH_CM,
            facecolor=(1.0, 0.6, 0.6, 0.3), edgecolor="red", linewidth=1.2,
        )
        self.robot_box.set_transform(t)
        ax.add_patch(self.robot_box)
        self.robot_arrow = ax.annotate(
            "",
            xy=(pos[0] + ori[0] * 4, pos[1] + ori[1] * 4),
            xytext=(pos[0], pos[1]),
            arrowprops=dict(arrowstyle="->", color="magenta", lw=2.5),
        )
        ax.plot(pos[0], pos[1], "mo", markersize=6)
        # 动作文字
        self.action_text = ax.text(
            0.02, 0.98, action_text, transform=ax.transAxes,
            fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )
        # 交互后端用 pause 触发重绘；非交互后端用 draw
        if ANIM_PAUSE_SEC > 0 and matplotlib.get_backend().lower() != "agg":
            plt.pause(ANIM_PAUSE_SEC)
        else:
            self.fig.canvas.draw_idle()


# 全局可视化实例
viz = None


# ---------------------------------------------------------------------
# 桩函数（monkey-patch 目标）：替换 PART A 的 I/O 函数
# ---------------------------------------------------------------------

def sim_solve_pnp():
    """桩：直接返回模拟器真实状态（可注入噪声），不拍照不检测

    同时兼作死循环检测点：每次 navigate_to_target 循环都会调用定位，
    若定位次数超限则抛异常中止，防止算法逻辑死循环（不执行动作时
    MAX_SIM_STEPS 无法触发）。
    """
    global current_position, current_orientation
    sim.locate_count += 1
    if sim.locate_count > MAX_SIM_STEPS * 3:
        raise RuntimeError(
            f"定位次数超限({sim.locate_count})，算法可能陷入死循环。中止以防卡死。"
        )
    pos = sim.pos.copy()
    ori = sim.orientation.copy()
    # 注入定位噪声
    if LOCATE_NOISE_STD > 0:
        pos = pos + np.random.normal(0, LOCATE_NOISE_STD, size=2)
    if LOCATE_ANGLE_NOISE_STD > 0:
        ang = np.radians(np.random.normal(0, LOCATE_ANGLE_NOISE_STD))
        c, s = np.cos(ang), np.sin(ang)
        R = np.array([[c, -s], [s, c]])
        ori = R @ ori
        n = np.linalg.norm(ori)
        if n != 0:
            ori /= n
    current_position = pos
    current_orientation = ori
    return True


def sim_set_head(pulse, move_time_ms=HEAD_MOVE_TIME_MS):
    """桩：只更新头部脉宽记录，不调舵机"""
    global current_head_pulse
    sim.head_pulse = pulse
    current_head_pulse = pulse


def sim_run_action(name, times=1):
    """桩：解析动作名，更新模拟器状态，并刷新可视化"""
    for _ in range(times):
        if sim.step_count >= MAX_SIM_STEPS:
            raise RuntimeError(
                f"模拟步数超限({MAX_SIM_STEPS})，算法可能不收敛。中止以防卡死。"
            )
        sim.step_count += 1
        if name == "stand":
            sim._record("stand")
        elif name == "go_forward_one_step":
            sim.apply_forward(FORWARD_ONE_STEP_CM)
        elif name == "go_forward_one_small_step":
            sim.apply_forward(FORWARD_ONE_SMALL_STEP_CM)
        elif name == "go_forward":
            # 连续前进：按一步常量模拟
            sim.apply_forward(FORWARD_ONE_STEP_CM)
        elif name == "back_one_step":
            sim.apply_back(BACK_ONE_STEP_CM)
        elif name == "back":
            sim.apply_back(BACK_ONE_STEP_CM)
        elif name == "left_move":
            sim.apply_left_move(LEFT_MOVE_CM)
        elif name == "right_move":
            sim.apply_right_move(RIGHT_MOVE_CM)
        elif name == "turn_left":
            sim.apply_turn(TURN_LEFT_DEG)
        elif name == "turn_left_small_step":
            sim.apply_turn(TURN_LEFT_SMALL_STEP_DEG)
        elif name == "turn_right":
            sim.apply_turn(-TURN_RIGHT_DEG)
        elif name == "turn_right_small_step":
            sim.apply_turn(-TURN_RIGHT_SMALL_STEP_DEG)
        else:
            print(f"[sim] 未知动作，忽略: {name}")
            sim._record(name)
        # 累加动作耗时
        sim.elapsed_time += ACTION_TIME_SEC.get(name, 1.0)
        if viz is not None:
            viz.update(sim, f"动作: {name} (第{len(sim.trajectory)}步)\n"
                            f"位置: ({sim.pos[0]:.1f}, {sim.pos[1]:.1f})  "
                            f"朝向: ({sim.orientation[0]:.2f}, {sim.orientation[1]:.2f})\n"
                            f"已用时间: {sim.elapsed_time:.1f}s")


def save_trajectory_png(path=TRAJECTORY_PNG_PATH):
    """保存当前 matplotlib 图为 PNG（含完整赛道+最终轨迹+终点姿态）"""
    if viz is None:
        print("[save_trajectory_png] 可视化未初始化，跳过保存。")
        return
    viz.fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[save_trajectory_png] 轨迹图已保存: {path}")


class TeeWriter:
    """将 stdout 同时写入终端和文件"""

    def __init__(self, file_path, original_stdout):
        self.file = open(file_path, "w", encoding="utf-8")
        self.stdout = original_stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def run_simulation():
    """主模拟流程：初始化 → monkey-patch → 执行原主流程 → 保存轨迹图"""
    global sim, viz

    # 确保输出目录存在
    os.makedirs(RESULT_DIR, exist_ok=True)

    # 日志 tee：同时输出到终端和文件
    tee = None
    original_stdout = sys.stdout
    if LOG_TO_FILE:
        tee = TeeWriter(LOG_FILE_PATH, original_stdout)
        sys.stdout = tee
        print(f"[log] 日志同时输出到文件: {LOG_FILE_PATH}")

    print("=" * 60)
    print("寻路算法模拟器启动")
    print(f"噪声配置: 定位位置σ={LOCATE_NOISE_STD}cm, 定位朝向σ={LOCATE_ANGLE_NOISE_STD}°, "
          f"动作步长σ={ACTION_ERROR_STD}, 转向σ={TURN_ERROR_STD}°")
    print("=" * 60)

    # 1. 初始化模拟器状态与可视化
    sim = SimState(INITIAL_POS, INITIAL_ORIENTATION)
    viz = Visualizer()
    viz.update(sim, "模拟器就绪\n等待启动...")

    # 2. monkey-patch：用桩函数替换 PART A 的 I/O 函数
    #    算法函数内部调用 run_action()/solve_pnp()/set_head() 时，
    #    Python 按模块全局命名空间解析，命中此处赋值的桩函数。
    g = globals()
    g["run_action"] = sim_run_action
    g["solve_pnp"] = sim_solve_pnp
    g["set_head"] = sim_set_head
    print("[patch] 已替换 I/O 函数: run_action / solve_pnp / set_head")

    try:
        # 3. 执行原主流程逻辑（与 goodluck.py 末尾一致）
        run_action("stand")
        set_head(HEAD_CENTER)

        for tid in ["1", "2", "3", "4"]:
            # 停靠阶段：到达 stand_poses，停留 3 秒
            if not navigate_to_target(tid, stand_poses, STOP_TIME):
                print(f"导航至停靠点 {tid} 失败，程序终止。")
                break
            print(f"已到达停靠点 {tid}。")

            # 准备转向阶段：到达 target_poses，不停留
            if not navigate_to_target(tid, target_poses, 0):
                print(f"导航至转向点 {tid} 失败，程序终止。")
                break
            print(f"已到达转向点 {tid}。")

        # 第5点：开环走出出口
        print("\n===== 到达第4个转向点，准备开环走出出口 =====")
        run_action("go_forward", times=3)
        run_action("turn_left", times=3)
        run_action("go_forward", times=6)
        run_action("stand")
        print("===== 全程完成 =====")

    except Exception as e:
        print(f"[run_simulation] 模拟过程异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 4. 保存最终轨迹图
        save_trajectory_png()
        print(f"\n总耗时: {sim.elapsed_time:.1f}s  总动作步数: {sim.step_count}")
        print("\n模拟结束。")
        # 恢复 stdout
        if tee is not None:
            sys.stdout = original_stdout
            tee.close()
            print(f"[log] 日志已保存: {LOG_FILE_PATH}")
        # 交互后端保持窗口；非交互后端直接退出
        if matplotlib.get_backend().lower() != "agg":
            print("关闭图形窗口退出。")
            plt.show()


if __name__ == "__main__":
    run_simulation()
