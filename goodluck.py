import time
import hiwonder.ActionGroupControl as AGC    #动作库，必须包含该库
import subprocess   #拍照
import apriltag
import cv2
import numpy as np
import hiwonder.ros_robot_controller_sdk as rrc
from hiwonder.Controller import Controller
board = rrc.Board()
ctl = Controller(board)

# =====================================================================
# 决策算法常量
# =====================================================================
ORIENTATION_THRESHOLD = 0.19    # 朝向差异模长阈值，约11°，是2sin(11°/2)的值
POSITION_THRESHOLD = 3.0        # 位置差异模长阈值，单位cm
STOP_TIME = 3                    # 到达目标点后停留时间，单位秒
MAX_LOCATE_RETRIES = 5           # 定位失败最大重试次数
PANNING_ANGLE_THRESHOLD = 30.0  # 朝向到目标点夹角阈值，单位度

# =====================================================================
# 动作组参数常量（所有的数值都需要重新标定！！！）
# =====================================================================
FORWARD_ONE_STEP_CM = 4.0    # go_forward_one_step（待标定）
FORWARD_ONE_SMALL_STEP_CM = 0.375  # go_forward_one_small_step（待标定）
BACK_ONE_STEP_CM = 4.0       # back_one_step（待标定）
LEFT_MOVE_CM = 2.9           # left_move（待标定）
RIGHT_MOVE_CM = 2.1          # right_move（待标定）
TURN_LEFT_SMALL_STEP_DEG = 21.0   # turn_left_small_step（待标定）
TURN_RIGHT_SMALL_STEP_DEG = 21.0  # turn_right_small_step（待标定）
TURN_LEFT_DEG = 30.0             # turn_left（估算值，待标定）
TURN_RIGHT_DEG = 30.0            # turn_right（估算值，待标定）
FORWARD_BIAS = 0.5           # 前进方向偏好权重，避免原地转圈

# =====================================================================
# 头部舵机参数常量
# =====================================================================
HEAD_CENTER = 1500
HEAD_RIGHT = 600
HEAD_LEFT = 1700
HEAD_MOVE_TIME_MS = 500
# 舵机脉宽→角度线性映射：angle_deg = (pulse - 1500) * SERVO_DEG_PER_US
# 标准500-2500μs对应±90°，即 90°/1000μs = 0.09
SERVO_DEG_PER_US = 0.09

# =====================================================================
# 赛道数据
# =====================================================================
#这里是定义每个AprilTag的世界坐标，单位为厘米。
#添加AprilTag的世界坐标，四个点的顺序为左上，右上，右下，左下。也即从左上角开始顺时针。

tag_poses = {}
tag_poses['36'] = np.array([[44.7,23.8,41.5], [44.7, 18.8, 41.5], [44.7, 18.8, 36.5], [44.7, 23.8, 36.5]], dtype=np.float64)
tag_poses['37'] = np.array([[20.6,100,41.7], [25.6, 100, 41.7], [25.6, 100, 36.7], [20.6, 100, 36.7]], dtype=np.float64)
tag_poses['38'] = np.array([[95, 82.2,41.7], [95, 77.2, 41.7], [95, 77.2, 36.7], [95, 82.2, 36.7]], dtype=np.float64)
tag_poses['39'] = np.array([[76.5, 0, 42.1], [71.5, 0, 42.1], [71.5, 0, 37.1], [76.5, 0, 37.1]], dtype=np.float64)

#以下定义机器人定点站立课目的目标点坐标和朝向。
centre_poses = {}
centre_poses['1'] = np.array([14.7,21.3], dtype=np.float64)
centre_poses['2'] = np.array([23.1, 70], dtype=np.float64)
centre_poses['3'] = np.array([65,79.7], dtype=np.float64)
centre_poses['4'] = np.array([74, 30], dtype=np.float64)
target_orientations = {}
target_orientations['1'] = np.array([1,0], dtype=np.float64)
target_orientations['2'] = np.array([0,1], dtype=np.float64)
target_orientations['3'] = np.array([1,0], dtype=np.float64)
target_orientations['4'] = np.array([0,-1], dtype=np.float64)
target_orientations['5'] = np.array([1,0], dtype=np.float64)

current_position = None
current_orientation = None
next_stop = '0'

def run_action(name, times=1):
    """执行动作组"""
    AGC.runActionGroup(name, times=times)

def calculate_diff():
    """计算当前位置/朝向与目标点的差异"""
    position_diff = np.array(current_position) - np.array(centre_poses[next_stop])
    orientation_diff = np.array(current_orientation) - np.array(target_orientations[next_stop])
    return position_diff, orientation_diff

def calc_distance(position_diff):
    """计算水平距离"""
    return np.linalg.norm(position_diff)

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
    """转动头部舵机并等待到位"""
    ctl.set_pwm_servo_pulse(2, pulse, move_time_ms)
    time.sleep(move_time_ms / 1000.0 + 0.2)

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
    R_neg = np.array([[cos_t, sin_t],
                      [-sin_t, cos_t]])
    current_orientation = R_neg @ current_orientation
    norm = np.linalg.norm(current_orientation)
    if norm != 0:
        current_orientation /= norm

def solve_pnp():
    """拍照并执行 AprilTag 检测 + PnP 求解，成功返回 True

    成功时设置 current_position 和 current_orientation（相机坐标系）。
    """
    intrinsic = np.array(([1.944903664123011e+03, 0, 1.283069051100245e+03],
                          [0, 1.950095436307893e+03, 9.831983420212778e+02],
                          [0, 0, 1]), dtype=np.double)
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
        intrinsic, distortion)
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
        print("定位成功（头部回正）。位置：", current_position, "朝向：", current_orientation)
        return True

    # 2. 头部右转拍照
    set_head(HEAD_RIGHT)
    if solve_pnp():
        compensate_head_offset(HEAD_RIGHT)
        print("定位成功（头部右转）。位置：", current_position, "机体朝向：", current_orientation)
        set_head(HEAD_CENTER)
        return True

    # 3. 头部左转拍照
    set_head(HEAD_LEFT)
    if solve_pnp():
        compensate_head_offset(HEAD_LEFT)
        print("定位成功（头部左转）。位置：", current_position, "机体朝向：", current_orientation)
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
            run_action('turn_left_small_step')
        else:
            print("头部扫描失败，身体右转小步尝试重新定位。")
            run_action('turn_right_small_step')
    print("定位重试超限，程序终止。")
    return False

def decide_panning_action(position_diff, orientation_xOy):
    """贪心平移策略：模拟各方向动作后的预期位置，选最接近目标的

    恢复后退和侧移动作，避免纯前进在朝向偏差略大于阈值时陷入
    旋转-过冲-反向旋转的原地转圈。前进方向加偏好权重 FORWARD_BIAS。
    """
    # 各动作的位移向量（机体坐标系）
    # orientation_xOy 为机体朝向，左转为 [-oy[1], oy[0]]，右转为 [oy[1], -oy[0]]
    oy = orientation_xOy
    left_dir = np.array([-oy[1], oy[0]])
    right_dir = np.array([oy[1], -oy[0]])

    # 模拟各动作后的预期 position_diff（2D）
    candidates = {
        'go_forward_one_step':       position_diff - FORWARD_ONE_STEP_CM * oy,
        'go_forward_one_small_step': position_diff - FORWARD_ONE_SMALL_STEP_CM * oy,
        'back_one_step':             position_diff + BACK_ONE_STEP_CM * oy,
        'left_move':                 position_diff - LEFT_MOVE_CM * left_dir,
        'right_move':                position_diff - RIGHT_MOVE_CM * right_dir,
    }

    # 计算每个候选动作执行后到目标的距离，前进方向减去偏好权重
    best_action = None
    best_score = float('inf')
    for action, new_pd in candidates.items():
        score = calc_distance(new_pd)
        if action.startswith('go_forward'):
            score -= FORWARD_BIAS
        print(f"  {action}: 执行后距离 {calc_distance(new_pd):.2f}cm (score={score:.2f})")
        if score < best_score:
            best_score = score
            best_action = action

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
        direction = 'left'
        print(f"需左转 {angle_deg:.1f}°")
    else:
        direction = 'right'
        print(f"需右转 {angle_deg:.1f}°")

    if direction == 'left':
        if angle_deg > TURN_LEFT_DEG:
            return 'turn_left', 1
        else:
            return 'turn_left_small_step', 1
    else:
        if angle_deg > TURN_RIGHT_DEG:
            return 'turn_right', 1
        else:
            return 'turn_right_small_step', 1

def navigate_to_target(target_id):
    """通用导航函数：定位→对准朝向→平移接近→到达检查→停留3秒"""
    global next_stop
    next_stop = target_id
    print(f"\n===== 开始导航至目标点 {target_id} =====")
    print(f"目标坐标：{centre_poses[target_id]}，目标朝向：{target_orientations[target_id]}")

    while True:
        # 1. 定位（含头部扫描+身体转动重试）
        if not locate_with_retry():
            print(f"无法定位，导航至目标点 {target_id} 失败。")
            return False

        print("当前机体位置：", current_position)
        print("当前机体朝向：", current_orientation)

        # 2. 计算差异
        pd, od = calculate_diff()
        print("位置差异pd：", pd)
        print("朝向差异od：", od)

        # 3. 朝向优先：od 模 > 阈值 → 旋转修正
        if np.linalg.norm(od) > ORIENTATION_THRESHOLD:
            action, times = decide_rotation_action(od)
            run_action(action, times)
            continue

        # 4. 平移接近：pd 模 > 阈值 → 贪心选最优方向
        if np.linalg.norm(pd) > POSITION_THRESHOLD:
            action = decide_panning_action(pd, current_orientation)
            run_action(action)
            continue

        # 5. 到达
        print(f"===== 到达目标点 {target_id}，停留 {STOP_TIME} 秒 =====")
        run_action('stand')
        time.sleep(STOP_TIME)
        return True

# =====================================================================
# 主流程
# =====================================================================
run_action('stand')
set_head(HEAD_CENTER)

for tid in ['1', '2', '3', '4']:
    if not navigate_to_target(tid):
        print(f"导航至目标点 {tid} 失败，程序终止。")
        break
    print(f"已到达目标点 {tid}。")

# 第5点：开环走出出口
print("\n===== 到达第4个目标点，准备开环走出出口 =====")
run_action('turn_left', times=4)
run_action('go_forward', times=3)
run_action('stand')
print("===== 全程完成 =====")