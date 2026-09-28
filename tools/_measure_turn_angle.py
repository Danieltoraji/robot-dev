# -*- coding: utf-8 -*-
"""在机器人上实测 turn_left / turn_right / 小步动作的单次真实转角。

方法：读取 IMU 陀螺仪 z 轴角速度(gz, 单位 deg/s)，在动作执行期间后台线程
高频采样并积分，得到净 yaw 变化。先测 2 秒静止漂移用于扣除零偏。

用法（远程）：
  exec_on_robot.py --code "$(cat tools/_measure_turn_angle.py 的正文)"
"""
import time
import threading

import hiwonder.ros_robot_controller_sdk as rrc
import hiwonder.ActionGroupControl as AGC

board = rrc.Board()
board.enable_reception(True)
time.sleep(2.0)

# gz 在 imu 元组中的下标（6 轴：ax,ay,az,gx,gy,gz）
GZ_INDEX = 5


def _read_gz():
    imu = board.get_imu()
    if imu is None or len(imu) <= GZ_INDEX:
        return None
    return float(imu[GZ_INDEX])


def _integrate_gz(seconds):
    """静止积分 seconds 秒，返回 (平均角速度 deg/s, 累计角度 deg)。"""
    samples = []
    t0 = time.time()
    while time.time() - t0 < seconds:
        gz = _read_gz()
        if gz is not None:
            samples.append((time.time() - t0, gz))
        time.sleep(0.005)
    if len(samples) < 2:
        return 0.0, 0.0
    total = 0.0
    for i in range(1, len(samples)):
        dt = samples[i][0] - samples[i - 1][0]
        total += 0.5 * (samples[i][1] + samples[i - 1][1]) * dt
    mean = total / (samples[-1][0] - samples[0][0]) if samples[-1][0] > samples[0][0] else 0.0
    return mean, total


def measure_action(name, settle=1.0):
    """执行一次动作并返回净 yaw 角（已扣除零偏）。"""
    samples = []
    stop = threading.Event()

    def sampler():
        t0 = time.time()
        while not stop.is_set():
            gz = _read_gz()
            if gz is not None:
                samples.append((time.time() - t0, gz))
            time.sleep(0.005)

    th = threading.Thread(target=sampler)
    th.start()
    time.sleep(0.3)  # 让采样先跑起来
    AGC.runActionGroup(name, times=1, with_stand=True)
    time.sleep(0.5)  # 等动作收尾
    stop.set()
    th.join()

    if len(samples) < 2:
        print('%-22s no samples' % name)
        return None

    total = 0.0
    for i in range(1, len(samples)):
        dt = samples[i][0] - samples[i - 1][0]
        total += 0.5 * (samples[i][1] + samples[i - 1][1]) * dt

    dur = samples[-1][0] - samples[0][0]
    # 扣除静止零偏
    total -= BIAS * dur
    print('%-22s net_yaw=%+7.2f deg  dur=%.2fs  samples=%d'
          % (name, total, dur, len(samples)))
    return total


# 1) 测静止零偏
BIAS, bias_deg = _integrate_gz(2.0)
print('静止零偏 gz_mean=%+.4f deg/s (2s 累计 %+.2f deg)' % (BIAS, bias_deg))

# 2) 逐个动作测 3 次取平均
for name in ['turn_left', 'turn_right', 'turn_left_small_step', 'turn_right_small_step']:
    vals = []
    for i in range(3):
        v = measure_action(name)
        if v is not None:
            vals.append(v)
        time.sleep(1.2)
    if vals:
        avg = sum(vals) / len(vals)
        print('==> %-22s avg=%+7.2f deg  (%s)' % (name, avg, ', '.join('%+.1f' % v for v in vals)))
    print('---')

print('done')
