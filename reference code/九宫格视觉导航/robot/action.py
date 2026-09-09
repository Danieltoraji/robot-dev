import time
import hiwonder.ActionGroupControl as AGC    #动作库，必须包含该库
import subprocess   #拍照

import hiwonder.ros_robot_controller_sdk as rrc
from hiwonder.Controller import Controller

board = rrc.Board()
ctl = Controller(board)  

# ---------------- 云台参数 ----------------

YAW_SERVO = 2
PITCH_SERVO = 1

YAW_CENTER = 1500


# ---------------- 云台俯仰 ----------------

# 导航姿态（略微下俯，保证转弯时不会拍到机器人自己）
LOOK_FORWARD_PULSE = 1200      # TODO：现场标定

# 检测姿态（最低头）
LOOK_DOWN_PULSE = 1000

CURRENT_YAW = 0
def over_hurdle():
    AGC.runActionGroup('climb_stairs',times=1,with_stand = True)
    AGC.runActionGroup('back_one_step', times=2, with_stand=False)
def move_forward():
    AGC.runActionGroup('go_forward_one_step', times=5, with_stand=False)
    #AGC.runActionGroup('stand')
def move_backward_one():
    AGC.runActionGroup('back_one_step', times=1, with_stand=False)
def move_forward_1():
    AGC.runActionGroup('go_forward', times=3, with_stand=True)
    #AGC.runActionGroup('stand')
def move_forward_fast():
    AGC.runActionGroup('go_forward_fast', times=3, with_stand=True) 
    #AGC.runActionGroup('stand')
def move_forward_small():
    """
    小步前进。

    接近目标后使用，提高停车精度。
    """
    AGC.runActionGroup('go_forward_one_step', times=2, with_stand=True)
    #AGC.runActionGroup('stand')

def turn_small_angle_right():
    AGC.runActionGroup('turn_right_small_step',times=1)
    #AGC.runActionGroup('stand')

def turn_small_angle_left():
    AGC.runActionGroup('turn_left_small_step',times=1)
    #AGC.runActionGroup('stand')

def turn_big_angle_left():
    AGC.runActionGroup('turn_left',times=2)
    #AGC.runActionGroup('stand')
    
def turn_big_angle_right():
    AGC.runActionGroup('turn_right',times=2)
    #AGC.runActionGroup('stand')

def angle_to_pulse(angle):
    if(angle <= 0):
        pulse = 1500 + 10 * angle
    else:
        pulse = 1500 + 10 * angle
    
    return pulse

MAX_YAW = 90

def turn_to_angle(angle):
    global CURRENT_YAW

    angle = max(-MAX_YAW, min(MAX_YAW, angle))

    pulse = angle_to_pulse(angle)

    ctl.set_pwm_servo_pulse(
        YAW_SERVO,
        pulse,
        500
    )

    CURRENT_YAW = angle

    time.sleep(0.5)

def servo_look_forward():
    """
    导航姿态。

    用于：
        1. YOLO识别数字
        2. 转向
        3. 搜索目标
    """

    ctl.set_pwm_servo_pulse(
        PITCH_SERVO,
        LOOK_FORWARD_PULSE,
        500
    )

    time.sleep(0.1)


def servo_look_down():
    """
    检测姿态。

    用于颜色识别。
    """

    ctl.set_pwm_servo_pulse(
        PITCH_SERVO,
        LOOK_DOWN_PULSE,
        500
    )

    time.sleep(0.5)

def reset_servo():
    """
    恢复导航姿态。

    当前策略：
        水平回正；
        云台恢复导航俯角。
    """

    global CURRENT_YAW

    ctl.set_pwm_servo_pulse(
        YAW_SERVO,
        YAW_CENTER,
        500
    )

    ctl.set_pwm_servo_pulse(
        PITCH_SERVO,
        LOOK_FORWARD_PULSE,
        500
    )

    CURRENT_YAW = 0

    time.sleep(0.5)
    
def init_servo():
    """
    比赛开始调用一次。

    初始时为导航姿态。
    """

    ctl.set_pwm_servo_pulse(
        PITCH_SERVO,
        LOOK_FORWARD_PULSE,
        500
    )

    ctl.set_pwm_servo_pulse(
        YAW_SERVO,
        YAW_CENTER,
        500
    )

    time.sleep(0.5)


