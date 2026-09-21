#!/usr/bin/python3
# coding=utf8
import os
import sys
import time
import threading
from pathlib import Path

import cv2
import hiwonder.ActionGroupControl as AGC
import hiwonder.Camera as Camera
import hiwonder.ros_robot_controller_sdk as rrc
from hiwonder.Controller import Controller

# ---- 终端按键输入（替代 cv2.imshow / waitKey 的 GUI 依赖） ----
import termios
import select

_stdin_fd = sys.stdin.fileno()
_old_term = None


def _setup_terminal():
    """设置为非规范模式：逐字符读取，不回显，保留输出处理"""
    global _old_term
    _old_term = termios.tcgetattr(_stdin_fd)
    new = termios.tcgetattr(_stdin_fd)
    # 关闭规范模式（不需要回车）和回显，保持输出处理正常
    new[3] &= ~(termios.ICANON | termios.ECHO)
    termios.tcsetattr(_stdin_fd, termios.TCSADRAIN, new)


def _restore_terminal():
    """恢复终端原始设置"""
    global _old_term
    if _old_term is not None:
        termios.tcsetattr(_stdin_fd, termios.TCSADRAIN, _old_term)


def _get_key():
    """非阻塞读取一个按键，无输入返回 None"""
    if select.select([sys.stdin], [], [], 0.01)[0]:
        try:
            return sys.stdin.read(1)
        except (IOError, OSError):
            return None
    return None
# ---------------------------------------------------------

PHOTO_DIR = Path(__file__).resolve().parent / "debug_vision_pic"
HEAD_CENTER_X = 1500
HEAD_CENTER_Y = 1500
HEAD_MIN_X = 500
HEAD_MAX_X = 2500
HEAD_MIN_Y = 1000
HEAD_MAX_Y = 2000
HEAD_STEP = 100
HEAD_MOVE_TIME_MS = 300

ACTION_MAP = {
    "w": ("go_forward", 1, False, "前进"),
    "s": ("back_fast", 1, False, "后退"),
    "a": ("left_move_fast", 1, False, "左移"),
    "d": ("right_move_fast", 1, False, "右移"),
    "q": ("turn_left", 1, True, "左转"),
    "e": ("turn_right", 1, True, "右转"),
}

board = rrc.Board()
ctl = Controller(board)
head_x = HEAD_CENTER_X
head_y = HEAD_CENTER_Y
action_thread = None


def print_help():
    print("""
cmd_keyboard.py 键盘控制说明（终端模式，无 GUI）：
  p       拍照，保存到 debug_vision_pic 文件夹
  w / s   前进 / 后退
  a / d   左移 / 右移
  q / e   左转 / 右转
  u / l   头部向上 / 头部向下
  空格    停止动作，身体和头部回到默认值
  Esc     退出程序

直接在终端按对应按键即可控制。
""")


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def set_head(x, y):
    ctl.set_pwm_servo_pulse(1, y, HEAD_MOVE_TIME_MS)
    ctl.set_pwm_servo_pulse(2, x, HEAD_MOVE_TIME_MS)


def reset_robot():
    global head_x, head_y
    AGC.stopActionGroup()
    head_x = HEAD_CENTER_X
    head_y = HEAD_CENTER_Y
    set_head(head_x, head_y)
    AGC.runActionGroup("stand_slow")
    print("已停止动作并回到默认值")


def move_head(direction):
    global head_y
    if direction == "up":
        head_y = clamp(head_y - HEAD_STEP, HEAD_MIN_Y, HEAD_MAX_Y)
        print("头部向上")
    elif direction == "down":
        head_y = clamp(head_y + HEAD_STEP, HEAD_MIN_Y, HEAD_MAX_Y)
        print("头部向下")
    set_head(head_x, head_y)


def run_action(action_name, times, with_stand):
    global action_thread
    if action_thread is not None and action_thread.is_alive():
        print("当前动作还未完成，请稍后再按键")
        return
    action_thread = threading.Thread(
        target=AGC.runActionGroup,
        args=(action_name, times, with_stand),
        daemon=True,
    )
    action_thread.start()


def save_photo(frame):
    if frame is None:
        print("当前没有摄像头画面，拍照失败")
        return

    PHOTO_DIR.mkdir(parents=True, exist_ok=True)
    filename = PHOTO_DIR / time.strftime("photo_%Y%m%d_%H%M%S.jpg")
    if cv2.imwrite(str(filename), frame):
        print(f"照片已保存: {filename}")
    else:
        print("照片保存失败")


def handle_key(key, frame):
    if key == ord("p"):
        save_photo(frame)
    elif key in (ord("w"), ord("s"), ord("a"), ord("d"), ord("q"), ord("e")):
        action_name, times, with_stand, desc = ACTION_MAP[chr(key)]
        print(desc)
        run_action(action_name, times, with_stand)
    elif key == ord("u"):
        move_head("up")
    elif key == ord("l"):
        move_head("down")
    elif key == 32:
        reset_robot()


def main():
    print_help()
    set_head(head_x, head_y)

    my_camera = Camera.Camera()
    my_camera.camera_open()

    try:
        _setup_terminal()
        while True:
            ret, frame = my_camera.read()
            ch = _get_key()
            if ch is not None:
                key = ord(ch)
                if key == 27 or key == 3:          # ESC 或 Ctrl+C
                    break
                handle_key(key, frame if ret else None)
            time.sleep(0.01)
    finally:
        _restore_terminal()
        AGC.stopActionGroup()
        my_camera.camera_close()
        print("\r程序已退出")


if __name__ == "__main__":
    if sys.version_info.major == 2:
        print("Please run this program with python3!")
        sys.exit(0)
    main()
