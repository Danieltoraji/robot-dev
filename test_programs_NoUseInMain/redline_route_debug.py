#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""巡线决策复盘浏览器（RedLinePatrolV3 离线模拟器）。

读取机器人回传的循迹照片，直接 import 比赛代码 levels/football_codes/
RedLinePatrolV3.py，逐帧执行同一套视觉管线 run() 与决策链 decide_action()，
在 PC 上可视化"计算出的路线"与"机器人当时的下一步行进方向"。

与 HSV_threhold_test_program.py 不同：本工具没有任何阈值调节轨道条，所有
阈值只读（直接来自比赛代码常量），定位是决策复盘浏览器——以后只改比赛
代码，这里自动同步验证。

机器人端采集照片：
    1. RedLinePatrolV3.py 顶部 SAVE_DEBUG_FRAMES = True
    2. 正常跑一次巡线（demoV4 或独立运行），校正帧自动存到
       /home/pi/codes/pictures/patrol/patrol_%06d.jpg
    3. PC 拉取：python tools/pull_from_robot.py --remote
       /home/pi/codes/pictures/patrol --dest <本地目录>

用法：
    python redline_route_debug.py --dir <照片目录>
    python redline_route_debug.py --image <单张照片>
    python redline_route_debug.py --dir <照片目录> --frames 30 --log replay.log

按键：n/b 前后翻页，空格 自动播放/暂停，r 重置会话状态，h 帮助，q 退出。
注意：b 后退只回退照片序号，会话状态不回滚（决策依赖跨帧状态）；想从头
回放请按 r。
"""

from __future__ import print_function

import argparse
import glob
import importlib.util
import os
import re
import sys
import time

import cv2
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REDLINE_DEFAULT = os.path.normpath(os.path.join(
    CURRENT_DIR, '..', 'levels', 'football_codes', 'RedLinePatrolV3.py'))
FALLBACK_PHOTO_DIR = os.path.normpath(os.path.join(
    CURRENT_DIR, '..', 'archive', 'result', 'pulled_photos'))
DEFAULT_PHOTO_DIR = os.path.join(CURRENT_DIR, 'patrol_photos')

WINDOW_ROUTE = '巡线路线 (RedLinePatrolV3 复盘)'
WINDOW_PANEL = '决策面板 (Decision Panel)'
PANEL_WIDTH = 900
PANEL_HEIGHT = 620
AUTO_STEP_SECONDS = 0.3

# 决策面板里只读展示的关键阈值（名字直接取自比赛代码模块常量）。
PANEL_THRESHOLD_NAMES = (
    'RED_H_LOW1', 'RED_H_HIGH1', 'RED_H_LOW2', 'RED_H_HIGH2',
    'RED_S_LOW', 'RED_V_LOW', 'VISION_TOP_Y', 'VISION_SIDE_CROP',
    'MIN_PATH_AREA', 'MIN_BAND_PIXELS', 'SAMPLE_BAND_HEIGHT',
    'CENTER_DEADBAND', 'HEADING_TURN_THRESHOLD', 'HEADING_MAX',
    'HEADING_HINT_THRESHOLD', 'VERTICAL_HEADING_TOLERANCE',
    'FILTER_ALPHA', 'FILTER_RESET_JUMP',
    'HORIZONTAL_MIN_WIDTH', 'HORIZONTAL_FILL_MIN',
    'CORNER_DIRECTION_DEADBAND', 'CORNER_TURN_TRIGGER_Y',
    'CORNER_LATCH_SECONDS', 'POST_TURN_CLEAR_FRAMES',
    'POST_TURN_MAX_EXTRA_TURNS', 'CORNER_MAIN_STEPS_LEFT',
    'CORNER_MAIN_STEPS_RIGHT', 'CORNER_FINE_MAX_STEPS',
    'CORNER_EXIT_HEADING_PX', 'CORNER_MIN_VERTICAL_SPAN',
    'CORNER_TRIGGER_CENTER_DEADBAND', 'CORNER_TRIGGER_DEEP_Y',
    'CORNER_CONFLICT_MIN_OFFSET', 'CORNER_CONFLICT_HOLD_SECONDS',
    'CORNER_CONFLICT_DEEP_SECONDS', 'CORNER_FINE_BAR_MAX_STEPS',
    'CORNER_BAR_FREE_REARM_FRAMES',
    'LOST_CONFIRM_SECONDS', 'SEARCH_PRIMARY_MAX_TURNS',
    'SEARCH_REVERSE_MAX_TURNS', 'SEARCH_OBSERVE_SECONDS',
    'MIN_CONTOUR_AREA', 'NORMAL_ACTION_SETTLE',
)


def load_redline(redline_path=None):
    """直接加载比赛代码 RedLinePatrolV3.py（不做任何算法拷贝）。

    Windows 上该文件的 hiwonder 导入被守卫，离线模式下只提供视觉与决策。
    """
    redline_path = os.path.normpath(redline_path or REDLINE_DEFAULT)
    if not os.path.isfile(redline_path):
        raise FileNotFoundError('找不到巡线代码：{}'.format(redline_path))
    football_dir = os.path.dirname(redline_path)
    if football_dir not in sys.path:
        sys.path.insert(0, football_dir)
    spec = importlib.util.spec_from_file_location('RedLinePatrolV3', redline_path)
    redline = importlib.util.module_from_spec(spec)
    sys.modules['RedLinePatrolV3'] = redline
    spec.loader.exec_module(redline)
    print('已加载比赛代码：{}（HARDWARE_AVAILABLE={}）'.format(
        redline_path, redline.HARDWARE_AVAILABLE))
    return redline


def print_key_thresholds(redline):
    print('当前比赛代码关键阈值（只读，改动请修改 RedLinePatrolV3.py）：')
    for name in PANEL_THRESHOLD_NAMES:
        print('  {} = {}'.format(name, getattr(redline, name, '<缺失>')))
    print('  CORNER_TURN_SEQUENCE = {}'.format(
        getattr(redline, 'CORNER_TURN_SEQUENCE', '<缺失>')))
    print('  roi = {}'.format(getattr(redline, 'roi', '<缺失>')))


def _frame_sort_key(name):
    digits = re.findall(r'\d+', name)
    return int(digits[-1]) if digits else name


def collect_photos(photo_dir=None, image_path=None):
    """收集照片并按下标排序。优先 --image，其次 --dir，
    再回退脚本旁 patrol_photos/ 与 archive/result/pulled_photos/。"""
    if image_path:
        if not os.path.isfile(image_path):
            raise FileNotFoundError('找不到照片：{}'.format(image_path))
        return [os.path.abspath(image_path)]

    search_dirs = []
    if photo_dir:
        search_dirs.append(photo_dir)
    search_dirs.append(DEFAULT_PHOTO_DIR)
    search_dirs.append(FALLBACK_PHOTO_DIR)

    for directory in search_dirs:
        if not directory or not os.path.isdir(directory):
            continue
        found = [p for p in glob.glob(os.path.join(directory, '*'))
                 if p.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        if found:
            found.sort(key=lambda p: _frame_sort_key(os.path.basename(p)))
            print('使用照片目录：{}（{} 张）'.format(directory, len(found)))
            return found

    print('没有找到照片：请用 --dir 指定目录、--image 指定单张，'
          '或把照片放到 {}'.format(DEFAULT_PHOTO_DIR))
    return []


def draw_overlays(redline, frame, state, decision, session):
    """在 run() 的基础标注上补画：决策横幅、heading 箭头、
    采样带左右边界、最宽带框、路口触发线。"""
    height, width = frame.shape[:2]

    # 顶部横幅：本帧动作 + 状态机摘要（cv2.putText 不支持中文，用英文）。
    cv2.rectangle(frame, (0, 0), (width, 66), (40, 40, 40), -1)
    action_names = ', '.join(a[0] for a in decision.actions) or '(none)'
    cv2.putText(frame, 'action: {}'.format(action_names), (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    if decision.wait:
        state_text = 'wait {:.2f}s'.format(decision.wait)
    elif session.turn_active:
        state_text = 'turn corner#{} steps={} fine_left={}'.format(
            session.corner_index + 1, session.turn_steps,
            session.turn_fine_remaining)
    elif session.post_turn_active:
        state_text = 'post_turn clear={}/{} extra={}/{}'.format(
            session.post_turn_clear_frames, redline.POST_TURN_CLEAR_FRAMES,
            session.post_turn_extra_turns, redline.POST_TURN_MAX_EXTRA_TURNS)
    else:
        state_text = 'next corner #{} dir={}'.format(
            session.corner_index + 1,
            redline.CORNER_TURN_SEQUENCE[session.corner_index]
            if session.corner_index < len(redline.CORNER_TURN_SEQUENCE)
            else 'END')
    cv2.putText(frame, state_text, (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 200), 1, cv2.LINE_AA)

    if not state.get('visible'):
        cv2.putText(frame, 'LINE LOST', (width // 2 - 70, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        return

    # heading 箭头：raw_heading 是真实视觉方向；超过 HEADING_MAX 被置 0
    # 忽略时画橙色并标注 IGNORED，绿色表示参与决策。
    near_x = int(state.get('near_x', width / 2.0))
    near_y = int(state.get('near_y', height - 1))
    raw = float(state.get('raw_heading', 0.0))
    tip_x = max(0, min(width - 1, int(near_x + raw * 0.25)))
    tip_y = max(0, near_y - 45)
    used = abs(raw) <= redline.HEADING_MAX
    color = (0, 255, 0) if used else (0, 165, 255)
    cv2.arrowedLine(frame, (near_x, near_y), (tip_x, tip_y), color, 2,
                    cv2.LINE_AA, tipLength=0.15)
    cv2.putText(frame, 'raw_h={:.0f}{}'.format(raw, '' if used else ' IGNORED'),
                (near_x + 6, near_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                color, 1, cv2.LINE_AA)

    # 采样带左右边界（数据来自比赛代码 run() 的真实采样结果）。
    for y, _cx, lx, rx, _count in state.get('samples', []):
        cv2.line(frame, (int(lx), int(y)), (int(rx), int(y)),
                 (255, 0, 0), 1, cv2.LINE_AA)

    # 最宽带（路口横条候选）青色框 + 宽度/填充率。
    max_width = float(state.get('max_width', 0.0))
    if max_width >= redline.HORIZONTAL_MIN_WIDTH:
        max_y = int(state.get('max_width_y', 0))
        cv2.rectangle(frame,
                      (int(state.get('max_width_left_x', 0)), max_y - 25),
                      (int(state.get('max_width_right_x', 0)), max_y + 25),
                      (255, 255, 0), 2)
        cv2.putText(frame, 'w={:.0f} fill={:.2f}'.format(
            max_width, float(state.get('max_width_fill_ratio', 0.0))),
            (int(state.get('max_width_left_x', 0)), max_y - 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)

    # 路口触发线（corner_y 达到该深度才触发转弯）。
    trigger_y = int(redline.CORNER_TURN_TRIGGER_Y)
    cv2.line(frame, (0, trigger_y), (width, trigger_y), (0, 255, 255), 1)
    cv2.putText(frame, 'trigger y={}'.format(trigger_y),
                (width - 135, trigger_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (0, 255, 255), 1, cv2.LINE_AA)
    # 横条深度豁免线（corner_y 达到该深度后不再要求近端居中）。
    deep_y = int(getattr(redline, 'CORNER_TRIGGER_DEEP_Y', 430))
    cv2.line(frame, (0, deep_y), (width, deep_y), (255, 128, 0), 1)
    cv2.putText(frame, 'deep y={}'.format(deep_y),
                (width - 135, deep_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 128, 0), 1, cv2.LINE_AA)
    if state.get('corner_ready'):
        cv2.putText(frame, 'CORNER READY y={:.0f}'.format(
            float(state.get('corner_y', 0.0))), (8, height - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)


def draw_help(frame):
    """在路线窗口底部叠加按键帮助。"""
    height, width = frame.shape[:2]
    cv2.rectangle(frame, (0, height - 22), (width, height), (0, 0, 0), -1)
    cv2.putText(frame, 'n: next  b: prev  space: play/pause  r: reset  '
                       'h: help  q: quit',
                (8, height - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 255, 255), 1, cv2.LINE_AA)


def draw_panel(redline, state, decision, session, now, photo_name):
    """决策面板：本帧量测值、会话状态、决策命中与只读阈值。"""
    panel = np.zeros((PANEL_HEIGHT, PANEL_WIDTH, 3), np.uint8)
    y_pos = 0

    def line(text, color=(220, 220, 220), step=19):
        nonlocal y_pos
        y_pos += step
        cv2.putText(panel, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    color, 1, cv2.LINE_AA)

    near_x = float(state.get('near_x', 0.0) or 0.0)

    line('--- VISION (from RedLinePatrolV3.run) ---', (0, 220, 255))
    line('file: {}'.format(os.path.basename(photo_name)))
    line('now={:.3f}s visible={} frame={}'.format(
        now, state.get('visible'), state.get('frame_id')))
    line('near_x={:.1f} center_error={:+.1f}'.format(
        near_x, near_x - redline.IMAGE_CENTER_X))
    line('heading={:+.1f} raw_heading={:+.1f} vspan={:.1f}'.format(
        float(state.get('heading', 0.0)), float(state.get('raw_heading', 0.0)),
        float(state.get('vertical_span', 0.0))))
    line('samples={} max_width={:.0f}@y={:.0f} fill={:.2f}'.format(
        state.get('sample_count', 0), float(state.get('max_width', 0.0)),
        float(state.get('max_width_y', 0.0)),
        float(state.get('max_width_fill_ratio', 0.0))))
    line('corner_y={:.0f} ready={} horiz={} pending={}'.format(
        float(state.get('corner_y', 0.0)), state.get('corner_ready'),
        state.get('horizontal_corner'), state.get('corner_pending')))
    line('corner_dir={} offset={:.0f}'.format(
        state.get('corner_direction', 0),
        float(state.get('corner_offset_x', 0.0))))

    line('--- SESSION (PatrolSession) ---', (0, 220, 255))
    seq = redline.CORNER_TURN_SEQUENCE
    next_dir = (seq[session.corner_index]
                if session.corner_index < len(seq) else 'END')
    line('next corner #{} dir={} seq={}'.format(
        session.corner_index + 1, next_dir, seq))
    line('turn_active={} steps={} fine_left={} pre_fwd={} main={}'.format(
        session.turn_active, session.turn_steps, session.turn_fine_remaining,
        session.turn_pre_forward_remaining, session.turn_main_issued))
    line('small_streak={} base_err={:.1f}'.format(
        session.turn_small_streak, session.turn_streak_base_error))
    line('post_turn={} clear={}/{} extra={}/{} progress={}/{}'.format(
        session.post_turn_active, session.post_turn_clear_frames,
        redline.POST_TURN_CLEAR_FRAMES, session.post_turn_extra_turns,
        redline.POST_TURN_MAX_EXTRA_TURNS, session.post_turn_progress_steps,
        redline.POST_TURN_MIN_FORWARD_STEPS))
    line('search active={} primary={}/{} reverse={}/{}'.format(
        session.search.active, session.search.primary_turns,
        redline.SEARCH_PRIMARY_MAX_TURNS, session.search.reverse_turns,
        redline.SEARCH_REVERSE_MAX_TURNS))
    filtered = (redline._filtered_near_x
                if redline._filtered_near_x is not None else 'None')
    line('no_progress={}/{} filtered_near={}'.format(
        session.no_progress_turns, redline.MAX_NO_PROGRESS_TURNS, filtered))

    line('--- DECISION (decide_action) ---', (0, 220, 255))
    line('label: {}'.format(decision.label), (0, 255, 255))
    if decision.actions:
        for act in decision.actions:
            line('  -> {} times={} with_stand={}'.format(act[0], act[1], act[2]),
                 (0, 255, 0))
    else:
        line('  -> (no action)', (0, 255, 0))
    line('wait={:.2f}s settle_after={:.2f}s observe={}'.format(
        decision.wait, decision.settle_after, decision.observe))

    line('--- KEY THRESHOLDS (read-only) ---', (0, 220, 255))
    buffer = ''
    for name in PANEL_THRESHOLD_NAMES:
        item = '{}={}'.format(name, getattr(redline, name, '?'))
        if len(buffer) + len(item) + 2 > 95:
            line('  ' + buffer)
            buffer = item
        else:
            buffer = (buffer + '  ' + item) if buffer else item
    if buffer:
        line('  ' + buffer)
    return panel


def process_frame(redline, patrol_session, frame, now):
    """离线执行一帧：视觉 run() → 决策 decide_action() → 叠加可视化。

    与真机完全同一套代码：run() 是比赛代码的视觉管线，decide_action()
    是 move() 的决策链本体。
    """
    display = redline.run(frame, now=now)
    state = redline.get_vision_state()
    decision = redline.decide_action(state, patrol_session, now)
    draw_overlays(redline, display, state, decision, patrol_session)
    return display, state, decision


def _log_line(index, photo, now, state, decision):
    return ('frame={} file={} now={:.3f} visible={} near_x={} heading={:.1f} '
            'raw_h={:.1f} actions=[{}] label={}\n').format(
        index + 1, os.path.basename(photo), now, state.get('visible'),
        state.get('near_x'), state.get('heading', 0.0),
        state.get('raw_heading', 0.0),
        ','.join(a[0] for a in decision.actions) or '-', decision.label)


def _read_frame(redline, photo):
    img = cv2.imread(photo)
    if img is None:
        return None
    return cv2.resize(img, redline.FRAME_SIZE,
                      interpolation=cv2.INTER_NEAREST)


def _run_batch(redline, photos, args):
    """非交互批量回放：逐帧打印决策（不弹窗），供验证与日志 diff。"""
    redline.reset()
    redline.enter = True
    redline.running = True
    patrol_session = redline.PatrolSession()
    log_file = open(args.log, 'w', encoding='utf-8') if args.log else None
    try:
        for i in range(min(args.frames, len(photos))):
            frame = _read_frame(redline, photos[i])
            if frame is None:
                print('[帧 {}/{}] 读取失败，跳过：{}'.format(
                    i + 1, len(photos), photos[i]))
                continue
            print('[帧 {}/{}] {}'.format(
                i + 1, len(photos), os.path.basename(photos[i])))
            _display, state, decision = process_frame(
                redline, patrol_session, frame, i / args.fps)
            if log_file is not None:
                log_file.write(_log_line(i, photos[i], i / args.fps,
                                         state, decision))
                log_file.flush()
    finally:
        if log_file is not None:
            log_file.close()


def main():
    parser = argparse.ArgumentParser(
        description='巡线决策复盘浏览器（离线，直接加载比赛代码）')
    parser.add_argument('--dir', help='照片目录（默认脚本旁 patrol_photos/，'
                                      '缺省回退 archive/result/pulled_photos/）')
    parser.add_argument('--image', help='单张照片路径')
    parser.add_argument('--redline', default=REDLINE_DEFAULT,
                        help='RedLinePatrolV3.py 路径（默认相对脚本定位）')
    parser.add_argument('--fps', type=float, default=15.0,
                        help='虚拟帧率：帧序号折算时间的除数，默认 15')
    parser.add_argument('--log', help='逐帧决策日志输出文件（utf-8）')
    parser.add_argument('--frames', type=int, default=0,
                        help='非交互批量回放前 N 帧后退出（0 为交互模式）')
    args = parser.parse_args()

    redline = load_redline(args.redline)
    print_key_thresholds(redline)
    photos = collect_photos(args.dir, args.image)
    if not photos:
        return

    if args.frames:
        _run_batch(redline, photos, args)
        return

    # 交互模式。离线会话：不调 init/start（不碰舵机），只重置视觉状态并
    # 置位运行标志；虚拟时间按"帧序号/fps"推进，与回放速度无关。
    redline.reset()
    redline.enter = True
    redline.running = True
    patrol_session = redline.PatrolSession()

    cv2.namedWindow(WINDOW_ROUTE, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOW_PANEL, cv2.WINDOW_NORMAL)

    index = 0
    playing = False
    show_help = False
    last_advance = time.time()
    log_file = open(args.log, 'w', encoding='utf-8') if args.log else None

    def process(idx):
        frame = _read_frame(redline, photos[idx])
        if frame is None:
            return None
        print('[帧 {}/{}] {}'.format(
            idx + 1, len(photos), os.path.basename(photos[idx])))
        display, state, decision = process_frame(
            redline, patrol_session, frame, idx / args.fps)
        panel = draw_panel(redline, state, decision, patrol_session,
                           idx / args.fps, photos[idx])
        if log_file is not None:
            log_file.write(_log_line(idx, photos[idx], idx / args.fps,
                                     state, decision))
            log_file.flush()
        return display, panel

    current = process(index)
    try:
        while True:
            if current is None:
                index += 1
                if index >= len(photos):
                    print('没有可显示的帧')
                    break
                current = process(index)
                continue
            display, panel = current
            if show_help:
                draw_help(display)
            cv2.imshow(WINDOW_ROUTE, display)
            cv2.imshow(WINDOW_PANEL, panel)
            key = cv2.waitKey(20) & 0xFF
            if key in (27, ord('q')):
                break
            elif key == ord('n'):
                if index + 1 < len(photos):
                    index += 1
                    current = process(index)
            elif key == ord('b'):
                if index > 0:
                    index -= 1
                    current = process(index)
            elif key == ord(' '):
                playing = not playing
                last_advance = time.time()
                print('自动播放：{}'.format('开' if playing else '关'))
            elif key == ord('r'):
                print('重置会话状态，从头回放')
                redline.reset()
                redline.enter = True
                redline.running = True
                patrol_session = redline.PatrolSession()
                index = 0
                current = process(index)
            elif key == ord('h'):
                show_help = not show_help

            if playing and time.time() - last_advance >= AUTO_STEP_SECONDS:
                last_advance = time.time()
                if index + 1 < len(photos):
                    index += 1
                    current = process(index)
                else:
                    playing = False
    finally:
        if log_file is not None:
            log_file.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
