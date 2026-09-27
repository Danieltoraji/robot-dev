#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""calib_tuner.py —— apriltag_sorting_task 阈值调参工具(PC 端 GUI)

用途: 不碰代码就能调节 levels/apriltag_sorting_task.py 的全部阈值参数,
     实时查看照片上的检测效果与状态机决策, 导出 calib_config.json 供机器人加载
     (主程序启动时若同目录存在 calib_config.json 会自动应用, 命令行显式参数优先)。

默认值自动同步: 本工具用 ast 解析主文件源码提取常量与 argparse 默认值,
不 import 主文件(避免 PC 上无 hiwonder SDK 的问题), 因此主程序改参数后工具永远一致。

用法:
    .venv/Scripts/python.exe tools/calib_tuner.py --image <照片或目录>   # GUI 模式
    .venv/Scripts/python.exe tools/calib_tuner.py --dump                # 无GUI: 打印默认配置JSON

按键(图片窗口 calib_tuner):
    n / b     下一张 / 上一张照片
    TAB       切换视图: 原图 → 蓝LAB掩膜 → 红HSV掩膜+ROI
    1~6       切换参数页: 1视觉 2颜色接近 3Tag放置 4巡线 5终点+动作 6卡尔曼
    s         保存 levels/calib_config.json(需 sync 到机器人)
    p         控制台打印可粘贴的 Python 常量块
    r         重置全部参数为代码默认值
    q / ESC   退出
鼠标: 在图片上点击, 打印该像素 BGR/LAB/HSV 值(标颜色阈值的关键操作)
Tag模拟窗口: dist/dx/cy 三个滑条, 实时显示 approach_tag 将命中的分支与动作
"""

import argparse
import ast
import glob
import json
import math
import os
import sys

import cv2
import numpy as np

TASK_PY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "levels", "apriltag_sorting_task.py")
CONFIG_PATH = os.path.join(os.path.dirname(TASK_PY), "calib_config.json")

WIN_IMG = "calib_tuner"
# 窗口标题/滑条标签用ASCII: OpenCV在Windows上对中文控件文字会乱码
WIN_PARAM = "Params"
WIN_TAG = "TagSim"


def _safe_destroy_window(name):
    """OpenCV 5 对不存在的窗口调用 destroyWindow 会抛 cv2.error, 这里忽略"""
    try:
        cv2.destroyWindow(name)
    except cv2.error:
        pass

# (key, page, label, scale, tmin, tb_max, idx)  idx=None 为平坦参数
# trackbar 范围 [0, tb_max], 参数值 = tb/scale + tmin
PARAMS = [
    # 1 视觉
    ("BLUE_LAB_MIN", 1, "Blue L min", 1, 0, 255, 0),
    ("BLUE_LAB_MIN", 1, "Blue A min", 1, 0, 255, 1),
    ("BLUE_LAB_MIN", 1, "Blue B min", 1, 0, 255, 2),
    ("BLUE_LAB_MAX", 1, "Blue L max", 1, 0, 255, 0),
    ("BLUE_LAB_MAX", 1, "Blue A max", 1, 0, 255, 1),
    ("BLUE_LAB_MAX", 1, "Blue B max", 1, 0, 255, 2),
    ("RED_H_LOW1", 1, "Red H1 lo", 1, 0, 180, None),
    ("RED_H_HIGH1", 1, "Red H1 hi", 1, 0, 180, None),
    ("RED_H_LOW2", 1, "Red H2 lo", 1, 0, 180, None),
    ("RED_H_HIGH2", 1, "Red H2 hi", 1, 0, 180, None),
    ("RED_S_LOW", 1, "Red S lo", 1, 0, 255, None),
    ("RED_S_HIGH", 1, "Red S hi", 1, 0, 255, None),
    ("RED_V_LOW", 1, "Red V lo", 1, 0, 255, None),
    ("RED_V_HIGH", 1, "Red V hi", 1, 0, 255, None),
    ("COLOR_AREA_MIN", 1, "Blue areaMin", 1, 0, 10000, None),
    ("CENTER_X", 1, "CENTER_X", 1, 0, 640, None),
    # 2 颜色接近
    ("COLOR_FAR_Y", 2, "FAR_Y", 1, 0, 480, None),
    ("COLOR_NEAR_Y", 2, "NEAR_Y", 1, 0, 480, None),
    ("COLOR_TOO_NEAR_Y", 2, "TOO_NEAR_Y", 1, 0, 480, None),
    ("COLOR_X_TURN", 2, "X_TURN", 1, 0, 320, None),
    ("COLOR_X_LARGE", 2, "X_LARGE", 1, 0, 320, None),
    ("COLOR_X_FINE", 2, "X_FINE", 1, 0, 320, None),
    ("pick_area_threshold", 2, "pick_area", 1, 0, 20000, None),
    ("pick_y_threshold", 2, "pick_y", 1, 0, 480, None),
    ("pick_too_near_y", 2, "pick_too_near_y", 1, 0, 480, None),
    ("STEP4_MAX_FORWARDS", 2, "STEP4_MAX_FWD", 1, 1, 30, None),
    # 3 Tag 放置
    ("BOARD_DEPTH_M", 3, "BOARD_DEPTH_mm", 1000, 0, 500, None),
    ("TAG_PLACE_NEAR_M", 3, "PLACE_NEAR_mm", 1000, 0, 2000, None),
    ("TAG_PLACE_FAR_M", 3, "PLACE_FAR_mm", 1000, 0, 2000, None),
    ("TAG_PLACE_BIG_M", 3, "PLACE_BIG_mm", 1000, 0, 3000, None),
    ("TAG_FAR_Y", 3, "TAG_FAR_Y", 1, 0, 480, None),
    ("TAG_NEAR_Y", 3, "TAG_NEAR_Y", 1, 0, 480, None),
    ("TAG_TOO_NEAR_Y", 3, "TAG_TOO_NEAR_Y", 1, 0, 480, None),
    ("TAG_X_TURN", 3, "TAG_X_TURN", 1, 0, 320, None),
    ("TAG_X_LARGE", 3, "TAG_X_LARGE", 1, 0, 320, None),
    ("TAG_X_FINE", 3, "TAG_X_FINE", 1, 0, 320, None),
    ("HEAD_SCAN_LEFT", 3, "HEAD_SCAN_LEFT", 1, 0, 1000, None),
    ("HEAD_SCAN_RIGHT", 3, "HEAD_SCAN_RIGHT", 1, 0, 1000, None),
    # 4 巡线
    ("LINE_CENTER_X", 4, "LINE_CENTER_X", 1, 0, 640, None),
    ("LINE_TURN_THRESHOLD", 4, "LINE_TURN_THR", 1, 0, 320, None),
    ("SEARCH_LINE_ALIGN_THRESHOLD", 4, "SEARCH_ALIGN", 1, 0, 320, None),
    ("VERTICAL_ANGLE_TOL", 4, "VERT_ANGLE_TOL_deg", 1, 0, 90, None),
    ("LINE_LOST_HOLD", 4, "LOST_HOLD_x10", 10, 0, 100, None),
    ("LINE_LOST_TIMEOUT", 4, "NOFRAME_TO_x10", 10, 0, 100, None),
    ("MAX_LOST_TURNS", 4, "MAX_LOST_TURNS", 1, 0, 50, None),
    ("MAX_CONSEC_MOVES", 4, "MAX_CONSEC_MOVES", 1, 1, 10, None),
    ("FOLLOW_CONFIRM_S", 4, "FOLLOW_CONFIRM_x0.1s", 10, 0, 100, None),
    ("FOLLOW_MAX_ADJUST", 4, "FOLLOW_MAX_ADJUST", 1, 0, 100, None),
    ("line_head_delta", 4, "LINE_HEAD_DELTA", 1, 0, 500, None),
    # 5 终点 + 动作计数
    ("END_YAW_LOWER", 5, "END_YAW_LO", 1, -90, 180, None),
    ("END_YAW_UPPER", 5, "END_YAW_HI", 1, -90, 180, None),
    ("WALK_STEPS", 5, "WALK_STEPS", 1, 1, 50, None),
    ("MAX_TURN", 5, "MAX_TURN", 1, 1, 100, None),
    ("PICK_FINAL_STEPS", 5, "PICK_FINAL", 1, 0, 10, None),
    ("PLACE_FINAL_STEPS", 5, "PLACE_FINAL", 1, 0, 10, None),
    ("MAX_PICK_RETRIES", 5, "PICK_RETRIES", 1, 0, 10, None),
    ("post_pick_left_turns", 5, "POST_PICK_TURNS", 1, 0, 50, None),
    ("post_pick_forward_steps", 5, "POST_PICK_STEPS", 1, 0, 50, None),
    ("back_steps_after_place", 5, "BACK_AFTER", 1, 0, 50, None),
    ("line_final_steps", 5, "LINE_FINAL_STEPS", 1, 0, 50, None),
    ("line_search_turns", 5, "LINE_SEARCH_TURNS", 1, 0, 100, None),
    # 6 卡尔曼 (Q=过程噪声越大越跟手, R=观测噪声越大越平滑)
    ("KF_CX_Q", 6, "cx Q x1000", 1000, 0, 20000, None),
    ("KF_CX_R", 6, "cx R x1000", 1000, 0, 20000, None),
    ("KF_CY_Q", 6, "cy Q x1000", 1000, 0, 20000, None),
    ("KF_CY_R", 6, "cy R x1000", 1000, 0, 20000, None),
    ("KF_DIST_Q", 6, "dist Q x100000", 100000, 0, 5000, None),
    ("KF_DIST_R", 6, "dist R x100000", 100000, 0, 5000, None),
    ("KF_OFFSET_Q", 6, "offset Q x100000", 100000, 0, 5000, None),
    ("KF_OFFSET_R", 6, "offset R x100000", 100000, 0, 5000, None),
    ("KF_ANGLE_Q", 6, "angle Q x1000", 1000, 0, 20000, None),
    ("KF_ANGLE_QV", 6, "angle Qv x1000", 1000, 0, 10000, None),
    ("KF_ANGLE_R", 6, "angle R x1000", 1000, 0, 50000, None),
    ("KF_AREA_Q", 6, "area Q", 1, 0, 2000, None),
    ("KF_AREA_R", 6, "area R", 1, 0, 5000, None),
    ("KF_LINE_Q", 6, "line Q x1000", 1000, 0, 20000, None),
    ("KF_LINE_R", 6, "line R x1000", 1000, 0, 50000, None),
]


def _literal(node):
    """安全求值 ast 字面量(常量/列表/元组), 失败返回 None"""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        vals = [_literal(e) for e in node.elts]
        if all(v is not None for v in vals):
            return vals
    return None


def _eval_expr(node, consts):
    """求值"常量+已提取常量名"的简单表达式, 如 0.17 + BOARD_DEPTH_M; 失败返回 None"""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return consts.get(node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = _eval_expr(node.operand, consts)
        return -v if v is not None else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        left = _eval_expr(node.left, consts)
        right = _eval_expr(node.right, consts)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        return left / right
    return None


def _assign_targets(node):
    """返回赋值语句的目标名列表(兼容 Python 3.14 的 Tuple 目标形式)"""
    targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
    if not targets and len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple):
        targets = [e.id for e in node.targets[0].elts if isinstance(e, ast.Name)]
    return targets


def load_defaults(task_py):
    """ast 解析主文件: 返回(模块常量dict, argparse默认值dict)"""
    with open(task_py, encoding="utf-8") as f:
        src = f.read()
    tree = ast.parse(src)
    consts, arg_defaults = {}, {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            val = _literal(node.value)
            if val is None:
                val = _eval_expr(node.value, consts)
            if val is None:
                continue
            targets = _assign_targets(node)
            if isinstance(val, list) and len(targets) > 1:
                for t, v in zip(targets, val):
                    if v is not None:
                        consts[t] = v
            elif len(targets) == 1:
                consts[targets[0]] = val
        elif isinstance(node, ast.FunctionDef) and node.name == "parse_args":
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                        and sub.func.attr == "add_argument"):
                    name = dflt = None
                    for a in sub.args:
                        if isinstance(a, ast.Constant) and isinstance(a.value, str) \
                                and a.value.startswith("--"):
                            name = a.value.lstrip("-").replace("-", "_")
                    for kw in sub.keywords:
                        if kw.arg == "default":
                            dflt = _literal(kw.value)
                    if name and dflt is not None:
                        arg_defaults[name] = dflt
    return consts, arg_defaults


def detect_blue_raw(img, v):
    """复刻主程序 detect_blue(去掉卡尔曼), 返回(cx, cy, area, mask)"""
    h, w = img.shape[:2]
    small = cv2.resize(img, (320, 240), interpolation=cv2.INTER_NEAREST)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    gb = cv2.GaussianBlur(lab, (3, 3), 3)
    mask = cv2.inRange(gb, tuple(v["BLUE_LAB_MIN"]), tuple(v["BLUE_LAB_MAX"]))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.dilate(cv2.erode(mask, k), k)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
    best, best_area = None, 0.0
    for c in cnts:
        a = abs(cv2.contourArea(c))
        if a > best_area:
            best_area, best = a, c
    if best is None or best_area < v["COLOR_AREA_MIN"]:
        return -1, -1, best_area, mask
    rect = cv2.minAreaRect(best)
    box = cv2.boxPoints(rect).astype(np.intp)
    for i in range(4):
        box[i, 0] = int(box[i, 0] * w / 320)
        box[i, 1] = int(box[i, 1] * h / 240)
    cx = int((box[0, 0] + box[2, 0]) / 2)
    cy = int((box[0, 1] + box[2, 1]) / 2)
    return cx, cy, best_area, mask


def detect_red_line_raw(img, v, roi):
    """复刻主程序 detect_red_line(去掉卡尔曼), 返回(line_cx, vertical, mask)"""
    gb = cv2.GaussianBlur(img, (3, 3), 3)
    hsv = cv2.cvtColor(gb, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array([v["RED_H_LOW1"], v["RED_S_LOW"], v["RED_V_LOW"]]),
                     np.array([v["RED_H_HIGH1"], v["RED_S_HIGH"], v["RED_V_HIGH"]]))
    m2 = cv2.inRange(hsv, np.array([v["RED_H_LOW2"], v["RED_S_LOW"], v["RED_V_LOW"]]),
                     np.array([v["RED_H_HIGH2"], v["RED_S_HIGH"], v["RED_V_HIGH"]]))
    mask = cv2.bitwise_or(m1, m2)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.dilate(cv2.erode(mask, k), k)
    mask[:, 0:160] = 0
    mask[:, 480:640] = 0
    sx = sw = 0.0
    vertical = False
    for r in roi:
        sub = mask[r[0]:r[1], r[2]:r[3]]
        cnts = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)[-2]
        best, ba = None, 0.0
        for c in cnts:
            a = abs(cv2.contourArea(c))
            if a > ba:
                ba, best = a, c
        if best is not None:
            box = np.intp(cv2.boxPoints(cv2.minAreaRect(best)))
            cx = float(box[0][0] + box[2][0]) / 2.0
            sx += cx * r[4]
            sw += r[4]
    # 竖直判定: 三段ROI合并区域的轮廓PCA主轴方向(单带40px会被带宽截断, 线宽大时主轴翻转)
    _y1 = min(r[0] for r in roi)
    _y2 = max(r[1] for r in roi)
    _mcnts = cv2.findContours(mask[_y1:_y2, :], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)[-2]
    _mbest, _marea = None, 0.0
    for c in _mcnts:
        a = abs(cv2.contourArea(c))
        if a > _marea:
            _marea, _mbest = a, c
    if _mbest is not None:
        m = cv2.moments(_mbest)
        if m["m00"] > 0:
            mu11 = m["mu11"] / m["m00"]
            mu20 = m["mu20"] / m["m00"]
            mu02 = m["mu02"] / m["m00"]
            angle = math.degrees(0.5 * math.atan2(2.0 * mu11, mu20 - mu02))
            if abs(angle) >= 90.0 - v["VERTICAL_ANGLE_TOL"]:
                vertical = True
    line_cx = int(sx / sw) if sw > 0 else -1
    return line_cx, vertical, mask


def simulate_color_at(cx, cy, area, v, step):
    """纯逻辑复刻 approach_color 的 step 1~4 分支, 返回(动作名, 是否到位)"""
    if cx < 0:
        return "看不见(头扫/转身)", False
    dx = cx - v["CENTER_X"]
    if step == 1:
        if abs(dx) > 170 and cy > v["COLOR_NEAR_Y"]:
            return "back_fast", False
        if abs(dx) > v["COLOR_X_TURN"]:
            if cy <= v["COLOR_FAR_Y"]:
                return "turn_right" if dx > 0 else "turn_left", False
            return "turn_right_small_step" if dx > 0 else "turn_left_small_step", False
        if cy <= v["COLOR_FAR_Y"]:
            return "go_forward", False
        return "-> step2", False
    if step == 2:
        if cy > v["COLOR_NEAR_Y"]:
            return "back_fast", False
        if abs(dx) > 150:
            return "right_move_30" if dx > 0 else "left_move_30", False
        if abs(dx) > v["COLOR_X_LARGE"]:
            return "right_move_30" if dx > 0 else "left_move_30", False
        return "-> step3", False
    if step == 3:
        if cy > v["COLOR_TOO_NEAR_Y"]:
            return "back_fast", False
        if cy <= v["COLOR_FAR_Y"]:
            return "go_forward", False
        if abs(dx) > v["COLOR_X_LARGE"]:
            return "right_move_30" if dx > 0 else "left_move_30", False
        if abs(dx) > v["COLOR_X_FINE"]:
            return "right_move" if dx > 0 else "left_move", False
        return "-> step4", False
    if cy > v["COLOR_TOO_NEAR_Y"]:
        return "back_fast", False
    if area >= v["pick_area_threshold"]:
        return "DONE(面积达标)", True
    if cy < v["COLOR_NEAR_Y"]:
        return "go_forward_one_step(计数兜底%d步)" % v["STEP4_MAX_FORWARDS"], False
    if abs(dx) > v["COLOR_X_FINE"]:
        return "回 step3", False
    return "DONE(cy对正)", True


def simulate_tag_at(dist, dx, cy, v):
    """纯逻辑复刻 approach_tag 分支, 返回(动作名, 是否到位)"""
    if dist > v["TAG_PLACE_FAR_M"] or dist == 0:
        if abs(dx) > v["TAG_X_TURN"]:
            if dist > v["TAG_PLACE_BIG_M"] and abs(dx) > 150:
                return "turn_right" if dx > 0 else "turn_left", False
            return "turn_right_small_step" if dx > 0 else "turn_left_small_step", False
        if dist > v["TAG_PLACE_FAR_M"]:
            return "go_forward" if dist > v["TAG_PLACE_BIG_M"] else "go_forward_one_step", False
        if cy <= v["TAG_FAR_Y"]:
            return "go_forward(像素模式)", False
        return "DONE(到位放置)", True
    if dist >= v["TAG_PLACE_NEAR_M"]:
        if abs(dx) > v["TAG_X_TURN"]:
            return "turn_right_small_step" if dx > 0 else "turn_left_small_step", False
        if abs(dx) > v["TAG_X_LARGE"]:
            return "right_move_30" if dx > 0 else "left_move_30", False
        if abs(dx) > v["TAG_X_FINE"]:
            return "right_move" if dx > 0 else "left_move", False
        return "DONE(到位放置)", True
    return "back_fast(太近)", False


class Tuner:
    def __init__(self, consts, arg_defaults, image_paths):
        self.consts = consts
        self.arg_defaults = arg_defaults
        self.page = 1
        self.view = 0  # 0原图 1蓝mask 2红mask
        self.images = image_paths
        self.idx = 0
        self.cur_img = None
        self.values = {}
        for key, _page, _label, scale, tmin, _tbmax, idx in PARAMS:
            if idx is not None:
                if key not in self.values:
                    if key not in self.consts or not isinstance(self.consts[key], (list, tuple)):
                        raise SystemExit("无法从主文件提取默认值: %s" % key)
                    self.values[key] = list(self.consts[key])
            elif key not in self.values:
                dflt = self.consts.get(key, self.arg_defaults.get(key))
                if dflt is None:
                    raise SystemExit("无法从主文件提取默认值: %s" % key)
                self.values[key] = dflt
        self.roi = [tuple(r) for r in self.consts.get("LINE_ROI", [])]

    # ---- 取值/设值 ----
    def _get(self, key, idx):
        if idx is not None:
            return self.values[key][idx]
        return self.values[key]

    def _set(self, key, idx, value):
        if idx is not None:
            self.values[key][idx] = int(value)
        else:
            self.values[key] = value

    # ---- GUI ----
    def _build_param_page(self):
        _safe_destroy_window(WIN_PARAM)
        cv2.namedWindow(WIN_PARAM)
        for key, page, label, scale, tmin, tb_max, idx in PARAMS:
            if page != self.page:
                continue
            cur = self._get(key, idx)
            tb = int(round((cur - tmin) * scale))

            def make_cb(k, i, sc, tmn):
                def cb(val):
                    v = val / sc + tmn
                    self._set(k, i, int(v) if sc == 1 else v)
                return cb
            cv2.createTrackbar(label, WIN_PARAM, max(0, min(tb, tb_max)), tb_max,
                               make_cb(key, idx, scale, tmin))

    def _build_tag_window(self):
        _safe_destroy_window(WIN_TAG)
        cv2.namedWindow(WIN_TAG)
        cv2.createTrackbar("dist(mm)", WIN_TAG, 1000, 3000, lambda v: None)
        cv2.createTrackbar("dx(px)", WIN_TAG, 320, 640, lambda v: None)
        cv2.createTrackbar("cy(px)", WIN_TAG, 280, 480, lambda v: None)

    def _on_mouse(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and self.cur_img is not None:
            h, w = self.cur_img.shape[:2]
            if 0 <= x < w and 0 <= y < h:
                px = self.cur_img[y:y + 1, x:x + 1]
                bgr = px[0, 0]
                lab = cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0, 0]
                hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, 0]
                print("[%d,%d] BGR=(%d,%d,%d) LAB=(%d,%d,%d) HSV=(%d,%d,%d)"
                      % (x, y, bgr[0], bgr[1], bgr[2],
                         lab[0], lab[1], lab[2], hsv[0], hsv[1], hsv[2]))

    def _load_current(self):
        if not self.images:
            self.cur_img = None
            return None
        p = self.images[self.idx % len(self.images)]
        img = cv2.imread(p)
        if img is None:
            print("[WARN] 读图失败:", p)
            self.cur_img = None
            return None
        self.cur_img = img
        return img

    def build_config(self):
        cfg = {}
        for key, _page, _label, _scale, _tmin, _tbmax, idx in PARAMS:
            if idx is None:
                cfg[key] = self.values[key]
        cfg["BLUE_LAB_MIN"] = self.values["BLUE_LAB_MIN"]
        cfg["BLUE_LAB_MAX"] = self.values["BLUE_LAB_MAX"]
        return cfg

    def save_config(self):
        cfg = self.build_config()
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=1)
        print("已保存", CONFIG_PATH, "(%d 项) -- 记得用 sync 工具同步到机器人" % len(cfg))

    def print_block(self):
        cfg = self.build_config()
        print("---- Python 常量块(可粘贴回主文件) ----")
        for k in sorted(cfg):
            print("%s = %r" % (k, cfg[k]))
        print("---- 结束 ----")

    def reset(self):
        self.values = {}
        for key, _page, _label, scale, tmin, _tbmax, idx in PARAMS:
            if idx is not None:
                if key not in self.values:
                    self.values[key] = list(self.consts[key])
            elif key not in self.values:
                dflt = self.consts.get(key, self.arg_defaults.get(key))
                if dflt is not None:
                    self.values[key] = dflt
        self._build_param_page()
        print("已重置为代码默认值")

    def run(self):
        cv2.namedWindow(WIN_IMG)
        cv2.setMouseCallback(WIN_IMG, self._on_mouse)
        self._build_param_page()
        self._build_tag_window()
        while True:
            img = self._load_current()
            if img is None:
                canvas = np.full((480, 640, 3), 40, dtype=np.uint8)
                cv2.putText(canvas, "no image; use --image <dir>", (60, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                display = canvas
            else:
                cx, cy, area, blue_mask = detect_blue_raw(img, self.values)
                line_cx, vertical, red_mask = detect_red_line_raw(img, self.values, self.roi)
                if self.view == 1:
                    display = cv2.cvtColor(
                        cv2.resize(blue_mask, (img.shape[1], img.shape[0]),
                                   interpolation=cv2.INTER_NEAREST),
                        cv2.COLOR_GRAY2BGR)
                elif self.view == 2:
                    display = cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR)
                    for r in self.roi:
                        cv2.rectangle(display, (r[2], r[0]), (r[3], r[1]), (0, 255, 255), 1)
                else:
                    display = img.copy()
                    if cx >= 0:
                        cv2.circle(display, (cx, cy), 5, (0, 255, 255), -1)
                        cv2.putText(display, "cx=%d cy=%d area=%.0f dx=%d"
                                    % (cx, cy, area, cx - self.values["CENTER_X"]),
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                    (0, 255, 255), 2)
                    cv2.line(display, (int(self.values["CENTER_X"]), 0),
                             (int(self.values["CENTER_X"]), display.shape[0]), (255, 0, 0), 1)
                    y = 60
                    for s in (1, 2, 3, 4):
                        act, done = simulate_color_at(cx, cy, area, self.values, s)
                        cv2.putText(display, "step%d: %s" % (s, act), (10, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                    (0, 255, 0) if done else (255, 255, 255), 1)
                        y += 22
                    cv2.putText(display, "line_cx=%d vertical=%s" % (line_cx, vertical),
                                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
                cv2.putText(display,
                            "view%d page%d [n/b切图 TAB视图 1-6参数页 s存 p打印 r重置]"
                            % (self.view, self.page),
                            (10, display.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            # Tag 模拟窗口
            tdist = cv2.getTrackbarPos("dist(mm)", WIN_TAG) / 1000.0
            tdx = cv2.getTrackbarPos("dx(px)", WIN_TAG)
            tcy = cv2.getTrackbarPos("cy(px)", WIN_TAG)
            tact, tdone = simulate_tag_at(tdist, tdx, tcy, self.values)
            tcanvas = np.full((160, 640, 3), 40, dtype=np.uint8)
            cv2.putText(tcanvas, "Tag sim: dist=%.2fm dx=%d cy=%d -> %s"
                        % (tdist, tdx, tcy, tact), (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if tdone else (0, 165, 255), 2)
            cv2.putText(tcanvas, "dist:0~3m dx:0~640 cy:0~480", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.imshow(WIN_TAG, tcanvas)
            cv2.imshow(WIN_IMG, display)

            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("n"):
                self.idx += 1
            elif key == ord("b"):
                self.idx -= 1
            elif key == 9:
                self.view = (self.view + 1) % 3
            elif key == ord("s"):
                self.save_config()
            elif key == ord("p"):
                self.print_block()
            elif key == ord("r"):
                self.reset()
            elif ord("1") <= key <= ord("6"):
                self.page = key - ord("0")
                self._build_param_page()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="apriltag_sorting_task 阈值调参工具")
    ap.add_argument("--image", help="照片文件或目录(通常是从机器人拉回的快照)")
    ap.add_argument("--task-py", default=TASK_PY, help="主文件路径(提取默认值)")
    ap.add_argument("--dump", action="store_true", help="打印默认配置JSON后退出(无GUI)")
    args = ap.parse_args()

    consts, arg_defaults = load_defaults(args.task_py)
    images = []
    if args.image:
        if os.path.isdir(args.image):
            images = sorted(glob.glob(os.path.join(args.image, "*.jpg"))
                            + glob.glob(os.path.join(args.image, "*.jpeg"))
                            + glob.glob(os.path.join(args.image, "*.png")))
        elif os.path.isfile(args.image):
            images = [args.image]
        else:
            raise SystemExit("找不到图片: %s" % args.image)
    tuner = Tuner(consts, arg_defaults, images)
    if args.dump:
        print(json.dumps(tuner.build_config(), ensure_ascii=False, indent=1))
        return
    tuner.run()


if __name__ == "__main__":
    main()
