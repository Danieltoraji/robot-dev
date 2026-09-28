#!/usr/bin/python3
# coding=utf8
"""
goal_line_judge.py — 足球整体越过球门线判断

这是一个独立的、无机器人硬件依赖的几何判断模块。
它不控制舵机、不执行踢球动作，也不修改 finalkick.py 的状态机。

输入：
    ball      : (cx, cy, w, h[, conf])，足球 YOLO 检测结果
    goalposts : [(cx, cy, w, h[, conf]), ...]，球门柱 YOLO 检测结果

标准：
    用两个球门柱检测框的底部中心连成球门线，并将该线无限延长。
    足球检测模型目前只提供矩形框，因此采用保守标准：
    检测框四个角全部越过球门线，才认为足球已经完全越线；
    这是当前任务采用的成功标准。partial_crossed 仍作为调试信息保留。
"""

from math import hypot


__all__ = ["check_ball_crossed_goal_line"]


def _post_geometry(post):
    """读取一个球门柱检测结果，兼容 tuple/list 和 dict。"""
    if isinstance(post, dict):
        return (
            float(post["cx"]),
            float(post["cy"]),
            float(post["w"]),
            float(post["h"]),
        )
    if len(post) < 4:
        raise ValueError("goalpost detection must contain cx, cy, w and h")
    return float(post[0]), float(post[1]), float(post[2]), float(post[3])


def _estimate_goal_line(goalposts, image_width):
    """根据球门柱底部中心估计球门线及其左右延长端点。"""
    valid = []
    for post in goalposts or []:
        try:
            cx, cy, width, height = _post_geometry(post)
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        if width > 0 and height > 0:
            valid.append((cx, cy, width, height))

    if len(valid) < 2:
        return None

    # 多个候选框时取横向距离最大的一对，尽量避免重复框影响结果。
    p1, p2 = max(
        ((a, b) for i, a in enumerate(valid) for b in valid[i + 1:]),
        key=lambda pair: abs(pair[0][0] - pair[1][0]),
    )
    if p1[0] > p2[0]:
        p1, p2 = p2, p1

    x1 = p1[0]
    y1 = p1[1] + p1[3] / 2.0
    x2 = p2[0]
    y2 = p2[1] + p2[3] / 2.0
    dx = x2 - x1
    dy = y2 - y1
    norm = hypot(dx, dy)
    if norm < 1.0 or abs(dx) < 1e-9:
        # 当前实现默认球门线可以从画面左边延长到右边。
        # 近似竖直的线无法可靠判断 above/below，因此放弃本帧。
        return None

    left_y = y1 + (0.0 - x1) * dy / dx
    right_y = y1 + (float(image_width - 1) - x1) * dy / dx

    return {
        "p1": (round(x1), round(y1)),
        "p2": (round(x2), round(y2)),
        "extended_p1": (0, round(left_y)),
        "extended_p2": (image_width - 1, round(right_y)),
        "x1": x1,
        "y1": y1,
        "dx": dx,
        "dy": dy,
        "norm": norm,
    }


def _signed_distance(point, line):
    """
    计算点到球门线的带符号距离。

    球门柱已经按图像 x 坐标从左到右排列。对于近似水平的球门线：
        正值 = 图像下方
        负值 = 图像上方
    """
    px, py = point
    return (
        (px - line["x1"]) * (-line["dy"])
        + (py - line["y1"]) * line["dx"]
    ) / line["norm"]


def check_ball_crossed_goal_line(
    ball,
    goalposts,
    *,
    image_width=640,
    goal_line=None,
    goal_line_source=None,
    goal_side="above",
    margin_px=3.0,
    previous_streak=0,
    confirm_frames=3,
):
    """
    判断足球检测框是否整体越过球门线及其延长线。

    参数：
        ball:
            足球检测结果 (cx, cy, w, h[, conf])；没有检测到时传 None。
        goalposts:
            球门柱检测结果列表，每项为 (cx, cy, w, h[, conf])。
        image_width:
            输入图像宽度，用于计算球门线的左右延长端点。
        goal_line:
            可选的外部球门线几何结果。测试网页可传入红色像素拟合线；
            不传时继续使用两个球门柱底部中心估计。
        goal_line_source:
            外部球门线来源名称，仅用于返回调试信息。
        goal_side:
            球门内部位于图像的哪一侧，只支持 "above" 或 "below"。
        margin_px:
            检测框四个角至少离球门线多少像素才算越过。
        previous_streak:
            上一帧的整体越线连续帧数，由调用方保存。
        confirm_frames:
            连续多少帧整体越线后将 confirmed 置为 True。

    返回：
        dict，包括：
            line_valid       : 是否成功估计球门线
            partial_crossed  : 是否有检测框角点已到球门另一侧
            whole_crossed    : 检测框四个角是否全部到另一侧
            confirmed        : 按连续帧规则是否确认
            streak           : 当前连续整体越线帧数
            line             : 球门线和延长线坐标
            bbox             : 足球检测框左上、右下坐标
            signed_distances : 四个框角到球门线的带符号距离
            line_source     : 球门线来源
    """
    if image_width < 2:
        raise ValueError("image_width must be at least 2")
    if goal_side not in ("above", "below"):
        raise ValueError("goal_side must be 'above' or 'below'")
    if margin_px < 0:
        raise ValueError("margin_px must be non-negative")
    if confirm_frames < 1:
        raise ValueError("confirm_frames must be at least 1")

    if goal_line is not None:
        line = goal_line
        line_source = goal_line_source or "provided"
    else:
        line = _estimate_goal_line(goalposts, image_width)
        line_source = goal_line_source or "goalposts"

    result = {
        "line_valid": line is not None,
        "line_source": line_source if line is not None else "unavailable",
        "partial_crossed": False,
        "whole_crossed": False,
        "confirmed": False,
        "streak": 0,
        "line": line,
        "bbox": None,
        "corners": [],
        "signed_distances": [],
    }

    if ball is None or line is None or len(ball) < 4:
        return result

    cx, cy, width, height = map(float, ball[:4])
    if width <= 0 or height <= 0:
        return result

    left = cx - width / 2.0
    top = cy - height / 2.0
    right = cx + width / 2.0
    bottom = cy + height / 2.0
    corners = [(left, top), (right, top), (right, bottom), (left, bottom)]
    distances = [_signed_distance(point, line) for point in corners]

    result["bbox"] = (round(left), round(top), round(right), round(bottom))
    result["corners"] = corners
    result["signed_distances"] = distances

    if goal_side == "above":
        on_goal_side = [distance <= -margin_px for distance in distances]
    else:
        on_goal_side = [distance >= margin_px for distance in distances]

    result["partial_crossed"] = any(on_goal_side)
    result["whole_crossed"] = all(on_goal_side)

    if result["whole_crossed"]:
        result["streak"] = max(0, int(previous_streak)) + 1
    else:
        result["streak"] = 0
    result["confirmed"] = result["streak"] >= confirm_frames
    return result
