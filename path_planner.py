# -*- coding: utf-8 -*-
"""
path_planner.py —— A* 路线层：膨胀地图上的最短路 + 子目标序列（v6 终版）

混合导航架构（已经红队三轮评审通过 + R1-R5 验收条件）中的规划层：
    只回答"走哪些格子、路过哪些门、在哪几个拐角停下转向"，
    不输出动作原语、不含朝向状态——执行由旧 navigate_to_target 完成
    （批量直行/批量转向/倒退/横移/危险逃离全部复用实机验证过的代码）。

场地几何事实（决定参数，2026-08-30 实测确认）：
    全场主走廊净宽 ~40cm（西 x∈[5,45] / 北 y∈[60,100] / 东 x∈[55,95]），
    机身宽 26cm（半宽 13）。40cm 走廊中：
    - 物理中心带 ±7cm（两侧各剩 7cm 间隙）；
    - 硬膨胀 17cm 后规划中心带 ±3cm（2.5cm 格下 2~3 列可行格）；
    - 净空 ≥21cm 的格子不存在 → 软圈是全场常态而非例外。
    由此 v6 的关键设计：软圈罚分只做"居中引导"（不做逼绕开）；
    激进度（批量步数）由执行层按航向对准度自适应，不由地理标签决定。

关键参数与依据（调参表详见计划文档）：
    格 2.5cm 40×40，格心 = 2.5 的倍数（R1：走廊中线 x=25/75、y=80 必须
    是格心，否则走廊内可行格净空全相同 → 代价平局 → 居中失效）；
    硬禁行 <17cm（半宽 13 + 执行误差预算 4）；软圈 17~21cm 代价
    1.5×(21-d)/4（线性分级，通道中心最便宜 → 自动居中，N1 解决）；
    启发 = 欧氏距离（可采纳，1600 格毫秒级无需加权）；
    拐角预检：14cm 扫掠圆（= 转弯扫掠外沿 √(13²+5²)≈13.9）+ 判据
    "扫掠区不碰墙（净空>0）"——旧 20cm/净空≥13 判据在 40cm 走廊会把
    正常拐角全部误杀（圆必然伸进两侧墙内）。失败处置 = 该点标"禁转弯"
    （可经过不可转向），A* 对该拐角追加转弯代价重规划绕开（R3：不拉黑
    经过权，否则东走廊门洞 [72,78]×[77,83] 被圆覆盖 → 全网无解）。

对外接口：
    plan_route(start_xy, goal_xy) -> Route 或 None
    Route: waypoints=[Waypoint(坐标, 语义 pass/corner/goal, 净空)],
           cells=栅格路径, length_cm=折线长
    clearance(xy) -> 净空 cm（供执行层分档/护栏用）
"""

import heapq
import math
from collections import namedtuple

import numpy as np

# =====================================================================
# 场地墙体（与 levels/goodluck.WALLS 同源；独立定义避免反向依赖关卡层）
# =====================================================================
# 矩形 [x_min, x_max, y_min, y_max]；外框底/顶边全宽；左/右边仅 y∈[40,100]
#（y<40 为出入口开口，不约束——入口 x=0 / 出口 x=100）
FIELD_WALLS = [
    [0, 5, 40, 100],      # 左墙
    [45, 55, 0, 60],      # 中墙
    [95, 100, 40, 100],   # 右墙
    [0, 100, 0, 0],       # 外框底边
    [0, 100, 100, 100],   # 外框顶边
    [0, 0, 40, 100],      # 外框左边（入口开口以下无墙）
    [100, 100, 40, 100],  # 外框右边（出口开口以下无墙）
]

# =====================================================================
# 参数（A* 调参表，v6 定值；修改前先读模块 docstring 的依据说明）
# =====================================================================
CELL = 2.5                 # 格边长 cm（走廊中心带 6cm 需 ≥2 列格，5cm 格只剩 1 列）
GRID_N = 40                # 100 / 2.5
INFLATE_HARD = 17.0        # 硬禁行：净空 < 此值（半宽 13 + 执行误差预算 4）
SOFT_OUTER = 21.0          # 软圈外沿（净空 ≥ 此值无罚分）
SOFT_PENALTY = 1.5         # 软圈罚分强度（只做居中引导，无逼绕开职能）
TURN_SWEEP_RADIUS = 14.0   # 拐角预检半径（扫掠外沿 √(13²+5²)≈13.9）
SEGMENT_LEN_CM = 25.0      # 直段子目标间隔
CORNER_TURN_COST = 8.0     # 拐角"禁转弯"标记后的附加转弯代价（迫使 A* 绕开该拐角）

# 子目标语义
#   pass  —— 直段切点：法向越线即切换，不精确到达不微调
#   corner—— 拐点：精确到达（6cm）+ 微调权限 + 预检
#   goal  —— 终点：精确到达（6cm）
Waypoint = namedtuple("Waypoint", ["xy", "semantics", "clearance"])
Route = namedtuple("Route", ["waypoints", "cells", "length_cm"])


# =====================================================================
# 几何基础
# =====================================================================

def _dist_to_rect(px, py, rect):
    """点到矩形最短距离（退化矩形 = 线段也适用）"""
    x0, x1, y0, y1 = rect
    dx = max(x0 - px, 0.0, px - x1)
    dy = max(y0 - py, 0.0, py - y1)
    return math.hypot(dx, dy)


def clearance(x, y):
    """点 (x,y) 到最近墙的净空 cm（执行层分档/护栏共用此语义）"""
    return min(_dist_to_rect(x, y, w) for w in FIELD_WALLS)


def _cell_center(i, j):
    """格心坐标——对齐 2.5 的倍数（R1：走廊中线必须是格心）"""
    return CELL * (i + 1), CELL * (j + 1)


def _to_cell(x, y):
    """连续坐标 → 最近格下标"""
    i = int(round(x / CELL - 1))
    j = int(round(y / CELL - 1))
    return min(max(i, 0), GRID_N - 1), min(max(j, 0), GRID_N - 1)


# =====================================================================
# 地图构建（净空场 + 可行格 + 软圈罚分场）
# =====================================================================

def build_maps():
    """构建净空场与罚分场（格心语义；一次构建，模块级缓存）

    返回 (clear_grid, penalty_grid)：
      clear_grid[i,j]  = 格心净空 cm（<INFLATE_HARD 为不可行）
      penalty_grid[i,j] = 该格额外代价（软圈线性分级；0 = 圈外）
    """
    clear = np.zeros((GRID_N, GRID_N), dtype=np.float64)
    pen = np.zeros((GRID_N, GRID_N), dtype=np.float64)
    for i in range(GRID_N):
        for j in range(GRID_N):
            cx, cy = _cell_center(i, j)
            d = min(_dist_to_rect(cx, cy, w) for w in FIELD_WALLS)
            clear[i, j] = d
            if INFLATE_HARD <= d < SOFT_OUTER:
                pen[i, j] = SOFT_PENALTY * (SOFT_OUTER - d) / (SOFT_OUTER - INFLATE_HARD)
    return clear, pen


_CLEAR, _PEN = build_maps()


def feasible(i, j):
    """格是否可行（净空 ≥ 硬膨胀）"""
    return _CLEAR[i, j] >= INFLATE_HARD


def _nearest_feasible(x, y):
    """最近可行格（起点/终点落在膨胀区内时吸附；不可行返回 None）"""
    i, j = _to_cell(x, y)
    if feasible(i, j):
        return i, j
    best, best_d = None, float("inf")
    for r in range(1, GRID_N):
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                if max(abs(di), abs(dj)) != r:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < GRID_N and 0 <= nj < GRID_N and feasible(ni, nj):
                    cx, cy = _cell_center(ni, nj)
                    d = math.hypot(cx - x, cy - y)
                    if d < best_d:
                        best_d, best = d, (ni, nj)
        if best is not None:
            return best
    return None


# =====================================================================
# 拐角预检（14cm 扫掠圆，判据 = 圆内净空 > 0，即转弯扫掠不碰墙）
# =====================================================================

def corner_precheck(x, y):
    """以 (x,y) 为圆心、TURN_SWEEP_RADIUS 为半径的圆内净空是否 > 0

    实现为净空场网格采样（半径 14 内取 5×5 邻域格的最小净空；
    格 2.5cm，5×5 覆盖 ±5cm 不足——按圆周 16 方向 + 圆心采样近似，
    采样点距 ≤2.5cm 保证不漏 <1.25cm 的净空正区）。
    """
    if clearance(x, y) <= 0:
        return False
    # 圆周 + 内部环形采样：半径 14 太大用 16 方向圆周 + 8 方向半径 7
    for r in (TURN_SWEEP_RADIUS, TURN_SWEEP_RADIUS / 2):
        for k in range(16):
            a = math.radians(k * 22.5)
            px, py = x + r * math.cos(a), y + r * math.sin(a)
            if clearance(px, py) <= 0:
                return False
    return True


# =====================================================================
# A* 核心（无朝向状态；对角移动防穿角）
# =====================================================================

def _astar(start_cell, goal_cell, blocked_corners=None):
    """纯几何 A*。blocked_corners: set[(i,j)] 拐角禁转弯格（附加转弯代价）。

    对角移动要求两侧直邻格也可行（防斜穿墙角）。
    返回格序列 [(i,j),...]（含起终点），不可达返回 None。
    """
    blocked = blocked_corners or set()
    si, sj = start_cell
    gi, gj = goal_cell
    gx, gy = _cell_center(gi, gj)

    def h(i, j):
        return math.hypot(gx - _cell_center(i, j)[0], gy - _cell_center(i, j)[1]) / CELL

    open_heap = [(h(si, sj), 0.0, (si, sj))]
    best_g = {(si, sj): 0.0}
    parent = {}
    closed = set()
    NEIGH = [(-1, 0), (1, 0), (0, -1), (0, 1),
             (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while open_heap:
        _, g, state = heapq.heappop(open_heap)
        if state in closed:
            continue
        closed.add(state)
        i, j = state
        if (i, j) == (gi, gj):
            cells = []
            s = state
            while s is not None:
                cells.append(s)
                s = parent.get(s)
            cells.reverse()
            return cells
        for di, dj in NEIGH:
            ni, nj = i + di, j + dj
            if not (0 <= ni < GRID_N and 0 <= nj < GRID_N) or not feasible(ni, nj):
                continue
            if di != 0 and dj != 0:
                # 对角防穿角：两侧直邻格也须可行
                if not (feasible(i + di, j) and feasible(i, j + dj)):
                    continue
                step = math.sqrt(2.0)
            else:
                step = 1.0
            cost = step + _PEN[ni, nj]
            if (ni, nj) in blocked:
                cost += CORNER_TURN_COST  # 禁转弯拐角：重罚使 A* 绕开
            ns = (ni, nj)
            ng = g + cost
            if ng < best_g.get(ns, float("inf")):
                best_g[ns] = ng
                parent[ns] = state
                heapq.heappush(open_heap, (ng + h(ni, nj), ng, ns))
    return None


# =====================================================================
# 路径 → 子目标序列
# =====================================================================

def _rdp(points, epsilon):
    """Ramer-Douglas-Peucker 折线简化（points 为 (x,y) 序列）"""
    if len(points) <= 2:
        return list(points)
    ax, ay = points[0]
    bx, by = points[-1]
    dx, dy = bx - ax, by - ay
    seg_len = math.hypot(dx, dy)
    best_d, best_i = -1.0, -1
    for k in range(1, len(points) - 1):
        px, py = points[k]
        if seg_len < 1e-9:
            d = math.hypot(px - ax, py - ay)
        else:
            # 点到直线距离
            d = abs((px - ax) * dy - (py - ay) * dx) / seg_len
        if d > best_d:
            best_d, best_i = d, k
    if best_d <= epsilon:
        return [points[0], points[-1]]
    left = _rdp(points[:best_i + 1], epsilon)
    right = _rdp(points[best_i:], epsilon)
    return left[:-1] + right


def _simplify(cells, epsilon_cm=2.5):
    """RDP 折线简化：压平 45° 锯齿（幅度 ~1.8cm < ε）、保留 90° 转角

    角度阈值法对此路径失效（45° 锯齿把 90° 转折分解成两个 45°，
    阈值 <90° 会融合真转角、≥90° 无法压平锯齿），故用 RDP。
    ε=2.5cm = 一格边长：锯齿被压平，真转折顶点保留。
    返回拐点格序列（保留首尾；格吸附——拐点必在原路径格上，净空 ≥17）。
    """
    if len(cells) <= 2:
        return list(cells)
    pts = [_cell_center(i, j) for i, j in cells]
    simplified = _rdp(pts, epsilon_cm)
    # 顶点吸附回最近的原始格（保证拐点净空 ≥ 硬膨胀）
    cell_set = {c: c for c in cells}
    out = []
    for (x, y) in simplified:
        i, j = _to_cell(x, y)
        if (i, j) in cell_set:
            out.append((i, j))
        else:
            # 搜索最近的原路径格
            best, best_d = None, float("inf")
            for (ci, cj) in cells:
                cx, cy = _cell_center(ci, cj)
                d = math.hypot(cx - x, cy - y)
                if d < best_d:
                    best_d, best = d, (ci, cj)
            out.append(best)
    # 保持顺序且去重（RDP 顶点吸附后可能重复）
    seen = []
    for c in out:
        if not seen or seen[-1] != c:
            seen.append(c)
    return seen


def _split_segment(p0, p1):
    """直段子目标切分：整段一个 pass 子目标（调度层批量截短保证不冲过段末），
    仅超长直段（>45cm，批量上限 30cm 走不完）保留中切点。
    切点吸附最近可行格心（净空 ≥17）。返回切点坐标列表（不含端点）。
    """
    x0, y0 = _cell_center(*p0)
    x1, y1 = _cell_center(*p1)
    length = math.hypot(x1 - x0, y1 - y0)
    if length <= 45.0:
        return []
    n_cut = int(length // SEGMENT_LEN_CM)
    pts = []
    for k in range(1, n_cut + 1):
        t = k * SEGMENT_LEN_CM / length
        if t >= 1.0:
            break
        px, py = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
        i, j = _to_cell(px, py)
        if feasible(i, j):
            cx, cy = _cell_center(i, j)
            if math.hypot(cx - px, cy - py) <= CELL * 1.5:
                pts.append((cx, cy))
                continue
        best, best_d = None, float("inf")
        for r in range(1, 4):
            found = False
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    if max(abs(di), abs(dj)) != r:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < GRID_N and 0 <= nj < GRID_N and feasible(ni, nj):
                        cx, cy = _cell_center(ni, nj)
                        d = math.hypot(cx - px, cy - py)
                        if d < best_d:
                            best_d, best = d, (cx, cy)
            if best is not None:
                break
        if best is not None and best_d <= CELL * 2.0:
            pts.append(best)
    return pts


def plan_route(start_xy, goal_xy):
    """主接口：起点 → 终点的子目标序列（含拐角预检与禁转弯重规划）

    返回 Route(waypoints, cells, length_cm)；不可达返回 None。
    waypoint 语义：pass（直段切点，越线即切换）/ corner（拐点，精确到达）
    / goal（终点，精确到达）。
    """
    start = _nearest_feasible(*start_xy)
    goal = _nearest_feasible(*goal_xy)
    if start is None or goal is None:
        return None

    blocked = set()
    for _ in range(4):  # 最多 4 轮：每轮预检失败禁一个拐角后重规划
        cells = _astar(start, goal, blocked_corners=blocked)
        if cells is None:
            return None
        corners = _simplify(cells)
        # 预检所有拐角（首尾不算——首是起点附近，尾由 goal 语义处理）
        bad = None
        for p in corners[1:-1]:
            cx, cy = _cell_center(*p)
            if not corner_precheck(cx, cy):
                bad = p
                break
        if bad is None:
            break
        blocked.add(bad)  # 禁转弯（可经过），A* 以附加代价绕开
    else:
        return None
    if cells is None:
        return None

    # 折线 → 子目标序列（起点不输出：它是机器人当前位置，不是目标）
    corners = _simplify(cells)
    waypoints = []
    for idx, p in enumerate(corners):
        cx, cy = _cell_center(*p)
        if idx == 0:
            continue  # 起点跳过
        if idx == len(corners) - 1:
            semantics = "goal"
        else:
            semantics = "corner"
        waypoints.append(Waypoint((cx, cy), semantics, clearance(cx, cy)))
        if idx < len(corners) - 1:
            for cut in _split_segment(p, corners[idx + 1]):
                waypoints.append(Waypoint(cut, "pass", clearance(*cut)))

    length = 0.0
    pts = [_cell_center(i, j) for i, j in cells]
    length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))
    return Route(waypoints, cells, length)


# =====================================================================
# 执行层辅助（分档/护栏用的净空查询）
# =====================================================================

def route_offset(xy, wp_a, wp_b):
    """点 xy 到线段 wp_a→wp_b 的（横向偏移, 弧长投影）——护栏与进度用

    返回 (lateral_cm, along_cm)。lateral 符号 = 相对路径方向的左侧为正。
    """
    ax, ay = wp_a
    bx, by = wp_b
    px, py = xy
    dx, dy = bx - ax, by - ay
    seg2 = dx * dx + dy * dy
    if seg2 < 1e-9:
        return math.hypot(px - ax, py - ay), 0.0
    t = ((px - ax) * dx + (py - ay) * dy) / seg2
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * dx, ay + t * dy
    lateral = (px - cx) * (-dy) + (py - cy) * dx  # 未归一化的叉积侧偏
    lateral = lateral / math.sqrt(seg2)
    along = t * math.sqrt(seg2)
    return lateral, along
