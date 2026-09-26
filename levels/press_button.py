# -*- coding: utf-8 -*-
"""机器人智按按钮：**只到"识别出 tag 的四个角"** 为止的骨架。

上面没有任何控制逻辑、没有判据、没有阈值、没有搜索——按用户要求全部删除，
后续由用户自己写。这里只负责三件事：

    1. 拍照落盘（真相机 / PC 图片）
    2. 检测 tag，**只按目标 id 取它**（不挑、不排序、不加工）
    3. 把角点规范成图像坐标的 左上 → 右上 → 右下 → 左下

用到的机器人接口（真机 `core.robot_core.RobotState`）：
    state.capture_image() -> 照片路径（失败返回 None）
    state.detect_apriltag(path) -> apriltag 库的检测结果列表，每项有 .tag_id / .corners

PC 上没有 apriltag 库时，`detect_tags()` 自动退回 cv2.aruco 读同一张照片。

自带一个最小跑法（真机）：
    python levels/press_button.py --probe            # 拍一张，打印检出的 id 与四角

参考：`docs/关卡算法/机器人智按按钮-press_button/` 下的设计与实测记录。
"""

import os
import sys

import numpy as np

# 允许直接运行本文件时找到仓库根
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =====================================================================
# 1. 角点顺序：左上 → 右上 → 右下 → 左下（图像坐标）
# =====================================================================

def order_corners_tl_tr_br_bl(pts):
    """四个角点规范成 左上/右上/右下/左下。

    做法：和最小的点是左上、最大的点是右下；剩下两点按 x 分左右。
    （apriltag 库本身返回的顺序已经是这个约定，见 core/robot_core.TAG_CORNER_PERM
    的真机实测记录；此函数用于 cv2.aruco 兜底路径统一口径。）
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    s = pts.sum(axis=1)
    i_tl, i_br = int(np.argmin(s)), int(np.argmax(s))
    rest = [i for i in range(4) if i not in (i_tl, i_br)]
    i_tr = rest[0] if pts[rest[0]][0] > pts[rest[1]][0] else rest[1]
    i_bl = rest[1] if i_tr == rest[0] else rest[0]
    return pts[[i_tl, i_tr, i_br, i_bl]]


def _corners_usable(corners):
    """是否是 4×2 且有限的角点（畸形输入返回 False，不抛异常）"""
    try:
        arr = np.asarray(corners, dtype=np.float64)
    except Exception:
        return False
    return arr.shape == (4, 2) and bool(np.all(np.isfinite(arr)))


# =====================================================================
# 2. 检测：从 apriltag 结果里取目标 id 的四个角
# =====================================================================

def corners_from_apriltag(results, target_id):
    """apriltag 库结果 → (四角 | None, 本帧检出的全部 id)

    只按 target_id 取；取不到返回 None。不做任何别的加工。
    """
    seen = []
    for r in results:
        try:
            tid = int(r.tag_id)
        except Exception:
            continue
        seen.append(tid)
        if tid == int(target_id):
            corners = getattr(r, "corners", None)
            if not _corners_usable(corners):
                return None, seen
            return np.asarray(corners, dtype=np.float64).reshape(4, 2), seen
    return None, seen


_ARUCO_DETECTOR = None
_ARUCO_NOTICE_PRINTED = False


def corners_from_aruco(frame, target_id):
    """PC 兜底：cv2.aruco 的 DICT_APRILTAG_36h11（与 apriltag 库同一个字典）→ 同一口径的四角"""
    global _ARUCO_DETECTOR, _ARUCO_NOTICE_PRINTED
    import cv2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if _ARUCO_DETECTOR is None:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        params = getattr(cv2.aruco, "DetectorParameters", None)
        params = params() if params is not None else cv2.aruco.DetectorParameters_create()
        _ARUCO_DETECTOR = cv2.aruco.ArucoDetector(dictionary, params)
    corners, ids, _ = _ARUCO_DETECTOR.detectMarkers(gray)
    if ids is None:
        return None, []
    seen = [int(i) for i in np.asarray(ids).flatten()]
    for pts, tid in zip(corners, seen):
        if tid == int(target_id):
            if not _ARUCO_NOTICE_PRINTED:
                _ARUCO_NOTICE_PRINTED = True
                print("[press_button] 使用 cv2.aruco 兜底检测（真机应走 apriltag 库）")
            return order_corners_tl_tr_br_bl(pts), seen
    return None, seen


def detect_tags(state, target_id):
    """拍一张 → 返回 (四角 | None, 本帧全部 id, 照片路径)

    真机链路：state.capture_image() → state.detect_apriltag(path)。
    检测链路不可用或读图失败时，退回 cv2.aruco 读同一张照片。
    """
    path = state.capture_image()
    if path is None:
        return None, [], None

    results = None
    try:
        results = state.detect_apriltag(path)
    except Exception as e:
        print(f"[press_button] apriltag 链路不可用（{type(e).__name__}: {e}），改用 cv2.aruco")

    if results is not None:
        corners, seen = corners_from_apriltag(results, target_id)
        return corners, seen, path

    try:
        import cv2
        frame = cv2.imread(path)
    except Exception as e:
        print(f"[press_button] cv2 不可用（{e}）")
        return None, [], path
    if frame is None:
        print(f"[press_button] 读图失败：{path}")
        return None, [], path
    corners, seen = corners_from_aruco(frame, target_id)
    return corners, seen, path


# =====================================================================
# 3. 从 PC 上的一张图片直接取（离线调参用，不碰机器人）
# =====================================================================

def corners_from_image(path, target_id):
    """读本地图片 → (四角 | None, 本帧全部 id)。PC 上没有 apriltag 库，走 cv2.aruco。"""
    import cv2

    frame = cv2.imread(path)
    if frame is None:
        print(f"[press_button] 读图失败：{path}")
        return None, []
    return corners_from_aruco(frame, target_id)


# =====================================================================
# main.py 需要它；目前是空字典（本关还没接控制逻辑）
# =====================================================================

tag_poses = {}


def run_level(state):
    """占位：控制逻辑待用户自己实现。

    可用的输入已经从 detect_tags() 拿到——每个 tag 的四个角：
        [左上, 右上, 右下, 左下]（图像坐标，像素）
    """
    raise NotImplementedError("press_button 的控制逻辑尚未实现（当前只提供取角点的骨架）")


if __name__ == "__main__":
    if "--probe" in sys.argv:
        idx = sys.argv.index("--probe")
        target = int(sys.argv[idx + 1]) if len(sys.argv) > idx + 1 else 102
        from core.robot_core import RobotState

        st = RobotState(tag_poses={})
        try:
            st.set_head(1500, force=True)
        except TypeError:
            st.current_head_pulse = None
            st.set_head(1500)
        corners, seen, path = detect_tags(st, target)
        print(f"照片: {path}")
        print(f"本帧检出的 id: {seen}")
        if corners is None:
            print(f"未检出目标 id={target}")
        else:
            print(f"目标 id={target} 的四个角（左上/右上/右下/左下，像素）:")
            for name, p in zip(("TL", "TR", "BR", "BL"), corners):
                print("   %s  u=%8.2f  v=%8.2f" % (name, p[0], p[1]))
        raise SystemExit(0)

    print("本模块目前只提供『取 tag 四角』的骨架（控制逻辑待实现）。")
    print("真机取一次角点：python levels/press_button.py --probe [tag_id]")
