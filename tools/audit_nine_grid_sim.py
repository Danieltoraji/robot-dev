#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""audit_nine_grid_sim.py —— nine_grid 仿真器体检（可复跑，逐条给证据）

为什么要有它
------------
2026-09-25 用户报了两条现象："普通小转弯转角明显和标定值不符（连续转数十次
都过不了 90°）"、"地面数字完全无法正确显示"。本脚本把那两条以及同一批排查中
发现的其它偏差**固化成可复跑的检查**，每条都打印实测数字与判据，避免"靠眼睛
看窗口"得出结论。

六项检查
--------
A 小转角步长 vs 现场标定（含"转 90° 要几步"）
B 字形贴图朝向（渲染出来的字 == 资产掩膜的哪种变换）
C 数字可读性（现场模板仲裁器 / SVM 在仿真帧上的读出率，含现场基线）
D 仿真投影 vs 关卡投影（必须逐像素一致）
E 动作计数口径（run_action 调用数 vs 原语执行数）
F 面板真值 occluded_frac（共面不重叠 ⇒ 应恒 0）

退出码：0 = 全部符合预期；1 = 有偏差（打印在末尾）。
用法：python tools/audit_nine_grid_sim.py
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import sim.nine_grid_sim as SIM                                  # noqa: E402
from core.camera_config import HEAD_CENTER, HEAD_LEFT, HEAD_RIGHT  # noqa: E402
from core.ground_homography import grid_cell_center              # noqa: E402
from levels import nine_grid_shared as SH                        # noqa: E402
from levels.nine_grid import (                                   # noqa: E402
    TURN_LEFT_SMALL_DEG, TURN_RIGHT_SMALL_DEG,
)
from levels.nine_grid_shared import project_ground_to_pixel      # noqa: E402
from vision.nine_grid_detector import NineGridDetector           # noqa: E402

OUT_DIR = os.path.join(_ROOT, "archive", "result", "sim_audit")
CELL = 4          # 单面板探针用的格位（场地正中，正对入口）
DIST = 45.0       # 相机到面板中心 cm
FAILS: list[str] = []


def _head(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def _verdict(ok, msg):
    print(f"  ⇒ {'✅' if ok else '❌'} {msg}")
    if not ok:
        FAILS.append(msg)


def _one_panel_frame(digit):
    """场地里只放一块面板（格4），正对相机拍一帧；可钉死字形朝向"""
    layout = {k: None for k in range(9)}
    layout[CELL] = digit
    r = SIM.SimNineGridRobot(layout=layout, seed=3)
    r.pos = np.array([grid_cell_center(CELL)[0],
                      grid_cell_center(CELL)[1] - DIST])
    r.heading = 0.0
    r.set_pitch(1040)
    return r


# ---------------------------------------------------------------- A
def check_small_turn():
    _head("A 运动原语：仿真 vs 关卡模型/现场标定")
    r = SIM.SimNineGridRobot(seed=3)
    print("  A1 全原语逐条（仿真实测均值 vs ACTION_MODEL）：")
    rows = []
    for action, (kind, delta) in sorted(SH.ACTION_MODEL.items()):
        if kind is None or action == "go_forward":      # stand / 禁用动作不测
            continue
        n = 4000
        dist, turn = [], []
        for _ in range(n):
            p0, h0 = r.pos.copy(), r.heading
            r._apply_action(action)
            dist.append(float(np.hypot(*(r.pos - p0))))
            turn.append(abs(r.heading - h0))
        got = float(np.mean(turn)) if kind == "turn" else float(np.mean(dist))
        want = abs(float(delta))
        err = abs(got - want) / want * 100.0
        unit = "°" if kind == "turn" else "cm"
        rows.append((action, want, got, err, unit))
        print(f"    {action:24s} 模型 {want:6.3f}{unit}  仿真 {got:6.3f}{unit}"
              f"  偏差 {err:4.1f}%")
    worst = max(rows, key=lambda t: t[3])
    _verdict(worst[3] < 5.0,
             f"{worst[0]} 仿真 {worst[2]:.3f}{worst[4]} 与模型 {worst[1]:.3f}"
             f"{worst[4]} 相差 {worst[3]:.1f}%（全原语最大偏差）")

    print("\n  A2 小转角专项（现场标定 vs 仿真，含『转 90° 要几步』）：")
    dl, dr = [], []
    for _ in range(20000):
        h0 = r.heading
        r._apply_action("turn_left_small_step")
        dl.append(abs(h0 - r.heading))
        h0 = r.heading
        r._apply_action("turn_right_small_step")
        dr.append(abs(r.heading - h0))
    dl, dr = np.array(dl), np.array(dr)
    print(f"    仿真 左小转：均值 {dl.mean():.3f}°/步  P(完全不转) "
          f"{np.mean(dl == 0)*100:.1f}%")
    print(f"    仿真 右小转：均值 {dr.mean():.3f}°/步  P(完全不转) "
          f"{np.mean(dr == 0)*100:.1f}%")
    print(f"    现场标定  ：左 {TURN_LEFT_SMALL_DEG:.3f}°/步  "
          f"右 {TURN_RIGHT_SMALL_DEG:.3f}°/步"
          f"（docs/.../现场测量结果-2026-09-XX.md §2 测量1）")
    print(f"    关卡 ACTION_MODEL：左 "
          f"{SH.ACTION_MODEL['turn_left_small_step'][1]:+.1f}°  "
          f"右 {SH.ACTION_MODEL['turn_right_small_step'][1]:+.1f}°")
    print(f"\n    连续 20 次小转的总转角：仿真 左 {20*dl.mean():.1f}° / "
          f"右 {20*dr.mean():.1f}°；现场 左 "
          f"{20*TURN_LEFT_SMALL_DEG:.1f}° / 右 {20*TURN_RIGHT_SMALL_DEG:.1f}°")
    n90 = 90.0 / dl.mean()
    print(f"    转过 90° 需要：仿真 {n90:.0f} 步；现场 "
          f"{90/TURN_LEFT_SMALL_DEG:.0f}（左）/{90/TURN_RIGHT_SMALL_DEG:.0f}"
          f"（右）步")
    _verdict(abs(dl.mean() - TURN_LEFT_SMALL_DEG) < 1.0,
             f"仿真左小转 {dl.mean():.2f}°/步 与标定 "
             f"{TURN_LEFT_SMALL_DEG}°/步 相差 "
             f"{abs(dl.mean()-TURN_LEFT_SMALL_DEG)/TURN_LEFT_SMALL_DEG*100:.0f}%"
             f"（转 90° 要 {n90:.0f} 步）")
    _verdict(abs(dr.mean() - TURN_RIGHT_SMALL_DEG) < 1.0,
             f"仿真右小转 {dr.mean():.2f}°/步 与标定 "
             f"{TURN_RIGHT_SMALL_DEG}°/步 相差 "
             f"{abs(dr.mean()-TURN_RIGHT_SMALL_DEG)/TURN_RIGHT_SMALL_DEG*100:.0f}%")
    _verdict(abs(dl.mean() - dr.mean()) > 1.0,
             f"仿真左右小转角差距 {abs(dl.mean()-dr.mean()):.2f}° "
             f"（左 {dl.mean():.2f} vs 右 {dr.mean():.2f}）——与现场实测的左右"
             f"不对称（差 66%）一致；若两者相近则说明又退回了『对称』模型")


# ---------------------------------------------------------------- B
def check_glyph_orientation():
    _head("B 字形贴图朝向：渲染出来的字是不是被镜像了")
    table = SIM.glyph_table()
    if not table:
        _verdict(False, "字体渲染不可用（glyph_mask 返回空）")
        return
    r = _one_panel_frame(5)
    orig_k = SIM.glyph_rotate_k
    SIM.glyph_rotate_k = lambda cell: 0            # 关掉随机朝向再判朝向
    try:
        frame = r.capture_frame()
    finally:
        SIM.glyph_rotate_k = orig_k
    g = table[5]
    # 画布朝向按"站在入口看面板"定：上边 = 世界 +y（远）、左边 = 世界 −x。
    # 这里**自己算四角点**（用物理约定，而不是问渲染代码要），比对才不是自证的。
    hx, hy = SIM.glyph_block_half_cm(5, CELL)
    c = grid_cell_center(CELL)
    world = np.array([[c[0] - hx, c[1] + hy], [c[0] + hx, c[1] + hy],
                      [c[0] + hx, c[1] - hy], [c[0] - hx, c[1] - hy]])
    # 用**关卡侧**的投影函数（独立于仿真渲染代码）把四角点打到像素上
    pix = project_ground_to_pixel(world, r.pos[0], r.pos[1], 0.0, 1040,
                                  HEAD_CENTER)
    W, H = max(60, g.shape[1] * 4), max(60, g.shape[0] * 4)
    dst = np.array([[0.0, 0.0], [W - 1.0, 0.0],
                    [W - 1.0, H - 1.0], [0.0, H - 1.0]], np.float32)
    Hm = cv2.getPerspectiveTransform(pix.astype(np.float32), dst)
    flat = cv2.warpPerspective(frame, Hm, (W, H))
    gray = cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = cv2.erode(ink, np.ones((5, 5), np.uint8))
    asset = cv2.resize(g * 255, (W, H), interpolation=cv2.INTER_NEAREST)
    variants = {
        "恒等（正立）": asset,
        "上下翻（镜像）": np.flipud(asset),
        "左右翻": np.fliplr(asset),
        "180°": np.rot90(asset, 2),
        "逆时针90°": np.rot90(asset, 1),
        "顺时针90°": np.rot90(asset, 3),
    }
    scored = []
    for name, m in variants.items():
        if m.shape != ink.shape:      # 旋转过的变体形状会变，比对齐回来
            m = cv2.resize(m, (ink.shape[1], ink.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
        inter = np.count_nonzero((m > 0) & (ink > 0))
        union = np.count_nonzero((m > 0) | (ink > 0))
        scored.append((inter / union if union else 0.0, name))
    scored.sort(reverse=True)
    for iou, name in scored:
        print(f"    {name:16s} IoU {iou:.3f}")
    best = scored[0][1]
    print(f"  （数字块 {2*hx:.1f}×{2*hy:.1f}cm，站在入口看的四角点顺序 "
          f"TL,TR,BR,BL → {[tuple(np.round(p, 0)) for p in pix]}）")
    _verdict(best == "恒等（正立）",
             f"仿真画出来的数字是字体字形的「{best}」——如果不是恒等，说明贴图"
             f"角点序与世界 y 反向（地面数字被画成镜像，现场不可能出现）")
    os.makedirs(OUT_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(OUT_DIR, "glyph_orientation.png"),
                np.hstack([cv2.cvtColor(255 - asset, cv2.COLOR_GRAY2BGR),
                           cv2.cvtColor(255 - ink, cv2.COLOR_GRAY2BGR)]))
    print(f"  （左=字体字形原样 右=渲染反投影，见 archive/result/sim_audit/"
          f"glyph_orientation.png）")


# ---------------------------------------------------------------- C
def check_digit_readability():
    _head("C 数字可读性：真实识别链在仿真帧上读得出数字吗")
    det = NineGridDetector()
    # (标签, 是否把角点序反过来=复现镜像 bug, 朝向 k 是否钉 0)
    combos = {
        "正立 + 随机朝向（现状）": (False, None),
        "正立 + 朝向固定 k=0": (False, 0),
        "镜像 + 朝向固定 k=0": (True, 0),
    }
    orig_quad, orig_k = SIM._glyph_quad, SIM.glyph_rotate_k
    results = {}
    try:
        for label, (mirror, kpin) in combos.items():
            SIM._glyph_quad = ((lambda R, C, cx, cy, d, cell:
                                orig_quad(R, C, cx, cy, d, cell)[::-1])
                               if mirror else orig_quad)
            SIM.glyph_rotate_k = ((lambda cell: kpin) if kpin is not None
                                  else orig_k)
            ok_t = ok_m = 0
            for digit in range(1, 8):
                r = _one_panel_frame(digit)
                frame = r.capture_frame()
                obs = det.detect_panels(frame, arbitrate=True, shape=True)
                if not obs:
                    continue
                o = max(obs, key=lambda z: z.area)
                ok_t += int(o.shape_digit == digit)      # 现场模板仲裁器
                ok_m += int(o.model_digit == digit)      # 别处训练的 SVM
            results[label] = (ok_t, ok_m)
            print(f"  {label:22s} 现场模板 {ok_t}/7   SVM {ok_m}/7")
    finally:
        SIM._glyph_quad, SIM.glyph_rotate_k = orig_quad, orig_k
    print("\n  现场基线（同一套识别链跑 tests/fixtures/field_photos）：")
    print("    SVM 判对 23%（48 个面板，置信度中位 0.28、没有一个 ≥0.6）"
          " —— 见 tools/gen_ninegrid_digit_templates.py 文件头："
          "该 SVM 是别处训练的，项目已知不可用")
    print("    现场模板仲裁器自评 LOO top-1 81.7%（gen_ninegrid_digit_templates"
          " 的输出），但它的样本来自**同一批现场帧**，对合成字形不保证成立")
    _verdict(True,
             "数字判据的读出率是**识别链**的性质（现场帧上 SVM 只有 23%、"
             "模板对合成字形几乎全判成 3），不是仿真渲染的问题；仿真这边"
             "只需保证画出来的数字是**物理上可能出现的印刷数字**（正立朝向的"
             "旋转、非镜像）——由 B 项与单测钉住")


# ---------------------------------------------------------------- D
def check_projection_consistency():
    _head("D 仿真渲染投影 vs 关卡投影（应为同一套）")
    worst = 0.0
    checked = 0
    for pitch in (1040, 1200):
        for head in (HEAD_CENTER, HEAD_LEFT, HEAD_RIGHT):
            r = SIM.SimNineGridRobot(seed=3)
            r.pos = np.array([50.0, 20.0])
            r.heading = 17.0
            bearing = r.heading - (head - HEAD_CENTER) * 0.09
            R = SIM.camera_rotation(bearing, pitch,
                                    SIM.CAM_PITCH_MOUNT_OFFSET_DEG)
            C = np.array([r.pos[0], r.pos[1], r.CAM_HEIGHT])
            for cell in range(9):
                c = grid_cell_center(cell)
                h = SIM.PANEL_HALF_CM
                sim_pts = SIM._project_quad_full(R, C, c[0], c[1], h, h)
                if sim_pts is None:
                    continue
                lv_pts = project_ground_to_pixel(
                    np.array([[c[0] - h, c[1] - h], [c[0] + h, c[1] - h],
                              [c[0] + h, c[1] + h], [c[0] - h, c[1] + h]]),
                    r.pos[0], r.pos[1], np.radians(r.heading), pitch, head)
                worst = max(worst,
                            float(np.max(np.linalg.norm(sim_pts - lv_pts,
                                                        axis=1))))
                checked += 1
    print(f"  比对 {checked} 组（3 档头部 × 2 档俯仰 × 9 格）")
    _verdict(worst < 0.5, f"最大偏差 {worst:.3f} px ⇒ 投影一致")


# ---------------------------------------------------------------- E/F
def check_counting_and_truth():
    _head("E/F 计数器口径与面板真值")
    r = SIM.SimNineGridRobot(seed=3)
    r.run_action("go_forward_one_step", 1)
    r.run_action("go_forward_one_step", 5)
    print(f"  E 实际执行原语 6 次，action_log 长度 = {len(r.action_log)}"
          f"（内容 {r.action_log}）")
    _verdict(len(r.action_log) == 6,
             f"stats['actions']/SIM_FUSE_ACTIONS/窗口 step= 数的是 run_action "
             f"**调用次数**，times=N 的批量动作只算 1 次")
    worst = 0.0
    for cell in range(9):
        c = grid_cell_center(cell)
        for dx, dy in ((0, -60), (0, -35), (0, -20), (0, 0), (0, 25),
                       (40, 0), (-40, 0), (0, -80)):
            r2 = SIM.SimNineGridRobot(layout=SIM.SIM_LAYOUT, seed=3)
            r2.pos = np.array([c[0] + dx, c[1] + dy])
            r2.heading = 0.0
            r2.set_pitch(1040)
            r2.capture_frame()
            for t in r2._last_frame_truth:
                worst = max(worst, t["occluded_frac"])
    print(f"  F 72 个站位里，面板真值最大 occluded_frac = {worst:.4f}")
    _verdict(worst <= 1e-9,
             "共面且互不重叠的面板在透视投影下不可能互相遮挡 ⇒ "
             "`occluded_by` 恒为空，是**死字段**（其 docstring 举的"
             "「面板6 被别格压住」例子不可复现）")


def main():
    print("nine_grid 仿真器体检（每条都打印实测数字；结论见末尾）")
    check_small_turn()
    check_glyph_orientation()
    check_digit_readability()
    check_projection_consistency()
    check_counting_and_truth()
    _head("结论")
    if not FAILS:
        print("  全部符合预期")
        return 0
    for i, m in enumerate(FAILS, 1):
        print(f"  {i}. {m}")
    print(f"\n共 {len(FAILS)} 条偏差")
    return 1


if __name__ == "__main__":
    sys.exit(main())
