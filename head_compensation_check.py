#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
head_compensation_check.py —— 头部转动定位补偿有效性验证（真机）

原理：
    转头不改变机体位置与朝向。机器人固定不动，头部依次转到 7 个档位
    （回正 / 右浅 / 右中 / 右深 / 左浅 / 左中 / 左深，均在相机水平半视场
    ≈33.7° 内），每个档位拍照定位：
      - 补偿后的「机体朝向」在所有头位下应一致（差异 <5°）；
      - 「相机位置」会因相机偏心而移动（位置差 ≈ 2·d·|sin(θ/2)|），
        脚本按每个头位反推偏心距离 d，供位置补偿标定使用；
      - 多点位可画出「朝向差 vs 转角」趋势：斜率 ≈0 补偿正确；
        斜率 ≈1 说明实际转角与标称不符；斜率 ≈2 说明方向约定反了。
    reproj > 2.0px 的档位判为坏解（2026-08-25 实测分隔线），不参与判定。

用法（机器人项目根目录）：
    python head_compensation_check.py            # 每档位 1 次
    python head_compensation_check.py --repeat 3 # 每档位 3 次取中值

输出：
    每个头位一行数据（先），再输出结论（后）；
    日志保存 result/head_comp_check_<时间戳>.txt

判定标准（汇总段结论）：
    朝向差/转角 比例 r = median(朝向差 / |标称转角|)：
      r < 0.15  → 补偿方向正确、标称角度基本准确
      0.15~0.6 → 标称角度不准（舵机实际转角 ≠ 标称，需修正系数）
      r > 0.8   → 「脉宽→转向」方向约定反了（补偿符号需反转）
    位置差 ≥2cm → 相机偏心明显，按反推值做位置补偿
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np

try:
    import cv2
    from robot_core import (RobotState, HEAD_CENTER, HEAD_RIGHT, HEAD_LEFT,
                            TAG_CORNER_PERM, solve_pnp_pose, pnp_pose_problems)
    from levels.goodluck import tag_poses
except Exception as e:
    print(f"[head-check] 导入失败: {e}")
    print("[head-check] 请在机器人项目根目录运行")
    sys.exit(1)


SERVO_DEG_PER_US = 0.09

# 头位表：名称, 脉宽, 标称转角（度）。左右各 3 档 + 回正。
# 档位限制在相机水平半视场(≈33.7°)内：转头 ±27° 后回正时看到的标签仍在
# 画面内，保证各档位对同一标签定位、质量一致，对比才有意义。
# 深档(±54°)曾导致标签出视野、定位质量断崖，故弃用。
HEAD_POSES = [
    ("回正", HEAD_CENTER, 0.0),
    ("右浅", 1400, (1400 - HEAD_CENTER) * SERVO_DEG_PER_US),
    ("右中", 1300, (1300 - HEAD_CENTER) * SERVO_DEG_PER_US),
    ("右深", 1200, (1200 - HEAD_CENTER) * SERVO_DEG_PER_US),
    ("左浅", 1600, (1600 - HEAD_CENTER) * SERVO_DEG_PER_US),
    ("左中", 1700, (1700 - HEAD_CENTER) * SERVO_DEG_PER_US),
    ("左深", 1800, (1800 - HEAD_CENTER) * SERVO_DEG_PER_US),
]

# 坏解判定：重投影误差超过此值（与 robot_core.PNP_REPROJ_ERR_MAX_PX 对齐）的行
# 判为坏解，不参与朝向/偏心判定（2026-08-25 实测：好解 ≤0.92px，坏解 ≥2.79px）
GOOD_REPROJ_PX = 2.0

HEAD_OK_DEG = 5.0      # 朝向差通过阈值（°）
POS_OK_CM = 2.0        # 位置差通过阈值（cm）
RATIO_DIR_FLIP = 0.8   # 朝向差/转角 比例 ≥ 此值判为方向约定反了
RATIO_SCALE_BAD = 0.15 # 比例 ≥ 此值判为标称角度不准


class TeeWriter:
    """stdout 同时输出到终端和日志文件"""

    def __init__(self, file_path):
        self.file = open(file_path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def locate_once(state):
    """拍照 + 检测 + PnP + 门控，返回 (pos_3d, ori_3d, reproj) 或 (None, 失败原因)"""
    filename = state.capture_image()
    if filename is None:
        return None, "拍照失败"
    dets = state.detect_apriltag(filename)
    known = [d for d in dets if str(d.tag_id) in tag_poses]
    if not known:
        return None, f"无已知标签（检测到 {len(dets)} 个）"
    objlist, imglist = [], []
    for d in known:
        tid = str(d.tag_id)
        objlist.extend(tag_poses[tid])
        imglist.extend(np.asarray(d.corners, dtype=np.float64)[TAG_CORNER_PERM])
    result = solve_pnp_pose(objlist, imglist)
    if result is None:
        return None, "PnP 求解失败"
    pos, ori, reproj = result
    problems = pnp_pose_problems(pos, ori, reproj)
    if problems:
        return None, "门控拒绝: " + "；".join(problems)
    return (pos, ori, reproj), None


def body_orientation(cam_ori_3d, pulse):
    """相机朝向(XY投影) 经头部偏转补偿 → 机体朝向（与 compensate_head_offset 同公式）"""
    c = cam_ori_3d[:2]
    n = np.linalg.norm(c)
    if n < 1e-9:
        return None
    c = c / n
    theta = np.radians((pulse - HEAD_CENTER) * SERVO_DEG_PER_US)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    b = np.array([cos_t * c[0] + sin_t * c[1],
                  -sin_t * c[0] + cos_t * c[1]])
    return b


def angle_diff_deg(a, b):
    """两个 2D 单位向量的夹角（度）"""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def median_record(records):
    """多组成功定位取中值（位置/朝向），朝向中值后归一化"""
    pos = np.median(np.array([r["pos"] for r in records]), axis=0)
    body = np.median(np.array([r["body"] for r in records]), axis=0)
    body = body / np.linalg.norm(body)
    reproj = float(np.median([r["reproj"] for r in records]))
    return {"pos": pos, "body": body, "reproj": reproj}


def fmt_row(fields):
    return "\t".join(str(f) for f in fields)


def main():
    parser = argparse.ArgumentParser(description="头部转动定位补偿有效性验证（真机）")
    parser.add_argument("--repeat", type=int, default=1,
                        help="每档位拍照次数（默认 1；结果临界时建议 3 取中值）")
    args = parser.parse_args()
    repeat = max(1, args.repeat)

    os.makedirs("result", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("result", f"head_comp_check_{ts}.txt")
    tee = TeeWriter(log_path)
    old_stdout = sys.stdout
    sys.stdout = tee
    try:
        print("=" * 70)
        print("头部转动定位补偿有效性验证（7 头位）")
        for name, pulse, deg in HEAD_POSES:
            print(f"  {name}: 脉宽 {pulse}, 标称转角 {deg:+.1f}°")
        print(f"每档位拍照次数: {repeat}")
        print(f"日志: {log_path}")
        print("提醒: 测试全程机器人保持静止（不执行任何动作），头部由脚本自动转动。")
        print("=" * 70)

        state = RobotState(tag_poses=tag_poses)
        state.set_head(HEAD_CENTER)

        # ---- 数据采集：每个头位一行 ----
        print()
        print(fmt_row(["头位", "脉宽", "标称角", "成功", "相机位置x", "相机位置y",
                       "机体朝向°", "reproj", "失败原因"]))
        results = []  # (name, pulse, deg, rec|None, fail_reason)
        for name, pulse, deg in HEAD_POSES:
            ok_records = []
            fail_reasons = []
            for i in range(repeat):
                state.set_head(pulse)
                result, reason = locate_once(state)
                if result is None:
                    fail_reasons.append(reason)
                    continue
                pos, ori, reproj = result
                body = body_orientation(ori, pulse)
                if body is None:
                    fail_reasons.append("朝向接近垂直")
                    continue
                ok_records.append({"pos": pos[:2], "body": body, "reproj": reproj})
            if ok_records:
                rec = median_record(ok_records)
                heading = np.degrees(np.arctan2(rec["body"][1], rec["body"][0]))
                print(fmt_row([name, pulse, f"{deg:+.1f}", f"{len(ok_records)}/{repeat}",
                               f"{rec['pos'][0]:.2f}", f"{rec['pos'][1]:.2f}",
                               f"{heading:.1f}", f"{rec['reproj']:.2f}", ""]))
                results.append((name, pulse, deg, rec, None))
            else:
                reason = fail_reasons[0] if fail_reasons else "未知"
                print(fmt_row([name, pulse, f"{deg:+.1f}", "0/" + str(repeat),
                               "-", "-", "-", "-", reason]))
                results.append((name, pulse, deg, None, reason))

        state.set_head(HEAD_CENTER)

        # ---- 汇总：先逐行数据，再结论 ----
        print()
        print("=" * 70)
        print("汇总（基准 = 回正）")
        base = next((rec for n, p, d, rec, _ in results if n == "回正" and rec is not None), None)
        if base is None:
            print("回正基准缺失（定位全部失败），测试无效。")
            print("请把机器人摆到能看到标签、距离 30~50cm 的位置后重试。")
            return

        base_heading = np.degrees(np.arctan2(base["body"][1], base["body"][0]))
        print(fmt_row(["头位", "标称角", "相机位置(x,y)", "机体朝向°",
                       "reproj", "朝向差°", "朝向差/转角", "位置差cm", "反推偏心cm", "判定"]))
        print(fmt_row(["回正", "0.0", f"({base['pos'][0]:.2f},{base['pos'][1]:.2f})",
                       f"{base_heading:.1f}", f"{base['reproj']:.2f}", "基准", "—", "基准", "—", ""]))

        rows = []
        for name, pulse, deg, rec, _ in results:
            if name == "回正" or rec is None:
                continue
            d_ori = angle_diff_deg(base["body"], rec["body"])
            d_pos = float(np.linalg.norm(base["pos"] - rec["pos"]))
            ratio = d_ori / abs(deg) if abs(deg) > 1e-6 else float("nan")
            theta_rad = np.radians(deg)
            lever = d_pos / (2.0 * abs(np.sin(theta_rad / 2.0))) if abs(deg) > 1e-6 else float("nan")
            reproj = rec["reproj"]
            bad = reproj > GOOD_REPROJ_PX
            rows.append((name, deg, d_ori, ratio, d_pos, lever, reproj, bad))
            print(fmt_row([name, f"{deg:+.1f}",
                           f"({rec['pos'][0]:.2f},{rec['pos'][1]:.2f})",
                           f"{np.degrees(np.arctan2(rec['body'][1], rec['body'][0])):.1f}",
                           f"{reproj:.2f}", f"{d_ori:.1f}", f"{ratio:.2f}", f"{d_pos:.2f}",
                           f"{lever:.1f}", "坏解(≥2px，不参与判定)" if bad else ""]))

        # ---- 结论 ----
        print()
        print("=" * 70)
        print("结论")
        n_bad = sum(1 for r in rows if r[7])
        valid_rows = [r for r in rows
                      if np.isfinite(r[2]) and np.isfinite(r[3]) and not r[7]]
        if n_bad:
            print(f"注: {n_bad} 个档位被判为坏解（reproj>{GOOD_REPROJ_PX:.1f}px），已排除，不参与判定。")
        if not valid_rows:
            print("无有效档位（全部失败或坏解），无法判定。请调整摆放位置/距离后重试。")
            return

        ratios = [r[3] for r in valid_rows]
        d_oris = [r[2] for r in valid_rows]
        d_poses = [r[4] for r in valid_rows]
        levers = [r[5] for r in valid_rows if np.isfinite(r[5])]

        ratio_med = float(np.median(ratios))
        d_ori_max = float(np.max(d_oris))
        d_pos_max = float(np.max(d_poses))
        lever_med = float(np.median(levers)) if levers else float("nan")

        print(f"[1] 补偿方向: 朝向差/转角 中值比例 = {ratio_med:.2f}", end="  ")
        if ratio_med < RATIO_DIR_FLIP * 0.5:
            print("→ 方向约定正确（右负左正成立），补偿公式无需反转")
        else:
            print("→ ✗ 疑似「脉宽→转向」方向约定反了（补偿符号需要反转）")

        print(f"[2] 标称角度: 朝向差/转角 比例 {ratio_med:.2f}", end="  ")
        if ratio_med < RATIO_SCALE_BAD:
            print("→ SERVO_DEG_PER_US=0.09 基本准确（误差 <15%）")
        elif ratio_med < RATIO_DIR_FLIP:
            print(f"→ ⚠ 实际转角与标称偏差约 {ratio_med*100:.0f}%（建议修正角度系数）")
        else:
            print("→ 被方向问题主导，先修正 [1] 再复测")

        print(f"[3] 相机偏心: 位置差最大 {d_pos_max:.2f}cm", end="  ")
        if d_pos_max < POS_OK_CM:
            print("→ 偏心影响可忽略（<2cm），可不做位置补偿")
        else:
            print(f"→ 存在偏心位移；反推光心偏心 中值 ≈ {lever_med:.1f}cm"
                  f"（建议按此值在 locate_with_scan 中做位置补偿）")

        print(f"[4] 补偿有效性: 朝向差最大 {d_ori_max:.1f}°", end="  ")
        if d_ori_max < HEAD_OK_DEG:
            print("→ ✓ 各头位补偿后机体朝向一致，补偿有效")
        else:
            print(f"→ ⚠ 存在 {d_ori_max:.1f}° 的偏差，结合 [1][2] 定位原因")

        print()
        print("说明: 右/左两侧结论一致才可信；若互斥，请 --repeat 3 复测或检查机械限位。")
        print("建议: 若 [3] 偏心 ≥2cm，后续在 locate_with_scan 的转头分支中，"
              "按 lever 中值对相机位置做反向补偿，再统一为机体位置。")
    finally:
        sys.stdout = old_stdout
        tee.close()
        print(f"[head-check] 日志已保存: {log_path}")


if __name__ == "__main__":
    main()
