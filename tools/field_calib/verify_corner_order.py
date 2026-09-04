#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_corner_order.py —— 真机验证 AprilTag 角点顺序（P0 验证脚本）

背景：
    solve_pnp 把 apriltag 库返回的 r.corners 顺序直接当成与 tag_poses
    「左上→右上→右下→左下」一致。该假设从未验证；顺序错 = 位姿错，
    且正方形标签的错误解重投影误差≈0，无法自查。
    本脚本用「已知位姿拍照对比」：对每个标签 × 8 种角点排列求解 PnP，
    与手工测量的真实相机位姿对比，跑多组后投票得出正确排列。

用法（在机器人项目根目录）：
    python -m tools.field_calib.verify_corner_order               # 默认 3 组
    python -m tools.field_calib.verify_corner_order --n 5         # 跑 5 组
    python -m tools.field_calib.verify_corner_order --photo /home/pi/Pictures/xxx.jpg  # 分析已存照片

每组步骤：
    1. 摆好机器人到任意可测位姿（尽量同时看到 ≥2 个标签）；
    2. 输入「相机镜头中心」真实坐标 x y z（cm）与相机水平朝向 dx dy；
    3. 脚本拍照 → 检测标签 → 8 种排列逐一求解并对比真值；
    4. 换位置/朝向再测下一组（位置差异拉大，投票才可靠）。

输出：
    每组明细 + 汇总投票表 + 推荐 TAG_CORNER_PERM；
    日志同步保存 archive/result/verify_corner_order_<时间戳>.txt。

注意：
    - 真值量「相机镜头中心」而不是机器人中心，最准；
      即使有 5~10cm 测量误差也不影响结论（错误排列误差通常 20cm+）。
    - 拍照前务必头部回正（HEAD_CENTER=1500），否则相机朝向 ≠ 机体朝向。
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np

# 允许直接运行本文件
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.paths import RESULT_DIR

try:
    import cv2
    from core.robot_core import RobotState, CAMERA_INTRINSIC, CAMERA_DISTORTION
    from levels.goodluck import tag_poses
except Exception as e:
    print(f"[verify] 导入失败: {e}")
    print("[verify] 请在机器人项目根目录运行（需 cv2 / numpy / apriltag / robot_core / levels）")
    sys.exit(1)


# =====================================================================
# 8 种候选角点排列（前 4 = 4 旋转，后 4 = 4 镜像旋转，
# 与 debug_pnp_permutations.py 的编号一致，便于对照）
# =====================================================================
PERMS = [
    np.array([0, 1, 2, 3], dtype=np.int64),
    np.array([1, 2, 3, 0], dtype=np.int64),
    np.array([2, 3, 0, 1], dtype=np.int64),
    np.array([3, 0, 1, 2], dtype=np.int64),
    np.array([3, 2, 1, 0], dtype=np.int64),
    np.array([2, 1, 0, 3], dtype=np.int64),
    np.array([1, 0, 3, 2], dtype=np.int64),
    np.array([0, 3, 2, 1], dtype=np.int64),
]

# 位姿合理性门控参数（与计划写入 robot_core.py 的默认值一致）
FIELD_MIN, FIELD_MAX = -10.0, 110.0   # 场地范围（cm，含余量）
CAM_Z_MIN, CAM_Z_MAX = 5.0, 80.0      # 相机高度合理范围（cm）
ORI_Z_MAX = 0.4                       # 相机朝向近水平（z 分量上限；实测斜仰视角可达 ±0.32）
ORI_XY_MIN = 0.1                      # 相机朝向 XY 分量下限（防垂直朝下）
REPROJ_MAX_PX = 20.0                  # 平均重投影误差上限（px）

# 判定「该排列与真值一致」的容差
POS_OK_CM = 8.0
HEAD_OK_DEG = 12.0


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


# =====================================================================
# 求解与校验（与 robot_core.solve_pnp 同一套数学）
# =====================================================================

def solve_pose(world_pts, img_pts):
    """solvePnP + 提取相机位姿 + 平均重投影误差；失败返回 None

    返回 (pos_3d, ori_3d, reproj_err)；pos 为世界系相机位置，
    ori 为世界系光轴单位向量。
    """
    world = np.asarray(world_pts, dtype=np.float64)
    img = np.asarray(img_pts, dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(world, img, CAMERA_INTRINSIC, CAMERA_DISTORTION)
    if not ok:
        return None
    proj, _ = cv2.projectPoints(world, rvec, tvec, CAMERA_INTRINSIC, CAMERA_DISTORTION)
    reproj_err = float(np.mean(np.linalg.norm(proj[:, 0, :] - img, axis=1)))
    R = cv2.Rodrigues(rvec)[0]
    pos = (-np.linalg.inv(R) @ tvec).flatten()
    ori = (np.linalg.inv(R) @ (np.array([[0.0], [0.0], [1.0]]) - tvec)).flatten() - pos
    n = np.linalg.norm(ori)
    if n != 0:
        ori = ori / n
    return pos, ori, reproj_err


def sanity_problems(pos, ori, reproj_err):
    """位姿合理性检查，返回问题列表（空 = 通过）"""
    problems = []
    x, y, z = pos
    if not (FIELD_MIN <= x <= FIELD_MAX and FIELD_MIN <= y <= FIELD_MAX):
        problems.append(f"位置({x:.1f},{y:.1f})超出场地范围")
    if not (CAM_Z_MIN <= z <= CAM_Z_MAX):
        problems.append(f"相机高度z={z:.1f}cm不合理")
    if abs(ori[2]) > ORI_Z_MAX:
        problems.append(f"朝向不水平(z分量{ori[2]:.2f})")
    if np.linalg.norm(ori[:2]) < ORI_XY_MIN:
        problems.append("朝向接近垂直")
    if reproj_err > REPROJ_MAX_PX:
        problems.append(f"重投影误差{reproj_err:.1f}px过大")
    return problems


def heading_err_deg(ori, gt_heading):
    """相机朝向（XY 投影）与真值水平朝向的夹角（度）"""
    h = ori[:2]
    hn = np.linalg.norm(h)
    if hn < 1e-6:
        return 180.0
    h = h / hn
    gt = np.asarray(gt_heading, dtype=np.float64)
    gt = gt / np.linalg.norm(gt)
    return float(np.degrees(np.arccos(np.clip(np.dot(h, gt), -1.0, 1.0))))


def analyze_tag(tag_id, world, img, gt_pos, gt_heading):
    """对单个标签 × 8 种排列求解，返回结果列表（与 PERMS 对齐）"""
    results = []
    for perm in PERMS:
        sol = solve_pose(world, img[perm])
        if sol is None:
            results.append({
                "perm": perm.copy(),
                "pos_err": None,
                "head_err": None,
                "reproj": None,
                "problems": ["PnP求解失败"],
            })
            continue
        pos, ori, reproj = sol
        pos_err = float(np.linalg.norm(pos - np.asarray(gt_pos, dtype=np.float64)))
        head_err = heading_err_deg(ori, gt_heading)
        problems = sanity_problems(pos, ori, reproj)
        results.append({
            "perm": perm.copy(),
            "pos_err": pos_err,
            "head_err": head_err,
            "reproj": reproj,
            "problems": problems,
        })
    return results


# =====================================================================
# 输入 / 单组流程
# =====================================================================

def ask_truth(k):
    """输入本组真值：相机镜头中心世界坐标 + 水平朝向"""
    print(f"\n[第 {k} 组] 请确认机器人已摆好，输入真值：")
    while True:
        raw = input("  相机镜头中心世界坐标 x y z (cm，空格分隔)：").strip()
        try:
            vals = [float(v) for v in raw.replace(",", " ").split()]
            if len(vals) != 3:
                raise ValueError
            gt_pos = np.array(vals, dtype=np.float64)
            break
        except Exception:
            print("  格式错误，示例：24.7 21.3 39.0")
    while True:
        raw = input("  相机水平朝向 dx dy（如 1 0）：").strip()
        try:
            vals = [float(v) for v in raw.replace(",", " ").split()]
            if len(vals) != 2 or np.linalg.norm(vals) < 1e-6:
                raise ValueError
            gt_head = np.array(vals, dtype=np.float64)
            gt_head = gt_head / np.linalg.norm(gt_head)
            break
        except Exception:
            print("  格式错误，示例：1 0（东）；0 1（北）；-1 0（西）")
    return gt_pos, gt_head


def run_one_sample(state, k, photo_path=None):
    """执行一组：输入真值 → 拍照/读图 → 检测 → 逐标签逐排列对比

    返回观测记录列表（每标签一条）。
    """
    gt_pos, gt_head = ask_truth(k)

    if photo_path:
        filename = photo_path
        print(f"[第 {k} 组] 分析已存照片: {filename}")
    else:
        print(f"[第 {k} 组] 正在拍照...")
        filename = state.capture_image()
        if filename is None:
            print("  拍照失败，本组无效。")
            return []

    image = cv2.imread(filename)
    if image is None:
        print(f"  读图失败: {filename}")
        return []
    h, w = image.shape[:2]
    if (w, h) != (2592, 1944):
        print(f"  [WARN] 实际分辨率 {w}x{h} 与内参对应 2592x1944 不一致，PnP 结果不可信！")

    dets = state.detect_apriltag(filename)
    if not dets:
        print("  未检测到 AprilTag，本组无效。")
        return []
    known = [d for d in dets if str(d.tag_id) in tag_poses]
    print(f"  检测到 {len(dets)} 个标签: " + ", ".join(str(d.tag_id) for d in dets))
    if len(known) < 2:
        print("  [提示] 本组仅 1 个可用标签：单标签+真值仍可判定（镜像解位置会明显偏离真值），"
              "但多标签投票更稳。")

    observations = []
    for d in known:
        tid = str(d.tag_id)
        world = np.asarray(tag_poses[tid], dtype=np.float64)
        img = np.asarray(d.corners, dtype=np.float64)
        per_perm = analyze_tag(tid, world, img, gt_pos, gt_head)
        observations.append({
            "sample": k,
            "tag": tid,
            "gt_pos": gt_pos,
            "gt_head": gt_head,
            "tag_center": world.mean(axis=0),
            "per_perm": per_perm,
        })

        tag_dist = float(np.linalg.norm(gt_pos - world.mean(axis=0)))
        print(f"\n--- Tag {tid}（相机到标签中心约 {tag_dist:.1f}cm） ---")
        for r in per_perm:
            if r["pos_err"] is None:
                print(f"  perm {r['perm'].tolist()}  求解失败")
                continue
            ok = (r["pos_err"] <= POS_OK_CM and r["head_err"] <= HEAD_OK_DEG and not r["problems"])
            flag = "✓" if ok else "✗"
            prob = ("；" + "；".join(r["problems"])) if r["problems"] else ""
            print(f"  perm {r['perm'].tolist()}  pos_err={r['pos_err']:6.1f}cm  "
                  f"head_err={r['head_err']:5.1f}°  reproj={r['reproj']:.2f}px  {flag}{prob}")
        valid = [r for r in per_perm if r["pos_err"] is not None]
        if valid:
            best = min(valid, key=lambda r: r["pos_err"])
            print(f"  >>> Tag {tid} 按位置误差最佳: perm {best['perm'].tolist()}  "
                  f"(pos_err={best['pos_err']:.1f}cm head_err={best['head_err']:.1f}°)")
    return observations


# =====================================================================
# 汇总投票
# =====================================================================

def summarize(observations):
    n_perm = len(PERMS)
    n_obs = len(observations)
    print("\n" + "=" * 72)
    print(f"汇总投票：共 {n_obs} 次标签观测")
    if n_obs == 0:
        print("无有效观测，无法给出结论。")
        return

    ranks = {pi: [] for pi in range(n_perm)}
    pos_errs = {pi: [] for pi in range(n_perm)}
    head_errs = {pi: [] for pi in range(n_perm)}
    reprojs = {pi: [] for pi in range(n_perm)}
    rank1 = {pi: 0 for pi in range(n_perm)}
    sanity_ok = {pi: 0 for pi in range(n_perm)}

    for obs in observations:
        errs = [r["pos_err"] for r in obs["per_perm"]]
        # 每次观测内按位置误差排名（求解失败排最后）
        order = sorted(
            range(n_perm),
            key=lambda i: (errs[i] is None, errs[i] if errs[i] is not None else float("inf")),
        )
        for rank, pi in enumerate(order, 1):
            ranks[pi].append(rank)
        rank1[order[0]] += 1
        for pi in range(n_perm):
            r = obs["per_perm"][pi]
            if r["pos_err"] is None:
                continue
            pos_errs[pi].append(r["pos_err"])
            head_errs[pi].append(r["head_err"])
            reprojs[pi].append(r["reproj"])
            if not r["problems"]:
                sanity_ok[pi] += 1

    print(f"\n{'排列':<12}{'平均位置误差':>13}{'平均朝向误差':>13}{'平均排名':>9}{'排名第1':>9}{'门控通过':>9}")
    for pi, perm in enumerate(PERMS):
        mp = float(np.mean(pos_errs[pi])) if pos_errs[pi] else float("nan")
        mh = float(np.mean(head_errs[pi])) if head_errs[pi] else float("nan")
        mr = float(np.mean(ranks[pi])) if ranks[pi] else float("nan")
        print(f"{str(perm.tolist()):<12}{mp:>9.1f}cm{mh:>9.1f}°{mr:>9.2f}"
              f"{f'{rank1[pi]}/{n_obs}':>9}{f'{sanity_ok[pi]}/{n_obs}':>9}")

    # 冠军：平均排名最小；并列时取平均位置误差更小者
    def key(pi):
        return (
            float(np.mean(ranks[pi])) if ranks[pi] else 1e9,
            float(np.mean(pos_errs[pi])) if pos_errs[pi] else 1e9,
        )

    winner = min(range(n_perm), key=key)
    w_perm = PERMS[winner]
    w_mp = float(np.mean(pos_errs[winner])) if pos_errs[winner] else float("nan")
    w_mh = float(np.mean(head_errs[winner])) if head_errs[winner] else float("nan")
    w_mr = float(np.mean(ranks[winner])) if ranks[winner] else float("nan")
    w_r1 = rank1[winner]
    w_ok = sanity_ok[winner]

    print()
    print(f">>> 结论：TAG_CORNER_PERM = {w_perm.tolist()}")
    print(f"    平均排名 {w_mr:.2f}，{w_r1}/{n_obs} 次观测排名第 1，"
          f"平均位置误差 {w_mp:.1f}cm，平均朝向误差 {w_mh:.1f}°，门控通过 {w_ok}/{n_obs}")

    if w_ok < n_obs or w_mp > POS_OK_CM or w_mh > HEAD_OK_DEG:
        print("    [WARN] 冠军排列未全部达标：")
        if w_ok < n_obs:
            print("      - 存在门控问题：请核对真值输入与标签数据")
        if w_mp > POS_OK_CM:
            w_reproj = float(np.mean(reprojs[winner])) if reprojs[winner] else float("nan")
            if w_reproj < 2.0:
                print("      - 重投影误差很小(<2px)但位置误差大 → 多半是真值量错了，请重测该组")
            else:
                print("      - 重投影误差也偏大 → 检查内参/畸变/标签坐标是否与实物一致")

    # 亚军差距检查：两个排列误差接近时结论不可靠
    others = [pi for pi in range(n_perm) if pi != winner and pos_errs[pi]]
    if others:
        runner = min(others, key=key)
        runner_mp = float(np.mean(pos_errs[runner])) if pos_errs[runner] else 1e9
        gap = abs(w_mp - runner_mp)
        if gap < 5.0:
            print(f"    [WARN] 与亚军排列 {PERMS[runner].tolist()} 平均位置误差仅差 {gap:.1f}cm，"
                  f"难以唯一确定。建议：换更斜的视角、拉开位置差异、或让照片同时出现 ≥2 个标签后重测。")

    print()
    print("当前代码行为：直接使用 r.corners 原始顺序（等价 perm [0 1 2 3]，未重排）。")
    if winner == 0:
        print("验证结果与当前实现一致 → 顺序无需重排；后续只需补位姿合理性门控。")
    else:
        print(f"验证结果与当前实现不一致 → 需要显式顺序映射：")
        print(f"    1) 在 robot_core.py 常量区加入：")
        print(f"       TAG_CORNER_PERM = np.array({w_perm.tolist()}, dtype=np.int64)")
        print(f"    2) 在 solve_pnp 中把：")
        print(f"       imglist.extend(r.corners)")
        print(f"       改为：")
        print(f"       imglist.extend(r.corners[TAG_CORNER_PERM])")


# =====================================================================
# 主流程
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="AprilTag 角点顺序真机验证（P0）")
    parser.add_argument("--n", type=int, default=3, help="测试组数（默认 3）")
    parser.add_argument("--photo", type=str, default=None,
                        help="分析已存照片（单张，忽略 --n；需输入拍照时的真值）")
    args = parser.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(RESULT_DIR, f"verify_corner_order_{ts}.txt")
    tee = TeeWriter(log_path)
    old_stdout = sys.stdout
    sys.stdout = tee
    try:
        print("=" * 72)
        print("AprilTag 角点顺序验证（P0）")
        print(f"相机内参: fx={CAMERA_INTRINSIC[0, 0]:.1f}, fy={CAMERA_INTRINSIC[1, 1]:.1f}（对应 2592x1944）")
        print(f"候选排列: {[p.tolist() for p in PERMS]}")
        print("提醒：拍照前请把头部舵机回正（HEAD_CENTER=1500）。")
        print(f"日志: {log_path}")
        print("=" * 72)

        state = RobotState(tag_poses=tag_poses)
        observations = []

        if args.photo:
            obs = run_one_sample(state, 1, photo_path=args.photo)
            observations.extend(obs)
        else:
            n = max(1, args.n)
            k = 1
            while k <= n:
                try:
                    obs = run_one_sample(state, k)
                except (EOFError, KeyboardInterrupt):
                    print("\n用户中断，结束采集。")
                    break
                if k == n:
                    observations.extend(obs)
                    break
                while True:
                    cmd = input(f"\n[第 {k} 组完成] 回车=下一组，r=重拍本组，q=结束采集：").strip().lower()
                    if cmd == "q":
                        k = n + 1
                        break
                    if cmd == "r":
                        break  # 重拍本组（不保存刚才的结果）
                    if cmd == "":
                        observations.extend(obs)
                        k += 1
                        break
                    print("  输入无效")
                if k == n + 1:
                    break  # 用户 q 退出

        summarize(observations)
    finally:
        sys.stdout = old_stdout
        tee.close()
        print(f"[verify] 日志已保存: {log_path}")


if __name__ == "__main__":
    main()
