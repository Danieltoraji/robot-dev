#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""调试镜像的九宫格「统一判据」叠加（tools/debug_server.py）

背景：现场排查时光看 digit 认出的数字不够 —— 要能一眼看出**关卡会怎么决策**。
所以 debug_server 的 digit 结果默认附带叠加数据：**用于决策的点**（十字）与
**图片分区**（白色细线）。⚠️ 开不开是**页面图层**的事（默认勾上），命令行没有
这个开关 —— 数据默认总是算（开销可忽略），只有 `--no-core` 时不加载关卡判据。

这个文件钉住四件事：
  1. **判据只有一份**：叠加用的 `zone_annotation` 必须就是 `levels.nine_grid.
     zone_at_pixel` 的结果（不许在调试服务器里另抄一套几何/阈值）；
  2. **默认就有**：不带任何开关，digit 结果里就有 point/zone/zone_lines，且每个
     色块的点与判据自洽；
  3. **唯一的例外**：`--no-core` 时不加载关卡判据 ⇒ 没有叠加数据，且 state 里
     `zone_overlay=false` + 人话原因（页面据此置灰勾选框）；
  4. **页面接线**：勾选框存在、默认勾上、进了 onchange→draw 的 id 列表、画的是
     归一化分区线与十字 —— 防"加了勾选框没接线"这类静默半成品。

照片夹具缺失时相关用例自动跳过（与 tests/test_nine_grid_replay.py 同风格）。
"""

import io
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))     # 仓库根

import pytest

import levels.nine_grid as NG
from tools import debug_server as DS

W, H = 2592.0, 1944.0
PHOTO = os.path.join(_HERE, "fixtures", "field_photos",
                     "field_1789281048_1040_1500.jpg")


class _Args:
    """Analyzers/Hub 需要的最小参数集（只跑 digit，不碰相机/网络）"""
    no_tag = True
    no_yolo = True
    no_line = True
    enable_digit = True
    model = DS.DEFAULT_MODEL
    conf = 0.45
    photo_dir = "."

    def __init__(self, no_core=False):
        self.no_core = no_core


@pytest.fixture(scope="module")
def digit_payload():
    """真机照片上跑一次 digit 分析器；返回 (默认参数, --no-core) 两份 payload"""
    cv2 = pytest.importorskip("cv2")
    if not os.path.exists(PHOTO):
        pytest.skip(f"现场照片夹具缺失: {PHOTO}")
    frame = cv2.imread(PHOTO)
    assert frame is not None and frame.shape[:2] == (1944, 2592), frame.shape
    out = {}
    for no_core in (False, True):
        an = DS.Analyzers(_Args(no_core))
        res, _ms = an.run("digit", frame)
        assert an.status["digit"] == "ok", an.status
        out[no_core] = res["digit"]
    return out[False], out[True]


def test_zone_annotation_is_the_level_criterion():
    """叠加的判据 == 关卡判据（同一函数、同一常量；含边界附近与画幅四角）"""
    pts = [(W / 2.0, 0.99 * H), (W / 2.0 + 0.09 * W, 0.99 * H),
           (W / 2.0 + 0.20 * W, 0.95 * H), (W / 2.0 + 0.20 * W, 0.70 * H),
           (W / 2.0 + 0.30 * W, 0.99 * H), (0.0, 0.0), (W - 1.0, H - 1.0),
           (W / 2.0 + 0.1200 * W, 0.999 * H),      # 走廊边界附近
           (W / 2.0 + 0.2747 * W, 0.7556 * H)]     # 蓝外沿/蓝上沿交点附近
    for (px, py) in pts:
        ann = DS.zone_annotation(px, py, W, H)
        assert ann is not None, "判据不可用（import levels.nine_grid 失败）"
        assert ann["zone"] == NG.zone_at_pixel(px, py, W, H), (px, py)
        assert ann["zone_cn"] == NG.ZONE_NAME_CN[ann["zone"]]
        assert 0.0 <= ann["t"] <= 1.0 and 0.0 <= ann["dx_frac"] <= 1.0


def test_zone_lines_are_the_level_lines():
    """叠加用的分割线就是关卡的 zone_lines（归一化、7 段）"""
    assert DS.zone_lines() == NG.zone_lines()
    assert len(DS.zone_lines()) == 7


def test_digit_payload_always_carries_overlay(digit_payload):
    """默认（不带任何开关）就有叠加：7 段归一化分区线 + 每个色块自洽的判据点"""
    base, _ = digit_payload
    assert "zone_note" not in base, f"默认不该有异常说明: {base.get('zone_note')}"
    lines = base["zone_lines"]
    assert len(lines) == 7
    for seg in lines:
        assert len(seg) == 2
        for q in seg:
            assert 0.0 <= q[0] <= 1.0 and 0.0 <= q[1] <= 1.0
    assert base["panels"]
    for p in base["panels"]:
        # 判据点必须落在画幅内（裁切面板取凸包质心，仍在画面里）
        assert 0.0 <= p["point"][0] <= W and 0.0 <= p["point"][1] <= H, p["point"]
        assert p["zone"] == NG.zone_at_pixel(p["point"][0], p["point"][1], W, H)
        assert p["zone_cn"] == NG.ZONE_NAME_CN[p["zone"]]
        assert p["t"] == pytest.approx(p["point"][1] / H, abs=1e-3)
        assert p["dx_frac"] == pytest.approx(
            abs(p["point"][0] - W / 2.0) / W, abs=1e-3)


def test_no_core_turns_overlay_off(digit_payload):
    """唯一例外：--no-core（本进程不碰关卡/core 层）⇒ 不加载判据、没有叠加数据"""
    _, off = digit_payload
    assert "zone_lines" not in off
    assert "分区叠加" in (off.get("zone_note") or ""), off.get("zone_note")
    for p in off["panels"]:
        assert "point" not in p and "zone" not in p \
            and "zone_cn" not in p and "dx_frac" not in p and "t" not in p


def test_state_payload_reports_overlay_availability():
    """页面靠 zone_overlay 决定勾选框能不能用（能用时默认勾上，见下一个用例）"""
    hub = DS.Hub(_Args(no_core=False))
    s = hub.state_payload()
    assert s["zone_overlay"] is True and not s["zone_note"]
    hub = DS.Hub(_Args(no_core=True))
    s = hub.state_payload()
    assert s["zone_overlay"] is False
    assert "no-core" in (s["zone_note"] or ""), s["zone_note"]


def test_page_wires_the_zone_layer():
    """页面接线：勾选框**默认勾上** + 进了图层 id 列表 + 画的是归一化线与十字"""
    html = DS.PAGE_HTML
    assert 'id="ly_zone" checked' in html, "分区判据勾选框缺失或没默认勾上"
    assert "'ly_zone'" in html, "勾选框没进 onchange→draw 的 id 列表（改了不重绘）"
    assert "zone_lines" in html, "没画分区线"
    assert "state.zone_overlay" in html, "没按服务器能力置灰勾选框"
    assert "#ffffff" in html, "分区线不是白的（用户要求白色细线）"
    # 命令行不该再有这个开关
    assert "--digit-zone" not in html


def test_every_response_is_uncacheable():
    """HTML 也必须发 Cache-Control: no-cache

    踩坑（2026-09-25）：HTML 以前不发这个头（只有 image/json 发），浏览器按启发式
    规则缓存整页 —— 代码同步 + 服务重启之后，页面还是旧的（图层少了新增的
    「分区判据」勾选框），看起来像"改了没生效"。
    """
    h = DS.Handler.__new__(DS.Handler)          # 不建 socket，只验发头逻辑
    sent = []
    h.send_response = lambda code: sent.append(("status", code))
    h.send_header = lambda k, v: sent.append((k, v))
    h.end_headers = lambda: None
    h.wfile = io.BytesIO()
    for ctype in ("text/html; charset=utf-8", "application/json",
                  "image/jpeg", "text/plain"):
        sent.clear()
        h._send(200, b"x", ctype)
        assert ("Cache-Control", "no-cache") in sent, ctype
