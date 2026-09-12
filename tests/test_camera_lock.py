# -*- coding: utf-8 -*-
"""相机参数锁定测试（纯逻辑，不需要真相机）

现场问题：fswebcam 每次调用都重新测光/白平衡，同一场地白底板 BGR 实测从
[153,155,149] 漂到 [209,186,180]（R/B 差 16%）→ 粉贴纸 H 168→141 漏检。
对策是一局开始用 v4l2-ctl 钉死白平衡/对焦（可选曝光）。本测试只验证
命令构造、读回解析与降级路径——真机生效性由现场 `--list-ctrls` 复验。

运行：python tests/test_camera_lock.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import core.robot_core as rc
from core.robot_core import (
    camera_lock_controls, build_camera_lock_cmd, build_camera_readback_cmd,
    parse_v4l2_values, lock_camera_controls,
)


def test_command_construction():
    device, ctrls = camera_lock_controls()
    assert device == rc.CAM_V4L2_DEVICE
    names = [k for k, _v in ctrls]
    for k in ("white_balance_automatic", "white_balance_temperature",
              "focus_automatic_continuous", "focus_absolute"):
        assert k in names, f"锁定清单缺 {k}: {names}"
    assert dict(ctrls)["white_balance_automatic"] == 0, "白平衡必须切手动"
    cmd = build_camera_lock_cmd(device, ctrls)
    assert cmd.startswith("v4l2-ctl -d ") and "-c " in cmd
    assert "white_balance_temperature=" in cmd and "," in cmd
    rb = build_camera_readback_cmd(device, ctrls)
    assert "--get-ctrl=" in rb
    # 曝光默认不锁，显式要求时才加入
    _d, c2 = camera_lock_controls(lock_exposure=True)
    assert ("auto_exposure", 1) in c2 and ("exposure_time_absolute", 313) in c2
    _d, c3 = camera_lock_controls(lock_exposure=False)
    assert not any(k.startswith("exposure") or k == "auto_exposure"
                   for k, _v in c3)
    print(f"  命令构造：{len(ctrls)} 项控制、曝光默认不锁 ✓")


def test_readback_parse():
    text = ("white_balance_automatic: 0\n"
            "white_balance_temperature: 4000\n"
            "focus_absolute: 166\n"
            "garbage line\n"
            "unknown_ctrl: n/a\n")
    got = parse_v4l2_values(text)
    assert got == {"white_balance_automatic": 0,
                   "white_balance_temperature": 4000,
                   "focus_absolute": 166}, got
    assert parse_v4l2_values("") == {} and parse_v4l2_values(None) == {}
    print("  读回解析：容忍空行/非数值/空输入 ✓")


def test_graceful_degradation():
    """PC 开发环境（无 /dev/video0）必须优雅降级，不抛异常"""
    ok, info = lock_camera_controls(device="/dev/definitely-not-here",
                                    force=True)
    assert ok is False
    assert info["applied"] is False and info["reason"]
    assert "readback" in info and "mismatch" in info
    # 关闭开关时也是"未锁但不报错"
    old = rc.CAM_LOCK_CONTROLS_ENABLED
    try:
        rc.CAM_LOCK_CONTROLS_ENABLED = False
        ok2, info2 = lock_camera_controls(force=False)
        assert ok2 is False and "关闭" in info2["reason"]
    finally:
        rc.CAM_LOCK_CONTROLS_ENABLED = old
    print(f"  降级路径：设备缺失/开关关闭均安全返回（{info['reason'][:24]}…） ✓")


if __name__ == "__main__":
    test_command_construction()
    test_readback_parse()
    test_graceful_degradation()
    print("相机参数锁定测试通过 ✓")
