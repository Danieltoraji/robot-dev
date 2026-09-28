#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""set_robot_host.py —— 设置机器人地址（持久环境变量，sync/exec/pull 三工具共用）

用法：
    python tools/set_robot_host.py http://192.168.31.209:8888   # 持久设置
    python tools/set_robot_host.py                              # 查看当前值

说明：setx 写入用户环境变量，**已打开的终端不会生效**，重开终端后生效；
机器人 IP 是 DHCP 动态的，换网络后跑一次本脚本即可。
"""
import os
import sys

if len(sys.argv) > 1:
    host = sys.argv[1].rstrip("/")
    assert host.startswith("http://") or host.startswith("https://"), \
        "地址需以 http:// 开头"
    r = os.system(f'setx ROBOT_HOST "{host}"')
    os.environ["ROBOT_HOST"] = host  # 当前进程也生效
    print("setx 完成" if r == 0 else "setx 失败", "->", host)
    print("注意：已打开的其他终端需重开后生效")
else:
    print("当前 ROBOT_HOST =", os.environ.get("ROBOT_HOST", "(未设置，用内置默认)"))
