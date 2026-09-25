# -*- coding: utf-8 -*-
"""完整 dump 转弯动作的 6 帧舵机脉冲，对比 turn_left/turn_right 步幅。"""
import sqlite3

D = '/home/pi/TonyPi/ActionGroups/'


def dump(name):
    db = sqlite3.connect(D + name + '.d6a')
    cur = db.cursor()
    cur.execute("SELECT * FROM ActionGroup ORDER BY rowid")
    rows = cur.fetchall()
    print('==== %s  (%d 帧)' % (name, len(rows)))
    base = None
    for r in rows:
        vals = list(r)
        # 第1列是帧序号/时间，其余是舵机脉冲
        frame_id = vals[0]
        pulses = vals[1:]
        if base is None:
            base = pulses
        delta = [p - b for p, b in zip(pulses, base)]
        # 汇总：总绝对偏移、最大绝对偏移、左/右腿（按奇偶粗略分组）偏移
        total_abs = sum(abs(d) for d in delta)
        max_abs = max(abs(d) for d in delta)
        print('frame%s: total_abs=%d max_abs=%d' % (frame_id, total_abs, max_abs))
        print('   pulses=', pulses)
        print('   delta =', delta)
    db.close()


for n in ['turn_left', 'turn_right', 'turn_left_small_step', 'turn_right_small_step']:
    dump(n)
    print()
