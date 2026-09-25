# -*- coding: utf-8 -*-
"""比对 turn_left/turn_right 及小步动作的舵机帧数据，判断是否对称。"""
import sqlite3

D = '/home/pi/TonyPi/ActionGroups/'


def dump(name):
    db = sqlite3.connect(D + name + '.d6a')
    cur = db.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    print('==', name, 'tables:', tables)
    for t in tables:
        cur.execute('SELECT * FROM %s' % t)
        rows = cur.fetchall()
        print('   table', t, 'rows=', len(rows))
        for r in rows[:3]:
            print('     ', r)
    db.close()


for n in ['turn_left', 'turn_right']:
    dump(n)
