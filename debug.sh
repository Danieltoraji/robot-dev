#!/usr/bin/env bash
# debug.sh —— 视觉调试镜像服务器启停（机器人在 Robot_Competition 根目录执行）
# 用法: bash debug.sh [start|stop|status|log]   （默认 start）
set -u
cd "$(dirname "$0")"
PY=/home/pi/jupyter-env/bin/python3
PORT=8081

case "${1:-start}" in
  start)
    pkill -f "[d]ebug_server" 2>/dev/null   # 方括号防自匹配；杀掉旧实例
    sleep 1
    # 2026-09-23：去掉 --digit-templates（digit 分析器改走关卡链路，不再用
    # 字体模板：色块 → 字形掩膜 → 颜色主判/形状仲裁），并默认打开 digit
    # 分析器 —— 它的叠加现在反映关卡会认定的数字，是现场排查的关键视图。
    nohup $PY tools/debug_server.py --port $PORT --enable-digit \
      > /tmp/debug_server.log 2>&1 &
    sleep 3
    IP=$(hostname -I | awk '{print $1}')
    if curl -s --max-time 5 "http://127.0.0.1:$PORT/api/state" > /dev/null; then
      echo "调试镜像已启动: http://$IP:$PORT"
    else
      echo "启动异常，查看日志: bash debug.sh log"
    fi
    echo "停止: bash debug.sh stop"
    ;;
  stop)
    pkill -f "[d]ebug_server" && echo "已停止" || echo "本就没在运行"
    ;;
  status)
    if pgrep -f "[d]ebug_server" > /dev/null; then
      echo "运行中:"; pgrep -af "[d]ebug_server"
    else
      echo "未运行（bash debug.sh 启动）"
    fi
    ;;
  log)
    tail -30 /tmp/debug_server.log
    ;;
  *)
    echo "用法: bash debug.sh [start|stop|status|log]"
    ;;
esac
