#!/usr/bin/env bash
# debug.sh —— 视觉调试镜像服务器启停（机器人在 Robot_Competition 根目录执行）
# 用法: bash debug.sh [start|stop|status|log|layout]   （默认 start）
#   九宫格「统一判据」叠加（判据点十字 + 分区白线）**默认就在**：它在网页的
#   「图层显示」里（「分区判据（十字＋白线）」，默认勾上），不想要就取消勾选
#   —— 开关在图形界面，不在命令行。`zone` 仍接受，但已等价于 start。
#   `layout` 只打印"布局流程总开关"的现状与用法（开关本身在关卡代码里，不在本脚本）。
# 额外参数原样透传给 tools/debug_server.py，例如: bash debug.sh --no-yolo
set -u
cd "$(dirname "$0")"
PY=/home/pi/jupyter-env/bin/python3
PORT=8081

start_server() {          # $1 = 追加给 debug_server.py 的参数（可为空）
  pkill -f "[d]ebug_server" 2>/dev/null   # 方括号防自匹配；杀掉旧实例
  sleep 1
  # 2026-09-23：去掉 --digit-templates（digit 分析器改走关卡链路，不再用
  # 字体模板：色块 → 字形掩膜 → 颜色主判/形状仲裁），并默认打开 digit
  # 分析器 —— 它的叠加现在反映关卡会认定的数字，是现场排查的关键视图。
  # 2026-09-25：digit 结果默认附带「分区判据」数据（判据点 + 分区线，debug_server
  # 直接复用 levels.nine_grid 的 zone_at_pixel/zone_lines，不另抄一份几何）；
  # 显示与否在网页图层里勾，命令行不再有开关。
  nohup $PY tools/debug_server.py --port $PORT --enable-digit $1 \
    > /tmp/debug_server.log 2>&1 &
  sleep 3
  IP=$(hostname -I | awk '{print $1}')
  if curl -s --max-time 5 "http://127.0.0.1:$PORT/api/state" > /dev/null; then
    echo "调试镜像已启动: http://$IP:$PORT"
    echo "分区判据叠加: 页面图层「分区判据（十字＋白线）」默认勾上（可随时取消勾选）"
  else
    echo "启动异常，查看日志: bash debug.sh log"
  fi
  echo "停止: bash debug.sh stop"
}

EXTRA=""
if [ "$#" -gt 1 ]; then EXTRA="${*:2}"; fi   # 额外参数透传（bash 切片；set -u 安全）

case "${1:-start}" in
  start)
    start_server "$EXTRA"
    ;;
  zone)
    # 分区叠加改成页面图层之后，这条子命令不再需要；保留成等价别名，
    # 只是不让"习惯了 bash debug.sh zone"的人撞一句用法错误。
    echo "提示: 分区叠加已是页面图层（默认勾上），zone 与 start 等价"
    start_server "$EXTRA"
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
  layout)
    # 布局流程总开关（2026-09-26）：真正的开关在代码里
    # （levels/nine_grid_shared.LAYOUT_ENABLED）+ 运行时环境变量
    # NINEGRID_NO_LAYOUT。本子命令只是"查一下现在是什么状态、怎么开"，
    # 免得现场翻代码找名字。
    echo "布局流程总开关（只影响 main.py 跑关卡，不影响本调试镜像）："
    echo "  开（默认）: 跑布局扫/格阵拟合/相机自标定/位姿自举"
    echo "  关        : 全部短路——逐格导航全靠视觉，布局投票不足不再终止整局"
    echo "  代码开关  : levels/nine_grid_shared.py 的 LAYOUT_ENABLED = True/False"
    echo "  临场关闭  : NINEGRID_NO_LAYOUT=1 python main.py nine_grid_three_stage"
    echo "  当前 shell: ${NINEGRID_NO_LAYOUT:+已设置 NINEGRID_NO_LAYOUT=$NINEGRID_NO_LAYOUT}${NINEGRID_NO_LAYOUT:-未设置（按代码常量）}"
    ;;
  *)
    echo "用法: bash debug.sh [start|zone|stop|status|log|layout]"
    echo "  （zone 与 start 等价；layout 只打印布局总开关的用法）"
    ;;
esac
