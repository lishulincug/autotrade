"""
一键启动脚本：
  1) 拉取真实K线（东方财富）+ 失败回退模拟
  2) 执行全部5个策略回测
  3) 生成 dashboard_data.json
  4) 启动本机 HTTP 服务端口 8765，浏览器打开看板
使用：
  cd d:\autotrade
  python run_and_serve.py
  # 然后浏览器访问 http://localhost:8765/
"""
import os
import sys
import json
import time
import threading
import webbrowser
import http.server
import socketserver

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from web_dashboard.dashboard_generator import (
    run_strategies_and_build_dashboard,
    save_payload,
)

PORT = 8765
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(ROOT_DIR, 'web_dashboard')


def ensure_web_dir():
    """把 web_dashboard/index.html 复制一份到根目录方便直接访问；同时 dashboard_data.json 放到 web_dashboard/"""
    pass


def start_http_server(port=PORT):
    """在 web_dashboard 目录启动 HTTP 服务器，同时把 / 路由到 index.html 和 dashboard_data.json"""

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=WEB_DIR, **kwargs)

        def end_headers(self):
            # 禁用缓存，方便每次刷新看最新数据
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Expires', '0')
            super().end_headers()

        def log_message(self, fmt, *args):
            # 静默访问日志
            pass

    # 重用地址
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(('127.0.0.1', port), Handler)
    return httpd


def main(port=PORT, open_browser=True, initial_cash=1_000_000):
    # 打印 ASCII Banner
    print("""
╔══════════════════════════════════════════════════════════════════╗
║      ██████  ██    ██  █████  ███    ██ ████████    ████████╗    ║
║      ██   ██ ██    ██ ██   ██ ████   ██    ██          ██       ║
║      ██████  ██    ██ ███████ ██ ██  ██    ██          ██       ║
║      ██   ██  ██  ██  ██   ██ ██  ██ ██    ██          ██       ║
║      ██   ██   ████   ██   ██ ██   ████    ██          ██       ║
║                ETF量化策略 · 回测终端 v2.0                        ║
║         真实K线 (东方财富API) · 5大策略 · Web可视化看板              ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    print(f"工作目录: {ROOT_DIR}")
    print(f"初始资金: ¥{initial_cash:,.0f}")
    print()

    t0 = time.time()

    # ============ 1. 生成回测数据 ============
    print("=" * 70)
    print("  STAGE 1/3 : 拉取真实数据 + 运行5个策略回测")
    print("=" * 70)

    payload, results = run_strategies_and_build_dashboard(initial_cash=initial_cash)

    # ============ 2. 保存JSON ============
    print()
    print("=" * 70)
    print("  STAGE 2/3 : 生成 dashboard_data.json (供ECharts消费)")
    print("=" * 70)

    data_path = save_payload(payload, WEB_DIR)
    file_size = os.path.getsize(data_path) / 1024
    print(f"  ✅ 已保存: {data_path}")
    print(f"     大小: {file_size:.1f} KB")
    print(f"     策略数: {len(payload['strategies'])}")
    for s in payload['strategies']:
        trades = len(s.get('trades', []))
        kpts = len(s['kline']['dates']) if s.get('kline') else 0
        print(f"       · {s['short_name']:<10} 交易{trades:>4d}笔  K线{kpts:>4d}根")

    # 再复制一份到根目录
    with open(os.path.join(ROOT_DIR, 'dashboard_data.json'), 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False)

    # 同步保存对比CSV
    try:
        from common.metrics import compare_strategies
        strat_results = []
        for s in payload['strategies']:
            strat_results.append((s['short_name'], s['metrics'], s['trade_stats']))
        df = compare_strategies(strat_results)
        csv_path = os.path.join(WEB_DIR, 'backtest_comparison.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"  ✅ 对比表: {csv_path}")
    except Exception as e:
        print(f"  ⚠ 对比CSV生成失败: {e}")

    # ============ 3. 启动HTTP服务器 ============
    print()
    print("=" * 70)
    print(f"  STAGE 3/3 : 启动 Web 看板 @ http://127.0.0.1:{port}")
    print("=" * 70)

    httpd = start_http_server(port)
    url = f'http://127.0.0.1:{port}/index.html'

    def open_later():
        time.sleep(1.5)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    if open_browser:
        threading.Thread(target=open_later, daemon=True).start()

    elapsed = time.time() - t0
    print()
    print(f"  🚀 看板地址: {url}")
    print(f"  📁 静态目录: {WEB_DIR}")
    print(f"  ⏱  全部完成耗时: {elapsed:.1f} 秒")
    print(f"  💡 按 Ctrl+C 停止服务器")
    print("-" * 70)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  ✋ 收到 Ctrl+C，服务器已停止。")
        httpd.server_close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='ETF量化策略回测·Web看板启动器')
    parser.add_argument('--port', type=int, default=PORT, help='HTTP端口（默认 8765）')
    parser.add_argument('--no-browser', action='store_true', help='不自动打开浏览器')
    parser.add_argument('--cash', type=int, default=1_000_000, help='初始资金（默认100万）')
    args = parser.parse_args()

    main(port=args.port, open_browser=not args.no_browser, initial_cash=args.cash)
