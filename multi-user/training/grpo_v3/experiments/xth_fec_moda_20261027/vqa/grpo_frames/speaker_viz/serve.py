"""speaker-only 可视化页面的只读服务（仅标准库）。

  python serve.py --port 8770
本机 ssh -L 8770:localhost:8770 <user>@<同一个登录节点>，浏览器打开打印出来的 http://localhost:8770/?token=... 。
只监听 127.0.0.1，并要求启动时随机生成的 token（登录节点多人共用）；视频支持 Range 便于拖动。
人工审核：页面上的 通过/不通过/备注 POST 到 /review，存进同目录 reviews.json（每题最新一条，带时间）。
"""
import argparse
import http.cookies
import json
import mimetypes
import os
import re
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
SAFE = re.compile(r'^[A-Za-z0-9_.-]+$')
REVIEWS = HERE / 'reviews.json'
VERDICTS = ('pass', 'fail', None)
LOCK = threading.Lock()


def load_reviews():
    return json.loads(REVIEWS.read_text()) if REVIEWS.is_file() else {}


class Handler(BaseHTTPRequestHandler):
    token = ''

    def log_message(self, fmt, *args):
        pass

    def authorized(self, query):
        if secrets.compare_digest(query.get('token', [''])[0], self.token):
            return True
        cookie = http.cookies.SimpleCookie(self.headers.get('Cookie', ''))
        return 'viz_token' in cookie and secrets.compare_digest(cookie['viz_token'].value, self.token)

    def do_GET(self):
        url = urlparse(self.path)
        if not self.authorized(parse_qs(url.query)):
            self.send_response(403)
            self.end_headers()
            self.wfile.write('缺少或错误的 token：请用 serve.py 打印的链接打开。'.encode())
            return
        if url.path in ('/', '/index.html'):
            return self.send_file(HERE / 'index.html', 'text/html; charset=utf-8', cookie=True)
        if url.path == '/data.json':
            return self.send_file(HERE / 'data.json', 'application/json; charset=utf-8')
        if url.path == '/reviews.json':
            with LOCK:
                return self.send_json(load_reviews())
        parts = url.path.split('/')[1:]
        # videos6/<evidence_id>/<user>.mp4
        if (len(parts) == 3 and parts[0] == 'videos6' and all(SAFE.match(x) for x in parts[1:])
                and parts[2].endswith('.mp4') and '..' not in parts[1:]):
            return self.send_file(HERE / 'videos6' / parts[1] / parts[2], 'video/mp4')
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        url = urlparse(self.path)
        if not self.authorized(parse_qs(url.query)) or url.path != '/review':
            self.send_response(403)
            self.end_headers()
            return
        n = int(self.headers.get('Content-Length') or 0)
        try:
            body = json.loads(self.rfile.read(min(n, 1 << 16)))
            qid, verdict, note = body['id'], body.get('verdict'), str(body.get('note') or '')[:4000]
            assert isinstance(qid, str) and qid in self.ids and verdict in VERDICTS
        except Exception:
            self.send_response(400)
            self.end_headers()
            return
        with LOCK:
            reviews = load_reviews()
            if verdict is None and not note:
                reviews.pop(qid, None)
            else:
                reviews[qid] = {'verdict': verdict, 'note': note, 'time': time.strftime('%Y-%m-%d %H:%M:%S')}
            tmp = REVIEWS.with_suffix('.tmp')
            tmp.write_text(json.dumps(reviews, ensure_ascii=False, indent=1))
            tmp.replace(REVIEWS)
        self.send_json(reviews)

    def send_json(self, obj):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_file(self, path, ctype, cookie=False):
        if not path.is_file():
            self.send_response(404)
            self.end_headers()
            return
        size = path.stat().st_size
        start, end = 0, size - 1
        m = re.match(r'bytes=(\d*)-(\d*)', self.headers.get('Range', ''))
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                start = int(m.group(1))
                end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
            else:
                start = max(size - int(m.group(2)), 0)
            if start > end:
                self.send_response(416)
                self.send_header('Content-Range', f'bytes */{size}')
                self.end_headers()
                return
            self.send_response(206)
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        else:
            self.send_response(200)
        self.send_header('Content-Type', ctype or mimetypes.guess_type(path.name)[0])
        self.send_header('Accept-Ranges', 'bytes')
        if cookie:
            self.send_header('Set-Cookie', f'viz_token={self.token}; HttpOnly; SameSite=Strict; Path=/')
        self.send_header('Content-Length', str(end - start + 1))
        self.end_headers()
        with path.open('rb') as f:
            f.seek(start)
            remaining = end - start + 1
            try:
                while remaining:
                    chunk = f.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8770)
    a = ap.parse_args()
    # 重启时可沿用旧链接：VIZ_TOKEN 环境变量（不放命令行，避免出现在 ps 里）
    Handler.token = os.environ.get('VIZ_TOKEN') or secrets.token_urlsafe(16)
    Handler.ids = {i['id'] for i in json.loads((HERE / 'data.json').read_text())['items']}
    server = ThreadingHTTPServer(('127.0.0.1', a.port), Handler)
    print(f'页面: http://localhost:{a.port}/?token={Handler.token}', flush=True)
    print('（需先在本机建立 ssh -L 隧道；Ctrl-C 停止）', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
