"""人工审核网页服务（仅标准库）。

  python serve.py --port 8765
然后本机 ssh -L 8765:localhost:8765 <user>@<同一个登录节点>，浏览器打开打印出来的 http://localhost:8765/?token=... 。

只监听127.0.0.1，并要求启动时随机生成的token（登录节点是多人共用的）。
审核结果写在 reviews/<set>.json（每次保存原子替换），重启服务不会丢。
"""
import argparse
import http.cookies
import json
import mimetypes
import os
import re
import secrets
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
VIDEOS = HERE / 'videos'
REVIEWS = HERE / 'reviews'
SAFE = re.compile(r'^[A-Za-z0-9_.-]+$')
LOCK = threading.Lock()


def load_set(name):
    data = json.loads((HERE / (name + '.json')).read_text())
    for it in data['items']:
        for v in it['videos']:
            v['generator_frame_count'] = len(v.pop('generator_frames'))
            v['video'] = f"/video/{it['packet_id']}/{v['agent_dir']}.mp4"
            v['video_ready'] = (VIDEOS / it['packet_id'] / (v['agent_dir'] + '.mp4')).is_file()
    return data


def read_reviews(name):
    p = REVIEWS / (name + '.json')
    return json.loads(p.read_text()) if p.exists() else {}


class Handler(BaseHTTPRequestHandler):
    token = ''
    sets = ()
    reviewer = ''

    def log_message(self, fmt, *args):
        pass

    def authorized(self, query):
        if query.get('token', [''])[0] == self.token:
            return True
        cookie = http.cookies.SimpleCookie(self.headers.get('Cookie', ''))
        return 'review_token' in cookie and secrets.compare_digest(cookie['review_token'].value, self.token)

    def send_json(self, value, status=200):
        body = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def deny(self):
        self.send_response(403)
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        self.end_headers()
        self.wfile.write('缺少或错误的token：请使用 serve.py 启动时打印的链接打开。'.encode())

    def do_GET(self):
        url = urlparse(self.path)
        query = parse_qs(url.query)
        if not self.authorized(query):
            return self.deny()
        if url.path == '/':
            body = (HERE / 'index.html').read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Set-Cookie', f'review_token={self.token}; HttpOnly; SameSite=Strict; Path=/')
            self.end_headers()
            self.wfile.write(body)
        elif url.path == '/api/config':
            self.send_json({'sets': list(self.sets), 'reviewer': self.reviewer})
        elif url.path == '/api/set':
            name = query.get('name', [''])[0]
            if name not in self.sets:
                return self.send_json({'error': 'unknown set'}, 404)
            self.send_json({**load_set(name), 'reviews': read_reviews(name)})
        elif url.path.startswith('/video/'):
            parts = url.path.split('/')[2:]
            if len(parts) != 2 or not all(SAFE.match(p) for p in parts):
                return self.send_json({'error': 'bad path'}, 400)
            self.send_file(VIDEOS / parts[0] / parts[1])
        else:
            self.send_json({'error': 'not found'}, 404)

    def send_file(self, path):
        """支持Range请求，浏览器才能拖动进度条。"""
        if not path.is_file():
            return self.send_json({'error': 'video not ready'}, 404)
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
        self.send_header('Content-Type', mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
        self.send_header('Accept-Ranges', 'bytes')
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

    def do_POST(self):
        url = urlparse(self.path)
        if not self.authorized(parse_qs(url.query)):
            return self.deny()
        if url.path != '/api/review':
            return self.send_json({'error': 'not found'}, 404)
        body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')
        name, rid, verdict = body.get('set'), body.get('review_id'), body.get('verdict')
        if name not in self.sets or verdict not in ('pass', 'fail', None):
            return self.send_json({'error': 'bad request'}, 400)
        ids = {it['review_id'] for it in json.loads((HERE / (name + '.json')).read_text())['items']}
        if rid not in ids:
            return self.send_json({'error': 'unknown review_id'}, 400)
        with LOCK:
            (REVIEWS / name).parent.mkdir(parents=True, exist_ok=True)
            reviews = read_reviews(name)
            if verdict is None:
                reviews.pop(rid, None)
            else:
                reviews[rid] = {'verdict': verdict, 'notes': str(body.get('notes', ''))[:5000],
                                'reviewer': str(body.get('reviewer', ''))[:100],
                                'updated_at': datetime.now(timezone.utc).isoformat(timespec='seconds')}
            tmp = REVIEWS / (name + '.json.tmp')
            tmp.write_text(json.dumps(reviews, indent=2, ensure_ascii=False))
            os.replace(tmp, REVIEWS / (name + '.json'))
        self.send_json({'ok': True, 'review': reviews.get(rid)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--sets', nargs='+', default=['trained_1', 'untrained_1'])
    parser.add_argument('--reviewer', default=os.environ.get('USER', ''))
    a = parser.parse_args()
    for name in a.sets:
        if not (HERE / (name + '.json')).is_file():
            raise SystemExit(f'找不到 {name}.json')
    Handler.token = secrets.token_urlsafe(16)
    Handler.sets = tuple(a.sets)
    Handler.reviewer = a.reviewer
    server = ThreadingHTTPServer(('127.0.0.1', a.port), Handler)
    print(f'审核页面: http://localhost:{a.port}/?token={Handler.token}', flush=True)
    print('（需先在本机建立 ssh -L 隧道；Ctrl-C 停止）', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
