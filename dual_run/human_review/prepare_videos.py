"""把审核集中每个packet的六个用户各拼成一条10分钟视频（20个30秒clip，640x640 H.264，faststart，便于浏览器边播边拖）。

clip来源：本地可读缓存优先，否则从packet.json里的HuggingFace video_url下载。输出 videos/<packet_id>/<agent_dir>.mp4，已存在则跳过（可重跑续做）。
"""
import argparse
import json
import os
import shutil
import subprocess
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
BIN = Path('/scratch/hm2991/conda/envs/egoqa-ms-swift-v4.2.2-vllm024/bin')
LOCAL = [Path('/scratch/tx856/multi_user_qa/dual_full120/segments'), Path('/scratch/hm2991/grpo_dual_full120_combined_20261016/segments')]


def source(agent_dir, day, seg, tmp):
    for root in LOCAL:
        p = root / agent_dir / day / (seg['clip_id'] + '.mp4')
        if os.access(p, os.R_OK):
            return p, 'local'
    p = tmp / (seg['clip_id'] + '.mp4')
    for attempt in range(4):
        try:
            with urllib.request.urlopen(seg['video_url'], timeout=120) as r, p.open('wb') as f:
                shutil.copyfileobj(r, f, 1 << 20)
            return p, 'download'
        except OSError:
            if attempt == 3:
                raise
    raise AssertionError


def duration(path):
    out = subprocess.run([str(BIN / 'ffprobe'), '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', str(path)],
                         capture_output=True, text=True, check=True).stdout
    return float(out)


def build(task, out_root, threads):
    packet_id, day, video = task
    out = out_root / packet_id / (video['agent_dir'] + '.mp4')
    if out.exists():
        return out, 'skip'
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as d:
        tmp = Path(d)
        kinds = []
        lines = []
        for seg in sorted(video['segments'], key=lambda s: s['segment_index']):
            p, kind = source(video['agent_dir'], day, seg, tmp)
            kinds.append(kind)
            lines.append(f"file '{p}'\n")
        (tmp / 'list.txt').write_text(''.join(lines))
        part = out.with_suffix('.part.mp4')
        subprocess.run([str(BIN / 'ffmpeg'), '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', str(tmp / 'list.txt'),
                        '-vf', 'scale=640:640', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '28', '-threads', str(threads),
                        '-c:a', 'aac', '-b:a', '64k', '-movflags', '+faststart', str(part)], check=True)
        d = duration(part)
        if abs(d - 30 * len(lines)) > 2:
            raise ValueError(f'{out} 时长{d}s，应为{30 * len(lines)}s')
        part.rename(out)
    return out, f"{kinds.count('local')} local / {kinds.count('download')} download, {d:.1f}s"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sets', nargs='+', default=['trained_1', 'untrained_1'])
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--manifest', type=Path, help='用 video_manifest.json 的清单代替 --sets')
    a = parser.parse_args()
    tasks = {}
    if a.manifest:
        for v in json.loads(a.manifest.read_text())['videos']:
            tasks[(v['packet_id'], v['agent_dir'])] = (v['packet_id'], v['day'], v)
        a.sets = []
    for name in a.sets:
        for it in json.loads((HERE / (name + '.json')).read_text())['items']:
            for v in it['videos']:
                tasks[(it['packet_id'], v['agent_dir'])] = (it['packet_id'], it['window']['day'], v)
    print(len(tasks), 'videos', flush=True)
    failed = 0
    with ThreadPoolExecutor(a.workers) as pool:
        futures = {pool.submit(build, t, HERE / 'videos', a.threads): k for k, t in tasks.items()}
        for i, f in enumerate(as_completed(futures), 1):
            try:
                out, info = f.result()
                print(i, out.relative_to(HERE), info, flush=True)
            except Exception as exc:
                failed += 1
                print(i, futures[f], 'FAILED', type(exc).__name__, exc, flush=True)
    print('done, failed =', failed, flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
