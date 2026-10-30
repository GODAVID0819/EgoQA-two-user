"""把 human_review/curated_pass 的 JSON + 480 段视频整理成公开 HF dataset 的目录（视频用硬链接，不复制）。
公开版 dataset.json 去掉 reviewer 身份（reviewer_id / reviewer / updated_at）。
  python stage_curated_pass.py <stage_dir>
"""
import json
import os
import sys
from pathlib import Path

SRC = Path('/scratch/tx856/multi_user_qa/dual_full120/human_review')
DROP = ('reviewer_id', 'reviewer', 'updated_at')


def scrub(x):
    if isinstance(x, dict):
        return {k: scrub(v) for k, v in x.items() if k not in DROP}
    if isinstance(x, list):
        return [scrub(v) for v in x]
    return x


def main():
    stage = Path(sys.argv[1])
    stage.mkdir(parents=True, exist_ok=True)
    for name in ('vqa.json', 'dataset.json', 'video_manifest.json'):
        data = scrub(json.loads((SRC / 'curated_pass' / name).read_text()))
        (stage / name).write_text(json.dumps(data, indent=2, ensure_ascii=False))
    manifest = json.loads((SRC / 'curated_pass/video_manifest.json').read_text())
    for v in manifest['videos']:
        dst = stage / v['review_video']
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            os.link(SRC / v['review_video'], dst)
    text = ''.join(p.read_text() for p in stage.glob('*.json'))
    for bad in ('/scratch', '/home/', 'hm2991', 'tx856'):
        if bad in text:
            raise ValueError(f'公开 JSON 里仍有 {bad}')
    print('staged', len(manifest['videos']), 'videos')


if __name__ == '__main__':
    main()
