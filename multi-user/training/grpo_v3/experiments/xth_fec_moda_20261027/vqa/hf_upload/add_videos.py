"""把子集用到的完整 10 分钟原始视频（每人一段，没有丢帧，640², 20 fps, 带音频）硬链接进 stage 目录。
来源是 egolife_six_user_mcqa_human_pass/videos（同一批 mp4）；路径与 vqa.json 里 videos[*].path 相同。
  python add_videos.py <stage_dir> <ids.json>
"""
import json
import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / 'egolife_six_user_mcqa_human_pass'


def main():
    stage, ids = Path(sys.argv[1]), set(json.loads(Path(sys.argv[2]).read_text()))
    items = [i for i in json.loads((SRC / 'vqa.json').read_text())['items'] if i['id'] in ids]
    assert len(items) == len(ids), (len(items), len(ids))
    paths = sorted({v['path'] for i in items for v in i['videos']})
    for p in paths:
        dst = stage / p
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.link(SRC / p, dst)
    print('linked videos', len(paths))


if __name__ == '__main__':
    main()
