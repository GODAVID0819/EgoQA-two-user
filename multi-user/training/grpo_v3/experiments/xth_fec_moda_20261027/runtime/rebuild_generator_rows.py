"""用当前运行根的 prompts.build_video_generation_prompt 重建生成器输入行（不加 generation_profile，即无 brevity 段）。

只换 messages 里的文本 prompt：图片、帧数、speaker/provider 顺序必须与原行逐项相同；行顺序与其余字段（如 original_split）保留。
  python rebuild_generator_rows.py --input <旧jsonl> --output <新jsonl>
"""
import argparse
import json
from pathlib import Path

from training.grpo_v3.six_user_binary.data import make_row, validate_row

KEEP = ('images', 'generator_image_paths', 'generator_frame_counts', 'required_users', 'evidence_id',
        'source_packet_id', 'asker_index', 'dataset_root')


def rebuild(row):
    new = make_row(row['dataset_root'], row['source_packet_id'], row['asker_index'])
    changed = [k for k in KEEP if new[k] != row[k]]
    if changed:
        raise ValueError(f"{row['evidence_id']} 重建后媒体/身份字段变化：{changed}")
    extra = {k: v for k, v in row.items() if k not in new and k != 'generation_profile'}
    out = {**new, **extra}
    validate_row(out)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    rows = [json.loads(s) for s in a.input.read_text().splitlines() if s.strip()]
    out = []
    for i, row in enumerate(rows):
        out.append(rebuild(row))
        if i % 60 == 0:
            print('REBUILT', i, flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in out))
    print('REBUILD_PASSED', json.dumps({'rows': len(out), 'input': str(a.input), 'output': str(a.output)}), flush=True)


if __name__ == '__main__':
    main()
