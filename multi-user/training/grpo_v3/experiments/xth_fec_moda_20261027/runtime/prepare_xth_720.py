"""用我们已有的600训练+120验证行（同120窗口、同speaker划分）按 lxr condition_definition.json 的原顺序排成720条件训练数据。

lxr 的原始来源数据在 /scratch/xl6775 下不可读；窗口与 (窗口, speaker) 条件逐条核对一致后，直接沿用其 length-paired 顺序，
不重新计算长度配对。每行再用本 runtime 的 data.validate_row 核对 prompt/图片与当前 packet 一致（需要CPU作业，读取全部帧文件）。
  python prepare_xth_720.py --config configs/train_continuous.json --train <train.jsonl> --validation <validation.jsonl>
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from dual_runtime import condition_binding

HERE = Path(__file__).resolve().parent


def ordered_rows(train_rows, validation_rows, definition):
    if len(train_rows) != 600 or len(validation_rows) != 120:
        raise ValueError('来源应为600训练和120验证条件')
    by_key = {}
    for split, rows in (('train', train_rows), ('validation', validation_rows)):
        for row in rows:
            key = condition_binding(row)
            if key in by_key:
                raise ValueError(f'条件重复：{key}')
            by_key[key] = {**row, 'original_split': split}
    order = [condition_binding(item) for item in definition['train']]
    if len(order) != 720 or set(order) != set(by_key):
        raise ValueError('我们的720条件与 lxr condition_definition 不一致')
    for item in definition['train']:
        if by_key[condition_binding(item)]['original_split'] != item['original_split']:
            raise ValueError(f'原split不一致：{condition_binding(item)}')
    rows = [by_key[key] for key in order]
    if any(condition_binding(rows[i])[0] == condition_binding(rows[i + 1])[0] for i in range(0, 720, 2)):
        raise ValueError('同一更新的两条件来自同一窗口')
    if Counter(r['asker_index'] for r in rows) != Counter({i: 120 for i in range(6)}):
        raise ValueError('每个speaker应为120条件')
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--validate-media', action='store_true', help='逐行 validate_row（读帧文件，需在CPU作业里跑）')
    parser.add_argument('--output', type=Path, help='默认写到配置的 train_dataset')
    a = parser.parse_args()
    c = json.loads(a.config.read_text())
    read = lambda p: [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
    definition = json.loads((HERE / 'condition_definition.json').read_text())
    rows = ordered_rows(read(a.train), read(a.validation), definition)
    if a.validate_media:
        from training.grpo_v3.six_user_binary.data import validate_row
        for i, row in enumerate(rows):
            validate_row(row)
            if i % 60 == 0:
                print('VALIDATED_CONDITION', i, flush=True)
    destination = a.output or Path(c['train_dataset'])
    if destination.exists():
        if read(destination) != rows:
            raise FileExistsError(f'{destination} 已存在且内容不同')
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    print('DATA_720_PASSED', json.dumps({'rows': len(rows), 'media_validated': a.validate_media,
                                         'speakers': dict(Counter(r['asker_index'] for r in rows)),
                                         'original_split': dict(Counter(r['original_split'] for r in rows))}), flush=True)


if __name__ == '__main__':
    main()
