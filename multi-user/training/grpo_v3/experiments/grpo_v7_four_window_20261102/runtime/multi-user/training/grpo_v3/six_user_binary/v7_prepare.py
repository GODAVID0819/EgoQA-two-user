"""从现场配置和真实 packet 准备 base 四窗训练输入。"""
from copy import deepcopy
import argparse
import json
from pathlib import Path, PurePosixPath
from collections import Counter


def configure(judge, train, *, root, c_adapter, speaker_model, steps=180):
    if type(steps) is not int or not 5 < steps <= 180:
        raise ValueError('总更新数须为6–180，默认完整一遍180步')
    for value in (root, c_adapter, speaker_model):
        if not PurePosixPath(value).is_absolute():
            raise ValueError('必须显式使用实际绝对路径')
    if not PurePosixPath(root).is_relative_to('/scratch') or train['max_pixels'] != 50176:
        raise ValueError('必须使用scratch和50176生成器像素策略')
    j, t = deepcopy(judge), deepcopy(train)
    if set(j['adapters']) != {'formality', 'groundedness', 'answerability'}:
        raise ValueError('必须保留三项Judge路由')
    j['adapters']['answerability'] = c_adapter
    j['v7'] = dict(speaker=dict(model_id=speaker_model, max_model_len=32768,
                              max_images=300, gpu_memory_utilization=.4),
                   c_prompt='v7_c_prompt.build_c_prompt')
    t.update(project_root=root, judge_config=root+'/configs/judge_v7.json',
             output_root=root+'/outputs', train_dataset=root+'/data/train_v7.jsonl',
             max_steps=steps, stop_after_steps=steps, warmup_steps=5,
             resume_from_checkpoint=None, initial_fresh_base=True, diversity_lambda=0.,
             gradient_accumulation_steps=8, dataset_shuffle=False,
             history_pool_directory=root+'/unused_history', run_attempt='v7_base_four_windows',
             v7_history_updates=8)
    for key in list(t):
        if key.startswith(('resume_', 'expected_resume_', 'recovery_', 'warm_start', 'v7_arm')):
            del t[key]
    for key in ('adapters','policy_adapter','initial_adapter'):
        t.pop(key,None)
    t['resume_from_checkpoint'] = None
    if 'save_steps' in t:
        t['save_steps'] = max(1, (int(t['save_steps'])+1)//2)
    t['walltime_basis'] = '待按本轮实际16槽耗时和明确预算重估，旧walltime只作参考'
    return j, t


def order_four_windows(rows, window_key):
    """优先保持原清单顺序，仅跳过会冲突或使余下清单无法分组的条件。"""
    if not rows or len(rows) % 4:
        raise ValueError('条件数必须可被四整除')
    remaining = list(enumerate(rows))
    counts = Counter(window_key(row) for row in rows)
    if max(counts.values()) > len(rows)//4:
        raise ValueError('同一窗口过多，无法分成四个不同窗口的组')
    output, mapping = [], []
    while remaining:
        groups_left = len(remaining)//4
        mandatory = {key for key,count in counts.items() if count == groups_left}
        selected, windows = [], set()
        for original,row in remaining:
            key = window_key(row)
            if key in windows:
                continue
            missing = mandatory-windows
            if key not in mandatory and len(selected)+len(missing) >= 4:
                continue
            selected.append((original,row))
            windows.add(key)
            if len(selected) == 4:
                break
        if len(selected) != 4:
            raise ValueError('四窗重排失败')
        ids = {index for index,_ in selected}
        remaining = [(index,row) for index,row in remaining if index not in ids]
        for index,row in selected:
            counts[window_key(row)] -= 1
            mapping.append(index)
            output.append(row)
    return output, mapping


def rebuild_row(row):
    from training.grpo_v3.six_user_binary.data import make_row
    rebuilt = make_row(row['dataset_root'], row['source_packet_id'], row['asker_index'],
                       generation_profile=row.get('generation_profile'))
    for key in ('evidence_id','required_users','images','generator_image_paths','generator_frame_counts'):
        if row[key] != rebuilt[key]:
            raise ValueError('重渲染改变真实输入字段：'+key)
    result = {**row, **rebuilt, 'prompt_version':'v7_attachment'}
    return result


def validate_dependencies(judge, train):
    from training.grpo_v3.six_user_binary.predictor import validate_config
    checked=validate_config(judge)
    adapter=Path(checked['adapters']['answerability'])
    meta=json.loads((adapter/'adapter_config.json').read_text())
    if meta.get('r')!=128 or meta.get('layers_to_transform')!=list(range(56,64)):
        raise ValueError('C adapter 与提供的630结构不符')
    speaker=Path(judge['v7']['speaker']['model_id'])
    model=json.loads((speaker/'config.json').read_text())
    if model.get('model_type')!='qwen2_5_vl' or model.get('vision_config',{}).get('patch_size')!=14:
        raise ValueError('盲答模型不是已指定的Qwen2.5-VL')
    if not (Path(train['policy_model'])/'config.json').is_file():
        raise FileNotFoundError('生成器base模型缺少config.json')
    return checked


def main():
    parser = argparse.ArgumentParser(description='准备v7配置与训练数据')
    for name in ('judge','train','root','c-adapter','speaker-model','output','dataset'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--steps', type=int, default=180)
    args = parser.parse_args()
    j,t = configure(json.loads(Path(args.judge).read_text()), json.loads(Path(args.train).read_text()),
                    root=args.root, c_adapter=args.c_adapter, speaker_model=args.speaker_model,
                    steps=args.steps)
    j=validate_dependencies(j,t)
    from training.grpo_v3.six_user_binary.diversity import window_id
    rows = [rebuild_row(json.loads(line)) for line in Path(args.dataset).read_text().splitlines() if line.strip()]
    if len(rows) != 720 or len({(r['source_packet_id'],r['asker_index']) for r in rows}) != 720:
        raise ValueError('正式输入必须为720个唯一条件')
    ordered,mapping = order_four_windows(rows, lambda row:window_id(row['source_packet_id']))
    out = Path(args.output)
    out.mkdir(exist_ok=False)
    (out/'configs').mkdir()
    (out/'data').mkdir()
    for name,value in [('configs/judge_v7.json',j),('configs/train_v7.json',t),('data/input_mapping.json',mapping)]:
        (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    with (out/'data/train_v7.jsonl').open('x',encoding='utf-8') as stream:
        for row in ordered:
            stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    (out/'rendered_generator_sample.txt').write_text(ordered[0]['messages'][0]['content'],encoding='utf-8')
    print('已准备配置、720行数据和原输入映射')

if __name__ == '__main__':
    main()

