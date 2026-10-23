"""把 base / step-204 pass-rate 评估生成的题整理成6个人工审核用json（每份50题），每题配六个用户的视频信息。

untrained_{1,2,3}.json / trained_{1,2,3}.json：
  1 = eval_20276697 验证输入 0-49；2 = eval_20276697 验证输入 50-99；3 = eval_20278753 验证输入 0-49（与1同输入同种子的重复生成）。
"""
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path('/scratch/tx856/multi_user_qa/dual_full120')
EVAL = ROOT / 'outputs/eval_passrate'
OUT = ROOT / 'human_review'
VALIDATION = ROOT / 'data/trainval/validation.jsonl'
SEGMENT_SECONDS = 30.0
SETS = [('1', '20276697', range(0, 50)), ('2', '20276697', range(50, 100)), ('3', '20278753', range(0, 50))]
MODELS = [('untrained', 'baseline'), ('trained', 'policy')]
JUDGES = ('formality', 'groundedness', 'all_six')


def parse_completion(text):
    body = re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()
    try:
        return json.loads(body), None
    except json.JSONDecodeError as exc:
        return None, str(exc)


def token_seconds(token):
    """EgoLife time_token HHMMSSFF -> 当天秒数。"""
    return int(token[:2]) * 3600 + int(token[2:4]) * 60 + int(token[4:6]) + int(token[6:8]) / 100


def videos(row):
    """六路视频，顺序与Generator看到的一致（speaker在前）。路径全部相对 packets/<packet_id>/，不含本机绝对路径。

    clip在窗口内的位置由time_token算出（data/frames下13个packet的帧source_segment_index全为0，不能用）。
    """
    packet_dir = Path(row['dataset_root']) / 'packets' / row['source_packet_id']
    packet = json.loads((packet_dir / 'packet.json').read_text())
    rel = [Path(p).relative_to(packet_dir) for p in row['generator_image_paths']]
    seen = defaultdict(list)
    for r in rel:
        seen[r.parts[1]].append(str(r))
    users = {u['frame_directory'].split('/')[-1]: u for u in packet['users']}
    sources = {u['agent_dir']: u for u in packet['source']['users']}
    window_start = token_seconds(packet['time_token'])
    result = []
    for dirname in dict.fromkeys(r.parts[1] for r in rel):
        u = users[dirname]
        timestamps = {f['path']: f['timestamp_seconds'] for f in u['frames']}
        segments = []
        for i, seg in enumerate(sources[u['agent_dir']]['segments']):
            start = token_seconds(seg['time_token']) - window_start
            if abs(start - i * SEGMENT_SECONDS) > 1e-6:
                raise ValueError(row['evidence_id'] + ' clip不是连续30秒')
            segments.append({'segment_index': i, 'clip_id': seg['clip_id'], 'time_token': seg['time_token'],
                             'window_start_seconds': start, 'window_end_seconds': start + SEGMENT_SECONDS,
                             'video_url': seg['video_url']})
        if len(segments) * SEGMENT_SECONDS != packet['duration_seconds']:
            raise ValueError(row['evidence_id'] + ' clip总长与窗口长度不符')
        result.append({'user_index': u['user_index'], 'agent_dir': u['agent_dir'], 'agent_id': u['agent_id'],
                       'agent_name': u['agent_name'], 'role': 'speaker' if u['user_index'] == row['asker_index'] else 'provider',
                       'segments': segments, 'frames_directory': u['frame_directory'], 'total_sampled_frames': u['frame_count'],
                       'generator_frames': [{'path': p, 'timestamp_seconds': timestamps[p]} for p in seen[dirname]]})
    if len(result) != 6 or result[0]['role'] != 'speaker':
        raise ValueError(row['evidence_id'] + ' 视频不是speaker在前的六路')
    window = {'packet_id': row['source_packet_id'], 'day': packet['day'], 'start_clock': packet['clip_clock'],
              'duration_seconds': packet['duration_seconds'], 'sampling_interval_seconds': packet['preprocessing']['sampling']['interval_seconds'],
              'note': 'frame路径相对 packets/<packet_id>/；window_start/end_seconds 与 timestamp_seconds 相对10分钟窗口起点；'
                      '每个clip 30秒，clip内时间 = 窗口时间 - window_start_seconds'}
    return window, result


def write_packet_manifest():
    """120个验证packet的当前存放位置（分在两个目录），供打包/分发。"""
    rows = [json.loads(s) for s in VALIDATION.read_text().splitlines() if s.strip()]
    packets = [{'validation_index': i, 'packet_id': r['source_packet_id'], 'asker_index': r['asker_index'],
                'evidence_id': r['evidence_id'], 'location': str(Path(r['dataset_root']) / 'packets' / r['source_packet_id']),
                'used_by_passrate_eval': i < 100} for i, r in enumerate(rows)]
    if len({p['packet_id'] for p in packets}) != len(packets):
        raise ValueError('验证packet有重复')
    (OUT / 'packets_manifest.json').write_text(json.dumps(packets, indent=2))


def load(job, label):
    rows, inputs = {}, {}
    for rank_dir in sorted((EVAL / ('eval_' + job)).glob('rank*')):
        record = json.loads((rank_dir / f'validation_{label}.json').read_text())
        if record['status'] != 'completed':
            raise ValueError(f'{job} {rank_dir.name} {label} 未完成')
        for line in (rank_dir / 'validation.jsonl').read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                inputs[r['evidence_id']] = r
        for r in record['rows']:
            rows[inputs[r['evidence_id']]['global_validation_index']] = (r, record['adapter'])
    return rows, {v['global_validation_index']: v for v in inputs.values()}


def main():
    write_packet_manifest()
    summary = {}
    for set_name, job, indices in SETS:
        for model, label in MODELS:
            rows, inputs = load(job, label)
            items = []
            for index in indices:
                r, adapter = rows[index]
                inp = inputs[index]
                qa, error = parse_completion(r['completion'])
                window, six = videos(inp)
                passed = {k: r['status'] == 'scored' and r['probabilities'][k] > .5 for k in JUDGES}
                options = qa.get('options') if isinstance(qa, dict) else None
                items.append({
                    'review_id': f'{model}_{set_name}_{index:03d}', 'model': model,
                    'checkpoint': 'base (untrained)' if adapter is None else 'step204 LoRA (job 20161348)', 'eval_job': job, 'validation_index': index,
                    'evidence_id': r['evidence_id'], 'seed': r['seed'],
                    'packet_id': inp['source_packet_id'], 'speaker': six[0]['agent_name'], 'window': window,
                    'question': None if qa is None else {
                        'question': qa.get('question'),
                        'options': [{'letter': 'ABCDE'[i], 'text': t} for i, t in enumerate(options)] if isinstance(options, list) and len(options) <= 5 else options,
                        'correct': qa.get('correct'), 'answer': qa.get('answer'), 'question_type': qa.get('question_type'),
                        'required_users': qa.get('required_users'), 'evidence': qa.get('evidence'),
                        'referred_timestamps': qa.get('referred_timestamps')},
                    'generator_output': qa, 'parse_error': error, 'raw_completion': r['completion'] if qa is None else None,
                    'judge': {'status': r['status'], 'pass_probability': r['probabilities'], 'pass': passed,
                              'pass_all_three': all(passed.values()), 'reward': r['reward'],
                              'criterion': 'pass = P(pass) > 0.5 (judge SFT criterion); unscored = fail'},
                    'videos': six,
                    'human_review': {'question_correct': None, 'answer_correct': None, 'needs_all_six': None, 'notes': ''}})
            name = f'{model}_{set_name}.json'
            meta = {'file': name, 'model': model, 'adapter': adapter, 'eval_job': job,
                    'validation_indices': [indices.start, indices.stop - 1], 'count': len(items),
                    'pass_all_three': sum(i['judge']['pass_all_three'] for i in items),
                    'packet_root': 'packets/<packet_id>/ (see packets_manifest.json)'}
            (OUT / name).write_text(json.dumps({'meta': meta, 'items': items}, indent=2, ensure_ascii=False))
            summary[name] = meta
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
