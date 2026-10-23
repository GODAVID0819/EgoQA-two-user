"""把三轮Judge反馈环评估中被接受（三个Judge都 P(pass)>0.5）的题整理成人工审核json，trained / untrained 各分三份，并写视频清单。

fbloop/untrained_{1,2,3}.json：job 20519345（base）；fbloop/trained_{1,2,3}.json：job 20519348（step204 LoRA）。
按 (验证输入序号, 样本号) 排序后连续三等分；每题给出被接受那一轮的QA、判分，以及之前各轮的QA和Judge反馈。
fbloop/video_manifest.json：这些题用到的全部六路视频（packets/<packet_id>/ 与 videos/<packet_id>/<agent_dir>.mp4 均为相对路径）。
"""
import json
from pathlib import Path

from build_review_json import JUDGES, ROOT, VALIDATION, parse_completion, videos

OUT = ROOT / 'human_review/fbloop'
RUNS = [('untrained', '20519345', 'base (untrained)'), ('trained', '20519348', 'step204 LoRA (job 20161348)')]
CRITERION = 'accepted = all three trained judges P(pass) > 0.5 within at most 3 rounds of generate -> judge -> feedback -> regenerate'


def question(qa):
    if not isinstance(qa, dict):
        return None
    options = qa.get('options')
    return {'question': qa.get('question'),
            'options': [{'letter': 'ABCDE'[i], 'text': t} for i, t in enumerate(options)] if isinstance(options, list) and len(options) <= 5 else options,
            'correct': qa.get('correct'), 'answer': qa.get('answer'), 'question_type': qa.get('question_type'),
            'required_users': qa.get('required_users'), 'evidence': qa.get('evidence'),
            'referred_timestamps': qa.get('referred_timestamps')}


def judge(a):
    return {'status': a['status'], 'invalid_reason': a['invalid_reason'], 'pass_probability': a['probabilities'],
            'pass': a['pass'], 'pass_all_three': a['all_pass']}


def load(job):
    record = {}
    for f in sorted((ROOT / f'outputs/feedback_loop/fbloop_{job}').glob('rank*/feedback_loop.json')):
        r = json.loads(f.read_text())
        if r['status'] != 'completed':
            raise ValueError(f'{f} 未完成')
        record.setdefault('adapter', r['adapter'])
        record.setdefault('items', []).extend(r['items'])
    return record


def main():
    inputs = [json.loads(s) for s in VALIDATION.read_text().splitlines() if s.strip()]
    by_evidence = {r['evidence_id']: r for r in inputs}
    OUT.mkdir(exist_ok=True)
    manifest, summary = {}, {}
    for model, job, checkpoint in RUNS:
        record = load(job)
        accepted = sorted((i for i in record['items'] if i['accepted_attempt']),
                          key=lambda i: (i['global_validation_index'], i['sample']))
        third = -(-len(accepted) // 3)
        for n in range(3):
            items = []
            for it in accepted[n * third:(n + 1) * third]:
                inp = by_evidence[it['evidence_id']]
                window, six = videos(inp)
                final = it['attempts'][it['accepted_attempt'] - 1]
                if not final['all_pass'] or any(not final['pass'][k] for k in JUDGES):
                    raise ValueError(it['evidence_id'] + ' 接受轮并非三项通过')
                qa, error = parse_completion(final['completion'])
                history = []
                for a in it['attempts'][:-1]:
                    prev, prev_error = parse_completion(a['completion'])
                    history.append({'attempt': a['attempt'], 'seed': a['seed'], 'question': question(prev),
                                    'parse_error': prev_error, 'raw_completion': a['completion'] if prev is None else None,
                                    'judge': judge(a), 'feedback_to_generator': a['feedback_out']})
                items.append({
                    'review_id': f"{model}_{n + 1}_{it['global_validation_index']:03d}_s{it['sample']}", 'model': model,
                    'checkpoint': checkpoint, 'eval_job': job, 'validation_index': it['global_validation_index'],
                    'sample': it['sample'], 'evidence_id': it['evidence_id'], 'accepted_attempt': it['accepted_attempt'],
                    'seed': final['seed'], 'packet_id': inp['source_packet_id'], 'speaker': six[0]['agent_name'], 'window': window,
                    'question': question(qa), 'generator_output': qa, 'parse_error': error,
                    'feedback_received': final['feedback_in'],
                    'judge': {**judge(final), 'criterion': CRITERION},
                    'previous_attempts': history,
                    'videos': six,
                    'human_review': {'question_correct': None, 'answer_correct': None, 'needs_all_six': None, 'notes': ''}})
                for v in six:
                    key = (inp['source_packet_id'], v['agent_dir'])
                    entry = manifest.setdefault(key, {
                        'packet_id': inp['source_packet_id'], 'agent_dir': v['agent_dir'], 'agent_name': v['agent_name'],
                        'day': window['day'], 'start_clock': window['start_clock'], 'duration_seconds': window['duration_seconds'],
                        'review_video': f"videos/{inp['source_packet_id']}/{v['agent_dir']}.mp4",
                        'frames_directory': f"packets/{inp['source_packet_id']}/{v['frames_directory']}",
                        'segments': [{k: s[k] for k in ('segment_index', 'clip_id', 'window_start_seconds', 'video_url')} for s in v['segments']],
                        'used_by': []})
                    entry['used_by'].append(f'{model}_{n + 1}')
            name = f'{model}_{n + 1}.json'
            meta = {'file': name, 'model': model, 'adapter': record['adapter'] and Path(record['adapter']).name, 'eval_job': job, 'criterion': CRITERION,
                    'count': len(items), 'total_accepted': len(accepted), 'total_questions': len(record['items']),
                    'accepted_at_attempt': {str(k): sum(i['accepted_attempt'] == k for i in items) for k in (1, 2, 3)},
                    'packet_root': 'packets/<packet_id>/ (see ../packets_manifest.json)',
                    'video_root': 'videos/<packet_id>/<agent_dir>.mp4 (see video_manifest.json)'}
            (OUT / name).write_text(json.dumps({'meta': meta, 'items': items}, indent=2, ensure_ascii=False))
            summary[name] = {k: meta[k] for k in ('count', 'total_accepted', 'accepted_at_attempt')}
    videos_list = sorted(manifest.values(), key=lambda v: (v['packet_id'], v['agent_dir']))
    for v in videos_list:
        v['used_by'] = sorted(set(v['used_by']))
    (OUT / 'video_manifest.json').write_text(json.dumps({
        'note': '每条视频 = 该用户在10分钟窗口内的20个30秒EgoLife clip按顺序拼接（审核用转码：640x640 H.264 crf28 faststart）；'
                'review_video 与 frames_directory 为相对路径；原始clip为 segments[].video_url',
        'packet_count': len({v['packet_id'] for v in videos_list}), 'video_count': len(videos_list), 'videos': videos_list},
        indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2))
    print('packets', len({v['packet_id'] for v in videos_list}), 'videos', len(videos_list))


if __name__ == '__main__':
    main()
