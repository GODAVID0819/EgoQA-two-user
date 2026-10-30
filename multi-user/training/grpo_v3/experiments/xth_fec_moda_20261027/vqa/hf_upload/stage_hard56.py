"""把 hard56 子集（56 题）整理成公开 HF dataset 目录：题目 JSON + evidence + 每题六人输入帧（= GRPO 生成器输入帧）。

选题（vqa_eval/grpo_frames/subsets/hard56.json）：Qwen3-VL-8B 与 Qwen2.5-VL-7B 的 text_only（不给帧）都答错，且
  speaker-only（GRPO 帧）两个都答错（44 题），或 speaker-only 至少一个答对但人工复审通过（12 题）。
帧：frames.json 里每题各用户实际输入给生成器的帧（speaker 300 帧，provider 只有 keep_mask 存活帧），原始 1408² jpg，
  按 frames/<packet_id>/<agent_dir>/<文件名> 存放（同一 packet 的不同提问人共用文件）；文件属于别的用户，复制而不是硬链接。
公开 JSON 去掉 reviewer 身份、人工备注和机器绝对路径。
  python stage_hard56.py <stage_dir> [--no-frames]
"""
import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path('/scratch/tx856/multi_user_qa')
CURATED = ROOT / 'dual_full120/human_review/curated_pass'
GF = ROOT / 'vqa_eval/grpo_frames'
IDS = GF / 'subsets/hard56.json'
REVIEWS = GF / 'subsets/human_review_speaker_only_20261030.json'
RUNS = {'speaker_only': {'Qwen3-VL-8B-Instruct': 'qwen3vl8b_speaker1_shuf0_21321632',
                         'Qwen2.5-VL-7B-Instruct': 'qwen25vl7b_speaker1_shuf0_21321633'},
        'text_only': {'Qwen3-VL-8B-Instruct': 'qwen3vl8b_text_only_shuf0_21326471',
                      'Qwen2.5-VL-7B-Instruct': 'qwen25vl7b_text_only_shuf0_21326472'}}
DROP = ('reviewer_id', 'reviewer', 'updated_at')


def scrub(x):
    if isinstance(x, dict):
        return {k: scrub(v) for k, v in x.items() if k not in DROP}
    if isinstance(x, list):
        return [scrub(v) for v in x]
    return x


def shuffle(item, seed=0):
    """与 vqa_grpo_frames.py / 之前公开的 vqa_shuffled.json 相同的重排。"""
    letters = list(item['options'])
    order = letters[:]
    random.Random(f'{seed}:{item["id"]}').shuffle(order)
    return {**item, 'options': {n: item['options'][o] for n, o in zip(letters, order)},
            'correct': letters[order.index(item['correct'])], 'original_correct': item['correct'], 'option_order': order}


def main():
    stage = Path(sys.argv[1])
    copy_frames = '--no-frames' not in sys.argv
    stage.mkdir(parents=True, exist_ok=True)
    ids = json.loads(IDS.read_text())
    passed = set(json.loads(REVIEWS.read_text())['pass'])
    vqa = {i['id']: i for i in json.loads((CURATED / 'vqa.json').read_text())['items']}
    full = {i['id']: i for i in json.loads((CURATED / 'dataset.json').read_text())['items']}
    store = json.loads((GF / 'frames.json').read_text())
    preds = {s: {m: {r['id']: r for r in map(json.loads, (GF / 'outputs' / d / 'predictions.jsonl').open())}
                 for m, d in runs.items()} for s, runs in RUNS.items()}

    vqa_items, full_items, copies = [], [], {}
    for i in ids:
        it, fr = vqa[i], store['items'][i]
        agent_dir = {v['agent_name']: v['agent_dir'] for v in it['videos']}
        frames = {}
        for user in fr['required_users']:
            rel = []
            for p in fr['frames'][user]:
                dst = f"frames/{it['packet_id']}/{agent_dir[user]}/{Path(p).name}"
                copies[dst] = p
                rel.append(dst)
            frames[user] = rel
        speaker_correct = any(preds['speaker_only'][m][i]['is_correct'] for m in preds['speaker_only'])
        selection = 'human_pass_speaker_only_correct' if speaker_correct else 'both_models_wrong_text_and_speaker_only'
        assert selection == 'both_models_wrong_text_and_speaker_only' or i in passed, i
        providers = [u for u in fr['required_users'] if u in {v['agent_name'] for v in it['evidence_videos']} - {it['speaker']}]
        # videos[*].path 指向姊妹 repo egolife-six-user-mcqa-human-pass 里的 mp4（本 repo 只放帧）
        entry = {**it, 'selection': selection,
                 'evidence_provider_users': providers, 'frame_counts': dict(zip(fr['required_users'], fr['frame_counts'])),
                 'frames': frames}
        vqa_items.append(entry)
        baselines = {s: {m: {'pred_shuffled_seed0': preds[s][m][i]['pred'], 'correct': preds[s][m][i]['is_correct']}
                         for m in preds[s]} for s in preds}
        full_items.append({**scrub(full[i]), 'selection': selection, 'evidence_users': it['evidence_users'],
                           'evidence_users_selection': it['evidence_users_selection'],
                           'evidence_provider_users': providers, 'generator_media_metadata': fr['metadata'],
                           'frame_counts': entry['frame_counts'], 'frames': frames, 'baseline_results': baselines})

    meta = {'n_items': len(vqa_items), 'frame_resolution_raw': store['resolution']['raw'],
            'grpo_model_input_resolution': store['resolution'],
            'max_pixels': store['max_pixels'], 'sample_interval_seconds': 2.0}
    (stage / 'vqa.json').write_text(json.dumps({'meta': meta, 'items': vqa_items}, indent=2, ensure_ascii=False))
    (stage / 'vqa_shuffled.json').write_text(json.dumps(
        {'meta': {**meta, 'shuffle': 'random.Random(f"0:{id}").shuffle(letters)'},
         'items': [shuffle(x) for x in vqa_items]}, indent=2, ensure_ascii=False))
    (stage / 'dataset.json').write_text(json.dumps({'meta': meta, 'items': full_items}, indent=2, ensure_ascii=False))
    text = ''.join(p.read_text() for p in stage.glob('*.json'))
    for bad in ('/scratch', '/home/', 'hm2991', 'tx856'):
        if bad in text:
            raise ValueError(f'公开 JSON 里仍有 {bad}')
    print('items', len(vqa_items), 'selection', {s: sum(x['selection'] == s for x in vqa_items) for s in
                                                 {x['selection'] for x in vqa_items}}, 'frames', len(copies), flush=True)
    if copy_frames:
        for n, (dst, src) in enumerate(sorted(copies.items()), 1):
            out = stage / dst
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, out)
            if n % 5000 == 0:
                print('copied', n, flush=True)
        print('staged frames', len(copies))


if __name__ == '__main__':
    main()
