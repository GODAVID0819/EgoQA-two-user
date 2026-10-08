"""Write the video-disjoint manifests from split_assignment.json (rows unchanged, only re-assigned)."""
import os, collections, json, shutil
from pathlib import Path
SRC = Path(os.environ.get('QUESTION_SPLIT', '/scratch/tx856/multi_user_qa/manifests_842_question_split'))
OUT = Path('manifests_842_video_split')
test_packets = set(json.load(open('split_assignment.json'))['test_packets'])
stats = json.load(open('packet_stats.json'))
rows = [json.loads(l) for s in ('train', 'test') for l in open(SRC / s / 'train.jsonl')]
split = {'train': [r for r in rows if r['group_id'] not in test_packets],
         'test': [r for r in rows if r['group_id'] in test_packets]}
# no time-overlapping footage may cross the split
for c in stats['components']:
    assert len({p in test_packets for p in c}) == 1, f'overlap group split across train/test: {c}'
assert not ({r['group_id'] for r in split['train']} & {r['group_id'] for r in split['test']})
assert not ({r['provenance']['candidate_id'] for r in split['train']} & {r['provenance']['candidate_id'] for r in split['test']})
for s, rs in split.items():
    (OUT / s).mkdir(parents=True, exist_ok=True)
    with open(OUT / s / 'train.jsonl', 'w', encoding='utf-8') as f:
        for r in rs: f.write(json.dumps(r, ensure_ascii=False) + '\n')
    for name in ('prompt_snapshots.json',):
        shutil.copy(SRC / s / name, OUT / s / name)
def summary(rs):
    c = collections.Counter((r['task'], r.get('condition_type') or '-', r['verdict']) for r in rs)
    return {'rows': len(rs), 'packets': len({r['group_id'] for r in rs}),
            'counts': {f'{t}|{ct}|{v}': n for (t, ct, v), n in sorted(c.items())}}
audit = {'schema': 'egolife_judge_sft_video_disjoint_split_v1',
         'source_manifests': str(SRC),
         'policy': 'split by video packet; packets whose 10-minute windows overlap in time are kept in the same split; '
                   'test row counts per task and label match the question split',
         'test_packets': sorted(test_packets),
         'train': summary(split['train']), 'test': summary(split['test'])}
json.dump(audit, open(OUT / 'split_audit.json', 'w'), indent=1, ensure_ascii=False)
print({s: len(rs) for s, rs in split.items()}, 'packets', {s: audit[s]['packets'] for s in split})
