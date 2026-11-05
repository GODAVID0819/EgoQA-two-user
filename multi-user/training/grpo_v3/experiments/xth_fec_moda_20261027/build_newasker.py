"""新一批生成的输入：120 个窗口每个挑 1 个新 asker（没被任何跑完的 feedback-loop 用过；
优先也没被取消的 21073148/9 用过），行取自 data/train_length_paired_ga.jsonl（generalanswer prompt）。
顺序 = lxr_run/data/validation_v2.jsonl 的窗口顺序（前 100 窗口与之前 cp348/continuous/gated 相同）。"""
import collections, glob, json, random
from pathlib import Path
B = Path('/scratch/tx856/multi_user_qa')
COMPLETE = ['20519345', '20519348', '20772096', '20819555', '21179169', '21676581', '21676583']
CANCELLED = ['21073148', '21073149']
def usage(jobs):
    u = collections.defaultdict(set)
    for j in jobs:
        files = glob.glob(str(B / f'*/outputs/feedback_loop/fbloop_{j}/rank*/validation.jsonl'))
        assert files, j
        for f in files:
            for l in open(f):
                w, a = json.loads(l)['evidence_id'].split('__ASKER_'); u[w].add(a)
    return u
done, canc = usage(COMPLETE), usage(CANCELLED)
rows = [json.loads(l) for l in open(B / 'lxr_run_ga/data/train_length_paired_ga.jsonl') if l.strip()]
by = {r['evidence_id']: r for r in rows}
order = [json.loads(l)['evidence_id'].split('__ASKER_')[0] for l in open(B / 'lxr_run/data/validation_v2.jsonl')]
rng = random.Random(20261104)
out, stats = [], collections.Counter()
for w in order:
    askers = sorted(e.split('__ASKER_')[1] for e in by if e.startswith(w + '__ASKER_'))
    assert len(askers) == 6, w
    fresh = [a for a in askers if a not in done[w] and a not in canc[w]]
    ok = [a for a in askers if a not in done[w]]
    assert ok, w
    pick = rng.choice(fresh or ok); stats['never_used' if fresh else 'only_in_cancelled'] += 1
    r = dict(by[f'{w}__ASKER_{pick}']); r.pop('original_split', None); out.append(r)
dst = B / 'lxr_run_ga/data/fbloop_newasker120_ga.jsonl'
dst.write_text(''.join(json.dumps(r) + '\n' for r in out))
print(len(out), dict(stats), dst)
