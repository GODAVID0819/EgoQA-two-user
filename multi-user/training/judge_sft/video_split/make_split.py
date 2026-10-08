"""Search a video-disjoint train/test split with the same test row counts as the question split."""
import os, json, collections, random, numpy as np
from pathlib import Path
SRC = Path(os.environ.get('QUESTION_SPLIT', '/scratch/tx856/multi_user_qa/manifests_842_question_split'))
stats = json.load(open('packet_stats.json'))
P = stats['packets']; comps = [sorted(c) for c in stats['components']]
rows = {s: [json.loads(l) for l in open(SRC / s / 'train.jsonl')] for s in ('train', 'test')}
allrows = rows['train'] + rows['test']

def key(r):
    t = r['task']
    if t == 'answerability': t += '/' + ('all6' if r['condition_type'] == 'combined_all_six_users' else 'speaker')
    return f"{t}:{r['verdict']}"
pc = collections.defaultdict(collections.Counter)
for r in allrows: pc[r['group_id']][key(r)] += 1
KEYS = sorted({k for c in pc.values() for k in c})
def task_n(c, t): return sum(v for k, v in c.items() if k.startswith(t))
target = collections.Counter(key(r) for r in rows['test'])           # current test composition
T = {t: task_n(target, t) for t in ('groundedness', 'formality', 'answerability')}
day_of = {p: P[p]['day'] for p in P}
day_total = collections.Counter(); [day_total.update({day_of[p]: task_n(pc[p], 'groundedness')}) for p in P]
frac = T['groundedness'] / sum(day_total.values())
ccount = [sum((pc[p] for p in c), collections.Counter()) for c in comps]
S = {p: np.array(P[p]['sig']) for p in P}

def score(test_idx):
    tot = sum((ccount[i] for i in test_idx), collections.Counter())
    hard = sum(abs(task_n(tot, t) - T[t]) for t in T)
    label = sum(abs(tot[k] - target[k]) for k in KEYS)               # per-label count mismatch
    dc = collections.Counter()
    for i in test_idx:
        for p in comps[i]: dc[day_of[p]] += task_n(pc[p], 'groundedness')
    day = sum(abs(dc[d] - frac * day_total[d]) for d in day_total)
    test_p = [p for i in test_idx for p in comps[i]]; train_p = [p for i, c in enumerate(comps) if i not in test_idx for p in c]
    Tr = np.stack([S[p] for p in train_p]); nn = np.array([float((Tr @ S[p]).max()) for p in test_p])
    vis = float((0.99 - nn).clip(min=0).sum())                       # test packets lacking a close train look-alike
    return 1000 * hard + 2 * label + 1.0 * day + 50 * vis, dict(hard=hard, label=label, day=round(day, 1), vis=round(vis, 3))

best = None
rng = random.Random(0)
idx_all = list(range(len(comps)))
for restart in range(60):
    order = idx_all[:]; rng.shuffle(order); cur = set(); g = 0
    for i in order:
        n = task_n(ccount[i], 'groundedness')
        if g + n <= T['groundedness']: cur.add(i); g += n
    s, _ = score(cur)
    for it in range(4000):
        a = rng.choice(list(cur)); b = rng.choice([i for i in idx_all if i not in cur])
        cand = (cur - {a}) | {b}
        if rng.random() < 0.3:                                        # occasional 2-for-1 / 1-for-2 moves
            c2 = rng.choice([i for i in idx_all if i not in cand]); cand = cand | {c2}
            if rng.random() < 0.5 and len(cand) > 2: cand = cand - {rng.choice(list(cand))}
        cs, _ = score(cand)
        if cs <= s: cur, s = cand, cs
    if best is None or s < best[0]: best = (s, set(cur)); print('restart', restart, 'score', round(s, 2), score(cur)[1], flush=True)
s, test_idx = best
test_packets = sorted(p for i in test_idx for p in comps[i])
json.dump({'test_packets': test_packets, 'score': s, 'detail': score(test_idx)[1]}, open('split_assignment.json', 'w'), indent=1)
print('FINAL', round(s, 2), score(test_idx)[1], 'test packets', len(test_packets))
