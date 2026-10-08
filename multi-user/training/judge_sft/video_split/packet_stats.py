"""Per-packet stats for a video-disjoint split: time windows, overlaps, row counts, CLIP signature."""
import os, json, collections, numpy as np
from pathlib import Path
SRC = Path(os.environ.get('QUESTION_SPLIT', '/scratch/tx856/multi_user_qa/manifests_842_question_split'))
rows = []
for split in ('train', 'test'):
    for l in open(SRC / split / 'train.jsonl'):
        r = json.loads(l); r['_orig_split'] = split; rows.append(r)
by_packet = collections.defaultdict(list)
for r in rows:
    by_packet[r['group_id']].append(r)
# formality rows have no frame_packet; their group_id is still the packet id
packets = {}
for pid, rs in by_packet.items():
    pdir = next((r['frame_packet'] for r in rs if r.get('frame_packet')), f'/scratch/hm2991/egolife_rlhf_evidence_v1/packets/{pid}')
    meta = json.load(open(Path(pdir) / 'packet.json'))
    h, m, s = map(float, meta['clip_clock'].split(':'))
    start = h * 3600 + m * 60 + s
    emb = np.load(Path(pdir) / 'clip_embeddings.f16.npy').astype(np.float32).reshape(-1, 512)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
    sig = emb.mean(0); sig /= np.linalg.norm(sig)
    cnt = collections.Counter((r['task'], r['verdict']) for r in rs)
    packets[pid] = dict(dir=pdir, day=meta['day'], start=start, end=start + meta['duration_seconds'],
                        tier=meta['selection']['tier'], sig=sig.tolist(),
                        counts={f'{t}:{v}': n for (t, v), n in cnt.items()},
                        orig={s: sum(r['_orig_split'] == s for r in rs) for s in ('train', 'test')})
ids = sorted(packets)
# connected components of time-overlapping packets (same day, windows intersect)
parent = {p: p for p in ids}
def find(x):
    while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
    return x
overlaps = []
for i, a in enumerate(ids):
    for b in ids[i + 1:]:
        A, B = packets[a], packets[b]
        if A['day'] == B['day']:
            ov = min(A['end'], B['end']) - max(A['start'], B['start'])
            if ov > 0:
                overlaps.append((a, b, ov)); parent[find(a)] = find(b)
comp = collections.defaultdict(list)
for p in ids: comp[find(p)].append(p)
json.dump({'packets': packets, 'components': list(comp.values()), 'overlaps': overlaps}, open('packet_stats.json', 'w'))
print('packets', len(ids), ' rows', len(rows))
print('overlapping packet pairs', len(overlaps), ' overlap seconds: min', min(o[2] for o in overlaps) if overlaps else 0, 'max', max(o[2] for o in overlaps) if overlaps else 0)
sizes = collections.Counter(len(c) for c in comp.values()); print('component sizes', dict(sorted(sizes.items())), ' n_components', len(comp))
print('days', dict(sorted(collections.Counter(p['day'] for p in packets.values()).items())))
print('tiers', dict(collections.Counter(p['tier'] for p in packets.values())))
g = [sum(v for k, v in p['counts'].items() if k.startswith('groundedness')) for p in packets.values()]
t = [sum(p['counts'].values()) for p in packets.values()]
print('groundedness rows per packet: min', min(g), 'max', max(g), 'mean', round(np.mean(g), 2), ' total rows per packet: min', min(t), 'max', max(t))
S = np.array([packets[p]['sig'] for p in ids]); C = S @ S.T; np.fill_diagonal(C, np.nan)
print('packet CLIP-signature cosine: mean', round(np.nanmean(C), 3), ' nearest-neighbour mean', round(np.nanmean(np.nanmax(C, 1)), 3), ' min pair', round(np.nanmin(C), 3))
