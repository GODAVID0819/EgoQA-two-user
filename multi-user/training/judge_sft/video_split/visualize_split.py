"""Self-contained HTML report to inspect a video-disjoint train/test split."""
import os, base64, collections, html, io, json, sys
import numpy as np
from pathlib import Path
from PIL import Image

QUESTION_SPLIT = Path(os.environ.get('QUESTION_SPLIT', '/scratch/tx856/multi_user_qa/manifests_842_question_split'))
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('manifests_842_video_split')
stats = json.load(open('packet_stats.json')); P = stats['packets']
comps = [sorted(c) for c in stats['components']]; comp_of = {p: i for i, c in enumerate(comps) for p in c}
test_set = set(json.load(open('split_assignment.json'))['test_packets'])
split = {p: ('test' if p in test_set else 'train') for p in P}

def load(path): return [json.loads(l) for l in open(path)]
old = {s: load(QUESTION_SPLIT / s / 'train.jsonl') for s in ('train', 'test')}
new = {s: load(OUT_DIR / s / 'train.jsonl') for s in ('train', 'test')}
def key(r):
    t = r['task']
    if t == 'answerability': t += ' / ' + ('all-six' if r['condition_type'] == 'combined_all_six_users' else 'speaker-only')
    return t, r['verdict']

# ---------- frame-level nearest-neighbour similarity ----------
def frames(p, step=15):
    e = np.load(Path(P[p]['dir']) / 'clip_embeddings.f16.npy').astype(np.float32)[:, ::step].reshape(-1, 512)
    return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)
F = {p: frames(p) for p in P}
train_p = [p for p in P if split[p] == 'train']; test_p = [p for p in P if split[p] == 'test']
TrM = np.concatenate([F[p] for p in train_p]); owner = np.concatenate([[i] * len(F[p]) for i, p in enumerate(train_p)])
def nn_sim(p, exclude_comp):
    sims = F[p] @ TrM.T
    if exclude_comp is not None:
        mask = np.array([comp_of[train_p[o]] == exclude_comp for o in owner]); sims[:, mask] = -1
    return sims.max(1)
test_nn = {p: float(np.median(nn_sim(p, None))) for p in test_p}
train_nn = {p: float(np.median(nn_sim(p, comp_of[p]))) for p in train_p}   # leave own overlap group out

# ---------- nearest train packet by signature (for side-by-side) ----------
S = {p: np.array(P[p]['sig']) for p in P}
def nearest_train(p):
    c = [(float(S[p] @ S[q]), q) for q in train_p if comp_of[q] != comp_of[p]]
    return max(c)

def thumb(p, frame_idx=150, w=104):
    meta = json.load(open(Path(P[p]['dir']) / 'packet.json'))
    out = []
    for u in meta['users']:
        fr = sorted(u['frames'], key=lambda x: int(x['frame_index']))[frame_idx]
        im = Image.open(Path(P[p]['dir']) / fr['path']).convert('RGB'); im.thumbnail((w, w))
        b = io.BytesIO(); im.save(b, 'JPEG', quality=70)
        out.append((u['agent_name'], base64.b64encode(b.getvalue()).decode()))
    return out
def strip(p):
    return '<div class="strip">' + ''.join(
        f'<figure><img src="data:image/jpeg;base64,{b}" alt="{html.escape(n)}"><figcaption>{html.escape(n)}</figcaption></figure>'
        for n, b in thumb(p)) + '</div>'

# ---------- tables ----------
def comp_table():
    keys = sorted({key(r) for s in old for r in old[s]})
    trs = []
    for k in keys:
        c = [sum(key(r) == k for r in d[s]) for d in (old, new) for s in ('train', 'test')]
        trs.append(f'<tr><td>{k[0]}</td><td>{k[1]}</td>' + ''.join(f'<td class="num">{x}</td>' for x in c) + '</tr>')
    tot = [len(d[s]) for d in (old, new) for s in ('train', 'test')]
    trs.append('<tr class="total"><td colspan=2>total rows</td>' + ''.join(f'<td class="num">{x}</td>' for x in tot) + '</tr>')
    vids = [len({r['group_id'] for r in d[s]}) for d in (old, new) for s in ('train', 'test')]
    trs.append('<tr class="total"><td colspan=2>video packets</td>' + ''.join(f'<td class="num">{x}</td>' for x in vids) + '</tr>')
    ov = [len({r['group_id'] for r in d['train']} & {r['group_id'] for r in d['test']}) for d in (old, new)]
    trs.append(f'<tr class="total"><td colspan=2>packets shared by train and test</td><td class="num" colspan=2>{ov[0]}</td><td class="num" colspan=2>{ov[1]}</td></tr>')
    return ('<table><tr><th rowspan=2>task</th><th rowspan=2>label</th><th colspan=2>question split (old)</th><th colspan=2>video split (new)</th></tr>'
            '<tr><th>train</th><th>test</th><th>train</th><th>test</th></tr>' + ''.join(trs) + '</table>')
def day_table():
    days = sorted({P[p]['day'] for p in P}); trs = []
    for d in days:
        g = {s: sum(1 for r in new[s] if r['task'] == 'groundedness' and P[r['group_id']]['day'] == d) for s in new}
        n = {s: sum(1 for p in P if P[p]['day'] == d and split[p] == s) for s in ('train', 'test')}
        trs.append(f'<tr><td>{d}</td><td class="num">{n["train"]}</td><td class="num">{n["test"]}</td><td class="num">{g["train"]}</td><td class="num">{g["test"]}</td><td class="num">{g["test"]/max(1,g["train"]+g["test"]):.2f}</td></tr>')
    return '<table><tr><th>day</th><th>train packets</th><th>test packets</th><th>train groundedness</th><th>test groundedness</th><th>test share</th></tr>' + ''.join(trs) + '</table>'

# ---------- timeline SVG ----------
def timeline():
    days = sorted({P[p]['day'] for p in P}); W, row_h, lane_h, left = 980, 0, 9, 60
    t0 = min(P[p]['start'] for p in P); t1 = max(P[p]['end'] for p in P)
    x = lambda t: left + (t - t0) / (t1 - t0) * (W - left - 10)
    parts = []; y = 10
    for d in days:
        ps = sorted((p for p in P if P[p]['day'] == d), key=lambda p: P[p]['start'])
        lanes = []
        for p in ps:
            for li, end in enumerate(lanes):
                if P[p]['start'] >= end: lanes[li] = P[p]['end']; break
            else: lanes.append(P[p]['end']); li = len(lanes) - 1
            cls = 'bar-test' if split[p] == 'test' else 'bar-train'
            parts.append(f'<rect class="{cls}" x="{x(P[p]["start"]):.1f}" y="{y + li*lane_h}" width="{max(2, x(P[p]["end"]) - x(P[p]["start"])):.1f}" height="{lane_h-2}"><title>{p} ({split[p]})</title></rect>')
        parts.append(f'<text class="lbl" x="4" y="{y + 8}">{d}</text>')
        y += max(1, len(lanes)) * lane_h + 8
    for h in range(int(t0 // 3600), int(t1 // 3600) + 1):
        parts.append(f'<line class="grid" x1="{x(h*3600):.1f}" x2="{x(h*3600):.1f}" y1="4" y2="{y}"/><text class="tick" x="{x(h*3600)+2:.1f}" y="{y+12}">{h:02d}:00</text>')
    return f'<svg viewBox="0 0 {W} {y+18}" class="chart" role="img" aria-label="packet timeline by day">{"".join(parts)}</svg>'

# ---------- PCA scatter SVG ----------
def scatter():
    ids = sorted(P); X = np.stack([S[p] for p in ids]); X = X - X.mean(0)
    _, _, vt = np.linalg.svd(X, full_matrices=False); Y = X @ vt[:2].T
    W, H, m = 480, 360, 24; lo, hi = Y.min(0), Y.max(0)
    px = lambda v: m + (v[0] - lo[0]) / (hi[0] - lo[0]) * (W - 2*m); py = lambda v: H - m - (v[1] - lo[1]) / (hi[1] - lo[1]) * (H - 2*m)
    dots = ''.join(f'<circle class="{"dot-test" if split[p]=="test" else "dot-train"}" cx="{px(y):.1f}" cy="{py(y):.1f}" r="{5 if split[p]=="test" else 3.5}"><title>{p} ({split[p]}, {P[p]["day"]})</title></circle>' for p, y in zip(ids, Y))
    return f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="CLIP PCA scatter">{dots}</svg>'

def hist():
    bins = np.linspace(0.80, 1.0, 21); W, H, m = 480, 220, 30
    a, _ = np.histogram(list(train_nn.values()), bins); b, _ = np.histogram(list(test_nn.values()), bins)
    a = a / max(1, a.sum()); b = b / max(1, b.sum()); top = max(a.max(), b.max())
    bw = (W - 2*m) / len(a); parts = []
    for i in range(len(a)):
        x0 = m + i * bw
        parts.append(f'<rect class="bar-train" x="{x0:.1f}" y="{H-m - a[i]/top*(H-2*m):.1f}" width="{bw/2-1:.1f}" height="{a[i]/top*(H-2*m):.1f}"/>')
        parts.append(f'<rect class="bar-test" x="{x0+bw/2:.1f}" y="{H-m - b[i]/top*(H-2*m):.1f}" width="{bw/2-1:.1f}" height="{b[i]/top*(H-2*m):.1f}"/>')
    for v in (0.80, 0.85, 0.90, 0.95, 1.0):
        xx = m + (v - 0.80) / 0.20 * (W - 2*m); parts.append(f'<text class="tick" x="{xx-10:.1f}" y="{H-m+14}">{v:.2f}</text>')
    return f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="nearest-frame similarity histogram">{"".join(parts)}</svg>'

pairs = []
for p in sorted(test_p, key=lambda p: (P[p]['day'], P[p]['start'])):
    sim, q = nearest_train(p)
    pairs.append(f'<section class="pair"><h3>{html.escape(p)} <span class="tag test">test</span> &nbsp; median nearest-frame sim to train = {test_nn[p]:.3f}</h3>{strip(p)}'
                 f'<h4>most similar train packet: {html.escape(q)} <span class="tag train">train</span> &nbsp; signature cosine {sim:.3f}</h4>{strip(q)}</section>')

doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Video Split Check</title><style>
:root{{--bg:#fbfbfa;--fg:#1d1d1b;--muted:#6b6b66;--line:#ddddd8;--train:#2f6fb0;--test:#d9752b;--card:#ffffff}}
@media (prefers-color-scheme: dark){{:root{{--bg:#161615;--fg:#ececea;--muted:#a3a39e;--line:#3a3a37;--train:#6aa6e0;--test:#f0a05c;--card:#1f1f1d}}}}
body{{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif;margin:0;padding:24px 16px;max-width:1040px;margin-inline:auto}}
h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:17px;margin:32px 0 8px;border-top:1px solid var(--line);padding-top:16px}} h3{{font-size:14px;margin:12px 0 4px}} h4{{font-size:13px;margin:8px 0 4px;color:var(--muted);font-weight:500}}
p.note{{color:var(--muted);margin:4px 0 12px}} table{{border-collapse:collapse;margin:8px 0;font-size:13px}} th,td{{border:1px solid var(--line);padding:3px 8px}} th{{background:var(--card)}} td.num{{text-align:right;font-variant-numeric:tabular-nums}} tr.total td{{font-weight:600}}
.chart{{width:100%;height:auto;background:var(--card);border:1px solid var(--line);border-radius:6px}} .bar-train,.dot-train{{fill:var(--train)}} .bar-test,.dot-test{{fill:var(--test)}} .grid{{stroke:var(--line)}} .lbl,.tick{{fill:var(--muted);font-size:10px}}
.legend span{{display:inline-block;margin-right:14px}} .sw{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}}
.row2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}}
.pair{{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:8px 12px;margin:12px 0}} .strip{{display:flex;gap:4px;flex-wrap:wrap}} figure{{margin:0}} figure img{{display:block;width:104px;height:auto;border-radius:3px}} figcaption{{font-size:10px;color:var(--muted);text-align:center}}
.tag{{font-size:11px;padding:1px 6px;border-radius:8px;color:#fff}} .tag.test{{background:var(--test)}} .tag.train{{background:var(--train)}}
</style></head><body>
<h1>Video-disjoint split check</h1>
<p class="note">Train and test share no video packet and no time-overlapping footage (packets that overlap in time are kept in the same split). Test row counts match the old question split.</p>
<div class="legend"><span><i class="sw" style="background:var(--train)"></i>train</span><span><i class="sw" style="background:var(--test)"></i>test</span></div>
<h2>1. Row counts</h2>{comp_table()}
<h2>2. Day balance</h2>{day_table()}
<h2>3. Timeline (each bar = one 10-minute packet; stacked bars overlap in time)</h2>{timeline()}
<div class="row2"><div><h2>4. CLIP signature PCA</h2><p class="note">One dot per packet (mean CLIP embedding of all 1,800 frames). Test should be spread among train.</p>{scatter()}</div>
<div><h2>5. Nearest-frame similarity</h2><p class="note">Per packet: median over its frames of the best cosine match among train frames. Train packets are scored against train packets outside their own overlap group (baseline); test against all train. Similar distributions = test looks like train without sharing footage. Median train {np.median(list(train_nn.values())):.3f}, test {np.median(list(test_nn.values())):.3f}.</p>{hist()}</div></div>
<h2>6. Test packets next to their most similar train packet</h2><p class="note">One frame per user at t≈300 s (middle of the 10-minute window).</p>{''.join(pairs)}
</body></html>"""
out = OUT_DIR / 'split_report.html'; out.write_text(doc, encoding='utf-8'); print(out, f'{out.stat().st_size/1e6:.1f} MB')
