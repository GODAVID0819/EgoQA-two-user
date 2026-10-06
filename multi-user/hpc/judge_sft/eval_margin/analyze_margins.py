"""Threshold analysis of per-example margins from evaluate_groundedness_multi_025fps."""
import json, math, random, sys
from collections import defaultdict

W_FAIL, W_PASS = 1.5440, 0.7399          # class weights used in training
SHIFT = math.log(W_PASS / W_FAIL)        # ≈ -0.736: optimum offset induced by weighted BCE

rows = defaultdict(list)
for line in open(sys.argv[1]):
    r = json.loads(line)
    rows[r["adapter"]].append((r["margin"], r["gold"]))

def metrics(data, t):
    tp = sum(m > t and g == 1 for m, g in data); fn = sum(m <= t and g == 1 for m, g in data)
    tn = sum(m <= t and g == 0 for m, g in data); fp = sum(m > t and g == 0 for m, g in data)
    rp, rf = tp / (tp + fn), tn / (tn + fp)
    return (tp + tn) / len(data), (rp + rf) / 2, (tp + fp) / len(data)

def auroc(data):
    pos = [m for m, g in data if g == 1]; neg = [m for m, g in data if g == 0]
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))

def best_t(data):
    cands = sorted({m for m, _ in data})
    cands = [cands[0] - 1] + [(a + b) / 2 for a, b in zip(cands, cands[1:])] + [cands[-1] + 1]
    return max(cands, key=lambda t: metrics(data, t)[0])

def cv_accuracy(data, folds=2, repeats=50):
    """Pick the threshold on one half, score on the other: an honest tuned-threshold estimate."""
    accs = []
    for seed in range(repeats):
        d = data[:]; random.Random(seed).shuffle(d)
        for k in range(folds):
            test = d[k::folds]; train = [x for i, x in enumerate(d) if i % folds != k]
            accs.append(metrics(test, best_t(train))[0])
    return sum(accs) / len(accs)

print(f"weighted-BCE offset log(w_pass/w_fail) = {SHIFT:+.3f}\n")
hdr = f"{'model':8} {'AUROC':>6} | {'acc@0':>6} {'bal@0':>6} {'pass%@0':>7} | {'acc@-.736':>9} {'bal':>6} {'pass%':>6} | {'acc(CV-tuned t)':>15}"
print(hdr); print("-" * len(hdr))
for name in ["base", "epoch1", "epoch2", "epoch3"]:
    d = rows.get(name)
    if not d: continue
    a0, b0, p0 = metrics(d, 0.0); a1, b1, p1 = metrics(d, SHIFT)
    print(f"{name:8} {auroc(d):6.3f} | {a0:6.3f} {b0:6.3f} {p0:7.3f} | {a1:9.3f} {b1:6.3f} {p1:6.3f} | {cv_accuracy(d):15.3f}")
n = len(next(iter(rows.values()))); g = sum(x[1] for x in next(iter(rows.values())))
print(f"\nn={n}, gold pass rate={g/n:.3f} (always-pass accuracy)")
