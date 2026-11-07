"""核对评测子集，并将同一旧Judge的互补预测合并为全200题指标。"""
import argparse
import json
import math
from pathlib import Path

from .metrics import _binary_metrics


def validate_evaluation_set(examples, expected_count, expected_pass):
    if not 0 < expected_count or not 0 <= expected_pass <= expected_count:
        raise ValueError('invalid expected evaluation counts')
    if len(examples) != expected_count or sum(ex.target for ex in examples) != expected_pass:
        raise ValueError(f'expected {expected_count} questions with {expected_pass} positives')
    if len({ex.example_id for ex in examples}) != len(examples):
        raise ValueError('duplicate evaluation example IDs')
    if any(ex.condition_type != 'cross_view_requirement' for ex in examples):
        raise ValueError('evaluation must use the joint C requirement')


def merge_predictions(gold, partitions):
    by_id = {}
    for rows in partitions:
        for row in rows:
            key = row['example_id']
            if key in by_id or key not in gold or row['gold'] != gold[key]:
                raise ValueError(f'duplicate, unknown or mismatched gold: {key}')
            probability = row['pass_probability']
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError(f'invalid probability: {key}')
            if row['prediction'] not in (0, 1) or row['prediction'] != int(probability >= .5):
                raise ValueError(f'prediction disagrees with the fixed 0.5 threshold: {key}')
            by_id[key] = row
    if set(by_id) != set(gold):
        raise ValueError('predictions do not cover every labelled example exactly once')
    rows = [by_id[key] for key in gold]
    metrics = _binary_metrics([r['gold'] for r in rows], [r['prediction'] for r in rows])
    metrics['confusion'] = {
        name: sum(r['gold'] == g and r['prediction'] == p for r in rows)
        for name, g, p in [('tp', 1, 1), ('tn', 0, 0), ('fp', 0, 1), ('fn', 1, 0)]
    }
    metrics['pass_precision'] = metrics['confusion']['tp'] / max(1, sum(r['prediction'] for r in rows))
    return rows, metrics


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifests', type=Path, nargs=2, required=True)
    parser.add_argument('--evaluations', type=Path, nargs=2, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    manifests = [r for path in args.manifests for r in read_rows(path)]
    gold = {r['example_id']: int(r['verdict'] == 'pass') for r in manifests}
    if len(manifests) != 200 or len(gold) != 200 or sum(gold.values()) != 50:
        raise ValueError('expected the approved 200 unique questions with 50 positives')
    if any(r['verdict'] not in ('pass', 'fail') or r['condition_type'] != 'cross_view_requirement'
           for r in manifests):
        raise ValueError('invalid manifest verdict or condition')
    summaries = [json.loads((path / 'summary.json').read_text()) for path in args.evaluations]
    if len({s['adapter'] for s in summaries}) != 1 or any(s['threshold'] != .5 for s in summaries):
        raise ValueError('only the same adapter with the same threshold may be combined')
    partitions = [read_rows(path / 'predictions.jsonl') for path in args.evaluations]
    if any(s['examples'] != len(rows) for s, rows in zip(summaries, partitions)):
        raise ValueError('source evaluation counts disagree')
    rows, metrics = merge_predictions(gold, partitions)
    summary = dict(label='C630_full200', adapter=summaries[0]['adapter'], examples=200,
                   positives=50, threshold=.5, metrics=metrics,
                   meets_requested_accuracy=metrics['accuracy'] >= .8,
                   source_evaluations=[str(p) for p in args.evaluations],
                   scope='same_scene_full_labelled_pool_old_judge_no_training')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
