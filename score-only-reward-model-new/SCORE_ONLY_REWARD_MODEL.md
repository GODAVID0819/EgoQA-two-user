# Binary multimodal reviewer

This repository contains the isolated reviewer-training experiment. Its current checkpoint and split-manifest contract is `binary_reviewer_v2`.

## Training signal

The source annotations remain the three per-candidate human score columns:

- `evidence_grounding_score` -> Evidence Quality
- `answerability_score` -> Answerability
- `formality_score` -> QA Formality

The CSV values remain integers in `1..3`; no source sample is rewritten or deleted. The training view maps `1 -> fail (0)` and `2/3 -> pass (1)`. Each head emits two logits in `[fail, pass]` order.

For each active head, the code counts fail/pass labels from the training split only and computes `weight[c] = N / (2 * N_c)`. It then applies cross-entropy with those weights. Per-sample weighted losses are divided by sample count, not by the sum of selected weights, so weighting remains effective for the current one-candidate training steps. The active head losses are averaged equally; the former `0.4/0.4/0.2` task-priority weights are gone.

`aggregate_rank` and derived `fea_total_score` remain ignored. There is no utility head, pairwise/tie loss, ranking loss, `lambda_rank`, or GRPO objective.

## Checkpoints and evaluation

Checkpoints save `binary_heads.pt`, the fail/pass mapping, training-split class counts and weights, class-weight reduction, and equal head aggregation. Old ordinal checkpoints are rejected rather than silently loaded.

Per-head evaluation reports CE loss, accuracy, balanced accuracy, macro-F1, AUROC, a 2x2 confusion matrix, and fail/pass precision, recall, F1, and support. Training/validation uses training-split class weights. External holdout comparison uses the same unweighted CE for base and trained models so loss values remain comparable.

The head loss can influence deployment behavior through shared LoRA parameters, but it does not directly supervise generated `reason` or `fix` tokens. Deployment prompts separately require `verdict` as the first field.

## Stages and data boundaries

- Stage 0: Evidence head only, frozen backbone, no LoRA.
- Stage 1: Evidence head plus q/v LoRA in the last two shared language blocks.
- Stage 2: all three binary heads plus the shared LoRA path.

The checked 70-packet split remains packet-disjoint: 60 evidence packets for training and 10 for validation. The evidence IDs are unchanged; only active manifest metadata and label-support counts were migrated to binary form.

The exact current contract is documented in `training/grpo_v3/experiments/human_preference_reviewer/BINARY_REVIEWER_V2_CONTRACT.md`. The previous ordinal contract and exact old manifest snapshots are preserved in `training/grpo_v3/experiments/human_preference_reviewer/ARCHIVED_ORDINAL_REVIEWER_V1.md` and `archive/ordinal_v1_manifests/`.

## Local verification

```bash
python -m unittest discover -s tests/training/grpo_v3/experiments/human_preference_reviewer/v1 -p "test_*.py" -v
python -m compileall -q training/grpo_v3/experiments/human_preference_reviewer/v1
```

GPU smoke, overfit, and full training commands remain under the existing Stage 0 and v1 runbooks. Directory names are kept for operational compatibility; the semantic contract version is v2.
