# Merged 70-packet score-only training dataset

Training annotation file: `data_RLHF/reviewer_v1/rlhf_candidate_scores_merged_70_packets.csv`

CSV SHA-256: `32679019FD7C665A0632E9885405BDF13C77B51386EFC56E7B29B24192210CD7`

## Provenance and eligibility

- `rlhf_candidate_scores_day5_7_first_50.csv` contributes 50 completed evidence packets and 300 scored candidates.
- `rlhf_candidate_scores_day1_4_full_day_98_packets_xl6775.csv` contributes its 20 completed evidence packets and 120 scored candidates; its 78 pending packets are excluded.
- The two sources have no overlapping evidence IDs or candidate IDs.
- The merged file contains exactly 70 completed evidence packets, six candidates per packet, and 420 supervised candidates.
- `fea_total_score` and `aggregate_rank` are omitted from the merged CSV. The model receives only the three score columns as labels.
- The deterministic seed-42 split keeps each six-candidate packet intact: 60 packets (360 candidates) for training and 10 packets (60 candidates) for validation. Locked-test and reserve lists are empty.
- The 70 packets reference 140 unique full-video sources arranged as 70 ordered synchronized pairs.

## Score support

| Criterion | Score 1 | Score 2 | Score 3 | `score > 1` | `score > 2` |
|---|---:|---:|---:|---:|---:|
| Evidence grounding | 152 | 81 | 187 | 268 | 187 |
| Answerability | 226 | 178 | 16 | 194 | 16 |
| Formality | 99 | 204 | 117 | 321 | 117 |

The support-aware split retains every class on both sides. Train answerability support for scores 1/2/3 is `199/152/9`; validation support is `27/26/7`. Answerability score 3 remains the rarest class, but it is materially better supported than in the original 50-packet run.

## Held-out evaluation

The ten validation packets are used for early stopping and checkpoint selection, not as a final test set. A separately annotated 5–10 packet `external_holdout` should remain untouched until the training configuration is frozen. The evaluator rejects holdout evidence IDs that overlap the checkpoint's recorded training IDs.
