# Day 5–7 first-50 training dataset

Source annotation file: `rlhf_candidate_scores_day5_7_first_50.csv`

CSV SHA-256: `4D78A3BB104DDB064B2760F9F4AC1AFA1D83DFC01B1518F8376AC23FBA9CF80D`

## Eligibility and training scope

- 600 CSV rows and 100 six-candidate evidence packets are present.
- Packet orders 1–50 are `completed`: 50 evidence packets and 300 fully scored candidates.
- Packet orders 51–100 are `pending`: 50 packets and 300 candidates with blank scores.
- No completed packet is malformed, partially scored, skipped, or duplicated.
- All 50 completed packets are used: 45 packets (270 candidates) for training and 5 packets (30 candidates) for validation. Locked-test and reserve lists are empty.
- The deterministic packet-level split uses seed 42 and keeps all six candidates from each evidence packet together.
- The model consumes 100 unique full-video sources arranged as 50 ordered synchronized pairs.

The local `outputs/RLHF_dataset/human_labeling_day5-7_full` metadata has the same assignment ID (`day5_7_full_100`) and dataset fingerprint (`dbea4d3c4ab21c99f39c`). Among the 50 completed packets, 42 come from Day 5 and 8 from Day 6. Generation metadata describes 295 `neutral` and 5 `commonality` candidates. This lineage is retained for auditing and statistics only; it is not added to the reviewer prompt or loss.

## Human score support

| Criterion | Score 1 | Score 2 | Score 3 | `score > 1` | `score > 2` |
|---|---:|---:|---:|---:|---:|
| Evidence grounding | 114 | 59 | 127 | 186 | 127 |
| Answerability | 151 | 147 | 2 | 149 | 2 |
| Formality | 91 | 127 | 82 | 209 | 82 |

The second answerability threshold has only two positive examples out of 300. The support-aware split assigns one to training and one to validation; train support for answerability scores 1/2/3 is `134/135/1`, and validation support is `17/12/1`. Metrics involving answerability score 3 must still be treated as under-supported. The first run does not apply inverse-frequency `pos_weight`, because extreme weighting would make the result highly sensitive to annotation noise.

## Held-out evaluation

The five-packet internal validation split supports training diagnostics and checkpoint comparison, but it is small and comes from the same annotation batch. There is no internal test set. The future 5–10 packet annotation batch must use distinct evidence IDs and an `external_holdout` manifest. The evaluator compares those IDs with the checkpoint's stored training IDs and stops on any overlap.
