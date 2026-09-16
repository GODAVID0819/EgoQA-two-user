# Exploratory reshuffled 90-packet run

The current 70-packet development CSV and the previously evaluated 20-packet CSV are pooled, then split at the `evidence_id` packet level with deterministic seed `20260809`:

- 60 training packets / 360 candidates
- 10 validation packets / 60 candidates
- 20 test packets / 120 candidates

The pooled score-only CSV is `data_RLHF/reviewer_v1/rlhf_candidate_scores_reshuffled_90_packets_seed_20260809.csv` with SHA-256 `82E5493F9211B39DB8A0572D7AF0A1F78E1625ABCAFEE2BFC41FF5C3200D2F89`. The isolated test CSV is `data_RLHF/reviewer_v1/rlhf_candidate_scores_reshuffled_test_20_seed_20260809.csv` with SHA-256 `F5644442067D81BAAFB2034D42A0589AA92387DB7D6BB929E4F9CFB2B9B376E9`.

The split mixes the former partitions as follows:

| New split | Former development-70 | Former test-20 |
| --- | ---: | ---: |
| Train | 46 | 14 |
| Validation | 7 | 3 |
| Test | 17 | 3 |

Both binary classes occur for all three heads in every split after applying `1 → fail` and `2/3 → pass`. The original 1–3 score counts remain preserved in the CSV and archived dataset notes.

Because the old test annotations and results have already been inspected, this is an exploratory robustness rerun, not a new untouched benchmark. It must not replace a future independently annotated holdout when reporting final generalization.
