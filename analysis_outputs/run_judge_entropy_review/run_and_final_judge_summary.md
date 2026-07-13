# Run and Final-Judge Summary

## Basic run data

`total_items` is the number of evidence packets. `total_candidate_generations` counts attempt traces containing a generator call.

| Run | Items | Accepted | Rejected | Acceptance rate | Candidate generations | Total attempts | Average attempts | Median attempts | First-attempt accepts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| greedy | 50 | 21 | 29 | 0.420000 | 123 | 123 | 2.460000 | 3.000000 | 9 |
| sampling | 50 | 22 | 28 | 0.440000 | 121 | 121 | 2.420000 | 3.000000 | 12 |

## Final-attempt judge PASS/FAIL statistics

Each evidence packet contributes only its final attempt to this table.

| Run | Judge | PASS | FAIL | UNCERTAIN | MISSING | PASS rate |
|---|---|---:|---:|---:|---:|---:|
| greedy | qa_formality | 40 | 10 | 0 | 0 | 0.800000 |
| greedy | evidence_groundedness | 38 | 12 | 0 | 0 | 0.760000 |
| greedy | answerability | 28 | 22 | 0 | 0 | 0.560000 |
| sampling | qa_formality | 42 | 8 | 0 | 0 | 0.840000 |
| sampling | evidence_groundedness | 35 | 15 | 0 | 0 | 0.700000 |
| sampling | answerability | 28 | 22 | 0 | 0 | 0.560000 |

## Joint final-attempt outcomes

- **greedy**: all three PASS for 21 items (0.420000); failure combinations: `{"answerability": 8, "evidence_groundedness": 4, "evidence_groundedness|answerability": 7, "none": 21, "qa_formality": 3, "qa_formality|answerability": 6, "qa_formality|evidence_groundedness|answerability": 1}`
- **sampling**: all three PASS for 22 items (0.440000); failure combinations: `{"answerability": 8, "evidence_groundedness": 3, "evidence_groundedness|answerability": 9, "none": 22, "qa_formality": 3, "qa_formality|answerability": 2, "qa_formality|evidence_groundedness|answerability": 3}`
