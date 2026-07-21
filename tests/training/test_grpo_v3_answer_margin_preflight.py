from __future__ import annotations

import math
import unittest

from training.grpo_v3_answer_margin_preflight import (
    FIXED_EVIDENCE_ID,
    validate_calibration,
    validate_scorer_probe,
)


LABELS = ("A", "B", "C", "D", "E")


def _score_request(offset: float = 0.0) -> dict:
    scores = {label: -float(index) + offset for index, label in enumerate(LABELS)}
    return {
        "evidence_id": FIXED_EVIDENCE_ID,
        "ordered_videos": [
            {
                "user": "A1",
                "path": "/scratch/evidence/A1.mp4",
                "sha256": "1" * 64,
                "size_bytes": 100,
                "processor_metadata": {"fps": 2.0, "num_frames": 16, "max_pixels": 1024, "min_pixels": 16},
            },
            {
                "user": "A5",
                "path": "/scratch/evidence/A5.mp4",
                "sha256": "2" * 64,
                "size_bytes": 200,
                "processor_metadata": {"fps": 2.0, "num_frames": 16, "max_pixels": 1024, "min_pixels": 16},
            },
        ],
        "sequence_scores": scores,
        "top1": "A",
        "label_details": {
            label: {"token_ids": [index + 1], "token_logprobs": [scores[label]], "sequence_logprob": scores[label]}
            for index, label in enumerate(LABELS)
        },
        "prompt_audit": {"passed": True, "leaked_fields": []},
    }


def passing_probe() -> dict:
    return {
        "schema_version": "answer_margin_scorer_probe_v1",
        "evidence_id": FIXED_EVIDENCE_ID,
        "health": {"status": "passed", "model": "Qwen/Qwen3-VL-2B-Instruct"},
        "trainable_parameter_count": 0,
        "runtime": {"device": "cuda:0", "gpu_name": "NVIDIA H100", "peak_memory_bytes": 123, "elapsed_seconds": 1.5},
        "requests": [_score_request(), _score_request(5e-6)],
    }


def calibration_rows() -> list[dict]:
    rows = []
    for group in range(8):
        for candidate in range(4):
            reward = group * 0.01 + (candidate * 0.1 if group < 6 else 0.0)
            rows.append(
                {
                    "reward": reward,
                    "record": {
                        "evidence_id": FIXED_EVIDENCE_ID,
                        "experiment_condition_id": "t05",
                        "temperature": 0.5,
                        "reward_call_index": group,
                        "candidate_index": candidate,
                        "masked": False,
                        "normalized_reward": reward,
                        "format_validation": {"status": "raw_valid"},
                        "permutation": [0, 1, 2, 3, 4],
                        "inverse_permutation": [0, 1, 2, 3, 4],
                        "label_scores": {
                            label: {"sequence_logprob": -float(index), "token_ids": [index], "token_logprobs": [-float(index)]}
                            for index, label in enumerate(LABELS)
                        },
                    },
                }
            )
    return rows


class AnswerMarginPreflightTests(unittest.TestCase):
    def test_scorer_probe_requires_repeated_real_dual_video_scores(self) -> None:
        result = validate_scorer_probe(passing_probe())
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["evidence_id"], FIXED_EVIDENCE_ID)
        self.assertEqual(result["request_count"], 2)

    def test_scorer_probe_rejects_score_drift_leakage_and_missing_runtime(self) -> None:
        probe = passing_probe()
        probe["requests"][1]["sequence_scores"]["A"] += 1e-2
        probe["requests"][1]["prompt_audit"] = {"passed": False, "leaked_fields": ["correct"]}
        probe["runtime"].pop("peak_memory_bytes")
        result = validate_scorer_probe(probe)
        self.assertEqual(result["status"], "failed")
        self.assertIn("repeat_scores_within_tolerance", result["failed_checks"])
        self.assertIn("prompt_leakage_scan_passed", result["failed_checks"])
        self.assertIn("runtime_metadata_complete", result["failed_checks"])

    def test_scorer_probe_rejects_nonfinite_wrong_evidence_and_trainable_scorer(self) -> None:
        probe = passing_probe()
        probe["evidence_id"] = "wrong"
        probe["trainable_parameter_count"] = 1
        probe["requests"][0]["sequence_scores"]["C"] = math.inf
        result = validate_scorer_probe(probe)
        self.assertEqual(result["status"], "failed")
        self.assertIn("fixed_evidence", result["failed_checks"])
        self.assertIn("zero_trainable_parameters", result["failed_checks"])
        self.assertIn("all_sequence_scores_finite", result["failed_checks"])

    def test_calibration_separates_completeness_from_research_thresholds(self) -> None:
        result = validate_calibration(calibration_rows())
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["completeness_status"], "passed")
        self.assertEqual(result["research_signal_status"], "passed")
        self.assertEqual(result["row_count"], 32)
        self.assertEqual(result["positive_variance_group_count"], 6)

        rows = calibration_rows()
        for row in rows:
            row["reward"] = 0.0
            row["record"]["normalized_reward"] = 0.0
        failed = validate_calibration(rows)
        self.assertEqual(failed["completeness_status"], "passed")
        self.assertEqual(failed["research_signal_status"], "failed")
        self.assertEqual(failed["status"], "failed")
        self.assertIn("at_least_6_positive_variance_groups", failed["failed_research_checks"])
        self.assertIn("at_least_2_distinct_rewards", failed["failed_research_checks"])

    def test_calibration_requires_exact_group_shape_and_complete_trace(self) -> None:
        rows = calibration_rows()
        rows.pop()
        rows[0]["record"]["label_scores"].pop("E")
        rows[1]["record"]["masked"] = True
        result = validate_calibration(rows)
        self.assertEqual(result["completeness_status"], "failed")
        self.assertIn("exact_8_groups_x_4_candidates", result["failed_completeness_checks"])
        self.assertIn("zero_infrastructure_masks", result["failed_completeness_checks"])
        self.assertIn("trace_audit_complete", result["failed_completeness_checks"])


if __name__ == "__main__":
    unittest.main()
