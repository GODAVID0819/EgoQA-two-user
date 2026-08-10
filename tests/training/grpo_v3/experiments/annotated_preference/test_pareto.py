from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import unittest

from training.grpo_v3.experiments.human_preference_reviewer.v1.data import CandidateRecord
from training.grpo_v3.experiments.annotated_preference.pareto import (
    build_pareto_pairs,
    compact_fingerprint,
    dominates,
)
from tests.training.grpo_v3.experiments.annotated_preference.fixtures import (
    candidate,
    evidence,
)


class DominatesTests(unittest.TestCase):
    def test_strict_improvement_dominates(self) -> None:
        self.assertTrue(dominates((3, 3, 2), (2, 3, 2)))
        self.assertFalse(dominates((2, 3, 2), (3, 3, 2)))

    def test_equal_and_crossing_vectors_do_not_dominate(self) -> None:
        self.assertFalse(dominates((3, 2, 1), (3, 2, 1)))
        self.assertFalse(dominates((3, 1, 3), (2, 3, 2)))
        self.assertFalse(dominates((2, 3, 2), (3, 1, 3)))

    def test_rejects_invalid_grade_values_including_bools(self) -> None:
        for invalid in (0, 4, None, True):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "1, 2, or 3"):
                    dominates((3, 3, invalid), (3, 3, 3))


class FingerprintTests(unittest.TestCase):
    def test_fingerprint_uses_only_evidence_and_model_features(self) -> None:
        original = candidate("candidate-z", display_order=99)
        changed_noncontent = replace(
            original,
            candidate_id="candidate-a",
            display_order=1,
            qa_formality=1,
            evidence_quality=1,
            answerability=1,
            overall_rank=99,
        )
        same_features_different_keyword_order = CandidateRecord(
            answer="A",
            correct="A",
            options=("A", "B", "C"),
            question="Who performed the action?",
            overall_rank=7,
            qa_formality=2,
            answerability=2,
            evidence_quality=2,
            display_order=1,
            evidence_id="evidence-1",
            candidate_id="candidate-a",
        )

        expected = compact_fingerprint("evidence-1", original)
        self.assertEqual(expected, compact_fingerprint("evidence-1", changed_noncontent))
        self.assertEqual(
            expected,
            compact_fingerprint("evidence-1", same_features_different_keyword_order),
        )
        self.assertNotEqual(expected, compact_fingerprint("evidence-2", original))

    def test_fingerprint_matches_flat_utf8_canonical_json(self) -> None:
        non_ascii = candidate(
            "candidate-zh",
            evidence_id="证据-一",
            question="谁拿起了杯子？",
            options=("甲", "乙", "丙"),
            correct="乙",
            answer="乙正在拿起杯子。",
        )
        payload = {"evidence_id": "证据-一", **non_ascii.model_features()}
        expected = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()

        self.assertEqual(expected, compact_fingerprint("证据-一", non_ascii))


class BuildParetoPairsTests(unittest.TestCase):
    def test_keeps_smallest_candidate_id_for_duplicate_content(self) -> None:
        duplicate_large = candidate(
            "candidate-z", qa_formality=1, evidence_quality=1, answerability=1
        )
        duplicate_small = replace(
            duplicate_large,
            candidate_id="candidate-a",
            display_order=200,
            qa_formality=3,
            evidence_quality=3,
            answerability=3,
            overall_rank=99,
        )
        inferior = candidate(
            "candidate-b", qa_formality=2, evidence_quality=2, answerability=2,
            question="What did the provider do?",
        )

        pairs, audit = build_pareto_pairs(evidence(duplicate_large, inferior, duplicate_small))

        self.assertEqual(1, len(pairs))
        self.assertEqual("candidate-a", pairs[0].chosen.candidate_id)
        self.assertEqual("candidate-b", pairs[0].rejected.candidate_id)
        self.assertEqual(1, audit.duplicate_candidate_count)
        self.assertEqual(1, audit.total_combinations)

    def test_does_not_deduplicate_candidates_across_evidence(self) -> None:
        shared = candidate("candidate-a", evidence_id="evidence-1")
        self.assertNotEqual(
            compact_fingerprint("evidence-1", shared),
            compact_fingerprint("evidence-2", replace(shared, evidence_id="evidence-2")),
        )

    def test_pair_order_is_independent_of_display_and_input_order(self) -> None:
        best = candidate(
            "candidate-c", display_order=30, qa_formality=3, evidence_quality=3,
            answerability=3, question="Best question",
        )
        middle = candidate(
            "candidate-a", display_order=20, qa_formality=2, evidence_quality=2,
            answerability=2, question="Middle question",
        )
        worst = candidate(
            "candidate-b", display_order=10, qa_formality=1, evidence_quality=1,
            answerability=1, question="Worst question",
        )

        first_pairs, _ = build_pareto_pairs(evidence(best, middle, worst))
        second_pairs, _ = build_pareto_pairs(
            evidence(
                replace(worst, display_order=300),
                replace(best, display_order=100),
                replace(middle, display_order=200),
            )
        )

        first_order = [
            (pair.chosen_fingerprint, pair.rejected_fingerprint) for pair in first_pairs
        ]
        second_order = [
            (pair.chosen_fingerprint, pair.rejected_fingerprint) for pair in second_pairs
        ]
        self.assertEqual(first_order, second_order)
        self.assertEqual(first_order, sorted(first_order))

    def test_equal_and_incomparable_vectors_do_not_make_pairs_and_audit_conserves(self) -> None:
        best = candidate(
            "candidate-a", qa_formality=3, evidence_quality=3, answerability=2,
            question="Best question",
        )
        equal = candidate(
            "candidate-b", qa_formality=3, evidence_quality=3, answerability=2,
            question="Equal question",
        )
        incomparable = candidate(
            "candidate-c", qa_formality=3, evidence_quality=1, answerability=3,
            question="Incomparable question",
        )

        pairs, audit = build_pareto_pairs(evidence(best, equal, incomparable))

        self.assertEqual(0, len(pairs))
        self.assertEqual(3, audit.total_combinations)
        self.assertEqual(0, audit.dominance_pair_count)
        self.assertEqual(1, audit.equal_vector_pair_count)
        self.assertEqual(2, audit.incomparable_pair_count)
        self.assertEqual(
            audit.total_combinations,
            audit.dominance_pair_count + audit.equal_vector_pair_count + audit.incomparable_pair_count,
        )

    def test_rejects_candidate_from_another_evidence(self) -> None:
        foreign = candidate("candidate-a", evidence_id="evidence-2")
        with self.assertRaisesRegex(ValueError, "evidence_id"):
            build_pareto_pairs(evidence(foreign, evidence_id="evidence-1"))


if __name__ == "__main__":
    unittest.main()
