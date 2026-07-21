import json
import math
import subprocess
import sys
import unittest

from training.grpo_v3_answer_margin import (
    ANSWER_MARGIN_REWARD_REVISION,
    LABELS,
    MARGIN_CLIP,
    PermutationKey,
    compute_answer_margin,
    extract_core_qa,
    permute_options,
)


class ExtractCoreQATests(unittest.TestCase):
    def setUp(self):
        self.qa = {
            "question": "Who opened the door?",
            "options": ["Ava", "Bo", "Cy", "Di", "Em"],
            "correct": "C",
        }

    def test_extracts_raw_valid_object(self):
        raw = json.dumps({**self.qa, "correct": "c"})

        result = extract_core_qa(raw)

        self.assertTrue(result.ok)
        self.assertEqual(result.status, "raw_valid")
        self.assertEqual(result.as_qa(), {
            "question": "Who opened the door?",
            "options": ["Ava", "Bo", "Cy", "Di", "Em"],
            "correct": "C",
        })

    def test_strips_core_fields_without_changing_internal_characters(self):
        raw = json.dumps({
            "question": "  Who  opened the door?  ",
            "options": ["  Ava ", " Bo  Junior ", "\tCy\n", " Di", "Em  "],
            "correct": " c ",
        })

        result = extract_core_qa(raw)

        self.assertTrue(result.ok)
        self.assertEqual(result.as_qa(), {
            "question": "Who  opened the door?",
            "options": ["Ava", "Bo  Junior", "Cy", "Di", "Em"],
            "correct": "C",
        })

    def test_extracts_complete_fence_and_extra_text(self):
        encoded = json.dumps(self.qa)
        fenced = extract_core_qa(f"```json\n{encoded}\n```")
        embedded = extract_core_qa(f"preface {{not JSON}} then {encoded} epilogue")

        self.assertTrue(fenced.ok)
        self.assertEqual(fenced.status, "repaired")
        self.assertFalse(embedded.ok, "the first complete object must be selected")

        embedded = extract_core_qa(f"preface only: {encoded} epilogue")
        self.assertTrue(embedded.ok)
        self.assertEqual(embedded.as_qa(), self.qa)

    def test_reuses_conservative_missing_and_trailing_comma_repairs(self):
        missing_comma = (
            '{"question":"Q","options":["1","2","3","4","5"] '
            '"correct":"A"}'
        )
        trailing_comma = (
            '{"question":"Q","options":["1","2","3","4","5"],'
            '"correct":"A",}'
        )

        for raw in (missing_comma, trailing_comma):
            with self.subTest(raw=raw):
                result = extract_core_qa(raw)
                self.assertTrue(result.ok)
                self.assertEqual(result.status, "repaired")
                self.assertTrue(result.format_validation.repair_operations)

    def test_scanner_ignores_braces_and_escaped_quotes_inside_strings(self):
        qa = {
            **self.qa,
            "question": 'What does "{escaped}" mean, and what about \\?',
        }
        result = extract_core_qa("prefix " + json.dumps(qa) + " suffix")

        self.assertTrue(result.ok)
        self.assertEqual(result.question, qa["question"])

    def test_scanner_ignores_braces_inside_quoted_prefix_text(self):
        raw = (
            'prefix "{not an object}" then '
            + json.dumps(self.qa)
            + " suffix"
        )

        result = extract_core_qa(raw)

        self.assertTrue(result.ok)
        self.assertEqual(result.as_qa(), self.qa)

    def test_prefix_string_escapes_do_not_break_object_scanning(self):
        prefix_string = json.dumps('escaped quote: " brace { and slash \\ }')
        raw = "prefix " + prefix_string + " then " + json.dumps(self.qa) + " suffix"

        result = extract_core_qa(raw)

        self.assertTrue(result.ok)
        self.assertEqual(result.as_qa(), self.qa)

    def test_rejects_unclosed_string_or_object(self):
        unclosed_string = '{"question":"unterminated }'
        unclosed_object = json.dumps(self.qa)[:-1]

        for raw in (unclosed_string, unclosed_object):
            with self.subTest(raw=raw):
                result = extract_core_qa(raw)
                self.assertFalse(result.ok)
                self.assertEqual(result.status, "unrecoverable")

    def test_selects_first_complete_object(self):
        first = {**self.qa, "question": "first"}
        second = {**self.qa, "question": "second"}
        result = extract_core_qa(json.dumps(first) + "\n" + json.dumps(second))

        self.assertTrue(result.ok)
        self.assertEqual(result.question, "first")

    def test_rejects_missing_empty_and_invalid_fields(self):
        invalid_cases = [
            ({"options": self.qa["options"], "correct": "A"}, "invalid_question"),
            ({**self.qa, "question": "  "}, "invalid_question"),
            ({**self.qa, "options": ["1", "2", "", "4", "5"]}, "invalid_options"),
            ({**self.qa, "correct": "AA"}, "invalid_correct"),
            ({**self.qa, "correct": "F"}, "invalid_correct"),
            ({**self.qa, "correct": 0}, "invalid_correct"),
        ]
        for value, reason in invalid_cases:
            with self.subTest(value=value):
                result = extract_core_qa(json.dumps(value))
                self.assertFalse(result.ok)
                self.assertEqual(result.failure_reason, reason)
                self.assertIsNone(result.as_qa())

    def test_requires_exactly_five_options(self):
        for count in (4, 6):
            with self.subTest(count=count):
                result = extract_core_qa(json.dumps({
                    **self.qa,
                    "options": [str(index) for index in range(count)],
                }))
                self.assertFalse(result.ok)
                self.assertEqual(result.failure_reason, "invalid_options")

    def test_does_not_guess_correct_from_answer_or_option_text(self):
        value = {
            "question": "Q",
            "options": ["A", "B", "C is correct", "D", "E"],
            "answer": "C",
            "explanation": "The correct answer is C.",
        }
        result = extract_core_qa(json.dumps(value))

        self.assertFalse(result.ok)
        self.assertEqual(result.failure_reason, "invalid_correct")


class PermuteOptionsTests(unittest.TestCase):
    def setUp(self):
        self.key = PermutationKey(
            experiment_condition_id="condition-7",
            phase="train",
            evidence_id="clip-19",
            generation_seed_or_call_index=23,
            candidate_index=2,
            reward_revision=ANSWER_MARGIN_REWARD_REVISION,
        )
        self.options = ["zero", "one", "two", "three", "four"]

    def test_permutation_is_bijection_with_correct_inverse_and_mapping(self):
        result = permute_options(self.options, "C", self.key)

        self.assertEqual(sorted(result.permutation), list(range(5)))
        self.assertEqual(sorted(result.inverse), list(range(5)))
        for new_index, old_index in enumerate(result.permutation):
            self.assertEqual(result.inverse[old_index], new_index)
            self.assertEqual(result.permuted_options[new_index], self.options[old_index])
        self.assertEqual(
            result.permuted_options[LABELS.index(result.mapped_correct)],
            self.options[2],
        )
        self.assertEqual(len(result.digests), 5)
        self.assertTrue(all(len(digest) == 64 for digest in result.digests))

    def test_matches_sha256_digest_then_index_contract(self):
        import hashlib

        expected_digests = [
            hashlib.sha256(
                self.key.stable_text().encode("utf-8") + b"\0" + str(index).encode("ascii")
            ).hexdigest()
            for index in range(5)
        ]
        expected_permutation = sorted(range(5), key=lambda index: (expected_digests[index], index))

        result = permute_options(self.options, "A", self.key)

        self.assertEqual(result.digests, expected_digests)
        self.assertEqual(result.permutation, expected_permutation)

    def test_same_key_is_stable_across_python_processes(self):
        script = (
            "import json; "
            "from training.grpo_v3_answer_margin import PermutationKey, permute_options; "
            "k=PermutationKey(experiment_condition_id='condition-7',phase='train',"
            "evidence_id='clip-19',generation_seed_or_call_index=23,candidate_index=2,"
            "reward_revision='combined_video_answer_margin_v1'); "
            "r=permute_options(['zero','one','two','three','four'],'C',k); "
            "print(json.dumps([r.permutation,r.inverse,r.mapped_correct,r.digests]))"
        )
        expected = subprocess.check_output(
            [sys.executable, "-c", script], text=True, cwd="."
        ).strip()
        actual = subprocess.check_output(
            [sys.executable, "-c", script], text=True, cwd="."
        ).strip()

        self.assertEqual(actual, expected)

    def test_reward_revision_changes_stable_key_and_digests(self):
        changed_revision = PermutationKey(
            experiment_condition_id=self.key.experiment_condition_id,
            phase=self.key.phase,
            evidence_id=self.key.evidence_id,
            generation_seed_or_call_index=self.key.generation_seed_or_call_index,
            candidate_index=self.key.candidate_index,
            reward_revision="combined_video_answer_margin_v2",
        )

        original = permute_options(self.options, "C", self.key)
        changed = permute_options(self.options, "C", changed_revision)

        self.assertNotEqual(self.key.stable_text(), changed_revision.stable_text())
        self.assertNotEqual(original.digests, changed.digests)

    def test_stable_text_uses_formal_experiment_condition_audit_key(self):
        stable_payload = json.loads(self.key.stable_text())

        self.assertEqual(stable_payload["experiment_condition_id"], "condition-7")
        self.assertNotIn("condition_id", stable_payload)
        self.assertEqual(set(stable_payload), {
            "experiment_condition_id",
            "phase",
            "evidence_id",
            "generation_seed_or_call_index",
            "candidate_index",
            "reward_revision",
        })

    def test_rejects_invalid_options_and_correct(self):
        with self.assertRaises(ValueError):
            permute_options(self.options[:4], "A", self.key)
        with self.assertRaises(ValueError):
            permute_options(self.options, "F", self.key)


class AnswerMarginTests(unittest.TestCase):
    def test_reward_revision_is_auditable(self):
        self.assertEqual(ANSWER_MARGIN_REWARD_REVISION, "combined_video_answer_margin_v1")

    def test_computes_positive_margin_reward_and_local_log_probabilities(self):
        scores = {"A": 3.0, "B": 1.0, "C": 0.0, "D": -1.0, "E": -2.0}
        result = compute_answer_margin(scores, "A")
        log_z = 3.0 + math.log(sum(math.exp(score - 3.0) for score in scores.values()))

        self.assertEqual(result.raw_margin, 2.0)
        self.assertEqual(result.clipped_margin, 2.0)
        self.assertEqual(result.reward, 0.25)
        for label, score in scores.items():
            self.assertAlmostEqual(result.log_probabilities[label], score - log_z)
        self.assertEqual(result.unique_top1, "A")
        self.assertFalse(result.tie)

    def test_computes_negative_margin_and_clips_both_sides(self):
        negative = compute_answer_margin(
            {"A": -2.0, "B": 2.0, "C": 1.0, "D": 0.0, "E": -1.0}, "A"
        )
        high = compute_answer_margin(
            {"A": 99.0, "B": 0.0, "C": -1.0, "D": -2.0, "E": -3.0}, "A"
        )
        low = compute_answer_margin(
            {"A": -99.0, "B": 0.0, "C": -1.0, "D": -2.0, "E": -3.0}, "A"
        )

        self.assertEqual(negative.raw_margin, -4.0)
        self.assertEqual(negative.reward, -0.5)
        self.assertEqual((high.clipped_margin, high.reward), (MARGIN_CLIP, 1.0))
        self.assertEqual((low.clipped_margin, low.reward), (-MARGIN_CLIP, -1.0))

    def test_reports_ties_with_one_micro_tolerance(self):
        tied = compute_answer_margin(
            {"A": 1.0, "B": 1.0 + 1e-6, "C": 0.0, "D": -1.0, "E": -2.0}, "A"
        )
        unique = compute_answer_margin(
            {"A": 1.0, "B": 1.0 + 1.1e-6, "C": 0.0, "D": -1.0, "E": -2.0}, "A"
        )

        self.assertTrue(tied.tie)
        self.assertIsNone(tied.unique_top1)
        self.assertFalse(unique.tie)
        self.assertEqual(unique.unique_top1, "B")

    def test_rejects_bad_key_space_correct_and_non_finite_scores(self):
        valid = {label: float(index) for index, label in enumerate(LABELS)}
        invalid_scores = [
            {key: value for key, value in valid.items() if key != "E"},
            {**valid, "F": 0.0},
            {**valid, "A": math.nan},
            {**valid, "A": math.inf},
            {**valid, "A": -math.inf},
        ]
        for scores in invalid_scores:
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                compute_answer_margin(scores, "A")
        for correct in ("a", "F", "AA"):
            with self.subTest(correct=correct), self.assertRaises(ValueError):
                compute_answer_margin(valid, correct)

    def test_rejects_non_finite_margin_created_by_finite_score_subtraction(self):
        scores = {"A": 1e308, "B": -1e308, "C": -1e308, "D": -1e308, "E": -1e308}

        with self.assertRaises(ValueError):
            compute_answer_margin(scores, "A")

    def test_rejects_non_finite_log_probability_derived_from_finite_scores(self):
        scores = {"A": 1e308, "B": 1e308, "C": -1e308, "D": -1e308, "E": -1e308}

        with self.assertRaises(ValueError):
            compute_answer_margin(scores, "A")


if __name__ == "__main__":
    unittest.main()
