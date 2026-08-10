from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import unittest

from training.grpo_v3.experiments.annotated_preference.prompting import (
    COMPACT_GENERATION_PROMPT,
    COMPACT_QA_CONTRACT,
    PROMPT_REVISION,
    build_compact_generation_prompt,
    prompt_sha256,
    serialize_compact_completion,
)
from tests.training.grpo_v3.experiments.annotated_preference.fixtures import candidate


class CompactCompletionSerializationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.record = candidate(
            "evidence-1::candidate-17",
            evidence_id="evidence-1",
            display_order=17,
            qa_formality=1,
            evidence_quality=2,
            answerability=3,
            overall_rank=99,
            question="谁把杯子递给了对方？",
            options=(
                "甲端把杯子递给乙端",
                "乙端把杯子递给甲端",
                "两人同时拿起杯子",
                "没有人移动杯子",
                "无法从交互判断",
            ),
            correct="B",
            answer="乙端把杯子递给甲端",
        )

    def test_serializes_only_ordered_model_features_as_compact_utf8_json(self) -> None:
        completion = serialize_compact_completion(self.record)

        self.assertEqual(
            '{"question":"谁把杯子递给了对方？","options":["甲端把杯子递给乙端","乙端把杯子递给甲端","两人同时拿起杯子","没有人移动杯子","无法从交互判断"],"correct":"B","answer":"乙端把杯子递给甲端"}',
            completion,
        )
        self.assertEqual(
            ["question", "options", "correct", "answer"],
            list(json.loads(completion)),
        )
        self.assertIn("谁把杯子递给了对方？", completion)
        self.assertNotIn("\\u", completion)
        self.assertFalse(completion.startswith((" ", "\n", "\t")))
        self.assertFalse(completion.endswith((" ", "\n", "\t")))
        self.assertNotIn("```", completion)
        for forbidden in (
            self.record.candidate_id,
            self.record.evidence_id,
            "display_order",
            "qa_formality",
            "evidence_quality",
            "answerability",
            "overall_rank",
            "evidence",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, completion)

    def test_noncontent_changes_do_not_affect_completion_but_each_content_change_does(self) -> None:
        original = serialize_compact_completion(self.record)
        noncontent_changed = replace(
            self.record,
            candidate_id="other-evidence::candidate-1",
            evidence_id="other-evidence",
            display_order=1,
            qa_formality=3,
            evidence_quality=3,
            answerability=1,
            overall_rank=1,
        )

        self.assertEqual(original, serialize_compact_completion(noncontent_changed))
        content_changes = {
            "question": replace(self.record, question="谁拿起了杯子？"),
            "options": replace(
                self.record,
                options=(
                    "甲端拿起杯子",
                    "乙端拿起杯子",
                    "两人同时拿起杯子",
                    "没有人拿起杯子",
                    "无法从交互判断",
                ),
            ),
            "correct": replace(self.record, correct="A"),
            "answer": replace(self.record, answer="甲端把杯子递给乙端"),
        }
        for field, changed in content_changes.items():
            with self.subTest(field=field):
                self.assertNotEqual(original, serialize_compact_completion(changed))


class CompactGenerationPromptTests(unittest.TestCase):
    def test_prompt_has_exact_dual_video_and_compact_qa_contract(self) -> None:
        prompt = COMPACT_GENERATION_PROMPT

        self.assertEqual(2, prompt.count("<video>"))
        self.assertTrue(prompt.startswith("<video>\n<video>"))
        self.assertIn("first video is the Speaker", prompt)
        self.assertIn("second video is the Provider", prompt)
        self.assertIn("same interaction", prompt)
        self.assertIn("synchronized complete dual views", prompt)
        self.assertIn("grounded multiple-choice QA", prompt)
        self.assertIn("Return only one JSON object", prompt)
        self.assertIn(
            '{"question":"...","options":["...","...","...","...","..."],"correct":"A","answer":"..."}',
            prompt,
        )
        self.assertIn("exactly five non-empty", prompt)
        self.assertIn("mutually exclusive", prompt)
        self.assertIn("same semantic type", prompt)
        self.assertIn("correct must be exactly one of A, B, C, D, or E", prompt)
        self.assertIn("answer must exactly equal", prompt)
        self.assertIn("must not contain names, timestamps, or meta-language", prompt)
        self.assertNotIn("evidence", prompt.lower())

    def test_build_and_hash_are_constant_and_utf8_stable(self) -> None:
        expected_hash = hashlib.sha256(
            COMPACT_GENERATION_PROMPT.encode("utf-8")
        ).hexdigest()

        self.assertEqual("compact_qa_v1", COMPACT_QA_CONTRACT)
        self.assertEqual("annotated_pareto_compact_qa_v1", PROMPT_REVISION)
        self.assertIs(COMPACT_GENERATION_PROMPT, build_compact_generation_prompt())
        self.assertEqual(expected_hash, prompt_sha256())
        self.assertEqual(prompt_sha256(), prompt_sha256())


if __name__ == "__main__":
    unittest.main()
