import json
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if "egolife_two_user_qa" not in sys.modules:
    package = types.ModuleType("egolife_two_user_qa")
    package.__path__ = [str(ROOT)]
    sys.modules["egolife_two_user_qa"] = package

from egolife_two_user_qa.prompts import (
    ANSWERABILITY_SUFFICIENCY_SCHEMA,
    build_answerability_prompt,
    build_evidence_groundedness_judge_prompt,
    build_qa_formality_judge_prompt,
    judge_schema_for_check,
)
from egolife_two_user_qa.video_qa_loop import (
    answerability_sufficiency_output_errors,
    merge_parallel_judges,
    parse_single_judge_output,
)


def qa_item() -> dict:
    return {
        "qa_id": "qa-review",
        "question_type": "neutral",
        "question": "Where was the mug placed after I handed it over?",
        "options": ["On the desk", "By the sink", "Near the door", "On the sofa", "In a bag"],
        "correct": "B",
        "answer": "By the sink",
        "required_users": [
            "Speaker",
            "ProviderOne",
            "ProviderTwo",
            "ProviderThree",
            "ProviderFour",
            "ProviderFive",
        ],
    }


def packet() -> dict:
    return {
        "evidence_id": "evidence-review",
        "required_users": qa_item()["required_users"],
        "clips": [{"user": user} for user in qa_item()["required_users"]],
    }


class JudgeOutputContractTests(unittest.TestCase):
    def test_final_judges_share_flat_first_verdict_contract(self) -> None:
        expected = {
            "verdict": "pass/fail",
            "reason": (
                "specific instance-level failure reason when verdict is fail; "
                "null when verdict is pass"
            ),
            "fix": (
                "specific repair for that failure when verdict is fail; "
                "null when verdict is pass"
            ),
        }
        self.assertEqual(judge_schema_for_check("qa_formality"), expected)
        self.assertEqual(judge_schema_for_check("evidence_groundedness"), expected)
        self.assertEqual(list(expected), ["verdict", "reason", "fix"])

    def test_formality_prompt_keeps_all_criteria_without_output_subchecks(self) -> None:
        prompt = build_qa_formality_judge_prompt(qa_item(), packet(), schema_errors=[])
        for criterion in (
            "first_person_perspective",
            "naturalness_and_clarity",
            "other_person_activity_query",
            "direct_name_leakage",
            "timestamp_citation",
            "ambiguous_reference",
        ):
            self.assertIn(criterion, prompt)
        output_contract = prompt.rsplit(
            "Return exactly one valid JSON object with this exact shape:", 1
        )[1]
        self.assertEqual(json.loads(output_contract), judge_schema_for_check("qa_formality"))
        self.assertNotIn("semantic_subchecks", output_contract)
        self.assertNotIn("review_passed", output_contract)
        self.assertNotIn("feedback_to_generator", output_contract)

    def test_grounding_prompt_uses_same_first_verdict_contract(self) -> None:
        prompt = build_evidence_groundedness_judge_prompt(qa_item(), packet())
        output_contract = prompt.rsplit(
            "Return exactly one valid JSON object with this exact shape:", 1
        )[1]
        self.assertEqual(
            json.loads(output_contract),
            judge_schema_for_check("evidence_groundedness"),
        )

    def test_live_parser_enforces_order_and_conditional_reason_fix(self) -> None:
        self.assertEqual(
            parse_single_judge_output(
                '{"verdict":"pass","reason":null,"fix":null}',
                "qa_formality",
            )["verdict"],
            "pass",
        )
        self.assertEqual(
            parse_single_judge_output(
                '{"verdict":"fail","reason":"The question names a participant.",'
                '"fix":"Replace the name with a local descriptive reference."}',
                "qa_formality",
            )["verdict"],
            "fail",
        )
        with self.assertRaisesRegex(ValueError, "exactly verdict, reason, fix"):
            parse_single_judge_output(
                '{"reason":null,"verdict":"pass","fix":null}',
                "qa_formality",
            )
        with self.assertRaisesRegex(ValueError, "requires reason to be null"):
            parse_single_judge_output(
                '{"verdict":"pass","reason":"Looks good.","fix":null}',
                "qa_formality",
            )
        with self.assertRaisesRegex(ValueError, "generic placeholder"):
            parse_single_judge_output(
                '{"verdict":"fail","reason":"The decisive reason this candidate '
                'fails this judge.","fix":"Revise the candidate."}',
                "qa_formality",
            )

    def test_merger_computes_internal_status_and_generator_feedback(self) -> None:
        answerability = {
            "evaluations": [],
            "gate": {
                "passed": True,
                "reason": "The asker-only condition is insufficient and all-six is sufficient.",
            },
        }
        merged = merge_parallel_judges(
            qa_formality_judge={"verdict": "pass", "reason": None, "fix": None},
            evidence_groundedness_judge={
                "verdict": "fail",
                "reason": "The claimed placement is not visible.",
                "fix": "Use an answer supported by a visible placement.",
            },
            answerability=answerability,
            schema_errors=[],
            qa_item=qa_item(),
            participant_names=qa_item()["required_users"],
        )
        self.assertEqual(merged["checks"]["qa_formality"]["status"], "PASS")
        self.assertEqual(merged["checks"]["evidence_groundedness"]["status"], "FAIL")
        self.assertFalse(merged["review_passed"])
        self.assertEqual(merged["blocking_failures"], ["evidence_groundedness"])
        self.assertIn("The claimed placement is not visible.", merged["feedback_to_generator"])
        self.assertNotIn("why_generator_asked_this", merged)

    def test_legacy_zero_shot_answerability_contract_is_preserved(self) -> None:
        self.assertEqual(
            ANSWERABILITY_SUFFICIENCY_SCHEMA["required"],
            ["answerable", "reason", "available_evidence", "missing_evidence"],
        )
        self.assertEqual(
            list(ANSWERABILITY_SUFFICIENCY_SCHEMA["properties"]),
            ["answerable", "reason", "available_evidence", "missing_evidence"],
        )
        prompt = build_answerability_prompt(
            qa_item(),
            {
                "condition_id": "speaker_only::Speaker",
                "condition_type": "speaker_only",
                "users": ["Speaker"],
            },
        )
        self.assertIn("The first JSON field must be `answerable`", prompt)
        value = {
            "answerable": False,
            "reason": "The recipient's later action is not visible in the speaker video.",
            "available_evidence": ["The initial handoff is visible."],
            "missing_evidence": ["The recipient's later placement is not visible."],
        }
        self.assertEqual(answerability_sufficiency_output_errors(value), [])


if __name__ == "__main__":
    unittest.main()
