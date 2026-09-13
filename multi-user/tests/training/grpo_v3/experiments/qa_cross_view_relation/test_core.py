from __future__ import annotations

import json
import unittest

from training.grpo_v3.experiments.qa_cross_view_relation.anchors import load_anchor_set
from training.grpo_v3.experiments.qa_cross_view_relation.deterministic import assess_completion
from training.grpo_v3.experiments.qa_cross_view_relation.domain import (
    CandidateSemanticScore,
    JudgeCandidate,
    GroupJudgeResult,
    TEXT_CHECK_NAMES,
)
from training.grpo_v3.experiments.qa_cross_view_relation.judge import (
    _build_schema_repair_prompt,
    NonThinkingTextJudgeRunner,
    judge_candidate_group,
)
from training.grpo_v3.experiments.qa_cross_view_relation.prompt import build_group_judge_prompt
from training.grpo_v3.experiments.qa_cross_view_relation.reward import compute_group_rewards


def qa(**overrides):
    value = {
        "qa_id": "qa-1",
        "question_type": "neutral",
        "question": "After I left the counter, where did the mug end up?",
        "options": ["counter", "sink", "laptop table", "shelf", "being carried"],
        "correct": "C",
        "answer": "laptop table",
        "required_users": ["Jake", "Katrina"],
        "per_user_evidence_claims": [{"user": "Katrina", "claim": "The mug was near the laptop."}],
    }
    value.update(overrides)
    return value


def passing_checks():
    return {
        name: {"status": "PASS", "reason": "The candidate passes this text-only check."}
        for name in TEXT_CHECK_NAMES
    }


def judge_score(candidate_id="c0", **overrides):
    value = {
        "candidate_id": candidate_id,
        "cross_view_relation_score": 2,
        "semantic_naturalness_score": 2,
        "internal_consistency_score": 2,
        "anchor_tier": 2,
        "pairwise_preferences": {},
        "checks": passing_checks(),
        "reasons": {"summary": "strong"},
    }
    value.update(overrides)
    return value


class CrossViewRelationCoreTests(unittest.TestCase):
    def test_text_judge_explicitly_disables_qwen3_thinking_mode(self):
        runner = NonThinkingTextJudgeRunner(model_id="Qwen3-32B")
        self.assertEqual(
            runner._extra_request_payload(),
            {"chat_template_kwargs": {"enable_thinking": False}},
        )

    def test_domain_rejects_invalid_scores_and_missing_candidate_ids(self):
        valid = CandidateSemanticScore.from_mapping(
            {
                "candidate_id": "c0",
                "cross_view_relation_score": 2,
                "semantic_naturalness_score": 1,
                "internal_consistency_score": 0,
                "anchor_tier": 2,
                "pairwise_preferences": {"c1": "WIN"},
                "reasons": {"summary": "ok"},
            }
        )
        self.assertEqual(valid.candidate_id, "c0")
        with self.assertRaises(ValueError):
            CandidateSemanticScore.from_mapping({**valid.to_dict(), "semantic_naturalness_score": 3})
        with self.assertRaises(ValueError):
            GroupJudgeResult.from_mapping({"candidate_scores": [valid.to_dict()]}, ["c0", "c1"])
        with self.assertRaises(ValueError):
            GroupJudgeResult.from_mapping({"candidate_scores": [{**valid.to_dict(), "candidate_id": "strong_anchor"}]}, ["strong_anchor"])
        with self.assertRaisesRegex(ValueError, "candidate score must be an object"):
            GroupJudgeResult.from_mapping({"candidate_scores": ["not-an-object"]}, ["c0"])
        c0 = judge_score("c0", pairwise_preferences={})
        c1 = judge_score("c1", pairwise_preferences={"c0": "LOSS"})
        with self.assertRaisesRegex(ValueError, "pairwise preference keys"):
            GroupJudgeResult.from_mapping(
                {"candidate_scores": [c0, c1]},
                ["c0", "c1"],
                require_text_checks=True,
            )

    def test_deterministic_checks_repair_json_but_blocks_semantic_invalids(self):
        raw = json.dumps(qa())
        repaired = "```json\n" + raw + "\n```"
        self.assertTrue(assess_completion(raw, required_users=["Jake", "Katrina"]).eligible_for_semantic_judge)
        self.assertEqual(
            assess_completion(repaired, required_users=["Jake", "Katrina"]).format_status,
            "repaired",
        )
        duplicate_options = assess_completion(
            json.dumps(qa(options=["same", "same", "c", "d", "e"])),
            required_users=["Jake", "Katrina"],
        )
        self.assertIn("options_must_be_unique", duplicate_options.blocking_errors)
        bad_answer = assess_completion(
            json.dumps(qa(answer="not the selected option")),
            required_users=["Jake", "Katrina"],
        )
        self.assertIn("answer_must_equal_options_correct", bad_answer.blocking_errors)
        leaked_name = assess_completion(
            json.dumps(qa(question="After I left, where did Katrina put the mug?")),
            required_users=["Jake", "Katrina"],
        )
        self.assertIn("question_mentions_required_user", leaked_name.blocking_errors)
        dataset_language = assess_completion(
            json.dumps(qa(question="In camera clip frame 12, where is the mug?")),
            required_users=["Jake", "Katrina"],
        )
        self.assertIn("question_uses_dataset_language", dataset_language.blocking_errors)

    def test_anchors_are_frozen_and_auditable(self):
        anchors = load_anchor_set()
        self.assertEqual(anchors.strong.anchor_id, "strong_cross_view_followup_v1")
        self.assertEqual(anchors.weak.anchor_id, "weak_other_person_activity_v1")
        self.assertEqual(len(anchors.sha256), 64)

    def test_group_judge_prompt_requests_strict_naturalness_and_consistency_audit(self):
        candidate = JudgeCandidate(candidate_id="c0", raw_completion=json.dumps(qa()), qa=qa())
        prompt, _order = build_group_judge_prompt(
            candidates=[candidate],
            anchors=load_anchor_set(),
            order_seed="fixture",
        )

        self.assertIn("subject-verb", prompt)
        self.assertIn("answer exactly matches", prompt)
        self.assertIn("user names", prompt)
        self.assertIn("timestamps", prompt)
        self.assertIn("dataset language", prompt)

    def test_v3_judge_contract_requires_all_text_checks_and_scopes_out_video_truth(self):
        prompt, _order = build_group_judge_prompt(
            candidates=[JudgeCandidate(candidate_id="c0", raw_completion=json.dumps(qa()), qa=qa())],
            anchors=load_anchor_set(),
            order_seed="fixture",
            require_text_checks=True,
        )
        self.assertIn("Do not verify video truth, groundedness, or actual answerability.", prompt)
        self.assertIn("Run every absolute text check before assigning scores or pairwise preferences.", prompt)
        for name in TEXT_CHECK_NAMES:
            self.assertIn(name, prompt)
        repair_prompt = _build_schema_repair_prompt(
            original_prompt=prompt,
            raw_output="{}",
            expected_candidate_ids=["c0"],
            error=ValueError("missing text checks"),
            require_text_checks=True,
        )
        self.assertIn("question_answer_type_match", repair_prompt)
        self.assertIn("pairwise_preferences keys", repair_prompt)

        missing = judge_score()
        missing["checks"] = {}
        with self.assertRaisesRegex(ValueError, "missing text checks"):
            GroupJudgeResult.from_mapping(
                {"candidate_scores": [missing]},
                ["c0"],
                require_text_checks=True,
            )
        malformed = judge_score()
        malformed["checks"]["premise_relevance"] = "PASS"
        with self.assertRaisesRegex(ValueError, "premise_relevance must be an object"):
            GroupJudgeResult.from_mapping(
                {"candidate_scores": [malformed]},
                ["c0"],
                require_text_checks=True,
            )

    def test_reward_formula_for_semantic_judged_candidates(self):
        valid0 = assess_completion(json.dumps(qa()), required_users=["Jake", "Katrina"])
        valid1 = assess_completion(
            json.dumps(qa(question="While I was washing dishes, what was the other person doing?")),
            required_users=["Jake", "Katrina"],
        )
        judge = GroupJudgeResult.from_mapping(
            {
                "candidate_scores": [
                    {
                        "candidate_id": "c0",
                        "cross_view_relation_score": 2,
                        "semantic_naturalness_score": 2,
                        "internal_consistency_score": 2,
                        "anchor_tier": 2,
                        "pairwise_preferences": {"c1": "WIN"},
                        "reasons": {"summary": "strong"},
                    },
                    {
                        "candidate_id": "c1",
                        "cross_view_relation_score": 0,
                        "semantic_naturalness_score": 2,
                        "internal_consistency_score": 2,
                        "anchor_tier": 0,
                        "pairwise_preferences": {"c0": "LOSS"},
                        "reasons": {"summary": "shallow activity"},
                    },
                ]
            },
            ["c0", "c1"],
        )
        rewards = compute_group_rewards(
            candidate_ids=["c0", "c1"],
            deterministic_results={"c0": valid0, "c1": valid1},
            judge_result=judge,
        )
        self.assertAlmostEqual(rewards["c0"].reward_total, 1.0)
        self.assertAlmostEqual(rewards["c1"].semantic_quality, 0.4)
        self.assertTrue(all(0.0 <= item.reward_total <= 1.0 for item in rewards.values()))

    def test_v3_blocking_text_checks_cap_reward_without_changing_v2_formula(self):
        assessment = assess_completion(json.dumps(qa()), required_users=["Jake", "Katrina"])
        blocked = judge_score()
        blocked["checks"]["question_answer_type_match"] = {
            "status": "FAIL",
            "reason": "The question asks for a person but the answer is an activity.",
        }
        judge = GroupJudgeResult.from_mapping(
            {"candidate_scores": [blocked]},
            ["c0"],
            require_text_checks=True,
        )

        v2 = compute_group_rewards(
            candidate_ids=["c0"],
            deterministic_results={"c0": assessment},
            judge_result=judge,
        )["c0"]
        v3 = compute_group_rewards(
            candidate_ids=["c0"],
            deterministic_results={"c0": assessment},
            judge_result=judge,
            apply_text_caps=True,
            reward_revision="qa_cross_view_relation_v3",
        )["c0"]

        self.assertAlmostEqual(v2.reward_total, 0.925)
        self.assertEqual(v2.reward_cap, 1.0)
        self.assertEqual(v3.reward_total, 0.40)
        self.assertAlmostEqual(v3.semantic_quality, 0.8)
        self.assertEqual(v3.anchor_score, 0.5)
        self.assertAlmostEqual(v3.reward_before_cap, 0.68)
        self.assertEqual(v3.reward_cap, 0.40)
        self.assertIn("question_answer_type_match", v3.cap_reasons)

    def test_v3_shallow_and_naturalness_failures_use_declared_caps(self):
        assessment = assess_completion(json.dumps(qa()), required_users=["Jake", "Katrina"])
        for check_name, expected_cap in (
            ("shallow_activity_relation", 0.40),
            ("natural_first_person_wording", 0.55),
        ):
            with self.subTest(check_name=check_name):
                value = judge_score()
                value["checks"][check_name] = {
                    "status": "FAIL",
                    "reason": "Deliberate regression fixture.",
                }
                judge = GroupJudgeResult.from_mapping(
                    {"candidate_scores": [value]},
                    ["c0"],
                    require_text_checks=True,
                )
                result = compute_group_rewards(
                    candidate_ids=["c0"],
                    deterministic_results={"c0": assessment},
                    judge_result=judge,
                    apply_text_caps=True,
                    reward_revision="qa_cross_view_relation_v3",
                )["c0"]
                self.assertEqual(result.reward_cap, expected_cap)
                self.assertLessEqual(result.reward_total, expected_cap)

    def test_unrecoverable_json_skips_whole_reward_group(self):
        valid0 = assess_completion(json.dumps(qa()), required_users=["Jake", "Katrina"])
        valid1 = assess_completion(
            json.dumps(qa(question="After I left the counter, where did the plate end up?")),
            required_users=["Jake", "Katrina"],
        )
        invalid = assess_completion("{not-json", required_users=["Jake", "Katrina"])
        judge = GroupJudgeResult.from_mapping(
            {
                "candidate_scores": [
                    {
                        "candidate_id": "c0",
                        "cross_view_relation_score": 2,
                        "semantic_naturalness_score": 2,
                        "internal_consistency_score": 2,
                        "anchor_tier": 2,
                        "pairwise_preferences": {"c1": "WIN"},
                        "reasons": {"summary": "strong"},
                    },
                    {
                        "candidate_id": "c1",
                        "cross_view_relation_score": 2,
                        "semantic_naturalness_score": 2,
                        "internal_consistency_score": 2,
                        "anchor_tier": 2,
                        "pairwise_preferences": {"c0": "LOSS"},
                        "reasons": {"summary": "also valid"},
                    },
                ]
            },
            ["c0", "c1"],
        )

        rewards = compute_group_rewards(
            candidate_ids=["c0", "c1", "c2"],
            deterministic_results={"c0": valid0, "c1": valid1, "c2": invalid},
            judge_result=judge,
        )

        self.assertTrue(all(item.reward_total == 0.0 for item in rewards.values()))
        self.assertTrue(all(item.reward_source == "group_skipped_unrecoverable_json" for item in rewards.values()))
        v3_rewards = compute_group_rewards(
            candidate_ids=["c0", "c1", "c2"],
            deterministic_results={"c0": valid0, "c1": valid1, "c2": invalid},
            judge_result=judge,
            apply_text_caps=True,
            reward_revision="qa_cross_view_relation_v3",
        )
        self.assertTrue(
            all(item.reward_revision == "qa_cross_view_relation_v3" for item in v3_rewards.values())
        )

    def test_deterministic_blocking_errors_receive_low_fixed_penalty(self):
        valid = assess_completion(json.dumps(qa()), required_users=["Jake", "Katrina"])
        invalid = assess_completion(
            json.dumps(qa(answer="C")),
            required_users=["Jake", "Katrina"],
        )
        self.assertIn("answer_must_equal_options_correct", invalid.blocking_errors)
        judge = GroupJudgeResult.from_mapping(
            {
                "candidate_scores": [
                    {
                        "candidate_id": "c0",
                        "cross_view_relation_score": 2,
                        "semantic_naturalness_score": 2,
                        "internal_consistency_score": 2,
                        "anchor_tier": 2,
                        "pairwise_preferences": {},
                        "reasons": {"summary": "strong"},
                    }
                ]
            },
            ["c0"],
        )

        rewards = compute_group_rewards(
            candidate_ids=["c0", "c1"],
            deterministic_results={"c0": valid, "c1": invalid},
            judge_result=judge,
        )

        self.assertAlmostEqual(rewards["c0"].reward_total, 0.925)
        self.assertLessEqual(rewards["c1"].reward_total, 0.2)
        self.assertEqual(rewards["c1"].reward_source, "deterministic_blocking_penalty")

    def test_judge_order_instability_is_traced_and_stabilized(self):
        assessment = assess_completion(json.dumps(qa()), required_users=["Jake", "Katrina"])
        candidate = JudgeCandidate(candidate_id="c0", raw_completion=json.dumps(qa()), qa=qa())

        outputs = [
            {
                "candidate_scores": [
                    {
                        "candidate_id": "c0",
                        "cross_view_relation_score": 2,
                        "semantic_naturalness_score": 2,
                        "internal_consistency_score": 2,
                        "anchor_tier": 2,
                        "pairwise_preferences": {},
                        "reasons": {"summary": "first"},
                    }
                ]
            },
            {
                "candidate_scores": [
                    {
                        "candidate_id": "c0",
                        "cross_view_relation_score": 1,
                        "semantic_naturalness_score": 2,
                        "internal_consistency_score": 2,
                        "anchor_tier": 1,
                        "pairwise_preferences": {},
                        "reasons": {"summary": "second"},
                    }
                ]
            },
        ]

        def runner(_prompt):
            return json.dumps(outputs.pop(0))

        result = judge_candidate_group(
            candidates=[candidate],
            deterministic_results={"c0": assessment},
            runner=runner,
            order_seed="fixture",
        )

        self.assertTrue(result.order_instability)
        self.assertEqual(len(result.raw_outputs), 2)
        self.assertEqual(result.candidate_scores["c0"].cross_view_relation_score, 1)

    def test_reason_wording_difference_alone_is_not_order_instability(self):
        assessment = assess_completion(json.dumps(qa()), required_users=["Jake", "Katrina"])
        candidate = JudgeCandidate(candidate_id="c0", raw_completion=json.dumps(qa()), qa=qa())
        first = judge_score("c0")
        second = judge_score("c0")
        second["checks"]["premise_relevance"]["reason"] = "Same PASS decision, different wording."
        outputs = [{"candidate_scores": [first]}, {"candidate_scores": [second]}]

        result = judge_candidate_group(
            candidates=[candidate],
            deterministic_results={"c0": assessment},
            runner=lambda _prompt: json.dumps(outputs.pop(0)),
            order_seed="fixture",
            require_text_checks=True,
        )

        self.assertFalse(result.order_instability)

    def test_judge_repairs_one_schema_invalid_response(self):
        assessment = assess_completion(json.dumps(qa()), required_users=["Jake", "Katrina"])
        candidate = JudgeCandidate(candidate_id="c0", raw_completion=json.dumps(qa()), qa=qa())

        invalid_missing_candidate_id = {
            "candidate_scores": [
                {
                    "cross_view_relation_score": 2,
                    "semantic_naturalness_score": 2,
                    "internal_consistency_score": 2,
                    "anchor_tier": 2,
                    "pairwise_preferences": {},
                    "reasons": {"summary": "missing id"},
                }
            ]
        }
        repaired = {
            "candidate_scores": [
                {
                    "candidate_id": "c0",
                    "cross_view_relation_score": 2,
                    "semantic_naturalness_score": 2,
                    "internal_consistency_score": 2,
                    "anchor_tier": 2,
                    "pairwise_preferences": {},
                    "reasons": {"summary": "repaired"},
                }
            ]
        }
        reverse = {
            "candidate_scores": [
                {
                    "candidate_id": "c0",
                    "cross_view_relation_score": 2,
                    "semantic_naturalness_score": 2,
                    "internal_consistency_score": 2,
                    "anchor_tier": 2,
                    "pairwise_preferences": {},
                    "reasons": {"summary": "reverse"},
                }
            ]
        }
        outputs = [invalid_missing_candidate_id, repaired, reverse]
        prompts = []

        def runner(prompt):
            prompts.append(prompt)
            return json.dumps(outputs.pop(0))

        result = judge_candidate_group(
            candidates=[candidate],
            deterministic_results={"c0": assessment},
            runner=runner,
            order_seed="fixture",
        )

        self.assertEqual(result.candidate_scores["c0"].candidate_id, "c0")
        self.assertEqual(len(prompts), 3)
        self.assertIn("candidate_id", prompts[1])
        self.assertIn("c0", prompts[1])
        self.assertEqual(result.raw_outputs[0]["raw_output"], json.dumps(invalid_missing_candidate_id))
        self.assertEqual(result.raw_outputs[0]["repair_parsed_output"], repaired)


if __name__ == "__main__":
    unittest.main()
