import math
from unittest.mock import patch

import egolife_two_user_qa.video_qa_loop as video_qa_loop_module
from egolife_two_user_qa.analyze_entropy_trials import recompute_entropy
from egolife_two_user_qa.prompts import (
    GENERATION_MODES,
    POSITIVE_EXAMPLES_GUIDANCE,
    QA_FORMALITY_SEMANTIC_SUBCHECK_NAMES,
    VIDEO_GENERATION_SCHEMA,
    build_answerability_prompt,
    build_evidence_groundedness_judge_prompt,
    build_qa_formality_judge_prompt,
    build_video_generation_prompt,
    qa_formality_errors,
    quality_quota_prompt,
)
from egolife_two_user_qa.qwen3vl_runner import (
    choice_logprobs_from_top_candidates,
    locate_choice_token,
)
from egolife_two_user_qa.schema import validate_qa_item
from egolife_two_user_qa.video_qa_loop import (
    answerability_gate,
    answerability_uncertainty_from_choice_logits,
    attach_quality_quota_metadata,
    check_from_single_judge,
    decision_uncertainty_from_choice_logits,
    generate_video_qa_loop,
    merge_parallel_judges,
    qa_for_judger_prompt,
    quality_quota_counts_from_rows,
    quality_quota_snapshot,
    run_answerability_eval,
    run_model_judge_branch,
)


def passing_formality_subchecks() -> dict[str, dict[str, str]]:
    return {
        name: {"status": "PASS", "reason": "passed"}
        for name in QA_FORMALITY_SEMANTIC_SUBCHECK_NAMES
    }


def test_locate_status_token_preserves_json_punctuation() -> None:
    pieces = ['{"review_passed":true,"checks":{"evidence_groundedness":{"status":', ' "PASS"', '}}}']

    located = locate_choice_token(pieces)

    assert located is not None
    assert located["token_index"] == 1
    assert located["generated_choice"] == "PASS"
    assert located["alternative_token_pieces"] == {
        "PASS": ' "PASS"',
        "FAIL": ' "FAIL"',
    }


def test_api_top_candidates_extract_both_status_logprobs() -> None:
    pieces = ['{"checks":{"evidence_groundedness":{"status":', ' "PASS"', '}}}']
    candidates = [
        [],
        [
            {"token": ' "PASS"', "logprob": -0.2},
            {"token": ' "FAIL"', "logprob": -2.0},
        ],
        [],
    ]

    signal = choice_logprobs_from_top_candidates(pieces, candidates)

    assert signal["available"] is True
    assert signal["choice_logprobs"] == {"PASS": -0.2, "FAIL": -2.0}


def test_locate_choice_token_rejects_schema_placeholder() -> None:
    assert locate_choice_token(['{"status": "PASS/FAIL"}']) is None


def test_locate_choice_token_uses_first_status_if_text_repeats_it() -> None:
    located = locate_choice_token(
        ['{"status":"PASS","reason":"do not repeat \\"status\\": \\"FAIL\\""}']
    )

    assert located is not None
    assert located["generated_choice"] == "PASS"


def test_entropy_is_normalized_over_only_pass_and_fail() -> None:
    uncertainty = decision_uncertainty_from_choice_logits(
        {
            "available": True,
            "choice_logits": {"PASS": 0.0, "FAIL": 0.0},
            "generated_choice": "PASS",
            "token_index": 12,
        }
    )

    assert uncertainty["available"] is True
    assert math.isclose(uncertainty["entropy_nats"], math.log(2.0), rel_tol=1e-7)
    assert uncertainty["normalized_entropy"] == 1.0
    assert uncertainty["generated_decision"] == "PASS"
    assert uncertainty["choice_set"] == ["PASS", "FAIL"]
    assert "selection_sort_key" not in uncertainty


def test_answerability_entropy_is_normalized_over_a_through_e() -> None:
    uncertainty = answerability_uncertainty_from_choice_logits(
        {
            "available": True,
            "choice_logits": {choice: 0.0 for choice in "ABCDE"},
            "generated_choice": "C",
            "token_index": 4,
        }
    )

    assert uncertainty["available"] is True
    assert math.isclose(uncertainty["entropy_nats"], math.log(5.0), rel_tol=1e-7)
    assert uncertainty["normalized_entropy"] == 1.0
    assert uncertainty["generated_choice"] == "C"
    assert uncertainty["choice_set"] == list("ABCDE")
    assert "selection_sort_key" not in uncertainty

    analysis = recompute_entropy(uncertainty)
    assert analysis["entropy_mode"] == "answerability_choice"
    assert math.isclose(analysis["normalized_entropy_recalculated"], 1.0)
    assert analysis["argmax_choice_recalculated"] == "A"
    assert analysis["selection_sort_key"] is None


def test_emitted_status_controls_gate_without_argmax_override() -> None:
    judge = {
        "review_passed": True,
        "checks": {
            "evidence_groundedness": {
                "status": "PASS",
                "reason": "clear support",
                "fix": "",
                "quality_score": 3,
            }
        },
        "choice_logit_signal": {
            "available": True,
            "choice_logits": {"PASS": -1.0, "FAIL": 2.0},
            "generated_choice": "PASS",
            "token_index": 5,
        },
    }

    check = check_from_single_judge(
        judge,
        "evidence_groundedness",
        include_decision_uncertainty=True,
    )

    assert check["status"] == "PASS"
    assert check["decision_uncertainty"]["argmax_decision"] == "FAIL"
    assert check["decision_uncertainty"]["generated_decision"] == "PASS"
    assert check["status_matches_effective_status"] is True
    assert check["quality_score"] == 3


def test_nested_status_does_not_need_to_be_first_json_field() -> None:
    judge = {
        "review_passed": True,
        "checks": {
            "evidence_groundedness": {
                "status": "PASS",
                "reason": "clear support",
                "fix": "",
            }
        },
        "choice_logit_signal": {
            "available": True,
            "choice_logits": {"PASS": 2.0, "FAIL": -2.0},
            "generated_choice": "PASS",
            "token_index": 8,
        },
    }

    check = check_from_single_judge(
        judge,
        "evidence_groundedness",
        include_decision_uncertainty=True,
    )

    assert check["status"] == "PASS"
    assert check["decision_uncertainty"]["available"] is True
    assert check["decision_uncertainty"]["token_index"] == 8


def test_fail_status_with_high_entropy_keeps_fail_gate() -> None:
    judge = {
        "checks": {
            "evidence_groundedness": {
                "status": "FAIL",
                "reason": "The video is too unclear to establish the claimed object.",
                "fix": "Use clearer visual evidence.",
            }
        },
        "choice_logit_signal": {
            "available": True,
            "choice_logits": {"PASS": -0.1, "FAIL": 0.1},
            "generated_choice": "FAIL",
        },
    }

    check = check_from_single_judge(
        judge,
        "evidence_groundedness",
        include_decision_uncertainty=True,
    )

    assert check["status"] == "FAIL"
    assert "unclear" in check["reason"]
    assert check["decision_uncertainty"]["normalized_entropy"] > 0.9


def test_strict_schema_does_not_require_legacy_pass_fail_entropy() -> None:
    uncertainty = {
        "available": True,
        "probabilities": {"PASS": 0.9, "FAIL": 0.1},
        "normalized_entropy": 0.47,
    }
    item = {
        "qa_id": "q1",
        "question": "What did we put on the table?",
        "options": ["A plate", "A cup", "A book", "A key", "A bag"],
        "correct": "A",
        "answer": "A plate",
        "required_users": ["u1", "u2"],
        "evidence": {},
        "single_user_answerability": {
            "u1": "insufficient evidence",
            "u2": "insufficient evidence",
        },
        "combined_answerability": "sufficient support",
        "model_id": "test-model",
        "source_urls": [],
        "question_type": "neutral",
        "generator_rationale": "test",
        "why_two_users_needed": "test",
        "per_user_evidence_claims": {},
        "attempt_count": 1,
        "video_evidence": [{}],
        "referred_timestamps": [],
        "human_audit": {},
        "generation_trace": [{}],
        "review": {
            "review_passed": True,
            "status": "passed",
            "judger": {
                "gate": {"passed": True},
                "checks": {
                    "qa_formality": {
                        "status": "PASS",
                        "decision_uncertainty": uncertainty,
                    },
                    "evidence_groundedness": {
                        "status": "PASS",
                        "decision_uncertainty": uncertainty,
                    },
                    "answerability": {"status": "PASS"},
                },
            },
            "answerability": {"gate": {"passed": True}, "evaluations": []},
            "schema_validation": {"passed": True},
        },
    }

    assert validate_qa_item(item, strict_review=True) == []

    item["review"]["judger"]["checks"]["evidence_groundedness"]["decision_uncertainty"] = {
        "available": False,
        "reason": "OpenRouter model does not support logprobs",
    }
    assert validate_qa_item(item, strict_review=True) == []

    del item["review"]["judger"]["checks"]["qa_formality"]["decision_uncertainty"]
    del item["review"]["judger"]["checks"]["evidence_groundedness"]["decision_uncertainty"]
    assert validate_qa_item(item, strict_review=True) == []
    entropy_errors = validate_qa_item(
        item,
        strict_review=True,
        require_decision_entropy=True,
    )
    assert len(
        [error for error in entropy_errors if "must include PASS/FAIL status entropy" in error]
    ) == 2
    assert validate_qa_item(item, strict_review=True) == []


def test_legacy_scored_opt_in_cannot_reactivate_production_prompts() -> None:
    prompt = build_evidence_groundedness_judge_prompt({}, {}, pass_fail_only=False)
    formality_prompt = build_qa_formality_judge_prompt({}, {}, pass_fail_only=False)
    output_contract = prompt[prompt.rfind("Return exactly one valid JSON object") :]

    assert '"status": "PASS/FAIL"' in output_contract
    assert '"decision"' not in output_contract
    assert 'Encode PASS as decision "P" and FAIL as decision "F"' not in prompt
    assert "first field in the JSON object must be decision" not in prompt
    assert "UNCERTAIN" not in prompt
    assert "PASS only when the question stem and declared correct answer" in prompt
    assert "evidence_groundedness quality_score rubric" not in prompt
    assert "set checks.qa_formality.status to FAIL" in formality_prompt
    assert "PASS qa_formality only when" in formality_prompt
    assert "qa_formality quality_score rubric" not in formality_prompt
    assert "Global 3-point quota" not in prompt
    assert "Previous 3-point assignments" not in prompt
    assert "quality_score" not in output_contract
    assert "*" * 2 not in prompt
    assert '"quota_rebuttal"' not in output_contract
    assert "Do not assign a numerical score" in prompt


def test_archived_quota_helpers_remain_available_off_pipeline() -> None:
    prompt = build_evidence_groundedness_judge_prompt(
        {},
        {},
        pass_fail_only=False,
        previous_three_point_assignments=48,
        quality_quota=48,
    )
    assert "Previous 3-point assignments" not in prompt
    assert "quota_rebuttal is mandatory" not in prompt

    archived_quota_prompt = quality_quota_prompt(
        previous_three_point_assignments=48,
        quota=48,
    )
    assert "Previous 3-point assignments already observed: 48." in archived_quota_prompt
    assert "Remaining 3-point capacity before this candidate: 0." in archived_quota_prompt
    assert "*" * 2 not in prompt
    assert "quota_rebuttal is mandatory" in archived_quota_prompt

    missing_rebuttal = attach_quality_quota_metadata(
        {
            "status": "PASS",
            "quality_score": 3,
            "quality_reason": "The visual evidence is exceptionally clear.",
            "quota_rebuttal": "",
        },
        quota_state=quality_quota_snapshot(48, 48),
    )
    assert missing_rebuttal["quality_quota"]["quota_exceeded_by_this_assignment"] is True
    assert missing_rebuttal["quality_quota"]["quota_rebuttal_required"] is True
    assert missing_rebuttal["quality_quota"]["output_contract_satisfied"] is False

    with_rebuttal = attach_quality_quota_metadata(
        {
            "status": "PASS",
            "quality_score": 3,
            "quality_reason": "The visual evidence is exceptionally clear.",
            "quota_rebuttal": (
                "Despite the exhausted quota, downgrading this plainly visible and fully "
                "aligned evidence would misrepresent the rubric."
            ),
        },
        quota_state=quality_quota_snapshot(48, 48),
    )
    assert with_rebuttal["quality_quota"]["output_contract_satisfied"] is True


def test_resume_quota_counts_every_scored_judge_attempt() -> None:
    def trace(formality_score: int, grounding_score: int) -> dict:
        return {
            "judge": {
                "merged": {
                    "checks": {
                        "qa_formality": {"quality_score": formality_score},
                        "evidence_groundedness": {"quality_score": grounding_score},
                    }
                }
            }
        }

    rows = [
        {"attempts": [trace(3, 2), trace(3, 3)]},
        {"generation_trace": [trace(1, 3)]},
    ]
    assert quality_quota_counts_from_rows(rows) == {
        "qa_formality": 2,
        "evidence_groundedness": 2,
    }
    compact_rows = [
        {
            "production_judge_config": {
                "observed_three_point_assignments": {
                    "qa_formality": 48,
                    "evidence_groundedness": 49,
                }
            }
        }
    ]
    assert quality_quota_counts_from_rows(compact_rows) == {
        "qa_formality": 48,
        "evidence_groundedness": 49,
    }
    compact_and_trace_rows = [
        {
            "production_judge_config": {
                "observed_three_point_assignments": {
                    "qa_formality": 2,
                    "evidence_groundedness": 2,
                }
            },
            "attempts": [trace(3, 2), trace(3, 3), trace(1, 3)],
        }
    ]
    assert quality_quota_counts_from_rows(compact_and_trace_rows) == {
        "qa_formality": 2,
        "evidence_groundedness": 2,
    }


def test_scored_quota_metadata_does_not_override_pass_fail_gate() -> None:
    def branch(check_name: str) -> dict:
        semantic_subchecks = (
            passing_formality_subchecks() if check_name == "qa_formality" else None
        )
        return {
            "review_passed": True,
            "checks": {
                check_name: {
                    "status": "PASS",
                    "reason": "The gate passes independently.",
                    "fix": "",
                    "quality_score": 3,
                    "quality_flag": "3_strong",
                    "quality_reason": "The candidate exactly matches the strong rubric.",
                    "quota_rebuttal": (
                        "Despite the exhausted quota, this candidate remains unambiguously strong."
                    ),
                    **(
                        {"semantic_subchecks": semantic_subchecks}
                        if semantic_subchecks is not None
                        else {}
                    ),
                }
            },
            "blocking_failures": [],
            "feedback_to_generator": "",
        }

    merged = merge_parallel_judges(
        qa_formality_judge=branch("qa_formality"),
        evidence_groundedness_judge=branch("evidence_groundedness"),
        answerability={"gate": {"passed": True, "reason": "passed"}},
        schema_errors=[],
        quality_quota_by_check={
            "qa_formality": quality_quota_snapshot(48, 48),
            "evidence_groundedness": quality_quota_snapshot(48, 48),
        },
    )

    assert merged["gate"]["passed"] is True
    assert "decision_uncertainty_summary" not in merged
    for check_name in ("qa_formality", "evidence_groundedness"):
        check = merged["checks"][check_name]
        assert "decision_uncertainty" not in check
        assert check["quality_quota"]["quota_exceeded_by_this_assignment"] is True
        assert check["quality_quota"]["output_contract_satisfied"] is True


def test_formality_is_text_only_while_grounding_may_receive_generator_rationale() -> None:
    generator_claim = "GENERATOR CLAIM: the bowl contains chips"
    qa = {
        "qa_id": "q1",
        "evidence_id": "e1",
        "question_type": "neutral",
        "question": "What was in the bowl?",
        "options": ["Chips", "Dough", "Fruit", "Keys", "Water"],
        "correct": "A",
        "answer": "Chips",
        "required_users": ["u1", "u2"],
        "evidence": generator_claim,
        "single_user_answerability": generator_claim,
        "combined_answerability": generator_claim,
        "generator_rationale": generator_claim,
        "why_two_users_needed": generator_claim,
        "per_user_evidence_claims": generator_claim,
        "referred_timestamps": generator_claim,
        "review": {"generator_self_check": generator_claim},
    }

    judge_qa = qa_for_judger_prompt(qa)
    formality_prompt = build_qa_formality_judge_prompt(judge_qa, {})
    grounding_prompt = build_evidence_groundedness_judge_prompt(judge_qa, {})

    assert judge_qa == {
        key: qa[key]
        for key in (
            "qa_id",
            "evidence_id",
            "question_type",
            "question",
            "options",
            "correct",
            "answer",
            "required_users",
            "generator_rationale",
        )
    }
    assert generator_claim not in formality_prompt
    assert generator_claim in grounding_prompt
    assert "pure text-only semantic judge" in formality_prompt
    assert "Do not use hidden generator intent" in formality_prompt
    assert "treat every claim in it as unverified" in grounding_prompt
    assert "Treat every object, action, person, state, identity, and continuity description" in grounding_prompt


def test_production_judges_hide_rationale_and_remove_quality_scoring() -> None:
    generator_claim = "GENERATOR CLAIM: the bowl contains chips"
    qa = {
        "qa_id": "q1",
        "question": "What was in the bowl?",
        "options": ["Chips", "Dough", "Fruit", "Keys", "Water"],
        "correct": "A",
        "answer": "Chips",
        "required_users": ["u1", "u2"],
        "generator_rationale": generator_claim,
    }
    judge_qa = qa_for_judger_prompt(qa, include_generator_rationale=False)
    formality_prompt = build_qa_formality_judge_prompt(
        judge_qa,
        {},
        pass_fail_only=True,
    )
    grounding_prompt = build_evidence_groundedness_judge_prompt(
        judge_qa,
        {},
        pass_fail_only=True,
    )

    assert "generator_rationale" not in judge_qa
    assert generator_claim not in formality_prompt
    assert generator_claim not in grounding_prompt
    assert "quality_score" not in formality_prompt
    assert "quality_score" not in grounding_prompt
    assert "why_generator_asked_this" not in formality_prompt
    assert "why_generator_asked_this" not in grounding_prompt
    assert "1 / 1_weak_or_reject" not in grounding_prompt
    assert "Do not assign a numerical score" in formality_prompt
    assert "Do not assign a numerical score" in grounding_prompt
    assert '"status": "PASS/FAIL"' in formality_prompt
    assert '"status": "PASS/FAIL"' in grounding_prompt


def test_answerability_prompt_puts_choice_before_explanation() -> None:
    prompt = build_answerability_prompt(
        {"question": "Question?", "options": [f"Option {choice}" for choice in "ABCDE"]},
        {"condition_id": "combined", "users": ["u1", "u2"]},
    )
    output_contract = prompt[prompt.rfind("Return exactly one valid JSON object") :]

    assert "first field in the JSON object must be choice" not in prompt
    assert output_contract.index('"choice"') < output_contract.index('"answer_text"')


def test_judge_branch_requests_and_attaches_pass_fail_status_entropy() -> None:
    class Runner:
        requested_field = None
        requested_choices = None

        def generate_with_choice_logits(self, *args, **kwargs):
            self.requested_field = kwargs.get("field_name")
            self.requested_choices = kwargs.get("choices")
            return {
                "text": (
                    '{"review_passed": true, "checks": '
                    '{"evidence_groundedness": {"status": "PASS", "reason": "visible", "fix": ""}}, '
                    '"blocking_failures": [], "why_generator_asked_this": "", '
                    '"feedback_to_generator": ""}'
                ),
                "choice_logits": {
                    "available": True,
                    "choice_logits": {"PASS": 1.0, "FAIL": -3.0},
                    "weight_type": "logit",
                    "generated_choice": "PASS",
                    "token_index": 2,
                },
            }

    runner = Runner()
    judge = run_model_judge_branch(
        check_name="evidence_groundedness",
        prompt="judge",
        runner=runner,
        image_paths=[],
        video_paths=[],
        evidence_id="e1",
        qa_id="q1",
        attempt=1,
        collect_choice_logits=True,
    )
    check = check_from_single_judge(
        judge,
        "evidence_groundedness",
        include_decision_uncertainty=True,
    )

    assert runner.requested_field == "status"
    assert runner.requested_choices == ("PASS", "FAIL")
    assert check["status"] == "PASS"
    assert check["decision_uncertainty"]["available"] is True
    assert check["decision_uncertainty"]["generated_decision"] == "PASS"
    assert check["status_matches_effective_status"] is True


def test_pass_fail_only_judge_branch_skips_logits_and_entropy() -> None:
    class Runner:
        generate_calls = 0

        def generate_with_choice_logits(self, *args, **kwargs):
            raise AssertionError("pass/fail-only mode must not request choice logits")

        def generate(self, *args, **kwargs):
            self.generate_calls += 1
            return (
                '{"review_passed": true, "checks": '
                '{"evidence_groundedness": {"status": "PASS", "reason": "visible", "fix": ""}}, '
                '"blocking_failures": [], "why_generator_asked_this": "", '
                '"feedback_to_generator": ""}'
            )

    runner = Runner()
    judge = run_model_judge_branch(
        check_name="evidence_groundedness",
        prompt="judge",
        runner=runner,
        image_paths=[],
        video_paths=[],
        evidence_id="e1",
        qa_id="q1",
        attempt=1,
        collect_choice_logits=False,
    )
    check = check_from_single_judge(
        judge,
        "evidence_groundedness",
        include_decision_uncertainty=False,
    )

    assert runner.generate_calls == 1
    assert "choice_logit_signal" not in judge
    assert "decision_uncertainty" not in check
    assert "status_matches_effective_status" not in check


def test_malformed_judge_json_gets_one_text_only_repair_attempt() -> None:
    class Runner:
        def __init__(self) -> None:
            self.calls = []

        def generate(self, prompt, **kwargs):
            self.calls.append((prompt, kwargs))
            if len(self.calls) == 1:
                return "PASS because the evidence is visible"
            return (
                '{"review_passed":true,"checks":{"evidence_groundedness":'
                '{"status":"PASS","reason":"visible","fix":""}},'
                '"blocking_failures":[],"feedback_to_generator":""}'
            )

    runner = Runner()
    judge = run_model_judge_branch(
        check_name="evidence_groundedness",
        prompt="original judge prompt",
        runner=runner,
        image_paths=["full-frame.jpg"],
        video_paths=["full-video.mp4"],
        evidence_id="e1",
        qa_id="q1",
        attempt=1,
    )

    assert len(runner.calls) == 2
    assert runner.calls[0][1] == {
        "image_paths": ["full-frame.jpg"],
        "video_paths": ["full-video.mp4"],
    }
    repair_prompt, repair_kwargs = runner.calls[1]
    assert "Preserve the same decision and content" in repair_prompt
    assert "PASS because the evidence is visible" in repair_prompt
    assert repair_kwargs == {"image_paths": [], "video_paths": []}
    assert judge["checks"]["evidence_groundedness"]["status"] == "PASS"
    assert judge["format_repair"]["attempted"] is True
    assert judge["format_repair"]["succeeded"] is True
    assert judge["initial_raw_output"] == "PASS because the evidence is visible"


def test_judge_json_repair_is_bounded_and_schema_aware() -> None:
    class Runner:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, prompt, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return '{"review_passed":true,"checks":{"qa_formality":{"status":"PASS"}}}'
            return "still malformed"

    runner = Runner()
    judge = run_model_judge_branch(
        check_name="qa_formality",
        prompt="original judge prompt",
        runner=runner,
        image_paths=[],
        video_paths=[],
        evidence_id="e1",
        qa_id="q1",
        attempt=1,
    )

    assert runner.calls == 2
    assert judge["checks"]["qa_formality"]["status"] == "FAIL"
    assert "after one JSON repair attempt" in judge["checks"]["qa_formality"]["reason"]
    assert judge["format_repair"]["attempted"] is True
    assert judge["format_repair"]["succeeded"] is False
    assert "judge JSON contract errors" in judge["format_repair"]["initial_error"]


def test_all_formality_subchecks_are_deterministic_blockers() -> None:
    def branch(check_name: str, *, failed_subcheck: str | None = None) -> dict:
        check = {"status": "PASS", "reason": "model said pass", "fix": ""}
        if check_name == "qa_formality":
            check["semantic_subchecks"] = passing_formality_subchecks()
            if failed_subcheck:
                check["semantic_subchecks"][failed_subcheck] = {
                    "status": "FAIL",
                    "reason": "deliberate regression probe",
                }
        return {
            "review_passed": True,
            "checks": {check_name: check},
            "blocking_failures": [],
            "feedback_to_generator": "",
        }

    for failed_subcheck in QA_FORMALITY_SEMANTIC_SUBCHECK_NAMES:
        merged = merge_parallel_judges(
            qa_formality_judge=branch(
                "qa_formality",
                failed_subcheck=failed_subcheck,
            ),
            evidence_groundedness_judge=branch("evidence_groundedness"),
            answerability={"gate": {"passed": True, "reason": "passed"}},
            schema_errors=[],
        )
        assert merged["checks"]["qa_formality"]["status"] == "FAIL"
        assert merged["gate"]["passed"] is False
        assert failed_subcheck in merged["checks"]["qa_formality"]["reason"]


def test_timestamp_is_judge_decided_while_known_names_remain_deterministic() -> None:
    qa_item = {
        "question": "What happened 35 seconds into the recording?",
        "options": ["One", "Two", "Three", "Four", "Five"],
    }
    assert qa_formality_errors(qa_item, []) == []
    formality_check = {
        "status": "PASS",
        "reason": "model said pass",
        "fix": "",
        "semantic_subchecks": passing_formality_subchecks(),
    }
    merged = merge_parallel_judges(
        qa_formality_judge={
            "review_passed": True,
            "checks": {"qa_formality": formality_check},
            "blocking_failures": [],
            "feedback_to_generator": "",
        },
        evidence_groundedness_judge={
            "review_passed": True,
            "checks": {
                "evidence_groundedness": {
                    "status": "PASS",
                    "reason": "visible",
                    "fix": "",
                }
            },
            "blocking_failures": [],
            "feedback_to_generator": "",
        },
        answerability={"gate": {"passed": True, "reason": "passed"}},
        schema_errors=[],
        qa_item=qa_item,
    )
    assert merged["checks"]["qa_formality"]["schema_branch"]["status"] == "PASS"
    assert merged["gate"]["passed"] is True

    formality_check["semantic_subchecks"]["timestamp_citation"] = {
        "status": "FAIL",
        "reason": "The question cites a seconds-from-start coordinate.",
    }
    judge_rejected = merge_parallel_judges(
        qa_formality_judge={
            "review_passed": False,
            "checks": {"qa_formality": formality_check},
            "blocking_failures": ["qa_formality"],
            "feedback_to_generator": "Remove the timestamp citation.",
        },
        evidence_groundedness_judge={
            "review_passed": True,
            "checks": {
                "evidence_groundedness": {
                    "status": "PASS",
                    "reason": "visible",
                    "fix": "",
                }
            },
            "blocking_failures": [],
            "feedback_to_generator": "",
        },
        answerability={"gate": {"passed": True, "reason": "passed"}},
        schema_errors=[],
        qa_item=qa_item,
    )
    assert judge_rejected["checks"]["qa_formality"]["schema_branch"]["status"] == "PASS"
    assert judge_rejected["checks"]["qa_formality"]["status"] == "FAIL"
    assert judge_rejected["gate"]["passed"] is False

    name_qa = {
        "question": "After I left, where did Alice put the cup?",
        "options": ["One", "Two", "Three", "Four", "Five"],
    }
    name_errors = qa_formality_errors(
        name_qa,
        [],
        participant_names=["Alice", "Jake"],
    )
    assert name_errors and "participant name" in name_errors[0]


def test_production_answerability_uses_json_generation_without_logits() -> None:
    class Runner:
        generate_calls = 0

        def generate_with_choice_logits(self, *args, **kwargs):
            raise AssertionError("production answerability must not request choice logits")

        def generate(self, *args, **kwargs):
            self.generate_calls += 1
            return (
                '{"choice": "A", "answer_text": "Option A", '
                '"evidence_used": "visible", "insufficient_reason": ""}'
            )

    runner = Runner()
    qa_item = {
        "qa_id": "q1",
        "required_users": ["u1", "u2"],
        "question": "Question?",
        "options": [f"Option {choice}" for choice in "ABCDE"],
        "correct": "A",
    }
    result = run_answerability_eval(
        qa_item=qa_item,
        packet={"clips": []},
        runner=runner,
        media_backend="transformers-local",
        allow_openai_video_input=False,
        prompt_rows=[],
    )

    assert len(result["evaluations"]) == 3
    assert runner.generate_calls == 3
    assert all("choice_uncertainty" not in evaluation for evaluation in result["evaluations"])
    assert result["gate"]["passed"] is False
    assert "asker/subset condition answered correctly" in result["gate"]["reason"]
    assert answerability_gate(qa_item, result["evaluations"]) == result["gate"]


def test_generation_prompt_is_category_free_with_underrepresented_implicit_families() -> None:
    prompt = build_video_generation_prompt(
        {"evidence_id": "e1", "required_users": ["u1", "u2"], "clips": []},
        "neutral",
    )

    assert POSITIVE_EXAMPLES_GUIDANCE in prompt
    assert "privately consider several substantively different information needs" in prompt
    assert "Cross-view comparison or asymmetry." in prompt
    assert "Cross-view identity or role linkage." in prompt
    assert "Post-handoff recipient follow-up." in prompt
    assert "Concrete state verification or change." in prompt
    assert "Cross-view concurrent activity." in prompt
    assert "Single-anchor matching" in prompt
    assert "Pair matching" in prompt
    assert "every option contains one concrete activity associated with each view" in prompt
    assert "Do not force a family or default automatically to an object" in prompt
    assert "Lead with the information request" in prompt
    assert "First-person perspective does not require putting I or we first" in prompt
    assert 'Example structure: "I could tell' not in prompt
    assert 'Example structure: "I could see' not in prompt
    assert 'Example structure: "After I' not in prompt
    assert "Clear visible quantity:" not in prompt
    assert "three open boxes" not in prompt
    assert "guitar" not in prompt.lower()
    assert "microwave" not in prompt.lower()
    assert "Historical unsteered runs" not in prompt
    assert "Optional underrepresented question direction:" not in prompt
    assert "Concurrent-activity guidance:" not in prompt
    assert "A concurrent-activity relation is allowed only when" not in prompt
    assert "A concurrent-activity question may be valid" not in prompt
    assert "For a time-anchored concurrent question" not in prompt
    assert "Choose the strongest natural grounded relation supported by the videos" in prompt
    assert "Do not stitch unrelated scenes together" in prompt
    assert "required_users[1] may be sufficient alone" in prompt
    assert "Prefer a strict two-view relation" not in prompt
    assert "fallback only if no grounded strict relation exists" not in prompt
    assert "Broad two-user reasoning categories:" not in prompt
    assert "cross_view_concurrent_activity" not in prompt
    assert '"category"' not in prompt
    assert '"category_rationale"' not in prompt
    assert "category" not in VIDEO_GENERATION_SCHEMA
    assert "category_rationale" not in VIDEO_GENERATION_SCHEMA


def test_underrepresented_relational_families_are_reconciled_with_judges() -> None:
    qa = {
        "question": "Was the laptop user also the person preparing food, or were they different people?",
        "options": [
            "They were two different people in separate rooms",
            "They were the same person moving between rooms",
            "Only the laptop activity was visibly established",
            "Only the food preparation was visibly established",
            "Neither activity involved a clearly visible person",
        ],
        "correct": "A",
        "answer": "They were two different people in separate rooms",
        "required_users": ["u1", "u2"],
        "generator_rationale": "Each view supplies one role and the combined views link them.",
    }
    formality_prompt = build_qa_formality_judge_prompt(qa, {})
    grounding_prompt = build_evidence_groundedness_judge_prompt(qa, {})
    answerability_prompt = build_answerability_prompt(
        qa,
        {"condition": "single_user", "included_users": ["u1"]},
    )

    assert "PASS linked task outcomes, interactions, and post-handoff follow-ups" in formality_prompt
    assert "Judge semantic form only, not whether the described facts are true" in formality_prompt
    assert "For identity or role linkage" in grounding_prompt
    assert "verify the initial exchange, same recipient, same object" in grounding_prompt
    assert "verify the exact object and observed state" in grounding_prompt
    assert "comparison or identity-linkage question" in answerability_prompt
    assert "post-handoff follow-up" in answerability_prompt


def test_time_anchored_concurrent_activity_is_allowed_but_must_be_strict() -> None:
    qa = {
        "question": "During the part where I was outside scrolling on my phone, what happened in the kitchen?",
        "options": [
            "A covered bowl was placed in the microwave",
            "A red refrigerator door was opened",
            "A tray of pastries was carried outside",
            "A cutting board was rinsed in the sink",
            "A grocery basket was placed on the table",
        ],
        "correct": "A",
        "answer": "A covered bowl was placed in the microwave",
        "required_users": ["u1", "u2"],
        "generator_rationale": (
            "u1 shows the distinctive temporal anchor; u2 shows the concrete event during the "
            "same synchronized interval; neither view alone supplies both pieces"
        ),
    }
    formality_prompt = build_qa_formality_judge_prompt(qa, {})
    grounding_prompt = build_evidence_groundedness_judge_prompt(qa, {})
    answerability_prompt = build_answerability_prompt(
        qa,
        {"condition": "single_user", "included_users": ["u2"]},
    )

    assert "PASS a concrete temporal relation in either direction" in formality_prompt
    assert "Do not judge whether the anchor truly localizes an interval" in formality_prompt
    assert "For a single-anchor concurrent question" in grounding_prompt
    assert "Either view may supply the fixed event" in grounding_prompt
    assert "relative temporal key" in answerability_prompt
    assert "Do not treat the start or end of a cropped condition as an implicit temporal key" in answerability_prompt


def test_concurrent_activity_pair_matching_may_use_actions_seen_by_either_wearer() -> None:
    qa = {
        "question": "Which pair of things I saw happening took place at about the same time?",
        "options": [
            "Someone outside played guitar while someone inside started the microwave",
            "Someone outside opened a box while someone inside washed a pan",
            "Someone outside played guitar while someone inside washed a pan",
            "Someone outside opened a box while someone inside started the microwave",
            "Someone outside moved a chair while someone inside carried a plate",
        ],
        "correct": "A",
        "answer": "Someone outside played guitar while someone inside started the microwave",
        "required_users": ["u1", "u2"],
        "generator_rationale": (
            "One concrete action is visible in each video and their original synchronized "
            "intervals overlap; each single view lacks the other half of the pairing"
        ),
    }
    formality_prompt = build_qa_formality_judge_prompt(qa, {})
    grounding_prompt = build_evidence_groundedness_judge_prompt(qa, {})
    answerability_prompt = build_answerability_prompt(
        qa,
        {"condition": "single_user", "included_users": ["u1"]},
    )

    assert "PASS a pair-matching form when every option states a complete cross-view activity pair" in formality_prompt
    assert "every component activity used across the options actually occurs" in grounding_prompt
    assert "exactly the declared pair overlaps on the original synchronized timeline" in grounding_prompt
    assert "equal playback positions in independently pruned videos" in grounding_prompt
    assert "shows only one side's activities" in answerability_prompt


def test_identical_explicit_judge_config_reuses_generator_runner() -> None:
    class Runner:
        def __init__(self, model_id: str) -> None:
            self.model_id = model_id

    calls: list[tuple[str, str]] = []

    def fake_make_runner(backend: str, **kwargs):
        calls.append((backend, kwargs["model_id"]))
        return Runner(kwargs["model_id"])

    class MemoryRows(list):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

        def load_existing(self) -> None:
            return None

    with (
        patch.object(video_qa_loop_module, "make_runner", side_effect=fake_make_runner),
        patch.object(video_qa_loop_module, "StreamingJsonlRows", MemoryRows),
        patch.object(video_qa_loop_module, "iter_jsonl", return_value=iter(())),
    ):
        rows = generate_video_qa_loop(
            evidence_path="unused-evidence.jsonl",
            output_path="unused-accepted.jsonl",
            prompts_path="unused-prompts.jsonl",
            rejected_path="unused-rejected.jsonl",
            intermediate_path="unused-intermediate.jsonl",
            backend="transformers-local",
            model_id="Qwen/Qwen3.6-27B",
            max_new_tokens=4096,
            judge_backend="transformers-local",
            judge_model_id="Qwen/Qwen3.6-27B",
            judge_max_new_tokens=4096,
            target_count=50,
            question_types=("neutral",),
        )

    assert rows == []
    assert calls == [("transformers-local", "Qwen/Qwen3.6-27B")]


def test_judges_do_not_receive_or_request_category_fields() -> None:
    qa = {
        "category": "object_tracking_and_location",
        "category_rationale": (
            "u1 identifies the handed-off object and u2 shows its final destination"
        ),
        "question": "Where did the object end up?",
        "options": ["On the desk", "In the bag", "By the door", "Under the chair", "On the shelf"],
        "correct": "A",
        "answer": "On the desk",
        "required_users": ["u1", "u2"],
        "generator_rationale": "u1 supplies the context and u2 shows the destination",
    }
    judge_item = qa_for_judger_prompt(qa)
    formality_prompt = build_qa_formality_judge_prompt(judge_item, {})
    grounding_prompt = build_evidence_groundedness_judge_prompt(judge_item, {})

    assert "category" not in judge_item
    assert "category_rationale" not in judge_item
    for prompt in (formality_prompt, grounding_prompt):
        assert "Generator-declared category" not in prompt
        assert "Broad category-selection guidance" not in prompt
        assert '"category"' not in prompt
        assert '"category_rationale"' not in prompt
        assert "cross_view_concurrent_activity" not in prompt


def test_discovery_and_clip_guided_modes_are_archived() -> None:
    packet = {
        "evidence_id": "e1",
        "required_users": ["u1", "u2"],
        "clips": [],
        "clip_exclusive": {"available": True, "score": 1.0},
    }

    assert GENERATION_MODES == ("baseline",)
    baseline_prompt = build_video_generation_prompt(packet, "neutral", generation_mode="baseline")
    assert "CLIP retrieval hints" not in baseline_prompt
    for archived_mode in ("clip_guided", "discovery", "discovery_control"):
        try:
            build_video_generation_prompt(packet, "neutral", generation_mode=archived_mode)
        except ValueError as exc:
            assert "unknown generation_mode" in str(exc)
        else:
            raise AssertionError(f"archived mode remained active: {archived_mode}")


def test_parallel_merge_keeps_existing_gate_behavior_with_pass_fail_summary() -> None:
    def branch(check_name: str) -> dict:
        semantic_subchecks = (
            passing_formality_subchecks() if check_name == "qa_formality" else None
        )
        return {
            "review_passed": True,
            "checks": {
                check_name: {
                    "status": "PASS",
                    "reason": "passed",
                    "fix": "",
                    **(
                        {"semantic_subchecks": semantic_subchecks}
                        if semantic_subchecks is not None
                        else {}
                    ),
                }
            },
            "blocking_failures": [],
            "choice_logit_signal": {
                "available": True,
                "choice_logits": {"PASS": 2.0, "FAIL": -2.0},
                "generated_choice": "PASS",
            },
        }

    merged = merge_parallel_judges(
        qa_formality_judge=branch("qa_formality"),
        evidence_groundedness_judge=branch("evidence_groundedness"),
        answerability={"gate": {"passed": True, "reason": "passed"}},
        schema_errors=[],
        include_decision_uncertainty=True,
    )

    assert merged["gate"]["passed"] is True
    assert merged["review_passed"] is True
    assert merged["blocking_failures"] == []
    assert merged["decision_uncertainty_summary"]["available"] is True
    assert "quality_uncertainty_summary" not in merged


def test_postrun_selection_sort_key_implements_requested_ordering() -> None:
    cases = [
        ("PASS_low_H", "PASS", {"PASS": 5.0, "FAIL": -5.0}),
        ("PASS_high_H", "PASS", {"PASS": 0.1, "FAIL": 0.0}),
        ("FAIL_high_H", "FAIL", {"PASS": 0.0, "FAIL": 0.1}),
        ("FAIL_low_H", "FAIL", {"PASS": -5.0, "FAIL": 5.0}),
    ]
    ranked = []
    for label, decision, logits in cases:
        runtime_uncertainty = decision_uncertainty_from_choice_logits(
            {"available": True, "choice_logits": logits, "generated_choice": decision}
        )
        analysis = recompute_entropy(
            {
                **runtime_uncertainty,
                "available": True,
                "log_weights": runtime_uncertainty["log_weights"],
                "generated_decision": decision,
            }
        )
        assert "selection_sort_key" not in runtime_uncertainty
        ranked.append((analysis["selection_sort_key"], label))

    assert [label for _, label in sorted(ranked)] == [
        "PASS_low_H",
        "PASS_high_H",
        "FAIL_high_H",
        "FAIL_low_H",
    ]
