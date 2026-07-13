import math

from egolife_two_user_qa.analyze_entropy_trials import recompute_entropy
from egolife_two_user_qa.prompts import (
    build_answerability_prompt,
    build_evidence_groundedness_judge_prompt,
    build_qa_formality_judge_prompt,
)
from egolife_two_user_qa.qwen3vl_runner import (
    choice_logprobs_from_top_candidates,
    locate_choice_token,
)
from egolife_two_user_qa.schema import validate_qa_item
from egolife_two_user_qa.video_qa_loop import (
    answerability_gate,
    answerability_uncertainty_from_choice_logits,
    check_from_single_judge,
    decision_uncertainty_from_choice_logits,
    merge_parallel_judges,
    run_answerability_eval,
    run_model_judge_branch,
)


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

    check = check_from_single_judge(judge, "evidence_groundedness")

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

    check = check_from_single_judge(judge, "evidence_groundedness")

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

    check = check_from_single_judge(judge, "evidence_groundedness")

    assert check["status"] == "FAIL"
    assert "unclear" in check["reason"]
    assert check["decision_uncertainty"]["normalized_entropy"] > 0.9


def test_strict_schema_accepts_pass_fail_status_entropy_without_proxy_decision() -> None:
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


def test_judge_prompt_restores_historical_status_based_contract() -> None:
    prompt = build_evidence_groundedness_judge_prompt({}, {})
    formality_prompt = build_qa_formality_judge_prompt({}, {})
    output_contract = prompt[prompt.rfind("Return exactly one valid JSON object") :]

    assert '"status": "PASS/FAIL"' in output_contract
    assert '"decision"' not in output_contract
    assert 'Encode PASS as decision "P" and FAIL as decision "F"' not in prompt
    assert "first field in the JSON object must be decision" not in prompt
    assert "UNCERTAIN" not in prompt
    assert "PASS only when the correct answer and all material evidence" in prompt
    assert "evidence_groundedness quality_score rubric" in prompt
    assert "set checks.qa_formality.status to FAIL" in formality_prompt
    assert "PASS qa_formality only if" in formality_prompt
    assert "qa_formality quality_score rubric" in formality_prompt


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
    )
    check = check_from_single_judge(judge, "evidence_groundedness")

    assert runner.requested_field == "status"
    assert runner.requested_choices == ("PASS", "FAIL")
    assert check["status"] == "PASS"
    assert check["decision_uncertainty"]["available"] is True
    assert check["decision_uncertainty"]["generated_decision"] == "PASS"
    assert check["status_matches_effective_status"] is True


def test_answerability_conditions_capture_greedy_a_e_entropy_without_changing_gate() -> None:
    class Runner:
        requests = []

        def generate_with_choice_logits(self, *args, **kwargs):
            self.requests.append((kwargs.get("field_name"), kwargs.get("choices")))
            return {
                "text": (
                    '{"choice": "A", "answer_text": "Option A", '
                    '"evidence_used": "visible", "insufficient_reason": ""}'
                ),
                "choice_logits": {
                    "available": True,
                    "choice_logits": {
                        "A": 2.0,
                        "B": 1.0,
                        "C": 0.0,
                        "D": -1.0,
                        "E": -2.0,
                    },
                    "generated_choice": "A",
                    "weight_type": "logit",
                    "token_index": 2,
                },
            }

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
    assert runner.requests == [("choice", tuple("ABCDE"))] * 3
    assert all(
        evaluation["choice_uncertainty"]["available"] is True
        for evaluation in result["evaluations"]
    )
    assert result["gate"]["passed"] is False
    assert "asker/subset condition answered correctly" in result["gate"]["reason"]
    without_uncertainty = [
        {key: value for key, value in evaluation.items() if key != "choice_uncertainty"}
        for evaluation in result["evaluations"]
    ]
    assert answerability_gate(qa_item, without_uncertainty) == result["gate"]


def test_parallel_merge_keeps_existing_gate_behavior_with_pass_fail_summary() -> None:
    def branch(check_name: str) -> dict:
        return {
            "review_passed": True,
            "checks": {
                check_name: {
                    "status": "PASS",
                    "reason": "passed",
                    "fix": "",
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
