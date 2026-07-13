import math

from egolife_two_user_qa.analyze_entropy_trials import recompute_entropy
from egolife_two_user_qa.prompts import (
    build_answerability_prompt,
    build_evidence_groundedness_judge_prompt,
)
from egolife_two_user_qa.qwen3vl_runner import (
    choice_logprobs_from_top_candidates,
    locate_choice_token,
)
from egolife_two_user_qa.video_qa_loop import (
    answerability_gate,
    answerability_uncertainty_from_choice_logits,
    check_from_single_judge,
    decision_uncertainty_from_choice_logits,
    merge_parallel_judges,
    run_answerability_eval,
    run_model_judge_branch,
)


def test_locate_first_decision_token_preserves_json_punctuation() -> None:
    pieces = ['{\n  "decision"', ":", ' "P"', ',\n  "checks": {}\n}']

    located = locate_choice_token(pieces)

    assert located is not None
    assert located["token_index"] == 2
    assert located["generated_choice"] == "P"
    assert located["alternative_token_pieces"] == {"P": ' "P"', "F": ' "F"'}


def test_api_top_candidates_extract_both_first_decision_logprobs() -> None:
    pieces = ['{"decision":', ' "P"', ',"checks":{}}']
    candidates = [
        [],
        [
            {"token": ' "P"', "logprob": -0.2},
            {"token": ' "F"', "logprob": -2.0},
        ],
        [],
    ]

    signal = choice_logprobs_from_top_candidates(pieces, candidates)

    assert signal["available"] is True
    assert signal["choice_logprobs"] == {"P": -0.2, "F": -2.0}


def test_locate_choice_token_rejects_schema_placeholder() -> None:
    assert locate_choice_token(['{"decision": "P/F"}']) is None


def test_locate_choice_token_uses_first_decision_if_text_repeats_it() -> None:
    located = locate_choice_token(
        ['{"decision":"P","reason":"do not repeat \\"decision\\": \\"F\\""}']
    )

    assert located is not None
    assert located["generated_choice"] == "P"


def test_entropy_is_normalized_over_only_p_and_f() -> None:
    uncertainty = decision_uncertainty_from_choice_logits(
        {
            "available": True,
            "choice_logits": {"P": 0.0, "F": 0.0},
            "generated_choice": "P",
            "token_index": 12,
        }
    )

    assert uncertainty["available"] is True
    assert math.isclose(uncertainty["entropy_nats"], math.log(2.0), rel_tol=1e-7)
    assert uncertainty["normalized_entropy"] == 1.0
    assert uncertainty["generated_decision"] == "P"
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


def test_first_decision_derives_status_without_argmax_override() -> None:
    judge = {
        "decision": "P",
        "checks": {
            "evidence_groundedness": {
                "reason": "clear support",
                "fix": "",
            }
        },
        "choice_logit_signal": {
            "available": True,
            "choice_logits": {"P": -1.0, "F": 2.0},
            "generated_choice": "P",
            "token_index": 5,
        },
    }

    check = check_from_single_judge(judge, "evidence_groundedness")

    assert check["decision"] == "P"
    assert check["status"] == "PASS"
    assert check["decision_uncertainty"]["argmax_decision"] == "F"
    assert check["decision_uncertainty"]["generated_decision"] == "P"
    assert check["decision_matches_effective_status"] is True
    assert "quality_score" not in check


def test_nonfirst_decision_is_rejected_as_invalid_entropy_measurement() -> None:
    judge = {
        "review_passed": True,
        "decision": "P",
        "checks": {
            "evidence_groundedness": {
                "reason": "clear support",
                "fix": "",
            }
        },
        "choice_logit_signal": {
            "available": True,
            "choice_logits": {"P": 2.0, "F": -2.0},
            "generated_choice": "P",
        },
    }

    check = check_from_single_judge(judge, "evidence_groundedness")

    assert check["status"] == "FAIL"
    assert check["decision_uncertainty"]["available"] is False
    assert "first JSON field" in check["reason"]


def test_unclear_evidence_is_binary_f_with_explanation() -> None:
    judge = {
        "decision": "F",
        "checks": {
            "evidence_groundedness": {
                "reason": "The video is too unclear to establish the claimed object.",
                "fix": "Use clearer visual evidence.",
            }
        },
        "choice_logit_signal": {
            "available": True,
            "choice_logits": {"P": -0.1, "F": 0.1},
            "generated_choice": "F",
        },
    }

    check = check_from_single_judge(judge, "evidence_groundedness")

    assert check["status"] == "FAIL"
    assert "unclear" in check["reason"]
    assert check["decision_uncertainty"]["normalized_entropy"] > 0.9


def test_judge_prompt_puts_decision_before_explanation_without_quality_rubric() -> None:
    prompt = build_evidence_groundedness_judge_prompt({}, {})
    output_contract = prompt[prompt.rfind("Return exactly one valid JSON object") :]

    assert 'first field in the JSON object must be decision' in prompt
    assert output_contract.index('"decision"') < output_contract.index('"checks"')
    assert '"status"' not in output_contract
    assert "quality_score rubric" not in prompt
    assert "final_quality_score" not in prompt


def test_answerability_prompt_puts_choice_before_explanation() -> None:
    prompt = build_answerability_prompt(
        {"question": "Question?", "options": [f"Option {choice}" for choice in "ABCDE"]},
        {"condition_id": "combined", "users": ["u1", "u2"]},
    )
    output_contract = prompt[prompt.rfind("Return exactly one valid JSON object") :]

    assert "first field in the JSON object must be choice" in prompt
    assert output_contract.index('"choice"') < output_contract.index('"answer_text"')


def test_judge_branch_requests_and_attaches_first_p_f_entropy() -> None:
    class Runner:
        requested_field = None
        requested_choices = None

        def generate_with_choice_logits(self, *args, **kwargs):
            self.requested_field = kwargs.get("field_name")
            self.requested_choices = kwargs.get("choices")
            return {
                "text": (
                    '{"decision": "P", "review_passed": true, "checks": '
                    '{"evidence_groundedness": {"reason": "visible", "fix": ""}}, '
                    '"blocking_failures": [], "why_generator_asked_this": "", '
                    '"feedback_to_generator": ""}'
                ),
                "choice_logits": {
                    "available": True,
                    "choice_logits": {"P": 1.0, "F": -3.0},
                    "weight_type": "logit",
                    "generated_choice": "P",
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

    assert runner.requested_field == "decision"
    assert runner.requested_choices == ("P", "F")
    assert check["status"] == "PASS"
    assert check["decision_uncertainty"]["available"] is True
    assert check["decision_uncertainty"]["generated_decision"] == "P"
    assert check["decision_matches_effective_status"] is True


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


def test_parallel_merge_keeps_existing_gate_behavior_with_p_f_summary() -> None:
    def branch(check_name: str) -> dict:
        return {
            "decision": "P",
            "review_passed": True,
            "checks": {
                check_name: {
                    "reason": "passed",
                    "fix": "",
                }
            },
            "blocking_failures": [],
            "choice_logit_signal": {
                "available": True,
                "choice_logits": {"P": 2.0, "F": -2.0},
                "generated_choice": "P",
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
        ("P_low_H", "P", {"P": 5.0, "F": -5.0}),
        ("P_high_H", "P", {"P": 0.1, "F": 0.0}),
        ("F_high_H", "F", {"P": 0.0, "F": 0.1}),
        ("F_low_H", "F", {"P": -5.0, "F": 5.0}),
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
        "P_low_H",
        "P_high_H",
        "F_high_H",
        "F_low_H",
    ]
