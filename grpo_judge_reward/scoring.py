from __future__ import annotations

import json
import re
from typing import Any

from .records import JudgeRewardRecord

OPTION_LETTERS = ("A", "B", "C", "D", "E")


def normalize_choice(value: Any) -> tuple[str | None, bool]:
    text = str(value or "").strip()
    if text.lower() in {"insufficient", "not enough", "unknown", "cannot answer", "can't answer", "无法判断"}:
        return None, True
    upper = text.upper()
    if upper in OPTION_LETTERS:
        return upper, False
    match = re.search(r"\b([A-E])\b", upper)
    return (match.group(1), False) if match else (None, False)


def _status(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).strip().upper() or None


def _merged_judger(review: dict[str, Any]) -> dict[str, Any]:
    judger = review.get("judger") if isinstance(review.get("judger"), dict) else review.get("judge")
    if not isinstance(judger, dict):
        judger = review.get("merged")
    if isinstance(judger, dict) and isinstance(judger.get("merged"), dict):
        return judger["merged"]
    return judger if isinstance(judger, dict) else {}


def _checks(review: dict[str, Any]) -> dict[str, Any]:
    judger = _merged_judger(review)
    checks = judger.get("checks")
    return checks if isinstance(checks, dict) else {}


def _answerability(data: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    if isinstance(data.get("answerability"), dict):
        return data["answerability"]
    value = review.get("answerability")
    return value if isinstance(value, dict) else {}


def _schema_pass(data: dict[str, Any], review: dict[str, Any], checks: dict[str, Any]) -> tuple[bool, tuple[str, ...]]:
    errors = tuple(str(item) for item in data.get("schema_errors") or ())
    schema_validation = review.get("schema_validation") if isinstance(review.get("schema_validation"), dict) else {}
    if schema_validation.get("passed") is False:
        errors = tuple(str(item) for item in schema_validation.get("errors") or errors or ("schema_validation_failed",))
    qa_formality = checks.get("qa_formality") if isinstance(checks.get("qa_formality"), dict) else {}
    branch = qa_formality.get("schema_branch") if isinstance(qa_formality.get("schema_branch"), dict) else {}
    if _status(branch.get("status")) == "FAIL":
        errors = tuple(str(item) for item in branch.get("errors") or errors or ("schema_branch_failed",))
    return not errors, errors


def _correct_option(qa: dict[str, Any]) -> str | None:
    correct, insufficient = normalize_choice(qa.get("correct"))
    return None if insufficient else correct


def _choice_is_correct(value: Any, correct: str | None) -> tuple[str | None, bool | None, bool]:
    choice, insufficient = normalize_choice(value)
    if insufficient:
        return None, False, True
    if not correct or not choice:
        return choice, None, False
    return choice, choice == correct, False


def _condition_users(row: dict[str, Any]) -> list[str]:
    users = row.get("users")
    if isinstance(users, list):
        return [str(user) for user in users]
    condition_id = str(row.get("condition_id") or "")
    if "::" in condition_id:
        return condition_id.split("::", 1)[1].split("+")
    return []


def _find_combined(evaluations: list[dict[str, Any]]) -> dict[str, Any] | None:
    combined = [row for row in evaluations if row.get("condition_type") == "combined_all_users"]
    return combined[-1] if combined else None


def _mask_record(data: dict[str, Any], reason: str, warnings: tuple[str, ...] = ()) -> JudgeRewardRecord:
    qa = data.get("qa") if isinstance(data.get("qa"), dict) else {}
    return JudgeRewardRecord(
        candidate_id=str(data.get("candidate_id") or ""),
        group_id=str(data.get("group_id") or data.get("evidence_id") or ""),
        evidence_id=str(data.get("evidence_id") or data.get("group_id") or ""),
        qa_id=str(data.get("qa_id") or qa.get("qa_id") or ""),
        attempt=data.get("attempt") if isinstance(data.get("attempt"), int) else None,
        masked=True,
        mask_reason=reason,
        eligible_for_grpo=False,
        reward_total=None,
        warnings=warnings,
        raw_qa=str(data.get("raw_qa") or ""),
        review=data.get("review") if isinstance(data.get("review"), dict) else {},
        answerability=data.get("answerability") if isinstance(data.get("answerability"), dict) else {},
    )


def compute_judge_reward(data: dict[str, Any]) -> JudgeRewardRecord:
    if data.get("mask_reason"):
        return _mask_record(data, str(data["mask_reason"]))

    qa = data.get("qa") if isinstance(data.get("qa"), dict) else {}
    review = data.get("review") if isinstance(data.get("review"), dict) else {}
    checks = _checks(review)
    answerability = _answerability(data, review)
    evaluations = answerability.get("evaluations") if isinstance(answerability.get("evaluations"), list) else []
    warnings: list[str] = []

    schema_pass, schema_errors = _schema_pass(data, review, checks)
    if not schema_pass:
        return _mask_record(data, "schema_fail", tuple(schema_errors))
    if not qa or not checks or not evaluations:
        return _mask_record(data, "missing_review_signals")

    correct = _correct_option(qa)
    if not correct:
        return _mask_record(data, "missing_correct_option")

    qa_formality = checks.get("qa_formality") if isinstance(checks.get("qa_formality"), dict) else {}
    groundedness = checks.get("evidence_groundedness") if isinstance(checks.get("evidence_groundedness"), dict) else {}
    qa_formality_status = _status(qa_formality.get("status"))
    groundedness_status = _status(groundedness.get("status"))
    semantic = qa_formality.get("semantic_subchecks") if isinstance(qa_formality.get("semantic_subchecks"), dict) else {}
    shallow = semantic.get("other_person_activity_query") if isinstance(semantic.get("other_person_activity_query"), dict) else {}
    shallow_status = _status(shallow.get("status"))

    components = {
        "groundedness": {"PASS": 1.0, "UNCERTAIN": -0.7, "FAIL": -1.2}.get(groundedness_status, -1.2),
        "combined_answerability": 0.0,
        "grounded_answerable_bonus": 0.0,
        "subset_leakage": 0.0,
        "qa_formality": {"PASS": 0.5, "FAIL": -0.5}.get(qa_formality_status, 0.0),
        "shallow_activity_query": -0.8 if shallow_status == "FAIL" else 0.0,
        "provider_only_cap": 0.0,
        "shallow_activity_cap": 0.0,
        "speaker_leakage_cap": 0.0,
    }
    if groundedness_status not in {"PASS", "UNCERTAIN", "FAIL"}:
        warnings.append("missing_or_unknown_groundedness_status")
    if qa_formality_status not in {"PASS", "FAIL"}:
        warnings.append("missing_or_unknown_qa_formality_status")

    combined = _find_combined(evaluations)
    combined_choice, combined_correct, combined_insufficient = _choice_is_correct(combined.get("choice") if combined else None, correct)
    components["combined_answerability"] = 1.0 if combined_correct is True else -1.2

    gate = answerability.get("gate") if isinstance(answerability.get("gate"), dict) else {}
    speaker_user = str(gate.get("speaker_user") or (qa.get("required_users") or [""])[0] or "")
    provider_user = str(gate.get("evidence_provider_user") or ((qa.get("required_users") or ["", ""])[1] if len(qa.get("required_users") or []) > 1 else ""))
    speaker_choice = None
    speaker_only_correct = False
    proper_subset_correct = False
    provider_only_correct = False

    for row in evaluations:
        choice, is_correct, _ = _choice_is_correct(row.get("choice"), correct)
        users = _condition_users(row)
        if row.get("condition_type") == "single_user" and users == [speaker_user]:
            speaker_choice = choice
            speaker_only_correct = is_correct is True
        elif row.get("condition_type") == "single_user" and provider_user and users == [provider_user]:
            provider_only_correct = is_correct is True
        elif row.get("condition_type") == "proper_subset" and is_correct is True:
            proper_subset_correct = True

    if proper_subset_correct:
        components["subset_leakage"] = -0.8
    if groundedness_status == "PASS" and combined_correct is True and not provider_only_correct:
        components["grounded_answerable_bonus"] = 0.5

    total = sum(components.values())
    if provider_only_correct:
        total = min(total, 2.0)
        components["provider_only_cap"] = 2.0
    if shallow_status == "FAIL":
        total = min(total, 1.5)
        components["shallow_activity_cap"] = 1.5
    if speaker_only_correct:
        total = min(total, 0.5)
        components["speaker_leakage_cap"] = 0.5
    total = round(total, 6)

    judger = _merged_judger(review)
    final = review.get("final_decision") if isinstance(review.get("final_decision"), dict) else {}
    return JudgeRewardRecord(
        candidate_id=str(data.get("candidate_id") or ""),
        group_id=str(data.get("group_id") or data.get("evidence_id") or ""),
        evidence_id=str(data.get("evidence_id") or data.get("group_id") or ""),
        qa_id=str(data.get("qa_id") or qa.get("qa_id") or ""),
        attempt=data.get("attempt") if isinstance(data.get("attempt"), int) else None,
        masked=False,
        mask_reason=None,
        eligible_for_grpo=True,
        reward_total=total,
        reward_components=components,
        warnings=tuple(warnings),
        schema_pass=True,
        groundedness_status=groundedness_status,
        combined_choice=combined_choice,
        combined_correct=combined_correct,
        combined_insufficient=combined_insufficient,
        speaker_user=speaker_user,
        speaker_only_choice=speaker_choice,
        speaker_only_correct=speaker_only_correct,
        proper_subset_correct=proper_subset_correct,
        provider_only_correct=provider_only_correct,
        qa_formality_status=qa_formality_status,
        shallow_activity_status=shallow_status,
        review_passed=review.get("review_passed") if isinstance(review.get("review_passed"), bool) else judger.get("review_passed"),
        rejection_stage=final.get("rejection_stage"),
        feedback_to_generator=str(judger.get("feedback_to_generator") or ""),
        raw_qa=str(data.get("raw_qa") or json.dumps(qa, ensure_ascii=False, sort_keys=True)),
        review=review,
        answerability=answerability,
    )
