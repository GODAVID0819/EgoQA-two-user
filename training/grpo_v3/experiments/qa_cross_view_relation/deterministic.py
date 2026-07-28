from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal, Sequence

from training.grpo_v3.shared.json_format import validate_completion_json


FormatStatus = Literal["raw_valid", "repaired", "unrecoverable"]
QUESTION_TYPES = {"commonality", "difference", "neutral"}
DATASET_LANGUAGE_PATTERN = re.compile(
    r"\b(timestamp|frame|camera|dataset|clip|video|user\s*id|agent\s*id)\b|\b\d{1,2}:\d{2}:\d{2}\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DeterministicAssessment:
    format_status: FormatStatus
    qa: dict[str, Any] | None
    eligible_for_semantic_judge: bool
    blocking_errors: tuple[str, ...]
    audit_flags: tuple[str, ...]
    format_validation: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_status": self.format_status,
            "qa": self.qa,
            "eligible_for_semantic_judge": self.eligible_for_semantic_judge,
            "blocking_errors": list(self.blocking_errors),
            "audit_flags": list(self.audit_flags),
            "format_validation": self.format_validation,
        }


def _norm(text: Any) -> str:
    return str(text).strip()


def _mentions_required_user(question: str, users: Sequence[str]) -> bool:
    lowered = question.lower()
    for user in users:
        name = str(user).strip()
        if name and re.search(rf"\b{re.escape(name.lower())}\b", lowered):
            return True
    return False


def assess_completion(
    raw_completion: str,
    *,
    required_users: Sequence[str] | None = None,
) -> DeterministicAssessment:
    format_result = validate_completion_json(raw_completion)
    format_dict = format_result.to_dict()
    if format_result.status == "unrecoverable" or format_result.value is None:
        return DeterministicAssessment(
            format_status="unrecoverable",
            qa=None,
            eligible_for_semantic_judge=False,
            blocking_errors=("unrecoverable_json",),
            audit_flags=(),
            format_validation=format_dict,
        )

    qa = dict(format_result.value)
    errors: list[str] = []
    flags: list[str] = []
    question = _norm(qa.get("question"))
    if not question:
        errors.append("question_required")

    options_raw = qa.get("options")
    options = [_norm(item) for item in options_raw] if isinstance(options_raw, list) else []
    if len(options) != 5 or any(not item for item in options):
        errors.append("exactly_five_nonempty_options_required")
    elif len({item.lower() for item in options}) != 5:
        errors.append("options_must_be_unique")

    correct = _norm(qa.get("correct")).upper()
    if correct not in {"A", "B", "C", "D", "E"}:
        errors.append("correct_must_be_A_to_E")
    elif len(options) == 5 and _norm(qa.get("answer")) != options[ord(correct) - ord("A")]:
        errors.append("answer_must_equal_options_correct")

    question_type = _norm(qa.get("question_type"))
    if question_type not in QUESTION_TYPES:
        errors.append("question_type_invalid")

    users = [str(item).strip() for item in (required_users or qa.get("required_users") or []) if str(item).strip()]
    if len(set(users)) < 2:
        errors.append("required_users_must_have_two_distinct_users")
    if question and _mentions_required_user(question, users):
        errors.append("question_mentions_required_user")
    if question and DATASET_LANGUAGE_PATTERN.search(question):
        errors.append("question_uses_dataset_language")

    claims = qa.get("per_user_evidence_claims")
    if isinstance(claims, list) and users:
        for claim in claims:
            if isinstance(claim, dict) and str(claim.get("user") or "") not in set(users):
                errors.append("evidence_claim_user_not_required")
                break

    if format_result.status == "repaired":
        flags.append("json_repaired_without_penalty")

    return DeterministicAssessment(
        format_status=format_result.status,
        qa=qa,
        eligible_for_semantic_judge=not errors,
        blocking_errors=tuple(errors),
        audit_flags=tuple(flags),
        format_validation=format_dict,
    )
