from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _attempt_review(attempt: dict[str, Any]) -> dict[str, Any]:
    judge_trace = attempt.get("judge") if isinstance(attempt.get("judge"), dict) else {}
    judger = judge_trace.get("merged") if isinstance(judge_trace.get("merged"), dict) else judge_trace
    answerability = attempt.get("answerability") if isinstance(attempt.get("answerability"), dict) else {}
    schema_errors = list(attempt.get("schema_errors") or ())
    result = attempt.get("result") if isinstance(attempt.get("result"), dict) else {}
    return {
        "review_passed": result.get("accepted") if isinstance(result.get("accepted"), bool) else judger.get("review_passed"),
        "judger": judger if isinstance(judger, dict) else {},
        "answerability": answerability,
        "schema_validation": {"passed": not schema_errors, "errors": schema_errors},
        "final_decision": {
            "accepted": result.get("accepted"),
            "rejection_stage": None if result.get("accepted") is True else None,
            "reason": result.get("reason"),
        },
    }


def _attempt_qa(attempt: dict[str, Any]) -> dict[str, Any]:
    generation = attempt.get("generation") if isinstance(attempt.get("generation"), dict) else {}
    parsed = generation.get("parsed_qa")
    if isinstance(parsed, dict):
        return dict(parsed)
    normalized = generation.get("normalized_qa")
    return dict(normalized) if isinstance(normalized, dict) else {}


def iter_intermediate_attempt_inputs(path: str | Path) -> Iterable[dict[str, Any]]:
    for row_index, row in enumerate(iter_jsonl(path), 1):
        evidence_id = str(row.get("evidence_id") or row.get("qa_id") or f"row_{row_index}")
        attempts = row.get("attempts") if isinstance(row.get("attempts"), list) else []
        for attempt_index, attempt in enumerate(attempts, 1):
            if not isinstance(attempt, dict):
                continue
            qa = _attempt_qa(attempt)
            answerability = attempt.get("answerability") if isinstance(attempt.get("answerability"), dict) else {}
            review = _attempt_review(attempt)
            has_signals = bool(qa and review.get("judger") and answerability.get("evaluations"))
            yield {
                "candidate_id": f"{evidence_id}::{attempt.get('attempt', attempt_index)}",
                "group_id": evidence_id,
                "evidence_id": evidence_id,
                "qa_id": str(attempt.get("qa_id") or qa.get("qa_id") or row.get("qa_id") or ""),
                "attempt": attempt.get("attempt") if isinstance(attempt.get("attempt"), int) else attempt_index,
                "qa": qa,
                "review": review,
                "answerability": answerability,
                "schema_errors": list(attempt.get("schema_errors") or ()),
                "raw_qa": str((attempt.get("generation") or {}).get("raw_output") or ""),
                "mask_reason": None if has_signals else "missing_review_signals",
            }


def iter_accepted_qa_inputs(path: str | Path) -> Iterable[dict[str, Any]]:
    for index, qa in enumerate(iter_jsonl(path), 1):
        qa_id = str(qa.get("qa_id") or f"qa_{index}")
        evidence_id = str(qa.get("evidence_id") or qa_id)
        review = qa.get("review") if isinstance(qa.get("review"), dict) else {}
        answerability = review.get("answerability") if isinstance(review.get("answerability"), dict) else {}
        has_signals = bool(review.get("judger") and answerability.get("evaluations"))
        yield {
            "candidate_id": f"{evidence_id}::{index}",
            "group_id": evidence_id,
            "evidence_id": evidence_id,
            "qa_id": qa_id,
            "attempt": index,
            "qa": qa,
            "review": review,
            "answerability": answerability,
            "schema_errors": [],
            "raw_qa": json.dumps(qa, ensure_ascii=False, sort_keys=True),
            "mask_reason": None if has_signals else "missing_review_signals",
        }


def iter_reward_inputs(path: str | Path, input_kind: str) -> Iterable[dict[str, Any]]:
    if input_kind == "intermediate":
        yield from iter_intermediate_attempt_inputs(path)
    elif input_kind == "accepted":
        yield from iter_accepted_qa_inputs(path)
    else:
        raise ValueError(f"unsupported input_kind: {input_kind}")

