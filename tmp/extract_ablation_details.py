from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def traces(row: dict[str, Any]) -> list[dict[str, Any]]:
    direct_attempts = [
        value
        for value in row.get("attempts") or []
        if isinstance(value, dict) and isinstance(value.get("generation"), dict)
    ]
    if direct_attempts:
        return direct_attempts
    values = row.get("generation_trace")
    if isinstance(values, list) and values:
        return [value for value in values if isinstance(value, dict)]
    recovered: list[dict[str, Any]] = []
    for wrapper in row.get("attempts") or []:
        if not isinstance(wrapper, dict):
            continue
        qa = wrapper.get("qa") if isinstance(wrapper.get("qa"), dict) else {}
        attempt = wrapper.get("attempt")
        matches = [
            value
            for value in qa.get("generation_trace") or []
            if isinstance(value, dict) and value.get("attempt") == attempt
        ]
        if matches:
            recovered.append(matches[-1])
    return recovered


def qa_from_trace(trace: dict[str, Any]) -> dict[str, Any]:
    generation = trace.get("generation") or {}
    parsed = generation.get("parsed_qa")
    if isinstance(parsed, dict):
        qa = dict(parsed)
        normalized = generation.get("normalized_qa")
        if isinstance(normalized, dict):
            for key, value in normalized.items():
                qa.setdefault(key, value)
        return qa
    raw = generation.get("raw_output")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return {}


def compact_qa(qa: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "question",
        "options",
        "correct",
        "answer",
        "required_users",
        "evidence",
        "referred_timestamps",
        "single_user_answerability",
        "combined_answerability",
        "generator_rationale",
        "why_two_users_needed",
        "per_user_evidence_claims",
    )
    return {key: qa.get(key) for key in keys}


def qa_hash(qa: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(qa, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def summarize_trace(
    trace: dict[str, Any], qa_override: dict[str, Any] | None = None
) -> dict[str, Any]:
    qa = qa_override or qa_from_trace(trace)
    generation = trace.get("generation") if isinstance(trace.get("generation"), dict) else {}
    raw_output = generation.get("raw_output")
    judge = trace.get("judge") if isinstance(trace.get("judge"), dict) else {}
    merged = judge.get("merged") if isinstance(judge.get("merged"), dict) else {}
    checks = merged.get("checks") if isinstance(merged.get("checks"), dict) else {}
    compact_checks: dict[str, Any] = {}
    for name, value in checks.items():
        if not isinstance(value, dict):
            continue
        compact_checks[name] = {
            key: value.get(key)
            for key in (
                "score",
                "pass_fail",
                "reason",
                "quality_reason",
                "blocking_failure",
            )
            if key in value
        }
    result = trace.get("result") if isinstance(trace.get("result"), dict) else {}
    answerability = trace.get("answerability")
    return {
        "attempt": trace.get("attempt"),
        "accepted": result.get("accepted") is True,
        "result_reason": result.get("reason"),
        "qa_hash": qa_hash(compact_qa(qa)) if qa else None,
        "raw_output_hash": (
            hashlib.sha256(raw_output.encode("utf-8")).hexdigest()
            if isinstance(raw_output, str)
            else None
        ),
        "qa": compact_qa(qa),
        "checks": compact_checks,
        "blocking_failures": merged.get("blocking_failures"),
        "judge_feedback": merged.get("feedback_to_generator"),
        "answerability": answerability,
    }


def summarize_row(row: dict[str, Any]) -> dict[str, Any]:
    row_traces = traces(row)
    wrapper_qas: dict[Any, dict[str, Any]] = {}
    for wrapper in row.get("attempts") or []:
        if isinstance(wrapper, dict) and isinstance(wrapper.get("qa"), dict):
            wrapper_qas[wrapper.get("attempt")] = wrapper["qa"]
    attempts = [
        summarize_trace(trace, wrapper_qas.get(trace.get("attempt")))
        for trace in row_traces
    ]
    accepted = next((attempt for attempt in attempts if attempt["accepted"]), None)
    audit = row.get("human_audit") if isinstance(row.get("human_audit"), dict) else {}
    videos = []
    for value in audit.get("video_evidence") or []:
        if isinstance(value, dict):
            videos.append(
                {
                    "user": value.get("user"),
                    "agent_id": value.get("agent_id"),
                    "day": value.get("day"),
                    "time_token": value.get("time_token"),
                    "original_local_video": value.get("original_local_video"),
                    "local_video": value.get("local_video"),
                }
            )
    return {
        "evidence_id": row.get("evidence_id"),
        "status": row.get("status"),
        "attempt_count": len(attempts),
        "accepted_attempt": accepted,
        "final_attempt": attempts[-1] if attempts else None,
        "attempts": attempts,
        "videos": videos,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--without", dest="without_path", type=Path, required=True)
    parser.add_argument("--with", dest="with_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    without_rows = [summarize_row(row) for row in load_jsonl(args.without_path)]
    with_rows = [summarize_row(row) for row in load_jsonl(args.with_path)]
    without_by_id = {row["evidence_id"]: row for row in without_rows}
    with_by_id = {row["evidence_id"]: row for row in with_rows}
    ids = [row["evidence_id"] for row in without_rows]
    if ids != [row["evidence_id"] for row in with_rows]:
        raise ValueError("evidence order differs")

    comparisons = []
    for evidence_id in ids:
        without = without_by_id[evidence_id]
        with_rationale = with_by_id[evidence_id]
        without_first = without["attempts"][0] if without["attempts"] else None
        with_first = with_rationale["attempts"][0] if with_rationale["attempts"] else None
        comparisons.append(
            {
                "evidence_id": evidence_id,
                "outcome_pair": f"without_{without['status']}__with_{with_rationale['status']}",
                "first_attempt_exact_match": bool(
                    without_first
                    and with_first
                    and without_first["raw_output_hash"]
                    == with_first["raw_output_hash"]
                ),
                "without_rationale": without,
                "with_rationale": with_rationale,
            }
        )

    payload = {
        "comparison_count": len(comparisons),
        "discordant": [
            row
            for row in comparisons
            if row["without_rationale"]["status"]
            != row["with_rationale"]["status"]
        ],
        "first_attempt_exact_matches": [
            row for row in comparisons if row["first_attempt_exact_match"]
        ],
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
