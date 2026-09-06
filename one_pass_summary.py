"""六用户 one-pass 结果的固定分母汇总。"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
from typing import Any, Iterable


REJECTION_BUCKETS = (
    "parse_failed",
    "rejected_by_formality",
    "rejected_by_evidence",
    "rejected_by_answerability",
    "rejected_by_other",
)


def _rows_by_slot(rows: Iterable[dict[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        slot_id = str(row.get("generation_slot_id") or "")
        if not slot_id:
            raise ValueError(f"{label} row is missing generation_slot_id")
        if slot_id in indexed:
            raise ValueError(f"duplicate {label} row for generation_slot_id={slot_id}")
        indexed[slot_id] = row
    return indexed


def _last_attempt(row: dict[str, Any]) -> dict[str, Any]:
    attempts = row.get("attempts")
    if isinstance(attempts, list) and attempts and isinstance(attempts[-1], dict):
        return attempts[-1]
    return {}


def _review_for_row(row: dict[str, Any]) -> dict[str, Any]:
    review = row.get("review")
    if isinstance(review, dict):
        return review
    attempt = _last_attempt(row)
    qa = attempt.get("qa") if isinstance(attempt, dict) else None
    review = qa.get("review") if isinstance(qa, dict) else None
    return review if isinstance(review, dict) else {}


def classify_one_pass_rejection(row: dict[str, Any]) -> str:
    """从已落盘的最终 rejected row 推导不改变 gate 的统计分类。"""

    attempts = row.get("attempts") if isinstance(row.get("attempts"), list) else []
    attempt_text = " ".join(
        str(item.get("reason") or "")
        for item in attempts
        if isinstance(item, dict)
    ).lower()
    if "valid json" in attempt_text or "parse failed" in attempt_text:
        return "parse_failed"

    review = _review_for_row(row)
    final_decision = review.get("final_decision")
    rejection_stage = (
        str(final_decision.get("rejection_stage") or "")
        if isinstance(final_decision, dict)
        else ""
    )
    checks = review.get("judger", {}).get("checks", {})
    if not isinstance(checks, dict):
        checks = {}
    failed_checks = {
        name
        for name in ("qa_formality", "evidence_groundedness", "answerability")
        if isinstance(checks.get(name), dict)
        and str(checks[name].get("status") or "").upper() == "FAIL"
    }
    if rejection_stage == "answerability" or "answerability" in failed_checks:
        return "rejected_by_answerability"
    if rejection_stage == "judger" and "qa_formality" in failed_checks:
        return "rejected_by_formality"
    if "evidence_groundedness" in failed_checks:
        return "rejected_by_evidence"
    if "qa_formality" in failed_checks:
        return "rejected_by_formality"
    return "rejected_by_other"


def _attempt_count(row: dict[str, Any]) -> int:
    value = row.get("attempt_count")
    if isinstance(value, int) and value > 0:
        return value
    attempts = row.get("attempts")
    if isinstance(attempts, list) and attempts:
        return max(
            [
                int(item.get("attempt", 0))
                for item in attempts
                if isinstance(item, dict) and str(item.get("attempt", "")).isdigit()
            ]
            or [1]
        )
    return 1


def _timing_by_slot(prompt_rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, float]]:
    timing: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in prompt_rows:
        slot_id = str(row.get("generation_slot_id") or "")
        if not slot_id:
            continue
        try:
            elapsed = float(row.get("elapsed_seconds"))
        except (TypeError, ValueError):
            continue
        stage = str(row.get("stage") or "")
        if stage == "generation":
            timing[slot_id]["generation_seconds"] += elapsed
        elif stage in {
            "qa_formality_judge",
            "evidence_groundedness_judge",
            "answerability",
        }:
            timing[slot_id]["judge_seconds"] += elapsed
    return {slot_id: dict(values) for slot_id, values in timing.items()}


def _mean_or_none(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 3) if values else None


def update_one_pass_manifest(
    manifest_path: str | Path,
    result_path: str | Path,
    *,
    status: str,
) -> None:
    """把 one-pass 结果状态同步回同一 Job 的 manifest。"""

    manifest_file = Path(manifest_path)
    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    payload["status"] = str(status)
    payload["result_path"] = str(Path(result_path))
    manifest_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def summarize_one_pass_rows(
    *,
    evidence_rows: list[dict[str, Any]],
    accepted_rows: list[dict[str, Any]],
    rejected_rows: list[dict[str, Any]],
    prompt_rows: list[dict[str, Any]],
    attempt_rows: list[dict[str, Any]],
    expected_slot_count: int = 30,
    generation_exit_code: int = 0,
) -> dict[str, Any]:
    """按预展开 evidence 的固定 slot 统计 one-pass 结果。"""

    if expected_slot_count <= 0:
        raise ValueError("expected_slot_count must be positive")
    evidence_by_slot = _rows_by_slot(evidence_rows, label="evidence")
    if len(evidence_by_slot) != expected_slot_count:
        raise ValueError(
            f"one-pass evidence must contain {expected_slot_count} unique slots, "
            f"got {len(evidence_by_slot)}"
        )
    accepted_by_slot = _rows_by_slot(accepted_rows, label="accepted")
    rejected_by_slot = _rows_by_slot(rejected_rows, label="rejected")
    overlap = set(accepted_by_slot) & set(rejected_by_slot)
    if overlap:
        raise ValueError(f"slot appears in both accepted and rejected rows: {sorted(overlap)}")

    timing = _timing_by_slot(prompt_rows)
    status_by_slot: dict[str, str] = {}
    rejection_counts = Counter()
    attempt_counts = Counter()
    completed_rows: dict[str, dict[str, Any]] = {}
    for slot_id in evidence_by_slot:
        if slot_id in accepted_by_slot:
            status_by_slot[slot_id] = "accepted"
            completed_rows[slot_id] = accepted_by_slot[slot_id]
        elif slot_id in rejected_by_slot:
            row = rejected_by_slot[slot_id]
            bucket = classify_one_pass_rejection(row)
            status_by_slot[slot_id] = bucket
            rejection_counts[bucket] += 1
            completed_rows[slot_id] = row
        else:
            status_by_slot[slot_id] = "missing"
        if slot_id in completed_rows:
            attempt_counts[str(_attempt_count(completed_rows[slot_id]))] += 1

    missing_count = sum(status == "missing" for status in status_by_slot.values())
    accepted_count = sum(status == "accepted" for status in status_by_slot.values())
    rejected_count = sum(
        status in REJECTION_BUCKETS for status in status_by_slot.values()
    )
    generated_valid_json_count = rejected_count - rejection_counts["parse_failed"] + accepted_count
    result: dict[str, Any] = {
        "status": (
            "completed"
            if missing_count == 0 and int(generation_exit_code) == 0
            else "incomplete"
        ),
        "objective": "fixed_one_pass_30_slot_generation",
        "slot_count": expected_slot_count,
        "completed_slot_count": len(completed_rows),
        "missing_slot_count": missing_count,
        "generated_valid_json_count": generated_valid_json_count,
        "parse_failed_count": rejection_counts["parse_failed"],
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "rejected_by_formality_count": rejection_counts["rejected_by_formality"],
        "rejected_by_evidence_count": rejection_counts["rejected_by_evidence"],
        "rejected_by_answerability_count": rejection_counts["rejected_by_answerability"],
        "rejected_by_other_count": rejection_counts["rejected_by_other"],
        "acceptance_rate": accepted_count / expected_slot_count,
        "valid_generation_rate": generated_valid_json_count / expected_slot_count,
        "attempt_count_distribution": dict(attempt_counts),
        "generation_exit_code": int(generation_exit_code),
        "generation_slot_target_reached": missing_count == 0,
        "status_by_generation_slot": status_by_slot,
    }

    groups: dict[str, dict[str, Any]] = {}
    speakers: dict[str, dict[str, Any]] = {}
    for identity, destination in (("generation_group_id", groups), ("speaker_user", speakers)):
        buckets: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
        for slot_id, evidence_row in evidence_by_slot.items():
            key = str(evidence_row.get(identity) or "missing")
            buckets[key].append((slot_id, evidence_row))
        for key, rows in buckets.items():
            statuses = [status_by_slot[slot_id] for slot_id, _ in rows]
            generation_seconds = [
                timing[slot_id]["generation_seconds"]
                for slot_id, _ in rows
                if timing.get(slot_id, {}).get("generation_seconds") is not None
            ]
            destination[key] = {
                "slot_count": len(rows),
                "completed_slot_count": sum(status != "missing" for status in statuses),
                "accepted_count": statuses.count("accepted"),
                "parse_failed_count": statuses.count("parse_failed"),
                "rejected_count": sum(status in REJECTION_BUCKETS for status in statuses),
                "mean_generation_seconds": _mean_or_none(generation_seconds),
            }
    result["by_generation_group"] = groups
    result["by_speaker"] = speakers

    generation_times = [
        values["generation_seconds"]
        for values in timing.values()
        if values.get("generation_seconds") is not None
    ]
    judge_times = [
        values["judge_seconds"]
        for values in timing.values()
        if values.get("judge_seconds") is not None
    ]
    result["timing"] = {
        "generation_slot_count_with_timing": len(generation_times),
        "mean_generation_seconds": _mean_or_none(generation_times),
        "median_generation_seconds": (
            round(statistics.median(generation_times), 3) if generation_times else None
        ),
        "mean_judge_seconds": _mean_or_none(judge_times),
        "prompt_row_count": len(prompt_rows),
        "attempt_row_count": len(attempt_rows),
    }
    return result
