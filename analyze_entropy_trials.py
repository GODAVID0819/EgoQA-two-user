"""Summarize generation runs, final judges, and final-attempt entropy.

This analyzer deliberately keeps three concerns separate:

1. basic run statistics (items, generations, attempts, acceptance);
2. final-attempt PASS/FAIL/UNCERTAIN statistics for all three judges;
3. raw first-decision P/F logits and independently recalculated binary entropy for the
   two model judges. Legacy 1/2/3 run artifacts remain readable.

Only the Python standard library is required.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


JUDGE_NAMES = ("qa_formality", "evidence_groundedness", "answerability")
DIRECT_DECISION_ENTROPY_JUDGES = ("qa_formality", "evidence_groundedness")
ENTROPY_JUDGE_NAMES = (*DIRECT_DECISION_ENTROPY_JUDGES, "answerability")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value


def safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    upper_weight = position - lower
    return ordered[lower] * (1.0 - upper_weight) + ordered[upper] * upper_weight


def describe(values: Sequence[float]) -> dict[str, Any]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return {
            "count": 0,
            "mean": None,
            "standard_deviation": None,
            "minimum": None,
            "q1": None,
            "median": None,
            "q3": None,
            "maximum": None,
        }
    return {
        "count": len(clean),
        "mean": statistics.fmean(clean),
        "standard_deviation": statistics.stdev(clean) if len(clean) > 1 else 0.0,
        "minimum": min(clean),
        "q1": percentile(clean, 0.25),
        "median": statistics.median(clean),
        "q3": percentile(clean, 0.75),
        "maximum": max(clean),
    }


def full_attempt_traces(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Handle the two layouts used by accepted and rejected intermediate rows."""

    generation_trace = row.get("generation_trace")
    if isinstance(generation_trace, list) and generation_trace:
        return [value for value in generation_trace if isinstance(value, dict)]
    attempts = row.get("attempts")
    if not isinstance(attempts, list):
        return []
    return [
        value
        for value in attempts
        if isinstance(value, dict) and ("judge" in value or "generation" in value)
    ]


def final_judger(row: dict[str, Any], traces: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if traces:
        judge_trace = traces[-1].get("judge")
        if isinstance(judge_trace, dict) and isinstance(judge_trace.get("merged"), dict):
            return judge_trace["merged"]
    review = row.get("review")
    if isinstance(review, dict) and isinstance(review.get("judger"), dict):
        return review["judger"]
    return {}


def final_answerability(row: dict[str, Any], judger: dict[str, Any]) -> dict[str, Any]:
    branches = judger.get("branches")
    if isinstance(branches, dict) and isinstance(branches.get("answerability"), dict):
        return branches["answerability"]
    review = row.get("review")
    if isinstance(review, dict) and isinstance(review.get("answerability"), dict):
        return review["answerability"]
    return {}


def generated_candidate_count(traces: Sequence[dict[str, Any]]) -> int:
    count = 0
    for trace in traces:
        generation = trace.get("generation")
        if isinstance(generation, dict) and (
            generation.get("raw_output") is not None
            or generation.get("parsed_qa") is not None
            or generation.get("prompt") is not None
        ):
            count += 1
    return count


def status_of(check: Any) -> str:
    if not isinstance(check, dict):
        return "MISSING"
    status = str(check.get("status") or "MISSING").strip().upper()
    return status if status in {"PASS", "FAIL", "UNCERTAIN"} else status


def recompute_entropy(uncertainty: Any) -> dict[str, Any]:
    base = {
        "entropy_available": False,
        "entropy_unavailable_reason": "missing decision_uncertainty",
        "entropy_mode": None,
        "choice_labels": None,
        "raw_weights": None,
        "probabilities_recalculated": None,
        "weight_type": None,
        "raw_weight_p": None,
        "raw_weight_f": None,
        "raw_weight_1": None,
        "raw_weight_2": None,
        "raw_weight_3": None,
        "probability_p": None,
        "probability_f": None,
        "probability_1": None,
        "probability_2": None,
        "probability_3": None,
        "entropy_nats_recalculated": None,
        "normalized_entropy_recalculated": None,
        "argmax_choice_recalculated": None,
        "top_probability_recalculated": None,
        "top_two_probability_margin_recalculated": None,
        "stored_entropy_nats": None,
        "stored_normalized_entropy": None,
        "entropy_nats_absolute_error": None,
        "normalized_entropy_absolute_error": None,
        "stored_argmax_choice": None,
        "generated_choice": None,
        "generated_matches_recalculated_argmax": None,
        "selection_sort_key": None,
        "selection_order": "P low H > P high H > F high H > F low H",
    }
    if not isinstance(uncertainty, dict):
        return base
    if uncertainty.get("available") is not True:
        base["entropy_unavailable_reason"] = str(
            uncertainty.get("reason") or "uncertainty.available is not true"
        )
        return base
    raw = uncertainty.get("log_weights")
    if not isinstance(raw, dict):
        base["entropy_unavailable_reason"] = "missing log_weights"
        return base
    if all(proxy in raw for proxy in ("P", "F")):
        labels = ("P", "F")
        entropy_mode = "first_decision"
    elif all(choice in raw for choice in ("A", "B", "C", "D", "E")):
        labels = ("A", "B", "C", "D", "E")
        entropy_mode = "answerability_choice"
    elif all(str(score) in raw for score in (1, 2, 3)):
        labels = ("1", "2", "3")
        entropy_mode = "legacy_quality_score"
    else:
        base["entropy_unavailable_reason"] = (
            "log_weights contains neither P/F, A-E, nor legacy 1/2/3"
        )
        return base
    try:
        logits = {label: float(raw[label]) for label in labels}
    except (KeyError, TypeError, ValueError):
        base["entropy_unavailable_reason"] = "log_weights contains a non-numeric value"
        return base
    if not all(math.isfinite(value) for value in logits.values()):
        base["entropy_unavailable_reason"] = "log_weights contains a non-finite value"
        return base

    maximum = max(logits.values())
    exponentials = {score: math.exp(value - maximum) for score, value in logits.items()}
    denominator = sum(exponentials.values())
    probabilities = {score: value / denominator for score, value in exponentials.items()}
    entropy_nats = -sum(
        probability * math.log(probability)
        for probability in probabilities.values()
        if probability > 0.0
    )
    normalized_entropy = entropy_nats / math.log(float(len(labels)))
    ordered_probabilities = sorted(probabilities.values(), reverse=True)
    argmax_choice = str(max(probabilities, key=probabilities.get))
    stored_entropy = safe_float(uncertainty.get("entropy_nats"))
    stored_normalized = safe_float(uncertainty.get("normalized_entropy"))
    generated_choice = uncertainty.get(
        "generated_decision",
        uncertainty.get(
            "generated_choice",
            uncertainty.get("generated_proxy", uncertainty.get("generated_score")),
        ),
    )
    generated_choice = str(generated_choice) if generated_choice is not None else None
    stored_argmax = uncertainty.get(
        "argmax_decision",
        uncertainty.get(
            "argmax_choice",
            uncertainty.get("argmax_proxy", uncertainty.get("argmax_score")),
        ),
    )
    selection_sort_key = None
    if generated_choice in {"P", "F"} and entropy_mode == "first_decision":
        selection_sort_key = [
            0 if generated_choice == "P" else 1,
            normalized_entropy if generated_choice == "P" else -normalized_entropy,
        ]

    return {
        "entropy_available": True,
        "entropy_unavailable_reason": "",
        "entropy_mode": entropy_mode,
        "choice_labels": list(labels),
        "raw_weights": logits,
        "probabilities_recalculated": probabilities,
        "weight_type": uncertainty.get("weight_type"),
        "raw_weight_p": logits.get("P"),
        "raw_weight_f": logits.get("F"),
        "raw_weight_1": logits.get("1"),
        "raw_weight_2": logits.get("2"),
        "raw_weight_3": logits.get("3"),
        "probability_p": probabilities.get("P"),
        "probability_f": probabilities.get("F"),
        "probability_1": probabilities.get("1"),
        "probability_2": probabilities.get("2"),
        "probability_3": probabilities.get("3"),
        "entropy_nats_recalculated": entropy_nats,
        "normalized_entropy_recalculated": normalized_entropy,
        "argmax_choice_recalculated": argmax_choice,
        "top_probability_recalculated": ordered_probabilities[0],
        "top_two_probability_margin_recalculated": (
            ordered_probabilities[0] - ordered_probabilities[1]
        ),
        "stored_entropy_nats": stored_entropy,
        "stored_normalized_entropy": stored_normalized,
        "entropy_nats_absolute_error": (
            abs(entropy_nats - stored_entropy) if stored_entropy is not None else None
        ),
        "normalized_entropy_absolute_error": (
            abs(normalized_entropy - stored_normalized)
            if stored_normalized is not None
            else None
        ),
        "stored_argmax_choice": str(stored_argmax) if stored_argmax is not None else None,
        "generated_choice": generated_choice,
        "generated_matches_recalculated_argmax": (
            generated_choice == argmax_choice if generated_choice is not None else None
        ),
        "selection_sort_key": selection_sort_key,
        "selection_order": "P low H > P high H > F high H > F low H",
    }


def analyze_run(label: str, path: Path) -> dict[str, Any]:
    item_rows: list[dict[str, Any]] = []
    judge_rows: list[dict[str, Any]] = []
    entropy_rows: list[dict[str, Any]] = []

    for row_index, row in enumerate(iter_jsonl(path), 1):
        evidence_id = str(row.get("evidence_id") or f"row_{row_index:04d}")
        traces = full_attempt_traces(row)
        final_trace = traces[-1] if traces else {}
        status = str(row.get("status") or "unknown").lower()
        accepted = status in {"accepted", "passed"}
        if not accepted:
            review = row.get("review")
            accepted = isinstance(review, dict) and review.get("review_passed") is True
        judger = final_judger(row, traces)
        checks = judger.get("checks") if isinstance(judger.get("checks"), dict) else {}
        check_statuses = {name: status_of(checks.get(name)) for name in JUDGE_NAMES}
        final_result = (
            final_trace.get("result") if isinstance(final_trace.get("result"), dict) else {}
        )
        attempt_count = len(traces)
        item_rows.append(
            {
                "run": label,
                "source_path": str(path),
                "row_index": row_index,
                "evidence_id": evidence_id,
                "question_type": row.get("question_type"),
                "generation_mode": row.get("generation_mode"),
                "generator_decode_mode": (
                    row.get("generator_decode", {}).get("mode")
                    if isinstance(row.get("generator_decode"), dict)
                    else None
                ),
                "status": status,
                "accepted": accepted,
                "attempt_count": attempt_count,
                "generation_count": generated_candidate_count(traces),
                "accepted_on_first_attempt": accepted and attempt_count == 1,
                "final_attempt": final_trace.get("attempt", attempt_count or None),
                "final_result_accepted": final_result.get("accepted"),
                **{f"{name}_status": check_statuses[name] for name in JUDGE_NAMES},
                "all_three_judges_pass": all(
                    check_statuses[name] == "PASS" for name in JUDGE_NAMES
                ),
                "failed_or_uncertain_judges": "|".join(
                    name for name in JUDGE_NAMES if check_statuses[name] != "PASS"
                ),
            }
        )

        for judge_name in JUDGE_NAMES:
            judge_rows.append(
                {
                    "run": label,
                    "evidence_id": evidence_id,
                    "final_attempt": final_trace.get("attempt", attempt_count or None),
                    "final_item_accepted": accepted,
                    "judge": judge_name,
                    "status": check_statuses[judge_name],
                    "reason": (
                        checks.get(judge_name, {}).get("reason")
                        if isinstance(checks.get(judge_name), dict)
                        else None
                    ),
                }
            )

        for judge_name in DIRECT_DECISION_ENTROPY_JUDGES:
            check = checks.get(judge_name) if isinstance(checks.get(judge_name), dict) else {}
            uncertainty = None
            if isinstance(check, dict):
                uncertainty = check.get("decision_uncertainty")
                if not isinstance(uncertainty, dict):
                    uncertainty = check.get("quality_uncertainty")
            entropy_rows.append(
                {
                    "run": label,
                    "evidence_id": evidence_id,
                    "final_attempt": final_trace.get("attempt", attempt_count or None),
                    "final_item_accepted": accepted,
                    "judge": judge_name,
                    "judge_status": check_statuses[judge_name],
                    **recompute_entropy(uncertainty),
                }
            )

        answerability = final_answerability(row, judger)
        evaluations = answerability.get("evaluations")
        if isinstance(evaluations, list):
            for evaluation in evaluations:
                if not isinstance(evaluation, dict):
                    continue
                entropy_rows.append(
                    {
                        "run": label,
                        "evidence_id": evidence_id,
                        "final_attempt": final_trace.get("attempt", attempt_count or None),
                        "final_item_accepted": accepted,
                        "judge": "answerability",
                        "judge_status": str(evaluation.get("choice") or "MISSING").upper(),
                        "condition_id": evaluation.get("condition_id"),
                        "condition_type": evaluation.get("condition_type"),
                        "condition_users": "|".join(
                            str(user) for user in evaluation.get("users") or []
                        ),
                        **recompute_entropy(evaluation.get("choice_uncertainty")),
                    }
                )

    if not item_rows:
        raise ValueError(f"{path}: no JSONL records found")
    return {
        "label": label,
        "source_path": str(path),
        "item_rows": item_rows,
        "judge_rows": judge_rows,
        "entropy_rows": entropy_rows,
        "basic_summary": summarize_basics(item_rows),
        "judge_summary": summarize_judges(item_rows),
        "entropy_summary": summarize_entropy(entropy_rows, len(item_rows)),
    }


def summarize_basics(item_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total_items = len(item_rows)
    accepted_count = sum(bool(row["accepted"]) for row in item_rows)
    rejected_count = total_items - accepted_count
    attempts = [float(row["attempt_count"]) for row in item_rows]
    total_attempts = sum(int(row["attempt_count"]) for row in item_rows)
    total_generations = sum(int(row["generation_count"]) for row in item_rows)
    return {
        "total_items": total_items,
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "acceptance_rate": accepted_count / total_items,
        "rejection_rate": rejected_count / total_items,
        "total_attempts": total_attempts,
        "total_candidate_generations": total_generations,
        "average_attempts_per_item": statistics.fmean(attempts),
        "median_attempts_per_item": statistics.median(attempts),
        "minimum_attempts": int(min(attempts)),
        "maximum_attempts": int(max(attempts)),
        "accepted_on_first_attempt_count": sum(
            bool(row["accepted_on_first_attempt"]) for row in item_rows
        ),
        "accepted_on_first_attempt_rate": (
            sum(bool(row["accepted_on_first_attempt"]) for row in item_rows) / total_items
        ),
        "attempt_count_distribution": dict(
            sorted(Counter(str(row["attempt_count"]) for row in item_rows).items())
        ),
        "question_type_distribution": dict(
            Counter(str(row.get("question_type")) for row in item_rows)
        ),
        "status_distribution": dict(Counter(str(row["status"]) for row in item_rows)),
    }


def summarize_judges(item_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total_items = len(item_rows)
    result: dict[str, Any] = {}
    for judge_name in JUDGE_NAMES:
        counts = Counter(str(row[f"{judge_name}_status"]) for row in item_rows)
        result[judge_name] = {
            "counts": dict(counts),
            "pass_count": counts.get("PASS", 0),
            "fail_count": counts.get("FAIL", 0),
            "uncertain_count": counts.get("UNCERTAIN", 0),
            "missing_count": counts.get("MISSING", 0),
            "pass_rate_over_all_items": counts.get("PASS", 0) / total_items,
        }
    result["joint"] = {
        "all_three_pass_count": sum(bool(row["all_three_judges_pass"]) for row in item_rows),
        "all_three_pass_rate": (
            sum(bool(row["all_three_judges_pass"]) for row in item_rows) / total_items
        ),
        "failure_combination_counts": dict(
            Counter(
                str(row["failed_or_uncertain_judges"] or "none") for row in item_rows
            ).most_common()
        ),
    }
    return result


def summarize_entropy(
    entropy_rows: Sequence[dict[str, Any]],
    item_count: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for judge_name in ENTROPY_JUDGE_NAMES:
        rows = [row for row in entropy_rows if row["judge"] == judge_name]
        available = [row for row in rows if row["entropy_available"]]
        result[judge_name] = {
            "row_count": len(rows),
            "available_count": len(available),
            "availability_rate": len(available) / len(rows) if rows else None,
            "coverage_over_items": (
                len(available) / item_count
                if judge_name in DIRECT_DECISION_ENTROPY_JUDGES
                else None
            ),
            "entropy_mode_counts": dict(
                Counter(str(row.get("entropy_mode")) for row in available)
            ),
            "weight_type_counts": dict(
                Counter(str(row.get("weight_type")) for row in available)
            ),
            **{
                key: describe(
                    [row[key] for row in available if row.get(key) is not None]
                )
                for key in (
                    "raw_weight_p",
                    "raw_weight_f",
                    "raw_weight_1",
                    "raw_weight_2",
                    "raw_weight_3",
                    "probability_p",
                    "probability_f",
                    "probability_1",
                    "probability_2",
                    "probability_3",
                )
            },
            "entropy_nats_recalculated": describe(
                [row["entropy_nats_recalculated"] for row in available]
            ),
            "normalized_entropy_recalculated": describe(
                [row["normalized_entropy_recalculated"] for row in available]
            ),
            "argmax_choice_counts": dict(
                sorted(
                    Counter(str(row["argmax_choice_recalculated"]) for row in available).items()
                )
            ),
            "generated_argmax_match_rate": (
                sum(bool(row["generated_matches_recalculated_argmax"]) for row in available)
                / len(available)
                if available
                else None
            ),
            "maximum_entropy_nats_absolute_error": max(
                (
                    row["entropy_nats_absolute_error"]
                    for row in available
                    if row["entropy_nats_absolute_error"] is not None
                ),
                default=None,
            ),
            "maximum_normalized_entropy_absolute_error": max(
                (
                    row["normalized_entropy_absolute_error"]
                    for row in available
                    if row["normalized_entropy_absolute_error"] is not None
                ),
                default=None,
            ),
        }
    return result


def display_number(value: Any, digits: int = 6) -> str:
    number = safe_float(value)
    return "NA" if number is None else f"{number:.{digits}f}"


def run_summary_markdown(runs: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# Run and Final-Judge Summary",
        "",
        "## Basic run data",
        "",
        "`total_items` is the number of evidence packets. `total_candidate_generations` counts "
        "attempt traces containing a generator call.",
        "",
        "| Run | Items | Accepted | Rejected | Acceptance rate | Candidate generations | Total attempts | Average attempts | Median attempts | First-attempt accepts |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        summary = run["basic_summary"]
        lines.append(
            f"| {run['label']} | {summary['total_items']} | {summary['accepted_count']} | "
            f"{summary['rejected_count']} | {display_number(summary['acceptance_rate'])} | "
            f"{summary['total_candidate_generations']} | {summary['total_attempts']} | "
            f"{display_number(summary['average_attempts_per_item'])} | "
            f"{display_number(summary['median_attempts_per_item'])} | "
            f"{summary['accepted_on_first_attempt_count']} |"
        )
    lines.extend(
        [
            "",
            "## Final-attempt judge PASS/FAIL statistics",
            "",
            "Each evidence packet contributes only its final attempt to this table.",
            "",
            "| Run | Judge | PASS | FAIL | UNCERTAIN | MISSING | PASS rate |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for run in runs:
        for judge_name in JUDGE_NAMES:
            judge = run["judge_summary"][judge_name]
            lines.append(
                f"| {run['label']} | {judge_name} | {judge['pass_count']} | "
                f"{judge['fail_count']} | {judge['uncertain_count']} | "
                f"{judge['missing_count']} | {display_number(judge['pass_rate_over_all_items'])} |"
            )
    lines.extend(["", "## Joint final-attempt outcomes", ""])
    for run in runs:
        joint = run["judge_summary"]["joint"]
        lines.append(
            f"- **{run['label']}**: all three PASS for {joint['all_three_pass_count']} items "
            f"({display_number(joint['all_three_pass_rate'])}); failure combinations: "
            f"`{json.dumps(joint['failure_combination_counts'], sort_keys=True)}`"
        )
    lines.append("")
    return "\n".join(lines)


def entropy_markdown(runs: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# Final-Attempt Entropy Review",
        "",
        "`qa_formality` and `evidence_groundedness` use first-decision P/F entropy. "
        "Answerability uses per-condition A-E choice entropy when it emits a direct option; "
        "`insufficient` rows are marked unavailable.",
        "",
        "For every row below, probabilities and entropy are recalculated from the stored raw "
        "choice weights using a softmax over the stored choice set. Legacy 1/2/3 rows are "
        "labeled separately.",
        "",
        "## Raw logits",
        "",
    ]
    for run in runs:
        lines.extend(
            [
                f"### {run['label']}",
                "",
                "| Evidence ID | Condition | Accepted | Judge | Output | Mode | Weight type | Raw weights |",
                "|---|---|---:|---|---|---|---|---|",
            ]
        )
        for row in run["entropy_rows"]:
            lines.append(
                f"| {row['evidence_id']} | {row.get('condition_id') or 'NA'} | "
                f"{str(bool(row['final_item_accepted'])).lower()} | {row['judge']} | "
                f"{row['judge_status']} | {row.get('entropy_mode') or 'NA'} | "
                f"{row.get('weight_type') or 'NA'} | "
                f"`{json.dumps(row.get('raw_weights'), sort_keys=True)}` |"
            )
        lines.append("")

    lines.extend(["## Recalculated probabilities and entropy", ""])
    for run in runs:
        lines.extend(
            [
                f"### {run['label']}",
                "",
                "| Evidence ID | Condition | Judge | Probabilities | Entropy nats | Normalized entropy | Argmax | Stored entropy error |",
                "|---|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in run["entropy_rows"]:
            lines.append(
                f"| {row['evidence_id']} | {row.get('condition_id') or 'NA'} | "
                f"{row['judge']} | "
                f"`{json.dumps(row.get('probabilities_recalculated'), sort_keys=True)}` | "
                f"{display_number(row['entropy_nats_recalculated'])} | "
                f"{display_number(row['normalized_entropy_recalculated'])} | "
                f"{row['argmax_choice_recalculated'] or 'NA'} | "
                f"{display_number(row['entropy_nats_absolute_error'], 10)} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Aggregate entropy statistics",
            "",
            "| Run | Judge | Coverage | Modes | Mean H | Median H | Mean normalized H | Argmax counts | Max stored-H error |",
            "|---|---|---:|---|---:|---:|---:|---|---:|",
        ]
    )
    for run in runs:
        for judge_name in ENTROPY_JUDGE_NAMES:
            summary = run["entropy_summary"][judge_name]
            lines.append(
                f"| {run['label']} | {judge_name} | "
                f"{summary['available_count']}/{summary['row_count']} | "
                f"`{json.dumps(summary['entropy_mode_counts'], sort_keys=True)}` | "
                f"{display_number(summary['entropy_nats_recalculated']['mean'])} | "
                f"{display_number(summary['entropy_nats_recalculated']['median'])} | "
                f"{display_number(summary['normalized_entropy_recalculated']['mean'])} | "
                f"`{json.dumps(summary['argmax_choice_counts'], sort_keys=True)}` | "
                f"{display_number(summary['maximum_entropy_nats_absolute_error'], 10)} |"
            )
    lines.append("")
    return "\n".join(lines)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize run counts, final judges, and recalculated final-attempt entropy."
    )
    parser.add_argument("--greedy", type=Path, required=True)
    parser.add_argument("--sampling", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs/run_judge_entropy_review"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for path in (args.greedy, args.sampling):
        if not path.is_file():
            raise SystemExit(f"input file not found: {path}")

    runs = [
        analyze_run("greedy", args.greedy),
        analyze_run("sampling", args.sampling),
    ]
    basic_report = {
        run["label"]: {
            "source_path": run["source_path"],
            "basic_summary": run["basic_summary"],
            "final_attempt_judges": run["judge_summary"],
        }
        for run in runs
    }
    entropy_report = {
        "calculation": {
            "probabilities": (
                "softmax over stored raw weights for first P/F decisions or direct A-E "
                "answerability choices"
            ),
            "entropy_nats": "-sum(p_i * ln(p_i))",
            "normalized_entropy": "entropy_nats / ln(choice_count)",
            "legacy_compatibility": "legacy 1/2/3 rows are detected and normalized by ln(3)",
            "scope": "final attempt only",
        },
        "runs": {
            run["label"]: {
                "source_path": run["source_path"],
                "summary": run["entropy_summary"],
                "rows": run["entropy_rows"],
            }
            for run in runs
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = args.output_dir / "run_and_final_judge_summary.json"
    summary_markdown = args.output_dir / "run_and_final_judge_summary.md"
    item_csv = args.output_dir / "run_items.csv"
    judges_csv = args.output_dir / "final_attempt_judges.csv"
    entropy_json = args.output_dir / "final_attempt_entropy.json"
    entropy_markdown_path = args.output_dir / "final_attempt_entropy.md"
    entropy_csv = args.output_dir / "final_attempt_entropy.csv"

    summary_json.write_text(
        json.dumps(basic_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_markdown.write_text(run_summary_markdown(runs), encoding="utf-8")
    write_csv(item_csv, [row for run in runs for row in run["item_rows"]])
    write_csv(judges_csv, [row for run in runs for row in run["judge_rows"]])
    entropy_json.write_text(
        json.dumps(entropy_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    entropy_markdown_path.write_text(entropy_markdown(runs), encoding="utf-8")
    write_csv(entropy_csv, [row for run in runs for row in run["entropy_rows"]])

    print(f"run_summary_json={summary_json}")
    print(f"run_summary_markdown={summary_markdown}")
    print(f"run_items_csv={item_csv}")
    print(f"final_judges_csv={judges_csv}")
    print(f"entropy_json={entropy_json}")
    print(f"entropy_markdown={entropy_markdown_path}")
    print(f"entropy_csv={entropy_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
