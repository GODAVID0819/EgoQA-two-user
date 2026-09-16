"""Paired PASS/FAIL ablation for generator rationale disclosure.

Every parseable question in one intermediate JSONL file is judged twice against
the same full videos.  The two prompts differ only in whether the generated QA
payload contains ``generator_rationale``.  There is no scoring, quota, ranking,
cross-candidate memory, or dense prompt/raw-output trace.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from .io_utils import append_jsonl, iter_jsonl, write_jsonl
from .qwen3vl_runner import DEFAULT_MODEL_ID, Qwen3VLTransformersRunner, make_runner
from .schema import extract_json_object


CONDITIONS = ("without_rationale", "with_rationale")
FORMAT_REPAIR_SUFFIX = """

FORMAT REPAIR RETRY:
Your preceding response could not be parsed as the required JSON object. Re-evaluate the
same videos and question, then begin your response with `{` and return exactly these two
fields with no Markdown, preamble, or trailing commentary:
{"status":"PASS or FAIL","reason":"brief concrete visual reason"}
"""
CORE_QA_FIELDS = (
    "qa_id",
    "evidence_id",
    "question_type",
    "question",
    "options",
    "correct",
    "answer",
    "required_users",
)


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _trace_has_generation(trace: Any) -> bool:
    return isinstance(trace, dict) and isinstance(trace.get("generation"), dict)


def _canonical_attempt_traces(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Recover attempt traces from accepted and rejected intermediate layouts."""

    direct = [trace for trace in _as_list(row.get("attempts")) if _trace_has_generation(trace)]
    if direct:
        return direct

    generation_trace = [
        trace for trace in _as_list(row.get("generation_trace")) if _trace_has_generation(trace)
    ]
    if generation_trace:
        return generation_trace

    recovered: list[dict[str, Any]] = []
    for wrapper in _as_list(row.get("attempts")):
        if not isinstance(wrapper, dict):
            continue
        qa = wrapper.get("qa") if isinstance(wrapper.get("qa"), dict) else {}
        wrapper_attempt = wrapper.get("attempt")
        matches = [
            trace
            for trace in _as_list(qa.get("generation_trace"))
            if _trace_has_generation(trace) and trace.get("attempt") == wrapper_attempt
        ]
        if matches:
            recovered.append(matches[-1])
    return recovered


def _qa_from_trace(trace: dict[str, Any]) -> dict[str, Any] | None:
    generation = trace.get("generation") if isinstance(trace.get("generation"), dict) else {}
    parsed = generation.get("parsed_qa")
    if not isinstance(parsed, dict):
        raw = generation.get("raw_output")
        if not isinstance(raw, str) or not raw.strip():
            return None
        try:
            parsed = extract_json_object(raw)
        except Exception:
            return None
    qa = dict(parsed)
    normalized = generation.get("normalized_qa")
    if isinstance(normalized, dict):
        for key, value in normalized.items():
            qa.setdefault(key, value)
    question = qa.get("question")
    if not isinstance(question, str) or not question.strip():
        return None
    return qa


def _video_evidence(trace: dict[str, Any], row: dict[str, Any]) -> list[dict[str, str]]:
    media = trace.get("media") if isinstance(trace.get("media"), dict) else {}
    audit = media.get("human_audit") if isinstance(media.get("human_audit"), dict) else {}
    if not audit and isinstance(row.get("human_audit"), dict):
        audit = row["human_audit"]

    evidence_rows: list[dict[str, str]] = []
    labeled_paths: set[str] = set()
    for evidence in _as_list(audit.get("video_evidence")):
        if not isinstance(evidence, dict):
            continue
        for key in (
            "full_local_video",
            "original_local_video",
            "source_local_video",
            "local_video",
        ):
            value = evidence.get(key)
            if value:
                path = str(value)
                if path not in labeled_paths:
                    evidence_rows.append(
                        {"path": path, "user": str(evidence.get("user") or "unknown_user")}
                    )
                    labeled_paths.add(path)
                break

    for value in _as_list(media.get("full_video_paths")):
        if value and str(value) not in labeled_paths:
            evidence_rows.append({"path": str(value), "user": "unknown_user"})
            labeled_paths.add(str(value))
    return evidence_rows


def load_generated_questions(intermediate_path: str | Path) -> list[dict[str, Any]]:
    """Load only attempt 1 from each packet when it produced a parseable question."""

    path = Path(intermediate_path)
    candidates: list[dict[str, Any]] = []
    for line_number, row in enumerate(iter_jsonl(path), 1):
        traces = _canonical_attempt_traces(row)
        if not traces:
            raise ValueError(f"{path}:{line_number} contains no recoverable generation attempts")
        first_trace = next(
            (
                trace
                for trace in traces
                if str(trace.get("attempt", "")).strip() == "1"
            ),
            None,
        )
        # Older intermediate files may omit the attempt number but preserve
        # chronological trace order. In that legacy case only, the first trace
        # is attempt 1. An explicitly numbered retry never substitutes for it.
        if first_trace is None and traces[0].get("attempt") is None:
            first_trace = traces[0]
        if first_trace is None:
            continue
        qa = _qa_from_trace(first_trace)
        if qa is None:
            continue
        qa.setdefault("evidence_id", row.get("evidence_id") or first_trace.get("evidence_id"))
        qa.setdefault("question_type", row.get("question_type") or first_trace.get("question_type"))
        result = first_trace.get("result") if isinstance(first_trace.get("result"), dict) else {}
        candidates.append(
            {
                "candidate_id": f"Q{len(candidates) + 1:06d}",
                "source_file": str(path),
                "source_line": line_number,
                "source_row_status": row.get("status"),
                "attempt_number": first_trace.get("attempt"),
                "prior_attempt_accepted": result.get("accepted") is True,
                "qa": qa,
                "video_evidence": _video_evidence(first_trace, row),
            }
        )
    return candidates


def _resolve_video_path(value: str, media_root: str | Path | None) -> str:
    path = Path(value)
    candidates = [path]
    if media_root is not None and not path.is_absolute():
        candidates.insert(0, Path(media_root) / path)
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"full video not found; checked: {checked}")


def compact_qa_payload(candidate: dict[str, Any], *, include_rationale: bool) -> dict[str, Any]:
    qa = candidate.get("qa") if isinstance(candidate.get("qa"), dict) else {}
    payload = {key: qa[key] for key in CORE_QA_FIELDS if key in qa}
    if include_rationale:
        payload["generator_rationale"] = qa.get("generator_rationale")
    return payload


def build_judge_prompt(candidate: dict[str, Any], *, include_rationale: bool) -> str:
    """Build the shared binary rubric; only the candidate payload is ablated."""

    payload = compact_qa_payload(candidate, include_rationale=include_rationale)
    media_order = [
        {"video": index, "user": evidence.get("user")}
        for index, evidence in enumerate(candidate.get("video_evidence", []), 1)
    ]
    return f"""You are an independent evidence-groundedness judge for a generated two-user egocentric-video question.

Return only PASS or FAIL. Do not score, rank, compare with other questions, enforce a quota, or use information from any previous candidate.

Judge the generated question against the FULL ORIGINAL VIDEOS:
- PASS only if every material visual claim in the question and declared correct answer is visibly supported, and the two views form the claimed coherent situation.
- FAIL if an object, action, state, location, person, or event is missing, ambiguous, unrelated, or visually misidentified.
- Inspect the nouns in the question independently. A fluent question can still be wrong: for example, a generator may call a bowl of dough a bowl of chips. That must FAIL even if the rest of the question sounds plausible.
- The answer options and declared answer are generator claims, not evidence. Verify them from the videos.
- If a generator_rationale field is present, it was written by the same fallible generator. It may explain intent but must never be treated as visual evidence or used to excuse a mismatch.
- Do not infer from filenames, prior pipeline decisions, outside knowledge, or common sense.
- If a material claim cannot be verified confidently, FAIL.

Video order and user labels:
{json.dumps(media_order, ensure_ascii=False, indent=2)}

Generated question-answer payload:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Return exactly one JSON object and no other text:
{{"status":"PASS or FAIL","reason":"brief concrete visual reason"}}
"""


def build_judge_content(
    candidate: dict[str, Any],
    *,
    include_rationale: bool,
    media_root: str | Path | None,
    max_image_pixels: int,
    fps: float,
) -> tuple[list[dict[str, Any]], str]:
    evidence_rows = candidate.get("video_evidence") or []
    if not evidence_rows:
        raise ValueError(f"{candidate['candidate_id']} has no full-video evidence")
    content: list[dict[str, Any]] = []
    for index, evidence in enumerate(evidence_rows, 1):
        path = _resolve_video_path(str(evidence["path"]), media_root)
        content.append(
            {
                "type": "text",
                "text": f"FULL ORIGINAL VIDEO {index}: user={evidence.get('user') or 'unknown_user'}",
            }
        )
        content.append(
            {
                "type": "video",
                "video": path,
                "max_pixels": max_image_pixels,
                "fps": fps,
            }
        )
    prompt = build_judge_prompt(candidate, include_rationale=include_rationale)
    content.append({"type": "text", "text": prompt})
    return content, prompt


def parse_pass_fail(raw_output: str) -> dict[str, str]:
    parsed = extract_json_object(raw_output)
    extra_fields = set(parsed) - {"status", "reason"}
    if extra_fields:
        raise ValueError(
            "judge output must contain only status and reason; "
            f"unexpected fields: {sorted(extra_fields)}"
        )
    status = str(parsed.get("status") or "").strip().upper()
    if status not in {"PASS", "FAIL"}:
        raise ValueError(f"judge status must be PASS or FAIL, got {status!r}")
    reason = str(parsed.get("reason") or "").strip()
    if not reason:
        raise ValueError("judge reason must be non-empty")
    return {"status": status, "reason": reason}


def content_with_format_repair(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    repaired = [dict(item) for item in content]
    if not repaired or repaired[-1].get("type") != "text":
        raise ValueError("judge content must end with the text prompt")
    repaired[-1]["text"] = str(repaired[-1].get("text") or "") + FORMAT_REPAIR_SUFFIX
    return repaired


def generate_pass_fail_with_retries(
    generate_content: Any,
    content: list[dict[str, Any]],
    *,
    candidate_id: str,
    condition: str,
    max_format_attempts: int,
) -> tuple[dict[str, str], int]:
    """Retry only malformed judge serialization; never retain invalid raw outputs."""

    if max_format_attempts < 1:
        raise ValueError("max_format_attempts must be positive")
    active_content = content
    for format_attempt in range(1, max_format_attempts + 1):
        raw_output = generate_content(active_content)
        try:
            return parse_pass_fail(raw_output), format_attempt
        except ValueError as exc:
            response_chars = len(str(raw_output or ""))
            print(
                "rationale_ablation_format_retry "
                f"candidate_id={candidate_id} condition={condition} "
                f"format_attempt={format_attempt} response_chars={response_chars} "
                f"error_type={type(exc).__name__}",
                flush=True,
            )
            if format_attempt >= max_format_attempts:
                raise RuntimeError(
                    "paired judge exhausted JSON format attempts: "
                    f"candidate_id={candidate_id} condition={condition} "
                    f"attempts={max_format_attempts} response_chars={response_chars}"
                ) from exc
            active_content = content_with_format_repair(content)
    raise AssertionError("unreachable format-retry state")


def condition_order(candidate_index: int, *, seed: int) -> tuple[str, str]:
    """Alternate first condition while allowing a reproducible parity flip."""

    if (candidate_index + seed) % 2:
        return CONDITIONS
    return tuple(reversed(CONDITIONS))


def status_pair(without_rationale: str, with_rationale: str) -> str:
    return f"without_{without_rationale}__with_{with_rationale}"


def summarize_results(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    pair_counts = {
        "without_PASS__with_PASS": 0,
        "without_PASS__with_FAIL": 0,
        "without_FAIL__with_PASS": 0,
        "without_FAIL__with_FAIL": 0,
    }
    total = 0
    actual_judge_calls = 0
    for row in rows:
        pair = row.get("status_pair")
        if pair not in pair_counts:
            raise ValueError(f"invalid or missing status_pair: {pair!r}")
        pair_counts[pair] += 1
        total += 1
        format_attempts = row.get("format_attempts")
        if isinstance(format_attempts, dict):
            actual_judge_calls += sum(
                max(1, int(format_attempts.get(condition) or 1))
                for condition in CONDITIONS
            )
        else:
            actual_judge_calls += 2
    return {
        "question_count": total,
        "judge_decision_count": total * 2,
        "judge_call_count": actual_judge_calls,
        "format_retry_count": actual_judge_calls - (total * 2),
        "status_pairs": pair_counts,
        "changed_when_rationale_removed": (
            pair_counts["without_PASS__with_FAIL"]
            + pair_counts["without_FAIL__with_PASS"]
        ),
        "removal_changed_pass_to_fail": pair_counts["without_FAIL__with_PASS"],
        "removal_changed_fail_to_pass": pair_counts["without_PASS__with_FAIL"],
    }


def summarize_baseline_packets(
    intermediate_path: str | Path,
    *,
    packet_count: int = 50,
) -> dict[str, Any]:
    rows = list(iter_jsonl(intermediate_path))[:packet_count]
    if len(rows) != packet_count:
        raise ValueError(
            f"baseline intermediate has {len(rows)} packets; expected exactly {packet_count}"
        )
    accepted = sum(row.get("status") == "accepted" for row in rows)
    attempt_counts = [len(_canonical_attempt_traces(row)) for row in rows]
    return {
        "packet_count": len(rows),
        "accepted_count": accepted,
        "rejected_count": len(rows) - accepted,
        "acceptance_rate": round(accepted / len(rows), 6),
        "total_generation_attempts": sum(attempt_counts),
        "average_generation_attempts_per_packet": round(
            sum(attempt_counts) / len(rows),
            6,
        ),
        "evidence_ids": [row.get("evidence_id") for row in rows],
    }


def summarize_production_outputs(
    accepted_path: str | Path,
    rejected_path: str | Path,
    *,
    packet_count: int = 50,
) -> dict[str, Any]:
    accepted_rows = list(iter_jsonl(accepted_path)) if Path(accepted_path).exists() else []
    rejected_rows = list(iter_jsonl(rejected_path)) if Path(rejected_path).exists() else []
    observed = len(accepted_rows) + len(rejected_rows)
    if observed != packet_count:
        raise ValueError(
            f"production outputs contain {observed} packets; expected exactly {packet_count}"
        )
    attempt_counts = [
        max(1, int(row.get("attempt_count") or len(_canonical_attempt_traces(row)) or 1))
        for row in accepted_rows
    ]
    attempt_counts.extend(
        max(1, int(row.get("attempt_count") or len(_canonical_attempt_traces(row)) or 1))
        for row in rejected_rows
    )
    accepted = len(accepted_rows)
    return {
        "packet_count": observed,
        "accepted_count": accepted,
        "rejected_count": len(rejected_rows),
        "acceptance_rate": round(accepted / observed, 6),
        "total_generation_attempts": sum(attempt_counts),
        "average_generation_attempts_per_packet": round(
            sum(attempt_counts) / observed,
            6,
        ),
        "evidence_ids": [
            row.get("evidence_id") for row in [*accepted_rows, *rejected_rows]
        ],
    }


def compact_production_outputs(
    accepted_path: str | Path,
    rejected_path: str | Path,
    *,
    generator_rationale_included: bool | None = None,
    pass_fail_only: bool | None = None,
    quality_quota: int | None = None,
    pass_fail_entropy_logits: str | None = None,
) -> dict[str, int]:
    """Remove dense prior-attempt traces while retaining resume/aggregate metadata."""

    counts: dict[str, int] = {}
    for label, path_value in (("accepted", accepted_path), ("rejected", rejected_path)):
        path = Path(path_value)
        rows = list(iter_jsonl(path)) if path.exists() else []
        compact_rows = []
        for row in rows:
            traces = _canonical_attempt_traces(row)
            judge_configs = [
                trace.get("judge")
                for trace in traces
                if isinstance(trace.get("judge"), dict)
            ]
            existing_config = (
                row.get("production_judge_config")
                if isinstance(row.get("production_judge_config"), dict)
                else {}
            )

            def observed_config(key: str) -> Any:
                values = {config.get(key) for config in judge_configs}
                return next(iter(values)) if len(values) == 1 else None

            compact = dict(row)
            compact["attempt_count"] = max(
                1,
                int(row.get("attempt_count") or len(traces) or 1),
            )
            production_judge_config = {
                "generator_rationale_included": (
                    generator_rationale_included
                    if generator_rationale_included is not None
                    else (
                        observed_config("generator_rationale_included")
                        if judge_configs
                        else existing_config.get("generator_rationale_included")
                    )
                ),
                "pass_fail_only": (
                    pass_fail_only
                    if pass_fail_only is not None
                    else (
                        observed_config("pass_fail_only")
                        if judge_configs
                        else existing_config.get("pass_fail_only")
                    )
                ),
            }
            observed_quota_values = {
                (config.get("quality_quota") or {}).get("limit_per_judge_category")
                for config in judge_configs
                if isinstance(config.get("quality_quota"), dict)
            }
            observed_quota = (
                next(iter(observed_quota_values))
                if len(observed_quota_values) == 1
                else existing_config.get("quality_quota")
            )
            effective_quota = quality_quota if quality_quota is not None else observed_quota
            if effective_quota is not None:
                production_judge_config["quality_quota"] = int(effective_quota)
            observed_entropy_contract = (
                observed_config("pass_fail_entropy_logits")
                if judge_configs
                else existing_config.get("pass_fail_entropy_logits")
            )
            effective_entropy_contract = (
                pass_fail_entropy_logits
                if pass_fail_entropy_logits is not None
                else observed_entropy_contract
            )
            if effective_entropy_contract is not None:
                production_judge_config["pass_fail_entropy_logits"] = effective_entropy_contract
            observed_count_snapshots = [
                (config.get("quality_quota") or {}).get(
                    "observed_three_point_assignments_after_candidate"
                )
                for config in judge_configs
                if isinstance(config.get("quality_quota"), dict)
                and isinstance(
                    (config.get("quality_quota") or {}).get(
                        "observed_three_point_assignments_after_candidate"
                    ),
                    dict,
                )
            ]
            compact_observed_counts = existing_config.get(
                "observed_three_point_assignments"
            )
            if observed_count_snapshots:
                check_names = {
                    str(check_name)
                    for snapshot in observed_count_snapshots
                    for check_name in snapshot
                }
                compact_observed_counts = {
                    check_name: max(
                        int(snapshot.get(check_name, 0))
                        for snapshot in observed_count_snapshots
                    )
                    for check_name in check_names
                }
            if isinstance(compact_observed_counts, dict):
                production_judge_config["observed_three_point_assignments"] = dict(
                    compact_observed_counts
                )
            compact["production_judge_config"] = production_judge_config
            compact.pop("attempts", None)
            compact.pop("generation_trace", None)
            compact_rows.append(compact)
        write_jsonl(path, compact_rows)
        counts[f"{label}_rows"] = len(compact_rows)
    return counts


def compare_acceptance_summaries(
    baseline: dict[str, Any],
    no_rationale_production: dict[str, Any],
    *,
    production_label: str = "production_without_rationale",
) -> dict[str, Any]:
    baseline_rate = float(baseline["acceptance_rate"])
    production_rate = float(no_rationale_production["acceptance_rate"])
    delta = round(production_rate - baseline_rate, 6)
    direction = "higher" if delta > 0 else "lower" if delta < 0 else "no_change"
    baseline_compact = {key: value for key, value in baseline.items() if key != "evidence_ids"}
    production_compact = {
        key: value for key, value in no_rationale_production.items() if key != "evidence_ids"
    }
    result = {
        "baseline_with_rationale": baseline_compact,
        production_label: production_compact,
        "acceptance_rate_delta": delta,
        "acceptance_rate_delta_percentage_points": round(delta * 100, 3),
        "acceptance_direction": direction,
        "same_evidence_id_set": (
            set(baseline.get("evidence_ids") or [])
            == set(no_rationale_production.get("evidence_ids") or [])
        ),
        "interpretation_note": (
            "This is an end-to-end production comparison on the same evidence packets. "
            "Questions are freshly sampled, so the acceptance-rate change is not a paired "
            "causal estimate of any one prompt change by itself."
        ),
    }
    if production_label == "production_without_rationale":
        result["no_rationale_acceptance_direction"] = direction
    return result


def run_paired_rationale_ablation(
    intermediate_path: str | Path,
    *,
    output_path: str | Path,
    runner: Any,
    media_root: str | Path | None = None,
    max_image_pixels: int = 262144,
    fps: float = 1.0,
    seed: int = 1729,
    max_format_attempts: int = 3,
    resume: bool = False,
    max_items: int | None = None,
) -> dict[str, Any]:
    candidates = load_generated_questions(intermediate_path)
    if max_items is not None:
        if max_items < 1:
            raise ValueError("max_items must be positive")
        candidates = candidates[:max_items]
    if not candidates:
        raise ValueError("intermediate file contains no parseable generated questions")
    if max_format_attempts < 1:
        raise ValueError("max_format_attempts must be positive")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = list(iter_jsonl(output)) if resume and output.exists() else []
    if not resume:
        output.write_text("", encoding="utf-8")
    completed_ids = {str(row.get("candidate_id")) for row in existing_rows}
    if len(completed_ids) != len(existing_rows):
        raise ValueError("resume output contains duplicate candidate IDs")
    expected_ids = {candidate["candidate_id"] for candidate in candidates}
    unexpected = completed_ids - expected_ids
    if unexpected:
        raise ValueError(f"resume output contains candidate IDs not present in input: {sorted(unexpected)}")
    candidates_by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    for row in existing_rows:
        candidate = candidates_by_id[str(row.get("candidate_id"))]
        qa = candidate["qa"]
        if row.get("qa_id") != qa.get("qa_id") or row.get("question") != qa.get("question"):
            raise ValueError(
                f"resume output does not match current input for {candidate['candidate_id']}"
            )

    generate_content = getattr(runner, "generate_content", None)
    if not callable(generate_content):
        raise TypeError("paired rationale ablation requires a runner with generate_content()")

    for candidate_index, candidate in enumerate(candidates, 1):
        candidate_id = candidate["candidate_id"]
        if candidate_id in completed_ids:
            print(f"rationale_ablation_resume_skip candidate_id={candidate_id}", flush=True)
            continue
        order = condition_order(candidate_index, seed=seed)
        judgments: dict[str, dict[str, str]] = {}
        format_attempts: dict[str, int] = {}
        for condition in order:
            include_rationale = condition == "with_rationale"
            content, _ = build_judge_content(
                candidate,
                include_rationale=include_rationale,
                media_root=media_root,
                max_image_pixels=max_image_pixels,
                fps=fps,
            )
            print(
                "rationale_ablation_judge_start "
                f"candidate_id={candidate_id} condition={condition}",
                flush=True,
            )
            judgments[condition], format_attempts[condition] = generate_pass_fail_with_retries(
                generate_content,
                content,
                candidate_id=candidate_id,
                condition=condition,
                max_format_attempts=max_format_attempts,
            )
            print(
                "rationale_ablation_judge_done "
                f"candidate_id={candidate_id} condition={condition} "
                f"status={judgments[condition]['status']} "
                f"format_attempts={format_attempts[condition]}",
                flush=True,
            )

        qa = candidate["qa"]
        without_status = judgments["without_rationale"]["status"]
        with_status = judgments["with_rationale"]["status"]
        row = {
            "candidate_id": candidate_id,
            "source_file": candidate["source_file"],
            "source_line": candidate["source_line"],
            "source_row_status": candidate.get("source_row_status"),
            "attempt_number": candidate.get("attempt_number"),
            "prior_attempt_accepted": candidate.get("prior_attempt_accepted"),
            "qa_id": qa.get("qa_id"),
            "evidence_id": qa.get("evidence_id"),
            "question": qa.get("question"),
            "options": qa.get("options"),
            "correct": qa.get("correct"),
            "answer": qa.get("answer"),
            "generator_rationale": qa.get("generator_rationale"),
            "video_count": len(candidate.get("video_evidence") or []),
            "condition_order": list(order),
            "format_attempts": format_attempts,
            "without_rationale": judgments["without_rationale"],
            "with_rationale": judgments["with_rationale"],
            "status_pair": status_pair(without_status, with_status),
        }
        append_jsonl(output, row)
        existing_rows.append(row)

    summary = summarize_results(existing_rows)
    summary.update(
        {
            "model_id": getattr(runner, "model_id", None),
            "output_path": str(output),
            "seed": seed,
            "max_format_attempts": max_format_attempts,
            "scoring": False,
            "quota": False,
        }
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Judge each parseable attempt-1 question twice: with and without generator rationale"
        )
    )
    parser.add_argument("--intermediate", required=True, help="One qa_mcq.intermediate.jsonl file")
    parser.add_argument("--output", required=True, help="Compact paired PASS/FAIL JSONL")
    parser.add_argument("--media-root", help="Resolve relative full-video paths below this directory")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-image-pixels", type=int, default=262144)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--min-free-gib", type=float, default=12.0)
    parser.add_argument("--max-input-tokens", type=int, default=245760)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--max-format-attempts",
        type=int,
        default=3,
        help="Maximum judge calls per condition when repairing malformed JSON output",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--max-items",
        type=int,
        help="Optional smoke-test limit; default judges all parseable attempt-1 questions",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runner = make_runner(
        "transformers-local",
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        max_image_pixels=args.max_image_pixels,
        dtype=args.dtype,
        allow_cpu=args.allow_cpu,
        disable_thinking=args.disable_thinking,
        video_fps=args.fps,
        max_input_tokens=args.max_input_tokens,
        min_free_gib=args.min_free_gib,
        device_map=args.device_map,
    )
    if not isinstance(runner, Qwen3VLTransformersRunner):
        raise TypeError("paired rationale ablation requires the local Qwen Transformers runner")
    summary = run_paired_rationale_ablation(
        args.intermediate,
        output_path=args.output,
        runner=runner,
        media_root=args.media_root,
        max_image_pixels=args.max_image_pixels,
        fps=args.fps,
        seed=args.seed,
        max_format_attempts=args.max_format_attempts,
        resume=args.resume,
        max_items=args.max_items,
    )
    print("rationale_ablation_summary=" + json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
