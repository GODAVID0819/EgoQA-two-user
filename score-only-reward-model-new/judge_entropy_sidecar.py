"""Detailed first-verdict pass/fail entropy sidecar.

The sidecar reuses stored judge prompts and media from generation traces, but
does not reuse the generated judge response. It preserves the detailed output
contract while replacing the leading review_passed boolean with an authoritative
lowercase pass/fail verdict. Logits are measured at that first field. The
original production judge remains the reference and the sidecar cannot affect
production acceptance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from .io_utils import append_jsonl, iter_jsonl, write_json, write_jsonl
from .prompts import build_judge_first_verdict_sidecar_prompt
from .qwen3vl_runner import DEFAULT_MODEL_ID, MEMORY_SAFE_BACKEND, make_runner
from .video_qa_loop import (
    FIRST_VERDICT_ENTROPY_VERSION,
    decision_uncertainty_from_choice_logits,
    run_first_verdict_entropy_sidecar_call,
)


PROBE_JUDGES = ("qa_formality", "evidence_groundedness")
CALIBRATION_BUCKETS = {0, 1}
CALIBRATION_BUCKET_COUNT = 5


def full_attempt_traces(row: dict[str, Any]) -> list[dict[str, Any]]:
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


def normalized_status(value: Any) -> str | None:
    status = str(value or "").strip().upper()
    return status if status in {"PASS", "FAIL"} else None


def model_status_from_entry(entry: Any, judge_name: str) -> str | None:
    if not isinstance(entry, dict):
        return None
    parsed = entry.get("parsed")
    if not isinstance(parsed, dict):
        return None
    checks = parsed.get("checks")
    check = checks.get(judge_name) if isinstance(checks, dict) else None
    return normalized_status(check.get("status")) if isinstance(check, dict) else None


def effective_status_from_trace(judge_trace: Any, judge_name: str) -> str | None:
    if not isinstance(judge_trace, dict):
        return None
    merged = judge_trace.get("merged")
    checks = merged.get("checks") if isinstance(merged, dict) else None
    check = checks.get(judge_name) if isinstance(checks, dict) else None
    return normalized_status(check.get("status")) if isinstance(check, dict) else None


def legacy_nested_status_entropy(
    judge_trace: Any,
    judge_name: str,
) -> dict[str, Any]:
    """Read the old, verdict-leaked status entropy as a paired control only."""

    if not isinstance(judge_trace, dict):
        return {}
    merged = judge_trace.get("merged")
    checks = merged.get("checks") if isinstance(merged, dict) else None
    check = checks.get(judge_name) if isinstance(checks, dict) else None
    uncertainty = (
        check.get("decision_uncertainty")
        if isinstance(check, dict)
        else None
    )
    if not isinstance(uncertainty, dict) or uncertainty.get("available") is not True:
        return {}
    normalized_entropy = uncertainty.get("normalized_entropy")
    entropy_nats = uncertainty.get("entropy_nats")
    if not isinstance(normalized_entropy, (int, float)):
        return {}
    return {
        "legacy_nested_status_normalized_entropy": float(normalized_entropy),
        "legacy_nested_status_entropy_nats": (
            float(entropy_nats)
            if isinstance(entropy_nats, (int, float))
            else None
        ),
        "legacy_nested_status_token_index": uncertainty.get("token_index"),
        "legacy_nested_status_measurement_valid": False,
    }


def _list_of_paths(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(path) for path in value if str(path or "").strip()]


def judge_media_from_trace(
    trace: dict[str, Any],
    judge_name: str,
) -> tuple[list[str], list[str], str]:
    if judge_name == "qa_formality":
        return [], [], "text_only"
    media = trace.get("media")
    if not isinstance(media, dict):
        return [], [], "missing"
    image_paths = _list_of_paths(
        media.get("judge_image_paths", media.get("full_image_paths"))
    )
    video_paths = _list_of_paths(
        media.get("judge_video_paths", media.get("full_video_paths"))
    )
    media_role = str(media.get("judge_media_role") or "full")
    return image_paths, video_paths, media_role


def _stable_digest(*parts: Any) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def extract_probe_tasks(
    inputs: Sequence[str | Path],
    *,
    final_attempt_only: bool = False,
) -> list[dict[str, Any]]:
    """Extract independent judge calls from accepted and rejected trace layouts."""

    tasks = []
    seen = set()
    for input_path_value in inputs:
        input_path = Path(input_path_value)
        for row_index, row in enumerate(iter_jsonl(input_path), 1):
            traces = full_attempt_traces(row)
            judge_trace_indices = [
                index
                for index, trace in enumerate(traces)
                if isinstance(trace.get("judge"), dict)
            ]
            if final_attempt_only and judge_trace_indices:
                judge_trace_indices = [judge_trace_indices[-1]]
            for trace_index in judge_trace_indices:
                trace = traces[trace_index]
                judge_trace = trace.get("judge") or {}
                for judge_name in PROBE_JUDGES:
                    entry = judge_trace.get(judge_name)
                    if not isinstance(entry, dict):
                        continue
                    review_prompt = entry.get("prompt")
                    model_status = model_status_from_entry(entry, judge_name)
                    if not isinstance(review_prompt, str) or not review_prompt.strip():
                        continue
                    if model_status is None:
                        continue
                    try:
                        probe_prompt = build_judge_first_verdict_sidecar_prompt(
                            review_prompt,
                            judge_name,
                        )
                    except ValueError:
                        continue
                    evidence_id = str(
                        trace.get("evidence_id")
                        or row.get("evidence_id")
                        or f"row_{row_index:06d}"
                    )
                    qa_id = str(trace.get("qa_id") or "")
                    attempt = int(trace.get("attempt") or trace_index + 1)
                    task_key = _stable_digest(
                        input_path.resolve(),
                        row_index,
                        trace_index,
                        judge_name,
                        evidence_id,
                        qa_id,
                        attempt,
                    )
                    if task_key in seen:
                        continue
                    seen.add(task_key)
                    image_paths, video_paths, media_role = judge_media_from_trace(
                        trace,
                        judge_name,
                    )
                    effective_status = effective_status_from_trace(
                        judge_trace,
                        judge_name,
                    )
                    legacy_entropy = legacy_nested_status_entropy(
                        judge_trace,
                        judge_name,
                    )
                    tasks.append(
                        {
                            "task_key": task_key,
                            "source_path": str(input_path),
                            "source_row": row_index,
                            "source_trace_index": trace_index,
                            "evidence_id": evidence_id,
                            "qa_id": qa_id,
                            "attempt": attempt,
                            "is_final_attempt": (
                                bool(judge_trace_indices)
                                and trace_index == judge_trace_indices[-1]
                            ),
                            "item_status": str(row.get("status") or ""),
                            "judge": judge_name,
                            "official_model_status": model_status,
                            "official_effective_status": effective_status or model_status,
                            "effective_status_overrode_model": bool(
                                effective_status and effective_status != model_status
                            ),
                            "review_prompt_sha256": _stable_digest(review_prompt),
                            "probe_prompt_sha256": _stable_digest(probe_prompt),
                            "probe_prompt": probe_prompt,
                            "image_paths": image_paths,
                            "video_paths": video_paths,
                            "media_role": media_role,
                            **legacy_entropy,
                        }
                    )
    return tasks


def select_probe_tasks(
    tasks: Sequence[dict[str, Any]],
    *,
    balance_statuses: bool,
    max_per_status_per_judge: int | None,
    random_seed: int,
) -> list[dict[str, Any]]:
    """Select a deterministic, optionally balanced all-attempt cohort."""

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        grouped[(str(task["judge"]), str(task["official_model_status"]))].append(task)
    for key, rows in grouped.items():
        rows.sort(key=lambda row: _stable_digest(random_seed, key, row["task_key"]))

    selected = []
    for judge_name in PROBE_JUDGES:
        pass_rows = grouped.get((judge_name, "PASS"), [])
        fail_rows = grouped.get((judge_name, "FAIL"), [])
        if balance_statuses:
            per_status = min(len(pass_rows), len(fail_rows))
            if max_per_status_per_judge is not None:
                per_status = min(per_status, max_per_status_per_judge)
            selected.extend(pass_rows[:per_status])
            selected.extend(fail_rows[:per_status])
        else:
            for rows in (pass_rows, fail_rows):
                limit = (
                    len(rows)
                    if max_per_status_per_judge is None
                    else min(len(rows), max_per_status_per_judge)
                )
                selected.extend(rows[:limit])
    selected.sort(
        key=lambda row: (
            str(row["judge"]),
            str(row["official_model_status"]),
            _stable_digest(random_seed, row["task_key"]),
        )
    )
    return selected


def missing_media(task: dict[str, Any]) -> list[str]:
    return [
        path
        for path in (*task.get("image_paths", []), *task.get("video_paths", []))
        if not Path(path).is_file()
    ]


def verify_runner_lowercase_verdict_tokens(runner: Any) -> dict[str, Any]:
    """Verify local tokenizer support before spending inference on the cohort."""

    processor = getattr(runner, "processor", None)
    tokenizer = getattr(processor, "tokenizer", processor)
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        return {
            "checked": False,
            "reason": "runner does not expose a local tokenizer; response-time capture will validate choices",
        }
    choices = {}
    for choice in ("pass", "fail"):
        token_ids = [int(value) for value in encode(choice, add_special_tokens=False)]
        leading_space_ids = [
            int(value)
            for value in encode(f" {choice}", add_special_tokens=False)
        ]
        choices[choice] = {
            "token_ids": token_ids,
            "single_token": len(token_ids) == 1,
            "leading_space_token_ids": leading_space_ids,
            "leading_space_single_token": len(leading_space_ids) == 1,
        }
    if not all(value["single_token"] for value in choices.values()):
        raise RuntimeError(
            "lowercase pass and fail must each be one tokenizer token: "
            + json.dumps(choices, sort_keys=True)
        )
    return {
        "checked": True,
        "tokenizer_class": type(tokenizer).__name__,
        "model_id": getattr(runner, "model_id", None),
        "choices": choices,
    }


def run_probe_task(task: dict[str, Any], runner: Any) -> dict[str, Any]:
    probe = run_first_verdict_entropy_sidecar_call(
        runner=runner,
        prompt=str(task["probe_prompt"]),
        image_paths=list(task.get("image_paths") or []),
        video_paths=list(task.get("video_paths") or []),
        check_name=str(task["judge"]),
    )
    signal = probe.get("choice_logit_signal")
    uncertainty = decision_uncertainty_from_choice_logits(signal)
    probabilities = uncertainty.get("probabilities") if isinstance(uncertainty, dict) else {}
    raw_weights = uncertainty.get("log_weights") if isinstance(uncertainty, dict) else {}
    sidecar_verdict = str(probe.get("parsed_verdict") or "").strip()
    generated_decision = normalized_status(sidecar_verdict)
    model_status = str(task["official_model_status"])
    effective_status = str(task["official_effective_status"])
    return {
        **{key: value for key, value in task.items() if key != "probe_prompt"},
        "probe_version": FIRST_VERDICT_ENTROPY_VERSION,
        "measurement_context": "authoritative_first_detailed_judge_verdict",
        "independent_from_acceptance_gate": True,
        "sidecar_verdict_is_authoritative": True,
        "probe_available": bool(uncertainty.get("available")),
        "probe_unavailable_reason": str(uncertainty.get("reason") or ""),
        "sidecar_verdict": sidecar_verdict or None,
        "sidecar_verdict_status": generated_decision,
        "probe_matches_model_status": generated_decision == model_status,
        "probe_matches_effective_status": generated_decision == effective_status,
        "sidecar_nested_check_status": probe.get("nested_check_status"),
        "sidecar_verdict_matches_nested_status": probe.get(
            "verdict_matches_nested_status"
        ),
        "raw_weight_pass": (raw_weights or {}).get("PASS"),
        "raw_weight_fail": (raw_weights or {}).get("FAIL"),
        "probability_pass": (probabilities or {}).get("PASS"),
        "probability_fail": (probabilities or {}).get("FAIL"),
        "entropy_nats": uncertainty.get("entropy_nats"),
        "normalized_entropy": uncertainty.get("normalized_entropy"),
        "entropy_change_from_legacy_nested_status": (
            float(uncertainty["normalized_entropy"])
            - float(task["legacy_nested_status_normalized_entropy"])
            if isinstance(uncertainty.get("normalized_entropy"), (int, float))
            and isinstance(
                task.get("legacy_nested_status_normalized_entropy"),
                (int, float),
            )
            else None
        ),
        "token_index": uncertainty.get("token_index"),
        "field_name": (signal or {}).get("field_name"),
        "prior_generated_verdict": (signal or {}).get("prior_generated_verdict"),
        "probe_output_contract_valid": (signal or {}).get(
            "probe_output_contract_valid"
        ),
        "probe_prompt_sha256": (signal or {}).get(
            "probe_prompt_sha256",
            task.get("probe_prompt_sha256"),
        ),
        "calibration_split": calibration_split(task),
        "sidecar_detailed_output": probe.get("raw_output"),
    }


def calibration_split(row: dict[str, Any]) -> str:
    # Keep all attempts from one evidence packet in the same split.
    bucket = int(
        _stable_digest(row.get("judge"), row.get("evidence_id"))[:8],
        16,
    ) % CALIBRATION_BUCKET_COUNT
    return "calibration" if bucket in CALIBRATION_BUCKETS else "evaluation"


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _binary_metrics(
    rows: Sequence[dict[str, Any]],
    *,
    target_field: str,
    temperature: float = 1.0,
) -> dict[str, Any]:
    usable = []
    for row in rows:
        pass_weight = row.get("raw_weight_pass")
        fail_weight = row.get("raw_weight_fail")
        target = normalized_status(row.get(target_field))
        if (
            isinstance(pass_weight, (int, float))
            and isinstance(fail_weight, (int, float))
            and target is not None
        ):
            gap = (float(pass_weight) - float(fail_weight)) / temperature
            probability_pass = _sigmoid(gap)
            usable.append((probability_pass, target == "PASS"))
    if not usable:
        return {
            "count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "agreement_rate": None,
            "balanced_accuracy": None,
            "roc_auc": None,
            "confusion_matrix": None,
            "negative_log_likelihood": None,
            "brier_score": None,
            "expected_calibration_error_10_bin": None,
        }
    epsilon = 1e-12
    agreements = []
    losses = []
    briers = []
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(10)]
    for probability_pass, is_pass in usable:
        predicted_pass = probability_pass >= 0.5
        correct = predicted_pass == is_pass
        agreements.append(correct)
        target_probability = probability_pass if is_pass else 1.0 - probability_pass
        losses.append(-math.log(min(1.0 - epsilon, max(epsilon, target_probability))))
        briers.append((probability_pass - float(is_pass)) ** 2)
        confidence = max(probability_pass, 1.0 - probability_pass)
        bin_index = min(9, int(confidence * 10))
        bins[bin_index].append((confidence, correct))
    ece = 0.0
    for values in bins:
        if not values:
            continue
        mean_confidence = statistics.fmean(value[0] for value in values)
        accuracy = statistics.fmean(float(value[1]) for value in values)
        ece += len(values) / len(usable) * abs(mean_confidence - accuracy)
    true_positives = sum(
        probability_pass >= 0.5 and is_pass
        for probability_pass, is_pass in usable
    )
    false_positives = sum(
        probability_pass >= 0.5 and not is_pass
        for probability_pass, is_pass in usable
    )
    true_negatives = sum(
        probability_pass < 0.5 and not is_pass
        for probability_pass, is_pass in usable
    )
    false_negatives = sum(
        probability_pass < 0.5 and is_pass
        for probability_pass, is_pass in usable
    )
    positives = [probability for probability, is_pass in usable if is_pass]
    negatives = [probability for probability, is_pass in usable if not is_pass]
    roc_auc = (
        statistics.fmean(
            1.0 if positive > negative else 0.5 if positive == negative else 0.0
            for positive in positives
            for negative in negatives
        )
        if positives and negatives
        else None
    )
    balanced_accuracy = (
        (
            true_positives / len(positives)
            + true_negatives / len(negatives)
        )
        / 2.0
        if positives and negatives
        else None
    )
    return {
        "count": len(usable),
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "agreement_rate": statistics.fmean(float(value) for value in agreements),
        "balanced_accuracy": balanced_accuracy,
        "roc_auc": roc_auc,
        "confusion_matrix": {
            "true_pass_predicted_pass": true_positives,
            "true_fail_predicted_pass": false_positives,
            "true_fail_predicted_fail": true_negatives,
            "true_pass_predicted_fail": false_negatives,
        },
        "negative_log_likelihood": statistics.fmean(losses),
        "brier_score": statistics.fmean(briers),
        "expected_calibration_error_10_bin": ece,
    }


def fit_temperature(
    rows: Sequence[dict[str, Any]],
    *,
    target_field: str,
) -> float | None:
    labels = Counter(
        normalized_status(row.get(target_field))
        for row in rows
        if row.get("probe_available")
    )
    if labels.get("PASS", 0) < 2 or labels.get("FAIL", 0) < 2:
        return None
    candidates = [
        math.exp(math.log(0.05) + index * (math.log(20.0) - math.log(0.05)) / 400)
        for index in range(401)
    ]
    scored = [
        (
            _binary_metrics(
                rows,
                target_field=target_field,
                temperature=temperature,
            )["negative_log_likelihood"],
            temperature,
        )
        for temperature in candidates
    ]
    return min(scored, key=lambda item: float(item[0]))[1]


def _describe(values: Iterable[float]) -> dict[str, Any]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return {"count": 0, "mean": None, "median": None, "minimum": None, "maximum": None}
    return {
        "count": len(clean),
        "mean": statistics.fmean(clean),
        "median": statistics.median(clean),
        "minimum": min(clean),
        "maximum": max(clean),
    }


def summarize_probe_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "experiment_contract": {
            "probe_version": FIRST_VERDICT_ENTROPY_VERSION,
            "measurement": (
                "lowercase pass/fail logits at the authoritative first field "
                "of a detailed sidecar judge"
            ),
            "reference_labels": {
                "official_model_status": "status emitted by the original detailed model judge",
                "official_effective_status": "status after deterministic pipeline overrides",
            },
            "gate_effect": "none",
            "default_scope": "all attempts, not final attempts only",
            "calibration_split": (
                "evidence-ID-grouped deterministic 40% calibration / 60% evaluation"
            ),
        },
        "row_count": len(rows),
        "judges": {},
        "warnings": [],
    }
    for judge_name in PROBE_JUDGES:
        judge_rows = [row for row in rows if row.get("judge") == judge_name]
        available = [row for row in judge_rows if row.get("probe_available")]
        model_counts = Counter(str(row.get("official_model_status")) for row in judge_rows)
        effective_counts = Counter(
            str(row.get("official_effective_status")) for row in judge_rows
        )
        split_counts = Counter(calibration_split(row) for row in available)
        for row in available:
            row["calibration_split"] = calibration_split(row)
        calibration_rows = [
            row for row in available if row.get("calibration_split") == "calibration"
        ]
        evaluation_rows = [
            row for row in available if row.get("calibration_split") == "evaluation"
        ]
        temperature = fit_temperature(
            calibration_rows,
            target_field="official_model_status",
        )
        judge_summary = {
            "row_count": len(judge_rows),
            "available_count": len(available),
            "unavailable_count": len(judge_rows) - len(available),
            "official_model_status_counts": dict(model_counts),
            "official_effective_status_counts": dict(effective_counts),
            "effective_status_override_count": sum(
                bool(row.get("effective_status_overrode_model")) for row in judge_rows
            ),
            "sidecar_verdict_counts": dict(
                Counter(str(row.get("sidecar_verdict")) for row in available)
            ),
            "normalized_entropy": _describe(
                float(row["normalized_entropy"])
                for row in available
                if isinstance(row.get("normalized_entropy"), (int, float))
            ),
            "legacy_nested_status_normalized_entropy": _describe(
                float(row["legacy_nested_status_normalized_entropy"])
                for row in judge_rows
                if isinstance(
                    row.get("legacy_nested_status_normalized_entropy"),
                    (int, float),
                )
            ),
            "paired_entropy_change_first_minus_legacy": _describe(
                float(row["entropy_change_from_legacy_nested_status"])
                for row in available
                if isinstance(
                    row.get("entropy_change_from_legacy_nested_status"),
                    (int, float),
                )
            ),
            "normalized_entropy_by_model_status": {
                status: _describe(
                    float(row["normalized_entropy"])
                    for row in available
                    if row.get("official_model_status") == status
                    and isinstance(row.get("normalized_entropy"), (int, float))
                )
                for status in ("PASS", "FAIL")
            },
            "split_counts": dict(split_counts),
            "reference_model_status": {
                "all_uncalibrated": _binary_metrics(
                    available,
                    target_field="official_model_status",
                ),
                "evaluation_uncalibrated": _binary_metrics(
                    evaluation_rows,
                    target_field="official_model_status",
                ),
                "fitted_temperature": temperature,
                "evaluation_temperature_scaled": (
                    _binary_metrics(
                        evaluation_rows,
                        target_field="official_model_status",
                        temperature=temperature,
                    )
                    if temperature is not None
                    else None
                ),
            },
            "reference_effective_status": {
                "all_uncalibrated": _binary_metrics(
                    available,
                    target_field="official_effective_status",
                )
            },
        }
        if min(model_counts.get("PASS", 0), model_counts.get("FAIL", 0)) < 20:
            summary["warnings"].append(
                f"{judge_name} has fewer than 20 examples in one model-status class"
            )
        if len(available) != len(judge_rows):
            summary["warnings"].append(
                f"{judge_name} has {len(judge_rows) - len(available)} unavailable probes"
            )
        summary["judges"][judge_name] = judge_summary
    return summary


def write_probe_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = [
        "task_key",
        "source_path",
        "source_row",
        "source_trace_index",
        "evidence_id",
        "qa_id",
        "attempt",
        "is_final_attempt",
        "item_status",
        "judge",
        "official_model_status",
        "official_effective_status",
        "effective_status_overrode_model",
        "sidecar_verdict",
        "sidecar_verdict_status",
        "probe_matches_model_status",
        "probe_matches_effective_status",
        "sidecar_nested_check_status",
        "sidecar_verdict_matches_nested_status",
        "sidecar_verdict_is_authoritative",
        "probe_available",
        "probe_unavailable_reason",
        "raw_weight_pass",
        "raw_weight_fail",
        "probability_pass",
        "probability_fail",
        "entropy_nats",
        "normalized_entropy",
        "legacy_nested_status_normalized_entropy",
        "legacy_nested_status_entropy_nats",
        "legacy_nested_status_token_index",
        "legacy_nested_status_measurement_valid",
        "entropy_change_from_legacy_nested_status",
        "calibration_split",
        "token_index",
        "field_name",
        "prior_generated_verdict",
        "probe_output_contract_valid",
        "probe_version",
        "measurement_context",
        "media_role",
        "review_prompt_sha256",
        "probe_prompt_sha256",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def report_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Detailed Sidecar First-Verdict Entropy",
        "",
        "Lowercase pass/fail logits are measured at the first field of an independent "
        "detailed judge call. That verdict is authoritative for the sidecar call; later "
        "checks and explanations must agree with it. Sidecar output cannot affect production "
        "acceptance.",
        "",
        "The original model-judge status is the primary reference label. Metrics against the "
        "post-merge effective status are reported separately because deterministic formality "
        "rules can override the model branch.",
        "",
        "| Judge | Available | PASS / FAIL reference | Legacy nested-status mean H | First-decision mean H | Agreement | Test AUROC | Test NLL | Calibrated test NLL | Temperature |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for judge_name in PROBE_JUDGES:
        item = summary["judges"].get(judge_name, {})
        counts = item.get("official_model_status_counts") or {}
        entropy = item.get("normalized_entropy") or {}
        legacy_entropy = item.get("legacy_nested_status_normalized_entropy") or {}
        reference = item.get("reference_model_status") or {}
        all_metrics = reference.get("all_uncalibrated") or {}
        test_metrics = reference.get("evaluation_uncalibrated") or {}
        calibrated = reference.get("evaluation_temperature_scaled") or {}
        temperature = reference.get("fitted_temperature")
        lines.append(
            f"| {judge_name} | {item.get('available_count', 0)}/{item.get('row_count', 0)} | "
            f"{counts.get('PASS', 0)} / {counts.get('FAIL', 0)} | "
            f"{_display(legacy_entropy.get('mean'))} | "
            f"{_display(entropy.get('mean'))} | "
            f"{_display(all_metrics.get('agreement_rate'))} | "
            f"{_display(test_metrics.get('roc_auc'))} | "
            f"{_display(test_metrics.get('negative_log_likelihood'))} | "
            f"{_display(calibrated.get('negative_log_likelihood'))} | "
            f"{_display(temperature)} |"
        )
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- {warning}" for warning in summary.get("warnings") or []] or ["- none"])
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Entropy measures uncertainty, not whether pass or fail is the predicted class; use `probability_pass` or the argmax for that.",
            "- Legacy nested-status entropy is shown only as a paired control demonstrating the leak; it is not a valid uncertainty estimate.",
            "- Agreement and calibration compare the sidecar's authoritative verdict with the original production model judge, not human ground truth.",
            "- Temperature is fitted separately per judge on an evidence-ID-grouped calibration split.",
            "- All attempts are included by default to avoid final-attempt survivorship bias.",
            "",
        ]
    )
    return "\n".join(lines)


def _display(value: Any) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):.6f}"


def analyze_output(output_path: Path, output_dir: Path) -> dict[str, Any]:
    rows = list(iter_jsonl(output_path)) if output_path.is_file() else []
    summary = summarize_probe_rows(rows)
    write_json(output_dir / "summary.json", summary)
    write_probe_csv(output_dir / "first_verdict_sidecar.csv", rows)
    (output_dir / "report.md").write_text(
        report_markdown(summary),
        encoding="utf-8",
    )
    return summary


def run_sidecar(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "first_verdict_sidecar.jsonl"
    tasks = extract_probe_tasks(
        args.input,
        final_attempt_only=args.final_attempt_only,
    )
    selected = select_probe_tasks(
        tasks,
        balance_statuses=args.balance_statuses,
        max_per_status_per_judge=args.max_per_status_per_judge,
        random_seed=args.random_seed,
    )
    write_jsonl(output_dir / "task_manifest.jsonl", selected)
    preflight = {
        "source_task_count": len(tasks),
        "selected_task_count": len(selected),
        "selected_by_judge_and_status": {
            f"{judge}:{status}": count
            for (judge, status), count in sorted(
                Counter(
                    (str(task["judge"]), str(task["official_model_status"]))
                    for task in selected
                ).items()
            )
        },
        "balance_statuses": args.balance_statuses,
        "max_per_status_per_judge": args.max_per_status_per_judge,
        "min_per_status_per_judge": args.min_per_status_per_judge,
        "final_attempt_only": args.final_attempt_only,
        "random_seed": args.random_seed,
        "probe_version": FIRST_VERDICT_ENTROPY_VERSION,
    }
    write_json(output_dir / "preflight.json", preflight)
    if not selected:
        raise SystemExit("no eligible judge prompts found in the supplied trace files")
    selected_counts = Counter(
        (str(task["judge"]), str(task["official_model_status"]))
        for task in selected
    )
    shortages = [
        (
            judge_name,
            status,
            selected_counts.get((judge_name, status), 0),
        )
        for judge_name in PROBE_JUDGES
        for status in ("PASS", "FAIL")
        if selected_counts.get((judge_name, status), 0)
        < args.min_per_status_per_judge
    ]
    if shortages:
        detail = ", ".join(
            f"{judge}/{status}={count}"
            for judge, status, count in shortages
        )
        raise SystemExit(
            "selected cohort is too small for the requested experiment; "
            f"minimum per judge/status is {args.min_per_status_per_judge}: {detail}"
        )
    if args.plan_only:
        print(json.dumps(preflight, indent=2))
        return 0

    existing_keys = set()
    if args.resume and output_path.is_file():
        existing_rows = list(iter_jsonl(output_path))
        selected_by_key = {
            str(task["task_key"]): task
            for task in selected
        }
        stale_keys = [
            str(row.get("task_key"))
            for row in existing_rows
            if str(row.get("task_key")) not in selected_by_key
        ]
        changed_prompts = [
            str(row.get("task_key"))
            for row in existing_rows
            if str(row.get("task_key")) in selected_by_key
            and row.get("probe_prompt_sha256")
            != selected_by_key[str(row.get("task_key"))].get("probe_prompt_sha256")
        ]
        existing_key_counts = Counter(
            str(row.get("task_key"))
            for row in existing_rows
            if row.get("task_key")
        )
        duplicate_keys = [
            key for key, count in existing_key_counts.items() if count > 1
        ]
        if stale_keys or changed_prompts or duplicate_keys:
            raise SystemExit(
                "resume output does not match the current task manifest; use a new "
                f"output directory (stale={len(stale_keys)}, "
                f"changed_prompts={len(changed_prompts)}, "
                f"duplicates={len(duplicate_keys)})"
            )
        existing_keys = {
            str(row.get("task_key"))
            for row in existing_rows
            if row.get("task_key")
        }
    elif output_path.exists():
        output_path.write_text("", encoding="utf-8")

    pending = [task for task in selected if task["task_key"] not in existing_keys]
    if not args.skip_missing_media:
        missing = [
            (task["task_key"], path)
            for task in pending
            for path in missing_media(task)
        ]
        if missing:
            preview = "; ".join(f"{key}: {path}" for key, path in missing[:10])
            raise SystemExit(
                f"{len(missing)} required media paths are missing before model load: {preview}"
            )
    else:
        pending = [task for task in pending if not missing_media(task)]

    runner = make_runner(
        args.backend,
        model_id=args.model_id,
        base_url=args.base_url,
        max_new_tokens=args.max_new_tokens,
        max_image_pixels=args.max_image_pixels,
        dtype=args.dtype,
        allow_cpu=args.allow_cpu,
        allow_openai_video_input=args.allow_openai_video_input,
        disable_thinking=args.disable_thinking,
        api_key=args.api_key,
        video_fps=args.video_fps,
    )
    preflight["tokenizer_verdict_choices"] = verify_runner_lowercase_verdict_tokens(
        runner
    )
    write_json(output_dir / "preflight.json", preflight)
    print(
        "entropy_tokenizer_preflight "
        + json.dumps(preflight["tokenizer_verdict_choices"], sort_keys=True),
        flush=True,
    )
    for index, task in enumerate(pending, 1):
        print(
            "entropy_probe_start "
            f"index={index}/{len(pending)} judge={task['judge']} "
            f"evidence_id={task['evidence_id']} attempt={task['attempt']}",
            flush=True,
        )
        row = run_probe_task(task, runner)
        append_jsonl(output_path, row)
        print(
            "entropy_probe_done "
            f"available={row['probe_available']} verdict={row['sidecar_verdict']} "
            f"normalized_entropy={row['normalized_entropy']}",
            flush=True,
        )
    summary = analyze_output(output_path, output_dir)
    print(
        f"first_verdict_sidecar_rows={summary['row_count']} output={output_path}",
        flush=True,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run or analyze detailed sidecar judges with an authoritative first "
            "lowercase pass/fail verdict."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--input", action="append", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument(
        "--backend",
        default="transformers-local",
        choices=[
            "transformers-local",
            MEMORY_SAFE_BACKEND,
            "openai-compatible-local",
            "gemini",
        ],
    )
    run.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    run.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    run.add_argument("--api-key")
    run.add_argument("--max-new-tokens", type=int, default=4096)
    run.add_argument("--max-image-pixels", type=int, default=262144)
    run.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    run.add_argument("--allow-cpu", action="store_true")
    run.add_argument("--allow-openai-video-input", action="store_true")
    run.add_argument("--disable-thinking", action="store_true")
    run.add_argument("--video-fps", type=float, default=1.0)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--plan-only", action="store_true")
    run.add_argument("--skip-missing-media", action="store_true")
    run.add_argument("--final-attempt-only", action="store_true")
    run.add_argument("--balance-statuses", action="store_true")
    run.add_argument("--max-per-status-per-judge", type=int)
    run.add_argument(
        "--min-per-status-per-judge",
        type=int,
        default=20,
        help="Fail preflight if any judge/status cell has fewer selected examples.",
    )
    run.add_argument("--random-seed", type=int, default=20260728)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--input", required=True)
    analyze.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        if (
            args.max_per_status_per_judge is not None
            and args.max_per_status_per_judge < 1
        ):
            raise SystemExit("--max-per-status-per-judge must be positive")
        if args.min_per_status_per_judge < 0:
            raise SystemExit("--min-per-status-per-judge cannot be negative")
        return run_sidecar(args)
    if args.command == "analyze":
        summary = analyze_output(Path(args.input), Path(args.output_dir))
        print(json.dumps(summary, indent=2))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
