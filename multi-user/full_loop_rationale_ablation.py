"""Independent end-to-end generator-rationale ablation trials.

This module deliberately leaves the historical attempt-1 ablation and the
production video loop unchanged.  ``run`` seeds one process, verifies that it
can see exactly one GPU, and then invokes a fresh video-first generation loop.
``compare`` summarizes every generation/review attempt and the final outcome
for every evidence packet in the two condition directories.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Iterable

from .io_utils import iter_jsonl, write_json


CONDITIONS = ("with_rationale", "without_rationale")
SCORED_CHECKS = ("qa_formality", "evidence_groundedness")


def condition_includes_rationale(condition: str) -> bool:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown rationale condition: {condition}")
    return condition == "with_rationale"


def trial_paths(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    return {
        "accepted": root / "qa_mcq.jsonl",
        "rejected": root / "qa_mcq.rejected.jsonl",
        "intermediate": root / "qa_mcq.intermediate.jsonl",
        "prompts": root / "video_first_prompts.jsonl",
        "summary": root / "trial_summary.json",
    }


def build_video_loop_argv(
    *,
    condition: str,
    evidence_path: str | Path,
    output_dir: str | Path,
    target_count: int,
    max_attempts: int,
    model_id: str,
    max_new_tokens: int,
    max_image_pixels: int,
    dtype: str,
    temperature: float,
    top_p: float,
    top_k: int,
    quality_quota: int,
    resume: bool = False,
) -> list[str]:
    """Construct one untouched production-loop invocation for a condition."""

    paths = trial_paths(output_dir)
    argv = [
        "--evidence",
        str(evidence_path),
        "--output",
        str(paths["accepted"]),
        "--prompts-output",
        str(paths["prompts"]),
        "--rejected-output",
        str(paths["rejected"]),
        "--intermediate-output",
        str(paths["intermediate"]),
        "--target-count",
        str(target_count),
        "--max-attempts",
        str(max_attempts),
        "--backend",
        "transformers-local",
        "--model-id",
        model_id,
        "--max-new-tokens",
        str(max_new_tokens),
        "--max-image-pixels",
        str(max_image_pixels),
        "--dtype",
        dtype,
        "--disable-thinking",
        "--generation-mode",
        "baseline",
        "--fixed-question-type-schedule",
        "--question-types",
        "neutral",
        "--generator-decode-mode",
        "sampling",
        "--generator-temperature",
        str(temperature),
        "--generator-top-p",
        str(top_p),
        "--generator-top-k",
        str(top_k),
        # Archived scoring/quota pipeline flags:
        # "--experimental-scored-judge",
        # "--judge-quality-quota",
        # str(quality_quota),
    ]
    if not condition_includes_rationale(condition):
        argv.append("--judge-hide-generator-rationale")
    if resume:
        argv.append("--resume")
    return argv


def seed_single_gpu_process(seed: int) -> dict[str, Any]:
    """Seed one Slurm step and reject accidental two-GPU model visibility."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for a local-Qwen trial") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("the full-loop trial requires a CUDA GPU")
    visible_count = int(torch.cuda.device_count())
    if visible_count != 1:
        raise RuntimeError(
            "each rationale-ablation Slurm step must see exactly one GPU; "
            f"torch.cuda.device_count()={visible_count} "
            f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES', '<unset>')}"
        )
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return {
        "seed": seed,
        "visible_gpu_count": visible_count,
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name(0),
    }


def _canonical_attempt_traces(row: dict[str, Any]) -> list[dict[str, Any]]:
    direct = [
        trace
        for trace in row.get("attempts") or []
        if isinstance(trace, dict) and isinstance(trace.get("generation"), dict)
    ]
    if direct:
        return direct
    return [
        trace
        for trace in row.get("generation_trace") or []
        if isinstance(trace, dict) and isinstance(trace.get("generation"), dict)
    ]


def _counter_json(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda row: str(row[0]))}


def _rows(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    return list(iter_jsonl(file_path)) if file_path.exists() else []


def _first_attempt_hashes(rows: Iterable[dict[str, Any]]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for row in rows:
        evidence_id = str(row.get("evidence_id") or "")
        traces = _canonical_attempt_traces(row)
        if not evidence_id or not traces:
            continue
        first = next((trace for trace in traces if trace.get("attempt") == 1), traces[0])
        raw = (first.get("generation") or {}).get("raw_output")
        if isinstance(raw, str):
            hashes[evidence_id] = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return hashes


def summarize_trial(
    intermediate_path: str | Path,
    *,
    condition: str,
    packet_count: int,
    quality_quota: int,
) -> dict[str, Any]:
    """Summarize final packet outcomes and every attempt in one fresh loop."""

    expected_rationale = condition_includes_rationale(condition)
    rows = _rows(intermediate_path)
    if len(rows) != packet_count:
        raise ValueError(
            f"{condition} intermediate has {len(rows)} packets; expected {packet_count}"
        )

    statuses = Counter(str(row.get("status") or "missing") for row in rows)
    attempt_histogram: Counter[int] = Counter()
    accepted_by_attempt: Counter[int] = Counter()
    score_counts = {check: Counter() for check in SCORED_CHECKS}
    pass_fail_counts = {check: Counter() for check in SCORED_CHECKS}
    three_positions = {check: [] for check in SCORED_CHECKS}
    quota_counts = {check: 0 for check in SCORED_CHECKS}
    quality_reason_present = {check: 0 for check in SCORED_CHECKS}
    quota_rebuttal_required = {check: 0 for check in SCORED_CHECKS}
    quota_rebuttal_present = {check: 0 for check in SCORED_CHECKS}
    quota_rebuttal_present_when_required = {check: 0 for check in SCORED_CHECKS}
    quota_exceedance_records = {check: [] for check in SCORED_CHECKS}
    quota_metadata_mismatch_records = {check: [] for check in SCORED_CHECKS}
    final_rejection_failures: Counter[str] = Counter()
    all_attempt_grounding_failures = 0
    grounding_hallucination_mentions = 0
    total_attempts = 0
    judged_attempts = 0

    for packet_index, row in enumerate(rows, 1):
        traces = _canonical_attempt_traces(row)
        attempt_histogram[len(traces)] += 1
        total_attempts += len(traces)
        accepted_attempt = next(
            (
                int(trace.get("attempt") or trace_index)
                for trace_index, trace in enumerate(traces, 1)
                if (trace.get("result") or {}).get("accepted") is True
            ),
            None,
        )
        if accepted_attempt is not None:
            accepted_by_attempt[accepted_attempt] += 1

        last_checks: dict[str, Any] = {}
        for trace in traces:
            judge = trace.get("judge") if isinstance(trace.get("judge"), dict) else {}
            merged = judge.get("merged") if isinstance(judge.get("merged"), dict) else {}
            checks = merged.get("checks") if isinstance(merged.get("checks"), dict) else {}
            if not checks:
                continue
            judged_attempts += 1
            last_checks = checks
            if judge.get("generator_rationale_included") is not expected_rationale:
                raise ValueError(
                    f"{condition} packet {packet_index} attempt {trace.get('attempt')} "
                    "has the wrong generator-rationale disclosure state"
                )
            if judge.get("pass_fail_only") is not False:
                raise ValueError(f"{condition} unexpectedly used the legacy binary-only judge")
            if judge.get("pass_fail_entropy_logits") != "legacy_archived_not_collected":
                raise ValueError(f"{condition} unexpectedly retained PASS/FAIL entropy logits")

            for check_name in SCORED_CHECKS:
                check = checks.get(check_name) if isinstance(checks.get(check_name), dict) else {}
                score = check.get("quality_score")
                score_counts[check_name][score] += 1
                pass_fail_counts[check_name][check.get("status")] += 1
                if str(check.get("quality_reason") or "").strip():
                    quality_reason_present[check_name] += 1
                quota = check.get("quality_quota") if isinstance(check.get("quality_quota"), dict) else {}
                previous = quota.get("previous_three_point_assignments")
                remaining = quota.get("remaining_before_candidate")
                recorded_limit = quota.get("quota")
                if recorded_limit is not None and recorded_limit != quality_quota:
                    raise ValueError(
                        f"{condition} {check_name} quota limit mismatch: "
                        f"observed={recorded_limit} expected={quality_quota}"
                    )
                if previous != quota_counts[check_name]:
                    raise ValueError(
                        f"{condition} {check_name} quota counter mismatch: "
                        f"observed={previous} expected={quota_counts[check_name]}"
                    )
                expected_remaining = max(0, quality_quota - quota_counts[check_name])
                if remaining != expected_remaining:
                    # Older completed trials used more than one descriptive
                    # remaining-capacity convention. The running `previous` counter is
                    # the authoritative resume state; retain this discrepancy for the
                    # audit instead of preventing an otherwise safe resume.
                    quota_metadata_mismatch_records[check_name].append(
                        {
                            "judged_attempt": judged_attempts,
                            "packet_index": packet_index,
                            "evidence_id": row.get("evidence_id"),
                            "attempt": trace.get("attempt"),
                            "field": "remaining_before_candidate",
                            "observed": remaining,
                            "expected": expected_remaining,
                        }
                    )
                rebuttal = str(check.get("quota_rebuttal") or "").strip()
                if quota.get("quota_rebuttal_required"):
                    quota_rebuttal_required[check_name] += 1
                    if rebuttal:
                        quota_rebuttal_present_when_required[check_name] += 1
                    quota_exceedance_records[check_name].append(
                        {
                            "judged_attempt": judged_attempts,
                            "packet_index": packet_index,
                            "evidence_id": row.get("evidence_id"),
                            "attempt": trace.get("attempt"),
                            "status": check.get("status"),
                            "quality_score": score,
                            "quality_reason": check.get("quality_reason"),
                            "quota_rebuttal": check.get("quota_rebuttal"),
                        }
                    )
                if rebuttal:
                    quota_rebuttal_present[check_name] += 1
                if score == 3:
                    quota_counts[check_name] += 1
                    three_positions[check_name].append(
                        {
                            "judged_attempt": judged_attempts,
                            "packet_index": packet_index,
                            "attempt": trace.get("attempt"),
                        }
                    )

            grounding = checks.get("evidence_groundedness") or {}
            if grounding.get("status") == "FAIL":
                all_attempt_grounding_failures += 1
                grounding_text = " ".join(
                    str(grounding.get(field) or "")
                    for field in ("reason", "fix", "quality_reason")
                ).lower()
                if "hallucinat" in grounding_text:
                    grounding_hallucination_mentions += 1

        if row.get("status") == "rejected" and last_checks:
            for check_name in (*SCORED_CHECKS, "answerability"):
                check = last_checks.get(check_name) or {}
                if check.get("status") == "FAIL":
                    final_rejection_failures[check_name] += 1

    accepted_count = statuses.get("accepted", 0)
    summary: dict[str, Any] = {
        "condition": condition,
        "generator_rationale_included": expected_rationale,
        "scope": "all_attempts_and_final_packet_outcomes",
        "packet_count": len(rows),
        "accepted_count": accepted_count,
        "rejected_count": statuses.get("rejected", 0),
        "acceptance_rate": round(accepted_count / len(rows), 6),
        "total_generation_attempts": total_attempts,
        "judged_attempts": judged_attempts,
        "average_generation_attempts_per_packet": round(total_attempts / len(rows), 6),
        "generation_attempt_histogram": _counter_json(attempt_histogram),
        "accepted_by_attempt": _counter_json(accepted_by_attempt),
        "final_rejection_failure_counts": _counter_json(final_rejection_failures),
        "all_attempt_grounding_failure_count": all_attempt_grounding_failures,
        "all_attempt_grounding_hallucination_mention_count": grounding_hallucination_mentions,
        "evidence_ids": [str(row.get("evidence_id") or "") for row in rows],
        "checks": {},
    }
    for check_name in SCORED_CHECKS:
        positions = three_positions[check_name]
        summary["checks"][check_name] = {
            "score_distribution": _counter_json(score_counts[check_name]),
            "pass_fail_distribution": _counter_json(pass_fail_counts[check_name]),
            "quality_reason_present_count": quality_reason_present[check_name],
            "quality_reason_missing_count": (
                judged_attempts - quality_reason_present[check_name]
            ),
            "three_point_assignment_count": quota_counts[check_name],
            "three_point_positions": positions,
            "last_three_point_position": positions[-1] if positions else None,
            "quota_rebuttal_required_count": quota_rebuttal_required[check_name],
            "quota_rebuttal_present_count": quota_rebuttal_present[check_name],
            "quota_rebuttal_present_when_required_count": (
                quota_rebuttal_present_when_required[check_name]
            ),
            "quota_rebuttal_missing_when_required_count": (
                quota_rebuttal_required[check_name]
                - quota_rebuttal_present_when_required[check_name]
            ),
            "quota_exceedance_records": quota_exceedance_records[check_name],
            "quota_metadata_mismatch_records": quota_metadata_mismatch_records[
                check_name
            ],
        }
    return summary


def compare_trials(
    *,
    with_rationale_intermediate: str | Path,
    without_rationale_intermediate: str | Path,
    packet_count: int,
    quality_quota: int,
) -> dict[str, Any]:
    with_summary = summarize_trial(
        with_rationale_intermediate,
        condition="with_rationale",
        packet_count=packet_count,
        quality_quota=quality_quota,
    )
    without_summary = summarize_trial(
        without_rationale_intermediate,
        condition="without_rationale",
        packet_count=packet_count,
        quality_quota=quality_quota,
    )
    if with_summary["evidence_ids"] != without_summary["evidence_ids"]:
        raise ValueError("the two full-loop trials did not process the same evidence order")

    with_rows = _rows(with_rationale_intermediate)
    without_rows = _rows(without_rationale_intermediate)
    outcome_pairs: Counter[str] = Counter()
    for without_row, with_row in zip(without_rows, with_rows):
        outcome_pairs[
            f"without_{without_row.get('status')}__with_{with_row.get('status')}"
        ] += 1

    with_hashes = _first_attempt_hashes(with_rows)
    without_hashes = _first_attempt_hashes(without_rows)
    shared_hash_ids = sorted(set(with_hashes) & set(without_hashes))
    exact_matches = [
        evidence_id
        for evidence_id in shared_hash_ids
        if with_hashes[evidence_id] == without_hashes[evidence_id]
    ]
    mismatches = [
        evidence_id
        for evidence_id in shared_hash_ids
        if with_hashes[evidence_id] != without_hashes[evidence_id]
    ]

    with_rate = float(with_summary["acceptance_rate"])
    without_rate = float(without_summary["acceptance_rate"])
    return {
        "scope": "two_independent_full_generation_loops_all_attempts",
        "packet_count": packet_count,
        "same_evidence_order": True,
        "with_rationale": with_summary,
        "without_rationale": without_summary,
        "final_packet_outcome_pairs": _counter_json(outcome_pairs),
        "acceptance_rate_delta_with_minus_without": round(with_rate - without_rate, 6),
        "acceptance_rate_delta_percentage_points": round((with_rate - without_rate) * 100, 3),
        "generation_attempt_delta_with_minus_without": (
            int(with_summary["total_generation_attempts"])
            - int(without_summary["total_generation_attempts"])
        ),
        "first_attempt_generation_pairing": {
            "shared_parseable_raw_output_count": len(shared_hash_ids),
            "exact_match_count": len(exact_matches),
            "mismatch_count": len(mismatches),
            "mismatch_evidence_ids": mismatches,
        },
        "interpretation_note": (
            "Both conditions regenerate from the same videos with the same seed and settings. "
            "The treatment is whether judges see generator_rationale. Retry trajectories may "
            "diverge after a different judgment or feedback message, so the primary endpoints "
            "are full-loop acceptance, attempt usage, and all-attempt judge behavior."
        ),
    }


def run_condition(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gpu = seed_single_gpu_process(args.seed)
    print("full_loop_trial_gpu=" + json.dumps(gpu, sort_keys=True), flush=True)
    argv = build_video_loop_argv(
        condition=args.condition,
        evidence_path=args.evidence,
        output_dir=output_dir,
        target_count=args.target_count,
        max_attempts=args.max_attempts,
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        max_image_pixels=args.max_image_pixels,
        dtype=args.dtype,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        quality_quota=args.quality_quota,
        resume=bool(getattr(args, "resume", False)),
    )
    from .video_qa_loop import main as video_loop_main

    result = video_loop_main(argv)
    if result != 0:
        raise RuntimeError(f"video loop returned nonzero status {result}")
    paths = trial_paths(output_dir)
    summary = summarize_trial(
        paths["intermediate"],
        condition=args.condition,
        packet_count=args.target_count,
        quality_quota=args.quality_quota,
    )
    summary["seed"] = args.seed
    summary["gpu"] = gpu
    write_json(paths["summary"], summary)
    print("full_loop_trial_summary=" + json.dumps(summary, sort_keys=True), flush=True)
    return summary


def run_sequential_conditions(args: argparse.Namespace) -> dict[str, Any]:
    """Run both full-loop arms on one GPU while loading Qwen only once."""

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    with_dir = output_root / "with_rationale"
    without_dir = output_root / "without_rationale"
    comparison_path = output_root / "full_loop_comparison.json"

    from . import video_qa_loop

    original_make_runner = video_qa_loop.make_runner
    cached_runner: Any | None = None
    cached_spec: tuple[Any, ...] | None = None
    model_load_count = 0

    def make_runner_once(backend: str, *runner_args: Any, **runner_kwargs: Any) -> Any:
        nonlocal cached_runner, cached_spec, model_load_count
        spec = (backend, runner_args, tuple(sorted(runner_kwargs.items())))
        if cached_runner is None:
            cached_runner = original_make_runner(backend, *runner_args, **runner_kwargs)
            cached_spec = spec
            model_load_count += 1
            print("sequential_model_cache=loaded", flush=True)
        elif spec != cached_spec:
            raise RuntimeError(
                "the sequential ablation attempted to change runner configuration "
                "after the single shared model was loaded"
            )
        else:
            print("sequential_model_cache=reused", flush=True)
        return cached_runner

    video_qa_loop.make_runner = make_runner_once
    summaries: dict[str, Any] = {}
    try:
        seed_single_gpu_process(args.seed)
        make_runner_once(
            "transformers-local",
            model_id=args.model_id,
            base_url="http://127.0.0.1:8000/v1",
            max_new_tokens=args.max_new_tokens,
            max_image_pixels=args.max_image_pixels,
            dtype=args.dtype,
            allow_cpu=False,
            allow_openai_video_input=False,
            disable_thinking=True,
            api_key=None,
        )
        print("sequential_model_preloaded_before_conditions=true", flush=True)
        for condition, condition_dir in (
            ("with_rationale", with_dir),
            ("without_rationale", without_dir),
        ):
            print(f"sequential_condition_start={condition}", flush=True)
            condition_args = argparse.Namespace(
                **{
                    **vars(args),
                    "condition": condition,
                    "output_dir": str(condition_dir),
                }
            )
            summaries[condition] = run_condition(condition_args)
            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except (ImportError, RuntimeError):
                pass
            print(f"sequential_condition_done={condition}", flush=True)
    finally:
        video_qa_loop.make_runner = original_make_runner

    if model_load_count != 1:
        raise RuntimeError(f"expected exactly one shared model load; observed {model_load_count}")
    comparison = compare_trials(
        with_rationale_intermediate=trial_paths(with_dir)["intermediate"],
        without_rationale_intermediate=trial_paths(without_dir)["intermediate"],
        packet_count=args.target_count,
        quality_quota=args.quality_quota,
    )
    comparison["execution"] = {
        "topology": "one_gpu_two_full_loops_sequential",
        "condition_order": ["with_rationale", "without_rationale"],
        "shared_model_load_count": model_load_count,
        "model_preloaded_before_condition_seeding": True,
        "seed_reset_before_each_condition": True,
    }
    write_json(comparison_path, comparison)
    print("full_loop_comparison=" + json.dumps(comparison, sort_keys=True), flush=True)
    return comparison


def resume_sequential_without_rationale(args: argparse.Namespace) -> dict[str, Any]:
    """Resume only the interrupted without-rationale arm, then compare both arms."""

    output_root = Path(args.output_root)
    with_dir = output_root / "with_rationale"
    without_dir = output_root / "without_rationale"
    comparison_path = output_root / "full_loop_comparison.json"
    with_intermediate = trial_paths(with_dir)["intermediate"]
    without_intermediate = trial_paths(without_dir)["intermediate"]

    with_rows = _rows(with_intermediate)
    without_rows = _rows(without_intermediate)
    if len(with_rows) != args.target_count:
        raise ValueError(
            "cannot resume: completed with-rationale arm has "
            f"{len(with_rows)} packets; expected {args.target_count}"
        )
    if not 0 < len(without_rows) < args.target_count:
        raise ValueError(
            "cannot resume: without-rationale arm must be partial; "
            f"observed {len(without_rows)} completed packets"
        )
    summarize_trial(
        with_intermediate,
        condition="with_rationale",
        packet_count=args.target_count,
        quality_quota=args.quality_quota,
    )
    partial_summary = summarize_trial(
        without_intermediate,
        condition="without_rationale",
        packet_count=len(without_rows),
        quality_quota=args.quality_quota,
    )
    evidence_rows = _rows(args.evidence)
    expected_ids = [
        str(row.get("evidence_id") or "") for row in evidence_rows[: args.target_count]
    ]
    with_ids = [str(row.get("evidence_id") or "") for row in with_rows]
    without_ids = [str(row.get("evidence_id") or "") for row in without_rows]
    if with_ids != expected_ids:
        raise ValueError("completed with-rationale IDs do not match the evidence order")
    if without_ids != expected_ids[: len(without_ids)]:
        raise ValueError("partial without-rationale IDs are not a prefix of the evidence order")

    accepted_ids = {
        str(row.get("evidence_id") or "")
        for row in _rows(trial_paths(without_dir)["accepted"])
    }
    rejected_ids = {
        str(row.get("evidence_id") or "")
        for row in _rows(trial_paths(without_dir)["rejected"])
    }
    if accepted_ids & rejected_ids:
        raise ValueError("partial without-rationale outputs duplicate accepted/rejected IDs")
    if accepted_ids | rejected_ids != set(without_ids):
        raise ValueError(
            "partial intermediate IDs do not match accepted/rejected terminal outputs; "
            "resume would risk regenerating completed packets"
        )

    from . import video_qa_loop

    original_make_runner = video_qa_loop.make_runner
    cached_runner: Any | None = None
    cached_spec: tuple[Any, ...] | None = None
    model_load_count = 0

    def make_runner_once(backend: str, *runner_args: Any, **runner_kwargs: Any) -> Any:
        nonlocal cached_runner, cached_spec, model_load_count
        spec = (backend, runner_args, tuple(sorted(runner_kwargs.items())))
        if cached_runner is None:
            cached_runner = original_make_runner(backend, *runner_args, **runner_kwargs)
            cached_spec = spec
            model_load_count += 1
            print("resume_model_cache=loaded", flush=True)
        elif spec != cached_spec:
            raise RuntimeError("resume attempted to change the shared runner configuration")
        else:
            print("resume_model_cache=reused", flush=True)
        return cached_runner

    video_qa_loop.make_runner = make_runner_once
    try:
        seed_single_gpu_process(args.seed)
        make_runner_once(
            "transformers-local",
            model_id=args.model_id,
            base_url="http://127.0.0.1:8000/v1",
            max_new_tokens=args.max_new_tokens,
            max_image_pixels=args.max_image_pixels,
            dtype=args.dtype,
            allow_cpu=False,
            allow_openai_video_input=False,
            disable_thinking=True,
            api_key=None,
        )
        resume_args = argparse.Namespace(
            **{
                **vars(args),
                "condition": "without_rationale",
                "output_dir": str(without_dir),
                "resume": True,
            }
        )
        run_condition(resume_args)
    finally:
        video_qa_loop.make_runner = original_make_runner

    if model_load_count != 1:
        raise RuntimeError(f"expected one model load during resume; observed {model_load_count}")
    comparison = compare_trials(
        with_rationale_intermediate=with_intermediate,
        without_rationale_intermediate=without_intermediate,
        packet_count=args.target_count,
        quality_quota=args.quality_quota,
    )
    comparison["execution"] = {
        "topology": "one_gpu_resume_without_rationale_only",
        "resumed_from_completed_packet_count": len(without_rows),
        "resumed_from_generation_attempt_count": partial_summary[
            "total_generation_attempts"
        ],
        "completed_with_rationale_reused": True,
        "shared_model_load_count_this_job": model_load_count,
        "quota_state_restored_from_intermediate": True,
        "sampling_rng_note": (
            "The interrupted process did not persist torch RNG state. Remaining generator "
            "calls start from the configured resume seed; completed candidates are not regenerated."
        ),
    }
    write_json(comparison_path, comparison)
    print("full_loop_resume_comparison=" + json.dumps(comparison, sort_keys=True), flush=True)
    return comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run one fresh full-loop condition on one GPU")
    run.add_argument("--condition", required=True, choices=CONDITIONS)
    run.add_argument("--evidence", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--target-count", type=int, default=50)
    run.add_argument("--max-attempts", type=int, default=3)
    run.add_argument("--model-id", default="Qwen/Qwen3.6-27B")
    run.add_argument("--max-new-tokens", type=int, default=4096)
    run.add_argument("--max-image-pixels", type=int, default=262144)
    run.add_argument("--dtype", default="bfloat16")
    run.add_argument("--temperature", type=float, default=0.7)
    run.add_argument("--top-p", type=float, default=0.9)
    run.add_argument("--top-k", type=int, default=40)
    run.add_argument("--quality-quota", type=int, default=48)
    run.add_argument("--seed", type=int, default=1729)

    sequential = subparsers.add_parser(
        "run-sequential",
        help="run both fresh full-loop conditions sequentially on one shared GPU model",
    )
    sequential.add_argument("--evidence", required=True)
    sequential.add_argument("--output-root", required=True)
    sequential.add_argument("--target-count", type=int, default=50)
    sequential.add_argument("--max-attempts", type=int, default=3)
    sequential.add_argument("--model-id", default="Qwen/Qwen3.6-27B")
    sequential.add_argument("--max-new-tokens", type=int, default=4096)
    sequential.add_argument("--max-image-pixels", type=int, default=262144)
    sequential.add_argument("--dtype", default="bfloat16")
    sequential.add_argument("--temperature", type=float, default=0.7)
    sequential.add_argument("--top-p", type=float, default=0.9)
    sequential.add_argument("--top-k", type=int, default=40)
    sequential.add_argument("--quality-quota", type=int, default=48)
    sequential.add_argument("--seed", type=int, default=1729)

    resume = subparsers.add_parser(
        "resume-sequential",
        help="resume only a partial without-rationale arm in a sequential-run output root",
    )
    resume.add_argument("--evidence", required=True)
    resume.add_argument("--output-root", required=True)
    resume.add_argument("--target-count", type=int, default=50)
    resume.add_argument("--max-attempts", type=int, default=3)
    resume.add_argument("--model-id", default="Qwen/Qwen3.6-27B")
    resume.add_argument("--max-new-tokens", type=int, default=4096)
    resume.add_argument("--max-image-pixels", type=int, default=262144)
    resume.add_argument("--dtype", default="bfloat16")
    resume.add_argument("--temperature", type=float, default=0.7)
    resume.add_argument("--top-p", type=float, default=0.9)
    resume.add_argument("--top-k", type=int, default=40)
    resume.add_argument("--quality-quota", type=int, default=48)
    resume.add_argument("--seed", type=int, default=1729)

    compare = subparsers.add_parser("compare", help="compare both completed full loops")
    compare.add_argument("--with-rationale-intermediate", required=True)
    compare.add_argument("--without-rationale-intermediate", required=True)
    compare.add_argument("--output", required=True)
    compare.add_argument("--packet-count", type=int, default=50)
    compare.add_argument("--quality-quota", type=int, default=48)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        run_condition(args)
        return 0
    if args.command == "run-sequential":
        run_sequential_conditions(args)
        return 0
    if args.command == "resume-sequential":
        resume_sequential_without_rationale(args)
        return 0
    comparison = compare_trials(
        with_rationale_intermediate=args.with_rationale_intermediate,
        without_rationale_intermediate=args.without_rationale_intermediate,
        packet_count=args.packet_count,
        quality_quota=args.quality_quota,
    )
    write_json(args.output, comparison)
    print("full_loop_comparison=" + json.dumps(comparison, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
