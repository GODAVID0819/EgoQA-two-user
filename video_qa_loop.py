"""Video-first generation loop for EgoLife two-user multiple-choice construction."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import nullcontext
from inspect import signature
import itertools
import json
import math
import re
import statistics
import tempfile
from threading import BoundedSemaphore
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from .io_utils import append_jsonl, iter_jsonl, write_json, write_jsonl
from .prompts import (
    DEFAULT_QUALITY_QUOTA,
    GENERATION_MODES,
    JUDGE_SCHEMA,
    JUDGE_OUTPUT_SCHEMA_MARKER,
    QA_FORMALITY_SEMANTIC_SUBCHECK_NAMES,
    build_answerability_condition_aggregation_prompt,
    build_answerability_fact_plan_prompt,
    build_answerability_prompt,
    build_answerability_user_fact_audit_prompt,
    build_evidence_observation_aggregation_prompt,
    build_evidence_segment_observation_prompt,
    build_judge_minimal_verdict_probe_prompt,
    build_evidence_groundedness_judge_prompt,
    build_judge_json_repair_prompt,
    build_qa_formality_judge_prompt,
    build_sequential_direct_judge_prompt,
    build_video_generation_prompt,
    formality_participant_names,
    judge_schema_for_check,
    qa_formality_errors,
)
# Archived discovery-mode imports:
# from .prompts import build_relation_discovery_prompt, build_relation_mcq_prompt
from .qwen3vl_runner import (
    DEFAULT_MODEL_ID,
    DEFAULT_SAMPLING_TEMPERATURE,
    DEFAULT_SAMPLING_TOP_P,
    GENERATOR_DECODING_MODES,
    OpenRouterRequestError,
    OPENROUTER_REASONING_EFFORTS,
    make_runner,
)
from .schema import OPTION_LETTERS, extract_json_object, normalize_correct, validate_qa_item


SIX_USER_JUDGE_MODE_TIME_AWARE = "time-aware-map-reduce"
SIX_USER_JUDGE_MODE_LEGACY = "legacy-zero-shot"
SIX_USER_JUDGE_MODE_SEQUENTIAL = "sequential-separated-fact-audit"
SIX_USER_JUDGE_MODES = (
    SIX_USER_JUDGE_MODE_TIME_AWARE,
    SIX_USER_JUDGE_MODE_LEGACY,
    SIX_USER_JUDGE_MODE_SEQUENTIAL,
)


class JudgeInfrastructureError(RuntimeError):
    """A model-runtime failure that must not be converted into a semantic reject."""

    def __init__(self, *, stage: str, cause: BaseException) -> None:
        self.stage = stage
        self.cause = cause
        super().__init__(f"{stage} infrastructure failure: {cause}")


def is_cuda_oom_error(exc: BaseException) -> bool:
    """Recognize catchable CUDA allocation failures across Torch/cuDNN wrappers."""

    current: BaseException | None = exc
    seen: set[int] = set()
    messages: list[str] = []
    class_names: list[str] = []
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        messages.append(str(current).lower())
        class_names.append(type(current).__name__.lower())
        current = current.__cause__ or current.__context__
    if any(name in {"outofmemoryerror", "cudaoutofmemoryerror"} for name in class_names):
        return True
    joined = "\n".join(messages)
    return any(
        marker in joined
        for marker in (
            "cuda out of memory",
            "cuda error: out of memory",
            "cudnn_status_alloc_failed",
            "cuda allocator",
        )
    )


class StreamingJsonlRows(list[dict[str, Any]]):
    """Keep an in-memory row list while also flushing each row to disk."""

    def __init__(self, path: str | Path | None, *, reset: bool = True) -> None:
        super().__init__()
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if reset or not self.path.exists():
                self.path.write_text("", encoding="utf-8")

    def load_existing(self) -> None:
        if not self.path or not self.path.exists():
            return
        for row in iter_jsonl(self.path):
            super().append(row)

    def append(self, row: dict[str, Any]) -> None:
        super().append(row)
        if self.path:
            append_jsonl(self.path, row)


def compact_prompt_record(row: dict[str, Any]) -> dict[str, Any]:
    """Keep prompt text and aggregate media counts, never exact media mappings."""

    if (
        "media_summary" in row
        and "image_paths" not in row
        and "video_paths" not in row
        and "condition_media" not in row
    ):
        return dict(row)
    compact = {
        key: value
        for key, value in row.items()
        if key not in {"image_paths", "video_paths", "condition_media"}
    }
    image_paths = row.get("image_paths")
    video_paths = row.get("video_paths")
    condition_media = row.get("condition_media")
    media_summary = {
        "image_count": len(image_paths) if isinstance(image_paths, list) else 0,
        "video_count": len(video_paths) if isinstance(video_paths, list) else 0,
        "media_role": row.get("media_role"),
        "exact_media_mapping": "omitted; resolve by evidence_id from the evidence artifact",
    }
    if isinstance(condition_media, dict):
        media_summary.update(
            {
                "condition_id": condition_media.get("condition_id"),
                "condition_type": condition_media.get("condition_type"),
                "users": condition_media.get("users", []),
                "total_duration_seconds": condition_media.get("total_duration_seconds"),
            }
        )
    compact["media_summary"] = media_summary
    return compact


def compact_answerability_for_checkpoint(value: Any) -> dict[str, Any]:
    """Remove repeated per-frame/video provenance from an answerability result."""

    if not isinstance(value, dict):
        return {}
    evaluations = []
    for evaluation in value.get("evaluations") or []:
        if not isinstance(evaluation, dict):
            continue
        compact_evaluation = {
            key: item
            for key, item in evaluation.items()
            if key not in {"condition_media", "raw_output", "initial_raw_output"}
        }
        raw_output = evaluation.get("raw_output")
        if isinstance(raw_output, str):
            compact_evaluation["raw_output_chars"] = len(raw_output)
        condition_media = evaluation.get("condition_media")
        if isinstance(condition_media, dict):
            compact_evaluation["media_summary"] = {
                "media_role": condition_media.get("media_role"),
                "image_count": len(condition_media.get("image_paths") or []),
                "video_count": len(condition_media.get("video_paths") or []),
                "total_duration_seconds": condition_media.get("total_duration_seconds"),
                "exact_media_mapping": "omitted; resolve by evidence_id",
            }
        evaluations.append(compact_evaluation)
    compact = {
        "evaluations": evaluations,
        "gate": value.get("gate", {}),
    }
    if isinstance(value.get("fact_plan"), dict):
        compact["fact_plan"] = dict(value["fact_plan"])
    if isinstance(value.get("user_audits"), list):
        compact["user_audits"] = [
            {
                key: item
                for key, item in audit.items()
                if key not in {"raw_output", "initial_raw_output", "condition_media"}
            }
            for audit in value["user_audits"]
            if isinstance(audit, dict)
        ]
    if isinstance(value.get("minimum_required_users"), list):
        compact["minimum_required_users"] = list(value["minimum_required_users"])
    if isinstance(value.get("trial_order"), list):
        compact["trial_order"] = list(value["trial_order"])
    return compact


def compact_review_for_checkpoint(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    compact = compact_trace_payload(value)
    compact["answerability"] = compact_answerability_for_checkpoint(
        value.get("answerability")
    )
    return compact


def compact_qa_for_checkpoint(value: Any) -> dict[str, Any]:
    """Keep the generated item while dropping embedded audit/trace copies."""

    if not isinstance(value, dict):
        return {}
    compact = {
        key: item
        for key, item in value.items()
        if key not in {"generation_trace", "human_audit", "video_evidence"}
    }
    if "review" in compact:
        compact["review"] = compact_review_for_checkpoint(compact.get("review"))
    return compact


def compact_trace_payload(value: Any) -> Any:
    """Recursively remove prompts, raw generations, and exact media provenance."""

    if isinstance(value, list):
        return [compact_trace_payload(item) for item in value]
    if not isinstance(value, dict):
        return value
    omitted_keys = {
        "prompt",
        "aggregation_prompt",
        "entropy_probe_prompt",
        "raw_output",
        "initial_raw_output",
        "image_paths",
        "video_paths",
        "condition_media",
        "video_evidence",
    }
    compact = {}
    for key, item in value.items():
        if key in omitted_keys:
            if isinstance(item, str):
                compact[f"{key}_chars"] = len(item)
            continue
        compact[key] = compact_trace_payload(item)
    return compact


def compact_judge_trace_for_checkpoint(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    compact = compact_trace_payload({
        key: item
        for key, item in value.items()
        if key
        not in {
            "qa_formality",
            "evidence_groundedness",
            "answerability",
        }
    })
    for branch_name in ("qa_formality", "evidence_groundedness"):
        branch = value.get(branch_name)
        if not isinstance(branch, dict):
            continue
        compact[branch_name] = compact_trace_payload(branch)
        prompt = branch.get("prompt")
        if isinstance(prompt, str):
            compact[branch_name]["prompt_chars"] = len(prompt)
            compact[branch_name]["prompt_recorded_in_prompts_jsonl"] = True
        entropy_prompt = branch.get("entropy_probe_prompt")
        if isinstance(entropy_prompt, str):
            compact[branch_name]["entropy_probe_prompt_chars"] = len(entropy_prompt)
    return compact


def compact_attempt_trace_for_checkpoint(value: Any) -> dict[str, Any]:
    """Project a full attempt trace to a bounded, resume-friendly checkpoint."""

    if not isinstance(value, dict):
        return {}
    compact = {
        key: value.get(key)
        for key in (
            "evidence_id",
            "qa_id",
            "question_type",
            "generation_mode",
            "attempt",
            "feedback_in",
            "generator_decode",
            "schema_errors",
            "judge_entropy",
            "result",
        )
        if value.get(key) is not None
    }
    previous_generation = value.get("previous_generation_in")
    if isinstance(previous_generation, str):
        compact["previous_generation_chars"] = len(previous_generation)

    media = value.get("media") if isinstance(value.get("media"), dict) else {}
    compact["media_summary"] = {
        "generator_image_count": len(media.get("image_paths") or []),
        "generator_video_count": len(media.get("video_paths") or []),
        "judge_image_count": len(media.get("judge_image_paths") or []),
        "judge_video_count": len(media.get("judge_video_paths") or []),
        "generator_media_role": media.get("media_role"),
        "judge_media_role": media.get("judge_media_role"),
        "prepared_video_uploads": bool(media.get("prepared_video_uploads")),
        "exact_media_mapping": "omitted; resolve by evidence_id",
    }

    generation = value.get("generation") if isinstance(value.get("generation"), dict) else {}
    compact_generation = {
        key: item
        for key, item in generation.items()
        if key not in {"prompt", "raw_output", "initial_raw_output"}
    }
    prompt = generation.get("prompt")
    if isinstance(prompt, str):
        compact_generation["prompt_chars"] = len(prompt)
        compact_generation["prompt_recorded_in_prompts_jsonl"] = True
    raw_output = generation.get("raw_output")
    if isinstance(raw_output, str):
        compact_generation["raw_output_chars"] = len(raw_output)
    compact["generation"] = compact_generation
    compact["judge"] = compact_judge_trace_for_checkpoint(value.get("judge"))
    compact["answerability"] = compact_answerability_for_checkpoint(
        value.get("answerability")
    )
    return compact


def intermediate_checkpoint_row(
    *,
    evidence_id: Any,
    question_type: str,
    generation_mode: str,
    status: str,
    attempts: list[dict[str, Any]],
    qa_id: Any = None,
    qa: dict[str, Any] | None = None,
    rejections: list[dict[str, Any]] | None = None,
    reason: str | None = None,
    generator_decode: dict[str, Any] | None = None,
    judge_video_source: str | None = None,
    review: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one compact intermediate JSONL row with no embedded frame manifest."""

    row: dict[str, Any] = {
        "checkpoint_version": 3,
        "evidence_id": evidence_id,
        "qa_id": qa_id,
        "question_type": question_type,
        "generation_mode": generation_mode,
        "status": status,
        "attempt_count": len(attempts),
        "attempts": [compact_attempt_trace_for_checkpoint(item) for item in attempts],
        "evidence_reference": (
            "Use evidence_id to resolve exact frames and videos from the input evidence JSONL; "
            "the intermediate checkpoint intentionally stores no frame manifest."
        ),
    }
    if qa is not None:
        row["qa"] = compact_qa_for_checkpoint(qa)
    if rejections:
        compact_rejections = []
        for rejection in rejections:
            if not isinstance(rejection, dict):
                continue
            compact_rejection = compact_trace_payload(
                {key: item for key, item in rejection.items() if key != "qa"}
            )
            if isinstance(rejection.get("qa"), dict):
                compact_rejection["qa"] = compact_qa_for_checkpoint(
                    rejection.get("qa")
                )
            compact_rejections.append(compact_rejection)
        row["rejections"] = compact_rejections
    if reason:
        row["reason"] = reason
    if generator_decode is not None:
        row["generator_decode"] = generator_decode
    if judge_video_source is not None:
        row["judge_video_source"] = judge_video_source
    if review is not None:
        row["review"] = compact_review_for_checkpoint(review)
    return row


def compact_existing_intermediate_row(row: dict[str, Any]) -> dict[str, Any]:
    """Upgrade one legacy intermediate row to the compact checkpoint contract."""

    if row.get("checkpoint_version") == 3:
        return row
    status = str(row.get("status") or "dry_run")
    qa = row.get("qa") if isinstance(row.get("qa"), dict) else None
    generation_trace = row.get("generation_trace")
    legacy_attempts = row.get("attempts")
    if isinstance(generation_trace, list):
        attempts = generation_trace
        rejections = legacy_attempts if isinstance(legacy_attempts, list) else None
    elif isinstance(legacy_attempts, list):
        attempts = legacy_attempts
        rejections = None
    elif qa is not None and isinstance(qa.get("generation_trace"), list):
        attempts = qa["generation_trace"]
        rejections = None
    else:
        attempts = [row] if "attempt" in row or "stage" in row else []
        rejections = None
    return intermediate_checkpoint_row(
        evidence_id=row.get("evidence_id") or (qa or {}).get("evidence_id"),
        qa_id=row.get("qa_id") or (qa or {}).get("qa_id"),
        question_type=str(row.get("question_type") or (qa or {}).get("question_type") or ""),
        generation_mode=str(
            row.get("generation_mode") or (qa or {}).get("generation_mode") or "baseline"
        ),
        status=status,
        attempts=attempts,
        qa=qa,
        rejections=rejections,
        reason=row.get("reason"),
        generator_decode=row.get("generator_decode"),
        judge_video_source=row.get("judge_video_source"),
        review=row.get("review"),
    )


def compact_existing_jsonl(
    path: str | Path | None,
    transform: Callable[[dict[str, Any]], dict[str, Any]],
) -> None:
    """Atomically compact an existing JSONL before a resume appends new rows."""

    if not path:
        return
    source = Path(path)
    if not source.exists() or source.stat().st_size == 0:
        return
    temporary = source.with_name(f".{source.name}.compact.tmp")
    try:
        write_jsonl(temporary, (transform(row) for row in iter_jsonl(source)))
        temporary.replace(source)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


QUESTION_TYPES = ("commonality", "difference", "neutral")
DEFAULT_QUESTION_TYPES = ("commonality", "difference")
DEFAULT_JUDGE_MODEL_ID = "Qwen/Qwen3.6-27B"
JUDGE_VIDEO_SOURCES = ("full", "pruned")
BLOCKING_JUDGE_CHECKS = (
    "qa_formality",
    "evidence_groundedness",
    "answerability",
)
DIRECT_REVIEW_CHECKS = (
    "qa_formality",
    "evidence_groundedness",
)
QUALITY_SCORED_JUDGE_CHECKS = {
    "qa_formality",
    "evidence_groundedness",
}
# PASS/FAIL entropy is opt-in for production runs. The old detailed judge remains
# the production gate. A second independent call emits only a lowercase verdict;
# that call is diagnostic and cannot affect acceptance, retries, or feedback.
LEGACY_DECISION_ENTROPY_JUDGE_CHECKS = set(QUALITY_SCORED_JUDGE_CHECKS)
FIRST_VERDICT_ENTROPY_VERSION = "first_verdict_detailed_v2"
MINIMAL_VERDICT_ENTROPY_VERSION = "independent_minimal_verdict_v1"
FIRST_VERDICT_FIELD = "verdict"
FIRST_VERDICT_CHOICES = ("pass", "fail")
TEMPORAL_REASONING_MODE = "temporal_reasoning"
MAX_SAFE_PACKETS_IN_FLIGHT = 4
MAX_SAFE_REVIEW_LANES = 3


def verify_first_verdict_tokenization(runner: Any) -> dict[str, Any]:
    """Verify that lowercase pass/fail are single tokens when a tokenizer exists."""

    processor = getattr(runner, "processor", None)
    tokenizer = getattr(processor, "tokenizer", processor)
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        return {
            "checked": False,
            "reason": (
                "runner does not expose a local tokenizer; the generated-token "
                "capture will validate the choices at response time"
            ),
        }
    choices = {}
    for choice in FIRST_VERDICT_CHOICES:
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


def quality_score_value(value: Any) -> int | None:
    """Return a valid integer quality score without changing the judge decision."""

    if isinstance(value, bool):
        return None
    try:
        score = int(value)
    except (TypeError, ValueError):
        return None
    return score if score in {1, 2, 3} else None


def quality_quota_snapshot(previous: int, quota: int) -> dict[str, int]:
    """Capture the prompt-time state for one judge category."""

    limit = max(1, int(quota))
    observed = max(0, int(previous))
    return {
        "quota": limit,
        "previous_three_point_assignments": observed,
        "remaining_before_candidate": max(0, limit - observed),
    }


def attach_quality_quota_metadata(
    check: dict[str, Any],
    *,
    quota_state: dict[str, int],
) -> dict[str, Any]:
    """Audit score rationale and post-quota rebuttal without gating acceptance."""

    score = quality_score_value(check.get("quality_score"))
    if score is not None:
        check["quality_score"] = score
    previous = int(quota_state["previous_three_point_assignments"])
    quota = int(quota_state["quota"])
    assigned_three = score == 3
    exceeded = bool(assigned_three and previous >= quota)
    quality_reason_present = bool(str(check.get("quality_reason") or "").strip())
    quota_rebuttal_present = bool(str(check.get("quota_rebuttal") or "").strip())
    if not exceeded and check.get("quota_rebuttal") is None:
        check["quota_rebuttal"] = ""
    check["quality_quota"] = {
        **quota_state,
        "assigned_score": score,
        "assigned_three_points": assigned_three,
        "quota_exceeded_by_this_assignment": exceeded,
        "quality_reason_present": quality_reason_present,
        "quota_rebuttal_required": exceeded,
        "quota_rebuttal_present": quota_rebuttal_present,
        "output_contract_satisfied": bool(
            score is not None
            and quality_reason_present
            and (not exceeded or quota_rebuttal_present)
        ),
        "acceptance_effect": "none; PASS/FAIL and answerability remain the only gates",
    }
    return check


def quality_quota_counts_from_rows(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Restore observed 3-point assignments from canonical run traces on resume."""

    restored_compact_counts = {
        check_name: 0 for check_name in QUALITY_SCORED_JUDGE_CHECKS
    }
    trace_counts = {check_name: 0 for check_name in QUALITY_SCORED_JUDGE_CHECKS}
    for row in rows:
        compact_config = (
            row.get("production_judge_config")
            if isinstance(row.get("production_judge_config"), dict)
            else {}
        )
        row_compact_counts = compact_config.get("observed_three_point_assignments")
        if isinstance(row_compact_counts, dict):
            for check_name in QUALITY_SCORED_JUDGE_CHECKS:
                try:
                    restored_compact_counts[check_name] = max(
                        restored_compact_counts[check_name],
                        int(row_compact_counts.get(check_name, 0)),
                    )
                except (TypeError, ValueError):
                    pass
        traces = row.get("generation_trace")
        if not isinstance(traces, list):
            traces = row.get("attempts")
        if not isinstance(traces, list):
            nested_qa = row.get("qa") if isinstance(row.get("qa"), dict) else {}
            traces = nested_qa.get("generation_trace")
        if not isinstance(traces, list):
            continue
        for trace in traces:
            if not isinstance(trace, dict):
                continue
            judge_trace = trace.get("judge")
            if not isinstance(judge_trace, dict):
                continue
            merged = judge_trace.get("merged")
            checks = merged.get("checks") if isinstance(merged, dict) else None
            if not isinstance(checks, dict):
                continue
            for check_name in QUALITY_SCORED_JUDGE_CHECKS:
                check = checks.get(check_name)
                if isinstance(check, dict) and quality_score_value(check.get("quality_score")) == 3:
                    trace_counts[check_name] += 1
    # A row may contain both the cumulative compact counter and the canonical traces.
    # They describe the same assignments, so take the larger reconstruction rather
    # than adding them and double-counting the pre-resume quota state.
    return {
        check_name: max(restored_compact_counts[check_name], trace_counts[check_name])
        for check_name in QUALITY_SCORED_JUDGE_CHECKS
    }


def existing_path(value: str | None) -> str | None:
    if not value:
        return None
    path = Path(value)
    if path.exists():
        return str(path)
    return None


def clip_video_path(clip: dict[str, Any], *, media_role: str = "generator") -> str | None:
    if media_role == "full":
        for key in ("full_local_video", "original_local_video", "source_local_video", "local_video"):
            if path := existing_path(clip.get(key)):
                return path
        return None
    return existing_path(clip.get("local_video"))


def clip_image_paths(clip: dict[str, Any]) -> list[str]:
    paths = []
    for frame in clip.get("frames", []):
        path = existing_path(frame.get("path"))
        if path:
            paths.append(path)
    return paths


def clips_require_frame_inputs(clips: list[dict[str, Any]]) -> bool:
    return any(
        clip.get("generator_media_mode") == "frames_only" or clip.get("force_frame_inputs")
        for clip in clips
    )


def media_for_clips(
    clips: list[dict[str, Any]],
    *,
    backend: str,
    allow_openai_video_input: bool,
    media_role: str = "generator",
) -> tuple[list[str], list[str]]:
    videos = [path for clip in clips if (path := clip_video_path(clip, media_role=media_role))]
    images = [path for clip in clips for path in clip_image_paths(clip)]
    if media_role == "generator" and clips_require_frame_inputs(clips):
        return images, []
    if backend in {"openai-compatible-local", "openrouter"} and not allow_openai_video_input:
        return images, []
    return images if not videos else [], videos


def prepare_runner_video_uploads(
    *,
    runner: Any,
    evidence_id: Any,
    generator_video_paths: list[str],
    full_video_paths: list[str],
    judge_media_role: str = "full",
) -> dict[str, Any] | None:
    """Let remote runners pre-upload all packet videos before generation starts."""

    prepare_videos = getattr(runner, "prepare_videos", None)
    if not callable(prepare_videos):
        return None
    all_video_paths = list(dict.fromkeys([*generator_video_paths, *full_video_paths]))
    if not all_video_paths:
        return None
    stage_start = time.time()
    print(
        "qa_stage_start "
        f"stage=prepare_media evidence_id={evidence_id} "
        f"generator_videos={len(generator_video_paths)} "
        f"judge_videos={len(full_video_paths)} "
        f"judge_media_role={judge_media_role} "
        f"unique_videos={len(all_video_paths)}",
        flush=True,
    )
    prepared = prepare_videos(all_video_paths)
    print(
        "qa_stage_done "
        f"stage=prepare_media evidence_id={evidence_id} "
        f"seconds={time.time() - stage_start:.1f} "
        f"prepared_videos={len(prepared or [])}",
        flush=True,
    )
    return {
        "stage": "prepare_media",
        "generator_video_paths": generator_video_paths,
        "full_video_paths": full_video_paths,
        "judge_video_paths": full_video_paths,
        "judge_media_role": judge_media_role,
        "unique_video_paths": all_video_paths,
        "prepared_video_count": len(prepared or []),
        "purpose": (
            "pre-upload generator media and the explicitly selected visual-judge media before "
            "generation starts"
        ),
    }


def time_map_segments_from_keep_intervals(
    keep_intervals: list[list[float]] | list[tuple[float, float]] | None,
) -> list[dict[str, float]]:
    """Map concatenated pruned-video time back to original-video time."""

    segments = []
    pruned_cursor = 0.0
    for interval in keep_intervals or []:
        if not isinstance(interval, (list, tuple)) or len(interval) < 2:
            continue
        original_start = float(interval[0])
        original_end = float(interval[1])
        if original_end <= original_start:
            continue
        duration = original_end - original_start
        pruned_start = pruned_cursor
        pruned_end = pruned_cursor + duration
        segments.append(
            {
                "pruned_start_seconds": round(pruned_start, 3),
                "pruned_end_seconds": round(pruned_end, 3),
                "original_start_seconds": round(original_start, 3),
                "original_end_seconds": round(original_end, 3),
            }
        )
        pruned_cursor = pruned_end
    return segments


def _temporal_keep_intervals_for_clip(clip: dict[str, Any]) -> list[list[float]] | list[tuple[float, float]]:
    pruning = clip.get("temporal_pruning")
    if not isinstance(pruning, dict):
        return []
    keep_intervals = pruning.get("keep_intervals")
    if isinstance(keep_intervals, list):
        return keep_intervals
    return []


def packet_with_temporal_reasoning_media(packet: dict[str, Any]) -> dict[str, Any]:
    """Return a packet whose prompt metadata exposes original timestamps.

    This is intentionally opt-in for temporal_reasoning mode. Other modes use
    the input packet unchanged, so no original_timestamp metadata leaks into
    neutral/baseline prompts or media traces. Discovery modes are archived.
    """

    updated = dict(packet)
    updated["generation_mode"] = TEMPORAL_REASONING_MODE
    clips = []
    for index, clip in enumerate(packet.get("clips", [])):
        next_clip = dict(clip)
        keep_intervals = _temporal_keep_intervals_for_clip(next_clip)
        time_map_segments = time_map_segments_from_keep_intervals(keep_intervals)
        local_video = next_clip.get("local_video")
        pruned_video = existing_path(local_video) or (str(local_video) if local_video else None)
        if time_map_segments and pruned_video:
            next_clip["temporal_reasoning"] = {
                "enabled": True,
                "mapping_type": "contiguous_interval_map",
                "generator_video": pruned_video,
                "time_map_segments": time_map_segments,
                "instruction": (
                    "Each time_map_segments row says that the contiguous pruned-video interval "
                    "[pruned_start_seconds, pruned_end_seconds] corresponds to the original-video "
                    "interval [original_start_seconds, original_end_seconds]. Use these intervals "
                    "to reason about original temporal order and jumps."
                ),
            }
            next_clip["generator_media_mode"] = "temporal_reasoning_pruned_video_with_sidecar_time_map"
        clips.append(next_clip)
    updated["clips"] = clips
    return updated


def video_evidence_for_packet(packet: dict[str, Any]) -> list[dict[str, Any]]:
    """Return deterministic clip/video provenance for the generated question-answer row."""

    rows = []
    for clip in packet.get("clips", []):
        local_video = clip.get("local_video")
        rows.append(
            {
                "user": clip.get("agent_name"),
                "agent_dir": clip.get("agent_dir"),
                "agent_id": clip.get("agent_id"),
                "day": clip.get("day"),
                "time_token": clip.get("time_token"),
                "clip_clock": clip.get("clip_clock"),
                "duration_seconds": clip.get("duration_seconds"),
                "segment_count": clip.get("segment_count"),
                "video_url": clip.get("video_url"),
                "source_video_urls": clip.get("source_video_urls"),
                "local_video": local_video,
                "local_video_exists": bool(existing_path(local_video)),
                "source_local_video": clip.get("source_local_video"),
                "original_local_video": clip.get("original_local_video"),
                "original_local_video_exists": bool(existing_path(clip.get("original_local_video"))),
                "full_local_video": clip.get("full_local_video"),
                "full_local_video_exists": bool(existing_path(clip.get("full_local_video"))),
                "benchmark_media": clip.get("benchmark_media"),
                "generator_media_mode": clip.get("generator_media_mode"),
                "generator_local_video": clip.get("generator_local_video"),
                "media_role": clip.get("media_role"),
                "is_pruned": clip.get("is_pruned"),
                "temporal_pruning": clip.get("temporal_pruning"),
                "temporal_reasoning": clip.get("temporal_reasoning"),
                "gaze_url": clip.get("gaze_url"),
                "source_gaze_urls": clip.get("source_gaze_urls"),
                "gaze_summary": clip.get("gaze_summary"),
                "source_segments": clip.get("source_segments"),
                "sampled_frames": [
                    {
                        "timestamp_seconds": frame.get("timestamp_seconds"),
                        "path": frame.get("path"),
                        "path_exists": bool(existing_path(frame.get("path"))),
                    }
                    for frame in clip.get("frames", [])
                ],
            }
        )
    return rows


def six_user_role_metadata(
    packet: dict[str, Any],
    required_users: list[str],
) -> dict[str, Any]:
    """读取并校验六用户 packet 的显式角色合同。"""

    if len(required_users) != 6:
        return {}

    expected = {
        "input_users": required_users,
        "speaker_user": required_users[0],
        "provider_users": required_users[1:],
        "evidence_provider_user": required_users[1],
        "evidence_provider_users": required_users[1:],
    }
    for field, expected_value in expected.items():
        actual_value = packet.get(field)
        if actual_value != expected_value:
            raise ValueError(
                f"six-user packet {field} must equal ordered required_users contract: "
                f"expected {expected_value!r}, got {actual_value!r}"
            )

    expected_media_roles = {
        required_users[0]: "speaker_all_clustering_frames",
        required_users[1]: "provider_retained_cluster_frames",
        required_users[2]: "provider_retained_cluster_frames",
        required_users[3]: "provider_retained_cluster_frames",
        required_users[4]: "provider_retained_cluster_frames",
        required_users[5]: "provider_retained_cluster_frames",
    }
    media_roles = packet.get("media_roles")
    if media_roles != expected_media_roles:
        raise ValueError(
            "six-user packet media_roles must cover the ordered speaker and provider "
            f"roles: expected {expected_media_roles!r}, got {media_roles!r}"
        )

    return {**expected, "media_roles": dict(expected_media_roles)}


def human_audit_packet(packet: dict[str, Any]) -> dict[str, Any]:
    """Compact evidence bundle intended for manual review of one generated question-answer item."""

    required_users = list(packet.get("required_users") or [])
    speaker_user = required_users[0] if required_users else None
    evidence_provider_user = required_users[1] if len(required_users) > 1 else None
    role_metadata = six_user_role_metadata(packet, required_users)
    review_instructions = [
        "Open each listed full_local_video, local_video, or video_url for the required users.",
        "Check the referred_timestamps and per_user_evidence_claims against the visible content.",
    ]
    if len(required_users) == 6:
        review_instructions.extend(
            [
                "Verify that the speaker video alone lacks enough visible evidence to answer the question.",
                "Verify that the six full input videos together contain enough visible evidence to answer the question.",
                "Assess evidence sufficiency without asking the answerability judge to select an option.",
                "Do not require every provider to contribute evidence.",
            ]
        )
    else:
        review_instructions.extend(
            [
                "Verify that required_users[0], the asker, cannot answer from their own video alone.",
                "If required_users[1], the evidence provider, can answer alone, confirm that this is logged in review.answerability.gate.evidence_provider_answerable.",
            ]
        )

    return {
        "evidence_id": packet.get("evidence_id"),
        "required_users": required_users,
        "speaker_user": speaker_user,
        "evidence_provider_user": evidence_provider_user,
        **role_metadata,
        "requirement": packet.get("requirement"),
        "source_urls": packet.get("source_urls", {}),
        "video_evidence": video_evidence_for_packet(packet),
        "review_instructions": review_instructions,
    }


def complete_generator_metadata(
    qa: dict[str, Any],
    *,
    packet: dict[str, Any],
    question_type: str,
) -> dict[str, Any]:
    """Fill review metadata that the generator may omit before the real gates run."""

    # Category selection is an offline analysis concern. Strip legacy or
    # hallucinated category keys so production artifacts remain category-free.
    qa.pop("category", None)
    qa.pop("category_rationale", None)
    required_users = list(packet.get("required_users") or qa.get("required_users") or [])
    role_metadata = six_user_role_metadata(packet, required_users)
    qa["question_type"] = question_type
    qa["required_users"] = required_users
    qa.update(role_metadata)
    qa.setdefault("referred_timestamps", [])
    if not isinstance(qa.get("referred_timestamps"), list):
        qa["referred_timestamps"] = []

    try:
        correct = normalize_correct(qa.get("correct"))
        qa["correct"] = correct
        options = qa.get("options")
        if isinstance(options, list) and len(options) == len(OPTION_LETTERS):
            qa["answer"] = options[OPTION_LETTERS.index(correct)]
    except ValueError:
        pass

    single = qa.get("single_user_answerability")
    if not isinstance(single, dict):
        single = {}
    asker_user = required_users[0] if required_users else None
    evidence_provider_user = required_users[1] if len(required_users) > 1 else None
    for index, user in enumerate(required_users):
        text = str(single.get(user, "")).strip()
        if index == 0 and (
            not text or not any(marker in text.lower() for marker in ("insufficient", "cannot", "not enough"))
        ):
            single[user] = (
                "insufficient because the asker's video alone does not provide "
                "the missing visual detail from the evidence provider"
            )
        elif index > 0 and not text:
            if len(required_users) == 6:
                single[user] = (
                    "not preclassified; assess only this provider's view and report truthfully "
                    "whether it independently supports the answer"
                )
            else:
                single[user] = (
                    "may be sufficient because this user is the evidence provider; "
                    "answerability is logged by the evaluator"
                )
    qa["single_user_answerability"] = single

    combined = str(qa.get("combined_answerability", "")).strip()
    if "sufficient" not in combined.lower() and "support" not in combined.lower():
        qa["combined_answerability"] = (
            "sufficient because the six-video input combines the speaker-side reason to ask "
            "with answer-bearing evidence from one or more provider views"
            if len(required_users) == 6
            else "sufficient because combining the required users' videos provides "
            "the speaker-side anchor event plus the missing visual detail needed "
            "to select exactly one option"
        )

    if not qa.get("generator_rationale"):
        qa["generator_rationale"] = (
            "The question is a natural first-person information need grounded in the full "
            "speaker experience, while one or more provider views supply evidence the speaker "
            "does not have."
            if len(required_users) == 6
            else "The question is framed as a natural first-person memory gap anchored "
            "in the asker's experience and answered with another user's visual evidence."
        )
    if not qa.get("why_two_users_needed"):
        if len(required_users) == 6:
            qa["why_two_users_needed"] = (
                "The speaker needs at least one external view because the speaker video alone "
                "does not contain the answer-bearing evidence; the six-video input supplies it."
            )
        else:
            qa["why_two_users_needed"] = (
                "At least two required users are needed because the first required user supplies "
                "the speaker-side anchor event while the second required user supplies the missing "
                "visual detail."
            )
    claims = qa.get("per_user_evidence_claims")
    if not isinstance(claims, list) or not claims:
        claims = []
        if len(required_users) != 6:
            for user in required_users:
                claims.append(
                    {
                        "user": user,
                        "claim": f"{user}'s own video contributes a necessary visual fact listed in the evidence field.",
                    }
                )
        qa["per_user_evidence_claims"] = claims
    if len(required_users) == 6 and not isinstance(qa.get("supporting_user_claims"), list):
        qa["supporting_user_claims"] = [
            dict(row)
            for row in claims
            if isinstance(row, dict) and row.get("user") != asker_user
        ]

    review = qa.get("review")
    if not isinstance(review, dict):
        review = {}
    review.setdefault(
        "generator_self_check",
        (
            "This draft should express a question the speaker would naturally want to ask after "
            "their own experience. The full speaker video should motivate the question without "
            "revealing the answer; one or more provider views should supply the missing evidence."
            if len(required_users) == 6
            else "This draft should be unanswerable from the first required user's video alone; "
            "the second required user's video may contain the answer as evidence-provider context."
        ),
    )
    review.setdefault("speaker_user", asker_user)
    review.setdefault("evidence_provider_user", evidence_provider_user)
    if len(required_users) == 6:
        review.setdefault("evidence_provider_users", required_users[1:])
        review.setdefault("provider_users", required_users[1:])
    review.setdefault("status", "draft")
    qa["review"] = review
    return qa


def condition_media_for_clips(
    *,
    condition: dict[str, Any],
    clips: list[dict[str, Any]],
    image_paths: list[str],
    video_paths: list[str],
    media_role: str = "generator",
) -> dict[str, Any]:
    total_duration_seconds = round(
        sum(
            float(clip.get("duration_seconds") or 0.0)
            for clip in clips
            if isinstance(clip, dict)
        ),
        3,
    )
    return {
        "condition_id": condition.get("condition_id"),
        "condition_type": condition.get("condition_type"),
        "users": condition.get("users", []),
        "media_role": media_role,
        "image_paths": image_paths,
        "video_paths": video_paths,
        "total_duration_seconds": total_duration_seconds,
        "video_evidence": video_evidence_for_packet({"clips": clips}),
    }


def qa_for_judger_prompt(
    qa: dict[str, Any],
    *,
    include_generator_rationale: bool = False,
) -> dict[str, Any]:
    """Return only independently judgeable candidate fields.

    ``include_generator_rationale`` is retained for call compatibility, but the
    generator's interpretation is deliberately never supplied to a reviewer.
    """

    wanted = [
        "qa_id",
        "evidence_id",
        "question_type",
        "question",
        "options",
        "correct",
        "answer",
        "required_users",
        # Generator-authored evidence fields remain excluded because they can anchor
        # judges to a mistaken interpretation instead of letting them inspect the media
        # independently.
        # "evidence",
        # "single_user_answerability",
        # "combined_answerability",
        # "why_two_users_needed",
        # "per_user_evidence_claims",
        # "referred_timestamps",
        # "review",
    ]
    return {key: qa[key] for key in wanted if key in qa}


def clips_for_users(packet: dict[str, Any], users: list[str]) -> list[dict[str, Any]]:
    wanted = set(users)
    return [clip for clip in packet.get("clips", []) if clip.get("agent_name") in wanted]


def answerability_qa_for_prompt(qa: dict[str, Any]) -> dict[str, Any]:
    """Remove gold answer fields before six-user sufficiency planning/auditing."""

    return {
        key: qa[key]
        for key in ("qa_id", "question", "options", "required_users")
        if key in qa
    }


def ordered_source_segment_media(clip: dict[str, Any]) -> tuple[list[str], int]:
    """Return existing ordered source segments, falling back to one full video.

    The returned paths are used only as model attachments.  They are deliberately
    never rendered into a prompt or persisted in a compact checkpoint.
    """

    source_segments = clip.get("source_segments")
    if isinstance(source_segments, list) and source_segments:
        indexed_rows = []
        for fallback_index, row in enumerate(source_segments):
            if not isinstance(row, dict):
                continue
            path_value = row.get("local_video")
            if not path_value or not Path(path_value).is_file():
                continue
            try:
                segment_index = int(row.get("segment_index", fallback_index))
            except (TypeError, ValueError):
                segment_index = fallback_index
            indexed_rows.append((segment_index, str(path_value)))
        if len(indexed_rows) != len(source_segments):
            missing_count = len(source_segments) - len(indexed_rows)
            raise FileNotFoundError(
                f"{clip.get('agent_name')} is missing {missing_count} cached source segment(s)"
            )
        indexed_rows.sort(key=lambda item: item[0])
        return [path for _, path in indexed_rows], len(indexed_rows)

    full_video = clip_video_path(clip, media_role="full")
    if not full_video or not Path(full_video).is_file():
        raise FileNotFoundError(
            f"{clip.get('agent_name')} has neither cached source segments nor a full video"
        )
    return [str(full_video)], 1


def six_user_source_segment_media(
    packet: dict[str, Any],
    required_users: list[str],
) -> dict[str, dict[str, Any]]:
    """Resolve each required user's independent ordered visual sequence."""

    clips_by_user = {
        str(clip.get("agent_name")): clip
        for clip in packet.get("clips", [])
        if isinstance(clip, dict) and clip.get("agent_name")
    }
    media = {}
    for user in required_users:
        clip = clips_by_user.get(str(user))
        if clip is None:
            raise ValueError(f"missing packet clip for required user {user}")
        video_paths, segment_count = ordered_source_segment_media(clip)
        media[str(user)] = {
            "video_paths": video_paths,
            "segment_count": segment_count,
        }
    return media


def parse_question_types(value: str | None) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_QUESTION_TYPES
    question_types = tuple(part.strip() for part in value.split(",") if part.strip())
    if not question_types:
        raise ValueError("question_types must include at least one question type")
    unknown = [question_type for question_type in question_types if question_type not in QUESTION_TYPES]
    if unknown:
        raise ValueError(f"unknown question_types: {unknown}")
    return question_types


def target_type_counts(target_count: int, question_types: tuple[str, ...] = DEFAULT_QUESTION_TYPES) -> dict[str, int]:
    base, remainder = divmod(target_count, len(question_types))
    return {
        question_type: base + (1 if index < remainder else 0)
        for index, question_type in enumerate(question_types)
    }


def choose_question_type(
    counts: dict[str, int],
    targets: dict[str, int],
    question_types: tuple[str, ...] = DEFAULT_QUESTION_TYPES,
) -> str | None:
    remaining = {
        question_type: targets[question_type] - counts.get(question_type, 0)
        for question_type in question_types
    }
    remaining = {key: value for key, value in remaining.items() if value > 0}
    if not remaining:
        return None
    return sorted(remaining.items(), key=lambda item: (-item[1], item[0]))[0][0]


def build_answerability_conditions(required_users: list[str]) -> list[dict[str, Any]]:
    users = list(required_users)
    if len(users) == 6:
        return [
            {
                "condition_id": f"speaker_only::{users[0]}",
                "condition_type": "speaker_only",
                "users": [users[0]],
            },
            {
                "condition_id": "combined_all_six_users::" + "+".join(users),
                "condition_type": "combined_all_six_users",
                "users": users,
            },
        ]
    conditions = [
        {
            "condition_id": f"single_user::{user}",
            "condition_type": "single_user",
            "users": [user],
        }
        for user in users
    ]
    if len(users) > 2:
        for size in range(2, len(users)):
            for combo in itertools.combinations(users, size):
                combo_users = list(combo)
                conditions.append(
                    {
                        "condition_id": "proper_subset::" + "+".join(combo_users),
                        "condition_type": "proper_subset",
                        "users": combo_users,
                    }
                )
    conditions.append(
        {
            "condition_id": "combined_all_users::" + "+".join(users),
            "condition_type": "combined_all_users",
            "users": users,
        }
    )
    return conditions


def parsed_choice(value: Any) -> tuple[str | None, bool]:
    text = str(value or "").strip().upper()
    if text in OPTION_LETTERS:
        return text, False
    return None, True


def validated_answerability_fact_plan(
    value: Any,
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate and freeze a consecutive, answer-neutral F1..Fn fact plan."""

    if not isinstance(value, dict):
        return None, "fact plan must be a JSON object"
    if set(value) != {"reason", "needed_facts"}:
        return None, "fact plan must contain only reason and needed_facts"
    if not str(value.get("reason") or "").strip():
        return None, "fact plan reason must be a non-empty string"
    facts = value.get("needed_facts")
    if not isinstance(facts, list) or not facts:
        return None, "needed_facts must be a non-empty array"
    expected_ids = [f"F{index}" for index in range(1, len(facts) + 1)]
    actual_ids = []
    for fact in facts:
        if not isinstance(fact, dict):
            return None, "every needed_fact must be an object"
        if set(fact) != {"fact_id", "fact", "why_needed"}:
            return None, "every needed_fact must contain only fact_id, fact, and why_needed"
        actual_ids.append(str(fact.get("fact_id") or ""))
        if not str(fact.get("fact") or "").strip():
            return None, "every fact must be a non-empty string"
        if not str(fact.get("why_needed") or "").strip():
            return None, "every why_needed must be a non-empty string"
    if actual_ids != expected_ids:
        return None, f"fact IDs must be consecutive: {expected_ids}"
    return {
        "reason": str(value["reason"]).strip(),
        "needed_facts": [dict(fact) for fact in facts],
    }, None


def validated_answerability_fact_audit(
    value: Any,
    *,
    expected_fact_ids: list[str],
    allowed_users: list[str],
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate one per-user or condition-level frozen-fact visibility audit."""

    if not isinstance(value, dict):
        return None, "fact audit must be a JSON object"
    forbidden_fields = [
        key
        for key in ("answerable", "choice", "answer", "answer_text", "option")
        if key in value
    ]
    if forbidden_fields:
        return None, "response included forbidden answer fields: " + ", ".join(
            forbidden_fields
        )
    if set(value) != {"reason", "fact_audits"}:
        return None, "fact audit must contain only reason and fact_audits"
    if not str(value.get("reason") or "").strip():
        return None, "fact audit reason must be a non-empty string"
    audits = value.get("fact_audits")
    if not isinstance(audits, list) or not audits:
        return None, "fact_audits must be a non-empty array"
    actual_ids = [
        str(row.get("fact_id") or "") if isinstance(row, dict) else ""
        for row in audits
    ]
    if actual_ids != expected_fact_ids:
        return None, (
            "fact_audits must preserve the frozen fact IDs and order: "
            f"{expected_fact_ids}"
        )
    allowed_user_set = set(allowed_users)
    compact_rows = []
    required_keys = {
        "fact_id",
        "visibility",
        "source_users",
        "segment_references",
        "visual_description",
    }
    for row in audits:
        if set(row) != required_keys:
            return None, f"fact audit {row.get('fact_id')} has unexpected fields"
        visibility = str(row.get("visibility") or "")
        if visibility not in {"VISIBLE", "NOT_VISIBLE", "AMBIGUOUS"}:
            return None, f"fact audit {row.get('fact_id')} has invalid visibility"
        source_users = row.get("source_users")
        if not isinstance(source_users, list) or any(
            not isinstance(user, str) or user not in allowed_user_set
            for user in source_users
        ):
            return None, f"fact audit {row.get('fact_id')} has invalid source_users"
        if visibility == "VISIBLE" and not source_users:
            return None, f"VISIBLE fact {row.get('fact_id')} requires source_users"
        if visibility != "VISIBLE" and source_users:
            return None, f"non-visible fact {row.get('fact_id')} must not name source_users"
        references = row.get("segment_references")
        if not isinstance(references, list) or any(
            not isinstance(reference, str)
            or re.fullmatch(r"segment_[0-9]{3,}", reference) is None
            for reference in references
        ):
            return None, f"fact audit {row.get('fact_id')} has invalid segment references"
        if not str(row.get("visual_description") or "").strip():
            return None, f"fact audit {row.get('fact_id')} needs a visual_description"
        compact_rows.append(dict(row))
    return {
        "reason": str(value["reason"]).strip(),
        "fact_audits": compact_rows,
    }, None


def parsed_answerability_sufficiency(
    evaluation: dict[str, Any],
) -> tuple[bool | None, str | None]:
    """Derive sufficiency only after validating the shared frozen-fact audit."""

    forbidden_fields = [
        key
        for key in ("answerable", "choice", "answer", "answer_text", "option")
        if key in evaluation
    ]
    if forbidden_fields:
        return None, "response included forbidden answer fields: " + ", ".join(
            forbidden_fields
        )
    expected_fact_ids = evaluation.get("shared_fact_ids")
    condition_users = evaluation.get("users")
    if not isinstance(expected_fact_ids, list) or not expected_fact_ids:
        return None, "shared_fact_ids must be a non-empty array"
    if not isinstance(condition_users, list) or not condition_users:
        return None, "condition users must be a non-empty array"
    audit_value = {
        "reason": evaluation.get("reason"),
        "fact_audits": evaluation.get("fact_audits"),
    }
    validated, error = validated_answerability_fact_audit(
        audit_value,
        expected_fact_ids=[str(fact_id) for fact_id in expected_fact_ids],
        allowed_users=[str(user) for user in condition_users],
    )
    if error or validated is None:
        return None, error or "fact audit could not be validated"
    sufficient = all(
        row.get("visibility") == "VISIBLE"
        for row in validated["fact_audits"]
    )
    return sufficient, None


def minimum_required_users_from_fact_audits(
    *,
    required_users: list[str],
    user_audits: list[dict[str, Any]],
    fact_ids: list[str],
) -> list[str]:
    """Choose the smallest speaker-inclusive user set covering all frozen facts."""

    if len(required_users) != 6:
        raise ValueError("minimum six-user factual set requires exactly six users")
    required_facts = {str(fact_id) for fact_id in fact_ids}
    visible_by_user: dict[str, set[str]] = {}
    for audit in user_audits:
        user = str(audit.get("user") or "")
        visible_by_user[user] = {
            str(row.get("fact_id"))
            for row in audit.get("fact_audits") or []
            if isinstance(row, dict) and row.get("visibility") == "VISIBLE"
        }

    speaker = required_users[0]
    speaker_facts = visible_by_user.get(speaker, set())
    providers = required_users[1:]
    for provider_count in range(1, len(providers) + 1):
        for provider_subset in itertools.combinations(providers, provider_count):
            covered = set(speaker_facts)
            for provider in provider_subset:
                covered.update(visible_by_user.get(provider, set()))
            if required_facts.issubset(covered):
                return [speaker, *provider_subset]
    return list(required_users)


def answerability_gate(
    qa_item: dict[str, Any],
    evaluations: list[dict[str, Any]],
    *,
    six_user_judge_mode: str = SIX_USER_JUDGE_MODE_TIME_AWARE,
) -> dict[str, Any]:
    required_users = list(qa_item.get("required_users") or [])
    if len(required_users) == 6:
        if six_user_judge_mode == SIX_USER_JUDGE_MODE_LEGACY:
            speaker_rows = [
                row
                for row in evaluations
                if row.get("condition_type") == "speaker_only"
            ]
            all_six_rows = [
                row
                for row in evaluations
                if row.get("condition_type") == "combined_all_six_users"
            ]
            base = {
                "answerability_mode": "direct_video_sufficiency_zero_shot",
                "answerability_evaluated_condition_count": len(evaluations),
            }
            if not speaker_rows:
                return {
                    "passed": False,
                    "reason": "missing speaker-only evaluation",
                    "failure_label": "speaker_only_missing",
                    **base,
                }
            if not all_six_rows:
                return {
                    "passed": False,
                    "reason": "missing all-six evaluation",
                    "failure_label": "all_six_missing",
                    **base,
                }

            def parsed_direct_sufficiency(
                row: dict[str, Any],
            ) -> tuple[bool | None, str | None]:
                forbidden_fields = [
                    key
                    for key in ("choice", "answer", "answer_text", "option", "correct")
                    if key in row
                ]
                if forbidden_fields:
                    return None, "response included forbidden answer fields: " + ", ".join(
                        forbidden_fields
                    )
                answerable = row.get("answerable")
                if not isinstance(answerable, bool):
                    return None, "answerable must be a JSON boolean"
                return answerable, None

            speaker_answerable, speaker_error = parsed_direct_sufficiency(
                speaker_rows[-1]
            )
            if speaker_error:
                return {
                    "passed": False,
                    "reason": f"invalid speaker-only sufficiency evaluation: {speaker_error}",
                    "failure_label": "speaker_only_unparsed",
                    "speaker_only_answerable": None,
                    **base,
                }
            metrics = {"speaker_only_answerable": speaker_answerable, **base}
            if speaker_answerable:
                return {
                    "passed": False,
                    "reason": "speaker-only video was judged sufficient to answer the question",
                    "failure_label": "speaker_only_answerable",
                    **metrics,
                }
            all_six_answerable, all_six_error = parsed_direct_sufficiency(
                all_six_rows[-1]
            )
            if all_six_error:
                return {
                    "passed": False,
                    "reason": f"invalid all-six sufficiency evaluation: {all_six_error}",
                    "failure_label": "all_six_unparsed",
                    "all_six_answerable": None,
                    **metrics,
                }
            combined_metrics = {
                "all_six_answerable": all_six_answerable,
                **metrics,
            }
            if not all_six_answerable:
                return {
                    "passed": False,
                    "reason": "all-six videos were judged insufficient to answer the question",
                    "failure_label": "all_six_not_answerable",
                    **combined_metrics,
                }
            return {
                "passed": True,
                "reason": (
                    "speaker-only video was judged insufficient and the combined six "
                    "videos were judged sufficient"
                ),
                "failure_label": None,
                **combined_metrics,
            }
        if six_user_judge_mode not in {
            SIX_USER_JUDGE_MODE_TIME_AWARE,
            SIX_USER_JUDGE_MODE_SEQUENTIAL,
        }:
            raise ValueError(f"unknown six_user_judge_mode: {six_user_judge_mode}")
        speaker_rows = [
            row for row in evaluations if row.get("condition_type") == "speaker_only"
        ]
        all_six_rows = [
            row
            for row in evaluations
            if row.get("condition_type") == "combined_all_six_users"
        ]
        minimum_rows = [
            row
            for row in evaluations
            if row.get("condition_type") == "minimum_required_users"
        ]
        base = {
            "answerability_mode": "shared_fact_visibility_audit",
            "answerability_evaluated_condition_count": len(evaluations),
        }
        if not speaker_rows:
            return {
                "passed": False,
                "reason": "missing speaker-only evaluation",
                "failure_label": "speaker_only_missing",
                **base,
            }

        speaker_answerable, speaker_error = parsed_answerability_sufficiency(
            speaker_rows[-1]
        )
        if speaker_error:
            return {
                "passed": False,
                "reason": f"invalid speaker-only sufficiency evaluation: {speaker_error}",
                "failure_label": "speaker_only_unparsed",
                "speaker_only_answerable": None,
                **base,
            }

        metrics = {
            "speaker_only_answerable": speaker_answerable,
            **base,
        }
        if speaker_answerable:
            return {
                "passed": False,
                "reason": "speaker-only videos were judged sufficient to answer the question",
                "failure_label": "speaker_only_answerable",
                **metrics,
            }
        if not all_six_rows:
            return {
                "passed": False,
                "reason": "missing all-six evaluation",
                "failure_label": "all_six_missing",
                **metrics,
            }
        speaker_fact_ids = speaker_rows[-1].get("shared_fact_ids")
        all_six_fact_ids = all_six_rows[-1].get("shared_fact_ids")
        if speaker_fact_ids != all_six_fact_ids:
            return {
                "passed": False,
                "reason": (
                    "speaker-only and all-six evaluations did not use the same frozen facts"
                ),
                "failure_label": "shared_fact_plan_mismatch",
                **metrics,
            }
        all_six_answerable, all_six_error = parsed_answerability_sufficiency(
            all_six_rows[-1]
        )
        if all_six_error:
            return {
                "passed": False,
                "reason": f"invalid all-six sufficiency evaluation: {all_six_error}",
                "failure_label": "all_six_unparsed",
                "all_six_answerable": None,
                **metrics,
            }
        combined_metrics = {
            "all_six_answerable": all_six_answerable,
            "shared_fact_ids": list(speaker_fact_ids or []),
            "shared_fact_count": len(speaker_fact_ids or []),
            **metrics,
        }
        if not all_six_answerable:
            return {
                "passed": False,
                "reason": "all-six videos were judged insufficient to answer the question",
                "failure_label": "all_six_not_answerable",
                **combined_metrics,
            }
        minimum_metrics: dict[str, Any] = {}
        if six_user_judge_mode == SIX_USER_JUDGE_MODE_SEQUENTIAL:
            if not minimum_rows:
                return {
                    "passed": False,
                    "reason": "missing minimum-required-users evaluation",
                    "failure_label": "minimum_required_users_missing",
                    **combined_metrics,
                }
            minimum_row = minimum_rows[-1]
            minimum_users = [str(user) for user in minimum_row.get("users") or []]
            if (
                len(minimum_users) < 2
                or minimum_users[0] != required_users[0]
                or any(user not in required_users for user in minimum_users)
            ):
                return {
                    "passed": False,
                    "reason": "minimum-required-users trial has an invalid user set",
                    "failure_label": "minimum_required_users_invalid",
                    **combined_metrics,
                }
            if minimum_row.get("shared_fact_ids") != speaker_fact_ids:
                return {
                    "passed": False,
                    "reason": (
                        "minimum-required-users and all-six evaluations did not use "
                        "the same frozen facts"
                    ),
                    "failure_label": "minimum_fact_plan_mismatch",
                    **combined_metrics,
                }
            minimum_answerable, minimum_error = parsed_answerability_sufficiency(
                minimum_row
            )
            minimum_metrics = {
                "minimum_required_users": minimum_users,
                "minimum_required_user_count": len(minimum_users),
                "minimum_required_users_answerable": minimum_answerable,
            }
            if minimum_error:
                return {
                    "passed": False,
                    "reason": (
                        "invalid minimum-required-users sufficiency evaluation: "
                        f"{minimum_error}"
                    ),
                    "failure_label": "minimum_required_users_unparsed",
                    **minimum_metrics,
                    **combined_metrics,
                }
            if not minimum_answerable:
                return {
                    "passed": False,
                    "reason": (
                        "the computed minimum required user set was judged insufficient"
                    ),
                    "failure_label": "minimum_required_users_not_answerable",
                    **minimum_metrics,
                    **combined_metrics,
                }
        speaker_visibility = {
            str(row.get("fact_id")): row.get("visibility")
            for row in speaker_rows[-1].get("fact_audits") or []
            if isinstance(row, dict)
        }
        all_six_visibility = {
            str(row.get("fact_id")): row.get("visibility")
            for row in all_six_rows[-1].get("fact_audits") or []
            if isinstance(row, dict)
        }
        provider_resolved_fact_ids = [
            fact_id
            for fact_id in speaker_fact_ids or []
            if speaker_visibility.get(str(fact_id)) != "VISIBLE"
            and all_six_visibility.get(str(fact_id)) == "VISIBLE"
        ]
        return {
            "passed": True,
            "reason": (
                "speaker-only evidence was judged insufficient; all-six evidence and "
                "the minimum required user set were judged sufficient"
                if six_user_judge_mode == SIX_USER_JUDGE_MODE_SEQUENTIAL
                else "speaker-only evidence was judged insufficient and all-six evidence "
                "was judged sufficient"
            ),
            "failure_label": None,
            "provider_resolved_fact_ids": provider_resolved_fact_ids,
            **minimum_metrics,
            **combined_metrics,
        }

    try:
        correct = normalize_correct(qa_item.get("correct"))
    except ValueError as exc:
        return {"passed": False, "reason": str(exc)}

    combined = [row for row in evaluations if row.get("condition_type") == "combined_all_users"]
    if not combined:
        return {"passed": False, "reason": "missing combined_all_users evaluation"}

    combined_choice, combined_invalid = parsed_choice(combined[-1].get("choice"))
    if combined_invalid:
        return {
            "passed": False,
            "reason": "combined_all_users did not select exactly one A-E answer",
            "invalid_evaluations": [
                {
                    "condition_id": combined[-1].get("condition_id"),
                    "choice": combined[-1].get("choice"),
                }
            ],
        }
    if combined_choice != correct:
        return {
            "passed": False,
            "reason": f"combined_all_users did not select correct answer {correct}",
        }

    asker_user = required_users[0] if required_users else None
    evidence_provider_user = required_users[1] if len(required_users) > 1 else None
    blocking_leaks = []
    evidence_provider_answerable = []
    invalid_evaluations = []
    for row in evaluations:
        if row.get("condition_type") == "combined_all_users":
            continue
        choice, invalid = parsed_choice(row.get("choice"))
        if invalid:
            invalid_evaluations.append(
                {
                    "condition_id": row.get("condition_id"),
                    "choice": row.get("choice"),
                }
            )
            continue
        if choice == correct:
            condition_id = row.get("condition_id")
            users = list(row.get("users") or [])
            if not users and isinstance(condition_id, str) and condition_id.startswith("single_user::"):
                users = [condition_id.split("::", 1)[1]]
            leak = {
                "condition_id": condition_id,
                "users": users,
                "choice": choice,
                "answer_text": row.get("answer_text"),
                "evidence_used": row.get("evidence_used"),
            }
            if (
                row.get("condition_type") == "single_user"
                and evidence_provider_user
                and users == [evidence_provider_user]
            ):
                evidence_provider_answerable.append(leak)
            else:
                blocking_leaks.append(leak)
    if invalid_evaluations:
        return {
            "passed": False,
            "reason": "answerability condition did not select exactly one A-E answer: "
            + ", ".join(str(item.get("condition_id")) for item in invalid_evaluations),
            "invalid_evaluations": invalid_evaluations,
        }
    if blocking_leaks:
        return {
            "passed": False,
            "reason": "asker/subset condition answered correctly: "
            + ", ".join(str(item.get("condition_id")) for item in blocking_leaks),
            "blocking_single_or_subset_answerable": blocking_leaks,
            "evidence_provider_answerable": evidence_provider_answerable,
            "speaker_user": asker_user,
            "evidence_provider_user": evidence_provider_user,
        }

    gate = {
        "passed": True,
        "reason": "combined videos answer correctly and all single/subset conditions chose an incorrect answer",
        "evidence_provider_answerable": evidence_provider_answerable,
        "speaker_user": asker_user,
        "evidence_provider_user": evidence_provider_user,
    }
    if evidence_provider_answerable:
        gate["reason"] = (
            "combined videos answer correctly; the evidence provider alone also answered correctly "
            "and this is logged as acceptable evidence-provider answerability"
        )
        gate["warning"] = "evidence_provider_alone_can_answer"
    return gate


def judge_gate(
    judge: dict[str, Any],
    *,
    required_checks: tuple[str, ...] = BLOCKING_JUDGE_CHECKS,
) -> dict[str, Any]:
    """Deterministically gate structured judger output.

    The model still proposes review_passed, but when structured checks are
    present the checks are authoritative. Some VLM outputs mark every blocking
    check PASS while leaving the top-level review_passed flag false; that flag
    is treated as a diagnostic inconsistency rather than a veto.
    """

    checks = judge.get("checks")
    if not isinstance(checks, dict):
        if judge.get("review_passed") is not True:
            return {
                "passed": False,
                "reason": str(judge.get("feedback_to_generator") or "judger review_passed is not true"),
                "failed_checks": list(judge.get("blocking_failures") or []),
            }
        return {
            "passed": True,
            "reason": "legacy judger output passed without structured checks",
            "failed_checks": [],
        }

    failed = []
    missing = []
    blocking_failures = list(judge.get("blocking_failures") or [])
    for name in required_checks:
        check = checks.get(name)
        if not isinstance(check, dict):
            missing.append(name)
            continue
        status = str(check.get("status", "")).strip().upper()
        if status != "PASS":
            failed.append(name)
    if missing or failed:
        details = []
        if failed:
            details.append("failed checks: " + ", ".join(failed))
        if missing:
            details.append("missing checks: " + ", ".join(missing))
        return {
            "passed": False,
            "reason": "; ".join(details),
            "failed_checks": failed + missing,
        }
    gate = {
        "passed": True,
        "reason": "all structured judger checks passed",
        "failed_checks": [],
    }
    warnings = []
    if blocking_failures:
        gate["model_blocking_failures"] = blocking_failures
        warnings.append(
            "ignored inconsistent blocking_failures because all required structured "
            "checks passed"
        )
    if judge.get("review_passed") is not True:
        gate["model_review_passed"] = judge.get("review_passed")
        warnings.append(
            "ignored inconsistent top-level review_passed because all required "
            "structured checks passed"
        )
    if warnings:
        gate["warnings"] = warnings
        gate["warning"] = "; ".join(warnings)
    return gate


def schema_formality_branch(schema_errors: list[str]) -> dict[str, Any]:
    """Return the deterministic schema/formality branch for qa_formality."""

    schema_errors = list(schema_errors)
    return {
        "status": "PASS" if not schema_errors else "FAIL",
        "errors": schema_errors,
        "reason": (
            "deterministic schema/formality checks passed"
            if not schema_errors
            else "deterministic schema/formality checks failed: " + "; ".join(schema_errors)
        ),
    }


# Archived inactive scoring pipeline:
#
# def quality_uncertainty_from_choice_logits(signal):
#     raw = signal.get("choice_logits") or signal.get("choice_logprobs")
#     weights = {str(score): float(raw[str(score)]) for score in (1, 2, 3)}
#     probabilities = softmax(weights)
#     entropy_nats = -sum(p * log(p) for p in probabilities.values())
#     return {
#         "choice_set": [1, 2, 3],
#         "probabilities": probabilities,
#         "normalized_entropy": entropy_nats / log(3),
#         "argmax_score": argmax(probabilities),
#         "generated_score": signal.get("generated_choice"),
#     }
#
# def normalize_quality_fields(check, check_name, *, choice_signal=None, emitted_score=None):
#     uncertainty = quality_uncertainty_from_choice_logits(choice_signal)
#     score = uncertainty["argmax_score"] if uncertainty["available"] else emitted_score
#     check["quality_score"] = clamp(score, 1, 3)
#     check["quality_flag"] = QUALITY_FLAGS[check["quality_score"]]
#     check["quality_score_source"] = "choice_logits_argmax" or a fallback source
#     check["quality_uncertainty"] = uncertainty
#     check["quality_reason"] = the model reason or a score-derived fallback
#     return check


def validate_first_verdict_sidecar_generation(
    generation: dict[str, Any],
    *,
    prompt: str,
    check_name: str,
) -> dict[str, Any]:
    """Validate a detailed sidecar judge whose first field is its real verdict.

    The experiment is invalid unless lowercase pass/fail is the first generated
    verdict and the later detailed check agrees with that authoritative field.
    """

    raw_output = str(generation.get("text") or "")
    source_signal = generation.get("choice_logits")
    signal = dict(source_signal) if isinstance(source_signal, dict) else {}
    output_contract_errors = []
    measurement_errors = []
    try:
        parsed = json.loads(raw_output.strip())
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        parsed = None
        output_contract_errors.append(f"judge output is not exact JSON: {exc}")
    if not isinstance(parsed, dict) or not parsed or list(parsed)[0] != FIRST_VERDICT_FIELD:
        output_contract_errors.append("judge must contain verdict as its first field")
        parsed_verdict = None
    else:
        parsed_verdict = str(parsed.get(FIRST_VERDICT_FIELD) or "").strip()
        if parsed_verdict not in FIRST_VERDICT_CHOICES:
            output_contract_errors.append("verdict must be lowercase pass or fail")
    if isinstance(parsed, dict) and "review_passed" in parsed:
        output_contract_errors.append("first-verdict judge must not emit review_passed")

    expected_schema = None
    if JUDGE_OUTPUT_SCHEMA_MARKER in prompt:
        try:
            expected_schema = json.loads(
                prompt.rsplit(JUDGE_OUTPUT_SCHEMA_MARKER, 1)[1].strip()
            )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            output_contract_errors.append(f"judge prompt schema is not valid JSON: {exc}")
    else:
        output_contract_errors.append("judge prompt is missing its output-schema marker")

    def require_schema_keys(
        expected: Any,
        actual: Any,
        path: str = "",
    ) -> None:
        if not isinstance(expected, dict):
            return
        if not isinstance(actual, dict):
            output_contract_errors.append(f"{path or 'output'} must be an object")
            return
        for key, expected_value in expected.items():
            key_path = f"{path}.{key}" if path else key
            if key not in actual:
                output_contract_errors.append(f"judge omitted required field {key_path}")
            elif isinstance(expected_value, dict):
                require_schema_keys(expected_value, actual[key], key_path)

    if isinstance(expected_schema, dict) and isinstance(parsed, dict):
        require_schema_keys(expected_schema, parsed)

    nested_status = None
    if isinstance(parsed, dict) and parsed_verdict in FIRST_VERDICT_CHOICES:
        checks = parsed.get("checks")
        check = checks.get(check_name) if isinstance(checks, dict) else None
        if not isinstance(check, dict):
            output_contract_errors.append(f"checks.{check_name} must be an object")
        else:
            reason = check.get("reason")
            fix = check.get("fix")
            if not isinstance(reason, str) or not reason.strip():
                output_contract_errors.append(
                    f"checks.{check_name}.reason must be a non-empty string"
                )
            if not isinstance(fix, str):
                output_contract_errors.append(f"checks.{check_name}.fix must be a string")
            semantic_subchecks = check.get("semantic_subchecks")
            if semantic_subchecks is not None:
                if not isinstance(semantic_subchecks, dict):
                    output_contract_errors.append(
                        f"checks.{check_name}.semantic_subchecks must be an object"
                    )
                else:
                    for subcheck_name, subcheck in semantic_subchecks.items():
                        if not isinstance(subcheck, dict):
                            output_contract_errors.append(
                                f"checks.{check_name}.semantic_subchecks."
                                f"{subcheck_name} must be an object"
                            )
                            continue
                        subcheck_status = str(
                            subcheck.get("status") or ""
                        ).strip().upper()
                        if subcheck_status not in {"PASS", "FAIL"}:
                            output_contract_errors.append(
                                f"checks.{check_name}.semantic_subchecks."
                                f"{subcheck_name}.status must be PASS or FAIL"
                            )
                        subcheck_reason = subcheck.get("reason")
                        if (
                            not isinstance(subcheck_reason, str)
                            or not subcheck_reason.strip()
                        ):
                            output_contract_errors.append(
                                f"checks.{check_name}.semantic_subchecks."
                                f"{subcheck_name}.reason must be a non-empty string"
                            )
        nested_status = (
            str(check.get("status") or "").strip().upper()
            if isinstance(check, dict)
            else None
        )
        if nested_status not in {"PASS", "FAIL"}:
            output_contract_errors.append(
                f"checks.{check_name}.status must be PASS or FAIL"
            )
        if nested_status in {"PASS", "FAIL"} and nested_status != parsed_verdict.upper():
            output_contract_errors.append(
                "authoritative verdict disagrees with the later detailed check status"
            )
        blocking_failures = parsed.get("blocking_failures")
        if not isinstance(blocking_failures, list):
            output_contract_errors.append("blocking_failures must be an array")
        else:
            if parsed_verdict == "pass" and blocking_failures:
                output_contract_errors.append(
                    "pass verdict must not list blocking_failures"
                )
            if parsed_verdict == "fail" and check_name not in blocking_failures:
                output_contract_errors.append(
                    "fail verdict must list the failed judge in blocking_failures"
                )
        if not isinstance(parsed.get("feedback_to_generator"), str):
            output_contract_errors.append("feedback_to_generator must be a string")
    else:
        check = None

    field_name = str(signal.get("field_name") or "")
    if field_name != FIRST_VERDICT_FIELD:
        measurement_errors.append(
            f"choice logits targeted {field_name or '<missing>'!r}, "
            f"expected {FIRST_VERDICT_FIELD!r}"
        )
    prefix = signal.get("generated_prefix_before_choice")
    if not isinstance(prefix, str):
        measurement_errors.append(
            "runner did not retain the generated prefix before the verdict token"
        )
        prefix = ""
    prior_verdict = bool(
        re.search(r'"review_passed"\s*:', prefix, re.IGNORECASE)
        or re.search(
            r'["\'](?:status|decision|verdict)["\']\s*:\s*'
            r'["\'](?:pass|fail)["\']',
            prefix,
            re.IGNORECASE,
        )
    )
    if prior_verdict:
        measurement_errors.append(
            "generated prefix contains a verdict before the measured first verdict"
        )
    generated_choice = str(signal.get("generated_choice") or "").strip()
    if (
        parsed_verdict in FIRST_VERDICT_CHOICES
        and generated_choice != parsed_verdict
    ):
        measurement_errors.append(
            "captured verdict token does not match the parsed verdict"
        )
    errors = [*output_contract_errors, *measurement_errors]
    if signal.get("available") is not True:
        errors.append(str(signal.get("reason") or "runner did not return available choice logits"))

    signal.update(
        {
            "available": not errors,
            "reason": "; ".join(dict.fromkeys(errors)),
            "probe_version": FIRST_VERDICT_ENTROPY_VERSION,
            "measurement_context": "authoritative_first_detailed_judge_verdict",
            "field_name": FIRST_VERDICT_FIELD,
            "prior_generated_verdict": prior_verdict,
            "probe_output_contract_valid": not output_contract_errors,
            "measurement_contract_valid": not measurement_errors,
        }
    )
    return {
        "prompt": prompt,
        "raw_output": raw_output,
        "parsed_verdict": parsed_verdict,
        "parsed_detailed_judge": parsed,
        "nested_check_status": nested_status,
        "verdict_matches_nested_status": (
            nested_status == parsed_verdict.upper()
            if parsed_verdict in FIRST_VERDICT_CHOICES
            and nested_status in {"PASS", "FAIL"}
            else None
        ),
        "choice_logit_signal": signal,
        "probe_version": FIRST_VERDICT_ENTROPY_VERSION,
        "sidecar_verdict_is_authoritative": True,
        "independent_from_acceptance_gate": True,
    }


def run_first_verdict_entropy_sidecar_call(
    *,
    runner: Any,
    prompt: str,
    image_paths: list[str],
    video_paths: list[str],
    check_name: str,
) -> dict[str, Any]:
    """Run one detailed judge call and capture its first lowercase verdict."""

    generate_with_choice_logits = getattr(runner, "generate_with_choice_logits", None)
    supports_choice_logits = getattr(runner, "supports_choice_logits", True)
    if not callable(generate_with_choice_logits) or not supports_choice_logits:
        reason = (
            f"runner {type(runner).__name__} disables choice logits for this provider/model"
            if callable(generate_with_choice_logits) and not supports_choice_logits
            else f"runner {type(runner).__name__} does not expose choice logits"
        )
        return {
            "prompt": prompt,
            "raw_output": "",
            "parsed_verdict": None,
            "parsed_detailed_judge": None,
            "choice_logit_signal": {
                "available": False,
                "reason": reason,
                "probe_version": FIRST_VERDICT_ENTROPY_VERSION,
                "measurement_context": "authoritative_first_detailed_judge_verdict",
                "field_name": FIRST_VERDICT_FIELD,
                "prior_generated_verdict": False,
                "probe_output_contract_valid": False,
            },
            "probe_version": FIRST_VERDICT_ENTROPY_VERSION,
            "sidecar_verdict_is_authoritative": True,
            "independent_from_acceptance_gate": True,
        }
    generation = generate_with_choice_logits(
        prompt,
        image_paths=image_paths,
        video_paths=video_paths,
        field_name=FIRST_VERDICT_FIELD,
        choices=FIRST_VERDICT_CHOICES,
    )
    if not isinstance(generation, dict):
        generation = {
            "text": "",
            "choice_logits": {
                "available": False,
                "reason": "runner returned a non-object first-verdict sidecar result",
            },
        }
    return validate_first_verdict_sidecar_generation(
        generation,
        prompt=prompt,
        check_name=check_name,
    )


def validate_minimal_verdict_probe_generation(
    generation: dict[str, Any],
    *,
    prompt: str,
    check_name: str,
) -> dict[str, Any]:
    """Validate an independent judge probe whose entire output is one verdict."""

    raw_output = str(generation.get("text") or "")
    source_signal = generation.get("choice_logits")
    signal = dict(source_signal) if isinstance(source_signal, dict) else {}
    output_contract_errors = []
    measurement_errors = []
    try:
        parsed = json.loads(raw_output.strip())
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        parsed = None
        output_contract_errors.append(f"probe output is not exact JSON: {exc}")

    parsed_verdict = None
    if not isinstance(parsed, dict):
        output_contract_errors.append("probe output must be a JSON object")
    elif list(parsed) != [FIRST_VERDICT_FIELD]:
        output_contract_errors.append(
            "probe output must contain exactly one field: verdict"
        )
    else:
        parsed_verdict = parsed.get(FIRST_VERDICT_FIELD)
        if (
            not isinstance(parsed_verdict, str)
            or parsed_verdict not in FIRST_VERDICT_CHOICES
        ):
            output_contract_errors.append(
                "verdict must be exactly lowercase pass or fail"
            )

    expected_schema = None
    if JUDGE_OUTPUT_SCHEMA_MARKER not in prompt:
        output_contract_errors.append(
            "probe prompt is missing its output-schema marker"
        )
    else:
        try:
            expected_schema = json.loads(
                prompt.rsplit(JUDGE_OUTPUT_SCHEMA_MARKER, 1)[1].strip()
            )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            output_contract_errors.append(
                f"probe prompt schema is not valid JSON: {exc}"
            )
    if expected_schema != {FIRST_VERDICT_FIELD: "pass/fail"}:
        output_contract_errors.append(
            "probe prompt must request exactly the single verdict field"
        )

    field_name = str(signal.get("field_name") or "")
    if field_name != FIRST_VERDICT_FIELD:
        measurement_errors.append(
            f"choice logits targeted {field_name or '<missing>'!r}, "
            f"expected {FIRST_VERDICT_FIELD!r}"
        )
    prefix = signal.get("generated_prefix_before_choice")
    if not isinstance(prefix, str):
        measurement_errors.append(
            "runner did not retain the generated prefix before the verdict token"
        )
        prefix = ""
    prior_verdict = bool(
        re.search(r'"review_passed"\s*:', prefix, re.IGNORECASE)
        or re.search(
            r'["\'](?:status|decision|verdict)["\']\s*:\s*'
            r'["\'](?:pass|fail)["\']',
            prefix,
            re.IGNORECASE,
        )
    )
    if prior_verdict:
        measurement_errors.append(
            "generated prefix contains a verdict before the measured verdict"
        )
    generated_choice = str(signal.get("generated_choice") or "").strip()
    if (
        parsed_verdict in FIRST_VERDICT_CHOICES
        and generated_choice != parsed_verdict
    ):
        measurement_errors.append(
            "captured verdict token does not match the parsed verdict"
        )

    errors = [*output_contract_errors, *measurement_errors]
    if signal.get("available") is not True:
        errors.append(
            str(signal.get("reason") or "runner did not return available choice logits")
        )
    signal.update(
        {
            "available": not errors,
            "reason": "; ".join(dict.fromkeys(errors)),
            "probe_version": MINIMAL_VERDICT_ENTROPY_VERSION,
            "measurement_context": "independent_minimal_judge_verdict",
            "field_name": FIRST_VERDICT_FIELD,
            "judge": check_name,
            "prior_generated_verdict": prior_verdict,
            "probe_output_contract_valid": not output_contract_errors,
            "measurement_contract_valid": not measurement_errors,
            "decision_role": "independent_diagnostic_probe",
            "verdict_affects_acceptance": False,
            "entropy_used_as_threshold": False,
            "independent_from_acceptance_gate": True,
        }
    )
    return {
        "prompt": prompt,
        "raw_output": raw_output,
        "parsed_verdict": (
            parsed_verdict
            if parsed_verdict in FIRST_VERDICT_CHOICES
            else None
        ),
        "choice_logit_signal": signal,
        "probe_version": MINIMAL_VERDICT_ENTROPY_VERSION,
        "decision_role": "independent_diagnostic_probe",
        "verdict_affects_acceptance": False,
        "entropy_used_as_threshold": False,
        "independent_from_acceptance_gate": True,
    }


def run_minimal_verdict_entropy_probe_call(
    *,
    runner: Any,
    prompt: str,
    image_paths: list[str],
    video_paths: list[str],
    check_name: str,
) -> dict[str, Any]:
    """Run the second, minimal judge call and capture pass/fail logits."""

    generate_with_choice_logits = getattr(runner, "generate_with_choice_logits", None)
    supports_choice_logits = getattr(runner, "supports_choice_logits", True)
    if not callable(generate_with_choice_logits) or not supports_choice_logits:
        reason = (
            f"runner {type(runner).__name__} disables choice logits for this provider/model"
            if callable(generate_with_choice_logits) and not supports_choice_logits
            else f"runner {type(runner).__name__} does not expose choice logits"
        )
        return validate_minimal_verdict_probe_generation(
            {
                "text": "",
                "choice_logits": {
                    "available": False,
                    "reason": reason,
                    "field_name": FIRST_VERDICT_FIELD,
                    "generated_prefix_before_choice": "",
                },
            },
            prompt=prompt,
            check_name=check_name,
        )
    generation = generate_with_choice_logits(
        prompt,
        image_paths=image_paths,
        video_paths=video_paths,
        field_name=FIRST_VERDICT_FIELD,
        choices=FIRST_VERDICT_CHOICES,
    )
    if not isinstance(generation, dict):
        generation = {
            "text": "",
            "choice_logits": {
                "available": False,
                "reason": "runner returned a non-object minimal-verdict result",
                "field_name": FIRST_VERDICT_FIELD,
                "generated_prefix_before_choice": "",
            },
        }
    return validate_minimal_verdict_probe_generation(
        generation,
        prompt=prompt,
        check_name=check_name,
    )


def decision_uncertainty_from_choice_logits(signal: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize first-verdict pass/fail or legacy PASS/FAIL logits."""

    statuses = ("PASS", "FAIL")
    if not isinstance(signal, dict):
        return {"available": False, "reason": "runner returned no choice-logit signal"}
    generated_raw = str(signal.get("generated_choice") or "").strip()
    generated = generated_raw.upper()
    signal_metadata = {
        **({"generated_decision": generated} if generated in statuses else {}),
        "probe_version": signal.get("probe_version"),
        "measurement_context": signal.get("measurement_context"),
        "field_name": signal.get("field_name"),
        "prior_generated_verdict": signal.get("prior_generated_verdict"),
        "probe_output_contract_valid": signal.get("probe_output_contract_valid"),
        "measurement_contract_valid": signal.get("measurement_contract_valid"),
        "decision_role": signal.get("decision_role"),
        "verdict_affects_acceptance": signal.get("verdict_affects_acceptance"),
        "entropy_used_as_threshold": signal.get("entropy_used_as_threshold"),
        "independent_from_acceptance_gate": signal.get(
            "independent_from_acceptance_gate"
        ),
    }
    if signal.get("available") is not True:
        return {
            "available": False,
            "reason": str(signal.get("reason") or "choice logits unavailable"),
            **signal_metadata,
        }
    raw = signal.get("choice_logits") or signal.get("choice_logprobs")
    if not isinstance(raw, dict):
        return {
            "available": False,
            "reason": "runner did not return pass/fail weights",
            **signal_metadata,
        }
    if all(choice in raw for choice in FIRST_VERDICT_CHOICES):
        source_keys = {"PASS": "pass", "FAIL": "fail"}
        verdict_token_case = "lowercase"
    elif all(status in raw for status in statuses):
        source_keys = {"PASS": "PASS", "FAIL": "FAIL"}
        verdict_token_case = "uppercase_legacy"
    else:
        return {
            "available": False,
            "reason": "runner did not return both PASS and FAIL weights",
            **signal_metadata,
        }
    try:
        weights = {
            status: float(raw[source_keys[status]])
            for status in statuses
        }
    except (TypeError, ValueError):
        return {
            "available": False,
            "reason": "choice weights contain a non-numeric value",
            **signal_metadata,
        }
    if not all(math.isfinite(value) for value in weights.values()):
        return {
            "available": False,
            "reason": "choice weights contain non-finite values",
            **signal_metadata,
        }
    max_weight = max(weights.values())
    exp_weights = {status: math.exp(value - max_weight) for status, value in weights.items()}
    denominator = sum(exp_weights.values())
    probabilities = {status: value / denominator for status, value in exp_weights.items()}
    entropy_nats = -sum(
        probability * math.log(probability)
        for probability in probabilities.values()
        if probability > 0.0
    )
    normalized_entropy = entropy_nats / math.log(2.0)
    return {
        "available": True,
        "choice_set": list(statuses),
        "captured_choice_set": [source_keys[status] for status in statuses],
        "verdict_token_case": verdict_token_case,
        "weight_type": str(signal.get("weight_type") or "logit_or_log_probability"),
        "log_weights": {status: round(value, 8) for status, value in weights.items()},
        "probabilities": {status: round(value, 8) for status, value in probabilities.items()},
        "entropy_nats": round(entropy_nats, 8),
        "entropy_bits": round(entropy_nats / math.log(2.0), 8),
        "normalized_entropy": round(normalized_entropy, 8),
        "argmax_decision": max(probabilities, key=probabilities.get),
        **signal_metadata,
        "token_index": signal.get("token_index"),
        "distribution_scope": (
            "softmax restricted to pass and fail at the only verdict field "
            "of the independent minimal judge probe"
        ),
    }


def answerability_uncertainty_from_choice_logits(
    signal: dict[str, Any] | None,
) -> dict[str, Any]:
    """Normalize direct A-E answer logits without affecting the answerability gate."""

    choices = tuple(OPTION_LETTERS)
    if not isinstance(signal, dict):
        return {"available": False, "reason": "runner returned no choice-logit signal"}
    generated = str(signal.get("generated_choice") or "").upper()
    if signal.get("available") is not True:
        return {
            "available": False,
            "reason": str(signal.get("reason") or "choice logits unavailable"),
            **({"generated_choice": generated} if generated in choices else {}),
        }
    raw = signal.get("choice_logits") or signal.get("choice_logprobs")
    if not isinstance(raw, dict) or any(choice not in raw for choice in choices):
        return {"available": False, "reason": "runner did not return all A-E weights"}
    weights = {choice: float(raw[choice]) for choice in choices}
    if not all(math.isfinite(value) for value in weights.values()):
        return {"available": False, "reason": "choice weights contain non-finite values"}
    max_weight = max(weights.values())
    exp_weights = {choice: math.exp(value - max_weight) for choice, value in weights.items()}
    denominator = sum(exp_weights.values())
    probabilities = {choice: value / denominator for choice, value in exp_weights.items()}
    entropy_nats = -sum(
        probability * math.log(probability)
        for probability in probabilities.values()
        if probability > 0.0
    )
    return {
        "available": True,
        "choice_set": list(choices),
        "weight_type": str(signal.get("weight_type") or "logit_or_log_probability"),
        "log_weights": {choice: round(value, 8) for choice, value in weights.items()},
        "probabilities": {
            choice: round(value, 8) for choice, value in probabilities.items()
        },
        "entropy_nats": round(entropy_nats, 8),
        "entropy_bits": round(entropy_nats / math.log(2.0), 8),
        "normalized_entropy": round(entropy_nats / math.log(float(len(choices))), 8),
        "argmax_choice": max(probabilities, key=probabilities.get),
        "generated_choice": generated if generated in choices else None,
        "token_index": signal.get("token_index"),
        "distribution_scope": "softmax restricted to direct answer tokens A, B, C, D, and E",
        "note": "diagnostic only; legacy non-six-user answerability is forced-choice over A-E",
    }


def attach_decision_uncertainty(
    check: dict[str, Any],
    check_name: str,
    *,
    choice_signal: dict[str, Any] | None = None,
    emitted_status: Any = None,
) -> dict[str, Any]:
    """Attach entropy metadata without overriding the effective gate status."""

    if check_name not in LEGACY_DECISION_ENTROPY_JUDGE_CHECKS:
        return check
    existing = check.get("decision_uncertainty")
    uncertainty = (
        existing
        if choice_signal is None and isinstance(existing, dict)
        else decision_uncertainty_from_choice_logits(choice_signal)
    )
    check["decision_uncertainty"] = uncertainty
    # Compare the independent probe with the effective production check after
    # any deterministic schema or semantic-subcheck overrides.
    generated_status = str(uncertainty.get("generated_decision") or "").upper()
    effective_status = str(check.get("status") or "").upper()
    probe_matches = bool(
        generated_status in {"PASS", "FAIL"}
        and effective_status in {"PASS", "FAIL"}
        and generated_status == effective_status
    )
    check["probe_matches_effective_status"] = probe_matches
    # Compatibility alias retained for older artifact readers.
    check["status_matches_effective_status"] = probe_matches
    return check


def failed_single_judge(check_name: str, reason: str, *, raw_output: str | None = None) -> dict[str, Any]:
    failed_check = {
        "status": "FAIL",
        "reason": reason,
        "fix": f"Repair the question-answer item so the {check_name} judge can pass.",
    }
    judge = {
        "review_passed": False,
        "checks": {
            check_name: failed_check
        },
        "blocking_failures": [check_name],
        "why_generator_asked_this": "",
        "feedback_to_generator": reason,
    }
    if raw_output is not None:
        judge["raw_output"] = raw_output
    return judge


def single_judge_output_errors(judge: dict[str, Any], check_name: str) -> list[str]:
    """Validate the production JSON contract before a judge result reaches the merger."""

    errors = []
    if not isinstance(judge.get("review_passed"), bool):
        errors.append("review_passed must be boolean")
    checks = judge.get("checks")
    check = checks.get(check_name) if isinstance(checks, dict) else None
    if not isinstance(check, dict):
        errors.append(f"checks.{check_name} must be an object")
    else:
        status = str(check.get("status") or "").strip().upper()
        if status not in {"PASS", "FAIL"}:
            errors.append(f"checks.{check_name}.status must be PASS or FAIL")
        reason = check.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            errors.append(f"checks.{check_name}.reason must be a non-empty string")
        fix = check.get("fix")
        if not isinstance(fix, str):
            errors.append(f"checks.{check_name}.fix must be a string")
        if check_name == "qa_formality":
            semantic_subchecks = check.get("semantic_subchecks")
            if not isinstance(semantic_subchecks, dict):
                errors.append("checks.qa_formality.semantic_subchecks must be an object")
            else:
                for subcheck_name in QA_FORMALITY_SEMANTIC_SUBCHECK_NAMES:
                    subcheck = semantic_subchecks.get(subcheck_name)
                    if not isinstance(subcheck, dict):
                        errors.append(
                            f"checks.qa_formality.semantic_subchecks.{subcheck_name} "
                            "must be an object"
                        )
                        continue
                    subcheck_status = str(subcheck.get("status") or "").strip().upper()
                    if subcheck_status not in {"PASS", "FAIL"}:
                        errors.append(
                            f"checks.qa_formality.semantic_subchecks.{subcheck_name}.status "
                            "must be PASS or FAIL"
                        )
                    subcheck_reason = subcheck.get("reason")
                    if not isinstance(subcheck_reason, str) or not subcheck_reason.strip():
                        errors.append(
                            f"checks.qa_formality.semantic_subchecks.{subcheck_name}.reason "
                            "must be a non-empty string"
                        )
    if not isinstance(judge.get("blocking_failures"), list):
        errors.append("blocking_failures must be an array")
    if not isinstance(judge.get("feedback_to_generator"), str):
        errors.append("feedback_to_generator must be a string")
    return errors


def parse_single_judge_output(raw: str, check_name: str) -> dict[str, Any]:
    judge = extract_json_object(raw)
    contract_errors = single_judge_output_errors(judge, check_name)
    if contract_errors:
        raise ValueError("judge JSON contract errors: " + "; ".join(contract_errors))
    return judge


def run_model_judge_branch(
    *,
    check_name: str,
    prompt: str,
    runner: Any,
    image_paths: list[str],
    video_paths: list[str],
    evidence_id: Any,
    qa_id: Any,
    attempt: int,
    collect_choice_logits: bool = False,
    minimal_verdict_probe_prompt: str | None = None,
) -> dict[str, Any]:
    """Run one model judge.

    The detailed call always remains authoritative. When entropy collection is
    enabled, a second independent call returns only ``{"verdict":"pass/fail"}``;
    its output and entropy are diagnostic and cannot alter the detailed result.
    """

    stage = f"{check_name}_judge"
    stage_start = time.time()
    print(
        "qa_stage_start "
        f"stage={stage} evidence_id={evidence_id} "
        f"qa_id={qa_id} attempt={attempt} "
        f"images={len(image_paths)} videos={len(video_paths)}",
        flush=True,
    )
    raw = runner.generate(prompt, image_paths=image_paths, video_paths=video_paths)
    print(
        "qa_stage_done "
        f"stage={stage} evidence_id={evidence_id} "
        f"qa_id={qa_id} attempt={attempt} "
        f"seconds={time.time() - stage_start:.1f}",
        flush=True,
    )
    initial_raw = raw
    final_raw = raw
    format_repair = {
        "attempted": False,
        "succeeded": False,
    }
    try:
        judge = parse_single_judge_output(raw, check_name)
    except Exception as initial_exc:
        format_repair = {
            "attempted": True,
            "succeeded": False,
            "initial_error": f"{type(initial_exc).__name__}: {initial_exc}",
        }
        repair_prompt = build_judge_json_repair_prompt(
            raw,
            judge_schema_for_check(check_name, pass_fail_only=True),
        )
        repair_start = time.time()
        print(
            "qa_format_repair_start "
            f"stage={stage} evidence_id={evidence_id} "
            f"qa_id={qa_id} attempt={attempt}",
            flush=True,
        )
        try:
            final_raw = runner.generate(
                repair_prompt,
                image_paths=[],
                video_paths=[],
            )
            judge = parse_single_judge_output(final_raw, check_name)
            format_repair["succeeded"] = True
        except OpenRouterRequestError:
            raise
        except Exception as repair_exc:
            format_repair["repair_error"] = f"{type(repair_exc).__name__}: {repair_exc}"
            judge = failed_single_judge(
                check_name,
                (
                    f"{check_name} judge output remained invalid after one JSON repair attempt: "
                    f"{repair_exc}"
                ),
            )
        print(
            "qa_format_repair_done "
            f"stage={stage} evidence_id={evidence_id} "
            f"qa_id={qa_id} attempt={attempt} "
            f"succeeded={format_repair['succeeded']} "
            f"seconds={time.time() - repair_start:.1f}",
            flush=True,
        )
    judge["raw_output"] = final_raw
    if format_repair["attempted"]:
        judge["initial_raw_output"] = initial_raw
        judge["format_repair"] = format_repair

    if not collect_choice_logits:
        judge["elapsed_seconds"] = round(time.time() - stage_start, 3)
        return judge

    effective_probe_prompt = minimal_verdict_probe_prompt
    probe_stage = f"{check_name}_entropy_probe"
    probe_start = time.time()
    print(
        "qa_stage_start "
        f"stage={probe_stage} evidence_id={evidence_id} "
        f"qa_id={qa_id} attempt={attempt} "
        f"images={len(image_paths)} videos={len(video_paths)}",
        flush=True,
    )
    try:
        if effective_probe_prompt is None:
            effective_probe_prompt = build_judge_minimal_verdict_probe_prompt(
                prompt,
                check_name,
            )
        entropy_probe = run_minimal_verdict_entropy_probe_call(
            runner=runner,
            prompt=effective_probe_prompt,
            image_paths=image_paths,
            video_paths=video_paths,
            check_name=check_name,
        )
    except Exception as exc:
        fallback_prompt = effective_probe_prompt or ""
        entropy_probe = validate_minimal_verdict_probe_generation(
            {
                "text": "",
                "choice_logits": {
                    "available": False,
                    "reason": (
                        "independent minimal-verdict probe crashed: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                    "field_name": FIRST_VERDICT_FIELD,
                    "generated_prefix_before_choice": "",
                },
            },
            prompt=fallback_prompt,
            check_name=check_name,
        )
    signal = entropy_probe.get("choice_logit_signal")
    if not isinstance(signal, dict):
        signal = {
            "available": False,
            "reason": "independent minimal-verdict probe returned no choice-logit signal",
            "probe_version": MINIMAL_VERDICT_ENTROPY_VERSION,
            "measurement_context": "independent_minimal_judge_verdict",
            "field_name": FIRST_VERDICT_FIELD,
            "probe_output_contract_valid": False,
            "decision_role": "independent_diagnostic_probe",
            "verdict_affects_acceptance": False,
            "entropy_used_as_threshold": False,
            "independent_from_acceptance_gate": True,
        }
    judge["choice_logit_signal"] = signal
    judge["entropy_probe_verdict"] = entropy_probe.get("parsed_verdict")
    judge["minimal_entropy_probe"] = {
        key: value
        for key, value in entropy_probe.items()
        if key not in {"prompt", "choice_logit_signal"}
    }
    print(
        "qa_stage_done "
        f"stage={probe_stage} evidence_id={evidence_id} "
        f"qa_id={qa_id} attempt={attempt} "
        f"probe_verdict={entropy_probe.get('parsed_verdict') or 'invalid'} "
        f"entropy_available={signal.get('available') is True} "
        f"seconds={time.time() - probe_start:.1f}",
        flush=True,
    )
    judge["elapsed_seconds"] = round(time.time() - stage_start, 3)
    return judge


def combined_direct_judge_output_errors(judge: dict[str, Any]) -> list[str]:
    """Validate the one-call qa_formality + evidence_groundedness contract."""

    errors = []
    for check_name in ("qa_formality", "evidence_groundedness"):
        errors.extend(single_judge_output_errors(judge, check_name))
    return list(dict.fromkeys(errors))


def parse_combined_direct_judge_output(raw: str) -> dict[str, Any]:
    judge = extract_json_object(raw)
    contract_errors = combined_direct_judge_output_errors(judge)
    if contract_errors:
        raise ValueError(
            "combined direct judge JSON contract errors: "
            + "; ".join(contract_errors)
        )
    return judge


def failed_combined_direct_judge(reason: str) -> dict[str, Any]:
    semantic_subchecks = {
        name: {"status": "FAIL", "reason": reason}
        for name in QA_FORMALITY_SEMANTIC_SUBCHECK_NAMES
    }
    return {
        "review_passed": False,
        "checks": {
            "qa_formality": {
                "status": "FAIL",
                "reason": reason,
                "fix": "Return the complete combined direct-judge JSON contract.",
                "semantic_subchecks": semantic_subchecks,
            },
            "evidence_groundedness": {
                "status": "FAIL",
                "reason": reason,
                "fix": "Return the complete combined direct-judge JSON contract.",
            },
        },
        "blocking_failures": ["qa_formality", "evidence_groundedness"],
        "why_generator_asked_this": "",
        "feedback_to_generator": reason,
    }


def run_combined_direct_judge(
    *,
    prompt: str,
    runner: Any,
    image_paths: list[str],
    video_paths: list[str],
    evidence_id: Any,
    qa_id: Any,
    attempt: int,
) -> dict[str, Any]:
    """Run the old one-pass judge, repairing JSON without replaying media."""

    stage = "combined_direct_judge"
    stage_start = time.time()
    print(
        "qa_stage_start "
        f"stage={stage} evidence_id={evidence_id} qa_id={qa_id} "
        f"attempt={attempt} images={len(image_paths)} videos={len(video_paths)}",
        flush=True,
    )
    raw = runner.generate(
        prompt,
        image_paths=image_paths,
        video_paths=video_paths,
    )
    print(
        "qa_stage_done "
        f"stage={stage} evidence_id={evidence_id} qa_id={qa_id} "
        f"attempt={attempt} seconds={time.time() - stage_start:.1f}",
        flush=True,
    )
    initial_raw = raw
    format_repair = {"attempted": False, "succeeded": False}
    try:
        judge = parse_combined_direct_judge_output(raw)
    except Exception as initial_exc:
        format_repair = {
            "attempted": True,
            "succeeded": False,
            "initial_error": f"{type(initial_exc).__name__}: {initial_exc}",
        }
        repair_prompt = build_judge_json_repair_prompt(raw, JUDGE_SCHEMA)
        repair_start = time.time()
        print(
            "qa_format_repair_start "
            f"stage={stage} evidence_id={evidence_id} qa_id={qa_id} "
            f"attempt={attempt}",
            flush=True,
        )
        try:
            raw = runner.generate(
                repair_prompt,
                image_paths=[],
                video_paths=[],
            )
            judge = parse_combined_direct_judge_output(raw)
            format_repair["succeeded"] = True
        except OpenRouterRequestError:
            raise
        except Exception as repair_exc:
            format_repair["repair_error"] = (
                f"{type(repair_exc).__name__}: {repair_exc}"
            )
            judge = failed_combined_direct_judge(
                "combined direct judge output remained invalid after one "
                f"text-only JSON repair: {repair_exc}"
            )
        print(
            "qa_format_repair_done "
            f"stage={stage} evidence_id={evidence_id} qa_id={qa_id} "
            f"attempt={attempt} succeeded={format_repair['succeeded']} "
            f"seconds={time.time() - repair_start:.1f}",
            flush=True,
        )
    judge["raw_output"] = raw
    if format_repair["attempted"]:
        judge["initial_raw_output"] = initial_raw
        judge["format_repair"] = format_repair
    judge["elapsed_seconds"] = round(time.time() - stage_start, 3)
    judge["gate"] = judge_gate(judge, required_checks=DIRECT_REVIEW_CHECKS)
    return judge


def check_from_single_judge(
    judge: dict[str, Any],
    check_name: str,
    *,
    include_decision_uncertainty: bool = False,
) -> dict[str, Any]:
    def finalize(check: dict[str, Any], *, emitted_status: Any = None) -> dict[str, Any]:
        if not include_decision_uncertainty:
            check.pop("decision_uncertainty", None)
            check.pop("probe_matches_effective_status", None)
            check.pop("status_matches_effective_status", None)
            return check
        return attach_decision_uncertainty(
            check,
            check_name,
            choice_signal=judge.get("choice_logit_signal"),
            emitted_status=emitted_status,
        )

    checks = judge.get("checks")
    if isinstance(checks, dict) and isinstance(checks.get(check_name), dict):
        check = dict(checks[check_name])
        status = str(check.get("status") or "").strip().upper()
        if status not in {"PASS", "FAIL"}:
            return finalize(
                {
                    "status": "FAIL",
                    "reason": f"{check_name} judge did not return status PASS or FAIL",
                    "fix": f"Return checks.{check_name}.status as PASS or FAIL.",
                }
            )
        check["status"] = status
        return finalize(check, emitted_status=status)
    return finalize(
        {
            "status": "FAIL",
            "reason": f"{check_name} judge did not return checks.{check_name}",
            "fix": f"Return a valid {check_name} check object.",
        }
    )


def merge_parallel_judges(
    *,
    qa_formality_judge: dict[str, Any],
    evidence_groundedness_judge: dict[str, Any],
    answerability: dict[str, Any],
    schema_errors: list[str],
    qa_item: dict[str, Any] | None = None,
    participant_names: list[str] | tuple[str, ...] | None = None,
    include_decision_uncertainty: bool = False,
    quality_quota_by_check: dict[str, dict[str, int]] | None = None,
) -> dict[str, Any]:
    schema_errors = qa_formality_errors(
        qa_item or {},
        schema_errors,
        participant_names=participant_names,
    )
    schema_branch = schema_formality_branch(schema_errors)
    qa_formality_check = check_from_single_judge(
        qa_formality_judge,
        "qa_formality",
        include_decision_uncertainty=include_decision_uncertainty,
    )
    model_qa_formality_check = dict(qa_formality_check)
    semantic_subchecks = qa_formality_check.get("semantic_subchecks")
    semantic_failures = []
    for subcheck_name in QA_FORMALITY_SEMANTIC_SUBCHECK_NAMES:
        subcheck = (
            semantic_subchecks.get(subcheck_name)
            if isinstance(semantic_subchecks, dict)
            else None
        )
        if not isinstance(subcheck, dict):
            semantic_failures.append(f"{subcheck_name} missing")
            continue
        status = str(subcheck.get("status") or "").strip().upper()
        if status != "PASS":
            detail = str(subcheck.get("reason") or "").strip()
            semantic_failures.append(
                f"{subcheck_name} {status.lower() if status else 'invalid'}"
                + (f": {detail}" if detail else "")
            )
    if semantic_failures:
        qa_formality_check["status"] = "FAIL"
        existing_reason = str(qa_formality_check.get("reason") or "").strip()
        semantic_reason = "semantic subchecks failed: " + "; ".join(semantic_failures)
        qa_formality_check["reason"] = (
            f"{existing_reason}; {semantic_reason}" if existing_reason else semantic_reason
        )
        qa_formality_check["fix"] = (
            "Repair every failed or missing formality subcheck: use natural first-person or "
            "shared-memory wording, clarify references and options, replace any concurrent-activity "
            "report with a concrete missing object, identity, state, location, outcome, consequence, "
            "explanation, interaction result, or follow-up, and remove participant names and "
            "timestamp citations."
        )
    if schema_branch["status"] != "PASS":
        qa_formality_check["status"] = "FAIL"
        qa_formality_check["reason"] = (
            schema_branch["reason"]
            + "; model qa_formality branch: "
            + str(model_qa_formality_check.get("reason", ""))
        )
        qa_formality_check["fix"] = (
            "Repair the generated JSON shape, multiple-choice options, correct letter, "
            "answer text, required users, required question-answer metadata, and any known "
            "participant-name leakage."
        )
    qa_formality_check["schema_branch"] = schema_branch
    qa_formality_check["model_branch"] = model_qa_formality_check
    if include_decision_uncertainty:
        qa_formality_check = attach_decision_uncertainty(qa_formality_check, "qa_formality")

    evidence_check = check_from_single_judge(
        evidence_groundedness_judge,
        "evidence_groundedness",
        include_decision_uncertainty=include_decision_uncertainty,
    )
    if quality_quota_by_check:
        qa_formality_quota = quality_quota_by_check.get("qa_formality")
        if isinstance(qa_formality_quota, dict):
            qa_formality_check = attach_quality_quota_metadata(
                qa_formality_check,
                quota_state=qa_formality_quota,
            )
        evidence_quota = quality_quota_by_check.get("evidence_groundedness")
        if isinstance(evidence_quota, dict):
            evidence_check = attach_quality_quota_metadata(
                evidence_check,
                quota_state=evidence_quota,
            )
    answerability_check = answerability_check_from_gate(answerability)

    combined = {
        "review_passed": True,
        "checks": {
            "qa_formality": qa_formality_check,
            "evidence_groundedness": evidence_check,
            "answerability": answerability_check,
        },
        "blocking_failures": [],
        "why_generator_asked_this": (
            qa_formality_judge.get("why_generator_asked_this")
            or evidence_groundedness_judge.get("why_generator_asked_this")
            or ""
        ),
        "feedback_to_generator": "",
        "branches": {
            "qa_formality": qa_formality_judge,
            "evidence_groundedness": evidence_groundedness_judge,
            "answerability": answerability,
        },
    }

    if include_decision_uncertainty:
        entropy_by_check = {}
        unavailable_entropy_checks = []
        for check_name in sorted(LEGACY_DECISION_ENTROPY_JUDGE_CHECKS):
            uncertainty = combined["checks"][check_name].get("decision_uncertainty") or {}
            if uncertainty.get("available") is True:
                entropy_by_check[check_name] = float(uncertainty["normalized_entropy"])
            else:
                unavailable_entropy_checks.append(check_name)
        entropy_values = list(entropy_by_check.values())
        combined["decision_uncertainty_summary"] = {
            "available": not unavailable_entropy_checks,
            "normalized_entropy_by_check": entropy_by_check,
            "mean_normalized_entropy": (
                round(sum(entropy_values) / len(entropy_values), 8) if entropy_values else None
            ),
            "max_normalized_entropy": round(max(entropy_values), 8) if entropy_values else None,
            "unavailable_checks": unavailable_entropy_checks,
            "note": (
                "each entropy value comes from a second independent minimal-verdict "
                "call; the detailed production judges alone drive the gate"
            ),
        }

    feedback = []
    for check_name, check in combined["checks"].items():
        if str(check.get("status", "")).upper() != "PASS":
            combined["blocking_failures"].append(check_name)
            reason = str(check.get("reason") or "")
            fix = str(check.get("fix") or "")
            feedback.append(f"{check_name}: {reason} {fix}".strip())
    combined["review_passed"] = not combined["blocking_failures"]
    combined["feedback_to_generator"] = " | ".join(feedback)
    combined["gate"] = judge_gate(combined)
    return combined


def answerability_check_from_gate(answerability: dict[str, Any] | None) -> dict[str, Any]:
    """Expose the deterministic answerability gate as a structured judge check."""

    if not isinstance(answerability, dict):
        return {
            "status": "FAIL",
            "reason": "answerability judge did not return a result",
            "fix": "Run the answerability evaluator and return its gate result.",
        }
    gate = answerability.get("gate")
    if not isinstance(gate, dict):
        return {
            "status": "FAIL",
            "reason": "answerability judge did not return gate metadata",
            "fix": "Return answerability.gate with passed and reason fields.",
        }
    reason = str(gate.get("reason") or "")
    if gate.get("passed") is True:
        check = {
            "status": "PASS",
            "reason": reason or "answerability gate passed",
            "fix": "",
        }
        if gate.get("warning"):
            check["warning"] = gate.get("warning")
        if gate.get("evidence_provider_answerable"):
            check["evidence_provider_answerable"] = gate.get("evidence_provider_answerable")
        return check
    return {
        "status": "FAIL",
        "reason": reason or "answerability gate failed",
        "fix": (
            (
                "Revise the question-answer item so the speaker video alone lacks the needed "
                "visual evidence and the six combined videos contain it."
            )
            if gate.get("answerability_mode") == "shared_fact_visibility_audit"
            else (
                "Revise the question-answer item so the combined required users select the correct "
                "answer and the asker/subset conditions do not."
            )
        ),
    }


def build_review_from_gates(
    *,
    judge: dict[str, Any] | None,
    answerability: dict[str, Any] | None,
    schema_errors: list[str] | None,
    accepted: bool,
    rejection_stage: str | None = None,
    final_reason: str | None = None,
) -> dict[str, Any]:
    """Build the final review object stored inside each question-answer row.

    Generator self-checks stay in generation_trace. The final review is derived
    from the model/deterministic judges, answerability evaluator, and final schema validation.
    """

    schema_errors = list(schema_errors or [])
    schema_passed = not schema_errors
    if accepted:
        status = "passed"
    elif rejection_stage == "judger":
        status = "rejected_by_judger"
    elif rejection_stage == "answerability":
        status = "rejected_by_answerability"
    else:
        status = "rejected_by_schema"

    return {
        "status": status,
        "review_passed": bool(accepted),
        "judger": compact_trace_payload(judge) if isinstance(judge, dict) else {},
        "answerability": compact_answerability_for_checkpoint(answerability),
        "schema_validation": {
            "passed": schema_passed,
            "errors": schema_errors,
        },
        "final_decision": {
            "accepted": bool(accepted),
            "rejection_stage": None if accepted else (rejection_stage or "schema"),
            "reason": final_reason or ("passed all gates" if accepted else "rejected"),
        },
    }


def production_entropy_rows_for_attempt(
    *,
    judge: dict[str, Any],
    evidence_id: Any,
    qa_id: Any,
    attempt: int,
    attempt_outcome: str,
) -> list[dict[str, Any]]:
    """Flatten the two independent minimal-verdict measurements for analysis."""

    checks = judge.get("checks") if isinstance(judge.get("checks"), dict) else {}
    branches = judge.get("branches") if isinstance(judge.get("branches"), dict) else {}
    rows = []
    for judge_name in sorted(LEGACY_DECISION_ENTROPY_JUDGE_CHECKS):
        check = checks.get(judge_name) if isinstance(checks.get(judge_name), dict) else {}
        branch = (
            branches.get(judge_name)
            if isinstance(branches.get(judge_name), dict)
            else {}
        )
        uncertainty = (
            check.get("decision_uncertainty")
            if isinstance(check.get("decision_uncertainty"), dict)
            else {
                "available": False,
                "reason": "effective production check omitted decision_uncertainty",
            }
        )
        probabilities = (
            uncertainty.get("probabilities")
            if isinstance(uncertainty.get("probabilities"), dict)
            else {}
        )
        log_weights = (
            uncertainty.get("log_weights")
            if isinstance(uncertainty.get("log_weights"), dict)
            else {}
        )
        probe_verdict = str(
            branch.get("entropy_probe_verdict")
            or uncertainty.get("generated_decision")
            or ""
        ).lower()
        probe_verdict_status = (
            probe_verdict.upper()
            if probe_verdict in FIRST_VERDICT_CHOICES
            else ""
        )
        production_status = str(check.get("status") or "").upper()
        probe_matches_production = (
            probe_verdict_status == production_status
            if probe_verdict_status in {"PASS", "FAIL"}
            and production_status in {"PASS", "FAIL"}
            else None
        )
        minimal_probe = (
            branch.get("minimal_entropy_probe")
            if isinstance(branch.get("minimal_entropy_probe"), dict)
            else {}
        )
        rows.append(
            {
                "evidence_id": evidence_id,
                "qa_id": qa_id,
                "attempt": attempt,
                "judge": judge_name,
                "probe_verdict": probe_verdict,
                "probe_verdict_status": probe_verdict_status,
                "production_status": production_status,
                "probe_matches_production_status": probe_matches_production,
                "minimal_probe_raw_output": minimal_probe.get("raw_output"),
                # Compatibility aliases for older entropy-analysis readers.
                "model_verdict": probe_verdict,
                "model_verdict_status": probe_verdict_status,
                "effective_status": production_status,
                "model_verdict_matches_effective_status": probe_matches_production,
                "attempt_outcome": attempt_outcome,
                "combined_judge_gate_passed": (
                    (judge.get("gate") or {}).get("passed") is True
                ),
                "entropy_available": uncertainty.get("available") is True,
                "entropy_unavailable_reason": str(
                    uncertainty.get("reason") or ""
                ),
                "probability_pass": probabilities.get("PASS"),
                "probability_fail": probabilities.get("FAIL"),
                "log_weight_pass": log_weights.get("PASS"),
                "log_weight_fail": log_weights.get("FAIL"),
                "entropy_nats": uncertainty.get("entropy_nats"),
                "entropy_bits": uncertainty.get("entropy_bits"),
                "normalized_entropy": uncertainty.get("normalized_entropy"),
                "argmax_decision": uncertainty.get("argmax_decision"),
                "generated_decision": uncertainty.get("generated_decision"),
                "generated_matches_argmax": (
                    uncertainty.get("generated_decision")
                    == uncertainty.get("argmax_decision")
                    if uncertainty.get("available") is True
                    else None
                ),
                "token_index": uncertainty.get("token_index"),
                "probe_version": uncertainty.get("probe_version"),
                "measurement_context": uncertainty.get("measurement_context"),
                "decision_role": uncertainty.get("decision_role"),
                "verdict_affects_acceptance": uncertainty.get(
                    "verdict_affects_acceptance"
                ),
                "entropy_used_as_threshold": uncertainty.get(
                    "entropy_used_as_threshold"
                ),
                "probe_output_contract_valid": uncertainty.get(
                    "probe_output_contract_valid"
                ),
                "measurement_contract_valid": uncertainty.get(
                    "measurement_contract_valid"
                ),
                "independent_from_acceptance_gate": uncertainty.get(
                    "independent_from_acceptance_gate"
                ),
            }
        )
    return rows


def _production_entropy_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    available = [
        row
        for row in rows
        if row.get("entropy_available") is True
        and isinstance(row.get("normalized_entropy"), (int, float))
    ]
    entropies = [float(row["normalized_entropy"]) for row in available]
    return {
        "row_count": len(rows),
        "available_count": len(available),
        "availability_rate": len(available) / len(rows) if rows else None,
        "mean_normalized_entropy": (
            statistics.fmean(entropies) if entropies else None
        ),
        "median_normalized_entropy": (
            statistics.median(entropies) if entropies else None
        ),
        "minimum_normalized_entropy": min(entropies) if entropies else None,
        "maximum_normalized_entropy": max(entropies) if entropies else None,
        "probe_verdict_counts": dict(
            sorted(Counter(str(row.get("probe_verdict_status")) for row in rows).items())
        ),
        "production_status_counts": dict(
            sorted(Counter(str(row.get("production_status")) for row in rows).items())
        ),
        "probe_production_mismatch_count": sum(
            row.get("probe_matches_production_status") is False
            for row in rows
        ),
        # Compatibility aliases for older report consumers.
        "model_verdict_counts": dict(
            sorted(Counter(str(row.get("probe_verdict_status")) for row in rows).items())
        ),
        "effective_status_counts": dict(
            sorted(Counter(str(row.get("production_status")) for row in rows).items())
        ),
        "model_effective_mismatch_count": sum(
            row.get("probe_matches_production_status") is False
            for row in rows
        ),
        "generated_argmax_mismatch_count": sum(
            row.get("generated_matches_argmax") is False for row in available
        ),
    }


def summarize_production_judge_entropy(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize entropy by judge, attempt result, and final packet outcome."""

    def grouped(field: str) -> dict[str, Any]:
        values = sorted({str(row.get(field) or "missing") for row in rows})
        return {
            value: _production_entropy_group(
                [row for row in rows if str(row.get(field) or "missing") == value]
            )
            for value in values
        }

    joint_keys = sorted(
        {
            (
                str(row.get("judge") or "missing"),
                str(row.get("packet_final_status") or "missing"),
            )
            for row in rows
        }
    )
    return {
        "experiment": "independent_minimal_judge_verdict_entropy",
        "measurement": (
            "restricted softmax over lowercase pass/fail logits at the only "
            "verdict field in a second minimal judge call"
        ),
        "causal_role": (
            "the old detailed production judge drives acceptance, retries, and feedback; "
            "the independent probe verdict and entropy are diagnostic only"
        ),
        "overall": _production_entropy_group(rows),
        "by_judge": grouped("judge"),
        "by_probe_verdict": grouped("probe_verdict_status"),
        "by_production_status": grouped("production_status"),
        # Compatibility aliases.
        "by_model_verdict": grouped("probe_verdict_status"),
        "by_effective_status": grouped("production_status"),
        "by_attempt_outcome": grouped("attempt_outcome"),
        "by_retry_followed": grouped("retry_followed"),
        "by_packet_final_status": grouped("packet_final_status"),
        "by_judge_and_packet_final_status": {
            f"{judge}:{status}": _production_entropy_group(
                [
                    row
                    for row in rows
                    if str(row.get("judge") or "missing") == judge
                    and str(row.get("packet_final_status") or "missing") == status
                ]
            )
            for judge, status in joint_keys
        },
    }


def production_entropy_report_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Independent Minimal Judge-Probe Entropy",
        "",
        (
            "Each model judge runs twice. The old detailed judge controls production; "
            "the second call returns only `{\"verdict\":\"pass\"}` or "
            "`{\"verdict\":\"fail\"}`. The second verdict and its entropy cannot "
            "change acceptance, retries, or feedback."
        ),
        "",
        "| Group | Rows | Available | Mean normalized entropy | Median | Probe/production mismatches |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for group_name, group in summary.get("by_judge", {}).items():
        lines.append(
            f"| judge={group_name} | {group['row_count']} | "
            f"{group['available_count']} | "
            f"{group['mean_normalized_entropy'] if group['mean_normalized_entropy'] is not None else 'NA'} | "
            f"{group['median_normalized_entropy'] if group['median_normalized_entropy'] is not None else 'NA'} | "
            f"{group['probe_production_mismatch_count']} |"
        )
    for group_name, group in summary.get("by_attempt_outcome", {}).items():
        lines.append(
            f"| attempt={group_name} | {group['row_count']} | "
            f"{group['available_count']} | "
            f"{group['mean_normalized_entropy'] if group['mean_normalized_entropy'] is not None else 'NA'} | "
            f"{group['median_normalized_entropy'] if group['median_normalized_entropy'] is not None else 'NA'} | "
            f"{group['probe_production_mismatch_count']} |"
        )
    for group_name, group in summary.get("by_packet_final_status", {}).items():
        lines.append(
            f"| final={group_name} | {group['row_count']} | "
            f"{group['available_count']} | "
            f"{group['mean_normalized_entropy'] if group['mean_normalized_entropy'] is not None else 'NA'} | "
            f"{group['median_normalized_entropy'] if group['median_normalized_entropy'] is not None else 'NA'} | "
            f"{group['probe_production_mismatch_count']} |"
        )
    lines.append("")
    return "\n".join(lines)


def generator_decode_config(
    *,
    generator_decode_mode: str,
    generator_temperature: float,
    generator_top_p: float,
    generator_top_k: int | None,
) -> dict[str, Any]:
    return {
        "mode": generator_decode_mode,
        "temperature": generator_temperature,
        "top_p": generator_top_p,
        "top_k": generator_top_k,
    }


def dry_run_discovered_relation(packet: dict[str, Any], question_type: str) -> dict[str, Any]:
    """Archived discovery-mode dry-run fixture for old artifact tests."""

    users = packet.get("required_users", [])[:2]
    speaker = users[0] if users else "User A"
    other = users[1] if len(users) > 1 else "User B"
    return {
        "category": "reference_and_viewpoint_resolution",
        "need": "dry-run discovered cross-user information need",
        "speaker_user": speaker,
        "other_required_users": [other],
        "what_speaker_knows_sees": "dry-run speaker-side visual anchor",
        "what_others_know_see": {other: "dry-run missing visual detail"},
        "only_clear_when_combining": f"dry-run {question_type} relation requiring both users",
        "why_natural_to_ask": "dry-run placeholder for prompt plumbing",
        "likely_answerable_by_one_video_alone": "no, dry-run placeholder",
    }


def dry_run_qa(packet: dict[str, Any], question_type: str, generation_mode: str = "baseline") -> dict[str, Any]:
    users = packet.get("required_users", [])[:2]
    clips = packet.get("clips", [])
    if clips_require_frame_inputs(clips):
        dry_run_media = {
            "image_paths": [path for clip in clips for path in clip_image_paths(clip)],
            "video_paths": [],
        }
    else:
        dry_run_media = {
            "image_paths": [],
            "video_paths": [
                path for clip in clips if (path := clip_video_path(clip))
            ],
        }
    return {
        "qa_id": f"DRYRUN_{packet.get('evidence_id')}_{question_type}",
        "question_type": question_type,
        "generation_mode": generation_mode,
        "question": "Which option can be determined only after comparing what we each experienced?",
        "options": ["Option A", "Option B", "Option C", "Option D", "Option E"],
        "correct": "A",
        "answer": "Option A",
        "required_users": users,
        "evidence": [{"user": user, "needed_fact": "dry-run video evidence", "frames_used": []} for user in users],
        "single_user_answerability": {user: "insufficient in dry-run mode" for user in users},
        "combined_answerability": "sufficient in dry-run prompt construction only",
        "generator_rationale": "dry-run placeholder",
        "why_two_users_needed": "dry-run placeholder",
        "per_user_evidence_claims": [{"user": user, "claim": "dry-run placeholder"} for user in users],
        "attempt_count": 0,
        "review": {
            "review_passed": False,
            "status": "dry_run",
            "judger": {},
            "answerability": {},
            "schema_validation": {"passed": False, "errors": []},
            "final_decision": {
                "accepted": False,
                "rejection_stage": "dry_run",
                "reason": "No model review was run in dry-run mode.",
            },
        },
        "model_id": "dry-run-no-model",
        "source_urls": packet.get("source_urls", {}),
        "video_evidence": video_evidence_for_packet(packet),
        "referred_timestamps": [],
        "human_audit": human_audit_packet(packet),
        "generation_trace": [
            {
                "attempt": 0,
                "stage": "dry_run",
                "question_type": question_type,
                "note": "No model was called; prompts and media paths were generated for plumbing validation.",
                "media": dry_run_media,
            }
        ],
    }


def run_sequential_fact_answerability_eval(
    *,
    qa_item: dict[str, Any],
    packet: dict[str, Any],
    runner: Any,
    prompt_rows: list[dict[str, Any]],
    attempt: int | None,
) -> dict[str, Any]:
    """Run speaker first, then batch the independent remainder when supported."""

    required_users = [str(user) for user in qa_item.get("required_users") or []]
    if len(required_users) != 6:
        raise ValueError("sequential factual answerability requires exactly six users")
    qa_for_prompt = answerability_qa_for_prompt(qa_item)
    plan_prompt = build_answerability_fact_plan_prompt(qa_for_prompt)
    plan_start = time.time()
    print(
        "qa_stage_start "
        f"stage=answerability_fact_plan qa_id={qa_item.get('qa_id')} videos=0",
        flush=True,
    )
    plan_raw = runner.generate(plan_prompt, image_paths=[], video_paths=[])
    plan_elapsed = round(time.time() - plan_start, 3)
    prompt_rows.append(
        compact_prompt_record(
            {
                "stage": "answerability_fact_plan",
                "qa_id": qa_item.get("qa_id"),
                "attempt": attempt,
                "generation_mode": qa_item.get("generation_mode"),
                "prompt": plan_prompt,
                "image_paths": [],
                "video_paths": [],
                "media_role": "text_only",
                "elapsed_seconds": plan_elapsed,
            }
        )
    )
    print(
        "qa_stage_done "
        f"stage=answerability_fact_plan qa_id={qa_item.get('qa_id')} "
        f"seconds={plan_elapsed:.1f}",
        flush=True,
    )
    try:
        plan_value = extract_json_object(plan_raw)
        fact_plan, plan_error = validated_answerability_fact_plan(plan_value)
    except Exception as exc:
        fact_plan, plan_error = None, f"parse_failed: {exc}"
    if fact_plan is None:
        return {
            "fact_plan": {
                "status": "invalid",
                "error": plan_error,
                "raw_output_chars": len(plan_raw),
            },
            "user_audits": [],
            "evaluations": [],
            "gate": {
                "passed": False,
                "reason": f"shared answerability fact plan was invalid: {plan_error}",
                "failure_label": "shared_fact_plan_invalid",
                "answerability_mode": "shared_fact_visibility_audit",
            },
        }

    fact_ids = [str(row["fact_id"]) for row in fact_plan["needed_facts"]]
    segment_media = six_user_source_segment_media(packet, required_users)

    def audit_user(user: str) -> dict[str, Any]:
        user_media = segment_media[user]
        video_paths = list(user_media["video_paths"])
        audit_prompt = build_answerability_user_fact_audit_prompt(
            qa_for_prompt,
            user=user,
            fact_plan=fact_plan,
            segment_count=int(user_media["segment_count"]),
        )
        audit_start = time.time()
        print(
            "qa_stage_start "
            f"stage=answerability_user_fact_audit qa_id={qa_item.get('qa_id')} "
            f"user={user} videos={len(video_paths)}",
            flush=True,
        )
        audit_raw = runner.generate(
            audit_prompt,
            image_paths=[],
            video_paths=video_paths,
        )
        audit_elapsed = round(time.time() - audit_start, 3)
        prompt_rows.append(
            compact_prompt_record(
                {
                    "stage": "answerability_user_fact_audit",
                    "qa_id": qa_item.get("qa_id"),
                    "attempt": attempt,
                    "generation_mode": qa_item.get("generation_mode"),
                    "user": user,
                    "segment_count": int(user_media["segment_count"]),
                    "prompt": audit_prompt,
                    "image_paths": [],
                    "video_paths": video_paths,
                    "media_role": "ordered_source_segments_one_user",
                    "elapsed_seconds": audit_elapsed,
                }
            )
        )
        print(
            "qa_stage_done "
            f"stage=answerability_user_fact_audit qa_id={qa_item.get('qa_id')} "
            f"user={user} seconds={audit_elapsed:.1f}",
            flush=True,
        )
        try:
            audit_value = extract_json_object(audit_raw)
            audit, audit_error = validated_answerability_fact_audit(
                audit_value,
                expected_fact_ids=fact_ids,
                allowed_users=[user],
            )
        except Exception as exc:
            audit, audit_error = None, f"parse_failed: {exc}"
        if audit is None:
            audit = {
                "reason": f"user audit invalid: {audit_error}",
                "fact_audits": [],
            }
        return {
            "user": user,
            **audit,
            "segment_count": int(user_media["segment_count"]),
            "elapsed_seconds": audit_elapsed,
            "raw_output_chars": len(audit_raw),
            "validation_error": audit_error,
        }

    def aggregate_condition(
        condition: dict[str, Any],
        user_audits: list[dict[str, Any]],
    ) -> dict[str, Any]:
        included_users = set(str(user) for user in condition["users"])
        included_audits = [
            {
                "user": audit["user"],
                "reason": audit["reason"],
                "fact_audits": audit["fact_audits"],
            }
            for audit in user_audits
            if audit["user"] in included_users
        ]
        aggregation_prompt = build_answerability_condition_aggregation_prompt(
            qa_for_prompt,
            condition=condition,
            fact_plan=fact_plan,
            user_audits=included_audits,
        )
        aggregation_start = time.time()
        print(
            "qa_stage_start "
            f"stage=answerability_condition_aggregation qa_id={qa_item.get('qa_id')} "
            f"condition_id={condition['condition_id']} videos=0",
            flush=True,
        )
        aggregation_raw = runner.generate(
            aggregation_prompt,
            image_paths=[],
            video_paths=[],
        )
        aggregation_elapsed = round(time.time() - aggregation_start, 3)
        prompt_rows.append(
            compact_prompt_record(
                {
                    "stage": "answerability_condition_aggregation",
                    "qa_id": qa_item.get("qa_id"),
                    "attempt": attempt,
                    "generation_mode": qa_item.get("generation_mode"),
                    "condition_id": condition["condition_id"],
                    "prompt": aggregation_prompt,
                    "image_paths": [],
                    "video_paths": [],
                    "media_role": "text_only_user_audit_reduction",
                    "elapsed_seconds": aggregation_elapsed,
                }
            )
        )
        print(
            "qa_stage_done "
            f"stage=answerability_condition_aggregation qa_id={qa_item.get('qa_id')} "
            f"condition_id={condition['condition_id']} "
            f"seconds={aggregation_elapsed:.1f}",
            flush=True,
        )
        try:
            aggregation_value = extract_json_object(aggregation_raw)
            aggregation, aggregation_error = validated_answerability_fact_audit(
                aggregation_value,
                expected_fact_ids=fact_ids,
                allowed_users=[str(user) for user in condition["users"]],
            )
        except Exception as exc:
            aggregation, aggregation_error = None, f"parse_failed: {exc}"
        if aggregation is None:
            aggregation = {
                "reason": f"condition aggregation invalid: {aggregation_error}",
                "fact_audits": [],
            }
        return {
            **condition,
            **aggregation,
            "shared_fact_ids": fact_ids,
            "elapsed_seconds": aggregation_elapsed,
            "raw_output_chars": len(aggregation_raw),
            "validation_error": aggregation_error,
            "media_summary": {
                "media_role": "per_user_ordered_source_segments",
                "user_count": len(condition["users"]),
                "segment_count": sum(
                    int(segment_media[str(user)]["segment_count"])
                    for user in condition["users"]
                ),
                "exact_media_mapping": "omitted; resolve by evidence_id",
            },
        }

    conditions = build_answerability_conditions(required_users)
    user_audits = [audit_user(required_users[0])]
    evaluations = [aggregate_condition(conditions[0], user_audits)]
    speaker_answerable, speaker_error = parsed_answerability_sufficiency(
        evaluations[0]
    )
    if speaker_error or speaker_answerable:
        gate = answerability_gate(
            qa_item,
            evaluations,
            six_user_judge_mode=SIX_USER_JUDGE_MODE_SEQUENTIAL,
        )
        return {
            "fact_plan": {
                **fact_plan,
                "raw_output_chars": len(plan_raw),
                "elapsed_seconds": plan_elapsed,
            },
            "user_audits": user_audits,
            "evaluations": evaluations,
            "gate": gate,
            "early_exit": "speaker_only",
            "remaining_user_audit_execution": "skipped_speaker_only",
            "condition_aggregation_execution": "skipped_speaker_only",
        }

    remaining_users = required_users[1:]
    supports_batching = bool(
        getattr(runner, "supports_concurrent_batching", False)
    )
    if supports_batching:
        begin_batch = getattr(runner, "begin_concurrent_batch", None)
        release_batch = getattr(runner, "release_concurrent_batch", None)
        batch_held = bool(
            callable(begin_batch) and begin_batch(len(remaining_users))
        )
        with ThreadPoolExecutor(max_workers=len(remaining_users)) as executor:
            remaining_futures = [
                executor.submit(audit_user, user) for user in remaining_users
            ]
            if batch_held and callable(release_batch):
                queued = release_batch()
                print(
                    "qa_batch_barrier "
                    f"stage=answerability_remaining_user_audits "
                    f"expected={len(remaining_users)} queued={queued}",
                    flush=True,
                )
            user_audits.extend(future.result() for future in remaining_futures)
        remaining_audit_execution = "concurrent_vllm_batch"
    else:
        user_audits.extend(audit_user(user) for user in remaining_users)
        remaining_audit_execution = "sequential_backend_fallback"

    minimum_required_users = minimum_required_users_from_fact_audits(
        required_users=required_users,
        user_audits=user_audits,
        fact_ids=fact_ids,
    )
    minimum_condition = {
        "condition_id": "minimum_required_users::" + "+".join(minimum_required_users),
        "condition_type": "minimum_required_users",
        "users": minimum_required_users,
    }
    if supports_batching:
        batch_held = bool(callable(begin_batch) and begin_batch(2))
        with ThreadPoolExecutor(max_workers=2) as executor:
            aggregation_futures = [
                executor.submit(aggregate_condition, condition, user_audits)
                for condition in (conditions[1], minimum_condition)
            ]
            if batch_held and callable(release_batch):
                queued = release_batch()
                print(
                    "qa_batch_barrier "
                    "stage=answerability_condition_aggregations "
                    f"expected=2 queued={queued}",
                    flush=True,
                )
            evaluations.extend(future.result() for future in aggregation_futures)
        aggregation_execution = "concurrent_vllm_batch"
    else:
        evaluations.append(aggregate_condition(conditions[1], user_audits))
        evaluations.append(aggregate_condition(minimum_condition, user_audits))
        aggregation_execution = "sequential_backend_fallback"
    gate = answerability_gate(
        qa_item,
        evaluations,
        six_user_judge_mode=SIX_USER_JUDGE_MODE_SEQUENTIAL,
    )
    return {
        "fact_plan": {
            **fact_plan,
            "raw_output_chars": len(plan_raw),
            "elapsed_seconds": plan_elapsed,
        },
        "user_audits": user_audits,
        "evaluations": evaluations,
        "gate": gate,
        "early_exit": None,
        "minimum_required_users": minimum_required_users,
        "remaining_user_audit_execution": remaining_audit_execution,
        "condition_aggregation_execution": aggregation_execution,
        "trial_order": [
            "combined_all_six_users",
            "minimum_required_users",
        ],
    }


def run_answerability_eval(
    *,
    qa_item: dict[str, Any],
    packet: dict[str, Any],
    runner: Any,
    media_backend: str,
    allow_openai_video_input: bool,
    prompt_rows: list[dict[str, Any]],
    judge_media_role: str = "full",
    attempt: int | None = None,
    six_user_judge_mode: str = SIX_USER_JUDGE_MODE_TIME_AWARE,
) -> dict[str, Any]:
    required_users = [str(user) for user in qa_item.get("required_users") or []]
    if (
        len(required_users) == 6
        and six_user_judge_mode == SIX_USER_JUDGE_MODE_SEQUENTIAL
    ):
        return run_sequential_fact_answerability_eval(
            qa_item=qa_item,
            packet=packet,
            runner=runner,
            prompt_rows=prompt_rows,
            attempt=attempt,
        )
    if (
        len(required_users) == 6
        and six_user_judge_mode == SIX_USER_JUDGE_MODE_TIME_AWARE
    ):
        qa_for_prompt = answerability_qa_for_prompt(qa_item)
        plan_prompt = build_answerability_fact_plan_prompt(qa_for_prompt)
        plan_start = time.time()
        print(
            "qa_stage_start "
            f"stage=answerability_fact_plan qa_id={qa_item.get('qa_id')} videos=0",
            flush=True,
        )
        plan_raw = runner.generate(plan_prompt, image_paths=[], video_paths=[])
        plan_elapsed = round(time.time() - plan_start, 3)
        prompt_rows.append(
            compact_prompt_record(
                {
                    "stage": "answerability_fact_plan",
                    "qa_id": qa_item.get("qa_id"),
                    "attempt": attempt,
                    "generation_mode": qa_item.get("generation_mode"),
                    "prompt": plan_prompt,
                    "image_paths": [],
                    "video_paths": [],
                    "media_role": "text_only",
                    "elapsed_seconds": plan_elapsed,
                }
            )
        )
        print(
            "qa_stage_done "
            f"stage=answerability_fact_plan qa_id={qa_item.get('qa_id')} "
            f"seconds={plan_elapsed:.1f}",
            flush=True,
        )
        try:
            plan_value = extract_json_object(plan_raw)
            fact_plan, plan_error = validated_answerability_fact_plan(plan_value)
        except Exception as exc:
            fact_plan, plan_error = None, f"parse_failed: {exc}"
        if fact_plan is None:
            return {
                "fact_plan": {
                    "status": "invalid",
                    "error": plan_error,
                    "raw_output_chars": len(plan_raw),
                },
                "user_audits": [],
                "evaluations": [],
                "gate": {
                    "passed": False,
                    "reason": f"shared answerability fact plan was invalid: {plan_error}",
                    "failure_label": "shared_fact_plan_invalid",
                    "answerability_mode": "shared_fact_visibility_audit",
                },
            }

        fact_ids = [
            str(row["fact_id"])
            for row in fact_plan["needed_facts"]
        ]
        segment_media = six_user_source_segment_media(packet, required_users)
        user_audits = []
        for user in required_users:
            user_media = segment_media[user]
            video_paths = list(user_media["video_paths"])
            audit_prompt = build_answerability_user_fact_audit_prompt(
                qa_for_prompt,
                user=user,
                fact_plan=fact_plan,
                segment_count=int(user_media["segment_count"]),
            )
            audit_start = time.time()
            print(
                "qa_stage_start "
                f"stage=answerability_user_fact_audit qa_id={qa_item.get('qa_id')} "
                f"user={user} videos={len(video_paths)}",
                flush=True,
            )
            audit_raw = runner.generate(
                audit_prompt,
                image_paths=[],
                video_paths=video_paths,
            )
            audit_elapsed = round(time.time() - audit_start, 3)
            prompt_rows.append(
                compact_prompt_record(
                    {
                        "stage": "answerability_user_fact_audit",
                        "qa_id": qa_item.get("qa_id"),
                        "attempt": attempt,
                        "generation_mode": qa_item.get("generation_mode"),
                        "user": user,
                        "segment_count": int(user_media["segment_count"]),
                        "prompt": audit_prompt,
                        "image_paths": [],
                        "video_paths": video_paths,
                        "media_role": "ordered_source_segments_one_user",
                        "elapsed_seconds": audit_elapsed,
                    }
                )
            )
            print(
                "qa_stage_done "
                f"stage=answerability_user_fact_audit qa_id={qa_item.get('qa_id')} "
                f"user={user} seconds={audit_elapsed:.1f}",
                flush=True,
            )
            try:
                audit_value = extract_json_object(audit_raw)
                audit, audit_error = validated_answerability_fact_audit(
                    audit_value,
                    expected_fact_ids=fact_ids,
                    allowed_users=[user],
                )
            except Exception as exc:
                audit, audit_error = None, f"parse_failed: {exc}"
            if audit is None:
                audit = {
                    "reason": f"user audit invalid: {audit_error}",
                    "fact_audits": [],
                }
            user_audits.append(
                {
                    "user": user,
                    **audit,
                    "segment_count": int(user_media["segment_count"]),
                    "elapsed_seconds": audit_elapsed,
                    "raw_output_chars": len(audit_raw),
                    "validation_error": audit_error,
                }
            )

        evaluations = []
        for condition in build_answerability_conditions(required_users):
            included_users = set(str(user) for user in condition["users"])
            included_audits = [
                {
                    "user": audit["user"],
                    "reason": audit["reason"],
                    "fact_audits": audit["fact_audits"],
                }
                for audit in user_audits
                if audit["user"] in included_users
            ]
            aggregation_prompt = build_answerability_condition_aggregation_prompt(
                qa_for_prompt,
                condition=condition,
                fact_plan=fact_plan,
                user_audits=included_audits,
            )
            aggregation_start = time.time()
            print(
                "qa_stage_start "
                f"stage=answerability_condition_aggregation qa_id={qa_item.get('qa_id')} "
                f"condition_id={condition['condition_id']} videos=0",
                flush=True,
            )
            aggregation_raw = runner.generate(
                aggregation_prompt,
                image_paths=[],
                video_paths=[],
            )
            aggregation_elapsed = round(time.time() - aggregation_start, 3)
            prompt_rows.append(
                compact_prompt_record(
                    {
                        "stage": "answerability_condition_aggregation",
                        "qa_id": qa_item.get("qa_id"),
                        "attempt": attempt,
                        "generation_mode": qa_item.get("generation_mode"),
                        "condition_id": condition["condition_id"],
                        "prompt": aggregation_prompt,
                        "image_paths": [],
                        "video_paths": [],
                        "media_role": "text_only_user_audit_reduction",
                        "elapsed_seconds": aggregation_elapsed,
                    }
                )
            )
            print(
                "qa_stage_done "
                f"stage=answerability_condition_aggregation qa_id={qa_item.get('qa_id')} "
                f"condition_id={condition['condition_id']} "
                f"seconds={aggregation_elapsed:.1f}",
                flush=True,
            )
            try:
                aggregation_value = extract_json_object(aggregation_raw)
                aggregation, aggregation_error = validated_answerability_fact_audit(
                    aggregation_value,
                    expected_fact_ids=fact_ids,
                    allowed_users=[str(user) for user in condition["users"]],
                )
            except Exception as exc:
                aggregation, aggregation_error = None, f"parse_failed: {exc}"
            if aggregation is None:
                aggregation = {
                    "reason": f"condition aggregation invalid: {aggregation_error}",
                    "fact_audits": [],
                }
            evaluations.append(
                {
                    **condition,
                    **aggregation,
                    "shared_fact_ids": fact_ids,
                    "elapsed_seconds": aggregation_elapsed,
                    "raw_output_chars": len(aggregation_raw),
                    "validation_error": aggregation_error,
                    "media_summary": {
                        "media_role": "per_user_ordered_source_segments",
                        "user_count": len(condition["users"]),
                        "segment_count": sum(
                            int(segment_media[str(user)]["segment_count"])
                            for user in condition["users"]
                        ),
                        "exact_media_mapping": "omitted; resolve by evidence_id",
                    },
                }
            )
        gate = answerability_gate(
            qa_item,
            evaluations,
            six_user_judge_mode=six_user_judge_mode,
        )
        return {
            "fact_plan": {
                **fact_plan,
                "raw_output_chars": len(plan_raw),
                "elapsed_seconds": plan_elapsed,
            },
            "user_audits": user_audits,
            "evaluations": evaluations,
            "gate": gate,
        }

    evaluations = []
    for condition in build_answerability_conditions(qa_item.get("required_users", [])):
        clips = clips_for_users(packet, condition["users"])
        image_paths, video_paths = media_for_clips(
            clips,
            backend=media_backend,
            allow_openai_video_input=allow_openai_video_input,
            media_role=judge_media_role,
        )
        prompt = build_answerability_prompt(qa_item, condition)
        prompt_row = {
            "stage": "answerability",
            "qa_id": qa_item.get("qa_id"),
            "attempt": attempt,
            "generation_mode": qa_item.get("generation_mode"),
            "condition_id": condition["condition_id"],
            "prompt": prompt,
            "image_paths": image_paths,
            "video_paths": video_paths,
            "media_role": judge_media_role,
            "condition_media": condition_media_for_clips(
                condition=condition,
                clips=clips,
                image_paths=image_paths,
                video_paths=video_paths,
                media_role=judge_media_role,
            ),
        }
        stage_start = time.time()
        print(
            "qa_stage_start "
            f"stage=answerability qa_id={qa_item.get('qa_id')} "
            f"condition_id={condition['condition_id']} "
            f"images={len(image_paths)} videos={len(video_paths)}",
            flush=True,
        )
        # Archived inactive answerability-logit experiment:
        # generation = runner.generate_with_choice_logits(..., choices=tuple(OPTION_LETTERS))
        # choice_signal = generation.get("choice_logits")
        # choice_uncertainty = answerability_uncertainty_from_choice_logits(choice_signal)
        # Production answerability now uses ordinary JSON generation only.
        raw = runner.generate(prompt, image_paths=image_paths, video_paths=video_paths)
        elapsed_seconds = round(time.time() - stage_start, 3)
        prompt_row["elapsed_seconds"] = elapsed_seconds
        prompt_rows.append(compact_prompt_record(prompt_row))
        print(
            "qa_stage_done "
            f"stage=answerability qa_id={qa_item.get('qa_id')} "
            f"condition_id={condition['condition_id']} seconds={elapsed_seconds:.1f}",
            flush=True,
        )
        try:
            answer = extract_json_object(raw)
        except Exception as exc:
            answer = {
                "choice": None,
                "answer_text": "",
                "evidence_used": f"parse_failed: {exc}",
            }
        evaluations.append(
            {
                **condition,
                **answer,
                "raw_output": raw,
                "elapsed_seconds": elapsed_seconds,
                "condition_media": condition_media_for_clips(
                    condition=condition,
                    clips=clips,
                    image_paths=image_paths,
                    video_paths=video_paths,
                    media_role=judge_media_role,
                ),
            }
        )
    gate = answerability_gate(
        qa_item,
        evaluations,
        six_user_judge_mode=six_user_judge_mode,
    )
    return {"evaluations": evaluations, "gate": gate}


def validated_evidence_segment_observation(
    value: Any,
    *,
    expected_user: str,
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate a compact, path-free map-stage groundedness observation."""

    if not isinstance(value, dict) or set(value) != {"user", "claims"}:
        return None, "observation must contain only user and claims"
    if str(value.get("user") or "") != expected_user:
        return None, f"observation user must be {expected_user}"
    claims = value.get("claims")
    if not isinstance(claims, list) or not claims:
        return None, "claims must be a non-empty array"
    required_keys = {
        "claim",
        "status",
        "segment_references",
        "visual_description",
    }
    compact_claims = []
    for claim in claims:
        if not isinstance(claim, dict) or set(claim) != required_keys:
            return None, "each claim has unexpected fields"
        if not str(claim.get("claim") or "").strip():
            return None, "each claim must be a non-empty string"
        if claim.get("status") not in {
            "SUPPORTED",
            "CONTRADICTED",
            "NOT_VISIBLE",
            "AMBIGUOUS",
        }:
            return None, "each claim must use a supported status"
        references = claim.get("segment_references")
        if not isinstance(references, list) or any(
            not isinstance(reference, str)
            or re.fullmatch(r"segment_[0-9]{3,}", reference) is None
            for reference in references
        ):
            return None, "segment_references must contain only segment_### labels"
        if not str(claim.get("visual_description") or "").strip():
            return None, "each claim needs a visual_description"
        compact_claims.append(dict(claim))
    return {"user": expected_user, "claims": compact_claims}, None


def run_six_user_groundedness_eval(
    *,
    qa_item: dict[str, Any],
    packet: dict[str, Any],
    runner: Any,
    prompt_rows: list[dict[str, Any]],
    attempt: int,
    collect_choice_logits: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Map each full user timeline separately, then reduce without visual media."""

    required_users = [str(user) for user in qa_item.get("required_users") or []]
    if len(required_users) != 6:
        raise ValueError("six-user groundedness map/reduce requires exactly six users")
    segment_media = six_user_source_segment_media(packet, required_users)
    observations = []
    total_map_elapsed = 0.0
    for user in required_users:
        user_media = segment_media[user]
        video_paths = list(user_media["video_paths"])
        segment_count = int(user_media["segment_count"])
        prompt = build_evidence_segment_observation_prompt(
            qa_item,
            user=user,
            segment_count=segment_count,
        )
        started = time.time()
        print(
            "qa_stage_start "
            f"stage=evidence_segment_observation qa_id={qa_item.get('qa_id')} "
            f"user={user} videos={len(video_paths)}",
            flush=True,
        )
        raw = runner.generate(prompt, image_paths=[], video_paths=video_paths)
        elapsed = round(time.time() - started, 3)
        total_map_elapsed += elapsed
        prompt_rows.append(
            compact_prompt_record(
                {
                    "stage": "evidence_segment_observation",
                    "evidence_id": packet.get("evidence_id"),
                    "qa_id": qa_item.get("qa_id"),
                    "attempt": attempt,
                    "user": user,
                    "segment_count": segment_count,
                    "prompt": prompt,
                    "image_paths": [],
                    "video_paths": video_paths,
                    "media_role": "ordered_source_segments_one_user",
                    "elapsed_seconds": elapsed,
                }
            )
        )
        print(
            "qa_stage_done "
            f"stage=evidence_segment_observation qa_id={qa_item.get('qa_id')} "
            f"user={user} seconds={elapsed:.1f}",
            flush=True,
        )
        try:
            parsed = extract_json_object(raw)
            observation, observation_error = validated_evidence_segment_observation(
                parsed,
                expected_user=user,
            )
        except Exception as exc:
            observation, observation_error = None, f"parse_failed: {exc}"
        observations.append(
            {
                **(
                    observation
                    if observation is not None
                    else {"user": user, "claims": []}
                ),
                "segment_count": segment_count,
                "elapsed_seconds": elapsed,
                "raw_output_chars": len(raw),
                "validation_error": observation_error,
            }
        )

    aggregation_prompt = build_evidence_observation_aggregation_prompt(
        qa_item,
        packet,
        observations=observations,
    )
    entropy_prompt = (
        build_judge_minimal_verdict_probe_prompt(
            aggregation_prompt,
            "evidence_groundedness",
        )
        if collect_choice_logits
        else None
    )
    prompt_rows.append(
        compact_prompt_record(
            {
                "stage": "evidence_groundedness_aggregation",
                "evidence_id": packet.get("evidence_id"),
                "qa_id": qa_item.get("qa_id"),
                "attempt": attempt,
                "prompt": aggregation_prompt,
                "image_paths": [],
                "video_paths": [],
                "media_role": "text_only_per_user_observation_reduction",
                "map_user_count": len(observations),
                "map_segment_count": sum(
                    int(row.get("segment_count") or 0) for row in observations
                ),
                "decision_entropy_requested": collect_choice_logits,
            }
        )
    )
    if entropy_prompt is not None:
        prompt_rows.append(
            compact_prompt_record(
                {
                    "stage": "evidence_groundedness_entropy_probe",
                    "evidence_id": packet.get("evidence_id"),
                    "qa_id": qa_item.get("qa_id"),
                    "attempt": attempt,
                    "prompt": entropy_prompt,
                    "image_paths": [],
                    "video_paths": [],
                    "media_role": "text_only_per_user_observation_reduction",
                    "decision_entropy_requested": True,
                }
            )
        )
    judge = run_model_judge_branch(
        check_name="evidence_groundedness",
        prompt=aggregation_prompt,
        runner=runner,
        image_paths=[],
        video_paths=[],
        evidence_id=packet.get("evidence_id"),
        qa_id=qa_item.get("qa_id"),
        attempt=attempt,
        collect_choice_logits=collect_choice_logits,
        minimal_verdict_probe_prompt=entropy_prompt,
    )
    trace = {
        "mode": "per_user_source_segment_map_reduce",
        "user_count": len(observations),
        "segment_count": sum(
            int(row.get("segment_count") or 0) for row in observations
        ),
        "max_visual_call_segment_count": max(
            int(row.get("segment_count") or 0) for row in observations
        ),
        "map_elapsed_seconds": round(total_map_elapsed, 3),
        "observations": observations,
        "aggregation_prompt": aggregation_prompt,
        "entropy_probe_prompt": entropy_prompt,
    }
    return judge, trace


def run_sequential_separated_review_judges(
    *,
    qa_item: dict[str, Any],
    packet: dict[str, Any],
    schema_errors: list[str],
    runner: Any,
    media_backend: str,
    allow_openai_video_input: bool,
    prompt_rows: list[dict[str, Any]],
    generator_image_paths: list[str],
    generator_video_paths: list[str],
    attempt: int,
    judge_media_role: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Run formality, grounding, and factual answerability as strict serial gates."""

    def skipped_judge(check_name: str, reason: str) -> dict[str, Any]:
        judge = failed_single_judge(check_name, reason)
        judge["skipped"] = True
        return judge

    def skipped_answerability(reason: str, failure_label: str) -> dict[str, Any]:
        return {
            "evaluations": [],
            "gate": {
                "passed": False,
                "reason": reason,
                "failure_label": failure_label,
            },
        }

    def merge_results(
        qa_formality_judge: dict[str, Any],
        evidence_groundedness_judge: dict[str, Any],
        answerability: dict[str, Any],
    ) -> dict[str, Any]:
        return merge_parallel_judges(
            qa_formality_judge=qa_formality_judge,
            evidence_groundedness_judge=evidence_groundedness_judge,
            answerability=answerability,
            schema_errors=schema_errors,
            qa_item=qa_item,
            participant_names=participant_names,
            include_decision_uncertainty=False,
            quality_quota_by_check=None,
        )

    passing_evidence_placeholder = {
        "review_passed": True,
        "checks": {
            "evidence_groundedness": {
                "status": "PASS",
                "reason": "placeholder used only to evaluate the preceding serial gate",
                "fix": "",
            }
        },
        "blocking_failures": [],
        "why_generator_asked_this": "",
        "feedback_to_generator": "",
    }
    passing_answerability_placeholder = {
        "evaluations": [],
        "gate": {
            "passed": True,
            "reason": "placeholder used only to evaluate the preceding serial gate",
        },
    }

    participant_names = formality_participant_names(packet, qa_item)
    schema_errors = qa_formality_errors(
        qa_item,
        schema_errors,
        participant_names=participant_names,
    )
    if schema_errors:
        reason = "deterministic schema/formality checks failed: " + "; ".join(
            schema_errors
        )
        qa_formality_judge = skipped_judge("qa_formality", reason)
        evidence_groundedness_judge = skipped_judge(
            "evidence_groundedness",
            "skipped because deterministic schema checks failed",
        )
        answerability = skipped_answerability(
            "skipped because deterministic schema checks failed",
            "upstream_schema_failed",
        )
        judge = merge_results(
            qa_formality_judge,
            evidence_groundedness_judge,
            answerability,
        )
        return judge, answerability, {
            "parallel": False,
            "execution_order": ["deterministic_schema"],
            "six_user_judge_mode": SIX_USER_JUDGE_MODE_SEQUENTIAL,
            "schema_branch": schema_formality_branch(schema_errors),
            "qa_formality": None,
            "evidence_groundedness": None,
            "answerability": answerability,
            "merged": judge,
        }

    qa_for_prompt = qa_for_judger_prompt(
        qa_item,
        include_generator_rationale=False,
    )
    qa_formality_prompt = build_qa_formality_judge_prompt(
        qa_for_prompt,
        packet,
        schema_errors=schema_errors,
        pass_fail_only=True,
    )
    prompt_rows.append(
        compact_prompt_record(
            {
                "stage": "qa_formality_judge",
                "evidence_id": packet.get("evidence_id"),
                "qa_id": qa_item.get("qa_id"),
                "question_type": qa_item.get("question_type"),
                "generation_mode": qa_item.get("generation_mode"),
                "attempt": attempt,
                "prompt": qa_formality_prompt,
                "image_paths": [],
                "video_paths": [],
                "media_role": "text_only",
                "model_id": getattr(runner, "model_id", None),
                "schema_branch": schema_formality_branch(schema_errors),
                "generator_rationale_included": False,
                "pass_fail_only": True,
                "judge_contract": "legacy_review_passed",
                "authoritative_for_acceptance": True,
            }
        )
    )
    try:
        qa_formality_judge = run_model_judge_branch(
            check_name="qa_formality",
            prompt=qa_formality_prompt,
            runner=runner,
            image_paths=[],
            video_paths=[],
            evidence_id=packet.get("evidence_id"),
            qa_id=qa_item.get("qa_id"),
            attempt=attempt,
        )
    except OpenRouterRequestError:
        raise
    except Exception as exc:
        if is_cuda_oom_error(exc):
            raise JudgeInfrastructureError(
                stage="qa_formality_judge",
                cause=exc,
            ) from exc
        qa_formality_judge = failed_single_judge(
            "qa_formality",
            f"qa_formality judge crashed: {exc}",
        )

    formality_probe = merge_results(
        qa_formality_judge,
        passing_evidence_placeholder,
        passing_answerability_placeholder,
    )
    if (
        str(
            ((formality_probe.get("checks") or {}).get("qa_formality") or {}).get(
                "status"
            )
            or ""
        ).upper()
        != "PASS"
    ):
        evidence_groundedness_judge = skipped_judge(
            "evidence_groundedness",
            "skipped because the text-only qa_formality judge failed",
        )
        answerability = skipped_answerability(
            "skipped because the text-only qa_formality judge failed",
            "upstream_qa_formality_failed",
        )
        judge = merge_results(
            qa_formality_judge,
            evidence_groundedness_judge,
            answerability,
        )
        trace = {
            "parallel": False,
            "execution_order": ["deterministic_schema", "qa_formality_judge"],
            "six_user_judge_mode": SIX_USER_JUDGE_MODE_SEQUENTIAL,
            "schema_branch": schema_formality_branch(schema_errors),
            "qa_formality": {
                "model_id": getattr(runner, "model_id", None),
                "elapsed_seconds": qa_formality_judge.get("elapsed_seconds"),
                "prompt": qa_formality_prompt,
                "parsed": qa_formality_judge,
            },
            "evidence_groundedness": None,
            "answerability": answerability,
            "merged": judge,
        }
        return judge, answerability, trace

    evidence_groundedness_prompt = build_evidence_groundedness_judge_prompt(
        qa_for_prompt,
        packet,
        pass_fail_only=True,
    )
    prompt_rows.append(
        compact_prompt_record(
            {
                "stage": "evidence_groundedness_judge",
                "evidence_id": packet.get("evidence_id"),
                "qa_id": qa_item.get("qa_id"),
                "question_type": qa_item.get("question_type"),
                "generation_mode": qa_item.get("generation_mode"),
                "attempt": attempt,
                "prompt": evidence_groundedness_prompt,
                "image_paths": generator_image_paths,
                "video_paths": generator_video_paths,
                "media_role": "same_sampled_media_as_generator",
                "model_id": getattr(runner, "model_id", None),
                "generator_rationale_included": False,
                "pass_fail_only": True,
                "judge_contract": "legacy_review_passed",
                "authoritative_for_acceptance": True,
            }
        )
    )
    try:
        evidence_groundedness_judge = run_model_judge_branch(
            check_name="evidence_groundedness",
            prompt=evidence_groundedness_prompt,
            runner=runner,
            image_paths=generator_image_paths,
            video_paths=generator_video_paths,
            evidence_id=packet.get("evidence_id"),
            qa_id=qa_item.get("qa_id"),
            attempt=attempt,
        )
    except OpenRouterRequestError:
        raise
    except Exception as exc:
        if is_cuda_oom_error(exc):
            raise JudgeInfrastructureError(
                stage="evidence_groundedness_judge",
                cause=exc,
            ) from exc
        evidence_groundedness_judge = failed_single_judge(
            "evidence_groundedness",
            f"evidence_groundedness judge crashed: {exc}",
        )

    groundedness_probe = merge_results(
        qa_formality_judge,
        evidence_groundedness_judge,
        passing_answerability_placeholder,
    )
    if (
        str(
            (
                (groundedness_probe.get("checks") or {}).get(
                    "evidence_groundedness"
                )
                or {}
            ).get("status")
            or ""
        ).upper()
        != "PASS"
    ):
        answerability = skipped_answerability(
            "skipped because the visual evidence_groundedness judge failed",
            "upstream_evidence_groundedness_failed",
        )
        judge = merge_results(
            qa_formality_judge,
            evidence_groundedness_judge,
            answerability,
        )
        trace = {
            "parallel": False,
            "execution_order": [
                "deterministic_schema",
                "qa_formality_judge",
                "evidence_groundedness_judge",
            ],
            "six_user_judge_mode": SIX_USER_JUDGE_MODE_SEQUENTIAL,
            "schema_branch": schema_formality_branch(schema_errors),
            "judge_media_role": "same_sampled_media_as_generator",
            "qa_formality": {
                "model_id": getattr(runner, "model_id", None),
                "elapsed_seconds": qa_formality_judge.get("elapsed_seconds"),
                "prompt": qa_formality_prompt,
                "parsed": qa_formality_judge,
            },
            "evidence_groundedness": {
                "model_id": getattr(runner, "model_id", None),
                "elapsed_seconds": evidence_groundedness_judge.get(
                    "elapsed_seconds"
                ),
                "prompt": evidence_groundedness_prompt,
                "parsed": evidence_groundedness_judge,
            },
            "answerability": answerability,
            "merged": judge,
        }
        return judge, answerability, trace

    answerability_prompt_rows: list[dict[str, Any]] = []
    try:
        answerability = run_answerability_eval(
            qa_item=qa_item,
            packet=packet,
            runner=runner,
            media_backend=media_backend,
            allow_openai_video_input=allow_openai_video_input,
            prompt_rows=answerability_prompt_rows,
            judge_media_role=judge_media_role,
            attempt=attempt,
            six_user_judge_mode=SIX_USER_JUDGE_MODE_SEQUENTIAL,
        )
    except OpenRouterRequestError:
        raise
    except Exception as exc:
        if is_cuda_oom_error(exc):
            raise JudgeInfrastructureError(
                stage="answerability",
                cause=exc,
            ) from exc
        answerability = {
            "evaluations": [],
            "gate": {
                "passed": False,
                "reason": f"answerability judge crashed: {exc}",
            },
        }
    for row in answerability_prompt_rows:
        prompt_rows.append(compact_prompt_record(row))

    judge = merge_results(
        qa_formality_judge,
        evidence_groundedness_judge,
        answerability,
    )
    trace = {
        "parallel": False,
        "execution_order": [
            "deterministic_schema",
            "qa_formality_judge",
            "evidence_groundedness_judge",
            "answerability_fact_plan",
            "answerability_speaker_audit",
            "answerability_remaining_user_audits_if_needed",
            "answerability_all_six_trial",
            "answerability_minimum_required_users_trial",
        ],
        "six_user_judge_mode": SIX_USER_JUDGE_MODE_SEQUENTIAL,
        "schema_branch": schema_formality_branch(schema_errors),
        "judge_media_role": "same_sampled_media_as_generator",
        "qa_formality": {
            "model_id": getattr(runner, "model_id", None),
            "elapsed_seconds": qa_formality_judge.get("elapsed_seconds"),
            "prompt": qa_formality_prompt,
            "parsed": qa_formality_judge,
        },
        "evidence_groundedness": {
            "model_id": getattr(runner, "model_id", None),
            "elapsed_seconds": evidence_groundedness_judge.get("elapsed_seconds"),
            "prompt": evidence_groundedness_prompt,
            "parsed": evidence_groundedness_judge,
        },
        "answerability": answerability,
        "merged": judge,
    }
    return judge, answerability, trace


def run_parallel_review_judges(
    *,
    qa_item: dict[str, Any],
    packet: dict[str, Any],
    schema_errors: list[str],
    runner: Any,
    qa_formality_runner: Any | None = None,
    media_backend: str,
    allow_openai_video_input: bool,
    prompt_rows: list[dict[str, Any]],
    full_image_paths: list[str],
    full_video_paths: list[str],
    attempt: int,
    judge_media_role: str = "full",
    include_generator_rationale: bool = False,
    pass_fail_only: bool = True,
    quality_quota_counts: dict[str, int] | None = None,
    quality_quota: int = DEFAULT_QUALITY_QUOTA,
    record_decision_entropy: bool = False,
    six_user_judge_mode: str = SIX_USER_JUDGE_MODE_TIME_AWARE,
    generator_image_paths: list[str] | None = None,
    generator_video_paths: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Run qa_formality, evidence_groundedness, and answerability in parallel."""

    is_six_user_sequential = (
        len(qa_item.get("required_users") or []) == 6
        and six_user_judge_mode == SIX_USER_JUDGE_MODE_SEQUENTIAL
    )
    if is_six_user_sequential:
        if record_decision_entropy:
            raise ValueError(
                "sequential-separated-fact-audit does not support decision-entropy probes"
            )
        if not getattr(runner, "supports_concurrent_batching", False):
            return run_sequential_separated_review_judges(
                qa_item=qa_item,
                packet=packet,
                schema_errors=schema_errors,
                runner=runner,
                media_backend=media_backend,
                allow_openai_video_input=allow_openai_video_input,
                prompt_rows=prompt_rows,
                generator_image_paths=(
                    list(generator_image_paths)
                    if generator_image_paths is not None
                    else list(full_image_paths)
                ),
                generator_video_paths=(
                    list(generator_video_paths)
                    if generator_video_paths is not None
                    else list(full_video_paths)
                ),
                attempt=attempt,
                judge_media_role=judge_media_role,
            )

    active_qa_formality_runner = qa_formality_runner or runner
    include_generator_rationale = False
    participant_names = formality_participant_names(packet, qa_item)
    schema_errors = qa_formality_errors(
        qa_item,
        schema_errors,
        participant_names=participant_names,
    )
    # Production is unconditionally binary-only. Compatibility parameters remain so
    # old offline callers still import cleanly, but cannot reactivate scoring here.
    # Archived scored/quota activation:
    # active_quota_counts = quality_quota_counts
    # if not pass_fail_only:
    #     active_quota_counts = active_quota_counts or {name: 0 for name in QUALITY_SCORED_JUDGE_CHECKS}
    #     quality_quota_by_check = {
    #         name: quality_quota_snapshot(active_quota_counts[name], quality_quota)
    #         for name in QUALITY_SCORED_JUDGE_CHECKS
    #     }
    pass_fail_only = True
    active_quota_counts: dict[str, int] | None = None
    quality_quota_by_check: dict[str, dict[str, int]] | None = None
    point_scoring_mode = "legacy_archived_not_active"
    qa_for_prompt = qa_for_judger_prompt(
        qa_item,
        include_generator_rationale=include_generator_rationale,
    )
    is_six_user = len(qa_for_prompt.get("required_users") or []) == 6
    six_user_map_reduce = (
        is_six_user
        and six_user_judge_mode == SIX_USER_JUDGE_MODE_TIME_AWARE
    )
    groundedness_image_paths = (
        list(generator_image_paths)
        if is_six_user_sequential and generator_image_paths is not None
        else list(full_image_paths)
    )
    groundedness_video_paths = (
        list(generator_video_paths)
        if is_six_user_sequential and generator_video_paths is not None
        else list(full_video_paths)
    )
    qa_formality_prompt = build_qa_formality_judge_prompt(
        qa_for_prompt,
        packet,
        schema_errors=schema_errors,
        pass_fail_only=True,
    )
    evidence_groundedness_prompt = (
        None
        if six_user_map_reduce
        else build_evidence_groundedness_judge_prompt(
            qa_for_prompt,
            packet,
            pass_fail_only=True,
        )
    )
    qa_formality_entropy_probe_prompt = None
    evidence_groundedness_entropy_probe_prompt = None
    if record_decision_entropy:
        qa_formality_entropy_probe_prompt = build_judge_minimal_verdict_probe_prompt(
            qa_formality_prompt,
            "qa_formality",
        )
        if evidence_groundedness_prompt is not None:
            evidence_groundedness_entropy_probe_prompt = (
                build_judge_minimal_verdict_probe_prompt(
                    evidence_groundedness_prompt,
                    "evidence_groundedness",
                )
            )
    prompt_rows.append(
        compact_prompt_record({
            "stage": "qa_formality_judge",
            "evidence_id": packet.get("evidence_id"),
            "qa_id": qa_item.get("qa_id"),
            "question_type": qa_item.get("question_type"),
            "generation_mode": qa_item.get("generation_mode"),
            "attempt": attempt,
            "prompt": qa_formality_prompt,
            "image_paths": [],
            "video_paths": [],
            "media_role": "text_only",
            "model_id": getattr(active_qa_formality_runner, "model_id", None),
            "schema_branch": schema_formality_branch(schema_errors),
            "generator_rationale_included": False,
            "pass_fail_only": True,
            "judge_contract": "legacy_review_passed",
            "decision_entropy_requested": False,
            "authoritative_for_acceptance": True,
            "entropy_probe_affects_acceptance": False,
            "point_scoring": point_scoring_mode,
        })
    )
    if evidence_groundedness_prompt is not None:
        prompt_rows.append(
            compact_prompt_record({
                "stage": "evidence_groundedness_judge",
                "evidence_id": packet.get("evidence_id"),
                "qa_id": qa_item.get("qa_id"),
                "question_type": qa_item.get("question_type"),
                "generation_mode": qa_item.get("generation_mode"),
                "attempt": attempt,
                "prompt": evidence_groundedness_prompt,
                "image_paths": groundedness_image_paths,
                "video_paths": groundedness_video_paths,
                "media_role": judge_media_role,
                "model_id": getattr(runner, "model_id", None),
                "generator_rationale_included": include_generator_rationale,
                "pass_fail_only": True,
                "judge_contract": "legacy_review_passed",
                "decision_entropy_requested": False,
                "authoritative_for_acceptance": True,
                "entropy_probe_affects_acceptance": False,
                "point_scoring": point_scoring_mode,
            })
        )
    if record_decision_entropy:
        entropy_prompt_rows = [
                {
                    "stage": "qa_formality_entropy_probe",
                    "evidence_id": packet.get("evidence_id"),
                    "qa_id": qa_item.get("qa_id"),
                    "question_type": qa_item.get("question_type"),
                    "generation_mode": qa_item.get("generation_mode"),
                    "attempt": attempt,
                    "prompt": qa_formality_entropy_probe_prompt,
                    "image_paths": [],
                    "video_paths": [],
                    "media_role": "text_only",
                    "model_id": getattr(active_qa_formality_runner, "model_id", None),
                    "pass_fail_only": True,
                    "judge_contract": "minimal_verdict_only",
                    "decision_entropy_requested": True,
                    "authoritative_for_acceptance": False,
                    "entropy_probe_affects_acceptance": False,
                    "point_scoring": point_scoring_mode,
                },
            ]
        if evidence_groundedness_entropy_probe_prompt is not None:
            entropy_prompt_rows.append(
                {
                    "stage": "evidence_groundedness_entropy_probe",
                    "evidence_id": packet.get("evidence_id"),
                    "qa_id": qa_item.get("qa_id"),
                    "question_type": qa_item.get("question_type"),
                    "generation_mode": qa_item.get("generation_mode"),
                    "attempt": attempt,
                    "prompt": evidence_groundedness_entropy_probe_prompt,
                    "image_paths": groundedness_image_paths,
                    "video_paths": groundedness_video_paths,
                    "media_role": judge_media_role,
                    "model_id": getattr(runner, "model_id", None),
                    "pass_fail_only": True,
                    "judge_contract": "minimal_verdict_only",
                    "decision_entropy_requested": True,
                    "authoritative_for_acceptance": False,
                    "entropy_probe_affects_acceptance": False,
                    "point_scoring": point_scoring_mode,
                }
            )
        for entropy_prompt_row in entropy_prompt_rows:
            prompt_rows.append(compact_prompt_record(entropy_prompt_row))

    answerability_prompt_rows: list[dict[str, Any]] = []
    groundedness_prompt_rows: list[dict[str, Any]] = []
    begin_batch = getattr(runner, "begin_concurrent_batch", None)
    release_batch = getattr(runner, "release_concurrent_batch", None)
    expected_runner_requests = 3 if active_qa_formality_runner is runner else 2
    batch_held = bool(
        callable(begin_batch) and begin_batch(expected_runner_requests)
    )
    with ThreadPoolExecutor(max_workers=3) as executor:
        qa_formality_future = executor.submit(
            run_model_judge_branch,
            check_name="qa_formality",
            prompt=qa_formality_prompt,
            runner=active_qa_formality_runner,
            image_paths=[],
            video_paths=[],
            evidence_id=packet.get("evidence_id"),
            qa_id=qa_item.get("qa_id"),
            attempt=attempt,
            collect_choice_logits=record_decision_entropy,
            minimal_verdict_probe_prompt=qa_formality_entropy_probe_prompt,
        )
        if six_user_map_reduce:
            evidence_groundedness_future = executor.submit(
                run_six_user_groundedness_eval,
                qa_item=qa_for_prompt,
                packet=packet,
                runner=runner,
                prompt_rows=groundedness_prompt_rows,
                attempt=attempt,
                collect_choice_logits=record_decision_entropy,
            )
        else:
            evidence_groundedness_future = executor.submit(
                run_model_judge_branch,
                check_name="evidence_groundedness",
                prompt=evidence_groundedness_prompt,
                runner=runner,
                image_paths=groundedness_image_paths,
                video_paths=groundedness_video_paths,
                evidence_id=packet.get("evidence_id"),
                qa_id=qa_item.get("qa_id"),
                attempt=attempt,
                collect_choice_logits=record_decision_entropy,
                minimal_verdict_probe_prompt=evidence_groundedness_entropy_probe_prompt,
            )
        answerability_future = executor.submit(
            run_answerability_eval,
            qa_item=qa_item,
            packet=packet,
            runner=runner,
            media_backend=media_backend,
            allow_openai_video_input=allow_openai_video_input,
            prompt_rows=answerability_prompt_rows,
            judge_media_role=judge_media_role,
            attempt=attempt,
            six_user_judge_mode=six_user_judge_mode,
        )
        if batch_held and callable(release_batch):
            queued = release_batch()
            print(
                "qa_batch_barrier "
                f"stage=parallel_review_entry expected={expected_runner_requests} "
                f"queued={queued}",
                flush=True,
            )

        try:
            qa_formality_judge = qa_formality_future.result()
        except OpenRouterRequestError:
            raise
        except Exception as exc:
            if is_cuda_oom_error(exc):
                raise JudgeInfrastructureError(
                    stage="qa_formality_judge",
                    cause=exc,
                ) from exc
            qa_formality_judge = failed_single_judge("qa_formality", f"qa_formality judge crashed: {exc}")
        groundedness_map_trace: dict[str, Any] = {}
        try:
            groundedness_result = evidence_groundedness_future.result()
            if six_user_map_reduce:
                evidence_groundedness_judge, groundedness_map_trace = groundedness_result
            else:
                evidence_groundedness_judge = groundedness_result
        except OpenRouterRequestError:
            raise
        except Exception as exc:
            if is_cuda_oom_error(exc):
                raise JudgeInfrastructureError(
                    stage=(
                        "evidence_groundedness_map_reduce"
                        if six_user_map_reduce
                        else "evidence_groundedness_judge"
                    ),
                    cause=exc,
                ) from exc
            evidence_groundedness_judge = failed_single_judge(
                "evidence_groundedness",
                f"evidence_groundedness judge crashed: {exc}",
            )
        try:
            answerability = answerability_future.result()
        except OpenRouterRequestError:
            raise
        except Exception as exc:
            if is_cuda_oom_error(exc):
                raise JudgeInfrastructureError(
                    stage="answerability",
                    cause=exc,
                ) from exc
            answerability = {
                "evaluations": [],
                "gate": {
                    "passed": False,
                    "reason": f"answerability judge crashed: {exc}",
                },
            }

    for row in answerability_prompt_rows:
        prompt_rows.append(compact_prompt_record(row))
    for row in groundedness_prompt_rows:
        prompt_rows.append(compact_prompt_record(row))

    judge = merge_parallel_judges(
        qa_formality_judge=qa_formality_judge,
        evidence_groundedness_judge=evidence_groundedness_judge,
        answerability=answerability,
        schema_errors=schema_errors,
        qa_item=qa_item,
        participant_names=participant_names,
        include_decision_uncertainty=record_decision_entropy,
        quality_quota_by_check=None,
    )
    for check_name in QUALITY_SCORED_JUDGE_CHECKS:
        check = (judge.get("checks") or {}).get(check_name)
        if not isinstance(check, dict):
            continue
        # Do not retain stray fields from the archived point/logit contracts even if
        # a model emits them despite the binary production schema.
        archived_fields = [
            "quality_score",
            "quality_flag",
            "quality_reason",
            "quota_rebuttal",
            "quality_quota",
            "quality_uncertainty",
        ]
        if not record_decision_entropy:
            archived_fields.extend(
                [
                    "decision_uncertainty",
                    "probe_matches_effective_status",
                    "status_matches_effective_status",
                ]
            )
        for archived_field in archived_fields:
            check.pop(archived_field, None)
    # Archived quota-counter update:
    # if quality_quota_by_check and active_quota_counts is not None: ...
    trace = {
        "parallel": True,
        "single_packet_batching": bool(
            getattr(runner, "supports_concurrent_batching", False)
        ),
        "parallel_entry_batch_barrier": batch_held,
        "parallel_entry_expected_requests": expected_runner_requests,
        "six_user_judge_mode": six_user_judge_mode if is_six_user else None,
        "schema_branch": schema_formality_branch(schema_errors),
        "generator_rationale_included": include_generator_rationale,
        "judge_media_role": judge_media_role,
        "pass_fail_only": True,
        "judge_contract": (
            "legacy_detailed_production_plus_independent_minimal_probe"
            if record_decision_entropy
            else "legacy_review_passed"
        ),
        "pass_fail_entropy_logits": (
            "independent_minimal_probe_recorded"
            if record_decision_entropy
            else "not_collected"
        ),
        "verdict_entropy_gate_relation": (
            "independent_probe_does_not_affect_gate"
            if record_decision_entropy
            else "not_applicable"
        ),
        "answerability_choice_logits": "legacy_archived_not_collected",
        "point_scoring": point_scoring_mode,
        "qa_formality": {
            "model_id": getattr(active_qa_formality_runner, "model_id", None),
            "elapsed_seconds": qa_formality_judge.get("elapsed_seconds"),
            "generator_rationale_included": False,
            "prompt": qa_formality_prompt,
            "entropy_probe_prompt": qa_formality_entropy_probe_prompt,
            "raw_output": qa_formality_judge.get("raw_output"),
            "parsed": qa_formality_judge,
        },
        "evidence_groundedness": {
            "model_id": getattr(runner, "model_id", None),
            "elapsed_seconds": evidence_groundedness_judge.get("elapsed_seconds"),
            "generator_rationale_included": include_generator_rationale,
            "mode": (
                "per_user_source_segment_map_reduce"
                if six_user_map_reduce
                else (
                    "single_generator_media_call"
                    if is_six_user_sequential
                    else "single_full_media_call"
                )
            ),
            "prompt": (
                groundedness_map_trace.get("aggregation_prompt")
                if six_user_map_reduce
                else evidence_groundedness_prompt
            ),
            "entropy_probe_prompt": (
                groundedness_map_trace.get("entropy_probe_prompt")
                if six_user_map_reduce
                else evidence_groundedness_entropy_probe_prompt
            ),
            "map_reduce": groundedness_map_trace if six_user_map_reduce else None,
            "raw_output": evidence_groundedness_judge.get("raw_output"),
            "parsed": evidence_groundedness_judge,
        },
        "answerability": answerability,
        "answerability_model_id": getattr(runner, "model_id", None),
        "merged": judge,
    }
    # Archived quota trace emission:
    # trace["quality_quota"] = {...}
    return judge, answerability, trace


def _read_jsonl_if_present(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    return list(iter_jsonl(path))


def _run_bounded_packet_pipeline(
    call_args: dict[str, Any],
    *,
    active_question_types: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Run a bounded packet pipeline with one generation and N review lanes."""

    width = int(call_args["max_packets_in_flight"])
    review_lanes = int(call_args["max_review_lanes"])
    target_count = int(call_args["target_count"])
    packet_limit = call_args.get("packet_limit")
    skipped_ids = {
        str(value).strip()
        for value in (call_args.get("skip_evidence_ids") or ())
        if str(value).strip()
    }
    selected: list[tuple[int, dict[str, Any]]] = []
    for packet_index, packet in enumerate(iter_jsonl(call_args["evidence_path"])):
        if packet_index >= target_count:
            break
        if str(packet.get("evidence_id") or "") in skipped_ids:
            continue
        if packet_limit is not None and len(selected) >= int(packet_limit):
            break
        selected.append((packet_index, packet))

    group_ids = [
        str(packet.get("generation_group_id") or packet.get("evidence_id") or "")
        for _, packet in selected
    ]
    if len(set(group_ids)) != len(group_ids):
        raise ValueError(
            "max_packets_in_flight > 1 requires unique generation_group_id values"
        )

    output_path = Path(call_args["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    optional_outputs = {
        "prompts_path": "prompts.jsonl",
        "rejected_path": "rejected.jsonl",
        "intermediate_path": "intermediate.jsonl",
        "infrastructure_skipped_path": "infrastructure_skipped.jsonl",
    }
    write_jsonl(output_path, [])
    for argument_name in optional_outputs:
        if call_args.get(argument_name):
            write_jsonl(call_args[argument_name], [])

    generation_gate = BoundedSemaphore(value=1)
    review_gate = BoundedSemaphore(value=review_lanes)
    pipeline_started = time.time()
    print(
        "qa_packet_pipeline_start "
        f"packets={len(selected)} max_packets_in_flight={width} "
        f"generation_lanes=1 review_lanes={review_lanes}",
        flush=True,
    )
    accepted_rows: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(
        prefix=".qa_packet_pipeline_",
        dir=output_path.parent,
    ) as temporary_root_value:
        temporary_root = Path(temporary_root_value)

        def run_packet(
            packet_index: int,
            packet: dict[str, Any],
        ) -> dict[str, list[dict[str, Any]]]:
            packet_root = temporary_root / f"packet_{packet_index:06d}"
            packet_root.mkdir(parents=True, exist_ok=True)
            evidence_path = packet_root / "evidence.jsonl"
            write_jsonl(evidence_path, [packet])
            packet_args = dict(call_args)
            packet_args.update(
                {
                    "evidence_path": evidence_path,
                    "output_path": packet_root / "accepted.jsonl",
                    "judge_entropy_path": None,
                    "judge_entropy_summary_path": None,
                    "judge_entropy_report_path": None,
                    "target_count": 1,
                    "fixed_question_type_schedule": True,
                    "question_types": (
                        active_question_types[
                            packet_index % len(active_question_types)
                        ],
                    ),
                    "resume": False,
                    "skip_evidence_ids": (),
                    "packet_limit": 1,
                    "max_packets_in_flight": 1,
                    "max_review_lanes": 1,
                    "_packet_index_offset": packet_index,
                    "_generation_gate": generation_gate,
                    "_review_gate": review_gate,
                }
            )
            for argument_name, filename in optional_outputs.items():
                packet_args[argument_name] = (
                    packet_root / filename
                    if call_args.get(argument_name)
                    else None
                )
            print(
                "qa_packet_pipeline_packet_start "
                f"packet_index={packet_index} evidence_id={packet.get('evidence_id')}",
                flush=True,
            )
            packet_accepted = generate_video_qa_loop(**packet_args)
            result = {"accepted": packet_accepted}
            for argument_name in optional_outputs:
                result[argument_name] = _read_jsonl_if_present(
                    packet_args.get(argument_name)
                )
            print(
                "qa_packet_pipeline_packet_done "
                f"packet_index={packet_index} evidence_id={packet.get('evidence_id')} "
                f"accepted={len(packet_accepted)}",
                flush=True,
            )
            return result

        with ThreadPoolExecutor(max_workers=width) as executor:
            active_futures = {
                executor.submit(run_packet, packet_index, packet): selection_index
                for selection_index, (packet_index, packet) in enumerate(
                    selected[:width]
                )
            }
            next_selection_index = len(active_futures)
            next_merge_index = 0
            completed_by_index: dict[int, dict[str, list[dict[str, Any]]]] = {}
            while active_futures:
                done, _ = wait(active_futures, return_when=FIRST_COMPLETED)
                for future in done:
                    selection_index = active_futures.pop(future)
                    completed_by_index[selection_index] = future.result()
                    if next_selection_index < len(selected):
                        packet_index, packet = selected[next_selection_index]
                        replacement = executor.submit(
                            run_packet,
                            packet_index,
                            packet,
                        )
                        active_futures[replacement] = next_selection_index
                        next_selection_index += 1

                # Refill immediately after any completion so a slow first packet
                # cannot drain the worker window.  Merge in source selection order.
                while next_merge_index in completed_by_index:
                    result = completed_by_index.pop(next_merge_index)
                    for row in result["accepted"]:
                        accepted_rows.append(row)
                        append_jsonl(output_path, row)
                    for argument_name in optional_outputs:
                        destination = call_args.get(argument_name)
                        for row in result[argument_name]:
                            if destination:
                                append_jsonl(destination, row)
                    next_merge_index += 1

    print(
        "qa_packet_pipeline_done "
        f"packets={len(selected)} accepted={len(accepted_rows)} "
        f"seconds={time.time() - pipeline_started:.1f}",
        flush=True,
    )
    return accepted_rows


def generate_video_qa_loop(
    *,
    evidence_path: str | Path,
    output_path: str | Path,
    prompts_path: str | Path | None,
    rejected_path: str | Path | None,
    intermediate_path: str | Path | None = None,
    infrastructure_skipped_path: str | Path | None = None,
    judge_entropy_path: str | Path | None = None,
    judge_entropy_summary_path: str | Path | None = None,
    judge_entropy_report_path: str | Path | None = None,
    backend: str,
    model_id: str = DEFAULT_MODEL_ID,
    base_url: str = "http://127.0.0.1:8000/v1",
    target_count: int = 20,
    max_attempts: int = 3,
    max_new_tokens: int = 1536,
    max_image_pixels: int = 262144,
    dtype: str = "bfloat16",
    allow_cpu: bool = False,
    allow_openai_video_input: bool = False,
    disable_thinking: bool = False,
    api_key: str | None = None,
    judge_backend: str | None = None,
    judge_model_id: str | None = None,
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
    judge_max_new_tokens: int | None = None,
    judge_reasoning_effort: str | None = None,
    qa_formality_use_generator: bool = False,
    judge_video_source: str = "full",
    six_user_judge_mode: str = SIX_USER_JUDGE_MODE_TIME_AWARE,
    judge_include_generator_rationale: bool = False,
    judge_pass_fail_only: bool = True,
    judge_quality_quota: int = DEFAULT_QUALITY_QUOTA,
    record_judge_decision_entropy: bool = False,
    dry_run: bool = False,
    generation_mode: str = "baseline",
    fixed_question_type_schedule: bool = False,
    question_types: tuple[str, ...] | None = None,
    resume: bool = False,
    generator_decode_mode: str = "greedy",
    generator_temperature: float = DEFAULT_SAMPLING_TEMPERATURE,
    generator_top_p: float = DEFAULT_SAMPLING_TOP_P,
    generator_top_k: int | None = None,
    skip_evidence_ids: Sequence[str] | None = None,
    packet_limit: int | None = None,
    max_packets_in_flight: int = 1,
    max_review_lanes: int = 1,
    _packet_index_offset: int | None = None,
    _generation_gate: BoundedSemaphore | None = None,
    _review_gate: BoundedSemaphore | None = None,
) -> list[dict[str, Any]]:
    judge_include_generator_rationale = False
    # Archived scored/quota production switch:
    # judge_pass_fail_only = caller-provided value
    # judge_quality_quota = caller-provided value
    # The live pipeline always uses ordinary JSON PASS/FAIL review.
    judge_pass_fail_only = True
    if generation_mode not in GENERATION_MODES:
        raise ValueError(f"unknown generation_mode: {generation_mode}")
    if generator_decode_mode not in GENERATOR_DECODING_MODES:
        raise ValueError(f"unknown generator_decode_mode: {generator_decode_mode}")
    if judge_video_source not in JUDGE_VIDEO_SOURCES:
        raise ValueError(
            f"unknown judge_video_source {judge_video_source!r}; "
            f"expected one of {JUDGE_VIDEO_SOURCES}"
        )
    if six_user_judge_mode not in SIX_USER_JUDGE_MODES:
        raise ValueError(
            f"unknown six_user_judge_mode {six_user_judge_mode!r}; "
            f"expected one of {SIX_USER_JUDGE_MODES}"
        )
    if (
        six_user_judge_mode == SIX_USER_JUDGE_MODE_SEQUENTIAL
        and record_judge_decision_entropy
    ):
        raise ValueError(
            "sequential-separated-fact-audit does not support decision-entropy probes"
        )
    if packet_limit is not None and packet_limit < 1:
        raise ValueError("packet_limit must be at least 1 when provided")
    configured_skip_evidence_ids = {
        str(evidence_id).strip()
        for evidence_id in (skip_evidence_ids or ())
        if str(evidence_id).strip()
    }
    # Archived scored-quota validation:
    # if not judge_pass_fail_only and judge_quality_quota < 1: ...
    active_question_types = tuple(question_types or DEFAULT_QUESTION_TYPES)
    if not active_question_types:
        raise ValueError("question_types must include at least one question type")
    unknown_question_types = [
        question_type for question_type in active_question_types if question_type not in QUESTION_TYPES
    ]
    if unknown_question_types:
        raise ValueError(f"unknown question_types: {unknown_question_types}")
    if not 1 <= max_packets_in_flight <= MAX_SAFE_PACKETS_IN_FLIGHT:
        raise ValueError(
            "max_packets_in_flight must be between 1 and 4 for the bounded VLM pipeline"
        )
    max_review_lanes_for_width = max(1, max_packets_in_flight - 1)
    if not 1 <= max_review_lanes <= min(
        MAX_SAFE_REVIEW_LANES,
        max_review_lanes_for_width,
    ):
        raise ValueError(
            "max_review_lanes must be between 1 and min(3, "
            "max_packets_in_flight - 1), with one lane allowed for width 1"
        )
    if max_packets_in_flight > 1:
        if backend != "openai-compatible-local":
            raise ValueError(
                "max_packets_in_flight > 1 is limited to openai-compatible-local"
            )
        active_judge_backend = judge_backend or backend
        if active_judge_backend not in {
            "openai-compatible-local",
            "openrouter",
            "gemini",
        }:
            raise ValueError(
                "max_packets_in_flight > 1 requires an HTTP judge backend"
            )
        if not fixed_question_type_schedule:
            raise ValueError(
                "max_packets_in_flight > 1 requires fixed_question_type_schedule"
            )
        if resume:
            raise ValueError(
                "max_packets_in_flight > 1 currently requires a fresh evidence pass"
            )
        if record_judge_decision_entropy:
            raise ValueError(
                "max_packets_in_flight > 1 does not yet merge entropy metadata"
            )
        current_locals = locals().copy()
        call_args = {
            name: current_locals[name]
            for name in signature(generate_video_qa_loop).parameters
        }
        return _run_bounded_packet_pipeline(
            call_args,
            active_question_types=active_question_types,
        )
    decode_config = generator_decode_config(
        generator_decode_mode=generator_decode_mode,
        generator_temperature=generator_temperature,
        generator_top_p=generator_top_p,
        generator_top_k=generator_top_k,
    )
    active_backend = "dry-run" if dry_run else backend
    active_judge_backend = "dry-run" if dry_run else (judge_backend or backend)
    runner = make_runner(
        active_backend,
        model_id=model_id,
        base_url=base_url,
        max_new_tokens=max_new_tokens,
        max_image_pixels=max_image_pixels,
        dtype=dtype,
        allow_cpu=allow_cpu,
        allow_openai_video_input=allow_openai_video_input,
        disable_thinking=disable_thinking,
        api_key=api_key,
    )
    effective_judge_model_id = judge_model_id or (
        DEFAULT_JUDGE_MODEL_ID if active_judge_backend != active_backend else model_id
    )
    effective_judge_base_url = judge_base_url or base_url
    effective_judge_max_new_tokens = judge_max_new_tokens or max_new_tokens
    effective_judge_api_key = judge_api_key if judge_api_key is not None else api_key
    judge_runner_matches_generator = (
        active_judge_backend == active_backend
        and effective_judge_model_id == model_id
        and effective_judge_base_url == base_url
        and effective_judge_max_new_tokens == max_new_tokens
        and effective_judge_api_key == api_key
        and not judge_reasoning_effort
    )
    judge_runner = runner
    if not judge_runner_matches_generator:
        judge_runner = make_runner(
            active_judge_backend,
            model_id=effective_judge_model_id,
            base_url=effective_judge_base_url,
            max_new_tokens=effective_judge_max_new_tokens,
            max_image_pixels=max_image_pixels,
            dtype=dtype,
            allow_cpu=allow_cpu,
            allow_openai_video_input=allow_openai_video_input,
            disable_thinking=disable_thinking,
            api_key=effective_judge_api_key,
            reasoning_effort=judge_reasoning_effort,
        )
    qa_formality_runner = runner if qa_formality_use_generator else judge_runner
    entropy_tokenizer_preflight: dict[str, Any] = {}
    if record_judge_decision_entropy:
        runners_by_role = {
            "qa_formality": qa_formality_runner,
            "evidence_groundedness": judge_runner,
        }
        verified_runner_ids: dict[int, dict[str, Any]] = {}
        for role, active_runner in runners_by_role.items():
            runner_id = id(active_runner)
            if runner_id not in verified_runner_ids:
                verified_runner_ids[runner_id] = verify_first_verdict_tokenization(
                    active_runner
                )
            entropy_tokenizer_preflight[role] = verified_runner_ids[runner_id]
        print(
            "production_entropy_tokenizer_preflight "
            + json.dumps(entropy_tokenizer_preflight, sort_keys=True),
            flush=True,
        )
    point_scoring_mode = "legacy_archived_not_active"
    judge_contract = (
        "legacy_detailed_production_plus_independent_minimal_probe"
        if record_judge_decision_entropy
        else "binary_pass_fail"
    )
    print(
        "qa_runner_config "
        f"generator_backend={active_backend} generator_model={runner.model_id} "
        f"qa_formality_model={qa_formality_runner.model_id} "
        f"visual_judge_backend={active_judge_backend} visual_judge_model={judge_runner.model_id} "
        f"judge_runner_shared_with_generator={judge_runner is runner} "
        f"visual_judge_reasoning_effort={judge_reasoning_effort or 'provider_default'} "
        f"judge_video_source={judge_video_source} "
        f"six_user_judge_mode={six_user_judge_mode} "
        f"configured_skip_count={len(configured_skip_evidence_ids)} "
        f"packet_limit={packet_limit if packet_limit is not None else 'none'} "
        f"generator_rationale_included={judge_include_generator_rationale} "
        f"judge_contract={judge_contract} "
        f"point_scoring={point_scoring_mode} "
        f"pass_fail_entropy_logits={'independent_minimal_probe_recorded' if record_judge_decision_entropy else 'not_collected'} "
        "answerability_choice_logits=legacy_archived_not_collected",
        flush=True,
    )
    prompts = StreamingJsonlRows(prompts_path, reset=not resume)
    intermediate_rows = StreamingJsonlRows(intermediate_path, reset=not resume)
    accepted = StreamingJsonlRows(output_path, reset=not resume)
    rejected = StreamingJsonlRows(rejected_path, reset=not resume)
    infrastructure_skipped = StreamingJsonlRows(
        infrastructure_skipped_path,
        reset=not resume,
    )
    judge_entropy_rows = StreamingJsonlRows(
        judge_entropy_path if record_judge_decision_entropy else None,
        reset=not resume,
    )
    if resume:
        compact_existing_jsonl(prompts_path, compact_prompt_record)
        compact_existing_jsonl(intermediate_path, compact_existing_intermediate_row)
        accepted.load_existing()
        rejected.load_existing()
        infrastructure_skipped.load_existing()
        judge_entropy_rows.load_existing()
    quality_quota_counts: dict[str, int] | None = None
    # Archived resume-time quota restoration:
    # quota_source_rows = list(intermediate_rows) or [*accepted, *rejected]
    # quality_quota_counts = quality_quota_counts_from_rows(quota_source_rows)
    processed_evidence_ids = {
        str(row.get("evidence_id"))
        for row in [*accepted, *rejected, *infrastructure_skipped]
        if row.get("evidence_id")
    }
    targets = target_type_counts(target_count, active_question_types)
    counts = {question_type: 0 for question_type in active_question_types}
    for row in accepted:
        question_type = row.get("question_type")
        if question_type in counts:
            counts[question_type] += 1
    judge_media_backend = judge_backend or backend
    evaluated_packet_count = 0

    for packet_index, packet in enumerate(iter_jsonl(evidence_path)):
        if fixed_question_type_schedule and packet_index >= target_count:
            break
        if len(accepted) >= target_count:
            break
        evidence_id = str(packet.get("evidence_id") or "")
        if evidence_id in configured_skip_evidence_ids:
            print(
                f"configured_skip evidence_id={evidence_id} "
                f"six_user_judge_mode={six_user_judge_mode}",
                flush=True,
            )
            continue
        if resume and evidence_id in processed_evidence_ids:
            print(f"resume_skip evidence_id={evidence_id}", flush=True)
            continue
        if packet_limit is not None and evaluated_packet_count >= packet_limit:
            print(
                f"packet_limit_reached evaluated_packets={evaluated_packet_count}",
                flush=True,
            )
            break
        evaluated_packet_count += 1
        question_type = (
            active_question_types[packet_index % len(active_question_types)]
            if fixed_question_type_schedule
            else choose_question_type(counts, targets, active_question_types)
        )
        if question_type is None:
            break
        if generation_mode == TEMPORAL_REASONING_MODE:
            packet = packet_with_temporal_reasoning_media(packet)
        clips = packet.get("clips", [])
        image_paths, video_paths = media_for_clips(
            clips,
            backend=backend,
            allow_openai_video_input=allow_openai_video_input,
            media_role="generator",
        )
        full_image_paths, full_video_paths = media_for_clips(
            clips,
            backend=judge_media_backend,
            allow_openai_video_input=allow_openai_video_input,
            media_role=judge_video_source,
        )
        if judge_runner is runner:
            prepared_video_uploads = prepare_runner_video_uploads(
                runner=runner,
                evidence_id=packet.get("evidence_id"),
                generator_video_paths=video_paths,
                full_video_paths=full_video_paths,
                judge_media_role=judge_video_source,
            )
        else:
            prepared_video_uploads = {
                "generator": prepare_runner_video_uploads(
                    runner=runner,
                    evidence_id=packet.get("evidence_id"),
                    generator_video_paths=video_paths,
                    full_video_paths=[],
                    judge_media_role=judge_video_source,
                ),
                "judge": prepare_runner_video_uploads(
                    runner=judge_runner,
                    evidence_id=packet.get("evidence_id"),
                    generator_video_paths=[],
                    full_video_paths=full_video_paths,
                    judge_media_role=judge_video_source,
                ),
            }
        feedback = None
        previous_generation = None
        if dry_run:
            qa = dry_run_qa(packet, question_type, generation_mode=generation_mode)
            # Archived discovery dry-run routing called build_relation_discovery_prompt
            # followed by build_relation_mcq_prompt. Production is baseline-only.
            gen_prompt = build_video_generation_prompt(
                packet,
                question_type,
                generation_mode=generation_mode,
            )
            schema_errors = qa_formality_errors(
                qa,
                validate_qa_item(qa),
                participant_names=formality_participant_names(packet, qa),
            )
            qa_for_prompt = qa_for_judger_prompt(
                qa,
                include_generator_rationale=judge_include_generator_rationale,
            )
            qa_formality_prompt = build_qa_formality_judge_prompt(
                qa_for_prompt,
                packet,
                schema_errors=schema_errors,
                pass_fail_only=True,
            )
            is_six_user_dry_run = len(qa_for_prompt.get("required_users") or []) == 6
            six_user_map_reduce_dry_run = (
                is_six_user_dry_run
                and six_user_judge_mode == SIX_USER_JUDGE_MODE_TIME_AWARE
            )
            sequential_dry_run = (
                is_six_user_dry_run
                and six_user_judge_mode == SIX_USER_JUDGE_MODE_SEQUENTIAL
            )
            six_user_fact_dry_run = (
                six_user_map_reduce_dry_run or sequential_dry_run
            )
            dry_fact_plan = {
                "reason": "The speaker-side reference and answer-bearing detail are required.",
                "needed_facts": [
                    {
                        "fact_id": "F1",
                        "fact": "The speaker-side question reference is visible.",
                        "why_needed": "It grounds the question in the speaker's experience.",
                    },
                    {
                        "fact_id": "F2",
                        "fact": "The answer-bearing external detail is visible.",
                        "why_needed": "It resolves the question.",
                    },
                ],
            }
            dry_observations = [
                {
                    "user": str(user),
                    "claims": [
                        {
                            "claim": "A material question or answer claim is visible.",
                            "status": "SUPPORTED",
                            "segment_references": ["segment_001"],
                            "visual_description": "Placeholder showing the reduce-stage contract.",
                        }
                    ],
                }
                for user in qa_for_prompt.get("required_users") or []
            ]
            evidence_groundedness_prompt = (
                build_evidence_observation_aggregation_prompt(
                    qa_for_prompt,
                    packet,
                    observations=dry_observations,
                )
                if six_user_map_reduce_dry_run
                else (
                    build_sequential_direct_judge_prompt(
                        qa_for_prompt,
                        packet,
                        schema_errors=schema_errors,
                    )
                    if sequential_dry_run
                    else build_evidence_groundedness_judge_prompt(
                        qa_for_prompt,
                        packet,
                        pass_fail_only=True,
                    )
                )
            )
            dry_trace = {
                "evidence_id": packet.get("evidence_id"),
                "qa_id": qa.get("qa_id"),
                "question_type": question_type,
                "generation_mode": generation_mode,
                "six_user_judge_mode": (
                    six_user_judge_mode if is_six_user_dry_run else None
                ),
                "attempt": 1,
                "feedback_in": None,
                "media": {
                    "image_paths": image_paths,
                    "video_paths": video_paths,
                    "media_role": "generator",
                    "full_image_paths": full_image_paths,
                    "full_video_paths": full_video_paths,
                    "judge_image_paths": full_image_paths,
                    "judge_video_paths": full_video_paths,
                    "judge_media_role": judge_video_source,
                    "prepared_video_uploads": prepared_video_uploads,
                    "human_audit": human_audit_packet(packet),
                },
                "generation": {"prompt": gen_prompt, "raw_output": None},
                "generator_decode": decode_config,
                "judge": {
                    "parallel": not sequential_dry_run,
                    "schema_branch": schema_formality_branch(schema_errors),
                    "generator_rationale_included": judge_include_generator_rationale,
                    "judge_media_role": judge_video_source,
                    "pass_fail_only": True,
                    "pass_fail_entropy_logits": "legacy_archived_not_collected",
                    "answerability_choice_logits": "legacy_archived_not_collected",
                    "point_scoring": point_scoring_mode,
                    "qa_formality": {
                        "generator_rationale_included": False,
                        "prompt": qa_formality_prompt,
                        "raw_output": None,
                    },
                    "evidence_groundedness": {
                        "generator_rationale_included": judge_include_generator_rationale,
                        "mode": (
                            "per_user_source_segment_map_reduce"
                            if six_user_map_reduce_dry_run
                            else (
                                "single_generator_media_call"
                                if sequential_dry_run
                                else "single_full_media_call"
                            )
                        ),
                        "prompt": evidence_groundedness_prompt,
                        "raw_output": None,
                    },
                },
                "answerability": {"conditions": []},
                "result": {"accepted": False, "dry_run": True},
            }
            # Archived discovery prompt-row emission removed from the production trace.
            prompts.append(
                compact_prompt_record({
                    "stage": "generation",
                    "evidence_id": packet.get("evidence_id"),
                    "question_type": question_type,
                    "generation_mode": generation_mode,
                    "attempt": 1,
                    "prompt": gen_prompt,
                    "image_paths": image_paths,
                    "video_paths": video_paths,
                    "generator_decode": decode_config,
                })
            )
            if sequential_dry_run:
                prompts.append(
                    compact_prompt_record(
                        {
                            "stage": "qa_formality_judge",
                            "evidence_id": packet.get("evidence_id"),
                            "qa_id": qa.get("qa_id"),
                            "question_type": question_type,
                            "generation_mode": generation_mode,
                            "attempt": 1,
                            "prompt": qa_formality_prompt,
                            "image_paths": [],
                            "video_paths": [],
                            "media_role": "text_only",
                            "schema_branch": schema_formality_branch(schema_errors),
                            "pass_fail_only": True,
                            "point_scoring": point_scoring_mode,
                        }
                    )
                )
                prompts.append(
                    compact_prompt_record(
                        {
                            "stage": "evidence_groundedness_judge",
                            "evidence_id": packet.get("evidence_id"),
                            "qa_id": qa.get("qa_id"),
                            "question_type": question_type,
                            "generation_mode": generation_mode,
                            "attempt": 1,
                            "prompt": evidence_groundedness_prompt,
                            "image_paths": image_paths,
                            "video_paths": video_paths,
                            "media_role": "same_sampled_media_as_generator",
                            "pass_fail_only": True,
                            "point_scoring": point_scoring_mode,
                        }
                    )
                )
            else:
                prompts.append(
                    compact_prompt_record({
                        "stage": "qa_formality_judge",
                        "evidence_id": packet.get("evidence_id"),
                        "qa_id": qa.get("qa_id"),
                        "question_type": question_type,
                        "generation_mode": generation_mode,
                        "attempt": 1,
                        "prompt": qa_formality_prompt,
                        "image_paths": [],
                        "video_paths": [],
                        "media_role": "text_only",
                        "schema_branch": schema_formality_branch(schema_errors),
                        "generator_rationale_included": False,
                        "pass_fail_only": True,
                        "point_scoring": point_scoring_mode,
                    })
                )
            if six_user_fact_dry_run:
                required_users = [
                    str(user) for user in qa_for_prompt.get("required_users") or []
                ]
                source_media = six_user_source_segment_media(packet, required_users)
                prompts.append(
                    compact_prompt_record(
                        {
                            "stage": "answerability_fact_plan",
                            "evidence_id": packet.get("evidence_id"),
                            "qa_id": qa.get("qa_id"),
                            "attempt": 1,
                            "prompt": build_answerability_fact_plan_prompt(
                                answerability_qa_for_prompt(qa)
                            ),
                            "image_paths": [],
                            "video_paths": [],
                            "media_role": "text_only",
                        }
                    )
                )
                dry_user_audits = []
                for user in required_users:
                    user_video_paths = list(source_media[user]["video_paths"])
                    segment_count = int(source_media[user]["segment_count"])
                    prompts.append(
                        compact_prompt_record(
                            {
                                "stage": "answerability_user_fact_audit",
                                "evidence_id": packet.get("evidence_id"),
                                "qa_id": qa.get("qa_id"),
                                "attempt": 1,
                                "user": user,
                                "segment_count": segment_count,
                                "prompt": build_answerability_user_fact_audit_prompt(
                                    answerability_qa_for_prompt(qa),
                                    user=user,
                                    fact_plan=dry_fact_plan,
                                    segment_count=segment_count,
                                ),
                                "image_paths": [],
                                "video_paths": user_video_paths,
                                "media_role": "ordered_source_segments_one_user",
                            }
                        )
                    )
                    if six_user_map_reduce_dry_run:
                        prompts.append(
                            compact_prompt_record(
                                {
                                    "stage": "evidence_segment_observation",
                                    "evidence_id": packet.get("evidence_id"),
                                    "qa_id": qa.get("qa_id"),
                                    "attempt": 1,
                                    "user": user,
                                    "segment_count": segment_count,
                                    "prompt": build_evidence_segment_observation_prompt(
                                        qa_for_prompt,
                                        user=user,
                                        segment_count=segment_count,
                                    ),
                                    "image_paths": [],
                                    "video_paths": user_video_paths,
                                    "media_role": "ordered_source_segments_one_user",
                                }
                            )
                        )
                    dry_user_audits.append(
                        {
                            "user": user,
                            "reason": "Placeholder showing the reduction contract.",
                            "fact_audits": [
                                {
                                    "fact_id": fact["fact_id"],
                                    "visibility": "NOT_VISIBLE",
                                    "source_users": [],
                                    "segment_references": [],
                                    "visual_description": "Placeholder visibility audit.",
                                }
                                for fact in dry_fact_plan["needed_facts"]
                            ],
                        }
                    )
                if six_user_map_reduce_dry_run:
                    prompts.append(
                        compact_prompt_record(
                            {
                                "stage": "evidence_groundedness_aggregation",
                                "evidence_id": packet.get("evidence_id"),
                                "qa_id": qa.get("qa_id"),
                                "attempt": 1,
                                "prompt": evidence_groundedness_prompt,
                                "image_paths": [],
                                "video_paths": [],
                                "media_role": "text_only_per_user_observation_reduction",
                            }
                        )
                    )
                dry_conditions = build_answerability_conditions(required_users)
                if sequential_dry_run:
                    dry_conditions.append(
                        {
                            "condition_id": (
                                "minimum_required_users::"
                                + "+".join(required_users[:2])
                            ),
                            "condition_type": "minimum_required_users",
                            "users": required_users[:2],
                        }
                    )
                for condition in dry_conditions:
                    included = set(condition["users"])
                    included_audits = [
                        audit for audit in dry_user_audits if audit["user"] in included
                    ]
                    prompts.append(
                        compact_prompt_record(
                            {
                                "stage": "answerability_condition_aggregation",
                                "evidence_id": packet.get("evidence_id"),
                                "qa_id": qa.get("qa_id"),
                                "attempt": 1,
                                "condition_id": condition["condition_id"],
                                "prompt": build_answerability_condition_aggregation_prompt(
                                    answerability_qa_for_prompt(qa),
                                    condition=condition,
                                    fact_plan=dry_fact_plan,
                                    user_audits=included_audits,
                                ),
                                "image_paths": [],
                                "video_paths": [],
                                "media_role": "text_only_user_audit_reduction",
                            }
                        )
                    )
                    dry_trace["answerability"]["conditions"].append(
                        {
                            "condition_id": condition["condition_id"],
                            "condition_type": condition["condition_type"],
                            "users": condition["users"],
                            "media_role": "per_user_ordered_source_segments",
                            "segment_count": sum(
                                int(source_media[str(user)]["segment_count"])
                                for user in condition["users"]
                            ),
                        }
                    )
            else:
                prompts.append(
                    compact_prompt_record({
                        "stage": "evidence_groundedness_judge",
                        "evidence_id": packet.get("evidence_id"),
                        "qa_id": qa.get("qa_id"),
                        "question_type": question_type,
                        "generation_mode": generation_mode,
                        "attempt": 1,
                        "prompt": evidence_groundedness_prompt,
                        "image_paths": full_image_paths,
                        "video_paths": full_video_paths,
                        "media_role": judge_video_source,
                        "generator_rationale_included": judge_include_generator_rationale,
                        "pass_fail_only": True,
                        "point_scoring": point_scoring_mode,
                    })
                )
                for condition in build_answerability_conditions(packet.get("required_users", [])):
                    condition_clips = clips_for_users(packet, condition["users"])
                    cond_images, cond_videos = media_for_clips(
                        condition_clips,
                        backend=judge_media_backend,
                        allow_openai_video_input=allow_openai_video_input,
                        media_role=judge_video_source,
                    )
                    prompts.append(
                        compact_prompt_record({
                            "stage": "answerability",
                            "evidence_id": packet.get("evidence_id"),
                            "question_type": question_type,
                            "generation_mode": generation_mode,
                            "condition_id": condition["condition_id"],
                            "prompt": build_answerability_prompt(qa, condition),
                            "image_paths": cond_images,
                            "video_paths": cond_videos,
                            "media_role": judge_video_source,
                            "condition_media": condition_media_for_clips(
                                condition=condition,
                                clips=condition_clips,
                                image_paths=cond_images,
                                video_paths=cond_videos,
                                media_role=judge_video_source,
                            ),
                        })
                    )
                    dry_trace["answerability"]["conditions"].append(
                        condition_media_for_clips(
                            condition=condition,
                            clips=condition_clips,
                            image_paths=cond_images,
                            video_paths=cond_videos,
                            media_role=judge_video_source,
                        )
                    )
            qa["generation_trace"] = [dry_trace]
            qa["human_audit"] = human_audit_packet(packet)
            qa["generator_decode"] = decode_config
            qa["judge_video_source"] = judge_video_source
            if is_six_user_dry_run:
                qa["six_user_judge_mode"] = six_user_judge_mode
            intermediate_rows.append(
                intermediate_checkpoint_row(
                    evidence_id=packet.get("evidence_id"),
                    qa_id=qa.get("qa_id"),
                    question_type=question_type,
                    generation_mode=generation_mode,
                    status="dry_run",
                    attempts=[dry_trace],
                    qa=qa,
                    generator_decode=decode_config,
                    judge_video_source=judge_video_source,
                )
            )
            counts[question_type] += 1
            accepted.append(qa)
            continue

        packet_rejections = []
        packet_trace = []
        packet_entropy_rows: list[dict[str, Any]] = []
        packet_final_status = "unknown"
        packet_final_attempt: int | None = None
        last_review = None
        for attempt in range(1, max_attempts + 1):
            attempt_trace: dict[str, Any] = {
                "evidence_id": packet.get("evidence_id"),
                "question_type": question_type,
                "generation_mode": generation_mode,
                "attempt": attempt,
                "feedback_in": feedback,
                "previous_generation_in": previous_generation,
                "media": {
                    "image_paths": image_paths,
                    "video_paths": video_paths,
                    "media_role": "generator",
                    "full_image_paths": full_image_paths,
                    "full_video_paths": full_video_paths,
                    "judge_image_paths": full_image_paths,
                    "judge_video_paths": full_video_paths,
                    "judge_media_role": judge_video_source,
                    "prepared_video_uploads": prepared_video_uploads,
                    "human_audit": human_audit_packet(packet),
                },
                "generation": {},
                "generator_decode": decode_config,
                "judge": {},
                "answerability": {},
                "result": {},
            }
            packet_trace.append(attempt_trace)
            # Archived discovery mode previously made a planning call here and then
            # converted selected_relation with build_relation_mcq_prompt. Production
            # now makes the single baseline generation call only.
            gen_prompt = build_video_generation_prompt(
                packet,
                question_type,
                feedback=feedback,
                generation_mode=generation_mode,
                previous_generation=previous_generation,
            )
            attempt_trace["generation"]["prompt"] = gen_prompt
            prompts.append(
                compact_prompt_record({
                    "stage": "generation",
                    "evidence_id": packet.get("evidence_id"),
                    "question_type": question_type,
                    "generation_mode": generation_mode,
                    "attempt": attempt,
                    "prompt": gen_prompt,
                    "image_paths": image_paths,
                    "video_paths": video_paths,
                    "generator_decode": decode_config,
                })
            )
            generation_queued_at = time.time()
            with _generation_gate if _generation_gate is not None else nullcontext():
                stage_start = time.time()
                generation_queue_seconds = round(stage_start - generation_queued_at, 3)
                print(
                    "qa_stage_start "
                    f"stage=generation evidence_id={packet.get('evidence_id')} "
                    f"question_type={question_type} attempt={attempt} "
                    f"images={len(image_paths)} videos={len(video_paths)} "
                    f"queue_seconds={generation_queue_seconds:.1f}",
                    flush=True,
                )
                if generator_decode_mode == "sampling":
                    raw_generation = runner.generate(
                        gen_prompt,
                        image_paths=image_paths,
                        video_paths=video_paths,
                        decoding_mode=generator_decode_mode,
                        temperature=generator_temperature,
                        top_p=generator_top_p,
                        top_k=generator_top_k,
                    )
                else:
                    raw_generation = runner.generate(
                        gen_prompt,
                        image_paths=image_paths,
                        video_paths=video_paths,
                    )
            generation_elapsed_seconds = round(time.time() - stage_start, 3)
            print(
                "qa_stage_done "
                f"stage=generation evidence_id={packet.get('evidence_id')} "
                f"question_type={question_type} attempt={attempt} "
                f"seconds={generation_elapsed_seconds:.1f}",
                flush=True,
            )
            attempt_trace["generation"]["raw_output"] = raw_generation
            attempt_trace["generation"]["elapsed_seconds"] = generation_elapsed_seconds
            if _generation_gate is not None:
                attempt_trace["generation"]["queue_seconds"] = generation_queue_seconds
            previous_generation = str(raw_generation)
            try:
                qa = extract_json_object(raw_generation)
            except Exception as exc:
                feedback = f"Generator output was not valid JSON: {exc}"
                attempt_trace["result"] = {"accepted": False, "reason": feedback}
                packet_rejections.append({"attempt": attempt, "reason": feedback, "raw_output": raw_generation})
                continue

            default_qa_sequence = (
                _packet_index_offset + 1
                if _packet_index_offset is not None
                else len(accepted) + 1
            )
            qa.setdefault(
                "qa_id",
                f"QA_{default_qa_sequence:03d}_{packet.get('evidence_id')}",
            )
            attempt_trace["qa_id"] = qa.get("qa_id")
            attempt_trace["generation"]["parsed_qa"] = {
                "qa_id": qa.get("qa_id"),
                "question": qa.get("question"),
                "options": qa.get("options"),
                "correct": qa.get("correct"),
                "answer": qa.get("answer"),
                "required_users": qa.get("required_users"),
                "question_type": qa.get("question_type"),
                "generator_rationale": qa.get("generator_rationale"),
                "why_two_users_needed": qa.get("why_two_users_needed"),
                "per_user_evidence_claims": qa.get("per_user_evidence_claims"),
                "referred_timestamps": qa.get("referred_timestamps"),
            }
            qa["evidence_id"] = packet.get("evidence_id")
            qa["question_type"] = question_type
            qa["generation_mode"] = generation_mode
            qa["generator_decode"] = decode_config
            qa["required_users"] = packet.get("required_users", qa.get("required_users", []))
            qa["model_id"] = runner.model_id
            qa["review_model_id"] = judge_runner.model_id
            qa["review_model_ids"] = {
                "qa_formality": qa_formality_runner.model_id,
                "evidence_groundedness": judge_runner.model_id,
                "answerability": judge_runner.model_id,
            }
            qa["judge_video_source"] = judge_video_source
            if len(qa.get("required_users") or []) == 6:
                qa["six_user_judge_mode"] = six_user_judge_mode
            qa["source_urls"] = packet.get("source_urls", {})
            qa["video_evidence"] = video_evidence_for_packet(packet)
            qa.setdefault("referred_timestamps", [])
            qa["human_audit"] = human_audit_packet(packet)
            qa["generation_trace"] = packet_trace
            qa["attempt_count"] = attempt
            qa.pop("judge_feedback", None)
            qa.pop("answerability_eval", None)
            complete_generator_metadata(qa, packet=packet, question_type=question_type)
            attempt_trace["generation"]["normalized_qa"] = {
                "qa_id": qa.get("qa_id"),
                "single_user_answerability": qa.get("single_user_answerability"),
                "combined_answerability": qa.get("combined_answerability"),
                "generator_rationale": qa.get("generator_rationale"),
                "why_two_users_needed": qa.get("why_two_users_needed"),
                "per_user_evidence_claims": qa.get("per_user_evidence_claims"),
                "review": qa.get("review"),
            }

            schema_errors = qa_formality_errors(
                qa,
                validate_qa_item(qa),
                participant_names=formality_participant_names(packet, qa),
            )
            if schema_errors:
                attempt_trace["schema_errors"] = schema_errors

            try:
                review_queued_at = time.time()
                with _review_gate if _review_gate is not None else nullcontext():
                    review_queue_seconds = round(time.time() - review_queued_at, 3)
                    print(
                        "qa_packet_review_dispatch "
                        f"evidence_id={packet.get('evidence_id')} attempt={attempt} "
                        f"queue_seconds={review_queue_seconds:.1f}",
                        flush=True,
                    )
                    judge, answerability, judge_trace = run_parallel_review_judges(
                        qa_item=qa,
                        packet=packet,
                        schema_errors=schema_errors,
                        runner=judge_runner,
                        qa_formality_runner=qa_formality_runner,
                        media_backend=judge_media_backend,
                        allow_openai_video_input=allow_openai_video_input,
                        prompt_rows=prompts,
                        full_image_paths=full_image_paths,
                        full_video_paths=full_video_paths,
                        attempt=attempt,
                        judge_media_role=judge_video_source,
                        include_generator_rationale=judge_include_generator_rationale,
                        pass_fail_only=True,
                        quality_quota_counts=None,
                        record_decision_entropy=record_judge_decision_entropy,
                        six_user_judge_mode=six_user_judge_mode,
                        generator_image_paths=image_paths,
                        generator_video_paths=video_paths,
                    )
            except JudgeInfrastructureError as exc:
                reason = str(exc)
                input_token_match = re.search(r"input_tokens=(\d+)", reason)
                attempt_trace["result"] = {
                    "accepted": False,
                    "infrastructure_error": True,
                    "stage": exc.stage,
                    "reason": reason,
                }
                qa["generation_trace"] = packet_trace
                infrastructure_skipped.append(
                    {
                        "status": "judge_infrastructure_skipped",
                        "evidence_id": packet.get("evidence_id"),
                        "qa_id": qa.get("qa_id"),
                        "question_type": question_type,
                        "generation_mode": generation_mode,
                        "six_user_judge_mode": six_user_judge_mode,
                        "judge_video_source": judge_video_source,
                        "attempt": attempt,
                        "stage": exc.stage,
                        "error_type": type(exc.cause).__name__,
                        "reason": reason,
                        "input_tokens": (
                            int(input_token_match.group(1))
                            if input_token_match
                            else None
                        ),
                        "retryable": True,
                        "qa": compact_qa_for_checkpoint(qa),
                    }
                )
                intermediate_rows.append(
                    intermediate_checkpoint_row(
                        evidence_id=packet.get("evidence_id"),
                        qa_id=qa.get("qa_id"),
                        question_type=question_type,
                        generation_mode=generation_mode,
                        status="judge_infrastructure_skipped",
                        attempts=packet_trace,
                        qa=qa,
                        reason=reason,
                        generator_decode=decode_config,
                        judge_video_source=judge_video_source,
                    )
                )
                packet_final_status = "infrastructure_skipped"
                packet_final_attempt = attempt
                print(
                    "qa_packet_skipped "
                    f"evidence_id={packet.get('evidence_id')} stage={exc.stage} "
                    f"reason={reason}",
                    flush=True,
                )
                break
            except OpenRouterRequestError as exc:
                # This is an infrastructure failure, not a negative judgment. Preserve the
                # generated candidate for recovery and stop instead of spending a new Qwen
                # generation attempt on misleading "judge crashed" feedback.
                reason = f"OpenRouter judge infrastructure failure after retries: {exc}"
                attempt_trace["result"] = {
                    "accepted": False,
                    "infrastructure_error": True,
                    "reason": reason,
                }
                qa["generation_trace"] = packet_trace
                intermediate_rows.append(
                    intermediate_checkpoint_row(
                        evidence_id=packet.get("evidence_id"),
                        qa_id=qa.get("qa_id"),
                        question_type=question_type,
                        generation_mode=generation_mode,
                        status="judge_infrastructure_error",
                        attempts=packet_trace,
                        qa=qa,
                        reason=reason,
                        generator_decode=decode_config,
                        judge_video_source=judge_video_source,
                    )
                )
                raise
            attempt_trace["judge"] = judge_trace
            if _review_gate is not None:
                attempt_trace["judge"]["packet_pipeline_queue_seconds"] = (
                    review_queue_seconds
                )
            attempt_trace["answerability"] = answerability
            if isinstance(answerability.get("minimum_required_users"), list):
                qa["minimum_required_users"] = list(
                    answerability["minimum_required_users"]
                )

            judge_failed = judge.get("gate", {}).get("passed") is not True
            if judge_failed:
                feedback = str(
                    judge.get("feedback_to_generator")
                    or judge["gate"].get("reason")
                    or "Judger rejected the question."
                )
                qa["review"] = build_review_from_gates(
                    judge=judge,
                    answerability=answerability,
                    schema_errors=schema_errors,
                    accepted=False,
                    rejection_stage="judger",
                    final_reason=feedback,
                )
                last_review = qa["review"]
                attempt_trace["result"] = {"accepted": False, "reason": feedback}
                packet_rejections.append(
                    {
                        "attempt": attempt,
                        "reason": feedback,
                        "qa": compact_qa_for_checkpoint(qa),
                    }
                )
                if record_judge_decision_entropy:
                    attempt_entropy_rows = production_entropy_rows_for_attempt(
                        judge=judge,
                        evidence_id=packet.get("evidence_id"),
                        qa_id=qa.get("qa_id"),
                        attempt=attempt,
                        attempt_outcome="judge_gate_failed",
                    )
                    attempt_trace["judge_entropy"] = attempt_entropy_rows
                    packet_entropy_rows.extend(attempt_entropy_rows)
                continue

            qa["review"] = build_review_from_gates(
                judge=judge,
                answerability=answerability,
                schema_errors=[],
                accepted=True,
                final_reason="passed all gates",
            )
            strict_errors = validate_qa_item(
                qa,
                strict_review=True,
                require_decision_entropy=record_judge_decision_entropy,
            )
            if strict_errors:
                feedback = "Strict validation errors: " + "; ".join(strict_errors)
                qa["review"] = build_review_from_gates(
                    judge=judge,
                    answerability=answerability,
                    schema_errors=strict_errors,
                    accepted=False,
                    rejection_stage="schema",
                    final_reason=feedback,
                )
                last_review = qa["review"]
                attempt_trace["schema_errors"] = strict_errors
                attempt_trace["result"] = {"accepted": False, "reason": feedback}
                packet_rejections.append(
                    {
                        "attempt": attempt,
                        "reason": feedback,
                        "qa": compact_qa_for_checkpoint(qa),
                    }
                )
                if record_judge_decision_entropy:
                    attempt_entropy_rows = production_entropy_rows_for_attempt(
                        judge=judge,
                        evidence_id=packet.get("evidence_id"),
                        qa_id=qa.get("qa_id"),
                        attempt=attempt,
                        attempt_outcome="post_judge_schema_failed",
                    )
                    attempt_trace["judge_entropy"] = attempt_entropy_rows
                    packet_entropy_rows.extend(attempt_entropy_rows)
                continue

            attempt_trace["result"] = {"accepted": True, "reason": "passed all gates"}
            if record_judge_decision_entropy:
                attempt_entropy_rows = production_entropy_rows_for_attempt(
                    judge=judge,
                    evidence_id=packet.get("evidence_id"),
                    qa_id=qa.get("qa_id"),
                    attempt=attempt,
                    attempt_outcome="accepted",
                )
                attempt_trace["judge_entropy"] = attempt_entropy_rows
                packet_entropy_rows.extend(attempt_entropy_rows)
            qa["generation_trace"] = packet_trace
            last_review = qa["review"]
            accepted.append(qa)
            intermediate_rows.append(
                intermediate_checkpoint_row(
                    evidence_id=packet.get("evidence_id"),
                    qa_id=qa.get("qa_id"),
                    question_type=question_type,
                    generation_mode=generation_mode,
                    status="accepted",
                    attempts=packet_trace,
                    qa=qa,
                    generator_decode=decode_config,
                    judge_video_source=judge_video_source,
                    review=qa.get("review"),
                )
            )
            counts[question_type] += 1
            packet_final_status = "accepted"
            packet_final_attempt = attempt
            break
        else:
            rejected_row = {
                "evidence_id": packet.get("evidence_id"),
                "question_type": question_type,
                "generation_mode": generation_mode,
                "generator_decode": decode_config,
                "judge_video_source": judge_video_source,
                "attempts": packet_rejections,
                "generation_trace": packet_trace,
                "human_audit": human_audit_packet(packet),
            }
            if last_review is not None:
                rejected_row["review"] = last_review
            rejected.append(rejected_row)
            intermediate_rows.append(
                intermediate_checkpoint_row(
                    evidence_id=packet.get("evidence_id"),
                    question_type=question_type,
                    generation_mode=generation_mode,
                    status="rejected",
                    attempts=packet_trace,
                    rejections=packet_rejections,
                    generator_decode=decode_config,
                    judge_video_source=judge_video_source,
                    review=last_review,
                )
            )
            packet_final_status = "rejected"
            packet_final_attempt = max_attempts

        if record_judge_decision_entropy:
            for entropy_row in packet_entropy_rows:
                entropy_row["packet_final_status"] = packet_final_status
                entropy_row["packet_final_attempt"] = packet_final_attempt
                entropy_row["is_final_attempt"] = (
                    entropy_row.get("attempt") == packet_final_attempt
                )
                entropy_row["retry_followed"] = (
                    isinstance(entropy_row.get("attempt"), int)
                    and isinstance(packet_final_attempt, int)
                    and entropy_row["attempt"] < packet_final_attempt
                )
                judge_entropy_rows.append(entropy_row)

    # Prompt and intermediate rows are already flushed one at a time. Rewriting either
    # complete file here used to multiply I/O and required loading legacy giant rows.
    write_jsonl(output_path, accepted)
    if rejected_path and rejected:
        write_jsonl(rejected_path, rejected)
    if record_judge_decision_entropy:
        entropy_summary = summarize_production_judge_entropy(judge_entropy_rows)
        entropy_summary["tokenizer_preflight"] = entropy_tokenizer_preflight
        if judge_entropy_summary_path:
            write_json(judge_entropy_summary_path, entropy_summary)
        if judge_entropy_report_path:
            report_path = Path(judge_entropy_report_path)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                production_entropy_report_markdown(entropy_summary),
                encoding="utf-8",
            )
    return accepted


def add_video_loop_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", default="transformers-local", choices=["transformers-local", "transformers-local-memory-safe", "vllm-local", "openai-compatible-local", "openrouter", "gemini"])
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--generation-mode", default="baseline", choices=GENERATION_MODES)
    parser.add_argument("--generator-decode-mode", default="greedy", choices=GENERATOR_DECODING_MODES)
    parser.add_argument("--generator-temperature", type=float, default=DEFAULT_SAMPLING_TEMPERATURE)
    parser.add_argument("--generator-top-p", type=float, default=DEFAULT_SAMPLING_TOP_P)
    parser.add_argument("--generator-top-k", type=int)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--max-image-pixels", type=int, default=262144)
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--allow-openai-video-input", action="store_true")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--api-key", help="Provider API key; OpenRouter reads OPENROUTER_API_KEY and Gemini reads GEMINI_API_KEY or GOOGLE_API_KEY")
    parser.add_argument("--judge-backend", choices=["transformers-local", "transformers-local-memory-safe", "vllm-local", "openai-compatible-local", "openrouter", "gemini"])
    parser.add_argument("--judge-model-id", help=f"Model for review judges/evaluators; defaults to {DEFAULT_JUDGE_MODEL_ID} when judge backend differs")
    parser.add_argument("--judge-base-url")
    parser.add_argument("--judge-api-key")
    parser.add_argument("--judge-max-new-tokens", type=int)
    parser.add_argument(
        "--judge-reasoning-effort",
        choices=OPENROUTER_REASONING_EFFORTS,
        help="OpenRouter reasoning effort for the visual judges; omitted uses the provider default.",
    )
    parser.add_argument(
        "--qa-formality-use-generator",
        action="store_true",
        help="Run the text-only qa_formality judge on the generator runner instead of the visual judge runner.",
    )
    parser.add_argument(
        "--judge-video-source",
        choices=JUDGE_VIDEO_SOURCES,
        default="full",
        help=(
            "Video source for evidence_groundedness and all answerability conditions. "
            "'full' preserves the production default; 'pruned' is the judge-media ablation."
        ),
    )
    parser.add_argument(
        "--six-user-judge-mode",
        choices=SIX_USER_JUDGE_MODES,
        default=SIX_USER_JUDGE_MODE_TIME_AWARE,
        help=(
            "Six-user review design. time-aware-map-reduce uses separate per-user "
            "evidence and factual-answerability maps; legacy-zero-shot uses direct "
            "six-video checks; sequential-separated-fact-audit runs text-only formality, "
            "then visual grounding, then speaker-first factual answerability as serial gates."
        ),
    )
    parser.add_argument(
        "--infrastructure-skipped-output",
        help="JSONL for retryable judge runtime failures such as catchable CUDA OOMs.",
    )
    parser.add_argument(
        "--skip-evidence-id",
        action="append",
        default=[],
        help="Evidence ID to exclude before any generator or judge call; repeat as needed.",
    )
    parser.add_argument(
        "--packet-limit",
        type=int,
        help="Stop after this many non-excluded evidence packets (useful for a smoke run).",
    )
    parser.add_argument(
        "--judge-hide-generator-rationale",
        dest="judge_include_generator_rationale",
        action="store_false",
        default=False,
        help="Compatibility no-op: generator_rationale is always withheld from review judges.",
    )
    parser.add_argument(
        "--record-judge-decision-entropy",
        action="store_true",
        help=(
            "Keep each detailed model judge as the production gate, then run a "
            "second independent verdict-only call to record pass/fail entropy."
        ),
    )
    parser.add_argument(
        "--judge-entropy-output",
        help="Attempt-level JSONL for integrated production judge entropy.",
    )
    parser.add_argument(
        "--judge-entropy-summary-output",
        help="Aggregate JSON summary grouped by judge, attempt result, and final packet result.",
    )
    parser.add_argument(
        "--judge-entropy-report-output",
        help="Markdown summary of the integrated production entropy run.",
    )
    # Archived scored/quota CLI plumbing. Keeping these commented prevents an old
    # launcher flag from reactivating score prompts in the production pipeline.
    # parser.add_argument("--experimental-scored-judge", ...)
    # parser.add_argument("--judge-quality-quota", ...)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fixed-question-type-schedule", action="store_true")
    parser.add_argument(
        "--question-types",
        default="commonality,difference",
        help="Comma-separated question types to schedule. Use 'neutral' to disable commonality/difference subtype constraints.",
    )
    parser.add_argument("--resume", action="store_true", help="Append to existing JSONL outputs and skip completed evidence IDs")
    parser.add_argument(
        "--max-packets-in-flight",
        type=int,
        default=1,
        help=(
            "Bounded packet pipeline depth. Use 2 only with the async "
            "openai-compatible-local vLLM server."
        ),
    )
    parser.add_argument(
        "--max-review-lanes",
        type=int,
        default=1,
        help=(
            "Maximum packet review stages allowed concurrently. Keep one generation "
            "lane; use 2 with three packets in flight or 3 with four."
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Video-first EgoLife two-user question-answer generation loop")
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompts-output")
    parser.add_argument("--rejected-output")
    parser.add_argument("--intermediate-output")
    parser.add_argument("--target-count", type=int, default=20)
    parser.add_argument("--max-attempts", type=int, default=3)
    add_video_loop_args(parser)
    args = parser.parse_args(argv)
    rows = generate_video_qa_loop(
        evidence_path=args.evidence,
        output_path=args.output,
        prompts_path=args.prompts_output,
        rejected_path=args.rejected_output,
        intermediate_path=args.intermediate_output,
        infrastructure_skipped_path=args.infrastructure_skipped_output,
        judge_entropy_path=args.judge_entropy_output,
        judge_entropy_summary_path=args.judge_entropy_summary_output,
        judge_entropy_report_path=args.judge_entropy_report_output,
        backend=args.backend,
        model_id=args.model_id,
        base_url=args.base_url,
        target_count=args.target_count,
        max_attempts=args.max_attempts,
        max_new_tokens=args.max_new_tokens,
        max_image_pixels=args.max_image_pixels,
        dtype=args.dtype,
        allow_cpu=args.allow_cpu,
        allow_openai_video_input=args.allow_openai_video_input,
        disable_thinking=args.disable_thinking,
        api_key=args.api_key,
        judge_backend=args.judge_backend,
        judge_model_id=args.judge_model_id,
        judge_base_url=args.judge_base_url,
        judge_api_key=args.judge_api_key,
        judge_max_new_tokens=args.judge_max_new_tokens,
        judge_reasoning_effort=args.judge_reasoning_effort,
        qa_formality_use_generator=args.qa_formality_use_generator,
        judge_video_source=args.judge_video_source,
        six_user_judge_mode=args.six_user_judge_mode,
        judge_include_generator_rationale=args.judge_include_generator_rationale,
        record_judge_decision_entropy=args.record_judge_decision_entropy,
        # Archived scored/quota CLI plumbing:
        # judge_pass_fail_only=args.judge_pass_fail_only,
        # judge_quality_quota=args.judge_quality_quota,
        dry_run=args.dry_run,
        generation_mode=args.generation_mode,
        fixed_question_type_schedule=args.fixed_question_type_schedule,
        question_types=parse_question_types(args.question_types),
        resume=args.resume,
        generator_decode_mode=args.generator_decode_mode,
        generator_temperature=args.generator_temperature,
        generator_top_p=args.generator_top_p,
        generator_top_k=args.generator_top_k,
        skip_evidence_ids=args.skip_evidence_id,
        packet_limit=args.packet_limit,
        max_packets_in_flight=args.max_packets_in_flight,
        max_review_lanes=args.max_review_lanes,
    )
    print(f"accepted {len(rows)} video-first question-answer rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
