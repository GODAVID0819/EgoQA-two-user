"""Video-first generation loop for EgoLife two-user multiple-choice construction."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
import math
import re
import statistics
import time
from pathlib import Path
from typing import Any

from .io_utils import append_jsonl, iter_jsonl, write_json, write_jsonl
from .prompts import (
    DEFAULT_QUALITY_QUOTA,
    GENERATION_MODES,
    JUDGE_OUTPUT_SCHEMA_MARKER,
    QA_FORMALITY_SEMANTIC_SUBCHECK_NAMES,
    build_answerability_prompt,
    build_judge_minimal_verdict_probe_prompt,
    build_evidence_groundedness_judge_prompt,
    build_judge_json_repair_prompt,
    build_qa_formality_judge_prompt,
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


QUESTION_TYPES = ("commonality", "difference", "neutral")
DEFAULT_QUESTION_TYPES = ("commonality", "difference")
DEFAULT_JUDGE_MODEL_ID = "Qwen/Qwen3.6-27B"
JUDGE_VIDEO_SOURCES = ("full", "pruned")
BLOCKING_JUDGE_CHECKS = (
    "qa_formality",
    "evidence_groundedness",
    "answerability",
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


def human_audit_packet(packet: dict[str, Any]) -> dict[str, Any]:
    """Compact evidence bundle intended for manual review of one generated question-answer item."""

    required_users = list(packet.get("required_users") or [])
    speaker_user = required_users[0] if required_users else None
    evidence_provider_user = required_users[1] if len(required_users) > 1 else None
    return {
        "evidence_id": packet.get("evidence_id"),
        "required_users": required_users,
        "speaker_user": speaker_user,
        "evidence_provider_user": evidence_provider_user,
        "requirement": packet.get("requirement"),
        "source_urls": packet.get("source_urls", {}),
        "video_evidence": video_evidence_for_packet(packet),
        "review_instructions": [
            "Open each listed local_video or video_url for the required users.",
            "Check the referred_timestamps and per_user_evidence_claims against the visible content.",
            "Verify that required_users[0], the asker, cannot answer from their own video alone.",
            "If required_users[1], the evidence provider, can answer alone, confirm that this is logged in review.answerability.gate.evidence_provider_answerable.",
        ],
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
    qa["question_type"] = question_type
    qa["required_users"] = required_users
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
            single[user] = (
                "may be sufficient because this user is the evidence provider; "
                "answerability is logged by the evaluator"
            )
    qa["single_user_answerability"] = single

    combined = str(qa.get("combined_answerability", "")).strip()
    if "sufficient" not in combined.lower() and "support" not in combined.lower():
        qa["combined_answerability"] = (
            "sufficient because combining the required users' videos provides "
            "the speaker-side anchor event plus the missing visual detail needed "
            "to select exactly one option"
        )

    if not qa.get("generator_rationale"):
        qa["generator_rationale"] = (
            "The question is framed as a natural first-person memory gap anchored "
            "in the asker's experience and answered with another user's visual evidence."
        )
    if not qa.get("why_two_users_needed"):
        qa["why_two_users_needed"] = (
            "At least two required users are needed because the first required user supplies "
            "the speaker-side anchor event while the second required user supplies the missing "
            "visual detail."
        )
    claims = qa.get("per_user_evidence_claims")
    if not isinstance(claims, list) or not claims:
        claims = []
        for user in required_users:
            claims.append(
                {
                    "user": user,
                    "claim": f"{user}'s own video contributes a necessary visual fact listed in the evidence field.",
                }
            )
        qa["per_user_evidence_claims"] = claims

    review = qa.get("review")
    if not isinstance(review, dict):
        review = {}
    review.setdefault(
        "generator_self_check",
        "This draft should be unanswerable from the first required user's video alone; "
        "the second required user's video may contain the answer as evidence-provider context.",
    )
    review.setdefault("speaker_user", asker_user)
    review.setdefault("evidence_provider_user", evidence_provider_user)
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
    return {
        "condition_id": condition.get("condition_id"),
        "condition_type": condition.get("condition_type"),
        "users": condition.get("users", []),
        "media_role": media_role,
        "image_paths": image_paths,
        "video_paths": video_paths,
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


def answerability_gate(qa_item: dict[str, Any], evaluations: list[dict[str, Any]]) -> dict[str, Any]:
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

    required_users = list(qa_item.get("required_users") or [])
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


def judge_gate(judge: dict[str, Any]) -> dict[str, Any]:
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
    for name in BLOCKING_JUDGE_CHECKS:
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
    if blocking_failures:
        return {
            "passed": False,
            "reason": "blocking_failures listed despite structured checks passing: "
            + ", ".join(str(item) for item in blocking_failures),
            "failed_checks": blocking_failures,
        }

    gate = {
        "passed": True,
        "reason": "all structured judger checks passed",
        "failed_checks": [],
    }
    if judge.get("review_passed") is not True:
        gate["model_review_passed"] = judge.get("review_passed")
        gate["warning"] = "ignored inconsistent top-level review_passed because all structured checks passed"
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
            "probe_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
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
                "probe_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
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
            "probe_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
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
        "probe_prompt_sha256": signal.get("probe_prompt_sha256"),
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
        "note": "diagnostic only; production answerability is forced-choice over A-E",
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
            "shared-memory wording, clarify references and options, express a concrete activity "
            "relation, and remove participant names and timestamp citations."
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
            "Revise the question-answer item so the combined required users select the correct answer "
            "and the asker/subset conditions do not."
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
        "judger": judge if isinstance(judge, dict) else {},
        "answerability": answerability if isinstance(answerability, dict) else {},
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
                "probe_prompt_sha256": uncertainty.get("probe_prompt_sha256"),
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


def run_answerability_eval(
    *,
    qa_item: dict[str, Any],
    packet: dict[str, Any],
    runner: Any,
    media_backend: str,
    allow_openai_video_input: bool,
    prompt_rows: list[dict[str, Any]],
    judge_media_role: str = "full",
) -> dict[str, Any]:
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
        prompt_rows.append(
            {
                "stage": "answerability",
                "qa_id": qa_item.get("qa_id"),
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
        )
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
        print(
            "qa_stage_done "
            f"stage=answerability qa_id={qa_item.get('qa_id')} "
            f"condition_id={condition['condition_id']} seconds={time.time() - stage_start:.1f}",
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
                "condition_media": condition_media_for_clips(
                    condition=condition,
                    clips=clips,
                    image_paths=image_paths,
                    video_paths=video_paths,
                    media_role=judge_media_role,
                ),
            }
        )
    gate = answerability_gate(qa_item, evaluations)
    return {"evaluations": evaluations, "gate": gate}


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
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Run qa_formality, evidence_groundedness, and answerability in parallel."""

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
    qa_formality_prompt = build_qa_formality_judge_prompt(
        qa_for_prompt,
        packet,
        schema_errors=schema_errors,
        pass_fail_only=True,
    )
    evidence_groundedness_prompt = build_evidence_groundedness_judge_prompt(
        qa_for_prompt,
        packet,
        pass_fail_only=True,
    )
    qa_formality_entropy_probe_prompt = None
    evidence_groundedness_entropy_probe_prompt = None
    if record_decision_entropy:
        qa_formality_entropy_probe_prompt = build_judge_minimal_verdict_probe_prompt(
            qa_formality_prompt,
            "qa_formality",
        )
        evidence_groundedness_entropy_probe_prompt = (
            build_judge_minimal_verdict_probe_prompt(
                evidence_groundedness_prompt,
                "evidence_groundedness",
            )
        )
    prompt_rows.append(
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
            "model_id": getattr(active_qa_formality_runner, "model_id", None),
            "schema_branch": schema_formality_branch(schema_errors),
            "generator_rationale_included": False,
            "pass_fail_only": True,
            "judge_contract": "legacy_review_passed",
            "decision_entropy_requested": False,
            "authoritative_for_acceptance": True,
            "entropy_probe_affects_acceptance": False,
            "point_scoring": point_scoring_mode,
        }
    )
    prompt_rows.append(
        {
            "stage": "evidence_groundedness_judge",
            "evidence_id": packet.get("evidence_id"),
            "qa_id": qa_item.get("qa_id"),
            "question_type": qa_item.get("question_type"),
            "generation_mode": qa_item.get("generation_mode"),
            "attempt": attempt,
            "prompt": evidence_groundedness_prompt,
            "image_paths": full_image_paths,
            "video_paths": full_video_paths,
            "media_role": judge_media_role,
            "model_id": getattr(runner, "model_id", None),
            "generator_rationale_included": include_generator_rationale,
            "pass_fail_only": True,
            "judge_contract": "legacy_review_passed",
            "decision_entropy_requested": False,
            "authoritative_for_acceptance": True,
            "entropy_probe_affects_acceptance": False,
            "point_scoring": point_scoring_mode,
        }
    )
    if record_decision_entropy:
        prompt_rows.extend(
            [
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
                {
                    "stage": "evidence_groundedness_entropy_probe",
                    "evidence_id": packet.get("evidence_id"),
                    "qa_id": qa_item.get("qa_id"),
                    "question_type": qa_item.get("question_type"),
                    "generation_mode": qa_item.get("generation_mode"),
                    "attempt": attempt,
                    "prompt": evidence_groundedness_entropy_probe_prompt,
                    "image_paths": full_image_paths,
                    "video_paths": full_video_paths,
                    "media_role": judge_media_role,
                    "model_id": getattr(runner, "model_id", None),
                    "pass_fail_only": True,
                    "judge_contract": "minimal_verdict_only",
                    "decision_entropy_requested": True,
                    "authoritative_for_acceptance": False,
                    "entropy_probe_affects_acceptance": False,
                    "point_scoring": point_scoring_mode,
                },
            ]
        )

    answerability_prompt_rows: list[dict[str, Any]] = []
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
        evidence_groundedness_future = executor.submit(
            run_model_judge_branch,
            check_name="evidence_groundedness",
            prompt=evidence_groundedness_prompt,
            runner=runner,
            image_paths=full_image_paths,
            video_paths=full_video_paths,
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
        )

        try:
            qa_formality_judge = qa_formality_future.result()
        except OpenRouterRequestError:
            raise
        except Exception as exc:
            qa_formality_judge = failed_single_judge("qa_formality", f"qa_formality judge crashed: {exc}")
        try:
            evidence_groundedness_judge = evidence_groundedness_future.result()
        except OpenRouterRequestError:
            raise
        except Exception as exc:
            evidence_groundedness_judge = failed_single_judge(
                "evidence_groundedness",
                f"evidence_groundedness judge crashed: {exc}",
            )
        try:
            answerability = answerability_future.result()
        except OpenRouterRequestError:
            raise
        except Exception as exc:
            answerability = {
                "evaluations": [],
                "gate": {
                    "passed": False,
                    "reason": f"answerability judge crashed: {exc}",
                },
            }

    for row in answerability_prompt_rows:
        prompt_rows.append(row)

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
            "generator_rationale_included": False,
            "prompt": qa_formality_prompt,
            "entropy_probe_prompt": qa_formality_entropy_probe_prompt,
            "raw_output": qa_formality_judge.get("raw_output"),
            "parsed": qa_formality_judge,
        },
        "evidence_groundedness": {
            "model_id": getattr(runner, "model_id", None),
            "generator_rationale_included": include_generator_rationale,
            "prompt": evidence_groundedness_prompt,
            "entropy_probe_prompt": evidence_groundedness_entropy_probe_prompt,
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


def generate_video_qa_loop(
    *,
    evidence_path: str | Path,
    output_path: str | Path,
    prompts_path: str | Path | None,
    rejected_path: str | Path | None,
    intermediate_path: str | Path | None = None,
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
    judge_entropy_rows = StreamingJsonlRows(
        judge_entropy_path if record_judge_decision_entropy else None,
        reset=not resume,
    )
    if resume:
        accepted.load_existing()
        rejected.load_existing()
        prompts.load_existing()
        intermediate_rows.load_existing()
        judge_entropy_rows.load_existing()
    quality_quota_counts: dict[str, int] | None = None
    # Archived resume-time quota restoration:
    # quota_source_rows = list(intermediate_rows) or [*accepted, *rejected]
    # quality_quota_counts = quality_quota_counts_from_rows(quota_source_rows)
    processed_evidence_ids = {
        str(row.get("evidence_id"))
        for row in [*accepted, *rejected]
        if row.get("evidence_id")
    }
    targets = target_type_counts(target_count, active_question_types)
    counts = {question_type: 0 for question_type in active_question_types}
    for row in accepted:
        question_type = row.get("question_type")
        if question_type in counts:
            counts[question_type] += 1
    judge_media_backend = judge_backend or backend

    for packet_index, packet in enumerate(iter_jsonl(evidence_path)):
        if fixed_question_type_schedule and packet_index >= target_count:
            break
        if len(accepted) >= target_count:
            break
        evidence_id = str(packet.get("evidence_id") or "")
        if resume and evidence_id in processed_evidence_ids:
            print(f"resume_skip evidence_id={evidence_id}", flush=True)
            continue
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
            evidence_groundedness_prompt = build_evidence_groundedness_judge_prompt(
                qa_for_prompt,
                packet,
                pass_fail_only=True,
            )
            dry_trace = {
                "evidence_id": packet.get("evidence_id"),
                "qa_id": qa.get("qa_id"),
                "question_type": question_type,
                "generation_mode": generation_mode,
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
                    "parallel": True,
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
                        "prompt": evidence_groundedness_prompt,
                        "raw_output": None,
                    },
                },
                "answerability": {"conditions": []},
                "result": {"accepted": False, "dry_run": True},
            }
            # Archived discovery prompt-row emission removed from the production trace.
            prompts.append(
                {
                    "stage": "generation",
                    "evidence_id": packet.get("evidence_id"),
                    "question_type": question_type,
                    "generation_mode": generation_mode,
                    "attempt": 1,
                    "prompt": gen_prompt,
                    "image_paths": image_paths,
                    "video_paths": video_paths,
                    "generator_decode": decode_config,
                }
            )
            prompts.append(
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
                    "generator_rationale_included": False,
                    "pass_fail_only": True,
                    "point_scoring": point_scoring_mode,
                }
            )
            prompts.append(
                {
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
                }
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
                    {
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
                    }
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
            intermediate_rows.append(dry_trace)
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
                {
                    "stage": "generation",
                    "evidence_id": packet.get("evidence_id"),
                    "question_type": question_type,
                    "generation_mode": generation_mode,
                    "attempt": attempt,
                    "prompt": gen_prompt,
                    "image_paths": image_paths,
                    "video_paths": video_paths,
                    "generator_decode": decode_config,
                }
            )
            stage_start = time.time()
            print(
                "qa_stage_start "
                f"stage=generation evidence_id={packet.get('evidence_id')} "
                f"question_type={question_type} attempt={attempt} "
                f"images={len(image_paths)} videos={len(video_paths)}",
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
                raw_generation = runner.generate(gen_prompt, image_paths=image_paths, video_paths=video_paths)
            print(
                "qa_stage_done "
                f"stage=generation evidence_id={packet.get('evidence_id')} "
                f"question_type={question_type} attempt={attempt} "
                f"seconds={time.time() - stage_start:.1f}",
                flush=True,
            )
            attempt_trace["generation"]["raw_output"] = raw_generation
            previous_generation = str(raw_generation)
            try:
                qa = extract_json_object(raw_generation)
            except Exception as exc:
                feedback = f"Generator output was not valid JSON: {exc}"
                attempt_trace["result"] = {"accepted": False, "reason": feedback}
                packet_rejections.append({"attempt": attempt, "reason": feedback, "raw_output": raw_generation})
                continue

            qa.setdefault("qa_id", f"QA_{len(accepted) + 1:03d}_{packet.get('evidence_id')}")
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
                )
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
                    {
                        "evidence_id": packet.get("evidence_id"),
                        "qa_id": qa.get("qa_id"),
                        "question_type": question_type,
                        "generation_mode": generation_mode,
                        "status": "judge_infrastructure_error",
                        "reason": reason,
                        "qa": qa,
                    }
                )
                raise
            attempt_trace["judge"] = judge_trace
            attempt_trace["answerability"] = answerability

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
                packet_rejections.append({"attempt": attempt, "reason": feedback, "qa": qa})
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
                packet_rejections.append({"attempt": attempt, "reason": feedback, "qa": qa})
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
                {
                    "evidence_id": packet.get("evidence_id"),
                    "qa_id": qa.get("qa_id"),
                    "question_type": question_type,
                    "generation_mode": generation_mode,
                    "generator_decode": decode_config,
                    "status": "accepted",
                    "attempts": packet_trace,
                }
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
            intermediate_rows.append({**rejected_row, "status": "rejected"})
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

    if prompts_path:
        write_jsonl(prompts_path, prompts)
    if intermediate_path:
        write_jsonl(intermediate_path, intermediate_rows)
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
    parser.add_argument("--backend", default="transformers-local", choices=["transformers-local", "transformers-local-memory-safe", "openai-compatible-local", "openrouter", "gemini"])
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
    parser.add_argument("--judge-backend", choices=["transformers-local", "transformers-local-memory-safe", "openai-compatible-local", "openrouter", "gemini"])
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
    )
    print(f"accepted {len(rows)} video-first question-answer rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
