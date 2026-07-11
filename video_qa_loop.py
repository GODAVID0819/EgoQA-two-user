"""Video-first generation loop for EgoLife two-user multiple-choice construction."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import itertools
import time
from pathlib import Path
from typing import Any

from .io_utils import append_jsonl, iter_jsonl, write_jsonl
from .prompts import (
    GENERATION_MODES,
    build_answerability_prompt,
    build_evidence_groundedness_judge_prompt,
    build_qa_formality_judge_prompt,
    build_relation_discovery_prompt,
    build_relation_mcq_prompt,
    build_video_generation_prompt,
)
from .qwen3vl_runner import (
    DEFAULT_MODEL_ID,
    DEFAULT_SAMPLING_TEMPERATURE,
    DEFAULT_SAMPLING_TOP_P,
    GENERATOR_DECODING_MODES,
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
BLOCKING_JUDGE_CHECKS = (
    "qa_formality",
    "evidence_groundedness",
    "answerability",
)
QUALITY_SCORED_JUDGE_CHECKS = {
    "qa_formality",
    "evidence_groundedness",
}
QUALITY_FLAGS = {
    1: "1_weak_or_reject",
    2: "2_acceptable",
    3: "3_strong",
}
TEMPORAL_REASONING_MODE = "temporal_reasoning"


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
    if clips_require_frame_inputs(clips):
        return images, []
    if backend == "openai-compatible-local" and not allow_openai_video_input:
        return images, []
    return images if not videos else [], videos


def prepare_runner_video_uploads(
    *,
    runner: Any,
    evidence_id: Any,
    generator_video_paths: list[str],
    full_video_paths: list[str],
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
        f"full_videos={len(full_video_paths)} "
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
        "unique_video_paths": all_video_paths,
        "prepared_video_count": len(prepared or []),
        "purpose": (
            "pre-upload generator pruned videos and full original videos so the generator sees pruned media "
            "and judges/answerability see complete originals"
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
    neutral/baseline/discovery prompts or media traces.
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
                "video_url": clip.get("video_url"),
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
                "gaze_summary": clip.get("gaze_summary"),
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


def qa_for_judger_prompt(qa: dict[str, Any]) -> dict[str, Any]:
    """Return only the generated question-answer fields the judger needs to evaluate."""

    wanted = [
        "qa_id",
        "evidence_id",
        "question_type",
        "question",
        "options",
        "correct",
        "answer",
        "required_users",
        "evidence",
        "single_user_answerability",
        "combined_answerability",
        "generator_rationale",
        "why_two_users_needed",
        "per_user_evidence_claims",
        "referred_timestamps",
        "review",
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
    text = str(value or "").strip()
    if text.lower() in {"insufficient", "not enough", "unknown", "cannot answer", "can't answer"}:
        return None, True
    try:
        return normalize_correct(text), False
    except ValueError:
        return None, False


def answerability_gate(qa_item: dict[str, Any], evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        correct = normalize_correct(qa_item.get("correct"))
    except ValueError as exc:
        return {"passed": False, "reason": str(exc)}

    combined = [row for row in evaluations if row.get("condition_type") == "combined_all_users"]
    if not combined:
        return {"passed": False, "reason": "missing combined_all_users evaluation"}

    combined_choice, combined_insufficient = parsed_choice(combined[-1].get("choice"))
    if combined_insufficient or combined_choice != correct:
        return {
            "passed": False,
            "reason": f"combined_all_users did not select correct answer {correct}",
        }

    required_users = list(qa_item.get("required_users") or [])
    asker_user = required_users[0] if required_users else None
    evidence_provider_user = required_users[1] if len(required_users) > 1 else None
    blocking_leaks = []
    evidence_provider_answerable = []
    for row in evaluations:
        if row.get("condition_type") == "combined_all_users":
            continue
        choice, insufficient = parsed_choice(row.get("choice"))
        if not insufficient and choice == correct:
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
        "reason": "combined videos answer correctly and all single/subset conditions are insufficient or incorrect",
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


def normalize_quality_fields(check: dict[str, Any], check_name: str) -> dict[str, Any]:
    """Ensure non-answerability judge checks always carry comparable quality metadata."""

    if check_name not in QUALITY_SCORED_JUDGE_CHECKS:
        return check
    raw_score = check.get("quality_score")
    try:
        score = int(raw_score)
    except (TypeError, ValueError):
        score = 2 if str(check.get("status", "")).upper() == "PASS" else 1
    score = min(3, max(1, score))
    check["quality_score"] = score
    check["quality_flag"] = QUALITY_FLAGS[score]
    if not str(check.get("quality_reason") or "").strip():
        if score == 3:
            check["quality_reason"] = f"{check_name} judged the question-answer item as strong on its rubric"
        elif score == 2:
            check["quality_reason"] = f"{check_name} judged the question-answer item as acceptable but not especially strong"
        else:
            check["quality_reason"] = f"{check_name} found a weak or rejecting issue"
    return check


def failed_single_judge(check_name: str, reason: str, *, raw_output: str | None = None) -> dict[str, Any]:
    failed_check = normalize_quality_fields(
        {
            "status": "FAIL",
            "reason": reason,
            "fix": f"Repair the question-answer item so the {check_name} judge can pass.",
        },
        check_name,
    )
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
) -> dict[str, Any]:
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
    try:
        judge = extract_json_object(raw)
    except Exception as exc:
        judge = failed_single_judge(check_name, f"{check_name} judge output was not valid JSON: {exc}")
    judge["raw_output"] = raw
    return judge


def check_from_single_judge(judge: dict[str, Any], check_name: str) -> dict[str, Any]:
    checks = judge.get("checks")
    if isinstance(checks, dict) and isinstance(checks.get(check_name), dict):
        return normalize_quality_fields(dict(checks[check_name]), check_name)
    return normalize_quality_fields(
        {
            "status": "FAIL",
            "reason": f"{check_name} judge did not return checks.{check_name}",
            "fix": f"Return a valid {check_name} check object.",
        },
        check_name,
    )


def merge_parallel_judges(
    *,
    qa_formality_judge: dict[str, Any],
    evidence_groundedness_judge: dict[str, Any],
    answerability: dict[str, Any],
    schema_errors: list[str],
) -> dict[str, Any]:
    schema_branch = schema_formality_branch(schema_errors)
    qa_formality_check = check_from_single_judge(qa_formality_judge, "qa_formality")
    model_qa_formality_check = dict(qa_formality_check)
    semantic_subchecks = qa_formality_check.get("semantic_subchecks")
    if isinstance(semantic_subchecks, dict):
        other_activity = semantic_subchecks.get("other_person_activity_query")
        if isinstance(other_activity, dict) and str(other_activity.get("status", "")).upper() == "FAIL":
            qa_formality_check["status"] = "FAIL"
            qa_formality_check["quality_score"] = 1
            reason = str(other_activity.get("reason") or "")
            qa_formality_check["reason"] = (
                str(qa_formality_check.get("reason") or "")
                + ("; " if qa_formality_check.get("reason") else "")
                + "semantic_subchecks.other_person_activity_query failed"
                + (f": {reason}" if reason else "")
            )
            qa_formality_check["fix"] = (
                "Replace the shallow concurrent-activity question with a concrete speaker-side "
                "information need whose answer depends on the evidence provider's missing detail."
            )
        direct_name = semantic_subchecks.get("direct_name_leakage")
        if isinstance(direct_name, dict) and str(direct_name.get("status", "")).upper() == "FAIL":
            qa_formality_check["status"] = "FAIL"
            qa_formality_check["quality_score"] = 1
            reason = str(direct_name.get("reason") or "")
            qa_formality_check["reason"] = (
                str(qa_formality_check.get("reason") or "")
                + ("; " if qa_formality_check.get("reason") else "")
                + "semantic_subchecks.direct_name_leakage failed"
                + (f": {reason}" if reason else "")
            )
            qa_formality_check["fix"] = (
                "Remove direct person names from the question and preserve the first-person "
                "or shared-memory perspective with pronouns, roles, or descriptions."
            )
    if schema_branch["status"] != "PASS":
        qa_formality_check["status"] = "FAIL"
        qa_formality_check["quality_score"] = 1
        qa_formality_check["reason"] = (
            schema_branch["reason"]
            + "; model qa_formality branch: "
            + str(model_qa_formality_check.get("reason", ""))
        )
        qa_formality_check["fix"] = (
            "Repair the generated JSON shape, multiple-choice options, correct letter, "
            "answer text, required users, and required question-answer metadata."
        )
    qa_formality_check["schema_branch"] = schema_branch
    qa_formality_check["model_branch"] = model_qa_formality_check
    qa_formality_check = normalize_quality_fields(qa_formality_check, "qa_formality")

    evidence_check = check_from_single_judge(evidence_groundedness_judge, "evidence_groundedness")
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
    users = packet.get("required_users", [])[:2]
    speaker = users[0] if users else "User A"
    other = users[1] if len(users) > 1 else "User B"
    return {
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
    backend: str,
    allow_openai_video_input: bool,
    prompt_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    evaluations = []
    for condition in build_answerability_conditions(qa_item.get("required_users", [])):
        clips = clips_for_users(packet, condition["users"])
        image_paths, video_paths = media_for_clips(
            clips,
            backend=backend,
            allow_openai_video_input=allow_openai_video_input,
            media_role="full",
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
                "condition_media": condition_media_for_clips(
                    condition=condition,
                    clips=clips,
                    image_paths=image_paths,
                    video_paths=video_paths,
                    media_role="full",
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
                "choice": "insufficient",
                "answer_text": "",
                "evidence_used": "",
                "insufficient_reason": f"parse_failed: {exc}",
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
                    media_role="full",
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
    backend: str,
    allow_openai_video_input: bool,
    prompt_rows: list[dict[str, Any]],
    full_image_paths: list[str],
    full_video_paths: list[str],
    attempt: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Run qa_formality, evidence_groundedness, and answerability in parallel."""

    qa_for_prompt = qa_for_judger_prompt(qa_item)
    qa_formality_prompt = build_qa_formality_judge_prompt(
        qa_for_prompt,
        packet,
        schema_errors=schema_errors,
    )
    evidence_groundedness_prompt = build_evidence_groundedness_judge_prompt(qa_for_prompt, packet)
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
            "schema_branch": schema_formality_branch(schema_errors),
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
            "media_role": "full",
        }
    )

    answerability_prompt_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        qa_formality_future = executor.submit(
            run_model_judge_branch,
            check_name="qa_formality",
            prompt=qa_formality_prompt,
            runner=runner,
            image_paths=[],
            video_paths=[],
            evidence_id=packet.get("evidence_id"),
            qa_id=qa_item.get("qa_id"),
            attempt=attempt,
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
        )
        answerability_future = executor.submit(
            run_answerability_eval,
            qa_item=qa_item,
            packet=packet,
            runner=runner,
            backend=backend,
            allow_openai_video_input=allow_openai_video_input,
            prompt_rows=answerability_prompt_rows,
        )

        try:
            qa_formality_judge = qa_formality_future.result()
        except Exception as exc:
            qa_formality_judge = failed_single_judge("qa_formality", f"qa_formality judge crashed: {exc}")
        try:
            evidence_groundedness_judge = evidence_groundedness_future.result()
        except Exception as exc:
            evidence_groundedness_judge = failed_single_judge(
                "evidence_groundedness",
                f"evidence_groundedness judge crashed: {exc}",
            )
        try:
            answerability = answerability_future.result()
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
    )
    trace = {
        "parallel": True,
        "schema_branch": schema_formality_branch(schema_errors),
        "qa_formality": {
            "prompt": qa_formality_prompt,
            "raw_output": qa_formality_judge.get("raw_output"),
            "parsed": qa_formality_judge,
        },
        "evidence_groundedness": {
            "prompt": evidence_groundedness_prompt,
            "raw_output": evidence_groundedness_judge.get("raw_output"),
            "parsed": evidence_groundedness_judge,
        },
        "answerability": answerability,
        "merged": judge,
    }
    return judge, answerability, trace


def generate_video_qa_loop(
    *,
    evidence_path: str | Path,
    output_path: str | Path,
    prompts_path: str | Path | None,
    rejected_path: str | Path | None,
    intermediate_path: str | Path | None = None,
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
    if generation_mode not in GENERATION_MODES:
        raise ValueError(f"unknown generation_mode: {generation_mode}")
    if generator_decode_mode not in GENERATOR_DECODING_MODES:
        raise ValueError(f"unknown generator_decode_mode: {generator_decode_mode}")
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
    runner = make_runner(
        "dry-run" if dry_run else backend,
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
    prompts = StreamingJsonlRows(prompts_path, reset=not resume)
    intermediate_rows = StreamingJsonlRows(intermediate_path, reset=not resume)
    accepted = StreamingJsonlRows(output_path, reset=not resume)
    rejected = StreamingJsonlRows(rejected_path, reset=not resume)
    if resume:
        accepted.load_existing()
        rejected.load_existing()
        prompts.load_existing()
        intermediate_rows.load_existing()
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
            backend=backend,
            allow_openai_video_input=allow_openai_video_input,
            media_role="full",
        )
        prepared_video_uploads = prepare_runner_video_uploads(
            runner=runner,
            evidence_id=packet.get("evidence_id"),
            generator_video_paths=video_paths,
            full_video_paths=full_video_paths,
        )
        feedback = None
        if dry_run:
            qa = dry_run_qa(packet, question_type, generation_mode=generation_mode)
            discovery_prompt = None
            discovered_relation = None
            if generation_mode == "discovery":
                discovery_prompt = build_relation_discovery_prompt(packet, question_type)
                discovered_relation = dry_run_discovered_relation(packet, question_type)
                gen_prompt = build_relation_mcq_prompt(packet, question_type, discovered_relation)
            else:
                gen_prompt = build_video_generation_prompt(
                    packet,
                    question_type,
                    generation_mode=generation_mode,
                )
            schema_errors = validate_qa_item(qa)
            qa_formality_prompt = build_qa_formality_judge_prompt(
                qa_for_judger_prompt(qa),
                packet,
                schema_errors=schema_errors,
            )
            evidence_groundedness_prompt = build_evidence_groundedness_judge_prompt(
                qa_for_judger_prompt(qa),
                packet,
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
                    "prepared_video_uploads": prepared_video_uploads,
                    "human_audit": human_audit_packet(packet),
                },
                "discovery": (
                    {
                        "prompt": discovery_prompt,
                        "raw_output": None,
                        "parsed": {"selected_relation": discovered_relation},
                    }
                    if generation_mode == "discovery"
                    else {}
                ),
                "generation": {"prompt": gen_prompt, "raw_output": None},
                "generator_decode": decode_config,
                "judge": {
                    "parallel": True,
                    "schema_branch": schema_formality_branch(schema_errors),
                    "qa_formality": {"prompt": qa_formality_prompt, "raw_output": None},
                    "evidence_groundedness": {"prompt": evidence_groundedness_prompt, "raw_output": None},
                },
                "answerability": {"conditions": []},
                "result": {"accepted": False, "dry_run": True},
            }
            if generation_mode == "discovery":
                prompts.append(
                    {
                        "stage": "discovery",
                        "evidence_id": packet.get("evidence_id"),
                        "question_type": question_type,
                        "generation_mode": generation_mode,
                        "attempt": 1,
                        "prompt": discovery_prompt,
                        "image_paths": image_paths,
                        "video_paths": video_paths,
                    }
                )
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
                    "media_role": "full",
                }
            )
            for condition in build_answerability_conditions(packet.get("required_users", [])):
                condition_clips = clips_for_users(packet, condition["users"])
                cond_images, cond_videos = media_for_clips(
                    condition_clips,
                    backend=backend,
                    allow_openai_video_input=allow_openai_video_input,
                    media_role="full",
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
                        "media_role": "full",
                        "condition_media": condition_media_for_clips(
                            condition=condition,
                            clips=condition_clips,
                            image_paths=cond_images,
                            video_paths=cond_videos,
                            media_role="full",
                        ),
                    }
                )
                dry_trace["answerability"]["conditions"].append(
                    condition_media_for_clips(
                        condition=condition,
                        clips=condition_clips,
                        image_paths=cond_images,
                        video_paths=cond_videos,
                        media_role="full",
                    )
                )
            qa["generation_trace"] = [dry_trace]
            qa["human_audit"] = human_audit_packet(packet)
            qa["generator_decode"] = decode_config
            intermediate_rows.append(dry_trace)
            counts[question_type] += 1
            accepted.append(qa)
            continue

        packet_rejections = []
        packet_trace = []
        last_review = None
        for attempt in range(1, max_attempts + 1):
            attempt_trace: dict[str, Any] = {
                "evidence_id": packet.get("evidence_id"),
                "question_type": question_type,
                "generation_mode": generation_mode,
                "attempt": attempt,
                "feedback_in": feedback,
                "media": {
                    "image_paths": image_paths,
                    "video_paths": video_paths,
                    "media_role": "generator",
                    "full_image_paths": full_image_paths,
                    "full_video_paths": full_video_paths,
                    "prepared_video_uploads": prepared_video_uploads,
                    "human_audit": human_audit_packet(packet),
                },
                "discovery": {},
                "generation": {},
                "generator_decode": decode_config,
                "judge": {},
                "answerability": {},
                "result": {},
            }
            packet_trace.append(attempt_trace)
            if generation_mode == "discovery":
                discovery_prompt = build_relation_discovery_prompt(packet, question_type, feedback=feedback)
                attempt_trace["discovery"]["prompt"] = discovery_prompt
                prompts.append(
                    {
                        "stage": "discovery",
                        "evidence_id": packet.get("evidence_id"),
                        "question_type": question_type,
                        "generation_mode": generation_mode,
                        "attempt": attempt,
                        "prompt": discovery_prompt,
                        "image_paths": image_paths,
                        "video_paths": video_paths,
                    }
                )
                stage_start = time.time()
                print(
                    "qa_stage_start "
                    f"stage=discovery evidence_id={packet.get('evidence_id')} "
                    f"question_type={question_type} attempt={attempt} "
                    f"images={len(image_paths)} videos={len(video_paths)}",
                    flush=True,
                )
                raw_discovery = runner.generate(discovery_prompt, image_paths=image_paths, video_paths=video_paths)
                print(
                    "qa_stage_done "
                    f"stage=discovery evidence_id={packet.get('evidence_id')} "
                    f"question_type={question_type} attempt={attempt} "
                    f"seconds={time.time() - stage_start:.1f}",
                    flush=True,
                )
                attempt_trace["discovery"]["raw_output"] = raw_discovery
                try:
                    discovery = extract_json_object(raw_discovery)
                except Exception as exc:
                    feedback = f"Discovery output was not valid JSON: {exc}"
                    attempt_trace["result"] = {"accepted": False, "reason": feedback}
                    packet_rejections.append({"attempt": attempt, "reason": feedback, "raw_output": raw_discovery})
                    continue
                discovered_relation = discovery.get("selected_relation") if isinstance(discovery, dict) else None
                if not isinstance(discovered_relation, dict) or not discovered_relation:
                    discovered_relation = discovery
                attempt_trace["discovery"]["parsed"] = discovery
                gen_prompt = build_relation_mcq_prompt(
                    packet,
                    question_type,
                    discovered_relation,
                    feedback=feedback,
                )
            else:
                gen_prompt = build_video_generation_prompt(
                    packet,
                    question_type,
                    feedback=feedback,
                    generation_mode=generation_mode,
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

            schema_errors = validate_qa_item(qa)
            if schema_errors:
                attempt_trace["schema_errors"] = schema_errors

            judge, answerability, judge_trace = run_parallel_review_judges(
                qa_item=qa,
                packet=packet,
                schema_errors=schema_errors,
                runner=runner,
                backend=backend,
                allow_openai_video_input=allow_openai_video_input,
                prompt_rows=prompts,
                full_image_paths=full_image_paths,
                full_video_paths=full_video_paths,
                attempt=attempt,
            )
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
                continue

            qa["review"] = build_review_from_gates(
                judge=judge,
                answerability=answerability,
                schema_errors=[],
                accepted=True,
                final_reason="passed all gates",
            )
            strict_errors = validate_qa_item(qa, strict_review=True)
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
                continue

            attempt_trace["result"] = {"accepted": True, "reason": "passed all gates"}
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
            break
        else:
            rejected_row = {
                "evidence_id": packet.get("evidence_id"),
                "question_type": question_type,
                "generation_mode": generation_mode,
                "generator_decode": decode_config,
                "attempts": packet_rejections,
                "generation_trace": packet_trace,
                "human_audit": human_audit_packet(packet),
            }
            if last_review is not None:
                rejected_row["review"] = last_review
            rejected.append(rejected_row)
            intermediate_rows.append({**rejected_row, "status": "rejected"})

    if prompts_path:
        write_jsonl(prompts_path, prompts)
    if intermediate_path:
        write_jsonl(intermediate_path, intermediate_rows)
    write_jsonl(output_path, accepted)
    if rejected_path and rejected:
        write_jsonl(rejected_path, rejected)
    return accepted


def add_video_loop_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", default="transformers-local", choices=["transformers-local", "openai-compatible-local", "gemini"])
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
    parser.add_argument("--api-key", help="Provider API key; Gemini also reads GEMINI_API_KEY or GOOGLE_API_KEY")
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
