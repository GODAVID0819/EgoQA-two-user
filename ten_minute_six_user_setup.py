"""Prepare six synchronized 10-minute users for time-aware QA generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
import types
import unicodedata
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

# The checked-in multi-user tree is also used directly from its filesystem path.
# Give direct script execution the same package context as ``python -m`` so the
# relative imports below remain the single source of truth.
if __package__ in {None, ""}:
    package_root = Path(__file__).resolve().parent
    package = types.ModuleType("egolife_two_user_qa")
    package.__path__ = [str(package_root)]
    sys.modules.setdefault("egolife_two_user_qa", package)
    __package__ = "egolife_two_user_qa"

from .clip_gap_demo import DEFAULT_CLIP_MODEL, ImageEncoder, TransformersClipEncoder
from .evidence import LONG_CONTEXT_EVIDENCE_DURATION_SECONDS, iter_evidence_packets
from .group_relative_clip_sampling import (
    SIX_USER_DEFAULT_CLUSTERS_PER_WINDOW,
    SIX_USER_DEFAULT_CLUSTER_WINDOW_SECONDS,
    SIX_USER_DEFAULT_GENERATOR_FRAME_BUDGET,
    SIX_USER_DEFAULT_MAX_CLUSTER_MEMBER_GAP_SECONDS,
    SIX_USER_DEFAULT_MAX_PAIR_TIME_DIFFERENCE_SECONDS,
    SIX_USER_DEFAULT_SAMPLE_INTERVAL_SECONDS,
    SIX_USER_GENERATOR_MEDIA_MODE,
    SIX_USER_PROVIDER_FRAME_MODE,
    SIX_USER_PROVIDER_MEDIA_ROLE,
    SIX_USER_SPEAKER_FRAME_MODE,
    SIX_USER_SPEAKER_MEDIA_ROLE,
    SIX_USER_TIME_AWARE_TEMPORAL_POLICY,
    analyze_group_relative_similarity,
    build_candidate_packet,
)
from .io_utils import write_json, write_jsonl
from .prompts import (
    LONG_HORIZON_FORMALITY_GUIDANCE,
    LONG_HORIZON_GROUNDEDNESS_GUIDANCE,
    OPTIONAL_LONG_HORIZON_GUIDANCE,
    build_answerability_prompt,
    build_answerability_condition_aggregation_prompt,
    build_answerability_fact_plan_prompt,
    build_answerability_user_fact_audit_prompt,
    build_evidence_groundedness_judge_prompt,
    build_evidence_observation_aggregation_prompt,
    build_evidence_segment_observation_prompt,
    build_qa_formality_judge_prompt,
    build_video_generation_prompt,
)
from .qwen3vl_runner import (
    MEMORY_SAFE_DEFAULT_IMAGE_CONTEXT_TARGET_FRACTION,
    MEMORY_SAFE_DEFAULT_IMAGE_ITEM_TOKEN_OVERHEAD,
    MEMORY_SAFE_DEFAULT_IMAGE_TEXT_TOKEN_RESERVE,
    QWEN_VISION_TOKEN_PIXEL_AREA,
    memory_safe_image_max_pixels,
)


SIX_USER_COUNT = 6
DEFAULT_PRUNING_CLUSTERS_PER_WINDOW = SIX_USER_DEFAULT_CLUSTERS_PER_WINDOW
DEFAULT_MIN_PRUNED_VIDEO_PERCENT = 40.0
DEFAULT_CLIP_BATCH_SIZE = 32
DEFAULT_MODEL_VIDEO_FPS = 0.75
DEFAULT_MAX_INPUT_TOKENS = 262_144
DEFAULT_TEXT_TOKEN_RESERVE = MEMORY_SAFE_DEFAULT_IMAGE_TEXT_TOKEN_RESERVE
ESTIMATED_VISUAL_TOKENS_PER_FRAME = 64
DEFAULT_GENERATOR_MAX_IMAGE_PIXELS = 65_536
LONG_HORIZON_SELECTION_TITLES = (
    "1. Object trajectory",
    "2. Cross-user before/after state",
    "3. Same-user revisit with a cross-user intervention",
    "4. Last-seen or most-recent interaction",
    "5. Cross-user temporal ordering",
)
PROCESSED_EVIDENCE_FILENAMES = (
    "six_user_10min_time_aware.jsonl",
    "six_user_10min_source_windows.jsonl",
    "qa_mcq.jsonl",
    "qa_mcq.rejected.jsonl",
    "qa_mcq.infrastructure_skipped.jsonl",
)
_CONSENSUS_EVIDENCE_ID = re.compile(
    r"^EGOLIFE6U_CONSENSUS_(?P<day>.+)_(?P<time_token>\d{8})_S\d+$"
)
_SOURCE_WINDOW_EVIDENCE_ID = re.compile(
    r"^EGOLIFE2U_(?P<day>.+)_(?P<time_token>\d{8})_600S(?:_|$)"
)


class BatchedImageEncoder:
    """Bound CLIP working memory while retaining the configured analysis coverage."""

    def __init__(self, encoder: ImageEncoder, *, batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("CLIP batch size must be positive")
        self.encoder = encoder
        self.batch_size = int(batch_size)
        self.model_id = encoder.model_id

    def encode(self, image_paths: list[str]) -> list[list[float]]:
        rows: list[list[float]] = []
        for start in range(0, len(image_paths), self.batch_size):
            rows.extend(self.encoder.encode(image_paths[start : start + self.batch_size]))
        return rows


def normalize_question(question: str) -> str:
    """Return the stable form used to prevent exact question reuse."""

    return " ".join(unicodedata.normalize("NFKC", question).casefold().split())


def question_fingerprint(question: str) -> str:
    normalized = normalize_question(question)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _iter_history_mappings(row: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """Visit only QA wrappers, never bulky frame, clip, or model-trace payloads."""

    yield row
    for field in (
        "qa",
        "qa_item",
        "candidate",
        "human_audit",
        "previous_generation",
        "final_qa",
    ):
        nested = row.get(field)
        if isinstance(nested, dict):
            yield from _iter_history_mappings(nested)
        elif isinstance(nested, list):
            for item in nested:
                if isinstance(item, dict):
                    yield from _iter_history_mappings(item)


def evidence_window_key(row: dict[str, Any]) -> tuple[str, str] | None:
    """Resolve a packet, candidate, or QA row to its synchronized source window."""

    day = str(row.get("day") or "").strip()
    time_token = str(row.get("time_token") or "").strip()
    if day and time_token:
        return day, time_token

    for field in ("source_window_evidence_id", "evidence_id"):
        evidence_id = str(row.get(field) or "").strip()
        if not evidence_id:
            continue
        for pattern in (_CONSENSUS_EVIDENCE_ID, _SOURCE_WINDOW_EVIDENCE_ID):
            match = pattern.match(evidence_id)
            if match:
                return match.group("day"), match.group("time_token")
    return None


def _iter_processed_files(sources: Sequence[str | Path]) -> list[Path]:
    files: dict[str, Path] = {}
    for source_value in sources:
        source = Path(source_value).expanduser()
        if not source.exists():
            raise FileNotFoundError(f"processed-evidence source does not exist: {source}")
        candidates: Iterable[Path]
        if source.is_file():
            candidates = (source,)
        else:
            candidates = (
                candidate
                for filename in PROCESSED_EVIDENCE_FILENAMES
                for candidate in source.rglob(filename)
            )
        for candidate in candidates:
            if candidate.is_file():
                resolved = candidate.resolve()
                files[str(resolved)] = resolved
    return [files[key] for key in sorted(files)]


def _read_processed_rows(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else [payload]
        for row in rows:
            if isinstance(row, dict):
                yield row
        return
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid processed-evidence JSONL at {path}:{line_number}: {exc}"
            ) from exc
        if isinstance(row, dict):
            yield row


def collect_processed_evidence(
    sources: Sequence[str | Path],
) -> dict[str, Any]:
    """Collect prior source windows and question hashes before media preparation."""

    files = _iter_processed_files(sources)
    window_keys: set[tuple[str, str]] = set()
    question_hashes: set[str] = set()
    row_count = 0
    for path in files:
        for row in _read_processed_rows(path):
            row_count += 1
            for mapping in _iter_history_mappings(row):
                key = evidence_window_key(mapping)
                if key is not None:
                    window_keys.add(key)
                question = mapping.get("question")
                if isinstance(question, str) and normalize_question(question):
                    question_hashes.add(question_fingerprint(question))
    return {
        "sources": [str(Path(source)) for source in sources],
        "scanned_files": [str(path) for path in files],
        "scanned_file_count": len(files),
        "scanned_row_count": row_count,
        "window_keys": sorted(window_keys),
        "question_hashes": sorted(question_hashes),
    }


def validate_fresh_output(
    *,
    qa_path: str | Path,
    freshness_manifest_path: str | Path,
) -> dict[str, Any]:
    """Fail if generated questions repeat a prior or same-batch question."""

    freshness = json.loads(Path(freshness_manifest_path).read_text(encoding="utf-8"))
    prior_hashes = set(freshness.get("prior_question_hashes") or [])
    current_hashes: list[str] = []
    for row in _read_processed_rows(Path(qa_path)):
        question = row.get("question")
        if not isinstance(question, str) or not normalize_question(question):
            raise ValueError("generated QA row is missing a non-empty question")
        current_hashes.append(question_fingerprint(question))
    repeated_prior = sorted(prior_hashes.intersection(current_hashes))
    repeated_batch = sorted(
        fingerprint
        for fingerprint in set(current_hashes)
        if current_hashes.count(fingerprint) > 1
    )
    if repeated_prior or repeated_batch:
        raise RuntimeError(
            "fresh-question validation failed: "
            f"prior_duplicates={len(repeated_prior)} "
            f"within_batch_duplicates={len(repeated_batch)}"
        )
    return {
        "passed": True,
        "question_count": len(current_hashes),
        "prior_question_fingerprint_count": len(prior_hashes),
        "prior_duplicate_count": 0,
        "within_batch_duplicate_count": 0,
    }


def six_user_context_plan(
    *,
    generator_frame_budget: int = SIX_USER_DEFAULT_GENERATOR_FRAME_BUDGET,
    judge_video_fps: float = DEFAULT_MODEL_VIDEO_FPS,
    generator_max_image_pixels: int = DEFAULT_GENERATOR_MAX_IMAGE_PIXELS,
) -> dict[str, Any]:
    if generator_frame_budget <= 0:
        raise ValueError("generator_frame_budget must be positive")
    if judge_video_fps <= 0:
        raise ValueError("judge_video_fps must be positive")
    aggregate_seconds = SIX_USER_COUNT * LONG_CONTEXT_EVIDENCE_DURATION_SECONDS
    pruning_sample_fps = 1.0 / SIX_USER_DEFAULT_SAMPLE_INTERVAL_SECONDS
    pruning_frames = round(aggregate_seconds * pruning_sample_fps)
    judge_frames_per_visual_call = round(
        LONG_CONTEXT_EVIDENCE_DURATION_SECONDS * judge_video_fps
    )
    judge_aggregate_frames = SIX_USER_COUNT * judge_frames_per_visual_call
    adaptive_generator_max_pixels = memory_safe_image_max_pixels(
        image_count=generator_frame_budget,
        configured_max_image_pixels=generator_max_image_pixels,
        max_input_tokens=DEFAULT_MAX_INPUT_TOKENS,
    )
    generator_tokens_per_frame = (
        adaptive_generator_max_pixels // QWEN_VISION_TOKEN_PIXEL_AREA
    )
    estimated_generator_visual_tokens = (
        generator_frame_budget * generator_tokens_per_frame
    )
    estimated_generator_input_tokens = (
        estimated_generator_visual_tokens
        + DEFAULT_TEXT_TOKEN_RESERVE
        + generator_frame_budget
        * MEMORY_SAFE_DEFAULT_IMAGE_ITEM_TOKEN_OVERHEAD
    )
    estimated_judge_visual_tokens = (
        judge_frames_per_visual_call * ESTIMATED_VISUAL_TOKENS_PER_FRAME
    )
    estimated_judge_input_tokens = (
        estimated_judge_visual_tokens + DEFAULT_TEXT_TOKEN_RESERVE
    )
    estimated_input_tokens = max(
        estimated_generator_input_tokens,
        estimated_judge_input_tokens,
    )
    context_target_tokens = int(
        DEFAULT_MAX_INPUT_TOKENS
        * MEMORY_SAFE_DEFAULT_IMAGE_CONTEXT_TARGET_FRACTION
    )
    if estimated_input_tokens > DEFAULT_MAX_INPUT_TOKENS:
        raise ValueError(
            "six-user context plan exceeds the input-token ceiling: "
            f"estimated={estimated_input_tokens} max={DEFAULT_MAX_INPUT_TOKENS}"
        )
    return {
        "users_per_packet": SIX_USER_COUNT,
        "duration_seconds_per_user": LONG_CONTEXT_EVIDENCE_DURATION_SECONDS,
        "aggregate_media_seconds": aggregate_seconds,
        "aggregate_media_minutes": aggregate_seconds / 60.0,
        "clip_analysis_sample_fps": pruning_sample_fps,
        "pruning_sample_fps": pruning_sample_fps,
        "pruning_sampled_frames_per_user": round(
            LONG_CONTEXT_EVIDENCE_DURATION_SECONDS * pruning_sample_fps
        ),
        "pruning_aggregate_sampled_frames": pruning_frames,
        "pruning_cluster_window_seconds": SIX_USER_DEFAULT_CLUSTER_WINDOW_SECONDS,
        "pruning_clusters_per_window": SIX_USER_DEFAULT_CLUSTERS_PER_WINDOW,
        "generator_aggregate_frame_budget": generator_frame_budget,
        "generator_frame_policy": "complete_surviving_sampled_frames_no_thinning",
        "generator_frame_count_is_data_dependent_after_provider_pruning": True,
        "generator_configured_max_image_pixels": generator_max_image_pixels,
        "generator_worst_case_adaptive_max_image_pixels": (
            adaptive_generator_max_pixels
        ),
        "generator_worst_case_tokens_per_frame": generator_tokens_per_frame,
        "generator_adaptive_image_context_target_fraction": (
            MEMORY_SAFE_DEFAULT_IMAGE_CONTEXT_TARGET_FRACTION
        ),
        "context_target_tokens": context_target_tokens,
        "generator_image_item_token_overhead": (
            MEMORY_SAFE_DEFAULT_IMAGE_ITEM_TOKEN_OVERHEAD
        ),
        "estimated_generator_visual_tokens": estimated_generator_visual_tokens,
        "estimated_generator_input_tokens": estimated_generator_input_tokens,
        "judge_video_fps": judge_video_fps,
        "judge_temporal_coverage": "full_600_seconds_per_user",
        "judge_review_mode": "per_user_source_segment_map_reduce",
        "judge_visual_scope": "one_user_per_visual_call",
        "judge_source_segments_per_visual_call": 20,
        "estimated_judge_frames_per_visual_call": judge_frames_per_visual_call,
        "estimated_judge_aggregate_frames": judge_aggregate_frames,
        "estimated_judge_visual_tokens": estimated_judge_visual_tokens,
        "estimated_judge_input_tokens": estimated_judge_input_tokens,
        "largest_visual_frame_count": max(
            generator_frame_budget,
            judge_frames_per_visual_call,
        ),
        "estimated_visual_tokens": max(
            estimated_generator_visual_tokens,
            estimated_judge_visual_tokens,
        ),
        "text_token_reserve": DEFAULT_TEXT_TOKEN_RESERVE,
        "estimated_input_tokens": estimated_input_tokens,
        "max_input_tokens": DEFAULT_MAX_INPUT_TOKENS,
        "estimated_context_headroom_tokens": (
            DEFAULT_MAX_INPUT_TOKENS - estimated_input_tokens
        ),
        "estimated_target_headroom_tokens": (
            context_target_tokens - estimated_input_tokens
        ),
        "conservative_judge_video_fps": DEFAULT_MODEL_VIDEO_FPS,
    }


def write_memory_safe_context_script(
    plan: dict[str, Any], output_dir: str | Path
) -> str:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "qwen_memory_safe_six_user_10min.sh"
    path.write_text(
        "# Generated six-user ten-minute model-context controls.\n"
        "# Source before generate_video_qa_loop. Judge FPS and generator image\n"
        "# resolution are context controls; the sampled generator frame set is unchanged.\n"
        "# CLIP pruning remains 1 FPS with K=12 per fixed 30-second block.\n"
        f"export QWEN_MEMORY_SAFE_VIDEO_FPS={float(plan['judge_video_fps']):.6g}\n"
        f"export QWEN_MEMORY_SAFE_MAX_INPUT_TOKENS={DEFAULT_MAX_INPUT_TOKENS}\n"
        "export QWEN_MEMORY_SAFE_ADAPTIVE_IMAGE_PIXELS=1\n"
        "export QWEN_MEMORY_SAFE_IMAGE_CONTEXT_TARGET_FRACTION="
        f"{MEMORY_SAFE_DEFAULT_IMAGE_CONTEXT_TARGET_FRACTION:.6g}\n"
        "export QWEN_MEMORY_SAFE_IMAGE_TEXT_TOKEN_RESERVE="
        f"{MEMORY_SAFE_DEFAULT_IMAGE_TEXT_TOKEN_RESERVE}\n"
        "export QWEN_MEMORY_SAFE_IMAGE_ITEM_TOKEN_OVERHEAD="
        f"{MEMORY_SAFE_DEFAULT_IMAGE_ITEM_TOKEN_OVERHEAD}\n"
        "export QWEN_MEMORY_SAFE_MIN_AVAILABLE_RAM_GIB=160\n"
        "export QWEN_MEMORY_SAFE_DEVICE_MAP=cuda\n"
        "export MALLOC_ARENA_MAX=2\n",
        encoding="utf-8",
        newline="\n",
    )
    return str(path)


def representative_six_user_packet() -> dict[str, Any]:
    users = ["<SPEAKER>", *[f"<PROVIDER_{index}>" for index in range(1, 6)]]
    roles = [SIX_USER_SPEAKER_MEDIA_ROLE, *([SIX_USER_PROVIDER_MEDIA_ROLE] * 5)]
    modes = [SIX_USER_SPEAKER_FRAME_MODE, *([SIX_USER_PROVIDER_FRAME_MODE] * 5)]
    clips = []
    for user_index, (user, role, mode) in enumerate(zip(users, roles, modes)):
        clips.append(
            {
                "agent_name": user,
                "duration_seconds": LONG_CONTEXT_EVIDENCE_DURATION_SECONDS,
                "segment_count": 20,
                "full_local_video": f"<FULL_VIDEO_{user_index + 1}>",
                "generator_media_mode": mode,
                "media_role": role,
                "is_pruned": user_index != 0,
                "force_frame_inputs": True,
                "frames": [
                    {
                        "path": f"<FRAME_{user_index + 1}_{frame_index + 1}>",
                        "timestamp_seconds": float(frame_index * 180),
                        "is_cluster_medoid": True,
                    }
                    for frame_index in range(3)
                ],
                "context_sampling": {
                    "policy": "complete_surviving_sampled_frames",
                    "analysis_sample_fps": (
                        1.0 / SIX_USER_DEFAULT_SAMPLE_INTERVAL_SECONDS
                    ),
                    "source_frame_count": 3,
                    "model_input_frame_count": 3,
                    "effective_model_input_fps": 0.005,
                    "aggregate_frame_budget": (
                        SIX_USER_DEFAULT_GENERATOR_FRAME_BUDGET
                    ),
                },
                "temporal_pruning": {
                    "method": "speaker_provider_time_aware_provider_only",
                    "comparison_scope": (
                        "every_asker_cluster_x_every_provider_cluster_"
                        "within_plus_minus_time_window"
                    ),
                    "temporal_policy": SIX_USER_TIME_AWARE_TEMPORAL_POLICY,
                    "pruned_side": "providers_only",
                    "asker_preserved": True,
                    "max_pair_time_difference_seconds": (
                        SIX_USER_DEFAULT_MAX_PAIR_TIME_DIFFERENCE_SECONDS
                    ),
                    "mutual_nearest_only": False,
                    "split_noncontiguous_clusters": True,
                    "max_cluster_member_gap_seconds": (
                        SIX_USER_DEFAULT_MAX_CLUSTER_MEMBER_GAP_SECONDS
                    ),
                    "cluster_count_per_window": (
                        SIX_USER_DEFAULT_CLUSTERS_PER_WINDOW
                    ),
                    "cluster_window_seconds": (
                        SIX_USER_DEFAULT_CLUSTER_WINDOW_SECONDS
                    ),
                    "pruning_protection_mode": "min_percent",
                    "min_pruned_video_percent": DEFAULT_MIN_PRUNED_VIDEO_PERCENT,
                    "kept_duration_seconds": (
                        600.0 if user_index == 0 else 360.0
                    ),
                    "removed_duration_seconds": (
                        0.0 if user_index == 0 else 240.0
                    ),
                },
            }
        )
    return {
        "evidence_id": "<SIX_USER_TEN_MINUTE_PACKET>",
        "candidate_type": "six_user_speaker_consensus",
        "duration_seconds": LONG_CONTEXT_EVIDENCE_DURATION_SECONDS,
        "input_users": users,
        "required_users": users,
        "speaker_user": users[0],
        "provider_users": users[1:],
        "generator_media_mode": SIX_USER_GENERATOR_MEDIA_MODE,
        "generator_context_budget": {
            "policy": "complete_surviving_sampled_frames",
            "analysis_sample_fps": (
                1.0 / SIX_USER_DEFAULT_SAMPLE_INTERVAL_SECONDS
            ),
            "aggregate_frame_budget": SIX_USER_DEFAULT_GENERATOR_FRAME_BUDGET,
            "source_frame_count": 18,
            "model_input_frame_count": 18,
        },
        "speaker_consensus_pruning": {
            "method": "speaker_provider_time_aware_provider_only",
            "comparison_scope": (
                "every_asker_cluster_x_every_provider_cluster_"
                "within_plus_minus_time_window"
            ),
            "temporal_policy": SIX_USER_TIME_AWARE_TEMPORAL_POLICY,
            "pruned_side": "providers_only",
            "asker_preserved": True,
            "max_pair_time_difference_seconds": (
                SIX_USER_DEFAULT_MAX_PAIR_TIME_DIFFERENCE_SECONDS
            ),
            "mutual_nearest_only": False,
            "split_noncontiguous_clusters": True,
            "max_cluster_member_gap_seconds": (
                SIX_USER_DEFAULT_MAX_CLUSTER_MEMBER_GAP_SECONDS
            ),
            "cluster_count_per_window": SIX_USER_DEFAULT_CLUSTERS_PER_WINDOW,
            "cluster_window_seconds": SIX_USER_DEFAULT_CLUSTER_WINDOW_SECONDS,
            "pruning_protection_mode": "min_percent",
            "min_pruned_video_percent": DEFAULT_MIN_PRUNED_VIDEO_PERCENT,
        },
        "clips": clips,
    }


def validate_prompt_contract(packet: dict[str, Any] | None = None) -> dict[str, Any]:
    active_packet = packet or representative_six_user_packet()
    if OPTIONAL_LONG_HORIZON_GUIDANCE.count("Example structure:") != 5:
        raise RuntimeError("six-user long-horizon guidance must contain five examples")
    for title in LONG_HORIZON_SELECTION_TITLES:
        if title not in OPTIONAL_LONG_HORIZON_GUIDANCE:
            raise RuntimeError(f"six-user long-horizon guidance is missing {title}")

    prompts = {}
    for question_type in ("neutral", "commonality", "difference"):
        prompt = build_video_generation_prompt(active_packet, question_type)
        if OPTIONAL_LONG_HORIZON_GUIDANCE not in prompt:
            raise RuntimeError(
                f"{question_type} six-user prompt omitted long-horizon examples"
            )
        for internal_pruning_value in (
            SIX_USER_TIME_AWARE_TEMPORAL_POLICY,
            (
                "every_asker_cluster_x_every_provider_cluster_"
                "within_plus_minus_time_window"
            ),
            '"speaker_consensus_pruning"',
            '"six_user_pruning_policy"',
            '"pruning_summary"',
        ):
            if internal_pruning_value in prompt:
                raise RuntimeError(
                    f"{question_type} six-user prompt exposed internal pruning metadata"
                )
        if "every sampled frame from the speaker" not in prompt:
            raise RuntimeError(
                f"{question_type} six-user prompt omitted asker preservation"
            )
        if "samples at one frame per second" not in prompt:
            raise RuntimeError(
                f"{question_type} six-user prompt omitted the pruning sample rate"
            )
        if "clusters each 30-second block independently with K=12" not in prompt:
            raise RuntimeError(
                f"{question_type} six-user prompt omitted clustering density"
            )
        if "only full-video judge decoding is downsampled" not in prompt:
            raise RuntimeError(
                f"{question_type} six-user prompt confused pruning and judge sampling"
            )
        if "complete_surviving_sampled_frames" not in prompt:
            raise RuntimeError(
                f"{question_type} six-user prompt omitted context sampling metadata"
            )
        prompts[question_type] = prompt

    preview_qa = representative_preview_qa(active_packet)
    judge_prompt = build_evidence_segment_observation_prompt(
        preview_qa,
        user=str((active_packet.get("required_users") or ["<SPEAKER>"])[0]),
        segment_count=int(active_packet.get("segment_count") or 20),
    )
    reduce_prompt = build_evidence_observation_aggregation_prompt(
        preview_qa,
        active_packet,
        observations=[representative_observation(preview_qa["required_users"][0])],
    )
    for groundedness_prompt in (judge_prompt, reduce_prompt):
        if LONG_HORIZON_GROUNDEDNESS_GUIDANCE not in groundedness_prompt:
            raise RuntimeError("six-user groundedness prompt omitted long-horizon checks")
        for forbidden in ("local_video", "timestamp_seconds", "window_start_seconds"):
            if forbidden in groundedness_prompt:
                raise RuntimeError(
                    f"six-user groundedness prompt exposed exact media metadata: {forbidden}"
                )

    legacy_formality_prompt = build_qa_formality_judge_prompt(
        preview_qa,
        active_packet,
    )
    legacy_groundedness_prompt = build_evidence_groundedness_judge_prompt(
        preview_qa,
        active_packet,
    )
    legacy_answerability_prompts = {
        condition_type: build_answerability_prompt(
            preview_qa,
            {
                "condition_id": condition_type,
                "condition_type": condition_type,
                "users": list(active_packet.get("required_users") or []),
            },
        )
        for condition_type in ("speaker_only", "combined_all_six_users")
    }
    if LONG_HORIZON_FORMALITY_GUIDANCE not in legacy_formality_prompt:
        raise RuntimeError(
            "six-user legacy formality prompt omitted five long-horizon structures"
        )
    if LONG_HORIZON_GROUNDEDNESS_GUIDANCE not in legacy_groundedness_prompt:
        raise RuntimeError(
            "six-user legacy groundedness prompt omitted five long-horizon checks"
        )
    for condition_type, answerability_prompt in legacy_answerability_prompts.items():
        if LONG_HORIZON_GROUNDEDNESS_GUIDANCE not in answerability_prompt:
            raise RuntimeError(
                f"six-user legacy {condition_type} answerability prompt omitted five long-horizon checks"
            )
    return {
        "passed": True,
        "selection_count": len(LONG_HORIZON_SELECTION_TITLES),
        "long_horizon_example_count": OPTIONAL_LONG_HORIZON_GUIDANCE.count(
            "Example structure:"
        ),
        "temporal_policy": SIX_USER_TIME_AWARE_TEMPORAL_POLICY,
        "prompt_characters": {
            question_type: len(prompt)
            for question_type, prompt in prompts.items()
        },
        "judge_map_prompt_characters": len(judge_prompt),
        "judge_reduce_prompt_characters": len(reduce_prompt),
        "legacy_judge_prompt_characters": {
            "qa_formality": len(legacy_formality_prompt),
            "evidence_groundedness": len(legacy_groundedness_prompt),
            **{
                f"answerability_{condition_type}": len(prompt)
                for condition_type, prompt in legacy_answerability_prompts.items()
            },
        },
    }


def representative_preview_qa(packet: dict[str, Any]) -> dict[str, Any]:
    return {
        "qa_id": "<QA_ID>",
        "required_users": list(packet.get("required_users") or []),
        "question": "Which item completed the setup I had been working on?",
        "options": ["<OPTION_A>", "<OPTION_B>", "<OPTION_C>", "<OPTION_D>", "<OPTION_E>"],
        "correct": "A",
        "answer": "<OPTION_A>",
    }


def representative_fact_plan() -> dict[str, Any]:
    return {
        "reason": "The reference context and answer-bearing detail must both be visible.",
        "needed_facts": [
            {
                "fact_id": "F1",
                "fact": "The speaker-side setup reference is visible.",
                "why_needed": "It anchors the question to the speaker's experience.",
            },
            {
                "fact_id": "F2",
                "fact": "The answer-bearing completing item is visible.",
                "why_needed": "It resolves the missing detail.",
            },
        ],
    }


def representative_observation(user: str) -> dict[str, Any]:
    return {
        "user": user,
        "claims": [
            {
                "claim": "The material setup event is visible.",
                "status": "SUPPORTED",
                "segment_references": ["segment_001"],
                "visual_description": "The setup appears clearly in the ordered sequence.",
            }
        ],
    }


def write_prompt_previews(packet: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for question_type in ("neutral", "commonality", "difference"):
        path = output_dir / f"generator_prompt_{question_type}.txt"
        path.write_text(
            build_video_generation_prompt(packet, question_type),
            encoding="utf-8",
        )
        paths[question_type] = str(path)
    qa_item = representative_preview_qa(packet)
    fact_plan = representative_fact_plan()
    speaker = str((packet.get("required_users") or ["<SPEAKER>"])[0])
    segment_count = int(packet.get("segment_count") or 20)
    preview_prompts = {
        "answerability_fact_plan": build_answerability_fact_plan_prompt(qa_item),
        "answerability_user_fact_audit": build_answerability_user_fact_audit_prompt(
            qa_item,
            user=speaker,
            fact_plan=fact_plan,
            segment_count=segment_count,
        ),
        "answerability_condition_aggregation": (
            build_answerability_condition_aggregation_prompt(
                qa_item,
                condition={
                    "condition_id": f"speaker_only::{speaker}",
                    "condition_type": "speaker_only",
                    "users": [speaker],
                },
                fact_plan=fact_plan,
                user_audits=[],
            )
        ),
        "evidence_segment_observation": build_evidence_segment_observation_prompt(
            qa_item,
            user=speaker,
            segment_count=segment_count,
        ),
        "evidence_groundedness_aggregation": (
            build_evidence_observation_aggregation_prompt(
                qa_item,
                packet,
                observations=[representative_observation(speaker)],
            )
        ),
    }
    for name, prompt in preview_prompts.items():
        path = output_dir / f"{name}_prompt.txt"
        path.write_text(prompt, encoding="utf-8")
        paths[name] = str(path)
    return paths


def prepare_ten_minute_six_user(
    *,
    manifest_path: str | Path,
    output_root: str | Path,
    cache_dir: str | Path,
    target_count: int = 20,
    source_window_count: int | None = None,
    max_groups: int | None = None,
    random_seed: int = 42,
    model_id: str = DEFAULT_CLIP_MODEL,
    device: str = "auto",
    ffmpeg_binary: str = "ffmpeg",
    clip_batch_size: int = DEFAULT_CLIP_BATCH_SIZE,
    pruning_clusters_per_window: int = DEFAULT_PRUNING_CLUSTERS_PER_WINDOW,
    high_similarity_threshold: float = 0.82,
    min_pruned_video_percent: float = DEFAULT_MIN_PRUNED_VIDEO_PERCENT,
    generator_frame_budget: int = SIX_USER_DEFAULT_GENERATOR_FRAME_BUDGET,
    generator_max_image_pixels: int = DEFAULT_GENERATOR_MAX_IMAGE_PIXELS,
    judge_video_fps: float = DEFAULT_MODEL_VIDEO_FPS,
    exclude_processed_from: Sequence[str | Path] = (),
) -> dict[str, Any]:
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    if source_window_count is None:
        source_window_count = target_count
    if source_window_count <= 0:
        raise ValueError("source_window_count must be positive")
    if pruning_clusters_per_window <= 0:
        raise ValueError("pruning_clusters_per_window must be positive")
    if not 0 < min_pruned_video_percent <= 100:
        raise ValueError("min_pruned_video_percent must be in (0, 100]")

    output_root = Path(output_root)
    source_dir = output_root / "source_windows"
    candidate_dir = output_root / "time_aware_candidates"
    prompt_dir = output_root / "prompt_preflight"
    raw_windows_path = source_dir / "six_user_10min_source_windows.jsonl"
    candidates_path = candidate_dir / "six_user_10min_time_aware.jsonl"
    freshness_manifest_path = output_root / "freshness_exclusions.json"
    source_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    processed = collect_processed_evidence(exclude_processed_from)
    excluded_window_keys = {
        (str(day), str(time_token))
        for day, time_token in processed["window_keys"]
    }
    write_json(
        freshness_manifest_path,
        {
            "exclude_processed_sources": processed["sources"],
            "scanned_files": processed["scanned_files"],
            "scanned_file_count": processed["scanned_file_count"],
            "scanned_row_count": processed["scanned_row_count"],
            "excluded_source_windows": [
                {"day": day, "time_token": time_token}
                for day, time_token in sorted(excluded_window_keys)
            ],
            "excluded_source_window_count": len(excluded_window_keys),
            "prior_question_hashes": processed["question_hashes"],
            "prior_question_fingerprint_count": len(processed["question_hashes"]),
        },
    )
    print(
        "ten_minute_prepare status=freshness_exclusions_resolved "
        f"sources={len(processed['sources'])} "
        f"files={processed['scanned_file_count']} "
        f"windows={len(excluded_window_keys)} "
        f"questions={len(processed['question_hashes'])}",
        flush=True,
    )

    prompt_contract = validate_prompt_contract()
    context_plan = six_user_context_plan(
        generator_frame_budget=generator_frame_budget,
        judge_video_fps=judge_video_fps,
        generator_max_image_pixels=generator_max_image_pixels,
    )
    context_script = write_memory_safe_context_script(context_plan, output_root)

    print(
        "ten_minute_prepare status=encoder_loading "
        f"model_id={model_id} device={device}",
        flush=True,
    )
    encoder_started = time.monotonic()
    encoder = BatchedImageEncoder(
        TransformersClipEncoder(model_id, device=device),
        batch_size=clip_batch_size,
    )
    print(
        "ten_minute_prepare status=encoder_loaded "
        f"model_id={encoder.model_id} seconds={time.monotonic() - encoder_started:.3f}",
        flush=True,
    )

    source_packet_iterator = iter_evidence_packets(
        manifest_path=manifest_path,
        cache_dir=cache_dir,
        output_root=source_dir,
        target_count=source_window_count,
        users_per_case=SIX_USER_COUNT,
        frames_per_clip=1,
        evidence_duration_seconds=LONG_CONTEXT_EVIDENCE_DURATION_SECONDS,
        max_groups=max_groups,
        download_media=True,
        random_seed=random_seed,
        stratify_by_day=True,
        progress=True,
        excluded_group_keys=excluded_window_keys,
        skip_failed_groups=True,
    )
    rng = random.Random(random_seed)
    source_packets: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    group_reports = []
    while len(candidates) < target_count:
        try:
            source_packet = next(source_packet_iterator)
        except StopIteration:
            break
        source_index = len(source_packets)
        source_key = evidence_window_key(source_packet)
        if source_key is None:
            raise RuntimeError("prepared source packet has no synchronized window key")
        if source_key in excluded_window_keys:
            raise RuntimeError(
                "processed evidence window reached media analysis despite exclusion: "
                f"day={source_key[0]} time_token={source_key[1]}"
            )
        source_packets.append(source_packet)
        write_jsonl(raw_windows_path, source_packets)
        group_started = time.monotonic()
        print(
            "ten_minute_prepare "
            f"status=source_analysis_started source_index={source_index} "
            f"evidence_id={source_packet.get('evidence_id')} "
            f"prepared_windows={len(source_packets)} candidates={len(candidates)}",
            flush=True,
        )
        group = {
            "day": source_packet.get("day"),
            "time_token": source_packet.get("time_token"),
            "clip_clock": source_packet.get("clip_clock"),
            "duration_seconds": LONG_CONTEXT_EVIDENCE_DURATION_SECONDS,
            "clips": source_packet.get("clips") or [],
        }
        try:
            result = analyze_group_relative_similarity(
                group,
                output_dir=candidate_dir / "assets",
                cache_dir=cache_dir,
                encoder=encoder,
                duration_seconds=LONG_CONTEXT_EVIDENCE_DURATION_SECONDS,
                sample_interval_seconds=SIX_USER_DEFAULT_SAMPLE_INTERVAL_SECONDS,
                start_seconds=0.0,
                selected_count=SIX_USER_COUNT,
                high_similarity_interval_threshold=high_similarity_threshold,
                pruning_clusters_per_video=pruning_clusters_per_window,
                min_pruned_video_seconds=(
                    LONG_CONTEXT_EVIDENCE_DURATION_SECONDS
                    * min_pruned_video_percent
                    / 100.0
                ),
                pruning_protection_mode="min_percent",
                min_pruned_video_percent=min_pruned_video_percent,
                max_pair_time_difference_seconds=(
                    SIX_USER_DEFAULT_MAX_PAIR_TIME_DIFFERENCE_SECONDS
                ),
                mutual_nearest_only=False,
                split_noncontiguous_clusters=True,
                max_cluster_member_gap_seconds=(
                    SIX_USER_DEFAULT_MAX_CLUSTER_MEMBER_GAP_SECONDS
                ),
                cluster_window_seconds=SIX_USER_DEFAULT_CLUSTER_WINDOW_SECONDS,
                generator_frame_budget=generator_frame_budget,
                random_pair_first=True,
                rng=rng,
                ffmpeg_binary=ffmpeg_binary,
                download_media=False,
            )
        except Exception as exc:
            group_reports.append(
                {
                    "source_index": source_index,
                    "evidence_id": source_packet.get("evidence_id"),
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(
                "ten_minute_prepare "
                f"status=source_analysis_failed source_index={source_index} "
                f"error_type={type(exc).__name__} "
                f"seconds={time.monotonic() - group_started:.3f}",
                flush=True,
            )
            continue

        report_path = candidate_dir / "diagnostics" / f"source_{source_index:04d}.json"
        write_json(report_path, result)
        accepted_from_group = 0
        for candidate_result in result.get("speaker_candidates") or []:
            if len(candidates) >= target_count:
                break
            packet = build_candidate_packet(candidate_result)
            packet["source_window_evidence_id"] = source_packet.get("evidence_id")
            packet["group_relative_clip_similarity"]["result_path"] = str(report_path)
            pruning = packet.get("speaker_consensus_pruning") or {}
            if pruning.get("temporal_policy") != SIX_USER_TIME_AWARE_TEMPORAL_POLICY:
                raise RuntimeError("candidate omitted the required six-user temporal policy")
            context_budget = packet.get("generator_context_budget") or {}
            if int(context_budget.get("model_input_frame_count") or 0) > generator_frame_budget:
                raise RuntimeError("candidate exceeded the generator frame budget")
            candidates.append(packet)
            accepted_from_group += 1
        group_reports.append(
            {
                "source_index": source_index,
                "evidence_id": source_packet.get("evidence_id"),
                "status": "succeeded" if accepted_from_group else "no_candidate",
                "candidate_count": accepted_from_group,
                "diagnostics": str(report_path),
            }
        )
        write_jsonl(candidates_path, candidates)
        print(
            "ten_minute_prepare "
            f"status=source_analysis_completed source_index={source_index} "
            f"new_candidates={accepted_from_group} total_candidates={len(candidates)} "
            f"target={target_count} seconds={time.monotonic() - group_started:.3f}",
            flush=True,
        )

    if not source_packets:
        raise RuntimeError("no complete synchronized six-user ten-minute windows were found")
    if len(candidates) < target_count:
        raise RuntimeError(
            f"requested {target_count} six-user candidates but produced {len(candidates)}; "
            "increase --source-window-count or --max-groups"
        )
    write_jsonl(candidates_path, candidates)
    prepared_window_keys = {
        key
        for packet in source_packets
        if (key := evidence_window_key(packet)) is not None
    }
    freshness_overlap = sorted(prepared_window_keys.intersection(excluded_window_keys))
    if freshness_overlap:
        raise RuntimeError(
            f"prepared {len(freshness_overlap)} previously processed source windows"
        )
    actual_prompt_contract = validate_prompt_contract(candidates[0])
    prompt_previews = write_prompt_previews(candidates[0], prompt_dir)
    summary = {
        "passed": True,
        "users_per_packet": SIX_USER_COUNT,
        "duration_seconds_per_user": LONG_CONTEXT_EVIDENCE_DURATION_SECONDS,
        "aggregate_media_minutes_total_coverage": 60.0,
        "maximum_visual_media_minutes_per_judge_call": 10.0,
        "target_count": target_count,
        "source_window_limit": source_window_count,
        "source_window_count": len(source_packets),
        "candidate_count": len(candidates),
        "raw_source_windows": str(raw_windows_path),
        "candidates": str(candidates_path),
        "group_reports": group_reports,
        "fresh_evidence": {
            "passed": True,
            "freshness_manifest": str(freshness_manifest_path),
            "exclude_processed_sources": processed["sources"],
            "scanned_file_count": processed["scanned_file_count"],
            "scanned_row_count": processed["scanned_row_count"],
            "excluded_source_window_count": len(excluded_window_keys),
            "prepared_source_window_count": len(prepared_window_keys),
            "source_window_overlap_count": 0,
            "prior_question_fingerprint_count": len(processed["question_hashes"]),
        },
        "acceleration": {
            "processed_windows_filtered_before_download_assembly_and_clip": True,
            "lazy_source_window_preparation": True,
            "ffmpeg_processes_per_sampled_video": 1,
            "clusters_computed_once_per_video_and_reused_for_all_askers": True,
            "full_judge_video_materialization": "hardlink_with_copy_fallback",
            "per_user_source_segment_map_reduce_judges": True,
            "pruning_contract_changed": False,
        },
        "time_aware_pruning": {
            "temporal_policy": SIX_USER_TIME_AWARE_TEMPORAL_POLICY,
            "analysis_sample_fps": (
                1.0 / SIX_USER_DEFAULT_SAMPLE_INTERVAL_SECONDS
            ),
            "cluster_window_seconds": SIX_USER_DEFAULT_CLUSTER_WINDOW_SECONDS,
            "clusters_per_window": pruning_clusters_per_window,
            "max_pair_time_difference_seconds": (
                SIX_USER_DEFAULT_MAX_PAIR_TIME_DIFFERENCE_SECONDS
            ),
            "mutual_nearest_only": False,
            "split_noncontiguous_clusters": True,
            "max_cluster_member_gap_seconds": (
                SIX_USER_DEFAULT_MAX_CLUSTER_MEMBER_GAP_SECONDS
            ),
            "pruning_protection_mode": "min_percent",
            "min_pruned_video_percent": min_pruned_video_percent,
            "pruned_side": "providers_only",
            "speaker_preserved": True,
        },
        "context_plan": context_plan,
        "context_environment_script": context_script,
        "prompt_contract_preflight": prompt_contract,
        "actual_packet_prompt_preflight": actual_prompt_contract,
        "prompt_previews": prompt_previews,
        "media_routing": {
            "generator": (
                "complete 1 FPS asker samples and complete surviving provider-cluster samples"
            ),
            "groundedness": (
                "six independent 600-second source-segment map calls followed by one text reduction"
            ),
            "answerability": (
                "one shared fact plan, six independent 600-second source-segment audits, and two text condition reductions"
            ),
        },
    }
    write_json(output_root / "six_user_10min_setup_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare six-user ten-minute time-aware QA evidence"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser(
        "preflight", help="Validate prompts and per-user judge context budgets"
    )
    preflight.add_argument("--output-dir")

    prepare = subparsers.add_parser(
        "prepare", help="Assemble, cluster, prune, and context-sample six-user windows"
    )
    prepare.add_argument("--manifest", required=True)
    prepare.add_argument("--output-root", required=True)
    prepare.add_argument("--cache-dir", required=True)
    prepare.add_argument("--target-count", type=int, default=20)
    prepare.add_argument("--source-window-count", type=int)
    prepare.add_argument("--max-groups", type=int)
    prepare.add_argument("--random-seed", type=int, default=42)
    prepare.add_argument("--model-id", default=DEFAULT_CLIP_MODEL)
    prepare.add_argument("--device", default="auto")
    prepare.add_argument("--ffmpeg-binary", default="ffmpeg")
    prepare.add_argument("--clip-batch-size", type=int, default=DEFAULT_CLIP_BATCH_SIZE)
    prepare.add_argument(
        "--pruning-clusters-per-window",
        "--pruning-clusters-per-video",
        dest="pruning_clusters_per_window",
        type=int,
        default=DEFAULT_PRUNING_CLUSTERS_PER_WINDOW,
        help="Clusters per fixed 30-second pruning block (default: 12)",
    )
    prepare.add_argument("--high-similarity-threshold", type=float, default=0.82)
    prepare.add_argument(
        "--min-pruned-video-percent",
        type=float,
        default=DEFAULT_MIN_PRUNED_VIDEO_PERCENT,
    )
    prepare.add_argument(
        "--generator-frame-budget",
        type=int,
        default=SIX_USER_DEFAULT_GENERATOR_FRAME_BUDGET,
    )
    prepare.add_argument(
        "--generator-max-image-pixels",
        type=int,
        default=DEFAULT_GENERATOR_MAX_IMAGE_PIXELS,
    )
    prepare.add_argument(
        "--judge-video-fps", type=float, default=DEFAULT_MODEL_VIDEO_FPS
    )
    prepare.add_argument(
        "--exclude-processed-from",
        action="append",
        default=[],
        help=(
            "Prior run directory or JSON/JSONL file. Repeat to exclude every source "
            "window prepared or processed there before downloading or CLIP analysis."
        ),
    )

    validate_fresh = subparsers.add_parser(
        "validate-fresh-output",
        help="Reject exact question reuse against the preparation-time history snapshot",
    )
    validate_fresh.add_argument("--qa", required=True)
    validate_fresh.add_argument("--freshness-manifest", required=True)
    validate_fresh.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preflight":
        report = {
            "prompt_contract": validate_prompt_contract(),
            "context_plan": six_user_context_plan(),
        }
        if args.output_dir:
            output_dir = Path(args.output_dir)
            report["prompt_previews"] = write_prompt_previews(
                representative_six_user_packet(), output_dir
            )
            report["context_environment_script"] = write_memory_safe_context_script(
                report["context_plan"], output_dir
            )
            write_json(output_dir / "six_user_10min_preflight.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "prepare":
        summary = prepare_ten_minute_six_user(
            manifest_path=args.manifest,
            output_root=args.output_root,
            cache_dir=args.cache_dir,
            target_count=args.target_count,
            source_window_count=args.source_window_count,
            max_groups=args.max_groups,
            random_seed=args.random_seed,
            model_id=args.model_id,
            device=args.device,
            ffmpeg_binary=args.ffmpeg_binary,
            clip_batch_size=args.clip_batch_size,
            pruning_clusters_per_window=args.pruning_clusters_per_window,
            high_similarity_threshold=args.high_similarity_threshold,
            min_pruned_video_percent=args.min_pruned_video_percent,
            generator_frame_budget=args.generator_frame_budget,
            generator_max_image_pixels=args.generator_max_image_pixels,
            judge_video_fps=args.judge_video_fps,
            exclude_processed_from=args.exclude_processed_from,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "validate-fresh-output":
        report = validate_fresh_output(
            qa_path=args.qa,
            freshness_manifest_path=args.freshness_manifest,
        )
        if args.output:
            write_json(args.output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
