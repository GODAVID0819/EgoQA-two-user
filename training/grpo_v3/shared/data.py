"""Convert retained-frame EgoLife packets to ms-swift multimodal rows."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import inspect
import json
import sys
from collections.abc import Iterable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from .media import (
    clip_for_user,
    generator_frame_counts,
    ordered_generator_frame_paths,
    ordered_reviewer_video_paths,
    required_users,
    validate_bound_paths,
)


PRODUCTION_PROMPT_CONTRACT = "neutral_retained_frames_no_perspective_mixing_v1"
PRODUCTION_PROMPT_REQUIRED_FRAGMENTS = (
    "Treat required_users[0] as the asker",
    "Every first-person factual claim about what the speaker saw, noticed, did, handled, or experienced must be directly supported by required_users[0]",
    "Never present an action, observation, object, person, or event visible only in required_users[1]",
    "Hard attribution audit before returning",
    "Combined-video support does not make a provider-only fact part of the asker's experience",
    "Do not use shared-memory wording to reattribute provider-only evidence to the asker",
)


@lru_cache(maxsize=1)
def _production_prompt_builder() -> Any:
    try:
        module = importlib.import_module("egolife_two_user_qa.prompts")
    except ModuleNotFoundError as error:
        if error.name != "egolife_two_user_qa":
            raise
        # Supports deliberate execution from the package checkout itself.
        project_root = str(Path(__file__).resolve().parents[4])
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        module = importlib.import_module("egolife_two_user_qa.prompts")
    return module.build_video_generation_prompt


@lru_cache(maxsize=1)
def production_prompt_source_contract() -> dict[str, str]:
    """Identify and hash the exact production prompt implementation in use."""

    builder = _production_prompt_builder()
    source_value = inspect.getsourcefile(builder)
    if not source_value:
        raise RuntimeError("production prompt builder has no inspectable source file")
    source = Path(source_value).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"production prompt source is missing: {source}")
    return {
        "prompt_builder": f"{builder.__module__}.{builder.__name__}",
        "prompt_source_path": str(source),
        "prompt_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


def _validate_production_prompt_text(prompt: str) -> None:
    missing = [
        fragment
        for fragment in PRODUCTION_PROMPT_REQUIRED_FRAGMENTS
        if fragment not in prompt
    ]
    if missing:
        raise ValueError(
            "production generator prompt is missing no-perspective-mixing safeguards: "
            f"{missing}"
        )


def _canonical_packet(packet: Mapping[str, Any]) -> dict[str, Any]:
    """Order clips by required user and store absolute frame/full-video paths."""

    canonical = copy.deepcopy(dict(packet))
    users = required_users(canonical)
    ordered_clips = []
    for user in users:
        source_clip = clip_for_user(canonical, user)
        if not isinstance(source_clip, dict):
            raise ValueError(f"clip for required user {user!r} must be a JSON object")
        ordered_clips.append(source_clip)
    canonical["clips"] = ordered_clips

    generator_frames = ordered_generator_frame_paths(canonical)
    reviewer_videos = ordered_reviewer_video_paths(canonical)
    frame_cursor = 0
    for user, frame_count, reviewer_video in zip(
        users, generator_frame_counts(canonical), reviewer_videos
    ):
        clip = clip_for_user(canonical, user)
        for frame, frame_path in zip(
            clip["frames"],
            generator_frames[frame_cursor:frame_cursor + frame_count],
        ):
            frame["path"] = frame_path
        frame_cursor += frame_count
        # Normalize full-video aliases without deleting provenance fields.
        clip["full_local_video"] = reviewer_video
    return canonical


def packet_to_swift_row(
    packet: dict[str, Any],
    *,
    question_type: str,
    generation_mode: str = "baseline",
) -> dict[str, Any]:
    evidence_id = str(packet.get("evidence_id") or "").strip()
    if not evidence_id:
        raise ValueError("packet is missing evidence_id")
    canonical = _canonical_packet(packet)
    users = required_users(canonical)
    generator_frames = ordered_generator_frame_paths(canonical)
    reviewer_videos = ordered_reviewer_video_paths(canonical)
    frame_counts = generator_frame_counts(canonical)
    builder = _production_prompt_builder()
    prompt = builder(canonical, question_type, generation_mode=generation_mode)
    _validate_production_prompt_text(prompt)
    source_contract = production_prompt_source_contract()
    image_placeholders = "\n".join("<image>" for _ in generator_frames)
    row = {
        "messages": [{"role": "user", "content": f"{image_placeholders}\n{prompt}"}],
        # ms-swift consumes only retained CLIP-cluster frames in this column.
        "images": generator_frames,
        # Immutable path columns are passed to the reward plugin for rebinding.
        "generator_image_paths": generator_frames,
        "reviewer_video_paths": reviewer_videos,
        "evidence_id": evidence_id,
        "packet_json": json.dumps(canonical, ensure_ascii=False),
        "question_type": str(question_type),
        "generation_mode": str(generation_mode),
        "prompt_contract": PRODUCTION_PROMPT_CONTRACT,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        **source_contract,
        "required_users": users,
        "image_order": users,
        "generator_frame_counts": frame_counts,
        "reviewer_video_order": users,
    }
    validate_swift_row(row)
    return row


def _packet_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    raw_packet = row.get("packet_json")
    try:
        packet = json.loads(raw_packet) if isinstance(raw_packet, str) else raw_packet
    except json.JSONDecodeError as error:
        raise ValueError("packet_json is not valid JSON") from error
    if not isinstance(packet, dict):
        raise ValueError("packet_json must encode one JSON object")
    return packet


def validate_swift_row(
    row: dict[str, Any],
    *,
    require_files: bool = True,
    require_current_prompt_source: bool = True,
) -> None:
    messages = row.get("messages")
    if (
        not isinstance(messages, list)
        or len(messages) != 1
        or not isinstance(messages[0], dict)
        or messages[0].get("role") != "user"
    ):
        raise ValueError("messages must contain exactly one user message")
    content = str(messages[0].get("content") or "")
    if "videos" in row:
        raise ValueError("retained-frame GRPO rows must not contain a videos column")
    if "<video>" in content:
        raise ValueError("retained-frame GRPO prompts must not contain <video> placeholders")

    packet = _packet_from_row(row)
    evidence_id = str(row.get("evidence_id") or "").strip()
    if not evidence_id or str(packet.get("evidence_id") or "").strip() != evidence_id:
        raise ValueError("row evidence_id does not match packet_json.evidence_id")
    users = required_users(packet)
    if row.get("required_users") != users:
        raise ValueError("required_users does not match packet_json.required_users")
    if row.get("image_order") != users:
        raise ValueError("image_order must exactly match required_users")
    if row.get("reviewer_video_order") != users:
        raise ValueError("reviewer_video_order must exactly match required_users")

    expected_generator = ordered_generator_frame_paths(
        packet, require_files=require_files
    )
    expected_reviewer = ordered_reviewer_video_paths(
        packet, require_files=require_files
    )
    image_prefix = "\n".join("<image>" for _ in expected_generator) + "\n"
    if not content.startswith(image_prefix):
        raise ValueError(
            "the user message must begin with one ordered <image> placeholder per "
            "retained generator frame"
        )
    prompt = content[len(image_prefix):]
    if content.count("<image>") != len(expected_generator):
        raise ValueError(
            "the user message must contain exactly one <image> placeholder per "
            "retained generator frame"
        )
    _validate_production_prompt_text(prompt)
    question_type = str(row.get("question_type") or "")
    generation_mode = str(row.get("generation_mode") or "")
    current_prompt = _production_prompt_builder()(
        packet,
        question_type,
        generation_mode=generation_mode,
    )
    _validate_production_prompt_text(current_prompt)
    if prompt != current_prompt:
        raise ValueError(
            "serialized prompt does not match the current production prompt output"
        )
    if row.get("prompt_contract") != PRODUCTION_PROMPT_CONTRACT:
        raise ValueError(
            f"prompt_contract must be {PRODUCTION_PROMPT_CONTRACT!r}"
        )
    if row.get("prompt_sha256") != hashlib.sha256(prompt.encode("utf-8")).hexdigest():
        raise ValueError("prompt_sha256 does not match the serialized production prompt")
    if require_current_prompt_source:
        source_contract = production_prompt_source_contract()
        for field, expected in source_contract.items():
            if row.get(field) != expected:
                raise ValueError(
                    f"{field} does not match the current production prompt implementation"
                )
    if row.get("generator_frame_counts") != generator_frame_counts(packet):
        raise ValueError(
            "generator_frame_counts must match clips[*].frames in required_users order"
        )
    validate_bound_paths(
        row.get("images"), expected_generator, field="images"
    )
    validate_bound_paths(
        row.get("generator_image_paths"),
        expected_generator,
        field="generator_image_paths",
    )
    validate_bound_paths(
        row.get("reviewer_video_paths"),
        expected_reviewer,
        field="reviewer_video_paths",
    )


def convert_packets(
    packets: Iterable[dict[str, Any]],
    *,
    question_type: str,
    generation_mode: str,
    max_prompts: int,
) -> list[dict[str, Any]]:
    if (
        isinstance(max_prompts, bool)
        or not isinstance(max_prompts, int)
        or max_prompts <= 0
    ):
        raise ValueError("max_prompts must be a positive integer")
    rows: list[dict[str, Any]] = []
    for packet in packets:
        if len(rows) >= max_prompts:
            break
        rows.append(
            packet_to_swift_row(
                packet,
                question_type=question_type,
                generation_mode=generation_mode,
            )
        )
    if not rows:
        raise ValueError("no evidence packets could be converted")
    return rows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an ms-swift retained-frame GRPO JSONL"
    )
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--question-type", choices=("neutral",), default="neutral")
    parser.add_argument("--generation-mode", choices=("baseline",), default="baseline")
    parser.add_argument("--max-prompts", type=int, default=1)
    args = parser.parse_args()
    rows = convert_packets(
        read_jsonl(args.evidence),
        question_type=args.question_type,
        generation_mode=args.generation_mode,
        max_prompts=args.max_prompts,
    )
    write_jsonl(args.output, rows)
    preview = args.output.with_name("dataset_preview.json")
    preview.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"rows": len(rows), "output": str(args.output), "preview": str(preview)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
