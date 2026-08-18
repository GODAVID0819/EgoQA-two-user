"""Blindly test a selected model on accepted QAs using two or six videos.

In six-video mode, the tester retrieves the two unpruned original pair videos
and four exact-or-sampled context videos. In pair mode, it supplies only the two
selected-pair videos, using either the exact pruned generator inputs or their
full originals. It anonymizes media filenames, places the asker's recording
first, and reveals no identity, evidence, rationale, review, timestamp, or
correctness metadata beyond the asker's name.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from .io_utils import append_jsonl, iter_jsonl, write_json, write_jsonl
from .qwen3vl_runner import (
    DEFAULT_MAX_IMAGE_PIXELS,
    DEFAULT_MODEL_ID,
    DEFAULT_SAMPLING_TEMPERATURE,
    DEFAULT_SAMPLING_TOP_P,
    DEFAULT_VIDEO_FPS,
    GENERATOR_DECODING_MODES,
    MEMORY_SAFE_BACKEND,
    OPENROUTER_REASONING_EFFORTS,
    make_runner,
)
from .schema import OPTION_LETTERS, extract_json_object, normalize_correct
from .small_video_model_runner import SMALL_VIDEO_BACKENDS, make_small_video_runner


TEST_BACKENDS = (
    "transformers-local",
    MEMORY_SAFE_BACKEND,
    *SMALL_VIDEO_BACKENDS,
    "openai-compatible-local",
    "openrouter",
    "gemini",
    "dry-run",
)
VIDEO_INPUT_SCOPES = ("six", "pair")
PAIR_VIDEO_SOURCES = ("generator", "full")
_SAFE_COMPONENT_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _nonempty_file(path: str | Path | None) -> bool:
    if not path:
        return False
    candidate = Path(str(path))
    return candidate.is_file() and candidate.stat().st_size > 0


def _safe_component(value: Any) -> str:
    text = _SAFE_COMPONENT_RE.sub("_", str(value or "").strip()).strip("._")
    if not text:
        raise ValueError(f"cannot create a safe component from {value!r}")
    return text


def _unique_index(
    rows: list[dict[str, Any]],
    *,
    key: str,
    source: str | Path,
) -> dict[str, dict[str, Any]]:
    index = {}
    for row_number, row in enumerate(rows, 1):
        value = str(row.get(key) or "")
        if not value:
            raise ValueError(f"{source}:{row_number}: missing {key}")
        if value in index:
            raise ValueError(f"{source}: duplicate {key}={value}")
        index[value] = row
    return index


def _evaluation_id(
    *,
    qa_path: Path,
    source_index: int,
    row_number: int,
    qa: dict[str, Any],
) -> str:
    return "_".join(
        (
            "EVAL",
            f"{source_index:03d}",
            f"{row_number:06d}",
            _safe_component(qa.get("qa_id") or qa_path.stem),
        )
    )


def load_qa_rows(qa_paths: list[str | Path]) -> list[dict[str, Any]]:
    """Load every accepted row with a stable source-qualified evaluation ID."""

    rows = []
    seen_evaluation_ids = set()
    for source_index, qa_path in enumerate(qa_paths, 1):
        qa_path = Path(qa_path).resolve()
        for row_number, qa in enumerate(iter_jsonl(qa_path), 1):
            qa_id = str(qa.get("qa_id") or "")
            if not qa_id:
                raise ValueError(f"{qa_path}:{row_number}: accepted QA is missing qa_id")
            evaluation_id = _evaluation_id(
                qa_path=qa_path,
                source_index=source_index,
                row_number=row_number,
                qa=qa,
            )
            if evaluation_id in seen_evaluation_ids:
                raise ValueError(f"duplicate evaluation_id generated: {evaluation_id}")
            seen_evaluation_ids.add(evaluation_id)
            qa_copy = dict(qa)
            qa_copy["_qa_source_path"] = str(qa_path)
            qa_copy["_qa_source_row"] = row_number
            qa_copy["_evaluation_id"] = evaluation_id
            rows.append(qa_copy)
    if not rows:
        raise ValueError("no accepted QAs were found")
    return rows


def _asker_name(qa: dict[str, Any]) -> str:
    required_users = qa.get("required_users")
    if not isinstance(required_users, list) or not required_users:
        raise ValueError(f"{qa.get('qa_id')}: required_users does not identify an asker")
    asker = str(required_users[0] or "").strip()
    if not asker:
        raise ValueError(f"{qa.get('qa_id')}: asker name is empty")
    return asker


def _full_pair_video(packet_id: str, clip: dict[str, Any]) -> Path:
    for key in ("full_local_video", "original_local_video", "source_local_video"):
        value = clip.get(key)
        if _nonempty_file(value):
            return Path(str(value))
    raise FileNotFoundError(
        f"{packet_id}: no existing full original video for pair agent "
        f"{clip.get('agent_dir')}"
    )


def _generator_pair_video(packet_id: str, clip: dict[str, Any]) -> Path:
    for key in ("local_video", "pruned_local_video"):
        value = clip.get(key)
        if _nonempty_file(value):
            return Path(str(value))
    raise FileNotFoundError(
        f"{packet_id}: no existing pruned generator video for pair agent "
        f"{clip.get('agent_dir')}"
    )


def _validated_question(qa: dict[str, Any]) -> tuple[str, list[str], str]:
    qa_id = str(qa.get("qa_id") or "<missing qa_id>")
    question = str(qa.get("question") or "").strip()
    if not question:
        raise ValueError(f"{qa_id}: question is empty")
    options = qa.get("options")
    if (
        not isinstance(options, list)
        or len(options) != 5
        or any(not isinstance(option, str) or not option.strip() for option in options)
    ):
        raise ValueError(f"{qa_id}: options must contain five non-empty strings")
    correct = normalize_correct(qa.get("correct"))
    return question, list(options), correct


def _ordered_six_media(
    *,
    qa: dict[str, Any],
    evidence_packet: dict[str, Any],
    six_view_packet: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    qa_id = str(qa.get("qa_id") or "<missing qa_id>")
    evidence_id = str(qa.get("evidence_id") or "")
    asker = _asker_name(qa)

    pair_clips = evidence_packet.get("clips")
    if not isinstance(pair_clips, list) or len(pair_clips) != 2:
        raise ValueError(f"{evidence_id}: evidence packet must contain two pair clips")
    remaining_clips = six_view_packet.get("remaining_full_clips")
    if not isinstance(remaining_clips, list) or len(remaining_clips) != 4:
        raise ValueError(f"{evidence_id}: six-view manifest must contain four context clips")

    media = []
    for clip in pair_clips:
        if not isinstance(clip, dict):
            raise ValueError(f"{evidence_id}: pair clip is not an object")
        media.append(
            {
                "agent_dir": clip.get("agent_dir"),
                "agent_name": clip.get("agent_name"),
                "source_role": "selected_pair_full_original",
                "alignment": "selected_pair_exact",
                "synchronized_with_selected_pair": True,
                "source_video": str(_full_pair_video(evidence_id, clip)),
            }
        )
    for clip in remaining_clips:
        if not isinstance(clip, dict):
            raise ValueError(f"{evidence_id}: context clip is not an object")
        alignment = str(clip.get("alignment") or "")
        if alignment != "exact_synchronized" and not alignment.startswith(
            "fallback_sampled_"
        ):
            raise ValueError(
                f"{evidence_id}: context clip has unsupported alignment {alignment!r}"
            )
        local_video = clip.get("local_video")
        if not _nonempty_file(local_video):
            raise FileNotFoundError(
                f"{evidence_id}: context video is missing for {clip.get('agent_dir')}: "
                f"{local_video}"
            )
        media.append(
            {
                "agent_dir": clip.get("agent_dir"),
                "agent_name": clip.get("agent_name"),
                "source_role": "remaining_context",
                "alignment": alignment,
                "synchronized_with_selected_pair": bool(
                    clip.get("synchronized_with_selected_pair")
                ),
                "source_video": str(local_video),
                "source_day": clip.get("day"),
                "source_time_token": clip.get("time_token"),
            }
        )

    agent_dirs = [str(clip.get("agent_dir") or "") for clip in media]
    if any(not agent_dir for agent_dir in agent_dirs):
        raise ValueError(f"{evidence_id}: a media row is missing agent_dir")
    if len(agent_dirs) != 6 or len(set(agent_dirs)) != 6:
        raise ValueError(
            f"{evidence_id}: expected six distinct participant videos, found {agent_dirs}"
        )

    asker_matches = [
        clip
        for clip in media
        if str(clip.get("agent_name") or "").strip().casefold() == asker.casefold()
    ]
    if len(asker_matches) != 1:
        raise ValueError(
            f"{qa_id}: asker {asker!r} matched {len(asker_matches)} of the six videos"
        )
    asker_clip = asker_matches[0]
    if asker_clip["source_role"] != "selected_pair_full_original":
        raise ValueError(f"{qa_id}: asker is not part of the original selected pair")

    others = sorted(
        (clip for clip in media if clip is not asker_clip),
        key=lambda clip: str(clip.get("agent_dir") or ""),
    )
    ordered = [asker_clip, *others]
    for index, clip in enumerate(ordered, 1):
        clip["input_video_index"] = index
        clip["is_asker_video"] = index == 1
    return asker, ordered


def _ordered_pair_media(
    *,
    qa: dict[str, Any],
    evidence_packet: dict[str, Any],
    pair_video_source: str,
) -> tuple[str, list[dict[str, Any]]]:
    qa_id = str(qa.get("qa_id") or "<missing qa_id>")
    evidence_id = str(qa.get("evidence_id") or "")
    asker = _asker_name(qa)

    pair_clips = evidence_packet.get("clips")
    if not isinstance(pair_clips, list) or len(pair_clips) != 2:
        raise ValueError(f"{evidence_id}: evidence packet must contain two pair clips")

    media = []
    for clip in pair_clips:
        if not isinstance(clip, dict):
            raise ValueError(f"{evidence_id}: pair clip is not an object")
        if pair_video_source == "generator":
            source_video = _generator_pair_video(evidence_id, clip)
            source_role = "selected_pair_generator_video"
        elif pair_video_source == "full":
            source_video = _full_pair_video(evidence_id, clip)
            source_role = "selected_pair_full_original"
        else:
            raise ValueError(f"unsupported pair_video_source: {pair_video_source}")
        media.append(
            {
                "agent_dir": clip.get("agent_dir"),
                "agent_name": clip.get("agent_name"),
                "source_role": source_role,
                "alignment": "selected_pair_exact",
                "synchronized_with_selected_pair": True,
                "source_video": str(source_video),
            }
        )

    agent_dirs = [str(clip.get("agent_dir") or "") for clip in media]
    if any(not agent_dir for agent_dir in agent_dirs):
        raise ValueError(f"{evidence_id}: a pair media row is missing agent_dir")
    if len(agent_dirs) != 2 or len(set(agent_dirs)) != 2:
        raise ValueError(
            f"{evidence_id}: expected two distinct selected-pair videos, found {agent_dirs}"
        )

    asker_matches = [
        clip
        for clip in media
        if str(clip.get("agent_name") or "").strip().casefold() == asker.casefold()
    ]
    if len(asker_matches) != 1:
        raise ValueError(
            f"{qa_id}: asker {asker!r} matched {len(asker_matches)} of the two videos"
        )
    asker_clip = asker_matches[0]
    other_clip = next(clip for clip in media if clip is not asker_clip)
    ordered = [asker_clip, other_clip]
    for index, clip in enumerate(ordered, 1):
        clip["input_video_index"] = index
        clip["is_asker_video"] = index == 1
    return asker, ordered


def build_blind_video_prompt(
    qa: dict[str, Any],
    *,
    asker: str,
    video_input_scope: str,
) -> str:
    """Build a test prompt with no gold, evidence, or participant metadata."""

    question, options, _ = _validated_question(qa)
    option_lines = "\n".join(
        f"{letter}. {option}" for letter, option in zip(OPTION_LETTERS, options)
    )
    if video_input_scope == "six":
        media_description = (
            "Answer one multiple-choice question using the six video inputs.\n\n"
            f"The person asking the question is {asker}. Video 1 is that person's "
            "own recording. Videos 2 through 6 are additional unlabeled recordings.\n"
        )
    elif video_input_scope == "pair":
        media_description = (
            "Answer one multiple-choice question using the two video inputs.\n\n"
            f"The person asking the question is {asker}. Video 1 is that person's "
            "own recording. Video 2 is an additional unlabeled recording.\n"
        )
    else:
        raise ValueError(f"unsupported video_input_scope: {video_input_scope}")
    return (
        media_description
        + "No other identity or pairing information is provided. Base your answer "
        "only on the visible video content and the question.\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{option_lines}\n\n"
        'Return exactly one JSON object in this form: {"choice":"A"}\n'
        "Replace A with one of A, B, C, D, or E. Do not explain your answer."
    )


def build_blind_six_video_prompt(
    qa: dict[str, Any],
    *,
    asker: str,
) -> str:
    """Build a test prompt with no gold/evidence/pairing metadata."""

    return build_blind_video_prompt(
        qa,
        asker=asker,
        video_input_scope="six",
    )


def parse_model_choice(raw_output: str) -> tuple[str | None, str]:
    try:
        parsed = extract_json_object(raw_output)
    except Exception:
        parsed = None
    if isinstance(parsed, dict):
        for key in ("choice", "answer", "option"):
            if key in parsed:
                try:
                    return normalize_correct(parsed[key]), f"json_{key}"
                except ValueError:
                    pass

    stripped = raw_output.strip()
    direct = re.fullmatch(r"[\s`*]*([A-Ea-e])(?:[\s.`*)]*)", stripped)
    if direct:
        return direct.group(1).upper(), "direct_letter"
    explicit = re.search(
        r"\b(?:final\s+answer|answer|choice|option)\s*(?:is|=|:)?\s*"
        r"[\[(]?([A-E])[\])]?.?\b",
        raw_output,
        re.IGNORECASE,
    )
    if explicit:
        return explicit.group(1).upper(), "explicit_answer_text"
    return None, "unparsed"


def _anonymous_video_alias(source: Path, alias_dir: Path) -> tuple[Path, str]:
    if not _nonempty_file(source):
        raise FileNotFoundError(f"cannot alias missing video: {source}")
    stat = source.stat()
    suffix = source.suffix.lower() or ".mp4"
    alias_name = f"{stat.st_dev:x}_{stat.st_ino:x}_{stat.st_size:x}_{stat.st_mtime_ns:x}"
    destination = alias_dir / f"{alias_name}{suffix}"
    if destination.exists():
        if not destination.is_file() or destination.stat().st_size != stat.st_size:
            raise ValueError(f"anonymous media alias conflicts with source: {destination}")
        return destination, "existing"

    alias_dir.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f"{destination.name}.",
        suffix=".tmp",
        dir=alias_dir,
    )
    os.close(fd)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        try:
            os.link(source, temporary)
            status = "hardlink"
        except OSError:
            shutil.copy2(source, temporary)
            status = "copy_fallback"
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination, status


def _prepare_cases(
    *,
    qa_rows: list[dict[str, Any]],
    evidence_index: dict[str, dict[str, Any]],
    six_view_index: dict[str, dict[str, Any]] | None,
    video_input_scope: str,
    pair_video_source: str,
) -> list[dict[str, Any]]:
    cases = []
    for qa in qa_rows:
        qa_id = str(qa.get("qa_id") or "")
        evaluation_id = str(qa.get("_evaluation_id") or "")
        if not evaluation_id:
            raise ValueError(f"{qa_id}: accepted QA is missing internal evaluation_id")
        evidence_id = str(qa.get("evidence_id") or "")
        if not evidence_id:
            raise ValueError(f"{qa_id}: accepted QA is missing evidence_id")
        evidence_packet = evidence_index.get(evidence_id)
        if evidence_packet is None:
            raise ValueError(f"{qa_id}: evidence_id not found in evidence JSONL: {evidence_id}")
        if video_input_scope == "six":
            if six_view_index is None:
                raise ValueError("six-view index is required in six-video mode")
            six_view_packet = six_view_index.get(evidence_id)
            if six_view_packet is None:
                raise ValueError(
                    f"{qa_id}: evidence_id not found in six-view manifest: {evidence_id}"
                )
            asker, media = _ordered_six_media(
                qa=qa,
                evidence_packet=evidence_packet,
                six_view_packet=six_view_packet,
            )
        elif video_input_scope == "pair":
            asker, media = _ordered_pair_media(
                qa=qa,
                evidence_packet=evidence_packet,
                pair_video_source=pair_video_source,
            )
        else:
            raise ValueError(f"unsupported video_input_scope: {video_input_scope}")
        question, options, correct = _validated_question(qa)
        cases.append(
            {
                "qa": qa,
                "qa_id": qa_id,
                "evaluation_id": evaluation_id,
                "evidence_id": evidence_id,
                "asker": asker,
                "question": question,
                "options": options,
                "correct_choice": correct,
                "media": media,
                "prompt": build_blind_video_prompt(
                    qa,
                    asker=asker,
                    video_input_scope=video_input_scope,
                ),
            }
        )
    return cases


def _result_summary(
    *,
    results: list[dict[str, Any]],
    model_id: str,
    backend: str,
    source_qa_count: int,
    selected_qa_count: int,
    output_path: Path,
    prompts_path: Path,
    video_input_scope: str,
    pair_video_source: str,
) -> dict[str, Any]:
    successful = [row for row in results if row.get("status") == "completed"]
    parsed = [row for row in successful if row.get("predicted_choice") in OPTION_LETTERS]
    correct = [row for row in parsed if row.get("is_correct") is True]
    errors = [row for row in results if row.get("status") == "error"]
    fallback_cases = [
        row
        for row in results
        if any(
            str(media.get("alignment") or "").startswith("fallback_sampled_")
            for media in row.get("media", [])
        )
    ]
    return {
        "backend": backend,
        "model_id": model_id,
        "source_qa_count": source_qa_count,
        "selected_qa_count": selected_qa_count,
        "result_count": len(results),
        "completed_count": len(successful),
        "model_error_count": len(errors),
        "parsed_choice_count": len(parsed),
        "parse_failure_count": len(successful) - len(parsed),
        "correct_count": len(correct),
        "accuracy_all_selected": (
            len(correct) / selected_qa_count if selected_qa_count else None
        ),
        "accuracy_completed": (
            len(correct) / len(successful) if successful else None
        ),
        "accuracy_parsed": len(correct) / len(parsed) if parsed else None,
        "case_count_with_fallback_video": len(fallback_cases),
        "output_path": str(output_path),
        "prompts_path": str(prompts_path),
        "video_input_scope": video_input_scope,
        "pair_video_source": pair_video_source,
        "testee_disclosure": (
            "asker name and the fact that Video 1 belongs to the asker; "
            + (
                "Videos 2-6 are unlabeled"
                if video_input_scope == "six"
                else "Video 2 is unlabeled"
            )
        ),
        "media_contract": (
            "two full unpruned originals from evidence JSONL plus four exact-or-sampled "
            "context videos from the six-view manifest"
            if video_input_scope == "six"
            else (
                "only the two exact pruned selected-pair videos used by the generator"
                if pair_video_source == "generator"
                else "only the two full unpruned selected-pair originals"
            )
        ),
    }


def run_six_video_qa_test(
    *,
    qa_paths: list[str | Path],
    evidence_path: str | Path,
    six_view_manifest_path: str | Path | None,
    output_path: str | Path,
    summary_path: str | Path | None = None,
    prompts_path: str | Path | None = None,
    media_alias_dir: str | Path | None = None,
    backend: str = "transformers-local",
    model_id: str = DEFAULT_MODEL_ID,
    base_url: str = "http://127.0.0.1:8000/v1",
    api_key: str | None = None,
    max_new_tokens: int = 64,
    max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
    dtype: str = "bfloat16",
    allow_cpu: bool = False,
    disable_thinking: bool = True,
    reasoning_effort: str | None = None,
    video_fps: float = DEFAULT_VIDEO_FPS,
    max_frames_per_video: int = 16,
    max_input_tokens: int | None = None,
    min_free_gib: float = 0.0,
    kv_bytes_per_token: int = 0,
    min_available_ram_gib: float = 0.0,
    attn_implementation: str = "sdpa",
    device_map: str = "auto",
    decoding_mode: str = "greedy",
    temperature: float = DEFAULT_SAMPLING_TEMPERATURE,
    top_p: float = DEFAULT_SAMPLING_TOP_P,
    top_k: int | None = None,
    start_index: int = 0,
    max_items: int | None = None,
    resume: bool = False,
    fail_fast: bool = False,
    runner: Any | None = None,
    video_input_scope: str = "six",
    pair_video_source: str = "generator",
) -> dict[str, Any]:
    """Run blind A-E evaluation and write one durable result row per QA."""

    if backend not in TEST_BACKENDS:
        raise ValueError(f"unsupported tester backend: {backend}")
    if decoding_mode not in GENERATOR_DECODING_MODES:
        raise ValueError(f"unsupported decoding_mode: {decoding_mode}")
    if video_input_scope not in VIDEO_INPUT_SCOPES:
        raise ValueError(f"unsupported video_input_scope: {video_input_scope}")
    if pair_video_source not in PAIR_VIDEO_SOURCES:
        raise ValueError(f"unsupported pair_video_source: {pair_video_source}")
    if video_input_scope == "six" and not six_view_manifest_path:
        raise ValueError("six_view_manifest_path is required in six-video mode")
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    if max_items is not None and max_items <= 0:
        raise ValueError("max_items must be positive when provided")
    if max_frames_per_video <= 0:
        raise ValueError("max_frames_per_video must be positive")

    qa_rows = load_qa_rows(qa_paths)
    selected_qa_rows = qa_rows[start_index:]
    if max_items is not None:
        selected_qa_rows = selected_qa_rows[:max_items]
    if not selected_qa_rows:
        raise ValueError("the requested QA slice is empty")

    evidence_rows = list(iter_jsonl(evidence_path))
    evidence_index = _unique_index(
        evidence_rows,
        key="evidence_id",
        source=evidence_path,
    )
    six_view_index = None
    if video_input_scope == "six":
        six_view_rows = list(iter_jsonl(six_view_manifest_path))
        six_view_index = _unique_index(
            six_view_rows,
            key="evidence_id",
            source=six_view_manifest_path,
        )
    cases = _prepare_cases(
        qa_rows=selected_qa_rows,
        evidence_index=evidence_index,
        six_view_index=six_view_index,
        video_input_scope=video_input_scope,
        pair_video_source=pair_video_source,
    )

    output_path = Path(output_path)
    summary_path = (
        Path(summary_path)
        if summary_path is not None
        else output_path.with_name(f"{output_path.stem}.summary.json")
    )
    prompts_path = (
        Path(prompts_path)
        if prompts_path is not None
        else output_path.with_name(f"{output_path.stem}.prompts.jsonl")
    )
    media_alias_dir = (
        Path(media_alias_dir)
        if media_alias_dir is not None
        else output_path.parent / f"{output_path.stem}_anonymous_media"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prompts_path.parent.mkdir(parents=True, exist_ok=True)

    if runner is not None:
        active_runner = runner
    elif backend in SMALL_VIDEO_BACKENDS:
        active_runner = make_small_video_runner(
            backend,
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            dtype=dtype,
            video_fps=video_fps,
            max_frames_per_video=max_frames_per_video,
            attn_implementation=attn_implementation,
            device_map=device_map,
        )
    else:
        active_runner = make_runner(
            backend,
            model_id=model_id,
            base_url=base_url,
            max_new_tokens=max_new_tokens,
            max_image_pixels=max_image_pixels,
            dtype=dtype,
            allow_cpu=allow_cpu,
            allow_openai_video_input=True,
            disable_thinking=disable_thinking,
            api_key=api_key,
            reasoning_effort=reasoning_effort,
            video_fps=video_fps,
            max_input_tokens=max_input_tokens,
            min_free_gib=min_free_gib,
            kv_bytes_per_token=kv_bytes_per_token,
            min_available_ram_gib=min_available_ram_gib,
            attn_implementation=attn_implementation,
            device_map=device_map,
        )
    effective_model_id = str(getattr(active_runner, "model_id", model_id))

    completed_evaluation_ids = set()
    if resume and output_path.exists():
        existing_results = list(iter_jsonl(output_path))
        selected_evaluation_ids = {case["evaluation_id"] for case in cases}
        cases_by_source_and_qa = {
            (
                str(case["qa"].get("_qa_source_path") or ""),
                case["qa_id"],
            ): case["evaluation_id"]
            for case in cases
        }
        evaluation_ids_by_qa_id: dict[str, list[str]] = {}
        for case in cases:
            evaluation_ids_by_qa_id.setdefault(case["qa_id"], []).append(
                case["evaluation_id"]
            )
        retained_results = []
        results_changed = False
        seen_existing_evaluation_ids = set()
        for result in existing_results:
            if str(result.get("backend") or "") != backend:
                raise ValueError("resume output contains a different backend")
            if str(result.get("model_id") or "") != effective_model_id:
                raise ValueError("resume output contains a different model_id")
            existing_qa_id = str(result.get("qa_id") or "")
            if not existing_qa_id:
                raise ValueError("resume output contains a row without qa_id")
            status = str(result.get("status") or "")
            if status not in {"completed", "error"}:
                raise ValueError(
                    f"resume output contains unsupported status {status!r} "
                    f"for qa_id={existing_qa_id}"
                )
            if status == "error":
                results_changed = True
                continue
            existing_evaluation_id = str(result.get("evaluation_id") or "")
            if not existing_evaluation_id:
                source_path = str(result.get("qa_source_path") or "")
                if source_path:
                    source_path = str(Path(source_path).resolve())
                existing_evaluation_id = cases_by_source_and_qa.get(
                    (source_path, existing_qa_id),
                    "",
                )
                if not existing_evaluation_id:
                    candidates = evaluation_ids_by_qa_id.get(existing_qa_id, [])
                    if len(candidates) == 1:
                        existing_evaluation_id = candidates[0]
                if not existing_evaluation_id:
                    raise ValueError(
                        "cannot map legacy resume row to a unique evaluation case: "
                        f"qa_id={existing_qa_id}"
                    )
                result = {
                    **result,
                    "evaluation_id": existing_evaluation_id,
                    "evaluation_id_migrated_from_legacy_result": True,
                }
                results_changed = True
            if existing_evaluation_id not in selected_evaluation_ids:
                raise ValueError(
                    "resume output contains evaluation outside the selected slice: "
                    f"{existing_evaluation_id}"
                )
            if existing_evaluation_id in seen_existing_evaluation_ids:
                raise ValueError(
                    "resume output contains duplicate evaluation_id: "
                    f"{existing_evaluation_id}"
                )
            seen_existing_evaluation_ids.add(existing_evaluation_id)
            completed_evaluation_ids.add(existing_evaluation_id)
            retained_results.append(result)
        if results_changed or len(retained_results) != len(existing_results):
            write_jsonl(output_path, retained_results)

        if prompts_path.exists():
            existing_prompts = list(iter_jsonl(prompts_path))
            upgraded_prompts = []
            prompts_changed = False
            retained_evaluation_ids_by_qa_id: dict[str, list[str]] = {}
            for result in retained_results:
                retained_evaluation_ids_by_qa_id.setdefault(
                    str(result.get("qa_id") or ""),
                    [],
                ).append(str(result.get("evaluation_id") or ""))
            for prompt_row in existing_prompts:
                prompt_evaluation_id = str(prompt_row.get("evaluation_id") or "")
                if prompt_evaluation_id in selected_evaluation_ids:
                    upgraded_prompts.append(prompt_row)
                    continue
                if prompt_evaluation_id:
                    prompts_changed = True
                    continue
                prompt_qa_id = str(prompt_row.get("qa_id") or "")
                candidates = retained_evaluation_ids_by_qa_id.get(prompt_qa_id, [])
                if len(candidates) != 1:
                    candidates = evaluation_ids_by_qa_id.get(prompt_qa_id, [])
                if len(candidates) != 1:
                    prompts_changed = True
                    continue
                upgraded_prompts.append(
                    {
                        **prompt_row,
                        "evaluation_id": candidates[0],
                        "evaluation_id_migrated_from_legacy_prompt": True,
                    }
                )
                prompts_changed = True
            if prompts_changed:
                write_jsonl(prompts_path, upgraded_prompts)
    else:
        output_path.unlink(missing_ok=True)
        prompts_path.unlink(missing_ok=True)

    for case_index, case in enumerate(cases, start=start_index):
        if case["evaluation_id"] in completed_evaluation_ids:
            continue
        media_audit = []
        anonymous_paths = []
        for media in case["media"]:
            alias, alias_status = _anonymous_video_alias(
                Path(media["source_video"]),
                media_alias_dir,
            )
            anonymous_paths.append(str(alias))
            media_audit.append(
                {
                    **media,
                    "anonymous_video": str(alias),
                    "alias_status": alias_status,
                }
            )

        prompt_row = {
            "evaluation_id": case["evaluation_id"],
            "qa_id": case["qa_id"],
            "evidence_id": case["evidence_id"],
            "asker": case["asker"],
            "prompt": case["prompt"],
            "video_paths_supplied_to_model": anonymous_paths,
            "video_count": len(anonymous_paths),
            "video_input_scope": video_input_scope,
            "pair_video_source": pair_video_source,
            "disclosed_identity_count": 1,
            "gold_fields_in_prompt": False,
        }
        append_jsonl(prompts_path, prompt_row)

        started = time.time()
        try:
            prepare_videos = getattr(active_runner, "prepare_videos", None)
            if callable(prepare_videos):
                prepare_videos(anonymous_paths)
            raw_output = active_runner.generate(
                case["prompt"],
                image_paths=[],
                video_paths=anonymous_paths,
                decoding_mode=decoding_mode,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            predicted_choice, parse_method = parse_model_choice(str(raw_output))
            result = {
                "case_index": case_index,
                "evaluation_id": case["evaluation_id"],
                "qa_id": case["qa_id"],
                "evidence_id": case["evidence_id"],
                "qa_source_path": case["qa"].get("_qa_source_path"),
                "asker": case["asker"],
                "backend": backend,
                "model_id": effective_model_id,
                "video_input_scope": video_input_scope,
                "pair_video_source": pair_video_source,
                "status": "completed",
                "question": case["question"],
                "options": case["options"],
                "correct_choice": case["correct_choice"],
                "predicted_choice": predicted_choice,
                "is_correct": (
                    predicted_choice == case["correct_choice"]
                    if predicted_choice is not None
                    else False
                ),
                "parse_method": parse_method,
                "raw_output": str(raw_output),
                "model_call_seconds": round(time.time() - started, 3),
                "media": media_audit,
                "prompt_disclosure": {
                    "asker_name": True,
                    "asker_video_position": 1,
                    "other_video_identities": False,
                    "original_pair_identity": False,
                    "gold_answer": False,
                    "generator_rationale": False,
                    "evidence_claims": False,
                    "review_metadata": False,
                },
            }
        except Exception as exc:
            result = {
                "case_index": case_index,
                "evaluation_id": case["evaluation_id"],
                "qa_id": case["qa_id"],
                "evidence_id": case["evidence_id"],
                "qa_source_path": case["qa"].get("_qa_source_path"),
                "asker": case["asker"],
                "backend": backend,
                "model_id": effective_model_id,
                "video_input_scope": video_input_scope,
                "pair_video_source": pair_video_source,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "model_call_seconds": round(time.time() - started, 3),
                "media": media_audit,
            }
            append_jsonl(output_path, result)
            if fail_fast:
                raise
            continue
        append_jsonl(output_path, result)

    results = list(iter_jsonl(output_path)) if output_path.exists() else []
    summary = _result_summary(
        results=results,
        model_id=effective_model_id,
        backend=backend,
        source_qa_count=len(qa_rows),
        selected_qa_count=len(cases),
        output_path=output_path,
        prompts_path=prompts_path,
        video_input_scope=video_input_scope,
        pair_video_source=pair_video_source,
    )
    summary.update(
        {
            "evidence_path": str(evidence_path),
            "six_view_manifest_path": (
                str(six_view_manifest_path) if six_view_manifest_path else None
            ),
            "qa_paths": [str(path) for path in qa_paths],
            "media_alias_dir": str(media_alias_dir),
            "decoding_mode": decoding_mode,
            "temperature": temperature if decoding_mode == "sampling" else 0,
            "top_p": top_p if decoding_mode == "sampling" else None,
            "top_k": top_k if decoding_mode == "sampling" else None,
            "start_index": start_index,
            "max_items": max_items,
            "max_frames_per_video": max_frames_per_video,
            "resume": resume,
        }
    )
    write_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    return summary


def add_six_video_tester_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--qa",
        action="append",
        required=True,
        help="Accepted qa_mcq.jsonl; repeat for multiple generation batches",
    )
    parser.add_argument("--evidence", required=True)
    parser.add_argument(
        "--six-view-manifest",
        help="Required only when --video-input-scope=six",
    )
    parser.add_argument(
        "--video-input-scope",
        choices=VIDEO_INPUT_SCOPES,
        default="six",
        help="Use all six videos or only the selected generation pair",
    )
    parser.add_argument(
        "--pair-video-source",
        choices=PAIR_VIDEO_SOURCES,
        default="generator",
        help="In pair mode, use the pruned generator inputs or full originals",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary")
    parser.add_argument("--prompts-output")
    parser.add_argument("--media-alias-dir")
    parser.add_argument("--backend", choices=TEST_BACKENDS, default="transformers-local")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-image-pixels", type=int, default=DEFAULT_MAX_IMAGE_PIXELS)
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--allow-cpu", action="store_true")
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument(
        "--disable-thinking",
        dest="disable_thinking",
        action="store_true",
    )
    thinking.add_argument(
        "--enable-thinking",
        dest="disable_thinking",
        action="store_false",
    )
    parser.set_defaults(disable_thinking=True)
    parser.add_argument("--video-fps", type=float, default=DEFAULT_VIDEO_FPS)
    parser.add_argument("--max-frames-per-video", type=int, default=16)
    parser.add_argument("--max-input-tokens", type=int)
    parser.add_argument("--min-free-gib", type=float, default=0.0)
    parser.add_argument("--kv-bytes-per-token", type=int, default=0)
    parser.add_argument("--min-available-ram-gib", type=float, default=0.0)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--reasoning-effort",
        choices=OPENROUTER_REASONING_EFFORTS,
    )
    parser.add_argument(
        "--decoding-mode",
        choices=GENERATOR_DECODING_MODES,
        default="greedy",
    )
    parser.add_argument("--temperature", type=float, default=DEFAULT_SAMPLING_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_SAMPLING_TOP_P)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Blindly test a selected video model on accepted QAs with two or six videos"
    )
    add_six_video_tester_args(parser)
    return parser


def run_six_video_test_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return run_six_video_qa_test(
        qa_paths=args.qa,
        evidence_path=args.evidence,
        six_view_manifest_path=args.six_view_manifest,
        video_input_scope=args.video_input_scope,
        pair_video_source=args.pair_video_source,
        output_path=args.output,
        summary_path=args.summary,
        prompts_path=args.prompts_output,
        media_alias_dir=args.media_alias_dir,
        backend=args.backend,
        model_id=args.model_id,
        base_url=args.base_url,
        api_key=args.api_key,
        max_new_tokens=args.max_new_tokens,
        max_image_pixels=args.max_image_pixels,
        dtype=args.dtype,
        allow_cpu=args.allow_cpu,
        disable_thinking=args.disable_thinking,
        reasoning_effort=args.reasoning_effort,
        video_fps=args.video_fps,
        max_frames_per_video=args.max_frames_per_video,
        max_input_tokens=args.max_input_tokens,
        min_free_gib=args.min_free_gib,
        kv_bytes_per_token=args.kv_bytes_per_token,
        min_available_ram_gib=args.min_available_ram_gib,
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
        decoding_mode=args.decoding_mode,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        start_index=args.start_index,
        max_items=args.max_items,
        resume=args.resume,
        fail_fast=args.fail_fast,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_six_video_test_from_args(args)
    print(
        f"tested={summary['result_count']} correct={summary['correct_count']} "
        f"accuracy_all_selected={summary['accuracy_all_selected']} "
        f"summary={summary['summary_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
