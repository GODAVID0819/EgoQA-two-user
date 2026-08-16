"""Collect fixed-size reward-model QA candidate sets with judge feedback.

This module is deliberately separate from ``generate_video_qa_loop`` and the
cyclic production pipeline.  It reuses their prompt, normalization, judging,
and answerability functions without changing their acceptance semantics.

For every frozen retained-cluster-frame evidence packet, the collector starts
independent feedback loops.  A loop contains at most three generation calls,
stops on the first production-gate pass, and otherwise feeds the immediately
preceding raw generation plus exact judge feedback into the next call.  New
independent loops are started until the packet has a fixed quota of valid QA
candidates.  Every valid candidate is retained, irrespective of its judge
outcome.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import re
import subprocess
import tempfile
import time
from typing import Any, Iterable

from .cluster_member_frame_sidecar import GENERATOR_MEDIA_MODE
from .cyclic_video_qa import SerializedJudgeRunner
from .io_utils import iter_jsonl, write_json, write_jsonl
from .manifest import seconds_from_time_token
from .prompts import (
    VIDEO_GENERATION_SCHEMA,
    build_video_generation_prompt,
    formality_participant_names,
    qa_formality_errors,
)
from .qwen3vl_runner import (
    DEFAULT_MAX_IMAGE_PIXELS,
    DEFAULT_MODEL_ID,
    DEFAULT_VIDEO_FPS,
    make_runner,
)
from .schema import OPTION_LETTERS, extract_json_object, validate_qa_item
from .staged_video_qa import _normalized_candidate_qa
from .video_qa_loop import (
    build_review_from_gates,
    generator_decode_config,
    media_for_clips,
    run_parallel_review_judges,
)


REWARD_COLLECTION_SCHEMA_VERSION = "reward_candidate_collection_v1"
DEFAULT_CANDIDATES_PER_PACKET = 6
DEFAULT_PACKETS_PER_OUTPUT_GROUP = 25
DEFAULT_MAX_ATTEMPTS_PER_LOOP = 3
DEFAULT_MAX_RAW_CALLS_PER_PACKET = 12
DEFAULT_MAX_MALFORMED_PER_PACKET = 3
DEFAULT_GENERATION_RETRIES = 2
DEFAULT_JUDGE_RETRIES = 2
DEFAULT_TEMPERATURE = 0.70
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 40


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_file_sha256(value: Any) -> str:
    path = inspect.getsourcefile(value)
    if not path:
        raise ValueError(f"could not resolve source file for {value!r}")
    return file_sha256(path)


def stable_generation_seed(
    *,
    run_seed: int,
    evidence_id: str,
    loop_index: int,
    attempt_in_loop: int,
    generation_index: int,
) -> int:
    """Return a process-stable 63-bit seed for one generation identity."""

    payload = "|".join(
        [
            str(run_seed),
            evidence_id,
            str(loop_index),
            str(attempt_in_loop),
            str(generation_index),
        ]
    )
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def set_generation_seed(seed: int) -> dict[str, Any]:
    """Seed Python, NumPy when available, and the active Torch CUDA process."""

    random.seed(seed)
    seeded = {"python": seed, "numpy": None, "torch": None, "cuda": False}
    try:
        import numpy as np

        numpy_seed = seed % (2**32)
        np.random.seed(numpy_seed)
        seeded["numpy"] = numpy_seed
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        seeded["torch"] = seed
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            seeded["cuda"] = True
    except ImportError:
        pass
    return seeded


def parse_clock_seconds(value: str, *, allow_24h: bool = False) -> float:
    """Parse HH:MM[:SS[.fraction]] for packet time-of-day filtering."""

    text = str(value).strip()
    match = re.fullmatch(
        r"(?P<hour>\d{1,2}):(?P<minute>\d{2})(?::(?P<second>\d{2}(?:\.\d+)?))?",
        text,
    )
    if not match:
        raise ValueError(f"invalid clock value {value!r}; expected HH:MM[:SS]")
    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    second = float(match.group("second") or 0.0)
    if allow_24h and hour == 24 and minute == 0 and second == 0:
        return 24 * 3600.0
    if not 0 <= hour <= 23 or not 0 <= minute <= 59 or not 0 <= second < 60:
        raise ValueError(f"clock value is out of range: {value!r}")
    return hour * 3600.0 + minute * 60.0 + second


def packet_clock_seconds(packet: dict[str, Any]) -> float:
    value = packet.get("clock_seconds")
    if value is not None:
        return float(value)
    token = str(packet.get("time_token") or "")
    return seconds_from_time_token(token)


def filter_evidence_packets(
    rows: Iterable[dict[str, Any]],
    *,
    days: tuple[str, ...] | None = None,
    start_clock_seconds: float = 0.0,
    end_clock_seconds: float = 24 * 3600.0,
    max_packets: int | None = None,
) -> list[dict[str, Any]]:
    """Return packets in an end-exclusive daily clock window."""

    if not 0 <= start_clock_seconds < end_clock_seconds <= 24 * 3600:
        raise ValueError("clock window must satisfy 00:00 <= start < end <= 24:00")
    if max_packets is not None and max_packets <= 0:
        raise ValueError("max_packets must be positive when provided")
    allowed_days = {str(day).strip().upper() for day in days or () if str(day).strip()}
    selected = []
    for packet in rows:
        day = str(packet.get("day") or "").upper()
        if allowed_days and day not in allowed_days:
            continue
        clock = packet_clock_seconds(packet)
        if start_clock_seconds <= clock < end_clock_seconds:
            selected.append(packet)
            if max_packets is not None and len(selected) >= max_packets:
                break
    return selected


def validate_raw_generated_qa(qa: dict[str, Any]) -> list[str]:
    """Validate the unmodified generator JSON before pipeline normalization."""

    errors: list[str] = []
    missing = sorted(set(VIDEO_GENERATION_SCHEMA) - set(qa))
    if missing:
        errors.append("missing raw fields: " + ", ".join(missing))

    for field in ("qa_id", "question", "answer"):
        value = qa.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{field} must be a non-empty string in raw output")

    question_type = qa.get("question_type")
    if question_type not in {"commonality", "difference", "neutral"}:
        errors.append("question_type must be commonality, difference, or neutral in raw output")

    options = qa.get("options")
    if not isinstance(options, list) or len(options) != 5:
        errors.append("raw options must contain exactly five entries")
    elif any(not isinstance(option, str) or not option.strip() for option in options):
        errors.append("all raw options must be non-empty strings")

    correct = qa.get("correct")
    if not isinstance(correct, str) or correct.strip().upper() not in OPTION_LETTERS:
        errors.append("raw correct must be exactly one option letter A-E")
    elif isinstance(options, list) and len(options) == 5:
        expected = options[OPTION_LETTERS.index(correct.strip().upper())]
        if qa.get("answer") != expected:
            errors.append("raw answer must exactly equal options[correct]")

    required_users = qa.get("required_users")
    if not isinstance(required_users, list) or len(required_users) < 2:
        errors.append("raw required_users must list at least two users")
    if not isinstance(qa.get("evidence"), list) or not qa.get("evidence"):
        errors.append("raw evidence must be a non-empty list")
    if not isinstance(qa.get("single_user_answerability"), dict):
        errors.append("raw single_user_answerability must be an object")
    if not isinstance(qa.get("review"), dict):
        errors.append("raw review must be an object")
    if not isinstance(qa.get("per_user_evidence_claims"), list):
        errors.append("raw per_user_evidence_claims must be a list")
    if not isinstance(qa.get("referred_timestamps"), list):
        errors.append("raw referred_timestamps must be a list")
    for field in ("combined_answerability", "generator_rationale", "why_two_users_needed"):
        value = qa.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"raw {field} must be a non-empty string")
    return errors


def answerability_class(
    qa: dict[str, Any], answerability: dict[str, Any]
) -> dict[str, Any]:
    """Preserve the current gate and add the agreed deterministic 3/2/1 class."""

    correct = str(qa.get("correct") or "").strip().upper()
    required_users = list(qa.get("required_users") or [])
    asker = required_users[0] if required_users else None
    provider = required_users[1] if len(required_users) > 1 else None
    evaluations = list(answerability.get("evaluations") or [])

    def choice_for(condition_type: str, user: Any = None) -> str | None:
        for row in evaluations:
            if row.get("condition_type") != condition_type:
                continue
            users = list(row.get("users") or [])
            if user is not None and users != [user]:
                continue
            choice = str(row.get("choice") or "").strip().upper()
            return choice if choice in OPTION_LETTERS else None
        return None

    asker_choice = choice_for("single_user", asker)
    provider_choice = choice_for("single_user", provider)
    combined_choice = choice_for("combined_all_users")
    if (
        correct in OPTION_LETTERS
        and combined_choice == correct
        and asker_choice in OPTION_LETTERS
        and asker_choice != correct
        and provider_choice in OPTION_LETTERS
        and provider_choice != correct
    ):
        score = 3
        label = "requires_both_videos"
    elif (
        correct in OPTION_LETTERS
        and combined_choice == correct
        and asker_choice in OPTION_LETTERS
        and asker_choice != correct
        and provider_choice == correct
    ):
        score = 2
        label = "provider_alone_answerable"
    else:
        score = 1
        label = "asker_answerable_unsupported_or_invalid"
    return {
        "score": score,
        "label": label,
        "correct": correct,
        "asker_user": asker,
        "provider_user": provider,
        "asker_choice": asker_choice,
        "provider_choice": provider_choice,
        "combined_choice": combined_choice,
        "current_answerability_gate": deepcopy(answerability.get("gate") or {}),
    }


def _existing_media(packet: dict[str, Any]) -> tuple[list[str], list[str]]:
    clips = packet.get("clips")
    if (
        packet.get("generator_media_mode") != GENERATOR_MEDIA_MODE
        or not isinstance(clips, list)
        or len(clips) != 2
    ):
        raise ValueError(
            f"{packet.get('evidence_id')}: expected a two-user retained-cluster-frame packet"
        )
    generator_images, generator_videos = media_for_clips(
        clips,
        backend="transformers-local",
        allow_openai_video_input=False,
        media_role="generator",
    )
    reward_images, reward_videos = media_for_clips(
        clips,
        backend="transformers-local",
        allow_openai_video_input=False,
        media_role="full",
    )
    if generator_videos or not generator_images:
        raise ValueError(
            f"{packet.get('evidence_id')}: generator must receive retained images only"
        )
    if reward_images or len(reward_videos) != 2:
        raise ValueError(
            f"{packet.get('evidence_id')}: full-video reward/judge media must contain two videos"
        )
    return generator_images, reward_videos


def _call_with_retries(
    operation: Any,
    *,
    retries: int,
    label: str,
) -> tuple[Any, list[dict[str, Any]]]:
    failures = []
    for retry_index in range(retries + 1):
        try:
            return operation(), failures
        except Exception as exc:
            failures.append(
                {
                    "retry_index": retry_index,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            if retry_index >= retries:
                raise RuntimeError(
                    f"{label} failed after {retries + 1} infrastructure attempts"
                ) from exc
    raise AssertionError("unreachable")


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=path.name,
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _packet_record_path(packet_dir: Path, evidence_id: str) -> Path:
    digest = _sha256_text(evidence_id)[:20]
    return packet_dir / f"packet_{digest}.json"


def _packet_group_dir(
    output_dir: Path,
    *,
    packet_index: int,
    total_packet_count: int,
    packets_per_group: int,
) -> Path:
    start_index = (packet_index // packets_per_group) * packets_per_group
    end_index = min(start_index + packets_per_group, total_packet_count)
    width = max(3, len(str(total_packet_count)))
    name = f"packets_{start_index + 1:0{width}d}_{end_index:0{width}d}"
    return output_dir / "packet_groups" / name


def _packet_group_specs(
    selected_packets: list[dict[str, Any]],
    *,
    output_dir: Path,
    packets_per_group: int,
) -> list[dict[str, Any]]:
    specs = []
    total = len(selected_packets)
    for start_index in range(0, total, packets_per_group):
        end_index = min(start_index + packets_per_group, total)
        specs.append(
            {
                "group_index": len(specs),
                "start_index": start_index,
                "end_index": end_index,
                "packets": selected_packets[start_index:end_index],
                "group_dir": _packet_group_dir(
                    output_dir,
                    packet_index=start_index,
                    total_packet_count=total,
                    packets_per_group=packets_per_group,
                ),
            }
        )
    return specs


def _normalized_duplicate_key(raw_qa: dict[str, Any]) -> str:
    question = " ".join(str(raw_qa.get("question") or "").lower().split())
    options = [" ".join(str(value).lower().split()) for value in raw_qa.get("options") or []]
    return _sha256_text(_canonical_json({"question": question, "options": options}))


def _model_revision_metadata(runner: Any) -> dict[str, Any]:
    model = getattr(runner, "model", None)
    processor = getattr(runner, "processor", None)
    model_config = getattr(model, "config", None)
    processor_config = getattr(processor, "config", None)
    return {
        "model_id": str(getattr(runner, "model_id", "")),
        "model_revision": getattr(model_config, "_commit_hash", None),
        "processor_revision": getattr(processor_config, "_commit_hash", None),
    }


def _git_commit() -> str | None:
    try:
        root = Path(__file__).resolve().parent
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def _compact_candidate(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": row["schema_version"],
        "candidate_id": row["candidate_id"],
        "detail_record_id": row["candidate_id"],
        "evidence_id": row["evidence_id"],
        "packet_index": row["packet_index"],
        "candidate_index": row["candidate_index"],
        "loop_index": row["loop_index"],
        "attempt_in_loop": row["attempt_in_loop"],
        "raw_call_index": row["raw_call_index"],
        "depth": row["depth"],
        "generation_id": row["generation_id"],
        "parent_generation_id": row["parent_generation_id"],
        "parent_candidate_id": row["parent_candidate_id"],
        "generation_type": row["generation_type"],
        "replaces_malformed_generation_ids": row[
            "replaces_malformed_generation_ids"
        ],
        "generation_seed": row["generation_seed"],
        "generator_model_id": row["generator_model_id"],
        "generator_decode": row["generator_decode"],
        "prompt_sha256": row["prompt_sha256"],
        "automatic_outcome": row["automatic_outcome"],
        "automatic_reason": row["automatic_reason"],
        "answerability_class": row["answerability_class"],
        "exact_duplicate_of": row["exact_duplicate_of"],
        "reward_model_media": row["reward_model_media"],
        "raw_qa": row["raw_qa"],
    }


def _collect_one_packet(
    *,
    packet: dict[str, Any],
    packet_index: int,
    runner: Any,
    judge_runner: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    evidence_id = str(packet.get("evidence_id") or "")
    if not evidence_id:
        raise ValueError(f"selected packet index {packet_index} has no evidence_id")
    generator_images, full_videos = _existing_media(packet)
    decode_config = generator_decode_config(
        generator_decode_mode="sampling",
        generator_temperature=float(config["temperature"]),
        generator_top_p=float(config["top_p"]),
        generator_top_k=int(config["top_k"]),
    )
    valid_candidates: list[dict[str, Any]] = []
    malformed_outputs: list[dict[str, Any]] = []
    duplicate_by_key: dict[str, str] = {}
    raw_generation_count = 0
    loop_index = 0
    infrastructure_failures: list[dict[str, Any]] = []
    started = time.time()

    while (
        len(valid_candidates) < int(config["candidates_per_packet"])
        and raw_generation_count < int(config["max_raw_calls_per_packet"])
        and len(malformed_outputs) <= int(config["max_malformed_per_packet"])
    ):
        loop_index += 1
        valid_attempt_count = 0
        judge_feedback: str | None = None
        previous_valid_generation: str | None = None
        previous_valid_generation_id: str | None = None
        previous_candidate_id: str | None = None
        format_repair_feedback: str | None = None
        format_repair_generation: str | None = None
        format_repair_generation_id: str | None = None
        malformed_generation_ids_since_valid: list[str] = []

        while valid_attempt_count < int(config["max_attempts_per_loop"]):
            if len(valid_candidates) >= int(config["candidates_per_packet"]):
                break
            if raw_generation_count >= int(config["max_raw_calls_per_packet"]):
                break
            raw_generation_count += 1
            raw_call_index = raw_generation_count
            next_valid_attempt = valid_attempt_count + 1
            previous_generation_in = previous_valid_generation
            parent_generation_id = previous_valid_generation_id
            parent_candidate_id = previous_candidate_id
            format_repair_generation_in = format_repair_generation
            format_repair_generation_id_in = format_repair_generation_id
            format_repair_feedback_in = format_repair_feedback
            prompt_feedback_parts = []
            if judge_feedback:
                prompt_feedback_parts.append("Judge feedback: " + judge_feedback)
            if format_repair_feedback_in:
                prompt_feedback_parts.append(
                    "Raw structural validation errors: " + format_repair_feedback_in
                )
            prompt_feedback = "\n".join(prompt_feedback_parts) or None
            # Only a structurally valid, judged candidate can be the semantic
            # revision target. Malformed text is retained separately for audit,
            # while its validation errors provide format-repair feedback.
            prompt_previous_generation = previous_generation_in
            generation_id = f"{evidence_id}::loop_{loop_index:02d}::raw_{raw_call_index:02d}"
            seed = stable_generation_seed(
                run_seed=int(config["run_seed"]),
                evidence_id=evidence_id,
                loop_index=loop_index,
                attempt_in_loop=next_valid_attempt,
                generation_index=raw_call_index,
            )
            effective_seeds = set_generation_seed(seed)
            prompt = build_video_generation_prompt(
                packet,
                "neutral",
                feedback=prompt_feedback,
                generation_mode="baseline",
                previous_generation=prompt_previous_generation,
            )
            prompt_sha256 = _sha256_text(prompt)
            generation_started = time.time()

            def generate() -> str:
                set_generation_seed(seed)
                return runner.generate(
                    prompt,
                    image_paths=generator_images,
                    video_paths=[],
                    decoding_mode="sampling",
                    temperature=float(config["temperature"]),
                    top_p=float(config["top_p"]),
                    top_k=int(config["top_k"]),
                )

            raw_generation, generation_failures = _call_with_retries(
                generate,
                retries=int(config["generation_retries"]),
                label=f"generation {generation_id}",
            )
            infrastructure_failures.extend(
                {"stage": "generation", "generation_id": generation_id, **row}
                for row in generation_failures
            )
            generation_seconds = time.time() - generation_started

            try:
                raw_qa = extract_json_object(raw_generation)
            except Exception as exc:
                raw_errors = [f"{type(exc).__name__}: {exc}"]
                raw_qa = None
            else:
                raw_errors = validate_raw_generated_qa(raw_qa)

            if raw_errors:
                malformed = {
                    "schema_version": REWARD_COLLECTION_SCHEMA_VERSION,
                    "evidence_id": evidence_id,
                    "packet_index": packet_index,
                    "generation_id": generation_id,
                    "loop_index": loop_index,
                    "raw_call_index": raw_call_index,
                    "valid_attempts_completed_in_loop": valid_attempt_count,
                    "next_valid_attempt_in_loop": next_valid_attempt,
                    "consumes_valid_attempt": False,
                    "attempt_in_loop": None,
                    "depth": None,
                    "generation_type": "malformed_raw_call",
                    "parent_generation_id": None,
                    "parent_candidate_id": None,
                    "generation_seed": seed,
                    "effective_seeds": effective_seeds,
                    "generator_decode": decode_config,
                    "prompt": prompt,
                    "prompt_sha256": prompt_sha256,
                    "raw_output": raw_generation,
                    "raw_parsed_qa": raw_qa,
                    "raw_structure_errors": raw_errors,
                    "generation_seconds": generation_seconds,
                    "infrastructure_retries": generation_failures,
                    "judge_feedback_source_candidate_id": parent_candidate_id,
                    "format_repair_source_generation_id": (
                        format_repair_generation_id_in
                    ),
                }
                malformed_outputs.append(malformed)
                malformed_generation_ids_since_valid.append(generation_id)
                format_repair_feedback = "; ".join(raw_errors)
                format_repair_generation = str(raw_generation)
                format_repair_generation_id = generation_id
                if len(malformed_outputs) > int(config["max_malformed_per_packet"]):
                    break
                continue

            valid_attempt_count += 1
            attempt_in_loop = valid_attempt_count
            if attempt_in_loop == 1 and parent_candidate_id is not None:
                raise ValueError("independent root unexpectedly has a parent candidate")
            if attempt_in_loop > 1 and parent_candidate_id is None:
                raise ValueError("feedback refinement is missing a valid parent candidate")
            candidate_index = len(valid_candidates) + 1
            candidate_id = f"{evidence_id}::candidate_{candidate_index:02d}"
            attempt_trace: dict[str, Any] = {
                "evidence_id": evidence_id,
                "question_type": "neutral",
                "generation_mode": "baseline",
                "loop_index": loop_index,
                "attempt_in_loop": attempt_in_loop,
                "raw_call_index": raw_call_index,
                "candidate_index": candidate_index,
                "feedback_in": prompt_feedback,
                "judge_feedback_in": judge_feedback,
                "format_repair_feedback_in": format_repair_feedback_in,
                "previous_generation_in": previous_generation_in,
                "format_repair_previous_generation_in": (
                    format_repair_generation_in
                ),
                "replaces_malformed_generation_ids": list(
                    malformed_generation_ids_since_valid
                ),
                "media": {
                    "image_paths": generator_images,
                    "video_paths": [],
                    "media_role": "generator",
                    "reward_model_video_paths": full_videos,
                    "reward_model_media_role": "full",
                },
                "generation": {"prompt": prompt, "raw_output": raw_generation},
                "generator_decode": decode_config,
                "judge": {},
                "answerability": {},
                "result": {},
            }
            normalized_qa = _normalized_candidate_qa(
                deepcopy(raw_qa),
                packet=packet,
                packet_index=packet_index,
                question_type="neutral",
                generation_mode="baseline",
                attempt=attempt_in_loop,
                model_id=str(getattr(runner, "model_id", config["model_id"])),
                decode_config=decode_config,
                attempt_trace=attempt_trace,
            )
            normalized_qa["qa_id"] = (
                f"QA_{packet_index + 1:04d}_{evidence_id}_"
                f"L{loop_index:02d}_A{attempt_in_loop:02d}_C{candidate_index:02d}"
            )
            schema_errors = qa_formality_errors(
                normalized_qa,
                validate_qa_item(normalized_qa),
                participant_names=formality_participant_names(packet, normalized_qa),
            )
            judge_prompt_rows: list[dict[str, Any]] = []
            judge_started = time.time()

            def judge_candidate() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
                return run_parallel_review_judges(
                    qa_item=normalized_qa,
                    packet=packet,
                    schema_errors=schema_errors,
                    runner=judge_runner,
                    qa_formality_runner=judge_runner,
                    media_backend="transformers-local",
                    allow_openai_video_input=False,
                    prompt_rows=judge_prompt_rows,
                    full_image_paths=[],
                    full_video_paths=full_videos,
                    attempt=attempt_in_loop,
                    judge_media_role="full",
                    include_generator_rationale=False,
                    pass_fail_only=True,
                    quality_quota_counts=None,
                    record_decision_entropy=False,
                )

            (judge, answerability, judge_trace), judge_failures = _call_with_retries(
                judge_candidate,
                retries=int(config["judge_retries"]),
                label=f"judge {candidate_id}",
            )
            infrastructure_failures.extend(
                {"stage": "judge", "candidate_id": candidate_id, **row}
                for row in judge_failures
            )
            judge_seconds = time.time() - judge_started
            attempt_trace["judge"] = judge_trace
            attempt_trace["answerability"] = answerability
            passed = judge.get("gate", {}).get("passed") is True
            reason = str(
                judge.get("feedback_to_generator")
                or judge.get("gate", {}).get("reason")
                or ("passed all gates" if passed else "Judger rejected the question.")
            )
            if passed:
                normalized_qa["review"] = build_review_from_gates(
                    judge=judge,
                    answerability=answerability,
                    schema_errors=[],
                    accepted=True,
                    final_reason="passed all gates",
                )
                strict_errors = validate_qa_item(normalized_qa, strict_review=True)
                if strict_errors:
                    passed = False
                    reason = "Strict validation errors: " + "; ".join(strict_errors)
                    normalized_qa["review"] = build_review_from_gates(
                        judge=judge,
                        answerability=answerability,
                        schema_errors=strict_errors,
                        accepted=False,
                        rejection_stage="schema",
                        final_reason=reason,
                    )
            else:
                normalized_qa["review"] = build_review_from_gates(
                    judge=judge,
                    answerability=answerability,
                    schema_errors=schema_errors,
                    accepted=False,
                    rejection_stage="judger",
                    final_reason=reason,
                )
            attempt_trace["result"] = {"accepted": passed, "reason": reason}
            normalized_qa["generation_trace"] = [attempt_trace]
            duplicate_key = _normalized_duplicate_key(raw_qa)
            duplicate_of = duplicate_by_key.get(duplicate_key)
            duplicate_by_key.setdefault(duplicate_key, candidate_id)
            candidate = {
                "schema_version": REWARD_COLLECTION_SCHEMA_VERSION,
                "candidate_id": candidate_id,
                "evidence_id": evidence_id,
                "packet_index": packet_index,
                "candidate_index": candidate_index,
                "loop_index": loop_index,
                "attempt_in_loop": attempt_in_loop,
                "raw_call_index": raw_call_index,
                "depth": attempt_in_loop,
                "generation_id": generation_id,
                "parent_generation_id": parent_generation_id,
                "parent_candidate_id": parent_candidate_id,
                "generation_type": (
                    "independent_root"
                    if parent_candidate_id is None
                    else "feedback_refinement"
                ),
                "replaces_malformed_generation_ids": list(
                    malformed_generation_ids_since_valid
                ),
                "format_repair_feedback_in": format_repair_feedback_in,
                "generation_seed": seed,
                "effective_seeds": effective_seeds,
                "generator_model_id": str(getattr(runner, "model_id", config["model_id"])),
                "generator_decode": decode_config,
                "generator_media": {
                    "role": "retained_cluster_member_frames",
                    "image_paths": generator_images,
                    "video_paths": [],
                },
                "reward_model_media": {
                    "role": "full_original_videos",
                    "video_paths": full_videos,
                },
                "prompt": prompt,
                "prompt_sha256": prompt_sha256,
                "raw_output": raw_generation,
                "raw_qa": raw_qa,
                "raw_structure_errors": [],
                "normalized_qa": normalized_qa,
                "pre_judge_schema_errors": schema_errors,
                "judge": judge,
                "answerability": answerability,
                "answerability_class": answerability_class(normalized_qa, answerability),
                "judge_trace": judge_trace,
                "judge_prompt_rows": judge_prompt_rows,
                "automatic_outcome": "PASS" if passed else "FAIL",
                "automatic_reason": reason,
                "exact_duplicate_of": duplicate_of,
                "generation_seconds": generation_seconds,
                "judge_seconds": judge_seconds,
                "generation_infrastructure_retries": generation_failures,
                "judge_infrastructure_retries": judge_failures,
            }
            valid_candidates.append(candidate)
            previous_valid_generation = str(raw_generation)
            previous_valid_generation_id = generation_id
            previous_candidate_id = candidate_id
            judge_feedback = None if passed else reason
            format_repair_feedback = None
            format_repair_generation = None
            format_repair_generation_id = None
            malformed_generation_ids_since_valid = []
            if passed:
                break

        if len(malformed_outputs) > int(config["max_malformed_per_packet"]):
            break

    retained = len(valid_candidates) == int(config["candidates_per_packet"])
    if retained:
        failure_reason = None
    elif len(malformed_outputs) > int(config["max_malformed_per_packet"]):
        failure_reason = "maximum malformed-output count exceeded"
    else:
        failure_reason = "maximum raw generation-call count reached before candidate quota"
    score_counts = {"1": 0, "2": 0, "3": 0}
    for candidate in valid_candidates:
        score_counts[str(candidate["answerability_class"]["score"])] += 1
    summary = {
        "evidence_id": evidence_id,
        "packet_index": packet_index,
        "day": packet.get("day"),
        "time_token": packet.get("time_token"),
        "clock_seconds": packet_clock_seconds(packet),
        "retained": retained,
        "failure_reason": failure_reason,
        "valid_candidate_count": len(valid_candidates),
        "candidate_quota": int(config["candidates_per_packet"]),
        "raw_generation_count": raw_generation_count,
        "malformed_output_count": len(malformed_outputs),
        "loop_count": loop_index,
        "independent_root_count": sum(
            candidate["generation_type"] == "independent_root"
            for candidate in valid_candidates
        ),
        "feedback_refinement_count": sum(
            candidate["generation_type"] == "feedback_refinement"
            for candidate in valid_candidates
        ),
        "automatic_pass_count": sum(
            candidate["automatic_outcome"] == "PASS" for candidate in valid_candidates
        ),
        "automatic_fail_count": sum(
            candidate["automatic_outcome"] == "FAIL" for candidate in valid_candidates
        ),
        "answerability_score_counts": score_counts,
        "exact_duplicate_count": sum(
            candidate["exact_duplicate_of"] is not None for candidate in valid_candidates
        ),
        "maximum_depth": max(
            (int(candidate["depth"]) for candidate in valid_candidates),
            default=0,
        ),
        "elapsed_seconds": time.time() - started,
    }
    return {
        "schema_version": REWARD_COLLECTION_SCHEMA_VERSION,
        "config_sha256": config["config_sha256"],
        "evidence_id": evidence_id,
        "retained": retained,
        "failure_reason": failure_reason,
        "candidates": valid_candidates,
        "malformed_outputs": malformed_outputs,
        "infrastructure_failures": infrastructure_failures,
        "summary": summary,
    }


def _consolidate_group(
    *,
    group_spec: dict[str, Any],
    output_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    group_dir = Path(group_spec["group_dir"])
    packet_dir = group_dir / "packet_records"
    packets = list(group_spec["packets"])
    start_index = int(group_spec["start_index"])
    group_dir.mkdir(parents=True, exist_ok=True)
    compact_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    retained_details: list[dict[str, Any]] = []
    malformed_rows: list[dict[str, Any]] = []
    discarded_records: list[dict[str, Any]] = []
    for local_index, packet in enumerate(packets):
        packet_index = start_index + local_index
        evidence_id = str(packet.get("evidence_id") or "")
        path = _packet_record_path(packet_dir, evidence_id)
        if not path.exists():
            raise ValueError(f"missing atomic packet result for {evidence_id}: {path}")
        record = _read_json(path)
        if record.get("config_sha256") != config["config_sha256"]:
            raise ValueError(f"packet result config mismatch for {evidence_id}")
        if int(record.get("summary", {}).get("packet_index", -1)) != packet_index:
            raise ValueError(f"packet result index mismatch for {evidence_id}")
        summaries.append(record["summary"])
        malformed_rows.extend(record.get("malformed_outputs") or [])
        if record.get("retained") is True:
            candidates = list(record.get("candidates") or [])
            if len(candidates) != int(config["candidates_per_packet"]):
                raise ValueError(f"retained packet {evidence_id} does not have the candidate quota")
            observed_indices = [int(row.get("candidate_index") or 0) for row in candidates]
            if observed_indices != list(range(1, int(config["candidates_per_packet"]) + 1)):
                raise ValueError(f"candidate indices are not contiguous for {evidence_id}")
            for candidate in candidates:
                parent_id = candidate.get("parent_candidate_id")
                generation_type = candidate.get("generation_type")
                if generation_type == "independent_root" and parent_id is not None:
                    raise ValueError(
                        f"independent root has a parent: {candidate['candidate_id']}"
                    )
                if generation_type == "feedback_refinement" and not parent_id:
                    raise ValueError(
                        f"feedback refinement has no parent: {candidate['candidate_id']}"
                    )
                if parent_id and not any(
                    row.get("candidate_id") == parent_id for row in candidates
                ):
                    raise ValueError(f"candidate parent is missing for {candidate['candidate_id']}")
                compact_rows.append(_compact_candidate(candidate))
                retained_details.append(candidate)
        else:
            discarded_records.append(record)

    write_jsonl(group_dir / "evidence_manifest.jsonl", packets)
    write_jsonl(group_dir / "candidates.jsonl", compact_rows)
    write_jsonl(group_dir / "packet_summaries.jsonl", summaries)
    write_jsonl(group_dir / "candidate_details.jsonl", retained_details)
    write_jsonl(group_dir / "malformed_outputs.jsonl", malformed_rows)
    write_jsonl(group_dir / "discarded_packets.jsonl", discarded_records)
    retained_packet_count = sum(row["retained"] is True for row in summaries)
    expected_candidate_count = len(packets) * int(config["candidates_per_packet"])
    relative_dir = str(group_dir.relative_to(output_dir))
    group_summary = {
        "schema_version": REWARD_COLLECTION_SCHEMA_VERSION,
        "group_index": int(group_spec["group_index"]) + 1,
        "group_name": group_dir.name,
        "relative_dir": relative_dir,
        "packet_start_number": start_index + 1,
        "packet_end_number": int(group_spec["end_index"]),
        "selected_packet_count": len(packets),
        "retained_packet_count": retained_packet_count,
        "discarded_packet_count": len(packets) - retained_packet_count,
        "candidate_count": len(compact_rows),
        "expected_candidate_count": expected_candidate_count,
        "quota_complete": len(compact_rows) == expected_candidate_count,
        "malformed_output_count": len(malformed_rows),
        "automatic_pass_count": sum(row["automatic_outcome"] == "PASS" for row in compact_rows),
        "automatic_fail_count": sum(row["automatic_outcome"] == "FAIL" for row in compact_rows),
        "answerability_score_counts": {
            str(score): sum(
                row["answerability_class"]["score"] == score for row in compact_rows
            )
            for score in (1, 2, 3)
        },
        "compact_candidate_index": str(group_dir / "candidates.jsonl"),
        "full_candidate_details": str(group_dir / "candidate_details.jsonl"),
        "detail_storage": "plain JSONL; one full record per candidate",
        "config_sha256": config["config_sha256"],
    }
    write_json(group_dir / "group_summary.json", group_summary)
    return group_summary


def _consolidate(
    *,
    selected_packets: list[dict[str, Any]],
    output_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    group_specs = _packet_group_specs(
        selected_packets,
        output_dir=output_dir,
        packets_per_group=int(config["packets_per_output_group"]),
    )
    group_summaries = [
        _consolidate_group(
            group_spec=group_spec,
            output_dir=output_dir,
            config=config,
        )
        for group_spec in group_specs
    ]
    write_jsonl(output_dir / "packet_groups.jsonl", group_summaries)
    candidate_count = sum(int(row["candidate_count"]) for row in group_summaries)
    expected_candidate_count = len(selected_packets) * int(
        config["candidates_per_packet"]
    )
    collection_summary = {
        "schema_version": REWARD_COLLECTION_SCHEMA_VERSION,
        "selected_packet_count": len(selected_packets),
        "retained_packet_count": sum(
            int(row["retained_packet_count"]) for row in group_summaries
        ),
        "discarded_packet_count": sum(
            int(row["discarded_packet_count"]) for row in group_summaries
        ),
        "candidate_count": candidate_count,
        "expected_candidate_count": expected_candidate_count,
        "quota_complete": candidate_count == expected_candidate_count,
        "packets_per_output_group": int(config["packets_per_output_group"]),
        "packet_group_count": len(group_summaries),
        "packet_group_index": str(output_dir / "packet_groups.jsonl"),
        "malformed_output_count": sum(
            int(row["malformed_output_count"]) for row in group_summaries
        ),
        "automatic_pass_count": sum(
            int(row["automatic_pass_count"]) for row in group_summaries
        ),
        "automatic_fail_count": sum(
            int(row["automatic_fail_count"]) for row in group_summaries
        ),
        "answerability_score_counts": {
            str(score): sum(
                int(row["answerability_score_counts"][str(score)])
                for row in group_summaries
            )
            for score in (1, 2, 3)
        },
        "output_layout": (
            "packet_groups/packets_NNN_NNN contains at most 25 packets and "
            "six candidates per packet"
        ),
        "config_sha256": config["config_sha256"],
    }
    write_json(output_dir / "collection_summary.json", collection_summary)
    if not collection_summary["quota_complete"]:
        raise ValueError(
            "collection is incomplete: expected "
            f"{expected_candidate_count} candidates from {len(selected_packets)} packets, "
            f"found {candidate_count}; inspect discarded_packets.jsonl in the packet groups"
        )
    return collection_summary


def collect_reward_candidates(
    *,
    evidence_path: str | Path,
    output_dir: str | Path,
    model_id: str = DEFAULT_MODEL_ID,
    backend: str = "transformers-local",
    base_url: str = "http://127.0.0.1:8000/v1",
    dtype: str = "bfloat16",
    max_new_tokens: int = 4096,
    max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
    video_fps: float = DEFAULT_VIDEO_FPS,
    disable_thinking: bool = True,
    candidates_per_packet: int = DEFAULT_CANDIDATES_PER_PACKET,
    packets_per_output_group: int = DEFAULT_PACKETS_PER_OUTPUT_GROUP,
    max_attempts_per_loop: int = DEFAULT_MAX_ATTEMPTS_PER_LOOP,
    max_raw_calls_per_packet: int = DEFAULT_MAX_RAW_CALLS_PER_PACKET,
    max_malformed_per_packet: int = DEFAULT_MAX_MALFORMED_PER_PACKET,
    generation_retries: int = DEFAULT_GENERATION_RETRIES,
    judge_retries: int = DEFAULT_JUDGE_RETRIES,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    run_seed: int = 20260804,
    days: tuple[str, ...] | None = None,
    start_clock: str = "00:00:00",
    end_clock: str = "24:00:00",
    max_packets: int | None = None,
    expected_packet_count: int | None = None,
    resume: bool = False,
    allow_cpu: bool = False,
) -> dict[str, Any]:
    """Collect, atomically checkpoint, and consolidate reward candidates."""

    if backend != "transformers-local":
        raise ValueError(
            "reward collection currently requires one resident transformers-local model"
        )
    if candidates_per_packet <= 0 or max_attempts_per_loop <= 0:
        raise ValueError("candidate quota and attempts per loop must be positive")
    if packets_per_output_group <= 0:
        raise ValueError("packets_per_output_group must be positive")
    if max_raw_calls_per_packet < candidates_per_packet:
        raise ValueError("max_raw_calls_per_packet cannot be smaller than the candidate quota")
    if max_malformed_per_packet < 0 or generation_retries < 0 or judge_retries < 0:
        raise ValueError("malformed and infrastructure retry limits must be non-negative")
    if temperature <= 0 or not 0 < top_p <= 1 or top_k <= 0:
        raise ValueError("sampling parameters are out of range")

    evidence_path = Path(evidence_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start_seconds = parse_clock_seconds(start_clock)
    end_seconds = parse_clock_seconds(end_clock, allow_24h=True)
    selected_packets = filter_evidence_packets(
        iter_jsonl(evidence_path),
        days=days,
        start_clock_seconds=start_seconds,
        end_clock_seconds=end_seconds,
        max_packets=max_packets,
    )
    if not selected_packets:
        raise ValueError("no evidence packets matched the requested day/time window")
    if expected_packet_count is not None and len(selected_packets) != expected_packet_count:
        raise ValueError(
            f"expected {expected_packet_count} selected packets, found {len(selected_packets)}"
        )
    evidence_ids = [str(packet.get("evidence_id") or "") for packet in selected_packets]
    if any(not value for value in evidence_ids) or len(evidence_ids) != len(set(evidence_ids)):
        raise ValueError("selected evidence IDs must be non-empty and unique")

    config_without_hash = {
        "schema_version": REWARD_COLLECTION_SCHEMA_VERSION,
        "evidence_path": str(evidence_path.resolve()),
        "evidence_sha256": file_sha256(evidence_path),
        "selected_evidence_ids": evidence_ids,
        "model_id": model_id,
        "backend": backend,
        "dtype": dtype,
        "max_new_tokens": max_new_tokens,
        "max_image_pixels": max_image_pixels,
        "video_fps": video_fps,
        "disable_thinking": disable_thinking,
        "candidates_per_packet": candidates_per_packet,
        "packets_per_output_group": packets_per_output_group,
        "max_attempts_per_loop": max_attempts_per_loop,
        "max_raw_calls_per_packet": max_raw_calls_per_packet,
        "max_malformed_per_packet": max_malformed_per_packet,
        "generation_retries": generation_retries,
        "judge_retries": judge_retries,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "run_seed": run_seed,
        "days": list(days or ()),
        "start_clock": start_clock,
        "end_clock": end_clock,
        "time_window_semantics": "daily start-inclusive/end-exclusive",
        "generator_media": "retained CLIP-cluster member frames",
        "judge_media": "full original videos",
        "reward_model_media": "full original videos",
        "answerability_policy": (
            "unchanged production gate plus auxiliary deterministic 3/2/1 class"
        ),
        "question_types": ["neutral"],
        "generation_mode": "baseline",
        "implementation_files_sha256": {
            "collector": file_sha256(Path(__file__).resolve()),
            "prompts": source_file_sha256(build_video_generation_prompt),
            "normalization": source_file_sha256(_normalized_candidate_qa),
            "judging_and_answerability": source_file_sha256(
                run_parallel_review_judges
            ),
            "model_runner": source_file_sha256(make_runner),
        },
        "git_commit": _git_commit(),
    }
    config = {
        **config_without_hash,
        "config_sha256": _sha256_text(_canonical_json(config_without_hash)),
    }

    existing_manifest = output_dir / "run_manifest.json"
    if existing_manifest.exists():
        existing = json.loads(existing_manifest.read_text(encoding="utf-8"))
        if not resume:
            raise ValueError(
                f"output directory already contains a run manifest; use --resume: {output_dir}"
            )
        if existing.get("config_sha256") != config["config_sha256"]:
            raise ValueError("resume configuration does not match the existing run")
    else:
        manifest = {
            **config,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "selected_packet_count": len(selected_packets),
            "storage": {
                "packet_group_index": "packet_groups.jsonl",
                "compact_indexes": "packet_groups/packets_NNN_NNN/candidates.jsonl",
                "full_details": "packet_groups/packets_NNN_NNN/candidate_details.jsonl",
                "atomic_packet_records": (
                    "packet_groups/packets_NNN_NNN/packet_records/*.json"
                ),
                "compression": "none",
            },
        }
        write_json(existing_manifest, manifest)
    write_jsonl(output_dir / "evidence_manifest.jsonl", selected_packets)
    group_specs = _packet_group_specs(
        selected_packets,
        output_dir=output_dir,
        packets_per_group=packets_per_output_group,
    )

    pending_packets: list[tuple[int, dict[str, Any]]] = []
    for packet_index, packet in enumerate(selected_packets):
        evidence_id = str(packet["evidence_id"])
        group_dir = _packet_group_dir(
            output_dir,
            packet_index=packet_index,
            total_packet_count=len(selected_packets),
            packets_per_group=packets_per_output_group,
        )
        path = _packet_record_path(group_dir / "packet_records", evidence_id)
        if not path.exists():
            pending_packets.append((packet_index, packet))
            continue
        if not resume:
            raise ValueError(f"packet record already exists without --resume: {path}")
        record = _read_json(path)
        if record.get("evidence_id") != evidence_id:
            raise ValueError(f"packet record evidence mismatch: {path}")
        if record.get("config_sha256") != config["config_sha256"]:
            raise ValueError(f"packet record configuration mismatch: {path}")
        print(f"reward_collection_resume_skip evidence_id={evidence_id}", flush=True)

    if not pending_packets:
        return _consolidate(
            selected_packets=selected_packets,
            output_dir=output_dir,
            config=config,
        )

    runner = make_runner(
        backend,
        model_id=model_id,
        base_url=base_url,
        max_new_tokens=max_new_tokens,
        max_image_pixels=max_image_pixels,
        dtype=dtype,
        allow_cpu=allow_cpu,
        allow_openai_video_input=False,
        disable_thinking=disable_thinking,
        video_fps=video_fps,
    )
    judge_runner = SerializedJudgeRunner(runner)
    model_metadata = _model_revision_metadata(runner)
    manifest = json.loads(existing_manifest.read_text(encoding="utf-8"))
    recorded_model = manifest.get("resolved_model")
    if isinstance(recorded_model, dict):
        mismatches = []
        for field in ("model_id", "model_revision", "processor_revision"):
            recorded = recorded_model.get(field)
            current = model_metadata.get(field)
            if recorded is not None and recorded != current:
                mismatches.append(f"{field}: recorded={recorded!r} current={current!r}")
        if mismatches:
            raise ValueError("resolved model changed during resume: " + "; ".join(mismatches))
    manifest["resolved_model"] = model_metadata
    write_json(existing_manifest, manifest)

    for packet_index, packet in pending_packets:
        evidence_id = str(packet["evidence_id"])
        group_spec = group_specs[packet_index // packets_per_output_group]
        group_dir = Path(group_spec["group_dir"])
        path = _packet_record_path(group_dir / "packet_records", evidence_id)
        print(
            "reward_collection_packet_start "
            f"packet={packet_index + 1}/{len(selected_packets)} evidence_id={evidence_id}",
            flush=True,
        )
        record = _collect_one_packet(
            packet=packet,
            packet_index=packet_index,
            runner=runner,
            judge_runner=judge_runner,
            config=config,
        )
        _atomic_write_json(path, record)
        print(
            "reward_collection_packet_done "
            f"evidence_id={evidence_id} retained={record['retained']} "
            f"valid={record['summary']['valid_candidate_count']} "
            f"raw_calls={record['summary']['raw_generation_count']}",
            flush=True,
        )
        if packet_index + 1 == int(group_spec["end_index"]):
            group_summary = _consolidate_group(
                group_spec=group_spec,
                output_dir=output_dir,
                config=config,
            )
            print(
                "reward_collection_group_done "
                f"group={group_summary['group_name']} "
                f"packets={group_summary['selected_packet_count']} "
                f"candidates={group_summary['candidate_count']}",
                flush=True,
            )

    return _consolidate(
        selected_packets=selected_packets,
        output_dir=output_dir,
        config=config,
    )


def _parse_days(value: str | None) -> tuple[str, ...] | None:
    if not value:
        return None
    days = tuple(part.strip().upper() for part in value.split(",") if part.strip())
    return days or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect fixed-six reward-model QA candidates with repeated judge loops"
    )
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--backend", default="transformers-local", choices=["transformers-local"])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--max-image-pixels", type=int, default=DEFAULT_MAX_IMAGE_PIXELS)
    parser.add_argument("--video-fps", type=float, default=DEFAULT_VIDEO_FPS)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--candidates-per-packet", type=int, default=DEFAULT_CANDIDATES_PER_PACKET)
    parser.add_argument(
        "--packets-per-output-group",
        type=int,
        default=DEFAULT_PACKETS_PER_OUTPUT_GROUP,
    )
    parser.add_argument(
        "--max-attempts-per-loop",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS_PER_LOOP,
    )
    parser.add_argument(
        "--max-raw-calls-per-packet",
        type=int,
        default=DEFAULT_MAX_RAW_CALLS_PER_PACKET,
    )
    parser.add_argument(
        "--max-malformed-per-packet",
        type=int,
        default=DEFAULT_MAX_MALFORMED_PER_PACKET,
    )
    parser.add_argument("--generation-retries", type=int, default=DEFAULT_GENERATION_RETRIES)
    parser.add_argument("--judge-retries", type=int, default=DEFAULT_JUDGE_RETRIES)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--run-seed", type=int, default=20260804)
    parser.add_argument("--days", help="Comma-separated DAY labels; omitted selects every day")
    parser.add_argument("--start-clock", default="00:00:00")
    parser.add_argument("--end-clock", default="24:00:00")
    parser.add_argument("--max-packets", type=int)
    parser.add_argument("--expected-packet-count", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = collect_reward_candidates(
        evidence_path=args.evidence,
        output_dir=args.output_dir,
        model_id=args.model_id,
        backend=args.backend,
        base_url=args.base_url,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        max_image_pixels=args.max_image_pixels,
        video_fps=args.video_fps,
        disable_thinking=not args.enable_thinking,
        candidates_per_packet=args.candidates_per_packet,
        packets_per_output_group=args.packets_per_output_group,
        max_attempts_per_loop=args.max_attempts_per_loop,
        max_raw_calls_per_packet=args.max_raw_calls_per_packet,
        max_malformed_per_packet=args.max_malformed_per_packet,
        generation_retries=args.generation_retries,
        judge_retries=args.judge_retries,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        run_seed=args.run_seed,
        days=_parse_days(args.days),
        start_clock=args.start_clock,
        end_clock=args.end_clock,
        max_packets=args.max_packets,
        expected_packet_count=args.expected_packet_count,
        resume=args.resume,
        allow_cpu=args.allow_cpu,
    )
    print(
        "reward_collection_complete "
        f"retained_packets={summary['retained_packet_count']} "
        f"discarded_packets={summary['discarded_packet_count']} "
        f"candidates={summary['candidate_count']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
