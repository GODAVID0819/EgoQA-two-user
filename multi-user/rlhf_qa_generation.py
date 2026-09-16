"""Spend six generation attempts across bounded QA loops for each RLHF packet.

The source-packet worker is the unit of pipeline concurrency.  Each worker
runs the existing generation/review/retry loop with at most three generations.
Failures stay with the same asker inside a loop.  Acceptance or three failed
passes ends that loop; a fresh loop rotates to the next asker and consumes the
remaining part of the six-attempt packet budget.  Shared semaphores bound
generation and review stages across source packets.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
import os
from pathlib import Path
import re
from threading import BoundedSemaphore
import tarfile
import time
from typing import Any, Iterable, Sequence
import uuid

# Keep this file runnable after being copied into the HPC package directory.
if __package__ in {None, ""}:
    import sys
    import types

    package_root = Path(__file__).resolve().parent
    package = types.ModuleType("egolife_two_user_qa")
    package.__path__ = [str(package_root)]
    sys.modules.setdefault("egolife_two_user_qa", package)
    __package__ = "egolife_two_user_qa"

from .io_utils import iter_jsonl, read_json, write_jsonl
from .rlhf_evidence_preprocessing import USER_COUNT, load_asker_view
from .video_qa_loop import (
    SIX_USER_JUDGE_MODE_LEGACY,
    compact_qa_for_checkpoint,
    compact_trace_payload,
    generate_video_qa_loop,
)


DEFAULT_MODEL_ID = "Qwen/Qwen3.8-27B"
DEFAULT_PACKETS_IN_FLIGHT = 6
DEFAULT_GENERATION_LANES = 4
DEFAULT_REVIEW_LANES = 5
DEFAULT_MAX_ATTEMPTS_PER_LOOP = 3
DEFAULT_ATTEMPTS_PER_PACKET = 6
DEFAULT_CHECKPOINT_PACKET_COUNT = 30


def _safe_component(value: Any) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("._")
    return cleaned or "unknown"


def asker_order_for_packet(source_packet_id: str) -> list[int]:
    """Return a reproducible packet-specific permutation of the six askers."""

    return sorted(
        range(USER_COUNT),
        key=lambda asker_index: hashlib.sha256(
            f"{source_packet_id}\0{asker_index}".encode("utf-8")
        ).digest(),
    )


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl_if_present(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    return list(iter_jsonl(path))


def _annotate_rows(
    rows: Iterable[dict[str, Any]],
    *,
    source_packet_id: str,
    asker_index: int,
    asker_user: str,
) -> list[dict[str, Any]]:
    annotated = []
    for row in rows:
        annotated.append(
            {
                **row,
                "rlhf_source_packet_id": source_packet_id,
                "rlhf_asker_index": asker_index,
                "rlhf_asker_user": asker_user,
            }
        )
    return annotated


def _compact_rejected_row(row: dict[str, Any]) -> dict[str, Any]:
    compact = compact_trace_payload(row)
    if not isinstance(compact, dict):
        return {}
    return compact


def select_complete_source_packets(
    dataset_root: str | Path,
    *,
    start_index: int = 0,
    packet_limit: int | None = None,
) -> list[dict[str, Any]]:
    """Return deterministic complete packet rows without mutating the dataset."""

    root = Path(dataset_root).resolve()
    packet_root = root / "packets"
    if not packet_root.is_dir():
        raise FileNotFoundError(f"RLHF packet directory is missing: {packet_root}")
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    if packet_limit is not None and packet_limit < 1:
        raise ValueError("packet_limit must be positive when provided")

    rows = []
    for packet_dir in sorted(packet_root.iterdir()):
        if not packet_dir.is_dir() or not (packet_dir / "COMPLETE").is_file():
            continue
        packet_path = packet_dir / "packet.json"
        if not packet_path.is_file():
            raise FileNotFoundError(f"complete packet has no packet.json: {packet_dir}")
        packet = read_json(packet_path)
        users = packet.get("users") or []
        if len(users) != USER_COUNT:
            raise ValueError(
                f"packet {packet_dir.name} has {len(users)} users; expected {USER_COUNT}"
            )
        rows.append(
            {
                "packet_id": str(packet.get("packet_id") or packet_dir.name),
                "day": str(packet.get("day") or ""),
                "time_token": str(packet.get("time_token") or ""),
                "users": [str(user.get("agent_dir") or "") for user in users],
            }
        )
    rows.sort(key=lambda row: (row["day"], row["time_token"], row["packet_id"]))
    selected = rows[start_index:]
    if packet_limit is not None:
        selected = selected[:packet_limit]
    if not selected:
        raise ValueError(
            f"no complete RLHF source packets selected from {root} at start_index={start_index}"
        )
    return selected


def _loop_paths(loop_dir: Path) -> dict[str, Path]:
    return {
        "evidence": loop_dir / "evidence.runtime.jsonl",
        "accepted": loop_dir / "accepted.jsonl",
        "rejected": loop_dir / "rejected.jsonl",
        "prompts": loop_dir / "prompts.jsonl",
        "intermediate": loop_dir / "intermediate.jsonl",
        "infrastructure": loop_dir / "infrastructure_skipped.jsonl",
        "complete": loop_dir / "LOOP_COMPLETE.json",
    }


def _validate_existing_marker(
    marker_path: Path,
    *,
    source_packet_id: str,
    loop_index: int,
) -> dict[str, Any]:
    marker = read_json(marker_path)
    if marker.get("source_packet_id") != source_packet_id:
        raise ValueError(f"loop marker packet mismatch: {marker_path}")
    if int(marker.get("loop_index", -1)) != loop_index:
        raise ValueError(f"loop marker index mismatch: {marker_path}")
    if marker.get("outcome") not in {"accepted", "rejected"}:
        raise ValueError(f"loop marker has invalid outcome: {marker_path}")
    attempts_consumed = int(marker.get("attempts_consumed", 0))
    if attempts_consumed < 1:
        raise ValueError(f"loop marker has invalid attempt count: {marker_path}")
    return marker


def _compact_finished_loop_files(
    paths: dict[str, Path],
    *,
    source_packet_id: str,
    asker_index: int,
    asker_user: str,
) -> dict[str, int]:
    accepted = _annotate_rows(
        (
            compact_qa_for_checkpoint(row)
            for row in _read_jsonl_if_present(paths["accepted"])
        ),
        source_packet_id=source_packet_id,
        asker_index=asker_index,
        asker_user=asker_user,
    )
    rejected = _annotate_rows(
        (
            _compact_rejected_row(row)
            for row in _read_jsonl_if_present(paths["rejected"])
        ),
        source_packet_id=source_packet_id,
        asker_index=asker_index,
        asker_user=asker_user,
    )
    intermediate = _annotate_rows(
        _read_jsonl_if_present(paths["intermediate"]),
        source_packet_id=source_packet_id,
        asker_index=asker_index,
        asker_user=asker_user,
    )
    prompts = _annotate_rows(
        _read_jsonl_if_present(paths["prompts"]),
        source_packet_id=source_packet_id,
        asker_index=asker_index,
        asker_user=asker_user,
    )
    infrastructure = _annotate_rows(
        _read_jsonl_if_present(paths["infrastructure"]),
        source_packet_id=source_packet_id,
        asker_index=asker_index,
        asker_user=asker_user,
    )
    write_jsonl(paths["accepted"], accepted)
    write_jsonl(paths["rejected"], rejected)
    write_jsonl(paths["intermediate"], intermediate)
    write_jsonl(paths["prompts"], prompts)
    write_jsonl(paths["infrastructure"], infrastructure)
    paths["evidence"].unlink(missing_ok=True)
    return {
        "accepted": len(accepted),
        "rejected": len(rejected),
        "intermediate": len(intermediate),
        "prompts": len(prompts),
        "infrastructure": len(infrastructure),
    }


def _aggregate_outputs(
    *,
    output_dir: Path,
    selected_packets: Sequence[dict[str, Any]],
    attempts_per_packet: int,
    artifact_dir: Path | None = None,
) -> dict[str, Any]:
    artifact_dir = artifact_dir or output_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    aggregate_names = {
        "accepted": "qa_mcq.jsonl",
        "rejected": "qa_mcq.rejected.jsonl",
        "intermediate": "qa_mcq.intermediate.jsonl",
        "prompts": "video_first_prompts.jsonl",
        "infrastructure": "qa_mcq.infrastructure_skipped.jsonl",
    }
    aggregate_rows = {name: [] for name in aggregate_names}
    outcomes = []
    completed_attempts_by_packet: dict[str, int] = {}
    completed_loops_by_packet: dict[str, int] = {}
    for packet in selected_packets:
        source_packet_id = str(packet["packet_id"])
        completed_attempts_by_packet[source_packet_id] = 0
        completed_loops_by_packet[source_packet_id] = 0
        packet_dir = output_dir / "loops" / _safe_component(source_packet_id)
        loop_dirs = sorted(packet_dir.glob("loop_*_asker_*"))
        for loop_index, loop_dir in enumerate(loop_dirs):
            paths = _loop_paths(loop_dir)
            if not paths["complete"].is_file():
                continue
            marker = _validate_existing_marker(
                paths["complete"],
                source_packet_id=source_packet_id,
                loop_index=loop_index,
            )
            outcomes.append(marker)
            completed_loops_by_packet[source_packet_id] += 1
            completed_attempts_by_packet[source_packet_id] += int(
                marker["attempts_consumed"]
            )
            for name in aggregate_names:
                aggregate_rows[name].extend(_read_jsonl_if_present(paths[name]))

    for name, filename in aggregate_names.items():
        write_jsonl(artifact_dir / filename, aggregate_rows[name])
    write_jsonl(artifact_dir / "loop_outcomes.jsonl", outcomes)
    planned_attempts = len(selected_packets) * attempts_per_packet
    completed_attempts = sum(completed_attempts_by_packet.values())
    summary = {
        "schema_version": "egolife_rlhf_qa_generation_v1",
        "source_packet_count": len(selected_packets),
        "attempts_per_source_packet": attempts_per_packet,
        "max_attempts_per_loop": DEFAULT_MAX_ATTEMPTS_PER_LOOP,
        "planned_generation_attempt_count": planned_attempts,
        "completed_generation_attempt_count": completed_attempts,
        "remaining_generation_attempt_count": planned_attempts - completed_attempts,
        "completed_question_loop_count": len(outcomes),
        "accepted_question_count": len(aggregate_rows["accepted"]),
        "exhausted_rejected_loop_count": len(aggregate_rows["rejected"]),
        "infrastructure_skipped_count": len(aggregate_rows["infrastructure"]),
        "completed_generation_attempts_by_source_packet": completed_attempts_by_packet,
        "completed_question_loops_by_source_packet": completed_loops_by_packet,
        "answerability_design": "concurrent_asker_only_and_all_six_legacy_sufficiency",
        "generator_media": "full_unpruned_asker_plus_asker_relative_pruned_providers",
        "judge_media": "full_unpruned_sampled_frames_for_all_six_users",
    }
    _atomic_write_json(artifact_dir / "run_summary.json", summary)
    return summary


def _ensure_checkpoint_archive(
    *,
    checkpoints_dir: Path,
    checkpoint_dir: Path,
) -> tuple[Path, str]:
    archive_path = checkpoints_dir / f"{checkpoint_dir.name}.tgz"
    checksum_path = archive_path.with_name(f"{archive_path.name}.sha256")
    if archive_path.is_file() and checksum_path.is_file():
        recorded = checksum_path.read_text(encoding="utf-8").strip().split()
        if recorded:
            return archive_path, recorded[0]

    temporary = checkpoints_dir / f".{archive_path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with tarfile.open(temporary, mode="w:gz") as archive:
            archive.add(checkpoint_dir, arcname=checkpoint_dir.name)
        os.replace(temporary, archive_path)
    finally:
        temporary.unlink(missing_ok=True)
    digest = _sha256_file(archive_path)
    _atomic_write_text(checksum_path, f"{digest}  {archive_path.name}\n")
    return archive_path, digest


def _publish_ready_checkpoints(
    *,
    output_dir: Path,
    selected_packets: Sequence[dict[str, Any]],
    attempts_per_packet: int,
    completed_attempts_by_packet: dict[str, int],
    checkpoint_packet_count: int,
    start_index: int,
) -> list[dict[str, Any]]:
    """Publish immutable labeling bundles for each fully completed packet cohort."""

    checkpoints_dir = output_dir / "checkpoints"
    published = []
    for cohort_offset in range(0, len(selected_packets), checkpoint_packet_count):
        cohort = selected_packets[
            cohort_offset : cohort_offset + checkpoint_packet_count
        ]
        packet_ids = [str(packet["packet_id"]) for packet in cohort]
        if any(
            completed_attempts_by_packet.get(packet_id, 0) != attempts_per_packet
            for packet_id in packet_ids
        ):
            continue

        first_packet_number = start_index + cohort_offset + 1
        last_packet_number = first_packet_number + len(cohort) - 1
        checkpoint_name = (
            f"packets_{first_packet_number:06d}_{last_packet_number:06d}"
        )
        checkpoint_dir = checkpoints_dir / checkpoint_name
        ready_path = checkpoint_dir / "CHECKPOINT_READY.json"
        manifest: dict[str, Any] | None = None
        if ready_path.is_file():
            candidate = read_json(ready_path)
            if candidate.get("source_packet_ids") != packet_ids:
                raise ValueError(
                    f"checkpoint packet selection mismatch: {ready_path}"
                )
            if int(candidate.get("attempts_per_source_packet", -1)) != attempts_per_packet:
                raise ValueError(f"checkpoint attempt budget mismatch: {ready_path}")
            manifest = candidate
        else:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_summary = _aggregate_outputs(
                output_dir=output_dir,
                artifact_dir=checkpoint_dir,
                selected_packets=cohort,
                attempts_per_packet=attempts_per_packet,
            )
            if checkpoint_summary["remaining_generation_attempt_count"]:
                raise RuntimeError(
                    f"refusing to publish incomplete checkpoint {checkpoint_name}"
                )
            accepted_rows = _read_jsonl_if_present(checkpoint_dir / "qa_mcq.jsonl")
            write_jsonl(checkpoint_dir / "labeling_queue.jsonl", accepted_rows)
            payload_names = (
                "labeling_queue.jsonl",
                "qa_mcq.jsonl",
                "qa_mcq.rejected.jsonl",
                "qa_mcq.intermediate.jsonl",
                "video_first_prompts.jsonl",
                "qa_mcq.infrastructure_skipped.jsonl",
                "loop_outcomes.jsonl",
                "run_summary.json",
            )
            payload_sha256 = {
                name: _sha256_file(checkpoint_dir / name) for name in payload_names
            }
            manifest = {
                "schema_version": "egolife_rlhf_labeling_checkpoint_v1",
                "checkpoint_name": checkpoint_name,
                "first_selected_packet_number": first_packet_number,
                "last_selected_packet_number": last_packet_number,
                "source_packet_count": len(cohort),
                "source_packet_ids": packet_ids,
                "attempts_per_source_packet": attempts_per_packet,
                "completed_generation_attempt_count": checkpoint_summary[
                    "completed_generation_attempt_count"
                ],
                "accepted_question_count": checkpoint_summary[
                    "accepted_question_count"
                ],
                "exhausted_rejected_loop_count": checkpoint_summary[
                    "exhausted_rejected_loop_count"
                ],
                "labeling_file": "labeling_queue.jsonl",
                "trajectory_file": "qa_mcq.intermediate.jsonl",
                "payload_sha256": payload_sha256,
                "answerability_design": checkpoint_summary["answerability_design"],
                "generator_media": checkpoint_summary["generator_media"],
                "judge_media": checkpoint_summary["judge_media"],
            }
            _atomic_write_json(ready_path, manifest)

        archive_path, archive_sha256 = _ensure_checkpoint_archive(
            checkpoints_dir=checkpoints_dir,
            checkpoint_dir=checkpoint_dir,
        )
        published.append(
            {
                **manifest,
                "archive_path": str(archive_path),
                "archive_sha256": archive_sha256,
            }
        )
    return published


def run_rlhf_packet_generation(
    *,
    dataset_root: str | Path,
    output_dir: str | Path,
    model_id: str = DEFAULT_MODEL_ID,
    base_url: str = "http://127.0.0.1:18000/v1",
    max_attempts: int = DEFAULT_MAX_ATTEMPTS_PER_LOOP,
    attempts_per_packet: int = DEFAULT_ATTEMPTS_PER_PACKET,
    max_new_tokens: int = 2048,
    max_image_pixels: int = 262144,
    start_index: int = 0,
    packet_limit: int | None = None,
    max_packets_in_flight: int = DEFAULT_PACKETS_IN_FLIGHT,
    max_generation_lanes: int = DEFAULT_GENERATION_LANES,
    max_review_lanes: int = DEFAULT_REVIEW_LANES,
    checkpoint_packet_count: int = DEFAULT_CHECKPOINT_PACKET_COUNT,
    generator_temperature: float = 0.7,
    generator_top_p: float = 0.9,
    generator_top_k: int | None = 40,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run or resume a fixed generation-attempt budget for each source packet."""

    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    if attempts_per_packet < 1:
        raise ValueError("attempts_per_packet must be positive")
    if checkpoint_packet_count < 1:
        raise ValueError("checkpoint_packet_count must be positive")
    if not 1 <= max_packets_in_flight <= 8:
        raise ValueError("max_packets_in_flight must be between 1 and 8")
    if not 1 <= max_generation_lanes <= min(6, max_packets_in_flight):
        raise ValueError(
            "max_generation_lanes must be between 1 and min(6, max_packets_in_flight)"
        )
    if not 1 <= max_review_lanes <= min(7, max_packets_in_flight):
        raise ValueError(
            "max_review_lanes must be between 1 and min(7, max_packets_in_flight)"
        )

    dataset_root = Path(dataset_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_packets = select_complete_source_packets(
        dataset_root,
        start_index=start_index,
        packet_limit=packet_limit,
    )
    generation_gate = BoundedSemaphore(max_generation_lanes)
    review_gate = BoundedSemaphore(max_review_lanes)
    started = time.time()
    print(
        "rlhf_qa_pipeline_start "
        f"source_packets={len(selected_packets)} "
        f"generation_attempts={len(selected_packets) * attempts_per_packet} "
        f"packets_in_flight={max_packets_in_flight} generation_lanes={max_generation_lanes} "
        f"review_lanes={max_review_lanes} max_attempts_per_loop={max_attempts} "
        f"checkpoint_packets={checkpoint_packet_count}",
        flush=True,
    )

    announced_checkpoints: set[str] = set()

    def refresh_outputs() -> tuple[dict[str, Any], list[dict[str, Any]]]:
        run_summary = _aggregate_outputs(
            output_dir=output_dir,
            selected_packets=selected_packets,
            attempts_per_packet=attempts_per_packet,
        )
        checkpoints = _publish_ready_checkpoints(
            output_dir=output_dir,
            selected_packets=selected_packets,
            attempts_per_packet=attempts_per_packet,
            completed_attempts_by_packet=run_summary[
                "completed_generation_attempts_by_source_packet"
            ],
            checkpoint_packet_count=checkpoint_packet_count,
            start_index=start_index,
        )
        for checkpoint in checkpoints:
            checkpoint_name = str(checkpoint["checkpoint_name"])
            if checkpoint_name in announced_checkpoints:
                continue
            announced_checkpoints.add(checkpoint_name)
            print(
                "rlhf_labeling_checkpoint_ready "
                f"checkpoint={checkpoint_name} "
                f"packets={checkpoint['source_packet_count']} "
                f"attempts={checkpoint['completed_generation_attempt_count']} "
                f"accepted={checkpoint['accepted_question_count']} "
                f"archive={checkpoint['archive_path']} "
                f"sha256={checkpoint['archive_sha256']}",
                flush=True,
            )
        return run_summary, checkpoints

    def run_source_packet(selection_index: int, packet_row: dict[str, Any]) -> dict[str, Any]:
        source_packet_id = str(packet_row["packet_id"])
        packet_started = time.time()
        packet_output_dir = output_dir / "loops" / _safe_component(source_packet_id)
        packet_output_dir.mkdir(parents=True, exist_ok=True)
        asker_order = asker_order_for_packet(source_packet_id)
        completed_loops = 0
        attempts_consumed = 0
        accepted_count = 0
        rejected_count = 0
        print(
            "rlhf_source_packet_start "
            f"selection_index={selection_index} packet_id={source_packet_id}",
            flush=True,
        )
        existing_loop_dirs: dict[int, Path] = {}
        for candidate in sorted(packet_output_dir.glob("loop_*_asker_*")):
            match = re.match(r"loop_([0-9]+)_asker_", candidate.name)
            if match is None:
                continue
            loop_index = int(match.group(1))
            if loop_index in existing_loop_dirs:
                raise ValueError(
                    f"duplicate loop index {loop_index} for packet {source_packet_id}"
                )
            existing_loop_dirs[loop_index] = candidate
        for loop_index in range(len(existing_loop_dirs)):
            loop_dir = existing_loop_dirs.get(loop_index)
            if loop_dir is None:
                raise ValueError(
                    f"non-contiguous loop history for packet {source_packet_id}"
                )
            paths = _loop_paths(loop_dir)
            if not paths["complete"].is_file():
                if loop_index != len(existing_loop_dirs) - 1:
                    raise ValueError(
                        f"incomplete non-final loop for packet {source_packet_id}: {loop_dir}"
                    )
                break
            marker = _validate_existing_marker(
                paths["complete"],
                source_packet_id=source_packet_id,
                loop_index=loop_index,
            )
            expected_asker_index = asker_order[loop_index % USER_COUNT]
            if int(marker.get("asker_index", -1)) != expected_asker_index:
                raise ValueError(
                    f"loop marker asker mismatch for packet {source_packet_id}: "
                    f"loop={loop_index} expected={expected_asker_index} "
                    f"actual={marker.get('asker_index')}"
                )
            completed_loops += 1
            attempts_consumed += int(marker["attempts_consumed"])
            accepted_count += int(marker["outcome"] == "accepted")
            rejected_count += int(marker["outcome"] == "rejected")
            print(
                "rlhf_question_loop_skip_complete "
                f"packet_id={source_packet_id} loop_index={loop_index} "
                f"asker={marker.get('asker_user')} outcome={marker['outcome']} "
                f"attempts_consumed={marker['attempts_consumed']}",
                flush=True,
            )
        if attempts_consumed > attempts_per_packet:
            raise ValueError(
                f"completed loop history exceeds the {attempts_per_packet}-attempt "
                f"budget for packet {source_packet_id}"
            )

        loop_index = completed_loops
        while attempts_consumed < attempts_per_packet:
            asker_index = asker_order[loop_index % USER_COUNT]
            view = load_asker_view(dataset_root, source_packet_id, asker_index)
            asker_user = str(view["asker_user"])
            loop_dir = (
                packet_output_dir
                / (
                    f"loop_{loop_index:02d}_asker_{asker_index:02d}_"
                    f"{_safe_component(view['asker_agent_dir'])}"
                )
            )
            loop_dir.mkdir(parents=True, exist_ok=True)
            paths = _loop_paths(loop_dir)
            loop_attempt_cap = min(
                max_attempts,
                attempts_per_packet - attempts_consumed,
            )
            write_jsonl(paths["evidence"], [view])
            global_loop_index = (
                (start_index + selection_index) * attempts_per_packet + loop_index
            )
            print(
                "rlhf_question_loop_start "
                f"packet_id={source_packet_id} loop_index={loop_index} "
                f"asker_index={asker_index} asker={asker_user} "
                f"attempt_budget_remaining={attempts_per_packet - attempts_consumed} "
                f"loop_attempt_cap={loop_attempt_cap} evidence_id={view['evidence_id']}",
                flush=True,
            )
            generate_video_qa_loop(
                evidence_path=paths["evidence"],
                output_path=paths["accepted"],
                prompts_path=paths["prompts"],
                rejected_path=paths["rejected"],
                intermediate_path=paths["intermediate"],
                infrastructure_skipped_path=paths["infrastructure"],
                backend="openai-compatible-local",
                model_id=model_id,
                base_url=base_url,
                target_count=1,
                max_attempts=loop_attempt_cap,
                max_new_tokens=max_new_tokens,
                max_image_pixels=max_image_pixels,
                dtype="bfloat16",
                allow_cpu=False,
                allow_openai_video_input=False,
                disable_thinking=True,
                judge_video_source="full",
                six_user_judge_mode=SIX_USER_JUDGE_MODE_LEGACY,
                dry_run=dry_run,
                generation_mode="baseline",
                fixed_question_type_schedule=True,
                question_types=("neutral",),
                resume=False,
                generator_decode_mode="sampling",
                generator_temperature=generator_temperature,
                generator_top_p=generator_top_p,
                generator_top_k=generator_top_k,
                packet_limit=1,
                max_packets_in_flight=1,
                max_generation_lanes=1,
                max_review_lanes=1,
                _packet_index_offset=global_loop_index,
                _generation_gate=generation_gate,
                _review_gate=review_gate,
            )
            counts = _compact_finished_loop_files(
                paths,
                source_packet_id=source_packet_id,
                asker_index=asker_index,
                asker_user=asker_user,
            )
            if counts["infrastructure"]:
                raise RuntimeError(
                    f"question loop ended in an infrastructure skip: {source_packet_id} "
                    f"asker={asker_user}; rerun the same output directory to retry"
                )
            if counts["accepted"] == 1 and counts["rejected"] == 0:
                outcome = "accepted"
            elif counts["accepted"] == 0 and counts["rejected"] == 1:
                outcome = "rejected"
            else:
                raise RuntimeError(
                    f"question loop produced an invalid terminal count for {source_packet_id} "
                    f"asker={asker_user}: accepted={counts['accepted']} "
                    f"rejected={counts['rejected']}"
                )
            intermediate_rows = _read_jsonl_if_present(paths["intermediate"])
            if len(intermediate_rows) != 1:
                raise RuntimeError(
                    f"question loop must produce exactly one intermediate row: "
                    f"packet={source_packet_id} loop={loop_index} "
                    f"rows={len(intermediate_rows)}"
                )
            loop_attempts_consumed = int(
                intermediate_rows[0].get("attempt_count") or 0
            )
            if not 1 <= loop_attempts_consumed <= loop_attempt_cap:
                raise RuntimeError(
                    f"question loop consumed an invalid attempt count: "
                    f"packet={source_packet_id} loop={loop_index} "
                    f"consumed={loop_attempts_consumed} cap={loop_attempt_cap}"
                )
            marker = {
                "schema_version": "egolife_rlhf_qa_loop_v1",
                "source_packet_id": source_packet_id,
                "evidence_id": view["evidence_id"],
                "loop_index": loop_index,
                "asker_index": asker_index,
                "asker_agent_dir": view["asker_agent_dir"],
                "asker_user": asker_user,
                "packet_asker_order": asker_order,
                "outcome": outcome,
                "attempts_consumed": loop_attempts_consumed,
                "max_attempts_for_loop": loop_attempt_cap,
                "packet_attempt_budget": attempts_per_packet,
                "answerability_design": "concurrent_asker_only_and_all_six_legacy_sufficiency",
                "generator_media": "full_unpruned_asker_plus_asker_relative_pruned_providers",
                "judge_media": "full_unpruned_sampled_frames_for_all_six_users",
                "row_counts": counts,
            }
            _atomic_write_json(paths["complete"], marker)
            completed_loops += 1
            attempts_consumed += loop_attempts_consumed
            accepted_count += int(outcome == "accepted")
            rejected_count += int(outcome == "rejected")
            print(
                "rlhf_question_loop_done "
                f"packet_id={source_packet_id} loop_index={loop_index} "
                f"asker={asker_user} outcome={outcome} "
                f"loop_attempts={loop_attempts_consumed} "
                f"packet_attempts={attempts_consumed}/{attempts_per_packet}",
                flush=True,
            )
            loop_index += 1
        result = {
            "source_packet_id": source_packet_id,
            "completed_question_loops": completed_loops,
            "generation_attempts_consumed": attempts_consumed,
            "accepted": accepted_count,
            "rejected": rejected_count,
            "elapsed_seconds": round(time.time() - packet_started, 3),
        }
        print(
            "rlhf_source_packet_done "
            f"packet_id={source_packet_id} loops={completed_loops} "
            f"generation_attempts={attempts_consumed}/{attempts_per_packet} "
            f"accepted={accepted_count} rejected={rejected_count} "
            f"seconds={result['elapsed_seconds']:.1f}",
            flush=True,
        )
        return result

    with ThreadPoolExecutor(max_workers=max_packets_in_flight) as executor:
        active = {
            executor.submit(run_source_packet, index, packet): index
            for index, packet in enumerate(selected_packets[:max_packets_in_flight])
        }
        next_index = len(active)
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                finished_index = active.pop(future)
                future.result()
                refresh_outputs()
                if next_index < len(selected_packets):
                    active[
                        executor.submit(
                            run_source_packet,
                            next_index,
                            selected_packets[next_index],
                        )
                    ] = next_index
                    next_index += 1
                print(
                    "rlhf_qa_pipeline_progress "
                    f"finished_source_packet_index={finished_index} "
                    f"next_source_packet_index={next_index}",
                    flush=True,
                )

    summary, published_checkpoints = refresh_outputs()
    if summary["remaining_generation_attempt_count"]:
        raise RuntimeError(
            "RLHF QA generation is incomplete: "
            f"{summary['remaining_generation_attempt_count']} generation attempts remain"
        )
    summary["elapsed_seconds_this_invocation"] = round(time.time() - started, 3)
    summary["checkpoint_packet_count"] = checkpoint_packet_count
    summary["published_labeling_checkpoints"] = published_checkpoints
    _atomic_write_json(output_dir / "run_summary.json", summary)
    print(
        "rlhf_qa_pipeline_done "
        f"source_packets={summary['source_packet_count']} "
        f"question_loops={summary['completed_question_loop_count']} "
        f"generation_attempts={summary['completed_generation_attempt_count']} "
        f"accepted={summary['accepted_question_count']} "
        f"rejected={summary['exhausted_rejected_loop_count']} "
        f"seconds={summary['elapsed_seconds_this_invocation']:.1f}",
        flush=True,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Spend six generation attempts across RLHF QA loops per packet"
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--base-url", default="http://127.0.0.1:18000/v1")
    parser.add_argument(
        "--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS_PER_LOOP
    )
    parser.add_argument(
        "--attempts-per-packet", type=int, default=DEFAULT_ATTEMPTS_PER_PACKET
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-image-pixels", type=int, default=262144)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--packet-limit", type=int)
    parser.add_argument(
        "--max-packets-in-flight", type=int, default=DEFAULT_PACKETS_IN_FLIGHT
    )
    parser.add_argument(
        "--max-generation-lanes", type=int, default=DEFAULT_GENERATION_LANES
    )
    parser.add_argument("--max-review-lanes", type=int, default=DEFAULT_REVIEW_LANES)
    parser.add_argument(
        "--checkpoint-packet-count",
        type=int,
        default=DEFAULT_CHECKPOINT_PACKET_COUNT,
    )
    parser.add_argument("--generator-temperature", type=float, default=0.7)
    parser.add_argument("--generator-top-p", type=float, default=0.9)
    parser.add_argument("--generator-top-k", type=int, default=40)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_rlhf_packet_generation(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        model_id=args.model_id,
        base_url=args.base_url,
        max_attempts=args.max_attempts,
        attempts_per_packet=args.attempts_per_packet,
        max_new_tokens=args.max_new_tokens,
        max_image_pixels=args.max_image_pixels,
        start_index=args.start_index,
        packet_limit=args.packet_limit,
        max_packets_in_flight=args.max_packets_in_flight,
        max_generation_lanes=args.max_generation_lanes,
        max_review_lanes=args.max_review_lanes,
        checkpoint_packet_count=args.checkpoint_packet_count,
        generator_temperature=args.generator_temperature,
        generator_top_p=args.generator_top_p,
        generator_top_k=args.generator_top_k,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
