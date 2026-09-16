"""Build a blinded six-user binary-judge human-labeling package.

The preferred input is one or more metadata-only RLHF checkpoint bundles. The
bundles contain generated questions plus ordered Hugging Face source-video
URLs, but no image or video bytes. Automatic judge outputs and generation
traces are deliberately excluded from the annotator payload. The legacy
extracted-checkpoint and sampled-frame dataset input remains available for
backward compatibility.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from urllib.parse import quote

import numpy as np


LABEL_SCHEMA_VERSION = "egolife_six_user_binary_labeling_v2"
CHECKPOINT_SCHEMA_VERSION = "egolife_rlhf_labeling_checkpoint_v1"
METADATA_BUNDLE_SCHEMA_VERSION = "egolife_labeling_metadata_bundle_v1"
FAIL_SCORE = 1
PASS_SCORE = 2

FORBIDDEN_BUNDLE_MEDIA_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".gif",
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
}

EXPORT_COLUMNS = [
    "schema_version",
    "dataset_fingerprint",
    "assignment_id",
    "reviewer_id",
    "annotation_status",
    "packet_order",
    "evidence_id",
    "display_order",
    "candidate_id",
    "source_qa_id",
    "source_evidence_id",
    "generation_attempt",
    "asker_user",
    "formality_verdict",
    "evidence_grounding_verdict",
    "answerability_verdict",
    "formality_score",
    "evidence_grounding_score",
    "answerability_score",
    "asker_only_answerable",
    "all_six_answerable",
    "failure_modes",
    "candidate_skipped",
    "skip_reason",
    "notes",
    "started_at",
    "completed_at",
    "active_seconds",
    "question",
    "options",
    "correct",
    "answer",
    "required_users",
    "source_video_urls",
]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_hex(*parts: Any, length: int = 20) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected one JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def _verified_checkpoint(checkpoint_dir: Path) -> tuple[dict[str, Any], Path]:
    checkpoint_dir = checkpoint_dir.resolve()
    ready_path = checkpoint_dir / "CHECKPOINT_READY.json"
    if not ready_path.is_file():
        raise FileNotFoundError(f"checkpoint readiness marker is missing: {ready_path}")
    manifest = _read_json(ready_path)
    if manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"unsupported checkpoint schema: {ready_path}")
    payloads = manifest.get("payload_sha256")
    if not isinstance(payloads, dict) or not payloads:
        raise ValueError(f"checkpoint has no payload checksums: {ready_path}")
    for name, expected in payloads.items():
        payload_path = checkpoint_dir / str(name)
        if not payload_path.is_file():
            raise FileNotFoundError(f"checkpoint payload is missing: {payload_path}")
        actual = _sha256_file(payload_path)
        if actual.casefold() != str(expected).casefold():
            raise ValueError(f"checkpoint payload checksum mismatch: {payload_path}")
    labeling_name = str(manifest.get("labeling_file") or "labeling_queue.jsonl")
    labeling_path = checkpoint_dir / labeling_name
    if not labeling_path.is_file():
        raise FileNotFoundError(f"checkpoint labeling queue is missing: {labeling_path}")
    return manifest, labeling_path


def _jsonl_bytes(payload: bytes, *, source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{source}:{line_number}: expected a JSON object")
        rows.append(value)
    return rows


def _verified_metadata_bundle(
    bundle_path: Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    str,
]:
    """Read one metadata-only archive without extracting it to disk."""

    bundle_path = bundle_path.resolve()
    if not bundle_path.is_file():
        raise FileNotFoundError(f"metadata bundle is missing: {bundle_path}")
    archive_sha256 = _sha256_file(bundle_path)
    checksum_path = bundle_path.with_name(f"{bundle_path.name}.sha256")
    if not checksum_path.is_file():
        raise FileNotFoundError(f"metadata bundle checksum is missing: {checksum_path}")
    checksum_tokens = checksum_path.read_text(encoding="utf-8-sig").strip().split()
    if not checksum_tokens or checksum_tokens[0].casefold() != archive_sha256.casefold():
        raise ValueError(f"metadata bundle checksum mismatch: {bundle_path}")

    with tarfile.open(bundle_path, mode="r:gz") as archive:
        members = {member.name: member for member in archive.getmembers()}
        for member in members.values():
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe archive member: {member.name}")
            suffix = relative.suffix.casefold()
            if (
                suffix in FORBIDDEN_BUNDLE_MEDIA_SUFFIXES
                or "frames" in relative.parts
                or relative.name == "clip_embeddings.f16.npy"
            ):
                raise ValueError(f"metadata bundle contains forbidden media: {member.name}")

        def member_bytes(name: str) -> bytes:
            member = members.get(name)
            if member is None or not member.isfile():
                raise FileNotFoundError(f"{bundle_path}: missing archive member {name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise FileNotFoundError(f"{bundle_path}: cannot read archive member {name}")
            return stream.read()

        bundle_info = json.loads(member_bytes("BUNDLE_INFO.json").decode("utf-8-sig"))
        if bundle_info.get("schema_version") != METADATA_BUNDLE_SCHEMA_VERSION:
            raise ValueError(f"unsupported metadata bundle schema: {bundle_path}")

        ready_names = sorted(
            name
            for name in members
            if re.fullmatch(r"checkpoint/[^/]+/CHECKPOINT_READY\.json", name)
        )
        if len(ready_names) != 1:
            raise ValueError(
                f"{bundle_path}: expected exactly one checkpoint readiness marker"
            )
        ready_name = ready_names[0]
        checkpoint_prefix = ready_name.rsplit("/", 1)[0]
        manifest = json.loads(member_bytes(ready_name).decode("utf-8-sig"))
        if manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(f"unsupported checkpoint schema in {bundle_path}")

        payloads = manifest.get("payload_sha256")
        if not isinstance(payloads, dict) or not payloads:
            raise ValueError(f"checkpoint has no payload checksums in {bundle_path}")
        for name, expected in payloads.items():
            relative = PurePosixPath(str(name))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe checkpoint payload in {bundle_path}: {name}")
            payload = member_bytes(f"{checkpoint_prefix}/{relative.as_posix()}")
            actual = hashlib.sha256(payload).hexdigest()
            if actual.casefold() != str(expected).casefold():
                raise ValueError(f"checkpoint payload checksum mismatch in {bundle_path}: {name}")

        labeling_name = str(manifest.get("labeling_file") or "labeling_queue.jsonl")
        labeling_payload = member_bytes(f"{checkpoint_prefix}/{labeling_name}")
        labeling_rows = _jsonl_bytes(
            labeling_payload,
            source=f"{bundle_path}!{checkpoint_prefix}/{labeling_name}",
        )
        trajectory_name = str(
            manifest.get("trajectory_file") or "qa_mcq.intermediate.jsonl"
        )
        trajectory_member = f"{checkpoint_prefix}/{trajectory_name}"
        trajectory_rows = (
            _jsonl_bytes(
                member_bytes(trajectory_member),
                source=f"{bundle_path}!{trajectory_member}",
            )
            if trajectory_member in members
            else []
        )
        media_rows = _jsonl_bytes(
            member_bytes("MEDIA_MANIFEST.jsonl"),
            source=f"{bundle_path}!MEDIA_MANIFEST.jsonl",
        )
    return manifest, labeling_rows, trajectory_rows, media_rows, archive_sha256


def _trajectory_attempt_rows(
    trajectory_rows: Iterable[dict[str, Any]],
    labeling_rows: Iterable[dict[str, Any]] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split all generation attempts into parseable QA rows and format failures."""

    postprocessed_by_attempt: dict[tuple[str, int], dict[str, Any]] = {}
    for row in labeling_rows:
        evidence_id = str(row.get("evidence_id") or "").strip()
        attempt_number = int(row.get("attempt_count") or 0)
        if evidence_id and attempt_number:
            postprocessed_by_attempt[(evidence_id, attempt_number)] = row
    trajectory_rows = list(trajectory_rows)
    for loop in trajectory_rows:
        evidence_id = str(loop.get("evidence_id") or "").strip()
        for rejection in loop.get("rejections") or []:
            if not isinstance(rejection, dict) or not isinstance(rejection.get("qa"), dict):
                continue
            attempt_number = int(rejection.get("attempt") or 0)
            if evidence_id and attempt_number:
                postprocessed_by_attempt[(evidence_id, attempt_number)] = rejection["qa"]

    candidates: list[dict[str, Any]] = []
    unparseable: list[dict[str, Any]] = []
    for loop_ordinal, loop in enumerate(trajectory_rows, 1):
        packet_id = str(loop.get("rlhf_source_packet_id") or "").strip()
        evidence_id = str(loop.get("evidence_id") or "").strip()
        if not packet_id or not evidence_id:
            raise ValueError(
                f"trajectory row {loop_ordinal} is missing packet/evidence provenance"
            )
        raw_attempts = loop.get("attempts")
        if not isinstance(raw_attempts, list):
            raise ValueError(f"{evidence_id}: trajectory attempts must be a list")
        for fallback_attempt, attempt in enumerate(raw_attempts, 1):
            if not isinstance(attempt, dict):
                raise ValueError(f"{evidence_id}: attempt {fallback_attempt} is not an object")
            attempt_number = int(attempt.get("attempt") or fallback_attempt)
            generation = attempt.get("generation")
            parsed_qa = generation.get("parsed_qa") if isinstance(generation, dict) else None
            provenance = {
                "rlhf_source_packet_id": packet_id,
                "rlhf_asker_index": loop.get("rlhf_asker_index"),
                "rlhf_asker_user": loop.get("rlhf_asker_user"),
                "source_evidence_id": evidence_id,
                "generation_attempt": attempt_number,
                "candidate_instance_id": f"{evidence_id}::attempt_{attempt_number:02d}",
            }
            if not isinstance(parsed_qa, dict):
                unparseable.append(provenance)
                continue
            candidates.append(
                {
                    **provenance,
                    "qa": parsed_qa,
                    "postprocessed_qa": postprocessed_by_attempt.get(
                        (evidence_id, attempt_number), {}
                    ),
                }
            )
    return candidates, unparseable


def _qa_payload(row: dict[str, Any]) -> dict[str, Any]:
    for key in ("qa", "raw_qa", "normalized_qa"):
        value = row.get(key)
        if isinstance(value, dict):
            return value
    return row


def _normalized_candidate(
    row: dict[str, Any],
    *,
    source_packet_id: str,
    ordinal: int,
    strict: bool = True,
) -> dict[str, Any]:
    qa = _qa_payload(row)
    question = str(qa.get("question") or "").strip()
    raw_options = qa.get("options")
    options = (
        [str(value).strip() for value in raw_options]
        if isinstance(raw_options, list)
        else []
    )
    correct = str(qa.get("correct") or "").strip().upper()
    if strict and not question:
        raise ValueError(f"{source_packet_id}: accepted row {ordinal} has no question")
    if strict and (len(options) != 5 or any(not value for value in options)):
        raise ValueError(f"{source_packet_id}: accepted row {ordinal} needs five options")
    if strict and correct not in tuple("ABCDE"):
        raise ValueError(f"{source_packet_id}: accepted row {ordinal} has invalid correct")
    correct_index = ord(correct) - ord("A") if len(correct) == 1 and correct.isalpha() else -1
    answer = str(qa.get("answer") or "").strip()
    if not answer and 0 <= correct_index < len(options):
        answer = options[correct_index]
    if strict and answer != options[correct_index]:
        raise ValueError(f"{source_packet_id}: answer does not match correct option")
    if not strict:
        question = question or "[Missing question]"
        options = [value or f"[Empty option {index}]" for index, value in enumerate(options, 1)]
        correct = correct or "[Missing correct option]"
        answer = answer or "[Missing answer]"
    source_qa_id = str(qa.get("qa_id") or row.get("qa_id") or "").strip()
    if not source_qa_id:
        source_qa_id = "QA_" + _stable_hex(source_packet_id, question, options, ordinal)
    asker_index = int(row.get("rlhf_asker_index", qa.get("asker_index", -1)))
    if not 0 <= asker_index < 6:
        raise ValueError(f"{source_packet_id}: row {ordinal} has invalid asker index")
    return {
        "candidate_id": str(
            row.get("candidate_instance_id") or f"{source_packet_id}::{source_qa_id}"
        ),
        "source_qa_id": source_qa_id,
        "source_evidence_id": str(row.get("source_evidence_id") or ""),
        "generation_attempt": row.get("generation_attempt"),
        "asker_index": asker_index,
        "asker_user": str(row.get("rlhf_asker_user") or qa.get("asker_user") or ""),
        "question": question,
        "options": options,
        "correct": correct,
        "answer": answer,
        "question_type": str(qa.get("question_type") or "neutral"),
        "required_users": [str(value) for value in qa.get("required_users") or []],
        "reported_timestamps": [
            value
            for value in qa.get("referred_timestamps") or []
            if isinstance(value, dict)
        ],
        "reported_evidence": [
            value
            for value in (row.get("postprocessed_qa") or {}).get("evidence") or []
            if isinstance(value, dict)
        ],
    }


def _clock_seconds(value: str) -> int:
    match = re.search(r"(?<!\d)([01]?\d|2[0-3]):([0-5]\d):([0-5]\d)", value)
    if match is None:
        raise ValueError(f"cannot parse source clock: {value!r}")
    hours, minutes, seconds = (int(part) for part in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def _format_clock(total_seconds: float) -> str:
    value = int(round(total_seconds)) % 86400
    return f"{value // 3600:02d}:{value // 60 % 60:02d}:{value % 60:02d}"


def _format_relative_time(total_seconds: float) -> str:
    value = max(0, int(round(total_seconds)))
    return f"{value // 60:02d}:{value % 60:02d}"


def _normalize_reported_seconds(value: Any, *, start_clock_seconds: int) -> tuple[float | None, str]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None, "unresolved"
    if 0.0 <= numeric <= 600.0:
        return numeric, "relative_seconds"
    integer = int(numeric)
    hours, minutes, seconds = integer // 10000, integer // 100 % 100, integer % 100
    if 0 <= hours < 24 and 0 <= minutes < 60 and 0 <= seconds < 60:
        delta = (hours * 3600 + minutes * 60 + seconds - start_clock_seconds) % 86400
        if 0 <= delta <= 600:
            return float(delta), "absolute_hhmmss"
    minutes, seconds = integer // 100, integer % 100
    mmss = minutes * 60 + seconds
    if 0 <= seconds < 60 and 0 <= mmss <= 600:
        return float(mmss), "relative_mmss"
    return None, "unresolved"


def _candidate_evidence_locations(
    candidate: dict[str, Any],
    *,
    users: list[dict[str, Any]],
    clip_clock: str,
) -> list[dict[str, Any]]:
    """Turn generator-reported users/times into packet-relative jump targets."""

    start_clock_seconds = _clock_seconds(clip_clock)
    user_index_by_name: dict[str, int] = {}
    for user in users:
        user_index = int(user["user_index"])
        for name in (user.get("agent_name"), user.get("agent_dir"), user.get("agent_id")):
            if name:
                user_index_by_name[str(name).casefold()] = user_index

    locations: list[dict[str, Any]] = []
    dedupe: dict[tuple[str, int | None, str], dict[str, Any]] = {}

    def add_location(
        *,
        user_name: str,
        relative_seconds: float | None,
        normalization: str,
        source: str,
        raw_timestamp: Any,
        moment: str,
    ) -> None:
        user_index = user_index_by_name.get(user_name.casefold())
        rounded = int(round(relative_seconds)) if relative_seconds is not None else None
        key = (
            user_name.casefold(),
            rounded,
            "" if rounded is not None else str(raw_timestamp),
        )
        existing = dedupe.get(key)
        if existing is not None:
            if source not in existing["sources"]:
                existing["sources"].append(source)
            if moment and not existing["moment"]:
                existing["moment"] = moment
            return
        location = {
            "user": user_name or "[Missing user]",
            "user_index": user_index,
            "relative_seconds": relative_seconds,
            "relative_time": (
                _format_relative_time(relative_seconds)
                if relative_seconds is not None
                else ""
            ),
            "absolute_clock": (
                _format_clock(start_clock_seconds + relative_seconds)
                if relative_seconds is not None
                else ""
            ),
            "segment_index": (
                min(19, int(relative_seconds // 30))
                if relative_seconds is not None
                else None
            ),
            "segment_offset_seconds": (
                round(relative_seconds % 30, 3)
                if relative_seconds is not None
                else None
            ),
            "normalization": normalization,
            "sources": [source],
            "raw_timestamp": str(raw_timestamp),
            "moment": moment,
        }
        dedupe[key] = location
        locations.append(location)

    for reference in candidate.pop("reported_timestamps", []):
        raw_timestamp = reference.get("timestamp_seconds")
        relative_seconds, normalization = _normalize_reported_seconds(
            raw_timestamp,
            start_clock_seconds=start_clock_seconds,
        )
        add_location(
            user_name=str(reference.get("user") or "").strip(),
            relative_seconds=relative_seconds,
            normalization=normalization,
            source="referred_timestamp",
            raw_timestamp=raw_timestamp,
            moment=str(reference.get("moment") or "").strip(),
        )

    clock_pattern = re.compile(
        r"(?<!\d)([01]?\d|2[0-3]):([0-5]\d):([0-5]\d)(?::\d+)?"
    )
    for evidence in candidate.pop("reported_evidence", []):
        user_name = str(evidence.get("user") or "").strip()
        moment = str(evidence.get("needed_fact") or "").strip()
        for frame in evidence.get("frames_used") or []:
            match = clock_pattern.search(str(frame))
            if match is None:
                continue
            hours, minutes, seconds = (int(part) for part in match.groups())
            relative_seconds = (
                hours * 3600 + minutes * 60 + seconds - start_clock_seconds
            ) % 86400
            if relative_seconds > 600:
                continue
            add_location(
                user_name=user_name,
                relative_seconds=float(relative_seconds),
                normalization="evidence_frame_clock",
                source="evidence_frame",
                raw_timestamp=str(frame),
                moment=moment,
            )

    return sorted(
        locations,
        key=lambda row: (
            row["relative_seconds"] is None,
            row["relative_seconds"] if row["relative_seconds"] is not None else 0,
            row["user"],
        ),
    )


def _media_url(packet_id: str, relative_path: str) -> str:
    relative = Path("packets") / packet_id / Path(relative_path)
    return "/evidence/" + quote(relative.as_posix(), safe="/")


def _packet_media(packet_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    packet = _read_json(packet_dir / "packet.json")
    clusters_path = packet_dir / "clusters.json"
    clusters = _read_json(clusters_path) if clusters_path.is_file() else {"users": []}
    cluster_by_user = {
        int(row["user_index"]): row
        for row in clusters.get("users") or []
        if isinstance(row, dict) and "user_index" in row
    }
    source_by_agent = {
        str(row.get("agent_dir")): row
        for row in (packet.get("source") or {}).get("users") or []
        if isinstance(row, dict)
    }
    users = []
    for fallback_index, user in enumerate(packet.get("users") or []):
        if not isinstance(user, dict):
            continue
        user_index = int(user.get("user_index", fallback_index))
        cluster = cluster_by_user.get(user_index, {})
        labels = [int(value) for value in cluster.get("labels") or []]
        representatives = {
            int(row["frame_index"])
            for row in cluster.get("representatives") or []
            if isinstance(row, dict) and "frame_index" in row
        }
        frames = []
        for frame in user.get("frames") or []:
            frame_index = int(frame.get("frame_index", len(frames)))
            frames.append(
                {
                    "frame_index": frame_index,
                    "timestamp_seconds": float(frame.get("timestamp_seconds", 0.0)),
                    "source_segment_index": int(frame.get("source_segment_index", 0)),
                    "cluster_index": labels[frame_index] if frame_index < len(labels) else None,
                    "cluster_representative": frame_index in representatives,
                    "media_url": _media_url(packet_dir.name, str(frame.get("path") or "")),
                }
            )
        source = source_by_agent.get(str(user.get("agent_dir")), {})
        segments = []
        for segment_index, segment in enumerate(source.get("segments") or []):
            if not isinstance(segment, dict):
                continue
            segments.append(
                {
                    "segment_index": segment_index,
                    "clip_id": str(segment.get("clip_id") or f"segment_{segment_index:03d}"),
                    "time_token": str(segment.get("time_token") or ""),
                    "video_url": str(segment.get("video_url") or ""),
                }
            )
        users.append(
            {
                "user_index": user_index,
                "agent_dir": str(user.get("agent_dir") or f"user_{user_index}"),
                "agent_id": str(user.get("agent_id") or ""),
                "agent_name": str(user.get("agent_name") or user.get("agent_dir") or f"User {user_index + 1}"),
                "frame_count": len(frames),
                "cluster_count": int(cluster.get("cluster_count") or 0),
                "frames": frames,
                "source_segments": segments,
            }
        )
    if len(users) != 6:
        raise ValueError(f"{packet_dir}: expected exactly six users")
    return packet, users


def _keep_indices(packet_dir: Path, asker_index: int) -> list[list[int]]:
    masks_path = packet_dir / "keep_masks.npz"
    if not masks_path.is_file():
        raise FileNotFoundError(f"packet keep masks are missing: {masks_path}")
    with np.load(masks_path, allow_pickle=False) as values:
        masks = values["keep_masks"]
        if masks.ndim != 3 or masks.shape[0] != 6 or masks.shape[1] != 6:
            raise ValueError(f"invalid keep-mask shape in {masks_path}: {masks.shape}")
        selected = masks[asker_index]
        return [
            [int(index) for index, keep in enumerate(user_mask) if bool(keep)]
            for user_mask in selected
        ]


def build_labeling_data(
    checkpoint_dirs: Iterable[Path],
    *,
    dataset_root: Path,
    assignment_id: str,
    assignment_seed: int = 20260915,
    packet_start: int = 1,
    packet_count: int | None = None,
) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    if packet_start < 1:
        raise ValueError("packet_start is one-based and must be positive")
    grouped: dict[str, list[dict[str, Any]]] = {}
    source_manifests = []
    fingerprint_parts = []
    packet_order_hint: list[str] = []
    for checkpoint_dir in checkpoint_dirs:
        manifest, labeling_path = _verified_checkpoint(checkpoint_dir)
        source_manifests.append({"checkpoint_dir": str(checkpoint_dir.resolve()), **manifest})
        fingerprint_parts.append(_sha256_file(labeling_path))
        for packet_id in manifest.get("source_packet_ids") or []:
            packet_id = str(packet_id)
            if packet_id not in packet_order_hint:
                packet_order_hint.append(packet_id)
        for row in _read_jsonl(labeling_path):
            packet_id = str(row.get("rlhf_source_packet_id") or "").strip()
            if not packet_id:
                raise ValueError(f"{labeling_path}: accepted row has no rlhf_source_packet_id")
            grouped.setdefault(packet_id, []).append(row)

    ordered_ids = [packet_id for packet_id in packet_order_hint if packet_id in grouped]
    ordered_ids.extend(sorted(set(grouped) - set(ordered_ids)))
    selected_ids = ordered_ids[packet_start - 1 :]
    if packet_count is not None:
        if packet_count < 1:
            raise ValueError("packet_count must be positive")
        selected_ids = selected_ids[:packet_count]
    if not selected_ids:
        raise ValueError("the requested assignment contains no accepted questions")

    entries = []
    for packet_order, packet_id in enumerate(selected_ids, 1):
        packet_dir = dataset_root / "packets" / packet_id
        if not (packet_dir / "COMPLETE").is_file():
            raise FileNotFoundError(f"preprocessed packet is incomplete: {packet_dir}")
        packet, users = _packet_media(packet_dir)
        fingerprint_parts.extend(
            _sha256_file(packet_dir / name)
            for name in ("packet.json", "clusters.json", "keep_masks.npz")
            if (packet_dir / name).is_file()
        )
        candidates = [
            _normalized_candidate(row, source_packet_id=packet_id, ordinal=index)
            for index, row in enumerate(grouped[packet_id], 1)
        ]
        if len({row["candidate_id"] for row in candidates}) != len(candidates):
            raise ValueError(f"{packet_id}: duplicate QA IDs in labeling queue")
        candidates.sort(
            key=lambda row: _stable_hex(
                assignment_seed, packet_id, row["candidate_id"], length=40
            )
        )
        for display_order, candidate in enumerate(candidates, 1):
            candidate["display_order"] = display_order
            candidate["keep_frame_indices_by_user"] = _keep_indices(
                packet_dir, int(candidate["asker_index"])
            )
            if not candidate["asker_user"]:
                candidate["asker_user"] = users[int(candidate["asker_index"])]["agent_name"]
            candidate["evidence_locations"] = _candidate_evidence_locations(
                candidate,
                users=users,
                clip_clock=str(packet.get("clip_clock") or ""),
            )
        entries.append(
            {
                "packet_order": packet_order,
                "evidence_id": packet_id,
                "source_packet_id": packet_id,
                "day": str(packet.get("day") or ""),
                "time_token": str(packet.get("time_token") or ""),
                "clip_clock": str(packet.get("clip_clock") or ""),
                "duration_seconds": float(packet.get("duration_seconds") or 0.0),
                "users": users,
                "candidate_count": len(candidates),
                "candidates": candidates,
            }
        )

    fingerprint = hashlib.sha256("\n".join(sorted(fingerprint_parts)).encode()).hexdigest()[:20]
    return {
        "schema_version": LABEL_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_fingerprint": fingerprint,
        "assignment": {
            "assignment_id": assignment_id,
            "assignment_seed": assignment_seed,
            "packet_start": packet_start,
            "packet_count": len(entries),
            "binary_score_encoding": {"fail": FAIL_SCORE, "pass": PASS_SCORE},
            "answerability_gate": "asker_only.answerable == false and all_six.answerable == true",
            "blinding": "automatic judge outputs, subchecks, prompts, and generation traces are excluded",
        },
        "summary": {
            "packet_count": len(entries),
            "candidate_count": sum(row["candidate_count"] for row in entries),
            "zero_acceptance_packet_count": len(packet_order_hint) - len(ordered_ids),
        },
        "packets": entries,
        "sources": source_manifests,
    }


def _source_video_users(media_row: dict[str, Any], *, packet_id: str) -> list[dict[str, Any]]:
    users = []
    for user_index, raw_user in enumerate(media_row.get("users") or []):
        if not isinstance(raw_user, dict):
            raise ValueError(f"{packet_id}: invalid user metadata at index {user_index}")
        segments = []
        for fallback_index, raw_segment in enumerate(raw_user.get("segments") or []):
            if not isinstance(raw_segment, dict):
                raise ValueError(
                    f"{packet_id}: invalid source segment for user {user_index}"
                )
            segment_index = int(raw_segment.get("segment_index", fallback_index))
            video_url = str(raw_segment.get("video_url") or "").strip()
            if not video_url.startswith("https://huggingface.co/"):
                raise ValueError(
                    f"{packet_id}: source segment {user_index}/{segment_index} "
                    "does not have a Hugging Face URL"
                )
            start_seconds = float(raw_segment.get("start_seconds", segment_index * 30))
            end_seconds = float(raw_segment.get("end_seconds", start_seconds + 30))
            if end_seconds <= start_seconds:
                raise ValueError(
                    f"{packet_id}: invalid source segment interval "
                    f"{user_index}/{segment_index}"
                )
            segments.append(
                {
                    "segment_index": segment_index,
                    "start_seconds": start_seconds,
                    "end_seconds": end_seconds,
                    "clip_id": str(
                        raw_segment.get("clip_id") or f"segment_{segment_index:03d}"
                    ),
                    "time_token": str(raw_segment.get("time_token") or ""),
                    "video_url": video_url,
                }
            )
        segments.sort(key=lambda row: row["segment_index"])
        if len(segments) != 20 or [row["segment_index"] for row in segments] != list(
            range(20)
        ):
            raise ValueError(
                f"{packet_id}: user {user_index} must have 20 ordered 30-second segments"
            )
        users.append(
            {
                "user_index": user_index,
                "agent_dir": str(raw_user.get("agent_dir") or f"user_{user_index}"),
                "agent_id": str(raw_user.get("agent_id") or ""),
                "agent_name": str(
                    raw_user.get("agent_name")
                    or raw_user.get("agent_dir")
                    or f"User {user_index + 1}"
                ),
                "source_segments": segments,
            }
        )
    if len(users) != 6:
        raise ValueError(f"{packet_id}: expected exactly six source-video users")
    return users


def build_labeling_data_from_metadata_bundles(
    metadata_bundles: Iterable[Path],
    *,
    assignment_id: str,
    assignment_seed: int = 20260915,
    packet_start: int = 1,
    packet_count: int | None = None,
) -> dict[str, Any]:
    """Build a blinded assignment directly from metadata-only ``.tgz`` files."""

    if packet_start < 1:
        raise ValueError("packet_start is one-based and must be positive")
    grouped: dict[str, list[dict[str, Any]]] = {}
    attempt_count_by_packet: dict[str, int] = {}
    unparseable_count_by_packet: dict[str, int] = {}
    unparseable_by_packet: dict[str, list[dict[str, Any]]] = {}
    media_by_packet: dict[str, dict[str, Any]] = {}
    packet_order_hint: list[str] = []
    source_manifests = []
    fingerprint_parts = []

    for bundle_path in metadata_bundles:
        (
            manifest,
            labeling_rows,
            trajectory_rows,
            media_rows,
            archive_sha256,
        ) = _verified_metadata_bundle(bundle_path)
        source_manifests.append(
            {
                "metadata_bundle": str(bundle_path.resolve()),
                "archive_sha256": archive_sha256,
                **manifest,
            }
        )
        fingerprint_parts.append(archive_sha256)
        for packet_id in manifest.get("source_packet_ids") or []:
            packet_id = str(packet_id)
            if packet_id not in packet_order_hint:
                packet_order_hint.append(packet_id)
        if trajectory_rows:
            candidate_rows, unparseable_rows = _trajectory_attempt_rows(
                trajectory_rows,
                labeling_rows,
            )
            for row in candidate_rows:
                packet_id = str(row["rlhf_source_packet_id"])
                grouped.setdefault(packet_id, []).append(row)
                attempt_count_by_packet[packet_id] = (
                    attempt_count_by_packet.get(packet_id, 0) + 1
                )
            for row in unparseable_rows:
                packet_id = str(row["rlhf_source_packet_id"])
                attempt_count_by_packet[packet_id] = (
                    attempt_count_by_packet.get(packet_id, 0) + 1
                )
                unparseable_count_by_packet[packet_id] = (
                    unparseable_count_by_packet.get(packet_id, 0) + 1
                )
                unparseable_by_packet.setdefault(packet_id, []).append(row)
        else:
            # Compatibility fallback for old bundles that predate full trajectory export.
            for row in labeling_rows:
                packet_id = str(row.get("rlhf_source_packet_id") or "").strip()
                if not packet_id:
                    raise ValueError(
                        f"{bundle_path}: accepted row has no source packet ID"
                    )
                grouped.setdefault(packet_id, []).append(row)
                attempt_count_by_packet[packet_id] = (
                    attempt_count_by_packet.get(packet_id, 0) + 1
                )
        for media_row in media_rows:
            packet_id = str(media_row.get("packet_id") or "").strip()
            if not packet_id:
                raise ValueError(f"{bundle_path}: media row has no packet ID")
            if packet_id in media_by_packet:
                raise ValueError(f"duplicate media metadata for packet {packet_id}")
            media_by_packet[packet_id] = media_row

    ordered_ids = list(packet_order_hint)
    ordered_ids.extend(sorted(set(grouped) - set(ordered_ids)))
    selected_ids = ordered_ids[packet_start - 1 :]
    if packet_count is not None:
        if packet_count < 1:
            raise ValueError("packet_count must be positive")
        selected_ids = selected_ids[:packet_count]
    if not selected_ids:
        raise ValueError("the requested assignment contains no source packets")

    entries = []
    for packet_order, packet_id in enumerate(selected_ids, 1):
        media_row = media_by_packet.get(packet_id)
        if media_row is None:
            raise FileNotFoundError(f"missing source-video metadata for packet {packet_id}")
        duration_seconds = float(media_row.get("duration_seconds") or 0.0)
        if duration_seconds != 600.0:
            raise ValueError(
                f"{packet_id}: expected a 600-second source timeline, got {duration_seconds}"
            )
        users = _source_video_users(media_row, packet_id=packet_id)
        candidates = [
            _normalized_candidate(
                row,
                source_packet_id=packet_id,
                ordinal=index,
                strict=not bool(row.get("candidate_instance_id")),
            )
            for index, row in enumerate(grouped.get(packet_id, []), 1)
        ]
        if len({row["candidate_id"] for row in candidates}) != len(candidates):
            raise ValueError(f"{packet_id}: duplicate generation-attempt candidate IDs")
        candidates.sort(
            key=lambda row: _stable_hex(
                assignment_seed, packet_id, row["candidate_id"], length=40
            )
        )
        for display_order, candidate in enumerate(candidates, 1):
            candidate["display_order"] = display_order
            if not candidate["asker_user"]:
                candidate["asker_user"] = users[int(candidate["asker_index"])][
                    "agent_name"
                ]
            candidate["evidence_locations"] = _candidate_evidence_locations(
                candidate,
                users=users,
                clip_clock=str(media_row.get("clip_clock") or ""),
            )
        entries.append(
            {
                "packet_order": packet_order,
                "evidence_id": packet_id,
                "source_packet_id": packet_id,
                "day": str(media_row.get("day") or ""),
                "time_token": str(media_row.get("time_token") or ""),
                "clip_clock": str(media_row.get("clip_clock") or ""),
                "duration_seconds": duration_seconds,
                "media_policy": "full_ten_minute_huggingface_source_timelines",
                "users": users,
                "candidate_count": len(candidates),
                "candidates": candidates,
            }
        )

    fingerprint = hashlib.sha256(
        "\n".join(sorted(fingerprint_parts)).encode()
    ).hexdigest()[:20]
    unlabelable_attempts = [
        {
            "candidate_id": row["candidate_instance_id"],
            "source_packet_id": row["rlhf_source_packet_id"],
            "source_evidence_id": row["source_evidence_id"],
            "generation_attempt": row["generation_attempt"],
            "asker_index": row["rlhf_asker_index"],
            "asker_user": row["rlhf_asker_user"],
            "reason": "generator output was not parsed as a QA JSON object",
        }
        for packet_id in selected_ids
        for row in unparseable_by_packet.get(packet_id, [])
    ]
    return {
        "schema_version": LABEL_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_fingerprint": fingerprint,
        "assignment": {
            "assignment_id": assignment_id,
            "assignment_seed": assignment_seed,
            "packet_start": packet_start,
            "packet_count": len(entries),
            "binary_score_encoding": {"fail": FAIL_SCORE, "pass": PASS_SCORE},
            "answerability_gate": (
                "asker_only.answerable == false and all_six.answerable == true"
            ),
            "media_policy": "full_ten_minute_huggingface_source_timelines",
            "blinding": (
                "automatic judge outputs, acceptance/rejection status, subchecks, "
                "prompts, and generation traces are excluded"
            ),
        },
        "summary": {
            "packet_count": len(entries),
            "candidate_count": sum(row["candidate_count"] for row in entries),
            "generation_attempt_count": sum(
                attempt_count_by_packet.get(packet_id, 0) for packet_id in selected_ids
            ),
            "unparseable_generation_attempt_count": sum(
                unparseable_count_by_packet.get(packet_id, 0)
                for packet_id in selected_ids
            ),
        },
        "packets": entries,
        "unlabelable_generation_attempts": unlabelable_attempts,
        "sources": source_manifests,
    }


def _blank_csv_rows(data: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for packet in data["packets"]:
        for candidate in packet["candidates"]:
            ordered_users = sorted(
                packet["users"],
                key=lambda user: user["user_index"] != candidate["asker_index"],
            )
            yield {
                "schema_version": data["schema_version"],
                "dataset_fingerprint": data["dataset_fingerprint"],
                "assignment_id": data["assignment"]["assignment_id"],
                "annotation_status": "pending",
                "packet_order": packet["packet_order"],
                "evidence_id": packet["evidence_id"],
                "display_order": candidate["display_order"],
                "candidate_id": candidate["candidate_id"],
                "source_qa_id": candidate["source_qa_id"],
                "source_evidence_id": candidate.get("source_evidence_id", ""),
                "generation_attempt": candidate.get("generation_attempt", ""),
                "asker_user": candidate["asker_user"],
                "failure_modes": "[]",
                "question": candidate["question"],
                "options": _canonical_json(candidate["options"]),
                "correct": candidate["correct"],
                "answer": candidate["answer"],
                "required_users": _canonical_json(candidate["required_users"]),
                "source_video_urls": _canonical_json(
                    [
                        segment["video_url"]
                        for user in ordered_users
                        for segment in user["source_segments"]
                        if segment["video_url"]
                    ]
                ),
            }


def _write_csv_template(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPORT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_blank_csv_rows(data))


def _write_local_server(path: Path) -> None:
    server = '''"""Serve the six-user RLHF source-video labeler."""

from __future__ import annotations

import http.server
from pathlib import Path
import sys
import threading
import webbrowser

PACKAGE_ROOT = Path(__file__).resolve().parent


def main() -> int:
    html_path = PACKAGE_ROOT / "rlhf_labeling.html"
    if not html_path.is_file():
        print(f"Missing labeler: {html_path}")
        return 1
    handler = lambda *args, **kwargs: http.server.SimpleHTTPRequestHandler(
        *args, directory=str(PACKAGE_ROOT), **kwargs
    )
    with http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        url = f"http://127.0.0.1:{server.server_address[1]}/rlhf_labeling.html"
        print(f"Serving labeler: {url}")
        print("Video segments load directly from Hugging Face when selected.")
        print("Keep this window open while labeling. Press Ctrl+C to stop.")
        if "--no-browser" not in sys.argv:
            threading.Timer(0.35, lambda: webbrowser.open(url)).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\\nLabeling server stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
    path.write_text(server, encoding="utf-8")


def write_labeling_package(
    data: dict[str, Any],
    *,
    output_dir: Path,
    template_path: Path,
    dataset_root: Path | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "labeling_data.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "assignment_manifest.json").write_text(
        json.dumps(
            {key: data[key] for key in ("schema_version", "generated_at", "dataset_fingerprint", "assignment", "summary", "sources")},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    unlabelable = data.get("unlabelable_generation_attempts") or []
    (output_dir / "unlabelable_generation_attempts.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in unlabelable),
        encoding="utf-8",
    )
    _write_csv_template(output_dir / "human_labels_template.csv", data)
    template = template_path.read_text(encoding="utf-8")
    embedded = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = template.replace("__LABELING_DATA__", embedded)
    if html == template:
        raise ValueError(f"{template_path}: missing __LABELING_DATA__ placeholder")
    (output_dir / "rlhf_labeling.html").write_text(html, encoding="utf-8")
    _write_local_server(output_dir / "serve_rlhf_labeling.py")
    (output_dir / "open_rlhf_labeling.cmd").write_text(
        '@echo off\r\nwhere py >nul 2>&1\r\nif %errorlevel%==0 (py "%~dp0serve_rlhf_labeling.py") else (python "%~dp0serve_rlhf_labeling.py")\r\npause\r\n',
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--metadata-bundle",
        action="append",
        type=Path,
        help="metadata-only .tgz; repeat for multiple checkpoints",
    )
    source.add_argument(
        "--checkpoint-dir",
        action="append",
        type=Path,
        help="legacy extracted checkpoint; repeat for multiple checkpoints",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="legacy sampled-frame dataset root; not used with --metadata-bundle",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--assignment-id", required=True)
    parser.add_argument("--assignment-seed", type=int, default=20260915)
    parser.add_argument("--packet-start", type=int, default=1)
    parser.add_argument("--packet-count", type=int)
    parser.add_argument(
        "--template",
        type=Path,
        default=Path(__file__).with_name("rlhf_human_labeling_template.html"),
    )
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.assignment_id):
        raise ValueError("assignment_id may contain only letters, numbers, dot, dash, underscore")
    if args.metadata_bundle:
        if args.dataset_root is not None:
            parser.error("--dataset-root cannot be used with --metadata-bundle")
        data = build_labeling_data_from_metadata_bundles(
            args.metadata_bundle,
            assignment_id=args.assignment_id,
            assignment_seed=args.assignment_seed,
            packet_start=args.packet_start,
            packet_count=args.packet_count,
        )
    else:
        if args.dataset_root is None:
            parser.error("--dataset-root is required with --checkpoint-dir")
        data = build_labeling_data(
            args.checkpoint_dir,
            dataset_root=args.dataset_root,
            assignment_id=args.assignment_id,
            assignment_seed=args.assignment_seed,
            packet_start=args.packet_start,
            packet_count=args.packet_count,
        )
    write_labeling_package(
        data,
        output_dir=args.output_dir,
        template_path=args.template,
        dataset_root=args.dataset_root,
    )
    print(_canonical_json({"output_dir": str(args.output_dir.resolve()), **data["summary"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
