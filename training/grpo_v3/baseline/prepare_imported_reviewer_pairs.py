"""Prepare reviewer-imported full-video pairs for production CLIP pruning.

The reviewer media importer intentionally writes only the two ordered full
videos needed by the frozen reviewer.  GRPO generation needs a richer packet:
an explicit asker/provider order plus the metadata consumed by the established
30-second paired-pruning pipeline.  This adapter adds only that deterministic
routing metadata.  It does not create generator media or bypass pruning.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ..shared.data import read_jsonl, write_jsonl


IMPORTED_SCHEMA_VERSION = "reviewer_imported_annotation_media_v1"
PREPARED_SCHEMA_VERSION = "grpo_v3_imported_reviewer_full_video_pairs_v1"
EVIDENCE_ID_PATTERN = re.compile(
    r"^EGOLIFE2U_RANDOM_PAIR_CLIP_PRUNED_"
    r"(?P<day>DAY\d+)_(?P<time>\d{8})_(?P<left>A\d+)_(?P<right>A\d+)_0-1$"
)
SPLIT_ID_FIELDS = (
    "train_evidence_ids",
    "validation_evidence_ids",
    "locked_test_evidence_ids",
)
REVIEWER_VIDEO_FIELDS = (
    "full_local_video",
    "original_local_video",
    "source_local_video",
)


def _split_ids(path: Path, *, day: str) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    split = value.get("split_manifest", value) if isinstance(value, dict) else None
    if not isinstance(split, dict):
        raise ValueError("split audit must contain a split_manifest object")
    ids: list[str] = []
    for field in SPLIT_ID_FIELDS:
        values = split.get(field)
        if not isinstance(values, list):
            raise ValueError(f"split audit is missing {field}")
        ids.extend(str(item).strip() for item in values)
    if any(not evidence_id for evidence_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("split audit evidence IDs must be unique and non-empty")
    selected = [
        evidence_id
        for evidence_id in ids
        if (match := EVIDENCE_ID_PATTERN.fullmatch(evidence_id))
        and match.group("day") == day
    ]
    if not selected:
        raise ValueError(f"split audit contains no {day} evidence IDs")
    return selected


def _agent_dir_from_url(value: Any) -> str:
    url = str(value or "").strip()
    marker = "/resolve/main/"
    path = urlparse(url).path
    if marker not in path:
        raise ValueError(
            "reviewer-imported clip video_url must use the EgoLife resolve/main layout"
        )
    parts = [part for part in path.split(marker, 1)[1].split("/") if part]
    if len(parts) < 3:
        raise ValueError(f"unexpected EgoLife video_url layout: {url}")
    return parts[-3]


def _reviewer_video(clip: dict[str, Any], *, evidence_id: str, user: str) -> str:
    values = [
        Path(str(clip[field])).expanduser().resolve()
        for field in REVIEWER_VIDEO_FIELDS
        if str(clip.get(field) or "").strip()
    ]
    if not values:
        raise ValueError(f"{evidence_id}: imported clip for {user!r} has no full video")
    if len(set(values)) != 1:
        raise ValueError(
            f"{evidence_id}: imported full-video aliases disagree for {user!r}"
        )
    path = values[0]
    if path.suffix.lower() != ".mp4" or not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(
            f"{evidence_id}: imported full video is missing or invalid for {user!r}: {path}"
        )
    return str(path)


def prepare_imported_pairs(
    *,
    import_manifest: str | Path,
    split_audit: str | Path,
    output_path: str | Path,
    day: str = "DAY1",
    expected_count: int | None = None,
    duration_seconds: float = 30.0,
) -> dict[str, Any]:
    """Create ordered, unpruned two-video packets for ``paired_evidence_pruning``."""

    day = str(day).strip().upper()
    if not re.fullmatch(r"DAY\d+", day):
        raise ValueError("day must look like DAY1")
    if expected_count is not None and expected_count <= 0:
        raise ValueError("expected_count must be positive")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")

    import_path = Path(import_manifest)
    audit_path = Path(split_audit)
    target_ids = _split_ids(audit_path, day=day)
    if expected_count is not None and len(target_ids) != expected_count:
        raise ValueError(
            f"split audit must contain {expected_count} {day} IDs; found {len(target_ids)}"
        )

    imported = read_jsonl(import_path)
    by_id: dict[str, dict[str, Any]] = {}
    for row in imported:
        evidence_id = str(row.get("evidence_id") or "").strip()
        if not evidence_id:
            raise ValueError("reviewer import row is missing evidence_id")
        if evidence_id in by_id:
            raise ValueError(f"duplicate reviewer import evidence_id: {evidence_id}")
        by_id[evidence_id] = row
    missing = [evidence_id for evidence_id in target_ids if evidence_id not in by_id]
    if missing:
        raise ValueError(f"reviewer import is missing named {day} IDs: {missing}")

    prepared: list[dict[str, Any]] = []
    for evidence_id in target_ids:
        source = by_id[evidence_id]
        if source.get("schema_version") != IMPORTED_SCHEMA_VERSION:
            raise ValueError(
                f"{evidence_id}: expected schema_version={IMPORTED_SCHEMA_VERSION!r}"
            )
        match = EVIDENCE_ID_PATTERN.fullmatch(evidence_id)
        if match is None or match.group("day") != day:
            raise ValueError(f"unsupported evidence ID: {evidence_id}")
        clips = source.get("clips")
        if not isinstance(clips, list) or len(clips) != 2 or not all(
            isinstance(clip, dict) for clip in clips
        ):
            raise ValueError(f"{evidence_id}: reviewer import must contain two clips")

        expected_agent_ids = [match.group("left"), match.group("right")]
        users: list[str] = []
        prepared_clips: list[dict[str, Any]] = []
        for index, (clip, expected_agent_id) in enumerate(
            zip(clips, expected_agent_ids), start=1
        ):
            user = str(clip.get("agent_name") or clip.get("user") or "").strip()
            if not user:
                raise ValueError(f"{evidence_id}: imported clip {index} has no user")
            agent_dir = _agent_dir_from_url(clip.get("video_url"))
            agent_id = agent_dir.split("_", 1)[0]
            if agent_id != expected_agent_id:
                raise ValueError(
                    f"{evidence_id}: clip {index} is {agent_id}, expected "
                    f"{expected_agent_id}; refusing to swap asker/provider order"
                )
            full_video = _reviewer_video(
                clip, evidence_id=evidence_id, user=user
            )
            current = dict(clip)
            current.update(
                {
                    "agent_dir": agent_dir,
                    "agent_id": agent_id,
                    "agent_name": user,
                    "duration_seconds": float(duration_seconds),
                    "local_video": full_video,
                    "source_local_video": full_video,
                    "original_local_video": full_video,
                    "full_local_video": full_video,
                }
            )
            current.pop("generator_local_video", None)
            current.pop("generator_media_mode", None)
            current.pop("frames", None)
            prepared_clips.append(current)
            users.append(user)
        if len(set(users)) != 2:
            raise ValueError(f"{evidence_id}: asker and provider users must be distinct")

        prepared.append(
            {
                **source,
                "schema_version": PREPARED_SCHEMA_VERSION,
                "day": day,
                "time_token": match.group("time"),
                "required_users": users,
                "speaker_user": users[0],
                "evidence_provider_user": users[1],
                "duration_seconds": float(duration_seconds),
                "candidate_type": "fixed_annotated_full_video_pair_for_production_pruning",
                "clips": prepared_clips,
            }
        )

    output = Path(output_path)
    write_jsonl(output, prepared)
    report = {
        "status": "passed",
        "schema_version": PREPARED_SCHEMA_VERSION,
        "source_import_manifest": str(import_path.resolve()),
        "source_split_audit": str(audit_path.resolve()),
        "output": str(output.resolve()),
        "day": day,
        "packet_count": len(prepared),
        "video_count": 2 * len(prepared),
        "duration_seconds": float(duration_seconds),
        "evidence_ids": target_ids,
        "media_routing": {
            "current_stage": "full videos prepared for production CLIP pruning",
            "generator": "unset until pruning and retained-frame sidecar complete",
            "reviewer": "clips[*].full_local_video",
        },
    }
    report_path = output.with_name(f"{output.stem}_audit.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--import-manifest", type=Path, required=True)
    parser.add_argument("--split-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--day", default="DAY1")
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--duration-seconds", type=float, default=30.0)
    args = parser.parse_args()
    report = prepare_imported_pairs(
        import_manifest=args.import_manifest,
        split_audit=args.split_audit,
        output_path=args.output,
        day=args.day,
        expected_count=args.expected_count,
        duration_seconds=args.duration_seconds,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(
        "IMPORTED_PAIR_CONTRACT_PASSED "
        f"day={report['day']} packets={report['packet_count']} "
        f"videos={report['video_count']}"
    )


if __name__ == "__main__":
    main()
