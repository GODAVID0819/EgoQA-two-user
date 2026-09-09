"""Prepare exact 10-minute media manifests for the curated six-vs-required test.

The preparer reuses the 10-minute composites from job 16463998 whenever they
exist and have the expected duration.  For older questions whose generator job
only retained a 30-second asset, it materializes the same 10-minute window once
from the existing EgoLife cache.  An opt-in fallback downloads only missing
30-second source clips from the official EgoLife dataset, validates them, and
then assembles the composite.  Optional skip mode records evidence that cannot
be recovered after all enabled fallbacks and continues.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

try:
    from egolife_two_user_qa.io_utils import download_file, hf_resolve_url
except ModuleNotFoundError:
    # Keep direct execution from the package directory working for local checks.
    from io_utils import download_file, hf_resolve_url


AGENTS = (
    ("A1", "A1_JAKE", "Jake"),
    ("A2", "A2_ALICE", "Alice"),
    ("A3", "A3_TASHA", "Tasha"),
    ("A4", "A4_LUCIA", "Lucia"),
    ("A5", "A5_KATRINA", "Katrina"),
    ("A6", "A6_SHURE", "Shure"),
)
AGENT_BY_NAME = {name: (agent_id, agent_dir) for agent_id, agent_dir, name in AGENTS}
AGENT_NUMBER_BY_NAME = {name: index for index, (_, _, name) in enumerate(AGENTS, 1)}
GROUP_RE = re.compile(r"^(DAY(?P<day>[1-9][0-9]*))::(?P<token>[0-9]{8})$")
SPEAKER_RE = re.compile(r"_S(?P<speaker>[1-6])$")


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def parse_window(
    row: dict[str, Any],
    *,
    evidence_id: str | None = None,
) -> tuple[str, int, str, int]:
    group = str(row.get("generation_group") or "")
    match = GROUP_RE.fullmatch(group)
    if match is None:
        raise ValueError(f"{row.get('qa_id')}: invalid generation_group {group!r}")
    evidence_id = str(evidence_id or row.get("evidence_id") or "")
    speaker_match = SPEAKER_RE.search(evidence_id)
    if speaker_match is None:
        raise ValueError(f"{row.get('qa_id')}: cannot parse speaker from {evidence_id!r}")
    day_label = match.group(1)
    return day_label, int(match.group("day")), match.group("token"), int(
        speaker_match.group("speaker")
    )


def time_tokens(start_token: str, count: int = 20) -> list[str]:
    hour = int(start_token[0:2])
    minute = int(start_token[2:4])
    second = int(start_token[4:6])
    centiseconds = int(start_token[6:8])
    start = datetime(2000, 1, 1, hour, minute, second, centiseconds * 10_000)
    tokens = []
    for index in range(count):
        value = start + timedelta(seconds=30 * index)
        if value.day != start.day:
            raise ValueError(f"10-minute window crosses midnight: {start_token}")
        tokens.append(
            f"{value.hour:02d}{value.minute:02d}{value.second:02d}"
            f"{value.microsecond // 10_000:02d}"
        )
    return tokens


def probe_duration(path: Path, ffprobe: str) -> float:
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def is_ten_minute(path: Path, ffprobe: str, minimum: float, maximum: float) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        duration = probe_duration(path, ffprobe)
    except (OSError, ValueError, subprocess.CalledProcessError):
        return False
    return minimum <= duration <= maximum


def is_source_clip(path: Path, ffprobe: str) -> bool:
    """Return whether a cached/downloaded source segment is decodable."""

    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        return probe_duration(path, ffprobe) > 0
    except (OSError, ValueError, subprocess.CalledProcessError):
        return False


def source_position(agent_number: int, speaker_number: int) -> int:
    if agent_number == speaker_number:
        return 0
    remaining = [number for number in range(1, 7) if number != speaker_number]
    return 1 + remaining.index(agent_number)


def reused_composite_path(
    root: Path,
    *,
    day_label: str,
    start_token: str,
    speaker_number: int,
    agent_number: int,
    agent_dir: str,
) -> Path:
    position = source_position(agent_number, speaker_number)
    return (
        root
        / f"{day_label}_{start_token}"
        / "six_user_speaker_consensus"
        / f"speaker_{speaker_number:02d}"
        / f"{position:02d}_{agent_dir}_full.mp4"
    )


def concat_escape(path: Path) -> str:
    return str(path.resolve()).replace("'", "'\\''")


def materialize_from_cache(
    *,
    cache_root: Path,
    output_root: Path,
    day_label: str,
    day_number: int,
    start_token: str,
    agent_dir: str,
    ffmpeg: str,
    ffprobe: str,
    minimum_duration: float,
    maximum_duration: float,
    download_missing: bool,
    download_cache_root: Path,
    dataset: str,
    dataset_revision: str,
) -> tuple[Path, list[str], int]:
    output = output_root / f"{day_label}_{start_token}" / f"{agent_dir}_10min.mp4"
    if is_ten_minute(output, ffprobe, minimum_duration, maximum_duration):
        return output, ["existing_materialized_composite"], 0

    tokens = time_tokens(start_token)
    sources = []
    missing = []
    downloaded_count = 0
    for token in tokens:
        filename = f"{day_label}_{agent_dir}_{token}.mp4"
        cached = cache_root / agent_dir / f"DAY_{day_number}" / filename
        if is_source_clip(cached, ffprobe):
            sources.append(cached)
            continue
        if not download_missing:
            missing.append(str(cached))
            continue

        downloaded = (
            download_cache_root / agent_dir / f"DAY_{day_number}" / filename
        )
        if not is_source_clip(downloaded, ffprobe):
            url = hf_resolve_url(
                dataset,
                f"{agent_dir}/{day_label}/{filename}",
                revision=dataset_revision,
            )
            download_file(url, downloaded)
            downloaded_count += 1
        if not is_source_clip(downloaded, ffprobe):
            raise RuntimeError(
                f"downloaded source clip is not decodable: {downloaded}"
            )
        sources.append(downloaded)

    if missing:
        raise FileNotFoundError(
            f"missing {len(missing)} cached source clips for "
            f"{day_label}_{start_token}/{agent_dir}: {missing[:3]}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    concat_list = output.with_suffix(".concat.txt")
    temporary = output.with_suffix(".tmp.mp4")
    concat_list.write_text(
        "".join(f"file '{concat_escape(path)}'\n" for path in sources),
        encoding="utf-8",
        newline="\n",
    )
    temporary.unlink(missing_ok=True)
    try:
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list),
                "-map",
                "0:v:0",
                "-an",
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(temporary),
            ],
            check=True,
        )
        if not is_ten_minute(temporary, ffprobe, minimum_duration, maximum_duration):
            duration = probe_duration(temporary, ffprobe) if temporary.is_file() else None
            raise RuntimeError(
                f"concatenated video has unexpected duration: {temporary} duration={duration}"
            )
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
        concat_list.unlink(missing_ok=True)
    return output, [str(path) for path in sources], downloaded_count


def clip_row(
    *,
    user: str,
    path: Path,
    source_kind: str,
    source_paths: list[str],
    duration_seconds: float,
) -> dict[str, Any]:
    agent_id, agent_dir = AGENT_BY_NAME[user]
    return {
        "agent_id": agent_id,
        "agent_dir": agent_dir,
        "agent_name": user,
        "full_local_video": str(path),
        "original_local_video": str(path),
        "source_local_video": str(path),
        "source_kind": source_kind,
        "source_paths": source_paths,
        "duration_seconds": round(duration_seconds, 3),
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    qa_rows = iter_jsonl(args.qa)
    if len(qa_rows) != args.expected_count:
        raise ValueError(
            f"expected {args.expected_count} curated QAs, found {len(qa_rows)} in {args.qa}"
        )

    evidence_rows = []
    six_view_rows = []
    provenance_rows = []
    skipped_rows = []
    materialized_cache: dict[
        tuple[str, str, str], tuple[Path, list[str], int]
    ] = {}
    seen_evidence_ids = set()

    for qa in qa_rows:
        qa_id = str(qa.get("qa_id") or "")
        source_evidence_id = str(qa.get("evidence_id") or "")
        evidence_id = (
            f"{source_evidence_id}__MINPAIR__{qa_id}"
            if args.qa_specific_evidence_ids
            else source_evidence_id
        )
        if not qa_id or not source_evidence_id:
            raise ValueError("every curated row needs qa_id and evidence_id")
        if evidence_id in seen_evidence_ids:
            raise ValueError(f"duplicate evidence_id: {evidence_id}")
        seen_evidence_ids.add(evidence_id)

        required_users = qa.get("required_users")
        if (
            not isinstance(required_users, list)
            or len(required_users) != 2
            or len(set(required_users)) != 2
            or any(user not in AGENT_BY_NAME for user in required_users)
        ):
            raise ValueError(f"{qa_id}: required_users must contain two known users")

        day_label, day_number, start_token, speaker_number = parse_window(
            qa,
            evidence_id=source_evidence_id,
        )
        asker = str(required_users[0])
        if AGENT_NUMBER_BY_NAME[asker] != speaker_number:
            raise ValueError(
                f"{qa_id}: asker {asker} does not match evidence speaker S{speaker_number}"
            )

        resolved: dict[str, dict[str, Any]] = {}
        users_to_resolve = (
            [name for _, _, name in AGENTS]
            if args.scope == "six"
            else [str(user) for user in required_users]
        )
        try:
            for agent_number, (_, agent_dir, user) in enumerate(AGENTS, 1):
                if user not in users_to_resolve:
                    continue
                reused = reused_composite_path(
                    args.ten_minute_asset_root,
                    day_label=day_label,
                    start_token=start_token,
                    speaker_number=speaker_number,
                    agent_number=agent_number,
                    agent_dir=agent_dir,
                )
                if is_ten_minute(
                    reused,
                    args.ffprobe,
                    args.minimum_duration,
                    args.maximum_duration,
                ):
                    path = reused
                    source_kind = "reused_job_16463998_ten_minute_composite"
                    source_paths = [str(reused)]
                else:
                    cache_key = (day_label, start_token, agent_dir)
                    if cache_key not in materialized_cache:
                        materialized_cache[cache_key] = materialize_from_cache(
                            cache_root=args.source_cache_root,
                            output_root=args.materialized_media_root,
                            day_label=day_label,
                            day_number=day_number,
                            start_token=start_token,
                            agent_dir=agent_dir,
                            ffmpeg=args.ffmpeg,
                            ffprobe=args.ffprobe,
                            minimum_duration=args.minimum_duration,
                            maximum_duration=args.maximum_duration,
                            download_missing=args.download_missing,
                            download_cache_root=args.download_cache_root,
                            dataset=args.dataset,
                            dataset_revision=args.dataset_revision,
                        )
                    path, source_paths, downloaded_count = materialized_cache[cache_key]
                    source_kind = (
                        "materialized_with_downloaded_30_second_source_clips"
                        if downloaded_count
                        else "materialized_from_existing_30_second_cache_clips"
                    )
                duration = probe_duration(path, args.ffprobe)
                resolved[user] = clip_row(
                    user=user,
                    path=path,
                    source_kind=source_kind,
                    source_paths=source_paths,
                    duration_seconds=duration,
                )
        except (
            FileNotFoundError,
            OSError,
            RuntimeError,
            ValueError,
            subprocess.CalledProcessError,
        ) as exc:
            if not getattr(args, "skip_unavailable", False):
                raise
            skipped_rows.append(
                {
                    "qa_id": qa_id,
                    "evidence_id": evidence_id,
                    "source_evidence_id": source_evidence_id,
                    "generation_group": qa.get("generation_group"),
                    "required_users": required_users,
                    "preparation_scope": args.scope,
                    "status": "skipped_unavailable_media",
                    "error_type": type(exc).__name__,
                    "reason": str(exc),
                }
            )
            continue

        pair_clips = [resolved[str(user)] for user in required_users]
        remaining = [
            {
                **resolved[user],
                "local_video": resolved[user]["full_local_video"],
                "alignment": "exact_synchronized",
                "synchronized_with_selected_pair": True,
                "day": day_label,
                "time_token": start_token,
            }
            for _, _, user in AGENTS
            if user not in required_users
        ] if args.scope == "six" else []
        evidence_rows.append(
            {
                "evidence_id": evidence_id,
                "source_evidence_id": source_evidence_id,
                "generation_group": qa["generation_group"],
                "required_users": required_users,
                "clips": pair_clips,
            }
        )
        if args.scope == "six":
            six_view_rows.append(
                {
                    "evidence_id": evidence_id,
                    "source_evidence_id": source_evidence_id,
                    "remaining_full_clips": remaining,
                }
            )
        provenance_rows.append(
            {
                "qa_id": qa_id,
                "evidence_id": evidence_id,
                "source_evidence_id": source_evidence_id,
                "generation_group": qa["generation_group"],
                "required_users": required_users,
                "preparation_scope": args.scope,
                "views": [resolved[user] for user in users_to_resolve],
            }
        )

    write_jsonl(args.evidence_output, evidence_rows)
    write_jsonl(args.six_view_output, six_view_rows)
    write_jsonl(args.provenance_output, provenance_rows)
    skipped_output = getattr(args, "skipped_output", None)
    if skipped_output is not None:
        write_jsonl(skipped_output, skipped_rows)
    summary = {
        "preparation_scope": args.scope,
        "qa_count": len(qa_rows),
        "evidence_count": len(evidence_rows),
        "skipped_evidence_count": len(skipped_rows),
        "pair_video_count": sum(len(row["clips"]) for row in evidence_rows),
        "context_video_count": sum(
            len(row["remaining_full_clips"]) for row in six_view_rows
        ),
        "unique_materialized_view_count": len(materialized_cache),
        "downloaded_source_clip_count": sum(
            downloaded_count
            for _, _, downloaded_count in materialized_cache.values()
        ),
        "download_missing": bool(args.download_missing),
        "download_cache_root": str(args.download_cache_root),
        "dataset": args.dataset,
        "dataset_revision": args.dataset_revision,
        "ten_minute_asset_root": str(args.ten_minute_asset_root),
        "source_cache_root": str(args.source_cache_root),
        "evidence_output": str(args.evidence_output),
        "six_view_output": str(args.six_view_output),
        "provenance_output": str(args.provenance_output),
        "skipped_output": str(skipped_output) if skipped_output is not None else None,
    }
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", type=Path, required=True)
    parser.add_argument("--ten-minute-asset-root", type=Path, required=True)
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument(
        "--scope",
        choices=("six", "pair"),
        default="six",
        help="Prepare all six views or only each question's required pair.",
    )
    parser.add_argument(
        "--qa-specific-evidence-ids",
        action="store_true",
        help=(
            "Give every QA a unique prepared evidence ID; required when repeated "
            "source windows use different minimal user pairs."
        ),
    )
    parser.add_argument("--materialized-media-root", type=Path, required=True)
    parser.add_argument("--evidence-output", type=Path, required=True)
    parser.add_argument("--six-view-output", type=Path, required=True)
    parser.add_argument("--provenance-output", type=Path, required=True)
    parser.add_argument("--skipped-output", type=Path)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--ffprobe", required=True)
    parser.add_argument(
        "--download-missing",
        action="store_true",
        help="Download missing 30-second clips from the configured EgoLife dataset.",
    )
    parser.add_argument("--download-cache-root", type=Path)
    parser.add_argument("--dataset", default="lmms-lab/EgoLife")
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--expected-count", type=int, default=17)
    parser.add_argument("--minimum-duration", type=float, default=570.0)
    parser.add_argument("--maximum-duration", type=float, default=630.0)
    parser.add_argument(
        "--skip-unavailable",
        action="store_true",
        help=(
            "Record and skip evidence whose existing and cached media cannot be prepared."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.skip_unavailable and args.skipped_output is None:
        raise SystemExit("--skip-unavailable requires --skipped-output")
    if args.download_cache_root is None:
        args.download_cache_root = args.materialized_media_root / "downloaded_source_clips"
    if args.qa_specific_evidence_ids and args.scope != "pair":
        raise SystemExit("--qa-specific-evidence-ids is only valid with --scope pair")
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(prepare(args), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
