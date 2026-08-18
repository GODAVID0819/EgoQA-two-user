"""Materialize four additional videos for each two-video packet.

The 300-packet CLIP-pruning job stores the selected two full (unpruned) videos
inside each packet. This module fetches one video for each of the four
non-selected EgoLife participants and writes a six-video manifest for later
model evaluation. Exact DAY/time matches are preferred. When an exact view is
absent, the default policy deterministically samples an available video from
that same participant, preferring the packet's day.

No question filtering or ambiguity checking is performed here.
"""

from __future__ import annotations

import argparse
import errno
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .evidence import local_cache_path
from .io_utils import download_file, iter_jsonl, read_json, write_json, write_jsonl
from .manifest import AGENTS


DEFAULT_PACKET_COUNT = 300
DEFAULT_OUTPUT_SUBDIR = "remaining_four_full_videos"
DEFAULT_FALLBACK_RANDOM_SEED = 42
SIX_AGENT_DIRS = tuple(AGENTS)
_SAFE_COMPONENT_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _nonempty_file(path: str | Path | None) -> bool:
    if not path:
        return False
    candidate = Path(str(path))
    return candidate.is_file() and candidate.stat().st_size > 0


def _safe_component(value: Any) -> str:
    text = _SAFE_COMPONENT_RE.sub("_", str(value or "").strip()).strip("._")
    if not text:
        raise ValueError(f"cannot create a directory name from {value!r}")
    return text


def _manifest_index(
    manifest: dict[str, Any],
) -> dict[tuple[str, str], dict[str, dict[str, Any]]]:
    index: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for clip_index, clip in enumerate(manifest.get("clips", [])):
        if not isinstance(clip, dict):
            raise ValueError(f"manifest clip {clip_index} is not an object")
        day = str(clip.get("day") or "")
        time_token = str(clip.get("time_token") or "")
        agent_dir = str(clip.get("agent_dir") or "")
        if not day or not time_token or not agent_dir:
            raise ValueError(
                f"manifest clip {clip_index} is missing day, time_token, or agent_dir"
            )
        group = index.setdefault((day, time_token), {})
        if agent_dir in group:
            raise ValueError(
                f"manifest contains duplicate clip for {day}/{time_token}/{agent_dir}"
            )
        group[agent_dir] = clip
    return index


def _manifest_clips_by_agent(
    manifest_index: dict[tuple[str, str], dict[str, dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    clips_by_agent = {agent_dir: [] for agent_dir in SIX_AGENT_DIRS}
    for group in manifest_index.values():
        for agent_dir, clip in group.items():
            if agent_dir in clips_by_agent:
                clips_by_agent[agent_dir].append(clip)
    for clips in clips_by_agent.values():
        clips.sort(
            key=lambda clip: (
                str(clip.get("day") or ""),
                str(clip.get("time_token") or ""),
                str(clip.get("video_path") or ""),
            )
        )
    return clips_by_agent


def _sample_fallback_clip(
    *,
    packet_id: str,
    target_day: str,
    target_time_token: str,
    agent_dir: str,
    clips_by_agent: dict[str, list[dict[str, Any]]],
    fallback_random_seed: int,
) -> tuple[dict[str, Any], str]:
    candidates = [
        clip
        for clip in clips_by_agent.get(agent_dir, [])
        if (
            str(clip.get("day") or ""),
            str(clip.get("time_token") or ""),
        )
        != (target_day, target_time_token)
    ]
    same_day = [
        clip for clip in candidates if str(clip.get("day") or "") == target_day
    ]
    pool = same_day or candidates
    if not pool:
        raise ValueError(
            f"{packet_id}: no fallback video is available for missing agent {agent_dir}"
        )
    selected_index = fallback_random_seed % len(pool)
    alignment = (
        "fallback_sampled_same_day" if same_day else "fallback_sampled_other_day"
    )
    return pool[selected_index], alignment


def _selected_pair(packet: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    packet_id = str(packet.get("evidence_id") or "<missing evidence_id>")
    clips = packet.get("clips")
    if not isinstance(clips, list) or len(clips) != 2:
        count = len(clips) if isinstance(clips, list) else 0
        raise ValueError(f"{packet_id}: expected exactly 2 selected clips, found {count}")
    if not all(isinstance(clip, dict) for clip in clips):
        raise ValueError(f"{packet_id}: selected clips must be objects")
    agent_dirs = [str(clip.get("agent_dir") or "") for clip in clips]
    if any(not agent_dir for agent_dir in agent_dirs):
        raise ValueError(f"{packet_id}: a selected clip is missing agent_dir")
    if len(set(agent_dirs)) != 2:
        raise ValueError(f"{packet_id}: selected clips do not have two distinct agents")
    return clips[0], clips[1]


def _selected_full_video(packet_id: str, clip: dict[str, Any]) -> Path:
    """Resolve only an explicitly unpruned selected-pair video."""

    for key in ("full_local_video", "original_local_video", "source_local_video"):
        value = clip.get(key)
        if _nonempty_file(value):
            return Path(str(value))
    agent_dir = clip.get("agent_dir")
    raise FileNotFoundError(
        f"{packet_id}: selected agent {agent_dir} has no existing unpruned video in "
        "full_local_video, original_local_video, or source_local_video"
    )


def _source_filename(clip: dict[str, Any]) -> str:
    video_path = clip.get("video_path")
    if not video_path:
        raise ValueError(
            "manifest clip is missing video_path for "
            f"{clip.get('day')}/{clip.get('time_token')}/{clip.get('agent_dir')}"
        )
    filename = Path(str(video_path)).name
    if not filename:
        raise ValueError(f"manifest video_path has no filename: {video_path}")
    return filename


def _build_packet_specs(
    packets: list[dict[str, Any]],
    *,
    manifest_index: dict[tuple[str, str], dict[str, dict[str, Any]]],
    clips_by_agent: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    missing_view_policy: str,
    fallback_random_seed: int,
) -> list[dict[str, Any]]:
    """Resolve every six-video row before downloading any media."""

    seen_ids: set[str] = set()
    seen_groups: set[tuple[str, str]] = set()
    seen_packet_dirs: set[str] = set()
    specs = []

    for packet_index, packet in enumerate(packets):
        packet_id = str(packet.get("evidence_id") or "")
        if not packet_id:
            raise ValueError(f"packet {packet_index} is missing evidence_id")
        if packet_id in seen_ids:
            raise ValueError(f"duplicate evidence_id: {packet_id}")
        seen_ids.add(packet_id)

        day = str(packet.get("day") or "")
        time_token = str(packet.get("time_token") or "")
        if not day or not time_token:
            raise ValueError(f"{packet_id}: packet is missing day or time_token")
        group_key = (day, time_token)
        if group_key in seen_groups:
            raise ValueError(
                f"{packet_id}: duplicate synchronized packet group {day}/{time_token}"
            )
        seen_groups.add(group_key)

        pair = _selected_pair(packet)
        pair_by_agent = {str(clip["agent_dir"]): clip for clip in pair}
        unknown_pair_agents = sorted(set(pair_by_agent) - set(SIX_AGENT_DIRS))
        if unknown_pair_agents:
            raise ValueError(
                f"{packet_id}: selected pair contains unknown agents {unknown_pair_agents}"
            )

        manifest_group = manifest_index.get(group_key, {})
        missing_agents = [
            agent_dir for agent_dir in SIX_AGENT_DIRS if agent_dir not in manifest_group
        ]
        extra_agents = sorted(set(manifest_group) - set(SIX_AGENT_DIRS))
        if extra_agents:
            raise ValueError(
                f"{packet_id}: exact manifest group {day}/{time_token} contains "
                f"unknown participants: {extra_agents}"
            )
        missing_context_agents = [
            agent_dir for agent_dir in missing_agents if agent_dir not in pair_by_agent
        ]
        if missing_context_agents and missing_view_policy == "error":
            raise ValueError(
                f"{packet_id}: exact manifest group {day}/{time_token} does not contain "
                f"all four context participants; missing={missing_context_agents}"
            )

        packet_component = _safe_component(packet_id)
        if packet_component in seen_packet_dirs:
            raise ValueError(
                f"{packet_id}: sanitized packet directory collides with another evidence_id"
            )
        seen_packet_dirs.add(packet_component)
        packet_dir = output_dir / "packets" / packet_component

        selected_rows = []
        remaining_rows = []
        all_rows = []
        for slot, agent_dir in enumerate(SIX_AGENT_DIRS, 1):
            selected_clip = pair_by_agent.get(agent_dir)
            if selected_clip is not None:
                manifest_clip = manifest_group.get(agent_dir, selected_clip)
                alignment = "selected_pair_exact"
            elif agent_dir in manifest_group:
                manifest_clip = manifest_group[agent_dir]
                alignment = "exact_synchronized"
            else:
                manifest_clip, alignment = _sample_fallback_clip(
                    packet_id=packet_id,
                    target_day=day,
                    target_time_token=time_token,
                    agent_dir=agent_dir,
                    clips_by_agent=clips_by_agent,
                    fallback_random_seed=fallback_random_seed,
                )
            base_row = {
                "slot": slot,
                "agent_dir": agent_dir,
                "agent_id": manifest_clip.get("agent_id"),
                "agent_name": manifest_clip.get("agent_name") or AGENTS[agent_dir],
                "target_day": day,
                "target_time_token": time_token,
                "day": manifest_clip.get("day"),
                "time_token": manifest_clip.get("time_token"),
                "clip_clock": manifest_clip.get("clip_clock"),
                "video_path": manifest_clip.get("video_path"),
                "video_url": manifest_clip.get("video_url"),
                "alignment": alignment,
                "synchronized_with_selected_pair": alignment
                in {"selected_pair_exact", "exact_synchronized"},
            }
            if selected_clip is not None:
                full_video = _selected_full_video(packet_id, selected_clip)
                row = {
                    **base_row,
                    "role": "selected_pair",
                    "local_video": str(full_video),
                    "source": "existing_unpruned_pair_video",
                }
                selected_rows.append(row)
            else:
                if not manifest_clip.get("video_url"):
                    raise ValueError(
                        f"{packet_id}: remaining agent {agent_dir} is missing video_url"
                    )
                filename = _source_filename(manifest_clip)
                destination = packet_dir / filename
                row = {
                    **base_row,
                    "role": "remaining_context",
                    "local_video": str(destination),
                    "cache_local_video": None,
                    "source": (
                        "manifest_exact_day_time_join"
                        if alignment == "exact_synchronized"
                        else "deterministic_same_participant_fallback_sample"
                    ),
                }
                remaining_rows.append(row)
            all_rows.append(row)

        if len(selected_rows) != 2 or len(remaining_rows) != 4:
            raise AssertionError(
                f"{packet_id}: expected 2 selected and 4 remaining videos, found "
                f"{len(selected_rows)} and {len(remaining_rows)}"
            )
        specs.append(
            {
                "packet_index": packet_index,
                "evidence_id": packet_id,
                "day": day,
                "time_token": time_token,
                "clip_clock": packet.get("clip_clock"),
                "packet_directory": str(packet_dir),
                "selected_pair_agent_dirs": [
                    str(clip["agent_dir"]) for clip in pair
                ],
                "remaining_agent_dirs": [
                    row["agent_dir"] for row in remaining_rows
                ],
                "selected_pair_full_clips": selected_rows,
                "remaining_full_clips": remaining_rows,
                "all_six_full_clips": all_rows,
            }
        )
    return specs


def _atomic_materialize(
    source: Path,
    destination: Path,
    *,
    mode: str,
) -> str:
    if not _nonempty_file(source):
        raise FileNotFoundError(f"source video is missing or empty: {source}")
    if destination.exists():
        if not destination.is_file() or destination.stat().st_size <= 0:
            raise ValueError(f"existing destination is not a non-empty file: {destination}")
        if destination.stat().st_size != source.stat().st_size:
            raise ValueError(
                f"existing destination size differs from source: {destination}"
            )
        return "existing"

    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f"{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(fd)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        if mode == "copy":
            shutil.copy2(source, temporary)
            materialized_as = "copy"
        elif mode == "hardlink":
            try:
                os.link(source, temporary)
                materialized_as = "hardlink"
            except OSError as exc:
                if exc.errno not in {
                    errno.EXDEV,
                    errno.EPERM,
                    errno.EACCES,
                    errno.ENOTSUP,
                }:
                    raise
                shutil.copy2(source, temporary)
                materialized_as = "copy_fallback"
        else:
            raise ValueError(f"unsupported materialization mode: {mode}")
        temporary.replace(destination)
        return materialized_as
    finally:
        temporary.unlink(missing_ok=True)


def prepare_six_view_packets(
    *,
    evidence_path: str | Path,
    manifest_path: str | Path,
    cache_dir: str | Path,
    output_dir: str | Path | None = None,
    expected_packet_count: int = DEFAULT_PACKET_COUNT,
    download_missing: bool = True,
    materialize_mode: str = "hardlink",
    missing_view_policy: str = "sample",
    fallback_random_seed: int = DEFAULT_FALLBACK_RANDOM_SEED,
) -> dict[str, Any]:
    """Fetch and store four additional same-participant videos per pair packet."""

    if expected_packet_count <= 0:
        raise ValueError("expected_packet_count must be positive")
    if materialize_mode not in {"hardlink", "copy"}:
        raise ValueError("materialize_mode must be 'hardlink' or 'copy'")
    if missing_view_policy not in {"sample", "error"}:
        raise ValueError("missing_view_policy must be 'sample' or 'error'")

    evidence_path = Path(evidence_path)
    manifest_path = Path(manifest_path)
    cache_dir = Path(cache_dir)
    output_dir = (
        Path(output_dir)
        if output_dir is not None
        else evidence_path.parent / DEFAULT_OUTPUT_SUBDIR
    )

    packets = list(iter_jsonl(evidence_path))
    if len(packets) != expected_packet_count:
        raise ValueError(
            f"expected exactly {expected_packet_count} evidence packets, found {len(packets)}"
        )
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError(f"manifest must be a JSON object: {manifest_path}")
    manifest_index = _manifest_index(manifest)
    specs = _build_packet_specs(
        packets,
        manifest_index=manifest_index,
        clips_by_agent=_manifest_clips_by_agent(manifest_index),
        output_dir=output_dir,
        missing_view_policy=missing_view_policy,
        fallback_random_seed=fallback_random_seed,
    )

    stats = {
        "downloaded_video_count": 0,
        "reused_cache_video_count": 0,
        "existing_output_video_count": 0,
        "hardlinked_video_count": 0,
        "copied_video_count": 0,
        "copy_fallback_video_count": 0,
    }
    output_rows = []
    for spec in specs:
        remaining_by_agent = {
            row["agent_dir"]: row for row in spec["remaining_full_clips"]
        }
        for agent_dir, row in remaining_by_agent.items():
            video_path = str(row["video_path"])
            cache_video = local_cache_path(cache_dir, video_path)
            row["cache_local_video"] = str(cache_video)
            if _nonempty_file(cache_video):
                stats["reused_cache_video_count"] += 1
            else:
                if not download_missing:
                    raise FileNotFoundError(
                        f"{spec['evidence_id']}: remaining agent {agent_dir} is not cached at "
                        f"{cache_video}; rerun without --no-download"
                    )
                download_file(str(row["video_url"]), cache_video)
                if not _nonempty_file(cache_video):
                    raise RuntimeError(
                        f"download did not produce a non-empty video: {cache_video}"
                    )
                stats["downloaded_video_count"] += 1

            status = _atomic_materialize(
                cache_video,
                Path(str(row["local_video"])),
                mode=materialize_mode,
            )
            row["materialization"] = status
            if status == "existing":
                stats["existing_output_video_count"] += 1
            elif status == "hardlink":
                stats["hardlinked_video_count"] += 1
            elif status == "copy":
                stats["copied_video_count"] += 1
            elif status == "copy_fallback":
                stats["copy_fallback_video_count"] += 1

        output_rows.append(spec)

    output_dir.mkdir(parents=True, exist_ok=True)
    packet_manifest = output_dir / "six_view_packet_manifest.jsonl"
    temporary_manifest = packet_manifest.with_name(f"{packet_manifest.name}.tmp")
    temporary_manifest.unlink(missing_ok=True)
    try:
        written = write_jsonl(temporary_manifest, output_rows)
        if written != expected_packet_count:
            raise RuntimeError(
                f"expected to write {expected_packet_count} six-view rows, wrote {written}"
            )
        temporary_manifest.replace(packet_manifest)
    finally:
        temporary_manifest.unlink(missing_ok=True)

    summary = {
        "packet_count": len(output_rows),
        "selected_pair_video_count": len(output_rows) * 2,
        "remaining_video_count": len(output_rows) * 4,
        "all_six_video_count": len(output_rows) * 6,
        "expected_agent_order": list(SIX_AGENT_DIRS),
        "exact_join_key": ["day", "time_token", "agent_dir"],
        "missing_view_policy": missing_view_policy,
        "fallback_random_seed": fallback_random_seed,
        "exact_synchronized_remaining_video_count": sum(
            row["alignment"] == "exact_synchronized"
            for packet in output_rows
            for row in packet["remaining_full_clips"]
        ),
        "fallback_sampled_remaining_video_count": sum(
            str(row["alignment"]).startswith("fallback_sampled_")
            for packet in output_rows
            for row in packet["remaining_full_clips"]
        ),
        "ambiguity_check": False,
        "evidence_path": str(evidence_path),
        "manifest_path": str(manifest_path),
        "cache_dir": str(cache_dir),
        "output_dir": str(output_dir),
        "packet_manifest": str(packet_manifest),
        "materialize_mode": materialize_mode,
        **stats,
    }
    write_json(output_dir / "preparation_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch one additional EgoLife video for each non-selected participant "
            "in every two-video evidence packet"
        )
    )
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument(
        "--output-dir",
        help=(
            "Defaults to <evidence directory>/"
            f"{DEFAULT_OUTPUT_SUBDIR}"
        ),
    )
    parser.add_argument(
        "--expected-packet-count",
        type=int,
        default=DEFAULT_PACKET_COUNT,
    )
    parser.add_argument(
        "--materialize-mode",
        choices=["hardlink", "copy"],
        default="hardlink",
        help="Hardlink cached videos when possible (default), or make full copies",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Require every remaining video to exist in the cache",
    )
    parser.add_argument(
        "--missing-view-policy",
        choices=["sample", "error"],
        default="sample",
        help=(
            "Sample an available video from the same participant when the exact "
            "DAY/time view is absent (default), or fail"
        ),
    )
    parser.add_argument(
        "--fallback-random-seed",
        type=int,
        default=DEFAULT_FALLBACK_RANDOM_SEED,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = prepare_six_view_packets(
        evidence_path=args.evidence,
        manifest_path=args.manifest,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        expected_packet_count=args.expected_packet_count,
        download_missing=not args.no_download,
        materialize_mode=args.materialize_mode,
        missing_view_policy=args.missing_view_policy,
        fallback_random_seed=args.fallback_random_seed,
    )
    print(
        f"prepared {summary['packet_count']} six-view packet mappings with "
        f"{summary['remaining_video_count']} stored remaining videos at "
        f"{summary['output_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
