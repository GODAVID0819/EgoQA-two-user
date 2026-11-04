"""Build a video/gaze manifest for EgoLife two-user QA generation."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

from .io_utils import fetch_json, hf_resolve_url, stable_id, write_json


VIDEO_DATASET = "lmms-lab/EgoLife"
GAZE_DATASET = "Wangtwohappy/EgoLife_EyeTracking_EyeGaze"
DEFAULT_REVISION = "main"

AGENTS = {
    "A1_JAKE": "Jake",
    "A2_ALICE": "Alice",
    "A3_TASHA": "Tasha",
    "A4_LUCIA": "Lucia",
    "A5_KATRINA": "Katrina",
    "A6_SHURE": "Shure",
}

FILENAME_RE = re.compile(
    r"(?P<day>DAY[1-7])_"
    r"(?P<agent_id>A[1-6])_"
    r"(?P<agent_name>[A-Z]+)_"
    r"(?P<time_token>\d{8})"
    r"\.(?P<ext>mp4|csv)$"
)


@dataclass(frozen=True)
class EgoLifeKey:
    day: str
    agent_dir: str
    agent_id: str
    agent_name: str
    time_token: str
    ext: str

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.day, self.agent_dir, self.time_token)

    @property
    def clip_clock(self) -> str:
        token = self.time_token
        return f"{token[0:2]}:{token[2:4]}:{token[4:6]}.{token[6:8]}"


def parse_egolife_path(repo_path: str) -> EgoLifeKey:
    """Parse paths like A1_JAKE/DAY1/DAY1_A1_JAKE_11094208.mp4."""

    parts = Path(repo_path).parts
    if len(parts) < 3:
        raise ValueError(f"Not an EgoLife clip path: {repo_path}")
    agent_dir = parts[-3]
    filename = parts[-1]
    match = FILENAME_RE.fullmatch(filename)
    if not match:
        raise ValueError(f"Not an EgoLife clip filename: {repo_path}")
    day = match.group("day")
    agent_id = match.group("agent_id")
    agent_name = match.group("agent_name").title()
    expected_dir = f"{agent_id}_{agent_name.upper()}"
    if agent_dir != expected_dir:
        raise ValueError(f"Agent dir mismatch in {repo_path}: expected {expected_dir}")
    return EgoLifeKey(
        day=day,
        agent_dir=agent_dir,
        agent_id=agent_id,
        agent_name=agent_name,
        time_token=match.group("time_token"),
        ext=match.group("ext"),
    )


def seconds_from_time_token(time_token: str) -> float:
    """Return seconds since midnight for an EgoLife HHMMSScc token."""

    if len(time_token) != 8 or not time_token.isdigit():
        raise ValueError(f"Invalid time token: {time_token}")
    hours = int(time_token[0:2])
    minutes = int(time_token[2:4])
    seconds = int(time_token[4:6])
    centiseconds = int(time_token[6:8])
    return hours * 3600 + minutes * 60 + seconds + centiseconds / 100.0


def hf_tree(dataset: str, repo_path: str, *, revision: str = DEFAULT_REVISION) -> list[dict[str, Any]]:
    quoted = quote(repo_path.strip("/"))
    url = f"https://huggingface.co/api/datasets/{dataset}/tree/{revision}/{quoted}?recursive=1&expand=false"
    data = fetch_json(url)
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected Hugging Face tree response for {dataset}/{repo_path}")
    return data


def _iter_clip_files(
    dataset: str,
    root_prefix: str,
    agent_dirs: Iterable[str],
    days: Iterable[str],
    suffix: str,
    *,
    revision: str,
    max_per_agent_day: int | None,
) -> Iterable[tuple[EgoLifeKey, str]]:
    for agent_dir in agent_dirs:
        for day in days:
            repo_dir = f"{root_prefix}/{agent_dir}/{day}" if root_prefix else f"{agent_dir}/{day}"
            rows = hf_tree(dataset, repo_dir, revision=revision)
            emitted = 0
            for row in rows:
                if row.get("type") != "file":
                    continue
                path = row.get("path", "")
                if not path.endswith(suffix):
                    continue
                parsed = parse_egolife_path(path)
                yield parsed, path
                emitted += 1
                if max_per_agent_day is not None and emitted >= max_per_agent_day:
                    break


def build_manifest(
    *,
    output_path: str | Path,
    agents: list[str] | None = None,
    days: list[str] | None = None,
    revision: str = DEFAULT_REVISION,
    max_per_agent_day: int | None = None,
    include_overlays: bool = True,
) -> dict[str, Any]:
    agent_dirs = agents or list(AGENTS)
    day_names = days or [f"DAY{i}" for i in range(1, 8)]

    videos: dict[tuple[str, str, str], str] = {}
    for parsed, path in _iter_clip_files(
        VIDEO_DATASET,
        "",
        agent_dirs,
        day_names,
        ".mp4",
        revision=revision,
        max_per_agent_day=max_per_agent_day,
    ):
        videos[parsed.key] = path

    gazes: dict[tuple[str, str, str], str] = {}
    for parsed, path in _iter_clip_files(
        GAZE_DATASET,
        "EyeGaze",
        agent_dirs,
        day_names,
        ".csv",
        revision=revision,
        max_per_agent_day=max_per_agent_day,
    ):
        gazes[parsed.key] = path

    overlays: dict[tuple[str, str, str], str] = {}
    if include_overlays:
        for parsed, path in _iter_clip_files(
            GAZE_DATASET,
            "EyeTracking",
            agent_dirs,
            day_names,
            ".mp4",
            revision=revision,
            max_per_agent_day=max_per_agent_day,
        ):
            overlays[parsed.key] = path

    clips: list[dict[str, Any]] = []
    for key, video_path in sorted(videos.items()):
        parsed = parse_egolife_path(video_path)
        gaze_path = gazes.get(key)
        if not gaze_path:
            continue
        overlay_path = overlays.get(key)
        clips.append(
            {
                "clip_id": stable_id(parsed.day, parsed.agent_dir, parsed.time_token),
                "day": parsed.day,
                "agent_dir": parsed.agent_dir,
                "agent_id": parsed.agent_id,
                "agent_name": parsed.agent_name,
                "time_token": parsed.time_token,
                "clip_clock": parsed.clip_clock,
                "clock_seconds": seconds_from_time_token(parsed.time_token),
                "video_path": video_path,
                "video_url": hf_resolve_url(VIDEO_DATASET, video_path, revision),
                "gaze_path": gaze_path,
                "gaze_url": hf_resolve_url(GAZE_DATASET, gaze_path, revision),
                "overlay_path": overlay_path,
                "overlay_url": hf_resolve_url(GAZE_DATASET, overlay_path, revision) if overlay_path else None,
            }
        )

    manifest = {
        "datasets": {
            "video": VIDEO_DATASET,
            "gaze": GAZE_DATASET,
            "revision": revision,
        },
        "agents": AGENTS,
        "clips": clips,
        "summary": {
            "clip_count": len(clips),
            "days": sorted({clip["day"] for clip in clips}),
            "agents": sorted({clip["agent_dir"] for clip in clips}),
        },
    }
    write_json(output_path, manifest)
    return manifest


def _parse_csv_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build EgoLife video/gaze manifest")
    parser.add_argument("--output", required=True, help="Manifest JSON output path")
    parser.add_argument("--agents", help="Comma-separated agent dirs, e.g. A1_JAKE,A2_ALICE")
    parser.add_argument("--days", help="Comma-separated days, e.g. DAY1,DAY2")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--max-per-agent-day", type=int)
    parser.add_argument("--no-overlays", action="store_true")
    args = parser.parse_args(argv)
    manifest = build_manifest(
        output_path=args.output,
        agents=_parse_csv_list(args.agents),
        days=_parse_csv_list(args.days),
        revision=args.revision,
        max_per_agent_day=args.max_per_agent_day,
        include_overlays=not args.no_overlays,
    )
    print(f"wrote {len(manifest['clips'])} aligned clips to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
