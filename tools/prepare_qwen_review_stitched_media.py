"""从 EgoLife 公开 30 秒片段确定性拼接双条件评审所需十分钟视频。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

HF_BASE = "https://huggingface.co/datasets/lmms-lab/EgoLife/resolve/main"
USER_LAYOUT = (
    ("Jake", "A1_JAKE"),
    ("Alice", "A2_ALICE"),
    ("Tasha", "A3_TASHA"),
    ("Lucia", "A4_LUCIA"),
    ("Katrina", "A5_KATRINA"),
    ("Shure", "A6_SHURE"),
)


@dataclass(frozen=True)
class MediaTask:
    group_id: str
    group_dir: str
    day: str
    user: str
    agent_dir: str
    urls: tuple[str, ...]
    output_path: Path


def segment_timestamps(start_code: str) -> tuple[str, ...]:
    if len(start_code) != 8 or not start_code.isdigit() or start_code[-2:] != "00":
        raise ValueError(f"invalid EgoLife timestamp: {start_code}")
    start = datetime.strptime(start_code[:6], "%H%M%S")
    return tuple(
        (start + timedelta(seconds=30 * index)).strftime("%H%M%S") + "00"
        for index in range(20)
    )


def segment_url(day: str, user: str, timestamp: str) -> str:
    agent_by_user = dict(USER_LAYOUT)
    if user not in agent_by_user:
        raise ValueError(f"unknown user: {user}")
    agent = agent_by_user[user]
    filename = f"{day}_{agent}_{timestamp}.mp4"
    return f"{HF_BASE}/{agent}/{day}/{filename}"


def _group_parts(group_id: str) -> tuple[str, str, str]:
    try:
        day, start = group_id.split("::", 1)
    except ValueError as exc:
        raise ValueError(f"invalid generation group: {group_id}") from exc
    if not day.startswith("DAY"):
        raise ValueError(f"invalid generation group day: {group_id}")
    segment_timestamps(start)
    return day, start, group_id.replace("::", "_")


def build_media_tasks(
    selection_path: str | Path,
    media_root: str | Path,
) -> tuple[MediaTask, ...]:
    groups: list[str] = []
    seen: set[str] = set()
    for line in Path(selection_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        group_id = str(json.loads(line)["generation_group_id"])
        if group_id not in seen:
            groups.append(group_id)
            seen.add(group_id)
    tasks: list[MediaTask] = []
    root = Path(media_root)
    for group_id in groups:
        day, start, group_dir = _group_parts(group_id)
        timestamps = segment_timestamps(start)
        for user, agent_dir in USER_LAYOUT:
            tasks.append(
                MediaTask(
                    group_id=group_id,
                    group_dir=group_dir,
                    day=day,
                    user=user,
                    agent_dir=agent_dir,
                    urls=tuple(segment_url(day, user, value) for value in timestamps),
                    output_path=root / group_dir / f"{user}.mp4",
                )
            )
    if not tasks:
        raise ValueError("selection contains no generation groups")
    return tuple(tasks)


def _probe_duration(ffprobe: str, path: Path) -> float:
    completed = subprocess.run(
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
    return float(completed.stdout.strip())


def _usable_output(ffprobe: str, path: Path) -> tuple[bool, float | None]:
    if not path.is_file() or path.stat().st_size <= 0:
        return False, None
    try:
        duration = _probe_duration(ffprobe, path)
    except (OSError, subprocess.SubprocessError, ValueError):
        return False, None
    return 590.0 <= duration <= 615.0, duration


def _download_segment(url: str, destination: Path, timeout: int) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "egoqa-qwen-review/1"})
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                with temporary.open("wb") as handle:
                    shutil.copyfileobj(response, handle, length=1024 * 1024)
            if temporary.stat().st_size <= 0:
                raise RuntimeError(f"empty download: {url}")
            os.replace(temporary, destination)
            return
        except Exception as exc:
            last_error = exc
            if temporary.exists():
                temporary.unlink()
            if attempt < 3:
                time.sleep(2 * attempt)
    raise RuntimeError(f"download failed after 3 attempts: {url}: {last_error}")


def _concat_segments(
    ffmpeg: str,
    ffprobe: str,
    segments: Iterable[Path],
    output_path: Path,
) -> float:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    segment_list = tuple(segments)
    concat_path = output_path.with_suffix(".concat.txt")
    concat_path.write_text(
        "".join(f"file '{path.as_posix()}'\n" for path in segment_list),
        encoding="utf-8",
    )
    temporary = output_path.with_suffix(".tmp.mp4")
    if temporary.exists():
        temporary.unlink()
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
            str(concat_path),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(temporary),
        ],
        check=True,
    )
    duration = _probe_duration(ffprobe, temporary)
    if not 590.0 <= duration <= 615.0:
        raise RuntimeError(
            f"stitched duration out of range: path={temporary} duration={duration}"
        )
    os.replace(temporary, output_path)
    concat_path.unlink(missing_ok=True)
    return duration


def _write_manifest(
    path: Path,
    status: str,
    results: list[dict],
    error: str | None,
    expected_task_count: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "completed_task_count": len(results),
        "expected_task_count": expected_task_count,
        "error": error,
        "items": results,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def prepare_media(
    tasks: Iterable[MediaTask],
    *,
    work_root: Path,
    ffmpeg: str,
    ffprobe: str,
    workers: int,
    timeout: int,
    manifest_path: Path,
) -> list[dict]:
    task_list = tuple(tasks)
    results: list[dict] = []
    try:
        for index, task in enumerate(task_list, 1):
            usable, duration = _usable_output(ffprobe, task.output_path)
            status = "reused" if usable else "prepared"
            if not usable:
                segment_root = work_root / task.group_dir / task.user
                segment_root.mkdir(parents=True, exist_ok=True)
                segment_paths = tuple(
                    segment_root / Path(url).name for url in task.urls
                )
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = [
                        executor.submit(_download_segment, url, path, timeout)
                        for url, path in zip(task.urls, segment_paths, strict=True)
                    ]
                    for future in futures:
                        future.result()
                duration = _concat_segments(
                    ffmpeg,
                    ffprobe,
                    segment_paths,
                    task.output_path,
                )
                shutil.rmtree(segment_root)
            result = {
                "index": index,
                "group_id": task.group_id,
                "user": task.user,
                "status": status,
                "source_segment_count": len(task.urls),
                "output_path": str(task.output_path),
                "duration_seconds": duration,
                "bytes": task.output_path.stat().st_size,
            }
            results.append(result)
            _write_manifest(
                manifest_path,
                "running",
                results,
                None,
                len(task_list),
            )
            print(
                f"MEDIA_TASK_DONE index={index}/{len(task_list)} "
                f"group={task.group_id} user={task.user} status={status} "
                f"duration={duration:.3f}",
                flush=True,
            )
    except Exception as exc:
        _write_manifest(
            manifest_path,
            "failed",
            results,
            str(exc),
            len(task_list),
        )
        raise
    _write_manifest(
        manifest_path,
        "passed",
        results,
        None,
        len(task_list),
    )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--media-root", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--ffprobe", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args(argv)
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    tasks = build_media_tasks(args.selection, args.media_root)
    prepare_media(
        tasks,
        work_root=Path(args.work_root),
        ffmpeg=args.ffmpeg,
        ffprobe=args.ffprobe,
        workers=args.workers,
        timeout=args.timeout,
        manifest_path=Path(args.manifest),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
