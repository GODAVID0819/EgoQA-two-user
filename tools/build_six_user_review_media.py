"""从六用户 candidate 元数据下载 30 秒原片并拼接人工审核视频。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def extract_review_media_plan(
    candidates: Iterable[dict[str, Any]],
    *,
    generation_group_id: str,
) -> dict[str, Any]:
    matches = [
        row
        for row in candidates
        if str(row.get("generation_group_id") or "") == generation_group_id
    ]
    if not matches:
        raise ValueError(f"generation group not found: {generation_group_id}")
    first = matches[0]
    clips = list(first.get("clips") or [])
    if len(clips) != 6:
        raise ValueError("review media plan requires exactly six user clips")
    users: dict[str, dict[str, Any]] = {}
    seen_urls: set[str] = set()
    for clip in sorted(clips, key=lambda row: str(row.get("agent_dir") or "")):
        user = str(clip.get("agent_name") or clip.get("user") or "").strip()
        agent_dir = str(clip.get("agent_dir") or "").strip()
        segments = sorted(
            list(clip.get("segments") or []),
            key=lambda row: (float(row.get("clock_seconds") or 0), str(row.get("time_token") or "")),
        )
        if not user or not agent_dir or len(segments) != 6:
            raise ValueError(f"user clip must contain a name, agent_dir, and six segments: {user!r}")
        planned_segments = []
        for index, segment in enumerate(segments):
            token = str(segment.get("time_token") or "").strip()
            url = str(segment.get("video_url") or "").strip()
            if not token or not url:
                raise ValueError(f"missing time_token or video_url for {user} segment {index}")
            if url in seen_urls:
                raise ValueError(f"duplicate segment URL: {url}")
            seen_urls.add(url)
            planned_segments.append(
                {
                    "segment_index": index,
                    "time_token": token,
                    "clip_clock": segment.get("clip_clock"),
                    "clock_seconds": segment.get("clock_seconds"),
                    "video_url": url,
                }
            )
        users[user] = {
            "agent_dir": agent_dir,
            "segments": planned_segments,
        }
    if len(users) != 6 or len(seen_urls) != 36:
        raise ValueError("review media plan must contain six users and 36 unique URLs")
    return {
        "generation_group_id": generation_group_id,
        "source_evidence_ids": [str(row.get("evidence_id") or "") for row in matches],
        "users": users,
    }


def download_atomic(
    url: str,
    destination: str | Path,
    *,
    open_url: Callable[..., Any] = urllib.request.urlopen,
    chunk_size: int = 1024 * 1024,
) -> Path:
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size > 0:
        return target
    temporary = target.with_name(f".{target.name}.part")
    request = urllib.request.Request(url, headers={"User-Agent": "EgoQA-review-media/1.0"})
    try:
        with open_url(request, timeout=120) as response, temporary.open("wb") as stream:
            status = int(getattr(response, "status", 200))
            if status != 200:
                raise RuntimeError(f"download returned HTTP {status}: {url}")
            while chunk := response.read(chunk_size):
                stream.write(chunk)
        if temporary.stat().st_size <= 0:
            raise RuntimeError(f"download produced an empty file: {url}")
        temporary.replace(target)
        return target
    finally:
        if temporary.exists():
            temporary.unlink()


def remote_content_length(
    url: str,
    *,
    open_url: Callable[..., Any] = urllib.request.urlopen,
) -> int | None:
    request = urllib.request.Request(
        url,
        method="HEAD",
        headers={"User-Agent": "EgoQA-review-media/1.0"},
    )
    with open_url(request, timeout=60) as response:
        value = response.headers.get("Content-Length")
    return int(value) if value and value.isdigit() else None


def probe_video(
    path: str | Path,
    *,
    ffprobe_binary: str,
    command_runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    result = command_runner(
        [
            ffprobe_binary,
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=index,codec_type,codec_name,width,height,r_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _concat_quote(path: Path) -> str:
    return str(path.resolve()).replace("'", "'\\''")


def concat_segments(
    segments: list[Path],
    output: str | Path,
    *,
    ffmpeg_binary: str,
    command_runner: Callable[..., Any] = subprocess.run,
    allow_transcode: bool = False,
) -> Path:
    if len(segments) != 6 or any(not path.is_file() for path in segments):
        raise ValueError("concat requires six existing segment files")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    concat_file = target.with_suffix(".concat.txt")
    concat_file.write_text(
        "".join(f"file '{_concat_quote(path)}'\n" for path in segments),
        encoding="utf-8",
    )
    command = [
        ffmpeg_binary,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-c",
        "copy",
        str(target),
    ]
    try:
        command_runner(command, check=True, capture_output=True, text=True)
    except Exception:
        if not allow_transcode:
            raise
        command_runner(
            [
                ffmpeg_binary,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "20",
                "-c:a",
                "aac",
                str(target),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    if not target.is_file() or target.stat().st_size <= 0:
        raise RuntimeError(f"ffmpeg did not create stitched video: {target}")
    return target


def render_review_media_markdown(
    plan: dict[str, Any],
    *,
    stitched_paths: dict[str, str | Path],
) -> str:
    lines = [
        "## 2. 完整三分钟六路审核媒体",
        "",
        (
            "以下 URL 直接来自 candidate 的 `clips[].segments[].video_url`。"
            "每位用户包含六个连续 30 秒原始片段；本地成片由这六段按原始顺序拼接。"
        ),
        "",
    ]
    for user, row in plan["users"].items():
        lines.extend([f"### {user}（{row['agent_dir']}）", ""])
        for segment in row["segments"]:
            lines.append(
                f"- [{segment['time_token']}]({segment['video_url']})"
            )
        stitched = stitched_paths.get(user)
        if stitched:
            lines.append(f"- 本地三分钟成片：`{Path(stitched)}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def update_report_media_section(report_path: str | Path, media_markdown: str) -> None:
    path = Path(report_path)
    text = path.read_text(encoding="utf-8")
    start = text.find("## 2.")
    end = text.find("## 3.", start + 1)
    if start < 0 or end < 0:
        raise ValueError("report must contain ordered section 2 and section 3 headings")
    updated = text[:start] + media_markdown.rstrip() + "\n\n" + text[end:]
    path.write_text(updated, encoding="utf-8")


def build_output_paths(plan: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    segment_paths: dict[str, list[Path]] = {}
    stitched_paths: dict[str, Path] = {}
    group_label = plan["generation_group_id"].replace("::", "_")
    for user, row in plan["users"].items():
        agent_dir = row["agent_dir"]
        segment_paths[user] = [
            output_dir / "segments" / agent_dir / f"{segment['time_token']}.mp4"
            for segment in row["segments"]
        ]
        stitched_paths[user] = (
            output_dir / "stitched" / f"{agent_dir}_{user}_{group_label}_180s.mp4"
        )
    return {"segments": segment_paths, "stitched": stitched_paths}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--generation-group-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--stitch", action="store_true")
    parser.add_argument("--allow-transcode", action="store_true")
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg"))
    parser.add_argument("--ffprobe", default=shutil.which("ffprobe"))
    parser.add_argument("--probe-remote-sizes", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = extract_review_media_plan(
        read_jsonl(args.candidates),
        generation_group_id=args.generation_group_id,
    )
    paths = build_output_paths(plan, output_dir)
    if args.probe_remote_sizes:
        total = 0
        unknown = 0
        for row in plan["users"].values():
            for segment in row["segments"]:
                size = remote_content_length(segment["video_url"])
                segment["content_length"] = size
                if size is None:
                    unknown += 1
                else:
                    total += size
        print(f"REMOTE_SIZE_KNOWN_BYTES={total} UNKNOWN_COUNT={unknown}")
    if args.download:
        for user, row in plan["users"].items():
            for segment, destination in zip(row["segments"], paths["segments"][user]):
                download_atomic(segment["video_url"], destination)
                if args.ffprobe:
                    segment["ffprobe"] = probe_video(
                        destination,
                        ffprobe_binary=args.ffprobe,
                    )
    if args.stitch:
        if not args.ffmpeg or not args.ffprobe:
            raise SystemExit("ffmpeg and ffprobe are required for --stitch")
        for user in plan["users"]:
            concat_segments(
                paths["segments"][user],
                paths["stitched"][user],
                ffmpeg_binary=args.ffmpeg,
                allow_transcode=args.allow_transcode,
            )
            probe = probe_video(paths["stitched"][user], ffprobe_binary=args.ffprobe)
            duration = float((probe.get("format") or {}).get("duration") or 0)
            if not 175.0 <= duration <= 185.0:
                raise RuntimeError(f"stitched duration is not approximately 180s: {user}={duration}")
            plan["users"][user]["stitched_ffprobe"] = probe
    plan_path = output_dir / "review_media_manifest.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown = render_review_media_markdown(plan, stitched_paths=paths["stitched"] if args.stitch else {})
    markdown_path = output_dir / "review_media_urls.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    if args.report:
        update_report_media_section(args.report, markdown)
    print(f"REVIEW_MEDIA_MANIFEST={plan_path}")
    print(f"REVIEW_MEDIA_URLS={markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
