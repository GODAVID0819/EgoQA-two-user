"""从指定仓库commit原样提取的CPU解码函数；只保留必要依赖。"""
from __future__ import annotations
import math,shutil,subprocess,time
from pathlib import Path
from typing import Any,Sequence,BinaryIO

def _read_ppm_token(stream: BinaryIO) -> bytes | None:
    """Read one whitespace-delimited PPM token, skipping comments."""

    token = bytearray()
    while True:
        value = stream.read(1)
        if not value:
            return bytes(token) if token else None
        if value == b"#":
            stream.readline()
            if token:
                return bytes(token)
            continue
        if value.isspace():
            if token:
                return bytes(token)
            continue
        token.extend(value)

def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            break
        chunks.extend(chunk)
    if len(chunks) != size:
        raise RuntimeError(
            f"ffmpeg RGB stream ended early: expected={size} actual={len(chunks)}"
        )
    return bytes(chunks)

def _decode_ffmpeg_ppm_stream(
    command: list[str],
    *,
    expected_frame_count: int,
    source_description: str,
) -> list[Any]:
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Pillow is required for direct RGB video sampling") from exc

    started = time.monotonic()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdout is None:
        raise RuntimeError("ffmpeg did not expose its RGB output pipe")
    frames: list[Any] = []
    try:
        while True:
            magic = _read_ppm_token(process.stdout)
            if magic is None:
                break
            if magic != b"P6":
                raise RuntimeError(f"unexpected ffmpeg PPM magic: {magic!r}")
            width_token = _read_ppm_token(process.stdout)
            height_token = _read_ppm_token(process.stdout)
            max_value_token = _read_ppm_token(process.stdout)
            if None in (width_token, height_token, max_value_token):
                raise RuntimeError("incomplete ffmpeg PPM header")
            width = int(width_token)
            height = int(height_token)
            max_value = int(max_value_token)
            if width <= 0 or height <= 0 or max_value != 255:
                raise RuntimeError(
                    f"unsupported ffmpeg PPM frame: {width}x{height} max={max_value}"
                )
            pixels = _read_exact(process.stdout, width * height * 3)
            frames.append(Image.frombytes("RGB", (width, height), pixels))
    finally:
        process.stdout.close()
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    if process.stderr:
        process.stderr.close()
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command, stderr=stderr)
    minimum = min(expected_frame_count, max(1, expected_frame_count - 2))
    if len(frames) < minimum:
        raise RuntimeError(
            "ffmpeg direct RGB sampling returned too few frames: "
            f"source={source_description} expected={expected_frame_count} "
            f"minimum={minimum} actual={len(frames)}"
        )
    print(
        "sampled_video_rgb "
        f"status=decoded source={source_description} frames={len(frames)} "
        f"ffmpeg_processes=1 seconds={time.monotonic() - started:.3f}",
        flush=True,
    )
    return frames

def decode_source_segments_rgb_frames(
    video_paths: Sequence[str | Path],
    *,
    duration_seconds: float,
    sample_interval_seconds: float,
    segment_durations_seconds: Sequence[float] | None = None,
    ffmpeg_binary: str = "ffmpeg",
) -> list[Any]:
    """Continuously sample ordered clips with one padded concat-filter process."""

    if duration_seconds <= 0 or sample_interval_seconds <= 0:
        raise ValueError("duration and sample interval must be positive")
    if not video_paths:
        raise ValueError("at least one source video is required")
    ffmpeg = shutil.which(ffmpeg_binary)
    if not ffmpeg:
        explicit = Path(ffmpeg_binary)
        if explicit.exists():
            ffmpeg = str(explicit)
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for direct RGB video sampling")
    resolved = [Path(path).resolve() for path in video_paths]
    nominal_durations = (
        [float(value) for value in segment_durations_seconds]
        if segment_durations_seconds is not None
        else [None for _ in resolved]
    )
    if len(nominal_durations) != len(resolved) or any(
        value is not None and value <= 0 for value in nominal_durations
    ):
        raise ValueError("nominal segment durations must be positive and align with videos")
    for path in resolved:
        if not path.is_file():
            raise FileNotFoundError(f"source segment is missing: {path}")
    expected = max(1, int(math.ceil(duration_seconds / sample_interval_seconds)))
    input_args = [value for path in resolved for value in ("-i", str(path))]
    prepared_labels = []
    filters = []
    for index, nominal_duration in enumerate(nominal_durations):
        label = f"v{index}"
        prepared_labels.append(f"[{label}]")
        chain = f"[{index}:v]setpts=PTS-STARTPTS"
        if nominal_duration is not None:
            # Pad each logical 30-second slot before concatenation.  This keeps
            # later users/times aligned even when an encoded MP4 ends slightly
            # early, without writing a repaired intermediate video.
            chain += (
                f",tpad=stop_mode=clone:stop_duration={nominal_duration:.6f}"
                f",trim=duration={nominal_duration:.6f}"
            )
        filters.append(f"{chain}[{label}]")
    filters.append(
        "".join(prepared_labels)
        + f"concat=n={len(resolved)}:v=1:a=0,"
        + f"fps={1.0 / float(sample_interval_seconds):.12g}:start_time=0,"
        + "format=rgb24[outv]"
    )
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        *input_args,
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[outv]",
        "-t",
        f"{float(duration_seconds):.6f}",
        "-frames:v",
        str(expected),
        "-f",
        "image2pipe",
        "-vcodec",
        "ppm",
        "pipe:1",
    ]
    return _decode_ffmpeg_ppm_stream(
        command,
        expected_frame_count=expected,
        source_description=f"{len(resolved)}_ordered_segments",
    )
