"""Per-user semantic observation stage for the upgraded QA pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .evidence import extract_frames, ffprobe_duration, local_cache_path, summarize_gaze_csv
from .gaze_projection import find_clip_calibration
from .io_utils import download_file, iter_jsonl, read_json, write_jsonl
from .qwen3vl_runner import DEFAULT_MODEL_ID, make_runner


OBSERVATION_SCHEMA = {
    "agent_name": "string",
    "day": "DAY1",
    "time_token": "HHMMSScc",
    "location_guess": "short visual location description",
    "visible_people": ["names or descriptions"],
    "salient_objects": ["objects that matter"],
    "actions": ["short action/event facts"],
    "gaze_focus": ["what the wearer appears to be attending to"],
    "projected_gaze_use": "only use projected 2D gaze when projection_status is projected",
    "key_facts": ["atomic facts useful for QA construction"],
    "uncertain_details": ["things that are unclear or should not be used as facts"],
}


def select_manifest_clips(manifest: dict[str, Any], target_clip_count: int | None = None) -> list[dict[str, Any]]:
    clips = sorted(
        manifest.get("clips", []),
        key=lambda clip: (clip["day"], clip["time_token"], clip["agent_dir"]),
    )
    if target_clip_count is not None:
        return clips[:target_clip_count]
    return clips


def build_observation_prompt(clip_packet: dict[str, Any]) -> str:
    frame_lines = [
        f"{idx + 1}. {frame['path']} at {frame['timestamp_seconds']} seconds"
        for idx, frame in enumerate(clip_packet.get("frames", []))
    ]
    prompt_packet = {
        "agent_name": clip_packet.get("agent_name"),
        "day": clip_packet.get("day"),
        "clip_clock": clip_packet.get("clip_clock"),
        "video_url": clip_packet.get("video_url"),
        "gaze_summary": clip_packet.get("gaze_summary"),
        "local_calibration": clip_packet.get("local_calibration"),
        "frames": frame_lines,
    }
    return f"""You are summarizing one user's egocentric clip for downstream multi-user QA construction.

Describe only what is supported by the provided images and metadata. Prefer atomic facts that could later be combined with facts from another user.
The gaze CSV contains Aria CPF yaw/pitch/depth values, not image pixels. Treat projected 2D gaze as usable only when `projection_status` is `projected`; otherwise, describe gaze only as an unprojected 3D/angle summary.

Return one valid JSON object only with this shape:
{json.dumps(OBSERVATION_SCHEMA, ensure_ascii=False, indent=2)}

Clip packet:
{json.dumps(prompt_packet, ensure_ascii=False, indent=2)}
"""


def prepare_clip_packet(
    clip: dict[str, Any],
    *,
    cache_dir: str | Path,
    output_root: str | Path,
    frames_per_clip: int,
    aria_calibration_dir: str | Path | None,
    download_media: bool,
) -> dict[str, Any]:
    local_video = local_cache_path(cache_dir, clip["video_path"])
    local_gaze = local_cache_path(cache_dir, clip["gaze_path"])
    if download_media:
        download_file(clip["video_url"], local_video)
        download_file(clip["gaze_url"], local_gaze)

    duration = ffprobe_duration(local_video) if local_video.exists() else None
    frames: list[dict[str, Any]] = []
    if local_video.exists():
        frame_dir = Path(output_root) / "observation_assets" / clip["clip_id"] / clip["agent_dir"]
        frames = extract_frames(
            local_video,
            frame_dir,
            frames_per_clip=frames_per_clip,
            duration=duration,
        )
    calibration_path = find_clip_calibration(aria_calibration_dir, clip)
    gaze_summary = summarize_gaze_csv(local_gaze, calibration_path=calibration_path) if local_gaze.exists() else {}
    return {
        **clip,
        "duration_seconds": duration,
        "local_video": str(local_video) if local_video.exists() else None,
        "local_gaze": str(local_gaze) if local_gaze.exists() else None,
        "local_calibration": str(calibration_path) if calibration_path else None,
        "frames": frames,
        "gaze_summary": gaze_summary,
    }


def observe_clips(
    *,
    manifest_path: str | Path,
    output_path: str | Path,
    prompts_path: str | Path | None,
    cache_dir: str | Path,
    output_root: str | Path,
    target_clip_count: int | None,
    frames_per_clip: int,
    aria_calibration_dir: str | Path | None,
    backend: str,
    model_id: str = DEFAULT_MODEL_ID,
    base_url: str = "http://127.0.0.1:8000/v1",
    max_new_tokens: int = 768,
    max_image_pixels: int = 262144,
    dtype: str = "bfloat16",
    allow_cpu: bool = False,
    disable_thinking: bool = False,
    dry_run: bool = False,
    download_media: bool = True,
) -> list[dict[str, Any]]:
    manifest = read_json(manifest_path)
    clips = select_manifest_clips(manifest, target_clip_count)
    runner = make_runner(
        "dry-run" if dry_run else backend,
        model_id=model_id,
        base_url=base_url,
        max_new_tokens=max_new_tokens,
        max_image_pixels=max_image_pixels,
        dtype=dtype,
        allow_cpu=allow_cpu,
        disable_thinking=disable_thinking,
    )
    rows = []
    prompt_rows = []
    for clip in clips:
        packet = prepare_clip_packet(
            clip,
            cache_dir=cache_dir,
            output_root=output_root,
            frames_per_clip=frames_per_clip,
            aria_calibration_dir=aria_calibration_dir,
            download_media=download_media,
        )
        prompt = build_observation_prompt(packet)
        image_paths = [frame["path"] for frame in packet.get("frames", []) if Path(frame["path"]).exists()]
        prompt_rows.append({"clip_id": clip["clip_id"], "prompt": prompt, "image_paths": image_paths})
        observation: dict[str, Any] = {
            "status": "dry_run",
            "agent_name": packet.get("agent_name"),
            "day": packet.get("day"),
            "time_token": packet.get("time_token"),
            "location_guess": "dry-run shared synchronized window",
            "visible_people": [],
            "salient_objects": ["dry-run sampled frame evidence"],
            "actions": [f"{packet.get('agent_name')} has sampled frames in the shared EgoLife time window"],
            "projected_gaze_use": "dry-run; use projected gaze only when projection_status is projected",
            "gaze_focus": [f"dry-run gaze projection status: {packet.get('gaze_summary', {}).get('projection_status')}"],
            "key_facts": [
                f"{packet.get('agent_name')} contributes sampled frames from {packet.get('day')} {packet.get('time_token')}",
                "shared synchronized EgoLife time window",
            ],
            "uncertain_details": ["No Qwen3-VL model was run in dry-run mode."],
        }
        raw_output = ""
        if not dry_run:
            raw_output = runner.generate(prompt, image_paths)
            try:
                from .schema import extract_json_object

                observation = extract_json_object(raw_output)
                observation["status"] = "ok"
            except Exception as exc:
                observation = {
                    "status": "parse_failed",
                    "key_facts": [],
                    "uncertain_details": [str(exc)],
                }
        rows.append(
            {
                "clip_id": clip["clip_id"],
                "clip": packet,
                "observation": observation,
                "model_id": runner.model_id,
                "raw_output": raw_output,
            }
        )
    if prompts_path:
        write_jsonl(prompts_path, prompt_rows)
    write_jsonl(output_path, rows)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize EgoLife clips into per-user observations")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompts-output")
    parser.add_argument("--cache-dir", default=".cache/egolife_two_user_qa")
    parser.add_argument("--output-root", default="egolife_two_user_qa/outputs/pilot_20")
    parser.add_argument("--target-clip-count", type=int)
    parser.add_argument("--frames-per-clip", type=int, default=4)
    parser.add_argument("--aria-calibration-dir")
    parser.add_argument("--backend", default="transformers-local", choices=["transformers-local", "openai-compatible-local"])
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--max-image-pixels", type=int, default=262144)
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-download-media", action="store_true")
    args = parser.parse_args(argv)
    rows = observe_clips(
        manifest_path=args.manifest,
        output_path=args.output,
        prompts_path=args.prompts_output,
        cache_dir=args.cache_dir,
        output_root=args.output_root,
        target_clip_count=args.target_clip_count,
        frames_per_clip=args.frames_per_clip,
        aria_calibration_dir=args.aria_calibration_dir,
        backend=args.backend,
        model_id=args.model_id,
        base_url=args.base_url,
        max_new_tokens=args.max_new_tokens,
        max_image_pixels=args.max_image_pixels,
        dtype=args.dtype,
        allow_cpu=args.allow_cpu,
        disable_thinking=args.disable_thinking,
        dry_run=args.dry_run,
        download_media=not args.no_download_media,
    )
    print(f"wrote {len(rows)} observations to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
