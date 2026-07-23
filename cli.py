"""Unified CLI for the EgoLife two-user question-answer pilot."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from .evidence import DEFAULT_EVIDENCE_DURATION_SECONDS, local_cache_path, prepare_evidence
from .io_utils import iter_jsonl
from .manifest import build_manifest
from .candidate_mining import mine_candidates
from .clip_gap_demo import (
    run_clip_gap_demo,
    run_diverse_packet_trials,
    run_random_clip_gap_trials,
)
from .clip_exclusive_mining import mine_clip_exclusive_candidates
from .group_relative_clip_sampling import mine_group_relative_clip_candidates
from .pruning_k_grid import (
    DEFAULT_K_VALUES,
    DEFAULT_RANDOM_SEED,
    parse_k_values,
    run_pruning_k_grid,
)
from .pruning_ablation import (
    DEFAULT_FPS_VALUES as DEFAULT_PRUNING_ABLATION_FPS_VALUES,
    DEFAULT_RANDOM_SEED as DEFAULT_PRUNING_ABLATION_RANDOM_SEED,
    DEFAULT_TEMPORAL_POLICIES,
    DEFAULT_THRESHOLD_VALUES,
    parse_float_values as parse_pruning_ablation_float_values,
    parse_temporal_policies,
    run_pruning_ablation,
)
from .observations import observe_clips
from .object_hints import (
    DEFAULT_LOCAL_BASE_URL,
    DEFAULT_OBJECT_DETECTION_MODEL,
    DEFAULT_REID_MODEL_ID,
    DEFAULT_REID_TEXT_THRESHOLD,
    DEFAULT_REID_VISUAL_THRESHOLD,
    enrich_evidence_with_object_hints,
)
from .qa_pipeline import add_runner_args, validate_outputs
from .review_media import materialize_review_videos
from .video_qa_loop import add_video_loop_args, generate_video_qa_loop, parse_question_types


def _csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def _source_video_name(clip: dict) -> str | None:
    for key in ("video_path", "local_video", "video_url"):
        value = clip.get(key)
        if value:
            name = Path(str(value)).name
            if name.lower().endswith((".mp4", ".mov", ".mkv", ".avi", ".webm")):
                return name
    day = clip.get("day")
    agent_dir = clip.get("agent_dir")
    time_token = clip.get("time_token")
    if day and agent_dir and time_token:
        return f"{day}_{agent_dir}_{time_token}.mp4"
    return None


def _cache_day_dir_candidates(day: str) -> list[str]:
    text = str(day or "")
    candidates = []
    if text.startswith("DAY_") and text[4:].isdigit():
        candidates.extend([text, "DAY" + text[4:]])
    elif text.startswith("DAY") and text[3:].isdigit():
        candidates.extend(["DAY_" + text[3:], text])
    elif text:
        candidates.append(text)
    return list(dict.fromkeys(candidates))


def cache_video_candidates(clip: dict, cache_dir: str | Path | None) -> list[Path]:
    """Return expected cache paths, accepting both DAY_N and DAYN folders."""

    candidates: list[Path] = []
    local_video = clip.get("local_video")
    if local_video:
        candidates.append(Path(local_video))
    if not cache_dir:
        return candidates

    agent_dir = clip.get("agent_dir")
    day = clip.get("day")
    video_name = _source_video_name(clip)
    if not (agent_dir and day):
        return candidates

    day_dirs = [Path(cache_dir) / str(agent_dir) / day_name for day_name in _cache_day_dir_candidates(str(day))]
    if video_name:
        candidates.extend(day_dir / video_name for day_dir in day_dirs)
    else:
        candidates.extend(day_dir / "placeholder.mp4" for day_dir in day_dirs)
    for day_dir in day_dirs:
        if day_dir.is_dir():
            candidates.extend(sorted(day_dir.glob("*.mp4")))
    return list(dict.fromkeys(candidates))


def resolve_cached_local_video(clip: dict, cache_dir: str | Path | None) -> Path | None:
    """Resolve a clip to an existing local video in the expected cache layout."""

    for candidate in cache_video_candidates(clip, cache_dir):
        if candidate.is_file():
            return candidate
    return None


def preflight_cached_evidence(
    evidence_path: str | Path,
    target_count: int,
    *,
    cache_dir: str | Path | None = None,
    resolved_output: str | Path | None = None,
) -> int:
    """Verify evidence rows and optionally write resolved cache local_video paths."""

    missing: list[str] = []
    checked_packets = 0
    checked_videos = 0
    resolved_packets: list[dict] = []
    for packet_index, packet in enumerate(iter_jsonl(evidence_path)):
        packet_out = copy.deepcopy(packet)
        should_check = packet_index < target_count
        if should_check:
            checked_packets += 1
        evidence_id = packet.get("evidence_id") or f"row_{packet_index + 1}"
        clips = packet.get("clips")
        clips_out = packet_out.get("clips")
        if not isinstance(clips, list) or not clips:
            if should_check:
                missing.append(f"{evidence_id}: missing clips list")
            resolved_packets.append(packet_out)
            continue
        if not isinstance(clips_out, list):
            clips_out = []
            packet_out["clips"] = clips_out
        for clip_index, clip in enumerate(clips, start=1):
            user = clip.get("agent_name") or clip.get("agent_dir") or f"clip_{clip_index}"
            resolved = resolve_cached_local_video(clip, cache_dir)
            if resolved and clip_index <= len(clips_out) and isinstance(clips_out[clip_index - 1], dict):
                clips_out[clip_index - 1]["local_video"] = str(resolved)
            if should_check:
                checked_videos += 1
                if not resolved:
                    candidates = [str(path) for path in cache_video_candidates(clip, cache_dir)]
                    video_name = _source_video_name(clip) or "<video_file>"
                    expected_path = (
                        local_cache_path(
                            cache_dir,
                            f"{clip.get('agent_dir')}/{clip.get('day')}/{video_name}",
                        )
                        if cache_dir and clip.get("agent_dir") and clip.get("day")
                        else None
                    )
                    expected = (
                        str(expected_path.parent) + "/ or matching DAYN folder/"
                        if expected_path
                        else "no --cache-dir provided"
                    )
                    missing.append(
                        f"{evidence_id}/{user}: local_video not found; expected cache layout {expected}; "
                        f"tried {candidates}"
                    )
        resolved_packets.append(packet_out)
        if should_check and checked_packets >= target_count and not resolved_output:
            break

    if checked_packets < target_count:
        missing.append(
            f"only found {checked_packets} evidence rows, but target_count is {target_count}"
        )

    if resolved_output:
        from .io_utils import write_jsonl

        write_jsonl(resolved_output, resolved_packets)

    print(
        "cached_evidence_preflight "
        f"packets={checked_packets} videos={checked_videos} missing={len(missing)}"
    )
    for item in missing[:20]:
        print(f"missing_cached_evidence {item}")
    if len(missing) > 20:
        print(f"missing_cached_evidence ... {len(missing) - 20} more")
    return 1 if missing else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="egolife-two-user-qa")
    sub = parser.add_subparsers(dest="command", required=True)

    manifest = sub.add_parser("build_manifest", help="Build EgoLife video/gaze manifest")
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--agents")
    manifest.add_argument("--days")
    manifest.add_argument("--revision", default="main")
    manifest.add_argument("--max-per-agent-day", type=int)
    manifest.add_argument("--no-overlays", action="store_true")

    evidence = sub.add_parser("prepare_evidence", help="Prepare multi-user evidence packets")
    evidence.add_argument("--manifest", required=True)
    evidence.add_argument("--output", required=True)
    evidence.add_argument("--cache-dir", default=".cache/egolife_two_user_qa")
    evidence.add_argument("--output-root", default="egolife_two_user_qa/outputs/pilot_20")
    evidence.add_argument("--target-count", type=int, default=20)
    evidence.add_argument("--users-per-case", type=int, default=2)
    evidence.add_argument("--frames-per-clip", type=int, default=3)
    evidence.add_argument(
        "--evidence-duration-seconds",
        type=float,
        default=DEFAULT_EVIDENCE_DURATION_SECONDS,
        help="Complete synchronized evidence-window duration (default: 30 seconds)",
    )
    evidence.add_argument("--aria-calibration-dir")
    evidence.add_argument("--max-groups", type=int)
    evidence.add_argument("--no-download-media", action="store_true")
    evidence.add_argument("--random-seed", type=int)
    evidence.add_argument("--stratify-by-day", action="store_true")

    obs = sub.add_parser("observe_clips", help="Summarize individual user clips with Qwen3-VL")
    obs.add_argument("--manifest", required=True)
    obs.add_argument("--output", required=True)
    obs.add_argument("--prompts-output")
    obs.add_argument("--cache-dir", default=".cache/egolife_two_user_qa")
    obs.add_argument("--output-root", default="egolife_two_user_qa/outputs/pilot_20")
    obs.add_argument("--target-clip-count", type=int)
    obs.add_argument("--frames-per-clip", type=int, default=4)
    obs.add_argument("--aria-calibration-dir")
    obs.add_argument("--no-download-media", action="store_true")
    add_runner_args(obs)

    mine = sub.add_parser("mine_candidates", help="Mine semantically complementary evidence packets")
    mine.add_argument("--observations", required=True)
    mine.add_argument("--output", required=True)
    mine.add_argument("--target-count", type=int, default=20)
    mine.add_argument("--users-per-case", type=int, default=2)
    mine.add_argument("--max-time-gap-seconds", type=float, default=90.0)
    mine.add_argument("--min-score", type=int, default=5)

    clip_gap = sub.add_parser("clip_gap_demo", help="Find CLIP anchors and cross-user evidence gaps")
    clip_gap.add_argument("--evidence", required=True)
    clip_gap.add_argument("--output-dir", required=True)
    clip_gap.add_argument("--packet-index", type=int, default=0)
    clip_gap.add_argument("--model-id", default="openai/clip-vit-base-patch32")
    clip_gap.add_argument("--start-seconds", type=float, default=0.0)
    clip_gap.add_argument("--duration-seconds", type=float, default=12.0)
    clip_gap.add_argument("--sample-interval-seconds", type=float, default=1.5)
    clip_gap.add_argument("--clusters-per-user", type=int, default=4)
    clip_gap.add_argument("--anchor-threshold", type=float, default=0.75)
    clip_gap.add_argument("--top-k", type=int, default=3)
    clip_gap.add_argument("--ffmpeg-binary", default="ffmpeg")
    clip_gap.add_argument("--use-existing-frames", action="store_true")
    clip_gap.add_argument("--random-trials", type=int, default=0)
    clip_gap.add_argument("--random-seed", type=int, default=42)
    clip_gap.add_argument("--diverse-packet-trials", type=int, default=0)

    clip_exclusive = sub.add_parser(
        "mine_clip_exclusive_candidates",
        help="Rank evidence packets by CLIP cross-user exclusiveness",
    )
    clip_exclusive.add_argument("--evidence", required=True)
    clip_exclusive.add_argument("--output", required=True)
    clip_exclusive.add_argument("--output-dir", required=True)
    clip_exclusive.add_argument("--model-id", default="openai/clip-vit-base-patch32")
    clip_exclusive.add_argument("--duration-seconds", type=float, default=12.0)
    clip_exclusive.add_argument("--sample-interval-seconds", type=float, default=1.5)
    clip_exclusive.add_argument("--start-seconds", type=float, default=0.0)
    clip_exclusive.add_argument("--clusters-per-user", type=int, default=4)
    clip_exclusive.add_argument("--anchor-threshold", type=float, default=0.75)
    clip_exclusive.add_argument("--top-k", type=int, default=3)
    clip_exclusive.add_argument("--max-packets", type=int)
    clip_exclusive.add_argument("--min-score", type=float)
    clip_exclusive.add_argument("--contact-sheet-count", type=int, default=5)
    clip_exclusive.add_argument("--ffmpeg-binary", default="ffmpeg")
    clip_exclusive.add_argument("--use-existing-frames", action="store_true")
    clip_exclusive.add_argument("--preserve-order", action="store_true")

    benchmark = sub.add_parser(
        "prepare_clip_pruned_benchmark",
        help="Prepare paired original/pruned CLIP-guided video-pair evidence packets",
    )
    benchmark.add_argument("--manifest", required=True)
    benchmark.add_argument("--output", required=True)
    benchmark.add_argument("--output-dir", required=True)
    benchmark.add_argument("--cache-dir", default=".cache/egolife_two_user_qa")
    benchmark.add_argument("--model-id", default="openai/clip-vit-base-patch32")
    benchmark.add_argument("--target-count", type=int, default=100)
    benchmark.add_argument("--max-groups", type=int)
    benchmark.add_argument("--min-group-size", type=int, default=2)
    benchmark.add_argument("--selected-count", type=int, default=2)
    benchmark.add_argument("--duration-seconds", type=float, default=30.0)
    benchmark.add_argument("--sample-interval-seconds", type=float, default=1.0)
    benchmark.add_argument("--start-seconds", type=float, default=0.0)
    benchmark.add_argument("--pairs-per-group", type=int, default=1)
    benchmark.add_argument("--topk", type=int, default=3)
    benchmark.add_argument("--min-topk-sim", type=float, default=0.65)
    benchmark.add_argument("--min-mean-sim", type=float, default=0.25)
    benchmark.add_argument("--max-mean-sim", type=float, default=0.90)
    benchmark.add_argument("--high-similarity-interval-threshold", type=float, default=0.82)
    benchmark.add_argument("--pruning-clusters-per-video", type=int, default=12)
    benchmark.add_argument("--preserve-shared-anchor-seconds", type=float, default=0.0)
    benchmark.add_argument("--min-pruned-video-seconds", type=float, default=8.0)
    benchmark.add_argument(
        "--pruning-protection-mode",
        choices=["reject", "min_seconds", "min_percent"],
        default="reject",
    )
    benchmark.add_argument("--min-pruned-video-percent", type=float)
    benchmark.add_argument(
        "--max-pair-time-difference-seconds",
        type=float,
        help=(
            "Only prune high-similarity centroid pairs whose timestamps differ by at most "
            "this many seconds; omit for timestamp-agnostic pruning"
        ),
    )
    benchmark.add_argument("--compare-all-pairs", action="store_true")
    benchmark.add_argument("--random-seed", type=int, default=42)
    benchmark.add_argument("--ffmpeg-binary", default="ffmpeg")
    benchmark.add_argument("--download-media", action="store_true")
    benchmark.add_argument("--review-dir")

    k_grid = sub.add_parser(
        "run_pruning_k_grid",
        help="Sample fixed synchronized pairs and materialize clustered-pruning variants across K",
    )
    k_grid.add_argument("--manifest", required=True)
    k_grid.add_argument("--output-dir", required=True)
    k_grid.add_argument("--cache-dir", default=".cache/egolife_two_user_qa")
    k_grid.add_argument("--pair-count", type=int, default=10)
    k_grid.add_argument("--max-groups", type=int)
    k_grid.add_argument("--min-group-size", type=int, default=2)
    k_grid.add_argument("--k-values", default=",".join(str(value) for value in DEFAULT_K_VALUES))
    k_grid.add_argument("--model-id", default="openai/clip-vit-base-patch32")
    k_grid.add_argument("--duration-seconds", type=float, default=30.0)
    k_grid.add_argument("--sample-interval-seconds", type=float, default=1.0)
    k_grid.add_argument("--start-seconds", type=float, default=0.0)
    k_grid.add_argument("--high-similarity-interval-threshold", type=float, default=0.82)
    k_grid.add_argument("--preserve-shared-anchor-seconds", type=float, default=0.0)
    k_grid.add_argument("--min-pruned-video-seconds", type=float, default=8.0)
    k_grid.add_argument(
        "--pruning-protection-mode",
        choices=["reject", "min_seconds", "min_percent"],
        default="min_seconds",
    )
    k_grid.add_argument("--min-pruned-video-percent", type=float)
    k_grid.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    k_grid.add_argument("--ffmpeg-binary", default="ffmpeg")
    k_grid.add_argument("--download-media", action="store_true")

    pruning_ablation = sub.add_parser(
        "run_pruning_ablation",
        help=(
            "Run separate temporal-policy, similarity-threshold, sampling-rate, and K sweeps "
            "on fixed synchronized 30-second pairs"
        ),
    )
    pruning_ablation.add_argument("--manifest", required=True)
    pruning_ablation.add_argument("--output-dir", required=True)
    pruning_ablation.add_argument("--cache-dir", default=".cache/egolife_two_user_qa")
    pruning_ablation.add_argument("--pair-count", type=int, default=10)
    pruning_ablation.add_argument("--max-groups", type=int)
    pruning_ablation.add_argument("--min-group-size", type=int, default=2)
    pruning_ablation.add_argument("--model-id", default="openai/clip-vit-base-patch32")
    pruning_ablation.add_argument("--duration-seconds", type=float, default=30.0)
    pruning_ablation.add_argument("--start-seconds", type=float, default=0.0)
    pruning_ablation.add_argument("--baseline-fps", type=float, default=1.0)
    pruning_ablation.add_argument("--baseline-k", type=int, default=12)
    pruning_ablation.add_argument("--baseline-threshold", type=float, default=0.82)
    pruning_ablation.add_argument(
        "--baseline-temporal-policy",
        choices=list(DEFAULT_TEMPORAL_POLICIES),
        default="current",
    )
    pruning_ablation.add_argument(
        "--fps-values",
        default=",".join(f"{value:g}" for value in DEFAULT_PRUNING_ABLATION_FPS_VALUES),
    )
    pruning_ablation.add_argument(
        "--k-values",
        default=",".join(str(value) for value in DEFAULT_K_VALUES),
    )
    pruning_ablation.add_argument(
        "--threshold-values",
        default=",".join(f"{value:.2f}" for value in DEFAULT_THRESHOLD_VALUES),
    )
    pruning_ablation.add_argument(
        "--temporal-policies",
        default=",".join(DEFAULT_TEMPORAL_POLICIES),
    )
    pruning_ablation.add_argument("--preserve-shared-anchor-seconds", type=float, default=0.0)
    pruning_ablation.add_argument("--min-pruned-video-seconds", type=float, default=8.0)
    pruning_ablation.add_argument(
        "--pruning-protection-mode",
        choices=["reject", "min_seconds", "min_percent"],
        default="min_seconds",
    )
    pruning_ablation.add_argument("--min-pruned-video-percent", type=float)
    pruning_ablation.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_PRUNING_ABLATION_RANDOM_SEED,
    )
    pruning_ablation.add_argument("--ffmpeg-binary", default="ffmpeg")
    pruning_ablation.add_argument("--download-media", action="store_true")

    video_gen = sub.add_parser(
        "generate_video_qa_loop",
        help="Generate video-first question-answer items with judge/evaluator retry loop",
    )
    video_gen.add_argument("--evidence", required=True)
    video_gen.add_argument("--output", required=True)
    video_gen.add_argument("--prompts-output")
    video_gen.add_argument("--rejected-output")
    video_gen.add_argument("--intermediate-output")
    video_gen.add_argument("--target-count", type=int, default=20)
    video_gen.add_argument("--max-attempts", type=int, default=3)
    add_video_loop_args(video_gen)

    object_hints = sub.add_parser(
        "add_object_hints",
        help="Add EgoEverything-style key-object detection hints to evidence packets",
    )
    object_hints.add_argument("--evidence", required=True)
    object_hints.add_argument("--output", required=True)
    object_hints.add_argument("--output-dir", default="outputs/object_hint_assets")
    object_hints.add_argument(
        "--backend",
        default="dry-run",
        choices=["dry-run", "openrouter", "transformers-local", "openai-compatible-local"],
    )
    object_hints.add_argument("--api-key")
    object_hints.add_argument("--model-id", default=DEFAULT_OBJECT_DETECTION_MODEL)
    object_hints.add_argument("--base-url", default=DEFAULT_LOCAL_BASE_URL)
    object_hints.add_argument("--max-new-tokens", type=int, default=1024)
    object_hints.add_argument("--max-image-pixels", type=int, default=262144)
    object_hints.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    object_hints.add_argument("--allow-cpu", action="store_true")
    object_hints.add_argument("--disable-thinking", action="store_true")
    object_hints.add_argument(
        "--frame-sampling-mode",
        default="evidence_frames",
        choices=["evidence_frames", "clip_medoids"],
    )
    object_hints.add_argument(
        "--frames-per-clip",
        type=int,
        default=3,
        help="Maximum sampled frames per clip; with clip_medoids, 0 means all kept pruning clusters.",
    )
    object_hints.add_argument("--objects-per-clip", type=int, default=3)
    object_hints.add_argument("--objects-per-packet", type=int, default=5)
    object_hints.add_argument("--n-workers", type=int, default=5)
    object_hints.add_argument("--max-packets", type=int)
    object_hints.add_argument("--random-seed", type=int)
    object_hints.add_argument("--disable-reid", action="store_true")
    object_hints.add_argument("--reid-model-id", default=DEFAULT_REID_MODEL_ID)
    object_hints.add_argument("--reid-device")
    object_hints.add_argument("--reid-visual-threshold", type=float, default=DEFAULT_REID_VISUAL_THRESHOLD)
    object_hints.add_argument("--reid-text-threshold", type=float, default=DEFAULT_REID_TEXT_THRESHOLD)
    object_hints.add_argument("--reid-batch-size", type=int, default=32)

    val = sub.add_parser("validate_outputs", help="Validate question-answer JSONL and write report/CSV")
    val.add_argument("--qa", required=True)
    val.add_argument("--report", required=True)
    val.add_argument("--csv-output")
    val.add_argument("--human-review-output")
    val.add_argument("--strict-review", action="store_true")

    review_media = sub.add_parser(
        "materialize_review_videos",
        help="Copy/download videos for manual question-answer review",
    )
    review_media.add_argument("--evidence")
    review_media.add_argument("--qa")
    review_media.add_argument("--output-dir", required=True)
    review_media.add_argument("--no-download", action="store_true")

    preflight = sub.add_parser(
        "hpc_preflight_cached_evidence",
        help="Verify evidence rows reference existing cached local_video files",
    )
    preflight.add_argument("--evidence", required=True)
    preflight.add_argument("--target-count", type=int, default=5)
    preflight.add_argument("--cache-dir")
    preflight.add_argument("--resolved-output")

    args = parser.parse_args(argv)
    if args.command == "build_manifest":
        result = build_manifest(
            output_path=args.output,
            agents=_csv(args.agents),
            days=_csv(args.days),
            revision=args.revision,
            max_per_agent_day=args.max_per_agent_day,
            include_overlays=not args.no_overlays,
        )
        print(f"wrote {len(result['clips'])} aligned clips to {args.output}")
        return 0
    if args.command == "prepare_evidence":
        rows = prepare_evidence(
            manifest_path=args.manifest,
            output_path=args.output,
            cache_dir=args.cache_dir,
            output_root=args.output_root,
            target_count=args.target_count,
            users_per_case=args.users_per_case,
            frames_per_clip=args.frames_per_clip,
            evidence_duration_seconds=args.evidence_duration_seconds,
            aria_calibration_dir=args.aria_calibration_dir,
            max_groups=args.max_groups,
            download_media=not args.no_download_media,
            random_seed=args.random_seed,
            stratify_by_day=args.stratify_by_day,
        )
        print(f"wrote {len(rows)} evidence packets to {args.output}")
        return 0
    if args.command == "observe_clips":
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
            api_key=args.api_key,
            dry_run=args.dry_run,
            download_media=not args.no_download_media,
        )
        print(f"wrote {len(rows)} observations to {args.output}")
        return 0
    if args.command == "mine_candidates":
        rows = mine_candidates(
            observations_path=args.observations,
            output_path=args.output,
            target_count=args.target_count,
            users_per_case=args.users_per_case,
            max_time_gap_seconds=args.max_time_gap_seconds,
            min_score=args.min_score,
        )
        print(f"wrote {len(rows)} semantic candidates to {args.output}")
        return 0
    if args.command == "clip_gap_demo":
        if args.diverse_packet_trials:
            result = run_diverse_packet_trials(
                evidence_path=args.evidence,
                output_dir=args.output_dir,
                model_id=args.model_id,
                trial_count=args.diverse_packet_trials,
                random_seed=args.random_seed,
                duration_seconds=args.duration_seconds,
                sample_interval_seconds=args.sample_interval_seconds,
                clusters_per_user=args.clusters_per_user,
                anchor_threshold=args.anchor_threshold,
                top_k=args.top_k,
                ffmpeg_binary=args.ffmpeg_binary,
                resample_videos=not args.use_existing_frames,
            )
            print(f"wrote {result['trial_count']} diverse-packet CLIP trials to {args.output_dir}")
            return 0
        if args.random_trials:
            result = run_random_clip_gap_trials(
                evidence_path=args.evidence,
                output_dir=args.output_dir,
                packet_index=args.packet_index,
                model_id=args.model_id,
                trial_count=args.random_trials,
                random_seed=args.random_seed,
                duration_seconds=args.duration_seconds,
                sample_interval_seconds=args.sample_interval_seconds,
                clusters_per_user=args.clusters_per_user,
                anchor_threshold=args.anchor_threshold,
                top_k=args.top_k,
                ffmpeg_binary=args.ffmpeg_binary,
                resample_videos=not args.use_existing_frames,
            )
            print(f"wrote {result['trial_count']} random CLIP gap trials to {args.output_dir}")
            return 0
        result = run_clip_gap_demo(
            evidence_path=args.evidence,
            output_dir=args.output_dir,
            packet_index=args.packet_index,
            model_id=args.model_id,
            start_seconds=args.start_seconds,
            duration_seconds=args.duration_seconds,
            sample_interval_seconds=args.sample_interval_seconds,
            clusters_per_user=args.clusters_per_user,
            anchor_threshold=args.anchor_threshold,
            top_k=args.top_k,
            ffmpeg_binary=args.ffmpeg_binary,
            resample_videos=not args.use_existing_frames,
        )
        print(
            f"wrote CLIP gap demo for {result['left_user']} and {result['right_user']} "
            f"to {args.output_dir}"
        )
        return 0
    if args.command == "mine_clip_exclusive_candidates":
        rows = mine_clip_exclusive_candidates(
            evidence_path=args.evidence,
            output_path=args.output,
            output_dir=args.output_dir,
            model_id=args.model_id,
            duration_seconds=args.duration_seconds,
            sample_interval_seconds=args.sample_interval_seconds,
            start_seconds=args.start_seconds,
            clusters_per_user=args.clusters_per_user,
            anchor_threshold=args.anchor_threshold,
            top_k=args.top_k,
            max_packets=args.max_packets,
            min_score=args.min_score,
            contact_sheet_count=args.contact_sheet_count,
            ffmpeg_binary=args.ffmpeg_binary,
            resample_videos=not args.use_existing_frames,
            preserve_order=args.preserve_order,
        )
        print(f"wrote {len(rows)} CLIP-exclusive evidence packets to {args.output}")
        return 0
    if args.command == "prepare_clip_pruned_benchmark":
        rows = mine_group_relative_clip_candidates(
            manifest_path=args.manifest,
            output_path=args.output,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            model_id=args.model_id,
            target_count=args.target_count,
            max_groups=args.max_groups,
            min_group_size=args.min_group_size,
            selected_count=args.selected_count,
            duration_seconds=args.duration_seconds,
            sample_interval_seconds=args.sample_interval_seconds,
            start_seconds=args.start_seconds,
            pairs_per_group=args.pairs_per_group,
            topk=args.topk,
            min_topk_sim=args.min_topk_sim,
            min_mean_sim=args.min_mean_sim,
            max_mean_sim=args.max_mean_sim,
            high_similarity_interval_threshold=args.high_similarity_interval_threshold,
            pruning_clusters_per_video=args.pruning_clusters_per_video,
            preserve_shared_anchor_seconds=args.preserve_shared_anchor_seconds,
            min_pruned_video_seconds=args.min_pruned_video_seconds,
            pruning_protection_mode=args.pruning_protection_mode,
            min_pruned_video_percent=args.min_pruned_video_percent,
            max_pair_time_difference_seconds=args.max_pair_time_difference_seconds,
            random_pair_first=not args.compare_all_pairs,
            random_seed=args.random_seed,
            ffmpeg_binary=args.ffmpeg_binary,
            download_media=args.download_media,
            review_dir=args.review_dir,
        )
        print(f"wrote {len(rows)} CLIP-pruned benchmark evidence packets to {args.output}")
        return 0
    if args.command == "run_pruning_k_grid":
        summary = run_pruning_k_grid(
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            pair_count=args.pair_count,
            max_groups=args.max_groups,
            min_group_size=args.min_group_size,
            k_values=parse_k_values(args.k_values),
            model_id=args.model_id,
            duration_seconds=args.duration_seconds,
            sample_interval_seconds=args.sample_interval_seconds,
            start_seconds=args.start_seconds,
            high_similarity_threshold=args.high_similarity_interval_threshold,
            preserve_shared_anchor_seconds=args.preserve_shared_anchor_seconds,
            min_pruned_video_seconds=args.min_pruned_video_seconds,
            pruning_protection_mode=args.pruning_protection_mode,
            min_pruned_video_percent=args.min_pruned_video_percent,
            random_seed=args.random_seed,
            ffmpeg_binary=args.ffmpeg_binary,
            download_media=args.download_media,
        )
        print(
            f"wrote {summary['pair_count']} synchronized pairs and "
            f"{summary['variant_count']} K-grid variants to {args.output_dir}"
        )
        return 0
    if args.command == "run_pruning_ablation":
        summary = run_pruning_ablation(
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            pair_count=args.pair_count,
            max_groups=args.max_groups,
            min_group_size=args.min_group_size,
            model_id=args.model_id,
            duration_seconds=args.duration_seconds,
            start_seconds=args.start_seconds,
            baseline_fps=args.baseline_fps,
            baseline_k=args.baseline_k,
            baseline_threshold=args.baseline_threshold,
            baseline_temporal_policy=args.baseline_temporal_policy,
            fps_values=parse_pruning_ablation_float_values(
                args.fps_values,
                name="FPS",
                minimum=1e-9,
            ),
            k_values=parse_k_values(args.k_values),
            threshold_values=parse_pruning_ablation_float_values(
                args.threshold_values,
                name="threshold",
                minimum=-1.0,
                maximum=1.0,
            ),
            temporal_policies=parse_temporal_policies(args.temporal_policies),
            preserve_shared_anchor_seconds=args.preserve_shared_anchor_seconds,
            min_pruned_video_seconds=args.min_pruned_video_seconds,
            pruning_protection_mode=args.pruning_protection_mode,
            min_pruned_video_percent=args.min_pruned_video_percent,
            random_seed=args.random_seed,
            ffmpeg_binary=args.ffmpeg_binary,
            download_media=args.download_media,
        )
        print(
            f"wrote {summary['pair_count']} synchronized pairs and "
            f"{summary['variant_count']} controlled pruning variants to {args.output_dir}"
        )
        return 0
    if args.command == "add_object_hints":
        rows = enrich_evidence_with_object_hints(
            evidence_path=args.evidence,
            output_path=args.output,
            output_dir=args.output_dir,
            backend=args.backend,
            api_key=args.api_key,
            model_id=args.model_id,
            base_url=args.base_url,
            max_new_tokens=args.max_new_tokens,
            max_image_pixels=args.max_image_pixels,
            dtype=args.dtype,
            allow_cpu=args.allow_cpu,
            disable_thinking=args.disable_thinking,
            frame_sampling_mode=args.frame_sampling_mode,
            frames_per_clip=args.frames_per_clip,
            objects_per_clip=args.objects_per_clip,
            objects_per_packet=args.objects_per_packet,
            n_workers=args.n_workers,
            max_packets=args.max_packets,
            random_seed=args.random_seed,
            enable_reid=not args.disable_reid,
            reid_model_id=args.reid_model_id,
            reid_device=args.reid_device,
            reid_visual_threshold=args.reid_visual_threshold,
            reid_text_threshold=args.reid_text_threshold,
            reid_batch_size=args.reid_batch_size,
        )
        print(f"wrote {len(rows)} object-hinted evidence packets to {args.output}")
        return 0
    if args.command == "generate_video_qa_loop":
        rows = generate_video_qa_loop(
            evidence_path=args.evidence,
            output_path=args.output,
            prompts_path=args.prompts_output,
            rejected_path=args.rejected_output,
            intermediate_path=args.intermediate_output,
            backend=args.backend,
            model_id=args.model_id,
            base_url=args.base_url,
            target_count=args.target_count,
            max_attempts=args.max_attempts,
            max_new_tokens=args.max_new_tokens,
            max_image_pixels=args.max_image_pixels,
            dtype=args.dtype,
            allow_cpu=args.allow_cpu,
            allow_openai_video_input=args.allow_openai_video_input,
            disable_thinking=args.disable_thinking,
            api_key=args.api_key,
            judge_backend=args.judge_backend,
            judge_model_id=args.judge_model_id,
            judge_base_url=args.judge_base_url,
            judge_api_key=args.judge_api_key,
            judge_max_new_tokens=args.judge_max_new_tokens,
            judge_reasoning_effort=args.judge_reasoning_effort,
            qa_formality_use_generator=args.qa_formality_use_generator,
            judge_include_generator_rationale=args.judge_include_generator_rationale,
            # Archived point-scoring CLI plumbing:
            # judge_pass_fail_only=args.judge_pass_fail_only,
            # judge_quality_quota=args.judge_quality_quota,
            dry_run=args.dry_run,
            generation_mode=args.generation_mode,
            fixed_question_type_schedule=args.fixed_question_type_schedule,
            question_types=parse_question_types(args.question_types),
            resume=args.resume,
            generator_decode_mode=args.generator_decode_mode,
            generator_temperature=args.generator_temperature,
            generator_top_p=args.generator_top_p,
            generator_top_k=args.generator_top_k,
        )
        print(f"accepted {len(rows)} video-first question-answer rows")
        return 0
    if args.command == "validate_outputs":
        return validate_outputs(
            qa_path=args.qa,
            report_path=args.report,
            csv_path=args.csv_output,
            human_review_path=args.human_review_output,
            strict_review=args.strict_review,
        )
    if args.command == "materialize_review_videos":
        manifest = materialize_review_videos(
            evidence_path=args.evidence,
            qa_path=args.qa,
            output_dir=args.output_dir,
            download_missing=not args.no_download,
        )
        print(
            "materialized "
            f"{manifest['video_count_ok']} videos to {manifest['videos_dir']} "
            f"({manifest['video_count_error']} errors, {manifest['video_count_missing']} missing)"
        )
        return 0
    if args.command == "hpc_preflight_cached_evidence":
        return preflight_cached_evidence(
            evidence_path=args.evidence,
            target_count=args.target_count,
            cache_dir=args.cache_dir,
            resolved_output=args.resolved_output,
        )
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
