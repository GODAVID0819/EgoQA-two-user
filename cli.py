"""Unified CLI for the EgoLife two-user QA pilot."""

from __future__ import annotations

import argparse

from .evidence import prepare_evidence
from .manifest import build_manifest
from .candidate_mining import mine_candidates
from .clip_gap_demo import (
    run_clip_gap_demo,
    run_diverse_packet_trials,
    run_random_clip_gap_trials,
)
from .clip_exclusive_mining import mine_clip_exclusive_candidates
from .observations import observe_clips
from .qa_pipeline import add_runner_args, validate_outputs
from .review_media import materialize_review_videos
from .video_qa_loop import add_video_loop_args, generate_video_qa_loop


def _csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


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

    video_gen = sub.add_parser("generate_video_qa_loop", help="Generate video-first QA with judge/eval retry loop")
    video_gen.add_argument("--evidence", required=True)
    video_gen.add_argument("--output", required=True)
    video_gen.add_argument("--prompts-output")
    video_gen.add_argument("--rejected-output")
    video_gen.add_argument("--intermediate-output")
    video_gen.add_argument("--target-count", type=int, default=20)
    video_gen.add_argument("--max-attempts", type=int, default=3)
    add_video_loop_args(video_gen)

    val = sub.add_parser("validate_outputs", help="Validate QA JSONL and write report/CSV")
    val.add_argument("--qa", required=True)
    val.add_argument("--report", required=True)
    val.add_argument("--csv-output")
    val.add_argument("--human-review-output")
    val.add_argument("--strict-review", action="store_true")

    review_media = sub.add_parser("materialize_review_videos", help="Copy/download videos for manual QA review")
    review_media.add_argument("--evidence")
    review_media.add_argument("--qa")
    review_media.add_argument("--output-dir", required=True)
    review_media.add_argument("--no-download", action="store_true")

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
            dry_run=args.dry_run,
            generation_mode=args.generation_mode,
            fixed_question_type_schedule=args.fixed_question_type_schedule,
            resume=args.resume,
        )
        print(f"accepted {len(rows)} video-first QA rows")
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
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
