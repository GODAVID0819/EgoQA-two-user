"""Summarize the H200 six-user preprocessing throughput sweep.

The launcher supplies paths and profile definitions through environment
variables. The summary treats CUDA-keeper utilization as scheduler overhead
and independently validates that every generator packet uses complete,
unpruned sampled-frame sets.
"""

from __future__ import annotations

import csv
import json
import os
import re
import statistics
from pathlib import Path
from typing import Any


FULL_FRAME_POLICY = "complete_full_unpruned_sampled_frames"
FULL_FRAME_MODE = "full_unpruned_sampled_frames_only"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def matched_floats(pattern: str, text: str) -> list[float]:
    return [float(value) for value in re.findall(pattern, text)]


def log_breakdown(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    encoder = matched_floats(
        r"ten_minute_prepare status=encoder_loaded .*?seconds=([0-9.]+)", text
    )
    source = matched_floats(
        r"ten_minute_prepare status=source_analysis_completed .*?seconds=([0-9.]+)",
        text,
    )
    evidence = matched_floats(
        r"evidence_window status=prepared .*?seconds=([0-9.]+)", text
    )
    decode = matched_floats(
        r"sampled_video_rgb status=decoded .*?seconds=([0-9.]+)", text
    )
    persist = matched_floats(
        r"sampled_video_rgb status=persisted .*?persist_seconds=([0-9.]+)", text
    )
    sample_total = matched_floats(
        r"sampled_video_rgb status=persisted .*?total_seconds=([0-9.]+)", text
    )
    sampling_wall = matched_floats(
        r"six_user_analysis status=sampling_completed .*?seconds=([0-9.]+)", text
    )
    embedding = matched_floats(
        r"six_user_analysis status=embedding_completed .*?seconds=([0-9.]+)", text
    )
    media_prepare = matched_floats(
        r"evidence_window status=parallel_user_media_preparation_completed .*?seconds=([0-9.]+)",
        text,
    )
    return {
        "encoder_load_seconds": round(sum(encoder), 6),
        "source_analysis_wall_seconds": round(sum(source), 6),
        "evidence_window_work_seconds": round(sum(evidence), 6),
        "direct_rgb_decode_worker_seconds": round(sum(decode), 6),
        "png_persist_worker_seconds": round(sum(persist), 6),
        "decode_plus_persist_worker_seconds": round(sum(sample_total), 6),
        "video_sampling_wall_seconds": round(sum(sampling_wall), 6),
        "clip_embedding_wall_seconds": round(sum(embedding), 6),
        "parallel_media_prepare_wall_seconds": round(sum(media_prepare), 6),
        "source_analysis_count": len(source),
        "sampled_video_count": len(decode),
        "sampled_video_cache_hit_count": len(
            re.findall(r"sampled_video_rgb status=cache_hit", text)
        ),
        "embedding_cache_hit_window_count": len(
            re.findall(
                r"six_user_analysis status=embedding_completed .*?cache_hit=true", text
            )
        ),
    }


def validate_full_unpruned_packets(
    packets: list[dict[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    packet_frame_counts: list[int] = []
    for packet_index, packet in enumerate(packets):
        budget = packet.get("generator_context_budget") or {}
        source_counts = budget.get("per_user_source_frame_counts")
        model_counts = budget.get("per_user_model_input_frame_counts")
        if budget.get("policy") != FULL_FRAME_POLICY:
            errors.append(f"packet {packet_index}: wrong generator context policy")
        if source_counts != model_counts:
            errors.append(f"packet {packet_index}: source/model frame counts differ")
        clips = list(packet.get("clips") or [])
        if len(clips) != 6:
            errors.append(f"packet {packet_index}: expected 6 clips, got {len(clips)}")
        total = 0
        for clip_index, clip in enumerate(clips):
            frames = list(clip.get("frames") or [])
            total += len(frames)
            context = clip.get("context_sampling") or {}
            pruning = clip.get("temporal_pruning") or {}
            if clip.get("generator_media_mode") != FULL_FRAME_MODE:
                errors.append(
                    f"packet {packet_index} clip {clip_index}: wrong generator media mode"
                )
            if clip.get("is_pruned") is not False:
                errors.append(
                    f"packet {packet_index} clip {clip_index}: generator marked pruned"
                )
            if context.get("policy") != FULL_FRAME_POLICY:
                errors.append(
                    f"packet {packet_index} clip {clip_index}: wrong context policy"
                )
            if int(context.get("source_frame_count") or 0) != len(frames):
                errors.append(
                    f"packet {packet_index} clip {clip_index}: source count mismatch"
                )
            if int(context.get("model_input_frame_count") or 0) != len(frames):
                errors.append(
                    f"packet {packet_index} clip {clip_index}: model count mismatch"
                )
            if pruning.get("analysis_pruning_applied_to_generator") is not False:
                errors.append(
                    f"packet {packet_index} clip {clip_index}: pruning reached generator"
                )
            if int(pruning.get("analysis_source_sampled_frame_count") or 0) != len(
                frames
            ):
                errors.append(
                    f"packet {packet_index} clip {clip_index}: analysis count mismatch"
                )
        packet_frame_counts.append(total)
        if int(budget.get("model_input_frame_count") or 0) != total:
            errors.append(f"packet {packet_index}: aggregate frame count mismatch")
    return {
        "passed": not errors,
        "packet_count": len(packets),
        "packet_frame_counts": packet_frame_counts,
        "error_count": len(errors),
        "errors": errors[:50],
    }


def gpu_summary(rows: list[dict[str, float]]) -> dict[str, Any] | None:
    if not rows:
        return None
    utils = [row["util"] for row in rows]
    return {
        "samples": len(rows),
        "sm_util_mean_percent": round(statistics.fmean(utils), 6),
        "fraction_below_10_percent": round(
            sum(value < 10 for value in utils) / len(utils), 6
        ),
        "fraction_at_least_80_percent": round(
            sum(value >= 80 for value in utils) / len(utils), 6
        ),
        "max_memory_used_mib": max(row["memory"] for row in rows),
    }


def main() -> None:
    root = Path(os.environ["EXPERIMENT_ROOT"])
    profile_root = Path(os.environ["PROFILE_OUTPUT_BASE"])
    experiment_name = os.environ["EXPERIMENT_NAME"]
    job_id = os.environ["SLURM_JOB_ID"]
    labels = os.environ["RUN_PROFILE_LABELS"].split()
    configs = {
        label: {
            "clip_batch_size": int(batch),
            "video_sample_workers": int(video_workers),
            "media_prepare_workers": int(media_workers),
        }
        for label, batch, video_workers, media_workers in zip(
            labels,
            os.environ["RUN_PROFILE_CLIP_BATCHES"].split(),
            os.environ["RUN_PROFILE_VIDEO_WORKERS"].split(),
            os.environ["RUN_PROFILE_MEDIA_PREPARE_WORKERS"].split(),
            strict=True,
        )
    }
    with Path(os.environ["STATUS_PATH"]).open(encoding="utf-8", newline="") as handle:
        statuses = list(csv.DictReader(handle, delimiter="\t"))
    status_by_label = {row["profile"]: row for row in statuses}

    profiles: dict[str, dict[str, Any]] = {}
    for label in labels:
        status = status_by_label.get(label)
        child = profile_root / f"{experiment_name}_{label}_{job_id}"
        timing_path = child / "preprocessing_timing.json"
        if status is None or not timing_path.is_file():
            profiles[label] = {
                "status": status.get("status") if status else "not_run",
                "exit_code": int(status["exit_code"]) if status else None,
                "configuration": configs[label],
                "preprocessing_completed": False,
            }
            continue
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
        packets = read_jsonl(Path(timing["candidate_jsonl"]))
        profiles[label] = {
            "status": status["status"],
            "exit_code": int(status["exit_code"]),
            "configuration": configs[label],
            "preprocessing_completed": True,
            "runtime_wall_seconds": int(status["end_epoch"])
            - int(status["start_epoch"]),
            "useful_preprocessing_seconds": int(timing["elapsed_seconds"]),
            "breakdown": log_breakdown(root / f"{label}_stdout_timeline.log"),
            "candidate_count": len(packets),
            "source_window_evidence_ids": sorted(
                {str(row.get("source_window_evidence_id") or "") for row in packets}
            ),
            "candidate_evidence_ids": [
                str(row.get("evidence_id") or "") for row in packets
            ],
            "full_unpruned_generator_contract": validate_full_unpruned_packets(
                packets
            ),
        }

    baseline_labels = ["baseline_a_b128_w3_m3", "baseline_b_b128_w3_m3"]
    baseline_seconds = [
        profiles[label]["useful_preprocessing_seconds"]
        for label in baseline_labels
        if profiles.get(label, {}).get("preprocessing_completed")
    ]
    baseline_mean = statistics.fmean(baseline_seconds) if baseline_seconds else None
    completed = {
        label: profile
        for label, profile in profiles.items()
        if profile.get("preprocessing_completed")
    }
    reference = completed.get("baseline_a_b128_w3_m3")
    for profile in completed.values():
        seconds = profile["useful_preprocessing_seconds"]
        profile["speedup_vs_bracketed_baseline_mean"] = (
            round(baseline_mean / seconds, 6) if baseline_mean else None
        )
        profile["candidate_ids_match_baseline_a"] = bool(
            reference
            and profile["candidate_evidence_ids"]
            == reference["candidate_evidence_ids"]
        )
        profile["source_windows_match_baseline_a"] = bool(
            reference
            and profile["source_window_evidence_ids"]
            == reference["source_window_evidence_ids"]
        )

    fastest_label = min(
        completed,
        key=lambda label: completed[label]["useful_preprocessing_seconds"],
        default=None,
    )
    all_contracts_pass = bool(completed) and all(
        profile["full_unpruned_generator_contract"]["passed"]
        for profile in completed.values()
    )
    all_outputs_match = bool(reference) and all(
        profile["candidate_ids_match_baseline_a"]
        and profile["source_windows_match_baseline_a"]
        for profile in completed.values()
    )

    gpu_rows: list[dict[str, float]] = []
    gpu_path = Path(os.environ["JOB_GPU_LOG"])
    if gpu_path.is_file():
        with gpu_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(
                line for line in handle if not line.startswith("#")
            ):
                try:
                    gpu_rows.append(
                        {
                            "epoch": float(row["sample_epoch"]),
                            "util": float(row["sm_util_percent"]),
                            "memory": float(row["memory_used_mib"]),
                        }
                    )
                except (KeyError, TypeError, ValueError):
                    pass
    gpu_profiles: dict[str, Any] = {}
    for label, status in status_by_label.items():
        start = float(status["start_epoch"])
        end = float(status["end_epoch"])
        gpu_profiles[label] = gpu_summary(
            [row for row in gpu_rows if start <= row["epoch"] <= end]
        )

    payload = {
        "status": (
            "completed"
            if int(os.environ["OVERALL_EXIT_CODE"]) == 0
            else "partial_failure"
        ),
        "baseline": {
            "replicate_labels": baseline_labels,
            "useful_preprocessing_seconds": baseline_seconds,
            "mean_useful_preprocessing_seconds": baseline_mean,
        },
        "fastest_profile": (
            {
                "label": fastest_label,
                "useful_preprocessing_seconds": completed[fastest_label][
                    "useful_preprocessing_seconds"
                ],
                "speedup_vs_bracketed_baseline_mean": completed[fastest_label][
                    "speedup_vs_bracketed_baseline_mean"
                ],
                "configuration": completed[fastest_label]["configuration"],
            }
            if fastest_label
            else None
        ),
        "all_candidate_and_source_outputs_match": all_outputs_match,
        "all_full_unpruned_generator_contracts_pass": all_contracts_pass,
        "profiles": profiles,
        "scheduler_gpu_including_keeper": {
            "job_wide": gpu_summary(gpu_rows),
            "profiles": gpu_profiles,
        },
        "interpretation": (
            "Baseline replicates bracket the sweep. Each arm uses a separate cold "
            "preprocessing cache. CUDA-keeper work is scheduler overhead, not useful "
            "preprocessing throughput."
        ),
    }
    output = root / "preprocessing_throughput_sweep.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("preprocessing_throughput_sweep", json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
