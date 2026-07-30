"""Turn fixed-cohort pruning sweeps into controlled QA-generation arms."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from pathlib import Path
from typing import Any, Iterable, Sequence

from .evidence import group_manifest_clips
from .io_utils import iter_jsonl, read_json, stable_id, write_json, write_jsonl


GENERATION_SWEEPS = ("threshold", "sampling", "k")


def parse_generation_sweeps(value: str | Iterable[str]) -> list[str]:
    raw = value.split(",") if isinstance(value, str) else value
    sweeps: list[str] = []
    for item in raw:
        sweep = str(item).strip()
        if not sweep:
            continue
        if sweep not in GENERATION_SWEEPS:
            raise ValueError(
                f"unknown generation ablation sweep {sweep!r}; "
                f"expected one of {GENERATION_SWEEPS}"
            )
        if sweep not in sweeps:
            sweeps.append(sweep)
    if not sweeps:
        raise ValueError("at least one generation ablation sweep is required")
    return sweeps


def _manifest_clip_index(manifest: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for group in group_manifest_clips(manifest):
        day = str(group.get("day") or "")
        time_token = str(group.get("time_token") or "")
        for clip in group.get("clips", []):
            agent_dir = str(clip.get("agent_dir") or "")
            key = (day, time_token, agent_dir)
            if all(key):
                index[key] = dict(clip)
    return index


def _variant_is_usable(row: dict[str, Any]) -> bool:
    if row.get("passed") is not True:
        return False
    if row.get("left_materialization_status") != "materialized":
        return False
    if row.get("right_materialization_status") != "materialized":
        return False
    for key in ("left_pruned_video", "right_pruned_video"):
        value = row.get(key)
        if not value or not Path(value).is_file():
            return False
    return True


def _is_baseline(row: dict[str, Any], settings: dict[str, Any]) -> bool:
    return (
        float(row["fps"]) == float(settings["baseline_fps"])
        and int(row["k"]) == int(settings["baseline_k"])
        and float(row["high_similarity_threshold"]) == float(settings["baseline_threshold"])
        and str(row["temporal_policy"]) == str(settings["baseline_temporal_policy"])
    )


def _clip_for_variant(
    source_clip: dict[str, Any],
    *,
    side: str,
    pair_id: str,
    original_video: str,
    pruned_video: str,
    pruning: dict[str, Any],
) -> dict[str, Any]:
    clip = dict(source_clip)
    source_local_video = clip.get("local_video")
    keep_intervals = pruning.get(f"{side}_keep_intervals", [])
    remove_intervals = pruning.get(f"{side}_remove_intervals", [])
    cluster_decisions = pruning.get(f"{side}_cluster_decisions", [])
    kept_cluster_representatives = [
        decision.get("representative")
        for decision in cluster_decisions
        if decision.get("pruned") is not True and isinstance(decision.get("representative"), dict)
    ]
    clip.update(
        {
            "source_local_video": source_local_video,
            "original_local_video": original_video,
            "full_local_video": original_video,
            "local_video": pruned_video,
            "generator_local_video": pruned_video,
            "generator_media_mode": "pruned_video",
            "temporal_pruning": {
                "side": side,
                "pair_key": pair_id,
                "source_local_video": source_local_video,
                "original_local_video": original_video,
                "pruned_local_video": pruned_video,
                "method": pruning.get("method"),
                "high_similarity_threshold": pruning.get("high_similarity_threshold"),
                "pruning_protection_mode": pruning.get("pruning_protection_mode"),
                "min_pruned_video_percent": pruning.get("min_pruned_video_percent"),
                "protection_target_kept_seconds": pruning.get("protection_target_kept_seconds"),
                "required_kept_duration_seconds": pruning.get("required_kept_duration_seconds"),
                "keep_intervals": keep_intervals,
                "remove_intervals": remove_intervals,
                "cluster_decisions": cluster_decisions,
                "kept_cluster_representatives": kept_cluster_representatives,
                "kept_cluster_count": len(kept_cluster_representatives),
                "restored_frame_indices": pruning.get(f"{side}_restored_frame_indices", []),
                "restored_frames": pruning.get(f"{side}_restored_frames", []),
                "preserved_shared_intervals": pruning.get("preserved_shared_intervals", []),
                "kept_duration_seconds": pruning.get(f"{side}_kept_duration_seconds"),
                "removed_duration_seconds": pruning.get(f"{side}_removed_duration_seconds"),
            },
            "benchmark_media": {
                "generator_video": pruned_video,
                "judge_video": original_video,
                "answerability_video": original_video,
                "source_cache_video": source_local_video,
            },
        }
    )
    return clip


def _packet_for_variant(
    pair: dict[str, Any],
    variant: dict[str, Any],
    *,
    clip_index: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[str, Any]:
    diagnostics = read_json(variant["diagnostics_path"])
    pruning = diagnostics["temporal_pruning"]
    day = str(pair.get("day") or "")
    time_token = str(pair.get("time_token") or "")
    agents = [str(pair["left_agent"]), str(pair["right_agent"])]
    source_clips = []
    for agent in agents:
        key = (day, time_token, agent)
        if key not in clip_index:
            raise ValueError(
                f"fixed ablation pair {pair['pair_id']} is absent from the manifest: "
                f"{day}/{time_token}/{agent}"
            )
        source_clips.append(clip_index[key])
    clips = [
        _clip_for_variant(
            source_clips[0],
            side="left",
            pair_id=str(pair["pair_id"]),
            original_video=str(pair["left_original_video"]),
            pruned_video=str(variant["left_pruned_video"]),
            pruning=pruning,
        ),
        _clip_for_variant(
            source_clips[1],
            side="right",
            pair_id=str(pair["pair_id"]),
            original_video=str(pair["right_original_video"]),
            pruned_video=str(variant["right_pruned_video"]),
            pruning=pruning,
        ),
    ]
    required_users = [clip.get("agent_name") for clip in clips]
    evidence_id = stable_id(
        "EGOLIFE2U_GENERATION_ABLATION",
        pair["pair_id"],
        variant["variant_id"],
    )
    return {
        "evidence_id": evidence_id,
        "candidate_type": "fixed_cohort_pruning_generation_ablation",
        "day": pair.get("day"),
        "time_token": pair.get("time_token"),
        "clip_clock": source_clips[0].get("clip_clock"),
        "required_users": required_users,
        "speaker_user": required_users[0],
        "evidence_provider_user": required_users[1],
        "requirement": (
            "Controlled ablation packet from a fixed synchronized pair cohort. Generation uses "
            "the two pruned videos. The runtime judge-video-source setting determines whether "
            "visual judges receive the pruned videos or the original 30-second windows."
        ),
        "generator_media_mode": "pruned_video",
        "clips": clips,
        "source_urls": {
            "videos": [clip.get("video_url") for clip in clips],
            "gazes": [clip.get("gaze_url") for clip in clips],
            "overlays": [clip.get("overlay_url") for clip in clips if clip.get("overlay_url")],
        },
        "generation_ablation": {
            "pair_id": pair["pair_id"],
            "sweep": variant["sweep"],
            "variant_id": variant["variant_id"],
            "variant_label": variant["variant_label"],
            "fps": variant["fps"],
            "sample_interval_seconds": variant["sample_interval_seconds"],
            "k": variant["k"],
            "high_similarity_threshold": variant["high_similarity_threshold"],
            "temporal_policy": variant["temporal_policy"],
            "max_pair_time_difference_seconds": variant.get(
                "max_pair_time_difference_seconds"
            ),
            "pruning_passed": variant["passed"],
            "diagnostics_path": variant["diagnostics_path"],
            "cluster_trace_path": variant["cluster_trace_path"],
        },
    }


def prepare_generation_ablation(
    *,
    manifest_path: str | Path,
    cohort_path: str | Path,
    metrics_path: str | Path,
    pruning_summary_path: str | Path,
    output_dir: str | Path,
    sweeps: Sequence[str] = GENERATION_SWEEPS,
) -> dict[str, Any]:
    """Write matched evidence files and a Slurm-readable generation arm manifest."""

    selected_sweeps = parse_generation_sweeps(sweeps)
    manifest = read_json(manifest_path)
    pruning_summary = read_json(pruning_summary_path)
    settings = pruning_summary["settings"]
    clip_index = _manifest_clip_index(manifest)
    pairs = {str(row["pair_id"]): row for row in iter_jsonl(cohort_path)}
    selected_metrics = [
        row for row in iter_jsonl(metrics_path) if str(row.get("sweep")) in selected_sweeps
    ]
    metrics_by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_metrics:
        metrics_by_variant[str(row["variant_id"])].append(row)
    if not metrics_by_variant:
        raise ValueError("no pruning variants matched the requested generation sweeps")

    usable_pair_ids_by_variant = {
        variant_id: {
            str(row["pair_id"]) for row in rows if _variant_is_usable(row)
        }
        for variant_id, rows in metrics_by_variant.items()
    }
    common_pair_ids = set(pairs)
    for pair_ids in usable_pair_ids_by_variant.values():
        common_pair_ids &= pair_ids
    ordered_pair_ids = [pair_id for pair_id in pairs if pair_id in common_pair_ids]
    if not ordered_pair_ids:
        raise RuntimeError(
            "no synchronized pair is usable across every requested pruning variant"
        )

    output_root = Path(output_dir)
    evidence_paths: dict[str, str] = {}
    variants: dict[str, dict[str, Any]] = {}
    for variant_id, rows in metrics_by_variant.items():
        by_pair = {str(row["pair_id"]): row for row in rows}
        first = rows[0]
        variant_dir = output_root / "evidence" / str(first["sweep"]) / variant_id
        evidence_path = variant_dir / "evidence_pruned_pairs.jsonl"
        packets = [
            _packet_for_variant(
                pairs[pair_id],
                by_pair[pair_id],
                clip_index=clip_index,
            )
            for pair_id in ordered_pair_ids
        ]
        write_jsonl(evidence_path, packets)
        evidence_paths[variant_id] = str(evidence_path)
        variants[variant_id] = first

    baseline_candidates = [
        variant_id
        for variant_id, row in variants.items()
        if _is_baseline(row, settings)
    ]
    if not baseline_candidates:
        raise ValueError(
            "the selected sweeps do not include the declared baseline configuration"
        )
    baseline_variant_id = min(
        baseline_candidates,
        key=lambda variant_id: (
            selected_sweeps.index(str(variants[variant_id]["sweep"])),
            variant_id,
        ),
    )

    arms = [
        {
            "arm_id": "control_full_judges",
            "ablation": "control",
            "sweep": "control",
            "variant_id": baseline_variant_id,
            "judge_video_source": "full",
        },
        {
            "arm_id": "judge_pruned_videos",
            "ablation": "judge_video_source",
            "sweep": "judge_video_source",
            "variant_id": baseline_variant_id,
            "judge_video_source": "pruned",
        },
    ]
    for variant_id, variant in variants.items():
        if _is_baseline(variant, settings):
            continue
        arms.append(
            {
                "arm_id": variant_id,
                "ablation": str(variant["sweep"]),
                "sweep": str(variant["sweep"]),
                "variant_id": variant_id,
                "judge_video_source": "full",
            }
        )

    arm_rows = []
    for arm_index, arm in enumerate(arms):
        variant = variants[arm["variant_id"]]
        arm_rows.append(
            {
                "arm_index": arm_index,
                **arm,
                "evidence_path": evidence_paths[arm["variant_id"]],
                "target_count": len(ordered_pair_ids),
                "pair_ids": ordered_pair_ids,
                "configuration": {
                    "fps": variant["fps"],
                    "sample_interval_seconds": variant["sample_interval_seconds"],
                    "k": variant["k"],
                    "high_similarity_threshold": variant[
                        "high_similarity_threshold"
                    ],
                    "temporal_policy": variant["temporal_policy"],
                },
            }
        )
    arm_manifest_path = output_root / "generation_arms.jsonl"
    write_jsonl(arm_manifest_path, arm_rows)
    summary = {
        "manifest_path": str(manifest_path),
        "cohort_path": str(cohort_path),
        "metrics_path": str(metrics_path),
        "pruning_summary_path": str(pruning_summary_path),
        "output_dir": str(output_root),
        "selected_sweeps": selected_sweeps,
        "source_pair_count": len(pairs),
        "common_usable_pair_count": len(ordered_pair_ids),
        "excluded_pair_ids": [pair_id for pair_id in pairs if pair_id not in common_pair_ids],
        "variant_count": len(variants),
        "baseline_variant_id": baseline_variant_id,
        "generation_arm_count": len(arm_rows),
        "generation_arm_manifest": str(arm_manifest_path),
        "arms": arm_rows,
    }
    write_json(output_root / "generation_ablation_summary.json", summary)
    return summary


GENERATION_RESULT_FIELDS = (
    "arm_index",
    "arm_id",
    "ablation",
    "variant_id",
    "judge_video_source",
    "fps",
    "sample_interval_seconds",
    "k",
    "high_similarity_threshold",
    "target_count",
    "accepted_count",
    "rejected_count",
    "completed_count",
    "acceptance_rate",
    "coverage_complete",
    "generation_attempt_count",
    "mean_attempts_per_completed_packet",
    "qa_formality_failure_count",
    "evidence_groundedness_failure_count",
    "answerability_failure_count",
    "unclassified_rejection_count",
    "judge_media_mismatch_count",
    "generator_rationale_exposure_count",
    "output_dir",
)


def _optional_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path)) if path.is_file() else []


def _write_result_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=GENERATION_RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in GENERATION_RESULT_FIELDS})


def summarize_generation_ablation(
    *,
    arm_manifest_path: str | Path,
    generation_output_root: str | Path,
    output_dir: str | Path,
    require_complete: bool = False,
) -> dict[str, Any]:
    """Aggregate completed or partial QA-generation arms with routing audits."""

    arms = list(iter_jsonl(arm_manifest_path))
    if not arms:
        raise ValueError("generation arm manifest is empty")
    generation_root = Path(generation_output_root)
    results = []
    incomplete_arm_ids = []
    for arm in arms:
        arm_id = str(arm["arm_id"])
        arm_output = generation_root / arm_id
        evidence = _optional_jsonl(Path(arm["evidence_path"]))
        accepted = _optional_jsonl(arm_output / "qa_mcq.jsonl")
        rejected = _optional_jsonl(arm_output / "qa_mcq.rejected.jsonl")
        prompts = _optional_jsonl(arm_output / "video_first_prompts.jsonl")
        final = [*accepted, *rejected]
        evidence_ids = {str(row.get("evidence_id") or "") for row in evidence}
        final_ids = [str(row.get("evidence_id") or "") for row in final]
        coverage_complete = (
            len(final_ids) == len(set(final_ids))
            and set(final_ids) == evidence_ids
            and len(final) == int(arm["target_count"])
        )
        if not coverage_complete:
            incomplete_arm_ids.append(arm_id)

        failure_counts: Counter[str] = Counter()
        for row in rejected:
            checks = (row.get("review") or {}).get("checks") or {}
            failed = [
                name
                for name in (
                    "qa_formality",
                    "evidence_groundedness",
                    "answerability",
                )
                if isinstance(checks.get(name), dict)
                and checks[name].get("status") == "FAIL"
            ]
            if failed:
                failure_counts.update(failed)
            else:
                failure_counts["unclassified"] += 1

        attempt_counts = [
            int(row.get("attempt_count") or 0) for row in accepted
        ] + [len(row.get("attempts") or []) for row in rejected]
        visual_prompt_rows = [
            row
            for row in prompts
            if row.get("stage") in {"evidence_groundedness_judge", "answerability"}
        ]
        expected_media_role = str(arm["judge_video_source"])
        judge_media_mismatch_count = sum(
            str(row.get("media_role") or "") != expected_media_role
            for row in visual_prompt_rows
        )
        generator_rationale_exposure_count = sum(
            row.get("generator_rationale_included") is True
            for row in prompts
            if row.get("stage") in {
                "qa_formality_judge",
                "evidence_groundedness_judge",
                "answerability",
            }
        )
        output_source_mismatch_count = sum(
            str(row.get("judge_video_source") or "") != expected_media_role
            for row in final
        )
        judge_media_mismatch_count += output_source_mismatch_count
        configuration = arm["configuration"]
        completed_count = len(final)
        results.append(
            {
                "arm_index": arm["arm_index"],
                "arm_id": arm_id,
                "ablation": arm["ablation"],
                "variant_id": arm["variant_id"],
                "judge_video_source": expected_media_role,
                "fps": configuration["fps"],
                "sample_interval_seconds": configuration["sample_interval_seconds"],
                "k": configuration["k"],
                "high_similarity_threshold": configuration[
                    "high_similarity_threshold"
                ],
                "target_count": arm["target_count"],
                "accepted_count": len(accepted),
                "rejected_count": len(rejected),
                "completed_count": completed_count,
                "acceptance_rate": (
                    round(len(accepted) / completed_count, 6)
                    if completed_count
                    else None
                ),
                "coverage_complete": coverage_complete,
                "generation_attempt_count": sum(
                    row.get("stage") == "generation" for row in prompts
                ),
                "mean_attempts_per_completed_packet": (
                    round(sum(attempt_counts) / len(attempt_counts), 6)
                    if attempt_counts
                    else None
                ),
                "qa_formality_failure_count": failure_counts["qa_formality"],
                "evidence_groundedness_failure_count": failure_counts[
                    "evidence_groundedness"
                ],
                "answerability_failure_count": failure_counts["answerability"],
                "unclassified_rejection_count": failure_counts["unclassified"],
                "judge_media_mismatch_count": judge_media_mismatch_count,
                "generator_rationale_exposure_count": generator_rationale_exposure_count,
                "output_dir": str(arm_output),
            }
        )

    output_root = Path(output_dir)
    write_jsonl(output_root / "generation_ablation_results.jsonl", results)
    _write_result_csv(output_root / "generation_ablation_results.csv", results)
    summary = {
        "arm_manifest_path": str(arm_manifest_path),
        "generation_output_root": str(generation_root),
        "arm_count": len(arms),
        "complete_arm_count": len(arms) - len(incomplete_arm_ids),
        "incomplete_arm_count": len(incomplete_arm_ids),
        "incomplete_arm_ids": incomplete_arm_ids,
        "all_media_routing_checks_passed": all(
            row["judge_media_mismatch_count"] == 0 for row in results
        ),
        "generator_rationale_exposure_count": sum(
            row["generator_rationale_exposure_count"] for row in results
        ),
        "results": results,
    }
    write_json(output_root / "generation_ablation_results_summary.json", summary)
    if require_complete and incomplete_arm_ids:
        raise RuntimeError(
            "generation ablation has incomplete arms: "
            + ", ".join(incomplete_arm_ids)
        )
    return summary
