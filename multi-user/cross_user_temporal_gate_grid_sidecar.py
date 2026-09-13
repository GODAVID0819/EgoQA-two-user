"""Fixed-cohort hard temporal-gate sweep for cross-user cluster pruning.

The experiment reuses complete CLIP embedding caches from a prior temporal
K-means run.  It compares the exact production control (cosine-only K-means
and cosine-only cross-user pruning) with time-aware within-video clustering at
``time_weight=0.1`` followed by either a center-gap or interval-gap hard gate.
The production modules remain untouched; a gate value of ``None`` delegates to
the existing sidecar pruning semantics exactly.
"""

from __future__ import annotations

import argparse
import html
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .group_relative_clip_sampling import _side_best_frame_matches
from .io_utils import iter_jsonl, read_json, write_json
from .temporal_kmeans_grid_sidecar import (
    CLUSTER_DISTANCE_FORMULA,
    DEFAULT_CLIP_MODEL,
    DEFAULT_DURATIONS_SECONDS,
    DEFAULT_SECONDS_PER_CLUSTER,
    DEFAULT_SIMILARITY_THRESHOLDS,
    DEFAULT_TEMPORAL_UNIT_SECONDS,
    _cosine_matrix,
    _duration_prefix,
    _expected_frame_count,
    _load_pair_embedding_cache,
    _number_slug,
    _write_csv_atomic,
    _write_json_atomic,
    _write_jsonl_atomic,
    cluster_quality_metrics,
    cross_cluster_temporal_gap_matrices,
    parse_float_grid,
    prune_time_aware_cluster_pair,
    time_aware_clustered_frame_representatives,
)


DEFAULT_PAIR_COUNT = 50
DEFAULT_WITHIN_TIME_WEIGHT = 0.1
DEFAULT_CROSS_GAP_MODES = ("center", "interval")
DEFAULT_CROSS_GAP_SECONDS = (0.0, 5.0, 10.0, 15.0, 30.0, 60.0, 120.0)

METRIC_FIELDS = (
    "pair_id",
    "day",
    "time_token",
    "left_agent",
    "right_agent",
    "duration_seconds",
    "sample_interval_seconds",
    "seconds_per_cluster",
    "k",
    "variant_kind",
    "within_time_weight",
    "temporal_unit_seconds",
    "high_similarity_threshold",
    "cross_gap_mode",
    "max_cross_gap_seconds",
    "left_cluster_count",
    "right_cluster_count",
    "mean_cluster_span_seconds",
    "p95_cluster_span_seconds",
    "mean_member_medoid_gap_seconds",
    "p95_member_medoid_gap_seconds",
    "mean_member_medoid_visual_similarity",
    "representative_pair_count",
    "cross_gap_eligible_pair_count",
    "cross_gap_eligible_pair_fraction",
    "ungated_high_similarity_pair_count",
    "accepted_high_similarity_pair_count",
    "rejected_high_similarity_pair_count",
    "accepted_trigger_fraction",
    "left_triggered_cluster_count",
    "right_triggered_cluster_count",
    "mean_trigger_raw_cosine_similarity",
    "mean_trigger_medoid_gap_seconds",
    "p95_trigger_medoid_gap_seconds",
    "max_trigger_medoid_gap_seconds",
    "mean_trigger_center_gap_seconds",
    "p95_trigger_center_gap_seconds",
    "max_trigger_center_gap_seconds",
    "mean_trigger_interval_gap_seconds",
    "p95_trigger_interval_gap_seconds",
    "max_trigger_interval_gap_seconds",
    "trigger_center_gap_gt_unit_fraction",
    "trigger_center_gap_gt_quarter_duration_fraction",
    "trigger_interval_gap_gt_unit_fraction",
    "trigger_interval_gap_gt_quarter_duration_fraction",
    "left_marked_frame_count",
    "right_marked_frame_count",
    "left_restored_frame_count",
    "right_restored_frame_count",
    "left_removed_percent",
    "right_removed_percent",
    "mean_removed_percent",
    "left_keep_segment_count",
    "right_keep_segment_count",
    "mean_keep_segment_count",
    "no_removal",
    "passed",
    "diagnostics_path",
)

AGGREGATE_FIELDS = (
    "duration_seconds",
    "seconds_per_cluster",
    "k",
    "variant_kind",
    "within_time_weight",
    "temporal_unit_seconds",
    "high_similarity_threshold",
    "cross_gap_mode",
    "max_cross_gap_seconds",
    "pair_count",
    "pass_rate",
    "no_removal_rate",
    "mean_cluster_span_seconds",
    "mean_p95_cluster_span_seconds",
    "mean_member_medoid_gap_seconds",
    "mean_p95_member_medoid_gap_seconds",
    "mean_member_medoid_visual_similarity",
    "mean_cross_gap_eligible_pair_fraction",
    "mean_ungated_high_similarity_pair_count",
    "mean_accepted_high_similarity_pair_count",
    "mean_rejected_high_similarity_pair_count",
    "mean_accepted_trigger_fraction",
    "mean_left_triggered_cluster_count",
    "mean_right_triggered_cluster_count",
    "mean_trigger_raw_cosine_similarity",
    "mean_trigger_medoid_gap_seconds",
    "mean_p95_trigger_medoid_gap_seconds",
    "mean_max_trigger_medoid_gap_seconds",
    "mean_trigger_center_gap_seconds",
    "mean_p95_trigger_center_gap_seconds",
    "mean_max_trigger_center_gap_seconds",
    "mean_trigger_interval_gap_seconds",
    "mean_p95_trigger_interval_gap_seconds",
    "mean_max_trigger_interval_gap_seconds",
    "mean_trigger_center_gap_gt_unit_fraction",
    "mean_trigger_center_gap_gt_quarter_duration_fraction",
    "mean_trigger_interval_gap_gt_unit_fraction",
    "mean_trigger_interval_gap_gt_quarter_duration_fraction",
    "mean_marked_frame_count",
    "mean_restored_frame_count",
    "mean_removed_percent",
    "mean_keep_segment_count",
    "member_gap_reduction_vs_production_seconds",
    "visual_similarity_delta_vs_production",
    "trigger_count_delta_vs_production",
    "removed_percent_delta_vs_production",
    "pass_rate_delta_vs_production",
    "trigger_count_delta_vs_temporal_ungated",
    "center_gap_reduction_vs_temporal_ungated_seconds",
    "interval_gap_reduction_vs_temporal_ungated_seconds",
    "center_far_fraction_reduction_vs_temporal_ungated",
    "interval_far_fraction_reduction_vs_temporal_ungated",
    "removed_percent_delta_vs_temporal_ungated",
    "pass_rate_delta_vs_temporal_ungated",
)


def parse_gap_modes(value: str | Sequence[str]) -> list[str]:
    raw = value.split(",") if isinstance(value, str) else value
    output: list[str] = []
    for item in raw:
        mode = str(item).strip().lower()
        if not mode:
            continue
        if mode not in {"center", "interval"}:
            raise ValueError("cross gap modes must be center and/or interval")
        if mode not in output:
            output.append(mode)
    if not output:
        raise ValueError("at least one cross gap mode is required")
    return output


def build_cross_user_gap_variants(
    durations_seconds: Sequence[float],
    seconds_per_cluster_values: Sequence[float],
    similarity_thresholds: Sequence[float],
    cross_gap_modes: Sequence[str],
    cross_gap_seconds: Sequence[float],
    *,
    within_time_weight: float,
) -> list[dict[str, Any]]:
    """Build one production control, one temporal-only control, and gated arms."""

    if not math.isfinite(within_time_weight) or within_time_weight <= 0:
        raise ValueError("within time weight must be finite and positive")
    modes = parse_gap_modes(cross_gap_modes)
    gaps = parse_float_grid(
        cross_gap_seconds,
        name="cross gap seconds",
        minimum=0.0,
    )
    variants: list[dict[str, Any]] = []
    for duration in durations_seconds:
        for seconds_per_cluster in seconds_per_cluster_values:
            k = max(1, int(math.ceil(float(duration) / float(seconds_per_cluster))))
            for threshold in similarity_thresholds:
                shared = {
                    "duration_seconds": float(duration),
                    "seconds_per_cluster": float(seconds_per_cluster),
                    "k": k,
                    "high_similarity_threshold": float(threshold),
                }
                variants.append(
                    {
                        **shared,
                        "variant_kind": "production_baseline",
                        "within_time_weight": 0.0,
                        "cross_gap_mode": "none",
                        "max_cross_gap_seconds": None,
                    }
                )
                variants.append(
                    {
                        **shared,
                        "variant_kind": "temporal_ungated",
                        "within_time_weight": float(within_time_weight),
                        "cross_gap_mode": "none",
                        "max_cross_gap_seconds": None,
                    }
                )
                for mode in modes:
                    for gap in gaps:
                        variants.append(
                            {
                                **shared,
                                "variant_kind": f"temporal_{mode}_gate",
                                "within_time_weight": float(within_time_weight),
                                "cross_gap_mode": mode,
                                "max_cross_gap_seconds": float(gap),
                            }
                        )
    return variants


def _mean(values: Sequence[float]) -> float | None:
    return round(float(statistics.fmean(values)), 6) if values else None


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(percentile) / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        value = ordered[lower]
    else:
        fraction = position - lower
        value = ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
    return round(float(value), 6)


def _fraction_over(values: Sequence[float], threshold: float) -> float | None:
    if not values:
        return None
    return round(sum(float(value) > threshold for value in values) / len(values), 6)


def _configuration_key(row: dict[str, Any]) -> tuple[Any, ...]:
    max_gap = row.get("max_cross_gap_seconds")
    return (
        float(row["duration_seconds"]),
        float(row["seconds_per_cluster"]),
        int(row["k"]),
        str(row["variant_kind"]),
        float(row["within_time_weight"]),
        str(row["cross_gap_mode"]),
        None if max_gap is None or max_gap == "" else float(max_gap),
        float(row["temporal_unit_seconds"]),
        float(row["high_similarity_threshold"]),
    )


def _metric_row(
    *,
    pair: dict[str, Any],
    variant: dict[str, Any],
    temporal_unit_seconds: float,
    sample_interval_seconds: float,
    left_quality: dict[str, Any],
    right_quality: dict[str, Any],
    pruning: dict[str, Any],
    diagnostics_path: str | None,
) -> dict[str, Any]:
    spans = left_quality["cluster_spans"] + right_quality["cluster_spans"]
    member_gaps = (
        left_quality["member_medoid_gaps"] + right_quality["member_medoid_gaps"]
    )
    visual = (
        left_quality["member_medoid_visual_similarities"]
        + right_quality["member_medoid_visual_similarities"]
    )
    triggers = pruning["high_similarity_representative_pairs"]
    cosine_values = [float(row["similarity"]) for row in triggers]
    medoid_gaps = [float(row["timestamp_difference_seconds"]) for row in triggers]
    center_gaps = [float(row["center_gap_seconds"]) for row in triggers]
    interval_gaps = [float(row["interval_gap_seconds"]) for row in triggers]
    total_pairs = int(pruning["representative_pair_count"])
    eligible_pairs = int(pruning["cross_gap_eligible_pair_count"])
    ungated_pairs = int(pruning["ungated_high_similarity_representative_pair_count"])
    accepted_pairs = int(pruning["high_similarity_representative_pair_count"])
    duration = float(variant["duration_seconds"])
    left_removed_percent = (
        100.0 * float(pruning["left_removed_duration_seconds"]) / duration
    )
    right_removed_percent = (
        100.0 * float(pruning["right_removed_duration_seconds"]) / duration
    )
    return {
        "pair_id": pair["pair_id"],
        "day": pair.get("day"),
        "time_token": pair.get("time_token"),
        "left_agent": pair.get("left_agent"),
        "right_agent": pair.get("right_agent"),
        "duration_seconds": duration,
        "sample_interval_seconds": float(sample_interval_seconds),
        "seconds_per_cluster": float(variant["seconds_per_cluster"]),
        "k": int(variant["k"]),
        "variant_kind": variant["variant_kind"],
        "within_time_weight": float(variant["within_time_weight"]),
        "temporal_unit_seconds": float(temporal_unit_seconds),
        "high_similarity_threshold": float(variant["high_similarity_threshold"]),
        "cross_gap_mode": variant["cross_gap_mode"],
        "max_cross_gap_seconds": variant["max_cross_gap_seconds"],
        "left_cluster_count": int(pruning["left_cluster_count"]),
        "right_cluster_count": int(pruning["right_cluster_count"]),
        "mean_cluster_span_seconds": _mean(spans),
        "p95_cluster_span_seconds": _percentile(spans, 95),
        "mean_member_medoid_gap_seconds": _mean(member_gaps),
        "p95_member_medoid_gap_seconds": _percentile(member_gaps, 95),
        "mean_member_medoid_visual_similarity": _mean(visual),
        "representative_pair_count": total_pairs,
        "cross_gap_eligible_pair_count": eligible_pairs,
        "cross_gap_eligible_pair_fraction": round(eligible_pairs / total_pairs, 6),
        "ungated_high_similarity_pair_count": ungated_pairs,
        "accepted_high_similarity_pair_count": accepted_pairs,
        "rejected_high_similarity_pair_count": int(
            pruning["cross_gap_rejected_high_similarity_pair_count"]
        ),
        "accepted_trigger_fraction": (
            round(accepted_pairs / ungated_pairs, 6) if ungated_pairs else None
        ),
        "left_triggered_cluster_count": int(pruning["left_marked_cluster_count"]),
        "right_triggered_cluster_count": int(pruning["right_marked_cluster_count"]),
        "mean_trigger_raw_cosine_similarity": _mean(cosine_values),
        "mean_trigger_medoid_gap_seconds": _mean(medoid_gaps),
        "p95_trigger_medoid_gap_seconds": _percentile(medoid_gaps, 95),
        "max_trigger_medoid_gap_seconds": (
            round(max(medoid_gaps), 6) if medoid_gaps else None
        ),
        "mean_trigger_center_gap_seconds": _mean(center_gaps),
        "p95_trigger_center_gap_seconds": _percentile(center_gaps, 95),
        "max_trigger_center_gap_seconds": (
            round(max(center_gaps), 6) if center_gaps else None
        ),
        "mean_trigger_interval_gap_seconds": _mean(interval_gaps),
        "p95_trigger_interval_gap_seconds": _percentile(interval_gaps, 95),
        "max_trigger_interval_gap_seconds": (
            round(max(interval_gaps), 6) if interval_gaps else None
        ),
        "trigger_center_gap_gt_unit_fraction": _fraction_over(
            center_gaps, temporal_unit_seconds
        ),
        "trigger_center_gap_gt_quarter_duration_fraction": _fraction_over(
            center_gaps, duration / 4.0
        ),
        "trigger_interval_gap_gt_unit_fraction": _fraction_over(
            interval_gaps, temporal_unit_seconds
        ),
        "trigger_interval_gap_gt_quarter_duration_fraction": _fraction_over(
            interval_gaps, duration / 4.0
        ),
        "left_marked_frame_count": len(pruning["left_marked_frame_indices"]),
        "right_marked_frame_count": len(pruning["right_marked_frame_indices"]),
        "left_restored_frame_count": len(pruning["left_restored_frame_indices"]),
        "right_restored_frame_count": len(pruning["right_restored_frame_indices"]),
        "left_removed_percent": round(left_removed_percent, 6),
        "right_removed_percent": round(right_removed_percent, 6),
        "mean_removed_percent": round(
            (left_removed_percent + right_removed_percent) / 2.0, 6
        ),
        "left_keep_segment_count": len(pruning["left_keep_intervals"]),
        "right_keep_segment_count": len(pruning["right_keep_intervals"]),
        "mean_keep_segment_count": round(
            (len(pruning["left_keep_intervals"]) + len(pruning["right_keep_intervals"]))
            / 2.0,
            6,
        ),
        "no_removal": float(pruning["removed_duration_seconds"]) <= 0,
        "passed": bool(pruning["passed"]),
        "diagnostics_path": diagnostics_path,
    }


def aggregate_metrics(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_configuration_key(row)].append(row)

    def average(selected: Sequence[dict[str, Any]], field: str) -> float | None:
        values = [float(row[field]) for row in selected if row.get(field) is not None]
        return _mean(values)

    variant_order = {
        "production_baseline": 0,
        "temporal_ungated": 1,
        "temporal_center_gate": 2,
        "temporal_interval_gate": 3,
    }

    def group_sort_key(item: tuple[tuple[Any, ...], list[dict[str, Any]]]) -> tuple[Any, ...]:
        key = item[0]
        return (
            key[0],
            key[1],
            key[2],
            variant_order.get(str(key[3]), 99),
            str(key[5]),
            -1.0 if key[6] is None else float(key[6]),
            key[8],
        )

    aggregates: list[dict[str, Any]] = []
    for key, selected in sorted(groups.items(), key=group_sort_key):
        (
            duration,
            seconds_per_cluster,
            k,
            variant_kind,
            within_weight,
            gap_mode,
            max_gap,
            unit,
            threshold,
        ) = key
        aggregates.append(
            {
                "duration_seconds": duration,
                "seconds_per_cluster": seconds_per_cluster,
                "k": k,
                "variant_kind": variant_kind,
                "within_time_weight": within_weight,
                "temporal_unit_seconds": unit,
                "high_similarity_threshold": threshold,
                "cross_gap_mode": gap_mode,
                "max_cross_gap_seconds": max_gap,
                "pair_count": len(selected),
                "pass_rate": round(
                    sum(bool(row["passed"]) for row in selected) / len(selected), 6
                ),
                "no_removal_rate": round(
                    sum(bool(row["no_removal"]) for row in selected) / len(selected), 6
                ),
                "mean_cluster_span_seconds": average(
                    selected, "mean_cluster_span_seconds"
                ),
                "mean_p95_cluster_span_seconds": average(
                    selected, "p95_cluster_span_seconds"
                ),
                "mean_member_medoid_gap_seconds": average(
                    selected, "mean_member_medoid_gap_seconds"
                ),
                "mean_p95_member_medoid_gap_seconds": average(
                    selected, "p95_member_medoid_gap_seconds"
                ),
                "mean_member_medoid_visual_similarity": average(
                    selected, "mean_member_medoid_visual_similarity"
                ),
                "mean_cross_gap_eligible_pair_fraction": average(
                    selected, "cross_gap_eligible_pair_fraction"
                ),
                "mean_ungated_high_similarity_pair_count": average(
                    selected, "ungated_high_similarity_pair_count"
                ),
                "mean_accepted_high_similarity_pair_count": average(
                    selected, "accepted_high_similarity_pair_count"
                ),
                "mean_rejected_high_similarity_pair_count": average(
                    selected, "rejected_high_similarity_pair_count"
                ),
                "mean_accepted_trigger_fraction": average(
                    selected, "accepted_trigger_fraction"
                ),
                "mean_left_triggered_cluster_count": average(
                    selected, "left_triggered_cluster_count"
                ),
                "mean_right_triggered_cluster_count": average(
                    selected, "right_triggered_cluster_count"
                ),
                "mean_trigger_raw_cosine_similarity": average(
                    selected, "mean_trigger_raw_cosine_similarity"
                ),
                "mean_trigger_medoid_gap_seconds": average(
                    selected, "mean_trigger_medoid_gap_seconds"
                ),
                "mean_p95_trigger_medoid_gap_seconds": average(
                    selected, "p95_trigger_medoid_gap_seconds"
                ),
                "mean_max_trigger_medoid_gap_seconds": average(
                    selected, "max_trigger_medoid_gap_seconds"
                ),
                "mean_trigger_center_gap_seconds": average(
                    selected, "mean_trigger_center_gap_seconds"
                ),
                "mean_p95_trigger_center_gap_seconds": average(
                    selected, "p95_trigger_center_gap_seconds"
                ),
                "mean_max_trigger_center_gap_seconds": average(
                    selected, "max_trigger_center_gap_seconds"
                ),
                "mean_trigger_interval_gap_seconds": average(
                    selected, "mean_trigger_interval_gap_seconds"
                ),
                "mean_p95_trigger_interval_gap_seconds": average(
                    selected, "p95_trigger_interval_gap_seconds"
                ),
                "mean_max_trigger_interval_gap_seconds": average(
                    selected, "max_trigger_interval_gap_seconds"
                ),
                "mean_trigger_center_gap_gt_unit_fraction": average(
                    selected, "trigger_center_gap_gt_unit_fraction"
                ),
                "mean_trigger_center_gap_gt_quarter_duration_fraction": average(
                    selected, "trigger_center_gap_gt_quarter_duration_fraction"
                ),
                "mean_trigger_interval_gap_gt_unit_fraction": average(
                    selected, "trigger_interval_gap_gt_unit_fraction"
                ),
                "mean_trigger_interval_gap_gt_quarter_duration_fraction": average(
                    selected, "trigger_interval_gap_gt_quarter_duration_fraction"
                ),
                "mean_marked_frame_count": _mean(
                    [
                        (
                            float(row["left_marked_frame_count"])
                            + float(row["right_marked_frame_count"])
                        )
                        / 2.0
                        for row in selected
                    ]
                ),
                "mean_restored_frame_count": _mean(
                    [
                        (
                            float(row["left_restored_frame_count"])
                            + float(row["right_restored_frame_count"])
                        )
                        / 2.0
                        for row in selected
                    ]
                ),
                "mean_removed_percent": average(selected, "mean_removed_percent"),
                "mean_keep_segment_count": average(
                    selected, "mean_keep_segment_count"
                ),
            }
        )

    baseline_key_fields = (
        "duration_seconds",
        "seconds_per_cluster",
        "k",
        "temporal_unit_seconds",
        "high_similarity_threshold",
    )

    def shared_key(row: dict[str, Any]) -> tuple[Any, ...]:
        return tuple(row[field] for field in baseline_key_fields)

    production = {
        shared_key(row): row
        for row in aggregates
        if row["variant_kind"] == "production_baseline"
    }
    temporal_ungated = {
        shared_key(row): row
        for row in aggregates
        if row["variant_kind"] == "temporal_ungated"
    }

    def delta(
        row: dict[str, Any],
        baseline: dict[str, Any] | None,
        field: str,
        *,
        reduction: bool = False,
    ) -> float | None:
        if baseline is None or row.get(field) is None or baseline.get(field) is None:
            return None
        current = float(row[field])
        control = float(baseline[field])
        return round(control - current if reduction else current - control, 6)

    for row in aggregates:
        production_row = production.get(shared_key(row))
        ungated_row = temporal_ungated.get(shared_key(row))
        row["member_gap_reduction_vs_production_seconds"] = delta(
            row,
            production_row,
            "mean_member_medoid_gap_seconds",
            reduction=True,
        )
        row["visual_similarity_delta_vs_production"] = delta(
            row, production_row, "mean_member_medoid_visual_similarity"
        )
        row["trigger_count_delta_vs_production"] = delta(
            row, production_row, "mean_accepted_high_similarity_pair_count"
        )
        row["removed_percent_delta_vs_production"] = delta(
            row, production_row, "mean_removed_percent"
        )
        row["pass_rate_delta_vs_production"] = delta(
            row, production_row, "pass_rate"
        )
        row["trigger_count_delta_vs_temporal_ungated"] = delta(
            row, ungated_row, "mean_accepted_high_similarity_pair_count"
        )
        row["center_gap_reduction_vs_temporal_ungated_seconds"] = delta(
            row,
            ungated_row,
            "mean_trigger_center_gap_seconds",
            reduction=True,
        )
        row["interval_gap_reduction_vs_temporal_ungated_seconds"] = delta(
            row,
            ungated_row,
            "mean_trigger_interval_gap_seconds",
            reduction=True,
        )
        row["center_far_fraction_reduction_vs_temporal_ungated"] = delta(
            row,
            ungated_row,
            "mean_trigger_center_gap_gt_unit_fraction",
            reduction=True,
        )
        row["interval_far_fraction_reduction_vs_temporal_ungated"] = delta(
            row,
            ungated_row,
            "mean_trigger_interval_gap_gt_unit_fraction",
            reduction=True,
        )
        row["removed_percent_delta_vs_temporal_ungated"] = delta(
            row, ungated_row, "mean_removed_percent"
        )
        row["pass_rate_delta_vs_temporal_ungated"] = delta(
            row, ungated_row, "pass_rate"
        )
    return aggregates


def write_summary_html(output_dir: Path, aggregates: Sequence[dict[str, Any]]) -> Path:
    fields = (
        "duration_seconds",
        "variant_kind",
        "cross_gap_mode",
        "max_cross_gap_seconds",
        "pair_count",
        "mean_accepted_trigger_fraction",
        "mean_trigger_center_gap_seconds",
        "mean_p95_trigger_center_gap_seconds",
        "mean_trigger_interval_gap_seconds",
        "mean_p95_trigger_interval_gap_seconds",
        "mean_removed_percent",
        "pass_rate",
        "no_removal_rate",
        "mean_member_medoid_gap_seconds",
        "mean_member_medoid_visual_similarity",
    )
    lines = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Cross-user temporal gate grid</title>",
        "<style>body{font-family:system-ui;margin:24px}table{border-collapse:collapse}",
        "th,td{border:1px solid #ccc;padding:6px 8px;text-align:right}",
        "th:first-child,td:first-child{text-align:left}</style></head><body>",
        "<h1>Cross-user hard temporal gate grid</h1><table><thead><tr>",
    ]
    lines.extend(f"<th>{html.escape(field)}</th>" for field in fields)
    lines.append("</tr></thead><tbody>")
    for row in aggregates:
        lines.append("<tr>")
        for field in fields:
            value = row.get(field)
            lines.append(f"<td>{html.escape('' if value is None else str(value))}</td>")
        lines.append("</tr>")
    lines.extend(["</tbody></table></body></html>"])
    path = output_dir / "summary.html"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_checkpoint(
    output_root: Path,
    *,
    cohort_rows: Sequence[dict[str, Any]],
    metric_rows: Sequence[dict[str, Any]],
    failures: Sequence[dict[str, Any]],
    pair_count_target: int,
    configuration_count_per_pair: int,
    source_experiment_dir: Path,
    resume_experiment_dir: Path | None,
    status: str,
) -> list[dict[str, Any]]:
    aggregates = aggregate_metrics(metric_rows)
    _write_jsonl_atomic(output_root / "cohort.jsonl", cohort_rows)
    _write_jsonl_atomic(output_root / "grid_metrics.jsonl", metric_rows)
    _write_jsonl_atomic(output_root / "aggregate_metrics.jsonl", aggregates)
    _write_csv_atomic(output_root / "grid_metrics.csv", metric_rows, METRIC_FIELDS)
    _write_csv_atomic(
        output_root / "aggregate_metrics.csv", aggregates, AGGREGATE_FIELDS
    )
    write_summary_html(output_root, aggregates)
    progress = {
        "status": status,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pair_count_target": int(pair_count_target),
        "pair_count_completed": len(cohort_rows),
        "configuration_count_per_pair": int(configuration_count_per_pair),
        "expected_metric_count": int(pair_count_target)
        * int(configuration_count_per_pair),
        "metric_count_completed": len(metric_rows),
        "aggregate_count": len(aggregates),
        "last_completed_pair_id": (
            cohort_rows[-1]["pair_id"] if cohort_rows else None
        ),
        "source_experiment_dir": str(source_experiment_dir),
        "resume_experiment_dir": (
            str(resume_experiment_dir) if resume_experiment_dir else None
        ),
        "failure_count": len(failures),
        "failures": list(failures),
    }
    _write_json_atomic(output_root / "progress.json", progress)
    return aggregates


def _compact_pruning_diagnostics(pruning: dict[str, Any]) -> dict[str, Any]:
    compact = {
        key: value
        for key, value in pruning.items()
        if key
        not in {
            "representative_similarity_matrix",
            "high_similarity_representative_pairs",
        }
    }
    pairs = pruning["high_similarity_representative_pairs"]
    compact["high_similarity_representative_pairs_sample"] = pairs[:100]
    compact["high_similarity_representative_pairs_sample_truncated"] = len(pairs) > 100
    return compact


def run_cross_user_temporal_gate_grid(
    *,
    source_experiment_dir: str | Path,
    output_dir: str | Path,
    pair_count: int = DEFAULT_PAIR_COUNT,
    durations_seconds: Sequence[float] = DEFAULT_DURATIONS_SECONDS,
    seconds_per_cluster_values: Sequence[float] = DEFAULT_SECONDS_PER_CLUSTER,
    similarity_thresholds: Sequence[float] = DEFAULT_SIMILARITY_THRESHOLDS,
    within_time_weight: float = DEFAULT_WITHIN_TIME_WEIGHT,
    cross_gap_modes: Sequence[str] = DEFAULT_CROSS_GAP_MODES,
    cross_gap_seconds: Sequence[float] = DEFAULT_CROSS_GAP_SECONDS,
    temporal_unit_seconds: float = DEFAULT_TEMPORAL_UNIT_SECONDS,
    sample_interval_seconds: float = 1.0,
    max_iterations: int = 25,
    min_pruned_video_seconds: float = 8.0,
    pruning_protection_mode: str = "min_percent",
    min_pruned_video_percent: float | None = 20.0,
    trace_pair_limit: int = 3,
    resume_experiment_dir: str | Path | None = None,
) -> dict[str, Any]:
    if pair_count <= 0:
        raise ValueError("pair count must be positive")
    if sample_interval_seconds <= 0 or temporal_unit_seconds <= 0:
        raise ValueError("sample interval and temporal unit must be positive")
    if max_iterations <= 0 or trace_pair_limit < 0:
        raise ValueError("max iterations must be positive and trace limit non-negative")
    durations = sorted(
        parse_float_grid(durations_seconds, name="duration", strictly_positive=True)
    )
    cluster_densities = parse_float_grid(
        seconds_per_cluster_values,
        name="seconds per cluster",
        strictly_positive=True,
    )
    thresholds = parse_float_grid(
        similarity_thresholds,
        name="similarity threshold",
        minimum=-1.0,
        maximum=1.0,
    )
    modes = parse_gap_modes(cross_gap_modes)
    gaps = parse_float_grid(
        cross_gap_seconds, name="cross gap seconds", minimum=0.0
    )
    variants = build_cross_user_gap_variants(
        durations,
        cluster_densities,
        thresholds,
        modes,
        gaps,
        within_time_weight=within_time_weight,
    )

    source_root = Path(source_experiment_dir)
    output_root = Path(output_dir)
    resume_root = Path(resume_experiment_dir) if resume_experiment_dir else None
    if not source_root.is_dir():
        raise FileNotFoundError(f"source experiment directory does not exist: {source_root}")
    source_cohort_path = source_root / "cohort.jsonl"
    source_summary_path = source_root / "summary.json"
    if not source_cohort_path.is_file() or not source_summary_path.is_file():
        raise FileNotFoundError(
            "source experiment must contain cohort.jsonl and summary.json"
        )
    source_summary = read_json(source_summary_path)
    model_id = str(source_summary.get("model_id") or DEFAULT_CLIP_MODEL)
    source_cohort = list(iter_jsonl(source_cohort_path))
    if len(source_cohort) < pair_count:
        raise RuntimeError(
            f"source cohort has {len(source_cohort)} pairs; requested {pair_count}"
        )
    selected_source_cohort = source_cohort[:pair_count]
    expected_frames = _expected_frame_count(
        max(durations), sample_interval_seconds
    )
    for pair in selected_source_cohort:
        pair_dir = source_root / "pairs" / str(pair["pair_id"])
        if not (pair_dir / "embedding_cache.json").is_file() or not (
            pair_dir / "embedding_cache.npz"
        ).is_file():
            raise FileNotFoundError(f"embedding cache is incomplete: {pair_dir}")

    output_root.mkdir(parents=True, exist_ok=True)
    cohort_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    completed_pair_ids: set[str] = set()
    expected_configuration_keys = {
        (
            float(variant["duration_seconds"]),
            float(variant["seconds_per_cluster"]),
            int(variant["k"]),
            str(variant["variant_kind"]),
            float(variant["within_time_weight"]),
            str(variant["cross_gap_mode"]),
            variant["max_cross_gap_seconds"],
            float(temporal_unit_seconds),
            float(variant["high_similarity_threshold"]),
        )
        for variant in variants
    }
    if resume_root is not None:
        resume_cohort_path = resume_root / "cohort.jsonl"
        resume_metrics_path = resume_root / "grid_metrics.jsonl"
        if not resume_cohort_path.is_file() or not resume_metrics_path.is_file():
            raise FileNotFoundError(
                "resume experiment must contain cohort.jsonl and grid_metrics.jsonl"
            )
        resume_cohort = list(iter_jsonl(resume_cohort_path))
        resume_metrics = list(iter_jsonl(resume_metrics_path))
        metrics_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in resume_metrics:
            metrics_by_pair[str(row.get("pair_id"))].append(row)
        source_pair_ids = {str(row["pair_id"]) for row in selected_source_cohort}
        for cached_pair in resume_cohort:
            pair_id = str(cached_pair.get("pair_id") or "")
            selected_metrics = metrics_by_pair.get(pair_id, [])
            if pair_id not in source_pair_ids or len(selected_metrics) != len(variants):
                continue
            if {_configuration_key(row) for row in selected_metrics} != expected_configuration_keys:
                continue
            cohort_rows.append({**cached_pair, "metrics_reused": True})
            metric_rows.extend(selected_metrics)
            completed_pair_ids.add(pair_id)
            pair_dir = output_root / "pairs" / pair_id
            pair_dir.mkdir(parents=True, exist_ok=True)
            _write_json_atomic(
                pair_dir / "pair_complete.json",
                {
                    "pair_id": pair_id,
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "configuration_count": len(selected_metrics),
                    "metrics_reused": True,
                    "resume_experiment_dir": str(resume_root),
                },
            )
        print(
            f"resume_checkpoint_pairs={len(completed_pair_ids)} "
            f"resume_checkpoint_metrics={len(metric_rows)}",
            flush=True,
        )

    _write_checkpoint(
        output_root,
        cohort_rows=cohort_rows,
        metric_rows=metric_rows,
        failures=failures,
        pair_count_target=pair_count,
        configuration_count_per_pair=len(variants),
        source_experiment_dir=source_root,
        resume_experiment_dir=resume_root,
        status="running",
    )

    for source_pair in selected_source_cohort:
        pair_id = str(source_pair["pair_id"])
        if pair_id in completed_pair_ids:
            continue
        try:
            source_pair_dir = source_root / "pairs" / pair_id
            frames_by_side, embeddings_by_side, _ = _load_pair_embedding_cache(
                source_pair_dir,
                expected_model_id=model_id,
                expected_frame_count=expected_frames,
            )
            pair = {
                **source_pair,
                "pair_id": pair_id,
                "source_embedding_cache_dir": str(source_pair_dir),
                "embedding_cache_reused": True,
                "metrics_reused": False,
            }
            pair_metric_rows: list[dict[str, Any]] = []
            duration_cache: dict[float, dict[str, Any]] = {}
            cluster_cache: dict[
                tuple[float, float, float],
                tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]],
            ] = {}

            for variant in variants:
                duration = float(variant["duration_seconds"])
                if duration not in duration_cache:
                    prefix_frames = []
                    prefix_embeddings = []
                    for frames, embeddings in zip(frames_by_side, embeddings_by_side):
                        current_frames, current_embeddings = _duration_prefix(
                            frames,
                            embeddings,
                            duration,
                            sample_interval_seconds,
                        )
                        prefix_frames.append(current_frames)
                        prefix_embeddings.append(current_embeddings)
                    full_matrix = _cosine_matrix(
                        prefix_embeddings[0], prefix_embeddings[1]
                    )
                    duration_cache[duration] = {
                        "frames": prefix_frames,
                        "embeddings": prefix_embeddings,
                        "full_matrix": full_matrix,
                        "left_best_matches": _side_best_frame_matches(
                            full_matrix,
                            side="left",
                            left_frames=prefix_frames[0],
                            right_frames=prefix_frames[1],
                            max_pair_time_difference_seconds=None,
                        ),
                        "right_best_matches": _side_best_frame_matches(
                            full_matrix,
                            side="right",
                            left_frames=prefix_frames[0],
                            right_frames=prefix_frames[1],
                            max_pair_time_difference_seconds=None,
                        ),
                    }
                context = duration_cache[duration]
                prefix_frames = context["frames"]
                prefix_embeddings = context["embeddings"]
                cluster_key = (
                    duration,
                    float(variant["seconds_per_cluster"]),
                    float(variant["within_time_weight"]),
                )
                if cluster_key not in cluster_cache:
                    left_clusters = time_aware_clustered_frame_representatives(
                        prefix_frames[0],
                        prefix_embeddings[0],
                        cluster_count=int(variant["k"]),
                        time_weight=float(variant["within_time_weight"]),
                        temporal_unit_seconds=temporal_unit_seconds,
                        max_iterations=max_iterations,
                    )
                    right_clusters = time_aware_clustered_frame_representatives(
                        prefix_frames[1],
                        prefix_embeddings[1],
                        cluster_count=int(variant["k"]),
                        time_weight=float(variant["within_time_weight"]),
                        temporal_unit_seconds=temporal_unit_seconds,
                        max_iterations=max_iterations,
                    )
                    cluster_cache[cluster_key] = (
                        left_clusters,
                        right_clusters,
                        cluster_quality_metrics(left_clusters, prefix_embeddings[0]),
                        cluster_quality_metrics(right_clusters, prefix_embeddings[1]),
                    )
                left_clusters, right_clusters, left_quality, right_quality = cluster_cache[
                    cluster_key
                ]
                representative_matrix_key = ("representative_matrix", cluster_key)
                if representative_matrix_key not in context:
                    context[representative_matrix_key] = _cosine_matrix(
                        left_clusters["representative_embeddings"],
                        right_clusters["representative_embeddings"],
                    )
                representative_matrix = context[representative_matrix_key]
                temporal_gap_matrix_key = ("temporal_gap_matrices", cluster_key)
                if temporal_gap_matrix_key not in context:
                    context[temporal_gap_matrix_key] = (
                        cross_cluster_temporal_gap_matrices(
                            left_clusters["representatives"],
                            right_clusters["representatives"],
                        )
                    )
                temporal_gap_matrices = context[temporal_gap_matrix_key]
                pruning = prune_time_aware_cluster_pair(
                    prefix_frames[0],
                    prefix_frames[1],
                    prefix_embeddings[0],
                    prefix_embeddings[1],
                    left_clusters,
                    right_clusters,
                    full_frame_matrix=context["full_matrix"],
                    start_seconds=0.0,
                    duration_seconds=duration,
                    sample_interval_seconds=sample_interval_seconds,
                    high_similarity_threshold=float(
                        variant["high_similarity_threshold"]
                    ),
                    min_pruned_video_seconds=min_pruned_video_seconds,
                    pruning_protection_mode=pruning_protection_mode,
                    min_pruned_video_percent=min_pruned_video_percent,
                    cross_gap_mode=str(variant["cross_gap_mode"]),
                    max_cross_gap_seconds=variant["max_cross_gap_seconds"],
                    representative_similarity_matrix=representative_matrix,
                    representative_temporal_gap_matrices=temporal_gap_matrices,
                    left_best_frame_matches=context["left_best_matches"],
                    right_best_frame_matches=context["right_best_matches"],
                )
                diagnostics_path: Path | None = None
                if len(cohort_rows) < trace_pair_limit:
                    gap_slug = (
                        "inf"
                        if variant["max_cross_gap_seconds"] is None
                        else _number_slug(float(variant["max_cross_gap_seconds"]))
                    )
                    diagnostics_path = (
                        output_root
                        / "pairs"
                        / pair_id
                        / "diagnostics"
                        / f"duration_{_number_slug(duration)}s"
                        / str(variant["variant_kind"])
                        / f"mode_{variant['cross_gap_mode']}"
                        / f"gap_{gap_slug}s.json"
                    )
                    write_json(
                        diagnostics_path,
                        {
                            "pair": pair,
                            "configuration": {
                                **variant,
                                "temporal_unit_seconds": temporal_unit_seconds,
                                "within_cluster_distance_formula": CLUSTER_DISTANCE_FORMULA,
                                "cross_gate_rule": (
                                    "cosine >= threshold AND selected gap <= max gap"
                                    if variant["cross_gap_mode"] != "none"
                                    else "cosine >= threshold"
                                ),
                            },
                            "left_cluster_quality": left_quality,
                            "right_cluster_quality": right_quality,
                            "pruning": _compact_pruning_diagnostics(pruning),
                        },
                    )
                pair_metric_rows.append(
                    _metric_row(
                        pair=pair,
                        variant=variant,
                        temporal_unit_seconds=temporal_unit_seconds,
                        sample_interval_seconds=sample_interval_seconds,
                        left_quality=left_quality,
                        right_quality=right_quality,
                        pruning=pruning,
                        diagnostics_path=(
                            str(diagnostics_path) if diagnostics_path else None
                        ),
                    )
                )

            if len(pair_metric_rows) != len(variants):
                raise RuntimeError(
                    f"incomplete pair grid: pair={pair_id} expected={len(variants)} "
                    f"actual={len(pair_metric_rows)}"
                )
            metric_rows.extend(pair_metric_rows)
            cohort_rows.append(pair)
            pair_output_dir = output_root / "pairs" / pair_id
            pair_output_dir.mkdir(parents=True, exist_ok=True)
            _write_json_atomic(
                pair_output_dir / "pair_complete.json",
                {
                    "pair_id": pair_id,
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "configuration_count": len(pair_metric_rows),
                    "source_embedding_cache_dir": str(source_pair_dir),
                    "metrics_reused": False,
                },
            )
            _write_checkpoint(
                output_root,
                cohort_rows=cohort_rows,
                metric_rows=metric_rows,
                failures=failures,
                pair_count_target=pair_count,
                configuration_count_per_pair=len(variants),
                source_experiment_dir=source_root,
                resume_experiment_dir=resume_root,
                status="running",
            )
            print(
                f"progress_pairs={len(cohort_rows)}/{pair_count} pair_id={pair_id} "
                f"metrics={len(metric_rows)}",
                flush=True,
            )
        except Exception as exc:
            failures.append(
                {
                    "pair_id": pair_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            _write_checkpoint(
                output_root,
                cohort_rows=cohort_rows,
                metric_rows=metric_rows,
                failures=failures,
                pair_count_target=pair_count,
                configuration_count_per_pair=len(variants),
                source_experiment_dir=source_root,
                resume_experiment_dir=resume_root,
                status="running",
            )
            print(
                f"progress_skip pair_id={pair_id} error={type(exc).__name__}: {exc}",
                flush=True,
            )

    target_met = len(cohort_rows) == pair_count
    aggregates = _write_checkpoint(
        output_root,
        cohort_rows=cohort_rows,
        metric_rows=metric_rows,
        failures=failures,
        pair_count_target=pair_count,
        configuration_count_per_pair=len(variants),
        source_experiment_dir=source_root,
        resume_experiment_dir=resume_root,
        status="complete" if target_met else "incomplete",
    )
    summary = {
        "source_experiment_dir": str(source_root),
        "output_dir": str(output_root),
        "summary_html": str(output_root / "summary.html"),
        "pair_count_requested": pair_count,
        "pair_count": len(cohort_rows),
        "target_met": target_met,
        "same_pair_cohort_for_all_variants": True,
        "model_id": model_id,
        "experiment_contract": (
            "production baseline uses w=0 and no cross gate; all gated arms use "
            "the fixed within-video time weight and preserve the cosine threshold "
            "and duration-protection policy"
        ),
        "settings": {
            "durations_seconds": durations,
            "seconds_per_cluster_values": cluster_densities,
            "similarity_thresholds": thresholds,
            "production_within_time_weight": 0.0,
            "gated_within_time_weight": float(within_time_weight),
            "cross_gap_modes": modes,
            "cross_gap_seconds": gaps,
            "temporal_unit_seconds": temporal_unit_seconds,
            "sample_interval_seconds": sample_interval_seconds,
            "max_iterations": max_iterations,
            "pruning_protection_mode": pruning_protection_mode,
            "min_pruned_video_seconds": min_pruned_video_seconds,
            "min_pruned_video_percent": min_pruned_video_percent,
            "configuration_count_per_pair": len(variants),
            "trace_pair_limit": trace_pair_limit,
            "resume_experiment_dir": str(resume_root) if resume_root else None,
        },
        "expected_metric_count": pair_count * len(variants),
        "metric_count": len(metric_rows),
        "aggregate_count": len(aggregates),
        "failure_count": len(failures),
        "failures": failures,
        "progress_path": str(output_root / "progress.json"),
    }
    _write_json_atomic(output_root / "summary.json", summary)
    if not target_met:
        raise RuntimeError(
            f"requested {pair_count} cached pairs but completed {len(cohort_rows)}; "
            f"inspect {output_root / 'summary.json'}"
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare production pruning with center- and interval-gap hard gates "
            "on a fixed cached cohort"
        )
    )
    parser.add_argument("--source-experiment-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pair-count", type=int, default=DEFAULT_PAIR_COUNT)
    parser.add_argument(
        "--durations-seconds",
        default=",".join(f"{value:g}" for value in DEFAULT_DURATIONS_SECONDS),
    )
    parser.add_argument(
        "--seconds-per-cluster-values",
        default=",".join(f"{value:g}" for value in DEFAULT_SECONDS_PER_CLUSTER),
    )
    parser.add_argument(
        "--similarity-thresholds",
        default=",".join(f"{value:g}" for value in DEFAULT_SIMILARITY_THRESHOLDS),
    )
    parser.add_argument(
        "--within-time-weight", type=float, default=DEFAULT_WITHIN_TIME_WEIGHT
    )
    parser.add_argument(
        "--cross-gap-modes", default=",".join(DEFAULT_CROSS_GAP_MODES)
    )
    parser.add_argument(
        "--cross-gap-seconds",
        default=",".join(f"{value:g}" for value in DEFAULT_CROSS_GAP_SECONDS),
    )
    parser.add_argument(
        "--temporal-unit-seconds", type=float, default=DEFAULT_TEMPORAL_UNIT_SECONDS
    )
    parser.add_argument("--sample-interval-seconds", type=float, default=1.0)
    parser.add_argument("--max-iterations", type=int, default=25)
    parser.add_argument("--min-pruned-video-seconds", type=float, default=8.0)
    parser.add_argument(
        "--pruning-protection-mode",
        choices=["reject", "min_seconds", "min_percent"],
        default="min_percent",
    )
    parser.add_argument("--min-pruned-video-percent", type=float, default=20.0)
    parser.add_argument("--trace-pair-limit", type=int, default=3)
    parser.add_argument(
        "--resume-experiment-dir",
        help="Reuse complete per-pair metrics from an earlier gate experiment",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_cross_user_temporal_gate_grid(
        source_experiment_dir=args.source_experiment_dir,
        output_dir=args.output_dir,
        pair_count=args.pair_count,
        durations_seconds=parse_float_grid(
            args.durations_seconds, name="duration", strictly_positive=True
        ),
        seconds_per_cluster_values=parse_float_grid(
            args.seconds_per_cluster_values,
            name="seconds per cluster",
            strictly_positive=True,
        ),
        similarity_thresholds=parse_float_grid(
            args.similarity_thresholds,
            name="similarity threshold",
            minimum=-1.0,
            maximum=1.0,
        ),
        within_time_weight=args.within_time_weight,
        cross_gap_modes=parse_gap_modes(args.cross_gap_modes),
        cross_gap_seconds=parse_float_grid(
            args.cross_gap_seconds, name="cross gap seconds", minimum=0.0
        ),
        temporal_unit_seconds=args.temporal_unit_seconds,
        sample_interval_seconds=args.sample_interval_seconds,
        max_iterations=args.max_iterations,
        min_pruned_video_seconds=args.min_pruned_video_seconds,
        pruning_protection_mode=args.pruning_protection_mode,
        min_pruned_video_percent=args.min_pruned_video_percent,
        trace_pair_limit=args.trace_pair_limit,
        resume_experiment_dir=args.resume_experiment_dir,
    )
    print(
        f"wrote {summary['metric_count']} rows for {summary['pair_count']} fixed pairs "
        f"to {summary['output_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
