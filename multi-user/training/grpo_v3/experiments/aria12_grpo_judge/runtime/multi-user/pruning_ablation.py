"""Controlled 30-second ablations for paired CLIP-guided video pruning.

The runner selects one fixed synchronized pair cohort, samples every video once
at the densest requested frame rate, and materializes four independent sweeps:
temporal matching policy, CLIP threshold, sampling rate, and cosine K-means K.
Only the named factor changes inside a sweep; all other values stay at the
declared baseline. No QA generation or VLM judge is invoked.
"""

from __future__ import annotations

import html
import math
import random
import statistics
from pathlib import Path
from typing import Any, Iterable, Sequence

from .clip_gap_demo import DEFAULT_CLIP_MODEL, ImageEncoder, TransformersClipEncoder
from .evidence import group_manifest_clips
from .group_relative_clip_sampling import (
    clustered_temporal_similarity_pruning,
    group_clip_frames,
)
from .io_utils import read_json, stable_id, write_json, write_jsonl
from .pruning_k_grid import (
    _materialize,
    _sample_pair,
    _safe_part,
    _tabular_trace_rows,
    _write_csv,
    build_cluster_trace,
    materialize_centroid_frames,
    parse_k_values,
)


DEFAULT_FPS_VALUES = (0.5, 1.0, 2.0, 4.0)
DEFAULT_THRESHOLD_VALUES = (0.78, 0.80, 0.82, 0.84, 0.86, 0.88)
DEFAULT_TEMPORAL_POLICIES = (
    "current",
    "gate_1s",
    "gate_2s",
    "gate_5s",
    "gate_2s_mnn",
    "gate_2s_mnn_contiguous",
)
DEFAULT_RANDOM_SEED = 20260720

TEMPORAL_POLICY_SETTINGS: dict[str, dict[str, Any]] = {
    "current": {
        "max_pair_time_difference_seconds": None,
        "mutual_nearest_only": False,
        "split_noncontiguous_clusters": False,
    },
    "gate_1s": {
        "max_pair_time_difference_seconds": 1.0,
        "mutual_nearest_only": False,
        "split_noncontiguous_clusters": False,
    },
    "gate_2s": {
        "max_pair_time_difference_seconds": 2.0,
        "mutual_nearest_only": False,
        "split_noncontiguous_clusters": False,
    },
    "gate_5s": {
        "max_pair_time_difference_seconds": 5.0,
        "mutual_nearest_only": False,
        "split_noncontiguous_clusters": False,
    },
    "gate_2s_mnn": {
        "max_pair_time_difference_seconds": 2.0,
        "mutual_nearest_only": True,
        "split_noncontiguous_clusters": False,
    },
    "gate_2s_mnn_contiguous": {
        "max_pair_time_difference_seconds": 2.0,
        "mutual_nearest_only": True,
        "split_noncontiguous_clusters": True,
    },
}

METRIC_FIELDS = (
    "pair_id",
    "sweep",
    "variant_id",
    "variant_label",
    "fps",
    "sample_interval_seconds",
    "k",
    "high_similarity_threshold",
    "temporal_policy",
    "max_pair_time_difference_seconds",
    "mutual_nearest_only",
    "split_noncontiguous_clusters",
    "left_sampled_frame_count",
    "right_sampled_frame_count",
    "left_cluster_count",
    "right_cluster_count",
    "high_similarity_representative_pair_count",
    "trigger_pair_count_beyond_2s",
    "mean_trigger_time_difference_seconds",
    "median_trigger_time_difference_seconds",
    "max_trigger_time_difference_seconds",
    "left_multirun_cluster_count",
    "right_multirun_cluster_count",
    "left_mean_cluster_span_seconds",
    "right_mean_cluster_span_seconds",
    "left_marked_frame_count",
    "right_marked_frame_count",
    "left_restored_frame_count",
    "right_restored_frame_count",
    "left_kept_duration_seconds",
    "right_kept_duration_seconds",
    "left_removed_duration_seconds",
    "right_removed_duration_seconds",
    "removed_duration_seconds",
    "left_keep_segment_count",
    "right_keep_segment_count",
    "no_removal",
    "collapsed",
    "passed",
    "left_materialization_status",
    "right_materialization_status",
    "left_pruned_video",
    "right_pruned_video",
    "diagnostics_path",
    "cluster_trace_path",
)

AGGREGATE_FIELDS = (
    "sweep",
    "variant_id",
    "variant_label",
    "fps",
    "k",
    "high_similarity_threshold",
    "temporal_policy",
    "pair_count",
    "passed_count",
    "no_removal_count",
    "collapsed_count",
    "mean_high_similarity_representative_pair_count",
    "mean_trigger_pair_count_beyond_2s",
    "mean_left_kept_duration_seconds",
    "mean_right_kept_duration_seconds",
    "mean_removed_duration_seconds",
    "mean_left_keep_segment_count",
    "mean_right_keep_segment_count",
    "mean_left_multirun_cluster_count",
    "mean_right_multirun_cluster_count",
)

TRACE_PREFIX_FIELDS = (
    "sweep",
    "variant_id",
    "variant_label",
    "fps",
    "high_similarity_threshold",
    "temporal_policy",
)

ASSIGNMENT_FIELDS = TRACE_PREFIX_FIELDS + (
    "pair_id",
    "k",
    "side",
    "cluster_index",
    "visual_cluster_index",
    "temporal_component_index",
    "cluster_triggered_for_pruning",
    "medoid_frame_index",
    "medoid_timestamp_seconds",
    "member_frame_index",
    "member_timestamp_seconds",
    "member_is_medoid",
    "member_final_status",
    "member_path",
)

TRIGGER_FIELDS = TRACE_PREFIX_FIELDS + (
    "pair_id",
    "k",
    "left_cluster_index",
    "right_cluster_index",
    "similarity",
    "left_representative_frame_index",
    "right_representative_frame_index",
    "left_representative_timestamp_seconds",
    "right_representative_timestamp_seconds",
    "timestamp_difference_seconds",
)

CENTROID_FIELDS = TRACE_PREFIX_FIELDS + (
    "pair_id",
    "k",
    "side",
    "cluster_index",
    "cluster_triggered_for_pruning",
    "centroid_frame_index",
    "centroid_frame_timestamp_seconds",
    "centroid_frame_output",
    "source_sampled_frame_path",
    "selection_method",
)


def parse_float_values(
    value: str | Iterable[float],
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> list[float]:
    """Parse and de-duplicate a numeric sweep while preserving order."""

    raw_values: Iterable[Any]
    if isinstance(value, str):
        raw_values = [part.strip() for part in value.split(",") if part.strip()]
    else:
        raw_values = value
    parsed_values: list[float] = []
    for raw in raw_values:
        try:
            parsed = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid {name} value: {raw!r}") from exc
        if not math.isfinite(parsed):
            raise ValueError(f"{name} values must be finite")
        if minimum is not None and parsed < minimum:
            raise ValueError(f"{name} values must be at least {minimum}")
        if maximum is not None and parsed > maximum:
            raise ValueError(f"{name} values must be at most {maximum}")
        if parsed not in parsed_values:
            parsed_values.append(parsed)
    if not parsed_values:
        raise ValueError(f"at least one {name} value is required")
    return parsed_values


def parse_temporal_policies(value: str | Iterable[str]) -> list[str]:
    raw = [part.strip() for part in value.split(",")] if isinstance(value, str) else list(value)
    policies = []
    for policy in raw:
        if not policy:
            continue
        if policy not in TEMPORAL_POLICY_SETTINGS:
            choices = ", ".join(TEMPORAL_POLICY_SETTINGS)
            raise ValueError(f"unknown temporal policy {policy!r}; expected one of: {choices}")
        if policy not in policies:
            policies.append(policy)
    if not policies:
        raise ValueError("at least one temporal policy is required")
    return policies


def _number_slug(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def build_ablation_variants(
    *,
    baseline_fps: float,
    baseline_k: int,
    baseline_threshold: float,
    baseline_temporal_policy: str,
    fps_values: Sequence[float],
    k_values: Sequence[int],
    threshold_values: Sequence[float],
    temporal_policies: Sequence[str],
) -> list[dict[str, Any]]:
    """Return four one-factor-at-a-time sweeps, including a control in each."""

    variants = []

    def append(
        sweep: str,
        parameter_label: str,
        *,
        fps: float = baseline_fps,
        k: int = baseline_k,
        threshold: float = baseline_threshold,
        temporal_policy: str = baseline_temporal_policy,
    ) -> None:
        policy = TEMPORAL_POLICY_SETTINGS[temporal_policy]
        variants.append(
            {
                "sweep": sweep,
                "variant_id": f"{sweep}_{parameter_label}",
                "variant_label": parameter_label,
                "fps": float(fps),
                "sample_interval_seconds": 1.0 / float(fps),
                "k": int(k),
                "high_similarity_threshold": float(threshold),
                "temporal_policy": temporal_policy,
                **policy,
            }
        )

    for policy in temporal_policies:
        append("temporal", policy, temporal_policy=policy)
    for threshold in threshold_values:
        append("threshold", f"threshold_{_number_slug(threshold)}", threshold=threshold)
    for fps in fps_values:
        append("sampling", f"fps_{_number_slug(fps)}", fps=fps)
    for k in k_values:
        append("k", f"k_{int(k):02d}", k=k)
    return variants


def _derive_sampling_rate(
    frames: Sequence[dict[str, Any]],
    embeddings: Sequence[list[float]],
    *,
    fps: float,
    start_seconds: float,
    duration_seconds: float,
) -> tuple[list[dict[str, Any]], list[list[float]]]:
    """Select a lower-rate timeline from the shared dense master cache."""

    if len(frames) != len(embeddings):
        raise ValueError("dense frame and embedding counts must match")
    target_count = max(1, int(math.ceil(duration_seconds * fps)))
    selected_indices = []
    for target_index in range(target_count):
        target_timestamp = start_seconds + target_index / fps
        if target_timestamp >= start_seconds + duration_seconds - 1e-9:
            break
        closest_index = min(
            range(len(frames)),
            key=lambda index: (
                abs(float(frames[index].get("timestamp_seconds", 0.0)) - target_timestamp),
                index,
            ),
        )
        if closest_index not in selected_indices:
            selected_indices.append(closest_index)
    return (
        [dict(frames[index]) for index in selected_indices],
        [list(embeddings[index]) for index in selected_indices],
    )


def _cluster_temporal_metrics(
    decisions: Sequence[dict[str, Any]],
    *,
    sample_interval_seconds: float,
) -> tuple[int, float]:
    spans = []
    multirun_count = 0
    gap_limit = 1.5 * sample_interval_seconds
    for decision in decisions:
        timestamps = sorted(float(value) for value in decision.get("member_timestamps", []))
        if not timestamps:
            continue
        spans.append(timestamps[-1] - timestamps[0])
        run_count = 1 + sum(
            right - left > gap_limit + 1e-9
            for left, right in zip(timestamps, timestamps[1:])
        )
        multirun_count += run_count > 1
    return multirun_count, round(statistics.mean(spans), 6) if spans else 0.0


def _mean(rows: Sequence[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return round(statistics.mean(values), 6) if values else None


def aggregate_ablation_metrics(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["sweep"]), str(row["variant_id"])), []).append(row)
    aggregates = []
    for _, selected in groups.items():
        first = selected[0]
        aggregates.append(
            {
                "sweep": first["sweep"],
                "variant_id": first["variant_id"],
                "variant_label": first["variant_label"],
                "fps": first["fps"],
                "k": first["k"],
                "high_similarity_threshold": first["high_similarity_threshold"],
                "temporal_policy": first["temporal_policy"],
                "pair_count": len(selected),
                "passed_count": sum(bool(row.get("passed")) for row in selected),
                "no_removal_count": sum(bool(row.get("no_removal")) for row in selected),
                "collapsed_count": sum(bool(row.get("collapsed")) for row in selected),
                "mean_high_similarity_representative_pair_count": _mean(
                    selected, "high_similarity_representative_pair_count"
                ),
                "mean_trigger_pair_count_beyond_2s": _mean(
                    selected, "trigger_pair_count_beyond_2s"
                ),
                "mean_left_kept_duration_seconds": _mean(selected, "left_kept_duration_seconds"),
                "mean_right_kept_duration_seconds": _mean(selected, "right_kept_duration_seconds"),
                "mean_removed_duration_seconds": _mean(selected, "removed_duration_seconds"),
                "mean_left_keep_segment_count": _mean(selected, "left_keep_segment_count"),
                "mean_right_keep_segment_count": _mean(selected, "right_keep_segment_count"),
                "mean_left_multirun_cluster_count": _mean(selected, "left_multirun_cluster_count"),
                "mean_right_multirun_cluster_count": _mean(selected, "right_multirun_cluster_count"),
            }
        )
    return aggregates


def _relative_media_path(output_dir: Path, value: str | None) -> str | None:
    if not value:
        return None
    path = Path(value)
    try:
        return path.resolve().relative_to(output_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def write_ablation_review_html(output_dir: Path, pair_rows: Sequence[dict[str, Any]]) -> Path:
    """Create a pair-by-pair visual review page organized by controlled sweep."""

    lines = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        "<title>30-second CLIP pruning ablations</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;margin:20px;background:#f6f7f9;color:#17202a}",
        "section{background:white;border:1px solid #d8dee6;border-radius:10px;padding:16px;margin:18px 0}",
        ".originals,.videos{display:grid;grid-template-columns:repeat(2,minmax(280px,1fr));gap:12px}",
        ".cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:14px}",
        ".card{border:1px solid #cbd3dd;border-radius:8px;padding:12px;background:#fbfcfe}",
        ".control{border-color:#3b82f6;background:#eff6ff}",
        "video{width:100%;max-height:300px;background:#111}",
        "code{font-size:.9em} .metrics{font-size:.9em;line-height:1.45}",
        "summary{font-weight:700;cursor:pointer;padding:8px 0}",
        "</style></head><body>",
        "<h1>30-second CLIP pruning ablations</h1>",
        (
            "<p>Each section uses the same synchronized pair. Inside a sweep, only the named "
            "factor changes; all other pruning settings remain at the baseline.</p>"
        ),
    ]
    for pair in pair_rows:
        lines.extend(
            [
                "<section>",
                f"<h2>{html.escape(str(pair['pair_id']))}</h2>",
                '<div class="originals">',
            ]
        )
        for side in ("left", "right"):
            src = _relative_media_path(output_dir, pair.get(f"{side}_original_video"))
            lines.append(
                f"<div><h3>{side.title()} original</h3><video controls preload=\"metadata\" "
                f"src=\"{html.escape(str(src or ''))}\"></video></div>"
            )
        lines.append("</div>")
        for sweep in ("temporal", "threshold", "sampling", "k"):
            variants = [row for row in pair.get("variants", []) if row.get("sweep") == sweep]
            lines.extend([f"<details open><summary>{sweep.title()} sweep</summary>", '<div class="cards">'])
            for variant in variants:
                is_control = (
                    float(variant["fps"]) == 1.0
                    and int(variant["k"]) == 12
                    and float(variant["high_similarity_threshold"]) == 0.82
                    and variant["temporal_policy"] == "current"
                )
                card_class = "card control" if is_control else "card"
                lines.extend(
                    [
                        f'<div class="{card_class}">',
                        f"<h3>{html.escape(str(variant['variant_label']))}</h3>",
                        (
                            '<p class="metrics">'
                            f"FPS={variant['fps']} · K={variant['k']} · threshold="
                            f"{variant['high_similarity_threshold']} · temporal="
                            f"{html.escape(str(variant['temporal_policy']))}<br>"
                            f"kept L/R={variant['left_kept_duration_seconds']}s/"
                            f"{variant['right_kept_duration_seconds']}s · removed total="
                            f"{variant['removed_duration_seconds']}s · triggers="
                            f"{variant['high_similarity_representative_pair_count']} · "
                            f"off-time(&gt;2s)={variant['trigger_pair_count_beyond_2s']}</p>"
                        ),
                        '<div class="videos">',
                    ]
                )
                for side in ("left", "right"):
                    src = _relative_media_path(output_dir, variant.get(f"{side}_pruned_video"))
                    lines.append(
                        f"<div><h4>{side.title()} pruned</h4><video controls preload=\"metadata\" "
                        f"src=\"{html.escape(str(src or ''))}\"></video></div>"
                    )
                diagnostics = _relative_media_path(output_dir, variant.get("diagnostics_path"))
                trace = _relative_media_path(output_dir, variant.get("cluster_trace_path"))
                lines.extend(
                    [
                        "</div>",
                        f'<p><a href="{html.escape(str(diagnostics or ""))}">pruning JSON</a> · '
                        f'<a href="{html.escape(str(trace or ""))}">cluster trace</a></p>',
                        "</div>",
                    ]
                )
            lines.extend(["</div></details>"])
        lines.append("</section>")
    lines.append("</body></html>")
    path = output_dir / "review.html"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_pruning_ablation(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    cache_dir: str | Path,
    pair_count: int = 10,
    max_groups: int | None = None,
    min_group_size: int = 2,
    model_id: str = DEFAULT_CLIP_MODEL,
    duration_seconds: float = 30.0,
    start_seconds: float = 0.0,
    baseline_fps: float = 1.0,
    baseline_k: int = 12,
    baseline_threshold: float = 0.82,
    baseline_temporal_policy: str = "current",
    fps_values: Sequence[float] = DEFAULT_FPS_VALUES,
    k_values: Sequence[int] = (4, 8, 12, 16, 20, 24, 30),
    threshold_values: Sequence[float] = DEFAULT_THRESHOLD_VALUES,
    temporal_policies: Sequence[str] = DEFAULT_TEMPORAL_POLICIES,
    preserve_shared_anchor_seconds: float = 0.0,
    min_pruned_video_seconds: float = 8.0,
    pruning_protection_mode: str = "min_seconds",
    min_pruned_video_percent: float | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    ffmpeg_binary: str = "ffmpeg",
    download_media: bool = False,
    encoder: ImageEncoder | None = None,
) -> dict[str, Any]:
    """Run independent temporal, threshold, FPS, and K sweeps on fixed pairs."""

    if pair_count <= 0:
        raise ValueError("pair_count must be positive")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if baseline_fps <= 0 or baseline_k <= 0:
        raise ValueError("baseline_fps and baseline_k must be positive")
    fps_grid = parse_float_values(fps_values, name="FPS", minimum=1e-9)
    k_grid = parse_k_values(k_values)
    threshold_grid = parse_float_values(
        threshold_values,
        name="threshold",
        minimum=-1.0,
        maximum=1.0,
    )
    policies = parse_temporal_policies(temporal_policies)
    if baseline_temporal_policy not in TEMPORAL_POLICY_SETTINGS:
        raise ValueError(f"unknown baseline temporal policy: {baseline_temporal_policy}")
    if not -1.0 <= baseline_threshold <= 1.0:
        raise ValueError("baseline_threshold must be between -1 and 1")

    variants = build_ablation_variants(
        baseline_fps=baseline_fps,
        baseline_k=baseline_k,
        baseline_threshold=baseline_threshold,
        baseline_temporal_policy=baseline_temporal_policy,
        fps_values=fps_grid,
        k_values=k_grid,
        threshold_values=threshold_grid,
        temporal_policies=policies,
    )
    master_fps = max([baseline_fps, *fps_grid])
    master_interval = 1.0 / master_fps
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = read_json(manifest_path)
    groups = [
        group
        for group in group_manifest_clips(manifest)
        if len(group.get("clips", [])) >= min_group_size
    ]
    rng = random.Random(random_seed)
    rng.shuffle(groups)
    if max_groups is not None:
        groups = groups[:max_groups]

    clip_encoder = encoder or TransformersClipEncoder(model_id)
    pair_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    trigger_rows: list[dict[str, Any]] = []
    centroid_rows: list[dict[str, Any]] = []
    skipped = []

    for group_index, group in enumerate(groups):
        if len(pair_rows) >= pair_count:
            break
        try:
            selected_clips = _sample_pair(group, rng)
            selected_group = {**group, "clips": selected_clips}
            master_rows = group_clip_frames(
                selected_group,
                output_root / "sampled_frames" / f"master_{_number_slug(master_fps)}fps",
                cache_dir=cache_dir,
                duration_seconds=duration_seconds,
                sample_interval_seconds=master_interval,
                start_seconds=start_seconds,
                ffmpeg_binary=ffmpeg_binary,
                download_media=download_media,
            )
            if len(master_rows) != 2:
                raise ValueError(f"expected two sampled videos, got {len(master_rows)}")
            master_embeddings = [
                clip_encoder.encode([str(frame["path"]) for frame in row["frames"]])
                for row in master_rows
            ]
            rate_cache: dict[float, tuple[list[list[dict[str, Any]]], list[list[list[float]]]]] = {}
            for fps in sorted({float(variant["fps"]) for variant in variants}):
                rate_frames = []
                rate_embeddings = []
                for side_index in range(2):
                    frames, embeddings = _derive_sampling_rate(
                        master_rows[side_index]["frames"],
                        master_embeddings[side_index],
                        fps=fps,
                        start_seconds=start_seconds,
                        duration_seconds=duration_seconds,
                    )
                    rate_frames.append(frames)
                    rate_embeddings.append(embeddings)
                rate_cache[fps] = (rate_frames, rate_embeddings)

            agents = [
                str(row["clip"].get("agent_dir") or row["user"])
                for row in master_rows
            ]
            pair_id = stable_id(group.get("day"), group.get("time_token"), *agents)
            pair_dir = output_root / "pairs" / _safe_part(pair_id)
            pair_dir.mkdir(parents=True, exist_ok=True)
            full_window = [[float(start_seconds), round(start_seconds + duration_seconds, 3)]]
            original_paths = []
            for side, row in zip(("left", "right"), master_rows):
                original_path = pair_dir / f"{side}_{_safe_part(row['clip'].get('agent_dir') or row['user'])}_original.mp4"
                result = _materialize(
                    row["clip"]["local_video"],
                    original_path,
                    full_window,
                    ffmpeg_binary=ffmpeg_binary,
                )
                if result["status"] != "materialized":
                    raise RuntimeError(f"failed to materialize {side} original: {result['error']}")
                original_paths.append(str(original_path))

            pair_row: dict[str, Any] = {
                "pair_id": pair_id,
                "day": group.get("day"),
                "time_token": group.get("time_token"),
                "left_agent": agents[0],
                "right_agent": agents[1],
                "left_original_video": original_paths[0],
                "right_original_video": original_paths[1],
                "master_fps": master_fps,
                "left_master_frame_count": len(master_rows[0]["frames"]),
                "right_master_frame_count": len(master_rows[1]["frames"]),
                "variants": [],
            }

            for variant in variants:
                fps = float(variant["fps"])
                frames, embeddings = rate_cache[fps]
                pruning = clustered_temporal_similarity_pruning(
                    frames[0],
                    frames[1],
                    embeddings[0],
                    embeddings[1],
                    start_seconds=start_seconds,
                    duration_seconds=duration_seconds,
                    sample_interval_seconds=float(variant["sample_interval_seconds"]),
                    cluster_count=int(variant["k"]),
                    high_similarity_threshold=float(variant["high_similarity_threshold"]),
                    preserve_shared_anchor_seconds=preserve_shared_anchor_seconds,
                    min_pruned_video_seconds=min_pruned_video_seconds,
                    pruning_protection_mode=pruning_protection_mode,
                    min_pruned_video_percent=min_pruned_video_percent,
                    max_pair_time_difference_seconds=variant[
                        "max_pair_time_difference_seconds"
                    ],
                    mutual_nearest_only=bool(variant["mutual_nearest_only"]),
                    split_noncontiguous_clusters=bool(
                        variant["split_noncontiguous_clusters"]
                    ),
                )
                variant_dir = pair_dir / str(variant["sweep"]) / _safe_part(variant["variant_id"])
                variant_dir.mkdir(parents=True, exist_ok=True)
                left_result = _materialize(
                    master_rows[0]["clip"]["local_video"],
                    variant_dir / "left_pruned.mp4",
                    pruning["left_keep_intervals"],
                    ffmpeg_binary=ffmpeg_binary,
                )
                right_result = _materialize(
                    master_rows[1]["clip"]["local_video"],
                    variant_dir / "right_pruned.mp4",
                    pruning["right_keep_intervals"],
                    ffmpeg_binary=ffmpeg_binary,
                )
                diagnostics_path = variant_dir / "pruning.json"
                cluster_trace_path = variant_dir / "cluster_trace.json"
                write_json(
                    diagnostics_path,
                    {
                        "pair_id": pair_id,
                        "configuration": variant,
                        "left_output": left_result,
                        "right_output": right_result,
                        "temporal_pruning": pruning,
                    },
                )
                trace = build_cluster_trace(
                    pair_id,
                    int(variant["k"]),
                    frames[0],
                    frames[1],
                    pruning,
                )
                trace["configuration"] = variant
                centroid_index, current_centroid_rows = materialize_centroid_frames(
                    variant_dir,
                    trace,
                )
                write_json(cluster_trace_path, trace)
                write_json(variant_dir / "centroid_frames.json", centroid_index)

                current_assignments, current_triggers = _tabular_trace_rows(trace)
                prefix = {key: variant.get(key) for key in TRACE_PREFIX_FIELDS}
                assignment_rows.extend([{**prefix, **row} for row in current_assignments])
                trigger_rows.extend([{**prefix, **row} for row in current_triggers])
                centroid_rows.extend([{**prefix, **row} for row in current_centroid_rows])

                trigger_differences = [
                    float(row.get("timestamp_difference_seconds", 0.0))
                    for row in pruning["high_similarity_representative_pairs"]
                ]
                left_multirun, left_mean_span = _cluster_temporal_metrics(
                    pruning["left_cluster_decisions"],
                    sample_interval_seconds=float(variant["sample_interval_seconds"]),
                )
                right_multirun, right_mean_span = _cluster_temporal_metrics(
                    pruning["right_cluster_decisions"],
                    sample_interval_seconds=float(variant["sample_interval_seconds"]),
                )
                no_removal = float(pruning["removed_duration_seconds"]) <= 0.0
                collapsed = not pruning["left_keep_intervals"] or not pruning["right_keep_intervals"]
                metric = {
                    "pair_id": pair_id,
                    **variant,
                    "left_sampled_frame_count": len(frames[0]),
                    "right_sampled_frame_count": len(frames[1]),
                    "left_cluster_count": pruning["left_cluster_count"],
                    "right_cluster_count": pruning["right_cluster_count"],
                    "high_similarity_representative_pair_count": len(trigger_differences),
                    "trigger_pair_count_beyond_2s": sum(value > 2.0 for value in trigger_differences),
                    "mean_trigger_time_difference_seconds": (
                        round(statistics.mean(trigger_differences), 6) if trigger_differences else None
                    ),
                    "median_trigger_time_difference_seconds": (
                        round(statistics.median(trigger_differences), 6) if trigger_differences else None
                    ),
                    "max_trigger_time_difference_seconds": (
                        round(max(trigger_differences), 6) if trigger_differences else None
                    ),
                    "left_multirun_cluster_count": left_multirun,
                    "right_multirun_cluster_count": right_multirun,
                    "left_mean_cluster_span_seconds": left_mean_span,
                    "right_mean_cluster_span_seconds": right_mean_span,
                    "left_marked_frame_count": len(pruning["left_marked_frame_indices"]),
                    "right_marked_frame_count": len(pruning["right_marked_frame_indices"]),
                    "left_restored_frame_count": len(pruning["left_restored_frame_indices"]),
                    "right_restored_frame_count": len(pruning["right_restored_frame_indices"]),
                    "left_kept_duration_seconds": pruning["left_kept_duration_seconds"],
                    "right_kept_duration_seconds": pruning["right_kept_duration_seconds"],
                    "left_removed_duration_seconds": pruning["left_removed_duration_seconds"],
                    "right_removed_duration_seconds": pruning["right_removed_duration_seconds"],
                    "removed_duration_seconds": pruning["removed_duration_seconds"],
                    "left_keep_segment_count": len(pruning["left_keep_intervals"]),
                    "right_keep_segment_count": len(pruning["right_keep_intervals"]),
                    "no_removal": no_removal,
                    "collapsed": collapsed,
                    "passed": pruning["passed"],
                    "left_materialization_status": left_result["status"],
                    "right_materialization_status": right_result["status"],
                    "left_pruned_video": left_result["path"],
                    "right_pruned_video": right_result["path"],
                    "diagnostics_path": str(diagnostics_path),
                    "cluster_trace_path": str(cluster_trace_path),
                }
                metric_rows.append(metric)
                pair_row["variants"].append(metric)

            write_json(pair_dir / "pair_summary.json", pair_row)
            pair_rows.append(pair_row)
        except Exception as exc:
            skipped.append(
                {
                    "group_index": group_index,
                    "day": group.get("day"),
                    "time_token": group.get("time_token"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    aggregates = aggregate_ablation_metrics(metric_rows)
    write_jsonl(output_root / "cohort.jsonl", pair_rows)
    write_jsonl(output_root / "ablation_metrics.jsonl", metric_rows)
    _write_csv(output_root / "ablation_metrics.csv", metric_rows, METRIC_FIELDS)
    _write_csv(output_root / "sweep_aggregates.csv", aggregates, AGGREGATE_FIELDS)
    _write_csv(output_root / "cluster_assignments.csv", assignment_rows, ASSIGNMENT_FIELDS)
    _write_csv(output_root / "trigger_pairs.csv", trigger_rows, TRIGGER_FIELDS)
    _write_csv(output_root / "centroid_frames.csv", centroid_rows, CENTROID_FIELDS)
    review_path = write_ablation_review_html(output_root, pair_rows)
    failures = sum(
        row["left_materialization_status"] == "failed"
        or row["right_materialization_status"] == "failed"
        for row in metric_rows
    )
    summary = {
        "manifest_path": str(manifest_path),
        "output_dir": str(output_root),
        "review_path": str(review_path),
        "settings": {
            "model_id": clip_encoder.model_id,
            "duration_seconds": duration_seconds,
            "start_seconds": start_seconds,
            "pair_count_requested": pair_count,
            "max_groups": max_groups,
            "min_group_size": min_group_size,
            "baseline_fps": baseline_fps,
            "baseline_k": baseline_k,
            "baseline_threshold": baseline_threshold,
            "baseline_temporal_policy": baseline_temporal_policy,
            "fps_values": fps_grid,
            "k_values": k_grid,
            "threshold_values": threshold_grid,
            "temporal_policies": policies,
            "master_fps": master_fps,
            "configuration_count_per_pair": len(variants),
            "preserve_shared_anchor_seconds": preserve_shared_anchor_seconds,
            "min_pruned_video_seconds": min_pruned_video_seconds,
            "pruning_protection_mode": pruning_protection_mode,
            "min_pruned_video_percent": min_pruned_video_percent,
            "random_seed": random_seed,
            "download_media": download_media,
        },
        "pair_count": len(pair_rows),
        "target_met": len(pair_rows) >= pair_count,
        "variant_count": len(metric_rows),
        "expected_variant_count": pair_count * len(variants),
        "materialization_failure_count": failures,
        "skipped_group_count": len(skipped),
        "skipped_groups": skipped,
        "sweep_aggregates": aggregates,
    }
    write_json(output_root / "summary.json", summary)
    if not summary["target_met"]:
        raise RuntimeError(
            f"requested {pair_count} synchronized pairs but produced {len(pair_rows)}; "
            f"inspect {output_root / 'summary.json'}"
        )
    if failures:
        raise RuntimeError(
            f"{failures} ablation variants failed video materialization; "
            f"inspect {output_root / 'ablation_metrics.csv'}"
        )
    return summary

