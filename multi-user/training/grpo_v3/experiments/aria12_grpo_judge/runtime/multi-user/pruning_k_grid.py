"""Paired K-grid experiment for clustered temporal video pruning.

The experiment samples each synchronized video pair once, extracts and embeds
its frames once, and then applies every requested cluster count to those exact
same inputs. This avoids the cohort drift that would result from independently
running the K-dependent benchmark miner for each cluster count.
"""

from __future__ import annotations

import csv
import html
import random
import shutil
import statistics
from pathlib import Path
from typing import Any, Iterable, Sequence

from .clip_gap_demo import DEFAULT_CLIP_MODEL, ImageEncoder, TransformersClipEncoder
from .evidence import group_manifest_clips
from .group_relative_clip_sampling import (
    clustered_temporal_similarity_pruning,
    group_clip_frames,
    materialize_pruned_video,
)
from .io_utils import read_json, stable_id, write_json, write_jsonl


DEFAULT_K_VALUES = (4, 8, 12, 16, 20, 24, 30)
DEFAULT_RANDOM_SEED = 20260716

GRID_CSV_FIELDS = (
    "pair_id",
    "day",
    "time_token",
    "left_agent",
    "right_agent",
    "k",
    "high_similarity_threshold",
    "left_cluster_count",
    "right_cluster_count",
    "high_similarity_representative_pair_count",
    "left_marked_cluster_count",
    "right_marked_cluster_count",
    "left_marked_frame_count",
    "right_marked_frame_count",
    "left_restored_frame_count",
    "right_restored_frame_count",
    "left_kept_duration_seconds",
    "right_kept_duration_seconds",
    "left_removed_duration_seconds",
    "right_removed_duration_seconds",
    "left_keep_segment_count",
    "right_keep_segment_count",
    "removed_duration_seconds",
    "no_removal",
    "collapsed",
    "passed",
    "left_materialization_status",
    "right_materialization_status",
    "left_pruned_video",
    "right_pruned_video",
    "diagnostics_path",
)

ASSIGNMENT_CSV_FIELDS = (
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

TRIGGER_PAIR_CSV_FIELDS = (
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

CENTROID_FRAME_CSV_FIELDS = (
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


def parse_k_values(value: str | Iterable[int]) -> list[int]:
    """Parse, validate, and de-duplicate a K grid while preserving its order."""

    raw_values: Iterable[Any]
    if isinstance(value, str):
        raw_values = [part.strip() for part in value.split(",") if part.strip()]
    else:
        raw_values = value
    values: list[int] = []
    for raw in raw_values:
        try:
            parsed = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid K value: {raw!r}") from exc
        if parsed <= 0:
            raise ValueError(f"K values must be positive, got {parsed}")
        if parsed not in values:
            values.append(parsed)
    if not values:
        raise ValueError("at least one K value is required")
    return values


def _safe_part(value: Any) -> str:
    text = str(value or "unknown").strip()
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
    return safe.strip("_") or "unknown"


def _sample_pair(group: dict[str, Any], rng: random.Random) -> list[dict[str, Any]]:
    clips = sorted(group.get("clips", []), key=lambda row: str(row.get("agent_dir")))
    if len(clips) < 2:
        raise ValueError("synchronized group contains fewer than two videos")
    selected = clips if len(clips) == 2 else rng.sample(clips, 2)
    return sorted(selected, key=lambda row: str(row.get("agent_dir")))


def _materialize(
    source_video: str | Path,
    output_video: Path,
    keep_intervals: list[list[float]] | list[tuple[float, float]],
    *,
    ffmpeg_binary: str,
) -> dict[str, Any]:
    if not keep_intervals:
        return {
            "status": "skipped_empty_keep_intervals",
            "path": None,
            "error": "pruning removed the entire requested window",
        }
    try:
        materialize_pruned_video(
            source_video,
            output_video,
            keep_intervals,
            ffmpeg_binary=ffmpeg_binary,
        )
    except Exception as exc:
        return {
            "status": "failed",
            "path": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {"status": "materialized", "path": str(output_video), "error": None}


def _sampled_frame_trace(frames: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "frame_index": index,
            "timestamp_seconds": frame.get("timestamp_seconds"),
            "path": frame.get("path"),
        }
        for index, frame in enumerate(frames)
    ]


def _cluster_assignment_trace(
    cluster_decisions: Sequence[dict[str, Any]],
    frames: Sequence[dict[str, Any]],
    *,
    final_marked_indices: Sequence[int],
    restored_indices: Sequence[int],
) -> list[dict[str, Any]]:
    """Expose medoids and every member's final interval-pruning status."""

    marked = {int(index) for index in final_marked_indices}
    restored = {int(index) for index in restored_indices}
    clusters = []
    for decision in cluster_decisions:
        representative_index = int(decision["frame_index"])
        members = []
        for raw_index in decision.get("member_indices", []):
            frame_index = int(raw_index)
            frame = frames[frame_index]
            status = "removed" if frame_index in marked else "restored" if frame_index in restored else "kept"
            members.append(
                {
                    "frame_index": frame_index,
                    "timestamp_seconds": frame.get("timestamp_seconds"),
                    "path": frame.get("path"),
                    "is_medoid": frame_index == representative_index,
                    "final_status": status,
                }
            )
        clusters.append(
            {
                "cluster_index": int(decision["cluster_index"]),
                "visual_cluster_index": int(
                    decision.get("visual_cluster_index", decision["cluster_index"])
                ),
                "temporal_component_index": int(decision.get("temporal_component_index", 0)),
                "triggered_for_pruning": decision.get("status") == "marked_for_pruning",
                "representative_frame_index": representative_index,
                "representative_timestamp_seconds": decision.get("timestamp_seconds"),
                "representative_path": decision.get("path"),
                "member_count": len(members),
                "removed_member_count": sum(member["final_status"] == "removed" for member in members),
                "restored_member_count": sum(member["final_status"] == "restored" for member in members),
                "members": members,
            }
        )
    return clusters


def build_cluster_trace(
    pair_id: str,
    k: int,
    left_frames: Sequence[dict[str, Any]],
    right_frames: Sequence[dict[str, Any]],
    pruning: dict[str, Any],
) -> dict[str, Any]:
    """Build a human-readable trace of the actual medoid clustering decision."""

    return {
        "pair_id": pair_id,
        "k": int(k),
        "note": (
            "The displayed cluster center is the medoid sampled frame. These medoid embeddings, "
            "rather than the intermediate k-means centroid vectors, are what pruning compares."
        ),
        "left_sampled_frames": _sampled_frame_trace(left_frames),
        "right_sampled_frames": _sampled_frame_trace(right_frames),
        "left_clusters": _cluster_assignment_trace(
            pruning["left_cluster_decisions"],
            left_frames,
            final_marked_indices=pruning["left_marked_frame_indices"],
            restored_indices=pruning["left_restored_frame_indices"],
        ),
        "right_clusters": _cluster_assignment_trace(
            pruning["right_cluster_decisions"],
            right_frames,
            final_marked_indices=pruning["right_marked_frame_indices"],
            restored_indices=pruning["right_restored_frame_indices"],
        ),
        "high_similarity_representative_pairs": pruning["high_similarity_representative_pairs"],
        "left_remove_intervals": pruning["left_remove_intervals"],
        "right_remove_intervals": pruning["right_remove_intervals"],
        "left_keep_intervals": pruning["left_keep_intervals"],
        "right_keep_intervals": pruning["right_keep_intervals"],
    }


def materialize_centroid_frames(
    variant_dir: Path,
    cluster_trace: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Copy the sampled medoid nearest each k-means centroid into an explicit output folder."""

    index: dict[str, Any] = {
        "pair_id": cluster_trace["pair_id"],
        "k": cluster_trace["k"],
        "definition": (
            "A k-means centroid is an embedding vector, not an image. Each exported centroid frame "
            "is the sampled medoid nearest that centroid and is the representative frame actually "
            "used by cross-video pruning."
        ),
        "selection_method": "sampled_medoid_nearest_cosine_kmeans_centroid",
        "left": [],
        "right": [],
    }
    rows = []
    for side in ("left", "right"):
        side_dir = variant_dir / "centroid_frames" / side
        side_dir.mkdir(parents=True, exist_ok=True)
        for cluster in cluster_trace[f"{side}_clusters"]:
            source = Path(str(cluster["representative_path"]))
            if not source.exists() or source.stat().st_size == 0:
                raise FileNotFoundError(f"missing centroid/medoid sampled frame: {source}")
            cluster_index = int(cluster["cluster_index"])
            frame_index = int(cluster["representative_frame_index"])
            timestamp = float(cluster["representative_timestamp_seconds"])
            suffix = source.suffix or ".png"
            output = side_dir / (
                f"cluster_{cluster_index:02d}_centroid_frame_{frame_index:03d}_{timestamp:.2f}s{suffix}"
            )
            shutil.copy2(source, output)
            cluster["centroid_frame_output"] = str(output)
            row = {
                "pair_id": cluster_trace["pair_id"],
                "k": cluster_trace["k"],
                "side": side,
                "cluster_index": cluster_index,
                "cluster_triggered_for_pruning": cluster["triggered_for_pruning"],
                "centroid_frame_index": frame_index,
                "centroid_frame_timestamp_seconds": timestamp,
                "centroid_frame_output": str(output),
                "source_sampled_frame_path": str(source),
                "selection_method": index["selection_method"],
            }
            index[side].append(row)
            rows.append(row)
    return index, rows


def _write_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
    fieldnames: Sequence[str] = GRID_CSV_FIELDS,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _tabular_trace_rows(cluster_trace: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    assignments = []
    for side in ("left", "right"):
        for cluster in cluster_trace[f"{side}_clusters"]:
            for member in cluster["members"]:
                assignments.append(
                    {
                        "pair_id": cluster_trace["pair_id"],
                        "k": cluster_trace["k"],
                        "side": side,
                        "cluster_index": cluster["cluster_index"],
                        "visual_cluster_index": cluster.get("visual_cluster_index"),
                        "temporal_component_index": cluster.get("temporal_component_index"),
                        "cluster_triggered_for_pruning": cluster["triggered_for_pruning"],
                        "medoid_frame_index": cluster["representative_frame_index"],
                        "medoid_timestamp_seconds": cluster["representative_timestamp_seconds"],
                        "member_frame_index": member["frame_index"],
                        "member_timestamp_seconds": member["timestamp_seconds"],
                        "member_is_medoid": member["is_medoid"],
                        "member_final_status": member["final_status"],
                        "member_path": member["path"],
                    }
                )
    trigger_pairs = [
        {
            "pair_id": cluster_trace["pair_id"],
            "k": cluster_trace["k"],
            **row,
        }
        for row in cluster_trace["high_similarity_representative_pairs"]
    ]
    return assignments, trigger_pairs


def _mean(rows: Sequence[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return round(statistics.mean(values), 6) if values else None


def _median(rows: Sequence[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return round(statistics.median(values), 6) if values else None


def aggregate_k_metrics(rows: Sequence[dict[str, Any]], k_values: Sequence[int]) -> list[dict[str, Any]]:
    aggregates = []
    for k in k_values:
        selected = [row for row in rows if int(row["k"]) == int(k)]
        aggregates.append(
            {
                "k": int(k),
                "pair_count": len(selected),
                "passed_count": sum(bool(row.get("passed")) for row in selected),
                "both_videos_materialized_count": sum(
                    row.get("left_materialization_status") == "materialized"
                    and row.get("right_materialization_status") == "materialized"
                    for row in selected
                ),
                "no_removal_count": sum(bool(row.get("no_removal")) for row in selected),
                "collapsed_count": sum(bool(row.get("collapsed")) for row in selected),
                "mean_left_kept_duration_seconds": _mean(selected, "left_kept_duration_seconds"),
                "mean_right_kept_duration_seconds": _mean(selected, "right_kept_duration_seconds"),
                "median_left_kept_duration_seconds": _median(selected, "left_kept_duration_seconds"),
                "median_right_kept_duration_seconds": _median(selected, "right_kept_duration_seconds"),
                "mean_removed_duration_seconds": _mean(selected, "removed_duration_seconds"),
                "mean_left_marked_frame_count": _mean(selected, "left_marked_frame_count"),
                "mean_right_marked_frame_count": _mean(selected, "right_marked_frame_count"),
                "mean_left_keep_segment_count": _mean(selected, "left_keep_segment_count"),
                "mean_right_keep_segment_count": _mean(selected, "right_keep_segment_count"),
            }
        )
    return aggregates


def _relative_media_path(output_dir: Path, value: str | None) -> str | None:
    if not value:
        return None
    path = Path(value)
    try:
        return path.relative_to(output_dir).as_posix()
    except ValueError:
        return path.as_posix()


def write_review_html(output_dir: Path, pair_rows: Sequence[dict[str, Any]]) -> Path:
    """Write a review index for side-by-side inspection of every K."""

    lines = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        "<title>Pruning K-grid review</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;margin:24px;background:#f6f7f9;color:#17202a}",
        ".pair{background:white;border:1px solid #dfe4ea;border-radius:12px;padding:18px;margin:20px 0}",
        ".videos{display:grid;grid-template-columns:repeat(2,minmax(280px,1fr));gap:14px}",
        "video{width:100%;max-height:320px;background:#111;border-radius:8px}",
        ".variant{border-top:1px solid #e6e9ed;margin-top:18px;padding-top:14px}",
        ".metrics{font-family:ui-monospace,monospace;font-size:13px;color:#445}",
        ".missing{padding:32px;background:#fff3cd;border-radius:8px;color:#664d03}",
        ".filmstrip{display:flex;gap:7px;overflow-x:auto;padding:8px 2px 14px}",
        ".frame{min-width:112px;font-family:ui-monospace,monospace;font-size:11px}",
        ".frame img{display:block;width:112px;height:76px;object-fit:cover;border:3px solid transparent;border-radius:6px}",
        ".frame.medoid img{border-color:#0d6efd}.frame.removed{color:#a61b1b}.frame.restored{color:#8a5a00}",
        ".cluster-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:10px}",
        ".cluster{border:1px solid #dfe4ea;border-radius:8px;padding:9px;background:#fafbfc}",
        ".cluster.triggered{border-color:#dc3545;background:#fff7f7}",
        "details{margin-top:10px}summary{cursor:pointer;font-weight:650}",
        "table{border-collapse:collapse;font-size:13px}th,td{border:1px solid #dfe4ea;padding:5px 8px;text-align:left}",
        "@media(max-width:760px){.videos{grid-template-columns:1fr}}",
        "</style></head><body>",
        "<h1>Cluster-pruning K-grid review</h1>",
        "<p>Every K uses the same synchronized pair, sampled frames, CLIP embeddings, threshold, and fixed pruning interval.</p>",
    ]

    def video_card(label: str, path_value: str | None, status: str = "materialized") -> list[str]:
        label_html = html.escape(label)
        relative = _relative_media_path(output_dir, path_value)
        if status != "materialized" or relative is None:
            return [f"<div><h4>{label_html}</h4><div class=\"missing\">{html.escape(status)}</div></div>"]
        source = html.escape(relative, quote=True)
        return [
            f"<div><h4>{label_html}</h4>",
            f'<video controls preload="metadata" src="{source}"></video></div>',
        ]

    def frame_card(frame: dict[str, Any], *, show_status: bool = False) -> str:
        relative = _relative_media_path(output_dir, frame.get("path"))
        status = str(frame.get("final_status") or "") if show_status else ""
        classes = ["frame"]
        if frame.get("is_medoid"):
            classes.append("medoid")
        if status:
            classes.append(status)
        label = f"#{int(frame['frame_index']):02d} | {frame.get('timestamp_seconds')}s"
        if status:
            label += f" | {status}"
        if relative is None:
            return f'<div class="{" ".join(classes)}">{html.escape(label)}<br>missing image</div>'
        return (
            f'<div class="{" ".join(classes)}"><img loading="lazy" '
            f'src="{html.escape(relative, quote=True)}" alt="{html.escape(label, quote=True)}">'
            f"{html.escape(label)}</div>"
        )

    def cluster_panel(side_label: str, clusters: Sequence[dict[str, Any]]) -> list[str]:
        panel = [f"<h4>{html.escape(side_label)} clusters</h4>", '<div class="cluster-grid">']
        for cluster in clusters:
            triggered = bool(cluster.get("triggered_for_pruning"))
            css = "cluster triggered" if triggered else "cluster"
            title = (
                f"Cluster {cluster['cluster_index']} | medoid #{cluster['representative_frame_index']} "
                f"@ {cluster.get('representative_timestamp_seconds')}s | "
                f"{'triggered' if triggered else 'not triggered'}"
            )
            panel.extend(
                [
                    f'<div class="{css}"><strong>{html.escape(title)}</strong>',
                    f"<p>{cluster.get('removed_member_count', 0)}/{cluster.get('member_count', 0)} removed; "
                    f"{cluster.get('restored_member_count', 0)} restored</p>",
                    "<h5>Exported centroid frame (nearest sampled medoid)</h5>",
                    '<div class="filmstrip">',
                    frame_card(
                        {
                            "frame_index": cluster["representative_frame_index"],
                            "timestamp_seconds": cluster.get("representative_timestamp_seconds"),
                            "path": cluster.get("centroid_frame_output"),
                            "is_medoid": True,
                        }
                    ),
                    "</div><h5>Assigned sampled frames</h5>",
                    '<div class="filmstrip">',
                    *(frame_card(member, show_status=True) for member in cluster.get("members", [])),
                    "</div></div>",
                ]
            )
        panel.append("</div>")
        return panel

    for pair in pair_rows:
        pair_id = html.escape(str(pair["pair_id"]))
        lines.extend(
            [
                '<section class="pair">',
                f"<h2>{pair_id}</h2>",
                f"<p>{html.escape(str(pair.get('day')))} {html.escape(str(pair.get('time_token')))}</p>",
                '<div class="videos">',
                *video_card(str(pair["left_agent"]) + " original", pair.get("left_original_video")),
                *video_card(str(pair["right_agent"]) + " original", pair.get("right_original_video")),
                "</div>",
                f"<h3>Sampled frames | {html.escape(str(pair['left_agent']))} ({len(pair.get('left_sampled_frames', []))})</h3>",
                '<div class="filmstrip">',
                *(frame_card(frame) for frame in pair.get("left_sampled_frames", [])),
                "</div>",
                f"<h3>Sampled frames | {html.escape(str(pair['right_agent']))} ({len(pair.get('right_sampled_frames', []))})</h3>",
                '<div class="filmstrip">',
                *(frame_card(frame) for frame in pair.get("right_sampled_frames", [])),
                "</div>",
            ]
        )
        for variant in pair.get("variants", []):
            lines.extend(
                [
                    '<div class="variant">',
                    f"<h3>K={int(variant['k'])}</h3>",
                    '<div class="videos">',
                    *video_card(
                        str(pair["left_agent"]) + " pruned",
                        variant.get("left_pruned_video"),
                        str(variant.get("left_materialization_status")),
                    ),
                    *video_card(
                        str(pair["right_agent"]) + " pruned",
                        variant.get("right_pruned_video"),
                        str(variant.get("right_materialization_status")),
                    ),
                    "</div>",
                    '<p class="metrics">'
                    f"kept L/R={variant.get('left_kept_duration_seconds')}/{variant.get('right_kept_duration_seconds')}s; "
                    f"marked frames L/R={variant.get('left_marked_frame_count')}/{variant.get('right_marked_frame_count')}; "
                    f"segments L/R={variant.get('left_keep_segment_count')}/{variant.get('right_keep_segment_count')}; "
                    f"passed={str(bool(variant.get('passed'))).lower()}"
                    "</p>",
                ]
            )
            trace = variant.get("cluster_trace", {})
            high_pairs = trace.get("high_similarity_representative_pairs", [])
            lines.extend(
                [
                    "<details><summary>Cluster centers, assignments, and trigger pairs</summary>",
                    "<p>Blue borders mark medoids. Red cluster cards were nominated for pruning. "
                    "Member labels show the final removed/restored/kept decision.</p>",
                    "<h4>High-similarity medoid pairs</h4>",
                    "<table><thead><tr><th>Left cluster</th><th>Right cluster</th><th>Similarity</th>"
                    "<th>Left medoid time</th><th>Right medoid time</th></tr></thead><tbody>",
                    *(
                        "<tr>"
                        f"<td>{int(row['left_cluster_index'])}</td>"
                        f"<td>{int(row['right_cluster_index'])}</td>"
                        f"<td>{float(row['similarity']):.4f}</td>"
                        f"<td>{row.get('left_representative_timestamp_seconds')}</td>"
                        f"<td>{row.get('right_representative_timestamp_seconds')}</td>"
                        "</tr>"
                        for row in high_pairs
                    ),
                    "</tbody></table>",
                    *cluster_panel(str(pair["left_agent"]), trace.get("left_clusters", [])),
                    *cluster_panel(str(pair["right_agent"]), trace.get("right_clusters", [])),
                    "</details></div>",
                ]
            )
        lines.append("</section>")
    lines.append("</body></html>")
    review_path = output_dir / "review.html"
    review_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return review_path


def run_pruning_k_grid(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    cache_dir: str | Path,
    pair_count: int = 10,
    max_groups: int | None = None,
    min_group_size: int = 2,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    model_id: str = DEFAULT_CLIP_MODEL,
    duration_seconds: float = 30.0,
    sample_interval_seconds: float = 1.0,
    start_seconds: float = 0.0,
    high_similarity_threshold: float = 0.82,
    preserve_shared_anchor_seconds: float = 0.0,
    min_pruned_video_seconds: float = 8.0,
    pruning_protection_mode: str = "min_seconds",
    min_pruned_video_percent: float | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    ffmpeg_binary: str = "ffmpeg",
    download_media: bool = False,
    encoder: ImageEncoder | None = None,
) -> dict[str, Any]:
    """Sample a fixed pair cohort and materialize all requested K variants."""

    if pair_count <= 0:
        raise ValueError("pair_count must be positive")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if sample_interval_seconds <= 0:
        raise ValueError("sample_interval_seconds must be positive")
    grid = parse_k_values(k_values)
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
    trigger_pair_rows: list[dict[str, Any]] = []
    centroid_frame_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    frames_root = output_root / "sampled_frames"
    pairs_root = output_root / "pairs"

    for group_index, group in enumerate(groups):
        if len(pair_rows) >= pair_count:
            break
        try:
            selected_clips = _sample_pair(group, rng)
            selected_group = {**group, "clips": selected_clips}
            rows = group_clip_frames(
                selected_group,
                frames_root,
                cache_dir=cache_dir,
                duration_seconds=duration_seconds,
                sample_interval_seconds=sample_interval_seconds,
                start_seconds=start_seconds,
                ffmpeg_binary=ffmpeg_binary,
                download_media=download_media,
            )
            if len(rows) != 2:
                raise ValueError(f"expected two sampled videos, got {len(rows)}")
            embeddings = [
                clip_encoder.encode([str(frame["path"]) for frame in row["frames"]])
                for row in rows
            ]

            agents = [str(row["clip"].get("agent_dir") or row["user"]) for row in rows]
            pair_id = stable_id(group.get("day"), group.get("time_token"), *agents)
            pair_dir = pairs_root / _safe_part(pair_id)
            pair_dir.mkdir(parents=True, exist_ok=True)
            full_window = [[float(start_seconds), round(float(start_seconds) + float(duration_seconds), 3)]]
            original_outputs = []
            for side, row in zip(("left", "right"), rows):
                original_path = pair_dir / f"{side}_{_safe_part(row['clip'].get('agent_dir') or row['user'])}_original.mp4"
                original_result = _materialize(
                    row["clip"]["local_video"],
                    original_path,
                    full_window,
                    ffmpeg_binary=ffmpeg_binary,
                )
                if original_result["status"] != "materialized":
                    raise RuntimeError(f"failed to materialize {side} original: {original_result['error']}")
                original_outputs.append(str(original_path))

            pair_row: dict[str, Any] = {
                "pair_id": pair_id,
                "day": group.get("day"),
                "time_token": group.get("time_token"),
                "clip_clock": group.get("clip_clock"),
                "left_agent": agents[0],
                "right_agent": agents[1],
                "left_clip_id": rows[0]["clip"].get("clip_id"),
                "right_clip_id": rows[1]["clip"].get("clip_id"),
                "left_source_video": rows[0]["clip"].get("local_video"),
                "right_source_video": rows[1]["clip"].get("local_video"),
                "left_original_video": original_outputs[0],
                "right_original_video": original_outputs[1],
                "left_sampled_frame_count": len(rows[0]["frames"]),
                "right_sampled_frame_count": len(rows[1]["frames"]),
                "left_sampled_frames": _sampled_frame_trace(rows[0]["frames"]),
                "right_sampled_frames": _sampled_frame_trace(rows[1]["frames"]),
                "variants": [],
            }
            pair_metrics: list[dict[str, Any]] = []
            pair_assignments: list[dict[str, Any]] = []
            pair_trigger_pairs: list[dict[str, Any]] = []
            pair_centroid_frames: list[dict[str, Any]] = []

            for k in grid:
                pruning = clustered_temporal_similarity_pruning(
                    rows[0]["frames"],
                    rows[1]["frames"],
                    embeddings[0],
                    embeddings[1],
                    start_seconds=start_seconds,
                    duration_seconds=duration_seconds,
                    sample_interval_seconds=sample_interval_seconds,
                    cluster_count=k,
                    high_similarity_threshold=high_similarity_threshold,
                    preserve_shared_anchor_seconds=preserve_shared_anchor_seconds,
                    min_pruned_video_seconds=min_pruned_video_seconds,
                    pruning_protection_mode=pruning_protection_mode,
                    min_pruned_video_percent=min_pruned_video_percent,
                )
                variant_dir = pair_dir / f"K_{k:02d}"
                variant_dir.mkdir(parents=True, exist_ok=True)
                diagnostics_path = variant_dir / "pruning.json"
                cluster_trace_path = variant_dir / "cluster_trace.json"
                left_result = _materialize(
                    rows[0]["clip"]["local_video"],
                    variant_dir / "left_pruned.mp4",
                    pruning["left_keep_intervals"],
                    ffmpeg_binary=ffmpeg_binary,
                )
                right_result = _materialize(
                    rows[1]["clip"]["local_video"],
                    variant_dir / "right_pruned.mp4",
                    pruning["right_keep_intervals"],
                    ffmpeg_binary=ffmpeg_binary,
                )
                write_json(
                    diagnostics_path,
                    {
                        "pair_id": pair_id,
                        "k": k,
                        "left_output": left_result,
                        "right_output": right_result,
                        "temporal_pruning": pruning,
                    },
                )
                cluster_trace = build_cluster_trace(
                    pair_id,
                    k,
                    rows[0]["frames"],
                    rows[1]["frames"],
                    pruning,
                )
                centroid_frame_index, variant_centroid_frames = materialize_centroid_frames(
                    variant_dir,
                    cluster_trace,
                )
                write_json(cluster_trace_path, cluster_trace)
                write_json(variant_dir / "centroid_frames.json", centroid_frame_index)
                variant_assignments, variant_trigger_pairs = _tabular_trace_rows(cluster_trace)
                pair_assignments.extend(variant_assignments)
                pair_trigger_pairs.extend(variant_trigger_pairs)
                pair_centroid_frames.extend(variant_centroid_frames)
                no_removal = float(pruning["removed_duration_seconds"]) <= 0.0
                collapsed = not pruning["left_keep_intervals"] or not pruning["right_keep_intervals"]
                metric = {
                    "pair_id": pair_id,
                    "day": group.get("day"),
                    "time_token": group.get("time_token"),
                    "left_agent": agents[0],
                    "right_agent": agents[1],
                    "k": k,
                    "high_similarity_threshold": high_similarity_threshold,
                    "left_cluster_count": pruning["left_cluster_count"],
                    "right_cluster_count": pruning["right_cluster_count"],
                    "high_similarity_representative_pair_count": pruning[
                        "high_similarity_representative_pair_count"
                    ],
                    "left_marked_cluster_count": pruning["left_marked_cluster_count"],
                    "right_marked_cluster_count": pruning["right_marked_cluster_count"],
                    "left_marked_frame_count": len(pruning["left_marked_frame_indices"]),
                    "right_marked_frame_count": len(pruning["right_marked_frame_indices"]),
                    "left_restored_frame_count": len(pruning["left_restored_frame_indices"]),
                    "right_restored_frame_count": len(pruning["right_restored_frame_indices"]),
                    "left_kept_duration_seconds": pruning["left_kept_duration_seconds"],
                    "right_kept_duration_seconds": pruning["right_kept_duration_seconds"],
                    "left_removed_duration_seconds": pruning["left_removed_duration_seconds"],
                    "right_removed_duration_seconds": pruning["right_removed_duration_seconds"],
                    "left_keep_segment_count": len(pruning["left_keep_intervals"]),
                    "right_keep_segment_count": len(pruning["right_keep_intervals"]),
                    "removed_duration_seconds": pruning["removed_duration_seconds"],
                    "no_removal": no_removal,
                    "collapsed": collapsed,
                    "passed": pruning["passed"],
                    "left_materialization_status": left_result["status"],
                    "right_materialization_status": right_result["status"],
                    "left_pruned_video": left_result["path"],
                    "right_pruned_video": right_result["path"],
                    "diagnostics_path": str(diagnostics_path),
                }
                pair_row["variants"].append(
                    {
                        **metric,
                        "cluster_trace_path": str(cluster_trace_path),
                        "cluster_trace": cluster_trace,
                    }
                )
                pair_metrics.append(metric)

            write_json(pair_dir / "pair_summary.json", pair_row)
            pair_rows.append(pair_row)
            metric_rows.extend(pair_metrics)
            assignment_rows.extend(pair_assignments)
            trigger_pair_rows.extend(pair_trigger_pairs)
            centroid_frame_rows.extend(pair_centroid_frames)
        except Exception as exc:
            skipped.append(
                {
                    "group_index": group_index,
                    "day": group.get("day"),
                    "time_token": group.get("time_token"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    write_jsonl(output_root / "cohort.jsonl", pair_rows)
    write_jsonl(output_root / "grid_metrics.jsonl", metric_rows)
    _write_csv(output_root / "grid_metrics.csv", metric_rows)
    _write_csv(output_root / "cluster_assignments.csv", assignment_rows, ASSIGNMENT_CSV_FIELDS)
    _write_csv(output_root / "trigger_pairs.csv", trigger_pair_rows, TRIGGER_PAIR_CSV_FIELDS)
    _write_csv(output_root / "centroid_frames.csv", centroid_frame_rows, CENTROID_FRAME_CSV_FIELDS)
    review_path = write_review_html(output_root, pair_rows)
    aggregates = aggregate_k_metrics(metric_rows, grid)
    materialization_failure_count = sum(
        row.get("left_materialization_status") == "failed"
        or row.get("right_materialization_status") == "failed"
        for row in metric_rows
    )
    summary = {
        "manifest_path": str(manifest_path),
        "output_dir": str(output_root),
        "review_path": str(review_path),
        "settings": {
            "model_id": clip_encoder.model_id,
            "pair_count_requested": pair_count,
            "max_groups": max_groups,
            "min_group_size": min_group_size,
            "k_values": grid,
            "duration_seconds": duration_seconds,
            "sample_interval_seconds": sample_interval_seconds,
            "pruning_half_width_seconds": sample_interval_seconds / 2.0,
            "start_seconds": start_seconds,
            "high_similarity_threshold": high_similarity_threshold,
            "preserve_shared_anchor_seconds": preserve_shared_anchor_seconds,
            "min_pruned_video_seconds": min_pruned_video_seconds,
            "pruning_protection_mode": pruning_protection_mode,
            "min_pruned_video_percent": min_pruned_video_percent,
            "random_seed": random_seed,
            "download_media": download_media,
        },
        "group_count_available": len(groups),
        "pair_count": len(pair_rows),
        "pair_count_requested": pair_count,
        "target_met": len(pair_rows) >= pair_count,
        "variant_count": len(metric_rows),
        "cluster_assignment_count": len(assignment_rows),
        "trigger_pair_count": len(trigger_pair_rows),
        "centroid_frame_count": len(centroid_frame_rows),
        "materialization_failure_count": materialization_failure_count,
        "skipped_group_count": len(skipped),
        "skipped_groups": skipped,
        "k_aggregates": aggregates,
    }
    write_json(output_root / "summary.json", summary)
    if not summary["target_met"]:
        raise RuntimeError(
            f"requested {pair_count} synchronized pairs but produced {len(pair_rows)}; "
            f"inspect {output_root / 'summary.json'} for skipped-group errors"
        )
    if materialization_failure_count:
        raise RuntimeError(
            f"{materialization_failure_count} K variants had an ffmpeg materialization failure; "
            f"inspect {output_root / 'grid_metrics.csv'} and per-K pruning.json traces"
        )
    return summary
