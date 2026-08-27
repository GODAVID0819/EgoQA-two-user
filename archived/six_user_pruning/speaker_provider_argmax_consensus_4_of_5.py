"""历史归档：早期 speaker-provider argmax 共识剪枝。

该文件仅供追溯，不得被生产流程导入。当时实验使用 4-of-5；
当前方法已改为 provider-only all-pairs。
"""

from typing import Any

from egolife_two_user_qa.group_relative_clip_sampling import (
    _intervals_for_frame_indices,
    _subtract_intervals,
    clustered_frame_representatives,
    frame_similarity_matrix,
)


def clustered_speaker_consensus_pruning(
    frames_by_video: list[list[dict[str, Any]]],
    embeddings_by_video: list[list[list[float]]],
    *,
    speaker_index: int,
    start_seconds: float,
    duration_seconds: float,
    sample_interval_seconds: float,
    cluster_count: int = 12,
    high_similarity_threshold: float = 0.82,
    min_high_provider_matches: int = 3,
    min_pruned_video_seconds: float = 8.0,
) -> dict[str, Any]:
    """按 speaker cluster 对五个 provider 的 argmax 共识生成裁剪区间。"""

    if len(frames_by_video) != 6 or len(embeddings_by_video) != 6:
        raise ValueError("speaker consensus pruning requires exactly 6 videos")
    if speaker_index < 0 or speaker_index >= 6:
        raise ValueError(f"speaker_index must be between 0 and 5, got {speaker_index}")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if sample_interval_seconds <= 0:
        raise ValueError("sample_interval_seconds must be positive")
    if min_high_provider_matches < 1 or min_high_provider_matches > 5:
        raise ValueError("min_high_provider_matches must be between 1 and 5")

    clusters_by_video = [
        clustered_frame_representatives(frames, embeddings, cluster_count=cluster_count)
        for frames, embeddings in zip(frames_by_video, embeddings_by_video)
    ]
    provider_indices = [index for index in range(6) if index != speaker_index]
    speaker_clusters = clusters_by_video[speaker_index]
    matrices = {
        provider_index: frame_similarity_matrix(
            speaker_clusters["representative_embeddings"],
            clusters_by_video[provider_index]["representative_embeddings"],
        )
        for provider_index in provider_indices
    }
    events: list[dict[str, Any]] = []
    marked_clusters = [set() for _ in range(6)]
    trigger_events = [set() for _ in range(6)]
    for speaker_cluster_index in range(int(speaker_clusters["cluster_count"])):
        provider_matches = []
        for provider_index in provider_indices:
            provider_cluster_index, similarity = max(
                enumerate(matrices[provider_index][speaker_cluster_index]),
                key=lambda item: float(item[1]),
            )
            provider_matches.append(
                {
                    "provider_index": provider_index,
                    "provider_cluster_index": int(provider_cluster_index),
                    "similarity": round(float(similarity), 6),
                    "meets_threshold": float(similarity) >= float(high_similarity_threshold),
                }
            )
        high_matches = [match for match in provider_matches if match["meets_threshold"]]
        if len(high_matches) < min_high_provider_matches:
            continue
        event_index = len(events)
        deleted_clusters = [
            {"video_index": speaker_index, "cluster_index": speaker_cluster_index}
        ] + [
            {
                "video_index": int(match["provider_index"]),
                "cluster_index": int(match["provider_cluster_index"]),
            }
            for match in high_matches
        ]
        events.append(
            {
                "event_index": event_index,
                "speaker_cluster_index": speaker_cluster_index,
                "provider_matches": provider_matches,
                "high_provider_count": len(high_matches),
                "deleted_clusters": deleted_clusters,
            }
        )
        for deleted in deleted_clusters:
            video_index = int(deleted["video_index"])
            marked_clusters[video_index].add(int(deleted["cluster_index"]))
            trigger_events[video_index].add(event_index)

    window_start = float(start_seconds)
    window_end = round(window_start + float(duration_seconds), 3)
    video_results = []
    for video_index, (frames, clusters) in enumerate(zip(frames_by_video, clusters_by_video)):
        marked_frame_indices: set[int] = set()
        for marked_cluster_index in marked_clusters[video_index]:
            marked_frame_indices.update(
                int(index)
                for index in clusters["representatives"][marked_cluster_index].get(
                    "member_indices", []
                )
            )
        remove_intervals = _intervals_for_frame_indices(
            frames,
            marked_frame_indices,
            window_start=window_start,
            window_end=window_end,
            sample_interval_seconds=sample_interval_seconds,
        )
        keep_intervals = _subtract_intervals((window_start, window_end), remove_intervals)
        kept_duration = round(sum(end - start for start, end in keep_intervals), 3)
        removed_duration = round(sum(end - start for start, end in remove_intervals), 3)
        video_results.append(
            {
                "video_index": video_index,
                "cluster_count": int(clusters["cluster_count"]),
                "clusters": clusters["representatives"],
                "marked_cluster_indices": sorted(marked_clusters[video_index]),
                "marked_frame_indices": sorted(marked_frame_indices),
                "trigger_event_indices": sorted(trigger_events[video_index]),
                "remove_intervals": remove_intervals,
                "keep_intervals": keep_intervals,
                "kept_duration_seconds": kept_duration,
                "removed_duration_seconds": removed_duration,
                "passed": kept_duration >= float(min_pruned_video_seconds),
            }
        )

    return {
        "method": "speaker_cluster_provider_argmax_consensus",
        "speaker_index": speaker_index,
        "provider_indices": provider_indices,
        "cluster_count": cluster_count,
        "high_similarity_threshold": high_similarity_threshold,
        "min_high_provider_matches": min_high_provider_matches,
        "events": events,
        "videos": video_results,
        "passed": bool(events) and all(video["passed"] for video in video_results),
    }
