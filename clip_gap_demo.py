"""Toy CLIP preprocessing for shared anchors and cross-user evidence gaps.

This module consumes the evidence JSONL already produced by ``prepare_evidence``.
It optionally resamples a short prefix of each local video, groups redundant
frames within each user, and compares the representative CLIP embeddings across
users.

The output is retrieval evidence, not a semantic claim that an object or event
is truly absent from another user's experience.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any, Protocol

from .io_utils import iter_jsonl, write_json


DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"


class ImageEncoder(Protocol):
    model_id: str

    def encode(self, image_paths: list[str]) -> list[list[float]]:
        ...


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0:
        return [0.0 for _ in vector]
    return [value / norm for value in vector]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("embedding dimensions must match")
    a_norm = _normalize(a)
    b_norm = _normalize(b)
    return sum(x * y for x, y in zip(a_norm, b_norm))


def pairwise_similarity(
    left_embeddings: list[list[float]],
    right_embeddings: list[list[float]],
) -> list[list[float]]:
    return [
        [round(cosine_similarity(left, right), 6) for right in right_embeddings]
        for left in left_embeddings
    ]


def _mean(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        raise ValueError("cannot average an empty vector list")
    return [
        sum(vector[index] for vector in vectors) / len(vectors)
        for index in range(len(vectors[0]))
    ]


def cluster_embedding_medoids(
    embeddings: list[list[float]],
    cluster_count: int,
    *,
    max_iterations: int = 25,
) -> tuple[list[int], list[int]]:
    """Return deterministic cosine-k-means labels and medoid indices."""

    if not embeddings:
        return [], []
    vectors = [_normalize(vector) for vector in embeddings]
    k = max(1, min(cluster_count, len(vectors)))

    centers = [vectors[0]]
    center_indices = [0]
    while len(centers) < k:
        next_index = max(
            (index for index in range(len(vectors)) if index not in center_indices),
            key=lambda index: min(1.0 - cosine_similarity(vectors[index], center) for center in centers),
        )
        center_indices.append(next_index)
        centers.append(vectors[next_index])

    labels = [0 for _ in vectors]
    for _ in range(max_iterations):
        new_labels = [
            max(range(k), key=lambda cluster: cosine_similarity(vector, centers[cluster]))
            for vector in vectors
        ]
        # Exact duplicate centers can otherwise lose a cluster through tie-breaking.
        for cluster, center_index in enumerate(center_indices):
            new_labels[center_index] = cluster
        new_centers = []
        for cluster in range(k):
            members = [vectors[index] for index, label in enumerate(new_labels) if label == cluster]
            new_centers.append(_normalize(_mean(members)) if members else centers[cluster])
        if new_labels == labels:
            labels = new_labels
            centers = new_centers
            break
        labels = new_labels
        centers = new_centers

    medoids = []
    for cluster in range(k):
        members = [index for index, label in enumerate(labels) if label == cluster]
        medoids.append(
            max(members, key=lambda index: cosine_similarity(vectors[index], centers[cluster]))
        )
    return labels, medoids


def mine_anchors_and_gaps(
    left_frames: list[dict[str, Any]],
    right_frames: list[dict[str, Any]],
    left_embeddings: list[list[float]],
    right_embeddings: list[list[float]],
    *,
    anchor_threshold: float = 0.75,
    top_k: int = 3,
) -> dict[str, Any]:
    if not left_frames or not right_frames:
        raise ValueError("both users need at least one representative frame")
    matrix = pairwise_similarity(left_embeddings, right_embeddings)

    left_best = [max(range(len(right_frames)), key=lambda j: matrix[i][j]) for i in range(len(left_frames))]
    right_best = [max(range(len(left_frames)), key=lambda i: matrix[i][j]) for j in range(len(right_frames))]

    anchors = []
    for left_index, right_index in enumerate(left_best):
        similarity = matrix[left_index][right_index]
        if right_best[right_index] == left_index and similarity >= anchor_threshold:
            anchors.append(
                {
                    "left_index": left_index,
                    "right_index": right_index,
                    "similarity": similarity,
                    "left_frame": left_frames[left_index],
                    "right_frame": right_frames[right_index],
                }
            )
    anchors.sort(key=lambda row: row["similarity"], reverse=True)
    anchor_left_indices = {int(row["left_index"]) for row in anchors}
    anchor_right_indices = {int(row["right_index"]) for row in anchors}

    left_novelty_ranked = []
    for left_index, right_index in enumerate(left_best):
        similarity = matrix[left_index][right_index]
        left_novelty_ranked.append(
            {
                "left_index": left_index,
                "closest_right_index": right_index,
                "closest_similarity": similarity,
                "novelty": round(1.0 - similarity, 6),
                "is_anchor": left_index in anchor_left_indices,
                "left_frame": left_frames[left_index],
                "closest_right_frame": right_frames[right_index],
            }
        )
    left_novelty_ranked.sort(key=lambda row: row["novelty"], reverse=True)
    left_gaps = [row for row in left_novelty_ranked if not row["is_anchor"]]

    right_novelty_ranked = []
    for right_index, left_index in enumerate(right_best):
        similarity = matrix[left_index][right_index]
        right_novelty_ranked.append(
            {
                "right_index": right_index,
                "closest_left_index": left_index,
                "closest_similarity": similarity,
                "novelty": round(1.0 - similarity, 6),
                "is_anchor": right_index in anchor_right_indices,
                "right_frame": right_frames[right_index],
                "closest_left_frame": left_frames[left_index],
            }
        )
    right_novelty_ranked.sort(key=lambda row: row["novelty"], reverse=True)
    right_gaps = [row for row in right_novelty_ranked if not row["is_anchor"]]

    return {
        "similarity_matrix": matrix,
        "anchors": anchors[:top_k],
        "left_novelty_ranked": left_novelty_ranked,
        "right_novelty_ranked": right_novelty_ranked,
        "left_evidence_gaps": left_gaps[:top_k],
        "right_evidence_gaps": right_gaps[:top_k],
        "interpretation_warning": (
            "A high novelty score means no close visual match was found in the other user's sampled "
            "representatives. Anchor representatives are excluded from evidence_gaps. Novelty does "
            "not prove that the other user never saw the event or object."
        ),
    }


class TransformersClipEncoder:
    def __init__(self, model_id: str = DEFAULT_CLIP_MODEL, *, device: str = "auto") -> None:
        try:
            import torch
            from PIL import Image
            from transformers import CLIPModel, CLIPProcessor
        except Exception as exc:
            raise RuntimeError(
                "CLIP encoding requires a compatible torch/torchvision/transformers installation. "
                "Install matching torch and torchvision builds, then transformers>=4.57."
            ) from exc
        self.torch = torch
        self.model_id = model_id
        self.image_class = Image
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()

    def _coerce_image_features(self, features: Any) -> Any:
        if hasattr(features, "norm"):
            return features
        if hasattr(features, "image_embeds"):
            return features.image_embeds
        if isinstance(features, (tuple, list)) and features:
            return self._coerce_image_features(features[0])
        if hasattr(features, "pooler_output"):
            pooled = features.pooler_output
            projection = getattr(self.model, "visual_projection", None)
            last_dim = None
            if hasattr(pooled, "shape") and len(pooled.shape) > 0:
                last_dim = int(pooled.shape[-1])
            if projection is not None:
                in_features = getattr(projection, "in_features", None)
                out_features = getattr(projection, "out_features", None)
                if in_features is None or last_dim == int(in_features):
                    return projection(pooled)
                if out_features is not None and last_dim == int(out_features):
                    return pooled
                raise RuntimeError(
                    "CLIP image encoder pooler_output dimension "
                    f"{last_dim} is incompatible with visual_projection "
                    f"{in_features}->{out_features}"
                )
            return pooled
        return features

    def encode(self, image_paths: list[str]) -> list[list[float]]:
        images = [self.image_class.open(path).convert("RGB") for path in image_paths]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            features = self._coerce_image_features(self.model.get_image_features(**inputs))
            if not hasattr(features, "norm"):
                raise RuntimeError(
                    f"CLIP image encoder returned unsupported output type: {type(features).__name__}"
                )
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().tolist()


def load_evidence_packet(path: str | Path, packet_index: int = 0) -> dict[str, Any]:
    for index, packet in enumerate(iter_jsonl(path)):
        if index == packet_index:
            return packet
    raise IndexError(f"evidence packet index {packet_index} not found in {path}")


def sample_short_video(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    duration_seconds: float = 12.0,
    sample_interval_seconds: float = 1.5,
    start_seconds: float = 0.0,
    ffmpeg_binary: str = "ffmpeg",
) -> list[dict[str, Any]]:
    ffmpeg = shutil.which(ffmpeg_binary)
    if not ffmpeg:
        explicit = Path(ffmpeg_binary)
        if explicit.exists():
            ffmpeg = str(explicit)
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to resample short video windows")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    count = max(2, int(math.floor(duration_seconds / sample_interval_seconds)) + 1)
    frames = []
    for index in range(count):
        timestamp = start_seconds + index * sample_interval_seconds
        if timestamp > start_seconds + duration_seconds + 1e-9:
            break
        path = output_dir / f"frame_{index:03d}_{timestamp:.2f}s.png"
        if not path.exists() or path.stat().st_size == 0:
            subprocess.run(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-ss",
                    f"{timestamp:.3f}",
                    "-i",
                    str(video_path),
                    "-frames:v",
                    "1",
                    "-vf",
                    "format=rgb24",
                    "-f",
                    "image2",
                    str(path),
                ],
                check=True,
            )
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"ffmpeg did not write sampled frame: {path}")
        frames.append({"timestamp_seconds": round(timestamp, 3), "path": str(path)})
    return frames


def packet_frames(
    packet: dict[str, Any],
    output_dir: str | Path,
    *,
    duration_seconds: float,
    sample_interval_seconds: float,
    start_seconds: float,
    ffmpeg_binary: str,
    resample_videos: bool,
) -> list[dict[str, Any]]:
    clips = packet.get("clips", [])
    if len(clips) != 2:
        raise ValueError("the toy demo currently expects exactly two clips/users")
    rows = []
    for clip in clips:
        user = str(clip.get("agent_name") or clip.get("agent_dir"))
        local_video = clip.get("local_video")
        if resample_videos:
            if not local_video or not Path(local_video).exists():
                raise FileNotFoundError(f"local video is unavailable for {user}: {local_video}")
            frames = sample_short_video(
                local_video,
                Path(output_dir) / "sampled_frames" / user,
                duration_seconds=duration_seconds,
                sample_interval_seconds=sample_interval_seconds,
                start_seconds=start_seconds,
                ffmpeg_binary=ffmpeg_binary,
            )
        else:
            frames = [
                frame
                for frame in clip.get("frames", [])
                if Path(str(frame.get("path", ""))).exists()
                and start_seconds <= float(frame.get("timestamp_seconds", 0.0)) <= start_seconds + duration_seconds
            ]
            if not frames:
                frames = [
                    frame
                    for frame in clip.get("frames", [])
                    if Path(str(frame.get("path", ""))).exists()
                ]
        if not frames:
            raise ValueError(f"no usable frames found for {user}")
        rows.append({"user": user, "clip": clip, "frames": frames})
    return rows


def _cluster_user_frames(
    frames: list[dict[str, Any]],
    embeddings: list[list[float]],
    cluster_count: int,
) -> tuple[list[dict[str, Any]], list[list[float]], list[dict[str, Any]]]:
    labels, medoids = cluster_embedding_medoids(embeddings, cluster_count)
    representatives = []
    representative_embeddings = []
    groups = []
    for cluster, medoid in enumerate(medoids):
        members = [index for index, label in enumerate(labels) if label == cluster]
        representative = dict(frames[medoid])
        representative["source_index"] = medoid
        representative["cluster_id"] = cluster
        representatives.append(representative)
        representative_embeddings.append(embeddings[medoid])
        groups.append(
            {
                "cluster_id": cluster,
                "representative_index": medoid,
                "member_indices": members,
                "member_timestamps": [frames[index].get("timestamp_seconds") for index in members],
            }
        )
    representatives.sort(key=lambda row: float(row.get("timestamp_seconds", 0.0)))
    order = [int(row["source_index"]) for row in representatives]
    representative_embeddings = [embeddings[index] for index in order]
    return representatives, representative_embeddings, groups


def write_contact_sheet(
    result: dict[str, Any],
    output_path: str | Path,
    *,
    thumb_size: tuple[int, int] = (280, 180),
) -> None:
    try:
        from PIL import Image, ImageDraw, ImageOps
    except ImportError as exc:
        raise RuntimeError("Pillow is required to write the CLIP contact sheet") from exc
    rows = []
    for label, candidates in [
        ("SHARED ANCHORS", result.get("anchors", [])),
        (f"{result['left_user']} GAPS", result.get("left_evidence_gaps", [])),
        (f"{result['right_user']} GAPS", result.get("right_evidence_gaps", [])),
    ]:
        rows.append((label, candidates))
    width = thumb_size[0] * 2 + 60
    row_height = thumb_size[1] + 55
    height = 45 + sum(max(1, len(items)) * row_height + 35 for _, items in rows)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    y = 15
    for label, items in rows:
        draw.text((15, y), label, fill="black")
        y += 30
        if not items:
            draw.text((25, y), "none above threshold", fill="gray")
            y += row_height
            continue
        for item in items:
            if label == "SHARED ANCHORS":
                left = item["left_frame"]
                right = item["right_frame"]
                score = f"similarity={item['similarity']:.3f}"
            elif label.endswith(" GAPS") and label.startswith(result["left_user"]):
                left = item["left_frame"]
                right = item["closest_right_frame"]
                score = f"novelty={item['novelty']:.3f}"
            else:
                left = item["closest_left_frame"]
                right = item["right_frame"]
                score = f"novelty={item['novelty']:.3f}"
            for column, frame in enumerate([left, right]):
                image = Image.open(frame["path"]).convert("RGB")
                image = ImageOps.contain(image, thumb_size)
                x = 15 + column * (thumb_size[0] + 30)
                sheet.paste(image, (x, y))
                caption = f"{frame.get('timestamp_seconds', '?')}s"
                draw.text((x, y + thumb_size[1] + 4), caption, fill="black")
            draw.text((15, y + thumb_size[1] + 22), score, fill="navy")
            y += row_height
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.crop((0, 0, width, min(height, y + 10))).save(output_path)


def run_clip_gap_demo(
    *,
    evidence_path: str | Path,
    output_dir: str | Path,
    packet_index: int = 0,
    model_id: str = DEFAULT_CLIP_MODEL,
    duration_seconds: float = 12.0,
    sample_interval_seconds: float = 1.5,
    start_seconds: float = 0.0,
    clusters_per_user: int = 4,
    anchor_threshold: float = 0.75,
    top_k: int = 3,
    ffmpeg_binary: str = "ffmpeg",
    resample_videos: bool = True,
    encoder: ImageEncoder | None = None,
) -> dict[str, Any]:
    packet = load_evidence_packet(evidence_path, packet_index)
    output_dir = Path(output_dir)
    users = packet_frames(
        packet,
        output_dir,
        duration_seconds=duration_seconds,
        sample_interval_seconds=sample_interval_seconds,
        start_seconds=start_seconds,
        ffmpeg_binary=ffmpeg_binary,
        resample_videos=resample_videos,
    )
    encoder = encoder or TransformersClipEncoder(model_id)
    encoded_users = []
    for row in users:
        embeddings = encoder.encode([str(frame["path"]) for frame in row["frames"]])
        representatives, representative_embeddings, groups = _cluster_user_frames(
            row["frames"], embeddings, clusters_per_user
        )
        encoded_users.append(
            {
                **row,
                "representatives": representatives,
                "representative_embeddings": representative_embeddings,
                "groups": groups,
            }
        )

    left, right = encoded_users
    mined = mine_anchors_and_gaps(
        left["representatives"],
        right["representatives"],
        left["representative_embeddings"],
        right["representative_embeddings"],
        anchor_threshold=anchor_threshold,
        top_k=top_k,
    )
    result = {
        "evidence_id": packet.get("evidence_id"),
        "day": packet.get("day"),
        "time_token": packet.get("time_token"),
        "model_id": encoder.model_id,
        "left_user": left["user"],
        "right_user": right["user"],
        "window": {
            "start_seconds": start_seconds,
            "duration_seconds": duration_seconds,
            "sample_interval_seconds": sample_interval_seconds,
        },
        "clustering": {
            "clusters_per_user": clusters_per_user,
            left["user"]: left["groups"],
            right["user"]: right["groups"],
        },
        "representative_frames": {
            left["user"]: left["representatives"],
            right["user"]: right["representatives"],
        },
        **mined,
    }
    write_json(output_dir / "clip_gap_results.json", result)
    write_contact_sheet(result, output_dir / "clip_gap_contact_sheet.jpg")
    return result


def random_window_starts(
    *,
    max_start_seconds: float,
    trial_count: int,
    sample_interval_seconds: float,
    seed: int,
) -> list[float]:
    """Select reproducible starts across the full available time range.

    The candidate grid is divided into ``trial_count`` temporal strata and one
    start is sampled from each. This retains randomness while avoiding five
    trials that accidentally cluster near the beginning of a short clip.
    """

    if trial_count <= 0:
        raise ValueError("trial_count must be positive")
    if sample_interval_seconds <= 0:
        raise ValueError("sample_interval_seconds must be positive")
    if max_start_seconds < 0:
        raise ValueError("duration_seconds exceeds the available video duration")
    step_count = int(math.floor(max_start_seconds / sample_interval_seconds))
    candidates = [round(index * sample_interval_seconds, 3) for index in range(step_count + 1)]
    candidate_count = len(candidates)
    if candidate_count < trial_count:
        raise ValueError(
            f"only {candidate_count} distinct window starts are available, but {trial_count} were requested"
        )
    rng = random.Random(seed)
    starts = []
    for stratum in range(trial_count):
        start_index = math.floor(stratum * candidate_count / trial_count)
        end_index = math.floor((stratum + 1) * candidate_count / trial_count)
        starts.append(rng.choice(candidates[start_index:end_index]))
    return sorted(starts)


def summarize_trial(result: dict[str, Any], trial_index: int) -> dict[str, Any]:
    anchors = result.get("anchors", [])
    left_gaps = result.get("left_evidence_gaps", [])
    right_gaps = result.get("right_evidence_gaps", [])
    anchor_scores = [float(row["similarity"]) for row in anchors]
    left_novelties = [float(row["novelty"]) for row in left_gaps]
    right_novelties = [float(row["novelty"]) for row in right_gaps]
    matrix_values = [
        float(value)
        for row in result.get("similarity_matrix", [])
        for value in row
    ]
    largest_novelty = max(left_novelties + right_novelties, default=None)
    max_anchor = max(anchor_scores, default=None)
    return {
        "trial": trial_index,
        "start_seconds": result.get("window", {}).get("start_seconds"),
        "end_seconds": round(
            float(result.get("window", {}).get("start_seconds", 0.0))
            + float(result.get("window", {}).get("duration_seconds", 0.0)),
            3,
        ),
        "anchor_count": len(anchors),
        "mean_cross_user_similarity": (
            round(sum(matrix_values) / len(matrix_values), 6) if matrix_values else None
        ),
        "minimum_cross_user_similarity": (
            round(min(matrix_values), 6) if matrix_values else None
        ),
        "max_anchor_similarity": round(max_anchor, 6) if max_anchor is not None else None,
        "mean_anchor_similarity": (
            round(sum(anchor_scores) / len(anchor_scores), 6) if anchor_scores else None
        ),
        f"max_{result['left_user']}_novelty": (
            round(max(left_novelties), 6) if left_novelties else None
        ),
        f"max_{result['right_user']}_novelty": (
            round(max(right_novelties), 6) if right_novelties else None
        ),
        "largest_user_novelty": (
            round(largest_novelty, 6) if largest_novelty is not None else None
        ),
        "review_priority": (
            round(largest_novelty * max_anchor, 6)
            if largest_novelty is not None and max_anchor is not None
            else 0.0
        ),
    }


def run_random_clip_gap_trials(
    *,
    evidence_path: str | Path,
    output_dir: str | Path,
    packet_index: int = 0,
    model_id: str = DEFAULT_CLIP_MODEL,
    trial_count: int = 5,
    random_seed: int = 42,
    duration_seconds: float = 12.0,
    sample_interval_seconds: float = 1.5,
    clusters_per_user: int = 4,
    anchor_threshold: float = 0.75,
    top_k: int = 3,
    ffmpeg_binary: str = "ffmpeg",
    resample_videos: bool = True,
    encoder: ImageEncoder | None = None,
) -> dict[str, Any]:
    packet = load_evidence_packet(evidence_path, packet_index)
    clips = packet.get("clips", [])
    durations = [
        float(clip["duration_seconds"])
        for clip in clips
        if clip.get("duration_seconds") is not None
    ]
    if len(durations) != len(clips) or not durations:
        raise ValueError("all clips need duration_seconds for random-window trials")
    available_duration = min(durations)
    starts = random_window_starts(
        max_start_seconds=available_duration - duration_seconds,
        trial_count=trial_count,
        sample_interval_seconds=sample_interval_seconds,
        seed=random_seed,
    )
    encoder = encoder or TransformersClipEncoder(model_id)
    output_dir = Path(output_dir)
    trials = []
    summaries = []
    for trial_index, start_seconds in enumerate(starts, 1):
        trial_dir = output_dir / f"trial_{trial_index:02d}_{start_seconds:.2f}s"
        result = run_clip_gap_demo(
            evidence_path=evidence_path,
            output_dir=trial_dir,
            packet_index=packet_index,
            model_id=model_id,
            duration_seconds=duration_seconds,
            sample_interval_seconds=sample_interval_seconds,
            start_seconds=start_seconds,
            clusters_per_user=clusters_per_user,
            anchor_threshold=anchor_threshold,
            top_k=top_k,
            ffmpeg_binary=ffmpeg_binary,
            resample_videos=resample_videos,
            encoder=encoder,
        )
        result["trial"] = trial_index
        result["output_dir"] = str(trial_dir)
        trials.append(result)
        summaries.append(summarize_trial(result, trial_index))

    aggregate = {
        "evidence_id": packet.get("evidence_id"),
        "model_id": encoder.model_id,
        "trial_count": trial_count,
        "random_seed": random_seed,
        "available_duration_seconds": available_duration,
        "window_duration_seconds": duration_seconds,
        "sample_interval_seconds": sample_interval_seconds,
        "window_starts_seconds": starts,
        "trial_summaries": summaries,
        "trials": trials,
    }
    write_json(output_dir / "random_trials_summary.json", aggregate)
    return aggregate


def run_diverse_packet_trials(
    *,
    evidence_path: str | Path,
    output_dir: str | Path,
    model_id: str = DEFAULT_CLIP_MODEL,
    trial_count: int = 5,
    random_seed: int = 42,
    duration_seconds: float = 12.0,
    sample_interval_seconds: float = 1.5,
    clusters_per_user: int = 4,
    anchor_threshold: float = 0.75,
    top_k: int = 3,
    ffmpeg_binary: str = "ffmpeg",
    resample_videos: bool = True,
    encoder: ImageEncoder | None = None,
) -> dict[str, Any]:
    """Run one random short window from each of several evidence packets."""

    packets = list(iter_jsonl(evidence_path))
    if len(packets) < trial_count:
        raise ValueError(
            f"{evidence_path} contains {len(packets)} packets, but {trial_count} diverse trials were requested"
        )
    rng = random.Random(random_seed)
    encoder = encoder or TransformersClipEncoder(model_id)
    output_dir = Path(output_dir)
    trials = []
    summaries = []
    for trial_index, packet in enumerate(packets[:trial_count], 1):
        clips = packet.get("clips", [])
        durations = [
            float(clip["duration_seconds"])
            for clip in clips
            if clip.get("duration_seconds") is not None
        ]
        if len(durations) != len(clips) or not durations:
            raise ValueError(f"packet {trial_index} is missing clip durations")
        available_duration = min(durations)
        max_start = available_duration - duration_seconds
        starts = random_window_starts(
            max_start_seconds=max_start,
            trial_count=1,
            sample_interval_seconds=sample_interval_seconds,
            seed=rng.randrange(0, 2**31),
        )
        start_seconds = starts[0]
        users = "-".join(str(user) for user in packet.get("required_users", []))
        trial_dir = output_dir / (
            f"trial_{trial_index:02d}_{packet.get('day', 'DAY')}_"
            f"{packet.get('time_token', 'time')}_{users}"
        )
        result = run_clip_gap_demo(
            evidence_path=evidence_path,
            output_dir=trial_dir,
            packet_index=trial_index - 1,
            model_id=model_id,
            duration_seconds=duration_seconds,
            sample_interval_seconds=sample_interval_seconds,
            start_seconds=start_seconds,
            clusters_per_user=clusters_per_user,
            anchor_threshold=anchor_threshold,
            top_k=top_k,
            ffmpeg_binary=ffmpeg_binary,
            resample_videos=resample_videos,
            encoder=encoder,
        )
        result["trial"] = trial_index
        result["output_dir"] = str(trial_dir)
        trials.append(result)
        summary = summarize_trial(result, trial_index)
        summary.update(
            {
                "evidence_id": packet.get("evidence_id"),
                "day": packet.get("day"),
                "time_token": packet.get("time_token"),
                "users": ", ".join(packet.get("required_users", [])),
            }
        )
        summaries.append(summary)

    aggregate = {
        "model_id": encoder.model_id,
        "trial_count": trial_count,
        "random_seed": random_seed,
        "window_duration_seconds": duration_seconds,
        "sample_interval_seconds": sample_interval_seconds,
        "trial_summaries": summaries,
        "trials": trials,
    }
    write_json(output_dir / "diverse_packet_trials_summary.json", aggregate)
    return aggregate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Find CLIP shared anchors and cross-user evidence gaps")
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--packet-index", type=int, default=0)
    parser.add_argument("--model-id", default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--duration-seconds", type=float, default=12.0)
    parser.add_argument("--sample-interval-seconds", type=float, default=1.5)
    parser.add_argument("--clusters-per-user", type=int, default=4)
    parser.add_argument("--anchor-threshold", type=float, default=0.75)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--ffmpeg-binary", default="ffmpeg")
    parser.add_argument(
        "--use-existing-frames",
        action="store_true",
        help="Use frame paths already stored in the evidence packet instead of resampling local videos",
    )
    parser.add_argument("--random-trials", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--diverse-packet-trials", type=int, default=0)
    args = parser.parse_args(argv)
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
        print(
            f"wrote {result['trial_count']} diverse-packet CLIP trials to "
            f"{Path(args.output_dir) / 'diverse_packet_trials_summary.json'}"
        )
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
        print(
            f"wrote {result['trial_count']} random CLIP gap trials to "
            f"{Path(args.output_dir) / 'random_trials_summary.json'}"
        )
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
        f"to {Path(args.output_dir) / 'clip_gap_results.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
