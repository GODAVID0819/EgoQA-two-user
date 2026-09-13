"""Object detection hints for video-first QA generation.

This module follows the EgoEverything object-first prepass shape: sample
keyframes, ask a VLM for normalized object boxes, keep both Qwen/Gemini-format and
pixel-format boxes, then select a small set of key objects that the QA generator
can use as attention anchors.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import urllib.request
from typing import Any

from .evidence import extract_frames, ffprobe_duration
from .gaze_projection import gaussian_bbox_score
from .io_utils import iter_jsonl, write_jsonl
from .object_reid import DEFAULT_REID_MODEL_ID, EgoEverythingClipReidentifier, reidentify_objects
from .qwen3vl_runner import DEFAULT_MODEL_ID, image_to_data_url, make_runner


DEFAULT_OBJECT_DETECTION_MODEL = DEFAULT_MODEL_ID
DEFAULT_OPENROUTER_OBJECT_DETECTION_MODEL = "google/gemini-2.5-flash"
DEFAULT_LOCAL_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_REID_VISUAL_THRESHOLD = 0.8
DEFAULT_REID_TEXT_THRESHOLD = 0.85


DETECTION_PROMPT = """Please analyze this image and identify all clearly visible objects with their locations.

IMPORTANT REQUIREMENTS:
1. Focus on objects with distinct, recognizable features
2. Exclude background elements like walls, floors, ceilings, or lighting unless they are specific objects
3. Only include objects that are clearly visible and well-defined
4. Provide accurate bounding box coordinates for each object

For each detected object, provide the results in this exact format:
[ymin,xmin,ymax,xmax]:<object_name>

Where:
- ymin, xmin, ymax, xmax are coordinates in range 0-1000 (normalized coordinates)
- ymin, xmin: top-left corner coordinates
- ymax, xmax: bottom-right corner coordinates
- object_name: clear, specific name of the object

Example format:
[100,200,300,400]:coffee_mug
[50,150,250,350]:laptop
[200,300,400,500]:chair

Please analyze the image carefully and provide all clearly visible objects with their bounding boxes."""


def resolve_existing_path(value: str | Path | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.exists():
        return path
    package_parent_candidate = Path(__file__).resolve().parent.parent / path
    if package_parent_candidate.exists():
        return package_parent_candidate
    return None


def _clean_object_name(name: str) -> str:
    name = re.sub(r"^[,\s]*\"?label\"?\s*:\s*\"?", "", str(name))
    name = re.sub(r"\"?\s*[,}]\s*$", "", name)
    return name.strip(" \t\r\n\",}{:")


def normalize_object_name(name: str) -> str:
    text = _clean_object_name(name).lower().replace("-", " ").replace("_", " ")
    text = re.sub(r"\b(the|a|an|one|visible|small|large|nearby)\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def image_dimensions(path: str | Path | None) -> tuple[int, int]:
    """Return image width and height, using optional imaging deps when present."""

    if path:
        image_path = resolve_existing_path(path)
        if image_path is not None:
            try:
                from PIL import Image

                with Image.open(image_path) as img:
                    return int(img.size[0]), int(img.size[1])
            except Exception:
                pass
            try:
                import cv2

                img = cv2.imread(str(image_path))
                if img is not None:
                    height, width = img.shape[:2]
                    return int(width), int(height)
            except Exception:
                pass
    return 1000, 1000


def parse_detection_results(
    response_text: str,
    *,
    image_path: str | Path | None = None,
    image_width: int | None = None,
    image_height: int | None = None,
) -> list[dict[str, Any]]:
    """Parse EgoEverything/Gemini detection rows into structured objects."""

    width, height = image_dimensions(image_path)
    if image_width:
        width = int(image_width)
    if image_height:
        height = int(image_height)

    patterns = [
        r"\[(\d+),(\d+),(\d+),(\d+)\]:([^,\n\r]+)",
        r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]:\s*([^\n\r]+)",
        r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\s*([^\n\r\[]+)",
        r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\].*?\"label\"\s*:\s*\"([^\"]+)\"",
    ]
    matches: list[tuple[str, str, str, str, str]] = []
    for pattern in patterns:
        matches = re.findall(pattern, response_text)
        if matches:
            break

    objects: list[dict[str, Any]] = []
    for match in matches:
        ymin_raw, xmin_raw, ymax_raw, xmax_raw, name_raw = match
        try:
            ymin = int(ymin_raw)
            xmin = int(xmin_raw)
            ymax = int(ymax_raw)
            xmax = int(xmax_raw)
        except ValueError:
            continue
        object_name = _clean_object_name(name_raw)
        if not object_name:
            continue
        if not (0 <= ymin <= 1000 and 0 <= xmin <= 1000 and 0 <= ymax <= 1000 and 0 <= xmax <= 1000):
            continue
        if ymin >= ymax or xmin >= xmax:
            continue

        pixel_bbox = [
            int(xmin / 1000 * width),
            int(ymin / 1000 * height),
            int(xmax / 1000 * width),
            int(ymax / 1000 * height),
        ]
        objects.append(
            {
                "name": object_name,
                "object_name": object_name,
                "bbox": pixel_bbox,
                "normalized_bbox": [xmin, ymin, xmax, ymax],
                "gemini_bbox": [ymin, xmin, ymax, xmax],
                "image_width": width,
                "image_height": height,
                "normalized_name": normalize_object_name(object_name),
            }
        )
    return objects


class OpenRouterObjectDetector:
    """OpenRouter detector using the EgoEverything prompt and response format."""

    def __init__(
        self,
        *,
        api_key: str | None,
        model_id: str = DEFAULT_OPENROUTER_OBJECT_DETECTION_MODEL,
        base_url: str = DEFAULT_OPENROUTER_BASE_URL,
        timeout: int = 120,
    ) -> None:
        self.api_key = api_key
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def detect_text(self, image_path: str | Path) -> str:
        if not self.api_key:
            raise RuntimeError("object detection requires --api-key or OPENROUTER_API_KEY")
        payload = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": DETECTION_PROMPT},
                        {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
                    ],
                }
            ],
            "temperature": 1,
            "max_tokens": 4096,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "https://github.com/ElecBeholder/EgoEverything",
                "X-Title": "egolife-two-user-qa-object-hints",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return str(data["choices"][0]["message"].get("content") or "").strip()

    def detect_objects(self, frame: dict[str, Any]) -> list[dict[str, Any]]:
        response_text = self.detect_text(frame["path"])
        objects = parse_detection_results(
            response_text,
            image_path=frame["path"],
            image_width=frame.get("image_width"),
            image_height=frame.get("image_height"),
        )
        for obj in objects:
            obj["raw_detector_output"] = response_text
            obj["detector_model"] = self.model_id
            obj["detector_backend"] = "openrouter"
        return objects


class RunnerObjectDetector:
    """Detector backed by the repo's Qwen/local VLM runner stack."""

    def __init__(
        self,
        *,
        backend: str,
        model_id: str = DEFAULT_OBJECT_DETECTION_MODEL,
        base_url: str = "http://127.0.0.1:8000/v1",
        max_new_tokens: int = 1024,
        max_image_pixels: int = 262144,
        dtype: str = "bfloat16",
        allow_cpu: bool = False,
        disable_thinking: bool = False,
    ) -> None:
        self.backend = backend
        self.model_id = model_id
        self.runner = make_runner(
            backend,
            model_id=model_id,
            base_url=base_url,
            max_new_tokens=max_new_tokens,
            max_image_pixels=max_image_pixels,
            dtype=dtype,
            allow_cpu=allow_cpu,
            disable_thinking=disable_thinking,
        )

    def detect_objects(self, frame: dict[str, Any]) -> list[dict[str, Any]]:
        response_text = self.runner.generate(DETECTION_PROMPT, image_paths=[frame["path"]], video_paths=[])
        objects = parse_detection_results(
            response_text,
            image_path=frame["path"],
            image_width=frame.get("image_width"),
            image_height=frame.get("image_height"),
        )
        for obj in objects:
            obj["raw_detector_output"] = response_text
            obj["detector_model"] = self.model_id
            obj["detector_backend"] = self.backend
        return objects


class DryRunObjectDetector:
    """Deterministic no-model detector for prompt and packet plumbing tests."""

    model_id = "dry-run-object-detector"

    def detect_objects(self, frame: dict[str, Any]) -> list[dict[str, Any]]:
        width, height = image_dimensions(frame.get("path"))
        user = str(frame.get("user") or "user").lower()
        stem = Path(str(frame.get("path") or "frame")).stem.lower()
        seed = sum(ord(ch) for ch in f"{user}:{stem}")
        labels = ["mug", "phone", "notebook", "bottle", "remote"]
        label = labels[seed % len(labels)]
        ymin = 180 + (seed % 160)
        xmin = 160 + (seed % 220)
        ymax = min(940, ymin + 260)
        xmax = min(940, xmin + 300)
        return parse_detection_results(
            f"[{ymin},{xmin},{ymax},{xmax}]:{label}",
            image_path=frame.get("path"),
            image_width=width,
            image_height=height,
        )


def detector_from_args(
    *,
    backend: str,
    api_key: str | None,
    model_id: str,
    base_url: str,
    max_new_tokens: int = 1024,
    max_image_pixels: int = 262144,
    dtype: str = "bfloat16",
    allow_cpu: bool = False,
    disable_thinking: bool = False,
) -> OpenRouterObjectDetector | RunnerObjectDetector | DryRunObjectDetector:
    if backend == "dry-run":
        return DryRunObjectDetector()
    if backend == "openrouter":
        effective_model = (
            DEFAULT_OPENROUTER_OBJECT_DETECTION_MODEL
            if model_id == DEFAULT_OBJECT_DETECTION_MODEL
            else model_id
        )
        effective_base_url = (
            DEFAULT_OPENROUTER_BASE_URL
            if base_url == DEFAULT_LOCAL_BASE_URL
            else base_url
        )
        return OpenRouterObjectDetector(
            api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
            model_id=effective_model,
            base_url=effective_base_url,
        )
    if backend in {"transformers-local", "openai-compatible-local"}:
        return RunnerObjectDetector(
            backend=backend,
            model_id=model_id,
            base_url=base_url,
            max_new_tokens=max_new_tokens,
            max_image_pixels=max_image_pixels,
            dtype=dtype,
            allow_cpu=allow_cpu,
            disable_thinking=disable_thinking,
        )
    raise ValueError(f"unknown object detector backend: {backend}")


def _uniform_take(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0 or len(rows) <= count:
        return list(rows)
    if count == 1:
        return [rows[len(rows) // 2]]
    indices = [round(index * (len(rows) - 1) / (count - 1)) for index in range(count)]
    return [rows[index] for index in sorted(set(indices))]


def _frame_score(row: dict[str, Any]) -> float:
    for key in ("centroid_similarity", "representative_similarity", "similarity", "novelty", "member_count"):
        try:
            return float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
    return 0.0


def select_frame_rows(rows: list[dict[str, Any]], *, count: int, mode: str) -> list[dict[str, Any]]:
    if mode == "clip_medoids":
        representative_rows = [
            row
            for row in rows
            if any(key in row for key in ("cluster_id", "source_index", "centroid_similarity", "representative_similarity"))
        ]
        if representative_rows:
            ranked = sorted(
                representative_rows,
                key=lambda row: (-_frame_score(row), float(row.get("timestamp_seconds") or 0.0)),
            )
            return ranked[:count] if count > 0 else ranked
    return _uniform_take(rows, count)


def _original_to_pruned_timestamp(timestamp: float, keep_intervals: list[Any]) -> float | None:
    elapsed = 0.0
    for interval in keep_intervals:
        if not isinstance(interval, (list, tuple)) or len(interval) < 2:
            continue
        try:
            start = float(interval[0])
            end = float(interval[1])
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        if start <= timestamp <= end:
            return round(elapsed + max(0.0, timestamp - start), 3)
        elapsed += end - start
    return None


def _kept_cluster_representative_rows(clip: dict[str, Any]) -> list[dict[str, Any]]:
    pruning = clip.get("temporal_pruning")
    if not isinstance(pruning, dict):
        return []
    rows = pruning.get("kept_cluster_representatives")
    if not isinstance(rows, list):
        cluster_decisions = pruning.get("cluster_decisions")
        rows = [
            row
            for row in cluster_decisions
            if isinstance(row, dict) and row.get("status") == "kept"
        ] if isinstance(cluster_decisions, list) else []
    keep_intervals = pruning.get("keep_intervals") if isinstance(pruning.get("keep_intervals"), list) else []
    selected = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        timestamp = _timestamp_from_row(row)
        if timestamp is None:
            continue
        pruned_timestamp = _original_to_pruned_timestamp(timestamp, keep_intervals) if keep_intervals else timestamp
        if pruned_timestamp is None:
            continue
        out = dict(row)
        out["source_timestamp_seconds"] = round(float(timestamp), 3)
        out["pruned_timestamp_seconds"] = pruned_timestamp
        out["timestamp_seconds"] = pruned_timestamp
        out["source"] = "kept_pruning_cluster_representative"
        selected.append(out)
    return sorted(
        selected,
        key=lambda row: (int(row.get("cluster_index", row.get("cluster_id", 0))), float(row.get("timestamp_seconds") or 0.0)),
    )


def _attached_frame_rows(clip: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for frame in clip.get("frames", []):
        resolved_path = resolve_existing_path(frame.get("path"))
        if resolved_path is not None:
            row = dict(frame)
            row["path"] = str(resolved_path)
            row.setdefault("source", "evidence_packet_frame")
            rows.append(row)
    return rows


def _timestamp_from_row(row: dict[str, Any]) -> float | None:
    for key in ("pruned_timestamp_seconds", "timestamp_seconds"):
        try:
            return float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
    return None


def _extract_frames_at_timestamps(
    video_path: str | Path,
    output_dir: str | Path,
    timestamps: list[float],
) -> list[dict[str, Any]]:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required for timestamp frame extraction")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for idx, timestamp in enumerate(timestamps, 1):
        safe_timestamp = max(0.0, float(timestamp))
        frame_path = output_dir / f"frame_{idx:02d}_{safe_timestamp:.2f}s.jpg"
        if not frame_path.exists():
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{safe_timestamp:.3f}",
                    "-i",
                    str(video_path),
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    str(frame_path),
                ],
                check=True,
            )
        rows.append({"timestamp_seconds": round(safe_timestamp, 3), "path": str(frame_path)})
    return rows


def sample_object_hint_frames(
    clip: dict[str, Any],
    *,
    frames_per_clip: int,
    output_dir: str | Path,
    evidence_id: str,
    frame_sampling_mode: str,
) -> list[dict[str, Any]]:
    frame_dir = Path(output_dir) / "object_hint_frames" / evidence_id / str(clip.get("agent_dir") or clip.get("agent_name"))
    resolved_video = resolve_existing_path(clip.get("local_video"))
    if resolved_video is not None:
        if frame_sampling_mode == "clip_medoids":
            kept_cluster_rows = _kept_cluster_representative_rows(clip)
            if kept_cluster_rows:
                selected_rows = kept_cluster_rows if frames_per_clip <= 0 else kept_cluster_rows[:frames_per_clip]
            else:
                selected_rows = select_frame_rows(_attached_frame_rows(clip), count=frames_per_clip, mode=frame_sampling_mode)
            timestamps = [
                timestamp
                for timestamp in (_timestamp_from_row(row) for row in selected_rows)
                if timestamp is not None
            ]
            if timestamps:
                selected_timestamps = timestamps if frames_per_clip <= 0 else timestamps[:frames_per_clip]
                extracted = _extract_frames_at_timestamps(resolved_video, frame_dir, selected_timestamps)
                for index, row in enumerate(extracted):
                    if index < len(selected_rows):
                        for key in (
                            "cluster_index",
                            "frame_index",
                            "member_indices",
                            "member_timestamps",
                            "member_count",
                            "source_timestamp_seconds",
                            "pruned_timestamp_seconds",
                        ):
                            if key in selected_rows[index]:
                                row[key] = selected_rows[index][key]
                    row["source"] = "pruned_local_video_medoid_timestamp_frame"
                    row["video_path"] = str(resolved_video)
                return extracted
        duration = ffprobe_duration(resolved_video)
        fallback_count = frames_per_clip if frames_per_clip > 0 else 3
        extracted = extract_frames(resolved_video, frame_dir, frames_per_clip=fallback_count, duration=duration)
        for row in extracted:
            row["source"] = "pruned_local_video_frame"
            row["video_path"] = str(resolved_video)
        return extracted

    rows = select_frame_rows(_attached_frame_rows(clip), count=frames_per_clip, mode=frame_sampling_mode)
    if rows:
        for row in rows:
            row.setdefault("source", "evidence_packet_frame_fallback")
    return rows


def existing_frame_rows(
    clip: dict[str, Any],
    *,
    frames_per_clip: int,
    output_dir: str | Path,
    evidence_id: str,
    frame_sampling_mode: str,
) -> list[dict[str, Any]]:
    return sample_object_hint_frames(
        clip,
        frames_per_clip=frames_per_clip,
        output_dir=output_dir,
        evidence_id=evidence_id,
        frame_sampling_mode=frame_sampling_mode,
    )


def _clip_gaze_point(clip: dict[str, Any]) -> tuple[float, float] | None:
    summary = clip.get("gaze_summary")
    if not isinstance(summary, dict) or summary.get("projection_status") != "projected":
        return None
    projected = summary.get("projected_gaze_summary")
    if not isinstance(projected, dict):
        projected = summary
    try:
        x = float(projected["median_x"])
        y = float(projected["median_y"])
    except (KeyError, TypeError, ValueError):
        return None
    return x, y


def _object_area_ratio(obj: dict[str, Any]) -> float:
    width = float(obj.get("image_width") or 1)
    height = float(obj.get("image_height") or 1)
    x0, y0, x1, y1 = obj.get("bbox") or [0, 0, 0, 0]
    return max(0.0, (float(x1) - float(x0)) * (float(y1) - float(y0)) / max(1.0, width * height))


def score_object_hint(obj: dict[str, Any], clip: dict[str, Any], *, gaze_sigma: float) -> float:
    """Score visible, gaze-near, and not-tiny objects higher."""

    area = _object_area_ratio(obj)
    area_score = min(1.0, math.sqrt(max(area, 0.0)) * 5.0)
    gaze_point = _clip_gaze_point(clip)
    if gaze_point is None:
        return 0.25 + area_score
    try:
        gaze_score = gaussian_bbox_score(gaze_point, tuple(float(v) for v in obj["bbox"]), sigma=gaze_sigma)
    except Exception:
        gaze_score = 0.0
    return 0.25 + area_score + gaze_score


def _dedupe_and_rank_objects(objects: list[dict[str, Any]], *, max_objects: int) -> list[dict[str, Any]]:
    best_by_name: dict[str, dict[str, Any]] = {}
    for obj in objects:
        if obj.get("object_id") is not None:
            key = f"object_id::{obj.get('object_id')}"
        else:
            key = obj.get("normalized_name") or normalize_object_name(str(obj.get("object_name") or obj.get("name") or ""))
        if not key:
            continue
        if obj.get("user"):
            key = f"{obj.get('user')}::{key}"
        current = best_by_name.get(key)
        if current is None or float(obj.get("selection_score", 0.0)) > float(current.get("selection_score", 0.0)):
            best_by_name[key] = obj
    ranked = sorted(
        best_by_name.values(),
        key=lambda row: (-float(row.get("selection_score", 0.0)), str(row.get("object_name") or "")),
    )
    return ranked[:max_objects]


def _attach_frame_context(obj: dict[str, Any], frame: dict[str, Any], clip: dict[str, Any]) -> dict[str, Any]:
    row = dict(obj)
    row["user"] = clip.get("agent_name")
    row["agent_dir"] = clip.get("agent_dir")
    row["frame_path"] = frame.get("path")
    row["timestamp_seconds"] = frame.get("timestamp_seconds")
    row["frame_source"] = frame.get("source")
    row["selection_score"] = round(score_object_hint(row, clip, gaze_sigma=400.0), 6)
    row.pop("raw_detector_output", None)
    return row


def enrich_packet_with_object_hints(
    packet: dict[str, Any],
    *,
    detector: OpenRouterObjectDetector | RunnerObjectDetector | DryRunObjectDetector,
    output_dir: str | Path,
    frame_sampling_mode: str = "evidence_frames",
    frames_per_clip: int = 3,
    objects_per_clip: int = 3,
    objects_per_packet: int = 5,
    n_workers: int = 5,
    enable_reid: bool = False,
    reidentifier: EgoEverythingClipReidentifier | None = None,
    reid_error: str | None = None,
    reid_visual_threshold: float = DEFAULT_REID_VISUAL_THRESHOLD,
    reid_text_threshold: float = DEFAULT_REID_TEXT_THRESHOLD,
    reid_batch_size: int = 32,
) -> dict[str, Any]:
    evidence_id = str(packet.get("evidence_id") or "packet")
    clips = packet.get("clips") if isinstance(packet.get("clips"), list) else []
    packet_out = json.loads(json.dumps(packet))
    detection_tasks: list[tuple[int, dict[str, Any]]] = []
    for clip_index, clip in enumerate(clips):
        frames = sample_object_hint_frames(
            clip,
            frames_per_clip=frames_per_clip,
            output_dir=output_dir,
            evidence_id=evidence_id,
            frame_sampling_mode=frame_sampling_mode,
        )
        for frame in frames:
            frame = dict(frame)
            frame["user"] = clip.get("agent_name")
            detection_tasks.append((clip_index, frame))

    detections_by_clip: dict[int, list[dict[str, Any]]] = {index: [] for index in range(len(clips))}
    failures: list[dict[str, Any]] = []
    if detection_tasks:
        with ThreadPoolExecutor(max_workers=max(1, n_workers)) as executor:
            future_map = {
                executor.submit(detector.detect_objects, frame): (clip_index, frame)
                for clip_index, frame in detection_tasks
            }
            for future in as_completed(future_map):
                clip_index, frame = future_map[future]
                clip = clips[clip_index]
                try:
                    detected = future.result()
                except Exception as exc:
                    failures.append(
                        {
                            "user": clip.get("agent_name"),
                            "frame_path": frame.get("path"),
                            "error": str(exc),
                        }
                    )
                    continue
                detections_by_clip[clip_index].extend(
                    _attach_frame_context(obj, frame, clip)
                    for obj in detected
                )

    reid_failures: list[dict[str, Any]] = []
    next_object_id = 0
    if enable_reid and reidentifier is not None:
        for clip_index, clip in enumerate(clips):
            clip_objects = detections_by_clip.get(clip_index, [])
            if not clip_objects:
                continue
            crop_dir = (
                Path(output_dir)
                / "object_reid_crops"
                / evidence_id
                / str(clip.get("agent_dir") or clip.get("agent_name") or clip_index)
            )
            try:
                merged_objects = reidentify_objects(
                    clip_objects,
                    crop_dir,
                    reidentifier=reidentifier,
                    visual_threshold=reid_visual_threshold,
                    text_threshold=reid_text_threshold,
                    batch_size=reid_batch_size,
                    object_id_offset=next_object_id,
                )
            except Exception as exc:
                reid_failures.append({"user": clip.get("agent_name"), "error": str(exc)})
                continue
            object_ids = [
                int(obj["object_id"])
                for obj in merged_objects
                if obj.get("object_id") is not None
            ]
            if object_ids:
                next_object_id = max(object_ids) + 1
            detections_by_clip[clip_index] = merged_objects

    reid_metadata = {
        "enabled": enable_reid,
        "available": enable_reid and reidentifier is not None,
        "method": "EgoEverything CLIP crop/text similarity merge",
        "model_id": getattr(reidentifier, "model_id", DEFAULT_REID_MODEL_ID) if reidentifier is not None else DEFAULT_REID_MODEL_ID,
        "visual_threshold": reid_visual_threshold,
        "text_threshold": reid_text_threshold,
        "failures": reid_failures[:10],
    }
    if reid_error:
        reid_metadata["error"] = reid_error

    packet_key_objects: list[dict[str, Any]] = []
    for clip_index, clip in enumerate(clips):
        clip_objects = detections_by_clip.get(clip_index, [])
        key_objects = _dedupe_and_rank_objects(clip_objects, max_objects=objects_per_clip)
        reid_object_count = len({obj.get("object_id") for obj in clip_objects if obj.get("object_id") is not None})
        if clip_index < len(packet_out.get("clips", [])):
            packet_out["clips"][clip_index]["object_hints"] = {
                "available": bool(key_objects),
                "detector_model": getattr(detector, "model_id", DEFAULT_OBJECT_DETECTION_MODEL),
                "sampled_frame_count": len([task for task in detection_tasks if task[0] == clip_index]),
                "detected_object_count": len(clip_objects),
                "reid_object_count": reid_object_count,
                "selection_method": (
                    "projected_gaze_gaussian_and_area"
                    if _clip_gaze_point(clip) is not None
                    else "area_and_name_dedup"
                ),
                "frame_source_preference": "pruned_local_video_then_packet_frames",
                "frame_sampling_mode": frame_sampling_mode,
                "reid": reid_metadata,
                "key_objects": key_objects,
            }
        packet_key_objects.extend(key_objects)

    packet_out["object_hints"] = {
        "available": bool(packet_key_objects),
        "detector_model": getattr(detector, "model_id", DEFAULT_OBJECT_DETECTION_MODEL),
        "detector_philosophy": (
            "EgoEverything-style VLM object detection on sampled keyframes with "
            "[ymin,xmin,ymax,xmax] normalized boxes; hints must be verified from raw media."
        ),
        "frame_source_preference": "pruned_local_video_then_packet_frames",
        "frame_sampling_mode": frame_sampling_mode,
        "reid": reid_metadata,
        "key_objects": _dedupe_and_rank_objects(packet_key_objects, max_objects=objects_per_packet),
        "failures": failures[:10],
    }
    return packet_out


def enrich_evidence_with_object_hints(
    *,
    evidence_path: str | Path,
    output_path: str | Path,
    output_dir: str | Path,
    backend: str = "dry-run",
    api_key: str | None = None,
    model_id: str = DEFAULT_OBJECT_DETECTION_MODEL,
    base_url: str = DEFAULT_LOCAL_BASE_URL,
    max_new_tokens: int = 1024,
    max_image_pixels: int = 262144,
    dtype: str = "bfloat16",
    allow_cpu: bool = False,
    disable_thinking: bool = False,
    frame_sampling_mode: str = "evidence_frames",
    frames_per_clip: int = 3,
    objects_per_clip: int = 3,
    objects_per_packet: int = 5,
    n_workers: int = 5,
    max_packets: int | None = None,
    random_seed: int | None = None,
    enable_reid: bool = True,
    reid_model_id: str = DEFAULT_REID_MODEL_ID,
    reid_device: str | None = None,
    reid_visual_threshold: float = DEFAULT_REID_VISUAL_THRESHOLD,
    reid_text_threshold: float = DEFAULT_REID_TEXT_THRESHOLD,
    reid_batch_size: int = 32,
) -> list[dict[str, Any]]:
    rows = list(iter_jsonl(evidence_path))
    if max_packets is not None:
        rows = rows[:max_packets]
    if random_seed is not None:
        random.Random(random_seed).shuffle(rows)
    detector = detector_from_args(
        backend=backend,
        api_key=api_key,
        model_id=model_id,
        base_url=base_url,
        max_new_tokens=max_new_tokens,
        max_image_pixels=max_image_pixels,
        dtype=dtype,
        allow_cpu=allow_cpu,
        disable_thinking=disable_thinking,
    )
    reidentifier = None
    reid_error = None
    if enable_reid:
        try:
            reidentifier = EgoEverythingClipReidentifier(model_id=reid_model_id, device=reid_device)
        except Exception as exc:
            reid_error = str(exc)
    enriched = [
        enrich_packet_with_object_hints(
            packet,
            detector=detector,
            output_dir=output_dir,
            frame_sampling_mode=frame_sampling_mode,
            frames_per_clip=frames_per_clip,
            objects_per_clip=objects_per_clip,
            objects_per_packet=objects_per_packet,
            n_workers=n_workers,
            enable_reid=enable_reid,
            reidentifier=reidentifier,
            reid_error=reid_error,
            reid_visual_threshold=reid_visual_threshold,
            reid_text_threshold=reid_text_threshold,
            reid_batch_size=reid_batch_size,
        )
        for packet in rows
    ]
    write_jsonl(output_path, enriched)
    return enriched


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Add EgoEverything-style object hints to evidence packets")
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-dir", default="outputs/object_hint_assets")
    parser.add_argument(
        "--backend",
        default="dry-run",
        choices=["dry-run", "openrouter", "transformers-local", "openai-compatible-local"],
    )
    parser.add_argument("--api-key")
    parser.add_argument("--model-id", default=DEFAULT_OBJECT_DETECTION_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_LOCAL_BASE_URL)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-image-pixels", type=int, default=262144)
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument(
        "--frame-sampling-mode",
        default="evidence_frames",
        choices=["evidence_frames", "clip_medoids"],
        help=(
            "Currently evidence_frames uses frames already attached to packets, or extracts uniform fallback frames. "
            "clip_medoids is recorded for CLIP representative-frame packet inputs."
        ),
    )
    parser.add_argument(
        "--frames-per-clip",
        type=int,
        default=3,
        help="Maximum sampled frames per clip; with clip_medoids, 0 means all kept pruning clusters.",
    )
    parser.add_argument("--objects-per-clip", type=int, default=3)
    parser.add_argument("--objects-per-packet", type=int, default=5)
    parser.add_argument("--n-workers", type=int, default=5)
    parser.add_argument("--max-packets", type=int)
    parser.add_argument("--random-seed", type=int)
    parser.add_argument("--disable-reid", action="store_true")
    parser.add_argument("--reid-model-id", default=DEFAULT_REID_MODEL_ID)
    parser.add_argument("--reid-device")
    parser.add_argument("--reid-visual-threshold", type=float, default=DEFAULT_REID_VISUAL_THRESHOLD)
    parser.add_argument("--reid-text-threshold", type=float, default=DEFAULT_REID_TEXT_THRESHOLD)
    parser.add_argument("--reid-batch-size", type=int, default=32)
    args = parser.parse_args(argv)
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


if __name__ == "__main__":
    raise SystemExit(main())
