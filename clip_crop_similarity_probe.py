"""Probe whether frame border artifacts affect CLIP image embeddings.

The script writes two images from the same source:

1. the original frame;
2. a center crop intended to remove black rounded/fisheye corners and corner
   metadata overlays.

It then computes CLIP cosine similarity between the two images when the local
torch/transformers/CLIP dependencies are available.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from PIL import Image

try:
    from .clip_gap_demo import DEFAULT_CLIP_MODEL, TransformersClipEncoder, cosine_similarity
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from egolife_two_user_qa.clip_gap_demo import DEFAULT_CLIP_MODEL, TransformersClipEncoder, cosine_similarity


def extract_video_frame(
    video_path: str | Path,
    output_path: str | Path,
    *,
    timestamp_seconds: float,
    ffmpeg_binary: str = "ffmpeg",
) -> Path:
    ffmpeg = shutil.which(ffmpeg_binary)
    if not ffmpeg:
        explicit = Path(ffmpeg_binary)
        if explicit.exists():
            ffmpeg = str(explicit)
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required when --video is used")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{timestamp_seconds:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(output_path),
        ],
        check=True,
    )
    return output_path


def center_crop_image(
    input_path: str | Path,
    output_path: str | Path,
    *,
    crop_scale: float = 0.70,
) -> dict[str, Any]:
    if not 0.1 <= crop_scale <= 1.0:
        raise ValueError("--crop-scale must be between 0.1 and 1.0")

    image = Image.open(input_path).convert("RGB")
    width, height = image.size
    side = int(math.floor(min(width, height) * crop_scale))
    left = (width - side) // 2
    top = (height - side) // 2
    box = (left, top, left + side, top + side)
    cropped = image.crop(box)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(output_path, quality=95)
    return {
        "input_size": [width, height],
        "crop_box_xyxy": list(box),
        "crop_size": [side, side],
        "crop_scale": crop_scale,
    }


def compute_clip_similarity(
    original_path: str | Path,
    cropped_path: str | Path,
    *,
    model_id: str = DEFAULT_CLIP_MODEL,
) -> dict[str, Any]:
    encoder = TransformersClipEncoder(model_id)
    original_embedding, cropped_embedding = encoder.encode([str(original_path), str(cropped_path)])
    return {
        "model_id": encoder.model_id,
        "cosine_similarity": round(cosine_similarity(original_embedding, cropped_embedding), 6),
    }


def compute_clip_pairwise_similarity(
    image_paths: dict[str, str | Path],
    *,
    model_id: str = DEFAULT_CLIP_MODEL,
) -> dict[str, Any]:
    labels = list(image_paths)
    encoder = TransformersClipEncoder(model_id)
    embeddings = encoder.encode([str(image_paths[label]) for label in labels])
    comparisons = {}
    for left_index, left_label in enumerate(labels):
        for right_index in range(left_index + 1, len(labels)):
            right_label = labels[right_index]
            comparisons[f"{left_label}_vs_{right_label}"] = round(
                cosine_similarity(embeddings[left_index], embeddings[right_index]),
                6,
            )
    return {
        "model_id": encoder.model_id,
        "comparisons": comparisons,
    }


def write_original_and_crop(
    *,
    image_path: str | Path,
    original_path: str | Path,
    cropped_path: str | Path,
    crop_scale: float,
) -> dict[str, Any]:
    image = Image.open(image_path).convert("RGB")
    original_path = Path(original_path)
    original_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(original_path, quality=95)
    return center_crop_image(original_path, cropped_path, crop_scale=crop_scale)


def run_probe(
    *,
    image_path: str | Path | None,
    compare_image_path: str | Path | None,
    video_path: str | Path | None,
    output_dir: str | Path,
    timestamp_seconds: float,
    crop_scale: float,
    model_id: str,
    ffmpeg_binary: str,
    skip_clip: bool,
) -> dict[str, Any]:
    if bool(image_path) == bool(video_path):
        raise ValueError("pass exactly one of --image or --video")
    if compare_image_path and video_path:
        raise ValueError("--compare-image can only be used with --image")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    original_path = output_dir / "original_frame.jpg"
    if video_path:
        extract_video_frame(
            video_path,
            original_path,
            timestamp_seconds=timestamp_seconds,
            ffmpeg_binary=ffmpeg_binary,
        )
        cropped_path = output_dir / "cropped_center_frame.jpg"
        crop_info = center_crop_image(original_path, cropped_path, crop_scale=crop_scale)
    else:
        cropped_path = output_dir / "cropped_center_frame.jpg"
        crop_info = write_original_and_crop(
            image_path=image_path,
            original_path=original_path,
            cropped_path=cropped_path,
            crop_scale=crop_scale,
        )

    result: dict[str, Any] = {
        "source": str(video_path or image_path),
        "timestamp_seconds": timestamp_seconds if video_path else None,
        "original_path": str(original_path),
        "cropped_path": str(cropped_path),
        "crop": crop_info,
    }
    if compare_image_path:
        compare_original_path = output_dir / "compare_original_frame.jpg"
        compare_cropped_path = output_dir / "compare_cropped_center_frame.jpg"
        compare_crop_info = write_original_and_crop(
            image_path=compare_image_path,
            original_path=compare_original_path,
            cropped_path=compare_cropped_path,
            crop_scale=crop_scale,
        )
        result["compare_source"] = str(compare_image_path)
        result["compare_original_path"] = str(compare_original_path)
        result["compare_cropped_path"] = str(compare_cropped_path)
        result["compare_crop"] = compare_crop_info
    if not skip_clip:
        try:
            if compare_image_path:
                result["clip"] = compute_clip_pairwise_similarity(
                    {
                        "source_original": original_path,
                        "source_cropped": cropped_path,
                        "compare_original": compare_original_path,
                        "compare_cropped": compare_cropped_path,
                    },
                    model_id=model_id,
                )
            else:
                result["clip"] = compute_clip_similarity(original_path, cropped_path, model_id=model_id)
        except Exception as exc:
            result["clip_error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }

    report_path = output_dir / "clip_crop_similarity.json"
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare CLIP similarity before/after center crop")
    parser.add_argument("--image", help="Existing image/frame path")
    parser.add_argument("--compare-image", help="Optional second image to compare before/after cropping")
    parser.add_argument("--video", help="Video path to sample")
    parser.add_argument("--timestamp-seconds", type=float, default=3.0)
    parser.add_argument("--output-dir", default="outputs/clip_crop_similarity_probe")
    parser.add_argument("--crop-scale", type=float, default=0.70)
    parser.add_argument("--model-id", default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--ffmpeg-binary", default="ffmpeg")
    parser.add_argument("--skip-clip", action="store_true", help="Only write image variants")
    args = parser.parse_args(argv)
    result = run_probe(
        image_path=args.image,
        compare_image_path=args.compare_image,
        video_path=args.video,
        output_dir=args.output_dir,
        timestamp_seconds=args.timestamp_seconds,
        crop_scale=args.crop_scale,
        model_id=args.model_id,
        ffmpeg_binary=args.ffmpeg_binary,
        skip_clip=args.skip_clip,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
