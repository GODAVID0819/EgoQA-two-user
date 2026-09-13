"""Leak-free dual-video prompt and Qwen3-VL processor adapter."""

from __future__ import annotations

import json
from typing import Any, Callable

from .data import CandidateRecord


def build_messages(
    candidate: CandidateRecord,
    *,
    video_a_path: str,
    video_b_path: str,
    video_a_user: str,
    video_b_user: str,
) -> list[dict[str, Any]]:
    if not all(str(value).strip() for value in (video_a_path, video_b_path, video_a_user, video_b_user)):
        raise ValueError("two materialized videos and their users are required")
    qa = candidate.model_features()
    instruction = "\n".join((
        "You are reviewing one two-user multiple-choice QA candidate.",
        f"Video A (speaker): {video_a_user}",
        f"Video B (provider): {video_b_user}",
        "Use both synchronized videos and the complete QA below.",
        "Judge visual evidence, which view or views are required, and instruction-following formality.",
        "Do not generate an explanation; return hidden states for the ordinal scoring heads.",
        "Candidate QA:",
        json.dumps(qa, ensure_ascii=False, separators=(",", ":")),
    ))
    return [{
        "role": "user",
        "content": [
            {"type": "video", "video": str(video_a_path)},
            {"type": "video", "video": str(video_b_path)},
            {"type": "text", "text": instruction},
        ],
    }]


def _render(processor: Any, messages: list[dict[str, Any]]) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": False}
    try:
        return processor.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return processor.apply_chat_template(messages, **kwargs)


def _qwen3vl_patch_size(processor: Any) -> int:
    image_processor = getattr(processor, "image_processor", None)
    value = getattr(image_processor, "patch_size", None)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError("Qwen3-VL processor does not expose a positive image patch size")
    return value


def _split_qwen3vl_video_metadata(videos: Any) -> tuple[list[Any], list[Any]]:
    """Unpack the ``(video, metadata)`` pairs required by Qwen3-VL."""

    if not isinstance(videos, (list, tuple)) or len(videos) != 2:
        raise RuntimeError(
            "qwen-vl-utils must return exactly two reviewer video/metadata pairs"
        )
    video_inputs: list[Any] = []
    video_metadata: list[Any] = []
    for index, item in enumerate(videos, start=1):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise RuntimeError(
                "qwen-vl-utils did not return Qwen3-VL video metadata for "
                f"reviewer video {index}"
            )
        video, metadata = item
        shape = getattr(video, "shape", None)
        if shape is not None and (len(shape) != 4 or int(shape[0]) <= 0):
            raise RuntimeError(
                f"reviewer video {index} decoded to an invalid frame tensor: {shape}"
            )
        video_inputs.append(video)
        video_metadata.append(metadata)
    return video_inputs, video_metadata


def encode_candidate(
    processor: Any,
    process_vision_info: Callable[..., Any],
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    rendered = _render(processor, messages)
    vision = process_vision_info(
        messages,
        image_patch_size=_qwen3vl_patch_size(processor),
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    if not isinstance(vision, tuple) or len(vision) != 3:
        raise RuntimeError("qwen_vl_utils.process_vision_info returned unsupported output")
    images, packed_videos, video_kwargs = vision
    videos, video_metadata = _split_qwen3vl_video_metadata(packed_videos)
    if not isinstance(video_kwargs, dict):
        raise RuntimeError("qwen-vl-utils returned invalid Qwen3-VL video kwargs")
    encoded = processor(
        text=[rendered],
        images=images,
        videos=videos,
        video_metadata=video_metadata,
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
        # qwen-vl-utils already resizes both videos. Resizing them again in the
        # Transformers processor violates the Qwen3-VL 0.0.14 contract.
        do_resize=False,
        **video_kwargs,
    )
    if "input_ids" not in encoded or "attention_mask" not in encoded:
        raise RuntimeError("processor response lacks input_ids or attention_mask")
    if hasattr(encoded, "pop"):
        encoded.pop("video_metadata", None)
    return dict(encoded)
