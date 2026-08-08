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
        "Do not generate an explanation; return hidden states for the fixed classification heads.",
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


def encode_candidate(
    processor: Any,
    process_vision_info: Callable[..., Any],
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    rendered = _render(processor, messages)
    try:
        vision = process_vision_info(
            messages, return_video_kwargs=True, return_video_metadata=True
        )
    except TypeError:
        try:
            vision = process_vision_info(messages, return_video_kwargs=True)
        except TypeError:
            vision = process_vision_info(messages)
    if not isinstance(vision, tuple) or len(vision) not in {2, 3}:
        raise RuntimeError("qwen_vl_utils.process_vision_info returned unsupported output")
    if len(vision) == 3:
        images, videos, video_kwargs = vision
    else:
        images, videos = vision
        video_kwargs = {}
    try:
        from qwen3vl_runner import split_video_inputs_and_metadata
    except ImportError:
        split_video_inputs_and_metadata = None
    if split_video_inputs_and_metadata is not None:
        videos, video_kwargs = split_video_inputs_and_metadata(videos, dict(video_kwargs or {}))
    encoded = processor(
        text=[rendered],
        images=images,
        videos=videos,
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
        **dict(video_kwargs or {}),
    )
    if "input_ids" not in encoded or "attention_mask" not in encoded:
        raise RuntimeError("processor response lacks input_ids or attention_mask")
    if hasattr(encoded, "pop"):
        encoded.pop("video_metadata", None)
    return dict(encoded)
