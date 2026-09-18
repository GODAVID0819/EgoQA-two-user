"""Sampled-frame collation for next-token verdict supervision."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from .contracts import JudgeTask, VERDICT_ASSISTANT_PREFIX
from .data import JudgeExample
from .loss import BinaryClassWeights, sample_weight_for_example


QWEN_VISION_TOKEN_PIXEL_AREA = 28 * 28
DEFAULT_IMAGE_CONTEXT_TARGET_FRACTION = 0.85
DEFAULT_IMAGE_TEXT_TOKEN_RESERVE = 8_192
DEFAULT_IMAGE_ITEM_TOKEN_OVERHEAD = 2


def adaptive_image_max_pixels(
    *,
    image_count: int,
    configured_max_pixels: int,
    min_pixels: int,
    max_input_tokens: int,
    target_fraction: float = DEFAULT_IMAGE_CONTEXT_TARGET_FRACTION,
    text_token_reserve: int = DEFAULT_IMAGE_TEXT_TOKEN_RESERVE,
    item_token_overhead: int = DEFAULT_IMAGE_ITEM_TOKEN_OVERHEAD,
) -> int:
    """Mirror the production Qwen runner's per-call dynamic image cap."""

    if image_count < 0:
        raise ValueError("image_count must be non-negative")
    if not 0 < min_pixels <= configured_max_pixels:
        raise ValueError("pixel bounds must satisfy 0 < min_pixels <= max_pixels")
    if max_input_tokens <= 0:
        raise ValueError("max_input_tokens must be positive")
    if not 0 < target_fraction <= 1:
        raise ValueError("target_fraction must be in (0, 1]")
    if text_token_reserve < 0 or item_token_overhead < 0:
        raise ValueError("token reserves must be non-negative")
    if image_count == 0:
        return int(configured_max_pixels)

    target_tokens = int(max_input_tokens * target_fraction)
    visual_token_budget = (
        target_tokens - text_token_reserve - image_count * item_token_overhead
    )
    minimum_tokens_per_image = max(
        1, int(min_pixels) // QWEN_VISION_TOKEN_PIXEL_AREA
    )
    if visual_token_budget < image_count * minimum_tokens_per_image:
        raise RuntimeError(
            "too many sampled frames to fit at the minimum resolution: "
            f"frames={image_count} max_input_tokens={max_input_tokens}"
        )
    tokens_per_image = max(minimum_tokens_per_image, visual_token_budget // image_count)
    adaptive_cap = tokens_per_image * QWEN_VISION_TOKEN_PIXEL_AREA
    return int(max(min_pixels, min(configured_max_pixels, adaptive_cap)))


def render_frame_order_blocks(blocks: list[tuple[str, int]]) -> str:
    if not blocks:
        return ""
    rows = [
        "Sampled-frame block order (authoritative; every block is chronological):"
    ]
    cursor = 1
    for index, (label, frame_count) in enumerate(blocks, start=1):
        end = cursor + frame_count - 1
        rows.append(
            f"- block_{index}: {label}; attachments {cursor}-{end}; "
            f"{frame_count} frames sampled at 0.5 FPS"
        )
        cursor = end + 1
    return "\n".join(rows) + "\n\n"


def render_frame_order(example: JudgeExample) -> str:
    return render_frame_order_blocks(
        [(frame_set.label, len(frame_set.frames)) for frame_set in example.frame_sets]
    )


def model_visible_prompt(example: JudgeExample) -> str:
    return render_frame_order(example) + example.prompt


def _apply_chat_template(processor: Any, messages: list[dict[str, Any]]) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return processor.apply_chat_template(
            messages,
            **kwargs,
            enable_thinking=False,
        )
    except TypeError:
        try:
            return processor.apply_chat_template(
                messages,
                **kwargs,
                template_kwargs={"enable_thinking": False},
            )
        except TypeError:
            return processor.apply_chat_template(messages, **kwargs)


class JudgeFrameCollator:
    """Load the exact packet-owned 0.5 FPS frames for one judge invocation.

    Batch size one is intentional: an all-six sample contains 1,800 images.
    Gradient accumulation provides the effective batch.
    """

    def __init__(
        self,
        *,
        processor: Any,
        class_weights: Mapping[JudgeTask, BinaryClassWeights],
        task_scales: Mapping[JudgeTask, float],
        min_pixels: int,
        max_pixels: int,
        max_input_tokens: int,
        image_context_target_fraction: float = DEFAULT_IMAGE_CONTEXT_TARGET_FRACTION,
        image_text_token_reserve: int = DEFAULT_IMAGE_TEXT_TOKEN_RESERVE,
        image_item_token_overhead: int = DEFAULT_IMAGE_ITEM_TOKEN_OVERHEAD,
        process_vision_info: Callable[..., Any] | None = None,
    ) -> None:
        if not 0 < min_pixels <= max_pixels:
            raise ValueError("pixel bounds must satisfy 0 < min_pixels <= max_pixels")
        if max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be positive")
        if process_vision_info is None:
            from qwen_vl_utils import process_vision_info as qwen_process_vision_info

            process_vision_info = qwen_process_vision_info
        self.processor = processor
        self.class_weights = class_weights
        self.task_scales = task_scales
        self.min_pixels = int(min_pixels)
        self.max_pixels = int(max_pixels)
        self.max_input_tokens = int(max_input_tokens)
        self.image_context_target_fraction = float(image_context_target_fraction)
        self.image_text_token_reserve = int(image_text_token_reserve)
        self.image_item_token_overhead = int(image_item_token_overhead)
        self.process_vision_info = process_vision_info

    def __call__(self, features: list[JudgeExample]) -> dict[str, Any]:
        if len(features) != 1:
            raise ValueError(
                "long-context sampled-frame judge training requires per-device batch "
                f"size 1; received {len(features)}"
            )
        import torch

        example = features[0]
        effective_max_pixels = adaptive_image_max_pixels(
            image_count=example.frame_count,
            configured_max_pixels=self.max_pixels,
            min_pixels=self.min_pixels,
            max_input_tokens=self.max_input_tokens,
            target_fraction=self.image_context_target_fraction,
            text_token_reserve=self.image_text_token_reserve,
            item_token_overhead=self.image_item_token_overhead,
        )
        content: list[dict[str, Any]] = [
            {
                "type": "image",
                "image": path,
                "min_pixels": self.min_pixels,
                "max_pixels": effective_max_pixels,
            }
            for path in example.frames
        ]
        content.append({"type": "text", "text": model_visible_prompt(example)})
        messages = [{"role": "user", "content": content}]
        rendered = _apply_chat_template(self.processor, messages)
        if "<think>" in rendered or "</think>" in rendered:
            raise RuntimeError(
                "judge chat template unexpectedly enabled thinking before the fixed verdict prefix"
            )
        rendered += VERDICT_ASSISTANT_PREFIX
        try:
            image_inputs, video_inputs, vision_kwargs = self.process_vision_info(
                messages,
                return_video_kwargs=True,
            )
        except TypeError:
            image_inputs, video_inputs = self.process_vision_info(messages)
            vision_kwargs = {}
        batch = self.processor(
            text=[rendered],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **vision_kwargs,
        )
        batch.pop("video_metadata", None)
        input_tokens = int(batch["input_ids"].shape[-1])
        if input_tokens > self.max_input_tokens:
            raise RuntimeError(
                "judge input exceeds max_input_tokens: "
                f"example_id={example.example_id} input_tokens={input_tokens} "
                f"max_input_tokens={self.max_input_tokens}"
            )
        batch["labels"] = torch.tensor(
            [[example.target, example.task_id]],
            dtype=torch.long,
        )
        batch["sample_weights"] = torch.tensor(
            [
                sample_weight_for_example(
                    example,
                    class_weights=self.class_weights,
                    task_scales=self.task_scales,
                )
            ],
            dtype=torch.float32,
        )
        return dict(batch)
