"""Independent-image collation for next-token verdict supervision."""

from __future__ import annotations

import hashlib
import inspect
from typing import Any, Callable, Mapping

from .contracts import JudgeTask, VERDICT_ASSISTANT_PREFIX
from .data import JudgeExample
from .loss import BinaryClassWeights, sample_weight_for_example


QWEN_VISION_PATCH_SIZE = 16
QWEN_VISION_SPATIAL_MERGE_SIZE = 2
QWEN_VISION_TEMPORAL_PATCH_SIZE = 2
QWEN_VISION_TOKEN_PIXEL_AREA = (
    QWEN_VISION_PATCH_SIZE * QWEN_VISION_SPATIAL_MERGE_SIZE
) ** 2
DEFAULT_IMAGE_CONTEXT_TARGET_FRACTION = 0.85
DEFAULT_IMAGE_TEXT_TOKEN_RESERVE = 8_192
DEFAULT_IMAGE_ITEM_TOKEN_OVERHEAD = 2
QWEN_NO_THINK_ASSISTANT_SUFFIX = "<think>\n\n</think>\n\n"


def adaptive_image_max_pixels(
    *,
    image_count: int,
    configured_max_pixels: int,
    min_pixels: int,
    max_input_tokens: int,
    target_fraction: float = DEFAULT_IMAGE_CONTEXT_TARGET_FRACTION,
    text_token_reserve: int = DEFAULT_IMAGE_TEXT_TOKEN_RESERVE,
    item_token_overhead: int = DEFAULT_IMAGE_ITEM_TOKEN_OVERHEAD,
    vision_token_pixel_area: int = QWEN_VISION_TOKEN_PIXEL_AREA,
) -> int:
    """Compute the per-frame cap for Qwen's independent-image representation."""

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
    if vision_token_pixel_area <= 0:
        raise ValueError("vision_token_pixel_area must be positive")
    if image_count == 0:
        return int(configured_max_pixels)

    target_tokens = int(max_input_tokens * target_fraction)
    visual_token_budget = (
        target_tokens - text_token_reserve - image_count * item_token_overhead
    )
    minimum_tokens_per_image = max(
        1, int(min_pixels) // vision_token_pixel_area
    )
    if visual_token_budget < image_count * minimum_tokens_per_image:
        raise RuntimeError(
            "too many sampled frames to fit at the minimum resolution: "
            f"frames={image_count} max_input_tokens={max_input_tokens}"
        )
    tokens_per_image = max(minimum_tokens_per_image, visual_token_budget // image_count)
    adaptive_cap = tokens_per_image * vision_token_pixel_area
    return int(max(min_pixels, min(configured_max_pixels, adaptive_cap)))


def _positive_processor_int(processor: Any, *names: str) -> int:
    image_processor = getattr(processor, "image_processor", None)
    for name in names:
        value = getattr(image_processor, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    raise RuntimeError(
        "Qwen processor image configuration is missing a positive "
        + "/".join(names)
    )


def qwen_vision_geometry(processor: Any) -> dict[str, int]:
    """Read and validate the Qwen3.8 vision geometry used by preprocessing."""

    geometry = {
        "patch_size": _positive_processor_int(processor, "patch_size"),
        "spatial_merge_size": _positive_processor_int(
            processor, "merge_size", "spatial_merge_size"
        ),
        "temporal_patch_size": _positive_processor_int(
            processor, "temporal_patch_size"
        ),
    }
    expected = {
        "patch_size": QWEN_VISION_PATCH_SIZE,
        "spatial_merge_size": QWEN_VISION_SPATIAL_MERGE_SIZE,
        "temporal_patch_size": QWEN_VISION_TEMPORAL_PATCH_SIZE,
    }
    if geometry != expected:
        raise RuntimeError(
            f"unexpected Qwen3.8 vision geometry: observed={geometry} "
            f"expected={expected}"
        )
    geometry["merged_token_pixel_area"] = (
        geometry["patch_size"] * geometry["spatial_merge_size"]
    ) ** 2
    return geometry


def render_frame_order_blocks(blocks: list[tuple[str, int]]) -> str:
    if not blocks:
        return ""
    rows = [
        "Sampled-image group order (authoritative; every group is chronological):"
    ]
    for index, (label, frame_count) in enumerate(blocks, start=1):
        rows.append(
            f"- image_group_{index}: {label}; "
            f"the next {frame_count} images are frames sampled at 0.5 FPS"
        )
    return "\n".join(rows) + "\n\n"


def render_frame_order(example: JudgeExample) -> str:
    return render_frame_order_blocks(
        [(frame_set.label, len(frame_set.frames)) for frame_set in example.frame_sets]
    )


def model_visible_prompt(example: JudgeExample) -> str:
    return render_frame_order(example) + example.prompt


def _supports_structured_chat_template_kwargs(processor: Any) -> bool:
    try:
        parameter = inspect.signature(processor.apply_chat_template).parameters.get(
            "kwargs"
        )
    except (TypeError, ValueError):
        return False
    return parameter is not None and "AllKwargsForChatTemplate" in str(
        parameter.annotation
    )


def _apply_chat_template(processor: Any, messages: list[dict[str, Any]]) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if _supports_structured_chat_template_kwargs(processor):
        try:
            return processor.apply_chat_template(
                messages,
                **kwargs,
                template_kwargs={"enable_thinking": False},
            )
        except TypeError:
            pass
    try:
        return processor.apply_chat_template(
            messages,
            **kwargs,
            enable_thinking=False,
        )
    except TypeError:
        return processor.apply_chat_template(messages, **kwargs)


def _assert_thinking_disabled(rendered: str) -> None:
    """Accept Qwen3.8's canonical closed empty no-thinking assistant block."""

    if "<think>" not in rendered and "</think>" not in rendered:
        return
    if rendered.endswith(QWEN_NO_THINK_ASSISTANT_SUFFIX):
        before_suffix = rendered[: -len(QWEN_NO_THINK_ASSISTANT_SUFFIX)]
        if "<think>" not in before_suffix and "</think>" not in before_suffix:
            return
    raise RuntimeError(
        "judge chat template left an active or non-empty thinking block before "
        "the fixed verdict prefix"
    )


class JudgeFrameCollator:
    """Load the exact packet-owned 0.5 FPS frames for one judge invocation.

    Every sampled JPEG is a separate Qwen image item. Batch size one is
    intentional: an all-six sample contains 1,800 image items. Gradient
    accumulation provides the effective batch.
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
        self.vision_geometry = qwen_vision_geometry(processor)

    def effective_max_pixels(self, example: JudgeExample) -> int:
        effective_max_pixels = adaptive_image_max_pixels(
            image_count=example.frame_count,
            configured_max_pixels=self.max_pixels,
            min_pixels=self.min_pixels,
            max_input_tokens=self.max_input_tokens,
            target_fraction=self.image_context_target_fraction,
            text_token_reserve=self.image_text_token_reserve,
            item_token_overhead=self.image_item_token_overhead,
            vision_token_pixel_area=self.vision_geometry["merged_token_pixel_area"],
        )
        return effective_max_pixels

    @staticmethod
    def _example_fingerprint(example_id: str) -> int:
        """Return a stable signed-int64-safe identity for TP rank audits."""

        digest = hashlib.sha256(example_id.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)

    def __call__(self, features: list[JudgeExample]) -> dict[str, Any]:
        if len(features) != 1:
            raise ValueError(
                "long-context sampled-frame judge training requires per-device batch "
                f"size 1; received {len(features)}"
            )
        import torch

        example = features[0]
        effective_max_pixels = self.effective_max_pixels(example)
        content: list[dict[str, Any]] = []
        for frame_set in example.frame_sets:
            content.extend(
                {
                    "type": "image",
                    "image": frame,
                    "min_pixels": self.min_pixels,
                    "max_pixels": effective_max_pixels,
                }
                for frame in frame_set.frames
            )
        content.append({"type": "text", "text": model_visible_prompt(example)})
        messages = [{"role": "user", "content": content}]
        rendered = _apply_chat_template(self.processor, messages)
        _assert_thinking_disabled(rendered)
        rendered += VERDICT_ASSISTANT_PREFIX
        image_inputs, video_inputs = self.process_vision_info(
            messages,
            image_patch_size=self.vision_geometry["patch_size"],
        )
        if video_inputs is not None and len(video_inputs) > 0:
            raise RuntimeError("independent-image judge input unexpectedly produced videos")
        processor_kwargs: dict[str, Any] = {
            "text": [rendered],
            "padding": True,
            "return_tensors": "pt",
        }
        if image_inputs is not None and len(image_inputs) > 0:
            processor_kwargs["images"] = image_inputs
        batch = self.processor(**processor_kwargs)
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
        batch["tp_example_fingerprint"] = torch.tensor(
            [self._example_fingerprint(example.example_id)],
            dtype=torch.long,
        )
        return dict(batch)
