"""Sampled-frame collation for next-token verdict supervision."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
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
MAX_ALL_SIX_VIDEO_INPUT_TOKENS = 140_000
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
    """Compute the conservative per-frame cap on Qwen's merged spatial grid.

    ``image_count`` deliberately remains the number of raw sampled frames, not
    the number of two-frame video tubelets. That preserves activation-memory
    headroom for training even though Qwen's native video patch embedding later
    halves the temporal positions.
    """

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
    rows = ["Sampled-video block order (authoritative; every block is chronological):"]
    for index, (label, frame_count) in enumerate(blocks, start=1):
        rows.append(
            f"- video_block_{index}: {label}; "
            f"{frame_count} frames sampled at 0.5 FPS"
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


def _coerce_video_metadata(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    frames_indices = value.get("frames_indices")
    if frames_indices is not None:
        frames_indices = list(frames_indices)
    total_num_frames = value.get("total_num_frames")
    if total_num_frames is None and frames_indices is not None:
        total_num_frames = len(frames_indices)
    try:
        total_num_frames = int(round(float(total_num_frames)))
    except (TypeError, ValueError):
        total_num_frames = 0
    kwargs = {
        "total_num_frames": total_num_frames,
        "fps": value.get("fps"),
        "width": value.get("width"),
        "height": value.get("height"),
        "duration": value.get("duration"),
        "video_backend": value.get("video_backend"),
        "frames_indices": frames_indices,
    }
    try:
        from transformers.video_utils import VideoMetadata

        return VideoMetadata(**kwargs)
    except Exception:
        return SimpleNamespace(**kwargs)


def _split_video_inputs_and_metadata(
    video_inputs: Any,
    video_kwargs: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Mirror the production Qwen runner's version-compatible video handling."""

    normalized_kwargs = dict(video_kwargs)
    if isinstance(normalized_kwargs.get("fps"), list):
        fps_values = normalized_kwargs["fps"]
        normalized_kwargs["fps"] = fps_values[0] if fps_values else 0.5
    if video_inputs is None:
        return None, normalized_kwargs
    fixed_video_inputs = []
    metadata_rows = []
    found_metadata = False
    for item in video_inputs:
        if isinstance(item, tuple) and len(item) == 2:
            video, metadata = item
            fixed_video_inputs.append(video)
            metadata_rows.append(_coerce_video_metadata(metadata))
            found_metadata = True
        else:
            fixed_video_inputs.append(item)
            metadata_rows.append(None)
    if found_metadata:
        normalized_kwargs["video_metadata"] = metadata_rows
        normalized_kwargs["return_metadata"] = True
    return fixed_video_inputs, normalized_kwargs


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
        self.vision_geometry = qwen_vision_geometry(processor)

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
            vision_token_pixel_area=self.vision_geometry["merged_token_pixel_area"],
        )
        content: list[dict[str, Any]] = [
            {
                "type": "video",
                # qwen-vl-utils officially accepts a list of pre-extracted
                # frames as one video. This retains all 300 frames while using
                # the model's temporal video patching, matching inference.
                "video": list(frame_set.frames),
                "min_pixels": self.min_pixels,
                "max_pixels": effective_max_pixels,
                "sample_fps": 0.5,
                "raw_fps": 0.5,
            }
            for frame_set in example.frame_sets
        ]
        content.append({"type": "text", "text": model_visible_prompt(example)})
        messages = [{"role": "user", "content": content}]
        rendered = _apply_chat_template(self.processor, messages)
        _assert_thinking_disabled(rendered)
        rendered += VERDICT_ASSISTANT_PREFIX
        try:
            image_inputs, video_inputs, vision_kwargs = self.process_vision_info(
                messages,
                image_patch_size=self.vision_geometry["patch_size"],
                return_video_kwargs=True,
                return_video_metadata=True,
            )
        except TypeError:
            try:
                image_inputs, video_inputs, vision_kwargs = self.process_vision_info(
                    messages,
                    image_patch_size=self.vision_geometry["patch_size"],
                    return_video_kwargs=True,
                )
            except TypeError:
                image_inputs, video_inputs = self.process_vision_info(
                    messages,
                    image_patch_size=self.vision_geometry["patch_size"],
                )
                vision_kwargs = {}
        video_inputs, vision_kwargs = _split_video_inputs_and_metadata(
            video_inputs,
            vision_kwargs,
        )
        processor_kwargs: dict[str, Any] = {
            "text": [rendered],
            "padding": True,
            "return_tensors": "pt",
        }
        if image_inputs is not None and len(image_inputs) > 0:
            processor_kwargs["images"] = image_inputs
        if video_inputs is not None and len(video_inputs) > 0:
            processor_kwargs["videos"] = video_inputs
            processor_kwargs.update(vision_kwargs)
        batch = self.processor(**processor_kwargs)
        batch.pop("video_metadata", None)
        input_tokens = int(batch["input_ids"].shape[-1])
        if input_tokens > self.max_input_tokens:
            raise RuntimeError(
                "judge input exceeds max_input_tokens: "
                f"example_id={example.example_id} input_tokens={input_tokens} "
                f"max_input_tokens={self.max_input_tokens}"
            )
        if (
            len(example.frame_sets) == 6
            and example.frame_count == 1_800
            and input_tokens > MAX_ALL_SIX_VIDEO_INPUT_TOKENS
        ):
            raise RuntimeError(
                "all-six input did not receive the expected Qwen temporal video "
                "packing: "
                f"example_id={example.example_id} input_tokens={input_tokens} "
                f"expected_at_most={MAX_ALL_SIX_VIDEO_INPUT_TOKENS}"
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
