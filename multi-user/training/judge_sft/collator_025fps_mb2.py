from __future__ import annotations

from typing import Any

from .collator import (
    DEFAULT_IMAGE_CONTEXT_TARGET_FRACTION,
    DEFAULT_IMAGE_ITEM_TOKEN_OVERHEAD,
    DEFAULT_IMAGE_TEXT_TOKEN_RESERVE,
    QWEN_VISION_TOKEN_PIXEL_AREA,
    JudgeFrameCollator as _BaseCollator,
    _apply_chat_template,
    _assert_thinking_disabled,
    adaptive_image_max_pixels,
)
from .contracts import VERDICT_ASSISTANT_PREFIX
from .loss import sample_weight_for_example


def _prompt_025fps(example) -> str:
    rows = [
        "Sampled-image group order (authoritative; every group is chronological):"
    ]
    for i, frame_set in enumerate(example.frame_sets, 1):
        n = len(frame_set.frames[::2])
        rows.append(
            f"- image_group_{i}: {frame_set.label}; "
            f"the next {n} images are frames sampled at 0.25 FPS"
        )
    return "\n".join(rows) + "\n\n" + example.prompt


class JudgeFrameCollator(_BaseCollator):
    """0.25 FPS independent-image collator with microbatch <= 2."""

    def __call__(self, features: list[Any]) -> dict[str, Any]:
        import torch

        if not 1 <= len(features) <= 2:
            raise ValueError(f"expected microbatch size 1 or 2, got {len(features)}")

        rendered_batch = []
        all_images = []
        labels = []
        sample_weights = []
        fingerprints = []

        for example in features:
            # Intentionally compute this from the ORIGINAL frame count.
            # Therefore halving FPS does NOT increase per-image resolution.
            effective_max_pixels = self.effective_max_pixels(example)

            content = []
            sampled_frames = []

            for frame_set in example.frame_sets:
                frames = tuple(frame_set.frames[::2])
                sampled_frames.extend(frames)

                content.extend(
                    {
                        "type": "image",
                        "image": frame,
                        "min_pixels": self.min_pixels,
                        "max_pixels": effective_max_pixels,
                    }
                    for frame in frames
                )

            content.append(
                {"type": "text", "text": _prompt_025fps(example)}
            )

            messages = [{"role": "user", "content": content}]
            rendered = _apply_chat_template(self.processor, messages)
            _assert_thinking_disabled(rendered)
            rendered_batch.append(rendered + VERDICT_ASSISTANT_PREFIX)

            if sampled_frames:
                cache_key = (effective_max_pixels, tuple(sampled_frames))
                cached = self._decoded_image_cache.get(cache_key)

                if cached is not None:
                    self._decoded_image_cache.move_to_end(cache_key)
                    image_inputs = list(cached)
                else:
                    image_inputs, video_inputs = self.process_vision_info(
                        messages,
                        image_patch_size=self.vision_geometry["patch_size"],
                    )

                    if video_inputs:
                        raise RuntimeError(
                            "independent-image input unexpectedly produced videos"
                        )

                    if (
                        self.decoded_image_cache_entries > 0
                        and image_inputs is not None
                    ):
                        self._decoded_image_cache[cache_key] = tuple(image_inputs)
                        self._decoded_image_cache.move_to_end(cache_key)

                        while (
                            len(self._decoded_image_cache)
                            > self.decoded_image_cache_entries
                        ):
                            self._decoded_image_cache.popitem(last=False)

                if image_inputs:
                    all_images.extend(image_inputs)

            labels.append([example.target, example.task_id])
            sample_weights.append(
                sample_weight_for_example(
                    example,
                    class_weights=self.class_weights,
                    task_scales=self.task_scales,
                )
            )
            fingerprints.append(
                self._example_fingerprint(example.example_id)
            )

        kwargs = {
            "text": rendered_batch,
            "padding": True,
            "return_tensors": "pt",
        }

        if all_images:
            kwargs["images"] = all_images

        batch = self.processor(**kwargs)
        batch.pop("video_metadata", None)

        if int(batch["input_ids"].shape[-1]) > self.max_input_tokens:
            raise RuntimeError(
                "judge batch exceeds max_input_tokens: "
                f"{batch['input_ids'].shape[-1]} > {self.max_input_tokens}"
            )

        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        batch["sample_weights"] = torch.tensor(
            sample_weights,
            dtype=torch.float32,
        )
        batch["tp_example_fingerprint"] = torch.tensor(
            fingerprints,
            dtype=torch.long,
        )

        return batch
