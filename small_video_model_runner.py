"""Local six-video inference adapters for smaller non-Qwen video models."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

from .qwen3vl_runner import generation_kwargs


SMALL_VIDEO_BACKENDS = (
    "videollama3-local",
    "llava-next-video-local",
)


def _torch_dtype(torch: Any, dtype: str) -> Any:
    return {
        "auto": torch.bfloat16,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(dtype, torch.bfloat16)


def _first_real_device(model: Any) -> Any:
    for parameter in model.parameters():
        if str(parameter.device) != "meta":
            return parameter.device
    raise RuntimeError("loaded model has no materialized parameters")


def _move_inputs(inputs: Any, *, device: Any, torch: Any, vision_dtype: Any) -> dict[str, Any]:
    moved = {}
    for key, value in dict(inputs).items():
        if isinstance(value, torch.Tensor):
            if value.is_floating_point() and "pixel_values" in key:
                moved[key] = value.to(device=device, dtype=vision_dtype)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def _decode_new_tokens(processor: Any, generated: Any, input_ids: Any) -> str:
    if input_ids is not None and generated.shape[-1] >= input_ids.shape[-1]:
        generated = generated[:, input_ids.shape[-1] :]
    return processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def _validated_video_paths(video_paths: list[str] | None) -> list[str]:
    paths = [str(path) for path in (video_paths or [])]
    if not paths:
        raise ValueError("at least one video is required")
    missing = [
        path
        for path in paths
        if not Path(path).is_file() or Path(path).stat().st_size <= 0
    ]
    if missing:
        raise FileNotFoundError(f"video inputs are missing or empty: {missing[:5]}")
    return paths


def build_videollama3_conversation(
    *,
    prompt: str,
    video_paths: list[str],
    video_fps: float,
    max_frames_per_video: int,
) -> list[dict[str, Any]]:
    content = [
        {
            "type": "video",
            "video": {
                "video_path": path,
                "fps": video_fps,
                "max_frames": max_frames_per_video,
            },
        }
        for path in video_paths
    ]
    content.append({"type": "text", "text": prompt})
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": content},
    ]


def build_llava_next_video_prompt(*, prompt: str, video_count: int) -> str:
    if video_count <= 0:
        raise ValueError("video_count must be positive")
    video_tokens = "\n".join("<video>" for _ in range(video_count))
    return f"USER: {video_tokens}\n{prompt}\nASSISTANT:"


def decode_video_uniform(
    path: str | Path,
    *,
    target_fps: float,
    max_frames: int,
) -> Any:
    """Decode uniformly distributed RGB frames without retaining the full video."""

    if target_fps <= 0:
        raise ValueError("target_fps must be positive")
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    try:
        import av
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "LLaVA-NeXT-Video local inference requires PyAV and NumPy"
        ) from exc

    source_path = Path(path)
    container = av.open(str(source_path))
    try:
        stream = container.streams.video[0]
        total_frames = int(stream.frames or 0)
        source_fps = float(stream.average_rate) if stream.average_rate else 0.0
        if total_frames > 0 and source_fps > 0:
            duration_seconds = total_frames / source_fps
            desired = min(max_frames, max(1, int(math.ceil(duration_seconds * target_fps))))
            if desired == 1:
                wanted = [0]
            else:
                wanted = [
                    round(index * (total_frames - 1) / (desired - 1))
                    for index in range(desired)
                ]
            wanted = sorted(set(wanted))
            selected = []
            wanted_index = 0
            for frame_index, frame in enumerate(container.decode(video=0)):
                if wanted_index >= len(wanted):
                    break
                if frame_index == wanted[wanted_index]:
                    selected.append(frame.to_ndarray(format="rgb24"))
                    wanted_index += 1
            if selected:
                return np.stack(selected)

        # Metadata-free fallback: sample at the requested FPS until the cap.
        selected = []
        next_sample_time = 0.0
        fallback_source_fps = source_fps if source_fps > 0 else 30.0
        for frame_index, frame in enumerate(container.decode(video=0)):
            timestamp = (
                float(frame.time)
                if frame.time is not None
                else frame_index / fallback_source_fps
            )
            if timestamp + 1e-9 < next_sample_time:
                continue
            selected.append(frame.to_ndarray(format="rgb24"))
            if len(selected) >= max_frames:
                break
            next_sample_time += 1.0 / target_fps
        if not selected:
            raise RuntimeError(f"no video frames decoded from {source_path}")
        return np.stack(selected)
    finally:
        container.close()


class VideoLLaMA3Runner:
    def __init__(
        self,
        model_id: str,
        *,
        max_new_tokens: int = 64,
        dtype: str = "bfloat16",
        video_fps: float = 1.0,
        max_frames_per_video: int = 16,
        attn_implementation: str = "sdpa",
        device_map: str = "auto",
        **_: Any,
    ) -> None:
        if video_fps <= 0:
            raise ValueError("video_fps must be positive")
        if max_frames_per_video <= 0:
            raise ValueError("max_frames_per_video must be positive")
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.video_fps = float(video_fps)
        self.max_frames_per_video = int(max_frames_per_video)
        self.torch = torch
        self.vision_dtype = _torch_dtype(torch, dtype)
        started = time.time()
        print(f"loading_processor={model_id}", flush=True)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        print(f"loading_model={model_id}", flush=True)
        load_kwargs = {
            "trust_remote_code": True,
            "device_map": device_map,
            "attn_implementation": attn_implementation,
        }
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=self.vision_dtype,
                **load_kwargs,
            )
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=self.vision_dtype,
                **load_kwargs,
            )
        self.model.eval()
        self.device = _first_real_device(self.model)
        print(f"model_first_param_device={self.device}", flush=True)
        print(f"model_loaded_seconds={time.time() - started:.1f}", flush=True)

    def prepare_videos(self, video_paths: list[str] | None = None) -> list[str]:
        return _validated_video_paths(video_paths)

    def generate(
        self,
        prompt: str,
        image_paths: list[str] | None = None,
        video_paths: list[str] | None = None,
        decoding_mode: str = "greedy",
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int | None = None,
    ) -> str:
        if image_paths:
            raise ValueError("VideoLLaMA3 tester backend accepts video inputs only")
        paths = _validated_video_paths(video_paths)
        conversation = build_videollama3_conversation(
            prompt=prompt,
            video_paths=paths,
            video_fps=self.video_fps,
            max_frames_per_video=self.max_frames_per_video,
        )
        started = time.time()
        inputs = self.processor(conversation=conversation, return_tensors="pt")
        inputs = _move_inputs(
            inputs,
            device=self.device,
            torch=self.torch,
            vision_dtype=self.vision_dtype,
        )
        input_ids = inputs.get("input_ids")
        kwargs = generation_kwargs(
            max_new_tokens=self.max_new_tokens,
            decoding_mode=decoding_mode,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, **kwargs)
        print(
            f"videollama3_generate_seconds={time.time() - started:.1f} "
            f"videos={len(paths)}",
            flush=True,
        )
        return _decode_new_tokens(self.processor, generated, input_ids)


class LlavaNextVideoRunner:
    def __init__(
        self,
        model_id: str,
        *,
        max_new_tokens: int = 64,
        dtype: str = "bfloat16",
        video_fps: float = 1.0,
        max_frames_per_video: int = 16,
        attn_implementation: str = "sdpa",
        device_map: str = "auto",
        **_: Any,
    ) -> None:
        if video_fps <= 0:
            raise ValueError("video_fps must be positive")
        if max_frames_per_video <= 0:
            raise ValueError("max_frames_per_video must be positive")
        import torch
        from transformers import (
            LlavaNextVideoForConditionalGeneration,
            LlavaNextVideoProcessor,
        )

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.video_fps = float(video_fps)
        self.max_frames_per_video = int(max_frames_per_video)
        self.torch = torch
        self.vision_dtype = _torch_dtype(torch, dtype)
        started = time.time()
        print(f"loading_processor={model_id}", flush=True)
        self.processor = LlavaNextVideoProcessor.from_pretrained(model_id)
        print(f"loading_model={model_id}", flush=True)
        load_kwargs = {
            "device_map": device_map,
            "attn_implementation": attn_implementation,
        }
        try:
            self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
                model_id,
                dtype=self.vision_dtype,
                **load_kwargs,
            )
        except TypeError:
            self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=self.vision_dtype,
                **load_kwargs,
            )
        self.model.eval()
        self.device = _first_real_device(self.model)
        print(f"model_first_param_device={self.device}", flush=True)
        print(f"model_loaded_seconds={time.time() - started:.1f}", flush=True)

    def prepare_videos(self, video_paths: list[str] | None = None) -> list[str]:
        return _validated_video_paths(video_paths)

    def generate(
        self,
        prompt: str,
        image_paths: list[str] | None = None,
        video_paths: list[str] | None = None,
        decoding_mode: str = "greedy",
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int | None = None,
    ) -> str:
        if image_paths:
            raise ValueError("LLaVA-NeXT-Video tester backend accepts video inputs only")
        paths = _validated_video_paths(video_paths)
        videos = [
            decode_video_uniform(
                path,
                target_fps=self.video_fps,
                max_frames=self.max_frames_per_video,
            )
            for path in paths
        ]
        text = build_llava_next_video_prompt(prompt=prompt, video_count=len(paths))
        started = time.time()
        inputs = self.processor(
            text=text,
            videos=videos,
            padding=True,
            return_tensors="pt",
        )
        inputs = _move_inputs(
            inputs,
            device=self.device,
            torch=self.torch,
            vision_dtype=self.vision_dtype,
        )
        input_ids = inputs.get("input_ids")
        kwargs = generation_kwargs(
            max_new_tokens=self.max_new_tokens,
            decoding_mode=decoding_mode,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, **kwargs)
        print(
            f"llava_next_video_generate_seconds={time.time() - started:.1f} "
            f"videos={len(paths)} frames_per_video_cap={self.max_frames_per_video}",
            flush=True,
        )
        return _decode_new_tokens(self.processor, generated, input_ids)


def make_small_video_runner(
    backend: str,
    *,
    model_id: str,
    max_new_tokens: int,
    dtype: str,
    video_fps: float,
    max_frames_per_video: int,
    attn_implementation: str,
    device_map: str,
) -> Any:
    kwargs = {
        "model_id": model_id,
        "max_new_tokens": max_new_tokens,
        "dtype": dtype,
        "video_fps": video_fps,
        "max_frames_per_video": max_frames_per_video,
        "attn_implementation": attn_implementation,
        "device_map": device_map,
    }
    if backend == "videollama3-local":
        return VideoLLaMA3Runner(**kwargs)
    if backend == "llava-next-video-local":
        return LlavaNextVideoRunner(**kwargs)
    raise ValueError(f"unsupported small-video backend: {backend}")
