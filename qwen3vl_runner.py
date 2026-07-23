"""Qwen3-VL runner backends for local/open-source inference."""

from __future__ import annotations

import base64
import gc
import hashlib
import inspect
import json
import mimetypes
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol


DEFAULT_MODEL_ID = "Qwen/Qwen3.6-27B"
DEFAULT_GEMINI_MODEL_ID = "gemini-3.5-flash"
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_GEMINI_UPLOAD_URL = "https://generativelanguage.googleapis.com/upload/v1beta"
DEFAULT_GEMINI_503_RETRY_DELAY_SECONDS = 60.0
DEFAULT_OPENROUTER_MODEL_ID = "google/gemini-2.5-flash"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MAX_RETRIES = 4
DEFAULT_OPENROUTER_RETRY_DELAY_SECONDS = 2.0
OPENROUTER_REASONING_EFFORTS = ("max", "xhigh", "high", "medium", "low", "minimal", "none")
DEFAULT_MAX_IMAGE_PIXELS = 262144
DEFAULT_VIDEO_FPS = 1.0
MEMORY_SAFE_BACKEND = "transformers-local-memory-safe"
MEMORY_SAFE_DEFAULT_VIDEO_FPS = 1.0
MEMORY_SAFE_DEFAULT_MAX_INPUT_TOKENS = 131_072
MEMORY_SAFE_DEFAULT_MIN_FREE_GIB = 5.0
MEMORY_SAFE_DEFAULT_KV_BYTES_PER_TOKEN = 65_536
MEMORY_SAFE_DEFAULT_MIN_AVAILABLE_RAM_GIB = 16.0
MEMORY_SAFE_DEFAULT_TRANSCODE_MAX_EDGE = 512
MEMORY_SAFE_DEFAULT_TRANSCODE_CRF = 23
MEMORY_SAFE_DEFAULT_ATTN_IMPLEMENTATION = "flash_attention_2"
GENERATOR_DECODING_MODES = ("greedy", "sampling")
DEFAULT_SAMPLING_TEMPERATURE = 0.7
DEFAULT_SAMPLING_TOP_P = 0.9
# Archived inactive score-token defaults:
# DEFAULT_CHOICE_FIELD = "final_quality_score"
# DEFAULT_SCORE_CHOICES = ("1", "2", "3")
DEFAULT_CHOICE_FIELD = "status"
DEFAULT_DECISION_CHOICES = ("PASS", "FAIL")


class OpenRouterRequestError(RuntimeError):
    """An OpenRouter transport/provider failure, not a model judgment."""

    def __init__(
        self,
        message: str,
        *,
        code: int | None = None,
        error_type: str | None = None,
        retryable: bool = True,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.error_type = error_type
        self.retryable = retryable


def locate_choice_token(
    token_texts: list[str],
    *,
    field_name: str = DEFAULT_CHOICE_FIELD,
    choices: tuple[str, ...] = DEFAULT_DECISION_CHOICES,
) -> dict[str, Any] | None:
    """Locate a JSON choice value and the token piece containing it.

    Token APIs expose pieces rather than character offsets.  Returning the
    exact alternative token spellings lets each backend look up PASS/FAIL at the same
    decoding step, including tokenizers that fold JSON punctuation or leading
    whitespace into a choice token.
    """

    text = "".join(token_texts)
    choice_pattern = "|".join(re.escape(choice) for choice in choices)
    pattern = re.compile(
        rf'["\']{re.escape(field_name)}["\']\s*:\s*["\']?({choice_pattern})["\']?'
        r'(?![A-Za-z0-9_/])'
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    match = matches[0]
    choice = match.group(1)
    choice_start = match.start(1)
    cursor = 0
    for index, piece in enumerate(token_texts):
        piece_end = cursor + len(piece)
        if cursor <= choice_start < piece_end:
            offset = choice_start - cursor
            alternatives = {
                candidate: piece[:offset] + candidate + piece[offset + len(choice) :]
                for candidate in choices
            }
            return {
                "token_index": index,
                "generated_choice": choice,
                "token_piece": piece,
                "alternative_token_pieces": alternatives,
            }
        cursor = piece_end
    return None


def choice_logprobs_from_top_candidates(
    token_texts: list[str],
    top_candidates: list[list[dict[str, Any]]],
    *,
    field_name: str = DEFAULT_CHOICE_FIELD,
    choices: tuple[str, ...] = DEFAULT_DECISION_CHOICES,
) -> dict[str, Any]:
    """Extract log probabilities for all choices from an API top-k response."""

    located = locate_choice_token(
        token_texts,
        field_name=field_name,
        choices=choices,
    )
    if located is None:
        return {"available": False, "reason": f"could not locate JSON field {field_name!r}"}
    index = int(located["token_index"])
    if index >= len(top_candidates):
        return {"available": False, "reason": "choice token has no matching logprob step"}
    candidates = top_candidates[index]
    by_token = {
        str(candidate.get("token")): float(candidate["logprob"])
        for candidate in candidates
        if candidate.get("token") is not None and candidate.get("logprob") is not None
    }
    expected = located["alternative_token_pieces"]
    missing = [choice for choice in choices if expected[choice] not in by_token]
    if missing:
        return {
            "available": False,
            "reason": "top-logprobs response omitted choice token(s): " + ", ".join(missing),
            "generated_choice": located["generated_choice"],
            "token_index": index,
        }
    return {
        "available": True,
        "choice_logprobs": {choice: by_token[expected[choice]] for choice in choices},
        "weight_type": "log_probability",
        "generated_choice": located["generated_choice"],
        "token_index": index,
    }


class Generator(Protocol):
    model_id: str

    def generate(
        self,
        prompt: str,
        image_paths: list[str] | None = None,
        video_paths: list[str] | None = None,
        decoding_mode: str = "greedy",
        temperature: float = DEFAULT_SAMPLING_TEMPERATURE,
        top_p: float = DEFAULT_SAMPLING_TOP_P,
        top_k: int | None = None,
    ) -> str:
        ...


def generation_kwargs(
    *,
    max_new_tokens: int,
    decoding_mode: str = "greedy",
    temperature: float = DEFAULT_SAMPLING_TEMPERATURE,
    top_p: float = DEFAULT_SAMPLING_TOP_P,
    top_k: int | None = None,
) -> dict[str, Any]:
    if decoding_mode not in GENERATOR_DECODING_MODES:
        raise ValueError(f"unknown decoding_mode: {decoding_mode}")
    kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
    if decoding_mode == "sampling":
        kwargs.update(
            {
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
        if top_k is not None:
            kwargs["top_k"] = top_k
    else:
        kwargs["do_sample"] = False
    return kwargs


def image_to_data_url(path: str | Path) -> str:
    path = Path(path)
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def file_to_data_url(path: str | Path) -> str:
    path = Path(path)
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def read_json_response(req: urllib.request.Request, *, timeout: int) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {req.full_url}: {detail}") from exc
    return json.loads(raw) if raw else {}


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def available_host_memory_bytes() -> int | None:
    """Return the tightest system/cgroup host-memory availability estimate."""

    candidates: list[int] = []
    try:
        import psutil

        candidates.append(int(psutil.virtual_memory().available))
    except (ImportError, AttributeError, OSError, ValueError):
        pass

    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        try:
            fields = {}
            for line in meminfo.read_text(encoding="utf-8").splitlines():
                key, value = line.split(":", 1)
                fields[key] = value.strip()
            available_kib = int(fields["MemAvailable"].split()[0])
            candidates.append(available_kib * 1024)
        except (KeyError, OSError, TypeError, ValueError):
            pass

    cgroup_file = Path("/proc/self/cgroup")
    if cgroup_file.is_file():
        try:
            for line in cgroup_file.read_text(encoding="utf-8").splitlines():
                _, controllers, relative = line.split(":", 2)
                relative_path = relative.lstrip("/")
                if controllers == "":
                    base = Path("/sys/fs/cgroup") / relative_path
                    limit_path = base / "memory.max"
                    usage_path = base / "memory.current"
                elif "memory" in controllers.split(","):
                    base = Path("/sys/fs/cgroup/memory") / relative_path
                    limit_path = base / "memory.limit_in_bytes"
                    usage_path = base / "memory.usage_in_bytes"
                else:
                    continue
                limit_text = limit_path.read_text(encoding="utf-8").strip()
                if limit_text == "max":
                    continue
                limit = int(limit_text)
                usage = int(usage_path.read_text(encoding="utf-8").strip())
                # Very large v1 limits conventionally mean "unlimited".
                if limit < 2**60:
                    candidates.append(max(0, limit - usage))
        except (OSError, TypeError, ValueError):
            pass

    return min(candidates) if candidates else None


def normalize_video_kwargs(video_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep Qwen video kwargs compatible across qwen-vl-utils/Transformers versions."""

    normalized = dict(video_kwargs)
    if isinstance(normalized.get("fps"), list):
        fps_values = normalized["fps"]
        normalized["fps"] = fps_values[0] if fps_values else 1.0
    return normalized


def coerce_video_metadata(value: Any) -> Any:
    """Return a Transformers-compatible video metadata object when possible."""

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


def split_video_inputs_and_metadata(
    video_inputs: Any,
    video_kwargs: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Split qwen-vl-utils ``(video, metadata)`` pairs for Qwen3-VL processors."""

    if video_inputs is None:
        return video_inputs, normalize_video_kwargs(video_kwargs)
    normalized_kwargs = normalize_video_kwargs(video_kwargs)
    fixed_video_inputs = []
    metadata_rows = []
    found_metadata = False
    for item in video_inputs:
        if isinstance(item, tuple) and len(item) == 2:
            video, metadata = item
            fixed_video_inputs.append(video)
            metadata_rows.append(coerce_video_metadata(metadata))
            found_metadata = True
        else:
            fixed_video_inputs.append(item)
            metadata_rows.append(None)
    if found_metadata:
        normalized_kwargs["video_metadata"] = metadata_rows
        normalized_kwargs["return_metadata"] = True
    return fixed_video_inputs, normalized_kwargs


def load_transformers_model(
    model_id: str,
    dtype: str = "bfloat16",
    *,
    attn_implementation: str = "sdpa",
    device_map: str = "auto",
):
    try:
        import torch
        from transformers import AutoModelForImageTextToText

        torch_dtype = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(dtype, torch.bfloat16)
        kwargs: dict[str, Any] = {
            "device_map": device_map,
            "attn_implementation": attn_implementation,
            "trust_remote_code": True,
        }

        def from_pretrained(model_cls):
            try:
                return model_cls.from_pretrained(model_id, dtype=torch_dtype, **kwargs)
            except TypeError:
                return model_cls.from_pretrained(model_id, torch_dtype=torch_dtype, **kwargs)

        model_id_lower = model_id.lower()
        prefer_explicit_qwen3vl = "qwen3-vl" in model_id_lower
        if prefer_explicit_qwen3vl:
            try:
                from transformers import Qwen3VLForConditionalGeneration

                return from_pretrained(Qwen3VLForConditionalGeneration)
            except ImportError:
                pass

        try:
            return from_pretrained(AutoModelForImageTextToText)
        except ValueError:
            if prefer_explicit_qwen3vl:
                raise
            try:
                from transformers import Qwen3VLForConditionalGeneration

                return from_pretrained(Qwen3VLForConditionalGeneration)
            except ImportError:
                raise
    except ImportError as exc:
        raise RuntimeError(
            "transformers-local backend requires torch, transformers>=4.57, and qwen-vl-utils"
        ) from exc


def supports_structured_chat_template_kwargs(processor: Any) -> bool:
    """Detect the newer ProcessorMixin ``template_kwargs`` routing contract."""

    try:
        kwargs_parameter = inspect.signature(processor.apply_chat_template).parameters.get("kwargs")
    except (TypeError, ValueError):
        return False
    if kwargs_parameter is None:
        return False
    return "AllKwargsForChatTemplate" in str(kwargs_parameter.annotation)


def apply_chat_template_compat(processor: Any, messages: list[dict[str, Any]], *, disable_thinking: bool) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if disable_thinking:
        if supports_structured_chat_template_kwargs(processor):
            try:
                return processor.apply_chat_template(
                    messages,
                    **kwargs,
                    template_kwargs={"enable_thinking": False},
                )
            except TypeError:
                # Some intermediate Transformers versions advertise the structured
                # annotation but still implement the legacy direct-template route.
                pass
        try:
            return processor.apply_chat_template(
                messages,
                **kwargs,
                enable_thinking=False,
            )
        except TypeError:
            pass
    return processor.apply_chat_template(messages, **kwargs)


class Qwen3VLTransformersRunner:
    """Run Qwen3-VL directly through Hugging Face Transformers."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        max_new_tokens: int = 1024,
        max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
        dtype: str = "bfloat16",
        allow_cpu: bool = False,
        disable_thinking: bool = False,
        video_fps: float = DEFAULT_VIDEO_FPS,
        max_input_tokens: int | None = None,
        min_free_gib: float = 0.0,
        kv_bytes_per_token: int = 0,
        min_available_ram_gib: float = 0.0,
        attn_implementation: str = "sdpa",
        device_map: str = "auto",
    ) -> None:
        if not allow_cpu and not cuda_available():
            raise RuntimeError(
                "CUDA is not available. Use --dry-run, --backend openai-compatible-local, "
                "or pass allow_cpu=True only for tiny tests."
            )
        import torch
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.max_image_pixels = max_image_pixels
        self.disable_thinking = disable_thinking
        self.video_fps = float(video_fps)
        self.max_input_tokens = max_input_tokens
        self.min_free_gib = float(min_free_gib)
        self.kv_bytes_per_token = int(kv_bytes_per_token)
        self.min_available_ram_gib = float(min_available_ram_gib)
        self.attn_implementation = attn_implementation
        self.device_map = device_map
        if self.video_fps <= 0:
            raise ValueError("video_fps must be positive")
        if self.max_input_tokens is not None and self.max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be positive when set")
        if self.min_free_gib < 0:
            raise ValueError("min_free_gib must be non-negative")
        if self.kv_bytes_per_token < 0:
            raise ValueError("kv_bytes_per_token must be non-negative")
        if self.min_available_ram_gib < 0:
            raise ValueError("min_available_ram_gib must be non-negative")
        self.process_vision_info = process_vision_info
        start = time.time()
        print(f"loading_processor={model_id}", flush=True)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        print(f"loading_model={model_id}", flush=True)
        self.model = load_transformers_model(
            model_id,
            dtype=dtype,
            attn_implementation=attn_implementation,
            device_map=device_map,
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.torch = torch
        print(f"model_first_param_device={self.device}", flush=True)
        print(f"model_loaded_seconds={time.time() - start:.1f}", flush=True)

    def _enforce_available_host_memory(self, *, stage: str) -> None:
        if self.min_available_ram_gib <= 0:
            return
        available_bytes = available_host_memory_bytes()
        if available_bytes is None:
            raise RuntimeError(
                "Cannot determine available host RAM for memory-safe inference"
            )
        available_gib = available_bytes / 1024**3
        print(
            "qwen_host_ram "
            f"stage={stage} available_gib={available_gib:.3f} "
            f"required_available_gib={self.min_available_ram_gib:.3f}",
            flush=True,
        )
        if available_gib < self.min_available_ram_gib:
            raise RuntimeError(
                "Insufficient available host RAM for memory-safe inference: "
                f"stage={stage} available_gib={available_gib:.3f} "
                f"required_available_gib={self.min_available_ram_gib:.3f}"
            )

    def _estimated_kv_gib(self, *, input_tokens: int) -> float:
        return (
            (input_tokens + self.max_new_tokens) * self.kv_bytes_per_token / 1024**3
        )

    def _required_free_vram_gib(self, *, input_tokens: int) -> float:
        return self._estimated_kv_gib(input_tokens=input_tokens) + self.min_free_gib

    def generate(
        self,
        prompt: str,
        image_paths: list[str] | None = None,
        video_paths: list[str] | None = None,
        decoding_mode: str = "greedy",
        temperature: float = DEFAULT_SAMPLING_TEMPERATURE,
        top_p: float = DEFAULT_SAMPLING_TOP_P,
        top_k: int | None = None,
    ) -> str:
        result = self._generate(
            prompt,
            image_paths=image_paths,
            video_paths=video_paths,
            decoding_mode=decoding_mode,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        return str(result["text"])

    def generate_with_choice_logits(
        self,
        prompt: str,
        image_paths: list[str] | None = None,
        video_paths: list[str] | None = None,
        *,
        field_name: str = DEFAULT_CHOICE_FIELD,
        choices: tuple[str, ...] = DEFAULT_DECISION_CHOICES,
    ) -> dict[str, Any]:
        """Generate text and capture exact local logits at a JSON choice token."""

        return self._generate(
            prompt,
            image_paths=image_paths,
            video_paths=video_paths,
            choice_field=field_name,
            choices=choices,
        )

    def generate_content(
        self,
        content: list[dict[str, Any]],
        *,
        decoding_mode: str = "greedy",
        temperature: float = DEFAULT_SAMPLING_TEMPERATURE,
        top_p: float = DEFAULT_SAMPLING_TOP_P,
        top_k: int | None = None,
    ) -> str:
        """Generate from explicitly interleaved text/image/video content.

        Grouped comparative review uses this path so each candidate label sits
        immediately beside that candidate's full videos in the multimodal prompt.
        """

        result = self._generate(
            "",
            decoding_mode=decoding_mode,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            multimodal_content=content,
        )
        return str(result["text"])

    def _generate(
        self,
        prompt: str,
        image_paths: list[str] | None = None,
        video_paths: list[str] | None = None,
        decoding_mode: str = "greedy",
        temperature: float = DEFAULT_SAMPLING_TEMPERATURE,
        top_p: float = DEFAULT_SAMPLING_TOP_P,
        top_k: int | None = None,
        *,
        choice_field: str | None = None,
        choices: tuple[str, ...] = DEFAULT_DECISION_CHOICES,
        multimodal_content: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        image_paths = image_paths or []
        video_paths = video_paths or []
        if multimodal_content is None:
            content: list[dict[str, Any]] = [
                {"type": "image", "image": image_path, "max_pixels": self.max_image_pixels}
                for image_path in image_paths
            ]
            content.extend(
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": self.max_image_pixels,
                    "fps": self.video_fps,
                }
                for video_path in video_paths
            )
            content.append({"type": "text", "text": prompt})
        else:
            content = [dict(item) for item in multimodal_content]
            image_paths = [str(item.get("image")) for item in content if item.get("type") == "image"]
            video_paths = [str(item.get("video")) for item in content if item.get("type") == "video"]
        prompt_chars = sum(
            len(str(item.get("text", "")))
            for item in content
            if item.get("type") == "text"
        )
        messages = [{"role": "user", "content": content}]
        start = time.time()
        print(
            "qwen_generate_start "
            f"images={len(image_paths)} videos={len(video_paths)} "
            f"prompt_chars={prompt_chars} disable_thinking={self.disable_thinking} "
            f"decoding_mode={decoding_mode}",
            flush=True,
        )
        text = apply_chat_template_compat(
            self.processor,
            messages,
            disable_thinking=self.disable_thinking,
        )
        try:
            vision_info = self.process_vision_info(
                messages,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            image_inputs, video_inputs, video_kwargs = vision_info
        except TypeError:
            try:
                image_inputs, video_inputs, video_kwargs = self.process_vision_info(
                    messages,
                    return_video_kwargs=True,
                )
            except TypeError:
                image_inputs, video_inputs = self.process_vision_info(messages)
                video_kwargs = {}
        vision_seconds = time.time() - start
        print(f"qwen_vision_processed_seconds={vision_seconds:.1f}", flush=True)
        video_inputs, video_kwargs = split_video_inputs_and_metadata(video_inputs, video_kwargs)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        del image_inputs, video_inputs, video_kwargs
        if self.min_available_ram_gib > 0:
            gc.collect()
        self._enforce_available_host_memory(stage="after_processor")
        inputs = inputs.to(self.device)
        encode_seconds = time.time() - start
        input_tokens = int(inputs.input_ids.shape[-1]) if hasattr(inputs, "input_ids") else -1
        inputs.pop("video_metadata", None)
        print(
            f"qwen_processor_encoded_seconds={encode_seconds:.1f} input_tokens={input_tokens}",
            flush=True,
        )
        if self.max_input_tokens is not None and input_tokens > self.max_input_tokens:
            raise RuntimeError(
                "Qwen input exceeds the memory-safe token ceiling: "
                f"input_tokens={input_tokens} max_input_tokens={self.max_input_tokens}. "
                "Reduce video FPS or max image pixels before retrying."
            )
        if (self.min_free_gib > 0 or self.kv_bytes_per_token > 0) and self.torch.cuda.is_available():
            free_bytes, total_bytes = self.torch.cuda.mem_get_info(self.device)
            free_gib = free_bytes / 1024**3
            estimated_kv_gib = self._estimated_kv_gib(input_tokens=input_tokens)
            required_free_gib = self._required_free_vram_gib(input_tokens=input_tokens)
            print(
                "qwen_pre_generate_vram "
                f"free_gib={free_gib:.3f} total_gib={total_bytes / 1024**3:.3f} "
                f"estimated_kv_gib={estimated_kv_gib:.3f} "
                f"workspace_reserve_gib={self.min_free_gib:.3f} "
                f"required_free_gib={required_free_gib:.3f}",
                flush=True,
            )
            if free_gib < required_free_gib:
                raise RuntimeError(
                    "Insufficient free CUDA memory for memory-safe generation: "
                    f"free_gib={free_gib:.3f} required_free_gib={required_free_gib:.3f} "
                    f"estimated_kv_gib={estimated_kv_gib:.3f} "
                    f"workspace_reserve_gib={self.min_free_gib:.3f}"
                )
        generate_kwargs = generation_kwargs(
            max_new_tokens=self.max_new_tokens,
            decoding_mode=decoding_mode,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        capture = None
        if choice_field:
            tokenizer = getattr(self.processor, "tokenizer", self.processor)
            token_ids: set[int] = set()
            whitespace_prefixes = ["", " ", "  ", "   ", "    ", "\n", "\n  ", "\n    ", "\t"]
            json_prefixes = ["", '"', ":", ':"', ': "', '{"', '{ "']
            json_suffixes = ["", '"', '",', '"}', '"\n']
            for whitespace in whitespace_prefixes:
                for json_prefix in json_prefixes:
                    for json_suffix in json_suffixes:
                        for choice in choices:
                            encoded = tokenizer.encode(
                                whitespace + json_prefix + choice + json_suffix,
                                add_special_tokens=False,
                            )
                            if len(encoded) == 1:
                                token_ids.add(int(encoded[0]))

            class ChoiceLogitCapture:
                def __init__(self, ids: list[int]) -> None:
                    self.ids = ids
                    self.steps: list[Any] = []

                def __call__(self, input_ids: Any, scores: Any) -> Any:
                    # Advanced indexing creates a tiny tensor; retaining it does
                    # not retain the full vocabulary logits for every step.
                    self.steps.append(scores[0, self.ids].detach())
                    return scores

            capture = ChoiceLogitCapture(sorted(token_ids))
            from transformers import LogitsProcessorList

            generate_kwargs["logits_processor"] = LogitsProcessorList([capture])
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                **generate_kwargs,
            )
        total_seconds = time.time() - start
        output_tokens = int(generated.shape[-1] - inputs.input_ids.shape[-1])
        print(
            f"qwen_model_generate_seconds={total_seconds - encode_seconds:.1f} "
            f"total_seconds={total_seconds:.1f} output_tokens={output_tokens}",
            flush=True,
        )
        trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated)
        ]
        decoded = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        result: dict[str, Any] = {"text": decoded}
        if not choice_field or capture is None:
            return result

        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        trimmed_token_ids = [int(token_id) for token_id in trimmed[0].tolist()]
        token_texts = [
            tokenizer.decode(
                [token_id],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for token_id in trimmed_token_ids
        ]
        located = locate_choice_token(token_texts, field_name=choice_field, choices=choices)
        if located is None:
            result["choice_logits"] = {
                "available": False,
                "reason": f"could not locate JSON field {choice_field!r}",
            }
            return result
        index = int(located["token_index"])
        if index >= len(capture.steps):
            result["choice_logits"] = {
                "available": False,
                "reason": "choice token has no captured logits step",
            }
            return result
        id_to_position = {token_id: position for position, token_id in enumerate(capture.ids)}
        alternative_ids: dict[str, int] = {}
        for choice, piece in located["alternative_token_pieces"].items():
            encoded = tokenizer.encode(piece, add_special_tokens=False)
            if len(encoded) != 1 or int(encoded[0]) not in id_to_position:
                result["choice_logits"] = {
                    "available": False,
                    "reason": f"choice {choice!r} is not a captured single token at the status step",
                    "generated_choice": located["generated_choice"],
                    "token_index": index,
                }
                return result
            alternative_ids[choice] = int(encoded[0])
        selected = capture.steps[index].float().cpu().tolist()
        result["choice_logits"] = {
            "available": True,
            "choice_logits": {
                choice: float(selected[id_to_position[token_id]])
                for choice, token_id in alternative_ids.items()
            },
            "weight_type": "logit",
            "generated_choice": located["generated_choice"],
            "token_index": index,
        }
        return result


class Qwen3VLMemorySafeTransformersRunner(Qwen3VLTransformersRunner):
    """Serialized, guarded local inference for long-video Qwen3.6-27B runs.

    This is deliberately a separate backend. The historical ``transformers-local``
    runner keeps its 1-FPS, SDPA, non-serialized behavior unchanged.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        max_new_tokens: int = 1024,
        max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
        dtype: str = "bfloat16",
        allow_cpu: bool = False,
        disable_thinking: bool = False,
    ) -> None:
        video_fps = float(
            os.getenv("QWEN_MEMORY_SAFE_VIDEO_FPS", str(MEMORY_SAFE_DEFAULT_VIDEO_FPS))
        )
        max_input_tokens = int(
            os.getenv(
                "QWEN_MEMORY_SAFE_MAX_INPUT_TOKENS",
                str(MEMORY_SAFE_DEFAULT_MAX_INPUT_TOKENS),
            )
        )
        min_free_gib = float(
            os.getenv(
                "QWEN_MEMORY_SAFE_GPU_RESERVE_GIB",
                os.getenv(
                    "QWEN_MEMORY_SAFE_MIN_FREE_GIB",
                    str(MEMORY_SAFE_DEFAULT_MIN_FREE_GIB),
                ),
            )
        )
        min_available_ram_gib = float(
            os.getenv(
                "QWEN_MEMORY_SAFE_MIN_AVAILABLE_RAM_GIB",
                str(MEMORY_SAFE_DEFAULT_MIN_AVAILABLE_RAM_GIB),
            )
        )
        attn_implementation = os.getenv(
            "QWEN_MEMORY_SAFE_ATTN_IMPLEMENTATION",
            MEMORY_SAFE_DEFAULT_ATTN_IMPLEMENTATION,
        )
        self.transcode_local_videos = os.getenv(
            "QWEN_MEMORY_SAFE_TRANSCODE_LOCAL_VIDEOS", "1"
        ).strip().lower() not in {"0", "false", "no"}
        self.transcode_max_edge = int(
            os.getenv(
                "QWEN_MEMORY_SAFE_TRANSCODE_MAX_EDGE",
                str(MEMORY_SAFE_DEFAULT_TRANSCODE_MAX_EDGE),
            )
        )
        self.transcode_crf = int(
            os.getenv(
                "QWEN_MEMORY_SAFE_TRANSCODE_CRF",
                str(MEMORY_SAFE_DEFAULT_TRANSCODE_CRF),
            )
        )
        self.transcode_cache_dir = Path(
            os.getenv(
                "QWEN_MEMORY_SAFE_VIDEO_CACHE_DIR",
                ".qwen_memory_safe_video_cache",
            )
        )
        if self.transcode_max_edge <= 0:
            raise ValueError("QWEN_MEMORY_SAFE_TRANSCODE_MAX_EDGE must be positive")
        if not 0 <= self.transcode_crf <= 51:
            raise ValueError("QWEN_MEMORY_SAFE_TRANSCODE_CRF must be between 0 and 51")
        super().__init__(
            model_id,
            max_new_tokens=max_new_tokens,
            max_image_pixels=max_image_pixels,
            dtype=dtype,
            allow_cpu=allow_cpu,
            disable_thinking=disable_thinking,
            video_fps=video_fps,
            max_input_tokens=max_input_tokens,
            min_free_gib=min_free_gib,
            kv_bytes_per_token=MEMORY_SAFE_DEFAULT_KV_BYTES_PER_TOKEN,
            min_available_ram_gib=min_available_ram_gib,
            attn_implementation=attn_implementation,
        )
        self._inference_lock = threading.Lock()
        print(
            "qwen_memory_safe_config "
            f"model_id={self.model_id} video_fps={self.video_fps:g} "
            f"max_image_pixels={self.max_image_pixels} "
            f"max_input_tokens={self.max_input_tokens} "
            f"gpu_workspace_reserve_gib={self.min_free_gib:g} "
            f"kv_bytes_per_token={self.kv_bytes_per_token} "
            f"min_available_ram_gib={self.min_available_ram_gib:g} "
            f"transcode_local_videos={str(self.transcode_local_videos).lower()} "
            f"transcode_max_edge={self.transcode_max_edge} "
            f"attn_implementation={self.attn_implementation} serialized=true",
            flush=True,
        )

    def _prepare_video_for_memory_safe_decode(self, path: str | Path) -> str:
        source = Path(path).resolve()
        if not self.transcode_local_videos:
            return str(source)
        stat = source.stat()
        cache_key = "|".join(
            (
                str(source),
                str(stat.st_size),
                str(stat.st_mtime_ns),
                str(self.video_fps),
                str(self.transcode_max_edge),
                str(self.transcode_crf),
            )
        )
        digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:20]
        output = self.transcode_cache_dir / f"{source.stem}.{digest}.mp4"
        if output.is_file() and output.stat().st_size > 0:
            return str(output)
        self.transcode_cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.transcode_cache_dir / (
            f".{source.stem}.{digest}.{threading.get_ident()}.tmp.mp4"
        )
        filters = (
            f"fps={self.video_fps:g}",
            (
                f"scale={self.transcode_max_edge}:{self.transcode_max_edge}:"
                "force_original_aspect_ratio=decrease"
            ),
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        )
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            ",".join(filters),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            str(self.transcode_crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temporary),
        ]
        print(
            "qwen_memory_safe_video_transcode_start "
            f"source={source} source_bytes={stat.st_size} "
            f"fps={self.video_fps:g} max_edge={self.transcode_max_edge} "
            f"crf={self.transcode_crf}",
            flush=True,
        )
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            if not temporary.is_file() or temporary.stat().st_size <= 0:
                raise RuntimeError(f"ffmpeg did not create a usable file: {temporary}")
            temporary.replace(output)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "ffmpeg is required by the memory-safe long-video backend"
            ) from exc
        except subprocess.CalledProcessError as exc:
            detail = str(exc.stderr or exc.stdout or exc).strip()[-2000:]
            raise RuntimeError(f"memory-safe video ffmpeg transcode failed: {detail}") from exc
        finally:
            if temporary.exists():
                temporary.unlink()
        print(
            "qwen_memory_safe_video_transcode_done "
            f"output={output} output_bytes={output.stat().st_size}",
            flush=True,
        )
        return str(output)

    def _generate(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        multimodal_content = kwargs.get("multimodal_content")
        if multimodal_content is not None:
            guarded_content = []
            for original_item in multimodal_content:
                item = dict(original_item)
                if item.get("type") == "video":
                    requested_fps = float(item.get("fps", self.video_fps))
                    requested_pixels = int(item.get("max_pixels", self.max_image_pixels))
                    item["fps"] = min(requested_fps, self.video_fps)
                    item["max_pixels"] = min(requested_pixels, self.max_image_pixels)
                guarded_content.append(item)
            kwargs = {**kwargs, "multimodal_content": guarded_content}
        queued_at = time.time()
        with self._inference_lock:
            wait_seconds = time.time() - queued_at
            self._enforce_available_host_memory(stage="before_video_transcode")
            args = list(args)
            if len(args) > 2 and args[2]:
                args[2] = [
                    self._prepare_video_for_memory_safe_decode(path) for path in args[2]
                ]
            elif kwargs.get("video_paths"):
                kwargs = {
                    **kwargs,
                    "video_paths": [
                        self._prepare_video_for_memory_safe_decode(path)
                        for path in kwargs["video_paths"]
                    ],
                }
            if kwargs.get("multimodal_content") is not None:
                prepared_content = []
                for original_item in kwargs["multimodal_content"]:
                    item = dict(original_item)
                    if item.get("type") == "video" and item.get("video"):
                        item["video"] = self._prepare_video_for_memory_safe_decode(
                            item["video"]
                        )
                    prepared_content.append(item)
                kwargs = {**kwargs, "multimodal_content": prepared_content}
            cuda_active = bool(self.torch.cuda.is_available())
            if cuda_active:
                self.torch.cuda.synchronize(self.device)
                self.torch.cuda.reset_peak_memory_stats(self.device)
                allocated_before = self.torch.cuda.memory_allocated(self.device)
            else:
                allocated_before = 0
            print(
                "qwen_memory_safe_inference_start "
                f"lock_wait_seconds={wait_seconds:.3f}",
                flush=True,
            )
            try:
                self._enforce_available_host_memory(stage="before_video_decode")
                return super()._generate(*args, **kwargs)
            finally:
                if cuda_active:
                    self.torch.cuda.synchronize(self.device)
                    peak_allocated = self.torch.cuda.max_memory_allocated(self.device)
                    allocated_after = self.torch.cuda.memory_allocated(self.device)
                    print(
                        "qwen_memory_safe_vram "
                        f"allocated_before_gib={allocated_before / 1024**3:.3f} "
                        f"peak_allocated_gib={peak_allocated / 1024**3:.3f} "
                        f"allocated_after_gib={allocated_after / 1024**3:.3f}",
                        flush=True,
                    )
                    gc.collect()
                    self.torch.cuda.empty_cache()
                print("qwen_memory_safe_inference_done", flush=True)


class OpenAICompatibleLocalRunner:
    """Call a local vLLM/SGLang/llama.cpp OpenAI-compatible server."""

    supports_choice_logits = True

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        base_url: str = "http://127.0.0.1:8000/v1",
        max_new_tokens: int = 1024,
        timeout: int = 600,
        api_key: str | None = None,
        allow_video_input: bool = False,
    ) -> None:
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout
        self.api_key = api_key or os.getenv("LOCAL_VLM_API_KEY") or "none"
        self.allow_video_input = allow_video_input

    def generate(
        self,
        prompt: str,
        image_paths: list[str] | None = None,
        video_paths: list[str] | None = None,
        decoding_mode: str = "greedy",
        temperature: float = DEFAULT_SAMPLING_TEMPERATURE,
        top_p: float = DEFAULT_SAMPLING_TOP_P,
        top_k: int | None = None,
    ) -> str:
        data = self._generate_response(
            prompt,
            image_paths=image_paths,
            video_paths=video_paths,
            decoding_mode=decoding_mode,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        return data["choices"][0]["message"]["content"].strip()

    def generate_with_choice_logits(
        self,
        prompt: str,
        image_paths: list[str] | None = None,
        video_paths: list[str] | None = None,
        *,
        field_name: str = DEFAULT_CHOICE_FIELD,
        choices: tuple[str, ...] = DEFAULT_DECISION_CHOICES,
    ) -> dict[str, Any]:
        data = self._generate_response(
            prompt,
            image_paths=image_paths,
            video_paths=video_paths,
            include_logprobs=True,
        )
        choice = data["choices"][0]
        text = str(choice["message"]["content"]).strip()
        content_logprobs = ((choice.get("logprobs") or {}).get("content") or [])
        token_texts = [str(item.get("token") or "") for item in content_logprobs]
        top_candidates = []
        for item in content_logprobs:
            rows = list(item.get("top_logprobs") or [])
            if item.get("token") is not None and item.get("logprob") is not None:
                rows.append({"token": item["token"], "logprob": item["logprob"]})
            top_candidates.append(rows)
        return {
            "text": text,
            "choice_logits": choice_logprobs_from_top_candidates(
                token_texts,
                top_candidates,
                field_name=field_name,
                choices=choices,
            ),
        }

    def _generate_response(
        self,
        prompt: str,
        image_paths: list[str] | None = None,
        video_paths: list[str] | None = None,
        decoding_mode: str = "greedy",
        temperature: float = DEFAULT_SAMPLING_TEMPERATURE,
        top_p: float = DEFAULT_SAMPLING_TOP_P,
        top_k: int | None = None,
        *,
        include_logprobs: bool = False,
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for path in image_paths or []:
            content.append({"type": "image_url", "image_url": {"url": image_to_data_url(path)}})
        if video_paths and not self.allow_video_input:
            raise RuntimeError(
                "openai-compatible-local backend received video_paths, but video input is disabled. "
                "Use image fallback or pass --allow-openai-video-input for a server that supports video data URLs."
            )
        for path in video_paths or []:
            content.append({"type": "video_url", "video_url": {"url": file_to_data_url(path)}})
        if decoding_mode not in GENERATOR_DECODING_MODES:
            raise ValueError(f"unknown decoding_mode: {decoding_mode}")
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": content}],
            "temperature": temperature if decoding_mode == "sampling" else 0,
            "max_tokens": self.max_new_tokens,
        }
        if decoding_mode == "sampling":
            payload["top_p"] = top_p
            if top_k is not None:
                payload["top_k"] = top_k
        if include_logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = 20
        payload.update(self._extra_request_payload())
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data

    def _extra_request_payload(self) -> dict[str, Any]:
        """Provider-specific request fields for OpenAI-compatible APIs."""

        return {}


class OpenRouterRunner(OpenAICompatibleLocalRunner):
    """Call a multimodal model through OpenRouter's OpenAI-compatible API."""

    # Gemini 2.5 Flash does not advertise logprobs/top_logprobs through OpenRouter.
    # Judges still emit their PASS/FAIL or A-E decision; only entropy diagnostics are absent.
    supports_choice_logits = False

    def __init__(
        self,
        model_id: str = DEFAULT_OPENROUTER_MODEL_ID,
        *,
        base_url: str = DEFAULT_OPENROUTER_BASE_URL,
        max_new_tokens: int = 1024,
        timeout: int = 600,
        api_key: str | None = None,
        allow_video_input: bool = False,
        reasoning_effort: str | None = None,
        max_retries: int | None = None,
        retry_delay_seconds: float | None = None,
    ) -> None:
        if reasoning_effort not in (None, *OPENROUTER_REASONING_EFFORTS):
            raise ValueError(f"unsupported OpenRouter reasoning effort: {reasoning_effort}")
        effective_api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not effective_api_key:
            raise RuntimeError("OpenRouter backend requires --api-key or OPENROUTER_API_KEY")
        super().__init__(
            model_id,
            base_url=base_url,
            max_new_tokens=max_new_tokens,
            timeout=timeout,
            api_key=effective_api_key,
            allow_video_input=allow_video_input,
        )
        self.reasoning_effort = reasoning_effort
        self.max_retries = int(
            os.getenv("OPENROUTER_MAX_RETRIES", str(DEFAULT_OPENROUTER_MAX_RETRIES))
            if max_retries is None
            else max_retries
        )
        self.retry_delay_seconds = float(
            os.getenv(
                "OPENROUTER_RETRY_DELAY_SECONDS",
                str(DEFAULT_OPENROUTER_RETRY_DELAY_SECONDS),
            )
            if retry_delay_seconds is None
            else retry_delay_seconds
        )
        if self.max_retries < 0:
            raise ValueError("OpenRouter max_retries must be non-negative")
        if self.retry_delay_seconds < 0:
            raise ValueError("OpenRouter retry_delay_seconds must be non-negative")
        self.video_max_edge = int(os.getenv("OPENROUTER_VIDEO_MAX_EDGE", "0"))
        self.video_fps = float(os.getenv("OPENROUTER_VIDEO_FPS", "0"))
        self.video_crf = int(os.getenv("OPENROUTER_VIDEO_CRF", "28"))
        self.video_cache_dir = Path(
            os.getenv("OPENROUTER_VIDEO_CACHE_DIR", ".openrouter_video_cache")
        )
        self._video_cache_lock = threading.Lock()
        if self.video_max_edge < 0:
            raise ValueError("OPENROUTER_VIDEO_MAX_EDGE must be non-negative")
        if self.video_fps < 0:
            raise ValueError("OPENROUTER_VIDEO_FPS must be non-negative")
        if not 0 <= self.video_crf <= 51:
            raise ValueError("OPENROUTER_VIDEO_CRF must be between 0 and 51")

    def _prepare_video_for_upload(self, path: str | Path) -> str:
        """Create and cache a smaller MP4 before OpenRouter base64 encoding."""

        source = Path(path).resolve()
        if self.video_max_edge == 0 and self.video_fps == 0:
            return str(source)
        stat = source.stat()
        cache_key = "|".join(
            (
                str(source),
                str(stat.st_size),
                str(stat.st_mtime_ns),
                str(self.video_max_edge),
                str(self.video_fps),
                str(self.video_crf),
            )
        )
        digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:20]
        output = self.video_cache_dir / f"{source.stem}.{digest}.mp4"
        with self._video_cache_lock:
            if output.is_file() and output.stat().st_size > 0:
                return str(output)
            self.video_cache_dir.mkdir(parents=True, exist_ok=True)
            temporary = self.video_cache_dir / f".{source.stem}.{digest}.{threading.get_ident()}.tmp.mp4"
            filters = []
            if self.video_fps > 0:
                filters.append(f"fps={self.video_fps:g}")
            if self.video_max_edge > 0:
                filters.extend(
                    (
                        f"scale={self.video_max_edge}:{self.video_max_edge}:force_original_aspect_ratio=decrease",
                        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    )
                )
            command = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-an",
            ]
            if filters:
                command.extend(("-vf", ",".join(filters)))
            command.extend(
                (
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    str(self.video_crf),
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(temporary),
                )
            )
            print(
                "openrouter_video_transcode_start "
                f"source={source} source_bytes={stat.st_size} "
                f"max_edge={self.video_max_edge} fps={self.video_fps:g} crf={self.video_crf}",
                flush=True,
            )
            try:
                subprocess.run(command, check=True, capture_output=True, text=True)
                if not temporary.is_file() or temporary.stat().st_size <= 0:
                    raise RuntimeError(f"ffmpeg did not create a usable file: {temporary}")
                temporary.replace(output)
            except FileNotFoundError as exc:
                raise RuntimeError(
                    "ffmpeg is required when OpenRouter video compression is enabled"
                ) from exc
            except subprocess.CalledProcessError as exc:
                detail = str(exc.stderr or exc.stdout or exc).strip()[-2000:]
                raise RuntimeError(f"OpenRouter video ffmpeg transcode failed: {detail}") from exc
            finally:
                if temporary.exists():
                    temporary.unlink()
            print(
                "openrouter_video_transcode_done "
                f"source={source} output={output} output_bytes={output.stat().st_size}",
                flush=True,
            )
        return str(output)

    @staticmethod
    def _response_error(data: Any, *, status: int | None = None) -> OpenRouterRequestError | None:
        if not isinstance(data, dict):
            return OpenRouterRequestError(
                "OpenRouter returned a non-object response",
                code=status,
                retryable=True,
            )
        error = data.get("error")
        if isinstance(error, dict):
            metadata = error.get("metadata") if isinstance(error.get("metadata"), dict) else {}
            raw_code = error.get("code", status)
            try:
                code = int(raw_code) if raw_code is not None else None
            except (TypeError, ValueError):
                code = status
            error_type = str(metadata.get("error_type") or "").strip() or None
            message = str(error.get("message") or "OpenRouter provider error").strip()
            retryable_types = {
                "rate_limit_exceeded",
                "provider_unavailable",
                "provider_timeout",
                "server_error",
                "upstream_error",
            }
            nonretryable_types = {
                "authentication",
                "permission_denied",
                "payment_required",
                "invalid_request",
                "context_length_exceeded",
                "max_tokens_exceeded",
                "token_limit_exceeded",
                "string_too_long",
            }
            # Unknown provider-side errors are treated as transient. Only known request,
            # account, and permission problems fail immediately because retries cannot fix them.
            retryable = True
            if code is not None and 400 <= code < 500 and code not in {408, 429}:
                retryable = False
            if error_type in retryable_types:
                retryable = True
            if error_type in nonretryable_types:
                retryable = False
            details = [f"code={code}" if code is not None else "code=unknown"]
            if error_type:
                details.append(f"type={error_type}")
            return OpenRouterRequestError(
                f"OpenRouter error ({', '.join(details)}): {message}",
                code=code,
                error_type=error_type,
                retryable=retryable,
            )
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return OpenRouterRequestError(
                "OpenRouter returned no choices and no structured error",
                code=status,
                retryable=True,
            )
        first_choice = choices[0]
        message = first_choice.get("message") if isinstance(first_choice, dict) else None
        if not isinstance(message, dict) or message.get("content") is None:
            return OpenRouterRequestError(
                "OpenRouter returned a choice without message content",
                code=status,
                retryable=True,
            )
        return None

    def _generate_response(
        self,
        prompt: str,
        image_paths: list[str] | None = None,
        video_paths: list[str] | None = None,
        decoding_mode: str = "greedy",
        temperature: float = DEFAULT_SAMPLING_TEMPERATURE,
        top_p: float = DEFAULT_SAMPLING_TOP_P,
        top_k: int | None = None,
        *,
        include_logprobs: bool = False,
    ) -> dict[str, Any]:
        prepared_video_paths = [self._prepare_video_for_upload(path) for path in (video_paths or [])]
        attempts = self.max_retries + 1
        last_error: OpenRouterRequestError | None = None
        for attempt_index in range(attempts):
            try:
                data = super()._generate_response(
                    prompt,
                    image_paths=image_paths,
                    video_paths=prepared_video_paths,
                    decoding_mode=decoding_mode,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    include_logprobs=include_logprobs,
                )
                response_error = self._response_error(data)
                if response_error is not None:
                    raise response_error
                return data
            except urllib.error.HTTPError as exc:
                try:
                    raw_body = exc.read().decode("utf-8", errors="replace")
                    body = json.loads(raw_body)
                except Exception:
                    body = {"error": {"code": exc.code, "message": str(exc)}}
                last_error = self._response_error(body, status=exc.code) or OpenRouterRequestError(
                    f"OpenRouter HTTP error {exc.code}: {exc.reason}",
                    code=exc.code,
                    retryable=exc.code == 408 or exc.code == 429 or exc.code >= 500,
                )
            except OpenRouterRequestError as exc:
                last_error = exc
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = OpenRouterRequestError(
                    f"OpenRouter transport error: {exc}",
                    retryable=True,
                )
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                last_error = OpenRouterRequestError(
                    f"OpenRouter returned an unreadable JSON response: {exc}",
                    retryable=True,
                )

            assert last_error is not None
            if not last_error.retryable or attempt_index + 1 >= attempts:
                raise last_error
            delay = self.retry_delay_seconds * (2**attempt_index)
            print(
                "openrouter_retry "
                f"model={self.model_id} attempt={attempt_index + 2}/{attempts} "
                f"delay_seconds={delay:.1f} reason={last_error}",
                flush=True,
            )
            if delay:
                time.sleep(delay)
        raise last_error or OpenRouterRequestError("OpenRouter request failed")

    def _extra_request_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"provider": {"allow_fallbacks": True}}
        if self.reasoning_effort is not None:
            payload["reasoning"] = {
                "effort": self.reasoning_effort,
                # The judges need only the final JSON; do not return their private scratchpad.
                "exclude": True,
            }
        return payload


class GeminiRunner:
    """Call Gemini through the native generateContent and Files APIs."""

    def __init__(
        self,
        model_id: str = DEFAULT_GEMINI_MODEL_ID,
        *,
        base_url: str = DEFAULT_GEMINI_BASE_URL,
        upload_url: str = DEFAULT_GEMINI_UPLOAD_URL,
        max_new_tokens: int = 1024,
        timeout: int = 600,
        api_key: str | None = None,
        file_poll_interval_seconds: float = 2.0,
        file_poll_timeout_seconds: float = 300.0,
        service_unavailable_retry_delay_seconds: float = DEFAULT_GEMINI_503_RETRY_DELAY_SECONDS,
        service_unavailable_max_retries: int | None = None,
    ) -> None:
        self.model_id = model_id or DEFAULT_GEMINI_MODEL_ID
        self.base_url = base_url.rstrip("/")
        self.upload_url = upload_url.rstrip("/")
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.file_poll_interval_seconds = file_poll_interval_seconds
        self.file_poll_timeout_seconds = file_poll_timeout_seconds
        self.service_unavailable_retry_delay_seconds = float(service_unavailable_retry_delay_seconds)
        self.service_unavailable_max_retries = service_unavailable_max_retries
        self._file_cache: dict[tuple[str, int, int], dict[str, Any]] = {}
        self._file_cache_lock = threading.Lock()
        if self.service_unavailable_retry_delay_seconds < 0:
            raise ValueError("service_unavailable_retry_delay_seconds must be non-negative")
        if self.service_unavailable_max_retries is not None and self.service_unavailable_max_retries < 0:
            raise ValueError("service_unavailable_max_retries must be non-negative or None")
        if not self.api_key:
            raise RuntimeError("Gemini backend requires --api-key, GEMINI_API_KEY, or GOOGLE_API_KEY")

    def _headers(self, *, content_type: str = "application/json") -> dict[str, str]:
        return {
            "Content-Type": content_type,
            "x-goog-api-key": str(self.api_key),
        }

    def _request_json(
        self,
        url: str,
        *,
        payload: dict[str, Any] | None = None,
        method: str = "POST",
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        data = json.dumps(payload or {}).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(
            url,
            data=data,
            headers=headers or self._headers(),
            method=method,
        )
        retries = 0
        while True:
            try:
                return read_json_response(req, timeout=self.timeout)
            except Exception as exc:
                if self._http_status_code(exc) != 503:
                    raise
                if (
                    self.service_unavailable_max_retries is not None
                    and retries >= self.service_unavailable_max_retries
                ):
                    raise
                retries += 1
                print(
                    "gemini_http_503_retry "
                    f"retry={retries} delay_seconds={self.service_unavailable_retry_delay_seconds:g} "
                    f"url={url}",
                    flush=True,
                )
                time.sleep(self.service_unavailable_retry_delay_seconds)

    @staticmethod
    def _http_status_code(exc: BaseException) -> int | None:
        """Find an HTTP status preserved in a wrapped exception chain."""

        current: BaseException | None = exc
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if isinstance(current, urllib.error.HTTPError):
                return int(current.code)
            current = current.__cause__ or current.__context__
        return None

    def _file_cache_key(self, path: str | Path) -> tuple[str, int, int]:
        resolved = Path(path).resolve()
        stat = resolved.stat()
        return str(resolved), int(stat.st_size), int(stat.st_mtime_ns)

    def _upload_file_once(self, path: str | Path) -> dict[str, Any]:
        path = Path(path).resolve()
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        size = path.stat().st_size
        start_req = urllib.request.Request(
            f"{self.upload_url}/files",
            data=json.dumps({"file": {"display_name": path.name}}).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": str(self.api_key),
                "X-Goog-Upload-Protocol": "resumable",
                "X-Goog-Upload-Command": "start",
                "X-Goog-Upload-Header-Content-Length": str(size),
                "X-Goog-Upload-Header-Content-Type": mime,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(start_req, timeout=self.timeout) as resp:
                upload_session_url = resp.headers.get("X-Goog-Upload-URL")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini file upload start failed for {path}: HTTP {exc.code}: {detail}") from exc
        if not upload_session_url:
            raise RuntimeError(f"Gemini file upload start did not return X-Goog-Upload-URL for {path}")

        upload_req = urllib.request.Request(
            upload_session_url,
            data=path.read_bytes(),
            headers={
                "Content-Length": str(size),
                "Content-Type": mime,
                "X-Goog-Upload-Offset": "0",
                "X-Goog-Upload-Command": "upload, finalize",
            },
            method="POST",
        )
        uploaded = read_json_response(upload_req, timeout=self.timeout)
        file_obj = uploaded.get("file") if isinstance(uploaded, dict) else None
        if not isinstance(file_obj, dict) or not file_obj.get("uri"):
            raise RuntimeError(f"Gemini file upload returned no file.uri for {path}: {uploaded}")
        return self._wait_for_file_active(file_obj)

    def _wait_for_file_active(self, file_obj: dict[str, Any]) -> dict[str, Any]:
        name = str(file_obj.get("name") or "")
        if not name:
            return file_obj
        deadline = time.time() + self.file_poll_timeout_seconds
        last_state = str(file_obj.get("state") or "")
        while time.time() < deadline:
            if last_state in {"", "ACTIVE"}:
                return file_obj
            if last_state == "FAILED":
                raise RuntimeError(f"Gemini file processing failed for {name}: {file_obj}")
            time.sleep(self.file_poll_interval_seconds)
            file_obj = self._request_json(
                f"{self.base_url}/{name}",
                method="GET",
                headers=self._headers(),
            )
            last_state = str(file_obj.get("state") or "")
        raise RuntimeError(f"Gemini file {name} did not become ACTIVE; last_state={last_state}")

    def _file_part(self, path: str | Path) -> dict[str, Any]:
        key = self._file_cache_key(path)
        with self._file_cache_lock:
            cached = self._file_cache.get(key)
            if cached is None:
                print(f"gemini_upload_start path={key[0]} bytes={key[1]}", flush=True)
                cached = self._upload_file_once(key[0])
                self._file_cache[key] = cached
                print(
                    "gemini_upload_done "
                    f"path={key[0]} uri={cached.get('uri')} state={cached.get('state')}",
                    flush=True,
                )
        mime = cached.get("mimeType") or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        return {"file_data": {"mime_type": mime, "file_uri": cached["uri"]}}

    def _image_part(self, path: str | Path) -> dict[str, Any]:
        path = Path(path)
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        return {
            "inline_data": {
                "mime_type": mime,
                "data": base64.b64encode(path.read_bytes()).decode("ascii"),
            }
        }

    def prepare_videos(self, video_paths: list[str] | None = None) -> list[dict[str, Any]]:
        """Upload video files now so later generation/review calls reuse them."""

        prepared = []
        for path in dict.fromkeys(video_paths or []):
            prepared.append(self._file_part(path)["file_data"])
        return prepared

    def generate(
        self,
        prompt: str,
        image_paths: list[str] | None = None,
        video_paths: list[str] | None = None,
        decoding_mode: str = "greedy",
        temperature: float = DEFAULT_SAMPLING_TEMPERATURE,
        top_p: float = DEFAULT_SAMPLING_TOP_P,
        top_k: int | None = None,
    ) -> str:
        result = self._generate(
            prompt,
            image_paths=image_paths,
            video_paths=video_paths,
            decoding_mode=decoding_mode,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        return str(result["text"])

    def generate_with_choice_logits(
        self,
        prompt: str,
        image_paths: list[str] | None = None,
        video_paths: list[str] | None = None,
        *,
        field_name: str = DEFAULT_CHOICE_FIELD,
        choices: tuple[str, ...] = DEFAULT_DECISION_CHOICES,
    ) -> dict[str, Any]:
        return self._generate(
            prompt,
            image_paths=image_paths,
            video_paths=video_paths,
            include_logprobs=True,
            choice_field=field_name,
            choices=choices,
        )

    def _generate(
        self,
        prompt: str,
        image_paths: list[str] | None = None,
        video_paths: list[str] | None = None,
        decoding_mode: str = "greedy",
        temperature: float = DEFAULT_SAMPLING_TEMPERATURE,
        top_p: float = DEFAULT_SAMPLING_TOP_P,
        top_k: int | None = None,
        *,
        include_logprobs: bool = False,
        choice_field: str = DEFAULT_CHOICE_FIELD,
        choices: tuple[str, ...] = DEFAULT_DECISION_CHOICES,
    ) -> dict[str, Any]:
        if decoding_mode not in GENERATOR_DECODING_MODES:
            raise ValueError(f"unknown decoding_mode: {decoding_mode}")
        image_paths = image_paths or []
        video_paths = video_paths or []
        start = time.time()
        print(
            "gemini_generate_start "
            f"model={self.model_id} images={len(image_paths)} videos={len(video_paths)} "
            f"prompt_chars={len(prompt)} decoding_mode={decoding_mode}",
            flush=True,
        )
        parts = [self._image_part(path) for path in image_paths]
        parts.extend(self._file_part(path) for path in video_paths)
        parts.append({"text": prompt})
        generation_config: dict[str, Any] = {
            "maxOutputTokens": self.max_new_tokens,
            "temperature": temperature if decoding_mode == "sampling" else 0,
        }
        if decoding_mode == "sampling":
            generation_config["topP"] = top_p
            if top_k is not None:
                generation_config["topK"] = top_k
        if include_logprobs:
            generation_config["responseLogprobs"] = True
            generation_config["logprobs"] = 20
        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": generation_config,
        }
        data = self._request_json(
            f"{self.base_url}/models/{self.model_id}:generateContent",
            payload=payload,
            method="POST",
        )
        candidates = data.get("candidates") if isinstance(data, dict) else None
        if not candidates:
            raise RuntimeError(f"Gemini returned no candidates: {data}")
        content = candidates[0].get("content") if isinstance(candidates[0], dict) else None
        out_parts = content.get("parts") if isinstance(content, dict) else None
        text = "".join(str(part.get("text") or "") for part in out_parts or [] if isinstance(part, dict)).strip()
        if not text:
            raise RuntimeError(f"Gemini returned no text content: {data}")
        print(
            f"gemini_generate_done model={self.model_id} seconds={time.time() - start:.1f} output_chars={len(text)}",
            flush=True,
        )
        result: dict[str, Any] = {"text": text}
        if include_logprobs:
            logprobs_result = candidates[0].get("logprobsResult") or {}
            chosen = list(logprobs_result.get("chosenCandidates") or [])
            top = list(logprobs_result.get("topCandidates") or [])
            token_texts = [str(item.get("token") or "") for item in chosen]
            top_candidates = []
            for index, row in enumerate(top):
                candidate_rows = [
                    {
                        "token": item.get("token"),
                        "logprob": item.get("logProbability"),
                    }
                    for item in row.get("candidates") or []
                    if isinstance(item, dict)
                ]
                if index < len(chosen):
                    candidate_rows.append(
                        {
                            "token": chosen[index].get("token"),
                            "logprob": chosen[index].get("logProbability"),
                        }
                    )
                top_candidates.append(candidate_rows)
            result["choice_logits"] = choice_logprobs_from_top_candidates(
                token_texts,
                top_candidates,
                field_name=choice_field,
                choices=choices,
            )
        return result


class DryRunRunner:
    """A no-model runner used only to write prompts and test plumbing."""

    model_id = "dry-run-no-model"

    def generate(
        self,
        prompt: str,
        image_paths: list[str] | None = None,
        video_paths: list[str] | None = None,
        decoding_mode: str = "greedy",
        temperature: float = DEFAULT_SAMPLING_TEMPERATURE,
        top_p: float = DEFAULT_SAMPLING_TOP_P,
        top_k: int | None = None,
    ) -> str:
        return json.dumps(
            {
                "dry_run": True,
                "prompt_preview": prompt[:1000],
                "image_count": len(image_paths or []),
                "video_count": len(video_paths or []),
                "decoding_mode": decoding_mode,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            },
            ensure_ascii=False,
        )


def make_runner(
    backend: str,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    base_url: str = "http://127.0.0.1:8000/v1",
    max_new_tokens: int = 1024,
    max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
    dtype: str = "bfloat16",
    allow_cpu: bool = False,
    allow_openai_video_input: bool = False,
    disable_thinking: bool = False,
    api_key: str | None = None,
    reasoning_effort: str | None = None,
    video_fps: float = DEFAULT_VIDEO_FPS,
    max_input_tokens: int | None = None,
    min_free_gib: float = 0.0,
    kv_bytes_per_token: int = 0,
    min_available_ram_gib: float = 0.0,
    attn_implementation: str = "sdpa",
    device_map: str = "auto",
) -> Generator:
    if backend == "transformers-local":
        return Qwen3VLTransformersRunner(
            model_id,
            max_new_tokens=max_new_tokens,
            max_image_pixels=max_image_pixels,
            dtype=dtype,
            allow_cpu=allow_cpu,
            disable_thinking=disable_thinking,
            video_fps=video_fps,
            max_input_tokens=max_input_tokens,
            min_free_gib=min_free_gib,
            kv_bytes_per_token=kv_bytes_per_token,
            min_available_ram_gib=min_available_ram_gib,
            attn_implementation=attn_implementation,
            device_map=device_map,
        )
    if backend == MEMORY_SAFE_BACKEND:
        return Qwen3VLMemorySafeTransformersRunner(
            model_id,
            max_new_tokens=max_new_tokens,
            max_image_pixels=max_image_pixels,
            dtype=dtype,
            allow_cpu=allow_cpu,
            disable_thinking=disable_thinking,
        )
    if backend == "openai-compatible-local":
        return OpenAICompatibleLocalRunner(
            model_id,
            base_url=base_url,
            max_new_tokens=max_new_tokens,
            api_key=api_key,
            allow_video_input=allow_openai_video_input,
        )
    if backend == "openrouter":
        effective_base_url = (
            DEFAULT_OPENROUTER_BASE_URL
            if base_url == "http://127.0.0.1:8000/v1"
            else base_url
        )
        effective_model_id = (
            DEFAULT_OPENROUTER_MODEL_ID if model_id == DEFAULT_MODEL_ID else model_id
        )
        return OpenRouterRunner(
            effective_model_id,
            base_url=effective_base_url,
            max_new_tokens=max_new_tokens,
            api_key=api_key,
            allow_video_input=allow_openai_video_input,
            reasoning_effort=reasoning_effort,
        )
    if backend == "gemini":
        effective_base_url = (
            DEFAULT_GEMINI_BASE_URL
            if base_url == "http://127.0.0.1:8000/v1"
            else base_url
        )
        effective_model_id = DEFAULT_GEMINI_MODEL_ID if model_id == DEFAULT_MODEL_ID else model_id
        return GeminiRunner(
            effective_model_id,
            base_url=effective_base_url,
            max_new_tokens=max_new_tokens,
            api_key=api_key,
        )
    if backend == "dry-run":
        return DryRunRunner()
    raise ValueError(f"Unsupported backend: {backend}")
