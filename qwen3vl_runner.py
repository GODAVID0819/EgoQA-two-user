"""Qwen3-VL runner backends for local/open-source inference."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
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
DEFAULT_MAX_IMAGE_PIXELS = 262144
GENERATOR_DECODING_MODES = ("greedy", "sampling")
DEFAULT_SAMPLING_TEMPERATURE = 0.7
DEFAULT_SAMPLING_TOP_P = 0.9


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


def load_transformers_model(model_id: str, dtype: str = "bfloat16"):
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
            "device_map": "auto",
            "attn_implementation": "sdpa",
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


def apply_chat_template_compat(processor: Any, messages: list[dict[str, Any]], *, disable_thinking: bool) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if disable_thinking:
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
        self.process_vision_info = process_vision_info
        start = time.time()
        print(f"loading_processor={model_id}", flush=True)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        print(f"loading_model={model_id}", flush=True)
        self.model = load_transformers_model(model_id, dtype=dtype)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.torch = torch
        print(f"model_first_param_device={self.device}", flush=True)
        print(f"model_loaded_seconds={time.time() - start:.1f}", flush=True)

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
        image_paths = image_paths or []
        video_paths = video_paths or []
        content: list[dict[str, Any]] = [
            {"type": "image", "image": image_path, "max_pixels": self.max_image_pixels}
            for image_path in image_paths
        ]
        content.extend(
            {"type": "video", "video": video_path, "max_pixels": self.max_image_pixels, "fps": 1.0}
            for video_path in video_paths
        )
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        start = time.time()
        print(
            "qwen_generate_start "
            f"images={len(image_paths)} videos={len(video_paths)} "
            f"prompt_chars={len(prompt)} disable_thinking={self.disable_thinking} "
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
        ).to(self.device)
        encode_seconds = time.time() - start
        input_tokens = int(inputs.input_ids.shape[-1]) if hasattr(inputs, "input_ids") else -1
        inputs.pop("video_metadata", None)
        print(
            f"qwen_processor_encoded_seconds={encode_seconds:.1f} input_tokens={input_tokens}",
            flush=True,
        )
        generate_kwargs = generation_kwargs(
            max_new_tokens=self.max_new_tokens,
            decoding_mode=decoding_mode,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
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
        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()


class OpenAICompatibleLocalRunner:
    """Call a local vLLM/SGLang/llama.cpp OpenAI-compatible server."""

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
        return data["choices"][0]["message"]["content"].strip()


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
    ) -> None:
        self.model_id = model_id or DEFAULT_GEMINI_MODEL_ID
        self.base_url = base_url.rstrip("/")
        self.upload_url = upload_url.rstrip("/")
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.file_poll_interval_seconds = file_poll_interval_seconds
        self.file_poll_timeout_seconds = file_poll_timeout_seconds
        self._file_cache: dict[tuple[str, int, int], dict[str, Any]] = {}
        self._file_cache_lock = threading.Lock()
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
        return read_json_response(req, timeout=self.timeout)

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
        return text


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
) -> Generator:
    if backend == "transformers-local":
        return Qwen3VLTransformersRunner(
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
