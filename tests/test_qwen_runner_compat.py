from concurrent.futures import ThreadPoolExecutor
import inspect
from pathlib import Path
from types import SimpleNamespace
import sys
import threading
import time

import pytest

import egolife_two_user_qa.qwen3vl_runner as qwen_runner_module
from egolife_two_user_qa.qwen3vl_runner import apply_chat_template_compat
from egolife_two_user_qa.qwen3vl_runner import (
    MEMORY_SAFE_DEFAULT_ATTN_IMPLEMENTATION,
    MEMORY_SAFE_DEFAULT_KV_BYTES_PER_TOKEN,
    MEMORY_SAFE_DEFAULT_MAX_INPUT_TOKENS,
    MEMORY_SAFE_DEFAULT_MIN_AVAILABLE_RAM_GIB,
    MEMORY_SAFE_DEFAULT_MIN_FREE_GIB,
    MEMORY_SAFE_DEFAULT_MIN_VIDEO_PIXELS,
    MEMORY_SAFE_DEFAULT_VIDEO_FPS,
    Qwen3VLLocalVLLMRunner,
    Qwen3VLMemorySafeTransformersRunner,
    Qwen3VLTransformersRunner,
    OpenAICompatibleLocalRunner,
    OpenRouterRunner,
)


MESSAGES = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]


def test_new_processor_receives_thinking_flag_in_template_kwargs() -> None:
    class StructuredProcessor:
        def __init__(self) -> None:
            self.kwargs = None

        def apply_chat_template(
            self,
            messages,
            **kwargs: "Unpack[AllKwargsForChatTemplate]",
        ) -> str:
            self.kwargs = kwargs
            return "structured"

    processor = StructuredProcessor()
    assert apply_chat_template_compat(processor, MESSAGES, disable_thinking=True) == "structured"
    assert processor.kwargs == {
        "tokenize": False,
        "add_generation_prompt": True,
        "template_kwargs": {"enable_thinking": False},
    }


def test_legacy_processor_receives_direct_thinking_flag() -> None:
    class LegacyProcessor:
        def __init__(self) -> None:
            self.kwargs = None

        def apply_chat_template(self, messages, **kwargs) -> str:
            self.kwargs = kwargs
            return "legacy"

    processor = LegacyProcessor()
    assert apply_chat_template_compat(processor, MESSAGES, disable_thinking=True) == "legacy"
    assert processor.kwargs == {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }


def test_thinking_enabled_does_not_add_provider_specific_kwargs() -> None:
    class Processor:
        def __init__(self) -> None:
            self.kwargs = None

        def apply_chat_template(self, messages, **kwargs) -> str:
            self.kwargs = kwargs
            return "default"

    processor = Processor()
    assert apply_chat_template_compat(processor, MESSAGES, disable_thinking=False) == "default"
    assert processor.kwargs == {
        "tokenize": False,
        "add_generation_prompt": True,
    }


def test_original_local_runner_defaults_remain_unchanged() -> None:
    parameters = inspect.signature(Qwen3VLTransformersRunner.__init__).parameters

    assert parameters["video_fps"].default == 1.0
    assert parameters["max_input_tokens"].default is None
    assert parameters["min_free_gib"].default == 0.0
    assert parameters["kv_bytes_per_token"].default == 0
    assert parameters["min_available_ram_gib"].default == 0.0
    assert parameters["attn_implementation"].default == "sdpa"


def test_memory_safe_runner_has_long_context_guards_by_default(monkeypatch) -> None:
    captured = {}

    def fake_base_init(self, model_id, **kwargs):
        captured.update(kwargs)
        self.model_id = model_id
        self.max_image_pixels = kwargs["max_image_pixels"]
        self.min_video_pixels = kwargs["min_video_pixels"]
        self.video_fps = kwargs["video_fps"]
        self.max_input_tokens = kwargs["max_input_tokens"]
        self.min_free_gib = kwargs["min_free_gib"]
        self.kv_bytes_per_token = kwargs["kv_bytes_per_token"]
        self.min_available_ram_gib = kwargs["min_available_ram_gib"]
        self.attn_implementation = kwargs["attn_implementation"]
        self.device_map = kwargs["device_map"]
        self.required_cuda_device_count = kwargs["required_cuda_device_count"]
        self.cuda_devices = ()
        self.torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))

    for name in (
        "QWEN_MEMORY_SAFE_VIDEO_FPS",
        "QWEN_MEMORY_SAFE_MIN_VIDEO_PIXELS",
        "QWEN_MEMORY_SAFE_MAX_INPUT_TOKENS",
        "QWEN_MEMORY_SAFE_GPU_RESERVE_GIB",
        "QWEN_MEMORY_SAFE_MIN_FREE_GIB",
        "QWEN_MEMORY_SAFE_MIN_AVAILABLE_RAM_GIB",
        "QWEN_MEMORY_SAFE_ATTN_IMPLEMENTATION",
        "QWEN_MEMORY_SAFE_TRANSCODE_LOCAL_VIDEOS",
        "QWEN_MEMORY_SAFE_TRANSCODE_MAX_EDGE",
        "QWEN_MEMORY_SAFE_TRANSCODE_CRF",
        "QWEN_MEMORY_SAFE_VIDEO_CACHE_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(Qwen3VLTransformersRunner, "__init__", fake_base_init)

    runner = Qwen3VLMemorySafeTransformersRunner("Qwen/Qwen3.6-27B")

    assert runner.model_id == "Qwen/Qwen3.6-27B"
    assert captured["video_fps"] == MEMORY_SAFE_DEFAULT_VIDEO_FPS
    assert captured["min_video_pixels"] == MEMORY_SAFE_DEFAULT_MIN_VIDEO_PIXELS
    assert captured["min_video_pixels"] <= captured["max_image_pixels"]
    assert captured["max_input_tokens"] == MEMORY_SAFE_DEFAULT_MAX_INPUT_TOKENS
    assert captured["min_free_gib"] == MEMORY_SAFE_DEFAULT_MIN_FREE_GIB
    assert captured["kv_bytes_per_token"] == MEMORY_SAFE_DEFAULT_KV_BYTES_PER_TOKEN
    assert captured["min_available_ram_gib"] == MEMORY_SAFE_DEFAULT_MIN_AVAILABLE_RAM_GIB
    assert captured["attn_implementation"] == MEMORY_SAFE_DEFAULT_ATTN_IMPLEMENTATION


def test_memory_safe_runner_serializes_complete_generate_calls(monkeypatch) -> None:
    runner = Qwen3VLMemorySafeTransformersRunner.__new__(
        Qwen3VLMemorySafeTransformersRunner
    )
    runner._inference_lock = threading.Lock()
    runner.torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    runner.cuda_devices = ()
    runner.min_available_ram_gib = 0.0
    runner.transcode_local_videos = False
    active = 0
    max_active = 0
    state_lock = threading.Lock()

    def fake_generate(self, marker):
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.04)
        with state_lock:
            active -= 1
        return {"text": marker}

    monkeypatch.setattr(Qwen3VLTransformersRunner, "_generate", fake_generate)
    monkeypatch.setattr(
        qwen_runner_module,
        "release_unused_host_memory",
        lambda: (0, 0, False),
    )

    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(runner._generate, ["one", "two", "three"]))

    assert [result["text"] for result in results] == ["one", "two", "three"]
    assert max_active == 1


def test_memory_safe_runner_caps_explicit_multimodal_video_budget(monkeypatch) -> None:
    runner = Qwen3VLMemorySafeTransformersRunner.__new__(
        Qwen3VLMemorySafeTransformersRunner
    )
    runner._inference_lock = threading.Lock()
    runner.torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    runner.cuda_devices = ()
    runner.video_fps = 0.5
    runner.max_image_pixels = 131_072
    runner.min_available_ram_gib = 0.0
    runner.transcode_local_videos = False

    def fake_generate(self, *args, **kwargs):
        return {"text": "ok", "content": kwargs["multimodal_content"]}

    monkeypatch.setattr(Qwen3VLTransformersRunner, "_generate", fake_generate)
    monkeypatch.setattr(
        qwen_runner_module,
        "release_unused_host_memory",
        lambda: (0, 0, False),
    )

    result = runner._generate(
        "",
        multimodal_content=[
            {"type": "video", "video": "one.mp4", "fps": 1.0, "max_pixels": 262_144},
            {"type": "text", "text": "Question"},
        ],
    )

    assert result["content"] == [
        {
            "type": "video",
            "video": str(Path("one.mp4").resolve()),
            "fps": 0.5,
            "max_pixels": 131_072,
        },
        {"type": "text", "text": "Question"},
    ]


def test_memory_safe_vram_requirement_is_kv_plus_five_gib_reserve() -> None:
    runner = Qwen3VLTransformersRunner.__new__(Qwen3VLTransformersRunner)
    runner.max_new_tokens = 2_048
    runner.kv_bytes_per_token = MEMORY_SAFE_DEFAULT_KV_BYTES_PER_TOKEN
    runner.min_free_gib = MEMORY_SAFE_DEFAULT_MIN_FREE_GIB

    assert runner._estimated_kv_gib(input_tokens=84_992) == pytest.approx(5.3125)
    assert runner._required_free_vram_gib(input_tokens=84_992) == pytest.approx(10.3125)
    assert runner._required_free_vram_gib(input_tokens=131_072) == pytest.approx(13.125)


def test_memory_safe_host_ram_guard_fails_before_decode(monkeypatch) -> None:
    runner = Qwen3VLTransformersRunner.__new__(Qwen3VLTransformersRunner)
    runner.min_available_ram_gib = 16.0
    monkeypatch.setattr(
        qwen_runner_module,
        "available_host_memory_bytes",
        lambda: 8 * 1024**3,
    )

    with pytest.raises(RuntimeError, match="Insufficient available host RAM"):
        runner._enforce_available_host_memory(stage="before_video_decode")


def test_memory_safe_runner_physically_caches_low_fps_decoder_input(
    tmp_path,
    monkeypatch,
) -> None:
    source = tmp_path / "ten_minutes.mp4"
    source.write_bytes(b"source-video")
    runner = Qwen3VLMemorySafeTransformersRunner.__new__(
        Qwen3VLMemorySafeTransformersRunner
    )
    runner.transcode_local_videos = True
    runner.video_fps = 1.0
    runner.transcode_max_edge = 512
    runner.transcode_crf = 23
    runner.transcode_cache_dir = tmp_path / "cache"
    calls = []

    def fake_run(command, *, check, capture_output, text):
        assert check is True
        assert capture_output is True
        assert text is True
        calls.append(command)
        Path(command[-1]).write_bytes(b"one-fps-video")

    monkeypatch.setattr(qwen_runner_module.subprocess, "run", fake_run)

    first = runner._prepare_video_for_memory_safe_decode(source)
    second = runner._prepare_video_for_memory_safe_decode(source)

    assert first == second
    assert Path(first).read_bytes() == b"one-fps-video"
    assert len(calls) == 1
    filters = calls[0][calls[0].index("-vf") + 1]
    assert "fps=1" in filters
    assert "scale=512:512:force_original_aspect_ratio=decrease" in filters


def test_vllm_runner_uses_tp_flash_attention_and_cached_local_media(
    tmp_path,
    monkeypatch,
) -> None:
    captured = {}

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeLLM:
        def __init__(self, **kwargs):
            captured["engine_kwargs"] = kwargs
            captured["engine"] = self
            self.enqueue_calls = []
            self.pending = []
            self.wait_batch_sizes = []

        def get_tokenizer(self):
            return SimpleNamespace(name="fake-tokenizer")

        def enqueue_chat(self, messages, **kwargs):
            external_request_id = str(len(self.enqueue_calls))
            internal_request_id = f"{external_request_id}-deadbeef"
            self.enqueue_calls.append((messages, kwargs))
            self.pending.append(
                SimpleNamespace(
                    request_id=external_request_id,
                    prompt_token_ids=[1, 2, 3],
                    outputs=[
                        SimpleNamespace(text="  accepted  ", token_ids=[4, 5])
                    ],
                )
            )
            return [internal_request_id]

        def wait_for_completion(self, **kwargs):
            self.wait_batch_sizes.append(len(self.pending))
            outputs = self.pending
            self.pending = []
            return outputs

    fake_vllm = SimpleNamespace(LLM=FakeLLM, SamplingParams=FakeSamplingParams)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setattr(qwen_runner_module, "cuda_available", lambda: True)

    environment = {
        "VLLM_ALLOWED_LOCAL_MEDIA_PATH": str(tmp_path),
        "VLLM_TENSOR_PARALLEL_SIZE": "2",
        "VLLM_GPU_MEMORY_UTILIZATION": "0.9",
        "VLLM_MAX_MODEL_LEN": "262144",
        "VLLM_MAX_NUM_SEQS": "8",
        "VLLM_BATCH_WAIT_MS": "100",
        "VLLM_MAX_NUM_BATCHED_TOKENS": "32768",
        "VLLM_MAX_IMAGES": "3600",
        "VLLM_MAX_VIDEOS": "20",
        "VLLM_MM_PROCESSOR_CACHE_GB": "8",
        "VLLM_MM_PROCESSOR_CACHE_TYPE": "shm",
        "VLLM_MM_ENCODER_TP_MODE": "data",
        "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
        "VLLM_MM_ENCODER_ATTN_BACKEND": "FLASH_ATTN",
        "VLLM_GDN_PREFILL_BACKEND": "flashinfer",
        "VLLM_MTP_SPECULATIVE_TOKENS": "1",
        "VLLM_ENABLE_PREFIX_CACHING": "0",
        "VLLM_VIDEO_FPS": "0.5",
        "VLLM_MIN_IMAGE_PIXELS": "3136",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    image = tmp_path / "frame 001.png"
    video = tmp_path / "segment 001.mp4"
    image.write_bytes(b"image")
    video.write_bytes(b"video")

    runner = Qwen3VLLocalVLLMRunner(
        "Qwen/Qwen3.8-27B",
        max_new_tokens=128,
        max_image_pixels=65_536,
        dtype="bfloat16",
        disable_thinking=True,
    )

    engine_kwargs = captured["engine_kwargs"]
    assert runner.supports_vllm_randomized_request_ids is True
    assert runner.caps_video_pixels_per_frame is True
    assert engine_kwargs["tensor_parallel_size"] == 2
    assert engine_kwargs["attention_backend"] == "FLASH_ATTN"
    assert engine_kwargs["mm_encoder_attn_backend"] == "FLASH_ATTN"
    assert engine_kwargs["gdn_prefill_backend"] == "flashinfer"
    assert engine_kwargs["mm_encoder_tp_mode"] == "data"
    assert engine_kwargs["mm_processor_cache_type"] == "shm"
    assert engine_kwargs["mm_processor_kwargs"]["cap_pixels_per_frame"] is True
    assert engine_kwargs["max_num_seqs"] == 8
    assert engine_kwargs["max_num_batched_tokens"] == 32_768
    assert engine_kwargs["enable_chunked_prefill"] is True
    assert engine_kwargs["enable_prefix_caching"] is False
    assert engine_kwargs["skip_mm_profiling"] is True
    assert engine_kwargs["speculative_config"] == {
        "method": "mtp",
        "num_speculative_tokens": 1,
    }

    assert runner.generate(
        "Review this evidence.",
        image_paths=[str(image)],
        video_paths=[str(video)],
        decoding_mode="sampling",
        temperature=0.7,
        top_p=0.9,
        top_k=40,
    ) == "accepted"

    messages, call_kwargs = captured["engine"].enqueue_calls[0]
    content = messages[0]["content"]
    assert content[0]["image_url"]["url"] == image.resolve().as_uri()
    assert content[1]["video_url"]["url"] == video.resolve().as_uri()
    assert content[0]["uuid"] == runner._media_uuid(image.resolve())
    assert content[1]["uuid"] == runner._media_uuid(video.resolve())
    assert call_kwargs["chat_template_kwargs"] == {"enable_thinking": False}
    assert call_kwargs["mm_processor_kwargs"]["fps"] == 0.5
    assert call_kwargs["mm_processor_kwargs"]["cap_pixels_per_frame"] is True
    assert call_kwargs["sampling_params"].kwargs == {
        "max_tokens": 128,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
    }

    assert runner.begin_concurrent_batch(3) is True
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(runner.generate, f"Judge branch {index}.") for index in range(3)]
        release = runner.release_concurrent_batch(timeout_seconds=2.0)
        results = [future.result() for future in futures]

    assert release == 3
    assert results == ["accepted", "accepted", "accepted"]
    assert captured["engine"].wait_batch_sizes == [1, 3]
    assert len(captured["engine"].enqueue_calls) == 4
    assert not hasattr(captured["engine"], "chat_calls")


def test_local_vllm_server_runner_sends_concurrent_zero_copy_media_requests(
    tmp_path,
    monkeypatch,
) -> None:
    image = tmp_path / "frame 001.png"
    video = tmp_path / "segment 001.mp4"
    image.write_bytes(b"image")
    video.write_bytes(b"video")
    monkeypatch.setenv("VLLM_OPENAI_USE_LOCAL_MEDIA_URIS", "1")
    monkeypatch.setenv("VLLM_ALLOWED_LOCAL_MEDIA_PATH", str(tmp_path))
    monkeypatch.setenv("VLLM_VIDEO_FPS", "0.5")
    monkeypatch.setenv("VLLM_MIN_IMAGE_PIXELS", "3136")

    active = 0
    max_active = 0
    state_lock = threading.Lock()
    payloads = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":" accepted "}}]}'

    def fake_urlopen(request, timeout):
        nonlocal active, max_active
        assert timeout == 3600
        payloads.append(qwen_runner_module.json.loads(request.data.decode("utf-8")))
        with state_lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.04)
        with state_lock:
            active -= 1
        return FakeResponse()

    monkeypatch.setattr(qwen_runner_module.urllib.request, "urlopen", fake_urlopen)
    runner = OpenAICompatibleLocalRunner(
        "Qwen/Qwen3.8-27B",
        allow_video_input=True,
        disable_thinking=True,
        max_image_pixels=65_536,
    )

    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(
            executor.map(
                lambda index: runner.generate(
                    f"Judge {index}",
                    image_paths=[str(image)],
                    video_paths=[str(video)],
                ),
                range(3),
            )
        )

    assert results == ["accepted", "accepted", "accepted"]
    assert max_active == 3
    assert runner.supports_concurrent_batching is True
    assert runner.uses_async_multimodal_frontend is True
    assert OpenRouterRunner.supports_concurrent_batching is False
    assert len(payloads) == 3
    for payload in payloads:
        content = payload["messages"][0]["content"]
        assert content[0]["image_url"]["url"] == image.resolve().as_uri()
        assert content[1]["video_url"]["url"] == video.resolve().as_uri()
        assert content[2]["type"] == "text"
        assert content[2]["text"].startswith("Judge ")
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        assert payload["mm_processor_kwargs"] == {
            "min_pixels": 3136,
            "max_pixels": 65536,
            "fps": 0.5,
            "cap_pixels_per_frame": True,
        }
