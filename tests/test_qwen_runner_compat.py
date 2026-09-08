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
    GenerationCallProfile,
    MEMORY_SAFE_DEFAULT_ATTN_IMPLEMENTATION,
    MEMORY_SAFE_DEFAULT_KV_BYTES_PER_TOKEN,
    MEMORY_SAFE_DEFAULT_MAX_INPUT_TOKENS,
    MEMORY_SAFE_DEFAULT_MIN_AVAILABLE_RAM_GIB,
    MEMORY_SAFE_DEFAULT_MIN_FREE_GIB,
    MEMORY_SAFE_DEFAULT_VIDEO_FPS,
    GeminiRunner,
    OpenAICompatibleLocalRunner,
    Qwen3VLMemorySafeTransformersRunner,
    Qwen3VLTransformersRunner,
)


MESSAGES = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]


def test_generation_call_profile_validates_output_budget() -> None:
    profile = GenerationCallProfile(max_new_tokens=8192, disable_thinking=False)

    assert profile.max_new_tokens == 8192
    assert profile.disable_thinking is False


def test_generation_call_profile_can_override_video_quality_per_stage() -> None:
    profile = GenerationCallProfile(
        max_new_tokens=8192,
        disable_thinking=True,
        video_fps=0.5,
        max_image_pixels=131_072,
    )

    assert profile.video_fps == 0.5
    assert profile.max_image_pixels == 131_072


def test_generation_call_profile_rejects_non_positive_budget() -> None:
    with pytest.raises(ValueError, match="max_new_tokens must be positive"):
        GenerationCallProfile(max_new_tokens=0, disable_thinking=False)


def test_local_runner_generate_accepts_per_call_profile() -> None:
    parameters = inspect.signature(Qwen3VLTransformersRunner.generate).parameters

    assert "call_profile" in parameters


def test_api_runners_accept_per_call_profile() -> None:
    assert "call_profile" in inspect.signature(
        OpenAICompatibleLocalRunner.generate
    ).parameters
    assert "call_profile" in inspect.signature(GeminiRunner.generate).parameters


def test_vllm_runner_uses_flash_kernels_batching_and_per_call_profile(
    tmp_path,
    monkeypatch,
) -> None:
    runner_class = getattr(qwen_runner_module, "Qwen3VLLocalVLLMRunner")
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
            self.enqueue_calls.append((messages, kwargs))
            self.pending.append(
                SimpleNamespace(
                    request_id=external_request_id,
                    prompt_token_ids=[1, 2, 3],
                    outputs=[SimpleNamespace(text="  accepted  ", token_ids=[4, 5])],
                )
            )
            return [f"{external_request_id}-deadbeef"]

        def wait_for_completion(self, **kwargs):
            self.wait_batch_sizes.append(len(self.pending))
            outputs = self.pending
            self.pending = []
            return outputs

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(LLM=FakeLLM, SamplingParams=FakeSamplingParams),
    )
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

    image = tmp_path / "frame.png"
    video = tmp_path / "segment.mp4"
    image.write_bytes(b"image")
    video.write_bytes(b"video")
    runner = runner_class(
        "Qwen/Qwen3.8-27B",
        max_new_tokens=128,
        max_image_pixels=65_536,
        dtype="bfloat16",
        disable_thinking=False,
    )

    engine_kwargs = captured["engine_kwargs"]
    assert runner.supports_vllm_randomized_request_ids is True
    assert runner.caps_video_pixels_per_frame is True
    assert engine_kwargs["tensor_parallel_size"] == 2
    assert engine_kwargs["attention_backend"] == "FLASH_ATTN"
    assert engine_kwargs["mm_encoder_attn_backend"] == "FLASH_ATTN"
    assert engine_kwargs["gdn_prefill_backend"] == "flashinfer"
    assert engine_kwargs["enable_chunked_prefill"] is True
    assert engine_kwargs["skip_mm_profiling"] is True
    assert engine_kwargs["mm_processor_kwargs"]["cap_pixels_per_frame"] is True
    assert engine_kwargs["speculative_config"] == {
        "method": "mtp",
        "num_speculative_tokens": 1,
    }

    profile = GenerationCallProfile(
        max_new_tokens=64,
        disable_thinking=True,
        video_fps=0.25,
        max_image_pixels=32_768,
    )
    assert runner.generate(
        "Review this evidence.",
        image_paths=[str(image)],
        video_paths=[str(video)],
        decoding_mode="sampling",
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        call_profile=profile,
    ) == "accepted"

    messages, call_kwargs = captured["engine"].enqueue_calls[0]
    content = messages[0]["content"]
    assert content[0]["uuid"] == runner._media_uuid(
        image.resolve(), modality="image", video_fps=0.25, max_pixels=32_768
    )
    assert content[0]["uuid"] != runner._media_uuid(
        image.resolve(), modality="image", video_fps=0.5, max_pixels=65_536
    )
    assert call_kwargs["chat_template_kwargs"] == {"enable_thinking": False}
    assert call_kwargs["mm_processor_kwargs"]["fps"] == 0.25
    assert call_kwargs["mm_processor_kwargs"]["max_pixels"] == 32_768
    assert call_kwargs["sampling_params"].kwargs == {
        "max_tokens": 64,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
    }

    assert runner.begin_concurrent_batch(3) is True
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(runner.generate, f"Judge branch {index}.")
            for index in range(3)
        ]
        release = runner.release_concurrent_batch(timeout_seconds=2.0)
        results = [future.result() for future in futures]

    assert release == 3
    assert results == ["accepted", "accepted", "accepted"]
    assert captured["engine"].wait_batch_sizes == [1, 3]


def test_make_runner_registers_vllm_backend(monkeypatch) -> None:
    sentinel = object()
    runner_class = getattr(qwen_runner_module, "Qwen3VLLocalVLLMRunner")
    monkeypatch.setattr(qwen_runner_module, "Qwen3VLLocalVLLMRunner", lambda *a, **k: sentinel)

    assert qwen_runner_module.make_runner("vllm-local") is sentinel
    assert runner_class is not None


def test_memory_safe_vram_estimate_uses_per_call_output_budget() -> None:
    runner = Qwen3VLTransformersRunner.__new__(Qwen3VLTransformersRunner)
    runner.kv_bytes_per_token = MEMORY_SAFE_DEFAULT_KV_BYTES_PER_TOKEN

    assert runner._estimated_kv_gib(
        input_tokens=84_992,
        max_new_tokens=8_192,
    ) == pytest.approx(5.6875)


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
        self.min_video_pixels = kwargs.get("min_video_pixels")
        self.video_fps = kwargs["video_fps"]
        self.max_input_tokens = kwargs["max_input_tokens"]
        self.min_free_gib = kwargs["min_free_gib"]
        self.kv_bytes_per_token = kwargs["kv_bytes_per_token"]
        self.min_available_ram_gib = kwargs["min_available_ram_gib"]
        self.attn_implementation = kwargs["attn_implementation"]
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
    assert captured["min_video_pixels"] is None
    assert captured["max_input_tokens"] == MEMORY_SAFE_DEFAULT_MAX_INPUT_TOKENS
    assert captured["min_free_gib"] == MEMORY_SAFE_DEFAULT_MIN_FREE_GIB
    assert captured["kv_bytes_per_token"] == MEMORY_SAFE_DEFAULT_KV_BYTES_PER_TOKEN
    assert captured["min_available_ram_gib"] == MEMORY_SAFE_DEFAULT_MIN_AVAILABLE_RAM_GIB
    assert captured["attn_implementation"] == MEMORY_SAFE_DEFAULT_ATTN_IMPLEMENTATION


def test_memory_safe_runner_passes_explicit_video_minimum(monkeypatch) -> None:
    captured = {}

    def fake_base_init(self, model_id, **kwargs):
        captured.update(kwargs)
        self.model_id = model_id
        self.max_image_pixels = kwargs["max_image_pixels"]
        self.min_video_pixels = kwargs.get("min_video_pixels")
        self.video_fps = kwargs["video_fps"]
        self.max_input_tokens = kwargs["max_input_tokens"]
        self.min_free_gib = kwargs["min_free_gib"]
        self.kv_bytes_per_token = kwargs["kv_bytes_per_token"]
        self.min_available_ram_gib = kwargs["min_available_ram_gib"]
        self.attn_implementation = kwargs["attn_implementation"]
        self.torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))

    monkeypatch.setenv("QWEN_MEMORY_SAFE_MIN_VIDEO_PIXELS", "3136")
    monkeypatch.setattr(Qwen3VLTransformersRunner, "__init__", fake_base_init)

    runner = Qwen3VLMemorySafeTransformersRunner(
        "Qwen/Qwen3.6-27B",
        max_image_pixels=65_536,
    )

    assert runner.min_video_pixels == 3_136
    assert captured["min_video_pixels"] == 3_136
    assert captured["max_image_pixels"] == 65_536


def test_memory_safe_runner_serializes_complete_generate_calls(monkeypatch) -> None:
    runner = Qwen3VLMemorySafeTransformersRunner.__new__(
        Qwen3VLMemorySafeTransformersRunner
    )
    runner._inference_lock = threading.Lock()
    runner.torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
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
    runner.video_fps = 0.5
    runner.max_image_pixels = 131_072
    runner.min_available_ram_gib = 0.0
    runner.transcode_local_videos = False

    def fake_generate(self, *args, **kwargs):
        return {"text": "ok", "content": kwargs["multimodal_content"]}

    monkeypatch.setattr(Qwen3VLTransformersRunner, "_generate", fake_generate)

    result = runner._generate(
        "",
        multimodal_content=[
            {"type": "video", "video": "one.mp4", "fps": 1.0, "max_pixels": 262_144},
            {"type": "text", "text": "Question"},
        ],
        call_profile=GenerationCallProfile(
            max_new_tokens=2048,
            disable_thinking=True,
            video_fps=0.25,
            max_image_pixels=65_536,
        ),
    )

    assert result["content"] == [
        {
            "type": "video",
            "video": str(Path("one.mp4").resolve()),
            "fps": 0.25,
            "max_pixels": 65_536,
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
