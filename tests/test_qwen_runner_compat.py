from concurrent.futures import ThreadPoolExecutor
import inspect
from pathlib import Path
from types import SimpleNamespace
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
    MEMORY_SAFE_DEFAULT_VIDEO_FPS,
    Qwen3VLMemorySafeTransformersRunner,
    Qwen3VLTransformersRunner,
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
        self.video_fps = kwargs["video_fps"]
        self.max_input_tokens = kwargs["max_input_tokens"]
        self.min_free_gib = kwargs["min_free_gib"]
        self.kv_bytes_per_token = kwargs["kv_bytes_per_token"]
        self.min_available_ram_gib = kwargs["min_available_ram_gib"]
        self.attn_implementation = kwargs["attn_implementation"]
        self.torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))

    for name in (
        "QWEN_MEMORY_SAFE_VIDEO_FPS",
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
    )

    assert result["content"] == [
        {"type": "video", "video": "one.mp4", "fps": 0.5, "max_pixels": 131_072},
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
