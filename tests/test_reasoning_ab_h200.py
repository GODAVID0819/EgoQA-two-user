from __future__ import annotations

from pathlib import Path
import json

import egolife_two_user_qa.qwen3vl_runner as qwen_runner
import egolife_two_user_qa.video_qa_loop as video_qa_loop


ROOT = Path(__file__).resolve().parents[1]
NR_JOB = ROOT / "hpc/qa/experiments/run_six_user_qa_reasoning_nr_h200.sbatch"
R_JOB = ROOT / "hpc/qa/experiments/run_six_user_qa_reasoning_r_h200.sbatch"
COMMON = ROOT / "hpc/qa/experiments/run_six_user_qa_reasoning_ab_h200_common.sh"
KEEPER = ROOT / "hpc/shared/cuda.py"


class MetadataRunner:
    model_id = "test-model"

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate_with_metadata(self, prompt: str, **kwargs: object) -> dict[str, object]:
        self.calls.append({"prompt": prompt, **kwargs})
        index = len(self.calls)
        return {
            "text": '{"ok": true}' if index > 1 else "reasoning draft",
            "prompt_tokens": 100 * index,
            "completion_tokens": 10 * index,
            "elapsed_seconds": float(index),
        }


def _profiles(mode: str):
    factory = getattr(video_qa_loop, "reasoning_ab_stage_profiles", None)
    assert callable(factory), "缺少 reasoning_ab_stage_profiles"
    return factory(mode)


def _run_stage(mode: str, runner: MetadataRunner) -> dict[str, object]:
    function = getattr(video_qa_loop, "run_profiled_structured_stage", None)
    assert callable(function), "缺少 run_profiled_structured_stage"
    return function(
        runner=runner,
        task_prompt="Return JSON",
        output_schema={"ok": True},
        stage_name="generator",
        image_paths=["frame.jpg"],
        video_paths=["clip.mp4"],
        profile=_profiles(mode)["generator"],
        generation_kwargs={"decoding_mode": "sampling", "temperature": 0.7},
    )


def test_nr_is_one_non_reasoning_structured_call() -> None:
    runner = MetadataRunner()
    result = _run_stage("nr", runner)

    assert len(runner.calls) == 1
    call = runner.calls[0]
    assert call["image_paths"] == ["frame.jpg"]
    assert call["video_paths"] == ["clip.mp4"]
    profile = call["call_profile"]
    assert isinstance(profile, qwen_runner.GenerationCallProfile)
    assert profile.disable_thinking is True
    assert profile.max_new_tokens == 2048
    assert result["reasoning_mode"] == "nr"
    assert result["execution_mode"] == "single_call"
    assert result["reasoning_prompt_tokens"] is None
    assert result["finalizer_prompt_tokens"] is None


def test_r_reasons_with_full_media_then_finalizes_text_only() -> None:
    runner = MetadataRunner()
    result = _run_stage("r", runner)

    assert len(runner.calls) == 2
    reasoning, finalizer = runner.calls
    assert reasoning["image_paths"] == ["frame.jpg"]
    assert reasoning["video_paths"] == ["clip.mp4"]
    assert reasoning["call_profile"].disable_thinking is False
    assert finalizer["image_paths"] == []
    assert finalizer["video_paths"] == []
    assert finalizer["call_profile"].disable_thinking is True
    assert result["reasoning_mode"] == "r"
    assert result["execution_mode"] == "reasoned_then_finalize"
    assert result["reasoning_prompt_tokens"] == 100
    assert result["reasoning_completion_tokens"] == 10
    assert result["reasoning_elapsed_seconds"] == 1.0
    assert result["finalizer_prompt_tokens"] == 200
    assert result["finalizer_completion_tokens"] == 20
    assert result["finalizer_elapsed_seconds"] == 2.0


def test_reasoning_profiles_do_not_override_frozen_media_settings() -> None:
    for mode in ("nr", "r"):
        for stage_profile in _profiles(mode).values():
            for call_profile in (
                stage_profile.single_call,
                stage_profile.reasoning,
                stage_profile.finalizer,
            ):
                if call_profile is None:
                    continue
                assert call_profile.video_fps is None
                assert call_profile.max_image_pixels is None


def test_local_vllm_request_applies_per_call_profile_and_returns_usage(monkeypatch) -> None:
    payloads: list[dict[str, object]] = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {
                    "choices": [{"message": {"content": " reasoning "}}],
                    "usage": {
                        "prompt_tokens": 321,
                        "completion_tokens": 45,
                    },
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        assert timeout == 3600
        payloads.append(json.loads(request.data.decode("utf-8")))
        return FakeResponse()

    monkeypatch.setattr(qwen_runner.urllib.request, "urlopen", fake_urlopen)
    runner = qwen_runner.OpenAICompatibleLocalRunner(
        "Qwen/Qwen3.8-27B",
        disable_thinking=True,
    )
    result = runner.generate_with_metadata(
        "Analyze carefully.",
        call_profile=qwen_runner.GenerationCallProfile(
            max_new_tokens=77,
            disable_thinking=False,
        ),
    )

    assert result["text"] == "reasoning"
    assert result["prompt_tokens"] == 321
    assert result["completion_tokens"] == 45
    assert result["elapsed_seconds"] >= 0
    assert payloads[0]["max_tokens"] == 77
    assert payloads[0]["chat_template_kwargs"] == {"enable_thinking": True}


def test_h200_condition_wrappers_differ_only_by_label_and_treatment() -> None:
    nr = NR_JOB.read_text(encoding="utf-8")
    reasoning = R_JOB.read_text(encoding="utf-8")
    assert "#SBATCH --account=torch_pr_674_tandon_advanced" in nr
    assert "#SBATCH --cpus-per-task=24" in nr
    assert "#SBATCH --gres=gpu:1" in nr
    assert "#SBATCH --constraint=h200" in nr
    assert "#SBATCH --mem=480G" in nr
    assert "#SBATCH --time=08:00:00" in nr
    assert "--partition" not in nr
    assert "--nodelist" not in nr
    normalized_nr = nr.replace("reasoning_nr_10x3_h200", "CONDITION_LABEL").replace(
        'REASONING_MODE="nr"', 'REASONING_MODE="MODE"'
    )
    normalized_r = reasoning.replace("reasoning_r_10x3_h200", "CONDITION_LABEL").replace(
        'REASONING_MODE="r"', 'REASONING_MODE="MODE"'
    )
    assert normalized_nr == normalized_r
    assert 'source "${SCRIPT_DIR}/run_six_user_qa_reasoning_ab_h200_common.sh"' in nr


def test_common_h200_body_freezes_ab_inputs_and_records_reasoning_mode() -> None:
    text = COMMON.read_text(encoding="utf-8")
    for exact in (
        'TARGET_COUNT="10"',
        'MAX_ATTEMPTS="3"',
        'JUDGE_VIDEO_FPS="0.50"',
        'GENERATOR_MAX_IMAGE_PIXELS="91728"',
        'VLLM_MIN_IMAGE_PIXELS="3136"',
        'MAX_PACKETS_IN_FLIGHT="3"',
        'MAX_REVIEW_LANES="2"',
        'VLLM_SERVER_API_COUNT="3"',
        'VLLM_RENDERER_NUM_WORKERS="1"',
        'VLLM_MEDIA_LOADING_THREAD_COUNT="4"',
        'VLLM_SERVER_OMP_NUM_THREADS="2"',
        'VLLM_MM_PROCESSOR_CACHE_GB="2"',
        'VLLM_GPU_MEMORY_UTILIZATION="0.82"',
        'VLLM_MAX_NUM_SEQS="12"',
        'VLLM_MAX_NUM_BATCHED_TOKENS="65536"',
        'VLLM_ENABLE_PREFIX_CACHING="1"',
        '--reasoning-mode "${REASONING_MODE}"',
    ):
        assert exact in text
    assert "17109425/shared_preprocessing" in text


def test_keeper_uses_elapsed_time_not_memory_as_activation_condition() -> None:
    common = COMMON.read_text(encoding="utf-8")
    keeper = KEEPER.read_text(encoding="utf-8")
    assert 'CUDA_KEEPER_ENABLE="1"' in common
    assert 'CUDA_KEEPER_GPUS="0"' in common
    assert 'CUDA_KEEPER_START_AFTER_SECONDS="7200"' in common
    assert '--start-after-seconds "${CUDA_KEEPER_START_AFTER_SECONDS}"' in common
    assert "--start-used-mib" not in common
    assert "--start-used-mib" not in keeper
    assert 'add_argument("--start-after-seconds"' in keeper
    assert "elapsed_seconds < self.start_after_seconds" in keeper
