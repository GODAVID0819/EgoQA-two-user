import base64
import json
from pathlib import Path
from unittest.mock import patch

from egolife_two_user_qa import qwen3vl_runner
from egolife_two_user_qa.qwen3vl_runner import (
    DEFAULT_OPENROUTER_BASE_URL,
    DEFAULT_OPENROUTER_MODEL_ID,
    OpenRouterRunner,
    OpenRouterRequestError,
    make_runner,
)
from egolife_two_user_qa.video_qa_loop import (
    run_answerability_eval,
    run_model_judge_branch,
    run_parallel_review_judges,
)


class _JsonResponse:
    def __init__(self, payload: dict) -> None:
        self._raw = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return self._raw


def test_openrouter_factory_uses_gemini25_defaults() -> None:
    runner = make_runner(
        "openrouter",
        api_key="test-key",
        allow_openai_video_input=True,
    )

    assert isinstance(runner, OpenRouterRunner)
    assert runner.model_id == DEFAULT_OPENROUTER_MODEL_ID == "google/gemini-2.5-flash"
    assert runner.base_url == DEFAULT_OPENROUTER_BASE_URL == "https://openrouter.ai/api/v1"
    assert runner.allow_video_input is True
    assert runner.supports_choice_logits is False


def test_openrouter_runner_sends_local_video_as_base64_video_url() -> None:
    runner = OpenRouterRunner(
        api_key="test-key",
        allow_video_input=True,
        reasoning_effort="high",
    )
    video_path = Path(__file__).resolve().parent / "fixtures" / "openrouter_test.mp4"
    video_bytes = video_path.read_bytes()

    with patch.object(
        qwen3vl_runner.urllib.request,
        "urlopen",
        return_value=_JsonResponse({"choices": [{"message": {"content": "ok"}}]}),
    ) as urlopen:
        assert runner.generate("Judge independently.", video_paths=[str(video_path)]) == "ok"

    request = urlopen.call_args.args[0]
    payload = json.loads(request.data.decode("utf-8"))
    content = payload["messages"][0]["content"]
    video_part = next(part for part in content if part["type"] == "video_url")
    data_url = video_part["video_url"]["url"]

    assert request.full_url == "https://openrouter.ai/api/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer test-key"
    assert payload["model"] == "google/gemini-2.5-flash"
    assert payload["temperature"] == 0
    assert payload["reasoning"] == {"effort": "high", "exclude": True}
    assert payload["provider"] == {"allow_fallbacks": True}
    assert data_url.startswith("data:video/mp4;base64,")
    assert base64.b64decode(data_url.split(",", 1)[1]) == video_bytes


def test_openrouter_runner_forwards_generator_sampling_top_k() -> None:
    runner = OpenRouterRunner(api_key="test-key")

    with patch.object(
        qwen3vl_runner.urllib.request,
        "urlopen",
        return_value=_JsonResponse({"choices": [{"message": {"content": "ok"}}]}),
    ) as urlopen:
        assert runner.generate(
            "Generate one candidate.",
            decoding_mode="sampling",
            temperature=0.7,
            top_p=0.9,
            top_k=40,
        ) == "ok"

    request = urlopen.call_args.args[0]
    payload = json.loads(request.data.decode("utf-8"))
    assert payload["temperature"] == 0.7
    assert payload["top_p"] == 0.9
    assert payload["top_k"] == 40


def test_openrouter_transcodes_and_caches_judge_video_before_upload() -> None:
    video_path = Path(__file__).resolve().parent / "fixtures" / "openrouter_test.mp4"
    cache_dir = Path(__file__).resolve().parent / "fixtures" / "tmp_openrouter_video_cache"
    environment = {
        "OPENROUTER_VIDEO_MAX_EDGE": "640",
        "OPENROUTER_VIDEO_FPS": "2",
        "OPENROUTER_VIDEO_CRF": "28",
        "OPENROUTER_VIDEO_CACHE_DIR": str(cache_dir),
    }
    with patch.dict(qwen3vl_runner.os.environ, environment):
        runner = OpenRouterRunner(api_key="test-key", allow_video_input=True)

    def fake_ffmpeg(command, **kwargs):
        assert command[command.index("-vf") + 1] == (
            "fps=2,scale=640:640:force_original_aspect_ratio=decrease,"
            "scale=trunc(iw/2)*2:trunc(ih/2)*2"
        )
        assert command[command.index("-crf") + 1] == "28"
        Path(command[-1]).write_bytes(b"compressed-video")

    try:
        with patch.object(qwen3vl_runner.subprocess, "run", side_effect=fake_ffmpeg) as ffmpeg:
            first = runner._prepare_video_for_upload(video_path)
            second = runner._prepare_video_for_upload(video_path)

        assert first == second
        assert Path(first).read_bytes() == b"compressed-video"
        assert ffmpeg.call_count == 1
    finally:
        for cached_file in cache_dir.glob("*.mp4"):
            cached_file.unlink()


def test_openrouter_retries_error_body_without_choices() -> None:
    runner = OpenRouterRunner(
        api_key="test-key",
        max_retries=2,
        retry_delay_seconds=0,
    )
    responses = [
        _JsonResponse(
            {
                "error": {
                    "code": 429,
                    "message": "Rate limit exceeded",
                    "metadata": {"error_type": "rate_limit_exceeded"},
                }
            }
        ),
        _JsonResponse({"choices": [{"message": {"content": "recovered"}}]}),
    ]

    with patch.object(qwen3vl_runner.urllib.request, "urlopen", side_effect=responses) as urlopen:
        assert runner.generate("Judge independently.") == "recovered"

    assert urlopen.call_count == 2


def test_openrouter_exhausted_error_remains_infrastructure_error() -> None:
    runner = OpenRouterRunner(
        api_key="test-key",
        max_retries=1,
        retry_delay_seconds=0,
    )
    error_response = {
        "error": {
            "code": 503,
            "message": "Provider unavailable",
            "metadata": {"error_type": "provider_unavailable"},
        }
    }

    with patch.object(
        qwen3vl_runner.urllib.request,
        "urlopen",
        side_effect=[_JsonResponse(error_response), _JsonResponse(error_response)],
    ):
        try:
            runner.generate("Judge independently.")
        except OpenRouterRequestError as exc:
            assert "Provider unavailable" in str(exc)
        else:
            raise AssertionError("exhausted OpenRouter errors must remain infrastructure failures")


def test_parallel_review_does_not_convert_openrouter_error_to_judge_fail() -> None:
    class FailingOpenRouterRunner:
        model_id = "google/gemini-2.5-flash"
        supports_choice_logits = False

        def generate(self, *args, **kwargs):
            raise OpenRouterRequestError(
                "OpenRouter error (code=503): provider unavailable",
                code=503,
                retryable=True,
            )

    runner = FailingOpenRouterRunner()
    try:
        run_parallel_review_judges(
            qa_item={
                "qa_id": "q1",
                "question_type": "neutral",
                "question": "What was in the bowl?",
                "options": ["Chips", "Dough", "Fruit", "Keys", "Water"],
                "correct": "B",
                "answer": "Dough",
                "required_users": ["u1", "u2"],
            },
            packet={"evidence_id": "e1", "clips": []},
            schema_errors=[],
            runner=runner,
            qa_formality_runner=runner,
            media_backend="openrouter",
            allow_openai_video_input=True,
            prompt_rows=[],
            full_image_paths=[],
            full_video_paths=[],
            attempt=1,
        )
    except OpenRouterRequestError as exc:
        assert "provider unavailable" in str(exc)
    else:
        raise AssertionError("OpenRouter infrastructure errors must not become semantic FAIL results")


def test_production_judge_does_not_request_or_attach_pass_fail_logprobs() -> None:
    class Runner:
        supports_choice_logits = False

        def generate(self, *args, **kwargs):
            return json.dumps(
                {
                    "review_passed": True,
                    "checks": {
                        "qa_formality": {
                            "status": "PASS",
                            "reason": "ok",
                            "fix": "",
                            "semantic_subchecks": {
                                name: {"status": "PASS", "reason": "ok"}
                                for name in (
                                    "first_person_perspective",
                                    "naturalness_and_clarity",
                                    "other_person_activity_query",
                                    "direct_name_leakage",
                                    "timestamp_citation",
                                )
                            },
                        }
                    },
                    "blocking_failures": [],
                    "feedback_to_generator": "",
                }
            )

        def generate_with_choice_logits(self, *args, **kwargs):
            raise AssertionError("OpenRouter Gemini must not be asked for logprobs")

    judge = run_model_judge_branch(
        check_name="qa_formality",
        prompt="prompt",
        runner=Runner(),
        image_paths=[],
        video_paths=[],
        evidence_id="e1",
        qa_id="q1",
        attempt=1,
    )

    assert judge["checks"]["qa_formality"]["status"] == "PASS"
    assert "choice_logit_signal" not in judge


def test_openrouter_answerability_skips_unsupported_logprobs() -> None:
    class Runner:
        supports_choice_logits = False

        def generate(self, *args, **kwargs):
            return (
                '{"choice":"insufficient","answer_text":"","evidence_used":"",'
                '"insufficient_reason":"No video was supplied in this unit test."}'
            )

        def generate_with_choice_logits(self, *args, **kwargs):
            raise AssertionError("OpenRouter Gemini must not be asked for logprobs")

    result = run_answerability_eval(
        qa_item={
            "qa_id": "q1",
            "question": "What was in the bowl?",
            "options": ["Chips", "Dough", "Fruit", "Keys", "Water"],
            "correct": "B",
            "required_users": ["u1", "u2"],
        },
        packet={"clips": []},
        runner=Runner(),
        media_backend="openrouter",
        allow_openai_video_input=True,
        prompt_rows=[],
    )

    assert len(result["evaluations"]) == 3
    assert all(row["choice"] == "insufficient" for row in result["evaluations"])


def test_parallel_review_routes_formality_to_qwen_and_visual_checks_to_openrouter() -> None:
    class QwenFormalityRunner:
        model_id = "Qwen/Qwen3.6-27B"

        def __init__(self) -> None:
            self.calls = 0

        def generate(self, prompt, **kwargs):
            self.calls += 1
            assert "qa_formality judge" in prompt
            return json.dumps(
                {
                    "review_passed": True,
                    "checks": {
                        "qa_formality": {
                            "status": "PASS",
                            "reason": "ok",
                            "fix": "",
                            "semantic_subchecks": {
                                name: {"status": "PASS", "reason": "ok"}
                                for name in (
                                    "first_person_perspective",
                                    "naturalness_and_clarity",
                                    "other_person_activity_query",
                                    "direct_name_leakage",
                                    "timestamp_citation",
                                )
                            },
                        }
                    },
                    "blocking_failures": [],
                    "why_generator_asked_this": "",
                    "feedback_to_generator": "",
                }
            )

    class OpenRouterVisualRunner:
        model_id = "google/gemini-2.5-flash"
        supports_choice_logits = False

        def __init__(self) -> None:
            self.calls = 0

        def generate(self, prompt, **kwargs):
            self.calls += 1
            if "evidence_groundedness judge" in prompt:
                return (
                    '{"review_passed":true,"checks":{"evidence_groundedness":'
                    '{"status":"PASS","reason":"ok","fix":""}},'
                    '"blocking_failures":[],"why_generator_asked_this":"",'
                    '"feedback_to_generator":""}'
                )
            return (
                '{"choice":"insufficient","answer_text":"","evidence_used":"",'
                '"insufficient_reason":"No video was supplied in this unit test."}'
            )

        def generate_with_choice_logits(self, *args, **kwargs):
            raise AssertionError("OpenRouter Gemini must not be asked for logprobs")

    qwen = QwenFormalityRunner()
    openrouter = OpenRouterVisualRunner()
    prompt_rows = []
    _, _, trace = run_parallel_review_judges(
        qa_item={
            "qa_id": "q1",
            "question_type": "neutral",
            "question": "What was in the bowl?",
            "options": ["Chips", "Dough", "Fruit", "Keys", "Water"],
            "correct": "B",
            "answer": "Dough",
            "required_users": ["u1", "u2"],
        },
        packet={"evidence_id": "e1", "clips": []},
        schema_errors=[],
        runner=openrouter,
        qa_formality_runner=qwen,
        media_backend="openrouter",
        allow_openai_video_input=True,
        prompt_rows=prompt_rows,
        full_image_paths=[],
        full_video_paths=[],
        attempt=1,
        # Legacy opt-ins must be ignored by the production review path.
        pass_fail_only=False,
        quality_quota_counts={"qa_formality": 48, "evidence_groundedness": 48},
        quality_quota=48,
    )

    assert qwen.calls == 1
    assert openrouter.calls == 4
    assert trace["qa_formality"]["model_id"] == "Qwen/Qwen3.6-27B"
    assert trace["evidence_groundedness"]["model_id"] == "google/gemini-2.5-flash"
    assert trace["answerability_model_id"] == "google/gemini-2.5-flash"
    assert trace["qa_formality"]["generator_rationale_included"] is False
    assert trace["evidence_groundedness"]["generator_rationale_included"] is True
    assert trace["pass_fail_only"] is True
    assert trace["point_scoring"] == "legacy_archived_not_active"
    assert "quality_quota" not in trace
    for row in prompt_rows[:2]:
        assert row["pass_fail_only"] is True
        assert "quality_score rubric" not in row["prompt"]
        assert "Global 3-point quota" not in row["prompt"]
    formality_row = next(row for row in prompt_rows if row["stage"] == "qa_formality_judge")
    assert formality_row["generator_rationale_included"] is False
