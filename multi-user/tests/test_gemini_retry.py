import io
import json
import urllib.error
from unittest.mock import patch

from egolife_two_user_qa import qwen3vl_runner
from egolife_two_user_qa.qwen3vl_runner import GeminiRunner


class _JsonResponse:
    def __init__(self, payload: dict) -> None:
        self._raw = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return self._raw


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://example.test/generateContent",
        code,
        "temporary error",
        {},
        io.BytesIO(b'{"error":{"message":"model is in high demand"}}'),
    )


def test_gemini_retries_every_503_until_request_succeeds() -> None:
    runner = GeminiRunner(
        api_key="test-key",
        service_unavailable_retry_delay_seconds=60,
    )

    with (
        patch.object(
            qwen3vl_runner.urllib.request,
            "urlopen",
            side_effect=[_http_error(503), _http_error(503), _JsonResponse({"ok": True})],
        ) as urlopen,
        patch.object(qwen3vl_runner.time, "sleep") as sleep,
    ):
        result = runner._request_json("https://example.test/generateContent", payload={"input": "x"})

    assert result == {"ok": True}
    assert urlopen.call_count == 3
    assert [call.args for call in sleep.call_args_list] == [(60.0,), (60.0,)]


def test_gemini_does_not_retry_non_503_http_errors() -> None:
    runner = GeminiRunner(
        api_key="test-key",
        service_unavailable_retry_delay_seconds=60,
    )

    with (
        patch.object(qwen3vl_runner.urllib.request, "urlopen", side_effect=_http_error(429)) as urlopen,
        patch.object(qwen3vl_runner.time, "sleep") as sleep,
    ):
        try:
            runner._request_json("https://example.test/generateContent", payload={"input": "x"})
        except RuntimeError as exc:
            assert "HTTP 429" in str(exc)
        else:
            raise AssertionError("expected a non-503 Gemini HTTP error to propagate")

    assert urlopen.call_count == 1
    sleep.assert_not_called()


def test_gemini_optional_503_retry_limit_is_honored() -> None:
    runner = GeminiRunner(
        api_key="test-key",
        service_unavailable_retry_delay_seconds=60,
        service_unavailable_max_retries=1,
    )

    with (
        patch.object(
            qwen3vl_runner.urllib.request,
            "urlopen",
            side_effect=[_http_error(503), _http_error(503)],
        ) as urlopen,
        patch.object(qwen3vl_runner.time, "sleep") as sleep,
    ):
        try:
            runner._request_json("https://example.test/generateContent", payload={"input": "x"})
        except RuntimeError as exc:
            assert "HTTP 503" in str(exc)
        else:
            raise AssertionError("expected Gemini to stop after the configured 503 retry limit")

    assert urlopen.call_count == 2
    assert [call.args for call in sleep.call_args_list] == [(60.0,)]
