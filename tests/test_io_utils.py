from __future__ import annotations

import sys
import types
import unittest
import urllib.error
from inspect import signature
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if "egolife_two_user_qa" not in sys.modules:
    package = types.ModuleType("egolife_two_user_qa")
    package.__path__ = [str(ROOT)]
    sys.modules["egolife_two_user_qa"] = package

from egolife_two_user_qa import io_utils  # noqa: E402


class HuggingFaceDownloadTests(unittest.TestCase):
    def test_request_reads_token_from_explicit_token_path(self) -> None:
        with (
            mock.patch.dict(
                "os.environ",
                {"HF_TOKEN_PATH": "/login-home/.cache/huggingface/token"},
                clear=True,
            ),
            mock.patch.object(
                io_utils.Path,
                "read_text",
                return_value="hf_example_token\n",
            ),
        ):
            request = io_utils._request("https://huggingface.co/example")

        self.assertEqual(
            request.get_header("Authorization"),
            "Bearer hf_example_token",
        )

    def test_rate_limit_retry_honors_retry_after(self) -> None:
        error = urllib.error.HTTPError(
            "https://huggingface.co/example",
            429,
            "Too Many Requests",
            {"Retry-After": "120"},
            None,
        )
        self.assertEqual(io_utils._retry_delay_seconds(error, 0), 120.0)

    def test_rate_limit_retry_is_capped(self) -> None:
        error = urllib.error.HTTPError(
            "https://huggingface.co/example",
            429,
            "Too Many Requests",
            {"Retry-After": "3600"},
            None,
        )
        self.assertEqual(io_utils._retry_delay_seconds(error, 0), 300.0)

    def test_download_default_allows_nine_attempts(self) -> None:
        retries = signature(io_utils.download_file).parameters["retries"].default
        self.assertEqual(retries, 8)


if __name__ == "__main__":
    unittest.main()
