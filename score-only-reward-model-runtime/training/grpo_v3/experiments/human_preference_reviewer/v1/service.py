"""Loopback-only HTTP service for the frozen score-only ordinal reviewer."""

from __future__ import annotations

import argparse
import json
import logging
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping

from .deployment import SCORE_FIELDS, SCORE_FORMAT_VERSION

LOOPBACK_HOST = "127.0.0.1"
LOGGER = logging.getLogger(__name__)


def parse_score_response(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "format_version", "review_key", "evidence_id", "cumulative_probabilities",
        "class_probabilities", "hard_scores", "expected_scores", "reward", "reward_policy",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise RuntimeError("score-only reviewer response has an invalid structure")
    if value.get("format_version") != SCORE_FORMAT_VERSION:
        raise RuntimeError("score-only reviewer response has an incompatible format version")
    if not str(value.get("review_key") or "").strip() or not str(value.get("evidence_id") or "").strip():
        raise RuntimeError("score-only reviewer response lacks identity")
    if set(value.get("expected_scores") or {}) != set(SCORE_FIELDS):
        raise RuntimeError("score-only reviewer response lacks expected scores")
    reward = value.get("reward")
    if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not 0.0 <= float(reward) <= 1.0:
        raise RuntimeError("score-only reviewer response has invalid reward")
    return dict(value)


class ScoreOnlyReviewerClient:
    def __init__(self, base_url: str, *, timeout_seconds: float = 300.0) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.timeout_seconds = float(timeout_seconds)

    def _request(self, path: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="GET" if payload is None else "POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"score-only reviewer HTTP {error.code}") from error
        if not isinstance(result, dict):
            raise RuntimeError("score-only reviewer returned non-object JSON")
        return result

    def health(self) -> dict[str, Any]:
        result = self._request("/health")
        if result.get("status") != "ok" or result.get("trainable_parameter_count") != 0:
            raise RuntimeError("score-only reviewer is not frozen and ready")
        return result

    def score(self, candidate: Mapping[str, Any]) -> dict[str, Any]:
        return parse_score_response(self._request("/score", {"candidate": dict(candidate)}))


def _handler(scorer: Any, score_lock: threading.Lock) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _json(self, status: int, value: Mapping[str, Any]) -> None:
            body = json.dumps(dict(value), ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path != "/health":
                self._json(404, {"error": "not_found"})
                return
            try:
                readiness = scorer.readiness()
                healthy = (
                    readiness.get("status") == "ok"
                    and readiness.get("trainable_parameter_count") == 0
                    and all((readiness.get("checks") or {}).values())
                )
                self._json(200 if healthy else 503, readiness)
            except Exception as error:
                LOGGER.exception("score-only reviewer health failed")
                self._json(503, {"status": "unhealthy", "error": type(error).__name__})

        def do_POST(self) -> None:
            if self.path != "/score":
                self._json(404, {"error": "not_found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(request, Mapping) or set(request) != {"candidate"} or not isinstance(request["candidate"], Mapping):
                    raise ValueError("score request must contain exactly one candidate object")
                with score_lock:
                    score = scorer.score(dict(request["candidate"]))
                self._json(200, parse_score_response(score))
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                LOGGER.exception("score-only reviewer rejected request")
                self._json(400, {"error": type(error).__name__})
            except Exception as error:
                LOGGER.exception("score-only reviewer inference failed")
                self._json(500, {"error": type(error).__name__})

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def create_server(scorer: Any, *, host: str = LOOPBACK_HOST, port: int = 8766) -> ThreadingHTTPServer:
    if host != LOOPBACK_HOST:
        raise ValueError("score-only reviewer service must use loopback binding")
    return ThreadingHTTPServer((host, int(port)), _handler(scorer, threading.Lock()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args(argv)
    from .deployment import load_frozen_score_only_reviewer

    scorer = load_frozen_score_only_reviewer(
        Path(args.checkpoint), model=args.model, device=args.device, torch_dtype=args.torch_dtype
    )
    server = create_server(scorer, port=args.port)
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
