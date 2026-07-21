"""仅绑定本机回环地址的冻结答题器 HTTP 服务与严格客户端。"""

from __future__ import annotations

import argparse
import json
import math
import socket
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping

from training.grpo_v3_answer_margin import LABELS
from training.grpo_v3_answer_scorer import (
    LabelScore,
    ScoreRequest,
    load_frozen_answer_scorer,
)


LOOPBACK_HOST = "127.0.0.1"


def parse_score_response(payload: Mapping[str, Any]) -> dict[str, LabelScore]:
    if not isinstance(payload, Mapping) or set(payload) != {"scores"}:
        raise RuntimeError("score response must contain only scores")
    raw_scores = payload["scores"]
    if not isinstance(raw_scores, Mapping) or set(raw_scores) != set(LABELS):
        raise RuntimeError("score response must contain exactly A-E")
    parsed: dict[str, LabelScore] = {}
    required_fields = {"label", "token_ids", "token_logprobs", "sequence_logprob"}
    for label in LABELS:
        item = raw_scores[label]
        if not isinstance(item, Mapping) or set(item) != required_fields:
            raise RuntimeError(f"{label} score has an invalid response structure")
        if item["label"] != label:
            raise RuntimeError(f"{label} score label does not match its key")
        token_ids = item["token_ids"]
        token_logprobs = item["token_logprobs"]
        if (
            not isinstance(token_ids, list)
            or not token_ids
            or any(isinstance(value, bool) or not isinstance(value, int) for value in token_ids)
        ):
            raise RuntimeError(f"{label} token_ids must be a non-empty integer list")
        if not isinstance(token_logprobs, list) or len(token_logprobs) != len(token_ids):
            raise RuntimeError(f"{label} token_logprobs must match token_ids")
        numeric_logprobs: list[float] = []
        for value in token_logprobs:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RuntimeError(f"{label} token logprob must be numeric")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise RuntimeError(f"{label} token logprob must be finite")
            numeric_logprobs.append(numeric)
        sequence = item["sequence_logprob"]
        if isinstance(sequence, bool) or not isinstance(sequence, (int, float)):
            raise RuntimeError(f"{label} sequence logprob must be numeric")
        sequence_value = float(sequence)
        if not math.isfinite(sequence_value):
            raise RuntimeError(f"{label} sequence logprob must be finite")
        if not math.isclose(sequence_value, sum(numeric_logprobs), rel_tol=1e-7, abs_tol=1e-7):
            raise RuntimeError(f"{label} sequence logprob does not equal its token sum")
        parsed[label] = LabelScore(label, list(token_ids), numeric_logprobs, sequence_value)
    return parsed


class AnswerScorerClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 30.0,
        opener: Any | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        if self.timeout_seconds <= 0 or not math.isfinite(self.timeout_seconds):
            raise ValueError("timeout_seconds must be finite and positive")
        self.opener = opener or urllib.request.build_opener()

    def score(self, request: ScoreRequest) -> dict[str, LabelScore]:
        body = json.dumps(request.to_payload(), ensure_ascii=False).encode("utf-8")
        http_request = urllib.request.Request(
            self.base_url + "/score",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self.opener.open(http_request, timeout=self.timeout_seconds) as response:
                if response.status != 200:
                    raise RuntimeError(f"answer scorer returned HTTP {response.status}")
                payload = json.load(response)
        except (TimeoutError, socket.timeout) as error:
            raise TimeoutError("answer scorer request timed out") from error
        except urllib.error.HTTPError as error:
            code = error.code
            error.close()
            raise RuntimeError(f"answer scorer returned HTTP {code}") from error
        except urllib.error.URLError as error:
            if isinstance(error.reason, (TimeoutError, socket.timeout)):
                raise TimeoutError("answer scorer request timed out") from error
            raise RuntimeError("answer scorer request failed") from error
        except json.JSONDecodeError as error:
            raise RuntimeError("answer scorer request failed") from error
        return parse_score_response(payload)


def _handler_for(scorer: Any) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _json(self, status: int, payload: Mapping[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path != "/health":
                self._json(404, {"error": "not_found"})
                return
            self._json(200, {"status": "ok", "trainable_parameter_count": scorer.trainable_parameter_count if hasattr(scorer, "trainable_parameter_count") else 0})

        def do_POST(self) -> None:
            if self.path != "/score":
                self._json(404, {"error": "not_found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                request = ScoreRequest.from_payload(payload)
                scores = scorer.score(request)
                if set(scores) != set(LABELS):
                    raise RuntimeError("scorer returned incomplete labels")
                response = {"scores": {label: scores[label].to_payload() for label in LABELS}}
                # 复用客户端解析，服务端同样拒绝非有限或不完整结构。
                parse_score_response(response)
                self._json(200, response)
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                self._json(400, {"error": type(error).__name__})
            except Exception as error:
                self._json(500, {"error": type(error).__name__})

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def create_server(
    scorer: Any,
    *,
    host: str = LOOPBACK_HOST,
    port: int = 8765,
) -> ThreadingHTTPServer:
    if host != LOOPBACK_HOST:
        raise ValueError("answer scorer service may only bind to 127.0.0.1")
    return ThreadingHTTPServer((host, int(port)), _handler_for(scorer))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="启动本机冻结双视频答案 scorer")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--port", type=int, default=8765)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scorer = load_frozen_answer_scorer(
        args.model,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )
    server = create_server(scorer, port=args.port)
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
