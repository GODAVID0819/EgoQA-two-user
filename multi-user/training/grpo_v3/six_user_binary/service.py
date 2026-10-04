"""仅监听本机的冻结 Judge 服务；模型或媒体失败不伪造成低奖励。"""
from __future__ import annotations

import argparse
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from .judge import InvalidCompletion, build_examples
from .reward import validate_probabilities


def validate_response(result: dict, request: dict) -> dict:
    if result.get("request_id") != request["request_id"] or result.get("evidence_id") != request["evidence_id"]:
        raise RuntimeError("Judge response identity mismatch")
    if result.get("status") == "scored":
        validate_probabilities(result["probabilities"])
    elif result.get("status") != "invalid_completion" or not result.get("reason"):
        raise RuntimeError("未知或不完整的 Judge 结果")
    if not isinstance(result.get("judge"), dict):
        raise RuntimeError("缺少 Judge 身份信息")
    return result


class JudgeClient:
    def __init__(self, base_url: str, timeout_seconds: float = 1800, *, expected_instance=None):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.expected_instance = expected_instance

    def _check_instance(self, identity):
        if self.expected_instance is not None and identity.get("instance_id") != self.expected_instance:
            raise RuntimeError("Judge instance 与当前作业不一致")

    def _request(self, path, payload=None):
        body = None if payload is None else json.dumps(payload, allow_nan=False).encode("utf-8")
        request = Request(self.base_url + path, data=body, headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return json.load(response)
        except HTTPError as exc:
            with exc:
                detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Judge 服务失败 HTTP {exc.code}: {detail}") from exc

    def health(self):
        value = self._request("/health")
        if value.get("status") != "ready" or value.get("frozen") is not True:
            raise RuntimeError("Judge 未就绪或未冻结")
        self._check_instance(value)
        return value

    def resource(self, action):
        if not self.expected_instance or action not in {'sleep', 'wake'}:
            raise ValueError('资源切换必须绑定当前Judge实例及已知动作')
        value = self._request('/resource', {'instance_id': self.expected_instance, 'action': action})
        self._check_instance(value)
        if value.get('sleeping') is not (action == 'sleep'):
            raise RuntimeError('Judge未确认请求的显存状态')
        return value

    def score(self, request):
        result = validate_response(self._request("/score", request), request)
        self._check_instance(result["judge"])
        return result

    def score_many(self, requests):
        values = self._request("/score_batch", {"requests": requests}).get("results")
        if not isinstance(values, list) or len(values) != len(requests):
            raise RuntimeError("Judge 批次返回数量不一致")
        for request, result in zip(requests, values):
            validate_response(result, request)
            self._check_instance(result["judge"])
        return values


class CandidateScorer:
    def __init__(self, predictor, instance_id):
        self.predictor = predictor
        self.instance_id = instance_id

    def health(self):
        return {"status": "ready", "frozen": True, **self.predictor.identity, "instance_id": self.instance_id}

    def resource(self, request):
        if request.get('instance_id') != self.instance_id:
            raise ValueError('资源控制不属于当前Judge实例')
        action = request.get('action')
        if action not in {'sleep', 'wake'}:
            raise ValueError('未知的资源控制动作')
        return {**self.predictor.resource(action), 'instance_id': self.instance_id}

    def score(self, request):
        return self.score_many([request])[0]

    def score_many(self, requests):
        start = time.perf_counter()
        if not isinstance(requests, list) or not 1 <= len(requests) <= 64:
            raise ValueError("评分批次必须包含 1 至 64 个候选")
        if len({r['request_id'] for r in requests}) != len(requests):
            raise ValueError("评分批次 request_id 重复")
        results, valid = [], []
        for request in requests:
            identity = {"request_id": request["request_id"], "evidence_id": request["evidence_id"], "judge": self.health()}
            try:
                examples = build_examples(request)
            except InvalidCompletion as exc:
                results.append({**identity, "status": "invalid_completion", "reason": str(exc)})
                continue
            valid.append((len(results), examples))
            results.append({**identity, "status": "scored", "predictions": {}})
        # 同一任务的候选相邻，复用视觉前缀；不能把候选位置用作身份替代。
        for key in ("formality", "groundedness", "speaker_only", "all_six"):
            examples = [row[key] for _, row in valid]
            if not examples:
                continue
            predictions = self.predictor.predict_many(examples)
            if len(predictions) != len(valid):
                raise RuntimeError("Judge predictor 批次返回数量不一致")
            for (index, _), prediction in zip(valid, predictions):
                results[index]["predictions"][key] = prediction
        for result in results:
            if result["status"] == "scored":
                result["probabilities"] = validate_probabilities({k:v["pass_probability"] for k,v in result["predictions"].items()})
            result["batch_elapsed_seconds"] = time.perf_counter() - start
            result["batch_size"] = len(requests)
        return results


def make_server(scorer, *, port: int):
    lock = threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def respond(self, code, value):
            body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def do_GET(self):
            self.respond(200, scorer.health()) if self.path == "/health" else self.respond(404, {"error": "not_found"})
        def do_POST(self):
            if self.path not in {"/score", "/score_batch", "/resource"}:
                self.respond(404, {"error": "not_found"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 16 * 1024 * 1024:
                    raise ValueError("评分请求大小不合理")
                request = json.loads(self.rfile.read(size))
                with lock:
                    if self.path == '/resource':
                        result = scorer.resource(request)
                    elif self.path == "/score_batch":
                        requests = request["requests"]
                        results = scorer.score_many(requests)
                        result = {"results": [validate_response(v, r) for v,r in zip(results, requests)]}
                    else:
                        result = validate_response(scorer.score(request), request)
                self.respond(200, result)
            except Exception as exc:
                logging.exception("Judge 请求失败")
                self.respond(500, {"error": type(exc).__name__, "message": str(exc)})
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description="冻结六用户 Judge 服务")
    parser.add_argument("--config", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--instance-id", required=True)
    args = parser.parse_args()
    from .predictor import VllmJudge
    scorer = CandidateScorer(VllmJudge.from_config(args.config), args.instance_id)
    if scorer.predictor.config.get('shared_gpu'):
        scorer.predictor.resource('sleep')
    with make_server(scorer, port=args.port) as server:
        print(json.dumps(scorer.health(), ensure_ascii=False), flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
