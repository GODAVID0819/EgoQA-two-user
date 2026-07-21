"""ms-swift 4.2.2 的 Gate 1/2 reward 插件。"""

from __future__ import annotations

import json
import hashlib
import math
import os
import threading
from pathlib import Path
from typing import Any, Callable, Sequence

try:
    from swift.rewards import ORM, orms
except ImportError:  # 登录节点/本地纯逻辑测试不安装 ms-swift。
    class ORM:  # type: ignore[no-redef]
        def __init__(self, args: Any = None, **kwargs: Any) -> None:
            self.args = args

    orms: dict[str, type] = {}


def _expand(values: Any, length: int, *, name: str) -> list[Any]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        rows = [values]
    else:
        rows = list(values)
    if len(rows) == length:
        return rows
    if len(rows) == 1:
        return rows * length
    raise ValueError(f"{name} 数量为 {len(rows)}，无法展开到 {length}")


def _trace_path(value: str | Path | None) -> Path:
    path = Path(value or os.environ.get("EGOQA_GRPO_V3_REWARD_TRACE", "reward_trace.jsonl"))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_rows(path: Path, rows: list[dict[str, Any]], lock: threading.Lock) -> None:
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _answer_margin_metadata_snapshot(value: Any) -> Any:
    """仅供 answer-margin metadata 失败 trace 使用的稳定 JSON 快照。"""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return {"type": "Path", "value": str(value)}
    if isinstance(value, bytes):
        return {
            "type": "bytes",
            "length": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(key, str):
                safe_key = key
            else:
                safe_key = json.dumps(
                    _answer_margin_metadata_snapshot(key),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            result[safe_key] = _answer_margin_metadata_snapshot(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_answer_margin_metadata_snapshot(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_answer_margin_metadata_snapshot(item) for item in value]
        items.sort(key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True))
        return {"type": "set", "items": items}
    value_type = type(value)
    return {"type": f"{value_type.__module__}.{value_type.__qualname__}"}


class CompletionGroupSizeError(ValueError):
    """answer-margin ORM 没有收到固定的四 completion group。"""


class ControlledGateReward(ORM):
    """Gate 1 边界探针；分数只验证训练闭环，不代表任务质量。"""

    def __init__(self, args: Any = None, *, trace_path: str | Path | None = None, **kwargs: Any) -> None:
        super().__init__(args, **kwargs)
        self.trace_path = _trace_path(trace_path)
        self._lock = threading.Lock()

    def __call__(self, completions: Sequence[str], **kwargs: Any) -> list[float]:
        count = len(completions)
        evidence_ids = _expand(kwargs.get("evidence_id", "unknown"), count, name="evidence_id")
        midpoint = (count - 1) / 2.0
        rewards = [float(index - midpoint) for index in range(count)]
        rows = [
            {
                "reward_kind": "gate1_controlled",
                "evidence_id": str(evidence_ids[index]),
                "candidate_index": index,
                "reward": rewards[index],
                "completion": str(completions[index]),
                "formal_result": False,
            }
            for index in range(count)
        ]
        _write_rows(self.trace_path, rows, self._lock)
        return rewards


class RepoNativeJudgeReward(ORM):
    def __init__(
        self,
        args: Any = None,
        *,
        trace_path: str | Path | None = None,
        score_fn: Callable[..., dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(args, **kwargs)
        self.trace_path = _trace_path(trace_path)
        self._lock = threading.Lock()
        self._reward_call_index = 0
        self.score_fn = score_fn or self._build_score_fn()

    def _next_call_index(self) -> int:
        with self._lock:
            value = self._reward_call_index
            self._reward_call_index += 1
        return value

    @staticmethod
    def _build_score_fn() -> Callable[..., dict[str, Any]]:
        from training.grpo_v3_repo_reward import make_repo_score_fn

        return make_repo_score_fn(
            review_model_id=os.environ.get("EGOQA_REVIEW_MODEL", "Qwen/Qwen3-VL-8B-Instruct"),
            review_base_url=os.environ.get("EGOQA_REVIEW_BASE_URL", "http://127.0.0.1:8001/v1"),
            policy_model_id=os.environ.get("EGOQA_POLICY_MODEL", "Qwen/Qwen3-VL-2B-Instruct"),
            review_max_new_tokens=int(os.environ.get("EGOQA_REVIEW_MAX_NEW_TOKENS", "2048")),
        )

    def __call__(self, completions: Sequence[str], **kwargs: Any) -> list[float]:
        count = len(completions)
        call_index = self._next_call_index()
        packets = _expand(kwargs["packet_json"], count, name="packet_json")
        evidence_ids = _expand(kwargs["evidence_id"], count, name="evidence_id")
        question_types = _expand(kwargs["question_type"], count, name="question_type")
        generation_modes = _expand(kwargs["generation_mode"], count, name="generation_mode")
        eval_ids = {
            item.strip()
            for item in os.environ.get("EGOQA_EVAL_EVIDENCE_IDS", "").split(",")
            if item.strip()
        }
        phases = ["eval" if str(item) in eval_ids else "train" for item in evidence_ids]
        if len(set(phases)) != 1:
            rows = [
                {
                    "reward_kind": "repo_native_judge",
                    "reward_call_index": call_index,
                    "phase": phases[index],
                    "evidence_id": str(evidence_ids[index]),
                    "candidate_index": index,
                    "completion_length_chars": len(str(completions[index])),
                    "reward": None,
                    "record": {
                        "masked": True,
                        "eligible_for_grpo": False,
                        "infrastructure_error": {
                            "type": "MixedPhaseGroupError",
                            "message": "同一 reward 调用混入 train/eval evidence",
                        },
                    },
                }
                for index in range(count)
            ]
            _write_rows(self.trace_path, rows, self._lock)
            raise ValueError("同一 reward 调用不得混入 train/eval evidence")
        phase = phases[0] if phases else "train"
        rewards: list[float | None] = []
        traces: list[dict[str, Any]] = []
        for index, completion in enumerate(completions):
            try:
                packet_value = packets[index]
                packet = json.loads(packet_value) if isinstance(packet_value, str) else packet_value
                result = self.score_fn(
                    raw_completion=str(completion),
                    packet=packet,
                    evidence_id=str(evidence_ids[index]),
                    question_type=str(question_types[index]),
                    generation_mode=str(generation_modes[index]),
                    candidate_index=index,
                )
            except Exception as exc:
                traces.append(
                    {
                        "reward_kind": "repo_native_judge",
                        "reward_call_index": call_index,
                        "phase": phase,
                        "evidence_id": str(evidence_ids[index]),
                        "candidate_index": index,
                        "completion_length_chars": len(str(completion)),
                        "reward": None,
                        "record": {
                            "masked": True,
                            "eligible_for_grpo": False,
                            "infrastructure_error": {
                                "type": type(exc).__name__,
                                "message": str(exc),
                            },
                        },
                    }
                )
                _write_rows(self.trace_path, traces, self._lock)
                raise
            reward = result.get("reward")
            if reward is not None:
                reward = float(reward)
                if not math.isfinite(reward):
                    traces.append(
                        {
                            "reward_kind": "repo_native_judge",
                            "reward_call_index": call_index,
                            "phase": phase,
                            "evidence_id": str(evidence_ids[index]),
                            "candidate_index": index,
                            "completion_length_chars": len(str(completion)),
                            "reward": None,
                            "record": {
                                **_json_safe(dict(result.get("record", {}))),
                                "masked": True,
                                "eligible_for_grpo": False,
                                "infrastructure_error": {
                                    "type": "NonFiniteRewardError",
                                    "message": f"reward 非有限值: {reward}",
                                },
                            },
                        }
                    )
                    _write_rows(self.trace_path, traces, self._lock)
                    raise ValueError(f"reward 非有限值: {reward}")
            rewards.append(reward)
            traces.append(
                {
                    "reward_kind": "repo_native_judge",
                    "reward_call_index": call_index,
                    "phase": phase,
                    "evidence_id": str(evidence_ids[index]),
                    "candidate_index": index,
                    "completion_length_chars": len(str(completion)),
                    "reward": reward,
                    "record": result.get("record", {}),
                }
            )
        _write_rows(self.trace_path, traces, self._lock)
        if any(value is None for value in rewards):
            reasons = [
                str(row.get("record", {}).get("mask_reason") or "unknown")
                for row in traces
                if row["reward"] is None
            ]
            raise RuntimeError(
                "repo-native reward group 包含 masked completion；ms-swift 4.2.2 ORM 不接受 None，"
                f"为避免污染组内归一化已中止本组: {reasons}"
            )
        return [float(value) for value in rewards]


class FormalityConfidenceReward(ORM):
    """仅使用 qa_formality PASS/FAIL logprob margin 的正式 reward。"""

    def __init__(
        self,
        args: Any = None,
        *,
        trace_path: str | Path | None = None,
        score_fn: Callable[..., dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(args, **kwargs)
        self.trace_path = _trace_path(trace_path)
        self._lock = threading.Lock()
        self._reward_call_index = 0
        self.score_fn = score_fn or self._build_score_fn()

    def _next_call_index(self) -> int:
        with self._lock:
            value = self._reward_call_index
            self._reward_call_index += 1
        return value

    @staticmethod
    def _build_score_fn() -> Callable[..., dict[str, Any]]:
        from training.grpo_v3_formality_reward import make_formality_score_fn

        return make_formality_score_fn(
            review_model_id=os.environ.get("EGOQA_REVIEW_MODEL", "Qwen/Qwen3-VL-8B-Instruct"),
            review_base_url=os.environ.get("EGOQA_REVIEW_BASE_URL", "http://127.0.0.1:8001/v1"),
            policy_model_id=os.environ.get("EGOQA_POLICY_MODEL", "Qwen/Qwen3-VL-2B-Instruct"),
            review_max_new_tokens=int(os.environ.get("EGOQA_REVIEW_MAX_NEW_TOKENS", "2048")),
        )

    def __call__(self, completions: Sequence[str], **kwargs: Any) -> list[float]:
        count = len(completions)
        call_index = self._next_call_index()
        packets = _expand(kwargs["packet_json"], count, name="packet_json")
        evidence_ids = _expand(kwargs["evidence_id"], count, name="evidence_id")
        question_types = _expand(kwargs["question_type"], count, name="question_type")
        generation_modes = _expand(kwargs["generation_mode"], count, name="generation_mode")
        eval_ids = {
            item.strip()
            for item in os.environ.get("EGOQA_EVAL_EVIDENCE_IDS", "").split(",")
            if item.strip()
        }
        phases = ["eval" if str(item) in eval_ids else "train" for item in evidence_ids]
        if len(set(phases)) != 1:
            rows = [
                {
                    "reward_kind": "qa_formality_confidence",
                    "reward_call_index": call_index,
                    "phase": phases[index],
                    "evidence_id": str(evidence_ids[index]),
                    "candidate_index": index,
                    "completion_length_chars": len(str(completions[index])),
                    "reward": None,
                    "record": {
                        "masked": True,
                        "eligible_for_grpo": False,
                        "infrastructure_error": {
                            "type": "MixedPhaseGroupError",
                            "message": "同一 formality reward 调用混入 train/eval evidence",
                        },
                    },
                }
                for index in range(count)
            ]
            _write_rows(self.trace_path, rows, self._lock)
            raise ValueError("同一 formality reward 调用不得混入 train/eval evidence")
        phase = phases[0] if phases else "train"
        rewards: list[float] = []
        rows: list[dict[str, Any]] = []
        for index, completion in enumerate(completions):
            try:
                packet_value = packets[index]
                packet = json.loads(packet_value) if isinstance(packet_value, str) else packet_value
                result = self.score_fn(
                    raw_completion=str(completion),
                    packet=packet,
                    evidence_id=str(evidence_ids[index]),
                    question_type=str(question_types[index]),
                    generation_mode=str(generation_modes[index]),
                    candidate_index=index,
                )
                reward = result.get("reward")
                if reward is None or not math.isfinite(float(reward)):
                    raise ValueError(f"qa_formality reward 非有限: {reward}")
                reward_value = float(reward)
            except Exception as exc:
                error_type = (
                    "NonFiniteRewardError"
                    if isinstance(exc, ValueError) and "reward 非有限" in str(exc)
                    else type(exc).__name__
                )
                error_row = {
                    "reward_kind": "qa_formality_confidence",
                    "reward_call_index": call_index,
                    "phase": phase,
                    "evidence_id": str(evidence_ids[index]),
                    "candidate_index": index,
                    "completion_length_chars": len(str(completion)),
                    "reward": None,
                    "record": {
                        "masked": True,
                        "eligible_for_grpo": False,
                        "infrastructure_error": {
                            "type": error_type,
                            "message": str(exc),
                        },
                    },
                }
                _write_rows(self.trace_path, [*rows, error_row], self._lock)
                raise
            rewards.append(reward_value)
            rows.append(
                {
                    "reward_kind": "qa_formality_confidence",
                    "reward_call_index": call_index,
                    "phase": phase,
                    "evidence_id": str(evidence_ids[index]),
                    "candidate_index": index,
                    "completion_length_chars": len(str(completion)),
                    "reward": reward_value,
                    "record": result.get("record", {}),
                }
            )
        _write_rows(self.trace_path, rows, self._lock)
        return rewards


class AnswerMarginReward(ORM):
    """双原生视频 frozen-scorer answer-margin ORM。"""

    def __init__(
        self,
        args: Any = None,
        *,
        trace_path: str | Path | None = None,
        score_fn: Callable[..., dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(args, **kwargs)
        self.trace_path = _trace_path(trace_path)
        self._lock = threading.Lock()
        self._reward_call_index = 0
        self.score_fn = score_fn or self._build_score_fn()

    def _next_call_index(self) -> int:
        with self._lock:
            value = self._reward_call_index
            self._reward_call_index += 1
        return value

    @staticmethod
    def _build_score_fn() -> Callable[..., dict[str, Any]]:
        from training.grpo_v3_answer_margin_reward import make_answer_margin_score_fn

        base_url = os.environ.get("EGOQA_ANSWER_SCORER_BASE_URL")
        timeout = os.environ.get("EGOQA_ANSWER_SCORER_TIMEOUT_SECONDS")
        if not base_url:
            raise RuntimeError("缺少 EGOQA_ANSWER_SCORER_BASE_URL")
        if not timeout:
            raise RuntimeError("缺少 EGOQA_ANSWER_SCORER_TIMEOUT_SECONDS")
        try:
            timeout_seconds = float(timeout)
        except ValueError as exc:
            raise RuntimeError("EGOQA_ANSWER_SCORER_TIMEOUT_SECONDS 必须是数字") from exc
        return make_answer_margin_score_fn(base_url=base_url, timeout_seconds=timeout_seconds)

    def __call__(self, completions: Sequence[str], **kwargs: Any) -> list[float]:
        from training.grpo_v3_answer_margin import (
            ANSWER_MARGIN_REWARD_REVISION,
            PermutationKey,
        )
        from training.grpo_v3_answer_margin_reward import (
            EXPERIMENT_CONDITION_ID,
            EXPERIMENT_REVISION,
            TRACE_SCHEMA_VERSION,
            resolve_ordered_videos,
            summarize_packet,
            video_input_summary,
        )

        count = len(completions)
        call_index = self._next_call_index()

        def single_available(name: str) -> Any:
            value = kwargs.get(name)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                return value[0] if len(value) == 1 else None
            return value

        def metadata_failure_row(error: Exception) -> dict[str, Any]:
            raw_packet = single_available("packet_json")
            packet_summary = None
            video_inputs: list[dict[str, str]] = []
            if raw_packet is not None:
                try:
                    packet_value = json.loads(raw_packet) if isinstance(raw_packet, str) else raw_packet
                except Exception:
                    packet_value = raw_packet
                packet_summary = {
                    "metadata_snapshot": _answer_margin_metadata_snapshot(packet_value)
                }
                if isinstance(packet_value, dict):
                    users = packet_value.get("required_users")
                    clips = packet_value.get("clips")
                    if isinstance(users, list) and isinstance(clips, list):
                        by_user = {
                            str(clip.get("agent_name")): clip.get("local_video")
                            for clip in clips
                            if isinstance(clip, dict)
                        }
                        if len(users) == 2 and all(str(user) in by_user for user in users):
                            video_inputs = []
                            for user in users:
                                raw_path = by_user[str(user)]
                                safe_path = _answer_margin_metadata_snapshot(raw_path)
                                video_inputs.append({
                                    "user": str(user),
                                    "path": safe_path,
                                    "basename": (
                                        Path(str(raw_path)).name
                                        if isinstance(raw_path, (str, Path))
                                        else None
                                    ),
                                })
            step = single_available("global_step")
            if isinstance(step, bool) or not isinstance(step, int) or step < 0:
                step = None
            evidence = single_available("evidence_id")
            return {
                "schema_version": TRACE_SCHEMA_VERSION,
                "reward_revision": ANSWER_MARGIN_REWARD_REVISION,
                "experiment_version": EXPERIMENT_REVISION,
                "experiment_condition_id": EXPERIMENT_CONDITION_ID,
                "reward_kind": "combined_video_answer_margin",
                "phase": None,
                "global_step": step,
                "reward_call_index": call_index,
                "candidate_index": None,
                "evidence_id": (
                    _answer_margin_metadata_snapshot(evidence)
                    if evidence is not None else None
                ),
                "raw_completion": None,
                "raw_completions": [
                    _answer_margin_metadata_snapshot(item) for item in completions
                ],
                "packet_summary": packet_summary,
                "video_inputs": video_inputs,
                "permutation_key": None,
                "available_metadata": _answer_margin_metadata_snapshot({
                    name: kwargs.get(name)
                    for name in (
                        "packet_json", "evidence_id", "question_type",
                        "generation_mode", "global_step",
                    )
                    if name in kwargs
                }),
                "failure_stage": "metadata",
                "reward": None,
                "record": {
                    "masked": True,
                    "eligible_for_grpo": False,
                    "infrastructure_error": {
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                },
            }

        if count != 4:
            error = CompletionGroupSizeError(
                f"answer-margin 固定要求4个completions，实际count={count}"
            )
            row = metadata_failure_row(error)
            row["expected_completion_count"] = 4
            row["actual_completion_count"] = count
            _write_rows(self.trace_path, [row], self._lock)
            raise error

        try:
            packets = _expand(kwargs["packet_json"], count, name="packet_json")
            evidence_ids = _expand(kwargs["evidence_id"], count, name="evidence_id")
            question_types = _expand(kwargs["question_type"], count, name="question_type")
            generation_modes = _expand(kwargs["generation_mode"], count, name="generation_mode")
            if "global_step" not in kwargs:
                raise ValueError("缺少 ms-swift ORM 展开字段 global_step")
            global_steps = _expand(kwargs["global_step"], count, name="global_step")
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in global_steps
            ):
                raise ValueError("global_step 必须逐 completion 展开为非负整数")
        except Exception as exc:
            _write_rows(self.trace_path, [metadata_failure_row(exc)], self._lock)
            raise

        eval_ids = {
            item.strip()
            for item in os.environ.get("EGOQA_EVAL_EVIDENCE_IDS", "").split(",")
            if item.strip()
        }
        phases = ["eval" if str(item) in eval_ids else "train" for item in evidence_ids]

        def key_payload(key: PermutationKey | None) -> dict[str, Any] | None:
            if key is None:
                return None
            return {
                "reward_revision": key.reward_revision,
                "experiment_condition_id": key.experiment_condition_id,
                "phase": key.phase,
                "evidence_id": key.evidence_id,
                "generation_seed_or_call_index": key.generation_seed_or_call_index,
                "candidate_index": key.candidate_index,
            }

        def base_row(
            index: int,
            *,
            global_step: int | None,
            packet: Any = None,
            videos: tuple[str, str] | None = None,
            key: PermutationKey | None = None,
        ) -> dict[str, Any]:
            if packet is None:
                raw_packet = packets[index]
                try:
                    packet = json.loads(raw_packet) if isinstance(raw_packet, str) else raw_packet
                except Exception:
                    packet = raw_packet
            video_inputs: list[dict[str, str]] = []
            if isinstance(packet, dict) and videos is not None:
                video_inputs = video_input_summary(packet, videos)
            elif isinstance(packet, dict):
                users = packet.get("required_users")
                clips = packet.get("clips")
                if isinstance(users, list) and isinstance(clips, list):
                    by_user = {
                        str(clip.get("agent_name")): clip.get("local_video")
                        for clip in clips
                        if isinstance(clip, dict)
                    }
                    if len(users) == 2 and all(str(user) in by_user for user in users):
                        video_inputs = [
                            {
                                "user": str(user),
                                "path": str(by_user[str(user)]),
                                "basename": Path(str(by_user[str(user)])).name,
                            }
                            for user in users
                        ]
            return {
                "schema_version": TRACE_SCHEMA_VERSION,
                "reward_revision": ANSWER_MARGIN_REWARD_REVISION,
                "experiment_version": EXPERIMENT_REVISION,
                "experiment_condition_id": EXPERIMENT_CONDITION_ID,
                "reward_kind": "combined_video_answer_margin",
                "phase": phases[index],
                "global_step": global_step,
                "reward_call_index": call_index,
                "candidate_index": index,
                "evidence_id": str(evidence_ids[index]),
                "raw_completion": str(completions[index]),
                "completion_length_chars": len(str(completions[index])),
                "question_type": str(question_types[index]),
                "generation_mode": str(generation_modes[index]),
                "packet_summary": summarize_packet(packet),
                "video_inputs": video_inputs,
                "permutation_key": key_payload(key),
            }

        def masked_row(
            index: int,
            error: Exception,
            *,
            stage: str,
            global_step: int | None,
            packet: Any = None,
            videos: tuple[str, str] | None = None,
            key: PermutationKey | None = None,
            prior_record: Any = None,
        ) -> dict[str, Any]:
            error_type = (
                "NonFiniteRewardError"
                if isinstance(error, ValueError) and "reward 非有限" in str(error)
                else type(error).__name__
            )
            record = dict(prior_record) if isinstance(prior_record, dict) else {}
            record.update({
                "masked": True,
                "eligible_for_grpo": False,
                "infrastructure_error": {"type": error_type, "message": str(error)},
            })
            return {
                **base_row(
                    index,
                    global_step=global_step,
                    packet=packet,
                    videos=videos,
                    key=key,
                ),
                "failure_stage": stage,
                "reward": None,
                "record": record,
            }

        def preview_global_step() -> int | None:
            raw = kwargs.get("global_step")
            if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
                raw = raw[0] if raw else None
            return raw if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0 else None

        condition_value = os.environ.get(
            "EGOQA_ANSWER_MARGIN_CONDITION_ID", EXPERIMENT_CONDITION_ID
        )
        if condition_value != EXPERIMENT_CONDITION_ID:
            error = ValueError("EGOQA_ANSWER_MARGIN_CONDITION_ID 必须严格等于 t05")
            _write_rows(
                self.trace_path,
                [masked_row(
                    0,
                    error,
                    stage="configuration",
                    global_step=preview_global_step(),
                )],
                self._lock,
            )
            raise error

        if len(set(phases)) != 1:
            error = ValueError("同一 answer-margin reward 调用不得混入 train/eval evidence")
            rows = [masked_row(
                index,
                error,
                stage="metadata",
                global_step=int(global_steps[index]),
            ) for index in range(count)]
            _write_rows(self.trace_path, rows, self._lock)
            raise error
        phase = phases[0] if phases else "train"
        rewards: list[float] = []
        rows: list[dict[str, Any]] = []
        for index, completion in enumerate(completions):
            packet = None
            videos = None
            key = None
            result = None
            stage = "packet_json_parse"
            try:
                key = PermutationKey(
                    experiment_condition_id=EXPERIMENT_CONDITION_ID,
                    phase=phase,
                    evidence_id=str(evidence_ids[index]),
                    generation_seed_or_call_index=call_index,
                    candidate_index=index,
                    reward_revision=ANSWER_MARGIN_REWARD_REVISION,
                )
                packet_value = packets[index]
                packet = json.loads(packet_value) if isinstance(packet_value, str) else packet_value
                stage = "media_mapping"
                videos = resolve_ordered_videos(packet, str(evidence_ids[index]))
                stage = "scoring"
                result = self.score_fn(
                    raw_completion=str(completion),
                    packet=packet,
                    evidence_id=str(evidence_ids[index]),
                    question_type=str(question_types[index]),
                    generation_mode=str(generation_modes[index]),
                    candidate_index=index,
                    key=key,
                    global_step=int(global_steps[index]),
                    reward_call_index=call_index,
                )
                stage = "response_validation"
                reward = result.get("reward")
                if reward is None or not math.isfinite(float(reward)):
                    raise ValueError(f"answer-margin reward 非有限: {reward}")
                reward_value = float(reward)
            except Exception as exc:
                error_row = masked_row(
                    index,
                    exc,
                    stage=stage,
                    global_step=int(global_steps[index]),
                    packet=packet,
                    videos=videos,
                    key=key,
                    prior_record=result.get("record") if isinstance(result, dict) else None,
                )
                _write_rows(self.trace_path, [*rows, error_row], self._lock)
                raise
            rewards.append(reward_value)
            rows.append({
                **base_row(
                    index,
                    global_step=int(global_steps[index]),
                    packet=packet,
                    videos=videos,
                    key=key,
                ),
                "failure_stage": None,
                "reward": reward_value,
                "record": result.get("record", {}),
            })
        _write_rows(self.trace_path, rows, self._lock)
        return rewards


orms["egoqa_gate1_controlled"] = ControlledGateReward
orms["egoqa_repo_native_judge"] = RepoNativeJudgeReward
orms["egoqa_qa_formality_confidence"] = FormalityConfidenceReward
orms["egoqa_combined_video_answer_margin"] = AnswerMarginReward
