"""已归档的 qa_formality 置信度 reward 插件。"""

from __future__ import annotations

import json
import math
import os
import threading
from pathlib import Path
from typing import Any, Callable, Sequence

from training.grpo_v3.runtime.reward_plugin import ORM, _expand, _trace_path, _write_rows, orms


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
        from training.grpo_v3.experiments.archived.formality.reward import make_formality_score_fn

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


orms["egoqa_qa_formality_confidence"] = FormalityConfidenceReward
