"""Stable contracts shared by judge training and inference."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping


VERDICT_ASSISTANT_PREFIX = '{"verdict":"'
VERDICT_CLOSING_QUOTE = '"'
VERDICT_FIELD_PATTERN = re.compile(
    r'"verdict"\s*:\s*"(?P<verdict>pass|fail)"',
    flags=re.IGNORECASE,
)


class Verdict(str, Enum):
    FAIL = "fail"
    PASS = "pass"

    @property
    def target(self) -> int:
        return int(self is Verdict.PASS)


class JudgeTask(str, Enum):
    FORMALITY = "formality"
    GROUNDEDNESS = "groundedness"
    ANSWERABILITY = "answerability"


TASK_IDS: dict[JudgeTask, int] = {
    JudgeTask.FORMALITY: 0,
    JudgeTask.GROUNDEDNESS: 1,
    JudgeTask.ANSWERABILITY: 2,
}

DEFAULT_TASK_WEIGHTS: dict[JudgeTask, float] = {
    JudgeTask.FORMALITY: 0.2,
    JudgeTask.GROUNDEDNESS: 0.4,
    JudgeTask.ANSWERABILITY: 0.4,
}


@dataclass(frozen=True)
class JudgeTrainingDefaults:
    model_id: str = "Qwen/Qwen3.8-27B"
    torch_dtype: str = "bfloat16"
    fps: float = 0.5
    min_pixels: int = 3_136
    max_pixels: int = 262_144
    max_input_tokens: int = 262_144
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    freeze_vision: bool = True
    freeze_aligner: bool = True
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    epochs: float = 3.0
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    class_weight_smoothing: float = 1.0
    max_class_weight: float = 10.0
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULTS = JudgeTrainingDefaults()


def normalize_verdict(value: Any) -> Verdict:
    normalized = str(value or "").strip().lower()
    try:
        return Verdict(normalized)
    except ValueError as exc:
        raise ValueError(f"verdict must be exactly pass or fail, got {value!r}") from exc


def render_locked_verdict_prefix(verdict: Verdict | str) -> str:
    """Return the canonical JSON prefix with both verdict quotes supplied by code."""

    normalized = verdict if isinstance(verdict, Verdict) else normalize_verdict(verdict)
    return f"{VERDICT_ASSISTANT_PREFIX}{normalized.value}{VERDICT_CLOSING_QUOTE}"


def parse_verdict(text: str, *, require_first_field: bool = True) -> Verdict:
    """Parse the pass/fail token following ``verdict`` from generated JSON.

    The strict path requires valid JSON and keeps ``verdict`` as the first field.
    A regex fallback is retained only to report otherwise recoverable generation
    output; it still requires the literal verdict field and an exact pass/fail value.
    """

    candidate = str(text or "").strip()
    if not candidate:
        raise ValueError("judge output is empty")
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        match = VERDICT_FIELD_PATTERN.search(candidate)
        if match is None:
            raise ValueError("judge output does not contain a pass/fail verdict field")
        if require_first_field:
            opening = candidate.find("{")
            first_key = re.search(r'"([^"\\]+)"\s*:', candidate[opening + 1 :])
            if first_key is None or first_key.group(1) != "verdict":
                raise ValueError("verdict must be the first JSON field")
        return normalize_verdict(match.group("verdict"))
    if not isinstance(parsed, dict):
        raise ValueError("judge output must be a JSON object")
    if require_first_field and (not parsed or next(iter(parsed)) != "verdict"):
        raise ValueError("verdict must be the first JSON field")
    return normalize_verdict(parsed.get("verdict"))


def parse_complete_reason_fix_output(text: str) -> dict[str, Any]:
    """Validate the complete formality/groundedness output contract."""

    candidate = str(text or "").strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"judge output is not complete valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("judge output must be a JSON object")
    if list(parsed) != ["verdict", "reason", "fix"]:
        raise ValueError("judge output fields must be exactly verdict, reason, fix in order")
    verdict = normalize_verdict(parsed["verdict"])
    reason = parsed["reason"]
    fix = parsed["fix"]
    if verdict is Verdict.PASS:
        if reason is not None or fix is not None:
            raise ValueError("pass requires reason and fix to both be null")
    else:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("fail requires a non-empty reason")
        if not isinstance(fix, str) or not fix.strip():
            raise ValueError("fail requires a non-empty fix")
        if len(reason.split()) > 40 or len(fix.split()) > 40:
            raise ValueError("fail reason and fix must each contain at most 40 words")
    return parsed


def validate_task_weights(
    weights: Mapping[JudgeTask | str, float],
) -> dict[JudgeTask, float]:
    normalized: dict[JudgeTask, float] = {}
    for raw_task, raw_weight in weights.items():
        task = raw_task if isinstance(raw_task, JudgeTask) else JudgeTask(str(raw_task))
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"task weight for {task.value} must be finite and positive")
        normalized[task] = weight
    if set(normalized) != set(JudgeTask):
        missing = sorted(task.value for task in set(JudgeTask) - set(normalized))
        extra = sorted(str(task) for task in set(normalized) - set(JudgeTask))
        raise ValueError(f"task weights must cover all judges; missing={missing} extra={extra}")
    total = sum(normalized.values())
    return {task: weight / total for task, weight in normalized.items()}
