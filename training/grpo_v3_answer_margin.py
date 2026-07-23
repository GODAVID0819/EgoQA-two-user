"""GRPO v3 的宽松 QA 提取、确定性选项排列和答案 margin。"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping

from training.grpo_v3.shared.json_format import FormatValidationResult, validate_completion_json


LABELS = ("A", "B", "C", "D", "E")
MARGIN_CLIP = 8.0
ANSWER_MARGIN_REWARD_REVISION = "combined_video_answer_margin_v1"
REWARD_REVISION = ANSWER_MARGIN_REWARD_REVISION


@dataclass(frozen=True)
class CoreQAExtraction:
    ok: bool
    status: str
    question: str | None
    options: list[str] | None
    correct: str | None
    failure_reason: str | None
    format_validation: FormatValidationResult
    raw_completion: str
    inner_format_status: str

    def as_qa(self) -> dict[str, Any] | None:
        if not self.ok:
            return None
        return {
            "question": self.question,
            "options": list(self.options) if self.options is not None else None,
            "correct": self.correct,
        }


def _first_complete_object(text: str) -> str | None:
    start: int | None = None
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if start is None:
                start = index
            depth += 1
        elif char == "}" and start is not None:
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
            if depth < 0:
                return None
    if start is None and in_string:
        search_start = 0
        while True:
            fallback_start = text.find("{", search_start)
            if fallback_start < 0:
                break
            candidate = _first_complete_object(text[fallback_start:])
            if (
                candidate is not None
                and validate_completion_json(candidate).value is not None
            ):
                return candidate
            search_start = fallback_start + 1
    return None


def extract_core_qa(raw_completion: str) -> CoreQAExtraction:
    raw = str(raw_completion)
    validation = validate_completion_json(raw)
    extra_text_recovered = False
    if validation.value is None:
        object_text = _first_complete_object(raw)
        if object_text is not None and object_text.strip() != raw.strip():
            validation = validate_completion_json(object_text)
            extra_text_recovered = validation.value is not None
    status = "extra_text_recovered" if extra_text_recovered else validation.status
    value = validation.value
    if value is None:
        return CoreQAExtraction(
            ok=False,
            status=status,
            question=None,
            options=None,
            correct=None,
            failure_reason="unrecoverable_json",
            format_validation=validation,
            raw_completion=raw,
            inner_format_status=validation.status,
        )
    question = value.get("question")
    options = value.get("options")
    correct = value.get("correct")
    normalized_question = question.strip() if isinstance(question, str) else None
    normalized_options = (
        [option.strip() for option in options]
        if isinstance(options, list) and all(isinstance(option, str) for option in options)
        else None
    )
    normalized_correct = correct.strip().upper() if isinstance(correct, str) else None
    if not normalized_question:
        reason = "invalid_question"
    elif (
        normalized_options is None
        or len(normalized_options) != 5
        or any(not option for option in normalized_options)
    ):
        reason = "invalid_options"
    elif normalized_correct not in LABELS or len(normalized_correct) != 1:
        reason = "invalid_correct"
    else:
        return CoreQAExtraction(
            ok=True,
            status=status,
            question=normalized_question,
            options=normalized_options,
            correct=normalized_correct,
            failure_reason=None,
            format_validation=validation,
            raw_completion=raw,
            inner_format_status=validation.status,
        )
    return CoreQAExtraction(
        ok=False,
        status=status,
        question=normalized_question,
        options=normalized_options,
        correct=normalized_correct,
        failure_reason=reason,
        format_validation=validation,
        raw_completion=raw,
        inner_format_status=validation.status,
    )


@dataclass(frozen=True)
class PermutationKey:
    experiment_condition_id: str
    phase: str
    evidence_id: str
    generation_seed_or_call_index: str | int
    candidate_index: int
    reward_revision: str

    def stable_text(self) -> str:
        return json.dumps(
            {
                "experiment_condition_id": self.experiment_condition_id,
                "phase": self.phase,
                "evidence_id": self.evidence_id,
                "generation_seed_or_call_index": self.generation_seed_or_call_index,
                "candidate_index": self.candidate_index,
                "reward_revision": self.reward_revision,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )


@dataclass(frozen=True)
class OptionPermutation:
    permuted_options: list[str]
    mapped_correct: str
    permutation: list[int]
    inverse: list[int]
    digests: list[str]

    @property
    def options(self) -> list[str]:
        return list(self.permuted_options)

    @property
    def correct(self) -> str:
        return self.mapped_correct


def permute_options(
    options: list[str],
    correct: str,
    key: PermutationKey,
) -> OptionPermutation:
    if (
        not isinstance(options, list)
        or len(options) != len(LABELS)
        or any(not isinstance(option, str) or not option.strip() for option in options)
    ):
        raise ValueError("options must contain exactly five non-empty strings")
    if correct not in LABELS:
        raise ValueError("correct must be one of A-E")
    if not isinstance(key, PermutationKey):
        raise ValueError("key must be a PermutationKey")

    stable_prefix = key.stable_text().encode("utf-8") + b"\0"
    digests = [
        hashlib.sha256(stable_prefix + str(index).encode("ascii")).hexdigest()
        for index in range(len(options))
    ]
    permutation = sorted(range(len(options)), key=lambda index: (digests[index], index))
    inverse = [0] * len(options)
    for new_index, old_index in enumerate(permutation):
        inverse[old_index] = new_index
    old_correct_index = LABELS.index(correct)
    mapped_correct = LABELS[inverse[old_correct_index]]
    return OptionPermutation(
        permuted_options=[options[index] for index in permutation],
        mapped_correct=mapped_correct,
        permutation=permutation,
        inverse=inverse,
        digests=digests,
    )


@dataclass(frozen=True)
class AnswerMargin:
    raw_margin: float
    clipped_margin: float
    reward: float
    log_probabilities: dict[str, float]
    unique_top1: str | None
    tie: bool

    @property
    def raw(self) -> float:
        return self.raw_margin

    @property
    def clipped(self) -> float:
        return self.clipped_margin


def compute_answer_margin(scores: Mapping[str, Real], correct: str) -> AnswerMargin:
    if not isinstance(scores, Mapping) or set(scores) != set(LABELS) or len(scores) != len(LABELS):
        raise ValueError("scores must have exactly the labels A-E")
    if correct not in LABELS:
        raise ValueError("correct must be one of A-E")
    numeric_scores: dict[str, float] = {}
    for label in LABELS:
        value = scores[label]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("scores must be finite real numbers")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("scores must be finite real numbers")
        numeric_scores[label] = numeric

    best_other = max(score for label, score in numeric_scores.items() if label != correct)
    raw_margin = numeric_scores[correct] - best_other
    if not math.isfinite(raw_margin):
        raise ValueError("answer margin must be finite")
    clipped_margin = max(-MARGIN_CLIP, min(MARGIN_CLIP, raw_margin))
    reward = clipped_margin / MARGIN_CLIP

    maximum = max(numeric_scores.values())
    log_normalizer = maximum + math.log(
        sum(math.exp(score - maximum) for score in numeric_scores.values())
    )
    log_probabilities = {
        label: numeric_scores[label] - log_normalizer for label in LABELS
    }
    derived_values = [
        raw_margin,
        clipped_margin,
        reward,
        log_normalizer,
        *log_probabilities.values(),
    ]
    if not all(math.isfinite(value) for value in derived_values):
        raise ValueError("derived answer-margin statistics must be finite")
    ranked = sorted(LABELS, key=lambda label: numeric_scores[label], reverse=True)
    tie = numeric_scores[ranked[0]] - numeric_scores[ranked[1]] <= 1e-6
    return AnswerMargin(
        raw_margin=raw_margin,
        clipped_margin=clipped_margin,
        reward=reward,
        log_probabilities=log_probabilities,
        unique_top1=None if tie else ranked[0],
        tie=tie,
    )
