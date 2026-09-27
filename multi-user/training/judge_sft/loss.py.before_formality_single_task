"""Verdict-token BCE and imbalance correction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .contracts import (
    VERDICT_ASSISTANT_PREFIX,
    JudgeTask,
    Verdict,
    validate_task_weights,
)
from .data import JudgeExample


@dataclass(frozen=True)
class BinaryClassWeights:
    fail: float
    passed: float
    fail_count: int
    pass_count: int

    def for_target(self, target: int) -> float:
        if target == 0:
            return self.fail
        if target == 1:
            return self.passed
        raise ValueError(f"binary target must be 0 or 1, got {target}")


def balanced_binary_class_weights(
    targets: Iterable[int],
    *,
    smoothing: float = 1.0,
    max_weight: float = 10.0,
) -> BinaryClassWeights:
    """Return symmetric inverse-frequency weights with mean observed weight one."""

    values = [int(value) for value in targets]
    if any(value not in {0, 1} for value in values):
        raise ValueError("targets must contain only 0/1")
    fail_count = values.count(0)
    pass_count = values.count(1)
    if fail_count == 0 or pass_count == 0:
        raise ValueError(
            "both pass and fail examples are required to estimate class weights; "
            f"fail={fail_count} pass={pass_count}"
        )
    if not math.isfinite(smoothing) or smoothing < 0:
        raise ValueError("smoothing must be finite and non-negative")
    if not math.isfinite(max_weight) or max_weight < 1:
        raise ValueError("max_weight must be finite and at least one")
    smoothed_total = fail_count + pass_count + 2.0 * smoothing
    raw_fail = smoothed_total / (2.0 * (fail_count + smoothing))
    raw_pass = smoothed_total / (2.0 * (pass_count + smoothing))
    observed_mean = (fail_count * raw_fail + pass_count * raw_pass) / len(values)
    normalized_fail = raw_fail / observed_mean
    normalized_pass = raw_pass / observed_mean
    if normalized_fail > max_weight:
        normalized_fail = max_weight
        normalized_pass = (
            len(values) - fail_count * normalized_fail
        ) / pass_count
    elif normalized_pass > max_weight:
        normalized_pass = max_weight
        normalized_fail = (
            len(values) - pass_count * normalized_pass
        ) / fail_count
    if normalized_fail <= 0 or normalized_pass <= 0:
        raise ValueError(
            "max_weight is too small to preserve a positive mean-one class weighting"
        )
    return BinaryClassWeights(
        fail=normalized_fail,
        passed=normalized_pass,
        fail_count=fail_count,
        pass_count=pass_count,
    )


def class_weights_by_task(
    examples: Iterable[JudgeExample],
    *,
    smoothing: float = 1.0,
    max_weight: float = 10.0,
) -> dict[JudgeTask, BinaryClassWeights]:
    rows = list(examples)
    result: dict[JudgeTask, BinaryClassWeights] = {}
    for task in JudgeTask:
        targets = [row.target for row in rows if row.task is task]
        if not targets:
            raise ValueError(f"training manifest has no {task.value} examples")
        result[task] = balanced_binary_class_weights(
            targets,
            smoothing=smoothing,
            max_weight=max_weight,
        )
    return result


def task_sampling_scales(
    examples: Iterable[JudgeExample],
    task_weights: Mapping[JudgeTask | str, float],
) -> dict[JudgeTask, float]:
    """Scale naturally sampled examples into an unbiased weighted task mean.

    For a full epoch, ``mean(scale_t * loss_i)`` equals
    ``sum_t task_weight_t * mean_t(loss_i)``.  This prevents the two
    answerability conditions from silently doubling answerability's 0.4 share.
    """

    rows = list(examples)
    if not rows:
        raise ValueError("cannot compute task scales for an empty dataset")
    normalized = validate_task_weights(task_weights)
    counts = {task: sum(row.task is task for row in rows) for task in JudgeTask}
    missing = [task.value for task, count in counts.items() if count == 0]
    if missing:
        raise ValueError(f"training manifest is missing tasks: {missing}")
    total = len(rows)
    return {
        task: normalized[task] * total / counts[task]
        for task in JudgeTask
    }


def sample_weight_for_example(
    example: JudgeExample,
    *,
    class_weights: Mapping[JudgeTask, BinaryClassWeights],
    task_scales: Mapping[JudgeTask, float],
) -> float:
    return float(example.loss_weight_multiplier) * (
        class_weights[example.task].for_target(example.target)
        * float(task_scales[example.task])
    )


def _decode_ids(tokenizer: Any, token_ids: list[int]) -> str:
    return str(
        tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    )


def resolve_verdict_token_ids(
    tokenizer: Any,
    *,
    assistant_prefix: str = VERDICT_ASSISTANT_PREFIX,
) -> dict[Verdict, int]:
    """Require pass/fail to be distinct one-token continuations of the exact prefix."""

    result: dict[Verdict, int] = {}
    prefix_ids = tokenizer.encode(assistant_prefix, add_special_tokens=False)
    if not prefix_ids:
        raise RuntimeError("assistant verdict prefix encoded to no tokens")
    decoded_prefix = _decode_ids(tokenizer, [int(value) for value in prefix_ids])
    for verdict in Verdict:
        encoded = tokenizer.encode(verdict.value, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(
                f"{verdict.value!r} must encode as one token for verdict-token BCE; "
                f"got token_ids={encoded}"
            )
        token_id = int(encoded[0])
        decoded = _decode_ids(tokenizer, [token_id]).strip().lower()
        if decoded != verdict.value:
            raise RuntimeError(
                f"token {token_id} decodes to {decoded!r}, expected {verdict.value!r}"
            )
        continued = _decode_ids(
            tokenizer,
            [int(value) for value in prefix_ids] + [token_id],
        )
        expected = decoded_prefix + verdict.value
        if continued != expected:
            raise RuntimeError(
                f"token {token_id} is not the exact {verdict.value!r} continuation "
                f"after assistant prefix {assistant_prefix!r}: got {continued!r}, "
                f"expected {expected!r}"
            )
        result[verdict] = token_id
    if result[Verdict.FAIL] == result[Verdict.PASS]:
        raise RuntimeError("pass and fail resolved to the same token id")
    return result


def verdict_pair_logits(
    final_token_logits: Any,
    *,
    fail_token_id: int,
    pass_token_id: int,
) -> Any:
    """Select logits in stable ``[fail, pass]`` order."""

    return final_token_logits[..., [int(fail_token_id), int(pass_token_id)]]


def verdict_bce_from_pair_logits(
    pair_logits: Any,
    targets: Any,
    *,
    sample_weights: Any | None = None,
) -> Any:
    """BCE over ``logit(pass) - logit(fail)``.

    This is algebraically the same decision loss as two-class cross entropy but
    uses the language model's existing vocabulary logits and adds no head.
    """

    import torch
    import torch.nn.functional as functional

    if pair_logits.ndim != 2 or pair_logits.shape[-1] != 2:
        raise ValueError(f"pair_logits must have shape [batch, 2], got {pair_logits.shape}")
    target_tensor = targets.to(device=pair_logits.device, dtype=pair_logits.dtype).view(-1)
    if target_tensor.shape[0] != pair_logits.shape[0]:
        raise ValueError("targets and pair_logits batch sizes differ")
    margin = pair_logits[:, 1] - pair_logits[:, 0]
    losses = functional.binary_cross_entropy_with_logits(
        margin,
        target_tensor,
        reduction="none",
    )
    if sample_weights is not None:
        weights = sample_weights.to(device=losses.device, dtype=losses.dtype).view(-1)
        if weights.shape != losses.shape:
            raise ValueError("sample_weights and losses must have equal shape")
        losses = losses * weights
    return losses.mean()
