"""Training-split class-weighted cross-entropy for binary Reviewer v2."""

from __future__ import annotations

from typing import Any, Mapping

from .modeling import ReviewerOutput

LOSS_FIELDS = {
    "evidence_quality": ("evidence_logits", "evidence_loss"),
    "answerability": ("answerability_logits", "answerability_loss"),
    "qa_formality": ("formality_logits", "formality_loss"),
}


def active_loss_names(active_heads: tuple[str, ...]) -> tuple[str, ...]:
    unsupported = [name for name in active_heads if name not in LOSS_FIELDS]
    if unsupported:
        raise ValueError(f"unsupported active head: {unsupported}")
    if not active_heads:
        raise ValueError("at least one active head is required")
    return tuple(LOSS_FIELDS[name][1] for name in active_heads)


def human_score_to_binary_label(grade: object) -> int:
    if type(grade) is not int or grade not in {1, 2, 3}:
        raise ValueError("human score must be integer 1, 2, or 3")
    return int(grade >= 2)


def validate_class_weights(
    active_heads: tuple[str, ...], class_weights: Mapping[str, Any]
) -> dict[str, tuple[float, float]]:
    import math

    active_loss_names(active_heads)
    unsupported = sorted(set(class_weights) - set(LOSS_FIELDS))
    if unsupported:
        raise ValueError(f"unsupported class-weight fields: {unsupported}")
    missing = [name for name in active_heads if name not in class_weights]
    if missing:
        raise ValueError(f"missing class weights: {missing}")
    result: dict[str, tuple[float, float]] = {}
    for name in active_heads:
        value = class_weights[name]
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"class weights for {name} must be [fail_weight, pass_weight]")
        pair = (float(value[0]), float(value[1]))
        if any(not math.isfinite(weight) or weight <= 0.0 for weight in pair):
            raise ValueError("class weights must be finite and positive")
        result[name] = pair
    return result


def reviewer_losses(
    output: ReviewerOutput,
    labels: Mapping[str, Any],
    *,
    active_heads: tuple[str, ...] = ("evidence_quality", "answerability", "qa_formality"),
    class_weights: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional
    weights = validate_class_weights(
        active_heads,
        class_weights or {field: (1.0, 1.0) for field in active_heads},
    )
    result: dict[str, Any] = {}
    for field in active_heads:
        logits_name, loss_name = LOSS_FIELDS[field]
        logits = getattr(output, logits_name)
        if logits is None:
            raise ValueError(f"active head {field} has no class logits")
        if logits.ndim != 2 or logits.shape[-1] != 2:
            raise ValueError(f"active head {field} must emit [fail, pass] logits")
        targets = labels[field]
        if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
            raise ValueError(f"active head {field} logits and labels have different batch shapes")
        if targets.numel() and not bool(torch.all((targets == 0) | (targets == 1))):
            raise ValueError("binary labels must be 0 (fail) or 1 (pass)")
        weight_tensor = torch.tensor(weights[field], device=logits.device, dtype=logits.dtype)
        # Use a fixed sample-count denominator. PyTorch's weighted-mean reduction
        # divides by the selected weights, which would cancel class weighting for
        # the current one-candidate training steps.
        result[loss_name] = functional.cross_entropy(
            logits, targets.long(), weight=weight_tensor, reduction="none"
        ).mean()
    values = [result[LOSS_FIELDS[field][1]] for field in active_heads]
    result["loss"] = values[0] if len(values) == 1 else sum(values) / len(values)
    return {"loss": result.pop("loss"), **result}
