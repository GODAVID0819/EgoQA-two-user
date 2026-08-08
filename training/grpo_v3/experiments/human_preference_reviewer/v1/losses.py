"""Absolute three-class supervision for Reviewer v1."""

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


def grade_to_target(grade: object) -> int:
    if type(grade) is not int or grade not in {1, 2, 3}:
        raise ValueError("human grade must be integer 1, 2, or 3")
    return grade - 1


def mean_three_losses_reference(evidence: float, answerability: float, formality: float) -> float:
    return (float(evidence) + float(answerability) + float(formality)) / 3.0


def grades_to_targets(grades: Any) -> Any:
    import torch

    if grades.numel() and not bool(torch.all((grades >= 1) & (grades <= 3))):
        raise ValueError("human grades must be 1, 2, or 3")
    return grades.long() - 1


def reviewer_losses(
    output: ReviewerOutput,
    labels: Mapping[str, Any],
    *,
    active_heads: tuple[str, ...] = ("evidence_quality", "answerability", "qa_formality"),
) -> dict[str, Any]:
    import torch.nn.functional as functional
    active_loss_names(active_heads)
    result: dict[str, Any] = {}
    for field in active_heads:
        logits_name, loss_name = LOSS_FIELDS[field]
        logits = getattr(output, logits_name)
        if logits is None:
            raise ValueError(f"active head {field} has no logits")
        result[loss_name] = functional.cross_entropy(
            logits, grades_to_targets(labels[field])
        )
    values = list(result.values())
    result["loss"] = values[0] if len(values) == 1 else sum(values) / float(len(values))
    return {"loss": result.pop("loss"), **result}
