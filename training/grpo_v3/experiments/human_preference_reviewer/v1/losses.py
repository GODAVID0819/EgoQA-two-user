"""Absolute three-class supervision for Reviewer v1."""

from __future__ import annotations

from typing import Any, Mapping

from .modeling import ReviewerOutput


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


def reviewer_losses(output: ReviewerOutput, labels: Mapping[str, Any]) -> dict[str, Any]:
    import torch.nn.functional as functional

    evidence = functional.cross_entropy(
        output.evidence_logits, grades_to_targets(labels["evidence_quality"])
    )
    answerability = functional.cross_entropy(
        output.answerability_logits, grades_to_targets(labels["answerability"])
    )
    formality = functional.cross_entropy(
        output.formality_logits, grades_to_targets(labels["qa_formality"])
    )
    total = (evidence + answerability + formality) / 3.0
    return {
        "loss": total,
        "evidence_loss": evidence,
        "answerability_loss": answerability,
        "formality_loss": formality,
    }
