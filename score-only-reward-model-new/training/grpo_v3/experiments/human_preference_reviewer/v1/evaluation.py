"""Metrics for binary fail/pass reviewer predictions."""

from __future__ import annotations

import math
from typing import Any, Sequence


def _binary_auroc(labels: Sequence[int], probabilities: Sequence[float]) -> float | None:
    positive = sum(label == 1 for label in labels)
    negative = len(labels) - positive
    if positive == 0 or negative == 0:
        return None
    favorable = 0.0
    for label, probability in zip(labels, probabilities):
        if label != 1:
            continue
        for other_label, other_probability in zip(labels, probabilities):
            if other_label != 0:
                continue
            favorable += float(probability > other_probability)
            favorable += 0.5 * float(probability == other_probability)
    return favorable / (positive * negative)


def binary_metrics(
    labels: Sequence[int],
    pass_probabilities: Sequence[float],
    *,
    loss: float | None = None,
) -> dict[str, Any]:
    if len(labels) != len(pass_probabilities) or not labels:
        raise ValueError("labels and pass probabilities must have the same non-zero length")
    if any(type(label) is not int or label not in {0, 1} for label in labels):
        raise ValueError("labels must be 0 (fail) or 1 (pass)")
    probabilities = [float(value) for value in pass_probabilities]
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities):
        raise ValueError("pass probabilities must be finite and in [0, 1]")
    predictions = [int(value >= 0.5) for value in probabilities]
    confusion = [[0, 0], [0, 0]]
    for label, prediction in zip(labels, predictions):
        confusion[label][prediction] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    recalls: list[float] = []
    f1_values: list[float] = []
    for label, name in ((0, "fail"), (1, "pass")):
        true_positive = confusion[label][label]
        support = sum(confusion[label])
        predicted_count = sum(row[label] for row in confusion)
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {
            "label": label,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "predicted_count": predicted_count,
        }
        if support:
            recalls.append(recall)
        f1_values.append(f1)

    return {
        "count": len(labels),
        "loss": loss,
        "accuracy": sum(a == b for a, b in zip(labels, predictions)) / len(labels),
        "balanced_accuracy": sum(recalls) / len(recalls),
        "macro_f1": sum(f1_values) / 2.0,
        "auroc": _binary_auroc(labels, probabilities),
        "confusion_matrix": confusion,
        "per_class": per_class,
        "predictions": predictions,
        "pass_probabilities": probabilities,
        "insufficient_class_support": len(recalls) != 2,
    }
