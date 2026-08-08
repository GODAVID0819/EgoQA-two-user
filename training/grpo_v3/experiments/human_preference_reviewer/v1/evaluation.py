"""Fixed-label three-class metrics for Reviewer v1."""

from __future__ import annotations

import math
from typing import Any, Sequence


def _rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            ranks[order[position]] = average
        cursor = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    x = _rank(left)
    y = _rank(right)
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    x_var = sum((a - x_mean) ** 2 for a in x)
    y_var = sum((b - y_mean) ** 2 for b in y)
    if x_var == 0 or y_var == 0:
        return None
    return numerator / math.sqrt(x_var * y_var)


def classification_metrics(
    labels: Sequence[int],
    probabilities: Sequence[Sequence[float]],
    *,
    loss: float | None = None,
) -> dict[str, Any]:
    if len(labels) != len(probabilities) or not labels:
        raise ValueError("labels and probabilities must have the same non-zero length")
    normalized: list[list[float]] = []
    for row in probabilities:
        if len(row) != 3 or any(not math.isfinite(float(value)) or float(value) < 0 for value in row):
            raise ValueError("each probability row must contain three finite nonnegative values")
        total = sum(float(value) for value in row)
        if total <= 0:
            raise ValueError("probability row must have positive mass")
        normalized.append([float(value) / total for value in row])
    if any(type(label) is not int or label not in {1, 2, 3} for label in labels):
        raise ValueError("labels must be integer grades 1, 2, or 3")
    predictions = [max(range(3), key=lambda index: row[index]) + 1 for row in normalized]
    confusion = [[0, 0, 0] for _ in range(3)]
    for label, prediction in zip(labels, predictions):
        confusion[label - 1][prediction - 1] += 1
    per_level: dict[str, dict[str, float | int]] = {}
    f1_values = []
    for level in (1, 2, 3):
        true_positive = confusion[level - 1][level - 1]
        support = sum(confusion[level - 1])
        predicted = sum(row[level - 1] for row in confusion)
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_level[str(level)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "predicted_count": predicted,
        }
        f1_values.append(f1)
    expected_scores = [sum((index + 1) * value for index, value in enumerate(row)) for row in normalized]
    return {
        "count": len(labels),
        "loss": loss,
        "accuracy": sum(a == b for a, b in zip(labels, predictions)) / len(labels),
        "macro_f1": sum(f1_values) / 3.0,
        "confusion_matrix": confusion,
        "per_level": per_level,
        "predictions": predictions,
        "expected_scores": expected_scores,
        "expected_score_mae": sum(abs(a - b) for a, b in zip(expected_scores, labels)) / len(labels),
        "spearman": _spearman([float(value) for value in labels], expected_scores),
        "insufficient_class_support": any(per_level[str(level)]["support"] == 0 for level in (1, 2, 3)),
    }
