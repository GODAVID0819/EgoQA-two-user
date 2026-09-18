"""Small dependency-free metrics for verdict-token evaluation."""

from __future__ import annotations

from typing import Any

from .contracts import JudgeTask, TASK_IDS


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _binary_metrics(targets: list[int], predictions: list[int]) -> dict[str, float]:
    tp = sum(target == 1 and pred == 1 for target, pred in zip(targets, predictions))
    tn = sum(target == 0 and pred == 0 for target, pred in zip(targets, predictions))
    fp = sum(target == 0 and pred == 1 for target, pred in zip(targets, predictions))
    fn = sum(target == 1 and pred == 0 for target, pred in zip(targets, predictions))
    pass_recall = _safe_ratio(tp, tp + fn)
    fail_recall = _safe_ratio(tn, tn + fp)
    precision = _safe_ratio(tp, tp + fp)
    recall = pass_recall
    f1 = _safe_ratio(2 * precision * recall, precision + recall)
    return {
        "accuracy": _safe_ratio(tp + tn, tp + tn + fp + fn),
        "balanced_accuracy": (pass_recall + fail_recall) / 2.0,
        "pass_recall": pass_recall,
        "fail_recall": fail_recall,
        "pass_f1": f1,
    }


def compute_verdict_metrics(eval_prediction: Any) -> dict[str, float]:
    predictions = eval_prediction.predictions
    label_ids = eval_prediction.label_ids
    predicted = (predictions[:, 1] > predictions[:, 0]).astype("int64")
    targets = label_ids[:, 0].astype("int64")
    task_ids = label_ids[:, 1].astype("int64")
    result = _binary_metrics(targets.tolist(), predicted.tolist())
    task_balanced: list[float] = []
    for task in JudgeTask:
        mask = task_ids == TASK_IDS[task]
        if not bool(mask.any()):
            continue
        metrics = _binary_metrics(targets[mask].tolist(), predicted[mask].tolist())
        for name, value in metrics.items():
            result[f"{task.value}_{name}"] = value
        task_balanced.append(metrics["balanced_accuracy"])
    result["macro_balanced_accuracy"] = (
        sum(task_balanced) / len(task_balanced) if task_balanced else 0.0
    )
    return result
