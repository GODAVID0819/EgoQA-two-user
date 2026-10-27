"""显式选择二分类奖励合成方式，不将训练 loss 权重隐式变成实验配置。"""
from __future__ import annotations
import math

KEYS = ("formality", "groundedness", "combined_answerability")
MODES = ("continuous", "binary", "formality", "groundedness")


def validate_probabilities(values: dict) -> dict[str, float]:
    if set(values) != set(KEYS):
        raise ValueError("需要新F/E/联合C三项Judge分数")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in values.values()):
        raise ValueError("Judge 概率必须是 [0,1] 内的有限数值")
    return {k: float(values[k]) for k in KEYS}


def aggregate(probabilities: dict, *, mode: str) -> float:
    p = validate_probabilities(probabilities)
    if mode not in MODES:
        raise ValueError(f"必须显式选择 reward mode：{MODES}")
    if mode in {"formality", "groundedness"}:
        return p[mode]
    if mode == "binary":
        p = {key: float(value >= .5) for key, value in p.items()}
    return .2 * p["formality"] + .4 * p["groundedness"] + .4 * p["combined_answerability"]
