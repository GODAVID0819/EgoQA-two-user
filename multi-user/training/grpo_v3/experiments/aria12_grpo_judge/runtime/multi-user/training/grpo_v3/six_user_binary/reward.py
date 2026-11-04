"""显式选择二分类奖励合成方式，不将训练 loss 权重隐式变成实验配置。"""
from __future__ import annotations
import math

KEYS = ("formality", "groundedness", "combined_answerability")
MODES = ("continuous", "continuous_gated", "binary", "formality", "groundedness")
# 训练奖励可选的质量项：continuous = 0.2F+0.4E+0.4C；continuous_gated = 0.2F+0.4E+0.4C·[E>0.5]
QUALITY_MODES = ("continuous", "continuous_gated")


def validate_probabilities(values):
    if set(values) != set(KEYS):
        raise ValueError("候选必须保留三个评分字段")
    result = dict(values)
    for name in KEYS:
        v = result[name]
        if name == "combined_answerability" and result["groundedness"] <= .5:
            if v is not None:
                raise ValueError("跳过的 C 必须记录为 null")
            continue
        if type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1:
            raise ValueError(f"{name} 缺失或越界")
    return result


def aggregate(probabilities, *, mode):
    if mode != "continuous_gated":
        raise ValueError("此运行只支持配置 A")
    p = validate_probabilities(probabilities)
    return .2*p["formality"] + .4*p["groundedness"] + .4*(p["combined_answerability"] or 0.)
