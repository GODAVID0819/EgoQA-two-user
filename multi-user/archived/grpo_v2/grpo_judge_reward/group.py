from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from .records import JudgeGroupRecord


def compute_group_records(records: list[dict[str, Any]], *, epsilon: float = 1e-8) -> list[JudgeGroupRecord]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row.get("group_id") or "")].append(row)

    results: list[JudgeGroupRecord] = []
    for group_id, rows in sorted(grouped.items()):
        valid = [
            row for row in rows
            if row.get("masked") is not True and isinstance(row.get("reward_total"), (int, float))
        ]
        masked_count = sum(row.get("masked") is True for row in rows)
        mean = None
        std = None
        advantages: dict[str, float] = {}
        skip_reason = None
        if len(valid) < 2:
            skip_reason = "too_few_valid_candidates"
        else:
            rewards = [float(row["reward_total"]) for row in valid]
            mean = sum(rewards) / len(rewards)
            std = math.sqrt(sum((value - mean) ** 2 for value in rewards) / len(rewards))
            if std < epsilon:
                skip_reason = "zero_reward_variance"
            else:
                advantages = {
                    str(row.get("candidate_id")): round((float(row["reward_total"]) - mean) / (std + epsilon), 6)
                    for row in valid
                }
        results.append(
            JudgeGroupRecord(
                group_id=group_id,
                raw_candidate_count=len(rows),
                masked_candidate_count=masked_count,
                valid_candidate_count=len(valid),
                reward_mean=round(mean, 6) if mean is not None else None,
                reward_std=round(std, 6) if std is not None else None,
                trainer_eligible=skip_reason is None,
                skip_reason=skip_reason,
                advantages=advantages,
            )
        )
    return results

