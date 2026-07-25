"""ms-swift 4.2.2 的 text-only A-density ORM。"""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from collections.abc import Sequence
from typing import Any

from training.grpo_v3.experiments.text_only_a_density.domain import (
    REWARD_KIND,
    REWARD_REVISION,
    score_completion,
)
from training.grpo_v3.runtime.reward_plugin import ORM, _trace_path, _write_rows, orms


def _strict_list(value: Any, count: int, *, name: str) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        values = [value]
    else:
        values = list(value)
    if len(values) == 1:
        values *= count
    if len(values) != count:
        raise ValueError(f"{name} 数量为 {len(values)}，要求与 4 个 completion 对齐")
    return values


class TextOnlyADensityReward(ORM):
    def __init__(
        self,
        args: Any = None,
        *,
        trace_path: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(args, **kwargs)
        self.trace_path = _trace_path(trace_path)
        self._lock = threading.Lock()
        self._reward_call_index = 0

    def __call__(self, completions: Sequence[str], **kwargs: Any) -> list[float]:
        if len(completions) != 4:
            raise ValueError(f"text-only A-density 固定要求 4 个 completions，实际为 {len(completions)}")
        trial_ids = _strict_list(kwargs.get("trial_id"), 4, name="trial_id")
        phases = _strict_list(kwargs.get("phase", "train"), 4, name="phase")
        candidates = _strict_list(kwargs.get("candidate_index", [0, 1, 2, 3]), 4, name="candidate_index")
        if len({str(item) for item in trial_ids}) != 1:
            raise ValueError("同一四候选 group 的 trial_id 必须一致")
        if len({str(item) for item in phases}) != 1:
            raise ValueError("同一四候选 group 的 phase 必须一致")
        if [int(item) for item in candidates] != [0, 1, 2, 3]:
            raise ValueError("candidate_index 必须严格为 0,1,2,3")
        call_index = self._reward_call_index
        self._reward_call_index += 1
        rows = []
        rewards = []
        for index, raw in enumerate(completions):
            completion = str(raw)
            score = score_completion(completion)
            rewards.append(score.reward)
            rows.append(
                {
                    "reward_kind": REWARD_KIND,
                    "reward_revision": REWARD_REVISION,
                    "phase": str(phases[index]),
                    "reward_call_index": call_index,
                    "candidate_index": int(candidates[index]),
                    "trial_id": str(trial_ids[index]),
                    "completion": completion,
                    "completion_sha256": hashlib.sha256(completion.encode("utf-8")).hexdigest(),
                    **score.to_dict(),
                    "formal_result": False,
                }
            )
        _write_rows(self.trace_path, rows, self._lock)
        return rewards


if "egoqa_text_only_a_density" in orms and orms["egoqa_text_only_a_density"] is not TextOnlyADensityReward:
    raise RuntimeError("egoqa_text_only_a_density ORM 名称已被其他实现占用")
orms["egoqa_text_only_a_density"] = TextOnlyADensityReward
