"""ms-swift ORM for the frozen score-only ordinal reviewer."""

from __future__ import annotations

import json
import math
import os
import threading
from pathlib import Path
from typing import Any, Sequence

try:
    from egolife_two_user_qa.training.grpo_v3.experiments.human_preference_reviewer.v1.grpo_reward import (
        REWARD_REVISION,
        score_completion,
    )
    from egolife_two_user_qa.training.grpo_v3.experiments.human_preference_reviewer.v1.service import (
        ScoreOnlyReviewerClient,
    )
except ModuleNotFoundError as error:
    if error.name != "egolife_two_user_qa":
        raise
    # Supports deliberate execution from the package checkout itself.
    from training.grpo_v3.experiments.human_preference_reviewer.v1.grpo_reward import (
        REWARD_REVISION,
        score_completion,
    )
    from training.grpo_v3.experiments.human_preference_reviewer.v1.service import (
        ScoreOnlyReviewerClient,
    )

try:
    from swift.rewards import ORM, orms
except ImportError:
    class ORM:  # type: ignore[no-redef]
        def __init__(self, args: Any = None, **kwargs: Any) -> None:
            self.args = args

    orms: dict[str, type] = {}


def _expand(value: Any, count: int, name: str) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        rows = list(value)
    else:
        rows = [value]
    if len(rows) == count:
        return rows
    if len(rows) == 1:
        return rows * count
    raise ValueError(f"{name} count {len(rows)} cannot expand to {count}")


def _expand_path_list(value: Any, count: int, name: str) -> list[Any]:
    # A single row's media value is itself a variable-length path sequence.
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and all(isinstance(item, (str, Path)) for item in value)
    ):
        return [list(value) for _ in range(count)]
    return _expand(value, count, name)


class ScoreOnlyOrdinalReward(ORM):
    def __init__(
        self,
        args: Any = None,
        *,
        client: Any | None = None,
        trace_path: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(args, **kwargs)
        self.client = client or ScoreOnlyReviewerClient(
            os.environ.get("EGOQA_SCORE_ONLY_REVIEWER_BASE_URL", "http://127.0.0.1:8766"),
            timeout_seconds=float(os.environ.get("EGOQA_SCORE_ONLY_REVIEWER_TIMEOUT_SECONDS", "300")),
        )
        self.trace_path = Path(
            trace_path or os.environ.get("EGOQA_GRPO_V3_REWARD_TRACE", "reward_trace.jsonl")
        )
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._call_index = 0

    def __call__(self, completions: Sequence[str], **kwargs: Any) -> list[float]:
        count = len(completions)
        packets = _expand(kwargs["packet_json"], count, "packet_json")
        evidence_ids = _expand(kwargs["evidence_id"], count, "evidence_id")
        generator_images = _expand_path_list(
            kwargs["generator_image_paths"], count, "generator_image_paths"
        )
        reviewer_videos = _expand_path_list(
            kwargs["reviewer_video_paths"], count, "reviewer_video_paths"
        )
        with self._lock:
            call_index = self._call_index
            self._call_index += 1
        rewards = []
        rows = []
        for index, completion in enumerate(completions):
            packet = json.loads(packets[index]) if isinstance(packets[index], str) else packets[index]
            result = score_completion(
                str(completion), packet,
                evidence_id=str(evidence_ids[index]),
                candidate_index=index,
                client=self.client,
                dataset_generator_images=generator_images[index],
                dataset_reviewer_videos=reviewer_videos[index],
            )
            reward = float(result["reward"])
            if not math.isfinite(reward):
                raise RuntimeError("score-only ORM received a non-finite reward")
            rewards.append(reward)
            rows.append({
                "reward_kind": "score_only_ordinal_reviewer",
                "reward_revision": REWARD_REVISION,
                "reward_call_index": call_index,
                "evidence_id": str(evidence_ids[index]),
                "candidate_index": index,
                "reward": reward,
                "record": result["record"],
            })
        with self._lock, self.trace_path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        return rewards


orms["egoqa_score_only_ordinal_v1"] = ScoreOnlyOrdinalReward
