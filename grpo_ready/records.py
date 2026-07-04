"""GRPO-ready 离线分析使用的稳定记录类型。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class AttemptRecord:
    attempt_id: str
    evidence_id: str
    packet_status: str
    question_type: str | None
    mode: str | None
    attempt_index: int
    feedback: str
    generator_prompt: str
    generator_image_paths: tuple[str, ...]
    generator_video_paths: tuple[str, ...]
    evaluator_image_paths: tuple[str, ...]
    evaluator_video_paths: tuple[str, ...]
    raw_qa: str
    parsed_qa: dict[str, Any] | None
    schema_errors: tuple[str, ...]
    judge: dict[str, Any] | None
    answerability: dict[str, Any] | None
    accepted: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RewardRecord:
    """Reward v0 记录；具体字段在奖励任务中补齐。"""

    attempt_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
