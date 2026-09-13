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
    attempt_id: str
    parse_success: bool
    schema_pass: bool | None
    schema_error_count: int | None
    formality_pass: bool | None
    groundedness_pass: bool | None
    combined_correct: bool | None
    speaker_alone_correct: bool | None
    provider_alone_correct: bool | None
    parse_reward: float
    schema_reward: float | None
    formality_reward: float | None
    groundedness_reward: float | None
    combined_reward: float | None
    speaker_leakage_reward: float | None
    provider_alone_reward: float | None
    total: float
    reward_version: str
    missing_components: tuple[str, ...]
    is_complete_reward: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
