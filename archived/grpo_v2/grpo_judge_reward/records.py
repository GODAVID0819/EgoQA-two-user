from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class JudgeRewardRecord:
    candidate_id: str
    group_id: str
    evidence_id: str
    qa_id: str
    attempt: int | None
    masked: bool
    mask_reason: str | None
    eligible_for_grpo: bool
    reward_total: float | None
    reward_components: dict[str, float] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    schema_pass: bool | None = None
    groundedness_status: str | None = None
    combined_choice: str | None = None
    combined_correct: bool | None = None
    combined_insufficient: bool | None = None
    speaker_user: str | None = None
    speaker_only_choice: str | None = None
    speaker_only_correct: bool | None = None
    proper_subset_correct: bool = False
    provider_only_correct: bool = False
    qa_formality_status: str | None = None
    shallow_activity_status: str | None = None
    review_passed: bool | None = None
    rejection_stage: str | None = None
    feedback_to_generator: str = ""
    raw_qa: str = ""
    review: dict[str, Any] = field(default_factory=dict)
    answerability: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JudgeGroupRecord:
    group_id: str
    raw_candidate_count: int
    masked_candidate_count: int
    valid_candidate_count: int
    reward_mean: float | None
    reward_std: float | None
    trainer_eligible: bool
    skip_reason: str | None
    advantages: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

