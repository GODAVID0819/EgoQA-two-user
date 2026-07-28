from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .deterministic import DeterministicAssessment
from .domain import GroupJudgeResult, REWARD_COMPONENT, REWARD_REVISION


CONSISTENCY_BLOCKERS = frozenset(
    {
        "question_answer_type_match",
        "options_answer_same_question",
        "semantic_option_uniqueness",
        "answer_resolves_question",
        "premise_relevance",
        "text_claim_consistency",
    }
)


@dataclass(frozen=True)
class RewardResult:
    candidate_id: str
    semantic_quality: float
    anchor_score: float
    borda_score: float
    reward_before_cap: float
    reward_cap: float
    cap_reasons: tuple[str, ...]
    reward_total: float
    reward_source: str
    format_status: str
    reward_revision: str = REWARD_REVISION

    def to_record(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "semantic_quality": self.semantic_quality,
            "anchor_score": self.anchor_score,
            "borda_score": self.borda_score,
            "reward_before_cap": self.reward_before_cap,
            "reward_cap": self.reward_cap,
            "cap_reasons": list(self.cap_reasons),
            "reward_total": self.reward_total,
            "reward_components": {REWARD_COMPONENT: self.reward_total},
            "reward_source": self.reward_source,
            "format_status": self.format_status,
            "reward_revision": self.reward_revision,
        }


def _clamp_reward(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("reward must be finite")
    return max(0.0, min(1.0, value))


def _semantic_quality(relation: int, naturalness: int, consistency: int) -> float:
    return 0.6 * (relation / 2.0) + 0.2 * (naturalness / 2.0) + 0.2 * (consistency / 2.0)


def _borda(candidate_id: str, preferences: Mapping[str, str], valid_ids: Sequence[str]) -> float:
    opponents = [item for item in valid_ids if item != candidate_id]
    if not opponents:
        return 0.5
    wins = 0.0
    for opponent in opponents:
        value = preferences.get(opponent)
        if value == "WIN":
            wins += 1.0
        elif value == "TIE":
            wins += 0.5
        elif value == "LOSS":
            wins += 0.0
        else:
            raise ValueError(f"missing pairwise preference for {candidate_id} vs {opponent}")
    return wins / len(opponents)


def _has_unrecoverable_json(assessment: DeterministicAssessment) -> bool:
    return "unrecoverable_json" in assessment.blocking_errors


def _deterministic_blocking_penalty(assessment: DeterministicAssessment) -> float:
    errors = set(assessment.blocking_errors)
    if not errors:
        return 0.0
    if "question_uses_dataset_language" in errors:
        return 0.2
    return 0.0


def _zero_group_result(
    candidate_id: str,
    assessment: DeterministicAssessment,
    *,
    reward_revision: str,
) -> RewardResult:
    return RewardResult(
        candidate_id=candidate_id,
        semantic_quality=0.0,
        anchor_score=0.0,
        borda_score=0.0,
        reward_before_cap=0.0,
        reward_cap=1.0,
        cap_reasons=(),
        reward_total=0.0,
        reward_source="group_skipped_unrecoverable_json",
        format_status=assessment.format_status,
        reward_revision=reward_revision,
    )


def compute_group_rewards(
    *,
    candidate_ids: Sequence[str],
    deterministic_results: Mapping[str, DeterministicAssessment],
    judge_result: GroupJudgeResult | None,
    invalid_cost: float = 0.05,
    apply_text_caps: bool = False,
    reward_revision: str = REWARD_REVISION,
) -> dict[str, RewardResult]:
    ids = [str(item) for item in candidate_ids]
    if any(_has_unrecoverable_json(deterministic_results[item]) for item in ids):
        return {
            item: _zero_group_result(
                item,
                deterministic_results[item],
                reward_revision=reward_revision,
            )
            for item in ids
        }

    valid_ids = [
        item for item in ids
        if deterministic_results[item].eligible_for_semantic_judge
    ]
    if valid_ids and judge_result is None:
        raise ValueError("judge_result is required when the group has valid candidates")
    if judge_result is not None and sorted(judge_result.candidate_scores) != sorted(valid_ids):
        raise ValueError("judge_result must match valid candidates")

    results: dict[str, RewardResult] = {}
    valid_rewards: list[float] = []
    for candidate_id in valid_ids:
        assert judge_result is not None
        score = judge_result.candidate_scores[candidate_id]
        relation_score = score.cross_view_relation_score
        naturalness_score = score.semantic_naturalness_score
        consistency_score = score.internal_consistency_score
        anchor_tier = score.anchor_tier
        failed = (
            {name for name, check in score.checks.items() if check.status == "FAIL"}
            if apply_text_caps
            else set()
        )
        consistency_failures = sorted(failed & CONSISTENCY_BLOCKERS)
        if consistency_failures:
            consistency_score = 0
            anchor_tier = min(anchor_tier, 1)
        if "shallow_activity_relation" in failed:
            relation_score = 0
            anchor_tier = 0
        if "natural_first_person_wording" in failed:
            naturalness_score = 0
        semantic = _semantic_quality(
            relation_score,
            naturalness_score,
            consistency_score,
        )
        anchor = anchor_tier / 2.0
        borda = _borda(candidate_id, score.pairwise_preferences, valid_ids)
        reward_before_cap = _clamp_reward(0.60 * semantic + 0.25 * anchor + 0.15 * borda)
        reward_cap = 1.0
        cap_reasons: list[str] = []
        if apply_text_caps:
            if consistency_failures:
                reward_cap = min(reward_cap, 0.40)
                cap_reasons.extend(consistency_failures)
            if "shallow_activity_relation" in failed:
                reward_cap = min(reward_cap, 0.40)
                cap_reasons.append("shallow_activity_relation")
            if "natural_first_person_wording" in failed:
                reward_cap = min(reward_cap, 0.55)
                cap_reasons.append("natural_first_person_wording")
        reward = min(reward_before_cap, reward_cap)
        valid_rewards.append(reward)
        results[candidate_id] = RewardResult(
            candidate_id=candidate_id,
            semantic_quality=semantic,
            anchor_score=anchor,
            borda_score=borda,
            reward_before_cap=reward_before_cap,
            reward_cap=reward_cap,
            cap_reasons=tuple(cap_reasons),
            reward_total=reward,
            reward_source=reward_revision,
            format_status=deterministic_results[candidate_id].format_status,
            reward_revision=reward_revision,
        )

    for candidate_id in ids:
        if candidate_id in results:
            continue
        assessment = deterministic_results[candidate_id]
        penalty = _deterministic_blocking_penalty(assessment)
        results[candidate_id] = RewardResult(
            candidate_id=candidate_id,
            semantic_quality=0.0,
            anchor_score=0.0,
            borda_score=0.0,
            reward_before_cap=penalty,
            reward_cap=1.0,
            cap_reasons=(),
            reward_total=penalty,
            reward_source="deterministic_blocking_penalty" if assessment.blocking_errors else "all_invalid_zero",
            format_status=assessment.format_status,
            reward_revision=reward_revision,
        )
    return results
