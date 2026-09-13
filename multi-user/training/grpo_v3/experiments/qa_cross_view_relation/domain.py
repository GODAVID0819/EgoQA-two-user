from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence


REWARD_REVISION = "qa_cross_view_relation_v2"
REWARD_COMPONENT = "qa_cross_view_relation"
ANCHOR_IDS = frozenset({"strong_anchor", "weak_anchor", "strong_cross_view_followup_v1", "weak_other_person_activity_v1"})
TEXT_CHECK_NAMES = frozenset(
    {
        "question_answer_type_match",
        "options_answer_same_question",
        "semantic_option_uniqueness",
        "answer_resolves_question",
        "premise_relevance",
        "text_claim_consistency",
        "natural_first_person_wording",
        "shallow_activity_relation",
    }
)
Score = Literal[0, 1, 2]
AnchorTier = Literal[0, 1, 2]
Preference = Literal["WIN", "TIE", "LOSS"]


def _score(value: Any, name: str) -> Score:
    if isinstance(value, bool) or not isinstance(value, int) or value not in {0, 1, 2}:
        raise ValueError(f"{name} must be one of 0, 1, 2")
    return value  # type: ignore[return-value]


def _preference(value: Any) -> Preference:
    if value not in {"WIN", "TIE", "LOSS"}:
        raise ValueError("pairwise preference must be WIN, TIE, or LOSS")
    return value  # type: ignore[return-value]


@dataclass(frozen=True)
class TextCheck:
    status: Literal["PASS", "FAIL"]
    reason: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], name: str) -> "TextCheck":
        status = str(value.get("status", "")).upper()
        if status not in {"PASS", "FAIL"}:
            raise ValueError(f"{name}.status must be PASS or FAIL")
        reason = str(value.get("reason", "")).strip()
        if not reason:
            raise ValueError(f"{name}.reason is required")
        return cls(status=status, reason=reason)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, str]:
        return {"status": self.status, "reason": self.reason}


@dataclass(frozen=True)
class CandidateSemanticScore:
    candidate_id: str
    cross_view_relation_score: Score
    semantic_naturalness_score: Score
    internal_consistency_score: Score
    anchor_tier: AnchorTier
    pairwise_preferences: dict[str, Preference]
    reasons: dict[str, str]
    checks: dict[str, TextCheck] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        require_text_checks: bool = False,
    ) -> "CandidateSemanticScore":
        candidate_id = str(value.get("candidate_id") or "")
        if not candidate_id:
            raise ValueError("candidate_id is required")
        if candidate_id in ANCHOR_IDS:
            raise ValueError("candidate_id cannot be an anchor id")
        raw_preferences = value.get("pairwise_preferences")
        if not isinstance(raw_preferences, Mapping):
            raise ValueError("pairwise_preferences must be an object")
        preferences: dict[str, Preference] = {}
        for key, item in raw_preferences.items():
            other = str(key)
            if other == candidate_id:
                raise ValueError("candidate cannot compare against itself")
            preferences[other] = _preference(item)
        raw_reasons = value.get("reasons") or {}
        if not isinstance(raw_reasons, Mapping):
            raise ValueError("reasons must be an object")
        raw_checks = value.get("checks") or {}
        if not isinstance(raw_checks, Mapping):
            raise ValueError("checks must be an object")
        check_names = {str(key) for key in raw_checks}
        missing = TEXT_CHECK_NAMES - check_names
        extra = check_names - TEXT_CHECK_NAMES
        if require_text_checks and missing:
            raise ValueError(f"missing text checks: {', '.join(sorted(missing))}")
        if extra:
            raise ValueError(f"unknown text checks: {', '.join(sorted(extra))}")
        for name, item in raw_checks.items():
            if not isinstance(item, Mapping):
                raise ValueError(f"{name} must be an object")
        return cls(
            candidate_id=candidate_id,
            cross_view_relation_score=_score(value.get("cross_view_relation_score"), "cross_view_relation_score"),
            semantic_naturalness_score=_score(value.get("semantic_naturalness_score"), "semantic_naturalness_score"),
            internal_consistency_score=_score(value.get("internal_consistency_score"), "internal_consistency_score"),
            anchor_tier=_score(value.get("anchor_tier"), "anchor_tier"),
            pairwise_preferences=preferences,
            reasons={str(key): str(item) for key, item in raw_reasons.items()},
            checks={
                str(key): TextCheck.from_mapping(item, str(key))
                for key, item in raw_checks.items()
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "cross_view_relation_score": self.cross_view_relation_score,
            "semantic_naturalness_score": self.semantic_naturalness_score,
            "internal_consistency_score": self.internal_consistency_score,
            "anchor_tier": self.anchor_tier,
            "pairwise_preferences": dict(self.pairwise_preferences),
            "reasons": dict(self.reasons),
            "checks": {name: check.to_dict() for name, check in self.checks.items()},
        }


@dataclass(frozen=True)
class GroupJudgeResult:
    candidate_scores: dict[str, CandidateSemanticScore]
    raw_outputs: tuple[dict[str, Any], ...] = ()
    item_orders: tuple[tuple[str, ...], ...] = ()
    order_instability: bool = False

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        expected_candidate_ids: Sequence[str],
        *,
        raw_outputs: Sequence[Mapping[str, Any]] = (),
        item_orders: Sequence[Sequence[str]] = (),
        order_instability: bool = False,
        require_text_checks: bool = False,
    ) -> "GroupJudgeResult":
        expected = [str(item) for item in expected_candidate_ids]
        if any(item in ANCHOR_IDS for item in expected):
            raise ValueError("policy candidate ids cannot include anchors")
        raw_scores = value.get("candidate_scores")
        if not isinstance(raw_scores, Sequence) or isinstance(raw_scores, (str, bytes)):
            raise ValueError("candidate_scores must be a sequence")
        if any(not isinstance(item, Mapping) for item in raw_scores):
            raise ValueError("candidate score must be an object")
        scores = [
            CandidateSemanticScore.from_mapping(
                item,
                require_text_checks=require_text_checks,
            )
            for item in raw_scores
        ]
        actual = [item.candidate_id for item in scores]
        if sorted(actual) != sorted(expected) or len(actual) != len(set(actual)):
            raise ValueError("judge result must contain every expected candidate exactly once")
        expected_set = set(expected)
        for score in scores:
            required_opponents = expected_set - {score.candidate_id}
            actual_opponents = set(score.pairwise_preferences)
            if actual_opponents != required_opponents:
                raise ValueError(
                    f"pairwise preference keys for {score.candidate_id} must be exactly "
                    f"{sorted(required_opponents)}"
                )
        return cls(
            candidate_scores={item.candidate_id: item for item in scores},
            raw_outputs=tuple(dict(item) for item in raw_outputs),
            item_orders=tuple(tuple(str(candidate) for candidate in order) for order in item_orders),
            order_instability=bool(order_instability),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_scores": [score.to_dict() for score in self.candidate_scores.values()],
            "raw_outputs": list(self.raw_outputs),
            "item_orders": [list(order) for order in self.item_orders],
            "order_instability": self.order_instability,
        }


@dataclass(frozen=True)
class JudgeCandidate:
    candidate_id: str
    raw_completion: str
    qa: dict[str, Any]
    deterministic_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class Anchor:
    anchor_id: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class AnchorSet:
    strong: Anchor
    weak: Anchor
    sha256: str
