"""从人工 F/E/A 标签构造确定性的 Pareto 偏好对。"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass

from training.grpo_v3.experiments.human_preference_reviewer.v1.data import (
    CandidateRecord,
    EvidenceRecord,
)

ScoreVector = tuple[int, int, int]
_VALID_GRADES = frozenset((1, 2, 3))


@dataclass(frozen=True)
class PreferencePair:
    evidence_id: str
    chosen: CandidateRecord
    rejected: CandidateRecord
    chosen_fingerprint: str
    rejected_fingerprint: str


@dataclass(frozen=True)
class PairAudit:
    total_combinations: int
    dominance_pair_count: int
    equal_vector_pair_count: int
    incomparable_pair_count: int
    duplicate_candidate_count: int


def _validate_vector(vector: object, *, name: str) -> ScoreVector:
    if not isinstance(vector, tuple) or len(vector) != 3:
        raise ValueError(f"{name} must be a three-grade score vector")
    if any(type(grade) is not int or grade not in _VALID_GRADES for grade in vector):
        raise ValueError(f"{name} grades must be integers 1, 2, or 3")
    return vector


def _score_vector(candidate: CandidateRecord) -> ScoreVector:
    return _validate_vector(
        (candidate.qa_formality, candidate.evidence_quality, candidate.answerability),
        name=f"candidate {candidate.candidate_id}",
    )


def dominates(left: ScoreVector, right: ScoreVector) -> bool:
    """当且仅当左侧每项不差且至少一项更高时返回真。"""
    valid_left = _validate_vector(left, name="left")
    valid_right = _validate_vector(right, name="right")
    return all(a >= b for a, b in zip(valid_left, valid_right)) and any(
        a > b for a, b in zip(valid_left, valid_right)
    )


def compact_fingerprint(evidence_id: str, candidate: CandidateRecord) -> str:
    """基于证据与模型可见内容生成稳定的 SHA-256 内容指纹。"""
    payload = {"evidence_id": evidence_id, **candidate.model_features()}
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_pareto_pairs(evidence: EvidenceRecord) -> tuple[tuple[PreferencePair, ...], PairAudit]:
    """在一个 evidence 内去重后，提取可比较的 Pareto 偏好对。"""
    by_fingerprint: dict[str, CandidateRecord] = {}
    duplicate_candidate_count = 0

    for candidate in sorted(evidence.candidates, key=lambda item: item.candidate_id):
        if candidate.evidence_id != evidence.evidence_id:
            raise ValueError(
                "candidate evidence_id does not match container evidence_id: "
                f"{candidate.candidate_id}"
            )
        fingerprint = compact_fingerprint(evidence.evidence_id, candidate)
        if fingerprint in by_fingerprint:
            duplicate_candidate_count += 1
        else:
            by_fingerprint[fingerprint] = candidate

    candidates = tuple(sorted(by_fingerprint.items(), key=lambda item: item[1].candidate_id))
    dominance_pair_count = 0
    equal_vector_pair_count = 0
    incomparable_pair_count = 0
    pairs: list[PreferencePair] = []

    for (left_fingerprint, left), (right_fingerprint, right) in itertools.combinations(candidates, 2):
        left_vector = _score_vector(left)
        right_vector = _score_vector(right)
        if left_vector == right_vector:
            equal_vector_pair_count += 1
        elif dominates(left_vector, right_vector):
            dominance_pair_count += 1
            pairs.append(
                PreferencePair(
                    evidence_id=evidence.evidence_id,
                    chosen=left,
                    rejected=right,
                    chosen_fingerprint=left_fingerprint,
                    rejected_fingerprint=right_fingerprint,
                )
            )
        elif dominates(right_vector, left_vector):
            dominance_pair_count += 1
            pairs.append(
                PreferencePair(
                    evidence_id=evidence.evidence_id,
                    chosen=right,
                    rejected=left,
                    chosen_fingerprint=right_fingerprint,
                    rejected_fingerprint=left_fingerprint,
                )
            )
        else:
            incomparable_pair_count += 1

    pairs.sort(key=lambda pair: (pair.chosen_fingerprint, pair.rejected_fingerprint))
    total_combinations = len(candidates) * (len(candidates) - 1) // 2
    audit = PairAudit(
        total_combinations=total_combinations,
        dominance_pair_count=dominance_pair_count,
        equal_vector_pair_count=equal_vector_pair_count,
        incomparable_pair_count=incomparable_pair_count,
        duplicate_candidate_count=duplicate_candidate_count,
    )
    return tuple(pairs), audit
