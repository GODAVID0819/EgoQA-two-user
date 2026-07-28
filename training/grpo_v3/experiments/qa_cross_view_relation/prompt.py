from __future__ import annotations

import hashlib
import json
from typing import Sequence

from .domain import AnchorSet, JudgeCandidate


def stable_candidate_order(candidate_ids: Sequence[str], order_seed: str) -> tuple[str, ...]:
    return tuple(sorted(candidate_ids, key=lambda item: (hashlib.sha256(f"{order_seed}\0{item}".encode()).hexdigest(), item)))


def build_group_judge_prompt(
    *,
    candidates: Sequence[JudgeCandidate],
    anchors: AnchorSet,
    order_seed: str,
    reverse: bool = False,
) -> tuple[str, tuple[str, ...]]:
    by_id = {item.candidate_id: item for item in candidates}
    order = stable_candidate_order(list(by_id), order_seed)
    if reverse:
        order = tuple(reversed(order))
    payload = {
        "task": "Score QA candidates for cross-view relation quality. Return JSON only.",
        "scores": {
            "cross_view_relation_score": "0 no concrete relation, 1 vague relation, 2 concrete missing object/state/location/outcome relation",
            "semantic_naturalness_score": "0 unnatural, impossible, wrong subject-verb/object pairing, or malformed question; 1 understandable but stiff/vague/awkward; 2 natural human question",
            "internal_consistency_score": "0 QA/options/answer contradiction, answer does not exactly match the correct option text, duplicate/conflicting options, or impossible role logic; 1 minor issue; 2 fully consistent",
            "anchor_tier": "0 not better than weak anchor, 1 between anchors, 2 at or above strong anchor",
            "pairwise_preferences": "For every other policy candidate: WIN, TIE, or LOSS",
        },
        "audit_guidelines": {
            "semantic_naturalness": [
                "Penalize impossible subject-verb or object-action pairings, such as objects doing human actions.",
                "Penalize vague placeholder wording like option A/option B when it makes the question unnatural.",
                "Penalize questions that sound like dataset annotations rather than a first-person human question.",
            ],
            "internal_consistency": [
                "Check that answer exactly matches the option selected by correct, not only the letter.",
                "Penalize duplicate options, contradictory options, or an answer that is not one of the options.",
                "Check that question, options, answer, evidence, and why_two_users_needed describe the same fact.",
            ],
            "privacy_and_dataset_language": [
                "Penalize questions that expose required user names.",
                "Penalize questions that mention timestamps, frame numbers, camera, clip, or dataset language.",
                "If deterministic_flags already report a repaired format, do not penalize merely for JSON repair.",
            ],
        },
        "strong_anchor": anchors.strong.payload,
        "weak_anchor": anchors.weak.payload,
        "candidates": [
            {
                "candidate_id": candidate_id,
                "qa": by_id[candidate_id].qa,
                "deterministic_flags": list(by_id[candidate_id].deterministic_flags),
            }
            for candidate_id in order
        ],
        "output_shape": {
            "candidate_scores": [
                {
                    "candidate_id": "candidate id",
                    "cross_view_relation_score": 0,
                    "semantic_naturalness_score": 0,
                    "internal_consistency_score": 0,
                    "anchor_tier": 0,
                    "pairwise_preferences": {"other candidate id": "WIN/TIE/LOSS"},
                    "reasons": {"summary": "short reason"},
                }
            ]
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2), order
