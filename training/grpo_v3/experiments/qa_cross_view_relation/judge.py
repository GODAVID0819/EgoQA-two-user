from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from .anchors import load_anchor_set
from .deterministic import DeterministicAssessment
from .domain import AnchorSet, GroupJudgeResult, JudgeCandidate
from .prompt import build_group_judge_prompt


def _call_runner(runner: Any, prompt: str) -> str:
    if callable(runner):
        return str(runner(prompt))
    if hasattr(runner, "generate"):
        return str(runner.generate(prompt))
    if hasattr(runner, "complete"):
        return str(runner.complete(prompt))
    raise TypeError("runner must be callable or expose generate/complete")


def _parse_json_object(raw: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("judge output must be a JSON object")
    return value


def _build_schema_repair_prompt(
    *,
    original_prompt: str,
    raw_output: str,
    expected_candidate_ids: Sequence[str],
    error: BaseException,
) -> str:
    payload = {
        "task": (
            "Your previous answer failed JSON/schema validation. Re-run the original "
            "judge request and return one valid JSON object only."
        ),
        "validation_error": f"{type(error).__name__}: {error}",
        "required_candidate_ids": list(expected_candidate_ids),
        "rules": [
            "candidate_scores must contain exactly one item for each required candidate_id.",
            "Each candidate_id must be copied exactly from required_candidate_ids.",
            "Do not use anchor ids as candidate ids.",
            "Return JSON only, with no markdown fences or explanation.",
        ],
        "required_output_shape": {
            "candidate_scores": [
                {
                    "candidate_id": "candidate_0",
                    "cross_view_relation_score": 0,
                    "semantic_naturalness_score": 0,
                    "internal_consistency_score": 0,
                    "anchor_tier": 0,
                    "pairwise_preferences": {},
                    "reasons": {"summary": ""},
                }
            ]
        },
        "original_judge_request": original_prompt,
        "previous_invalid_response": raw_output,
    }
    return json.dumps(payload, ensure_ascii=True, indent=2)


def _parse_validate_or_repair(
    *,
    runner: Any,
    prompt: str,
    raw: str,
    expected_candidate_ids: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        parsed = _parse_json_object(raw)
        GroupJudgeResult.from_mapping(parsed, expected_candidate_ids)
        return parsed, {"prompt": prompt, "raw_output": raw, "parsed_output": parsed}
    except (TypeError, ValueError) as initial_error:
        repair_prompt = _build_schema_repair_prompt(
            original_prompt=prompt,
            raw_output=raw,
            expected_candidate_ids=expected_candidate_ids,
            error=initial_error,
        )
        repair_raw = _call_runner(runner, repair_prompt)
        repair_parsed = _parse_json_object(repair_raw)
        try:
            GroupJudgeResult.from_mapping(repair_parsed, expected_candidate_ids)
        except (TypeError, ValueError) as repair_error:
            raise ValueError(
                "judge output failed schema validation after one repair attempt: "
                f"initial={type(initial_error).__name__}: {initial_error}; "
                f"repair={type(repair_error).__name__}: {repair_error}"
            ) from repair_error
        return repair_parsed, {
            "prompt": prompt,
            "raw_output": raw,
            "parsed_output": repair_parsed,
            "initial_validation_error": {
                "type": type(initial_error).__name__,
                "message": str(initial_error),
            },
            "repair_prompt": repair_prompt,
            "repair_raw_output": repair_raw,
            "repair_parsed_output": repair_parsed,
        }


def _stabilize_unstable_result(
    *,
    first_result: GroupJudgeResult,
    second_result: GroupJudgeResult,
    expected_candidate_ids: Sequence[str],
) -> dict[str, Any]:
    scores: list[dict[str, Any]] = []
    for candidate_id in expected_candidate_ids:
        first_score = first_result.candidate_scores[candidate_id]
        second_score = second_result.candidate_scores[candidate_id]
        reasons = dict(first_score.reasons)
        reasons["order_stability"] = (
            "Two judge passes disagreed after reversing candidate order; scalar scores "
            "were conservatively minimized and pairwise preferences were neutralized."
        )
        scores.append(
            {
                "candidate_id": candidate_id,
                "cross_view_relation_score": min(
                    first_score.cross_view_relation_score,
                    second_score.cross_view_relation_score,
                ),
                "semantic_naturalness_score": min(
                    first_score.semantic_naturalness_score,
                    second_score.semantic_naturalness_score,
                ),
                "internal_consistency_score": min(
                    first_score.internal_consistency_score,
                    second_score.internal_consistency_score,
                ),
                "anchor_tier": min(first_score.anchor_tier, second_score.anchor_tier),
                "pairwise_preferences": {
                    other_id: "TIE"
                    for other_id in expected_candidate_ids
                    if other_id != candidate_id
                },
                "reasons": reasons,
            }
        )
    return {"candidate_scores": scores}


def judge_candidate_group(
    *,
    candidates: Sequence[JudgeCandidate],
    deterministic_results: Mapping[str, DeterministicAssessment],
    anchors: AnchorSet | None = None,
    runner: Any,
    order_seed: str,
) -> GroupJudgeResult:
    valid = [
        item for item in candidates
        if deterministic_results[item.candidate_id].eligible_for_semantic_judge
    ]
    if not valid:
        raise ValueError("no valid candidates to judge")
    anchor_set = anchors or load_anchor_set()
    expected_candidate_ids = [item.candidate_id for item in valid]
    outputs: list[dict[str, Any]] = []
    orders: list[tuple[str, ...]] = []
    for reverse in (False, True):
        prompt, order = build_group_judge_prompt(
            candidates=valid,
            anchors=anchor_set,
            order_seed=order_seed,
            reverse=reverse,
        )
        raw = _call_runner(runner, prompt)
        _parsed, trace = _parse_validate_or_repair(
            runner=runner,
            prompt=prompt,
            raw=raw,
            expected_candidate_ids=expected_candidate_ids,
        )
        outputs.append(trace)
        orders.append(order)
    first = outputs[0]["parsed_output"]
    second = outputs[1]["parsed_output"]
    first_result = GroupJudgeResult.from_mapping(first, expected_candidate_ids)
    second_result = GroupJudgeResult.from_mapping(second, expected_candidate_ids)
    instability = any(
        {
            "cross_view_relation_score": first_result.candidate_scores[cid].cross_view_relation_score,
            "semantic_naturalness_score": first_result.candidate_scores[cid].semantic_naturalness_score,
            "internal_consistency_score": first_result.candidate_scores[cid].internal_consistency_score,
            "anchor_tier": first_result.candidate_scores[cid].anchor_tier,
            "pairwise_preferences": first_result.candidate_scores[cid].pairwise_preferences,
        }
        != {
            "cross_view_relation_score": second_result.candidate_scores[cid].cross_view_relation_score,
            "semantic_naturalness_score": second_result.candidate_scores[cid].semantic_naturalness_score,
            "internal_consistency_score": second_result.candidate_scores[cid].internal_consistency_score,
            "anchor_tier": second_result.candidate_scores[cid].anchor_tier,
            "pairwise_preferences": second_result.candidate_scores[cid].pairwise_preferences,
        }
        for cid in first_result.candidate_scores
    )
    selected = (
        _stabilize_unstable_result(
            first_result=first_result,
            second_result=second_result,
            expected_candidate_ids=expected_candidate_ids,
        )
        if instability
        else first
    )
    return GroupJudgeResult.from_mapping(
        selected,
        expected_candidate_ids,
        raw_outputs=outputs,
        item_orders=orders,
        order_instability=instability,
    )
