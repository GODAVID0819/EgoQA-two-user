from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .deterministic import DeterministicAssessment
from .domain import JudgeCandidate
from .judge import NonThinkingTextJudgeRunner, judge_candidate_group
from .reward import compute_group_rewards


CALIBRATION_PADDING_QA = {
    "question_type": "neutral",
    "question": "After I left the counter, where did the mug end up?",
    "options": ["counter", "sink", "beside the laptop", "shelf", "being carried"],
    "answer": "beside the laptop",
}


def _assessment(qa: dict[str, Any]) -> DeterministicAssessment:
    return DeterministicAssessment(
        format_status="raw_valid",
        qa=qa,
        eligible_for_semantic_judge=True,
        blocking_errors=(),
        audit_flags=(),
        format_validation={"status": "reviewer_fixture"},
    )


def run_audit(
    fixture: dict[str, Any],
    *,
    runner: Any,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    group_traces: list[dict[str, Any]] = []
    cases = list(fixture["cases"])
    for group_index, offset in enumerate(range(0, len(cases), 4)):
        group_cases = [
            {**case, "_is_padding": False}
            for case in cases[offset:offset + 4]
        ]
        while len(group_cases) < 4:
            padding_index = len(group_cases)
            group_cases.append(
                {
                    "case_id": f"calibration_padding_{group_index}_{padding_index}",
                    "question": CALIBRATION_PADDING_QA["question"],
                    "options": CALIBRATION_PADDING_QA["options"],
                    "answer": CALIBRATION_PADDING_QA["answer"],
                    "expected_text_issue": False,
                    "_is_padding": True,
                }
            )
        case_by_id: dict[str, dict[str, Any]] = {}
        assessments: dict[str, DeterministicAssessment] = {}
        candidates: list[JudgeCandidate] = []
        for case in group_cases:
            candidate_id = str(case["case_id"])
            qa = {
                "question_type": "neutral",
                "question": case["question"],
                "options": case["options"],
                "answer": case["answer"],
            }
            case_by_id[candidate_id] = case
            assessments[candidate_id] = _assessment(qa)
            candidates.append(
                JudgeCandidate(
                    candidate_id=candidate_id,
                    raw_completion=json.dumps(qa, ensure_ascii=False),
                    qa=qa,
                )
            )
        judged = judge_candidate_group(
            candidates=candidates,
            deterministic_results=assessments,
            runner=runner,
            order_seed=f"reviewer_audit_group_{group_index}",
            require_text_checks=True,
        )
        candidate_ids = [candidate.candidate_id for candidate in candidates]
        rewards = compute_group_rewards(
            candidate_ids=candidate_ids,
            deterministic_results=assessments,
            judge_result=judged,
            apply_text_caps=True,
            reward_revision="qa_cross_view_relation_v3",
        )
        group_traces.append(
            {
                "group_index": group_index,
                "candidate_ids": candidate_ids,
                "raw_outputs": list(judged.raw_outputs),
                "item_orders": [list(order) for order in judged.item_orders],
                "order_instability": judged.order_instability,
            }
        )
        for candidate_id in candidate_ids:
            case = case_by_id[candidate_id]
            if case["_is_padding"]:
                continue
            reward = rewards[candidate_id]
            results.append(
                {
                    "case_id": candidate_id,
                    "group_index": group_index,
                    "expected_text_issue": bool(case["expected_text_issue"]),
                    "reward": reward.reward_total,
                    "reward_before_cap": reward.reward_before_cap,
                    "reward_cap": reward.reward_cap,
                    "cap_reasons": list(reward.cap_reasons),
                    "judge_score": judged.candidate_scores[candidate_id].to_dict(),
                }
            )

    negatives = [item for item in results if item["expected_text_issue"]]
    nonnegatives = [item for item in results if not item["expected_text_issue"]]
    return {
        "schema_version": "qa_cross_view_relation_reviewer_audit_result_v1",
        "source_job": fixture.get("source_job"),
        "case_count": len(results),
        "negative_count": len(negatives),
        "nonnegative_count": len(nonnegatives),
        "negative_high_reward_count": sum(item["reward"] > 0.9 for item in negatives),
        "negative_cap_trigger_count": sum(item["reward_cap"] < 1.0 for item in negatives),
        "nonnegative_false_rejection_count": sum(
            item["reward_cap"] < 1.0 for item in nonnegatives
        ),
        "order_instability_rate": (
            sum(item["order_instability"] for item in group_traces) / len(group_traces)
            if group_traces
            else None
        ),
        "groups": group_traces,
        "cases": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen text-only reviewer audit.")
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args()
    fixture = json.loads(args.fixture.read_text(encoding="utf-8-sig"))
    runner = NonThinkingTextJudgeRunner(
        model_id=args.model,
        base_url=args.base_url,
        max_new_tokens=args.max_new_tokens,
        timeout=args.timeout_seconds,
        allow_video_input=False,
    )
    result = run_audit(fixture, runner=runner)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in {"cases", "groups"}
            },
            indent=2,
        )
    )
    if (
        result["negative_high_reward_count"] != 0
        or result["negative_cap_trigger_count"] != result["negative_count"]
        or result["nonnegative_false_rejection_count"] != 0
        or float(result["order_instability_rate"] or 0.0) > 0.20
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
