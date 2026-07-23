"""从历史 GRPO v3 trace 离线回放 qa_formality 连续 reward。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from training.grpo_v3.experiments.archived.formality.reward import FORMALITY_COMPONENT, confidence_reward


def _choice_logprobs(row: dict[str, Any]) -> tuple[float, float] | None:
    parsed = (
        (row.get("record") or {})
        .get("judge_trace", {})
        .get("qa_formality", {})
        .get("parsed", {})
    )
    signal = parsed.get("choice_logit_signal") if isinstance(parsed, dict) else None
    values = signal.get("choice_logprobs") if isinstance(signal, dict) else None
    if not isinstance(values, dict) or "PASS" not in values or "FAIL" not in values:
        return None
    try:
        pass_logprob = float(values["PASS"])
        fail_logprob = float(values["FAIL"])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(pass_logprob) or not math.isfinite(fail_logprob):
        return None
    return pass_logprob, fail_logprob


def replay_trace(
    rows: list[dict[str, Any]],
    *,
    input_sha256: str,
) -> dict[str, Any]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    finite_rewards: list[float] = []
    missing_logprob_count = 0
    for row in rows:
        values = _choice_logprobs(row)
        if values is None:
            missing_logprob_count += 1
            continue
        reward = confidence_reward(*values)
        replayed = {
            "candidate_index": int(row.get("candidate_index", -1)),
            "reward": reward,
        }
        grouped[int(row["reward_call_index"])].append(replayed)
        finite_rewards.append(reward)

    group_rows: list[dict[str, Any]] = []
    for call_index in sorted(grouped):
        members = sorted(grouped[call_index], key=lambda item: item["candidate_index"])
        indices = [item["candidate_index"] for item in members]
        if len(members) != 4 or indices != [0, 1, 2, 3]:
            continue
        rewards = [float(item["reward"]) for item in members]
        group_rows.append(
            {
                "reward_call_index": call_index,
                "reward_mean": statistics.fmean(rewards),
                "reward_std": statistics.pstdev(rewards),
                "rewards": rewards,
            }
        )

    positive_std_count = sum(group["reward_std"] > 0 for group in group_rows)
    positive_std_ratio = positive_std_count / len(group_rows) if group_rows else 0.0
    checks = {
        "has_complete_group": bool(group_rows),
        "all_replayed_rewards_finite": all(math.isfinite(value) for value in finite_rewards),
        "all_rewards_within_unit_interval": all(-1.0 <= value <= 1.0 for value in finite_rewards),
        "positive_std_ratio_at_least_0_8": positive_std_ratio >= 0.8,
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": "grpo_v3_formality_replay_v1",
        "status": "passed" if not failed_checks else "failed",
        "checks": checks,
        "failed_checks": failed_checks,
        "input_sha256": input_sha256,
        "input_row_count": len(rows),
        "finite_reward_count": len(finite_rewards),
        "missing_logprob_count": missing_logprob_count,
        "complete_group_count": len(group_rows),
        "positive_std_group_count": positive_std_count,
        "positive_std_ratio": positive_std_ratio,
        "reward_min": min(finite_rewards) if finite_rewards else None,
        "reward_max": max(finite_rewards) if finite_rewards else None,
        "reward_mean": statistics.fmean(finite_rewards) if finite_rewards else None,
        "reward_components": [FORMALITY_COMPONENT],
        "groups": group_rows,
    }


def replay_file(trace_path: Path, output_path: Path) -> dict[str, Any]:
    payload = trace_path.read_bytes()
    rows = [
        json.loads(line)
        for line in payload.decode("utf-8").splitlines()
        if line.strip()
    ]
    result = replay_trace(
        rows,
        input_sha256=hashlib.sha256(payload).hexdigest(),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="离线回放 qa_formality 连续 reward")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay_file(args.trace, args.output)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
