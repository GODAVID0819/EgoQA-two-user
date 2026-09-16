"""Compare one trained Reviewer evaluation with the shared untrained baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


AGGREGATE_DIRECTIONS = {
    "weighted_loss": "lower",
    "weighted_accuracy": "higher",
    "weighted_macro_f1": "higher",
    "weighted_balanced_accuracy": "higher",
}
PER_HEAD_METRICS = (
    "loss",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "auroc",
)


def _comparison(before: float, after: float, direction: str) -> dict[str, Any]:
    delta = after - before
    return {
        "base_untrained": before,
        "trained": after,
        "trained_minus_base": delta,
        "better_direction": direction,
        "trained_improved": delta < 0 if direction == "lower" else delta > 0,
    }


def compare_evaluation_results(
    base: Mapping[str, Any],
    trained: Mapping[str, Any],
    *,
    test_set: str,
    learning_rate: float | None = None,
) -> dict[str, Any]:
    if base.get("checkpoint_mode") != "base_untrained":
        raise ValueError("first result is not the base_untrained evaluation")
    if trained.get("checkpoint_mode") != "trained":
        raise ValueError("second result is not the trained-checkpoint evaluation")
    for field in (
        "evaluation_csv_sha256",
        "evaluation_split_sha256",
        "evaluation_evidence_count",
        "evaluation_candidate_count",
    ):
        if base.get(field) != trained.get(field):
            raise ValueError(f"evaluation inputs differ for {field}")

    aggregate = {
        metric: _comparison(
            float(base[metric]), float(trained[metric]), direction
        )
        for metric, direction in AGGREGATE_DIRECTIONS.items()
    }
    per_head: dict[str, dict[str, Any]] = {}
    if set(base["metrics"]) != set(trained["metrics"]):
        raise ValueError("evaluation head sets differ")
    for head in sorted(base["metrics"]):
        per_head[head] = {}
        for metric in PER_HEAD_METRICS:
            before = base["metrics"][head].get(metric)
            after = trained["metrics"][head].get(metric)
            if before is None or after is None:
                per_head[head][metric] = {
                    "base_untrained": before,
                    "trained": after,
                    "trained_minus_base": None,
                }
                continue
            direction = (
                "lower"
                if metric == "loss"
                else "higher"
            )
            per_head[head][metric] = _comparison(
                float(before), float(after), direction
            )

    return {
        "status": "passed",
        "test_set": test_set,
        "learning_rate": learning_rate,
        "evaluation_csv_sha256": base["evaluation_csv_sha256"],
        "evaluation_split_sha256": base["evaluation_split_sha256"],
        "evaluation_evidence_count": base["evaluation_evidence_count"],
        "evaluation_candidate_count": base["evaluation_candidate_count"],
        "trained_checkpoint_dir": trained["checkpoint_dir"],
        "aggregate": aggregate,
        "per_head": per_head,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--trained", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--test-set", required=True)
    parser.add_argument("--learning-rate", type=float)
    args = parser.parse_args(argv)

    base_path = Path(args.base)
    trained_path = Path(args.trained)
    output_path = Path(args.output)
    result = compare_evaluation_results(
        json.loads(base_path.read_text(encoding="utf-8")),
        json.loads(trained_path.read_text(encoding="utf-8")),
        test_set=args.test_set,
        learning_rate=args.learning_rate,
    )
    result["base_result"] = str(base_path.resolve())
    result["trained_result"] = str(trained_path.resolve())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
