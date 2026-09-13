"""Summarize paired all-six versus minimal-required video QA results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def index_rows(path: Path) -> dict[str, dict[str, Any]]:
    index = {}
    for row in read_jsonl(path):
        evaluation_id = str(row.get("evaluation_id") or "")
        if not evaluation_id:
            raise ValueError(f"{path}: result row is missing evaluation_id")
        if evaluation_id in index:
            raise ValueError(f"{path}: duplicate evaluation_id {evaluation_id}")
        index[evaluation_id] = row
    return index


def exact_mcnemar_pvalue(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = min(left_only, right_only)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / (
        2**discordant
    )
    return min(1.0, 2.0 * probability)


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    six = index_rows(args.six)
    required = index_rows(args.required)
    if set(six) != set(required):
        raise ValueError(
            "condition result IDs differ: "
            f"six_only={sorted(set(six) - set(required))} "
            f"required_only={sorted(set(required) - set(six))}"
        )
    if len(six) != args.expected_count:
        raise ValueError(f"expected {args.expected_count} paired rows, found {len(six)}")

    paired_rows = []
    both_correct = 0
    six_only_correct = 0
    required_only_correct = 0
    neither_correct = 0
    for evaluation_id, six_row in sorted(six.items(), key=lambda item: item[1]["case_index"]):
        required_row = required[evaluation_id]
        if six_row.get("status") != "completed" or required_row.get("status") != "completed":
            raise ValueError(
                f"incomplete pair {evaluation_id}: "
                f"six={six_row.get('status')} required={required_row.get('status')}"
            )
        if six_row.get("qa_id") != required_row.get("qa_id"):
            raise ValueError(f"QA mismatch for {evaluation_id}")
        six_correct = six_row.get("is_correct") is True
        required_correct = required_row.get("is_correct") is True
        if six_correct and required_correct:
            both_correct += 1
        elif six_correct:
            six_only_correct += 1
        elif required_correct:
            required_only_correct += 1
        else:
            neither_correct += 1
        paired_rows.append(
            {
                "evaluation_id": evaluation_id,
                "qa_id": six_row.get("qa_id"),
                "evidence_id": six_row.get("evidence_id"),
                "asker": six_row.get("asker"),
                "correct_choice": six_row.get("correct_choice"),
                "six_users_choice": six_row.get("predicted_choice"),
                "six_users_correct": six_correct,
                "required_users_choice": required_row.get("predicted_choice"),
                "required_users_correct": required_correct,
                "six_users_seconds": six_row.get("model_call_seconds"),
                "required_users_seconds": required_row.get("model_call_seconds"),
            }
        )

    count = len(paired_rows)
    six_correct_count = both_correct + six_only_correct
    required_correct_count = both_correct + required_only_correct
    model_ids = {
        str(row.get("model_id") or "") for row in [*six.values(), *required.values()]
    }
    if len(model_ids) != 1:
        raise ValueError(f"condition outputs contain different models: {sorted(model_ids)}")
    summary = {
        "model_id": next(iter(model_ids)),
        "question_count": count,
        "six_users": {
            "correct_count": six_correct_count,
            "accuracy": six_correct_count / count,
        },
        "required_users": {
            "correct_count": required_correct_count,
            "accuracy": required_correct_count / count,
        },
        "required_minus_six_accuracy": (
            required_correct_count - six_correct_count
        )
        / count,
        "paired_outcomes": {
            "both_correct": both_correct,
            "six_only_correct": six_only_correct,
            "required_only_correct": required_only_correct,
            "neither_correct": neither_correct,
        },
        "exact_mcnemar_pvalue": exact_mcnemar_pvalue(
            six_only_correct, required_only_correct
        ),
        "six_results": str(args.six),
        "required_results": str(args.required),
        "per_question_output": str(args.per_question_output),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with args.per_question_output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in paired_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--six", type=Path, required=True)
    parser.add_argument("--required", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-question-output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=17)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(summarize(args), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
