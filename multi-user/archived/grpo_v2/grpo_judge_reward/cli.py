from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .extract import iter_reward_inputs
from .group import compute_group_records
from .scoring import compute_judge_reward


def _git_commit() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})


def _summary(records: list[dict[str, Any]], groups: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = [float(row["reward_total"]) for row in records if isinstance(row.get("reward_total"), (int, float))]
    return {
        "raw_candidates": len(records),
        "masked_candidates": sum(row.get("masked") is True for row in records),
        "valid_candidates": len(rewards),
        "schema_masked": sum(row.get("mask_reason") == "schema_fail" for row in records),
        "missing_review_signals": sum(row.get("mask_reason") == "missing_review_signals" for row in records),
        "groundedness_pass": sum(row.get("groundedness_status") == "PASS" for row in records),
        "groundedness_uncertain": sum(row.get("groundedness_status") == "UNCERTAIN" for row in records),
        "groundedness_fail": sum(row.get("groundedness_status") == "FAIL" for row in records),
        "combined_correct": sum(row.get("combined_correct") is True for row in records),
        "combined_wrong_or_insufficient": sum(row.get("combined_correct") is False for row in records),
        "speaker_leakage_count": sum(row.get("speaker_only_correct") is True for row in records),
        "subset_leakage_count": sum(row.get("proper_subset_correct") is True for row in records),
        "provider_only_correct_count": sum(row.get("provider_only_correct") is True for row in records),
        "shallow_activity_fail_count": sum(row.get("shallow_activity_status") == "FAIL" for row in records),
        "trainer_eligible_groups": sum(row.get("trainer_eligible") is True for row in groups),
        "zero_variance_groups": sum(row.get("skip_reason") == "zero_reward_variance" for row in groups),
        "too_few_valid_groups": sum(row.get("skip_reason") == "too_few_valid_candidates" for row in groups),
        "reward_mean": round(sum(rewards) / len(rewards), 6) if rewards else None,
        "reward_min": min(rewards) if rewards else None,
        "reward_max": max(rewards) if rewards else None,
    }


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = ["# GRPO v2 Judger Reward 摘要", ""]
    for key, value in summary.items():
        lines.append(f"- `{key}`: {value}")
    return "\n".join(lines) + "\n"


def _case_studies(records: list[dict[str, Any]]) -> str:
    scored = sorted((row for row in records if isinstance(row.get("reward_total"), (int, float))), key=lambda row: row["reward_total"])
    lines = ["# GRPO v2 人工审查案例", "", "## 最高 reward", ""]
    for row in reversed(scored[-4:]):
        lines.append(f"- `{row.get('candidate_id')}`: reward={row.get('reward_total')}, groundedness={row.get('groundedness_status')}, combined={row.get('combined_correct')}, speaker_leakage={row.get('speaker_only_correct')}")
    lines.extend(["", "## 最低 reward", ""])
    for row in scored[:4]:
        lines.append(f"- `{row.get('candidate_id')}`: reward={row.get('reward_total')}, groundedness={row.get('groundedness_status')}, combined={row.get('combined_correct')}, speaker_leakage={row.get('speaker_only_correct')}")
    lines.extend(["", "## Speaker leakage", ""])
    for row in [item for item in records if item.get("speaker_only_correct") is True][:4]:
        lines.append(f"- `{row.get('candidate_id')}`: reward={row.get('reward_total')}, speaker={row.get('speaker_user')}, choice={row.get('speaker_only_choice')}")
    return "\n".join(lines) + "\n"


def run_score(input_path: str | Path, input_kind: str, output_dir: str | Path) -> dict[str, Any]:
    source = Path(input_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    records = [compute_judge_reward(row).to_dict() for row in iter_reward_inputs(source, input_kind)]
    groups = [row.to_dict() for row in compute_group_records(records)]
    summary = _summary(records, groups)

    _write_jsonl(destination / "judge_reward_records.jsonl", records)
    _write_csv(destination / "judge_reward_records.csv", records)
    _write_jsonl(destination / "judge_group_records.jsonl", groups)
    (destination / "judge_reward_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (destination / "judge_reward_summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    (destination / "judge_case_studies.md").write_text(_case_studies(records), encoding="utf-8")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(source.resolve()),
        "input_kind": input_kind,
        "git_commit": _git_commit(),
        "python_version": platform.python_version(),
        "reward_version": "judge_linear_v2_1_provider_shallow",
    }
    (destination / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score repo-native judge outputs for GRPO v2")
    sub = parser.add_subparsers(dest="command", required=True)
    score = sub.add_parser("score")
    score.add_argument("--input", required=True, type=Path)
    score.add_argument("--input-kind", choices=["intermediate", "accepted"], required=True)
    score.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "score":
        print(json.dumps(run_score(args.input, args.input_kind, args.output_dir), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
