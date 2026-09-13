"""历史 attempt 的 Reward v0 回放命令行入口。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .extract_attempts import iter_attempt_records
from .rewards import REWARD_VERSION, compute_reward


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(cwd: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _label_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals = [float(row["total"]) for row in rows]
    complete_count = sum(bool(row["is_complete_reward"]) for row in rows)
    return {
        "count": len(rows),
        "observed_total_mean": statistics.fmean(totals) if totals else None,
        "observed_total_median": statistics.median(totals) if totals else None,
        "complete_reward_count": complete_count,
        "complete_reward_coverage": complete_count / len(rows) if rows else 0.0,
    }


def _contradiction_cases(rows: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    accepted = sorted((row for row in rows if row["accepted"]), key=lambda row: row["total"])
    rejected = sorted((row for row in rows if not row["accepted"]), key=lambda row: row["total"], reverse=True)
    candidates: list[dict[str, Any]] = []
    if rows:
        center = statistics.median(float(row["total"]) for row in rows)
        for row in accepted:
            candidates.append({
                "attempt_id": row["attempt_id"],
                "historical_label": "accepted",
                "observed_total": row["total"],
                "distance_from_global_median": center - float(row["total"]),
                "missing_components": row["missing_components"],
            })
        for row in rejected:
            candidates.append({
                "attempt_id": row["attempt_id"],
                "historical_label": "rejected",
                "observed_total": row["total"],
                "distance_from_global_median": float(row["total"]) - center,
                "missing_components": row["missing_components"],
            })
    candidates.sort(key=lambda row: float(row["distance_from_global_median"]), reverse=True)
    return candidates[:limit]


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [row for row in rows if row["accepted"]]
    rejected = [row for row in rows if not row["accepted"]]
    missing_counts = Counter(
        component
        for row in rows
        for component in row["missing_components"]
    )
    failed_counts = Counter()
    for row in rows:
        for field, component in (
            ("parse_success", "parse"),
            ("schema_pass", "schema"),
            ("formality_pass", "formality"),
            ("groundedness_pass", "groundedness"),
            ("combined_correct", "combined"),
        ):
            if row[field] is False:
                failed_counts[component] += 1
        if row["speaker_alone_correct"] is True:
            failed_counts["speaker_leakage"] += 1
    complete_count = sum(bool(row["is_complete_reward"]) for row in rows)
    return {
        "reward_version": REWARD_VERSION,
        "attempt_count": len(rows),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "complete_reward_count": complete_count,
        "complete_reward_coverage": complete_count / len(rows) if rows else 0.0,
        "by_historical_label": {
            "accepted": _label_stats(accepted),
            "rejected": _label_stats(rejected),
        },
        "missing_component_counts": dict(sorted(missing_counts.items())),
        "failed_component_counts": dict(sorted(failed_counts.items())),
        "contradiction_cases": _contradiction_cases(rows),
    }


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})


def _summary_markdown(summary: dict[str, Any]) -> str:
    accepted = summary["by_historical_label"]["accepted"]
    rejected = summary["by_historical_label"]["rejected"]
    lines = [
        "# GRPO-Ready P0 历史奖励回放摘要",
        "",
        f"- Reward 版本：`{summary['reward_version']}`",
        f"- 总 attempts：{summary['attempt_count']}",
        f"- 历史 accepted：{summary['accepted_count']}",
        f"- 历史 rejected：{summary['rejected_count']}",
        f"- 完整奖励覆盖率：{summary['complete_reward_coverage']:.2%}",
        "",
        "## 按历史标签统计",
        "",
        "| 标签 | 数量 | observed reward 均值 | 中位数 | 完整覆盖率 |",
        "|---|---:|---:|---:|---:|",
        f"| accepted | {accepted['count']} | {accepted['observed_total_mean']:.4f} | {accepted['observed_total_median']:.4f} | {accepted['complete_reward_coverage']:.2%} |",
        f"| rejected | {rejected['count']} | {rejected['observed_total_mean']:.4f} | {rejected['observed_total_median']:.4f} | {rejected['complete_reward_coverage']:.2%} |",
        "",
        "## 缺失组件",
        "",
    ]
    if summary["missing_component_counts"]:
        lines.extend(
            f"- `{name}`：{count}"
            for name, count in summary["missing_component_counts"].items()
        )
    else:
        lines.append("- 无")
    lines.extend(["", "## Reward 与历史标签矛盾候选", ""])
    for case in summary["contradiction_cases"]:
        lines.append(
            f"- `{case['attempt_id']}`：历史 {case['historical_label']}，"
            f"observed reward={case['observed_total']:.4f}"
        )
    return "\n".join(lines) + "\n"


def run_replay(input_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    source = Path(input_path).resolve()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for attempt in iter_attempt_records(source):
        row = attempt.to_dict()
        row.update(compute_reward(attempt).to_dict())
        rows.append(row)
    if not rows:
        raise ValueError(f"no attempts found in {source}")

    summary = _summary(rows)
    _write_jsonl(destination / "reward_replay_results.jsonl", rows)
    _write_csv(destination / "reward_replay_results.csv", rows)
    (destination / "reward_replay_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (destination / "reward_replay_summary.md").write_text(
        _summary_markdown(summary), encoding="utf-8"
    )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(source),
        "input_sha256": _sha256(source),
        "git_commit": _git_commit(Path.cwd()),
        "reward_version": REWARD_VERSION,
        "python_version": platform.python_version(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "attempt_count": len(rows),
    }
    (destination / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="回放历史 EgoQA attempts 的 Reward v0")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    summary = run_replay(args.input, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
