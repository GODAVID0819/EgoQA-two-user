"""严格配对比较多个 LoRA 在同一 greedy 固定集上的结果。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


def _key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("evidence_id") or ""), str(row.get("question_type") or "")


def _index(rows: list[dict[str, Any]], label: str) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = _key(row)
        if not all(key):
            raise ValueError(f"{label} 存在缺少 evidence_id/question_type 的行")
        if key in result:
            raise ValueError(f"{label} 存在重复评估键: {key}")
        reward = row.get("reward")
        if not isinstance(reward, (int, float)):
            raise ValueError(f"{label} {key} reward 不是有限数字")
        result[key] = row
    return result


def _components(row: dict[str, Any]) -> dict[str, float]:
    values = (row.get("record") or {}).get("reward_components") or {}
    return {str(key): float(value) for key, value in values.items() if isinstance(value, (int, float))}


def compare_runs(runs: dict[str, list[dict[str, Any]]], *, baseline_label: str) -> dict[str, Any]:
    if baseline_label not in runs:
        raise ValueError(f"缺少 baseline: {baseline_label}")
    indexed = {label: _index(rows, label) for label, rows in runs.items()}
    baseline_keys = set(indexed[baseline_label])
    for label, rows in indexed.items():
        if set(rows) != baseline_keys:
            raise ValueError(f"{label} 与 {baseline_label} 评估集合不一致")
    summaries: dict[str, Any] = {}
    for label, rows in indexed.items():
        ordered = [rows[key] for key in sorted(baseline_keys)]
        grounded = Counter(str((row.get("record") or {}).get("groundedness_status") or "") for row in ordered)
        formats = Counter(str(((row.get("record") or {}).get("format_validation") or {}).get("status") or "") for row in ordered)
        component_names = sorted({name for row in ordered for name in _components(row)})
        summaries[label] = {
            "reward_mean": mean(float(row["reward"]) for row in ordered),
            "groundedness_counts": dict(grounded),
            "format_counts": dict(formats),
            "component_means": {
                name: mean(_components(row).get(name, 0.0) for row in ordered) for name in component_names
            },
        }
    comparisons: dict[str, Any] = {}
    baseline = indexed[baseline_label]
    for label, rows in indexed.items():
        if label == baseline_label:
            continue
        deltas = [float(rows[key]["reward"]) - float(baseline[key]["reward"]) for key in sorted(baseline_keys)]
        component_names = sorted({name for key in baseline_keys for name in {*_components(rows[key]), *_components(baseline[key])}})
        comparisons[f"{label}_vs_{baseline_label}"] = {
            "wins": sum(delta > 0 for delta in deltas),
            "ties": sum(delta == 0 for delta in deltas),
            "losses": sum(delta < 0 for delta in deltas),
            "reward_delta_mean": mean(deltas),
            "component_delta_means": {
                name: mean(_components(rows[key]).get(name, 0.0) - _components(baseline[key]).get(name, 0.0) for key in sorted(baseline_keys))
                for name in component_names
            },
        }
    paired = []
    for key in sorted(baseline_keys):
        paired.append({
            "evidence_id": key[0], "question_type": key[1],
            "runs": {
                label: {
                    "reward": rows[key]["reward"],
                    "raw_completion": rows[key].get("raw_completion"),
                    "groundedness_status": (rows[key].get("record") or {}).get("groundedness_status"),
                    "combined_correct": (rows[key].get("record") or {}).get("combined_correct"),
                    "reward_components": _components(rows[key]),
                } for label, rows in indexed.items()
            },
        })
    return {"schema_version": "grpo_v3_greedy_compare_v1", "baseline": baseline_label, "paired_count": len(baseline_keys), "summaries": summaries, "comparisons": comparisons, "paired_rows": paired}


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _markdown(result: dict[str, Any]) -> str:
    lines = ["# LoRA 固定集 greedy 配对评估", "", f"配对样本：{result['paired_count']}；baseline：`{result['baseline']}`。", "", "## 总览", "", "| run | reward 均值 | groundedness | JSON 格式 |", "|---|---:|---|---|"]
    for label, summary in result["summaries"].items():
        lines.append(f"| {label} | {summary['reward_mean']:.4f} | {json.dumps(summary['groundedness_counts'], ensure_ascii=False)} | {json.dumps(summary['format_counts'], ensure_ascii=False)} |")
    lines.extend(["", "## 配对变化", ""])
    for name, comparison in result["comparisons"].items():
        lines.append(f"- `{name}`：胜 {comparison['wins']}，平 {comparison['ties']}，负 {comparison['losses']}，平均 reward 变化 {comparison['reward_delta_mean']:+.4f}。")
    lines.extend(["", "## 逐题", ""])
    for row in result["paired_rows"]:
        lines.extend([f"### {row['evidence_id']} / {row['question_type']}", ""])
        for label, value in row["runs"].items():
            lines.append(f"- {label}：reward={value['reward']}，groundedness={value['groundedness_status']}，combined_correct={value['combined_correct']}；QA：{value['raw_completion']}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="比较多个 LoRA 的固定集 greedy 结果")
    parser.add_argument("--run", action="append", required=True, help="label=/path/to/results.jsonl")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    runs: dict[str, list[dict[str, Any]]] = {}
    for spec in args.run:
        label, separator, value = spec.partition("=")
        if not separator or not label or label in runs:
            raise ValueError(f"无效或重复 --run: {spec}")
        runs[label] = _read(Path(value))
    result = compare_runs(runs, baseline_label=args.baseline)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_md.write_text(_markdown(result), encoding="utf-8")
    print(json.dumps({"paired_count": result["paired_count"], "json": str(args.output_json), "markdown": str(args.output_md)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
