"""Answer-margin scorer probe 与 calibration Gate 验证。"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from training.grpo_v3_answer_margin import LABELS


FIXED_EVIDENCE_ID = "EGOLIFE2U_DAY2_11350000_A1_A5"


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _complete_scores(value: Any) -> bool:
    return isinstance(value, Mapping) and set(value) == set(LABELS) and all(_finite_number(value[label]) for label in LABELS)


def _video_contract(request: Mapping[str, Any]) -> bool:
    videos = request.get("ordered_videos")
    if not isinstance(videos, list) or len(videos) != 2:
        return False
    for video in videos:
        if not isinstance(video, Mapping):
            return False
        metadata = video.get("processor_metadata")
        if not (
            str(video.get("path") or "").lower().endswith(".mp4")
            and len(str(video.get("sha256") or "")) == 64
            and int(video.get("size_bytes") or 0) > 0
            and isinstance(metadata, Mapping)
            and all(name in metadata for name in ("fps", "num_frames", "max_pixels", "min_pixels"))
        ):
            return False
    return True


def validate_scorer_probe(payload: Mapping[str, Any]) -> dict[str, Any]:
    requests = payload.get("requests") if isinstance(payload, Mapping) else None
    rows = requests if isinstance(requests, list) else []
    scores = [row.get("sequence_scores") if isinstance(row, Mapping) else None for row in rows]
    scores_finite = len(rows) == 2 and all(_complete_scores(value) for value in scores)
    computed_top1 = [max(LABELS, key=lambda label: float(value[label])) for value in scores] if scores_finite else []
    checks = {
        "fixed_evidence": payload.get("evidence_id") == FIXED_EVIDENCE_ID and all(row.get("evidence_id") == FIXED_EVIDENCE_ID for row in rows if isinstance(row, Mapping)),
        "health_passed": isinstance(payload.get("health"), Mapping) and payload["health"].get("status") == "passed",
        "exact_two_repeat_requests": len(rows) == 2,
        "ordered_native_dual_video_complete": len(rows) == 2 and all(isinstance(row, Mapping) and _video_contract(row) for row in rows) and rows[0].get("ordered_videos") == rows[1].get("ordered_videos"),
        "all_sequence_scores_finite": scores_finite,
        "repeat_top1_consistent": scores_finite and len(set(computed_top1)) == 1 and all(row.get("top1") == computed_top1[index] for index, row in enumerate(rows)),
        "repeat_scores_within_tolerance": scores_finite and all(math.isclose(float(scores[0][label]), float(scores[1][label]), rel_tol=1e-4, abs_tol=1e-4) for label in LABELS),
        "zero_trainable_parameters": payload.get("trainable_parameter_count") == 0,
        "prompt_leakage_scan_passed": len(rows) == 2 and all(isinstance(row.get("prompt_audit"), Mapping) and row["prompt_audit"].get("passed") is True and not row["prompt_audit"].get("leaked_fields") for row in rows if isinstance(row, Mapping)),
        "label_audit_complete": len(rows) == 2 and all(isinstance(row.get("label_details"), Mapping) and set(row["label_details"]) == set(LABELS) for row in rows if isinstance(row, Mapping)),
        "runtime_metadata_complete": isinstance(payload.get("runtime"), Mapping) and all(name in payload["runtime"] for name in ("device", "gpu_name", "peak_memory_bytes", "elapsed_seconds")) and _finite_number(payload["runtime"].get("peak_memory_bytes")) and _finite_number(payload["runtime"].get("elapsed_seconds")),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": "answer_margin_scorer_probe_result_v1",
        "status": "passed" if not failed else "failed",
        "evidence_id": payload.get("evidence_id"),
        "request_count": len(rows),
        "checks": checks,
        "failed_checks": failed,
    }


def _trace_complete(row: Mapping[str, Any]) -> bool:
    record = row.get("record")
    if not isinstance(record, Mapping):
        return False
    labels = record.get("label_scores")
    permutation = record.get("permutation")
    inverse = record.get("inverse_permutation")
    return (
        record.get("evidence_id") == FIXED_EVIDENCE_ID
        and record.get("experiment_condition_id") == "t05"
        and record.get("temperature") == 0.5
        and isinstance(record.get("format_validation"), Mapping)
        and isinstance(permutation, list) and sorted(permutation) == list(range(5))
        and isinstance(inverse, list) and sorted(inverse) == list(range(5))
        and isinstance(labels, Mapping) and set(labels) == set(LABELS)
        and all(isinstance(labels[label], Mapping) and _finite_number(labels[label].get("sequence_logprob")) for label in LABELS)
    )


def validate_calibration(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [float(row["reward"]) for row in rows if _finite_number(row.get("reward"))]
    groups: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        record = row.get("record") if isinstance(row, Mapping) else None
        if isinstance(record, Mapping) and isinstance(record.get("reward_call_index"), int):
            groups.setdefault(int(record["reward_call_index"]), []).append(row)
    exact_shape = set(groups) == set(range(8)) and all(len(group) == 4 and {item.get("record", {}).get("candidate_index") for item in group} == set(range(4)) for group in groups.values())
    masks = sum(1 for row in rows if isinstance(row.get("record"), Mapping) and row["record"].get("masked") is True)
    completeness = {
        "exact_8_groups_x_4_candidates": len(rows) == 32 and exact_shape,
        "all_32_rewards_finite": len(values) == len(rows) == 32,
        "zero_infrastructure_masks": masks == 0,
        "trace_audit_complete": len(rows) == 32 and all(_trace_complete(row) for row in rows),
    }
    positive_groups = 0
    for group in groups.values():
        group_values = [float(item["reward"]) for item in group if _finite_number(item.get("reward"))]
        positive_groups += int(len(group_values) == 4 and statistics.pstdev(group_values) > 0)
    clipped = sum(abs(value) == 1.0 for value in values)
    research = {
        "at_least_6_positive_variance_groups": positive_groups >= 6,
        "at_least_2_distinct_rewards": len(set(values)) >= 2,
        "clip_ratio_at_most_20_percent": len(values) == 32 and clipped / 32 <= 0.20,
    }
    failed_completeness = [name for name, passed in completeness.items() if not passed]
    failed_research = [name for name, passed in research.items() if not passed]
    return {
        "schema_version": "answer_margin_calibration_result_v1",
        "status": "passed" if not failed_completeness and not failed_research else "failed",
        "completeness_status": "passed" if not failed_completeness else "failed",
        "research_signal_status": "passed" if not failed_research else "failed",
        "checks": {**completeness, **research},
        "failed_completeness_checks": failed_completeness,
        "failed_research_checks": failed_research,
        "row_count": len(rows),
        "finite_reward_count": len(values),
        "masked_reward_count": masks,
        "positive_variance_group_count": positive_groups,
        "distinct_reward_count": len(set(values)),
        "clip_count": clipped,
        "clip_ratio": clipped / len(values) if values else None,
    }


def _strict_json(path: Path) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"JSON 不允许非有限常量: {value}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def _strict_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line, parse_constant=lambda item: (_ for _ in ()).throw(ValueError(f"第 {number} 行包含 {item}")))
        if not isinstance(value, dict):
            raise ValueError(f"第 {number} 行必须是 JSON object")
        rows.append(value)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="验证 answer-margin scorer probe 或 calibration")
    parser.add_argument("--kind", choices=("scorer-probe", "calibration"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate_scorer_probe(_strict_json(args.input)) if args.kind == "scorer-probe" else validate_calibration(_strict_jsonl(args.input))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
