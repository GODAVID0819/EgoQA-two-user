"""验证 answer-margin 1/5/40-step 训练产物的完整性与研究信号。"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping

from training.grpo_v3_answer_margin import ANSWER_MARGIN_REWARD_REVISION, LABELS
from training.grpo_v3_answer_margin_preflight import FIXED_EVIDENCE_ID


MODE_COUNTS = {"smoke1": (1, 4, 1), "smoke5": (5, 20, 4), "probe40": (40, 160, 32)}


def expected_counts(mode: str) -> tuple[int, int, int]:
    try:
        return MODE_COUNTS[mode]
    except KeyError as exc:
        raise ValueError("mode 只能是 smoke1、smoke5 或 probe40") from exc


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"JSON 包含 {value}")))


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    result = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line, parse_constant=lambda item: (_ for _ in ()).throw(ValueError(f"第 {number} 行包含 {item}")))
        if not isinstance(value, dict):
            raise ValueError(f"第 {number} 行必须是 JSON object")
        result.append(value)
    return result


def _latest_state(output: Path) -> dict[str, Any]:
    states = [_json(path) for path in output.rglob("trainer_state.json")]
    return max(states, key=lambda value: int(value.get("global_step") or -1), default={})


def _trace_complete(row: Mapping[str, Any]) -> bool:
    record = row.get("record")
    if not isinstance(record, Mapping):
        return False
    labels = record.get("label_scores")
    return (
        record.get("reward_revision") == ANSWER_MARGIN_REWARD_REVISION
        and record.get("experiment_condition_id") == "t05"
        and record.get("temperature") == 0.5
        and record.get("evidence_id") == FIXED_EVIDENCE_ID
        and isinstance(record.get("format_validation"), Mapping)
        and sorted(record.get("permutation") or []) == list(range(5))
        and sorted(record.get("inverse_permutation") or []) == list(range(5))
        and isinstance(labels, Mapping) and set(labels) == set(LABELS)
        and all(isinstance(labels[label], Mapping) and isinstance(labels[label].get("sequence_logprob"), (int, float)) and math.isfinite(float(labels[label]["sequence_logprob"])) for label in LABELS)
    )


def validate_training_artifacts(output_dir: Path, *, mode: str) -> dict[str, Any]:
    steps, expected_rows, required_positive = expected_counts(mode)
    rows = _rows(output_dir / "reward_trace.jsonl")
    state = _latest_state(output_dir)
    config = _json(output_dir / "resolved_config.json")
    storage = _json(output_dir / "storage_preflight.json")
    environment = _json(output_dir / "environment_audit.json")
    inventory = _json(output_dir / "checkpoint_inventory.json")
    reload = _json(output_dir / "adapter_reload.json")
    manifest = _json(output_dir / "run_manifest.json")
    scorer_probe = _json(output_dir / "scorer_runtime_probe.json")
    groups: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        record = row.get("record")
        if isinstance(record, Mapping) and isinstance(record.get("reward_call_index"), int):
            groups.setdefault(int(record["reward_call_index"]), []).append(row)
    values = [float(row["reward"]) for row in rows if isinstance(row.get("reward"), (int, float)) and not isinstance(row.get("reward"), bool) and math.isfinite(float(row["reward"]))]
    masks = sum(isinstance(row.get("record"), Mapping) and row["record"].get("masked") is True for row in rows)
    exact_groups = set(groups) == set(range(steps)) and all(len(group) == 4 and {row.get("record", {}).get("candidate_index") for row in group} == set(range(4)) for group in groups.values())
    positive_groups = sum(len(group) == 4 and statistics.pstdev([float(row["reward"]) for row in group]) > 0 for group in groups.values() if all(isinstance(row.get("reward"), (int, float)) and math.isfinite(float(row["reward"])) for row in group))
    adapter_dirs = [path.parent for path in output_dir.rglob("adapter_config.json") if (path.parent / "adapter_model.safetensors").is_file() or (path.parent / "adapter_model.bin").is_file()]
    adapter_paths = {str(path.resolve()) for path in adapter_dirs}
    processor_dir = Path(str(reload.get("processor_dir") or output_dir / "processor"))
    processor_files = list(processor_dir.glob("*processor*config*.json")) + list(processor_dir.glob("preprocessor_config.json"))
    policy_env = environment.get("policy_environment") if isinstance(environment.get("policy_environment"), Mapping) else {}
    scorer_env = environment.get("scorer_environment") if isinstance(environment.get("scorer_environment"), Mapping) else {}
    integrity = {
        "exact_trace_count": len(rows) == expected_rows,
        "exact_four_candidates_per_group": exact_groups,
        "all_rewards_finite": len(values) == len(rows) == expected_rows,
        "zero_infrastructure_masks": masks == 0,
        "trace_audit_complete": len(rows) == expected_rows and all(_trace_complete(row) for row in rows),
        "required_global_step": state.get("global_step") == steps,
        "locked_condition_config": config.get("condition_id") == "t05" and config.get("temperature") == 0.5 and config.get("num_generations") == 4 and config.get("reward_revision") == ANSWER_MARGIN_REWARD_REVISION and config.get("evidence_id") == FIXED_EVIDENCE_ID,
        "storage_preflight_passed": storage.get("status") == "passed",
        "separate_environment_audits_passed": environment.get("status") == "passed" and policy_env.get("status") == "passed" and scorer_env.get("status") == "passed" and policy_env.get("python") != scorer_env.get("python"),
        "scorer_runtime_probe_passed": scorer_probe.get("status") == "passed" and scorer_probe.get("evidence_id") == FIXED_EVIDENCE_ID,
        "parent_inventory_frozen": inventory.get("status") == "passed" and inventory.get("parent_job") == "gate2_14119442" and inventory.get("parent_checkpoint") == "checkpoint-1" and inventory.get("source") == "manifest_and_hash_inventory" and inventory.get("gate2_result", {}).get("status") == "passed" and len(inventory.get("adapter_files") or []) > 0 and config.get("parent_job") == "gate2_14119442" and config.get("parent_checkpoint") == "checkpoint-1",
        "adapter_complete": bool(adapter_dirs),
        "processor_config_saved": bool(processor_files),
        "adapter_and_processor_reload_passed": reload.get("status") == "passed" and int(reload.get("lora_parameters") or 0) > 0 and str(Path(str(reload.get("adapter_dir") or ".")).resolve()) in adapter_paths and reload.get("processor_reloaded") is True and reload.get("inference_check", {}).get("status") == "passed",
        "run_manifest_complete": manifest.get("status") == "completed" and manifest.get("reward_revision") == ANSWER_MARGIN_REWARD_REVISION and manifest.get("condition_id") == "t05" and manifest.get("checkpoint_inventory") == "checkpoint_inventory.json",
        "trainer_state_complete": isinstance(state.get("log_history"), list) and bool(state["log_history"]),
    }
    research = {"required_positive_variance_groups": positive_groups >= required_positive}
    failed_integrity = [name for name, passed in integrity.items() if not passed]
    failed_research = [name for name, passed in research.items() if not passed]
    return {
        "schema_version": "answer_margin_training_artifacts_v1",
        "mode": mode,
        "run_status": "passed" if not failed_integrity else "invalid",
        "research_signal_status": "passed" if not failed_research else "failed",
        "integrity_checks": integrity,
        "research_checks": research,
        "failed_integrity_checks": failed_integrity,
        "failed_research_checks": failed_research,
        "trace_count": len(rows),
        "finite_reward_count": len(values),
        "masked_reward_count": masks,
        "global_step": int(state.get("global_step") or 0),
        "positive_variance_group_count": positive_groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="验证 answer-margin smoke1/smoke5/probe40 训练产物")
    parser.add_argument("--mode", choices=tuple(MODE_COUNTS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    result = validate_training_artifacts(args.output_dir, mode=args.mode)
    destination = args.result or args.output_dir / f"answer_margin_{args.mode}_result.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    if result["run_status"] != "passed" or result["research_signal_status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
