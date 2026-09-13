"""Post-run acceptance checks for score-only GRPO smoke jobs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, TextIO

from safetensors.torch import load_file


def _normalized_csv_row(row: dict[str | None, str | None]) -> dict[str, str]:
    """Normalize nvidia-smi CSV fields across driver formatting variants."""
    return {
        str(key).strip(): str(value).strip()
        for key, value in row.items()
        if key is not None and value is not None
    }


def parse_gpu_peaks(handle: TextIO) -> dict[str, dict[str, int]]:
    peaks: dict[str, dict[str, int]] = {}
    reader = csv.DictReader(handle, skipinitialspace=True)
    for raw_row in reader:
        row = _normalized_csv_row(raw_row)
        missing = {
            "index",
            "memory.total [MiB]",
            "memory.used [MiB]",
        }.difference(row)
        if missing:
            raise ValueError(
                "gpu metrics CSV is missing required fields "
                f"{sorted(missing)}; fields={sorted(row)}"
            )
        index = row["index"]
        total = int(row["memory.total [MiB]"].split()[0])
        used = int(row["memory.used [MiB]"].split()[0])
        previous = peaks.get(
            index,
            {"total_mib": total, "peak_used_mib": 0},
        )
        if previous["total_mib"] != total:
            raise ValueError(f"GPU {index} total memory changed within metrics CSV")
        previous["peak_used_mib"] = max(previous["peak_used_mib"], used)
        peaks[index] = previous
    if not peaks:
        raise ValueError("gpu metrics CSV has no data rows")
    return peaks


def load_gpu_peaks(path: str | Path) -> dict[str, dict[str, int]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return parse_gpu_peaks(handle)


def validate_smoke_run(
    output_dir: str | Path,
    *,
    expected_groups: int,
    num_generations: int,
    minimum_policy_gpu_headroom_mib: int,
) -> dict[str, Any]:
    output = Path(output_dir)
    expected_rows = expected_groups * num_generations
    rows = [
        json.loads(line)
        for line in (output / "reward_trace.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    adapters = list(output.rglob("adapter_model.safetensors"))
    states = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in output.rglob("trainer_state.json")
    ]
    history = [row for state in states for row in state.get("log_history", [])]
    grad_norms = [
        float(row["grad_norm"])
        for row in history
        if row.get("grad_norm") is not None
    ]
    global_steps = [int(state.get("global_step", 0)) for state in states]
    valid_rows = [
        row
        for row in rows
        if row.get("record", {}).get("reward_source")
        == "score_only_ordinal_reviewer"
    ]
    rewards = [float(row["reward"]) for row in rows]
    reward_mean = sum(rewards) / len(rewards) if rewards else None
    reward_std = (
        math.sqrt(
            sum((value - reward_mean) ** 2 for value in rewards) / len(rewards)
        )
        if rewards and reward_mean is not None
        else None
    )

    lora_b_nonzero = False
    for path in adapters:
        tensors = load_file(str(path), device="cpu")
        lora_b_nonzero |= any(
            "lora_B" in name and bool(tensor.count_nonzero())
            for name, tensor in tensors.items()
        )

    gpu_peaks = load_gpu_peaks(output / "gpu_metrics.csv")
    policy_gpu = gpu_peaks.get("0")
    policy_headroom_mib = (
        policy_gpu["total_mib"] - policy_gpu["peak_used_mib"]
        if policy_gpu
        else None
    )
    checks = {
        "expected_reward_rows": len(rows) == expected_rows,
        "score_only_revision": all(
            row.get("reward_revision") == "score_only_ordinal_reviewer_v1"
            for row in rows
        ),
        "finite_rewards": all(math.isfinite(value) for value in rewards),
        "reward_range": all(0.0 <= value <= 1.0 for value in rewards),
        "three_head_scores_present": all(
            row["record"]["reward_source"] == "deterministic_rejection"
            or set(row["record"]["reviewer_score"]["expected_scores"])
            == {"evidence_quality", "answerability", "qa_formality"}
            for row in rows
        ),
        "valid_reviewer_rewards_at_least_two": len(valid_rows) >= 2,
        "reward_std_positive": reward_std is not None and reward_std > 0.0,
        "global_step_at_least_one": bool(global_steps) and max(global_steps) >= 1,
        "adapter_model_safetensors": bool(adapters),
        "finite_nonzero_grad_norm": any(
            math.isfinite(value) and value > 0.0 for value in grad_norms
        ),
        "lora_B_nonzero": lora_b_nonzero,
        "policy_gpu_headroom_at_least_requested": (
            policy_headroom_mib is not None
            and policy_headroom_mib >= minimum_policy_gpu_headroom_mib
        ),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "reward_count": len(rows),
        "valid_reviewer_reward_count": len(valid_rows),
        "reward_mean": reward_mean,
        "reward_std": reward_std,
        "global_step": max(global_steps) if global_steps else 0,
        "num_generations": num_generations,
        "gpu_peaks": gpu_peaks,
        "policy_gpu_headroom_mib": policy_headroom_mib,
        "minimum_policy_gpu_headroom_mib": minimum_policy_gpu_headroom_mib,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-groups", type=int, required=True)
    parser.add_argument("--num-generations", type=int, required=True)
    parser.add_argument("--minimum-policy-gpu-headroom-mib", type=int, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = validate_smoke_run(
        args.output_dir,
        expected_groups=args.expected_groups,
        num_generations=args.num_generations,
        minimum_policy_gpu_headroom_mib=args.minimum_policy_gpu_headroom_mib,
    )
    result_path = args.output_dir / "smoke_result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
