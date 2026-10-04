"""检查实际训练产物，单独报告候选数、视频组覆盖和更新证据。"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path


def finite_updated_adapter(path):
    import torch
    from safetensors.torch import load_file
    tensors = load_file(str(path), device="cpu")
    if not tensors or not all(bool(torch.isfinite(value).all()) for value in tensors.values()):
        return False
    return any("lora_B" in name and bool(value.count_nonzero()) for name, value in tensors.items())


def validate(output, *, adapter_check=finite_updated_adapter):
    output = Path(output)
    config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (output / "reward_trace.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    states = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in (output / "swift").rglob("trainer_state.json")]
    selected = max(states, key=lambda item: int(item[1].get("global_step", 0))) if states else None
    state = selected[1] if selected else {}
    checkpoint = selected[0].parent if selected else output / "missing_checkpoint"
    history = state.get("log_history", [])
    rewards = [r.get("reward") for r in rows]
    finite = all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and 0 <= x <= 1 for x in rewards)
    groups = defaultdict(list)
    for row in rows:
        key = (row["request_id"].rsplit(":", 1)[0], row["evidence_id"])
        groups[key].append(row["reward"])
    allowed_sizes = {config["num_generations"], config["num_generations_eval"]}
    nonconstant = sum(len(values) > 1 and max(values) > min(values) for values in groups.values()) if finite else 0
    adapter = checkpoint / "adapter_model.safetensors"
    checks = {
        "nonempty_reward_trace": bool(rows),
        "finite_rewards": finite,
        "unique_request_ids": len({r["request_id"] for r in rows}) == len(rows),
        "complete_observed_groups": bool(groups) and all(len(v) in allowed_sizes for v in groups.values()),
        "observed_group_variance_positive": nonconstant > 0,
        "at_least_two_judged_candidates": sum(r["judge_result"]["status"] == "scored" for r in rows) >= 2,
        "requested_steps_reached": int(state.get("global_step", 0)) >= config["max_steps"],
        "finite_nonzero_gradient": any(isinstance(r.get("grad_norm"), (int, float)) and math.isfinite(r["grad_norm"]) and r["grad_norm"] > 0 for r in history),
        "finite_reported_metrics": all(math.isfinite(v) for row in history for v in row.values() if isinstance(v, (int, float))),
        "validation_metrics_present": any(any(k.startswith("eval_") for k in row) for row in history),
        "finite_nonzero_lora_b": adapter.is_file() and adapter_check(adapter),
    }
    return {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value], "global_step": state.get("global_step", 0),
        "checkpoint": str(checkpoint), "candidate_count": len(rows), "observed_group_count": len(groups),
        "nonconstant_group_count": nonconstant, "source_packet_count": len({r["source_packet_id"] for r in rows}),
        "candidate_scope": "训练和验证评分合计；不将该计数称为终态 QA 数",
        "evidence_boundary": "仅训练工程证据；不证明固定测试集、人类 QA 质量或跨视频泛化"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = validate(args.output)
    except Exception as exc:
        result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    (args.output / "training_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
