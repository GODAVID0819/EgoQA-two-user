"""Deterministic fixed-rollout GRPO sign probe for the A-density experiment.

This probe intentionally does not call generation. It checks the shortest
reward -> advantage -> policy-gradient loss -> checkpoint reload path with a
fixed four-candidate rollout: A, B, A, B.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def fixed_rollout_spec() -> dict[str, Any]:
    return {
        "schema_version": "text_only_a_density_fixed_rollout_spec_v1",
        "prompt_count": 1,
        "completion_texts": ["A", "B", "A", "B"],
        "completion_token_ids": [0, 1, 0, 1],
        "rewards": [1.0, -1.0, 1.0, -1.0],
        "advantages": [1.0, -1.0, 1.0, -1.0],
        "uses_generation": False,
        "uses_video": False,
        "uses_reviewer_or_judge": False,
        "parses_json": False,
    }


def _load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def analyze_fixed_rollout_summary(path: Path) -> dict[str, Any]:
    summary = _load_summary(path)
    initial = float(summary.get("initial_margin", float("nan")))
    final = float(summary.get("final_margin", float("nan")))
    reloaded = float(summary.get("reloaded_margin", float("nan")))
    delta = final - initial
    reload_delta = abs(reloaded - final)
    checks = {
        "ran_10_steps": int(summary.get("steps", -1)) == 10,
        "margins_are_finite": all(math.isfinite(value) for value in (initial, final, reloaded)),
        "margin_improved_by_at_least_0_5": math.isfinite(delta) and delta >= 0.5,
        "reload_preserves_final_margin": math.isfinite(reload_delta) and reload_delta <= 1e-6,
        "nonzero_trainable_delta": bool(summary.get("nonzero_trainable_delta")),
        "all_grad_norms_finite": bool(summary.get("all_grad_norms_finite")),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": "text_only_a_density_fixed_rollout_analysis_v1",
        "status": "passed" if not failed else "not_converged",
        "checks": checks,
        "failed_checks": failed,
        "initial_margin": initial,
        "final_margin": final,
        "reloaded_margin": reloaded,
        "final_minus_initial_margin": delta,
        "reload_abs_margin_delta": reload_delta,
    }


def run_probe(output_dir: Path, *, steps: int = 10, learning_rate: float = 0.2, seed: int = 42) -> dict[str, Any]:
    import torch

    torch.manual_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = fixed_rollout_spec()
    completion_ids = torch.tensor(spec["completion_token_ids"], dtype=torch.long)
    advantages = torch.tensor(spec["advantages"], dtype=torch.float32)
    theta = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.SGD([theta], lr=learning_rate)

    trace: list[dict[str, Any]] = []

    def margin() -> float:
        logits = torch.stack([theta, -theta])
        return float((logits[0] - logits[1]).detach().cpu())

    initial_margin = margin()
    initial_theta = float(theta.detach().cpu())
    all_grad_norms_finite = True
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = torch.stack([theta, -theta])
        log_probs = torch.nn.functional.log_softmax(logits, dim=0)
        selected = log_probs[completion_ids]
        loss = -(selected * advantages).mean()
        loss.backward()
        grad_norm = float(theta.grad.detach().abs().cpu())
        all_grad_norms_finite = all_grad_norms_finite and math.isfinite(grad_norm)
        before = margin()
        optimizer.step()
        after = margin()
        trace.append(
            {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "grad_norm": grad_norm,
                "margin_before": before,
                "margin_after": after,
            }
        )

    checkpoint = output_dir / "fixed_rollout_checkpoint.pt"
    torch.save({"theta": theta.detach().cpu(), "steps": steps, "spec": spec}, checkpoint)
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    reloaded_theta = loaded["theta"].float()
    reloaded_margin = float(reloaded_theta - (-reloaded_theta))
    final_theta = float(theta.detach().cpu())
    summary = {
        "schema_version": "text_only_a_density_fixed_rollout_summary_v1",
        "steps": steps,
        "learning_rate": learning_rate,
        "seed": seed,
        "initial_theta": initial_theta,
        "final_theta": final_theta,
        "initial_margin": initial_margin,
        "final_margin": margin(),
        "reloaded_margin": reloaded_margin,
        "nonzero_trainable_delta": abs(final_theta - initial_theta) > 0,
        "all_grad_norms_finite": all_grad_norms_finite,
        "checkpoint": checkpoint.name,
        "spec": spec,
        "trace": trace,
    }
    summary_path = output_dir / "fixed_rollout_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    analysis = analyze_fixed_rollout_summary(summary_path)
    (output_dir / "fixed_rollout_analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic fixed-rollout GRPO sign probe")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run_probe(args.output_dir, steps=args.steps, learning_rate=args.learning_rate, seed=args.seed)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
