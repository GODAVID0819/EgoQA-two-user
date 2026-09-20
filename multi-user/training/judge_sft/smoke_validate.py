"""Validate evidence from a one-step judge SFT cluster smoke."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_smoke(
    *,
    trainer_dir: Path,
    adapter_reload_path: Path,
    expected_model_id: str,
    expected_example_id: str,
) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file

    contract_path = trainer_dir / "training_contract.json"
    state_path = trainer_dir / "trainer_state.json"
    adapter_dir = trainer_dir / "final_adapter"
    adapter_path = adapter_dir / "adapter_model.safetensors"
    required = (
        contract_path,
        state_path,
        adapter_dir / "adapter_config.json",
        adapter_path,
        adapter_reload_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing smoke artifacts: " + ", ".join(missing))

    contract = _read_json(contract_path)
    state = _read_json(state_path)
    reload_result = _read_json(adapter_reload_path)
    history = list(state.get("log_history") or [])
    grad_norms = [
        float(row["grad_norm"])
        for row in history
        if row.get("grad_norm") is not None
    ]
    losses = [
        float(row["loss"])
        for row in history
        if row.get("loss") is not None
    ]
    tensors = load_file(str(adapter_path), device="cpu")
    lora_b = {
        name: tensor
        for name, tensor in tensors.items()
        if "lora_B" in name
    }
    finite_adapter = bool(tensors) and all(
        bool(torch.isfinite(tensor).all())
        for tensor in tensors.values()
        if tensor.is_floating_point()
    )
    lora_b_nonzero = bool(lora_b) and any(
        bool(torch.count_nonzero(tensor))
        for tensor in lora_b.values()
    )
    token_ids = contract.get("verdict_token_ids") or {}
    checks = {
        "model_id_exact": contract.get("model_id") == expected_model_id,
        "pure_tp2_contract": (
            ((contract.get("distributed_contract") or {}).get("tensor_parallel_size") == 2)
            and ((contract.get("distributed_contract") or {}).get("data_parallel_size") == 1)
        ),
        "independent_image_contract": (
            "independent Qwen image item" in str(contract.get("media_contract") or "")
        ),
        "bf16_lora_without_autocast_promotion": (
            (((contract.get("parameter_counts") or {}).get("lora_dtype_audit") or {}).get("expected") == "torch.bfloat16")
            and (((contract.get("parameter_counts") or {}).get("lora_dtype_audit") or {}).get("autocast_adapter_dtype") is False)
            and set(
                (((contract.get("parameter_counts") or {}).get("lora_dtype_audit") or {}).get("parameter_counts_by_dtype") or {})
            ) == {"torch.bfloat16"}
        ),
        "upper_16_decoder_layers_only": (
            (((contract.get("decoder_training_contract") or {}).get("total_decoder_layers")) == 64)
            and (((contract.get("decoder_training_contract") or {}).get("trainable_decoder_layers")) == 16)
            and (((contract.get("decoder_training_contract") or {}).get("frozen_prefix_layer_indices")) == list(range(48)))
            and (((contract.get("decoder_training_contract") or {}).get("trainable_layer_indices")) == list(range(48, 64)))
            and ((((contract.get("decoder_training_contract") or {}).get("frozen_prefix_autograd_guard") or {}).get("required_input_requires_grad")) is False)
        ),
        "single_selected_all_six_example": (
            ((contract.get("optimization_train") or {}).get("examples") == 1)
            and (
                (contract.get("hyperparameters") or {}).get("train_example_id")
                == expected_example_id
            )
        ),
        "global_step_exactly_one": int(state.get("global_step") or 0) == 1,
        "finite_positive_grad_norm": any(
            math.isfinite(value) and value > 0 for value in grad_norms
        ),
        "finite_loss": bool(losses) and all(math.isfinite(value) for value in losses),
        "distinct_verdict_tokens": (
            isinstance(token_ids.get("pass"), int)
            and isinstance(token_ids.get("fail"), int)
            and token_ids["pass"] != token_ids["fail"]
        ),
        "adapter_tensors_finite": finite_adapter,
        "lora_B_nonzero_after_step": lora_b_nonzero,
        "adapter_reload_passed": reload_result.get("status") == "passed",
        "adapter_reload_kept_bf16": (
            reload_result.get("lora_dtypes") == ["torch.bfloat16"]
            and reload_result.get("autocast_adapter_dtype") is False
        ),
        "reload_exercised_binary_logits": (
            ((reload_result.get("decision") or {}).get("verdict") in {"pass", "fail"})
            and isinstance(
                (reload_result.get("decision") or {}).get("pass_minus_fail_logit"),
                (int, float),
            )
        ),
        "reload_generated_complete_contract": (
            ((reload_result.get("generation") or {}).get("full_contract_valid") is True)
            and isinstance(
                (reload_result.get("generation") or {}).get("text"),
                str,
            )
            and bool((reload_result.get("generation") or {}).get("parsed"))
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": "judge_sft_one_step_smoke_result_v1",
        "status": "passed" if not failed else "failed",
        "checks": checks,
        "failed_checks": failed,
        "trainer_dir": str(trainer_dir.resolve()),
        "adapter_dir": str(adapter_dir.resolve()),
        "global_step": int(state.get("global_step") or 0),
        "grad_norms": grad_norms,
        "losses": losses,
        "adapter_tensor_count": len(tensors),
        "lora_B_tensor_count": len(lora_b),
        "evidence_scope": (
            "Proves runtime, one optimizer step, nonzero LoRA update, checkpoint, "
            "adapter reload, PASS/FAIL first-token selection, and continuous generation "
            "of one complete verdict-first JSON contract. It does not prove judge quality."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate one-step judge SFT smoke")
    parser.add_argument("--trainer-dir", type=Path, required=True)
    parser.add_argument("--adapter-reload", type=Path, required=True)
    parser.add_argument("--expected-model-id", required=True)
    parser.add_argument("--expected-example-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate_smoke(
        trainer_dir=args.trainer_dir,
        adapter_reload_path=args.adapter_reload,
        expected_model_id=args.expected_model_id,
        expected_example_id=args.expected_example_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
