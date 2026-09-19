"""Reload a judge adapter and exercise binary-first continuous JSON generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .collator import _apply_chat_template, _assert_thinking_disabled
from .contracts import (
    VERDICT_ASSISTANT_PREFIX,
    Verdict,
    parse_complete_reason_fix_output,
)
from .inference import generate_complete_judge_output
from .loss import resolve_verdict_token_ids


def _model_class() -> Any:
    import transformers

    result = getattr(transformers, "AutoModelForMultimodalLM", None)
    if result is None:
        result = getattr(transformers, "AutoModelForImageTextToText", None)
    if result is None:
        raise RuntimeError("no compatible Transformers multimodal auto-model class")
    return result


def reload_and_probe(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        args.adapter_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = _model_class().from_pretrained(
        args.model_id,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(
        model,
        str(args.adapter_dir),
        is_trainable=False,
    )
    model.eval()
    lora_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if "lora_" in name.lower()
    )
    if lora_parameters <= 0:
        raise RuntimeError("reloaded adapter exposes no LoRA parameters")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Smoke probe only. Judge whether this sentence is grammatical: "
                        "Where did I leave the mug? Return exactly one JSON object with "
                        "verdict first, followed by reason and fix. Use lowercase pass or "
                        "fail. For pass, reason and fix must both be null. For fail, both "
                        "must be non-empty strings."
                    ),
                }
            ],
        }
    ]
    rendered = _apply_chat_template(processor, messages)
    _assert_thinking_disabled(rendered)
    rendered += VERDICT_ASSISTANT_PREFIX
    batch = processor(text=[rendered], return_tensors="pt")
    model_device = next(model.parameters()).device
    model_inputs = {
        key: value.to(model_device)
        for key, value in batch.items()
        if isinstance(value, torch.Tensor)
    }
    token_ids = resolve_verdict_token_ids(getattr(processor, "tokenizer", processor))
    with torch.inference_mode():
        generation = generate_complete_judge_output(
            model,
            getattr(processor, "tokenizer", processor),
            model_inputs,
            verdict_token_ids=token_ids,
            max_new_tokens=args.max_new_tokens,
        )
    parsed_output = parse_complete_reason_fix_output(generation.text)
    decision = generation.decision
    if parsed_output["verdict"] != decision.verdict.value:
        raise RuntimeError("generated JSON verdict differs from the locked verdict token")
    return {
        "schema_version": "judge_sft_adapter_reload_v2",
        "status": "passed",
        "model_id": args.model_id,
        "adapter_dir": str(args.adapter_dir.resolve()),
        "model_class": type(model).__name__,
        "processor_class": type(processor).__name__,
        "lora_parameters": lora_parameters,
        "generation_contract": "binary-first token lock followed by continuous JSON generation",
        "verdict_token_ids": {
            verdict.value: token_ids[verdict]
            for verdict in (Verdict.FAIL, Verdict.PASS)
        },
        "decision": {
            "verdict": decision.verdict.value,
            "pass_probability_within_binary_pair": decision.pass_probability,
            "pass_minus_fail_logit": decision.pass_minus_fail_logit,
            "unrestricted_top_token_id": decision.unrestricted_top_token_id,
            "unrestricted_top_is_verdict": decision.unrestricted_top_is_verdict,
            "locked_json_prefix": decision.locked_json_prefix,
        },
        "generation": {
            "full_contract_valid": True,
            "text": generation.text,
            "parsed": parsed_output,
            "generated_token_count": len(generation.generated_token_ids),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Reload judge LoRA and probe verdict logits")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    args = parser.parse_args()
    result = reload_and_probe(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
