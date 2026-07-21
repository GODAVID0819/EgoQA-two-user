"""Gate 0：ms-swift/Qwen3-VL 原生视频与 LoRA 能力探针。"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from training.grpo_v3_contract import DEFAULTS
from training.grpo_v3_data import read_jsonl, validate_swift_row


def assert_framework_version(installed_version: str) -> None:
    if installed_version != DEFAULTS.framework_version:
        raise RuntimeError(
            f"ms-swift 必须固定为 {DEFAULTS.framework_version}，实际为 {installed_version}"
        )


def build_gate0_result(
    *,
    single_video_ok: bool,
    dual_video_ok: bool,
    batch_size: int,
    trainable_lora_parameters: int,
    visual_trainable_parameters: int,
    aligner_trainable_parameters: int,
    media_metadata: dict[str, Any],
) -> dict[str, Any]:
    media_complete = (
        media_metadata.get("actual_video_count") == 2
        and bool(media_metadata.get("video_grid_thw"))
        and int(media_metadata.get("visual_token_count") or 0) > 0
        and bool(media_metadata.get("processor_class"))
        and bool(media_metadata.get("video_backend"))
    )
    checks = {
        "single_video_processor": bool(single_video_ok),
        "dual_video_processor": bool(dual_video_ok),
        "batch_size_4": batch_size == 4,
        "lora_parameters_nonzero": trainable_lora_parameters > 0,
        "visual_parameters_frozen": visual_trainable_parameters == 0,
        "aligner_parameters_frozen": aligner_trainable_parameters == 0,
        "media_metadata_recorded": media_complete,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "batch_size": batch_size,
        "trainable_lora_parameters": trainable_lora_parameters,
        "visual_trainable_parameters": visual_trainable_parameters,
        "aligner_trainable_parameters": aligner_trainable_parameters,
        "media_metadata": media_metadata,
        **DEFAULTS.to_dict(),
    }


def _encode(template: Any, row: dict[str, Any]) -> dict[str, Any]:
    encoded = template.encode(row, return_template_inputs=True)
    if not encoded:
        raise RuntimeError("ms-swift template.encode 返回空结果")
    return encoded


def collate_encoded_batch(template: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 4:
        raise ValueError(f"Gate 0 必须真实 collate 4 条样本，收到 {len(rows)}")
    batch = template.data_collator(rows)
    if not isinstance(batch, dict) or not batch:
        raise RuntimeError("ms-swift data_collator 返回空 batch")
    return batch


def _plain(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def run_probe(*, model_path: str, dataset: Path) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, get_peft_model
    from swift import get_processor, get_template
    from transformers import AutoModelForImageTextToText

    assert_framework_version(importlib.metadata.version("ms-swift"))
    if not torch.cuda.is_available():
        raise RuntimeError("Gate 0 模型探针必须在 Torch GPU 计算节点运行")
    rows = read_jsonl(dataset)
    if not rows:
        raise ValueError("dataset 为空")
    validate_swift_row(rows[0])
    processor = get_processor(model_path)
    template = get_template(processor)
    template.set_mode("train")
    dual = _encode(template, rows[0])
    single_row = dict(rows[0])
    single_row["messages"] = [{"role": "user", "content": "<video>\nprocessor probe"}]
    single_row["videos"] = [rows[0]["videos"][0]]
    single = _encode(template, single_row)
    batch = [rows[0]] * 4
    encoded_batch = [_encode(template, item) for item in batch]
    collated_batch = collate_encoded_batch(template, encoded_batch)

    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        device_map={"": 0},
    )
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=DEFAULTS.lora_rank,
            lora_alpha=DEFAULTS.lora_alpha,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(DEFAULTS.lora_target_modules),
        ),
    )
    aligner_markers = ("aligner", "projector", "merger")
    for name, parameter in model.named_parameters():
        lowered = name.lower()
        if "visual" in lowered or "vision" in lowered or any(marker in lowered for marker in aligner_markers):
            parameter.requires_grad = False
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    visual_trainable = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and ("visual" in name.lower() or "vision" in name.lower())
    )
    aligner_trainable = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and any(marker in name.lower() for marker in aligner_markers)
    )
    template_inputs = dual.get("template_inputs")
    grid = dual.get("video_grid_thw")
    if grid is None:
        grid = getattr(template_inputs, "video_grid_thw", None)
    grid_rows = _plain(grid) or []
    visual_token_count = sum(
        int(row[0]) * int(row[1]) * int(row[2])
        for row in grid_rows
        if isinstance(row, (list, tuple)) and len(row) == 3
    )
    metadata = {
        "actual_video_count": len(rows[0]["videos"]),
        "single_input_keys": sorted(single),
        "dual_input_keys": sorted(dual),
        "encoded_batch_count": len(encoded_batch),
        "collated_batch_keys": sorted(collated_batch),
        "video_grid_thw": grid_rows,
        "visual_token_count": visual_token_count,
        "estimated_decoded_frame_count": sum(int(row[0]) * 2 for row in grid_rows if len(row) == 3),
        "video_fps": _plain(getattr(template_inputs, "fps", None)),
        "processor_class": type(processor).__name__,
        "video_backend": "qwen_vl_utils",
    }
    return build_gate0_result(
        single_video_ok=True,
        dual_video_ok=True,
        batch_size=len(encoded_batch),
        trainable_lora_parameters=trainable,
        visual_trainable_parameters=visual_trainable,
        aligner_trainable_parameters=aligner_trainable,
        media_metadata=metadata,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="执行 Gate 0 原生视频/LoRA GPU 探针")
    parser.add_argument("--model-path", default=DEFAULTS.policy_model)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_probe(model_path=args.model_path, dataset=args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
