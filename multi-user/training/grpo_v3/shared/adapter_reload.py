"""定位并真实重载 ms-swift 产出的 LoRA adapter。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _step(path: Path) -> int:
    match = re.search(r"checkpoint-(\d+)", path.as_posix())
    return int(match.group(1)) if match else -1


def discover_adapter_dir(root: Path) -> Path:
    candidates = []
    for config in root.rglob("adapter_config.json"):
        parent = config.parent
        if (parent / "adapter_model.safetensors").is_file() or (parent / "adapter_model.bin").is_file():
            candidates.append(parent)
    if not candidates:
        raise FileNotFoundError(f"{root} 下没有完整 LoRA adapter")
    return max(candidates, key=lambda path: (_step(path), path.stat().st_mtime_ns))


def reload_adapter(*, model_path: str, adapter_dir: Path, processor_output: Path | None = None) -> dict:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("adapter 重载必须在 Torch GPU 计算节点运行")
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        device_map={"": 0},
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    lora_parameters = sum(
        parameter.numel() for name, parameter in model.named_parameters() if "lora_" in name.lower()
    )
    if lora_parameters <= 0:
        raise RuntimeError("adapter 已加载但 LoRA 参数为 0")
    processor_output = processor_output or adapter_dir.parent / "processor"
    processor_output.mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(processor_output)
    reloaded_processor = AutoProcessor.from_pretrained(processor_output, local_files_only=True)
    return {
        "status": "passed",
        "model_path": model_path,
        "adapter_dir": str(adapter_dir.resolve()),
        "lora_parameters": lora_parameters,
        "processor_class": type(processor).__name__,
        "reloaded_processor_class": type(reloaded_processor).__name__,
        "processor_reloaded": True,
        "model_class": type(model).__name__,
        "processor_dir": str(processor_output.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="定位并重载 GRPO v3 LoRA adapter")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--search-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    adapter_dir = discover_adapter_dir(args.search_root)
    result = reload_adapter(
        model_path=args.model_path,
        adapter_dir=adapter_dir,
        processor_output=args.output.parent / "processor",
    )
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
