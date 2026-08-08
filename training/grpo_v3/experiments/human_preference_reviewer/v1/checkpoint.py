"""Checkpoint contract for three heads and in-place PEFT LoRA adapters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


HEAD_NAMES = ("evidence_head", "answerability_head", "formality_head")


def _json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_checkpoint(
    reviewer: object,
    output_dir: str | Path,
    *,
    config: Mapping[str, Any],
    csv_sha256: str,
    split_sha256: str,
    parameter_audit: Mapping[str, Any],
    optimizer: object | None = None,
    scheduler: object | None = None,
    trainer_state: Mapping[str, Any] | None = None,
    processor: object | None = None,
) -> None:
    import torch

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    heads = {name: getattr(reviewer, name).state_dict() for name in HEAD_NAMES}
    torch.save(heads, output / "classification_heads.pt")
    adapter = {
        name: parameter.detach().cpu()
        for name, parameter in reviewer.state_dict().items()  # type: ignore[attr-defined]
        if ".lora_A." in name or ".lora_B." in name
    }
    if not adapter:
        raise ValueError("checkpoint has no LoRA adapter parameters")
    torch.save(adapter, output / "lora_adapter.pt")
    contract = {
        **dict(config),
        "csv_sha256": csv_sha256,
        "split_sha256": split_sha256,
        "label_mapping": {"1": 0, "2": 1, "3": 2},
        "head_names": list(HEAD_NAMES),
        "lora_parameter_names": sorted(adapter),
    }
    _json(output / "reviewer_v1_config.json", contract)
    _json(output / "parameter_audit.json", parameter_audit)
    torch.save({
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "trainer_state": dict(trainer_state or {}),
    }, output / "trainer_state.pt")
    if processor is not None:
        processor.save_pretrained(output / "processor")


def load_classification_heads(reviewer: object, checkpoint_dir: str | Path) -> None:
    import torch

    state = torch.load(Path(checkpoint_dir) / "classification_heads.pt", map_location="cpu", weights_only=True)
    if set(state) != set(HEAD_NAMES):
        raise ValueError("checkpoint classification-head contract mismatch")
    for name in HEAD_NAMES:
        getattr(reviewer, name).load_state_dict(state[name], strict=True)


def load_lora_adapter(reviewer: object, checkpoint_dir: str | Path) -> None:
    import torch

    state = torch.load(Path(checkpoint_dir) / "lora_adapter.pt", map_location="cpu", weights_only=True)
    current = reviewer.state_dict()  # type: ignore[attr-defined]
    missing = sorted(set(state) - set(current))
    if missing:
        raise ValueError(f"checkpoint LoRA names are absent from model: {missing}")
    current.update(state)
    reviewer.load_state_dict(current, strict=True)  # type: ignore[attr-defined]


def load_checkpoint_contract(checkpoint_dir: str | Path) -> dict[str, Any]:
    path = Path(checkpoint_dir) / "reviewer_v1_config.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("contract_version") != "human_preference_reviewer_absolute_v1":
        raise ValueError("incompatible Reviewer v1 checkpoint contract")
    if value.get("label_mapping") != {"1": 0, "2": 1, "3": 2}:
        raise ValueError("checkpoint label mapping drift")
    return value
