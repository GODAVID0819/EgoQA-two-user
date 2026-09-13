"""Checkpoint contract for score-only heads and in-place PEFT LoRA adapters."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from .data import (
    CONTRACT_VERSION,
    IGNORED_ANNOTATION_COLUMNS,
    ORDINAL_TARGET_MAPPING,
    SCORE_COLUMNS,
    SUPERVISION_TYPE,
)


HEAD_NAMES = ("evidence_head", "answerability_head", "formality_head")
HEAD_BY_FIELD = {
    "evidence_quality": "evidence_head",
    "answerability": "answerability_head",
    "qa_formality": "formality_head",
}


def _validated_loss_weights(value: object) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != set(HEAD_BY_FIELD):
        raise ValueError("checkpoint score-loss weight fields drift")
    try:
        numeric = {name: float(weight) for name, weight in value.items()}
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint score-loss weights must be numeric") from error
    if any(not math.isfinite(weight) or weight <= 0.0 for weight in numeric.values()):
        raise ValueError("checkpoint score-loss weights must be finite and positive")
    if not math.isclose(sum(numeric.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("checkpoint score-loss weights must sum to one")
    return numeric


def checkpoint_head_names(active_heads: tuple[str, ...]) -> tuple[str, ...]:
    unsupported = [name for name in active_heads if name not in HEAD_BY_FIELD]
    if unsupported:
        raise ValueError(f"unsupported active head: {unsupported}")
    return tuple(HEAD_BY_FIELD[name] for name in active_heads)


def validate_checkpoint_runtime_contract(
    contract: Mapping[str, Any],
    *,
    stage: str,
    active_heads: tuple[str, ...],
    lora_enabled: bool,
    model_name_or_path: str,
    reviewer_config: Mapping[str, Any] | None = None,
    actual_lora_targets: tuple[str, ...] | None = None,
) -> None:
    expected = {
        "stage": stage,
        "active_heads": list(active_heads),
        "lora_enabled": lora_enabled,
        "model_name_or_path": model_name_or_path,
    }
    if reviewer_config is not None:
        for key in (
            "last_n_shared_blocks", "lora_target_modules", "lora_r", "lora_alpha",
            "lora_dropout", "lora_bias", "loss_weights",
        ):
            expected[key] = reviewer_config.get(key)
    if actual_lora_targets is not None:
        expected["actual_lora_targets"] = list(actual_lora_targets)
    mismatches = {
        key: {"checkpoint": contract.get(key), "runtime": value}
        for key, value in expected.items()
        if contract.get(key) != value
    }
    if mismatches:
        raise ValueError(f"checkpoint runtime contract mismatch: {mismatches}")


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
    active_heads: tuple[str, ...] = ("evidence_quality", "answerability", "qa_formality"),
    lora_enabled: bool = True,
) -> None:
    import torch

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    head_names = checkpoint_head_names(active_heads)
    heads = {name: getattr(reviewer, name).state_dict() for name in head_names}
    torch.save(heads, output / "ordinal_heads.pt")
    adapter = {
        name: parameter.detach().cpu()
        for name, parameter in reviewer.state_dict().items()  # type: ignore[attr-defined]
        if ".lora_A." in name or ".lora_B." in name
    }
    if lora_enabled and not adapter:
        raise ValueError("checkpoint has no LoRA adapter parameters")
    if lora_enabled:
        torch.save(adapter, output / "lora_adapter.pt")
    loss_weights = _validated_loss_weights(config.get("loss_weights"))
    contract = {
        **dict(config),
        "csv_sha256": csv_sha256,
        "split_sha256": split_sha256,
        "training_signal": SUPERVISION_TYPE,
        "head_type": "shared_score_ordered_thresholds",
        "num_score_levels": 3,
        "num_ordinal_thresholds": 2,
        "training_label_columns": list(SCORE_COLUMNS),
        "ignored_annotation_columns": list(IGNORED_ANNOTATION_COLUMNS),
        "ordinal_target_mapping": ORDINAL_TARGET_MAPPING,
        "loss_weights": loss_weights,
        "loss_aggregation": "normalized_weighted_sum",
        "active_heads": list(active_heads),
        "lora_enabled": lora_enabled,
        "head_names": list(head_names),
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


def load_ordinal_heads(reviewer: object, checkpoint_dir: str | Path) -> None:
    import torch

    state = torch.load(Path(checkpoint_dir) / "ordinal_heads.pt", map_location="cpu", weights_only=True)
    contract = load_checkpoint_contract(checkpoint_dir)
    head_names = checkpoint_head_names(tuple(contract.get("active_heads") or ()))
    if set(state) != set(head_names):
        raise ValueError("checkpoint ordinal-head contract mismatch")
    for name in head_names:
        getattr(reviewer, name).load_state_dict(state[name], strict=True)


def load_lora_adapter(reviewer: object, checkpoint_dir: str | Path) -> None:
    import torch

    state = torch.load(Path(checkpoint_dir) / "lora_adapter.pt", map_location="cpu", weights_only=True)
    current = reviewer.state_dict()  # type: ignore[attr-defined]
    current_lora = {
        name for name in current if ".lora_A." in name or ".lora_B." in name
    }
    contract_lora = set(load_checkpoint_contract(checkpoint_dir).get("lora_parameter_names") or [])
    if set(state) != current_lora or set(state) != contract_lora:
        raise ValueError(
            "checkpoint LoRA parameter set mismatch; "
            f"missing_from_file={sorted(current_lora - set(state))}, "
            f"unexpected_in_file={sorted(set(state) - current_lora)}, "
            f"contract_difference={sorted(set(state) ^ contract_lora)}"
        )
    current.update(state)
    reviewer.load_state_dict(current, strict=True)  # type: ignore[attr-defined]


def load_checkpoint_contract(checkpoint_dir: str | Path) -> dict[str, Any]:
    path = Path(checkpoint_dir) / "reviewer_v1_config.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("contract_version") != CONTRACT_VERSION:
        raise ValueError("incompatible Reviewer v1 checkpoint contract")
    if value.get("training_signal") != SUPERVISION_TYPE:
        raise ValueError("checkpoint training signal drift")
    if value.get("head_type") != "shared_score_ordered_thresholds":
        raise ValueError("checkpoint ordinal head type drift")
    if value.get("num_score_levels") != 3 or value.get("num_ordinal_thresholds") != 2:
        raise ValueError("checkpoint ordinal output shape drift")
    if value.get("training_label_columns") != list(SCORE_COLUMNS):
        raise ValueError("checkpoint training label columns drift")
    if value.get("ignored_annotation_columns") != list(IGNORED_ANNOTATION_COLUMNS):
        raise ValueError("checkpoint ignored annotation columns drift")
    if value.get("ordinal_target_mapping") != ORDINAL_TARGET_MAPPING:
        raise ValueError("checkpoint ordinal target mapping drift")
    _validated_loss_weights(value.get("loss_weights"))
    if value.get("loss_aggregation") != "normalized_weighted_sum":
        raise ValueError("checkpoint score-loss aggregation drift")
    training_ids = value.get("training_evidence_ids")
    if (
        not isinstance(training_ids, list)
        or not training_ids
        or any(not str(evidence_id).strip() for evidence_id in training_ids)
        or len(training_ids) != len(set(training_ids))
    ):
        raise ValueError("checkpoint training evidence IDs drift")
    if value.get("training_split_mode") not in {
        "train_validation_test", "train_validation", "train_only", "smoke"
    }:
        raise ValueError("checkpoint training split mode drift")
    return value
