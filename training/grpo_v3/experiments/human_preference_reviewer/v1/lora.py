"""Exact LoRA placement and parameter-freezing audits."""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

from .config import ReviewerV1Config
from .modeling import head_module_names


def locate_shared_language_layers(model: object) -> tuple[str, Sequence[object]]:
    candidates = (
        ("model.language_model.layers", ("model", "language_model", "layers")),
        ("language_model.layers", ("language_model", "layers")),
    )
    for path, attributes in candidates:
        current: Any = model
        for attribute in attributes:
            current = getattr(current, attribute, None)
            if current is None:
                break
        if current is not None and hasattr(current, "__len__") and len(current) > 0:
            return path, current
    raise ValueError("unable to locate shared language Transformer layers")


def expected_lora_targets(
    model: object,
    *,
    last_n: int,
    projections: Sequence[str],
    expected_layer_count: int | None = None,
) -> tuple[str, ...]:
    path, layers = locate_shared_language_layers(model)
    if expected_layer_count is not None and len(layers) != expected_layer_count:
        raise ValueError(
            f"Reviewer v1 expected {expected_layer_count} shared language blocks; found {len(layers)}"
        )
    if last_n <= 0 or len(layers) < last_n:
        raise ValueError("invalid number of final shared blocks")
    targets: list[str] = []
    for index in range(len(layers) - last_n, len(layers)):
        attention = getattr(layers[index], "self_attn", None)
        if attention is None:
            raise ValueError(f"{path}.{index} has no self_attn")
        for projection in projections:
            if getattr(attention, projection, None) is None:
                raise ValueError(f"{path}.{index}.self_attn has no {projection}")
            targets.append(f"{path}.{index}.self_attn.{projection}")
    return tuple(targets)


def target_layer_indices(model: object, *, last_n: int) -> tuple[int, ...]:
    _, layers = locate_shared_language_layers(model)
    if len(layers) < last_n:
        raise ValueError("model has fewer shared layers than requested")
    return tuple(range(len(layers) - last_n, len(layers)))


def inject_reviewer_lora(model: object, config: ReviewerV1Config) -> tuple[object, tuple[str, ...]]:
    """Freeze base weights and inject PEFT adapters without wrapping away `.model`."""

    expected = expected_lora_targets(
        model,
        last_n=config.last_n_shared_blocks,
        projections=config.lora_target_modules,
        expected_layer_count=config.expected_shared_block_count,
    )
    indices = target_layer_indices(model, last_n=config.last_n_shared_blocks)
    for parameter in model.parameters():  # type: ignore[attr-defined]
        parameter.requires_grad = False
    try:
        from peft import LoraConfig, inject_adapter_in_model
    except ImportError as error:
        raise RuntimeError("PEFT is required to inject Reviewer v1 LoRA") from error
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        target_modules=list(config.lora_target_modules),
        layers_to_transform=list(indices),
        layers_pattern="layers",
    )
    injected = inject_adapter_in_model(lora_config, model, adapter_name="default")
    found = {
        name.rsplit(".lora_A", 1)[0].rsplit(".lora_B", 1)[0]
        for name, _ in injected.named_parameters()
        if ".lora_A." in name or ".lora_B." in name
    }
    missing = [target for target in expected if not any(name.endswith(target) for name in found)]
    extra = [name for name in found if not any(name.endswith(target) for target in expected)]
    if missing or extra:
        raise RuntimeError(f"LoRA target mismatch; missing={missing}, extra={extra}")
    return injected, expected


def audit_trainable_parameter_names(
    names: Iterable[str],
    *,
    expected_layer_indices: Sequence[int],
    active_heads: Sequence[str] = ("evidence_quality", "answerability", "qa_formality"),
    lora_enabled: bool = True,
) -> dict[str, object]:
    names = tuple(str(name) for name in names)
    allowed_head_prefixes = tuple(f"{name}." for name in head_module_names(active_heads))
    layer_tokens = tuple(f".layers.{index}." for index in expected_layer_indices)
    unexpected = []
    lora_names = []
    head_names = []
    for name in names:
        if name.startswith(allowed_head_prefixes):
            head_names.append(name)
            continue
        is_lora = ".lora_A." in name or ".lora_B." in name
        if lora_enabled and is_lora and any(token in name for token in layer_tokens) and re.search(r"\.(q_proj|v_proj)\.", name):
            lora_names.append(name)
            continue
        unexpected.append(name)
    if unexpected:
        raise ValueError(f"unexpected trainable parameters: {unexpected}")
    return {
        "trainable_parameter_names": list(names),
        "head_parameter_names": head_names,
        "lora_parameter_names": lora_names,
        "unexpected_trainable_names": unexpected,
    }


def parameter_audit(
    reviewer: object,
    *,
    expected_layer_indices: Sequence[int],
    active_heads: Sequence[str] = ("evidence_quality", "answerability", "qa_formality"),
    lora_enabled: bool = True,
) -> dict[str, object]:
    total = 0
    trainable = 0
    names = []
    head_count = 0
    head_counts = {"evidence_head": 0, "answerability_head": 0, "formality_head": 0}
    lora_count = 0
    for name, parameter in reviewer.named_parameters():  # type: ignore[attr-defined]
        count = int(parameter.numel())
        total += count
        if not parameter.requires_grad:
            continue
        trainable += count
        names.append(name)
        if name.startswith(("evidence_head.", "answerability_head.", "formality_head.")):
            head_count += count
            head_counts[name.split(".", 1)[0]] += count
        if ".lora_A." in name or ".lora_B." in name:
            lora_count += count
    result = audit_trainable_parameter_names(
        names,
        expected_layer_indices=expected_layer_indices,
        active_heads=active_heads,
        lora_enabled=lora_enabled,
    )
    result.update({
        "total_parameter_count": total,
        "trainable_parameter_count": trainable,
        "head_parameter_count": head_count,
        "head_parameter_counts": head_counts,
        "lora_parameter_count": lora_count,
    })
    return result
