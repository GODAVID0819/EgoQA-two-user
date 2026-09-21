"""Train a no-head Qwen3-VL judge from normalized PASS/FAIL invocations."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
from pathlib import Path
from typing import Any

from .collator import (
    DEFAULT_IMAGE_CONTEXT_TARGET_FRACTION,
    DEFAULT_IMAGE_ITEM_TOKEN_OVERHEAD,
    DEFAULT_IMAGE_TEXT_TOKEN_RESERVE,
    JudgeFrameCollator,
    QWEN_VISION_TOKEN_PIXEL_AREA,
    adaptive_image_max_pixels,
)
from .contracts import (
    DEFAULTS,
    JudgeTask,
    Verdict,
    validate_task_weights,
)
from .data import (
    JudgeDataset,
    load_normalized_manifest,
    manifest_summary,
)
from .loss import (
    class_weights_by_task,
    resolve_verdict_token_ids,
    task_sampling_scales,
)
from .trainer import audit_tensor_parallel_sampler, build_verdict_trainer_class


VISION_MARKERS = ("visual", "vision_tower", "vision_model")
ALIGNER_MARKERS = ("aligner", "projector", "merger")
DECODER_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(?P<index>\d+)(?:\.|$)")
LORA_TP_FACTOR_BY_TARGET = {
    # Colwise base projections shard their output dimension, hence LoRA-B.
    "q_proj": "lora_B",
    "k_proj": "lora_B",
    "v_proj": "lora_B",
    "gate_proj": "lora_B",
    "up_proj": "lora_B",
    # Rowwise base projections shard their input dimension, hence LoRA-A.
    "o_proj": "lora_A",
    "down_proj": "lora_A",
}
LORA_TP_STYLE_BY_TARGET = {
    target: "colwise" if factor == "lora_B" else "rowwise"
    for target, factor in LORA_TP_FACTOR_BY_TARGET.items()
}


def _matches_any(name: str, markers: tuple[str, ...]) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in markers)


def _parameter_count(parameter: Any) -> int:
    """Count logical parameters, including distributed tensor parameters."""

    return int(getattr(parameter, "ds_numel", parameter.numel()))


def freeze_multimodal_adapters(
    model: Any,
    *,
    freeze_vision: bool,
    freeze_aligner: bool,
) -> dict[str, int]:
    for name, parameter in model.named_parameters():
        if freeze_vision and _matches_any(name, VISION_MARKERS):
            parameter.requires_grad = False
        if freeze_aligner and _matches_any(name, ALIGNER_MARKERS):
            parameter.requires_grad = False
    counts = {
        "total": sum(_parameter_count(parameter) for parameter in model.parameters()),
        "trainable": sum(
            _parameter_count(parameter)
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "vision_trainable": sum(
            _parameter_count(parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and _matches_any(name, VISION_MARKERS)
        ),
        "aligner_trainable": sum(
            _parameter_count(parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and _matches_any(name, ALIGNER_MARKERS)
        ),
        "lora_trainable": sum(
            _parameter_count(parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and "lora_" in name.lower()
        ),
    }
    if counts["trainable"] <= 0 or counts["lora_trainable"] <= 0:
        raise RuntimeError(f"LoRA injection produced no trainable parameters: {counts}")
    if freeze_vision and counts["vision_trainable"]:
        raise RuntimeError(f"vision parameters are unexpectedly trainable: {counts}")
    if freeze_aligner and counts["aligner_trainable"]:
        raise RuntimeError(f"aligner parameters are unexpectedly trainable: {counts}")
    unexpected = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" not in name.lower()
    ]
    if unexpected:
        raise RuntimeError(
            "default judge plan permits only LoRA parameters to train; unexpected="
            + ",".join(unexpected[:20])
        )
    return counts


def audit_language_lora_targets(
    model: Any,
    requested_targets: list[str] | tuple[str, ...],
) -> dict[str, int]:
    """Require every requested adapter family to exist on the language side.

    PEFT matches target names by suffix and can therefore also inject adapters
    into similarly named vision modules. Those modules are frozen separately;
    this audit proves that each requested attention/MLP family also has
    trainable language-side LoRA parameters.
    """

    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and "lora_" in name.lower()
        and not _matches_any(name, VISION_MARKERS)
        and not _matches_any(name, ALIGNER_MARKERS)
    ]
    counts = {
        target: sum(
            _parameter_count(parameter)
            for name, parameter in trainable
            if f".{target}." in name
        )
        for target in requested_targets
    }
    missing = [target for target, count in counts.items() if count <= 0]
    if missing:
        raise RuntimeError(
            "requested language LoRA targets produced no trainable parameters: "
            + ",".join(missing)
        )
    return counts


def audit_trainable_lora_dtypes(model: Any, expected_dtype: Any) -> dict[str, Any]:
    """Require every trainable adapter parameter to remain in the model dtype."""

    dtype_counts: dict[str, int] = {}
    mismatches: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or "lora_" not in name.lower():
            continue
        label = str(parameter.dtype)
        dtype_counts[label] = dtype_counts.get(label, 0) + _parameter_count(parameter)
        if parameter.dtype != expected_dtype:
            mismatches.append(f"{name}:{parameter.dtype}")
    if not dtype_counts:
        raise RuntimeError("LoRA dtype audit found no trainable adapter parameters")
    if mismatches:
        raise RuntimeError(
            "trainable LoRA parameters were promoted away from BF16: "
            + ",".join(mismatches[:20])
        )
    return {
        "expected": str(expected_dtype),
        "parameter_counts_by_dtype": dtype_counts,
        "autocast_adapter_dtype": False,
    }


def audit_trainable_lora_layers(
    model: Any,
    expected_layer_indices: list[int] | tuple[int, ...],
) -> dict[str, Any]:
    """Require language LoRA parameters on exactly the selected decoder layers."""

    expected = set(int(index) for index in expected_layer_indices)
    observed_counts: dict[int, int] = {}
    unclassified: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or "lora_" not in name.lower():
            continue
        if _matches_any(name, VISION_MARKERS) or _matches_any(name, ALIGNER_MARKERS):
            continue
        matches = {int(value) for value in DECODER_LAYER_PATTERN.findall(name)}
        if len(matches) != 1:
            unclassified.append(name)
            continue
        index = next(iter(matches))
        observed_counts[index] = observed_counts.get(index, 0) + _parameter_count(
            parameter
        )
    observed = set(observed_counts)
    if unclassified:
        raise RuntimeError(
            "could not assign trainable language LoRA parameters to one decoder layer: "
            + ",".join(unclassified[:20])
        )
    if observed != expected:
        raise RuntimeError(
            "trainable LoRA decoder layers differ from the requested selection: "
            f"expected={sorted(expected)} observed={sorted(observed)}"
        )
    return {
        "layer_indices": sorted(observed),
        "parameter_counts_by_layer": {
            str(index): observed_counts[index] for index in sorted(observed_counts)
        },
    }


def audit_lora_tensor_parallel_materialization(model: Any) -> dict[str, Any]:
    """Require TP-sharded LoRA factors on every selected projection family."""

    from torch.distributed.tensor import DTensor

    sharded_counts = {target: 0 for target in LORA_TP_FACTOR_BY_TARGET}
    unsharded: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        for target, factor in LORA_TP_FACTOR_BY_TARGET.items():
            if f".{target}." not in name or f".{factor}." not in name:
                continue
            if isinstance(parameter, DTensor):
                sharded_counts[target] += _parameter_count(parameter)
            else:
                unsharded.append(name)
    missing = [target for target, count in sharded_counts.items() if count <= 0]
    if missing or unsharded:
        raise RuntimeError(
            "PEFT did not materialize TP-aware LoRA factors: "
            f"missing_targets={missing} unsharded={unsharded[:20]}. "
            "The local PEFT/Transformers TP compatibility sharder did not apply."
        )
    return {
        "policy": "colwise LoRA-B and rowwise LoRA-A factors are DTensors",
        "sharded_parameter_counts_by_target": sharded_counts,
    }


def install_frozen_prefix_input_guard(
    model: Any,
    *,
    first_trainable_layer: int,
) -> dict[str, Any]:
    """Prove at runtime that the frozen lower stack is outside autograd."""

    matches: list[tuple[str, Any]] = []
    for name, module in model.named_modules():
        match = DECODER_LAYER_PATTERN.search(name)
        if (
            match is not None
            and match.end() == len(name)
            and int(match.group("index")) == first_trainable_layer
            and not _matches_any(name, VISION_MARKERS)
        ):
            matches.append((name, module))
    if len(matches) != 1:
        raise RuntimeError(
            "could not uniquely locate the first trainable decoder layer: "
            f"index={first_trainable_layer} matches={[name for name, _ in matches]}"
        )
    module_name, module = matches[0]

    def _guard(_module: Any, positional_args: tuple[Any, ...]) -> None:
        if not positional_args:
            raise RuntimeError("first trainable decoder layer received no hidden state")
        hidden_states = positional_args[0]
        if bool(getattr(hidden_states, "requires_grad", False)):
            raise RuntimeError(
                "frozen decoder prefix leaked into autograd; layer "
                f"{first_trainable_layer} input unexpectedly requires gradients"
            )

    handle = module.register_forward_pre_hook(_guard)
    # Keep an explicit reference for the entire training lifetime.
    model._judge_frozen_prefix_guard_handle = handle
    return {
        "first_trainable_layer": first_trainable_layer,
        "guarded_module": module_name,
        "required_input_requires_grad": False,
    }


def configure_safe_tensor_parallel_plan(config: Any) -> dict[str, Any]:
    """Use Qwen's TP plan while replicating its currently unsafe hybrid layers."""

    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        try:
            text_config = get_text_config(decoder=True)
        except TypeError:
            text_config = get_text_config()
    else:
        text_config = getattr(config, "text_config", config)
    original = dict(getattr(text_config, "base_model_tp_plan", None) or {})
    if not original:
        raise RuntimeError("Qwen text config does not define base_model_tp_plan")
    safe_plan = {
        name: style
        for name, style in original.items()
        if ".linear_attn." not in name
    }
    required_suffixes = (
        ".self_attn.q_proj",
        ".self_attn.k_proj",
        ".self_attn.v_proj",
        ".self_attn.o_proj",
        ".mlp.gate_proj",
        ".mlp.up_proj",
        ".mlp.down_proj",
    )
    missing = [
        suffix
        for suffix in required_suffixes
        if not any(name.endswith(suffix) for name in safe_plan)
    ]
    if missing:
        raise RuntimeError(
            "Qwen TP plan is missing required attention/MLP projections: "
            + ",".join(missing)
        )
    text_config.base_model_tp_plan = safe_plan
    return {
        "policy": "shard dense MLP and full-attention projections; replicate hybrid linear-attention projections",
        "active_plan": safe_plan,
        "replicated_for_compatibility": sorted(set(original) - set(safe_plan)),
    }


def audit_tensor_parallel_materialization(
    model: Any,
    requested_targets: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """Prove that every trainable projection family was actually TP-sharded."""

    from torch.distributed.tensor import DTensor

    sharded_counts = {target: 0 for target in requested_targets}
    accidentally_sharded_hybrid: list[str] = []
    for name, parameter in model.named_parameters():
        if not isinstance(parameter, DTensor):
            continue
        if ".linear_attn." in name:
            accidentally_sharded_hybrid.append(name)
        for target in requested_targets:
            if f".{target}.weight" in name:
                sharded_counts[target] += _parameter_count(parameter)
    missing = [target for target, count in sharded_counts.items() if count <= 0]
    if missing:
        raise RuntimeError(
            "requested TP projection families were not materialized as DTensors: "
            + ",".join(missing)
        )
    if accidentally_sharded_hybrid:
        raise RuntimeError(
            "safe TP plan unexpectedly sharded Qwen hybrid linear attention: "
            + ",".join(accidentally_sharded_hybrid[:20])
        )
    return {
        "sharded_parameter_counts_by_target": sharded_counts,
        "hybrid_linear_attention_is_replicated": True,
    }


def ensure_tensor_parallel_metadata(
    model: Any,
    *,
    requested_tp_size: int,
) -> dict[str, Any]:
    """Validate TP metadata and backport the Transformers 5.16 size marker.

    Transformers 5.16 materializes the device mesh and DTensor parameters but
    does not assign ``model._tp_size``. Trainer uses that marker to distinguish
    TP from DDP. This helper is deliberately called only after the DTensor
    materialization audit succeeds.
    """

    observed = getattr(model, "tp_size", None)
    if observed is None:
        observed = getattr(model, "_tp_size", None)
    if observed is not None:
        observed = int(observed)
        if observed != requested_tp_size:
            raise RuntimeError(
                "loaded model tensor-parallel metadata disagrees with the request: "
                f"requested={requested_tp_size} observed={observed}"
            )
        return {
            "requested_tp_size": requested_tp_size,
            "observed_tp_size": observed,
            "metadata_repaired": False,
        }

    distributed_config = getattr(
        getattr(model, "config", None),
        "distributed_config",
        None,
    )
    configured_tp_size = getattr(distributed_config, "tp_size", None)
    device_mesh = getattr(model, "_device_mesh", None)
    if device_mesh is None:
        raise RuntimeError(
            "TP-sharded parameters exist but the loaded model has no device mesh"
        )
    tp_mesh = tensor_parallel_mesh(device_mesh)
    mesh_tp_size = int(tp_mesh.size())
    if (
        int(configured_tp_size or 0) != requested_tp_size
        or mesh_tp_size != requested_tp_size
    ):
        raise RuntimeError(
            "loaded model tensor-parallel mesh disagrees with the request: "
            f"requested={requested_tp_size} configured={configured_tp_size} "
            f"mesh={mesh_tp_size}"
        )

    # Compatibility backport of the assignment added by Transformers 5.17.
    # Without it, Trainer treats this already-sharded model as DDP.
    model._tp_size = requested_tp_size
    return {
        "requested_tp_size": requested_tp_size,
        "observed_tp_size": requested_tp_size,
        "metadata_repaired": True,
    }


def tensor_parallel_mesh(device_mesh: Any) -> Any:
    """Return the TP dimension from a Transformers device mesh."""

    tp_mesh = device_mesh
    if int(getattr(device_mesh, "ndim", 1)) > 1:
        try:
            tp_mesh = device_mesh["tp"]
        except Exception as error:
            raise RuntimeError(
                "loaded multidimensional device mesh has no 'tp' dimension"
            ) from error
    return tp_mesh


def materialize_lora_tensor_parallelism(
    model: Any,
    *,
    tp_mesh: Any,
    tp_size: int,
    requested_targets: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """Shard LoRA factors for the Transformers 5.16 DTensor TP backend.

    Released PEFT 0.21 only discovers the older per-layer Transformers TP
    markers. This applies the same factor sharding used by newer PEFT and also
    supplies PEFT's checkpoint-gathering metadata.
    """

    from peft.tuners.lora.layer import LoraLayer
    from peft.utils.integrations import TpInfo
    from torch.distributed.tensor import DTensor
    from transformers.distributed.tensor_parallel import ALL_PARALLEL_STYLES

    get_base_model = getattr(model, "get_base_model", None)
    base_model = get_base_model() if callable(get_base_model) else model
    base_tp_plan = getattr(base_model, "tp_plan", None) or getattr(
        base_model,
        "_tp_plan",
        None,
    )
    if not base_tp_plan:
        raise RuntimeError("PEFT-wrapped model no longer exposes the active TP plan")
    # Trainer and PEFT's checkpoint gatherer inspect the outer model.
    model._tp_size = tp_size
    model._device_mesh = tp_mesh
    model._tp_plan = base_tp_plan

    requested = set(requested_targets)
    unsupported = requested - set(LORA_TP_FACTOR_BY_TARGET)
    if unsupported:
        raise RuntimeError(
            "no TP-aware LoRA factor policy for targets: "
            + ",".join(sorted(unsupported))
        )
    applied = {target: 0 for target in requested}
    already_sharded = {target: 0 for target in requested}
    seen = {target: 0 for target in requested}
    for name, module in model.named_modules():
        if not isinstance(module, LoraLayer):
            continue
        target = name.rsplit(".", 1)[-1]
        if target not in requested:
            continue
        seen[target] += 1
        factor_name = LORA_TP_FACTOR_BY_TARGET[target]
        style_name = LORA_TP_STYLE_BY_TARGET[target]
        factor_modules = getattr(module, factor_name)
        module_tp_plan: dict[str, str] = {}
        normalized_name = name.removeprefix("base_model.model.")
        generic_name = re.sub(
            r"\.\d+(\.|$)",
            lambda match: ".*" + match.group(1),
            normalized_name,
        )
        for adapter_name, factor_module in factor_modules.items():
            module_name = f"{normalized_name}.{factor_name}.{adapter_name}"
            parameters = list(factor_module.named_parameters(recurse=False))
            if not parameters:
                raise RuntimeError(f"LoRA factor has no parameters: {module_name}")
            if all(isinstance(parameter, DTensor) for _, parameter in parameters):
                already_sharded[target] += 1
            elif any(isinstance(parameter, DTensor) for _, parameter in parameters):
                raise RuntimeError(f"partially sharded LoRA factor: {module_name}")
            else:
                style = ALL_PARALLEL_STYLES[style_name]
                for parameter_name, _ in parameters:
                    style.validate_param(
                        factor_module,
                        parameter_name,
                        tp_mesh,
                        parameter_name=f"{module_name}.{parameter_name}",
                    )
                    style.shard_param(factor_module, parameter_name, tp_mesh)
                style.install_forward(factor_module, tp_mesh)
                applied[target] += 1
            module_tp_plan[
                f"{generic_name}.{factor_name}.{adapter_name}.weight"
            ] = style_name
        module._tp_info = TpInfo(
            tp_plan=module_tp_plan,
            device_mesh=tp_mesh,
            tp_size=tp_size,
        )
    missing = [target for target, count in seen.items() if count <= 0]
    if missing:
        raise RuntimeError(
            "LoRA injection omitted requested TP targets: " + ",".join(missing)
        )
    return {
        "policy": "DTensor-shard colwise LoRA-B and rowwise LoRA-A",
        "applied_module_counts_by_target": applied,
        "already_sharded_module_counts_by_target": already_sharded,
        "seen_module_counts_by_target": seen,
    }


def synchronize_lora_initialization(
    model: Any,
    *,
    tp_mesh: Any,
    requested_targets: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """Broadcast and exactly audit LoRA initialization before TP sharding.

    Each torchrun process constructs PEFT adapters independently. Even with a
    common seed, rank-specific RNG consumption in a future dependency could
    silently diverge replicated LoRA factors. Rank zero is therefore the
    authoritative initialization source for every still-local LoRA parameter;
    sharded factors are derived from those identical full tensors afterwards.
    """

    import torch
    from peft.tuners.lora.layer import LoraLayer
    from torch.distributed.tensor import DTensor

    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError("LoRA TP initialization sync requires a process group")
    process_group = tp_mesh.get_group()
    group_world_size = torch.distributed.get_world_size(process_group)
    if group_world_size != int(tp_mesh.size()):
        raise RuntimeError(
            "LoRA TP initialization group disagrees with the device mesh: "
            f"group={group_world_size} mesh={int(tp_mesh.size())}"
        )
    source_rank = torch.distributed.get_global_rank(process_group, 0)
    requested = set(requested_targets)
    seen = {target: 0 for target in requested}
    broadcast_counts = {target: 0 for target in requested}
    parameter_count = 0

    for module_name, module in model.named_modules():
        if not isinstance(module, LoraLayer):
            continue
        target = module_name.rsplit(".", 1)[-1]
        if target not in requested:
            continue
        seen[target] += 1
        for factor_name in ("lora_A", "lora_B"):
            factor_modules = getattr(module, factor_name)
            for adapter_name, factor_module in factor_modules.items():
                for parameter_name, parameter in factor_module.named_parameters(
                    recurse=False
                ):
                    full_name = (
                        f"{module_name}.{factor_name}.{adapter_name}.{parameter_name}"
                    )
                    if isinstance(parameter, DTensor):
                        raise RuntimeError(
                            "LoRA parameter was sharded before initialization sync: "
                            + full_name
                        )
                    with torch.no_grad():
                        torch.distributed.broadcast(
                            parameter,
                            src=source_rank,
                            group=process_group,
                        )
                    gathered = [
                        torch.empty_like(parameter) for _ in range(group_world_size)
                    ]
                    torch.distributed.all_gather(
                        gathered,
                        parameter.detach(),
                        group=process_group,
                    )
                    if any(
                        not torch.equal(gathered[0], candidate)
                        for candidate in gathered[1:]
                    ):
                        raise RuntimeError(
                            "LoRA initialization remained inconsistent after broadcast: "
                            + full_name
                        )
                    broadcast_counts[target] += parameter.numel()
                    parameter_count += 1

    missing = [target for target, count in seen.items() if count <= 0]
    if missing:
        raise RuntimeError(
            "LoRA initialization sync omitted requested targets: "
            + ",".join(sorted(missing))
        )
    if parameter_count <= 0:
        raise RuntimeError("LoRA initialization sync found no adapter parameters")
    return {
        "policy": (
            "reset every rank to the common seed, then broadcast every full "
            "LoRA factor from TP rank zero before sharding"
        ),
        "source_global_rank": source_rank,
        "tp_group_world_size": group_world_size,
        "audited_parameter_tensors": parameter_count,
        "broadcast_parameter_counts_by_target": broadcast_counts,
        "exact_post_broadcast_equality": True,
    }


def load_model_and_processor(
    args: argparse.Namespace,
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoProcessor, DistributedConfig, set_seed

    processor = AutoProcessor.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    config = AutoConfig.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        try:
            text_config = get_text_config(decoder=True)
        except TypeError:
            text_config = get_text_config()
    else:
        text_config = getattr(config, "text_config", config)
    total_decoder_layers = int(getattr(text_config, "num_hidden_layers", 0))
    if not 0 < args.trainable_decoder_layers <= total_decoder_layers:
        raise ValueError(
            "trainable_decoder_layers must be between one and the model depth: "
            f"requested={args.trainable_decoder_layers} total={total_decoder_layers}"
        )
    first_trainable_layer = total_decoder_layers - args.trainable_decoder_layers
    trainable_layer_indices = list(
        range(first_trainable_layer, total_decoder_layers)
    )
    tp_plan_audit = configure_safe_tensor_parallel_plan(config)
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
        "low_cpu_mem_usage": True,
        "attn_implementation": args.attn_implementation,
        "config": config,
        "distributed_config": DistributedConfig(tp_size=args.tensor_parallel_size),
    }
    dtype = torch.bfloat16
    model_class = getattr(transformers, "AutoModelForMultimodalLM", None)
    if model_class is None:
        model_class = getattr(transformers, "AutoModelForImageTextToText", None)
    if model_class is None:
        raise RuntimeError(
            "transformers must provide AutoModelForMultimodalLM or "
            "AutoModelForImageTextToText"
        )
    model_parameters = inspect.signature(model_class.from_pretrained).parameters
    if "dtype" in model_parameters:
        load_kwargs["dtype"] = dtype
    else:
        load_kwargs["torch_dtype"] = dtype
    model = model_class.from_pretrained(args.model_id, **load_kwargs)
    tp_plan_audit["materialization"] = audit_tensor_parallel_materialization(
        model,
        args.lora_target_modules,
    )
    tp_plan_audit["metadata"] = ensure_tensor_parallel_metadata(
        model,
        requested_tp_size=args.tensor_parallel_size,
    )
    observed_tp_size = int(tp_plan_audit["metadata"]["observed_tp_size"])
    tp_mesh = tensor_parallel_mesh(model._device_mesh)
    model.config.use_cache = False
    # Trainer seeds itself only after the model already exists. Reset every TP
    # process here so PEFT's random LoRA-A initialization is deterministic.
    set_seed(args.seed)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(args.lora_target_modules),
            layers_to_transform=trainable_layer_indices,
            layers_pattern="layers",
        ),
        # PEFT otherwise promotes BF16 adapters to FP32. For a long sequence,
        # that doubles the full LoRA-B output activation and caused the
        # observed 5.51 GiB up_proj allocation failure.
        autocast_adapter_dtype=False,
    )
    tp_plan_audit["lora_initialization"] = synchronize_lora_initialization(
        model,
        tp_mesh=tp_mesh,
        requested_targets=args.lora_target_modules,
    )
    tp_plan_audit["lora_materialization"] = materialize_lora_tensor_parallelism(
        model,
        tp_mesh=tp_mesh,
        tp_size=observed_tp_size,
        requested_targets=args.lora_target_modules,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        disable_input_grads = getattr(model, "disable_input_require_grads", None)
        if callable(disable_input_grads):
            disable_input_grads()
    counts = freeze_multimodal_adapters(
        model,
        freeze_vision=True,
        freeze_aligner=True,
    )
    counts["language_lora_by_target"] = audit_language_lora_targets(
        model,
        args.lora_target_modules,
    )
    counts["language_lora_by_layer"] = audit_trainable_lora_layers(
        model,
        trainable_layer_indices,
    )
    counts["lora_tensor_parallel_audit"] = (
        audit_lora_tensor_parallel_materialization(model)
    )
    counts["lora_dtype_audit"] = audit_trainable_lora_dtypes(model, dtype)
    counts["decoder_layer_selection"] = {
        "total_decoder_layers": total_decoder_layers,
        "trainable_decoder_layers": args.trainable_decoder_layers,
        "frozen_prefix_layer_indices": list(range(first_trainable_layer)),
        "trainable_layer_indices": trainable_layer_indices,
        "selection_policy": "upper contiguous decoder layers",
    }
    counts["frozen_prefix_autograd_guard"] = install_frozen_prefix_input_guard(
        model,
        first_trainable_layer=first_trainable_layer,
    )
    counts["tensor_parallel_size"] = observed_tp_size
    return model, processor, counts, tp_plan_audit


def _training_argument_kwargs(
    args: argparse.Namespace,
    parameter_names: set[str],
    *,
    parallelism_config: Any | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "adam_beta1": args.adam_beta1,
        "adam_beta2": args.adam_beta2,
        "adam_epsilon": args.adam_epsilon,
        "max_grad_norm": args.max_grad_norm,
        "lr_scheduler_type": args.lr_scheduler_type,
        "bf16": True,
        "tf32": True,
        "gradient_checkpointing": False,
        "remove_unused_columns": False,
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": True,
        "logging_steps": 1,
        "save_strategy": "steps" if args.max_steps > 0 else "epoch",
        "save_steps": args.max_steps if args.max_steps > 0 else 500,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "data_seed": args.seed,
        "report_to": [],
        "label_names": ["labels"],
        "optim": "adamw_torch",
        "load_best_model_at_end": False,
    }
    if "parallelism_config" not in parameter_names:
        raise RuntimeError("installed Transformers does not support ParallelismConfig")
    if "save_only_model" not in parameter_names:
        raise RuntimeError("installed Transformers does not support TP-safe model-only checkpoints")
    kwargs["parallelism_config"] = parallelism_config
    kwargs["save_only_model"] = True
    if "warmup_ratio" in parameter_names:
        kwargs["warmup_ratio"] = args.warmup_ratio
    elif "warmup_steps" in parameter_names:
        # Transformers v5.2+ removed warmup_ratio and accepts a float ratio
        # directly through warmup_steps.
        kwargs["warmup_steps"] = args.warmup_ratio
    else:
        raise RuntimeError(
            "installed transformers TrainingArguments supports neither "
            "warmup_ratio nor warmup_steps"
        )
    eval_name = "eval_strategy" if "eval_strategy" in parameter_names else "evaluation_strategy"
    kwargs[eval_name] = "no"
    return kwargs


def training_arguments(args: argparse.Namespace) -> Any:
    from accelerate import ParallelismConfig
    from transformers import TrainingArguments

    parameter_names = set(inspect.signature(TrainingArguments.__init__).parameters)
    kwargs = _training_argument_kwargs(
        args,
        parameter_names,
        parallelism_config=ParallelismConfig(tp_size=args.tensor_parallel_size),
    )
    return TrainingArguments(**kwargs)


def _jsonable_weights(values: dict[Any, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values.items():
        label = key.value if hasattr(key, "value") else str(key)
        if hasattr(value, "__dict__"):
            result[label] = dict(value.__dict__)
        else:
            result[label] = value
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="No-head PASS/FAIL token training for EgoLife judges"
    )
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULTS.model_id)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--min-pixels", type=int, default=DEFAULTS.min_pixels)
    parser.add_argument("--max-pixels", type=int, default=DEFAULTS.max_pixels)
    parser.add_argument("--max-input-tokens", type=int, default=DEFAULTS.max_input_tokens)
    parser.add_argument(
        "--image-context-target-fraction",
        type=float,
        default=DEFAULT_IMAGE_CONTEXT_TARGET_FRACTION,
    )
    parser.add_argument(
        "--image-text-token-reserve",
        type=int,
        default=DEFAULT_IMAGE_TEXT_TOKEN_RESERVE,
    )
    parser.add_argument(
        "--image-item-token-overhead",
        type=int,
        default=DEFAULT_IMAGE_ITEM_TOKEN_OVERHEAD,
    )
    parser.add_argument("--lora-rank", type=int, default=DEFAULTS.lora_rank)
    parser.add_argument("--lora-alpha", type=int, default=DEFAULTS.lora_alpha)
    parser.add_argument("--lora-dropout", type=float, default=DEFAULTS.lora_dropout)
    parser.add_argument(
        "--trainable-decoder-layers",
        type=int,
        default=DEFAULTS.trainable_decoder_layers,
        help="Apply LoRA only to this many upper contiguous decoder layers.",
    )
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=list(DEFAULTS.lora_target_modules),
    )
    parser.add_argument("--learning-rate", type=float, default=DEFAULTS.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=DEFAULTS.weight_decay)
    parser.add_argument("--adam-beta1", type=float, default=DEFAULTS.adam_beta1)
    parser.add_argument("--adam-beta2", type=float, default=DEFAULTS.adam_beta2)
    parser.add_argument("--adam-epsilon", type=float, default=DEFAULTS.adam_epsilon)
    parser.add_argument("--epochs", type=float, default=DEFAULTS.epochs)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Positive values override epochs; intended for bounded smoke runs.",
    )
    parser.add_argument(
        "--train-example-id",
        help=(
            "Optimize only this example after computing class/task weights from the full "
            "manifest. Intended for deterministic one-example smoke runs."
        ),
    )
    parser.add_argument("--warmup-ratio", type=float, default=DEFAULTS.warmup_ratio)
    parser.add_argument("--lr-scheduler-type", default=DEFAULTS.lr_scheduler_type)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=DEFAULTS.gradient_accumulation_steps,
    )
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULTS.max_grad_norm)
    parser.add_argument(
        "--class-weight-smoothing",
        type=float,
        default=DEFAULTS.class_weight_smoothing,
    )
    parser.add_argument(
        "--max-class-weight", type=float, default=DEFAULTS.max_class_weight
    )
    parser.add_argument("--formality-weight", type=float, default=0.2)
    parser.add_argument("--groundedness-weight", type=float, default=0.4)
    parser.add_argument("--answerability-weight", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=DEFAULTS.seed)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=DEFAULTS.gradient_checkpointing,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    if args.max_steps == 0 or args.max_steps < -1:
        raise ValueError("max_steps must be -1 or a positive integer")
    if args.tensor_parallel_size != 2:
        raise ValueError("this launcher is intentionally a pure two-GPU TP job")
    train_examples = load_normalized_manifest(args.train_manifest)
    task_weights = {
        JudgeTask.FORMALITY: args.formality_weight,
        JudgeTask.GROUNDEDNESS: args.groundedness_weight,
        JudgeTask.ANSWERABILITY: args.answerability_weight,
    }
    task_weights = validate_task_weights(task_weights)
    class_weights = class_weights_by_task(
        train_examples,
        smoothing=args.class_weight_smoothing,
        max_weight=args.max_class_weight,
    )
    task_scales = task_sampling_scales(train_examples, task_weights)
    optimization_examples = train_examples
    if args.train_example_id:
        optimization_examples = [
            example
            for example in train_examples
            if example.example_id == args.train_example_id
        ]
        if len(optimization_examples) != 1:
            raise ValueError(
                "train_example_id must match exactly one manifest row; "
                f"id={args.train_example_id!r} matches={len(optimization_examples)}"
            )
    launch_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if launch_world_size != args.tensor_parallel_size:
        raise RuntimeError(
            "pure TP requires WORLD_SIZE == tensor_parallel_size: "
            f"world_size={launch_world_size} tp_size={args.tensor_parallel_size}"
        )
    sampler_audit = audit_tensor_parallel_sampler(
        JudgeDataset(optimization_examples),
        tensor_parallel_size=args.tensor_parallel_size,
        seed=args.seed,
    )
    if int(os.environ.get("RANK", "0")) == 0:
        print(
            "tensor_parallel_sampler_preflight="
            + json.dumps(sampler_audit, sort_keys=True),
            flush=True,
        )
        visual_budget_audit = {
            "max_input_tokens": args.max_input_tokens,
            "target_fraction": args.image_context_target_fraction,
            "all_six_1800_image_max_pixels": adaptive_image_max_pixels(
                image_count=1_800,
                configured_max_pixels=args.max_pixels,
                min_pixels=args.min_pixels,
                max_input_tokens=args.max_input_tokens,
                target_fraction=args.image_context_target_fraction,
                text_token_reserve=args.image_text_token_reserve,
                item_token_overhead=args.image_item_token_overhead,
                vision_token_pixel_area=QWEN_VISION_TOKEN_PIXEL_AREA,
            ),
            "speaker_300_frame_max_pixels": adaptive_image_max_pixels(
                image_count=300,
                configured_max_pixels=args.max_pixels,
                min_pixels=args.min_pixels,
                max_input_tokens=args.max_input_tokens,
                target_fraction=args.image_context_target_fraction,
                text_token_reserve=args.image_text_token_reserve,
                item_token_overhead=args.image_item_token_overhead,
                vision_token_pixel_area=QWEN_VISION_TOKEN_PIXEL_AREA,
            ),
        }
        print(
            "judge_visual_budget_preflight="
            + json.dumps(visual_budget_audit, sort_keys=True),
            flush=True,
        )
    # DistributedConfig must shard the checkpoint while it is loaded. Trainer
    # then mirrors that mesh through ParallelismConfig instead of creating DDP.
    model, processor, parameter_counts, tp_plan_audit = load_model_and_processor(args)
    hf_training_args = training_arguments(args)
    if int(hf_training_args.world_size) != launch_world_size:
        raise RuntimeError(
            "TrainingArguments world size disagrees with torchrun: "
            f"arguments={hf_training_args.world_size} launch={launch_world_size}"
        )
    tokenizer = getattr(processor, "tokenizer", processor)
    token_ids = resolve_verdict_token_ids(tokenizer)
    collator = JudgeFrameCollator(
        processor=processor,
        class_weights=class_weights,
        task_scales=task_scales,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        max_input_tokens=args.max_input_tokens,
        image_context_target_fraction=args.image_context_target_fraction,
        image_text_token_reserve=args.image_text_token_reserve,
        image_item_token_overhead=args.image_item_token_overhead,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_contract = {
        "contract_version": "verdict_token_bce_independent_images_tp2_upper16_v8",
        "model_id": args.model_id,
        "media_contract": (
            "packet-owned 300-frame user timelines sampled at 0.5 FPS; every "
            "sampled JPEG is an independent Qwen image item; one speaker group "
            "or six complete chronological image groups"
        ),
        "assistant_prefix": '{"verdict":"',
        "inference_generation_contract": (
            "lock pass/fail from first-token logits, then continue the same "
            "autoregressive generation through the complete JSON contract"
        ),
        "verdict_token_ids": {key.value: value for key, value in token_ids.items()},
        "loss": "BCEWithLogits(logit_pass - logit_fail)",
        "distributed_contract": {
            "policy": "pure tensor parallelism; the same example is replicated on both ranks",
            "tensor_parallel_size": args.tensor_parallel_size,
            "data_parallel_size": 1,
            "rank_identity_check": "all-gather stable example fingerprint before every forward",
            "checkpoint_state": "adapter/model only; optimizer-state resume is unsupported",
            "tp_plan": tp_plan_audit,
            "preflight": sampler_audit,
        },
        "decoder_training_contract": parameter_counts["decoder_layer_selection"]
        | {
            "frozen_prefix_autograd_guard": parameter_counts[
                "frozen_prefix_autograd_guard"
            ]
        },
        "task_weights": _jsonable_weights(task_weights),
        "class_weights": _jsonable_weights(class_weights),
        "task_sampling_scales": _jsonable_weights(task_scales),
        "train": manifest_summary(train_examples),
        "optimization_train": manifest_summary(optimization_examples),
        "internal_validation": None,
        "checkpoint_policy": (
            "save and retain every epoch checkpoint; select later with separately "
            "labeled standalone validation and test sets"
        ),
        "parameter_counts": parameter_counts,
        "hyperparameters": DEFAULTS.to_dict() | vars(args),
    }
    # Path values are stringified explicitly to keep the audit manifest portable.
    run_contract["hyperparameters"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in run_contract["hyperparameters"].items()
    }
    is_world_process_zero = int(os.environ.get("RANK", "0")) == 0
    if is_world_process_zero:
        (args.output_dir / "training_contract.json").write_text(
            json.dumps(run_contract, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    TrainerClass = build_verdict_trainer_class()
    trainer = TrainerClass(
        model=model,
        args=hf_training_args,
        train_dataset=JudgeDataset(optimization_examples),
        data_collator=collator,
        verdict_token_ids=token_ids,
        tensor_parallel_size=args.tensor_parallel_size,
        optimizer_cls_and_kwargs=(
            __import__("torch").optim.AdamW,
            {
                "lr": args.learning_rate,
                "betas": (args.adam_beta1, args.adam_beta2),
                "eps": args.adam_epsilon,
                "foreach": False,
                "fused": False,
            },
        ),
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(str(args.output_dir / "final_adapter"))
    if trainer.is_world_process_zero():
        processor.save_pretrained(str(args.output_dir / "final_adapter"))


if __name__ == "__main__":
    main()
