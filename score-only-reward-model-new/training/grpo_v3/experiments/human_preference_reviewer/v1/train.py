"""Supervised training, smoke, validation, and locked-test CLI for Reviewer v1."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .checkpoint import (
    load_checkpoint_contract,
    load_binary_heads,
    load_lora_adapter,
    save_checkpoint,
    validate_checkpoint_runtime_contract,
)
from .config import ReviewerV1Config
from .data import (
    CONTRACT_VERSION,
    SUPERVISION_TYPE,
    EvidenceRecord,
    binary_class_statistics,
    load_annotation_csv,
    sha256_file,
    validate_split_manifest,
)
from .evaluation import binary_metrics
from .lora import inject_reviewer_lora, parameter_audit, target_layer_indices
from .losses import LOSS_FIELDS, reviewer_losses
from .modeling import ReviewerV1, resolve_hidden_size
from .prompting import build_messages, encode_candidate


def load_media_map(path: str | Path) -> dict[str, str]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise ValueError("media map must be a non-empty URL-to-path object")
    result: dict[str, str] = {}
    for source, local in raw.items():
        local_path = Path(str(local)).expanduser()
        if not local_path.is_file() or local_path.stat().st_size <= 0:
            raise ValueError(f"materialized video is missing or empty: {local_path}")
        result[str(source)] = str(local_path.resolve())
    return result


def select_evidence(
    records: Sequence[EvidenceRecord], manifest: Mapping[str, Any], split: str
) -> tuple[EvidenceRecord, ...]:
    if split not in {"train", "validation", "locked_test"}:
        raise ValueError(f"unsupported split: {split}")
    ids = [str(value) for value in manifest.get(f"{split}_evidence_ids") or []]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError(f"split {split} must contain unique evidence IDs")
    by_id = {record.evidence_id: record for record in records}
    missing = sorted(set(ids) - set(by_id))
    if missing:
        raise ValueError(f"split {split} references absent evidence IDs: {missing}")
    return tuple(by_id[evidence_id] for evidence_id in ids)


def validate_evaluation_data_provenance(
    contract: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    evaluation_provenance: str,
    split: str,
    evaluation_csv_sha256: str,
    evaluation_split_sha256: str,
) -> None:
    if evaluation_provenance == "checkpoint_split":
        if contract.get("csv_sha256") != evaluation_csv_sha256:
            raise ValueError("checkpoint CSV hash mismatch")
        if contract.get("split_sha256") != evaluation_split_sha256:
            raise ValueError("checkpoint split manifest hash mismatch")
        return
    if evaluation_provenance != "external_holdout":
        raise ValueError(f"unsupported evaluation provenance: {evaluation_provenance}")
    if split != "locked_test":
        raise ValueError("external holdout evaluation requires --split locked_test")
    if manifest.get("split_mode") != "external_holdout":
        raise ValueError("external holdout evaluation requires split_mode=external_holdout")
    training_ids = set(contract.get("training_evidence_ids") or ())
    holdout_ids = set(manifest.get("locked_test_evidence_ids") or ())
    overlap = sorted(training_ids & holdout_ids)
    if overlap:
        raise ValueError(f"external holdout overlaps checkpoint training evidence: {overlap}")


def _move(value: Any, device: str) -> Any:
    return value.to(device) if hasattr(value, "to") else value


def _encoded_inputs(
    evidence: EvidenceRecord,
    candidate: Any,
    *,
    media_map: Mapping[str, str],
    processor: Any,
    process_vision_info: Any,
    device: str,
) -> dict[str, Any]:
    missing = [source for source in (evidence.video_a_source, evidence.video_b_source) if source not in media_map]
    if missing:
        raise ValueError(f"media map lacks sources for {evidence.evidence_id}: {missing}")
    messages = build_messages(
        candidate,
        video_a_path=media_map[evidence.video_a_source],
        video_b_path=media_map[evidence.video_b_source],
        video_a_user=evidence.video_a_user,
        video_b_user=evidence.video_b_user,
    )
    return {
        name: _move(value, device)
        for name, value in encode_candidate(processor, process_vision_info, messages).items()
    }


def _labels(candidate: Any, *, torch_module: Any, device: str) -> dict[str, Any]:
    return {
        name: torch_module.tensor([grade], dtype=torch_module.long, device=device)
        for name, grade in candidate.labels().items()
    }


def _load_runtime(
    *, model_path: str, device: str, dtype_name: str, config: ReviewerV1Config
) -> tuple[Any, Any, tuple[int, ...], tuple[str, ...]]:
    import torch
    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as error:
        raise RuntimeError("Torch runtime requires transformers and PEFT") from error
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_name]
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    load_kwargs = {"trust_remote_code": True, "attn_implementation": "sdpa"}
    try:
        backbone = AutoModelForImageTextToText.from_pretrained(model_path, dtype=dtype, **load_kwargs)
    except TypeError:
        backbone = AutoModelForImageTextToText.from_pretrained(model_path, torch_dtype=dtype, **load_kwargs)
    if getattr(backbone.config, "model_type", None) != "qwen3_vl":
        raise ValueError(
            f'Reviewer v1 requires model_type="qwen3_vl"; found {getattr(backbone.config, "model_type", None)!r}'
        )
    backbone.to(device)
    backbone.config.use_cache = False
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    if config.lora_enabled:
        indices = target_layer_indices(backbone, last_n=config.last_n_shared_blocks)
        backbone, targets = inject_reviewer_lora(backbone, config)
    else:
        indices, targets = (), ()
    reviewer = ReviewerV1(
        backbone,
        resolve_hidden_size(backbone.config),
        active_heads=config.active_heads,
    ).to(device)
    return reviewer, processor, indices, targets


def _gradient_route_audit(
    reviewer: Any, losses: Mapping[str, Any], *, lora_enabled: bool
) -> dict[str, Any]:
    import torch

    named = [(name, parameter) for name, parameter in reviewer.named_parameters() if parameter.requires_grad]
    loss_names = {
        "evidence_loss": "evidence_head.",
        "answerability_loss": "answerability_head.",
        "formality_loss": "formality_head.",
    }
    result: dict[str, Any] = {}
    for loss_name, direct_prefix in loss_names.items():
        if loss_name not in losses:
            continue
        gradients = torch.autograd.grad(
            losses[loss_name], [parameter for _, parameter in named],
            retain_graph=True, allow_unused=True,
        )
        by_name = dict(zip((name for name, _ in named), gradients))
        direct = [name for name, gradient in by_name.items() if name.startswith(direct_prefix) and gradient is not None]
        other_heads = [
            name for name, gradient in by_name.items()
            if name.startswith(("evidence_head.", "answerability_head.", "formality_head."))
            and not name.startswith(direct_prefix) and gradient is not None
        ]
        lora = [
            name for name, gradient in by_name.items()
            if (".lora_A." in name or ".lora_B." in name)
            and gradient is not None and bool(torch.isfinite(gradient).all())
            and float(gradient.detach().float().norm().cpu()) > 0
        ]
        passed = bool(direct) and not other_heads and (bool(lora) if lora_enabled else not lora)
        result[loss_name] = {
            "status": "passed" if passed else "failed",
            "direct_head_gradient_names": direct,
            "unexpected_other_head_gradient_names": other_heads,
            "nonzero_lora_gradient_names": lora,
        }
        if not passed:
            raise RuntimeError(f"gradient route audit failed for {loss_name}: {result[loss_name]}")
    return result


def _task_gradient_diagnostics(
    reviewer: Any,
    losses: Mapping[str, Any],
    *,
    active_heads: tuple[str, ...],
    loss_weights: Mapping[str, float],
) -> dict[str, Any] | None:
    """Measure per-task gradient magnitudes and conflicts on shared LoRA parameters."""
    import torch

    if len(active_heads) < 2:
        return None
    shared = [
        parameter
        for name, parameter in reviewer.named_parameters()
        if parameter.requires_grad and (".lora_A." in name or ".lora_B." in name)
    ]
    if not shared:
        return None
    gradients: dict[str, tuple[Any | None, ...]] = {}
    norms: dict[str, float] = {}
    for field in active_heads:
        loss_name = LOSS_FIELDS[field][1]
        task_gradients = torch.autograd.grad(
            losses[loss_name], shared, retain_graph=True, allow_unused=True
        )
        gradients[field] = task_gradients
        squared_norm = sum(
            gradient.detach().float().square().sum()
            for gradient in task_gradients
            if gradient is not None
        )
        norms[field] = float(torch.sqrt(squared_norm).cpu()) if not isinstance(squared_norm, int) else 0.0

    pairwise: dict[str, float | None] = {}
    for left_index, left in enumerate(active_heads):
        for right in active_heads[left_index + 1:]:
            dot_product = sum(
                left_gradient.detach().float().mul(right_gradient.detach().float()).sum()
                for left_gradient, right_gradient in zip(gradients[left], gradients[right])
                if left_gradient is not None and right_gradient is not None
            )
            denominator = norms[left] * norms[right]
            if denominator <= 0.0 or isinstance(dot_product, int):
                cosine = None
            else:
                cosine = max(-1.0, min(1.0, float(dot_product.cpu()) / denominator))
            pairwise[f"{left}__vs__{right}"] = cosine
    return {
        "gradient_norms": norms,
        "weighted_gradient_norms": {
            field: norms[field] * float(loss_weights[field]) for field in active_heads
        },
        "pairwise_cosines": pairwise,
        "negative_pairs": sorted(name for name, value in pairwise.items() if value is not None and value < 0.0),
    }


def _summarize_gradient_diagnostics(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pair_names = sorted({
        name
        for sample in samples
        for name in (sample.get("pairwise_cosines") or {})
    })
    summary: dict[str, Any] = {}
    for name in pair_names:
        values = [
            float(sample["pairwise_cosines"][name])
            for sample in samples
            if (sample.get("pairwise_cosines") or {}).get(name) is not None
        ]
        if values:
            summary[name] = {
                "sample_count": len(values),
                "mean_cosine": sum(values) / len(values),
                "minimum_cosine": min(values),
                "maximum_cosine": max(values),
                "negative_fraction": sum(value < 0.0 for value in values) / len(values),
            }
    return summary


def _weighted_validation_metric(
    metrics: Mapping[str, Mapping[str, Any]] | None,
    weights: Mapping[str, float],
    metric_name: str,
) -> float | None:
    available = [
        (field, values.get(metric_name))
        for field, values in (metrics or {}).items()
        if values.get(metric_name) is not None
    ]
    if not available:
        return None
    denominator = sum(float(weights[field]) for field, _ in available)
    return sum(float(weights[field]) * float(value) for field, value in available) / denominator


def _epoch_metric_summary(
    metrics: Mapping[str, Mapping[str, Any]],
    weights: Mapping[str, float],
) -> dict[str, Any]:
    """Compact epoch metrics for either online training or held-out evaluation."""
    return {
        "weighted_loss": _weighted_validation_metric(metrics, weights, "loss"),
        "weighted_accuracy": _weighted_validation_metric(metrics, weights, "accuracy"),
        "weighted_macro_f1": _weighted_validation_metric(metrics, weights, "macro_f1"),
        "weighted_balanced_accuracy": _weighted_validation_metric(
            metrics, weights, "balanced_accuracy"
        ),
        "per_head_accuracy": {
            field: values["accuracy"] for field, values in metrics.items()
        },
        "per_head_loss": {
            field: values["loss"] for field, values in metrics.items()
        },
    }


def _validation_loss_improved(
    current: float, best: float | None, *, min_delta: float
) -> bool:
    if min_delta < 0.0:
        raise ValueError("early stopping min_delta must be non-negative")
    if not math.isfinite(current):
        raise ValueError("early stopping validation loss must be finite")
    return best is None or current < best - min_delta


def _evaluate(
    reviewer: Any,
    records: Iterable[EvidenceRecord],
    *,
    media_map: Mapping[str, str], processor: Any, process_vision_info: Any, device: str,
    active_heads: tuple[str, ...],
    class_weights: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    import torch

    labels = {field: [] for field in active_heads}
    pass_probabilities = {field: [] for field in labels}
    loss_values = {field: [] for field in labels}
    reviewer.eval()
    with torch.no_grad():
        for evidence in records:
            for candidate in evidence.candidates:
                output = reviewer(**_encoded_inputs(
                    evidence, candidate, media_map=media_map, processor=processor,
                    process_vision_info=process_vision_info, device=device,
                ))
                losses = reviewer_losses(
                    output,
                    _labels(candidate, torch_module=torch, device=device),
                    active_heads=active_heads,
                    class_weights=class_weights,
                )
                pairs = {
                    "evidence_quality": (
                        output.evidence_logits, losses.get("evidence_loss")
                    ),
                    "answerability": (
                        output.answerability_logits, losses.get("answerability_loss")
                    ),
                    "qa_formality": (
                        output.formality_logits, losses.get("formality_loss")
                    ),
                }
                for field in active_heads:
                    logits, loss = pairs[field]
                    labels[field].append(candidate.labels()[field])
                    pass_probabilities[field].append(
                        float(torch.softmax(logits[0].float(), dim=-1)[1].cpu())
                    )
                    loss_values[field].append(float(loss.detach().float().cpu()))
    return {
        field: binary_metrics(
            labels[field], pass_probabilities[field],
            loss=sum(loss_values[field]) / len(loss_values[field])
        )
        for field in labels
    }


def _verify_checkpoint_reload_in_place(
    reviewer: Any, checkpoint_dir: Path, *, lora_enabled: bool
) -> dict[str, Any]:
    import torch

    expected = {
        name: parameter.detach().cpu().clone()
        for name, parameter in reviewer.named_parameters()
        if parameter.requires_grad
    }
    with torch.no_grad():
        for name, parameter in reviewer.named_parameters():
            if name in expected:
                parameter.zero_()
    load_binary_heads(reviewer, checkpoint_dir)
    if lora_enabled:
        load_lora_adapter(reviewer, checkpoint_dir)
    mismatched = [
        name for name, parameter in reviewer.named_parameters()
        if name in expected and not torch.equal(parameter.detach().cpu(), expected[name])
    ]
    result = {"status": "passed" if not mismatched else "failed", "mismatched_names": mismatched}
    if mismatched:
        raise RuntimeError(f"checkpoint reload mismatch: {mismatched}")
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from qwen_vl_utils import process_vision_info

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gradient_diagnostics_interval < 0:
        raise ValueError("gradient diagnostics interval must be non-negative")
    if args.early_stopping_patience < 0:
        raise ValueError("early stopping patience must be non-negative")
    if args.early_stopping_min_epochs < 0:
        raise ValueError("early stopping min_epochs must be non-negative")
    if args.early_stopping_min_epochs > args.epochs:
        raise ValueError("early stopping min_epochs cannot exceed requested epochs")
    if args.early_stopping_min_delta < 0.0:
        raise ValueError("early stopping min_delta must be non-negative")
    if args.cosine_min_lr < 0.0 or args.cosine_min_lr > args.learning_rate:
        raise ValueError("cosine minimum learning rate must be between zero and learning rate")
    if args.lr_scheduler != "constant" and args.command != "fit":
        raise ValueError("learning-rate scheduling is only supported for fit")
    if args.save_best_checkpoint_per_epoch and args.command != "fit":
        raise ValueError("best-checkpoint-per-epoch saving is only supported for fit")
    if args.save_best_checkpoint_per_epoch and args.early_stopping_patience <= 0:
        raise ValueError("best-checkpoint-per-epoch saving requires validation early stopping")
    if args.log_training_metrics_per_epoch and args.command != "fit":
        raise ValueError("per-epoch training metrics are only supported for fit")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epoch_metrics_path = output_dir / "epoch_metrics.jsonl"
    if args.log_training_metrics_per_epoch:
        epoch_metrics_path.write_text("", encoding="utf-8")
    audit = load_annotation_csv(args.csv)
    media_map = load_media_map(args.media_map)
    config = ReviewerV1Config(
        stage=args.stage,
        model_name_or_path=args.model,
        seed=args.seed,
        train_evidence_count=args.train_evidence_count,
        validation_evidence_count=args.validation_evidence_count,
        locked_test_evidence_count=args.locked_test_evidence_count,
    )
    manifest = json.loads(Path(args.split_manifest).read_text(encoding="utf-8")) if args.split_manifest else None
    if manifest is not None:
        expected_counts = None if (
            args.command == "evaluate" and args.evaluation_provenance == "external_holdout"
        ) else (
            config.train_evidence_count,
            config.validation_evidence_count,
            config.locked_test_evidence_count,
        )
        validate_split_manifest(
            manifest,
            expected_counts=expected_counts,
            require_contract=True,
        )
    if manifest and manifest.get("csv_sha256") != audit.csv_sha256:
        raise ValueError("split manifest CSV hash does not match annotation CSV")
    reviewer, processor, indices, targets = _load_runtime(
        model_path=args.model, device=args.device, dtype_name=args.torch_dtype, config=config
    )
    param_audit = parameter_audit(
        reviewer,
        expected_layer_indices=indices,
        active_heads=config.active_heads,
        lora_enabled=config.lora_enabled,
    )
    param_audit["lora_target_modules"] = list(targets)
    trainable = [parameter for parameter in reviewer.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    baseline_lora_noop_verified = False
    if args.command == "smoke":
        records = (audit.eligible_evidence[0],)
    elif args.command == "fit":
        if manifest is None:
            raise ValueError("fit requires --split-manifest")
        records = select_evidence(audit.eligible_evidence, manifest, "train")
    else:
        if manifest is None:
            raise ValueError("evaluate requires --split-manifest")
        evaluation_split_hash = sha256_file(args.split_manifest)
        if args.checkpoint_mode == "trained":
            if not args.checkpoint:
                raise ValueError("trained evaluation requires --checkpoint")
            contract = load_checkpoint_contract(args.checkpoint)
            validate_checkpoint_runtime_contract(
                contract,
                stage=config.stage,
                active_heads=config.active_heads,
                lora_enabled=config.lora_enabled,
                model_name_or_path=config.model_name_or_path,
                reviewer_config=config.to_dict(),
                actual_lora_targets=targets,
            )
            validate_evaluation_data_provenance(
                contract,
                manifest,
                evaluation_provenance=args.evaluation_provenance,
                split=args.split,
                evaluation_csv_sha256=audit.csv_sha256,
                evaluation_split_sha256=evaluation_split_hash,
            )
            load_binary_heads(reviewer, args.checkpoint)
            if config.lora_enabled:
                load_lora_adapter(reviewer, args.checkpoint)
            training_class_weights = contract["class_weights"]
            evaluation_class_weights = {
                field: [1.0, 1.0] for field in config.active_heads
            }
        else:
            if args.checkpoint:
                raise ValueError("base_untrained evaluation must not receive --checkpoint")
            # The seeded reviewer heads are untrained and the newly injected LoRA
            # path is initially a no-op, so this measures the untouched base
            # backbone plus the initial reviewer parameterization.
            validate_evaluation_data_provenance(
                {},
                manifest,
                evaluation_provenance=args.evaluation_provenance,
                split=args.split,
                evaluation_csv_sha256=audit.csv_sha256,
                evaluation_split_sha256=evaluation_split_hash,
            )
            lora_b_parameters = [
                parameter
                for name, parameter in reviewer.named_parameters()
                if ".lora_B." in name
            ]
            if not lora_b_parameters or any(
                torch.count_nonzero(parameter.detach()).item() != 0
                for parameter in lora_b_parameters
            ):
                raise RuntimeError("base_untrained LoRA path is not an initial no-op")
            baseline_lora_noop_verified = True
            evaluation_class_weights = {
                field: [1.0, 1.0] for field in config.active_heads
            }
            training_class_weights = None
        evaluation_records = select_evidence(
            audit.eligible_evidence, manifest, args.split
        )
        metrics = _evaluate(
            reviewer, evaluation_records,
            media_map=media_map, processor=processor, process_vision_info=process_vision_info,
            device=args.device,
            active_heads=config.active_heads,
            class_weights=evaluation_class_weights,
        )
        result = {
            "status": "passed",
            "split": args.split,
            "evaluation_evidence_count": len(evaluation_records),
            "evaluation_candidate_count": sum(
                len(evidence.candidates) for evidence in evaluation_records
            ),
            "checkpoint_mode": args.checkpoint_mode,
            "baseline_seed": args.seed if args.checkpoint_mode == "base_untrained" else None,
            "baseline_lora_noop_verified": baseline_lora_noop_verified,
            "checkpoint_dir": (
                str(Path(args.checkpoint).resolve()) if args.checkpoint else None
            ),
            "metrics": metrics,
            "weighted_loss": _weighted_validation_metric(
                metrics, config.active_loss_weights, "loss"
            ),
            "weighted_accuracy": _weighted_validation_metric(
                metrics, config.active_loss_weights, "accuracy"
            ),
            "weighted_macro_f1": _weighted_validation_metric(
                metrics, config.active_loss_weights, "macro_f1"
            ),
            "weighted_balanced_accuracy": _weighted_validation_metric(
                metrics, config.active_loss_weights, "balanced_accuracy"
            ),
            "evaluation_loss_class_weights": evaluation_class_weights,
            "training_class_weights": training_class_weights,
            "head_loss_aggregation": "equal_mean",
            "evaluation_provenance": args.evaluation_provenance,
            "evaluation_csv_sha256": audit.csv_sha256,
            "evaluation_split_sha256": evaluation_split_hash,
            "parameter_audit": param_audit,
        }
        _write_json(Path(args.output_dir) / "evaluation_result.json", result)
        return result
    validation_ids = list((manifest or {}).get("validation_evidence_ids") or ())
    validation_records = (
        select_evidence(audit.eligible_evidence, manifest, "validation")
        if args.command == "fit" and manifest is not None and validation_ids
        else ()
    )
    if args.early_stopping_patience > 0 and not validation_records:
        raise ValueError("early stopping requires a non-empty validation split")
    training_class_counts, class_weights = binary_class_statistics(
        records, active_heads=config.active_heads
    )
    steps_per_epoch = sum(len(evidence.candidates) for evidence in records)
    scheduled_steps = min(args.max_steps, args.epochs * steps_per_epoch)
    scheduler = None
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, scheduled_steps),
            eta_min=args.cosine_min_lr,
        )
    split_hash = sha256_file(args.split_manifest) if args.split_manifest else "SMOKE_NO_SPLIT"
    checkpoint_config = {
        "contract_version": CONTRACT_VERSION,
        **config.to_dict(),
        "actual_lora_targets": list(targets),
        "training_evidence_ids": sorted(evidence.evidence_id for evidence in records),
        "training_split_mode": (manifest or {}).get("split_mode", "smoke"),
        "training_class_counts": training_class_counts,
        "class_weights": class_weights,
    }
    initial = {
        name: parameter.detach().cpu().clone()
        for name, parameter in reviewer.named_parameters() if parameter.requires_grad
    }
    history = []
    gradient_routes = None
    gradient_diagnostics = []
    epoch_training_history = []
    epoch_validation_history = []
    best_validation_loss = None
    best_validation_metrics = None
    best_trainable_state = None
    best_optimizer_state = None
    best_scheduler_state = None
    best_epoch = None
    best_global_step = None
    epochs_without_improvement = 0
    completed_epochs = 0
    stopped_early = False
    global_step = 0
    reviewer.train()
    training_started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        epoch_started = time.perf_counter()
        candidates = [(evidence, candidate) for evidence in records for candidate in evidence.candidates]
        random.Random(args.seed + epoch).shuffle(candidates)
        epoch_step_count = 0
        online_training_labels = {
            field: [] for field in config.active_heads
        } if args.log_training_metrics_per_epoch else None
        online_training_logits = {
            field: [] for field in config.active_heads
        } if args.log_training_metrics_per_epoch else None
        online_training_losses = {
            field: [] for field in config.active_heads
        } if args.log_training_metrics_per_epoch else None
        for evidence, candidate in candidates:
            output = reviewer(**_encoded_inputs(
                evidence, candidate, media_map=media_map, processor=processor,
                process_vision_info=process_vision_info, device=args.device,
            ))
            losses = reviewer_losses(
                output,
                _labels(candidate, torch_module=torch, device=args.device),
                active_heads=config.active_heads,
                class_weights=class_weights,
            )
            if args.log_training_metrics_per_epoch:
                if (
                    online_training_labels is None
                    or online_training_logits is None
                    or online_training_losses is None
                ):
                    raise RuntimeError("online training metric buffers are unavailable")
                candidate_labels = candidate.labels()
                for field in config.active_heads:
                    logits_name, loss_name = LOSS_FIELDS[field]
                    online_training_labels[field].append(candidate_labels[field])
                    online_training_logits[field].append(
                        getattr(output, logits_name).detach().float()
                    )
                    online_training_losses[field].append(
                        losses[loss_name].detach().float()
                    )
            if gradient_routes is None and args.command == "smoke":
                gradient_routes = _gradient_route_audit(
                    reviewer, losses, lora_enabled=config.lora_enabled
                )
            next_step = global_step + 1
            if (
                args.gradient_diagnostics_interval > 0
                and config.lora_enabled
                and len(config.active_heads) > 1
                and (next_step == 1 or next_step % args.gradient_diagnostics_interval == 0)
            ):
                diagnostics = _task_gradient_diagnostics(
                    reviewer,
                    losses,
                    active_heads=config.active_heads,
                    loss_weights=config.active_loss_weights,
                )
                if diagnostics is not None:
                    gradient_diagnostics.append({"step": next_step, **diagnostics})
            losses["loss"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            epoch_step_count += 1
            history.append({
                "step": global_step, "evidence_id": evidence.evidence_id,
                "candidate_id": candidate.candidate_id,
                **{name: float(value.detach().float().cpu()) for name, value in losses.items()},
                "grad_norm": float(grad_norm.detach().float().cpu()),
            })
            if global_step >= args.max_steps:
                break
        completed_epochs = epoch + 1
        training_pass_seconds = time.perf_counter() - epoch_started
        if args.log_training_metrics_per_epoch:
            metrics_started = time.perf_counter()
            if (
                online_training_labels is None
                or online_training_logits is None
                or online_training_losses is None
            ):
                raise RuntimeError("online training metric buffers are unavailable")
            training_epoch_metrics = {}
            for field in config.active_heads:
                logits = torch.cat(online_training_logits[field], dim=0)
                pass_probabilities = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
                mean_loss = float(
                    torch.stack(online_training_losses[field]).mean().cpu()
                )
                training_epoch_metrics[field] = binary_metrics(
                    online_training_labels[field],
                    pass_probabilities,
                    loss=mean_loss,
                )
            training_epoch_entry = {
                "epoch": completed_epochs,
                "global_step": global_step,
                "training_steps": epoch_step_count,
                "measurement": "online_training_pass",
                "training_pass_seconds": training_pass_seconds,
                "metrics_aggregation_seconds": time.perf_counter() - metrics_started,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                **_epoch_metric_summary(
                    training_epoch_metrics, config.active_loss_weights
                ),
            }
            epoch_training_history.append(training_epoch_entry)
            with epoch_metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "event": "training_metrics",
                    **training_epoch_entry,
                }, ensure_ascii=False) + "\n")
            print(
                "EPOCH_TRAINING_METRICS "
                + json.dumps(training_epoch_entry, ensure_ascii=False),
                flush=True,
            )
            reviewer.train()
        if args.early_stopping_patience > 0:
            validation_started = time.perf_counter()
            epoch_metrics = _evaluate(
                reviewer,
                validation_records,
                media_map=media_map,
                processor=processor,
                process_vision_info=process_vision_info,
                device=args.device,
                active_heads=config.active_heads,
                class_weights=class_weights,
            )
            weighted_loss = _weighted_validation_metric(
                epoch_metrics, config.active_loss_weights, "loss"
            )
            if weighted_loss is None:
                raise RuntimeError("early stopping validation loss is unavailable")
            improved = _validation_loss_improved(
                weighted_loss,
                best_validation_loss,
                min_delta=args.early_stopping_min_delta,
            )
            if improved:
                best_validation_loss = weighted_loss
                best_validation_metrics = epoch_metrics
                best_trainable_state = {
                    name: parameter.detach().cpu().clone()
                    for name, parameter in reviewer.named_parameters()
                    if parameter.requires_grad
                }
                best_optimizer_state = copy.deepcopy(optimizer.state_dict())
                best_scheduler_state = (
                    copy.deepcopy(scheduler.state_dict()) if scheduler is not None else None
                )
                best_epoch = completed_epochs
                best_global_step = global_step
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            validation_epoch_entry = {
                "epoch": completed_epochs,
                "global_step": global_step,
                "training_steps": epoch_step_count,
                "weighted_loss": weighted_loss,
                "weighted_macro_f1": _weighted_validation_metric(
                    epoch_metrics, config.active_loss_weights, "macro_f1"
                ),
                "weighted_balanced_accuracy": _weighted_validation_metric(
                    epoch_metrics,
                    config.active_loss_weights,
                    "balanced_accuracy",
                ),
                "per_head_loss": {
                    field: values["loss"] for field, values in epoch_metrics.items()
                },
                "improved": improved,
                "epochs_without_improvement": epochs_without_improvement,
                "training_pass_seconds": training_pass_seconds,
                "validation_seconds": time.perf_counter() - validation_started,
                "epoch_total_seconds": time.perf_counter() - epoch_started,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            epoch_validation_history.append(validation_epoch_entry)
            if args.log_training_metrics_per_epoch:
                with epoch_metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({
                        "event": "validation_metrics",
                        **validation_epoch_entry,
                    }, ensure_ascii=False) + "\n")
            if improved and args.save_best_checkpoint_per_epoch:
                epoch_checkpoint = (
                    output_dir / "checkpoints" / f"best_epoch_{completed_epochs:03d}"
                )
                save_checkpoint(
                    reviewer,
                    epoch_checkpoint,
                    config=checkpoint_config,
                    csv_sha256=audit.csv_sha256,
                    split_sha256=split_hash,
                    parameter_audit=param_audit,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    trainer_state={
                        "checkpoint_kind": "validation_best",
                        "global_step": global_step,
                        "completed_epochs": completed_epochs,
                        "best_validation_loss": weighted_loss,
                    },
                    processor=processor,
                    active_heads=config.active_heads,
                    lora_enabled=config.lora_enabled,
                )
                (output_dir / "best_checkpoint.txt").write_text(
                    str(epoch_checkpoint.resolve()) + "\n", encoding="utf-8"
                )
                print(
                    "BEST_CHECKPOINT_SAVED "
                    + json.dumps({
                        "epoch": completed_epochs,
                        "global_step": global_step,
                        "validation_loss": weighted_loss,
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "checkpoint": str(epoch_checkpoint.resolve()),
                    }),
                    flush=True,
                )
            reviewer.train()
            if (
                completed_epochs >= args.early_stopping_min_epochs
                and epochs_without_improvement >= args.early_stopping_patience
            ):
                stopped_early = True
                break
        if global_step >= args.max_steps:
            break
    if best_trainable_state is not None:
        with torch.no_grad():
            for name, parameter in reviewer.named_parameters():
                if name in best_trainable_state:
                    parameter.copy_(best_trainable_state[name].to(parameter.device))
        if best_optimizer_state is None:
            raise RuntimeError("best optimizer state is unavailable")
        optimizer.load_state_dict(best_optimizer_state)
        if scheduler is not None:
            if best_scheduler_state is None:
                raise RuntimeError("best scheduler state is unavailable")
            scheduler.load_state_dict(best_scheduler_state)
    deltas = {
        name: float((parameter.detach().cpu() - initial[name]).float().norm())
        for name, parameter in reviewer.named_parameters() if parameter.requires_grad
    }
    head_prefixes = {
        "evidence_quality": "evidence_head.",
        "answerability": "answerability_head.",
        "qa_formality": "formality_head.",
    }
    head_changed = all(
        any(name.startswith(head_prefixes[field]) and delta > 0 for name, delta in deltas.items())
        for field in config.active_heads
    )
    lora_changed = any((".lora_A." in name or ".lora_B." in name) and delta > 0 for name, delta in deltas.items())
    if (
        not history
        or not head_changed
        or (config.lora_enabled and not lora_changed)
        or (not config.lora_enabled and lora_changed)
        or any(not math.isfinite(row["loss"]) for row in history)
    ):
        raise RuntimeError("training update gate failed")
    elapsed_seconds = time.perf_counter() - training_started
    per_candidate: dict[str, list[float]] = {}
    for row in history:
        key = f'{row["evidence_id"]}::{row["candidate_id"]}'
        per_candidate.setdefault(key, []).append(row["loss"])
    repeated_candidate_loss = {
        key: {"first": values[0], "last": values[-1], "improved": values[-1] < values[0]}
        for key, values in per_candidate.items() if len(values) >= 2
    }
    validation_metrics = best_validation_metrics
    if validation_metrics is None and validation_records:
        validation_metrics = _evaluate(
            reviewer, validation_records,
            media_map=media_map, processor=processor, process_vision_info=process_vision_info,
            device=args.device,
            active_heads=config.active_heads,
            class_weights=class_weights,
        )
    save_checkpoint(
        reviewer, output_dir / "checkpoint", config=checkpoint_config,
        csv_sha256=audit.csv_sha256, split_sha256=split_hash,
        parameter_audit=param_audit, optimizer=optimizer, scheduler=scheduler,
        trainer_state={
            "global_step": best_global_step if best_global_step is not None else global_step,
            "completed_epochs": best_epoch if best_epoch is not None else completed_epochs,
        }, processor=processor,
        active_heads=config.active_heads,
        lora_enabled=config.lora_enabled,
    )
    checkpoint_reload = _verify_checkpoint_reload_in_place(
        reviewer, output_dir / "checkpoint", lora_enabled=config.lora_enabled
    )
    result = {
        "status": "passed", "mode": args.command, "global_step": global_step,
        "checkpoint_global_step": (
            best_global_step if best_global_step is not None else global_step
        ),
        "requested_epochs": args.epochs,
        "completed_epochs": completed_epochs,
        "checkpoint_epoch": best_epoch if best_epoch is not None else completed_epochs,
        "optimizer_config": {
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "lr_scheduler": args.lr_scheduler,
            "cosine_min_lr": args.cosine_min_lr if args.lr_scheduler == "cosine" else None,
            "scheduled_steps": scheduled_steps,
            "checkpoint_learning_rate": float(optimizer.param_groups[0]["lr"]),
        },
        "save_best_checkpoint_per_epoch": args.save_best_checkpoint_per_epoch,
        "best_checkpoint_pointer": (
            str((output_dir / "best_checkpoint.txt").resolve())
            if args.save_best_checkpoint_per_epoch else None
        ),
        "training_accuracy_logged_per_epoch": args.log_training_metrics_per_epoch,
        "epoch_metrics_log": (
            str(epoch_metrics_path.resolve())
            if args.log_training_metrics_per_epoch else None
        ),
        "epoch_training_history": epoch_training_history,
        "early_stopping": {
            "enabled": args.early_stopping_patience > 0,
            "monitor": "validation_mean_class_weighted_ce",
            "mode": "min",
            "patience_epochs": args.early_stopping_patience,
            "minimum_epochs": args.early_stopping_min_epochs,
            "min_delta": args.early_stopping_min_delta,
            "stopped_early": stopped_early,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "epochs_without_improvement_at_stop": epochs_without_improvement,
        },
        "epoch_validation_history": epoch_validation_history,
        "stage": config.stage,
        "supervision_type": SUPERVISION_TYPE,
        "active_heads": list(config.active_heads),
        "training_split_mode": (manifest or {}).get("split_mode", "smoke"),
        "training_evidence_count": len(records),
        "validation_status": "evaluated" if validation_metrics is not None else "not_available",
        "training_class_counts": training_class_counts,
        "class_weights": class_weights,
        "class_weight_source": "training_split_only",
        "head_loss_aggregation": "equal_mean",
        "stage0_framework_validation": config.stage == "stage0",
        "head_parameter_delta_nonzero": head_changed,
        "lora_parameter_delta_nonzero": lora_changed,
        "gradient_routes": gradient_routes,
        "gradient_diagnostics_interval": args.gradient_diagnostics_interval,
        "gradient_diagnostics": gradient_diagnostics,
        "gradient_conflict_summary": _summarize_gradient_diagnostics(gradient_diagnostics),
        "parameter_audit": param_audit,
        "validation_metrics": validation_metrics,
        "throughput": {
            "elapsed_seconds": elapsed_seconds,
            "candidate_steps_per_hour": global_step * 3600.0 / max(elapsed_seconds, 1e-9),
        },
        "repeated_candidate_loss": repeated_candidate_loss,
        "validation_macro_f1_mean": (
            sum(value["macro_f1"] for value in validation_metrics.values()) / len(validation_metrics)
            if validation_metrics else None
        ),
        "validation_balanced_accuracy_mean": (
            sum(value["balanced_accuracy"] for value in validation_metrics.values())
            / len(validation_metrics)
            if validation_metrics else None
        ),
        "validation_weighted_macro_f1": _weighted_validation_metric(
            validation_metrics, config.active_loss_weights, "macro_f1"
        ),
        "validation_weighted_balanced_accuracy": _weighted_validation_metric(
            validation_metrics, config.active_loss_weights, "balanced_accuracy"
        ),
        "checkpoint_reload": checkpoint_reload,
        "history": history,
    }
    _write_json(output_dir / "training_result.json", result)
    return result


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("smoke", "fit", "evaluate"):
        child = subparsers.add_parser(command)
        child.add_argument("--csv", required=True)
        child.add_argument("--media-map", required=True)
        child.add_argument("--model", required=True)
        child.add_argument("--output-dir", required=True)
        child.add_argument("--split-manifest")
        child.add_argument("--checkpoint")
        child.add_argument(
            "--checkpoint-mode",
            choices=("trained", "base_untrained"),
            default="trained",
        )
        child.add_argument("--split", choices=("validation", "locked_test"), default="validation")
        child.add_argument("--device", default="cuda:0")
        child.add_argument("--torch-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
        child.add_argument("--learning-rate", type=float, default=2e-4)
        child.add_argument(
            "--lr-scheduler", choices=("constant", "cosine"), default="constant"
        )
        child.add_argument("--cosine-min-lr", type=float, default=0.0)
        child.add_argument("--weight-decay", type=float, default=0.01)
        child.add_argument("--max-grad-norm", type=float, default=1.0)
        child.add_argument("--epochs", type=int, default=3)
        child.add_argument("--max-steps", type=int, default=1 if command == "smoke" else 1000000)
        child.add_argument("--early-stopping-patience", type=int, default=0)
        child.add_argument("--early-stopping-min-epochs", type=int, default=0)
        child.add_argument("--early-stopping-min-delta", type=float, default=0.0)
        child.add_argument("--log-training-metrics-per-epoch", action="store_true")
        child.add_argument("--save-best-checkpoint-per-epoch", action="store_true")
        child.add_argument("--seed", type=int, default=42)
        child.add_argument("--stage", choices=("stage0", "stage1", "stage2"), default="stage2")
        child.add_argument("--gradient-diagnostics-interval", type=int, default=50)
        child.add_argument("--train-evidence-count", type=int, default=60)
        child.add_argument("--validation-evidence-count", type=int, default=10)
        child.add_argument("--locked-test-evidence-count", type=int, default=0)
        child.add_argument(
            "--evaluation-provenance",
            choices=("checkpoint_split", "external_holdout"),
            default="checkpoint_split",
        )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
