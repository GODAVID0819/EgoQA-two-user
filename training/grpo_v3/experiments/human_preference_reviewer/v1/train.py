"""Supervised training, smoke, validation, and locked-test CLI for Reviewer v1."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .checkpoint import (
    load_checkpoint_contract,
    load_classification_heads,
    load_lora_adapter,
    save_checkpoint,
    validate_checkpoint_runtime_contract,
)
from .config import ReviewerV1Config
from .data import EvidenceRecord, load_annotation_csv, sha256_file, validate_split_manifest
from .evaluation import classification_metrics
from .lora import inject_reviewer_lora, parameter_audit, target_layer_indices
from .losses import reviewer_losses
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
    if not ids:
        raise ValueError(f"split {split} is empty and cannot be evaluated or trained")
    if len(ids) != len(set(ids)):
        raise ValueError(f"split {split} must contain unique evidence IDs")
    by_id = {record.evidence_id: record for record in records}
    missing = sorted(set(ids) - set(by_id))
    if missing:
        raise ValueError(f"split {split} references absent evidence IDs: {missing}")
    return tuple(by_id[evidence_id] for evidence_id in ids)


def _candidate_observation(
    *,
    evidence_id: str,
    candidate_id: str,
    label: int,
    probabilities: Sequence[float],
    loss: float,
) -> dict[str, Any]:
    values = [float(value) for value in probabilities]
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError("candidate probabilities must contain three finite values")
    if label not in (1, 2, 3) or not math.isfinite(float(loss)):
        raise ValueError("candidate observation requires a valid grade and finite loss")
    return {
        "evidence_id": str(evidence_id),
        "candidate_id": str(candidate_id),
        "label": int(label),
        "prediction": max(range(3), key=values.__getitem__) + 1,
        "probabilities": values,
        "loss": float(loss),
    }


def _controlled_overfit_gate(
    pre_metrics: Mapping[str, Any],
    post_metrics: Mapping[str, Any],
    *,
    minimum_loss_reduction: float = 0.30,
    minimum_improved_ratio: float = 0.80,
    minimum_accuracy_gain: float = 0.20,
    minimum_prediction_classes: int = 2,
) -> dict[str, Any]:
    """Compare one fixed probe set before and after head-only training."""

    def observations(metrics: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
        rows = metrics.get("candidate_results")
        if not isinstance(rows, list) or not rows:
            raise ValueError("controlled overfit metrics require non-empty candidate_results")
        indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("candidate_results entries must be mappings")
            key = (str(row.get("evidence_id") or ""), str(row.get("candidate_id") or ""))
            if not all(key) or key in indexed:
                raise ValueError("candidate_results require unique evidence and candidate identities")
            indexed[key] = row
        return indexed

    pre_rows = observations(pre_metrics)
    post_rows = observations(post_metrics)
    if set(pre_rows) != set(post_rows):
        raise ValueError("pre/post metrics must contain the same candidate identities")

    per_candidate: list[dict[str, Any]] = []
    for key in sorted(pre_rows):
        pre_row = pre_rows[key]
        post_row = post_rows[key]
        if int(pre_row["label"]) != int(post_row["label"]):
            raise ValueError(f"pre/post label mismatch for candidate {key}")
        pre_loss = float(pre_row["loss"])
        post_loss = float(post_row["loss"])
        per_candidate.append({
            "evidence_id": key[0],
            "candidate_id": key[1],
            "label": int(pre_row["label"]),
            "pre_loss": pre_loss,
            "post_loss": post_loss,
            "loss_delta": post_loss - pre_loss,
            "improved": post_loss < pre_loss,
            "pre_prediction": int(pre_row["prediction"]),
            "post_prediction": int(post_row["prediction"]),
        })

    pre_loss = float(pre_metrics["loss"])
    post_loss = float(post_metrics["loss"])
    pre_accuracy = float(pre_metrics["accuracy"])
    post_accuracy = float(post_metrics["accuracy"])
    finite = all(math.isfinite(value) for value in (pre_loss, post_loss, pre_accuracy, post_accuracy))
    finite = finite and all(
        math.isfinite(float(row["pre_loss"])) and math.isfinite(float(row["post_loss"]))
        for row in per_candidate
    )
    loss_reduction_ratio = (
        (pre_loss - post_loss) / pre_loss
        if finite and pre_loss > 0 else 0.0
    )
    improved_count = sum(bool(row["improved"]) for row in per_candidate)
    improved_ratio = improved_count / len(per_candidate)
    accuracy_gain = post_accuracy - pre_accuracy
    prediction_class_count = len({int(row["post_prediction"]) for row in per_candidate})

    failures: list[str] = []
    if not finite:
        failures.append("nonfinite_pre_or_post_metric")
    if not post_loss < pre_loss:
        failures.append("post_loss_not_lower_than_pre_loss")
    if not loss_reduction_ratio >= minimum_loss_reduction:
        failures.append("loss_reduction_below_minimum")
    if not improved_ratio >= minimum_improved_ratio:
        failures.append("improved_candidate_ratio_below_minimum")
    if not accuracy_gain >= minimum_accuracy_gain:
        failures.append("accuracy_gain_below_minimum")
    if prediction_class_count < minimum_prediction_classes:
        failures.append("prediction_class_count_below_minimum")

    return {
        "passed": not failures,
        "thresholds": {
            "minimum_loss_reduction": minimum_loss_reduction,
            "minimum_improved_candidate_ratio": minimum_improved_ratio,
            "minimum_accuracy_gain": minimum_accuracy_gain,
            "minimum_prediction_classes": minimum_prediction_classes,
        },
        "measurements": {
            "candidate_count": len(per_candidate),
            "pre_train_loss": pre_loss,
            "post_train_loss": post_loss,
            "loss_reduction_ratio": loss_reduction_ratio,
            "improved_candidate_count": improved_count,
            "improved_candidate_ratio": improved_ratio,
            "pre_train_accuracy": pre_accuracy,
            "post_train_accuracy": post_accuracy,
            "accuracy_gain": accuracy_gain,
            "post_prediction_class_count": prediction_class_count,
        },
        "failures": failures,
        "per_candidate_pre_post": per_candidate,
    }


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


def _evaluate(
    reviewer: Any,
    records: Iterable[EvidenceRecord],
    *,
    media_map: Mapping[str, str], processor: Any, process_vision_info: Any, device: str,
    active_heads: tuple[str, ...],
) -> dict[str, Any]:
    import torch

    labels = {field: [] for field in active_heads}
    probabilities = {field: [] for field in labels}
    loss_values = {field: [] for field in labels}
    candidate_results = {field: [] for field in labels}
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
                )
                pairs = {
                    "evidence_quality": (output.evidence_logits, losses.get("evidence_loss")),
                    "answerability": (output.answerability_logits, losses.get("answerability_loss")),
                    "qa_formality": (output.formality_logits, losses.get("formality_loss")),
                }
                for field in active_heads:
                    logits, loss = pairs[field]
                    label = candidate.labels()[field]
                    probability = torch.softmax(logits[0].float(), dim=-1).cpu().tolist()
                    loss_value = float(loss.detach().float().cpu())
                    labels[field].append(label)
                    probabilities[field].append(probability)
                    loss_values[field].append(loss_value)
                    candidate_results[field].append(_candidate_observation(
                        evidence_id=evidence.evidence_id,
                        candidate_id=candidate.candidate_id,
                        label=label,
                        probabilities=probability,
                        loss=loss_value,
                    ))
    result: dict[str, Any] = {}
    for field in labels:
        metrics = classification_metrics(
            labels[field], probabilities[field], loss=sum(loss_values[field]) / len(loss_values[field])
        )
        metrics["candidate_results"] = candidate_results[field]
        metrics["prediction_counts"] = {
            str(grade): sum(row["prediction"] == grade for row in candidate_results[field])
            for grade in (1, 2, 3)
        }
        result[field] = metrics
    return result


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
    load_classification_heads(reviewer, checkpoint_dir)
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
        validate_split_manifest(
            manifest,
            expected_counts=(
                config.train_evidence_count,
                config.validation_evidence_count,
                config.locked_test_evidence_count,
            ),
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
    if args.command == "smoke":
        records = (audit.eligible_evidence[0],)
    elif args.command == "fit":
        if manifest is None:
            raise ValueError("fit requires --split-manifest")
        records = select_evidence(audit.eligible_evidence, manifest, "train")
    else:
        if manifest is None or not args.checkpoint:
            raise ValueError("evaluate requires --split-manifest and --checkpoint")
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
        if contract["csv_sha256"] != audit.csv_sha256:
            raise ValueError("checkpoint CSV hash mismatch")
        if contract.get("split_sha256") != sha256_file(args.split_manifest):
            raise ValueError("checkpoint split manifest hash mismatch")
        load_classification_heads(reviewer, args.checkpoint)
        if config.lora_enabled:
            load_lora_adapter(reviewer, args.checkpoint)
        metrics = _evaluate(
            reviewer, select_evidence(audit.eligible_evidence, manifest, args.split),
            media_map=media_map, processor=processor, process_vision_info=process_vision_info,
            device=args.device,
            active_heads=config.active_heads,
        )
        result = {"status": "passed", "split": args.split, "metrics": metrics, "parameter_audit": param_audit}
        _write_json(Path(args.output_dir) / "evaluation_result.json", result)
        return result
    controlled_probe = args.command == "fit" and config.stage == "stage0"
    probe_evidence_ids = [record.evidence_id for record in records] if controlled_probe else []
    probe_label_support: dict[str, dict[str, int]] = {}
    if controlled_probe:
        for field in config.active_heads:
            probe_label_support[field] = {
                str(grade): sum(
                    candidate.labels()[field] == grade
                    for evidence in records
                    for candidate in evidence.candidates
                )
                for grade in (1, 2, 3)
            }
        missing_probe_grades = [
            f"{field}:{grade}"
            for field, support in probe_label_support.items()
            for grade, count in support.items()
            if count == 0
        ]
        if missing_probe_grades:
            raise ValueError(
                "controlled Stage 0 probe must cover all grades; "
                f"missing support: {missing_probe_grades}"
            )
    pre_train_metrics = (
        _evaluate(
            reviewer, records,
            media_map=media_map, processor=processor, process_vision_info=process_vision_info,
            device=args.device, active_heads=config.active_heads,
        )
        if controlled_probe else None
    )
    initial = {
        name: parameter.detach().cpu().clone()
        for name, parameter in reviewer.named_parameters() if parameter.requires_grad
    }
    history = []
    epoch_probe_metrics: list[dict[str, Any]] = []
    gradient_routes = None
    global_step = 0
    reviewer.train()
    training_started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        candidates = [(evidence, candidate) for evidence in records for candidate in evidence.candidates]
        random.Random(args.seed + epoch).shuffle(candidates)
        for evidence, candidate in candidates:
            output = reviewer(**_encoded_inputs(
                evidence, candidate, media_map=media_map, processor=processor,
                process_vision_info=process_vision_info, device=args.device,
            ))
            losses = reviewer_losses(
                output,
                _labels(candidate, torch_module=torch, device=args.device),
                active_heads=config.active_heads,
            )
            if gradient_routes is None and args.command == "smoke":
                gradient_routes = _gradient_route_audit(
                    reviewer, losses, lora_enabled=config.lora_enabled
                )
            losses["loss"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            history.append({
                "step": global_step, "evidence_id": evidence.evidence_id,
                "candidate_id": candidate.candidate_id,
                **{name: float(value.detach().float().cpu()) for name, value in losses.items()},
                "grad_norm": float(grad_norm.detach().float().cpu()),
            })
            if global_step >= args.max_steps:
                break
        if controlled_probe:
            epoch_metrics = _evaluate(
                reviewer, records,
                media_map=media_map, processor=processor,
                process_vision_info=process_vision_info, device=args.device,
                active_heads=config.active_heads,
            )
            epoch_probe_metrics.append({
                "epoch": epoch + 1,
                "global_step": global_step,
                "metrics": {
                    field: {
                        "loss": metrics["loss"],
                        "accuracy": metrics["accuracy"],
                        "macro_f1": metrics["macro_f1"],
                        "prediction_counts": metrics["prediction_counts"],
                    }
                    for field, metrics in epoch_metrics.items()
                },
            })
            reviewer.train()
        if global_step >= args.max_steps:
            break
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    post_train_metrics = (
        _evaluate(
            reviewer, records,
            media_map=media_map, processor=processor, process_vision_info=process_vision_info,
            device=args.device, active_heads=config.active_heads,
        )
        if controlled_probe else None
    )
    controlled_overfit_gate = None
    per_candidate_pre_post: list[dict[str, Any]] = []
    if controlled_probe:
        field = config.active_heads[0]
        controlled_overfit_gate = _controlled_overfit_gate(
            pre_train_metrics[field], post_train_metrics[field]
        )
        per_candidate_pre_post = controlled_overfit_gate["per_candidate_pre_post"]
    validation_metrics = None
    if args.command == "fit" and manifest is not None:
        validation_metrics = _evaluate(
            reviewer, select_evidence(audit.eligible_evidence, manifest, "validation"),
            media_map=media_map, processor=processor, process_vision_info=process_vision_info,
            device=args.device,
            active_heads=config.active_heads,
        )
    split_hash = sha256_file(args.split_manifest) if args.split_manifest else "SMOKE_NO_SPLIT"
    save_checkpoint(
        reviewer, output_dir / "checkpoint", config={
            "contract_version": "human_preference_reviewer_absolute_v1",
            **config.to_dict(), "actual_lora_targets": list(targets),
        }, csv_sha256=audit.csv_sha256, split_sha256=split_hash,
        parameter_audit=param_audit, optimizer=optimizer,
        trainer_state={"global_step": global_step}, processor=processor,
        active_heads=config.active_heads,
        lora_enabled=config.lora_enabled,
    )
    checkpoint_reload = _verify_checkpoint_reload_in_place(
        reviewer, output_dir / "checkpoint", lora_enabled=config.lora_enabled
    )
    result = {
        "status": (
            "passed"
            if controlled_overfit_gate is None or controlled_overfit_gate["passed"]
            else "failed_controlled_overfit_gate"
        ),
        "mode": args.command, "global_step": global_step,
        "stage": config.stage,
        "active_heads": list(config.active_heads),
        "stage0_framework_validation": config.stage == "stage0",
        "head_parameter_delta_nonzero": head_changed,
        "lora_parameter_delta_nonzero": lora_changed,
        "gradient_routes": gradient_routes,
        "parameter_audit": param_audit,
        "validation_metrics": validation_metrics,
        "probe_evidence_ids": probe_evidence_ids,
        "probe_label_support": probe_label_support,
        "pre_train_metrics": pre_train_metrics,
        "post_train_metrics": post_train_metrics,
        "epoch_probe_metrics": epoch_probe_metrics,
        "per_candidate_pre_post": per_candidate_pre_post,
        "controlled_overfit_gate": controlled_overfit_gate,
        "throughput": {
            "elapsed_seconds": elapsed_seconds,
            "candidate_steps_per_hour": global_step * 3600.0 / max(elapsed_seconds, 1e-9),
        },
        "repeated_candidate_loss": repeated_candidate_loss,
        "validation_macro_f1_mean": (
            sum(value["macro_f1"] for value in validation_metrics.values()) / len(validation_metrics)
            if validation_metrics else None
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
        child.add_argument("--split", choices=("validation", "locked_test"), default="validation")
        child.add_argument("--device", default="cuda:0")
        child.add_argument("--torch-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
        child.add_argument("--learning-rate", type=float, default=2e-4)
        child.add_argument("--weight-decay", type=float, default=0.01)
        child.add_argument("--max-grad-norm", type=float, default=1.0)
        child.add_argument("--epochs", type=int, default=3)
        child.add_argument("--max-steps", type=int, default=1 if command == "smoke" else 1000000)
        child.add_argument("--seed", type=int, default=42)
        child.add_argument("--stage", choices=("stage0", "stage1", "stage2"), default="stage2")
        child.add_argument("--train-evidence-count", type=int, default=60)
        child.add_argument("--validation-evidence-count", type=int, default=10)
        child.add_argument("--locked-test-evidence-count", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
