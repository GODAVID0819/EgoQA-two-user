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
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError(f"split {split} must contain unique evidence IDs")
    by_id = {record.evidence_id: record for record in records}
    missing = sorted(set(ids) - set(by_id))
    if missing:
        raise ValueError(f"split {split} references absent evidence IDs: {missing}")
    return tuple(by_id[evidence_id] for evidence_id in ids)


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
                    labels[field].append(candidate.labels()[field])
                    probabilities[field].append(torch.softmax(logits[0].float(), dim=-1).cpu().tolist())
                    loss_values[field].append(float(loss.detach().float().cpu()))
    return {
        field: classification_metrics(
            labels[field], probabilities[field], loss=sum(loss_values[field]) / len(loss_values[field])
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
        if contract.get("stage") != config.stage:
            raise ValueError("checkpoint stage mismatch")
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
    initial = {
        name: parameter.detach().cpu().clone()
        for name, parameter in reviewer.named_parameters() if parameter.requires_grad
    }
    history = []
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
        "status": "passed", "mode": args.command, "global_step": global_step,
        "stage": config.stage,
        "active_heads": list(config.active_heads),
        "stage0_framework_validation": config.stage == "stage0",
        "head_parameter_delta_nonzero": head_changed,
        "lora_parameter_delta_nonzero": lora_changed,
        "gradient_routes": gradient_routes,
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
        child.add_argument("--train-evidence-count", type=int, default=40)
        child.add_argument("--validation-evidence-count", type=int, default=10)
        child.add_argument("--locked-test-evidence-count", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
