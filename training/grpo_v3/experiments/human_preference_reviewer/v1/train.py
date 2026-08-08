"""Supervised training, smoke, validation, and locked-test CLI for Reviewer v1."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .checkpoint import (
    load_checkpoint_contract,
    load_classification_heads,
    load_lora_adapter,
    save_checkpoint,
)
from .config import ReviewerV1Config
from .data import EvidenceRecord, load_annotation_csv, sha256_file
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
    backbone.to(device)
    backbone.config.use_cache = False
    indices = target_layer_indices(backbone, last_n=config.last_n_shared_blocks)
    backbone, targets = inject_reviewer_lora(backbone, config)
    reviewer = ReviewerV1(backbone, resolve_hidden_size(backbone.config)).to(device)
    return reviewer, processor, indices, targets


def _gradient_route_audit(reviewer: Any, losses: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    named = [(name, parameter) for name, parameter in reviewer.named_parameters() if parameter.requires_grad]
    loss_names = {
        "evidence_loss": "evidence_head.",
        "answerability_loss": "answerability_head.",
        "formality_loss": "formality_head.",
    }
    result: dict[str, Any] = {}
    for loss_name, direct_prefix in loss_names.items():
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
        passed = bool(direct) and not other_heads and bool(lora)
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
) -> dict[str, Any]:
    import torch

    labels = {field: [] for field in ("evidence_quality", "answerability", "qa_formality")}
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
                losses = reviewer_losses(output, _labels(candidate, torch_module=torch, device=device))
                pairs = (
                    ("evidence_quality", output.evidence_logits, losses["evidence_loss"]),
                    ("answerability", output.answerability_logits, losses["answerability_loss"]),
                    ("qa_formality", output.formality_logits, losses["formality_loss"]),
                )
                for field, logits, loss in pairs:
                    labels[field].append(candidate.labels()[field])
                    probabilities[field].append(torch.softmax(logits[0].float(), dim=-1).cpu().tolist())
                    loss_values[field].append(float(loss.detach().float().cpu()))
    return {
        field: classification_metrics(
            labels[field], probabilities[field], loss=sum(loss_values[field]) / len(loss_values[field])
        )
        for field in labels
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from qwen_vl_utils import process_vision_info

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    audit = load_annotation_csv(args.csv)
    media_map = load_media_map(args.media_map)
    manifest = json.loads(Path(args.split_manifest).read_text(encoding="utf-8")) if args.split_manifest else None
    if manifest and manifest.get("csv_sha256") not in {None, audit.csv_sha256}:
        raise ValueError("split manifest CSV hash does not match annotation CSV")
    config = ReviewerV1Config(model_name_or_path=args.model, seed=args.seed)
    reviewer, processor, indices, targets = _load_runtime(
        model_path=args.model, device=args.device, dtype_name=args.torch_dtype, config=config
    )
    param_audit = parameter_audit(reviewer, expected_layer_indices=indices)
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
        if contract["csv_sha256"] != audit.csv_sha256:
            raise ValueError("checkpoint CSV hash mismatch")
        load_classification_heads(reviewer, args.checkpoint)
        load_lora_adapter(reviewer, args.checkpoint)
        metrics = _evaluate(
            reviewer, select_evidence(audit.eligible_evidence, manifest, args.split),
            media_map=media_map, processor=processor, process_vision_info=process_vision_info,
            device=args.device,
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
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        candidates = [(evidence, candidate) for evidence in records for candidate in evidence.candidates]
        random.Random(args.seed + epoch).shuffle(candidates)
        for evidence, candidate in candidates:
            output = reviewer(**_encoded_inputs(
                evidence, candidate, media_map=media_map, processor=processor,
                process_vision_info=process_vision_info, device=args.device,
            ))
            losses = reviewer_losses(output, _labels(candidate, torch_module=torch, device=args.device))
            if gradient_routes is None and args.command == "smoke":
                gradient_routes = _gradient_route_audit(reviewer, losses)
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
    head_changed = all(any(name.startswith(prefix) and delta > 0 for name, delta in deltas.items()) for prefix in (
        "evidence_head.", "answerability_head.", "formality_head.",
    ))
    lora_changed = any((".lora_A." in name or ".lora_B." in name) and delta > 0 for name, delta in deltas.items())
    if not history or not head_changed or not lora_changed or any(not math.isfinite(row["loss"]) for row in history):
        raise RuntimeError("training update gate failed")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_hash = sha256_file(args.split_manifest) if args.split_manifest else "SMOKE_NO_SPLIT"
    save_checkpoint(
        reviewer, output_dir / "checkpoint", config={
            "contract_version": "human_preference_reviewer_absolute_v1",
            **config.to_dict(), "actual_lora_targets": list(targets),
        }, csv_sha256=audit.csv_sha256, split_sha256=split_hash,
        parameter_audit=param_audit, optimizer=optimizer,
        trainer_state={"global_step": global_step}, processor=processor,
    )
    result = {
        "status": "passed", "mode": args.command, "global_step": global_step,
        "head_parameter_delta_nonzero": head_changed,
        "lora_parameter_delta_nonzero": lora_changed,
        "gradient_routes": gradient_routes,
        "parameter_audit": param_audit,
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
