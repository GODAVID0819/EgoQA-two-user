"""Zero-GPU data audit and Qwen3-VL module-structure probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .data import build_split_manifest, load_annotation_csv
from .lora import expected_lora_targets, locate_shared_language_layers


def annotation_audit_report(
    csv_path: str | Path,
    *,
    train_count: int,
    validation_count: int,
    locked_test_count: int,
    seed: int = 42,
) -> dict[str, Any]:
    audit = load_annotation_csv(csv_path)
    required = train_count + validation_count + locked_test_count
    eligible = len(audit.eligible_evidence)
    report = audit.to_dict()
    report["formal_split_gate"] = {
        "train_evidence_count": train_count,
        "validation_evidence_count": validation_count,
        "locked_test_evidence_count": locked_test_count,
        "required_evidence_count": required,
        "eligible_evidence_count": eligible,
        "missing_evidence_count": max(0, required - eligible),
        "status": "passed" if eligible >= required else "failed",
    }
    report["status"] = "passed" if eligible >= required else "insufficient_data"
    if eligible >= required:
        report["split_manifest"] = build_split_manifest(
            audit.eligible_evidence,
            train_count=train_count,
            validation_count=validation_count,
            locked_test_count=locked_test_count,
            seed=seed,
            csv_sha256=audit.csv_sha256,
        )
    return report


def structure_report(model_name_or_path: str, *, expected_layer_count: int = 36) -> dict[str, Any]:
    try:
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoModelForImageTextToText
    except ImportError as error:
        raise RuntimeError("transformers and accelerate are required for the structure probe") from error
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForImageTextToText.from_config(config, trust_remote_code=True)
    path, layers = locate_shared_language_layers(model)
    targets = expected_lora_targets(model, last_n=2, projections=("q_proj", "v_proj"))
    if len(layers) != expected_layer_count:
        raise RuntimeError(
            f"expected {expected_layer_count} shared language blocks; found {len(layers)} at {path}"
        )
    target_shapes = {}
    for target in targets:
        current: Any = model
        for component in target.split("."):
            current = current[int(component)] if component.isdigit() else getattr(current, component)
        target_shapes[target] = {
            "in_features": int(current.in_features),
            "out_features": int(current.out_features),
        }
    return {
        "status": "passed",
        "model_class": type(model).__name__,
        "shared_stack_path": path,
        "shared_layer_count": len(layers),
        "target_layer_indices": [len(layers) - 2, len(layers) - 1],
        "lora_targets": list(targets),
        "target_shapes": target_shapes,
        "vision_module_present": bool(getattr(model, "visual", None) is not None or getattr(getattr(model, "model", None), "visual", None) is not None),
    }


def _write(path: str | None, value: dict[str, Any]) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    annotation = subparsers.add_parser("annotation-csv", help="audit the human annotation CSV")
    annotation.add_argument("--csv", required=True)
    annotation.add_argument("--output")
    annotation.add_argument("--split-output")
    annotation.add_argument("--train-evidence-count", type=int, default=40)
    annotation.add_argument("--validation-evidence-count", type=int, default=10)
    annotation.add_argument("--locked-test-evidence-count", type=int, default=10)
    annotation.add_argument("--seed", type=int, default=42)
    annotation.add_argument("--require-formal-split", action="store_true")
    structure = subparsers.add_parser("structure", help="inspect Qwen3-VL shared blocks without loading weights")
    structure.add_argument("--model", required=True)
    structure.add_argument("--expected-layer-count", type=int, default=36)
    structure.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "annotation-csv":
        report = annotation_audit_report(
            args.csv,
            train_count=args.train_evidence_count,
            validation_count=args.validation_evidence_count,
            locked_test_count=args.locked_test_evidence_count,
            seed=args.seed,
        )
        _write(args.output, report)
        if args.split_output and "split_manifest" in report:
            _write(args.split_output, report["split_manifest"])
        return 2 if args.require_formal_split and report["status"] != "passed" else 0
    _write(args.output, structure_report(args.model, expected_layer_count=args.expected_layer_count))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
