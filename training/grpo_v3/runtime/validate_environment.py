"""Validate the pinned no-vLLM MS-SWIFT environment used by score-only GRPO."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


EXPECTED_DISTRIBUTIONS = {
    "ms-swift": "4.2.2",
    "transformers": "5.8.1",
    "peft": "0.19.1",
    "datasets": "4.8.4",
    "trl": "0.29.1",
}


def validate_environment(expected_env: str | Path) -> dict[str, Any]:
    expected_prefix = Path(expected_env).resolve()
    active_prefix = Path(sys.prefix).resolve()
    versions: dict[str, str | None] = {}
    errors: list[str] = []
    for distribution, expected_version in EXPECTED_DISTRIBUTIONS.items():
        try:
            actual_version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            actual_version = None
        versions[distribution] = actual_version
        if actual_version != expected_version:
            errors.append(
                f"{distribution} expected {expected_version}, found {actual_version}"
            )

    if active_prefix != expected_prefix:
        errors.append(
            f"active Python prefix {active_prefix} does not match {expected_prefix}"
        )

    # This environment intentionally has no vLLM. Importing the GRPO trainer is
    # the executable contract for the lazy no-vLLM MS-SWIFT fix.
    vllm_present = importlib.util.find_spec("vllm") is not None
    if vllm_present:
        errors.append("vllm must remain absent in the pinned no-vLLM environment")

    imports: dict[str, str] = {}
    for module_name in (
        "torchcodec.decoders._video_decoder",
        "qwen_vl_utils",
        "swift.rlhf_trainers.grpo_trainer",
    ):
        try:
            module = importlib.import_module(module_name)
            imports[module_name] = str(Path(module.__file__).resolve())
        except Exception as exc:  # pragma: no cover - exercised on HPC
            imports[module_name] = f"{type(exc).__name__}: {exc}"
            errors.append(f"failed to import {module_name}: {type(exc).__name__}: {exc}")

    scratch_variables = (
        "HOME",
        "PIP_CACHE_DIR",
        "XDG_CACHE_HOME",
        "HF_HOME",
        "HF_DATASETS_CACHE",
        "MODELSCOPE_CACHE",
        "TORCH_HOME",
        "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "CUDA_CACHE_PATH",
        "VLLM_CACHE_ROOT",
        "FLASHINFER_WORKSPACE_BASE",
        "TMPDIR",
        "TMP",
        "TEMP",
    )
    storage = {name: os.environ.get(name) for name in scratch_variables}
    non_scratch = [
        name
        for name, value in storage.items()
        if not value or not Path(value).is_absolute() or not str(value).startswith("/scratch/")
    ]
    if non_scratch:
        errors.append(f"storage variables are not scratch-routed: {non_scratch}")

    return {
        "schema_version": "egoqa_grpo_environment_contract_v1",
        "status": "passed" if not errors else "failed",
        "active_prefix": str(active_prefix),
        "expected_prefix": str(expected_prefix),
        "python": str(Path(sys.executable).resolve()),
        "versions": versions,
        "vllm_present": vllm_present,
        "imports": imports,
        "storage": storage,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-env", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate_environment(args.expected_env)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
