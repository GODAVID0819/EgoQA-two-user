"""在加载模型前验证 Torch 作业的 scratch-first 存储环境。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


REQUIRED_STORAGE_VARIABLES = (
    "HOME", "XDG_CACHE_HOME", "HF_HOME", "HF_DATASETS_CACHE", "MODELSCOPE_CACHE",
    "TORCH_HOME", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "VLLM_CACHE_ROOT",
    "CUDA_CACHE_PATH", "FLASHINFER_WORKSPACE_BASE", "TMPDIR", "TMP", "TEMP",
)


def _entry(variable: str, raw_path: str, allowed_root: Path) -> dict[str, Any]:
    path = Path(raw_path) if raw_path else Path()
    absolute = bool(raw_path) and path.is_absolute()
    resolved = path.resolve(strict=False) if absolute else path
    within_allowed_root = absolute and resolved.is_relative_to(allowed_root)
    writable = False
    error: str | None = None
    total_bytes: int | None = None
    free_bytes: int | None = None
    probe: Path | None = None
    if not raw_path:
        error = "environment variable is unset"
    elif not absolute:
        error = "path is not absolute"
    elif not within_allowed_root:
        error = f"path is outside allowed root: {allowed_root}"
    else:
        try:
            resolved.mkdir(parents=True, exist_ok=True)
            probe = resolved / f".egoqa-storage-probe-{uuid.uuid4().hex}"
            probe.write_bytes(b"ok")
            usage = shutil.disk_usage(resolved)
            total_bytes = int(usage.total)
            free_bytes = int(usage.free)
            writable = True
        except OSError as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if probe is not None and probe.is_file():
                probe.unlink()
    return {
        "variable": variable, "path": raw_path,
        "resolved_path": str(resolved) if absolute else None,
        "absolute": absolute, "within_allowed_root": within_allowed_root,
        "writable": writable, "filesystem_total_bytes": total_bytes,
        "filesystem_free_bytes": free_bytes, "error": error,
    }


def validate_storage_environment(
    *,
    allowed_root: Path,
    environ: Mapping[str, str],
    required_variables: Sequence[str] = REQUIRED_STORAGE_VARIABLES,
) -> dict[str, Any]:
    allowed = allowed_root.resolve(strict=False)
    paths = [_entry(name, str(environ.get(name) or ""), allowed) for name in required_variables]
    failed_checks = [
        str(entry["variable"])
        for entry in paths
        if not (entry["absolute"] and entry["within_allowed_root"] and entry["writable"])
    ]
    return {
        "schema_version": "torch_storage_preflight_v1",
        "status": "passed" if not failed_checks else "failed",
        "allowed_root": str(allowed),
        "required_variables": list(required_variables),
        "failed_checks": failed_checks,
        "paths": paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Torch scratch-first 存储预检")
    parser.add_argument("--allowed-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate_storage_environment(allowed_root=args.allowed_root, environ=os.environ)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
