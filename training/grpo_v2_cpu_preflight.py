"""无需 GPU 的 GRPO v2 远程 preflight。"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


EXPECTED_REVISION = "2026-07-14-multimodal-batch-preflight-v5"
FLASHINFER_GUARD = (
    'export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"'
)


class PreflightError(RuntimeError):
    """可操作的 preflight 失败。"""


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise PreflightError(f"{code}: {message}")


def load_first_packet(path: Path) -> dict[str, Any]:
    require(path.is_file() and path.stat().st_size > 0, "EVIDENCE_MISSING", str(path))
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                require(isinstance(value, dict), "EVIDENCE_SHAPE", "首行不是 JSON object")
                return value
    raise PreflightError("EVIDENCE_EMPTY: evidence JSONL 没有有效行")


def check_static_inputs(
    *,
    project_root: Path,
    model_path: Path,
    evidence_path: Path,
) -> dict[str, Any]:
    sbatch_path = project_root / "hpc" / "grpo_v2_lora_8b_smoke.sbatch"
    require(sbatch_path.is_file(), "SBATCH_MISSING", str(sbatch_path))
    script = sbatch_path.read_text(encoding="utf-8")
    require(
        f'SCRIPT_REVISION="{EXPECTED_REVISION}"' in script,
        "STALE_SBATCH",
        f"远程脚本不是 {EXPECTED_REVISION}",
    )
    require(
        re.search(r"VLLM_USE_FLASHINFER_SAMPLE(?!R)", script) is None,
        "SBATCH_VARIABLE_TYPO",
        "发现 VLLM_USE_FLASHINFER_SAMPLE；正确变量名末尾必须是 SAMPLER",
    )
    require(
        FLASHINFER_GUARD in script,
        "STALE_SBATCH",
        "缺少 VLLM_USE_FLASHINFER_SAMPLER=0 防护",
    )
    require(model_path.is_dir(), "MODEL_MISSING", str(model_path))
    packet = load_first_packet(evidence_path)
    require(bool(str(packet.get("evidence_id") or "").strip()), "EVIDENCE_SHAPE", "缺少 evidence_id")
    require(isinstance(packet.get("clips"), list) and packet["clips"], "EVIDENCE_SHAPE", "clips 为空")
    return {
        "script_revision": EXPECTED_REVISION,
        "sbatch": str(sbatch_path),
        "model_path": str(model_path),
        "evidence": str(evidence_path),
        "evidence_id": packet.get("evidence_id"),
        "clip_count": len(packet["clips"]),
    }


def run_checked(command: list[str], *, cwd: Path, name: str) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        env=os.environ.copy(),
    )
    output = (result.stdout + result.stderr).strip()
    require(result.returncode == 0, f"{name}_FAIL", output or f"exit={result.returncode}")
    return output


def check_training_environment(project_root: Path, model_path: Path) -> str:
    code = r'''
import importlib.metadata
import json
import torch, torchvision, transformers, trl, peft, datasets
from transformers import AutoConfig, AutoProcessor

model_path = __import__("sys").argv[1]
AutoConfig.from_pretrained(model_path, local_files_only=True)
AutoProcessor.from_pretrained(model_path, local_files_only=True)
print(json.dumps({
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "trl": trl.__version__,
    "peft": peft.__version__,
    "torchvision": torchvision.__version__,
    "bitsandbytes": importlib.metadata.version("bitsandbytes"),
    "cuda_available": torch.cuda.is_available(),
    "model_config": "ok",
    "processor": "ok",
}))
'''
    return run_checked(
        [sys.executable, "-c", code, str(model_path)],
        cwd=project_root,
        name="TRAIN_ENV",
    )


def check_vllm_environment(project_root: Path, inference_python: Path) -> str:
    require(inference_python.is_file(), "VLLM_PYTHON_MISSING", str(inference_python))
    code = r'''
import json, os, shutil
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
import vllm
from vllm import envs
value = envs.VLLM_USE_FLASHINFER_SAMPLER
if value is not False:
    raise SystemExit(f"VLLM_USE_FLASHINFER_SAMPLER 未解析为 False: {value!r}")
print(json.dumps({
    "vllm": vllm.__version__,
    "flashinfer_sampler": value,
    "ninja": shutil.which("ninja"),
}))
'''
    return run_checked(
        [str(inference_python), "-c", code],
        cwd=project_root,
        name="VLLM_ENV",
    )


def check_dataset_route(
    project_root: Path,
    evidence_path: Path,
    model_path: Path,
    processor_batch_size: int,
    policy_media_mode: str,
    max_policy_frames_per_clip: int,
) -> str:
    code = r'''
import json, sys
from pathlib import Path
from training.grpo_v2_lora import build_training_rows, read_jsonl
from transformers import AutoProcessor

rows = build_training_rows(
    read_jsonl(sys.argv[1]),
    max_prompts=1,
    question_type="commonality",
    policy_media_mode=sys.argv[4],
    max_policy_frames_per_clip=int(sys.argv[5]),
)
content = rows[0]["prompt"][0]["content"]
media = []
for item in content:
    if item.get("type") == "image":
        media.append(item.get("image"))
    elif item.get("type") == "video":
        media.append(item.get("video"))
missing = [path for path in media if not path or not Path(path).is_file()]
if not media:
    raise SystemExit("policy prompt 没有 image/video")
if missing:
    raise SystemExit("policy media 路径缺失: " + json.dumps(missing, ensure_ascii=False))
processor = AutoProcessor.from_pretrained(sys.argv[2], local_files_only=True)
batch_size = int(sys.argv[3])
batch_prompts = [rows[0]["prompt"] for _ in range(batch_size)]
tokenized = processor.apply_chat_template(
    conversation=batch_prompts,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)
if "input_ids" not in tokenized or tokenized["input_ids"].numel() == 0:
    raise SystemExit("processor 没有生成 input_ids")
print(json.dumps({
    "evidence_id": rows[0]["evidence_id"],
    "media_count": len(media),
    "media": media,
    "input_tokens": int(tokenized["input_ids"].numel()),
    "processor_batch_size": batch_size,
    "policy_media_mode": sys.argv[4],
    "status": "PROCESSOR_BATCH_MULTIMODAL_OK",
}, ensure_ascii=False))
'''
    return run_checked(
        [
            sys.executable,
            "-c",
            code,
            str(evidence_path),
            str(model_path),
            str(processor_batch_size),
            policy_media_mode,
            str(max_policy_frames_per_clip),
        ],
        cwd=project_root,
        name="DATASET_ROUTE",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument(
        "--inference-python",
        default="/scratch/xl6775/envs/egoqa-grpo/bin/python",
    )
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--processor-batch-size", type=int, default=4)
    parser.add_argument(
        "--policy-media-mode",
        choices=("sampled_frames", "native_video"),
        default="sampled_frames",
    )
    parser.add_argument("--max-policy-frames-per-clip", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = Path(args.project_root).resolve()
    model_path = Path(args.model_path).resolve()
    evidence_path = Path(args.evidence).resolve()
    try:
        summary = check_static_inputs(
            project_root=project_root,
            model_path=model_path,
            evidence_path=evidence_path,
        )
        print(json.dumps(summary, ensure_ascii=False))
        if args.static_only:
            print("CPU_PREFLIGHT_STATIC_OK")
            return 0

        print(check_training_environment(project_root, model_path))
        print(check_vllm_environment(project_root, Path(args.inference_python)))
        require(
            args.processor_batch_size >= 2,
            "BATCH_SIZE_INVALID",
            "processor-batch-size 必须 >= 2",
        )
        require(
            args.max_policy_frames_per_clip >= 1,
            "FRAME_LIMIT_INVALID",
            "max-policy-frames-per-clip 必须 >= 1",
        )
        print(
            check_dataset_route(
                project_root,
                evidence_path,
                model_path,
                args.processor_batch_size,
                args.policy_media_mode,
                args.max_policy_frames_per_clip,
            )
        )
        print(
            run_checked(
                [sys.executable, "-m", "unittest", "tests.training.test_grpo_v2_lora", "-v"],
                cwd=project_root,
                name="UNIT_TESTS",
            )
        )
        print("CPU_PREFLIGHT_OK")
        return 0
    except (OSError, ValueError, json.JSONDecodeError, PreflightError) as exc:
        print(f"CPU_PREFLIGHT_FAIL {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
