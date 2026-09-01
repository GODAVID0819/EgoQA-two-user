#!/usr/bin/env bash
#SBATCH --job-name=download_qwen38_27b
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err

set -euo pipefail

MODEL_DIR="/scratch/xl6775/models/Qwen3.8-27B"
HF_CLI="/scratch/xl6775/conda/envs/qwen3vl-smoke/bin/hf"
PYTHON="/scratch/xl6775/conda/envs/qwen3vl-smoke/bin/python"
JOB_SCRATCH_ROOT="/scratch/xl6775/job_scratch/qwen38_download_${SLURM_JOB_ID}"
RESULT_ROOT="/scratch/xl6775/models/download_manifests"
RESULT_PATH="${RESULT_ROOT}/qwen38_download_${SLURM_JOB_ID}/download_result.json"

export HOME="${JOB_SCRATCH_ROOT}/home"
export XDG_CACHE_HOME="${JOB_SCRATCH_ROOT}/xdg_cache"
export HF_HOME="${JOB_SCRATCH_ROOT}/hf_home"
export HF_HUB_CACHE="${JOB_SCRATCH_ROOT}/hf_hub"
export TRANSFORMERS_CACHE="${JOB_SCRATCH_ROOT}/transformers"
export TORCH_HOME="${JOB_SCRATCH_ROOT}/torch"
export TMPDIR="${JOB_SCRATCH_ROOT}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export CONDA_NO_PLUGINS=true
export CONDA_SOLVER=classic
mkdir -p \
  "${HOME}" "${XDG_CACHE_HOME}" "${HF_HOME}" "${HF_HUB_CACHE}" \
  "${TRANSFORMERS_CACHE}" "${TORCH_HOME}" "${TMPDIR}" \
  "${MODEL_DIR}" "$(dirname "${RESULT_PATH}")"

test -x "${HF_CLI}"
test -x "${PYTHON}"

"${HF_CLI}" download Qwen/Qwen3.8-27B --local-dir "${MODEL_DIR}"

"${PYTHON}" - "${MODEL_DIR}" "${RESULT_PATH}" "${SLURM_JOB_ID}" <<'PY'
import json
import sys
from pathlib import Path

from transformers import AutoConfig, AutoProcessor

model_dir = Path(sys.argv[1])
result_path = Path(sys.argv[2])
job_id = sys.argv[3]
expected_shards = 18
required_files = (
    "config.json",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
missing = [name for name in required_files if not (model_dir / name).is_file()]
shards = sorted(model_dir.glob("model-*-of-*.safetensors"))
empty_shards = [path.name for path in shards if path.stat().st_size <= 0]
if missing or len(shards) != expected_shards or empty_shards:
    raise SystemExit(
        f"model download incomplete missing={missing} shard_count={len(shards)} "
        f"empty_shards={empty_shards}"
    )
config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
if str(getattr(config, "model_type", "")) != "qwen3_5":
    raise SystemExit(f"unexpected model_type={getattr(config, 'model_type', None)!r}")
architectures = list(getattr(config, "architectures", None) or [])
if "Qwen3_5ForConditionalGeneration" not in architectures:
    raise SystemExit(f"unexpected architectures={architectures!r}")
result = {
    "status": "passed",
    "job_id": job_id,
    "model_dir": str(model_dir),
    "model_type": config.model_type,
    "architectures": architectures,
    "processor_class": type(processor).__name__,
    "weight_shard_count": len(shards),
    "weight_bytes": sum(path.stat().st_size for path in shards),
}
result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print("QWEN38_DOWNLOAD_AUDIT_PASSED", json.dumps(result, sort_keys=True), flush=True)
PY

printf 'DOWNLOAD_RESULT=%s\n' "${RESULT_PATH}"
