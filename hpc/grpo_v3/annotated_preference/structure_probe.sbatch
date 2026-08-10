#!/usr/bin/env bash
#SBATCH --job-name=pareto-dpo-structure
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/xl6775/projects/EgoQA-two-user-annotated-pareto-dpo/logs/pareto-dpo-structure-%j.out
#SBATCH --error=/scratch/xl6775/projects/EgoQA-two-user-annotated-pareto-dpo/logs/pareto-dpo-structure-%j.err
set -euo pipefail

MODE=structure
PAIR_LIMIT=1
export MODE PAIR_LIMIT
PROJECT_ROOT=${PROJECT_ROOT:-/scratch/xl6775/projects/EgoQA-two-user-annotated-pareto-dpo}
source "${PROJECT_ROOT}/hpc/grpo_v3/annotated_preference/common.sh"
test -s "${DPO_DATA_DIR}/train_dpo.jsonl"

echo "SLURM_JOB_ID=${SLURM_JOB_ID} OUTDIR=${OUTDIR} PAIR_LIMIT=${PAIR_LIMIT}"
"${PYTHON}" - "${DPO_DATA_DIR}/train_dpo.jsonl" "${MODEL_DIR}" \
  "${OUTDIR}/dataset_preview.json" "${OUTDIR}/structure_probe.json" <<'PY'
import importlib.metadata
import json
import sys
from pathlib import Path

from transformers import AutoProcessor

dataset, model_dir, preview_path, result_path = map(Path, sys.argv[1:])
row = json.loads(dataset.open("r", encoding="utf-8").readline())
messages = row["messages"]
chosen = messages[-1]["content"]
rejected = row["rejected_response"]
videos = [Path(value) for value in row["videos"]]
prompt = messages[0]["content"]
processor = AutoProcessor.from_pretrained(
    str(model_dir), local_files_only=True, trust_remote_code=True
)
encoded = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
preview = {"pair_limit": 1, "messages": messages, "rejected_response": rejected, "videos": row["videos"]}
result = {
    "status": "passed",
    "model_dir": str(model_dir.resolve()),
    "video_token_count": prompt.count("<video>"),
    "local_video_count": len(videos),
    "local_videos_nonempty": all(path.is_absolute() and path.stat().st_size > 0 for path in videos),
    "chosen_label_nonempty": bool(chosen.strip()),
    "rejected_label_nonempty": bool(rejected.strip()),
    "encoded_token_count": len(encoded),
    "versions": {name: importlib.metadata.version(name) for name in ("ms-swift", "torch", "transformers", "torchcodec")},
}
preview_path.write_text(json.dumps(preview, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
if result["video_token_count"] != 2 or result["local_video_count"] != 2:
    raise SystemExit(2)
if not all(result[key] for key in ("local_videos_nonempty", "chosen_label_nonempty", "rejected_label_nonempty")):
    raise SystemExit(2)
PY

test -s "${OUTDIR}/storage_preflight.json"
test -s "${OUTDIR}/dependencies.txt"
test -s "${OUTDIR}/dataset_preview.json"
test -s "${OUTDIR}/structure_probe.json"
