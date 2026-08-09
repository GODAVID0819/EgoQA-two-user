#!/usr/bin/env bash
#SBATCH --job-name=reviewer-media
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/xl6775/projects/EgoQA-two-user-reviewer-v1/logs/reviewer-media-%j.out
#SBATCH --error=/scratch/xl6775/projects/EgoQA-two-user-reviewer-v1/logs/reviewer-media-%j.err
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/xl6775/projects/EgoQA-two-user-reviewer-v1}"
TRAIN_ENV="${TRAIN_ENV:-/scratch/xl6775/envs/egoqa-ms-swift-v4.2.2-vllm024}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data_RLHF/reviewer_v1}"
CSV_PATH="${CSV_PATH:-${DATA_DIR}/rlhf_candidate_scores_merged_70_packets.csv}"
DATASET_ROOT="${DATASET_ROOT:-/scratch/xl6775/datasets/EgoLife}"
MEDIA_MAP="${MEDIA_MAP:-${DATA_DIR}/media_map.json}"
REQUIRED_LIST="${DATA_DIR}/required_media_paths.txt"
PYTHON="${TRAIN_ENV}/bin/python"

mkdir -p "${PROJECT_ROOT}/logs" "${DATA_DIR}" "${DATASET_ROOT}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

"${PYTHON}" - "${CSV_PATH}" "${REQUIRED_LIST}" <<'PY'
import csv
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
marker = "/resolve/main/"
paths = set()
with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
    for row in csv.DictReader(handle):
        for column in ("video_1_source", "video_2_source"):
            source = (row.get(column) or "").strip()
            if marker not in source:
                raise ValueError(f"unsupported media source: {source}")
            paths.add(source.split(marker, 1)[1])
output_path.write_text("".join(f"{path}\n" for path in sorted(paths)), encoding="utf-8")
print(f"required_videos={len(paths)}")
PY

"${PYTHON}" - "${REQUIRED_LIST}" "${DATASET_ROOT}" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

required = Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
snapshot_download(
    repo_id="lmms-lab/EgoLife",
    repo_type="dataset",
    local_dir=sys.argv[2],
    allow_patterns=required,
)
PY

"${PYTHON}" -m training.grpo_v3.experiments.human_preference_reviewer.v1.audit media-map \
  --csv "${CSV_PATH}" --dataset-root "${DATASET_ROOT}" --output "${MEDIA_MAP}"

"${PYTHON}" - "${REQUIRED_LIST}" "${MEDIA_MAP}" <<'PY'
import json
import sys
from pathlib import Path

required = Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
media_map = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
assert len(required) == 140, len(required)
assert len(media_map) == 140, len(media_map)
assert all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in media_map.values())
print("MEDIA_GATE_PASSED required=140 mapped=140")
PY
