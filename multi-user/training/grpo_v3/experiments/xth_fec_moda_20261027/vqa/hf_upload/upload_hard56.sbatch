#!/bin/bash
#SBATCH --job-name=hf_upload_hard56
#SBATCH --account=torch_pr_111_general
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/tx856/multi_user_qa/dual_full120/logs/%x_%j.out
# 公开上传 hard56（56 题 + 49113 帧）到 HF dataset tianxia2/egolife-six-user-mcqa-hard56
set -euo pipefail
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/tx856/.conda/envs/grpo-swift
REPO=tianxia2/egolife-six-user-mcqa-hard56
DIR=/scratch/tx856/multi_user_qa/dual_full120/hf_upload/egolife_six_user_mcqa_hard56
cd /scratch/tx856/multi_user_qa/dual_full120/hf_upload
python stage_hard56.py "$DIR"
N=$(find "$DIR/frames" -name '*.jpg' | wc -l); echo "local jpg $N"; [ "$N" = 49113 ]
python -c "from huggingface_hub import create_repo; print(create_repo('$REPO', repo_type='dataset', private=False, exist_ok=True))"
hf upload-large-folder "$REPO" "$DIR" --repo-type dataset --num-workers 8
python - <<PY
from huggingface_hub import HfApi
files = HfApi().list_repo_files('$REPO', repo_type='dataset')
n = sum(f.endswith('.jpg') for f in files)
print('remote files', len(files), 'jpg', n)
assert n == 49113 and {'vqa.json', 'vqa_shuffled.json', 'dataset.json', 'README.md'} <= set(files)
PY
echo UPLOAD_DONE
