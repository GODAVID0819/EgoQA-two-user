#!/bin/bash
#SBATCH --job-name=hf_upload_curated_pass
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/tx856/multi_user_qa/dual_full120/logs/%x_%j.out
# 公开上传 human_review/curated_pass（169 题 + 480 段视频）到 HF dataset tianxia2/egolife-six-user-mcqa-human-pass
set -euo pipefail
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/tx856/.conda/envs/grpo-swift
REPO=tianxia2/egolife-six-user-mcqa-human-pass
DIR=/scratch/tx856/multi_user_qa/dual_full120/hf_upload/egolife_six_user_mcqa_human_pass
python -c "from huggingface_hub import create_repo; print(create_repo('$REPO', repo_type='dataset', private=False, exist_ok=True))"
hf upload-large-folder "$REPO" "$DIR" --repo-type dataset --num-workers 8
python - <<PY
from huggingface_hub import HfApi
files = HfApi().list_repo_files('$REPO', repo_type='dataset')
print('remote files', len(files), 'mp4', sum(f.endswith('.mp4') for f in files))
assert sum(f.endswith('.mp4') for f in files) == 480
PY
echo UPLOAD_DONE
