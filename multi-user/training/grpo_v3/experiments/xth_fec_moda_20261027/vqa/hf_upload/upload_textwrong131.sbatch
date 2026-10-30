#!/bin/bash
#SBATCH --job-name=hf_upload_textwrong131
#SBATCH --account=torch_pr_111_general
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/tx856/multi_user_qa/dual_full120/logs/%x_%j.out
# 公开上传 textwrong131（131 题 + 89255 帧 + 414 段完整视频）到 HF dataset tianxia2/egolife-six-user-mcqa-textwrong131
set -euo pipefail
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/tx856/.conda/envs/grpo-swift
REPO=tianxia2/egolife-six-user-mcqa-textwrong131
DIR=/scratch/tx856/multi_user_qa/dual_full120/hf_upload/egolife_six_user_mcqa_textwrong131
cd /scratch/tx856/multi_user_qa/dual_full120/hf_upload
python stage_textwrong131.py "$DIR"
python add_videos.py "$DIR" /scratch/tx856/multi_user_qa/vqa_eval/grpo_frames/subsets/textwrong_any131.json
V=$(find "$DIR/videos" -name '*.mp4' | wc -l); echo "local mp4 $V"; [ "$V" = 414 ]
N=$(find "$DIR/frames" -name '*.jpg' | wc -l); echo "local jpg $N"; [ "$N" = 89255 ]
python -c "from huggingface_hub import create_repo; print(create_repo('$REPO', repo_type='dataset', private=False, exist_ok=True))"
hf upload-large-folder "$REPO" "$DIR" --repo-type dataset --num-workers 8
python - <<PY
from huggingface_hub import HfApi
files = HfApi().list_repo_files('$REPO', repo_type='dataset')
n = sum(f.endswith('.jpg') for f in files)
print('remote files', len(files), 'jpg', n)
assert n == 89255 and sum(f.endswith('.mp4') for f in files) == 414 and {'vqa.json', 'vqa_shuffled.json', 'dataset.json', 'README.md'} <= set(files)
PY
echo UPLOAD_DONE
