#!/bin/bash
#SBATCH --job-name=hf_upload_json_only
#SBATCH --account=torch_pr_111_general
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/tx856/multi_user_qa/dual_full120/logs/%x_%j.out
# 先只传 JSON + README（帧/视频由各自的 upload-large-folder 任务继续传）：hard56 补 video_manifest.json，textwrong131 建 repo 并传 JSON
set -euo pipefail
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/tx856/.conda/envs/grpo-swift
cd /scratch/tx856/multi_user_qa/dual_full120/hf_upload
python - <<'PY'
from huggingface_hub import HfApi, create_repo, CommitOperationAdd
api = HfApi()
for repo, d in (('tianxia2/egolife-six-user-mcqa-hard56', 'egolife_six_user_mcqa_hard56'),
                ('tianxia2/egolife-six-user-mcqa-textwrong131', 'egolife_six_user_mcqa_textwrong131')):
    create_repo(repo, repo_type='dataset', private=False, exist_ok=True)
    names = ['README.md', 'vqa.json', 'vqa_shuffled.json', 'dataset.json', 'video_manifest.json']
    api.create_commit(repo, repo_type='dataset', commit_message='Add/update JSON metadata and README',
                      operations=[CommitOperationAdd(n, f'{d}/{n}') for n in names])
    files = set(api.list_repo_files(repo, repo_type='dataset'))
    assert set(names) <= files, (repo, set(names) - files)
    print(repo, 'ok', len(files), 'files')
PY
echo JSON_DONE
