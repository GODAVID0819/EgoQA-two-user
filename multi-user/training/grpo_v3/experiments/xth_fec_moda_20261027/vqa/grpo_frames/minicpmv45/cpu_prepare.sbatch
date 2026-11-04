#!/bin/bash
#SBATCH --job-name=minicpm45_cpu131
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/xl6775/experiments/vqa_minicpmv45_grpo131_20261031/logs/cpu_%j.out
set -euo pipefail
export VQA_ROOT=/scratch/xl6775/experiments/vqa_minicpmv45_grpo131_20261031
export VQA_ENV_BIN=/scratch/xl6775/envs/videollama3_grpo131_gpu_20261030/bin
export VQA_FFMPEG_BIN=/scratch/xl6775/envs/egoqa-ffmpeg-runtime/bin
export VQA_FFMPEG_LIB=/scratch/xl6775/envs/egoqa-ffmpeg-runtime/lib
cd "$VQA_ROOT/code"
source runtime_env.sh
MINICPM_ENV=/scratch/xl6775/envs/minicpmv45_grpo131_20261031
if [ ! -x "$MINICPM_ENV/bin/python" ]; then
  python -m venv "$MINICPM_ENV"
  "$MINICPM_ENV/bin/python" -c 'import sysconfig,pathlib;pathlib.Path(sysconfig.get_path("purelib"),"previous_gpu_packages.pth").write_text("/scratch/xl6775/envs/videollama3_grpo131_gpu_20261030/lib/python3.11/site-packages\n")'
fi
export PATH="$MINICPM_ENV/bin:$PATH"
python -m pip install --disable-pip-version-check transformers==4.51.3 tokenizers==0.21.4
python -m pip check
python - <<'PY'
import json,os,torch,flash_attn,transformers
from pathlib import Path
from huggingface_hub import snapshot_download
root=Path(os.environ['VQA_ROOT']);config=json.loads((root/'code/experiment_config.json').read_text())
print('ENVIRONMENT',torch.__version__,torch.version.cuda,transformers.__version__,flash_attn.__version__,flush=True)
snapshot_download(config['model'],revision=config['model_revision'],local_dir=str(root/'model'),
    allow_patterns=['*.json','*.py','*.txt','*.safetensors'],token=False,max_workers=4)
print('MODEL_ASSETS_READY',flush=True)
PY
export HF_HUB_OFFLINE=1 MINICPM_MODEL="$VQA_ROOT/model"
export FRAME_CACHE=/scratch/xl6775/experiments/vqa_videollama3_grpo131_20261030/frames_156
python -m unittest test_packing.py
python prepare_cpu.py
python - <<'PY'
import json,os
from pathlib import Path
from upstream.modeling_minicpmv import MiniCPMV
from upstream.configuration_minicpm import MiniCPMVConfig
root=Path(os.environ['VQA_ROOT']);config=MiniCPMVConfig.from_pretrained(root/'model')
assert config.max_position_embeddings==40960 and config.query_num==64
assert config._name_or_path and (root/'code/cpu_ready.json').is_file()
print('CPU_READY_ALL_393_NO_GPU',flush=True)
PY
