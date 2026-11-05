#!/bin/bash
# 以后在Torch的Slurm作业内source；本文件不连接、不提交任何作业。
set -euo pipefail
: "${SLURM_JOB_ID:?仅在Slurm作业中调用}"
: "${VQA_ROOT:?填写本实验scratch根目录}"
: "${VQA_ENV_BIN:?填写专用VideoLLaMA3环境bin目录}"
: "${VQA_FFMPEG_BIN:?填写已核实的FFmpeg bin目录}"
: "${VQA_FFMPEG_LIB:?填写已核实的FFmpeg lib目录}"
export VQA_JOBROOT="$VQA_ROOT/runtime/$SLURM_JOB_ID"
mkdir -p "$VQA_JOBROOT"
export HOME="$VQA_JOBROOT/home" XDG_CACHE_HOME="$VQA_JOBROOT/xdg" HF_HOME="$VQA_JOBROOT/hf"
export HF_HUB_CACHE="$VQA_JOBROOT/hf/hub" HF_DATASETS_CACHE="$VQA_JOBROOT/hf/datasets"
export HF_TOKEN_PATH="$VQA_JOBROOT/hf/token" HF_XET_CACHE="$VQA_JOBROOT/hf/xet"
export HF_MODULES_CACHE="$VQA_JOBROOT/hf/modules" MODELSCOPE_CACHE="$VQA_JOBROOT/modelscope"
export TORCH_HOME="$VQA_JOBROOT/torch" TRITON_CACHE_DIR="$VQA_JOBROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$VQA_JOBROOT/inductor" VLLM_CACHE_ROOT="$VQA_JOBROOT/vllm"
export CUDA_CACHE_PATH="$VQA_JOBROOT/cuda" FLASHINFER_WORKSPACE_BASE="$VQA_JOBROOT/flashinfer"
export TMPDIR="$VQA_JOBROOT/tmp" TMP="$VQA_JOBROOT/tmp" TEMP="$VQA_JOBROOT/tmp"
export PIP_CACHE_DIR="$VQA_JOBROOT/pip"
export HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_XET=1 VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export DECORD_EOF_RETRY_MAX=65536
export PATH="$VQA_FFMPEG_BIN:$VQA_ENV_BIN:$PATH"
export LD_LIBRARY_PATH="$VQA_FFMPEG_LIB:${LD_LIBRARY_PATH:-}"
python - <<'PY'
import json, os, shutil
from pathlib import Path
root = Path(os.environ['VQA_JOBROOT']).resolve()
import getpass
allowed = (Path('/scratch') / getpass.getuser()).resolve()
if not root.is_relative_to(allowed):
    raise ValueError('运行时根目录超出本人scratch')
names = ['HOME','XDG_CACHE_HOME','HF_HOME','HF_HUB_CACHE','HF_DATASETS_CACHE','HF_XET_CACHE',
         'HF_MODULES_CACHE','MODELSCOPE_CACHE','TORCH_HOME','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR',
         'VLLM_CACHE_ROOT','CUDA_CACHE_PATH','FLASHINFER_WORKSPACE_BASE','TMPDIR','TMP','TEMP','PIP_CACHE_DIR']
records = []
for name in names:
    path = Path(os.environ[name]).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'{name}超出本次作业目录')
    path.mkdir(parents=True, exist_ok=True)
    probe = path / '.write_check'
    probe.write_text('ok')
    probe.unlink()
    records.append({'name':name,'path':str(path),'writable':True})
free = shutil.disk_usage(root).free
if free < 80 * 1024**3:
    raise OSError('可用空间不足80GiB')
(root / 'storage_preflight.json').write_text(json.dumps({
    'job_id':os.environ['SLURM_JOB_ID'],'root':str(root),'free_bytes':free,'paths':records,'passed':True},indent=2))
print('STORAGE_PREFLIGHT_OK', root, free, flush=True)
PY
