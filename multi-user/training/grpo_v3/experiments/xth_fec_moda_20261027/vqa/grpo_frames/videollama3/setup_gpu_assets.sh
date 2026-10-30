#!/bin/bash
# 仅由CPU准备作业调用；安装独立CUDA环境和权重，不申请或测量GPU。
set -euo pipefail
: "${VQA_ROOT:?}"
: "${VQA_GPU_ENV:?填写专用GPU环境目录}"
: "${VQA_BASE_PYTHON:?填写可用的Python3.11可执行文件}"
if [ ! -x "$VQA_GPU_ENV/bin/python" ]; then
  "$VQA_BASE_PYTHON" -m venv "$VQA_GPU_ENV"
fi
"$VQA_GPU_ENV/bin/python" -m pip install --disable-pip-version-check torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
"$VQA_GPU_ENV/bin/python" -m pip install --disable-pip-version-check -r requirements-inference.txt
"$VQA_GPU_ENV/bin/python" -m pip install --disable-pip-version-check 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3%2Bcu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl'
"$VQA_GPU_ENV/bin/python" -m pip check
"$VQA_GPU_ENV/bin/python" - <<'PY'
import json,os,pathlib,importlib.metadata
import torch,flash_attn
from huggingface_hub import snapshot_download
root=pathlib.Path(os.environ['VQA_ROOT'])
config=json.loads((root/'code/experiment_config.json').read_text())
if torch.version.cuda!='12.4' or torch._C._GLIBCXX_USE_CXX11_ABI:
    raise RuntimeError('CUDA或C++ ABI与预编译FlashAttention轮子不一致')
model=snapshot_download(config['model'],revision=config['model_revision'],local_dir=str(root/'model'),
                        allow_patterns=['*.json','*.py','*.txt','*.safetensors'],token=False,max_workers=4)
report={'status':'CPU_IMPORT_AND_ASSETS_READY_GPU_NOT_TESTED','model_dir':model,
        'model_revision':config['model_revision'],'cuda_build':torch.version.cuda,
        'packages':{p:importlib.metadata.version(p) for p in ['torch','torchvision','transformers','flash-attn']}}
(root/f'gpu_assets_{os.environ["SLURM_JOB_ID"]}.json').write_text(json.dumps(report,indent=2)+'\n')
print('GPU_ASSETS_CPU_READY',json.dumps(report),flush=True)
PY
