# Score-only Reward Model 驱动 Generator GRPO：Torch/H100 完整运行手册

本手册服从 [Torch 实验 Meta 规则](../EgoQA-two-user/docs/TORCH_EXPERIMENT_META_RULES_CN.md) 和 [Torch Runbook 模板](../EgoQA-two-user/docs/TORCH_RUNBOOK_TEMPLATE_CN.md)。这两个 Markdown 只供人工阅读，不是远端作业依赖。

## 0. 固定合同与当前证据边界

### 0.1 实验合同

本实验冻结 `checkpoints/ckpt1` 中的 Qwen3-VL-8B score-only ordinal reviewer，只训练 Qwen3-VL-8B generator 的 LoRA：

```text
GPU 0：Qwen3-VL-8B generator，ms-swift 4.2.2 GRPO
GPU 1：Qwen3-VL-8B frozen score-only reviewer HTTP 服务
generator 输入：每条数据的两个原生视频与 messages
reviewer 输入：packet_json 中两位 required_users 的完整原始视频与一个 generator completion
reward function：egoqa_score_only_ordinal_v1
reward revision：score_only_ordinal_reviewer_v1
```

每个 ordinal head 的期望分数和归一化分数为：

\[
\mathbb{E}[s_h]=1+P(s_h>1)+P(s_h>2),
\qquad
r_h=\frac{\mathbb{E}[s_h]-1}{2}.
\]

GRPO 的标量 reward 为：

\[
R=0.4r_{\mathrm{evidence}}+0.4r_{\mathrm{answerability}}+0.2r_{\mathrm{formality}}.
\]

没有 `overall_utility` head，不使用 pairwise、tie 或 Bradley–Terry loss。无法解析或不满足五选一 JSON 契约的 completion 由代码确定性记为 `0.0`；reviewer 加载、健康检查或真实推理失败时整个作业失败，不允许回退到伪 reward。

### 0.2 当前真实 Git 与文件状态

本地检查结果：

```text
仓库根：C:/Users/20661/Desktop/Research/AR/multiuser/score-only-reward-model
分支：codex/score-only-reward-model
HEAD：fc148f4564ab486fe70f0a87c283d08772e650d0
origin：https://github.com/GODAVID0819/EgoQA-two-user.git
worktree：仅当前 score-only-reward-model worktree
状态：dirty；本实验 GRPO 文件、Runbook 和 checkpoint 均未进入当前 HEAD
```

因此本轮选择**窄 SFTP 同步**，不把 `HEAD` 冒充为实际运行代码，不执行 `git reset`、`git clean`、merge 或 push。运行身份由以下三部分共同确定：

1. 上述基线 `HEAD`；
2. 上传前生成并在 Torch 复核的 `grpo_score_only_sync.sha256`；
3. 数据集与 checkpoint 的实际 SHA-256。

### 0.3 checkpoint 固定指纹

本地 `checkpoints/ckpt1` 当前恰有 9 个文件，总计 `12,888,119` bytes：

| 相对路径 | bytes | SHA-256 |
|---|---:|---|
| `lora_adapter.pt` | 430035 | `606aa54273198782238c68d063649bdb1a8d799200b0bdfcff3a5cb4abd0439b` |
| `ordinal_heads.pt` | 53155 | `2cb86e104dcda76160491afcff6108a7587bf38d997aabac5f882bcb335df9e0` |
| `parameter_audit.json` | 2788 | `ee6ce06451204ad603182cfd7013256c9dd6d304dffe0d295d0e222989a9f519` |
| `processor/chat_template.jinja` | 5292 | `3636d0f0bd6bef02654cdffdc447b79cb2cef8ab02cc75267345946291a489e4` |
| `processor/processor_config.json` | 1191 | `d89ef49ce9cd37fbf510158e13c1ef063d9286411c1ec9049932dbe0487143b1` |
| `processor/tokenizer.json` | 11422818 | `8579e1ca7cc5d82a9e0202eed555529996f4ffe7f563c2979a0290cf3db452d3` |
| `processor/tokenizer_config.json` | 734 | `7ce86d685d3fb775073b9b1c0f576dd89b7943194c771dd2573cc10e5cae4a48` |
| `reviewer_v1_config.json` | 6570 | `d006a2d60d80485afa825f8eebbefdca9b1ea4ec4db2f0c19bc5721c0318c16d` |
| `trainer_state.pt` | 965536 | `7f524e7420a5ef1dcb1c8804f221b4955cd76428b41303aa34df3d5647904293` |

`reviewer_v1_config.json` 的真实合同是：`contract_version=score_only_ordinal_reviewer_v1`、36 个 shared blocks、只在第 34/35 block 的 `q_proj/v_proj` 注入 rank-8 LoRA、三个 active heads、60 train evidence、10 validation evidence、0 locked test evidence、seed 42。checkpoint 的训练 CSV SHA-256 为 `32679019FD7C665A0632E9885405BDF13C77B51386EFC56E7B29B24192210CD7`，split SHA-256 为 `B174A09E697F5ECA69A5E140AC22245062B39E588EE5C62278A69DDBBCDCC3D1`。

### 0.4 Gate 顺序、资源和当前边界

必须依次执行：

```text
Gate 0 零 GPU 预检
→ 1×H100 reviewer-only 真实双视频 probe
→ 2×H100 / 1-step / 4 completions Smoke
→ 2×H100 / 5-step 短 Probe
→ 2×H100 / 20-step 长 Probe
→ 2×H100 / 120-step 候选正式训练
→ 固定 held-out 评估（当前未实现，不能伪造）
```

当前 Slurm 脚本中的资源：

| 作业 | GPU | CPU | 内存 | 时限 | Slurm account |
|---|---:|---:|---:|---:|---|
| reviewer-only probe | 1×H100 | 8 | 96G | 1h | `torch_pr_674_tandon_advanced` |
| GRPO 1/5/20/120 step | 2×H100 | 8 | 160G | 2h | `torch_pr_674_tandon_advanced` |

这些是脚本中的真实请求，不是远端实测最优值。`120-step` 只有在 20-step 的耗时和显存/内存证据表明 2h 足够时才提交；否则先根据 `sacct` 和 `gpu_metrics.csv` 修改资源并重新审计。

截至本手册生成时：checkpoint 结构和 SHA 已在本地确认；新增 GRPO/Slurm 合同测试 7/7 通过。尚未在 Torch 运行任何本实验作业，没有可报告的真实 JobID、远端 dataset SHA、GPU forward/backward、训练改善或 held-out 质量结论。Windows 的 Bash 实例因系统权限错误无法启动，因此 `bash -n` 必须在 Torch Gate 0 重做。

## 1. Windows：Git 检查与代码发布

在 Windows PowerShell 中直接执行：

```powershell
$Repo = 'C:\Users\20661\Desktop\Research\AR\multiuser\score-only-reward-model'
git -C $Repo rev-parse --show-toplevel
git -C $Repo branch --show-current
git -C $Repo rev-parse HEAD
git -C $Repo remote -v
git -C $Repo worktree list
git -C $Repo status --short --branch
```

预期仍能看到 `HEAD=fc148f4564ab486fe70f0a87c283d08772e650d0` 和 dirty 文件。若 HEAD 或文件内容已变化，不要沿用上面的静态描述；下面的同步清单会为执行时的真实内容重新生成 SHA manifest。

生成窄同步清单和 SHA-256：

```powershell
$Repo = 'C:\Users\20661\Desktop\Research\AR\multiuser\score-only-reward-model'
$SyncFiles = @(
  'GRPO_SCORE_ONLY_RUNBOOK_CN.md',
  'training/grpo_v3/experiments/human_preference_reviewer/__init__.py',
  'training/grpo_v3/experiments/human_preference_reviewer/v1/__init__.py',
  'training/grpo_v3/experiments/human_preference_reviewer/v1/checkpoint.py',
  'training/grpo_v3/experiments/human_preference_reviewer/v1/config.py',
  'training/grpo_v3/experiments/human_preference_reviewer/v1/data.py',
  'training/grpo_v3/experiments/human_preference_reviewer/v1/lora.py',
  'training/grpo_v3/experiments/human_preference_reviewer/v1/modeling.py',
  'training/grpo_v3/experiments/human_preference_reviewer/v1/prompting.py',
  'training/grpo_v3/experiments/human_preference_reviewer/v1/deployment.py',
  'training/grpo_v3/experiments/human_preference_reviewer/v1/grpo_reward.py',
  'training/grpo_v3/experiments/human_preference_reviewer/v1/service.py',
  'training/grpo_v3/runtime/__init__.py',
  'training/grpo_v3/runtime/score_only_reward_plugin.py',
  'training/torch_storage_preflight.py',
  'hpc/grpo_v3/score_only_reward/reviewer_probe.sbatch',
  'hpc/grpo_v3/score_only_reward/grpo_smoke1.sbatch',
  'checkpoints/ckpt1/lora_adapter.pt',
  'checkpoints/ckpt1/ordinal_heads.pt',
  'checkpoints/ckpt1/parameter_audit.json',
  'checkpoints/ckpt1/reviewer_v1_config.json',
  'checkpoints/ckpt1/trainer_state.pt',
  'checkpoints/ckpt1/processor/chat_template.jinja',
  'checkpoints/ckpt1/processor/processor_config.json',
  'checkpoints/ckpt1/processor/tokenizer.json',
  'checkpoints/ckpt1/processor/tokenizer_config.json'
)
$Missing = $SyncFiles | Where-Object { -not (Test-Path -LiteralPath (Join-Path $Repo $_)) }
if ($Missing.Count -eq 0) {
  $Manifest = foreach ($Relative in $SyncFiles) {
    $Hash = (Get-FileHash -LiteralPath (Join-Path $Repo $Relative) -Algorithm SHA256).Hash.ToLower()
    "$Hash  $($Relative.Replace('\','/'))"
  }
  $Manifest | Set-Content -LiteralPath (Join-Path $Repo 'grpo_score_only_sync.sha256') -Encoding ascii
  Get-Content -LiteralPath (Join-Path $Repo 'grpo_score_only_sync.sha256')
} else {
  'STOP: 以下窄同步文件缺失：'
  $Missing
}
```

出现 `STOP` 时不要开始上传。该 manifest 是传输校验产物，不代表 Git commit。

## 2. Torch：同步代码并确认隔离目录

从 Windows PowerShell 登录，按提示完成 NYU MFA：

```powershell
ssh xl6775@login.torch.hpc.nyu.edu
```

进入 Torch 登录节点后执行：

```bash
NETID=xl6775
TORCH_ACCOUNT=torch_pr_674_tandon_advanced
PROJECT_PARENT=/scratch/xl6775/projects
PROJECT_ROOT=/scratch/xl6775/projects/score-only-reward-model

if [[ -d "${PROJECT_PARENT}" ]]; then
  mkdir -p "${PROJECT_ROOT}"
  echo "PROJECT_ROOT_CREATED_OR_PRESENT=${PROJECT_ROOT}"
else
  echo "MISSING: ${PROJECT_PARENT}"
  echo "STOP: projects 父目录不存在，未创建 score-only 项目根。"
fi

if [[ -d "${PROJECT_ROOT}/.git" ]]; then
  git -C "${PROJECT_ROOT}" rev-parse --show-toplevel
  git -C "${PROJECT_ROOT}" rev-parse HEAD
  git -C "${PROJECT_ROOT}" status --short --branch
  git -C "${PROJECT_ROOT}" worktree list
else
  echo "INFO: 新建的窄 SFTP 运行目录没有 .git；代码身份由本地 HEAD 和同步 SHA manifest 共同记录。"
fi
```

这里不复用已有的 `EgoQA-two-user-reviewer-v1` 或 `EgoQA-two-user-reward-main`，而是在用户已确认存在的 `/scratch/xl6775/projects` 下创建独立 `score-only-reward-model` 运行目录。该目录目前不是 Git worktree；在 dirty 本地实现尚未整理成 commit 时，不创建一个会丢失未提交实现的伪“干净” worktree。

## 3. Torch：先创建 SFTP 目标目录

仍在 Torch 登录节点执行：

```bash
PROJECT_ROOT=/scratch/xl6775/projects/score-only-reward-model

if [[ -d "${PROJECT_ROOT}" ]]; then
  mkdir -p \
    "${PROJECT_ROOT}/training/grpo_v3/experiments/human_preference_reviewer/v1" \
    "${PROJECT_ROOT}/training/grpo_v3/runtime" \
    "${PROJECT_ROOT}/hpc/grpo_v3/score_only_reward" \
    "${PROJECT_ROOT}/checkpoints/ckpt1/processor" \
    "${PROJECT_ROOT}/outputs/grpo_score_only/pre_submit" \
    "${PROJECT_ROOT}/logs"
  echo "SFTP_TARGET_DIRS_READY"
else
  echo "STOP: PROJECT_ROOT 不存在，未创建子目录。"
fi
```

## 4. Windows：窄 SFTP 同步

回到 Windows PowerShell，启动 SFTP：

```powershell
sftp xl6775@dtn.torch.hpc.nyu.edu
```

进入 `sftp>` 后完整粘贴以下块：

```text
lcd C:/Users/20661/Desktop/Research/AR/multiuser/score-only-reward-model
cd /scratch/xl6775/projects/score-only-reward-model
put grpo_score_only_sync.sha256 grpo_score_only_sync.sha256
put GRPO_SCORE_ONLY_RUNBOOK_CN.md GRPO_SCORE_ONLY_RUNBOOK_CN.md
put training/grpo_v3/experiments/human_preference_reviewer/__init__.py training/grpo_v3/experiments/human_preference_reviewer/__init__.py
put training/grpo_v3/experiments/human_preference_reviewer/v1/__init__.py training/grpo_v3/experiments/human_preference_reviewer/v1/__init__.py
put training/grpo_v3/experiments/human_preference_reviewer/v1/checkpoint.py training/grpo_v3/experiments/human_preference_reviewer/v1/checkpoint.py
put training/grpo_v3/experiments/human_preference_reviewer/v1/config.py training/grpo_v3/experiments/human_preference_reviewer/v1/config.py
put training/grpo_v3/experiments/human_preference_reviewer/v1/data.py training/grpo_v3/experiments/human_preference_reviewer/v1/data.py
put training/grpo_v3/experiments/human_preference_reviewer/v1/lora.py training/grpo_v3/experiments/human_preference_reviewer/v1/lora.py
put training/grpo_v3/experiments/human_preference_reviewer/v1/modeling.py training/grpo_v3/experiments/human_preference_reviewer/v1/modeling.py
put training/grpo_v3/experiments/human_preference_reviewer/v1/prompting.py training/grpo_v3/experiments/human_preference_reviewer/v1/prompting.py
put training/grpo_v3/experiments/human_preference_reviewer/v1/deployment.py training/grpo_v3/experiments/human_preference_reviewer/v1/deployment.py
put training/grpo_v3/experiments/human_preference_reviewer/v1/grpo_reward.py training/grpo_v3/experiments/human_preference_reviewer/v1/grpo_reward.py
put training/grpo_v3/experiments/human_preference_reviewer/v1/service.py training/grpo_v3/experiments/human_preference_reviewer/v1/service.py
put training/grpo_v3/runtime/__init__.py training/grpo_v3/runtime/__init__.py
put training/grpo_v3/runtime/score_only_reward_plugin.py training/grpo_v3/runtime/score_only_reward_plugin.py
put training/torch_storage_preflight.py training/torch_storage_preflight.py
put hpc/grpo_v3/score_only_reward/reviewer_probe.sbatch hpc/grpo_v3/score_only_reward/reviewer_probe.sbatch
put hpc/grpo_v3/score_only_reward/grpo_smoke1.sbatch hpc/grpo_v3/score_only_reward/grpo_smoke1.sbatch
put checkpoints/ckpt1/lora_adapter.pt checkpoints/ckpt1/lora_adapter.pt
put checkpoints/ckpt1/ordinal_heads.pt checkpoints/ckpt1/ordinal_heads.pt
put checkpoints/ckpt1/parameter_audit.json checkpoints/ckpt1/parameter_audit.json
put checkpoints/ckpt1/reviewer_v1_config.json checkpoints/ckpt1/reviewer_v1_config.json
put checkpoints/ckpt1/trainer_state.pt checkpoints/ckpt1/trainer_state.pt
put checkpoints/ckpt1/processor/chat_template.jinja checkpoints/ckpt1/processor/chat_template.jinja
put checkpoints/ckpt1/processor/processor_config.json checkpoints/ckpt1/processor/processor_config.json
put checkpoints/ckpt1/processor/tokenizer.json checkpoints/ckpt1/processor/tokenizer.json
put checkpoints/ckpt1/processor/tokenizer_config.json checkpoints/ckpt1/processor/tokenizer_config.json
bye
```

不要上传整个 `outputs/`、模型目录、视频目录、Conda 环境、cache、`.git` 或其他 dirty 文件。

## 5. Torch：上传接收检查

重新进入 Torch 登录节点后执行：

```bash
PROJECT_ROOT=/scratch/xl6775/projects/score-only-reward-model
cd "${PROJECT_ROOT}"

if [[ -s grpo_score_only_sync.sha256 ]]; then
  sha256sum -c grpo_score_only_sync.sha256
else
  echo "MISSING: ${PROJECT_ROOT}/grpo_score_only_sync.sha256"
  echo "STOP: 未验证上传内容。"
fi

find checkpoints/ckpt1 -type f -printf '%P\t%s bytes\n' | sort
```

`sha256sum -c` 必须 26/26 全部显示 `OK`。再固定核验关键 checkpoint 文件：

```bash
cd "${PROJECT_ROOT}"
printf '%s  %s\n' \
  606aa54273198782238c68d063649bdb1a8d799200b0bdfcff3a5cb4abd0439b checkpoints/ckpt1/lora_adapter.pt \
  2cb86e104dcda76160491afcff6108a7587bf38d997aabac5f882bcb335df9e0 checkpoints/ckpt1/ordinal_heads.pt \
  d006a2d60d80485afa825f8eebbefdca9b1ea4ec4db2f0c19bc5721c0318c16d checkpoints/ckpt1/reviewer_v1_config.json \
  8579e1ca7cc5d82a9e0202eed555529996f4ffe7f563c2979a0290cf3db452d3 checkpoints/ckpt1/processor/tokenizer.json \
  | sha256sum -c -
```

任何 `FAILED` 都要重新上传对应文件，不能提交 GPU 作业。

## 6. Torch：设置已解析变量

每次新开 Torch 登录 Shell 都完整执行：

```bash
NETID=xl6775
TORCH_ACCOUNT=torch_pr_674_tandon_advanced
PROJECT_ROOT=/scratch/xl6775/projects/score-only-reward-model
OUTPUT_ROOT=${PROJECT_ROOT}/outputs/grpo_score_only
TRAIN_ENV=/scratch/xl6775/envs/egoqa-ms-swift-v4.2.2-vllm024
FFMPEG_ENV=/scratch/xl6775/envs/egoqa-ffmpeg-runtime
POLICY_MODEL=/scratch/xl6775/models/Qwen3-VL-8B-Instruct
REVIEWER_MODEL=/scratch/xl6775/models/Qwen3-VL-8B-Instruct
REVIEWER_CHECKPOINT=${PROJECT_ROOT}/checkpoints/ckpt1
DATA_DIR=/scratch/xl6775/projects/EgoQA-two-user/outputs/grpo_v3/gate3_v2_data
DATASET=${DATA_DIR}/gate3_v2_train_native_video.jsonl
SPLIT_MANIFEST=${DATA_DIR}/gate3_v2_split_manifest.json
PYTHON=${TRAIN_ENV}/bin/python
SWIFT=${TRAIN_ENV}/bin/swift
PRE_SUBMIT_DIR=${OUTPUT_ROOT}/pre_submit

export PROJECT_ROOT OUTPUT_ROOT TRAIN_ENV FFMPEG_ENV POLICY_MODEL REVIEWER_MODEL
export REVIEWER_CHECKPOINT DATA_DIR DATASET SPLIT_MANIFEST PYTHON SWIFT PRE_SUBMIT_DIR
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PATH="${FFMPEG_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${FFMPEG_ENV}/lib:${LD_LIBRARY_PATH:-}"

cd "${PROJECT_ROOT}"
mkdir -p logs "${OUTPUT_ROOT}" "${PRE_SUBMIT_DIR}"
printf 'PROJECT_ROOT=%s\nDATASET=%s\nCHECKPOINT=%s\n' \
  "${PROJECT_ROOT}" "${DATASET}" "${REVIEWER_CHECKPOINT}"
```

`DATASET` 和 `SPLIT_MANIFEST` 的路径来自仓库现有 Gate 3 v2 数据构造器和基线 Slurm 合同；本地 score-only 隔离副本中没有这两个数据文件。必须由下面的远端 Gate 0 证明它们真实存在并计算 SHA，不能因路径“看起来正确”就提交。

## 7. Gate 0：零 GPU 代码、存储和数据预检

### 7.1 路径与脚本语法

```bash
MISSING=0
for REQUIRED_PATH in \
  "${PROJECT_ROOT}/training/grpo_v3/experiments/human_preference_reviewer/v1/deployment.py" \
  "${PROJECT_ROOT}/training/grpo_v3/experiments/human_preference_reviewer/v1/grpo_reward.py" \
  "${PROJECT_ROOT}/training/grpo_v3/experiments/human_preference_reviewer/v1/service.py" \
  "${PROJECT_ROOT}/training/grpo_v3/runtime/score_only_reward_plugin.py" \
  "${PROJECT_ROOT}/training/torch_storage_preflight.py" \
  "${PROJECT_ROOT}/hpc/grpo_v3/score_only_reward/reviewer_probe.sbatch" \
  "${PROJECT_ROOT}/hpc/grpo_v3/score_only_reward/grpo_smoke1.sbatch" \
  "${REVIEWER_CHECKPOINT}/reviewer_v1_config.json" \
  "${REVIEWER_CHECKPOINT}/lora_adapter.pt" \
  "${REVIEWER_CHECKPOINT}/ordinal_heads.pt" \
  "${REVIEWER_CHECKPOINT}/processor/tokenizer.json" \
  "${POLICY_MODEL}/config.json" \
  "${REVIEWER_MODEL}/config.json" \
  "${DATASET}" \
  "${SPLIT_MANIFEST}" \
  "${PYTHON}" \
  "${SWIFT}" \
  "${FFMPEG_ENV}/bin/ffmpeg"
do
  if [[ -e "${REQUIRED_PATH}" ]]; then
    echo "OK: ${REQUIRED_PATH}"
  else
    echo "MISSING: ${REQUIRED_PATH}"
    MISSING=1
  fi
done

if [[ "${MISSING}" -eq 0 ]]; then
  bash -n hpc/grpo_v3/score_only_reward/reviewer_probe.sbatch
  bash -n hpc/grpo_v3/score_only_reward/grpo_smoke1.sbatch
  "${PYTHON}" -m py_compile \
    training/grpo_v3/experiments/human_preference_reviewer/v1/deployment.py \
    training/grpo_v3/experiments/human_preference_reviewer/v1/grpo_reward.py \
    training/grpo_v3/experiments/human_preference_reviewer/v1/service.py \
    training/grpo_v3/runtime/score_only_reward_plugin.py \
    training/torch_storage_preflight.py
  echo "PATH_AND_SYNTAX_GATE: PASSED"
else
  echo "PATH_AND_SYNTAX_GATE: FAILED; STOP; DO NOT SUBMIT"
fi
```

### 7.2 训练环境、运行时编译工具和包身份

```bash
if [[ "${MISSING:-1}" -eq 0 ]]; then
  "${PYTHON}" -c 'import importlib.metadata as m; print("ms-swift",m.version("ms-swift")); assert m.version("ms-swift")=="4.2.2"'
  "${PYTHON}" -c 'import torch,transformers,peft,accelerate,safetensors; print("torch",torch.__version__); print("transformers",transformers.__version__); print("peft",peft.__version__); print("accelerate",accelerate.__version__)'
  "${PYTHON}" -c 'from torchcodec.decoders import VideoDecoder; print("VideoDecoder",VideoDecoder.__module__)'
  "${PYTHON}" -c 'import training.grpo_v3.experiments.human_preference_reviewer.v1.deployment as m; print("deployment",m.__file__)'
  "${PYTHON}" -c 'import training.grpo_v3.runtime.score_only_reward_plugin as m; print("plugin",m.__file__)'
  "${SWIFT}" rlhf --help | grep -E -- '--rlhf_type|--external_plugins|--reward_funcs|--num_generations|--tuner_type'
  "${FFMPEG_ENV}/bin/ffmpeg" -version | head -n 2
  command -v cc
  command -v c++
  command -v ninja
else
  echo "STOP: 路径 Gate 未通过，跳过依赖导入。"
fi
```

`deployment` 和 `plugin` 打印出的文件必须位于 `${PROJECT_ROOT}`；若解析到其他 checkout，属于包身份错误。`cc`、`c++` 或 `ninja` 缺失时不要占用 H100 试错。

### 7.3 登录节点 scratch-first 存储预检

```bash
LOGIN_PREFLIGHT_SCRATCH=/scratch/xl6775/score_only_grpo_pre_submit
export HOME=${LOGIN_PREFLIGHT_SCRATCH}/home
export XDG_CACHE_HOME=${LOGIN_PREFLIGHT_SCRATCH}/xdg
export HF_HOME=${LOGIN_PREFLIGHT_SCRATCH}/huggingface
export HF_DATASETS_CACHE=${LOGIN_PREFLIGHT_SCRATCH}/huggingface/datasets
export MODELSCOPE_CACHE=${LOGIN_PREFLIGHT_SCRATCH}/modelscope
export TORCH_HOME=${LOGIN_PREFLIGHT_SCRATCH}/torch
export TRITON_CACHE_DIR=${LOGIN_PREFLIGHT_SCRATCH}/triton
export TORCHINDUCTOR_CACHE_DIR=${LOGIN_PREFLIGHT_SCRATCH}/torchinductor
export VLLM_CACHE_ROOT=${LOGIN_PREFLIGHT_SCRATCH}/vllm
export CUDA_CACHE_PATH=${LOGIN_PREFLIGHT_SCRATCH}/cuda
export FLASHINFER_WORKSPACE_BASE=${LOGIN_PREFLIGHT_SCRATCH}/flashinfer
export FLASHINFER_WORKSPACE_DIR=${FLASHINFER_WORKSPACE_BASE}
export VLLM_NO_USAGE_STATS=1
export TMPDIR=${LOGIN_PREFLIGHT_SCRATCH}/tmp
export TMP=${TMPDIR}
export TEMP=${TMPDIR}

mkdir -p "${PRE_SUBMIT_DIR}" "${LOGIN_PREFLIGHT_SCRATCH}"
"${PYTHON}" -m training.torch_storage_preflight \
  --allowed-root "${LOGIN_PREFLIGHT_SCRATCH}" \
  --output "${PRE_SUBMIT_DIR}/storage_preflight.json"
"${PYTHON}" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"]=="passed" and not r["failed_checks"]; print("STORAGE_GATE: PASSED")' \
  "${PRE_SUBMIT_DIR}/storage_preflight.json"
```

### 7.4 checkpoint 与数据全量审计

该审计读取**全部**数据行，不只抽第一行；同时验证 generator 的两个视频、reviewer 的两个完整视频、占位符、角色顺序、packet 身份和 20/10+10 split 合同：

```bash
if [[ "${MISSING:-1}" -eq 0 ]]; then
  sha256sum "${DATASET}" "${SPLIT_MANIFEST}" \
    | tee "${PRE_SUBMIT_DIR}/dataset_inputs.sha256"

  "${PYTHON}" - "${REVIEWER_CHECKPOINT}" "${DATASET}" "${SPLIT_MANIFEST}" "${PRE_SUBMIT_DIR}/data_checkpoint_audit.json" <<'PY'
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import torch

from training.grpo_v3.experiments.human_preference_reviewer.v1.checkpoint import load_checkpoint_contract
from training.grpo_v3.experiments.human_preference_reviewer.v1.grpo_reward import prepare_completion_for_review

checkpoint, dataset_path, split_path, output_path = map(Path, sys.argv[1:])
contract = load_checkpoint_contract(checkpoint)
assert contract["contract_version"] == "score_only_ordinal_reviewer_v1"
assert contract["active_heads"] == ["evidence_quality", "answerability", "qa_formality"]
assert contract["loss_weights"] == {"evidence_quality": 0.4, "answerability": 0.4, "qa_formality": 0.2}
assert contract["last_n_shared_blocks"] == 2
assert contract["expected_shared_block_count"] == 36
assert contract["lora_target_modules"] == ["q_proj", "v_proj"]
assert contract["lora_r"] == 8 and contract["lora_alpha"] == 16
lora = torch.load(checkpoint / "lora_adapter.pt", map_location="cpu", weights_only=True)
heads = torch.load(checkpoint / "ordinal_heads.pt", map_location="cpu", weights_only=True)
assert set(lora) == set(contract["lora_parameter_names"])
assert set(heads) == {"evidence_head", "answerability_head", "formality_head"}

rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
split = json.loads(split_path.read_text(encoding="utf-8"))
assert len(rows) == 20
assert len({row["evidence_id"] for row in rows}) == 20
assert Counter(row["question_type"] for row in rows) == {"commonality": 10, "difference": 10}
assert split["schema_version"] == "grpo_v3_gate3_v2_split_v1"
assert split["seed"] == 42 and split["train_count"] == 20 and split["eval_count"] == 8
assert split["train_evidence_ids"] == [row["evidence_id"] for row in rows]
required_fields = {"messages", "videos", "evidence_id", "packet_json", "question_type", "generation_mode", "required_users", "video_order"}
reviewer_video_paths = []
generator_video_paths = []
for row in rows:
    assert required_fields <= set(row)
    assert isinstance(row["messages"], list) and len(row["messages"]) == 1
    assert row["messages"][0]["role"] == "user"
    assert str(row["messages"][0]["content"]).count("<video>") == 2
    assert isinstance(row["videos"], list) and len(row["videos"]) == 2
    assert row["video_order"] == row["required_users"] and len(set(row["required_users"])) == 2
    for value in row["videos"]:
        path = Path(str(value))
        assert path.suffix.lower() == ".mp4" and path.is_file() and path.stat().st_size > 0
        generator_video_paths.append(str(path.resolve()))
    packet = json.loads(row["packet_json"])
    assert packet["evidence_id"] == row["evidence_id"]
    completion = json.dumps({
        "question": "Which relation best describes the two synchronized views?",
        "options": ["same activity", "first only", "second only", "unrelated", "uncertain"],
        "correct": "E",
        "answer": "uncertain",
    })
    for candidate_index in range(4):
        prepared = prepare_completion_for_review(
            completion, packet, evidence_id=row["evidence_id"], candidate_index=candidate_index
        )
        assert prepared["status"] == "eligible"
        reviewer_video_paths.extend([
            prepared["candidate"]["video_a_path"],
            prepared["candidate"]["video_b_path"],
        ])

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

result = {
    "status": "passed",
    "dataset": str(dataset_path.resolve()),
    "dataset_sha256": sha256(dataset_path),
    "split_manifest": str(split_path.resolve()),
    "split_manifest_sha256": sha256(split_path),
    "row_count": len(rows),
    "evidence_count": len({row["evidence_id"] for row in rows}),
    "question_type_counts": dict(Counter(row["question_type"] for row in rows)),
    "generator_video_reference_count": len(generator_video_paths),
    "reviewer_video_reference_count_for_four_candidate_shape": len(reviewer_video_paths),
    "checkpoint_contract_version": contract["contract_version"],
    "checkpoint_training_csv_sha256": contract["csv_sha256"],
    "checkpoint_training_split_sha256": contract["split_sha256"],
}
output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
PY

  "${PYTHON}" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"]=="passed" and r["row_count"]==20 and r["evidence_count"]==20 and r["reviewer_video_reference_count_for_four_candidate_shape"]==160; print("DATA_CHECKPOINT_GATE: PASSED")' \
    "${PRE_SUBMIT_DIR}/data_checkpoint_audit.json"
else
  echo "STOP: 路径 Gate 未通过，跳过数据审计。"
fi
```

如果数据文件缺失，本手册不会猜一个 SHA 或偷偷改用其他 JSONL。先把由 `training/grpo_v3/baseline/gate3_dataset.py` 真实生成的 `gate3_v2_train_native_video.jsonl` 与同一次生成的 `gate3_v2_split_manifest.json` 放到 `${DATA_DIR}`，再从 Gate 0 重新开始。若逐条审计出现 `accepted_count=0` 或全部 completion 被确定性拒绝，应检查 JSON 格式、`packet_json`、完整视频路径和拒绝原因，不得直接重投 H100。

## 8. GPU runtime probe：frozen reviewer 真实双视频请求

该作业只加载 reviewer，不启动 generator 或 trainer。它从 `${DATASET}` 第一条真实 `packet_json` 取两位用户的完整视频，通过 `/score` 完成一次实际 multimodal forward。

提交前确认没有同名活动作业：

```bash
squeue -u "${USER}" -n egoqa-score-rm-probe \
  -o '%.18i %.24j %.10T %.10M %.10l %R' 2>/dev/null || true
```

只在 Gate 0 产物通过时提交：

```bash
cd "${PROJECT_ROOT}"
if [[ -s "${PRE_SUBMIT_DIR}/data_checkpoint_audit.json" ]] \
  && "${PYTHON}" -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"]=="passed"' "${PRE_SUBMIT_DIR}/data_checkpoint_audit.json" \
  && "${PYTHON}" -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"]=="passed"' "${PRE_SUBMIT_DIR}/storage_preflight.json"
then
  REVIEWER_PROBE_JOB_RAW=$(sbatch --parsable \
    --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",OUTPUT_ROOT="${OUTPUT_ROOT}",TRAIN_ENV="${TRAIN_ENV}",FFMPEG_ENV="${FFMPEG_ENV}",REVIEWER_MODEL="${REVIEWER_MODEL}",REVIEWER_CHECKPOINT="${REVIEWER_CHECKPOINT}",DATASET="${DATASET}" \
    hpc/grpo_v3/score_only_reward/reviewer_probe.sbatch)
  REVIEWER_PROBE_JOB=${REVIEWER_PROBE_JOB_RAW%%;*}
  REVIEWER_PROBE_DIR=${OUTPUT_ROOT}/reviewer_probe_${REVIEWER_PROBE_JOB}
  printf 'REVIEWER_PROBE_JOB=%s\nREVIEWER_PROBE_DIR=%s\n' \
    "${REVIEWER_PROBE_JOB}" "${REVIEWER_PROBE_DIR}" \
    > "${OUTPUT_ROOT}/reviewer_probe_submission_${REVIEWER_PROBE_JOB}.env"
  echo "REVIEWER_PROBE_JOB=${REVIEWER_PROBE_JOB}"
  echo "REVIEWER_PROBE_DIR=${REVIEWER_PROBE_DIR}"
else
  echo "STOP: Gate 0 未通过，reviewer probe 未提交。"
fi
```

监控和验收：

```bash
squeue -j "${REVIEWER_PROBE_JOB}" -o '%.18i %.24j %.10T %.10M %.10l %R' 2>/dev/null || true
sacct -j "${REVIEWER_PROBE_JOB}" --units=G \
  --format=JobID,JobName%28,State,ExitCode,Elapsed,Timelimit,AllocTRES%40,MaxRSS
tail -n 100 "${PROJECT_ROOT}/logs/score-only-reviewer-probe-${REVIEWER_PROBE_JOB}.out"
tail -n 100 "${PROJECT_ROOT}/logs/score-only-reviewer-probe-${REVIEWER_PROBE_JOB}.err"
tail -n 100 "${REVIEWER_PROBE_DIR}/reviewer_service.log"

if [[ -s "${REVIEWER_PROBE_DIR}/reviewer_probe_result.json" ]]; then
  "${PYTHON}" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"]=="passed"; assert r["health"]["status"]=="ok"; assert r["health"]["trainable_parameter_count"]==0; assert len(r["video_paths"])==2; assert 0<=float(r["score"]["reward"])<=1; assert set(r["score"]["expected_scores"])=={"evidence_quality","answerability","qa_formality"}; print(json.dumps(r,ensure_ascii=False,indent=2))' \
    "${REVIEWER_PROBE_DIR}/reviewer_probe_result.json"
else
  echo "MISSING: ${REVIEWER_PROBE_DIR}/reviewer_probe_result.json"
  echo "STOP: reviewer probe 未通过。"
fi
```

只有 Slurm 顶层与 `.batch` 均为 `COMPLETED / 0:0`，且结果 JSON 断言通过，才能提交双卡 Smoke。该 Gate 证明 checkpoint 可以冻结加载并处理一条真实双视频请求；不证明 generator、GRPO 梯度或 reward 方差。

## 9. Gate 依赖与产物映射

| Gate | 前置硬条件 | JobID 推导输出 | 主要验收文件 |
|---|---|---|---|
| Gate 0 | 窄同步 SHA、路径、依赖、存储、20-row 数据审计通过 | 无 GPU JobID | `pre_submit/storage_preflight.json`、`data_checkpoint_audit.json`、`dataset_inputs.sha256` |
| reviewer probe | Gate 0 通过 | `${OUTPUT_ROOT}/reviewer_probe_${REVIEWER_PROBE_JOB}` | `reviewer_probe_result.json`、`reviewer_health.json`、`reviewer_service.log` |
| Gate 1 Smoke | reviewer probe 通过 | `${OUTPUT_ROOT}/smoke1_${SMOKE_JOB}` | `smoke_result.json`、`reward_trace.jsonl`、`run_manifest.json`、adapter |
| Gate 2 Probe 5 | Smoke 通过 | `${OUTPUT_ROOT}/smoke1_${PROBE5_JOB}` | 同上 |
| Gate 3 Probe 20 | Probe 5 通过 | `${OUTPUT_ROOT}/smoke1_${PROBE20_JOB}` | 同上 |
| Gate 4 Train 120 | Probe 20 通过且资源外推可容纳 | `${OUTPUT_ROOT}/smoke1_${TRAIN120_JOB}` | 同上 |
| Gate 5 held-out | 独立评估代码与固定 held-out 数据 | 当前不存在 | 当前不可执行 |

不要使用 `latest_*`、mtime 或“最像的目录”归因。本实验所有运行产物都从 `sbatch --parsable` 返回的真实 JobID 推导。

## 10. 通用 JobID 检查函数

在 Torch 登录 Shell 中定义一次：

```bash
check_job() {
  CHECK_JOB_ID="$1"
  CHECK_OUTPUT_DIR="$2"
  echo "JOB_ID=${CHECK_JOB_ID}"
  echo "OUTPUT_DIR=${CHECK_OUTPUT_DIR}"
  squeue -j "${CHECK_JOB_ID}" -o '%.18i %.24j %.10T %.10M %.10l %R' 2>/dev/null || true
  sacct -j "${CHECK_JOB_ID}" --units=G \
    --format=JobID,JobName%28,State,ExitCode,Elapsed,Timelimit,AllocTRES%40,MaxRSS
  if [[ -s "${CHECK_OUTPUT_DIR}/smoke_result.json" ]]; then
    "${PYTHON}" -c 'import json,sys; r=json.load(open(sys.argv[1])); print(json.dumps({"status":r.get("status"),"failed_checks":r.get("failed_checks"),"reward_count":r.get("reward_count"),"valid_reviewer_reward_count":r.get("valid_reviewer_reward_count"),"reward_mean":r.get("reward_mean"),"reward_std":r.get("reward_std"),"global_step":r.get("global_step"),"checks":r.get("checks")},ensure_ascii=False,indent=2))' \
      "${CHECK_OUTPUT_DIR}/smoke_result.json"
  else
    echo "MISSING: ${CHECK_OUTPUT_DIR}/smoke_result.json"
  fi
}
```

活动状态看 `squeue`；作业退出后的最终状态看 `sacct`。`squeue` 无输出只表示作业不在活动队列，不能解释为成功。

## 11. Gate 1：1-step / 4-completion Smoke

提交：

```bash
if [[ -s "${REVIEWER_PROBE_DIR}/reviewer_probe_result.json" ]] \
  && "${PYTHON}" -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"]=="passed"' "${REVIEWER_PROBE_DIR}/reviewer_probe_result.json"
then
  SMOKE_JOB_RAW=$(sbatch --parsable \
    --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",OUTPUT_ROOT="${OUTPUT_ROOT}",TRAIN_ENV="${TRAIN_ENV}",FFMPEG_ENV="${FFMPEG_ENV}",POLICY_MODEL="${POLICY_MODEL}",REVIEWER_MODEL="${REVIEWER_MODEL}",REVIEWER_CHECKPOINT="${REVIEWER_CHECKPOINT}",DATASET="${DATASET}",MAX_STEPS=1,EXPECTED_GROUPS=1 \
    hpc/grpo_v3/score_only_reward/grpo_smoke1.sbatch)
  SMOKE_JOB=${SMOKE_JOB_RAW%%;*}
  SMOKE_DIR=${OUTPUT_ROOT}/smoke1_${SMOKE_JOB}
  printf 'SMOKE_JOB=%s\nSMOKE_DIR=%s\nREVIEWER_PROBE_JOB=%s\n' \
    "${SMOKE_JOB}" "${SMOKE_DIR}" "${REVIEWER_PROBE_JOB}" \
    > "${OUTPUT_ROOT}/smoke_submission_${SMOKE_JOB}.env"
  echo "SMOKE_JOB=${SMOKE_JOB}"
  echo "SMOKE_DIR=${SMOKE_DIR}"
else
  echo "STOP: reviewer-only Gate 未通过，Smoke 未提交。"
fi
```

监控：

```bash
check_job "${SMOKE_JOB}" "${SMOKE_DIR}"
tail -n 100 "${PROJECT_ROOT}/logs/score-only-grpo-smoke1-${SMOKE_JOB}.out"
tail -n 100 "${PROJECT_ROOT}/logs/score-only-grpo-smoke1-${SMOKE_JOB}.err"
tail -n 100 "${SMOKE_DIR}/reviewer_service.log"
```

硬验收：

```bash
if [[ -s "${SMOKE_DIR}/smoke_result.json" ]] && [[ -s "${SMOKE_DIR}/run_manifest.json" ]]; then
  "${PYTHON}" -c 'import json,sys; r=json.load(open(sys.argv[1])); m=json.load(open(sys.argv[2])); assert r["status"]=="passed" and not r["failed_checks"]; assert r["reward_count"]==4; assert r["valid_reviewer_reward_count"]>=2; assert r["reward_std"]>0; assert r["global_step"]>=1; assert all(r["checks"].values()); assert m["status"]=="passed" and m["reward_revision"]=="score_only_ordinal_reviewer_v1" and m["max_steps"]==1; print("SMOKE_GATE: PASSED")' \
    "${SMOKE_DIR}/smoke_result.json" "${SMOKE_DIR}/run_manifest.json"
else
  echo "STOP: Smoke 验收文件缺失。"
fi
```

同时要求 `sacct` 顶层和 `.batch` 为 `COMPLETED / 0:0`，adapter 存在、至少一个 `lora_B` 非零、存在有限非零 `grad_norm`。即使全部通过，也只证明 1 个 GRPO step 的工程闭环，不证明收敛或真实 QA 改善。

## 12. Gate 2：5-step 短 Probe

```bash
if [[ -s "${SMOKE_DIR}/smoke_result.json" ]] \
  && "${PYTHON}" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"]=="passed" and r["checks"]["reward_std_positive"] and r["checks"]["valid_reviewer_rewards_at_least_two"]' "${SMOKE_DIR}/smoke_result.json"
then
  PROBE5_JOB_RAW=$(sbatch --parsable \
    --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",OUTPUT_ROOT="${OUTPUT_ROOT}",TRAIN_ENV="${TRAIN_ENV}",FFMPEG_ENV="${FFMPEG_ENV}",POLICY_MODEL="${POLICY_MODEL}",REVIEWER_MODEL="${REVIEWER_MODEL}",REVIEWER_CHECKPOINT="${REVIEWER_CHECKPOINT}",DATASET="${DATASET}",MAX_STEPS=5,EXPECTED_GROUPS=5 \
    hpc/grpo_v3/score_only_reward/grpo_smoke1.sbatch)
  PROBE5_JOB=${PROBE5_JOB_RAW%%;*}
  PROBE5_DIR=${OUTPUT_ROOT}/smoke1_${PROBE5_JOB}
  printf 'PROBE5_JOB=%s\nPROBE5_DIR=%s\nSMOKE_JOB=%s\n' \
    "${PROBE5_JOB}" "${PROBE5_DIR}" "${SMOKE_JOB}" \
    > "${OUTPUT_ROOT}/probe5_submission_${PROBE5_JOB}.env"
  echo "PROBE5_JOB=${PROBE5_JOB}"
  echo "PROBE5_DIR=${PROBE5_DIR}"
else
  echo "STOP: Smoke 未通过，5-step Probe 未提交。"
fi
```

```bash
check_job "${PROBE5_JOB}" "${PROBE5_DIR}"
tail -n 100 "${PROJECT_ROOT}/logs/score-only-grpo-smoke1-${PROBE5_JOB}.out"
tail -n 100 "${PROJECT_ROOT}/logs/score-only-grpo-smoke1-${PROBE5_JOB}.err"

if [[ -s "${PROBE5_DIR}/smoke_result.json" ]] && [[ -s "${PROBE5_DIR}/run_manifest.json" ]]; then
  "${PYTHON}" -c 'import json,sys; r=json.load(open(sys.argv[1])); m=json.load(open(sys.argv[2])); assert r["status"]=="passed" and not r["failed_checks"]; assert r["reward_count"]==20 and r["valid_reviewer_reward_count"]>=2 and r["reward_std"]>0 and r["global_step"]>=5; assert m["status"]=="passed" and m["max_steps"]==5; print("PROBE5_GATE: PASSED")' \
    "${PROBE5_DIR}/smoke_result.json" "${PROBE5_DIR}/run_manifest.json"
else
  echo "STOP: 5-step Probe 验收文件缺失。"
fi
```

除脚本硬 Gate 外，人工检查 20 条 trace 中 `deterministic_rejection` 的比例和 `rejection_reason`。若大多数候选因同一个 schema 原因被拒绝，即使有少量方差也不能升级。

## 13. Gate 3：20-step 长 Probe

```bash
if [[ -s "${PROBE5_DIR}/smoke_result.json" ]] \
  && "${PYTHON}" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"]=="passed" and r["global_step"]>=5 and r["reward_std"]>0' "${PROBE5_DIR}/smoke_result.json"
then
  PROBE20_JOB_RAW=$(sbatch --parsable \
    --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",OUTPUT_ROOT="${OUTPUT_ROOT}",TRAIN_ENV="${TRAIN_ENV}",FFMPEG_ENV="${FFMPEG_ENV}",POLICY_MODEL="${POLICY_MODEL}",REVIEWER_MODEL="${REVIEWER_MODEL}",REVIEWER_CHECKPOINT="${REVIEWER_CHECKPOINT}",DATASET="${DATASET}",MAX_STEPS=20,EXPECTED_GROUPS=20 \
    hpc/grpo_v3/score_only_reward/grpo_smoke1.sbatch)
  PROBE20_JOB=${PROBE20_JOB_RAW%%;*}
  PROBE20_DIR=${OUTPUT_ROOT}/smoke1_${PROBE20_JOB}
  printf 'PROBE20_JOB=%s\nPROBE20_DIR=%s\nPROBE5_JOB=%s\n' \
    "${PROBE20_JOB}" "${PROBE20_DIR}" "${PROBE5_JOB}" \
    > "${OUTPUT_ROOT}/probe20_submission_${PROBE20_JOB}.env"
  echo "PROBE20_JOB=${PROBE20_JOB}"
  echo "PROBE20_DIR=${PROBE20_DIR}"
else
  echo "STOP: 5-step Probe 未通过，20-step Probe 未提交。"
fi
```

```bash
check_job "${PROBE20_JOB}" "${PROBE20_DIR}"
tail -n 100 "${PROJECT_ROOT}/logs/score-only-grpo-smoke1-${PROBE20_JOB}.out"
tail -n 100 "${PROJECT_ROOT}/logs/score-only-grpo-smoke1-${PROBE20_JOB}.err"

if [[ -s "${PROBE20_DIR}/smoke_result.json" ]] && [[ -s "${PROBE20_DIR}/run_manifest.json" ]]; then
  "${PYTHON}" -c 'import json,sys; r=json.load(open(sys.argv[1])); m=json.load(open(sys.argv[2])); assert r["status"]=="passed" and not r["failed_checks"]; assert r["reward_count"]==80 and r["valid_reviewer_reward_count"]>=2 and r["reward_std"]>0 and r["global_step"]>=20; assert m["status"]=="passed" and m["max_steps"]==20; print("PROBE20_ENGINEERING_GATE: PASSED")' \
    "${PROBE20_DIR}/smoke_result.json" "${PROBE20_DIR}/run_manifest.json"
else
  echo "STOP: 20-step Probe 验收文件缺失。"
fi
```

当前脚本证明 reward 有效、梯度非零和 LoRA 改变，但尚未实现“前 10 组对后 10 组 reward 均值/斜率”的独立收敛分析。因此 20-step 通过只能称为**长 Probe 工程 Gate 通过**，不能称为收敛。提交 120-step 前必须人工统计 trace 与 trainer log，确认没有 reward collapse、拒绝率突增或 NaN。

## 14. Gate 4：120-step 候选正式训练

先读取真实资源消耗：

```bash
sacct -j "${PROBE20_JOB}" --units=G \
  --format=JobID,JobName%28,State,ExitCode,Elapsed,Timelimit,AllocTRES%40,MaxRSS
head -n 3 "${PROBE20_DIR}/gpu_metrics.csv"
tail -n 20 "${PROBE20_DIR}/gpu_metrics.csv"
```

只有 `PROBE20` 为 `COMPLETED / 0:0`、工程 Gate 通过，且按实测启动成本与每 step 耗时外推后 120 steps 能在脚本真实 `02:00:00` 时限内完成，才执行：

```bash
if [[ -s "${PROBE20_DIR}/smoke_result.json" ]] \
  && "${PYTHON}" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"]=="passed" and r["global_step"]>=20 and r["reward_std"]>0' "${PROBE20_DIR}/smoke_result.json"
then
  TRAIN120_JOB_RAW=$(sbatch --parsable \
    --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",OUTPUT_ROOT="${OUTPUT_ROOT}",TRAIN_ENV="${TRAIN_ENV}",FFMPEG_ENV="${FFMPEG_ENV}",POLICY_MODEL="${POLICY_MODEL}",REVIEWER_MODEL="${REVIEWER_MODEL}",REVIEWER_CHECKPOINT="${REVIEWER_CHECKPOINT}",DATASET="${DATASET}",MAX_STEPS=120,EXPECTED_GROUPS=120 \
    hpc/grpo_v3/score_only_reward/grpo_smoke1.sbatch)
  TRAIN120_JOB=${TRAIN120_JOB_RAW%%;*}
  TRAIN120_DIR=${OUTPUT_ROOT}/smoke1_${TRAIN120_JOB}
  printf 'TRAIN120_JOB=%s\nTRAIN120_DIR=%s\nPROBE20_JOB=%s\n' \
    "${TRAIN120_JOB}" "${TRAIN120_DIR}" "${PROBE20_JOB}" \
    > "${OUTPUT_ROOT}/train120_submission_${TRAIN120_JOB}.env"
  echo "TRAIN120_JOB=${TRAIN120_JOB}"
  echo "TRAIN120_DIR=${TRAIN120_DIR}"
else
  echo "STOP: 20-step Gate 未通过，120-step 训练未提交。"
fi
```

验收：

```bash
check_job "${TRAIN120_JOB}" "${TRAIN120_DIR}"
tail -n 100 "${PROJECT_ROOT}/logs/score-only-grpo-smoke1-${TRAIN120_JOB}.out"
tail -n 100 "${PROJECT_ROOT}/logs/score-only-grpo-smoke1-${TRAIN120_JOB}.err"

if [[ -s "${TRAIN120_DIR}/smoke_result.json" ]] && [[ -s "${TRAIN120_DIR}/run_manifest.json" ]]; then
  "${PYTHON}" -c 'import json,sys; r=json.load(open(sys.argv[1])); m=json.load(open(sys.argv[2])); assert r["status"]=="passed" and not r["failed_checks"]; assert r["reward_count"]==480 and r["valid_reviewer_reward_count"]>=2 and r["reward_std"]>0 and r["global_step"]>=120; assert m["status"]=="passed" and m["max_steps"]==120; print("TRAIN120_ENGINEERING_GATE: PASSED")' \
    "${TRAIN120_DIR}/smoke_result.json" "${TRAIN120_DIR}/run_manifest.json"
else
  echo "STOP: 120-step 训练验收文件缺失。"
fi
```

`TRAIN120_ENGINEERING_GATE: PASSED` 仍只证明训练工程产物完整。它不能单独证明平均 reward 改善具有统计意义，更不能证明真人评价下的 QA groundedness、answerability 或 formality 改善。

## 15. Gate 5：固定验证或评估

当前 score-only GRPO 实现没有 generator checkpoint 的固定 held-out 生成与 before/after 评估脚本；现有 `gate3_v2_eval_native_video.jsonl` 的构造代码在兄弟仓库，但本隔离副本中没有可验证的评估入口。因此本 Gate 明确为：

```text
状态：BLOCKED_BY_MISSING_IMPLEMENTATION
不能执行：不得把训练集 reward、Slurm COMPLETED 或 adapter 非零当成 held-out 质量评估
解锁条件：新增并测试固定 seed/解码参数的 base-vs-adapter held-out 生成、同一 frozen reviewer 评分、逐 evidence 配对统计和人工抽查产物
```

在该入口真实存在之前，本手册不伪造 `sbatch` 命令、输出路径、字段或验收阈值。

## 16. 通用失败证据收集

在 Torch 登录 Shell 定义：

```bash
collect_failure() {
  FAILED_STAGE="$1"
  FAILED_JOB="$2"
  FAILED_DIR="$3"
  DIAG_DIR=${OUTPUT_ROOT}/diagnostics/${FAILED_STAGE}_${FAILED_JOB}
  mkdir -p "${DIAG_DIR}"
  sacct -j "${FAILED_JOB}" --units=G \
    --format=JobID,JobName%28,State,ExitCode,Elapsed,Timelimit,AllocTRES%40,MaxRSS \
    > "${DIAG_DIR}/sacct.txt" 2>&1 || true
  scontrol show job -dd "${FAILED_JOB}" > "${DIAG_DIR}/scontrol.txt" 2>&1 || true
  cp -f "${PROJECT_ROOT}/logs/"*"${FAILED_JOB}"*.out "${DIAG_DIR}/" 2>/dev/null || true
  cp -f "${PROJECT_ROOT}/logs/"*"${FAILED_JOB}"*.err "${DIAG_DIR}/" 2>/dev/null || true
  cp -f "${FAILED_DIR}/storage_preflight.json" "${DIAG_DIR}/" 2>/dev/null || true
  cp -f "${FAILED_DIR}/dependencies.txt" "${DIAG_DIR}/" 2>/dev/null || true
  cp -f "${FAILED_DIR}/gpu_environment.csv" "${DIAG_DIR}/" 2>/dev/null || true
  cp -f "${FAILED_DIR}/ffmpeg_environment.txt" "${DIAG_DIR}/" 2>/dev/null || true
  cp -f "${FAILED_DIR}/gpu_metrics.csv" "${DIAG_DIR}/" 2>/dev/null || true
  cp -f "${FAILED_DIR}/reviewer_health.json" "${DIAG_DIR}/" 2>/dev/null || true
  cp -f "${FAILED_DIR}/reviewer_service.log" "${DIAG_DIR}/" 2>/dev/null || true
  cp -f "${FAILED_DIR}/reviewer_probe_result.json" "${DIAG_DIR}/" 2>/dev/null || true
  cp -f "${FAILED_DIR}/reward_trace.jsonl" "${DIAG_DIR}/" 2>/dev/null || true
  cp -f "${FAILED_DIR}/smoke_result.json" "${DIAG_DIR}/" 2>/dev/null || true
  cp -f "${FAILED_DIR}/run_manifest.json" "${DIAG_DIR}/" 2>/dev/null || true
  cp -f "${PRE_SUBMIT_DIR}/data_checkpoint_audit.json" "${DIAG_DIR}/" 2>/dev/null || true
  cp -f "${PRE_SUBMIT_DIR}/dataset_inputs.sha256" "${DIAG_DIR}/" 2>/dev/null || true
  tar -czf "${DIAG_DIR}.tar.gz" -C "$(dirname "${DIAG_DIR}")" "$(basename "${DIAG_DIR}")"
  echo "DIAGNOSTIC_BUNDLE=${DIAG_DIR}.tar.gz"
}
```

示例使用真实变量，不填虚构 JobID：

```bash
collect_failure smoke "${SMOKE_JOB}" "${SMOKE_DIR}"
collect_failure probe5 "${PROBE5_JOB}" "${PROBE5_DIR}"
collect_failure probe20 "${PROBE20_JOB}" "${PROBE20_DIR}"
collect_failure train120 "${TRAIN120_JOB}" "${TRAIN120_DIR}"
```

只对实际失败的阶段调用对应一行。不要在尚未赋值的变量上运行。

## 17. Windows：下载小型诊断包

先在 Torch 输出的 `DIAGNOSTIC_BUNDLE=` 行读取真实 JobID。然后在 Windows PowerShell 中执行；命令会交互读取 JobID，不含假数字：

```powershell
$JobId = Read-Host '输入 collect_failure 输出中的真实 JobID'
$Stage = Read-Host '输入阶段名：reviewer_probe、smoke、probe5、probe20 或 train120'
$LocalDiag = 'C:\Users\20661\Desktop\Research\AR\multiuser\score-only-reward-model\torch_diagnostics'
New-Item -ItemType Directory -Force -Path $LocalDiag | Out-Null
$Batch = Join-Path $env:TEMP "score-only-get-$JobId.sftp"
@(
  "lcd $($LocalDiag.Replace('\','/'))",
  'cd /scratch/xl6775/projects/score-only-reward-model/outputs/grpo_score_only/diagnostics',
  "get $Stage`_$JobId.tar.gz",
  'bye'
) | Set-Content -LiteralPath $Batch -Encoding ascii
sftp -b $Batch xl6775@dtn.torch.hpc.nyu.edu
Get-FileHash -LiteralPath (Join-Path $LocalDiag "$Stage`_$JobId.tar.gz") -Algorithm SHA256
```

只下载日志、JSON、CSV、manifest 和压缩诊断包；不要下载模型、adapter、视频或 cache 作为常规诊断。

## 18. 当前已证明、未证明与下一步

当前已经证明：

- 本地 `ckpt1` 是 9 文件完整目录，关键 SHA 与原始压缩包核对结果一致；
- checkpoint 合同是三个 score-only cumulative ordinal heads，部署权重为 0.4/0.4/0.2；
- GRPO reward/plugin/service 和 Slurm 合同测试通过；
- reviewer-only probe 与 1/5/20/120-step 提交命令均使用真实脚本、真实资源字段和 JobID 派生产物；
- 登录 Shell 命令不使用 `exit`、`logout`、`exec`、`|| exit 1` 或全局裸 `set -e`。

当前没有证明：

- Torch 远端代码、环境、模型、数据或视频现在可用；
- `egoqa-ms-swift-v4.2.2-vllm024` 环境实际含 `ms-swift==4.2.2`、TorchCodec 和运行时编译工具；
- reviewer 能在 H100 上加载，或 2×8B 模型能在 2×H100/160G 内完成 GRPO；
- 1、5、20 或 120 step 已成功；
- reward 随训练改善、generator 收敛或 held-out QA 质量改善。

下一步唯一动作：先执行第 1–7 节，取得真实 `data_checkpoint_audit.json` 和数据 SHA；两者通过后只提交第 8 节 reviewer-only probe，不要直接提交双卡训练。

## 19. 固定汇报格式

```text
阶段：
本地 branch / HEAD：codex/score-only-reward-model / fc148f4564ab486fe70f0a87c283d08772e650d0
窄同步 manifest SHA：
dataset / split SHA-256：
checkpoint 关键 SHA：
Job ID：
JobID 推导输出目录：
Slurm 顶层 / batch State 与 ExitCode：
第一个失败 Gate：
数据 rows / evidence / question_type / generator-video / reviewer-video：
reviewer frozen health / reward / expected scores：
reward rows / valid reviewer rows / deterministic rejection rows：
reward mean / std：
global_step / grad_norm / LoRA-B nonzero：
adapter 与 manifest：
Elapsed / MaxRSS / GPU peak：
本次能证明：
本次不能证明：
下一步唯一动作：
```
