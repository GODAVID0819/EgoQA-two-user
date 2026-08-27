# 六用户视频问答：Torch/H100 运行手册

## 0. 实验目标与证据边界

本手册只运行六用户视频问答数据生成，不训练或修改生成器、DPO、GRPO、reviewer、优化器与 checkpoint。

固定实验合同如下：

- 输入：同一同步组中的 6 名用户，即 1 名 speaker、2 名 anchor provider、3 名 additional provider。
- 候选边：计算 6 名用户之间全部 15 条无向边；speaker 至少有 2 个保留邻居。
- 生成媒体：speaker 裁剪视频、2 个 anchor provider 裁剪视频、3 个 additional provider 完整视频，共 6 个视频。
- groundedness：一次查看 6 个完整原视频。
- answerability：只调用两次 VLM；第一次只看 speaker 完整视频，第二次看 6 个完整视频。
- 接受条件：`speaker_only_correct=false` 且 `all_six_correct=true`。
- `all_six_wrong` 只记为中性失败，不自动归因为噪声。
- runtime probe：目标接受 1 条 QA，候选目标 1 条，最多检查 16 个同步组，只用于首次 H100、Qwen、已固定的 decord 后端和完整六视频链路验证。
- pilot：最多尝试 40 个候选，每个候选只运行 1 个完整 loop；目标接受 40 条 QA，但接受数为 1–39 时以 `partial` 正常保留全部结果，不是第二次 smoke。

当前本地已验证：

- 分支：`feature/multi-user-six-video-qa`。
- Job `15959693` 的失败已固化为 Torch 作业合同回归测试：六用户作业固定 `FORCE_QWENVL_VIDEO_READER=decord`，不再要求未被应用选择的 TorchCodec。
- Torch 登录节点已用真实 MP4 验证 `qwen-vl-utils` 选择 decord，且 decord 成功解码 600 帧视频中的首、中、末帧。
- 随后的零 GPU Qwen 预处理暴露 `max_pixels=65536` 小于隐藏默认 `min_pixels`；当前合同改为显式传入 `min_pixels=3136`、`max_pixels=65536`，不再读取隐藏默认常量。
- 作业脚本固定为 1 张 H100、8 CPU、160 GB 内存；probe 时限 1.5 小时，pilot 时限 4 小时。
- 时限依据：已有同链路 100 个完整 loop 约 8 小时，即每个 loop 约 4.8 分钟；40 个 loop 线性估计为 3.2 小时，剩余约 0.8 小时覆盖模型加载、候选准备、媒体预处理和运行波动。

当前尚未验证：

- 修复后的脚本尚未上传到 Torch，也尚未产生新的 JobID。
- 失败 provenance：Job `15959693` 获得 1 张 H100 后在 24 秒内因 `pip check` 的 decord 平台元数据提示退出；`storage_preflight.json` 为 `passed`，未进入模型加载或六视频推理。
- 尚未验证修复后的 H100 模型加载、六视频 Qwen 预处理、媒体下载和实际显存占用。
- 尚无通过的 `six_user_qa_result.json` 或人工质量结论。

三份强制依据均位于工作区公共文档目录：

- `C:\Users\20661\Desktop\Research\AR\multiuser\docs\Torch通用复现项目执行手册.md`
- `C:\Users\20661\Desktop\Research\AR\multiuser\docs\TORCH_EXPERIMENT_META_RULES_CN.md`
- `C:\Users\20661\Desktop\Research\AR\multiuser\docs\TORCH_RUNBOOK_TEMPLATE_CN.md`

模板引用的补充文件 `TORCHCODEC_FFMPEG_RUNTIME_RULES_CN.md` 当前不存在；它不是上述三份强制依据之一。本手册已按元规则和仓库 `training/grpo_v3/REMOTE_EXECUTION_GUARDRAILS_CN.md` 固定 FFmpeg 环境优先级、`LD_LIBRARY_PATH` 和实际 Qwen 视频后端 decord。

## 1. 固定路径与资源

| 项目 | 当前值 | 依据 |
|---|---|---|
| Torch 用户 | `xl6775` | 公共元规则 |
| 登录节点 | `login.torch.hpc.nyu.edu` | 公共执行手册 |
| 远端工程目录 | `/scratch/xl6775/projects/EgoQA-two-user-grpo-clean` | 公共元规则与既有 QA 作业 |
| 输出根目录 | `/scratch/xl6775/projects/EgoQA-two-user-grpo-clean/outputs/six_user_qa` | 本实验作业脚本 |
| QA 环境 | `/scratch/xl6775/conda/envs/qwen3vl-smoke` | 公共元规则 |
| FFmpeg 环境 | `/scratch/xl6775/envs/egoqa-ffmpeg-runtime` | 公共元规则 |
| Qwen 模型 | `/scratch/xl6775/models/Qwen3.6-27B` | 公共元规则 |
| CLIP 模型标识 | `openai/clip-vit-base-patch32` | 既有候选挖掘入口 |
| 脚本所写 account | `torch_pr_674_tandon_advanced` | 公共元规则；提交前必须以实时查询复核 |
| GPU | 1 张 H100 | 作业脚本与本轮实验要求 |

若实时查询与表中 account、可用 partition、QOS 或 H100 资源冲突，打印 `STOP:`，不要提交。不得用旧作业的 account 替代实时查询。

## 2. Windows：本地最终检查

运行位置：Windows PowerShell。

```powershell
$Repo = 'C:\Users\20661\Desktop\Research\AR\multiuser\EgoQA-two-user-multi-user-prompt'
Set-Location -LiteralPath $Repo

git rev-parse --show-toplevel
git branch --show-current
git status --short

& .\.venv\Scripts\python.exe -m pytest `
  tests/test_six_user_prompts.py `
  tests/test_six_user_group_relative_sampling.py `
  tests/test_six_user_video_qa_loop.py `
  tests/test_six_user_torch_job_contract.py -q

git diff --check
```

期望：分支为 `feature/multi-user-six-video-qa`；定向测试和 `git diff --check` 通过；未提交修改只包含 `AGENTS.md`、本手册、runtime probe、`qwen3vl_runner.py`、对应合同/runner 测试和远程运行约束。若出现其他文件，先核对来源，不上传。

## 3. Torch 登录节点：实时 account、partition、QOS 与路径审计

运行位置：Torch SSH 登录 shell。此块不会提交作业，也不会关闭登录会话。

```bash
PROJECT_ROOT=/scratch/xl6775/projects/EgoQA-two-user-grpo-clean
OUTPUT_ROOT=${PROJECT_ROOT}/outputs/six_user_qa
TRAIN_ENV=/scratch/xl6775/conda/envs/qwen3vl-smoke
FFMPEG_ENV=/scratch/xl6775/envs/egoqa-ffmpeg-runtime
MODEL_DIR=/scratch/xl6775/models/Qwen3.6-27B
TORCH_ACCOUNT=torch_pr_674_tandon_advanced
AUDIT_DIR=${OUTPUT_ROOT}/login_audits
AUDIT_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
AUDIT_FILE=${AUDIT_DIR}/resource_audit_${AUDIT_STAMP}.txt

mkdir -p "${AUDIT_DIR}"
{
  echo "user=${USER}"
  echo "account_expected=${TORCH_ACCOUNT}"
  echo "project_root=${PROJECT_ROOT}"
  echo "train_env=${TRAIN_ENV}"
  echo "ffmpeg_env=${FFMPEG_ENV}"
  echo "model_dir=${MODEL_DIR}"
  echo "--- associations ---"
  sacctmgr -nP show assoc where user="${USER}" format=User,Account,Partition,QOS,DefaultQOS
  echo "--- default account ---"
  sacctmgr -nP show user where name="${USER}" format=User,DefaultAccount
  echo "--- partitions and gpu features ---"
  sinfo -h -o '%P|%a|%l|%G|%f'
  echo "--- account qos ---"
  sacctmgr -nP show assoc where user="${USER}" account="${TORCH_ACCOUNT}" format=User,Account,Partition,QOS,DefaultQOS
} | tee "${AUDIT_FILE}"

ACCOUNT_OK=0
H100_OK=0
if sacctmgr -nP show assoc where user="${USER}" account="${TORCH_ACCOUNT}" format=Account | grep -q '^torch_pr_674_tandon_advanced|'; then
  ACCOUNT_OK=1
else
  echo "STOP: 当前用户没有查询到 ${TORCH_ACCOUNT} 关联；不要提交。"
fi
if sinfo -h -o '%a|%G|%f' | grep -i 'up' | grep -qi 'h100'; then
  H100_OK=1
else
  echo "STOP: 当前集群查询未发现可用 H100 资源描述；不要提交。"
fi

for REQUIRED_PATH in "${PROJECT_ROOT}" "${TRAIN_ENV}/bin/python" "${FFMPEG_ENV}/bin/ffmpeg" "${FFMPEG_ENV}/bin/ffprobe" "${MODEL_DIR}/config.json"; do
  if [[ -e "${REQUIRED_PATH}" ]]; then
    echo "FOUND: ${REQUIRED_PATH}"
  else
    echo "MISSING: ${REQUIRED_PATH}"
  fi
done

echo "audit_file=${AUDIT_FILE}"
echo "account_ok=${ACCOUNT_OK} h100_ok=${H100_OK}"
```

只有 `account_ok=1 h100_ok=1` 且五个固定路径均为 `FOUND:` 时才继续。若实时查询显示必须指定 partition 或 QOS，先按查询结果修改提交参数并重新执行 `sbatch --test-only`；不要猜值。

## 4. Torch 登录节点：建立窄范围 SFTP 目标目录

运行位置：Torch SSH 登录 shell。

```bash
PROJECT_ROOT=/scratch/xl6775/projects/EgoQA-two-user-grpo-clean
if [[ -d "${PROJECT_ROOT}" ]]; then
  mkdir -p \
    "${PROJECT_ROOT}/hpc/logs" \
    "${PROJECT_ROOT}/hpc/qa/smoke" \
    "${PROJECT_ROOT}/hpc/qa/experiments" \
    "${PROJECT_ROOT}/tests" \
    "${PROJECT_ROOT}/docs" \
    "${PROJECT_ROOT}/outputs/six_user_qa/submission_manifests"
  echo "READY: SFTP 目标目录已建立。"
else
  echo "STOP: 工程目录不存在，未建立任何上传目录。"
fi
```

## 5. Windows：窄范围交互式 SFTP 上传

运行位置：Windows PowerShell。只覆盖本功能明确列出的代码、测试、脚本和手册；不触碰远端数据与历史输出。

```text
sftp xl6775@login.torch.hpc.nyu.edu
lcd C:/Users/20661/Desktop/Research/AR/multiuser/EgoQA-two-user-multi-user-prompt
cd /scratch/xl6775/projects/EgoQA-two-user-grpo-clean
put AGENTS.md
put qwen3vl_runner.py
lcd C:/Users/20661/Desktop/Research/AR/multiuser/EgoQA-two-user-multi-user-prompt/tests
cd /scratch/xl6775/projects/EgoQA-two-user-grpo-clean/tests
put test_six_user_torch_job_contract.py
put test_qwen_runner_compat.py
lcd C:/Users/20661/Desktop/Research/AR/multiuser/EgoQA-two-user-multi-user-prompt/hpc/qa/smoke
cd /scratch/xl6775/projects/EgoQA-two-user-grpo-clean/hpc/qa/smoke
put run_six_user_qa_runtime_probe.sbatch
lcd C:/Users/20661/Desktop/Research/AR/multiuser/EgoQA-two-user-multi-user-prompt/hpc/qa/experiments
cd /scratch/xl6775/projects/EgoQA-two-user-grpo-clean/hpc/qa/experiments
put run_six_user_qa_pilot_40.sbatch
lcd C:/Users/20661/Desktop/Research/AR/multiuser/EgoQA-two-user-multi-user-prompt/training/grpo_v3
cd /scratch/xl6775/projects/EgoQA-two-user-grpo-clean/training/grpo_v3
put REMOTE_EXECUTION_GUARDRAILS_CN.md
lcd C:/Users/20661/Desktop/Research/AR/multiuser/EgoQA-two-user-multi-user-prompt/docs
cd /scratch/xl6775/projects/EgoQA-two-user-grpo-clean/docs
put SIX_USER_QA_TORCH_RUNBOOK_CN.md
bye
```

## 6. Torch 登录节点：上传接收与零 GPU 静态检查

运行位置：Torch SSH 登录 shell。

```bash
PROJECT_ROOT=/scratch/xl6775/projects/EgoQA-two-user-grpo-clean
TRAIN_ENV=/scratch/xl6775/conda/envs/qwen3vl-smoke
MISSING_COUNT=0
REQUIRED_FILES=(
  AGENTS.md
  prompts.py
  schema.py
  video_qa_loop.py
  group_relative_clip_sampling.py
  qwen3vl_runner.py
  six_video_qa_tester.py
  six_view_packet_prep.py
  tests/test_six_user_prompts.py
  tests/test_six_user_group_relative_sampling.py
  tests/test_six_user_video_qa_loop.py
  tests/test_six_user_torch_job_contract.py
  tests/test_qwen_runner_compat.py
  hpc/qa/smoke/run_six_user_qa_runtime_probe.sbatch
  hpc/qa/experiments/run_six_user_qa_pilot_40.sbatch
  training/grpo_v3/REMOTE_EXECUTION_GUARDRAILS_CN.md
  training/torch_storage_preflight.py
)

for RELATIVE_PATH in "${REQUIRED_FILES[@]}"; do
  if [[ -s "${PROJECT_ROOT}/${RELATIVE_PATH}" ]]; then
    echo "FOUND: ${RELATIVE_PATH}"
  else
    echo "MISSING: ${RELATIVE_PATH}"
    MISSING_COUNT=$((MISSING_COUNT + 1))
  fi
done

if [[ "${MISSING_COUNT}" -eq 0 && -x "${TRAIN_ENV}/bin/python" ]]; then
  cd "${PROJECT_ROOT}"
  source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
  conda activate "${TRAIN_ENV}"
  STATIC_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
  STATIC_ALIAS_ROOT=/scratch/${USER}/job_scratch/six_user_qa_login_static_${STATIC_STAMP}/pythonpath
  mkdir -p "${STATIC_ALIAS_ROOT}"
  ln -s "${PROJECT_ROOT}" "${STATIC_ALIAS_ROOT}/egolife_two_user_qa"
  export PYTHONPATH="${STATIC_ALIAS_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"
  STATIC_OK=1
  if bash -n \
    hpc/qa/smoke/run_six_user_qa_runtime_probe.sbatch \
    hpc/qa/experiments/run_six_user_qa_pilot_40.sbatch; then
    echo "PASSED: Bash 语法检查。"
  else
    echo "STOP: Bash 语法检查失败。"
    STATIC_OK=0
  fi
  if "${TRAIN_ENV}/bin/python" -m compileall -q \
    prompts.py schema.py video_qa_loop.py group_relative_clip_sampling.py \
    qwen3vl_runner.py six_video_qa_tester.py six_view_packet_prep.py; then
    echo "PASSED: Python 语法检查。"
  else
    echo "STOP: Python 语法检查失败。"
    STATIC_OK=0
  fi
  if "${TRAIN_ENV}/bin/python" -m pytest \
    tests/test_six_user_prompts.py \
    tests/test_six_user_group_relative_sampling.py \
    tests/test_six_user_video_qa_loop.py \
    tests/test_six_user_torch_job_contract.py \
    tests/test_qwen_runner_compat.py::test_memory_safe_runner_has_long_context_guards_by_default \
    tests/test_qwen_runner_compat.py::test_memory_safe_runner_passes_explicit_video_minimum -q; then
    echo "PASSED: 六用户定向测试。"
  else
    echo "STOP: 六用户定向测试失败。"
    STATIC_OK=0
  fi
  if [[ "${STATIC_OK}" -eq 1 ]]; then
    echo "REMOTE_STATIC_CHECK_PASSED"
  fi
else
  echo "STOP: 接收检查或 Python 环境失败，跳过静态测试。"
fi
```

这一步只证明远端文件能被目标 Python 解析且定向测试通过，不证明 CUDA、Qwen 模型或六视频推理可运行。

## 6.1 Torch 登录节点：零 GPU 视频后端与 Qwen 预处理检查

运行位置：Torch SSH 登录 shell。必须在提交 GPU probe 前执行；只读取一个已有视频，不加载模型、不申请 GPU，也不会关闭登录会话。

```bash
PROJECT_ROOT=/scratch/xl6775/projects/EgoQA-two-user-grpo-clean
TRAIN_ENV=/scratch/xl6775/conda/envs/qwen3vl-smoke
FFMPEG_ENV=/scratch/xl6775/envs/egoqa-ffmpeg-runtime
OUTPUT_ROOT=${PROJECT_ROOT}/outputs/six_user_qa
AUDIT_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
AUDIT_DIR=${OUTPUT_ROOT}/login_audits
PIP_CHECK_FILE=${AUDIT_DIR}/pip_check_${AUDIT_STAMP}.txt
PYTHON=${TRAIN_ENV}/bin/python

mkdir -p "${AUDIT_DIR}"
export PATH="${FFMPEG_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${FFMPEG_ENV}/lib:${LD_LIBRARY_PATH:-}"
export FORCE_QWENVL_VIDEO_READER=decord
export QWEN_MEMORY_SAFE_MIN_VIDEO_PIXELS=3136

PRECHECK_OK=1
if "${PYTHON}" -m pip check > "${PIP_CHECK_FILE}" 2>&1; then
  echo "pip_check_status=passed"
else
  UNEXPECTED_PIP_CHECK="$(
    grep -Ev '^decord [^[:space:]]+ is not supported on this platform$' \
      "${PIP_CHECK_FILE}" || true
  )"
  if [[ -n "${UNEXPECTED_PIP_CHECK}" ]]; then
    echo "STOP: pip check 发现 decord 平台元数据提示之外的依赖错误。"
    printf '%s\n' "${UNEXPECTED_PIP_CHECK}"
    PRECHECK_OK=0
  else
    echo "pip_check_status=known_decord_platform_metadata_warning"
  fi
fi
cat "${PIP_CHECK_FILE}"

VIDEO_PATH=$(
  find "${PROJECT_ROOT}/outputs" -maxdepth 8 -type f \
    \( -iname '*.mp4' -o -iname '*.mov' -o -iname '*.mkv' \) \
    -print -quit 2>/dev/null
)

if [[ "${PRECHECK_OK}" -eq 1 && -n "${VIDEO_PATH}" && -f "${VIDEO_PATH}" ]]; then
  "${PYTHON}" - "${VIDEO_PATH}" <<'PY'
import sys

import decord
from qwen_vl_utils import process_vision_info
from qwen_vl_utils.vision_process import get_video_reader_backend

video_path = sys.argv[1]
backend = get_video_reader_backend()
if backend != "decord":
    raise SystemExit(f"unexpected Qwen video backend: {backend}")

reader = decord.VideoReader(video_path)
frame_count = len(reader)
if frame_count <= 0:
    raise SystemExit("video frame count is zero")
indices = sorted({0, frame_count // 2, frame_count - 1})
frames = reader.get_batch(indices).asnumpy()

messages = [{
    "role": "user",
    "content": [
        {
            "type": "video",
            "video": video_path,
            "min_pixels": 3136,
            "max_pixels": 65536,
            "fps": 1.0,
        },
        {"type": "text", "text": "zero-GPU runtime preflight"},
    ],
}]
try:
    _, video_inputs, _ = process_vision_info(
        messages,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
except TypeError:
    try:
        _, video_inputs, _ = process_vision_info(messages, return_video_kwargs=True)
    except TypeError:
        _, video_inputs = process_vision_info(messages)
if video_inputs is None or len(video_inputs) != 1:
    raise SystemExit("Qwen video preprocessing did not return exactly one video")

print(f"video_path={video_path}")
print(f"video_backend={backend}")
print(f"frame_count={frame_count}")
print(f"decoded_shape={frames.shape}")
print("ZERO_GPU_RUNTIME_PREFLIGHT_PASSED")
PY
  PRECHECK_RC=$?
  if [[ "${PRECHECK_RC}" -ne 0 ]]; then
    echo "STOP: 零 GPU 视频后端或 Qwen 预处理检查失败，不提交 GPU probe。"
  fi
else
  echo "STOP: 依赖检查失败或没有找到真实视频，不提交 GPU probe。"
fi
```

只有出现 `ZERO_GPU_RUNTIME_PREFLIGHT_PASSED` 才提交 runtime probe。该结果证明当前环境可用 decord 和显式像素区间 `3136 <= pixels <= 65536` 走真实 Qwen 视频预处理，但不证明 CUDA、27B 模型或六视频推理可运行。

## 7. 阶段与唯一 smoke

| 阶段 | 脚本 | 输入 | 资源 | 通过条件 | 失败证据 |
|---|---|---|---|---|---|
| runtime probe | `hpc/qa/smoke/run_six_user_qa_runtime_probe.sbatch` | EgoLife 在线 manifest、目标 1 个候选、最多检查 16 个同步组、每候选 1 次 | 1×H100、8 CPU、160 GB、1.5 小时 | 作业成功且 `six_user_qa_result.json` 为 `passed`，接受数至少 1；模型加载前完成 6 段真实视频 Qwen 预处理，生成/groundedness/answerability 视频数分别为 6/6/两次调用 | `.out`、`.err`、`storage_preflight.json`、`pip_check.txt`、`job_manifest.json`、候选、接受、拒绝和 intermediate JSONL |
| pilot | `hpc/qa/experiments/run_six_user_qa_pilot_40.sbatch` | 同一合同、最多 40 个候选、每候选 1 次 | 1×H100、8 CPU、160 GB、4 小时 | probe 先通过；接受 40 条时结果为 `passed`；接受 1–39 条时结果为 `partial` 并正常保留产物；0 条仍失败 | 同上，另加人工复核表和 review videos |

probe 是本流程唯一 smoke。probe 通过后直接运行 pilot，不再增加其他规模的 smoke。`job_manifest.json` 在 `pip check` 之前以 `preflight` 状态落盘；前置检查通过后更新为 `running`，因此前置失败也保留可定位的任务身份和环境合同。

## 8. 提交 runtime probe

运行位置：Torch SSH 登录 shell。该块重新声明所有路径；失败只打印 `STOP:`，不会关闭登录会话。

```bash
PROJECT_ROOT=/scratch/xl6775/projects/EgoQA-two-user-grpo-clean
OUTPUT_ROOT=${PROJECT_ROOT}/outputs/six_user_qa
TRAIN_ENV=/scratch/xl6775/conda/envs/qwen3vl-smoke
FFMPEG_ENV=/scratch/xl6775/envs/egoqa-ffmpeg-runtime
MODEL_DIR=/scratch/xl6775/models/Qwen3.6-27B
TORCH_ACCOUNT=torch_pr_674_tandon_advanced
SCRIPT=${PROJECT_ROOT}/hpc/qa/smoke/run_six_user_qa_runtime_probe.sbatch
MANIFEST_ROOT=${OUTPUT_ROOT}/submission_manifests

ACCOUNT_OK=1
H100_OK=1
FILES_OK=1
if sacctmgr -nP show assoc where user="${USER}" account="${TORCH_ACCOUNT}" format=Account | grep -q '^torch_pr_674_tandon_advanced|'; then ACCOUNT_OK=1; fi
if sinfo -h -o '%a|%G|%f' | grep -i 'up' | grep -qi 'h100'; then H100_OK=1; fi
for REQUIRED_PATH in "${SCRIPT}" "${TRAIN_ENV}/bin/python" "${FFMPEG_ENV}/bin/ffmpeg" "${FFMPEG_ENV}/bin/ffprobe" "${MODEL_DIR}/config.json"; do
  if [[ ! -e "${REQUIRED_PATH}" ]]; then
    echo "MISSING: ${REQUIRED_PATH}"
    FILES_OK=0
  fi
done

if [[ "${ACCOUNT_OK}" -eq 1 && "${H100_OK}" -eq 1 && "${FILES_OK}" -eq 1 ]]; then
  mkdir -p "${PROJECT_ROOT}/hpc/logs" "${MANIFEST_ROOT}"
  if sbatch --test-only \
    --account="${TORCH_ACCOUNT}" \
    --constraint=h100 \
    --chdir="${PROJECT_ROOT}" \
    "${SCRIPT}"; then
    SUBMIT_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
    RAW_JOB_ID=$(sbatch --parsable \
      --account="${TORCH_ACCOUNT}" \
      --constraint=h100 \
      --chdir="${PROJECT_ROOT}" \
      --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",OUTPUT_ROOT="${OUTPUT_ROOT}",TRAIN_ENV="${TRAIN_ENV}",FFMPEG_ENV="${FFMPEG_ENV}",MODEL_DIR="${MODEL_DIR}" \
      "${SCRIPT}")
    SUBMIT_RC=$?
    JOB_ID=${RAW_JOB_ID%%;*}
    if [[ "${SUBMIT_RC}" -eq 0 && "${JOB_ID}" =~ ^[0-9]+$ ]]; then
      TASK_MANIFEST=${MANIFEST_ROOT}/runtime_probe_${SUBMIT_STAMP}_${JOB_ID}.env
      {
        printf 'RUN_KIND=%q\n' runtime_probe
        printf 'JOB_ID=%q\n' "${JOB_ID}"
        printf 'SBATCH_RESPONSE=%q\n' "${RAW_JOB_ID}"
        printf 'SCRIPT=%q\n' "${SCRIPT}"
        printf 'OUTPUT_DIR=%q\n' "${OUTPUT_ROOT}/six_user_qa_runtime_probe_${JOB_ID}"
        printf 'STDOUT=%q\n' "${PROJECT_ROOT}/hpc/logs/egoqa_6u_probe_${JOB_ID}.out"
        printf 'STDERR=%q\n' "${PROJECT_ROOT}/hpc/logs/egoqa_6u_probe_${JOB_ID}.err"
        printf 'SUBMITTED_AT=%q\n' "${SUBMIT_STAMP}"
      } > "${TASK_MANIFEST}"
      echo "SUBMITTED: job_id=${JOB_ID}"
      echo "TASK_MANIFEST=${TASK_MANIFEST}"
    else
      echo "STOP: sbatch 提交失败，未写入有效任务 manifest。"
    fi
  else
    echo "STOP: sbatch --test-only 未通过，未提交。"
  fi
else
  echo "STOP: account、H100 或固定路径检查未通过，未提交。"
fi
```

必须保存命令打印的 `TASK_MANIFEST` 路径。后续所有监控、Gate、pilot 和下载都从该文件读取 JobID，不要求手工记忆 JobID。
baocun:TASK_MANIFEST=/scratch/xl6775/projects/EgoQA-two-user-grpo-clean/outputs/six_user_qa/submission_manifests/runtime_probe_20260819T132419Z_16021016.env

## 9. 从任务 manifest 监控任一作业

运行位置：Torch SSH 登录 shell。

```bash
read -r -p '粘贴提交时打印的 TASK_MANIFEST 绝对路径: ' TASK_MANIFEST
EXPECTED_ROOT=/scratch/xl6775/projects/EgoQA-two-user-grpo-clean/outputs/six_user_qa/submission_manifests
if [[ -f "${TASK_MANIFEST}" && "${TASK_MANIFEST}" == "${EXPECTED_ROOT}/"*.env ]]; then
  . "${TASK_MANIFEST}"
  echo "job_id=${JOB_ID}"
  echo "output_dir=${OUTPUT_DIR}"
  sacct -j "${JOB_ID}" --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,AllocTRES -P
  if [[ -f "${STDOUT}" ]]; then tail -n 80 "${STDOUT}"; else echo "MISSING: ${STDOUT}"; fi
  if [[ -f "${STDERR}" ]]; then tail -n 80 "${STDERR}"; else echo "MISSING: ${STDERR}"; fi
  if [[ -f "${OUTPUT_DIR}/six_user_qa_result.json" ]]; then
    cat "${OUTPUT_DIR}/six_user_qa_result.json"
  else
    echo "MISSING: ${OUTPUT_DIR}/six_user_qa_result.json"
  fi
else
  echo "STOP: 任务 manifest 不存在或不在本实验 manifest 目录中。"
fi
```

`COMPLETED|0:0` 只表示批处理进程退出成功；仍必须检查结果 JSON、接受数、媒体路由指标和人工复核材料。

## 10. runtime probe 产物 Gate

运行位置：Torch SSH 登录 shell。输入 runtime probe 的任务 manifest 路径。

```bash
read -r -p '粘贴 runtime probe 的 TASK_MANIFEST 绝对路径: ' TASK_MANIFEST
EXPECTED_ROOT=/scratch/xl6775/projects/EgoQA-two-user-grpo-clean/outputs/six_user_qa/submission_manifests
if [[ -f "${TASK_MANIFEST}" && "${TASK_MANIFEST}" == "${EXPECTED_ROOT}/"*.env ]]; then
  . "${TASK_MANIFEST}"
  RESULT_FILE=${OUTPUT_DIR}/six_user_qa_result.json
  if [[ "${RUN_KIND}" == runtime_probe && -f "${RESULT_FILE}" ]]; then
    /scratch/xl6775/conda/envs/qwen3vl-smoke/bin/python - "${RESULT_FILE}" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "status": "passed",
    "generator_video_count": 6,
    "groundedness_video_count": 6,
    "answerability_call_count": 2,
    "speaker_only_correct": False,
    "all_six_correct": True,
    "cross_view_gain": 1,
    "answerability_evaluated_condition_count": 2,
}
errors = [f"{key}={result.get(key)!r}" for key, value in expected.items() if result.get(key) != value]
if int(result.get("accepted_count", 0)) < 1:
    errors.append(f"accepted_count={result.get('accepted_count')!r}")
if errors:
    raise SystemExit("PROBE_GATE_FAILED: " + ", ".join(errors))
print("PROBE_GATE_PASSED")
print(json.dumps(result, ensure_ascii=False, indent=2))
PY
  else
    echo "STOP: 不是 runtime probe manifest，或结果文件缺失。"
  fi
else
  echo "STOP: runtime probe manifest 无效。"
fi
```

`all_six_wrong_count` 和 `all_six_wrong_rate` 必须汇报，但没有固定通过阈值；它们用于观察六视频噪声风险，不单独证明候选有问题。

## 11. probe 通过后提交 pilot

运行位置：Torch SSH 登录 shell。只有上一节打印 `PROBE_GATE_PASSED` 后执行。

```bash
read -r -p '粘贴已通过 Gate 的 runtime probe TASK_MANIFEST 绝对路径: ' PROBE_TASK_MANIFEST
PROJECT_ROOT=/scratch/xl6775/projects/EgoQA-two-user-grpo-clean
OUTPUT_ROOT=${PROJECT_ROOT}/outputs/six_user_qa
TRAIN_ENV=/scratch/xl6775/conda/envs/qwen3vl-smoke
FFMPEG_ENV=/scratch/xl6775/envs/egoqa-ffmpeg-runtime
MODEL_DIR=/scratch/xl6775/models/Qwen3.6-27B
TORCH_ACCOUNT=torch_pr_674_tandon_advanced
SCRIPT=${PROJECT_ROOT}/hpc/qa/experiments/run_six_user_qa_pilot_40.sbatch
MANIFEST_ROOT=${OUTPUT_ROOT}/submission_manifests
PROBE_OK=0

if [[ -f "${PROBE_TASK_MANIFEST}" && "${PROBE_TASK_MANIFEST}" == "${MANIFEST_ROOT}/"runtime_probe_*.env ]]; then
  . "${PROBE_TASK_MANIFEST}"
  PROBE_RESULT=${OUTPUT_DIR}/six_user_qa_result.json
  if [[ -f "${PROBE_RESULT}" ]]; then
    PROBE_OK=$("${TRAIN_ENV}/bin/python" - "${PROBE_RESULT}" <<'PY'
import json
import sys
from pathlib import Path
result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
ok = result.get("status") == "passed" and int(result.get("accepted_count", 0)) >= 1
print(1 if ok else 0)
PY
)
  fi
fi

if [[ "${PROBE_OK}" -eq 1 ]]; then
  if sbatch --test-only \
    --account="${TORCH_ACCOUNT}" \
    --constraint=h100 \
    --chdir="${PROJECT_ROOT}" \
    "${SCRIPT}"; then
    SUBMIT_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
    RAW_JOB_ID=$(sbatch --parsable \
      --account="${TORCH_ACCOUNT}" \
      --constraint=h100 \
      --chdir="${PROJECT_ROOT}" \
      --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",OUTPUT_ROOT="${OUTPUT_ROOT}",TRAIN_ENV="${TRAIN_ENV}",FFMPEG_ENV="${FFMPEG_ENV}",MODEL_DIR="${MODEL_DIR}" \
      "${SCRIPT}")
    SUBMIT_RC=$?
    JOB_ID=${RAW_JOB_ID%%;*}
    if [[ "${SUBMIT_RC}" -eq 0 && "${JOB_ID}" =~ ^[0-9]+$ ]]; then
      TASK_MANIFEST=${MANIFEST_ROOT}/pilot_40_${SUBMIT_STAMP}_${JOB_ID}.env
      {
        printf 'RUN_KIND=%q\n' pilot_40
        printf 'JOB_ID=%q\n' "${JOB_ID}"
        printf 'SBATCH_RESPONSE=%q\n' "${RAW_JOB_ID}"
        printf 'SCRIPT=%q\n' "${SCRIPT}"
        printf 'OUTPUT_DIR=%q\n' "${OUTPUT_ROOT}/six_user_qa_pilot_40_${JOB_ID}"
        printf 'STDOUT=%q\n' "${PROJECT_ROOT}/hpc/logs/egoqa_6u_pilot40_${JOB_ID}.out"
        printf 'STDERR=%q\n' "${PROJECT_ROOT}/hpc/logs/egoqa_6u_pilot40_${JOB_ID}.err"
        printf 'PROBE_TASK_MANIFEST=%q\n' "${PROBE_TASK_MANIFEST}"
        printf 'SUBMITTED_AT=%q\n' "${SUBMIT_STAMP}"
      } > "${TASK_MANIFEST}"
      echo "SUBMITTED: job_id=${JOB_ID}"
      echo "TASK_MANIFEST=${TASK_MANIFEST}"
    else
      echo "STOP: pilot 提交失败，未写入有效任务 manifest。"
    fi
  else
    echo "STOP: pilot 的 sbatch --test-only 未通过，未提交。"
  fi
else
  echo "STOP: runtime probe 结果没有通过，未提交 pilot。"
fi
```

## 12. pilot 结果 Gate 与人工检查

运行位置：Torch SSH 登录 shell。输入 pilot 的任务 manifest 路径。

```bash
read -r -p '粘贴 pilot 的 TASK_MANIFEST 绝对路径: ' TASK_MANIFEST
EXPECTED_ROOT=/scratch/xl6775/projects/EgoQA-two-user-grpo-clean/outputs/six_user_qa/submission_manifests
if [[ -f "${TASK_MANIFEST}" && "${TASK_MANIFEST}" == "${EXPECTED_ROOT}/"pilot_40_*.env ]]; then
  . "${TASK_MANIFEST}"
  RESULT_FILE=${OUTPUT_DIR}/six_user_qa_result.json
  if [[ -f "${RESULT_FILE}" ]]; then
    /scratch/xl6775/conda/envs/qwen3vl-smoke/bin/python - "${RESULT_FILE}" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "generator_video_count": 6,
    "groundedness_video_count": 6,
    "answerability_call_count": 2,
    "speaker_only_correct": False,
    "all_six_correct": True,
    "cross_view_gain": 1,
    "answerability_evaluated_condition_count": 2,
}
errors = [f"{key}={result.get(key)!r}" for key, value in expected.items() if result.get(key) != value]
status = result.get("status")
accepted_count = int(result.get("accepted_count", 0))
if status not in {"passed", "partial"}:
    errors.append(f"status={status!r}")
if accepted_count < 1:
    errors.append(f"accepted_count={result.get('accepted_count')!r}")
if status == "passed" and accepted_count < 40:
    errors.append(f"passed_without_40={accepted_count}")
if status == "partial" and accepted_count >= 40:
    errors.append(f"partial_with_40={accepted_count}")
if errors:
    raise SystemExit("PILOT_GATE_FAILED: " + ", ".join(errors))
print("PILOT_RESULT_ACCEPTED")
print(json.dumps(result, ensure_ascii=False, indent=2))
PY
    for REQUIRED_FILE in qa_mcq.jsonl qa_mcq.csv human_review_sheet.md generation_report.md job_manifest.json storage_preflight.json; do
      if [[ -s "${OUTPUT_DIR}/${REQUIRED_FILE}" ]]; then
        echo "FOUND: ${OUTPUT_DIR}/${REQUIRED_FILE}"
      else
        echo "MISSING: ${OUTPUT_DIR}/${REQUIRED_FILE}"
      fi
    done
    find "${OUTPUT_DIR}/review_videos" -maxdepth 1 -type f -print | sort
  else
    echo "STOP: pilot 结果文件缺失。"
  fi
else
  echo "STOP: pilot 任务 manifest 无效。"
fi
```

人工检查本次全部已接受 QA（最多 40 条）：

1. question 是否必须结合非 speaker 视角，而不是从 speaker 单独推断。
2. 6 个视频中的无关信息是否诱导了错误答案或含糊措辞。
3. 正确答案是否由至少一个 provider 的可见证据支撑；不要求 5 个 provider 都贡献。
4. distractor 是否合理但能被六视频证据排除。
5. `all_six_wrong` 样本只作为失败案例单独查看，不直接标记为噪声数据。

pilot 只提供最多 40 条样本的能力与失败模式观察，不能证明总体质量提升或统计显著性。`partial` 表示候选用尽后得到 1–39 条有效 QA，不表示达到 40 条目标。

## 13. 通用失败证据收集

运行位置：Torch SSH 登录 shell。输入对应任务 manifest 路径。

```bash
read -r -p '粘贴失败作业的 TASK_MANIFEST 绝对路径: ' TASK_MANIFEST
EXPECTED_ROOT=/scratch/xl6775/projects/EgoQA-two-user-grpo-clean/outputs/six_user_qa/submission_manifests
if [[ -f "${TASK_MANIFEST}" && "${TASK_MANIFEST}" == "${EXPECTED_ROOT}/"*.env ]]; then
  . "${TASK_MANIFEST}"
  echo "--- scheduler ---"
  sacct -j "${JOB_ID}" --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,MaxVMSize,AllocTRES -P
  echo "--- stdout ---"
  if [[ -f "${STDOUT}" ]]; then tail -n 200 "${STDOUT}"; else echo "MISSING: ${STDOUT}"; fi
  echo "--- stderr ---"
  if [[ -f "${STDERR}" ]]; then tail -n 200 "${STDERR}"; else echo "MISSING: ${STDERR}"; fi
  echo "--- output files ---"
  if [[ -d "${OUTPUT_DIR}" ]]; then find "${OUTPUT_DIR}" -maxdepth 2 -type f -printf '%p|%s bytes\n' | sort; else echo "MISSING: ${OUTPUT_DIR}"; fi
  for EVIDENCE_FILE in storage_preflight.json pip_check.txt job_manifest.json six_user_qa_result.json; do
    if [[ -f "${OUTPUT_DIR}/${EVIDENCE_FILE}" ]]; then
      echo "--- ${EVIDENCE_FILE} ---"
      cat "${OUTPUT_DIR}/${EVIDENCE_FILE}"
    fi
  done
else
  echo "STOP: 任务 manifest 无效，未收集证据。"
fi
```

诊断顺序：先看 `sacct`、`.out`、`.err`，再看 `storage_preflight.json`、`job_manifest.json` 和阶段产物。重新运行必须产生新的 JobID 和新的任务 manifest；不要覆盖失败作业目录。

## 14. Windows：交互式 SFTP 下载小型结果

先在 Windows PowerShell 创建固定本地接收目录：

```powershell
New-Item -ItemType Directory -Force -Path 'C:\Users\20661\Desktop\Research\AR\multiuser\torch_downloads\six_user_qa' | Out-Null
```

然后在 Torch SSH 登录 shell 从任务 manifest 生成包含真实 JobID 和真实输出目录的下载命令：

```bash
read -r -p '粘贴要下载作业的 TASK_MANIFEST 绝对路径: ' TASK_MANIFEST
EXPECTED_ROOT=/scratch/xl6775/projects/EgoQA-two-user-grpo-clean/outputs/six_user_qa/submission_manifests
if [[ -f "${TASK_MANIFEST}" && "${TASK_MANIFEST}" == "${EXPECTED_ROOT}/"*.env ]]; then
  . "${TASK_MANIFEST}"
  printf '%s\n' \
    'sftp xl6775@login.torch.hpc.nyu.edu' \
    'lcd C:/Users/20661/Desktop/Research/AR/multiuser/torch_downloads/six_user_qa' \
    "cd ${OUTPUT_DIR}" \
    "get six_user_qa_result.json six_user_qa_result_${JOB_ID}.json" \
    "get job_manifest.json job_manifest_${JOB_ID}.json" \
    "get storage_preflight.json storage_preflight_${JOB_ID}.json" \
    "get pip_check.txt pip_check_${JOB_ID}.txt" \
    "get qa_mcq.jsonl qa_mcq_${JOB_ID}.jsonl" \
    "get qa_mcq.csv qa_mcq_${JOB_ID}.csv" \
    "get human_review_sheet.md human_review_sheet_${JOB_ID}.md" \
    "get generation_report.md generation_report_${JOB_ID}.md" \
    'bye'
else
  echo "STOP: 任务 manifest 无效，未生成下载命令。"
fi
```

把打印出的完整命令逐行粘贴到 Windows PowerShell。实际下载会严格按 `sftp`、`lcd`、`cd`、逐文件 `get`、`bye` 的顺序执行；每个本地文件名包含该作业的真实 JobID，避免覆盖另一作业结果。

## 15. 固定汇报格式

每个阶段按以下字段汇报：

```text
阶段：runtime probe 或 pilot
任务 manifest：提交块打印的绝对路径
JobID：从任务 manifest 读取
Slurm State / ExitCode：
输出目录：从任务 manifest 读取
结果 status：
accepted_target：
target_reached：
allow_partial：
candidate_count：
attempted_count：
accepted_count：
rejected_count：
generator_video_count：
groundedness_video_count：
answerability_call_count：
speaker_only_correct：
all_six_correct：
cross_view_gain：
answerability_evaluated_condition_count：
all_six_wrong_count：
all_six_wrong_rate：
人工复核发现：
可证明：
不可证明：
下一步：
```

## 16. 交付前自检结果

- 所有固定路径、环境、模型、资源和脚本名均来自当前权威文档或真实仓库文件。
- 未残留模板变量、待办标记或虚构 JobID。
- 登录 shell 命令不包含会关闭会话的命令或全局严格模式。
- 两次实际提交都使用 `sbatch --parsable`，兼容 cluster 后缀，并把 JobID 写入时间戳任务 manifest。
- 监控、Gate、pilot 依赖与下载都从任务 manifest 派生 JobID 和输出目录。
- 正式作业在模型加载前封闭 HOME、cache、临时目录并运行存储预检。
- FFmpeg 的 `PATH`、`LD_LIBRARY_PATH` 在 decord 与 Qwen 视频后端预检之前设置。
- 六用户 memory-safe 路径显式记录 `min_video_pixels=3136`、`max_image_pixels=65536`，不读取 `qwen-vl-utils` 隐藏默认下界。
- GPU 提交前先完成零 GPU 的真实视频解码和 Qwen `process_vision_info` 检查。
- runtime probe 是唯一 smoke；pilot 最多执行 40 个完整 loop，接受 1–39 条时显式记录为 `partial`。
- Job `16021016` 在 1 小时内仍停留于 8 候选挖掘阶段；probe 已缩为实际只消费的 1 个候选，并按失败实测的 1.5 倍把时限调整为 1.5 小时。pilot 仍为 4 小时；该调整不改变六用户媒体、判定或接受合同。
- 本地静态验证、登录节点检查、GPU runtime、作业完成、产物 Gate 和人工质量判断均明确分开。
- 未授权也未执行推送、上传、Slurm 提交、远端清理或扩大实验范围。
