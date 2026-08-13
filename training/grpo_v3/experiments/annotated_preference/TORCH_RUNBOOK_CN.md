# 标注 Pareto-DPO：Torch/H100 完整运行手册

本手册遵循 [Torch 实验元规则](../../../../docs/TORCH_EXPERIMENT_META_RULES_CN.md)。它从 Windows 本地代码开始，依次完成 Git 同步、Torch 独立 worktree、单文件 SFTP、数据生成与验证，以及 Gate 0–5。各段均可直接复制；不要提前运行后续 Gate。

## 0. 固定合同与执行边界

- 本地仓库：`C:\Users\20661\Desktop\Research\AR\multiuser\EgoQA-two-user-reviewer-v1`。
- Git 分支：`feature/annotated-pareto-dpo`。
- Torch 代码来源：`/scratch/xl6775/projects/EgoQA-two-user-reviewer-v1`。
- Torch 独立 worktree：`/scratch/xl6775/projects/EgoQA-two-user-annotated-pareto-dpo`。
- 环境：`/scratch/xl6775/envs/egoqa-ms-swift-v4.2.2-vllm024`。
- 模型：`/scratch/xl6775/models/Qwen3-VL-8B-Instruct`。
- CSV SHA-256：`32679019FD7C665A0632E9885405BDF13C77B51386EFC56E7B29B24192210CD7`。
- 数据：420 行、70 个 evidence、每个 evidence 6 个候选；划分固定为 `60/10/0`。
- 媒体：140 个唯一视频；`media_map.json` 必须精确覆盖 CSV 中的 URL。
- 有序角色：`video_1_*` 是 Speaker/asker，`video_2_*` 是 Provider；`video_1_user`、`video_2_user` 保存 Jake、Tasha 等真实参与者姓名，不是字面标签 `A / Speaker`、`B / Provider`。
- `compact_qa_v1` 是当前标注数据的紧凑提示词合同，不是生产 `expanded schema` 的替换。
- 登录 shell 中失败只打印 `STOP` 或 `MISSING`，并跳过依赖步骤，保持 SSH 会话可继续使用。
- 代码走 Git；SFTP 只传明确列出的单个小文件。不要递归上传项目、模型、视频、cache、`data_RLHF` 或 outputs。

## 1. Windows：检查并推送代码

在 Windows PowerShell 中整段执行：

```powershell
$Repo = 'C:\Users\20661\Desktop\Research\AR\multiuser\EgoQA-two-user-reviewer-v1'
$Branch = 'feature/annotated-pareto-dpo'
Set-Location -LiteralPath $Repo

git status --short --branch
$CurrentBranch = git branch --show-current
$Dirty = git status --porcelain

if ($CurrentBranch -ne $Branch) {
  Write-Host "STOP: 当前分支是 $CurrentBranch，需要 $Branch"
} elseif ($Dirty) {
  Write-Host 'STOP: 工作树不干净；先提交本次 Runbook 修改，再推送'
} else {
  git push -u origin $Branch
  $LocalHead = git rev-parse HEAD
  $RemoteHead = (git ls-remote origin "refs/heads/$Branch" | ForEach-Object { ($_ -split '\s+')[0] })
  if ($LocalHead -eq $RemoteHead) {
    Write-Host "GIT_PUSH_PASSED head=$LocalHead"
  } else {
    Write-Host "STOP: local=$LocalHead remote=$RemoteHead"
  }
}
```

只有看到 `GIT_PUSH_PASSED` 才继续。该步骤只发布当前分支，不上传数据、模型或输出。

## 2. Torch：拉取分支并建立独立 worktree

先 SSH 登录：

```text
ssh xl6775@greene.hpc.nyu.edu
```

登录后整段执行。用户此前的失败命令可能已创建一个不含 Git 的空实验目录；下面会先把它改名备份，不删除其中内容：

```bash
SOURCE_ROOT=/scratch/xl6775/projects/EgoQA-two-user-reviewer-v1
PROJECT_ROOT=/scratch/xl6775/projects/EgoQA-two-user-annotated-pareto-dpo
BRANCH=feature/annotated-pareto-dpo

if [ ! -e "${SOURCE_ROOT}/.git" ]; then
  echo "STOP: source repository missing: ${SOURCE_ROOT}"
else
  git -C "${SOURCE_ROOT}" fetch origin "${BRANCH}:refs/remotes/origin/${BRANCH}"
fi

if [ -e "${PROJECT_ROOT}/.git" ]; then
  echo "worktree already exists: ${PROJECT_ROOT}"
elif [ -e "${PROJECT_ROOT}" ]; then
  BACKUP_ROOT=${PROJECT_ROOT}.pre_worktree_$(date +%Y%m%d_%H%M%S)
  mv "${PROJECT_ROOT}" "${BACKUP_ROOT}"
  echo "existing non-Git directory preserved at ${BACKUP_ROOT}"
fi

if [ ! -e "${PROJECT_ROOT}/.git" ] && [ -e "${SOURCE_ROOT}/.git" ]; then
  if git -C "${SOURCE_ROOT}" show-ref --verify --quiet "refs/heads/${BRANCH}"; then
    git -C "${SOURCE_ROOT}" worktree add "${PROJECT_ROOT}" "${BRANCH}"
  else
    git -C "${SOURCE_ROOT}" worktree add -b "${BRANCH}" "${PROJECT_ROOT}" "origin/${BRANCH}"
  fi
fi

if [ -e "${PROJECT_ROOT}/.git" ]; then
  LOCAL_EXCLUDE=$(git -C "${PROJECT_ROOT}" rev-parse --git-path info/exclude)
  mkdir -p "$(dirname "${LOCAL_EXCLUDE}")"
  for PATTERN in "/data_RLHF/" "/outputs/" "/logs/"; do
    if ! grep -Fqx "${PATTERN}" "${LOCAL_EXCLUDE}" 2>/dev/null; then
      printf '%s\n' "${PATTERN}" >> "${LOCAL_EXCLUDE}"
      echo "ADDED_LOCAL_EXCLUDE: ${PATTERN}"
    fi
  done
  if git -C "${PROJECT_ROOT}" diff --quiet \
    && git -C "${PROJECT_ROOT}" diff --cached --quiet; then
    git -C "${PROJECT_ROOT}" merge --ff-only "origin/${BRANCH}"
  else
    echo "STOP: target worktree has tracked or staged changes; fast-forward skipped"
    git -C "${PROJECT_ROOT}" status --short --untracked-files=no
  fi
  if [ -n "${BACKUP_ROOT:-}" ]; then
    cp -a -n "${BACKUP_ROOT}/." "${PROJECT_ROOT}/"
    echo "preserved files copied without overwriting Git files; backup retained at ${BACKUP_ROOT}"
  fi
else
  echo "STOP: worktree creation failed"
fi

if [ "$(git -C "${PROJECT_ROOT}" branch --show-current 2>/dev/null)" = "${BRANCH}" ]; then
  LOCAL_HEAD=$(git -C "${PROJECT_ROOT}" rev-parse HEAD)
  REMOTE_HEAD=$(git -C "${PROJECT_ROOT}" rev-parse "origin/${BRANCH}")
  if [ "${LOCAL_HEAD}" = "${REMOTE_HEAD}" ]; then
    echo "GIT_SYNC_PASSED branch=${BRANCH} head=${LOCAL_HEAD}"
  else
    echo "STOP: worktree head does not match origin branch"
  fi
else
  echo "STOP: wrong or missing worktree branch"
fi

mkdir -p "${PROJECT_ROOT}/logs" \
  "${PROJECT_ROOT}/data_RLHF/annotated_preference" \
  "${PROJECT_ROOT}/outputs/annotated_preference"
git -C "${PROJECT_ROOT}" status --short --branch
```

只有看到 `GIT_SYNC_PASSED` 才继续。若输出了备份路径，暂时保留该目录，确认其中没有需要恢复的文件后再人工处理。

## 3. Windows：用 SFTP 只上传标注 CSV

Torch 的目标数据目录已由上一节创建。回到 Windows PowerShell，执行：

```text
sftp xl6775@greene.hpc.nyu.edu
lcd C:/Users/20661/Documents/xwechat_files/wxid_i096w25uhusk22_e748/msg/file/2026-08
cd /scratch/xl6775/projects/EgoQA-two-user-annotated-pareto-dpo/data_RLHF/annotated_preference
put "rlhf_candidate_scores_merged_70_packets.csv" rlhf_candidate_scores_merged_70_packets.csv
ls -l rlhf_candidate_scores_merged_70_packets.csv
bye
```

不要用 SFTP 上传 `split_60_10.json` 或 `media_map.json`：下一节会在 Torch 上从冻结 CSV 确定性生成，避免把旧数据合同误用到新实验。

## 4. Torch：设置变量并检查上传结果

重新登录或回到 Torch shell 后，整段执行：

```bash
export PROJECT_ROOT=/scratch/xl6775/projects/EgoQA-two-user-annotated-pareto-dpo
export SOURCE_ROOT=/scratch/xl6775/projects/EgoQA-two-user-reviewer-v1
export TRAIN_ENV=/scratch/xl6775/envs/egoqa-ms-swift-v4.2.2-vllm024
export FFMPEG_ENV=/scratch/xl6775/envs/egoqa-ffmpeg-runtime
export MODEL_DIR=/scratch/xl6775/models/Qwen3-VL-8B-Instruct
export DATASET_ROOT=/scratch/xl6775/datasets/EgoLife
export DATA_DIR=${PROJECT_ROOT}/data_RLHF/annotated_preference
export OUTPUT_ROOT=${PROJECT_ROOT}/outputs/annotated_preference
export CSV_PATH=${DATA_DIR}/rlhf_candidate_scores_merged_70_packets.csv
export SPLIT_PATH=${DATA_DIR}/split_60_10.json
export MEDIA_MAP=${DATA_DIR}/media_map.json
export DPO_DATA_DIR=${DATA_DIR}/dpo
export PYTHON=${TRAIN_ENV}/bin/python
export PATH=${FFMPEG_ENV}/bin:${PATH}
export LD_LIBRARY_PATH=${FFMPEG_ENV}/lib:${LD_LIBRARY_PATH:-}
EXPECTED_SHA=32679019FD7C665A0632E9885405BDF13C77B51386EFC56E7B29B24192210CD7

mkdir -p "${PROJECT_ROOT}/logs" "${DATA_DIR}" "${OUTPUT_ROOT}"
cd "${PROJECT_ROOT}"

git status --short --branch
git status --short -- \
  training/grpo_v3/experiments/annotated_preference/TORCH_RUNBOOK_CN.md \
  hpc/grpo_v3/annotated_preference \
  data_RLHF/annotated_preference/rlhf_candidate_scores_merged_70_packets.csv

bash -n hpc/grpo_v3/annotated_preference/common.sh
bash -n hpc/grpo_v3/annotated_preference/gate0_data.sbatch
bash -n hpc/grpo_v3/annotated_preference/structure_probe.sbatch
bash -n hpc/grpo_v3/annotated_preference/smoke1.sbatch
bash -n hpc/grpo_v3/annotated_preference/overfit_probe.sbatch
bash -n hpc/grpo_v3/annotated_preference/train.sbatch
bash -n hpc/grpo_v3/annotated_preference/evaluate.sbatch
bash -n hpc/grpo_v3/human_preference_reviewer/v1/prepare_media.sbatch

if [ -f "${CSV_PATH}" ]; then
  ACTUAL_SHA=$(sha256sum "${CSV_PATH}" | awk '{print toupper($1)}')
  if [ "${ACTUAL_SHA}" = "${EXPECTED_SHA}" ]; then
    echo "CSV_SHA_PASSED ${ACTUAL_SHA}"
  else
    echo "STOP: CSV SHA mismatch actual=${ACTUAL_SHA}"
  fi
else
  echo "MISSING: ${CSV_PATH}"
fi

if [ -x "${PYTHON}" ] && [ -d "${MODEL_DIR}" ] && [ -x "${FFMPEG_ENV}/bin/ffmpeg" ]; then
  echo "RUNTIME_PATHS_PASSED"
  "${PYTHON}" -c 'from torchcodec.decoders import VideoDecoder; print("TORCHCODEC_PASSED", VideoDecoder.__module__)'
  "${FFMPEG_ENV}/bin/ffmpeg" -version | head -n 1
  df -h /scratch/xl6775
else
  echo "MISSING: python, model, or FFmpeg runtime"
fi
```

验收：分支正确；Git 无意外代码修改；八个 `bash -n` 均无输出且返回成功；CSV 出现 `CSV_SHA_PASSED`；运行时出现 `RUNTIME_PATHS_PASSED` 和 `TORCHCODEC_PASSED`。此时还不要求 `${DPO_DATA_DIR}` 存在。

## 5. Torch：生成并验证 `split_60_10.json`

该步骤在登录节点使用 CPU，直接复用当前分支的 Reviewer v1 审计实现：

```bash
if [ -f "${CSV_PATH}" ] && [ "$(sha256sum "${CSV_PATH}" | awk '{print toupper($1)}')" = "${EXPECTED_SHA}" ]; then
  "${PYTHON}" -m training.grpo_v3.experiments.human_preference_reviewer.v1.audit annotation-csv \
    --csv "${CSV_PATH}" \
    --output "${DATA_DIR}/annotation_audit_60_10.json" \
    --split-output "${SPLIT_PATH}" \
    --train-evidence-count 60 \
    --validation-evidence-count 10 \
    --locked-test-evidence-count 0 \
    --seed 42 \
    --require-formal-split
else
  echo "STOP: valid CSV is required before split generation"
fi

if [ -s "${DATA_DIR}/annotation_audit_60_10.json" ] && [ -s "${SPLIT_PATH}" ]; then
  "${PYTHON}" - "${DATA_DIR}/annotation_audit_60_10.json" "${SPLIT_PATH}" "${CSV_PATH}" <<'PY'
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

audit = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
split = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
with Path(sys.argv[3]).open("r", encoding="utf-8-sig", newline="") as handle:
    rows = list(csv.DictReader(handle))
media_by_evidence = defaultdict(set)
for row in rows:
    media_by_evidence[row["evidence_id"]].add(tuple(
        row[column].strip() for column in (
            "video_1_user", "video_1_source", "video_2_user", "video_2_source",
        )
    ))
ordered_media = [next(iter(values)) for values in media_by_evidence.values()]
assert audit["status"] == "passed", audit
assert audit["csv_sha256"].upper() == "32679019FD7C665A0632E9885405BDF13C77B51386EFC56E7B29B24192210CD7"
assert audit["row_count"] == 420
assert audit["evidence_count"] == 70
assert len(media_by_evidence) == 70
assert all(len(values) == 1 for values in media_by_evidence.values())
assert all(all(value for value in media) for media in ordered_media)
assert all(media[0] != media[2] for media in ordered_media)
assert all(media[1] != media[3] for media in ordered_media)
assert len({source for media in ordered_media for source in (media[1], media[3])}) == 140
assert len(split["train_evidence_ids"]) == 60
assert len(split["validation_evidence_ids"]) == 10
assert split["locked_test_evidence_ids"] == []
assert split["reserve_evidence_ids"] == []
assert not set(split["train_evidence_ids"]) & set(split["validation_evidence_ids"])
print("ORDERED_MEDIA_PASSED video_1=speaker video_2=provider names=participants sources=140")
print("ANNOTATION_GATE_PASSED rows=420 evidence=70 split=60/10/0")
PY
else
  echo "MISSING: annotation audit or split manifest"
fi
```

只有同时看到 `ORDERED_MEDIA_PASSED video_1=speaker video_2=provider names=participants sources=140` 和 `ANNOTATION_GATE_PASSED rows=420 evidence=70 split=60/10/0` 才准备媒体。

## 6. Torch：准备 140 个视频并生成 `media_map.json`

视频不经 SFTP；作业按 CSV 中的 140 个 URL 在 Torch 数据集目录中下载或复用文件，并生成新的映射：

```bash
MEDIA_SCRIPT=${PROJECT_ROOT}/hpc/grpo_v3/human_preference_reviewer/v1/prepare_media.sbatch
if [ -s "${SPLIT_PATH}" ] && [ -f "${MEDIA_SCRIPT}" ]; then
  MEDIA_JOB_RAW=$(sbatch --parsable \
    --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",DATA_DIR="${DATA_DIR}",CSV_PATH="${CSV_PATH}",DATASET_ROOT="${DATASET_ROOT}",MEDIA_MAP="${MEDIA_MAP}" \
    --chdir="${PROJECT_ROOT}" \
    --output="${PROJECT_ROOT}/logs/pareto-media-%j.out" \
    --error="${PROJECT_ROOT}/logs/pareto-media-%j.err" \
    "${MEDIA_SCRIPT}")
  MEDIA_JOB=${MEDIA_JOB_RAW%%;*}
  echo "MEDIA_JOB=${MEDIA_JOB}"
else
  echo "STOP: split or media preparation script missing"
fi
```

记录打印出的数字 `MEDIA_JOB`。等待结束后执行：

```bash
if [[ "${MEDIA_JOB:-}" =~ ^[0-9]+$ ]]; then
  squeue -j "${MEDIA_JOB}" -o "%.18i %.9T %.24j %.10M %.20R"
  sacct -j "${MEDIA_JOB}" --format=JobID,JobName%28,State,ExitCode,Elapsed,MaxRSS
  tail -n 100 "${PROJECT_ROOT}/logs/pareto-media-${MEDIA_JOB}.out"
  tail -n 100 "${PROJECT_ROOT}/logs/pareto-media-${MEDIA_JOB}.err"
else
  echo "STOP: MEDIA_JOB is not a numeric JobID"
fi

if [ -s "${MEDIA_MAP}" ]; then
  "${PYTHON}" - "${CSV_PATH}" "${MEDIA_MAP}" <<'PY'
import csv
import json
import sys
from pathlib import Path

with Path(sys.argv[1]).open("r", encoding="utf-8-sig", newline="") as handle:
    sources = {
        row[column].strip()
        for row in csv.DictReader(handle)
        for column in ("video_1_source", "video_2_source")
    }
mapping = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
assert len(sources) == 140, len(sources)
assert len(mapping) == 140, len(mapping)
assert set(mapping) == sources
assert all(Path(path).is_absolute() for path in mapping.values())
assert all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in mapping.values())
print("MEDIA_MAP_PASSED count=140")
PY
else
  echo "MISSING: ${MEDIA_MAP}"
fi
```

只有 Slurm 状态为 `COMPLETED` 且出现 `MEDIA_MAP_PASSED count=140` 才进入 Gate 0。

### 后续 GPU Gate 的通用检查函数

所有阶段只认本次数字 JobID 对应的目录：

| 阶段 | 唯一产物目录 |
|---|---|
| Gate 0 | `${OUTPUT_ROOT}/gate0_${JOBID}` |
| Gate 1 | `${OUTPUT_ROOT}/structure_${JOBID}` |
| Gate 2 | `${OUTPUT_ROOT}/smoke_${JOBID}` |
| Gate 3 | `${OUTPUT_ROOT}/overfit_${JOBID}` |
| Gate 4 | `${OUTPUT_ROOT}/train_${JOBID}` |
| Gate 5 | `${OUTPUT_ROOT}/validation_${JOBID}` |

在提交 Gate 1–5 的 shell 中先粘贴一次。它只读取指定 JobID 的日志和产物：

```bash
inspect_job() {
  local jobid=$1
  local mode=$2
  local log_stem=$3
  local outdir=${OUTPUT_ROOT}/${mode}_${jobid}
  if [[ "${jobid}" =~ ^[0-9]+$ ]]; then
    squeue -j "${jobid}" -o "%.18i %.9T %.28j %.10M %.20R"
    sacct -j "${jobid}" --units=G \
      --format=JobID,JobName%30,State,ExitCode,Elapsed,Timelimit,AllocTRES%50,MaxRSS,MaxVMSize
    tail -n 120 "${PROJECT_ROOT}/logs/${log_stem}-${jobid}.out" 2>/dev/null || true
    tail -n 120 "${PROJECT_ROOT}/logs/${log_stem}-${jobid}.err" 2>/dev/null || true
    find "${outdir}" -maxdepth 2 -type f -printf '%P\n' 2>/dev/null | sort
    if [ -s "${outdir}/dpo_gate_result.json" ]; then
      "${PYTHON}" - "${outdir}/dpo_gate_result.json" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(json.dumps(result, ensure_ascii=False, indent=2))
assert result["status"] == "passed", result
print("MODEL_GATE_PASSED")
PY
    elif [ "${mode}" = "structure" ] && [ -s "${outdir}/structure_probe.json" ]; then
      "${PYTHON}" -m json.tool "${outdir}/structure_probe.json"
      echo "STRUCTURE_ARTIFACT_PRESENT"
    else
      echo "MISSING: expected Gate artifact under ${outdir}"
    fi
  else
    echo "STOP: JobID must be numeric"
  fi
}
```

## 7. Gate 0：构建并审计 DPO 数据

```bash
SCRIPT=${PROJECT_ROOT}/hpc/grpo_v3/annotated_preference/gate0_data.sbatch
if [ -s "${CSV_PATH}" ] && [ -s "${SPLIT_PATH}" ] && [ -s "${MEDIA_MAP}" ] && [ -f "${SCRIPT}" ]; then
  GATE0_JOB_RAW=$(sbatch --parsable --export=ALL --chdir="${PROJECT_ROOT}" "${SCRIPT}")
  GATE0_JOB=${GATE0_JOB_RAW%%;*}
  echo "GATE0_JOB=${GATE0_JOB}"
else
  echo "STOP: Gate 0 requires CSV, split, media map, and script"
fi
```

```bash
if [[ "${GATE0_JOB:-}" =~ ^[0-9]+$ ]]; then
  JOBID=${GATE0_JOB}
  squeue -j "${JOBID}" -o "%.18i %.9T %.24j %.10M %.20R"
  sacct -j "${JOBID}" --format=JobID,JobName%28,State,ExitCode,Elapsed,MaxRSS
  tail -n 120 "${PROJECT_ROOT}/logs/pareto-dpo-gate0-${JOBID}.out"
  tail -n 120 "${PROJECT_ROOT}/logs/pareto-dpo-gate0-${JOBID}.err"
else
  echo "STOP: GATE0_JOB is not a numeric JobID"
fi

if [ -s "${DPO_DATA_DIR}/dataset_manifest.json" ] && \
   [ -s "${DPO_DATA_DIR}/train_dpo.jsonl" ] && \
   [ -s "${DPO_DATA_DIR}/validation_dpo.jsonl" ] && \
   [ -s "${DPO_DATA_DIR}/overfit_4_dpo.jsonl" ]; then
  "${PYTHON}" - "${DPO_DATA_DIR}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
for name in (
    "train_dpo.jsonl", "validation_dpo.jsonl", "train_pair_index.jsonl",
    "validation_pair_index.jsonl", "overfit_4_dpo.jsonl", "pareto_audit.json",
):
    path = root / name
    assert path.is_file() and path.stat().st_size > 0, name
assert manifest["counts"]["train_evidence_count"] == 60
assert manifest["counts"]["validation_evidence_count"] == 10
print("DPO_DATA_PASSED train=60 validation=10 locked_test=0")
PY
else
  echo "MISSING: stable DPO data under ${DPO_DATA_DIR}"
fi
```

Gate 0 输出只认 `${OUTPUT_ROOT}/gate0_${JOBID}` 和稳定数据目录 `${DPO_DATA_DIR}`。它证明 CSV、split、媒体和 Pareto 配对可构建；不证明模型可加载或训练可收敛。

## 8. Gate 1：模型结构探针

```bash
SCRIPT=${PROJECT_ROOT}/hpc/grpo_v3/annotated_preference/structure_probe.sbatch
if [ -s "${DPO_DATA_DIR}/train_dpo.jsonl" ] && [ -f "${SCRIPT}" ]; then
  STRUCTURE_JOB_RAW=$(sbatch --parsable --export=ALL --chdir="${PROJECT_ROOT}" "${SCRIPT}")
  STRUCTURE_JOB=${STRUCTURE_JOB_RAW%%;*}
  echo "STRUCTURE_JOB=${STRUCTURE_JOB}"
else
  echo "STOP: Gate 1 requires Gate 0 data and script"
fi
```

提交后执行：

```bash
inspect_job "${STRUCTURE_JOB}" structure pareto-dpo-structure
```

确认 `sacct` 为 `COMPLETED`、`${OUTPUT_ROOT}/structure_${STRUCTURE_JOB}/structure_probe.json` 存在且出现 `STRUCTURE_ARTIFACT_PRESENT`。验收结构 JSON、目标模块和可训练参数审计。该 Gate 不证明视频训练链路可运行。

## 9. Gate 2：单步 Smoke

```bash
SCRIPT=${PROJECT_ROOT}/hpc/grpo_v3/annotated_preference/smoke1.sbatch
if [[ "${STRUCTURE_JOB:-}" =~ ^[0-9]+$ ]] && [ -f "${SCRIPT}" ]; then
  SMOKE_JOB_RAW=$(sbatch --parsable --export=ALL --chdir="${PROJECT_ROOT}" "${SCRIPT}")
  SMOKE_JOB=${SMOKE_JOB_RAW%%;*}
  echo "SMOKE_JOB=${SMOKE_JOB}"
else
  echo "STOP: set the completed STRUCTURE_JOB before Gate 2"
fi
```

提交后执行：

```bash
inspect_job "${SMOKE_JOB}" smoke pareto-dpo-smoke1
```

确认 `sacct` 为 `COMPLETED`、`${OUTPUT_ROOT}/smoke_${SMOKE_JOB}/dpo_gate_result.json` 出现 `MODEL_GATE_PASSED`。验收一次前向、损失、反向、优化器更新、非零 LoRA delta 和可重载 adapter。它不证明 Pareto 偏好能泛化。

## 10. Gate 3：四样本过拟合探针

```bash
SCRIPT=${PROJECT_ROOT}/hpc/grpo_v3/annotated_preference/overfit_probe.sbatch
if [[ "${SMOKE_JOB:-}" =~ ^[0-9]+$ ]] && [ -f "${SCRIPT}" ]; then
  OVERFIT_JOB_RAW=$(sbatch --parsable --export=ALL --chdir="${PROJECT_ROOT}" "${SCRIPT}")
  OVERFIT_JOBID=${OVERFIT_JOB_RAW%%;*}
  echo "OVERFIT_JOBID=${OVERFIT_JOBID}"
else
  echo "STOP: set the completed SMOKE_JOB before Gate 3"
fi
```

提交后执行：

```bash
inspect_job "${OVERFIT_JOBID}" overfit pareto-dpo-overfit
```

确认 `sacct` 为 `COMPLETED`、`${OUTPUT_ROOT}/overfit_${OVERFIT_JOBID}/dpo_gate_result.json` 出现 `MODEL_GATE_PASSED`。验收固定小集 reward margin 改善、最终 pair accuracy 大于 0.8、参数变化和 adapter 重载。

## 11. Gate 4：60-evidence 正式训练

若这是新 SSH 会话，先输入真实数字 JobID：

```bash
if ! [[ "${OVERFIT_JOBID:-}" =~ ^[0-9]+$ ]]; then
  read -r -p "输入已通过 Gate 3 的数字 JobID: " OVERFIT_JOBID
fi
OVERFIT_RESULT=${OUTPUT_ROOT}/overfit_${OVERFIT_JOBID}/dpo_gate_result.json
SCRIPT=${PROJECT_ROOT}/hpc/grpo_v3/annotated_preference/train.sbatch
if [[ "${OVERFIT_JOBID:-}" =~ ^[0-9]+$ ]] && [ -s "${OVERFIT_RESULT}" ] && [ -f "${SCRIPT}" ]; then
  TRAIN_JOB_RAW=$(sbatch --parsable \
    --export=ALL,OVERFIT_RESULT="${OVERFIT_RESULT}" \
    --chdir="${PROJECT_ROOT}" "${SCRIPT}")
  TRAIN_JOBID=${TRAIN_JOB_RAW%%;*}
  echo "TRAIN_JOBID=${TRAIN_JOBID}"
else
  echo "STOP: missing numeric OVERFIT_JOBID, passed result, or train script"
fi
```

提交后执行：

```bash
inspect_job "${TRAIN_JOBID=}" train pareto-dpo-train
```

确认 `sacct` 为 `COMPLETED`、`${OUTPUT_ROOT}/train_${TRAIN_JOBID}/dpo_gate_result.json` 出现 `MODEL_GATE_PASSED`。验收曲线无 NaN/Inf、global step 非零、validation 指标存在、参数审计通过、adapter 可重载，并保留 Git 提交与输入 SHA。

## 12. Gate 5：冻结 Gate 4 验证产物审计

Gate 4 已使用 `train_dpo.jsonl` 更新参数，并在每个 epoch 结束时对 `validation_dpo.jsonl` 计算 DPO 指标。Gate 5 只验证并封装 Gate 4 的最终 validation 指标、adapter 和来源 SHA；它不再调用 `swift rlhf`，不执行 optimizer step，也不生成新 checkpoint。因此本 Gate 只申请 CPU，通常应在几分钟内结束。

先在 Torch 登录节点执行只读预检。这里继续使用已完成的 Gate 4 JobID `15595900`；若以后更换训练任务，只改这一行：

```bash
TRAIN_JOBID=15595900
ADAPTER_DIR=${OUTPUT_ROOT}/train_${TRAIN_JOBID}/adapter
TRAIN_DIR=$(dirname "${ADAPTER_DIR}")
SCRIPT=${PROJECT_ROOT}/hpc/grpo_v3/annotated_preference/evaluate.sbatch

if [[ "${TRAIN_JOBID}" =~ ^[0-9]+$ ]] \
  && [ -s "${ADAPTER_DIR}/adapter_config.json" ] \
  && [ -s "${TRAIN_DIR}/trainer_state.json" ] \
  && [ -s "${TRAIN_DIR}/dpo_gate_result.json" ] \
  && [ -s "${TRAIN_DIR}/resolved_command.txt" ] \
  && [ -f "${SCRIPT}" ]; then
  echo "GATE5_INPUTS_PASSED train_job=${TRAIN_JOBID}"
else
  echo "STOP: missing Gate 4 adapter, trainer state, result, command, or Gate 5 script"
fi

if bash -n "${SCRIPT}"; then
  echo "GATE5_BASH_SYNTAX_PASSED"
else
  echo "STOP: Gate 5 script has invalid bash syntax"
fi

if grep -n 'swift rlhf' "${SCRIPT}"; then
  echo "STOP: Gate 5 must not invoke swift rlhf"
else
  echo "GATE5_ZERO_TRAINING_CONTRACT_PASSED"
fi

"${PYTHON}" - "${TRAIN_DIR}/dpo_gate_result.json" "${TRAIN_DIR}/trainer_state.json" <<'PY'
import json
import sys
from pathlib import Path

gate = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
state = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
history = state.get("log_history", [])
eval_rows = [row for row in history if isinstance(row, dict) and "eval_loss" in row]
if gate.get("status") == "passed" and gate.get("mode") == "train" and eval_rows:
    row = eval_rows[-1]
    print("GATE4_VALIDATION_SOURCE_PASSED")
    print(json.dumps({
        "global_step": state.get("global_step"),
        "eval_loss": row.get("eval_loss"),
        "eval_pair_accuracy": row.get("eval_rewards/accuracies"),
        "eval_reward_margin": row.get("eval_rewards/margins"),
    }, indent=2, allow_nan=False))
else:
    print("STOP: Gate 4 did not preserve a passed epoch-end validation result")
PY
```

只有上述输出包含 `GATE5_INPUTS_PASSED`、`GATE5_BASH_SYNTAX_PASSED`、`GATE5_ZERO_TRAINING_CONTRACT_PASSED` 和 `GATE4_VALIDATION_SOURCE_PASSED` 时才提交：

```bash
if [[ "${TRAIN_JOBID}" =~ ^[0-9]+$ ]] \
  && [ -s "${ADAPTER_DIR}/adapter_config.json" ] \
  && [ -s "${TRAIN_DIR}/trainer_state.json" ] \
  && [ -s "${TRAIN_DIR}/dpo_gate_result.json" ] \
  && [ -s "${TRAIN_DIR}/resolved_command.txt" ] \
  && [ -f "${SCRIPT}" ] \
  && ! grep -q 'swift rlhf' "${SCRIPT}"; then
  VALIDATION_JOB_RAW=$(sbatch --parsable \
    --export=ALL,TRAIN_JOB_ID=${TRAIN_JOBID},ADAPTER_DIR=${ADAPTER_DIR} \
    --chdir="${PROJECT_ROOT}" "${SCRIPT}")
  VALIDATION_JOB=${VALIDATION_JOB_RAW%%;*}
  echo "VALIDATION_JOB=${VALIDATION_JOB}"
else
  echo "STOP: Gate 5 preflight failed; job was not submitted"
fi
```

提交后执行：

```bash
if [[ "${VALIDATION_JOB:-}" =~ ^[0-9]+$ ]]; then
  inspect_job "${VALIDATION_JOB}" validation pareto-dpo-validation
  RESULT=${OUTPUT_ROOT}/validation_${VALIDATION_JOB}/dpo_gate_result.json
  STATE=${OUTPUT_ROOT}/validation_${VALIDATION_JOB}/trainer_state.json
  if [ -s "${RESULT}" ] && [ -s "${STATE}" ]; then
    "${PYTHON}" - "${RESULT}" "${STATE}" "${TRAIN_JOBID}" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
state = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
provenance = state.get("validation_provenance", {})
expected_train_job_id = sys.argv[3]
summary = {
    "status": result.get("status"),
    "eval_pair_count": result.get("eval_pair_count"),
    "manifest_validation_pair_count": result.get("manifest_validation_pair_count"),
    "eval_loss": result.get("final_eval_loss"),
    "eval_pair_accuracy": result.get("final_eval_pair_accuracy"),
    "eval_reward_margin": result.get("final_eval_reward_margin"),
    "evaluation_origin": provenance.get("evaluation_origin"),
    "gate5_optimizer_steps": provenance.get("gate5_optimizer_steps"),
    "source_train_job_id": provenance.get("source_train_job_id"),
    "source_trainer_state_sha256": provenance.get("source_trainer_state_sha256"),
}
print(json.dumps(summary, indent=2, allow_nan=False))
if (
    summary["status"] == "passed"
    and summary["eval_pair_count"] == summary["manifest_validation_pair_count"]
    and summary["evaluation_origin"] == "gate4_epoch_end"
    and summary["gate5_optimizer_steps"] == 0
    and summary["source_train_job_id"] == expected_train_job_id
):
    print("GATE5_FROZEN_VALIDATION_PASSED")
else:
    print("STOP: Gate 5 frozen-validation contract failed")
PY
  else
    echo "MISSING: Gate 5 result or trainer state"
  fi
else
  echo "STOP: VALIDATION_JOB must be numeric"
fi
```

最终验收必须同时满足：Slurm `COMPLETED 0:0`、`status=passed`、`eval_pair_count` 等于 manifest、`evaluation_origin=gate4_epoch_end`、`gate5_optimizer_steps=0`、`source_train_job_id=15595900`。Gate5 不读取零行 locked test，也不把 validation 数据送入任何训练入口。

## 13. 通用 JobID 监控与失败证据收集

对任意阶段，在同一 shell 设置该阶段的真实数字 JobID，或直接运行提示输入：

```bash
read -r -p "输入要检查的数字 JobID: " JOBID
if [[ "${JOBID}" =~ ^[0-9]+$ ]]; then
  squeue -j "${JOBID}" -o "%.18i %.9T %.28j %.10M %.20R"
  sacct -j "${JOBID}" --units=G \
    --format=JobID,JobName%30,State,ExitCode,Elapsed,Timelimit,AllocTRES%50,MaxRSS,MaxVMSize
  scontrol show job -dd "${JOBID}" | grep -E 'JobState=|Reason=|ExitCode=|RunTime=|TimeLimit=|WorkDir=|StdOut=|StdErr=|Command='

  JOB_OUT="${OUTPUT_ROOT}/jobs/${JOBID}"
  mkdir -p "${JOB_OUT}"
  cp "${PROJECT_ROOT}/logs/"*"${JOBID}"*.out "${JOB_OUT}/" 2>/dev/null || true
  cp "${PROJECT_ROOT}/logs/"*"${JOBID}"*.err "${JOB_OUT}/" 2>/dev/null || true
  find "${OUTPUT_ROOT}" -maxdepth 2 -type f -path "*_${JOBID}/*" \
    \( -name '*.json' -o -name '*.txt' -o -name '*.csv' \) \
    -print > "${JOB_OUT}/artifact_paths.txt"
  tar -czf "${OUTPUT_ROOT}/job_${JOBID}_diagnostics.tar.gz" \
    -C "${OUTPUT_ROOT}/jobs" "${JOBID}"
  echo "DIAGNOSTIC_BUNDLE=${OUTPUT_ROOT}/job_${JOBID}_diagnostics.tar.gz"
else
  echo "STOP: JOBID must be numeric"
fi
```

需要下载诊断包时，在 Windows PowerShell 输入真实数字：

```powershell
$JobId = Read-Host '输入数字 JobID'
$Remote = "xl6775@greene.hpc.nyu.edu:/scratch/xl6775/projects/EgoQA-two-user-annotated-pareto-dpo/outputs/annotated_preference/job_${JobId}_diagnostics.tar.gz"
$Local = Join-Path $env:USERPROFILE 'Downloads'
scp $Remote $Local
```

不要默认打包模型、完整 checkpoint、视频、Hugging Face cache 或全部 outputs。

## 14. Gate 6：明确未执行

Gate 6（生产 `expanded schema` 的端到端自由生成与人工终点评估）未执行，因此自由生成未验证。Gate 0–5 最多证明冻结标注数据上的 DPO 数据链、训练工程和固定 validation 评估；不能表述为生产替换成功、locked-test 成功或真实 QA 质量结论。

## 15. 每次汇报的固定格式

```text
阶段：
branch / commit：
Job ID：
Slurm State / ExitCode：
输出目录：
CSV SHA：
split / media / DPO 数据状态：
第一个失败 Gate：
训练工程证据：
验证证据：
本次能证明：
本次不能证明：
下一步唯一动作：
```

## 16. 临时低利用率约束：一次提交 9 个短任务的分段超参 Sweep

本节用于集群会审计运行超过约 2 小时且 GPU 利用率较低任务的情况。不要启动额外 CUDA 空转负载。训练拆为三条自动依赖链；每段只推进一个 epoch，完整恢复模型、optimizer、scheduler、随机状态和 global step：

```text
LR=3e-5：epoch 1（step 0→66）→ epoch 2（66→132）→ epoch 3（132→198）
LR=6e-5：epoch 1（step 0→66）→ epoch 2（66→132）→ epoch 3（132→198）
LR=1e-4：epoch 1（step 0→66）→ epoch 2（66→132）→ epoch 3（132→198）
```

每段申请 `01:50:00`。launcher 一次提交全部 9 个任务，用 `afterok` 自动衔接；不要求输入、回忆或记录 JobID。JobID 只用于 Slurm provenance 和唯一产物目录。

### 16.1 Windows：提交本节代码改动后推送

在 Windows PowerShell 执行：

```powershell
$Repo = 'C:\Users\20661\Desktop\Research\AR\multiuser\EgoQA-two-user-reviewer-v1'
$Branch = 'feature/annotated-pareto-dpo'
Set-Location -LiteralPath $Repo

git status --short --branch
git diff --check
git diff -- `
  hpc/grpo_v3/annotated_preference/staged_train.sbatch `
  hpc/grpo_v3/annotated_preference/submit_staged_sweep.sh `
  training/grpo_v3/experiments/annotated_preference/TORCH_RUNBOOK_CN.md

git add -f -- `
  hpc/grpo_v3/annotated_preference/staged_train.sbatch `
  hpc/grpo_v3/annotated_preference/submit_staged_sweep.sh

git add -- `
  training/grpo_v3/experiments/annotated_preference/TORCH_RUNBOOK_CN.md `
  tests/training/grpo_v3/experiments/annotated_preference/test_slurm.py `
  tests/training/grpo_v3/experiments/annotated_preference/test_runbook.py

git diff --cached --check
git diff --cached --stat
git commit -m "feat(training): add staged Pareto DPO sweep"
git push origin $Branch

$LocalHead = git rev-parse HEAD
$RemoteHead = (
  git ls-remote origin "refs/heads/$Branch" |
    ForEach-Object { ($_ -split '\s+')[0] }
)

if ($LocalHead -eq $RemoteHead) {
  Write-Host "STAGED_SWEEP_GIT_PUSH_PASSED head=$LocalHead"
} else {
  Write-Host "STOP: local=$LocalHead remote=$RemoteHead"
}
```

由于仓库 `.gitignore` 忽略整个 `hpc/`，这里必须对两个明确的新脚本使用 `git add -f`；不要对整个 `hpc/` 目录强制添加。只有看到 `STAGED_SWEEP_GIT_PUSH_PASSED` 才继续。本节脚本通过 Git 同步，不需要 SFTP 上传 `cuda.py` 或其他运行脚本。

### 16.2 Torch：同步代码并做只读预检

在 Torch 登录节点整段执行：

```bash
export PROJECT_ROOT=/scratch/xl6775/projects/EgoQA-two-user-annotated-pareto-dpo
export TRAIN_ENV=/scratch/xl6775/envs/egoqa-ms-swift-v4.2.2-vllm024
export MODEL_DIR=/scratch/xl6775/models/Qwen3-VL-8B-Instruct
export DATA_DIR=${PROJECT_ROOT}/data_RLHF/annotated_preference
export OUTPUT_ROOT=${PROJECT_ROOT}/outputs/annotated_preference
export DPO_DATA_DIR=${DATA_DIR}/dpo
export PYTHON=${TRAIN_ENV}/bin/python
BRANCH=feature/annotated-pareto-dpo

if [ -e "${PROJECT_ROOT}/.git" ]; then
  git -C "${PROJECT_ROOT}" fetch origin "${BRANCH}"
  if git -C "${PROJECT_ROOT}" diff --quiet \
    && git -C "${PROJECT_ROOT}" diff --cached --quiet; then
    git -C "${PROJECT_ROOT}" merge --ff-only "origin/${BRANCH}"
  else
    echo "STOP: worktree has tracked or staged changes; Git fast-forward skipped"
    git -C "${PROJECT_ROOT}" status --short --untracked-files=no
  fi
else
  echo "STOP: missing Git worktree at ${PROJECT_ROOT}"
fi

STAGE_SCRIPT=${PROJECT_ROOT}/hpc/grpo_v3/annotated_preference/staged_train.sbatch
SUBMIT_SCRIPT=${PROJECT_ROOT}/hpc/grpo_v3/annotated_preference/submit_staged_sweep.sh
READY=1

if ! bash -n "${STAGE_SCRIPT}"; then
  echo "STOP: staged_train.sbatch syntax failed"
  READY=0
fi

if ! bash -n "${SUBMIT_SCRIPT}"; then
  echo "STOP: submit_staged_sweep.sh syntax failed"
  READY=0
fi

if [ ! -x "${PYTHON}" ] || [ ! -d "${MODEL_DIR}" ]; then
  echo "MISSING: Python environment or model"
  READY=0
fi

for FILE in \
  "${DPO_DATA_DIR}/train_dpo.jsonl" \
  "${DPO_DATA_DIR}/validation_dpo.jsonl" \
  "${DPO_DATA_DIR}/dataset_manifest.json"; do
  if [ ! -s "${FILE}" ]; then
    echo "MISSING: ${FILE}"
    READY=0
  fi
done

OVERFIT_RESULT=$("${PYTHON}" - "${OUTPUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

candidates = []
for path in Path(sys.argv[1]).glob("overfit_*/dpo_gate_result.json"):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    if value.get("status") == "passed" and value.get("mode") == "overfit":
        candidates.append((path.stat().st_mtime_ns, path))
if candidates:
    print(max(candidates)[1])
PY
)

if [ -z "${OVERFIT_RESULT}" ] || [ ! -s "${OVERFIT_RESULT}" ]; then
  echo "STOP: no passed overfit Gate was discovered"
  READY=0
else
  echo "AUTO_DISCOVERED_OVERFIT_RESULT=${OVERFIT_RESULT}"
fi

if [ "${READY}" -eq 1 ]; then
  echo "STAGED_SWEEP_PREFLIGHT_PASSED"
else
  echo "STOP: staged sweep preflight failed"
fi
```

只有看到 `STAGED_SWEEP_PREFLIGHT_PASSED` 才提交。

### 16.3 Torch：一次提交全部 9 个任务

沿用上一节环境变量，执行：

```bash
if [ "${READY:-0}" -eq 1 ] && [ -f "${SUBMIT_SCRIPT}" ]; then
  bash "${SUBMIT_SCRIPT}"
else
  echo "STOP: preflight did not pass; no staged sweep was submitted"
fi
```

验收输出：

```text
STAGED_SWEEP_SUBMISSION_PASSED count=9
ACTIVE_MANIFEST=.../active_staged_sweep_manifest.txt
```

持久化文件为：

```text
${OUTPUT_ROOT}/active_staged_sweep_manifest.txt
${OUTPUT_ROOT}/staged_sweep_<时间戳>/jobs.tsv
${OUTPUT_ROOT}/staged_sweep_<时间戳>/sweep_manifest.tsv
```

### 16.4 Torch：无需 JobID 的统一监控

重新登录后也可直接执行：

```bash
export PROJECT_ROOT=/scratch/xl6775/projects/EgoQA-two-user-annotated-pareto-dpo
export OUTPUT_ROOT=${PROJECT_ROOT}/outputs/annotated_preference
ACTIVE_POINTER=${OUTPUT_ROOT}/active_staged_sweep_manifest.txt

if [ -s "${ACTIVE_POINTER}" ]; then
  JOBS=$(head -n 1 "${ACTIVE_POINTER}")
else
  JOBS=
fi

if [ -n "${JOBS}" ] && [ -s "${JOBS}" ]; then
  JOB_IDS=$(awk -F '\t' 'NR > 1 {print $5}' "${JOBS}" | paste -sd, -)
  echo "ACTIVE_STAGED_SWEEP=${JOBS}"
  column -t -s $'\t' "${JOBS}"
  squeue -j "${JOB_IDS}" -o "%.18i %.30j %.10T %.10M %.10l %.24R"
  sacct -j "${JOB_IDS}" --units=G \
    --format=JobIDRaw,JobName%30,State%28,ExitCode,Elapsed,Timelimit,MaxRSS

  while IFS=$'\t' read -r LR EPOCH INITIAL TARGET JOB_ID DEP_JOB RESUME OUTDIR SUBMITTED; do
    if [ "${JOB_ID}" != "job_id" ]; then
      LOG=${PROJECT_ROOT}/logs/pareto-dpo-stage-${JOB_ID}.out
      echo
      echo "========== lr=${LR} target_epoch=${EPOCH} =========="
      if [ -s "${LOG}" ]; then
        grep -F "'global_step/max_steps':" "${LOG}" | tail -n 1 || true
        grep -F "'eval_loss':" "${LOG}" | tail -n 1 || true
      else
        echo "WAITING: log has not been created"
      fi
      if [ -s "${OUTDIR}/stage_contract.json" ]; then
        cat "${OUTDIR}/stage_contract.json"
      fi
    fi
  done < "${JOBS}"
else
  echo "MISSING: active staged sweep manifest"
fi
```

`PENDING (Dependency)` 是正常状态。若上游失败，下游可能显示 `DependencyNeverSatisfied`；此时不要手工强制启动下游，因为它没有完整 checkpoint。

### 16.5 Torch：完成后自动汇总 9 行 validation 结果

```bash
export PROJECT_ROOT=/scratch/xl6775/projects/EgoQA-two-user-annotated-pareto-dpo
export TRAIN_ENV=/scratch/xl6775/envs/egoqa-ms-swift-v4.2.2-vllm024
export OUTPUT_ROOT=${PROJECT_ROOT}/outputs/annotated_preference
export PYTHON=${TRAIN_ENV}/bin/python
ACTIVE_POINTER=${OUTPUT_ROOT}/active_staged_sweep_manifest.txt

if [ -s "${ACTIVE_POINTER}" ]; then
  JOBS=$(head -n 1 "${ACTIVE_POINTER}")
else
  JOBS=
fi

if [ -n "${JOBS}" ] && [ -s "${JOBS}" ]; then
  SWEEP_DIR=$(dirname "${JOBS}")
  RESULTS=${SWEEP_DIR}/staged_sweep_results.csv
  BEST=${SWEEP_DIR}/best_config.env

  "${PYTHON}" - "${JOBS}" "${RESULTS}" "${BEST}" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

jobs_path, results_path, best_path = map(Path, sys.argv[1:])
jobs = list(csv.DictReader(
    jobs_path.open("r", encoding="utf-8", newline=""), delimiter="\t"
))
rows = []
missing = []

for job in jobs:
    output_dir = Path(job["output_dir"])
    state_path = output_dir / "trainer_state.json"
    gate_path = output_dir / "dpo_gate_result.json"
    if not state_path.is_file() or not gate_path.is_file():
        missing.append(str(output_dir))
        continue
    state = json.loads(state_path.read_text(encoding="utf-8"))
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    eval_rows = [
        item for item in state.get("log_history", [])
        if isinstance(item, dict) and "eval_loss" in item
    ]
    if not eval_rows:
        missing.append(str(output_dir))
        continue
    item = eval_rows[-1]
    rows.append({
        "learning_rate": job["learning_rate"],
        "epoch": int(job["target_epoch"]),
        "job_id": job["job_id"],
        "global_step": state.get("global_step"),
        "gate_status": gate.get("status"),
        "eval_loss": item.get("eval_loss"),
        "eval_pair_accuracy": item.get("eval_rewards/accuracies"),
        "eval_reward_margin": item.get("eval_rewards/margins"),
        "output_dir": str(output_dir),
    })

fields = [
    "learning_rate", "epoch", "job_id", "global_step", "gate_status",
    "eval_loss", "eval_pair_accuracy", "eval_reward_margin", "output_dir",
]
with results_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

valid = [
    row for row in rows
    if row["gate_status"] == "passed"
    and finite(row["eval_loss"])
    and finite(row["eval_pair_accuracy"])
    and finite(row["eval_reward_margin"])
]
ranked = sorted(valid, key=lambda row: (
    row["eval_loss"], -row["eval_pair_accuracy"],
    -row["eval_reward_margin"], row["epoch"],
))

for row in ranked:
    print(
        f'lr={row["learning_rate"]:>5} epoch={row["epoch"]} '
        f'loss={row["eval_loss"]:.9f} '
        f'acc={row["eval_pair_accuracy"]:.6f} '
        f'margin={row["eval_reward_margin"]:.9f}'
    )

if len(valid) == 9 and not missing:
    best = ranked[0]
    best_path.write_text(
        f'BEST_LR={best["learning_rate"]}\n'
        f'BEST_EPOCH={best["epoch"]}\n'
        f'BEST_SOURCE_JOB={best["job_id"]}\n',
        encoding="utf-8",
    )
    print("STAGED_SWEEP_RESULT_COUNT_PASSED count=9")
    print(f"RESULTS={results_path}")
    print(f"BEST_CONFIG={best_path}")
else:
    print(f"WAITING: valid_rows={len(valid)} missing={len(missing)}")
PY
else
  echo "MISSING: active staged sweep manifest"
fi
```

只有出现 `STAGED_SWEEP_RESULT_COUNT_PASSED count=9` 才进行最终配置选择。排序规则为：先最小化 `eval_loss`，再最大化 pair accuracy 和 reward margin，最后倾向更少 epoch。

### 16.6 Windows：无需 JobID 下载 sweep 表

先在 Torch 汇总成功并记下 `staged_sweep_<时间戳>` 目录名，然后在 Windows PowerShell 执行：

```powershell
$TorchHost = 'greene.hpc.nyu.edu'
$RemoteRoot = '/scratch/xl6775/projects/EgoQA-two-user-annotated-pareto-dpo/outputs/annotated_preference'
$SweepName = Read-Host '输入 staged_sweep_ 开头的目录名'
$LocalDir = Join-Path $env:USERPROFILE "Downloads\$SweepName"
New-Item -ItemType Directory -Force -Path $LocalDir | Out-Null

scp "xl6775@${TorchHost}:${RemoteRoot}/${SweepName}/jobs.tsv" $LocalDir
scp "xl6775@${TorchHost}:${RemoteRoot}/${SweepName}/staged_sweep_results.csv" $LocalDir
scp "xl6775@${TorchHost}:${RemoteRoot}/${SweepName}/best_config.env" $LocalDir
Get-ChildItem -LiteralPath $LocalDir
```

该下载只包含小型 manifest 和指标表，不下载 adapter、optimizer、视频或整个 outputs。

### 16.7 三条 epoch 2 失败后的可观测性恢复

2026-08-13 的三个 epoch 2 Job `15675191`、`15675194`、`15675197` 均在 Swift 日志到达目标 step 后以 `FAILED 1:0` 结束，且原 runner 未持久化 Swift 返回码或临时 checkpoint 清单。修复后的 `staged_train.sbatch` 必须在任何阶段验收前写出：

```text
${OUTPUT_ROOT}/staged_<JobID>/swift_return_code.txt
${OUTPUT_ROOT}/staged_<JobID>/checkpoint_inventory.txt
${OUTPUT_ROOT}/staged_<JobID>/checkpoint.partial
```

只有 Swift 返回 0、四个恢复文件齐全且 `trainer_state.global_step` 等于目标 step 时，`checkpoint.partial` 才会提升为正式 `checkpoint`。否则 Job 仍失败，但诊断证据留在持久输出目录。

先在 Windows 提交并推送修复：

```powershell
$Repo = 'C:\Users\20661\Desktop\Research\AR\multiuser\EgoQA-two-user-reviewer-v1'
$Branch = 'feature/annotated-pareto-dpo'
Set-Location -LiteralPath $Repo

git add -f -- `
  hpc/grpo_v3/annotated_preference/staged_train.sbatch `
  hpc/grpo_v3/annotated_preference/submit_staged_recovery.sh
git add -- `
  training/grpo_v3/experiments/annotated_preference/TORCH_RUNBOOK_CN.md `
  tests/training/grpo_v3/experiments/annotated_preference/test_slurm.py `
  tests/training/grpo_v3/experiments/annotated_preference/test_runbook.py `
  docs/superpowers/plans/2026-08-14-staged-sweep-observability-recovery.md

git diff --cached --check
python -m unittest `
  tests.training.grpo_v3.experiments.annotated_preference.test_slurm `
  tests.training.grpo_v3.experiments.annotated_preference.test_runbook
git commit -m "fix(training): preserve staged checkpoint diagnostics"
git push origin $Branch
```

然后在 Torch 登录节点整段执行。该块先快进代码并验证三个 epoch 1 完整 checkpoint；只有全部通过才一次提交六个恢复任务，不需要输入或记录 JobID：

```bash
export PROJECT_ROOT=/scratch/xl6775/projects/EgoQA-two-user-annotated-pareto-dpo
export OUTPUT_ROOT=${PROJECT_ROOT}/outputs/annotated_preference
BRANCH=feature/annotated-pareto-dpo
READY=1

if [ -e "${PROJECT_ROOT}/.git" ]; then
  git -C "${PROJECT_ROOT}" fetch origin "${BRANCH}"
  if git -C "${PROJECT_ROOT}" diff --quiet \
    && git -C "${PROJECT_ROOT}" diff --cached --quiet; then
    git -C "${PROJECT_ROOT}" merge --ff-only "origin/${BRANCH}"
  else
    echo "STOP: Torch worktree has tracked or staged changes"
    git -C "${PROJECT_ROOT}" status --short --untracked-files=no
    READY=0
  fi
else
  echo "MISSING: ${PROJECT_ROOT}/.git"
  READY=0
fi

STAGE_SCRIPT=${PROJECT_ROOT}/hpc/grpo_v3/annotated_preference/staged_train.sbatch
RECOVERY_SCRIPT=${PROJECT_ROOT}/hpc/grpo_v3/annotated_preference/submit_staged_recovery.sh

if ! bash -n "${STAGE_SCRIPT}"; then
  echo "STOP: staged_train.sbatch syntax failed"
  READY=0
fi
if ! bash -n "${RECOVERY_SCRIPT}"; then
  echo "STOP: submit_staged_recovery.sh syntax failed"
  READY=0
fi

if [ "${READY}" -eq 1 ]; then
  bash "${RECOVERY_SCRIPT}"
else
  echo "STOP: recovery was not submitted"
fi
```

成功提交标记固定为：

```text
RECOVERY_SUBMISSION_PASSED count=6
ACTIVE_RECOVERY_MANIFEST=.../active_staged_recovery_manifest.txt
```

重新登录后无需 JobID 监控：

```bash
export PROJECT_ROOT=/scratch/xl6775/projects/EgoQA-two-user-annotated-pareto-dpo
export OUTPUT_ROOT=${PROJECT_ROOT}/outputs/annotated_preference
ACTIVE_POINTER=${OUTPUT_ROOT}/active_staged_recovery_manifest.txt

if [ -s "${ACTIVE_POINTER}" ]; then
  JOBS=$(head -n 1 "${ACTIVE_POINTER}")
else
  JOBS=
fi

if [ -n "${JOBS}" ] && [ -s "${JOBS}" ]; then
  JOB_IDS=$(awk -F '\t' 'NR > 1 {print $5}' "${JOBS}" | paste -sd, -)
  column -t -s $'\t' "${JOBS}"
  squeue -j "${JOB_IDS}" -o "%.18i %.32j %.12T %.10M %.10l %.28R"
  sacct -j "${JOB_IDS}" --units=G \
    --format=JobIDRaw,JobName%32,State%28,ExitCode,Elapsed,Timelimit,MaxRSS
else
  echo "MISSING: active staged recovery manifest"
fi
```

若任一恢复任务失败，优先读取其 `swift_return_code.txt`、`checkpoint_inventory.txt` 和 `checkpoint.partial`，不要再次提交完整 9-job sweep。
