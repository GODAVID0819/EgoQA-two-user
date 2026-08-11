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
  if [ -z "$(git -C "${PROJECT_ROOT}" status --porcelain)" ]; then
    git -C "${PROJECT_ROOT}" merge --ff-only "origin/${BRANCH}"
  else
    echo "STOP: target worktree has local changes; fast-forward skipped"
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
