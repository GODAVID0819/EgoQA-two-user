# Reviewer v1 Torch 运行手册

本手册服从 `docs/TORCH_EXPERIMENT_META_RULES_CN.md`。Markdown 只供人工阅读，不是远端作业依赖。

## 1. 范围与硬 Gate

```text
CPU Prepare → CPU Structure Probe → 60-train/10-validation（唯一 H100 作业）→ External Locked Test（后续）
```

当前 `binary_reviewer_v2` 只训练 Qwen3-VL-8B 最后两个 shared language blocks 的 q/v LoRA 与三个独立 binary heads。每个 head 输出 `[fail, pass]` 两个 logits；旧人类分数按 `1 → fail`、`2/3 → pass` 映射。每个 head 使用只从 training split 计算的 balanced class-weighted cross-entropy，三个 head 等权平均。数据来源仍是 `evidence_grounding_score`、`answerability_score`、`formality_score`；`aggregate_rank` 和 `fea_total_score` 被数据层忽略。不运行 overall preference、pairwise/tie loss、`lambda_rank`、GRPO 或 no-video ablation。

当前 CSV SHA-256：

```text
32679019FD7C665A0632E9885405BDF13C77B51386EFC56E7B29B24192210CD7
```

当前严格可用数据为 70 completed evidence（420 candidates）：60 packets 用于训练、10 packets 用于 validation；test/reserve 均为空。合并 CSV 已移除 ranking 派生列。现在可以提交 Train，但在后续独立标注 5-10 个 packets 前不能运行 Locked Test，也不能声称最终 unseen-evidence 泛化。

## 2. 每次 SSH 初始化

先从 Windows PowerShell 登录，并按终端提示完成 Microsoft device login 与 NYU MFA：

```powershell
ssh hm2991@login.torch.hpc.nyu.edu
```

进入 Torch 登录节点后再设置运行变量：

```bash
NETID=hm2991
TORCH_ACCOUNT=torch_pr_111_tandon_advanced
cd /scratch/hm2991/Long-video-understanding-clip/egolife_two_user_qa/score-only-reward-model
REVIEWER_ROOT=$(pwd -P)
export REVIEWER_ROOT
HPC_ROOT=${REVIEWER_ROOT}
CODE_ROOT=${REVIEWER_ROOT}
OUTPUT_ROOT=${CODE_ROOT}/outputs/human_preference_reviewer/v1
DATA_DIR=${CODE_ROOT}/data_RLHF/reviewer_v1
TRAIN_ENV=/scratch/hm2991/conda/envs/qwen3vl-smoke
FFMPEG_ENV=${TRAIN_ENV}
MODEL_DIR=/scratch/hm2991/models/Qwen3-VL-8B-Instruct
PACKAGE_ROOT=$(cd "${REVIEWER_ROOT}/.." && pwd -P)
REWARD_OUTPUT_ROOT=${PACKAGE_ROOT}/outputs/reward_candidate_collection_qwen36_27b
IMPORTED_MEDIA_RUN=${REWARD_OUTPUT_ROOT}/imported_day1_annotations_20
CSV_PATH=${DATA_DIR}/rlhf_candidate_scores_merged_70_packets.csv
MEDIA_MAP=${DATA_DIR}/media_map.json
PYTHON=${TRAIN_ENV}/bin/python
export PYTHONPATH="${CODE_ROOT}:${HPC_ROOT}:${PYTHONPATH:-}"
mkdir -p "${HPC_ROOT}/hpc/logs" "${OUTPUT_ROOT}" "${DATA_DIR}" "${IMPORTED_MEDIA_RUN}"
cd "${CODE_ROOT}"
```

供直接粘贴的 SSH 命令不启用全局 strict mode，也不执行 `exit`。失败后保留会话收集证据。

## 3. 隔离目录与 CSV

SSH 登录节点先建目录：

```bash
test -d "${REVIEWER_ROOT}/training"
test -d "${REVIEWER_ROOT}/hpc"
mkdir -p "${REVIEWER_ROOT}/data_RLHF/reviewer_v1"
mkdir -p "${REVIEWER_ROOT}/hpc/logs"
```

本轮直接从完整的隔离副本运行，不再把 `training/` 和 `hpc/` 拆开复制到基线仓库：

```text
local score-only-reward-model/... -> /scratch/hm2991/Long-video-understanding-clip/egolife_two_user_qa/score-only-reward-model/...
```

Windows PowerShell：

```text
sftp hm2991@dtn.torch.hpc.nyu.edu
lcd "C:/Users/haoya/OneDrive/Desktop/Long context video understanding ARVR/Long-video-understanding/egolife_two_user_qa/score-only-reward-model"
cd /scratch/hm2991/Long-video-understanding-clip/egolife_two_user_qa/score-only-reward-model
put "data_RLHF/reviewer_v1/rlhf_candidate_scores_merged_70_packets.csv" data_RLHF/reviewer_v1/rlhf_candidate_scores_merged_70_packets.csv
bye
```

若完整目录已经复制，只需确认最新 launcher 已覆盖并把单个 CSV 放到上述位置。不要从 Windows 复制模型或视频。CPU Prepare 会复用 candidate-collection 中已有的 100 个视频源，并只把缺失的 40 个 Day-1 视频源按原 candidate-collection 目录结构写入 `${IMPORTED_MEDIA_RUN}`；再次运行会复用这些文件。

## 4. Gate 0：零 GPU 预检

### 4.1 接收、语法、依赖

```bash
cd "${CODE_ROOT}"
test -s training/grpo_v3/experiments/human_preference_reviewer/v1/train.py
test -s hpc/grpo_v3/human_preference_reviewer/v1/common.sh
bash -n "${REVIEWER_ROOT}/hpc/grpo_v3/human_preference_reviewer/v1/prepare.sbatch"
bash -n "${REVIEWER_ROOT}/hpc/grpo_v3/human_preference_reviewer/v1/submit_first_run.sh"
bash -n "${HPC_ROOT}/hpc/grpo_v3/human_preference_reviewer/v1/common.sh"
bash -n "${HPC_ROOT}/hpc/grpo_v3/human_preference_reviewer/v1/structure_probe.sbatch"
bash -n "${HPC_ROOT}/hpc/grpo_v3/human_preference_reviewer/v1/train.sbatch"
bash -n "${HPC_ROOT}/hpc/grpo_v3/human_preference_reviewer/v1/evaluate.sbatch"

export PATH="${FFMPEG_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${FFMPEG_ENV}/lib:${LD_LIBRARY_PATH:-}"
"${PYTHON}" -c 'import torch,transformers,peft,accelerate; print(torch.__version__,transformers.__version__,peft.__version__,accelerate.__version__)'
"${PYTHON}" -c 'from torchcodec.decoders import VideoDecoder; print(VideoDecoder.__module__)'
"${FFMPEG_ENV}/bin/ffmpeg" -version | head -n 2
test -s "${MODEL_DIR}/config.json" && echo MODEL_CONFIG_OK
command -v hf
```

登录节点没有分配 GPU，因此下面输出 `cuda_visible_on_login=False` 是正常的；真正的 CUDA Gate 在 H100 training job 内执行。`cuda.py` 位于 `Long-video-understanding-clip/hpc/`，training launcher 会在训练前启动并在作业退出时清理它：

```bash
CUDA_KEEPER=/scratch/hm2991/Long-video-understanding-clip/hpc/cuda.py
test -s "${CUDA_KEEPER}" && echo CUDA_KEEPER_SCRIPT_OK
"${PYTHON}" -c 'import pynvml,torch; print("pynvml=ok", "cuda_visible_on_login=", torch.cuda.is_available())'
```

模型缺失时只在 CPU 作业下载：

```bash
hf download Qwen/Qwen3-VL-8B-Instruct --local-dir "${MODEL_DIR}"
test -s "${MODEL_DIR}/config.json"
```

### 4.2 CPU Prepare 作业

```bash
cd "${REVIEWER_ROOT}"
PREPARE_JOB_RAW=$(sbatch --parsable hpc/grpo_v3/human_preference_reviewer/v1/prepare.sbatch)
PREPARE_JOB=${PREPARE_JOB_RAW%%;*}
echo "PREPARE_JOB=${PREPARE_JOB}"
sacct -j "${PREPARE_JOB}" -o JobID,JobName%30,State,ExitCode,Elapsed,MaxRSS
```

Prepare 作业完成后验证：

```bash
sha256sum "${CSV_PATH}"
"${PYTHON}" -c 'import json,sys; r=json.load(open(sys.argv[1])); m=r["split_manifest"]; s=m["label_support"]; assert r["row_count"]==420; assert r["evidence_count"]==70; assert r["eligible_evidence_count"]==70; assert r["eligible_candidate_count"]==420; assert r["csv_sha256"]=="32679019FD7C665A0632E9885405BDF13C77B51386EFC56E7B29B24192210CD7"; assert m["split_mode"]=="train_validation" and len(m["train_evidence_ids"])==60 and len(m["validation_evidence_ids"])==10; assert not m["locked_test_evidence_ids"] and not m["reserve_evidence_ids"]; assert all(int(n)>0 for split in ("train","validation") for grades in s[split].values() for n in grades.values()); print(json.dumps(r,indent=2))' "${DATA_DIR}/annotation_audit.json"
test -s "${DATA_DIR}/split_train_60_validation_10.json"
test -s "${DATA_DIR}/split_2_1_1.json"
"${PYTHON}" -c 'import json,sys,os; m=json.load(open(sys.argv[1])); assert len(m)==140; assert all(os.path.isfile(p) and os.path.getsize(p)>0 for p in m.values()); print("MEDIA_MAP_OK",len(m))' "${MEDIA_MAP}"
"${PYTHON}" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"]=="passed" and r["source_count"]==140; assert r["resolved_from_prior_jobs_count"]==100; assert r["reused_import_count"]+r["downloaded_count"]==40; print(json.dumps(r,indent=2))' "${DATA_DIR}/media_materialization_report.json"
```

本轮 60/10 split、仅供手动 smoke/overfit 的 2/1/1 probe split、140 个视频映射和 `media_materialization_report.json` 必须全部存在；任何一个检查失败都不要提交 H100 作业。正式训练只读取 60/10 split。

### 4.3 一次提交完整首轮链

若希望自动按依赖执行所有 Gate 与正式训练，不要逐个手动提交后续章节中的作业，直接运行：

```bash
cd "${REVIEWER_ROOT}"
bash hpc/grpo_v3/human_preference_reviewer/v1/submit_first_run.sh
```

该 helper 提交 `CPU prepare → CPU structure → H100 train`。两个验证作业都不申请 GPU，并使用 `afterok` 依赖；任一 CPU 验证失败时，正式 60/10 训练不会启动。

## 5. Gate 1：CPU Structure Probe

```bash
cd "${HPC_ROOT}"
STRUCTURE_JOB_RAW=$(sbatch --parsable hpc/grpo_v3/human_preference_reviewer/v1/structure_probe.sbatch)
STRUCTURE_JOB=${STRUCTURE_JOB_RAW%%;*}
STRUCTURE_DIR=${OUTPUT_ROOT}/structure_${STRUCTURE_JOB}
printf 'STRUCTURE_JOB=%s\nSTRUCTURE_DIR=%s\n' "${STRUCTURE_JOB}" "${STRUCTURE_DIR}" > "${OUTPUT_ROOT}/structure_submission_${STRUCTURE_JOB}.env"
echo "STRUCTURE_JOB=${STRUCTURE_JOB}"
```

```bash
squeue -j "${STRUCTURE_JOB}" -o '%.18i %.24j %.10T %.10M %.10l %R' 2>/dev/null || true
sacct -j "${STRUCTURE_JOB}" -o JobID,JobName%30,State,ExitCode,Elapsed,MaxRSS
tail -n 100 "${HPC_ROOT}/hpc/logs/reviewer-v1-structure-${STRUCTURE_JOB}.out"
tail -n 100 "${HPC_ROOT}/hpc/logs/reviewer-v1-structure-${STRUCTURE_JOB}.err"
"${PYTHON}" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"]=="passed"; assert r["shared_stack_path"]=="model.language_model.layers"; assert r["shared_layer_count"]==36; assert r["target_layer_indices"]==[34,35]; assert len(r["lora_targets"])==4; print(json.dumps(r,indent=2))' "${STRUCTURE_DIR}/structure_probe.json"
```

Structure Probe 使用 `accelerate.init_empty_weights()` 从 config 构建 meta model，只检查 36 层结构和最后两层 q/v LoRA targets；它不加载模型权重，也不申请 GPU。

## 6. Gate 2：正式 60-train / 10-validation

重新确认 manifest 与 CSV hash 后执行：

```bash
"${PYTHON}" -m training.grpo_v3.experiments.human_preference_reviewer.v1.audit annotation-csv \
  --csv "${CSV_PATH}" --output "${DATA_DIR}/annotation_audit_formal.json" \
  --split-output "${DATA_DIR}/split_train_60_validation_10.json" --split-mode train_validation \
  --train-evidence-count 60 --validation-evidence-count 10 --locked-test-evidence-count 0 \
  --require-formal-split
```

返回非零就停止，不把 pending 或空标签补入 split。通过后：

```bash
TRAIN_JOB_RAW=$(sbatch --parsable hpc/grpo_v3/human_preference_reviewer/v1/train.sbatch)
TRAIN_JOB=${TRAIN_JOB_RAW%%;*}
TRAIN_DIR=${OUTPUT_ROOT}/train_${TRAIN_JOB}
printf 'TRAIN_JOB=%s\nTRAIN_DIR=%s\n' "${TRAIN_JOB}" "${TRAIN_DIR}" > "${OUTPUT_ROOT}/train_submission_${TRAIN_JOB}.env"
```

```bash
sacct -j "${TRAIN_JOB}" -o JobID,JobName%30,State,ExitCode,Elapsed,MaxRSS
test -s "${TRAIN_DIR}/training_result.json"
test -s "${TRAIN_DIR}/checkpoint/reviewer_v1_config.json"
CHECKPOINT_DIR=${TRAIN_DIR}/checkpoint
```

确认 `training_result.json` 中 `training_split_mode=train_validation`、`training_evidence_count=60`、`validation_status=evaluated`，并检查逐 head `validation_*`、加权 `validation_weighted_*` 摘要与 `gradient_conflict_summary.negative_fraction`。十个 validation packets 可用于训练诊断、early stopping 和 checkpoint 比较，但不能替代独立 test 或用于宣传最终泛化性能。

## 7. 后续 External Locked Test

当前有 10-packet internal validation，但没有 test。独立标注 5–10 个新 packets 后，先上传其 CSV，构建独立 media map 和 `external_holdout` manifest：

```bash
TEST_CSV_PATH=${DATA_DIR}/rlhf_candidate_scores_external_test.csv
TEST_MEDIA_MAP=${DATA_DIR}/external_test_media_map.json
TEST_SPLIT_MANIFEST=${DATA_DIR}/split_external_test.json
"${PYTHON}" -m training.grpo_v3.experiments.human_preference_reviewer.v1.audit annotation-csv \
  --csv "${TEST_CSV_PATH}" --output "${DATA_DIR}/external_test_audit.json" \
  --split-output "${TEST_SPLIT_MANIFEST}" --split-mode external_holdout \
  --require-formal-split
"${PYTHON}" -m training.grpo_v3.experiments.human_preference_reviewer.v1.audit media-map \
  --csv "${TEST_CSV_PATH}" --reward-output-root "${REWARD_OUTPUT_ROOT}" --output "${TEST_MEDIA_MAP}"
```

External Locked Test 只运行一次；evaluator 会拒绝任何与 checkpoint training evidence IDs 重叠的 packet：

```bash
TEST_JOB_RAW=$(sbatch --parsable --export=ALL,CHECKPOINT_DIR="${CHECKPOINT_DIR}",CSV_PATH="${TEST_CSV_PATH}",MEDIA_MAP="${TEST_MEDIA_MAP}",EVAL_SPLIT_MANIFEST="${TEST_SPLIT_MANIFEST}" hpc/grpo_v3/human_preference_reviewer/v1/evaluate.sbatch)
TEST_JOB=${TEST_JOB_RAW%%;*}
TEST_DIR=${OUTPUT_ROOT}/evaluate_${TEST_JOB}
```

报告每个字段的 CE loss、accuracy、balanced accuracy、macro-F1、AUROC、2×2 confusion matrix，以及 fail/pass 两类的 precision/recall/F1/support。任一类 support 为 0 时，`insufficient_class_support=true`，不得写成完整二分类结论。

## 8. 失败证据收集

替换真实 JobID 和 mode：

```bash
JOB_ID=12345678
MODE=train
JOB_DIR=${OUTPUT_ROOT}/${MODE}_${JOB_ID}
DIAG_DIR=${OUTPUT_ROOT}/diagnostics/${MODE}_${JOB_ID}
mkdir -p "${DIAG_DIR}"
sacct -j "${JOB_ID}" -o JobID,JobName%30,State,ExitCode,Elapsed,Start,End,MaxRSS > "${DIAG_DIR}/sacct.txt" 2>&1 || true
scontrol show job -dd "${JOB_ID}" > "${DIAG_DIR}/scontrol.txt" 2>&1 || true
cp -f "${HPC_ROOT}/hpc/logs/"*"${JOB_ID}"*.out "${DIAG_DIR}/" 2>/dev/null || true
cp -f "${HPC_ROOT}/hpc/logs/"*"${JOB_ID}"*.err "${DIAG_DIR}/" 2>/dev/null || true
cp -f "${JOB_DIR}/storage_preflight.json" "${DIAG_DIR}/" 2>/dev/null || true
cp -f "${JOB_DIR}/training_result.json" "${DIAG_DIR}/" 2>/dev/null || true
cp -f "${JOB_DIR}/evaluation_result.json" "${DIAG_DIR}/" 2>/dev/null || true
cp -f "${JOB_DIR}/checkpoint/parameter_audit.json" "${DIAG_DIR}/" 2>/dev/null || true
tar -czf "${DIAG_DIR}.tar.gz" -C "$(dirname "${DIAG_DIR}")" "$(basename "${DIAG_DIR}")"
echo "DIAGNOSTIC_BUNDLE=${DIAG_DIR}.tar.gz"
```

`squeue` 只看活动作业，历史最终状态看 `sacct` 顶层和 `.batch`。所有输出由真实 JobID 推导，不从固定或 latest 目录归因。

## 9. 资源与汇报

首轮只有正式训练申请 1×H100。训练结束后用：

```bash
sacct -j "${TRAIN_JOB}" --units=G -o JobID,State,ExitCode,Elapsed,AllocTRES%40,MaxRSS
```

下一同形状作业内存取 `MaxRSS × 1.25` 后向常用档位取整；时限按实测启动成本与每 step 耗时外推。

```text
阶段：
branch / commit：
CSV SHA-256：
Job ID / 输出目录：
Slurm State / ExitCode：
第一个失败 Gate：
数据 evidence / candidate / 各级 support：
实际 LoRA blocks / modules：
total / trainable / heads / LoRA 参数量：
三个字段 validation/test 指标：
本次能证明：
本次不能证明：
Elapsed / MaxRSS / GPU peak：
下一步唯一动作：
```

CPU 验证通过不能证明真实 H100 forward/backward；首个 GPU 检查发生在正式训练作业内。Slurm `COMPLETED` 也不能替代 `storage_preflight.json`、`structure_probe.json`、`training_result.json`、`parameter_audit.json` 和 `evaluation_result.json`。
