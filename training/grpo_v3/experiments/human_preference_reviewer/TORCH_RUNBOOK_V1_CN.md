# Reviewer v1 Torch 运行手册

本手册服从 `docs/TORCH_EXPERIMENT_META_RULES_CN.md`。Markdown 只供人工阅读，不是远端作业依赖。

## 1. 范围与硬 Gate

```text
零 GPU → Structure Probe → Smoke → Overfit Probe → 40/10/10 Train → Validation → Locked Test
```

Reviewer v1 只训练 Qwen3-VL-8B 最后两个 shared language blocks 的 q/v LoRA 与三个独立 3-class heads。不运行 overall preference、pairwise/tie loss、GRPO 或 no-video ablation。

当前 CSV SHA-256：

```text
F3E006B3A488A3ACA86C8F3B1862392EF3576A73BA78EA202E40F7754DB730AC
```

当前严格可用数据为 44 completed evidence；正式 40/10/10 还缺 16 个。现在可以执行前四个 Gate，不能提交正式 Train 或 Locked Test。

## 2. 每次 SSH 初始化

```bash
NETID=xl6775
TORCH_ACCOUNT=torch_pr_674_tandon_advanced
CLEAN_ROOT=/scratch/xl6775/projects/EgoQA-two-user-grpo-clean
OUTPUT_ROOT=${CLEAN_ROOT}/outputs/human_preference_reviewer/v1
DATA_DIR=${CLEAN_ROOT}/data_RLHF/reviewer_v1
TRAIN_ENV=/scratch/xl6775/envs/egoqa-ms-swift-v4.2.2-vllm024
FFMPEG_ENV=/scratch/xl6775/envs/egoqa-ffmpeg-runtime
MODEL_DIR=/scratch/xl6775/models/Qwen3-VL-8B-Instruct
EGO_LIFE_ROOT=/scratch/xl6775/datasets/EgoLife
CSV_PATH=${DATA_DIR}/rlhf_candidate_scores_day5_7_full_100_HM.csv
MEDIA_MAP=${DATA_DIR}/media_map.json
PYTHON=${TRAIN_ENV}/bin/python
mkdir -p "${CLEAN_ROOT}/logs" "${OUTPUT_ROOT}" "${DATA_DIR}"
cd "${CLEAN_ROOT}"
```

供直接粘贴的 SSH 命令不启用全局 strict mode，也不执行 `exit`。失败后保留会话收集证据。

## 3. Windows → Torch 窄 SFTP

SSH 登录节点先建目录：

```bash
mkdir -p /scratch/xl6775/projects/EgoQA-two-user-grpo-clean/training/grpo_v3/experiments/human_preference_reviewer/v1
mkdir -p /scratch/xl6775/projects/EgoQA-two-user-grpo-clean/tests/training/grpo_v3/experiments/human_preference_reviewer/v1
mkdir -p /scratch/xl6775/projects/EgoQA-two-user-grpo-clean/hpc/grpo_v3/human_preference_reviewer/v1
mkdir -p /scratch/xl6775/projects/EgoQA-two-user-grpo-clean/data_RLHF/reviewer_v1
```

Windows PowerShell：

```text
sftp xl6775@login.torch.hpc.nyu.edu
lcd C:/Users/20661/Desktop/Research/AR/multiuser/EgoQA-two-user-reviewer-v1
cd /scratch/xl6775/projects/EgoQA-two-user-grpo-clean
put training/grpo_v3/experiments/human_preference_reviewer/__init__.py training/grpo_v3/experiments/human_preference_reviewer/__init__.py
put training/grpo_v3/experiments/human_preference_reviewer/v1/*.py training/grpo_v3/experiments/human_preference_reviewer/v1/
put tests/training/grpo_v3/experiments/human_preference_reviewer/v1/*.py tests/training/grpo_v3/experiments/human_preference_reviewer/v1/
put hpc/grpo_v3/human_preference_reviewer/v1/common.sh hpc/grpo_v3/human_preference_reviewer/v1/common.sh
put hpc/grpo_v3/human_preference_reviewer/v1/*.sbatch hpc/grpo_v3/human_preference_reviewer/v1/
put training/grpo_v3/experiments/human_preference_reviewer/TORCH_RUNBOOK_V1_CN.md training/grpo_v3/experiments/human_preference_reviewer/TORCH_RUNBOOK_V1_CN.md
lcd C:/Users/20661/Documents/xwechat_files/wxid_i096w25uhusk22_e748/msg/file/2026-08
put "rlhf_candidate_scores_day5_7_full_100_HM (1)(1).csv" /scratch/xl6775/projects/EgoQA-two-user-grpo-clean/data_RLHF/reviewer_v1/rlhf_candidate_scores_day5_7_full_100_HM.csv
bye
```

这里只上传窄代码集合和单个 CSV；不要上传模型、视频、cache、outputs 或整个 `data_RLHF`。

## 4. Gate 0：零 GPU 预检

### 4.1 接收、语法、依赖

```bash
cd "${CLEAN_ROOT}"
git status --short -- training/grpo_v3/experiments/human_preference_reviewer hpc/grpo_v3/human_preference_reviewer
bash -n hpc/grpo_v3/human_preference_reviewer/v1/common.sh
bash -n hpc/grpo_v3/human_preference_reviewer/v1/structure_probe.sbatch
bash -n hpc/grpo_v3/human_preference_reviewer/v1/smoke1.sbatch
bash -n hpc/grpo_v3/human_preference_reviewer/v1/overfit_probe.sbatch
bash -n hpc/grpo_v3/human_preference_reviewer/v1/train.sbatch
bash -n hpc/grpo_v3/human_preference_reviewer/v1/evaluate.sbatch

export PATH="${FFMPEG_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${FFMPEG_ENV}/lib:${LD_LIBRARY_PATH:-}"
"${PYTHON}" -c 'import torch,transformers,peft,accelerate; print(torch.__version__,transformers.__version__,peft.__version__,accelerate.__version__)'
"${PYTHON}" -c 'from torchcodec.decoders import VideoDecoder; print(VideoDecoder.__module__)'
"${FFMPEG_ENV}/bin/ffmpeg" -version | head -n 2
test -s "${MODEL_DIR}/config.json" && echo MODEL_CONFIG_OK
command -v hf
```

模型缺失时只在 CPU 作业下载：

```bash
hf download Qwen/Qwen3-VL-8B-Instruct --local-dir "${MODEL_DIR}"
test -s "${MODEL_DIR}/config.json"
```

### 4.2 Targeted tests

```bash
"${PYTHON}" -m unittest discover -s tests/training/grpo_v3/experiments/human_preference_reviewer/v1 -p 'test_*.py' -v
"${PYTHON}" -m compileall -q training/grpo_v3/experiments/human_preference_reviewer/v1
```

Torch 环境不得跳过 PyTorch-specific tests。

### 4.3 CSV audit 与 split

```bash
sha256sum "${CSV_PATH}"
"${PYTHON}" -m training.grpo_v3.experiments.human_preference_reviewer.v1.audit annotation-csv \
  --csv "${CSV_PATH}" --output "${DATA_DIR}/annotation_audit.json"
"${PYTHON}" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["row_count"]==600; assert r["evidence_count"]==100; assert r["csv_sha256"]=="F3E006B3A488A3ACA86C8F3B1862392EF3576A73BA78EA202E40F7754DB730AC"; print(json.dumps(r,indent=2))' "${DATA_DIR}/annotation_audit.json"
```

当前 formal split Gate 预期失败，这是数据完成度结论，不是代码失败。Overfit 使用独立 2/1/1 manifest：

```bash
"${PYTHON}" -m training.grpo_v3.experiments.human_preference_reviewer.v1.audit annotation-csv \
  --csv "${CSV_PATH}" --train-evidence-count 2 --validation-evidence-count 1 \
  --locked-test-evidence-count 1 --split-output "${DATA_DIR}/split_2_1_1.json" \
  --output "${DATA_DIR}/annotation_audit_2_1_1.json"
```

### 4.4 视频映射

```bash
"${PYTHON}" -m training.grpo_v3.experiments.human_preference_reviewer.v1.audit media-map \
  --csv "${CSV_PATH}" --dataset-root "${EGO_LIFE_ROOT}" --output "${MEDIA_MAP}"
"${PYTHON}" -c 'import json,sys,os; m=json.load(open(sys.argv[1])); assert len(m)==88; assert all(os.path.isfile(p) and os.path.getsize(p)>0 for p in m.values()); print("MEDIA_MAP_OK",len(m))' "${MEDIA_MAP}"
```

若缺文件，按 audit 报出的相对路径用 `hf download lmms-lab/EgoLife --repo-type dataset <relative-path> --local-dir "${EGO_LIFE_ROOT}"` 在 CPU 作业补齐，不占用 H100 下载。

## 5. Gate 1：Structure Probe

```bash
STRUCTURE_JOB_RAW=$(sbatch --parsable hpc/grpo_v3/human_preference_reviewer/v1/structure_probe.sbatch)
STRUCTURE_JOB=${STRUCTURE_JOB_RAW%%;*}
STRUCTURE_DIR=${OUTPUT_ROOT}/structure_${STRUCTURE_JOB}
printf 'STRUCTURE_JOB=%s\nSTRUCTURE_DIR=%s\n' "${STRUCTURE_JOB}" "${STRUCTURE_DIR}" > "${OUTPUT_ROOT}/structure_submission_${STRUCTURE_JOB}.env"
echo "STRUCTURE_JOB=${STRUCTURE_JOB}"
```

```bash
squeue -j "${STRUCTURE_JOB}" -o '%.18i %.24j %.10T %.10M %.10l %R' 2>/dev/null || true
sacct -j "${STRUCTURE_JOB}" -o JobID,JobName%30,State,ExitCode,Elapsed,MaxRSS
tail -n 100 "${CLEAN_ROOT}/logs/reviewer-v1-structure-${STRUCTURE_JOB}.out"
tail -n 100 "${CLEAN_ROOT}/logs/reviewer-v1-structure-${STRUCTURE_JOB}.err"
"${PYTHON}" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"]=="passed"; assert r["shared_stack_path"]=="model.language_model.layers"; assert r["shared_layer_count"]==36; assert r["target_layer_indices"]==[34,35]; assert len(r["lora_targets"])==4; print(json.dumps(r,indent=2))' "${STRUCTURE_DIR}/structure_probe.json"
```

## 6. Gate 2：真实双视频 1-step Smoke

```bash
SMOKE_JOB_RAW=$(sbatch --parsable hpc/grpo_v3/human_preference_reviewer/v1/smoke1.sbatch)
SMOKE_JOB=${SMOKE_JOB_RAW%%;*}
SMOKE_DIR=${OUTPUT_ROOT}/smoke_${SMOKE_JOB}
printf 'SMOKE_JOB=%s\nSMOKE_DIR=%s\n' "${SMOKE_JOB}" "${SMOKE_DIR}" > "${OUTPUT_ROOT}/smoke_submission_${SMOKE_JOB}.env"
```

```bash
sacct -j "${SMOKE_JOB}" -o JobID,JobName%30,State,ExitCode,Elapsed,MaxRSS
"${PYTHON}" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"]=="passed"; assert r["global_step"]==1; assert r["head_parameter_delta_nonzero"]; assert r["lora_parameter_delta_nonzero"]; assert all(v["status"]=="passed" for v in r["gradient_routes"].values()); assert not r["parameter_audit"]["unexpected_trainable_names"]; print(json.dumps(r,indent=2))' "${SMOKE_DIR}/training_result.json"
test -s "${SMOKE_DIR}/checkpoint/classification_heads.pt"
test -s "${SMOKE_DIR}/checkpoint/lora_adapter.pt"
test -s "${SMOKE_DIR}/checkpoint/parameter_audit.json"
```

Smoke 只证明真实视频 forward/backward、三个 heads、共享 LoRA、冻结审计和保存成立，不证明泛化。

## 7. Gate 3：24-step Overfit Probe

```bash
OVERFIT_JOB_RAW=$(sbatch --parsable hpc/grpo_v3/human_preference_reviewer/v1/overfit_probe.sbatch)
OVERFIT_JOB=${OVERFIT_JOB_RAW%%;*}
OVERFIT_DIR=${OUTPUT_ROOT}/overfit_${OVERFIT_JOB}
```

```bash
sacct -j "${OVERFIT_JOB}" -o JobID,JobName%30,State,ExitCode,Elapsed,MaxRSS
"${PYTHON}" -c 'import json,sys; r=json.load(open(sys.argv[1])); d=r["repeated_candidate_loss"]; improved=sum(x["improved"] for x in d.values()); print({"repeated":len(d),"improved":improved,"throughput":r["throughput"]}); assert r["status"]=="passed" and len(d)>=6 and improved/len(d)>=0.75' "${OVERFIT_DIR}/training_result.json"
```

该 Gate 比较同一 candidate 第一次和最后一次出现时的 loss，避免把随机样本难度误当成学习趋势。失败时先检查 loss、grad norm、parameter delta 与视频输入，不直接扩大规模。

同时用 `candidate_steps_per_hour` 做 40 evidence × 6 candidate × 3 epoch（共 720 candidate-step）的耗时外推；若预计训练时间超过作业时限的 70%，先增加时限或单独设计可验证的视频复用优化，再提交正式训练。

## 8. Gate 4：正式 40/10/10

至少 60 个 completed evidence 后才执行：

```bash
"${PYTHON}" -m training.grpo_v3.experiments.human_preference_reviewer.v1.audit annotation-csv \
  --csv "${CSV_PATH}" --output "${DATA_DIR}/annotation_audit_formal.json" \
  --split-output "${DATA_DIR}/split_40_10_10.json" --require-formal-split
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

## 9. Validation 与 Locked Test

```bash
VALID_JOB_RAW=$(sbatch --parsable --export=ALL,CHECKPOINT_DIR="${CHECKPOINT_DIR}",EVAL_SPLIT=validation hpc/grpo_v3/human_preference_reviewer/v1/evaluate.sbatch)
VALID_JOB=${VALID_JOB_RAW%%;*}
VALID_DIR=${OUTPUT_ROOT}/evaluate_${VALID_JOB}
sacct -j "${VALID_JOB}" -o JobID,JobName%30,State,ExitCode,Elapsed,MaxRSS
"${PYTHON}" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"]=="passed"; print(json.dumps(r["metrics"],indent=2))' "${VALID_DIR}/evaluation_result.json"
```

checkpoint 选择完成后，Locked Test 只运行一次：

```bash
TEST_JOB_RAW=$(sbatch --parsable --export=ALL,CHECKPOINT_DIR="${CHECKPOINT_DIR}",EVAL_SPLIT=locked_test hpc/grpo_v3/human_preference_reviewer/v1/evaluate.sbatch)
TEST_JOB=${TEST_JOB_RAW%%;*}
TEST_DIR=${OUTPUT_ROOT}/evaluate_${TEST_JOB}
```

报告每个字段的 loss、accuracy、macro-F1、3×3 confusion matrix、每级 precision/recall/F1/support、expected-score MAE 与 Spearman。任一级 support 为 0 时，`insufficient_class_support=true`，不得写成完整三分类结论。

## 10. 失败证据收集

替换真实 JobID 和 mode：

```bash
JOB_ID=12345678
MODE=smoke
JOB_DIR=${OUTPUT_ROOT}/${MODE}_${JOB_ID}
DIAG_DIR=${OUTPUT_ROOT}/diagnostics/${MODE}_${JOB_ID}
mkdir -p "${DIAG_DIR}"
sacct -j "${JOB_ID}" -o JobID,JobName%30,State,ExitCode,Elapsed,Start,End,MaxRSS > "${DIAG_DIR}/sacct.txt" 2>&1 || true
scontrol show job -dd "${JOB_ID}" > "${DIAG_DIR}/scontrol.txt" 2>&1 || true
cp -f "${CLEAN_ROOT}/logs/"*"${JOB_ID}"*.out "${DIAG_DIR}/" 2>/dev/null || true
cp -f "${CLEAN_ROOT}/logs/"*"${JOB_ID}"*.err "${DIAG_DIR}/" 2>/dev/null || true
cp -f "${JOB_DIR}/storage_preflight.json" "${DIAG_DIR}/" 2>/dev/null || true
cp -f "${JOB_DIR}/training_result.json" "${DIAG_DIR}/" 2>/dev/null || true
cp -f "${JOB_DIR}/evaluation_result.json" "${DIAG_DIR}/" 2>/dev/null || true
cp -f "${JOB_DIR}/checkpoint/parameter_audit.json" "${DIAG_DIR}/" 2>/dev/null || true
tar -czf "${DIAG_DIR}.tar.gz" -C "$(dirname "${DIAG_DIR}")" "$(basename "${DIAG_DIR}")"
echo "DIAGNOSTIC_BUNDLE=${DIAG_DIR}.tar.gz"
```

`squeue` 只看活动作业，历史最终状态看 `sacct` 顶层和 `.batch`。所有输出由真实 JobID 推导，不从固定或 latest 目录归因。

## 11. 资源与汇报

首轮统一 1×H100。Smoke 后用：

```bash
sacct -j "${SMOKE_JOB}" --units=G -o JobID,State,ExitCode,Elapsed,AllocTRES%40,MaxRSS
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

本地测试通过不能证明真实 H100 runtime；Slurm `COMPLETED` 也不能替代 `storage_preflight.json`、`structure_probe.json`、`training_result.json`、`parameter_audit.json` 和 `evaluation_result.json`。
