# 标注 Pareto-DPO：Torch/H100 运行手册

## 固定合同与边界

- 仅在工作树 `feature/annotated-pareto-dpo` 提交；登录集群后先确认 `git branch --show-current`。
- 标注 CSV 的 SHA-256 必须为 `32679019FD7C665A0632E9885405BDF13C77B51386EFC56E7B29B24192210CD7`；证据级划分固定为 `60/10/0`（训练/验证/锁定测试），锁定测试为零行，不能拿验证集替代它。
- 默认合同以 `hpc/grpo_v3/annotated_preference/common.sh` 为准：项目 `/scratch/xl6775/projects/EgoQA-two-user-annotated-pareto-dpo`；环境 `/scratch/xl6775/envs/egoqa-ms-swift-v4.2.2-vllm024`；模型 `/scratch/xl6775/models/Qwen3-VL-8B-Instruct`；数据 `${PROJECT_ROOT}/data_RLHF/annotated_preference`；输出 `${PROJECT_ROOT}/outputs/annotated_preference`。
- `compact_qa_v1` 只是当前标注数据的紧凑提示词合同，**不是**生产 `expanded schema` 的替换；生产端的扩展字段、解析和兼容性仍须单独验收。
- 本手册中的登录 shell 块故意只打印 `MISSING` 或 `STOP` 并跳过后续依赖，保持 SSH 会话可用。批处理脚本内部可采用严格失败策略。

## 登录节点预检（可直接粘贴）

```bash
export PROJECT_ROOT=/scratch/xl6775/projects/EgoQA-two-user-annotated-pareto-dpo
export TRAIN_ENV=/scratch/xl6775/envs/egoqa-ms-swift-v4.2.2-vllm024
export MODEL_DIR=/scratch/xl6775/models/Qwen3-VL-8B-Instruct
export DATA_DIR=${PROJECT_ROOT}/data_RLHF/annotated_preference
export OUTPUT_ROOT=${PROJECT_ROOT}/outputs/annotated_preference
export CSV_PATH=${DATA_DIR}/rlhf_candidate_scores_merged_70_packets.csv
export SPLIT_PATH=${DATA_DIR}/split_60_10.json
export MEDIA_MAP=${DATA_DIR}/media_map.json
export DPO_DATA_DIR=${DATA_DIR}/dpo
EXPECTED_SHA=32679019FD7C665A0632E9885405BDF13C77B51386EFC56E7B29B24192210CD7

if [ "$(git -C "${PROJECT_ROOT}" branch --show-current 2>/dev/null)" = "feature/annotated-pareto-dpo" ]; then
  echo "branch=OK"
else
  echo "STOP: wrong or missing worktree branch"
fi
if [ -x "${TRAIN_ENV}/bin/python" ] && [ -d "${MODEL_DIR}" ] && [ -f "${CSV_PATH}" ] && [ -f "${SPLIT_PATH}" ] && [ -f "${MEDIA_MAP}" ] && [ -d "${DPO_DATA_DIR}" ]; then
  ACTUAL_SHA=$(sha256sum "${CSV_PATH}" | awk '{print toupper($1)}')
  if [ "${ACTUAL_SHA}" = "${EXPECTED_SHA}" ]; then echo "data=OK split=60/10/0"; else echo "STOP: CSV SHA mismatch"; fi
else
  echo "MISSING: env, model, CSV, split, media map, or DPO data"
fi
mkdir -p "${OUTPUT_ROOT}"
```

每次提交均用 `--parsable` 得到唯一 JobID；以下通用监控/取证块中的 `${JOBID}` 必须替换为本次命令返回值。

```bash
if [ -n "${JOBID:-}" ]; then
  squeue -j "${JOBID}" -o "%.18i %.9T %.12j %.10M %.20R"
  sacct -j "${JOBID}" --format=JobID,JobName%28,State,Elapsed,AllocTRES
  JOB_OUT="${OUTPUT_ROOT}/jobs/${JOBID}"
  mkdir -p "${JOB_OUT}"
  cp "${PROJECT_ROOT}/logs/"*"${JOBID}"*.out "${JOB_OUT}/" 2>/dev/null || true
  cp "${PROJECT_ROOT}/logs/"*"${JOBID}"*.err "${JOB_OUT}/" 2>/dev/null || true
else
  echo "STOP: JOBID is empty; do not infer an output directory"
fi
```

## Gate 0：数据与划分审计

输入：上述 CSV、SHA 与 `60/10/0` 配置。资源：CPU/零 GPU。提交：

```bash
SCRIPT=${PROJECT_ROOT}/hpc/grpo_v3/annotated_preference/gate0_data.sbatch
if [ -f "${SCRIPT}" ]; then JOBID=$(sbatch --parsable "${SCRIPT}"); echo "JOBID=${JOBID}"; else echo "MISSING: ${SCRIPT}"; fi
```

监控使用上节的 `squeue -j`、结束后 `sacct -j`；输出只认 `${OUTPUT_ROOT}/gate0_${JOBID}`。验收：报告 CSV 指纹、每个 evidence 仅归属一个 split，且计数 `60/10/0`。失败时收集 `${JOBID}` 的 `.out/.err`、输入 SHA 和划分 JSON。它能证明输入冻结与划分正确；不能证明模型能加载或 DPO 能收敛。

## Gate 1：模型结构探针

输入：Gate 0 通过的数据、模型目录。资源：1×H100、4 CPU、64G、1 小时。提交：

```bash
SCRIPT=${PROJECT_ROOT}/hpc/grpo_v3/annotated_preference/structure_probe.sbatch
if [ -f "${SCRIPT}" ]; then JOBID=$(sbatch --parsable "${SCRIPT}"); echo "JOBID=${JOBID}"; else echo "MISSING: ${SCRIPT}"; fi
```

监控/输出：`squeue -j "${JOBID}"`、`sacct -j "${JOBID}"`，只读取 `${OUTPUT_ROOT}/structure_${JOBID}`。验收：结构 JSON 记录模型标识、层数、目标模块与可训练参数审计。失败收集 JobID 日志、结构 JSON（若有）和环境版本。它能证明模型结构与适配目标可见；不能证明视频样本、反传或训练质量。

## Gate 2：单步 Smoke

输入：Gate 0 数据、Gate 1 结构审计与至少一条多模态样本。资源：1×H100、8 CPU、128G、3 小时。提交：

```bash
SCRIPT=${PROJECT_ROOT}/hpc/grpo_v3/annotated_preference/smoke1.sbatch
if [ -f "${SCRIPT}" ]; then JOBID=$(sbatch --parsable "${SCRIPT}"); echo "JOBID=${JOBID}"; else echo "MISSING: ${SCRIPT}"; fi
```

监控/输出：用 `squeue -j`、`sacct -j`，只认 `${OUTPUT_ROOT}/smoke_${JOBID}`。验收：一次前向、损失、反向、优化器更新和检查点写入均完成。失败收集 JobID 日志、CUDA OOM、解码器与依赖版本。它能证明该最小路径可运行；不能证明 Pareto 偏好目标可学习或泛化。

## Gate 3：小集过拟合探针

输入：Gate 2 样本与固定小训练集。资源：1×H100、8 CPU、128G、5 小时。提交：

```bash
SCRIPT=${PROJECT_ROOT}/hpc/grpo_v3/annotated_preference/overfit_probe.sbatch
if [ -f "${SCRIPT}" ]; then JOBID=$(sbatch --parsable "${SCRIPT}"); echo "JOBID=${JOBID}"; else echo "MISSING: ${SCRIPT}"; fi
```

监控/输出：`squeue -j "${JOBID}"` 与 `sacct -j "${JOBID}"`；只读 `${OUTPUT_ROOT}/overfit_${JOBID}`。验收：固定小集的训练损失持续下降，并保存曲线、配置和检查点。失败收集曲线、最后检查点、随机种子和 JobID 日志。它能证明损失/梯度链路可拟合；不能证明 60% 训练后的验证效果。

## Gate 4：60% 训练

输入：Gate 0 的 60% 训练集、Gate 3 配置。资源：1×H100、8 CPU、160G、12 小时。提交：

```bash
OVERFIT_JOBID=替换为Gate3返回的JobID
OVERFIT_RESULT=${OUTPUT_ROOT}/overfit_${OVERFIT_JOBID}/dpo_gate_result.json
SCRIPT=${PROJECT_ROOT}/hpc/grpo_v3/annotated_preference/train.sbatch
if [ -n "${OVERFIT_JOBID}" ] && [ -f "${OVERFIT_RESULT}" ] && [ -f "${SCRIPT}" ]; then JOBID=$(sbatch --parsable --export=ALL,OVERFIT_RESULT="${OVERFIT_RESULT}" "${SCRIPT}"); echo "JOBID=${JOBID}"; else echo "STOP: missing OVERFIT_JOBID, result, or ${SCRIPT}"; fi
```

监控/输出：`squeue -j "${JOBID}"`、`sacct -j "${JOBID}"`；产物只能来自 `${OUTPUT_ROOT}/train_${JOBID}`，包括最终检查点、训练曲线、配置、Git 提交和输入 SHA。验收：作业 COMPLETED、检查点可读、无 NaN/Inf，且所有输入合同被记录。失败收集完整 `.out/.err`、`sacct`、最后检查点和曲线。它能证明一次受控训练完成；不能证明 10% 验证集质量，也不能证明生产 expanded schema 兼容。

## Gate 5：10% 验证

输入：Gate 4 的 JobID 检查点、固定 10% 验证集；不得读取零行锁定测试。资源：1×H100、8 CPU、128G、5 小时。提交时显式传递训练 JobID，避免歧义：

```bash
TRAIN_JOBID=替换为Gate4返回的JobID
ADAPTER_DIR=${OUTPUT_ROOT}/train_${TRAIN_JOBID}/adapter
SCRIPT=${PROJECT_ROOT}/hpc/grpo_v3/annotated_preference/evaluate.sbatch
if [ -n "${TRAIN_JOBID}" ] && [ -d "${ADAPTER_DIR}" ] && [ -f "${SCRIPT}" ]; then JOBID=$(sbatch --parsable --export=ALL,TRAIN_JOB_ID=${TRAIN_JOBID},ADAPTER_DIR=${ADAPTER_DIR} "${SCRIPT}"); echo "JOBID=${JOBID}"; else echo "STOP: missing TRAIN_JOBID, adapter, or ${SCRIPT}"; fi
```

监控/输出：`squeue -j "${JOBID}"`、`sacct -j "${JOBID}"`；只认 `${OUTPUT_ROOT}/validation_${JOBID}` 与其所指向的 `${TRAIN_JOBID}` 检查点。验收：报告验证集计数、Pareto 指标/每类支持数、混淆或失败样例，且不发生训练集泄漏。失败收集 JobID 日志、指标 JSON、预测样例和训练检查点元数据。它能证明固定验证集上的一次评估；不能证明锁定测试、真实生产流量或自由生成质量。

## Gate 6：明确未执行

Gate 6（使用生产 `expanded schema` 的端到端自由生成/人工终点评估）**未执行**，因此自由生成未验证。不得把 Gate 0–5 的静态、Smoke、过拟合、训练或验证结果表述为生产替换成功、自由生成成功或研究结论。
