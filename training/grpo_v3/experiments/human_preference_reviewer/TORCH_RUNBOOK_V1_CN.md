# Reviewer v1 Torch Runbook

本手册服从 `docs/TORCH_EXPERIMENT_META_RULES_CN.md`，用于运行完整的 **最后两层 LoRA + 三个三分类 heads**。Stage 0 单 head Gate 已通过后，再按本文执行。

```text
零 GPU → 媒体准备 → Structure Probe → Smoke → Overfit Probe → 60/10 Train → Validation
```

本轮只有 60 个 training evidence 和 10 个 validation evidence，没有 locked test。Validation 可用于开发判断和 checkpoint 选择，但不能写成独立测试集结果。

## 1. 固定合同

- 模型：`Qwen3-VL-8B-Instruct`；
- LoRA：shared language blocks 34、35 的 `q_proj`、`v_proj`；
- heads：Evidence Quality、Answerability、QA Formality；
- 总损失：三个三分类交叉熵等权平均；
- CSV：`rlhf_candidate_scores_merged_70_packets.csv`；
- CSV SHA-256：`32679019FD7C665A0632E9885405BDF13C77B51386EFC56E7B29B24192210CD7`；
- 数据：420 rows、70 evidence、每个 evidence 6 candidates；
- 划分：`60/10/0`，manifest 为 `split_60_10.json`；
- 视频：140 个唯一 MP4，覆盖 Day 1、Day 5、Day 6。

当前明确不训练 Overall Utility、Bradley–Terry、tie loss、GRPO reward，也不进行 full backbone finetuning。

## 2. 每次 SSH 登录后设置变量

下面命令不会关闭 SSH 会话。若某个 Gate 失败，保留当前终端检查日志。

```bash
PROJECT_ROOT=/scratch/xl6775/projects/EgoQA-two-user-reviewer-v1
OUTPUT_ROOT=${PROJECT_ROOT}/outputs/human_preference_reviewer/v1
DATA_DIR=${PROJECT_ROOT}/data_RLHF/reviewer_v1
TRAIN_ENV=/scratch/xl6775/envs/egoqa-ms-swift-v4.2.2-vllm024
FFMPEG_ENV=/scratch/xl6775/envs/egoqa-ffmpeg-runtime
MODEL_DIR=/scratch/xl6775/models/Qwen3-VL-8B-Instruct
DATASET_ROOT=/scratch/xl6775/datasets/EgoLife
CSV_PATH=${DATA_DIR}/rlhf_candidate_scores_merged_70_packets.csv
MEDIA_MAP=${DATA_DIR}/media_map.json
FORMAL_SPLIT=${DATA_DIR}/split_60_10.json
PYTHON=${TRAIN_ENV}/bin/python

export PROJECT_ROOT OUTPUT_ROOT DATA_DIR TRAIN_ENV FFMPEG_ENV MODEL_DIR
export DATASET_ROOT CSV_PATH MEDIA_MAP FORMAL_SPLIT PYTHON

mkdir -p "${PROJECT_ROOT}/logs" "${OUTPUT_ROOT}" "${DATA_DIR}"
cd "${PROJECT_ROOT}"
```

## 3. 只上传新 CSV

在 Windows PowerShell 中执行：

```text
sftp xl6775@greene.hpc.nyu.edu
lcd C:/Users/20661/Documents/xwechat_files/wxid_i096w25uhusk22_e748/msg/file/2026-08
cd /scratch/xl6775/projects/EgoQA-two-user-reviewer-v1/data_RLHF/reviewer_v1
put "rlhf_candidate_scores_merged_70_packets.csv" rlhf_candidate_scores_merged_70_packets.csv
```

代码应通过 Git 分支或本地生成的 bundle 同步，不要把整个项目目录、模型、视频、cache 或 outputs 递归上传。

## 4. Gate 0：零 GPU 数据与代码预检

```bash
cd "${PROJECT_ROOT}"

echo "branch=$(git branch --show-current)"
echo "head=$(git rev-parse HEAD)"
git status --short --branch

sha256sum "${CSV_PATH}"

"${PYTHON}" -m training.grpo_v3.experiments.human_preference_reviewer.v1.audit annotation-csv \
  --csv "${CSV_PATH}" \
  --output "${DATA_DIR}/annotation_audit_60_10.json" \
  --split-output "${FORMAL_SPLIT}" \
  --train-evidence-count 60 \
  --validation-evidence-count 10 \
  --locked-test-evidence-count 0 \
  --seed 42 \
  --require-formal-split

"${PYTHON}" - "${DATA_DIR}/annotation_audit_60_10.json" "${FORMAL_SPLIT}" <<'PY'
import json
import sys
from pathlib import Path

audit = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
split = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
assert audit["status"] == "passed", audit["status"]
assert audit["csv_sha256"].upper() == "32679019FD7C665A0632E9885405BDF13C77B51386EFC56E7B29B24192210CD7"
assert audit["row_count"] == 420
assert audit["evidence_count"] == 70
assert len(split["train_evidence_ids"]) == 60
assert len(split["validation_evidence_ids"]) == 10
assert split["locked_test_evidence_ids"] == []
assert split["reserve_evidence_ids"] == []
assert not set(split["train_evidence_ids"]) & set(split["validation_evidence_ids"])
print("ANNOTATION_GATE_PASSED rows=420 evidence=70 split=60/10/0")
print(json.dumps(split["label_support"], ensure_ascii=False, indent=2))
PY

"${PYTHON}" -m unittest discover \
  -s tests/training/grpo_v3/experiments/human_preference_reviewer/v1 \
  -p 'test_*.py' -v
"${PYTHON}" -m compileall -q training/grpo_v3/experiments/human_preference_reviewer/v1

bash -n hpc/grpo_v3/human_preference_reviewer/v1/common.sh
bash -n hpc/grpo_v3/human_preference_reviewer/v1/prepare_media.sbatch
bash -n hpc/grpo_v3/human_preference_reviewer/v1/structure_probe.sbatch
bash -n hpc/grpo_v3/human_preference_reviewer/v1/smoke1.sbatch
bash -n hpc/grpo_v3/human_preference_reviewer/v1/overfit_probe.sbatch
bash -n hpc/grpo_v3/human_preference_reviewer/v1/train.sbatch
bash -n hpc/grpo_v3/human_preference_reviewer/v1/evaluate.sbatch
```

验收：SHA、420/70、`60/10/0`、三个字段的 1/2/3 support、单元测试和语法检查全部通过。不要额外给 `unittest discover` 传 `-t`；本仓库的 `tests/` 不是可导入 package。

## 5. Gate 0.5：准备 140 个视频

该作业只负责按 CSV 精确下载 140 个视频并生成新的 `media_map.json`。即使旧目录已有 200 个视频，也必须重新生成 map，不能假设旧 map 覆盖新 CSV。

```bash
MEDIA_JOB_RAW=$(sbatch \
  --parsable \
  --export=ALL \
  --chdir="${PROJECT_ROOT}" \
  --output="${PROJECT_ROOT}/logs/reviewer-media-%j.out" \
  --error="${PROJECT_ROOT}/logs/reviewer-media-%j.err" \
  hpc/grpo_v3/human_preference_reviewer/v1/prepare_media.sbatch)
MEDIA_JOB=${MEDIA_JOB_RAW%%;*}
echo "MEDIA_JOB=${MEDIA_JOB}"
```

作业结束后检查：

```bash
sacct -j "${MEDIA_JOB}" --format=JobID,JobName%24,State,ExitCode,Elapsed,MaxRSS
tail -n 80 "${PROJECT_ROOT}/logs/reviewer-media-${MEDIA_JOB}.out"
tail -n 80 "${PROJECT_ROOT}/logs/reviewer-media-${MEDIA_JOB}.err"

"${PYTHON}" - "${MEDIA_MAP}" <<'PY'
import json
import sys
from pathlib import Path

mapping = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert len(mapping) == 140, len(mapping)
assert all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in mapping.values())
print("MEDIA_MAP_PASSED count=140")
PY
```

## 6. Gate 1：Structure Probe

```bash
STRUCTURE_JOB_RAW=$(sbatch \
  --parsable --export=ALL --chdir="${PROJECT_ROOT}" \
  --output="${PROJECT_ROOT}/logs/reviewer-v1-structure-%j.out" \
  --error="${PROJECT_ROOT}/logs/reviewer-v1-structure-%j.err" \
  hpc/grpo_v3/human_preference_reviewer/v1/structure_probe.sbatch)
STRUCTURE_JOB=${STRUCTURE_JOB_RAW%%;*}
STRUCTURE_DIR=${OUTPUT_ROOT}/structure_${STRUCTURE_JOB}
echo "STRUCTURE_JOB=${STRUCTURE_JOB}"
```

```bash
sacct -j "${STRUCTURE_JOB}" --format=JobID,JobName%24,State,ExitCode,Elapsed,MaxRSS
"${PYTHON}" - "${STRUCTURE_DIR}/structure_probe.json" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert result["status"] == "passed"
assert result["shared_stack_path"] == "model.language_model.layers"
assert result["shared_layer_count"] == 36
assert result["target_layer_indices"] == [34, 35]
assert len(result["lora_targets"]) == 4
print(json.dumps(result, ensure_ascii=False, indent=2))
PY
```

## 7. Gate 2：真实双视频 Smoke

```bash
SMOKE_JOB_RAW=$(sbatch \
  --parsable --export=ALL --chdir="${PROJECT_ROOT}" \
  --output="${PROJECT_ROOT}/logs/reviewer-v1-smoke-%j.out" \
  --error="${PROJECT_ROOT}/logs/reviewer-v1-smoke-%j.err" \
  hpc/grpo_v3/human_preference_reviewer/v1/smoke1.sbatch)
SMOKE_JOB=${SMOKE_JOB_RAW%%;*}
SMOKE_DIR=${OUTPUT_ROOT}/smoke_${SMOKE_JOB}
echo "SMOKE_JOB=${SMOKE_JOB}"
```

```bash
sacct -j "${SMOKE_JOB}" --format=JobID,JobName%24,State,ExitCode,Elapsed,MaxRSS
"${PYTHON}" - "${SMOKE_DIR}/training_result.json" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert result["status"] == "passed"
assert result["global_step"] == 1
assert result["head_parameter_delta_nonzero"]
assert result["lora_parameter_delta_nonzero"]
assert all(route["status"] == "passed" for route in result["gradient_routes"].values())
assert not result["parameter_audit"]["unexpected_trainable_names"]
print(result["parameter_audit"])
PY

test -s "${SMOKE_DIR}/storage_preflight.json"
test -s "${SMOKE_DIR}/checkpoint/parameter_audit.json"
```

## 8. Gate 3：Overfit Probe

先生成独立的小样本 manifest；它只用于可学习性诊断，不用于正式结果。

```bash
"${PYTHON}" -m training.grpo_v3.experiments.human_preference_reviewer.v1.audit annotation-csv \
  --csv "${CSV_PATH}" \
  --output "${DATA_DIR}/annotation_audit_2_1_1.json" \
  --split-output "${DATA_DIR}/split_2_1_1.json" \
  --train-evidence-count 2 --validation-evidence-count 1 --locked-test-evidence-count 1

OVERFIT_JOB_RAW=$(sbatch \
  --parsable --export=ALL --chdir="${PROJECT_ROOT}" \
  --output="${PROJECT_ROOT}/logs/reviewer-v1-overfit-%j.out" \
  --error="${PROJECT_ROOT}/logs/reviewer-v1-overfit-%j.err" \
  hpc/grpo_v3/human_preference_reviewer/v1/overfit_probe.sbatch)
OVERFIT_JOB=${OVERFIT_JOB_RAW%%;*}
OVERFIT_DIR=${OUTPUT_ROOT}/overfit_${OVERFIT_JOB}
echo "OVERFIT_JOB=${OVERFIT_JOB}"
```

```bash
sacct -j "${OVERFIT_JOB}" --format=JobID,JobName%24,State,ExitCode,Elapsed,MaxRSS
"${PYTHON}" - "${OVERFIT_DIR}/training_result.json" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert result["status"] == "passed"
assert result["head_parameter_delta_nonzero"]
assert result["lora_parameter_delta_nonzero"]
print("throughput=", result["throughput"])
print("validation_macro_f1_mean=", result["validation_macro_f1_mean"])
PY
```

Overfit Probe 只证明三 heads 与共享 LoRA 能收到梯度并降低受控样本的损失，不证明 unseen evidence 泛化。

## 9. Gate 4：正式 60/10 训练

只有前述 Gate 全部通过，才提交正式训练：

```bash
TRAIN_JOB_RAW=$(sbatch \
  --parsable --export=ALL --chdir="${PROJECT_ROOT}" \
  --output="${PROJECT_ROOT}/logs/reviewer-v1-train-%j.out" \
  --error="${PROJECT_ROOT}/logs/reviewer-v1-train-%j.err" \
  hpc/grpo_v3/human_preference_reviewer/v1/train.sbatch)
TRAIN_JOB=${TRAIN_JOB_RAW%%;*}
TRAIN_DIR=${OUTPUT_ROOT}/train_${TRAIN_JOB}
echo "TRAIN_JOB=${TRAIN_JOB}"
```

```bash
sacct -j "${TRAIN_JOB}" --format=JobID,JobName%24,State,ExitCode,Elapsed,MaxRSS
test -s "${TRAIN_DIR}/training_result.json"
test -s "${TRAIN_DIR}/checkpoint/reviewer_v1_config.json"
test -s "${TRAIN_DIR}/checkpoint/parameter_audit.json"
CHECKPOINT_DIR=${TRAIN_DIR}/checkpoint
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
```

`parameter_audit.json` 必须证明 trainable 参数只有三个 classification heads 与 blocks 34/35 的四个 LoRA projection；任何 unexpected trainable parameter 都是失败。

## 10. Gate 5：Validation

```bash
VALID_JOB_RAW=$(sbatch \
  --parsable \
  --export=ALL,CHECKPOINT_DIR="${CHECKPOINT_DIR}" \
  --chdir="${PROJECT_ROOT}" \
  --output="${PROJECT_ROOT}/logs/reviewer-v1-eval-%j.out" \
  --error="${PROJECT_ROOT}/logs/reviewer-v1-eval-%j.err" \
  hpc/grpo_v3/human_preference_reviewer/v1/evaluate.sbatch)
VALID_JOB=${VALID_JOB_RAW%%;*}
VALID_DIR=${OUTPUT_ROOT}/evaluate_${VALID_JOB}
echo "VALID_JOB=${VALID_JOB}"
```

```bash
sacct -j "${VALID_JOB}" --format=JobID,JobName%24,State,ExitCode,Elapsed,MaxRSS
"${PYTHON}" - "${VALID_DIR}/evaluation_result.json" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert result["status"] == "passed"
assert result["split"] == "validation"
print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
PY
```

分别报告 Evidence、Answerability、Formality 的 validation loss、accuracy、macro-F1、3×3 confusion matrix、每级 precision/recall、expected-score MAE 与 Spearman。Answerability 的等级 3 样本较少，必须同时报告 support，不只看 accuracy。

## 11. 简洁查错命令

把 JobID 与 mode 改成真实值：

```bash
JOB_ID=12345678
MODE=train

sacct -j "${JOB_ID}" \
  --format=JobID,JobName%24,State,ExitCode,Elapsed,MaxRSS

echo "===== stdout ====="
tail -n 120 "${PROJECT_ROOT}/logs/reviewer-v1-${MODE}-${JOB_ID}.out"

echo "===== stderr ====="
tail -n 160 "${PROJECT_ROOT}/logs/reviewer-v1-${MODE}-${JOB_ID}.err"
```

若路径不确定，再运行 `scontrol show job -dd "${JOB_ID}"` 查看 `WorkDir`、`Command`、`StdOut`、`StdErr`。不要先输出大量环境信息。

## 12. 结论边界

本轮能证明：

- 70 个 evidence 的 `60/10/0` 划分无重叠；
- 三个绝对评分 heads 与最后两层共享 LoRA 能联合训练；
- Reviewer 在 10 个 unseen validation evidence 上与人工标签的对齐程度。

本轮不能证明：

- 独立 locked-test 泛化；
- Overall ranking、pairwise preference 或 GRPO reward 有效；
- 当前超参数已经最优。

本地单元测试通过不能替代 H100 Gate；Slurm 状态 `COMPLETED` 也不能替代 `storage_preflight.json`、`structure_probe.json`、`training_result.json`、`parameter_audit.json` 和 `evaluation_result.json`。
