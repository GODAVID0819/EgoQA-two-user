# GRPO v3 新版 Gate 3：多信号人工审计、双 LoRA greedy 评估与 Torch 操作手册

> 远端执行状态（2026-07-20）：已在 `egoqa-ffmpeg-runtime` 环境修复 `ffmpeg` 缺失问题，并用原定参数（包括 `high_similarity_interval_threshold=0.82`）成功生成 32 条不同 evidence：
> `/scratch/xl6775/projects/EgoQA-two-user/outputs/grpo_v3/packet_build_32_seed42_ffmpeg_fixed/evidence_pruned_pairs.jsonl`。
> `wc -l` 已确认是 32。Hugging Face 未认证警告未阻止本次构建；无需降低 CLIP 阈值，也不要再上传或伪造 `selected_packets_pruned_gate4.jsonl`。当前从第 3 节初始化和验收开始，随后执行第 4 节的 20-train/8-eval 确定性拆分；分支 A 与分支 B 可在拆分完成后并行。

## 1. 本轮要回答什么

本轮不是直接扩大训练规模，而是用最低成本同时回答三个问题：

1. 8B reviewer 的 groundedness、answerability、speaker leakage、formality 与 shallow activity 信号是否值得信任；
2. Gate 2 与旧 Gate 3 LoRA 在同一固定集、原 repo greedy 解码下，谁的真实 QA 质量更好；
3. 将旧 Gate 3 的“同一个 evidence 重复 20 步”改为“20 个不同 evidence”，并把训练温度降到 `0.3` 后，reward 是否开始改善。

执行依赖如下：

```text
分支 A：旧 Gate 3 trace -> 导出 24 个多信号审计案例 -> 截取双视频 -> 人工逐项填写 -> 批准/不批准既定 reward 变更
分支 B：同一 8 条 held-out 集 -> Gate 2 greedy -> 旧 Gate 3 greedy -> 配对比较

分支 A 与 B 可以同时运行。
新版 Gate 3 必须等待分支 A 产生显式批准的 summary。
```

固定条件：

- policy：Qwen3-VL-2B-Instruct BF16 LoRA；
- reviewer：冻结 Qwen3-VL-8B-Instruct；
- 输入：两段有序原生 MP4；
- ms-swift：4.2.2；
- 像素预算：`50176`；
- greedy 评估：`do_sample=false`，每条只生成一次，不重试；
- 新版 GRPO：`temperature=0.3`、`top_p=1.0`、每组 4 completion。

## 2. 本地需要上传的文件

在 Windows PowerShell 中进入：

```powershell
cd C:\Users\20661\Desktop\Research\AR\multiuser\EgoQA-two-user
```

启动交互式 SFTP：

```powershell
sftp xl6775@torch-login-b-2
```

进入项目并上传：

```text
lcd C:/Users/20661/Desktop/Research/AR/multiuser/EgoQA-two-user
cd /scratch/xl6775/projects/EgoQA-two-user

put training/grpo_v3_groundedness_audit.py training/
put training/grpo_v3_gate3_dataset.py training/
put training/grpo_v3_greedy_eval.py training/
put training/grpo_v3_greedy_compare.py training/
put training/grpo_v3_repo_reward.py training/
put training/grpo_v3_summary.py training/

put hpc/grpo_v3_lora_greedy_eval.sbatch hpc/
put hpc/grpo_v3_ms_swift_gate3_v2.sbatch hpc/

put tests/training/test_grpo_v3_groundedness_audit.py tests/training/
put tests/training/test_grpo_v3_gate3_dataset.py tests/training/
put tests/training/test_grpo_v3_greedy_eval.py tests/training/
put tests/training/test_grpo_v3_greedy_compare.py tests/training/
put tests/training/test_grpo_v3_repo_reward.py tests/training/
put tests/training/test_grpo_v3_summary.py tests/training/
put tests/training/test_grpo_v3_slurm.py tests/training/
```

本轮需要至少 28 个不同 evidence。Torch 已经原生生成 32 条 packet，因此只上传上述源码、测试和 Slurm 脚本，**不再上传** `selected_packets_pruned_gate4.jsonl`。源码上传完成后执行 `bye` 退出 SFTP。

远端实际 evidence 文件固定为：

```text
/scratch/xl6775/projects/EgoQA-two-user/outputs/grpo_v3/packet_build_32_seed42_ffmpeg_fixed/evidence_pruned_pairs.jsonl
```

生成该文件时第一次得到 0 条的根因是当前 Python 运行环境找不到 `ffmpeg`，不是 manifest 为空或 `0.82` 阈值过严。修复后同一阈值一次生成 32 条。后续训练使用正常的 ms-swift 训练环境即可；只有重新构建 packet 或截取人工审计视频时才要求当前 shell 能找到 `ffmpeg`。

## 3. Torch 登录节点初始化与本地逻辑测试

SSH 登录后执行：

```bash
cd /scratch/${USER}/projects/EgoQA-two-user

export PROJECT_ROOT=/scratch/${USER}/projects/EgoQA-two-user
export GRPO_V3_ROOT=${PROJECT_ROOT}/outputs/grpo_v3
export TRAIN_ENV=/scratch/${USER}/envs/egoqa-ms-swift-v4.2.2-vllm024
export PYTHON=${TRAIN_ENV}/bin/python
export PACKET_ROOT=${GRPO_V3_ROOT}/packet_build_32_seed42_ffmpeg_fixed
export EVIDENCE_JSONL=${PACKET_ROOT}/evidence_pruned_pairs.jsonl

test -x "${PYTHON}"
test -s "${EVIDENCE_JSONL}"
test "$(wc -l < "${EVIDENCE_JSONL}")" -eq 32
```

先验收 32 条 packet 均为合法 JSON、`evidence_id` 唯一且每条包含两段视频：

```bash
${PYTHON} - "${EVIDENCE_JSONL}" <<'PY'
import json
import sys

path = sys.argv[1]
rows = []
with open(path, "r", encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"第 {line_number} 行不是合法 JSON: {exc}") from exc

ids = [row.get("evidence_id") for row in rows]
assert len(rows) == 32, f"期望 32 条，实际 {len(rows)} 条"
assert all(ids), "存在缺失 evidence_id 的 packet"
assert len(set(ids)) == 32, "evidence_id 不唯一"
assert all(len(row.get("clips") or []) == 2 for row in rows), "存在不是双视频的 packet"

print("packet_count       =", len(rows))
print("unique_evidence_id =", len(set(ids)))
print("two_clip_packets   =", sum(len(row.get("clips") or []) == 2 for row in rows))
print("PACKET_VALIDATION_PASSED")
PY
```

必须看到 `PACKET_VALIDATION_PASSED`。这只证明 32 条 evidence 的结构合同成立，不代表人工视频质量、reviewer 可靠性或 Gate 3 已通过。

先跑纯逻辑测试：

```bash
${PYTHON} -m unittest \
  tests.training.test_grpo_v3_groundedness_audit \
  tests.training.test_grpo_v3_gate3_dataset \
  tests.training.test_grpo_v3_greedy_eval \
  tests.training.test_grpo_v3_greedy_compare \
  tests.training.test_grpo_v3_repo_reward \
  tests.training.test_grpo_v3_summary \
  tests.training.test_grpo_v3_slurm -v
```

这里通过只说明本地逻辑合同正确，不代表 GPU 评估或新版 Gate 3 已通过。

## 4. 一次性构造 20 条训练集和 8 条固定评估集

执行：

```bash
export PACKET_ROOT=${GRPO_V3_ROOT}/packet_build_32_seed42_ffmpeg_fixed
export EVIDENCE_JSONL=${PACKET_ROOT}/evidence_pruned_pairs.jsonl
export DATA_DIR=${GRPO_V3_ROOT}/gate3_v2_data

${PYTHON} -m training.grpo_v3_gate3_dataset \
  --evidence "${EVIDENCE_JSONL}" \
  --output-dir "${DATA_DIR}" \
  --seed 42 \
  --train-count 20 \
  --eval-count 8
```

检查：

```bash
wc -l \
  "${DATA_DIR}/gate3_v2_train_native_video.jsonl" \
  "${DATA_DIR}/gate3_v2_eval_native_video.jsonl"

${PYTHON} -m json.tool "${DATA_DIR}/gate3_v2_split_manifest.json"
```

必须看到：

- train 为 20 行、20 个不同 evidence；
- eval 为 8 行、8 个不同 evidence；
- train/eval 无交叉；
- train 为 commonality 10、difference 10；
- eval 为 commonality 4、difference 4。

不要手动复制同一个 packet 凑足 28 条。

## 5. 分支 A：导出 24 个 groundedness 人工审计案例

先定位旧 Gate 3 trace。若 Torch 原始输出目录仍是 job `14194844`：

```bash
export OLD_GATE3_DIR=${GRPO_V3_ROOT}/gate3_14194844
export OLD_GATE3_TRACE=${OLD_GATE3_DIR}/reward_trace.jsonl
test -s "${OLD_GATE3_TRACE}"
```

如果路径不同，使用：

```bash
find "${GRPO_V3_ROOT}" -maxdepth 3 -type f -name reward_trace.jsonl -print
```

确认后导出 12 PASS＋12 FAIL：

```bash
export AUDIT_DIR=${GRPO_V3_ROOT}/groundedness_audit

${PYTHON} -m training.grpo_v3_groundedness_audit export \
  --trace "${OLD_GATE3_TRACE}" \
  --output-dir "${AUDIT_DIR}" \
  --pass-count 12 \
  --fail-count 12
```

导出物：

```bash
ls -lh "${AUDIT_DIR}"
```

应包括：

- `groundedness_audit_cases.jsonl`：完整机器可读案例，包含 groundedness、三类 answerability、formality、shallow activity 与 reward 信号；
- `groundedness_audit_review.csv`：需要人工逐项填写的多信号审计表；
- `groundedness_audit_guide_cn.md`：逐条 QA、角色、claim、各 reviewer 信号、answerability 条件证据和播放命令；
- `extract_audit_clips.sh`：根据 evidence timestamps 自动截取两段审计视频。

### 5.1 在 Torch 截取可下载的小视频

```bash
bash "${AUDIT_DIR}/extract_audit_clips.sh" "${AUDIT_DIR}/clips"

find "${AUDIT_DIR}/clips" -type f -name '*.mp4' | wc -l
tar -C "${AUDIT_DIR}" -czf "${AUDIT_DIR}/groundedness_audit_pack.tar.gz" \
  groundedness_audit_cases.jsonl \
  groundedness_audit_review.csv \
  groundedness_audit_guide_cn.md \
  clips
```

正常应有 48 个小视频，即每个案例两段。如果 `ffmpeg` 报某个时间窗口超出视频长度，先保留错误信息；不要把缺视频伪装为 reviewer FAIL。

### 5.2 下载审计包

Windows 中：

```powershell
cd C:\Users\20661\Desktop\Research\AR\multiuser\EgoQA-two-user
sftp xl6775@torch-login-b-2
```

交互式 SFTP：

```text
lcd C:/Users/20661/Desktop/Research/AR/multiuser/EgoQA-two-user/outputs/grpo_v3
cd /scratch/xl6775/projects/EgoQA-two-user/outputs/grpo_v3/groundedness_audit
get groundedness_audit_pack.tar.gz
bye
```

解压后，打开 `groundedness_audit_guide_cn.md` 和 `groundedness_audit_review.csv`。

## 6. 人工如何判断每个多信号案例

每条案例按以下顺序检查，建议每条 2–4 分钟：

1. 阅读问题、五个选项、正确项和 answer；
2. 阅读两个 `evidence claims`，明确每个用户声称看到了什么；
3. 同时打开该案例的 `__u1.mp4` 与 `__u2.mp4`；
4. 独立检查 groundedness：claim 与正确答案是否真的可见、被唯一支持；
5. 独立检查 combined answerability：两段视频合并后能否唯一作答；
6. 独立检查 speaker leakage：提问者自己的视频是否已经足以答对；
7. 独立检查 provider-only answerability：证据提供者是否能单独答对；
8. 不依赖视频结论，检查 QA formality 与 shallow activity；
9. 最后才阅读 reviewer 各项结论、条件选择与证据，判断是否一致。

各信号必须分别填写。不要因为 groundedness FAIL 就自动把 answerability、formality 全部填成 FAIL；也不要因为 provider 单独能回答就判定 answerability gate 失败。JSON format 是机器结构信号，不要求人工通过视频判断。

### 6.1 CSV 字段怎么填

| 字段 | 允许值 | 含义 |
|---|---|---|
| `human_groundedness` | `PASS` / `FAIL` / `UNCERTAIN` | 视频是否支持 claims 与正确答案 |
| `human_combined_answerability` | `PASS` / `FAIL` / `UNCERTAIN` | 两段视频合并后是否唯一支持正确项 |
| `human_speaker_leakage` | `LEAK` / `NO_LEAK` / `UNCERTAIN` | speaker 单独是否已经能够答对 |
| `human_provider_answerability` | `ANSWERABLE` / `NOT_ANSWERABLE` / `UNCERTAIN` | provider 单独是否能够答对；可答不等于 gate 失败 |
| `human_qa_formality` | `PASS` / `FAIL` / `UNCERTAIN` | 问题是否自然、五选项是否规范且唯一 |
| `human_shallow_activity` | `PASS` / `FAIL` / `UNCERTAIN` | PASS 表示不是浅层问题，FAIL 表示问题过浅 |
| `claim_visible` | `yes` / `no` / `partial` | 两用户 claims 是否清楚可见 |
| `answer_supported` | `yes` / `no` / `uncertain` | 正确项是否被视频支持 |
| `reviewer_agreement` | `yes` / `no` | 人工最终结论是否同 reviewer 一致 |
| `notes` | 中文自由文本 | 写明哪一段视频、哪个时刻支持或反驳 |

groundedness 判定建议：

- `PASS`：关键 claim 清楚可见，且正确答案与视频一致；
- `FAIL`：关键 claim 不可见、与视频冲突，或答案依赖无法验证的空间/因果推断；
- `UNCERTAIN`：画面遮挡、过短、模糊，人工无法可靠判断。

一个好的 `notes` 例子：

```text
u2 约 2–4 秒能看到多个花瓶，但无法判断它们位于桌子的左侧；因此 reviewer FAIL 合理。
```

不要只写“同意”或“不同意”，否则之后无法复核依据。

人工 answerability gate 由脚本推导：

\[
\text{Human Answerability Gate}
=
\text{Combined PASS}
\land
\text{Speaker NO\_LEAK}
\]

新增信号用于诊断和人机一致性分析，不会自动修改 reward 权重，也不会自动放行 Gate 3 或 Gate 4。`--approve-weight-change` 仍是操作者对既定 groundedness 权重变更的显式批准。

## 7. 分支 B：同时提交 Gate 2 与旧 Gate 3 greedy 评估

这一步可以在你人工看视频时运行。

### 7.1 定位两个 adapter

Gate 2：

```bash
export GATE2_DIR=$(head -n 1 "${GRPO_V3_ROOT}/latest_gate2_output.txt")
export GATE2_ADAPTER=$(${PYTHON} -c 'import json,sys; from pathlib import Path; print(json.load(open(Path(sys.argv[1])/"run_manifest.json"))["adapter_dir"])' "${GATE2_DIR}")
test -d "${GATE2_ADAPTER}"
echo "${GATE2_ADAPTER}"
```

旧 Gate 3：

```bash
export OLD_GATE3_DIR=${GRPO_V3_ROOT}/gate3_14194844
export OLD_GATE3_ADAPTER=$(${PYTHON} -c 'from pathlib import Path; import sys; from training.grpo_v3_adapter_reload import discover_adapter_dir; print(discover_adapter_dir(Path(sys.argv[1])))' "${OLD_GATE3_DIR}")
test -d "${OLD_GATE3_ADAPTER}"
echo "${OLD_GATE3_ADAPTER}"
```

### 7.2 提交两个相互独立的作业

```bash
GATE2_EVAL_JOB=$(sbatch --parsable \
  --export=ALL,DATA_DIR="${DATA_DIR}",ADAPTER_LABEL=gate2,ADAPTER_DIR="${GATE2_ADAPTER}" \
  hpc/grpo_v3_lora_greedy_eval.sbatch)

OLD_GATE3_EVAL_JOB=$(sbatch --parsable \
  --export=ALL,DATA_DIR="${DATA_DIR}",ADAPTER_LABEL=gate3_old,ADAPTER_DIR="${OLD_GATE3_ADAPTER}" \
  hpc/grpo_v3_lora_greedy_eval.sbatch)

echo "GATE2_EVAL_JOB=${GATE2_EVAL_JOB}"
echo "OLD_GATE3_EVAL_JOB=${OLD_GATE3_EVAL_JOB}"
```

这两个作业没有依赖关系。集群资源允许时会并行运行；资源不足时 Slurm 会排队，但结果仍可严格配对。

查看状态：

```bash
squeue -j "${GATE2_EVAL_JOB},${OLD_GATE3_EVAL_JOB}"
sacct -j "${GATE2_EVAL_JOB},${OLD_GATE3_EVAL_JOB}" --format=JobID,State,Elapsed,ExitCode,MaxRSS
```

日志：

```bash
tail -n 80 "logs/grpo-v3-greedy-${GATE2_EVAL_JOB}.out"
tail -n 80 "logs/grpo-v3-greedy-${GATE2_EVAL_JOB}.err"
tail -n 80 "logs/grpo-v3-greedy-${OLD_GATE3_EVAL_JOB}.out"
tail -n 80 "logs/grpo-v3-greedy-${OLD_GATE3_EVAL_JOB}.err"
```

必须看到：

```text
greedy_eval_status=passed
```

### 7.3 配对比较两个 adapter

```bash
export GATE2_EVAL_DIR=${GRPO_V3_ROOT}/greedy_gate2_${GATE2_EVAL_JOB}
export OLD_GATE3_EVAL_DIR=${GRPO_V3_ROOT}/greedy_gate3_old_${OLD_GATE3_EVAL_JOB}
export COMPARE_DIR=${GRPO_V3_ROOT}/greedy_compare_gate2_gate3_old

mkdir -p "${COMPARE_DIR}"

${PYTHON} -m training.grpo_v3_greedy_compare \
  --run gate2="${GATE2_EVAL_DIR}/greedy_results.jsonl" \
  --run gate3_old="${OLD_GATE3_EVAL_DIR}/greedy_results.jsonl" \
  --baseline gate2 \
  --output-json "${COMPARE_DIR}/comparison.json" \
  --output-md "${COMPARE_DIR}/comparison_cn.md"
```

阅读：

```bash
${PYTHON} -m json.tool "${COMPARE_DIR}/comparison.json" | less
sed -n '1,240p' "${COMPARE_DIR}/comparison_cn.md"
```

注意：8 条 held-out evidence 是快速 paired diagnostic，不是论文级统计显著性实验。它足以判断“旧 Gate 3 是否在相同 greedy 条件下普遍优于或劣于 Gate 2”。

## 8. 上传人工 CSV 并生成审计 summary

人工填写完后上传 CSV：

```powershell
sftp xl6775@torch-login-b-2
```

交互式 SFTP：

```text
lcd C:/Users/20661/Desktop/Research/AR/multiuser/EgoQA-two-user/outputs/grpo_v3/groundedness_audit
cd /scratch/xl6775/projects/EgoQA-two-user/outputs/grpo_v3/groundedness_audit
put groundedness_audit_review.csv
bye
```

先生成“不批准”的统计，分别查看各 reviewer 信号是否可信：

```bash
cd /scratch/${USER}/projects/EgoQA-two-user

${PYTHON} -m training.grpo_v3_groundedness_audit summarize \
  --review-csv "${AUDIT_DIR}/groundedness_audit_review.csv" \
  --output "${AUDIT_DIR}/groundedness_audit_summary_unapproved.json"

${PYTHON} -m json.tool "${AUDIT_DIR}/groundedness_audit_summary_unapproved.json"
```

建议同时满足以下条件才批准：

- 已完成至少 20 个；
- reviewer PASS 与 FAIL 子集各至少 8 个；
- 总体 agreement rate 不低于 0.80；
- PASS 和 FAIL 任一子集都没有明显系统性失真；
- UNCERTAIN 比例不高于约 0.15；
- 你阅读过所有 disagreement 的 notes。

同时检查 `signals`、`human_answerability_gate`、`cross_signal` 和 `invalid_values`。新增信号允许暂时部分填写，但其 `completed` 必须按实际数量解释，不能把空值或非法值当作人工 PASS。

如果满足，由你显式生成批准文件：

```bash
${PYTHON} -m training.grpo_v3_groundedness_audit summarize \
  --review-csv "${AUDIT_DIR}/groundedness_audit_review.csv" \
  --output "${AUDIT_DIR}/groundedness_audit_summary.json" \
  --approve-weight-change

${PYTHON} -m json.tool "${AUDIT_DIR}/groundedness_audit_summary.json"
```

必须确认：

```text
"approved_for_weight_change": true
```

如果 reviewer 不可信，不要添加 `--approve-weight-change`，也不要提交新版 Gate 3。应先修改 reviewer prompt，再重新审计。

## 9. 提交新版 Gate 3

只有第 8 节已经生成批准文件时执行：

```bash
export AUDIT_SUMMARY=${AUDIT_DIR}/groundedness_audit_summary.json
test -s "${AUDIT_SUMMARY}"

GATE3_V2_JOB=$(sbatch --parsable \
  --export=ALL,DATA_DIR="${DATA_DIR}",AUDIT_SUMMARY="${AUDIT_SUMMARY}",GATE2_DIR="${GATE2_DIR}" \
  hpc/grpo_v3_ms_swift_gate3_v2.sbatch)

echo "GATE3_V2_JOB=${GATE3_V2_JOB}"
```

监控：

```bash
squeue -j "${GATE3_V2_JOB}"
sacct -j "${GATE3_V2_JOB}" --format=JobID,State,Elapsed,ExitCode,MaxRSS
tail -f "logs/grpo-v3-gate3-v2-${GATE3_V2_JOB}.out"
```

作业参数固定为：

- 从 passed Gate 2 adapter 开始；
- 20 个不同 evidence；
- commonality/difference 各 10；
- `temperature=0.3`；
- `top_p=1.0`；
- `learning_rate=1e-5`；
- `lr_scheduler_type=constant`；
- `beta=0.0`；
- `VIDEO_MAX_PIXELS=50176`；
- `ground_answer_gap_v1`：groundedness 与 combined answerability 正负间隔增大；
- 三层 JSON format reward 保持不变。

### 9.1 新版 Gate 3 通过条件

```bash
export GATE3_V2_DIR=${GRPO_V3_ROOT}/gate3_v2_${GATE3_V2_JOB}

${PYTHON} -m json.tool "${GATE3_V2_DIR}/gate3_result.json"
${PYTHON} -m json.tool "${GATE3_V2_DIR}/convergence_metrics.json"
${PYTHON} -m json.tool "${GATE3_V2_DIR}/run_manifest.json"
```

需要同时满足：

- `gate3_result.status == "passed"`；
- 80 个 reward 全部有限；
- masked 为 0；
- 20 组均恰好 4 candidates；
- 多数组 reward std 大于 0；
- late reward mean 高于 early reward mean；
- adapter reload 成功；
- manifest 记录 `content_reward_revision=ground_answer_gap_v1`；
- manifest 记录人工审计 summary 路径和 SHA256。

Gate 3 v2 没通过时仍不能进入 Gate 4。

## 10. 用同一固定集评估新版 Gate 3

仅在新版 Gate 3 已产生完整 adapter 后执行：

```bash
export GATE3_V2_ADAPTER=$(${PYTHON} -c 'import json,sys; from pathlib import Path; print(json.load(open(Path(sys.argv[1])/"run_manifest.json"))["adapter_dir"])' "${GATE3_V2_DIR}")
test -d "${GATE3_V2_ADAPTER}"

NEW_GATE3_EVAL_JOB=$(sbatch --parsable \
  --export=ALL,DATA_DIR="${DATA_DIR}",ADAPTER_LABEL=gate3_v2,ADAPTER_DIR="${GATE3_V2_ADAPTER}" \
  hpc/grpo_v3_lora_greedy_eval.sbatch)

echo "NEW_GATE3_EVAL_JOB=${NEW_GATE3_EVAL_JOB}"
```

完成后做三方配对：

```bash
export NEW_GATE3_EVAL_DIR=${GRPO_V3_ROOT}/greedy_gate3_v2_${NEW_GATE3_EVAL_JOB}
export THREE_WAY_DIR=${GRPO_V3_ROOT}/greedy_compare_three_way
mkdir -p "${THREE_WAY_DIR}"

${PYTHON} -m training.grpo_v3_greedy_compare \
  --run gate2="${GATE2_EVAL_DIR}/greedy_results.jsonl" \
  --run gate3_old="${OLD_GATE3_EVAL_DIR}/greedy_results.jsonl" \
  --run gate3_v2="${NEW_GATE3_EVAL_DIR}/greedy_results.jsonl" \
  --baseline gate2 \
  --output-json "${THREE_WAY_DIR}/comparison.json" \
  --output-md "${THREE_WAY_DIR}/comparison_cn.md"
```

重点查看：

- 新版 Gate 3 相对 Gate 2 的 paired wins/losses；
- groundedness PASS 数是否回升；
- combined answerability 是否回升；
- raw-valid/repaired/unrecoverable 是否恶化；
- commonality 与 difference 是否出现单侧失败。

## 11. 常见失败如何处理

### 11.1 人工审计导出不足 12 个 FAIL

工具会直接报错。不要重复抽同一个案例。可以将目标改为实际可用 FAIL 数，但仍需确保最终已完成 PASS/FAIL 各至少 8 个；否则不能批准扩大 reward。

### 11.2 greedy eval 出现 masked reward

作业应立即失败。检查：

- 两段视频是否存在；
- evidence_id 是否错位；
- reviewer 是否超时；
- reviewer 是否返回缺失信号；
- reward 是否 NaN/Inf。

这些不是 candidate JSON 格式失败，不能转换成 `-3.0`。

### 11.3 completion JSON 不可修复

单条返回有限 `-3.0`，评估继续处理其他样本；trace 中应记录 `unrecoverable`，且不调用 reviewer。

### 11.4 reviewer OOM

先重提并降低并发：

```bash
sbatch --export=ALL,REVIEW_MAX_NUM_SEQS=1,DATA_DIR="${DATA_DIR}",ADAPTER_LABEL=gate2,ADAPTER_DIR="${GATE2_ADAPTER}" \
  hpc/grpo_v3_lora_greedy_eval.sbatch
```

不要先提高分辨率。检查 `gpu_metrics.csv` 和 reviewer log 中的 `out_of_memory`、`CUDA error`、`Traceback (most recent call last)`。

### 11.5 新版 Gate 3 reward 仍下降

先不要进入 Gate 4，也不要立即提高 LR。依次检查：

1. 三方 greedy 对比是否也下降；
2. 下降主要来自 groundedness、answerability 还是 format；
3. commonality/difference 哪类下降；
4. 人工 audit disagreement 是否集中在同一种空间或动作判断；
5. 新版 20 evidence 是否包含异常 packet。

只有当 greedy 行为改善但采样训练曲线仍下降时，才优先继续调 temperature 或 group/step 设计。

## 12. 需要下载回本地的结果

建议最终下载：

```text
outputs/grpo_v3/groundedness_audit/
outputs/grpo_v3/gate3_v2_data/gate3_v2_split_manifest.json
outputs/grpo_v3/greedy_gate2_<jobid>/
outputs/grpo_v3/greedy_gate3_old_<jobid>/
outputs/grpo_v3/greedy_compare_gate2_gate3_old/
outputs/grpo_v3/gate3_v2_<jobid>/
outputs/grpo_v3/greedy_gate3_v2_<jobid>/
outputs/grpo_v3/greedy_compare_three_way/
logs/grpo-v3-greedy-<jobid>.out
logs/grpo-v3-greedy-<jobid>.err
logs/grpo-v3-gate3-v2-<jobid>.out
logs/grpo-v3-gate3-v2-<jobid>.err
```

本地分析时必须区分：

- 人工 reviewer 校准结果；
- 固定集 greedy adapter 对比；
- GRPO sampling 训练曲线；
- Gate 3 正式验收结果。

任何一类结果都不能单独替代另外三类。
