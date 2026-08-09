# Reviewer 70-Packet 数据与 60/10 划分设计

## 1. 目标与评估边界

Reviewer v1 的正式数据源更新为 `rlhf_candidate_scores_merged_70_packets.csv`。70 个 `evidence_id` 全部参与 evidence-level 划分：60 个用于训练，10 个用于 validation，当前不创建 locked test。

Validation 可用于训练过程监控、checkpoint 选择和本轮 Reviewer v1 的开发评估。因此它不是无偏最终测试集，后续研究结论需要另外收集从未用于模型选择的 evidence 作为 locked test。

## 2. 已验证的数据事实

新 CSV 的 SHA-256 为：

```text
32679019FD7C665A0632E9885405BDF13C77B51386EFC56E7B29B24192210CD7
```

审计结果：

- 420 行、70 个唯一 `evidence_id`；
- 每个 evidence 恰好 6 个 candidate；
- 420 行全部为 `annotation_status=completed`；
- candidate ID 全部唯一；
- 每个 evidence 的 `display_order` 恰好为 1–6；
- `packet_skipped=false`、`evidence_reviewed=true`；
- 三个评分列都只包含 1、2、3；
- 共引用 140 个唯一视频：Day 1 为 40 个、Day 5 为 84 个、Day 6 为 16 个。

全量标签分布为：

| 字段 | Level 1 | Level 2 | Level 3 |
|---|---:|---:|---:|
| Evidence Quality | 152 | 81 | 187 |
| Answerability | 226 | 178 | 16 |
| QA Formality | 99 | 204 | 117 |

固定 seed 42 的 60/10 划分中，train 和 validation 的三个字段都覆盖 1、2、3。Validation 的 support 分别为 Evidence `20/12/28`、Answerability `27/26/7`、Formality `6/35/19`。

## 3. CSV 兼容合同

新 CSV 继续使用已有的三个源评分列：

- `evidence_grounding_score -> evidence_quality`
- `answerability_score -> answerability`
- `formality_score -> qa_formality`

新 CSV 没有 `fea_total_score`。该列属于可由三个评分求和得到的冗余校验字段，因此 parser 改为：存在时必须等于三个评分之和；缺失时不报错，也不人工写回原始 CSV。

新 CSV 没有 `aggregate_rank`。`overall_rank` 继续保留为可选字段并解析为 `None`。Reviewer v1 不训练 Overall Utility 或 ranking loss，因此不影响本阶段训练；未来 Reviewer v2 若使用 ranking supervision，必须换用包含明确 overall ranking 的数据源，不能从 `display_order` 推断 ranking。

## 4. Split manifest 合同

默认计数更新为：

```text
train_evidence_count=60
validation_evidence_count=10
locked_test_evidence_count=0
```

Manifest 文件名更新为 `split_60_10.json`，仍包含：

- `train_evidence_ids`：60 个；
- `validation_evidence_ids`：10 个；
- `locked_test_evidence_ids`：空列表；
- `reserve_evidence_ids`：空列表；
- CSV SHA、seed、selection attempt 和各 split label support。

Split 规则：

- train 和 validation count 必须是正整数；
- locked test count 可以为 0，但不能为负数；
- train 和 validation 必须非空；
- locked test 在 expected count 为 0 时必须为空；
- 所有非空 split 按 `evidence_id` 严格互斥；
- full-class-support 检查只覆盖非空 split；
- seed 42 必须生成三个字段均有完整 1/2/3 support 的 60/10 manifest。

请求评估空的 locked-test split 时，程序应给出明确错误，不能返回空指标或把 validation 静默当作 test。

## 5. 训练与评估合同

Stage 0/1/2 的模型架构和 loss 不因数据划分改变。正式 Stage 2 仍使用三个 heads 与最后两个 shared blocks 的 q/v LoRA。

训练作业显式传入 60/10/0 计数并读取 `split_60_10.json`。训练结束保存 validation 指标和 checkpoint。独立 evaluation 作业本轮只允许 `EVAL_SPLIT=validation`；Runbook 删除正式 locked-test 执行步骤，并写明需要未来独立数据后才能恢复。

## 6. 媒体与集群路径

默认 CSV 文件名更新为 `rlhf_candidate_scores_merged_70_packets.csv`。Runbook 通过单文件 SFTP 将其上传到独立 Reviewer 仓库：

```text
/scratch/xl6775/projects/EgoQA-two-user-reviewer-v1/data_RLHF/reviewer_v1/
```

零 GPU Gate 先从 CSV 生成 140 条 required media 清单，再检查或下载 Day 1/5/6 视频，最后生成新的 `media_map.json`。旧 CSV 生成的 media map 不得复用为新数据已覆盖的证据。

所有 Reviewer Slurm 默认根、日志路径和 Runbook 提交路径统一到 `/scratch/xl6775/projects/EgoQA-two-user-reviewer-v1`，避免再次回落到 `EgoQA-two-user-grpo-clean`。

## 7. 需要修改的文件

核心合同：

- `training/grpo_v3/experiments/human_preference_reviewer/v1/data.py`
- `training/grpo_v3/experiments/human_preference_reviewer/v1/config.py`
- `training/grpo_v3/experiments/human_preference_reviewer/v1/audit.py`
- `training/grpo_v3/experiments/human_preference_reviewer/v1/train.py`

集群入口：

- `hpc/grpo_v3/human_preference_reviewer/v1/common.sh`
- `hpc/grpo_v3/human_preference_reviewer/v1/*.sbatch`
- 必要的 Stage 0 默认路径文件

文档：

- `training/grpo_v3/experiments/human_preference_reviewer/README_CN.md`
- `training/grpo_v3/experiments/human_preference_reviewer/REVIEWER_STAGED_DESIGN_CN.md`
- `training/grpo_v3/experiments/human_preference_reviewer/TORCH_RUNBOOK_STAGE0_CN.md`
- `training/grpo_v3/experiments/human_preference_reviewer/TORCH_RUNBOOK_V1_CN.md`

测试：

- data、config、audit、train、Slurm 和 Runbook 对应测试。

## 8. 验收 Gate

零 GPU 验收必须证明：

1. 新 CSV audit 为 420 行、70 evidence、420 eligible candidates；
2. CSV SHA 与设计固定值一致；
3. `split_60_10.json` 精确包含 60/10/0/0 evidence；
4. 两个非空 split 的三个字段都有 1/2/3 support；
5. train、validation、locked-test、reserve 无 overlap；
6. 缺少 `fea_total_score` 不再导致拒绝，存在但错误时仍拒绝；
7. 请求空 locked test 明确失败；
8. 140 个视频全部存在且 media map 与新 CSV 一致；
9. Reviewer 完整单元测试通过；
10. Runbook 不再包含旧 CSV、40/10/10 正式合同或本轮 locked-test 执行命令。

## 9. 非目标

本次不修改三个评分定义、模型 head、LoRA target、loss 权重或 Overall Utility 设计；不从 display order 构造 overall ranking；不把 validation 结果描述为独立最终测试结果；不把新 CSV 本身提交到 Git 仓库。
