# Archived ordinal Reviewer v1 contract

本文件保留二分类迁移前的训练合同，供回滚和审计。它不是当前 active contract。原始人类标注 CSV 与既有 QA 样本均未删除。

## 旧标签与 loss

旧模型为每个 judge 训练 1–3 ordinal score：

```json
{
  "contract_version": "score_only_ordinal_reviewer_v1",
  "supervision_type": "cumulative_ordinal_scores",
  "ordinal_target_mapping": {
    "1": [0, 0],
    "2": [1, 0],
    "3": [1, 1]
  },
  "head_type": "shared_score_ordered_thresholds",
  "num_score_levels": 3,
  "num_ordinal_thresholds": 2,
  "loss_weights": {
    "evidence_quality": 0.4,
    "answerability": 0.4,
    "qa_formality": 0.2
  },
  "loss_aggregation": "normalized_weighted_sum"
}
```

每个 head 输出 `P(score > 1)` 与 `P(score > 2)`，使用 cumulative binary cross-entropy。旧 checkpoint 文件名为 `ordinal_heads.pt`；当前代码会拒绝加载这种旧合同，避免把旧 head 误当成二分类 head。

## 旧 manifest 快照

迁移前的两个 manifest 已原样复制到：

- `archive/ordinal_v1_manifests/day1_7_merged_70_train_60_validation_10.json`
- `archive/ordinal_v1_manifests/day5_7_first_50_train_45_validation_5.json`

当前 active manifest 仍使用完全相同的 evidence split，只把 label-support 与合同元数据转换成 `1 → fail`、`2/3 → pass`。

## 旧模型输入提示语

```text
Do not generate an explanation; return hidden states for the ordinal scoring heads.
```

当前分类-head 训练仍不生成 `reason` 或 `fix`；这些文本由部署时的生成式 judge prompt 产生。二分类 CE 会通过共享 LoRA 参数间接影响部署推理，但不是对后续文本 token 的直接监督。
