# Binary Reviewer v2 training contract

当前合同把每个旧人类 1–3 分数确定性映射成一个 deployment-aligned binary decision：

```json
{
  "contract_version": "binary_reviewer_v2",
  "supervision_type": "class_weighted_binary_decisions",
  "human_score_to_binary_label": {
    "1": 0,
    "2": 1,
    "3": 1
  },
  "binary_label_names": {
    "0": "fail",
    "1": "pass"
  },
  "head_type": "two_logit_binary_classifier",
  "class_logit_order": ["fail", "pass"],
  "loss_function": "cross_entropy",
  "class_weighting": "balanced_inverse_frequency_from_training_split_only",
  "class_weight_reduction": "sum_weighted_losses_divided_by_sample_count",
  "head_loss_aggregation": "equal_mean"
}
```

对于每个 active judge head 和训练 split，若 `N` 为样本总数、`N_c` 为类别 `c` 的样本数，则：

```text
weight[c] = N / (2 * N_c)
```

类别计数与权重只从 training split 计算，并写入 checkpoint。validation/test 不参与权重估计。weighted CE 先保留逐样本 loss，再以样本数为固定分母取均值；这样在当前每步一个 candidate 时类别权重不会被 PyTorch 的默认 weighted-mean 分母抵消。三个 judge head 的 class-weighted CE 等权平均；不再有 `0.4/0.4/0.2` task priority。

训练期间的 validation loss 使用 training-split class weights。最终外部 holdout 报告使用 unweighted CE，使 base-untrained 与 trained checkpoint 的 loss 可直接比较；同时报告 accuracy、balanced accuracy、macro-F1、AUROC、2×2 confusion matrix，以及 fail/pass 两类的 precision/recall/F1/support。

旧 CSV 的三个分数字段仍保留为数据来源：

```json
[
  "evidence_grounding_score",
  "answerability_score",
  "formality_score"
]
```

`aggregate_rank` 与 `fea_total_score` 继续被忽略。原始数据没有被删除或覆写。
