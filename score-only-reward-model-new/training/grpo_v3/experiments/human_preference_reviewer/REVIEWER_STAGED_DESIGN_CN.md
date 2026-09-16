# Reviewer 分阶段设计（binary_reviewer_v2）

## 目标

训练 Qwen3-VL-8B reviewer，使 Evidence Quality、Answerability、QA Formality 三个 judge 与部署的 pass/fail 决策一致。目录和 Python module 仍保留 `v1` 名称以兼容既有入口；checkpoint 合同版本已经提升为 `binary_reviewer_v2`。

## 标签

人类标注 CSV 继续保存 1–3 原始分数，训练 view 为：

```text
1 -> fail (0)
2 -> pass (1)
3 -> pass (1)
```

理由是三个 rubric 中 1 都表示不满足 gate，而 2/3 都表示可接受或更强。旧 ordinal 映射已归档在 `ARCHIVED_ORDINAL_REVIEWER_V1.md`。

## 模型

```text
two synchronized videos + candidate QA
-> Qwen3-VL shared representation
-> Evidence / Answerability / Formality independent Linear(hidden_dim, 2) heads
-> logits ordered as [fail, pass]
```

Stage 0 只实例化 Evidence head 且不注入 LoRA。Stage 1 在相同单 head 上启用最后两个 shared language blocks 的 q/v LoRA。Stage 2 实例化三个 heads，并让它们共享同一组 LoRA 参数。

## Loss

对每个 active head，类别权重只从 training split 统计：

```text
weight[c] = N / (2 * N_c)
head_loss = mean(cross_entropy(logits, binary_label, weight=[w_fail, w_pass], reduction=none))
total_loss = mean(active head losses)
```

固定按样本数取均值，避免 batch size 1 时默认 weighted-mean 把类别权重抵消。不再使用 ordinal BCE，也不再使用 Evidence/Answerability/Formality 的 `0.4/0.4/0.2` priority。class counts、class weights、映射和 loss 合同写入 checkpoint。

## 参数边界

- Stage 0：只允许 `evidence_head.*` 更新。
- Stage 1：允许 `evidence_head.*` 和 layers 34–35 的 `q_proj`/`v_proj` LoRA 更新。
- Stage 2：允许三个 heads 和同一组 LoRA 更新。
- vision tower、原始 backbone weights、LM head 与 base bias 均冻结。

## 评估

报告每个 head 的 accuracy、balanced accuracy、macro-F1、AUROC、2×2 confusion matrix，以及 fail/pass 两类的 precision/recall/F1/support。validation 的 early-stopping CE 使用 training-split class weights；外部 holdout 的 base/trained 对比使用相同的 unweighted CE，避免两种 checkpoint 因 loss weighting 不同而不可比。

## 部署关系

训练 head 本身不生成 JSON、`reason` 或 `fix`。它通过共享 LoRA 对部署推理产生间接影响。部署 prompt 的输出顺序单独固定为 `verdict`、`reason`、`fix`，并要求 fail 时给出具体原因与修复，pass 时两个字段为 `null`。
