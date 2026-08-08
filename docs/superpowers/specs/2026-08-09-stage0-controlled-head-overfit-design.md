# Stage 0B 受控单 Head 过拟合验证设计

## 目标

Stage 0B 只验证冻结 Qwen3-VL backbone 后，Evidence Quality 三分类 head 能否在一个固定小集合上被监督训练。它不评价 unseen evidence 泛化，也不启用 LoRA、Answerability、Formality 或 Overall Utility。

## 当前问题

现有 Overfit Gate 用单样本训练 step 的即时 loss，以及每个 candidate 第一次和最后一次出现时的 loss 判定通过。这两个观测点处于不同模型状态，不能构成统一的训练前/训练后比较。现有 validation 只有一个 evidence、缺少 Level 2，并且全预测 Level 3 仍可得到 66.7% accuracy，因此不能证明 head 可学习。

## 受控 Probe 合同

- 固定选择 4 个 train evidence，共 24 个 candidate，称为 `overfit probe set`。
- Probe set 必须覆盖 Evidence Quality 的 1、2、3 三类；这是诊断集合，不用于泛化结论。
- 随机初始化 head 后、任何 optimizer step 前，对完整 probe set 运行一次统一 evaluation，记为 `pre_train_metrics`。
- 每个 epoch 结束后对相同 probe set 运行统一 evaluation，写入 `epoch_probe_metrics`。
- 训练完成后再次对相同 probe set运行统一 evaluation，记为 `post_train_metrics`。
- 保存每个 candidate 的训练前/训练后 CE loss、label、prediction 和 probabilities，以便计算样本级改善比例。

## 核心指标

训练前后平均交叉熵分别为：

\[
L_{\mathrm{pre}}=\frac{1}{N}\sum_{i=1}^{N}\operatorname{CE}(z_i,y_i),
\qquad
L_{\mathrm{post}}=\frac{1}{N}\sum_{i=1}^{N}\operatorname{CE}(z'_i,y_i)
\]

相对下降比例为：

\[
r_{\mathrm{loss}}=\frac{L_{\mathrm{pre}}-L_{\mathrm{post}}}{L_{\mathrm{pre}}}
\]

输出还包括 accuracy 增量、预测类别数、样本级 loss 改善数量和比例。

## Gate 判定

Stage 0B 只有同时满足以下条件才通过：

- pre/post loss 均为有限数；
- `post_train_loss < pre_train_loss`；
- 相对 loss 至少下降 30%；
- 至少 80% candidate 的 post loss 小于 pre loss；
- post accuracy 至少比 pre accuracy 提高 20 个百分点；
- post prediction 至少覆盖两个等级，禁止单类塌缩；
- Evidence head 参数发生更新，backbone 保持冻结；
- checkpoint reload 验证通过。

阈值写入训练结果的 Gate contract，不作为可调训练超参数散落在 sbatch 中。首轮使用 4 evidence、20 epochs、最多 480 candidate steps；若 Gate 失败，先分析完整 pre/post 与 epoch 曲线，不同时修改多个训练因素。

## 输出合同

`training_result.json` 新增：

- `probe_evidence_ids`
- `probe_label_support`
- `pre_train_metrics`
- `post_train_metrics`
- `epoch_probe_metrics`
- `per_candidate_pre_post`
- `controlled_overfit_gate`

保留 `history` 和 `repeated_candidate_loss` 作为辅助诊断，但它们不再决定 Gate。

## 修改边界

只修改训练结果采集、Stage 0 Overfit sbatch、Stage 0 Runbook 和对应单元测试。不改模型结构、LoRA、Stage 1/2 配置、正式 40/10/10 划分或 locked test。

## 测试策略

- 使用纯 Python 合成 evaluation snapshot 测试 Gate 计算，不加载 8B 模型。
- 验证显著 loss 下降、accuracy 提升且预测非塌缩时通过。
- 验证全预测同一等级时拒绝，即便 accuracy 因类别不平衡看似较高。
- 验证 candidate 集合或顺序不一致时拒绝 pre/post 比较。
- 静态测试确保 sbatch 使用 4-evidence probe、20 epochs，并由 `controlled_overfit_gate.passed` 决定验收。
- 真实视频和 Qwen3-VL 的 pre/post 行为由单 H100 Stage 0B 作业验证。
