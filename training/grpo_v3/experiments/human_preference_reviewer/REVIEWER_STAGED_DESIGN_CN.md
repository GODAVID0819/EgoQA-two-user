# 多模态 Reviewer 分阶段训练设计

## 1. 目标

本分支保留完整 Reviewer v1 训练框架，但按风险逐步启用能力：

1. Stage 0：单个 Evidence Quality head，backbone 完全冻结；
2. Stage 1：单个 Evidence Quality head，启用最后两个 shared blocks 的 LoRA；
3. Stage 2：三个独立 absolute heads，共享最后两个 blocks 的 LoRA；
4. Stage 3：增加 Overall Utility head 和 ranking supervision，另行设计。

Stage 0 只验证框架正确性，不作为最终 Reviewer 效果实验。

## 2. 完整框架

输入始终是两段同步视频和一个完整 QA candidate：

```text
Video A（speaker）+ Video B（provider）+ QA
→ Qwen3-VL-8B shared multimodal representation
→ Evidence / Answerability / Formality 三个独立三分类 head
```

完整 Stage 2 默认在 `model.language_model.layers.34` 和 `.35` 的
`self_attn.q_proj`、`self_attn.v_proj` 注入 LoRA。原始 backbone 参数全部冻结。

三个 head 分别计算三分类交叉熵，完整模式的总损失为：

\[
L_{\mathrm{total}}=\frac{L_E+L_A+L_F}{3}
\]

Overall ranking 在数据合同中保留，但 Stage 0–2 不参与训练。

## 3. Stage 0 合同

Stage 0 通过显式配置选择 `evidence_quality`，而不是维护一套复制的模型代码：

- 仅构建或启用 `evidence_head = Linear(hidden_dim, 3)`；
- 只计算 Evidence Quality 的交叉熵；
- backbone 和所有 LoRA adapter 均不参与训练；
- optimizer 只能接收 Evidence head 参数；
- checkpoint 明确记录 `stage=stage0`、`active_heads=[evidence_quality]`、`lora_enabled=false`；
- 评估只输出 Evidence Quality 指标；
- prompt 不包含标签、ranking、candidate ID 或其他标注元数据。

Stage 0 优先选择 Evidence Quality，因为当前 completed 标注的 1/2/3 分布为
`106/51/107`，且该任务确实依赖视频证据。

## 4. Stage 0 Gates

依次执行：

1. 零 GPU CSV、路径和数据合同审计；
2. Qwen3-VL module structure 检查；
3. synthetic hidden unit tests；
4. 一条真实双视频 candidate 的 forward/backward；
5. Evidence head 参数必须更新；
6. backbone 所有参数必须保持冻结；
7. 2/1/1 evidence 的小样本 overfit；
8. checkpoint save/reload；
9. 输出 Evidence accuracy、macro-F1、confusion matrix、逐等级 precision/recall、expected-score MAE。

Stage 0 不执行正式 validation 结论，也不要求 LoRA 梯度。

## 5. 当前正式数据划分

当前标注文件含 420 条 candidate、70 个 evidence，每个 evidence 恰好 6 条。固定随机种子 42 后，正式合同为：

- training：60 个 evidence；
- validation：10 个 evidence；
- locked test：0 个 evidence。

因此本轮记为 `60/10/0`。validation 用于训练过程监控和 checkpoint 选择，不得同时宣称为独立 locked test；未来需要用新的未参与开发决策的数据补建 locked test。

## 6. 提交边界

新远端分支只提交 Reviewer 所需的白名单文件：

- `training/grpo_v3/experiments/human_preference_reviewer/`；
- `hpc/grpo_v3/human_preference_reviewer/`；
- `tests/training/grpo_v3/experiments/human_preference_reviewer/`；
- Reviewer 直接依赖且远端基线缺失的最小公共文件。

不提交 CSV、视频、模型、checkpoint、输出、日志、临时审计结果或 archived 副本。
不删除远端基线中与 Reviewer 无关的历史文件，避免让协作者审查无关的大规模删除。

## 7. 文档边界

分支提供两份面向协作者的短文档：

- `README_CN.md`：解释完整框架、目录、数据状态和阶段顺序；
- `TORCH_RUNBOOK_STAGE0_CN.md`：只包含 Stage 0 可复制的集群命令、Gate、产物和失败收集。

完整三 head + LoRA 的正式训练命令不混入 Stage 0 Runbook。
