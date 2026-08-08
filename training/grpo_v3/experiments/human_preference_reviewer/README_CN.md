# 多模态 Reviewer 训练框架

本目录实现两段同步 egocentric 视频与一个 QA candidate 的监督式 Reviewer。完整 v1 输出 Evidence Quality、Answerability、QA Formality 三个独立的三分类结果；Overall ranking 只保留在数据合同中，当前不训练。

## 分阶段启用

- Stage 0：只训练 Evidence Quality head，整个 backbone 完全冻结，用于验证框架。
- Stage 1：Evidence Quality head 加最后两个 shared blocks 的 LoRA。
- Stage 2：三个独立 heads 共同训练，共享最后两个 blocks 的 LoRA。
- Stage 3：以后增加 Overall Utility 与 ranking supervision。

Stage 2 的最小 LoRA 目标是：

```text
model.language_model.layers.34.self_attn.q_proj
model.language_model.layers.34.self_attn.v_proj
model.language_model.layers.35.self_attn.q_proj
model.language_model.layers.35.self_attn.v_proj
```

原始 backbone、vision tower 和三个分类 head 之外的参数保持冻结。Stage 0 连 LoRA 也不注入，只允许 `evidence_head.*` 更新。

## 主要入口

- `v1/data.py`：CSV、每个 evidence 恰好 6 candidates、evidence-level split。
- `v1/modeling.py`：共享 representation 与可选择的分类 heads。
- `v1/train.py`：Stage 0–2 的训练、评估和参数 Gate。
- `v1/audit.py`：零 GPU 数据审计、视频映射、Qwen3-VL 结构检查。
- `TORCH_RUNBOOK_STAGE0_CN.md`：首轮单 head 集群验证。
- `TORCH_RUNBOOK_V1_CN.md`：完整三 head + LoRA 流程；Stage 0 通过前不要执行正式训练。

## 当前数据边界

最新 CSV 有 100 个 evidence、每个 6 candidates；严格 completed 只有 44 个，因此正式 40/10/10 仍缺 16 个。Answerability 等级 3 目前只有 1 条。Stage 0 使用小样本 Gate，不代表最终 unseen-evidence 效果。
