# 多模态 Reviewer 训练框架

本目录实现 `两段同步第一视角视频 + 一条 QA candidate` 的监督式 Reviewer。Reviewer v1 输出三个彼此独立的三分类结果：

- Evidence Quality；
- Answerability；
- QA Formality。

每个标签均为 `1/2/3`，训练前显式映射到交叉熵所需的 `0/1/2`。Overall ranking 只保留在数据合同中，当前不训练 Overall Utility、pairwise 或 tie loss。

## 当前数据合同

- 文件：`rlhf_candidate_scores_merged_70_packets.csv`；
- SHA-256：`32679019FD7C665A0632E9885405BDF13C77B51386EFC56E7B29B24192210CD7`；
- 420 行、70 个 evidence、每个 evidence 恰好 6 条 candidate；
- 正式划分：60 个 training evidence、10 个 validation evidence、0 个 locked test，即 `60/10/0`；
- 按 `evidence_id` 划分，两个集合不得重叠；
- 本轮 validation 可用于 checkpoint 选择；未来取得独立新标注后，再建立 locked test。

## 分阶段启用

- Stage 0：只训练 Evidence Quality head，backbone 完全冻结，验证训练链路确实可学习。
- Stage 1：Evidence Quality head 加最后两个 shared blocks 的 LoRA。
- Stage 2：三个独立 heads 共同训练，共享最后两个 blocks 的 LoRA。
- Stage 3：以后增加 Overall Utility 与 ranking supervision，另行设计。

Stage 2 的最小 LoRA 目标为：

```text
model.language_model.layers.34.self_attn.q_proj
model.language_model.layers.34.self_attn.v_proj
model.language_model.layers.35.self_attn.q_proj
model.language_model.layers.35.self_attn.v_proj
```

原始 backbone、vision tower 和非 LoRA 参数保持冻结。Stage 0 连 LoRA 也不注入，只允许 `evidence_head.*` 更新。

## 主要入口

- `v1/data.py`：CSV 合同与 evidence-level split；
- `v1/modeling.py`：共享表示与可选择的分类 heads；
- `v1/train.py`：Stage 0–2 训练、评估和参数 Gate；
- `v1/audit.py`：零 GPU 数据审计、视频映射、模型结构检查；
- `TORCH_RUNBOOK_STAGE0_CN.md`：单 head 框架验证；
- `TORCH_RUNBOOK_V1_CN.md`：60/10 正式三 head + LoRA 训练与 validation。
