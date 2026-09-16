# 多模态 Reviewer 训练框架

本目录实现两段同步 egocentric 视频与一个 QA candidate 的监督式 Reviewer。当前 `binary_reviewer_v2` 为 Evidence Quality、Answerability、QA Formality 各训练一个二分类 head，输出顺序固定为 `[fail, pass]`。

旧人类 1–3 分数不会被删除，读取后确定性映射为：`1 → fail (0)`，`2/3 → pass (1)`。每个 head 使用训练 split 自己的 fail/pass 分布计算 `N / (2 * N_c)` class weight，并使用 weighted cross-entropy。validation/test 不参与 class-weight 估计。三个 active head 的 CE 等权平均，不再使用 `0.4/0.4/0.2` task priority。

CSV 中的 `aggregate_rank` 和派生的 `fea_total_score` 继续被忽略，不进入样本、模型、loss 或 checkpoint 合同。旧 ordinal 合同与 manifest 快照见 `ARCHIVED_ORDINAL_REVIEWER_V1.md`；当前精确合同见 `BINARY_REVIEWER_V2_CONTRACT.md`。

## 分阶段启用

- Stage 0：只训练 Evidence Quality binary head，整个 backbone 完全冻结，用于验证框架。
- Stage 1：Evidence Quality binary head 加最后两个 shared blocks 的 LoRA。
- Stage 2：三个独立 binary heads 共同训练，共享最后两个 blocks 的 LoRA，并执行一次 joint backward。

本仓库不提供 Overall Utility head、pairwise/tie loss、ranking supervision、`lambda_rank` 或 GRPO。

Stage 2 的最小 LoRA 目标是：

```text
model.language_model.layers.34.self_attn.q_proj
model.language_model.layers.34.self_attn.v_proj
model.language_model.layers.35.self_attn.q_proj
model.language_model.layers.35.self_attn.v_proj
```

原始 backbone、vision tower 和三个分类 head 之外的参数保持冻结。Stage 0 连 LoRA 也不注入，只允许 `evidence_head.*` 更新。

## 主要入口

- `v1/data.py`：CSV、1–3 到 binary 的映射、training-only class weights、evidence-level split。
- `v1/modeling.py`：共享 representation 与可选择的 `[fail, pass]` heads。
- `v1/losses.py`：per-head class-weighted CE 与 active heads 等权平均。
- `v1/train.py`：Stage 0–2 的训练、评估和参数 Gate。
- `v1/audit.py`：零 GPU 数据审计、视频映射、Qwen3-VL 结构检查。
- `TORCH_RUNBOOK_STAGE0_CN.md`：首轮单 head 集群验证。
- `TORCH_RUNBOOK_V1_CN.md`：完整三 head + LoRA 流程；目录名保留 `v1` 以避免更改现有作业入口，内部合同版本为 v2。

## 当前数据边界

当前 merged 数据仍是 70 个 completed/scored evidence packets、每个 6 candidates（420 candidates）。60 packets 用于训练、10 packets 用于 validation；split 以 evidence packet 为单位并要求 train/validation 的每个 head 都有 fail/pass 支持。原始 CSV 与 QA 样本保持不变，只有训练时 label view 和 active manifest 合同发生变化。

分类-head 训练不直接生成 `reason` 或 `fix`。CE 通过共享 LoRA 间接改变部署 judge 的表示；部署时的生成式 output contract 仍要求先输出 verdict，再输出具体 reason/fix。
