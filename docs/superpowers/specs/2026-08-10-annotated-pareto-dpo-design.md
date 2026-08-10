# 人工标注 Pareto 偏好直接训练双视频 QA Generator 设计

## 1. 目标与边界

本阶段直接使用 `rlhf_candidate_scores_merged_70_packets.csv` 中的人工 F/E/A 绝对评分训练双视频 QA generator，不先依赖 learned Reviewer，也不把静态 CSV 伪装成在线 GRPO 查表 reward。

本阶段采用同一 `evidence_id` 内的 Pareto 支配关系构造 DPO `chosen/rejected`。目标是验证 generator 是否能提高人工明确占优 QA 的条件概率，并在未见 evidence 的固定候选和自由生成评估中保持该趋势。

本阶段不实现：

- CSV exact-match 在线 reward；
- Overall Utility、Bradley–Terry 标注或人工总排名补造；
- 对不可比较候选强行设定权重；
- Reviewer v1 训练或 learned-reviewer 在线 GRPO；
- 将 validation 结果称为 locked-test 结果。

## 2. 固定数据合同

输入 CSV：

`rlhf_candidate_scores_merged_70_packets.csv`

固定 SHA-256：

`32679019FD7C665A0632E9885405BDF13C77B51386EFC56E7B29B24192210CD7`

必须验证：

- 420 行；
- 70 个唯一 `evidence_id`；
- 每个 evidence 恰好 6 个唯一 `candidate_id`；
- `annotation_status=completed`、`packet_skipped=false`、`evidence_reviewed=true`；
- `formality_score`、`evidence_grounding_score`、`answerability_score` 均为整数 `1/2/3`；
- 每个 QA 恰好五个非空选项；
- `correct` 为 `A`–`E`，且 `answer` 与正确选项文本一致；
- 每个 evidence 的两条视频 URL 和用户角色内部一致；
- `candidate_id` 前缀与 `evidence_id` 一致。

正式划分复用 seed 42 的 `split_60_10.json`：

- training：60 evidence；
- validation：10 evidence；
- locked test：0；
- reserve：0。

任何训练 pair 的两端必须属于同一个 training evidence。任何 validation candidate 不得进入训练 pair、训练 sampler 或 checkpoint 选择以外的训练计算。

## 3. QA completion 序列化

每个 candidate 必须被序列化成与生产 generator 兼容的确定性 JSON completion。规范字段为：

```json
{
  "question": "...",
  "options": ["...", "...", "...", "...", "..."],
  "correct": "C",
  "answer": "..."
}
```

序列化必须使用 UTF-8、固定字段顺序、禁止 NaN，并保持 candidate 原始语义。内容指纹由 `evidence_id` 与规范化 completion JSON 共同计算；不能只按问题文本去重。

若同一 evidence 内多个不同 `candidate_id` 具有相同 completion 指纹：

- 数据审计必须记录这些 candidate；
- 不在这些 candidate 之间构造偏好 pair；
- 若它们分别与第三个 candidate 构成相同方向的 Pareto 关系，只保留一个确定性代表，避免重复加权。

`display_order` 只表示标注页面展示顺序，禁止作为人工排名或 reward。

## 4. Pareto 偏好合同

候选 `i` 的人工评分向量为：

\[
\mathbf{s}_i=(F_i,E_i,A_i)
\]

其中三项均为越高越好。若：

\[
F_i\ge F_j,\qquad E_i\ge E_j,\qquad A_i\ge A_j
\]

且至少一个维度严格大于，则 `i` Pareto 支配 `j`，构造：

```text
chosen = i
rejected = j
```

以下情况不得构造有方向的 pair：

- 三维评分向量完全相同；
- 两个候选在不同维度互有优劣；
- completion 内容指纹相同；
- 任一候选 schema、标签、视频身份或 evidence 身份不合法。

当前 CSV 的只读审计基线为：1050 个同 evidence 候选组合，其中 643 个 Pareto 支配、207 个同评分向量、200 个不可比较。正式构建器必须重新计算并把实际计数写入审计 JSON；不能把这些基线数字硬编码为运行结果。

## 5. DPO 数据记录

每条输出记录至少包含：

```json
{
  "messages": [{"role": "user", "content": "<video><video>\n..."}],
  "videos": ["speaker.mp4", "provider.mp4"],
  "chosen": "{...}",
  "rejected": "{...}",
  "evidence_id": "...",
  "chosen_candidate_id": "...::candidate_03",
  "rejected_candidate_id": "...::candidate_06",
  "chosen_scores": {"formality": 3, "evidence": 3, "answerability": 2},
  "rejected_scores": {"formality": 2, "evidence": 1, "answerability": 1},
  "preference_source": "human_fea_pareto_v1"
}
```

双视频顺序必须保持现有角色合同：第一条为 A/Speaker，第二条为 B/Provider。训练与评估使用完整同步原视频，不用 pruned video、cluster member frames 或 centroid frames。

Generator prompt 必须来自一个显式、版本化的 prompt builder。若 CSV 不含原始 `question_type` 或 `generation_mode`，不得猜测逐行历史 prompt；首版统一使用现有生产双视频五选一 QA 生成合同，并在数据清单中记录 prompt revision 与 SHA-256。

## 6. 训练合同

首版只训练 generator LoRA，基础模型和非 LoRA 参数冻结。训练后必须保存：

- adapter/checkpoint；
- resolved training arguments；
- 输入 CSV SHA-256；
- split manifest SHA-256；
- prompt revision 与 SHA-256；
- DPO 数据清单及其 SHA-256；
- trainable-parameter audit；
- 每步 loss 和 chosen/rejected margin；
- JobID 派生的输出路径。

不在本设计中锁定具体 DPO `beta`、学习率、训练步数或 batch size。它们必须通过 Structure、1-step Smoke 和小样本 Overfit Gate 逐级确定，不能在一次失败后同时修改多项。

## 7. Gate 顺序

### Gate 0：零 GPU 数据审计

要求：

- CSV 和 split 合同全部通过；
- train/validation evidence 无交集；
- 每个输出 pair 满足 Pareto 支配；
- 不存在同 completion 指纹 pair；
- 每条记录恰好两个完整视频；
- 审计报告给出支配、同分、不可比较、重复内容、按 split 分组的精确计数。

### Gate 1：Structure Probe

要求：

- 模型与 processor 可加载；
- 双视频输入可以完成前向；
- chosen/rejected completion 可以计算有限 log probability；
- 只有预期 LoRA 参数可训练。

### Gate 2：1-step Smoke

要求：

- loss、梯度和 chosen/rejected margin 有限；
- LoRA 参数变化非零；
- 非 LoRA 参数变化为零；
- checkpoint 可以严格重载；
- 作业成功不能替代上述产物检查。

### Gate 3：4-evidence Overfit Probe

固定 4 个 training evidence。要求训练集 pair accuracy 和 chosen-rejected margin 明显改善，并报告是否出现长度捷径或 JSON 退化。此 Gate 只证明链路可学习，不证明泛化。

### Gate 4：60-evidence 正式训练

只有 Gate 0–3 全部通过后才能提交。训练不读取 validation completion 参与梯度更新。

### Gate 5：10-evidence 固定候选 Validation

至少报告：

- Pareto pair accuracy；
- chosen-rejected 长度归一化 log-prob margin；
- 每 evidence top-1 candidate 的 F/E/A 分布；
- log-prob 与三维等级的相关性；
- 按 F/E/A 支持度分解的结果；
- 与未训练 base generator 的同协议比较。

Validation 可用于 checkpoint 选择，但不是 locked test。

### Gate 6：自由生成盲评

在未参与训练的 evidence 上，由 base 与 adapter 按相同生成配置产生新 QA；混合、匿名、随机排序后使用同一 F/E/A rubric 重新人工标注。只有该 Gate 才能支持“自由生成质量改善”的研究结论。

## 8. 失败与停止规则

- Gate 0 失败：只修数据、身份或序列化合同，不提交 GPU 作业。
- Gate 1 失败：只修模型/processor/媒体加载，不调整训练超参数。
- Gate 2 无参数变化或 margin 非有限：停止，不进入 Overfit。
- Gate 3 不能过拟合：先检查 prompt、completion masking、视频顺序和 DPO loss，再考虑单项超参数变化。
- Gate 5 未优于 base：停止正式结论，保留失败 checkpoint 和全部产物；不得用 train loss 下降替代 validation。
- Gate 6 未改善：结论限定为固定候选偏好学习成功，不宣称新 QA 生成改善。

## 9. 代码隔离与版本边界

实现应从 `feature/multimodal-reviewer-training` 的当前干净提交创建独立分支和独立 worktree，例如：

```text
feature/annotated-pareto-dpo
EgoQA-two-user-annotated-pareto-dpo
```

不得在当前脏的 `grpo/qa-cross-view-relation-v2` 工作树中直接实现、merge 或 rebase。新实验使用独立目录，不修改既有 cross-view reward 语义，也不把 Reviewer v1 三头训练改造成 DPO trainer。

建议目录边界：

```text
training/grpo_v3/experiments/annotated_preference/
tests/training/grpo_v3/experiments/annotated_preference/
hpc/grpo_v3/annotated_preference/
```

## 10. 完成定义

本实现阶段完成需要同时满足：

1. 设计对应的数据构建器、审计器、测试和训练入口已进入独立分支；
2. 零 GPU 测试覆盖 Pareto、同分、不可比较、重复内容、split 泄漏和错误 schema；
3. 本地静态检查通过；
4. Torch 侧至少完成 Gate 0–3，且产物来自真实 JobID；
5. 若尚未执行 Gate 4–6，汇报中必须明确写为未验证，不得声称 generator 已改善。
