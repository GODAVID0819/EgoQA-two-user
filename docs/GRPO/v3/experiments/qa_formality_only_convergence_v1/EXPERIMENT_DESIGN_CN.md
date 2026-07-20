# GRPO v3 仅 `qa_formality` 连续置信度 Reward 最小收敛实验

> 规格状态：已获用户批准，等待落盘规格复核  
> 规格版本：`qa_formality_confidence_v1`  
> 日期：2026-07-20

## 1. 实验问题

本实验只回答：

> 在现有原生双视频、ms-swift、BF16 LoRA、GRPO 训练链路下，如果训练标量只来自冻结的 `qa_formality` judge，policy 是否能够提高该 reward？

本实验是训练链路的最小可收敛性探针，不是完整 QA 质量实验。即使通过，也只能说明现有 GRPO 链路可以优化 `qa_formality` judge 所定义的目标，不能说明：

- QA 的视频证据一致性提高；
- QA 的单/双用户可回答性提高；
- 完整 repo-native 联合 reward 收敛；
- reviewer 与人工判断一致；
- Gate 4 已解锁；
- 最终模型的综合 QA 质量提高。

## 2. 已知事实与设计动机

旧 Gate 3 作业 `14194844` 的真实 trace 中：

- 共有 80 个候选；
- 78 个候选包含可用 `qa_formality` 判断，其中 PASS 71 个、FAIL 7 个；
- 18 个可完整重建的四候选组中，原二值 `+0.5/-0.5` reward 只有 7 组具有正标准差；
- 原二值 reward 的正标准差组比例只有 `7/18 = 38.9%`；
- 同一 judge 的 PASS/FAIL logprob 在这 18 个完整组中均存在可见差异。

因此，直接保留二值 reward 会让多数 GRPO 组的组内优势为零。本实验改用由同一个 `qa_formality` judge 输出的 PASS/FAIL logprob 构造连续 reward，不引入其他 judge 信号。

## 3. Reward 契约

### 3.1 唯一训练标量

设 judge 输出：

\[
\ell_{P}=\log P(\mathrm{PASS}),\qquad
\ell_{F}=\log P(\mathrm{FAIL})
\]

定义置信度 margin：

\[
m=\ell_{P}-\ell_{F}
\]

训练 reward 为：

\[
r_{\mathrm{formality}}
=
\frac{\operatorname{clip}(m,-32,32)}{32}
\]

性质：

- reward 固定在 \([-1,1]\)；
- 正值表示 judge 更偏向 PASS；
- 负值表示 judge 更偏向 FAIL；
- 不会像直接使用 PASS 概率一样在高置信度区域数值饱和；
- 极端 logprob 不会无限放大；
- 旧 Gate 3 trace 回放时，18/18 个完整组保持正标准差；
- reward 变换的截断值 `32` 是固定实验常量，不允许在同一实验内调节。

每条 trace 必须同时保存：

- `pass_logprob`；
- `fail_logprob`；
- `logprob_margin_raw`；
- `logprob_margin_clipped`；
- `qa_formality_confidence`；
- `qa_formality_status`；
- `reward_source`；
- 原始 judge 输出和解析结果。

### 3.2 明确移除的信号

训练标量中不得出现：

- `groundedness`；
- `combined_answerability`；
- `grounded_answerable_bonus`；
- `subset_leakage`；
- 独立的 `shallow_activity_query` reward；
- `provider_only_cap`；
- `shallow_activity_cap`；
- `speaker_leakage_cap`；
- `format` penalty；
- `evidence_groundedness` judge reward；
- answerability judge reward。

`other_person_activity_query` 和 `direct_name_leakage` 仍可由 `qa_formality` judge 作为其既有 rubric 的内部语义子检查，但不得作为独立 reward component 再次相加。

### 3.3 JSON 与不可判定候选

候选先经过现有严格 JSON 解析和保守修复：

- 原始合法 JSON：直接交给 `qa_formality` judge，不加格式奖励或罚分；
- 可保守修复 JSON：修复后交给 `qa_formality` judge，不加格式奖励或罚分；
- 完全不可恢复：无法形成可判定 MCQ，按 `qa_formality` rubric 中的结构失败处理，reward 固定为 `-1.0`。

完全不可恢复候选的 trace 必须记录：

```json
{
  "qa_formality_status": "FAIL",
  "reward": -1.0,
  "reward_source": "deterministic_unjudgeable_floor",
  "judge_called": false
}
```

这不是独立的格式 reward。对应记录的 reward component 集合仍严格等于：

```text
{qa_formality_confidence}
```

### 3.4 Judge 缺失和基础设施故障

以下情况属于基础设施错误，必须中止当前作业，不得静默转换为低 reward：

- reviewer 服务不可达；
- judge 调用异常；
- judge 输出无法解析；
- PASS/FAIL logprob 缺失；
- logprob 或 reward 为 `NaN`/`Inf`；
- evidence ID 或 group 元数据错位；
- 同一 reward 调用混入不同训练阶段。

候选自身不可恢复的 JSON 属于可训练的 formality FAIL；reviewer 或数据管线故障不属于候选错误，两者必须严格分开。

## 4. 执行架构

新增专用 `qa_formality-only` reward 路径：

```text
原生有序双 MP4
    ↓
Qwen3-VL-2B policy 生成四个候选
    ↓
JSON 解析／保守修复
    ↓
qa_formality 纯文本 prompt
    ↓
冻结的 Qwen3-VL-8B qa_formality judge
    ↓
PASS/FAIL logprob margin
    ↓
qa_formality_confidence reward
    ↓
GRPO 组内归一化与 LoRA 更新
```

该路径不得调用：

- `evidence_groundedness` judge；
- answerability evaluator；
- reviewer 的视频推理分支。

Policy 输入仍是原生、有序的两段 `.mp4`，不改为 `sampled_frames`。Reviewer 的 `qa_formality` 判断为纯文本输入，但仍使用冻结的 8B reviewer，保持 judge 身份不变。

## 5. 实验阶梯

### 5.1 阶段 A：旧 trace 离线回放

不申请 GPU。读取作业 `14194844` 的 `reward_trace.jsonl`，重新计算：

- PASS/FAIL margin；
- 连续 formality reward；
- 每组 reward mean/std；
- 有效 logprob 比例；
- 正标准差组比例；
- reward 最小值和最大值；
- 缺失值、非有限值和不可判定候选数量。

通过条件：

1. 所有具有 PASS/FAIL logprob 的候选均产生有限 reward；
2. reward 全部位于 \([-1,1]\)；
3. 正标准差完整组比例至少为 `0.8`；
4. 输出 reward component 集合严格等于 `{qa_formality_confidence}`；
5. 回放报告记录输入 trace 的 SHA-256。

阶段 A 失败时，不进入 GPU 实验。

### 5.2 阶段 B：1-step 集成 Smoke

从已通过的 Gate 2 adapter 开始，执行 1 个 optimizer step，固定：

```text
num_generations=4
temperature=0.7
learning_rate=1e-5
lr_scheduler_type=constant
beta=0.0
seed=42
data_seed=42
dataset_shuffle=false
```

通过条件：

1. 4/4 reward 有限；
2. reward component 集合严格等于 `{qa_formality_confidence}`；
3. trace 只包含 `qa_formality` judge；
4. 没有 groundedness 或 answerability judge 调用；
5. 该组 reward 标准差大于 0；
6. optimizer 完成 1 step；
7. LoRA adapter 和 processor 均可重载；
8. manifest 记录父 Gate 2、reward revision、margin 截断值和完整命令。

阶段 B 失败时，不进入正式 probe。

### 5.3 阶段 C：40-step 最小过拟合 Probe

阶段 C 必须从与阶段 B 相同的已通过 Gate 2 adapter 重新开始，不能从以下 checkpoint 继续：

- 阶段 B smoke checkpoint；
- 旧失败 Gate 3 checkpoint；
- Gate 3 v2 checkpoint；
- 任意未通过验收的中间 adapter。

固定配置：

```text
训练数据：Gate 0 的同一个固定 evidence
训练步数：40
每组候选：4
temperature：0.7
top_p：1.0
learning_rate：1e-5
lr_scheduler_type：constant
beta：0.0
seed/data_seed：42
dataset_shuffle：false
精度与微调：BF16 LoRA
LoRA target：q_proj、v_proj
冻结模块：ViT、aligner
policy 输入：原生有序双 MP4
policy：Qwen3-VL-2B-Instruct
reviewer：冻结 Qwen3-VL-8B-Instruct
```

使用单个固定 evidence 是为了验证训练链路是否能过拟合单一、低噪声目标，而不是验证泛化。使用 `temperature=0.7` 是为了维持同组候选差异；本实验不沿用 Gate 3 v2 的 `temperature=0.3`。

## 6. 收敛验收

阶段 C 必须输出：

- 40 个 group 的 reward mean/std 序列；
- 前 10 组 reward 均值；
- 后 10 组 reward 均值；
- 首尾 reward delta；
- 对 40 个 group mean 做普通最小二乘得到的 slope；
- 正标准差组数量和比例；
- formality PASS/FAIL 数量和比例；
- raw margin 与 clipped margin 的分布；
- 不可恢复候选数量和比例；
- trainer 数值有限性；
- `global_step`；
- adapter/processor 重载结果；
- 父 adapter、代码版本、数据 SHA-256 和完整启动命令。

### 6.1 硬通过条件

以下条件必须全部成立：

1. `global_step == 40`；
2. 160/160 reward 有限；
3. infrastructure mask 数量为 0；
4. 40 个组均包含 4 个候选；
5. 正标准差组比例至少为 `0.8`；
6. 后 10 组均值高于前 10 组均值；
7. 40 组 reward mean 的线性回归 slope 大于 0；
8. reward component 集合严格等于 `{qa_formality_confidence}`；
9. trace 中没有 groundedness/answerability judge 调用；
10. adapter 和 processor 重载成功；
11. trainer 记录的相关数值均为有限值；
12. 不可恢复候选比例没有相对前 10 组上升。

形式化地：

\[
\overline r_{\mathrm{last10}}-
\overline r_{\mathrm{first10}}>0
\]

并且：

\[
\operatorname{slope}(r_1,\ldots,r_{40})>0
\]

第一轮不设置 `reward_delta > 0.05` 等绝对 effect-size 硬阈值，因为初始 policy 的 formality 已处于较高水平。报告必须给出实际 effect size，但不能用很小的正数夸大为显著质量提升。

### 6.2 失败分类

| 结果 | 结论 |
|---|---|
| reward 提升、正方差充足、adapter 更新 | 当前 GRPO 链路能学习 `qa_formality` 单一目标 |
| reward 未提升、正方差充足、adapter 更新 | 优化器在更新，但未形成可检测的 formality 改善 |
| 大量零方差组 | reward 信号设计或采样失败，不能据此否定 GRPO |
| adapter 未更新 | 训练更新链路失败，与 judge 质量分开诊断 |
| reward 提升但不可恢复率上升 | 存在退化或 reward hacking，不得判为通过 |
| reviewer/logprob 缺失 | 基础设施失败，不得转写成训练不收敛 |

## 7. 反 Reward Hacking 审计

训练前后必须导出可读样本，至少检查：

- 问题是否模板化重复；
- 问题长度是否异常缩短；
- 选项是否退化、重复或失去互斥性；
- 是否通过固定措辞欺骗 judge；
- QA 是否保持可读结构；
- 是否直接复制 formality prompt 中的好例子；
- 不可恢复候选是否增加。

人工检查只用于否决明显退化，不添加新的训练 reward，也不能代替本实验的数值验收。

## 8. 预定实现边界

实验代码应独立于旧 Gate 3/Gate 3 v2，预计新增：

```text
training/grpo_v3_formality_reward.py
training/grpo_v3_formality_convergence.py
hpc/grpo_v3_formality_smoke.sbatch
hpc/grpo_v3_formality_probe.sbatch
tests/training/test_grpo_v3_formality_reward.py
tests/training/test_grpo_v3_formality_convergence.py
```

允许对共享 plugin、summary 或 manifest 模块做最小接线修改，但必须保持原有 reward revision 的行为不变，并增加回归测试。

禁止：

- 覆盖或重命名旧 Gate 3 脚本；
- 修改旧实验产物；
- 同时调 learning rate、temperature、步数或 LoRA 配置；
- 将本实验通过解释成完整 Gate 3 或 Gate 4 通过；
- 在阶段 A/B 失败后绕过门槛直接提交阶段 C。

## 9. 实验文档目录

最终目录固定为：

```text
docs/GRPO/v3/experiments/qa_formality_only_convergence_v1/
├── README_CN.md
├── EXPERIMENT_DESIGN_CN.md
├── TORCH_RUNBOOK_CN.md
└── RESULT_INTERPRETATION_CN.md
```

- `README_CN.md`：实验入口、状态和最短执行顺序；
- `EXPERIMENT_DESIGN_CN.md`：本文件，定义不可临时更改的实验契约；
- `TORCH_RUNBOOK_CN.md`：可直接复制到 Torch 的上传、预检、提交、监控、验收和下载命令；
- `RESULT_INTERPRETATION_CN.md`：结果分类、允许结论和禁止结论。

## 10. 实施完成标准

本地实施只有同时满足以下条件才可称为“已准备好上传 Torch”：

1. 新 reward 的单元测试通过；
2. logprob 缺失、非有限值、截断边界和不可恢复候选测试通过；
3. 旧 trace 离线回放测试及实际回放通过；
4. 收敛分析器的窗口、slope、正方差组和失败分类测试通过；
5. 旧 GRPO v3 回归测试通过；
6. Python `compileall` 通过；
7. Slurm 参数契约检查通过；
8. runbook 中不含需要人工补写的占位符；
9. `git diff --check` 通过；
10. 本地结果只报告为静态/单元/回放验证，不提前声称 Torch 训练成功。

