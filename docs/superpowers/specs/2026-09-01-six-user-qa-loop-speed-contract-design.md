# 六用户 QA 单 Loop 提速与合同对齐设计

日期：2026-09-01

## 1. 目标与适用范围

本设计只优化六用户 QA 单个 generation slot 内最多三次 attempt 的执行路径，并修正当前代码与验收合同不一致的问题。已批准的行为变化包括：

1. answerability 继续保持两个视觉调用，但两次调用必须检查同一份 canonical needed facts；
2. evidence groundedness 统一采用一次六路完整原视频调用，不再把复杂的六用户分段观察加文本聚合作为生产路径；
3. generator 不再输出 `why_two_users_needed`，其他当前 generator 字段全部保留；
4. 同一同步六视频组中，所有通过 pruning 和媒体时长检查的 speaker packet 都保留并获得 QA generation slot；
5. 三个 generation group 仍各执行二十个 slot，每个 slot 仍最多执行三次 attempt。

本设计不修改 pruning 算法、视频时长、三组乘二十 slot 总量、generator sampling、当前 fast profile、第三次 attempt 的完整评审要求、远端作业或已有运行产物。

本设计取代 `2026-08-28-six-user-10min-reasoning-evidence-answerability-design.md` 中以下生产语义：

- 第 5 节的六用户分段 evidence 投票与聚合不再作为生产 evidence；
- 第 6 节的两条件 answerability 保留，但增加 canonical facts 同步合同；
- `why_two_users_needed` 不再属于 generator 或严格 QA schema；
- 一个同步视频组不再在 QA 调度前压缩成单一 speaker packet。

## 2. 单 Slot 数据流

一个 generation slot 最多执行三次 attempt。每次 attempt 的评审顺序保持：

```text
generator
  -> deterministic schema/formality preparation
  -> qa_formality
  -> speaker-only answerability
  -> all-six answerability
  -> simple evidence groundedness
```

前两次 attempt 继续使用 fail-fast：formality 失败后不调用 answerability/evidence；speaker-only 足够回答后不调用 all-six/evidence；all-six 不足后不调用 evidence。第三次 attempt 继续执行完整评审指标。

在全部阶段执行时，一次 attempt 的模型调用数为五次：

1. generator；
2. qa_formality；
3. speaker-only answerability；
4. all-six answerability；
5. simple evidence groundedness。

JSON repair 仍是失败时的额外调用，不计入固定五次。

## 3. 两次 Answerability 的 Canonical Facts 同步

### 3.1 调用顺序

继续采用 speaker-first，保持现有低成本 fail-fast：

1. speaker-only 调用负责从问题和五个选项中定义唯一 canonical facts，并审核这些 facts 在 speaker 完整原视频中的可见性；
2. speaker-only 已足够回答时，前两次 attempt 直接拒绝；
3. speaker-only 不足时，all-six 调用接收同一份 canonical facts，只审核它们在六个完整原视频中的可见性和来源；
4. 不增加 provider-only、subset 或额外 fact-extraction 调用。

### 3.2 Canonical fact 字段

speaker-only 的每个 fact 固定包含：

- `fact_id`：当前 answerability 内唯一且稳定的字符串，例如 `F1`、`F2`；
- `fact`：区分答案所必需的最小事实；
- `why_needed`：该事实为何是回答问题所必需；
- `visibility`；
- `confidence`；
- `source_user`；
- `original_time_range`；
- `visual_description`。

all-six prompt 接收 speaker-only 返回的 canonical facts。all-six 输出必须：

- 保持 fact 数量相同；
- 保持原始顺序；
- 原样保留 `fact_id`、`fact`、`why_needed`；
- 只允许更新 `visibility`、`confidence`、`source_user`、`original_time_range` 和 `visual_description`。

### 3.3 确定性同步检查

程序在计算 all-six sufficiency 前逐项验证：

- `fact_id` 集合、顺序和唯一性；
- `fact` 文本完全一致；
- `why_needed` 文本完全一致；
- 没有新增、删除、合并或拆分 fact。

任一项不一致时，all-six condition 无效，answerability 失败并记录 `answerability_fact_contract_mismatch`。程序不得按位置猜测、模糊匹配或静默接受改变后的 fact。

### 3.4 Gate 与最小用户集合

通过条件保持：

\[
\neg \mathrm{speaker\_sufficient}
\land
\mathrm{all\_six\_sufficient}.
\]

每个 condition 的 sufficiency 仍由全部 canonical facts 是否均为 `VISIBLE + HIGH` 确定。`minimum_required_users` 继续取 all-six 同步事实中所有 `VISIBLE + HIGH` `source_user` 的有序并集；它表示 needed-fact 来源集合，不表示已经穷举重跑所有用户子集。

## 4. Evidence 统一为简单单次版本

生产 evidence groundedness 固定执行一次调用：

- 输入：speaker 和五个 provider 的六段完整原视频；
- thinking：开启；
- `max_new_tokens`：4096；
- 输出：现有 `evidence_groundedness` PASS/FAIL、reason 和 fix 合同；
- generator rationale、generator evidence 和 generator 自检不作为视觉证据。

生产路径不再根据每用户 segment 数量切换到 chunked evidence。三分钟六段和十分钟二十段都走同一个简单 evidence 调用。

现有 `evidence_segment_specs()`、分段物化、用户投票和文本聚合代码可以暂时保留用于历史离线实验，但不得由正式六用户 QA loop 自动调用。

正式运行验收改为：

- accepted QA 恰好有一条 `evidence_groundedness_judge` prompt row；
- 该 row 使用六个完整原视频，`media_role="full"`；
- 不要求 `evidence_segment_observation`；
- 不要求 `evidence_groundedness_aggregation`；
- 结果摘要记录 `groundedness_video_count=6`、分段 observation 和 aggregation 数量为零。

## 5. Generator 字段调整

从生产 generator 输出 schema、严格 schema、归一化、trace 摘要、人工审核渲染和相关测试中删除：

```text
why_two_users_needed
```

以下 generator 字段全部保留：

- `qa_id`；
- `question_type`；
- `question`；
- `options`；
- `correct`；
- `answer`；
- `required_users`；
- `evidence`；
- `referred_timestamps`；
- `single_user_answerability`；
- `combined_answerability`；
- `generator_rationale`；
- `per_user_evidence_claims`；
- `review`。

其中 generator 仍必须自行输出原始时间证据；本设计不压缩 `evidence` 或 `referred_timestamps`，也不改变它们的含义。

旧 artifact 中存在 `why_two_users_needed` 时读取代码可以忽略该额外字段，但新 generator prompt、严格 accepted QA 和新渲染结果均不再要求或展示它。

## 6. 保留所有合格 Speaker Candidate

### 6.1 Candidate 语义

对每个同步六视频组，candidate miner 继续依次尝试 speaker 1 至 speaker 6。每个 speaker 独立执行 pruning 和媒体时长检查：

- 通过时生成一个独立 speaker packet；
- 失败时保留失败诊断但不生成 packet；
- 不因前一个 speaker 成功而停止后续 speaker；
- 同一组六视频可以产生多个 speaker packet。

生产 wrapper 不再使用 `ONE_CANDIDATE_PER_GROUP=1` 把同组候选压缩成第一条。所有通过的 speaker packet 都进入 QA 调度。

### 6.2 三组乘二十 Slot 的两层调度

三个 `generation_group_id` 仍各有二十个 slot，总计六十个。调度采用两层 round-robin：

1. group 层：每一轮依次为三个 generation group 各发出一个 slot；
2. speaker 层：每个 group 内按该组通过的 speaker packet 循环选择。

令组 \(g\) 有 \(S_g\) 个合格 speaker，则该组每个 speaker 获得的 slot 数为：

\[
\left\lfloor\frac{20}{S_g}\right\rfloor
\quad\text{或}\quad
\left\lceil\frac{20}{S_g}\right\rceil.
\]

因为 \(1\le S_g\le 6\)，每个合格 speaker 至少获得：

\[
\left\lfloor\frac{20}{6}\right\rfloor=3
\]

个完整 generation slot，满足用户批准的“至少两个 slot”要求。这里的 slot 可以最终 accepted 或 rejected；不要求每个 speaker 至少得到两条 accepted QA。

每个 slot 仍最多执行三次 attempt。slot ID 必须同时能定位 generation group、speaker packet 和 group 内 round，避免同组多 speaker 发生 identity 冲突。

### 6.3 调度异常

- 任一目标 group 没有合格 speaker packet：候选合同失败，不进入 QA；
- group 数不等于三个：正式 3×20 合同失败；
- 某组完成 slot 数不足二十：最终状态失败或按既有 deadline 规则记 partial；
- 任一合格 speaker 的 slot 数小于二：调度验收失败；
- 不通过复制 packet 或虚构 speaker 补足配额。

## 7. TDD 验证合同

实现必须先写并运行失败测试，再写最小生产代码。至少覆盖：

### Answerability

1. 六用户仍只建立 speaker-only 和 all-six 两个 condition；
2. speaker-only 输出稳定且唯一的 `fact_id`；
3. all-six prompt 收到 speaker-only canonical facts；
4. all-six 原样返回 canonical fact identity 时可计算 sufficiency；
5. all-six 新增、删除、改序、改写 `fact` 或 `why_needed` 时失败；
6. speaker sufficient 时前两次 attempt 不调用 all-six；
7. all-six minimum users 只来自同步 facts 的 `VISIBLE + HIGH` source users。

### Evidence

1. 每用户六段的三分钟 packet 只调用一次简单 evidence；
2. 每用户二十段的十分钟 packet 也只调用一次简单 evidence；
3. evidence 调用使用六个完整原视频、thinking 和 4096；
4. 正式 accepted QA 不要求 segment observations 或 aggregation；
5. 旧 chunked 工具不能被生产 dispatcher 自动选中。

### Generator schema

1. generator prompt 不包含 `why_two_users_needed`；
2. 新输出缺少该字段仍通过普通和严格 schema；
3. 其他 generator 字段仍保持原合同；
4. 渲染器不再显示该字段；
5. 带有旧额外字段的 artifact 仍可读取。

### Speaker 与 slot 调度

1. 同组所有 pruning/时长通过的 speaker packet 均被保留；
2. wrapper 不再截断为每组第一条；
3. 三组始终各获得二十个 slot；
4. 每组内部在所有合格 speaker 间轮转；
5. 合格 speaker 数为 1、2、4、6 时，每个 speaker 的 slot 数均不少于二；
6. 六 speaker 时分配为两个 speaker 四个 slot、四个 speaker 三个 slot，或等价的确定性均衡分配；
7. 每个 slot 仍保留最多三次 attempt。

## 8. 影响边界与交付

预计修改面限于：

- `prompts.py`；
- `schema.py`；
- `video_qa_loop.py`；
- `qa_generation_schedule.py`；
- `hpc/qa/smoke/run_six_user_qa_runtime_probe.sbatch`；
- 10min 3×20 wrappers；
- 人工审核渲染中对已删除字段的展示；
- 对应聚焦测试。

不修改 `qwen3vl_runner.py`、pruning 数学、模型目录、视频 FPS、像素、CUDA keeper、Slurm account/partition/QOS 或远端 worktree。实施不自动上传、提交新作业、取消旧作业、提交 Git 代码或清理产物。

本地测试只能证明代码合同、调度和 schema 一致，不能证明 Qwen3.8 远端 runtime、单 loop 实际提速、accepted QA 数量或人工质量。任何远端结论必须来自新的 JobID、实际日志和 JobID 派生产物。
