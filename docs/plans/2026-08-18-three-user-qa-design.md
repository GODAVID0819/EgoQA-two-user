# 1 speaker + 2 providers QA 生成链路设计

日期：2026-08-18
状态：用户已批准设计，等待实现

## 1. 目标

把当前 `1 speaker + 1 provider -> 1 QA` 的 QA 生成试验链路扩展为：

\[
1\ \text{speaker} + 2\ \text{providers} \rightarrow 1\ \text{QA}
\]

第一轮只验证三用户 QA 生成效果，不修改 GRPO、DPO 或 reviewer 训练数据合同。三位用户来自同一个同步组；generator 使用三段裁剪视频，groundedness judger 和 answerability evaluator 使用对应的三段完整原视频。

## 2. 非目标与验证边界

- 不修改 GRPO、DPO 和 reviewer 的训练、奖励或数据合同。
- 不声称任意数量 provider 已完成验证。本轮正式支持 `selected_count=2` 与 `selected_count=3`；大于 3 的值明确报错。
- 不改变历史二用户行为和既有产物语义。
- runtime probe 只证明三视频链路可执行。
- 自动门禁通过不等于 QA 已经具有人工认可的自然性和三人必要性。
- 5 条 accepted QA 仅用于初步定性观察，不构成统计显著性结论。

## 3. 候选选择：星型三用户结构

### 3.1 路线选择

采用星型三用户扩展，不采用全连接三元组过滤，也不采用“合格二用户对加随机第三人”。

从同一同步组随机抽取三段视频，枚举三种 speaker 角色。对于候选 speaker，只要求以下两条边分别通过当前两两 CLIP pair filter：

- `speaker-provider_1`
- `speaker-provider_2`

`provider_1-provider_2` 的两两评分和过滤结果写入 trace，但不作为候选阻断条件。原因是两个 provider 可以提供互补信息，不要求彼此观察到相似内容。

如果多个 speaker 都满足条件，使用固定随机种子从合格结构中选择一个，避免角色总由 manifest 顺序决定。

### 3.2 用户顺序合同

输出顺序固定为：

- `required_users[0]`：speaker；
- `required_users[1]`：provider 1；
- `required_users[2]`：provider 2。

clips、裁剪视频、完整视频、prompt 媒体和 review 媒体必须与该顺序一致。

## 4. 三视频裁剪与媒体路由

继续复用当前 pair scorer 和 pair-specific 时间裁剪结果：

- speaker 的删除区间取两条 speaker-provider 边对应删除区间的并集；
- provider 1 只使用 `speaker-provider_1` 边产生的删除区间；
- provider 2 只使用 `speaker-provider_2` 边产生的删除区间；
- 保留现有 `preserve_shared_anchor_seconds`、最短保留时长和裁剪保护；
- 任意一段视频违反最短保留约束时，整个三用户候选拒绝，不静默退回未裁剪视频。

媒体边界保持为：

- generator：三段裁剪视频；
- groundedness judger：三段对应完整原视频；
- answerability evaluator：按条件选择一段、两段或三段完整原视频。

二用户路径继续使用现有 pair 选择与裁剪逻辑。

## 5. 严格三用户回答性门禁

三用户候选共评估 7 个条件：

1. 仅 speaker；
2. 仅 provider 1；
3. 仅 provider 2；
4. speaker + provider 1；
5. speaker + provider 2；
6. provider 1 + provider 2；
7. speaker + provider 1 + provider 2。

三用户门禁定义为：

\[
\operatorname{Pass}
=
\operatorname{Correct}(S,P_1,P_2)
\land
\bigwedge_{X\subsetneq\{S,P_1,P_2\}}
\neg\operatorname{Correct}(X)
\]

即三用户全集必须答对，6 个真子集均不得答对。任意真子集答对都写入 `blocking_subset_leaks` 并拒绝候选。

二用户路径保留既有兼容行为：传统 missing-detail QA 中，provider 单独答对仍作为非阻断 warning。

当前 evaluator 在证据不足时仍被要求从 A-E 强制选择，因此真子集可能偶然猜中正确答案并造成保守拒绝。第一轮保持单次评估，通过 trace 测量该问题，不新增重复投票机制。

## 6. Prompt、schema 与 metadata

三用户 prompt 必须要求：

- speaker 提供自然的问题动机或共同经历锚点；
- 两个 provider 分别提供不同的答案承载事实；
- 删除任意一位参与者后，正确答案都不能唯一确定；
- 只有三用户全集能够支持唯一正确选项。

统一新增或使用：

- `speaker_user`
- `evidence_provider_users`
- `blocking_subset_leaks`

为兼容历史分析代码，保留：

- `evidence_provider_user`，三用户时指第一个 provider；
- `why_two_users_needed`，保留旧字段名，但内容解释三位用户为何缺一不可。

`per_user_evidence_claims` 必须覆盖全部 required users，每位用户至少有一条非空、可由其媒体验证的贡献描述。人工审查信息应明确列出 speaker、两个 providers、7 个回答性条件和每个真子集的泄漏结果。

## 7. 失败、重试与可观测性

- groundedness、formality 或 answerability 任一阻断检查失败时，沿用当前重试流程。
- 回答性反馈必须指出具体泄漏子集，例如 `speaker+provider_1` 或 `provider_1+provider_2`。
- 达到 `max_attempts` 后写入 rejected JSONL，不计入 accepted count。
- 保留每次 generation、judge、answerability 和媒体路由 trace。
- 三元组 trace 保存三条 pair edge 的评分、过滤结果、选择角色和裁剪区间。

## 8. 测试策略

按照测试先行方式覆盖：

### 8.1 候选选择

- 两条 speaker-provider 边均通过时接受；
- 任一 speaker-provider 边失败时拒绝；
- provider-provider 边失败不阻断；
- 固定随机种子产生确定角色选择。

### 8.2 裁剪与媒体

- speaker 使用两条边删除区间的并集；
- 两个 provider 分别使用自身边的区间；
- 三段裁剪视频与三段完整视频 provenance 齐全；
- 任一视频违反最短保留约束时拒绝整个三元组；
- generator 与 judge 媒体顺序匹配 `required_users`。

### 8.3 门禁与兼容

- 三用户生成全部 7 个条件；
- 仅全集答对时通过；
- 任意单用户或二用户子集答对时阻断；
- 二用户 provider-alone warning 行为不变；
- 三用户 metadata 覆盖两个 providers；
- `selected_count=2` 与既有输出保持兼容；
- `selected_count=3` 可运行；
- 大于 3 明确报错。

### 8.4 完整验证

- 相关定向测试；
- 完整本地测试套件；
- Python 静态语法检查；
- `.sbatch` 的 `bash -n`；
- `git diff --check`。

## 9. Torch 作业设计

### 9.1 唯一 runtime probe

新增一个三视频 runtime probe，只处理一个三用户候选，用于验证：

- generator 能同时解码三段裁剪视频；
- judger 和 answerability evaluator 能解码三段完整视频；
- 模型完成一次生成、审核和 7 个回答性条件调用；
- 媒体顺序与用户顺序一致。

该 probe 是本流程唯一 smoke。通过后直接进入正式试验，不串联其他规模 smoke。

### 9.2 正式小规模试验

正式作业默认目标为 5 条 accepted QA。除用户数和严格三用户门禁外，沿用当前二用户试验的模型、阈值、视频长度、采样间隔、随机种子、生成模式和最大重试次数。

两个 `.sbatch` 均需：

- 在模型加载前把 `HOME`、HF/Torch cache、临时目录和运行时编译目录封闭到 JobID 对应 scratch；
- 在模型加载前运行存储预检并保存 `storage_preflight.json`；
- 分别审计训练/推理环境、CUDA、GPU、FFmpeg 和 TorchCodec；
- 从 JobID 派生输出目录；
- 保存代码提交号、参数、输入 manifest、输出路径和运行环境到 `job_manifest.json`；
- 不使用 `latest_*` 或固定共享目录表示当前作业产物。

Runbook 必须使用 `sbatch --parsable`，兼容 cluster 后缀并自动提取 JobID、写入时间戳提交 manifest。登录 shell 命令不得关闭 SSH 会话。同步采用窄范围交互式 SFTP，不覆盖远端数据和历史产物。

## 10. 产物与验收

正式作业至少保存：

- `storage_preflight.json`
- `job_manifest.json`
- 输入数据 manifest
- 三用户候选 JSONL
- 三元组及三条 pair edge trace
- 原视频和裁剪视频 provenance
- generator prompts
- intermediate、accepted 和 rejected QA JSONL
- 7 条回答性条件 trace
- CSV、generation report、human review sheet
- review videos 或明确索引
- stdout/stderr。

### 10.1 运行时 Gate

- 三段 generator 媒体与三段 judge 媒体顺序正确；
- FFmpeg/TorchCodec 解码成功；
- 模型调用完成；
- 7 个回答性条件齐全；
- 无 CUDA、OOM、媒体缺失或 schema 异常。

### 10.2 正式试验 Gate

- Job 状态和退出码正常；
- 必需产物存在且非空；
- accepted、rejected 与 attempt 数量可核对；
- `accepted_count > 0`；
- 每条 accepted QA 均通过 groundedness、formality 和严格三用户门禁；
- 每条 accepted QA 均有三位用户完整 provenance。

### 10.3 研究结论

正式试验报告接受率、主要拒绝原因和人工审查观察。自动门禁结果只作为候选筛选证据；是否自然、是否真正需要三人，最终以三段完整原视频上的人工审查为准。
