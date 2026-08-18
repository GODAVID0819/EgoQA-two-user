# 六视频多用户 QA 生成链路设计

日期：2026-08-18
状态：用户已批准设计，等待实现

## 1. 目标

把当前 QA 生成输入从两位用户扩展为六位同步用户：

\[
1\ \text{speaker}
+ 2\ \text{anchor providers}
+ 3\ \text{additional providers}
\rightarrow 1\ \text{QA}
\]

本轮只修改 QA 候选挖掘、生成、自动审核和 Torch 试验链路，不修改 GRPO、DPO 或 reviewer 训练合同。

六位用户同时作为模型输入，但不要求五个 providers 全部贡献答案。核心 answerability 条件是：speaker 单独不能答对，而六视频实际输入能够答对。

## 2. 非目标与证据边界

- 不要求六位用户缺一不可。
- 不要求两个 anchor providers 都被最终 QA 使用。
- 不要求 additional providers 提供答案。
- 不遍历全部用户子集做 answerability 搜索。
- 不修改 GRPO、DPO 和 reviewer 训练数据、奖励或模型结构。
- 不把 runtime probe 通过表述为 QA 质量结论。
- 不把自动 reviewer 通过表述为人工确认的自然性或可回答性。
- 正式 5 条 accepted QA 仅用于初步定性观察，不支持统计显著性结论。

## 3. 六用户候选结构

### 3.1 同步用户采样

从包含至少六位用户的同一同步 group 中抽取六段视频，对六位用户计算全部：

\[
\binom{6}{2}=15
\]

条 CLIP pair edges。

候选 speaker 必须至少拥有两条通过现有 pair filter 的 speaker-provider edges。从其合格邻居中选择两位 anchor providers，其余三位成为 additional providers。

provider-provider edges 以及未被选择的 speaker-provider edges 只保存为诊断，不作为候选阻断条件。若存在多个合格角色结构，使用固定随机种子选择；若所有角色结构在媒体物化阶段失败，则拒绝该 group 并保存诊断。

### 3.2 用户顺序合同

六用户顺序固定为：

1. `input_users[0]`：speaker；
2. `input_users[1]`：anchor provider 1；
3. `input_users[2]`：anchor provider 2；
4. `input_users[3]`：additional provider 1；
5. `input_users[4]`：additional provider 2；
6. `input_users[5]`：additional provider 3。

`required_users` 保留为历史兼容字段，值与 `input_users` 相同并继续决定媒体顺序；在六用户模式中，它不再表示六位用户都逻辑必要。

## 4. 媒体裁剪与路由

### 4.1 Generator 输入

- speaker：使用两条 anchor edges 对应删除区间的归一化并集进行裁剪；
- anchor provider 1：使用其与 speaker 的 pair-specific 裁剪；
- anchor provider 2：使用其与 speaker 的 pair-specific 裁剪；
- 三个 additional providers：使用完整同步视频，不根据无关或未通过的 edges 强行裁剪。

因此 generator 接收三段裁剪视频和三段完整视频。每段媒体都保存显式角色：

- `speaker_pruned`
- `anchor_provider_pruned`
- `additional_provider_full`

speaker 合并裁剪后低于现有最短保留时长时，当前角色结构失败；实现继续尝试其他合格 speaker/anchor 结构，不静默回退未裁剪 speaker。全部结构失败时才拒绝 group。

### 4.2 Judge 输入

groundedness judge 查看同一顺序的六段完整原视频。

answerability judge 只运行两个条件：

- `speaker_only`：一段 speaker 完整原视频；
- `combined_all_six_users`：六段完整原视频。

所有媒体 trace 必须同时记录用户、角色、generator path、full judge path、是否裁剪及其来源 edges。

## 5. Answerability 门禁

六用户门禁定义为：

\[
\operatorname{Pass}
=
\neg\operatorname{Correct}(\{S\})
\land
\operatorname{Correct}(\{S,P_1,P_2,P_3,P_4,P_5\})
\]

具体行为：

- speaker-only 选择正确答案：拒绝；
- speaker-only 输出无法解析：门禁失败并触发生成重试；
- speaker-only 选择错误答案：继续 all-six；
- all-six 选择正确答案：通过；
- all-six 选择错误答案或无法解析：门禁失败并触发重试。

不评估 provider-only、speaker 加部分 providers 或其他子集。这些子集是否答对均不构成阻断，也不产生额外 VLM 调用。

该门禁只说明 speaker 需要外部视角，且六视频实际输入可恢复正确答案；它不说明每个 provider 都必要。

## 6. Prompt 与审核合同

### 6.1 Generator prompt

Prompt 必须：

- 从 speaker 的第一人称经历或共享场景提出自然问题；
- 要求 speaker 自身视频不能确定答案；
- 提示优先检查两个经过 CLIP 关系过滤的 anchor providers；
- 允许 additional providers 提供证据；
- 允许一个或多个 providers 支持答案；
- 禁止声称所有 providers 都不可替代；
- 禁止要求只有六视频全集才能回答。

### 6.2 Groundedness judge

Groundedness judge 使用六段完整视频检查：

- speaker 的问题动机或共享场景锚点是否真实；
- declared answer 是否被至少一个外部视角或外部视角组合支持；
- generator 声称使用的 provider 证据是否真实可见；
- 是否存在人物、物体、动作、时间或身份误认。

未使用某个 provider 不是拒绝理由。

## 7. Schema 与 metadata

六用户 packet 新增：

- `input_users`
- `speaker_user`
- `anchor_provider_users`
- `additional_provider_users`
- `media_roles`
- `selected_anchor_edges`
- `diagnostic_pair_edges`

兼容保留：

- `required_users`：六位有序输入用户；
- `evidence_provider_user`：第一个 anchor provider；
- `evidence_provider_users`：两个 anchor providers；
- `why_two_users_needed`：保留旧字段名，但只说明 speaker 需要至少一个外部视角。

生成与审核产物新增：

- `speaker_only_correct`
- `speaker_only_choice`
- `all_six_correct`
- `all_six_choice`
- `cross_view_gain`
- `answerability_evaluated_condition_count`
- `supporting_user_claims`

`supporting_user_claims` 至少包含一个非 speaker 用户；其中引用的用户必须属于六位输入用户。不要求为全部输入用户编造 evidence claim。

## 8. 噪声指标与解释

定义：

\[
\text{cross\_view\_gain}
=
\mathbf{1}[\text{all-six correct}]
-
\mathbf{1}[\text{speaker-only correct}]
\]

accepted QA 的 `cross_view_gain` 应为 1。

每个候选至少记录：

- speaker-only 与 all-six 的选择和正确性；
- 六段媒体顺序、角色和裁剪状态；
- all-six 输入总视频时长；
- generation、groundedness 和 answerability 阶段耗时；
- `all_six_wrong` 拒绝数量和比例；
- 解析失败、媒体解码、上下文长度、OOM 和其他运行错误。

`all_six_wrong` 只说明六视频条件未恢复正确答案，不能仅凭自动输出归因为噪声。可能原因还包括 QA 不清楚、证据不足、VLM 能力、媒体顺序或解码问题。

## 9. 测试策略

### 9.1 候选与角色

- 六人产生 15 条 pair edges；
- speaker 至少有两个合格 anchor neighbors；
- 两个 anchor 合格、其他 edges 失败时仍可接受；
- 固定随机种子产生稳定用户顺序；
- 无合格角色结构时保存诊断并拒绝；
- 二用户路径保持兼容；
- CLI 支持 `selected_count=2` 和 `selected_count=6`，其他值在媒体下载和模型加载前报错。

### 9.2 媒体

- speaker 使用两条 anchor 删除区间并集；
- 两个 anchors 各使用自身边区间；
- 三个 additional providers 使用完整视频；
- generator 顺序为三段 pruned 加三段 full；
- judge 顺序为六段 full；
- 最短保留保护失败时尝试其他角色结构；
- 所有路径与用户一一对应。

### 9.3 门禁与兼容

- 六用户只生成两个 answerability 条件；
- speaker-only 正确时拒绝；
- speaker-only 错误且 all-six 正确时通过；
- 任一条件无法解析时拒绝；
- all-six 错误时拒绝；
- 二用户回归测试保持通过；
- Prompt 不再包含三用户“所有 providers 必要”的旧措辞。

### 9.4 完整验证

- 定向测试；
- 完整本地测试套件；
- Python 静态语法检查；
- 两个 `.sbatch` 的 `bash -n`；
- `git diff --check`。

## 10. Torch 作业

### 10.1 唯一六视频 runtime probe

只处理一个六用户候选，验证：

- H100 上模型能够加载；
- generator 能处理三段裁剪和三段完整视频；
- groundedness judge 能处理六段完整视频；
- answerability 能分别处理一段和六段完整视频；
- 不发生 OOM、解码、输入顺序或上下文长度错误；
- 必需环境、媒体和 trace 产物存在。

该 probe 是唯一 smoke。通过后不再串联其他规模 smoke。

### 10.2 正式 5 条小试验

- 目标为 5 条 accepted QA；
- 除六视频输入、角色结构和 answerability 外，沿用当前二用户试验的模型、阈值、视频长度、采样间隔、随机种子、生成模式和重试次数；
- `accepted_count=0` 视为自动 Gate 失败；
- accepted 数量介于 1 与 4 时按实际结果报告，不声称达到目标；
- 不自动扩大样本或提交后续实验。

两个作业均需：

- 从 JobID 派生 output 与 job-specific scratch；
- 在模型加载前封闭 HOME、cache 和临时目录；
- 在模型加载前运行存储预检；
- 审计 Python、CUDA、GPU、FFmpeg 和 TorchCodec；
- 保存 `storage_preflight.json` 与 `job_manifest.json`；
- 使用 `sbatch --parsable` 提交并自动记录 JobID；
- 提交前根据远端查询复核 account、partition、QOS 和 H100 资源；
- 不自动 push、上传或提交作业。

## 11. 产物与验收

至少保存：

- 输入 manifest；
- 六用户 candidates JSONL；
- 15 条 pair edge 与角色选择 trace；
- 六段 generator/judge 媒体 provenance；
- prompts、intermediate、accepted 和 rejected JSONL；
- 两个 answerability condition traces；
- storage/environment/job manifests；
- CSV、generation report、human review sheet；
- stdout/stderr。

运行时 Gate 只证明六视频链路可执行。正式自动 Gate 要求作业正常、产物完整、计数一致、`accepted_count>0`，且 accepted QA 均满足 speaker-only 错误和 all-six 正确。

人工终点评估必须查看六段完整视频，重点区分 additional-provider 噪声、问题质量、证据不足、VLM 能力和媒体路由错误。远端运行前所有结论均属于本地静态或单元测试证据。

## 12. 历史与分支

- 三用户设计和实施计划保留为决策历史，但标记为已被本文档取代；
- 三用户实现尚未开始，不需要回滚实现代码；
- 当前开发分支重命名为 `feature/multi-user-six-video-qa`；
- 不 push，不创建或修改远端分支。
