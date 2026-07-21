# GRPO v3 Combined-Video Answer-Margin Convergence v1 实验设计

> 规格状态：已批准方向的正式落盘稿，等待用户复核
>
> Reward 版本：`combined_video_answer_margin_v1`
>
> 实验版本：`combined_video_answer_margin_convergence_v1`
>
> 日期：2026-07-21

## 1. 唯一实验问题

本实验只回答：

> 在现有有序双原生视频、ms-swift 4.2.2、Qwen3-VL-2B BF16 LoRA 和 GRPO 链路下，使用冻结视觉答题器给出的五选一答案 margin 作为唯一 reward 时，policy 能否在 40 个 optimizer step 内形成可重复检测的 reward 改善？

本轮目标是先用一个固定 generation temperature 证明“当前 native-video GRPO 框架能够优化一个清晰、连续、与 QA 答案正确性直接相关的 reward”。即使该单条件实验通过，也不能据此声称：

- 最终真实 QA 综合质量已经提高；
- 生成问题满足双用户 information-gap 目标；
- asker-only 不可回答性已经改善；
- groundedness、formality、leakage 或原 repo-native 联合 reward 已改善；
- 更大 policy 可以直接复用本结果而无需重新验证；
- Gate 4 或后续大规模训练已自动解锁。

## 2. 与既有策略基线的关系

`docs/GRPO/v3/NATIVE_VIDEO_MSSWIFT_GRPO_STRATEGY_CN.md` 仍是 v3 的全局策略基线。本实验只在已获批准的最小范围内作实验级覆盖：

1. 唯一训练标量从 repo-native 联合 reward 改为 `combined_video_answer_margin_v1`；
2. 冻结评分模型从原 8B reviewer 改为独立冻结的 `Qwen/Qwen3-VL-2B-Instruct` 答题器；
3. 不调用 groundedness、answerability、leakage、`qa_formality` 或 PASS/FAIL reviewer；
4. completion 只需宽松恢复 `question/options/correct` 三个核心字段，不以完整 schema 或严格 JSON 为实验目标。

以下基线约束保持不变：

- policy 输入必须是 evidence packet 中两段有序原生 `.mp4`；
- policy 固定为 `Qwen/Qwen3-VL-2B-Instruct`；
- 使用 ms-swift 4.2.2、BF16 LoRA，冻结 ViT 与 aligner；
- 默认不使用 QLoRA，不把 sampled frames 当成正式输入；
- 扩大实验必须逐级通过真实调用形状的前置门槛；
- adapter 必须保存、真实重载并产生固定评估证据；
- 本地检查、Torch runtime 和实验结论必须分层表述。

本文件只定义 v1 实验，不修改全局策略文档。若 v1 通过，是否将 answer-margin 纳入后续主策略，需要另行批准。

## 3. 已知失败证据与换 reward 的理由

此前 `qa_formality_confidence_v1` 40-step Probe 已完整训练并保存可重载 LoRA，但在线趋势和固定端点评估均未证明改善。已归档的固定端点评估主要结果为：

- step 0 mean 约为 `0.45825`；
- step 40 mean 约为 `0.38477`；
- 配对均值差约为 `-0.07349`；
- 配对 bootstrap 95% CI 约为 `[-0.25024, 0.05103]`；
- 8 wins / 2 ties / 6 losses；
- 不可判定率从 0 上升到约 6.25%；
- 结论为 `not_improved`。

人工检查还发现，PASS/FAIL formality 判断与 QA 内部答案一致性并不可靠。因此，本轮不继续只调 `qa_formality` 的 temperature、学习率或 reviewer 大小，而把 reward 改为视觉答题器对生成器声明答案的连续支持程度。

## 4. 方案比较与正式选择

### 4.1 方案 A：独立 Transformers 冻结答题器，计算 A–E 序列 logprob

GPU0 运行 policy 生成与训练，GPU1 常驻一个独立的冻结 2B 答题进程。答题器对 A–E 五个规范答案标签分别执行 teacher-forcing，返回完整序列 logprob。

优点：

- 不依赖生成采样，分数连续且可复现；
- A–E 即使被 tokenizer 分成多个 token 也能完整评分；
- 五个标签始终都有分数，不受 API `top_logprobs` 截断影响；
- scorer 与训练进程、optimizer 和 LoRA 参数物理隔离；
- 可直接记录 token ID、逐 token logprob 和完整 margin。

代价：需要实现一个只服务本机作业的轻量 scorer 接口，并独立验证真实双视频 forward 的显存和吞吐。

### 4.2 方案 B：vLLM OpenAI-compatible 接口生成一个答案字母并读取 top logprobs

优点是服务启动和并发模式与旧 reviewer 接近。主要风险是：A–E 不一定全部出现在返回的 top logprobs 中，多 token 标签的序列概率也不能由单个输出 token 可靠替代。缺失标签若被填默认低分会改变 reward 语义，因此 v1 不采用。

### 4.3 方案 C：在 policy 训练进程内共置冻结答题器

优点是少一个服务进程。缺点是 scorer 与 policy 争用 GPU0 显存，容易把模型加载、训练反向传播和评分故障混在一起，也更难证明 scorer 参数完全没有被 optimizer 接管，因此 v1 不采用。

### 4.4 正式决策

v1 固定采用方案 A：

```text
GPU0：Qwen3-VL-2B policy，BF16 LoRA，生成与 GRPO 更新
GPU1：独立 Qwen3-VL-2B 冻结答题器，BF16 inference-only
```

若方案 A 的 scorer-only runtime probe 失败，必须先定位失败层级；不得在同一实验版本中静默改成方案 B、伪造缺失 logprob 或将 scorer 移回 GPU0。

## 5. 数据与媒体契约

训练和评分均使用当前 fixed-evidence：

```text
evidence_id=EGOLIFE2U_DAY2_11350000_A1_A5
```

该 identity 已由归档 fixed-eval 的 `resolved_config.json` 复核。实际 JSONL 行、两段视频路径和文件 SHA-256 必须在后续 preflight 中解析并冻结，不能只凭文件顺序选择“第一条”。两段有序原生 MP4 的逻辑结构为：

```text
evidence packet
  ├── user-1 native MP4
  └── user-2 native MP4
```

硬性约束：

1. 两段视频的路径、用户身份、顺序和时间语义不得改变；
2. policy 和冻结答题器必须接收同一对视频及同一媒体预算；
3. 不允许用预抽取图片帧替代正式训练或评分输入；
4. processor 在运行时按固定 `fps/num_frames/max_pixels/min_pixels` 解码仍属于原生视频输入；
5. 同一 prompt 的四个 completion 必须共享完全相同的视频、prompt 和媒体参数；
6. scorer prompt 只能加入生成问题和打乱后的五个选项，不能加入生成器声明的答案、`answer` 文本、解释、rationale、self-check 或其他可能泄露答案的字段；
7. manifest 必须记录两个视频的绝对路径、文件大小、SHA-256、媒体预算和实际 processor 元数据。

## 6. 宽松核心 QA 提取契约

### 6.1 可进入正常评分的最小字段

completion 只需恢复：

```text
question：非空字符串
options：恰好五个非空字符串
correct：A、B、C、D、E 之一
```

规范化只允许：

- 去除字段值两端空白；
- 将 `correct` 的小写字母转成大写；
- 去除完整外层 Markdown fence；
- 忽略 JSON object 前后的额外说明文本；
- 复用现有 `training/grpo_v3_json_format.py` 的保守纯语法修复能力。

不得：

- 根据 `answer` 文本猜测 `correct`；
- 将四个或六个选项自动补成五个；
- 根据解释内容改写问题或选项；
- 修改任何字符串内部 token；
- 用 LLM 二次修复 completion；
- 因完整 schema 缺字段而拒绝一个已具备三项核心字段的 QA。

现有三层 JSON 逻辑只复用 object 定位、严格解析、保守修复及审计字段；其 `-0.5`、`-3.0` 格式 reward 不进入本实验。

### 6.2 核心字段不可恢复

若三项核心字段不能按上述规则恢复：

\[
R=-1
\]

此时不调用冻结答题器，并记录 `reward_source=core_qa_unrecoverable_floor`。这是 answer-margin reward 定义域内的有限失败值，不是独立 JSON reward，也不是 infrastructure mask。

以下情况仍属于基础设施错误，必须中止当前 group 或作业：

- completion、evidence ID、packet 或视频映射错位；
- scorer 不可达、模型未加载或真实视频请求失败；
- 任一 A–E 序列分数缺失或非有限；
- permutation 映射不能双向复原；
- reward、margin 或统计产物出现 `NaN/Inf`；
- 同一 reward 调用混入不同 evidence、阶段或实验条件。

## 7. 确定性选项打乱

### 7.1 目的

冻结答题器不能总看到生成器原始的正确标签位置，否则 policy 可能利用固定字母偏置。评分前必须打乱五个选项，并同步映射 `correct`。

### 7.2 稳定算法

不使用 Python 内置 `hash()`，也不依赖进程级随机状态。对每个原始选项索引 \(i\in\{0,1,2,3,4\}\) 计算：

\[
h_i=\operatorname{SHA256}(K\,\|\,\texttt{"\\0"}\,\|\,i)
\]

按 `(h_i, i)` 的字节序升序排列得到 permutation。稳定键 \(K\) 固定包含：

```text
reward_revision
experiment_condition_id
phase
evidence_id
generation_seed 或 reward_call_index
candidate_index
```

训练 trace 和固定评估记录必须同时保存：

- 稳定键各字段；
- 原始选项与原始正确标签；
- permutation 与 inverse permutation；
- 打乱后选项与打乱后正确标签；
- 每个 SHA-256 排序摘要。

相同键必须跨 Python 进程得到相同排列；正确选项文本在映射前后必须完全一致。任何 off-by-one、重复索引或非双射映射都是基础设施失败。

## 8. 冻结答题器 prompt 与标签评分

### 8.1 Prompt 边界

答题器只收到：

```text
两段有序原生视频
问题文本
A–E 五个打乱后的选项
“只选择一个答案标签，不解释”的指令
答案前缀
```

prompt 不得包含生成器声明的 `correct` 标签、原 `answer` 字段、解释、evidence claims、rationale、review 或 self-check。构造完成后需要执行结构化泄露扫描：检查被排除字段名、声明标签以及 question/options 之外的被排除文本是否被拼入。若 `answer` 文本本来就是五个 option 之一，该 option 仍必须正常出现，不能把合法选项误判为泄露。

### 8.2 多 token 标签的正式定义

规范候选集合固定为：

\[
\mathcal{L}=\{A,B,C,D,E\}
\]

对标签 \(k\) 在完整 scorer 上下文后形成的 token 序列
\(y^{(k)}_1,\ldots,y^{(k)}_{T_k}\)，定义：

\[
s_k=\sum_{t=1}^{T_k}
\log p\!\left(y^{(k)}_t\mid x,y^{(k)}_{<t}\right)
\]

其中 \(x\) 是双视频、问题、打乱后选项和答案前缀。使用序列 logprob 求和，不做长度归一化。五个标签的规范字符串、token ID 和 token 数量必须写入 trace。

实现必须验证：

1. prompt 与五个“prompt + 标签”编码存在可审计的公共前缀；
2. 标签 token span 非空；
3. 五个 \(s_k\) 均为有限值；
4. scorer 运行在 `eval` 与 inference-only 模式；
5. scorer 参数 `requires_grad=false`，且不出现在 policy optimizer 参数组中；
6. 同一请求重复评分时，五个序列分数在固定 runtime 下于预定数值容差内一致。

若 tokenizer 边界不能稳定定位，scorer-only preflight 必须失败；不得退化为只读取生成答案的第一个 token。

### 8.3 概率、top-1 与 margin

在五个序列分数上做局部 softmax：

\[
\log p(k)=s_k-\log\sum_{j\in\mathcal{L}}e^{s_j}
\]

设打乱后的正确标签为 \(c\)，原始 margin 为：

\[
M=\log p(c)-\max_{j\ne c}\log p(j)
\]

由于归一化项抵消，数值实现可等价计算：

\[
M=s_c-\max_{j\ne c}s_j
\]

训练 reward 固定为：

\[
R=\frac{\operatorname{clip}(M,-8,8)}{8}
\]

因此 \(R\in[-1,1]\)。截断值 `8` 是 v1 固定常量，不得按 temperature 或运行结果临时调整。

答题 top-1 为 \(\arg\max_k s_k\)。若最大值在 `1e-6` 容差内出现并列，则记录为 tie，正确选项 top-1 指标按未命中处理。

## 9. 在线训练架构与数据流

```text
同一条有序双原生视频 evidence
    ↓
Qwen3-VL-2B policy 按指定 temperature 生成 4 个 completion
    ↓
宽松核心 QA 提取
    ├── 失败：有限 reward = -1，保留 trace
    └── 成功
          ↓
      SHA-256 确定性选项打乱
          ↓
      独立冻结 2B 答题器读取双原生视频、问题、打乱后选项
          ↓
      A–E 五个序列 logprob → raw margin → normalized reward
          ↓
      GRPO 组内相对优势 → policy LoRA optimizer step
```

不得调用旧 repo-native reviewer 的任何 judge 分支，也不得将以下信号加入训练标量：

```text
qa_formality
groundedness
combined_answerability
single-user answerability
leakage
raw completion length
JSON format penalty
人工标签
```

completion length、格式状态和可提取率可以作为审计指标记录，但不能进入 v1 reward。

## 10. 固定训练配置

本轮只执行一个正式 generation temperature 条件，并从已验证 parent adapter 开始：

```text
framework=ms-swift
framework_version=4.2.2
policy=Qwen/Qwen3-VL-2B-Instruct
answer_scorer=Qwen/Qwen3-VL-2B-Instruct
policy_input=ordered_dual_native_video
scorer_input=ordered_dual_native_video
train_type=lora
torch_dtype=bfloat16
target_modules=q_proj,v_proj
lora_rank=8
lora_alpha=16
freeze_vit=true
freeze_aligner=true
num_generations=4
max_steps=40
learning_rate=1e-5
lr_scheduler_type=constant
beta=0.0
top_p=1.0
seed=42
data_seed=42
dataset_shuffle=false
reward_revision=combined_video_answer_margin_v1
margin_clip=8
```

共同 parent 固定为已通过的 Gate 2 作业 `gate2_14119442` 所保存的 `checkpoint-1`。后续实现必须从该作业的 `gate2_result.json`、`run_manifest.json` 和 adapter 文件哈希建立 `checkpoint_inventory.json`；不得仅信任可漂移的 `latest_gate2_output.txt`。禁止使用旧 Gate 3、Gate 3 v2、formality smoke/probe 或其他未批准 adapter 作为 step 0。

正式条件固定为：

| 条件 ID | generation temperature |
|---|---:|
| `t05` | `temperature=0.5` |

`temperature=0.5` 位于原批准范围中间，用于兼顾四候选多样性与结构稳定性。本轮不执行 `0.3` 或 `0.7`；它们只能在 v1 完整结束后作为另立版本的扩展条件。不得根据 calibration、smoke 或训练中间结果临时修改 temperature、学习率、步数、prompt、媒体预算、LoRA、scorer 或 reward。

## 11. 执行门槛

执行顺序是硬门槛：

```text
Gate A：本地纯逻辑与数据契约检查
→ Gate B：冻结答题器 scorer-only 真实双视频 runtime probe
→ Gate C：temperature=0.5 的 scorer calibration
→ Gate D：temperature=0.5 的 1-step 集成 smoke
→ Gate E：temperature=0.5 的 5-step smoke
→ Gate F：temperature=0.5 的 40-step 正式训练
→ Gate G：step 0 / step 40 固定端点评估与结论汇总
```

任一 Gate 失败时，不得绕过。修复基础设施后必须回到能覆盖原失败边界的最小 Gate。

### 11.1 Gate A：本地纯逻辑与数据契约

至少验证：

- 宽松核心字段提取的成功、修复、额外文本和失败边界；
- `options` 数量、空字符串和 `correct` 范围；
- SHA-256 permutation 跨进程稳定且为双射；
- 正确选项文本映射无 off-by-one；
- 多 token 序列 margin、clip 和有限值处理；
- 固定端点配对、bootstrap 和结论状态；
- ms-swift ORM 能稳定收到 `evidence_id`、`packet_json`、`question_type`、`generation_mode` 和 completion；
- 真实四 completion group 的字段展开形状与训练调用一致。

Gate A 只证明本地纯逻辑与静态契约，不证明 Torch scorer 或训练可用。

### 11.2 Gate B：scorer-only runtime probe

先在单张 H100 上启动冻结 2B scorer，不启动 trainer。依次验证：

1. 模型、processor 和两段真实原生视频加载；
2. 一条真实 QA 的五个标签均得到有限序列 logprob；
3. token span、token ID、raw scores、softmax log probabilities 和 top-1 可落盘；
4. scorer 参数全部冻结；
5. 重复请求的 top-1 一致，分数在固定容差内一致；
6. 未向 prompt 泄露 `correct/answer/rationale`；
7. scorer 进程 GPU 身份、峰值显存和耗时有证据。

任何标签分数缺失、视频请求失败或答案泄露均使 Gate B 失败。

### 11.3 Gate C：scorer calibration

`temperature=0.5` 从共同 parent adapter 生成 8 个四候选 group，共 32 个 baseline completion，使用 8 个显式 group seed。

calibration 必须同时满足：

1. 32/32 reward 为有限值；
2. infrastructure mask 为 0；
3. 至少 6/8 group 的 `reward_std > 0`；
4. reward 至少出现两个不同值；
5. 触及 `R=-1` 或 `R=1` clip 边界的候选比例不超过 20%；
6. scorer 的五标签分数、permutation 和核心提取状态完整落盘。

calibration 未通过时，正式训练暂停。不得根据 calibration 结果修改 clip 常量或改试另一个 temperature 来绕过失败。

### 11.4 Gate D：1-step 集成 smoke

`temperature=0.5` 从同一 parent adapter 执行：

- 1 prompt × 4 completions；
- 4/4 reward 有限；
- infrastructure mask 为 0；
- `reward_std > 0`；
- `global_step == 1`；
- LoRA adapter 和 processor 文件完整；
- scorer 与 policy 位于预期 GPU；
- adapter 可真实重载。

### 11.5 Gate E：5-step smoke

`temperature=0.5` 从共同 parent adapter 重新开始，不从 1-step checkpoint 继续。至少要求：

- 5 个完整四候选 group；
- 20/20 reward 有限；
- infrastructure mask 为 0；
- `global_step == 5`；
- 至少 4/5 group 的 `reward_std > 0`；
- scorer trace、trainer state、adapter 和重载证据完整。

### 11.6 Gate F：40-step 正式条件

`temperature=0.5` 再次从共同 parent adapter 重新开始，不从 smoke checkpoint 继续。正式条件必须产生：

- 40 个完整 group；
- 160 条 completion/reward trace；
- 40 个 optimizer step；
- 最终 LoRA adapter、processor、trainer state 和真实重载证据。

在线 reward 曲线作为诊断证据记录，但最终“收敛通过”以 Gate G 的固定端点配对结果为主，避免只用训练期采样曲线下结论。

## 12. 固定端点评估

### 12.1 配对设计

固定评估比较：

- step 0：共同 parent adapter；
- step 40：`temperature=0.5` 的最终 adapter。

使用同一条有序双视频 evidence 和同一组显式的 32 个评估 seed。每个 checkpoint × seed 生成一个 completion，因此必须得到：

\[
2\times32=64
\]

条固定评估记录。step 0 与 step 40 使用完全相同且顺序一致的 32 个 seed。

评估继续执行同一核心提取、permutation、冻结 scorer 和 reward 契约。permutation 稳定键明确包含固定 temperature 与 seed，保证 step 0/40 使用相同选项排列。

### 12.2 固定随机性

每次生成前显式设置 Python、NumPy、PyTorch CPU 和当前 CUDA device seed，并把实际 seed 写入结果。唯一键固定为：

```text
(condition_id, checkpoint_step, seed)
```

重复键、缺失键或额外键都属于评估基础设施失败。固定 seed 只约束当前固定依赖和硬件下的可控随机性，不能表述为跨 GPU、跨驱动或跨依赖版本逐 token 完全一致。

## 13. 指标与统计定义

### 13.1 每个端点的指标

对 step 0 和 step 40 分别输出：

- 完成记录数；
- mean raw answer margin；
- mean normalized reward；
- normalized reward 标准差；
- 正确选项唯一 top-1 命中数与命中率；
- scorer top-1 tie 数量；
- 核心 QA 可提取数量与比例；
- `raw_valid/repaired/extra_text_recovered/unrecoverable` 数量；
- clip 下界、clip 上界和非饱和数量；
- 各失败类型和完整性计数。

### 13.2 配对指标

对同一 seed 定义：

\[
d_i=R_{40,i}-R_{0,i}
\]

输出：

- 平均配对差 \(\overline d\)；
- 以 `1e-6` 为 tie 容差的 win/tie/loss；
- 10,000 次配对 bootstrap 的 95% percentile CI；
- 固定 bootstrap seed `20260721`；
- step 0/40 top-1 命中率差；
- step 0/40 核心 QA 可提取率差。

bootstrap 必须对 32 个配对差按 pair 重采样，不能分别重采样两个端点。

### 13.3 训练期信号指标

每个 40-step 条件还必须输出：

- 40 个 group 的 reward mean/std；
- `reward_std > 0` 的 group 数量与比例；
- 160 条 reward 的有限值与 mask 计数；
- raw margin、normalized reward 和 clip 比例；
- 核心 QA 可提取率；
- completion 长度，仅作审计；
- trainer 可提供的 grad norm、clip、entropy 指标；
- `beta=0.0`，明确记录 KL 被关闭；
- 缺失的 trainer 指标标记为“未测得”，不得伪造数值。

## 14. 通过条件与三层结论

### 14.1 `temperature=0.5` 通过

正式条件必须同时满足：

1. step-40 mean normalized reward 严格高于 step-0；
2. 配对 bootstrap 95% CI 下界严格大于 0；
3. step-40 正确选项 top-1 命中率不低于 step-0；
4. 至少 80% 的 40 个训练 group 满足 `reward_std > 0`；
5. 固定端点评估 64/64 完成，即 step 0 与 step 40 各 32/32；
6. step-40 核心 QA 可提取率相对 step-0 下降不超过 5 个百分点；
7. 160/160 训练 reward 有限，基础设施 mask 为 0；
8. 最终 adapter 保存、真实重载和推理检查通过。

32 对样本下“CI 下界严格大于 0”是较严格门槛，但本版保留已批准标准，不作静默放宽。

### 14.2 单条件结论

`experiment_conclusion` 只能是：

- `passed`：执行完整且八项条件全部满足；
- `not_converged`：执行完整，但至少一项数值/质量通过条件不满足；
- `invalid`：基础设施、完整性、模型身份、数据映射、非有限值或 adapter 证据失败。

`not_converged` 是合法实验结果，Slurm 作业可正常退出；`invalid` 必须非零退出，不能伪装成低 reward 或 `not_converged`。

### 14.3 v1 完成判定

本轮只有一个正式条件。因此，`temperature=0.5` 的 `experiment_conclusion=passed` 即表示 v1 达成“先把 answer-margin GRPO 链路跑通并观察到收敛”的目标；`not_converged` 表示链路有效执行但未证明收敛；`invalid` 表示尚无有效研究结论。

## 15. Reward hacking 与退化审计

训练前后必须导出可读 completion 和 scorer 输入，至少检查：

- 问题是否模板化或异常缩短；
- 五个选项是否重复、空泛或只在表面词形上不同；
- 生成器是否把答案写入 question 或 option 之外的可见位置；
- `correct` 是否长期固定在同一原始位置；
- permutation 后的正确标签是否近似均衡；
- policy 是否通过不可恢复输出大量取得固定 `-1`；
- margin 提高是否仅由错误选项变得荒谬造成；
- scorer top-1 命中提高时，问题和选项是否仍可读。

人工审计可以否决明显退化或泄露，但不增加新 reward，也不能把未满足数值门槛的条件改判为通过。

## 16. 失败分层

| 层级 | 示例 | 允许结论 |
|---|---|---|
| 调度 | `PENDING`、未获 GPU | 只说明资源状态 |
| 基础设施 | scorer 未 ready、视频 forward 失败、存储预检失败 | 尚未验证 reward 或训练 |
| Reward 语义 | clip 大面积饱和、多数组零方差、核心提取率过低 | 当前 reward/采样不适合正式训练 |
| 训练 | 无 optimizer step、非有限 trainer 指标、adapter 缺失 | 更新或保存链路失败 |
| 固定评估 | 记录缺失、seed 键不完整、配对失败 | 不能得出收敛结论 |
| 研究结果 | 单条件完整但未满足收敛门槛 | v1 有效完成但未证明收敛 |

修复后的本地检查只能报告“本地修复完成，等待 Torch runtime 验证”。只有远程作业越过原失败边界，才能报告对应 Gate 通过。

## 17. 存储、环境与资源安全

所有后续 Torch/Slurm 实现必须遵守 `docs/TORCH_EXPERIMENT_META_RULES_CN.md`：

- policy 训练环境和 scorer 推理环境分别审计；
- 加载模型前完成 scratch-first 存储预检；
- 将 HOME、XDG、Hugging Face、ModelScope、Torch、Triton、TorchInductor、vLLM、CUDA、FlashInfer 与临时目录统一指向 `JOB_SCRATCH_ROOT`；
- 使用 `FLASHINFER_WORKSPACE_BASE`，并设置 `VLLM_NO_USAGE_STATS=1`；
- `storage_preflight.json` 失败时不加载任何模型；
- 不使用 `/tmp` 保存正式实验产物；
- 不自动删除历史缓存、checkpoint 或用户文件；
- Markdown 仅供人工阅读，远端训练、预检和测试不得依赖本文件存在。

本实验默认申请两张 H100。资源实测若证明冻结 2B scorer 不能与现有训练配置并行运行，应先报告 GPU 显存、吞吐和失败边界，再另立资源版本；不得在 v1 内切换 scorer 模型或 reward 算法。

## 18. 证据与产物契约

每个 Gate 和正式条件使用独立输出目录。正式 40-step 条件至少保存：

```text
storage_preflight.json
environment_audit.json
cpu_preflight.json
scorer_runtime_probe.json
resolved_config.json
dataset_preview.json
permutation_preflight.json
reward_trace.jsonl
trainer_state.json
training_metrics.json
adapter_reload.json
run_manifest.json
gpu_metrics.csv
dependencies.txt
完整 stdout/stderr 与 scorer 服务日志
LoRA adapter 与 processor 文件
```

每条 reward trace 至少保存：

- schema/reward/experiment version；
- condition、phase、global step、reward call index、candidate index；
- evidence ID、两段视频身份和路径摘要；
- 原始 completion；
- 核心 QA 提取状态、修复操作和规范化字段；
- permutation 稳定键、正向/逆向映射和正确标签映射；
- scorer prompt 的 SHA-256 与泄露扫描结果；
- A–E 标签字符串、token ID、逐 token logprob、序列分数与局部 log probability；
- raw margin、clipped margin、normalized reward、top-1 与 tie 状态；
- reward source、masked/eligible 状态和异常信息。

固定端点评估至少保存：

```text
checkpoint_inventory.json
fixed_eval_results.jsonl
fixed_eval_summary.json
```

`fixed_eval_results.jsonl` 必须严格 64 行。`fixed_eval_summary.json` 必须列出运行完整性、八项通过条件、统一 seed/配置校验和实验三态结论。

只有产物完整性通过时才更新 latest 指针；latest 只表示“本轮证据完整结束”，不表示实验一定通过。

## 19. 预定实现边界

用户批准本规格后，实施计划应覆盖以下独立组件，但具体文件名由实施计划结合现有模块确定：

1. 宽松核心 QA extractor：复用现有 JSON 修复器，不引入第二套冲突语义；
2. SHA-256 确定性 option permutation；
3. 冻结 2B 多模态 answer scorer 与本地服务接口；
4. ms-swift 4.2.2 专用 ORM reward 接线；
5. scorer calibration、1-step、5-step、40-step 的产物验证器；
6. step 0/40 固定配对评估与 bootstrap 汇总；
7. 单条件固定端点评估汇总与三态结论；
8. 遵守 scratch-first 的 Slurm 脚本和中文 Torch Runbook。

允许复用：

- `training/grpo_v3_json_format.py` 的保守解析/修复；
- `training/grpo_v3_reward_plugin.py` 的 ORM 字段展开和 trace 形状；
- `training/grpo_v3_formality_fixed_eval.py` 的 adapter 加载、固定 seed、配对 bootstrap 和 manifest 骨架；
- 现有 storage preflight、adapter reload 和 native-video runner 能力。

禁止：

- 覆盖或重命名既有 Gate 3、formality Probe 或 fixed-eval 文件；
- 修改、删除或重新解释既有实验产物；
- 在旧 reward revision 上静默改语义；
- 让远端入口依赖 Markdown；
- calibration 或 smoke 失败后直接提交 40-step；
- 同时调整 temperature 之外的多个训练变量；
- 把 `not_converged` 写成基础设施失败，或把 `invalid` 写成有效负结果。

## 20. v1 之外的后续路线

v1 成功后，才考虑更贴近双用户研究目标的 information-gap reward：

\[
R_{\mathrm{gap}}=M_{\mathrm{both}}-M_{\mathrm{asker\text{-}only}}
\]

该方案需要额外一次 asker-only 视频评分，成本和噪声更高，不得混入本轮 v1。

若 v1 reward 已确认有方差、无大面积饱和、scorer 稳定但 `temperature=0.5` 仍不能收敛，后续先由用户决定是否扩展到 `temperature=0.3/0.7`，或按以下单变量容量顺序另立版本：

```text
q/v LoRA rank 8
→ all-linear LoRA rank 8
→ all-linear LoRA rank 16
→ 2B 全参数训练
```

不得在本轮直接切换全参数训练，也不得堆叠多个容量升级。

## 21. 本规格的完成边界

本文件通过用户复核后，下一步才使用 `writing-plans` 生成实施计划。当前阶段不实现：

- Python 代码；
- 单元测试或集成测试；
- Slurm 脚本；
- Torch Runbook；
- 远程 scorer、训练或评估作业。

后续本地实施完成也只能表述为“已完成本地静态/单元验证，准备上传 Torch”；实际 scorer、训练和收敛结论必须由对应远程 Gate 与证据产物证明。
