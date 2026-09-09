# 六用户长视频 QA 推理优化版本与效果记录

更新时间：2026-09-09

## 1. 文档目的

本文持续记录六用户十分钟 QA 推理链路的每个主要版本、实现变化、运行证据、性能效果、质量边界和后续决策。所有百分比均绑定具体运行；单样本结果不表述为统计显著或总体质量结论。

## 2. 当前固定任务合同

- 模型：Qwen3.8-27B，BF16。
- 输入：一个 speaker 与五个 provider，共六个同步十分钟视角。
- Generator：speaker 使用全部 sampled clustering frames，provider 使用正式 pruning metadata 保留的 frames。
- Judge：使用完整原视频或正式 sequential source-segment 合同。
- QA：保留 schema、qa_formality、evidence_groundedness、speaker-only、all-six、minimum-set 和自动 accepted/rejected 语义。
- 性能结论必须区分模型初始化、媒体处理、prefill、decode、finalizer、review batching 和端到端 wall time。

## 3. 版本历史

### V0：Transformers memory-safe 串行后端

主要实现：

- 单个常驻 Transformers runner。
- `_inference_lock` 串行化视频解码、processor 和 `model.generate`。
- FlashAttention 2 只覆盖标准 attention。
- 通过 FPS、像素、输入 token ceiling 和 KV reserve 控制峰值显存。

已知问题：

- Python three-judge 线程存在，但底层 inference lock 使模型调用串行。
- 长序列 Qwen3.5/Qwen3.8 曾在约 70k input tokens 时出现接近整卡显存的峰值。
- FlashAttention 不能单独优化 Gated DeltaNet 层。

证据边界：这是历史后端与问题来源，不是本轮 H20 A/B 的直接数值基线。

### V1：vLLM、FlashAttention、FlashInfer 与 continuous batching

本地提交说明：`feat: 六用户推理切换到 vLLM 和 FlashAttention`

主要实现：

- 新增常驻 `Qwen3VLLocalVLLMRunner`。
- vLLM paged KV cache、chunked prefill 和 continuous batching。
- 文本、视觉 encoder attention 使用 FlashAttention。
- Gated DeltaNet prefill 使用 FlashInfer。
- MTP speculative decoding 使用一个 draft token。
- Three-judge batch barrier 将 formality、groundedness、answerability 首批请求一起交给 scheduler。
- 多模态 processor cache identity 包含文件身份、模态、FPS 与像素预算。
- 每阶段 `GenerationCallProfile` 独立控制输出预算、thinking、FPS 和像素。

H20 真实运行环境：

- GPU：单张 NVIDIA H20，97,871 MiB。
- PyTorch：2.13.0+cu129。
- vLLM：0.28.0+cu129。
- FlashAttention：2.8.3.post1，运行日志实际选择 FlashAttention 3 kernel。
- FlashInfer：0.6.16.post3。
- 模型权重：51.87 GiB。
- KV cache：22.64 GiB，342,199 tokens。
- 峰值显存：81,829 MiB。
- Three-judge：真实 `batch_size=3`。
- MTP：大多数采样窗口的接受率约 76%–98%。

结论：vLLM、FlashAttention、FlashInfer、MTP、真实 sampled-frame generator、六视频 judges 与 answerability 已在 H20 上完成一个真实 slot。

### V1.1：vLLM IPC socket 短路径修复

本地提交说明：`fix: 缩短 vLLM IPC socket 路径`

真实失败：

```text
ZMQError: ipc path is longer than 107 characters
```

原因：job-specific `TMPDIR` 加 vLLM UUID 后超过 Unix socket 路径长度。

修复：

- 为 vLLM 创建短 `/tmp/egoqa_vllm_${SLURM_JOB_ID}` 符号链接。
- 链接目标仍是 job-specific scratch 的 `tmp`。
- Storage preflight 解析后仍确认实际写盘位于 allowed root。
- 清理仅删除目标完全匹配的符号链接，不删除 scratch。

效果：H20 真实运行越过原 ZMQ 初始化失败，完成模型加载和整个 one-slot。

### V1.2：Generator finalizer 改为 text-only

本地提交说明：`perf: 避免生成 finalizer 重复编码媒体`

瓶颈：reasoning 已读取 1,982 张真实 candidate frames，finalizer 只负责把 reasoning draft 整理成 schema JSON，却再次携带同一批图片，触发第二次长多模态 prefill。

代码变化：

- Generator reasoning 保留全部媒体和原输出预算。
- Generator finalizer 只接收原任务与 reasoning draft。
- 不改变 sampled frames、像素、sampling、QA schema、judges、answerability 或 acceptance gate。

H20 成对单槽结果：

| 指标 | 基线 | 优化后 | 变化 |
|---|---:|---:|---:|
| Finalizer 图片数 | 1,982 | 0 | 移除重复媒体 |
| Finalizer prompt tokens | 174,738 | 10,232 | -94.1% |
| Finalizer latency | 202.1 s | 16.5 s | -91.8% |
| Generation latency | 564.3 s | 353.1 s | -37.4% |
| End-to-end wall time | 1,320 s | 915 s | -30.7% |
| Peak GPU memory | 81,829 MiB | 81,829 MiB | 无增加 |
| Three-judge batch size | 3 | 3 | 保持 |

端到端 wall time 受到第二次运行 NFS page cache 已变热的影响，不能全部归因于代码；finalizer token 与 latency 是最干净的阶段级信号。

质量边界：

- 两次生成相同选项和相同声明答案，问题措辞轻微不同。
- 两次均为 qa_formality PASS。
- 两次均因酒瓶标签在六个视角中不可辨认而被 evidence_groundedness 与 answerability 拒绝。
- 这证明 gate 行为没有被放松，不证明总体 QA 质量等价。

远端证据：

```text
/nas/egoqa_xuerong_h20/outputs/h20_one_pass_baseline_20260909T012000CST
/nas/egoqa_xuerong_h20/outputs/h20_one_pass_optimized_20260909T014500CST
/nas/egoqa_xuerong_h20/outputs/h20_one_pass_optimized_20260909T014500CST/optimization_comparison.json
```

## 4. Reasoning 收益与成本实验

### 4.1 两个源版本

Non-reasoning 最新版本：

```text
C:\Users\20661\Desktop\Research\AR\multiuser\multi-user.zip
```

Reasoning 版本：

```text
C:\Users\20661\Desktop\Research\AR\multiuser\EgoQA-two-user-multi-user-prompt
branch: feature/six-user
```

参考提交脚本：

```text
C:\Users\20661\Desktop\Research\AR\multiuser\run_six_user_qa_1h200_throughput_probe_0p5.sbatch
C:\Users\20661\Desktop\Research\AR\multiuser\run_six_user_qa_10min_sequential_0p5_fresh30.sbatch
```

### 4.2 不能直接比较两个源版本

`multi-user.zip` 除了 non-reasoning，还包含当前 reasoning 仓库尚未完整具备的 async vLLM server、跨 packet 并发、sequential separated fact audit 和更新后的生产参数。若分别运行 ZIP 与当前仓库，差异将同时来自：

- reasoning 与 token budget；
- in-process runner 与 async server；
- packet 并发宽度；
- judge 调度模式；
- runtime 和产物格式。

这种比较无法把性能或质量变化归因于 reasoning。

### 4.3 正确 A/B 设计

以 `multi-user.zip` 为统一代码基线，把当前 reasoning 机制移植成一个显式开关，并同时保留 V1.2 text-only finalizer 与 V1.1 IPC 修复。两个条件必须使用同一份合并后代码。

固定不变：

- 固定十候选 cohort，来源为 Job `17109425` 的 shared preprocessing。
- 10 candidates，最多 3 attempts。
- Qwen3.8-27B BF16。
- 单张 H200、24 CPU、480G RAM、8 小时上限。
- `JUDGE_VIDEO_FPS=0.50`。
- `GENERATOR_MAX_IMAGE_PIXELS=91728`。
- 相同 vLLM、FlashAttention、FlashInfer、MTP、prefix cache 和 server 参数。
- 相同 sampling seed、question schedule、judges、schema 和 acceptance gate。
- 相同 packet concurrency：`MAX_PACKETS_IN_FLIGHT=3`、`MAX_REVIEW_LANES=2`。
- 每个条件独立 job-specific scratch、输出和 JobID manifest。

唯一 treatment：

| 条件 | Generator/Judge 行为 |
|---|---|
| NR | thinking disabled；每阶段单次结构化生成；沿用 non-reasoning 的 2,048-token 上限 |
| R | reasoning draft + text-only finalizer；使用当前 reasoning 分阶段预算 |

不得通过降低 R 的 FPS、像素、candidate 数、attempt 数或 judge 数来隐藏 reasoning 时间成本。

### 4.4 必须比较的指标

性能：

- engine 初始化与 compile 时间，单独报告，不混入 reasoning 结论。
- 每个 completed slot 的 wall time。
- 每个 accepted QA 的总时间；accepted 为 0 时写不可计算。
- generator reasoning、generator finalizer、formality、groundedness、speaker-only、all-six、minimum-set 的调用次数、tokens 和 latency。
- GPU 峰值显存、平均利用率、高利用率时间占比。
- vLLM batch size、prompt throughput、generation throughput、MM cache hit、MTP acceptance。

质量：

- attempted、completed、accepted、rejected、infrastructure-skipped。
- qa_formality、evidence_groundedness、answerability 各阶段通过率。
- all-six sufficient、speaker-only insufficient、minimum-set determined 的数量。
- schema/JSON repair 率。
- 问题重复率。
- 每个固定 candidate 的最终状态和拒绝原因。
- 对两组生成的有效 QA 做盲人工审查时，分别记录可回答性、证据真实性和问题自然度；不能把自动 accepted 当人工 gold。

### 4.5 Reasoning 保留或丢弃规则

十候选结果只作为方向性证据，不声称统计显著。建议使用以下工程决策规则：

保留 reasoning 进入更大评估，需至少满足一项质量收益：

- accepted 数比 NR 多至少 2/10；或
- groundedness 与 answerability 联合通过数提高至少 2/10；或
- 在可配对人工审核题中，R 的盲评胜率达到至少 70%。

同时报告成本，不以质量收益掩盖时间：

- 若 R 的 completed-slot 中位时间小于 NR 的 1.5 倍，成本可接受；
- 1.5–2.0 倍需要明确质量收益才能继续；
- 超过 2.0 倍必须先优化 reasoning budget，再考虑扩大实验。

丢弃当前 reasoning 配置：

- accepted 或联合 gate 通过数变化在 ±1/10 内，且 completed-slot 中位时间增加至少 50%；或
- R 的 schema repair、infrastructure failure 或 OOM 明显增加；或
- 人工审核没有稳定偏好，而时间达到 NR 的两倍以上。

若丢弃，只移除 reasoning draft + finalizer treatment；保留已经独立验证有效的 vLLM、FlashAttention、FlashInfer、MTP、batching、text-only structural finalization 和 IPC 修复。

## 5. 当前优化判断与下一优先级

已确认有效并保留：

1. vLLM 常驻引擎与 continuous batching。
2. FlashAttention 和 FlashInfer GDN prefill。
3. MTP=1，目前接受率为积极信号。
4. Generator text-only finalizer。
5. 短 IPC socket 路径。

下一优先级取决于 reasoning A/B：

- 若 reasoning 无明显质量收益，直接移除 reasoning treatment。
- 若 reasoning 有收益但成本过高，优先缩短 reasoning output budget，因为 H20 中 generator reasoning 打满 6,144 tokens，耗时约 337–362 秒。
- 每次只调整一个 reasoning budget，使用同一固定 cohort 验证。
- 不把启动时 NFS page cache 或 compile cache 命中算作 reasoning 加速。

## 6. 当前证据边界

- H20 A/B 只有一个正式 candidate、每个条件一次运行。
- 两个 QA 均 rejected，不能据此计算 accepted-QA 平均时间。
- 已证明工程链路与 text-only finalizer 性能改善。
- 尚未证明 reasoning 对十候选 QA 质量的收益。
- 尚未运行统一代码基线下的 NR/R H200 配对任务。
