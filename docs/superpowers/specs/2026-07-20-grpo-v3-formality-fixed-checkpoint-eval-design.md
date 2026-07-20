# GRPO v3 `qa_formality` 固定 Checkpoint 配对评估设计

## 1. 背景与目标

40-step `qa_formality_confidence_v1` Probe 已完整执行，但在线 reward 没有通过收敛验收：后 10 组均值低于前 10 组、全程斜率为负，末窗口还出现了一个不可判定候选。训练、160 条 reward trace、LoRA 保存和回载均正常，因此下一步不是重新训练或修改门槛，而是减少采样噪声，检验训练前后 policy 的固定条件差异。

本评估回答唯一问题：

> 在相同输入、相同 16 个随机种子、相同解码参数和相同 `qa_formality` judger 下，Probe 的最终 checkpoint 40 是否比训练前 parent adapter 获得更高的连续 PASS 置信度 reward？

本评估不重新训练，不修改原 Probe 结果，不引入其他 reward，不解锁更大规模训练，也不证明 groundedness、answerability 或综合 QA 质量改善。

## 2. 评估范围

远程清单确认 `--save_total_limit 1` 最终只保留了 checkpoint 40，因此评估以下 2 个 adapter：

- step 0：Probe 使用的 parent adapter；
- step 40：`formality_probe_14377903` 保存的最终 checkpoint。

每个 adapter 使用同一条 Probe native-video evidence 和同一组 16 个固定随机种子。每个“checkpoint × seed”生成一个候选，共得到：

\[
2\times16=32
\]

条独立候选记录。

解码和 reward 边界固定为：

- policy：Qwen3-VL-2B-Instruct 加对应 LoRA；
- 输入：原 Probe 的两段有序 native video；
- temperature：0.7；
- 每次生成只使用当前 seed；
- reward revision：`qa_formality_confidence_v1`；
- 只调用文本 `qa_formality` judger；
- reviewer：Qwen3-VL-8B-Instruct；
- 不调用 video reviewer；
- 不加入 JSON、groundedness、answerability、leakage 或其他 reward。

## 3. 执行架构

采用一个 Slurm 作业顺序评估 step 0 与 step 40。8B reviewer 只启动一次；policy 在同一 GPU 和依赖环境中切换两个 LoRA adapter，避免不同作业环境造成额外差异。

执行顺序为：

```text
统一 scratch 环境与存储预检
→ 数据及 checkpoint 清单预检
→ 启动 reviewer 并通过 /v1/models 与最小文本请求
→ 加载 step 0 adapter
→ 按 16 个固定种子生成、评分并落盘
→ 切换为 step 40 adapter
→ 使用相同 16 个种子生成、评分并落盘
→ 检查结果严格为 32 条
→ 生成配对统计、三态结论和 manifest
```

不会复用现有 greedy eval 作为核心实现，因为 greedy decoding 和完整 repo-native reward 与本问题不一致。可以复用其 LoRA 加载和 native-video runner 模式，但采样、评分和汇总必须使用新的固定评估契约。

## 4. 固定随机性契约

实现中保存长度严格为 16 的显式 seed 列表。每个 checkpoint 必须使用完全相同且顺序一致的列表。

每次生成前同时设置 Python、NumPy、PyTorch CPU 和当前 CUDA device 的 seed，并把实际 seed 写入结果行。若当前生成后端无法保证逐候选 seed 生效，预检必须失败，不能退化为仅设置一次全局 seed。

结果唯一键为：

```text
(checkpoint_step, seed)
```

任何重复键、缺失键或额外键都使基础设施验收失败。

固定 seed 只控制可控的软件随机性；不能据此宣称跨 GPU 型号或跨依赖版本逐 token 完全一致。因此 manifest 还必须记录节点 GPU、关键依赖版本和解码配置。

## 5. Reward 与不可判定契约

每条候选沿用现有 `qa_formality_confidence_v1`：

\[
r=\frac{\operatorname{clip}(\log p(\mathrm{PASS})-\log p(\mathrm{FAIL}),-32,32)}{32}
\]

不可恢复 JSON 或其他按现有 reward 契约定义的不可判定候选，主指标继续记为：

\[
r=-1
\]

不可判定候选不得从主指标中删除、重新调用其他 judger 或替换为同组均值。评估同时报告“仅可判定候选均值”作为敏感性分析，但它不能覆盖总 reward 和不可判定率的主结论。

## 6. 指标

对 step 0 和 step 40 分别输出：

- 候选数，必须为 16；
- 总 reward 均值和标准差；
- 可判定 reward 均值；
- 不可判定数量和比例；
- step 40 相对于 step 0 的逐 seed 配对差均值；
- step 40 相对于 step 0 的胜、平、负 seed 数量。

主端点为：

\[
\Delta R=R_{40}-R_0
\]

对 step 40 与 step 0 的 16 个逐 seed reward 差执行配对 bootstrap。bootstrap 使用固定分析 seed 和固定重采样次数，输出均值差的 95% percentile 置信区间，确保重复分析得到相同结果。

## 7. 三态实验结论

`fixed_eval_summary.json` 的 `experiment_conclusion` 只能取以下三值。

### 7.1 `improved`

必须同时满足：

\[
\Delta R>0
\]

- step 40 不可判定率不高于 step 0；
- step 40 相对 step 0 的配对 bootstrap 95% 置信区间下界大于 0。

### 7.2 `not_improved`

满足任一条件：

- \(\Delta R\le0\)；
- step 40 不可判定率高于 step 0。

仅可判定候选均值上升不能覆盖上述失败条件。

### 7.3 `inconclusive`

满足：

- \(\Delta R>0\)；
- 不可判定率没有上升；
- 但配对 bootstrap 95% 置信区间下界不大于 0。

该状态只允许表述为“方向积极，但 16 个种子的证据不足”。之后是否扩展到 32 个种子必须另行批准。

## 8. 作业状态与实验结论分离

Slurm 作业退出码只表示执行完整性，不表示研究假设是否成立。

以下条件全部满足时，`run_status=passed` 且作业返回 0：

- storage preflight 通过；
- reviewer readiness 通过；
- 2 个 adapter 均完整并成功加载；
- 32 个候选全部生成和评分；
- reward 全部为有限值；
- 结果键严格覆盖 2×16 笛卡尔积；
- 只出现 `qa_formality_confidence` reward；
- 没有基础设施 mask；
- summary 和 manifest 成功写入。

`experiment_conclusion=not_improved` 是合法实验结果，不导致非零退出码。

模型加载失败、reviewer 失败、reward 缺失或非有限、结果不足 32 条、重复键、checkpoint 清单不完整等属于基础设施失败，必须返回非零退出码；不得生成伪 reward 或把缺失候选当作低 reward 继续汇总。

## 9. 存储安全

新 Slurm 作业遵守 `docs/TORCH_EXPERIMENT_META_RULES_CN.md`：

- 作业只接收一个 `JOB_SCRATCH_ROOT`；
- 在加载 reviewer 和 policy 前，将 HOME、XDG、Hugging Face、ModelScope、Torch、Triton、TorchInductor、vLLM、CUDA、FlashInfer 和临时目录全部定向到该 scratch 根下；
- 使用 `FLASHINFER_WORKSPACE_BASE`；
- 设置 `VLLM_NO_USAGE_STATS=1`；
- 在模型加载前生成并验收 `storage_preflight.json`；
- 不把正式产物写入共享 `/tmp`；
- 不自动删除历史缓存、模型、adapter 或实验结果。

## 10. 产物契约

独立输出目录至少包含：

```text
storage_preflight.json
cpu_preflight.json
resolved_config.json
checkpoint_inventory.json
fixed_eval_results.jsonl
fixed_eval_summary.json
run_manifest.json
gpu_metrics.csv
dependencies.txt
```

`fixed_eval_results.jsonl` 严格为 32 行。每行至少保存：schema version、checkpoint label、checkpoint step、adapter path、seed、evidence ID、视频路径、解码配置、原始 completion、reward、是否不可判定、reward record 和 judge trace。

`fixed_eval_summary.json` 保存 step 0 和 step 40 的统计、逐 seed 配对结果、bootstrap 区间及三态结论。

`checkpoint_inventory.json` 保存 2 个 adapter 的绝对路径、必要文件存在性和 adapter 配置摘要。

`resolved_config.json` 保存模型路径、数据路径及 SHA-256、明确的 16 个 seed、temperature、生成长度、reward revision、reviewer 配置和依赖边界。

`run_manifest.json` 分开保存：

```text
run_status
experiment_conclusion
artifact_paths
row_count
checkpoint_count
seed_count
storage_preflight_status
```

只有基础设施验收通过且 32 行完整时，才更新 `latest_formality_fixed_eval_output.txt`。指针表示“评估完整结束”，不表示结论一定为 `improved`。

## 11. 测试策略

实现严格采用测试先行。自动测试至少覆盖：

- step 0 和 step 40 复用同一组 16 个 seed；
- 结果键必须严格覆盖 2×16；
- 重复键、缺失键和非有限 reward 被拒绝；
- 不可判定候选保留在总 reward 中；
- 仅可判定均值只作为敏感性指标；
- `improved`、`not_improved`、`inconclusive` 三种结论；
- 配对 bootstrap 在固定分析 seed 下可复现；
- step 40 不可判定率上升时不能判为 improved；
- manifest 将运行状态和实验结论分开；
- Slurm 包含完整 scratch-first 环境、预检、32 行断言和失败硬停止；
- Bash 静态语法检查通过。

本地测试只能证明纯逻辑、静态配置和脚本语法，不能宣称 Torch GPU、真实视频生成或 reviewer runtime 已通过。

## 12. 执行与完成边界

实施阶段只新增专用评估模块、测试、Slurm 文件和独立实验 Runbook，避免修改当前已有未提交变化的 Probe 训练逻辑。

本地完成条件：

- 新增测试先失败、实现后通过；
- formality 专项测试通过；
-完整 GRPO v3 测试通过；
- `compileall`、`bash -n` 和 `git diff --check` 通过。

远程完成条件：

- 远程静态预检通过；
- Slurm 作业完成且 `run_status=passed`；
- `fixed_eval_results.jsonl` 严格 32 行；
- step 0 和 step 40 各 16 行；
- summary、manifest 和 reviewer 日志完整；
- 最终报告明确区分基础设施状态和 `improved`、`not_improved` 或 `inconclusive` 实验结论。
