# GRPO v3 `qa_formality` 固定端点评估实施计划

> **执行要求：** 使用 `executing-plans` 按任务顺序内联执行。每个生产行为必须先有失败测试，再写最小实现。

**目标：** 在不重新训练的前提下，对 Probe 的 parent adapter（step 0）和 `checkpoint-40` 使用同一组 16 个固定种子，生成 32 条 `qa_formality_confidence_v1` 配对评估记录并给出三态结论。

**架构：** 新增一个独立 Python 模块负责 checkpoint 清单、固定 seed 采样、formality-only 评分、配对 bootstrap 和产物写入；新增一个双 GPU Slurm 作业，在 GPU 0 上加载一次 2B policy 并切换两个 LoRA，在 GPU 1 上启动一次 8B reviewer。运行完整性与实验结论分离。

**技术栈：** Python 3.11、unittest、PyTorch、Transformers、PEFT、现有 Qwen3-VL runner、现有 `qa_formality_confidence_v1` scorer、Bash/Slurm、vLLM。

---

### 任务 1：固定端点评估的纯统计契约

**文件：**

- 新增：`tests/training/test_grpo_v3_formality_fixed_eval.py`
- 新增：`training/grpo_v3_formality_fixed_eval.py`

- [ ] **步骤 1：先写失败测试**

测试使用 step 0 和 step 40、各 16 个相同 seed 的人工记录，覆盖：严格 2×16 键空间、不可判定保留在总 reward、可判定均值、配对胜平负、固定 bootstrap 可复现，以及 `improved`、`not_improved`、`inconclusive` 三态结论。另写缺失键、重复键、非有限 reward、额外 reward component、基础设施 mask 的拒绝测试。

- [ ] **步骤 2：运行测试并确认按预期失败**

运行：

```bash
python -m unittest tests.training.test_grpo_v3_formality_fixed_eval -v
```

预期：因 `training.grpo_v3_formality_fixed_eval` 尚不存在而失败。

- [ ] **步骤 3：实现最小统计 API**

模块提供：

```python
CHECKPOINT_STEPS = (0, 40)
FIXED_SEEDS = tuple(range(2026072000, 2026072016))
BOOTSTRAP_SEED = 20260720

def analyze_fixed_eval(
    rows: Iterable[dict[str, Any]],
    *,
    checkpoint_steps: Sequence[int] = CHECKPOINT_STEPS,
    seeds: Sequence[int] = FIXED_SEEDS,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_replicates: int = 10000,
) -> dict[str, Any]: ...
```

主结论严格执行已批准规格：delta 非正或不可判定率上升为 `not_improved`；方向和安全条件通过但置信区间跨 0 为 `inconclusive`；置信区间下界也大于 0 才是 `improved`。

- [ ] **步骤 4：运行测试并确认通过**

运行同一步骤 2；预期全部通过。

### 任务 2：Checkpoint 清单、固定 seed 生成和 formality-only 评分

**文件：**

- 修改：`tests/training/test_grpo_v3_formality_fixed_eval.py`
- 修改：`training/grpo_v3_formality_fixed_eval.py`

- [ ] **步骤 1：先扩展失败测试**

新增测试验证：

- 从 Probe `resolved_config.json` 找到 parent run，再从 parent `run_manifest.json` 读取 step 0 adapter；
- Probe 内必须恰好找到一个完整 `checkpoint-40`；
- adapter 必须同时有 `adapter_config.json` 和权重文件；
- 两个 checkpoint 使用完全相同、顺序一致的 16 个 seed；
- 每个候选生成前重设 Python、NumPy、PyTorch CPU/CUDA seed；
- runner 使用 `decoding_mode="sampling"`、`temperature=0.7`、`top_p=1.0` 和两段有序视频；
- scorer 结果包含 mask、非有限 reward 或其他 reward component 时立即失败；
- 每条结果保留 completion、完整 record、judge trace、adapter 路径和 seed。

- [ ] **步骤 2：运行测试并确认新测试失败**

运行：

```bash
python -m unittest tests.training.test_grpo_v3_formality_fixed_eval -v
```

预期：缺少 inventory、seed 和 evaluate API。

- [ ] **步骤 3：实现最小评估 API 与 CLI**

模块新增：

```python
def build_checkpoint_inventory(probe_dir: Path) -> dict[str, Any]: ...
def set_generation_seed(seed: int, *, torch_module: Any | None = None) -> None: ...
def evaluate_adapter(..., checkpoint_step: int, seeds: Sequence[int], temperature: float) -> list[dict[str, Any]]: ...
def load_multi_adapter_runner(..., adapters: Mapping[int, Path]) -> Any: ...
```

CLI 接受 `--probe-dir`、`--dataset`、`--model-path`、`--review-model`、`--review-base-url`、`--output-dir` 等参数。它只加载一次 base policy，通过 PEFT 为 step 0 和 step 40 注册两个 adapter；依次评估后写出 `checkpoint_inventory.json`、`resolved_config.json`、`fixed_eval_results.jsonl`、`fixed_eval_summary.json` 和 `run_manifest.json`。

`run_manifest.json.run_status` 只在 32 行及所有基础设施断言通过后为 `passed`，并单独复制 `experiment_conclusion`。

- [ ] **步骤 4：运行测试并确认通过**

运行同一步骤 2；预期全部通过。

### 任务 3：Slurm 运行契约

**文件：**

- 新增：`tests/training/test_grpo_v3_formality_fixed_eval_slurm.py`
- 新增：`hpc/grpo_v3_formality_fixed_eval.sbatch`

- [ ] **步骤 1：先写失败的静态契约测试**

测试要求 Slurm 文件包含：双 L40S、`FORMALITY_PROBE_DIR`、完整 scratch-first 环境变量、`FLASHINFER_WORKSPACE_BASE`、`VLLM_NO_USAGE_STATS=1`、模型加载前 storage preflight、reviewer `/models` 和最小文本请求、formality fixed eval CLI、32 行及两个 checkpoint 各 16 行断言、`run_status=passed` 断言和成功后 pointer 更新。

测试同时禁止：训练命令、完整 repo-native reward、video reviewer、`FLASHINFER_WORKSPACE_DIR`、根据 `experiment_conclusion` 返回非零退出码。

- [ ] **步骤 2：运行测试并确认失败**

```bash
python -m unittest tests.training.test_grpo_v3_formality_fixed_eval_slurm -v
```

预期：Slurm 文件尚不存在。

- [ ] **步骤 3：实现 Slurm 作业**

作业先定位 Probe、dataset、两个 adapter 并运行 CPU/storage preflight；GPU 1 启动 8B reviewer，GPU 0 执行固定端点评估。结束后用 Python 断言 32 行、step 0/40 各 16 行和 `run_status=passed`，再更新：

```text
outputs/grpo_v3/latest_formality_fixed_eval_output.txt
```

`not_improved` 和 `inconclusive` 均正常退出 0。

- [ ] **步骤 4：运行测试及 Bash 静态检查**

```bash
python -m unittest tests.training.test_grpo_v3_formality_fixed_eval_slurm -v
bash -n hpc/grpo_v3_formality_fixed_eval.sbatch
```

预期全部通过。

### 任务 4：人工可复制的 Torch 实验文档

**文件：**

- 新增：`docs/GRPO/v3/experiments/qa_formality_fixed_checkpoint_eval_v1/README_CN.md`
- 新增：`docs/GRPO/v3/experiments/qa_formality_fixed_checkpoint_eval_v1/TORCH_RUNBOOK_CN.md`

- [ ] **步骤 1：编写实验说明和 Runbook**

README 记录问题、step 0/40 边界、32 条配对设计、三态结论和不能外推的结论。Runbook 顶部链接 `docs/TORCH_EXPERIMENT_META_RULES_CN.md`，提供可整块复制到 Torch 的：上传清单、登录节点静态检查、checkpoint 清单、提交、监控、安全验收、失败证据收集和 SFTP 下载命令。

- [ ] **步骤 2：静态扫描文档契约**

```bash
rg -n "formality_probe_14377903|checkpoint-40|32|16|run_status|experiment_conclusion|JOB_SCRATCH_ROOT|latest_formality_fixed_eval_output" docs/GRPO/v3/experiments/qa_formality_fixed_checkpoint_eval_v1
```

预期所有关键边界均出现，且文档不要求上传 Markdown 才能运行代码。

### 任务 5：完整本地验证

**文件：** 无新增文件。

- [ ] **步骤 1：运行专项测试**

```bash
python -m unittest discover -s tests/training -p 'test_grpo_v3_formality_fixed_eval*.py' -v
```

- [ ] **步骤 2：运行 formality 与完整 GRPO v3 测试**

```bash
python -m unittest discover -s tests/training -p 'test_grpo_v3_formality_*.py' -v
python -m unittest discover -s tests/training -p 'test_grpo_v3_*.py' -v
```

- [ ] **步骤 3：运行语法与 diff 检查**

```bash
python -m compileall training tests/training
bash -n hpc/grpo_v3_formality_fixed_eval.sbatch
git diff --check
```

只把上述结果报告为本地逻辑和静态验证，不宣称远程 Torch runtime 已通过。

### 任务 6：远程执行边界

**文件：** 无本地代码修改。

- [ ] **步骤 1：探测远程连接能力**

若当前 Codex 环境可以连接 Torch，则先执行只读环境和路径检查，再同步新增代码并提交作业；若无法解析或认证 Torch 登录节点，则停止在“本地实现完成、远程待人工复制执行”，不伪造远程结果。

- [ ] **步骤 2：若可连接则执行并监控**

远程作业必须越过 storage preflight、reviewer readiness、真实 native-video generation、32 条评分和产物验收。只有实际产物存在时才报告三态结论。

