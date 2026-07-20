# GRPO v3 仅 qa_formality 连续置信度 Reward 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现只由冻结 `qa_formality` judge 的 PASS/FAIL logprob margin 产生 reward 的 GRPO v3 最小收敛实验，并提供人工可直接复制到 Torch 执行的完整 runbook。

**Architecture:** 新建独立的 formality score、离线回放、收敛分析和实验验收模块，仅在现有 ms-swift reward plugin 中增加一个独立 ORM 注册项。Smoke 和 40-step probe 从同一个已通过 Gate 2 adapter 分别启动，复用原生双视频 policy 输入，但 reviewer 只运行纯文本 `qa_formality` 分支；旧 Gate 3、Gate 3 v2 和 repo-native reward 行为保持不变。

**Tech Stack:** Python 3、`unittest`、ms-swift 4.2.2 ORM plugin、Qwen3-VL、vLLM、Slurm、JSON/JSONL、PowerShell 本地验证、Bash Torch runbook。

---

## 0. 文件结构与职责

**新建文件：**

- `training/grpo_v3_formality_reward.py`：logprob margin 映射、单一 judge score function、严格故障边界。
- `training/grpo_v3_formality_replay.py`：读取历史 trace，离线重建 formality reward 和组内方差报告。
- `training/grpo_v3_formality_convergence.py`：分析 smoke/probe trace、窗口增量、线性 slope 和硬验收条件。
- `training/grpo_v3_formality_artifacts.py`：验证 adapter/processor/trainer 产物并生成 formality 专用 result、manifest 和 resolved config。
- `hpc/grpo_v3_formality_smoke.sbatch`：从 Gate 2 adapter 启动 1-step smoke。
- `hpc/grpo_v3_formality_probe.sbatch`：从同一 Gate 2 adapter 独立启动 40-step probe。
- `tests/training/test_grpo_v3_formality_reward.py`：reward 公式、judge 调用、不可恢复候选和故障测试。
- `tests/training/test_grpo_v3_formality_plugin.py`：ms-swift ORM 展开、trace 和异常传播测试。
- `tests/training/test_grpo_v3_formality_replay.py`：历史 trace 回放和 SHA-256 测试。
- `tests/training/test_grpo_v3_formality_convergence.py`：窗口、slope、正方差、退化和失败分类测试。
- `tests/training/test_grpo_v3_formality_artifacts.py`：smoke/probe 产物与 manifest 契约测试。
- `tests/training/test_grpo_v3_formality_slurm.py`：两个 Slurm 脚本的参数和父 adapter 契约测试。
- `docs/GRPO/v3/experiments/qa_formality_only_convergence_v1/README_CN.md`：实验入口和执行顺序。
- `docs/GRPO/v3/experiments/qa_formality_only_convergence_v1/TORCH_RUNBOOK_CN.md`：SFTP、回放、提交、监控、验收、下载命令。
- `docs/GRPO/v3/experiments/qa_formality_only_convergence_v1/RESULT_INTERPRETATION_CN.md`：允许结论、禁止结论和失败分类。

**修改文件：**

- `training/grpo_v3_reward_plugin.py`：增加 `FormalityConfidenceReward`，注册 `egoqa_qa_formality_confidence`；不改变原两个 ORM。

---

### Task 1：实现连续 formality reward 纯函数和单一 judge score

**Files:**

- Create: `training/grpo_v3_formality_reward.py`
- Create: `tests/training/test_grpo_v3_formality_reward.py`

- [ ] **Step 1：先写 reward 数学契约的失败测试**

在测试文件中定义以下核心用例：

```python
import math
import unittest

from training.grpo_v3_formality_reward import confidence_reward


class FormalityConfidenceMathTests(unittest.TestCase):
    def test_margin_is_clipped_and_scaled_to_unit_interval(self) -> None:
        self.assertEqual(confidence_reward(40.0, 0.0), 1.0)
        self.assertEqual(confidence_reward(-40.0, 0.0), -1.0)
        self.assertEqual(confidence_reward(-2.0, -10.0), 0.25)

    def test_nonfinite_logprob_is_infrastructure_error(self) -> None:
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "非有限"):
                    confidence_reward(value, 0.0)
```

- [ ] **Step 2：运行测试并确认因模块不存在而失败**

Run:

```powershell
python -m unittest tests.training.test_grpo_v3_formality_reward -v
```

Expected: `ModuleNotFoundError: No module named 'training.grpo_v3_formality_reward'`。

- [ ] **Step 3：实现固定的 margin 映射**

在新模块中加入：

```python
FORMALITY_REWARD_REVISION = "qa_formality_confidence_v1"
FORMALITY_MARGIN_CLIP = 32.0
FORMALITY_COMPONENT = "qa_formality_confidence"
UNJUDGEABLE_FORMALITY_REWARD = -1.0


def confidence_reward(pass_logprob: float, fail_logprob: float) -> float:
    values = (float(pass_logprob), float(fail_logprob))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("qa_formality PASS/FAIL logprob 包含非有限值")
    raw_margin = values[0] - values[1]
    clipped_margin = max(-FORMALITY_MARGIN_CLIP, min(FORMALITY_MARGIN_CLIP, raw_margin))
    return clipped_margin / FORMALITY_MARGIN_CLIP
```

- [ ] **Step 4：补写 score function 的失败测试**

构造 fake modules 和 fake runner，测试：

```python
def valid_completion() -> str:
    return json.dumps({
        "question": "Which mug was still on the counter after I left?",
        "options": ["red", "blue", "green", "white", "black"],
        "correct": "white",
    })


class FakeRunner:
    def generate_with_choice_logits(self, prompt, **kwargs):
        return {
            "text": json.dumps({
                "review_passed": True,
                "checks": {"qa_formality": {"status": "PASS"}},
                "blocking_failures": [],
            }),
            "choice_logits": {
                "available": True,
                "choice_logprobs": {"PASS": -2.0, "FAIL": -10.0},
            },
        }
```

断言合法候选得到 `0.25`，且：

```python
self.assertEqual(result["record"]["reward_components"], {"qa_formality_confidence": 0.25})
self.assertEqual(result["record"]["reward_source"], "judge_pass_fail_logprob_margin")
self.assertEqual(result["record"]["judge_trace"].keys(), {"qa_formality"})
self.assertFalse(result["record"]["masked"])
```

再断言完全不可恢复 completion 得到 `-1.0`、`judge_called=False`、component 集合不变；judge 输出无效 JSON、缺失 choice logprob 或 runner 异常时抛出异常，而不是返回 `-1.0`。

- [ ] **Step 5：实现 `make_formality_score_fn()`**

实现边界固定为：

```python
def make_formality_score_fn(
    *,
    review_model_id: str,
    review_base_url: str,
    policy_model_id: str,
    review_max_new_tokens: int,
    modules: dict[str, Any] | None = None,
) -> Callable[..., dict[str, Any]]:
    repo_modules = _formality_modules() if modules is None else modules
    reviewer = repo_modules["OpenAICompatibleLocalRunner"](
        model_id=review_model_id,
        base_url=review_base_url,
        max_new_tokens=review_max_new_tokens,
        timeout=900,
        allow_video_input=False,
    )
    return _build_score_closure(
        reviewer=reviewer,
        policy_model_id=policy_model_id,
        modules=repo_modules,
    )
```

`modules` 必须提供：

```python
{
    "OpenAICompatibleLocalRunner": runner.OpenAICompatibleLocalRunner,
    "build_qa_formality_judge_prompt": prompts.build_qa_formality_judge_prompt,
    "run_model_judge_branch": video_loop.run_model_judge_branch,
    "qa_for_judger_prompt": video_loop.qa_for_judger_prompt,
    "validate_qa_item": schema.validate_qa_item,
    "complete_generator_metadata": video_loop.complete_generator_metadata,
}
```

合法/可修复候选的最小执行顺序必须写成：

```python
format_result = validate_completion_json(raw_completion)
if format_result.status == "unrecoverable":
    return {
        "reward": UNJUDGEABLE_FORMALITY_REWARD,
        "record": {
            "candidate_id": candidate_id,
            "group_id": evidence_id,
            "evidence_id": evidence_id,
            "masked": False,
            "eligible_for_grpo": True,
            "qa_formality_status": "FAIL",
            "reward_source": "deterministic_unjudgeable_floor",
            "judge_called": False,
            "reward_components": {
                FORMALITY_COMPONENT: UNJUDGEABLE_FORMALITY_REWARD,
            },
            "reward_total": UNJUDGEABLE_FORMALITY_REWARD,
            "format_validation": format_result.to_dict(),
        },
    }
qa = dict(format_result.value or {})
qa["qa_id"] = str(qa.get("qa_id") or f"GRPO_V3_FORMALITY_{evidence_id}_{candidate_index}")
qa["question_type"] = question_type
qa["generation_mode"] = generation_mode
qa["required_users"] = list(packet.get("required_users") or qa.get("required_users") or [])
qa["model_id"] = policy_model_id
modules["complete_generator_metadata"](qa, packet=packet, question_type=question_type)
schema_errors = modules["validate_qa_item"](qa)
prompt = modules["build_qa_formality_judge_prompt"](
    modules["qa_for_judger_prompt"](qa), packet, schema_errors=schema_errors
)
judge = modules["run_model_judge_branch"](
    check_name="qa_formality",
    prompt=prompt,
    runner=reviewer,
    image_paths=[],
    video_paths=[],
    evidence_id=evidence_id,
    qa_id=qa["qa_id"],
    attempt=candidate_index + 1,
)
```

随后严格校验 `checks.qa_formality.status`、`choice_logit_signal.available` 和两个 logprob。record 只写入 `{FORMALITY_COMPONENT: reward}`；禁止调用 `run_parallel_review_judges`、`build_review_from_gates` 或 `compute_judge_reward`。

- [ ] **Step 6：运行 reward 测试**

Run:

```powershell
python -m unittest tests.training.test_grpo_v3_formality_reward -v
```

Expected: 所有 formality reward 测试 `OK`。

- [ ] **Step 7：提交 Task 1**

```powershell
git add training/grpo_v3_formality_reward.py tests/training/test_grpo_v3_formality_reward.py
git commit -m "feat: add qa formality confidence reward"
```

---

### Task 2：接入 ms-swift ORM plugin，并保持旧 ORM 不变

**Files:**

- Modify: `training/grpo_v3_reward_plugin.py`
- Create: `tests/training/test_grpo_v3_formality_plugin.py`

- [ ] **Step 1：写 plugin 注册和 trace 的失败测试**

测试必须断言：

```python
self.assertIn("egoqa_gate1_controlled", plugin.orms)
self.assertIn("egoqa_repo_native_judge", plugin.orms)
self.assertIn("egoqa_qa_formality_confidence", plugin.orms)
```

使用注入的 `score_fn` 对四个 completion 返回 `[-0.5, 0.0, 0.5, 1.0]`，调用新 ORM 后断言：

```python
self.assertEqual(rewards, [-0.5, 0.0, 0.5, 1.0])
self.assertEqual(len(trace_rows), 4)
self.assertTrue(all(row["reward_kind"] == "qa_formality_confidence" for row in trace_rows))
self.assertEqual({row["reward_call_index"] for row in trace_rows}, {0})
```

同时测试任一 score 调用抛异常时：trace 记录 `infrastructure_error`、ORM 重新抛出原异常、没有返回部分 reward。

- [ ] **Step 2：运行 plugin 测试并确认缺少注册项**

```powershell
python -m unittest tests.training.test_grpo_v3_formality_plugin -v
```

Expected: 因 `egoqa_qa_formality_confidence` 未注册而失败。

- [ ] **Step 3：实现独立 ORM**

在旧 `RepoNativeJudgeReward` 之后增加：

```python
class FormalityConfidenceReward(ORM):
    def __init__(
        self,
        args: Any = None,
        *,
        trace_path: str | Path | None = None,
        score_fn: Callable[..., dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(args, **kwargs)
        self.trace_path = _trace_path(trace_path)
        self._lock = threading.Lock()
        self._reward_call_index = 0
        self.score_fn = score_fn or self._build_score_fn()
```

新类复用 `_expand`、`_write_rows` 和 `_json_safe`，但 `reward_kind` 固定为 `qa_formality_confidence`。`_build_score_fn()` 调用 `make_formality_score_fn()`；所有异常必须先写基础设施 trace 再重新抛出。

在文件末尾只追加：

```python
orms["egoqa_qa_formality_confidence"] = FormalityConfidenceReward
```

旧两个注册语句和旧类不得重写。

- [ ] **Step 4：运行 plugin 和旧 plugin 回归测试**

```powershell
python -m unittest tests.training.test_grpo_v3_formality_plugin tests.training.test_grpo_v3_reward_plugin -v
```

Expected: 全部 `OK`。

- [ ] **Step 5：提交 Task 2**

```powershell
git add training/grpo_v3_reward_plugin.py tests/training/test_grpo_v3_formality_plugin.py
git commit -m "feat: register formality confidence ORM"
```

---

### Task 3：实现旧 trace 离线回放 Gate A

**Files:**

- Create: `training/grpo_v3_formality_replay.py`
- Create: `tests/training/test_grpo_v3_formality_replay.py`

- [ ] **Step 1：写回放失败测试**

构造两个四候选 group，其中每条 record 包含：

```python
"judge_trace": {
    "qa_formality": {
        "parsed": {
            "checks": {"qa_formality": {"status": "PASS"}},
            "choice_logit_signal": {
                "available": True,
                "choice_logprobs": {"PASS": -2.0, "FAIL": -10.0},
            },
        }
    }
}
```

断言报告包含：

```python
self.assertEqual(report["schema_version"], "grpo_v3_formality_replay_v1")
self.assertEqual(report["complete_group_count"], 2)
self.assertEqual(report["finite_reward_count"], 8)
self.assertEqual(report["reward_components"], ["qa_formality_confidence"])
self.assertEqual(report["input_sha256"], expected_sha256)
```

另写一组所有 margin 相同的 fixture，断言正标准差比例低于 `0.8` 时报告 `status=failed`。

- [ ] **Step 2：运行测试并确认模块不存在**

```powershell
python -m unittest tests.training.test_grpo_v3_formality_replay -v
```

- [ ] **Step 3：实现回放器和 CLI**

公开接口固定为 `replay_trace(rows: list[dict[str, Any]], *, input_sha256: str) -> dict[str, Any]` 和 `replay_file(trace_path: Path, output_path: Path) -> dict[str, Any]`。前者执行下述分组与验收算法；后者计算文件 SHA-256、解析非空 JSONL 行、调用前者，并以 `allow_nan=False` 写入输出 JSON。

CLI：

```text
python -m training.grpo_v3_formality_replay --trace INPUT --output OUTPUT
```

回放只读取 `judge_trace.qa_formality.parsed.choice_logit_signal.choice_logprobs`，调用 Task 1 的 `confidence_reward()`，按 `reward_call_index` 和 `candidate_index` 重建组。缺失 logprob 的候选计入 `missing_logprob_count`；只有四条均有限的组才计入完整组。

通过条件固定为：至少一个完整组、所有已计算 reward 有限、reward 位于 `[-1,1]`、完整组正标准差比例至少 `0.8`。

- [ ] **Step 4：运行单元测试和真实旧 trace 回放**

```powershell
python -m unittest tests.training.test_grpo_v3_formality_replay -v
python -m training.grpo_v3_formality_replay --trace outputs/grpo_v3/gate3_14194844/gate3_14194844/reward_trace.jsonl --output outputs/grpo_v3/formality_replay_14194844.json
python -m json.tool outputs/grpo_v3/formality_replay_14194844.json
```

Expected: 单元测试 `OK`；真实报告 `status=passed`，完整组 `18`，正标准差组 `18`。

- [ ] **Step 5：提交 Task 3**

只提交源码和测试；`outputs/` 回放产物保留为本地验证证据，不纳入提交：

```powershell
git add training/grpo_v3_formality_replay.py tests/training/test_grpo_v3_formality_replay.py
git commit -m "feat: add formality reward trace replay"
```

---

### Task 4：实现 Smoke/Probe 收敛分析和实验产物契约

**Files:**

- Create: `training/grpo_v3_formality_convergence.py`
- Create: `training/grpo_v3_formality_artifacts.py`
- Create: `tests/training/test_grpo_v3_formality_convergence.py`
- Create: `tests/training/test_grpo_v3_formality_artifacts.py`

- [ ] **Step 1：写 40 组收敛通过的失败测试**

fixture 生成 40 个组、每组 4 个候选，前 10 组均值低于后 10 组，且每组标准差为正。断言：

```python
result = analyze_formality_convergence(rows, expected_steps=40)
self.assertEqual(result["status"], "passed")
self.assertEqual(result["group_count"], 40)
self.assertEqual(result["finite_reward_count"], 160)
self.assertGreater(result["reward_delta"], 0.0)
self.assertGreater(result["reward_slope"], 0.0)
self.assertGreaterEqual(result["positive_std_ratio"], 0.8)
```

分别构造以下失败 fixture：后 10 组不提高、slope 非正、正方差比例低于 0.8、component 污染、出现 groundedness judge trace、不可恢复率上升、非有限 reward、组不是四候选。每个 fixture 必须断言精确 `failed_checks` 名称。

- [ ] **Step 2：实现最小收敛分析器**

公开接口固定为 `analyze_formality_convergence(rows: list[dict[str, Any]], *, expected_steps: int) -> dict[str, Any]`。实现按 `reward_call_index` 分组、按 `candidate_index` 排序，严格检查四候选 cardinality、唯一 component、唯一 judge trace、有限 reward、窗口不可恢复率，然后计算下述 slope 和硬检查集合。

线性 slope 用普通最小二乘闭式计算，不引入 NumPy：

```python
def linear_slope(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    xs = list(range(1, len(values) + 1))
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(values)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values)) / denominator
```

CLI 固定为：

```text
python -m training.grpo_v3_formality_convergence --trace TRACE --expected-steps 40 --output OUTPUT
```

- [ ] **Step 3：写 smoke/probe artifact 验证失败测试**

构造临时 output tree，包含 trace、`trainer_state.json`、adapter、processor 和 `adapter_reload.json`。Smoke 断言恰好 4 个有限 reward、一个正方差组、global step 至少 1；Probe 断言恰好 160 个有限 reward、global step 等于 40、`convergence_metrics.json` passed。

Manifest 必须断言：

```python
self.assertEqual(manifest["reward_revision"], "qa_formality_confidence_v1")
self.assertEqual(manifest["reward_components"], ["qa_formality_confidence"])
self.assertEqual(manifest["margin_clip"], 32.0)
self.assertEqual(manifest["parent_run"], str(parent.resolve()))
self.assertEqual(manifest["dataset_sha256"], expected_sha256)
self.assertFalse(manifest["calls_video_reviewer"])
```

- [ ] **Step 4：实现 formality 专用 artifact 模块**

CLI 子命令：

```text
python -m training.grpo_v3_formality_artifacts validate --mode smoke --output-dir DIR
python -m training.grpo_v3_formality_artifacts validate --mode probe --output-dir DIR
python -m training.grpo_v3_formality_artifacts summarize --mode smoke --output-dir DIR --dataset DATASET --parent-run PARENT --policy-model POLICY --reviewer-model REVIEWER --job-id JOB
```

输出文件固定为：

```text
formality_smoke_result.json
formality_probe_result.json
resolved_config.json
run_manifest.json
```

模块只读取 formality trace，不修改现有 `grpo_v3_gate_validate.py` 和 `grpo_v3_summary.py`，避免更改旧 Gate 语义。

- [ ] **Step 5：运行 Task 4 测试**

```powershell
python -m unittest tests.training.test_grpo_v3_formality_convergence tests.training.test_grpo_v3_formality_artifacts -v
```

Expected: 全部 `OK`。

- [ ] **Step 6：提交 Task 4**

```powershell
git add training/grpo_v3_formality_convergence.py training/grpo_v3_formality_artifacts.py tests/training/test_grpo_v3_formality_convergence.py tests/training/test_grpo_v3_formality_artifacts.py
git commit -m "feat: validate formality convergence artifacts"
```

---

### Task 5：编写 Smoke/Probe Slurm 脚本和静态契约测试

**Files:**

- Create: `hpc/grpo_v3_formality_smoke.sbatch`
- Create: `hpc/grpo_v3_formality_probe.sbatch`
- Create: `tests/training/test_grpo_v3_formality_slurm.py`

- [ ] **Step 1：先写 Slurm 文本契约测试**

测试读取两个脚本文本，逐项断言：

```python
self.assertIn("--reward_funcs egoqa_qa_formality_confidence", text)
self.assertIn("--num_generations 4", text)
self.assertIn("--temperature 0.7", text)
self.assertIn("--learning_rate 1e-5", text)
self.assertIn("--lr_scheduler_type constant", text)
self.assertIn("--beta 0.0", text)
self.assertIn("--freeze_vit true", text)
self.assertIn("--freeze_aligner true", text)
self.assertIn("--use_vllm false", text)
self.assertNotIn("ground_answer_gap_v1", text)
self.assertNotIn("egoqa_repo_native_judge", text)
self.assertNotIn("EGOQA_GROUNDEDNESS_AUDIT_SUMMARY", text)
```

Smoke 必须包含 `--max_steps 1`；probe 必须包含 `--max_steps 40`。两个脚本都必须从 `gate2_result.json` 和 `run_manifest.json` 验证 parent，并且 probe 不能读取 smoke adapter。

- [ ] **Step 2：运行测试并确认脚本不存在**

```powershell
python -m unittest tests.training.test_grpo_v3_formality_slurm -v
```

- [ ] **Step 3：以现有 Gate 3 脚本为资源模板编写 smoke**

保留：2×L40S、8 CPU、64GB、同一训练/推理环境、policy GPU 0、reviewer GPU 1、GPU monitor、adapter reload 和清理 trap。

关键差异固定为：

```bash
EGOQA_FORMALITY_REWARD_REVISION="qa_formality_confidence_v1"
export EGOQA_FORMALITY_REWARD_REVISION
export EGOQA_GRPO_V3_REWARD_TRACE="${OUTPUT_DIR}/reward_trace.jsonl"
```

reviewer `vllm serve` 不需要 `--allowed-local-media-path`，因为 formality judge 只接收文本。训练命令使用 `--reward_funcs egoqa_qa_formality_confidence` 和 `--max_steps 1`。训练后依次执行 adapter reload、formality artifact validate、formality artifact summarize；失败仍保留 manifest 和日志。

- [ ] **Step 4：编写独立 40-step probe**

Probe 重新从 `${GATE2_DIR}` 解析 adapter，不能引用 `latest_formality_smoke_output.txt`。训练配置除 `--max_steps 40` 外与 smoke 保持一致。训练结束先运行：

```bash
"${PYTHON}" -m training.grpo_v3_formality_convergence \
  --trace "${EGOQA_GRPO_V3_REWARD_TRACE}" \
  --expected-steps 40 \
  --output "${OUTPUT_DIR}/convergence_metrics.json"
```

再运行 artifact validate 和 summarize。只有 probe 验收通过才写 `latest_formality_probe_output.txt`。

- [ ] **Step 5：运行 Slurm 契约和 shell 静态检查**

```powershell
python -m unittest tests.training.test_grpo_v3_formality_slurm -v
bash -n hpc/grpo_v3_formality_smoke.sbatch
bash -n hpc/grpo_v3_formality_probe.sbatch
```

Expected: unittest `OK`；两个 `bash -n` 退出码均为 0。若本机没有 Bash，只报告该项未运行，并在 Torch 登录节点预检中补做，不能伪报通过。

- [ ] **Step 6：提交 Task 5**

```powershell
git add hpc/grpo_v3_formality_smoke.sbatch hpc/grpo_v3_formality_probe.sbatch tests/training/test_grpo_v3_formality_slurm.py
git commit -m "feat: add formality convergence Slurm probes"
```

---

### Task 6：补齐实验目录和可复制 Torch Runbook

**Files:**

- Create: `docs/GRPO/v3/experiments/qa_formality_only_convergence_v1/README_CN.md`
- Create: `docs/GRPO/v3/experiments/qa_formality_only_convergence_v1/TORCH_RUNBOOK_CN.md`
- Create: `docs/GRPO/v3/experiments/qa_formality_only_convergence_v1/RESULT_INTERPRETATION_CN.md`

- [ ] **Step 1：编写 README**

README 顶部必须标记：

```text
本地状态：代码/单元测试/历史 trace 回放准备中或已完成
远程状态：尚未运行，不能声称 reward 已提升
正式顺序：Gate A 回放 → 1-step smoke → 40-step probe
```

列出四份文档的职责，并链接锁定规格 `EXPERIMENT_DESIGN_CN.md`。

- [ ] **Step 2：编写 Torch runbook**

Runbook 必须包含可逐块复制的命令，且人工只需要替换真实作业 ID，不需要现场写 Python 文件。结构固定为：

1. 本地 PowerShell `sftp` 上传文件清单；
2. Torch 登录节点设置 `PROJECT_ROOT`、`TRAIN_ENV`、`INFERENCE_ENV`、模型路径；
3. `git status` 和上传文件存在性检查；
4. Python import/compileall/unittest；
5. 旧作业 `14194844` trace 路径定位与 Gate A 回放；
6. Gate A JSON 硬断言；
7. `sbatch hpc/grpo_v3_formality_smoke.sbatch`；
8. `squeue`、`scontrol`、stdout/stderr/reviewer log 定位；
9. smoke result、trace、adapter reload 和 manifest 验收；
10. 只有 smoke passed 后提交 probe；
11. probe 监控和 `convergence_metrics.json` 验收；
12. 下载产物的交互式 SFTP 命令；
13. 失败时收集的最小证据清单。

所有 shell 变量必须在同一代码块中显式定义，不能依赖 PowerShell 变量。不得使用 `sftp -b`、临时 batch 文件或要求用户在 Torch 上编辑源码。

- [ ] **Step 3：编写结果解释文档**

明确列出：

- 通过时唯一允许的结论；
- reward 未提升但 adapter 更新时的结论；
- 零方差时不能否定 GRPO；
- 不可恢复率上升时必须否决；
- reviewer/logprob 故障属于基础设施失败；
- 本实验不解锁完整 Gate 4。

附上“汇报模板”，字段固定为本地验证、Torch 作业 ID、Gate A、smoke、probe、reward delta、slope、正方差比例、不可恢复率、adapter reload 和下一步。

- [ ] **Step 4：扫描 runbook 的不可执行占位内容**

Run:

```powershell
rg -n "TB[D]|TO[D]O|待[补]|稍后填写|your_path|path/to|JOB_ID_VALUE" docs/GRPO/v3/experiments/qa_formality_only_convergence_v1
```

Expected: 无匹配。作业 ID 通过 shell 命令返回值或明确的 `FORMALITY_SMOKE_JOB_ID=实际数字` 说明处理，不能在可复制命令中保留占位内容。

- [ ] **Step 5：提交 Task 6**

```powershell
git add docs/GRPO/v3/experiments/qa_formality_only_convergence_v1
git commit -m "docs: add formality convergence Torch runbook"
```

---

### Task 7：全量本地验证与最终审计

**Files:**

- Verify: `training/grpo_v3_formality_*.py`
- Verify: `training/grpo_v3_reward_plugin.py`
- Verify: `tests/training/test_grpo_v3_*.py`
- Verify: `hpc/grpo_v3_formality_*.sbatch`
- Verify: `docs/GRPO/v3/experiments/qa_formality_only_convergence_v1/*.md`

- [ ] **Step 1：运行 formality 专项测试**

```powershell
python -m unittest discover -s tests/training -p "test_grpo_v3_formality_*.py" -v
```

Expected: 全部 `OK`。

- [ ] **Step 2：运行全部 GRPO v3 回归测试**

```powershell
python -m unittest discover -s tests/training -p "test_grpo_v3_*.py" -v
```

Expected: `OK`，并记录实际测试数；不能沿用历史 `Ran 70 tests` 作为本轮证据。

- [ ] **Step 3：运行编译和源码预检**

```powershell
python -m compileall training tests/training
python -m training.grpo_v3_preflight --source-only --output tmp/grpo_v3_formality_source_preflight.json
```

Expected: compileall 退出码 0；preflight `status=passed`。

- [ ] **Step 4：重新运行真实旧 trace 回放并硬断言**

```powershell
python -m training.grpo_v3_formality_replay --trace outputs/grpo_v3/gate3_14194844/gate3_14194844/reward_trace.jsonl --output outputs/grpo_v3/formality_replay_14194844.json
python -c "import json; d=json.load(open(r'outputs/grpo_v3/formality_replay_14194844.json',encoding='utf-8')); assert d['status']=='passed'; assert d['complete_group_count']==18; assert d['positive_std_group_count']==18; print(d)"
```

Expected: 断言通过并打印报告。

- [ ] **Step 5：运行契约扫描和 diff 检查**

```powershell
rg -n "egoqa_repo_native_judge|ground_answer_gap_v1|EGOQA_GROUNDEDNESS_AUDIT_SUMMARY" hpc/grpo_v3_formality_smoke.sbatch hpc/grpo_v3_formality_probe.sbatch
rg -n "TB[D]|TO[D]O|待[补]|your_path|path/to|JOB_ID_VALUE" docs/GRPO/v3/experiments/qa_formality_only_convergence_v1
git diff --check
```

Expected: 前两条无匹配；`git diff --check` 无输出。

- [ ] **Step 6：检查改动范围**

```powershell
git status --short
git diff --stat HEAD~6..HEAD
git log -7 --oneline
```

确认未覆盖旧 Gate 3/Gate 3 v2 文件，未提交 `outputs/`、模型产物、缓存或用户已有无关修改。

- [ ] **Step 7：提交仅由验证发现的必要修正**

如果前述验证暴露实现缺陷，先增加能复现缺陷的测试，再修复并提交精确文件：

```powershell
git add training/grpo_v3_formality_reward.py training/grpo_v3_formality_replay.py training/grpo_v3_formality_convergence.py training/grpo_v3_formality_artifacts.py training/grpo_v3_reward_plugin.py tests/training/test_grpo_v3_formality_reward.py tests/training/test_grpo_v3_formality_plugin.py tests/training/test_grpo_v3_formality_replay.py tests/training/test_grpo_v3_formality_convergence.py tests/training/test_grpo_v3_formality_artifacts.py tests/training/test_grpo_v3_formality_slurm.py hpc/grpo_v3_formality_smoke.sbatch hpc/grpo_v3_formality_probe.sbatch docs/GRPO/v3/experiments/qa_formality_only_convergence_v1
git commit -m "fix: harden formality convergence validation"
```

如果没有修正，不创建空提交。

- [ ] **Step 8：最终报告边界**

最终汇报必须分开写：

```text
已完成：本地代码、单元测试、compileall、source preflight、真实历史 trace 回放、runbook 静态检查
未完成：Torch smoke、Torch 40-step probe、远程 reward 提升结论
人工下一步：按 TORCH_RUNBOOK_CN.md 从 Gate A 开始执行
```

不得把本地历史 trace 回放写成新训练成功。
