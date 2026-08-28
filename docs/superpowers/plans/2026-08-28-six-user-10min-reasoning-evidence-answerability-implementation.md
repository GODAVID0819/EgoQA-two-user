# 六用户十分钟 Reasoning、Evidence 与 Answerability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. 当前执行环境不启用子代理分派，使用同一会话内的批次执行和人工检查点。

**Goal:** 在不改变三分钟默认行为的前提下，实现六用户十分钟 `600s/K=240/G=30s` 路径、高置信度 evidence 多数投票、answerability 事实置信度，以及 generator/evidence/answerability reasoning 8192 与 formality 非 reasoning 2048 的单模型 stage-specific 调用。

**Architecture:** 继续复用现有六用户 candidate、QA loop 和一个驻留 Qwen 模型。媒体层把每用户 segment 数从固定六段推广为由窗口长度决定；evidence 视觉调用输出 segment observations 与单一用户级 vote，程序做确定性多数聚合；answerability 仍只运行 speaker-only 和 combined-all-six。Reasoning 与输出上限改为每次调用参数，最终 JSON 从 reasoning 后最后一个完整对象提取。

**Tech Stack:** Python 3.11+/3.14、pytest、Hugging Face Transformers、Qwen3.6-27B、decord、FFmpeg、Slurm、PowerShell、Bash。

---

## 文件结构与修改责任

- `schema.py`：从 reasoning 输出中提取最后一个完整 JSON object。
- `qwen3vl_runner.py`：定义 `GenerationCallProfile`，让所有 runner 支持每次调用的输出上限与 thinking 配置；内存估算使用实际调用上限。
- `video_qa_loop.py`：构造 stage profiles，向 generator、formality、evidence、answerability 和 JSON repair 传播配置；合并 evidence 投票结果和 answerability gate。
- `prompts.py`：扩展 evidence observation、aggregation 和 answerability schema/prompt。
- `evidence_chunk_review.py`：支持六段与二十段输入，校验用户级 vote，并执行确定性多数聚合。
- `group_relative_clip_sampling.py`：保持三分钟默认 `G=10s`，允许十分钟 wrapper 显式传入 `G=30s` 并记录 `K=240`。
- `hpc/qa/smoke/run_six_user_qa_runtime_probe.sbatch`：接收 cross-gap 与 stage profile 环境变量，取消硬编码六个源 segment 的假设。
- `hpc/qa/experiments/run_six_user_qa_10min_reasoning.sbatch`：独立十分钟入口，一个 group、一个 slot。
- `tests/test_schema_json_extraction.py`：reasoning 后 JSON 解析。
- `tests/test_qwen_runner_compat.py`：per-call profile 和 template thinking 路由。
- `tests/test_evidence_chunk_review.py`：segment 数、vote 校验和多数聚合。
- `tests/test_six_user_prompts.py`：新 schema 与 prompt 语义。
- `tests/test_six_user_video_qa_loop.py`：stage profile、answerability confidence 和 gate。
- `tests/test_zip_temporal_pruning.py`、`tests/test_three_minute_blockwise_pruning.py`：`G=30s` 边界与三分钟 `G=10s` 回归。
- `tests/test_ten_minute_reasoning_job_contract.py`：十分钟 wrapper 和 runtime 参数合同。

## Task 1：Reasoning 后最终 JSON 提取

**Files:**
- Create: `tests/test_schema_json_extraction.py`
- Modify: `schema.py`

- [ ] **Step 1: 写失败测试，覆盖 reasoning 中的大括号、示例 JSON、最终 JSON 和截断输出**

```python
import pytest

from egolife_two_user_qa.schema import extract_json_object


def test_extract_json_object_uses_last_complete_object_after_reasoning() -> None:
    raw = (
        "<think>Compare {A, B}; an example is {\"draft\": true}.</think>\n"
        '{"reason": "supported", "needed_facts": []}'
    )
    assert extract_json_object(raw) == {
        "reason": "supported",
        "needed_facts": [],
    }


def test_extract_json_object_uses_final_object_after_example_object() -> None:
    raw = (
        'Example: {"status": "FAIL"}\n'
        'Final: {"status": "PASS", "reason": "visible"}'
    )
    assert extract_json_object(raw)["status"] == "PASS"


def test_extract_json_object_rejects_truncated_final_object() -> None:
    with pytest.raises(ValueError, match="No complete JSON object"):
        extract_json_object('<think>done</think>\n{"status": "PASS"')
```

- [ ] **Step 2: 运行 RED 测试并确认旧 parser 失败**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_schema_json_extraction.py
```

Expected: 至少前两个测试失败，因为旧实现从第一个 `{` 截取到最后一个 `}`；截断错误消息也不匹配新合同。

- [ ] **Step 3: 实现最后一个完整 JSON object 解析**

在 `schema.py` 中使用以下逻辑替换首尾大括号截取：

```python
def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    decoder = json.JSONDecoder()
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    for match in re.finditer(r"\{", cleaned):
        try:
            value, consumed = decoder.raw_decode(cleaned[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append((match.start() + consumed, match.start(), value))
    if not candidates:
        raise ValueError("No complete JSON object found in model output")
    _, _, selected = max(candidates, key=lambda row: (row[0], -row[1]))
    return selected
```

- [ ] **Step 4: 运行 GREEN 测试和相关 schema 回归**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_schema_json_extraction.py tests/test_six_user_prompts.py
```

Expected: 全部通过。

- [ ] **Step 5: 提交 Task 1**

```powershell
git add -- schema.py tests/test_schema_json_extraction.py
git commit -m "fix: 解析 reasoning 后最终 JSON"
```

## Task 2：单模型 per-call stage profile

**Files:**
- Modify: `qwen3vl_runner.py`
- Modify: `video_qa_loop.py`
- Modify: `tests/test_qwen_runner_compat.py`
- Modify: `tests/test_six_user_video_qa_loop.py`

- [ ] **Step 1: 写失败测试，定义 profile API 和一个 runner identity**

在 `tests/test_qwen_runner_compat.py` 增加：

```python
from egolife_two_user_qa.qwen3vl_runner import GenerationCallProfile


def test_generation_call_profile_validates_output_budget() -> None:
    profile = GenerationCallProfile(max_new_tokens=8192, disable_thinking=False)
    assert profile.max_new_tokens == 8192
    assert profile.disable_thinking is False


def test_generation_call_profile_rejects_non_positive_budget() -> None:
    with pytest.raises(ValueError, match="max_new_tokens must be positive"):
        GenerationCallProfile(max_new_tokens=0, disable_thinking=False)
```

在 `tests/test_six_user_video_qa_loop.py` 增加 recording runner，并断言：

```python
assert profiles["generator"] == (8192, False)
assert profiles["evidence_segment_observation"] == (8192, False)
assert profiles["evidence_groundedness_aggregation"] == (8192, False)
assert profiles["answerability"] == (8192, False)
assert profiles["qa_formality"] == (2048, True)
assert profiles["json_repair"] == (2048, True)
assert len({call["runner_id"] for call in calls}) == 1
```

- [ ] **Step 2: 运行 RED 测试**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_qwen_runner_compat.py tests/test_six_user_video_qa_loop.py
```

Expected: `GenerationCallProfile` 不存在，且当前 judge 调用没有 per-call profile。

- [ ] **Step 3: 在 runner 层实现不可变 profile**

在 `qwen3vl_runner.py` 增加：

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationCallProfile:
    max_new_tokens: int
    disable_thinking: bool

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
```

所有 backend 的 `generate()` 与本地 `_generate()` 增加：

```python
call_profile: GenerationCallProfile | None = None
```

本地 runner 在每次调用中计算：

```python
effective_max_new_tokens = (
    call_profile.max_new_tokens if call_profile is not None else self.max_new_tokens
)
effective_disable_thinking = (
    call_profile.disable_thinking if call_profile is not None else self.disable_thinking
)
```

并将它们分别传给 `apply_chat_template_compat()`、`generation_kwargs()` 和 KV 预算函数。禁止临时修改 `self.max_new_tokens` 或 `self.disable_thinking`。

修改内存估算签名：

```python
def _estimated_kv_gib(self, *, input_tokens: int, max_new_tokens: int) -> float:
    return (input_tokens + max_new_tokens) * self.kv_bytes_per_token / 1024**3
```

- [ ] **Step 4: 在 QA loop 构造固定 stage profiles**

在 `video_qa_loop.py` 增加：

```python
def six_user_reasoning_profiles() -> dict[str, GenerationCallProfile]:
    reasoning = GenerationCallProfile(max_new_tokens=8192, disable_thinking=False)
    formality = GenerationCallProfile(max_new_tokens=2048, disable_thinking=True)
    return {
        "generator": reasoning,
        "evidence_segment_observation": reasoning,
        "evidence_groundedness_aggregation": reasoning,
        "answerability": reasoning,
        "qa_formality": formality,
        "json_repair": formality,
    }
```

该函数只在新十分钟 CLI 开关启用时使用；其他路径继续从现有 `max_new_tokens/disable_thinking` 构造统一 profile。

CLI 增加：

```python
parser.add_argument("--six-user-ten-minute-reasoning-profile", action="store_true")
parser.add_argument("--formality-max-new-tokens", type=int, default=2048)
```

- [ ] **Step 5: 传播 profile 到所有模型调用和 schema-only repair**

更新 generator、`run_model_judge_branch()`、`run_chunked_evidence_groundedness_eval()`、`run_answerability_eval()` 的调用，使每次 `runner.generate()` 都显式接收对应 profile。Prompt rows 与 attempt trace 写入：

```python
"reasoning_enabled": not profile.disable_thinking,
"max_new_tokens": profile.max_new_tokens,
```

- [ ] **Step 6: 运行 GREEN 测试**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_qwen_runner_compat.py tests/test_six_user_video_qa_loop.py
```

Expected: 全部通过，fake runner 和真实 runner 的旧调用继续兼容。

- [ ] **Step 7: 提交 Task 2**

```powershell
git add -- qwen3vl_runner.py video_qa_loop.py tests/test_qwen_runner_compat.py tests/test_six_user_video_qa_loop.py
git commit -m "feat: 按评审阶段配置 reasoning 输出"
```

## Task 3：可变 segment 数与用户级 evidence vote

**Files:**
- Create: `tests/test_evidence_chunk_review.py`
- Modify: `prompts.py`
- Modify: `evidence_chunk_review.py`
- Modify: `video_qa_loop.py`
- Modify: `tests/test_six_user_prompts.py`

- [ ] **Step 1: 写 segment 数和连续性失败测试**

在新测试文件中构造每用户六段、二十段和缺一段的 packet：

```python
def test_evidence_segment_specs_accepts_twenty_segments_for_ten_minutes() -> None:
    specs = evidence_segment_specs(packet_with_segments(20), six_full_video_paths())
    assert all(len(rows) == 20 for rows in specs.values())
    assert specs["Jake"][0]["start_seconds"] == 0.0
    assert specs["Jake"][-1]["start_seconds"] == 570.0


def test_evidence_segment_specs_preserves_six_segment_three_minute_contract() -> None:
    specs = evidence_segment_specs(packet_with_segments(6), six_full_video_paths())
    assert all(len(rows) == 6 for rows in specs.values())


def test_evidence_segment_specs_rejects_non_uniform_segment_counts() -> None:
    with pytest.raises(ValueError, match="same complete segment count"):
        evidence_segment_specs(packet_with_one_user_missing_segment(), six_full_video_paths())
```

- [ ] **Step 2: 写 user_vote 校验失败测试**

```python
def test_validate_segment_observation_requires_high_confidence_vote() -> None:
    observation = valid_observation(segment_count=20)
    observation["user_vote"] = {
        "visible": True,
        "confidence": "LOW",
        "supported_option": "B",
        "supporting_segment_indices": [3],
        "reason": "blurry",
    }
    errors = validate_segment_observation(
        observation,
        expected_user="Jake",
        expected_time_tokens=twenty_tokens(),
    )
    assert "non-HIGH vote must not select an option" in errors
```

- [ ] **Step 3: 写确定性多数聚合 RED 测试**

```python
@pytest.mark.parametrize(
    ("votes", "passed"),
    [
        (["B", "B", "B", None, None, None], True),
        (["B", "B", "A", None, None, None], True),
        (["B", "B", "A", "A", None, None], False),
        (["B", "B", "B", "A", "A", "A"], False),
        ([None, None, None, None, None, None], False),
    ],
)
def test_aggregate_evidence_user_votes(votes, passed) -> None:
    summary = aggregate_evidence_user_votes("B", make_user_votes(votes))
    assert summary["passed"] is passed
```

再增加 `MEDIUM/LOW` 不计票、同一用户多个 segment 只计一票、competitor 达阈值拒绝的独立测试。

- [ ] **Step 4: 运行 RED 测试**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_evidence_chunk_review.py tests/test_six_user_prompts.py
```

Expected: 固定六段错误、`user_vote` schema 缺失、聚合函数不存在。

- [ ] **Step 5: 扩展 evidence prompt schema**

在 `prompts.py` 中把 claim 扩展为：

```python
{
    "claim": "material claim",
    "status": "SUPPORTED, CONTRADICTED, NOT_VISIBLE, or AMBIGUOUS",
    "confidence": "HIGH, MEDIUM, or LOW",
    "visual_description": "direct visual evidence or why it is not visible",
    "original_time_range": "original interval",
}
```

并增加 `user_vote`。Prompt 明确规定：低置信度、遮挡、太远、太暗和无法区分时使用 NOT_VISIBLE；只有 HIGH 才能选择 A–E。

同时增加专用于 text-only premise audit 的 schema；它不再决定答案票数：

```python
EVIDENCE_AGGREGATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "premises_supported",
        "high_confidence_material_conflict",
        "reason",
    ],
    "properties": {
        "premises_supported": {"type": "boolean"},
        "high_confidence_material_conflict": {"type": "boolean"},
        "reason": {"type": "string", "minLength": 1},
    },
}
```

- [ ] **Step 6: 实现 segment 泛化和 vote 校验**

`evidence_segment_specs()` 从六个 clips 的真实 `segments` 读取统一 count，并检查每段 `start_seconds=index*30`、time token 顺序和用户间 count 一致。

实现：

```python
def aggregate_evidence_user_votes(
    correct: str,
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    counts = {letter: 0 for letter in OPTION_LETTERS}
    visible_users = []
    for observation in observations:
        vote = observation["user_vote"]
        if vote["visible"] is not True or vote["confidence"] != "HIGH":
            continue
        option = vote["supported_option"]
        if option in counts:
            counts[option] += 1
            visible_users.append(observation["user"])
    visible_count = len(visible_users)
    threshold_options = [
        option
        for option, count in counts.items()
        if count >= 3 or count > visible_count / 2
    ]
    passed = correct in threshold_options and threshold_options == [correct]
    return {
        "passed": passed,
        "visible_user_count": visible_count,
        "option_support_counts": counts,
        "threshold_options": threshold_options,
        "not_visible_user_count": len(observations) - visible_count,
    }
```

- [ ] **Step 7: 将 vote summary 设为 aggregation 的权威答案支持输入**

`run_chunked_evidence_groundedness_eval()` 在六个 observation 后计算 vote summary，并把它放入 text-only aggregation prompt。Aggregator 只判断题干 premise、身份/连续性/时间冲突；程序最终要求：

```python
final_pass = (
    vote_summary["passed"]
    and aggregation["premises_supported"] is True
    and aggregation["high_confidence_material_conflict"] is False
)
```

最终 check 保存 `vote_summary`，不允许 text aggregator 重新计算人数。

- [ ] **Step 8: 运行 GREEN 测试**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_evidence_chunk_review.py tests/test_six_user_prompts.py tests/test_six_user_video_qa_loop.py
```

Expected: 全部通过。

- [ ] **Step 9: 提交 Task 3**

```powershell
git add -- prompts.py evidence_chunk_review.py video_qa_loop.py tests/test_evidence_chunk_review.py tests/test_six_user_prompts.py tests/test_six_user_video_qa_loop.py
git commit -m "feat: 按高置信度可见用户聚合 evidence"
```

## Task 4：Answerability needed-fact 置信度

**Files:**
- Modify: `prompts.py`
- Modify: `video_qa_loop.py`
- Modify: `tests/test_six_user_prompts.py`
- Modify: `tests/test_six_user_video_qa_loop.py`

- [ ] **Step 1: 写失败测试**

```python
def test_answerability_requires_high_confidence_visible_facts() -> None:
    evaluation = sufficiency_evaluation(
        visibility="VISIBLE",
        confidence="MEDIUM",
        source_user="speaker",
    )
    sufficient, error = parsed_answerability_sufficiency(evaluation)
    assert error is None
    assert sufficient is False


def test_answerability_passes_when_combined_users_fill_all_high_confidence_facts() -> None:
    gate = answerability_gate(
        qa_item=six_user_qa(),
        evaluations=[
            condition_eval("speaker_only", visibility="NOT_VISIBLE", confidence="LOW"),
            condition_eval("combined_all_six_users", visibility="VISIBLE", confidence="HIGH"),
        ],
    )
    assert gate["passed"] is True
```

增加 speaker VISIBLE/HIGH 时拒绝、source user 不属于 condition 时拒绝、非六用户旧路径不变的测试。

- [ ] **Step 2: 运行 RED 测试**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_six_user_prompts.py tests/test_six_user_video_qa_loop.py
```

Expected: schema 不要求 confidence，MEDIUM 当前会被错误视为 sufficient。

- [ ] **Step 3: 修改 schema 与 prompt**

`ANSWERABILITY_SUFFICIENCY_SCHEMA.needed_facts[]` 增加必填：

```python
"confidence": {
    "type": "string",
    "enum": ["HIGH", "MEDIUM", "LOW"],
}
```

Prompt 明确：VISIBLE 只表示直接可见；最终 sufficiency 还要求 HIGH。MEDIUM/LOW、NOT_VISIBLE 和 AMBIGUOUS 均不能补齐 needed fact。

- [ ] **Step 4: 修改确定性 parser**

在 `parsed_answerability_sufficiency()` 中：

```python
if visibility != "VISIBLE" or confidence != "HIGH":
    all_visible = False
```

保留 source user、时间范围、禁止 answer fields 和两条件 gate 的现有校验。

- [ ] **Step 5: 运行 GREEN 测试**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_six_user_prompts.py tests/test_six_user_video_qa_loop.py
```

Expected: 全部通过。

- [ ] **Step 6: 提交 Task 4**

```powershell
git add -- prompts.py video_qa_loop.py tests/test_six_user_prompts.py tests/test_six_user_video_qa_loop.py
git commit -m "feat: 为 answerability 增加高置信度事实门禁"
```

## Task 5：十分钟 ZIP `K=240/G=30s` 合同

**Files:**
- Modify: `group_relative_clip_sampling.py`
- Modify: `hpc/qa/smoke/run_six_user_qa_runtime_probe.sbatch`
- Modify: `tests/test_zip_temporal_pruning.py`
- Modify: `tests/test_three_minute_blockwise_pruning.py`

- [ ] **Step 1: 写 `G=30s` 边界 RED 测试**

```python
def test_ten_minute_zip_accepts_center_gap_at_thirty_seconds() -> None:
    result = prune_time_aware_cluster_pair(
        *pair_inputs(center_gap_seconds=30.0),
        cross_gap_mode="center",
        max_cross_gap_seconds=30.0,
    )
    assert result["high_similarity_representative_pair_count"] == 1


def test_ten_minute_zip_rejects_center_gap_above_thirty_seconds() -> None:
    result = prune_time_aware_cluster_pair(
        *pair_inputs(center_gap_seconds=30.001),
        cross_gap_mode="center",
        max_cross_gap_seconds=30.0,
    )
    assert result["high_similarity_representative_pair_count"] == 0
```

在三分钟测试继续断言默认 `max_cross_gap_seconds == 10.0`。

- [ ] **Step 2: 写十分钟 K 值和 CLI 传播 RED 测试**

```python
def test_ten_minute_analysis_uses_full_window_k240_and_g30() -> None:
    calls = run_six_user_analysis(
        duration_seconds=600.0,
        pruning_max_cross_gap_seconds=30.0,
    )
    assert all(call["cluster_count"] == 240 for call in calls)
    assert all(call["max_cross_gap_seconds"] == 30.0 for call in calls)
```

Runtime script 测试要求环境变量传入：

```python
assert 'PRUNING_MAX_CROSS_GAP_SECONDS="${PRUNING_MAX_CROSS_GAP_SECONDS:-10}"' in runtime
assert '--pruning-max-cross-gap-seconds "${PRUNING_MAX_CROSS_GAP_SECONDS}"' in runtime
```

- [ ] **Step 3: 运行 RED 测试**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_zip_temporal_pruning.py tests/test_three_minute_blockwise_pruning.py
```

Expected: runtime 尚未传播环境变量，十分钟 helper/fixture 断言失败。

- [ ] **Step 4: 实现参数传播，保持默认 10 秒**

在 runtime script 增加：

```bash
PRUNING_MAX_CROSS_GAP_SECONDS="${PRUNING_MAX_CROSS_GAP_SECONDS:-10}"
export PRUNING_MAX_CROSS_GAP_SECONDS
```

candidate mining 命令增加：

```bash
--pruning-cross-gap-mode center \
--pruning-max-cross-gap-seconds "${PRUNING_MAX_CROSS_GAP_SECONDS}" \
```

不修改 `group_relative_clip_sampling.py` 中三分钟通用默认 `10.0`。十分钟由 wrapper 显式传 `30`。

- [ ] **Step 5: 运行 GREEN 测试**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_zip_temporal_pruning.py tests/test_three_minute_blockwise_pruning.py tests/test_six_user_group_relative_sampling.py
```

Expected: 全部通过。

- [ ] **Step 6: 提交 Task 5**

```powershell
git add -- group_relative_clip_sampling.py hpc/qa/smoke/run_six_user_qa_runtime_probe.sbatch tests/test_zip_temporal_pruning.py tests/test_three_minute_blockwise_pruning.py
git commit -m "feat: 传播十分钟 ZIP cross-gap 参数"
```

## Task 6：独立十分钟 reasoning wrapper

**Files:**
- Create: `hpc/qa/experiments/run_six_user_qa_10min_reasoning.sbatch`
- Create: `tests/test_ten_minute_reasoning_job_contract.py`
- Modify: `hpc/qa/smoke/run_six_user_qa_runtime_probe.sbatch`

- [ ] **Step 1: 写 wrapper 合同 RED 测试**

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "hpc/qa/experiments/run_six_user_qa_10min_reasoning.sbatch"
RUNTIME = (ROOT / "hpc/qa/smoke/run_six_user_qa_runtime_probe.sbatch").read_text(
    encoding="utf-8"
)


def test_ten_minute_reasoning_wrapper_contract() -> None:
    job = JOB.read_text(encoding="utf-8")
    assert "#SBATCH --mem=96G" in job
    assert "#SBATCH --time=04:00:00" in job
    assert 'EVIDENCE_DURATION_SECONDS="600"' in job
    assert 'PRUNING_BLOCK_SECONDS="30"' in job
    assert 'PRUNING_MAX_CROSS_GAP_SECONDS="30"' in job
    assert 'TARGET_GENERATION_GROUPS="1"' in job
    assert 'MAX_GENERATION_SLOTS="1"' in job
    assert 'MAX_NEW_TOKENS="8192"' in job
    assert 'FORMALITY_MAX_NEW_TOKENS="2048"' in job
    assert 'SIX_USER_TEN_MINUTE_REASONING_PROFILE="1"' in job
    assert "--nodelist" not in job and "#SBATCH -w" not in job


def test_runtime_passes_ten_minute_reasoning_profile() -> None:
    assert '--max-new-tokens "${MAX_NEW_TOKENS}"' in RUNTIME
    assert '--formality-max-new-tokens "${FORMALITY_MAX_NEW_TOKENS}"' in RUNTIME
    assert "--six-user-ten-minute-reasoning-profile" in RUNTIME
```

- [ ] **Step 2: 运行 RED 测试**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_ten_minute_reasoning_job_contract.py
```

Expected: 新 wrapper 不存在，runtime 尚未传 reasoning profile。

- [ ] **Step 3: 创建十分钟 wrapper**

Wrapper 使用已确认的本地实验值：

```bash
RUN_MODE="six_user_qa_10min_reasoning"
ACCEPTED_TARGET="1"
EVIDENCE_TARGET="6"
MAX_ATTEMPTS="3"
ALLOW_PARTIAL="1"
EVIDENCE_DURATION_SECONDS="600"
PRUNING_BLOCK_SECONDS="30"
PRUNING_MAX_CROSS_GAP_SECONDS="30"
TARGET_GENERATION_GROUPS="1"
SINGLE_CANDIDATE_GROUP="0"
MAX_GENERATION_SLOTS="1"
QWEN_MEMORY_SAFE_VIDEO_FPS="0.25"
QWEN_MEMORY_SAFE_MAX_IMAGE_PIXELS="131072"
QWEN_MEMORY_SAFE_MAX_INPUT_TOKENS="131072"
MAX_NEW_TOKENS="8192"
FORMALITY_MAX_NEW_TOKENS="2048"
SIX_USER_TEN_MINUTE_REASONING_PROFILE="1"
```

该入口只作为首次十分钟工程 smoke，不是正式十分钟资源合同。它沿用当前六用户三分钟作业已经被调度并实测过的账号、H100、`96G/4h` 分配作为第一条测量链路，不写 partition/QOS/nodelist：

```bash
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --mem=96G
#SBATCH --time=04:00:00
```

`96G/4h` 只用于首次十分钟 smoke 测量，不能直接升级为正式十分钟资源合同。提交前必须用当前 Torch 的 `sbatch --test-only --parsable` 验证资源可接受性，并解决 association 显示 `normal`、历史提交使用 `gpu48` 的 QOS 差异。冲突未解决时禁止实际提交；smoke 完成后依据实际 Elapsed、MaxRSS、GPU peak 和完整 stage 计数设计正式资源。

- [ ] **Step 4: runtime 接收 profile 环境变量并构造 CLI 参数**

Runtime 默认继续保持当前 30 秒/2048/disable-thinking 行为。只有 `SIX_USER_TEN_MINUTE_REASONING_PROFILE=1` 时增加：

```bash
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
FORMALITY_MAX_NEW_TOKENS="${FORMALITY_MAX_NEW_TOKENS:-2048}"
SIX_USER_TEN_MINUTE_REASONING_PROFILE="${SIX_USER_TEN_MINUTE_REASONING_PROFILE:-0}"
PRUNING_MAX_CROSS_GAP_SECONDS="${PRUNING_MAX_CROSS_GAP_SECONDS:-10}"
export MAX_NEW_TOKENS FORMALITY_MAX_NEW_TOKENS
export SIX_USER_TEN_MINUTE_REASONING_PROFILE PRUNING_MAX_CROSS_GAP_SECONDS
```

构造参数时：

```bash
qa_profile_args+=(
  --six-user-ten-minute-reasoning-profile
  --formality-max-new-tokens "${FORMALITY_MAX_NEW_TOKENS}"
)
```

十分钟 wrapper 不传 `--disable-thinking`；三分钟 wrapper 继续沿用当前 `--disable-thinking`。

- [ ] **Step 5: 更新 runtime 的 segment 验收**

保留 accepted QA 必须有六个 `evidence_segment_observation` prompt rows，因为一行对应一个用户调用；同时增加每行 `segments` 长度必须等于 `EVIDENCE_DURATION_SECONDS/30`。十分钟要求每行 20，三分钟要求每行 6。

- [ ] **Step 6: 运行 GREEN 测试和 Bash 语法检查**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_ten_minute_reasoning_job_contract.py tests/test_three_minute_4h_job_contract.py
```

在可用 Bash 环境运行：

```bash
bash -n hpc/qa/experiments/run_six_user_qa_10min_reasoning.sbatch
bash -n hpc/qa/smoke/run_six_user_qa_runtime_probe.sbatch
```

Expected: pytest 全部通过，两个 Bash 命令返回 0。

- [ ] **Step 7: 提交 Task 6**

```powershell
git add -- hpc/qa/experiments/run_six_user_qa_10min_reasoning.sbatch hpc/qa/smoke/run_six_user_qa_runtime_probe.sbatch tests/test_ten_minute_reasoning_job_contract.py
git commit -m "feat: 新增六用户十分钟 reasoning 入口"
```

## Task 7：聚焦回归、文档和提交边界

**Files:**
- Modify: `docs/superpowers/specs/2026-08-28-six-user-10min-reasoning-evidence-answerability-design.md` only if implementation reveals an already-approved detail that needs exact wording; do not change semantics silently.
- Verify: all modified source/test/wrapper files.

- [ ] **Step 1: 运行完整聚焦测试集**

```powershell
.\.venv\Scripts\python.exe -m pytest -q `
  tests/test_schema_json_extraction.py `
  tests/test_qwen_runner_compat.py `
  tests/test_evidence_chunk_review.py `
  tests/test_six_user_prompts.py `
  tests/test_six_user_video_qa_loop.py `
  tests/test_zip_temporal_pruning.py `
  tests/test_three_minute_blockwise_pruning.py `
  tests/test_six_user_group_relative_sampling.py `
  tests/test_ten_minute_reasoning_job_contract.py `
  tests/test_three_minute_4h_job_contract.py
```

Expected: 0 failed。

- [ ] **Step 2: 运行 Python 静态编译和 Git whitespace 检查**

```powershell
.\.venv\Scripts\python.exe -m compileall -q `
  schema.py qwen3vl_runner.py prompts.py evidence_chunk_review.py `
  video_qa_loop.py group_relative_clip_sampling.py
git diff --check
```

Expected: 两条命令均返回 0。

- [ ] **Step 3: 检查三分钟行为未被十分钟默认覆盖**

```powershell
rg -n 'EVIDENCE_DURATION_SECONDS="180"|PRUNING_BLOCK_SECONDS="30"|MAX_GENERATION_SLOTS="3"' hpc/qa/experiments/run_six_user_qa_3min_4h.sbatch
rg -n 'max_cross_gap_seconds: float = 10.0' group_relative_clip_sampling.py
```

Expected: 三分钟 wrapper 和通用默认仍保持原值。

- [ ] **Step 4: 审计暂存范围**

```powershell
git status --short
git diff --name-only
git diff --cached --name-only
```

禁止暂存：

- 当前用户未提交的 `hpc/qa/experiments/run_six_user_qa_3min_4h.sbatch`；
- 当前用户未提交的 `tests/test_three_minute_4h_job_contract.py`，除非实施确实需要在该文件增加不覆盖原改动的回归断言并单独审查；
- `prompts.py.before_zip_20260826.bak`；
- `.codex_runtime/`；
- `tools/render_six_user_human_review.py`；
- 任何运行产物、视频、缓存或日志。

- [ ] **Step 5: 生成本地验证摘要，不提交远端作业**

摘要必须分别报告：

- RED 测试及预期失败；
- GREEN 聚焦测试数；
- 三分钟回归；
- 十分钟 wrapper 静态合同；
- 远端未验证边界；
- QOS association 冲突仍需在提交前解决；
- 未执行 Slurm 提交或取消。

- [ ] **Step 6: 提交剩余经审查的实施文件**

```powershell
git add -- schema.py qwen3vl_runner.py prompts.py evidence_chunk_review.py video_qa_loop.py group_relative_clip_sampling.py hpc/qa/smoke/run_six_user_qa_runtime_probe.sbatch hpc/qa/experiments/run_six_user_qa_10min_reasoning.sbatch tests/test_schema_json_extraction.py tests/test_qwen_runner_compat.py tests/test_evidence_chunk_review.py tests/test_six_user_prompts.py tests/test_six_user_video_qa_loop.py tests/test_zip_temporal_pruning.py tests/test_three_minute_blockwise_pruning.py tests/test_six_user_group_relative_sampling.py tests/test_ten_minute_reasoning_job_contract.py
git commit -m "feat: 扩展六用户十分钟 reasoning QA"
```

如果前面各 Task 已分别提交且不存在剩余修改，则本步骤只做状态审计，不创建空提交。

## 执行检查点

执行使用 `executing-plans`，按以下三个批次汇报：

1. Batch A：Task 1–2，JSON parser 与 stage profile；
2. Batch B：Task 3–4，evidence 与 answerability；
3. Batch C：Task 5–7，十分钟 pruning/wrapper 与完整回归。

每个批次必须提供实际 RED 和 GREEN 输出。任何测试失败先停在该批次定位，不继续叠加修改。完成本地实现后停在远端提交门口；只有用户再次明确授权，才进行窄范围同步、一次最小 smoke 或正式 Slurm 提交。
