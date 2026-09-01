# Six-User QA Loop Speed Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改变 3×20 总量、每 slot 三次 attempt、pruning 和 fast profile 核心边界的前提下，同步两次 answerability 的 facts、统一简单 evidence、删除 `why_two_users_needed`，并让每组所有合格 speaker 均衡获得 QA slot。

**Architecture:** 保留现有 `video_qa_loop.py` 主流程，通过一个确定性的 canonical-fact 对齐 helper 连接 speaker-only 与 all-six；生产 evidence dispatcher 始终进入单次六视频 judge；group-aware slot scheduler 先按 generation group 均分二十个 slot，再在组内轮转 speaker packet。候选挖掘仍尝试全部 speaker，只取消正式 wrapper 的事后单候选截断。

**Tech Stack:** Python 3.11、pytest、JSONL、Bash/Slurm 合同文本测试；本地不加载 Qwen、不执行真实视频推理、不提交远端作业。

---

## 文件结构与修改职责

- `prompts.py`：增加 answerability canonical facts prompt 合同；删除 generator 的 `why_two_users_needed`。
- `schema.py`：删除严格 schema 和 Markdown renderer 对 `why_two_users_needed` 的要求。
- `video_qa_loop.py`：同步两次 answerability facts；生产 evidence 固定简单调用；停止补写/记录被删除字段。
- `qa_generation_schedule.py`：实现 generation-group 均衡、组内 speaker 轮转的 60-slot 调度。
- `group_relative_clip_sampling.py`：在 candidate packet 顶层显式保存 `speaker_index`，保留所有通过 speaker。
- `hpc/qa/smoke/run_six_user_qa_runtime_probe.sbatch`：取消生产单候选截断依赖，验收简单 evidence 和 group/speaker slot 配额。
- `hpc/qa/experiments/run_six_user_qa_10min_3groups_x20*.sbatch`：将 `ONE_CANDIDATE_PER_GROUP` 固定为 `0`。
- `tests/test_six_user_prompts.py`：prompt 与 generator schema 回归。
- `tests/test_six_user_video_qa_loop.py`：answerability facts 同步和简单 evidence 回归。
- `tests/test_time_budget_qa_loop.py`：三组各二十 slot、组内 speaker 均衡回归。
- `tests/test_six_user_group_relative_sampling.py`：candidate packet speaker identity 回归。
- `tests/test_ten_minute_reasoning_job_contract.py`、`tests/test_six_user_torch_job_contract.py`：wrapper 和运行验收合同回归。
- `tests/test_six_user_10min_review.py`：渲染结果不再展示被删除字段。

批准规格明确禁止自动提交实现代码；因此各任务的最后一步使用“检查窄 diff”代替 Git commit。不得暂存或提交用户现有 dirty changes。

### Task 1: Canonical facts prompt 与确定性对齐 helper

**Files:**
- Modify: `prompts.py:104-163,1865-1951`
- Modify: `video_qa_loop.py:1088-1278,3021-3152`
- Test: `tests/test_six_user_prompts.py`
- Test: `tests/test_six_user_video_qa_loop.py`

- [ ] **Step 1: 写 speaker fact ID 和 all-six 固定 facts 的失败测试**

在 `tests/test_six_user_prompts.py` 增加：

```python
def test_all_six_answerability_prompt_reuses_canonical_facts() -> None:
    condition = {
        "condition_id": "combined_all_six_users::speaker+provider_one+provider_two+provider_three+provider_four+provider_five",
        "condition_type": "combined_all_six_users",
        "users": list(SIX_USERS),
    }
    canonical = [
        {"fact_id": "F1", "fact": "the final destination", "why_needed": "it distinguishes the options"},
        {"fact_id": "F2", "fact": "the final object state", "why_needed": "it identifies the correct outcome"},
    ]

    prompt = build_answerability_prompt(
        six_user_qa(),
        condition,
        canonical_facts=canonical,
    )

    assert '"fact_id": "F1"' in prompt
    assert "must not add, delete, reorder, merge, split, or rewrite facts" in prompt
```

在 `tests/test_six_user_video_qa_loop.py` 增加：

```python
def test_answerability_fact_contract_rejects_rewritten_all_six_fact() -> None:
    speaker = sufficiency_evaluation(build_answerability_conditions(SIX_USERS)[0], False)
    speaker["needed_facts"][0]["fact_id"] = "F1"
    all_six = sufficiency_evaluation(build_answerability_conditions(SIX_USERS)[1], True)
    all_six["needed_facts"][0].update({"fact_id": "F1", "fact": "a rewritten fact"})

    result = answerability_gate(six_user_qa(), [speaker, all_six])

    assert result["passed"] is False
    assert result["failure_label"] == "answerability_fact_contract_mismatch"
```

- [ ] **Step 2: 运行测试并确认 RED**

运行：

```powershell
& .\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider `
  tests/test_six_user_prompts.py::test_all_six_answerability_prompt_reuses_canonical_facts `
  tests/test_six_user_video_qa_loop.py::test_answerability_fact_contract_rejects_rewritten_all_six_fact
```

预期：FAIL；第一个测试报告 `canonical_facts` 参数不存在，第二个测试未返回 `answerability_fact_contract_mismatch`。

- [ ] **Step 3: 扩展 answerability schema 和 prompt**

在 `ANSWERABILITY_SUFFICIENCY_SCHEMA` 的每个 fact 中增加：

```python
"fact_id": {
    "type": "string",
    "minLength": 1,
},
```

将 prompt builder 改为：

```python
def build_answerability_prompt(
    qa_item: dict[str, Any],
    condition: dict[str, Any],
    *,
    canonical_facts: list[dict[str, Any]] | None = None,
) -> str:
```

speaker-only prompt 要求按 `F1`、`F2` 顺序生成唯一 `fact_id`。all-six 在 `canonical_facts` 非空时插入只含 `fact_id/fact/why_needed` 的 JSON，并加入：

```text
You must return exactly these canonical facts in the same order. You must not add,
delete, reorder, merge, split, or rewrite facts. Preserve fact_id, fact, and
why_needed verbatim. Re-evaluate only visibility, confidence, source_user,
original_time_range, and visual_description from the supplied six videos.
```

- [ ] **Step 4: 增加确定性对齐 helper**

在 `video_qa_loop.py` 增加：

```python
def canonical_fact_contract_errors(
    speaker_evaluation: dict[str, Any],
    all_six_evaluation: dict[str, Any],
) -> list[str]:
    speaker_facts = speaker_evaluation.get("needed_facts")
    all_six_facts = all_six_evaluation.get("needed_facts")
    if not isinstance(speaker_facts, list) or not isinstance(all_six_facts, list):
        return ["both conditions must contain needed_facts arrays"]
    if len(speaker_facts) != len(all_six_facts):
        return [f"fact count changed: speaker={len(speaker_facts)} all_six={len(all_six_facts)}"]

    errors: list[str] = []
    seen_ids: set[str] = set()
    for index, (speaker_fact, all_six_fact) in enumerate(zip(speaker_facts, all_six_facts)):
        if not isinstance(speaker_fact, dict) or not isinstance(all_six_fact, dict):
            errors.append(f"needed_facts[{index}] must be objects")
            continue
        fact_id = str(speaker_fact.get("fact_id") or "").strip()
        if not fact_id:
            errors.append(f"speaker needed_facts[{index}].fact_id must be non-empty")
        elif fact_id in seen_ids:
            errors.append(f"speaker fact_id is duplicated: {fact_id}")
        seen_ids.add(fact_id)
        for key in ("fact_id", "fact", "why_needed"):
            if all_six_fact.get(key) != speaker_fact.get(key):
                errors.append(f"needed_facts[{index}].{key} changed")
    return errors
```

`answerability_gate()` 在解析 all-six sufficiency 前调用该 helper；有错误时返回：

```python
{
    "passed": False,
    "reason": "all-six answerability changed canonical facts: " + "; ".join(errors),
    "failure_label": "answerability_fact_contract_mismatch",
    **metrics,
}
```

`parsed_answerability_sufficiency()` 将 `fact_id` 加入必填字段，并拒绝空或重复 ID。

同时更新 `tests/test_six_user_video_qa_loop.py` 的 `sufficiency_evaluation()` 测试 helper，使默认 fact 带有 `fact_id="F1"`；所有直接构造 `needed_facts` 的六用户测试 fixture 均补稳定 ID，避免把旧 fixture 缺字段误判成新逻辑回归。

- [ ] **Step 5: 运行测试并确认 GREEN**

运行 Step 2 同一命令。预期：2 passed。

- [ ] **Step 6: 检查窄 diff**

运行：

```powershell
git diff --check -- prompts.py video_qa_loop.py tests/test_six_user_prompts.py tests/test_six_user_video_qa_loop.py
```

预期：无输出，exit code 0。不要暂存或提交。

### Task 2: 两次 Answerability 真实共享同一份 facts

**Files:**
- Modify: `video_qa_loop.py:3021-3152,3449-3690`
- Test: `tests/test_six_user_video_qa_loop.py`

- [ ] **Step 1: 写 run_answerability_eval 和 fail-fast 的失败测试**

增加测试，使用 fake condition evaluator 记录 `canonical_facts`：

```python
def test_six_user_answerability_passes_speaker_facts_to_all_six() -> None:
    observed: list[tuple[str, object]] = []

    def fake_condition_eval(*, condition, canonical_facts=None, **_kwargs):
        observed.append((condition["condition_type"], canonical_facts))
        row = sufficiency_evaluation(
            condition,
            condition["condition_type"] == "combined_all_six_users",
        )
        row["needed_facts"][0]["fact_id"] = "F1"
        if canonical_facts:
            row["needed_facts"][0].update(canonical_facts[0])
        return row

    with mock.patch.object(
        video_qa_loop,
        "run_answerability_condition_eval",
        side_effect=fake_condition_eval,
    ):
        result = run_answerability_eval(
            qa_item=six_user_qa(),
            packet=six_user_packet(),
            runner=object(),
            media_backend="transformers-local",
            allow_openai_video_input=False,
            prompt_rows=[],
        )

    assert result["gate"]["passed"] is True
    assert observed[0] == ("speaker_only", None)
    assert observed[1][0] == "combined_all_six_users"
    assert observed[1][1] == [
        {"fact_id": "F1", "fact": "the later destination", "why_needed": "the question asks where the object ended up"}
    ]
```

- [ ] **Step 2: 运行并确认 RED**

运行：

```powershell
& .\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider `
  tests/test_six_user_video_qa_loop.py::test_six_user_answerability_passes_speaker_facts_to_all_six
```

预期：FAIL，因为 `run_answerability_condition_eval()` 不接收 `canonical_facts`，或 all-six 收到 `None`。

- [ ] **Step 3: 写最小实现**

为 `run_answerability_condition_eval()` 增加：

```python
canonical_facts: list[dict[str, Any]] | None = None,
```

并传给 `build_answerability_prompt()`。prompt row 记录只含 identity 字段的 `canonical_facts`，不重复保存可见性结果。

为 `run_answerability_eval()` 的六用户分支显式按 speaker、all-six 顺序执行。speaker 结果生成：

```python
canonical_facts = [
    {key: fact.get(key) for key in ("fact_id", "fact", "why_needed")}
    for fact in speaker_evaluation.get("needed_facts") or []
    if isinstance(fact, dict)
]
```

all-six 调用传入该列表。非六用户路径保持现有循环行为。

`run_fail_fast_review_judges()` 在 speaker 不足后采用同一提取逻辑，并把 `canonical_facts` 传给 all-six；speaker sufficient 的前两轮仍不启动 all-six。

- [ ] **Step 4: 运行并确认 GREEN**

运行 Step 2 命令，并加上现有 fail-fast 两项测试：

```powershell
& .\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider `
  tests/test_six_user_video_qa_loop.py::test_six_user_answerability_passes_speaker_facts_to_all_six `
  tests/test_six_user_video_qa_loop.py::SixUserAnswerabilityTests::test_fail_fast_review_stops_after_formality_on_first_attempt `
  tests/test_six_user_video_qa_loop.py::SixUserAnswerabilityTests::test_fail_fast_review_third_attempt_runs_every_metric_after_failure
```

预期：3 passed。

- [ ] **Step 5: 检查窄 diff**

运行 `git diff --check -- video_qa_loop.py tests/test_six_user_video_qa_loop.py`。预期无输出；不提交。

### Task 3: 生产 Evidence 固定简单单次调用

**Files:**
- Modify: `video_qa_loop.py:3358-3438,3691-4062`
- Modify: `hpc/qa/smoke/run_six_user_qa_runtime_probe.sbatch:681-821,961-969`
- Test: `tests/test_six_user_video_qa_loop.py`
- Test: `tests/test_ten_minute_reasoning_job_contract.py`
- Test: `tests/test_six_user_torch_job_contract.py`

- [ ] **Step 1: 写六段和二十段都只调用简单 evidence 的失败测试**

增加一个覆盖六段和二十段的测试：

```python
def test_production_evidence_uses_one_full_video_call_for_all_segment_counts() -> None:
    for segment_count in (6, 20):
        packet = six_user_packet()
        for clip in packet["clips"]:
            clip["segments"] = [
                {"time_token": f"segment-{index:02d}"}
                for index in range(segment_count)
            ]
        prompt_rows: list[dict[str, object]] = []

        with (
            mock.patch.object(video_qa_loop, "run_model_judge_branch", return_value={
                "review_passed": True,
                "checks": {"evidence_groundedness": {"status": "PASS", "reason": "supported", "fix": ""}},
                "blocking_failures": [],
                "feedback_to_generator": "",
                "raw_output": "{}",
                "elapsed_seconds": 0.1,
            }) as simple,
            mock.patch.object(video_qa_loop, "run_chunked_evidence_groundedness_eval") as chunked,
        ):
            result = video_qa_loop.run_evidence_groundedness_review(
                qa_item=six_user_qa(),
                packet=packet,
                runner=object(),
                prompt_rows=prompt_rows,
                full_image_paths=[],
                full_video_paths=[f"/full/{user}.mp4" for user in SIX_USERS],
                attempt=1,
                judge_media_role="full",
                stage_profiles=video_qa_loop.six_user_ten_minute_fast_profiles(),
            )

        simple.assert_called_once()
        chunked.assert_not_called()
        assert result["checks"]["evidence_groundedness"]["status"] == "PASS"
        assert [row["stage"] for row in prompt_rows] == ["evidence_groundedness_judge"]
```

- [ ] **Step 2: 运行并确认 RED**

运行该测试。预期：循环中的 `segment_count=6` 分支失败，因为当前 dispatcher 会进入 chunked evidence。

- [ ] **Step 3: 删除生产 dispatcher 的 chunked 分支**

`run_evidence_groundedness_review()` 始终构造一条 `evidence_groundedness_judge` row 并调用 `run_model_judge_branch()`。保留 `run_chunked_evidence_groundedness_eval()` 供离线直接调用，但生产函数不再检查 segment 数量。

`run_parallel_review_judges()` 同样始终提交 simple evidence future，并移除 chunk rows 的生产 trace 拼接。fast profile 的 simple call 继续读取 `stage_profiles["evidence_groundedness"]`，保持 thinking/4096。

- [ ] **Step 4: 先更新 `.sbatch` 合同测试并确认 RED**

在 job-contract 测试中要求：

```python
assert 'stage="evidence_groundedness_judge"' in runtime
assert "accepted QA must have exactly 1 simple evidence groundedness call" in runtime
assert '"groundedness_video_count": 6' in runtime
assert '"evidence_segment_observation_count": 0' in runtime
assert '"evidence_groundedness_aggregation_count": 0' in runtime
assert "accepted QA must have 6 evidence segment observations" not in runtime
```

运行测试，预期因旧 validator 仍要求 6+1 而失败。

- [ ] **Step 5: 修改 runtime validator**

使用 `prompt_rows_by_generation_identity(..., stage="evidence_groundedness_judge")` 建立 identity 索引。每条 accepted QA 要求恰好一条 simple evidence row、六个 full video、`media_role="full"`。删除 accepted 路径中的 segment/aggregation 必需断言；摘要固定：

```python
"groundedness_video_count": 6,
"evidence_segment_observation_count": 0,
"evidence_groundedness_aggregation_count": 0,
```

- [ ] **Step 6: 运行并确认 GREEN**

运行：

```powershell
& .\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider `
  tests/test_six_user_video_qa_loop.py `
  tests/test_ten_minute_reasoning_job_contract.py `
  tests/test_six_user_torch_job_contract.py
```

预期：相关测试全部通过。

- [ ] **Step 7: 检查窄 diff**

运行 `git diff --check`，限定上述五个文件；不提交。

### Task 4: 删除 `why_two_users_needed`，保留其他 Generator 字段

**Files:**
- Modify: `prompts.py:12-68,1427-1455`
- Modify: `schema.py:64-73,449-463`
- Modify: `video_qa_loop.py:706-823,2961-3019,4700-4777`
- Modify: `tools/render_six_user_10min_review.py`
- Test: `tests/test_six_user_prompts.py`
- Test: `tests/test_six_user_video_qa_loop.py`
- Test: `tests/test_six_user_10min_review.py`

- [ ] **Step 1: 写 schema、prompt 和渲染失败测试**

增加断言：

```python
def test_generator_schema_removes_only_why_two_users_needed() -> None:
    assert "why_two_users_needed" not in VIDEO_GENERATION_SCHEMA
    for field in (
        "qa_id", "question_type", "question", "options", "correct", "answer",
        "required_users", "evidence", "referred_timestamps",
        "single_user_answerability", "combined_answerability",
        "generator_rationale", "per_user_evidence_claims", "review",
    ):
        assert field in VIDEO_GENERATION_SCHEMA
```

构造不含该字段、但含其他 strict fields 的 QA，断言 `validate_qa_item(..., strict_review=True)` 不报告 missing field。渲染测试断言结果中不出现 `Why Two Users Are Needed` 或 `why_two_users_needed`。

- [ ] **Step 2: 运行并确认 RED**

运行三份聚焦测试。预期：prompt/schema/renderer 仍含旧字段，测试失败。

- [ ] **Step 3: 写最小实现**

- 从 `VIDEO_GENERATION_SCHEMA` 和六用户 override 删除字段；
- 从 `VIDEO_FIRST_REQUIRED_FIELDS` 删除字段；
- `complete_generator_metadata()` 执行 `qa.pop("why_two_users_needed", None)`，不再补默认值；
- dry-run QA、parsed/normalized trace 摘要不再写该字段；
- `schema.py` 的通用 Markdown renderer 删除 `Why Two Users Are Needed` 标题和值；
- `tools/render_six_user_10min_review.py` 将 `qa.get("generator_rationale") or qa.get("why_two_users_needed")` 改为只读取 `qa.get("generator_rationale")`；
- 不改 `evidence`、`referred_timestamps` 或任何其他 generator 字段；
- 读取旧 artifact 时额外字段继续被忽略，不添加禁止-extra-fields 校验。

- [ ] **Step 4: 运行并确认 GREEN**

运行：

```powershell
& .\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider `
  tests/test_six_user_prompts.py `
  tests/test_six_user_video_qa_loop.py `
  tests/test_six_user_10min_review.py
```

预期：全部通过。

- [ ] **Step 5: 检查窄 diff**

运行 `git diff --check`，限定本任务文件；不提交。

### Task 5: 保留 Speaker Identity 并实现 Group-Balanced Slot Scheduler

**Files:**
- Modify: `group_relative_clip_sampling.py:2982-3055`
- Modify: `qa_generation_schedule.py:17-75`
- Modify: `video_qa_loop.py:4270-4320`
- Test: `tests/test_six_user_group_relative_sampling.py`
- Test: `tests/test_time_budget_qa_loop.py`

- [ ] **Step 1: 写 candidate packet speaker identity 的失败测试**

在现有 six-user candidate 测试中断言：

```python
packet = build_candidate_packet(candidate_result)
assert packet["speaker_index"] == candidate_result["selection"]["speaker_index"]
assert packet["speaker_user"] == packet["required_users"][0]
```

保留并重跑已有的全部 speaker 结果断言：当 speaker 1、2、5、6 通过时，`speaker_candidates` 的 speaker index 必须完整等于 `[1, 2, 4, 5]`，不能在 candidate miner 内提前停止。

运行测试，预期因 packet 顶层缺少 `speaker_index` 而失败。

- [ ] **Step 2: 写三组不等 speaker 数量的调度失败测试**

在 `tests/test_time_budget_qa_loop.py` 增加直接 scheduler 测试。构造：

```python
packets = [
    *packets_for_group("group-a", speakers=[0, 1, 2, 3, 4, 5]),
    *packets_for_group("group-b", speakers=[0, 2, 5]),
    *packets_for_group("group-c", speakers=[1, 4]),
]
slots = list(round_robin_generation_slots(packets, max_slots=60))
```

断言：

```python
assert Counter(slot["generation_group_id"] for slot in slots) == {
    "group-a": 20,
    "group-b": 20,
    "group-c": 20,
}
for group_id in ("group-a", "group-b", "group-c"):
    per_speaker = Counter(
        slot["speaker_index"] for slot in slots
        if slot["generation_group_id"] == group_id
    )
    assert min(per_speaker.values()) >= 2
```

运行测试，预期当前 packet-level round-robin 不能保证每组二十而失败。

- [ ] **Step 3: 写最小 candidate 与 scheduler 实现**

`build_candidate_packet()` 在 six-user speaker consensus packet 顶层增加：

```python
"speaker_index": int(selection["speaker_index"]),
```

将 `round_robin_generation_slots()` 改为按首次出现顺序建立 group buckets：

```python
grouped: dict[str, list[dict[str, Any]]] = {}
for packet in available:
    group_id = str(packet.get("generation_group_id") or packet["evidence_id"])
    grouped.setdefault(group_id, []).append(packet)

emitted = 0
group_round = 0
while max_slots is None or emitted < max_slots:
    for group_packets in grouped.values():
        if max_slots is not None and emitted >= max_slots:
            return
        packet = group_packets[group_round % len(group_packets)]
        slot = dict(packet)
        slot["base_evidence_id"] = str(packet["evidence_id"])
        slot["generation_round_index"] = group_round
        slot["generation_diversity_focus"] = diversity_focus_for_round(packet, group_round)
        slot["generation_slot_id"] = generation_slot_id(str(packet["evidence_id"]), group_round)
        yield slot
        emitted += 1
    group_round += 1
```

空输入保持现有空 iterator 行为；缺少 `generation_group_id` 的普通 packet 以自身 evidence ID 作为独立 group，保持旧单 packet 语义。

- [ ] **Step 4: 运行并确认 GREEN**

运行两份聚焦测试。预期通过，并确认六 speaker 的二十 slot 分布为两个 4、四个 3，顺序确定。

- [ ] **Step 5: 检查窄 diff**

运行 `git diff --check -- group_relative_clip_sampling.py qa_generation_schedule.py video_qa_loop.py tests/test_six_user_group_relative_sampling.py tests/test_time_budget_qa_loop.py`。预期无输出；不提交。

### Task 6: Wrapper 与最终 Slot 验收对齐

**Files:**
- Modify: `hpc/qa/experiments/run_six_user_qa_10min_3groups_x20.sbatch`
- Modify: `hpc/qa/experiments/run_six_user_qa_10min_3groups_x20_qwen38_fast.sbatch`
- Modify: `hpc/qa/experiments/run_six_user_qa_10min_3groups_x20_qwen38_fast_fix.sbatch`
- Modify: `hpc/qa/smoke/run_six_user_qa_runtime_probe.sbatch:184-205,365-445,860-969`
- Test: `tests/test_ten_minute_reasoning_job_contract.py`
- Test: `tests/test_time_budget_qa_loop.py`

- [ ] **Step 1: 写 wrapper 和结果验收失败测试**

断言每个 3×20 wrapper 包含：

```python
assert 'ONE_CANDIDATE_PER_GROUP="0"' in wrapper
assert 'MAX_GENERATION_SLOTS="60"' in wrapper
assert 'EXPECTED_QA_PER_GROUP="20"' in wrapper
```

runtime contract 断言候选过滤不会运行 `selected_by_group.setdefault`，结果摘要包含：

```text
completed_slots_by_group_and_speaker
minimum_slots_per_eligible_speaker
speaker_slot_target_reached
```

并要求 `minimum_slots_per_eligible_speaker=2`。

- [ ] **Step 2: 运行并确认 RED**

运行 `tests/test_ten_minute_reasoning_job_contract.py` 和相关 time-budget contract。预期旧 wrapper 的值为 `1`、runtime 缺少 per-group speaker 验收而失败。

- [ ] **Step 3: 修改 wrapper 与 candidate audit**

三个 3×20 wrapper 固定 `ONE_CANDIDATE_PER_GROUP="0"`。runtime candidate audit 不再重写 `six_user_candidates.jsonl` 为每组第一条；仍检查恰好三个 generation groups、每条 packet 六视频媒体完整、`speaker_index` 在 `0..5`、同组不重复 speaker。

- [ ] **Step 4: 修改最终结果验收**

从 accepted/rejected/time-budget-partial rows 按 `(generation_group_id, speaker_index)` 计数。对 candidate manifest 中所有合格 `(group, speaker)` 建立期望集合，要求：

```python
all(completed_slots_by_group[group_id] >= 20 for group_id in generation_group_ids)
all(completed_slots_by_group_and_speaker[key] >= 2 for key in eligible_group_speakers)
```

摘要写入每组和每 speaker 计数、最低配额 2 与布尔状态；不要求每 speaker 至少两条 accepted QA。

- [ ] **Step 5: 运行并确认 GREEN**

运行：

```powershell
& .\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider `
  tests/test_time_budget_qa_loop.py `
  tests/test_ten_minute_reasoning_job_contract.py `
  tests/test_six_user_torch_job_contract.py
```

预期全部通过。

- [ ] **Step 6: 检查窄 diff**

运行 `git diff --check`，限定 wrapper、runtime 和对应测试；不提交。

### Task 7: 聚焦回归、静态检查与交付审计

**Files:**
- Verify only; no new production behavior.

- [ ] **Step 1: 运行完整聚焦测试集合**

```powershell
& .\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider `
  tests/test_six_user_prompts.py `
  tests/test_six_user_video_qa_loop.py `
  tests/test_time_budget_qa_loop.py `
  tests/test_six_user_group_relative_sampling.py `
  tests/test_six_user_10min_review.py `
  tests/test_ten_minute_reasoning_job_contract.py `
  tests/test_six_user_torch_job_contract.py `
  tests/test_review_retry_loop.py
```

预期：exit code 0，零失败。必须记录实际 passed 数，不在计划中预填虚构计数。

- [ ] **Step 2: 运行 Python 编译检查**

```powershell
& .\.venv\Scripts\python.exe -m py_compile `
  prompts.py schema.py video_qa_loop.py qa_generation_schedule.py `
  group_relative_clip_sampling.py tools/render_six_user_10min_review.py
```

预期：无输出，exit code 0。

- [ ] **Step 3: 检查嵌入 Python 与 wrapper 合同**

`tests/test_ten_minute_reasoning_job_contract.py` 和 `tests/test_six_user_torch_job_contract.py` 必须解析 runtime `.sbatch` 中的 heredoc Python；不得用 Windows Bash 的可用性代替远端 `bash -n` 结论。本轮不连接 Torch，因此 Bash runtime 仍标记“远端未验证”。

- [ ] **Step 4: 检查全部任务 diff**

```powershell
git diff --check
git status --short --branch
git diff --stat
```

确认：

- 没有修改 `qwen3vl_runner.py` 或 pruning 数学；
- 没有删除或覆盖用户原有 dirty changes；
- 没有暂存实现文件；
- 没有提交、push、上传、提交 Slurm 或取消 Job；
- 新鲜测试证据与“远端未验证”边界分别报告。

- [ ] **Step 5: 逐项对照规格交付**

逐项确认：2-call facts 同步、simple evidence、删除单字段、所有 speaker packet、三组各二十 slot、每 speaker 至少两个 slot、每 slot 三 attempts。若任一项没有测试和代码证据，不得声称完成。
