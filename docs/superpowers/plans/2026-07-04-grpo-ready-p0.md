# GRPO-Ready P0 实施计划

> **面向代理执行者：** 必须使用 `executing-plans` 或 `subagent-driven-development` 逐项实施。本计划使用复选框跟踪进度。

**目标：** 将 55 个历史 packet 中的 121 个生成 attempt 恢复为稳定记录，计算 Reward v0，并输出可复现的回放结果与汇总。

**架构：** 新增独立 `grpo_ready` 包。提取层只负责把嵌套历史记录规范化；奖励层是无 I/O 纯函数；回放层负责流式读取、输出与统计。现有 `video_qa_loop.py` 保持不变。

**技术栈：** Python 3.10+、标准库 `dataclasses/json/csv/statistics/argparse/unittest`，复用 `schema.extract_json_object` 与 `schema.validate_qa_item`。

---

## 文件结构

- 新建 `grpo_ready/__init__.py`：公开稳定类型与版本。
- 新建 `grpo_ready/records.py`：`AttemptRecord`、`RewardRecord` 及字典序列化。
- 新建 `grpo_ready/extract_attempts.py`：packet/attempt 扁平化与真实输入校验。
- 新建 `grpo_ready/rewards.py`：Reward v0 纯函数及 evaluator 观测提取。
- 新建 `grpo_ready/replay.py`：CLI、JSONL/CSV、汇总、矛盾案例与 manifest。
- 新建 `tests/grpo_ready/__init__.py`：测试包。
- 新建 `tests/grpo_ready/test_extract_attempts.py`：构造数据与真实数据提取测试。
- 新建 `tests/grpo_ready/test_rewards.py`：奖励分项和缺失值测试。
- 新建 `tests/grpo_ready/test_replay.py`：输出契约集成测试。

## 任务 1：记录类型与 attempt 提取

**文件：**

- 新建：`grpo_ready/__init__.py`
- 新建：`grpo_ready/records.py`
- 新建：`grpo_ready/extract_attempts.py`
- 新建：`tests/grpo_ready/__init__.py`
- 新建：`tests/grpo_ready/test_extract_attempts.py`

- [ ] **步骤 1：先写失败测试**

测试构造 accepted 与 rejected packet，断言 attempt 自身的 `result.accepted`
决定标签，并断言 raw output、prompt、媒体、judge、answerability 均被保留：

```python
class ExtractAttemptsTests(unittest.TestCase):
    def test_attempt_acceptance_does_not_inherit_packet_status(self):
        packet = fixture_packet(status="accepted", accepted_attempt=1)
        rows = extract_packet_attempts(packet)
        self.assertEqual([row.accepted for row in rows], [False, True])
        self.assertEqual(rows[0].raw_qa, '{"question": "first"}')

    def test_real_intermediate_contains_all_attempts(self):
        rows = list(iter_attempt_records(INPUT_PATH))
        self.assertEqual(len(rows), 121)
        self.assertEqual(sum(row.accepted for row in rows), 27)
        self.assertTrue(all(row.raw_qa.strip() for row in rows))
```

- [ ] **步骤 2：运行测试并确认因模块不存在而失败**

运行：`python -m unittest tests.grpo_ready.test_extract_attempts -v`

预期：`ModuleNotFoundError: No module named 'grpo_ready'`。

- [ ] **步骤 3：实现最小记录类型**

`AttemptRecord` 明确定义以下字段，并使用 `dataclasses.asdict` 序列化：

```python
@dataclass(frozen=True)
class AttemptRecord:
    attempt_id: str
    evidence_id: str
    packet_status: str
    question_type: str | None
    mode: str | None
    attempt_index: int
    feedback: str
    generator_prompt: str
    generator_image_paths: tuple[str, ...]
    generator_video_paths: tuple[str, ...]
    evaluator_image_paths: tuple[str, ...]
    evaluator_video_paths: tuple[str, ...]
    raw_qa: str
    parsed_qa: dict[str, Any] | None
    schema_errors: tuple[str, ...]
    judge: dict[str, Any] | None
    answerability: dict[str, Any] | None
    accepted: bool
```

- [ ] **步骤 4：实现最小提取器**

`extract_packet_attempts(packet)` 遍历 `packet["attempts"]`；从
`generation.raw_output` 恢复 raw，从 `generation.parsed_qa` 优先获取 parsed，
否则调用 `extract_json_object`；schema errors 使用 `validate_qa_item`；accepted
只使用 `attempt.result.accepted is True`。`iter_attempt_records(path)` 对损坏行、
空 raw 和缺少 attempts 抛出包含行号/evidence_id 的 `ValueError`。

- [ ] **步骤 5：运行测试并确认通过**

运行：`python -m unittest tests.grpo_ready.test_extract_attempts -v`

预期：全部通过，真实数据断言 121 attempts、27 accepted、0 个空 raw。

- [ ] **步骤 6：提交该任务**

```powershell
git add grpo_ready tests/grpo_ready
git commit -m "feat: extract historical GRPO attempts"
```

## 任务 2：Reward v0 纯函数

**文件：**

- 修改：`grpo_ready/records.py`
- 新建：`grpo_ready/rewards.py`
- 新建：`tests/grpo_ready/test_rewards.py`

- [ ] **步骤 1：先写奖励失败测试**

```python
class RewardTests(unittest.TestCase):
    def test_parse_failure_only_scores_parse_component(self):
        reward = compute_reward(attempt(parse_success=False))
        self.assertEqual(reward.parse_reward, -2.0)
        self.assertIsNone(reward.schema_reward)
        self.assertIn("groundedness", reward.missing_components)
        self.assertFalse(reward.is_complete_reward)

    def test_full_pass_scores_expected_total(self):
        reward = compute_reward(full_attempt(speaker_choice="insufficient", provider_choice="A"))
        self.assertEqual(reward.total, 5.0)
        self.assertEqual(reward.provider_alone_reward, 0.0)
        self.assertTrue(reward.is_complete_reward)

    def test_speaker_correct_is_penalized(self):
        reward = compute_reward(full_attempt(speaker_choice="A"))
        self.assertEqual(reward.speaker_leakage_reward, -2.0)

    def test_missing_judge_is_not_treated_as_failure(self):
        reward = compute_reward(full_attempt(judge=None))
        self.assertIsNone(reward.formality_reward)
        self.assertIsNone(reward.groundedness_reward)
```

- [ ] **步骤 2：运行测试并确认因接口不存在而失败**

运行：`python -m unittest tests.grpo_ready.test_rewards -v`

预期：导入 `compute_reward` 失败。

- [ ] **步骤 3：实现 RewardRecord 与 Reward v0**

`RewardRecord` 保存 parse/schema/formality/groundedness/combined、speaker/provider
观测，各分项 reward、`total`、`reward_version="v0"`、missing components 与
complete 标记。`compute_reward` 使用以下映射：parse `+0.5/-2.0`；schema
`+1.0/-0.5`；formality `+0.5/-0.5`；groundedness `+2.0/-2.0`；combined
`+1.0/-1.0`；speaker correct `-2.0`，否则 `0.0`；provider correct 始终 `0.0`。
未观测项为 `None`，`total` 只求已观测分项之和。

- [ ] **步骤 4：运行奖励测试并确认通过**

运行：`python -m unittest tests.grpo_ready.test_rewards -v`

预期：全部通过。

- [ ] **步骤 5：运行提取与奖励联合回归**

运行：`python -m unittest tests.grpo_ready.test_extract_attempts tests.grpo_ready.test_rewards -v`

预期：全部通过。

- [ ] **步骤 6：提交该任务**

```powershell
git add grpo_ready tests/grpo_ready/test_rewards.py
git commit -m "feat: compute GRPO reward v0"
```

## 任务 3：Replay 输出与汇总

**文件：**

- 新建：`grpo_ready/replay.py`
- 新建：`tests/grpo_ready/test_replay.py`

- [ ] **步骤 1：先写 CLI 集成失败测试**

```python
class ReplayTests(unittest.TestCase):
    def test_replay_writes_complete_output_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_replay(INPUT_PATH, Path(tmp))
            expected = {
                "reward_replay_results.jsonl",
                "reward_replay_results.csv",
                "reward_replay_summary.json",
                "reward_replay_summary.md",
                "run_manifest.json",
            }
            self.assertTrue(expected.issubset({p.name for p in Path(tmp).iterdir()}))
            self.assertEqual(summary["attempt_count"], 121)
            self.assertEqual(summary["accepted_count"], 27)
            self.assertLessEqual(len(summary["contradiction_cases"]), 5)
```

- [ ] **步骤 2：运行测试并确认因 `run_replay` 不存在而失败**

运行：`python -m unittest tests.grpo_ready.test_replay -v`

预期：导入失败。

- [ ] **步骤 3：实现回放编排和文件输出**

`run_replay(input_path, output_dir)` 创建目录、逐 attempt 计算奖励、写 JSONL；
CSV 使用字段并集且复杂字段 JSON 编码；summary 分别统计 accepted/rejected 的
数量、observed total 均值/中位数、complete-case 覆盖率，并统计每个 missing/
失败组件。矛盾案例按“accepted 低分或 rejected 高分”的距离排序，最多保留 5 条。

- [ ] **步骤 4：实现可复现 manifest 与 Markdown 汇总**

manifest 保存 UTC 时间、输入 SHA-256、git commit、reward version、Python 版本、
命令参数和环境中的 `SLURM_JOB_ID`。Markdown 只复述 summary 中的数字，不重新
计算统计。

- [ ] **步骤 5：运行 Replay 测试并确认通过**

运行：`python -m unittest tests.grpo_ready.test_replay -v`

预期：全部通过，五个输出均可读取。

- [ ] **步骤 6：运行全量新测试**

运行：`python -m unittest discover -s tests/grpo_ready -v`

预期：全部通过。

- [ ] **步骤 7：提交该任务**

```powershell
git add grpo_ready/replay.py tests/grpo_ready/test_replay.py
git commit -m "feat: add historical reward replay outputs"
```

## 任务 4：真实回放与仓库回归

**文件：**

- 生成但不提交：`outputs/grpo_ready_local/*`

- [ ] **步骤 1：运行真实 P0 replay**

运行：

```powershell
python -m grpo_ready.replay --input outputs/qa_mcq.intermediate.jsonl --output-dir outputs/grpo_ready_local
```

预期：退出码 0，报告 121 attempts、27 accepted。

- [ ] **步骤 2：验证输出文件可解析且计数一致**

运行：

```powershell
python -c "import json,pathlib; p=pathlib.Path('outputs/grpo_ready_local'); rows=[json.loads(x) for x in (p/'reward_replay_results.jsonl').open(encoding='utf-8')]; s=json.loads((p/'reward_replay_summary.json').read_text(encoding='utf-8')); assert len(rows)==121; assert s['attempt_count']==121; assert sum(bool(r['accepted']) for r in rows)==27; print('P0_OK')"
```

预期：输出 `P0_OK`。

- [ ] **步骤 3：运行原仓库测试回归**

运行：`python -m unittest tests.test_core -v`

预期：原测试全部通过；若存在与本改动无关的环境型失败，记录完整失败名称与原因。

- [ ] **步骤 4：运行最终新测试和工作区检查**

运行：`python -m unittest discover -s tests/grpo_ready -v`

运行：`git status --short`

预期：新测试全部通过；工作区只包含用户原有未跟踪文件和本次明确生成的回放目录。

## 自检结论

- 设计中的提取、奖励、缺失值、五类输出、manifest 和真实数据验收均有对应任务。
- P1/P2 未混入本计划。
- 类型名称、字段名称与命令入口在各任务间保持一致。
