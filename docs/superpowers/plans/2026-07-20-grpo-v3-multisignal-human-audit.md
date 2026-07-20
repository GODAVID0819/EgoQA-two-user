# GRPO v3 多信号人工审计实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将现有 groundedness 专用审计包升级为保留人工结果的多信号审计包，并在本地真实 Gate 3 trace 上完成升级验证。

**Architecture:** 保持 `training/grpo_v3_groundedness_audit.py` 为唯一导出与汇总入口，在案例抽取层补回 reviewer 和 answerability 原始信号，在 CSV 层提供稳定的人工枚举字段和安全合并，在汇总层分别统计各信号并推导人工 answerability gate。现有 groundedness 顶层统计和显式批准语义继续兼容。

**Tech Stack:** Python 3 标准库、`unittest`、JSONL、CSV、Markdown、PowerShell 本地验证。

---

### Task 1: 抽取多信号案例

**Files:**
- Modify: `training/grpo_v3_groundedness_audit.py`
- Modify: `tests/training/test_grpo_v3_groundedness_audit.py`

- [ ] **Step 1: 写失败测试**

扩展测试 trace，使其包含 `qa_formality_status`、`shallow_activity_status`、`combined_correct`、`speaker_only_correct`、`provider_only_correct`、reward components，以及三种 answerability evaluation。断言导出的案例包含语义化机器状态、角色、gate reason 和精简 evaluation。

```python
self.assertEqual(case["reviewer_combined_answerability"], "PASS")
self.assertEqual(case["reviewer_speaker_leakage"], "LEAK")
self.assertEqual(case["reviewer_provider_answerability"], "ANSWERABLE")
self.assertEqual(case["reviewer_qa_formality"], "PASS")
self.assertEqual(case["reviewer_shallow_activity"], "NO_SHALLOW")
self.assertEqual(case["speaker_user"], "u1")
self.assertEqual(len(case["answerability_evaluations"]), 3)
```

- [ ] **Step 2: 验证测试失败**

Run: `python -m unittest tests.training.test_grpo_v3_groundedness_audit -v`

Expected: 新增断言因字段不存在而失败。

- [ ] **Step 3: 最小实现信号映射**

新增只负责类型安全提取和三值映射的辅助函数。布尔值缺失必须输出空字符串，不得默认为失败。

```python
def _boolean_label(value: Any, *, true_label: str, false_label: str) -> str:
    if value is True:
        return true_label
    if value is False:
        return false_label
    return ""
```

在 `_case()` 中抽取机器状态、角色、answerability gate/evaluations 和 reward components。

- [ ] **Step 4: 验证测试通过**

Run: `python -m unittest tests.training.test_grpo_v3_groundedness_audit -v`

Expected: 本任务测试通过。

- [ ] **Step 5: 提交**

```bash
git add training/grpo_v3_groundedness_audit.py tests/training/test_grpo_v3_groundedness_audit.py
git commit -m "feat: export GRPO audit reviewer signals"
```

### Task 2: 增加多信号 CSV 并安全保留人工结果

**Files:**
- Modify: `training/grpo_v3_groundedness_audit.py`
- Modify: `tests/training/test_grpo_v3_groundedness_audit.py`

- [ ] **Step 1: 写失败测试**

断言 `build_review_rows()` 生成全部机器/人工列；为旧 CSV 合并函数增加三个测试：保留已填值与未知列、case id 集合不同时报错、覆盖前产生备份。

```python
merged = merge_existing_reviews(new_rows, old_rows)
self.assertEqual(merged[0]["human_groundedness"], "PASS")
self.assertEqual(merged[0]["human_speaker_leakage"], "")
self.assertEqual(merged[0]["custom_note"], "保留")
```

- [ ] **Step 2: 验证测试失败**

Run: `python -m unittest tests.training.test_grpo_v3_groundedness_audit -v`

Expected: 合并函数未定义或新增列不存在。

- [ ] **Step 3: 实现稳定列和合并保护**

定义固定人工字段列表；机器字段始终来自新 trace，人工字段和未知附加列从旧 CSV 按 `case_id` 保留。若 case id 集合不一致，抛出包含新增/缺失 id 的 `ValueError`。`export_audit()` 在写已有 CSV 前复制时间戳备份。

```python
HUMAN_FIELDS = (
    "human_groundedness",
    "human_combined_answerability",
    "human_speaker_leakage",
    "human_provider_answerability",
    "human_qa_formality",
    "human_shallow_activity",
)
```

- [ ] **Step 4: 验证测试通过**

Run: `python -m unittest tests.training.test_grpo_v3_groundedness_audit -v`

Expected: 合并、错配保护和备份测试通过。

- [ ] **Step 5: 提交**

```bash
git add training/grpo_v3_groundedness_audit.py tests/training/test_grpo_v3_groundedness_audit.py
git commit -m "feat: preserve multisignal audit reviews"
```

### Task 3: 生成多信号指南

**Files:**
- Modify: `training/grpo_v3_groundedness_audit.py`
- Modify: `tests/training/test_grpo_v3_groundedness_audit.py`

- [ ] **Step 1: 写失败测试**

对 `_markdown()` 或公开包装函数断言指南包含角色、三种 answerability、各条件选择/证据、formality、shallow activity、format/reward 和人工字段说明。

```python
self.assertIn("Speaker leakage", guide)
self.assertIn("combined answerability", guide)
self.assertIn("human_speaker_leakage", guide)
self.assertIn("先独立判断，再阅读 reviewer", guide)
```

- [ ] **Step 2: 验证测试失败**

Run: `python -m unittest tests.training.test_grpo_v3_groundedness_audit -v`

Expected: 新指南文本断言失败。

- [ ] **Step 3: 实现分层 Markdown**

按设计中的固定顺序输出每条案例；answerability evaluation 只展示审计所需字段，未知值明确显示“未记录”。在顶部加入枚举值表和防止 reviewer 锚定的操作顺序。

- [ ] **Step 4: 验证测试通过并提交**

Run: `python -m unittest tests.training.test_grpo_v3_groundedness_audit -v`

Expected: 指南测试通过。

```bash
git add training/grpo_v3_groundedness_audit.py tests/training/test_grpo_v3_groundedness_audit.py
git commit -m "feat: render multisignal audit guide"
```

### Task 4: 汇总多信号人工结论

**Files:**
- Modify: `training/grpo_v3_groundedness_audit.py`
- Modify: `tests/training/test_grpo_v3_groundedness_audit.py`

- [ ] **Step 1: 写失败测试**

构造至少 20 条满足旧 groundedness 覆盖条件的记录，并部分填写新增信号。断言：旧顶层字段不变；每个信号分别统计 completed/counts/agreement；人工 answerability gate 按 combined PASS 且 speaker NO_LEAK 推导；不合法值进入诊断并从完成数排除。

```python
self.assertEqual(summary["schema_version"], "grpo_v3_multisignal_audit_v2")
self.assertEqual(summary["signals"]["speaker_leakage"]["counts"]["LEAK"], 1)
self.assertEqual(summary["human_answerability_gate"]["passed"], 1)
self.assertIn("agreement_rate", summary["signals"]["groundedness"])
```

- [ ] **Step 2: 验证测试失败**

Run: `python -m unittest tests.training.test_grpo_v3_groundedness_audit -v`

Expected: `signals` 和 gate 汇总不存在。

- [ ] **Step 3: 实现独立统计和 gate 推导**

为每个信号定义人工/机器字段和合法枚举映射；只对合法已填值计算 completed 与一致率。保留原 `completed_count`、reviewer PASS/FAIL 覆盖、agreement 和 uncertain 统计。

- [ ] **Step 4: 验证测试通过并提交**

Run: `python -m unittest tests.training.test_grpo_v3_groundedness_audit -v`

Expected: 所有审计单元测试通过。

```bash
git add training/grpo_v3_groundedness_audit.py tests/training/test_grpo_v3_groundedness_audit.py
git commit -m "feat: summarize multisignal human audit"
```

### Task 5: 更新运行手册并升级真实审计包

**Files:**
- Modify: `docs/GRPO/v3/GRPO_V3_GATE3_V2_AUDIT_GREEDY_TORCH_RUNBOOK_CN.md`
- Regenerate: `outputs/grpo_v3/groundedness_audit_pack/groundedness_audit_cases.jsonl`
- Regenerate: `outputs/grpo_v3/groundedness_audit_pack/groundedness_audit_review.csv`
- Regenerate: `outputs/grpo_v3/groundedness_audit_pack/groundedness_audit_guide_cn.md`
- Regenerate: `outputs/grpo_v3/groundedness_audit_pack/groundedness_audit_export.json`

- [ ] **Step 1: 更新手册**

将第 5、6 节改为多信号审计说明，列出全部人工枚举、推导 gate、保留现有结果和“不改变当前 Gate”的边界。

- [ ] **Step 2: 运行真实导出**

Run:

```powershell
python -m training.grpo_v3_groundedness_audit export `
  --trace outputs/grpo_v3/gate3_14194844/gate3_14194844/reward_trace.jsonl `
  --output-dir outputs/grpo_v3/groundedness_audit_pack `
  --pass-count 12 --fail-count 12
```

Expected: `cases=24`，现有 CSV 若存在则产生备份，case id 集合一致。

- [ ] **Step 3: 校验媒体与案例集合**

Run: `Get-ChildItem outputs/grpo_v3/groundedness_audit_pack/clips -Filter *.mp4 | Measure-Object`

Expected: `Count = 48`，媒体未被导出命令修改。

- [ ] **Step 4: 生成未批准诊断汇总（仅当已有至少 20 条 groundedness 人工结果）**

Run: `python -m training.grpo_v3_groundedness_audit summarize --review-csv outputs/grpo_v3/groundedness_audit_pack/groundedness_audit_review.csv --output outputs/grpo_v3/groundedness_audit_pack/multisignal_audit_summary_unapproved.json`

Expected: 若人工字段尚未填够，命令明确报告不足，不伪造汇总；若已填够，输出 `approved_for_weight_change=false`。

- [ ] **Step 5: 提交代码、测试和手册**

审计包位于现有未跟踪输出目录，不纳入代码提交；只提交明确的源代码、测试和手册。

```bash
git add training/grpo_v3_groundedness_audit.py tests/training/test_grpo_v3_groundedness_audit.py docs/GRPO/v3/GRPO_V3_GATE3_V2_AUDIT_GREEDY_TORCH_RUNBOOK_CN.md
git commit -m "docs: document multisignal GRPO audit"
```

### Task 6: 最终验证

**Files:**
- Verify: `training/grpo_v3_groundedness_audit.py`
- Verify: `tests/training/test_grpo_v3_groundedness_audit.py`
- Verify: `docs/GRPO/v3/GRPO_V3_GATE3_V2_AUDIT_GREEDY_TORCH_RUNBOOK_CN.md`

- [ ] **Step 1: 运行相关单元测试**

Run: `python -m unittest tests.training.test_grpo_v3_groundedness_audit tests.training.test_grpo_v3_repo_reward tests.training.test_grpo_v3_slurm -v`

Expected: 全部通过。

- [ ] **Step 2: 编译检查**

Run: `python -m compileall -q training/grpo_v3_groundedness_audit.py tests/training/test_grpo_v3_groundedness_audit.py`

Expected: exit code 0。

- [ ] **Step 3: 检查差异和生成物一致性**

Run: `git diff --check`

Expected: 本任务文件无 whitespace error。核对 24 条案例、CSV 24 行、指南 24 个案例标题和 48 个视频。

- [ ] **Step 4: 报告证据边界**

明确区分：本地单元测试/静态检查通过、真实本地 trace 审计包升级成功、没有重新调用远程 reviewer、没有运行新的训练或改变 Gate 结果。
