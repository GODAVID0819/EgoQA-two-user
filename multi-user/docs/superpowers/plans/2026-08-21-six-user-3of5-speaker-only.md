# Six-User 3-of-5 Speaker-Only Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将六用户共识裁剪改为 3-of-5，并把六用户回答性门禁缩减为一次 speaker-only 完整原视频判断，同时保留现有 QA formality 阻塞门禁。

**Architecture:** 只修改现有六用户专用分支，不改变两用户路径。共识函数继续返回事件和逐视频裁剪诊断；回答性条件构造器决定只发起一次 speaker-only 调用，gate 只消费该结果；Torch 脚本同步更新验收字段。

**Tech Stack:** Python 3、pytest、Slurm Bash、Git。

---

### Task 1: 将共识触发条件改为 3-of-5

**Files:**
- Modify: `tests/test_six_user_group_relative_sampling.py`
- Modify: `group_relative_clip_sampling.py`

- [ ] **Step 1: 写 3-of-5 和 2-of-5 失败测试**

将现有 4-of-5 测试改为构造五个 provider 分数，其中三个分数 `>= 0.82`，断言事件只删除 speaker 与这三个 provider 的对应 cluster；再增加两个过阈值时 `events == []` 的测试。保留现有 argmax 和重复 cluster 去重断言。

- [ ] **Step 2: 运行测试并确认旧实现失败**

运行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_six_user_group_relative_sampling.py -q -p no:cacheprovider --basetemp .test_tmp\red_3of5
```

预期：3-of-5 测试失败，因为当前默认值和调用点仍为 4。

- [ ] **Step 3: 最小修改共识实现**

在 `clustered_speaker_consensus_pruning` 中将默认值改为：

```python
min_high_provider_matches: int = 3
```

在六 speaker 遍历调用和返回的 selection 元数据中统一写入 `3`。删除集合继续由 `high_matches` 生成，不改变未过阈值 provider 的 cluster。

- [ ] **Step 4: 运行测试并确认通过**

重复 Step 2 命令，预期该文件全部通过。

### Task 2: 将六用户 answerability 改为 speaker-only

**Files:**
- Modify: `tests/test_six_user_video_qa_loop.py`
- Modify: `video_qa_loop.py`

- [ ] **Step 1: 写最小失败测试**

调整六用户测试，要求 `build_answerability_conditions` 只返回：

```python
[{"condition_id": "speaker_only::speaker", "condition_type": "speaker_only", "users": ["speaker"]}]
```

保留三项 gate 行为测试：speaker 选错通过、选对失败、不可解析失败。调整 runner 测试，断言只调用一次，且只收到 speaker 的一个完整原视频。

- [ ] **Step 2: 运行测试并确认旧双条件实现失败**

运行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_six_user_video_qa_loop.py -q -p no:cacheprovider --basetemp .test_tmp\red_speaker_only
```

预期：条件数量、runner 调用次数和旧 all-six gate 断言失败。

- [ ] **Step 3: 最小修改回答性实现**

六用户条件只返回 speaker-only。六用户 gate 仅查找 speaker-only 行：缺失、不可解析、选择正确均失败；选择其他合法选项通过。返回字段限定为：

```python
{
    "passed": True,
    "failure_label": None,
    "speaker_only_choice": speaker_choice,
    "speaker_only_correct": False,
    "answerability_evaluated_condition_count": 1,
}
```

不生成 all-six 和 cross-view-gain 字段。两用户 gate 不变。

- [ ] **Step 4: 增加 QA formality 阻塞回归测试**

用真实 `merge_parallel_judges` 合并一个 `qa_formality.status=FAIL`、`evidence_groundedness.status=PASS`、`answerability.gate.passed=True` 的六用户结果，断言 `review_passed is False` 且 `qa_formality` 位于 `blocking_failures`。

- [ ] **Step 5: 运行六用户 prompt/loop 测试**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_six_user_prompts.py tests/test_six_user_video_qa_loop.py -q -p no:cacheprovider --basetemp .test_tmp\green_speaker_only
```

预期：全部通过。

### Task 3: 更新 Torch 结果验收合同

**Files:**
- Modify: `tests/test_six_user_torch_job_contract.py`
- Modify: `hpc/qa/smoke/run_six_user_qa_runtime_probe.sbatch`

- [ ] **Step 1: 先修改合同测试**

要求脚本包含 `answerability_call_count=1`、`answerability_evaluated_condition_count=1` 和 `speaker_only_correct`，并拒绝 all-six、cross-view-gain、all-six-wrong 专属统计字段。增加 QA formality prompt 行数量和 accepted row gate 检查。

- [ ] **Step 2: 运行合同测试并确认失败**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_six_user_torch_job_contract.py -q -p no:cacheprovider --basetemp .test_tmp\red_torch_contract
```

预期：旧脚本仍要求两次回答性调用和 all-six 指标，因此失败。

- [ ] **Step 3: 最小修改脚本验收逻辑**

在结果检查中要求每个 QA 只有一个 speaker-only answerability prompt，其视频数为 1；accepted gate 必须满足：

```python
{
    "speaker_only_correct": False,
    "answerability_evaluated_condition_count": 1,
}
```

结果 JSON 记录 `answerability_call_count: 1`，删除 all-six 和 cross-view-gain 专属字段。QA formality prompt 必须存在，且 accepted review 的 `checks.qa_formality.status` 为 `PASS`。

- [ ] **Step 4: 运行合同测试并确认通过**

重复 Step 2 命令，预期全部通过。

### Task 4: 回归验证、受控提交和推送

**Files:**
- Verify: 上述所有修改文件
- Preserve: worktree 中与本功能无关的 dirty 文件

- [ ] **Step 1: 运行定向回归**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_six_user_group_relative_sampling.py tests/test_pruning_ablation.py tests/test_paired_evidence_pruning.py tests/test_six_user_prompts.py tests/test_six_user_video_qa_loop.py tests/test_six_user_torch_job_contract.py -q -p no:cacheprovider --basetemp .test_tmp\six_user_3of5_final
.\.venv\Scripts\python.exe -m py_compile group_relative_clip_sampling.py prompts.py schema.py video_qa_loop.py
git diff --check
```

预期：定向测试全部通过，编译和 diff 检查退出码为 0。

- [ ] **Step 2: 审计暂存范围**

只暂存本计划、相关设计文档和本次功能涉及的代码、测试、六用户脚本；不得暂存 `AGENTS.md`、旧 Runbook、GRPO guardrails 或其他无关 dirty 文件。使用 `git diff --cached --name-status` 逐项核对。

- [ ] **Step 3: 提交实现**

提交信息使用：

```text
feat: update six-user consensus and answerability
```

- [ ] **Step 4: 推送分支**

运行：

```powershell
git push -u origin feature/six-user
```

推送后核验本地分支跟踪 `origin/feature/six-user`。交付只报告分支名、文件和测试结果，不使用提交哈希作为标识。

- [ ] **Step 5: 停在实验审批前**

给出后续 Torch 登录节点验证、窄同步、runtime probe 与 pilot 的计划和预计资源。用户批准前不连接 Torch、不提交新任务；现有任务保持不变且不取消。
