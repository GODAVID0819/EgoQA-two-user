# 三用户 QA 生成链路实施计划

> **执行要求：** 使用 `executing-plans` 按任务顺序实施；每个行为改动先运行新增测试并观察预期失败，再写最小实现使其通过。

**目标：** 在不修改 GRPO、DPO 和 reviewer 训练合同的前提下，把 QA 生成链路扩展为经过严格门禁的 `1 speaker + 2 providers -> 1 QA`，并交付可在 Torch 上运行的三视频 runtime probe、5 条正式小试验和中文 Runbook。

**总体架构：** 保留当前二用户 pair 路径；在 `group_relative_clip_sampling.py` 中新增基于两条合格 speaker-provider 边的星型三元组选择与裁剪，在 `video_qa_loop.py` 中对三用户启用全部真子集阻断门禁。generator 读取三段裁剪视频，视觉 judger 和 answerability evaluator 读取对应完整视频。Torch 作业使用 JobID 派生输出和 scratch，并保存可核验 manifest。

**技术栈：** Python 3、pytest/unittest、CLIP、FFmpeg、Qwen3-VL、Slurm、Bash。

---

## 任务 1：锁定三用户回答性条件和严格门禁

**文件：**

- 新建：`tests/test_three_user_video_qa_loop.py`
- 修改：`video_qa_loop.py:680-815`

### 步骤 1：写失败测试

新增测试覆盖：

- `build_answerability_conditions(["speaker", "p1", "p2"])` 按稳定顺序返回 3 个 single、3 个 proper subset 和 1 个 combined 条件；
- 三用户全集答对、6 个真子集均答错时通过；
- 任意 single 或 proper subset 答对时失败，并写入 `blocking_subset_leaks`；
- `p1` 单独答对在三用户模式下必须阻断；
- 二用户 `provider` 单独答对仍只产生 `evidence_provider_alone_can_answer` warning。

运行：

```powershell
python -m pytest tests/test_three_user_video_qa_loop.py -q
```

预期：三用户门禁字段或严格 provider 逻辑尚未实现，测试失败。

### 步骤 2：最小实现

在 `answerability_gate()` 中按 `len(required_users)` 分支：

- 二用户保留现有 evidence-provider 例外；
- 三用户把全部非 `combined_all_users` 正确选择加入阻断列表；
- 返回稳定字段 `blocking_subset_leaks`，同时保留历史字段以兼容旧消费者；
- 三用户通过原因明确为“全集答对且全部真子集未答对”。

### 步骤 3：验证并提交

```powershell
python -m pytest tests/test_three_user_video_qa_loop.py tests/test_quality_entropy.py -q
git diff --check
git add video_qa_loop.py tests/test_three_user_video_qa_loop.py
git commit -m "feat: 严格校验三用户回答必要性"
```

## 任务 2：补齐三用户 metadata、人工审查和 schema

**文件：**

- 修改：`tests/test_three_user_video_qa_loop.py`
- 修改：`video_qa_loop.py:469-584`
- 修改：`schema.py:119-137`
- 修改：`tests/test_three_user_prompts.py`

### 步骤 1：写失败测试

覆盖：

- `human_audit_packet()` 输出 `speaker_user`、两个 `evidence_provider_users` 和三用户审查说明；
- `complete_generator_metadata()` 保留 `evidence_provider_user=p1`，同时写入 `evidence_provider_users=[p1,p2]`；
- 三用户默认 rationale 和 legacy `why_two_users_needed` 明确要求三位用户缺一不可；
- `per_user_evidence_claims` 覆盖三位用户；
- schema 拒绝缺少任意 required user claim 的三用户 QA；
- 二用户 metadata 保持兼容。

运行并确认失败：

```powershell
python -m pytest tests/test_three_user_video_qa_loop.py tests/test_three_user_prompts.py -q
```

### 步骤 2：最小实现

- 所有 provider 从 `required_users[1:]` 派生；
- 新增 plural metadata，不移除 singular legacy 字段；
- 三用户人工审查说明要求检查两个 provider 的独立贡献和全部真子集；
- schema 只增加三用户必要字段一致性校验，不改变二用户 accepted schema。

### 步骤 3：验证并提交

```powershell
python -m pytest tests/test_three_user_video_qa_loop.py tests/test_three_user_prompts.py -q
git diff --check
git add video_qa_loop.py schema.py tests/test_three_user_video_qa_loop.py tests/test_three_user_prompts.py
git commit -m "feat: 补齐三用户 QA 审查元数据"
```

## 任务 3：实现纯函数星型角色选择

**文件：**

- 新建：`tests/test_three_user_group_relative_sampling.py`
- 修改：`group_relative_clip_sampling.py:1125-1269`
- 修改：`group_relative_clip_sampling.py:1514-1667`

### 步骤 1：写失败测试

使用合成 `pair_scores` 测试新的纯函数：

- 两条从同一 speaker 出发的 kept 边形成合格星型；
- 任意一条 speaker-provider 边 rejected 时，该 speaker 不合格；
- provider-provider 边 rejected 不阻断；
- 三个 speaker 都合格时，固定 seed 得到确定角色顺序；
- 没有合格 speaker 时返回包含三条边状态的明确诊断。

运行并确认缺少函数：

```powershell
python -m pytest tests/test_three_user_group_relative_sampling.py -q
```

### 步骤 2：最小实现

新增无 I/O helper：

- 用无方向键索引三条 pair；
- 枚举三个 hub/speaker；
- 构造 `speaker_index`、`provider_indices`、两条 blocking edges 和一条 diagnostic edge；
- 按固定 RNG 从合格星型中选择；
- 输出 `selected_triplet` 和拒绝诊断。

不要改变 `score_video_pairs()` 的既有 pair 评分语义。

### 步骤 3：验证并提交

```powershell
python -m pytest tests/test_three_user_group_relative_sampling.py tests/test_pruning_ablation.py -q
git diff --check
git add group_relative_clip_sampling.py tests/test_three_user_group_relative_sampling.py
git commit -m "feat: 选择同步三用户星型候选"
```

## 任务 4：实现三视频区间合并与物化

**文件：**

- 修改：`tests/test_three_user_group_relative_sampling.py`
- 修改：`group_relative_clip_sampling.py:1369-1511`

### 步骤 1：写失败测试

mock `materialize_pruned_video()`，验证：

- speaker 的 remove intervals 是两条边对应 speaker 侧区间的归一化并集；
- speaker keep intervals 由完整窗口减去并集得到；
- 两个 providers 各使用自身边对应区间；
- 三个输出 clip 顺序严格为 speaker、provider 1、provider 2；
- 每个 clip 同时包含 generator pruned path 与 judge/full original path；
- 合并后 speaker 保留时长不足时拒绝整个三元组；
- pair 路径原测试保持通过。

### 步骤 2：最小实现

新增：

- 区间归一化/求并集 helper；
- 从 pair 和 clip index 解析 left/right pruning 侧的 helper；
- 三元组 clip 物化函数；
- 三用户 `temporal_pruning` provenance，记录两条来源边与合并区间。

保护规则沿用当前窗口、最短秒数和百分比设置；不在失败时回退完整视频。

### 步骤 3：验证并提交

```powershell
python -m pytest tests/test_three_user_group_relative_sampling.py tests/test_paired_evidence_pruning.py tests/test_pruning_ablation.py -q
git diff --check
git add group_relative_clip_sampling.py tests/test_three_user_group_relative_sampling.py
git commit -m "feat: 物化三用户裁剪视频"
```

## 任务 5：接通 candidate mining、CLI 和 trace

**文件：**

- 修改：`tests/test_three_user_group_relative_sampling.py`
- 修改：`group_relative_clip_sampling.py:1514-1667`
- 修改：`group_relative_clip_sampling.py:1670-1909`
- 修改：`group_relative_clip_sampling.py:1911-2200`

### 步骤 1：写失败测试

覆盖：

- `selected_count=3` 对三个 sampled clips 计算三条 pair edges 并选择星型；
- `build_candidate_packet()` 输出三位 ordered users、`speaker_user`、plural providers、三元组 trace 和新 candidate type；
- `selected_count=2` 仍输出当前 pair candidate；
- `selected_count>3` 在媒体下载或模型加载前明确报错；
- summary 统计三用户 accepted/skipped，跳过原因保留 star diagnostics。

### 步骤 2：最小实现

- 把 `analyze_group_relative_similarity()` 分成共享的 clip/frame/pair 评分阶段和 2/3 用户选择阶段；
- 二用户继续使用现有 sampled pair；
- 三用户使用星型选择和三视频物化；
- trace 增加 `selection.method=random_synchronized_triplet_then_star_pair_filter`、`selected_triplet` 与三条 edge；
- candidate ID 包含三个 agent ID 和稳定角色顺序；
- CLI help 和错误信息改为明确支持 2 或 3。

### 步骤 3：验证并提交

```powershell
python -m pytest tests/test_three_user_group_relative_sampling.py tests/test_pruning_ablation.py tests/test_paired_evidence_pruning.py -q
python -m compileall group_relative_clip_sampling.py video_qa_loop.py schema.py
git diff --check
git add group_relative_clip_sampling.py tests/test_three_user_group_relative_sampling.py
git commit -m "feat: 接通三用户候选挖掘链路"
```

## 任务 6：验证三用户媒体路由和端到端 dry-run

**文件：**

- 修改：`tests/test_three_user_video_qa_loop.py`
- 必要时修改：`video_qa_loop.py:258-345`
- 必要时修改：`video_qa_loop.py:2439-2521`

### 步骤 1：写失败测试

用 fake runner 和三个临时视频路径验证：

- generator media 顺序是三段 pruned video；
- full judge media 顺序是三段 original video；
- answerability 的 7 次调用分别收到 1、1、1、2、2、2、3 段完整视频；
- `condition_media` 保存与条件一致的用户与路径；
- 所有 prompt rows 能追溯 required user 顺序。

### 步骤 2：最小实现

只修复测试揭示的路由或 provenance 缺口。现有 `media_for_clips()` 和 `clips_for_users()` 若已自然支持三用户，不做无必要重构。

### 步骤 3：验证并提交

```powershell
python -m pytest tests/test_three_user_video_qa_loop.py tests/test_three_user_prompts.py tests/test_quality_entropy.py -q
git diff --check
git add video_qa_loop.py tests/test_three_user_video_qa_loop.py
git commit -m "test: 验证三用户 QA 媒体路由"
```

## 任务 7：新增符合当前规则的 Torch 作业

**文件：**

- 新建：`hpc/qa/smoke/run_three_user_qa_runtime_probe.sbatch`
- 新建：`hpc/qa/experiments/run_three_user_qa_pilot_5.sbatch`
- 新建：`tests/test_three_user_torch_job_contract.py`

### 步骤 1：写静态合同测试并确认失败

测试两个脚本均包含：

- `selected-count 3`；
- JobID 派生 `OUTPUT_DIR` 与 `JOB_SCRATCH_ROOT`；
- job-specific `HOME`、HF/Torch/Triton/CUDA/tmp cache；
- 模型加载前调用 `python -m training.torch_storage_preflight`；
- FFmpeg 和 TorchCodec 环境审计；
- `job_manifest.json`；
- 三用户 media/condition 产物检查；
- probe target 为 1，正式 pilot target 为 5；
- 不使用 `latest_*`。

```powershell
python -m pytest tests/test_three_user_torch_job_contract.py -q
```

### 步骤 2：实现两个 `.sbatch`

以当前 QA 作业参数为基础，但不复制其过期的固定输出和 cache 写法：

- account 使用权威规则当前记录的 `torch_pr_674_tandon_advanced`；
- Runbook 提交前必须用 `sacctmgr`、`sinfo`、`scontrol` 动态复核 account、partition、QOS 和 H100 资源；冲突时停止提交；
- 使用 `/scratch/${USER}/..._${SLURM_JOB_ID}` scratch；
- 永久输出路径包含 `${SLURM_JOB_ID}`；
- probe 是唯一 smoke，只处理一个候选；
- pilot 目标为 5 accepted，`accepted_count=0` 视为 Gate 失败；
- 保存环境、Git HEAD、参数、媒体数量、accepted/rejected/attempt 计数。

### 步骤 3：静态验证并提交

```powershell
python -m pytest tests/test_three_user_torch_job_contract.py -q
bash -n hpc/qa/smoke/run_three_user_qa_runtime_probe.sbatch
bash -n hpc/qa/experiments/run_three_user_qa_pilot_5.sbatch
git diff --check
git add hpc/qa/smoke/run_three_user_qa_runtime_probe.sbatch hpc/qa/experiments/run_three_user_qa_pilot_5.sbatch tests/test_three_user_torch_job_contract.py
git commit -m "feat: 添加三用户 QA Torch 作业"
```

若本机没有可用 Bash，只能把该项记录为本地未验证，并在 Torch 登录节点同步后先执行 `bash -n`；不得把 PowerShell 解析结果表述为 Bash 验证。

## 任务 8：编写中文 Torch Runbook

**文件：**

- 新建：`docs/THREE_USER_QA_TORCH_RUNBOOK_CN.md`

### 步骤 1：再次核对真实状态

在写 Runbook 前重新只读核对：

- 当前 HEAD、branch、dirty state；
- 两个 `.sbatch` 的实际参数、输出和资源；
- `training/torch_storage_preflight.py`；
- QA CLI、模型 ID、环境路径和输入 manifest 来源；
- 三份父目录 Torch 权威文档的当前内容；
- `training/grpo_v3/REMOTE_EXECUTION_GUARDRAILS_CN.md`。

### 步骤 2：按模板写 Runbook

Runbook 必须：

- 无 `{{...}}`、`TODO`、`TBD`、`<your_path>` 或虚构值；
- 给出本地窄范围变更检查和交互式 SFTP 同步块；
- 所有下载严格使用 `sftp -> lcd -> cd -> get -> bye`；
- 登录 shell 命令不使用 `exit`、`logout`、`exec`、全局 `set -e` 或 `|| exit 1`；
- 提交前动态打印并核对 account/partition/QOS；
- 所有提交使用 `sbatch --parsable`，自动去除 cluster 后缀、记录 JobID 和时间戳 manifest；
- 先提交唯一 runtime probe，按真实产物 Gate 验收后再提交 pilot；
- 监控和验收从提交 manifest/JobID 派生，不使用 `latest_*`；
- 区分本地测试、登录节点检查、Slurm 状态、GPU runtime、自动 QA Gate 和人工终点评估；
- 明确“远端未验证”，因为本轮不自动上传或提交。

### 步骤 3：模板自检并提交

```powershell
rg -n "\{\{|TODO|TBD|<your_path>|latest_|scp|rsync" docs/THREE_USER_QA_TORCH_RUNBOOK_CN.md
git diff --check
git add -f docs/THREE_USER_QA_TORCH_RUNBOOK_CN.md
git commit -m "docs: 添加三用户 QA Torch 运行手册"
```

`rg` 对禁止项应无输出；文中若必须解释禁用词，应改用中文描述，避免自检误报。

## 任务 9：完整验证与交付审计

**文件：**

- 必要时仅修改前述实现、测试、作业或 Runbook 文件。

### 步骤 1：运行完整验证

```powershell
python -m pytest tests/test_three_user_prompts.py tests/test_three_user_video_qa_loop.py tests/test_three_user_group_relative_sampling.py tests/test_three_user_torch_job_contract.py -q
python -m pytest -q
python -m compileall group_relative_clip_sampling.py video_qa_loop.py schema.py prompts.py
git diff --check
git status --short --branch
```

在可用 Bash 环境中再次运行两个 `.sbatch` 的 `bash -n`。

### 步骤 2：审计范围和历史兼容

```powershell
git diff 6bfb63d..HEAD --stat
git log --oneline 6bfb63d..HEAD
rg -n "selected_count != 2|exactly two|evidence_provider_user|required_users\[1\]" group_relative_clip_sampling.py video_qa_loop.py prompts.py
```

逐项确认遗留二用户措辞是兼容分支，不是三用户路径中的硬编码。

### 步骤 3：交付说明

报告：

- 当前 branch 与 HEAD；
- 修改文件和核心合同；
- 本地测试数量与结果；
- Bash 静态验证结果；
- Runbook 路径；
- 远端仍未上传、未提交、未验证；
- 用户下一步从 Runbook 的 runtime probe 提交块开始。

