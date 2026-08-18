# 六视频多用户 QA 生成链路实施计划

> **执行要求：** 使用 `executing-plans` 顺序实施。每个行为改动必须先写测试并观察预期失败，再写最小实现；旧的三用户实施计划已被取代，不得执行。

**目标：** 实现 `1 speaker + 2 anchor providers + 3 additional providers -> 1 QA`，使 generator 接收三段裁剪视频和三段完整视频，视觉审核使用六段完整视频，answerability 固定比较 speaker-only 与 all-six，并交付 Torch runtime probe、正式 5 条小试验和中文 Runbook。

**架构：** 保留当前二用户 pair 采样路径；六用户路径对一个同步六人集合计算 15 条 pair edges，选择拥有至少两个合格邻居的 speaker 和两个 anchors。speaker 合并两条 anchor 边的裁剪区间，anchors 分别裁剪，additional providers 保持完整。六用户 answerability 只生成两个条件，不改变 GRPO、DPO 或 reviewer 训练合同。

**技术栈：** Python、pytest/unittest、CLIP、FFmpeg、Qwen3-VL、Slurm、Bash。

---

## 任务 0：建立可重复的本地测试环境

**文件：**

- 可能新建但不提交：`.venv/`
- 只读核对：`.gitignore`

### 步骤 1：取得用户授权

当前系统只有 Python 3.14，且没有 pytest。未获授权前不得创建环境或安装依赖。

### 步骤 2：创建局部环境

获批后在当前 worktree 创建 `.venv`，确认 `.venv` 被 Git 忽略，只在该环境安装运行现有测试所需的最小依赖。先安装 pytest；如果收集阶段报告缺少直接运行依赖，再按实际错误逐项补充，不预装训练栈。

### 步骤 3：运行基线

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

若基线失败，停止实现并区分环境、收集和现有测试失败；未经用户允许不在失败基线上继续。

## 任务 1：把六用户 answerability 固定为两次调用

**文件：**

- 新建：`tests/test_six_user_video_qa_loop.py`
- 修改：`video_qa_loop.py:680-815`
- 修改：`video_qa_loop.py:2439-2521`

### 步骤 1：写失败测试

覆盖：

- 六用户只生成 `speaker_only` 和 `combined_all_six_users`；
- speaker-only 正确时拒绝；
- speaker-only 错误且 all-six 正确时通过；
- speaker-only 无法解析时拒绝；
- all-six 错误或无法解析时拒绝；
- 通过结果包含 `speaker_only_correct=false`、`all_six_correct=true`、`cross_view_gain=1` 和评估次数 2；
- 二用户 `build_answerability_conditions()` 与现有 gate 行为保持不变。

运行并确认失败：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_six_user_video_qa_loop.py -q
```

### 步骤 2：最小实现

- `build_answerability_conditions()` 对六用户返回两个条件；
- 六用户 gate 不使用 provider-alone 例外或 proper-subset 逻辑；
- 无法解析的 choice 作为阻断错误保存；
- 指标直接从两次结构化选择派生；
- 二用户分支不重构。

### 步骤 3：验证并提交

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_six_user_video_qa_loop.py tests/test_quality_entropy.py -q
git diff --check
git add video_qa_loop.py tests/test_six_user_video_qa_loop.py
git commit -m "feat: 固定六用户回答性双条件门禁"
```

## 任务 2：把三用户 prompt 改为六用户角色合同

**文件：**

- 删除：`tests/test_three_user_prompts.py`
- 新建：`tests/test_six_user_prompts.py`
- 修改：`prompts.py:931-1735`

### 步骤 1：写失败测试

六用户测试覆盖：

- prompt 把 `required_users[0]` 解释为 speaker；
- `required_users[1:3]` 是两个 anchor providers；
- `required_users[3:6]` 是 additional providers；
- speaker-only 必须不足；
- 六视频实际输入必须支持唯一正确答案；
- 不包含“所有 providers 必须贡献”“只有所有用户一起才能回答”或三用户专用措辞；
- groundedness prompt 不因未使用某个 provider 而失败；
- answerability prompt 分别准确描述 speaker-only 与 all-six；
- sampled image/video order 文本支持六位用户；
- 二用户 prompt 保留当前语义。

运行并确认当前三用户 prompt 与新合同冲突：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_six_user_prompts.py -q
```

### 步骤 2：最小实现

- 把 `len(required_users)==3` 专用分支替换为六用户角色分支；
- role contract 使用 plural anchors 和 additional providers；
- generator schema 不要求六位用户都具有 evidence claim；
- 保留 legacy 字段名，但改写内容为“speaker 需要至少一个外部视角”；
- 二用户分支不改变。

### 步骤 3：验证并提交

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_six_user_prompts.py -q
git diff --check
git add prompts.py tests/test_three_user_prompts.py tests/test_six_user_prompts.py
git commit -m "feat: 将生成提示扩展为六视频角色"
```

## 任务 3：补齐六用户 packet、metadata 和 schema

**文件：**

- 修改：`tests/test_six_user_video_qa_loop.py`
- 修改：`video_qa_loop.py:469-584`
- 修改：`schema.py:119-137`

### 步骤 1：写失败测试

覆盖：

- packet/audit 输出 `input_users`、speaker、两个 anchors、三个 additionals；
- `evidence_provider_user` 兼容指向第一个 anchor；
- `evidence_provider_users` 只包含两个 anchors；
- `media_roles` 覆盖六位用户；
- `supporting_user_claims` 至少引用一个非 speaker 输入用户；
- 不要求 claim 覆盖全部六位用户；
- 引用 packet 外用户时 schema 失败；
- 二用户 schema 与 metadata 保持通过。

### 步骤 2：最小实现

- 六用户角色从 packet 显式字段读取，并校验与有序用户列表一致；
- audit instruction 明确只要求 speaker 不可单独回答和 all-six 可回答；
- legacy `why_two_users_needed` 只解释外部视角需求；
- schema 只在六用户模式增加新字段一致性约束。

### 步骤 3：验证并提交

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_six_user_video_qa_loop.py tests/test_six_user_prompts.py -q
git diff --check
git add video_qa_loop.py schema.py tests/test_six_user_video_qa_loop.py
git commit -m "feat: 补齐六用户 QA 元数据"
```

## 任务 4：实现六人星型 anchor 选择纯函数

**文件：**

- 新建：`tests/test_six_user_group_relative_sampling.py`
- 修改：`group_relative_clip_sampling.py:1125-1269`
- 修改：`group_relative_clip_sampling.py:1514-1667`

### 步骤 1：写失败测试

使用合成的 15 条 pair scores 测试：

- speaker 至少有两个 kept neighbors 时形成候选；
- 恰好两条 anchor edges kept、其他 edges rejected 时仍接受；
- 只有一条 kept neighbor 时该 speaker 不合格；
- provider-provider edges 不阻断；
- 多个 speaker 和多组 anchor 合格时固定 seed 得到确定结果；
- 输出包含 speaker index、两个 ordered anchor indices、三个 ordered additional indices、selected anchor edges 和全部 diagnostic edges；
- 无合格结构时诊断包含各用户 kept degree。

运行并确认缺少六用户 helper：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_six_user_group_relative_sampling.py -q
```

### 步骤 2：最小实现

新增纯函数：

- 无方向 pair key 索引；
- kept 邻接表；
- 合格 speaker/anchor 组合枚举；
- seeded selection；
- 稳定 additional-provider 顺序；
- compact rejection diagnostics。

不要改变 `score_video_pairs()` 的评分、阈值和 pair trace 语义。

### 步骤 3：验证并提交

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_six_user_group_relative_sampling.py tests/test_pruning_ablation.py -q
git diff --check
git add group_relative_clip_sampling.py tests/test_six_user_group_relative_sampling.py
git commit -m "feat: 选择六用户双锚点角色结构"
```

## 任务 5：实现三段裁剪加三段完整的媒体物化

**文件：**

- 修改：`tests/test_six_user_group_relative_sampling.py`
- 修改：`group_relative_clip_sampling.py:1369-1511`

### 步骤 1：写失败测试

mock 文件复制和 FFmpeg 物化，验证：

- speaker remove intervals 为两条 anchor 边对应 speaker 侧区间的规范化并集；
- speaker keep intervals 从完整窗口减去 remove union；
- 两个 anchors 分别使用正确 left/right pair 侧区间；
- 三个 additional providers 不调用裁剪函数，generator path 指向完整同步视频；
- 六段输出顺序和角色固定；
- 三段 pruned 与三段 full 的媒体角色准确；
- speaker 合并后保留不足时当前结构失败；
- 原有 pair 物化测试保持通过。

### 步骤 2：最小实现

新增：

- interval normalize/union/complement helper；
- 按 clip index 解析 pair side 的 helper；
- speaker 双边合并裁剪；
- anchor 单边裁剪；
- additional full-context provenance；
- 每个 clip 的 generator/full judge 路径和 media role。

不得把 additional full video 标成 pruned，不得在 speaker 保护失败时静默回退。

### 步骤 3：验证并提交

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_six_user_group_relative_sampling.py tests/test_paired_evidence_pruning.py tests/test_pruning_ablation.py -q
git diff --check
git add group_relative_clip_sampling.py tests/test_six_user_group_relative_sampling.py
git commit -m "feat: 物化六用户混合媒体输入"
```

## 任务 6：接通六用户 candidate mining、CLI 和 trace

**文件：**

- 修改：`tests/test_six_user_group_relative_sampling.py`
- 修改：`group_relative_clip_sampling.py:1514-2200`

### 步骤 1：写失败测试

覆盖：

- `selected_count=6` 随机抽取六人并计算 15 条 edges；
- 物化失败后尝试下一个合格角色结构；
- candidate packet 有六位有序用户、两个 anchors、三个 additionals 和 media roles；
- candidate type、ID 和 selection method 明确为 six-user/two-anchor；
- summary 保存 15 条 edges、角色尝试和 skip diagnostics；
- `selected_count=2` 输出保持兼容；
- `selected_count=3` 或其他值在 encoder 初始化和媒体下载前明确报错。

### 步骤 2：最小实现

- 在共享 frame embedding/pair scoring 后按 2/6 分流；
- 二用户沿用现有 selected pair；
- 六用户枚举 seeded role structures，直到物化成功；
- `build_candidate_packet()` 按模式生成兼容字段与六用户新字段；
- CLI help 只声明支持 2 和 6。

### 步骤 3：验证并提交

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_six_user_group_relative_sampling.py tests/test_pruning_ablation.py tests/test_paired_evidence_pruning.py -q
.\.venv\Scripts\python.exe -m compileall group_relative_clip_sampling.py
git diff --check
git add group_relative_clip_sampling.py tests/test_six_user_group_relative_sampling.py
git commit -m "feat: 接通六用户候选挖掘链路"
```

## 任务 7：验证六视频媒体路由与噪声指标

**文件：**

- 修改：`tests/test_six_user_video_qa_loop.py`
- 必要时修改：`video_qa_loop.py:258-345`
- 必要时修改：`video_qa_loop.py:2439-2521`
- 必要时修改：`schema.py`

### 步骤 1：写失败测试

用 fake runner 和六个临时媒体验证：

- generator 顺序为 speaker pruned、两个 anchor pruned、三个 additional full；
- groundedness judge 顺序为六段 full；
- answerability 第一次调用一段 speaker full，第二次调用六段 full；
- runner 总 answerability 调用数严格为 2；
- prompt rows 和 condition media 保存用户、角色、路径和裁剪状态；
- `cross_view_gain`、两项 correct 指标和阶段耗时进入 review trace；
- all-six wrong 使用中性失败标签，不自动归因为 noise。

### 步骤 2：最小实现

只修复测试揭示的 routing/provenance 缺口；若现有 `media_for_clips()` 已自然支持混合路径，不做无必要重构。

### 步骤 3：验证并提交

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_six_user_video_qa_loop.py tests/test_six_user_prompts.py tests/test_quality_entropy.py -q
git diff --check
git add video_qa_loop.py schema.py tests/test_six_user_video_qa_loop.py
git commit -m "feat: 记录六视频回答性与媒体指标"
```

## 任务 8：新增六视频 Torch 作业

**文件：**

- 新建：`hpc/qa/smoke/run_six_user_qa_runtime_probe.sbatch`
- 新建：`hpc/qa/experiments/run_six_user_qa_pilot_5.sbatch`
- 新建：`tests/test_six_user_torch_job_contract.py`

### 步骤 1：写失败的静态合同测试

两个作业必须包含：

- `selected-count 6` 和最小 group size 6；
- probe target 1，pilot target 5；
- JobID 派生 output 和 scratch；
- job-specific HOME、HF/Torch/Triton/CUDA/tmp；
- 模型加载前调用 `training.torch_storage_preflight`；
- FFmpeg 与 TorchCodec 审计；
- `job_manifest.json`；
- 六段 generator、六段 judge 与两条 answerability trace 检查；
- accepted/rejected/attempt/all-six-wrong 计数；
- 不使用模糊当前产物路径。

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_six_user_torch_job_contract.py -q
```

### 步骤 2：实现作业

- 资源参数从当前 QA H100 作业和三份 Torch 权威文档派生；
- account 暂按权威文档当前记录写入，但 Runbook 必须要求提交前用远端查询复核；
- probe 是唯一 smoke，正式 pilot 不承担第一次六视频 runtime 验证；
- 输出路径包含 `${SLURM_JOB_ID}`；
- `accepted_count=0` 时自动 Gate 失败，1-4 条按实际数量报告；
- 不自动扩样或提交下一作业。

### 步骤 3：验证并提交

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_six_user_torch_job_contract.py -q
bash -n hpc/qa/smoke/run_six_user_qa_runtime_probe.sbatch
bash -n hpc/qa/experiments/run_six_user_qa_pilot_5.sbatch
git diff --check
git add hpc/qa/smoke/run_six_user_qa_runtime_probe.sbatch hpc/qa/experiments/run_six_user_qa_pilot_5.sbatch tests/test_six_user_torch_job_contract.py
git commit -m "feat: 添加六视频 QA Torch 作业"
```

本机没有可用 Bash 时，只能记录为本地未验证，并在 Torch 登录节点先执行 `bash -n`。

## 任务 9：编写中文 Torch Runbook

**文件：**

- 新建：`docs/SIX_USER_QA_TORCH_RUNBOOK_CN.md`

### 步骤 1：重新完整读取权威依据

写作前完整读取并按当前版本核对：

- 父目录 `docs/Torch通用复现项目执行手册.md`；
- 父目录 `docs/TORCH_EXPERIMENT_META_RULES_CN.md`；
- 父目录 `docs/TORCH_RUNBOOK_TEMPLATE_CN.md`；
- 仓库 `AGENTS.md`；
- `training/grpo_v3/REMOTE_EXECUTION_GUARDRAILS_CN.md`；
- 两个新 `.sbatch`、storage preflight、实际 CLI、模型和输出合同。

### 步骤 2：按模板写 Runbook

必须满足：

- 无模板占位符或虚构值；
- 本地窄范围检查和交互式 SFTP 同步；
- 下载严格使用交互式 SFTP 的 `lcd`、`cd`、逐文件 `get` 和 `bye` 顺序；
- 登录 shell 不关闭 SSH 会话，不使用全局严格模式；
- 提交前打印并核对 account、partition、QOS 和 H100 资源；
- 所有提交使用 `sbatch --parsable`，兼容 cluster 后缀，自动保存 JobID manifest；
- 先验收唯一 runtime probe 的真实产物，再提交 pilot；
- 监控、验收和下载均从 JobID manifest 派生；
- 明确本地、登录节点、Slurm、GPU runtime、自动 QA Gate 和人工终点评估边界；
- 明确远端未验证，因为本轮不自动上传或提交。

### 步骤 3：自检并提交

```powershell
rg -n "\{\{|TODO|TBD|<your_path>|latest_|scp|rsync" docs/SIX_USER_QA_TORCH_RUNBOOK_CN.md
git diff --check
git add -f docs/SIX_USER_QA_TORCH_RUNBOOK_CN.md
git commit -m "docs: 添加六视频 QA Torch 运行手册"
```

禁止项检查应无输出。

## 任务 10：完整验证与交付审计

### 步骤 1：运行验证

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_six_user_prompts.py tests/test_six_user_video_qa_loop.py tests/test_six_user_group_relative_sampling.py tests/test_six_user_torch_job_contract.py -q
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall group_relative_clip_sampling.py video_qa_loop.py schema.py prompts.py
git diff --check
git status --short --branch
```

在可用 Bash 环境再次运行两个 `.sbatch` 的 `bash -n`。

### 步骤 2：范围审计

```powershell
git diff 6bfb63d..HEAD --stat
git log --oneline 6bfb63d..HEAD
rg -n "len\(required_users\) == 3|selected_count != 2|exactly two|required_users\[2\]" prompts.py video_qa_loop.py group_relative_clip_sampling.py
```

确认三用户正式逻辑已移除、二用户残留措辞只存在于兼容分支、GRPO/DPO/reviewer 训练文件未被修改。

### 步骤 3：交付说明

报告：

- 分支与 HEAD；
- 核心输入、角色、媒体和 answerability 合同；
- 本地测试数量和结果；
- Bash 静态验证结果或未验证边界；
- Runbook 路径；
- 未 push、未上传、未提交 Slurm、远端未验证；
- 用户下一步从 Runbook 的六视频 runtime probe 开始。
