# GRPO v3 远程运行约束与工作偏好

本文是 `EgoQA-two-user` 中 GRPO/Torch 任务的强制运行手册。根目录 `AGENTS.md` 要求所有相关任务在行动前阅读本文。

## 1. 证据分层

每次状态汇报都必须把以下层级分开：

| 层级 | 必需证据 | 可以证明 | 不能证明 |
|---|---|---|---|
| 本地静态 | 单元测试、`compileall`、`bash -n`、`git diff --check` | 纯逻辑、语法和静态合同 | CUDA、视频、真实模型和远端环境 |
| 远端基础设施 | Slurm 状态、storage preflight、模型/环境/decoder/reviewer preflight | 作业获得资源并越过对应运行边界 | reward 正确、模型改善 |
| 训练工程 | completion/reward trace、有限梯度、LoRA delta、optimizer step、checkpoint、reload | 被测试的训练链确实更新并可重载 | 代理奖励收敛或真实 QA 改善 |
| 代理奖励 | 预注册趋势、组内方差、覆盖率、固定端点 | 冻结 reward 下是否改善 | groundedness、answerability 或人类质量 |
| 真实 QA | 视频复核、人工盲审、独立 groundedness/answerability Gate | 对应真实质量维度 | 未评估的其他质量维度 |

禁止把上一层的成功写成下一层的成功。

## 2. Git 与远端同步

### 本地提交前

```powershell
git rev-parse --show-toplevel
git status --short --branch
git branch -vv
git worktree list --porcelain
git diff --check
```

- 先确认真实仓库、分支、worktree 和 dirty state。
- 不覆盖、不删除未跟踪或忽略的研究文件。
- `outputs/`、`tmp/`、日志和大型模型产物不进入 Git。
- pull/rebase 前先保护 dirty state；不要假设 `--autostash` 能保护 ignored/untracked 文件。
- 不使用 `git reset --hard`、强制 push 或来源不明的递归删除。

### Torch checkout 同步前

```bash
cd /scratch/${USER}/projects/EgoQA-two-user
git rev-parse --show-toplevel
git status --short --branch
git remote -v
```

若 checkout 非干净状态，先停止并备份/提交/暂存明确范围，不能直接覆盖。同步完成后记录：

```text
branch
commit SHA
worktree
merged/rebased
pushed
remaining local state
cleanup
```

## 3. Job-specific scratch 与存储预检

每个 `.sbatch` 必须在脚本内部设置 job-specific `JOB_SCRATCH_ROOT`，并在模型加载前将以下变量固定到该目录：

```text
HOME
XDG_CACHE_HOME
HF_HOME
HF_DATASETS_CACHE
MODELSCOPE_CACHE
TORCH_HOME
TRITON_CACHE_DIR
TORCHINDUCTOR_CACHE_DIR
VLLM_CACHE_ROOT
CUDA_CACHE_PATH
FLASHINFER_WORKSPACE_BASE
TMPDIR
TMP
TEMP
```

同时设置：

```text
VLLM_NO_USAGE_STATS=1
TOKENIZERS_PARALLELISM=false
```

模型加载前必须运行：

```bash
"${PYTHON}" -m training.torch_storage_preflight \
  --allowed-root "${JOB_SCRATCH_ROOT}" \
  --output "${OUTPUT_DIR}/storage_preflight.json"
```

`storage_preflight.json` 是 smoke/probe 的必需证据。禁止把正式缓存或产物写入 `/tmp`，禁止在作业中自动删除历史缓存。

## 4. 原生视频运行时

原生视频 job 必须在同一个 shell 中同时设置：

```bash
FFMPEG_ENV="${FFMPEG_ENV:-/scratch/${USER}/envs/egoqa-ffmpeg-runtime}"
export PATH="${FFMPEG_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${FFMPEG_ENV}/lib:${LD_LIBRARY_PATH:-}"
```

模型或 trainer 启动前运行：

```bash
"${FFMPEG_ENV}/bin/ffmpeg" -version
"${PYTHON}" - <<'PY'
from torchcodec.decoders import VideoDecoder
print(VideoDecoder.__module__)
PY
```

`torchvision.io.read_video` 回退、`Could not load libtorchcodec` 或 `libavutil.so.*` 缺失，优先检查上述两个运行时路径，不能直接归因于模型或 reward。

## 5. Reviewer 与 scorer

- reviewer/scorer 环境和 trainer 环境分别审计，不能用一套环境成功推断另一套环境可用。
- 至少依次验证：
  1. 服务进程仍存活；
  2. `/v1/models` 成功；
  3. 最小文本请求成功；
  4. 与正式任务相同 schema/批大小/候选数量的请求成功；
  5. 输出通过本地 schema parser。
- HTTP 200 只证明请求成功，不证明：
  - candidate identity 正确；
  - 内部一致性检查执行；
  - reward 可计算；
  - forward/reverse 次序稳定。
- schema-invalid 输出不得按位置猜测 `candidate_id`。允许一次携带原始请求、错误、期望 ID 和输出形状的修复；仍无效则中止。
- 生成式 judge 的正序/逆序差异必须写入 trace。稳定化策略和选中结果必须可审计。

## 6. Slurm 状态和日志

- `PENDING (Priority)` 是排队，不是运行失败。
- `squeue` 找不到已结束 job 不代表失败；使用 `sacct`、`scontrol show job` 和真实输出目录。
- 日志定位顺序：
  1. 真实提交的 `.sbatch` 中 `--output/--error`；
  2. 脚本定义的 reviewer/scorer 日志；
  3. `scontrol show job -dd <JOB_ID>` 的 `WorkDir/StdOut/StdErr/Command`；
  4. 最后才 `tail`。
- 不使用 `latest_*` 推断历史结论。所有路径从真实数字 JobID 和 `run_manifest.json` 派生。
- validator 非零退出可能表示研究 Gate 未通过，不等于程序崩溃。必须读取 validator JSON。

## 7. 扩大规模的 Gate

默认顺序：

```text
静态检查
→ reviewer/scorer-only
→ 1 prompt × 4 completions × 1 optimizer step
→ 小规模多 evidence probe
→ 预注册 probe
→ held-out 固定端点
→ 独立真实质量审计
```

1-step 至少需要：

```text
4 个 completion trace
有限 reward
有意义的组内 reward 方差
global_step >= 1
非零可训练参数变化
adapter 文件完整
adapter/processor reload 成功
```

环境或代码修复后回到越过原失败边界的最小 Gate，不因排队昂贵直接跳到长作业。

## 8. 数据覆盖与泛化

每个 probe 必须报告：

```text
distinct evidence_id 数量
每个 evidence_id 的 group 数
question type 分布
训练/held-out evidence_id 划分
每个切片的 raw-valid/repaired/unrecoverable 比例
每个切片的正常 judge 覆盖率
```

- 单一 evidence pair 上的 reward 上升，只能证明该 pair 上的代理奖励可优化。
- 同一 pair 重复 40/120 次会放大对象、场景和措辞模板，不能作为跨 clip 泛化证据。
- train/held-out 必须按 `evidence_id` 划分，不能把同一视频对的不同 completion 分到两侧。
- 分析模板化时至少报告：
  - 问题前缀/骨架集中度；
  - distinct question 比例；
  - 对象词和关系类型分布；
  - 高分模板占比；
  - held-out evidence 上的固定端点变化。

## 9. 当前 cross-view reward 的责任边界

当前 text-only cross-view judge 可以负责：

- QA 格式与自然度；
- question/option/answer 的文本内部一致性；
- 选项是否互斥、平行且无语义重复；
- speaker-side 信息需求是否与问题有关；
- 是否表达具体的跨视角信息关系；
- 是否退化成浅层活动报告；
- 生成的 evidence/claims/rationale 是否在文本层面互相矛盾。

当前 judge 不看视频时，不负责：

- 视频事实是否真实；
- 实际 groundedness；
- 实际单用户/组合 answerability；
- 真实时间对齐；
- distractor 的视觉真值。

这些维度未来应作为独立、可校准的 judge 组件接入，而不是偷偷混入当前分数解释。

阻断性内部一致性错误、问题—答案类型错位、语义重复选项和浅层活动问题必须具有最终 reward 硬封顶；不能只依赖一个容易被总体印象覆盖的标量分。

## 10. 汇报模板

每次实验汇报至少包含：

```text
目标与冻结条件
branch / commit / job ID / output path
数据覆盖（evidence_id、question type、train/held-out）
本地验证
远端基础设施证据
训练工程证据
代理奖励证据
真实质量证据
能证明
不能证明
第一个失败 Gate
下一步唯一改动
```

已有失败应固化为单元测试、静态断言、preflight 或本文规则，不能只留在聊天记录中。
