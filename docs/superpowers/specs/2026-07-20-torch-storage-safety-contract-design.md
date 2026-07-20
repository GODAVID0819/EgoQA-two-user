# Torch 存储安全契约设计

## 1. 目标与范围

本设计把 Torch 上反复出现的 `/home/${USER}` 配额失败提升为跨实验基础设施契约。以后创建或修改 Torch Runbook、`.sbatch` 或远端执行文档前，必须先阅读 `docs/TORCH_EXPERIMENT_META_RULES_CN.md`。

本轮覆盖：

- 更新跨实验 Meta 规则；
- 为当前 `qa_formality` Smoke/Probe 补齐 scratch-first 环境；
- 增加便宜的登录节点存储预检；
- 增加自动契约测试；
- 更新当前 Runbook 的上传、预检、提交和失败解释。

本轮不迁移全部历史 `.sbatch`，不引入共享 Shell 公共库，不自动删除任何模型、adapter、缓存或实验结果。

## 2. 根因与设计原则

已观察到的写盘失败依次落在：

```text
/home/${USER}/.cache/flashinfer
/home/${USER}/.cache/modelscope
/home/${USER}/.config/vllm
```

这不是三个独立问题，而是一类“第三方库从 HOME 或默认缓存根派生写路径”的基础设施问题。只修单个库变量会让失败迁移到下一个未覆盖目录。

设计原则：

1. 作业内的 `HOME` 和所有已知缓存根都定向到 `/scratch/${USER}`。
2. 默认值必须写在 `.sbatch` 内，不能只依赖提交 shell 的环境继承。
3. 加载 reviewer 或 policy 前必须验证目录存在、可写且不位于 `/home`。
4. 预检失败立即阻止昂贵模型加载，但只终止 Slurm 作业，不退出人工 SSH shell。
5. 存储失败属于基础设施失败，不能写成 reward、trainer 或收敛失败。

## 3. 统一环境契约

每个新 Torch GPU 作业至少定义一个实验或作业级 scratch 根，并从中派生：

```bash
JOB_SCRATCH_ROOT=/scratch/${USER}/egoqa_job_runtime
JOB_HOME=${JOB_SCRATCH_ROOT}/home

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

同时默认设置 `VLLM_NO_USAGE_STATS=1`，避免非实验必需的 usage telemetry 写入用户配置目录。若某个固定依赖版本使用不同变量名，Runbook 必须记录版本、实际变量名和源码/运行证据，不能凭相似命名猜测。

## 4. 存储预检

当前 `qa_formality` 实验新增 `training/torch_storage_preflight.py`。它接收允许根目录和需要检查的环境变量，执行：

1. 环境变量存在且为绝对路径；
2. 解析后的路径位于 `/scratch/${USER}`；
3. 目录可创建；
4. 通过小型临时文件验证可写性并立即删除该临时文件；
5. 输出文件系统剩余空间；
6. 生成结构化 `storage_preflight.json`，记录变量名、解析路径、检查结果和失败原因。

预检不删除既有目录或缓存，不清理旧实验，不修改模型和 adapter。任何检查失败都应在 reviewer 与 trainer 启动前结束作业。

## 5. Meta 规则与 Runbook 作者门槛

`docs/TORCH_EXPERIMENT_META_RULES_CN.md` 新增“存储与缓存是提交前硬门槛”章节，并规定：

- 新建或修改 Torch Runbook 前必须先阅读 Meta 规则；
- 新 Runbook 顶部必须链接 Meta 规则；
- Runbook 必须提供 scratch-first 环境、登录节点预检和失败诊断；
- `.sbatch` 必须自包含安全默认值；
- 不允许只设置 `XDG_CACHE_HOME` 而保留原始 `HOME`；
- 不允许把大型临时文件写入 `/home` 或共享 `/tmp`；
- 空间不足时先生成 `df`、quota 和目录占用证据，再由人工决定清理对象。

Meta 规则不能保证所有未来作者主动阅读，因此自动测试承担可执行约束；文档承担解释、命令和汇报口径。

## 6. 自动测试

新增存储预检单元测试，覆盖：

- 接受全部位于允许 scratch 根内的可写目录；
- 拒绝 `/home`、相对路径和越出允许根的路径；
- 写探针失败时返回结构化失败，不继续模型启动；
- 不删除既有文件。

扩展 `test_grpo_v3_formality_slurm.py`，要求 Smoke/Probe：

- 设置完整的统一环境变量；
- 使用 `FLASHINFER_WORKSPACE_BASE`，禁止错误的 `FLASHINFER_WORKSPACE_DIR`；
- 在 vLLM 和 `swift rlhf` 之前运行存储预检；
- 将 `storage_preflight.json` 写入本次输出目录。

文档内容检查不放进远端训练测试，避免再次要求 Torch 上传 Markdown。

## 7. 当前实验落地

`grpo_v3_formality_smoke.sbatch` 和 `grpo_v3_formality_probe.sbatch` 将：

1. 设置统一 scratch 环境；
2. 创建所有目录；
3. 调用存储预检并保存 JSON；
4. 通过后再执行现有 CPU 数据预检；
5. 随后启动 reviewer 和 trainer；
6. 在 manifest/验收中保留存储预检产物。

Runbook 更新上传清单、登录节点轻量预检、作业提交和失败诊断。Markdown 仍只保存在本地，不成为远端 Python 测试依赖。

## 8. 验收与完成边界

本地完成条件：

```text
存储预检单元测试通过
Formality 全部单元测试通过
Slurm 契约测试通过
git diff --check 通过
```

远端完成条件：

```text
bash -n 通过
storage_preflight.json.status=passed
所有记录路径位于 /scratch/${USER}
reviewer 日志不再出现 /home 写盘失败
Smoke 越过数据加载并完成 1 个 optimizer step
```

本地验证只能表述为“存储安全契约已实现，等待 Torch runtime 验证”。只有新的远端 Smoke 越过原失败边界，才能写“远端存储问题验证通过”。
