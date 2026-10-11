# 六用户二分类 Judge 接入在线 GRPO

本分支保存六用户在线 GRPO 实现：复用 ms-swift、当前六用户 prompt 和冻结的 27B baseline Judge，支持直接批处理启动、完整断点恢复和固定验证比较。

## 2026-10-06：从基座重开的学习率搜索

用户已确认三组学习率 `3e-6 / 1e-5 / 3e-5`，分别从同一27B基座初始化新LoRA，不加载历史checkpoint-60。三组各完成100次更新后比较，选中组继续到总计200次更新。三个独立单H200作业可以并行；每个作业内仍让Policy/Judge分阶段交替运行。

统一使用 `lr_scheduler_type=cosine_with_min_lr`、`warmup_steps=10`、`lr_scheduler_kwargs={"min_lr_rate":0.1}`。`formal_max_steps=200` 定义完整调度长度，`stop_after_steps=100` 定义筛选阶段的正常结束步；两者不能混用。第50/100步作过程验证，保持每2步保存。其余模型、reward、G4、微批1/累积4、温度0.85、LoRA和2048输出设置沿用成功版本。最终固定验证保持相同输入、解码、种子和冻结Judge。

`workflow.training_config` 显式传递搜索参数，`launch.swift_command` 显式生成真实CLI；`stage_stop` 通过框架回调在筛选终点请求验证、完整保存和正常结束，不缩短调度器，也不取消Slurm作业。`stage_state.json` 保存训练开始/结束的实际global step、学习率和调度器位置。阶段验收检查第100步完整状态及200步调度目标；选中组恢复自身的新checkpoint-100，阶段终点改为200，保持同一调度曲线。

用户本轮明确选择沿用已验证直接入口，不新增keeper，保留实际GPU利用率和阶段日志。普通作业由Torch自动路由分区和QOS，申请资源仍须提交前现场核对。历史文件与checkpoint全部保留。

本地91项相关测试：90通过、1项Windows上跳过实际Bash生命周期；新增8项测试覆盖参数到CLI、正常阶段结束、完整断点验收、自动资源路由及拒绝误用另一学习率的检查结果。Torch实际CLI、调度曲线、100到101状态恢复和回调注册已通过；91项远端回归中89项初轮通过，补齐新版本遗漏的原有根部GPU身份辅助源码后另2项通过，两环境新鲜依赖检查通过。新作业3e-6/1e-5/3e-5分别为19262323/19262322/19262321，初次核验均PENDING；新的GPU训练及100步结束仍待实际产物验证，不能把提交称为训练完成或效果改善。真实根、清单和监控见运行记录首节。下文第46/60步进度叙述属于历史时间快照；10月4日最终60步配对结果已由本地真实证据确认，奖励未显示改善。

提交后的额度等待快照曾与可见计数不一致；随后19262321和19262322已同时RUNNING、各分配1张H200，启动存储检查通过，实际运行配置正确且进入Judge初始化。19262323仍排队，原因QOSMaxGRESPerUser；新优化器更新和100步结束尚未验证。没有取消或重提，不能将旧快照描述为三组全面阻塞，详细时序见运行记录首节及本地 `review_artifacts/grpo_lr_search_20261006/排队诊断.md`。

发布分支：`lxr-grpo`。实现位于 `multi-user/training/grpo_v3/six_user_binary`，引用同一 `multi-user` 下的 prompt；模块命令和测试从 `multi-user` 目录运行。

本轮使用 `continuous` 奖励，Policy 和冻结 baseline Judge 均为 `Qwen3.8-27B`。Policy 从基座初始化并训练 LoRA；Judge 不加载 adapter、不更新参数。六用户素材经剪枝后用于生成问题，冻结 Judge 评分，再由框架在组内计算 advantage 更新 Policy。当前实验为训练 18 输入、验证 6 输入，总目标 60 次更新，每 2 步保存、第 20/40/60 步验证。

## 当前执行路径与加速

- `execution_mode="direct"`：`train_direct.sbatch` 在 allocation 内直接执行 `resume`，无需额外登录节点 dispatcher。`submit_after_cpu` 使用 `sbatch --parsable` 并立即记录真实 JobID。
- `generation_profile="compact_full_qa_v1"`：保留全部 QA 字段，压缩辅助说明、使用紧凑 JSON；本轮训练和最终 baseline/Policy 验证均显式设置 2048 token 上限。
- `policy_image_cache_gb=4`：Policy 图像缓存 v2 支持 ms-swift 实际使用的 `bytes/path` 字典，并保留嵌入 bytes 优先级、文件变化失效、像素预算及容量边界。Judge 另外复用逐帧图片和只读视图缓存。
- `shared_gpu=true`：Policy 和 Judge 均使用分配内索引 `[0]`。Policy vLLM 睡眠并卸载 HF 模型与优化器后唤醒 Judge，评分结束且收到 Judge 释放确认后才恢复 Policy；释放失败时停止恢复另一模型。独立验证使用同样的切换顺序。
- `resume_from_checkpoint` 恢复完整 LoRA、优化器、调度器和随机状态；已通过的最小运行由 `skip_smoke=true` 复用。没有保存完整的更新不计入恢复进度。
- `paired_validation=true`、`baseline_after_training=true`：正式终点后以相同 6 个输入、每输入 4 候选、相同解码设置和种子比较未训练基座与最终 Policy。

当前代码已在 Torch 上完成单卡真实更新、验证、保存及适配器重载；本次已核验正式 checkpoint-46。60 步及最终完整配对结果仍未完成，工程通过不能代替质量提升结论。运行历史见 [27B Torch 执行记录](SIX_USER_GRPO_27B_TORCH_RUNBOOK_CN.md)。

## 实施清单

- [x] 数据：复用 `rlhf_evidence_preprocessing.load_asker_view`，保留 speaker 完整采样帧和 provider 剪枝帧；当前只保留训练与验证，以 source packet 隔离不同集合。
- [x] Judge：复用当前题面、证据和可回答性 prompt，四次判断分别使用纯文本、全六用户完整帧、speaker 完整帧和全六用户完整帧；可回答性不含选项或声明答案。
- [x] 评分服务：冻结 baseline Judge，取得 pass/fail 原始概率，保留每项预测、模型与输入身份；基础设施错误中止，非法生成单独计零分。
- [x] GRPO：复用 ORM 插件模式，核对每条 completion 对应的 packet/asker，输出逐候选 reward trace。奖励合成与生成模型仍须由实验配置明确选择，不以 Judge 训练 loss 权重代替实验决策。
- [x] 训练入口：复用在线采样、LoRA、验证、checkpoint 和 JobID 记录方式；资源和数据位置由当前实验配置显式传入，不复制历史作业值。
- [x] 验证：已按测试先行实现媒体路由、提示词泄漏、奖励方向、错误传播与 HTTP 接线检查；独立审查发现的问题均新增针对性回归。本地测试不证明 GPU runtime 或训练效果。

## 边界

本文是代码说明，不是提交命令模板。代码已同步到 Torch，当前执行事实见 [27B Torch 执行记录](SIX_USER_GRPO_27B_TORCH_RUNBOOK_CN.md)。历史测试或环境查询只证明对应时间的边界。

## 2026-09-29 推理加速

- Policy 可用 `use_vllm=true` 切换到 vLLM colocate：训练与生成共享第一张 GPU，生成时卸载训练模型/优化器，更新时让 vLLM sleep；第二张 GPU 专供冻结 Judge。首次同步基座，后续由 ms-swift 同步 LoRA 并清空受权重变化影响的缓存。
- Judge 的 `/score_batch` 一次接收整组候选，按 formality、groundedness、speaker-only、all-six 分别组成批次。四个合法候选原本需要 16 次 `LLM.generate` 调用，现在最多四次；这不是四倍端到端加速的实测结论。
- 每帧的媒体缓存键包含服务命名空间、绝对路径和图像预处理参数，保留帧顺序。当前服务读取已完成、只读的数据包；启动新服务会更换命名空间。缓存解码图片、vLLM 多模态预处理和可复用的视觉前缀，不缓存随题目变化的评分。
- 同步前后均逐项核验候选 `request_id`、`evidence_id` 和服务实例；无效生成保留原槽位。模型异常、数量不符、非有限概率仍报错，奖励公式与输入帧不变。
- 图像、上下文、生成长度、BF16、两份 27B 权重和每组四候选保持既有设置。注意力仍通过 SDPA 调度；已检查的新训练环境编译支持并启用 PyTorch Flash SDPA。Flash Linear Attention 0.5.2 位于项目独立目录，仅加入 Policy 的 Python 路径。
- `acceleration_backend.json` 记录训练进程看到的包版本、vLLM 开关和线性注意力算子。逐候选 reward trace 增加 `score_batch_id/score_batch_seconds`；Judge 子结果记录预处理时间、批次生成时间和缓存 token 数。批次耗时是共享值，统计时必须按批次去重，不能逐候选相加。
- 退出时清理整个本任务子进程组，包括 launcher 已退出但仍存活的 vLLM worker，避免 smoke 结束后残留显存阻塞正式阶段。

配置依据已对照远端实际 ms-swift 4.2.2 与 vLLM 源码。相关官方说明：[共卡生成、休眠、卸载与 LoRA 同步](https://swift.readthedocs.io/en/latest/Instruction/GRPO/GetStarted/GRPO.html)、[多模态缓存标识](https://docs.vllm.ai/en/latest/features/multimodal_inputs/)、[Flash Linear Attention](https://github.com/fla-org/flash-linear-attention)。真实 GPU 吞吐、显存峰值和加速倍数尚未测得，不能由 CPU 测试或配置开关推断。

## 代码入口

所有模块命令均从本仓库 `multi-user` 目录运行，确保 `training` 和 prompt 来自此处。

| 模块 | 责任 |
|---|---|
| `training/grpo_v3/six_user_binary/data.py` | 显式选择 source packet / asker，生成 ms-swift 图片数据，校验完整媒体与集合隔离 |
| `judge.py` | 从当前正式 prompt 构建四次评审，修正生成内容不能控制的用户身份与顺序 |
| `predictor.py` | 使用同版本图像预处理和 vLLM，读取 pass/fail 原始 logprob；支持三任务同一 adapter 或分别指定 adapter |
| `service.py` | 本机 HTTP 评分；每次作业独立实例身份，防止连到残留旧服务 |
| `reward.py`、`plugin.py` | 显式奖励合成、completion 与媒体对应关系、逐候选日志 |
| `launch.py` | 分开运行 policy 和 Judge，缓存隔离、环境记录、模型进程清理、产物检查 |
| `shared_gpu.py` | 单卡训练及独立验证的显存切换，异常时维持安全恢复顺序 |
| `policy_image_cache.py`、`generation.py` | 实际输入格式的有界图片缓存及完整 QA 紧凑输出设置 |
| `resume.py`、`workflow.py` | 工作流配置、已完成输入复用、完整断点选择及直接续训 |
| `evaluation.py` | 未训练 baseline 与最终适配器的完整同输入、同种子配对比较 |
| `zero_gpu.py`、`zero_gpu_runner.py` | 两套环境、数据、处理器、编译器和断点恢复检查 |
| `submit.py` | 以已核验配置提交单个作业，兼容 cluster 后缀并立即记录 JobID |
| `validate_run.py` | 检查完整候选组、有限奖励、梯度、LoRA、实际步数和验证指标 |

默认分卡模式要求 Policy 与 Judge 索引互不重叠；只有显式设置 `shared_gpu=true` 时才允许两者共用单卡索引 `[0]`。训练微批大小与梯度累积可配置，当前为微批 1、累积 4、每组 4 候选。

## 奖励选项：必须显式选择

每项概率均为 pass/fail 二元归一化概率：

\[
p=\sigma(z_{pass}-z_{fail}).
\]

`continuous` 使用：

\[
R=0.2p_{formality}+0.4p_{groundedness}+0.4p_{all\_six}(1-p_{speaker\_only}).
\]

该乘积是两个跨视角要求的软近似，不宣称两个事件已经满足统计独立性，也不把结果解释为校准后的 QA 正确概率。`binary` 先以 0.5 将四项概率转为布尔结果，再用同一函数。`formality`、`groundedness` 分别仅返回对应概率，供用户明确要求的单项训练。代码没有默认 reward mode；实际实验只能使用用户明确选择的配置。

无法解析的 QA、缺问题、选项不合法或声明答案不一致，记零奖励并保留原因。模型调用失败、媒体缺失、候选身份不一致和非有限概率会报错，不能混为低质量 QA。

## 实际运行配置需要什么

训练配置需要实际绝对路径：`project_root`、`train_python`、`judge_python`、`policy_model`、`judge_config`、`train_dataset`、`val_dataset`、`output_root`、`scratch_root`。模型、训练与推理环境可分别选择；不自动安装或升级依赖。

必须明确的实验参数包括 `reward_mode`、`max_steps`、两种 `num_generations`、`max_length`、`max_completion_length`、`max_pixels`、学习率、KL 系数和采样参数。提交还需要当前查询确认的 account、partition、QOS、GRES、CPU、内存、时限及其估时依据。本文不填入未经核验的远端值。

本轮 Judge 配置显式指定 `judge_mode="baseline"` 和真实基座目录 `model_id`，不配置 adapter，并关闭 vLLM LoRA 加载。保留的 `adapter` 模式仅在今后明确选择时使用，需要 `adapters` 中三个任务的实际路径。两种模式不会静默互相回退。`tensor_parallel_size` 必须匹配分配给 Judge 的 GPU 数。

Reward 插件返回原始连续奖励，由 ms-swift 的 `advantage_estimator=grpo`、`scale_rewards=group` 计算组内相对 advantage；不在插件内重复标准化。Judge 始终冻结，仅 policy 的 LoRA 参数参与更新。

当前直接训练入口不启动人为 GPU 空转负载。`hpc/shared/cuda_device_identity.py` 保留实际设备 UUID 映射支持。旧资源保留和外部 attach 记录仅用于历史事故追溯；本分支提供的当前批处理入口是 `train_direct.sbatch`。

## 长输入和验证边界

ms-swift 4.2 的 GRPO 截断策略只支持 `left` 和 `delete`。本实现采用 `delete` 避免截断视频证据；框架可能删除超长或编码失败输入并重采样，`strict=true` 不保证所有选中样本均被训练。必须结合实际输入长度、trainer 日志和 reward trace 中的来源覆盖解释分母；不能将配置行数当作实际完成数。参数依据：[ms-swift 官方参数文档](https://swift.readthedocs.io/en/v4.2/Instruction/Command-line-parameters.html)。

`training_result.json` 只验证工程产物；候选计数包含训练和验证评分，不是终态 QA 数。实际 GPU 更新及 adapter 重载已通过，最终 60 步固定配对改善和人工 QA 质量仍待验证。

## 本地针对性验证

运行位置：Windows PowerShell。本段只执行本地测试，不连接 Torch。

```powershell
Set-Location 'C:\Users\20661\Desktop\Research\AR\multiuser\EgoQA-main-20260928\multi-user'
python -B -m unittest discover -s tests/training/grpo_v3/six_user_binary -v
python -B -m unittest discover -s tests -p test_judge_output_contracts.py -v
```

未安装模型运行依赖的本地测试使用构造数据或推理替身；测试通过不能替代后续真实媒体与 GPU 运行。

2026-09-28 本地验证结果：新增 18/18 测试、现有 Judge 输出合同 7/7 测试通过；20 个相关 Python 文件语法解析通过，提交脚本 Bash 语法检查通过，四个主要命令行入口可加载。独立审查提出的服务实例串用、重复集合路径、GPU 重编号和清理竞态均已修复并回归。

同日已通过现有共享连接只读查询自身 Torch 目录与包元数据：8B 和 27B 基座目录存在；`egoqa-ms-swift-v4.2.2-vllm024` 环境记录 ms-swift 4.2.2 / vLLM 0.24.0，`egoqa-ms-swift-v4.2.2` 不含 vLLM，`qwen38-vllm` 记录 vLLM 0.28.0 且不含 ms-swift。以上只证明目录和包元数据存在，不证明当前模型加载或 GPU 运行成功。9 月 28 日协作者的权重和数据目录权限阻止了提交；9 月 29 日 baseline 模式取消了 adapter 依赖，当前待重新连接并核验可读六用户素材。
