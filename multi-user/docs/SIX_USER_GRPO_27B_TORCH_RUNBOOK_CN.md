# 六用户 27B baseline Judge GRPO：本次执行记录

遵循工作区 [Torch 实验元规则](../../../docs/TORCH_EXPERIMENT_META_RULES_CN.md)。以下首节为当前执行记录，后续旧 JobID、配置及命令仅用于历史追溯。

## 2026-10-04：lxr-grpo 分支与当前直接续训

当前源码发布分支为 `lxr-grpo`。推荐批处理入口是 `multi-user/hpc/grpo_v3/six_user_binary/train_direct.sbatch`，由 `submit_after_cpu` 验证指定零 GPU 结果后提交并记录真实 JobID，分配资源后在作业内直接续训。下面的 hold/attach、旧 8B、1024 token、三集合及 keeper 叙述属于历史记录，不是当前运行方式。

实际续训 Job **19136858** 的提交清单为 `/scratch/xl6775/projects/EgoQA-six-user-baseline-grpo-20260929/variants/shared-gpu-20261003T031000Z/submissions/grpo60_20261004T034110Z_1587a7c9/submission.json`；10 月 4 日 13:20 的新鲜检查确认 RUNNING，正式完整 checkpoint-46 已保存。训练输出为该版本目录下 `outputs/formal_shared40/train_19136858`，工作流为 `outputs/workflow_19136858_shared40`。这是当次观察，后续状态以该 JobID 的新鲜 Slurm 和产物查询为准。

配置为 Policy/Judge 同用27B基座、单H200分阶段共享、训练18输入/验证6输入、G4、紧凑完整QA、2048token预算、总60步、每2步保存及20/40/60验证，最后运行同设置6输入24候选baseline/Policy比较。模型、媒体、环境和训练产物位于原有 scratch；本分支发布源码、测试和说明，不包含这些大型文件。此前单卡最小更新/验证/adapter重载已完整通过，因此本次从40直接恢复，复用既有证据。最终60步及配对改善尚未完成。

## 2026-10-03 19:33：单卡续训19102046已提交

登录节点完整检查均通过，包括83项回归测试、两套实际环境的睡眠/唤醒接口和六项零GPU检查，以及完整checkpoint36的恢复检查。证据位于单卡版本目录的 `preflight/shared_gpu_checks`。认证恢复后只提交一次新作业 **19102046**，当前新鲜调度状态为 **PENDING**，不能当作训练已启动。

版本目录为 `/scratch/xl6775/projects/EgoQA-six-user-baseline-grpo-20260929/variants/shared-gpu-20261003T031000Z`，提交记录为该目录下 `submissions/grpo60_20261003T113036Z_ea8b7773/submission.json`，同目录保存标准输出/错误日志。Slurm已确认1张H200、16CPU、500G、16小时、账户pr674、实际QOS为gpu48、自动路由h200_tandon/h200_public、无固定节点；批处理Command直接执行 `train_direct.sbatch`。

工作流输出为版本目录下 `outputs/workflow_19102046_sharedgpu`；唯一新显存边界最小运行输出 `outputs/smoke_sharedgpu/train_19102046`，正式输出 `outputs/formal_sharedgpu/train_19102046`。先检查真实显存切换、最小更新/验证和adapter重载，通过后同一作业自动从36恢复到60。正式首更新应为37、首完整新checkpoint应为38，最小检查的1步不计入正式进度。最终仍需6输入24候选新baseline/Policy完整配对结果。助手持续监控；所有旧作业和产物保留，不自动取消任何作业。

## 2026-10-03 11:09：保存第36步，验证单GPU分阶段运行

最新已完成的正式进度为 **36/60**。`19077013` 于北京时间10:32:28被系统取消，运行2小时8分58秒；日志到37但37未保存，不能计入续训进度。完整断点位于 `/scratch/xl6775/projects/EgoQA-six-user-baseline-grpo-20260929/variants/policycache-v2-20261002T224256Z/outputs/formal_cachev2/train_19077013/swift/v0-20261002-203348/checkpoint-36`。前驱源码、日志及checkpoint全部保留，未执行取消命令。Slurm未给出明确取消原因；最后一小时两卡平均利用率49.47%和12.39%，与利用率政策风险相符，不能把残留的配额等待原因当作取消原因。

真实Policy图像缓存v2已生效。CPU作业19075375完成真实字典格式的1188帧×4候选编码对照，117.282秒降为31.514秒，五个编码字段一致；72项远端测试及六项零GPU检查通过。同输入的正式第21步观察到744.036秒降为627.051秒，即耗时减少15.7%，但节点与生成长度不同，不能视为严格因果加速基准。19077013后续步时约286–475秒，已评分72候选均合法；第20步过程验证奖励0.3392763只是代理指标，最终新设置baseline尚未完成。

为减少顺序执行时闲置的一张GPU，新代码显式启用 `shared_gpu`，在同一H200分阶段运行Policy与冻结Judge。Policy vLLM先睡眠，HF模型与优化器卸载CPU，再唤醒Judge；评分完成后必须收到Judge睡眠释放确认，才恢复Policy。任何释放失败都停止后续模型恢复。单独终评也使用相同切换逻辑。模型、reward、数据、生成设置和同步GRPO更新次序保持原有定义，新增阶段日志 `gpu_phase_trace.jsonl`。

当前是**验证中，未提交新GPU作业**。本地83项测试中82项通过、1项Windows平台条件跳过，包含真实HTTP实例隔离、部分唤醒失败清理和模型/优化器切换顺序。目标源码目录为 `/scratch/xl6775/projects/EgoQA-six-user-baseline-grpo-20260929/variants/shared-gpu-20261003T031000Z`；两环境接口、数据、处理器与完整checkpoint36在登录节点零GPU验证。新显存边界须在同一allocation内完成唯一一次最小更新、验证、adapter重载；通过后直接从正式36恢复至60，最小检查的更新不计入正式训练。终点仍为6输入24候选的同设置baseline/Policy配对比较。

拟使用1张H200、16 CPU、500G、16小时，自动选择可用H200 partition/QOS，不指定节点、不使用空转负载。提交入口仍为 `train_direct.sbatch`，通过 `--parsable` 将真实JobID立即写入时间戳目录的 `submission.json`；没有当前新JobID前不提供猜测监控路径。充分检查只能减少已知错误，实际单卡峰值显存与切换耗时仍需GPU运行确认。

## 2026-10-03：系统取消后的第 18 步恢复与 Policy 图像缓存

最新恢复作业 **19068662** 已在全部验证通过后提交。清单为 `submissions/grpo60_20261002T214811Z_f2187e18/submission.json`，首次核验 `PENDING/Priority`。请求 2 H200、16 CPU、500G、16 小时；仅指定已核验 account 和 H200 类型，由站点自动路由到 `h200_tandon,h200_public`，实际 QOS 为 `gpu48`。没有固定节点。实际 `Command` 仍是 `train_direct.sbatch`，从第 18 步继续，不需要后续手工接入。输出为 `outputs/formal_cache18/train_19068662` 和 `outputs/workflow_19068662_cache18`；当前等待实际第 19 步与完整 checkpoint-20。

`19003763` 最终为 `CANCELLED by 0`，运行 `02:28:38`，北京时间 04:02:22 结束；助手没有执行取消。日志完成第 19 步，最后完整保存的是 `outputs/formal_compacta/train_19003763/swift/v0-20261002-134526/checkpoint-18`，恢复只能从第 18 步开始。40 个已评分候选均为有效 JSON，已完成各步的截断率均为 0；没有第 20 步验证或最终比较。调度器没有提供具体取消说明；已有低 GPU 利用率记录与集群政策相符，但不能据此断言唯一原因。

原恢复配置 `preflight/grpo_resume18_20261002T201136Z/workflow.json` 的六项登录节点检查均通过，包括第 18 步优化器、调度器和适配器检查。新配置继续总计 60 步、每 2 步保存、每 20 步验证、18 个训练输入与 6 个验证输入、2048 生成上限。根据剩余 42 步及验证开销，恢复任务申请 16 小时。

纯 CPU 作业 `19064929` 在 `cs628` 以 `COMPLETED/0:0` 结束，运行 4 分 23 秒，峰值内存约 11.45 GiB，资源请求不含 GPU。完整 1188 帧、四候选的 Policy 编码对照为：原实现 141.814 秒，缓存缩放后图像 38.155 秒，减少约 103.66 秒。`input_ids`、`labels`、`pixel_values`、`image_grid_thw`、`mm_token_type_ids` 完全一致。证据在 `preflight/perf_19003763_20261003/cpu_19064929/policy_cpu_full_benchmark.json`。这是完整输入的 CPU 阶段测量，不是已测得的 GPU 端到端加速率。此前登录节点全量进程没有返回结果，其失败证据保留，不称为通过。

实际模板的 `load_images=True`，没有走临时 PNG 中转；优化针对重复 JPEG 解码和缩放。新缓存按路径、文件修改时间、大小、像素预算及模板类型区分，使用 4 GiB 有界缓存；返回图像副本，避免候选互相修改。对象坐标输入和只保留路径的模板仍走原处理。模型、媒体像素、reward、组大小、训练步数及数据划分均不改变。

本地 70 项测试中 69 项通过、1 项因平台条件跳过；远端 70 项全部通过。改动已窄范围同步，旧文件备份在 `incoming/pre_policy_cache_20261002T210321Z`；远端是 SFTP 源码快照，没有 Git 元数据。`preflight/grpo_policycache18_20261002T210321Z` 中的六项登录节点检查及实际实现的 CPU 等价性对照已全部通过：实际缓存实现的 128 帧四候选编码为 33.872 秒→9.040 秒，五字段相同；正式四候选组件确认缓存命中 6 次、解码 2 次；checkpoint-18 的 64 份优化器状态有限、调度器步数 18。上述结果通过后才提交恢复任务。完整 CPU 对照节省时间约占旧最长真实步时的 10.21%，这是保守估算依据；实际整体比例仍须在新 GPU 运行中测量，不能把 CPU 阶段提速直接当作整体实测。

## 当前作业：19003763（2026-10-02 提交）

本轮完整登录节点验证已通过后正式提交 **19003763**。提交记录为 `submissions/grpo60_20261002T033116Z_9d05a8ba/submission.json`，已核实 Slurm 的 `Command` 是 `multi-user/hpc/grpo_v3/six_user_binary/train_direct.sbatch`。初次查询为 `PENDING`，不表示训练已经启动。资源为 2 张 H200、16 CPU、500G、24 小时；使用当前可用 account `torch_pr_674_tandon_advanced`、partition `h200_tandon`、QOS `gpu48`，未固定节点。

批处理直接调用续训入口，环境检查失败或训练进程退出时作业随之退出，不依赖额外 dispatcher。Judge 和 Policy 使用本机空闲端口，避免固定端口冲突。当前正式输出为 `outputs/formal_compacta/train_19003763`，工作流状态为 `outputs/workflow_19003763_compacta/workflow_status.json`；标准日志为提交目录下 `slurm-19003763.out/.err`。首个新更新应为第 11 步，首个新完整 checkpoint 应为第 12 步，之后再核验第 60 步和 `validation_comparison.json`。

通过证据：远端完整 60 项测试及启动器追加 9 项测试；训练/Judge 两套环境导入、依赖检查及实际 C/C++ 共享库编译加载；全部 7200 张图片像素解码；4 候选处理器组件；全部输入含 2048 生成预算的保守长度上界不超过 58,592（配置上限 65,536）；checkpoint-10 的 64 份优化器状态、调度器步数 10、有限且非零 LoRA 更新、Python/NumPy/CPU 随机状态和保存训练参数加载；原失败 CUDA 内核源码再次在零 GPU 条件下编译通过。完整产物位于 `preflight/grpo_compactA_20261002`，这些检查不能替代本轮实际 GPU 续训和效果评估。

## 2026-10-02：新目标，直接续训与方案 A

用户已要求继续并创建新的训练 goal。本轮采用方案 A：保持全部 QA 字段，追加紧凑 JSON 和辅助说明长度要求，生成上限从 1024 改为 2048。新输入在 `data/grpo_trainval_compactA_20261002`，仍是训练 18、验证 6，媒体及 source packet 划分未改变。历史数据、旧 baseline 和完整 `checkpoint-10` 均保留；从第 11 步起记录新生成设置，最终不能把不同设置的旧 baseline 当作对照。

13 小时无训练的根因已经定位：`hold.sbatch` 只保留 allocation，训练依赖后续登录节点 dispatcher，而那次上传/认证中断后 dispatcher 从未启动。新脚本为 `multi-user/hpc/grpo_v3/six_user_binary/train_direct.sbatch`，批任务内直接执行 `resume`，传入 `RUN_CONFIG`，训练结束或失败均返回真实退出状态，不再依赖另一个接入进程。新的训练路径不启动人为 GPU 空转负载；历史 keeper 文件只作为事故证据保留。

此前已有的加速包括 Policy vLLM colocate、LoRA 同步/休眠与卸载、冻结 Judge 的 vLLM 批量评分、媒体 UUID/前缀缓存和 FLA 算子。本次新增逐帧有界缓存及只读视图缓存，减少提问者顺序变化带来的重复解码和 NFS 元数据访问。1800 帧重排的登录节点 CPU 实测为原实现 112.46 秒、新实现 0.032 秒；冷加载约 115 秒没有改善，逐张图像像素和顺序完全一致。此结果只证明被测图片准备阶段，不是整个训练的加速倍数。

本轮配置为 `preflight/grpo_compactA_20261002/workflow_compactA.json`，从 `outputs/formal_resumable/train_18901392/swift/v0-20260930-202348/checkpoint-10` 恢复，总目标 60 步；每 2 步保存 checkpoint，第 20/40/60 步作过程验证。最终训练完成后用未训练基座生成新的 baseline，再评估最终 Policy，两者均用新 prompt、2048 上限、同一 6 输入/24 候选及冻结 Judge，不要求指标必须提高才算运行完成。

新的直接执行链路及生成设置已通过本地测试；远端 `preflight/grpo_compactA_20261002/controller_status.json` 与 `checks/status.json` 均为 passed，随后才提交当前作业。登录节点的处理器组件检查不冒充完整 GPU 显存或算子验证，完整媒体链路已有历史真实 10 步运行依据。Slurm `--test-only` 返回的数字不是已提交 JobID，当前真实 JobID 仅为首节记录的 19003763。

## 2026-10-02 08:31 北京时间：状态核验与曲线

新鲜 `sacct` 确认 `18901392` 为系统 `CANCELLED by 0`，正式训练仅完成 10/60 步，完整 `checkpoint-10` 保留。其后 `18917012` 为 `TIMEOUT`，运行 13:00:16；当前 `squeue` 无用户活动作业。后继任务的 `submission.json` 仍为 `submitted_pending_zero_gpu_dispatch`，没有训练 `attachment.json`、训练 dispatcher 或新正式输出目录，只有早期 keeper sidecar。该 sidecar 的步骤退出码 124 对应设定的两小时交接定时退出，不能当作训练报错。13 小时资源保留没有产生新增训练更新，之前接入流程未完成。

本次从登录节点读取实际 `logging.jsonl` 数值并生成本地派生曲线：`review_artifacts/grpo_training_18901392_snapshot_20261002/training_curves.png`，同目录有 PDF、SVG 和 `training_metrics.csv`。10 步训练奖励均值 0.197006402，最后一步 0.27055272；40 个训练候选中 19 个到达 1024 token 上限，19 个为 `invalid_completion`，第 8/9 步的奖励与组内标准差均为零。每步平均 11.56 分钟。以上是单次中断训练的批次指标，不是固定验证提升。

未训练 baseline 仍为原 6 个输入、24 候选，平均奖励 0.0833480591；没有更新后 Policy 的固定验证结果，不能与训练奖励直接相减宣称改善。当前应先修复资源申请与训练接入的衔接，再处理高截断率/无效 JSON，并从完整正式 checkpoint 恢复；本次用户请求仅完成状态检查与曲线交付，没有新增 GPU 提交。

## 2026-10-01：系统取消后的断点续训

资源作业 **18883354** 在运行 `02:28:56` 后被 Slurm 标记为 `CANCELLED by 0`，正式步骤 `18883354.2` 随之为 `CANCELLED/0:15`。系统未在 `Comment`、`AdminComment`、`SystemComment` 留下具体取消说明；计算节点 `gh114` 后续仍为可用的混合占用态，24 小时 walltime 未到。[NYU Torch 官方作业政策](https://services.rt.nyu.edu/docs/hpc/submitting_jobs/slurm_submitting_jobs/)说明 gh 节点低 GPU 利用率会被自动取消；本次 keeper 日志末尾两小时平均值约 32%，低于官方的 60% 取消阈值，两者吻合，但无法证明这是唯一原因。助手没有执行取消命令。当前正式训练在 `0/60` 中断，没有正式更新或正式 checkpoint；旧最小训练 `1/1` 及其 checkpoint 是工程验证，不计入正式 60 步。

旧正式输出中的 `validation_baseline.json` 已确认 `status=completed`，保留原 6 个验证输入、24 个候选，连续奖励均值 `0.0833480591`。新作业 **18901392** 已用当前核验的 `torch_pr_674_tandon_advanced`、`h200_tandon`、`gpu48` 合同提交，当前状态 `PENDING`；提交清单在 `submissions/grpo60_resumable_20260930T230912Z_7836c92b/submission.json`，同目录有自动接入进程和实际配置，不能把提交当成训练通过。旧作业及全部产物均保留。

续训代码只跑正式阶段，复用已完成的最小训练证据和固定 baseline；正式目标仍为 **60 步**，第 20/40/60 步执行原定验证，每 2 步保存一个可恢复 checkpoint，最多保留最近 3 个。若后续 allocation 再被系统取消，新作业只从完整保存的 checkpoint 恢复训练状态；选择器检查步数与 `trainer_state.json` 一致，以及 LoRA 权重、优化器、调度器、随机状态和训练参数文件齐全且非空，不能把未保存的步数计入已完成更新。最终配对比较要求两次验证的模型、Judge 合同、输入绑定、解码设置和种子相同；跨作业的 Judge 进程实例 ID 可不同，但必须同为冻结基座。登录节点已核对旧 baseline 的 6/24 完整性、数据绑定、训练配置与旧最小训练通过记录；本地完整 48 项测试通过，Torch 登录节点部署后的完整 45 项测试通过，断点选择新增针对性 8 项测试通过。新的 GPU 正式续训仍待实际验证。

## 2026-10-01：首次反向传播失败与 CUDA 工具链修复

同一资源作业 **18883354** 的 `portfix1` 步骤 `18883354.1` 以 `FAILED/1:0` 结束。它已完成 Policy/Judge 初始化和首组四候选评分，奖励为 `0.2652、0、0.4970、0.5393`，组内有非零方差；但第一次反向传播的 TileLang GDN 内核编译失败，因此**没有任何参数更新或 checkpoint**。第一个实际错误是 CUDA 编译器与头文件版本不匹配：训练环境选择了 CUDA 13.2 的 `nvcc`，运行时头文件为 CUDA 13.0。失败日志在 `outputs/smoke_portfix1/train_18883354/trainer.log`，旧产物原样保留。

已在项目的 `preflight/nvcc130_18883354/packages/nvidia/cu13` 安装独立 CUDA 13.0 编译器及匹配头文件，没有改动原训练环境。将上次日志里的原始失败内核源码提取出来，使用该编译器在登录节点编译，退出码为 0，产出非空 CUBIN；记录见 `preflight/nvcc130_18883354/compile_status.json`。这属于零 GPU 验证，只证明该内核可以编译，不能替代实际反向传播和优化器更新。

启动器已将该工具链的 `CUDA_HOME`、`CUDA_PATH` 仅传给 Policy 子进程，Judge 环境保持原样。远端启动器和工作流的 11 项针对性测试通过后，在仍为 `RUNNING` 的 **18883354** allocation 内启动新的独立尝试 `nvcc130`；没有取消或重提 GPU 资源作业：

- 接入记录：`submissions/grpo60_samealloc_20260930T214944Z_669606f9`；实际步骤号和状态以该目录 `attachment.json`、Slurm 查询为准。
- 工作流：`outputs/workflow_18883354_nvcc130/workflow_status.json`。
- 最小训练：`outputs/smoke_nvcc130/train_18883354`。
- 60 步训练及固定验证：`outputs/formal_nvcc130/train_18883354`。

后续核验已确认：同一尝试的最小训练步骤完成 `global_step 1/1`，梯度范数约 `0.0983`，奖励均值约 `0.3254`、组内标准差约 `0.2481`；训练后验证和 `checkpoint-1` 均已完成，`outputs/smoke_nvcc130/train_18883354/run_manifest.json` 已写入。工作流随后切入 `formal` 阶段。未训练 baseline 的 `validation_baseline.json` 已标记 `completed`：原 6 个验证输入共产生 24 个候选，连续奖励均值 `0.0833480591`，非零奖励 5/24。正式 60 步训练器已启动，但尚未验证其首步更新。

仍须核验 60 步终点、最终 checkpoint，以及同一 6 个验证输入的 baseline/Policy 配对指标和实际分母；当前最小步骤通过并不构成正式训练完成或质量提升的证据。

## 2026-10-01：同一 allocation 的端口修复重试

JobID **18883354** 已在 `gh114` 获得2张H200。最初训练步骤 `18883354.0` 运行约8分29秒，以 `FAILED/1:0` 结束：Judge 已加载27B基座并返回 ready，Policy 的 `torchrun` 在固定端口29500创建TCPStore时遇到 `EADDRINUSE`。无Policy参数更新；当时资源作业仍为RUNNING。源码核对确认当前ms-swift会将环境变量 `MASTER_PORT` 传给 `torchrun`。

已补上计算节点启动Policy前的本机空闲端口选择，记录到 `policy_rendezvous.json`；本地和Torch登录节点的启动器/工作流针对性测试共11项通过。新尝试 `portfix1` 使用同一JobID、不申请新GPU作业：

- 接入记录：`submissions/grpo60_samealloc_20260930T205412Z_9aeb388f`；该步骤 `18883354.1` 的最终状态和失败原因见上节。
- 历史工作流：`outputs/workflow_18883354_portfix1/workflow_status.json`。
- 历史最小训练：`outputs/smoke_portfix1/train_18883354`。
- 历史正式训练目标：`outputs/formal_portfix1/train_18883354`。

旧 `.0`、`.1` 步骤、原始输出、失败日志和代码备份均保留。资源作业不得因步骤失败被自动取消。

## 2026-10-01：零 GPU 验证后再提交 60 步版本

本节替代下文旧作业配置。用户明确要求先完成零 GPU 验证，再提交 GPU；旧作业 `18837514` 已按用户指定取消，查询为 `CANCELLED by 4914731`、运行时间 0，未占用 GPU。五项零 GPU 检查全部通过后，已提交新的 60 步作业 **18883354**，最新核验为 PENDING，自动接入进程存活。

新配置为 `configs/workflow_trainval60_20261001.json`。本轮仅使用训练集和验证集：旧保留测试窗口并入训练集，训练为三个来源窗口、18 个提问者输入；验证保持原 DAY4_21400000 窗口的 6 个输入。活动输入目录为 `data/grpo_trainval_20261001`，没有 test.jsonl，旧实验输入与产物继续保留。

正式训练为 **60 次更新**，第 20、40、60 步验证并保存，保留三个 checkpoint。正式训练前执行独立 baseline 验证，完成后加载第 60 步 LoRA 做独立 Policy 验证：均为同一 6 个输入、每输入 4 候选、同一模板/图像预算/解码参数/逐输入固定种子、同一冻结 Judge 实例。输出 `validation_baseline.json`、`validation_policy.json` 和 `validation_comparison.json`。完整配对统计不会丢弃无效生成，负提升也正常报告；中间训练内验证用于过程趋势，不冒充独立终点评估。

零 GPU 检查入口为 `zero_gpu_runner.py`，检查两套 Python 的真实导入、pip check、ninja/FFmpeg、C/C++ 编译与共享库加载、Python 开发头文件、正式 CLI 参数类型、全部真实图片的像素解码、训练侧四候选编码和 Judge 图片组件编码。无 GPU 的节点不能证明 CUDA kernel、显存峰值或完整 1800 帧服务调度，保留原有唯一最小 smoke 覆盖这些边界。

两套环境中旧 Decord wheel 标记为 cp36，导致 Python 3.11/3.12 的 pip check 失败。已在 `runtime/cpu_tools_20261001/decord_backup_train` 与 `decord_backup_judge` 备份；重装官方 py3 wheel 后检查仍失败，因此移除了当前图片流程不使用的 Decord 可选包。已检查其依赖方仅将其列为额外可选依赖，没有移除必需依赖。随后训练和 Judge 两套环境的实际包导入、pip check、ninja/FFmpeg、C/C++ 编译执行、Python 头文件与共享库加载全部通过；训练 CLI 类型解析确认 60 步、每 20 步验证、use_vllm=true。

独立 GCC/G++ 12.4 已安装在 `/scratch/xl6775/envs/egoqa-host-toolchain-20261001`，新配置通过 `compiler_environment` 将已核实的 CC/CXX 路径传入预检和正式子进程。原始失败检查保存在 `preflight/trainval60_20261001`，当前检查目录是 `preflight/trainval60_20261001_retry1`；本地完整 39 项测试通过。

已启动 `submit_after_cpu.py`（启动时 PID `2272373`）：只在同一目录的五项检查全部通过、配置未改变时提交一次新 GPU 作业；任何失败或超时都不提交。该目录 `submission.claim` 指向唯一提交记录目录，实际 JobID 由其中的 `submission.json` 读取。当前最后核验到的阶段是 `train_data`，尚未确认新 JobID。条件提交的两项本地回归测试通过；它不以指标提升作为提交条件，也不取消任何作业。

### 01:44 更新：剩余编码移至纯 CPU 作业

`trainval60_20261001_retry1` 中两套环境、18/6 行数据和全部 **7200 张图片的真实像素解码均通过**；训练四候选完整编码进程随后被 `SIGKILL` 终止。系统证据不足以确定原因，不能直接声称 GPU OOM。旧条件提交程序因此退出，未生成 GPU JobID。

现已提交纯 CPU 验证作业 **18883207**：`cpu_short`、系统选择 `cpu48`、4 CPU、64G、20 分钟，ReqTRES 不含 GPU。它复用同一配置下已通过的两套环境和数据检查，只补齐训练与 Judge 编码。时限针对剩余两项处理器检查，先前训练编码运行约四分钟后被系统终止；不重复已完成的包导入和7200图解码。

CPU 提交记录：`submissions/cpu_preflight_20260930T174411Z_f12586d3/submission.json`；当前检查输出：`preflight/trainval60_cpu_18883207`。新的条件提交程序启动 PID 为 `684512`，会核对CPU作业是否失败或超时；仅五项检查全部通过后才提交 GPU。实际GPU任务目录从当前检查目录的 `submission.claim` 读取，不能使用旧失败目录。

### 01:49 更新：五项零 GPU 检查全部通过

CPU 作业 `18883207` 为 `COMPLETED/0:0`，用时 00:04:24，MaxRSS 为 13952784K（约13.3 GiB）。对应 `status.json` 明确为 passed：两套环境与数据检查复用此前通过结果，训练与 Judge 编码均以 exit 0 完成。

真实训练批次的 `input_ids` 为 `[4, 39921]`，每候选1249张图片，`image_grid_thw` 为 `[4996, 3]`；输入长度加1024生成预算仍低于65536。Judge四请求图片组件编码通过。以上证明CPU依赖、编译、输入文件与被检查的编码形状；不宣称完整1800帧Judge服务、CUDA显存或参数更新已经通过。

GPU任务实际提交目录为 `submissions/grpo60_20260930T174412Z_bc49f330`，其中 `submission.json` 已记录真实 JobID **18883354**、CPU验证来源 `18883207` 和启动配置。后台接入 PID `740868` 已在提交完成后独立确认存活，`attachment.json` 为 `waiting_for_allocation`。资源为2 × H200、16 CPU、500G、24小时；未固定节点。输出将写入 `outputs/workflow_18883354`、`outputs/smoke/train_18883354` 和 `outputs/formal/train_18883354`。

## 2026-09-30：Judge 启动故障与恢复

作业 `18790400` 已运行，不再是下文历史记录中的排队状态。它在 `gh123` 获得两张 H200，资源持续 02:13:25 后显示 `CANCELLED by 0`；实际训练步骤 `18790400.0` 运行 00:07:02，以 `FAILED/1:0` 结束。系统取消的具体原因未知，助手未执行取消。

本次已经证明：旧数据 12/6/6 行可复用、存储检查通过、共享 keeper 正常、27B Judge 的 18 个权重分片加载完成（约 51.1 GiB），vLLM 选用 FlashAttention 3 并完成了部分图编译。失败发生在服务就绪之前：FlashInfer 采样器预热需要执行 `ninja`，子进程报 `FileNotFoundError: ninja`。Policy 尚未启动，无 reward trace、参数更新或 checkpoint。

远端检查确认两个 Python 环境的 `bin/ninja` 均已安装。根因是启动器只使用解释器绝对路径，未将各自环境的 `bin` 加入 PATH。本次修复：为 Judge 与 Policy 分别构造 PATH，保留 FFmpeg 的 PATH/LD_LIBRARY_PATH；设置 `VLLM_USE_FLASHINFER_SAMPLER=0`，并将 GDN 预填充选为两套实际 vLLM 均支持的 Triton 后端，避免依赖当前未就绪的 CUDA C++ 即时编译链。vLLM、FlashAttention、LoRA 同步、批量评分和前缀缓存保持启用。每次启动保存 `runtime_tools.json` 记录各角色的工具路径。

本地与 Torch 登录节点的启动器、加速和预测器测试均为 14 项通过；新增测试先复现缺失行为，再通过修复。远端已分别成功执行训练环境和 Judge 环境的 `ninja --version`，并验证 FFmpeg 可见。完整 GPU 初始化和训练更新仍需新作业验证，不能把本地或登录节点检查写成 smoke 通过。

新恢复作业 **18837514** 已提交，保持 2 × H200、16 CPU、500G、24 小时以及两个 Qwen3.8-27B。新提交和自动接入记录：
`/scratch/xl6775/projects/EgoQA-six-user-baseline-grpo-20260929/submissions/grpo_runtime_recovery_20260930T004328Z_d97311d2`

该目录的 `submission.json` 保存新 JobID、后台接入 PID `3931142`、旧失败 JobID `18719659/18790400`、数据来源 `18719659` 和源码备份路径。复用已有数据，排到资源后自动接入；输出分别为 `outputs/workflow_18837514`、`outputs/smoke/train_18837514`、`outputs/formal/train_18837514`。没有取消旧任务，没有覆盖其日志或实验产物。

## 2026-09-29 晚间检查更新

### 当前加速配置：仍使用作业 18790400

用户要求推理加速后，已更新本作业的自动接入程序；没有取消或重新申请 Slurm allocation。当前接入目录为：
`/scratch/xl6775/projects/EgoQA-six-user-baseline-grpo-20260929/submissions/grpo_accel_20260929T114357Z_62ae166e`

该目录 `submission.json` 保存资源提交来源、配置和接入程序 PID `1338252`；旧接入目录的 `dispatcher_replaced.json` 指向本次更新。提交命令结束后已独立确认接入程序存活，状态 `waiting_for_allocation`；实际下一次训练命令包含 `use_vllm=true`、LoRA 同步、休眠和卸载设置。最后一次队列核验为 `PENDING (Priority)`。模型、奖励、帧输入、数据划分、采样参数和训练步数保持原设置。

| 加速项 | 当前配置 |
|---|---|
| Policy 生成 | vLLM colocate，TP=1，最多同时调度 4 个序列，分块预填充和前缀缓存 |
| 显存管理 | vLLM 占比 0.55；生成时卸载训练模型/优化器，更新时 sleep level 1 |
| 权重同步 | 首次同步基座，后续 LoRA 同步；ms-swift 在更新权重后清理相关缓存 |
| 训练环境 | `/scratch/xl6775/envs/egoqa-ms-swift-v4.2.2-vllm024/bin/python`，torch 2.11.0+cu130 / vLLM 0.24.0 / Transformers 5.8.1 / ms-swift 4.2.2 |
| 注意力 | SDPA；已验证 PyTorch 编译支持、开关启用 Flash SDPA，实际 GPU kernel 待运行日志确认 |
| 混合架构线性注意力 | flash-linear-attention / fla-core 0.5.2，独立安装在 `runtime/acceleration_20260929/packages`，只传给 Policy 进程；GPU 实际调用待验证 |
| Judge | 继续使用独立 vLLM 0.28.0 环境；最多 4 个序列、多模态处理缓存 4 GiB |
| Judge 批处理 | 整组候选一次 HTTP 请求，四项任务分别批处理；逐项核验候选身份与完整概率 |
| 媒体复用 | 解码图片缓存、包含图像处理参数与服务命名空间的媒体 UUID、vLLM 前缀缓存 |

Judge 当前配置文件：`configs/judge_baseline_27b_accel_18790400.json`。源码更新前备份保存在本次 `submission.json` 的 `code_backup` 字段所指目录。原环境未被 pip 升级；独立扩展包通过 `--no-deps` 安装。

本地完整测试 **32/32** 通过；远端加速/评分/启动器/预测器/工作流针对性测试 **23/23** 通过。真实 GPU 加速效果仍未验证，不能宣称固定加速倍数。四个合法候选的 16 项评分从 16 次逐条 `LLM.generate` 调用改为最多 4 次批调用，不意味着端到端耗时必然缩短为四分之一。

运行后查看当前 JobID 的 `acceleration_backend.json`、`trainer.log` 和 `reward_trace.jsonl`：后端记录包版本与算子可用性；训练日志记录生成/更新耗时；评分 trace 记录预处理、批生成耗时、缓存 token 数。共享的批耗时须按批次去重统计。资源未到位时不进行额外 GPU 测量或重复提交；原计划的唯一一次最小 smoke 直接使用加速设置。

以下为本次恢复前后的历史依据；第 5 节查看命令已更新为加速接入目录。

`18719659` 已获得过 2 张 H200；资源作业最终为 `CANCELLED by 0`，运行 02:27:46，`.batch` 为 `CANCELLED/0:9`。自动接入步骤 `18719659.0` 运行 00:08:19 后以 `FAILED/1:0` 结束。

四个窗口的采样、CLIP 编码、聚类和剪枝已完成，生成训练/验证/保留测试输入 12/6/6 行。失败发生在 smoke 的 `allocation_keeper()` 存活检查：`os.kill(pid, 0)` 抛出 `ProcessLookupError`。Keeper 日志进一步确认，GPU 身份辅助函数拒绝了 PyTorch 返回的 UUID 格式，导致两个控制线程退出，继而阻断训练入口。27B policy/Judge 尚未加载，没有 policy 更新或训练效果。系统随后取消资源作业的具体原因未确定；不得据此声称 OOM、模型不兼容或质量失败。

已修复 UUID 标准化，支持无前缀字符串、UUID 对象与原始 16 字节表示；仍不根据物理 GPU 编号猜测设备。训练入口发现共享 keeper 已退出时尝试在当前步骤恢复，不再由辅助进程的存活检查直接阻断主训练。相关本地和 Torch 登录节点测试均通过（GPU 身份 2 项、工作流 5 项、启动器 4 项）。修复后的 GPU 实际运行仍待验证。

新恢复作业 **18790400** 已提交，2 × H200、16 CPU、500G、24 小时；account、partition 和系统选择的 QOS 与原作业一致。提交记录为：
`/scratch/xl6775/projects/EgoQA-six-user-baseline-grpo-20260929/submissions/grpo_recovery_20260929T104334Z_66b3ad96/submission.json`

同目录 `workflow.json` 记录旧数据来源 `data/grpo_18719659`，`attachment.json` 记录自动接入状态。后台程序 PID 为 `818388`，登录主机为 `torch-login-b-1`。提交命令结束后独立核验该程序仍存活，状态为 `waiting_for_allocation`；新鲜队列状态为 `PENDING (QOSGrpGRES)`，表示等待 QOS 组级 GPU 资源额度。排到资源后自动接入，直接复用已完成的数据；执行一次最小更新检查后进入首轮 12 次更新，不再抽帧或运行 CLIP。两个模型均保持 Qwen3.8-27B。

新结果写入 `outputs/workflow_18790400`、`outputs/smoke/train_18790400` 和 `outputs/formal/train_18790400`，旧数据和失败证据保留。资源日志为提交目录下的 `keeper_18790400.out/.err`，步骤日志为 `step-18790400.0.out/.err`（实际步骤号以 `attachment.json` 和 Slurm 为准）。

以下章节保留此前提交配置和计划作为来源记录，其中“等待资源”是历史状态。已生成帧、数据划分与失败日志继续保留，未取消其他任务。

遵循外层工作区的 [Torch 通用手册](../../../docs/Torch通用复现项目执行手册.md)、[实验元规则](../../../docs/TORCH_EXPERIMENT_META_RULES_CN.md) 和 [Runbook 模板](../../../docs/TORCH_RUNBOOK_TEMPLATE_CN.md)。本文件记录已经上传、已经启动自动接入程序的实际任务；不是新的提交模板。

## 1. 本次模型与算法

- Policy：`/scratch/xl6775/models/Qwen3.8-27B`，从基座初始化并训练语言侧 LoRA。
- Judge：同一基座路径的独立冻结实例，`judge_mode=baseline`，不加载 adapter。
- 奖励：`0.2 × 题面通过概率 + 0.4 × 证据通过概率 + 0.4 × 全六视角可答概率 × (1 − 单视角可答概率)`。
- GRPO：`advantage_estimator=grpo`、`scale_rewards=group`；插件返回原始奖励，框架在组内标准化。
- 策略 LoRA：rank 8、alpha 16，目标 `q_proj/v_proj`；BF16、梯度检查点，视觉与对齐模块冻结。
- 每组 4 个候选；训练微批 1、累积 4；学习率 `1e-5`、KL 系数 `0.04`、温度 `0.85`、top-p `0.95`、top-k `40`。
- Policy 非思考模式，SDPA；输入上限 65536 token、生成上限 1024 token、单图像素上限 24576。Judge 独立使用当前完整帧与自适应图像预算。
- 以上是本次计划执行配置，不是训练已经成功的证据。

## 2. 代码、输入与环境

本地仓库：`C:/Users/20661/Desktop/Research/AR/multiuser/EgoQA-main-20260928`，分支 `codex/six-user-binary-grpo`，包含未提交修改；未 push。

远端独立代码目录：`/scratch/xl6775/projects/EgoQA-six-user-baseline-grpo-20260929`。使用窄 SFTP 上传源码包和九文件接入更新包，更新前旧文件备份在该目录 `incoming/pre_attach_20260929_011702`；历史项目未修改。

训练解释器：`/scratch/xl6775/envs/egoqa-ms-swift-v4.2.2/bin/python`。

Judge 解释器：`/scratch/xl6775/conda/envs/qwen38-vllm/bin/python`。

FFmpeg：`/scratch/xl6775/envs/egoqa-ffmpeg-runtime/bin/ffmpeg`，工作流同时设置其 `bin` 和 `lib` 环境。

实际配置文件：`multi-user/hpc/grpo_v3/six_user_binary/workflow_27b_18719659.json`，已上传远端。

数据清单：`/scratch/xl6775/datasets/egoqa_fps_newprompt_20260918/base_candidates.jsonl`。五条记录对应四个独立同步窗口，每个窗口六段十分钟成片；重复 speaker 记录去重。直接使用已有成片，不重新下载视频。

| 用途 | 独立窗口 | 计划 asker 输入数 |
|---|---|---:|
| 训练 | DAY1_17200000、DAY3_17000000 | 12 |
| 验证 | DAY4_21400000 | 6 |
| 保留测试 | DAY1_11400000 | 6 |

每个窗口每人按 0.5 FPS 采样 300 帧，再用仓库既有 CLIP 聚类与 provider-all-pairs 剪枝。Speaker 帧完整保留，Judge 使用完整采样帧。实际抽帧和媒体时长检查将在计算节点执行；表中的数值是计划输入数量，不是已生成或 accepted QA 数量。

## 3. 资源作业与自动接入

资源作业：`18719659`，`gpu-hold-grpo`。

| 配置 | 实际值 |
|---|---|
| account | torch_pr_674_tandon_advanced |
| partition | h200_tandon |
| QOS | gpu48，由系统选择 |
| GPU | 2 × H200，不固定节点 |
| CPU / 内存 | 16 / 500G |
| 时限 | 24 小时，用户明确要求的持续资源分配时限 |

第一次显式指定 `normal` QOS 被站点拒绝，没有产生 JobID；移除该参数后提交成功。不能把账户关联查询中的 `normal` 原样用于本任务。

资源提交记录：
`/scratch/xl6775/projects/EgoQA-six-user-baseline-grpo-20260929/submissions/hold_20260928T164805Z_a0f093b8/submission.json`

GRPO 自动接入记录目录：
`/scratch/xl6775/projects/EgoQA-six-user-baseline-grpo-20260929/submissions/grpo_attach_20260929T011841_65c451a9`

该目录的 `submission.json` 保存接入程序 PID 和配置，`attachment.json` 保存当前状态及实际 `srun --jobid=18719659 --overlap --exact` 命令。程序已经启动；验证时状态为 `waiting_for_allocation`。最新一次调度核验为 `PENDING (Priority)`，属于正常排队。此前的配额等待原因已经变化，未调整 partition、未取消或重复提交作业。

不需要用户再手工提交。后台程序等待资源后自动创建作业步骤；资源保留作业本身不执行 GRPO。

## 4. 到位后的执行顺序与产物

1. 在作业专属 scratch 中封闭 HOME/cache/tmp，检查存储可写性；复用本 allocation 的 keeper，不另启重复 keeper。
2. 检查 24 段同步成片并建立四个完整六用户帧与剪枝掩码数据包。
3. 生成按 source packet 隔离的训练、验证和保留测试输入。
4. 运行唯一一次最小 smoke：1 个训练输入、1 个验证输入、1 次 optimizer 更新。
5. smoke 的训练工程检查通过后，启动首轮 12 次更新。正式训练重新从 27B 基座初始化 LoRA，不把 smoke checkpoint 当作额外训练起点。保留测试不传入 trainer。

主要输出均由 JobID 派生：

- `outputs/workflow_18719659/workflow_status.json`：预处理、smoke、正式训练阶段状态。
- `data/frames_18719659`：实际帧、聚类与六种 asker 掩码。
- `data/grpo_18719659/split_manifest.json`：实际数据划分。
- `outputs/smoke/train_18719659`：最小更新检查。
- `outputs/formal/train_18719659`：首轮训练、逐候选 reward、验证和 checkpoint。
- 接入记录目录下 `step-%J.out/.err`：实际 Slurm step 日志，`%J` 由 Slurm 展开。

失败会保存状态和日志，自动接入程序停止后续阶段；不会取消 `18719659` 或其他作业。未知 GPU/模型错误不转成默认奖励。完成、失败或用户终止时，训练驱动清理自己启动的子进程；资源保留作业仍按其 24 小时时限运行，取消需要用户明确指定 JobID。

## 5. 单次查看状态

运行位置：Torch 登录节点 Bash。此块只读，不提交任务，也不关闭登录会话。

```bash
ROOT=/scratch/xl6775/projects/EgoQA-six-user-baseline-grpo-20260929
TASK=${ROOT}/submissions/grpo60_samealloc_20260930T205412Z_9aeb388f
sacct -X -j 18883354 -o JobID,State,ExitCode,Elapsed,NodeList
squeue -j 18883354 -o '%.18i %.10T %.10M %R' 2>/dev/null || true
if [ -s "${TASK}/attachment.json" ]; then cat "${TASK}/attachment.json"; fi
if [ -s "${ROOT}/outputs/workflow_18883354_portfix1/workflow_status.json" ]; then
  cat "${ROOT}/outputs/workflow_18883354_portfix1/workflow_status.json"
fi
tail -n 40 "${TASK}/dispatcher.log"
```

遵照用户要求，助手不持续轮询排队；后台接入程序的等待仅用于自动开始任务。

## 6. 当前证据边界

| 层级 | 当前证据 | 结论边界 |
|---|---|---|
| 本地 | 前轮完整 32 项通过，本次启动路径与后端修复相关 14 项通过 | 代码逻辑，不代表 GPU 可运行 |
| 远端零GPU | 两环境导入/pip check、编译与动态加载、7200图解码和两项编码通过；CPU作业18883207完成 | 不证明CUDA内核、完整Judge服务或梯度更新 |
| 调度 | 18883354已获2张H200；18883354.0失败后18883354.1已在同一allocation启动 | 不能仅凭步骤RUNNING推断训练通过 |
| GPU / 更新 | 18883354.0的Judge已ready；Policy在torchrun端口29500冲突时失败 | 当前portfix1步骤的更新与验证尚待产物核验 |
| QA 质量 | 未评估 | 代理奖励变化不能替代人工终点评估 |
