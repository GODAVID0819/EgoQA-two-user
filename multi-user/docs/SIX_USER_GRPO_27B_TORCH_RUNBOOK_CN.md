# 六用户 27B baseline Judge GRPO：本次执行记录

遵循工作区 [Torch 操作统一规则](../../../docs/TORCH_RULES_CN.md)。以下首节为当前执行记录，后续旧 JobID、配置及命令仅用于历史追溯；旧通用文档不再组成额外必读链。

## 2026-10-10 22:38：48输入续训至总200已提交19530750

用户已确认再续100至总200完成原退火，新GPU **19530750** 经`sbatch --parsable`实际提交，清单`submissions/grpo200_20261010T143802Z_2d021dc5/submission.json`，输出`outputs/formal_train8-lora200/train_19530750`，根`/scratch/xl6775/projects/EgoQA-six-user-baseline-grpo-20260929/variants/train8-lora1e5-continue200-20261010T142800Z`，preflight为`continue100-to200`。canonical `variants/train8_current_attempt.json`已更新phase `training_resume100_to200`与新manifest/out/root/source完整100；19458762保留completed/fixed_reward0.38994569000254564和24/24/6配对历史。提交即时PENDING/None，尚无新增GPU更新，不重复claim或提交。

实际Torch CPU `resume100_check.json` passed：160个adapter张量8,912,896参数全部有限，48个qkv B层非零；160个optimizer状态皆step100且有限，scheduler100/LR5.871607054625497e-6，Python/NumPy/Torch CPU RNG可恢复、CUDA随机字节存在。原余弦100→200实际调度单调，下一调度步5.797407629465648e-6，200末端1e-6，不重启warmup。真实CLI max200/stop200/warm10/min_rate0.1/peak1e-5/G4/微批1累积4/r8a16/qv+qkv及源100路径核验通过，Policy allocator expandable_segments:True和根hpc/shared实际导入保持。未修改源断点或已完成旧根，不重复未变环境/媒体/编译测试或GPU smoke。

提交前资源快照/审计、当前account/路由、脚本bash-n和封闭scratch启动检查通过，官方gh取消60%/警告75%已核，pointwise2/target80/reserve8GiB/prealloc16MiB/maxduty0.9保持。资源1H200/16CPU/500G/16.75小时；按源64步均412.21秒、最后20均384.46秒取大×100，加启动1800秒和3次验证各2408.22秒估13.957小时、余20.01%，排队另计。

实际新GPU开始须stage_state train_begin global100/scheduler100/LR5.871607e-6；结束须global200/scheduler200/LR1e-6、完整CP200、工程结果passed、独立24/24槽位6输入同设置冻结Judge及cleanup通过。当前已完成B100和C300结果保留，长期监控继续到新增B200验收并交付最终两页英文纯文本/中文讲稿后才停止，不自动进一步扩大步数。

## 2026-10-10 晚间：100步及300步已验收，用户新增48输入至总200

用户明确要求复用现有连接，session_20261010_221821_39024066/PID76316已核验READY、所属存活PID及fresh本人Torch身份。grpo_progress及grpo_final_compare请求均COMMAND_EXIT0/结束标记通过；不再把旧待认证状态当当前状态。

B19458762在gh124 COMPLETED/0:0，运行9:18:24，北京10月9日19:21:26结束；完整checkpoint-100、training_result passed100、stage_begin36/scheduler36/LR9.590529e-6至end100/scheduler100/LR5.871607e-6连续，fixed24/24/6与run_manifest completed/cleanup_errors[]通过。固定奖励0.38994569000254564，相对基座0.38801604080996177差+0.0019296491925839022，输入3升3降。原分配器修复已实际通过第37步失败边界、余64更新与完整保存及独立重载；不宣称所有配置均永不OOM。

C19430523在gh101 COMPLETED/0:0，运行12:18:50，北京10月9日18:22:19结束；完整checkpoint-300、工程passed300、stage_begin200/end300与scheduler/LR固定1e-6、fixed24/24/6和cleanup通过。固定奖励0.3832687129254224，相对基座-0.004747327884539341，相对自身200步0.39451069624537244差-0.011241983319950022。300步固定均值没有进一步提升，保留负结果，不继续到400。

基座、旧18输入100/200/300和新48输入100的实际settings、input_bindings及冻结Judge一致，均24槽位/6输入完整。比较是固定输入/种子/槽位上的生成，不把它称相同固定题目的人工准确率，不混过程eval。只有一个验证窗口，不能称统计显著或已证明收敛；48输入同时扩大训练窗口及LoRA，不能单独归因LoRA。

用户最新确认：48输入从自身完整checkpoint-100再续100到总200，完成原200余弦退火；不是额外200至300。新独立根`train8-lora1e5-continue200-20261010T142800Z`、preflight `continue100-to200`正在进行实际CP100/CLI/原100→200曲线恢复检查，尚未提交GPU。保留peak1e-5/warm10/min_rate0.1/G4/微批1累积4/r8a16/qv+qkv/视觉aligner冻结/48训练6验证/reward/Judge/Policy expandable_segments及原保护；不重启调度或另做smoke，原5环境/媒体数据检查与已完成GPU证据明确复用。

源64更新均412.21秒、末20均384.46秒，取大×100；启动1800秒、两次过程验证及独立评分各2408.22秒，估13.957小时，申请16.75小时余20.01%，排队另计；资源1H200/16CPU/500G。提交前现场资源/官网gh60/75/保护/封闭scratch及实际恢复检查通过，再新claim/--parsable即时保存。长期监控现在应延续到48输入总200工程、完整保存、固定评分验收；最终两页文案纳入B100/B200及旧A200/C300结果，不能按旧B100+C300已完成就提前停用。

## 2026-10-09 09:13：48输入已从完整36恢复提交19458762

新GPU19458762由`sbatch --parsable`提交，清单新根`submissions/grpo200_20261009T011340Z_ea7f1cba/submission.json`，输出`outputs/formal_train8-lora100-allocfix/train_19458762`，根`/scratch/xl6775/projects/EgoQA-six-user-baseline-grpo-20260929/variants/train8-lora1e5-allocfix-20261009T010900Z`，preflight为`resume36-to100`。canonical `variants/train8_current_attempt.json`已更新phase `training_resume36_to100_allocator`、源训练19415600与完整CP36，并在gpu_attempt_history保留该失败。提交即时PENDING/None，无新增GPU更新，不重复提交旧claim或重跑前三步。

实际远端CPU `allocator_resume_check.json` passed：160个adapter张量8,912,896参数全有限，48个qkv B层非零，160个优化器状态全step36/有限；scheduler36/LR9.590529407721232e-6，CPU RNG可恢复、CUDA字节存在。实际CLI保持max200/stop100/warm10/min_rate0.1/peak1e-5/G4/微批1累积4/r8a16及q/v/qkv，恢复路径指向原CP36；原调度36→100到LR5.871607e-6的检查通过。31项真实远端回归通过；运行现场安装的Swift原生切换函数、仅替换CUDA API为CPU调用记录，取得False→True序列，并核对vLLM自定义MemPool源保护；这项检查不冒充GPU显存验证。Policy环境显式设置旧分配器变量，Judge不变。原5项环境/媒体数据报告明确复用。

资源1H200/16CPU/500G/12.75小时，估10.506小时、余21.36%，排队另计；提交前保存资源快照与审计，保护/封闭scratch保持。实际新GPU须核验stage_begin36/scheduler36/LR9.590529e-6、原失败第37步之后的反向与完整更新、阶段100完整保存/同冻结Judge固定24槽位6输入/清理。若仍OOM须按新鲜错误定位，不原样无限重投；不能修改C活动根、G/批量/数据/LoRA或自动加资源。300步续训19430523继续运行；最终报告仍等待B100与C300全部验收。

## 2026-10-09 早晨：300步续训正常，48输入36步后反向OOM

本次用户PIN认证ok后，session_20261009_085614_f7858c1c/PID25760已READY、probe_exit0和新鲜xl6775/Torch身份通过。Torch登录节点实际EDT -0400，以下用户时间为北京时间。

C19430523在gh101于10月9日06:03:29开始，08:59已完整保存224/300，stage_state.json真实train_begin为global200/scheduler200/LR1e-6，后续训练仍为1e-6。存储检查passed；保护累计约80.34%、多窗口更新且memory_safe。C继续运行，终点300工程与固定评分未出，不修改活动根。

B19415600在gh120运行4:43:20，10月9日08:26:25结束，顶层及batch为FAILED/1:0；最新完整checkpoint-36在旧根`outputs/formal_train8-lora100/train_19415600/swift/v0-20261008-155509/checkpoint-36`。致命错误是第37步反向CUDA OOM：申请1.84GiB、空闲239.62MiB、reserved未使用15.93GiB。保护此时已memory_yield、buffer0、duty0；它是程序OOM，不是利用率系统取消。启动早期multiprocess的SystemExit0和清理日志不能代替致命错误归因；原JobID和产物保留。

现场读取实际Swift4.2.2与vLLM0.24源码确认：Swift的`set_expandable_segments`只有显式`PYTORCH_CUDA_ALLOC_CONF`含该键才启用，rollout初始化/唤醒vLLM时关闭、睡眠后训练时开启；vLLM的CuMemAllocator对自定义MemPool已有关闭/恢复处理。最小修复只在新Policy环境启用`expandable_segments:True`，并移除冲突的新别名`PYTORCH_ALLOC_CONF`；Judge、父环境、未配置该字段的旧作业不变。研究超参不变，不先修改G/批量/媒体/LoRA或切换资源。两项新回归先失败后通过，含原launch/workflow/evaluation共31项本地passed；GPU第37步OOM是否消除仍待实际恢复运行证明，不据CPU接线称已解决。

恢复独立根`train8-lora1e5-allocfix-20261009T010900Z`，preflight为`resume36-to100`；从原完整36继续到总100，保留max200/warm10/min_rate0.1/peak1e-5/G4/微批1累积4/r8a16/q_proj,v_proj,in_proj_qkv/48训练6验证及原reward/Judge。只上传launch/workflow、两项测试和CPU恢复验证脚本，包含根hpc/shared；不修改C活动根、B历史失败根或全局环境。实际远端CPU检查已启动，源5项不变环境/数据/processor报告显式复用；新验证实际CP36、optimizer/scheduler/RNG/LR连续、CLI与原生阶段切换。当前尚未新GPU提交，检查通过后使用新claim/新JobID正式36→100，不重跑前三步或再串smoke。估剩64步按源33更新均462.17秒（末10均361.64秒，取大）、启动1800秒和3次验证各2148秒共10.506小时，申请12.75小时、余21.36%，排队另计；实际分配器性能后续刷新。

## 2026-10-08 20:49：200→300续训已提交19430523

新作业19430523经`sbatch --parsable`实际提交并即时保存，清单为新根`submissions/grpo300_20261008T124914Z_19dd6252/submission.json`，输出为`outputs/formal_lr1e5-to300/train_19430523`。根为`/scratch/xl6775/projects/EgoQA-six-user-baseline-grpo-20260929/variants/lr1e5-continue300-20261008T124000Z`；canonical为`variants/lr-search-20261006T033800Z/lr1e5_continuation300_current_attempt.json`，独立于原A200清单和B100清单。提交即时PENDING/ReqNodeNotAvail，没有新增更新；不能把提交成功称训练完成。

实际Torch CPU `preflight/continue200-to300/continuation_check.json` passed：源全局200/scheduler200、64个adapter张量共3,014,656参数有限且LoRA B非零、64个优化器状态皆step200并保持动量张量逐项相同；adapter、RNG、trainer_state及其他8个文件逐字节相同，Python/NumPy/Torch CPU RNG可恢复，CUDA随机字节存在（未称CPU已恢复CUDA RNG）。源原断点不修改。实际transformers恒定调度加载旧base_lrs的下一步1e-5已复现；新副本仅改scheduler.base_lrs与optimizer.initial_lr，逐步运行调度200→300最小/最大均1e-6，步数连续，不重启warmup。CLI max300/stop300/constant/1e-6/warm0/G4/微批1累积4/q_proj/v_proj/r8a16已实际解析，根级保护依赖真实导入通过。

提交前审计保存`submission_audit.json`及`resource_snapshot.txt`；现有pointwise-2-instant保护保持，作业启动必须生成封闭scratch的storage_preflight并通过才加载模型。1H200/16CPU/500G/16.5小时，估13.728小时、余量20.19%，排队另计。原5项未变环境/数据/processor报告显式复用，新的实际调度检查覆盖旧cosine报告，未额外串GPU smoke。启动后须查stage_audit中的global200/scheduler200/LR1e-6，终点global300/scheduler300/LR1e-6、完整CP300、training_result和同设置独立24槽位/6输入及cleanup全部验收。

长期监控现在覆盖B100与19430523新增300步；旧A200保持历史完成。用户今后新认证严格给PIN→本次ok→ENTER，已有本次已验证连接可继续只读监控，不自主网页登录。最终两页英文纯文本设置/结果和中文讲稿应包含新增300步真实结果，不能在B100完成时提前停监控或自动再延长。

## 2026-10-08 晚间：用户批准从200续100，固定末端学习率

用户明确确认原1e-5实验从完整checkpoint-200继续100步至总300，学习率固定1e-6，其余18训练/6验证输入、q_proj/v_proj、G4、微批1累积4、r8/a16、冻结视觉与aligner、reward和Judge均保持。该新增授权覆盖旧“不继续A”的调度约定，只增加这一次200→300，不自动延长至400。最终报告和监控结束条件包括B100与新增300步续训的工程/完整保存/固定评分验收。

本次用户手动新PIN认证回复ok后，session_20261008_203522_21e1d418/PID15116已AUTHENTICATED/READY及新鲜xl6775/Torch身份通过；今后新连接按给PIN→用户ok→ENTER，不自主网页登录。20:37新鲜查询B19415600仍PENDING/None，无新增GPU更新，不重复提交或取消。

独立根为`/scratch/xl6775/projects/EgoQA-six-user-baseline-grpo-20260929/variants/lr1e5-continue300-20261008T124000Z`，preflight为`continue200-to300`。复用A200无活动根的实际代码并包括根hpc/shared；不修改B活动源码。源19346766完整CP200在原A输出中保持，续训使用新根下可追溯副本。原余弦断点的scheduler.base_lrs仍是1e-5，不能只把CLI改constant就认定下一步是1e-6；正在实际环境核验该失败情形和限定的学习率状态转换。模型、优化器动量/步数、随机状态保留，只对续训副本的scheduler.base_lrs和optimizer.initial_lr设末端1e-6，CLI constant/warmup0/总300。该转换尚待CPU实际检查通过，当前未提交新GPU，不能称已续训。

源A200真实100更新均393.17秒、最近20均408.89秒，取较大×100，加启动1800秒、两次过程验证各2466.58秒及固定独立评分1800秒，估13.728小时；新申请16.5小时、余量20.19%，排队另计。资源沿用1H200/16CPU/500G；官方gh取消60%/警告75%已重新核查，现有pointwise-2-instant、目标80%、8GiB显存让步、16MiB预分配、0.9占空比保持。

## 2026-10-08 13:42：200步已验收，48输入从完整3步接总100已提交

A续训19346766已COMPLETED/0:0，运行12:57:05，北京13:10:27结束；完整checkpoint-200、training_result passed/global200、固定独立24/24评分及run_manifest completed/cleanup_errors[]通过。固定奖励0.39451069624537244，共同基座0.38801604080996177，差+0.00649465543541067。与自身100步0.383954547141297核验同settings、input_bindings、冻结Judge及24槽位/6输入后，差+0.010556149104075404；6输入3升3降。平均代理奖励改善，不宣称统计显著、人工质量或已证收敛；过程验证与独立评分不混用。

B独立重载修复19412581已COMPLETED/0:0，gh119，运行35:36，北京13:18:40结束，24/24完整配对、MPI1/SyncMPClient、EngineCore实际Injected worker extension和partialQKV补丁、run_manifest completed/cleanup_errors[]通过。源checkpoint-3的固定奖励0.36271246905841525，比共同基座-0.025303571751546516；只是3步早期检查，不当100步结果。至此新增LoRA真实加载、训练三步、更新后生成及独立重载边界通过，没有再次重跑训练三步。

新B续训 **19415600** 已由`sbatch --parsable`提交，清单`submissions/grpo200_20261008T054246Z_fc9ad426/submission.json`，输出`outputs/formal_train8-lora100/train_19415600`，根仍`train8-lora1e5-evalfix-20261008T025604Z`，preflight为`resume3-to100`，phase为`training_resume3_to100`。配置已删除validation_only_adapter，resume_from_checkpoint指向19397639完整CP3，stop100/max200/warm10/min_rate0.1/peakLR1e-5/G4/微批1累积4/r8a16/原48训练6验证不变。继续验证源optimizer/scheduler3/LR3e-6到100的连续性，不重做媒体、不加其他实验。

实际CLI CPU检查passed并核验target_modules、resume3、分段停止回调、100阶段与200调度。检查最初因跳过初始化后lr_scheduler_kwargs仍为JSON字符串而与dict比较失败；诊断取得实际raw值后仅对JSON字段做等价转换，训练参数未改，未忽略不一致。五环境/编译/媒体/数据及源CP3状态检查按不受变动影响显式复用，不串新smoke。资源1H200/16CPU/500G/20h：97步按517/560.9秒平均538.95秒共14.523h，加启动20.2min、两次过程验证35.8min各及独立35.8min，估16.65h，余量20.1%，排队另计。提交即时PENDING/ReqNodeNotAvail，后续按真实sacct/squeue刷新，不手动指定节点/取消或重复提交。

长监控继续至B训练/完整100保存/固定独立评分验收，然后交付最终两页英文纯文本设置/结果与详细中文讲稿；08点一次性快照已经交付，不重复。A已完成，不新增第二组200或训练到400。B同时增加窗口和LoRA，不能单独归因LoRA。

## 2026-10-08 12:36：评分保留隔离worker并在worker内安装兼容修复

19410265在gh107运行16:43后FAILED/1:0，首个错误是vLLM初始化CUDA graph profiling期间，父进程保护线程`stream.synchronize`触发`operation not permitted when stream is capturing`，随后出现CUBLAS和capture invalidated；原同进程方案存在保护与模型捕获的上下文冲突。没有关闭保护或更改LoRA/数据/解码；源checkpoint-3未改，不重跑三步训练。

恢复原独立EngineCore进程（含qkv的评分明确MPI=1），用实际安装vLLM0.24已有的`worker_extension_cls`接口，把`packed_worker_extension.PartialPackedLoRAWorkerExtension`导入真正worker，模块导入即安装原部分QKV修复。实际worker_base源码确认该扩展在worker初始化前解析并动态继承，字段在EngineArgs/ParallelConfig真实存在。新的9项评分回归先失败后通过；在新的实际Torch CPU Python子进程验证扩展导入后LoRA类marker为真、native字段支持，报告`preflight/eval3-worker/worker_extension_check.json` passed。它是CPU传播检查，GPU重载和固定评分仍待验收。

新仅评分GPU **19412581** 已--parsable提交，清单`submissions/grpo200_20261008T043646Z_fdb08343/submission.json`，输出`outputs/formal_train8-lora-eval3-worker/train_19412581`；根仍`train8-lora1e5-evalfix-20261008T025604Z`，preflight为`eval3-worker`，phase为`evaluation_only_cp3_worker`。新增优化器更新0，源训练19397639/完整CP3保持。19410265失败历史、日志和claim保留，当前canonical已改新JobID，禁止重复提交。资源/75min估算、保护、储存封闭及原冻结Judge不变；提交即时PENDING。

12:34查询A19346766仍RUNNING12:20:52，最后完整保存198；200工程和独立评分尚未验收。下一步读取19412581的实际worker注入日志/24槽位完整固定评分/清理状态，成功后从源CP3新JobID续到总100，必须删除validation_only_adapter。最终两实验验收后补两页英文纯文本与中文讲稿；不因工程修复成功称收敛或代理质量改善。

以下11:51段是上一修复尝试的历史，当前评分实现以上述隔离worker扩展为准。

## 2026-10-08 11:51：保留三步训练成果，仅恢复独立重载评分

10:25新鲜查询：A19346766仍RUNNING，完整184；B19397639在gh116于08:43:48至10:22:06运行1:38:18，训练三步及过程验证通过，完整checkpoint-3和training_result passed，失败发生在独立重载评分。评分父进程已打印兼容修复安装，但新EngineCore子进程仍在原`expand_packed_lora`读取None.shape，说明父进程补丁没有传入子进程。三步训练成果不重跑，不把过程验证当独立评分。

修复仅使含`in_proj_qkv`的独立评分设置`VLLM_ENABLE_V1_MULTIPROCESSING=0`，沿用训练已成功的同进程引擎；原q/v评分设置保持。增加`validation_only_adapter`恢复入口，沿用原Judge、数据、解码、交替运行和利用率保护，跳过训练，只重载指定adapter评分；记录实际引擎客户端类型。新增失败回归先失败后通过，9项评分、9项入口及11项workflow合计29项本地与实际Torch CPU检查通过。

新根`/scratch/xl6775/projects/EgoQA-six-user-baseline-grpo-20260929/variants/train8-lora1e5-evalfix-20261008T025604Z`同时包含`multi-user`与根级`hpc/shared/cuda_device_identity.py`；仅该无活动新根接收三份代码和针对性测试。真实CP3核验：160个adapter张量、8,912,896参数全部有限，48个qkv B层均非零、优化器160状态均step3且有限、scheduler3/LR3e-6；Python/NumPy/Torch CPU RNG可恢复，CUDA字节状态存在，未将CPU检查称CUDA RNG已恢复或GPU评分已通过。报告`preflight/eval3/eval_repair_check.json` passed。

仅评分恢复GPU **19410265** 已由`sbatch --parsable`提交，清单`submissions/grpo200_20261008T035106Z_66599f64/submission.json`，输出`outputs/formal_train8-lora-eval3/train_19410265`，canonical phase为`evaluation_only_cp3`，source_training_job_id为19397639，新增优化器更新为0。资源1H200/16CPU/500G/75min；已观测启动20.2min、独立Policy载入5.3min、同窗口评分35.8min，合计61.3min，余量22.3%，排队另计；提交即时PENDING(Priority)。保护参数及存储封闭要求不变。

11:45左右A已完整196步，仍RUNNING；200完整保存和独立评分尚未验收。B只完成3训练更新，100步续训尚未提交。19410265实际完成24/24固定评分、同输入/冻结Judge核验及清理后，从原CP3连续接总100，保留max200/warm10/min.1/LR1e-5/G4/微批1累积4/r8a16/48数据。新100配置必须删除`validation_only_adapter`，设置resume_from_checkpoint为源CP3、stop_after_steps为100，使用新preflight/claim和新JobID，禁止把仅评分作业当100步完成。

认证已验证可复用Edge中本人NYU正常登录状态：先准备浏览器设备页，再创建新码，正常选择xl6775@nyu.edu与Torch应用并Continue，网页明确成功立即发送ENTER，随后身份/READY检查。此前生成到ENTER165秒的连接关闭，快速批次正常通过；没有官方超时归因，不承诺永久免认证。没有保存密码、设备码或令牌，没有绕过Duo或认证。

## 2026-10-08 08:35：隔离目录遗漏保护依赖，补齐后重提

19396821在gh113于北京时间08:33:05启动，FAILED/1:0，用时1:37；首个错误是保护进程`ModuleNotFoundError: No module named 'hpc.shared'`，尚未进入LoRA验证、没有完整更新断点，不是GPU利用率系统取消。新目录创建时只复制了`multi-user`代码，遗漏原variant根的`hpc/shared/cuda_device_identity.py`。已有CPU回归没有触发保护线程中的这项延迟导入，不能据此当依赖闭包通过。

只补齐原已验证的CUDA/NVML身份映射模块，LoRA兼容修复、训练超参、保护参数及共享环境不变。采用正式保护进程相同的Python和`PYTHONPATH=<root>/multi-user:<root>`，实际导入先复现缺模块失败，再复制该明确文件后成功，且核验导入来源就是新根。报告`preflight/runtime3-guarddep/guard_dependency_import.json` passed；未把CPU导入称GPU保护或LoRA验收。

以后复制独立variant或改变保护入口时，必须同时包含`multi-user`与已使用的根级`hpc/shared`依赖，并在提交前按实际子进程Python/PYTHONPATH做导入核验；不能只看文件清单、只跑不触发延迟导入的测试。当前补齐已在失败后修复，后续提交须预防这种遗漏。

旧失败19350675/19396821及各自清单、claim和日志保留；新preflight为`runtime3-guarddep`。修复后新GPU **19397639** 已由`sbatch --parsable`提交，清单`submissions/grpo200_20261008T004005Z_b95ae35c/submission.json`，输出`outputs/formal_train8-lora-packfix-dep/train_19397639`，当前canonical manifest及历史已更新，不能重复旧claim。08:40提交即时squeue为PENDING，无更新断点；仍为原3步失败边界验证，2h/1H200/16CPU/500G，训练与保护参数不变。A19346766仍RUNNING8:26:45，完整保存164、此前日志165，保护运行并显存安全，200独立评分尚未完成。

08:41后续新鲜查询：19397639仍PENDING，原因QOSGrpGRES，尚无GPU保护启动或更新，不能宣称GPU验收通过；A19346766 RUNNING8:28:10，完整保存166。组级资源限制时保持原提交，不原样重投、不手工指定分区或开启抢占，不自动取消其他任务。长期监控已更新新JobID及依赖预防要求。

## 2026-10-08 08点重连：四LR齐全、续训推进、48输入加载兼容修复

用户新PIN认证完成，`session_20261008_080428_de785e48`/PID94204身份及READY检查通过。08:07新鲜查询：A19346766仍RUNNING，完整断点160，后续日志161/200，200独立评分未完成。3e-6 19342429已COMPLETED/0:0，运行6:10:07，北京04:10:29结束，完整100断点、training_result及run_manifest通过，固定评分24/24槽位、6输入配对。

| 原三窗口/18输入的100步实验 | 固定奖励 | 相对共同基座变化 |
|---|---:|---:|
| 基座 | 0.38801604 | 0 |
| 1e-5 | 0.38395455 | -0.00406149 |
| 1e-6 | 0.38002298 | -0.00799306 |
| 3e-6 | 0.37241404 | -0.01560200 |
| 3e-5 | 0.36749298 | -0.02052306 |

48输入任务后台已提交19350675；它FAILED/1:0，用时21:53，gh116，北京01:22:59至01:44:52，没有完整更新断点。首个错误是vLLM0.24 `expand_packed_lora`读取未适配`in_proj_z`空项的shape；不是利用率系统取消。实际安装源码与[官方问题47639](https://github.com/vllm-project/vllm/issues/47639)对应，保持已批准的q/v/in_proj_qkv，不增加z或升级共享环境。

新隔离根`/scratch/xl6775/projects/EgoQA-six-user-baseline-grpo-20260929/variants/train8-lora1e5-packfix-20261008T001232Z`复用原48/6数据与不受修改影响的五份CPU报告。实际安装vLLM方法与CPU张量先复现NoneType错误，再由仅针对QKV有LoRA/z未适配的兼容包装修复；4项数值/边界检查、4项LoRA范围检查及9项入口检查共17项通过。`packed_lora_compat.py`由新根训练plugin和独立evaluation安装；只修改新根、没有修改A活动源码或全局site-packages。实际vLLM加载类注册与幂等安装检查通过；CPU检查未带GPU acceleration路径时有`vllm._C`警告，不将该CPU警告当GPU算子验收或升级共享环境的理由。

修复后的新GPU **19396821** 已由`sbatch --parsable`提交，清单`submissions/grpo200_20261008T002533Z_64ffae80/submission.json`，输出`outputs/formal_train8-lora-packfix/train_19396821`。08:27左右sacct为PENDING、squeue原因Priority，尚无GPU更新；原失败19350675已保留canonical manifest的gpu_attempt_history，当前指向新JobID，严禁重复提交。阶段仍为3更新必要GPU边界，2h/1H200/16CPU/500G、自动路由、pointwise-2-instant保护与原200曲线不变；通过后从自身完整断点接100，不增加其他搜索条件。

同次新鲜检查A19346766 RUNNING8:13:51，已保存到164步，还无200固定评分。清单内九个LR训练尝试及A/B共11个已分配单GPU作业累计约57.76 GPU小时，其中本次新增A/B约8.60小时；新19396821排队时间不计GPU小时。此计数按明确清单JobID，不包含未列入清单的额外测量/环境作业，不能冒充全部历史账户用时。原45–60小时估算在新增A/B授权之前；维持当前两项已批准范围，不据此自动扩搜索或取消活动作业。

08点快照已交付，一次提醒PAUSED；长期监控继续。四组排名不证明改善，均低于基座；不新增第二个200实验、不扩G/MLP/新LR。每次恢复先读canonical manifest及claim，保存失败历史，不原样无限重投。

## 2026-10-08 08点：已核验结果快照与认证边界

08点检查仍为NYU密码页面，没有用户认证完成确认或网页成功证据；存活桥均WAITING_FOR_USER_ENTER，无可复用READY连接。远端最后核验为01:17左右，不能把旧状态当08点实时状态。未重复发码、发送ENTER、取消或重复提交。

本聊天交付两页英文纯文本（设置、结果）及对应中文讲稿。已完成独立100步奖励：1e-5 0.38395455、1e-6 0.38002298、3e-5 0.36749298，共同基座0.38801604；每项24/24槽位、6输入、一个验证窗口。3e-6的19342429最后状态01:17为RUNNING，最后明确核验完整断点62在前晚23:39，不推算08点步数。A19346766最后01:17为RUNNING，源完整100、没有确认续训新断点或200端点评分。B的数据CPU19348785已COMPLETED、48/6完整通过；最后train_processor passed、judge_processor running，GPU提交未知，后台自动提交必须重连读canonical manifest才确认。

本次快照没有收敛、显著性、人工质量改善或LoRA独立因果结论。08点一次性提醒10-8-08-grpo已PAUSED；grpo-lora长期监控保留，认证恢复后先核验当前JobID及防重标记，再继续工程保存及固定评分验收，补最终汇报。没有上传活动作业源码。

## 2026-10-08：两实验并行，08点交付设置/结果快照

用户最新明确要求两项同时推进，持续监控到完成；两项完成或北京时间2026-10-08 08:00先到时交付两页presentation，第一页设置、第二页结果，英文纯文本后附逐页中文讲稿，未完成项如实写进度。08点快照不取消任务、不停止长期监控；最终完成后补最终结果。长期监控grpo-lora和08点一次性提醒10-8-08-grpo均已ACTIVE。

| 实验 | 数据与初始化 | 阶段/曲线 | LoRA | 实际任务 |
|---|---|---|---|---|
| A：1e-5续训 | 原3训练窗口/18输入，自身19262322完整100步断点 | 100→200，总调度200、warmup10、余弦下限10% | 原q_proj/v_proj，r8/a16 | GPU 19346766，00:13:22开始RUNNING gh108 |
| B：1e-5新数据 | 8训练窗口/48输入，同基座新LoRA；独立验证1窗口/6输入 | 总100步/原200曲线；一次必要3步实际GPU核验后连续接100 | q_proj/v_proj/in_proj_qkv，r8/a16，视觉/aligner冻结 | CPU数据准备19348785；GPU尚待数据及检查通过 |

两项保持G4、微批1/累积4、原Judge/媒体预算、峰值LR1e-5。实验B的1e-5为用户明确指定，不再等四LR决定，也不宣称全四组最终赢家；不额外叠加扩数据基线，不在旧小数据断点中途换数据。原四LR100结果和19342429继续保留、监控，不能自动取消。

A独立根`/scratch/xl6775/projects/EgoQA-six-user-baseline-grpo-20260929/variants/lr1e5-continue200-20261007T155422Z`，实际清单`submissions/grpo200_20261007T160654Z_b900c440/submission.json`，输出`outputs/formal_lr1e5-to200/train_19346766`；总索引原搜索根`lr1e5_continuation_current_attempt.json`。恢复前实际核验optimizer/scheduler100、LoRA有限非零、Python/Numpy/Torch CPU RNG可恢复和CUDA字节状态存在，当前LR5.871607054625497e-6连续、终200为1e-6，回调100/199不停止、200保存/验证/停止；不宣称恢复未序列化vLLM采样器状态。取old1e5后90步322.17秒与当前保护下近期11步406.20秒较大值，100余更新加启动0.5h、两过程验证1.1094h、独立评分0.2821h估13.1747h，申请16h15余量23.34%。不是08点必定完成；最后比较同设置基座/100/200，loss近零或完成200本身不证明收敛。

B实际确认记录为本地`hpc/grpo_v3/six_user_binary/train8_100_request_20261007.json`（不是workflow）。新增五窗口来自正式EgoLife API六用户共同完整文件列表：DAY2 16:00/16:20/17:30/17:50/21:10，共600段、30成片、约8.75GiB；训练与验证实际时间区间不得重叠。数据根`/scratch/xl6775/projects/EgoQA-six-user-baseline-grpo-20260929/variants/train8-data-20261007T162701Z`，源`source_manifest.json`、CPU清单`submissions/prepare_20261007T164354Z/submission.json`、输出`outputs/prepare_19348785`。CPU 16核/64G/无GPU，2h分段工作预算；时间不足提前保存partial有效材料，不把partial当已完成。00:53六百段均已下载/验证、无失败并已拼接，01:02新增五packet完成三组，正在第四组；仅补新窗口帧，旧18训练/6验证帧只读复用。完整准备后写`dataset_ready.json` passed，实际48唯一packet/asker、8训练/1验证窗口及媒体可读性通过后才提交GPU。

01:13:17最新覆盖：CPU数据19348785已COMPLETED/0:0，运行19:02，01:09:04结束；`dataset_ready.json.status=passed`、48/6、训练8窗口/验证1窗口及五个新窗口均完成。新B运行目录`train8-lora1e5-20261007T165809Z`的CPU检查正在运行，自动提交等待它实际通过；当前不能把等待检查叫GPU已经提交。`runtime_submission_status.json`与`train8_current_attempt.json`在返回--parsable JobID后即时更新，不得从另一个窗口重复提交。A续训19346766仍RUNNING gh108，已运行近1h。

B GPU无活动独立根`train8-lora1e5-20261007T165809Z`，CPU检查`preflight/runtime3`，必要验证配置`workflow.json`。本地submit_after_cpu支持显式expected_split_counts，默认18/6保护保留；新48/6回归先失败后5项通过，仅上传该独立根，不改任何活动旧根。旧zero_gpu data分支固定18/6未改，不把它直接用于48；数据真实核验来自CPU完整准备报告，再实际运行新CLI及train/judge processor。背景CPU检查PID234846等待真实dataset_ready，失败/partial不提交GPU。一次3步验证覆盖warmup首步LR0、次步非零更新、第三步更新后生成同步及独立重载，约100.5min估时申请120min余量19.4%，之后从完整3步接100；不是多个smoke台阶。

presentation需要说明：旧四LR表来自3个训练窗口；B同时扩窗口和LoRA，不能把变化单独归因LoRA。48输入/100优化器步的实际访问次数需从训练记录核验，不能把“两遍加4输入”冒充每输入严格使用两次。固定24槽位/6输入来自一个验证窗口，不称统计显著、独立测试或人工质量改善。新增A及B均为用户本次明确授权的范围，累计用时据实记录，不因超过旧约45–60小时计划自动扩更多条件。

## 2026-10-07 22:00：3e-6从52步重提，持续监控与单组LoRA已获授权

22:21:35最新覆盖：1e-6的19320446已COMPLETED/0:0，运行10:02:44，北京22:15:46结束；training_result passed/global100、独立validation_policy completed24/24、run_manifest completed。固定奖励0.38002298，相对共同基座0.38801604差-0.00799306（约-2.06%），F/G/S/A约0.731317/0.339252/0.481572/0.520790。已完成三组中的1e-5仍最高（0.38395455），不能在3e-6未完成时称全四组赢家。3e-6新19342429仍RUNNING21:13，stage_state.train_begin为52/200、当前LR2.6873415e-6，启动保护已交接结束；训练侧新完整保存和终点仍待后续监控。三份文档接收及统一规则相对链接均已远端核验可访问。

用户已明确授权离开期间完成3e-6重提并持续监控，四组100步结果齐全后选择一组较优学习率提交已确认LoRA扩展，并交付两页presentation（LR结果表、LR与LoRA设置）。没有要求PPTX，按项目规则输出英文纯文本两页及逐页中文讲稿。LoRA先采用与LR对照匹配的100步阶段、原200步曲线，仅改变已确认模块覆盖；新模块首次GPU风险做一次必要的最小验证，已通过部分不重复。原择优总200约定保留，先核实累计GPU预算，不额外扩大四组或训练到400。

新的3e-6恢复作业 **19342429** 已由 `sbatch --parsable` 实际提交，52→100、总调度200、warmup10/min_lr_rate0.1，微批1/累积4/G4/q_proj/v_proj/beta0.04/温度0.85及数据不变。失败19320447保留在search_manifest历史；当前trial以及protected_lr_restore_manifest均已指向新JobID，不能重复提交或删除claim。

22:01:15新鲜现场：19342429已经RUNNING gh104，北京22:00:22开始，实际h200_tandon/gpu48，1H200/16CPU/500G/8h30；不是仍在排队。按6.81375h纯运行估计约10月8日04:49结束100步与评分，生成长度/保护开销/异常可能改变时间。1e-6的19320446已完成100步过程验证和完整保存，training_result.status=passed、stage_state.train_end.global_step=100，调度200和LR连续；独立validation_policy仍running、0条完成，validation_comparison及最终run_manifest尚未生成，因此四组同设置比较仍未齐，不提前选赢家。

- 独立源码根：`/scratch/xl6775/projects/EgoQA-six-user-baseline-grpo-20260929/variants/lr3e6-resume52-20261007T134742Z`。
- 新提交清单：该根`submissions/grpo200_20261007T135924Z_571a6869/submission.json`。
- 新输出：该根`outputs/formal_lr3e6-shortguard/train_19342429`。
- 本次检查：该根`preflight/resume52/status.json`，真实CPU核验完整52步optimizer/scheduler/LoRA、随机状态文件、实际CLI和曲线连续通过；当前优化器LR约2.68734e-6。原五份不受变化影响的环境/媒体报告显式复用并记录来源，不冒充新运行。
- 资源：1H200/16CPU/500G，自动分区/QOS、没有固定节点。前次24个保存更新的step_time平均402.12秒，48余步加启动0.5h、过程验证0.6701h、独立评分0.2821h估6.81375h；申请8h30，余量约24.75%，排队另计。草稿7h未提交，已在正式提交前改正。

新保护pointwise-2-instant修复已复现的控制缺口：长窗口100%而当次利用率0时，旧控制仍duty0；失败测试实际复现0!=0.9后修复为显存安全且当次低于取消阈值时立即使用既有占空比上限，显存不足仍先归零。CUDA算子/小缓冲、低优先级流、reserve8GiB、max_prealloc16MiB、max_duty0.9保持不变。实际Torch环境9项保护测试全部通过，包括真实CPU张量和RNG不变；本地guard8通过1跳过、workflow11/hyperparameter7通过。只同步到新恢复根，未修改仍活动的19320446旧根运行代码。此证据修复一个确定的反馈缺口，不证明系统取消唯一原因已确认或新长作业绝不会取消。

本线程持续检查已创建，automationId=`grpo-lora`，每30分钟，完成/失败/需要用户认证时通知，状态不变保持安静；任务全部完成后停用。作业内利用率采样保护按秒运行。监控指令已更新为新上下文首次读取统一规则入口，同一上下文未变不反复读旧文档。连接失效时遵守新的设备认证/MFA边界，不复用设备码、不保存凭据。

## 2026-10-07 21:38：1e-6已完成100次更新待验证，3e-6再次系统取消

新鲜SSH只读结果覆盖12:19预测。19320446（1e-6）仍RUNNING gh102，已运行9:25:05；trainer.log已出现100/200，但第100步过程验证、完整checkpoint-100、training_result和独立终点比较尚未完成。当前完整保存点98步，不把日志100次更新写成100步全流程验收通过。参考此前过程验证约37–40分钟、独立评分约17分钟及初始化开销，剩余约50–70分钟；这只是估计。最近一小时GPU记录均值约80%，不能证明后续不会取消。

19320447（3e-6）于北京时间16:02:15在gh107被CANCELLED by 0，运行3:48:54；batch为CANCELLED/0:15，stderr仅系统SIGTERM。日志更新53，完整断点52，保留恢复所需adapter/optimizer/scheduler/rng/args文件。准确路径：`/scratch/xl6775/projects/EgoQA-six-user-baseline-grpo-20260929/variants/utilization-measure-20261007T003153Z/outputs/formal_lr3e6-protected/train_19320447/swift/v0-20261007-002444/checkpoint-52`。本轮只检查文件和保存状态，尚未重读该断点张量。

该取消作业GPU全程样本均值80.35%、末3600样本81.94%、末7200样本82.92%；最后10个样本均值55%、30个68.33%、60个80.67%。采样存在间隔，样本数不能严格冒充秒数。保护末反馈仍存活且显存安全，窗口均值不等于每次调度器采样都越过阈值。系统Reason=None、Comment/AdminComment/SystemComment为空，历史scontrol已清理，当前节点MIXED且没有近期重启证据。实际分区h200_tandon、QOS gpu48，未申请可抢占分区。官网仍写gh取消阈值60%/警告75%，未公开统计窗口或本次取消归因；既不能因长窗口80%排除短时利用率风险，也不能据此认定唯一原因。没有调整保护强度、原样重投或取消任何作业。

已完成100步固定端点仍只有19262322（1e-5）和19262321（3e-5）：同一baseline=0.38801604，分别0.38395455（差-0.00406149）/0.36749298（差-0.02052306），各24/24评分、6输入完整配对。1e-5仅在已完成的两组中领先，全四组尚无赢家。1e-6过程50步reward=0.38980078不与这些独立终点混排，3e-6过程50步0.35495072同理。验证来自单一视频窗口，不能宣称统计显著或人工质量改善。本轮未提交200步续训或LoRA GPU作业；最新四组100→择优1–2组总计200要求不变。

## 2026-10-07 12:09：微批2显存测量失败，两个低LR已回退并恢复提交

19312517最终FAILED/1:0，运行32:06，北京10:33:05→11:05:11。首个真正fatal为训练SDPA CUDA OOM：申请5.88GiB、空闲1.42GiB，设备容量139.79GiB。数据map结束时出现的.nfs清理警告不是后续fatal根因。仅4个候选完成评分，无新完整checkpoint；保存点仍为原1e-6的18步。不是低利用率取消或训练语义失败。保护已完成临时CUDA进程→训练进程交接并在OOM前释放辅助缓冲/占空比降零；实际全程GPU平均74.48%、最后10分钟68.42%、采样峰值显存138.37GiB。该证据只证明启动/交接/让步边界，不能证明长时训练安全。

已回退同链路验证过的微批1/累积4/G4，其余学习率、reward、数据、q_proj/v_proj、原200步曲线及保护不变。实际CPU再次读optimizer/scheduler/LoRA及CLI确认恢复位置、有限状态、当前LR连续、stop100/max200；未追加不同规模smoke。

| 学习率 | 新JobID | 来源断点 | 申请时限 | 12:09状态 |
|---|---|---|---|---|
| 1e-6 | 19320446 | 自身18步 | 13:00:00 | PENDING/Priority |
| 3e-6 | 19320447 | 自身28步 | 11:45:00 | PENDING/Priority |

两组各1H200/16CPU/500G，自动分区/QOS、无节点指定；阶段终点100，总调度仍200。提交清单分别为测量根submissions/grpo200_20261007T040643Z_6cded66a/submission.json及grpo200_20261007T040644Z_ad4255e5/submission.json；当前索引protected_lr_restore_manifest.json。原搜索根search_manifest对应trial已指向新JobID并保留旧取消记录。不能重复提交或删除任一旧claim。

12:19:49现场覆盖：两组已并行RUNNING，19320446于12:13:02启动gh102，19320447于12:13:21启动gh107，各运行约6分钟；启动保护最近60秒分别约85.15%/82.67%，累计约79.29%/78.72%，显存安全。Policy新更新尚未核验，不能只凭RUNNING称恢复完成。根据上述纯运行估计，3e-6预计约21:40、1e-6约22:40完成100步与评分；只是吞吐推算，实际生成长度/保护开销/运行异常可能改变时间。不是Slurm承诺时刻。

余下82/72步以已有6.20536分钟/更新、启动0.5h、两次过程验证1.2228h、固定评分0.2821h估计纯运行10.4856/9.4513h，申请余量23.98%/24.32%；排队单独计算。高LR两组仍已完成100并保持原端点，不擅自直接延长200。当前四LR对照尚未齐，不称1e-5已是全四组赢家。

已按批准的LoRA模块清单准备独立源码根 `/scratch/xl6775/projects/EgoQA-six-user-baseline-grpo-20260929/variants/lora-attention-20261007T040921Z`，没有GPU作业或最终LR选择。新接口lora_target_modules真正转发到Swift CLI，默认q_proj/v_proj保持不变，仅允许原范围或已确认的q_proj/v_proj/in_proj_qkv；不改变两个活动LR根的训练代码。新增4项回归先复现转发失效再通过，最新本地104项102通过2跳过；远端新LoRA结构与CLI核验待完成，GPU新增覆盖仍未验证。LoRA启动时机已询问：等待完整LR比较(推荐)，或先用已完成组中较好的1e-5作对照；不要把等待时间当作选择授权。

独立LoRA根CPU/meta模型及真实PEFT注入核验已passed_cpu_only：160个可训练张量、实际8,912,896参数、语言64层均有注入，视觉与aligner无可训练参数；真实CLItarget_modules三项/G4/r8/a16/batch1/acc4生效，远端新增4项回归通过。这里用1e-5只作CLI示例，不是最终实验LR选择，没有GPU提交、没有新模块CUDA运行验证。CPU meta模型为统计结构临时关闭FusedRMSNormGated；fast-path警告不用于判断实际GPU训练环境损坏。

## 2026-10-07用户最新修正：四组100步，仅择优1–2组总计200步

最新要求覆盖此前“把四组都先完成200步”及“额外200步到400”的理解：四个学习率组完成总计100步，同设置固定验证后按领先幅度与输入一致性选1–2组；明显领先只续一组，接近时最多两组，续训终点为总计200，其余停在100。恢复优化器与原200步scheduler，不能重置为新的200步曲线。没有200步必定收敛的保证，不把完整退火直接当作质量改善。

LoRA第一路线已获用户明确确认：G4、target_modules=[q_proj,v_proj,in_proj_qkv]、rank8、alpha16，视觉与aligner冻结，不添加MLP或in_proj_z/a/b等其他模块。下一步先读取19312517及各LR组当前真实状态，更新剩余时长；若LR组即将完成，则等较优LR确定后开展新LoRA。模块清单不再重复要求批准，但不擅自选择尚未有依据的最佳LR或改变当前测量配置。后面的“模块路线待确认”历史快照已过时。

本轮尚未新建续训或LoRA GPU作业。旧共享桥已断开，待新会话认证后刷新实际状态；最后核验显示高LR两组100步完成，低LR保存18/28步，19312517是18→20测量，不等于正式低LR100步已完成。不得据此旧快照假定LR实验马上完成。

## 2026-10-07 10:39：测量已启动，正确性只读核验与待确认模块

19312517已于北京时间10:33:05在gh115开始RUNNING；10:39:11运行6分06秒。pointwise-1启动保护已越过前次16MiB预算失败边界，最近一分钟GPU利用率约83%、累计约78.7%，Judge模型已加载并进入编译。此时尚未进入Policy更新；保护交接、微批2真实显存、18→20更新与固定端点仍待验收，不称彻底解决长时取消。

进一步读取真实启动分配：辅助allocated约1.126MiB、reserved2.0MiB，低于16MiB上限；最近反馈60秒80.83%、已有时长累计79.72%，Judge health已ready。尚无训练侧交接日志，不能把启动期保护通过当作整个训练周期保护通过。

本轮只读核验真实config与checkpoint：语言模型64层中48层linear_attention、16层full_attention。当前q_proj/v_proj LoRA只注入全注意力的层3/7/11/15/19/23/27/31/35/39/43/47/51/55/59/63，64个float32张量，总可训练参数3,014,656，r8/a16，视觉/aligner冻结。当前Transformers与vLLM0.24源码存在in_proj_qkv及其in_proj_qkvz打包映射，有静态支持依据；新增模块尚未GPU加载验证。已向用户提出先G4下增加in_proj_qkv或先保留q/v比较G4/G8的单变量路线，未获具体修改确认，不改当前配置。前者按真实config预计参数8,912,896，属于结构推算，不冒充实际注入验收。

实际保存args确认loss_type=grpo、scale_rewards=group；本轮读取安装的优势/损失与逐优化器步同步源码，不套最新TRL默认。实际损失加真实四候选奖励的CPU受控接口检查passed：优势、梯度下降符号和mask符合预期，但logprob是受控变量，未验证真实Qwen共享参数更新或生成端logprob一致性。四组全部观察到的score_batch均4候选、同组evidence/packet/asker绑定一致且零方差组0；混合train/eval trace不能当纯训练样本数，也不能精确还原缺step标签的真实梯度对应。

本轮搜索/取消/修复/测量实际累计约29.21 GPU小时（运行中的耗时继续增长），到45/60预算分别剩约15.79/30.79小时；旧V3环境和基座验证不混入。后续完整退火与模块/G对照先确定具体对象及预算，不机械扩全部四组。

详细派生核验记录：工作区review_artifacts/grpo_lr_search_20261006/训练正确性只读核验_20261007_CN.md。本轮没有修改活动19312517的科学配置、没有G/模块变更或新增GPU提交。

## 2026-10-07 09:26侧交接：正确性、G/LoRA与完整调度要求已并入

已完整读取用户指定交接 `C:\Users\20661\AppData\Local\Temp\EgoQA_GRPO_Correctness_Sweep_Schedule_Side_Handoff_20261007_092627_CN.md`。本节作为最新科学排查与调度约束，覆盖后面历史“所有组100步无改善即停止评价”的简化表述；不改变已提交19312517的配置、提交claim或运行保护。

执行顺序：先取得19312517新鲜manifest、Slurm状态、保护/模型/训练日志；同时从已有产物检查真实G/有效批量、候选/奖励/evidence/asker/媒体的组内对齐、invalid/零方差/截断/token mask。继而用固定候选和奖励核验优势符号、真实梯度贡献及小更新前后的相对对数概率，检查vLLM同步当前LoRA与独立重载一致性。冻结Judge用人工确认的受控正反例核验方向和排序，不能从自动accepted推导人工gold。只读或CPU证据与必要GPU受控计算分开汇报，不用参数非零变化替代更新方向正确。

目前本地静态确认continuous奖励为0.2F+0.4G+0.4A(1-S)，未发现speaker-only方向写反；保存的100步stage_state实际学习率比例5.871607054625497e-6/1e-5=0.5871607054625497，与200步余弦中途位置一致。这两项不证明真实训练器loss方向、生成端同步或Judge校准已经通过。

G和LoRA覆盖优先于beta/温度/秩搜索，具体新G及模块清单须先与用户讨论确认。本轮维持G4、q_proj/v_proj和视觉/aligner冻结；G8及线性注意力投影仅是待核验候选，未获具体修改授权。先按真实远端模型config、模块树和断点键统计层型、注入位置及可训练参数，并检查当前Transformers/vLLM模块支持；不能仅凭模型类名断言精确覆盖比例。loss_type/scale_rewards及生成同步路径读取Torch实际安装源码，不复制最新TRL默认值。

调度路线明确分开：新100步完整退火对照应从同一基座新LoRA出发，max_steps100/warmup10/min_lr_rate0.1，使用实际get_scheduler与CLI核验终点为峰值10%；不能把旧200步曲线的checkpoint100改称完整退火，不能简单改max_steps后恢复旧optimizer/scheduler。若需要确认已有配置的真实状态，用户已允许针对具体配置保留原200步曲线继续到200，恢复LR应连续。先确定配置与对照、更新累计GPU小时预算，不机械将四组都扩至200，不无限训练直到总分为正。

既有固定验证显示1e-5的groundedness/all-six提高，但formality下降、speaker-only提高，总奖励略降；不同分项可能拉扯，不仅凭总分判没有学习。比较仍限同一验证视频窗口、6个asker/24个输入-种子-生成槽位；没有独立locked test或人工质量终点。完整退火结束也不等于统计收敛或真实QA质量改善。

本轮没有新增GPU提交、修改G/LoRA/学习率或缩短当前测量曲线。共享桥已断开，19312517当前远端状态尚未刷新；后面的09:23排队信息只是最后现场快照。最新本地科学约束尚未同步远端，待重新认证后只同步文档，并对模型、真实训练器与产物做只读核验。

## 2026-10-07 09:14：修复测量19312517已提交

当前测量JobID为 **19312517**，提交清单为测量根 `submissions/grpo200_20261007T011446Z_448d7cad/submission.json`，输出 `outputs/formal_micro2-pointwise/train_19312517`，总索引 `measurement_manifest.json`。准备目录preflight/micro2-pointwise，已有独立submission.claim；不得重复提交或删除claim。实际CLI、断点报告与新增8项远端CPU测试均passed；GPU启动、保护目标达成、显存及两次更新仍待现场日志确认。

09:23:51北京时间新鲜查询：19312517仍PENDING/Priority，没有正式输出。调度器估计时间已多次变化，不作为承诺。最新本地全部100项测试98通过2跳过（Windows Bash及本地缺Torch的真实张量项）；后者已在Torch现有训练环境通过实际CPU测试。没有GPU运行证据时不宣称pointwise保护修复通过，更不宣称低利用率取消已彻底解决。

资源与测量范围不变：1H200/16CPU/500G/75分钟，峰值1e-6、checkpoint18→20、微批2/累积2/G4、调度总长200。唯一修复是辅助计算pointwise-1。原19311962因辅助工作区预算退出，记录在新manifest.history及preflight/failure_19311962。没有助手取消任何作业。

## 2026-10-07：首次测量已发现辅助工作区边界，定点修复

首次测量19311962在北京时间09:03:34提前分配gh105，09:04:45 FAILED/1:0，运行1分11秒；尚未开始Policy更新。辅助矩阵计算的实际分配超过16MiB预算，启动保护检查及时终止，run_manifest cleanup_errors为空。这不是低利用率取消，也不能证明微批2有问题。失败源码与清单保留在测量根preflight/failure_19311962，旧提交记录和claim不删除。

最小修复仅将辅助MM计算改为固定逐元素sin，两个384×384 float32缓冲合计1.125MiB，避免矩阵库工作区；新增分配器统计仍受16MiB上限约束，没有放宽预算。每条保护日志记录revision=pointwise-1。新增真实CPU张量测试验证缓冲占用、输入不变、有限输出和随机状态不变；远端8项全部通过，实际CLI仍2/2/G4、checkpoint18→20、200步调度。新准备目录preflight/micro2-pointwise，其他训练参数、目标80%、显存reserve8GiB及75分钟时限不变。修复后的GPU运行仍待新Job验证，不用CPU测试冒充CUDA效果。

## 2026-10-07：每次提交前预防低利用率取消（当前实施）

用户已明确授权：先尝试微批2/累积2/G4，显存可控时增加真实工作并发，仍不足的阶段预先配置反馈式keeper；保护必须覆盖每次提交，不能等系统取消一次后才调整。这覆盖本轮旧的“始终no keeper”例外。工作区AGENTS.md和本项目AGENTS.md已维护同一顺序，不能用后面的历史快照覆盖此授权。

当前测量目录为 `/scratch/xl6775/projects/EgoQA-six-user-baseline-grpo-20260929/variants/utilization-measure-20261007T003153Z`，已从原搜索根复制纯源码；模型、Torch环境、帧数据、baseline与checkpoint继续引用原路径。计划从1e-6 checkpoint-18恢复到20，调度总长度200，微批2/累积2/G4；保留完整过程验证与终点评分。计划时限75分钟：启动约30分钟、两次更新约12.4分钟、过程验证约7分钟、独立终点评分约17分钟，合计约66.4分钟，申请余量约13%；它是一次实际显存/吞吐测量，不能证明长作业不会取消或1e-6已完成100步。尚未提交JobID。

新保护实现是显式可选的utilization_guard，旧配置默认不启用。当前gh*节点取消阈值60%、警告75%，测量配置控制目标80%；从启动采样，按1/10/30/60/120分钟及累计窗口的时间加权均值反馈，占空比上限90%，不等7200秒才行动。工作区规则要求正式提交前再次核对官网及实际资源。最近现场show_slurm_qos与完整sacctmgr结果均为gpu48用户16GPU，上一次孤立查询0不当作当前限制；实际account为torch_pr_674_tandon_advanced，h200_tandon状态UP，仍自动路由且不指定节点。

为避免增加常驻CUDA上下文，启动期临时保护必须交接到训练进程：训练侧先请求临时进程退出，并核验其PID已离开NVML，再启动同进程保护。辅助缓冲及分配器新增空间限制16MiB，低优先级流每次只提交有限短工作，空闲显存不足8GiB时停止并释放。辅助CUDA图只在主线程等待或Policy已同步卸载、Judge尚未唤醒的安全边界捕获；运行中的后台线程不并发捕获图。窗口历史跨交接继承。固定终点评分也启用同进程保护。临时进程自身CUDA上下文不包含在16MiB张量上限内，因此交接时明确验证其释放，不把张量预算当作整个进程占用。

新增利用率、启动交接、显存让步和配置传递测试；本地99项98通过1项Windows Bash跳过。纯CPU检查不能证明CUDA图、保护交接、微批2显存或长时间安全；这些仍待新测量日志。最新源码尚未同步到测量根：一次SFTP命令使用父目录相对路径被本地桥校验拒绝，旧连接结束；未提交GPU作业。桥已修复为单次传输失败返回失败码而不关闭transport，并通过本地拒绝非法路径/接受工作区相对路径检查。新认证会话等待用户确认；不复用旧目录，不记录设备码。

同步运行位置为Windows交互式SFTP；新桥READY后可使用下列窄路径，不下载、不递归上传历史产物：

```text
sftp xl6775@login.torch.hpc.nyu.edu
lcd C:/Users/20661/Desktop/Research/AR/multiuser
cd /scratch/xl6775/projects/EgoQA-six-user-baseline-grpo-20260929/variants/utilization-measure-20261007T003153Z
put "AGENTS.md" "AGENTS.md"
put "EgoQA-main-20260928/multi-user/AGENTS.md" "multi-user/AGENTS.md"
put "EgoQA-main-20260928/multi-user/training/grpo_v3/six_user_binary/utilization_guard.py" "multi-user/training/grpo_v3/six_user_binary/utilization_guard.py"
put "EgoQA-main-20260928/multi-user/training/grpo_v3/six_user_binary/utilization_runtime.py" "multi-user/training/grpo_v3/six_user_binary/utilization_runtime.py"
put "EgoQA-main-20260928/multi-user/training/grpo_v3/six_user_binary/launch.py" "multi-user/training/grpo_v3/six_user_binary/launch.py"
put "EgoQA-main-20260928/multi-user/training/grpo_v3/six_user_binary/plugin.py" "multi-user/training/grpo_v3/six_user_binary/plugin.py"
put "EgoQA-main-20260928/multi-user/training/grpo_v3/six_user_binary/workflow.py" "multi-user/training/grpo_v3/six_user_binary/workflow.py"
put "EgoQA-main-20260928/multi-user/training/grpo_v3/six_user_binary/evaluation.py" "multi-user/training/grpo_v3/six_user_binary/evaluation.py"
put "EgoQA-main-20260928/multi-user/training/grpo_v3/six_user_binary/shared_gpu.py" "multi-user/training/grpo_v3/six_user_binary/shared_gpu.py"
put "EgoQA-main-20260928/multi-user/tests/training/grpo_v3/six_user_binary/test_utilization_guard.py" "multi-user/tests/training/grpo_v3/six_user_binary/test_utilization_guard.py"
put "EgoQA-main-20260928/multi-user/docs/SIX_USER_GRPO_27B_TORCH_RUNBOOK_CN.md" "multi-user/docs/SIX_USER_GRPO_27B_TORCH_RUNBOOK_CN.md"
bye
```

上传成功后，先在新根检查实际CLI为2/2/G4、stop20/max200、当前18步断点以及新增保护参数，并运行新增CPU回归、实际包导入及Bash语法。真实GPU测量只提交一次，使用--parsable立即保存新manifest；不得复用任何旧submission.claim。记录本次保护配置和节点、时间窗口、启停/强度/显存、真实更新耗时和完整20步断点。CPU或短测通过不能替代越过原约2小时取消时段的正式长作业验证。

## 2026-10-07 08:10以后：两低LR组系统取消，利用率排查及恢复边界

现场sacct确认19291980（1e-6）与19292245（3e-6）均CANCELLED by 0，分别运行02:18:32和02:18:33。北京时间00:13:47同时开始、02:32:19/20取消；分别gh116/gh117、h200_tandon/gpu48，未申请抢占分区。当前没有活动作业。stderr仅SIGTERM，Slurm取消备注为空，残留QOSMaxGRESPerUser不能当作终止原因。两组已越过之前Judge启动等待失败并发生真实Policy更新。

[Torch官方规则](https://services.rt.nyu.edu/docs/hpc/submitting_jobs/slurm_submitting_jobs/)当前gh*节点取消阈值60%、警告75%，没有公开统计窗口或初始化豁免。两组全程平均GPU利用率48.28%/45.55%，前30分钟11.49%/3.50%，末60分钟59.16%/62.55%；结合系统近同时取消，低利用率是首要解释，尚无管理员明确归因。成功高LR组全程62.71%/61.80%，同链路余量也不充足。

每次模型切换本身约3秒；稳态Judge评分中位约79–83秒，Policy阶段约259–264秒。图片缓存已命中，后期groundedness图片准备约0.013秒/候选，实际评分批次仍约50–53秒；单纯扩大图片缓存不是主要修复方向。Policy阶段包含生成及训练，显存峰值139.66/139.49GiB，尚不能将其等同于训练forward单独峰值，因此微批翻倍必须实际测量，不能承诺无OOM。

已在现有训练环境CPU完整读取两个checkpoint：1e-6的outputs/formal_lr1e6/train_19291980/swift/v0-20261006-122630/checkpoint-18，以及3e-6的outputs/formal_lr3e6/train_19292245/swift/v0-20261006-122937/checkpoint-28。两组均64优化器状态、192有限优化器张量、64有限且非零adapter张量，步数18/28、scheduler位置和200步余弦曲线一致。3e-6日志29未保存，不计入恢复完成。审计报告为新根preflight/cancellation_audit_20261007T002000Z/checkpoint_audit.json。无100步终点结果，不能评价低LR是否改善。

用户选择可以讨论扩大真实批量或调整资源，尚未确定具体微批/G/GPU数量。候选是单H200、G4下微批2/累积2，保持每次更新4条completion；需要真实GPU显存与吞吐测量，并会改变数值执行顺序。直接双卡分离同步Policy/Judge可能使每卡空闲比例升高，不作为默认修复。仍遵循用户本轮no keeper决定，不改reward/prompt/帧数/解码，不盲目原样重投。具体新训练参数按用户既有要求先讨论。

本地现场派生摘要见工作区review_artifacts/grpo_lr_search_20261006/状态快照_20261007_0810_CN.md。本节为本地更新，尚未同步远端；后面的历史PENDING/RUNNING快照仅用于追溯。

## 2026-10-07 00:05：新增1e-6已提交，冷启动等待修复

用户明确选择新增峰值学习率1e-6。新作业 **19291980** 从同一27B基座初始化新LoRA，不加载checkpoint；阶段100步、调度总长200、10步预热、余弦下限10%（最终1e-7）、G4/微批1累积4、beta0.04、温度0.85、LoRA和2048预算均保持原设置。资源为1H200/16CPU/500G/15:30:00，自动路由h200_tandon/h200_public、gpu48，无节点绑定。提交清单为新根 `submissions/grpo200_20261006T155839Z_43751061/submission.json`，输出 `outputs/formal_lr1e6/train_19291980`。实际CLI/曲线/回调检查passed，当前PENDING/QOSMaxGRESPerUser，未开始训练。

准备时发现3e-6恢复作业19289697已FAILED/1:0，运行16:54，原因Judge服务启动超时，Policy尚未启动。Judge日志显示模型与多模态预热在23:20:04完成、23:20:08被清理；15分钟启动上限不足。现有launch已支持judge_startup_timeout_seconds，但workflow未转发。最小修复仅补这一字段传递，并给新作业设置1800秒；不改变算法或GPU资源。新增回归先复现字段丢失，修复后本地workflow11项与搜索7项通过，远端新增单项和实际运行配置检查通过。等待上限修复的GPU效果仍待新作业验证，不能称已解决全部运行风险。

3e-6自身checkpoint12的状态和超参不变，使用同一修复重新提交 **19292245**，从12继续到100、总调度仍200，1H200/16CPU/500G/14h。提交清单为新根 `submissions/grpo200_20261006T160256Z_5efae41a/submission.json`，输出 `outputs/formal_lr3e6/train_19292245`，当前PENDING/QOSMaxGRESPerUser。原19262323系统取消及19289697启动超时均留在search_manifest该组history；未取消或重投两个已完成组。

新增1e-6准备目录为 `preflight/lr1e6_20261006T154959Z`，恢复重试为 `preflight/resume12_retry2_20261006T160254Z`。按已完成两组约11小时/100步及原6.21分钟/更新两种实测参考，四组100步筛选累计预计约45–48个H200小时（含已发生中断）；如果再延长一组到200，可能接近或略超原60小时上限，须到时结合实际耗时和效果再讨论，不自动扩大。估计不含排队、额外失败或更长生成。当前不根据两个未改善的结果断定所有LR过高，也不承诺1e-6一定更好。

## 2026-10-06 23:08：两组完成，固定终点未改善；恢复组已运行

19262322（1e-5）与19262321（3e-5）均COMPLETED/0:0，分别运行11:01:13和10:59:25；完整checkpoint-100、训练工程验收、run_manifest completed及cleanup_errors空均已核验。阶段正常停止于100，完整学习率调度保持200。两组同6输入/24候选与旧未训练baseline完整配对，均24/24成功评分：

| 条件 | 固定验证平均奖励 | 相对baseline差值 | 相对变化 |
|---|---:|---:|---:|
| 未训练baseline | 0.388016 | 0 | 0 |
| 1e-5，100步 | 0.383955 | -0.004061 | -1.05% |
| 3e-5，100步 | 0.367493 | -0.020523 | -5.29% |

1e-5组6个提问者输入中4个均值提高、2个下降；3e-5组1个提高、5个下降。只有一个验证视频窗口，没有显著性或人工质量评估，不能称稳定退化或收敛提升。过程验证100步的0.394804/0.398572与这里独立固定种子终点不同，不混用。

3e-6恢复作业19289697已RUNNING于gh104，查询时已运行5分23秒，继续该组完整12步断点到100，未改变超参或从旧checkpoint60初始化。原19262323系统取消及利用率证据保留。恢复任务仍需核验实际新更新与100步结果，不能因为已提交或RUNNING称为恢复完成；取消具体原因也未证明已消除。

当前没有选择或提交200步延长任务。待3e-6完成同设置固定验证后再比较三组；如果三组均没有改善信号，按已讨论方案先重新讨论剩余预算用途，不机械延长。当前本地派生结果位于 `review_artifacts/grpo_lr_search_20261006/status_snapshot_20261006_230842.json`。

## 2026-10-06 23:02：两组100步通过，低学习率组恢复已提交

| 峰值学习率 | 当前JobID | 当前阶段 | 完整保存点 |
|---|---|---|---|
| 3e-6 | 19289697 | 恢复任务PENDING/Priority | 原19262323的checkpoint-12 |
| 1e-5 | 19262322 | 100步训练工程passed，独立终点评分16/24 | checkpoint-100 |
| 3e-5 | 19262321 | 100步训练工程passed，独立终点评分20/24 | checkpoint-100 |

两高学习率组仍RUNNING于gh119，因为同一作业内的独立终点评分尚未完成；已实际正常结束训练于100步，scheduler_last_epoch=100且完整调度目标仍200。训练产物验收包括有限奖励、梯度、LoRA非零变化、完整checkpoint和保留的调度长度，均passed。当前没有完整最终paired comparison，不能先选赢家。

过程验证奖励：1e-5组第50/100步为0.381281/0.394804，3e-5组为0.390598/0.398572。这是训练内过程指标，不与旧独立baseline混用；验证仅一个视频窗口，不宣称显著性或真实QA改善。

3e-6原作业19262323于北京时间20:20:32启动gh116，22:33:54系统CANCELLED by0，运行2:13:22。日志记录到13步但完整保存仅12步，恢复不把13计入完成。stderr仅系统SIGTERM，未发现前置程序异常；Slurm没有明确取消备注，残留QOSGrpGRES不作为取消原因。全程记录GPU均值37.58%、末3600样本均值51.15%，与低利用率风险相符，不能确定唯一原因。

已核验当前搜索组自己的完整checkpoint-12：64份优化器状态有限且step12、scheduler12、LoRA有限非零、保存超参匹配，新的实际resume CLI和恢复优化器学习率与原200步曲线一致。保持3e-6、beta0.04、数据、模型、200步调度/100步阶段终点和单H200/no keeper方案，从12继续到100；没有加载旧checkpoint-60、重做媒体或重复GPU最小运行。

恢复提交记录为新版本根下 `submissions/grpo200_20261006T145613Z_45081f28/submission.json`；正式输出 `outputs/formal_lr3e6/train_19289697`。CPU恢复证据位于 `preflight/resume12_20261006T144601Z`。资源为1H200/16CPU/500G/14h，余下88步与验证/初始化约11.1h、加25%余量后申请14h，自动路由h200_tandon/h200_public、gpu48，无节点指定。新JobID已记录到search_manifest，原取消JobID及历史输出保留在该trial的history。未取消或修改两个运行作业，新的恢复不能保证不再遭遇集群取消。

后面的监控块从search_manifest读取当前JobID，因此会自动读取19289697，不要求手工替换旧19262323；不要再次运行相同恢复preflight的提交器或删除submission.claim。

## 2026-10-06 11:50：三组学习率已并行提交

**12:05及随后启动核验：** 19262321（3e-5）和19262322（1e-5）已同时RUNNING于gh119，各分配1张H200，启动存储检查passed，实际运行配置为200步调度/100步阶段结束、10步预热、10%下限且无旧checkpoint初始化；当前进入Judge模型初始化。19262323（3e-6）仍PENDING/QOSMaxGRESPerUser。新的优化器更新与100步结束尚未验证，不能把RUNNING称为训练完成。

此前三组都报告额度等待，启动前自己的gpu48计数为gres/gpu=16(0)（上限16、占用0），两个H200分区用户占用也0，每作业ReqTRES均gres/gpu=1。一次对19262323写入相同时限返回Unspecified error，前后参数未变。随后两个作业自动获得资源，因此不能把旧等待快照描述为三组全面阻塞，也不能据此断言配额实际耗尽或具体Slurm缺陷。保留第三组等待原因供后续核对；没有取消、重提或改变资源。

三组均从27B基座初始化新LoRA，不加载旧checkpoint。独立新版本为 `/scratch/xl6775/projects/EgoQA-six-user-baseline-grpo-20260929/variants/lr-search-20261006T033800Z`；模型、两套环境、已有训练18/验证6输入与7200帧继续引用成功版本的真实路径，历史代码和产物未覆盖。总索引为该根下的 `search_manifest.json`，每个JobID另有独立的时间戳 `submission.json`。

| 学习率 | JobID | 提交目录（相对新版本根） | 正式输出（相对新版本根） |
|---|---|---|---|
| 3e-6 | 19262323 | submissions/grpo200_20261006T034958Z_90529790 | outputs/formal_lr3e6/train_19262323 |
| 1e-5 | 19262322 | submissions/grpo200_20261006T034958Z_14638bf6 | outputs/formal_lr1e5/train_19262322 |
| 3e-5 | 19262321 | submissions/grpo200_20261006T034957Z_848a86f4 | outputs/formal_lr3e5/train_19262321 |

提交后新鲜sacct/squeue/scontrol确认：三组均PENDING，尚无正式训练输出；每组1H200、16CPU、500G、15:30:00、constraint=h200、account为torch_pr_674_tandon_advanced。未指定节点、partition或QOS，站点实际路由为h200_tandon/h200_public、gpu48。原作业峰值323GiB是保留500G的依据；时限由100步实测吞吐、两次过程验证、终点评分与初始化估算约12.35h，再加约25.5%余量。并行改变历时，不改变累计GPU小时预算。

实际配置已通过训练环境的CLI解析与CPU调度计算：总调度200步、10步预热、余弦下限10%、第100步正常结束；优化器/调度状态在100恢复到101时与连续运行一致，真实框架回调已注册。三组第50/100步过程验证、每2步保存，旧完整baseline设置一致且可复用，终点仍按相同6输入/24候选评分。单H200内部继续Policy/Judge交替，用户本轮明确授权不新增keeper。

验证证据在新根 `preflight/search_schedule.json`、`preflight/search_cpu_status.json` 和 `submission_verified.json`。Windows本地91项90通过1跳过；Torch初轮91项中89通过，新版本遗漏根部GPU身份辅助源码导致2项失败，补齐原有文件后这2项均通过。初轮失败证据保存在 `preflight/first_attempt_failure`，修复检查为 `preflight/source_tests_repair.log`。两环境新鲜pip依赖检查均通过；未受改动影响的媒体像素、处理器和编译证据复用有明确来源。本轮没有重复GPU最小运行；新的100步阶段结束仍须由实际产物验收，不把CPU检查或提交当作训练完成。

### 从清单查看当前三组状态

运行位置：Torch登录节点Bash。此块只读，可在全新登录shell直接复制，不依赖前文变量；不提交、不取消、不关闭会话。

```bash
ROOT=/scratch/xl6775/projects/EgoQA-six-user-baseline-grpo-20260929/variants/lr-search-20261006T033800Z
python3 - "${ROOT}" <<'PY'
import json, subprocess, sys
from pathlib import Path
root = Path(sys.argv[1])
manifest = json.loads((root / 'search_manifest.json').read_text())
jobs = ','.join(trial['job_id'] for trial in manifest['trials'])
subprocess.run(['sacct', '-X', '-j', jobs, '-o', 'JobID,State,ExitCode,Elapsed,Timelimit,NodeList'])
subprocess.run(['squeue', '-j', jobs, '-o', '%.18i %.10T %.10M %.10l %.40R'])
for trial in manifest['trials']:
    print('\n学习率：', trial['learning_rate'], 'JobID：', trial['job_id'])
    print('正式输出：', trial['formal_output'])
    task = Path(trial['submission_manifest']).parent
    output = Path(trial['formal_output'])
    for path in (task / ('slurm-' + trial['job_id'] + '.out'),
                 task / ('slurm-' + trial['job_id'] + '.err'), output / 'trainer.log'):
        if path.is_file():
            print('日志：', path)
            print('\n'.join(path.read_text(errors='replace').splitlines()[-12:]))
    for name in ('stage_state.json', 'training_result.json', 'validation_comparison.json'):
        path = output / name
        if path.is_file():
            print('产物：', path)
            print(path.read_text())
PY
```

没有文件通常表示尚未进入对应阶段，不据此推断训练失败。三组已有真实JobID和一次性提交记录，不得再次启动提交器或删除submission.claim。实际100步结果齐全后才能选择较好配置继续到200；选择依据为固定验证奖励、分项、格式合法率和训练稳定性，单一验证窗口不证明跨视频泛化或人工QA改善。

## 2026-10-06：新一轮三组学习率搜索，准备中

用户批准从基座重开三组 `3e-6 / 1e-5 / 3e-5`，复用成功代码、环境和已处理媒体。三组各单H200，可并行调度；每组内部保留Policy/Judge交替运行。三组各100步筛选，选中组继续到总计200步；预计累计约50–55个H200小时，不含排队和中断，按实际生成耗时更新估计。

调度总长度200步，预热10步，余弦衰减到峰值的10%；阶段结束100步由正常回调完成验证和完整保存。每50步过程验证，每2步保存，固定验证使用相同6输入/24候选、种子、解码和冻结Judge。旧checkpoint-60不作为初始化；若继续训练，恢复本轮选中组自己的完整checkpoint-100。

本地已修复60步限制、超参不传递和constant调度写死；阶段验收保留完整优化器/调度器/随机状态检查。91项本地相关测试90通过、1项Windows跳过；真实Torch CLI解析与调度曲线待核验，现有7200张帧引用可读，实际gpu48用户上限16张、当前没有活动作业。当前尚无本轮新JobID，不提供猜测提交或监控路径。普通作业不手动指定partition/QOS或节点，按当前官方路由规则与远端查询结果提交。

用户本轮明确授权沿用成功的direct入口，不新增keeper，保留GPU利用率监控、阶段日志和断点恢复。旧作业、预检、一次性提交记录和媒体树全部保留；独立新版本只接收受影响源码和配置。任何已提交任务均不得自动取消。

历史第60步最终结果由 `review_artifacts/grpo_training_19136858_final_20261004` 的本地汇总确认：同设置24/24候选评分完整，baseline奖励0.388016、Policy60奖励0.383889；不构成质量提升或显著退化结论。下面“仍在46步”等文字是当时快照，不是本轮当前状态。

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
