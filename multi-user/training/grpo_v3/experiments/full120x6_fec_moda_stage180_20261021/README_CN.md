# F/E/C与固定历史池diversity：最新GRPO源码快照

本目录是`lxr-grpo`分支上最新的六用户GRPO运行源码，接续原`codex/grpo-ftjudge-dual-speed-v5-20261016`的V5归档。V5及其他实验目录保留其历史行为；本次新增实现位于本目录`runtime/`，不应直接使用旧实验的提交器作为当前入口。

当前真实运行来源为正式JobID`20573337`。它复用了`20565267`通过的最长输入完整测量，然后从同基座重新初始化LoRA训练至180步，调度horizon仍为360步。该代码归档和CPU回归不等于180步最终验收完成，也不证明人工QA质量或泛化提升。

## 奖励及历史参照

质量项为

\[
R_{quality}=0.2F+0.4E+0.4C
\]

其中F为Formality，E为Groundedness，C为六视角联合Answerability；不再使用旧A乘以一减speaker-only分数的奖励。

仅编码每个候选的`question`，使用冻结CPU MiniLM得到归一化向量。在同一窗口内，参照为当前合法G4中的其他候选，加固定历史池；历史题不加入GRPO生成组或优势分母。非法completion不成为参照，也不给多样性奖励。质量未通过的合法当前候选仍作为其他候选的去重参照。

\[
d_i=\min_{j\in\mathcal R_i}\frac{1-\cos(q_i,q_j)}{2},\qquad
g_i=\mathbf 1[F_i\ge0.5\land E_i\ge0.5\land C_i\ge0.5]
\]

\[
R_i=R_{quality,i}+0.5g_id_i
\]

没有参照时距离和bonus为零；同题或近邻题会降低距离。当前G4及历史记录按真实ID绑定，不能用返回顺序猜候选身份。

历史池直接使用完整旧300步实验的旧分数筛选，不用当前Judge重评分：旧总reward≥0.4、旧F≥0.5、旧E≥0.4、旧A≥0.5且旧S<0.5，seed42，每窗最多8题、不凑满。当前来源共251题、覆盖120窗，其中部分窗口历史池为空；251是该实际来源的结果，不是所有新数据都固定为251题。

## 运行设置及布局

- 原720条件顺序、G4、两rank各4槽位/更新保持；阶段180覆盖前360条件、1440原始槽位，后181–360没有自动启动授权。
- Policy为同基座fresh LoRA，r64/alpha128/dropout0.05，q_proj/v_proj/in_proj_qkv；视觉与aligner冻结。每rank微批1、累积4，全部160个可训练张量在裁剪和优化器前做梯度均值。
- 50176像素、0.5FPS、completion上限2048、上下文131072；seed/data_seed42，峰值学习率1e-5、cosine_with_min_lr、warmup10、min_lr_rate0.1；stop_after_steps180、max_steps360。
- 单节点2H200、32CPU、750G；Policy与Judge同卡交替，不以低计算活动率推断显存可容纳两个同时驻留模型。
- 全64语言层原生gradient checkpoint输入CPU暂存，保留原checkpoint/RNG；每次forward检查包装归属，避免Swift临时关闭/开启checkpoint覆盖包装。数值恢复按原dtype/device进行。
- 所有配置、模型、数据、环境、输出和scratch路径必须来自实际新环境。`examples/train_config_20573337.json`保留该来源的配置，包含私有scratch路径，不能不经核对直接提交。

`runtime/`沿用真实独立variant根布局：顶层为双卡、历史池和阶段/恢复辅助模块，`multi-user/training/grpo_v3/six_user_binary/`为实际F/E/C及diversity包源码，`prompts/`包含原样Judge模板，`hpc/shared/`为共享设备识别依赖。当前运行布局不使用旧`package/`暂存目录，归档取后续实际部署的完整包源码。

如需在新环境复用，先在新的独立根准备本仓库`multi-user`与`hpc`依赖，再按此`runtime/`相对结构覆盖运行源码，配置真实模型、三Judge adapter、数据及已准备的固定池，并以根目录和`multi-user`设置PYTHONPATH。正式训练入口为`dual_launch.py --config <已核验的新配置>`。不要修改正在运行的实验源码。

当前单独正式任务预算由用户批准为总75小时，计入先前测量51分08秒后正式请求266932秒，Slurm按分钟取整。`combined_allocation.py`、`submit_combined.py`及早期`submit_training.py`保留原60小时测量合并流程的来源代码，**不是当前75小时单独正式任务的提交指令**；不要重跑已通过的GPU测量或按历史脚本重复提交。

## 验证

在有NumPy和Pillow的Python环境，从仓库根运行：

```powershell
python -B multi-user/training/grpo_v3/experiments/full120x6_fec_moda_stage180_20261021/run_cpu_regressions.py
```

轻量回归隐藏CUDA，核对奖励公式、G4邻居/历史窗口、非法候选、去重、固定旧分数池、三Judge模板和身份对应、720条件/不同窗口配对、180停止与360调度、阶段保存/恢复及容量输入绑定；使用合成fixture，不加载真实模型或提交Slurm。

旧`test_moda_validation.py`的fixture在本次归档中补充了逐rank评分轨迹，以满足后来实际训练代码新增的严格轨迹验收；训练实现及验收条件未因此放宽。

`runtime/test_checkpoint_input_offload.py`、`test_stage_native_state.py`为需要真实Torch等依赖的独立CPU测试，未包含在上述轻量入口。当前来源此前已有CPU作业`20564755`七项原生等价性验证、CPU作业`20442505`八项输入与恢复验证，以及`20565267`真实GPU完整循环/保存/CPU重载；这些历史证据与本次本地回归分开，不能当作在新环境重新验证通过。

训练验收仍须核顶层和batch终态、180真实更新/360条件/1440槽位、完整CP180及独立CPU重载；包括optimizer160状态、原调度、两rank RNG与数据位置。固定Judge代理reward和通过率不等于人工正确率。

## 归档范围

本目录不包含模型权重、checkpoint、视频/帧、QA原始输出、完整训练日志、认证文件、设备码、桥接程序或活动监控状态。`source_snapshot.json`记录来源及源码清单，`condition_definition.json`只保留实验条件定义。新增GitHub代码不授权提交、取消或改变任何Torch作业。
