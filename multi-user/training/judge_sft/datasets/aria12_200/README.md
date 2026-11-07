# Aria12 联合可回答性训练

使用已批准的主题替换版200题。训练160题（40通过、120失败），测试40题（10通过、30失败）。两边共享同一场景，按事实和近义问题分组，无重复事实组。修订题使用批准的版本。详见 `split_summary.json`。

联合C通过要求提问窗口不足且12路联合证据支持唯一选项。训练文本包含问题及五个选项。第一组固定为提问窗，每张图前有窗口和相对时刻，剪辑切点单独标明。

从原始拼接成片按0.5 FPS抽帧，12路各117帧。准备工具 `python -m training.judge_sft.prepare_aria12` 接收数据目录、来源映射和输出根目录。输出根下的 `media/packet.json` 与 `train.jsonl`、`test.jsonl` 一起放到 Torch 项目的 `input/`。媒体和本机绝对路径保留在实验目录。

Torch 使用 Qwen3.8-27B 基座，双 H200 TP2，微批1，累积4，LoRA r128/α128，末16层，学习率2e-5，普通BCE。固定5轮共200步。每轮保存断点，最终保存 `final_adapter` 后在40题上评估一次。输出 `test_predictions.jsonl` 和 `test_metrics.json`。

首个GPU作业22137615在数据加载阶段因临时套接字地址过长停留于0步，之后被系统取消。入口使用短路径 `/scratch/$USER/jt/$SLURM_JOB_ID` 指向本次作业缓存，存储检查仍核实实际路径归属。模型加载前通过 `check_worker_transport` 实际创建Unix套接字，并跨进程传递PyTorch张量。失败日志及原JobID保留。

22150875通过通信检查后，在首个反向传播中触发显存不足。堆栈定位于Qwen线性注意力的纯PyTorch回退实现。运行环境需有fla-core 0.6.0和einops 0.8.1，入口确认实际分发到FLA。登录节点可运行 `check_linear_backend` 做零GPU接口验证，正式作业在大模型加载前用小张量对照原实现检查前向与反向，再开始训练。

22155006通过FLA前向/反向对照后，在完整长序列MLP反向重计算中显存不足。入口对逐token独立的MLP按2048 token分块，并在块内使用非重入检查点。CPU回归验证嵌套检查点下的输出、输入梯度及所有参数梯度一致。全部帧和训练超参数继续沿用。

用户随后要求逐轮评测并对比旧C630，40题作为checkpoint选择评测集使用，启用GRPO的数值阈值待看对比结果后确定。独立 `watch_epoch_evaluations` 监视第40/80/120/160步完整checkpoint并安排单H200评测，当前训练的第200步继续由原入口完成评测。`evaluate_aria12` 对旧C630和新adapter使用相同40题、12窗顺序、模板、像素上限及0.5判定阈值。

Aria入口支持TP2或TP4。四卡吞吐检查使用 `--tensor-parallel-size 4 --benchmark-steps 2 --mlp-chunk-size 0`，保持微批1/累积4及全部输入，只执行两次更新，不读取评测分数或输出正式训练模型。正式4卡训练是否启动依据测得速度确定；当前2卡作业继续执行。

现场核对 `JUDGE_PROJECT`、`JUDGE_REPO`、`JUDGE_ENV`、`JUDGE_MODEL` 后提交 `hpc/judge_sft/preflight_aria12.sbatch` 做正式处理器CPU检查。通过后，把该作业的 `processor_preflight.json` 路径作为 `JUDGE_PREFLIGHT`，按现场账户与时限提交 `hpc/judge_sft/train_aria12_answerability.sbatch`。产物按作业号保存在 `jobs/`。GPU监控记录时间、卡UUID、利用率和显存。
