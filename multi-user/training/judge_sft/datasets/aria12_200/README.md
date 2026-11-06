# Aria12 联合可回答性训练

使用已批准的主题替换版200题。训练160题（40通过、120失败），测试40题（10通过、30失败）。两边共享同一场景，按事实和近义问题分组，无重复事实组。修订题使用批准的版本。详见 `split_summary.json`。

联合C通过要求提问窗口不足且12路联合证据支持唯一选项。训练文本包含问题及五个选项。第一组固定为提问窗，每张图前有窗口和相对时刻，剪辑切点单独标明。

从原始拼接成片按0.5 FPS抽帧，12路各117帧。准备工具 `python -m training.judge_sft.prepare_aria12` 接收数据目录、来源映射和输出根目录。输出根下的 `media/packet.json` 与 `train.jsonl`、`test.jsonl` 一起放到 Torch 项目的 `input/`。媒体和本机绝对路径保留在实验目录。

Torch 使用 Qwen3.8-27B 基座，双 H200 TP2，微批1，累积4，LoRA r128/α128，末16层，学习率2e-5，普通BCE。固定5轮共200步。每轮保存断点，最终保存 `final_adapter` 后在40题上评估一次。输出 `test_predictions.jsonl` 和 `test_metrics.json`。

现场核对 `JUDGE_PROJECT`、`JUDGE_REPO`、`JUDGE_ENV`、`JUDGE_MODEL` 后提交 `hpc/judge_sft/preflight_aria12.sbatch` 做正式处理器CPU检查。通过后，把该作业的 `processor_preflight.json` 路径作为 `JUDGE_PREFLIGHT`，按现场账户与时限提交 `hpc/judge_sft/train_aria12_answerability.sbatch`。产物按作业号保存在 `jobs/`。GPU监控记录时间、卡UUID、利用率和显存。
