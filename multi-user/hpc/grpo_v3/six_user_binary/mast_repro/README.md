# 用 ms-swift 框架复现 MAST GRPO run

目标：`judge_grpo_r64_L16_lr2e-5_p4g4_pipe_s900`（MAST，`mast_trainer.py`）。评估 reward（3 packet × 6 提问者 × 4 候选）：
第 0 步 0.423 ± 0.013，第 100 步 0.452 ± 0.014，第 200 步 0.486 ± 0.015，第 300 步 0.491 ± 0.015。

训练框架（ms-swift GRPOTrainer、`launch.py` 的作业驱动、reward、judge 视角）不变；下面的开关默认关闭，
不设置时原有实验的行为与之前完全相同。

## 文件

| 文件 | 内容 |
|---|---|
| `selection.json` | 8 个训练 packet × 6 提问者 = 48 行；3 个验证 packet × 6 提问者 = 18 行 |
| `judge_config.json` | 冻结 Qwen3.8-27B（不加 adapter）；`frame_stride 2`、`max_pixels 119808`，与 judge SFT 的 `collator_025fps_mb2` 一致 |
| `run_config.template.json` | 训练配置；`__XXX__` 为需要替换的本机路径 |

## 与 MAST 的对齐方式

| 项目 | 设置 | 实现 |
|---|---|---|
| policy 输入 | 6 人完整时间线隔帧，6 × 150 = 900 帧，提问者在前，不剪枝 | 数据行 `generator_view: full_025fps`（`data.full_025fps_view`，同 `mast_inputs.policy_row`） |
| policy prompt 元数据 | 每人 `model_input_frame_count 150`、`effective_model_input_fps 0.25`；packet 级 `model_input_frame_count 900`、`per_user_model_input_frame_counts` 6 × 150；`source_*` 保留 1800 / 300 | 同上；`prompts.py` 写死的说明文字（"one-per-second"、"survived pruning"）不改 |
| 简洁输出要求 | `compact_full_qa_v1` | 数据行 `generation_profile` |
| policy 每张图像素 | 65536 | `max_pixels` |
| judge 输入 | 每个 frame_set 隔帧；每张图上限按原 0.5 FPS 帧数 `adaptive_image_max_pixels(..., 119808)`；可见文本 `_prompt_025fps` | `judge_config.frame_stride 2`、`max_pixels 119808` |
| LoRA | r64、α64、dropout 0；层 48–63 的全部线性层，共 124 个模块 | `lora_target_regex`、`lora_dropout` |
| 优化 | lr 2e-5 常数、warmup 0、weight decay 0、ε 0.2、loss `grpo` | `learning_rate`、`lr_scheduler_type`、`warmup_steps`、`swift_overrides` |
| 每步候选 | 4 个 prompt × 4 = 16：每卡 batch 1、GA 16、单 policy GPU | `per_device_train_batch_size`、`gradient_accumulation_steps` |
| 长度 | 生成 2048；prompt 上限与 vLLM 上下文都是 71680（≥ 69632 + 2048） | `max_completion_length`、`max_length` |
| 生成端 LoRA | 每步把 LoRA 合并进权重后同步给 vLLM（与 MAST 相同；vLLM 0.24 对线性注意力层 LoRA 的支持未验证） | `swift_overrides.vllm_enable_lora false` |
| 其余 | β 0.04、T 0.85 / top_p 0.95 / top_k 40、seed 42、900 步、每 100 步评估、每 50 步保存、评估每 prompt 4 候选 | 原有配置项 |

ms-swift 4.2.2 中 `loss_type` 默认 `grpo`、`epsilon` 默认 0.2，与 MAST 相同，配置里仍显式写出；`weight_decay` 默认 0.1，必须设为 0。

## 步骤

1. 环境：按 `requirements/grpo_v3_ms_swift.txt` 建 ms-swift 4.2.2 训练环境，另备 judge 用的 vLLM 环境。
2. 生成数据（`dataset_root` 下需有 `packets/<packet_id>/packet.json` 与帧；不需要 CLIP keep mask）：

   ```bash
   cd <project_root>/multi-user
   python -m training.grpo_v3.six_user_binary.data \
     --selection hpc/grpo_v3/six_user_binary/mast_repro/selection.json \
     --dataset-root <dataset_root> --output-dir <data_dir> \
     --generator-view full_025fps --generation-profile compact_full_qa_v1
   ```

   输出 `train.jsonl`（48 行）和 `validation.jsonl`（18 行）。启动时 `launch.py` 会再用同样方式重建并逐行比对。
3. 把 `run_config.template.json` 和 `judge_config.json` 里的 `__XXX__` 换成本机路径，用 `hpc/grpo_v3/six_user_binary/train_direct.sbatch` 提交（2 张 GPU：policy 0、judge 1）。

## 正式训练前需要确认

- **显存**：单张 policy GPU 上要容纳 27B bf16、约 6.6 万 token 的前向/反向、r64 LoRA，以及休眠的 colocate vLLM（上下文 71680）。这一长度从未验证过，先跑 1–2 步测峰值显存与每步耗时。
- **每张图的视觉 token 数**：应为 64（65536 px），整条 prompt 约 6.45 万 token。在测试运行里打印确认 ms-swift 的缩放与 MAST 的 `process_vision_info` 一致。
- **LoRA 模块数**：PEFT 可训练模块应为 124，vision tower 不应出现。

## 判断复现是否成功

看 18 个验证 prompt × 4 候选的平均 reward：第 0 步约 0.42（主要用于确认 policy 与 judge 输入对齐），第 100 / 200 步约 0.45 / 0.49，
相差在 ±0.015（一个标准误）以内算一致。第 0 步对不上时，先查 judge 输入，再查 policy 输入。

仍与 MAST 不同、预计影响很小的地方：ms-swift 同步训练（ratio 恒为 1），MAST 有 1 步流水线延迟；vLLM 版本、TP 与 LoRA 合并方式带来的数值差异。
