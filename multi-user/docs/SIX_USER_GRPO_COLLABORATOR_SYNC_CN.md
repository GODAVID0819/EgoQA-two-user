# 六用户GRPO协作同步：当前视频、配置和入口

本文依据2026-10-11对当前作业19530750提交清单、运行配置及实际train/validation JSONL的只读核对。运行进度由主线程监控；此处说明数据与代码，不从文档重复提交当前作业。

## 1. 当前使用的视频

视频来源为[官方EgoLife数据集](https://huggingface.co/datasets/lmms-lab/EgoLife)。每个窗口是六名用户的同步10分钟视频，用户为Jake、Alice、Tasha、Lucia、Katrina、Shure；原目录标识分别为A1_JAKE、A2_ALICE、A3_TASHA、A4_LUCIA、A5_KATRINA、A6_SHURE。

| 划分 | 窗口标识 | 数据集内时间范围 | 训练输入数 |
|---|---|---|---:|
| 训练 | DAY1_11400000 | DAY1 11:40–11:50 | 6 |
| 训练 | DAY1_17200000 | DAY1 17:20–17:30 | 6 |
| 训练 | DAY2_16000000 | DAY2 16:00–16:10 | 6 |
| 训练 | DAY2_16200000 | DAY2 16:20–16:30 | 6 |
| 训练 | DAY2_17300000 | DAY2 17:30–17:40 | 6 |
| 训练 | DAY2_17500000 | DAY2 17:50–18:00 | 6 |
| 训练 | DAY2_21100000 | DAY2 21:10–21:20 | 6 |
| 训练 | DAY3_17000000 | DAY3 17:00–17:10 | 6 |
| 验证 | DAY4_21400000 | DAY4 21:40–21:50 | 6条验证输入 |

时间是EgoLife文件名中的数据集时间字段，不是作业提交时间。每个训练窗口分别以六个人为提问者构造输入，所以是8个窗口×6个提问者=48条不同packet/asker输入，不能把它称为48个独立视频窗口。

每条输入仍涉及同一窗口的六个视角。实际Policy输入是图片序列：提问者保留完整300张采样帧，其余视角采用原CLIP/pruner保留的帧；Judge依照原任务使用完整、未裁剪的媒体。`data.py`严格核验图片顺序、路径、prompt和packet/asker绑定，不直接把MP4传给Policy训练器。

原3个训练窗口和验证窗口的帧直接复用；新增5个DAY2窗口由20个30秒分段/用户拼接并预处理，实际新增600段、30个10分钟成片。数据准备作业19348785已保存passed的48/6验收记录。

当前数据根：

```text
/scratch/xl6775/projects/EgoQA-six-user-baseline-grpo-20260929/variants/train8-data-20261007T162701Z/data/trainval/
  train.jsonl       48条，8个窗口，每个asker_index 0..5
  validation.jsonl   6条，独立DAY4窗口
```

源目录下`source_manifest.json`记录公开视频分段映射，`dataset_ready.json`记录真实完成状态；视频、图片、checkpoint和运行日志不放进GitHub。`hpc/grpo_v3/six_user_binary/train8_100_request_20261007.json`是历史人类确认记录，包含当时的准备状态，**不是当前可直接提交的workflow**。

## 2. 当前训练设置

| 设置 | 当前值 |
|---|---|
| Policy / 冻结Judge | Qwen3.8-27B / Qwen3.8-27B |
| 执行方式 | 单H200，Policy/Judge交替运行，vLLM colocate |
| 当前作业 | 19530750，`training_resume100_to200` |
| 恢复来源 | 已完成作业19458762的完整checkpoint-100 |
| 终点 | 总计200个优化器更新，不是从100额外增加200 |
| 学习率 | 原200步余弦曲线，峰值1e-5、warmup 10、末端比例0.1 |
| 恢复LR / 末端LR | 100步约5.871607e-6 → 200步1e-6 |
| G / 微批 / 累积 | 4 / 1 / 4 |
| LoRA | q_proj、v_proj、in_proj_qkv；rank 8、alpha 16 |
| 冻结模块 | 视觉模块与aligner |
| KL beta | 0.04 |
| 解码 | temperature 0.85、top_p 0.95、top_k 40 |
| 长度 | max_length 65536、max_completion_length 2048 |
| Policy像素上限 | max_pixels 24576 |
| 显存分配器 | Policy显式expandable_segments:True，沿用Swift/vLLM原生阶段切换 |
| 资源 | 1 H200、16 CPU、500G主存、16.75小时申请时限 |

G=4表示每个输入生成4个候选，奖励采用组内归一化。完整恢复包含adapter、optimizer、scheduler和RNG；不会重启warmup。显存分配器配置及反馈式GPU保护保持原已验证设置，不在本次续训改G、数据、reward或LoRA范围。

## 3. 实际正在使用的脚本链

所有下列路径相对本仓库的`multi-user/`。

```text
training/grpo_v3/six_user_binary/submit_after_cpu.py
  → training/grpo_v3/six_user_binary/submit.py
  → hpc/grpo_v3/six_user_binary/train_direct.sbatch
  → python -m training.grpo_v3.six_user_binary.resume
  → workflow.training_config(...)生成本次formal配置
  → python -m training.grpo_v3.six_user_binary.launch
      ├─ 冻结Judge：python -m ...six_user_binary.service
      ├─ 实际优化器训练：已安装ms-swift的 swift rlhf --rlhf_type grpo
      │    └─ external_plugins加载 ...six_user_binary/plugin.py
      ├─ 训练工程验收：...six_user_binary/validate_run.py
      └─ 独立checkpoint评分：python -m ...six_user_binary.evaluation
```

| 文件 | 作用 |
|---|---|
| `hpc/grpo_v3/six_user_binary/train_direct.sbatch` | 当前真正提交给Slurm的入口；封闭scratch、存储预检、启动resume |
| `training/grpo_v3/six_user_binary/resume.py` | 复用已处理数据、核对48/6和划分、进入正式续训 |
| `training/grpo_v3/six_user_binary/workflow.py` | 将workflow转换为训练配置；恢复场景调用training_config，不重做8窗口媒体 |
| `training/grpo_v3/six_user_binary/launch.py` | 启动Judge和保护，拼接真实Swift CLI，管理释放和最终评分 |
| `training/grpo_v3/six_user_binary/plugin.py` | 注册SixUserBinaryReward并保持候选/媒体/奖励对应 |
| `training/grpo_v3/six_user_binary/shared_gpu.py` | 在Policy卸载后唤醒Judge，评分后释放Judge再恢复Policy |
| `training/grpo_v3/six_user_binary/stage_stop.py` | 达到指定优化器步数时正常保存、验证和结束，记录恢复LR/调度器 |
| `training/grpo_v3/six_user_binary/service.py`、`scorer.py`、`reward.py` | 冻结Judge服务、四项概率评分、连续奖励汇总 |
| `training/grpo_v3/six_user_binary/evaluation.py` | 独立重载adapter，按相同输入/seed/slot比较24个生成槽位 |
| `training/grpo_v3/six_user_binary/packed_lora_compat.py`、`packed_worker_extension.py` | 部分融合QKV LoRA兼容修复，并传播到实际vLLM评分worker |
| `training/grpo_v3/six_user_binary/utilization_guard.py`、`utilization_runtime.py` | 有界反馈保护、短长窗口记录和显存让步 |
| 根目录`hpc/shared/cuda_device_identity.py` | CUDA/NVML设备身份映射，复制运行根时必须一并包含 |
| `hpc/grpo_v3/six_user_binary/prepare_train8.py` | 先前补齐五个DAY2窗口的CPU媒体/数据准备，不是当前每步训练入口 |

当前使用`train_direct.sbatch`。同目录的`train.sbatch`、`hold.sbatch`以及根`hpc/shared/cuda.py`保留旧入口兼容，不把它们当19530750的启动入口。

## 4. 实际运行配置与结果定位

当前Torch运行根：

```text
/scratch/xl6775/projects/EgoQA-six-user-baseline-grpo-20260929/variants/train8-lora1e5-continue200-20261010T142800Z
```

相对该根：

```text
submissions/grpo200_20261010T143802Z_2d021dc5/
  submission.json   保存实际JobID、资源和sbatch命令
  workflow.json     本次实际workflow配置
preflight/continue100-to200/
  resume100_check.json
outputs/formal_train8-lora200/train_19530750/
  run_config.json   真正传入运行器的配置
  train_command.json
  trainer.log
  stage_state.json
  swift/.../checkpoint-*
  training_result.json
  validation_policy.json
  validation_comparison.json
  run_manifest.json
```

上述路径用于追溯现有作业，不能直接拿它们再次提交。合作者使用自己的账号、account、模型/环境/数据和独立输出根；真实已处理数据及完整checkpoint另行按授权传输。提交仍由`submit_after_cpu.py`在匹配的CPU报告通过后返回`--parsable` JobID，不能只凭JSON文件存在假装检查完成。

`workflow_27b_18719659.json`是早期基础配置，不包含当前8窗口和完整续训信息，不直接作为当前训练模板。

## 5. 验证与结论范围

在仓库的`multi-user/`目录运行本地单元测试：

```bash
python -m unittest discover -s tests/training/grpo_v3/six_user_binary -p 'test_*.py' -v
```

缺少实际Torch/vLLM源码时，数值兼容测试明确skip，不能据此声称该边界验证通过。具备实际安装环境时，`test_packed_lora_compat.py`默认验证修复；显式设置`EGOQA_TEST_PACKED_COMPAT=0`可复现未修复方法的历史错误。Bash特定检查在支持的Linux环境执行。

当前CPU参数检查会把CLI中保留的JSON字符串等价转换为字典后严格比较调度配置；非法JSON、数组和布尔值仍拒绝。该解析修复没有修改学习率曲线或训练器。

工程验收、完整保存和固定评分分开。验证是一个视频窗口的6个提问者输入、24个生成槽位，不是24个独立视频或人工gold题；固定Judge代理奖励不等于人工QA质量、统计显著性或收敛证明。当前48/6配置与另行讨论的speaker级划分方案不能混用。

本机统一Torch规则仍来自工作区；仅克隆GitHub时可读[仓库内协作副本](TORCH_RULES_CN.md)。更完整的历史设置、失败原因和验收记录见[Runbook](SIX_USER_GRPO_27B_TORCH_RUNBOOK_CN.md)。
