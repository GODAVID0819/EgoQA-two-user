# VideoLLaMA3-7B：原 GRPO 帧输入的三视角条件评测

本目录适配 `DAMO-NLP-SG/VideoLLaMA3-7B`，使用 `tianxia2/egolife-six-user-mcqa-textwrong131` 全部 131 题，对比两个证据视角（`evidence2`）、提问者视角（`speaker1`）和六视角（`all6`），共 393 条预测。

已完成作业 `21411288`：49/131、45/131、41/131；两个证据视角相对六视角为 +6.11 个百分点。完整结果及解释边界见 [RESULTS_CN.md](RESULTS_CN.md)。代码复用父目录 `vqa_grpo_frames.py`，不能只复制本子目录运行。

## 输入及模型设置

沿用参考提交 `2fff59962208eccea1b6fa495e7f4f96129d21da` 的 GRPO 帧流程：speaker 保留原 300 帧，provider 保留原掩码选中的帧；不改帧序、不重算剪枝。使用原 JPEG95/subsampling0 和面积 24576 预缩放，1408×1408 变为 156×156；全部独立图片在前，原描述、metadata、问题和选项在后。

使用既有 `vqa_shuffled.json`，选项已按 seed0 重排，不再打乱。原 `Answer + Reason` 提示词，greedy、最多 512 新 token、batch4、seed0、BF16、repetition_penalty=1.0；原生 32768 上下文，不截断或扩展。证据列表中有 1 题只有 speaker，仍保留在全 131 题分母。

**156×156 是预缩放尺寸。** 官方处理器按同一请求图片数自动缩图：speaker1 实际 98×98，evidence2 为 70/84/98/112 方图，all6 为 56×56。因此差值同时包含视角集合和原生缩图的影响。程序逐题记录最终尺寸和 CPU/GPU 实际上下文。

独立图片的原压缩掩码恒真；`image_batching.py` 仅绕过恒等压缩以支持原 batch4，并拒绝 video 输入。小模型 CPU 测试核对变长批次与官方单题生成一致；没有加载 7B 权重进行该测试。

## 环境与素材

正式已验证环境为 Python3.11、torch2.4.1/cu124、torchvision0.19.1、transformers4.46.3、FlashAttention2.7.3；其余依赖见 `requirements-inference.txt`。`setup_gpu_assets.sh` 仅适用于相同 Python/CUDA/ABI 组合，使用独立环境，不降级现有 Qwen 环境。CPU 测试需要 torch/torchvision 及推理依赖；重建测试还需要可用 FFmpeg（或本地 `imageio-ffmpeg`）。

在本目录执行以下素材下载。命令只获取两份题集 JSON 和官方源码/分词器，不下载图片、视频或模型权重；下载文件由 `.gitignore` 排除。

```bash
hf download tianxia2/egolife-six-user-mcqa-textwrong131 \
  --repo-type dataset --revision 90e6e9bfbbe4b2f009bb37fdf25c1920e972a4a5 \
  --local-dir . --include dataset.json vqa_shuffled.json
hf download DAMO-NLP-SG/VideoLLaMA3-7B \
  --revision d5b763e368861e7f5096e7ff1b49f92fbccf8ae6 \
  --local-dir upstream --include '*.py' '*.json' '*.txt'
python run_local_checks.py
```

`prompt_header.json` 保存本次原指令段；`original_video_sizes.json` 保存源视频路径和大小，不包含媒体。模型源码及分词器从官方固定版本获取，不在仓库重复存放。

## CPU 准备与正式运行

下面各命令在已安装相应依赖的计算作业内执行。先设置本次独立 scratch 根目录 `VQA_ROOT`，至少准备模型、帧缓存和输出所需空间；不要覆盖历史作业输出。

优先复用同版本、同原始清单的 156PNG 缓存。缺帧时，可以用已有 EgoLife 原视频及公开源重建，再完整检查 393 条输入与 89255 张唯一帧：

```bash
python rebuild_from_videos.py --cache "$VQA_ROOT/frames_156" \
  --video-cache "$VQA_ROOT/original_video_cache" \
  --original-videos "/scratch/$USER/datasets/EgoLife" \
  --workers 8 --report "$VQA_ROOT/rebuild_ready_$SLURM_JOB_ID.json"
python prepare_frames.py --model upstream --cache "$VQA_ROOT/frames_156" \
  --workers 8 --report "$VQA_ROOT/cpu_ready_$SLURM_JOB_ID.json"
```

重建保留原 20 段 padding/trim→concat→0.5FPS 时间轴；先复现 JPEG，再原样缩放。独立按每个 30 秒片段抽帧会在短段边界改变时间轴，本实现保留整窗口 concat。源视频固定 `lmms-lab/EgoLife@143fb319be7aa5ae210c936bf4f0f3a86092afb0`；最多涉及 7812 段、117.53GiB，完整已有窗口会直接复用。该原视频版本与题集帧版本不同，分别记录。

`prepare_frames.py` 也可以直接下载题集原帧，但匿名逐图请求较慢；它优先复用有效缓存。`--context-only` 仅检查容量，不产生可以正式推理的媒体通过报告。

Slurm 模板保持正式实验资源：CPU 16核/64GiB/3小时；GPU 1×H200、16核/128GiB/2小时。模板不连接、不自动提交、也不取消作业。提交前在当前集群核实账户、路径、环境及利用率规则。

模板需要通过环境传入 `VQA_ROOT`、`VQA_CODE_ROOT`（本目录绝对路径）、`VQA_ENV_BIN`、`VQA_FFMPEG_BIN`、`VQA_FFMPEG_LIB`；CPU 模板还需要 `VQA_GPU_ENV`、`VQA_BASE_PYTHON` 用于准备独立 GPU 环境和权重。正式 GPU 模板另需 `VQA_CPU_REPORT` 指向本次 `PASS` 报告。事先创建 `$VQA_ROOT/logs`。用 `sbatch --parsable --account="$VQA_SLURM_ACCOUNT" --output=...` 提交并立即保存真实 JobID，不沿用他人的账户或旧作业号。

需要直接调用推理时，使用相同模型环境、存储封闭与 FFmpeg 设置，在 GPU 作业内执行：

```bash
python evaluate_grpo.py --model "$VQA_ROOT/model" --cache "$VQA_ROOT/frames_156" \
  --cpu-report "$VQA_CPU_REPORT" --out "$VQA_ROOT/outputs/formal_$SLURM_JOB_ID"
python audit_results.py --out "$VQA_ROOT/outputs/formal_$SLURM_JOB_ID"
```

输出目录必须新建。中断后可添加 `--resume-from`，仅复用同配置、题号、gold 和原帧路径匹配的完整记录，忽略末尾未写完的行，保留来源 JobID。三组必须全部配对 131 题；解析失败记错，不缩小分母。

`runtime_env.sh` 保留原运行时缓存封闭和空间检查；个人 scratch 范围改由当前用户推导。发布版 Slurm 模板去除了个人账户和固定路径；GPU 采样器移入模型所在的 `srun` 步骤。上述发布整理仅做本地 CPU/语法验证，未重新提交远端 CPU/GPU 作业，也未重跑准确率。

## 文件与结果

- `grpo_inputs.py`：直接复用父目录的视角选择、提示词和解析器。
- `model_adapter.py`、`image_batching.py`、`evaluate_grpo.py`：官方图片处理、容量记录、批处理和正式推理。
- `rebuild_from_videos.py`、`reference_decode.py`、`prepare_frames.py`：原视频重建或原帧下载与检查。
- `resume_results.py`、`audit_results.py`：恢复和独立结果重算。
- `test_*.py`：真实边界检查；`run_local_checks.py` 生成本地检查记录，无 GPU 或 7B 权重加载。
- `RESULTS_CN.md`、`FINAL_STATISTICS.json`：历史已完成作业的报告和衍生汇总；不包含原始预测文件。
- `recorded_run/`：历史实际执行脚本及当时正式配置；保留作追溯，复现时使用上面的可配置入口。

历史原始逐题输出仍位于报告所列 Torch 路径；本 Git 提交不包含连接信息、认证材料、模型、媒体、缓存、日志或压缩包。

发布整理验证（2026-10-30）：13 项 CPU 测试全部通过，覆盖全部 393 个提示词、真实官方处理器容量、变长图片批处理、恢复与原 concat 重建边界；4 个 Shell/Slurm 脚本通过 `bash -n`。没有新增远端作业或 GPU 测量。
