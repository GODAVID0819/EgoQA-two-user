# 六用户 QA：Torch/H100 运行手册

## 1. 本轮冻结合同

本流程只挖掘六用户证据包并生成 QA，不训练或修改模型、reviewer、DPO、GRPO、优化器或 checkpoint。

- Torch 登录用户与 scratch 所有者：`hm2991`。
- Slurm 项目 allocation：`torch_pr_674_tandon_advanced`。用户名和 allocation 是两个不同概念；提交前必须用 `sacctmgr` 确认 `hm2991` 仍关联该 allocation。
- launcher 不固定 partition 或 QoS；由集群按 account、GRES 和 H100 constraint 调度。提交时的 account override 仍以本轮实际授权为准。
- 输入：同一同步组的 6 个视频。
- 对每个同步组，依次尝试 6 个人分别作为 asker/speaker。
- 每个视频独立进行 CLIP 帧聚类，默认请求 12 个 cluster；实际 cluster 数以诊断字段为准。
- 对一个 asker 候选，比较 asker 的每个 cluster 与其他 5 个 provider 的每个 cluster。若实际 cluster 数为 $N_A,N_{P_1},\ldots,N_{P_5}$，比较总数必须是：

  $$N_A\sum_{i=1}^{5}N_{P_i}$$

- 每个 provider cluster 只要与任一 asker cluster 的相似度达到或超过 `0.82`，该 provider cluster 的成员区间就被裁剪。不是只比较 argmax，也不要求多个 provider 达成共识。
- `topk`、mean-sim 上下界和 pair eligibility 是两用户 pair scorer 的旧参数；六用户分支在进入该 scorer 前已经返回，因此正式 launcher 不传这些参数。`0.82` 不是 pair-selection gate，而是决定 provider 的哪些 cluster 与 asker 重复、需要裁剪的实际阈值，所以仍然需要。
- asker cluster 永远不标记删除；生成器收到 asker 用于 CLIP 聚类和相似度计算的完整 1 FPS 采样帧集合（30 秒窗口通常为 30 张图）。
- 生成器的其余 5 组输入只包含各 provider 未被裁剪 cluster 的采样成员帧。生成阶段只发送图片，不发送任何 MP4；groundedness 使用六段完整原视频。answerability 分别检查 speaker 的完整原视频和六段完整原视频，只判断给定视频是否包含足以回答问题的视觉证据，不选择 A–E、不输出答案，也不与 gold 选项比较。
- 正式作业必须精确写出 100 个候选 packet。每个 packet 进入 feedback-driven QA loop，最多进行 3 次生成/审核；通过即提前停止，失败反馈和上一轮生成会传入下一轮。QA 接受数可以小于 100，但 packet 数和 packet-level generation coverage 不能 partial。

活动脚本：

- 复用 job `16220358` 全部 100 个已准备 packet 的正式作业：`hpc/qa/production/run_six_user_qa_reuse_packets_100.sbatch`
- 只有需要重新挖掘 packet 时才使用：`hpc/qa/production/run_six_user_qa_packets_100.sbatch`

正式 wrapper 会在同一个 Slurm allocation 内调用共享 runtime worker；不需要先提交独立的 1-packet probe。

## 2. 固定远端路径

```bash
PROJECT_ROOT=/scratch/hm2991/Long-video-understanding-clip
PACKAGE_ROOT=${PROJECT_ROOT}/egolife_two_user_qa/multi-user
OUTPUT_ROOT=${PACKAGE_ROOT}/outputs/six_user_qa
TRAIN_ENV=/scratch/hm2991/conda/envs/qwen3vl-smoke
FFMPEG_ENV=/scratch/hm2991/envs/egoqa-ffmpeg-runtime
MODEL_ID=Qwen/Qwen3.6-27B
MODEL_CACHE_ROOT=/scratch/hm2991/hf_cache
TORCH_ACCOUNT=torch_pr_674_tandon_advanced
```

这些默认值与本工程现有的 project-level `hpc/*.sbatch` 一致，不再假设上游 GitHub 作者拥有的本地 model checkout。`Qwen/Qwen3.6-27B` 使用 hm2991 既有的 `/scratch/hm2991/hf_cache`；其余 HOME、数据集、Torch、编译和临时 cache 位于 `/scratch/${USER}/job_scratch/<run>_<job-id>`。所有作业产物目录由真实 `${SLURM_JOB_ID}` 派生，不得使用 `latest_*` 作为结论依据。
正式 wrapper 将 `OUTPUT_ROOT` 固定为 `${PROJECT_ROOT}/egolife_two_user_qa/multi-user/outputs/six_user_qa`；路径中的目录名是带连字符的 `multi-user`，不是 `multiuser`。
`FFMPEG_ENV` 是首选位置，不是唯一允许的位置。runtime 在激活 Qwen 环境后依次查找 `${FFMPEG_ENV}/bin`、`${TRAIN_ENV}/bin` 和当前 `PATH`，并在日志中输出最终的 `source`、`ffmpeg` 与 `ffprobe` 路径；三处都找不到才停止。
runtime 通过绝对路径 `${TRAIN_ENV}/bin/python -P` 执行 CUDA keeper、preflight、挖掘、生成、验证和所有内嵌 Python，不依赖登录 shell 中的 `python`，也不让当前工作目录抢占 `PYTHONPATH`。`conda activate` 后会验证 `sys.prefix` 与 `CONDA_PREFIX` 都精确指向 `${TRAIN_ENV}`；`validate_multi_user_package_import` 还要求实际导入的 package 和 `qwen3vl_runner.py` 都解析到 `${PACKAGE_ROOT}`，否则在挖掘前停止。

## 3. 本地静态验证

在 `multi-user` 目录运行：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m unittest `
  tests.test_six_user_group_relative_sampling `
  tests.test_six_user_prompts `
  tests.test_six_user_video_qa_loop `
  tests.test_six_user_torch_job_contract -v

bash -n hpc/qa/smoke/run_six_user_qa_runtime_probe.sbatch
bash -n hpc/qa/production/run_six_user_qa_reuse_packets_100.sbatch
```

本地通过只能证明 Python/Slurm 静态合同，不能证明 CUDA、远端路径、模型、FFmpeg、TorchCodec、媒体下载或 QA 质量。

## 4. 登录节点资源与路径预检

```bash
PROJECT_ROOT=/scratch/hm2991/Long-video-understanding-clip
PACKAGE_ROOT=${PROJECT_ROOT}/egolife_two_user_qa/multi-user
TRAIN_ENV=/scratch/hm2991/conda/envs/qwen3vl-smoke
FFMPEG_ENV=/scratch/hm2991/envs/egoqa-ffmpeg-runtime
MODEL_ID=Qwen/Qwen3.6-27B
MODEL_CACHE_ROOT=/scratch/hm2991/hf_cache
TORCH_ACCOUNT=torch_pr_674_tandon_advanced

sacctmgr -nP show assoc where user=hm2991 \
  format=User,Account
sinfo -h -o '%P|%a|%l|%G|%f'

ACCOUNT_OK=0
if sacctmgr -nP show assoc where user=hm2991 account="${TORCH_ACCOUNT}" \
  format=Account | grep -q '^torch_pr_674_tandon_advanced|'; then
  ACCOUNT_OK=1
else
  echo "STOP: hm2991 未查询到 ${TORCH_ACCOUNT} 关联。"
fi

for REQUIRED_PATH in \
  "${PROJECT_ROOT}" \
  "${PROJECT_ROOT}/hpc/qa/smoke/run_six_user_qa_runtime_probe.sbatch" \
  "${PROJECT_ROOT}/hpc/qa/production/run_six_user_qa_reuse_packets_100.sbatch" \
  "${PACKAGE_ROOT}/__main__.py" \
  "${PACKAGE_ROOT}/group_relative_clip_sampling.py" \
  "${TRAIN_ENV}/bin/python"; do
  if [[ -e "${REQUIRED_PATH}" ]]; then
    echo "FOUND: ${REQUIRED_PATH}"
  else
    echo "MISSING: ${REQUIRED_PATH}"
  fi
done
echo "account_ok=${ACCOUNT_OK}"
```

只有 allocation、H100、工程和 Python 路径均通过后才能提交。模型不要求 `/scratch/hm2991/models/...` checkout；`validate_model_runtime` 会用 `AutoConfig.from_pretrained("Qwen/Qwen3.6-27B")` 验证既有 Hugging Face cache 或网络解析。FFmpeg/ffprobe 由正式作业在激活环境后解析；若解析失败，`resolve_ffmpeg_runtime` 阶段会列出检查过的位置。launcher 不要求显式 partition 或 QoS。

## 5. 上传范围

远端部署必须保留下面的两层布局；`hpc/` 直接位于工程根目录，六用户 Python 包位于 `egolife_two_user_qa/multi-user/`：

```text
Long-video-understanding-clip/
├── hpc/qa/...
└── egolife_two_user_qa/multi-user/...
```

只同步本轮相关源码、测试、脚本与本手册。下列 Python、测试、文档路径相对于 `${PACKAGE_ROOT}`：

```text
group_relative_clip_sampling.py
prompts.py
schema.py
video_qa_loop.py
tests/test_six_user_group_relative_sampling.py
tests/test_six_user_prompts.py
tests/test_six_user_video_qa_loop.py
tests/test_six_user_torch_job_contract.py
docs/SIX_USER_QA_TORCH_RUNBOOK_CN.md
```

下列 launcher 路径相对于 `${PROJECT_ROOT}`：

```text
hpc/qa/smoke/run_six_user_qa_runtime_probe.sbatch
hpc/qa/production/run_six_user_qa_reuse_packets_100.sbatch
```

SFTP 登录为 `sftp hm2991@login.torch.hpc.nyu.edu`，远端工程根目录为 `/scratch/hm2991/Long-video-understanding-clip`。不要覆盖历史输出或清理未跟踪研究文件。

## 6. 直接提交 100-packet 作业

```bash
cd /scratch/hm2991/Long-video-understanding-clip
mkdir -p hpc/logs egolife_two_user_qa/multi-user/outputs/six_user_qa/submission_manifests

sbatch --parsable \
  --chdir="$PWD" \
  hpc/qa/production/run_six_user_qa_reuse_packets_100.sbatch
```

不单独提交 probe，也不运行 `sbatch --test-only`。正式作业本身依次经过 storage preflight、FFmpeg/TorchCodec/decord、六视频解码、provider-only all-pairs 诊断、full-asker/pruned-provider 路由、生成、formality、groundedness 和两次 reasoning-only answerability；任一阶段失败都会留下阶段标记。speaker-only 必须返回 `answerable=false`，all-six 必须返回 `answerable=true`；输出只包含证据充分性判断、理由、已有证据和缺失证据。

六视频 memory-safe runner 显式把 `QWEN_MEMORY_SAFE_MIN_VIDEO_PIXELS=3136` 传给每个 video item，并验证它不超过 launcher 的 `max_image_pixels=65536`。这避免 `qwen-vl-utils` 回退到高于 65536 的默认 video floor。

作业启动后会立即打印带时间戳的 `stage=<name> status=started`。运行时持续覆盖 `${OUTDIR}/stage_status.json`：正常结束记录 `completed`，shell 错误记录 `failed`、stage、line 和 exit code，收到 `INT/TERM` 时记录 `cancelled` 和 signal。GPU keeper 在 storage/manifest 之前启动，日志写入 `${OUTDIR}/cuda_keeper_<job-id>.log`，并由 EXIT trap 清理。

## 7. 100-packet 与三轮 loop 合同

`run_six_user_qa_reuse_packets_100.sbatch` 固定复用 job `16220358` 的全部 100 个候选 packet（offset 0），不重新挖掘：

```text
EVIDENCE_TARGET=100
ACCEPTED_TARGET=100
MAX_GROUPS=800
MAX_ATTEMPTS=3
ALLOW_PARTIAL=1
REFERENCE_JOB_ID=16220358
PREPARED_EVIDENCE_OFFSET=0
```

`ALLOW_PARTIAL=1` 只允许最终接受 QA 少于 100；候选 validator 仍要求 `six_user_candidates.jsonl` 恰好包含 100 个 packet，而且每个 packet 必须至少生成一次。每个 packet 的 attempt 序列必须从 1 连续递增、不得超过 3；通过后提前停止，三次都未通过才写入 rejected。最坏情况下会执行 300 次生成。挖掘器在组内达到目标后立即停止追加，避免一个同步组追加多个 speaker 候选导致 100 被超出。

## 8. 必查产物与结论边界

真实输出目录中必须检查：

```text
stage_status.json
cuda_keeper_<job-id>.log
storage_preflight.json
job_manifest.json
group_relative_clip_summary.json
six_user_candidates.jsonl
video_first_prompts.jsonl
qa_mcq.jsonl
qa_mcq.rejected.jsonl
qa_mcq.intermediate.jsonl
six_user_qa_result.json
generation_report.md
human_review_sheet.md
```

每个 candidate 必须满足：

- `comparison_scope=every_speaker_cluster_x_every_provider_cluster`；
- `pairwise_comparison_count=expected_pairwise_comparison_count`；
- `pruned_side=providers_only` 且 `speaker_preserved=true`；
- asker 的 `marked_cluster_indices`、`remove_intervals` 为空；
- 每个 threshold event 只删除对应 provider cluster；
- 所有 clip 都设置 `force_frame_inputs=true`，且不存在 `local_video` 或 `generator_local_video` 生成器路由；
- asker 的图片数等于其全部 cluster member 总数；每个 provider 的图片数等于未标记删除 cluster 的 member 总数，且图片按原采样顺序排列。

这些产物能证明本次作业执行了声明的 cluster 比较、裁剪与媒体路由。它们不能单独证明生成问题自然、视觉 grounded、speaker 单独不可答或总体 QA 质量；这些仍需要 full-video judges、reasoning-only answerability Gate 和人工抽查。该 Gate 证明的是 judge 对证据充分性的判断，不证明模型实际选择了正确选项，因为 answerability judge 被明确禁止回答问题。
