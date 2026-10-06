# Qwen3.8-27B binary judge training

This package trains the existing language-model vocabulary logits directly. It
adds no classification head and supervises no scalar score.

## 在其他机器上复现训练

以下步骤复现当前的 groundedness 训练（0.25 FPS、batch 1、GA=4、cosine、
warmup 0.05、不做类别加权）。

### 1. 硬件

| 项 | 要求 |
|---|---|
| GPU | 2 × H200（141 GB）。单卡峰值约 137 GB，80 GB 显卡放不下 |
| CUDA 驱动 | 支持 CUDA 13.0（环境里是 `torch 2.13.0+cu130`） |
| 内存 / CPU | 256 GB 内存，约 16 核（两个 dataloader worker 负责解码 900 张图） |
| 磁盘 | 约 200 GB：模型 52 GB，数据 tar 75 GB，解压后 75 GB |

### 2. 代码

```bash
git clone -b xth https://github.com/GODAVID0819/EgoQA-two-user.git
cd EgoQA-two-user/multi-user      # 必须从这个目录启动：入口是 -m training.judge_sft.<module>
```

### 3. Python 环境

环境文件在 `multi-user/requirements/`：`qwen38-vllm.conda-explicit.txt`、
`qwen38-vllm.environment.yml`、`qwen38-vllm.pip-freeze.txt`。

```bash
conda create -y -p ./qwen38-vllm --file requirements/qwen38-vllm.conda-explicit.txt
grep -v "^-e \|^# " requirements/qwen38-vllm.pip-freeze.txt > /tmp/pip-reqs.txt
./qwen38-vllm/bin/pip install --no-deps \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  --extra-index-url https://flashinfer.ai/whl \
  -r /tmp/pip-reqs.txt
```

两个额外 index 都是必需的：`torch==2.13.0+cu130` 只在 PyTorch 源上有，
`flashinfer-cubin==0.6.16.post3` 只在 FlashInfer 源上有（PyPI 没有）。
`--no-deps` 保证装出来的版本与 freeze 完全一致。

### 4. 模型

```bash
hf download Qwen/Qwen3.8-27B --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --local-dir ./Qwen3.8-27B
```

### 5. 数据

数据集是私有的 HuggingFace dataset
`tianxia2/egolife-judge-sft-842-question-split`，需要有访问权限并先 `hf auth login`。

```bash
hf download tianxia2/egolife-judge-sft-842-question-split --repo-type dataset --local-dir ./judge_data
cd judge_data
mkdir -p packets_extracted && for t in packets/*.tar; do tar -xf "$t" -C packets_extracted; done
```

manifest 里的 `frame_packet` 是原集群的绝对路径
（`/scratch/hm2991/egolife_rlhf_evidence_v1/packets/<packet_id>`）。运行数据集
README 里的那段 Python，把它改写为 `packets_extracted/<packet_id>`，生成
`manifests/{train,test}/train.local.jsonl`。

数据量：train 2694 行（groundedness 674：456 pass / 218 fail），test 672 行
（groundedness 168：117 pass / 51 fail）。按问题切分，test 复用训练集中的视频
packet。

### 6. 启动训练

不依赖 SLURM 的启动脚本：`hpc/judge_sft/run_groundedness_portable.sh`。

```bash
REPO=$PWD \
ENV=/path/to/qwen38-vllm \
MODEL=/path/to/Qwen3.8-27B \
DATA=/path/to/judge_data/manifests \
bash hpc/judge_sft/run_groundedness_portable.sh
```

可选环境变量：`OUT`（输出目录）、`TASK`（默认 `groundedness`）、`EPOCHS`
（默认 10）、`EVAL=0`（关闭每个 epoch 的 test 评估）、`CUDA_VISIBLE_DEVICES`。

先跑一步冒烟测试，确认环境、显存和数据路径：

```bash
MAX_STEPS=1 EVAL=0 REPO=$PWD ENV=... MODEL=... DATA=... bash hpc/judge_sft/run_groundedness_portable.sh
```

在原集群上使用的 SLURM 版本是
`hpc/judge_sft/train_groundedness_025fps_mb1_ga4_noclassw.sbatch`（路径写死为
集群路径）。

### 7. 运行时间与输出

- 每个优化步（4 条样本）约 63 秒，每个 epoch 169 步约 3 小时；每次 test 评估
  （168 题）约 45–50 分钟；10 个 epoch 合计约 40 小时。
- 启动时会逐个检查所有帧文件是否存在，在网络文件系统上可能需要几分钟。
- 输出：`$OUT/trainer/checkpoint-<step>/`（每个 epoch 一个，含 LoRA adapter 与
  优化器状态）、`$OUT/trainer/training_contract.json`（全部超参与类别权重）、
  `$OUT/train.log`（训练 loss 与每个 epoch 的 test 指标）。
- 离线评估（逐题 margin、AUROC、阈值分析）见 `hpc/judge_sft/eval_margin/`。

## Exact target and media contract

Every invocation ends at the fixed assistant prefix:

```text
{"verdict":"
```

`pass` and `fail` must each be one tokenizer token. If their next-token logits
are `z_pass` and `z_fail`, the binary margin is `z_pass - z_fail` and training
uses `BCEWithLogits(margin, target)`, with PASS=1. At inference, code compares
only these two logits. Free-generated text never determines the verdict.

The production generation run did not retain six monolithic training videos.
Its durable judge input is in
`/scratch/$USER/egolife_rlhf_evidence_v1/packets`: six ordered, packet-owned
timelines of exactly 300 JPEGs each, already sampled from 600 seconds at 0.5
FPS. Every sampled JPEG is supplied to Qwen as an independent image item.
Groundedness and all-six answerability therefore see 1,800 ordered images in
six explicitly described chronological user groups; speaker-only
answerability sees 300 images; formality is text-only. Both
answerability conditions receive only the generated question from the QA item.
Options, the correct letter, answer text, rationale, and every other QA field
are withheld; condition metadata still identifies which media is present.

The collator uses Qwen3.8's processor-declared vision geometry: a 16-pixel
spatial patch and a 2-by-2 spatial merge. The image processor also declares a
temporal patch size of two, but each still image satisfies that internal tensor
dimension independently; adjacent sampled frames are never paired into a video
tubelet. The collator passes `image_patch_size=16` explicitly to
`qwen-vl-utils` and fails closed if the loaded processor reports different
geometry. The resulting language-model spatial-token area is 32-by-32 pixels,
not the older 28-by-28 assumption.

The dynamic per-frame resolution cap budgets all 1,800 independent image items
against the configured context target. With the defaults, the all-six cap is
119,808 pixels per frame. A one-user 300-frame call retains the configured
262,144-pixel cap:

```text
max_input_tokens=262144
target_fraction=0.85
text_token_reserve=8192
item_token_overhead=2
min_pixels=3136
configured_max_pixels=262144
```

For all-six rows, the cap is applied to every image. There is no two-frame
temporal tubelet packing and no video metadata. The processor-only runtime gate
records the resulting packed token count, and the collator rejects any row that
exceeds `max_input_tokens`.

Manifests are compact. A visual row stores `frame_packet`, ordered
`frame_user_indices`, and `frame_order`; the loader resolves and verifies all
300 frames per selected user from the relative paths recorded in `packet.json`.
The on-cluster packet layout is:

```text
/scratch/$USER/egolife_rlhf_evidence_v1/
└── packets/
    └── <packet-id>/
        ├── packet.json
        └── frames/
            ├── user_0_A1_JAKE/
            │   └── ...jpg
            ├── user_1_.../
            │   └── ...jpg
            └── ...
```

Frame paths inside `packet.json` are packet-relative (for example,
`frames/user_0_A1_JAKE/...jpg`); the training manifest references the packet
directory, not the `frames` directory directly.

## Human-label conversion

`prepare_real_data` joins each label by
`source_evidence_id::attempt_NN` to `qa_mcq.intermediate.jsonl`. This is
necessary because the 221 labeled candidates include rejected generation
attempts, not only rows in `qa_mcq.jsonl`. The join recovers the full QA and its
original schema errors, reconstructs current prompts, verifies packet/user/URL
provenance, and refuses missing or duplicate matches.

The preparation module resolves `prompts.py` from this exact `multi-user`
checkout and builds the full judge view directly from each persisted
`packet.json`. It does not import the legacy `rlhf_evidence_preprocessing.py`
pipeline, and every visual judge row references the complete unpruned sampled
frames rather than the generation-time provider-pruned view.
The annotation and packet provenance must contain the identical multiset of
source-video URLs, including duplicate counts; their serialized/display order
may differ because frame ordering comes from the packet's user and frame indices.

Each labeled candidate becomes up to four invocations:

- one formality PASS/FAIL row;
- one groundedness PASS/FAIL row;
- one speaker-only answerability row from `asker_only_answerable`;
- one all-six answerability row from `all_six_answerable`.

The aggregate `answerability_verdict` is checked against the gate
`not asker_only_answerable and all_six_answerable`, but it is not a training
target. If a human PASS contradicts a deterministic FAIL embedded in the
formality prompt, only that impossible task row is excluded and recorded in
`excluded_task_rows.jsonl`; the other three labels remain usable.

Every usable row from all 40 supplied packets is written to `train.jsonl`; no
internal validation manifest is created. `data_audit.json` records label and
generation hashes, full-training counts, exclusions, the media contract, and
the question-only answerability prompt contract.
`prompt_snapshots.json` contains one complete prompt per judge condition. A
future, separately labeled standalone validation/test collection will select
among the saved epoch checkpoints. Because prompts are embedded in
`train.jsonl`, manifests prepared before the question-only answerability
contract must not be reused. Rerun `prepare_real_manifests.sbatch` after
syncing this revision.

## Loss weighting and first run

Class imbalance is corrected independently per task with smoothed, capped,
mean-one inverse-frequency weights. Task sampling scales make the dataset-level
objective exactly:

```text
0.2 * mean(formality BCE)
+ 0.4 * mean(groundedness BCE)
+ 0.4 * mean(answerability BCE)
```

The first real run uses BF16 Qwen3.8-27B with language-side attention and MLP
LoRA on `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and
`down_proj`, rank 8, alpha 16, dropout 0.05. Adapters are applied only to the
upper 16 decoder layers, indices 48 through 63. Layers 0 through 47 still run
the pretrained forward pass but contain no adapters and remain outside the
autograd graph. A forward pre-hook fails the job if layer 48 receives a tensor
that already requires gradients, proving that the frozen prefix was not merely
excluded from the optimizer while still retaining its activation graph.
Non-reentrant checkpointing computes adapter parameter gradients without
requiring the frozen-prefix input to require gradients.

The launcher audits the exact 16-layer selection and that every requested
projection family has trainable language-side LoRA parameters. The vision
encoder, merger/aligner, embeddings, LM head, lower 48 decoder layers, and all
base weights remain frozen. PEFT is called with
`autocast_adapter_dtype=False`, and startup fails unless every trainable LoRA
parameter is BF16; this prevents the FP32 LoRA-B activation that caused the
measured 5.51-GiB allocation failure. Attention defaults to PyTorch native SDPA, avoiding an external
`flash-attn` build while still allowing H200 fused SDPA kernels. Set
`ATTN_IMPLEMENTATION=flash_attention_2` only in an environment with a compatible
`flash_attn` installation. The full launcher uses two H200s as one pure TP2
model replica, non-foreach/non-fused AdamW, LR `2e-5`, betas
`(0.9, 0.95)`, weight decay `0.01`, 10% warmup, cosine decay, ten epochs,
one shared microbatch across the TP ranks, gradient accumulation 32 (effective
batch 32), gradient checkpointing, and clipping at 1.0. There is no in-training
evaluation or early stopping. Every epoch checkpoint is saved without a
retention cap, and `final_adapter` stores the last epoch for convenience. TP
checkpoints contain model/adapter state only because optimizer-state resume is
not supported for models sharded at load time. Both TP ranks receive the same
example; the trainer all-gathers a stable example fingerprint before every
forward and fails if rank inputs differ. CPU image preprocessing uses two
persistent DataLoader workers per TP rank with one-batch prefetch. The sampler
keeps examples with identical packet frames adjacent, and each worker keeps a
two-entry LRU of decoded/resized CPU images. This overlaps JPEG work with GPU
training and avoids decoding shared media repeatedly; it does not cache CUDA
tensors or change the images, labels, loss weights, or optimizer schedule.

Training only the upper 16 layers reduces the rough worst-case decoder
checkpoint-boundary budget from about 131 GiB for all 64 layers to about 33
GiB. The full 1,800-image sequence, frozen model weights, vision forward, and
hybrid linear-attention working tensors still occupy both GPUs, so the all-six
one-step smoke remains a hard gate before the ten-epoch run.

Qwen's MLP and ordinary attention projections use the model's native TP plan.
The hybrid linear-attention projections remain replicated because current
native Transformers TP has a known reshape failure for that Qwen path. This
still shards every `gate_proj`, `up_proj`, and `down_proj`, including the exact
wide MLP operation responsible for the preceding OOM.
The launcher preserves the 10% warmup across Transformers APIs: it uses
`warmup_ratio=0.1` where supported and the Transformers v5.2+
`warmup_steps=0.1` ratio form otherwise.

## What the fixed assistant prefix actually does

The prefix is part of the tokenized input context; it is not claimed as model
output. The collator first asks Qwen's chat template to render the user message
and open an assistant turn with thinking disabled. Qwen3.8 represents this with
a canonical closed empty thinking block. The collator then concatenates the
literal verdict prefix:

```text
<rendered user content><assistant marker><think>\n\n</think>\n\n{"verdict":"
```

The empty block is template control syntax, not generated reasoning. An open or
non-empty thinking block is rejected before tokenization.

Because a causal LM's logits at the final input position predict the next
token, those logits answer: "what token follows the opening verdict quote?"
Training reads only the `pass` and `fail` entries of that vocabulary vector and
applies BCE to their difference. It does not append the human verdict, run
generation, or supervise the remainder of the JSON.

Inference uses one continuous autoregressive generation. At generation step
one, a logits processor records the original vocabulary scores, compares only
`pass` and `fail`, and permits exactly the selected verdict token. Starting at
step two it becomes a no-op, so the same `model.generate` call continues with
the closing quote, later contract fields, closing brace, and EOS. There is no
second prompt and the model is not re-asked for its verdict.

For example, the single generated continuation after the code-supplied prefix
may be:

```text
pass","reason":null,"fix":null}
```

Together, the input prefix and generated continuation form:

```json
{"verdict":"pass","reason":null,"fix":null}
```

For FAIL, the same uninterrupted generation writes the concrete `reason` and
`fix`. The complete object is schema-validated, while the authoritative verdict
always remains the first constrained token. The adapter-reload smoke test now
exercises both the binary logit decision and complete continuous JSON
generation. Training itself still computes loss only at the verdict position;
the available human annotations contain verdict labels, not supervised
reason/fix targets.

## Cluster sequence

Put exactly these two completed exports in a cluster directory (the filenames
may retain their original suffixes):

```text
six_user_binary_labels_annotator-01-packets-001-020-v2_hm2991.jsonl
six_user_binary_labels_annotator-05-packets-081-100-v2_ac.jsonl
```

The default is `/scratch/$USER/egolife_judge_sft_labels`. Then run each gate in
order, substituting the exact manifest output path printed by the first job:

```bash
mkdir -p hpc/logs
sbatch hpc/judge_sft/prepare_real_manifests.sbatch

sbatch --export=ALL,MANIFEST_DIR=/scratch/$USER/egolife_judge_sft/manifests_40_packets_JOBID \
  hpc/judge_sft/runtime_smoke_qwen38_27b.sbatch

sbatch --export=ALL,MANIFEST_DIR=/scratch/$USER/egolife_judge_sft/manifests_40_packets_JOBID \
  hpc/judge_sft/train_one_step_qwen38_27b.sbatch

sbatch --export=ALL,MANIFEST_DIR=/scratch/$USER/egolife_judge_sft/manifests_40_packets_JOBID \
  hpc/judge_sft/train_real_40_packets_qwen38_27b.sbatch
```

The runtime gate processes a real 1,800-frame example without loading model
weights. The one-step gate proves one optimizer update, finite loss/gradient,
nonzero LoRA-B tensors, save/reload, binary-logit selection, and continuous
generation of a complete verdict-first JSON object. Only after both pass should
the ten-epoch job run. These gates establish data/runtime/training plumbing;
they do not establish held-out judge quality. The full job writes
`checkpoint_inventory.json` and fails if it cannot find a checkpoint for every
configured epoch.

The collator treats these manifests as image-only. It rejects any unexpected
video output and omits `videos` and every video-only processor field.

Both GPU training launchers start the same utilization-aware CUDA keeper used
by the production question-generation job. It is enabled by default, watches
both allocated GPUs, uses the conservative generation defaults (including a
0.25 GiB maximum preallocation), writes `cuda_keeper.log` inside the job output,
and is stopped by the exit trap. The CPU manifest job and processor-only runtime
probe do not start it. The shared helper must exist at
`$project/hpc/shared/cuda_high_duty.py`, and the training environment must
provide `pynvml` and `psutil`. Set `ENABLE_CUDA_KEEPER=0` only when deliberately
running without the keeper.

All launchers default to the actual production locations:

```text
project:    /scratch/$USER/Long-video-understanding-clip
package:    $project/egolife_two_user_qa/multi-user
frames:     /scratch/$USER/egolife_rlhf_evidence_v1
generation: /scratch/$USER/egolife_rlhf_qa_generation/qwen38_legacy_two_pass_schema_v2
```

They default to the generation environment at
`/scratch/$USER/conda/envs/qwen38-vllm`. It must contain Transformers 5.4 or
newer, Accelerate 1.12 or newer, PEFT 0.19.0 or newer with TP-aware LoRA,
`qwen-vl-utils`, and Safetensors. The launchers do
not install or upgrade packages; if those training dependencies live in a
separate environment, pass its exact directory as `TRAIN_ENV`.
Transformers 5.16 materializes TP correctly but omits its `model._tp_size`
marker; the trainer validates the DTensor mesh and supplies that metadata
before constructing `Trainer`, matching the upstream 5.17 behavior.
Epoch and final adapter saves are collective under TP2: both ranks enter
PEFT's state-dict gathering path, while only rank zero writes the adapter,
processor, and training-argument files. This avoids a rank-zero-only DTensor
gather deadlock at checkpoint time. The collective receives an adapter-only
state dict so checkpointing never gathers the frozen 27B base model to CPU.
Immediately before PEFT creates the adapters, both ranks also reset to the
configured training seed. Rank zero then broadcasts every full LoRA A/B tensor
before any factor is sharded, and an exact all-gather equality audit must pass.
This prevents independently initialized TP ranks from being combined into an
incoherent adapter.
If the shared Hugging Face cache contains more than one Qwen3.8-27B snapshot,
set `MODEL_PATH` to one exact snapshot directory.
