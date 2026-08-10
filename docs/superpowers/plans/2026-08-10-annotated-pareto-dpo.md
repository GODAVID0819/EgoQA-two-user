# 人工标注 Pareto DPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 70-evidence F/E/A 人工标注转换为无权重补偿的 Pareto 偏好数据，并提供隔离的双完整视频 ms-swift DPO 数据、Smoke、Overfit、正式训练和 validation 入口。

**Architecture:** 复用 Reviewer v1 已验证的 CSV 与 `60/10/0` evidence split 解析合同，在新实验目录中增加纯函数 Pareto 层、`compact_qa_v1` prompt/序列化层和 ms-swift JSONL 构建 CLI。训练层不实现自定义 DPO loss，而是生成官方标准的 `messages + rejected_response + videos` 数据，调用固定环境中的 `swift rlhf --rlhf_type dpo`；所有 Slurm 作业都经过 storage、FFmpeg、TorchCodec、数据 SHA 和参数变化 Gate。

**Tech Stack:** Python 3、标准库 `csv/json/hashlib/dataclasses/unittest`、现有 Reviewer v1 数据合同、ms-swift 4.2.2 环境、Qwen3-VL-8B-Instruct、PEFT LoRA、Slurm/H100、TorchCodec/FFmpeg。

---

## 范围与官方兼容依据

本计划实现设计规格中的 Gate 0–5：数据审计、Structure、1-step Smoke、4-evidence Overfit、60-evidence 正式训练和 10-evidence DPO validation。Gate 6 的自由生成盲评依赖真实训练 checkpoint 和新的人工标注轮次，本计划保存其所需 checkpoint、配置和数据指纹，但不在没有远端产物时伪造端点结果。

ms-swift 官方合同：

- DPO chosen 是 `messages` 的最后一条 assistant，rejected 使用 `rejected_response`；共用多模态输入时两侧复用 `videos`：<https://swift.readthedocs.io/en/latest/Customization/Custom-dataset.html>
- 训练入口为 `swift rlhf --rlhf_type dpo`；`beta` 控制相对 reference model 的约束：<https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/RLHF.md>

新建目录：

```text
training/grpo_v3/experiments/annotated_preference/
tests/training/grpo_v3/experiments/annotated_preference/
hpc/grpo_v3/annotated_preference/
```

不得修改：

```text
training/grpo_v3/runtime/reward_plugin.py
training/grpo_v3/experiments/qa_cross_view_relation/
training/grpo_v3/experiments/human_preference_reviewer/v1/
hpc/grpo_v3/qa_cross_view_relation/
```

### Task 1：建立隔离工作树并验证基线

**Files:**
- Use: `C:/Users/20661/Desktop/Research/AR/multiuser/EgoQA-two-user-reviewer-v1`
- Create worktree: `C:/Users/20661/Desktop/Research/AR/multiuser/EgoQA-two-user-annotated-pareto-dpo`
- Create branch: `feature/annotated-pareto-dpo`

- [ ] **Step 1：确认源工作树干净且包含规格**

```powershell
git -C 'C:\Users\20661\Desktop\Research\AR\multiuser\EgoQA-two-user-reviewer-v1' status --short --branch
git -C 'C:\Users\20661\Desktop\Research\AR\multiuser\EgoQA-two-user-reviewer-v1' log -1 --format='%H %s'
```

Expected：无未提交文件，HEAD 包含 `d9fa43d` 或后续提交。

- [ ] **Step 2：通过 `using-git-worktrees` 再次检测后创建 worktree**

无原生 worktree 工具时：

```powershell
git -C 'C:\Users\20661\Desktop\Research\AR\multiuser\EgoQA-two-user-reviewer-v1' worktree add 'C:\Users\20661\Desktop\Research\AR\multiuser\EgoQA-two-user-annotated-pareto-dpo' -b feature/annotated-pareto-dpo
```

Expected：原脏的 cross-view 工作树未切换、未 stash、未修改。

- [ ] **Step 3：运行基线回归**

```powershell
python -m unittest discover -s tests/training/grpo_v3/experiments/human_preference_reviewer/v1 -p 'test_*.py' -v
python -m compileall -q training/grpo_v3/experiments/human_preference_reviewer/v1
git diff --check
```

Expected：全绿；若失败，保留原始失败并在得到用户指示前停止。

### Task 2：以 TDD 实现 Pareto 支配和重复内容去重

**Files:**
- Create: `training/grpo_v3/experiments/annotated_preference/__init__.py`
- Create: `training/grpo_v3/experiments/annotated_preference/pareto.py`
- Create: `tests/training/grpo_v3/experiments/annotated_preference/__init__.py`
- Create: `tests/training/grpo_v3/experiments/annotated_preference/fixtures.py`
- Create: `tests/training/grpo_v3/experiments/annotated_preference/test_pareto.py`

- [ ] **Step 1：写失败测试**

```python
class ParetoTests(unittest.TestCase):
    def test_dominance_requires_no_worse_and_one_strictly_better(self) -> None:
        self.assertTrue(dominates((3, 3, 2), (2, 3, 2)))
        self.assertFalse(dominates((2, 3, 2), (3, 3, 2)))

    def test_equal_and_incomparable_vectors_have_no_direction(self) -> None:
        self.assertFalse(dominates((2, 2, 2), (2, 2, 2)))
        self.assertFalse(dominates((3, 1, 3), (2, 3, 2)))
        self.assertFalse(dominates((2, 3, 2), (3, 1, 3)))
```

另写测试覆盖：无效等级；字段顺序不影响 completion 指纹；同 evidence 重复内容只保留最小 `candidate_id`；跨 evidence 不去重；pair 顺序不依赖 `display_order`；同分和不可比较不产生 pair。

- [ ] **Step 2：运行并确认 RED**

```powershell
python -m unittest tests.training.grpo_v3.experiments.annotated_preference.test_pareto -v
```

Expected：因模块或函数不存在而失败。

- [ ] **Step 3：实现最小接口**

```python
ScoreVector = tuple[int, int, int]

@dataclass(frozen=True)
class PreferencePair:
    evidence_id: str
    chosen: CandidateRecord
    rejected: CandidateRecord
    chosen_fingerprint: str
    rejected_fingerprint: str

@dataclass(frozen=True)
class PairAudit:
    total_combinations: int
    dominance_pair_count: int
    equal_vector_pair_count: int
    incomparable_pair_count: int
    duplicate_candidate_count: int

def dominates(left: ScoreVector, right: ScoreVector) -> bool:
    if any(value not in {1, 2, 3} for value in (*left, *right)):
        raise ValueError("Pareto grades must be 1, 2, or 3")
    return all(a >= b for a, b in zip(left, right)) and any(
        a > b for a, b in zip(left, right)
    )

def compact_fingerprint(evidence_id: str, candidate: CandidateRecord) -> str:
    payload = {"evidence_id": evidence_id, **candidate.model_features()}
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

def build_pareto_pairs(
    evidence: EvidenceRecord,
) -> tuple[Sequence[PreferencePair], PairAudit]:
    unique: dict[str, CandidateRecord] = {}
    duplicate_count = 0
    for candidate in sorted(evidence.candidates, key=lambda item: item.candidate_id):
        fingerprint = compact_fingerprint(evidence.evidence_id, candidate)
        if fingerprint in unique:
            duplicate_count += 1
            continue
        unique[fingerprint] = candidate

    pairs: list[PreferencePair] = []
    equal_count = 0
    incomparable_count = 0
    candidates = tuple(unique.values())
    for left, right in itertools.combinations(candidates, 2):
        left_vector = (left.qa_formality, left.evidence_quality, left.answerability)
        right_vector = (right.qa_formality, right.evidence_quality, right.answerability)
        if left_vector == right_vector:
            equal_count += 1
            continue
        if dominates(left_vector, right_vector):
            chosen, rejected = left, right
        elif dominates(right_vector, left_vector):
            chosen, rejected = right, left
        else:
            incomparable_count += 1
            continue
        pairs.append(
            PreferencePair(
                evidence_id=evidence.evidence_id,
                chosen=chosen,
                rejected=rejected,
                chosen_fingerprint=compact_fingerprint(evidence.evidence_id, chosen),
                rejected_fingerprint=compact_fingerprint(evidence.evidence_id, rejected),
            )
        )
    pairs.sort(key=lambda pair: (pair.chosen_fingerprint, pair.rejected_fingerprint))
    return tuple(pairs), PairAudit(
        total_combinations=len(tuple(itertools.combinations(candidates, 2))),
        dominance_pair_count=len(pairs),
        equal_vector_pair_count=equal_count,
        incomparable_pair_count=incomparable_count,
        duplicate_candidate_count=duplicate_count,
    )
```

候选组合使用 `itertools.combinations`；重复内容代表按 `candidate_id` 字典序确定。

- [ ] **Step 4：运行 GREEN 并提交**

```powershell
python -m unittest tests.training.grpo_v3.experiments.annotated_preference.test_pareto -v
git add training/grpo_v3/experiments/annotated_preference tests/training/grpo_v3/experiments/annotated_preference
git commit -m 'feat(training): add Pareto preference contract'
```

### Task 3：以 TDD 实现 compact_qa_v1 prompt 与序列化

**Files:**
- Create: `training/grpo_v3/experiments/annotated_preference/prompting.py`
- Create: `tests/training/grpo_v3/experiments/annotated_preference/test_prompting.py`

- [ ] **Step 1：写失败测试**

```python
def test_compact_completion_has_only_four_fields_in_fixed_order(self) -> None:
    text = serialize_compact_completion(candidate("e1::candidate_01"))
    self.assertEqual(
        list(json.loads(text)),
        ["question", "options", "correct", "answer"],
    )
    self.assertNotIn("formality_score", text)

def test_prompt_declares_video_roles_and_compact_schema(self) -> None:
    prompt = build_compact_generation_prompt()
    self.assertEqual(prompt.count("<video>"), 2)
    self.assertIn("first video is the Speaker", prompt)
    self.assertIn("second video is the Provider", prompt)
    self.assertIn('"question"', prompt)
    self.assertNotIn('"evidence"', prompt)
```

Prompt 还必须声明：只返回一个 JSON；恰好五个非空、互斥且同类型选项；`correct` 为 A–E；`answer` 等于正确选项；问题不包含姓名、时间码和 dataset/video/frame 语言。

- [ ] **Step 2：运行 RED 后实现**

```python
COMPACT_QA_CONTRACT = "compact_qa_v1"
PROMPT_REVISION = "annotated_pareto_compact_qa_v1"
COMPACT_GENERATION_PROMPT = """<video><video>
You receive two synchronized egocentric videos of the same interaction. The first
video is the Speaker view and the second video is the Provider view. Create one
grounded multiple-choice QA item answerable from the two videos together.

Return exactly one JSON object with fields in this order: "question", "options",
"correct", "answer". Provide exactly five non-empty, mutually exclusive options
of the same semantic type. Use A, B, C, D, or E in "correct"; "answer" must equal
the text of that option. Do not use names, timestamps, or dataset/video/frame
wording in the question. Return no markdown or commentary."""

def serialize_compact_completion(candidate: CandidateRecord) -> str:
    return json.dumps(
        candidate.model_features(),
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )

def build_compact_generation_prompt() -> str:
    return COMPACT_GENERATION_PROMPT

def prompt_sha256() -> str:
    return hashlib.sha256(COMPACT_GENERATION_PROMPT.encode("utf-8")).hexdigest()
```

- [ ] **Step 3：运行 GREEN 并提交**

```powershell
python -m unittest tests.training.grpo_v3.experiments.annotated_preference.test_prompting -v
git add training/grpo_v3/experiments/annotated_preference/prompting.py tests/training/grpo_v3/experiments/annotated_preference/test_prompting.py
git commit -m 'feat(training): add compact QA DPO prompt'
```

### Task 4：以 TDD 实现 train/validation JSONL 构建 CLI

**Files:**
- Create: `training/grpo_v3/experiments/annotated_preference/build_dataset.py`
- Create: `tests/training/grpo_v3/experiments/annotated_preference/test_build_dataset.py`

- [ ] **Step 1：写官方 ms-swift 格式失败测试**

```python
def test_build_row_matches_swift_multimodal_dpo_format(self) -> None:
    row = build_dpo_row(pair, evidence, media_map)
    self.assertEqual([x["role"] for x in row["messages"]], ["user", "assistant"])
    self.assertEqual(row["messages"][0]["content"].count("<video>"), 2)
    self.assertEqual(json.loads(row["messages"][1]["content"])["question"], "Chosen?")
    self.assertEqual(json.loads(row["rejected_response"])["question"], "Rejected?")
    self.assertEqual(row["videos"], ["/media/speaker.mp4", "/media/provider.mp4"])
    self.assertNotIn("rejected_videos", row)
```

训练 JSONL 仅保留 `messages/rejected_response/videos`。evidence、candidate IDs、F/E/A 和指纹进入平行 `pair_index.jsonl`，避免训练 preprocessor 消费审计字段。

- [ ] **Step 2：写 split/media/泄漏失败测试**

覆盖：

- split CSV SHA 不一致；
- train/validation overlap；
- locked-test 非空；
- media map 缺任一 URL、非本地绝对路径或两视频相同；
- validation evidence 出现在 train index；
- 输出行数、pair index 和审计计数不一致；
- 输出写入中断时旧文件不被部分覆盖。

- [ ] **Step 3：运行 RED 后实现**

```python
@dataclass(frozen=True)
class BuildOutputs:
    train_rows: Sequence[dict[str, Any]]
    validation_rows: Sequence[dict[str, Any]]
    train_index: Sequence[dict[str, Any]]
    validation_index: Sequence[dict[str, Any]]
    audit: dict[str, Any]

def build_dpo_row(
    pair: PreferencePair,
    evidence: EvidenceRecord,
    media_map: Mapping[str, str],
) -> dict[str, Any]:
    videos = [
        str(Path(media_map[evidence.video_a_source]).resolve()),
        str(Path(media_map[evidence.video_b_source]).resolve()),
    ]
    return {
        "messages": [
            {"role": "user", "content": build_compact_generation_prompt()},
            {"role": "assistant", "content": serialize_compact_completion(pair.chosen)},
        ],
        "rejected_response": serialize_compact_completion(pair.rejected),
        "videos": videos,
    }

def pair_index_row(pair: PreferencePair) -> dict[str, Any]:
    return {
        "evidence_id": pair.evidence_id,
        "chosen_candidate_id": pair.chosen.candidate_id,
        "rejected_candidate_id": pair.rejected.candidate_id,
        "chosen_scores": [
            pair.chosen.qa_formality,
            pair.chosen.evidence_quality,
            pair.chosen.answerability,
        ],
        "rejected_scores": [
            pair.rejected.qa_formality,
            pair.rejected.evidence_quality,
            pair.rejected.answerability,
        ],
        "chosen_fingerprint": pair.chosen_fingerprint,
        "rejected_fingerprint": pair.rejected_fingerprint,
    }
```

`build_outputs(csv_path, split_path, media_map_path)` 按以下固定数据流实现：

1. `load_annotation_csv(csv_path)`，读取 split JSON 后调用
   `validate_split_manifest(manifest, expected_counts=(60, 10, 0), require_contract=True)`；
2. 强制 `manifest["csv_sha256"] == audit.csv_sha256`，并将 70 个 split ID 逐一映射到
   `audit.eligible_evidence`，缺失或多余 ID 都报错；
3. 读取扁平 `URL -> 本地绝对路径` media map，逐个验证存在、非空、A/B 路径不同；
4. 分别按 manifest 中 train/validation evidence 顺序调用 `build_pareto_pairs`，每个 pair
   同时生成一个训练行和一个 `pair_index_row`，并验证两者计数一一对应；
5. 从 train 中按 `evidence_id` 字典序取前四个存在 pair 的 evidence 生成 overfit 子集；
6. 返回 `BuildOutputs`；CLI 先在同目录临时文件中写完并计算 SHA-256，全部成功后再以
   `Path.replace` 原子替换七个正式输出，任何异常不得覆盖旧产物。

CLI：

```powershell
python -m training.grpo_v3.experiments.annotated_preference.build_dataset build `
  --csv $env:CSV_PATH --split $env:SPLIT_PATH --media-map $env:MEDIA_MAP_PATH `
  --output-dir $env:DPO_DATA_DIR
```

原子输出：

```text
train_dpo.jsonl
validation_dpo.jsonl
train_pair_index.jsonl
validation_pair_index.jsonl
overfit_4_dpo.jsonl
pareto_audit.json
dataset_manifest.json
```

Manifest 记录输入与输出 SHA-256、prompt SHA、合同版本、精确 evidence/pair 计数。Overfit evidence 是 train split 内字典序最小的四个“至少有一个 pair”的 evidence。

- [ ] **Step 4：运行 GREEN；真实输入缺失时必须失败而非降级**

```powershell
python -m unittest tests.training.grpo_v3.experiments.annotated_preference.test_build_dataset -v
python -m training.grpo_v3.experiments.annotated_preference.build_dataset build `
  --csv 'C:\Users\20661\Documents\xwechat_files\wxid_i096w25uhusk22_e748\msg\file\2026-08\rlhf_candidate_scores_merged_70_packets.csv' `
  --split 'C:\Users\20661\Desktop\Research\AR\multiuser\EgoQA-two-user-annotated-pareto-dpo\data_RLHF\reviewer_v1\split_60_10.json' `
  --media-map 'C:\Users\20661\Desktop\Research\AR\multiuser\EgoQA-two-user-annotated-pareto-dpo\data_RLHF\reviewer_v1\media_map.json' `
  --output-dir 'C:\Users\20661\Desktop\Research\AR\multiuser\EgoQA-two-user-annotated-pareto-dpo\tmp\annotated_preference_build'
```

若本地没有 formal split 或 140 个本地媒体映射，Expected 是清晰的缺失输入错误；禁止回退到 URL、重分割或空数据。

- [ ] **Step 5：提交**

```powershell
git add training/grpo_v3/experiments/annotated_preference/build_dataset.py tests/training/grpo_v3/experiments/annotated_preference/test_build_dataset.py
git commit -m 'feat(training): build annotated Pareto DPO datasets'
```

### Task 5：以 TDD 实现训练指标和 Gate 摘要

**Files:**
- Create: `training/grpo_v3/experiments/annotated_preference/analyze.py`
- Create: `tests/training/grpo_v3/experiments/annotated_preference/test_analyze.py`

- [ ] **Step 1：写失败测试**

Fixture 包含 `loss/eval_loss/rewards/accuracies/rewards/margins/logps/chosen/logps/rejected/grad_norm`。断言：

```python
self.assertEqual(result["status"], "passed")
self.assertGreater(result["final_reward_margin"], 0)
self.assertGreater(result["final_pair_accuracy"], 0.5)
self.assertTrue(result["finite_metrics"])
```

另写缺 trainer state、NaN、无 eval 指标、LoRA delta 为零和非 LoRA delta 非零的失败测试。

- [ ] **Step 2：运行 RED 后实现 CLI**

```powershell
python -m training.grpo_v3.experiments.annotated_preference.analyze `
  --trainer-state $env:OUTPUT_DIR/trainer_state.json `
  --parameter-audit $env:OUTPUT_DIR/parameter_audit.json `
  --dataset-manifest $env:DPO_DATA_DIR/dataset_manifest.json `
  --mode smoke --output $env:OUTPUT_DIR/dpo_gate_result.json
```

Mode 合同：

- `smoke`：有限 loss/grad、LoRA delta 非零、非 LoRA delta 为零、checkpoint 存在；
- `overfit`：最终 margin 高于初始，pair accuracy 大于 0.8；
- `train`：训练步数非零且 validation 指标存在；
- `validation`：有限 eval loss/margin/accuracy，pair 数与 manifest 一致。

- [ ] **Step 3：运行 GREEN 并提交**

```powershell
python -m unittest tests.training.grpo_v3.experiments.annotated_preference.test_analyze -v
git add training/grpo_v3/experiments/annotated_preference/analyze.py tests/training/grpo_v3/experiments/annotated_preference/test_analyze.py
git commit -m 'feat(training): analyze Pareto DPO gates'
```

### Task 6：以 TDD 增加共享 Slurm preflight 与 Gate 0

**Files:**
- Create: `hpc/grpo_v3/annotated_preference/common.sh`
- Create: `hpc/grpo_v3/annotated_preference/gate0_data.sbatch`
- Create: `tests/training/grpo_v3/experiments/annotated_preference/test_slurm.py`

- [ ] **Step 1：写静态失败测试**

```python
self.assertIn("training.torch_storage_preflight", common)
self.assertIn("torchcodec.decoders import VideoDecoder", common)
self.assertIn("SLURM_JOB_ID", common)
self.assertIn("FPS_MAX_FRAMES", common)
self.assertIn("VIDEO_MAX_PIXELS", common)
self.assertNotIn("latest_", common)
```

Gate 0 必须运行 annotated-preference tests、`compileall`、builder，并逐个 `test -s` 七个输出文件。

- [ ] **Step 2：运行 RED 后实现 `common.sh`**

```bash
PROJECT_ROOT=${PROJECT_ROOT:-/scratch/xl6775/projects/EgoQA-two-user-annotated-pareto-dpo}
TRAIN_ENV=${TRAIN_ENV:-/scratch/xl6775/envs/egoqa-ms-swift-v4.2.2-vllm024}
MODEL_DIR=${MODEL_DIR:-/scratch/xl6775/models/Qwen3-VL-8B-Instruct}
DATA_DIR=${DATA_DIR:-${PROJECT_ROOT}/data_RLHF/annotated_preference}
OUTPUT_ROOT=${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/annotated_preference}
CSV_PATH=${CSV_PATH:-${DATA_DIR}/rlhf_candidate_scores_merged_70_packets.csv}
SPLIT_PATH=${SPLIT_PATH:-${DATA_DIR}/split_60_10.json}
MEDIA_MAP=${MEDIA_MAP:-${DATA_DIR}/media_map.json}
export FPS=${FPS:-1}
export FPS_MIN_FRAMES=${FPS_MIN_FRAMES:-4}
export FPS_MAX_FRAMES=${FPS_MAX_FRAMES:-64}
export VIDEO_MAX_PIXELS=${VIDEO_MAX_PIXELS:-50176}
```

全部 cache/temp 固定到 `${JOB_SCRATCH_ROOT}`；执行 storage preflight；同时设置 FFmpeg `PATH/LD_LIBRARY_PATH`；导入 `VideoDecoder`；输出目录来自真实 JobID。

- [ ] **Step 3：实现 Gate 0，运行 GREEN 并提交**

```powershell
python -m unittest tests.training.grpo_v3.experiments.annotated_preference.test_slurm -v
bash -n hpc/grpo_v3/annotated_preference/common.sh
bash -n hpc/grpo_v3/annotated_preference/gate0_data.sbatch
git add hpc/grpo_v3/annotated_preference tests/training/grpo_v3/experiments/annotated_preference/test_slurm.py
git commit -m 'feat(training): add Pareto DPO data preflight'
```

### Task 7：增加 Structure、Smoke 与 Overfit 作业

**Files:**
- Create: `hpc/grpo_v3/annotated_preference/structure_probe.sbatch`
- Create: `hpc/grpo_v3/annotated_preference/smoke1.sbatch`
- Create: `hpc/grpo_v3/annotated_preference/overfit_probe.sbatch`
- Modify: `tests/training/grpo_v3/experiments/annotated_preference/test_slurm.py`

- [ ] **Step 1：先写命令合同失败测试**

Smoke 必须含：

```text
swift rlhf
--rlhf_type dpo
--dataset train_dpo.jsonl
--val_dataset validation_dpo.jsonl
--tuner_type lora
--freeze_vit true
--freeze_aligner true
--target_modules q_proj v_proj
--lora_rank 8
--lora_alpha 16
--beta 0.1
--max_steps 1
--dataset_shuffle false
--seed 42
--data_seed 42
```

测试拒绝 reward plugin、reward function、GRPO、自动下载模型和 `latest_*`。

- [ ] **Step 2：实现 Structure Probe**

用一条 train pair 执行 ms-swift dataset/template 编码，写：

```text
storage_preflight.json
dependencies.txt
dataset_preview.json
structure_probe.json
```

Structure JSON 记录两 `<video>`、两本地视频、chosen/rejected 非空 labels、模型路径、ms-swift/torch/transformers/torchcodec 版本。

- [ ] **Step 3：实现 1-step Smoke**

固定首轮参数：BF16、`max_length=32768`、gradient checkpointing、单 H100、batch 1、accumulation 1、constant scheduler、learning rate `1e-5`、beta `0.1`。训练后严格检查 LoRA 名称集合、非零 delta、非 LoRA 零 delta与 checkpoint reload。

- [ ] **Step 4：实现 4-evidence Overfit**

输入 `overfit_4_dpo.jsonl`，默认 `MAX_STEPS=40`，其余参数与 Smoke 相同；只有 `analyze --mode overfit` 通过才允许正式训练。

- [ ] **Step 5：运行 GREEN 并提交**

```powershell
python -m unittest tests.training.grpo_v3.experiments.annotated_preference.test_slurm -v
bash -n hpc/grpo_v3/annotated_preference/structure_probe.sbatch
bash -n hpc/grpo_v3/annotated_preference/smoke1.sbatch
bash -n hpc/grpo_v3/annotated_preference/overfit_probe.sbatch
git diff --check
git add hpc/grpo_v3/annotated_preference tests/training/grpo_v3/experiments/annotated_preference/test_slurm.py
git commit -m 'feat(training): add Pareto DPO smoke gates'
```

### Task 8：增加正式训练与 validation 作业

**Files:**
- Create: `hpc/grpo_v3/annotated_preference/train.sbatch`
- Create: `hpc/grpo_v3/annotated_preference/evaluate.sbatch`
- Modify: `tests/training/grpo_v3/experiments/annotated_preference/test_slurm.py`

- [ ] **Step 1：写前置 Gate 失败测试**

`train.sbatch` 必须读取显式 `OVERFIT_RESULT` 并断言 `status=passed`；训练只读取 `train_dpo.jsonl`，validation 只通过 `--val_dataset validation_dpo.jsonl`。输出必须为 `${OUTPUT_ROOT}/train_${SLURM_JOB_ID}`。

- [ ] **Step 2：实现正式训练**

```bash
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
BETA=${BETA:-0.1}
```

保存 `resolved_command.txt/environment.txt/dataset_manifest.json/trainer_state.json/adapter/parameter_audit.json/dpo_gate_result.json`。这些是首轮 Gate 参数，不是最优超参数声明。

- [ ] **Step 3：实现独立 validation**

显式接收 `ADAPTER_DIR` 和 `TRAIN_JOB_ID`，禁止搜索 `latest_*`；输出 `${OUTPUT_ROOT}/validation_${SLURM_JOB_ID}`，只报告固定 pair accuracy/margin/eval loss，不宣称自由生成改善。

- [ ] **Step 4：运行 GREEN 并提交**

```powershell
python -m unittest tests.training.grpo_v3.experiments.annotated_preference.test_slurm -v
bash -n hpc/grpo_v3/annotated_preference/train.sbatch
bash -n hpc/grpo_v3/annotated_preference/evaluate.sbatch
git diff --check
git add hpc/grpo_v3/annotated_preference tests/training/grpo_v3/experiments/annotated_preference/test_slurm.py
git commit -m 'feat(training): add Pareto DPO train and validation jobs'
```

### Task 9：编写可复制 Torch Runbook

**Files:**
- Create: `training/grpo_v3/experiments/annotated_preference/TORCH_RUNBOOK_CN.md`
- Create: `tests/training/grpo_v3/experiments/annotated_preference/test_runbook.py`

- [ ] **Step 1：写 Runbook 失败测试**

要求包含：独立 worktree/branch、CSV SHA、`60/10/0`、路径、环境变量、Gate 0–5、`sbatch --parsable`、`squeue/sacct`、JobID 产物、失败收集、`compact_qa_v1` 边界和 Gate 6 未执行声明。

用户粘贴到 SSH 登录 shell 的命令不得含：

```text
exit
logout
exec
|| exit 1
set -e
set -euo pipefail
```

Sbatch 内可使用 `set -euo pipefail`；交互命令失败时打印 `MISSING/STOP` 并跳过依赖步骤，保留会话。

- [ ] **Step 2：运行 RED 后写完整 Runbook**

每个 Gate 提供：作用域/资源、输入 SHA、preflight、提交、监控、JobID 输出、验收断言、失败日志、最小下一步、能证明与不能证明的内容。

- [ ] **Step 3：运行 GREEN 并提交**

```powershell
python -m unittest tests.training.grpo_v3.experiments.annotated_preference.test_runbook -v
git add training/grpo_v3/experiments/annotated_preference/TORCH_RUNBOOK_CN.md tests/training/grpo_v3/experiments/annotated_preference/test_runbook.py
git commit -m 'docs(training): add Pareto DPO Torch runbook'
```

### Task 10：完整验证与交付

**Files:**
- Verify all files above

- [ ] **Step 1：运行新实验全套测试**

```powershell
python -m unittest discover -s tests/training/grpo_v3/experiments/annotated_preference -p 'test_*.py' -v
```

Expected：全部通过，无 skip/error/warning。

- [ ] **Step 2：运行 Reviewer v1 回归**

```powershell
python -m unittest discover -s tests/training/grpo_v3/experiments/human_preference_reviewer/v1 -p 'test_*.py' -v
```

- [ ] **Step 3：编译、shell 与 diff 检查**

```powershell
python -m compileall -q training/grpo_v3/experiments/annotated_preference
bash -n hpc/grpo_v3/annotated_preference/common.sh
bash -n hpc/grpo_v3/annotated_preference/gate0_data.sbatch
bash -n hpc/grpo_v3/annotated_preference/structure_probe.sbatch
bash -n hpc/grpo_v3/annotated_preference/smoke1.sbatch
bash -n hpc/grpo_v3/annotated_preference/overfit_probe.sbatch
bash -n hpc/grpo_v3/annotated_preference/train.sbatch
bash -n hpc/grpo_v3/annotated_preference/evaluate.sbatch
git diff --check
git status --short
```

- [ ] **Step 4：规格覆盖审计**

确认：无 CSV 在线 lookup reward；无 `display_order` 排名；无同分/不可比较 pair；无 train/validation 泄漏；无扩展 schema 伪造；无 locked-test 或自由生成过度声明。

- [ ] **Step 5：仅在验证产生必要修正时提交**

```powershell
git add training/grpo_v3/experiments/annotated_preference tests/training/grpo_v3/experiments/annotated_preference hpc/grpo_v3/annotated_preference
git commit -m 'test(training): verify annotated Pareto DPO workflow'
```

若无改动，不创建空提交。

## 证据等级

本地执行完成只证明数据、Pareto、DPO JSONL、Slurm 和 Runbook 合同通过静态/单元测试。远端 Gate 分别证明双视频编码、真实梯度、LoRA 更新、小样本可学习、60-evidence 作业和 10-evidence 固定偏好指标。即使 Gate 5 通过，也不能替代 Gate 6 的新 QA 自由生成盲评。
