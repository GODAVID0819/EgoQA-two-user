# 六用户高质量 Generator 单次 30 槽 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 复用 Job `16699348` 的三组 speaker candidate asset，以高质量 generator 媒体固定生成 30 个 QA 槽，并完整记录一次生成后的 judge 结果。

**Architecture:** 新增一个纯数据模块，从三个大型 group asset 逐个恢复 18 个紧凑 speaker packet，并预展开为每组 10 槽、全局每个 speaker 5 槽的 30 行 JSONL。扩展 Qwen runner 的单次调用媒体 profile，使同一已加载模型在 generator 使用 `0.5/131072`、judge 使用 `0.25/65536`；复用现有 parallel judge、两次同步 facts 的 Answerability 和简单 Evidence。新增正式 wrapper 通过现有 runtime 入口跳过 mining，使用固定 30 分母汇总。

**Tech Stack:** Python 3、JSONL、现有 `video_qa_loop.py`、Qwen3-VL Transformers memory-safe runner、Bash/Slurm、现有 pytest。

---

### Task 1: 恢复紧凑 speaker packet 并生成固定 30 槽输入

**Files:**
- Create: `one_pass_evidence.py`
- Create: `tools/build_six_user_one_pass_evidence.py`
- Create: `tests/test_six_user_one_pass.py`

- [ ] **Step 1: 写失败测试，锁定恢复和轮转合同**

在 `tests/test_six_user_one_pass.py` 放入最小合成 asset，验证 `compact_speaker_packets()` 从一个 group asset 的六个 `speaker_candidates` 产生六个不同 `speaker_index`，输出保留六路 `clips`、`full_local_video`、`generator_local_video`、`temporal_pruning` 和原始时间映射，并不保留 `similarity_matrix`、`block_diagnostics` 等大诊断字段。

同时验证 `expand_one_pass_slots()` 接收三个 group 的六个 packet 后返回 30 行：每组 10 行、每行唯一 `generation_slot_id`、每个 speaker 全局 5 行，且三组顺序为规格中固定的 DAY1、DAY3、DAY4 轮转顺序。

运行：`python -m pytest tests/test_six_user_one_pass.py -q`

预期：FAIL，模块或函数尚未存在。

- [ ] **Step 2: 实现最小数据模块**

在 `one_pass_evidence.py` 实现以下公开接口：

```python
def compact_speaker_packets(asset_path: str | Path, *, source_job_id: str) -> list[dict[str, Any]]: ...
def expand_one_pass_slots(packets_by_group: Iterable[dict[str, Any]], *, slots_per_group: int = 10) -> list[dict[str, Any]]: ...
def write_one_pass_evidence(asset_paths: Iterable[str | Path], *, compact_output: str | Path, expanded_output: str | Path, source_job_id: str) -> dict[str, Any]: ...
```

实现要求：逐个读取 asset；只接受 `speaker_candidates` 中状态成功、索引为 `0..5` 的六个 candidate；用 candidate 的 `selected_clips` 构造已有六用户 packet 形状；为重复使用的 packet 生成新的 `evidence_id`、`base_evidence_id`、`generation_slot_id`，但保持 `generation_group_id` 和视频身份不变；发现 group 不完整或视频路径缺失时抛出带 group/speaker/path 的错误。

固定轮转起点为：DAY1 的 speaker index `0`、DAY3 的 `4`、DAY4 的 `2`；每组按六人各一槽后，再给前四个轮转 speaker 一槽。输出 provenance 指向 Job `16699348` 和实际 asset 文件名。

- [ ] **Step 3: 实现窄 CLI 并运行测试**

`tools/build_six_user_one_pass_evidence.py` 只负责解析 `--asset`（可重复三次）、`--compact-output`、`--expanded-output`、`--source-job-id`，调用模块并打印 `one_pass_evidence_ready groups=3 packets=18 slots=30`。不扫描其他目录、不复制视频、不运行 mining。

运行：`python -m pytest tests/test_six_user_one_pass.py -q`

预期：PASS。

### Task 2: 为同一模型实例增加阶段级视频质量 profile

**Files:**
- Modify: `qwen3vl_runner.py`
- Modify: `video_qa_loop.py`
- Modify: `tests/test_six_user_video_qa_loop.py`

- [ ] **Step 1: 写失败测试，验证 profile 和单次内容参数**

为 `GenerationCallProfile` 增加可选 `video_fps`、`max_image_pixels` 后，测试 one-pass profile 的 generator 为 `0.5/131072`，judge 为 `0.25/65536`，两者均关闭 thinking；用不加载模型的 fake runner 检查 `generate_with_call_profile()` 将 profile 传到 runner。

运行：`python -m pytest tests/test_six_user_video_qa_loop.py -q -k "profile or media"`

预期：FAIL，当前 profile 没有阶段级媒体字段。

- [ ] **Step 2: 实现 profile 覆盖和 memory-safe 缓存隔离**

在 `qwen3vl_runner.py` 中让普通 runner 根据当前 `call_profile` 生成视频 content 的 `fps`/`max_pixels`；让 memory-safe runner 在转码和缓存 key 中使用当前 profile 的 fps，并对请求值执行不超过 runner 上限的约束。runner 本体仍只实例化一次，默认行为保持原样。

在 `video_qa_loop.py` 新增 one-pass stage profile 工厂，并由 `generate_video_qa_loop(..., six_user_one_pass_profile=True)` 选择它；generator 调用使用高质量 profile，现有 `run_parallel_review_judges()` 的 Evidence、两次 Answerability 和 formality 使用 judge profile。`run_answerability_eval()` 继续先生成 speaker facts，再将同一 `canonical_facts` 传给 all-six。

- [ ] **Step 3: 运行针对性测试**

运行：`python -m pytest tests/test_six_user_video_qa_loop.py -q -k "profile or answerability or evidence"`

预期：PASS；旧 fast/reasoning profile 测试保持通过。

### Task 3: 接入固定 30 槽 one-shot loop 和汇总

**Files:**
- Create: `tools/summarize_six_user_one_pass.py`
- Modify: `video_qa_loop.py`
- Modify: `hpc/qa/smoke/run_six_user_qa_runtime_probe.sbatch`
- Modify: `tests/test_six_user_one_pass.py`
- Modify: `tests/test_ten_minute_reasoning_job_contract.py`

- [ ] **Step 1: 写失败测试，锁定 one-shot 行为**

用 fake generator 让第一个槽输出合法 QA、第二个槽输出非法 JSON、第三个槽被 judge 拒绝，调用 one-pass loop 入口，验证三个槽均有记录、generator 调用次数等于槽数、`MAX_ATTEMPTS=1` 不重试，且汇总分母使用预展开槽数而不是 accepted 数量。

- [ ] **Step 2: 实现 loop 参数和输出身份**

新增 `--six-user-one-pass-profile` 及对应 CLI 传递；one-pass wrapper 设置 `target_count=30`、`max_attempts=1`、`fixed_question_type_schedule=True`、`question_types=neutral`，传入预展开 30 行 evidence，不使用 `repeat_evidence`。每个槽继续写 `video_first_prompts.jsonl`、`qa_mcq.attempts.jsonl`、accepted/rejected/intermediate；生成、解析、每个 judge 和槽总耗时写入已有 trace，并补充 one-pass profile 字段。

将 one-pass 的失败分类明确映射为 `parse_failed`、`rejected_by_formality`、`rejected_by_evidence`、`rejected_by_answerability` 或运行失败；judge 失败不再设置 generator feedback retry。

- [ ] **Step 3: 实现 one-pass 汇总并接入 runtime 分支**

`tools/summarize_six_user_one_pass.py` 读取 compact/expanded evidence、accepted、rejected、attempts 和 prompts，按 `generation_slot_id` 合并，检查 30 个槽均有最终状态，输出 `six_user_qa_result.json`，以 30 为固定分母，给出 group/speaker/失败类型/耗时统计。one-pass 分支跳过旧的 accepted-target 和“每个 speaker 至少两槽”判定；普通任务继续使用原分支。

在 runtime 脚本中，当 `ONE_PASS_30_SLOT_MODE=1` 时跳过 `build_manifest`、CLIP mining 和 repeat-evidence deadline 分支，先调用 Task 1 CLI 生成 `${OUTDIR}/six_user_candidates.jsonl` 与 `${OUTDIR}/one_pass_evidence.jsonl`，再调用现有 QA loop 与 one-pass 汇总。保留当前 job-specific HOME/cache/TMP、FFmpeg、CUDA keeper 和 JobID 输出目录逻辑。

- [ ] **Step 4: 运行最小本地验证**

运行：`python -m pytest tests/test_six_user_one_pass.py tests/test_six_user_video_qa_loop.py -q`

预期：新增合同与已有 answerability/evidence/loop 测试通过；不运行模型、不启动 GPU。

### Task 4: 新增正式 wrapper，完成登录节点零 GPU 验证后直接提交

**Files:**
- Create: `hpc/qa/experiments/run_six_user_qa_10min_one_pass_30.sbatch`
- Modify: `tests/test_ten_minute_reasoning_job_contract.py`

- [ ] **Step 1: 写 wrapper 合同测试**

验证新 wrapper 使用新 run mode、Job `16699348` 的三个明确 asset 路径、`ONE_PASS_30_SLOT_MODE=1`、`MAX_GENERATION_SLOTS=30`、`MAX_ATTEMPTS=1`、`EXPECTED_QA_PER_GROUP=10`、`FAIL_FAST_REVIEW=0`、generator `QWEN_MEMORY_SAFE_VIDEO_FPS=0.5`、`QWEN_MEMORY_SAFE_MAX_IMAGE_PIXELS=131072`，不含 `--nodelist`、`-w` 或旧 Job 输出覆盖。

- [ ] **Step 2: 写正式 wrapper**

基于当前 fast-fix wrapper 的资源和 job-specific scratch 结构，使用新的唯一 `RUN_MODE` 和输出目录；保留 `#SBATCH --time=2-00:00:00`、`#SBATCH --gres=gpu:1`、H100 constraint、96G 内存。显式设置并导出 `CUDA_KEEPER_ENABLE=1`、`${PROJECT_ROOT}/hpc/shared/cuda.py`、分配 GPU、reserve、max prealloc 和 `CUDA_KEEPER_START_AFTER_SECONDS=7200`，由现有 runtime 的 `trap` 清理 keeper。

- [ ] **Step 3: 代码完成后做一次必要的本地/登录节点验证**

本地只运行 Task 3 的针对性 pytest 和 wrapper shell 语法检查；随后按 Torch 共享连接 SOP 连接并在登录节点执行不触碰 GPU 的导入、CLI help、三 asset 路径/JSON 结构、30 槽展开和 wrapper 静态合同检查。验证失败先修复并重新执行同一检查，不增加 SHA、GPU smoke 或额外提交前作业。

- [ ] **Step 4: 零 GPU 验证通过后提交正式作业**

正式提交使用 `sbatch --parsable`，立即把返回 JobID 写入时间戳 manifest；不提交 SHA probe/smoke，不取消任何既有 Job。提交后只用新 JobID 派生的 stdout、stderr、manifest 和输出目录跟踪生成结果。
