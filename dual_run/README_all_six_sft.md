# full120 四卡 GRPO：all_six_sft 奖励

分支 `xth-dual-joint`，worktree `/scratch/tx856/multi_user_qa/EgoQA-joint`；运行目录 `/scratch/tx856/multi_user_qa/dual_full120`。
`run_all_six_sft/` 是实际提交所用 sbatch 与配置的副本（运行时读取的是运行目录里的同名文件）。

## 奖励

`R = 0.2·F + 0.4·G + 0.4·A`，`reward_mode = all_six_sft`

| 项 | Judge adapter | prompt |
|---|---|---|
| F formality | `adapters/formality_checkpoint-504`（r64，层16–63） | `prompts.build_qa_formality_judge_prompt`，与SFT文本逐字一致，见 `fixtures/formality_judge_sft_prompt.txt` |
| G groundedness | `adapters/groundedness_epoch4` | `prompts.build_evidence_groundedness_judge_prompt` |
| A answerability | `adapters/answerability_checkpoint-340`（r128，层56–63） | `fixtures/all_six_answerability_judge_prompt.txt` |

A 只判断“六人合并的时间线是否足以回答”。prompt 由 340 SFT 的联合模板（`fixtures/joint_answerability_judge_sft_prompt.txt`，
逐字取自 `answerability_842_question_split/manifests_video_split/train/answerability_joint.jsonl`）删去“speaker 单独不足”条件及两行相关说明得到；
340 没有用这个精确文本训练过。不再单独打 speaker_only。

## 设置

4×H200，每 rank 自己的 policy 和冻结 Judge 同卡交替；每步 4 个 video window（每卡 1 个，G4），GA4；
LoRA r128/α128/dropout0.05，q_proj/v_proj/in_proj_qkv，视觉冻结；lr 1e-5 cosine_with_min_lr 0.1，warmup 10，beta 0.04；
调度 900 步，stage1 停在 360；每 2 步存断点，保留 3 个。sbatch 内存 1200G（集群上限 300G/GPU）。

## 运行命令

```bash
cd /scratch/tx856/multi_user_qa/dual_full120

# 单测（只跑 dual_run 下的模块；不要 discover，multi-user 软链会把整个仓库的测试带进来）
cd /scratch/tx856/multi_user_qa/EgoQA-joint/dual_run
PYTHONPATH=$PWD/multi-user:$PWD /scratch/tx856/.conda/envs/grpo-swift/bin/python -m unittest \
  $(ls test_*.py | sed 's/\.py$//')
cd /scratch/tx856/multi_user_qa/dual_full120

# 4卡 smoke（2 步）
sbatch --parsable --time=06:00:00 \
  --export=ALL,RUN_CONFIG=/scratch/tx856/multi_user_qa/dual_full120/configs/smoke4_allsix_sft.json \
  quad_allsix_sft.sbatch

# 正式 stage1（0→360，调度 900）
sbatch --parsable \
  --export=ALL,RUN_CONFIG=/scratch/tx856/multi_user_qa/dual_full120/configs/formal4_allsix_sft_stage1.json \
  quad_allsix_sft.sbatch
```

每次提交把 JobID 记入 `submissions/submissions.txt`。日志：`logs/quad_allsix_sft_<JobID>.out`，
内存 `logs/hostmem_<JobID>.log`，训练 `outputs/<run>/train_<JobID>/trainer.log` 与 `swift/*/logging.jsonl`。
stage2/3 用断点续跑（`resume_from_checkpoint` 等字段 + CPU audit），不能改任何训练设置。

## 基座 vs LoRA 问题通过率

验证集前 100 条输入（未参与训练），基座和 LoRA 各生成 1 题（同输入、同种子、同训练采样设置），
三个冻结 Judge 打分；每个 Judge 以 `P(pass) > 0.5` 为通过（与 Judge SFT 评估的判定一致），未被打分的无效生成算不通过。
结果写到 `outputs/eval_passrate/eval_<JobID>/passrate_summary.json`。

```bash
cd /scratch/tx856/multi_user_qa/dual_full120
sbatch --parsable --export=ALL,RUN_CONFIG=/scratch/tx856/multi_user_qa/dual_full120/configs/formal4_allsix_sft_stage1.json,ADAPTER=/scratch/tx856/multi_user_qa/dual_full120/eval_ckpts/step204_20161348,COUNT=100 \
  eval_passrate.sbatch
```

## 断点续跑（20161348 在第215步 CUDA OOM 退出，从 checkpoint-214 续跑）

```bash
cd /scratch/tx856/multi_user_qa/dual_full120
S=$PWD/outputs/formal4_allsix_sft/train_20161348
# 1) CPU核验断点（登录节点会被kill，用CPU作业）：adapter/优化器有限、160个状态step=214、lr=原调度第214步、4个rng
sbatch --parsable --job-name=resume_audit --account=torch_pr_111_tandon_advanced --cpus-per-task=8 --mem=64G --time=01:00:00 \
  --output=$PWD/logs/resume_audit_%j.out --wrap "/scratch/tx856/.conda/envs/grpo-swift/bin/python \
  /scratch/tx856/multi_user_qa/EgoQA-joint/dual_run/resume_audit.py --checkpoint $S/swift/v0-20261017-035106/checkpoint-214 \
  --config $S/run_config.json --output $PWD/resume/cpu_audit_20161348_step214.json"
# 2) 续跑配置 = 原 run_config.json + resume_from_checkpoint/resume_source_output/recovery_from_job/expected_resume_step/
#    resume_cpu_audit/run_attempt（见 run_all_six_sft/formal4_allsix_sft_stage1_resume214.json；其余训练设置不变，recovery.py会拒绝任何改动）
RUN_CONFIG=$PWD/configs/formal4_allsix_sft_stage1_resume214.json sbatch --parsable --export=ALL,RUN_CONFIG quad_allsix_sft.sbatch
```
