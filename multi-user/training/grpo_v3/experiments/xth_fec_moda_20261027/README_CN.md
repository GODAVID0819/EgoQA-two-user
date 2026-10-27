# lxr F/E/C GRPO 训练 + 三轮 Judge 反馈环生成（xth 版）

本目录在 `lxr-grpo` 分支的 `full120x6_fec_moda_stage180_20261021`（正式运行 20573337 的源码快照）上做最小改动，用于：

1. 用与 lxr **相同的代码和超参数**训练六用户 QA 生成器（GRPO），质量奖励可选两种；
2. 用 lxr 的三个 Judge prompt（C 为 “both requirements” 联合 answerability）跑三轮反馈环生成问题。

提交 `aee568b` 是 lxr runtime 的原样拷贝，`git diff aee568b -- .` 即为全部改动。

## 与 lxr 的差别

| 项 | lxr（20573337） | 本目录 |
|---|---|---|
| 质量奖励 | `continuous`：0.2F + 0.4E + 0.4C | `reward_mode` 可选 `continuous`（不变）或 `continuous_gated`：0.2F + 0.4E + 0.4C·[E>0.5] |
| 多样性奖励 | +0.5·g·d，g = F/E/C 都 ≥0.5 | 不变（两种质量奖励都加） |
| 固定历史池 | 旧300步实验的251题（`/scratch/xl6775`，无读权限） | 我们 step204 生成器三轮反馈环中**被接受**的题（F/E/all-six 三个 Judge P>0.5），同样 seed42、每窗最多8、不补满：894题、119窗有题；score_source=`xth_fbloop_step204_F_E_A` |
| 训练数据 | 600训练+120原验证，按 `condition_definition.json` 长度配对顺序 | 我们的 `dual_full120/data/trainval/{train,validation}.jsonl`（同120窗口、同speaker划分），按同一 `condition_definition.json` 顺序排列；720行全部通过本 runtime 的 `data.validate_row`（prompt、图片与当前packet逐项一致） |
| Judge | F504 / E epoch4 / C340，lxr 的 `prompts/*.txt` | 同 adapter、同 prompt 文件 |
| 环境 | lxr 私有 conda / 编译器 / 加速包 | `/scratch/tx856/.conda/envs/grpo-swift`，`policy_cuda_home=/scratch/tx856/.conda/cuda130_home`，无额外加速包 |
| 其余超参 | 2×H200、r64/α128/dropout0.05、lr 1e-5 cosine_with_min_lr、warmup10、β0.04、G4、GA4、50176像素、上下文131072、completion 2048、temperature 0.85/top_p 0.95/top_k 40、stop 180 / 调度 360 | 完全相同（`make_xth_configs.py` 从 `examples/train_config_20573337.json` 只改路径、reward_mode、历史池） |

新增：`predictor.explain_many` 与 Judge 服务 `/explain_batch`（只为判 fail 的 Judge 生成反馈文本，不影响打分路径）、`feedback_loop.py`、`eval_feedback_loop.py`（2 或 4 卡）。

## 文件

| 文件 | 作用 |
|---|---|
| `runtime/` | 运行源码（lxr runtime + 上述改动） |
| `runtime/multi-user/training/grpo_v3/six_user_binary/reward.py` | `continuous` / `continuous_gated` |
| `runtime/.../diversity.py`、`plugin.py`、`launch.py`、`validate_run.py` | 按 `reward_mode` 计算/校验质量项 |
| `runtime/.../history_pool.py` | 新增 xth 历史池来源（`save_xth_history`，`load_history` 接受两种来源） |
| `runtime/build_xth_history.py` | 从反馈环输出构建固定历史池（CPU） |
| `runtime/prepare_xth_720.py` | 我们的720行按 lxr 顺序排列并逐行 `validate_row`（CPU） |
| `runtime/.../service.py`、`predictor.py`、`feedback_loop.py`，`runtime/eval_feedback_loop.py` | 三轮反馈环 |
| `runtime/test_xth_options.py` | 新增部分的单测 |
| `materialize_root.sh` | 按 lxr README 的“独立根”布局物化运行目录 |
| `make_xth_configs.py` | 在运行根写 `configs/`：`train_continuous.json`、`train_continuous_gated.json`、`smoke_capacity.json`、`judge_fec.json`、`fbloop.json` |
| `slurm/train_dual.sbatch`、`slurm/fbloop.sbatch` | 提交脚本 |

## 一次性准备（Torch）

```bash
REPO=/scratch/tx856/multi_user_qa/EgoQA-lxr          # 本分支的 worktree
EXP=$REPO/multi-user/training/grpo_v3/experiments/xth_fec_moda_20261027
ROOT=/scratch/tx856/multi_user_qa/lxr_run             # 运行根
P=/scratch/tx856/.conda/envs/grpo-swift/bin/python

# 1. 物化运行根（仓库 multi-user、hpc + runtime 覆盖；要求工作区已提交；有作业在跑时不要重跑）
bash $EXP/materialize_root.sh $ROOT && mkdir -p $ROOT/logs
$P $EXP/make_xth_configs.py $ROOT

# 2. 模型目录名必须是 Qwen3.8-27B（lxr predictor 用目录名核对 Judge adapter 的基座）：逐文件链接到 HF 快照
S=/scratch/hm2991/egolife_vllm_runtime/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
mkdir -p /scratch/tx856/models/Qwen3.8-27B && for f in $S/*; do ln -sfn "$(readlink -f $f)" /scratch/tx856/models/Qwen3.8-27B/$(basename $f); done

# 3. 多样性编码器（与 lxr 同 revision 的 MiniLM ONNX）与 onnxruntime
B=https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/1110a243fdf4706b3f48f1d95db1a4f5529b4d41
mkdir -p $ROOT/encoder/onnx && for f in config.json tokenizer.json tokenizer_config.json special_tokens_map.json vocab.txt onnx/model.onnx; do curl -sfL -o $ROOT/encoder/$f $B/$f; done
$P -m pip install --no-deps --target $ROOT/runtime_deps_train onnxruntime==1.31.0 coloredlogs humanfriendly flatbuffers

# 4. 数据与历史池（CPU作业，约30分钟；读全部帧文件）
D=/scratch/tx856/multi_user_qa/dual_full120; FB=$D/outputs/feedback_loop
sbatch --account=torch_pr_111_tandon_advanced --cpus-per-task=8 --mem=32G --time=02:00:00 --wrap "
  export CUDA_VISIBLE_DEVICES= PYTHONPATH=$ROOT/runtime_deps_train:$ROOT/multi-user:$ROOT; cd $ROOT
  $P prepare_xth_720.py --config configs/train_continuous.json --train $D/data/trainval/train.jsonl --validation $D/data/trainval/validation.jsonl --validate-media
  $P build_xth_history.py --config configs/train_continuous.json --runs $FB/fbloop_20772096 $FB/fbloop_20819555 $FB/fbloop_21073148 $FB/fbloop_21073149"
```

单步容量 smoke 还需要 `$ROOT/data/train_capacity2.jsonl`：`train_length_paired.jsonl` 里图片最多的一对（同一更新的）相邻条件，当前为 0 起第 126、127 行，各 1800 帧：

```bash
$P -c "
import json; rows=[json.loads(l) for l in open('$ROOT/data/train_length_paired.jsonl')]
i=max(range(0,720,2),key=lambda i:len(rows[i]['images'])+len(rows[i+1]['images']))
open('$ROOT/data/train_capacity2.jsonl','w').write(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows[i:i+2]))"
```

## 单测（CPU，约20秒）

```bash
cd $EXP && $P -B run_cpu_regressions.py      # lxr 原有回归 + test_xth_options，共61项
```

## GRPO 训练

```bash
# 单步 smoke（最长一对 1800+1800 帧，跑一次完整更新并做 lxr validate_run 全部检查）
sbatch --time=04:00:00 --export=ALL,RUN_CONFIG=$ROOT/configs/smoke_capacity.json $EXP/slurm/train_dual.sbatch

# 正式：任选一种质量奖励（各 2×H200、750G；stop_after_steps=180，调度 360）
sbatch --export=ALL,RUN_CONFIG=$ROOT/configs/train_continuous.json       $EXP/slurm/train_dual.sbatch
sbatch --export=ALL,RUN_CONFIG=$ROOT/configs/train_continuous_gated.json $EXP/slurm/train_dual.sbatch
```

- 入口是 lxr 的 `dual_launch.py`：每卡一个冻结 Judge 服务（与 Policy 同卡交替显存），两 rank DDP，每步两个不同窗口各 G4。
- 输出 `$ROOT/outputs/<reward_mode>/train_<JobID>/`：`swift/v0-*/checkpoint-*`（每2步存、保留3个）、`rank*/reward_trace.jsonl`（每条候选的 F/E/C 概率、quality_reward、diversity_distance/bonus、reward）、`rank*/group_metrics.jsonl`、`run_manifest.json`；结束时自动跑 `validate_run`。
- lxr 实测约 450 秒/步，180 步约 22–25 小时；sbatch 墙钟 47 小时（用户 >48h 作业限 4 卡）。
- **180→360 续跑**：复制原配置，改 `stop_after_steps: 360`，加 `resume_from_checkpoint`（`.../checkpoint-180`）、`resume_source_output`（`.../train_<原JobID>`）、`recovery_from_job`、`expected_resume_step: 180`，其余设置不得改变（`recovery.validated_resume` 会逐项核对，Judge 配置也须相同）。

## 三轮 Judge 反馈环生成

```bash
# ADAPTER：任一生成器 LoRA。HF 上 lxr 的 cp348：
#   https://huggingface.co/SherryDragon/egolife-grpo-v6-generator-cp348-r64 → /scratch/tx856/multi_user_qa/dual_full120/eval_ckpts/lxr_cp348_20857619
A=/scratch/tx856/multi_user_qa/dual_full120/eval_ckpts/lxr_cp348_20857619
# smoke（2卡）
sbatch --gres=gpu:h200:2 --mem=600G --time=04:00:00 \
  --export=ALL,RUN_CONFIG=$ROOT/configs/fbloop.json,ADAPTER=$A,COUNT=2,SAMPLES=2,ATTEMPTS=2 $EXP/slurm/fbloop.sbatch
# 正式（4卡）：前 COUNT 条验证输入（每窗口原验证 speaker）× SAMPLES 题，最多 ATTEMPTS 轮
sbatch --export=ALL,RUN_CONFIG=$ROOT/configs/fbloop.json,ADAPTER=$A,COUNT=100,SAMPLES=3,ATTEMPTS=3 $EXP/slurm/fbloop.sbatch
# 训练完自己的 checkpoint 后，把 ADAPTER 换成 $ROOT/outputs/<mode>/train_<JobID>/swift/v0-*/checkpoint-<step>
```

- 流程：生成 → 三个 Judge 取 verdict token 的 P(pass)，各自 >0.5 算通过，**三项都过才接受** → 判 fail 的 Judge 在 `{"verdict":"fail` 后贪心续写 reason/fix（C 为 reason + missing_evidence；groundedness 不写 reason 时预填后再续写一次）→ 拼成 `"{check}: {reason} {fix}"` 用原流水线 `_feedback_block` 插回 prompt 重生成。
- 生成设置与训练一致：50176 像素、上下文 131072、temperature 0.85 / top_p 0.95 / top_k 40、最多 2048 token；每题种子逐题记录。
- 换输入：把 `fbloop.json` 的 `val_dataset` 换成其他 jsonl（行格式同训练数据），`validation_seed` 换新种子。`COUNT` 须能被卡数（2 或 4）整除。
- 续跑：`RESUME_DIR=$ROOT/outputs/feedback_loop/fbloop_<JobID>`，其余参数与原运行相同。
- 输出 `$ROOT/outputs/feedback_loop/fbloop_<JobID>/`：`rank*/feedback_loop.json`（每题每轮的 seed、收到的反馈、原始输出、三个 Judge 概率/是否通过、Judge 原始续写、给下一轮的反馈）、`feedback_loop_summary.json`（每轮结束后各 Judge 通过率与各轮接受数）。

## 已验证

| JobID | 内容 | 结果 |
|---|---|---|
| 21141084 | CPU：720行按 lxr 顺序排列并逐行 `validate_row`；构建历史池 | 720/720 通过；历史池 894 题 |
| 21141156 | 反馈环 smoke，HF cp348，2输入×2题×2轮 | 完成；C 的反馈会明确指出 “speaker 自己的时间线已足够回答” |
| 21145082 | GRPO 单步 smoke，`continuous_gated`，最长一对 | lxr `validate_run` 25 项全部通过（含按 gated 公式重算的 reward 分量） |

这些只证明工程链路可用；Judge 代理通过率不等于人工正确率。
