#!/usr/bin/env bash
#SBATCH --job-name=egoqa-am-p40
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:2
#SBATCH --mem=96G
#SBATCH --time=18:00:00
#SBATCH --output=logs/grpo-v3-answer-margin-probe40-%j.out
#SBATCH --error=logs/grpo-v3-answer-margin-probe40-%j.err
# 本 Gate 复用同目录 smoke1 的完整 scratch-first 执行体；下列锁定项供提交前静态审计。
# export HOME= XDG_CACHE_HOME= HF_HOME= HF_DATASETS_CACHE= MODELSCOPE_CACHE= TORCH_HOME=
# export TRITON_CACHE_DIR= TORCHINDUCTOR_CACHE_DIR= VLLM_CACHE_ROOT= CUDA_CACHE_PATH=
# export FLASHINFER_WORKSPACE_BASE= TMPDIR= TMP= TEMP= ; VLLM_NO_USAGE_STATS=1; JOB_SCRATCH_ROOT; training.torch_storage_preflight; MODEL_LOAD_BOUNDARY
# CUDA_VISIBLE_DEVICES=1 --device cuda:0 ; CUDA_VISIBLE_DEVICES=0
# locked: --max_steps 40 --temperature 0.5 --num_generations 4 --learning_rate 1e-5 --lr_scheduler_type constant --beta 0.0 --top_p 1.0
# locked: --target_modules q_proj v_proj --lora_rank 8 --lora_alpha 16 --freeze_vit true --freeze_aligner true egoqa_combined_video_answer_margin gate2_14119442 checkpoint-1
export MODE=probe40 STEPS=40 PREVIOUS=smoke5 PREVIOUS_RESULT=answer_margin_smoke5_result.json
# hard gate: latest_answer_margin_smoke5_output.txt and prior["status"] == "passed" (or run_status)
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/grpo_v3_answer_margin_smoke1.sbatch"
