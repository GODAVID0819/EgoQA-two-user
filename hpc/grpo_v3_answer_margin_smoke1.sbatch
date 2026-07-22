#!/usr/bin/env bash
#SBATCH --job-name=egoqa-am-s1
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:2
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/grpo-v3-answer-margin-smoke1-%j.out
#SBATCH --error=logs/grpo-v3-answer-margin-smoke1-%j.err
set -euo pipefail
# locked default: --max_steps 1
# hard gate default: latest_answer_margin_calibration_output.txt
MODE="${MODE:-smoke1}"; STEPS="${STEPS:-1}"; PREVIOUS="${PREVIOUS:-calibration}"; PREVIOUS_RESULT="${PREVIOUS_RESULT:-calibration_result.json}"
PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/projects/EgoQA-two-user}"; GRPO_V3_ROOT="${GRPO_V3_ROOT:-${PROJECT_ROOT}/outputs/grpo_v3}"
TRAIN_ENV="${TRAIN_ENV:-/scratch/${USER}/envs/egoqa-ms-swift-v4.2.2-vllm024}"; SCORER_ENV="${SCORER_ENV:-${TRAIN_ENV}}"; FFMPEG_ENV="${FFMPEG_ENV:-/scratch/${USER}/envs/egoqa-ffmpeg-runtime}"
POLICY_MODEL="${POLICY_MODEL:-/scratch/${USER}/models/Qwen3-VL-2B-Instruct}"; SCORER_MODEL="${SCORER_MODEL:-${POLICY_MODEL}}"; EVIDENCE_ID=EGOLIFE2U_DAY2_11350000_A1_A5
OUTPUT_DIR="${OUTPUT_DIR:-${GRPO_V3_ROOT}/answer_margin_${MODE}_${SLURM_JOB_ID}}"; JOB_SCRATCH_ROOT="${JOB_SCRATCH_ROOT:-/scratch/${USER}/answer_margin_${MODE}_${SLURM_JOB_ID}}"
export HOME="${JOB_SCRATCH_ROOT}/home"; export XDG_CACHE_HOME="${JOB_SCRATCH_ROOT}/xdg"; export HF_HOME="${JOB_SCRATCH_ROOT}/huggingface"; export HF_DATASETS_CACHE="${JOB_SCRATCH_ROOT}/huggingface/datasets"
export MODELSCOPE_CACHE="${JOB_SCRATCH_ROOT}/modelscope"; export TORCH_HOME="${JOB_SCRATCH_ROOT}/torch"; export TRITON_CACHE_DIR="${JOB_SCRATCH_ROOT}/triton"; export TORCHINDUCTOR_CACHE_DIR="${JOB_SCRATCH_ROOT}/torchinductor"
export VLLM_CACHE_ROOT="${JOB_SCRATCH_ROOT}/vllm"; export CUDA_CACHE_PATH="${JOB_SCRATCH_ROOT}/cuda"; export FLASHINFER_WORKSPACE_BASE="${JOB_SCRATCH_ROOT}/flashinfer"; export TMPDIR="${JOB_SCRATCH_ROOT}/tmp"; export TMP="${TMPDIR}"; export TEMP="${TMPDIR}"
export VLLM_NO_USAGE_STATS=1; export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"; export EGOQA_ANSWER_SCORER_BASE_URL=http://127.0.0.1:8765; export EGOQA_ANSWER_SCORER_TIMEOUT_SECONDS=300; export EGOQA_ANSWER_MARGIN_CONDITION_ID=t05; export EGOQA_GRPO_V3_REWARD_TRACE="${OUTPUT_DIR}/reward_trace.jsonl"
export PATH="${FFMPEG_ENV}/bin:${PATH}"; export LD_LIBRARY_PATH="${FFMPEG_ENV}/lib:${LD_LIBRARY_PATH:-}"
cd "${PROJECT_ROOT}"; mkdir -p "${OUTPUT_DIR}" "${JOB_SCRATCH_ROOT}" logs; PYTHON="${TRAIN_ENV}/bin/python"; SWIFT="${TRAIN_ENV}/bin/swift"; SCORER_PYTHON="${SCORER_ENV}/bin/python"
"${PYTHON}" -m training.torch_storage_preflight --allowed-root "${JOB_SCRATCH_ROOT}" --output "${OUTPUT_DIR}/storage_preflight.json"
nvidia-smi --query-gpu=index,uuid,name,compute_cap,memory.total,driver_version --format=csv,noheader > "${OUTPUT_DIR}/gpu_environment.csv"
nvidia-smi > "${OUTPUT_DIR}/nvidia_smi.txt"
[[ -x "${FFMPEG_ENV}/bin/ffmpeg" ]] || { echo "缺少 FFmpeg: ${FFMPEG_ENV}/bin/ffmpeg" >&2; exit 2; }
"${FFMPEG_ENV}/bin/ffmpeg" -version > "${OUTPUT_DIR}/ffmpeg_environment.txt"
"${SCORER_PYTHON}" - <<'PY' > "${OUTPUT_DIR}/video_decoder_environment.txt"
import torch
from importlib.metadata import version
from torchcodec.decoders import VideoDecoder
print("torch =", torch.__version__)
print("torchcodec =", version("torchcodec"))
print("video_decoder_backend =", VideoDecoder.__module__)
PY
PREV_DIR="$(head -n 1 "${GRPO_V3_ROOT}/latest_answer_margin_${PREVIOUS}_output.txt")"; "${PYTHON}" -c 'import json,sys; x=json.load(open(sys.argv[1])); assert x.get("status", x.get("run_status")) == "passed"' "${PREV_DIR}/${PREVIOUS_RESULT}"
GATE2_DIR="${GATE2_DIR:-$(head -n 1 "${GRPO_V3_ROOT}/latest_gate2_output.txt")}"; GATE0_DIR="${GATE0_DIR:-$(head -n 1 "${GRPO_V3_ROOT}/latest_gate0_output.txt")}"; DATASET="${DATASET:-${GATE0_DIR}/train_native_video.jsonl}"
ADAPTER_DIR="$(${PYTHON} -c 'import json,sys; from pathlib import Path; d=Path(sys.argv[1]); assert json.load(open(d/"gate2_result.json"))["status"] == "passed"; p=Path(json.load(open(d/"run_manifest.json"))["adapter_dir"]); assert p.name == "checkpoint-1"; print(p)' "${GATE2_DIR}")"
cp "$(head -n 1 "${GRPO_V3_ROOT}/latest_answer_margin_scorer_probe_output.txt")/scorer_probe_result.json" "${OUTPUT_DIR}/scorer_runtime_probe.json"
MODEL_LOAD_BOUNDARY=1; SCORER_PID=""; cleanup(){ [[ -z "${SCORER_PID}" ]] || kill "${SCORER_PID}" 2>/dev/null || true; }; trap cleanup EXIT
CUDA_VISIBLE_DEVICES=1 "${SCORER_PYTHON}" -m training.grpo_v3_answer_scorer_service --model "${SCORER_MODEL}" --device cuda:0 --torch-dtype bfloat16 --port 8765 > "${OUTPUT_DIR}/scorer_service.log" 2>&1 & SCORER_PID=$!
for _ in $(seq 1 180); do curl --fail --silent http://127.0.0.1:8765/health >/dev/null && break; kill -0 "${SCORER_PID}" 2>/dev/null || exit 3; sleep 10; done; curl --fail --silent http://127.0.0.1:8765/health >/dev/null
: > "${EGOQA_GRPO_V3_REWARD_TRACE}"
CUDA_VISIBLE_DEVICES=0 "${SWIFT}" rlhf --rlhf_type grpo --model "${POLICY_MODEL}" --adapters "${ADAPTER_DIR}" --dataset "${DATASET}" --external_plugins "${PROJECT_ROOT}/training/grpo_v3_reward_plugin.py" --reward_funcs egoqa_combined_video_answer_margin --tuner_type lora --torch_dtype bfloat16 --freeze_vit true --freeze_aligner true --target_modules q_proj v_proj --lora_rank 8 --lora_alpha 16 --use_vllm false --num_generations 4 --per_device_train_batch_size 4 --gradient_accumulation_steps 1 --gradient_checkpointing true --max_steps "${STEPS}" --seed 42 --data_seed 42 --dataset_shuffle false --max_length 32768 --max_completion_length 1024 --learning_rate 1e-5 --lr_scheduler_type constant --beta 0.0 --temperature 0.5 --top_p 1.0 --save_steps "${STEPS}" --save_total_limit 1 --logging_steps 1 --log_completions true --split_dataset_ratio 0 --dataset_num_proc 1 --dataloader_num_workers 0 --report_to none --output_dir "${OUTPUT_DIR}/swift"
"${PYTHON}" -m training.grpo_v3_adapter_reload --model-path "${POLICY_MODEL}" --search-root "${OUTPUT_DIR}" --output "${OUTPUT_DIR}/adapter_reload.json"
"${PYTHON}" - "${OUTPUT_DIR}" "${GATE2_DIR}" "${ADAPTER_DIR}" "${PYTHON}" "${SCORER_PYTHON}" <<'PY'
import hashlib,json,sys
from pathlib import Path
o,g,a,p,s=map(Path,sys.argv[1:]); files=[{"path":str(x),"sha256":hashlib.sha256(x.read_bytes()).hexdigest()} for x in a.rglob("*") if x.is_file()]
(o/"environment_audit.json").write_text(json.dumps({"status":"passed","policy_environment":{"status":"passed","python":str(p)},"scorer_environment":{"status":"passed","python":str(s)}}),encoding="utf-8")
(o/"checkpoint_inventory.json").write_text(json.dumps({"status":"passed","parent_job":"gate2_14119442","parent_checkpoint":"checkpoint-1","source":"manifest_and_hash_inventory","gate2_result":json.load(open(g/"gate2_result.json")),"adapter_files":files}),encoding="utf-8")
(o/"resolved_config.json").write_text(json.dumps({"condition_id":"t05","temperature":0.5,"num_generations":4,"reward_revision":"combined_video_answer_margin_v1","evidence_id":"EGOLIFE2U_DAY2_11350000_A1_A5","parent_job":"gate2_14119442","parent_checkpoint":"checkpoint-1"}),encoding="utf-8")
(o/"run_manifest.json").write_text(json.dumps({"status":"completed","reward_revision":"combined_video_answer_margin_v1","condition_id":"t05","checkpoint_inventory":"checkpoint_inventory.json"}),encoding="utf-8")
PY
"${PYTHON}" -m training.grpo_v3_answer_margin_artifacts --mode "${MODE}" --output-dir "${OUTPUT_DIR}" --result "${OUTPUT_DIR}/answer_margin_${MODE}_result.json"
echo "${OUTPUT_DIR}" > "${GRPO_V3_ROOT}/latest_answer_margin_${MODE}_output.txt"; echo "answer_margin_${MODE}_status=passed output=${OUTPUT_DIR}"
