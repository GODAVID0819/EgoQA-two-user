#!/usr/bin/env bash
#SBATCH --job-name=egoqa-am-cal
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:2
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/grpo-v3-answer-margin-calibration-%j.out
#SBATCH --error=logs/grpo-v3-answer-margin-calibration-%j.err
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/projects/EgoQA-two-user}"; GRPO_V3_ROOT="${GRPO_V3_ROOT:-${PROJECT_ROOT}/outputs/grpo_v3}"
TRAIN_ENV="${TRAIN_ENV:-/scratch/${USER}/envs/egoqa-ms-swift-v4.2.2-vllm024}"; SCORER_ENV="${SCORER_ENV:-${TRAIN_ENV}}"; FFMPEG_ENV="${FFMPEG_ENV:-/scratch/${USER}/envs/egoqa-ffmpeg-runtime}"
POLICY_MODEL="${POLICY_MODEL:-/scratch/${USER}/models/Qwen3-VL-2B-Instruct}"; SCORER_MODEL="${SCORER_MODEL:-${POLICY_MODEL}}"
EVIDENCE_ID="EGOLIFE2U_DAY2_11350000_A1_A5"; OUTPUT_DIR="${OUTPUT_DIR:-${GRPO_V3_ROOT}/answer_margin_calibration_${SLURM_JOB_ID}}"
JOB_SCRATCH_ROOT="${JOB_SCRATCH_ROOT:-/scratch/${USER}/answer_margin_calibration_${SLURM_JOB_ID}}"
export HOME="${JOB_SCRATCH_ROOT}/home"; export XDG_CACHE_HOME="${JOB_SCRATCH_ROOT}/xdg"; export HF_HOME="${JOB_SCRATCH_ROOT}/huggingface"
export HF_DATASETS_CACHE="${JOB_SCRATCH_ROOT}/huggingface/datasets"; export MODELSCOPE_CACHE="${JOB_SCRATCH_ROOT}/modelscope"; export TORCH_HOME="${JOB_SCRATCH_ROOT}/torch"
export TRITON_CACHE_DIR="${JOB_SCRATCH_ROOT}/triton"; export TORCHINDUCTOR_CACHE_DIR="${JOB_SCRATCH_ROOT}/torchinductor"; export VLLM_CACHE_ROOT="${JOB_SCRATCH_ROOT}/vllm"
export CUDA_CACHE_PATH="${JOB_SCRATCH_ROOT}/cuda"; export FLASHINFER_WORKSPACE_BASE="${JOB_SCRATCH_ROOT}/flashinfer"; export TMPDIR="${JOB_SCRATCH_ROOT}/tmp"; export TMP="${TMPDIR}"; export TEMP="${TMPDIR}"
export VLLM_NO_USAGE_STATS=1; export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"; export EGOQA_ANSWER_SCORER_BASE_URL="http://127.0.0.1:8765"; export EGOQA_ANSWER_SCORER_TIMEOUT_SECONDS=300
export PATH="${FFMPEG_ENV}/bin:${PATH}"; export LD_LIBRARY_PATH="${FFMPEG_ENV}/lib:${LD_LIBRARY_PATH:-}"
cd "${PROJECT_ROOT}"; mkdir -p "${OUTPUT_DIR}" "${JOB_SCRATCH_ROOT}" logs
PYTHON="${TRAIN_ENV}/bin/python"; SCORER_PYTHON="${SCORER_ENV}/bin/python"
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
PREV_DIR="$(head -n 1 "${GRPO_V3_ROOT}/latest_answer_margin_scorer_probe_output.txt")"
"${PYTHON}" -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"] == "passed"' "${PREV_DIR}/scorer_probe_result.json"
if [[ -z "${GATE2_DIR:-}" ]]; then GATE2_DIR="$(head -n 1 "${GRPO_V3_ROOT}/latest_gate2_output.txt")"; fi
GATE0_DIR="${GATE0_DIR:-$(head -n 1 "${GRPO_V3_ROOT}/latest_gate0_output.txt")}"; DATASET="${DATASET:-${GATE0_DIR}/train_native_video.jsonl}"
ADAPTER_DIR="$(${PYTHON} -c 'import json,sys; from pathlib import Path; d=Path(sys.argv[1]); r=json.load(open(d/"gate2_result.json")); m=json.load(open(d/"run_manifest.json")); assert r["status"] == "passed"; p=Path(m["adapter_dir"]); assert p.name == "checkpoint-1"; print(p)' "${GATE2_DIR}")"
MODEL_LOAD_BOUNDARY=1
SCORER_PID=""; cleanup(){ [[ -z "${SCORER_PID}" ]] || kill "${SCORER_PID}" 2>/dev/null || true; }; trap cleanup EXIT
CUDA_VISIBLE_DEVICES=1 "${SCORER_PYTHON}" -m training.grpo_v3_answer_scorer_service --model "${SCORER_MODEL}" --device cuda:0 --torch-dtype bfloat16 --port 8765 > "${OUTPUT_DIR}/scorer_service.log" 2>&1 & SCORER_PID=$!
for _ in $(seq 1 180); do curl --fail --silent http://127.0.0.1:8765/health > "${OUTPUT_DIR}/scorer_health.json" && break; kill -0 "${SCORER_PID}" 2>/dev/null || exit 3; sleep 10; done
curl --fail --silent http://127.0.0.1:8765/health >/dev/null
CUDA_VISIBLE_DEVICES=0 "${PYTHON}" - "${DATASET}" "${EVIDENCE_ID}" "${POLICY_MODEL}" "${ADAPTER_DIR}" "${OUTPUT_DIR}/calibration_trace.jsonl" <<'PY'
import json, sys
from pathlib import Path
from peft import PeftModel
from qwen3vl_runner import Qwen3VLTransformersRunner
from training.grpo_v3_answer_margin import ANSWER_MARGIN_REWARD_REVISION, PermutationKey
from training.grpo_v3_answer_margin_reward import make_answer_margin_score_fn
from training.grpo_v3_formality_fixed_eval import _prompt, set_generation_seed
dataset, evidence, model, adapter, output = sys.argv[1:]
row = next(x for x in (json.loads(s) for s in Path(dataset).read_text(encoding="utf-8").splitlines() if s.strip()) if x.get("evidence_id") == evidence)
packet = json.loads(row["packet_json"]) if isinstance(row["packet_json"], str) else row["packet_json"]
runner = Qwen3VLTransformersRunner(model_id=model, max_new_tokens=1024, max_image_pixels=50176, min_video_pixels=3136, dtype="bfloat16")
runner.model = PeftModel.from_pretrained(runner.model, adapter, is_trainable=False); runner.model.eval(); runner.device = next(runner.model.parameters()).device
score = make_answer_margin_score_fn(base_url="http://127.0.0.1:8765", timeout_seconds=300)
with Path(output).open("w", encoding="utf-8") as f:
  for group in range(8):
    completions=[]
    for candidate in range(4):
      set_generation_seed(2026072200 + 4*group + candidate, torch_module=runner.torch)
      completions.append(runner.generate(_prompt(row), image_paths=[], video_paths=row["videos"], decoding_mode="sampling", temperature=0.5, top_p=1.0))
    for candidate, raw in enumerate(completions):
      key=PermutationKey(ANSWER_MARGIN_REWARD_REVISION,"t05","calibration",evidence,group,candidate)
      item=score(raw_completion=str(raw),packet=packet,evidence_id=evidence,candidate_index=candidate,key=key,global_step=0,reward_call_index=group,question_type=row["question_type"],generation_mode=row.get("generation_mode","baseline"))
      item["record"]["temperature"]=0.5; f.write(json.dumps(item,ensure_ascii=False,allow_nan=False)+"\n")
PY
"${PYTHON}" -m training.grpo_v3_answer_margin_preflight --kind calibration --input "${OUTPUT_DIR}/calibration_trace.jsonl" --output "${OUTPUT_DIR}/calibration_result.json"
echo "${OUTPUT_DIR}" > "${GRPO_V3_ROOT}/latest_answer_margin_calibration_output.txt"; echo "answer_margin_calibration_status=passed output=${OUTPUT_DIR}"
