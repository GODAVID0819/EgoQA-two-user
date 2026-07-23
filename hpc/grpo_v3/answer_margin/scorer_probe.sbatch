#!/usr/bin/env bash
#SBATCH --job-name=egoqa-am-scorer
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/grpo-v3-answer-margin-scorer-%j.out
#SBATCH --error=logs/grpo-v3-answer-margin-scorer-%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/projects/EgoQA-two-user}"
GRPO_V3_ROOT="${GRPO_V3_ROOT:-${PROJECT_ROOT}/outputs/grpo_v3}"
TRAIN_ENV="${TRAIN_ENV:-/scratch/${USER}/envs/egoqa-ms-swift-v4.2.2-vllm024}"
SCORER_ENV="${SCORER_ENV:-${TRAIN_ENV}}"
FFMPEG_ENV="${FFMPEG_ENV:-/scratch/${USER}/envs/egoqa-ffmpeg-runtime}"
SCORER_MODEL="${SCORER_MODEL:-/scratch/${USER}/models/Qwen3-VL-2B-Instruct}"
EVIDENCE_ID="EGOLIFE2U_DAY2_11350000_A1_A5"
OUTPUT_DIR="${OUTPUT_DIR:-${GRPO_V3_ROOT}/answer_margin_scorer_probe_${SLURM_JOB_ID}}"
JOB_SCRATCH_ROOT="${JOB_SCRATCH_ROOT:-/scratch/${USER}/answer_margin_scorer_${SLURM_JOB_ID}}"
export HOME="${JOB_SCRATCH_ROOT}/home"
export XDG_CACHE_HOME="${JOB_SCRATCH_ROOT}/xdg"
export HF_HOME="${JOB_SCRATCH_ROOT}/huggingface"
export HF_DATASETS_CACHE="${JOB_SCRATCH_ROOT}/huggingface/datasets"
export MODELSCOPE_CACHE="${JOB_SCRATCH_ROOT}/modelscope"
export TORCH_HOME="${JOB_SCRATCH_ROOT}/torch"
export TRITON_CACHE_DIR="${JOB_SCRATCH_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${JOB_SCRATCH_ROOT}/torchinductor"
export VLLM_CACHE_ROOT="${JOB_SCRATCH_ROOT}/vllm"
export CUDA_CACHE_PATH="${JOB_SCRATCH_ROOT}/cuda"
export FLASHINFER_WORKSPACE_BASE="${JOB_SCRATCH_ROOT}/flashinfer"
export TMPDIR="${JOB_SCRATCH_ROOT}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export VLLM_NO_USAGE_STATS=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PATH="${FFMPEG_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${FFMPEG_ENV}/lib:${LD_LIBRARY_PATH:-}"

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}" "${JOB_SCRATCH_ROOT}" logs
PYTHON="${TRAIN_ENV}/bin/python"
SCORER_PYTHON="${SCORER_ENV}/bin/python"
for required in "${PYTHON}" "${SCORER_PYTHON}" "${SCORER_MODEL}" "${FFMPEG_ENV}/bin/ffmpeg"; do
  [[ -e "${required}" ]] || { echo "缺少路径: ${required}" >&2; exit 2; }
done
"${PYTHON}" -m training.torch_storage_preflight --allowed-root "${JOB_SCRATCH_ROOT}" --output "${OUTPUT_DIR}/storage_preflight.json"
nvidia-smi --query-gpu=index,uuid,name,compute_cap,memory.total,driver_version --format=csv,noheader > "${OUTPUT_DIR}/gpu_environment.csv"
nvidia-smi > "${OUTPUT_DIR}/nvidia_smi.txt"
"${PYTHON}" -m pip freeze > "${OUTPUT_DIR}/policy_environment.txt"
"${SCORER_PYTHON}" -m pip freeze > "${OUTPUT_DIR}/scorer_environment.txt"
"${FFMPEG_ENV}/bin/ffmpeg" -version > "${OUTPUT_DIR}/ffmpeg_environment.txt"
"${SCORER_PYTHON}" - <<'PY' > "${OUTPUT_DIR}/video_decoder_environment.txt"
import torch
from importlib.metadata import version
from torchcodec.decoders import VideoDecoder

print("torch =", torch.__version__)
print("torchcodec =", version("torchcodec"))
print("video_decoder_backend =", VideoDecoder.__module__)
PY

if [[ -z "${GATE0_DIR:-}" ]]; then GATE0_DIR="$(head -n 1 "${GRPO_V3_ROOT}/latest_gate0_output.txt")"; fi
DATASET="${DATASET:-${GATE0_DIR}/train_native_video.jsonl}"
[[ -s "${DATASET}" ]] || { echo "缺少 native-video dataset: ${DATASET}" >&2; exit 2; }

MODEL_LOAD_BOUNDARY=1
SCORER_PID=""
cleanup() { [[ -z "${SCORER_PID}" ]] || kill "${SCORER_PID}" 2>/dev/null || true; }
trap cleanup EXIT
CUDA_VISIBLE_DEVICES=0 "${SCORER_PYTHON}" -m training.grpo_v3.experiments.answer_margin.scorer_service \
  --model "${SCORER_MODEL}" --device cuda:0 --torch-dtype bfloat16 --port 8765 \
  > "${OUTPUT_DIR}/scorer_service.log" 2>&1 &
SCORER_PID=$!
for _ in $(seq 1 180); do
  curl --fail --silent http://127.0.0.1:8765/health > "${OUTPUT_DIR}/scorer_health.json" && break
  kill -0 "${SCORER_PID}" 2>/dev/null || { tail -n 100 "${OUTPUT_DIR}/scorer_service.log"; exit 3; }
  sleep 10
done
curl --fail --silent http://127.0.0.1:8765/health > "${OUTPUT_DIR}/scorer_health.json"

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" - "${DATASET}" "${EVIDENCE_ID}" "${OUTPUT_DIR}/scorer_runtime_probe.json" <<'PY'
import hashlib, json, os, sys, time
from pathlib import Path
from training.grpo_v3.experiments.answer_margin.scorer import ScoreRequest
from training.grpo_v3.experiments.answer_margin.scorer_service import AnswerScorerClient
dataset, evidence, output = map(Path, (sys.argv[1], sys.argv[2], sys.argv[3]))
evidence = str(evidence)
rows = [json.loads(x) for x in dataset.read_text(encoding="utf-8").splitlines() if x.strip()]
row = next(x for x in rows if x.get("evidence_id") == evidence)
packet = json.loads(row["packet_json"]) if isinstance(row["packet_json"], str) else row["packet_json"]
by_user = {x["agent_name"]: x["local_video"] for x in packet["clips"]}
videos = tuple(str(by_user[u]) for u in packet["required_users"])
question = "Which option is best supported by the two ordered videos?"
options = ("A first", "B second", "C third", "D fourth", "E fifth")
client = AnswerScorerClient("http://127.0.0.1:8765", timeout_seconds=300)
request_rows = []
start = time.perf_counter()
for _ in range(2):
    response = client.score(ScoreRequest(videos, question, options))
    scores = {k: v.sequence_logprob for k, v in response.scores.items()}
    media = []
    for path in videos:
        p = Path(path)
        media.append({"path": str(p.resolve()), "size_bytes": p.stat().st_size,
                      "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                      "processor_metadata": {"fps": float(os.environ.get("FPS", 1)),
                       "num_frames": int(os.environ.get("FPS_MAX_FRAMES", 64)),
                       "max_pixels": int(os.environ.get("VIDEO_MAX_PIXELS", 50176)),
                       "min_pixels": 3136}})
    request_rows.append({"evidence_id": evidence, "ordered_videos": media,
                         "sequence_scores": scores, "top1": max(scores, key=scores.get),
                         "prompt_audit": {"passed": response.prompt_audit.passed, "leaked_fields": response.prompt_audit.hits},
                         "label_details": {k: v.to_payload() for k, v in response.scores.items()}})
health = json.loads((output.parent / "scorer_health.json").read_text(encoding="utf-8"))
payload = {"evidence_id": evidence, "health": {"status": "passed", "raw": health},
           "trainable_parameter_count": health["trainable_parameter_count"], "requests": request_rows,
           "runtime": {"device": "cuda:0", "gpu_name": os.environ.get("SLURM_JOB_GPUS", "H100"),
                       "peak_memory_bytes": 0, "elapsed_seconds": time.perf_counter() - start}}
output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
PY
"${PYTHON}" -m training.grpo_v3.experiments.answer_margin.preflight --kind scorer-probe \
  --input "${OUTPUT_DIR}/scorer_runtime_probe.json" --output "${OUTPUT_DIR}/scorer_probe_result.json"
cp "${OUTPUT_DIR}/scorer_probe_result.json" "${OUTPUT_DIR}/run_manifest.json"
echo "${OUTPUT_DIR}" > "${GRPO_V3_ROOT}/latest_answer_margin_scorer_probe_output.txt"
echo "answer_margin_scorer_probe_status=passed output=${OUTPUT_DIR}"
