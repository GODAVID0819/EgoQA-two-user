from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SBATCH = (
    REPO_ROOT
    / "hpc"
    / "qa"
    / "experiments"
    / "run_six_user_qa_10min_vllm_fps_resolution_3x10.sbatch"
)
BASE_RUNTIME = (
    REPO_ROOT
    / "hpc"
    / "qa"
    / "production"
    / "run_six_user_qa_10min_sequential_0p5_fresh30.sbatch"
)


def test_three_profile_ablation_uses_requested_fps_and_pixel_budgets() -> None:
    text = SBATCH.read_text(encoding="utf-8")

    assert 'PROFILE_FPS=("0.25" "0.50" "0.75")' in text
    assert 'PROFILE_TOTAL_FRAMES=("900" "1800" "2700")' in text
    assert 'PROFILE_MAX_TOKENS_PER_FRAME=("236" "117" "77")' in text
    assert 'PROFILE_MAX_PIXELS=("185024" "91728" "60368")' in text
    assert 'PROFILE_EQUIV_RESOLUTION=("430x430" "303x303" "246x246")' in text


def test_ablation_is_ten_packets_one_attempt_and_reuses_exact_evidence() -> None:
    text = SBATCH.read_text(encoding="utf-8")

    assert 'TARGET_COUNT="${TARGET_COUNT:-10}"' in text
    assert 'QA_PACKET_LIMIT="${QA_PACKET_LIMIT:-10}"' in text
    assert 'MAX_ATTEMPTS="${MAX_ATTEMPTS:-1}"' in text
    assert 'reuse_preprocessed_from="${SHARED_PREPROCESSING_DIR}"' in text
    assert "shared_candidates.sha256" in text
    assert "shared_candidate_changed_before_profile" in text
    assert "shared_candidate_changed_after_profile" in text


def test_ablation_pins_vllm_flash_attention_and_video_bounds() -> None:
    text = SBATCH.read_text(encoding="utf-8")

    assert 'VLLM_REQUIRED_VERSION="${VLLM_REQUIRED_VERSION:-0.28.0}"' in text
    assert 'FLASHINFER_REQUIRED_VERSION="${FLASHINFER_REQUIRED_VERSION:-0.6.16.post3}"' in text
    assert 'VLLM_SAMPLING_SEED="${VLLM_SAMPLING_SEED:-20260906}"' in text
    assert 'VLLM_MAX_NUM_SEQS="8"' in text
    assert 'VLLM_BATCH_WAIT_MS="20"' in text
    assert 'VLLM_ATTENTION_BACKEND="FLASH_ATTN"' in text
    assert 'VLLM_MM_ENCODER_ATTN_BACKEND="FLASH_ATTN"' in text
    assert 'VLLM_MIN_IMAGE_PIXELS="3136"' in text
    assert 'GENERATOR_MAX_IMAGE_PIXELS="${profile_max_pixels}"' in text
    assert 'VLLM_VIDEO_FPS="${profile_fps}"' in text
    assert "is_fa_version_supported" in text
    assert 'for package in ("flashinfer-python", "flashinfer-cubin")' in text
    assert "expected_pixels = max_tokens * 28 * 28" in text
    assert "expected_frames = round(profile_fps * 6 * 600)" in text
    assert '"cap_pixels_per_frame": True' in text


def test_ablation_has_remote_storage_and_video_preflights() -> None:
    raw = SBATCH.read_bytes()
    text = raw.decode("utf-8")

    assert raw.startswith(b"#!/usr/bin/env bash\n")
    assert b"\r" not in raw
    assert "#SBATCH --gres=gpu:2" in text
    assert "#SBATCH --time=24:00:00" in text
    assert 'PROJECT_ROOT="${QWEN3VL_PROJECT_ROOT:-/scratch/${USER}/Long-video-understanding-clip}"' in text
    assert 'HPC_ROOT="${PROJECT_ROOT}/hpc"' in text
    assert 'RUNTIME_SCRIPT="${RUNTIME_SCRIPT:-${HPC_ROOT}/qa/production/' in text
    assert '"${HPC_ROOT}/logs"' in text
    assert '${PACKAGE_ROOT}/hpc' not in text
    assert 'JOB_SCRATCH_ROOT="${JOB_SCRATCH_ROOT:-/scratch/${USER}/j/${SLURM_JOB_ID}}"' in text
    for variable in (
        "HOME",
        "XDG_CACHE_HOME",
        "HF_HOME",
        "HF_DATASETS_CACHE",
        "MODELSCOPE_CACHE",
        "TORCH_HOME",
        "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "VLLM_CACHE_ROOT",
        "CUDA_CACHE_PATH",
        "FLASHINFER_WORKSPACE_BASE",
        "TMPDIR",
        "TMP",
        "TEMP",
    ):
        assert f"export {variable}=" in text
    assert "-m training.torch_storage_preflight" in text
    assert '--allowed-root "${JOB_SCRATCH_ROOT}"' in text
    assert 'export PATH="${FFMPEG_ENV}/bin:${PATH}"' in text
    assert 'export LD_LIBRARY_PATH="${FFMPEG_ENV}/lib' in text
    assert "get_video_reader_backend" in text
    assert 'export FORCE_QWENVL_VIDEO_READER=decord' in text
    assert 'LOGIN_HF_TOKEN_PATH="${HF_TOKEN_PATH:-${HF_HOME:-${HOME}/.cache/huggingface}/token}"' in text
    assert 'export HF_TOKEN_PATH="${LOGIN_HF_TOKEN_PATH}"' in text


def test_base_runtime_accepts_each_profile_without_mutating_shared_evidence() -> None:
    text = BASE_RUNTIME.read_text(encoding="utf-8")

    assert (
        'JOB_SCRATCH_ROOT="${JOB_SCRATCH_ROOT:-/scratch/${USER}/job_scratch/'
        '${RUN_MODE}_${SLURM_JOB_ID}}"'
    ) in text
    assert (
        'VLLM_IPC_RUNTIME_ROOT="${VLLM_IPC_RUNTIME_ROOT:-${JOB_SCRATCH_ROOT}/v}"'
    ) in text
    assert 'HPC_ROOT="${PROJECT_ROOT}/hpc"' in text
    assert 'CUDA_KEEPER_SCRIPT="${CUDA_KEEPER_SCRIPT:-${HPC_ROOT}/shared/cuda.py}"' in text
    assert (
        'export PYTHONPATH="${PYTHON_ALIAS_ROOT}:${PACKAGE_ROOT}:${PROJECT_ROOT}:'
        '${PYTHONPATH:-}"'
    ) in text
    assert '${PACKAGE_ROOT}/hpc' not in text
    assert '--allowed-root "${JOB_SCRATCH_ROOT}"' in text
    assert 'FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"' in text
    assert "from qwen_vl_utils.vision_process import get_video_reader_backend" in text
    assert "from torchcodec.decoders import VideoDecoder" not in text
    assert 'expected_judge_video_fps = float(sys.argv[5])' in text
    assert 'expected_max_image_pixels = int(sys.argv[6])' in text
    assert 'reused_preprocessing = bool(sys.argv[7])' in text
    assert "if not reused_preprocessing and prepared_judge_video_fps" in text
    assert "if not reused_preprocessing and prepared_max_image_pixels" in text
    assert '"video_fps": float(os.environ["VLLM_VIDEO_FPS"])' in text
    assert '"min_image_pixels": int(os.environ["VLLM_MIN_IMAGE_PIXELS"])' in text
    assert '"cap_pixels_per_frame": True' in text
    assert "supports_vllm_randomized_request_ids" in text
    assert "caps_video_pixels_per_frame" in text
