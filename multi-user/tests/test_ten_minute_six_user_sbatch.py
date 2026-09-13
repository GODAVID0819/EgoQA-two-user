from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
SBATCH = (
    ROOT
    / "hpc"
    / "qa"
    / "production"
    / "run_six_user_qa_10min_time_aware.sbatch"
)
LEGACY_SBATCH = (
    ROOT
    / "hpc"
    / "qa"
    / "production"
    / "run_six_user_qa_10min_legacy_zero_shot.sbatch"
)
CUDA_KEEPER = ROOT.parent / "hpc" / "shared" / "cuda.py"
GITATTRIBUTES = ROOT / ".gitattributes"


class TenMinuteSixUserSbatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = SBATCH.read_bytes()
        cls.text = cls.raw.decode("utf-8")

    def test_shell_files_are_lf_only(self) -> None:
        self.assertTrue(self.raw.startswith(b"#!/usr/bin/env bash\n"))
        self.assertNotIn(b"\r", self.raw)
        attributes = GITATTRIBUTES.read_text(encoding="utf-8")
        self.assertIn("*.sbatch text eol=lf", attributes)
        self.assertIn("*.sh text eol=lf", attributes)

    def test_requests_two_h100_or_h200_gpus_and_balances_the_model(self) -> None:
        self.assertIn("#SBATCH --gres=gpu:2", self.text)
        self.assertIn('#SBATCH --constraint="h100|h200"', self.text)
        self.assertIn("REQUIRED_GPU_COUNT=2", self.text)
        self.assertIn('CUDA_KEEPER_GPUS="0,1"', self.text)
        self.assertIn(
            'export QWEN_MEMORY_SAFE_DEVICE_MAP="balanced"',
            self.text,
        )
        self.assertIn(
            'export QWEN_MEMORY_SAFE_REQUIRED_MODEL_GPU_COUNT="${REQUIRED_GPU_COUNT}"',
            self.text,
        )

    def test_hpc_helpers_resolve_from_cluster_project_root(self) -> None:
        self.assertIn(
            'PROJECT_ROOT="${QWEN3VL_PROJECT_ROOT:-/scratch/${USER}/Long-video-understanding-clip}"',
            self.text,
        )
        self.assertIn('HPC_ROOT="${PROJECT_ROOT}/hpc"', self.text)
        self.assertIn(
            'PACKAGE_ROOT="${PACKAGE_ROOT:-${PROJECT_ROOT}/egolife_two_user_qa/multi-user}"',
            self.text,
        )
        self.assertIn(
            'CUDA_KEEPER_SCRIPT="${CUDA_KEEPER_SCRIPT:-${HPC_ROOT}/shared/cuda.py}"',
            self.text,
        )
        self.assertIn('source "${HPC_ROOT}/shared/env_qwen3vl.sh"', self.text)
        self.assertNotIn('${PACKAGE_ROOT}/hpc', self.text)
        self.assertNotIn('egolife_two_user_qa/multi-user/hpc', self.text)

    def test_cuda_keeper_uses_the_guarded_python_invocation(self) -> None:
        command = (
            'PYTHONUNBUFFERED=1 "${PYTHON}" -P "${CUDA_KEEPER_SCRIPT}" \\\n'
            '  --threshold "${CUDA_KEEPER_THRESHOLD}" \\\n'
            '  --gpus "${CUDA_KEEPER_GPUS}" \\\n'
            '  --reserve "${CUDA_KEEPER_RESERVE}"'
        )
        self.assertIn(command, self.text)
        self.assertIn('CUDA_KEEPER_PID=$!', self.text)
        self.assertIn('kill -0 "${CUDA_KEEPER_PID}"', self.text)
        self.assertIn('trap cleanup EXIT INT TERM', self.text)
        self.assertLess(
            self.text.index('stage "start_cuda_keeper"'),
            self.text.index('stage "generate_six_user_qa"'),
        )
        self.assertIn("REQUIRED_GPU_COUNT=2", self.text)
        self.assertIn('CUDA_KEEPER_GPUS="0,1"', self.text)
        self.assertIn('export RUN_CUDA_KEEPER_GPUS="${CUDA_KEEPER_GPUS}"', self.text)
        self.assertIn("CUDA keeper must cover every requested local GPU", self.text)

    def test_slurm_storage_and_video_runtime_guardrails(self) -> None:
        self.assertIn("#SBATCH --mem=320G", self.text)
        self.assertIn(
            'OUTDIR="${OUTPUT_BASE}/${RUN_MODE}_${SLURM_JOB_ID}"', self.text
        )
        self.assertNotIn("latest_", self.text)
        self.assertIn("export PYTHONUNBUFFERED=1", self.text)
        for variable in (
            "HOME",
            "XDG_CACHE_HOME",
            "HF_HOME",
            "HF_HUB_CACHE",
            "TRANSFORMERS_CACHE",
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
            self.assertRegex(self.text, rf"export {variable}=", variable)
        storage_preflight = self.text.index(
            '"${PYTHON}" -P -m training.torch_storage_preflight'
        )
        model_call = self.text.index("generate_video_qa_loop")
        self.assertLess(storage_preflight, model_call)
        self.assertIn('--allowed-root "${JOB_SCRATCH_ROOT}"', self.text)
        self.assertIn('FFMPEG_SOURCE="FFMPEG_ENV"', self.text)
        self.assertIn('FFMPEG_SOURCE="TRAIN_ENV"', self.text)
        self.assertIn('FFMPEG_SOURCE="activated_PATH"', self.text)
        self.assertIn('FFMPEG_BINARY="$(command -v ffmpeg || true)"', self.text)
        self.assertIn('FFPROBE_BINARY="$(command -v ffprobe || true)"', self.text)
        self.assertNotIn('test -x "${FFMPEG_ENV}/bin/ffmpeg"', self.text)
        self.assertIn(
            'export PATH="$(dirname "${FFMPEG_BINARY}"):$(dirname "${FFPROBE_BINARY}"):${PATH}"',
            self.text,
        )
        self.assertNotIn(
            'LD_LIBRARY_PATH="${FFMPEG_ENV}/lib:',
            self.text,
        )
        self.assertNotIn(
            'LD_LIBRARY_PATH="${TRAIN_ENV}/lib:',
            self.text,
        )
        self.assertIn('stage "configure_pytorch_runtime_libraries"', self.text)
        self.assertIn('PYTORCH_CUDNN_LIB="${PYTHON_SITE_PACKAGES}/nvidia/cudnn/lib"', self.text)
        self.assertIn('PYTORCH_CORE_LIB="${PYTHON_SITE_PACKAGES}/torch/lib"', self.text)
        self.assertIn(
            'export LD_LIBRARY_PATH="${TORCH_RUNTIME_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"',
            self.text,
        )
        self.assertIn('LOGIN_HF_TOKEN_PATH="${HF_TOKEN_PATH:-', self.text)
        self.assertIn('export HF_TOKEN_PATH="${LOGIN_HF_TOKEN_PATH}"', self.text)
        self.assertIn("hf_authentication=", self.text)
        self.assertIn("from torchcodec.decoders import VideoDecoder", self.text)
        self.assertIn('export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"', self.text)
        self.assertIn(
            'export QWEN_MEMORY_SAFE_MIN_AVAILABLE_RAM_GIB="160"', self.text
        )
        self.assertIn(
            'export QWEN_MEMORY_SAFE_DEVICE_MAP="balanced"',
            self.text,
        )

    def test_ten_minute_pruning_and_judge_contract(self) -> None:
        self.assertIn(
            '"${PYTHON}" -P "${PACKAGE_ROOT}/ten_minute_six_user_setup.py" prepare',
            self.text,
        )
        self.assertIn("--pruning-clusters-per-window 12", self.text)
        self.assertIn('pruning_sample_fps": 1.0', self.text)
        self.assertIn('pruning_cluster_window_seconds": 30.0', self.text)
        self.assertIn('judge_temporal_coverage": "full_600_seconds_per_user"', self.text)
        self.assertIn('"per_user_source_segment_map_reduce"', self.text)
        self.assertIn('"one_user_per_visual_call"', self.text)
        self.assertIn('"legacy_direct_zero_shot"', self.text)
        self.assertIn('"speaker_only_then_all_six_full_videos"', self.text)
        self.assertIn('--judge-video-source full', self.text)
        self.assertIn(
            'export QWEN_MEMORY_SAFE_VIDEO_FPS="${JUDGE_VIDEO_FPS}"', self.text
        )
        self.assertIn('JUDGE_VIDEO_FPS="${JUDGE_VIDEO_FPS:-0.75}"', self.text)
        self.assertIn(
            'export QWEN_MEMORY_SAFE_MAX_INPUT_TOKENS="262144"', self.text
        )
        self.assertIn(
            'GENERATOR_MAX_IMAGE_PIXELS="${GENERATOR_MAX_IMAGE_PIXELS:-65536}"',
            self.text,
        )
        self.assertIn(
            'export QWEN_MEMORY_SAFE_ADAPTIVE_IMAGE_PIXELS="1"', self.text
        )
        self.assertIn(
            'export QWEN_MEMORY_SAFE_IMAGE_CONTEXT_TARGET_FRACTION="0.85"',
            self.text,
        )
        self.assertIn(
            '--generator-max-image-pixels "${GENERATOR_MAX_IMAGE_PIXELS}"',
            self.text,
        )
        self.assertIn(
            '--max-image-pixels "${GENERATOR_MAX_IMAGE_PIXELS}"', self.text
        )
        self.assertIn(
            'plan.get("estimated_generator_input_tokens", 0) > plan["max_input_tokens"]',
            self.text,
        )

    def test_long_context_prefill_requires_cudnn_v8_with_64_bit_indexing(self) -> None:
        self.assertIn("unset TORCH_CUDNN_V8_API_DISABLED", self.text)
        self.assertIn('export TORCH_CUDNN_V8_API_DEBUG=1', self.text)
        self.assertIn('stage "long_context_cudnn_preflight"', self.text)
        self.assertIn("cudnn_version = torch.backends.cudnn.version()", self.text)
        self.assertIn("cudnn_version < 90300", self.text)
        self.assertIn("requires cuDNN 9.3+", self.text)
        self.assertIn("causal_conv1d_available=", self.text)
        self.assertIn("fla_available=", self.text)
        self.assertLess(
            self.text.index("unset TORCH_CUDNN_V8_API_DISABLED"),
            self.text.index("import torch"),
        )
        self.assertLess(
            self.text.index('stage "configure_pytorch_runtime_libraries"'),
            self.text.index('stage "long_context_cudnn_preflight"'),
        )
        self.assertLess(
            self.text.index('stage "long_context_cudnn_preflight"'),
            self.text.index('stage "generate_six_user_qa"'),
        )

    def test_previous_completed_preprocessing_can_be_reused(self) -> None:
        self.assertIn(
            'REUSE_PREPROCESSED_FROM="${REUSE_PREPROCESSED_FROM:-}"',
            self.text,
        )
        self.assertIn('if [[ -n "${REUSE_PREPROCESSED_FROM}" ]]; then', self.text)
        self.assertIn('stage "reuse_preprocessed_candidates"', self.text)
        self.assertIn(
            '${REUSE_PREPROCESSED_FROM}/ten_minute_setup/time_aware_candidates/'
            'six_user_10min_time_aware.jsonl',
            self.text,
        )
        self.assertIn(
            '${REUSE_PREPROCESSED_FROM}/ten_minute_setup/'
            'six_user_10min_setup_summary.json',
            self.text,
        )
        self.assertIn('test -s "${CANDIDATES}"', self.text)
        self.assertIn('test -s "${SETUP_SUMMARY}"', self.text)
        self.assertIn("reused generator frame is missing", self.text)
        self.assertIn("reused full judge video is missing", self.text)
        self.assertIn("cached judge source segment is missing", self.text)
        self.assertIn("each ten-minute judge input must preserve its 20 source segments", self.text)
        self.assertIn('"preprocessing_mode": (', self.text)

    def test_new_preparation_excludes_prior_windows_and_repeated_questions(self) -> None:
        self.assertIn(
            'EXCLUDE_PROCESSED_FROM="${EXCLUDE_PROCESSED_FROM:-${OUTPUT_BASE}}"',
            self.text,
        )
        self.assertIn('REQUIRE_FRESH_EVIDENCE=1', self.text)
        self.assertIn('EVIDENCE_RANDOM_SEED="${EVIDENCE_RANDOM_SEED:-20260831}"', self.text)
        self.assertIn(
            'EVIDENCE_FRESHNESS_ARGS+=(--exclude-processed-from "${EXCLUDE_PROCESSED_FROM}")',
            self.text,
        )
        self.assertIn('--random-seed "${EVIDENCE_RANDOM_SEED}"', self.text)
        self.assertIn('"${EVIDENCE_FRESHNESS_ARGS[@]}"', self.text)
        self.assertIn('stage "validate_fresh_questions"', self.text)
        self.assertIn('validate-fresh-output', self.text)
        self.assertIn('fresh_question_validation.json', self.text)
        self.assertIn('"require_fresh_evidence":', self.text)
        self.assertIn('"evidence_random_seed":', self.text)
        self.assertIn('fresh_evidence.get("source_window_overlap_count", -1)', self.text)
        self.assertLess(
            self.text.index('stage "prepare_ten_minute_six_user_candidates"'),
            self.text.index('stage "generate_six_user_qa"'),
        )
        self.assertLess(
            self.text.index('stage "generate_six_user_qa"'),
            self.text.index('stage "validate_fresh_questions"'),
        )

    def test_timed_out_qa_can_resume_without_copying_giant_auxiliary_logs(self) -> None:
        self.assertIn('RESUME_QA_FROM="${RESUME_QA_FROM:-}"', self.text)
        self.assertIn(
            'if [[ -n "${RESUME_QA_FROM}" && -z "${REUSE_PREPROCESSED_FROM}" ]]',
            self.text,
        )
        self.assertIn('stage "seed_qa_resume_state"', self.text)
        self.assertIn(
            'cp -f "${RESUME_QA_FROM}/qa_mcq.jsonl" "${OUTDIR}/qa_mcq.jsonl"',
            self.text,
        )
        self.assertIn('QA_RESUME_ARGS=(--resume)', self.text)
        self.assertIn('"${QA_RESUME_ARGS[@]}"', self.text)
        self.assertIn(
            '"qa_resume_from": os.environ["RUN_RESUME_QA_FROM"] or None',
            self.text,
        )
        self.assertNotIn(
            'cp -f "${RESUME_QA_FROM}/video_first_prompts.jsonl"',
            self.text,
        )
        self.assertNotIn(
            'cp -f "${RESUME_QA_FROM}/qa_mcq.intermediate.jsonl"',
            self.text,
        )
        self.assertLess(
            self.text.index('stage "seed_qa_resume_state"'),
            self.text.index('stage "generate_six_user_qa"'),
        )
        self.assertIn('stage "validate_six_user_judge_contract"', self.text)
        self.assertIn("intermediate row {line_number} exceeds the 5 MB ceiling", self.text)

    def test_legacy_zero_shot_mode_is_explicit_and_resumable(self) -> None:
        self.assertIn(
            'SIX_USER_JUDGE_MODE="${SIX_USER_JUDGE_MODE:-time-aware-map-reduce}"',
            self.text,
        )
        self.assertIn('--six-user-judge-mode "${SIX_USER_JUDGE_MODE}"', self.text)
        self.assertIn(
            '--infrastructure-skipped-output "${OUTDIR}/qa_mcq.infrastructure_skipped.jsonl"',
            self.text,
        )
        self.assertIn('COHORT_SKIP_SOURCE="${COHORT_SKIP_SOURCE:-}"', self.text)
        self.assertIn('--skip-evidence-id "${evidence_id}"', self.text)
        self.assertIn('--packet-limit "${QA_PACKET_LIMIT}"', self.text)
        self.assertIn('stages.count("evidence_groundedness_judge") != 1', self.text)
        self.assertIn('"combined_all_six_users": 6', self.text)

    def test_reuse_validation_accepts_legacy_summary_and_existing_media(self) -> None:
        blocks = re.findall(r"<<'PY'\n(.*?)\nPY", self.text, flags=re.DOTALL)
        validation = next(
            block for block in blocks if "ten_minute_contract_passed" in block
        )
        temp_root = ROOT / "tmp"
        temp_root.mkdir(exist_ok=True)
        root = temp_root / f"reuse_validation_{uuid4().hex}"
        root.mkdir()

        def cleanup() -> None:
            for path in root.iterdir():
                path.unlink(missing_ok=True)
            root.rmdir()

        self.addCleanup(cleanup)
        clips = []
        for index in range(6):
            frame = root / f"frame_{index}.png"
            video = root / f"full_{index}.mp4"
            frame.write_bytes(b"png")
            video.write_bytes(b"mp4")
            source_segments = []
            for segment_index in range(20):
                segment = root / f"source_{index}_{segment_index}.mp4"
                segment.write_bytes(b"mp4")
                source_segments.append(
                    {
                        "segment_index": segment_index,
                        "local_video": str(segment),
                    }
                )
            clips.append(
                {
                    "is_pruned": index != 0,
                    "force_frame_inputs": True,
                    "frames": [{"path": str(frame)}],
                    "full_local_video": str(video),
                    "source_segments": source_segments,
                }
            )
        candidates = root / "candidates.jsonl"
        candidates.write_text(
            json.dumps({"clips": clips}) + "\n", encoding="utf-8"
        )
        summary = root / "summary.json"
        summary.write_text(
            json.dumps(
                {
                    "context_plan": {
                        "pruning_sample_fps": 1.0,
                        "judge_temporal_coverage": (
                            "full_600_seconds_per_user"
                        ),
                        "generator_aggregate_frame_budget": 3600,
                        "max_input_tokens": 131072,
                    },
                    "time_aware_pruning": {
                        "cluster_window_seconds": 30.0,
                        "clusters_per_window": 12,
                        "max_pair_time_difference_seconds": 30.0,
                        "speaker_preserved": True,
                        "pruned_side": "providers_only",
                    },
                }
            ),
            encoding="utf-8",
        )
        original_argv = sys.argv
        try:
            sys.argv = ["validate", str(summary), str(candidates), "1", "0"]
            exec(compile(validation, "reuse_validation", "exec"), {})
        finally:
            sys.argv = original_argv

    def test_embedded_python_is_syntactically_valid(self) -> None:
        blocks = re.findall(r"<<'PY'\n(.*?)\nPY", self.text, flags=re.DOTALL)
        self.assertGreaterEqual(len(blocks), 3)
        for index, block in enumerate(blocks):
            compile(block, f"{SBATCH.name}:heredoc-{index}", "exec")


class LegacyZeroShotSbatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = LEGACY_SBATCH.read_bytes()
        cls.text = cls.raw.decode("utf-8")

    def test_launcher_is_lf_only_and_uses_project_root_hpc(self) -> None:
        self.assertTrue(self.raw.startswith(b"#!/usr/bin/env bash\n"))
        self.assertNotIn(b"\r", self.raw)
        self.assertIn("#SBATCH --mem=320G", self.text)
        self.assertIn("#SBATCH --time=48:00:00", self.text)
        self.assertIn("#SBATCH --gres=gpu:2", self.text)
        self.assertIn('#SBATCH --constraint="h100|h200"', self.text)
        self.assertIn(
            'BASE_LAUNCHER="${PROJECT_ROOT}/hpc/qa/production/'
            'run_six_user_qa_10min_time_aware.sbatch"',
            self.text,
        )
        self.assertNotIn("egolife_two_user_qa/multi-user/hpc", self.text)

    def test_launcher_freezes_legacy_mode_and_requires_explicit_preprocessing_reuse(self) -> None:
        self.assertIn('export SIX_USER_JUDGE_MODE="legacy-zero-shot"', self.text)
        self.assertIn('export JUDGE_VIDEO_FPS="${JUDGE_VIDEO_FPS:-0.75}"', self.text)
        self.assertIn(
            'export CUDA_KEEPER_RESERVE="${CUDA_KEEPER_RESERVE:-12.0}"',
            self.text,
        )
        self.assertIn("export REQUIRED_GPU_COUNT=2", self.text)
        self.assertIn("export QWEN_MEMORY_SAFE_DEVICE_MAP=balanced", self.text)
        self.assertIn(
            "export QWEN_MEMORY_SAFE_REQUIRED_MODEL_GPU_COUNT=2",
            self.text,
        )
        self.assertIn("export CUDA_KEEPER_GPUS=0,1", self.text)
        self.assertIn('export QA_PACKET_LIMIT=""', self.text)
        self.assertNotIn('export QA_PACKET_LIMIT="${QA_PACKET_LIMIT:-}"', self.text)
        self.assertIn(
            'export REUSE_PREPROCESSED_FROM="${REUSE_PREPROCESSED_FROM:-}"',
            self.text,
        )
        self.assertIn("error=missing_reuse_preprocessed_from", self.text)
        self.assertIn('export RESUME_QA_FROM=""', self.text)
        self.assertIn("export REQUIRE_FRESH_EVIDENCE=0", self.text)
        self.assertNotIn("six_user_qa_10min_time_aware_16463998", self.text)
        self.assertNotIn("error=missing_cohort_exclusions", self.text)
        self.assertIn('source "${BASE_LAUNCHER}"', self.text)

    def test_project_cuda_keeper_supports_and_records_two_gpu_selection(self) -> None:
        text = CUDA_KEEPER.read_text(encoding="utf-8")

        compile(text, str(CUDA_KEEPER), "exec")
        self.assertIn("parse_gpus_arg", text)
        self.assertIn("for g, r in zip(gpu_list, reserves)", text)
        self.assertIn('"cuda_keeper_start "', text)
        self.assertIn("selected_gpus={gpu_list}", text)
        self.assertIn("reserve_gib={reserves}", text)


if __name__ == "__main__":
    unittest.main()
