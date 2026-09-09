from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT
SBATCH = (
    REPOSITORY_ROOT
    / "hpc"
    / "qa"
    / "production"
    / "run_six_user_qa_10min_sequential_0p5_fresh30.sbatch"
)
CUDA_KEEPER = REPOSITORY_ROOT / "hpc" / "shared" / "cuda.py"


class TenMinuteSixUserSequentialSbatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = SBATCH.read_bytes()
        cls.text = cls.raw.decode("utf-8")

    def test_shell_is_lf_only(self) -> None:
        self.assertTrue(self.raw.startswith(b"#!/usr/bin/env bash\n"))
        self.assertNotIn(b"\r", self.raw)

    def test_fresh_thirty_defaults_and_new_output_namespace(self) -> None:
        self.assertIn('TARGET_COUNT="${TARGET_COUNT:-30}"', self.text)
        self.assertIn('SOURCE_WINDOW_COUNT="${SOURCE_WINDOW_COUNT:-120}"', self.text)
        self.assertIn('JUDGE_VIDEO_FPS="${JUDGE_VIDEO_FPS:-0.5}"', self.text)
        self.assertIn(
            'RUN_MODE="${RUN_MODE:-six_user_qa_10min_sequential_0p5_fresh30}"',
            self.text,
        )
        self.assertIn('OUTDIR="${OUTPUT_BASE}/${RUN_MODE}_${SLURM_JOB_ID}"', self.text)
        self.assertIn(
            'JOB_SCRATCH_ROOT="${JOB_SCRATCH_ROOT:-/scratch/${USER}/job_scratch/${RUN_MODE}_${SLURM_JOB_ID}}"',
            self.text,
        )

    def test_vllm_ipc_socket_path_stays_below_linux_limit(self) -> None:
        self.assertIn(
            'VLLM_IPC_RUNTIME_ROOT="${VLLM_IPC_RUNTIME_ROOT:-${JOB_SCRATCH_ROOT}/v}"',
            self.text,
        )
        self.assertIn('export TMPDIR="${VLLM_IPC_RUNTIME_ROOT}"', self.text)
        self.assertIn('stage "vllm_ipc_path_preflight"', self.text)
        self.assertIn('getattr(zmq, "IPC_PATH_MAX_LEN", 107)', self.text)
        self.assertIn('len(os.fsencode(path))', self.text)

    def test_recovery_reuses_only_preprocessed_packets(self) -> None:
        self.assertIn(
            "reuse_preprocessed_requires_require_fresh_evidence_0", self.text
        )
        self.assertIn(
            "new_preprocessing_requires_require_fresh_evidence_1", self.text
        )
        self.assertIn("fresh30_run_forbids_qa_resume", self.text)
        self.assertIn('REQUIRE_FRESH_EVIDENCE="${REQUIRE_FRESH_EVIDENCE:-1}"', self.text)
        self.assertIn('EXCLUDE_PROCESSED_FROM="${EXCLUDE_PROCESSED_FROM:-${OUTPUT_BASE}}"', self.text)
        self.assertIn('--exclude-processed-from "${EXCLUDE_PROCESSED_FROM}"', self.text)
        self.assertIn('stage "reuse_preprocessed_candidates"', self.text)
        self.assertIn(
            'CANDIDATES="${REUSE_PREPROCESSED_FROM}/ten_minute_setup/'
            'time_aware_candidates/six_user_10min_time_aware.jsonl"',
            self.text,
        )
        self.assertIn(
            'SETUP_SUMMARY="${REUSE_PREPROCESSED_FROM}/ten_minute_setup/'
            'six_user_10min_setup_summary.json"',
            self.text,
        )

    def test_parallelizes_top_level_judges_and_preserves_speaker_first_audit(self) -> None:
        self.assertIn(
            'SIX_USER_JUDGE_MODE="${SIX_USER_JUDGE_MODE:-sequential-separated-fact-audit}"',
            self.text,
        )
        self.assertIn('--six-user-judge-mode "${SIX_USER_JUDGE_MODE}"', self.text)
        self.assertIn('stages.count("qa_formality_judge") != 1', self.text)
        self.assertIn('stages.count("evidence_groundedness_judge")', self.text)
        self.assertIn('"combined_direct_judge",', self.text)
        self.assertIn('(1, 1, 1),  # speaker-only early rejection', self.text)
        self.assertIn(
            '(1, 6, 3),  # complete all-six and minimum-set factual trials',
            self.text,
        )
        self.assertIn("partial_skipped = (", self.text)
        self.assertIn('"minimum_required_users::"', self.text)
        self.assertNotIn('stages.index("qa_formality_judge")', self.text)
        self.assertNotIn(
            "did not finish grounding before answerability",
            self.text,
        )
        self.assertIn(
            '"judge_review_mode": "parallel_formality_grounding_answerability_with_speaker_first_fact_audit"',
            self.text,
        )

    def test_generation_keeps_old_generator_sampling_and_disables_thinking(self) -> None:
        self.assertIn('MODEL_ID="${MODEL_ID:-Qwen/Qwen3.8-27B}"', self.text)
        self.assertIn(
            'GENERATOR_MAX_IMAGE_PIXELS="${GENERATOR_MAX_IMAGE_PIXELS:-262144}"',
            self.text,
        )
        self.assertIn("--generation-mode baseline", self.text)
        self.assertIn("--generator-decode-mode sampling", self.text)
        self.assertIn("--disable-thinking", self.text)
        self.assertIn("--fixed-question-type-schedule", self.text)

    def test_vllm_tensor_parallel_flash_attention_is_the_default(self) -> None:
        self.assertIn('INFERENCE_BACKEND="${INFERENCE_BACKEND:-vllm-local}"', self.text)
        self.assertIn('REQUIRED_GPU_COUNT="${REQUIRED_GPU_COUNT:-2}"', self.text)
        self.assertIn('--backend "${INFERENCE_BACKEND}"', self.text)
        self.assertNotIn("--backend transformers-local-memory-safe", self.text)
        self.assertIn(
            'VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-${REQUIRED_GPU_COUNT}}"',
            self.text,
        )
        self.assertIn(
            'VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"',
            self.text,
        )
        self.assertIn(
            'VLLM_MM_ENCODER_ATTN_BACKEND="${VLLM_MM_ENCODER_ATTN_BACKEND:-FLASH_ATTN}"',
            self.text,
        )
        self.assertIn(
            'VLLM_GDN_PREFILL_BACKEND="${VLLM_GDN_PREFILL_BACKEND:-flashinfer}"',
            self.text,
        )
        self.assertIn(
            'VLLM_MTP_SPECULATIVE_TOKENS="${VLLM_MTP_SPECULATIVE_TOKENS:-1}"',
            self.text,
        )
        self.assertIn(
            'VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-8}"',
            self.text,
        )
        self.assertIn(
            'VLLM_BATCH_WAIT_MS="${VLLM_BATCH_WAIT_MS:-20}"',
            self.text,
        )
        self.assertIn('getattr(LLM, "enqueue_chat", None)', self.text)
        self.assertIn('getattr(LLM, "wait_for_completion", None)', self.text)
        self.assertIn('stage "vllm_flash_attention_preflight"', self.text)
        self.assertLess(
            self.text.index('stage "vllm_flash_attention_preflight"'),
            self.text.index('stage "generate_six_user_qa"'),
        )

    def test_cuda_keeper_defaults_on_and_tracks_requested_gpus(self) -> None:
        self.assertIn(
            'CUDA_KEEPER_ENABLE="${CUDA_KEEPER_ENABLE:-${ENABLE_CUDA_KEEPER:-1}}"',
            self.text,
        )
        self.assertIn('CUDA_KEEPER_GPUS="${CUDA_KEEPER_GPUS:-all}"', self.text)
        self.assertNotIn("error=vllm_forbids_cuda_keeper", self.text)
        self.assertIn(
            'CUDA keeper must cover every requested local GPU exactly once',
            self.text,
        )
        keeper_launch = self.text.index('stage "start_cuda_keeper"')
        keeper_guard = self.text.rfind(
            'if [[ "${ENABLE_CUDA_KEEPER}" == "1" ]]', 0, keeper_launch
        )
        self.assertGreaterEqual(keeper_guard, 0)
        self.assertLess(keeper_launch, self.text.index('stage "generate_six_user_qa"'))

    def test_cuda_keeper_maps_torch_local_devices_to_nvml_devices(self) -> None:
        keeper_text = CUDA_KEEPER.read_text(encoding="utf-8")

        compile(keeper_text, str(CUDA_KEEPER), "exec")
        self.assertIn("_nvml_handle_for_logical_device", keeper_text)
        self.assertIn('os.environ.get("CUDA_VISIBLE_DEVICES", "")', keeper_text)
        self.assertIn("nvmlDeviceGetHandleByUUID", keeper_text)
        self.assertIn("cuda_keeper_device_mapping", keeper_text)

    def test_dependency_preflights_cover_vllm_media_and_keeper_stack(self) -> None:
        self.assertIn('"${PYTHON}" -P -m pip check', self.text)
        self.assertIn("import decord", self.text)
        self.assertIn("import pynvml", self.text)
        self.assertNotIn("from torchcodec.decoders import VideoDecoder", self.text)
        self.assertIn('if video_backend != "decord"', self.text)
        self.assertIn("from transformers import CLIPModel, CLIPProcessor", self.text)
        self.assertIn("from vllm.vllm_flash_attn import (", self.text)
        self.assertIn("is_fa_version_supported", self.text)

    def test_flashinfer_jit_does_not_inherit_base_conda_headers(self) -> None:
        sanitize = self.text.index(
            'stage "sanitize_native_extension_build_environment"'
        )
        engine = self.text.index('stage "generate_six_user_qa"')
        self.assertIn(
            "unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH "
            "OBJC_INCLUDE_PATH LIBRARY_PATH",
            self.text,
        )
        self.assertLess(sanitize, engine)

    def test_native_video_runtime_falls_back_to_training_environment(self) -> None:
        self.assertIn('FFMPEG_SOURCE="FFMPEG_ENV"', self.text)
        self.assertIn('FFMPEG_SOURCE="TRAIN_ENV"', self.text)
        self.assertIn('FFMPEG_SOURCE="activated_PATH"', self.text)
        self.assertIn(
            'export PATH="$(dirname "${FFMPEG_BINARY}"):$(dirname "${FFPROBE_BINARY}"):${PATH}"',
            self.text,
        )
        self.assertIn('FFMPEG_RUNTIME_ROOT="$(dirname "$(dirname "${FFMPEG_BINARY}")")"', self.text)
        self.assertIn(
            'export LD_LIBRARY_PATH="${FFMPEG_RUNTIME_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"',
            self.text,
        )
        self.assertNotIn("from torchcodec.decoders import VideoDecoder", self.text)
        self.assertIn('FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"', self.text)

    def test_storage_and_video_preflights_precede_model_work(self) -> None:
        storage = self.text.index('stage "storage_preflight"')
        video = self.text.index('stage "video_runtime_preflight"')
        encoder = self.text.index('stage "prepare_ten_minute_six_user_candidates"')
        generation = self.text.index('stage "generate_six_user_qa"')
        self.assertLess(storage, video)
        self.assertLess(video, encoder)
        self.assertLess(encoder, generation)

    def test_preparation_receives_half_fps_and_large_lazy_candidate_pool(self) -> None:
        self.assertIn('--target-count "${TARGET_COUNT}"', self.text)
        self.assertIn('--source-window-count "${SOURCE_WINDOW_COUNT}"', self.text)
        self.assertIn('--judge-video-fps "${JUDGE_VIDEO_FPS}"', self.text)
        self.assertIn('--ffmpeg-binary "${FFMPEG_BINARY}"', self.text)
        self.assertIn(
            "prepared_judge_video_fps != expected_judge_video_fps",
            self.text,
        )

    def test_individual_inference_infrastructure_failures_are_logged_for_skipping(self) -> None:
        self.assertIn(
            '--infrastructure-skipped-output "${OUTDIR}/qa_mcq.infrastructure_skipped.jsonl"',
            self.text,
        )


if __name__ == "__main__":
    unittest.main()
