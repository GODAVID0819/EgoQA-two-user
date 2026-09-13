from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
PROBE = PROJECT_ROOT / "hpc" / "qa" / "smoke" / "run_six_user_qa_runtime_probe.sbatch"
PILOT = ROOT / "hpc" / "qa" / "experiments" / "run_six_user_qa_pilot_40.sbatch"
PACKETS_100 = (
    PROJECT_ROOT / "hpc" / "qa" / "production" / "run_six_user_qa_packets_100.sbatch"
)
REUSE_PACKETS_100 = (
    PROJECT_ROOT
    / "hpc"
    / "qa"
    / "production"
    / "run_six_user_qa_reuse_packets_100.sbatch"
)
PACKAGE_PROBE_MIRROR = (
    ROOT / "hpc" / "qa" / "smoke" / "run_six_user_qa_runtime_probe.sbatch"
)
PACKAGE_PACKETS_100_MIRROR = (
    ROOT / "hpc" / "qa" / "production" / "run_six_user_qa_packets_100.sbatch"
)
PACKAGE_REUSE_PACKETS_100_MIRROR = (
    ROOT / "hpc" / "qa" / "production" / "run_six_user_qa_reuse_packets_100.sbatch"
)
OLD_PILOT = ROOT / "hpc" / "qa" / "experiments" / "run_six_user_qa_pilot_5.sbatch"
RUNBOOK = ROOT / "docs" / "SIX_USER_QA_TORCH_RUNBOOK_CN.md"
QWEN_RUNNER = ROOT / "qwen3vl_runner.py"


class SixUserTorchJobContractTests(unittest.TestCase):
    def read(self, path: Path) -> str:
        self.assertTrue(path.is_file(), path)
        return path.read_text(encoding="utf-8")

    def effective_text(self, path: Path) -> str:
        text = self.read(path)
        if path in {PILOT, PACKETS_100, REUSE_PACKETS_100}:
            text += "\n" + self.read(PROBE)
        return text

    def assert_common_contract(self, text: str) -> None:
        former_user = "".join(("x", "l", "6775"))
        self.assertIn("#SBATCH --account=torch_pr_674_tandon_advanced", text)
        self.assertNotIn("#SBATCH --partition=", text)
        self.assertNotIn("#SBATCH --qos=", text)
        self.assertIn("/scratch/${USER}/", text)
        self.assertNotIn(former_user, text)
        self.assertIn(
            'PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/Long-video-understanding-clip}"',
            text,
        )
        self.assertIn(
            'PACKAGE_ROOT="${PACKAGE_ROOT:-${PROJECT_ROOT}/egolife_two_user_qa/multi-user}"',
            text,
        )
        self.assertIn(
            'OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/egolife_two_user_qa/multi-user/outputs/six_user_qa}"',
            text,
        )
        self.assertNotIn("egolife_two_user_qa/multiuser", text.lower())
        self.assertIn(
            '"${PYTHON}" -P "${PACKAGE_ROOT}/group_relative_clip_sampling.py"',
            text,
        )
        self.assertNotIn("EgoQA-two-user-grpo-clean", text)
        self.assertIn("#SBATCH --constraint=h100", text)
        self.assertIn("#SBATCH --mem=64G", text)
        self.assertIn('OUTDIR="${OUTPUT_ROOT}/${RUN_MODE}_${SLURM_JOB_ID}"', text)
        self.assertIn('JOB_SCRATCH_ROOT="/scratch/${USER}/job_scratch/', text)
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
            self.assertRegex(text, rf"export {variable}=", variable)
        self.assertIn('"${PYTHON}" -P -m training.torch_storage_preflight', text)
        self.assertIn('--allowed-root "/scratch/${USER}"', text)
        self.assertIn(
            'MODEL_ID="${MODEL_ID:-${QWEN_MODEL_ID:-Qwen/Qwen3.6-27B}}"',
            text,
        )
        self.assertIn(
            'MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-/scratch/${USER}/hf_cache}"',
            text,
        )
        self.assertIn('stage_start "validate_model_runtime"', text)
        self.assertIn("AutoConfig.from_pretrained(model_id", text)
        self.assertIn('--model-id "${MODEL_ID}"', text)
        self.assertNotIn("MODEL_DIR", text)
        self.assertNotIn("/scratch/hm2991/models", text)
        self.assertIn("storage_preflight.json", text)
        self.assertIn("job_manifest.json", text)
        self.assertIn("six_user_qa_result.json", text)
        self.assertIn("--selected-count 6", text)
        self.assertIn("--min-group-size 6", text)
        self.assertIn("--high-similarity-interval-threshold 0.82", text)
        self.assertIn("--min-pruned-video-seconds 8", text)
        for inactive_pair_option in (
            "--pairs-per-group",
            "--topk",
            "--min-topk-sim",
            "--min-mean-sim",
            "--max-mean-sim",
            "--preserve-shared-anchor-seconds",
            "--pruning-protection-mode",
        ):
            self.assertNotIn(inactive_pair_option, text)
        self.assertIn(
            "speaker_all_clustering_frames_five_provider_retained_cluster_frames",
            text,
        )
        self.assertIn("speaker_all_clustering_frames", text)
        self.assertIn("provider_retained_cluster_frames", text)
        self.assertIn('clip.get("generator_media_mode") != "all_clustering_frames_only"', text)
        self.assertIn('clip.get("generator_media_mode") != "retained_cluster_frames_only"', text)
        self.assertIn("speaker generator must receive all clustering frames", text)
        self.assertIn("provider generator received a frame from a pruned cluster", text)
        self.assertIn("generator packet must not expose a generator MP4", text)
        self.assertIn("six_user_speaker_consensus", text)
        self.assertIn("speaker_provider_all_pairs_provider_only", text)
        self.assertIn("every_speaker_cluster_x_every_provider_cluster", text)
        self.assertIn("six-user pruning did not compare every asker-provider cluster pair", text)
        self.assertIn("asker clusters or intervals were pruned", text)
        self.assertIn("pruning event must delete only its matched provider cluster", text)
        self.assertIn("expected exactly", text)
        self.assertIn('"${EVIDENCE_TARGET}" <<\'PY\'', text)
        self.assertIn("generation_attempts_by_evidence", text)
        self.assertIn("every candidate packet must enter generation", text)
        self.assertIn("generation attempts must be contiguous and ordered", text)
        self.assertIn("generation attempts exceed MAX_ATTEMPTS", text)
        self.assertIn("rejected packet must exhaust MAX_ATTEMPTS", text)
        self.assertIn(
            '"generation_packet_count": len(generation_attempts_by_evidence)',
            text,
        )
        self.assertIn('"generation_attempt_count": len(generation)', text)
        self.assertIn('"retried_packet_count": sum(', text)
        self.assertIn('"max_attempts_per_packet": max_attempts', text)
        self.assertIn("speaker_attempts", text)
        self.assertIn("speaker_only_answerable", text)
        self.assertIn("all_six_answerable", text)
        self.assertIn('"answerability_mode": "shared_fact_visibility_audit"', text)
        self.assertIn("answerability_evaluated_condition_count", text)
        self.assertIn("generator_image_count", text)
        self.assertIn("generator_video_count must be 0", text)
        self.assertIn("groundedness_visual_call_count", text)
        self.assertIn("groundedness_max_videos_per_call", text)
        self.assertIn("per_user_source_segment_map_reduce", text)
        self.assertIn("answerability_call_count", text)
        self.assertIn('"answerability_call_count": answerability_call_count', text)
        self.assertIn('"answerability_evaluated_condition_count": 2', text)
        self.assertIn('row.get("stage") == "qa_formality_judge"', text)
        self.assertIn("speaker would naturally have and genuinely want to ask", text)
        self.assertIn("speaker's sampled frames alone must remain insufficient", text)
        self.assertIn("Concurrent-activity restriction", text)
        self.assertIn("other_person_activity_query", text)
        self.assertIn("qa_formality no longer blocks concurrent-activity questions", text)
        self.assertIn("concurrent-activity formality restriction is incomplete", text)
        self.assertIn("qa_formality prompt is missing the concurrent-activity blocker", text)
        self.assertIn("Six-user interaction-chain example", text)
        self.assertIn("Stage marker: answerability_fact_plan", text)
        self.assertIn("Stage marker: answerability_user_fact_audit", text)
        self.assertIn("Stage marker: answerability_condition_aggregation", text)
        self.assertIn("six-user answerability map prompt requests a forced-choice answer", text)
        self.assertIn('formality_check.get("status") != "PASS"', text)
        self.assertIn('accepted_answerability_call_counts', text)
        self.assertIn('rows = by_qa_attempt.get((qa_id, attempt), [])', text)
        self.assertIn('speaker_answerability.get("media_summary")', text)
        self.assertIn('all_six_answerability.get("media_summary")', text)
        self.assertNotIn('speaker_answerability.get("video_paths")', text)
        self.assertIn('"all_six_answerable": True', text)
        self.assertNotIn('"speaker_only_correct"', text)
        self.assertNotIn('"all_six_correct"', text)
        self.assertNotIn('"cross_view_gain"', text)
        self.assertNotIn('"all_six_wrong_count"', text)
        self.assertNotIn('"all_six_wrong_rate"', text)
        self.assertIn("SIX_USER_QA_JOB_FINISHED", text)
        self.assertIn(
            'CUDA_KEEPER_SCRIPT="${CUDA_KEEPER_SCRIPT:-${PROJECT_ROOT}/hpc/cuda.py}"',
            text,
        )
        self.assertIn('stage_start "start_cuda_keeper"', text)
        self.assertIn('"${PYTHON}" -P "${CUDA_KEEPER_SCRIPT}"', text)
        self.assertIn('stage_start "validate_python_environment"', text)
        self.assertIn('stage_start "validate_multi_user_package_import"', text)
        self.assertIn("wrong EgoLife package imported", text)
        self.assertIn("multi_user_package_dir=", text)
        self.assertIn("multi_user_runner_path=", text)
        self.assertIn('expected_prefix = Path(sys.argv[1]).resolve()', text)
        self.assertIn('actual_prefix = Path(sys.prefix).resolve()', text)
        self.assertIn('os.environ.get("CONDA_PREFIX", "")', text)
        self.assertNotRegex(text, r"(?m)^(?:if )?python(?:\s|$)")
        self.assertNotRegex(
            text,
            r'(?m)^(?:if )?"\$\{PYTHON\}" (?!-P(?:\s|$))',
        )
        self.assertIn("trap cleanup EXIT", text)
        self.assertIn("stage_status.json", text)
        self.assertIn("status=failed", text)
        self.assertIn('stage_finish "six_user_qa_complete"', text)
        self.assertLess(
            text.index('stage_start "start_cuda_keeper"'),
            text.index('stage_start "activate_environment"'),
        )
        self.assertNotIn("SIX_USER_QA_RUNTIME_PROBE_PASSED", text)
        self.assertIn("branch", text)
        self.assertIn("dirty_state", text)
        self.assertNotIn("latest_", text)

        self.assertIn('stage_start "resolve_ffmpeg_runtime"', text)
        self.assertIn('FFMPEG_SOURCE="FFMPEG_ENV"', text)
        self.assertIn('FFMPEG_SOURCE="TRAIN_ENV"', text)
        self.assertIn('FFMPEG_SOURCE="activated_PATH"', text)
        self.assertIn("FFmpeg runtime not found", text)
        self.assertIn('--ffmpeg-binary "${FFMPEG_BINARY}"', text)
        self.assertNotIn('test -x "${FFMPEG_ENV}/bin/ffmpeg"', text)
        path_export = text.index(
            'export PATH="$(dirname "${FFMPEG_BINARY}"):$(dirname "${FFPROBE_BINARY}"):${PATH}"'
        )
        library_export = text.index(
            'export LD_LIBRARY_PATH="${TORCH_RUNTIME_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"'
        )
        self.assertNotIn('LD_LIBRARY_PATH="${FFMPEG_ENV}/lib:', text)
        self.assertNotIn('LD_LIBRARY_PATH="${TRAIN_ENV}/lib:', text)
        self.assertIn('PYTORCH_CUDNN_LIB="${PYTHON_SITE_PACKAGES}/nvidia/cudnn/lib"', text)
        self.assertIn('export FORCE_QWENVL_VIDEO_READER="decord"', text)
        self.assertIn('export QWEN_MEMORY_SAFE_MIN_VIDEO_PIXELS="3136"', text)
        self.assertIn(
            'if "${PYTHON}" -P -m pip check > "${OUTDIR}/pip_check.txt" 2>&1; then',
            text,
        )
        self.assertIn("known_decord_platform_metadata_warning", text)
        self.assertIn("from qwen_vl_utils.vision_process import get_video_reader_backend", text)
        self.assertIn("from qwen_vl_utils import process_vision_info", text)
        self.assertIn("six_user_image_preflight_passed", text)
        self.assertIn('"type": "image"', text)
        self.assertIn('"image": item', text)
        self.assertIn('"max_pixels": 65536', text)
        self.assertIn("--max-image-pixels 65536", text)
        self.assertIn('if video_backend != "decord":', text)
        self.assertIn("from torchcodec.decoders import VideoDecoder", text)
        self.assertIn("torchcodec_decoder=", text)

        decoder_import = text.index("import decord")
        torchcodec_import = text.index("from torchcodec.decoders import VideoDecoder")
        storage_preflight = text.index(
            '"${PYTHON}" -P -m training.torch_storage_preflight'
        )
        job_manifest = text.index('"${PYTHON}" -P - "${OUTDIR}/job_manifest.json"')
        pip_check = text.index(
            'if "${PYTHON}" -P -m pip check > "${OUTDIR}/pip_check.txt" 2>&1; then'
        )
        image_preflight = text.index("six_user_image_preflight_passed")
        model_command = text.index("generate_video_qa_loop")
        self.assertLess(path_export, decoder_import)
        self.assertLess(library_export, decoder_import)
        self.assertLess(path_export, torchcodec_import)
        self.assertLess(library_export, torchcodec_import)
        self.assertLess(storage_preflight, model_command)
        self.assertLess(job_manifest, pip_check)
        self.assertLess(image_preflight, model_command)

    def test_runtime_probe_contract(self) -> None:
        text = self.effective_text(PROBE)
        self.assert_common_contract(text)
        self.assertIn('export QWEN_MEMORY_SAFE_ATTN_IMPLEMENTATION="sdpa"', text)
        self.assertNotIn(
            'export QWEN_MEMORY_SAFE_ATTN_IMPLEMENTATION="flash_attention_2"',
            text,
        )
        self.assertIn("#SBATCH --time=01:30:00", text)
        self.assertIn("six_user_qa_runtime_probe", text)
        self.assertIn("ACCEPTED_TARGET:-1", text)
        self.assertIn("EVIDENCE_TARGET:-1", text)
        self.assertIn("MAX_GROUPS:-16", text)
        self.assertIn("MAX_ATTEMPTS:-1", text)

    def test_project_level_launchers_match_package_mirrors(self) -> None:
        self.assertEqual(self.read(PROBE), self.read(PACKAGE_PROBE_MIRROR))
        self.assertEqual(
            self.read(PACKETS_100),
            self.read(PACKAGE_PACKETS_100_MIRROR),
        )
        self.assertEqual(
            self.read(REUSE_PACKETS_100),
            self.read(PACKAGE_REUSE_PACKETS_100_MIRROR),
        )

    def test_six_user_memory_safe_runner_forwards_video_pixel_floor(self) -> None:
        text = self.read(QWEN_RUNNER)
        self.assertIn("MEMORY_SAFE_DEFAULT_MIN_VIDEO_PIXELS = 4 * 28 * 28", text)
        self.assertIn('"QWEN_MEMORY_SAFE_MIN_VIDEO_PIXELS"', text)
        self.assertIn("min_video_pixels=min_video_pixels", text)
        self.assertIn('video_content["min_pixels"] = self.min_video_pixels', text)
        self.assertIn('"low_cpu_mem_usage": True', text)
        self.assertIn('os.getenv("QWEN_MEMORY_SAFE_DEVICE_MAP", "cuda")', text)
        self.assertIn("release_unused_host_memory()", text)
        self.assertIn("qwen_memory_safe_host_cleanup", text)

    def test_memory_safe_runner_monitors_every_balanced_cuda_device(self) -> None:
        text = self.read(QWEN_RUNNER)

        self.assertIn("self.cuda_devices = tuple(", text)
        self.assertIn("Loaded model did not span the required CUDA device count", text)
        self.assertIn('os.getenv("QWEN_MEMORY_SAFE_REQUIRED_MODEL_GPU_COUNT", "1")', text)
        self.assertIn('f"device={cuda_device} free_gib=', text)
        self.assertIn("for cuda_device in cuda_devices:", text)
        self.assertIn('"qwen_memory_safe_vram_total "', text)
        self.assertIn("with self.torch.cuda.device(cuda_device):", text)
        self.assertIn("0 < min_video_pixels <= max_image_pixels", text)

    def test_runtime_probe_embedded_python_is_syntactically_valid(self) -> None:
        text = self.read(PROBE)
        blocks = re.findall(r"<<'?PY'?\n(.*?)\nPY", text, flags=re.DOTALL)

        self.assertGreaterEqual(len(blocks), 6)
        for index, block in enumerate(blocks):
            compile(block, f"{PROBE.name}:heredoc-{index}", "exec")

    def test_runbook_submits_the_100_packet_job_directly(self) -> None:
        text = self.read(RUNBOOK)
        former_user = "".join(("x", "l", "6775"))
        self.assertNotIn("PROBE_OK", text)
        self.assertNotRegex(text, r"(?m)^\s*sbatch --test-only")
        self.assertNotIn("#SBATCH --partition=", self.read(PACKETS_100))
        self.assertNotIn("#SBATCH --qos=", self.read(PACKETS_100))
        self.assertIn("run_six_user_qa_reuse_packets_100.sbatch", text)
        self.assertIn("EVIDENCE_TARGET=100", text)
        self.assertIn("MAX_ATTEMPTS=3", text)
        self.assertIn("恰好包含 100 个 packet", text)
        self.assertIn("every_speaker_cluster_x_every_provider_cluster", text)
        self.assertIn("Torch 登录用户与 scratch 所有者：`hm2991`", text)
        self.assertIn("/scratch/hm2991/Long-video-understanding-clip", text)
        self.assertIn("egolife_two_user_qa/multi-user", text)
        self.assertNotIn(former_user, text)

    def test_pilot_contract(self) -> None:
        text = self.effective_text(PILOT)
        self.assert_common_contract(text)
        self.assertFalse(OLD_PILOT.is_file(), OLD_PILOT)
        self.assertIn("#SBATCH --time=24:00:00", text)
        self.assertIn('RUN_MODE="six_user_qa_pilot_40"', text)
        self.assertIn('ACCEPTED_TARGET="40"', text)
        self.assertIn('EVIDENCE_TARGET="40"', text)
        self.assertIn('MAX_GROUPS="320"', text)
        self.assertIn('MAX_ATTEMPTS="1"', text)
        self.assertIn('ALLOW_PARTIAL="1"', text)
        self.assertIn(
            'bash "${PROJECT_ROOT}/hpc/qa/smoke/run_six_user_qa_runtime_probe.sbatch"',
            self.read(PILOT),
        )
        self.assertIn('--max-attempts "${MAX_ATTEMPTS}"', text)
        self.assertIn('status = "partial"', text)
        self.assertIn('if status == "failed":', text)

    def test_100_packet_generation_contract(self) -> None:
        text = self.effective_text(PACKETS_100)
        wrapper_text = self.read(PACKETS_100)
        self.assert_common_contract(text)
        self.assertIn("#SBATCH --time=48:00:00", text)
        self.assertIn('RUN_MODE="six_user_qa_packets_100"', text)
        self.assertIn('ACCEPTED_TARGET="100"', text)
        self.assertIn('EVIDENCE_TARGET="100"', text)
        self.assertIn('MAX_GROUPS="800"', text)
        self.assertIn('MAX_ATTEMPTS="3"', text)
        self.assertIn('ALLOW_PARTIAL="1"', text)
        self.assertIn(
            'OUTPUT_ROOT="${PROJECT_ROOT}/egolife_two_user_qa/multi-user/outputs/six_user_qa"',
            wrapper_text,
        )
        self.assertIn("export PROJECT_ROOT PACKAGE_ROOT OUTPUT_ROOT", wrapper_text)
        self.assertIn(
            'bash "${PROJECT_ROOT}/hpc/qa/smoke/run_six_user_qa_runtime_probe.sbatch"',
            self.read(PACKETS_100),
        )
        self.assertIn("Mine exactly 100 six-user evidence packets", text)
        self.assertIn("up to three attempts per packet", text)
        self.assertIn('WRAPPER_STAGE="production_wrapper_bootstrap"', text)
        self.assertIn('WRAPPER_STAGE="handoff_to_six_user_runtime"', text)
        self.assertIn('--target-count "${EVIDENCE_TARGET}"', text)
        self.assertIn('--max-attempts "${MAX_ATTEMPTS}"', text)

    def test_100_packet_reuse_contract(self) -> None:
        text = self.effective_text(REUSE_PACKETS_100)
        wrapper_text = self.read(REUSE_PACKETS_100)
        self.assert_common_contract(text)
        self.assertIn("#SBATCH --time=48:00:00", text)
        self.assertIn('RUN_MODE="six_user_baseline_neutral_reuse_packets_100"', text)
        self.assertIn('ACCEPTED_TARGET="100"', text)
        self.assertIn('EVIDENCE_TARGET="100"', text)
        self.assertIn('MAX_GROUPS="800"', text)
        self.assertIn('MAX_ATTEMPTS="3"', text)
        self.assertIn('ALLOW_PARTIAL="1"', text)
        self.assertIn('REFERENCE_JOB_ID="16220358"', text)
        self.assertIn('PREPARED_EVIDENCE_OFFSET="0"', text)
        self.assertIn(
            'PREPARED_EVIDENCE_SOURCE="${OUTPUT_ROOT}/six_user_baseline_neutral_packets_100_${REFERENCE_JOB_ID}/six_user_candidates.jsonl"',
            wrapper_text,
        )
        self.assertIn("Reuse all 100 prepared evidence packets", wrapper_text)
        self.assertIn(
            'bash "${PROJECT_ROOT}/hpc/qa/smoke/run_six_user_qa_runtime_probe.sbatch"',
            wrapper_text,
        )

    def test_jobs_do_not_modify_training_contracts(self) -> None:
        for path in (PROBE, PILOT, PACKETS_100, REUSE_PACKETS_100):
            text = self.effective_text(path).lower()
            self.assertNotRegex(text, r"python\s+-m\s+training\.(?:grpo|dpo)")
            self.assertNotRegex(text, r"python\s+[^\n]*(?:reviewer|optimizer|checkpoint)")

    def test_every_sbatch_uses_one_job_specific_output_contract(self) -> None:
        for path in (PROBE, PILOT, PACKETS_100, REUSE_PACKETS_100):
            text = self.effective_text(path)
            output_assignments = re.findall(
                r'^OUTDIR="\$\{OUTPUT_ROOT\}/\$\{RUN_MODE\}_\$\{SLURM_JOB_ID\}"$',
                text,
                flags=re.MULTILINE,
            )
            self.assertEqual(output_assignments, [
                'OUTDIR="${OUTPUT_ROOT}/${RUN_MODE}_${SLURM_JOB_ID}"'
            ])


if __name__ == "__main__":
    unittest.main()
