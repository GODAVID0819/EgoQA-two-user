from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "hpc" / "qa" / "smoke" / "run_six_user_qa_runtime_probe.sbatch"
PILOT = ROOT / "hpc" / "qa" / "experiments" / "run_six_user_qa_pilot_40.sbatch"
OLD_PILOT = ROOT / "hpc" / "qa" / "experiments" / "run_six_user_qa_pilot_5.sbatch"
RUNBOOK = ROOT / "docs" / "SIX_USER_QA_TORCH_RUNBOOK_CN.md"


class SixUserTorchJobContractTests(unittest.TestCase):
    def read(self, path: Path) -> str:
        self.assertTrue(path.is_file(), path)
        return path.read_text(encoding="utf-8")

    def effective_text(self, path: Path) -> str:
        text = self.read(path)
        if path == PILOT:
            text += "\n" + self.read(PROBE)
        return text

    def assert_common_contract(self, text: str) -> None:
        self.assertIn("#SBATCH --account=torch_pr_674_tandon_advanced", text)
        self.assertIn("#SBATCH --constraint=h100", text)
        self.assertIn("#SBATCH --mem=64G", text)
        self.assertIn('OUTDIR="${OUTPUT_ROOT}/${RUN_MODE}_${SLURM_JOB_ID}"', text)
        self.assertIn('JOB_SCRATCH_ROOT="/scratch/${USER}/job_scratch/', text)
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
            self.assertRegex(text, rf"export {variable}=", variable)
        self.assertIn("python -m training.torch_storage_preflight", text)
        self.assertIn("storage_preflight.json", text)
        self.assertIn("job_manifest.json", text)
        self.assertIn("six_user_qa_result.json", text)
        self.assertIn("--selected-count 6", text)
        self.assertIn("--min-group-size 6", text)
        self.assertIn("speaker_full_five_provider_pruned_videos", text)
        self.assertIn("speaker_reference_unpruned", text)
        self.assertIn("provider_similarity_pruned", text)
        self.assertIn("six_user_speaker_consensus", text)
        self.assertIn("speaker_attempts", text)
        self.assertIn("speaker_only_correct", text)
        self.assertIn("answerability_evaluated_condition_count", text)
        self.assertIn("generator_video_count", text)
        self.assertIn("groundedness_video_count", text)
        self.assertIn("answerability_call_count", text)
        self.assertIn('"answerability_call_count": answerability_call_count', text)
        self.assertIn('"answerability_evaluated_condition_count": 2', text)
        self.assertIn('row.get("stage") == "qa_formality_judge"', text)
        self.assertIn('formality_check.get("status") != "PASS"', text)
        self.assertIn('accepted_answerability_call_counts', text)
        self.assertIn('rows = by_qa_attempt.get((qa_id, attempt), [])', text)
        self.assertIn('speaker_answerability.get("media_role") != "full"', text)
        self.assertIn('speaker_answerability.get("video_paths") != [speaker_full_video]', text)
        self.assertIn('all_six_answerability.get("video_paths") != full_videos', text)
        self.assertIn('"all_six_correct": True', text)
        self.assertNotIn('"cross_view_gain"', text)
        self.assertNotIn('"all_six_wrong_count"', text)
        self.assertNotIn('"all_six_wrong_rate"', text)
        self.assertIn("SIX_USER_QA_JOB_FINISHED", text)
        self.assertNotIn("SIX_USER_QA_RUNTIME_PROBE_PASSED", text)
        self.assertIn("branch", text)
        self.assertIn("dirty_state", text)
        self.assertNotIn("latest_", text)

        path_export = text.index('export PATH="${FFMPEG_ENV}/bin:${PATH}"')
        library_export = text.index(
            'export LD_LIBRARY_PATH="${FFMPEG_ENV}/lib:${LD_LIBRARY_PATH:-}"'
        )
        self.assertIn('export FORCE_QWENVL_VIDEO_READER="decord"', text)
        self.assertIn('export QWEN_MEMORY_SAFE_MIN_VIDEO_PIXELS="3136"', text)
        self.assertIn('if python -m pip check > "${OUTDIR}/pip_check.txt" 2>&1; then', text)
        self.assertIn("known_decord_platform_metadata_warning", text)
        self.assertIn("from qwen_vl_utils.vision_process import get_video_reader_backend", text)
        self.assertIn("from qwen_vl_utils import process_vision_info", text)
        self.assertIn("six_user_video_preflight_passed", text)
        self.assertIn('"min_pixels": 3136', text)
        self.assertIn('"max_pixels": 65536', text)
        self.assertIn("--max-image-pixels 65536", text)
        self.assertIn('if video_backend != "decord":', text)
        self.assertNotIn("from torchcodec", text)

        decoder_import = text.index("import decord")
        storage_preflight = text.index("python -m training.torch_storage_preflight")
        job_manifest = text.index('python - "${OUTDIR}/job_manifest.json"')
        pip_check = text.index('if python -m pip check > "${OUTDIR}/pip_check.txt" 2>&1; then')
        video_preflight = text.index("six_user_video_preflight_passed")
        model_command = text.index("generate_video_qa_loop")
        self.assertLess(path_export, decoder_import)
        self.assertLess(library_export, decoder_import)
        self.assertLess(storage_preflight, model_command)
        self.assertLess(job_manifest, pip_check)
        self.assertLess(video_preflight, model_command)

    def test_runtime_probe_contract(self) -> None:
        text = self.effective_text(PROBE)
        self.assert_common_contract(text)
        self.assertIn("#SBATCH --time=01:30:00", text)
        self.assertIn("six_user_qa_runtime_probe", text)
        self.assertIn("ACCEPTED_TARGET:-1", text)
        self.assertIn("EVIDENCE_TARGET:-1", text)
        self.assertIn("MAX_GROUPS:-16", text)
        self.assertIn("MAX_ATTEMPTS:-1", text)

    def test_runbook_requires_a_passed_probe_before_pilot_submission(self) -> None:
        text = self.read(RUNBOOK)
        self.assertIn('if [[ "${PROBE_OK}" -eq 1 ]]; then', text)
        self.assertIn('echo "STOP: runtime probe', text)
        self.assertNotIn('# if [[ "${PROBE_OK}" -eq 1 ]]; then', text)
        self.assertNotIn('PROBE_TASK_MANIFEST=TASK_MANIFEST=', text)

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
        self.assertIn('--max-attempts "${MAX_ATTEMPTS}"', text)
        self.assertIn('status = "partial"', text)
        self.assertIn('if status == "failed":', text)

    def test_jobs_do_not_modify_training_contracts(self) -> None:
        for path in (PROBE, PILOT):
            text = self.effective_text(path).lower()
            self.assertNotRegex(text, r"python\s+-m\s+training\.(?:grpo|dpo)")
            self.assertNotRegex(text, r"python\s+[^\n]*(?:reviewer|optimizer|checkpoint)")

    def test_every_sbatch_uses_one_job_specific_output_contract(self) -> None:
        for path in (PROBE, PILOT):
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
