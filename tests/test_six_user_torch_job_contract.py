from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "hpc" / "qa" / "smoke" / "run_six_user_qa_runtime_probe.sbatch"
PILOT = ROOT / "hpc" / "qa" / "experiments" / "run_six_user_qa_pilot_5.sbatch"


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
        self.assertIn("three_pruned_three_full_videos", text)
        self.assertIn("speaker_only_correct", text)
        self.assertIn("all_six_correct", text)
        self.assertIn("cross_view_gain", text)
        self.assertIn("answerability_evaluated_condition_count", text)
        self.assertIn("all_six_wrong_count", text)
        self.assertIn("all_six_wrong_rate", text)
        self.assertIn("generator_video_count", text)
        self.assertIn("groundedness_video_count", text)
        self.assertIn("answerability_call_count", text)
        self.assertIn("branch", text)
        self.assertIn("dirty_state", text)
        self.assertNotIn("latest_", text)

        path_export = text.index('export PATH="${FFMPEG_ENV}/bin:${PATH}"')
        library_export = text.index(
            'export LD_LIBRARY_PATH="${FFMPEG_ENV}/lib:${LD_LIBRARY_PATH:-}"'
        )
        decoder_import = text.index("from torchcodec.decoders import VideoDecoder")
        storage_preflight = text.index("python -m training.torch_storage_preflight")
        model_command = text.index("generate_video_qa_loop")
        self.assertLess(path_export, decoder_import)
        self.assertLess(library_export, decoder_import)
        self.assertLess(storage_preflight, model_command)

    def test_runtime_probe_contract(self) -> None:
        text = self.effective_text(PROBE)
        self.assert_common_contract(text)
        self.assertIn("six_user_qa_runtime_probe", text)
        self.assertIn("ACCEPTED_TARGET:-1", text)
        self.assertIn("EVIDENCE_TARGET:-8", text)

    def test_pilot_contract(self) -> None:
        text = self.effective_text(PILOT)
        self.assert_common_contract(text)
        self.assertIn('RUN_MODE="six_user_qa_pilot_5"', text)
        self.assertIn('ACCEPTED_TARGET="5"', text)
        self.assertIn('EVIDENCE_TARGET="30"', text)

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
