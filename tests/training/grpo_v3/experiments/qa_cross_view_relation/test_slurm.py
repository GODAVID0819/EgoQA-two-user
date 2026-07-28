from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[5]
SCRIPT = REPO_ROOT / "hpc" / "grpo_v3" / "qa_cross_view_relation" / "smoke1.sbatch"
V3_SMOKE = REPO_ROOT / "hpc" / "grpo_v3" / "qa_cross_view_relation" / "smoke1_v3.sbatch"
V3_PROBE20 = REPO_ROOT / "hpc" / "grpo_v3" / "qa_cross_view_relation" / "probe20_v3.sbatch"
V3_PROBE = REPO_ROOT / "hpc" / "grpo_v3" / "qa_cross_view_relation" / "probe120_v3.sbatch"


class CrossViewRelationSlurmTests(unittest.TestCase):
    def test_native_video_smoke_uses_ffmpeg_runtime_and_torchcodec_preflight(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")

        self.assertIn("FFMPEG_ENV", text)
        self.assertIn('PATH="${FFMPEG_ENV}/bin:${PATH}"', text)
        self.assertIn('LD_LIBRARY_PATH="${FFMPEG_ENV}/lib:${LD_LIBRARY_PATH:-}"', text)
        self.assertIn('"${FFMPEG_ENV}/bin/ffmpeg" -version', text)
        self.assertIn("from torchcodec.decoders import VideoDecoder", text)
        self.assertLess(text.index("from torchcodec.decoders import VideoDecoder"), text.index('"${SWIFT}" rlhf'))

    def test_v3_uses_8b_policy_32b_text_judge_and_pretraining_gates(self) -> None:
        smoke = V3_SMOKE.read_text(encoding="utf-8")
        probe20 = V3_PROBE20.read_text(encoding="utf-8")
        probe = V3_PROBE.read_text(encoding="utf-8")
        for expected in (
            "#SBATCH --gres=gpu:3",
            "#SBATCH --constraint=h100",
            "Qwen3-VL-8B-Instruct",
            "Qwen3-32B",
            "POLICY_ADAPTER",
            "CUDA_VISIBLE_DEVICES=1,2",
            "--tensor-parallel-size 2",
            "--reward_funcs egoqa_cross_view_relation_v3",
            "--save_safetensors true",
            "storage_preflight.json",
            "heldout_cpu_preflight.json",
            "dataset_audit.json",
            "reviewer_audit_result.json",
            "--expected-reward-revision qa_cross_view_relation_v3",
            "parameter_delta_nonzero",
            "from torchcodec.decoders import VideoDecoder",
        ):
            self.assertIn(expected, smoke)
        self.assertLess(smoke.index("dataset_audit.json"), smoke.index('"${SWIFT}" rlhf'))
        self.assertLess(smoke.index("reviewer_audit_result.json"), smoke.index('"${SWIFT}" rlhf'))
        self.assertIn("EXPECTED_GROUPS=120", probe)
        self.assertIn("smoke1_v3.sbatch", probe)
        self.assertIn("PROBE_DIR", probe)
        self.assertIn("training_gate_result.json", probe)
        self.assertIn("SMOKE_DIR", probe20)
        self.assertIn("smoke_result.json", probe20)
        self.assertIn("EXPECTED_GROUPS=20", probe20)
        self.assertIn("gate_validate", smoke)
        self.assertIn("reward_std_positive", smoke)
        self.assertNotIn("latest_", probe)


if __name__ == "__main__":
    unittest.main()
