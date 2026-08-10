from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
HPC = ROOT / "hpc/grpo_v3/annotated_preference"
SCRIPTS = (
    "common.sh",
    "gate0_data.sbatch",
    "structure_probe.sbatch",
    "smoke1.sbatch",
    "overfit_probe.sbatch",
    "train.sbatch",
    "evaluate.sbatch",
)


def _read(name: str) -> str:
    return (HPC / name).read_text(encoding="utf-8")


def _assert_bash_syntax_if_available(test: unittest.TestCase) -> None:
    bash = shutil.which("bash")
    if bash is None:
        return
    try:
        probe = subprocess.run(
            [bash, "--version"], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return
    if probe.returncode != 0:
        return
    for name in SCRIPTS:
        result = subprocess.run(
            [bash, "-n", str(HPC / name)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        test.assertEqual(result.returncode, 0, f"{name}: {result.stderr}")


class AnnotatedPreferenceSlurmTests(unittest.TestCase):
    def test_common_and_gate0_contract(self) -> None:
        common = _read("common.sh")
        gate0 = _read("gate0_data.sbatch")
        for value in (
            "/scratch/xl6775/projects/EgoQA-two-user-annotated-pareto-dpo",
            "/scratch/xl6775/envs/egoqa-ms-swift-v4.2.2-vllm024",
            "/scratch/xl6775/models/Qwen3-VL-8B-Instruct",
            "data_RLHF/annotated_preference",
            "outputs/annotated_preference",
            "rlhf_candidate_scores_merged_70_packets.csv",
            "split_60_10.json",
            "media_map.json",
            "FPS=${FPS:-1}",
            "FPS_MIN_FRAMES=${FPS_MIN_FRAMES:-4}",
            "FPS_MAX_FRAMES=${FPS_MAX_FRAMES:-64}",
            "VIDEO_MAX_PIXELS=${VIDEO_MAX_PIXELS:-50176}",
            "training.torch_storage_preflight",
            "torchcodec.decoders import VideoDecoder",
            "SLURM_JOB_ID",
            "JOB_SCRATCH_ROOT",
            "LD_LIBRARY_PATH",
        ):
            self.assertIn(value, common)
        for variable in (
            "HOME", "XDG_CACHE_HOME", "HF_HOME", "HF_DATASETS_CACHE",
            "MODELSCOPE_CACHE", "TORCH_HOME", "TRITON_CACHE_DIR",
            "TORCHINDUCTOR_CACHE_DIR", "VLLM_CACHE_ROOT", "CUDA_CACHE_PATH",
            "FLASHINFER_WORKSPACE_BASE", "TMPDIR", "TMP", "TEMP",
        ):
            self.assertIn(f"export {variable}=", common)
        self.assertNotIn("latest_", common.lower())
        for token in (
            "unittest discover",
            "annotated_preference",
            "compileall",
            "build_dataset build",
            "--csv \"${CSV_PATH}\"",
            "--split \"${SPLIT_PATH}\"",
            "--media-map \"${MEDIA_MAP}\"",
        ):
            self.assertIn(token, gate0)
        for output in (
            "train_dpo.jsonl", "validation_dpo.jsonl", "train_pair_index.jsonl",
            "validation_pair_index.jsonl", "overfit_4_dpo.jsonl",
            "pareto_audit.json", "dataset_manifest.json",
        ):
            self.assertIn(f'test -s "${{DPO_DATA_DIR}}/{output}"', gate0)
        _assert_bash_syntax_if_available(self)

    def test_structure_smoke_and_overfit_contract(self) -> None:
        structure = _read("structure_probe.sbatch")
        smoke = _read("smoke1.sbatch")
        overfit = _read("overfit_probe.sbatch")
        for artifact in (
            "storage_preflight.json", "dependencies.txt", "dataset_preview.json",
            "structure_probe.json",
        ):
            self.assertIn(artifact, structure)
        self.assertIn("train_dpo.jsonl", structure)
        self.assertIn("PAIR_LIMIT=1", structure)
        for text in (smoke, overfit):
            for token in (
                "swift rlhf", "--rlhf_type dpo", "--model \"${MODEL_DIR}\"",
                "--tuner_type lora", "--freeze_vit true", "--freeze_aligner true",
                "--target_modules q_proj v_proj", "--lora_rank 8", "--lora_alpha 16",
                "--torch_dtype bfloat16", "--max_length 32768",
                "--gradient_checkpointing true", "--learning_rate 1e-5", "--beta 0.1",
                "--dataset_shuffle false", "--seed 42", "--data_seed 42",
            ):
                self.assertIn(token, text)
            lowered = text.lower()
            for forbidden in (
                "--rlhf_type grpo", "reward_plugin", "reward_function", "reward_funcs",
                "snapshot_download", "from_pretrained(", "latest_",
            ):
                self.assertNotIn(forbidden, lowered)
        self.assertIn("--dataset \"${DPO_DATA_DIR}/train_dpo.jsonl\"", smoke)
        self.assertIn("--val_dataset \"${DPO_DATA_DIR}/validation_dpo.jsonl\"", smoke)
        self.assertIn("--max_steps 1", smoke)
        self.assertIn("--dataset \"${DPO_DATA_DIR}/overfit_4_dpo.jsonl\"", overfit)
        self.assertIn("MAX_STEPS=${MAX_STEPS:-40}", overfit)
        self.assertIn("annotated_preference.analyze", overfit)
        self.assertIn("--mode overfit", overfit)

    def test_train_and_evaluate_contract(self) -> None:
        train = _read("train.sbatch")
        evaluate = _read("evaluate.sbatch")
        for token in (
            "OVERFIT_RESULT", 'status") == "passed"',
            "NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}",
            "PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}",
            "GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}",
            "LEARNING_RATE=${LEARNING_RATE:-1e-5}", "BETA=${BETA:-0.1}",
            'OUTDIR="${OUTPUT_ROOT}/train_${SLURM_JOB_ID}"',
            "--dataset \"${DPO_DATA_DIR}/train_dpo.jsonl\"",
            "--val_dataset \"${DPO_DATA_DIR}/validation_dpo.jsonl\"",
            "resolved_command.txt", "environment.txt", "dataset_manifest.json",
            "trainer_state.json", "adapter", "parameter_audit.json", "dpo_gate_result.json",
        ):
            self.assertIn(token, train)
        for token in (
            'ADAPTER_DIR="${ADAPTER_DIR:?', 'TRAIN_JOB_ID="${TRAIN_JOB_ID:?',
            'OUTDIR="${OUTPUT_ROOT}/validation_${SLURM_JOB_ID}"',
            "validation_dpo.jsonl", "fixed_pair_accuracy", "chosen_rejected_margin",
            "eval_loss", "annotated_preference.analyze", "--mode validation",
        ):
            self.assertIn(token, evaluate)
        for text in (train, evaluate):
            lowered = text.lower()
            self.assertIn("swift rlhf", lowered)
            self.assertIn("--rlhf_type dpo", lowered)
            for forbidden in (
                "--rlhf_type grpo", "reward_plugin", "reward_function", "reward_funcs",
                "snapshot_download", "latest_",
            ):
                self.assertNotIn(forbidden, lowered)


if __name__ == "__main__":
    unittest.main()
