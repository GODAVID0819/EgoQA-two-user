from __future__ import annotations

import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_LAUNCHER = (
    REPO_ROOT
    / "multi-user"
    / "run_curated_10min_qwen36_27b_qwen3vl_8b.sbatch"
)
HPC_LAUNCHER = (
    REPO_ROOT
    / "hpc"
    / "qa"
    / "experiments"
    / "run_curated_10min_qwen36_27b_qwen3vl_8b.sbatch"
)
PACKAGE_HPC_LAUNCHER = (
    REPO_ROOT
    / "multi-user"
    / "hpc"
    / "qa"
    / "experiments"
    / "run_curated_10min_qwen36_27b_qwen3vl_8b.sbatch"
)


class CuratedTenMinuteAblationSbatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.text = PACKAGE_LAUNCHER.read_text(encoding="utf-8")

    def test_project_root_and_package_copies_match(self) -> None:
        self.assertEqual(
            PACKAGE_LAUNCHER.read_bytes(),
            HPC_LAUNCHER.read_bytes(),
        )
        self.assertEqual(
            PACKAGE_LAUNCHER.read_bytes(),
            PACKAGE_HPC_LAUNCHER.read_bytes(),
        )

    def test_requests_and_requires_two_h100_or_h200_gpus(self) -> None:
        self.assertIn("#SBATCH --gres=gpu:2", self.text)
        self.assertIn('#SBATCH --constraint="h100|h200"', self.text)
        self.assertIn("REQUIRED_GPU_COUNT=2", self.text)
        self.assertIn('CUDA_KEEPER_GPUS="0,1"', self.text)
        self.assertIn('QWEN_MEMORY_SAFE_DEVICE_MAP="balanced"', self.text)
        self.assertIn(
            'QWEN_MEMORY_SAFE_REQUIRED_MODEL_GPU_COUNT="${REQUIRED_GPU_COUNT}"',
            self.text,
        )
        self.assertIn("cuda_keeper_gpu_contract_passed", self.text)

    def test_uses_hardened_storage_video_and_cudnn_preflights(self) -> None:
        self.assertIn('HPC_ROOT="${PROJECT_ROOT}/hpc"', self.text)
        self.assertIn('source "${ENV_SCRIPT}"', self.text)
        self.assertIn('STORAGE_PREFLIGHT_MODULE="egolife_two_user_qa.training.', self.text)
        self.assertIn('STORAGE_PREFLIGHT_MODULE="training.torch_storage_preflight"', self.text)
        self.assertIn('-m "${STORAGE_PREFLIGHT_MODULE}"', self.text)
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
            self.assertRegex(self.text, rf"export {variable}=")
        self.assertIn('FORCE_QWENVL_VIDEO_READER="decord"', self.text)
        self.assertIn("get_video_reader_backend", self.text)
        self.assertIn("long_context_cudnn_preflight", self.text)
        self.assertIn("PYTORCH_CUDNN_LIB", self.text)
        self.assertIn('QWEN_MEMORY_SAFE_MIN_VIDEO_PIXELS="${MIN_VIDEO_PIXELS}"', self.text)

    def test_keeps_experiment_scope_and_records_manifest(self) -> None:
        self.assertIn('VIDEO_FPS="${VIDEO_FPS:-0.5}"', self.text)
        self.assertIn('MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-131072}"', self.text)
        self.assertIn(
            'for label, scope in (("six_users", "six"), ("required_users", "pair")):',
            self.text,
        )
        self.assertIn(
            'run_model "qwen3vl_8b" "Qwen/Qwen3-VL-8B-Instruct"\n'
            'run_model "qwen36_27b" "Qwen/Qwen3.6-27B"',
            self.text,
        )
        self.assertIn(
            '"model_order": ["Qwen/Qwen3-VL-8B-Instruct", "Qwen/Qwen3.6-27B"]',
            self.text,
        )
        self.assertIn(
            'for label in ("qwen3vl_8b", "qwen36_27b"):',
            self.text,
        )
        self.assertIn('RUN_MANIFEST_PATH="${OUTDIR}/run_manifest.json"', self.text)
        self.assertIn(
            'RUN_LABEL="${RUN_LABEL:-curated_trace_v3_qwen3vl_8b_qwen36_27b_',
            self.text,
        )
        self.assertIn("completed_model_outputs=", self.text)
        self.assertIn('"expected_model_calls": 68', self.text)

    def test_embedded_python_is_syntactically_valid_and_file_is_lf_only(self) -> None:
        self.assertNotIn(b"\r", PACKAGE_LAUNCHER.read_bytes())
        blocks = re.findall(r"<<'PY'\n(.*?)\nPY", self.text, flags=re.DOTALL)
        self.assertGreaterEqual(len(blocks), 7)
        for index, block in enumerate(blocks, 1):
            compile(block, f"embedded_python_{index}.py", "exec")


if __name__ == "__main__":
    unittest.main()
