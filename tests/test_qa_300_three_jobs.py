from __future__ import annotations

import re
import unittest
from pathlib import Path


class ThreeWayQaLauncherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]
        self.hpc = self.root / "hpc"

    def test_three_launchers_are_independent_ten_hour_h100_jobs(self) -> None:
        expected = {
            "run_qa_packets_001_100.sbatch": (0, "packets_001_100"),
            "run_qa_packets_101_200.sbatch": (100, "packets_101_200"),
            "run_qa_packets_201_300.sbatch": (200, "packets_201_300"),
        }
        covered = set()
        for name, (start, label) in expected.items():
            script = (self.hpc / name).read_text(encoding="utf-8")
            self.assertIn("#SBATCH --time=10:00:00", script)
            self.assertIn("#SBATCH --constraint=h100", script)
            self.assertIn("#SBATCH --mem=128G", script)
            self.assertIn(f"export PACKET_START_INDEX={start}", script)
            self.assertIn(f'export BATCH_LABEL="{label}"', script)
            self.assertIn('exec bash "${WORKER}"', script)
            covered.update(range(start, start + 100))
        self.assertEqual(covered, set(range(300)))

    def test_worker_enforces_exclusive_slices_and_current_production_settings(self) -> None:
        script = (self.hpc / "run_qa_packet_slice_100.sh").read_text(encoding="utf-8")
        self.assertIn("PACKET_COUNT=100", script)
        self.assertIn("if len(source) != 300:", script)
        self.assertIn('QWEN_MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen3.6-27B}"', script)
        self.assertIn('MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"', script)
        self.assertIn('MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"', script)
        self.assertIn("--fixed-question-type-schedule", script)
        self.assertIn("--question-types neutral", script)
        self.assertIn("--generator-decode-mode sampling", script)
        self.assertIn('SAMPLING_TEMPERATURE="${SAMPLING_TEMPERATURE:-0.7}"', script)
        self.assertIn('SAMPLING_TOP_P="${SAMPLING_TOP_P:-0.9}"', script)
        self.assertIn('echo "cuda_keeper=required threshold=${CUDA_KEEPER_THRESHOLD}', script)
        self.assertIn('trap cleanup EXIT INT TERM', script)
        self.assertIn('echo "stage=verify_complete_slice_coverage"', script)
        self.assertNotIn("prompts-previous", script)

        heredocs = script.split("<<'PY'\n")[1:]
        self.assertEqual(len(heredocs), 4)
        for index, block in enumerate(heredocs):
            source = block.split("\nPY\n", 1)[0]
            compile(source, f"qa-slice-heredoc-{index}", "exec")

    def test_wrapper_packet_ranges_do_not_overlap(self) -> None:
        starts = []
        for path in sorted(self.hpc.glob("run_qa_packets_*_*.sbatch")):
            script = path.read_text(encoding="utf-8")
            match = re.search(r"^export PACKET_START_INDEX=(\d+)$", script, re.MULTILINE)
            if match:
                starts.append(int(match.group(1)))
        self.assertEqual(starts, [0, 100, 200])


if __name__ == "__main__":
    unittest.main()

