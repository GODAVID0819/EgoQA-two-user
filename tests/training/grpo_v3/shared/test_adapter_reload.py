from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from training.grpo_v3.shared.adapter_reload import discover_adapter_dir


class AdapterDiscoveryTests(unittest.TestCase):
    def test_finds_highest_complete_nested_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for step in (1, 3, 2):
                checkpoint = root / "swift" / f"checkpoint-{step}"
                checkpoint.mkdir(parents=True)
                (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
                (checkpoint / "adapter_model.safetensors").write_bytes(b"weights")
            self.assertEqual(discover_adapter_dir(root).name, "checkpoint-3")

    def test_rejects_incomplete_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "checkpoint-1").mkdir()
            (root / "checkpoint-1" / "adapter_config.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(FileNotFoundError, "adapter"):
                discover_adapter_dir(root)


if __name__ == "__main__":
    unittest.main()
