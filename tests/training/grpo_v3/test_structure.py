from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


class GrpoV3StructureTests(unittest.TestCase):
    def test_shared_package_replaces_flat_shared_modules(self) -> None:
        expected = (
            ROOT / "training/grpo_v3/shared/adapter_reload.py",
            ROOT / "training/grpo_v3/shared/contract.py",
            ROOT / "training/grpo_v3/shared/data.py",
            ROOT / "training/grpo_v3/shared/json_format.py",
        )
        for path in expected:
            with self.subTest(path=path):
                self.assertTrue(path.is_file())
        for name in (
            "grpo_v3_adapter_reload.py",
            "grpo_v3_contract.py",
            "grpo_v3_data.py",
            "grpo_v3_json_format.py",
        ):
            with self.subTest(old=name):
                self.assertFalse((ROOT / "training" / name).exists())

if __name__ == "__main__":
    unittest.main()
