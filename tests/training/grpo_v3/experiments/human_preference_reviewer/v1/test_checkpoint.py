from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is verified in the Torch training environment")
class CheckpointTests(unittest.TestCase):
    def test_heads_round_trip_exactly(self) -> None:
        import torch
        from torch import nn
        from training.grpo_v3.experiments.human_preference_reviewer.v1.checkpoint import (
            load_classification_heads,
            save_checkpoint,
        )

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.evidence_head = nn.Linear(4, 3)
                self.answerability_head = nn.Linear(4, 3)
                self.formality_head = nn.Linear(4, 3)
                self.adapter = nn.Parameter(torch.ones(1))

        model = Model()
        directory = Path(tempfile.mkdtemp())
        save_checkpoint(
            model, directory,
            config={"contract_version": "human_preference_reviewer_absolute_v1"},
            csv_sha256="CSV", split_sha256="SPLIT", parameter_audit={"status": "passed"},
        )
        restored = Model()
        load_classification_heads(restored, directory)
        for name in ("evidence_head", "answerability_head", "formality_head"):
            for key, value in getattr(model, name).state_dict().items():
                torch.testing.assert_close(value, getattr(restored, name).state_dict()[key])


if __name__ == "__main__":
    unittest.main()
