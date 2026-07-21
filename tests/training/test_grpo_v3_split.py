from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

from training.grpo_v3_split import split_packets


class Gate4SplitTests(unittest.TestCase):
    def _packets(self, root: Path, count: int = 50) -> list[dict]:
        videos = []
        for name in ("alice.mp4", "bob.mp4"):
            path = root / name
            path.write_bytes(b"mp4")
            videos.append(str(path))
        return [
            {
                "evidence_id": f"E{index:03d}",
                "required_users": ["ALICE", "BOB"],
                "clips": [
                    {"agent_name": "ALICE", "local_video": videos[0]},
                    {"agent_name": "BOB", "local_video": videos[1]},
                ],
            }
            for index in range(count)
        ]

    def test_split_is_deterministic_disjoint_and_order_independent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            packets = self._packets(Path(tmp))
            shuffled = list(packets)
            random.Random(7).shuffle(shuffled)
            first = split_packets(packets, seed=42, prompt_builder=lambda *_a, **_k: "P")
            second = split_packets(shuffled, seed=42, prompt_builder=lambda *_a, **_k: "P")
        self.assertEqual(first["manifest"], second["manifest"])
        self.assertEqual(len(first["train_rows"]), 40)
        self.assertEqual(len(first["eval_rows"]), 10)
        train_ids = set(first["manifest"]["train_evidence_ids"])
        eval_ids = set(first["manifest"]["eval_evidence_ids"])
        self.assertFalse(train_ids & eval_ids)
        self.assertEqual(first["manifest"]["intersection_count"], 0)
        for row in first["train_rows"] + first["eval_rows"]:
            self.assertEqual(len(row["videos"]), 2)
            self.assertEqual(row["messages"][0]["content"].count("<video>"), 2)

    def test_rejects_duplicate_or_insufficient_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            packets = self._packets(Path(tmp))
            with self.assertRaisesRegex(ValueError, "duplicate|重复"):
                split_packets(packets + [packets[0]], prompt_builder=lambda *_a, **_k: "P")
            with self.assertRaisesRegex(ValueError, "50"):
                split_packets(packets[:49], prompt_builder=lambda *_a, **_k: "P")


if __name__ == "__main__":
    unittest.main()
