from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from training.grpo_v3_gate3_dataset import build_gate3_split


class Gate3DatasetTests(unittest.TestCase):
    def test_builds_reproducible_disjoint_balanced_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packets = []
            for index in range(32):
                clips = []
                for user in ("u1", "u2"):
                    video = root / f"E{index:03d}_{user}.mp4"
                    video.write_bytes(b"video")
                    clips.append({"agent_name": user, "local_video": str(video)})
                packets.append(
                    {
                        "evidence_id": f"E{index:03d}",
                        "required_users": ["u1", "u2"],
                        "clips": clips,
                    }
                )

            prompt_builder = lambda packet, question_type, generation_mode: (
                f"{packet['evidence_id']} {question_type} {generation_mode}"
            )
            train_a, eval_a, manifest_a = build_gate3_split(
                packets,
                seed=42,
                train_count=20,
                eval_count=8,
                prompt_builder=prompt_builder,
            )
            train_b, eval_b, manifest_b = build_gate3_split(
                list(reversed(packets)),
                seed=42,
                train_count=20,
                eval_count=8,
                prompt_builder=prompt_builder,
            )

            self.assertEqual(manifest_a["train_evidence_ids"], manifest_b["train_evidence_ids"])
            self.assertEqual(manifest_a["eval_evidence_ids"], manifest_b["eval_evidence_ids"])
            self.assertEqual(len(train_a), 20)
            self.assertEqual(len(eval_a), 8)
            self.assertEqual(len({row["evidence_id"] for row in train_a}), 20)
            self.assertTrue(
                {row["evidence_id"] for row in train_a}.isdisjoint(
                    {row["evidence_id"] for row in eval_a}
                )
            )
            self.assertEqual([row["question_type"] for row in train_a[::2]], ["commonality"] * 10)
            self.assertEqual([row["question_type"] for row in train_a[1::2]], ["difference"] * 10)
            self.assertEqual(manifest_a["train_question_type_counts"], {"commonality": 10, "difference": 10})
            self.assertEqual(manifest_a["eval_question_type_counts"], {"commonality": 4, "difference": 4})

    def test_requires_enough_unique_evidence(self) -> None:
        packets = [{"evidence_id": f"E{i}"} for i in range(27)]
        with self.assertRaisesRegex(ValueError, "至少需要 28"):
            build_gate3_split(packets, train_count=20, eval_count=8)


if __name__ == "__main__":
    unittest.main()
