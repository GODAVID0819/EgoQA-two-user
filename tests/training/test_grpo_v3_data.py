from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.grpo_v3_data import packet_to_swift_row, validate_swift_row


class NativeDualVideoDataTests(unittest.TestCase):
    def _packet(self, root: Path) -> dict:
        videos = []
        for name in ("asker.mp4", "provider.mp4"):
            path = root / name
            path.write_bytes(b"not-empty-mp4-probe")
            videos.append(str(path))
        return {
            "evidence_id": "E1",
            "required_users": ["ALICE", "BOB"],
            "clips": [
                {"agent_name": "ALICE", "local_video": videos[0]},
                {"agent_name": "BOB", "local_video": videos[1]},
            ],
        }

    def test_builds_exactly_two_ordered_native_videos(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            packet = self._packet(Path(tmp))
            row = packet_to_swift_row(
                packet,
                question_type="commonality",
                generation_mode="baseline",
                prompt_builder=lambda *_args, **_kwargs: "PROMPT",
            )
        self.assertEqual(row["messages"], [{"role": "user", "content": "<video><video>\nPROMPT"}])
        self.assertEqual(row["video_order"], ["ALICE", "BOB"])
        self.assertEqual(row["required_users"], ["ALICE", "BOB"])
        self.assertEqual(row["evidence_id"], "E1")
        self.assertEqual(row["question_type"], "commonality")
        self.assertEqual(row["generation_mode"], "baseline")
        self.assertEqual(json.loads(row["packet_json"])["evidence_id"], "E1")
        self.assertEqual(len(row["videos"]), 2)
        validate_swift_row(row, require_files=False)

    def test_rejects_missing_empty_or_frame_only_media(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packet = self._packet(root)
            Path(packet["clips"][1]["local_video"]).write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "为空"):
                packet_to_swift_row(packet, question_type="commonality", prompt_builder=lambda *_a, **_k: "P")

            packet = self._packet(root)
            packet["clips"][0]["generator_media_mode"] = "frames_only"
            packet["clips"][0]["frames"] = [{"path": "frame.jpg"}]
            with self.assertRaisesRegex(ValueError, "sampled_frames|frames_only"):
                packet_to_swift_row(packet, question_type="commonality", prompt_builder=lambda *_a, **_k: "P")

    def test_rejects_wrong_user_or_video_cardinality(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            packet = self._packet(Path(tmp))
            packet["required_users"] = ["ALICE"]
            with self.assertRaisesRegex(ValueError, "恰好两个"):
                packet_to_swift_row(packet, question_type="commonality", prompt_builder=lambda *_a, **_k: "P")

    def test_allows_historical_frame_provenance_when_formal_media_is_mp4(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            packet = self._packet(Path(tmp))
            packet["sampled_frames"] = [{"path": "historical-only.jpg"}]
            row = packet_to_swift_row(
                packet,
                question_type="commonality",
                prompt_builder=lambda *_args, **_kwargs: "PROMPT",
            )
            self.assertEqual(len(row["videos"]), 2)
            self.assertTrue(all(path.endswith(".mp4") for path in row["videos"]))

            packet = self._packet(Path(tmp))
            packet["clips"] = packet["clips"][:1]
            with self.assertRaisesRegex(ValueError, "视频"):
                packet_to_swift_row(packet, question_type="commonality", prompt_builder=lambda *_a, **_k: "P")


if __name__ == "__main__":
    unittest.main()
