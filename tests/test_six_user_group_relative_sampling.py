from __future__ import annotations

import random
import shutil
import sys
import types
import unittest
import uuid
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if "egolife_two_user_qa" not in sys.modules:
    package = types.ModuleType("egolife_two_user_qa")
    package.__path__ = [str(ROOT)]
    sys.modules["egolife_two_user_qa"] = package

from egolife_two_user_qa.group_relative_clip_sampling import (  # noqa: E402
    build_six_user_role_structures,
    materialize_six_user_role_structure,
)
from egolife_two_user_qa import group_relative_clip_sampling  # noqa: E402


def pair_scores(*, kept_keys: set[str]) -> list[dict[str, object]]:
    rows = []
    for left_index in range(6):
        for right_index in range(left_index + 1, 6):
            pair_key = f"{left_index}-{right_index}"
            kept = pair_key in kept_keys
            rows.append(
                {
                    "pair_key": pair_key,
                    "left_index": left_index,
                    "right_index": right_index,
                    "status": "kept" if kept else "rejected",
                    "rejection_reasons": [] if kept else ["synthetic_rejection"],
                }
            )
    return rows


class SixUserRoleSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp_root = ROOT / "tmp"
        tmp_root.mkdir(exist_ok=True)
        self.tmp_path = tmp_root / f"six_user_sampling_{uuid.uuid4().hex}"
        self.tmp_path.mkdir()
        self.addCleanup(shutil.rmtree, self.tmp_path, True)

    def six_rows(self) -> list[dict[str, object]]:
        rows = []
        for index in range(6):
            source = self.tmp_path / f"user_{index}.mp4"
            source.write_bytes(b"full-video")
            rows.append(
                {
                    "user": f"user_{index}",
                    "clip": {
                        "clip_id": f"clip_{index}",
                        "agent_dir": f"agent_{index}",
                        "agent_name": f"user_{index}",
                        "local_video": str(source),
                    },
                    "frames": [],
                }
            )
        return rows

    @staticmethod
    def anchor_edge(anchor_index: int, *, speaker_remove: list[list[float]]) -> dict[str, object]:
        return {
            "pair_key": f"0-{anchor_index}",
            "left_index": 0,
            "right_index": anchor_index,
            "status": "kept",
            "temporal_pruning": {
                "method": "synthetic",
                "left_remove_intervals": speaker_remove,
                "left_keep_intervals": [[0.0, 10.0]],
                "left_kept_duration_seconds": 10.0,
                "right_remove_intervals": [[1.0, 3.0]],
                "right_keep_intervals": [[0.0, 1.0], [3.0, 10.0]],
                "right_kept_duration_seconds": 8.0,
            },
        }

    def test_exactly_two_speaker_edges_are_enough(self) -> None:
        scores = pair_scores(kept_keys={"0-1", "0-2"})

        result = build_six_user_role_structures(scores, rng=random.Random(7))

        self.assertEqual(len(result["diagnostic_pair_edges"]), 15)
        self.assertEqual(result["kept_degrees"], [2, 1, 1, 0, 0, 0])
        self.assertEqual(len(result["role_structures"]), 1)
        structure = result["role_structures"][0]
        self.assertEqual(structure["speaker_index"], 0)
        self.assertEqual(structure["anchor_indices"], [1, 2])
        self.assertEqual(structure["additional_indices"], [3, 4, 5])
        self.assertEqual(
            [edge["pair_key"] for edge in structure["selected_anchor_edges"]],
            ["0-1", "0-2"],
        )

    def test_one_kept_neighbor_produces_no_role_structure(self) -> None:
        result = build_six_user_role_structures(
            pair_scores(kept_keys={"0-1"}),
            rng=random.Random(2),
        )

        self.assertEqual(result["role_structures"], [])
        self.assertEqual(result["kept_degrees"], [1, 1, 0, 0, 0, 0])
        self.assertEqual(result["eligible_speaker_indices"], [])

    def test_provider_provider_rejections_do_not_block_valid_star(self) -> None:
        result = build_six_user_role_structures(
            pair_scores(kept_keys={"3-4", "3-5"}),
            rng=random.Random(5),
        )

        self.assertEqual(len(result["role_structures"]), 1)
        self.assertEqual(result["role_structures"][0]["speaker_index"], 3)
        self.assertEqual(result["role_structures"][0]["anchor_indices"], [4, 5])
        self.assertEqual(result["role_structures"][0]["additional_indices"], [0, 1, 2])

    def test_seeded_order_is_deterministic_with_multiple_structures(self) -> None:
        scores = pair_scores(kept_keys={"0-1", "0-2", "0-3", "1-2", "1-4"})

        first = build_six_user_role_structures(scores, rng=random.Random(19))
        second = build_six_user_role_structures(scores, rng=random.Random(19))

        self.assertGreater(len(first["role_structures"]), 1)
        self.assertEqual(first["role_structures"], second["role_structures"])
        for structure in first["role_structures"]:
            self.assertEqual(len(structure["anchor_indices"]), 2)
            self.assertEqual(len(structure["additional_indices"]), 3)
            self.assertEqual(len(structure["selected_anchor_edges"]), 2)

    def test_missing_pair_edge_is_rejected(self) -> None:
        scores = pair_scores(kept_keys={"0-1", "0-2"})[:-1]

        with self.assertRaisesRegex(ValueError, "15"):
            build_six_user_role_structures(scores, rng=random.Random(1))

    def test_materializes_three_pruned_and_three_full_videos_in_role_order(self) -> None:
        first_edge = self.anchor_edge(1, speaker_remove=[[2.0, 4.0]])
        second_edge = self.anchor_edge(2, speaker_remove=[[3.0, 5.0]])
        structure = {
            "candidate_rank": 1,
            "speaker_index": 0,
            "anchor_indices": [1, 2],
            "additional_indices": [3, 4, 5],
            "selected_anchor_edges": [first_edge, second_edge],
        }
        materialize_calls = []

        def fake_materialize(source_video, output_video, keep_intervals, **kwargs):
            materialize_calls.append(
                {
                    "source": str(source_video),
                    "output": str(output_video),
                    "keep_intervals": list(keep_intervals),
                }
            )
            output = Path(output_video)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"pruned-video")
            return output

        with mock.patch.object(
            group_relative_clip_sampling,
            "materialize_pruned_video",
            fake_materialize,
        ):
            clips = materialize_six_user_role_structure(
                self.six_rows(),
                structure,
                output_dir=self.tmp_path / "assets",
                start_seconds=0.0,
                duration_seconds=10.0,
                min_pruned_video_seconds=2.0,
                ffmpeg_binary="ffmpeg",
            )

        self.assertEqual([clip["agent_name"] for clip in clips], [f"user_{i}" for i in range(6)])
        self.assertEqual(
            [clip["media_role"] for clip in clips],
            [
                "speaker_pruned",
                "anchor_provider_pruned",
                "anchor_provider_pruned",
                "additional_provider_full",
                "additional_provider_full",
                "additional_provider_full",
            ],
        )
        self.assertEqual(len(materialize_calls), 3)
        self.assertEqual(
            materialize_calls[0]["keep_intervals"],
            [(0.0, 2.0), (5.0, 10.0)],
        )
        self.assertEqual(
            [row["keep_intervals"] for row in materialize_calls[1:]],
            [[[0.0, 1.0], [3.0, 10.0]], [[0.0, 1.0], [3.0, 10.0]]],
        )
        for clip in clips[:3]:
            self.assertTrue(clip["is_pruned"])
            self.assertNotEqual(clip["generator_local_video"], clip["full_local_video"])
        for clip in clips[3:]:
            self.assertFalse(clip["is_pruned"])
            self.assertEqual(clip["generator_local_video"], clip["full_local_video"])

    def test_rejects_role_structure_when_merged_speaker_retention_is_too_short(self) -> None:
        first_edge = self.anchor_edge(1, speaker_remove=[[0.0, 6.0]])
        second_edge = self.anchor_edge(2, speaker_remove=[[5.0, 9.5]])
        structure = {
            "candidate_rank": 1,
            "speaker_index": 0,
            "anchor_indices": [1, 2],
            "additional_indices": [3, 4, 5],
            "selected_anchor_edges": [first_edge, second_edge],
        }

        with self.assertRaisesRegex(ValueError, "speaker.*too short"):
            materialize_six_user_role_structure(
                self.six_rows(),
                structure,
                output_dir=self.tmp_path / "assets",
                start_seconds=0.0,
                duration_seconds=10.0,
                min_pruned_video_seconds=2.0,
                ffmpeg_binary="ffmpeg",
            )


if __name__ == "__main__":
    unittest.main()
