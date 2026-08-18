from __future__ import annotations

import random
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if "egolife_two_user_qa" not in sys.modules:
    package = types.ModuleType("egolife_two_user_qa")
    package.__path__ = [str(ROOT)]
    sys.modules["egolife_two_user_qa"] = package

from egolife_two_user_qa.group_relative_clip_sampling import (  # noqa: E402
    build_six_user_role_structures,
)


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


if __name__ == "__main__":
    unittest.main()
