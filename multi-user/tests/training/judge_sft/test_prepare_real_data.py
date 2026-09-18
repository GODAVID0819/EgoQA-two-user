from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from training.judge_sft.prepare_real_data import (
    FULL_JUDGE_MEDIA_MODE,
    _load_full_judge_view,
    _repo_module,
    _validate_source_video_provenance,
    _validated_label_row,
    build_parser,
    load_generation,
)


ROOT = Path(__file__).resolve().parents[3]


def _label_row() -> dict:
    return {
        "schema_version": "egolife_six_user_binary_labeling_v2",
        "annotation_status": "completed",
        "candidate_skipped": False,
        "formality_verdict": "pass",
        "evidence_grounding_verdict": "fail",
        "answerability_verdict": "pass",
        "asker_only_answerable": False,
        "all_six_answerable": True,
        "options": json.dumps(["a", "b", "c", "d", "e"]),
        "required_users": json.dumps([f"u{i}" for i in range(6)]),
        "source_video_urls": json.dumps([f"v{i}" for i in range(120)]),
        "labels": {
            "qa_formality": {"verdict": "pass"},
            "evidence_groundedness": {"verdict": "fail"},
            "answerability": {
                "verdict": "pass",
                "speaker_only": {"answerable": False},
                "combined_all_six_users": {"answerable": True},
            },
        },
    }


class PrepareRealDataTests(unittest.TestCase):
    def test_source_video_provenance_allows_only_reordering(self) -> None:
        _validate_source_video_provenance(
            ["video_b", "video_a", "video_a"],
            ["video_a", "video_b", "video_a"],
            candidate_id="candidate",
        )
        with self.assertRaisesRegex(ValueError, "source video URL set mismatch"):
            _validate_source_video_provenance(
                ["video_a", "video_b"],
                ["video_a", "video_c"],
                candidate_id="candidate",
            )

    def test_repo_module_resolves_this_multi_user_checkout(self) -> None:
        prompts = _repo_module("prompts")
        self.assertEqual(Path(prompts.__file__).resolve(), ROOT / "prompts.py")

    def test_full_judge_view_needs_no_legacy_preprocessing_module(self) -> None:
        users = []
        source_users = []
        for index in range(6):
            users.append(
                {
                    "user_index": index,
                    "agent_dir": f"agent_{index}",
                    "agent_id": f"A{index}",
                    "agent_name": f"user_{index}",
                    "frames": [
                        {
                            "frame_index": 0,
                            "path": f"frames/user_{index}/000000.jpg",
                        }
                    ],
                }
            )
            source_users.append(
                {
                    "agent_dir": f"agent_{index}",
                    "segments": [{"video_url": f"video_{index}.mp4"}],
                }
            )
        packet = {
            "schema_version": "egolife_rlhf_evidence_v1",
            "packet_id": "P1",
            "duration_seconds": 600,
            "preprocessing": {"sampling": {"fps": 0.5}},
            "source": {"users": source_users},
            "users": users,
        }
        with patch(
            "training.judge_sft.prepare_real_data._read_json",
            return_value=packet,
        ):
            view = _load_full_judge_view(ROOT / "synthetic_dataset", "P1", 2)

        self.assertEqual([clip["user_index"] for clip in view["clips"]], [2, 0, 1, 3, 4, 5])
        self.assertEqual(view["required_users"][0], "user_2")
        self.assertEqual(view["generator_media_mode"], FULL_JUDGE_MEDIA_MODE)
        self.assertTrue(all(not clip["is_pruned"] for clip in view["clips"]))
        self.assertTrue(all(clip["frames"] == clip["full_frames"] for clip in view["clips"]))
        self.assertEqual(
            view["generator_context_budget"]["per_user_model_input_frame_counts"],
            [1, 1, 1, 1, 1, 1],
        )

    def test_answerability_aggregate_is_consistency_only_gate(self) -> None:
        row = _validated_label_row(_label_row(), location="test")
        self.assertFalse(row["asker_only_answerable"])
        self.assertTrue(row["all_six_answerable"])
        broken = _label_row()
        broken["answerability_verdict"] = "fail"
        broken["labels"]["answerability"]["verdict"] = "fail"
        with self.assertRaisesRegex(ValueError, "gate is inconsistent"):
            _validated_label_row(broken, location="test")

    def test_generation_join_recovers_attempt_and_schema_errors(self) -> None:
        root = ROOT / "tests" / "training" / "judge_sft" / "fixtures" / "generation_join"
        attempts, packets, _ = load_generation(root)
        item = attempts["P1__ASKER_A3::attempt_03"]
        self.assertEqual(item.qa["qa_id"], "Q1")
        self.assertEqual(item.schema_errors, ("six options",))
        self.assertEqual(packets, {"P1"})

    def test_preparation_has_no_internal_validation_split(self) -> None:
        parsed = build_parser().parse_args(
            [
                "--labels",
                "labels.jsonl",
                "--generation-root",
                "generation",
                "--dataset-root",
                "dataset",
                "--output-dir",
                "out",
            ]
        )
        self.assertFalse(hasattr(parsed, "eval_fraction"))
        self.assertFalse(hasattr(parsed, "seed"))


if __name__ == "__main__":
    unittest.main()
