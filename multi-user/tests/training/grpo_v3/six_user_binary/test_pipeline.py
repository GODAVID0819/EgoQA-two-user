"""六用户 GRPO 的媒体路由与线上评审边界。"""
from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


MODULE = "training.grpo_v3.six_user_binary"


def require_module(test, name):
    try:
        spec = importlib.util.find_spec(MODULE + "." + name)
    except ModuleNotFoundError:
        spec = None
    test.assertIsNotNone(spec, "六用户 GRPO 模块尚未实现：" + name)
    return importlib.import_module(MODULE + "." + name)


def packet_fixture(root: Path, packet_id="DAY1_TEST"):
    import numpy as np
    packet = root / "packets" / packet_id
    packet.mkdir(parents=True)
    users = []
    for u in range(6):
        frame_dir = packet / f"u{u}"
        frame_dir.mkdir()
        frames = []
        for i in range(300):
            path = frame_dir / f"{i:03d}.jpg"
            path.write_bytes(b"fixture-frame")
            frames.append({"frame_index": i, "timestamp_seconds": i * 2.0,
                           "path": path.relative_to(packet).as_posix()})
        users.append({"user_index": u, "agent_dir": f"A{u+1}",
                      "agent_id": f"A{u+1}", "agent_name": f"Person{u+1}", "frames": frames})
    (packet / "packet.json").write_text(json.dumps({
        "schema_version": "egolife_rlhf_evidence_v1", "packet_id": packet_id,
        "users": users, "duration_seconds": 600.0, "day": "DAY1",
        "clip_clock": "12:00:00", "time_token": "12000000",
        "preprocessing": {"sampling": {"fps": 0.5, "interval_seconds": 2.0}},
    }), encoding="utf-8")
    masks = np.ones((6, 6, 300), dtype=bool)
    for asker in range(6):
        for user in range(6):
            if user != asker:
                masks[asker, user, 1::2] = False
    np.savez(packet / "keep_masks.npz", keep_masks=masks)
    (packet / "asker_views.json").write_text('{"views":[]}', encoding="utf-8")
    return packet


def qa_text():
    return json.dumps({"question_type": "neutral", "question": "Where did I leave my mug?",
        "options": ["On the table", "In the drawer", "On the shelf", "In the sink", "In the bag"],
        "correct": "A", "answer": "On the table", "generator_rationale": "PRIVATE_RATIONALE",
        "why_two_users_needed": "PRIVATE_NEED", "evidence": [], "referred_timestamps": []})


class SixUserDataTests(unittest.TestCase):
    def test_duplicate_validation_paths_do_not_hide_split_leakage(self):
        data = require_module(self, "data")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.jsonl"
            path.write_text('{"source_packet_id":"P","asker_index":0}\n')
            with patch("sys.argv", ["data", "--validate", str(path), str(path)]), patch.object(data, "validate_row"):
                with self.assertRaisesRegex(ValueError, "重复"):
                    data.main()
    def test_generation_uses_pruned_providers_with_full_speaker(self):
        data = require_module(self, "data")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packet_fixture(root)
            row = data.make_row(root, "DAY1_TEST", 2)
            self.assertEqual(row["required_users"][0], "Person3")
            self.assertEqual(row["generator_frame_counts"], [300, 150, 150, 150, 150, 150])
            self.assertEqual(len(row["images"]), 1050)
            self.assertEqual(row["messages"][0]["content"].count("<image>"), 1050)
            self.assertNotIn("videos", row)
            data.validate_row(row)
            row["generator_image_paths"] = list(reversed(row["generator_image_paths"]))
            with self.assertRaisesRegex(ValueError, "generator_image_paths"):
                data.validate_row(row)

    def test_same_source_packet_cannot_cross_splits_with_different_askers(self):
        data = require_module(self, "data")
        rows = {"train": [{"source_packet_id": "P", "asker_index": 0}],
                "validation": [{"source_packet_id": "P", "asker_index": 1}]}
        with self.assertRaisesRegex(ValueError, "P"):
            data.validate_splits(rows)


class JudgeRoutingTests(unittest.TestCase):
    def test_judges_use_current_prompts_and_full_unpruned_media(self):
        judge = require_module(self, "judge")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packet_fixture(root)
            request = {"request_id": "case1", "dataset_root": str(root),
                       "source_packet_id": "DAY1_TEST", "asker_index": 2,
                       "evidence_id": "DAY1_TEST__ASKER_A3", "completion": qa_text()}
            examples = judge.build_examples(request)
            self.assertEqual([e.frame_count for e in examples.values()], [0, 1800, 300, 1800])
            for key in ("speaker_only", "all_six"):
                self.assertNotIn("On the table", examples[key].prompt)
                self.assertNotIn("PRIVATE_RATIONALE", examples[key].prompt)
            self.assertEqual(examples["speaker_only"].frame_sets[0].label, examples["all_six"].frame_sets[0].label)
            self.assertIn("evidence-sufficiency judge", examples["all_six"].prompt)
            request["evidence_id"] = "OTHER_PACKET"
            with self.assertRaisesRegex(ValueError, "evidence_id"):
                judge.build_examples(request)

    def test_missing_media_is_not_a_zero_reward_candidate(self):
        judge = require_module(self, "judge")
        with self.assertRaises(FileNotFoundError):
            judge.build_examples({"request_id": "x", "dataset_root": "missing_grpo_fixture",
                "source_packet_id": "P", "asker_index": 0, "evidence_id": "P",
                "completion": qa_text()})


if __name__ == "__main__":
    unittest.main()
