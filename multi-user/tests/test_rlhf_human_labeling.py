from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
import sys
import tarfile
import types
import unittest
import uuid
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if "egolife_two_user_qa" not in sys.modules:
    package = types.ModuleType("egolife_two_user_qa")
    package.__path__ = [str(ROOT)]
    sys.modules["egolife_two_user_qa"] = package

from egolife_two_user_qa.rlhf_human_labeling import (  # noqa: E402
    CHECKPOINT_SCHEMA_VERSION,
    LABEL_SCHEMA_VERSION,
    build_labeling_data,
    build_labeling_data_from_metadata_bundles,
    write_labeling_package,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RlhfHumanLabelingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = ROOT / "tmp" / f"rlhf_human_labeling_{uuid.uuid4().hex}"
        self.dataset = self.root / "dataset"
        self.checkpoint = self.root / "checkpoint"
        self.packet_id = "RLHF6U_DAY1_12000000_A1_A2_A3_A4_A5_A6"
        packet_dir = self.dataset / "packets" / self.packet_id
        users = []
        source_users = []
        cluster_users = []
        for user_index in range(6):
            agent_dir = f"A{user_index + 1}_USER{user_index + 1}"
            frames = []
            for frame_index in range(3):
                relative = Path("frames") / f"user_{user_index}" / f"frame_{frame_index}.jpg"
                frame_path = packet_dir / relative
                frame_path.parent.mkdir(parents=True, exist_ok=True)
                frame_path.write_bytes(b"synthetic-jpeg")
                frames.append(
                    {
                        "frame_index": frame_index,
                        "timestamp_seconds": float(frame_index * 2),
                        "source_segment_index": 0,
                        "path": relative.as_posix(),
                    }
                )
            users.append(
                {
                    "user_index": user_index,
                    "agent_dir": agent_dir,
                    "agent_id": f"A{user_index + 1}",
                    "agent_name": f"User {user_index + 1}",
                    "frame_count": 3,
                    "frames": frames,
                }
            )
            source_users.append(
                {
                    "agent_dir": agent_dir,
                    "segments": [
                        {
                            "clip_id": f"clip-{user_index}",
                            "time_token": "12000000",
                            "video_url": f"https://example.invalid/{agent_dir}.mp4",
                        }
                    ],
                }
            )
            cluster_users.append(
                {
                    "user_index": user_index,
                    "agent_dir": agent_dir,
                    "cluster_count": 2,
                    "labels": [0, 0, 1],
                    "representatives": [
                        {"cluster_index": 0, "frame_index": 0},
                        {"cluster_index": 1, "frame_index": 2},
                    ],
                }
            )
        write_json(
            packet_dir / "packet.json",
            {
                "schema_version": "egolife_rlhf_evidence_v1",
                "packet_id": self.packet_id,
                "day": "DAY1",
                "time_token": "12000000",
                "clip_clock": "12:00:00.00",
                "duration_seconds": 6,
                "source": {"users": source_users},
                "users": users,
            },
        )
        write_json(packet_dir / "clusters.json", {"users": cluster_users})
        masks = np.ones((6, 6, 3), dtype=np.bool_)
        for asker in range(6):
            for provider in range(6):
                if asker != provider:
                    masks[asker, provider, 0] = False
        np.savez_compressed(packet_dir / "keep_masks.npz", keep_masks=masks)
        (packet_dir / "COMPLETE").write_text("{}\n", encoding="utf-8")

        rows = []
        for index in range(2):
            rows.append(
                {
                    "qa_id": f"QA_{index + 1}",
                    "question": f"What happened in candidate {index + 1}?",
                    "options": ["one", "two", "three", "four", "five"],
                    "correct": "B",
                    "answer": "two",
                    "required_users": [f"User {user + 1}" for user in range(6)],
                    "referred_timestamps": [
                        {
                            "user": f"User {index + 1}",
                            "timestamp_seconds": 120100 if index == 0 else 9999,
                            "moment": "Synthetic evidence moment",
                        }
                    ],
                    "review": {"must_not_leak": True},
                    "generation_trace": {"must_not_leak": True},
                    "rlhf_source_packet_id": self.packet_id,
                    "rlhf_asker_index": index,
                    "rlhf_asker_user": f"User {index + 1}",
                }
            )
        queue = self.checkpoint / "labeling_queue.jsonl"
        write_jsonl(queue, rows)
        trajectory = self.checkpoint / "qa_mcq.intermediate.jsonl"
        write_jsonl(
            trajectory,
            [
                {
                    "evidence_id": f"{self.packet_id}__ASKER_A1",
                    "rlhf_source_packet_id": self.packet_id,
                    "rlhf_asker_index": 0,
                    "rlhf_asker_user": "User 1",
                    "attempts": [
                        {
                            "attempt": 1,
                            "generation": {"parsed_qa": rows[0]},
                            "judge": {"must_not_leak": True},
                            "result": {"accepted": False},
                        },
                        {
                            "attempt": 2,
                            "generation": {
                                "parsed_qa": {
                                    **rows[1],
                                    "options": ["one", "two", "three", "four"],
                                    "correct": "",
                                    "answer": "",
                                }
                            },
                            "judge": {"must_not_leak": True},
                            "result": {"accepted": True},
                        },
                        {
                            "attempt": 3,
                            "generation": {"parsed_qa": None},
                            "judge": {},
                            "result": {"accepted": False},
                        },
                    ],
                }
            ],
        )
        write_json(
            self.checkpoint / "CHECKPOINT_READY.json",
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "source_packet_ids": [self.packet_id],
                "labeling_file": "labeling_queue.jsonl",
                "trajectory_file": "qa_mcq.intermediate.jsonl",
                "payload_sha256": {
                    "labeling_queue.jsonl": sha256(queue),
                    "qa_mcq.intermediate.jsonl": sha256(trajectory),
                },
            },
        )
        self.metadata_bundle = self.root / "packets_000001_000030_metadata_only.tgz"
        media_rows = [
            {
                "checkpoint_name": "packets_000001_000030",
                "packet_id": self.packet_id,
                "day": "DAY1",
                "time_token": "12000000",
                "clip_clock": "12:00:00.00",
                "duration_seconds": 600,
                "users": [
                    {
                        "agent_dir": f"A{user_index + 1}_USER{user_index + 1}",
                        "agent_id": f"A{user_index + 1}",
                        "agent_name": f"User {user_index + 1}",
                        "segments": [
                            {
                                "segment_index": segment_index,
                                "start_seconds": segment_index * 30,
                                "end_seconds": (segment_index + 1) * 30,
                                "clip_id": f"clip-{user_index}-{segment_index}",
                                "time_token": f"12{segment_index:02d}0000",
                                "video_url": (
                                    "https://huggingface.co/datasets/example/resolve/main/"
                                    f"user-{user_index}/segment-{segment_index}.mp4"
                                ),
                            }
                            for segment_index in range(20)
                        ],
                    }
                    for user_index in range(6)
                ],
            }
        ]
        bundle_info = {
            "schema_version": "egolife_labeling_metadata_bundle_v1",
            "checkpoint_name": "packets_000001_000030",
        }
        with tarfile.open(self.metadata_bundle, "w:gz") as archive:
            archive.add(
                self.checkpoint,
                arcname="checkpoint/packets_000001_000030",
            )
            for name, payload in {
                "MEDIA_MANIFEST.jsonl": "".join(
                    json.dumps(row) + "\n" for row in media_rows
                ).encode(),
                "BUNDLE_INFO.json": (json.dumps(bundle_info) + "\n").encode(),
            }.items():
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
        (self.metadata_bundle.with_name(self.metadata_bundle.name + ".sha256")).write_text(
            f"{sha256(self.metadata_bundle)}  {self.metadata_bundle.name}\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_builds_variable_candidate_blinded_six_user_assignment(self) -> None:
        data = build_labeling_data(
            [self.checkpoint],
            dataset_root=self.dataset,
            assignment_id="review-a",
        )

        self.assertEqual(data["schema_version"], LABEL_SCHEMA_VERSION)
        self.assertEqual(data["summary"]["packet_count"], 1)
        self.assertEqual(data["summary"]["candidate_count"], 2)
        packet = data["packets"][0]
        self.assertEqual(len(packet["users"]), 6)
        self.assertEqual(len(packet["candidates"]), 2)
        for candidate in packet["candidates"]:
            self.assertEqual(len(candidate["keep_frame_indices_by_user"]), 6)
            self.assertEqual(
                len(candidate["keep_frame_indices_by_user"][candidate["asker_index"]]),
                3,
            )
        serialized = json.dumps(data)
        self.assertNotIn("must_not_leak", serialized)
        self.assertNotIn("generation_trace", serialized)
        self.assertNotIn('"review"', serialized)
        self.assertIn("/evidence/packets/", serialized)

    def test_rejects_mutated_checkpoint_payload(self) -> None:
        with (self.checkpoint / "labeling_queue.jsonl").open("a", encoding="utf-8") as handle:
            handle.write("{}\n")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            build_labeling_data(
                [self.checkpoint],
                dataset_root=self.dataset,
                assignment_id="review-a",
            )

    def test_builds_directly_from_metadata_only_bundle(self) -> None:
        data = build_labeling_data_from_metadata_bundles(
            [self.metadata_bundle],
            assignment_id="review-video",
        )

        self.assertEqual(data["summary"]["candidate_count"], 2)
        self.assertEqual(data["summary"]["generation_attempt_count"], 3)
        self.assertEqual(data["summary"]["unparseable_generation_attempt_count"], 1)
        self.assertEqual(len(data["unlabelable_generation_attempts"]), 1)
        self.assertIn("not parsed", data["unlabelable_generation_attempts"][0]["reason"])
        self.assertEqual(
            data["assignment"]["media_policy"],
            "full_ten_minute_huggingface_source_timelines",
        )
        packet = data["packets"][0]
        self.assertEqual(packet["duration_seconds"], 600)
        self.assertEqual(len(packet["users"]), 6)
        self.assertTrue(
            all(len(user["source_segments"]) == 20 for user in packet["users"])
        )
        serialized = json.dumps(data)
        self.assertNotIn("must_not_leak", serialized)
        self.assertNotIn('"review"', serialized)
        self.assertNotIn('"judge"', serialized)
        self.assertNotIn('"accepted"', serialized)
        self.assertNotIn("/evidence/packets/", serialized)
        self.assertTrue(
            all("::attempt_" in candidate["candidate_id"] for candidate in packet["candidates"])
        )
        malformed = next(
            candidate for candidate in packet["candidates"] if candidate["generation_attempt"] == 2
        )
        self.assertEqual(len(malformed["options"]), 4)
        self.assertEqual(malformed["correct"], "[Missing correct option]")
        first_attempt = next(
            candidate for candidate in packet["candidates"] if candidate["generation_attempt"] == 1
        )
        self.assertEqual(first_attempt["evidence_locations"][0]["user_index"], 0)
        self.assertEqual(first_attempt["evidence_locations"][0]["relative_seconds"], 60.0)
        self.assertEqual(first_attempt["evidence_locations"][0]["absolute_clock"], "12:01:00")

    def test_writes_runnable_package_and_binary_csv_template(self) -> None:
        data = build_labeling_data_from_metadata_bundles(
            [self.metadata_bundle],
            assignment_id="review-a",
        )
        output = self.root / "labeling"
        template = ROOT / "rlhf_human_labeling_template.html"
        write_labeling_package(
            data,
            output_dir=output,
            template_path=template,
        )

        html = (output / "rlhf_labeling.html").read_text(encoding="utf-8")
        self.assertNotIn("__LABELING_DATA__", html)
        self.assertNotIn("Blinded review.", html)
        self.assertNotIn("Specific failure reason", html)
        self.assertNotIn("Available evidence (one item per line)", html)
        self.assertIn("Previous packet", html)
        self.assertIn("candidate-tabs", html)
        self.assertIn("Common failure modes", html)
        self.assertIn("Manual notes for this question (optional)", html)
        self.assertNotIn("Attempt ${task.candidate.generation_attempt", html)
        self.assertIn("Unclear or unnatural wording / question", html)
        self.assertIn("Non-first-person question", html)
        self.assertIn("Asker asks about themselves", html)
        self.assertIn("Participant name leakage", html)
        self.assertIn("Timestamp leakage", html)
        self.assertNotIn("Evidence points to the wrong user or time", html)
        self.assertNotIn("Video unavailable or cannot be assessed", html)
        self.assertIn("Generator-reported evidence locations", html)
        self.assertIn("answerable: false", html)
        self.assertIn("Six-user source video", html)
        self.assertIn("Continue automatically", html)
        self.assertNotIn("<img", html)
        self.assertNotIn("/evidence/packets/", html)
        self.assertNotIn("Full judge view", html)
        self.assertIn(
            "Video segments load directly from Hugging Face",
            (output / "serve_rlhf_labeling.py").read_text(),
        )
        unlabelable = (output / "unlabelable_generation_attempts.jsonl").read_text(
            encoding="utf-8"
        )
        self.assertEqual(len([line for line in unlabelable.splitlines() if line]), 1)
        with (output / "human_labels_template.csv").open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        self.assertIn("formality_verdict", rows[0])
        self.assertIn("asker_only_answerable", rows[0])
        self.assertIn("failure_modes", rows[0])
        self.assertNotIn("asker_only_reason", rows[0])
        self.assertIn("generation_attempt", rows[0])
        self.assertEqual(rows[0]["annotation_status"], "pending")


if __name__ == "__main__":
    unittest.main()
