from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tarfile
import threading
import types
import unittest
import uuid
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if "egolife_two_user_qa" not in sys.modules:
    package = types.ModuleType("egolife_two_user_qa")
    package.__path__ = [str(ROOT)]
    sys.modules["egolife_two_user_qa"] = package

from egolife_two_user_qa.io_utils import (  # noqa: E402
    iter_jsonl,
    write_json,
    write_jsonl,
)
from egolife_two_user_qa.rlhf_evidence_preprocessing import (  # noqa: E402
    GENERATOR_MEDIA_MODE,
    SCHEMA_VERSION,
    build_preprocessing_config,
    load_asker_view,
)
from egolife_two_user_qa.rlhf_qa_generation import (  # noqa: E402
    run_rlhf_packet_generation,
)
from egolife_two_user_qa.video_qa_loop import (  # noqa: E402
    SIX_USER_JUDGE_MODE_LEGACY,
    media_for_clips,
    run_answerability_eval,
    six_user_role_metadata,
)


class _ConcurrentAnswerabilityRunner:
    def __init__(self) -> None:
        self.barrier = threading.Barrier(2, timeout=2.0)
        self.calls: list[int] = []
        self.lock = threading.Lock()

    def generate(self, prompt, image_paths=None, video_paths=None):
        with self.lock:
            self.calls.append(len(image_paths or []))
        self.barrier.wait()
        speaker_only = '"condition_type": "speaker_only"' in prompt
        return json.dumps(
            {
                "answerable": not speaker_only,
                "reason": "visible evidence was evaluated directly",
                "available_evidence": [],
                "missing_evidence": [],
            }
        )


class RlhfQaGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.test_root = ROOT / "tmp" / f"rlhf_qa_generation_{uuid.uuid4().hex}"
        self.dataset_root = self.test_root / "dataset"
        self.output_root = self.test_root / "output"
        self.packet_id = "RLHF6U_DAY1_12000000_A1_A2_A3_A4_A5_A6"
        packet_dir = self.dataset_root / "packets" / self.packet_id
        packet_dir.mkdir(parents=True)
        config = build_preprocessing_config(duration_seconds=6.0, sample_fps=0.5)
        users = []
        source_users = []
        for user_index in range(6):
            agent_dir = f"A{user_index + 1}_USER{user_index + 1}"
            frames = []
            frame_dir = packet_dir / "frames" / f"user_{user_index}"
            frame_dir.mkdir(parents=True)
            for frame_index in range(3):
                path = frame_dir / f"frame_{frame_index:04d}.jpg"
                path.write_bytes(b"synthetic-jpeg")
                frames.append(
                    {
                        "frame_index": frame_index,
                        "timestamp_seconds": float(frame_index * 2),
                        "source_segment_index": 0,
                        "path": path.relative_to(packet_dir).as_posix(),
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
                    "agent_id": f"A{user_index + 1}",
                    "agent_name": f"User {user_index + 1}",
                    "segments": [
                        {
                            "clip_id": f"clip-{user_index}",
                            "time_token": "12000000",
                            "video_url": f"https://example.invalid/{agent_dir}.mp4",
                        }
                    ],
                }
            )
        write_json(
            packet_dir / "packet.json",
            {
                "schema_version": SCHEMA_VERSION,
                "packet_id": self.packet_id,
                "day": "DAY1",
                "time_token": "12000000",
                "clip_clock": "12:00:00.00",
                "duration_seconds": 6.0,
                "selection": {"tier": "test"},
                "generator_media_mode": GENERATOR_MEDIA_MODE,
                "preprocessing": config,
                "source": {"users": source_users},
                "users": users,
            },
        )
        masks = np.ones((6, 6, 3), dtype=np.bool_)
        for asker_index in range(6):
            for provider_index in range(6):
                if provider_index != asker_index:
                    masks[asker_index, provider_index, 0] = False
        np.savez_compressed(packet_dir / "keep_masks.npz", keep_masks=masks)
        views = []
        for asker_index in range(6):
            media = []
            for user_index in range(6):
                is_asker = user_index == asker_index
                media.append(
                    {
                        "user_index": user_index,
                        "agent_dir": users[user_index]["agent_dir"],
                        "role": "asker" if is_asker else "provider",
                        "original_frame_count": 3,
                        "retained_frame_count": 3 if is_asker else 2,
                        "removed_frame_count": 0 if is_asker else 1,
                        "retained_percent": 100.0 if is_asker else 66.667,
                        "marked_cluster_count": 0 if is_asker else 1,
                        "restored_cluster_count": 0,
                    }
                )
            views.append(
                {
                    "asker_index": asker_index,
                    "asker_agent_dir": users[asker_index]["agent_dir"],
                    "generator_media_mode": GENERATOR_MEDIA_MODE,
                    "media": media,
                }
            )
        write_json(
            packet_dir / "asker_views.json",
            {"schema_version": SCHEMA_VERSION, "views": views},
        )
        (packet_dir / "COMPLETE").write_text("{}\n", encoding="utf-8")

    def tearDown(self) -> None:
        shutil.rmtree(self.test_root, ignore_errors=True)

    def test_media_contract_uses_pruned_generator_and_full_judge_frames(self) -> None:
        view = load_asker_view(self.dataset_root, self.packet_id, 2)

        six_user_role_metadata(view, view["required_users"])
        generator_images, generator_videos = media_for_clips(
            view["clips"],
            backend="openai-compatible-local",
            allow_openai_video_input=False,
            media_role="generator",
        )
        judge_images, judge_videos = media_for_clips(
            view["clips"],
            backend="openai-compatible-local",
            allow_openai_video_input=False,
            media_role="full",
        )

        self.assertEqual(len(generator_images), 3 + 5 * 2)
        self.assertEqual(len(judge_images), 6 * 3)
        self.assertEqual(generator_videos, [])
        self.assertEqual(judge_videos, [])

    def test_production_launcher_uses_one_h200_and_project_root_hpc(self) -> None:
        launcher = (
            ROOT
            / "hpc"
            / "qa"
            / "production"
            / "run_six_user_rlhf_qa_generation.sbatch"
        ).read_text(encoding="utf-8")

        self.assertIn("#SBATCH --gres=gpu:1", launcher)
        self.assertIn("#SBATCH --constraint=h200", launcher)
        self.assertNotIn("#SBATCH --gres=gpu:2", launcher)
        self.assertIn('REQUIRED_GPU_COUNT="${REQUIRED_GPU_COUNT:-1}"', launcher)
        self.assertIn('HPC_ROOT="${PROJECT_ROOT}/hpc"', launcher)
        self.assertIn(
            'EXPECTED_LAUNCHER="${HPC_ROOT}/qa/production/'
            'run_six_user_rlhf_qa_generation.sbatch"',
            launcher,
        )
        self.assertIn(
            'PACKAGE_ROOT="${PACKAGE_ROOT:-${PROJECT_ROOT}/'
            'egolife_two_user_qa/multi-user}"',
            launcher,
        )
        self.assertIn('export VLLM_TENSOR_PARALLEL_SIZE="${REQUIRED_GPU_COUNT}"', launcher)

    def test_legacy_six_user_answerability_calls_are_concurrent(self) -> None:
        view = load_asker_view(self.dataset_root, self.packet_id, 0)
        runner = _ConcurrentAnswerabilityRunner()
        prompt_rows: list[dict] = []
        qa = {
            "qa_id": "QA_TEST",
            "question": "What detail was visible?",
            "options": ["one", "two", "three", "four", "five"],
            "required_users": view["required_users"],
            "generation_mode": "baseline",
        }

        result = run_answerability_eval(
            qa_item=qa,
            packet=view,
            runner=runner,
            media_backend="openai-compatible-local",
            allow_openai_video_input=False,
            prompt_rows=prompt_rows,
            judge_media_role="full",
            attempt=1,
            six_user_judge_mode=SIX_USER_JUDGE_MODE_LEGACY,
        )

        self.assertTrue(result["gate"]["passed"])
        self.assertEqual(
            result["condition_execution"],
            "concurrent_asker_only_and_all_six",
        )
        self.assertEqual(sorted(runner.calls), [3, 18])
        self.assertEqual(len(prompt_rows), 2)

    def test_six_total_attempts_start_new_loops_after_early_acceptance(self) -> None:
        summary = run_rlhf_packet_generation(
            dataset_root=self.dataset_root,
            output_dir=self.output_root,
            packet_limit=1,
            max_packets_in_flight=1,
            max_generation_lanes=1,
            max_review_lanes=1,
            checkpoint_packet_count=1,
            dry_run=True,
        )

        self.assertEqual(summary["planned_generation_attempt_count"], 6)
        self.assertEqual(summary["completed_generation_attempt_count"], 6)
        self.assertEqual(summary["completed_question_loop_count"], 6)
        self.assertEqual(summary["accepted_question_count"], 6)
        markers = list(iter_jsonl(self.output_root / "loop_outcomes.jsonl"))
        self.assertEqual([row["attempts_consumed"] for row in markers], [1] * 6)
        self.assertEqual(sorted(row["asker_index"] for row in markers), list(range(6)))

        checkpoint_name = "packets_000001_000001"
        checkpoint_root = self.output_root / "checkpoints"
        checkpoint_dir = checkpoint_root / checkpoint_name
        ready = json.loads(
            (checkpoint_dir / "CHECKPOINT_READY.json").read_text(encoding="utf-8")
        )
        labeling_rows = list(iter_jsonl(checkpoint_dir / "labeling_queue.jsonl"))
        archive_path = checkpoint_root / f"{checkpoint_name}.tgz"
        checksum_path = checkpoint_root / f"{checkpoint_name}.tgz.sha256"
        expected_archive_sha256 = checksum_path.read_text(encoding="utf-8").split()[0]
        actual_archive_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        self.assertEqual(ready["source_packet_count"], 1)
        self.assertEqual(ready["completed_generation_attempt_count"], 6)
        self.assertEqual(len(labeling_rows), 6)
        self.assertEqual(expected_archive_sha256, actual_archive_sha256)
        with tarfile.open(archive_path, mode="r:gz") as archive:
            self.assertIn(
                f"{checkpoint_name}/labeling_queue.jsonl",
                archive.getnames(),
            )

        resumed = run_rlhf_packet_generation(
            dataset_root=self.dataset_root,
            output_dir=self.output_root,
            packet_limit=1,
            max_packets_in_flight=1,
            max_generation_lanes=1,
            max_review_lanes=1,
            checkpoint_packet_count=1,
            dry_run=True,
        )
        self.assertEqual(resumed["completed_generation_attempt_count"], 6)
        self.assertEqual(resumed["accepted_question_count"], 6)
        self.assertEqual(
            hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            expected_archive_sha256,
        )

    def test_two_three_pass_failures_exhaust_the_six_attempt_budget(self) -> None:
        def reject_entire_loop(**kwargs):
            attempt_count = int(kwargs["max_attempts"])
            write_jsonl(kwargs["output_path"], [])
            write_jsonl(
                kwargs["rejected_path"],
                [
                    {
                        "evidence_id": "synthetic",
                        "attempts": [
                            {"attempt": attempt, "reason": "judge rejected"}
                            for attempt in range(1, attempt_count + 1)
                        ],
                    }
                ],
            )
            write_jsonl(
                kwargs["intermediate_path"],
                [
                    {
                        "checkpoint_version": 3,
                        "evidence_id": "synthetic",
                        "status": "rejected",
                        "attempt_count": attempt_count,
                        "attempts": [
                            {"attempt": attempt, "result": {"accepted": False}}
                            for attempt in range(1, attempt_count + 1)
                        ],
                    }
                ],
            )
            write_jsonl(kwargs["prompts_path"], [])
            write_jsonl(kwargs["infrastructure_skipped_path"], [])
            return []

        with mock.patch(
            "egolife_two_user_qa.rlhf_qa_generation.generate_video_qa_loop",
            side_effect=reject_entire_loop,
        ):
            summary = run_rlhf_packet_generation(
                dataset_root=self.dataset_root,
                output_dir=self.output_root,
                packet_limit=1,
                max_packets_in_flight=1,
                max_generation_lanes=1,
                max_review_lanes=1,
            )

        self.assertEqual(summary["completed_generation_attempt_count"], 6)
        self.assertEqual(summary["completed_question_loop_count"], 2)
        self.assertEqual(summary["accepted_question_count"], 0)
        self.assertEqual(summary["exhausted_rejected_loop_count"], 2)
        markers = list(iter_jsonl(self.output_root / "loop_outcomes.jsonl"))
        self.assertEqual([row["attempts_consumed"] for row in markers], [3, 3])
        self.assertEqual(len({row["asker_index"] for row in markers}), 2)


if __name__ == "__main__":
    unittest.main()
