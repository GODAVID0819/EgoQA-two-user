from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from training.grpo_v3.baseline.gate3_dataset import (
    build_gate3_split,
    build_named_split,
    rebase_active_media_paths,
)
from training.grpo_v3.baseline.prepare_imported_reviewer_pairs import (
    IMPORTED_SCHEMA_VERSION,
    prepare_imported_pairs,
)
from training.grpo_v3.experiments.human_preference_reviewer.v1.config import (
    ReviewerV1Config,
)
from training.grpo_v3.experiments.human_preference_reviewer.v1.deployment import (
    reviewer_config_from_contract,
)
from training.grpo_v3.experiments.human_preference_reviewer.v1.grpo_reward import (
    prepare_completion_for_review,
)
from training.grpo_v3.experiments.human_preference_reviewer.v1.prompting import (
    encode_candidate,
)
from training.grpo_v3.runtime.score_only_reward_plugin import _expand_path_list
from training.grpo_v3.shared.data import (
    PRODUCTION_PROMPT_CONTRACT,
    PRODUCTION_PROMPT_REQUIRED_FRAGMENTS,
    packet_to_swift_row,
    production_prompt_source_contract,
    validate_swift_row,
    write_jsonl,
)
from training.grpo_v3.shared.media import GENERATOR_MEDIA_MODE
from training.grpo_v3.shared.validate_dataset import validate_dataset

def _completion() -> str:
    return json.dumps(
        {
            "question": "Which description is supported by the combined views?",
            "options": [
                "The first possible description.",
                "The second possible description.",
                "The third possible description.",
                "The fourth possible description.",
                "The fifth possible description.",
            ],
            "correct": "A",
            "answer": "The first possible description.",
        }
    )


class ReviewerEncodingContractTests(unittest.TestCase):
    def test_reviewer_encoder_uses_qwen3vl_video_metadata_contract(self) -> None:
        class FakeImageProcessor:
            patch_size = 16

        class FakeProcessor:
            image_processor = FakeImageProcessor()

            def __init__(self) -> None:
                self.call: dict | None = None

            def apply_chat_template(self, messages: list[dict], **kwargs: object) -> str:
                return "rendered"

            def __call__(self, **kwargs: object) -> dict:
                self.call = dict(kwargs)
                return {"input_ids": [[1]], "attention_mask": [[1]]}

        class FakeVideo:
            shape = (4, 3, 32, 32)

        def fake_process_vision_info(messages: list[dict], **kwargs: object) -> tuple:
            self.assertEqual(kwargs["image_patch_size"], 16)
            self.assertIs(kwargs["return_video_kwargs"], True)
            self.assertIs(kwargs["return_video_metadata"], True)
            return (
                None,
                [(FakeVideo(), {"fps": 30.0}), (FakeVideo(), {"fps": 30.0})],
                {"do_sample_frames": False},
            )

        processor = FakeProcessor()
        encoded = encode_candidate(processor, fake_process_vision_info, [])
        self.assertEqual(encoded["input_ids"], [[1]])
        assert processor.call is not None
        self.assertEqual(len(processor.call["videos"]), 2)
        self.assertEqual(len(processor.call["video_metadata"]), 2)
        self.assertIs(processor.call["do_resize"], False)
        self.assertIs(processor.call["do_sample_frames"], False)

    def test_reviewer_encoder_rejects_metadata_free_video_inputs(self) -> None:
        class FakeImageProcessor:
            patch_size = 16

        class FakeProcessor:
            image_processor = FakeImageProcessor()

            def apply_chat_template(self, messages: list[dict], **kwargs: object) -> str:
                return "rendered"

        class FakeVideo:
            shape = (4, 3, 32, 32)

        def fake_process_vision_info(messages: list[dict], **kwargs: object) -> tuple:
            return None, [FakeVideo(), FakeVideo()], {"do_sample_frames": False}

        with self.assertRaisesRegex(RuntimeError, "did not return Qwen3-VL video metadata"):
            encode_candidate(FakeProcessor(), fake_process_vision_info, [])


class ScoreOnlyMediaContractTests(unittest.TestCase):
    def setUp(self) -> None:
        package_root = Path(__file__).resolve().parents[3]
        test_tmp = Path(os.environ.get("EGOQA_TEST_TMPDIR", package_root / "tmp"))
        test_tmp.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=test_tmp)
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _video(self, name: str) -> str:
        path = self.root / name
        path.write_bytes(b"not-empty")
        return str(path)

    def _frames(self, evidence_id: str, user: str, timestamps: list[float]) -> list[dict]:
        frames = []
        for order, timestamp in enumerate(timestamps, start=1):
            path = self.root / f"{evidence_id}-{user}-frame-{order}.jpg"
            path.write_bytes(b"not-empty")
            frames.append(
                {
                    "path": str(path),
                    "frame_role": "retained_clip_cluster_member",
                    "input_order_within_user": order,
                    "timestamp_seconds": timestamp,
                }
            )
        return frames

    def _packet(self, evidence_id: str = "evidence-1") -> dict:
        # Reverse clip order intentionally; required_users controls both model orders.
        return {
            "evidence_id": evidence_id,
            "required_users": ["A1", "A2"],
            "generator_media_mode": GENERATOR_MEDIA_MODE,
            "clips": [
                {
                    "agent_name": "A2",
                    "source_local_video": self._video(f"{evidence_id}-a2-full.mp4"),
                    "generator_media_mode": GENERATOR_MEDIA_MODE,
                    "force_frame_inputs": True,
                    "frames": self._frames(evidence_id, "a2", [2.0, 4.0]),
                },
                {
                    "agent_name": "A1",
                    "original_local_video": self._video(f"{evidence_id}-a1-full.mp4"),
                    "generator_media_mode": GENERATOR_MEDIA_MODE,
                    "force_frame_inputs": True,
                    "frames": self._frames(evidence_id, "a1", [1.0, 3.0, 5.0]),
                },
            ],
        }

    def test_row_binds_retained_frames_and_full_videos_in_required_user_order(self) -> None:
        row = packet_to_swift_row(
            self._packet(),
            question_type="neutral",
        )
        self.assertNotIn("videos", row)
        self.assertTrue(row["images"][0].endswith("evidence-1-a1-frame-1.jpg"))
        self.assertTrue(row["images"][2].endswith("evidence-1-a1-frame-3.jpg"))
        self.assertTrue(row["images"][3].endswith("evidence-1-a2-frame-1.jpg"))
        self.assertEqual(row["images"], row["generator_image_paths"])
        self.assertEqual(row["generator_frame_counts"], [3, 2])
        self.assertEqual(row["messages"][0]["content"].count("<image>"), 5)
        self.assertNotIn("<video>", row["messages"][0]["content"])
        self.assertTrue(row["reviewer_video_paths"][0].endswith("evidence-1-a1-full.mp4"))
        self.assertTrue(row["reviewer_video_paths"][1].endswith("evidence-1-a2-full.mp4"))
        packet = json.loads(row["packet_json"])
        self.assertEqual([clip["agent_name"] for clip in packet["clips"]], ["A1", "A2"])
        self.assertEqual(packet["required_users"], row["image_order"])
        self.assertEqual(packet["required_users"], row["reviewer_video_order"])

    def test_swapped_dataset_frame_paths_are_rejected(self) -> None:
        row = packet_to_swift_row(
            self._packet(),
            question_type="neutral",
        )
        row["images"] = list(reversed(row["images"]))
        with self.assertRaisesRegex(ValueError, "does not match packet media"):
            validate_swift_row(row)

    def test_video_generator_media_is_rejected(self) -> None:
        packet = self._packet()
        packet["clips"][0]["local_video"] = self._video("stale-pruned.mp4")
        with self.assertRaisesRegex(ValueError, "generator video media"):
            packet_to_swift_row(
                packet,
                question_type="neutral",
            )

    def test_image_placeholder_count_is_rechecked(self) -> None:
        row = packet_to_swift_row(
            self._packet(),
            question_type="neutral",
        )
        row["messages"][0]["content"] = row["messages"][0]["content"].replace(
            "<image>\n", "", 1
        )
        with self.assertRaisesRegex(ValueError, "one ordered <image> placeholder"):
            validate_swift_row(row)

    def test_production_prompt_describes_frame_ranges_and_neutral_only(self) -> None:
        row = packet_to_swift_row(self._packet(), question_type="neutral")
        content = row["messages"][0]["content"]
        self.assertIn("A1: packet images 1", content)
        self.assertIn("A2: packet images 4", content)
        self.assertIn('"question_type": "neutral"', content)
        self.assertNotIn('"question_type": "commonality, difference, or neutral"', content)
        for fragment in PRODUCTION_PROMPT_REQUIRED_FRAGMENTS:
            self.assertIn(fragment, content)
        self.assertEqual(row["prompt_contract"], PRODUCTION_PROMPT_CONTRACT)
        source_contract = production_prompt_source_contract()
        for field, expected in source_contract.items():
            self.assertEqual(row[field], expected)

    def test_missing_perspective_safeguard_is_rejected(self) -> None:
        row = packet_to_swift_row(self._packet(), question_type="neutral")
        fragment = PRODUCTION_PROMPT_REQUIRED_FRAGMENTS[3]
        row["messages"][0]["content"] = row["messages"][0]["content"].replace(
            fragment, "removed attribution check", 1
        )
        prefix = "\n".join("<image>" for _ in row["images"]) + "\n"
        prompt = row["messages"][0]["content"][len(prefix):]
        row["prompt_sha256"] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        with self.assertRaisesRegex(ValueError, "no-perspective-mixing safeguards"):
            validate_swift_row(row)

    def test_semantically_stale_production_prompt_is_always_rejected(self) -> None:
        row = packet_to_swift_row(self._packet(), question_type="neutral")
        row["messages"][0]["content"] += "\nUntracked prompt mutation."
        prefix = "\n".join("<image>" for _ in row["images"]) + "\n"
        prompt = row["messages"][0]["content"][len(prefix):]
        row["prompt_sha256"] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        with self.assertRaisesRegex(
            ValueError, "does not match the current production prompt output"
        ):
            validate_swift_row(row, require_current_prompt_source=False)

    def test_stale_source_hash_can_be_allowed_for_exact_current_prompt(self) -> None:
        row = packet_to_swift_row(self._packet(), question_type="neutral")
        row["prompt_source_sha256"] = "0" * 64
        validate_swift_row(row, require_current_prompt_source=False)
        with self.assertRaisesRegex(
            ValueError, "prompt_source_sha256 does not match"
        ):
            validate_swift_row(row)

    def test_reviewer_never_falls_back_to_generator_media(self) -> None:
        packet = self._packet()
        for clip in packet["clips"]:
            clip.pop("source_local_video", None)
            clip.pop("original_local_video", None)
        with self.assertRaisesRegex(ValueError, "has no reviewer video"):
            packet_to_swift_row(
                packet,
                question_type="neutral",
            )

    def test_runtime_rechecks_dataset_packet_binding(self) -> None:
        row = packet_to_swift_row(
            self._packet(),
            question_type="neutral",
        )
        packet = json.loads(row["packet_json"])
        with self.assertRaisesRegex(ValueError, "does not match packet media"):
            prepare_completion_for_review(
                _completion(),
                packet,
                evidence_id=row["evidence_id"],
                candidate_index=0,
                dataset_generator_images=list(reversed(row["images"])),
                dataset_reviewer_videos=row["reviewer_video_paths"],
            )

    def test_train_and_eval_manifests_are_both_validated(self) -> None:
        packets = [self._packet(f"evidence-{index}") for index in range(2)]
        train, evaluation, manifest = build_gate3_split(
            packets,
            train_count=1,
            eval_count=1,
        )
        self.assertTrue(all(row["question_type"] == "neutral" for row in train + evaluation))
        self.assertTrue(all(row["generation_mode"] == "baseline" for row in train + evaluation))
        self.assertEqual(manifest["train_question_type_counts"], {"neutral": 1})
        self.assertEqual(manifest["eval_question_type_counts"], {"neutral": 1})
        train_path = self.root / "train.jsonl"
        eval_path = self.root / "eval.jsonl"
        manifest_path = self.root / "split.json"
        write_jsonl(train_path, train)
        write_jsonl(eval_path, evaluation)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        self.assertEqual(
            validate_dataset(
                train_path,
                expected_rows=1,
                split_manifest=manifest_path,
                split="train",
            )["status"],
            "passed",
        )
        self.assertEqual(
            validate_dataset(
                eval_path,
                expected_rows=1,
                split_manifest=manifest_path,
                split="eval",
            )["status"],
            "passed",
        )

        manifest["eval_evidence_ids"] = list(manifest["train_evidence_ids"])
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "train/eval leakage"):
            validate_dataset(
                train_path,
                split_manifest=manifest_path,
                split="train",
            )

    def test_split_counts_must_be_positive(self) -> None:
        for train_count, eval_count in ((0, 1), (-1, 1), (1, 0), (1, -1)):
            with self.subTest(train_count=train_count, eval_count=eval_count):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    build_gate3_split(
                        [], train_count=train_count, eval_count=eval_count
                    )

    def test_named_reviewer_split_preserves_train_eval_test_ids(self) -> None:
        packets = [self._packet(f"evidence-{index}") for index in range(3)]
        train, evaluation, test, manifest = build_named_split(
            packets,
            train_evidence_ids=["evidence-1"],
            eval_evidence_ids=["evidence-0"],
            test_evidence_ids=["evidence-2"],
            seed=20260809,
        )
        self.assertEqual([row["evidence_id"] for row in train], ["evidence-1"])
        self.assertEqual([row["evidence_id"] for row in evaluation], ["evidence-0"])
        self.assertEqual([row["evidence_id"] for row in test], ["evidence-2"])
        self.assertEqual(manifest["test_count"], 1)
        self.assertEqual(manifest["selection_strategy"], "external_named_evidence_split")
        heldout_path = self.root / "heldout.jsonl"
        manifest_path = self.root / "heldout_split.json"
        write_jsonl(heldout_path, evaluation + test)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        result = validate_dataset(
            heldout_path,
            expected_rows=2,
            split_manifest=manifest_path,
            split="heldout",
        )
        self.assertEqual(result["split"], "heldout")
        self.assertEqual(result["split_manifest_count"], 2)

    def test_active_media_paths_can_be_rebased_to_current_project(self) -> None:
        packet = self._packet()
        old_root = "/scratch/old-user/Long-video-understanding-clip"
        stale_packet = json.loads(json.dumps(packet))
        for clip in stale_packet["clips"]:
            for frame in clip["frames"]:
                frame["path"] = f"{old_root}/{Path(frame['path']).name}"
            for field in ("full_local_video", "original_local_video", "source_local_video"):
                if field in clip:
                    clip[field] = f"{old_root}/{Path(clip[field]).name}"
        rebased = rebase_active_media_paths(
            stale_packet,
            source_project_root=old_root,
            target_project_root=str(self.root),
        )
        row = packet_to_swift_row(rebased, question_type="neutral")
        self.assertTrue(all(path.startswith(str(self.root)) for path in row["images"]))
        self.assertTrue(
            all(path.startswith(str(self.root)) for path in row["reviewer_video_paths"])
        )

    def _imported_reviewer_pair(self, *, reverse: bool = False) -> tuple[Path, Path]:
        evidence_id = "EGOLIFE2U_RANDOM_PAIR_CLIP_PRUNED_DAY1_11153000_A1_A3_0-1"
        agents = [
            ("A1_ALICE", "Alice", self._video("left_A1_ALICE_original.mp4")),
            ("A3_BOB", "Bob", self._video("right_A3_BOB_original.mp4")),
        ]
        if reverse:
            agents.reverse()
        row = {
            "schema_version": IMPORTED_SCHEMA_VERSION,
            "evidence_id": evidence_id,
            "day": "DAY1",
            "time_token": "11153000",
            "clips": [
                {
                    "agent_name": user,
                    "video_url": (
                        "https://huggingface.co/datasets/lmms-lab/EgoLife/"
                        f"resolve/main/{agent_dir}/DAY1/DAY1_{agent_dir}_11153000.mp4"
                    ),
                    "source_local_video": video,
                    "original_local_video": video,
                    "full_local_video": video,
                }
                for agent_dir, user, video in agents
            ],
        }
        import_path = self.root / "imported.jsonl"
        write_jsonl(import_path, [row])
        audit_path = self.root / "split_audit.json"
        audit_path.write_text(
            json.dumps(
                {
                    "split_manifest": {
                        "train_evidence_ids": [evidence_id],
                        "validation_evidence_ids": [],
                        "locked_test_evidence_ids": [],
                    }
                }
            ),
            encoding="utf-8",
        )
        return import_path, audit_path

    def test_reviewer_import_is_prepared_without_exposing_generator_media(self) -> None:
        import_path, audit_path = self._imported_reviewer_pair()
        output = self.root / "prepared.jsonl"
        report = prepare_imported_pairs(
            import_manifest=import_path,
            split_audit=audit_path,
            output_path=output,
            expected_count=1,
        )
        packet = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report["packet_count"], 1)
        self.assertEqual(packet["required_users"], ["Alice", "Bob"])
        self.assertEqual(packet["speaker_user"], "Alice")
        self.assertEqual(packet["evidence_provider_user"], "Bob")
        self.assertEqual(
            [clip["agent_id"] for clip in packet["clips"]], ["A1", "A3"]
        )
        self.assertTrue(all(clip["local_video"] for clip in packet["clips"]))
        self.assertTrue(
            all("generator_media_mode" not in clip for clip in packet["clips"])
        )
        self.assertTrue(all("frames" not in clip for clip in packet["clips"]))

    def test_reviewer_import_refuses_to_swap_asker_and_provider(self) -> None:
        import_path, audit_path = self._imported_reviewer_pair(reverse=True)
        with self.assertRaisesRegex(ValueError, "refusing to swap asker/provider order"):
            prepare_imported_pairs(
                import_manifest=import_path,
                split_audit=audit_path,
                output_path=self.root / "prepared.jsonl",
                expected_count=1,
            )


class ScoreOnlyCheckpointContractTests(unittest.TestCase):
    def _contract(self) -> dict:
        config = ReviewerV1Config(
            lora_target_modules=("q_proj", "k_proj"),
            lora_r=4,
            lora_alpha=12,
            lora_dropout=0.125,
            seed=7,
        )
        return {
            **config.to_dict(),
            "active_heads": list(config.active_heads),
            "lora_enabled": config.lora_enabled,
        }

    def test_all_serialized_lora_fields_are_reconstructed(self) -> None:
        loaded = reviewer_config_from_contract(self._contract())
        self.assertEqual(loaded.lora_target_modules, ("q_proj", "k_proj"))
        self.assertEqual(loaded.lora_r, 4)
        self.assertEqual(loaded.lora_alpha, 12)
        self.assertEqual(loaded.lora_dropout, 0.125)
        self.assertEqual(loaded.seed, 7)

    def test_runtime_contract_drift_is_rejected_before_model_load(self) -> None:
        contract = self._contract()
        contract["active_heads"] = ["evidence_quality"]
        with self.assertRaisesRegex(ValueError, "checkpoint runtime contract mismatch"):
            reviewer_config_from_contract(contract)

    def test_missing_serialized_config_field_is_rejected(self) -> None:
        contract = self._contract()
        del contract["lora_alpha"]
        with self.assertRaisesRegex(ValueError, "missing ReviewerV1Config fields"):
            reviewer_config_from_contract(contract)


class ScoreOnlyPluginContractTests(unittest.TestCase):
    def test_one_variable_length_frame_list_repeats_for_all_completions(self) -> None:
        paths = ["frame-1.jpg", "frame-2.jpg", "frame-3.jpg"]
        self.assertEqual(_expand_path_list(paths, 4, "frames"), [paths] * 4)

    def test_batched_frame_lists_remain_row_specific(self) -> None:
        paths = [["a.jpg", "b.jpg"], ["c.jpg", "d.jpg", "e.jpg"]]
        self.assertEqual(_expand_path_list(paths, 2, "frames"), paths)


class ScoreOnlySmokeValidationTests(unittest.TestCase):
    def test_nvidia_smi_csv_headers_with_spaces_are_normalized(self) -> None:
        import io

        from training.grpo_v3.runtime.validate_grpo_smoke import (
            parse_gpu_peaks,
        )

        self.assertEqual(
            parse_gpu_peaks(
                io.StringIO(
                "timestamp, index, name, memory.total [MiB], "
                "memory.used [MiB], utilization.gpu [%]\n"
                "2026/08/15 08:20:26.559, 0, NVIDIA H200, "
                "143771 MiB, 3411 MiB, 100 %\n"
                "2026/08/15 08:20:27.559, 0, NVIDIA H200, "
                "143771 MiB, 48763 MiB, 100 %\n"
                )
            ),
            {"0": {"total_mib": 143771, "peak_used_mib": 48763}},
        )


class ScoreOnlyHpcLayoutTests(unittest.TestCase):
    def test_runtime_scripts_use_project_level_hpc_root(self) -> None:
        project_root = Path(__file__).resolve().parents[4]
        hpc_root = project_root / "hpc" / "grpo_v3" / "score_only_reward"
        paths = [
            hpc_root / "common.sh",
            hpc_root / "cuda_keeper.sh",
            hpc_root / "reviewer_probe.sbatch",
            hpc_root / "grpo_smoke1.sbatch",
            hpc_root / "grpo_train.sbatch",
            hpc_root / "grpo_train_lr_grid.sbatch",
            hpc_root / "prepare_day1_and_dataset.sbatch",
            hpc_root / "compare_policies.sbatch",
            hpc_root / "compare_policies_test.sbatch",
            hpc_root / "grpo_high_lr_grid_eval30.sbatch",
        ]
        for path in paths:
            self.assertTrue(path.is_file(), path)
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("/scratch/xl6775", text)
            self.assertNotIn("projects/score-only-reward-model", text)
            self.assertNotIn("EgoQA-two-user", text)
            self.assertNotIn("${PACKAGE_ROOT}/hpc", text)

        common = paths[0].read_text(encoding="utf-8")
        self.assertIn('HPC_ROOT="${PROJECT_ROOT}/hpc"', common)
        self.assertIn(
            'PACKAGE_ROOT="${EGOLIFE2U_PACKAGE_ROOT:-${PROJECT_ROOT}/egolife_two_user_qa}"',
            common,
        )
        self.assertIn(
            "egolife_two_user_qa.training.grpo_v3.shared.validate_dataset",
            common,
        )
        self.assertIn(
            "egolife_two_user_qa.training.grpo_v3.runtime.validate_environment",
            common,
        )
        self.assertIn('export PIP_CACHE_DIR="${JOB_SCRATCH_ROOT}/pip"', common)
        self.assertIn('export PYTHONDONTWRITEBYTECODE=1', common)
        keeper = paths[1].read_text(encoding="utf-8")
        self.assertIn('${HPC_ROOT}/cuda_slurm.py', keeper)
        self.assertIn('${HPC_ROOT}/cuda.py', keeper)
        self.assertIn('cuda_keeper_device_mapping', keeper)
        smoke = paths[3].read_text(encoding="utf-8")
        self.assertIn(
            '${PACKAGE_ROOT}/training/grpo_v3/runtime/score_only_reward_plugin.py',
            smoke,
        )
        self.assertIn('--max_pixels "${MAX_PIXELS}"', smoke)
        self.assertIn('BETA="${BETA:-0.04}"', smoke)
        self.assertIn('TEMPERATURE="${TEMPERATURE:-0.85}"', smoke)
        self.assertIn('MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1536}"', smoke)
        self.assertIn('--num_generations "${NUM_GENERATIONS}"', smoke)
        self.assertIn(
            "egolife_two_user_qa.training.grpo_v3.runtime.validate_grpo_smoke",
            smoke,
        )
        self.assertIn('CUDA_KEEPER_GPUS="0,1"', smoke)
        self.assertIn("#SBATCH --constraint=h200", smoke)
        self.assertIn('stop_cuda_keeper', smoke)
        train = paths[4].read_text(encoding="utf-8")
        self.assertIn('--val_dataset "${VAL_DATASET}"', train)
        self.assertIn('--eval_strategy epoch', train)
        self.assertIn('NUM_GENERATIONS="${NUM_GENERATIONS:-4}"', train)
        self.assertIn('BETA="${BETA:-0.04}"', train)
        self.assertIn('MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1536}"', train)
        self.assertIn('CUDA_KEEPER_GPUS="0,1"', train)
        self.assertIn("#SBATCH --constraint=h200", train)
        self.assertIn('stop_cuda_keeper', train)
        grid = paths[5].read_text(encoding="utf-8")
        self.assertIn('LEARNING_RATES="${LEARNING_RATES:-1e-6 1e-5 3e-5}"', grid)
        self.assertIn('NUM_GENERATIONS="${NUM_GENERATIONS:-4}"', grid)
        self.assertIn('for LEARNING_RATE in "${LEARNING_RATE_ARRAY[@]}"', grid)
        self.assertIn('--model "${POLICY_MODEL}"', grid)
        self.assertIn('--output_dir "${RUN_OUTPUT_DIR}/swift"', grid)
        self.assertIn('CUDA_KEEPER_GPUS="0,1"', grid)
        self.assertIn("#SBATCH --constraint=h200", grid)
        self.assertIn('stop_cuda_keeper', grid)
        probe = paths[2].read_text(encoding="utf-8")
        self.assertIn('dataset_generator_images=row["generator_image_paths"]', probe)
        self.assertIn('"generator_image_paths": row["images"]', probe)
        self.assertIn('CUDA_KEEPER_GPUS="0"', probe)
        prepare = paths[6].read_text(encoding="utf-8")
        self.assertIn('CUDA_KEEPER_GPUS="0"', prepare)
        comparison = paths[7].read_text(encoding="utf-8")
        self.assertIn("#SBATCH --constraint=h200", comparison)
        self.assertIn('CUDA_KEEPER_GPUS="0,1"', comparison)
        self.assertIn('DATASET_SPLIT="${COMPARISON_SPLIT}"', comparison)
        self.assertIn('gate3_v3_eval_retained_frames.jsonl', comparison)
        self.assertIn('gate3_v3_test_retained_frames.jsonl', comparison)
        self.assertIn('heldout30_eval_then_test.jsonl', comparison)
        self.assertIn('COMPARISON_POLICY_SPECS="${COMPARISON_POLICY_SPECS:-', comparison)
        self.assertIn('POLICY_NAMES=(baseline)', comparison)
        self.assertIn('DECODING_NAMES=(greedy sampling)', comparison)
        self.assertIn('SAMPLING_TEMPERATURE="${SAMPLING_TEMPERATURE:-0.85}"', comparison)
        self.assertIn('SAMPLING_TOP_P="${SAMPLING_TOP_P:-0.95}"', comparison)
        self.assertIn('SAMPLING_TOP_K="${SAMPLING_TOP_K:-40}"', comparison)
        self.assertIn('--temperature "${DECODING_TEMPERATURE}"', comparison)
        self.assertIn('--top_p "${SAMPLING_TOP_P}"', comparison)
        self.assertIn('--top_k "${SAMPLING_TOP_K}"', comparison)
        self.assertIn('--remove_unused_columns true', comparison)
        self.assertIn('--adapters "${ADAPTER_DIR}"', comparison)
        self.assertIn(
            "egolife_two_user_qa.training.grpo_v3.evaluation.policy_comparison",
            comparison,
        )
        self.assertIn('stop_cuda_keeper', comparison)
        test_comparison = paths[8].read_text(encoding="utf-8")
        self.assertIn("#SBATCH --constraint=h100", test_comparison)
        self.assertIn("#SBATCH --time=20:00:00", test_comparison)
        self.assertIn('export MODE=grpo_compare_test', test_comparison)
        self.assertIn('export COMPARISON_SPLIT=test', test_comparison)
        self.assertIn('export EXPECTED_ROWS=20', test_comparison)
        self.assertIn('compare_policies.sbatch', test_comparison)
        high_lr = paths[9].read_text(encoding="utf-8")
        self.assertIn("#SBATCH --constraint=h200", high_lr)
        self.assertIn('LEARNING_RATES="5e-5 7e-5 1e-4"', high_lr)
        self.assertIn(
            'COMPARISON_POLICY_SPECS="lr_5e-5:lr_5e_minus_5 '
            'lr_7e-5:lr_7e_minus_5 lr_1e-4:lr_1e_minus_4"',
            high_lr,
        )
        self.assertIn('LORA_TARGET_MODULES="q_proj v_proj"', high_lr)
        self.assertIn('COMPARISON_SPLIT=heldout', high_lr)
        self.assertIn('EXPECTED_ROWS=30', high_lr)
        self.assertIn('comparison.get("generation_count") != 240', high_lr)


if __name__ == "__main__":
    unittest.main()
