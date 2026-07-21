from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from training.grpo_v2_lora import (
    RepoJudgeReward,
    build_training_rows,
    completion_text,
    expand_to_length,
)


class CompletionTextTests(unittest.TestCase):
    def test_accepts_standard_and_conversational_completions(self) -> None:
        self.assertEqual(completion_text("plain"), "plain")
        self.assertEqual(
            completion_text([{"role": "assistant", "content": "answer"}]),
            "answer",
        )
        self.assertEqual(
            completion_text(
                [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "json answer"}],
                    }
                ]
            ),
            "json answer",
        )

    def test_expand_to_length_broadcasts_one_value(self) -> None:
        self.assertEqual(expand_to_length(["packet"], 3, name="packet_json"), ["packet"] * 3)
        with self.assertRaisesRegex(ValueError, "packet_json"):
            expand_to_length(["a", "b"], 3, name="packet_json")


class DatasetRowTests(unittest.TestCase):
    def test_builds_one_conversational_multimodal_prompt(self) -> None:
        packet = {
            "evidence_id": "e1",
            "required_users": ["P1", "P2"],
            "clips": [{"agent_name": "P1"}, {"agent_name": "P2"}],
        }
        rows = build_training_rows(
            [packet],
            max_prompts=1,
            question_type="commonality",
            prompt_builder=lambda *_args, **_kwargs: "PROMPT",
            media_selector=lambda *_args, **_kwargs: (["a.jpg"], ["b.mp4"]),
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["evidence_id"], "e1")
        self.assertEqual(rows[0]["question_type"], "commonality")
        content = rows[0]["prompt"][0]["content"]
        self.assertEqual(content[0], {"type": "image", "image": "a.jpg"})
        self.assertEqual(content[1], {"type": "video", "video": "b.mp4"})
        self.assertEqual(content[-1], {"type": "text", "text": "PROMPT"})
        self.assertEqual(json.loads(rows[0]["packet_json"])["evidence_id"], "e1")

    def test_sampled_frames_mode_avoids_native_video_batching(self) -> None:
        packet = {
            "evidence_id": "e1",
            "clips": [{"side": "A"}, {"side": "B"}],
        }
        frames = {
            "A": [f"a{i}.jpg" for i in range(6)],
            "B": [f"b{i}.jpg" for i in range(3)],
        }
        rows = build_training_rows(
            [packet],
            max_prompts=1,
            question_type="commonality",
            policy_media_mode="sampled_frames",
            max_policy_frames_per_clip=4,
            prompt_builder=lambda *_args, **_kwargs: "PROMPT",
            media_selector=lambda *_args, **_kwargs: ([], ["a.mp4", "b.mp4"]),
            frame_selector=lambda clip: frames[clip["side"]],
        )
        content = rows[0]["prompt"][0]["content"]
        self.assertEqual(
            [item["image"] for item in content if item["type"] == "image"],
            ["a0.jpg", "a2.jpg", "a3.jpg", "a5.jpg", "b0.jpg", "b1.jpg", "b2.jpg"],
        )
        self.assertFalse(any(item["type"] == "video" for item in content))


class RewardAdapterTests(unittest.TestCase):
    def test_maps_rewards_and_persists_a_trace(self) -> None:
        calls: list[dict[str, object]] = []

        def fake_score(**kwargs):
            calls.append(kwargs)
            raw = str(kwargs["raw_completion"])
            return {
                "reward": 1.25 if "good" in raw else None,
                "record": {"masked": "good" not in raw, "raw_qa": raw},
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            trace_path = Path(temp_dir) / "reward_trace.jsonl"
            reward = RepoJudgeReward(trace_path=trace_path, score_fn=fake_score)
            values = reward(
                completions=[
                    [{"role": "assistant", "content": "good"}],
                    "bad",
                ],
                packet_json=[json.dumps({"evidence_id": "e1"})],
                evidence_id=["e1"],
                question_type=["commonality"],
                generation_mode=["baseline"],
            )

            self.assertEqual(values, [1.25, None])
            self.assertEqual([call["candidate_index"] for call in calls], [0, 1])
            rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["reward"], 1.25)
            self.assertIsNone(rows[1]["reward"])


class SlurmScriptTests(unittest.TestCase):
    def test_slurm_and_preflight_share_revision_v5(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        script = (project_root / "hpc" / "grpo_v2_lora_8b_smoke.sbatch").read_text(
            encoding="utf-8"
        )
        preflight = (project_root / "training" / "grpo_v2_cpu_preflight.py").read_text(
            encoding="utf-8"
        )
        revision = "2026-07-14-multimodal-batch-preflight-v5"
        self.assertIn(f'SCRIPT_REVISION="{revision}"', script)
        self.assertIn(f'EXPECTED_REVISION = "{revision}"', preflight)

    def test_cpu_preflight_exercises_exact_multimodal_batch_shape(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        preflight = (project_root / "training" / "grpo_v2_cpu_preflight.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("apply_chat_template", preflight)
        self.assertIn('batch_prompts = [rows[0]["prompt"] for _ in range(batch_size)]', preflight)
        self.assertIn("conversation=batch_prompts", preflight)
        self.assertIn("PROCESSOR_BATCH_MULTIMODAL_OK", preflight)

    def test_disables_flashinfer_sampler_that_requires_ninja_jit(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        script = (project_root / "hpc" / "grpo_v2_lora_8b_smoke.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"',
            script,
        )

    def test_cpu_preflight_rejects_a_stale_slurm_script_without_gpu(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        preflight = project_root / "training" / "grpo_v2_cpu_preflight.py"
        self.assertTrue(preflight.is_file(), "缺少 CPU preflight 入口")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "hpc").mkdir()
            (root / "hpc" / "grpo_v2_lora_8b_smoke.sbatch").write_text(
                "#!/usr/bin/env bash\n",
                encoding="utf-8",
            )
            model = root / "model"
            model.mkdir()
            evidence = root / "evidence.jsonl"
            evidence.write_text('{"evidence_id":"e1","clips":[{}]}\n', encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(preflight),
                    "--project-root",
                    str(root),
                    "--model-path",
                    str(model),
                    "--evidence",
                    str(evidence),
                    "--static-only",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("STALE_SBATCH", result.stdout + result.stderr)

    def test_cpu_preflight_rejects_flashinfer_variable_typo(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        preflight = project_root / "training" / "grpo_v2_cpu_preflight.py"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "hpc").mkdir()
            (root / "hpc" / "grpo_v2_lora_8b_smoke.sbatch").write_text(
                'SCRIPT_REVISION="2026-07-14-multimodal-batch-preflight-v5"\n'
                'export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLE:-0}"\n',
                encoding="utf-8",
            )
            model = root / "model"
            model.mkdir()
            evidence = root / "evidence.jsonl"
            evidence.write_text('{"evidence_id":"e1","clips":[{}]}\n', encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(preflight),
                    "--project-root",
                    str(root),
                    "--model-path",
                    str(model),
                    "--evidence",
                    str(evidence),
                    "--static-only",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("SBATCH_VARIABLE_TYPO", result.stdout + result.stderr)

    def test_cpu_preflight_accepts_the_current_slurm_script_without_gpu(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        preflight = project_root / "training" / "grpo_v2_cpu_preflight.py"
        self.assertTrue(preflight.is_file(), "缺少 CPU preflight 入口")

        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "model"
            model.mkdir()
            evidence = Path(temp_dir) / "evidence.jsonl"
            evidence.write_text('{"evidence_id":"e1","clips":[{}]}\n', encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(preflight),
                    "--project-root",
                    str(project_root),
                    "--model-path",
                    str(model),
                    "--evidence",
                    str(evidence),
                    "--static-only",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("CPU_PREFLIGHT_STATIC_OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
