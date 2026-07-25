from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from qwen3vl_runner import Qwen3VLTransformersRunner
from training.grpo_v3_formality_fixed_eval import (
    FIXED_EVAL_MIN_VIDEO_PIXELS,
    FIXED_SEEDS,
    analyze_fixed_eval,
    build_run_manifest,
    build_checkpoint_inventory,
    evaluate_adapter,
    select_probe_row,
)


class _VisionBoundaryReached(RuntimeError):
    pass


class QwenVideoPixelBoundsTests(unittest.TestCase):
    def test_runner_passes_explicit_video_minimum_below_frozen_maximum(self) -> None:
        runner = object.__new__(Qwen3VLTransformersRunner)
        runner.max_image_pixels = 50176
        runner.min_video_pixels = 4096
        runner.disable_thinking = False
        runner.processor = type(
            "Processor",
            (),
            {"apply_chat_template": lambda self, messages, **kwargs: "prompt"},
        )()
        captured = {}

        def capture(messages, **kwargs):
            captured["messages"] = messages
            raise _VisionBoundaryReached

        runner.process_vision_info = capture

        with self.assertRaises(_VisionBoundaryReached):
            runner._generate("prompt", video_paths=["/v/u1.mp4", "/v/u2.mp4"])

        videos = captured["messages"][0]["content"][:2]
        self.assertEqual(
            videos,
            [
                {
                    "type": "video",
                    "video": "/v/u1.mp4",
                    "min_pixels": 4096,
                    "max_pixels": 50176,
                    "fps": 1.0,
                },
                {
                    "type": "video",
                    "video": "/v/u2.mp4",
                    "min_pixels": 4096,
                    "max_pixels": 50176,
                    "fps": 1.0,
                },
            ],
        )


def _record(reward: float, *, unjudgeable: bool = False) -> dict:
    return {
        "masked": False,
        "judge_called": not unjudgeable,
        "reward_source": (
            "deterministic_unjudgeable_floor"
            if unjudgeable
            else "judge_pass_fail_logprob_margin"
        ),
        "reward_components": {"qa_formality_confidence": reward},
        "judge_trace": {} if unjudgeable else {"qa_formality": {"parsed": {}}},
    }


def _rows(*, delta: float = 0.3, final_unjudgeable: bool = False) -> list[dict]:
    rows = []
    for step in (0, 40):
        for index, seed in enumerate(FIXED_SEEDS):
            reward = -0.2 + index * 0.01 + (delta if step == 40 else 0.0)
            unjudgeable = bool(step == 40 and final_unjudgeable and index == 0)
            if unjudgeable:
                reward = -1.0
            rows.append(
                {
                    "checkpoint_step": step,
                    "checkpoint_label": f"step_{step}",
                    "seed": seed,
                    "reward": reward,
                    "record": _record(reward, unjudgeable=unjudgeable),
                }
            )
    return rows


class FixedEvalAnalysisTests(unittest.TestCase):
    def test_improved_requires_positive_paired_confidence_interval(self) -> None:
        result = analyze_fixed_eval(_rows(delta=0.3), bootstrap_replicates=500)

        self.assertEqual(result["run_status"], "passed")
        self.assertEqual(result["experiment_conclusion"], "improved")
        self.assertEqual(result["row_count"], 32)
        self.assertAlmostEqual(result["reward_delta"], 0.3)
        self.assertGreater(result["paired_bootstrap_95_ci"][0], 0.0)
        self.assertEqual(result["paired_comparison"], {"wins": 16, "ties": 0, "losses": 0})

    def test_bootstrap_is_reproducible_and_small_positive_delta_is_inconclusive(self) -> None:
        rows = _rows(delta=0.02)
        for index, row in enumerate(rows[16:]):
            row["reward"] += -0.08 if index % 2 == 0 else 0.08
            row["record"] = _record(row["reward"])
        first = analyze_fixed_eval(rows, bootstrap_replicates=800)
        second = analyze_fixed_eval(rows, bootstrap_replicates=800)

        self.assertEqual(first["paired_bootstrap_95_ci"], second["paired_bootstrap_95_ci"])
        self.assertEqual(first["experiment_conclusion"], "inconclusive")
        self.assertLessEqual(first["paired_bootstrap_95_ci"][0], 0.0)

    def test_nonpositive_delta_or_rising_unjudgeable_rate_is_not_improved(self) -> None:
        self.assertEqual(
            analyze_fixed_eval(_rows(delta=-0.01), bootstrap_replicates=100)["experiment_conclusion"],
            "not_improved",
        )
        result = analyze_fixed_eval(
            _rows(delta=0.4, final_unjudgeable=True), bootstrap_replicates=100
        )
        self.assertEqual(result["experiment_conclusion"], "not_improved")
        self.assertEqual(result["checkpoints"]["40"]["unjudgeable_count"], 1)
        self.assertEqual(result["checkpoints"]["40"]["judgeable_count"], 15)

    def test_invalid_keyspace_mask_component_or_nonfinite_reward_is_rejected(self) -> None:
        cases = []
        missing = _rows()
        missing.pop()
        cases.append(missing)
        duplicate = _rows()
        duplicate[-1] = dict(duplicate[0])
        cases.append(duplicate)
        nonfinite = _rows()
        nonfinite[0]["reward"] = math.inf
        cases.append(nonfinite)
        masked = _rows()
        masked[0]["record"]["masked"] = True
        cases.append(masked)
        contaminated = _rows()
        contaminated[0]["record"]["reward_components"]["groundedness"] = 1.0
        cases.append(contaminated)

        for rows in cases:
            with self.subTest(case=cases.index(rows)):
                with self.assertRaises((ValueError, RuntimeError)):
                    analyze_fixed_eval(rows, bootstrap_replicates=20)

    def test_manifest_separates_run_status_from_experiment_conclusion(self) -> None:
        summary = analyze_fixed_eval(_rows(delta=-0.01), bootstrap_replicates=20)
        manifest = build_run_manifest(
            summary=summary,
            artifact_paths={"results": "/out/fixed_eval_results.jsonl"},
            storage_preflight={"status": "passed"},
        )

        self.assertEqual(manifest["run_status"], "passed")
        self.assertEqual(manifest["experiment_conclusion"], "not_improved")
        self.assertEqual(manifest["storage_preflight_status"], "passed")
        with self.assertRaisesRegex(RuntimeError, "storage preflight"):
            build_run_manifest(
                summary=summary,
                artifact_paths={},
                storage_preflight={"status": "failed"},
            )


class FixedEvalInventoryTests(unittest.TestCase):
    def test_inventory_resolves_parent_and_only_checkpoint_40(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "gate2"
            parent_adapter = parent / "swift" / "checkpoint-1"
            final_adapter = root / "probe" / "swift" / "run" / "checkpoint-40"
            for adapter in (parent_adapter, final_adapter):
                adapter.mkdir(parents=True)
                (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
                (adapter / "adapter_model.safetensors").write_bytes(b"weights")
            (parent / "run_manifest.json").write_text(
                json.dumps({"adapter_dir": str(parent_adapter)}), encoding="utf-8"
            )
            probe = root / "probe"
            (probe / "resolved_config.json").write_text(
                json.dumps({"parent_run": str(parent)}), encoding="utf-8"
            )

            inventory = build_checkpoint_inventory(probe)

            self.assertEqual([item["checkpoint_step"] for item in inventory["checkpoints"]], [0, 40])
            self.assertEqual(inventory["status"], "passed")

    def test_inventory_rejects_missing_or_extra_final_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            probe = Path(directory)
            (probe / "resolved_config.json").write_text(
                json.dumps({"parent_run": str(probe / "missing")}), encoding="utf-8"
            )
            with self.assertRaises((FileNotFoundError, ValueError)):
                build_checkpoint_inventory(probe)

    def test_probe_row_is_selected_by_trace_evidence_id_not_dataset_position(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            probe = Path(directory)
            (probe / "reward_trace.jsonl").write_text(
                json.dumps({"evidence_id": "E2"}) + "\n" + json.dumps({"evidence_id": "E2"}) + "\n",
                encoding="utf-8",
            )
            rows = [{"evidence_id": "E1"}, {"evidence_id": "E2"}]

            selected = select_probe_row(rows, probe)

            self.assertEqual(selected["evidence_id"], "E2")

    def test_probe_row_rejects_multiple_trace_evidence_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            probe = Path(directory)
            (probe / "reward_trace.jsonl").write_text(
                json.dumps({"evidence_id": "E1"}) + "\n" + json.dumps({"evidence_id": "E2"}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "唯一 evidence"):
                select_probe_row([{"evidence_id": "E1"}, {"evidence_id": "E2"}], probe)


class _Model:
    def __init__(self) -> None:
        self.adapters = []

    def set_adapter(self, name: str) -> None:
        self.adapters.append(name)


class _Runner:
    model_id = "policy"

    def __init__(self) -> None:
        self.model = _Model()
        self.calls = []

    def generate(self, prompt, image_paths=None, video_paths=None, **kwargs):
        self.calls.append(
            {"prompt": prompt, "image_paths": image_paths, "video_paths": video_paths, **kwargs}
        )
        return '{"question":"q","options":["a","b","c","d","e"],"correct":"a"}'


class FixedEvalGenerationTests(unittest.TestCase):
    def test_fixed_eval_video_minimum_is_valid_under_frozen_maximum(self) -> None:
        self.assertEqual(FIXED_EVAL_MIN_VIDEO_PIXELS, 4 * 32 * 32)
        self.assertLessEqual(FIXED_EVAL_MIN_VIDEO_PIXELS, 50176)

    def test_sampling_uses_fixed_seeds_native_videos_and_formality_only_records(self) -> None:
        runner = _Runner()
        score_calls = []

        def scorer(**kwargs):
            score_calls.append(kwargs)
            return {"reward": 0.25, "record": _record(0.25)}

        row = {
            "messages": [{"role": "user", "content": "<video><video>\nrepo prompt"}],
            "videos": ["/v/u1.mp4", "/v/u2.mp4"],
            "evidence_id": "E1",
            "packet_json": '{"evidence_id":"E1","required_users":["u1","u2"]}',
            "question_type": "commonality",
            "generation_mode": "baseline",
        }
        results = evaluate_adapter(
            row=row,
            runner=runner,
            score_fn=scorer,
            checkpoint_step=40,
            adapter_dir=Path("/adapter/checkpoint-40"),
            seeds=FIXED_SEEDS,
            temperature=0.7,
        )

        self.assertEqual(len(results), 16)
        self.assertEqual(runner.model.adapters, ["step_40"])
        self.assertEqual([item["seed"] for item in results], list(FIXED_SEEDS))
        self.assertEqual(len(score_calls), 16)
        for call in runner.calls:
            self.assertEqual(call["video_paths"], ["/v/u1.mp4", "/v/u2.mp4"])
            self.assertEqual(call["decoding_mode"], "sampling")
            self.assertEqual(call["temperature"], 0.7)
            self.assertEqual(call["top_p"], 1.0)
        self.assertEqual(results[0]["record"]["reward_components"], {"qa_formality_confidence": 0.25})


if __name__ == "__main__":
    unittest.main()
