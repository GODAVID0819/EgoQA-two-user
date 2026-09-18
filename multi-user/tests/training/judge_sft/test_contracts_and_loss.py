from __future__ import annotations

import json
import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from training.judge_sft.collator import (
    _apply_chat_template,
    adaptive_image_max_pixels,
    model_visible_prompt,
)
from training.judge_sft.contracts import (
    VERDICT_ASSISTANT_PREFIX,
    JudgeTask,
    Verdict,
    parse_complete_reason_fix_output,
    parse_verdict,
    render_locked_verdict_prefix,
)
from training.judge_sft.data import (
    FrameSet,
    JudgeExample,
    assert_group_disjoint,
    normalized_record_to_example,
)
from training.judge_sft.loss import (
    balanced_binary_class_weights,
    resolve_verdict_token_ids,
    task_sampling_scales,
    verdict_bce_from_pair_logits,
)
from training.judge_sft.inference import (
    generate_complete_judge_output,
    select_verdict_from_next_token_logits,
)
from training.judge_sft.trainer import logits_to_keep_argument
from training.judge_sft.train import audit_language_lora_targets


def _example(index: int, task: JudgeTask, verdict: Verdict) -> JudgeExample:
    frame_set = FrameSet(
        label="speaker: A1",
        frames=tuple(f"speaker-{frame}.jpg" for frame in range(300)),
    )
    if task is JudgeTask.FORMALITY:
        frame_sets = ()
        condition_type = None
    elif task is JudgeTask.GROUNDEDNESS:
        frame_sets = tuple(
            FrameSet(
                label=f"user-{user}",
                frames=tuple(f"user-{user}-{frame}.jpg" for frame in range(300)),
            )
            for user in range(6)
        )
        condition_type = None
    else:
        frame_sets = (frame_set,)
        condition_type = "speaker_only"
    return JudgeExample(
        example_id=f"e{index}",
        group_id=f"g{index}",
        task=task,
        prompt="prompt",
        verdict=verdict,
        frame_sets=frame_sets,
        condition_type=condition_type,
    )


class FakeTokenizer:
    ids = {VERDICT_ASSISTANT_PREFIX: [1, 2, 3], "fail": [7], "pass": [11]}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return self.ids[text]

    def decode(self, ids, **kwargs) -> str:
        del kwargs
        values = [int(value) for value in ids]
        if values == [1, 2, 3]:
            return VERDICT_ASSISTANT_PREFIX
        if values == [1, 2, 3, 7]:
            return VERDICT_ASSISTANT_PREFIX + "fail"
        if values == [1, 2, 3, 11]:
            return VERDICT_ASSISTANT_PREFIX + "pass"
        if values == [7]:
            return "fail"
        if values == [11]:
            return "pass"
        if values == [11, 12]:
            return 'pass","reason":null,"fix":null}'
        raise KeyError(values)


class VerdictContractsAndLossTests(unittest.TestCase):
    def test_parser_requires_verdict_first_and_exact_binary_value(self) -> None:
        self.assertIs(parse_verdict('{"verdict":"pass","reason":null}'), Verdict.PASS)
        self.assertIs(parse_verdict('{"verdict":"fail","reason":"x"}'), Verdict.FAIL)
        with self.assertRaisesRegex(ValueError, "first"):
            parse_verdict('{"reason":"x","verdict":"fail"}')
        with self.assertRaisesRegex(ValueError, "pass or fail"):
            parse_verdict('{"verdict":"maybe"}')

    def test_90_10_balancing_preserves_mean_scale(self) -> None:
        weights = balanced_binary_class_weights([1] * 90 + [0] * 10, smoothing=0)
        self.assertAlmostEqual(weights.passed, 100 / 180)
        self.assertAlmostEqual(weights.fail, 5.0)
        observed_mean = (90 * weights.passed + 10 * weights.fail) / 100
        self.assertAlmostEqual(observed_mean, 1.0)

    def test_extreme_imbalance_is_capped_without_changing_mean_scale(self) -> None:
        weights = balanced_binary_class_weights(
            [1] * 199 + [0], smoothing=0, max_weight=10
        )
        self.assertEqual(weights.fail, 10.0)
        self.assertLess(weights.passed, 1.0)
        observed_mean = (199 * weights.passed + weights.fail) / 200
        self.assertAlmostEqual(observed_mean, 1.0)

    def test_bce_uses_pass_minus_fail_margin(self) -> None:
        pair_logits = torch.tensor([[0.0, 2.0], [2.0, 0.0]])
        targets = torch.tensor([1, 0])
        loss = verdict_bce_from_pair_logits(pair_logits, targets)
        self.assertAlmostEqual(float(loss), math.log1p(math.exp(-2.0)), places=6)

    def test_task_scales_recover_point_two_point_four_point_four_means(self) -> None:
        examples = [
            _example(0, JudgeTask.FORMALITY, Verdict.PASS),
            _example(1, JudgeTask.FORMALITY, Verdict.FAIL),
            _example(2, JudgeTask.GROUNDEDNESS, Verdict.PASS),
            _example(3, JudgeTask.GROUNDEDNESS, Verdict.FAIL),
            _example(4, JudgeTask.ANSWERABILITY, Verdict.PASS),
            _example(5, JudgeTask.ANSWERABILITY, Verdict.PASS),
            _example(6, JudgeTask.ANSWERABILITY, Verdict.FAIL),
            _example(7, JudgeTask.ANSWERABILITY, Verdict.FAIL),
        ]
        scales = task_sampling_scales(
            examples,
            {
                JudgeTask.FORMALITY: 0.2,
                JudgeTask.GROUNDEDNESS: 0.4,
                JudgeTask.ANSWERABILITY: 0.4,
            },
        )
        self.assertAlmostEqual(scales[JudgeTask.FORMALITY], 0.8)
        self.assertAlmostEqual(scales[JudgeTask.GROUNDEDNESS], 1.6)
        self.assertAlmostEqual(scales[JudgeTask.ANSWERABILITY], 0.8)

    def test_token_contract_uses_two_distinct_single_tokens(self) -> None:
        token_ids = resolve_verdict_token_ids(FakeTokenizer())
        self.assertEqual(token_ids, {Verdict.FAIL: 7, Verdict.PASS: 11})

    def test_chat_template_disables_qwen_thinking_directly(self) -> None:
        class Processor:
            def __init__(self) -> None:
                self.enable_thinking = None

            def apply_chat_template(
                self,
                messages,
                *,
                tokenize,
                add_generation_prompt,
                enable_thinking,
            ):
                del messages, tokenize, add_generation_prompt
                self.enable_thinking = enable_thinking
                return "assistant-start"

        processor = Processor()
        self.assertEqual(_apply_chat_template(processor, []), "assistant-start")
        self.assertFalse(processor.enable_thinking)

    def test_verdict_selection_ignores_an_off_label_unrestricted_top_token(self) -> None:
        logits = torch.zeros(20)
        logits[3] = 10.0
        logits[7] = 1.0
        logits[11] = 2.0
        decision = select_verdict_from_next_token_logits(
            logits,
            verdict_token_ids={Verdict.FAIL: 7, Verdict.PASS: 11},
        )
        self.assertIs(decision.verdict, Verdict.PASS)
        self.assertFalse(decision.unrestricted_top_is_verdict)
        self.assertEqual(decision.locked_json_prefix, '{"verdict":"pass"')

    def test_code_supplies_both_quotes_around_locked_verdict(self) -> None:
        self.assertEqual(render_locked_verdict_prefix("fail"), '{"verdict":"fail"')

    def test_first_verdict_token_is_locked_then_same_generation_continues(self) -> None:
        class Model:
            def generate(self, *, input_ids, logits_processor, **kwargs):
                del kwargs
                scores = torch.zeros((1, 20))
                scores[0, 3] = 10.0  # Unrestricted top token must be ignored at step one.
                scores[0, 7] = 1.0
                scores[0, 11] = 2.0
                constrained = logits_processor[0](input_ids, scores)
                first = torch.argmax(constrained, dim=-1)[:, None]
                suffix = torch.tensor([[12]], dtype=torch.long)
                return SimpleNamespace(
                    sequences=torch.cat((input_ids, first, suffix), dim=-1)
                )

        generated = generate_complete_judge_output(
            Model(),
            FakeTokenizer(),
            {"input_ids": torch.tensor([[1, 2, 3]])},
            verdict_token_ids={Verdict.FAIL: 7, Verdict.PASS: 11},
        )
        self.assertIs(generated.decision.verdict, Verdict.PASS)
        self.assertFalse(generated.decision.unrestricted_top_is_verdict)
        self.assertEqual(
            generated.text,
            '{"verdict":"pass","reason":null,"fix":null}',
        )
        self.assertEqual(
            parse_complete_reason_fix_output(generated.text)["verdict"],
            "pass",
        )

    def test_normalized_visual_record_has_explicit_frame_block_order(self) -> None:
        def frame_set(user: int) -> dict:
            return {
                "label": "speaker: A1" if user == 0 else f"provider_{user}: A{user + 1}",
                "frames": [f"u{user}-{frame}.jpg" for frame in range(300)],
            }

        row = {
            "example_id": "x",
            "group_id": "set-1",
            "task": "answerability",
            "condition_type": "combined_all_six_users",
            "prompt": "judge this",
            "verdict": "pass",
            "frame_sets": [frame_set(user) for user in range(6)],
        }
        example = normalized_record_to_example(row, require_frame_files=False)
        prompt = model_visible_prompt(example)
        self.assertIn("block_1: speaker: A1; attachments 1-300", prompt)
        self.assertIn("block_2: provider_1: A2; attachments 301-600", prompt)
        self.assertEqual(example.frame_count, 1_800)

    def test_packet_relative_nested_frame_directories_are_resolved(self) -> None:
        packet_dir = (Path.cwd() / "nested-layout-packet").resolve()
        users = []
        for user_index in range(6):
            directory = f"user_{user_index}_A{user_index + 1}_NAME"
            users.append(
                {
                    "user_index": user_index,
                    "frames": [
                        {
                            "frame_index": frame_index,
                            "path": f"frames/{directory}/frame_{frame_index:06d}.jpg",
                        }
                        for frame_index in range(300)
                    ],
                }
            )
        packet_json = json.dumps(
            {
                "duration_seconds": 600.0,
                "preprocessing": {
                    "sampling": {"fps": 0.5, "interval_seconds": 2.0}
                },
                "users": users,
            }
        )
        row = {
            "example_id": "nested-frames",
            "group_id": "set-1",
            "task": "answerability",
            "condition_type": "speaker_only",
            "prompt": "judge this",
            "verdict": "pass",
            "frame_packet": str(packet_dir),
            "frame_user_indices": [0],
            "frame_order": ["speaker: A1"],
        }

        with patch.object(Path, "read_text", return_value=packet_json):
            example = normalized_record_to_example(row, require_frame_files=False)

        self.assertEqual(example.frame_count, 300)
        self.assertEqual(
            Path(example.frames[0]),
            (
                packet_dir
                / "frames"
                / "user_0_A1_NAME"
                / "frame_000000.jpg"
            ).resolve(),
        )
        self.assertEqual(
            Path(example.frames[-1]).name,
            "frame_000299.jpg",
        )

    def test_dynamic_resolution_cap_matches_production_1800_frame_call(self) -> None:
        self.assertEqual(
            adaptive_image_max_pixels(
                image_count=1_800,
                configured_max_pixels=262_144,
                min_pixels=3_136,
                max_input_tokens=262_144,
            ),
            91_728,
        )
        self.assertEqual(
            adaptive_image_max_pixels(
                image_count=300,
                configured_max_pixels=262_144,
                min_pixels=3_136,
                max_input_tokens=262_144,
            ),
            262_144,
        )

    def test_native_video_manifest_is_rejected_as_obsolete(self) -> None:
        with self.assertRaisesRegex(ValueError, "obsolete"):
            normalized_record_to_example(
                {
                    "example_id": "x",
                    "task": "groundedness",
                    "prompt": "p",
                    "verdict": "pass",
                    "videos": ["x.mp4"],
                    "video_order": ["speaker"],
                },
                require_frame_files=False,
            )

    def test_group_leakage_is_rejected(self) -> None:
        train = [_example(0, JudgeTask.FORMALITY, Verdict.PASS)]
        eval_row = JudgeExample(
            **{**train[0].__dict__, "example_id": "eval"}
        )
        with self.assertRaisesRegex(ValueError, "leakage"):
            assert_group_disjoint(train, [eval_row])

    def test_logits_to_keep_is_mandatory(self) -> None:
        class Good:
            def forward(self, input_ids, logits_to_keep=0):
                return None

        class Bad:
            def forward(self, input_ids):
                return None

        self.assertEqual(logits_to_keep_argument(Good()), "logits_to_keep")
        with self.assertRaisesRegex(RuntimeError, "Materializing"):
            logits_to_keep_argument(Bad())

    def test_every_requested_language_lora_family_must_be_present(self) -> None:
        class Model:
            def named_parameters(self):
                for target in ("q_proj", "gate_proj", "up_proj", "down_proj"):
                    yield (
                        f"base_model.model.layers.0.mlp.{target}.lora_A.default.weight",
                        torch.nn.Parameter(torch.ones(2)),
                    )
                yield (
                    "base_model.model.visual.q_proj.lora_A.default.weight",
                    torch.nn.Parameter(torch.ones(5)),
                )

        counts = audit_language_lora_targets(
            Model(),
            ["q_proj", "gate_proj", "up_proj", "down_proj"],
        )
        self.assertEqual(counts["q_proj"], 2)
        with self.assertRaisesRegex(RuntimeError, "v_proj"):
            audit_language_lora_targets(Model(), ["q_proj", "v_proj"])


if __name__ == "__main__":
    unittest.main()
