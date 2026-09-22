from __future__ import annotations

import json
import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from training.judge_sft.collator import (
    JudgeFrameCollator,
    QWEN_NO_THINK_ASSISTANT_SUFFIX,
    QWEN_VISION_TOKEN_PIXEL_AREA,
    _assert_thinking_disabled,
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
    JudgeDataset,
    JudgeExample,
    assert_group_disjoint,
    normalized_record_to_example,
)
from training.judge_sft.loss import (
    BinaryClassWeights,
    balanced_binary_class_weights,
    resolve_verdict_token_ids,
    sample_weight_for_example,
    task_sampling_scales,
    verdict_bce_from_pair_logits,
)
from training.judge_sft.inference import (
    generate_complete_judge_output,
    select_verdict_from_next_token_logits,
)
from training.judge_sft.trainer import (
    TensorParallelReplicatedSampler,
    audit_tensor_parallel_sampler,
    logits_to_keep_argument,
    model_execution_signature,
)
from training.judge_sft.train import (
    audit_language_lora_targets,
    audit_trainable_lora_layers,
    audit_trainable_lora_dtypes,
    configure_safe_tensor_parallel_plan,
    ensure_tensor_parallel_metadata,
    install_frozen_prefix_input_guard,
)


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

    def test_tensor_parallel_sampler_replicates_one_lossless_order(self) -> None:
        examples = [
            *[
                _example(index, JudgeTask.FORMALITY, Verdict.PASS)
                for index in range(3)
            ],
            *[
                _example(index + 3, JudgeTask.ANSWERABILITY, Verdict.FAIL)
                for index in range(3)
            ],
            *[
                _example(index + 6, JudgeTask.GROUNDEDNESS, Verdict.PASS)
                for index in range(3)
            ],
        ]
        dataset = JudgeDataset(examples)
        sampler = TensorParallelReplicatedSampler(dataset, seed=42)
        second_rank = TensorParallelReplicatedSampler(dataset, seed=42)

        indices = list(sampler)

        self.assertEqual(len(indices), 9)
        self.assertEqual(set(indices), set(range(9)))
        self.assertEqual(indices, list(second_rank))
        signatures = [model_execution_signature(dataset[index]) for index in indices]
        self.assertEqual(len(signatures), 9)
        audit = audit_tensor_parallel_sampler(
            dataset,
            tensor_parallel_size=2,
            seed=42,
        )
        self.assertEqual(audit["status"], "passed")
        self.assertEqual(audit["real_examples_per_epoch"], 9)
        self.assertEqual(audit["sampler_slots_per_rank_per_epoch"], 9)
        self.assertEqual(audit["zero_loss_padding_slots_per_epoch"], 0)
        self.assertTrue(audit["identical_order_on_every_tp_rank"])

    def test_sampler_keeps_shared_media_adjacent_for_cpu_cache_reuse(self) -> None:
        shared = _example(0, JudgeTask.GROUNDEDNESS, Verdict.PASS)
        same_media = JudgeExample(
            **{**shared.__dict__, "example_id": "same-media"}
        )
        other_packet = JudgeExample(
            **{
                **shared.__dict__,
                "example_id": "other-packet",
                "group_id": "other-packet",
            }
        )
        dataset = JudgeDataset([shared, other_packet, same_media])
        order = list(TensorParallelReplicatedSampler(dataset, seed=42))
        shared_positions = sorted((order.index(0), order.index(2)))

        self.assertEqual(shared_positions[1] - shared_positions[0], 1)
        audit = audit_tensor_parallel_sampler(
            dataset, tensor_parallel_size=2, seed=42
        )
        self.assertEqual(audit["media_locality_buckets"], 2)

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

    def test_chat_template_uses_transformers_v5_structured_kwargs(self) -> None:
        class Processor:
            def __init__(self) -> None:
                self.kwargs = None

            def apply_chat_template(
                self,
                messages,
                **kwargs: "Unpack[AllKwargsForChatTemplate]",
            ):
                del messages
                self.kwargs = kwargs
                if kwargs.get("template_kwargs") == {"enable_thinking": False}:
                    return "assistant-start" + QWEN_NO_THINK_ASSISTANT_SUFFIX
                return "assistant-start<think>\n"

        processor = Processor()
        rendered = _apply_chat_template(processor, [])
        _assert_thinking_disabled(rendered)
        self.assertEqual(
            processor.kwargs["template_kwargs"], {"enable_thinking": False}
        )

    def test_thinking_guard_accepts_only_closed_empty_qwen_block(self) -> None:
        _assert_thinking_disabled("assistant-start")
        _assert_thinking_disabled(
            "assistant-start" + QWEN_NO_THINK_ASSISTANT_SUFFIX
        )
        with self.assertRaisesRegex(RuntimeError, "active or non-empty"):
            _assert_thinking_disabled("assistant-start<think>\n")
        with self.assertRaisesRegex(RuntimeError, "active or non-empty"):
            _assert_thinking_disabled(
                "assistant-start<think>private reasoning</think>\n\n"
            )

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
        self.assertIn("image_group_1: speaker: A1; the next 300 images", prompt)
        self.assertIn("image_group_2: provider_1: A2; the next 300 images", prompt)
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
        self.assertEqual(QWEN_VISION_TOKEN_PIXEL_AREA, 32 * 32)
        self.assertEqual(
            adaptive_image_max_pixels(
                image_count=1_800,
                configured_max_pixels=262_144,
                min_pixels=3_136,
                max_input_tokens=262_144,
            ),
            119_808,
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

    def test_collator_uses_1800_independent_images_without_video_packing(self) -> None:
        class Processor:
            def __init__(self) -> None:
                self.call_kwargs = None
                self.image_processor = SimpleNamespace(
                    patch_size=16,
                    merge_size=2,
                    temporal_patch_size=2,
                )

            def apply_chat_template(
                self,
                messages,
                *,
                tokenize,
                add_generation_prompt,
                enable_thinking,
            ):
                del messages, tokenize, add_generation_prompt, enable_thinking
                return "assistant-start"

            def __call__(self, **kwargs):
                self.call_kwargs = kwargs
                return {"input_ids": torch.tensor([[1, 2, 3]])}

        observed_messages = []

        def process_vision_info(
            messages,
            *,
            image_patch_size,
        ):
            self.assertEqual(image_patch_size, 16)
            observed_messages.extend(messages)
            image_count = sum(
                item.get("type") == "image" for item in messages[0]["content"]
            )
            return [f"decoded-image-{index}" for index in range(image_count)], None

        processor = Processor()
        unit_weights = BinaryClassWeights(
            fail=1.0,
            passed=1.0,
            fail_count=1,
            pass_count=1,
        )
        collator = JudgeFrameCollator(
            processor=processor,
            class_weights={task: unit_weights for task in JudgeTask},
            task_scales={task: 1.0 for task in JudgeTask},
            min_pixels=3_136,
            max_pixels=262_144,
            max_input_tokens=262_144,
            process_vision_info=process_vision_info,
        )

        batch = collator([_example(0, JudgeTask.GROUNDEDNESS, Verdict.PASS)])

        image_items = observed_messages[0]["content"][:1_800]
        self.assertEqual(len(image_items), 1_800)
        self.assertTrue(all(item["type"] == "image" for item in image_items))
        self.assertTrue(
            all(
                item["max_pixels"] == 119_808
                for item in image_items
            )
        )
        self.assertEqual(
            processor.call_kwargs["images"],
            [f"decoded-image-{index}" for index in range(1_800)],
        )
        self.assertNotIn("videos", processor.call_kwargs)
        self.assertNotIn("do_resize", processor.call_kwargs)
        self.assertNotIn("cap_pixels_per_frame", processor.call_kwargs)
        self.assertEqual(batch["labels"].tolist(), [[1, 1]])
        self.assertEqual(batch["tp_example_fingerprint"].shape, (1,))

    def test_collator_reuses_packet_local_decoded_images(self) -> None:
        class Processor:
            image_processor = SimpleNamespace(
                patch_size=16,
                merge_size=2,
                temporal_patch_size=2,
            )

            def apply_chat_template(
                self,
                messages,
                *,
                tokenize,
                add_generation_prompt,
                enable_thinking,
            ):
                del messages, tokenize, add_generation_prompt, enable_thinking
                return "assistant-start"

            def __call__(self, **kwargs):
                del kwargs
                return {"input_ids": torch.tensor([[1, 2, 3]])}

        decode_calls = []

        def process_vision_info(messages, *, image_patch_size):
            del image_patch_size
            decode_calls.append(messages)
            return [object() for _ in range(1_800)], None

        unit_weights = BinaryClassWeights(
            fail=1.0,
            passed=1.0,
            fail_count=1,
            pass_count=1,
        )
        collator = JudgeFrameCollator(
            processor=Processor(),
            class_weights={task: unit_weights for task in JudgeTask},
            task_scales={task: 1.0 for task in JudgeTask},
            min_pixels=3_136,
            max_pixels=262_144,
            max_input_tokens=262_144,
            decoded_image_cache_entries=2,
            process_vision_info=process_vision_info,
        )
        example = _example(0, JudgeTask.GROUNDEDNESS, Verdict.PASS)

        first = collator([example])
        second = collator([example])

        self.assertEqual(len(decode_calls), 1)
        self.assertTrue(torch.equal(first["input_ids"], second["input_ids"]))

    def test_collator_rejects_independent_image_context_above_limit(self) -> None:
        class Processor:
            image_processor = SimpleNamespace(
                patch_size=16,
                merge_size=2,
                temporal_patch_size=2,
            )

            def apply_chat_template(
                self,
                messages,
                *,
                tokenize,
                add_generation_prompt,
                enable_thinking,
            ):
                del messages, tokenize, add_generation_prompt, enable_thinking
                return "assistant-start"

            def __call__(self, **kwargs):
                del kwargs
                return {"input_ids": torch.zeros((1, 262_145), dtype=torch.long)}

        unit_weights = BinaryClassWeights(
            fail=1.0,
            passed=1.0,
            fail_count=1,
            pass_count=1,
        )
        collator = JudgeFrameCollator(
            processor=Processor(),
            class_weights={task: unit_weights for task in JudgeTask},
            task_scales={task: 1.0 for task in JudgeTask},
            min_pixels=3_136,
            max_pixels=262_144,
            max_input_tokens=262_144,
            process_vision_info=lambda messages, **kwargs: (
                [f"image-{index}" for index in range(1_800)],
                None,
            ),
        )

        with self.assertRaisesRegex(RuntimeError, "exceeds max_input_tokens"):
            collator([_example(0, JudgeTask.GROUNDEDNESS, Verdict.PASS)])

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

    def test_lora_dtype_audit_rejects_fp32_promotion(self) -> None:
        class Model:
            def __init__(self, dtype):
                self.parameter = torch.nn.Parameter(torch.ones(2, dtype=dtype))

            def named_parameters(self):
                yield "model.layers.0.mlp.up_proj.lora_B.default.weight", self.parameter

        audit = audit_trainable_lora_dtypes(Model(torch.bfloat16), torch.bfloat16)
        self.assertEqual(audit["parameter_counts_by_dtype"], {"torch.bfloat16": 2})
        self.assertFalse(audit["autocast_adapter_dtype"])
        with self.assertRaisesRegex(RuntimeError, "promoted away from BF16"):
            audit_trainable_lora_dtypes(Model(torch.float32), torch.bfloat16)

    def test_only_upper_16_decoder_layers_have_trainable_lora(self) -> None:
        class Model:
            def named_parameters(self):
                for index in range(48, 64):
                    yield (
                        "base_model.model.language_model.layers."
                        f"{index}.mlp.up_proj.lora_A.default.weight",
                        torch.nn.Parameter(torch.ones(2)),
                    )

        audit = audit_trainable_lora_layers(Model(), list(range(48, 64)))
        self.assertEqual(audit["layer_indices"], list(range(48, 64)))
        with self.assertRaisesRegex(RuntimeError, "differ from"):
            audit_trainable_lora_layers(Model(), list(range(47, 64)))

    def test_frozen_prefix_guard_rejects_gradient_bearing_layer_48_input(self) -> None:
        class Stack(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = torch.nn.ModuleList(
                    [torch.nn.Identity() for _ in range(64)]
                )

        model = Stack()
        audit = install_frozen_prefix_input_guard(
            model,
            first_trainable_layer=48,
        )
        self.assertEqual(audit["guarded_module"], "layers.48")
        model.layers[48](torch.ones(1, requires_grad=False))
        with self.assertRaisesRegex(RuntimeError, "leaked into autograd"):
            model.layers[48](torch.ones(1, requires_grad=True))

    def test_nonreentrant_checkpoint_trains_params_without_input_grad(self) -> None:
        layer = torch.nn.Linear(2, 2, bias=False)
        inputs = torch.ones(1, 2, requires_grad=False)
        output = torch.utils.checkpoint.checkpoint(
            layer,
            inputs,
            use_reentrant=False,
        )
        output.sum().backward()
        self.assertIsNotNone(layer.weight.grad)
        self.assertFalse(inputs.requires_grad)

    def test_safe_tp_plan_replicates_only_hybrid_linear_attention(self) -> None:
        plan = {
            "layers.*.self_attn.q_proj": "colwise",
            "layers.*.self_attn.k_proj": "colwise",
            "layers.*.self_attn.v_proj": "colwise",
            "layers.*.self_attn.o_proj": "rowwise",
            "layers.*.mlp.gate_proj": "colwise",
            "layers.*.mlp.up_proj": "colwise",
            "layers.*.mlp.down_proj": "rowwise",
            "layers.*.linear_attn.in_proj_qkv": "colwise_gather_output",
        }
        text_config = SimpleNamespace(base_model_tp_plan=plan)
        config = SimpleNamespace(get_text_config=lambda **kwargs: text_config)
        audit = configure_safe_tensor_parallel_plan(config)
        self.assertNotIn(
            "layers.*.linear_attn.in_proj_qkv",
            text_config.base_model_tp_plan,
        )
        self.assertEqual(
            audit["replicated_for_compatibility"],
            ["layers.*.linear_attn.in_proj_qkv"],
        )

    def test_transformers_516_tp_metadata_is_repaired_after_mesh_audit(self) -> None:
        class Mesh:
            ndim = 1

            @staticmethod
            def size() -> int:
                return 2

        model = SimpleNamespace(
            config=SimpleNamespace(
                distributed_config=SimpleNamespace(tp_size=2),
            ),
            _device_mesh=Mesh(),
            _tp_size=None,
        )
        audit = ensure_tensor_parallel_metadata(model, requested_tp_size=2)
        self.assertEqual(model._tp_size, 2)
        self.assertTrue(audit["metadata_repaired"])
        self.assertEqual(audit["observed_tp_size"], 2)

    def test_tp_metadata_repair_rejects_a_wrong_mesh_size(self) -> None:
        class Mesh:
            ndim = 1

            @staticmethod
            def size() -> int:
                return 1

        model = SimpleNamespace(
            config=SimpleNamespace(
                distributed_config=SimpleNamespace(tp_size=2),
            ),
            _device_mesh=Mesh(),
            _tp_size=None,
        )
        with self.assertRaisesRegex(RuntimeError, "mesh disagrees"):
            ensure_tensor_parallel_metadata(model, requested_tp_size=2)


if __name__ == "__main__":
    unittest.main()
