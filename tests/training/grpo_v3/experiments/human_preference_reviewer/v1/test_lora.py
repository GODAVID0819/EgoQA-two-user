from __future__ import annotations

import unittest

from training.grpo_v3.experiments.human_preference_reviewer.v1.lora import (
    audit_trainable_parameter_names,
    expected_lora_targets,
    locate_shared_language_layers,
)


class Linear:
    pass


class Attention:
    def __init__(self) -> None:
        self.q_proj = Linear()
        self.k_proj = Linear()
        self.v_proj = Linear()
        self.o_proj = Linear()


class Layer:
    def __init__(self) -> None:
        self.self_attn = Attention()


class LanguageModel:
    def __init__(self, count: int) -> None:
        self.layers = [Layer() for _ in range(count)]


class Inner:
    def __init__(self, count: int) -> None:
        self.language_model = LanguageModel(count)


class FakeModel:
    def __init__(self, count: int) -> None:
        self.model = Inner(count)
        self.visual = object()


class LoraPlacementTests(unittest.TestCase):
    def test_locates_official_shared_stack_and_last_two_qv_targets(self) -> None:
        model = FakeModel(36)

        path, layers = locate_shared_language_layers(model)
        targets = expected_lora_targets(model, last_n=2, projections=("q_proj", "v_proj"))

        self.assertEqual(path, "model.language_model.layers")
        self.assertEqual(len(layers), 36)
        self.assertEqual(targets, (
            "model.language_model.layers.34.self_attn.q_proj",
            "model.language_model.layers.34.self_attn.v_proj",
            "model.language_model.layers.35.self_attn.q_proj",
            "model.language_model.layers.35.self_attn.v_proj",
        ))

    def test_missing_projection_fails_instead_of_broad_matching(self) -> None:
        model = FakeModel(4)
        del model.model.language_model.layers[3].self_attn.v_proj
        with self.assertRaisesRegex(ValueError, "v_proj"):
            expected_lora_targets(model, last_n=2, projections=("q_proj", "v_proj"))

    def test_rejects_non_qwen3_vl_8b_layer_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "36"):
            expected_lora_targets(
                FakeModel(32), last_n=2, projections=("q_proj", "v_proj"), expected_layer_count=36
            )

    def test_trainable_audit_allows_only_heads_and_expected_lora(self) -> None:
        names = (
            "evidence_head.weight", "answerability_head.bias", "formality_head.weight",
            "backbone.model.language_model.layers.34.self_attn.q_proj.lora_A.default.weight",
            "backbone.model.language_model.layers.35.self_attn.v_proj.lora_B.default.weight",
        )
        result = audit_trainable_parameter_names(names, expected_layer_indices=(34, 35))
        self.assertEqual(result["unexpected_trainable_names"], [])

        with self.assertRaisesRegex(ValueError, "unexpected trainable"):
            audit_trainable_parameter_names(names + ("backbone.visual.weight",), expected_layer_indices=(34, 35))

    def test_stage0_audit_allows_only_evidence_head_and_no_lora(self) -> None:
        result = audit_trainable_parameter_names(
            ("evidence_head.weight", "evidence_head.bias"),
            expected_layer_indices=(),
            active_heads=("evidence_quality",),
            lora_enabled=False,
        )
        self.assertEqual(result["lora_parameter_names"], [])
        for forbidden in ("answerability_head.weight", "backbone.x.lora_A.default.weight"):
            with self.subTest(forbidden=forbidden), self.assertRaisesRegex(ValueError, "unexpected trainable"):
                audit_trainable_parameter_names(
                    ("evidence_head.weight", forbidden),
                    expected_layer_indices=(),
                    active_heads=("evidence_quality",),
                    lora_enabled=False,
                )


if __name__ == "__main__":
    unittest.main()
