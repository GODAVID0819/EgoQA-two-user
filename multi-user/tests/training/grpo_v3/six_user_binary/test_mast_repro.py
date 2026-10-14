"""复现 MAST GRPO 所需的输入与参数开关；默认行为不变。"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from test_launch import run_config
from test_pipeline import packet_fixture, qa_text, require_module

MAST_TARGET_REGEX = (r"^model\.language_model\.layers\.(4[89]|5[0-9]|6[0-3])\."
                     r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|linear_attn\.(in_proj_qkv|in_proj_z|"
                     r"in_proj_a|in_proj_b|out_proj)|mlp\.(gate_proj|up_proj|down_proj))$")


def budget_from_prompt(content):
    prompt = content[content.rfind("<image>") + len("<image>\n"):]
    start = prompt.index('"generator_context_budget"')
    return json.loads("{" + prompt[start:prompt.index("}", start) + 1] + "}")["generator_context_budget"]


class FullQuarterFpsPolicyInputTests(unittest.TestCase):
    def test_full_025fps_row_matches_mast_policy_row(self):
        data = require_module(self, "data")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packet = packet_fixture(root)
            row = data.make_row(root, "DAY1_TEST", 2, generator_view="full_025fps",
                                generation_profile="compact_full_qa_v1")
            self.assertEqual(row["generator_view"], "full_025fps")
            self.assertEqual(row["generator_frame_counts"], [150] * 6)
            self.assertEqual(len(row["images"]), 900)
            self.assertEqual(row["messages"][0]["content"].count("<image>"), 900)
            # 提问者在前，每人完整时间线隔帧（0, 2, 4, ...）。
            order = [2, 0, 1, 3, 4, 5]
            expected = [str((packet / f"u{u}" / f"{i:03d}.jpg").resolve()) for u in order for i in range(0, 300, 2)]
            self.assertEqual(row["images"], expected)
            self.assertEqual(budget_from_prompt(row["messages"][0]["content"]), {
                "policy": "six_full_unpruned_sampled_frame_sets", "analysis_sample_fps": 0.5,
                "source_frame_count": 1800, "model_input_frame_count": 900,
                "per_user_source_frame_counts": [300] * 6,
                "per_user_model_input_frame_counts": [150] * 6})
            content = row["messages"][0]["content"]
            self.assertIn('"effective_model_input_fps": 0.25', content)
            self.assertIn('"is_pruned": false', content)
            self.assertIn("Output brevity requirements", content)
            data.validate_row(row)
            # 去掉视角标记会按原剪枝视角重建，必须被检出。
            row.pop("generator_view")
            with self.assertRaises(ValueError):
                data.validate_row(row)

    def test_reward_plugin_validates_full_025fps_rows(self):
        data = require_module(self, "data")
        plugin = require_module(self, "plugin")

        class Client:
            def score(self, request):
                return {"request_id": request["request_id"], "evidence_id": request["evidence_id"],
                        "status": "scored", "probabilities": {"formality": 1., "groundedness": 1.,
                        "speaker_only": 0., "all_six": 1.}, "judge": {"model_id": "fixture"}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packet_fixture(root)
            row = data.make_row(root, "DAY1_TEST", 2, generator_view="full_025fps",
                                generation_profile="compact_full_qa_v1")
            orm = plugin.SixUserBinaryReward(client=Client(), trace_path=root / "trace.jsonl",
                                             reward_mode="continuous")
            # ms-swift 按列传入整组（同一 prompt 的 2 个候选）元数据。
            kwargs = {k: [row[k]] * 2 for k in ("dataset_root", "source_packet_id", "asker_index", "evidence_id",
                                                "required_users", "generator_image_paths", "generator_view")}
            self.assertEqual(orm(["a", "b"], **kwargs), [1., 1.])
            # 缺少视角列时按默认剪枝视角校验，路径不一致必须报错。
            kwargs.pop("generator_view")
            with self.assertRaisesRegex(ValueError, "generator_image_paths"):
                plugin.SixUserBinaryReward(client=Client(), trace_path=root / "t2.jsonl",
                                           reward_mode="continuous")(["a"], **{k: v[:1] for k, v in kwargs.items()})

    def test_default_view_is_unchanged(self):
        data = require_module(self, "data")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packet_fixture(root)
            row = data.make_row(root, "DAY1_TEST", 2)
            self.assertNotIn("generator_view", row)
            self.assertEqual(row["generator_frame_counts"], [300, 150, 150, 150, 150, 150])
            with self.assertRaisesRegex(ValueError, "未知生成器视角"):
                data.make_row(root, "DAY1_TEST", 2, generator_view="other")


class JudgeFrameStrideTests(unittest.TestCase):
    def test_stride_two_matches_quarter_fps_judge_collator(self):
        judge = require_module(self, "judge")
        predictor = require_module(self, "predictor")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packet_fixture(root)
            examples = judge.build_examples({"request_id": "case1", "dataset_root": str(root),
                "source_packet_id": "DAY1_TEST", "asker_index": 2,
                "evidence_id": "DAY1_TEST__ASKER_A3", "completion": qa_text()})
            grounded = examples["groundedness"]
            frames = predictor.judge_frames(grounded, 2)
            self.assertEqual(len(frames), 900)
            self.assertEqual(frames, [f for s in grounded.frame_sets for f in s.frames[::2]])
            prompt = predictor.judge_prompt(grounded, 2)
            self.assertIn("the next 150 images are frames sampled at 0.25 FPS", prompt)
            self.assertTrue(prompt.endswith(grounded.prompt))
            self.assertEqual(len(predictor.judge_frames(examples["speaker_only"], 2)), 150)
            # 默认 stride=1 保持原 0.5 FPS 全部帧与原可见文本。
            self.assertEqual(predictor.judge_frames(grounded, 1), list(grounded.frames))
            self.assertIn("frames sampled at 0.5 FPS", predictor.judge_prompt(grounded, 1))


class SwiftOptionTests(unittest.TestCase):
    @staticmethod
    def mast_config():
        c = run_config()
        c.update(lora_rank=64, lora_alpha=64, lora_dropout=0, lora_target_regex=MAST_TARGET_REGEX,
                 swift_overrides={"weight_decay": 0, "epsilon": 0.2, "loss_type": "grpo",
                                  "vllm_enable_lora": False}, use_vllm=True)
        return c

    def test_mast_lora_and_optimizer_options_reach_swift(self):
        launch = require_module(self, "launch")
        cmd = launch.swift_command(self.mast_config(), "/scratch/test/outputs/train_123")
        value = lambda flag: cmd[cmd.index(flag) + 1]
        self.assertEqual(value("--target_regex"), MAST_TARGET_REGEX)
        self.assertNotIn("--target_modules", cmd)
        self.assertEqual((value("--lora_rank"), value("--lora_alpha"), value("--lora_dropout")), ("64", "64", "0"))
        self.assertEqual((value("--weight_decay"), value("--epsilon"), value("--loss_type")), ("0", "0.2", "grpo"))
        self.assertEqual(value("--vllm_enable_lora"), "false")
        self.assertEqual(cmd.count("--vllm_enable_lora"), 1)

    def test_defaults_are_unchanged(self):
        launch = require_module(self, "launch")
        cmd = launch.swift_command(run_config(), "/scratch/test/outputs/train_123")
        for flag in ("--target_regex", "--weight_decay", "--epsilon", "--loss_type", "--lora_dropout"):
            self.assertNotIn(flag, cmd)
        self.assertEqual(cmd[cmd.index("--target_modules") + 1:], ["q_proj", "v_proj"])

    def test_invalid_options_are_rejected(self):
        launch = require_module(self, "launch")
        for change, message in (({"swift_overrides": {"learning_rate": 1}}, "只允许"),
                                ({"swift_overrides": {"weight_decay": -1}}, "数值不合法"),
                                ({"swift_overrides": {"vllm_enable_lora": "no"}}, "类型不合法"),
                                ({"lora_target_modules": ["q_proj", "v_proj"]}, "不能与"),
                                ({"lora_dropout": 1.0}, "lora_dropout")):
            c = self.mast_config()
            c.update(change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, message):
                launch.validate_config(c)


if __name__ == "__main__":
    unittest.main()
