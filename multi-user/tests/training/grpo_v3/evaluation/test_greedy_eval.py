from __future__ import annotations

import math
import unittest

from training.grpo_v3.evaluation.greedy_eval import evaluate_rows


class _Runner:
    model_id = "policy"

    def __init__(self) -> None:
        self.calls = []

    def generate(self, prompt, image_paths=None, video_paths=None, decoding_mode="greedy"):
        self.calls.append(
            {"prompt": prompt, "image_paths": image_paths, "video_paths": video_paths, "decoding_mode": decoding_mode}
        )
        return '{"question":"q"}'


def _row(evidence_id: str = "E1") -> dict:
    return {
        "messages": [{"role": "user", "content": "<video><video>\n原 repo prompt"}],
        "videos": ["/v/u1.mp4", "/v/u2.mp4"],
        "evidence_id": evidence_id,
        "packet_json": '{"evidence_id":"%s","required_users":["u1","u2"]}' % evidence_id,
        "question_type": "commonality",
        "generation_mode": "baseline",
    }


class GreedyEvalTests(unittest.TestCase):
    def test_one_shot_greedy_generation_and_full_record_output(self) -> None:
        runner = _Runner()
        score_calls = []

        def scorer(**kwargs):
            score_calls.append(kwargs)
            return {
                "reward": 1.25,
                "record": {"reward_total": 1.25, "reward_components": {"groundedness": 1.5}},
            }

        results = evaluate_rows([_row()], runner=runner, score_fn=scorer, adapter_label="gate2")
        self.assertEqual(len(results), 1)
        self.assertEqual(len(runner.calls), 1)
        self.assertEqual(runner.calls[0]["decoding_mode"], "greedy")
        self.assertEqual(runner.calls[0]["video_paths"], ["/v/u1.mp4", "/v/u2.mp4"])
        self.assertEqual(len(score_calls), 1)
        self.assertEqual(results[0]["reward"], 1.25)
        self.assertEqual(results[0]["adapter_label"], "gate2")
        self.assertEqual(results[0]["decode_config"], {"mode": "greedy", "do_sample": False})
        self.assertEqual(results[0]["record"]["reward_components"]["groundedness"], 1.5)

    def test_masked_or_nonfinite_reward_aborts_evaluation(self) -> None:
        runner = _Runner()
        with self.assertRaisesRegex(RuntimeError, "masked"):
            evaluate_rows(
                [_row()], runner=runner,
                score_fn=lambda **kwargs: {"reward": None, "record": {"masked": True}},
                adapter_label="gate2",
            )
        with self.assertRaisesRegex(ValueError, "有限"):
            evaluate_rows(
                [_row()], runner=runner,
                score_fn=lambda **kwargs: {"reward": math.inf, "record": {}},
                adapter_label="gate2",
            )

    def test_duplicate_eval_key_aborts_before_generation(self) -> None:
        runner = _Runner()
        with self.assertRaisesRegex(ValueError, "重复"):
            evaluate_rows(
                [_row(), _row()], runner=runner,
                score_fn=lambda **kwargs: {"reward": 0.0, "record": {}},
                adapter_label="gate2",
            )
        self.assertEqual(runner.calls, [])


if __name__ == "__main__":
    unittest.main()
