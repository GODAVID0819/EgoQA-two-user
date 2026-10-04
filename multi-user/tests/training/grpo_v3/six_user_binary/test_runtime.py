"""评分服务和 GRPO 候选对齐的回归测试。"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from test_pipeline import require_module


class RuntimeTests(unittest.TestCase):
    def test_existing_judge_on_same_port_cannot_be_mistaken_for_new_instance(self):
        service = require_module(self, "service")
        client = service.JudgeClient("http://127.0.0.1:1", expected_instance="this-job")
        with patch.object(client, "_request", return_value={"status": "ready", "frozen": True, "instance_id": "older-job"}):
            with self.assertRaisesRegex(RuntimeError, "instance"):
                client.health()
        reply = {"request_id": "r", "evidence_id": "p", "status": "invalid_completion", "reason": "bad JSON",
                 "judge": {"instance_id": "older-job"}}
        with patch.object(client, "_request", return_value=reply):
            with self.assertRaisesRegex(RuntimeError, "instance"):
                client.score({"request_id": "r", "evidence_id": "p"})
    def test_four_probabilities_reject_nonfinite_and_keep_speaker_direction(self):
        reward = require_module(self, "reward")
        good = {"formality": 1., "groundedness": 1., "speaker_only": 0., "all_six": 1.}
        self.assertAlmostEqual(reward.aggregate(good, mode="continuous"), 1.)
        self.assertAlmostEqual(reward.aggregate({**good, "speaker_only": 1.}, mode="continuous"), .6)
        soft = {"formality": .9, "groundedness": .8, "speaker_only": .2, "all_six": .75}
        self.assertAlmostEqual(reward.aggregate(soft, mode="continuous"), .74)
        self.assertAlmostEqual(reward.aggregate(soft, mode="binary"), 1.)
        with self.assertRaises(ValueError):
            reward.aggregate({**good, "groundedness": float("nan")}, mode="continuous")

    def test_http_client_checks_request_identity_and_exposes_service_failure(self):
        service = require_module(self, "service")
        class Scorer:
            def health(self):
                return {"status": "ready", "model_id": "fixture", "frozen": True}
            def score(self, request):
                if request["request_id"] == "broken":
                    raise RuntimeError("CUDA fixture failure")
                return {"request_id": request["request_id"], "evidence_id": request["evidence_id"],
                        "status": "scored", "probabilities": {"formality": .8, "groundedness": .8,
                        "speaker_only": .1, "all_six": .9}, "judge": self.health()}
        server = service.make_server(Scorer(), port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = service.JudgeClient(f"http://127.0.0.1:{server.server_port}")
            self.assertTrue(client.health()["frozen"])
            result = client.score({"request_id": "one", "evidence_id": "P"})
            self.assertEqual(result["probabilities"]["speaker_only"], .1)
            with self.assertLogs(level="ERROR"), self.assertRaises(RuntimeError):
                client.score({"request_id": "broken", "evidence_id": "P"})
            with self.assertRaisesRegex(RuntimeError, "identity"):
                service.validate_response({**result, "request_id": "other"}, {"request_id": "one", "evidence_id": "P"})
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_plugin_preserves_batched_row_identity_and_writes_raw_completion(self):
        plugin = require_module(self, "plugin")
        seen = []
        class Client:
            def score(self, request):
                seen.append(request)
                return {"request_id": request["request_id"], "evidence_id": request["evidence_id"],
                    "status": "scored", "probabilities": {"formality": 1., "groundedness": 1.,
                    "speaker_only": 0., "all_six": 1.}, "judge": {"model_id": "fixture"}}
        with tempfile.TemporaryDirectory() as tmp:
            with patch(MODULE + ".plugin.validate_row") as check:
                orm = plugin.SixUserBinaryReward(client=Client(), trace_path=Path(tmp)/"trace.jsonl", reward_mode="continuous")
                values = orm(["a", "b"], dataset_root=[tmp, tmp], source_packet_id=["P1", "P2"],
                    asker_index=[0, 1], evidence_id=["E1", "E2"], required_users=[["A"], ["B"]],
                    generator_image_paths=[["a.jpg"], ["b.jpg"]])
                self.assertEqual(check.call_count, 2)
            self.assertEqual(values, [1., 1.])
            self.assertEqual([r["evidence_id"] for r in seen], ["E1", "E2"])
            self.assertNotEqual(seen[0]["request_id"], seen[1]["request_id"])
            rows = [json.loads(x) for x in (Path(tmp)/"trace.jsonl").read_text().splitlines()]
            self.assertEqual([r["completion"] for r in rows], ["a", "b"])
            self.assertEqual(rows[1]["source_packet_id"], "P2")

    def test_group_size_mismatch_is_not_silently_repeated(self):
        plugin = require_module(self, "plugin")
        with self.assertRaises(ValueError):
            plugin.expand(["P1", "P2"], 4, "packet")


MODULE = "training.grpo_v3.six_user_binary"
if __name__ == "__main__":
    unittest.main()
