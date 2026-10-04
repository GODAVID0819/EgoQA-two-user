"""已有同步成片应直接复用，不重新走下载和重建媒体路径。"""
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from training.judge_sft.prepare_real_data import _repo_module


class MaterializedInputTests(unittest.TestCase):
    def test_materialized_packet_routes_to_existing_sampler_without_downloading(self):
        prep = _repo_module("rlhf_evidence_preprocessing")
        self.assertIn("materialized_packet", inspect.signature(prep.prepare_packet).parameters)
        class SamplingReached(Exception):
            pass
        clips = [{"agent_dir": f"A{i+1}_USER", "agent_name": f"User{i+1}", "local_video": f"/{i}.mp4"} for i in range(6)]
        packet = {"day": "DAY1", "time_token": "12000000", "clips": clips}
        config = prep.build_preprocessing_config()
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(prep, "build_evidence_packet") as rebuild, patch.object(prep, "group_clip_frames", side_effect=SamplingReached) as sample:
                with self.assertRaises(SamplingReached):
                    prep.prepare_packet(packet, dataset_root=Path(tmp)/"data", cache_dir=Path(tmp)/"cache",
                        config=config, config_fingerprint=prep.fingerprint(config), materialized_packet=packet)
                rebuild.assert_not_called()
                routed = sample.call_args.args[0]["clips"]
                self.assertEqual([c["agent_dir"] for c in routed], [c["agent_dir"] for c in clips])
                self.assertEqual(routed[0]["source_segments"], [{"segment_index": 0,
                    "local_video": "/0.mp4", "window_start_seconds": 0., "window_end_seconds": 600.}])
                self.assertNotIn("source_segments", clips[0])


if __name__ == "__main__":
    unittest.main()
