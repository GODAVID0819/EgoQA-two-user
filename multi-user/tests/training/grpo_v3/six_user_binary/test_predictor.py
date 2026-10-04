"""不加载 GPU 的二分类输出与 adapter 配置检查。"""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from test_pipeline import require_module


class PredictorTests(unittest.TestCase):
    def test_explicit_baseline_never_loads_an_adapter(self):
        predictor = require_module(self, "predictor")
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "Qwen3.8-27B"
            base.mkdir()
            (base / "config.json").write_text('{}')
            config = {"model_id": str(base), "judge_mode": "baseline"}
            checked = predictor.validate_config(config)
            self.assertEqual(checked["adapters"], {})
            self.assertEqual(checked["judge_mode"], "baseline")
            engine = predictor.engine_options(checked)
            self.assertFalse(engine["enable_lora"])
            self.assertNotIn("max_lora_rank", engine)
            with self.assertRaisesRegex(ValueError, "baseline"):
                predictor.validate_config({**config, "adapters": {"formality": "unused"}})

    def test_reads_both_requested_raw_logprobs_without_top_token_fallback(self):
        predictor = require_module(self, "predictor")
        output = SimpleNamespace(outputs=[SimpleNamespace(logprobs=[{4: SimpleNamespace(logprob=-2.), 5: SimpleNamespace(logprob=-3.)}])])
        decision = predictor.binary_decision(output, pass_id=4, fail_id=5)
        self.assertAlmostEqual(decision["pass_probability"], .7310585786300049)
        with self.assertRaises(RuntimeError):
            predictor.binary_decision(output, pass_id=6, fail_id=5)

    def test_requires_adapter_for_every_task_and_rejects_wrong_base(self):
        predictor = require_module(self, "predictor")
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "Qwen3.8-27B"
            base.mkdir()
            (base / "config.json").write_text('{}')
            adapter = Path(tmp) / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "Qwen/Qwen3.8-27B", "r": 16}))
            (adapter / "adapter_model.safetensors").write_bytes(b"fixture")
            config = {"model_id": str(base), "adapters": {name: str(adapter) for name in ("formality", "groundedness", "answerability")}}
            self.assertEqual(predictor.validate_config(config)["max_lora_rank"], 16)
            config["adapters"].pop("groundedness")
            with self.assertRaises(ValueError):
                predictor.validate_config(config)
            config["adapters"]["groundedness"] = str(adapter)
            (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "Qwen/Qwen3-VL-8B", "r": 16}))
            with self.assertRaisesRegex(ValueError, "base"):
                predictor.validate_config(config)


if __name__ == "__main__":
    unittest.main()
