import unittest
import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPooling
from training.judge_sft.frozen_vision_cache import FrozenVisionCache


class Qwen3_5VisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.), requires_grad=False)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return BaseModelOutputWithPooling(last_hidden_state=x*self.scale, pooler_output=x.mean(0))


class VisionCacheTests(unittest.TestCase):
    def test_exact_key_reuse_and_distinct_window_order(self):
        model = Qwen3_5VisionModel().eval()
        cache = FrozenVisionCache(model)
        with torch.inference_mode():
            cache.key = ("W01", "W02")
            a = model(torch.arange(8.).reshape(2,4))
            b = model(torch.arange(8.).reshape(2,4))
            torch.testing.assert_close(a.last_hidden_state, b.last_hidden_state)
            self.assertEqual(model.calls, 1)
            cache.key = ("W02", "W01")
            model(torch.ones(2,4))
            self.assertEqual(model.calls, 2)


if __name__ == "__main__":
    unittest.main()
