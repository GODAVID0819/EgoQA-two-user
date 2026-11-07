"""验证分块检查点保留完整MLP输出、输入梯度和参数梯度。"""
import copy
import unittest

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from training.judge_sft.chunked_mlp import chunked_forward


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(7, 13)
        self.up = nn.Linear(7, 13)
        self.down = nn.Linear(13, 7)

    def forward(self, x):
        return self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))


class ChunkedMLPTests(unittest.TestCase):
    def test_nested_checkpoint_outputs_and_all_gradients(self):
        torch.manual_seed(17)
        reference = MLP().double()
        chunked = copy.deepcopy(reference)
        a = torch.randn(2, 37, 7, dtype=torch.float64, requires_grad=True)
        b = a.detach().clone().requires_grad_()
        expected = checkpoint(reference, a, use_reentrant=False)
        actual = checkpoint(lambda x: chunked_forward(chunked, x, chunk_size=8), b, use_reentrant=False)
        torch.testing.assert_close(actual, expected)
        expected.square().mean().backward()
        actual.square().mean().backward()
        torch.testing.assert_close(b.grad, a.grad)
        for x, y in zip(chunked.parameters(), reference.parameters()):
            torch.testing.assert_close(x.grad, y.grad)


if __name__ == "__main__":
    unittest.main()
