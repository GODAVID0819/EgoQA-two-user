"""验证Qwen线性注意力实际选择FLA，并可在加载大模型前检查小张量反向。"""
import argparse
import inspect
import json
from importlib.metadata import version
from pathlib import Path


def validate_linear_backend(cuda_device=None):
    import torch
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from transformers.models.qwen3_5 import modeling_qwen3_5

    dispatch = modeling_qwen3_5.torch_chunk_gated_delta_rule
    visited = set()

    def visit(function):
        if id(function) in visited:
            return
        visited.add(id(function))
        if inspect.isfunction(function):
            for value in inspect.getclosurevars(function).nonlocals.values():
                if callable(value):
                    visit(value)

    visit(dispatch)
    if id(chunk_gated_delta_rule) not in visited:
        raise RuntimeError("Qwen gated delta rule is using its high-memory Torch fallback")
    result = {"backend": "fla", "fla_core": version("fla-core"),
              "dispatcher_verified": True, "cuda_checked": False}
    if cuda_device is None:
        return result
    torch.cuda.set_device(cuda_device)
    with torch.random.fork_rng(devices=[cuda_device]):
        torch.manual_seed(17)
        originals = [torch.randn(1, 64, 4, 128, device=cuda_device, dtype=torch.bfloat16) for _ in range(3)]
        originals += [-torch.rand(1, 64, 4, device=cuda_device), torch.rand(1, 64, 4, device=cuda_device)]
        fast_inputs = [x.detach().clone().requires_grad_() for x in originals]
        reference_inputs = [x.detach().clone().requires_grad_() for x in originals]
        kwargs = dict(output_final_state=False, use_qk_l2norm_in_kernel=True)
        fast = dispatch(*fast_inputs, **kwargs)[0]
        reference = inspect.unwrap(dispatch)(*reference_inputs, **kwargs)[0]
        torch.testing.assert_close(fast, reference, rtol=.05, atol=.02)
        fast.float().square().mean().backward()
        reference.float().square().mean().backward()
        for actual, expected in zip(fast_inputs, reference_inputs):
            if actual.grad is None or not torch.isfinite(actual.grad).all():
                raise RuntimeError("FLA backward did not produce finite gradients")
            torch.testing.assert_close(actual.grad, expected.grad, rtol=.15, atol=1e-4)
        result["cuda_checked"] = True
        result["forward_max_abs_difference"] = float((fast-reference).abs().max())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate_linear_backend()
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
