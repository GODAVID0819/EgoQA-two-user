"""分块并检查点化逐token MLP，降低长序列反向的中间激活峰值。"""
import functools


def chunked_forward(forward, hidden_states, chunk_size=2048):
    import torch
    from torch.utils.checkpoint import checkpoint

    if hidden_states.shape[-2] <= chunk_size:
        return forward(hidden_states)
    outputs = []
    for chunk in hidden_states.split(chunk_size, dim=-2):
        if torch.is_grad_enabled() and chunk.requires_grad:
            output = checkpoint(forward, chunk, use_reentrant=False)
        else:
            output = forward(chunk)
        outputs.append(output)
    return torch.cat(outputs, dim=-2)


def install_chunked_mlp(model, chunk_size=2048):
    patched = []
    for name, module in model.named_modules():
        if type(module).__name__ != "Qwen3_5MLP":
            continue
        original = module.forward
        module.forward = functools.partial(chunked_forward, original, chunk_size=chunk_size)
        patched.append(name)
    if not patched:
        raise RuntimeError("No Qwen3_5MLP modules were found for activation chunking")
    return {"chunk_size": chunk_size, "modules": patched}
