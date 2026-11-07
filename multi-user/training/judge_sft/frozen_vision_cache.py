"""推理时按完整帧序列缓存冻结视觉编码器输出，缓存保存在CPU。"""
from collections import OrderedDict


def map_tensors(value, transform):
    import torch
    if isinstance(value, torch.Tensor):
        return transform(value)
    if isinstance(value, dict):
        mapped = {key: map_tensors(item, transform) for key, item in value.items()}
        return mapped if type(value) is dict else type(value)(**mapped)
    if isinstance(value, tuple):
        return tuple(map_tensors(item, transform) for item in value)
    if isinstance(value, list):
        return [map_tensors(item, transform) for item in value]
    return value


class FrozenVisionCache:
    def __init__(self, model, max_bytes=16*1024**3):
        candidates = [m for m in model.modules() if type(m).__name__ == "Qwen3_5VisionModel"]
        if len(candidates) != 1:
            raise ValueError("expected one Qwen vision encoder")
        self.module = candidates[0]
        if any(p.requires_grad for p in self.module.parameters()):
            raise ValueError("vision cache requires a frozen encoder")
        self.original = self.module.forward
        self.cache = OrderedDict()
        self.max_bytes = max_bytes
        self.bytes = 0
        self.key = None
        self.hits = 0
        self.misses = 0
        self.module.forward = self.forward

    def forward(self, *args, **kwargs):
        import torch
        if self.module.training or torch.is_grad_enabled() or self.key is None:
            return self.original(*args, **kwargs)
        device = next(self.module.parameters()).device
        if self.key in self.cache:
            self.hits += 1
            self.cache.move_to_end(self.key)
            return map_tensors(self.cache[self.key][0], lambda t: t.to(device))
        self.misses += 1
        output = self.original(*args, **kwargs)
        size = 0
        def copy_tensor(tensor):
            nonlocal size
            size += tensor.numel()*tensor.element_size()
            return tensor.detach().to("cpu", copy=True)
        cached = map_tensors(output, copy_tensor)
        if size <= self.max_bytes:
            while self.cache and self.bytes+size > self.max_bytes:
                _, (_, old_size) = self.cache.popitem(last=False)
                self.bytes -= old_size
            self.cache[self.key] = (cached, size)
            self.bytes += size
        return output
