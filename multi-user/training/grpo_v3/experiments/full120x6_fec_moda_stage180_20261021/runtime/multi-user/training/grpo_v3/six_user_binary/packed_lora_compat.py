"""仅兼容Qwen融合QKV投影有LoRA而z投影没有LoRA的vLLM加载。"""
from functools import wraps


def wrap_expand(original):
    @wraps(original)
    def expand(self, lora_a, lora_b):
        partial_qkv = (self.n_slices == 4 and len(lora_a) == 2 and len(lora_b) == 2
                       and lora_a[0] is not None and lora_b[0] is not None
                       and lora_a[1] is None and lora_b[1] is None)
        if not partial_qkv:
            return original(self, lora_a, lora_b)
        if lora_b[0].shape[0] != sum(self.output_sizes[:3]):
            raise ValueError('Qwen部分QKV LoRA的B行数与Q/K/V三个切片不一致')
        a, b = original(self, lora_a[:1], lora_b[:1])
        if len(a) != 3 or len(b) != 3:
            raise ValueError('Qwen部分QKV LoRA没有展开为三个切片')
        return a + [None], b + [None]
    return expand


def install():
    from vllm.lora.layers.column_parallel_linear import MergedColumnParallelLinearWithLoRA
    cls = MergedColumnParallelLinearWithLoRA
    if getattr(cls.expand_packed_lora, '_egoqa_partial_qkv_fix', False):
        return
    cls.expand_packed_lora = wrap_expand(cls.expand_packed_lora)
    cls.expand_packed_lora._egoqa_partial_qkv_fix = True
    print('EGOQA_PARTIAL_QKV_LORA_COMPAT installed, z remains unadapted', flush=True)
