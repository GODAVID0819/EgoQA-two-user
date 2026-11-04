"""vLLM导入worker扩展时，在实际GPU引擎进程安装部分融合LoRA修复。"""
from .packed_lora_compat import install

install()


class PartialPackedLoRAWorkerExtension:
    def egoqa_partial_packed_lora_compat_ready(self):
        from vllm.lora.layers.column_parallel_linear import MergedColumnParallelLinearWithLoRA
        return bool(getattr(MergedColumnParallelLinearWithLoRA.expand_packed_lora,
                            '_egoqa_partial_qkv_fix', False))
