"""将指定语言层的梯度检查点输入暂存CPU，保持原重计算与RNG。"""
from functools import wraps


def tensor_key(tensor):
    return (tensor.data_ptr(),tuple(tensor.shape),tuple(tensor.stride()),tensor.dtype,tensor.device)


class InputOffload:
    def __init__(self,model,layers,layer_stride):
        self.selected_layers=list(range(layer_stride-1,len(layers),layer_stride))
        self.stats={'packed_tensors':0,'restored_tensors':0,'offloaded_bytes':0,'wrapper_reinstalls':0}
        self.originals=[]
        for i in self.selected_layers:
            layer=layers[i];original=layer._gradient_checkpointing_func
            self.originals.append((layer,original))
            layer._gradient_checkpointing_func=self.wrap(original)
        self.forward_handle=model.register_forward_pre_hook(self.before_forward)

    def before_forward(self,module,args):
        # Swift旧/参考概率上下文退出时会重新enable，必须在实际forward前恢复包装。
        for index,(layer,previous) in enumerate(self.originals):
            current=layer._gradient_checkpointing_func
            if getattr(current,'_egoqa_input_offload_owner',None) is self:continue
            self.originals[index]=(layer,current)
            layer._gradient_checkpointing_func=self.wrap(current)
            self.stats['wrapper_reinstalls']+=1

    def wrap(self,original):
        import torch
        @wraps(original)
        def checkpoint(function,*args,**kwargs):
            if not args or not isinstance(args[0],torch.Tensor) or not args[0].is_floating_point():
                raise ValueError('所选语言层检查点的第一个输入必须是hidden_states张量')
            # hook闭包只保留指针与元数据，不能捕获GPU输入本身。
            key=tensor_key(args[0])
            def pack(tensor):
                if tensor_key(tensor)==key:
                    cpu=tensor.detach().to('cpu',copy=True)
                    self.stats['packed_tensors']+=1
                    self.stats['offloaded_bytes']+=tensor.numel()*tensor.element_size()
                    return (tensor.device,cpu)
                return tensor.detach()
            def unpack(packed):
                if isinstance(packed,tuple):
                    device,cpu=packed;self.stats['restored_tensors']+=1
                    return cpu.to(device=device,non_blocking=False)
                return packed
            with torch.autograd.graph.saved_tensors_hooks(pack,unpack):
                return original(function,*args,**kwargs)
        checkpoint._egoqa_input_offload_owner=self
        return checkpoint

    def close(self):
        self.forward_handle.remove()
        for layer,original in self.originals:
            if getattr(layer._gradient_checkpointing_func,'_egoqa_input_offload_owner',None) is self:
                layer._gradient_checkpointing_func=original
        self.originals.clear()


def install(model,*,expected_layer_count=64,layer_stride=4):
    if type(layer_stride) is not int or layer_stride not in (1,4):
        raise ValueError('只支持已验证的全部层或每四层一个输入转存')
    layers=[module for module in model.modules() if type(module).__name__=='Qwen3_5DecoderLayer']
    if len(layers)!=expected_layer_count or any(not callable(getattr(layer,'_gradient_checkpointing_func',None)) for layer in layers):
        raise ValueError('语言层数量或已启用的原生梯度检查点与预期不符')
    return InputOffload(model,layers,layer_stride)
