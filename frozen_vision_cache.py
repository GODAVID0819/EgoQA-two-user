"""默认关闭的冻结视觉forward单组CPU缓存；不缓存语言模型/KV或可训练特征。

显式install后仍需set_key(有序图路径tuple)才会启用。预算为保留的CPU输入及输出
张量总字节；精确核对输入的临时D2H传输和恢复输出仍有额外开销，须实际测量。
"""
from dataclasses import fields, is_dataclass
import functools
import random

_active = None


class _Unsafe(Exception):
    pass


def _tree_bytes(value, torch):
    if type(value) is torch.Tensor:
        if value.requires_grad or value.layout != torch.strided:
            raise _Unsafe('requires_grad_or_non_strided_tensor')
        return value.numel()*value.element_size()
    if value is None or type(value) in (bool,int,float,str):
        return 0
    if type(value) in (list,tuple,dict) or _named_tuple(value):
        items = value.values() if type(value) is dict else value
        if type(value) is dict and not all(type(key) is str for key in value):
            raise _Unsafe('opaque_mapping_key')
        return sum(_tree_bytes(item,torch) for item in items)
    if is_dataclass(value) and not isinstance(value,type) and all(f.init for f in fields(value)):
        return sum(_tree_bytes(getattr(value,f.name),torch) for f in fields(value))
    raise _Unsafe('opaque_input_or_output_type')


def _named_tuple(value):
    return isinstance(value,tuple) and isinstance(getattr(type(value),'_fields',None),tuple)


def _pack(value, torch, *, output=False):
    if type(value) is torch.Tensor:
        # 非连续输出可能依赖其stride；无法无损重建时直接回退。
        if output and not value.is_contiguous():
            raise _Unsafe('non_contiguous_output')
        return ('tensor', value.detach().to('cpu',copy=True),
                value.device, tuple(value.shape), tuple(value.stride()))
    if value is None or type(value) in (bool,int,float,str):
        return ('scalar',value)
    if type(value) is dict:
        return ('dict',tuple((key,_pack(item,torch,output=output)) for key,item in value.items()))
    if is_dataclass(value) and not isinstance(value,type):
        return ('dataclass',type(value),tuple((f.name,_pack(getattr(value,f.name),torch,output=output))
                                            for f in fields(value)))
    kind = 'namedtuple' if _named_tuple(value) else ('list' if type(value) is list else 'tuple')
    return (kind,type(value),tuple(_pack(item,torch,output=output) for item in value))


def _matches(value, packed, torch):
    kind = packed[0]
    if kind == 'tensor':
        _,cpu,device,shape,stride = packed
        return (type(value) is torch.Tensor and not value.requires_grad
            and value.layout == torch.strided and value.device == device
            and value.dtype == cpu.dtype and tuple(value.shape) == shape
            and tuple(value.stride()) == stride and torch.equal(value.detach().to('cpu'),cpu))
    if kind == 'scalar':
        return type(value) is type(packed[1]) and value == packed[1]
    if kind == 'dict':
        return (type(value) is dict and tuple(value) == tuple(key for key,_ in packed[1])
            and all(_matches(value[key],item,torch) for key,item in packed[1]))
    if kind == 'dataclass':
        return type(value) is packed[1] and all(_matches(getattr(value,key),item,torch)
                                               for key,item in packed[2])
    return (type(value) is packed[1] and len(value) == len(packed[2])
            and all(_matches(item,saved,torch) for item,saved in zip(value,packed[2])))


def _restore(packed, torch):
    kind = packed[0]
    if kind == 'tensor':
        # copy=True保证调用方改写CPU或GPU输出时不会污染保留缓存。
        return packed[1].to(device=packed[2],copy=True)
    if kind == 'scalar':
        return packed[1]
    if kind == 'dict':
        return {key:_restore(item,torch) for key,item in packed[1]}
    if kind == 'dataclass':
        return packed[1](**{key:_restore(item,torch) for key,item in packed[2]})
    values = [_restore(item,torch) for item in packed[2]]
    if kind == 'namedtuple':
        return packed[1](*values)
    return values if kind == 'list' else tuple(values)


class VisionCache:
    def __init__(self, visual, torch, max_bytes):
        self.visual, self.torch = visual, torch
        self.max_bytes = max_bytes
        self.enabled = visual is not None
        self.reason = None if self.enabled else 'visual_module_not_unique'
        self.key = None
        self.entry = None
        self.original = None
        self.stats = {'hits':0,'misses':0,'bypasses':0,'bytes':0}
        if self.enabled:
            try:
                self._state()
            except _Unsafe as exc:
                self.enabled, self.reason = False, str(exc)
        if self.enabled:
            self.original = visual.forward
            @functools.wraps(self.original)
            def cached_forward(*args,**kwargs):
                return self._forward(args,kwargs)
            visual.forward = cached_forward

    def clear(self):
        self.entry = None
        self.stats['bytes'] = 0

    def set_key(self, key):
        valid = type(key) is tuple and len(key)>0 and all(type(path) is str and path for path in key)
        key = key if valid else None
        if key != self.key:
            self.clear()
            self.key = key
        return bool(self.enabled and self.key is not None)

    def close(self):
        if self.original is not None:
            self.visual.forward = self.original
            self.original = None
        self.enabled = False
        self.clear()

    def _state(self):
        torch = self.torch
        state = []
        for module in self.visual.modules():
            if isinstance(module,torch.nn.modules.dropout._DropoutNd) and module.p > 0:
                raise _Unsafe('stochastic_dropout')
            if module.training and isinstance(module,torch.nn.modules.batchnorm._BatchNorm):
                raise _Unsafe('mutable_batchnorm')
            for name in ('dropout','attention_dropout','hidden_dropout_prob','drop_path_rate'):
                probability = getattr(module,name,None)
                if isinstance(probability,(int,float)) and probability > 0:
                    raise _Unsafe('stochastic_dropout')
            state.append((id(module),module.training))
        for tensor in list(self.visual.parameters())+list(self.visual.buffers()):
            if tensor.requires_grad:
                raise _Unsafe('visual_not_fully_frozen')
            try:
                state.append((id(tensor),tensor._version,tensor.device,tensor.dtype,tuple(tensor.shape)))
            except RuntimeError as exc:
                raise _Unsafe('opaque_tensor_version') from exc
        # autocast会改变冻结视觉输出，必须参与缓存有效性判断。
        for device in ('cpu','cuda'):
            state.append((device,torch.is_autocast_enabled(device),torch.get_autocast_dtype(device)))
        return tuple(state)

    def _rng(self):
        torch = self.torch
        devices = sorted({tensor.device.index for tensor in self.visual.parameters() if tensor.is_cuda})
        return (random.getstate(),torch.get_rng_state(),
                tuple(torch.cuda.get_rng_state(device) for device in devices))

    def _rng_equal(self,left,right):
        return (left[0] == right[0] and self.torch.equal(left[1],right[1])
            and all(self.torch.equal(a,b) for a,b in zip(left[2],right[2])))

    def _forward(self,args,kwargs):
        if not self.enabled or self.key is None:
            return self.original(*args,**kwargs)
        torch = self.torch
        inputs = (args,kwargs)
        try:
            state = self._state()
            input_bytes = _tree_bytes(inputs,torch)
            if input_bytes > self.max_bytes:
                raise _Unsafe('input_exceeds_budget')
            if self.entry is not None and state == self.entry[0] and _matches(inputs,self.entry[1],torch):
                result = _restore(self.entry[2],torch)
                self.stats['hits'] += 1
                return result
            self.clear()
            saved_inputs = _pack(inputs,torch)
            before = self._rng()
        except (Exception,MemoryError) as exc:
            self.clear()
            self.stats['bypasses'] += 1
            self.reason = str(exc)
            return self.original(*args,**kwargs)
        self.stats['misses'] += 1
        result = self.original(*args,**kwargs)
        try:
            if not self._rng_equal(before,self._rng()):
                self.enabled = False
                raise _Unsafe('forward_consumed_rng')
            if state != self._state() or not _matches(inputs,saved_inputs,torch):
                raise _Unsafe('forward_mutated_inputs_or_state')
            output_bytes = _tree_bytes(result,torch)
            if input_bytes+output_bytes > self.max_bytes:
                raise _Unsafe('inputs_and_outputs_exceed_budget')
            saved_output = _pack(result,torch,output=True)
            self.entry = (state,saved_inputs,saved_output)
            self.stats['bytes'] = input_bytes+output_bytes
            self.reason = None
        except (Exception,MemoryError) as exc:
            self.clear()
            self.stats['bypasses'] += 1
            self.reason = str(exc)
        return result


def install(model, max_bytes=2*1024**3):
    """安装不改变forward梯度的可选缓存；未设有效key时完全透明。"""
    global _active
    import torch
    if isinstance(max_bytes,bool) or not isinstance(max_bytes,int) or max_bytes <= 0:
        raise ValueError('max_bytes必须是正整数')
    if _active is not None:
        _active.close()
    visuals = [module for name,module in model.named_modules() if name == 'visual' or name.endswith('.visual')]
    _active = VisionCache(visuals[0] if len(visuals)==1 else None,torch,max_bytes)
    return _active


def set_key(key):
    """每rank每个完整G4组调用；有序路径变化或None立即清空。"""
    return _active.set_key(key) if _active is not None else False
