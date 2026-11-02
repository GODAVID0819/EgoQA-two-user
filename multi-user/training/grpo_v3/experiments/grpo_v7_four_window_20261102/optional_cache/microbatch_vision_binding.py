"""按 Swift 的真实单样本微批绑定既有冻结视觉缓存；不改变模型输入或采样。

仅支持已核对的非动态、非序列并行、每卡微批为一的正式路径。
编码字典只以对象身份关联，不添加可传入模型的额外字段；未知路径透明回退。
"""
from functools import wraps


def _binding(batch):
    if not isinstance(batch,(list,tuple)) or len(batch)!=1 or not isinstance(batch[0],dict):
        return None
    row=batch[0]
    packet=row.get('source_packet_id')
    asker=row.get('asker_index')
    paths=row.get('generator_image_paths') or row.get('images')
    if (not isinstance(packet,str) or not packet or type(asker) is not int
            or not isinstance(paths,(list,tuple)) or not paths
            or not all(isinstance(p,str) and p for p in paths)):
        return None
    return packet,asker,tuple(paths)


def install(trainer,controller,record=None):
    """只包装当前 trainer 实例；保留此前安装的计时、保护器和其他包装。"""
    if getattr(trainer,'_egoqa_microbatch_binding',None) is not None:
        return trainer._egoqa_microbatch_binding
    state={'preparing':False,'current':None,'active':None,'encoded':{}}
    prepare_base=trainer._prepare_batch_inputs
    split_base=trainer.split_by_mini_batches
    forward_base=trainer._get_per_token_logps_and_entropies

    def supported():
        return (controller.enabled and getattr(trainer,'dynamic_num_samples',True) is False
                and getattr(trainer.args,'per_device_train_batch_size',None)==1
                and getattr(trainer.template,'sequence_parallel_size',None)==1)

    def activate(binding,source):
        binding=binding if supported() else None
        # 不同窗口即使路径或像素恰好相同，也不能沿用上一窗口的缓存。
        if binding is None or binding!=state['active']:
            controller.set_key(None)
        controller.set_key(binding[2] if binding is not None else None)
        state['active']=binding
        if record is not None:
            record({'event':'microbatch_binding','source':source,'eligible':binding is not None,
                    'image_count':len(binding[2]) if binding is not None else 0})

    @wraps(split_base)
    def split(inputs,*args,**kwargs):
        chunks=split_base(inputs,*args,**kwargs)
        if not state['preparing']:
            return chunks
        def bound_chunks():
            for batch in chunks:
                state['current']=_binding(batch) if supported() else None
                yield batch
            state['current']=None
        return bound_chunks()

    @wraps(prepare_base)
    def prepare(inputs,*args,**kwargs):
        state['encoded'].clear()
        state['preparing']=True
        state['current']=None
        try:
            result=prepare_base(inputs,*args,**kwargs)
            # 只保留真实返回的同一编码字典；重建/切片等未知路径不会误用。
            returned={id(x):x for x in result} if isinstance(result,list) else {}
            state['encoded']={k:v for k,v in state['encoded'].items() if returned.get(k) is v[0]}
            return result
        except BaseException:
            state['encoded'].clear()
            activate(None,'prepare_failed')
            raise
        finally:
            state['preparing']=False
            state['current']=None

    @wraps(forward_base)
    def forward(model,inputs,*args,**kwargs):
        if state['preparing']:
            binding=state['current']
            state['encoded'][id(inputs)]=(inputs,binding)
            source='prepare'
        else:
            saved=state['encoded'].get(id(inputs))
            binding=saved[1] if saved is not None and saved[0] is inputs else None
            source='buffered' if saved is not None and saved[0] is inputs else 'unknown'
        activate(binding,source)
        return forward_base(model,inputs,*args,**kwargs)

    trainer.split_by_mini_batches=split
    trainer._prepare_batch_inputs=prepare
    trainer._get_per_token_logps_and_entropies=forward
    trainer._egoqa_microbatch_binding=state
    return state
